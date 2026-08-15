// Stages 5-9: Pippenger multi-scalar multiplication over BN254 G1 and G2, in CUDA.
//
// Port of crates/g16-metal/src/shaders/msm.metal. Same algorithm, same kernel names, same
// semantics, so a Metal-versus-CUDA number measures the hardware and the compiler and not
// two different MSMs.
//
// This file is compiled at run time by NVRTC, concatenated AFTER `bn254_fr.cuh`, which
// supplies `struct Fr` and the `fr_*` routines. NVRTC has no filesystem, so there is no
// `#include` anywhere here; kernels.rs pastes the two sources together and the `#ifndef`
// guard in the prelude makes that safe.
//
// ============================================================================
// THE CONSTANTS BELOW MIRROR g16_gpu_layout's FQ_MODULUS and FQ_N0. A Rust test greps
// this source for those exact lines, the same way the prelude is guarded, so the two
// cannot drift silently. Change them together or not at all.
// ============================================================================
//
// WHAT THIS FILE DECIDES, AND WHY
//
// 1. BUCKET WRITE CONFLICTS ARE REMOVED BY CONSTRUCTION, NOT BY ATOMICS.
//    There is no 32-byte atomic on any GPU, so the usual `buckets[d] += P` cannot be
//    written directly. Three approaches exist in the literature: sort the (digit, point)
//    pairs, build a sparse-matrix transpose in the cuZK style, or give every thread a
//    private bucket array and merge afterwards. Private buckets are out on arithmetic
//    grounds alone: `threads * 2^(c-1)` accumulators at 128 bytes each is hundreds of
//    megabytes and the merge costs more point additions than the accumulation it
//    parallelises. Sorting the pairs is what zkonduit's Metal MSM does, on the CPU.
//
//    What is implemented here is the third shape, the cuZK one, and it is cheaper than
//    a sort because the keys are already dense small integers: count how many points
//    land in every bucket (32-bit `atomicAdd`, on plain counters, never on a field
//    element), prefix-sum the counts into row offsets, scatter each point index into its
//    bucket's run (again a 32-bit `atomicAdd`, this time on a cursor), and then give one
//    thread exclusive ownership of one bucket. That is a counting sort by bucket index,
//    O(n) rather than O(n log n), and after the scatter every bucket is written by
//    exactly one thread, so the accumulation needs no synchronisation of any kind. Order
//    within a bucket is not preserved, which does not matter because bucket accumulation
//    is commutative.
//
//    If you ever find yourself wanting an atomic on a point here, something upstream has
//    been mis-ported. Every atomic in this file is `atomicAdd` on a `u32`.
//
// 2. THE SCALARS 0 AND 1 NEVER REACH A BUCKET.
//    Four of the five MSMs take the witness as scalars, and in a bit-heavy circuit over
//    99% of those are 0 or 1. `msm_count` and `msm_scatter` test for both before doing
//    anything else: a zero scalar reads its 8 limbs, fails the test and exits, costing
//    the digit scan and nothing more; a one scalar is routed to `msm_ones_*`, which
//    performs exactly one mixed addition for it.
//
//    Leaving the ones in Pippenger would have been correct but pathological, and this is
//    the specific trap worth naming: the signed recoding sends every scalar equal to 1 to
//    digit +1 of window 0, so bucket (0, 0) would collect *every* one-scalar, and since
//    one thread owns one bucket that single thread would serially accumulate 100k points
//    while the other quarter-million threads sat idle. Special-casing 1 is not a
//    micro-optimisation here, it is what stops the dispatch degenerating to one thread.
//
// 3. POINTS ACCUMULATE IN XYZZ, BASES STAY AFFINE.
//    Extended Jacobian (X, Y, ZZ, ZZZ) with x = X/ZZ, y = Y/ZZZ and ZZ^3 = ZZZ^2. Mixed
//    addition (madd-2008-s) is 7M + 2S against 7M + 4S for Jacobian madd-2007-bl, and
//    the general addition (add-2008-s) is 12M + 2S against 11M + 5S, with roughly a
//    third of the field additions. Bucket accumulation is essentially all mixed
//    additions, so this is the operation that decides the kernel.
//
//    The alternative is affine batch addition, which is what rapidsnark and sppark use
//    on CPU and which is why our own CPU MSM leaves 11-19% on the table. It is not taken
//    here, for the reason given in the MSL twin: a batch inversion is three more
//    dispatches and two more passes over the bucket array per accumulation round. The
//    honest statement is that it is untested here, not that it loses.
//
//    Identity is ZZ == 0. On Metal that comes free, because a freshly allocated MTLBuffer
//    is zero-filled. ON CUDA IT DOES NOT: cudaMalloc hands back whatever was there. Every
//    bucket array must be run through `msm_clear_g1` / `msm_clear_g2` and every count
//    array through `zero_u32` before first use. Getting this wrong yields a wrong proof,
//    not a crash, and it will look like a flaky failure. This is the single most likely
//    way to break the port, which is why it is said twice.
//
// 4. THE WINDOW REDUCTION ENDS ON THE HOST.
//    `msm_reduce_*` collapses each window's 2^(c-1) buckets to a single point through a
//    per-thread segment plus a block-level tree, so the host reads back `n_windows`
//    points per MSM and does only the Horner combination. The serial tail stays where
//    serial tails belong.
//
// NVRTC HOUSE RULES OBSERVED HERE
//   * No `#include`, so no `uint2` and no `<cmath>`. The entry pair is a POD struct
//     declared below, and the two helpers that would otherwise come from a header (`min`,
//     `clz`) are written out, so nothing depends on which declarations NVRTC happens to
//     pre-inject. nvcc supplies both silently, which is exactly why the compile check
//     cannot be trusted to catch their absence.
//   * Every entry point is `extern "C"`. Without it NVRTC mangles the name and
//     `load_function` fails at run time with a name lookup error, not at compile time.
//   * Every conditional reduction stays a ternary. See the prelude for the measurement.
//
// COMPILE TIME IS A REAL COST HERE, MEASURED, AND IT LANDS ON THE FIRST PROOF.
// NVRTC only emits PTX; the driver then runs the same ptxas at cuModuleLoad, so whatever
// ptxas costs is paid inside `prepare`. On the T4 box, `nvcc -arch=sm_75 -cubin -Xptxas -v`
// over this unit takes about 4m50s wall, and it is all in the Fq2 instantiations:
// msm_reduce_g2 85.6 s, msm_merge_g2 24.8 s, msm_ones_g2 16.6 s, against 4.7 ms for
// msm_scan. `-Xptxas -O1` only brings the total to 3m48s, so the optimisation level is
// not the lever. The cause is that every `f_*` and `pt_*` here is `__forceinline__`, so
// `pt_mul_small<Fq2>` inlines pt_dbl and pt_add, which inline six fq_mul each, and
// msm_reduce_g2 becomes one enormous function; ptxas is superlinear in that.
//
// It is left as it is, because the mandate for this file is a faithful port and because
// the two candidate fixes are both benchmarks rather than obvious wins: mark the Fq2
// curve routines `__noinline__` (which would also cut the 255-register, stack-spilling
// G2 kernels, but pays an ABI call with a 256-byte point by value), or cache the compiled
// module on disk so only the first run ever pays. Whoever measures the prepare stage will
// find this at the top of the profile; that is the point of writing the numbers down.
//
// Register footprint at -O3, same run, for whoever sizes the launches: zero_u32 and
// msm_clear_* 6, msm_scan 17, msm_count 28, msm_scatter 30, then the point kernels at
// 122-217 for G1 and pinned at the 255 ceiling with a small stack spill for every G2 one.
// 255 registers means at most 8 resident warps per SM out of 32, so the G2 kernels are
// occupancy-bound by construction and a bigger block size will not help them.

#ifndef G16_MSM_CU
#define G16_MSM_CU

// ---------------------------------------------------------------------------
// Fq: the BN254 base field. Same 8 x u32 CIOS Montgomery layout as `Fr` in the prelude,
// and deliberately a separate copy rather than a template. CUDA could carry the modulus
// as a template parameter where MSL could not, but keeping the two files structurally
// identical is worth more than the deduplication: a fix to one transfers to the other by
// inspection, and the drift test greps for these literal lines.
// ---------------------------------------------------------------------------

#define FQ_LIMBS 8

struct Fq {
    u32 v[8];
};

static_assert(sizeof(Fq) == 32, "Fq must be 32 bytes to match layout::PackedFq");

// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
__constant__ u32 FQ_N[8] = { 0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u };

// -q^{-1} mod 2^32. Cross-checked against ark-ff in the layout crate, not pasted from a blog.
__constant__ u32 FQ_N0 = 0xe4866389u;

// R mod q, the Montgomery representative of 1.
__constant__ u32 FQ_R[8] = { 0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u, 0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u };

__device__ __forceinline__ bool fq_is_zero(Fq a) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        acc |= a.v[i];
    }
    return acc == 0u;
}

__device__ __forceinline__ bool fq_eq(Fq a, Fq b) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        acc |= (a.v[i] ^ b.v[i]);
    }
    return acc == 0u;
}

// See fr_cond_sub_n in the prelude for why the borrow test is bit 63 of a wrapped u64 and
// why this is a ternary rather than an `if`.
__device__ __forceinline__ Fq fq_cond_sub_n(Fq a, u32 hi) {
    u32 red[8];
    u64 borrow = 0;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        u64 d = (u64)a.v[i] - (u64)FQ_N[i] - borrow;
        red[i] = (u32)d;
        borrow = (d >> 63) & (u64)1;
    }
    bool take = (hi != 0u) || (borrow == 0);
    Fq out;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = take ? red[i] : a.v[i];
    }
    return out;
}

__device__ __forceinline__ Fq fq_add(Fq a, Fq b) {
    Fq s;
    u64 c = 0;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        u64 t = (u64)a.v[i] + (u64)b.v[i] + c;
        s.v[i] = (u32)t;
        c = t >> 32;
    }
    return fq_cond_sub_n(s, (u32)c);
}

__device__ __forceinline__ Fq fq_sub(Fq a, Fq b) {
    u32 d[8];
    u64 borrow = 0;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        u64 t = (u64)a.v[i] - (u64)b.v[i] - borrow;
        d[i] = (u32)t;
        borrow = (t >> 63) & (u64)1;
    }
    u32 mask = (u32)(0u - (u32)borrow);
    Fq out;
    u64 c = 0;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        u64 t = (u64)d[i] + (u64)(FQ_N[i] & mask) + c;
        out.v[i] = (u32)t;
        c = t >> 32;
    }
    return out;
}

__device__ __forceinline__ Fq fq_zero() {
    Fq z;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        z.v[i] = 0u;
    }
    return z;
}

__device__ __forceinline__ Fq fq_one() {
    Fq o;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        o.v[i] = FQ_R[i];
    }
    return o;
}

__device__ __forceinline__ Fq fq_neg(Fq a) {
    return fq_sub(fq_zero(), a);
}

// CIOS Montgomery product, identical in shape to fr_mul. See the prelude for the
// justification of the 32-bit limb with a 64-bit accumulator, and for the status of the
// PTX carry-chain specialisation this shares with fr_mul behind `G16_FF_PTX`.
#ifdef G16_FF_PTX
__device__ __forceinline__ Fq fq_mul(Fq a, Fq b) {
    Fq out;
    u32 hi = g16_mont_mul_ptx(a.v, b.v, FQ_N, FQ_N0, out.v);
    return fq_cond_sub_n(out, hi);
}
__device__ __forceinline__ Fq fq_mul_portable(Fq a, Fq b) {
#else
__device__ __forceinline__ Fq fq_mul(Fq a, Fq b) {
#endif
    u32 t[FQ_LIMBS + 2];
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS + 2; i++) {
        t[i] = 0u;
    }

#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        u32 bi = b.v[i];

        u64 c = 0;
#pragma unroll
        for (u32 j = 0; j < FQ_LIMBS; j++) {
            u64 r = (u64)t[j] + (u64)a.v[j] * (u64)bi + c;
            t[j] = (u32)r;
            c = r >> 32;
        }
        u64 r = (u64)t[FQ_LIMBS] + c;
        t[FQ_LIMBS] = (u32)r;
        t[FQ_LIMBS + 1] = (u32)(r >> 32);

        u32 m = t[0] * FQ_N0;
        u64 d = (u64)t[0] + (u64)m * (u64)FQ_N[0];
        c = d >> 32;
#pragma unroll
        for (u32 j = 1; j < FQ_LIMBS; j++) {
            u64 r2 = (u64)t[j] + (u64)m * (u64)FQ_N[j] + c;
            t[j - 1] = (u32)r2;
            c = r2 >> 32;
        }
        u64 r3 = (u64)t[FQ_LIMBS] + c;
        t[FQ_LIMBS - 1] = (u32)r3;
        t[FQ_LIMBS] = t[FQ_LIMBS + 1] + (u32)(r3 >> 32);
    }

    Fq out;
#pragma unroll
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = t[i];
    }
    return fq_cond_sub_n(out, t[FQ_LIMBS]);
}

__device__ __forceinline__ Fq fq_sqr(Fq a) {
    return fq_mul(a, a);
}

// ---------------------------------------------------------------------------
// Fq2 = Fq[u] / (u^2 + 1). BN254's G2 lives over this, so stage 6 needs it and the field
// prelude does not have it.
//
// The reduction is `u^2 = -1`, so a multiplication is three Fq multiplications through
// Karatsuba rather than four, and a squaring is two.
// ---------------------------------------------------------------------------

struct Fq2 {
    Fq c0;
    Fq c1;
};

static_assert(sizeof(Fq2) == 64, "Fq2 must be 64 bytes to match layout::PackedFq2");

__device__ __forceinline__ Fq2 fq2_zero() {
    Fq2 r;
    r.c0 = fq_zero();
    r.c1 = fq_zero();
    return r;
}

__device__ __forceinline__ Fq2 fq2_one() {
    Fq2 r;
    r.c0 = fq_one();
    r.c1 = fq_zero();
    return r;
}

__device__ __forceinline__ bool fq2_is_zero(Fq2 a) {
    return fq_is_zero(a.c0) && fq_is_zero(a.c1);
}

__device__ __forceinline__ bool fq2_eq(Fq2 a, Fq2 b) {
    return fq_eq(a.c0, b.c0) && fq_eq(a.c1, b.c1);
}

__device__ __forceinline__ Fq2 fq2_add(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_add(a.c0, b.c0);
    r.c1 = fq_add(a.c1, b.c1);
    return r;
}

__device__ __forceinline__ Fq2 fq2_sub(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_sub(a.c0, b.c0);
    r.c1 = fq_sub(a.c1, b.c1);
    return r;
}

__device__ __forceinline__ Fq2 fq2_neg(Fq2 a) {
    Fq2 r;
    r.c0 = fq_neg(a.c0);
    r.c1 = fq_neg(a.c1);
    return r;
}

// (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u.
// Karatsuba: the cross term is (a0 + a1)(b0 + b1) - a0 b0 - a1 b1.
__device__ __forceinline__ Fq2 fq2_mul(Fq2 a, Fq2 b) {
    Fq v0 = fq_mul(a.c0, b.c0);
    Fq v1 = fq_mul(a.c1, b.c1);
    Fq2 r;
    r.c0 = fq_sub(v0, v1);
    r.c1 = fq_sub(fq_sub(fq_mul(fq_add(a.c0, a.c1), fq_add(b.c0, b.c1)), v0), v1);
    return r;
}

// (a0 + a1 u)^2 = (a0 + a1)(a0 - a1) + 2 a0 a1 u.
__device__ __forceinline__ Fq2 fq2_sqr(Fq2 a) {
    Fq t0 = fq_add(a.c0, a.c1);
    Fq t1 = fq_sub(a.c0, a.c1);
    Fq t2 = fq_mul(a.c0, a.c1);
    Fq2 r;
    r.c0 = fq_mul(t0, t1);
    r.c1 = fq_add(t2, t2);
    return r;
}

// ---------------------------------------------------------------------------
// One name per operation, overloaded on the field, so the curve arithmetic below can be
// written once as a template and instantiated for G1 (Fq) and G2 (Fq2). Overloads rather
// than a traits class because these are all trivially inlined and the overload set is the
// smallest thing that compiles identically under both toolchains.
// ---------------------------------------------------------------------------

__device__ __forceinline__ Fq  f_add(Fq a, Fq b)   { return fq_add(a, b); }
__device__ __forceinline__ Fq2 f_add(Fq2 a, Fq2 b) { return fq2_add(a, b); }
__device__ __forceinline__ Fq  f_sub(Fq a, Fq b)   { return fq_sub(a, b); }
__device__ __forceinline__ Fq2 f_sub(Fq2 a, Fq2 b) { return fq2_sub(a, b); }
__device__ __forceinline__ Fq  f_mul(Fq a, Fq b)   { return fq_mul(a, b); }
__device__ __forceinline__ Fq2 f_mul(Fq2 a, Fq2 b) { return fq2_mul(a, b); }
__device__ __forceinline__ Fq  f_sqr(Fq a)         { return fq_sqr(a); }
__device__ __forceinline__ Fq2 f_sqr(Fq2 a)        { return fq2_sqr(a); }
__device__ __forceinline__ Fq  f_neg(Fq a)         { return fq_neg(a); }
__device__ __forceinline__ Fq2 f_neg(Fq2 a)        { return fq2_neg(a); }
__device__ __forceinline__ bool f_is_zero(Fq a)    { return fq_is_zero(a); }
__device__ __forceinline__ bool f_is_zero(Fq2 a)   { return fq2_is_zero(a); }
__device__ __forceinline__ bool f_eq(Fq a, Fq b)   { return fq_eq(a, b); }
__device__ __forceinline__ bool f_eq(Fq2 a, Fq2 b) { return fq2_eq(a, b); }

// The MSL twin needs `thread F&` here because address spaces are part of the type there.
// CUDA infers, so the qualifier is simply dropped and the signature is otherwise the same.
__device__ __forceinline__ void f_set_zero(Fq& a)  { a = fq_zero(); }
__device__ __forceinline__ void f_set_zero(Fq2& a) { a = fq2_zero(); }
__device__ __forceinline__ void f_set_one(Fq& a)   { a = fq_one(); }
__device__ __forceinline__ void f_set_one(Fq2& a)  { a = fq2_one(); }

// ---------------------------------------------------------------------------
// Points.
//
// `Aff<Fq>` is 64 bytes and `Aff<Fq2>` is 128, matching layout::PackedG1Affine and
// layout::PackedG2Affine exactly, and (0, 0) is the point at infinity in both (it is off
// curve for y^2 = x^3 + b with b != 0, so the encoding is unambiguous, not a convention).
// snarkjs zkeys really do contain such points, so `aff_is_inf` is a live path and not a
// defensive one.
//
// `Xyzz<F>` is the accumulator: x = X/ZZ, y = Y/ZZZ, with the invariant ZZ^3 = ZZZ^2.
// ZZ == 0 is the identity.
// ---------------------------------------------------------------------------

template <typename F>
struct Aff {
    F x;
    F y;
};

template <typename F>
struct Xyzz {
    F x;
    F y;
    F zz;
    F zzz;
};

typedef Aff<Fq> AffG1;
typedef Aff<Fq2> AffG2;
typedef Xyzz<Fq> PtG1;
typedef Xyzz<Fq2> PtG2;

static_assert(sizeof(AffG1) == 64, "AffG1 must match layout::PackedG1Affine");
static_assert(sizeof(AffG2) == 128, "AffG2 must match layout::PackedG2Affine");
static_assert(sizeof(PtG1) == 128, "PtG1 must match msm::PackedXyzzG1");
static_assert(sizeof(PtG2) == 256, "PtG2 must match msm::PackedXyzzG2");

template <typename F>
__device__ __forceinline__ Xyzz<F> pt_zero() {
    Xyzz<F> r;
    f_set_zero(r.x);
    f_set_zero(r.y);
    f_set_zero(r.zz);
    f_set_zero(r.zzz);
    return r;
}

template <typename F>
__device__ __forceinline__ bool pt_is_zero(Xyzz<F> p) {
    return f_is_zero(p.zz);
}

template <typename F>
__device__ __forceinline__ bool aff_is_inf(Aff<F> p) {
    return f_is_zero(p.x) && f_is_zero(p.y);
}

template <typename F>
__device__ __forceinline__ Xyzz<F> pt_from_affine(Aff<F> p) {
    Xyzz<F> r;
    r.x = p.x;
    r.y = p.y;
    f_set_one(r.zz);
    f_set_one(r.zzz);
    return r;
}

template <typename F>
__device__ __forceinline__ Xyzz<F> pt_neg(Xyzz<F> p) {
    Xyzz<F> r = p;
    r.y = f_neg(p.y);
    return r;
}

// mdbl-2008-s: doubling an affine point straight into XYZZ. a = 0 for BN254, on both G1
// and G2, so the `a * ZZ^2` term of the general doubling disappears.
template <typename F>
__device__ __forceinline__ Xyzz<F> pt_dbl_affine(Aff<F> p) {
    if (f_is_zero(p.y)) {
        // Only reachable for a 2-torsion point, which BN254's odd-order groups do not
        // contain. Returning the identity is the mathematically correct answer anyway.
        return pt_zero<F>();
    }
    F u = f_add(p.y, p.y);
    F v = f_sqr(u);
    F w = f_mul(u, v);
    F s = f_mul(p.x, v);
    F xx = f_sqr(p.x);
    F m = f_add(f_add(xx, xx), xx);
    Xyzz<F> r;
    r.x = f_sub(f_sqr(m), f_add(s, s));
    r.y = f_sub(f_mul(m, f_sub(s, r.x)), f_mul(w, p.y));
    r.zz = v;
    r.zzz = w;
    return r;
}

// dbl-2008-s-1 with a = 0.
template <typename F>
__device__ __forceinline__ Xyzz<F> pt_dbl(Xyzz<F> p) {
    if (pt_is_zero(p) || f_is_zero(p.y)) {
        return pt_zero<F>();
    }
    F u = f_add(p.y, p.y);
    F v = f_sqr(u);
    F w = f_mul(u, v);
    F s = f_mul(p.x, v);
    F xx = f_sqr(p.x);
    F m = f_add(f_add(xx, xx), xx);
    Xyzz<F> r;
    r.x = f_sub(f_sqr(m), f_add(s, s));
    r.y = f_sub(f_mul(m, f_sub(s, r.x)), f_mul(w, p.y));
    r.zz = f_mul(v, p.zz);
    r.zzz = f_mul(w, p.zzz);
    return r;
}

// madd-2008-s: XYZZ += affine, 7M + 2S. This is the inner loop of bucket accumulation and
// therefore the single hottest routine in the whole backend.
//
// Every guard below is load-bearing and none may be dropped as unreachable. The
// exceptional cases are rare, which is precisely what makes a missing guard dangerous: it
// passes a casual test and then fails on one particular witness in production.
template <typename F>
__device__ __forceinline__ Xyzz<F> pt_madd(Xyzz<F> acc, Aff<F> p) {
    if (aff_is_inf(p)) {
        return acc;
    }
    if (pt_is_zero(acc)) {
        return pt_from_affine(p);
    }
    F u2 = f_mul(p.x, acc.zz);
    F s2 = f_mul(p.y, acc.zzz);
    F pp_ = f_sub(u2, acc.x);
    F rr = f_sub(s2, acc.y);
    if (f_is_zero(pp_)) {
        // Same x. Either the same point (double it) or its negative (they cancel).
        // A bucket can hit both: the signed digit recoding puts a point and its negation
        // in different buckets, but two *different* bases in one bucket can still be
        // equal or opposite, and a zkey is not guaranteed to have distinct bases.
        if (f_is_zero(rr)) {
            return pt_dbl_affine(p);
        }
        return pt_zero<F>();
    }
    F pp = f_sqr(pp_);
    F ppp = f_mul(pp_, pp);
    F q = f_mul(acc.x, pp);
    Xyzz<F> r;
    r.x = f_sub(f_sub(f_sqr(rr), ppp), f_add(q, q));
    r.y = f_sub(f_mul(rr, f_sub(q, r.x)), f_mul(acc.y, ppp));
    r.zz = f_mul(acc.zz, pp);
    r.zzz = f_mul(acc.zzz, ppp);
    return r;
}

// add-2008-s: XYZZ + XYZZ, 12M + 2S. Used only by the bucket reduction, which runs
// 2 * 2^(c-1) times per window against n mixed additions in the accumulation.
template <typename F>
__device__ __forceinline__ Xyzz<F> pt_add(Xyzz<F> a, Xyzz<F> b) {
    if (pt_is_zero(a)) {
        return b;
    }
    if (pt_is_zero(b)) {
        return a;
    }
    F u1 = f_mul(a.x, b.zz);
    F u2 = f_mul(b.x, a.zz);
    F s1 = f_mul(a.y, b.zzz);
    F s2 = f_mul(b.y, a.zzz);
    F pp_ = f_sub(u2, u1);
    F rr = f_sub(s2, s1);
    if (f_is_zero(pp_)) {
        if (f_is_zero(rr)) {
            return pt_dbl(a);
        }
        return pt_zero<F>();
    }
    F pp = f_sqr(pp_);
    F ppp = f_mul(pp_, pp);
    F q = f_mul(u1, pp);
    Xyzz<F> r;
    r.x = f_sub(f_sub(f_sqr(rr), ppp), f_add(q, q));
    r.y = f_sub(f_mul(rr, f_sub(q, r.x)), f_mul(s1, ppp));
    r.zz = f_mul(f_mul(a.zz, b.zz), pp);
    r.zzz = f_mul(f_mul(a.zzz, b.zzz), ppp);
    return r;
}

// Index of the highest set bit of a nonzero k, written out rather than called as `__clz`.
// NVRTC compiles without the CUDA headers and the set of declarations it pre-injects is
// not something this file should depend on; nvcc supplies `__clz` silently, so the
// offline compile check would not have caught the difference. 31 predicated iterations,
// fully unrolled, on a path that runs once per reduce thread.
__device__ __forceinline__ u32 msm_hibit(u32 k) {
    u32 h = 0u;
#pragma unroll
    for (u32 i = 1u; i < 32u; i++) {
        h = ((k >> i) != 0u) ? i : h;
    }
    return h;
}

// Same reasoning as msm_hibit: `min` over two u32 is one line, so write it here.
__device__ __forceinline__ u32 msm_min(u32 a, u32 b) {
    return (a < b) ? a : b;
}

// k * P for a small k (below 2^16 here: it is a bucket index inside one window).
// Plain MSB-first double-and-add. It runs once per reduce thread, against a whole segment
// of bucket additions, so a windowed ladder would not pay for itself.
template <typename F>
__device__ __forceinline__ Xyzz<F> pt_mul_small(Xyzz<F> p, u32 k) {
    Xyzz<F> acc = pt_zero<F>();
    if (k == 0u || pt_is_zero(p)) {
        return acc;
    }
    u32 hi = msm_hibit(k);
    for (int i = (int)hi; i >= 0; i--) {
        acc = pt_dbl(acc);
        if ((k >> (u32)i) & 1u) {
            acc = pt_add(acc, p);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Scalar window decomposition.
//
// The recoding is the carry-free signed one from `g16-msm`, reproduced digit for digit so
// the GPU and CPU MSMs decompose identically and a mismatch can only come from the point
// arithmetic:
//
//     b_i = bits [i*c, i*c + c) of the scalar
//     d_i = b_i - 2^c * [b_i >= 2^(c-1)]  +  bit (i*c - 1) of the scalar
//
// Every digit is a pure function of the scalar and the window index, so the windows stay
// independent and nothing has to be materialised. `d_i` lands in [-2^(c-1), 2^(c-1)], so
// bucket index |d_i| - 1 is in [0, 2^(c-1) - 1] and there are 2^(c-1) buckets, NOT
// 2^(c-1) + 1. That extra bucket is a real bug already found and fixed once in the CPU
// code: never written, but still walked by the running-sum reduction, which shifts every
// coefficient by one and yields a wrong proof.
//
// THE SCALARS READ HERE ARE IN STANDARD FORM, NOT MONTGOMERY. `fr_mont_to_std` exists to
// put them there. Feeding Montgomery limbs to this decomposition produces a proof wrong
// by a factor of R, which fails verification with no other symptom to go on.
// ---------------------------------------------------------------------------

// `width` bits of a 256-bit little-endian value starting at `bit_off`. Reads past the top
// as zero, which is what makes the highest window cheap instead of a special case.
__device__ __forceinline__ u32 sc_read_bits(const u32* v, u32 bit_off, u32 width) {
    u32 idx = bit_off >> 5;
    if (idx >= 8u) {
        return 0u;
    }
    u32 sh = bit_off & 31u;
    u64 buf = (u64)v[idx] >> sh;
    if (sh + width > 32u && idx + 1u < 8u) {
        buf |= (u64)v[idx + 1u] << (32u - sh);
    }
    return (u32)(buf & (((u64)1 << width) - (u64)1));
}

// Signed digit `i`, returned split into magnitude and sign so the caller never handles a
// negative index. `mag == 0` means the digit is zero and no bucket is touched.
__device__ __forceinline__ void sc_signed_digit(const u32* v, u32 i, u32 c, u32& mag, bool& neg) {
    u32 off = i * c;
    u32 b = sc_read_bits(v, off, c);
    u32 carry = (off == 0u) ? 0u : sc_read_bits(v, off - 1u, 1u);
    // The borrow of 2^c when the raw window is in the top half; the neighbour to the
    // right pays it back through the carry bit above.
    bool borrow = ((b >> (c - 1u)) & 1u) != 0u;
    if (borrow) {
        // d = b - 2^c + carry, which is in [-2^(c-1), 0].
        u32 m = (1u << c) - b - carry;
        mag = m;
        neg = true;
        if (m == 0u) {
            neg = false;
        }
    } else {
        mag = b + carry;
        neg = false;
    }
}

__device__ __forceinline__ bool sc_is_zero(const u32* v) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        acc |= v[i];
    }
    return acc == 0u;
}

__device__ __forceinline__ bool sc_is_one(const u32* v) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 1; i < 8u; i++) {
        acc |= v[i];
    }
    return acc == 0u && v[0] == 1u;
}

// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `msm::MsmParams` on the host, which must be `#[repr(C)]`
// with these ten `u32` fields in this order. Passed by value: CUDA kernel arguments live
// in a small parameter space, so there is no constant buffer to bind and no
// `[[buffer(n)]]` index to keep in sync, which removes a whole class of Metal-side
// mistake. The host declares it locally and can therefore implement cudarc's `DeviceRepr`
// for it directly.
// ---------------------------------------------------------------------------

struct MsmParams {
    u32 n;           // scalars in this MSM
    u32 c;           // window width in bits
    u32 n_windows;   // ceil(255 / c)
    u32 n_buckets;   // 2^(c-1)
    u32 cap;         // entries reserved per window, == n
    u32 scalar_off;  // element offset into the scalar buffer
    u32 base_off;    // element offset into the base buffer
    u32 ones_groups; // blocks in msm_ones_*
    u32 slice_len;   // entries per thread in the segmented accumulation
    u32 slices;      // ceil(cap / slice_len), threads per window there
};

static_assert(sizeof(MsmParams) == 40, "MsmParams must be ten u32 to match the host struct");

// The scatter entry, which is MSL's `uint2` in the twin. NVRTC has no `vector_types.h`
// and this file may not include one, so the pair is declared here as a POD struct; the
// host mirrors it as a `#[repr(C)]` pair of `u32`, since only the size and field order
// matter.
//
//   x = row  = w * n_buckets + (|digit| - 1)
//   y = (point index << 1) | (digit is negative)
//
// The row is stored rather than recomputed because the segmented accumulation walks a
// fixed-length slice of the entry array and has to discover where one bucket's run ends,
// which it cannot do from the point index alone.
struct MsmEntry {
    u32 x;
    u32 y;
};

static_assert(sizeof(MsmEntry) == 8, "MsmEntry must be two u32, the twin of MSL uint2");

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

// One thread per word. This is not an optimisation, it is a correctness requirement on
// CUDA: cudaMalloc does not zero, and the count histogram is accumulated with atomicAdd
// on top of whatever the allocation already held. Metal gets that zeroing free from a
// fresh MTLBuffer; this backend must clear every counter array itself, every time, before
// msm_count.
extern "C" __global__ void zero_u32(u32* buf, u32 len) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < len) {
        buf[gid] = 0u;
    }
}

// Montgomery limbs (layout::PackedFr) to standard limbs (layout::PackedScalar).
// The NTT leaves H in Montgomery form on the device; Pippenger needs the integer.
extern "C" __global__ void fr_mont_to_std(const Fr* in, Fr* out, u32 len) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < len) {
        out[gid] = fr_from_mont(in[gid]);
    }
}

// Stage 1 of the counting sort: how many points land in each (window, bucket).
// One thread per scalar. `counts` must have been zeroed first, see zero_u32.
extern "C" __global__ void msm_count(const u32* scalars, u32* counts, MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.n) {
        return;
    }
    u32 s[8];
    u32 base = (p.scalar_off + gid) * 8u;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        s[i] = scalars[base + i];
    }
    // The two cheap classes leave here. A zero has cost one 8-limb read and nothing
    // else; a one is handled by msm_ones_* in a single mixed addition.
    if (sc_is_zero(s) || sc_is_one(s)) {
        return;
    }
    for (u32 w = 0; w < p.n_windows; w++) {
        u32 mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) {
            continue;
        }
        // A plain 32-bit atomic on a plain counter. No atomic anywhere in this file
        // touches a field element, and none ever should.
        atomicAdd(&counts[w * p.n_buckets + (mag - 1u)], 1u);
    }
}

// Stage 2: exclusive prefix sum of the counts inside each window, biased by that window's
// base offset in the entry array. ONE BLOCK PER WINDOW.
//
// SCAN_TG is the shared array size, not the launch size: the host passes up to SCAN_TG
// threads and the kernel reads the real count from blockDim.x, so a smaller block still
// works. Launching more than SCAN_TG threads per block would run off the end of `tmp`.
#define SCAN_TG 256

extern "C" __global__ void msm_scan(const u32* counts, u32* cursor, MsmParams p) {
    __shared__ u32 tmp[SCAN_TG];
    u32 w = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 tcount = blockDim.x;

    u32 running = w * p.cap;
    u32 chunks = (p.n_buckets + tcount - 1u) / tcount;
    for (u32 ch = 0; ch < chunks; ch++) {
        u32 idx = ch * tcount + tid;
        u32 v = (idx < p.n_buckets) ? counts[w * p.n_buckets + idx] : 0u;
        tmp[tid] = v;
        __syncthreads();
        // Hillis-Steele inclusive scan. The read is separated from the write by a barrier
        // on both sides, which is what makes the in-place update safe. Every thread of the
        // block reaches every barrier: the trip count depends only on tcount, which is
        // uniform, so there is no divergent __syncthreads here.
        for (u32 d = 1; d < tcount; d <<= 1) {
            u32 x = (tid >= d) ? tmp[tid - d] : 0u;
            __syncthreads();
            tmp[tid] += x;
            __syncthreads();
        }
        if (idx < p.n_buckets) {
            cursor[w * p.n_buckets + idx] = running + tmp[tid] - v;
        }
        u32 total = tmp[tcount - 1u];
        // Not decoration: this is what stops the next chunk's `tmp[tid] = v` landing
        // before a slower lane has read tmp[tcount - 1].
        __syncthreads();
        running += total;
    }
}

// Stage 3: scatter each point index into its bucket's run. One thread per scalar. The
// cursor is bumped with a relaxed 32-bit atomicAdd, so points land inside their run in an
// arbitrary order, which is fine because bucket accumulation is commutative. After this
// kernel `cursor` holds each run's END, and the run's start is `cursor - counts`.
extern "C" __global__ void msm_scatter(const u32* scalars, u32* cursor, MsmEntry* entries, MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.n) {
        return;
    }
    u32 s[8];
    u32 base = (p.scalar_off + gid) * 8u;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        s[i] = scalars[base + i];
    }
    if (sc_is_zero(s) || sc_is_one(s)) {
        return;
    }
    for (u32 w = 0; w < p.n_windows; w++) {
        u32 mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) {
            continue;
        }
        u32 row = w * p.n_buckets + (mag - 1u);
        u32 slot = atomicAdd(&cursor[row], 1u);
        MsmEntry e;
        e.x = row;
        e.y = (gid << 1) | (neg ? 1u : 0u);
        entries[slot] = e;
    }
}

// Stage 4, the simple form: one thread owns one bucket, exclusively, so there is nothing
// to synchronise. Correct, and a third of the code of the segmented form below, but the
// dispatch finishes when the fattest bucket does.
//
// MEASURED on the Metal side, and the numbers are why the segmented kernel exists. On
// js_2x2_d32 at c=11 the busiest bucket holds 11,758 entries against a mean of 17.02, a
// 691x imbalance; this kernel takes 200.99 ms for that MSM and the segmented one takes
// 8.00 ms on identical inputs. End to end over all five MSMs at the tuned window widths,
// this kernel gives 455.9 ms on js_16x16_d32 against 126.8 ms segmented.
//
// Kept so the same comparison can be re-run here rather than assumed to carry over from
// Apple silicon. The imbalance is a property of the witness and does carry over; what it
// costs is a property of the machine and does not.
template <typename F>
__device__ __forceinline__ void msm_accumulate_impl(const MsmEntry* entries,
                                                    const Aff<F>* bases,
                                                    const u32* counts,
                                                    const u32* cursor,
                                                    Xyzz<F>* buckets,
                                                    MsmParams p,
                                                    u32 gid) {
    u32 total = p.n_windows * p.n_buckets;
    if (gid >= total) {
        return;
    }
    u32 end = cursor[gid];
    u32 cnt = counts[gid];
    u32 start = end - cnt;
    Xyzz<F> acc = pt_zero<F>();
    for (u32 i = start; i < end; i++) {
        u32 e = entries[i].y;
        Aff<F> b = bases[p.base_off + (e >> 1)];
        if ((e & 1u) != 0u) {
            b.y = f_neg(b.y);
        }
        acc = pt_madd(acc, b);
    }
    buckets[gid] = acc;
}

// A templated function cannot itself be `extern "C" __global__`, which is why every entry
// point below is a thin instantiating wrapper. Same pattern as the MSL twin.
extern "C" __global__ void msm_accumulate_g1(const MsmEntry* entries,
                                             const AffG1* bases,
                                             const u32* counts,
                                             const u32* cursor,
                                             PtG1* buckets,
                                             MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_accumulate_impl<Fq>(entries, bases, counts, cursor, buckets, p, gid);
}

extern "C" __global__ void msm_accumulate_g2(const MsmEntry* entries,
                                             const AffG2* bases,
                                             const u32* counts,
                                             const u32* cursor,
                                             PtG2* buckets,
                                             MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_accumulate_impl<Fq2>(entries, bases, counts, cursor, buckets, p, gid);
}

// Stage 4, the load-balanced form. This is the kernel the backend actually uses.
//
// THE PROBLEM IT SOLVES, measured rather than assumed. One thread per bucket makes
// per-thread work proportional to bucket occupancy, and occupancy is not uniform: a
// witness contains repeated values, and every copy of one value lands in the same bucket
// of every window. On js_2x2_d32 the fattest bucket holds 11,758 entries against a mean
// of 17. That single thread runs 11,758 serial mixed additions while the other 24,575
// threads finish in about 17 and then wait, and it costs 201 ms of a 253 ms MSM. On CUDA
// the shape of the problem is if anything worse: a straggler lane holds its whole warp,
// and the warp holds its registers and its slot on the SM until it retires.
//
// THE FIX, which is zkmopro's segmented SMVP adapted to our layout. Slice the entry array
// into fixed-length runs of `slice_len` and give one thread each slice, so per-thread work
// is uniform BY CONSTRUCTION rather than by hoping the digits spread. Within its slice a
// thread finds bucket boundaries by watching `entry.x` change.
//
//   * A run that neither starts at the slice's first entry nor ends at its last is
//     wholly contained here, so no other thread will ever touch that bucket, and it is
//     written straight to `buckets[row]` with no synchronisation.
//   * The first and last runs may continue into the neighbouring slices, so they are
//     written to that slice's two spill slots, tagged with their row. That is at most
//     two spills per thread, and a slice containing a single run spills once.
//
// `msm_merge_*` then adds a bucket's spills to whatever was direct-written. It only has
// to look at the slices its own run overlaps, which it computes from the run's start and
// end, so there is no search. The worst-case merge cost is `count / slice_len` additions
// for the fattest bucket, which turns the 11,758-step serial loop into about 180.
//
// `slice_len` trades the two against each other: accumulation is `slice_len` mixed
// additions per thread and the merge is `max_count / slice_len` full additions, so the
// balance point is near sqrt(max_count). 64 also keeps the thread count high enough to
// fill the machine at our smaller domains, and that half of the argument is stronger here
// than on Metal because there is more machine to fill.

__constant__ u32 MSM_NO_ROW = 0xffffffffu;

// Every bucket has to start at the identity, and on CUDA a fresh cudaMalloc is not zero,
// never mind a pooled buffer still holding the previous proof's points. Only `zz` is
// written: `pt_add`, `pt_madd` and the host conversion all test `zz` alone, and a bucket
// that is direct-written is overwritten in full anyway, so clearing the other three
// coordinates would be pure memory traffic. One thread per bucket row.
//
// THIS KERNEL IS MANDATORY HERE. On Metal it only matters for a reused buffer.
template <typename F>
__device__ __forceinline__ void msm_clear_impl(Xyzz<F>* buckets, MsmParams p, u32 gid) {
    if (gid < p.n_windows * p.n_buckets) {
        F z;
        f_set_zero(z);
        buckets[gid].zz = z;
    }
}

extern "C" __global__ void msm_clear_g1(PtG1* buckets, MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_clear_impl<Fq>(buckets, p, gid);
}

extern "C" __global__ void msm_clear_g2(PtG2* buckets, MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_clear_impl<Fq2>(buckets, p, gid);
}

// One thread per slice, gid = w * slices + k. Both spill slots are written before any
// early return that can reach them, so the spill arrays need no pre-zeroing of their own;
// the bucket array still does, through msm_clear_*.
template <typename F>
__device__ __forceinline__ void msm_segmented_impl(const MsmEntry* entries,
                                                   const Aff<F>* bases,
                                                   const u32* cursor,
                                                   Xyzz<F>* buckets,
                                                   Xyzz<F>* spill_pts,
                                                   u32* spill_rows,
                                                   MsmParams p,
                                                   u32 gid) {
    u32 w = gid / p.slices;
    u32 k = gid - w * p.slices;
    if (w >= p.n_windows) {
        return;
    }
    u32 base = w * p.cap;
    // The scatter left every cursor at its run's end, so the last bucket's cursor is the
    // end of the whole window region. Slices past it are empty.
    u32 used = cursor[w * p.n_buckets + p.n_buckets - 1u] - base;

    u32 head_slot = 2u * gid;
    u32 tail_slot = head_slot + 1u;
    spill_rows[head_slot] = MSM_NO_ROW;
    spill_rows[tail_slot] = MSM_NO_ROW;

    u32 lo = k * p.slice_len;
    if (lo >= used) {
        return;
    }
    u32 hi = msm_min(lo + p.slice_len, used);

    u32 cur_row = entries[base + lo].x;
    Xyzz<F> acc = pt_zero<F>();
    bool is_first_run = true;

    for (u32 i = lo; i < hi; i++) {
        MsmEntry e = entries[base + i];
        if (e.x != cur_row) {
            if (is_first_run) {
                spill_rows[head_slot] = cur_row;
                spill_pts[head_slot] = acc;
                is_first_run = false;
            } else {
                // Strictly interior: this thread is the only one that will ever see this
                // bucket, so the write needs no synchronisation and no spill slot.
                buckets[cur_row] = acc;
            }
            acc = pt_zero<F>();
            cur_row = e.x;
        }
        Aff<F> b = bases[p.base_off + (e.y >> 1)];
        if ((e.y & 1u) != 0u) {
            b.y = f_neg(b.y);
        }
        acc = pt_madd(acc, b);
    }

    // The run that ends at the slice boundary always spills, whether or not it actually
    // continues. Spilling one run that did not need to costs the merge one addition;
    // failing to spill one that did would lose it.
    if (is_first_run) {
        spill_rows[head_slot] = cur_row;
        spill_pts[head_slot] = acc;
    } else {
        spill_rows[tail_slot] = cur_row;
        spill_pts[tail_slot] = acc;
    }
}

extern "C" __global__ void msm_segmented_g1(const MsmEntry* entries,
                                            const AffG1* bases,
                                            const u32* cursor,
                                            PtG1* buckets,
                                            PtG1* spill_pts,
                                            u32* spill_rows,
                                            MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_segmented_impl<Fq>(entries, bases, cursor, buckets, spill_pts, spill_rows, p, gid);
}

extern "C" __global__ void msm_segmented_g2(const MsmEntry* entries,
                                            const AffG2* bases,
                                            const u32* cursor,
                                            PtG2* buckets,
                                            PtG2* spill_pts,
                                            u32* spill_rows,
                                            MsmParams p) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_segmented_impl<Fq2>(entries, bases, cursor, buckets, spill_pts, spill_rows, p, gid);
}

// Fold each bucket's spilled partials into it. One thread per bucket row, and it looks
// only at the slices its own run overlaps, so there is no search and no atomic.
template <typename F>
__device__ __forceinline__ void msm_merge_impl(Xyzz<F>* buckets,
                                               const Xyzz<F>* spill_pts,
                                               const u32* spill_rows,
                                               const u32* counts,
                                               const u32* cursor,
                                               MsmParams p,
                                               u32 row) {
    if (row >= p.n_windows * p.n_buckets) {
        return;
    }
    u32 cnt = counts[row];
    if (cnt == 0u) {
        return;
    }
    u32 w = row / p.n_buckets;
    u32 base = w * p.cap;
    u32 start = cursor[row] - cnt - base;
    u32 end = cursor[row] - base;
    u32 k_lo = start / p.slice_len;
    u32 k_hi = (end - 1u) / p.slice_len;

    Xyzz<F> acc = buckets[row];
    for (u32 k = k_lo; k <= k_hi; k++) {
        u32 slot = 2u * (w * p.slices + k);
        if (spill_rows[slot] == row) {
            acc = pt_add(acc, spill_pts[slot]);
        }
        if (spill_rows[slot + 1u] == row) {
            acc = pt_add(acc, spill_pts[slot + 1u]);
        }
    }
    buckets[row] = acc;
}

extern "C" __global__ void msm_merge_g1(PtG1* buckets,
                                        const PtG1* spill_pts,
                                        const u32* spill_rows,
                                        const u32* counts,
                                        const u32* cursor,
                                        MsmParams p) {
    u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    msm_merge_impl<Fq>(buckets, spill_pts, spill_rows, counts, cursor, p, row);
}

extern "C" __global__ void msm_merge_g2(PtG2* buckets,
                                        const PtG2* spill_pts,
                                        const u32* spill_rows,
                                        const u32* counts,
                                        const u32* cursor,
                                        MsmParams p) {
    u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    msm_merge_impl<Fq2>(buckets, spill_pts, spill_rows, counts, cursor, p, row);
}

// Stage 5: collapse one window's 2^(c-1) buckets to one point. ONE BLOCK PER WINDOW.
//
// The window sum is sum_j (j+1) B_j. Split the buckets into one segment per thread, at
// [lo, hi). Inside a segment the reverse running sum gives
// P = sum_j (j - lo + 1) B_j and Q = sum_j B_j in two additions per bucket, and the
// segment contributes P + lo * Q. The per-thread results are then tree-reduced in shared
// memory, so the host reads back one point per window and does nothing but the Horner
// combination.
//
// REDUCE_TG is the shared array size and therefore the largest block the host may launch
// at these kernels. 64 rather than 128 is an occupancy choice: at 64 the G2 array is
// 64 * 256 = 16 KB. An sm_75 SM has 64 KB of shared memory, so 16 KB per block leaves
// room for four resident blocks; 128 threads would be 32 KB and halve that. The MSL twin
// picks the same 64 off the same reasoning applied to a 32 KB threadgroup budget, which
// is a coincidence of the numbers rather than a shared derivation.
#define REDUCE_TG 64

template <typename F>
__device__ __forceinline__ void msm_reduce_impl(const Xyzz<F>* buckets,
                                                Xyzz<F>* window_sums,
                                                MsmParams p,
                                                Xyzz<F>* shared,
                                                u32 w,
                                                u32 tid,
                                                u32 tcount) {
    u32 seg_len = (p.n_buckets + tcount - 1u) / tcount;
    u32 lo = tid * seg_len;
    u32 hi = msm_min(lo + seg_len, p.n_buckets);

    Xyzz<F> mine = pt_zero<F>();
    if (lo < hi) {
        Xyzz<F> run = pt_zero<F>();
        Xyzz<F> tot = pt_zero<F>();
        for (u32 j = hi; j > lo; j--) {
            run = pt_add(run, buckets[w * p.n_buckets + (j - 1u)]);
            tot = pt_add(tot, run);
        }
        mine = pt_add(tot, pt_mul_small(run, lo));
    }
    shared[tid] = mine;

    // Every thread of the block runs this loop the same number of times, so no
    // __syncthreads() is reached by only part of the block. A thread whose segment was
    // empty still contributes its identity and still has to arrive at the barriers.
    for (u32 s = 1; s < tcount; s <<= 1) {
        __syncthreads();
        if ((tid & ((s << 1) - 1u)) == 0u && tid + s < tcount) {
            shared[tid] = pt_add(shared[tid], shared[tid + s]);
        }
    }
    __syncthreads();
    if (tid == 0u) {
        window_sums[w] = shared[0];
    }
}

extern "C" __global__ void msm_reduce_g1(const PtG1* buckets, PtG1* window_sums, MsmParams p) {
    __shared__ PtG1 shared[REDUCE_TG];
    msm_reduce_impl<Fq>(buckets, window_sums, p, shared, blockIdx.x, threadIdx.x, blockDim.x);
}

extern "C" __global__ void msm_reduce_g2(const PtG2* buckets, PtG2* window_sums, MsmParams p) {
    __shared__ PtG2 shared[REDUCE_TG];
    msm_reduce_impl<Fq2>(buckets, window_sums, p, shared, blockIdx.x, threadIdx.x, blockDim.x);
}

// The scalar-of-1 path: sum the bases whose scalar is exactly 1, one mixed addition each.
// Strided so consecutive lanes read consecutive scalars, which on CUDA also means a warp's
// 8-limb reads coalesce into contiguous 1 KB transactions rather than 32 scattered ones.
// Then the same block-level tree as the reduction, so the host adds only `ones_groups`
// points.
//
// This kernel is not optional: msm_count and msm_scatter deliberately drop every scalar
// equal to 1, so without it those terms are simply missing from the proof.
template <typename F>
__device__ __forceinline__ void msm_ones_impl(const u32* scalars,
                                              const Aff<F>* bases,
                                              Xyzz<F>* out,
                                              MsmParams p,
                                              Xyzz<F>* shared,
                                              u32 g,
                                              u32 tid,
                                              u32 tcount) {
    u32 stride = p.ones_groups * tcount;
    Xyzz<F> acc = pt_zero<F>();
    for (u32 i = g * tcount + tid; i < p.n; i += stride) {
        u32 s[8];
        u32 base = (p.scalar_off + i) * 8u;
#pragma unroll
        for (u32 k = 0; k < 8u; k++) {
            s[k] = scalars[base + k];
        }
        if (!sc_is_one(s)) {
            continue;
        }
        acc = pt_madd(acc, bases[p.base_off + i]);
    }
    shared[tid] = acc;
    for (u32 s = 1; s < tcount; s <<= 1) {
        __syncthreads();
        if ((tid & ((s << 1) - 1u)) == 0u && tid + s < tcount) {
            shared[tid] = pt_add(shared[tid], shared[tid + s]);
        }
    }
    __syncthreads();
    if (tid == 0u) {
        out[g] = shared[0];
    }
}

extern "C" __global__ void msm_ones_g1(const u32* scalars, const AffG1* bases, PtG1* out, MsmParams p) {
    __shared__ PtG1 shared[REDUCE_TG];
    msm_ones_impl<Fq>(scalars, bases, out, p, shared, blockIdx.x, threadIdx.x, blockDim.x);
}

extern "C" __global__ void msm_ones_g2(const u32* scalars, const AffG2* bases, PtG2* out, MsmParams p) {
    __shared__ PtG2 shared[REDUCE_TG];
    msm_ones_impl<Fq2>(scalars, bases, out, p, shared, blockIdx.x, threadIdx.x, blockDim.x);
}

#endif // G16_MSM_CU
