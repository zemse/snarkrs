// Stages 5-9: Pippenger multi-scalar multiplication over BN254 G1 and G2, in CUDA.
//
// Port of crates/metal/src/shaders/msm.metal. Same algorithm, same kernel names, same
// semantics, so a Metal-versus-CUDA number measures the hardware and the compiler and not
// two different MSMs.
//
// This file is compiled at run time by NVRTC, concatenated AFTER `bn254_fr.cuh` and
// `bn254_curve.cuh`, which supply `Fr`, the two curve fields and the point arithmetic.
// NVRTC has no filesystem, so there is no `#include` anywhere here;
// snarkrs_gpu_kernels::unit_msm pastes the three sources together and the `#ifndef` guards
// make that safe.
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
// 2. IN VARIABLE WORK, SCALARS 0 AND 1 NEVER REACH A BUCKET.
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
// CONSTANT WORK (opt in): every scalar emits in every window. Zero digits accumulate
// actual bases in dummy rows, discarded only by the mathematical reduction. Ones are
// ordinary digits. Fixed slice and fold geometry depends on n, not classification.
// Fold and reduce use Metal's complete homogeneous formulas. This is NOT constant time:
// atomics, addresses, run boundaries, mixed-add exceptions and host math depend on data.
// NVRTC/PTX and arithmetic validation on NVIDIA are required for this source port.
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

// Renes, Costello and Batina (2016), algorithms 7 and 9, ported from
// metal/src/shaders/msm.metal without changing the complete formulas.
// Local to the MSM unit; shared curve/FFT arithmetic is unchanged.
template <typename F>
struct Proj {
    F x;
    F y;
    F z;
};

typedef Proj<Fq> ProjG1;
typedef Proj<Fq2> ProjG2;

// 3b for each group: 9 on G1 (y^2 = x^3 + 3), and 3 * 3 / (9 + u) on G2, the twist's
// b, in Montgomery form. Cross-checked against arkworks by the CUDA CPU field-oracle tests.
__constant__ u32 FQ2_G2_B3_C0[8] = { 0xb62e0d6au, 0x3baa927cu, 0xd1b664fdu, 0xd71e7c52u, 0xd95d4664u, 0x03873e63u, 0x082ab8f4u, 0x0e75b5b1u };
__constant__ u32 FQ2_G2_B3_C1[8] = { 0x7596fe35u, 0xaab7c666u, 0xbb6a27bau, 0x31d21a78u, 0x680401ffu, 0x85dd7297u, 0xdf39a7e9u, 0x03c52d6au };

// 9a as four additions.
__device__ __forceinline__ Fq f_mul_b3(Fq a) {
    Fq a2 = fq_add(a, a);
    Fq a4 = fq_add(a2, a2);
    Fq a8 = fq_add(a4, a4);
    return fq_add(a8, a);
}

__device__ __forceinline__ Fq2 f_mul_b3(Fq2 a) {
    Fq2 b3;
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        b3.c0.v[i] = FQ2_G2_B3_C0[i];
        b3.c1.v[i] = FQ2_G2_B3_C1[i];
    }
    return fq2_mul(a, b3);
}

__device__ __forceinline__ Fq fq_select(bool take_a, Fq a, Fq b) {
    Fq out;
    for (u32 i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = (take_a ? a.v[i] : b.v[i]);
    }
    return out;
}

__device__ __forceinline__ Fq2 fq2_select(bool take_a, Fq2 a, Fq2 b) {
    Fq2 out;
    out.c0 = fq_select(take_a, a.c0, b.c0);
    out.c1 = fq_select(take_a, a.c1, b.c1);
    return out;
}

__device__ __forceinline__ Fq  f_select(bool take_a, Fq a, Fq b)   { return fq_select(take_a, a, b); }
__device__ __forceinline__ Fq2 f_select(bool take_a, Fq2 a, Fq2 b) { return fq2_select(take_a, a, b); }

template <typename F>
__device__ __forceinline__ Proj<F> proj_zero() {
    Proj<F> r;
    f_set_zero(r.x);
    f_set_one(r.y);
    f_set_zero(r.z);
    return r;
}

// (X/ZZ, Y/ZZZ) = (X ZZZ : Y ZZ : ZZ ZZZ). The XYZZ identity is all zero, which maps
// to (0 : 0 : 0), not a point, so Y becomes 1 for it; the select reads what the
// products just consumed, so nothing stays live.
template <typename F>
__device__ __forceinline__ Proj<F> proj_from_xyzz(Xyzz<F> p) {
    Proj<F> r;
    r.x = f_mul(p.x, p.zzz);
    r.y = f_mul(p.y, p.zz);
    r.z = f_mul(p.zz, p.zzz);
    F one;
    f_set_one(one);
    r.y = f_select(f_is_zero(p.zz), one, r.y);
    return r;
}

// (X/Z, Y/Z) = (X Z, Y Z^2, Z^2, Z^3) in XYZZ. Z = 0 gives all zero, the XYZZ identity.
template <typename F>
__device__ __forceinline__ Xyzz<F> xyzz_from_proj(Proj<F> p) {
    Xyzz<F> r;
    r.zz = f_sqr(p.z);
    r.zzz = f_mul(r.zz, p.z);
    r.x = f_mul(p.x, p.z);
    r.y = f_mul(p.y, r.zz);
    return r;
}

// Algorithm 7 of the paper, a = 0: 12M + 2 products by 3b + 19 additions.
//   X3 = (X1 Y2 + X2 Y1)(Y1 Y2 - 3b Z1 Z2) - 3b (Y1 Z2 + Y2 Z1)(X1 Z2 + X2 Z1)
//   Y3 = (Y1 Y2 + 3b Z1 Z2)(Y1 Y2 - 3b Z1 Z2) + 9b X1 X2 (X1 Z2 + X2 Z1)
//   Z3 = (Y1 Z2 + Y2 Z1)(Y1 Y2 + 3b Z1 Z2) + 3 X1 X2 (X1 Y2 + X2 Y1)
template <typename F>
__device__ __forceinline__ Proj<F> pt_add(Proj<F> a, Proj<F> b) {
    F t0 = f_mul(a.x, b.x);
    F t1 = f_mul(a.y, b.y);
    F t2 = f_mul(a.z, b.z);
    F t3 = f_mul(f_add(a.x, a.y), f_add(b.x, b.y));
    t3 = f_sub(t3, f_add(t0, t1));                  // X1 Y2 + X2 Y1
    F t4 = f_mul(f_add(a.y, a.z), f_add(b.y, b.z));
    t4 = f_sub(t4, f_add(t1, t2));                  // Y1 Z2 + Y2 Z1
    F t5 = f_mul(f_add(a.x, a.z), f_add(b.x, b.z));
    t5 = f_sub(t5, f_add(t0, t2));                  // X1 Z2 + X2 Z1
    t0 = f_add(f_add(t0, t0), t0);                  // 3 X1 X2
    t2 = f_mul_b3(t2);                              // 3b Z1 Z2
    F z3 = f_add(t1, t2);                           // Y1 Y2 + 3b Z1 Z2
    t1 = f_sub(t1, t2);                             // Y1 Y2 - 3b Z1 Z2
    t5 = f_mul_b3(t5);                              // 3b (X1 Z2 + X2 Z1)
    Proj<F> r;
    r.x = f_sub(f_mul(t3, t1), f_mul(t4, t5));
    r.y = f_add(f_mul(z3, t1), f_mul(t5, t0));
    r.z = f_add(f_mul(t4, z3), f_mul(t0, t3));
    return r;
}

// Algorithm 9, a = 0: 6M + 2S + 1 product by 3b, and the identity maps to itself.
//   X3 = 2 X Y (Y^2 - 9b Z^2)
//   Y3 = (Y^2 - 9b Z^2)(Y^2 + 3b Z^2) + 24b Y^2 Z^2
//   Z3 = 8 Y^3 Z
template <typename F>
__device__ __forceinline__ Proj<F> pt_dbl(Proj<F> p) {
    F t0 = f_sqr(p.y);                              // Y^2
    F z3 = f_add(t0, t0);
    z3 = f_add(z3, z3);
    z3 = f_add(z3, z3);                             // 8 Y^2
    F t1 = f_mul(p.y, p.z);                         // Y Z
    F t2 = f_mul_b3(f_sqr(p.z));                    // 3b Z^2
    F x3 = f_mul(t2, z3);                           // 24b Y^2 Z^2
    F y3 = f_add(t0, t2);                           // Y^2 + 3b Z^2
    z3 = f_mul(t1, z3);                             // 8 Y^3 Z
    t1 = f_add(t2, t2);
    t2 = f_add(t1, t2);                             // 9b Z^2
    t0 = f_sub(t0, t2);                             // Y^2 - 9b Z^2
    y3 = f_add(f_mul(t0, y3), x3);
    t1 = f_mul(p.x, p.y);
    x3 = f_mul(t0, t1);
    Proj<F> r;
    r.x = f_add(x3, x3);
    r.y = y3;
    r.z = z3;
    return r;
}

// `pt_mul_small` on the complete formulas: only `k`, a bucket offset the key fixes,
// shapes the ladder, and an identity `p` rides it to the identity.
template <typename F>
__device__ __forceinline__ Proj<F> pt_mul_small(Proj<F> p, u32 k) {
    Proj<F> acc = proj_zero<F>();
    if (k == 0u) {
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

// The reduce is written once over its point type `P`: `Xyzz<F>` with the shortcuts,
// or `Proj<F>` under constant work. Loads and stores are XYZZ either way.
template <typename P>
struct PtOps;

template <typename F>
struct PtOps<Xyzz<F>> {
    static __device__ __forceinline__ Xyzz<F> zero() { return pt_zero<F>(); }
    static __device__ __forceinline__ Xyzz<F> load(Xyzz<F> p) { return p; }
    static __device__ __forceinline__ Xyzz<F> store(Xyzz<F> p) { return p; }
};

template <typename F>
struct PtOps<Proj<F>> {
    static __device__ __forceinline__ Proj<F> zero() { return proj_zero<F>(); }
    static __device__ __forceinline__ Proj<F> load(Xyzz<F> p) { return proj_from_xyzz(p); }
    static __device__ __forceinline__ Xyzz<F> store(Proj<F> p) { return xyzz_from_proj(p); }
};


// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `msm::MsmParams` on the host, which must be `#[repr(C)]`
// with these eleven `u32` fields in this order. Passed by value: CUDA kernel arguments live
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
    u32 cap;         // entries per window, general count or n under constant work
    u32 scalar_off;  // element offset into the scalar buffer
    u32 base_off;    // element offset into the base buffer
    u32 ones_groups; // blocks in msm_ones_*
    u32 slice_len;   // entries per thread in the segmented accumulation
    u32 slices;      // ceil(cap / slice_len), threads per window there
    u32 dummy_rows;  // constant work: zero-digit rows per window
};

static_assert(sizeof(MsmParams) == 44, "MsmParams must be eleven u32 to match the host struct");

struct FoldParams {
    u32 slots;
    u32 groups;
    u32 len;
    u32 last;
};
static_assert(sizeof(FoldParams) == 16, "FoldParams must be four u32");

__device__ __forceinline__ u32 msm_dummy_row(MsmParams p, u32 w, u32 gid) {
    return p.n_windows * p.n_buckets + w * p.dummy_rows + (gid & (p.dummy_rows - 1u));
}

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
// Byte-identical twin of g16_mont_to_std in pointwise.cu, and the duplication is forced by
// the unit split: unit_msm does not carry pointwise.cu. A change to fr_from_mont's callers
// is made in both.
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
    // Variable work skips zeros and routes ones to msm_ones_*. Constant work
    // emits every scalar in every window, including zero digits.
    if (p.dummy_rows == 0u && (sc_is_zero(s) || sc_is_one(s))) {
        return;
    }
    for (u32 w = 0; w < p.n_windows; w++) {
        u32 mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        u32 row;
        if (mag != 0u) {
            row = w * p.n_buckets + (mag - 1u);
        } else if (p.dummy_rows != 0u) {
            row = msm_dummy_row(p, w, gid);
        } else {
            continue;
        }
        // A plain 32-bit atomic on a plain counter. No atomic anywhere in this file
        // touches a field element, and none ever should.
        atomicAdd(&counts[row], 1u);
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
    if (tid == 0u) {
        for (u32 j = 0; j < p.dummy_rows; j++) {
            u32 row = p.n_windows * p.n_buckets + w * p.dummy_rows + j;
            cursor[row] = running;
            running += counts[row];
        }
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
    if (p.dummy_rows == 0u && (sc_is_zero(s) || sc_is_one(s))) {
        return;
    }
    for (u32 w = 0; w < p.n_windows; w++) {
        u32 mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        u32 row;
        if (mag != 0u) {
            row = w * p.n_buckets + (mag - 1u);
        } else if (p.dummy_rows != 0u) {
            row = msm_dummy_row(p, w, gid);
        } else {
            continue;
        }
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
// Constant work clears every coordinate: homogeneous conversion consumes them
// even for the identity.
// THIS KERNEL IS MANDATORY HERE. On Metal it only matters for a reused buffer.
template <typename F>
__device__ __forceinline__ void msm_clear_impl(Xyzz<F>* buckets, MsmParams p, u32 gid) {
    if (gid < p.n_windows * (p.n_buckets + p.dummy_rows)) {
        if (p.dummy_rows != 0u) {
            buckets[gid] = pt_zero<F>();
            return;
        }
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
    // end of the whole window region. Constant work includes the dummy rows.
    u32 last = p.dummy_rows == 0u
        ? w * p.n_buckets + p.n_buckets - 1u
        : p.n_windows * p.n_buckets + w * p.dummy_rows + p.dummy_rows - 1u;
    u32 used = cursor[last] - base;

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

// Fixed spill tree, ported from Metal msm_fold_impl. No occupancy-sized merge.
template <typename F>
__device__ __forceinline__ void msm_fold_impl(const Xyzz<F>* in_pts,
                                            const u32* in_rows,
                                            Xyzz<F>* out_pts,
                                            u32* out_rows,
                                            Xyzz<F>* buckets,
                                            MsmParams p,
                                            FoldParams f,
                                            u32 gid) {
    u32 m_in = f.slots;
    u32 groups = f.groups;
    u32 len = f.len;
    bool last = f.last != 0u;
    u32 w = gid / groups;
    u32 g = gid - w * groups;
    if (w >= p.n_windows) {
        return;
    }
    u32 head_slot = 2u * gid;
    u32 tail_slot = head_slot + 1u;
    if (!last) {
        out_rows[head_slot] = MSM_NO_ROW;
        out_rows[tail_slot] = MSM_NO_ROW;
    }

    u32 lo = g * len;
    u32 hi = msm_min(lo + len, m_in);
    u32 cur = MSM_NO_ROW;
    bool is_first_run = true;
    Proj<F> acc = proj_zero<F>();
    // `len` iterations in every group, the window's short last one included, so a
    // thread's additions do not depend on where it sits either.
    for (u32 i = lo; i < lo + len; i++) {
        u32 j = w * m_in + msm_min(i, hi - 1u);
        u32 r = (i < hi) ? in_rows[j] : MSM_NO_ROW;
        Xyzz<F> acc_out = xyzz_from_proj(acc);
        bool empty = r == MSM_NO_ROW;
        if (!empty && r != cur) {
            if (cur != MSM_NO_ROW) {
                if (is_first_run && !last) {
                    out_pts[head_slot] = acc_out;
                    out_rows[head_slot] = cur;
                } else {
                    buckets[cur] = acc_out;
                }
                is_first_run = false;
            }
            acc = proj_zero<F>();
            cur = r;
        }
        Proj<F> q = proj_from_xyzz(in_pts[j]);
        if (empty) {
            q = proj_zero<F>();
        }
        acc = pt_add(acc, q);
    }

    if (cur == MSM_NO_ROW) {
        return;
    }
    Xyzz<F> acc_out = xyzz_from_proj(acc);
    if (last) {
        buckets[cur] = acc_out;
    } else if (is_first_run) {
        out_pts[head_slot] = acc_out;
        out_rows[head_slot] = cur;
    } else {
        out_pts[tail_slot] = acc_out;
        out_rows[tail_slot] = cur;
    }
}

extern "C" __global__ void msm_fold_g1(const PtG1* in_pts,
                                        const u32* in_rows,
                                        PtG1* out_pts,
                                        u32* out_rows,
                                        PtG1* buckets,
                                        MsmParams p,
                                        FoldParams f) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_fold_impl<Fq>(in_pts, in_rows, out_pts, out_rows, buckets, p, f, gid);
}

extern "C" __global__ void msm_fold_g2(const PtG2* in_pts,
                                        const u32* in_rows,
                                        PtG2* out_pts,
                                        u32* out_rows,
                                        PtG2* buckets,
                                        MsmParams p,
                                        FoldParams f) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    msm_fold_impl<Fq2>(in_pts, in_rows, out_pts, out_rows, buckets, p, f, gid);
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

template <typename F, typename P>
__device__ __forceinline__ void msm_reduce_impl(const Xyzz<F>* buckets,
                                                Xyzz<F>* window_sums,
                                                MsmParams p,
                                                P* shared,
                                                u32 w,
                                                u32 tid,
                                                u32 tcount) {
    u32 seg_len = (p.n_buckets + tcount - 1u) / tcount;
    u32 lo = tid * seg_len;
    u32 hi = msm_min(lo + seg_len, p.n_buckets);

    P mine = PtOps<P>::zero();
    if (lo < hi) {
        P run = PtOps<P>::zero();
        P tot = PtOps<P>::zero();
        for (u32 j = hi; j > lo; j--) {
            run = pt_add(run, PtOps<P>::load(buckets[w * p.n_buckets + (j - 1u)]));
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
        window_sums[w] = PtOps<P>::store(shared[0]);
    }
}

extern "C" __global__ void msm_reduce_g1(const PtG1* buckets, PtG1* window_sums, MsmParams p) {
    __shared__ union {
        PtG1 variable[REDUCE_TG];
        ProjG1 constant[REDUCE_TG];
    } shared;
    if (p.dummy_rows != 0u) {
        msm_reduce_impl<Fq, ProjG1>(buckets, window_sums, p, shared.constant, blockIdx.x, threadIdx.x, blockDim.x);
    } else {
        msm_reduce_impl<Fq, PtG1>(buckets, window_sums, p, shared.variable, blockIdx.x, threadIdx.x, blockDim.x);
    }
}

extern "C" __global__ void msm_reduce_g2(const PtG2* buckets, PtG2* window_sums, MsmParams p) {
    __shared__ union {
        PtG2 variable[REDUCE_TG];
        ProjG2 constant[REDUCE_TG];
    } shared;
    if (p.dummy_rows != 0u) {
        msm_reduce_impl<Fq2, ProjG2>(buckets, window_sums, p, shared.constant, blockIdx.x, threadIdx.x, blockDim.x);
    } else {
        msm_reduce_impl<Fq2, PtG2>(buckets, window_sums, p, shared.variable, blockIdx.x, threadIdx.x, blockDim.x);
    }
}

// The scalar-of-1 path: sum the bases whose scalar is exactly 1, one mixed addition each.
// Strided so consecutive lanes read consecutive scalars, which on CUDA also means a warp's
// 8-limb reads coalesce into contiguous 1 KB transactions rather than 32 scattered ones.
// Then the same block-level tree as the reduction, so the host adds only `ones_groups`
// points.
//
// In variable work this kernel is required: count and scatter drop every scalar
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
