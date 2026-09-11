// BN254 Fq, Fq2 and the G1/G2 point arithmetic, shared by the MSM and FFT units.
//
// Split out of msm.cu so that fft.cu can be its own translation unit without carrying
// the seventeen MSM entry points: an entry point in a unit is compiled whether or not
// the host ever loads it, and NVRTC over the MSM unit alone costs 113.6 s on a cold T4
// (context.rs). Everything here is a template or a __forceinline__ helper, so a unit
// pays only for what its own kernels instantiate.
//
// Compiled at run time by NVRTC, concatenated after `bn254_fr.cuh`, which supplies the
// u32/u64 typedefs and, behind `G16_FF_PTX`, the inline-PTX Montgomery round. NVRTC has
// no filesystem, so there is no `#include` anywhere here; g16_gpu_kernels assembles the
// units and the `#ifndef` guards make the concatenation safe.
//
// ============================================================================
// THE CONSTANTS BELOW MIRROR g16_gpu_layout's FQ_MODULUS and FQ_N0. A Rust test greps
// this source for those exact lines, the same way the prelude is guarded, so the two
// cannot drift silently. Change them together or not at all.
// ============================================================================

#ifndef G16_BN254_CURVE_CUH
#define G16_BN254_CURVE_CUH

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
// Jacobian with a = 0, for the GLV ladder in `fft.cu`.
//
// The Pippenger kernels stay in XYZZ, where the mixed addition that decides them
// is cheapest. The ladder is the opposite shape, ~263 doublings against ~58 additions
// per call, and Jacobian doubling with a = 0 (dbl-2009-l) is 2M + 5S against XYZZ's
// 6M + 3S, while the general addition (add-2007-bl, 11M + 5S against 12M + 2S) gives
// only a little back. With `fq_sqr` costing a full multiply that is 7 against 9 per
// doubling and 16 against 14 per addition. A Jacobian point is also 96 bytes to XYZZ's
// 128, a quarter off the ladder's spilled window table.
// ---------------------------------------------------------------------------

template <typename F>
struct Jac {
    F x;
    F y;
    F z;
};

// Identity is Z == 0 with X and Y both zero, so converting it to XYZZ lands exactly on
// `pt_zero`'s all-zero pattern rather than on garbage coordinates with a zero ZZ.
template <typename F>
__device__ __forceinline__ Jac<F> jac_zero() {
    Jac<F> r;
    f_set_zero(r.x);
    f_set_zero(r.y);
    f_set_zero(r.z);
    return r;
}

template <typename F>
__device__ __forceinline__ bool jac_is_zero(Jac<F> p) {
    return f_is_zero(p.z);
}

template <typename F>
__device__ __forceinline__ Jac<F> jac_neg(Jac<F> p) {
    Jac<F> r = p;
    r.y = f_neg(p.y);
    return r;
}

// XYZZ -> Jacobian without the inversion `Z = ZZZ / ZZ` would take: scaling the class
// by ZZ*ZZZ gives Z' = ZZ*ZZZ, X' = X*ZZ*ZZZ^2, Y' = Y*ZZ^3*ZZZ^2, and
// X'/Z'^2 == X/ZZ, Y'/Z'^3 == Y/ZZZ. A zero ZZ stays a zero Z'.
template <typename F>
__device__ __forceinline__ Jac<F> jac_from_xyzz(Xyzz<F> p) {
    F zzz2 = f_sqr(p.zzz);
    F zz2 = f_sqr(p.zz);
    Jac<F> r;
    r.x = f_mul(f_mul(p.x, p.zz), zzz2);
    r.y = f_mul(f_mul(p.y, f_mul(zz2, p.zz)), zzz2);
    r.z = f_mul(p.zz, p.zzz);
    return r;
}

// Jacobian -> XYZZ is the definition of XYZZ: ZZ = Z^2, ZZZ = Z^3.
template <typename F>
__device__ __forceinline__ Xyzz<F> xyzz_from_jac(Jac<F> p) {
    Xyzz<F> r;
    r.x = p.x;
    r.y = p.y;
    r.zz = f_sqr(p.z);
    r.zzz = f_mul(r.zz, p.z);
    return r;
}

// dbl-2009-l with a = 0. The identity exits early for the reason `pt_mul_glv` gives: the
// ladder doubles an identity accumulator until the scalar's first nonzero digit, and
// the exit keeps those doublings nearly free. A 2-torsion point (Y == 0) would come out
// as Z3 == 0, the identity, which is correct and which BN254's odd-order groups never
// reach anyway.
template <typename F>
__device__ __forceinline__ Jac<F> jac_dbl(Jac<F> p) {
    if (jac_is_zero(p)) {
        return p;
    }
    F a = f_sqr(p.x);
    F b = f_sqr(p.y);
    F c = f_sqr(b);
    F t = f_sub(f_sub(f_sqr(f_add(p.x, b)), a), c);
    F d = f_add(t, t);
    F e = f_add(f_add(a, a), a);
    F f = f_sqr(e);
    F c4 = f_add(c, c);
    c4 = f_add(c4, c4);
    Jac<F> r;
    r.x = f_sub(f, f_add(d, d));
    r.y = f_sub(f_mul(e, f_sub(d, r.x)), f_add(c4, c4));
    r.z = f_mul(f_add(p.y, p.y), p.z);
    return r;
}

// add-2007-bl. The same-x fork mirrors `pt_add`: equal points double, opposite points
// cancel to the identity, and both are live paths for an accumulator that can land on
// any multiple of the base.
template <typename F>
__device__ __forceinline__ Jac<F> jac_add(Jac<F> a, Jac<F> b) {
    if (jac_is_zero(a)) {
        return b;
    }
    if (jac_is_zero(b)) {
        return a;
    }
    F z1z1 = f_sqr(a.z);
    F z2z2 = f_sqr(b.z);
    F u1 = f_mul(a.x, z2z2);
    F u2 = f_mul(b.x, z1z1);
    F s1 = f_mul(f_mul(a.y, b.z), z2z2);
    F s2 = f_mul(f_mul(b.y, a.z), z1z1);
    F h = f_sub(u2, u1);
    F rr = f_sub(s2, s1);
    if (f_is_zero(h)) {
        if (f_is_zero(rr)) {
            return jac_dbl(a);
        }
        return jac_zero<F>();
    }
    F h2 = f_add(h, h);
    F i = f_sqr(h2);
    F j = f_mul(h, i);
    F r2 = f_add(rr, rr);
    F v = f_mul(u1, i);
    F s1j = f_mul(s1, j);
    Jac<F> out;
    out.x = f_sub(f_sub(f_sqr(r2), j), f_add(v, v));
    out.y = f_sub(f_mul(r2, f_sub(v, out.x)), f_add(s1j, s1j));
    out.z = f_mul(f_sub(f_sub(f_sqr(f_add(a.z, b.z)), z1z1), z2z2), h);
    return out;
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

#endif // G16_BN254_CURVE_CUH
