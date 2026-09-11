// Stages 5-9: Pippenger multi-scalar multiplication over BN254 G1 and G2, in MSL.
//
// This file is compiled at runtime, concatenated AFTER `bn254_fr.metal`, which supplies
// `struct Fr` and the `fr_*` routines. There is no include path at runtime, so the host
// pastes the two sources together; the `#ifndef` guard in the prelude makes that safe.
//
// ============================================================================
// THE CONSTANTS BELOW MIRROR crates/g16-metal/src/layout.rs (FQ_MODULUS, FQ_N0).
// `msm.rs` greps this source for the exact lines, so the two cannot drift silently.
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
//    parallelises. Sorting the pairs is what zkonduit's Metal MSM does, on the CPU,
//    through the same unified-memory pages.
//
//    What is implemented here is the third shape, the cuZK one, and it is cheaper than
//    a sort because the keys are already dense small integers: count how many points
//    land in every bucket (32-bit `atomic_fetch_add`, on plain counters, never on a
//    field element), prefix-sum the counts into row offsets, scatter each point index
//    into its bucket's run (again a 32-bit `atomic_fetch_add`, this time on a cursor),
//    and then give one thread exclusive ownership of one bucket. That is a counting
//    sort by bucket index, O(n) rather than O(n log n), and after the scatter every
//    bucket is written by exactly one thread, so the accumulation needs no
//    synchronisation of any kind. Order within a bucket is not preserved, which does not
//    matter because bucket accumulation is commutative.
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
//    addition (madd-2008-s) is 8M + 2S against 7M + 4S for Jacobian madd-2007-bl, and
//    the general addition (add-2008-s) is 12M + 2S against 11M + 5S, with roughly a
//    third of the field additions. Bucket accumulation is essentially all mixed
//    additions, so this is the operation that decides the kernel.
//
//    The alternative is affine batch addition, which is what rapidsnark and sppark use
//    on CPU and which is why our own CPU MSM leaves 11-19% on the table. It is not taken
//    here. A batch inversion is a prefix-product, a single inversion and a suffix pass,
//    so on a GPU it is three more dispatches plus two more passes over the bucket array
//    per accumulation round, and the rounds have to be re-planned as the runs shorten.
//    At the domain sizes this prover targets, where a bare command buffer already costs
//    0.16 ms and an extra dispatch 2-3 us, paying six extra dispatches per window to
//    save field products per addition is not obviously a win, and it is certainly not
//    the first thing to build. The honest statement is that this is untested here, not
//    that it loses.
//
//    Identity is ZZ == 0, which is what a freshly allocated Metal buffer already
//    contains, so a bucket array needs no memset kernel.
//
// 4. THE WINDOW REDUCTION ENDS ON THE HOST.
//    `msm_reduce_*` collapses each window's 2^(c-1) buckets to a single point through a
//    per-thread segment plus a threadgroup tree, so the host reads back `n_windows`
//    points per MSM and does only the Horner combination. The serial tail stays where
//    serial tails belong.

#ifndef G16_MSM_METAL
#define G16_MSM_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fq: the BN254 base field. Same 8 x uint32 CIOS Montgomery layout as `Fr` in the
// prelude, and deliberately a separate copy rather than a template: MSL has no way to
// carry a `constant`-address-space modulus array as a template parameter, and every
// alternative (macros, passing the modulus as an argument) either hides the constants
// from the drift test in `msm.rs` or costs a register file full of moduli.
// ---------------------------------------------------------------------------

#define FQ_LIMBS 8

struct Fq {
    uint v[8];
};

static_assert(sizeof(Fq) == 32, "Fq must be 32 bytes to match layout::PackedFq");

// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
constant uint FQ_N[8] = { 0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u };

// -q^{-1} mod 2^32. Cross-checked against ark-ff in layout.rs.
constant uint FQ_N0 = 0xe4866389u;

// R mod q, the Montgomery representative of 1.
constant uint FQ_R[8] = { 0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u, 0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u };

inline bool fq_is_zero(Fq a) {
    uint acc = 0u;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        acc |= a.v[i];
    }
    return acc == 0u;
}

inline bool fq_eq(Fq a, Fq b) {
    uint acc = 0u;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        acc |= (a.v[i] ^ b.v[i]);
    }
    return acc == 0u;
}

// See fr_cond_sub_n in the prelude for why the borrow test is bit 63 of a wrapped ulong
// and why this is a `select` rather than an `if`.
inline Fq fq_cond_sub_n(Fq a, uint hi) {
    uint red[8];
    ulong borrow = 0;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)FQ_N[i] - borrow;
        red[i] = (uint)d;
        borrow = (d >> 63) & (ulong)1;
    }
    bool take = (hi != 0u) || (borrow == 0);
    Fq out;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = select(a.v[i], red[i], take);
    }
    return out;
}

inline Fq fq_add(Fq a, Fq b) {
    Fq s;
    ulong c = 0;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        ulong t = (ulong)a.v[i] + (ulong)b.v[i] + c;
        s.v[i] = (uint)t;
        c = t >> 32;
    }
    return fq_cond_sub_n(s, (uint)c);
}

inline Fq fq_sub(Fq a, Fq b) {
    uint d[8];
    ulong borrow = 0;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        ulong t = (ulong)a.v[i] - (ulong)b.v[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 63) & (ulong)1;
    }
    uint mask = (uint)(0u - (uint)borrow);
    Fq out;
    ulong c = 0;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        ulong t = (ulong)d[i] + (ulong)(FQ_N[i] & mask) + c;
        out.v[i] = (uint)t;
        c = t >> 32;
    }
    return out;
}

inline Fq fq_zero() {
    Fq z;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        z.v[i] = 0u;
    }
    return z;
}

inline Fq fq_one() {
    Fq o;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        o.v[i] = FQ_R[i];
    }
    return o;
}

inline Fq fq_neg(Fq a) {
    return fq_sub(fq_zero(), a);
}

// CIOS Montgomery product, identical in shape to fr_mul. See the prelude for the
// justification of the 32-bit limb with a 64-bit accumulator.
inline Fq fq_mul(Fq a, Fq b) {
    uint t[FQ_LIMBS + 2];
    for (uint i = 0; i < FQ_LIMBS + 2; i++) {
        t[i] = 0u;
    }

    for (uint i = 0; i < FQ_LIMBS; i++) {
        uint bi = b.v[i];

        ulong c = 0;
        for (uint j = 0; j < FQ_LIMBS; j++) {
            ulong r = (ulong)t[j] + (ulong)a.v[j] * (ulong)bi + c;
            t[j] = (uint)r;
            c = r >> 32;
        }
        ulong r = (ulong)t[FQ_LIMBS] + c;
        t[FQ_LIMBS] = (uint)r;
        t[FQ_LIMBS + 1] = (uint)(r >> 32);

        uint m = t[0] * FQ_N0;
        ulong d = (ulong)t[0] + (ulong)m * (ulong)FQ_N[0];
        c = d >> 32;
        for (uint j = 1; j < FQ_LIMBS; j++) {
            ulong r2 = (ulong)t[j] + (ulong)m * (ulong)FQ_N[j] + c;
            t[j - 1] = (uint)r2;
            c = r2 >> 32;
        }
        ulong r3 = (ulong)t[FQ_LIMBS] + c;
        t[FQ_LIMBS - 1] = (uint)r3;
        t[FQ_LIMBS] = t[FQ_LIMBS + 1] + (uint)(r3 >> 32);
    }

    Fq out;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = t[i];
    }
    return fq_cond_sub_n(out, t[FQ_LIMBS]);
}

inline Fq fq_sqr(Fq a) {
    return fq_mul(a, a);
}

// ---------------------------------------------------------------------------
// Fq2 = Fq[u] / (u^2 + 1). BN254's G2 lives over this, so stage 6 needs it and the
// field prelude does not have it.
//
// The reduction is `u^2 = -1`, so a multiplication is three Fq multiplications through
// Karatsuba rather than four, and a squaring is two.
// ---------------------------------------------------------------------------

struct Fq2 {
    Fq c0;
    Fq c1;
};

static_assert(sizeof(Fq2) == 64, "Fq2 must be 64 bytes to match layout::PackedFq2");

inline Fq2 fq2_zero() {
    Fq2 r;
    r.c0 = fq_zero();
    r.c1 = fq_zero();
    return r;
}

inline Fq2 fq2_one() {
    Fq2 r;
    r.c0 = fq_one();
    r.c1 = fq_zero();
    return r;
}

inline bool fq2_is_zero(Fq2 a) {
    return fq_is_zero(a.c0) && fq_is_zero(a.c1);
}

inline bool fq2_eq(Fq2 a, Fq2 b) {
    return fq_eq(a.c0, b.c0) && fq_eq(a.c1, b.c1);
}

inline Fq2 fq2_add(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_add(a.c0, b.c0);
    r.c1 = fq_add(a.c1, b.c1);
    return r;
}

inline Fq2 fq2_sub(Fq2 a, Fq2 b) {
    Fq2 r;
    r.c0 = fq_sub(a.c0, b.c0);
    r.c1 = fq_sub(a.c1, b.c1);
    return r;
}

inline Fq2 fq2_neg(Fq2 a) {
    Fq2 r;
    r.c0 = fq_neg(a.c0);
    r.c1 = fq_neg(a.c1);
    return r;
}

// (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u.
// Karatsuba: the cross term is (a0 + a1)(b0 + b1) - a0 b0 - a1 b1.
inline Fq2 fq2_mul(Fq2 a, Fq2 b) {
    Fq v0 = fq_mul(a.c0, b.c0);
    Fq v1 = fq_mul(a.c1, b.c1);
    Fq2 r;
    r.c0 = fq_sub(v0, v1);
    r.c1 = fq_sub(fq_sub(fq_mul(fq_add(a.c0, a.c1), fq_add(b.c0, b.c1)), v0), v1);
    return r;
}

// (a0 + a1 u)^2 = (a0 + a1)(a0 - a1) + 2 a0 a1 u.
inline Fq2 fq2_sqr(Fq2 a) {
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
// written once as a template and instantiated for G1 (Fq) and G2 (Fq2). Overloads
// rather than a traits class because MSL has no partial specialisation worth relying on
// and these are all trivially inlined.
// ---------------------------------------------------------------------------

inline Fq  f_add(Fq a, Fq b)   { return fq_add(a, b); }
inline Fq2 f_add(Fq2 a, Fq2 b) { return fq2_add(a, b); }
inline Fq  f_sub(Fq a, Fq b)   { return fq_sub(a, b); }
inline Fq2 f_sub(Fq2 a, Fq2 b) { return fq2_sub(a, b); }
inline Fq  f_mul(Fq a, Fq b)   { return fq_mul(a, b); }
inline Fq2 f_mul(Fq2 a, Fq2 b) { return fq2_mul(a, b); }
inline Fq  f_sqr(Fq a)         { return fq_sqr(a); }
inline Fq2 f_sqr(Fq2 a)        { return fq2_sqr(a); }
inline Fq  f_neg(Fq a)         { return fq_neg(a); }
inline Fq2 f_neg(Fq2 a)        { return fq2_neg(a); }
inline bool f_is_zero(Fq a)    { return fq_is_zero(a); }
inline bool f_is_zero(Fq2 a)   { return fq2_is_zero(a); }
inline bool f_eq(Fq a, Fq b)   { return fq_eq(a, b); }
inline bool f_eq(Fq2 a, Fq2 b) { return fq2_eq(a, b); }

inline void f_set_zero(thread Fq& a)  { a = fq_zero(); }
inline void f_set_zero(thread Fq2& a) { a = fq2_zero(); }
inline void f_set_one(thread Fq& a)   { a = fq_one(); }
inline void f_set_one(thread Fq2& a)  { a = fq2_one(); }

// ---------------------------------------------------------------------------
// Points.
//
// `Aff<Fq>` is 64 bytes and `Aff<Fq2>` is 128, matching layout::PackedG1Affine and
// layout::PackedG2Affine exactly, and (0, 0) is the point at infinity in both (it is off
// curve for y^2 = x^3 + b with b != 0, so the encoding is unambiguous, not a convention).
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
inline Xyzz<F> pt_zero() {
    Xyzz<F> r;
    f_set_zero(r.x);
    f_set_zero(r.y);
    f_set_zero(r.zz);
    f_set_zero(r.zzz);
    return r;
}

template <typename F>
inline bool pt_is_zero(Xyzz<F> p) {
    return f_is_zero(p.zz);
}

template <typename F>
inline bool aff_is_inf(Aff<F> p) {
    return f_is_zero(p.x) && f_is_zero(p.y);
}

template <typename F>
inline Xyzz<F> pt_from_affine(Aff<F> p) {
    Xyzz<F> r;
    r.x = p.x;
    r.y = p.y;
    f_set_one(r.zz);
    f_set_one(r.zzz);
    return r;
}

template <typename F>
inline Xyzz<F> pt_neg(Xyzz<F> p) {
    Xyzz<F> r = p;
    r.y = f_neg(p.y);
    return r;
}

// mdbl-2008-s: doubling an affine point straight into XYZZ. a = 0 for BN254, on both G1
// and G2, so the `a * ZZ^2` term of the general doubling disappears.
template <typename F>
inline Xyzz<F> pt_dbl_affine(Aff<F> p) {
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

// dbl-2008-s-1 with a = 0, 6M + 3S.
template <typename F>
inline Xyzz<F> pt_dbl(Xyzz<F> p) {
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

// madd-2008-s: XYZZ += affine, 8M + 2S. fq_sqr is fq_mul, so G1 pays ten products.
// This is the inner loop of bucket accumulation and the hottest routine in the backend.
template <typename F>
inline Xyzz<F> pt_madd(Xyzz<F> acc, Aff<F> p) {
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
inline Xyzz<F> pt_add(Xyzz<F> a, Xyzz<F> b) {
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

// k * P for a small k (below 2^16 here: it is a bucket index inside one window).
// Plain MSB-first double-and-add. It runs once per reduce thread, against a whole
// segment of bucket additions, so a windowed ladder would not pay for itself.
template <typename F>
inline Xyzz<F> pt_mul_small(Xyzz<F> p, uint k) {
    Xyzz<F> acc = pt_zero<F>();
    if (k == 0u || pt_is_zero(p)) {
        return acc;
    }
    uint hi = 31u - clz(k);
    for (int i = (int)hi; i >= 0; i--) {
        acc = pt_dbl(acc);
        if ((k >> (uint)i) & 1u) {
            acc = pt_add(acc, p);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Jacobian with a = 0, for the fixed-window ladder in `ceremony.metal`.
//
// The Pippenger kernels above stay in XYZZ, where the mixed addition that decides them
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
inline Jac<F> jac_zero() {
    Jac<F> r;
    f_set_zero(r.x);
    f_set_zero(r.y);
    f_set_zero(r.z);
    return r;
}

template <typename F>
inline bool jac_is_zero(Jac<F> p) {
    return f_is_zero(p.z);
}

template <typename F>
inline Jac<F> jac_neg(Jac<F> p) {
    Jac<F> r = p;
    r.y = f_neg(p.y);
    return r;
}

// XYZZ -> Jacobian without the inversion `Z = ZZZ / ZZ` would take: scaling the class
// by ZZ*ZZZ gives Z' = ZZ*ZZZ, X' = X*ZZ*ZZZ^2, Y' = Y*ZZ^3*ZZZ^2, and
// X'/Z'^2 == X/ZZ, Y'/Z'^3 == Y/ZZZ. A zero ZZ stays a zero Z'.
template <typename F>
inline Jac<F> jac_from_xyzz(Xyzz<F> p) {
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
inline Xyzz<F> xyzz_from_jac(Jac<F> p) {
    Xyzz<F> r;
    r.x = p.x;
    r.y = p.y;
    r.zz = f_sqr(p.z);
    r.zzz = f_mul(r.zz, p.z);
    return r;
}

// dbl-2009-l with a = 0. The identity exits early for the reason `pt_mul` gives: the
// ladder doubles an identity accumulator until the scalar's first nonzero digit, and
// the exit keeps those doublings nearly free. A 2-torsion point (Y == 0) would come out
// as Z3 == 0, the identity, which is correct and which BN254's odd-order groups never
// reach anyway.
template <typename F>
inline Jac<F> jac_dbl(Jac<F> p) {
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
inline Jac<F> jac_add(Jac<F> a, Jac<F> b) {
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
// bucket index |d_i| - 1 is in [0, 2^(c-1) - 1] and there are 2^(c-1) buckets.
// ---------------------------------------------------------------------------

// `width` bits of a 256-bit little-endian value starting at `bit_off`. Reads past the top
// as zero, which is what makes the highest window cheap instead of a special case.
inline uint sc_read_bits(thread const uint* v, uint bit_off, uint width) {
    uint idx = bit_off >> 5;
    if (idx >= 8u) {
        return 0u;
    }
    uint sh = bit_off & 31u;
    ulong buf = (ulong)v[idx] >> sh;
    if (sh + width > 32u && idx + 1u < 8u) {
        buf |= (ulong)v[idx + 1u] << (32u - sh);
    }
    return (uint)(buf & (((ulong)1 << width) - (ulong)1));
}

// Signed digit `i`, returned split into magnitude and sign so the caller never handles a
// negative index. `mag == 0` means the digit is zero and no bucket is touched.
inline void sc_signed_digit(thread const uint* v, uint i, uint c, thread uint& mag, thread bool& neg) {
    uint off = i * c;
    uint b = sc_read_bits(v, off, c);
    uint carry = (off == 0u) ? 0u : sc_read_bits(v, off - 1u, 1u);
    // The borrow of 2^c when the raw window is in the top half; the neighbour to the
    // right pays it back through the carry bit above.
    bool borrow = ((b >> (c - 1u)) & 1u) != 0u;
    if (borrow) {
        // d = b - 2^c + carry, which is in [-2^(c-1), 0].
        uint m = (1u << c) - b - carry;
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

inline bool sc_is_zero(thread const uint* v) {
    uint acc = 0u;
    for (uint i = 0; i < 8u; i++) {
        acc |= v[i];
    }
    return acc == 0u;
}

inline bool sc_is_one(thread const uint* v) {
    uint acc = 0u;
    for (uint i = 1; i < 8u; i++) {
        acc |= v[i];
    }
    return acc == 0u && v[0] == 1u;
}

// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `msm::MsmParams` in msm.rs.
// ---------------------------------------------------------------------------

struct MsmParams {
    uint n;           // scalars in this MSM
    uint c;           // window width in bits
    uint n_windows;   // ceil(255 / c)
    uint n_buckets;   // 2^(c-1)
    uint cap;         // entries reserved per window, == n
    uint scalar_off;  // element offset into the scalar buffer
    uint base_off;    // element offset into the base buffer
    uint ones_groups; // threadgroups in msm_ones_*
    uint slice_len;   // entries per thread in the segmented accumulation
    uint slices;      // ceil(cap / slice_len), threads per window there
    uint reduce_groups; // threadgroups per window in msm_reduce_*
};

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

kernel void zero_u32(device uint* buf [[buffer(0)]],
                     constant uint& len [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < len) {
        buf[gid] = 0u;
    }
}

// Montgomery limbs (layout::PackedFr) to standard limbs (layout::PackedScalar).
// The NTT leaves H in Montgomery form on the device; Pippenger needs the integer.
kernel void fr_mont_to_std(device const Fr* in [[buffer(0)]],
                           device Fr* out [[buffer(1)]],
                           constant uint& len [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < len) {
        out[gid] = fr_from_mont(in[gid]);
    }
}

// Stage 1 of the counting sort: how many points land in each (window, bucket).
kernel void msm_count(device const uint* scalars [[buffer(0)]],
                      device atomic_uint* counts [[buffer(1)]],
                      constant MsmParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    uint s[8];
    uint base = (p.scalar_off + gid) * 8u;
    for (uint i = 0; i < 8u; i++) {
        s[i] = scalars[base + i];
    }
    // The two cheap classes leave here. A zero has cost one 8-limb read and nothing
    // else; a one is handled by msm_ones_* in a single mixed addition.
    if (sc_is_zero(s) || sc_is_one(s)) {
        return;
    }
    for (uint w = 0; w < p.n_windows; w++) {
        uint mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) {
            continue;
        }
        atomic_fetch_add_explicit(&counts[w * p.n_buckets + (mag - 1u)], 1u, memory_order_relaxed);
    }
}

// Stage 2: exclusive prefix sum of the counts inside each window, biased by that
// window's base offset in the entry array. One threadgroup per window.
//
// SCAN_TG is the array size, not the dispatch size: the host passes
// min(SCAN_TG, pipeline max) threads and the kernel reads the real count from
// [[threads_per_threadgroup]], so a pipeline that reports fewer than 256 still works.
#define SCAN_TG 256

kernel void msm_scan(device const uint* counts [[buffer(0)]],
                     device uint* cursor [[buffer(1)]],
                     constant MsmParams& p [[buffer(2)]],
                     uint w [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tcount [[threads_per_threadgroup]]) {
    threadgroup uint tmp[SCAN_TG];
    uint running = w * p.cap;
    uint chunks = (p.n_buckets + tcount - 1u) / tcount;
    for (uint ch = 0; ch < chunks; ch++) {
        uint idx = ch * tcount + tid;
        uint v = (idx < p.n_buckets) ? counts[w * p.n_buckets + idx] : 0u;
        tmp[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Hillis-Steele inclusive scan. The read is separated from the write by a
        // barrier on both sides, which is what makes the in-place update safe.
        for (uint d = 1; d < tcount; d <<= 1) {
            uint x = (tid >= d) ? tmp[tid - d] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tmp[tid] += x;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (idx < p.n_buckets) {
            cursor[w * p.n_buckets + idx] = running + tmp[tid] - v;
        }
        uint total = tmp[tcount - 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running += total;
    }
}

// Stage 3: scatter each point index into its bucket's run. The cursor is bumped with a
// relaxed 32-bit fetch_add, so points land inside their run in an arbitrary order, which
// is fine because bucket accumulation is commutative. After this kernel `cursor` holds
// each run's END, and the run's start is `cursor - counts`.
//
// An entry is `uint2(row, point << 1 | sign)`, where `row = w * n_buckets + bucket`. The
// row is stored rather than recomputed because the segmented accumulation below walks a
// fixed-length slice of the entry array and has to discover where one bucket's run ends,
// which it cannot do from the point index alone.
kernel void msm_scatter(device const uint* scalars [[buffer(0)]],
                        device atomic_uint* cursor [[buffer(1)]],
                        device uint2* entries [[buffer(2)]],
                        constant MsmParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    uint s[8];
    uint base = (p.scalar_off + gid) * 8u;
    for (uint i = 0; i < 8u; i++) {
        s[i] = scalars[base + i];
    }
    if (sc_is_zero(s) || sc_is_one(s)) {
        return;
    }
    for (uint w = 0; w < p.n_windows; w++) {
        uint mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) {
            continue;
        }
        uint row = w * p.n_buckets + (mag - 1u);
        uint slot = atomic_fetch_add_explicit(&cursor[row], 1u, memory_order_relaxed);
        entries[slot] = uint2(row, (gid << 1) | (neg ? 1u : 0u));
    }
}

// Stage 4, the simple form: one thread owns one bucket, exclusively, so there is nothing
// to synchronise. Correct, and a third of the code of the segmented form below, but the
// dispatch finishes when the fattest bucket does.
//
// MEASURED on this machine, and the numbers are why the segmented kernel exists. On
// js_2x2_d32 at c=11 the busiest bucket holds 11,758 entries against a mean of 17.02, a
// 691x imbalance; this kernel takes 200.99 ms for that MSM and the segmented one takes
// 8.00 ms on identical inputs. End to end over all five MSMs at the tuned window widths,
// this kernel gives 455.9 ms on js_16x16_d32 against 126.8 ms segmented, which is the
// difference between losing to the CPU and beating it 4.2x.
//
// Kept behind G16_METAL_MSM_LEGACY_ACC=1 so that comparison stays reproducible instead
// of being a claim in a comment.
template <typename F>
inline void msm_accumulate_impl(device const uint2* entries,
                                device const Aff<F>* bases,
                                device const uint* counts,
                                device const uint* cursor,
                                device Xyzz<F>* buckets,
                                constant MsmParams& p,
                                uint gid) {
    uint total = p.n_windows * p.n_buckets;
    if (gid >= total) {
        return;
    }
    uint end = cursor[gid];
    uint cnt = counts[gid];
    uint start = end - cnt;
    Xyzz<F> acc = pt_zero<F>();
    for (uint i = start; i < end; i++) {
        uint e = entries[i].y;
        Aff<F> b = bases[p.base_off + (e >> 1)];
        if ((e & 1u) != 0u) {
            b.y = f_neg(b.y);
        }
        acc = pt_madd(acc, b);
    }
    buckets[gid] = acc;
}

kernel void msm_accumulate_g1(device const uint2* entries [[buffer(0)]],
                              device const AffG1* bases [[buffer(1)]],
                              device const uint* counts [[buffer(2)]],
                              device const uint* cursor [[buffer(3)]],
                              device PtG1* buckets [[buffer(4)]],
                              constant MsmParams& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    msm_accumulate_impl<Fq>(entries, bases, counts, cursor, buckets, p, gid);
}

kernel void msm_accumulate_g2(device const uint2* entries [[buffer(0)]],
                              device const AffG2* bases [[buffer(1)]],
                              device const uint* counts [[buffer(2)]],
                              device const uint* cursor [[buffer(3)]],
                              device PtG2* buckets [[buffer(4)]],
                              constant MsmParams& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    msm_accumulate_impl<Fq2>(entries, bases, counts, cursor, buckets, p, gid);
}

// Stage 4, the load-balanced form. This is the kernel the backend actually uses.
//
// THE PROBLEM IT SOLVES, measured rather than assumed. One thread per bucket makes
// per-thread work proportional to bucket occupancy, and occupancy is not uniform: a
// witness contains repeated values, and every copy of one value lands in the same bucket
// of every window. On js_2x2_d32 the fattest bucket holds 11,758 entries against a mean
// of 17. That single thread runs 11,758 serial mixed additions while the other 24,575
// threads finish in about 17 and then wait, and it costs 201 ms of a 253 ms MSM.
//
// THE FIX, which is zkmopro's segmented SMVP adapted to our layout. Slice the entry
// array into fixed-length runs of `slice_len` and give one thread each slice, so
// per-thread work is uniform BY CONSTRUCTION rather than by hoping the digits spread.
// Within its slice a thread finds bucket boundaries by watching `entry.x` change.
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
// fill the machine at our smaller domains.

constant uint MSM_NO_ROW = 0xffffffffu;

// Every bucket has to start at the identity, and unlike a fresh allocation a pooled
// buffer holds the previous proof's points. Only `zz` is written: `pt_add` and the host
// conversion both test `zz` alone, and a bucket that is direct-written is overwritten in
// full anyway, so clearing the other three coordinates would be pure memory traffic.
template <typename F>
inline void msm_clear_impl(device Xyzz<F>* buckets, constant MsmParams& p, uint gid) {
    if (gid < p.n_windows * p.n_buckets) {
        // Built in thread space and then stored: MSL address spaces are part of the
        // type, so a `thread F&` overload cannot bind a `device` member.
        F z;
        f_set_zero(z);
        buckets[gid].zz = z;
    }
}

kernel void msm_clear_g1(device PtG1* buckets [[buffer(0)]],
                         constant MsmParams& p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    msm_clear_impl<Fq>(buckets, p, gid);
}

kernel void msm_clear_g2(device PtG2* buckets [[buffer(0)]],
                         constant MsmParams& p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    msm_clear_impl<Fq2>(buckets, p, gid);
}

template <typename F>
inline void msm_segmented_impl(device const uint2* entries,
                               device const Aff<F>* bases,
                               device const uint* cursor,
                               device Xyzz<F>* buckets,
                               device Xyzz<F>* spill_pts,
                               device uint* spill_rows,
                               constant MsmParams& p,
                               uint gid) {
    uint w = gid / p.slices;
    uint k = gid - w * p.slices;
    if (w >= p.n_windows) {
        return;
    }
    uint base = w * p.cap;
    // The scatter left every cursor at its run's end, so the last bucket's cursor is the
    // end of the whole window region. Slices past it are empty.
    uint used = cursor[w * p.n_buckets + p.n_buckets - 1u] - base;

    uint head_slot = 2u * gid;
    uint tail_slot = head_slot + 1u;
    spill_rows[head_slot] = MSM_NO_ROW;
    spill_rows[tail_slot] = MSM_NO_ROW;

    uint lo = k * p.slice_len;
    if (lo >= used) {
        return;
    }
    uint hi = min(lo + p.slice_len, used);

    uint cur_row = entries[base + lo].x;
    Xyzz<F> acc = pt_zero<F>();
    bool is_first_run = true;

    for (uint i = lo; i < hi; i++) {
        uint2 e = entries[base + i];
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

kernel void msm_segmented_g1(device const uint2* entries [[buffer(0)]],
                             device const AffG1* bases [[buffer(1)]],
                             device const uint* cursor [[buffer(2)]],
                             device PtG1* buckets [[buffer(3)]],
                             device PtG1* spill_pts [[buffer(4)]],
                             device uint* spill_rows [[buffer(5)]],
                             constant MsmParams& p [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    msm_segmented_impl<Fq>(entries, bases, cursor, buckets, spill_pts, spill_rows, p, gid);
}

kernel void msm_segmented_g2(device const uint2* entries [[buffer(0)]],
                             device const AffG2* bases [[buffer(1)]],
                             device const uint* cursor [[buffer(2)]],
                             device PtG2* buckets [[buffer(3)]],
                             device PtG2* spill_pts [[buffer(4)]],
                             device uint* spill_rows [[buffer(5)]],
                             constant MsmParams& p [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    msm_segmented_impl<Fq2>(entries, bases, cursor, buckets, spill_pts, spill_rows, p, gid);
}

// Fold each bucket's spilled partials into it. One thread per bucket, and it looks only
// at the slices its own run overlaps, so there is no search and no atomic.
template <typename F>
inline void msm_merge_impl(device Xyzz<F>* buckets,
                           device const Xyzz<F>* spill_pts,
                           device const uint* spill_rows,
                           device const uint* counts,
                           device const uint* cursor,
                           constant MsmParams& p,
                           uint row) {
    if (row >= p.n_windows * p.n_buckets) {
        return;
    }
    uint cnt = counts[row];
    if (cnt == 0u) {
        return;
    }
    uint w = row / p.n_buckets;
    uint base = w * p.cap;
    uint start = cursor[row] - cnt - base;
    uint end = cursor[row] - base;
    uint k_lo = start / p.slice_len;
    uint k_hi = (end - 1u) / p.slice_len;

    Xyzz<F> acc = buckets[row];
    for (uint k = k_lo; k <= k_hi; k++) {
        uint slot = 2u * (w * p.slices + k);
        if (spill_rows[slot] == row) {
            acc = pt_add(acc, spill_pts[slot]);
        }
        if (spill_rows[slot + 1u] == row) {
            acc = pt_add(acc, spill_pts[slot + 1u]);
        }
    }
    buckets[row] = acc;
}

kernel void msm_merge_g1(device PtG1* buckets [[buffer(0)]],
                         device const PtG1* spill_pts [[buffer(1)]],
                         device const uint* spill_rows [[buffer(2)]],
                         device const uint* counts [[buffer(3)]],
                         device const uint* cursor [[buffer(4)]],
                         constant MsmParams& p [[buffer(5)]],
                         uint row [[thread_position_in_grid]]) {
    msm_merge_impl<Fq>(buckets, spill_pts, spill_rows, counts, cursor, p, row);
}

kernel void msm_merge_g2(device PtG2* buckets [[buffer(0)]],
                         device const PtG2* spill_pts [[buffer(1)]],
                         device const uint* spill_rows [[buffer(2)]],
                         device const uint* counts [[buffer(3)]],
                         device const uint* cursor [[buffer(4)]],
                         constant MsmParams& p [[buffer(5)]],
                         uint row [[thread_position_in_grid]]) {
    msm_merge_impl<Fq2>(buckets, spill_pts, spill_rows, counts, cursor, p, row);
}

// Stage 5: collapse one window's 2^(c-1) buckets to one point.
//
// The window sum is sum_j (j+1) B_j. Split the buckets into one segment per thread, at
// [lo, hi). Inside a segment the reverse running sum gives
// P = sum_j (j - lo + 1) B_j and Q = sum_j B_j in two additions per bucket, and the
// segment contributes P + lo * Q, with `lo` the window-global bucket index, so the
// identity holds no matter how the segments are carved up.
//
// That freedom is what `reduce_groups` uses. One threadgroup per window is 20
// threadgroups at c=13, on a device with more cores than that, and it measured 3.96 ms
// flat in n: pure latency, not work. Each window is therefore split across
// `reduce_groups` threadgroups, each owning a contiguous chunk of its buckets and
// writing one partial to `window_sums[w * reduce_groups + g]`; the host adds the
// partials per window before its Horner combination, a few dozen cheap additions.
//
// REDUCE_TG is the threadgroup array size. 64 rather than 128 is a deliberate occupancy
// choice: at 64 the G2 array is 64 * 256 = 16 KB, half of this device's 32 KB
// threadgroup budget, so two threadgroups still fit per core. At 128 it would be the
// whole budget and only one would.
#define REDUCE_TG 64

template <typename F>
inline void msm_reduce_impl(device const Xyzz<F>* buckets,
                            device Xyzz<F>* window_sums,
                            constant MsmParams& p,
                            threadgroup Xyzz<F>* shared,
                            uint tg,
                            uint tid,
                            uint tcount) {
    uint w = tg / p.reduce_groups;
    uint g = tg - w * p.reduce_groups;
    uint chunk = (p.n_buckets + p.reduce_groups - 1u) / p.reduce_groups;
    uint tg_lo = g * chunk;
    uint tg_hi = min(tg_lo + chunk, p.n_buckets);
    uint seg_len = (chunk + tcount - 1u) / tcount;
    uint lo = tg_lo + tid * seg_len;
    uint hi = min(lo + seg_len, tg_hi);

    Xyzz<F> mine = pt_zero<F>();
    if (lo < hi) {
        Xyzz<F> run = pt_zero<F>();
        Xyzz<F> tot = pt_zero<F>();
        for (uint j = hi; j > lo; j--) {
            run = pt_add(run, buckets[w * p.n_buckets + (j - 1u)]);
            tot = pt_add(tot, run);
        }
        mine = pt_add(tot, pt_mul_small(run, lo));
    }
    shared[tid] = mine;

    for (uint s = 1; s < tcount; s <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & ((s << 1) - 1u)) == 0u && tid + s < tcount) {
            shared[tid] = pt_add(shared[tid], shared[tid + s]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        window_sums[tg] = shared[0];
    }
}

kernel void msm_reduce_g1(device const PtG1* buckets [[buffer(0)]],
                          device PtG1* window_sums [[buffer(1)]],
                          constant MsmParams& p [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG1 shared[REDUCE_TG];
    msm_reduce_impl<Fq>(buckets, window_sums, p, shared, tg, tid, tcount);
}

kernel void msm_reduce_g2(device const PtG2* buckets [[buffer(0)]],
                          device PtG2* window_sums [[buffer(1)]],
                          constant MsmParams& p [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG2 shared[REDUCE_TG];
    msm_reduce_impl<Fq2>(buckets, window_sums, p, shared, tg, tid, tcount);
}

// The scalar-of-1 path: sum the bases whose scalar is exactly 1, one mixed addition
// each. Strided so consecutive lanes read consecutive scalars, then the same threadgroup
// tree as the reduction, so the host adds only `ones_groups` points.
template <typename F>
inline void msm_ones_impl(device const uint* scalars,
                          device const Aff<F>* bases,
                          device Xyzz<F>* out,
                          constant MsmParams& p,
                          threadgroup Xyzz<F>* shared,
                          uint g,
                          uint tid,
                          uint tcount) {
    uint stride = p.ones_groups * tcount;
    Xyzz<F> acc = pt_zero<F>();
    for (uint i = g * tcount + tid; i < p.n; i += stride) {
        uint s[8];
        uint base = (p.scalar_off + i) * 8u;
        for (uint k = 0; k < 8u; k++) {
            s[k] = scalars[base + k];
        }
        if (!sc_is_one(s)) {
            continue;
        }
        acc = pt_madd(acc, bases[p.base_off + i]);
    }
    shared[tid] = acc;
    for (uint s = 1; s < tcount; s <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & ((s << 1) - 1u)) == 0u && tid + s < tcount) {
            shared[tid] = pt_add(shared[tid], shared[tid + s]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        out[g] = shared[0];
    }
}

kernel void msm_ones_g1(device const uint* scalars [[buffer(0)]],
                        device const AffG1* bases [[buffer(1)]],
                        device PtG1* out [[buffer(2)]],
                        constant MsmParams& p [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG1 shared[REDUCE_TG];
    msm_ones_impl<Fq>(scalars, bases, out, p, shared, g, tid, tcount);
}

kernel void msm_ones_g2(device const uint* scalars [[buffer(0)]],
                        device const AffG2* bases [[buffer(1)]],
                        device PtG2* out [[buffer(2)]],
                        constant MsmParams& p [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG2 shared[REDUCE_TG];
    msm_ones_impl<Fq2>(scalars, bases, out, p, shared, g, tid, tcount);
}

// The gather form of the ones path, used when the host classified the scalars: `idx`
// holds the base index of every live one-scalar contribution, with the bases at
// infinity already dropped, and `p.n` is its length. The scan form above pays a
// 32-byte scalar load and a dead branch for every zero and every infinity base; here
// every iteration is a real mixed addition, so the dependent per-thread chain is as
// short as the work allows.
template <typename F>
inline void msm_ones_idx_impl(device const uint* idx,
                              device const Aff<F>* bases,
                              device Xyzz<F>* out,
                              constant MsmParams& p,
                              threadgroup Xyzz<F>* shared,
                              uint g,
                              uint tid,
                              uint tcount) {
    uint stride = p.ones_groups * tcount;
    Xyzz<F> acc = pt_zero<F>();
    for (uint i = g * tcount + tid; i < p.n; i += stride) {
        acc = pt_madd(acc, bases[idx[i]]);
    }
    shared[tid] = acc;
    for (uint s = 1; s < tcount; s <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & ((s << 1) - 1u)) == 0u && tid + s < tcount) {
            shared[tid] = pt_add(shared[tid], shared[tid + s]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        out[g] = shared[0];
    }
}

kernel void msm_ones_idx_g1(device const uint* idx [[buffer(0)]],
                            device const AffG1* bases [[buffer(1)]],
                            device PtG1* out [[buffer(2)]],
                            constant MsmParams& p [[buffer(3)]],
                            uint g [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG1 shared[REDUCE_TG];
    msm_ones_idx_impl<Fq>(idx, bases, out, p, shared, g, tid, tcount);
}

kernel void msm_ones_idx_g2(device const uint* idx [[buffer(0)]],
                            device const AffG2* bases [[buffer(1)]],
                            device PtG2* out [[buffer(2)]],
                            constant MsmParams& p [[buffer(3)]],
                            uint g [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tcount [[threads_per_threadgroup]]) {
    threadgroup PtG2 shared[REDUCE_TG];
    msm_ones_idx_impl<Fq2>(idx, bases, out, p, shared, g, tid, tcount);
}

#endif // G16_MSM_METAL
