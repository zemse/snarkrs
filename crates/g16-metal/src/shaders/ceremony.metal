// The three ceremony primitives that are not a multiexp, in MSL.
//
// Compiled at runtime after `bn254_fr.metal` and `msm.metal`, in that order: this file
// adds no curve arithmetic of its own. `Aff<F>`, `Xyzz<F>`, `Jac<F>`, `pt_add`,
// `pt_from_affine`, the `jac_*` ladder representation, the `f_*` field overloads and the
// signed-digit recoding all come from `msm.metal` unchanged, which is what makes a
// mismatch against the CPU provably a scheduling bug here rather than a formula bug
// there.
//
// ============================================================================
// NOTHING IN THIS FILE MAY BE SHARED INTO g16-wgpu.
// A fixed-window ladder inlines `pt_add` and `pt_dbl` far more than three times over
// `Xyzz<Fq2>`, which is exactly 256 bytes. That is the shape of WebKit 323560, filed by
// this project: 3 or more inlined point-op call sites over a 256-byte struct miscompile
// on iOS. Metal on macOS and CUDA only.
// ============================================================================
//
// WHAT THIS FILE DECIDES, AND WHY
//
// 1. A FIXED SIGNED WINDOW, NOT wNAF.
//    `g16_ceremony::prepare::point_times_fr` (prepare.rs:149) is wNAF with an odd-multiple
//    table, and wNAF is the right choice on a core: measured over 4000 random `Fr`, width 5
//    leaves 42.7 nonzero digits in 252, so a core does 42.7 additions instead of 254.
//    A SIMD lane does not get that. With 32 lanes each holding a different scalar the
//    chance that some lane in the group has a nonzero digit at a given ladder position is
//    99.7%, so a ported wNAF costs the whole group 254 additions and only ever skips the
//    ones a lane did not need. A fixed window has no data-dependent skip to lose: every
//    lane adds once per window, at the same window, always.
//
//    The recoding is `sc_signed_digit` from `msm.metal`, digit for digit, so the ladder
//    and the Pippenger decomposition cannot disagree about what a scalar means.
//
// 2. THE WINDOW IS A TEMPLATE PARAMETER AND EVERY WIDTH IS COMPILED.
//    `Xyzz<Fq>` is 128 bytes and `Xyzz<Fq2>` is 256, so a table of `2^(c-1)` multiples is
//    1 KB per thread at c=4 on G1 and at c=3 on G2. That is past what the register file
//    holds and the spill is what actually decides the width, not the multiply count, so
//    the width cannot be reasoned out. `ceremony.rs` sweeps the compiled widths, and
//    `WINDOW_G2` there says what that sweep can and cannot be read for.
//
// 3. THE ONE INVERSION PER BATCH GOES BACK TO THE HOST.
//    There is no `f_inv` in any shader in this crate and this file does not add one. A
//    Fermat inverse is a 254-bit exponentiation, about 370 `Fq` multiplies, and Montgomery's
//    trick needs exactly ONE inversion for a whole batch. Spending 370 multiplies per
//    thread to avoid a single 0.149 ms round trip per chunk is the wrong trade at every
//    size this project runs: at the 2^18 points a chunk holds, the round trip is 0.0006 us
//    a point. `cer_affine_prefix_*` stops at the segment products, the host inverts one
//    element and seeds each segment, and `cer_affine_finish_*` walks back.
//
// 4. THE GEOMETRIC KEY IS RECOMPUTED PER THREAD, NOT CARRIED.
//    `first * inc^i` is a serial recurrence and a serial recurrence does not chunk. Each
//    thread raises `inc` to its own global index instead: at most 23 squarings and 23
//    multiplies in `Fr` against roughly 3,300 in `Fq` for the point multiplication that
//    follows, so it is under 1.5% of the thread and it makes the dispatch index-invariant.

#ifndef G16_CEREMONY_METAL
#define G16_CEREMONY_METAL

#include <metal_stdlib>
using namespace metal;

// Bits the signed recoding is laid out over. Must equal `g16_msm::RECODE_BITS` and the
// `RECODE_BITS` in `msm.rs`; `ceremony.rs` greps this exact line so the three cannot drift.
#define CER_RECODE_BITS 255

// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `ceremony::CerParams` in ceremony.rs.
// ---------------------------------------------------------------------------

struct CerParams {
    uint n;         // points in this dispatch
    uint seg_len;   // points one thread owns in the two batch-to-affine passes
    uint segments;  // ceil(n / seg_len)
    uint index_off; // global index of point 0, so a chunk knows its own exponents
    uint has_inc;   // 0 when inc == 1 and the key is constant across the whole array
    Fr first;       // Montgomery, as layout::PackedFr hands it over
    Fr inc;         // Montgomery
};

// ---------------------------------------------------------------------------
// General point scalar multiplication
// ---------------------------------------------------------------------------

// `base^e` in Fr, Montgomery in and out. `e` is a point index, so it is under 2^23 here
// and the loop is 23 iterations, not 254.
inline Fr fr_pow_u32(Fr base, uint e) {
    Fr r = fr_one();
    while (e != 0u) {
        if ((e & 1u) != 0u) {
            r = fr_mul(r, base);
        }
        base = fr_sqr(base);
        e >>= 1u;
    }
    return r;
}

// `[k] p` for a full 254-bit `k` in STANDARD form (layout::PackedScalar), fixed signed
// window of width `C`.
//
// The CPU twin is `point_times_fr`, which this must agree with as a curve point, not limb
// for limb: the representations here are projective and the two ladders reach the same
// point through different representatives. The ceremony's byte-identity survives that
// because every path out of here ends in `cer_affine_finish_*`, and affine is canonical.
//
// The point arrives and leaves in XYZZ, the crate's interchange representation, but the
// ladder itself runs in Jacobian with a = 0 (`jac_*` in `msm.metal`): the doubling chain
// is ~82% of the ladder's multiplies and dbl-2009-l is 7 against XYZZ's 9, which the
// slightly dearer general addition does not eat. The conversions cost 8 multiplies once
// per call against ~3,000 inside it.
//
// `tbl[i]` holds `(i+1) p`, so digit magnitude `m` in `1 ..= 2^(C-1)` indexes `tbl[m-1]`
// directly. An even multiple is one doubling of half of it and an odd one is an addition,
// which is 7 multiplies instead of 16 on half the table.
template <typename F, uint C>
inline Xyzz<F> pt_mul(Xyzz<F> p, thread const uint* k) {
    if (pt_is_zero(p)) {
        return pt_zero<F>();
    }
    Jac<F> q = jac_from_xyzz(p);
    Jac<F> acc = jac_zero<F>();

    Jac<F> tbl[1u << (C - 1u)];
    tbl[0] = q;
    for (uint i = 1u; i < (1u << (C - 1u)); i++) {
        tbl[i] = ((i & 1u) != 0u) ? jac_dbl(tbl[(i - 1u) >> 1u]) : jac_add(tbl[i - 1u], q);
    }

    const uint nw = (CER_RECODE_BITS + C - 1u) / C;
    for (uint w = 0; w < nw; w++) {
        for (uint j = 0; j < C; j++) {
            // A no-op while `acc` is still the identity, which `jac_dbl` exits on, so the
            // top window pays nothing for the loop staying uniform across every lane.
            acc = jac_dbl(acc);
        }
        uint mag;
        bool neg;
        sc_signed_digit(k, nw - 1u - w, C, mag, neg);
        if (mag != 0u) {
            Jac<F> t = tbl[mag - 1u];
            acc = jac_add(acc, neg ? jac_neg(t) : t);
        }
    }
    return xyzz_from_jac(acc);
}

// ---------------------------------------------------------------------------
// Batch projective to affine
// ---------------------------------------------------------------------------

// Montgomery's trick, split so the host does the one inversion. Pass 1: each thread owns
// `seg_len` consecutive points, writes the running product of every `t = ZZ*ZZZ` before
// each of them, and leaves its segment's whole product behind.
//
// `t` is one accumulator entry per point, not two, for the reason `batch_to_affine`
// (prepare.rs:198) documents: XYZZ wants both `ZZ^-1` and `ZZZ^-1`, and inverting their
// product gives `ZZ^-1 = ZZZ*t^-1` and `ZZZ^-1 = ZZ*t^-1` from the one value.
//
// Points at infinity are skipped rather than multiplied in, so a zero never enters the
// accumulator and poisons the whole segment. They are a live path here: ptau section 12's
// `power+1` block is padded with them.
template <typename F>
inline void cer_affine_prefix_impl(device const Xyzz<F>* pts,
                                   device F* prefix,
                                   device F* segprod,
                                   constant CerParams& p,
                                   uint gid) {
    if (gid >= p.segments) {
        return;
    }
    uint start = gid * p.seg_len;
    uint end = min(start + p.seg_len, p.n);
    F acc;
    f_set_one(acc);
    for (uint i = start; i < end; i++) {
        prefix[i] = acc;
        Xyzz<F> q = pts[i];
        if (!pt_is_zero(q)) {
            acc = f_mul(acc, f_mul(q.zz, q.zzz));
        }
    }
    segprod[gid] = acc;
}

// Pass 2: `seeds[gid]` is the inverse of the product of every `t` up to the end of this
// segment, which the host built from one inversion and a suffix walk over `segprod`. The
// backward pass is then identical to `batch_to_affine_serial`.
template <typename F>
inline void cer_affine_finish_impl(device const Xyzz<F>* pts,
                                   device const F* prefix,
                                   device const F* seeds,
                                   device Aff<F>* out,
                                   constant CerParams& p,
                                   uint gid) {
    if (gid >= p.segments) {
        return;
    }
    uint start = gid * p.seg_len;
    uint end = min(start + p.seg_len, p.n);
    F inv = seeds[gid];
    for (uint i = end; i > start; i--) {
        uint j = i - 1u;
        Xyzz<F> q = pts[j];
        Aff<F> a;
        if (pt_is_zero(q)) {
            f_set_zero(a.x);
            f_set_zero(a.y);
            out[j] = a;
            continue;
        }
        F t = f_mul(q.zz, q.zzz);
        F t_inv = f_mul(inv, prefix[j]);
        inv = f_mul(inv, t);
        a.x = f_mul(f_mul(q.x, q.zzz), t_inv);
        a.y = f_mul(f_mul(q.y, q.zz), t_inv);
        out[j] = a;
    }
}

kernel void cer_affine_prefix_g1(device const PtG1* pts [[buffer(0)]],
                                 device Fq* prefix [[buffer(1)]],
                                 device Fq* segprod [[buffer(2)]],
                                 constant CerParams& p [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    cer_affine_prefix_impl<Fq>(pts, prefix, segprod, p, gid);
}

kernel void cer_affine_prefix_g2(device const PtG2* pts [[buffer(0)]],
                                 device Fq2* prefix [[buffer(1)]],
                                 device Fq2* segprod [[buffer(2)]],
                                 constant CerParams& p [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    cer_affine_prefix_impl<Fq2>(pts, prefix, segprod, p, gid);
}

kernel void cer_affine_finish_g1(device const PtG1* pts [[buffer(0)]],
                                 device const Fq* prefix [[buffer(1)]],
                                 device const Fq* seeds [[buffer(2)]],
                                 device AffG1* out [[buffer(3)]],
                                 constant CerParams& p [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    cer_affine_finish_impl<Fq>(pts, prefix, seeds, out, p, gid);
}

kernel void cer_affine_finish_g2(device const PtG2* pts [[buffer(0)]],
                                 device const Fq2* prefix [[buffer(1)]],
                                 device const Fq2* seeds [[buffer(2)]],
                                 device AffG2* out [[buffer(3)]],
                                 constant CerParams& p [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    cer_affine_finish_impl<Fq2>(pts, prefix, seeds, out, p, gid);
}

// ---------------------------------------------------------------------------
// The two ladder kernels, one pair per compiled window width.
//
// A macro rather than a template because MSL kernels cannot be templates, and the host
// selects a pipeline by name (`cer_point_mul_g1_c4` and so on). `ceremony.rs` builds the
// same names from the same width list.
// ---------------------------------------------------------------------------

#define CER_LADDER_KERNELS(SUF, FT, AFFT, PTT, C)                                         \
    kernel void cer_point_mul_##SUF(device const PTT* pts [[buffer(0)]],                  \
                                    device const uint* scalars [[buffer(1)]],             \
                                    device PTT* out [[buffer(2)]],                        \
                                    constant CerParams& p [[buffer(3)]],                  \
                                    uint gid [[thread_position_in_grid]]) {               \
        if (gid >= p.n) {                                                                 \
            return;                                                                       \
        }                                                                                 \
        uint k[8];                                                                        \
        for (uint i = 0; i < 8u; i++) {                                                   \
            k[i] = scalars[gid * 8u + i];                                                 \
        }                                                                                 \
        out[gid] = pt_mul<FT, C>(pts[gid], k);                                            \
    }                                                                                     \
    kernel void cer_apply_key_##SUF(device const AFFT* pts [[buffer(0)]],                 \
                                    device PTT* out [[buffer(1)]],                        \
                                    constant CerParams& p [[buffer(2)]],                  \
                                    uint gid [[thread_position_in_grid]]) {               \
        if (gid >= p.n) {                                                                 \
            return;                                                                       \
        }                                                                                 \
        Fr s = p.first;                                                                   \
        if (p.has_inc != 0u) {                                                            \
            s = fr_mul(s, fr_pow_u32(p.inc, p.index_off + gid));                          \
        }                                                                                 \
        Fr std = fr_from_mont(s);                                                         \
        uint k[8];                                                                        \
        for (uint i = 0; i < 8u; i++) {                                                   \
            k[i] = std.v[i];                                                              \
        }                                                                                 \
        AFFT a = pts[gid];                                                                \
        Xyzz<FT> q = aff_is_inf(a) ? pt_zero<FT>() : pt_from_affine(a);                    \
        out[gid] = pt_mul<FT, C>(q, k);                                                   \
    }

CER_LADDER_KERNELS(g1_c2, Fq, AffG1, PtG1, 2u)
CER_LADDER_KERNELS(g1_c3, Fq, AffG1, PtG1, 3u)
CER_LADDER_KERNELS(g1_c4, Fq, AffG1, PtG1, 4u)
CER_LADDER_KERNELS(g1_c5, Fq, AffG1, PtG1, 5u)
CER_LADDER_KERNELS(g2_c2, Fq2, AffG2, PtG2, 2u)
CER_LADDER_KERNELS(g2_c3, Fq2, AffG2, PtG2, 3u)
CER_LADDER_KERNELS(g2_c4, Fq2, AffG2, PtG2, 4u)
CER_LADDER_KERNELS(g2_c5, Fq2, AffG2, PtG2, 5u)

#endif // G16_CEREMONY_METAL
