// The group inverse FFT behind `ptau prepare`, in MSL.
//
// Compiled at runtime after `bn254_fr.metal`, `msm.metal` and `ceremony.metal`, in that
// order. Every field and point operation comes from `msm.metal` unchanged: `Xyzz<F>`,
// `pt_add`, `pt_neg`, `pt_zero`, the `jac_*` Jacobian representation and the signed-digit
// recoding. What this file adds on top of them is one thing, the GLV ladder of item 4,
// and it is the only formula here that a mismatch against the CPU can be blamed on.
//
// ============================================================================
// NOTHING IN THIS FILE MAY BE SHARED INTO g16-wgpu.
// `pt_mul_glv` inlines `jac_add` and `jac_dbl` far more than three times over
// `Jac<Fq2>`, which is 192 bytes, and hands them XYZZ points that are 256. That is the
// shape of WebKit 323560, filed by this project. Metal on macOS and CUDA only.
// ============================================================================
//
// WHAT THIS FILE DECIDES, AND WHY
//
// 1. ONE PASS OF A BLOCK PER COMMAND BUFFER, AND OUT OF PLACE.
//    `ifft` (prepare.rs:320) is `bits` strictly sequential passes over one vector, and a
//    pass is `n/2` independent butterflies on disjoint index pairs. In place, one thread
//    per butterfly is correct and one command buffer could hold every pass of a block:
//    consecutive dispatches on a serial compute encoder are ordered with an implicit
//    barrier, which is the whole of the dependency the transform needs.
//
//    macOS does not allow that, and not for a reason a smaller command buffer fixes. A
//    submission that keeps the GPU busy while the GPU is also driving a display comes back
//    as `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, and it is not a duration
//    limit: measured on this M2 Max, the same 2^19-point G2 block was killed 6.8 s into a
//    run and completed in 16.6 s with exactly the same command-buffer split once the
//    machine was quiet. Shrinking the buffers by an order of magnitude did not save a
//    power-20 run either. The trigger is contention for the display, so the only reliable
//    answer is to make the kill survivable rather than to try to stay under it.
//
//    That is what decides the shape here. A pass reads one buffer and writes another, so a
//    killed command buffer has damaged only the destination and the pass can be run again
//    unchanged; `fft.rs` ping-pongs the two buffers and retries. In place, a kill leaves
//    the vector half-transformed with no way to tell which half. What follows from the
//    same requirement is about one block rather than about one buffer: a buffer holding
//    two passes of the SAME block has already overwritten the first one's input by the
//    time the second runs, while one pass each of several independent blocks stays
//    disjoint and re-runnable. `fft.rs` fills its buffers that way, so the 0.149 ms round
//    trip is spread across a whole round instead of paid per block.
//
//    A pass larger than the ladder budget is split further by `gid_off`. That is a
//    throughput and blast-radius knob, not a correctness one: the pieces stay idempotent
//    because the source buffer is read-only for the whole pass.
//
// 2. THE TWIDDLE COMES OUT OF A TABLE, NOT A RECURRENCE.
//    snarkjs walks `W` forward one `Fr` multiply at a time (`build_fft.js:657-748`), which
//    is serial by construction and which the CPU already replaces with `w^first` per task
//    (prepare.rs:273). A thread here cannot walk at all, so the host builds ONE table of
//    `W^i` for `i < n/2` with `W` the primitive `n`-th root, and pass `exp` reads it by a
//    stride: `roots[exp]^j == W^(j << (bits - exp))`. The table is in STANDARD form, the
//    `frm_fromMontgomery` that `g1m_timesFr` does (`build_bn128.js:60-76`) having been
//    done once on the host where it cannot be silently wrong by a factor of R.
//
// 3. THE `w^0` BUTTERFLY SKIPS THE LADDER, AND THE GRID IS DENSE OVER THE REST.
//    `[1]P == P`, so butterfly `j == 0` of every group needs no multiplication. The CPU
//    makes the same skip (prepare.rs:303), and this file used to take it as a branch in
//    a one-thread-per-butterfly grid, which is free only from `exp >= 6`: below span 32
//    an idle lane rides inside a live SIMD group for the whole pass, so a span-2 pass
//    paid every slot full ladder time and half of them did nothing (measured: a 2^16
//    block's mix pass took 31 ms whether 16,384 or 32,767 slots laddered). The mix grid
//    is therefore dense over the `j != 0` butterflies, with the ladderless ones riding
//    the first `groups` threads; see `fft_mix_impl`.
//
// 4. THE LADDER IS GLV, AND THE LATTICE IS THE HOST'S PROBLEM.
//    BN254 has an endomorphism `phi(x, y) = (beta x, y)`, `beta` a primitive cube root of
//    one in Fq, which acts on the r-torsion as multiplication by a primitive cube root of
//    one in Fr. Its characteristic polynomial `x^2 + x + 1` has two roots mod r and the
//    two eigenspaces are exactly G1 and G2, so the eigenvalue is `lambda` on one and
//    `lambda^2` on the other: one `beta`, two decomposition lattices. `[k]P` becomes
//    `[k1]P + [k2]phi(P)` with both magnitudes under 2^127 (the widest the lattice was
//    measured to produce; see `GLV_RECODE_BITS`), so the doubling chain is 130 long at
//    c=5 rather than 255, against one more addition per window.
//
//    The usual objection to GLV is that the decomposition is paid per multiplication.
//    Here it is not paid at all. The twiddles are powers of a fixed root, the host builds
//    the whole table already, and `twiddle_table` (fft.rs) hands each thread `|k1|`,
//    `|k2|` and two sign bits. Nothing in this file decomposes anything.
//
//    The second window table is free as well, which is what makes GLV affordable in a
//    kernel whose binding constraint is the window table's register footprint. `phi` is a
//    group homomorphism, so `phi(i P) == i phi(P)`: the `phi` side reads the SAME table
//    and applies one `Fq` multiply to X on the way out. Doubling the table would have
//    spent exactly what halving the doublings saved.
//
// 5. BIT REVERSAL AND THE OUTPUT ROTATION ARE NOT KERNELS.
//    `bit_reverse` (prepare.rs:256) and the `a[1..].reverse()` that finishes the inverse
//    (prepare.rs:340) are both pure permutations, and the host is copying every point into
//    and out of a buffer regardless. `fft.rs` folds them into that copy, so neither costs
//    a pass over 268 MB of device memory nor a kernel that can get an index wrong.

#ifndef G16_FFT_METAL
#define G16_FFT_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `fft::FftParams` in fft.rs.
// ---------------------------------------------------------------------------

// `span` rather than the obvious `half`: `half` is MSL's 16-bit float type, and a member
// named after it is a hard compile error inside the struct and an "expected unqualified-id"
// at every use, which is not a message that points at the name.
struct FftParams {
    uint n;        // points in this block, a power of two
    uint span;     // butterflies per group in this pass, 1 << (exp - 1)
    uint log_span; // exp - 1
    uint tw_shift; // bits - exp, the stride into the twiddle table
    uint gid_off;  // index of the first butterfly this dispatch owns
};

// ---------------------------------------------------------------------------
// The GLV ladder
// ---------------------------------------------------------------------------

// Words one twiddle takes in the table: `|k1|`, `|k2|`, then the sign word. Mirrors
// `layout::PackedGlv`, which `fft.rs` greps for this exact line.
#define GLV_WORDS 9

// Bits a magnitude is recoded over. The lattice bounds `|k1|` and `|k2|` by
// `|n11| + |n21|`, which is 1.4795e38, so 127 bits is the true width; 128 is what the
// window count is taken from, and every compiled `C` divides into it with `C * nw >= 128`,
// which is what stops the top window from borrowing off the end of the value.
// `fft.rs` greps this line.
#define GLV_RECODE_BITS 128

// `beta`, a primitive cube root of one in Fq, Montgomery form. Equal to
// `ark_bn254::g1::Config::ENDO_COEFFS[0]` and to the G2 one's `c0`, which `fft.rs`
// asserts rather than trusts. The G2 endomorphism is the same `beta` embedded in Fq2
// with `c1 == 0`, so it is two Fq multiplies there and one here, never a full Fq2 one.
constant uint FQ_BETA[8] = { 0x13e80b9cu, 0x3350c88eu, 0xdb5e56b9u, 0x7dce557cu, 0xb615564au, 0x6001b4b8u, 0x020217e0u, 0x2682e617u };

inline Fq fq_beta() {
    Fq b;
    for (uint i = 0; i < 8u; i++) {
        b.v[i] = FQ_BETA[i];
    }
    return b;
}

inline Fq  f_beta_mul(Fq x)  { return fq_mul(x, fq_beta()); }
inline Fq2 f_beta_mul(Fq2 x) {
    Fq b = fq_beta();
    Fq2 r;
    r.c0 = fq_mul(x.c0, b);
    r.c1 = fq_mul(x.c1, b);
    return r;
}

// `phi(X, Y, Z) = (beta X, Y, Z)`. In Jacobian the point is `(X/Z^2, Y/Z^3)`, so scaling
// X by beta scales the affine x by beta and leaves y alone, which is the endomorphism
// exactly. Negation is on Y, so `phi` and `jac_neg` commute and the digit's sign can be
// applied on either side.
template <typename F>
inline Jac<F> jac_endo(Jac<F> p) {
    Jac<F> r = p;
    r.x = f_beta_mul(p.x);
    return r;
}

// `width` bits at `bit_off` of the 128-bit magnitude in `v[0..4]`, reading past the top
// as zero.
//
// `sc_read_bits` cannot serve: it spans all eight limbs, and here the two magnitudes are
// consecutive halves of one array, so a top window of `|k1|` would pull in the bottom bits
// of `|k2|` and silently produce a different scalar.
inline uint glv_read_bits(thread const uint* v, uint bit_off, uint width) {
    uint idx = bit_off >> 5;
    if (idx >= 4u) {
        return 0u;
    }
    uint sh = bit_off & 31u;
    ulong buf = (ulong)v[idx] >> sh;
    if (sh + width > 32u && idx + 1u < 4u) {
        buf |= (ulong)v[idx + 1u] << (32u - sh);
    }
    return (uint)(buf & (((ulong)1 << width) - (ulong)1));
}

// `sc_signed_digit` over a 128-bit magnitude, digit for digit. Split out only because of
// the read above; the recoding itself is the same one the MSM and the ceremony ladder use.
inline void glv_signed_digit(thread const uint* v, uint i, uint c, thread uint& mag, thread bool& neg) {
    uint off = i * c;
    uint b = glv_read_bits(v, off, c);
    uint carry = (off == 0u) ? 0u : glv_read_bits(v, off - 1u, 1u);
    bool borrow = ((b >> (c - 1u)) & 1u) != 0u;
    if (borrow) {
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

// `[k] p` for a twiddle the host has already put through the lattice: `k[0..4]` is
// `|k1|`, `k[4..8]` is `|k2|`, `sign` bit 0 is `k1 < 0` and bit 1 is `k2 < 0`, and
// `k == +-k1 + lambda * (+-k2)` for that group's `lambda`.
//
// Interleaved, not joint: one accumulator, `C` doublings a window, then up to two
// additions off ONE table of `2^(C-1)` multiples of `p`. The `phi` side reads that same
// table because `phi` is a homomorphism, so the register footprint is a plain fixed-window
// ladder's at the same `C` while the doubling chain is half as long.
//
// The two signs cost nothing. `k1`'s is folded into the base point once, before the table
// is built, and `k2`'s then rides as `flip`: the table is multiples of `s1 p`, so a `phi`
// entry carries an extra factor of `s1` and the digit's own sign is XORed with
// `s1 != s2`. Both are a negated Y.
//
// The point arrives and leaves in XYZZ, and the ladder runs in Jacobian with a = 0, for
// the reason `pt_mul` (ceremony.metal) gives: the doubling chain dominates and dbl-2009-l
// is 7 multiplies against XYZZ's 9. That argument is weaker here, since GLV halves the
// doublings and adds an addition per window, but it is still the right way round.
template <typename F, uint C>
inline Xyzz<F> pt_mul_glv(Xyzz<F> p, thread const uint* k, uint sign) {
    if (pt_is_zero(p)) {
        return pt_zero<F>();
    }
    Jac<F> q = jac_from_xyzz(p);
    if ((sign & 1u) != 0u) {
        q = jac_neg(q);
    }
    bool flip = ((sign ^ (sign >> 1u)) & 1u) != 0u;
    Jac<F> acc = jac_zero<F>();

    Jac<F> tbl[1u << (C - 1u)];
    tbl[0] = q;
    for (uint i = 1u; i < (1u << (C - 1u)); i++) {
        tbl[i] = ((i & 1u) != 0u) ? jac_dbl(tbl[(i - 1u) >> 1u]) : jac_add(tbl[i - 1u], q);
    }

    const uint nw = (GLV_RECODE_BITS + C - 1u) / C;
    for (uint w = 0; w < nw; w++) {
        for (uint j = 0; j < C; j++) {
            // A no-op while `acc` is still the identity, which `jac_dbl` exits on.
            acc = jac_dbl(acc);
        }
        uint mag;
        bool neg;
        glv_signed_digit(k, nw - 1u - w, C, mag, neg);
        if (mag != 0u) {
            Jac<F> t = tbl[mag - 1u];
            acc = jac_add(acc, neg ? jac_neg(t) : t);
        }
        glv_signed_digit(k + 4u, nw - 1u - w, C, mag, neg);
        if (mag != 0u) {
            Jac<F> t = jac_endo(tbl[mag - 1u]);
            acc = jac_add(acc, (neg != flip) ? jac_neg(t) : t);
        }
    }
    return xyzz_from_jac(acc);
}

// ---------------------------------------------------------------------------
// One `fftMix` pass
// ---------------------------------------------------------------------------

// `out[lo], out[hi] <- in[lo] + [w^j] in[hi], in[lo] - [w^j] in[hi]` for one butterfly.
//
// The grid is DENSE over the `j != 0` butterflies, not over all of them. The obvious
// one-thread-per-butterfly map lets the `j == 0` lanes skip the ladder, and that skip
// buys nothing below span 32: an idle lane rides inside a live SIMD group for the whole
// pass, so a span-2 pass costs every thread-slot full ladder time and half of them do no
// work (measured on this M2 Max: a 2^16 block's mix pass took 31 ms whether 16,384 or
// 32,767 of its 32,768 slots actually laddered). So thread `d` here owns ladder
// butterfly `d` of the `groups * (span - 1)` that exist: group `d / (span - 1)`,
// position `1 + d % (span - 1)`. The division is against ~3,000 field multiplies per
// thread and does not show.
//
// The `n / 2^exp` ladderless `j == 0` butterflies ride along on the first `groups`
// threads, two additions each next to a full ladder. At `exp == 1` there is nothing to
// compact, every butterfly is `j == 0` and the pass is dispatched over all of them.
//
// Every index of `out` is still written by exactly one thread (the two families pair
// disjoint indices) and `in` is never written, so any sub-range of a pass can be
// dispatched again after a failure and produce the same answer.
template <typename F, uint C>
inline void fft_mix_impl(device const Xyzz<F>* in,
                         device Xyzz<F>* out,
                         device const uint* tw,
                         constant FftParams& p,
                         uint tid) {
    uint d = tid + p.gid_off;
    uint groups = (p.n >> 1u) >> p.log_span;

    if (p.span == 1u) {
        if (d >= groups) {
            return;
        }
        Xyzz<F> u = in[d << 1u];
        Xyzz<F> t = in[(d << 1u) + 1u];
        out[d << 1u] = pt_add(u, t);
        out[(d << 1u) + 1u] = pt_add(u, pt_neg(t));
        return;
    }

    if (d >= (p.n >> 1u) - groups) {
        return;
    }
    uint g = d / (p.span - 1u);
    uint j = 1u + (d - g * (p.span - 1u));
    uint lo = (g << (p.log_span + 1u)) + j;
    uint hi = lo + p.span;

    uint k[8];
    uint base = (j << p.tw_shift) * GLV_WORDS;
    for (uint i = 0; i < 8u; i++) {
        k[i] = tw[base + i];
    }
    Xyzz<F> t = pt_mul_glv<F, C>(in[hi], k, tw[base + 8u]);
    Xyzz<F> u = in[lo];
    out[lo] = pt_add(u, t);
    out[hi] = pt_add(u, pt_neg(t));

    if (d < groups) {
        uint lo0 = d << (p.log_span + 1u);
        uint hi0 = lo0 + p.span;
        Xyzz<F> u0 = in[lo0];
        Xyzz<F> t0 = in[hi0];
        out[lo0] = pt_add(u0, t0);
        out[hi0] = pt_add(u0, pt_neg(t0));
    }
}

// ---------------------------------------------------------------------------
// The last mix pass, with the `1/n` scaling fused in
// ---------------------------------------------------------------------------

// `out[lo], out[hi] <- [s] in[lo] + [s w^j] in[hi], [s] in[lo] - [s w^j] in[hi]` with
// `s = 1/n`, for pass `exp == bits` only. Exact by distributivity:
// `[s](u + [w^j]v) == [s]u + [s w^j]v`, and the exit through `batch_to_affine` is what
// keeps the different representative from mattering, as ever.
//
// The `fftFinal` scaling used to be its own pass, on the argument that scaling both
// butterfly inputs is `n` ladders and a separate pass is also `n` ladders. What that
// argument misses is the last mix pass it can absorb: separate, the two cost
// `n/2 - 1 + n` ladders; fused they cost `n`, because the host folds `s` into the
// twiddle table (`stw[j] = s * W^j`, so `stw[0] == s` covers the `u` side) and each
// thread runs exactly two ladders. That is `n/2 - 1` ladders saved per block, 1/bits of
// the block's total, and the `j == 0` skip is gone: `[s]v` is a real ladder now, so the
// pass has no idle lanes at all.
//
// The last pass has exactly one group, so `lo` is `gid` and `hi` is `gid + n/2` with no
// group arithmetic. Still out of place and read-only on `in`, so the retry story is
// unchanged; each thread is two ladders, so `fft.rs` halves this pass's thread budget to
// keep command-buffer duration flat.
template <typename F, uint C>
inline void fft_mix_scale_impl(device const Xyzz<F>* in,
                               device Xyzz<F>* out,
                               device const uint* stw,
                               constant FftParams& p,
                               uint tid) {
    uint gid = tid + p.gid_off;
    uint half_n = p.n >> 1u;
    if (gid >= half_n) {
        return;
    }
    uint s[8];
    uint k[8];
    uint base = gid * GLV_WORDS;
    for (uint i = 0; i < 8u; i++) {
        s[i] = stw[i];
        k[i] = stw[base + i];
    }
    Xyzz<F> t = pt_mul_glv<F, C>(in[gid + half_n], k, stw[base + 8u]);
    Xyzz<F> u = pt_mul_glv<F, C>(in[gid], s, stw[8]);
    out[gid] = pt_add(u, t);
    out[gid + half_n] = pt_add(u, pt_neg(t));
}

// ---------------------------------------------------------------------------
// The kernels, one pair per group per compiled window width.
//
// A macro rather than a template, for the reason `CER_LADDER_KERNELS` gives: MSL kernels
// cannot be templates and the host selects a pipeline by name. `fft.rs` builds the same
// names from the same width list `ceremony.rs` uses.
// ---------------------------------------------------------------------------

#define FFT_KERNELS(SUF, FT, PTT, C)                                         \
    kernel void fft_mix_##SUF(device const PTT* in [[buffer(0)]],            \
                              device PTT* out [[buffer(1)]],                 \
                              device const uint* tw [[buffer(2)]],           \
                              constant FftParams& p [[buffer(3)]],           \
                              uint tid [[thread_position_in_grid]]) {        \
        fft_mix_impl<FT, C>(in, out, tw, p, tid);                            \
    }                                                                        \
    kernel void fft_mix_scale_##SUF(device const PTT* in [[buffer(0)]],      \
                                    device PTT* out [[buffer(1)]],           \
                                    device const uint* stw [[buffer(2)]],    \
                                    constant FftParams& p [[buffer(3)]],     \
                                    uint tid [[thread_position_in_grid]]) {  \
        fft_mix_scale_impl<FT, C>(in, out, stw, p, tid);                     \
    }

FFT_KERNELS(g1_c2, Fq, PtG1, 2u)
FFT_KERNELS(g1_c3, Fq, PtG1, 3u)
FFT_KERNELS(g1_c4, Fq, PtG1, 4u)
FFT_KERNELS(g1_c5, Fq, PtG1, 5u)
FFT_KERNELS(g2_c2, Fq2, PtG2, 2u)
FFT_KERNELS(g2_c3, Fq2, PtG2, 3u)
FFT_KERNELS(g2_c4, Fq2, PtG2, 4u)
FFT_KERNELS(g2_c5, Fq2, PtG2, 5u)

#endif // G16_FFT_METAL
