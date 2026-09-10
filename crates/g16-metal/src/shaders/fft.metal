// The group inverse FFT behind `ptau prepare`, in MSL.
//
// Compiled at runtime after `bn254_fr.metal`, `msm.metal` and `ceremony.metal`, in that
// order. Every piece of arithmetic here comes from those three unchanged: `Xyzz<F>`,
// `pt_add`, `pt_neg` and `pt_zero` from `msm.metal`, and the fixed-window ladder
// `pt_mul<F, C>` from `ceremony.metal`. A mismatch against the CPU is therefore a
// scheduling bug in this file, not a formula bug in one of those.
//
// ============================================================================
// NOTHING IN THIS FILE MAY BE SHARED INTO g16-wgpu.
// It calls `pt_mul`, which inlines `pt_add` and `pt_dbl` far more than three times over
// `Xyzz<Fq2>`, which is exactly 256 bytes. That is the shape of WebKit 323560, filed by
// this project. Metal on macOS and CUDA only.
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
// 4. BIT REVERSAL AND THE OUTPUT ROTATION ARE NOT KERNELS.
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
    uint base = (j << p.tw_shift) * 8u;
    for (uint i = 0; i < 8u; i++) {
        k[i] = tw[base + i];
    }
    Xyzz<F> t = pt_mul<F, C>(in[hi], k);
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
    for (uint i = 0; i < 8u; i++) {
        s[i] = stw[i];
        k[i] = stw[gid * 8u + i];
    }
    Xyzz<F> t = pt_mul<F, C>(in[gid + half_n], k);
    Xyzz<F> u = pt_mul<F, C>(in[gid], s);
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
