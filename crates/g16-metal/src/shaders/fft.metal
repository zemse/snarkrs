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
// 1. ONE THREAD PER BUTTERFLY, IN PLACE, DISPATCHED ONCE PER PASS.
//    `ifft` (prepare.rs:320) is `bits` strictly sequential passes over one vector, and a
//    pass is `n/2` independent butterflies on disjoint index pairs. So each pass is one
//    `dispatch_threads` of `n/2` threads over the same buffer, and the passes go into one
//    serial compute encoder where consecutive dispatches are ordered with an implicit
//    barrier. `fft.rs` submits all `bits + 1` of them in a single command buffer: the
//    vector never leaves the device between passes, and a power-20 run pays 85
//    commit-and-waits in total rather than 946.
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
// 3. THE `w^0` BUTTERFLY SKIPS THE LADDER, AND THE DIVERGENCE IS FREE.
//    `[1]P == P`, so butterfly `j == 0` of every group needs no multiplication. That is
//    one lane in `span`, so from `exp >= 6` the branch costs a SIMD group nothing it was
//    not already paying, and at `exp == 1` it is EVERY butterfly and the whole pass is
//    additions. The CPU makes the same skip for the same reason (prepare.rs:303).
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
};

// ---------------------------------------------------------------------------
// One `fftMix` pass
// ---------------------------------------------------------------------------

// `lo, hi <- lo + [w^j] hi, lo - [w^j] hi` for the butterfly `gid` owns.
//
// `gid` splits into the group `gid >> log_span` and the position `j` inside it, which is
// the mapping that makes the pairs disjoint: group `g` occupies `[g << exp, (g+1) << exp)`
// and pairs its lower half with its upper half elementwise. `u` is read before either
// store, so the in-place write of `lo` cannot be seen by this thread's read of it, and no
// other thread touches either index.
template <typename F, uint C>
inline void fft_mix_impl(device Xyzz<F>* a,
                         device const uint* tw,
                         constant FftParams& p,
                         uint gid) {
    if (gid >= (p.n >> 1u)) {
        return;
    }
    uint j = gid & (p.span - 1u);
    uint lo = ((gid >> p.log_span) << (p.log_span + 1u)) + j;
    uint hi = lo + p.span;

    Xyzz<F> u = a[lo];
    Xyzz<F> t = a[hi];
    if (j != 0u) {
        uint k[8];
        uint base = (j << p.tw_shift) * 8u;
        for (uint i = 0; i < 8u; i++) {
            k[i] = tw[base + i];
        }
        t = pt_mul<F, C>(t, k);
    }
    a[lo] = pt_add(u, t);
    a[hi] = pt_add(u, pt_neg(t));
}

// ---------------------------------------------------------------------------
// The `1/n` pass
// ---------------------------------------------------------------------------

// `a[i] <- [1/n] a[i]`, the `fftFinal` scaling. One scalar for the whole dispatch, so it
// arrives as a single standard-form `Scalar` in a buffer rather than through the params
// struct, which holds only `uint`s.
//
// This is 12.8% of the command on the CPU and it is the same 12.8% here: a full-width
// ladder per point either way. It is not fused into the first pass, whose twiddles are all
// one, because scaling both butterfly inputs is `n` ladders and a separate pass is also
// `n` ladders, and the transform is 400x compute bound, so the pass over the vector it
// would save is not worth a second kernel that can be wrong.
template <typename F, uint C>
inline void fft_scale_impl(device Xyzz<F>* a,
                           device const uint* k,
                           constant FftParams& p,
                           uint gid) {
    if (gid >= p.n) {
        return;
    }
    uint s[8];
    for (uint i = 0; i < 8u; i++) {
        s[i] = k[i];
    }
    a[gid] = pt_mul<F, C>(a[gid], s);
}

// ---------------------------------------------------------------------------
// The kernels, one pair per group per compiled window width.
//
// A macro rather than a template, for the reason `CER_LADDER_KERNELS` gives: MSL kernels
// cannot be templates and the host selects a pipeline by name. `fft.rs` builds the same
// names from the same width list `ceremony.rs` uses.
// ---------------------------------------------------------------------------

#define FFT_KERNELS(SUF, FT, PTT, C)                                                      \
    kernel void fft_mix_##SUF(device PTT* a [[buffer(0)]],                                \
                              device const uint* tw [[buffer(1)]],                        \
                              constant FftParams& p [[buffer(2)]],                        \
                              uint gid [[thread_position_in_grid]]) {                     \
        fft_mix_impl<FT, C>(a, tw, p, gid);                                               \
    }                                                                                     \
    kernel void fft_scale_##SUF(device PTT* a [[buffer(0)]],                              \
                                device const uint* k [[buffer(1)]],                       \
                                constant FftParams& p [[buffer(2)]],                      \
                                uint gid [[thread_position_in_grid]]) {                   \
        fft_scale_impl<FT, C>(a, k, p, gid);                                              \
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
