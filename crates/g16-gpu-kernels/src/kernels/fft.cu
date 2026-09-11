// The group inverse FFT behind `ptau prepare`, in CUDA C.
//
// Port of crates/g16-metal/src/shaders/fft.metal, kernel for kernel. Compiled at run
// time by NVRTC after `bn254_fr.cuh` and `bn254_curve.cuh`, in that order. Every field
// and point operation comes from the curve header unchanged: `Xyzz<F>`, `pt_add`,
// `pt_neg`, `pt_zero` and the `jac_*` Jacobian representation. What this file adds on
// top of them is the GLV ladder, and it is the only formula here that a mismatch
// against the CPU can be blamed on.
//
// The design arguments are in the Metal twin's banner and are not repeated: the dense
// grid over the `j != 0` butterflies, the twiddle table in standard form, GLV with the
// lattice on the host, and the two permutations that are deliberately not kernels. Two
// places this file knowingly differs from that twin:
//
// 1. THE OUT-OF-PLACE, ONE-PASS-AT-A-TIME SHAPE IS INHERITED, NOT REQUIRED HERE. The
//    Metal design is forced by a macOS behaviour: a submission that keeps the GPU busy
//    while the GPU also drives a display is killed with
//    `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, so every pass has to be
//    re-runnable and the host retries. A headless Linux compute card has no analogue,
//    and the CUDA host driver (g16-cuda/src/fft.rs) already drops the retry loop and
//    the ladder budget that exist only to survive that kill. The out-of-place ping-pong
//    is kept anyway, because it is the shape proven byte-identical to the CPU at powers
//    13 to 20 on Metal; whoever collapses it in place is relaxing a Metal constraint,
//    not a CUDA one, and buys device memory rather than time with it.
//
// 2. ONLY ONE WINDOW WIDTH IS COMPILED BY DEFAULT. Metal compiles four widths per group
//    in 54 ms of runtime MSL; NVRTC costs 113.6 s of source-to-PTX plus about 175 s of
//    driver JIT for the MSM unit alone on a cold T4 (g16-cuda/src/context.rs), and
//    `pt_mul_glv` fully inlines `jac_dbl` and `jac_add`, which fully inline the CIOS
//    multiply. Eight instantiations would tax every fresh machine to keep a sweep knob
//    the shipped configuration does not use, so only the shipped c = 5 (swept on Metal
//    through a whole power-16 prepare, fft.rs `FFT_WINDOW`) is always instantiated.
//
// 3. THE EXPERIMENT VARIANTS AT THE BOTTOM, BEHIND `G16_FFT_VARIANTS`. Any edit to this
//    file costs a full NVRTC + ptxas rebuild on the measuring machine (the PTX cache in
//    g16-cuda/src/context.rs keys on the source), so a sweep is affordable only if one
//    compile carries every candidate. The `#ifdef G16_FFT_VARIANTS` block instantiates
//    the candidate set as extra entry points in this same unit; the host defines the
//    guard when `G16_CUDA_FFT_VARIANTS=1` (g16-cuda/src/kernels.rs) and picks an entry
//    point per run via `G16_CUDA_FFT_VARIANT`, so the whole sweep is two cached units
//    (one more with `G16_CUDA_FF_PTX=1`) and every run after the first is free. A user
//    who never sets the env compiles the two shipped entry-point pairs and nothing else.

#ifndef G16_FFT_CU
#define G16_FFT_CU

// ---------------------------------------------------------------------------
// Kernel parameters. Mirrors `fft::FftParams` in g16-cuda/src/fft.rs, passed by value
// the way `MsmParams` is; same field names as the MSL twin, `span` and not `half`, so
// the two files stay a readable diff.
// ---------------------------------------------------------------------------

struct FftParams {
    u32 n;        // points in this block, a power of two
    u32 span;     // butterflies per group in this pass, 1 << (exp - 1)
    u32 log_span; // exp - 1
    u32 tw_shift; // bits - exp, the stride into the twiddle table
    u32 gid_off;  // index of the first butterfly this dispatch owns
};

static_assert(sizeof(FftParams) == 20, "FftParams must be five u32 to match the host struct");

// ---------------------------------------------------------------------------
// The GLV ladder
// ---------------------------------------------------------------------------

// Words one twiddle takes in the table: `|k1|`, `|k2|`, then the sign word. Mirrors
// `g16_gpu_layout::PackedGlv`, which a Rust test greps for this exact line.
#define GLV_WORDS 9

// Bits a magnitude is recoded over. The lattice bounds `|k1|` and `|k2|` by
// `|n11| + |n21|`, which is 1.4795e38, so 127 bits is the true width; 128 is what the
// window count is taken from, and every compiled `C` divides into it with `C * nw >= 128`,
// which is what stops the top window from borrowing off the end of the value.
// A Rust test greps this line.
#define GLV_RECODE_BITS 128

// `beta`, a primitive cube root of one in Fq, Montgomery form. Equal to
// `ark_bn254::g1::Config::ENDO_COEFFS[0]` and to the G2 one's `c0`, which a Rust test
// asserts rather than trusts. The G2 endomorphism is the same `beta` embedded in Fq2
// with `c1 == 0`, so it is two Fq multiplies there and one here, never a full Fq2 one.
__constant__ u32 FQ_BETA[8] = { 0x13e80b9cu, 0x3350c88eu, 0xdb5e56b9u, 0x7dce557cu, 0xb615564au, 0x6001b4b8u, 0x020217e0u, 0x2682e617u };

__device__ __forceinline__ Fq fq_beta() {
    Fq b;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        b.v[i] = FQ_BETA[i];
    }
    return b;
}

__device__ __forceinline__ Fq  f_beta_mul(Fq x)  { return fq_mul(x, fq_beta()); }
__device__ __forceinline__ Fq2 f_beta_mul(Fq2 x) {
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
__device__ __forceinline__ Jac<F> jac_endo(Jac<F> p) {
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
__device__ __forceinline__ u32 glv_read_bits(const u32* v, u32 bit_off, u32 width) {
    u32 idx = bit_off >> 5;
    if (idx >= 4u) {
        return 0u;
    }
    u32 sh = bit_off & 31u;
    u64 buf = (u64)v[idx] >> sh;
    if (sh + width > 32u && idx + 1u < 4u) {
        buf |= (u64)v[idx + 1u] << (32u - sh);
    }
    return (u32)(buf & (((u64)1 << width) - (u64)1));
}

// `sc_signed_digit` over a 128-bit magnitude, digit for digit. Split out only because of
// the read above; the recoding itself is the same one the MSM uses.
__device__ __forceinline__ void glv_signed_digit(const u32* v, u32 i, u32 c, u32& mag, bool& neg) {
    u32 off = i * c;
    u32 b = glv_read_bits(v, off, c);
    u32 carry = (off == 0u) ? 0u : glv_read_bits(v, off - 1u, 1u);
    bool borrow = ((b >> (c - 1u)) & 1u) != 0u;
    if (borrow) {
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
// the reason the Jac section of `bn254_curve.cuh` gives: the doubling chain dominates and
// dbl-2009-l is 7 multiplies against XYZZ's 9. That argument is weaker here, since GLV
// halves the doublings and adds an addition per window, but it is still the right way
// round.
template <typename F, u32 C>
__device__ __forceinline__ Xyzz<F> pt_mul_glv(Xyzz<F> p, const u32* k, u32 sign) {
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
    for (u32 i = 1u; i < (1u << (C - 1u)); i++) {
        tbl[i] = ((i & 1u) != 0u) ? jac_dbl(tbl[(i - 1u) >> 1u]) : jac_add(tbl[i - 1u], q);
    }

    const u32 nw = (GLV_RECODE_BITS + C - 1u) / C;
    for (u32 w = 0; w < nw; w++) {
        for (u32 j = 0; j < C; j++) {
            // A no-op while `acc` is still the identity, which `jac_dbl` exits on.
            acc = jac_dbl(acc);
        }
        u32 mag;
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

// `tbl[mag - 1]` as an unrolled compare-select chain over constant indices, so the array
// is never dynamically indexed and ptxas may keep it in registers. `mag` is in
// [1, 2^(C-1)]; the chain is 2^(C-1) - 1 predicated struct copies against a ~60-multiply
// window body.
template <typename F, u32 C>
__device__ __forceinline__ Jac<F> glv_tbl_select(const Jac<F>* tbl, u32 mag) {
    Jac<F> t = tbl[0];
#pragma unroll
    for (u32 i = 1u; i < (1u << (C - 1u)); i++) {
        if (mag == i + 1u) {
            t = tbl[i];
        }
    }
    return t;
}

// `pt_mul_glv` with the window table pinned out of local memory. `tbl[mag - 1u]` above
// is dynamically indexed, which forces the table into local memory: 1.5 KB per thread on
// G1 and 3 KB on G2 at c = 5, so the tables of a single resident block already exceed
// Turing's whole L1 and every window rereads an entry through L2. Here the build loop
// and both lookups are fully unrolled with constant indices, so the table is eligible
// for the register file instead. That only has a chance of fitting at a narrow width,
// which is why only C == 3 is instantiated: four G1 entries are 96 registers of table,
// plus 48 for the accumulator and base, under the 255 cap with room for the multiply;
// the G2 table alone is 192 and ptxas will spill some of it, which the ptxas -v report
// says before any run does. The price is 43 windows instead of 26, ~34 more window
// additions per ladder. Which side of that trade the T4 takes is exactly what the
// `c3r` variant exists to measure.
template <typename F, u32 C>
__device__ __forceinline__ Xyzz<F> pt_mul_glv_reg(Xyzz<F> p, const u32* k, u32 sign) {
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
#pragma unroll
    for (u32 i = 1u; i < (1u << (C - 1u)); i++) {
        tbl[i] = ((i & 1u) != 0u) ? jac_dbl(tbl[(i - 1u) >> 1u]) : jac_add(tbl[i - 1u], q);
    }

    const u32 nw = (GLV_RECODE_BITS + C - 1u) / C;
    for (u32 w = 0; w < nw; w++) {
        for (u32 j = 0; j < C; j++) {
            acc = jac_dbl(acc);
        }
        u32 mag;
        bool neg;
        glv_signed_digit(k, nw - 1u - w, C, mag, neg);
        if (mag != 0u) {
            Jac<F> t = glv_tbl_select<F, C>(tbl, mag);
            acc = jac_add(acc, neg ? jac_neg(t) : t);
        }
        glv_signed_digit(k + 4u, nw - 1u - w, C, mag, neg);
        if (mag != 0u) {
            Jac<F> t = jac_endo(glv_tbl_select<F, C>(tbl, mag));
            acc = jac_add(acc, (neg != flip) ? jac_neg(t) : t);
        }
    }
    return xyzz_from_jac(acc);
}

// Compile-time ladder choice for the kernels below. `if constexpr` so an entry point
// instantiates only the ladder it uses; a plain `if` would compile both into every one.
template <typename F, u32 C, bool REG>
__device__ __forceinline__ Xyzz<F> fft_ladder(Xyzz<F> p, const u32* k, u32 sign) {
    if constexpr (REG) {
        return pt_mul_glv_reg<F, C>(p, k, sign);
    } else {
        return pt_mul_glv<F, C>(p, k, sign);
    }
}

// ---------------------------------------------------------------------------
// One `fftMix` pass
// ---------------------------------------------------------------------------

// `out[lo], out[hi] <- in[lo] + [w^j] in[hi], in[lo] - [w^j] in[hi]` for one butterfly.
//
// The grid is DENSE over the `j != 0` butterflies, not over all of them. The obvious
// one-thread-per-butterfly map lets the `j == 0` lanes skip the ladder, and that skip
// buys nothing below span 32: an idle lane rides inside a live warp for the whole pass,
// so a span-2 pass costs every thread-slot full ladder time and half of them do no work
// (measured on the Metal twin, whose SIMD group is the same 32 lanes: a 2^16 block's mix
// pass took 31 ms whether 16,384 or 32,767 of its 32,768 slots actually laddered). So
// thread `d` here owns ladder butterfly `d` of the `groups * (span - 1)` that exist:
// group `d / (span - 1)`, position `1 + d % (span - 1)`. The division is against ~3,000
// field multiplies per thread and does not show.
//
// The `n / 2^exp` ladderless `j == 0` butterflies ride along on the first `groups`
// threads, two additions each next to a full ladder. At `exp == 1` there is nothing to
// compact, every butterfly is `j == 0` and the pass is dispatched over all of them.
//
// Every index of `out` is still written by exactly one thread (the two families pair
// disjoint indices) and `in` is never written, so any sub-range of a pass can be
// dispatched again after a failure and produce the same answer.
template <typename F, u32 C, bool REG>
__device__ __forceinline__ void fft_mix_impl(const Xyzz<F>* __restrict__ in,
                                             Xyzz<F>* __restrict__ out,
                                             const u32* __restrict__ tw,
                                             FftParams p,
                                             u32 tid) {
    u32 d = tid + p.gid_off;
    u32 groups = (p.n >> 1u) >> p.log_span;

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
    u32 g = d / (p.span - 1u);
    u32 j = 1u + (d - g * (p.span - 1u));
    u32 lo = (g << (p.log_span + 1u)) + j;
    u32 hi = lo + p.span;

    u32 k[8];
    u32 base = (j << p.tw_shift) * GLV_WORDS;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        k[i] = tw[base + i];
    }
    Xyzz<F> t = fft_ladder<F, C, REG>(in[hi], k, tw[base + 8u]);
    Xyzz<F> u = in[lo];
    out[lo] = pt_add(u, t);
    out[hi] = pt_add(u, pt_neg(t));

    if (d < groups) {
        u32 lo0 = d << (p.log_span + 1u);
        u32 hi0 = lo0 + p.span;
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
// The host folds `s` into this pass's own twiddle table (`stw[j] = s * W^j`, so
// `stw[0] == s` covers the `u` side), which is what saves the separate scaling pass:
// see `fft_mix_scale_impl` in the Metal twin for the ladder count.
//
// The last pass has exactly one group, so `lo` is `gid` and `hi` is `gid + n/2` with no
// group arithmetic. Still out of place and read-only on `in`; each thread is two
// ladders.
template <typename F, u32 C, bool REG>
__device__ __forceinline__ void fft_mix_scale_impl(const Xyzz<F>* __restrict__ in,
                                                   Xyzz<F>* __restrict__ out,
                                                   const u32* __restrict__ stw,
                                                   FftParams p,
                                                   u32 tid) {
    u32 gid = tid + p.gid_off;
    u32 half_n = p.n >> 1u;
    if (gid >= half_n) {
        return;
    }
    u32 s[8];
    u32 k[8];
    u32 base = gid * GLV_WORDS;
#pragma unroll
    for (u32 i = 0; i < 8u; i++) {
        s[i] = stw[i];
        k[i] = stw[base + i];
    }
    Xyzz<F> t = fft_ladder<F, C, REG>(in[gid + half_n], k, stw[base + 8u]);
    Xyzz<F> u = fft_ladder<F, C, REG>(in[gid], s, stw[8]);
    out[gid] = pt_add(u, t);
    out[gid + half_n] = pt_add(u, pt_neg(t));
}

// ---------------------------------------------------------------------------
// The kernels, one pair per group per compiled variant.
//
// A macro rather than a template because a templated function cannot itself be
// `extern "C" __global__`, the same pattern as the MSM entry points; the host selects a
// function by name. g16-cuda/src/fft.rs's `VARIANTS` builds the same names from the
// same list, and a grep test on each side keeps them aligned.
// ---------------------------------------------------------------------------

#define FFT_KERNELS(SUF, FT, PTT, C, REG)                                     \
    extern "C" __global__ void fft_mix_##SUF(const PTT* __restrict__ in,      \
                                             PTT* __restrict__ out,           \
                                             const u32* __restrict__ tw,      \
                                             FftParams p) {                   \
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        fft_mix_impl<FT, C, REG>(in, out, tw, p, tid);                        \
    }                                                                         \
    extern "C" __global__ void fft_mix_scale_##SUF(const PTT* __restrict__ in,\
                                                   PTT* __restrict__ out,     \
                                                   const u32* __restrict__ stw,\
                                                   FftParams p) {             \
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        fft_mix_scale_impl<FT, C, REG>(in, out, stw, p, tid);                 \
    }

// The same pair under a `__launch_bounds__` cap. `MAXT` bounds the block and `MINB`
// blocks must co-reside per SM, so ptxas is forced down to 64K / (MAXT * MINB) registers
// per thread and spills the difference. Whether trading spill for occupancy pays on a
// latency-heavy dependent chain is a measurement, not a judgement call.
#define FFT_KERNELS_LB(SUF, FT, PTT, C, REG, MAXT, MINB)                      \
    extern "C" __global__ void __launch_bounds__(MAXT, MINB)                  \
    fft_mix_##SUF(const PTT* __restrict__ in,                                 \
                  PTT* __restrict__ out,                                      \
                  const u32* __restrict__ tw,                                 \
                  FftParams p) {                                              \
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        fft_mix_impl<FT, C, REG>(in, out, tw, p, tid);                        \
    }                                                                         \
    extern "C" __global__ void __launch_bounds__(MAXT, MINB)                  \
    fft_mix_scale_##SUF(const PTT* __restrict__ in,                           \
                        PTT* __restrict__ out,                                \
                        const u32* __restrict__ stw,                          \
                        FftParams p) {                                        \
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        fft_mix_scale_impl<FT, C, REG>(in, out, stw, p, tid);                 \
    }

FFT_KERNELS(g1_c5, Fq, PtG1, 5u, false)
FFT_KERNELS(g2_c5, Fq2, PtG2, 5u, false)

// The experiment set, one hypothesis per pair (see the banner's point 3 for why they are
// all in this one unit and how the host reaches them):
//
// * `c4`: the c = 5 table is what forces local-memory traffic; halving it to 768 B / 1.5 KB
//   per thread buys more than the arithmetic it costs (32 windows instead of 26, ~12
//   more window additions, 8 fewer table-build entries).
// * `c3r`: local memory is the wrong home for the table altogether; four entries selected
//   by predicated moves keep it in registers and beat both widths, or the ~34 extra
//   additions bury the saving.
// * `c5r128`: the default register allocation (up to 255 a thread, ~2 blocks of 128 per
//   SM) starves the SM of warps; capping at 128 registers doubles residency and the
//   added spill costs less than the latency it hides.
#ifdef G16_FFT_VARIANTS
FFT_KERNELS(g1_c4, Fq, PtG1, 4u, false)
FFT_KERNELS(g2_c4, Fq2, PtG2, 4u, false)
FFT_KERNELS(g1_c3r, Fq, PtG1, 3u, true)
FFT_KERNELS(g2_c3r, Fq2, PtG2, 3u, true)
FFT_KERNELS_LB(g1_c5r128, Fq, PtG1, 5u, false, 128, 4)
FFT_KERNELS_LB(g2_c5r128, Fq2, PtG2, 5u, false, 128, 4)
#endif // G16_FFT_VARIANTS

#endif // G16_FFT_CU
