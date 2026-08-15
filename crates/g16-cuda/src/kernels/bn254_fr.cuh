// BN254 scalar field arithmetic for CUDA. Compiled at runtime through NVRTC, the same
// shape as the Metal backend's MTLDevice::newLibraryWithSource: the kernel source ships
// as a string in the binary and is compiled for the GPU actually present, so one binary
// runs on Turing, Ampere and Ada without an offline toolchain or a fat cubin.
//
// ============================================================================
// THIS FILE AND crates/g16-gpu-layout/src/lib.rs ARE ONE DEFINITION IN TWO LANGUAGES.
// `struct Fr` below is `PackedFr` there. FR_N / FR_N0 below are FR_MODULUS / FR_N0
// there. A Rust test greps this source for those exact lines, so editing one side alone
// fails the build rather than producing wrong proofs. Change them together.
//
// The Metal backend carries a third copy of the same definition in
// crates/g16-metal/src/shaders/bn254_fr.metal, guarded by its own grep test. The point
// of the shared layout crate is that both GPUs are fed bit-identical bytes, which is the
// only thing that makes a Metal-versus-CUDA benchmark mean anything.
// ============================================================================
//
// LIMB WIDTH: 8 limbs of 32 bits with 64-bit accumulators, the same choice as the Metal
// port, and for the same reason rather than by copying.
//
// The MSL file claims this layout "is the opposite of what a CUDA port would do". That
// is wrong and is corrected here: NVIDIA GPUs have no native 64x64 integer multiplier
// either. A 64-bit multiply lowers to a sequence of mul.lo.u32 / mul.hi.u32 plus carry
// handling, exactly like on Apple silicon. What NVIDIA does have is native 32x32->64
// (mul.wide.u32), and nvcc reliably contracts
//
//     (unsigned long long)a.v[j] * (unsigned long long)bi
//
// into a single mul.wide.u32 when it can prove both factors are 32 bits, which it can
// here because both are `unsigned int` loads. So the accumulator stays 64-bit and every
// product stays native, which is the same bargain the Metal version strikes.
//
// The one thing NVIDIA offers that Metal does not is an addressable carry flag:
// mad.lo.cc.u32 / madc.hi.cc.u32 chains written as inline PTX remove the explicit
// shift-and-mask carry propagation entirely and are what sppark and the other fast CUDA
// MSM libraries use. That path now exists in this file, behind `G16_FF_PTX` (defined by
// the host when `G16_CUDA_FF_PTX=1`, see kernels.rs), OFF by default: it has been
// checked against a bit-exact host simulation of the exact chain shape, but it has NOT
// run on an NVIDIA card yet, and it stays opt-in until `fr_probe` has validated it and
// `bench_chain` has priced it on real hardware. The portable version below remains the
// default and the reference.
//
// R = 2^(32*8) = 2^256 is exactly arkworks' Montgomery radix, so a host-side Fr is
// handed to a kernel with no conversion in either direction.
//
// ALGORITHM: CIOS (Coarsely Integrated Operand Scanning), Acar's thesis, one 32-bit limb
// per outer iteration with a `unsigned int t[10]` accumulator. A single conditional
// subtraction at the end is enough: r is 254 bits and R is 2^256, so the CIOS
// intermediate stays below 2r and one subtraction lands in [0, r).
//
// BRANCHES: every conditional reduction is a ternary, never an `if` with a body. On a
// 32-lane warp a data-dependent `if (x >= mod) x -= mod` serialises the divergent lanes;
// ZKProphet (arXiv 2509.22684) attributes 70.5% of FF_add/FF_sub latency to exactly that
// branch. A ternary over a scalar lowers to `selp.b32`, which is predicated and costs the
// subtraction unconditionally and nothing else. Do not "simplify" these into ifs.

#ifndef G16_BN254_FR_CUH
#define G16_BN254_FR_CUH

// NVRTC compiles without the CUDA headers, so nothing here may rely on them. These two
// aliases keep the body character-for-character comparable with the MSL twin.
typedef unsigned int u32;
typedef unsigned long long u64;

#define FR_LIMBS 8

// Mirrors g16_gpu_layout::PackedFr. Little-endian 32-bit limbs, Montgomery form with
// R = 2^256, always fully reduced into [0, r).
//
// A plain POD struct with no constructor and no methods, deliberately. The Metal side
// needs this because a class with a constructor cannot be a `threadgroup` variable; CUDA
// has no such restriction, but keeping the two identical means a fix to one transfers to
// the other by inspection instead of by translation.
struct Fr {
    u32 v[8];
};

static_assert(sizeof(Fr) == 32, "Fr must be 32 bytes to match layout::PackedFr");

// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//   = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
__constant__ u32 FR_N[8] = { 0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u };

// -r^{-1} mod 2^32. Verified against ark-ff in the layout crate, not pasted from a blog.
__constant__ u32 FR_N0 = 0xefffffffu;

// R mod r, the Montgomery representative of 1.
__constant__ u32 FR_R[8] = { 0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u, 0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u };

// R^2 mod r, the multiplier that lifts a standard-form residue into Montgomery form.
__constant__ u32 FR_R2[8] = { 0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u, 0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u };

__device__ __forceinline__ Fr fr_zero() {
    Fr z;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        z.v[i] = 0u;
    }
    return z;
}

__device__ __forceinline__ Fr fr_one() {
    Fr o;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        o.v[i] = FR_R[i];
    }
    return o;
}

__device__ __forceinline__ bool fr_is_zero(Fr a) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        acc |= a.v[i];
    }
    return acc == 0u;
}

__device__ __forceinline__ bool fr_eq(Fr a, Fr b) {
    u32 acc = 0u;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        acc |= (a.v[i] ^ b.v[i]);
    }
    return acc == 0u;
}

// Returns `a - r` when the 288-bit value (hi : a) is at least r, else `a`. Branch-free.
//
// The borrow test uses bit 63 of a wrapped 64-bit difference: each limb difference lies
// in [-(2^32), 2^32), so a negative result wraps to something at least 2^64 - 2^32 - 1
// and a non-negative one stays below 2^32. There is no third case.
__device__ __forceinline__ Fr fr_cond_sub_n(Fr a, u32 hi) {
    u32 red[8];
    u64 borrow = 0;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        u64 d = (u64)a.v[i] - (u64)FR_N[i] - borrow;
        red[i] = (u32)d;
        borrow = (d >> 63) & (u64)1;
    }
    // Take the reduced value when the subtraction did not go negative (a >= r), or when
    // there is a nonzero 257th-and-up word, which puts the value above 2^256 > r.
    bool take = (hi != 0u) || (borrow == 0);
    Fr out;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        out.v[i] = take ? red[i] : a.v[i];
    }
    return out;
}

// (a + b) mod r. Inputs must already be reduced.
//
// The carry out of limb 7 is computed and fed to the conditional subtraction even though
// for BN254 it is provably always zero (a, b < r < 2^254 so a + b < 2^255). It costs one
// ternary input and it means this function stays correct if someone reuses the file for
// a 256-bit prime, which is exactly the silent-wrong-answer trap in one of the reference
// implementations.
__device__ __forceinline__ Fr fr_add(Fr a, Fr b) {
    Fr s;
    u64 c = 0;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        u64 t = (u64)a.v[i] + (u64)b.v[i] + c;
        s.v[i] = (u32)t;
        c = t >> 32;
    }
    return fr_cond_sub_n(s, (u32)c);
}

// (a - b) mod r. Inputs must already be reduced. Branch-free: the modulus is masked in
// rather than added under an `if`.
__device__ __forceinline__ Fr fr_sub(Fr a, Fr b) {
    u32 d[8];
    u64 borrow = 0;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        u64 t = (u64)a.v[i] - (u64)b.v[i] - borrow;
        d[i] = (u32)t;
        borrow = (t >> 63) & (u64)1;
    }
    u32 mask = (u32)(0u - (u32)borrow);
    Fr out;
    u64 c = 0;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        u64 t = (u64)d[i] + (u64)(FR_N[i] & mask) + c;
        out.v[i] = (u32)t;
        c = t >> 32;
    }
    return out;
}

// -a mod r, with -0 == 0. Falls straight out of fr_sub: 0 - 0 borrows nothing so the
// modulus is not added back, and 0 - a for a != 0 borrows and yields r - a.
__device__ __forceinline__ Fr fr_neg(Fr a) {
    return fr_sub(fr_zero(), a);
}

// ---------------------------------------------------------------------------
// Optional PTX carry-chain CIOS round, shared by fr_mul here and fq_mul in msm.cu.
//
// One outer CIOS round as a single asm block: multiply pass (lo chain, then hi chain,
// both propagating through t8 into t9), m = t0 * n0', reduction pass (same two chains).
// The caller shifts the accumulator down one word in C between rounds; ptxas erases
// register-to-register moves, so the shift costs nothing in SASS.
//
// This is EXACTLY the arithmetic of the portable 10-word CIOS below, checked against a
// bit-exact host simulation of this chain shape (including that t9 never overflows, and
// that the reduction annihilates t0) over the edge vectors and tens of thousands of
// random and biased-high operands on both BN254 fields. It has NOT yet executed on an
// NVIDIA GPU. Validation on real hardware is `fr_probe` / `fq2_probe` with
// `G16_CUDA_FF_PTX=1`; the measurement that would justify making it the default is
// `bench_chain` in the same configuration.
//
// Operand map: %0-%9 = t0..t9 (read-write), %10-%17 = a limbs, %18 = b's current limb,
// %19 = -N^{-1} mod 2^32, %20-%27 = modulus limbs.
// ---------------------------------------------------------------------------
#ifdef G16_FF_PTX
#define G16_MONT_ROUND_PTX(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, A, BI, N0, Q)       \
    asm("{\n\t"                                                                        \
        ".reg .u32 m;\n\t"                                                             \
        "mad.lo.cc.u32  %0, %10, %18, %0;\n\t"                                         \
        "madc.lo.cc.u32 %1, %11, %18, %1;\n\t"                                         \
        "madc.lo.cc.u32 %2, %12, %18, %2;\n\t"                                         \
        "madc.lo.cc.u32 %3, %13, %18, %3;\n\t"                                         \
        "madc.lo.cc.u32 %4, %14, %18, %4;\n\t"                                         \
        "madc.lo.cc.u32 %5, %15, %18, %5;\n\t"                                         \
        "madc.lo.cc.u32 %6, %16, %18, %6;\n\t"                                         \
        "madc.lo.cc.u32 %7, %17, %18, %7;\n\t"                                         \
        "addc.cc.u32    %8, %8, 0;\n\t"                                                \
        "addc.u32       %9, %9, 0;\n\t"                                                \
        "mad.hi.cc.u32  %1, %10, %18, %1;\n\t"                                         \
        "madc.hi.cc.u32 %2, %11, %18, %2;\n\t"                                         \
        "madc.hi.cc.u32 %3, %12, %18, %3;\n\t"                                         \
        "madc.hi.cc.u32 %4, %13, %18, %4;\n\t"                                         \
        "madc.hi.cc.u32 %5, %14, %18, %5;\n\t"                                         \
        "madc.hi.cc.u32 %6, %15, %18, %6;\n\t"                                         \
        "madc.hi.cc.u32 %7, %16, %18, %7;\n\t"                                         \
        "madc.hi.cc.u32 %8, %17, %18, %8;\n\t"                                         \
        "addc.u32       %9, %9, 0;\n\t"                                                \
        "mul.lo.u32     m, %0, %19;\n\t"                                               \
        "mad.lo.cc.u32  %0, m, %20, %0;\n\t"                                           \
        "madc.lo.cc.u32 %1, m, %21, %1;\n\t"                                           \
        "madc.lo.cc.u32 %2, m, %22, %2;\n\t"                                           \
        "madc.lo.cc.u32 %3, m, %23, %3;\n\t"                                           \
        "madc.lo.cc.u32 %4, m, %24, %4;\n\t"                                           \
        "madc.lo.cc.u32 %5, m, %25, %5;\n\t"                                           \
        "madc.lo.cc.u32 %6, m, %26, %6;\n\t"                                           \
        "madc.lo.cc.u32 %7, m, %27, %7;\n\t"                                           \
        "addc.cc.u32    %8, %8, 0;\n\t"                                                \
        "addc.u32       %9, %9, 0;\n\t"                                                \
        "mad.hi.cc.u32  %1, m, %20, %1;\n\t"                                           \
        "madc.hi.cc.u32 %2, m, %21, %2;\n\t"                                           \
        "madc.hi.cc.u32 %3, m, %22, %3;\n\t"                                           \
        "madc.hi.cc.u32 %4, m, %23, %4;\n\t"                                           \
        "madc.hi.cc.u32 %5, m, %24, %5;\n\t"                                           \
        "madc.hi.cc.u32 %6, m, %25, %6;\n\t"                                           \
        "madc.hi.cc.u32 %7, m, %26, %7;\n\t"                                           \
        "madc.hi.cc.u32 %8, m, %27, %8;\n\t"                                           \
        "addc.u32       %9, %9, 0;\n\t"                                                \
        "}\n\t"                                                                        \
        : "+r"(t0), "+r"(t1), "+r"(t2), "+r"(t3), "+r"(t4), "+r"(t5), "+r"(t6),        \
          "+r"(t7), "+r"(t8), "+r"(t9)                                                 \
        : "r"((A)[0]), "r"((A)[1]), "r"((A)[2]), "r"((A)[3]), "r"((A)[4]),             \
          "r"((A)[5]), "r"((A)[6]), "r"((A)[7]), "r"(BI), "r"(N0), "r"((Q)[0]),        \
          "r"((Q)[1]), "r"((Q)[2]), "r"((Q)[3]), "r"((Q)[4]), "r"((Q)[5]),             \
          "r"((Q)[6]), "r"((Q)[7]))

// Full 8-round Montgomery product over an arbitrary 8-limb modulus with the chain
// above. `q` and `n0` are the modulus limbs and -q^{-1} mod 2^32; the result is the
// 9-word value (carry:out), reduced by the caller's conditional subtraction.
__device__ __forceinline__ u32 g16_mont_mul_ptx(const u32* a, const u32* b, const u32* q,
                                                u32 n0, u32* out) {
    u32 t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0;
#pragma unroll
    for (u32 i = 0; i < 8; i++) {
        G16_MONT_ROUND_PTX(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, a, b[i], n0, q);
        // Shift the zero low word off: register renames, free after ptxas.
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = t8;
        t8 = t9; t9 = 0;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
    out[4] = t4; out[5] = t5; out[6] = t6; out[7] = t7;
    return t8;
}
#endif // G16_FF_PTX

// Montgomery product: returns a * b * R^{-1} mod r, so Montgomery in, Montgomery out.
//
// CIOS, interleaving one multiply pass and one reduction pass per outer limb of b. The
// accumulator is 10 words: 8 for the residue, 1 for the carry out of the multiply pass,
// and 1 more because that carry pass can itself carry.
#ifdef G16_FF_PTX
__device__ __forceinline__ Fr fr_mul(Fr a, Fr b) {
    Fr out;
    u32 hi = g16_mont_mul_ptx(a.v, b.v, FR_N, FR_N0, out.v);
    return fr_cond_sub_n(out, hi);
}
__device__ __forceinline__ Fr fr_mul_portable(Fr a, Fr b) {
#else
__device__ __forceinline__ Fr fr_mul(Fr a, Fr b) {
#endif
    u32 t[FR_LIMBS + 2];
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS + 2; i++) {
        t[i] = 0u;
    }

#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        u32 bi = b.v[i];

        // t += a * b[i]
        u64 c = 0;
#pragma unroll
        for (u32 j = 0; j < FR_LIMBS; j++) {
            u64 r = (u64)t[j] + (u64)a.v[j] * (u64)bi + c;
            t[j] = (u32)r;
            c = r >> 32;
        }
        u64 r = (u64)t[FR_LIMBS] + c;
        t[FR_LIMBS] = (u32)r;
        t[FR_LIMBS + 1] = (u32)(r >> 32);

        // m = t[0] * (-r^{-1}) mod 2^32, chosen so t + m*r has a zero low limb; then
        // shift that zero limb off, which is the division by 2^32 that makes this
        // Montgomery rather than schoolbook.
        u32 m = t[0] * FR_N0;
        u64 d = (u64)t[0] + (u64)m * (u64)FR_N[0];
        c = d >> 32;
#pragma unroll
        for (u32 j = 1; j < FR_LIMBS; j++) {
            u64 r2 = (u64)t[j] + (u64)m * (u64)FR_N[j] + c;
            t[j - 1] = (u32)r2;
            c = r2 >> 32;
        }
        u64 r3 = (u64)t[FR_LIMBS] + c;
        t[FR_LIMBS - 1] = (u32)r3;
        t[FR_LIMBS] = t[FR_LIMBS + 1] + (u32)(r3 >> 32);
    }

    Fr out;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        out.v[i] = t[i];
    }
    return fr_cond_sub_n(out, t[FR_LIMBS]);
}

// a^2. Kept as a named function so a dedicated squaring (which can skip roughly half the
// limb products) can replace the body later without touching any caller.
__device__ __forceinline__ Fr fr_sqr(Fr a) {
    return fr_mul(a, a);
}

// Standard form -> Montgomery form: a -> a*R. Only needed for values that did not come
// from the host through layout::PackedFr, which is already Montgomery.
__device__ __forceinline__ Fr fr_to_mont(Fr a) {
    Fr r2;
#pragma unroll
    for (u32 i = 0; i < FR_LIMBS; i++) {
        r2.v[i] = FR_R2[i];
    }
    return fr_mul(a, r2);
}

// Montgomery form -> standard form: a*R -> a. A Montgomery multiply by the integer 1 is
// exactly a Montgomery reduction.
__device__ __forceinline__ Fr fr_from_mont(Fr a) {
    Fr one;
    one.v[0] = 1u;
#pragma unroll
    for (u32 i = 1; i < FR_LIMBS; i++) {
        one.v[i] = 0u;
    }
    return fr_mul(a, one);
}

#endif // G16_BN254_FR_CUH
