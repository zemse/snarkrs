// BN254 scalar field arithmetic for Metal. Compiled at runtime through
// MTLDevice::newLibraryWithSource; there is no offline toolchain on the target machine
// and none is needed.
//
// ============================================================================
// THIS FILE AND crates/g16-metal/src/layout.rs ARE ONE DEFINITION IN TWO LANGUAGES.
// `struct Fr` below is `PackedFr` there. FR_N / FR_N0 below are FR_MODULUS / FR_N0
// there. A Rust test greps this source for those exact lines, so editing one side alone
// fails the build rather than producing wrong proofs. Change them together.
// ============================================================================
//
// LIMB WIDTH: 8 limbs of 32 bits, with 64-bit accumulators. Not 4 x 64.
//
// This is the opposite of what a CUDA port would do, and it is measured, not assumed.
// Apple GPUs have a native 32x32->64 multiply but emulate a full 64x64 one. Keeping every
// *stored* limb a `uint` and widening to `ulong` only at the multiply-accumulate site,
// as in
//
//     ulong r = (ulong)t[j] + (ulong)a.v[j] * (ulong)b.v[i] + c;
//
// means both factors are provably 32-bit, so the compiler emits the native path and the
// `ulong` only ever carries the accumulator. On this M2 Max that form ran at 4.51 G
// mul/s against 3.49 G mul/s for the same CIOS written with `mulhi` and explicit 32-bit
// carry chains, a 29% win for the wider accumulator. Both Apple-targeted references
// (zkmopro/gpu-acceleration, zkonduit/metal-msm) independently landed on 8 x u32; the one
// project that uses 4 x u64 (lambdaworks) hand-emulates a u128 class underneath and pays
// the emulated multiply on every limb product.
//
// The second reason is free interoperability: 8 limbs of 32 bits makes R = 2^256, which
// is exactly arkworks' Montgomery radix, so a host-side Fr can be handed to a kernel with
// no conversion in either direction.
//
// ALGORITHM: CIOS (Coarsely Integrated Operand Scanning), Acar's thesis, one 32-bit limb
// per outer iteration with a `uint t[10]` accumulator. A single conditional subtraction
// at the end is enough: r is 254 bits and R is 2^256, so the CIOS intermediate stays
// below 2r and one subtraction lands in [0, r).
//
// BRANCHES: every conditional reduction is a `select`, never an `if`. On a 32-wide SIMD
// group a data-dependent `if (x >= mod) x -= mod` serialises the divergent lanes, and
// profiling of CUDA ZK kernels attributes about 70% of add/sub latency to exactly that
// branch. `select` costs the subtraction unconditionally and nothing else.

#ifndef G16_BN254_FR_METAL
#define G16_BN254_FR_METAL

#include <metal_stdlib>
using namespace metal;

#define FR_LIMBS 8

// Mirrors g16_metal::layout::PackedFr. Little-endian 32-bit limbs, Montgomery form with
// R = 2^256, always fully reduced into [0, r).
//
// A plain POD struct with no constructor and no methods, deliberately. A class with a
// constructor cannot be declared as a `threadgroup` variable in MSL, which is a wall the
// reference implementations hit and worked around with address-space-qualified
// reinterpret casts. Free functions over a POD never have that problem.
struct Fr {
    uint v[8];
};

static_assert(sizeof(Fr) == 32, "Fr must be 32 bytes to match layout::PackedFr");

// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//   = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
constant uint FR_N[8] = { 0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u };

// -r^{-1} mod 2^32. Verified against ark-ff in layout.rs, not pasted from a blog post.
constant uint FR_N0 = 0xefffffffu;

// R mod r, the Montgomery representative of 1.
constant uint FR_R[8] = { 0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u, 0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u };

// R^2 mod r, the multiplier that lifts a standard-form residue into Montgomery form.
constant uint FR_R2[8] = { 0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u, 0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u };

inline Fr fr_zero() {
    Fr z;
    for (uint i = 0; i < FR_LIMBS; i++) {
        z.v[i] = 0u;
    }
    return z;
}

inline Fr fr_one() {
    Fr o;
    for (uint i = 0; i < FR_LIMBS; i++) {
        o.v[i] = FR_R[i];
    }
    return o;
}

inline bool fr_is_zero(Fr a) {
    uint acc = 0u;
    for (uint i = 0; i < FR_LIMBS; i++) {
        acc |= a.v[i];
    }
    return acc == 0u;
}

inline bool fr_eq(Fr a, Fr b) {
    uint acc = 0u;
    for (uint i = 0; i < FR_LIMBS; i++) {
        acc |= (a.v[i] ^ b.v[i]);
    }
    return acc == 0u;
}

// Returns `a - r` when the 288-bit value (hi : a) is at least r, else `a`. Branch-free.
//
// The borrow test uses bit 63 of a wrapped `ulong` difference: each limb difference lies
// in [-(2^32), 2^32), so a negative result wraps to something at least 2^64 - 2^32 - 1
// and a non-negative one stays below 2^32. There is no third case.
inline Fr fr_cond_sub_n(Fr a, uint hi) {
    uint red[8];
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)FR_N[i] - borrow;
        red[i] = (uint)d;
        borrow = (d >> 63) & (ulong)1;
    }
    // Take the reduced value when the subtraction did not go negative (a >= r), or when
    // there is a nonzero 257th-and-up word, which puts the value above 2^256 > r.
    bool take = (hi != 0u) || (borrow == 0);
    Fr out;
    for (uint i = 0; i < FR_LIMBS; i++) {
        out.v[i] = select(a.v[i], red[i], take);
    }
    return out;
}

// (a + b) mod r. Inputs must already be reduced.
//
// The carry out of limb 7 is computed and fed to the conditional subtraction even though
// for BN254 it is provably always zero (a, b < r < 2^254 so a + b < 2^255). It costs one
// `select` input and it means this function stays correct if someone reuses the file for
// a 256-bit prime, which is exactly the silent-wrong-answer trap in one of the reference
// implementations.
inline Fr fr_add(Fr a, Fr b) {
    Fr s;
    ulong c = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong t = (ulong)a.v[i] + (ulong)b.v[i] + c;
        s.v[i] = (uint)t;
        c = t >> 32;
    }
    return fr_cond_sub_n(s, (uint)c);
}

// (a - b) mod r. Inputs must already be reduced. Branch-free: the modulus is masked in
// rather than added under an `if`.
inline Fr fr_sub(Fr a, Fr b) {
    uint d[8];
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong t = (ulong)a.v[i] - (ulong)b.v[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 63) & (ulong)1;
    }
    uint mask = (uint)(0u - (uint)borrow);
    Fr out;
    ulong c = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong t = (ulong)d[i] + (ulong)(FR_N[i] & mask) + c;
        out.v[i] = (uint)t;
        c = t >> 32;
    }
    return out;
}

// -a mod r, with -0 == 0. Falls straight out of fr_sub: 0 - 0 borrows nothing so the
// modulus is not added back, and 0 - a for a != 0 borrows and yields r - a.
inline Fr fr_neg(Fr a) {
    return fr_sub(fr_zero(), a);
}

// Montgomery product: returns a * b * R^{-1} mod r, so Montgomery in, Montgomery out.
//
// CIOS, interleaving one multiply pass and one reduction pass per outer limb of b. The
// accumulator is 10 words: 8 for the residue, 1 for the carry out of the multiply pass,
// and 1 more because that carry pass can itself carry.
inline Fr fr_mul(Fr a, Fr b) {
    uint t[FR_LIMBS + 2];
    for (uint i = 0; i < FR_LIMBS + 2; i++) {
        t[i] = 0u;
    }

    for (uint i = 0; i < FR_LIMBS; i++) {
        uint bi = b.v[i];

        // t += a * b[i]
        ulong c = 0;
        for (uint j = 0; j < FR_LIMBS; j++) {
            ulong r = (ulong)t[j] + (ulong)a.v[j] * (ulong)bi + c;
            t[j] = (uint)r;
            c = r >> 32;
        }
        ulong r = (ulong)t[FR_LIMBS] + c;
        t[FR_LIMBS] = (uint)r;
        t[FR_LIMBS + 1] = (uint)(r >> 32);

        // m = t[0] * (-r^{-1}) mod 2^32, chosen so t + m*r has a zero low limb; then
        // shift that zero limb off, which is the division by 2^32 that makes this
        // Montgomery rather than schoolbook.
        uint m = t[0] * FR_N0;
        ulong d = (ulong)t[0] + (ulong)m * (ulong)FR_N[0];
        c = d >> 32;
        for (uint j = 1; j < FR_LIMBS; j++) {
            ulong r2 = (ulong)t[j] + (ulong)m * (ulong)FR_N[j] + c;
            t[j - 1] = (uint)r2;
            c = r2 >> 32;
        }
        ulong r3 = (ulong)t[FR_LIMBS] + c;
        t[FR_LIMBS - 1] = (uint)r3;
        t[FR_LIMBS] = t[FR_LIMBS + 1] + (uint)(r3 >> 32);
    }

    Fr out;
    for (uint i = 0; i < FR_LIMBS; i++) {
        out.v[i] = t[i];
    }
    return fr_cond_sub_n(out, t[FR_LIMBS]);
}

// a^2. Kept as a named function so a dedicated squaring (which can skip roughly half the
// limb products) can replace the body later without touching any caller.
inline Fr fr_sqr(Fr a) {
    return fr_mul(a, a);
}

// Standard form -> Montgomery form: a -> a*R. Only needed for values that did not come
// from the host through layout::PackedFr, which is already Montgomery.
inline Fr fr_to_mont(Fr a) {
    Fr r2;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r2.v[i] = FR_R2[i];
    }
    return fr_mul(a, r2);
}

// Montgomery form -> standard form: a*R -> a. A Montgomery multiply by the integer 1 is
// exactly a Montgomery reduction.
inline Fr fr_from_mont(Fr a) {
    Fr one;
    one.v[0] = 1u;
    for (uint i = 1; i < FR_LIMBS; i++) {
        one.v[i] = 0u;
    }
    return fr_mul(a, one);
}

#endif // G16_BN254_FR_METAL
