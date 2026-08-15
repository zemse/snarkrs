//! Branch-free BN254 field arithmetic on raw Montgomery limbs.
//!
//! # Why this exists
//!
//! Profiling (bench/results/profiling/README.md, round 4) put 62.8% of CPU proof time
//! inside `ark-ff`'s `Fp`, and showed the multiply itself is near the machine limit; what
//! is not is everything wrapped around it. `ark-ff` finishes every add, sub and multiply
//! with a *compare-and-branch* against the modulus: a four-limb lexicographic compare
//! whose first branch is taken with probability about one half on Montgomery outputs,
//! which is a guaranteed-unpredictable branch in the innermost loop of the prover.
//! `cargo asm` on the NTT butterfly counted 27 conditional branches per iteration from
//! exactly this. The measured symptom at the point level: a G1 mixed add costs 247 ns
//! against a 148 ns floor set by its own multiplies.
//!
//! The functions here are the same arithmetic with the reduction done branch-free: the
//! conditional subtraction computes both candidates and blends them with a mask, which
//! LLVM lowers to `csel` on aarch64 and `cmov` on x86_64. No secret-independent branch
//! remains in add/sub/mul/sqr (`inverse` and comparisons are not on the hot path).
//!
//! # What lives here
//!
//! `RawFq`, `RawFq2` and `RawFr`: plain limb arrays in Montgomery form, bit-compatible
//! with `ark-ff`'s in-memory representation (arkworks stores Montgomery limbs
//! little-endian, which is exactly what these wrap), so conversion in either direction is
//! a copy, never an arithmetic operation. The intended consumer is a hot loop that
//! converts on entry, works on raw limbs throughout, and converts back on exit; the MSM
//! bucket accumulation in `g16-msm` is the canonical one.
//!
//! Every operation keeps values fully reduced in `[0, p)`, matching `ark-ff` semantics
//! exactly. That makes cross-checking trivial (bit equality against ark on every sample)
//! and keeps this module drop-in safe; lazy `[0, 2p)` forms would buy a little more and
//! are the documented next step if measurement demands them.
//!
//! The multiply is the fused "no final carry" CIOS (gnark's shape), valid because BN254's
//! top word 0x30644e72e131a029 is below 2^63 - 1 on both fields. The same shape was
//! validated for the GPU ports against a bit-exact host simulation; here it is
//! additionally differential-tested against `ark-ff` in `tests/raw_matches_ark.rs`.

use crate::{Fq, Fq2, Fr};

/// BN254 base field modulus, little-endian u64 limbs.
pub const FQ_MOD: [u64; 4] = [
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
/// `-q^{-1} mod 2^64`.
pub const FQ_N0: u64 = 0x87d20782e4866389;

/// BN254 scalar field modulus, little-endian u64 limbs.
pub const FR_MOD: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
/// `-r^{-1} mod 2^64`.
pub const FR_N0: u64 = 0xc2e1f593efffffff;

// ---------------------------------------------------------------------------
// Limb kernels, shared by both fields. All #[inline(always)]: these exist to be
// swallowed whole by the caller's loop body.
// ---------------------------------------------------------------------------

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    // 128-bit subtract keeps the borrow as data rather than as a flag the compiler is
    // tempted to branch on.
    let t = (a as u128).wrapping_sub(b as u128).wrapping_sub(borrow as u128);
    (t as u64, (t >> 127) as u64)
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `if t >= m { t - m } else { t }`, branch-free: both candidates are computed and
/// blended by a mask. LLVM emits csel/cmov for the blend.
#[inline(always)]
fn cond_sub(t: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (r0, b) = sbb(t[0], m[0], 0);
    let (r1, b) = sbb(t[1], m[1], b);
    let (r2, b) = sbb(t[2], m[2], b);
    let (r3, b) = sbb(t[3], m[3], b);
    // b == 1 means t < m: keep t. The mask is all-ones in that case.
    let keep = b.wrapping_neg();
    [
        (t[0] & keep) | (r0 & !keep),
        (t[1] & keep) | (r1 & !keep),
        (t[2] & keep) | (r2 & !keep),
        (t[3] & keep) | (r3 & !keep),
    ]
}

/// `(a + b) mod m` for fully reduced inputs. The carry out of limb 3 is provably zero
/// for both BN254 fields (moduli are 254-bit), so the sum fits four limbs and one
/// conditional subtraction finishes it.
#[inline(always)]
fn add_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (s0, c) = adc(a[0], b[0], 0);
    let (s1, c) = adc(a[1], b[1], c);
    let (s2, c) = adc(a[2], b[2], c);
    let (s3, _) = adc(a[3], b[3], c);
    cond_sub([s0, s1, s2, s3], m)
}

/// `(a - b) mod m`, branch-free: the modulus is masked in when the subtraction borrows.
#[inline(always)]
fn sub_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (d0, bw) = sbb(a[0], b[0], 0);
    let (d1, bw) = sbb(a[1], b[1], bw);
    let (d2, bw) = sbb(a[2], b[2], bw);
    let (d3, bw) = sbb(a[3], b[3], bw);
    let mask = bw.wrapping_neg();
    let (r0, c) = adc(d0, m[0] & mask, 0);
    let (r1, c) = adc(d1, m[1] & mask, c);
    let (r2, c) = adc(d2, m[2] & mask, c);
    let (r3, _) = adc(d3, m[3] & mask, c);
    [r0, r1, r2, r3]
}

/// Montgomery product, fused no-final-carry CIOS. Montgomery in, Montgomery out, fully
/// reduced. Valid for moduli whose top word is at most (2^64 - 1)/2 - 1, which both
/// BN254 fields satisfy; see the module doc.
#[inline(always)]
fn mont_mul(a: [u64; 4], b: [u64; 4], m: &[u64; 4], n0: u64) -> [u64; 4] {
    let mut t = [0u64; 4];
    for i in 0..4 {
        let bi = b[i] as u128;
        let mut aa = t[0] as u128 + a[0] as u128 * bi;
        let t0 = aa as u64;
        aa >>= 64;
        let mm = t0.wrapping_mul(n0);
        let mut c = (t0 as u128 + (mm as u128) * (m[0] as u128)) >> 64;
        for j in 1..4 {
            aa = t[j] as u128 + a[j] as u128 * bi + aa;
            let tj = aa as u64;
            aa >>= 64;
            c = tj as u128 + (mm as u128) * (m[j] as u128) + c;
            t[j - 1] = c as u64;
            c >>= 64;
        }
        // Exact by the top-word bound: no carry out of limb 3.
        t[3] = (c as u64).wrapping_add(aa as u64);
    }
    cond_sub(t, m)
}

// ---------------------------------------------------------------------------
// The operation surface, as a trait, so curve arithmetic can be written once over a
// generic base field and instantiated for G1 (RawFq) and G2 (RawFq2), exactly as the
// Metal and CUDA kernels do with their f_add/f_mul overload sets.
// ---------------------------------------------------------------------------

pub trait RawField: Copy + PartialEq + Send + Sync {
    const ZERO: Self;
    /// The Montgomery representative of 1.
    const ONE: Self;
    fn is_zero(&self) -> bool;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn sqr(self) -> Self;
    fn double(self) -> Self;
}

// ---------------------------------------------------------------------------
// RawFq
// ---------------------------------------------------------------------------

/// BN254 base field element as raw Montgomery limbs, always fully reduced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFq(pub [u64; 4]);

impl RawFq {
    pub const ZERO: RawFq = RawFq([0; 4]);
    /// R mod q, the Montgomery representative of 1. Pinned against ark in tests.
    pub const ONE: RawFq = RawFq([
        0xd35d438dc58f0d9d,
        0x0a78eb28f5c70b3d,
        0x666ea36f7879462c,
        0x0e0a77c19a07df2f,
    ]);

    #[inline(always)]
    pub fn from_fq(x: &Fq) -> Self {
        // arkworks stores exactly these limbs (Montgomery form, little-endian).
        RawFq(x.0 .0)
    }

    #[inline(always)]
    pub fn to_fq(self) -> Fq {
        ark_ff::Fp(ark_ff::BigInt(self.0), core::marker::PhantomData)
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFq(add_mod(self.0, o.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFq(sub_mod(self.0, o.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        RawFq(sub_mod([0; 4], self.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        RawFq(mont_mul(self.0, o.0, &FQ_MOD, FQ_N0))
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        self.mul(self)
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        self.add(self)
    }
}

impl RawField for RawFq {
    const ZERO: Self = RawFq::ZERO;
    const ONE: Self = RawFq::ONE;
    #[inline(always)]
    fn is_zero(&self) -> bool {
        RawFq::is_zero(self)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        RawFq::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        RawFq::sub(self, o)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        RawFq::neg(self)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        RawFq::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        RawFq::sqr(self)
    }
    #[inline(always)]
    fn double(self) -> Self {
        RawFq::double(self)
    }
}

// ---------------------------------------------------------------------------
// RawFq2 = RawFq[u] / (u^2 + 1)
// ---------------------------------------------------------------------------

/// BN254 `Fq2` element as a pair of raw `Fq`. Multiplication is Karatsuba (three `Fq`
/// products), squaring the complex trick (two), the same shapes as `ark-ff` and the GPU
/// kernels use, so operation counts are comparable across all three.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFq2 {
    pub c0: RawFq,
    pub c1: RawFq,
}

impl RawFq2 {
    pub const ZERO: RawFq2 = RawFq2 {
        c0: RawFq::ZERO,
        c1: RawFq::ZERO,
    };
    pub const ONE: RawFq2 = RawFq2 {
        c0: RawFq::ONE,
        c1: RawFq::ZERO,
    };

    #[inline(always)]
    pub fn from_fq2(x: &Fq2) -> Self {
        RawFq2 {
            c0: RawFq::from_fq(&x.c0),
            c1: RawFq::from_fq(&x.c1),
        }
    }

    #[inline(always)]
    pub fn to_fq2(self) -> Fq2 {
        Fq2::new(self.c0.to_fq(), self.c1.to_fq())
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFq2 {
            c0: self.c0.add(o.c0),
            c1: self.c1.add(o.c1),
        }
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFq2 {
            c0: self.c0.sub(o.c0),
            c1: self.c1.sub(o.c1),
        }
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        RawFq2 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// `(a0 + a1 u)(b0 + b1 u)` with `u^2 = -1`: Karatsuba, three `Fq` products.
    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        let v0 = self.c0.mul(o.c0);
        let v1 = self.c1.mul(o.c1);
        let s = self.c0.add(self.c1);
        let t = o.c0.add(o.c1);
        RawFq2 {
            c0: v0.sub(v1),
            c1: s.mul(t).sub(v0).sub(v1),
        }
    }

    /// `(a0 + a1 u)^2 = (a0 + a1)(a0 - a1) + 2 a0 a1 u`.
    #[inline(always)]
    pub fn sqr(self) -> Self {
        let t0 = self.c0.add(self.c1);
        let t1 = self.c0.sub(self.c1);
        let t2 = self.c0.mul(self.c1);
        RawFq2 {
            c0: t0.mul(t1),
            c1: t2.add(t2),
        }
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        self.add(self)
    }
}

impl RawField for RawFq2 {
    const ZERO: Self = RawFq2::ZERO;
    const ONE: Self = RawFq2::ONE;
    #[inline(always)]
    fn is_zero(&self) -> bool {
        RawFq2::is_zero(self)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        RawFq2::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        RawFq2::sub(self, o)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        RawFq2::neg(self)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        RawFq2::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        RawFq2::sqr(self)
    }
    #[inline(always)]
    fn double(self) -> Self {
        RawFq2::double(self)
    }
}

// ---------------------------------------------------------------------------
// RawFr
// ---------------------------------------------------------------------------

/// BN254 scalar field element as raw Montgomery limbs. Same design as [`RawFq`]; exists
/// for `Fr`-heavy loops (NTT butterflies, coset shifts) that want the branch-free
/// reduction there too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFr(pub [u64; 4]);

impl RawFr {
    pub const ZERO: RawFr = RawFr([0; 4]);

    #[inline(always)]
    pub fn from_fr(x: &Fr) -> Self {
        RawFr(x.0 .0)
    }

    #[inline(always)]
    pub fn to_fr(self) -> Fr {
        ark_ff::Fp(ark_ff::BigInt(self.0), core::marker::PhantomData)
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFr(add_mod(self.0, o.0, &FR_MOD))
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFr(sub_mod(self.0, o.0, &FR_MOD))
    }

    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        RawFr(mont_mul(self.0, o.0, &FR_MOD, FR_N0))
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        self.mul(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m[0] * n0 == -1 mod 2^64` is the defining property of the Montgomery constant;
    /// pinned here so the hex literals above cannot rot silently.
    #[test]
    fn montgomery_constants_are_inverses() {
        assert_eq!(FQ_MOD[0].wrapping_mul(FQ_N0), u64::MAX);
        assert_eq!(FR_MOD[0].wrapping_mul(FR_N0), u64::MAX);
    }

    #[test]
    fn ones_match_ark() {
        use ark_ff::One;
        assert_eq!(RawFq::ONE.to_fq(), Fq::one());
        assert_eq!(RawFq2::ONE.to_fq2(), Fq2::one());
        assert_eq!(RawFq::ONE, <RawFq as RawField>::ONE);
    }

    #[test]
    fn moduli_match_ark() {
        use ark_ff::PrimeField;
        assert_eq!(Fq::MODULUS.0, FQ_MOD);
        assert_eq!(Fr::MODULUS.0, FR_MOD);
    }
}
