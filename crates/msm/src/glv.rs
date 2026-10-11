//! Variable-base scalar multiplication on raw limbs, split in two by the GLV endomorphism.
//!
//! Both BN254 groups have the order-3 automorphism `phi(x, y) = (beta x, y)`, which acts
//! on the group as multiplication by a cube root of unity `lambda` mod `r`. Writing
//! `k = k1 + k2 lambda` with `|k1|, |k2| < 2^127` turns one 254-bit ladder into two
//! interleaved 127-bit ones that share their doublings, which is half the doublings for
//! the price of computing `phi` once per table entry (one base field multiply).
//!
//! This is the kernel of every place a point is multiplied by its own scalar rather than
//! summed into an MSM: the ceremony's key scaling (`ptau contribute`, `zkey contribute`,
//! the beacons), the group FFT's twiddles (`ptau prepare`), the small slots of `groth16
//! setup`, and the public-input sum of `verify`. arkworks 0.5 multiplies an affine point
//! by plain double-and-add over all 254 bits, GLV only on the projective path.
//!
//! The decomposition is Babai rounding against the reduced basis `v1 = (A, -B)`,
//! `v2 = (B, C)` of the lattice `{(a, b) : a + b lambda = 0 mod r}` (ark-bn254's
//! `SCALAR_DECOMP_COEFFS`, `A C + B^2 = r`), with the two divisions by `r` replaced by
//! multiplications by `round(2^256 C / r)` and `round(2^256 B / r)`. That rounding is off
//! by at most 0.625 from the exact quotient, which bounds both halves by
//! `0.625 (A + B) < 2^126.4`: inside an `i128`, so the subtraction that produces them can
//! run modulo 2^128 and still be exact. The identity `k1 + k2 lambda = k` holds for any
//! integer quotients; only the bound depends on the rounding.

use ark_ec::short_weierstrass::Affine;
use ark_ff::{MontFp, PrimeField};
use snarkrs_field::raw::{RawField, RawFq, RawFq2};
use snarkrs_field::{Fq, Fr};

use crate::xyzz::{RawCurve, Xyzz};

/// `lambda`, the eigenvalue of `phi` on G1 (ark-bn254's G1 `LAMBDA`). On G2 the same
/// `beta` gives `lambda^2`, so G2's [`GlvField::endo`] uses `beta^2` to stay on this one.
pub const LAMBDA: Fr =
    MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

/// A cube root of unity in `Fq` with `(beta x, y) = [LAMBDA] (x, y)` on G1.
const BETA: Fq =
    MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

/// `beta^2 = -1 - beta`, which is `[LAMBDA]` on G2.
const BETA2: Fq = MontFp!("2203960485148121921418603742825762020974279258880205651966");

/// The reduced basis, `v1 = (A, -B)` and `v2 = (B, C)`.
const A: u128 = 0x6f4d8248eeb859fc8211bbeb7d4f1128;
const B: u128 = 0x89d3256894d213e3;
const C: u128 = 0x6f4d8248eeb859fd0be4e1541221250b;

/// `round(2^256 C / r)` and `round(2^256 B / r)`, little-endian.
const G1: [u64; 3] = [0x5398fd0300ff6565, 0x4ccef014a773d2d2, 0x2];
const G2: [u64; 2] = [0xd91d232ec7e0b3d7, 0x2];

/// wNAF window. Each half's table holds `2^(W-2)` odd multiples and its digits are one
/// nonzero in `W + 1` bits on average, so over two 127-bit halves 5 costs 7 table
/// additions against about 42 in the ladder; 4 and 6 are within a few percent.
const W: u32 = 5;

/// Digits per half: 127 bits plus the one wNAF can carry past the top.
const DIGITS: usize = 130;

/// The coordinate field of a group with the endomorphism: G1's `RawFq` or G2's `RawFq2`.
/// Keyed on the field rather than the curve so code generic over [`RawField`], like the
/// group FFT, can call it.
pub trait GlvField: RawField {
    /// `phi` on the XYZZ representation. `x = X / ZZ`, so scaling `X` scales `x`.
    fn endo(p: &Xyzz<Self>) -> Xyzz<Self>;
}

impl GlvField for RawFq {
    #[inline(always)]
    fn endo(p: &Xyzz<RawFq>) -> Xyzz<RawFq> {
        Xyzz {
            x: p.x.mul(RawFq(BETA.0 .0)),
            ..*p
        }
    }
}

impl GlvField for RawFq2 {
    #[inline(always)]
    fn endo(p: &Xyzz<RawFq2>) -> Xyzz<RawFq2> {
        let b = RawFq(BETA2.0 .0);
        Xyzz {
            x: RawFq2 {
                c0: p.x.c0.mul(b),
                c1: p.x.c1.mul(b),
            },
            ..*p
        }
    }
}

/// `k = k1 + k2 LAMBDA (mod r)` with both halves below 2^127 in magnitude.
pub fn decompose(k: &Fr) -> (i128, i128) {
    let k = k.into_bigint().0;
    let c1 = mul_shift_round(&k, &G1);
    let c2 = mul_shift_round(&k, &G2);
    let lo = k[0] as u128 | (k[1] as u128) << 64;
    // Exact modulo 2^128 because the true values fit an i128 (module doc).
    let k1 = lo
        .wrapping_sub(c1.wrapping_mul(A))
        .wrapping_sub(c2.wrapping_mul(B));
    let k2 = c1.wrapping_mul(B).wrapping_sub(c2.wrapping_mul(C));
    (k1 as i128, k2 as i128)
}

/// `(k g + 2^255) >> 256`, the rounded quotient. Below 2^128 for `k < 2^254`, `g < 2^130`.
#[inline]
fn mul_shift_round(k: &[u64; 4], g: &[u64]) -> u128 {
    let mut t = [0u64; 7];
    for (i, &gi) in g.iter().enumerate() {
        let mut carry = 0u128;
        for (j, &kj) in k.iter().enumerate() {
            let s = t[i + j] as u128 + kj as u128 * gi as u128 + carry;
            t[i + j] = s as u64;
            carry = s >> 64;
        }
        t[i + 4] = carry as u64;
    }
    // + 2^255, carried up through the limbs that are kept.
    let (s3, c) = t[3].overflowing_add(1 << 63);
    t[3] = s3;
    let mut carry = c as u64;
    for limb in &mut t[4..] {
        let (s, c) = limb.overflowing_add(carry);
        *limb = s;
        carry = c as u64;
    }
    t[4] as u128 | (t[5] as u128) << 64
}

/// Signed width-`W` NAF of `k`, least significant digit first. Returns the digit count.
#[inline]
fn wnaf(mut k: u128, out: &mut [i8; DIGITS]) -> usize {
    let mut n = 0;
    while k != 0 {
        let d = if k & 1 == 1 {
            let m = (k & ((1 << W) - 1)) as i8;
            let d = if m >= 1 << (W - 1) { m - (1 << W) } else { m };
            // `k - d` with `d` signed: a negative digit adds.
            k = if d >= 0 {
                k - d as u128
            } else {
                k + (-d) as u128
            };
            d
        } else {
            0
        };
        out[n] = d;
        n += 1;
        k >>= 1;
    }
    n
}

/// `[k] p`. Never inlined: a call is tens of microseconds, so the call costs nothing,
/// and two inlined copies in one loop body (the scaled butterfly) kept LLVM busy for
/// over twenty minutes on the ceremony crate.
#[inline(never)]
pub fn mul<F: GlvField>(p: &Xyzz<F>, k: &Fr) -> Xyzz<F> {
    if p.is_zero() {
        return Xyzz::ZERO;
    }
    let (k1, k2) = decompose(k);
    let mut d1 = [0i8; DIGITS];
    let mut d2 = [0i8; DIGITS];
    let n1 = wnaf(k1.unsigned_abs(), &mut d1);
    let n2 = wnaf(k2.unsigned_abs(), &mut d2);
    let n = n1.max(n2);
    if n == 0 {
        return Xyzz::ZERO;
    }

    // Only as many odd multiples as the digits ask for: a small scalar, which is most of
    // what `groth16 setup` multiplies by, then costs no table at all.
    let top = d1[..n1]
        .iter()
        .chain(&d2[..n2])
        .map(|d| d.unsigned_abs() as usize)
        .max()
        .unwrap_or(1);
    let len = top / 2 + 1;
    let mut t1 = [Xyzz::<F>::ZERO; 1 << (W - 2)];
    let mut t2 = [Xyzz::<F>::ZERO; 1 << (W - 2)];
    t1[0] = *p;
    if len > 1 {
        let two = p.dbl();
        for i in 1..len {
            let mut next = t1[i - 1];
            next.add_assign(&two);
            t1[i] = next;
        }
    }
    if n2 > 0 {
        for i in 0..len {
            t2[i] = F::endo(&t1[i]);
        }
    }
    let (neg1, neg2) = (k1 < 0, k2 < 0);

    let mut acc = Xyzz::<F>::ZERO;
    for i in (0..n).rev() {
        acc = acc.dbl();
        add_digit(&mut acc, &t1, d1[i], neg1);
        add_digit(&mut acc, &t2, d2[i], neg2);
    }
    acc
}

#[inline(always)]
fn add_digit<F: RawField>(acc: &mut Xyzz<F>, table: &[Xyzz<F>], d: i8, neg: bool) {
    if d == 0 {
        return;
    }
    let e = &table[(d.unsigned_abs() >> 1) as usize];
    if (d < 0) != neg {
        acc.add_assign(&Xyzz { y: e.y.neg(), ..*e });
    } else {
        acc.add_assign(e);
    }
}

/// `[k] p` for an affine `p`.
#[inline]
pub fn mul_affine<P: RawCurve>(p: &Affine<P>, k: &Fr) -> Xyzz<P::RF>
where
    P::RF: GlvField,
{
    if p.infinity {
        return Xyzz::ZERO;
    }
    let (x, y) = P::raw_xy(p);
    mul(
        &Xyzz {
            x,
            y,
            zz: P::RF::ONE,
            zzz: P::RF::ONE,
        },
        k,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{BigInt, Field, One, Zero};

    fn edge_scalars() -> Vec<Fr> {
        let r = Fr::MODULUS;
        let mut v = vec![
            Fr::zero(),
            Fr::one(),
            Fr::from(2u64),
            Fr::from(3u64),
            -Fr::one(),
            -Fr::from(2u64),
            LAMBDA,
            -LAMBDA,
            LAMBDA * LAMBDA,
            LAMBDA + Fr::one(),
            Fr::from(u64::MAX),
            Fr::from_bigint(BigInt::new([0, 0, 1, 0])).unwrap(),
            Fr::from_bigint(BigInt::new([u64::MAX, u64::MAX, 0, 0])).unwrap(),
            Fr::from_bigint(BigInt::new([0, 0, 0, 1 << 61])).unwrap(),
            Fr::from_bigint(BigInt::new([r.0[0] - 1, r.0[1], r.0[2], r.0[3]])).unwrap(),
            Fr::from_bigint(BigInt::new([
                r.0[0] >> 1 | r.0[1] << 63,
                r.0[1] >> 1 | r.0[2] << 63,
                r.0[2] >> 1 | r.0[3] << 63,
                r.0[3] >> 1,
            ]))
            .unwrap(),
            Fr::from(2u64).inverse().unwrap(),
        ];
        let mut x = Fr::from(0x9e3779b97f4a7c15u64);
        for _ in 0..2000 {
            x = x.square() + Fr::from(7u64);
            v.push(x);
        }
        v
    }

    #[test]
    fn lambda_is_a_cube_root_of_unity() {
        assert_eq!(LAMBDA * LAMBDA * LAMBDA, Fr::one());
        assert_ne!(LAMBDA, Fr::one());
        assert_eq!(BETA * BETA * BETA, Fq::one());
        assert_eq!(BETA * BETA, BETA2);
    }

    #[test]
    fn rounding_constants_are_what_the_doc_says() {
        // A C + B^2 = r, checked in Fr where r is zero.
        let f = |x: u128| Fr::from(x);
        assert_eq!(f(A) * f(C) + f(B) * f(B), Fr::zero());
        // The two basis vectors are in the lattice.
        assert_eq!(f(A) - f(B) * LAMBDA, Fr::zero());
        assert_eq!(f(B) + f(C) * LAMBDA, Fr::zero());
    }

    #[test]
    fn decomposition_recomposes_and_is_short() {
        for k in edge_scalars() {
            let (k1, k2) = decompose(&k);
            let back = |x: i128| {
                let m = Fr::from(x.unsigned_abs());
                if x < 0 {
                    -m
                } else {
                    m
                }
            };
            assert_eq!(back(k1) + back(k2) * LAMBDA, k, "k = {k}");
            assert!(k1.unsigned_abs() < 1 << 127 && k2.unsigned_abs() < 1 << 127);
        }
        assert_eq!(decompose(&-Fr::one()), (-1, 0));
        assert_eq!(decompose(&Fr::one()), (1, 0));
    }

    #[test]
    fn wnaf_digits_rebuild_the_scalar() {
        for k in [
            0u128,
            1,
            15,
            16,
            17,
            31,
            1 << 126,
            (1 << 127) - 1,
            0x1234_5678_9abc_def0,
        ] {
            let mut d = [0i8; DIGITS];
            let n = wnaf(k, &mut d);
            let mut back: i128 = 0;
            for i in (0..n).rev() {
                back = back * 2 + d[i] as i128;
            }
            assert_eq!(back as u128, k);
            assert!(d[..n]
                .iter()
                .all(|&x| x == 0 || (x & 1 == 1 && x.abs() < 1 << (W - 1))));
        }
    }
}
