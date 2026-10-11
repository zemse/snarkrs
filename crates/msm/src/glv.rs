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
use ark_ff::{Field, MontFp, PrimeField};
use snarkrs_field::raw::{RawField, RawFq, RawFq2};
use snarkrs_field::{Fq, Fq2, Fr};

use crate::xyzz::{RawCurve, Xyzz};

mod batch;
pub use batch::{
    mul_batch, mul_batch_in, mul_batch_xyzz, mul_batch_xyzz_in, mul_sum_batch_xyzz_in, BatchState,
};

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
    /// How many parts [`GlvField::split`] cuts a scalar into, for [`mul_batch_xyzz`]: two
    /// on G1, four on G2, which also has `psi`.
    const DIMS: usize;

    /// `phi` on an affine `x`: a multiplication by the cube root of unity.
    fn endo_x(x: Self) -> Self;

    /// The inverse of a nonzero element, through arkworks. Once per batch, never per point.
    fn inv(self) -> Self;

    /// `k = sum_j k_j e_j (mod r)`, with `e_j` the eigenvalue of [`GlvField::endo_j`]`(j)`.
    /// Parts past [`GlvField::DIMS`] are zero.
    fn split(k: &Fr) -> [i128; 4];

    /// The `j`-th endomorphism on an affine point: the identity, then `phi`, then (G2 only)
    /// `psi` and `phi psi`.
    fn endo_j(j: usize, x: Self, y: Self) -> (Self, Self);

    /// `phi` on the XYZZ representation. `x = X / ZZ`, so scaling `X` scales `x`.
    #[inline(always)]
    fn endo(p: &Xyzz<Self>) -> Xyzz<Self> {
        Xyzz {
            x: Self::endo_x(p.x),
            ..*p
        }
    }
}

impl GlvField for RawFq {
    const DIMS: usize = 2;

    #[inline(always)]
    fn endo_x(x: RawFq) -> RawFq {
        x.mul(RawFq(BETA.0 .0))
    }

    fn inv(self) -> RawFq {
        RawFq::from_fq(&self.to_fq().inverse().expect("inverse of zero"))
    }

    fn split(k: &Fr) -> [i128; 4] {
        let (k1, k2) = decompose(k);
        [k1, k2, 0, 0]
    }

    #[inline]
    fn endo_j(j: usize, x: RawFq, y: RawFq) -> (RawFq, RawFq) {
        match j {
            0 => (x, y),
            _ => (Self::endo_x(x), y),
        }
    }
}

impl GlvField for RawFq2 {
    const DIMS: usize = 4;

    #[inline(always)]
    fn endo_x(x: RawFq2) -> RawFq2 {
        let b = RawFq(BETA2.0 .0);
        RawFq2 {
            c0: x.c0.mul(b),
            c1: x.c1.mul(b),
        }
    }

    fn inv(self) -> RawFq2 {
        RawFq2::from_fq2(&self.to_fq2().inverse().expect("inverse of zero"))
    }

    fn split(k: &Fr) -> [i128; 4] {
        decompose4(k)
    }

    #[inline]
    fn endo_j(j: usize, x: RawFq2, y: RawFq2) -> (RawFq2, RawFq2) {
        // psi: Frobenius (conjugation on Fq2) between the twist isomorphisms, ark-bn254's
        // `p_power_endomorphism`.
        let psi = |x: RawFq2, y: RawFq2| {
            let conj = |a: RawFq2| RawFq2 {
                c0: a.c0,
                c1: a.c1.neg(),
            };
            (
                conj(x).mul(RawFq2::from_fq2(&PSI_X)),
                conj(y).mul(RawFq2::from_fq2(&PSI_Y)),
            )
        };
        match j {
            0 => (x, y),
            1 => (Self::endo_x(x), y),
            2 => psi(x, y),
            _ => {
                let (x, y) = psi(x, y);
                (Self::endo_x(x), y)
            }
        }
    }
}

/// `psi(x, y) = (conj(x) PSI_X, conj(y) PSI_Y)` on G2, `(u + 9)^((q - 1)/3)` and
/// `(u + 9)^((q - 1)/2)` (ark-bn254's `P_POWER_ENDOMORPHISM_COEFF_0/1`). It is `[MU]` on G2.
const PSI_X: Fq2 = Fq2::new(
    MontFp!("21575463638280843010398324269430826099269044274347216827212613867836435027261"),
    MontFp!("10307601595873709700152284273816112264069230130616436755625194854815875713954"),
);
const PSI_Y: Fq2 = Fq2::new(
    MontFp!("2821565182194536844548159561693502659359617185244120367078079554186484126554"),
    MontFp!("3505843767911556378687030309984248845540243509899259641013678093033130930403"),
);

/// `MU = 6 x^2 = q mod r`, the eigenvalue of `psi` on G2.
pub const MU: Fr = MontFp!("147946756881789318990833708069417712966");

/// An LLL-reduced basis of `{a : a0 + a1 LAMBDA + a2 MU + a3 LAMBDA MU = 0 mod r}`, every
/// entry below 2^64. Its determinant is `-r`.
const BASIS4: [[i128; 4]; 4] = [
    [9931322734385697762, -9931322734385697763, 1, 0],
    [
        4965661367192848881,
        -4965661367192848882,
        -9931322734385697762,
        -4965661367192848881,
    ],
    [
        -4965661367192848882,
        -9931322734385697763,
        -4965661367192848881,
        4965661367192848881,
    ],
    [
        4965661367192848881,
        9931322734385697762,
        -4965661367192848881,
        4965661367192848882,
    ],
];

/// `round(2^256 n_i / r)` for `(n_0..n_3) / r` the first row of `BASIS4`'s inverse, as
/// (negative, little-endian magnitude).
const G4: [(bool, [u64; 4]); 4] = [
    (
        false,
        [
            0x74f3e6bec8b942da,
            0x5e9aa4e2a056dc2d,
            0x3d00631561b25728,
            0x1,
        ],
    ),
    (false, [0x5398fd0300ff6565, 0x4ccef014a773d2d2, 0x2, 0]),
    (
        true,
        [
            0x071c4c43fac4db00,
            0x55b4ca7ba3e55782,
            0x9e80318ab0d92b95,
            0,
        ],
    ),
    (
        false,
        [
            0x2dff291532e42728,
            0x55b4ca7ba3e5577f,
            0x9e80318ab0d92b95,
            0,
        ],
    ),
];

/// `k = k0 + k1 LAMBDA + k2 MU + k3 LAMBDA MU (mod r)` with every part below 2^66, the
/// four-dimensional GLV-GLS split of G2. Babai rounding against [`BASIS4`] as in
/// [`decompose`]; each quotient is off by at most 0.625, which bounds the parts by
/// `0.625 * 4 * 2^63.2 < 2^65`, so they are exact computed modulo 2^128.
pub fn decompose4(k: &Fr) -> [i128; 4] {
    let kk = k.into_bigint().0;
    let mut v = [kk[0] as u128 | (kk[1] as u128) << 64, 0, 0, 0];
    for (i, (neg, g)) in G4.iter().enumerate() {
        let c = mul_shift_round(&kk, g);
        let c = if *neg { c.wrapping_neg() } else { c };
        for j in 0..4 {
            v[j] = v[j].wrapping_sub(c.wrapping_mul(BASIS4[i][j] as u128));
        }
    }
    v.map(|x| x as i128)
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

/// `(k g + 2^255) >> 256`, the rounded quotient, modulo 2^128 (the callers only need it
/// that far: what they subtract it into fits an `i128`).
#[inline]
fn mul_shift_round(k: &[u64; 4], g: &[u64]) -> u128 {
    let mut t = [0u64; 8];
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
    fn the_four_way_split_recomposes_and_is_short() {
        let mu = MU;
        assert_eq!(
            mu * mu,
            LAMBDA + Fr::one(),
            "psi^2 = phi + 1 on the eigenvalues"
        );
        for b in BASIS4 {
            let f = |x: i128| {
                let m = Fr::from(x.unsigned_abs());
                if x < 0 {
                    -m
                } else {
                    m
                }
            };
            assert_eq!(
                f(b[0]) + f(b[1]) * LAMBDA + f(b[2]) * mu + f(b[3]) * LAMBDA * mu,
                Fr::zero()
            );
        }
        for k in edge_scalars() {
            let p = decompose4(&k);
            let f = |x: i128| {
                let m = Fr::from(x.unsigned_abs());
                if x < 0 {
                    -m
                } else {
                    m
                }
            };
            assert_eq!(
                f(p[0]) + f(p[1]) * LAMBDA + f(p[2]) * mu + f(p[3]) * LAMBDA * mu,
                k,
                "k = {k}"
            );
            assert!(
                p.iter().all(|x| x.unsigned_abs() < 1 << 66),
                "k = {k}: {p:?}"
            );
        }
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
