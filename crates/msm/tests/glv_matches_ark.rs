//! `glv::mul` against arkworks' scalar multiplication, on both groups, over the scalars
//! a random sweep misses: zero, one, minus one, the eigenvalue itself and its neighbours,
//! and powers of two around the 127-bit split.

use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, FftField};
use ark_ff::{BigInt, Field, One, PrimeField, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;
use snarkrs_field::{Fr, G1Projective, G2Projective};
use snarkrs_msm::glv::{self, GlvField, LAMBDA};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};

fn scalars(rng: &mut StdRng) -> Vec<Fr> {
    let mut v = vec![
        Fr::zero(),
        Fr::one(),
        Fr::from(2u64),
        Fr::from(15u64),
        Fr::from(16u64),
        Fr::from(17u64),
        -Fr::one(),
        -Fr::from(16u64),
        LAMBDA,
        -LAMBDA,
        LAMBDA * LAMBDA,
        LAMBDA - Fr::one(),
        Fr::from(u64::MAX),
        Fr::from(2u64).inverse().unwrap(),
        Fr::from_bigint(BigInt::new([0, 0, 1, 0])).unwrap(),
        Fr::from_bigint(BigInt::new([u64::MAX, u64::MAX, 0, 0])).unwrap(),
        Fr::from_bigint(BigInt::new([0, 0, 0, 1 << 61])).unwrap(),
        Fr::TWO_ADIC_ROOT_OF_UNITY,
    ];
    v.extend((0..64).map(|_| Fr::rand(rng)));
    v
}

fn check<P: RawCurve + ark_ec::CurveConfig<ScalarField = Fr>>(
    label: &str,
    rand: impl Fn(&mut StdRng) -> Projective<P>,
) where
    Projective<P>: std::fmt::Debug,
    P::RF: GlvField,
{
    let mut rng = StdRng::seed_from_u64(0x61_76);
    for _ in 0..4 {
        let p: Affine<P> = rand(&mut rng).into_affine();
        for k in scalars(&mut rng) {
            let want = p * k;
            assert_eq!(
                to_projective(&glv::mul_affine::<P>(&p, &k)),
                want,
                "{label}: affine, k = {k}"
            );
            // A non-normalised input: the same point with ZZ != 1.
            let two = Xyzz::<P::RF>::ZERO;
            let (x, y) = P::raw_xy(&p);
            let mut q = two;
            q.madd(x, y);
            let q = q.dbl();
            assert_eq!(
                to_projective(&glv::mul(&q, &k)),
                want.double(),
                "{label}: xyzz, k = {k}"
            );
        }
        assert!(glv::mul_affine::<P>(&Affine::<P>::identity(), &Fr::one()).is_zero());
        // The endomorphism is [lambda] on this group.
        let (x, y) = P::raw_xy(&p);
        let mut a = Xyzz::<P::RF>::ZERO;
        a.madd(x, y);
        assert_eq!(
            to_projective(&<P::RF as GlvField>::endo(&a)),
            p * LAMBDA,
            "{label}: endo"
        );
    }
}

#[test]
fn g1_matches_ark() {
    check::<snarkrs_field::g1::Config>("G1", |r| G1Projective::rand(r));
}

#[test]
fn g2_matches_ark() {
    check::<snarkrs_field::g2::Config>("G2", |r| G2Projective::rand(r));
}
