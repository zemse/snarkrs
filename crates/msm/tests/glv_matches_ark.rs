//! `glv::mul` against arkworks' scalar multiplication, on both groups, over the scalars
//! a random sweep misses: zero, one, minus one, the eigenvalue itself and its neighbours,
//! and powers of two around the 127-bit split.

use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, FftField};
use ark_ff::{BigInt, Field, One, PrimeField, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;
use snarkrs_field::raw::RawField;
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

fn check_batch<P: RawCurve + ark_ec::CurveConfig<ScalarField = Fr>>(
    label: &str,
    rand: impl Fn(&mut StdRng) -> Projective<P>,
) where
    Projective<P>: std::fmt::Debug,
    P::RF: GlvField,
{
    let mut rng = StdRng::seed_from_u64(0xba7c);
    let mut ks = scalars(&mut rng);
    let n = ks.len();
    let mut ps: Vec<Affine<P>> = (0..n).map(|_| rand(&mut rng).into_affine()).collect();
    // The identity in, a repeated point, and one point under opposite scalars.
    ps[3] = Affine::<P>::identity();
    ps[5] = ps[4];
    ps[7] = ps[6];
    ks[7] = -ks[6];
    let mut out = vec![Affine::<P>::identity(); n];
    glv::mul_batch(&ps, &ks, &mut out);
    for i in 0..n {
        assert_eq!(
            out[i],
            (ps[i] * ks[i]).into_affine(),
            "{label}: lane {i}, k = {}",
            ks[i]
        );
    }

    // Non-normalised XYZZ in (each point doubled, so ZZ != 1), affine XYZZ out.
    let xyzz: Vec<Xyzz<P::RF>> = ps
        .iter()
        .map(|p| {
            let mut a = Xyzz::<P::RF>::ZERO;
            if !p.infinity {
                let (x, y) = P::raw_xy(p);
                a.madd(x, y);
            }
            a.dbl()
        })
        .collect();
    let mut res = vec![Xyzz::<P::RF>::ZERO; n];
    glv::mul_batch_xyzz(&xyzz, &ks, &mut res);
    for i in 0..n {
        assert_eq!(
            to_projective(&res[i]),
            (ps[i] * ks[i]).double(),
            "{label}: xyzz lane {i}, k = {}",
            ks[i]
        );
        assert!(res[i].is_zero() || (res[i].zz == P::RF::ONE && res[i].zzz == P::RF::ONE));
    }
}

#[test]
fn g1_batch_matches_ark() {
    check_batch::<snarkrs_field::g1::Config>("G1", |r| G1Projective::rand(r));
}

#[test]
fn g2_batch_matches_ark() {
    check_batch::<snarkrs_field::g2::Config>("G2", |r| G2Projective::rand(r));
}

/// `mul_sum_batch_xyzz_in` over sums of one to three terms: every edge scalar beside random
/// ones, the identity among the terms, a sum that cancels (`[k] P + [-k] P`), a term
/// repeated, and the empty batch.
fn check_sum<P: RawCurve + ark_ec::CurveConfig<ScalarField = Fr>>(
    label: &str,
    rand: impl Fn(&mut StdRng) -> Projective<P>,
) where
    Projective<P>: std::fmt::Debug,
    P::RF: GlvField,
{
    let mut rng = StdRng::seed_from_u64(0x5a3d);
    let mut st = glv::BatchState::default();
    glv::mul_sum_batch_xyzz_in::<P::RF>(&mut st, &[], &[], 2, &mut []);
    for terms in 1..=3 {
        let mut ks = scalars(&mut rng);
        ks.extend(scalars(&mut rng));
        ks.truncate(ks.len() / terms * terms);
        let m = ks.len();
        let mut ps: Vec<Affine<P>> = (0..m).map(|_| rand(&mut rng).into_affine()).collect();
        ps[1] = Affine::<P>::identity();
        if terms > 1 {
            // Output 2 cancels; output 3 repeats a term.
            ps[2 * terms + 1] = ps[2 * terms];
            ks[2 * terms + 1] = -ks[2 * terms];
            ps[3 * terms + 1] = ps[3 * terms];
            ks[3 * terms + 1] = ks[3 * terms];
        }
        let xyzz: Vec<Xyzz<P::RF>> = ps
            .iter()
            .map(|p| {
                let mut a = Xyzz::<P::RF>::ZERO;
                if !p.infinity {
                    let (x, y) = P::raw_xy(p);
                    a.madd(x, y);
                }
                a
            })
            .collect();
        let mut out = vec![Xyzz::<P::RF>::ZERO; m / terms];
        glv::mul_sum_batch_xyzz_in(&mut st, &xyzz, &ks, terms, &mut out);
        for (i, o) in out.iter().enumerate() {
            let want: Projective<P> = (0..terms)
                .map(|t| ps[i * terms + t] * ks[i * terms + t])
                .sum();
            assert_eq!(to_projective(o), want, "{label}: {terms} terms, output {i}");
            assert!(o.is_zero() || (o.zz == P::RF::ONE && o.zzz == P::RF::ONE));
        }
    }
}

/// `mul_sums_xyzz_in`, a term count per sum: empty sums, single terms, a sum of 31, every
/// edge scalar (so tables of one entry up to the full eight), the identity, a sum that
/// cancels, and repeats.
fn check_sums<P: RawCurve + ark_ec::CurveConfig<ScalarField = Fr>>(
    label: &str,
    rand: impl Fn(&mut StdRng) -> Projective<P>,
) where
    Projective<P>: std::fmt::Debug,
    P::RF: GlvField,
{
    let mut rng = StdRng::seed_from_u64(0x7a66);
    let mut st = glv::BatchState::default();
    let sizes = [0usize, 1, 2, 3, 0, 5, 1, 31, 4, 2, 0, 7, 1, 1, 3];
    let mut ks = scalars(&mut rng);
    ks.extend(scalars(&mut rng));
    let m: usize = sizes.iter().sum();
    ks.truncate(m);
    let mut ps: Vec<Affine<P>> = (0..m).map(|_| rand(&mut rng).into_affine()).collect();
    ps[10] = Affine::<P>::identity();
    // The sum of three at terms 3..6 cancels its first two; the sum of 31 repeats one.
    ps[4] = ps[3];
    ks[4] = -ks[3];
    ps[20] = ps[19];
    ks[20] = ks[19];
    let xyzz: Vec<Xyzz<P::RF>> = ps
        .iter()
        .map(|p| {
            let mut a = Xyzz::<P::RF>::ZERO;
            if !p.infinity {
                let (x, y) = P::raw_xy(p);
                a.madd(x, y);
            }
            a
        })
        .collect();
    let ends: Vec<usize> = sizes
        .iter()
        .scan(0, |e, s| {
            *e += s;
            Some(*e)
        })
        .collect();
    let mut out = vec![Xyzz::<P::RF>::ZERO; sizes.len()];
    glv::mul_sums_xyzz_in(&mut st, &xyzz, &ks, &ends, &mut out);
    let mut from = 0;
    for (i, &to) in ends.iter().enumerate() {
        let want: Projective<P> = (from..to).map(|u| ps[u] * ks[u]).sum();
        assert_eq!(to_projective(&out[i]), want, "{label}: sum {i}");
        from = to;
    }
    glv::mul_sums_xyzz_in::<P::RF>(&mut st, &[], &[], &[], &mut []);
}

#[test]
fn g1_sums_match_ark() {
    check_sums::<snarkrs_field::g1::Config>("G1", |r| G1Projective::rand(r));
}

#[test]
fn g2_sums_match_ark() {
    check_sums::<snarkrs_field::g2::Config>("G2", |r| G2Projective::rand(r));
}

#[test]
fn g1_sum_batch_matches_ark() {
    check_sum::<snarkrs_field::g1::Config>("G1", |r| G1Projective::rand(r));
}

#[test]
fn g2_sum_batch_matches_ark() {
    check_sum::<snarkrs_field::g2::Config>("G2", |r| G2Projective::rand(r));
}
