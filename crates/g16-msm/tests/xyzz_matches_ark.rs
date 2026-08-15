//! The XYZZ point layer against `ark_ec`, operation by operation, on both groups.
//!
//! The MSM tests already cross-check whole MSMs; this file pins the individual point
//! operations, including the corners a random MSM essentially never exercises: the
//! identity in either operand, adding a point to itself (the doubling branch), adding a
//! point to its negation (the cancellation branch), and long mixed chains.

use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::AdditiveGroup;
use ark_std::UniformRand;
use g16_field::{G1Projective, G2Projective};
use g16_msm::xyzz::{to_projective, RawCurve, Xyzz};

fn check_group<P: RawCurve>(label: &str, rand_proj: impl Fn(&mut ark_std::rand::rngs::StdRng) -> Projective<P>)
where
    Projective<P>: std::fmt::Debug,
{
    use ark_std::rand::SeedableRng;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xC0FFEE);

    // madd against ark, random points, accumulating so acc passes through many states.
    let pts: Vec<Affine<P>> = (0..256)
        .map(|_| rand_proj(&mut rng).into_affine())
        .collect();
    let mut acc = Xyzz::<P::RF>::ZERO;
    let mut want = Projective::<P>::default();
    // Projective::default() is not the identity in ark; build zero explicitly.
    want -= want;
    for (i, p) in pts.iter().enumerate() {
        let (x, y) = P::raw_xy(p);
        acc.madd(x, y);
        want += p;
        assert_eq!(to_projective(&acc), want, "{label}: madd chain diverged at {i}");
    }

    // The corners.
    let p = pts[0];
    let (px, py) = P::raw_xy(&p);

    // identity += p
    let mut a = Xyzz::<P::RF>::ZERO;
    a.madd(px, py);
    assert_eq!(to_projective(&a), p.into_group(), "{label}: identity + p");

    // p += p, the doubling branch of madd
    let mut d = Xyzz::<P::RF>::ZERO;
    d.madd(px, py);
    d.madd(px, py);
    assert_eq!(to_projective(&d), p.into_group() + p, "{label}: p + p doubles");

    // p += -p, the cancellation branch
    let neg = (-p.into_group()).into_affine();
    let (nx, ny) = P::raw_xy(&neg);
    let mut z = Xyzz::<P::RF>::ZERO;
    z.madd(px, py);
    z.madd(nx, ny);
    assert!(z.is_zero(), "{label}: p + (-p) must cancel to the identity");
    assert_eq!(to_projective(&z), Projective::<P>::from(p) - p, "{label}: cancel value");

    // full add: both identity cases, doubling, cancellation, and randoms
    let q = pts[1];
    let mut xa = Xyzz::<P::RF>::ZERO;
    xa.madd(px, py);
    let mut xb = Xyzz::<P::RF>::ZERO;
    let (qx, qy) = P::raw_xy(&q);
    xb.madd(qx, qy);

    let mut s = xa;
    s.add_assign(&xb);
    assert_eq!(to_projective(&s), p.into_group() + q, "{label}: add");

    let mut s = xa;
    s.add_assign(&Xyzz::<P::RF>::ZERO);
    assert_eq!(to_projective(&s), p.into_group(), "{label}: add identity rhs");

    let mut s = Xyzz::<P::RF>::ZERO;
    s.add_assign(&xa);
    assert_eq!(to_projective(&s), p.into_group(), "{label}: add identity lhs");

    let mut s = xa;
    s.add_assign(&xa);
    assert_eq!(to_projective(&s), p.into_group() + p, "{label}: add self doubles");

    let mut xneg = Xyzz::<P::RF>::ZERO;
    xneg.madd(nx, ny);
    let mut s = xa;
    s.add_assign(&xneg);
    assert!(s.is_zero(), "{label}: add of negation cancels");

    // dbl against ark, including through non-trivial ZZ (accumulate first)
    let mut acc = Xyzz::<P::RF>::ZERO;
    acc.madd(px, py);
    acc.madd(qx, qy);
    let d = acc.dbl();
    let want = (p.into_group() + q).double();
    assert_eq!(to_projective(&d), want, "{label}: dbl");
    assert!(Xyzz::<P::RF>::ZERO.dbl().is_zero(), "{label}: dbl identity");
}

#[test]
fn xyzz_g1_matches_ark() {
    check_group::<g16_field::g1::Config>("g1", |rng| G1Projective::rand(rng));
}

#[test]
fn xyzz_g2_matches_ark() {
    check_group::<g16_field::g2::Config>("g2", |rng| G2Projective::rand(rng));
}
