//! The portable multiply of an x86_64 build without `+adx,+bmi2`, against `ark-ff`.
//!
//! Such a build picks the `mulx` multiply at run time on a CPU with ADX and BMI2, which is
//! every machine the other tests run on, so they never reach the fallback. This binary turns
//! the `mulx` multiply off before anything else and checks that the fallback is taken and
//! still matches arkworks. A `+adx,+bmi2` build has no fallback and only checks the flag.
#![cfg(target_arch = "x86_64")]

use snarkrs_field::raw::{self, RawFq, RawFq2, RawFr};
use snarkrs_field::{Fq, Fq2, Fr, UniformRand};

#[test]
fn portable_multiply_matches_ark() {
    let compiled = cfg!(all(target_feature = "adx", target_feature = "bmi2"));
    raw::disable_mulx();
    assert_eq!(raw::mulx_multiply(), compiled);

    let mut rng = ark_std::test_rng();
    let one = Fq::from(1u64);
    let edges = [Fq::from(0u64), one, -one, -one - one, Fq::from(1u64 << 63)];
    let mut fq: Vec<(Fq, Fq)> = edges
        .iter()
        .flat_map(|a| edges.iter().map(move |b| (*a, *b)))
        .collect();
    fq.extend((0..50_000).map(|_| (Fq::rand(&mut rng), Fq::rand(&mut rng))));
    for (i, (a, b)) in fq.iter().enumerate() {
        let (ra, rb) = (RawFq::from_fq(a), RawFq::from_fq(b));
        assert_eq!(ra.mul(rb).to_fq(), *a * *b, "fq mul {i}");
        assert_eq!(ra.add(rb).to_fq(), *a + *b, "fq add {i}");
        assert_eq!(ra.sub(rb).to_fq(), *a - *b, "fq sub {i}");
        let (a2, b2) = (Fq2::new(*a, *b), Fq2::new(*b, *a));
        let (ra2, rb2) = (RawFq2::from_fq2(&a2), RawFq2::from_fq2(&b2));
        assert_eq!(ra2.mul(rb2).to_fq2(), a2 * b2, "fq2 mul {i}");
    }

    let rone = Fr::from(1u64);
    let edges = [Fr::from(0u64), rone, -rone, -rone - rone];
    let mut fr: Vec<(Fr, Fr)> = edges
        .iter()
        .flat_map(|a| edges.iter().map(move |b| (*a, *b)))
        .collect();
    fr.extend((0..50_000).map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng))));
    for (i, (a, b)) in fr.iter().enumerate() {
        let (ra, rb) = (RawFr::from_fr(a), RawFr::from_fr(b));
        assert_eq!(ra.mul(rb).to_fr(), *a * *b, "fr mul {i}");
    }
}
