//! The raw branch-free field layer against `ark-ff`, element by element.
//!
//! `snarkrs_field::raw` re-implements add/sub/neg/mul/sqr on raw Montgomery limbs so hot
//! loops can drop the compare-and-branch reduction. Wrong Montgomery arithmetic fails
//! quietly in the top limb, so every operation is checked against arkworks over the edge
//! vectors that a random sweep essentially never generates, plus a large random sweep,
//! for both fields and for Fq2.

use ark_ff::BigInt;
use snarkrs_field::raw::{RawFq, RawFq2, RawFr, FQ_MOD, FR_MOD};
use snarkrs_field::{Field, Fq, Fq2, Fr, UniformRand};

/// Structured operands: conditional-subtraction boundaries in both directions, the
/// largest representable values, small integers, and the limb patterns of
/// [`limb_edges`].
fn edge_fq() -> Vec<Fq> {
    let one = Fq::from(1u64);
    let m1 = -one;
    vec![
        Fq::from(0u64),
        one,
        Fq::from(2u64),
        m1,
        m1 - one,
        m1 - Fq::from(2u64),
        Fq::from(1u64 << 63),
    ]
    .into_iter()
    .chain(
        limb_edges(FQ_MOD)
            .into_iter()
            .map(|l| Fq::new_unchecked(BigInt::new(l))),
    )
    .collect()
}

/// Montgomery limb patterns at the boundaries of the carry and borrow chains, as stored:
/// the add, sub and reduction chains run on these limbs, not on the integers they stand
/// for. Every one is below `m`, so each is a valid element.
fn limb_edges(m: [u64; 4]) -> Vec<[u64; 4]> {
    let x = u64::MAX;
    let half = [
        m[0] >> 1 | m[1] << 63,
        m[1] >> 1 | m[2] << 63,
        m[2] >> 1 | m[3] << 63,
        m[3] >> 1,
    ];
    vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [m[0] - 1, m[1], m[2], m[3]],
        [m[0] - 2, m[1], m[2], m[3]],
        [x, x, x, m[3] - 1],
        [0, 0, 0, m[3]],
        half,
        [half[0] + 1, half[1], half[2], half[3]],
        [x, 0, 0, 0],
        [0, x, 0, 0],
        [0, 0, x, 0],
        [x, x, x, 0],
        [0, 0, 0, 1 << 61],
        [x, x, 0, 1 << 61],
    ]
}

fn edge_fr() -> Vec<Fr> {
    let one = Fr::from(1u64);
    let m1 = -one;
    vec![
        Fr::from(0u64),
        one,
        Fr::from(2u64),
        m1,
        m1 - one,
        Fr::from(1u64 << 63),
    ]
    .into_iter()
    .chain(
        limb_edges(FR_MOD)
            .into_iter()
            .map(|l| Fr::new_unchecked(BigInt::new(l))),
    )
    .collect()
}

#[test]
fn raw_fq_matches_ark() {
    let mut rng = ark_std::test_rng();
    let mut pairs: Vec<(Fq, Fq)> = Vec::new();
    for a in edge_fq() {
        for b in edge_fq() {
            pairs.push((a, b));
        }
    }
    for _ in 0..200_000 {
        pairs.push((Fq::rand(&mut rng), Fq::rand(&mut rng)));
    }
    for (i, (a, b)) in pairs.iter().enumerate() {
        let (ra, rb) = (RawFq::from_fq(a), RawFq::from_fq(b));
        assert_eq!(ra.add(rb).to_fq(), *a + *b, "add {i}");
        assert_eq!(ra.sub(rb).to_fq(), *a - *b, "sub {i}");
        assert_eq!(ra.neg().to_fq(), -*a, "neg {i}");
        assert_eq!(ra.mul(rb).to_fq(), *a * *b, "mul {i}");
        assert_eq!(ra.sqr().to_fq(), *a * *a, "sqr {i}");
        assert_eq!(ra.double().to_fq(), *a + *a, "double {i}");
        assert_eq!(ra.is_zero(), a.0 .0 == [0; 4], "is_zero {i}");
    }
}

#[test]
fn raw_fq2_matches_ark() {
    let mut rng = ark_std::test_rng();
    let mut pairs: Vec<(Fq2, Fq2)> = Vec::new();
    for a in edge_fq() {
        for b in edge_fq() {
            pairs.push((Fq2::new(a, b), Fq2::new(b, a)));
        }
    }
    for _ in 0..100_000 {
        pairs.push((Fq2::rand(&mut rng), Fq2::rand(&mut rng)));
    }
    for (i, (a, b)) in pairs.iter().enumerate() {
        let (ra, rb) = (RawFq2::from_fq2(a), RawFq2::from_fq2(b));
        assert_eq!(ra.add(rb).to_fq2(), *a + *b, "add {i}");
        assert_eq!(ra.sub(rb).to_fq2(), *a - *b, "sub {i}");
        assert_eq!(ra.neg().to_fq2(), -*a, "neg {i}");
        assert_eq!(ra.mul(rb).to_fq2(), *a * *b, "mul {i}");
        assert_eq!(ra.sqr().to_fq2(), *a * *a, "sqr {i}");
        assert_eq!(ra.double().to_fq2(), *a + *a, "double {i}");
    }
}

#[test]
fn raw_fr_matches_ark() {
    let mut rng = ark_std::test_rng();
    let mut pairs: Vec<(Fr, Fr)> = Vec::new();
    for a in edge_fr() {
        for b in edge_fr() {
            pairs.push((a, b));
        }
    }
    for _ in 0..200_000 {
        pairs.push((Fr::rand(&mut rng), Fr::rand(&mut rng)));
    }
    for (i, (a, b)) in pairs.iter().enumerate() {
        let (ra, rb) = (RawFr::from_fr(a), RawFr::from_fr(b));
        assert_eq!(ra.add(rb).to_fr(), *a + *b, "add {i}");
        assert_eq!(ra.sub(rb).to_fr(), *a - *b, "sub {i}");
        assert_eq!(ra.mul(rb).to_fr(), *a * *b, "mul {i}");
        assert_eq!(ra.sqr().to_fr(), *a * *a, "sqr {i}");
    }
}

/// A long dependent chain mixing every operation, so a bug that two independent
/// single-op checks could mask (a compensating pair) still surfaces.
#[test]
fn raw_fq_chain_matches_ark() {
    let mut rng = ark_std::test_rng();
    let mut a = Fq::rand(&mut rng);
    let b = Fq::rand(&mut rng);
    let mut ra = RawFq::from_fq(&a);
    let rb = RawFq::from_fq(&b);
    for i in 0..100_000u64 {
        a = a * b + a - b;
        a.square_in_place();
        ra = ra.mul(rb).add(ra).sub(rb).sqr();
        if i % 1000 == 0 {
            assert_eq!(ra.to_fq(), a, "chain diverged at {i}");
        }
    }
    assert_eq!(ra.to_fq(), a);
}
