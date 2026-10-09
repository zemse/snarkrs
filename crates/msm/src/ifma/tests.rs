//! The IFMA fill against the portable one, window by window, and its field layer against
//! ark. Every test returns early on a CPU without AVX-512 IFMA, so they only prove
//! anything where `available()` holds; the box runs print which.

use super::*;
use crate::{window_chunk_batch, CpuMsm, MsmBackend, RECODE_BITS};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use ark_std::{rand::Rng, test_rng, UniformRand};
use snarkrs_field::{Fr, G1Projective};

fn skip() -> bool {
    if available() {
        return false;
    }
    eprintln!("no AVX-512 IFMA on this CPU, skipped");
    true
}

/// Into the 2^260 domain, eight lanes.
fn lanes(xs: &[Fq; 8]) -> V {
    let mut cols = [[0u64; 8]; 5];
    for (l, x) in xs.iter().enumerate() {
        let limbs = to_52(RawFq::from_fq(x).0);
        for j in 0..5 {
            cols[j][l] = limbs[j];
        }
    }
    unsafe { mul(&load_lanes(&cols), &splat(&INTO)) }
}

/// And back, canonical.
fn unlanes(v: &V) -> [Fq; 8] {
    let cols = unsafe { store_lanes(&cond_sub(&mul(v, &splat(&OUT_OF)), &P)) };
    core::array::from_fn(|l| {
        RawFq(from_52([
            cols[0][l], cols[1][l], cols[2][l], cols[3][l], cols[4][l],
        ]))
        .to_fq()
    })
}

#[test]
fn constants_match_ark() {
    let two = Fq::from(2u64);
    for (e, got) in [(260u64, ONE), (264, INTO), (256, OUT_OF)] {
        // `2^e mod p` as an integer is the canonical form of the element 2^e.
        assert_eq!(from_52(got), two.pow([e]).into_bigint().0, "2^{e}");
    }
    assert_eq!(from_52(P), FQ_MOD);
    assert_eq!(P[0].wrapping_mul(K0) & MASK, MASK, "k0 is -p^-1 mod 2^52");
}

#[test]
fn field_ops_match_ark() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    let mut vals: Vec<Fq> = vec![Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::from(2u64), -Fq::from(2u64)];
    vals.extend((0..4091).map(|_| Fq::rand(&mut rng)));
    for (ch, other) in vals.chunks_exact(8).zip(vals.chunks_exact(8).rev()) {
        let a: [Fq; 8] = ch.try_into().unwrap();
        let b: [Fq; 8] = other.try_into().unwrap();
        let (va, vb) = (lanes(&a), lanes(&b));
        let want = |f: fn(Fq, Fq) -> Fq| -> [Fq; 8] { core::array::from_fn(|l| f(a[l], b[l])) };
        unsafe {
            assert_eq!(unlanes(&mul(&va, &vb)), want(|x, y| x * y));
            assert_eq!(unlanes(&add(&va, &vb)), want(|x, y| x + y));
            assert_eq!(unlanes(&sub(&va, &vb)), want(|x, y| x - y));
            assert_eq!(unlanes(&sub(&va, &va)), [Fq::ZERO; 8]);
            let z = is_zero(&va);
            for l in 0..8 {
                assert_eq!(z >> l & 1 != 0, a[l].is_zero());
            }
            // A lazy-range input: va + va is in [0, 2p) but rarely canonical.
            assert_eq!(is_zero(&sub(&add(&va, &va), &add(&va, &va))), 0xff);
            let mut inv = [Fq::ZERO; 8];
            for l in 0..8 {
                inv[l] = if a[l].is_zero() {
                    Fq::ONE
                } else {
                    a[l].inverse().unwrap()
                };
            }
            if a.iter().all(|x| !x.is_zero()) {
                assert_eq!(unlanes(&invert_lanes(&va)), inv);
            }
        }
    }
    // A long lazy chain.
    let a: [Fq; 8] = vals[8..16].try_into().unwrap();
    let b: [Fq; 8] = vals[16..24].try_into().unwrap();
    let (mut va, vb) = (lanes(&a), lanes(&b));
    let mut want = a;
    for _ in 0..2000 {
        unsafe {
            va = sub(&mul(&va, &vb), &add(&va, &vb));
        }
        for l in 0..8 {
            want[l] = want[l] * b[l] - (want[l] + b[l]);
        }
    }
    assert_eq!(unlanes(&va), want);
}

fn walk(n: usize, rng: &mut impl Rng) -> Vec<G1Affine> {
    let step = G1Projective::rand(rng);
    let mut cur = G1Projective::rand(rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    G1Projective::normalize_batch(&proj)
}

fn scan_of(scalars: &[Fr]) -> Prescan<G1> {
    Prescan {
        idx: (0..scalars.len() as u32).collect(),
        bigints: scalars.iter().map(|s| s.into_bigint()).collect(),
        ones_sum: Projective::zero(),
    }
}

/// Every window at every `c`, whole range and an offset sub-range, both work modes.
fn chunks_agree(bases: &[G1Affine], scalars: &[Fr], cs: &[u32]) -> usize {
    let scan = scan_of(scalars);
    let prep = G1::prepare(bases, &scan).expect("IFMA is available");
    let n = bases.len();
    let mut checked = 0;
    for &c in cs {
        for w in 0..RECODE_BITS.div_ceil(c as usize) {
            for (range, work) in [(0..n, Work::Variable), (n / 3..n, Work::Constant)] {
                assert_eq!(
                    G1::window_chunk(&prep, bases, &scan, range.clone(), w, c, work),
                    window_chunk_batch(bases, &scan, range, w, c, work),
                    "window {w} at c = {c}"
                );
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn fill_matches_the_portable_fill() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    let mut checked = 0;
    for c in 9u32..=15 {
        // Enough points per bucket that every window flushes several full rounds.
        let n = 3 << (c - 1);
        let bases = walk(n, &mut rng);
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        checked += chunks_agree(&bases, &scalars, &[c]);
    }
    eprintln!("ifma: {checked} random window chunks agree");
}

/// Doublings, cancellations, a hot bucket and zero digits, at the narrowest and the
/// widest batched window.
#[test]
fn fill_matches_on_degenerate_digits() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    let n = 4000;
    let mut bases = walk(n, &mut rng);
    let mut s: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    for i in 0..n / 4 {
        // One point under one scalar: every window funnels into a single bucket, all
        // doublings, through the retries and the side buckets.
        bases[i] = bases[0];
        s[i] = s[0];
    }
    for i in n / 4..n / 2 {
        // P and -P under one scalar: cancellations.
        bases[i] = if i % 2 == 0 {
            bases[n / 4]
        } else {
            -bases[n / 4]
        };
        s[i] = s[n / 4];
    }
    for i in n / 2..n / 2 + 300 {
        // The same point, alternately negated, under different scalars: doublings and
        // cancellations scattered through ordinary rounds.
        bases[i] = if i % 3 == 0 {
            -bases[n / 2]
        } else {
            bases[n / 2]
        };
    }
    for si in s.iter_mut().skip(n - 200) {
        *si = Fr::zero();
    }
    for si in s.iter_mut().skip(n - 400).take(100) {
        *si = Fr::from(3u64);
    }
    let checked = chunks_agree(&bases, &s, &[9, 10, 12, 15]);

    // All-equal scalars over distinct bases, and a single hot bucket.
    let k = Fr::rand(&mut rng);
    let eq = vec![k; n];
    let distinct = walk(n, &mut rng);
    let checked = checked + chunks_agree(&distinct, &eq, &[9, 13]);
    let hot = vec![distinct[0]; n];
    let checked = checked + chunks_agree(&hot, &eq, &[9, 13]);
    eprintln!("ifma: {checked} degenerate window chunks agree");
}

/// End to end through `msm_g1`, which dispatches here, against arkworks: the first
/// lengths at c = 9, 12 and 15, bases at infinity, and constant work.
#[test]
fn msm_g1_matches_arkworks() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    // Every 7th base is infinity, so `m = n - ceil(n / 7)` sets the window.
    for (n, c) in [(3_884usize, 9u32), (38_230, 12), (305_838, 15)] {
        assert_eq!(crate::window_size(n - n.div_ceil(7)), c, "n = {n}");
        let mut bases = walk(n, &mut rng);
        for i in (0..n).step_by(7) {
            bases[i] = G1Affine::identity();
        }
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let want = G1Projective::msm(&bases, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g1(&bases, &scalars), want, "n = {n}");
        assert_eq!(
            CpuMsm::constant_work().msm_g1(&bases, &scalars),
            want,
            "n = {n}, constant work"
        );
    }
    eprintln!("ifma: msm_g1 agrees with arkworks at c = 9, 12, 15");
}
