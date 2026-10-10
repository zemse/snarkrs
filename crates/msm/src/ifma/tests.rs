//! The IFMA fill against the portable one, window by window, and its field layer against
//! ark. Every test returns early on a CPU without AVX-512 IFMA, so they only prove
//! anything where `available()` holds; the box runs print which.

use super::*;
use crate::{window_chunk_batch, CpuMsm, MsmBackend, RECODE_BITS};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use ark_std::{rand::Rng, test_rng, UniformRand};
use snarkrs_field::Fr;

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
                assert_eq!(unlanes(&va.invert()), inv);
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

/// Karatsuba, the lane inversion and the domain round trip for Fq2, against ark.
#[test]
fn fq2_lanes_match_ark() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    for round in 0..512 {
        let mut a: [Fq2; 8] = core::array::from_fn(|_| Fq2::rand(&mut rng));
        let b: [Fq2; 8] = core::array::from_fn(|_| Fq2::rand(&mut rng));
        if round == 0 {
            // A zero component, as a G2 coordinate can have.
            a[1] = Fq2::new(Fq::ZERO, Fq::ONE);
            a[2] = Fq2::new(-Fq::ONE, Fq::ZERO);
        }
        let raw = |x: &[Fq2; 8]| x.map(|e| RawFq2::from_fq2(&e));
        unsafe {
            let (va, vb) = (V2::enter(&raw(&a)), V2::enter(&raw(&b)));
            assert_eq!(va.leave(), raw(&a));
            let want = |f: fn(Fq2, Fq2) -> Fq2| -> [RawFq2; 8] {
                core::array::from_fn(|l| RawFq2::from_fq2(&f(a[l], b[l])))
            };
            assert_eq!(va.mul(&vb).leave(), want(|x, y| x * y));
            assert_eq!(va.add(&vb).leave(), want(|x, y| x + y));
            assert_eq!(va.sub(&vb).leave(), want(|x, y| x - y));
            assert_eq!(va.sub(&va).is_zero(), 0xff);
            assert_eq!(va.invert().leave(), raw(&a.map(|x| x.inverse().unwrap())));
        }
    }
}

fn walk<P: RawCurve>(n: usize, rng: &mut impl Rng) -> Vec<Affine<P>>
where
    Projective<P>: UniformRand,
{
    let step = Projective::<P>::rand(rng);
    let mut cur = Projective::<P>::rand(rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    Projective::normalize_batch(&proj)
}

fn scan_of<P: RawCurve>(scalars: &[Fr]) -> Prescan<P> {
    Prescan {
        idx: (0..scalars.len() as u32).collect(),
        bigints: scalars.iter().map(|s| s.into_bigint()).collect(),
        ones_sum: Projective::zero(),
    }
}

/// Every window at every `c`, whole range and an offset sub-range, both work modes.
fn chunks_agree<P: FastFill>(bases: &[Affine<P>], scalars: &[Fr], cs: &[u32]) -> usize {
    let scan = scan_of::<P>(scalars);
    let prep = P::prepare(bases, &scan).expect("IFMA is available");
    let n = bases.len();
    let mut checked = 0;
    for &c in cs {
        for w in 0..RECODE_BITS.div_ceil(c as usize) {
            for (range, work) in [(0..n, Work::Variable), (n / 3..n, Work::Constant)] {
                assert_eq!(
                    P::window_chunk(&prep, bases, &scan, range.clone(), w, c, work),
                    window_chunk_batch(bases, &scan, range, w, c, work),
                    "window {w} at c = {c}"
                );
                checked += 1;
            }
        }
    }
    checked
}

fn random_chunks<P: FastFill>(cs: core::ops::RangeInclusive<u32>) -> usize
where
    Projective<P>: UniformRand,
{
    let mut rng = test_rng();
    let mut checked = 0;
    for c in cs {
        // Enough points per bucket that every window flushes several full rounds.
        let n = 3 << (c - 1);
        let bases = walk::<P>(n, &mut rng);
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        checked += chunks_agree(&bases, &scalars, &[c]);
    }
    checked
}

#[test]
fn fill_matches_the_portable_fill_g1() {
    if skip() {
        return;
    }
    let checked = random_chunks::<G1>(9..=15);
    eprintln!("ifma: {checked} random G1 window chunks agree");
}

#[test]
fn fill_matches_the_portable_fill_g2() {
    if skip() {
        return;
    }
    let checked = random_chunks::<G2>(9..=13);
    eprintln!("ifma: {checked} random G2 window chunks agree");
}

/// Doublings, cancellations, a hot bucket and zero digits, at the narrowest and the
/// widest batched window.
fn degenerate<P: FastFill>(n: usize, cs: &[u32]) -> usize
where
    Projective<P>: UniformRand,
{
    let mut rng = test_rng();
    let mut bases = walk::<P>(n, &mut rng);
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
    let checked = chunks_agree(&bases, &s, cs);

    // All-equal scalars over distinct bases, and a single hot bucket.
    let k = Fr::rand(&mut rng);
    let eq = vec![k; n];
    let distinct = walk::<P>(n, &mut rng);
    let checked = checked + chunks_agree(&distinct, &eq, &cs[..1]);
    let hot = vec![distinct[0]; n];
    checked + chunks_agree(&hot, &eq, &cs[..1])
}

#[test]
fn fill_matches_on_degenerate_digits_g1() {
    if skip() {
        return;
    }
    let checked = degenerate::<G1>(4000, &[9, 10, 12, 15]);
    eprintln!("ifma: {checked} degenerate G1 window chunks agree");
}

#[test]
fn fill_matches_on_degenerate_digits_g2() {
    if skip() {
        return;
    }
    let checked = degenerate::<G2>(2000, &[9, 13]);
    eprintln!("ifma: {checked} degenerate G2 window chunks agree");
}

/// End to end through `msm_g1`/`msm_g2`, which dispatch here, against arkworks: the
/// first lengths at c = 9, 12 and 15, bases at infinity, and constant work.
#[test]
fn msm_matches_arkworks() {
    if skip() {
        return;
    }
    let mut rng = test_rng();
    // Every 7th base is infinity, so `m = n - ceil(n / 7)` sets the window.
    for (n, c) in [(3_884usize, 9u32), (38_230, 12), (305_838, 15)] {
        assert_eq!(crate::window_size(n - n.div_ceil(7)), c, "n = {n}");
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let mut g1 = walk::<G1>(n, &mut rng);
        for i in (0..n).step_by(7) {
            g1[i] = Affine::identity();
        }
        let want = Projective::<G1>::msm(&g1, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g1(&g1, &scalars), want, "G1 n = {n}");
        assert_eq!(
            CpuMsm::constant_work().msm_g1(&g1, &scalars),
            want,
            "G1 n = {n}, constant work"
        );
        if c > 12 {
            // G2 at 2^18 is slow to build and to check; c = 9 and 12 cover its path.
            continue;
        }
        let mut g2 = walk::<G2>(n, &mut rng);
        for i in (0..n).step_by(7) {
            g2[i] = Affine::identity();
        }
        let want = Projective::<G2>::msm(&g2, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g2(&g2, &scalars), want, "G2 n = {n}");
        assert_eq!(
            CpuMsm::constant_work().msm_g2(&g2, &scalars),
            want,
            "G2 n = {n}, constant work"
        );
    }
    eprintln!("ifma: msm_g1 agrees with arkworks at c = 9, 12, 15, msm_g2 at c = 9, 12");
}
