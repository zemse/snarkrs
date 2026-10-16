//! Single-thread cost of one variable-base scalar multiplication, three ways: arkworks'
//! affine multiply, `glv::mul` (XYZZ) and `glv::mul_batch_xyzz` (affine, batched), on both
//! groups, over random points and scalars. Every result is checked against arkworks.
//!
//! `cargo run --release -p snarkrs-msm --example glv_cost -- [BATCH...]`
//!
//! One `group way batch us_per_mul` line each, the median of `REPS` (default 7) runs.

use std::time::Instant;

use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::CurveGroup;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;
use snarkrs_field::{Fr, G1Projective, G2Projective};
use snarkrs_msm::glv::{self, GlvField};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run<P: RawCurve + ark_ec::CurveConfig<ScalarField = Fr>>(
    name: &str,
    batches: &[usize],
    reps: usize,
    rand: impl Fn(&mut StdRng) -> Projective<P>,
) where
    P::RF: GlvField,
{
    let mut rng = StdRng::seed_from_u64(0x9c05);
    let n = *batches.iter().max().unwrap();
    let ps: Vec<Affine<P>> = (0..n).map(|_| rand(&mut rng).into_affine()).collect();
    let ks: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let want: Vec<Projective<P>> = ps.iter().zip(&ks).map(|(p, k)| *p * k).collect();
    let xyzz: Vec<Xyzz<P::RF>> = ps
        .iter()
        .map(|p| {
            let (x, y) = P::raw_xy(p);
            Xyzz {
                x,
                y,
                zz: <P::RF as snarkrs_field::raw::RawField>::ONE,
                zzz: <P::RF as snarkrs_field::raw::RawField>::ONE,
            }
        })
        .collect();

    let m = 256.min(n);
    let mut t = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let got: Vec<_> = ps[..m].iter().zip(&ks).map(|(p, k)| *p * k).collect();
        t.push(t0.elapsed().as_secs_f64() * 1e6 / m as f64);
        assert_eq!(got[0], want[0]);
    }
    println!("{name} ark 1 {:.2}", median(t));

    let mut t = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let got: Vec<_> = xyzz[..m]
            .iter()
            .zip(&ks)
            .map(|(p, k)| glv::mul(p, k))
            .collect();
        t.push(t0.elapsed().as_secs_f64() * 1e6 / m as f64);
        for i in 0..m {
            assert_eq!(to_projective::<P>(&got[i]), want[i]);
        }
    }
    println!("{name} xyzz 1 {:.2}", median(t));

    for &b in batches {
        let mut out = vec![Xyzz::<P::RF>::ZERO; b];
        let mut t = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            glv::mul_batch_xyzz(&xyzz[..b], &ks[..b], &mut out);
            t.push(t0.elapsed().as_secs_f64() * 1e6 / b as f64);
        }
        for i in 0..b {
            assert_eq!(to_projective::<P>(&out[i]), want[i]);
        }
        println!("{name} batch {b} {:.2}", median(t));
    }

    // Sums of two products, `b` of them from `2b` points: us per sum.
    for &b in batches.iter().filter(|&&b| 2 * b <= n) {
        let mut st = glv::BatchState::default();
        let mut out = vec![Xyzz::<P::RF>::ZERO; b];
        let mut t = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            glv::mul_sum_batch_xyzz_in(&mut st, &xyzz[..2 * b], &ks[..2 * b], 2, &mut out);
            t.push(t0.elapsed().as_secs_f64() * 1e6 / b as f64);
        }
        for i in 0..b {
            assert_eq!(to_projective::<P>(&out[i]), want[2 * i] + want[2 * i + 1]);
        }
        println!("{name} sum2 {b} {:.2}", median(t));
    }
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let mut batches: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("batch size"))
        .collect();
    if batches.is_empty() {
        batches = vec![64, 256, 1024, 2048, 4096];
    }
    run::<snarkrs_field::g1::Config>("G1", &batches, reps, G1Projective::rand);
    run::<snarkrs_field::g2::Config>("G2", &batches, reps, G2Projective::rand);
}
