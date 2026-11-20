//! Dense G1 MSM probe: `cargo run --release -p snarkrs-msm --example zz_inv -- LOG2N [REPS]`.
//! Prints ns per point per thread, the number the bucket fill is judged on.
use ark_ec::CurveGroup;
use ark_ff::{Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use snarkrs_field::{Fq, Fr, G1Affine, G1Projective};
use snarkrs_msm::{CpuMsm, MsmBackend};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let log: u32 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(18);
    let reps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let n = 1usize << log;
    let mut rng = StdRng::seed_from_u64(7);
    let mut x = Fq::rand(&mut rng);
    let t = Instant::now();
    for _ in 0..200 {
        x = x.inverse().unwrap() + Fq::ONE;
    }
    eprintln!("Fq inverse {:.0} ns", t.elapsed().as_nanos() as f64 / 200.0);
    let mut r = snarkrs_field::raw::RawFq::from_fq(&x);
    let t = Instant::now();
    for _ in 0..20000 {
        r = r.inv().unwrap().add(snarkrs_field::raw::RawFq::ONE);
    }
    eprintln!("RawFq inv {:.0} ns {:?}", t.elapsed().as_nanos() as f64 / 20000.0, r.0[0]);
    // Distinct points cheaply: a running sum of random points, normalised in one batch.
    let seeds: Vec<G1Projective> = (0..64).map(|_| G1Projective::rand(&mut rng)).collect();
    let mut acc = G1Projective::rand(&mut rng);
    let proj: Vec<G1Projective> = (0..n)
        .map(|i| {
            acc += seeds[i % 64];
            acc
        })
        .collect();
    let bases: Vec<G1Affine> = G1Projective::normalize_batch(&proj);
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let msm = CpuMsm::new();
    let want = msm.msm_g1(&bases, &scalars);
    if std::env::var("CHECK").is_ok() {
        use ark_ec::VariableBaseMSM;
        assert_eq!(want, G1Projective::msm(&bases, &scalars).unwrap(), "disagrees with arkworks");
        eprintln!("matches arkworks");
    }
    let mut ts = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        let got = msm.msm_g1(&bases, &scalars);
        ts.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(got, want);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!(
        "n=2^{log} threads={} median {:.1} ms  min {:.1}  {:.1} ns/point/thread",
        msm.threads,
        med,
        ts[0],
        med * 1e6 * msm.threads as f64 / n as f64
    );
}
