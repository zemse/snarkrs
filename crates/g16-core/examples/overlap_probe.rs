//! Throwaway probe: what would overlapping stages 0-4 with the witness MSMs buy?
//!
//! Of the five MSMs only stage 9 reads `H`; stages 5-8 read the witness, which exists
//! before `compute_h` starts. This measures three schedules on the CPU backend:
//!
//!   A  sequential, the shape `prove` has today: compute_h, then all five MSMs.
//!   B  join(compute_h, witness MSMs), then the H MSM alone.
//!   C  join(compute_h then H MSM, witness MSMs): the H MSM starts the moment H
//!      exists, and the witness MSMs fill whatever the pool has idle. This is the
//!      schedule a restructure would buy, so its delta against A is the ceiling.
//!
//! Rounds interleave A/B/C so thermal drift hits all three alike. Outputs are compared
//! across schedules each round: a probe that raced its way to a different proof would
//! be measuring a bug.
//!
//! `cargo run --release -p g16-core --example overlap_probe -- bench/artifacts/csp/sha256_128 ...`

use std::path::Path;
use std::time::Instant;

use g16_core::{cpu::CpuBackend, Backend, MsmOutputs, StageTimings};
use g16_field::{Fr, G1Projective, G2Projective};
use g16_msm::{CpuMsm, MsmBackend};
use g16_zkey::{wtns::Witness, ProvingKey};

const ROUNDS: usize = 9;

struct Outs {
    a_g1: G1Projective,
    b_g2: G2Projective,
    b_g1: G1Projective,
    l_g1: G1Projective,
    h_g1: G1Projective,
}

impl From<MsmOutputs> for Outs {
    fn from(m: MsmOutputs) -> Self {
        Outs {
            a_g1: m.a_g1,
            b_g2: m.b_g2,
            b_g1: m.b_g1,
            l_g1: m.l_g1,
            h_g1: m.h_g1,
        }
    }
}

fn witness_msms(msm: &CpuMsm, pk: &g16_zkey::ProvingKey, witness: &[Fr]) -> Outs {
    let l_scalars = &witness[pk.n_public + 1..];
    // Same nesting as the prover's, minus the H job.
    let ((a_g1, b_g2), (b_g1, l_g1)) = rayon::join(
        || {
            rayon::join(
                || msm.msm_g1(&pk.a_query, witness),
                || msm.msm_g2(&pk.b_g2_query, witness),
            )
        },
        || {
            rayon::join(
                || msm.msm_g1(&pk.b_g1_query, witness),
                || msm.msm_g1(&pk.l_query, l_scalars),
            )
        },
    );
    Outs {
        a_g1,
        b_g2,
        b_g1,
        l_g1,
        h_g1: G1Projective::default(),
    }
}

fn check(name: &str, a: &Outs, b: &Outs) {
    assert_eq!(a.a_g1, b.a_g1, "{name}: a_g1");
    assert_eq!(a.b_g2, b.b_g2, "{name}: b_g2");
    assert_eq!(a.b_g1, b.b_g1, "{name}: b_g1");
    assert_eq!(a.l_g1, b.l_g1, "{name}: l_g1");
    assert_eq!(a.h_g1, b.h_g1, "{name}: h_g1");
}

fn stats(label: &str, ms: &mut [f64]) {
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ms[ms.len() / 2];
    println!(
        "  {label}: min {:.2} ms  median {median:.2} ms  max {:.2} ms",
        ms[0],
        ms[ms.len() - 1]
    );
}

fn run(dir: &Path) {
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
    let circuit = CpuBackend::new().prepare(pk).unwrap();
    let pk = circuit.key();
    let msm = CpuMsm::new();
    println!(
        "{} (domain 2^{})",
        dir.file_name().unwrap().to_string_lossy(),
        pk.domain_size.trailing_zeros()
    );

    // Warm up: touch every page of the key once.
    let mut t = StageTimings::default();
    let h = circuit.compute_h(&witness, &mut t).unwrap();
    let _ = circuit.msms(&witness, &h, &mut t).unwrap();

    let mut ta = Vec::new();
    let mut tb = Vec::new();
    let mut tc = Vec::new();
    for _ in 0..ROUNDS {
        // A: today's schedule.
        let mut t = StageTimings::default();
        let start = Instant::now();
        let h = circuit.compute_h(&witness, &mut t).unwrap();
        let ma: Outs = circuit.msms(&witness, &h, &mut t).unwrap().into();
        ta.push(start.elapsed().as_secs_f64() * 1e3);

        // B: witness MSMs beside compute_h, H MSM after both.
        let start = Instant::now();
        let (h, mut mb) = rayon::join(
            || {
                let mut t = StageTimings::default();
                circuit.compute_h(&witness, &mut t).unwrap()
            },
            || witness_msms(&msm, pk, &witness),
        );
        mb.h_g1 = msm.msm_g1(&pk.h_query, h.to_host().unwrap());
        tb.push(start.elapsed().as_secs_f64() * 1e3);

        // C: the H MSM chained straight onto compute_h, witness MSMs alongside.
        let start = Instant::now();
        let (h_g1, mut mc) = rayon::join(
            || {
                let mut t = StageTimings::default();
                let h = circuit.compute_h(&witness, &mut t).unwrap();
                msm.msm_g1(&pk.h_query, h.to_host().unwrap())
            },
            || witness_msms(&msm, pk, &witness),
        );
        mc.h_g1 = h_g1;
        tc.push(start.elapsed().as_secs_f64() * 1e3);

        check("B", &ma, &mb);
        check("C", &ma, &mc);
    }
    stats("A sequential      ", &mut ta);
    stats("B join, then H    ", &mut tb);
    stats("C H chained on 0-4", &mut tc);
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "pass artifact directories");
    for d in &dirs {
        run(Path::new(d));
    }
}
