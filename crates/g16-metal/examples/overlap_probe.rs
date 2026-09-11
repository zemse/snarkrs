//! Throwaway probe: what would overlapping stages 0-4 with the witness MSMs buy on Metal?
//!
//! Only stage 9 reads `H`; stages 5-8 read the witness. `HStages` and `MetalMsm` own
//! separate command queues, so a second thread can submit the four witness MSMs while
//! the compute_h command buffers run, and the device is free to interleave them. The
//! cost of asking is that the batch splits in two: the witness jobs lose their seat in
//! the concurrent encoder next to H's accumulation, and a second submission is paid.
//!
//!   A  sequential, the shape `prove` has today: compute_h, then all five in one batch.
//!   C  thread 1 runs compute_h then the H MSM alone; thread 2 runs the witness batch.
//!
//! Rounds interleave A/C so thermal drift hits both alike, and outputs are compared
//! each round. Run with `G16_METAL_CB_TIMES=1` to see whether the driver's GPU windows
//! actually overlapped: if the split helps, additivity is exactly what should break.
//!
//! `cargo run --release -p g16-metal --example overlap_probe -- bench/artifacts/csp/sha256_128 ...`

#[cfg(target_os = "macos")]
mod probe {
    use std::path::Path;
    use std::time::Instant;

    use g16_core::StageTimings;
    use g16_metal::msm::{G1Bases, G2Bases, Job, JobG1, JobG2, MetalMsm, MsmResult};
    use g16_metal::stages::{HResident, HStages};
    use g16_zkey::{wtns::Witness, ProvingKey};
    use metal::Device;

    const ROUNDS: usize = 9;

    struct Bases {
        a: G1Bases,
        b_g2: G2Bases,
        b_g1: G1Bases,
        l: G1Bases,
        h: G1Bases,
    }

    fn witness_jobs<'a>(
        pk: &ProvingKey,
        bases: &'a Bases,
        w: &'a g16_metal::msm::ScalarBuf,
    ) -> [Job<'a>; 4] {
        [
            Job::G1(JobG1 {
                bases: &bases.a,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: pk.n_vars,
            }),
            Job::G2(JobG2 {
                bases: &bases.b_g2,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: pk.n_vars,
            }),
            Job::G1(JobG1 {
                bases: &bases.b_g1,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: pk.n_vars,
            }),
            Job::G1(JobG1 {
                bases: &bases.l,
                base_off: 0,
                scalars: w,
                scalar_off: pk.n_public + 1,
                n: bases.l.len(),
            }),
        ]
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

    fn run_dir(stages: &HStages, msm: &MetalMsm, dir: &Path) {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let resident: HResident = stages.prepare(&pk).unwrap();
        let bases = Bases {
            a: msm.upload_g1_bases(&pk.a_query),
            b_g2: msm.upload_g2_bases(&pk.b_g2_query),
            b_g1: msm.upload_g1_bases(&pk.b_g1_query),
            l: msm.upload_g1_bases(&pk.l_query),
            h: msm.upload_g1_bases(&pk.h_query),
        };
        println!(
            "{} (domain 2^{})",
            dir.file_name().unwrap().to_string_lossy(),
            pk.domain_size.trailing_zeros()
        );

        // Warm up both kernel sets and the scratch pools.
        let mut t = StageTimings::default();
        let h = resident.compute_h(stages, &witness, &mut t).unwrap();
        drop(h);

        let mut ta = Vec::new();
        let mut tc = Vec::new();
        for _ in 0..ROUNDS {
            // A: today's schedule, one batch of five.
            let start = Instant::now();
            let mut t = StageTimings::default();
            let h = resident.compute_h(stages, &witness, &mut t).unwrap();
            let handle = h
                .device_handle::<g16_metal::stages::HHandle>(g16_metal::stages::TAG)
                .unwrap();
            let w = msm.upload_scalars(&witness);
            let h_scalars = msm.scalars_from_device_std(handle.h_std(), handle.len());
            let [j0, j1, j2, j3] = witness_jobs(&pk, &bases, &w);
            let jobs = [
                j0,
                j1,
                j2,
                j3,
                Job::G1(JobG1 {
                    bases: &bases.h,
                    base_off: 0,
                    scalars: &h_scalars,
                    scalar_off: 0,
                    n: pk.domain_size,
                }),
            ];
            let out_a = msm.msm_batch(&jobs).unwrap();
            drop(h);
            ta.push(start.elapsed().as_secs_f64() * 1e3);

            // C: witness batch on its own queue beside compute_h, H batch after.
            let start = Instant::now();
            let (h_out, wit_out) = std::thread::scope(|s| {
                let wit = s.spawn(|| {
                    let w = msm.upload_scalars(&witness);
                    let jobs = witness_jobs(&pk, &bases, &w);
                    msm.msm_batch(&jobs).unwrap()
                });
                let mut t = StageTimings::default();
                let h = resident.compute_h(stages, &witness, &mut t).unwrap();
                let handle = h
                    .device_handle::<g16_metal::stages::HHandle>(g16_metal::stages::TAG)
                    .unwrap();
                let h_scalars = msm.scalars_from_device_std(handle.h_std(), handle.len());
                let jobs = [Job::G1(JobG1 {
                    bases: &bases.h,
                    base_off: 0,
                    scalars: &h_scalars,
                    scalar_off: 0,
                    n: pk.domain_size,
                })];
                let out = msm.msm_batch(&jobs).unwrap();
                (out, wit.join().unwrap())
            });
            tc.push(start.elapsed().as_secs_f64() * 1e3);

            // PartialEq on the projective points, not on their coordinates: two
            // representations of the same point differ coordinate-wise.
            let same = |a: &MsmResult, b: &MsmResult| match (a, b) {
                (MsmResult::G1(x), MsmResult::G1(y)) => x == y,
                (MsmResult::G2(x), MsmResult::G2(y)) => x == y,
                _ => false,
            };
            for i in 0..4 {
                assert!(same(&out_a[i], &wit_out[i]), "witness msm {i} differs");
            }
            assert!(same(&out_a[4], &h_out[0]), "h msm differs");
        }
        stats("A sequential ", &mut ta);
        stats("C overlapped ", &mut tc);
    }

    pub fn run() {
        let dirs: Vec<_> = std::env::args_os().skip(1).collect();
        assert!(!dirs.is_empty(), "pass one or more circuit directories");
        let device = Device::system_default().expect("no Metal device");
        let stages = HStages::with_device(device.clone()).unwrap();
        let msm = MetalMsm::with_device(device).unwrap();
        for dir in dirs {
            run_dir(&stages, &msm, Path::new(&dir));
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    probe::run();
    #[cfg(not(target_os = "macos"))]
    panic!("this probe needs a Metal device");
}
