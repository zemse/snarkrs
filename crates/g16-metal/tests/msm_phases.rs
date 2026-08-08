//! Splits the Metal MSM stage into its three submissions, to find the size-independent
//! intercept that makes the GPU lose below the crossover.
//!
//! `msms()` does three things: one `upload_scalars` (host pack + memcpy), one
//! `scalars_from_device_mont` command buffer, and one `msm_batch` command buffer holding
//! every dispatch for all five MSMs. Three commits at the measured 0.153 ms floor is
//! 0.46 ms, so if the stage costs ~7 ms on a 100-wire circuit the rest is somewhere else.

#![cfg(target_os = "macos")]

use std::time::Instant;

use g16_field::{AffineRepr, CurveGroup, Fr};
use g16_metal::msm::{Job, JobG1, JobG2, MetalMsm};
use metal::Device;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn msm_stage_intercept() {
    let msm = MetalMsm::with_device(Device::system_default().unwrap()).unwrap();

    println!(
        "{:>8}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}",
        "n", "c", "upload", "batch5", "batch1", "per-window"
    );
    for &n in &[16384usize, 24576, 32768, 49152, 65536, 131072, 262144] {
        let c = g16_metal::msm::window_size(n);
        let n_windows = 254usize.div_ceil(c as usize);

        let scalars: Vec<Fr> = (0..n).map(|i| Fr::from((i as u64) + 1)).collect();
        let g1: Vec<_> = (0..n)
            .map(|i| (g16_field::G1Affine::generator() * Fr::from((i as u64) + 3)).into_affine())
            .collect();
        let g2: Vec<_> = (0..n)
            .map(|i| (g16_field::G2Affine::generator() * Fr::from((i as u64) + 5)).into_affine())
            .collect();
        let b1 = msm.upload_g1_bases(&g1);
        let b2 = msm.upload_g2_bases(&g2);

        // warm
        for _ in 0..5 {
            let s = msm.upload_scalars(&scalars);
            let _ = msm
                .msm_batch(&[Job::G1(JobG1 {
                    bases: &b1,
                    base_off: 0,
                    scalars: &s,
                    scalar_off: 0,
                    n,
                })])
                .unwrap();
        }

        let mut up = Vec::new();
        let mut b5 = Vec::new();
        let mut b1t = Vec::new();
        for _ in 0..30 {
            let t = Instant::now();
            let s = msm.upload_scalars(&scalars);
            up.push(ms(t));

            let t = Instant::now();
            let _ = msm
                .msm_batch(&[Job::G1(JobG1 {
                    bases: &b1,
                    base_off: 0,
                    scalars: &s,
                    scalar_off: 0,
                    n,
                })])
                .unwrap();
            b1t.push(ms(t));

            let t = Instant::now();
            let _ = msm
                .msm_batch(&[
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G2(JobG2 {
                        bases: &b2,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                ])
                .unwrap();
            b5.push(ms(t));
        }
        println!(
            "{n:>8}  {c:>8}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9} ",
            median(up),
            median(b5),
            median(b1t),
            n_windows
        );
    }
}
