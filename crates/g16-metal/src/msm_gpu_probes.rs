//! Opt-in measurements for the review. None runs in the correctness suite.

use super::*;
use g16_field::{CurveGroup, PrimeField, PrimeGroup};
use g16_msm::MsmBackend;
use metal::objc::{msg_send, sel, sel_impl};
use std::time::Instant;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn review_plan_shape() {
    let n: usize = std::env::var("G16_REVIEW_N").unwrap().parse().unwrap();
    let bits: usize = std::env::var("G16_REVIEW_BITS")
        .unwrap_or_else(|_| "255".into())
        .parse()
        .unwrap();
    assert!(n > 0 && (3..=255).contains(&bits));
    let m = MetalMsm::new().unwrap();
    let mut seed = 0x1234_5678_9abc_def0u64;
    let scalars: Vec<_> = (0..n)
        .map(|i| {
            let mut bytes = [0u8; 32];
            for b in &mut bytes {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *b = seed as u8;
            }
            if bits < 255 {
                for bit in bits - 1..256 {
                    bytes[bit / 8] &= !(1 << (bit % 8));
                }
            }
            // Pin the bound with one scalar, without concentrating every scalar in
            // the high half of the top window. Full-width inputs reduce 256 random
            // bits modulo Fr, just as the existing randomized correctness tests do.
            if i == 0 {
                bytes.fill(0);
                bytes[(bits - 2) / 8] = 1 << ((bits - 2) % 8);
            }
            let s = Fr::from_le_bytes_mod_order(&bytes);
            if s.is_zero() || s.is_one() {
                Fr::from(2u64)
            } else {
                s
            }
        })
        .collect();
    let mut p = G1Projective::generator();
    let bases: Vec<_> = (0..n)
        .map(|_| {
            p += G1Projective::generator();
            p.into_affine()
        })
        .collect();
    let want = g16_msm::CpuMsm::new()
        .msm_g1(&bases, &scalars)
        .into_affine();
    let b = m.upload_g1_bases(&bases);
    let s = m.upload_scalars(&scalars);
    let plan = Plan::new(&s, 0, n);
    println!(
        "shape n={n} bits={bits} c={} w={} slice={} wide_rows={} host_tail={}",
        plan.c, plan.n_windows, plan.slice_len, plan.merge_wide_rows, plan.host_tail
    );
    let mut samples = Vec::new();
    for rep in 0..12 {
        let start = Instant::now();
        let got = m.msm_g1(&b, &s).unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(got.into_affine(), want);
        if rep >= 3 {
            samples.push(elapsed);
        }
    }
    println!("shape median_ms={:.4}", median(samples));
}

// Same bucket addresses and segment lengths as the production reduce, with three
// separately compiled bodies. Mode 0 reads all 128 bytes and emits their checksum;
// mode 1 emits Q; mode 2 emits P. The latter two have no scan or small multiplication.
// These outputs have their own CPU oracle and never enter a proof.
const REDUCE_PROBE: &str = r#"
template <uint MODE>
inline void reduce_body_probe(device const PtG1* buckets, device PtG1* out,
                              constant uint4& shape, uint tg, uint tid, uint threads) {
    uint w = tg / shape.y;
    uint g = tg % shape.y;
    uint chunk = (shape.x + shape.y - 1u) / shape.y;
    uint seg = (chunk + threads - 1u) / threads;
    uint lo = g * chunk + tid * seg;
    uint hi = min(min(lo + seg, (g + 1u) * chunk), shape.x);
    PtG1 run = pt_zero<Fq>();
    PtG1 total = pt_zero<Fq>();
    Fq checksum = fq_zero();
    for (uint j = hi; j > lo; j--) {
        PtG1 b = buckets[w * shape.x + j - 1u];
        if (MODE == 0u) {
            for (uint k = 0; k < 8u; k++) {
                checksum.v[k] ^= b.x.v[k] ^ b.y.v[k] ^ b.zz.v[k] ^ b.zzz.v[k];
            }
        } else {
            run = pt_add(run, b);
            if (MODE == 2u) { total = pt_add(total, run); }
        }
    }
    if (MODE == 0u) { total.x = checksum; }
    if (MODE == 1u) { total = run; }
    out[tg * threads + tid] = total;
}
#define PROBE_KERNEL(NAME, MODE) \
kernel void NAME(device const PtG1* buckets [[buffer(0)]], \
                 device PtG1* out [[buffer(1)]], constant uint4& shape [[buffer(2)]], \
                 uint tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_position_in_threadgroup]], \
                 uint threads [[threads_per_threadgroup]]) { \
    reduce_body_probe<MODE>(buckets, out, shape, tg, tid, threads); \
}
PROBE_KERNEL(probe_load, 0u)
PROBE_KERNEL(probe_run, 1u)
PROBE_KERNEL(probe_total, 2u)
"#;

#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn review_reduce_body() {
    let m = MetalMsm::new().unwrap();
    let library = m
        .device
        .new_library_with_source(
            &format!("{FR_MSL}\n{MSM_MSL}\n{REDUCE_PROBE}"),
            &CompileOptions::new(),
        )
        .unwrap();
    let pipelines: Vec<_> = ["probe_load", "probe_run", "probe_total"]
        .iter()
        .map(|name| {
            let f = library.get_function(name, None).unwrap();
            m.device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap()
        })
        .collect();
    for (windows, buckets) in [(20usize, 4096usize), (17, 16384)] {
        let groups = 8;
        let threads = 64;
        let rows = windows * buckets;
        let count = windows * groups * threads;
        let mut points = Vec::with_capacity(rows);
        let mut coefficients = Vec::with_capacity(rows);
        for j in 0..rows {
            // A permutation avoids consecutive equal points and predictable sums.
            let k = (j * 73 % 257 + 1) as u64;
            coefficients.push(k);
            points.push(k);
        }
        let table: Vec<_> = (1u64..=257)
            .map(|k| {
                let p = (G1Projective::generator() * Fr::from(k)).into_affine();
                PackedXyzzG1 {
                    x: PackedFq::from_fq(&p.x),
                    y: PackedFq::from_fq(&p.y),
                    zz: PackedFq::from_fq(&g16_field::Fq::one()),
                    zzz: PackedFq::from_fq(&g16_field::Fq::one()),
                }
            })
            .collect();
        let packed: Vec<_> = points.iter().map(|k| table[*k as usize - 1]).collect();
        let input = m.device.new_buffer_with_data(
            packed.as_ptr().cast(),
            (rows * 128) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output = m
            .device
            .new_buffer((count * 128) as u64, MTLResourceOptions::StorageModeShared);
        let shape = [buckets as u32, groups as u32, windows as u32, 0];
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for rep in 0..8 {
            for mode in 0..3 {
                let pso = &pipelines[mode];
                assert!(pso.max_total_threads_per_threadgroup() >= threads as u64);
                let cb = m.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&input), 0);
                enc.set_buffer(1, Some(&output), 0);
                enc.set_bytes(2, 16, shape.as_ptr().cast());
                // Repeated dispatches amortize a single submission floor. This is a
                // warm-cache diagnostic, not an estimate of end-to-end proof time.
                for _ in 0..16 {
                    enc.dispatch_thread_groups(
                        MTLSize::new((windows * groups) as u64, 1, 1),
                        MTLSize::new(threads as u64, 1, 1),
                    );
                }
                enc.end_encoding();
                cb.commit();
                crate::cb::wait_ok(cb, "reduce body probe").unwrap();
                // SAFETY: documented timestamps, read only after successful completion.
                let elapsed = unsafe {
                    let start: f64 = msg_send![cb, GPUStartTime];
                    let end: f64 = msg_send![cb, GPUEndTime];
                    (end - start) * 1e3 / 16.0
                };
                if rep >= 2 {
                    samples[mode].push(elapsed);
                }
                let got: &[PackedXyzzG1] = unsafe { read_back(&output, count) };
                let seg = buckets / groups / threads;
                for index in [0, 1, 31, 32, count / 2, count - 1] {
                    let lo = index * seg;
                    if mode == 0 {
                        let mut want = [0u32; 8];
                        for b in &packed[lo..lo + seg] {
                            for (k, x) in want.iter_mut().enumerate() {
                                *x ^= b.x.v[k] ^ b.y.v[k] ^ b.zz.v[k] ^ b.zzz.v[k];
                            }
                        }
                        assert_eq!(got[index].x.v, want);
                    } else {
                        let k: u64 = coefficients[lo..lo + seg]
                            .iter()
                            .enumerate()
                            .map(|(j, k)| k * if mode == 1 { 1 } else { (j + 1) as u64 })
                            .sum();
                        assert_eq!(
                            got[index].to_projective().into_affine(),
                            (G1Projective::generator() * Fr::from(k)).into_affine()
                        );
                    }
                }
            }
        }
        println!("reduce_body w={windows} buckets={buckets} groups=8 threads=64");
        for (mode, values) in samples.into_iter().enumerate() {
            println!(
                "  mode={mode} gpu_ms={:.4} max_threads={} simd={}",
                median(values),
                pipelines[mode].max_total_threads_per_threadgroup(),
                pipelines[mode].thread_execution_width()
            );
        }
    }
}
