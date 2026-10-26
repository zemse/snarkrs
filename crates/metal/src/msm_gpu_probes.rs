//! Opt-in GPU measurements of the plan shape, the reduce and scatter kernel bodies, and
//! what a first command buffer costs. Every one is `#[ignore]`d, so none runs in the
//! correctness suite.

use super::*;
use objc2_metal::{MTLBuffer, MTLCommandQueue};
use snarkrs_field::{CurveGroup, PrimeField, PrimeGroup};
use snarkrs_msm::MsmBackend;
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
    let want = snarkrs_msm::CpuMsm::new()
        .msm_g1(&bases, &scalars)
        .into_affine();
    let b = m.upload_g1_bases(&bases).expect("upload bases");
    let s = m.upload_scalars(&scalars).expect("upload scalars");
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
    let library = mtl::library(&m.device, &format!("{FR_MSL}\n{MSM_MSL}\n{REDUCE_PROBE}")).unwrap();
    let pipelines: Vec<_> = ["probe_load", "probe_run", "probe_total"]
        .iter()
        .map(|name| {
            let f = mtl::function(&library, name).unwrap();
            mtl::pipeline(&m.device, &f).unwrap()
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
                    zz: PackedFq::from_fq(&snarkrs_field::Fq::one()),
                    zzz: PackedFq::from_fq(&snarkrs_field::Fq::one()),
                }
            })
            .collect();
        let packed: Vec<_> = points.iter().map(|k| table[*k as usize - 1]).collect();
        let input =
            crate::alloc::shared_with_data(&m.device, packed.as_ptr().cast(), rows * 128).unwrap();
        let output = crate::alloc::shared(&m.device, count * 128).unwrap();
        let shape = [buckets as u32, groups as u32, windows as u32, 0];
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for rep in 0..8 {
            for mode in 0..3 {
                let pso = &pipelines[mode];
                assert!(pso.maxTotalThreadsPerThreadgroup() >= threads);
                let queue = m.queue.get();
                let cb = queue.commandBuffer().unwrap();
                let enc = mtl::encoder(&cb);
                enc.setComputePipelineState(pso);
                mtl::set_buffer(&enc, 0, &input, 0);
                mtl::set_buffer(&enc, 1, &output, 0);
                set_params(&enc, 2, &shape);
                // Repeated dispatches amortize a single submission floor. This is a
                // warm-cache diagnostic, not an estimate of end-to-end proof time.
                for _ in 0..16 {
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        mtl::size(windows * groups),
                        mtl::size(threads),
                    );
                }
                enc.endEncoding();
                cb.commit();
                crate::cb::wait_ok(&cb, "reduce body probe").unwrap();
                // Read only after successful completion.
                let elapsed = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 / 16.0;
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
                pipelines[mode].maxTotalThreadsPerThreadgroup(),
                pipelines[mode].threadExecutionWidth()
            );
        }
    }
}

// A 32-byte-per-thread read, one checksum store: enough traffic to expose page
// mapping, too little arithmetic to hide it.
const TOUCH_PROBE: &str = r#"
kernel void probe_touch(device const uint* in [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    uint acc = 0u;
    for (uint i = 0; i < 8u; i++) { acc ^= in[gid * 8u + i]; }
    out[gid] = acc;
}
"#;

// What the first command buffer of a batch pays for a scalar buffer in each of four
// states: fresh allocation vs reused, CPU-dirtied vs untouched. The witness upload is
// a fresh `new_buffer` plus a full CPU write every proof, so its first reader is the
// candidate for the first-plan surcharge.
#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn lane_first_cb_cost() {
    let m = MetalMsm::new().unwrap();
    let library = mtl::library(&m.device, TOUCH_PROBE).unwrap();
    let f = mtl::function(&library, "probe_touch").unwrap();
    let pso = mtl::pipeline(&m.device, &f).unwrap();
    let n: usize = 131072;
    let bytes = n * 32;
    let out = crate::alloc::shared(&m.device, n * 4).unwrap();
    let reused = crate::alloc::shared(&m.device, bytes).unwrap();
    let run = |buf: &Buffer| -> (f64, f64) {
        let queue = m.queue.get();
        let cb = queue.commandBuffer().unwrap();
        let enc = mtl::encoder(&cb);
        enc.setComputePipelineState(&pso);
        mtl::set_buffer(&enc, 0, buf, 0);
        mtl::set_buffer(&enc, 1, &out, 0);
        let len = n as u32;
        set_params(&enc, 2, &len);
        dispatch_1d(&enc, &pso, n, 64);
        enc.endEncoding();
        let t = Instant::now();
        cb.commit();
        crate::cb::wait_ok(&cb, "touch probe").unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        // Read only after successful completion.
        let gpu = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
        (wall, gpu)
    };
    let cpu_write = |buf: &Buffer| -> f64 {
        let t = Instant::now();
        // SAFETY: shared-mode buffer of `bytes` bytes, no dispatch in flight.
        unsafe {
            core::ptr::write_bytes(buf.contents().cast::<u8>().as_ptr(), 0x5a, bytes);
        }
        t.elapsed().as_secs_f64() * 1e3
    };
    for _ in 0..3 {
        run(&reused);
    }
    #[allow(clippy::type_complexity)]
    let mut rows: Vec<(&str, Vec<f64>, Vec<f64>, Vec<f64>)> =
        ["fresh+write", "reused+write", "reused", "fresh"]
            .into_iter()
            .map(|k| (k, Vec::new(), Vec::new(), Vec::new()))
            .collect();
    for _ in 0..10 {
        for (kind, walls, gpus, writes) in rows.iter_mut() {
            let fresh;
            let buf = match *kind {
                "reused+write" | "reused" => &reused,
                _ => {
                    fresh = crate::alloc::shared(&m.device, bytes).unwrap();
                    &fresh
                }
            };
            if kind.ends_with("write") {
                writes.push(cpu_write(buf));
            }
            let (w, g) = run(buf);
            walls.push(w);
            gpus.push(g);
        }
    }
    for (kind, walls, gpus, writes) in rows {
        let wr = if writes.is_empty() {
            "-".into()
        } else {
            format!("{:.3}", median(writes))
        };
        println!(
            "first_cb {kind:13} wall_ms={:.3} gpu_ms={:.3} cpu_write_ms={wr}",
            median(walls),
            median(gpus),
        );
    }
}

// The scatter's cost, split by suspect. Every mode reads and recodes the scalars the
// way the production kernel does; they differ only in what happens per digit. Mode 0
// is the production body. Mode 1 keeps the atomic but stores densely, so the store no
// longer waits on the atomic's result. Mode 2 drops the atomic. Mode 3 keeps the
// atomic and its result but drops the scattered store. Mode 4 is digit compute alone.
// Mode 5 is the production body with a 4-byte entry, to see how much of the scattered
// store is footprint. Dense and 4-byte modes write wrong entries by design; nothing
// here enters a proof.
const SCATTER_PROBE: &str = r#"
template <uint MODE>
inline void scatter_probe(device const uint* scalars, device atomic_uint* cursor,
                          device uint2* entries, constant MsmParams& p, uint gid) {
    if (gid >= p.n) { return; }
    uint s[8];
    uint base = (p.scalar_off + gid) * 8u;
    for (uint i = 0; i < 8u; i++) { s[i] = scalars[base + i]; }
    if (sc_is_zero(s) || (p.separate_ones != 0u && sc_is_one(s))) { return; }
    uint acc = 0u;
    for (uint w = 0; w < p.n_windows; w++) {
        uint mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) { continue; }
        uint row = w * p.n_buckets + (mag - 1u);
        uint payload = (gid << 1) | (neg ? 1u : 0u);
        if (MODE == 4u) { acc ^= row; continue; }
        if (MODE == 2u) { entries[w * p.n + gid] = uint2(row, payload); continue; }
        uint slot = atomic_fetch_add_explicit(&cursor[row], 1u, memory_order_relaxed);
        if (MODE == 1u) { entries[w * p.n + gid] = uint2(row, payload); }
        else if (MODE == 3u) { acc ^= slot; }
        else if (MODE == 5u) { ((device uint*)entries)[slot] = payload; }
        else { entries[slot] = uint2(row, payload); }
    }
    if (MODE == 3u || MODE == 4u) { entries[gid] = uint2(acc, 0u); }
}
#define SCATTER_PROBE_KERNEL(NAME, MODE) \
kernel void NAME(device const uint* scalars [[buffer(0)]], \
                 device atomic_uint* cursor [[buffer(1)]], \
                 device uint2* entries [[buffer(2)]], \
                 constant MsmParams& p [[buffer(3)]], \
                 uint gid [[thread_position_in_grid]]) { \
    scatter_probe<MODE>(scalars, cursor, entries, p, gid); \
}
SCATTER_PROBE_KERNEL(probe_scatter_prod, 0u)
SCATTER_PROBE_KERNEL(probe_scatter_dense, 1u)
SCATTER_PROBE_KERNEL(probe_scatter_noatomic, 2u)
SCATTER_PROBE_KERNEL(probe_scatter_nostore, 3u)
SCATTER_PROBE_KERNEL(probe_scatter_digits, 4u)
SCATTER_PROBE_KERNEL(probe_scatter_u32, 5u)

// The production body over a window subrange. Serially dispatching disjoint ranges
// shrinks the set of concurrently filling bucket rows, which is one cache line each.
kernel void probe_scatter_span(device const uint* scalars [[buffer(0)]],
                               device atomic_uint* cursor [[buffer(1)]],
                               device uint2* entries [[buffer(2)]],
                               constant MsmParams& p [[buffer(3)]],
                               constant uint2& span [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    uint s[8];
    uint base = (p.scalar_off + gid) * 8u;
    for (uint i = 0; i < 8u; i++) { s[i] = scalars[base + i]; }
    if (sc_is_zero(s) || (p.separate_ones != 0u && sc_is_one(s))) { return; }
    for (uint w = span.x; w < span.y; w++) {
        uint mag;
        bool neg;
        sc_signed_digit(s, w, p.c, mag, neg);
        if (mag == 0u) { continue; }
        uint row = w * p.n_buckets + (mag - 1u);
        uint slot = atomic_fetch_add_explicit(&cursor[row], 1u, memory_order_relaxed);
        entries[slot] = uint2(row, (gid << 1) | (neg ? 1u : 0u));
    }
}
"#;

#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn lane_scatter_body() {
    let m = MetalMsm::new().unwrap();
    let library =
        mtl::library(&m.device, &format!("{FR_MSL}\n{MSM_MSL}\n{SCATTER_PROBE}")).unwrap();
    let names = [
        "probe_scatter_prod",
        "probe_scatter_dense",
        "probe_scatter_noatomic",
        "probe_scatter_nostore",
        "probe_scatter_digits",
        "probe_scatter_u32",
    ];
    let mut pipelines: Vec<_> = names
        .iter()
        .map(|name| {
            let f = mtl::function(&library, name).unwrap();
            mtl::pipeline(&m.device, &f).unwrap()
        })
        .collect();
    // The shipped pipeline on the same inputs, to anchor the probe against the
    // production phase numbers.
    pipelines.push(m.pipelines.scatter.clone());
    let span_pso = {
        let f = mtl::function(&library, "probe_scatter_span").unwrap();
        mtl::pipeline(&m.device, &f).unwrap()
    };
    let mut labels: Vec<&str> = names.to_vec();
    labels.push("production msm_scatter");
    for n in [131072usize, 524288] {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let scalars: Vec<_> = (0..n)
            .map(|_| {
                let mut bytes = [0u8; 32];
                for b in &mut bytes {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *b = seed as u8;
                }
                Fr::from_le_bytes_mod_order(&bytes)
            })
            .collect();
        let s = m.upload_scalars(&scalars).expect("upload scalars");
        let mut plan = Plan::new(&s, 0, n);
        let mut keep = Vec::new();
        plan.alloc(&m.pool, &mut keep).expect("plan scratch");
        println!(
            "scatter_body n={n} c={} w={} buckets={} cap={}",
            plan.c, plan.n_windows, plan.n_buckets, plan.cap
        );
        let params = plan.params();
        // Short single-dispatch command buffers measured 3x slow at this size: the
        // clock sags between submissions. Eight rounds per command buffer keep the
        // device busy; a prep-only buffer of the same rounds is subtracted, so a
        // round's scatter is (timed - prep) / 8. The prep inside the timed buffer
        // also hands every scatter a freshly scanned cursor.
        const ROUNDS: usize = 8;
        let time_cb = |encode_scatter: Option<&ComputePipelineState>| -> f64 {
            let queue = m.queue.get();
            let cb = queue.commandBuffer().unwrap();
            let enc = mtl::encoder(&cb);
            for _ in 0..ROUNDS {
                plan.encode_zero(&m, &enc);
                plan.encode_count(&m, &enc);
                plan.encode_scan(&m, &enc);
                if let Some(pso) = encode_scatter {
                    enc.setComputePipelineState(pso);
                    mtl::set_buffer(&enc, 0, plan.scalars, 0);
                    mtl::set_buffer(&enc, 1, plan.cursor.as_ref().unwrap(), 0);
                    mtl::set_buffer(&enc, 2, plan.entries.as_ref().unwrap(), 0);
                    set_params(&enc, 3, &params);
                    // The production kernel takes a window span; the probe bodies
                    // ignore this slot.
                    let span = [0u32, plan.n_windows as u32];
                    set_params(&enc, 4, &span);
                    dispatch_1d(&enc, pso, n, 64);
                }
            }
            enc.endEncoding();
            cb.commit();
            crate::cb::wait_ok(&cb, "scatter probe").unwrap();
            // Read only after successful completion.
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        };
        // The span variant, dispatched over `parts` serial window ranges per round.
        let time_span = |parts: usize| -> f64 {
            let queue = m.queue.get();
            let cb = queue.commandBuffer().unwrap();
            let enc = mtl::encoder(&cb);
            for _ in 0..ROUNDS {
                plan.encode_zero(&m, &enc);
                plan.encode_count(&m, &enc);
                plan.encode_scan(&m, &enc);
                let step = plan.n_windows.div_ceil(parts);
                let mut lo = 0usize;
                while lo < plan.n_windows {
                    let hi = (lo + step).min(plan.n_windows);
                    enc.setComputePipelineState(&span_pso);
                    mtl::set_buffer(&enc, 0, plan.scalars, 0);
                    mtl::set_buffer(&enc, 1, plan.cursor.as_ref().unwrap(), 0);
                    mtl::set_buffer(&enc, 2, plan.entries.as_ref().unwrap(), 0);
                    set_params(&enc, 3, &params);
                    let span = [lo as u32, hi as u32];
                    set_params(&enc, 4, &span);
                    dispatch_1d(&enc, &span_pso, n, 64);
                    lo = hi;
                }
            }
            enc.endEncoding();
            cb.commit();
            crate::cb::wait_ok(&cb, "scatter span probe").unwrap();
            // Read only after successful completion.
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        };
        let span_parts = [2usize, 3, 4];
        let mut prep_samples = Vec::new();
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); pipelines.len()];
        let mut span_samples: Vec<Vec<f64>> = vec![Vec::new(); span_parts.len()];
        for rep in 0..6 {
            let prep = time_cb(None);
            if rep >= 2 {
                prep_samples.push(prep / ROUNDS as f64);
            }
            for (mode, pso) in pipelines.iter().enumerate() {
                let t = time_cb(Some(pso));
                if rep >= 2 {
                    samples[mode].push((t - prep) / ROUNDS as f64);
                }
            }
            for (i, parts) in span_parts.iter().enumerate() {
                let t = time_span(*parts);
                if rep >= 2 {
                    span_samples[i].push((t - prep) / ROUNDS as f64);
                }
            }
        }
        println!("  prep(zero+count+scan) gpu_ms={:.4}", median(prep_samples));
        for (label, values) in labels.iter().zip(samples) {
            println!("  {label:24} gpu_ms={:.4}", median(values));
        }
        for (parts, values) in span_parts.iter().zip(span_samples) {
            println!(
                "  span parts={parts}            gpu_ms={:.4}",
                median(values)
            );
        }
        m.pool.give(keep);
    }
}

/// What `Work::Constant` leaves to bucket occupancy, phase by phase: one constant-work
/// MSM in each group over the same bases, scalars all zero, bits, and dense random,
/// each phase in its own command buffer. `G16_REVIEW_N` sets the length (2^18 by
/// default, H's length on keccak256 and js_16x16_d32).
#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn constant_work_phase_occupancy() {
    let n: usize = std::env::var("G16_REVIEW_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let m = MetalMsm::new().unwrap();
    let mut p1 = G1Projective::generator();
    let mut p2 = G2Projective::generator();
    let (mut g1, mut g2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        p1 += G1Projective::generator();
        p2 += G2Projective::generator();
        g1.push(p1);
        g2.push(p2);
    }
    let b1 = m
        .upload_g1_bases(&G1Projective::normalize_batch(&g1))
        .unwrap();
    let b2 = m
        .upload_g2_bases(&G2Projective::normalize_batch(&g2))
        .unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let bits: Vec<Fr> = (0..n).map(|_| Fr::from(next() & 1)).collect();
    let dense: Vec<Fr> = (0..n)
        .map(|_| {
            let mut bytes = [0u8; 32];
            for b in &mut bytes {
                *b = next() as u8;
            }
            Fr::from_le_bytes_mod_order(&bytes)
        })
        .collect();
    let sets = [
        ("zeros", vec![Fr::from(0u64); n]),
        ("bits", bits),
        ("dense", dense),
    ];
    let time = |f: &mut dyn FnMut(&ComputeCommandEncoderRef)| -> f64 {
        let t = Instant::now();
        let queue = m.queue.get();
        let cb = queue.commandBuffer().unwrap();
        let enc = mtl::encoder(&cb);
        f(&enc);
        enc.endEncoding();
        cb.commit();
        crate::cb::wait_ok(&cb, "occupancy probe").unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    for group in ["g1", "g2"] {
        for (label, scalars) in &sets {
            let s = m.upload_scalars_with(scalars, Work::Constant).unwrap();
            let job = if group == "g1" {
                Job::G1(JobG1 {
                    bases: &b1,
                    base_off: 0,
                    scalars: &s,
                    scalar_off: 0,
                    n,
                })
            } else {
                Job::G2(JobG2 {
                    bases: &b2,
                    base_off: 0,
                    scalars: &s,
                    scalar_off: 0,
                    n,
                })
            };
            let mut plan = Plan::new(&s, 0, n);
            let mut keep = Vec::new();
            plan.alloc(&m.pool, &mut keep).unwrap();
            let out = Outputs::alloc(&m, &mut keep, &job, &plan, 0).unwrap();
            let mut t: [Vec<f64>; 5] = Default::default();
            for rep in 0..12 {
                let d = time(&mut |enc| plan.encode(&m, enc));
                let c = time(&mut |enc| out.encode_clear(&m, enc, &job, &plan));
                let a = time(&mut |enc| out.encode_accumulate(&m, enc, &job, &plan));
                let g = time(&mut |enc| out.encode_merge(&m, enc, &job, &plan));
                let r = time(&mut |enc| out.encode_reduce(&m, enc, &job, &plan));
                if rep >= 3 {
                    for (v, x) in t.iter_mut().zip([d, c, a, g, r]) {
                        v.push(x);
                    }
                }
            }
            let [d, c, a, g, r] = t.map(median);
            println!(
                "occupancy {group} {label:<5} n={n} c={} w={} digits {d:.2} clear {c:.2} \
                 accum {a:.2} merge {g:.2} reduce {r:.2} ms",
                plan.c, plan.n_windows
            );
            drop(out);
            drop(plan);
            m.pool.give(keep);
        }
    }
}

/// The constant-work MSM's phases across scalar classes at one `n`, interleaved so a
/// drift in the machine lands on every class alike: zeros, ones, bits, dense random,
/// one dense value everywhere (every entry of a window in one bucket) and small values
/// (window zero only). Every rep times each class's digit pipeline, clear,
/// accumulation, fold and reduce in a command buffer of its own; medians and ranges
/// over the reps after the warm-up. `G16_REVIEW_N` sets the length (2^18 by default)
/// and `G16_REVIEW_REPS` the reps (12).
#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn constant_work_phase_flatness() {
    let n: usize = std::env::var("G16_REVIEW_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let reps: usize = std::env::var("G16_REVIEW_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let warm = 2;
    let m = MetalMsm::new().unwrap();
    // Every seventh base at infinity, as the tests' `bases_with_infinity`.
    let mut p1 = G1Projective::generator();
    let mut p2 = G2Projective::generator();
    let (mut g1, mut g2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        p1 += G1Projective::generator();
        p2 += G2Projective::generator();
        g1.push(p1);
        g2.push(p2);
    }
    let mut g1 = G1Projective::normalize_batch(&g1);
    let mut g2 = G2Projective::normalize_batch(&g2);
    for i in (3..n).step_by(7) {
        g1[i] = G1Affine::identity();
        g2[i] = G2Affine::identity();
    }
    let b1 = m.upload_g1_bases(&g1).unwrap();
    let b2 = m.upload_g2_bases(&g2).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let bits: Vec<Fr> = (0..n).map(|_| Fr::from(next() & 1)).collect();
    let dense: Vec<Fr> = (0..n)
        .map(|_| {
            let mut bytes = [0u8; 32];
            for b in &mut bytes {
                *b = next() as u8;
            }
            Fr::from_le_bytes_mod_order(&bytes)
        })
        .collect();
    let one_value = -Fr::from(0x7f3a_19c5_d02b_6e41u64);
    let sets = [
        ("zeros", vec![Fr::from(0u64); n]),
        ("ones", vec![Fr::from(1u64); n]),
        ("bits", bits),
        ("dense", dense),
        ("onevalue", vec![one_value; n]),
        ("small", (0..n).map(|i| Fr::from((i % 5) as u64)).collect()),
    ];
    let time = |f: &mut dyn FnMut(&ComputeCommandEncoderRef)| -> f64 {
        let t = Instant::now();
        let queue = m.queue.get();
        let cb = queue.commandBuffer().unwrap();
        let enc = mtl::encoder(&cb);
        f(&enc);
        enc.endEncoding();
        cb.commit();
        crate::cb::wait_ok(&cb, "flatness probe").unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    let phases = ["digits", "clear", "accum", "fold", "reduce"];
    for group in ["g1", "g2"] {
        let scalars: Vec<ScalarBuf> = sets
            .iter()
            .map(|(_, s)| m.upload_scalars_with(s, Work::Constant).unwrap())
            .collect();
        let jobs: Vec<Job<'_>> = scalars
            .iter()
            .map(|s| {
                if group == "g1" {
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: s,
                        scalar_off: 0,
                        n,
                    })
                } else {
                    Job::G2(JobG2 {
                        bases: &b2,
                        base_off: 0,
                        scalars: s,
                        scalar_off: 0,
                        n,
                    })
                }
            })
            .collect();
        let mut keep = Vec::new();
        let mut plans: Vec<Plan<'_>> = scalars.iter().map(|s| Plan::new(s, 0, n)).collect();
        for p in &mut plans {
            p.alloc(&m.pool, &mut keep).unwrap();
        }
        let outs: Vec<Outputs> = jobs
            .iter()
            .zip(&plans)
            .map(|(job, plan)| Outputs::alloc(&m, &mut keep, job, plan, 0).unwrap())
            .collect();
        let mut t: Vec<[Vec<f64>; 5]> = sets.iter().map(|_| Default::default()).collect();
        for rep in 0..warm + reps {
            for (i, ((job, plan), out)) in jobs.iter().zip(&plans).zip(&outs).enumerate() {
                let d = time(&mut |enc| plan.encode(&m, enc));
                let c = time(&mut |enc| out.encode_clear(&m, enc, job, plan));
                let a = time(&mut |enc| out.encode_accumulate(&m, enc, job, plan));
                let g = time(&mut |enc| out.encode_merge(&m, enc, job, plan));
                let r = time(&mut |enc| out.encode_reduce(&m, enc, job, plan));
                if rep >= warm {
                    for (v, x) in t[i].iter_mut().zip([d, c, a, g, r]) {
                        v.push(x);
                    }
                }
            }
        }
        println!(
            "flatness {group} n={n} c={} w={} slices/window={} reps={reps} (median, min-max, ms)",
            plans[0].c, plans[0].n_windows, plans[0].slices
        );
        for (i, (label, _)) in sets.iter().enumerate() {
            let mut line = format!("  {label:<9}");
            for (name, v) in phases.iter().zip(&t[i]) {
                let mut s = v.clone();
                s.sort_by(f64::total_cmp);
                line += &format!(
                    "  {name} {:.2} ({:.2}-{:.2})",
                    median(v.clone()),
                    s[0],
                    s[s.len() - 1]
                );
            }
            println!("{line}");
        }
        drop(outs);
        drop(plans);
        m.pool.give(keep);
    }
}
