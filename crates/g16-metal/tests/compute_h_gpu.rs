//! Stages 0 to 4 on the GPU against the CPU backend, element by element, on real
//! artifacts.
//!
//! This is the deliverable for the H half of the Metal backend. The CPU backend is the
//! oracle: it already matches snarkjs' own `buildABC1 -> ifft -> batchApplyKey ->
//! fft -> joinABC` vector index by index, so agreeing with it pins down the CSR gather,
//! the transform ordering, the iNTT normalisation, the coset shift and the "no division
//! by Z" convention all at once.
//!
//! This is exact integer arithmetic in a prime field. Anything short of every element
//! matching is a failure, not a tolerance.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use g16_core::{cpu::CpuBackend, Backend, HPoly, StageTimings};
use g16_field::Fr;
use g16_metal::stages::{HHandle, HStages, TAG};
use g16_zkey::{wtns::Witness, ProvingKey};

fn artifacts() -> Vec<(String, PathBuf)> {
    let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
    else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

/// Both encodings of the GPU's H, unpacked. `h_std` is validated as a canonical residue
/// on the way out, so a broken `fr_from_mont` cannot hide behind a silent reduction.
fn read_back(h: &HPoly) -> (Vec<Fr>, Vec<Fr>) {
    let handle = h
        .device_handle::<HHandle>(TAG)
        .expect("compute_h returned something other than a metal device handle");
    let std_form = handle
        .to_host_std()
        .expect("h_std contains a value that is not a canonical residue below r");
    (handle.to_host(), std_form)
}

fn compare(name: &str, label: &str, got: &[Fr], want: &[Fr]) -> usize {
    assert_eq!(got.len(), want.len(), "{name} {label}: length");
    let mut matched = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{name} {label}: mismatch at index {i}");
        matched += 1;
    }
    matched
}

#[test]
fn gpu_compute_h_matches_the_cpu_backend() {
    let found = artifacts();
    assert!(!found.is_empty(), "no artifacts under bench/artifacts");

    let stages = HStages::new().expect("Metal device");
    println!(
        "DEVICE  {}  maxFusedPasses={}  threadgroupMem={}",
        stages.device().name(),
        stages.max_fused_passes(),
        stages.device().max_threadgroup_memory_length(),
    );

    let mut total = 0usize;
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;

        // The CPU oracle, on the same witness.
        let cpu = CpuBackend::new().prepare(pk).unwrap();
        let mut tc = StageTimings::default();
        let want = cpu.compute_h(&w, &mut tc).unwrap();
        let want = want.to_host().unwrap().to_vec();

        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let res = stages.prepare(&pk).unwrap();
        // The premise the one-thread-per-row gather is built on, restated from the key
        // itself rather than quoted. Short ragged rows with many empties is what rules
        // out every cooperative mapping.
        let stats = |m: usize| -> (f64, usize, usize) {
            let rp = &pk.coeffs.row_ptr[m];
            let rows = rp.len() - 1;
            let nnz = *rp.last().unwrap() as usize;
            let empty = (0..rows).filter(|&c| rp[c] == rp[c + 1]).count();
            let longest = (0..rows)
                .map(|c| (rp[c + 1] - rp[c]) as usize)
                .max()
                .unwrap_or(0);
            (nnz as f64 / rows as f64, empty, longest)
        };
        let (ma, ea, la) = stats(0);
        let (mb, eb, lb) = stats(1);
        println!(
            "{name}: n_vars {} domain 2^{} batches {:?} dispatches {}",
            pk.n_vars,
            res.domain_size().trailing_zeros(),
            res.pass_batches(),
            res.dispatch_count(),
        );
        println!(
            "  CSR rows  A mean {ma:.2} empty {ea} longest {la}   B mean {mb:.2} empty {eb} longest {lb}"
        );

        // The production path: one command buffer, stage 4 fused into the last NTT store.
        let mut tg = StageTimings::default();
        let h = res.compute_h(&stages, &w, &mut tg).unwrap();
        assert_eq!(h.len(), res.domain_size());
        let (mont, std_form) = read_back(&h);
        let m1 = compare(&name, "fused/montgomery", &mont, &want);
        let m2 = compare(&name, "fused/standard", &std_form, &want);
        drop(h);

        // The unfused path, which runs the standalone stage 4 kernel and three separate
        // command buffers. Same answer or the fusion is wrong.
        std::env::set_var("G16_METAL_PROFILE", "1");
        let mut tp = StageTimings::default();
        let h = res.compute_h(&stages, &w, &mut tp).unwrap();
        let (mont_u, std_u) = read_back(&h);
        let m3 = compare(&name, "unfused/montgomery", &mont_u, &want);
        let m4 = compare(&name, "unfused/standard", &std_u, &want);
        drop(h);
        std::env::remove_var("G16_METAL_PROFILE");

        println!(
            "H MATCHED  {name:<14} {}/{} elements, all four encodings ({} comparisons)",
            m1,
            want.len(),
            m1 + m2 + m3 + m4
        );
        println!(
            "STAGE US   {name:<14} gpu gather {} ntt {} pointwise {}   cpu gather {} ntt {} pointwise {}",
            tp.gather_us, tp.ntt_us, tp.pointwise_us, tc.gather_us, tc.ntt_us, tc.pointwise_us
        );
        println!(
            "FUSED US   {name:<14} host pack {} gpu stages 0-4 {}",
            tg.gather_us, tg.ntt_us
        );
        total += m1 + m2 + m3 + m4;
    }
    println!("TOTAL ELEMENT COMPARISONS MATCHED  {total}");
}

/// A witness of the wrong length must be refused before anything is dispatched, and the
/// buffer pool must survive the refusal.
#[test]
fn a_short_witness_is_rejected() {
    let found = artifacts();
    let Some((_, dir)) = found.first() else {
        return;
    };
    let stages = HStages::new().expect("Metal device");
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
    let res = stages.prepare(&pk).unwrap();
    let mut t = StageTimings::default();
    assert!(res.compute_h(&stages, &w[..w.len() - 1], &mut t).is_err());
    // and the circuit still works afterwards
    assert!(res.compute_h(&stages, &w, &mut t).is_ok());
}

/// Warm timings for stages 0 to 4 alone, against the same stages on the CPU. Not a proof
/// timing: stages 5 to 9 and witness generation are not here. Reported because a kernel
/// nobody timed is not a result.
#[test]
#[ignore = "timing, run with --ignored --nocapture"]
fn gpu_compute_h_timing() {
    let stages = HStages::new().expect("Metal device");
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let res = stages.prepare(&pk).unwrap();
        let cpu_pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let cpu = CpuBackend::new().prepare(cpu_pk).unwrap();

        // The two loops are separate, not interleaved. Interleaving looks fairer and is
        // not: the CPU backend saturates all twelve cores with rayon, and a GPU rep that
        // starts while those threads are still winding down contends with them for
        // memory bandwidth and power budget. Measured, that inflated the 2^18 GPU median
        // from 6.9 ms to about 13 ms, which is a measurement of the harness.
        let mut gpu_ms = Vec::new();
        let mut pack_us = 0u64;
        let mut dev_us = 0u64;
        let mut kept = 0u64;
        for rep in 0..15 {
            let mut t = StageTimings::default();
            let s = Instant::now();
            let h = res.compute_h(&stages, &w, &mut t).unwrap();
            let dt = s.elapsed().as_secs_f64() * 1e3;
            drop(h);
            if rep >= 5 {
                gpu_ms.push(dt);
                pack_us += t.gather_us;
                dev_us += t.ntt_us;
                kept += 1;
            }
        }
        let mut cpu_ms = Vec::new();
        for rep in 0..15 {
            let mut tc = StageTimings::default();
            let s = Instant::now();
            let hc = cpu.compute_h(&w, &mut tc).unwrap();
            let dtc = s.elapsed().as_secs_f64() * 1e3;
            drop(hc);
            if rep >= 5 {
                cpu_ms.push(dtc);
            }
        }
        println!(
            "  split  host witness pack {:.3} ms  gpu command buffer {:.3} ms",
            pack_us as f64 / kept as f64 / 1e3,
            dev_us as f64 / kept as f64 / 1e3,
        );
        gpu_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cpu_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let g = gpu_ms[gpu_ms.len() / 2];
        let c = cpu_ms[cpu_ms.len() / 2];
        println!(
            "STAGES 0-4 WARM  {name:<14} domain 2^{:<2} dispatches {:<3} gpu {g:.3} ms  cpu {c:.3} ms  gpu/cpu {:.2}x",
            res.domain_size().trailing_zeros(),
            res.dispatch_count(),
            g / c
        );
    }
}

/// Does fusing stage 4 into the last NTT store actually pay?
///
/// The fused path saves one dispatch, one write of C and one read of A, B and C, but the
/// unfused path is also two extra command buffers, so the two effects have to be
/// separated by measurement rather than argued. Reps alternate so a thermal drift over
/// the run cannot be mistaken for a difference between the paths.
#[test]
#[ignore = "timing, run with --ignored --nocapture"]
fn stage_four_fusion_ab() {
    let stages = HStages::new().expect("Metal device");
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let res = stages.prepare(&pk).unwrap();

        let mut fused = Vec::new();
        let mut unfused = Vec::new();
        for rep in 0..31 {
            for (profile, out) in [(false, &mut fused), (true, &mut unfused)] {
                if profile {
                    std::env::set_var("G16_METAL_PROFILE", "1");
                } else {
                    std::env::remove_var("G16_METAL_PROFILE");
                }
                let mut t = StageTimings::default();
                let s = Instant::now();
                let h = res.compute_h(&stages, &w, &mut t).unwrap();
                let dt = s.elapsed().as_secs_f64() * 1e3;
                drop(h);
                if rep >= 5 {
                    out.push(dt);
                }
            }
        }
        std::env::remove_var("G16_METAL_PROFILE");
        fused.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unfused.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let f = fused[fused.len() / 2];
        let u = unfused[unfused.len() / 2];
        println!(
            "FUSION A/B  {name:<14} fused {f:.3} ms (p10 {:.3} p90 {:.3})  unfused+3cb {u:.3} ms  ratio {:.2}x",
            fused[fused.len() / 10],
            fused[fused.len() * 9 / 10],
            u / f
        );
    }
}
