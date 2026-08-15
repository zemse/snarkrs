//! Where the ~7 ms fixed cost in the Metal MSM stage goes.
//!
//! `g16 bench` shows `msm` flat at roughly 7.5 ms from domain 128 to domain 4096, which
//! is ~50x the 0.153 ms command-buffer dispatch floor, so it is neither the submission
//! overhead nor the input size. This separates three candidates:
//!
//!   * back-to-back submission with no host work between reps (isolates GPU clock ramp
//!     and idle-to-active latency from the pairing verify the bench loop does per rep),
//!   * `compute_h` alone vs `msms` alone,
//!   * the host-side Horner combination over `n_windows` points, which scales with the
//!     window count and not with the input.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use g16_core::{cpu::CpuBackend, Backend, StageTimings};
use g16_metal::MetalBackend;
use g16_zkey::{wtns::Witness, ProvingKey};

fn dir_for(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .join(name)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_one(name: &str, dir: PathBuf) {
    if !dir.join("circuit.zkey").is_file() {
        println!("MSM-FLOOR {name}: no artifact, skipped");
        return;
    }
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let domain = pk.domain_size;
    let n_vars = pk.n_vars;
    let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;

    let metal = MetalBackend::new().unwrap();
    let c = metal.prepare(pk).unwrap();

    // Warmup: first few submissions after process start are not representative.
    for _ in 0..10 {
        let mut t = StageTimings::default();
        let h = c.compute_h(&w, &mut t).unwrap();
        let _ = c.msms(&w, &h, &mut t).unwrap();
    }

    // Back-to-back, nothing between reps.
    let mut h_ms = Vec::new();
    let mut m_ms = Vec::new();
    let mut all_ms = Vec::new();
    for _ in 0..100 {
        let mut t = StageTimings::default();
        let t0 = Instant::now();
        let h = c.compute_h(&w, &mut t).unwrap();
        let t1 = Instant::now();
        let _o = c.msms(&w, &h, &mut t).unwrap();
        let t2 = Instant::now();
        h_ms.push((t1 - t0).as_secs_f64() * 1e3);
        m_ms.push((t2 - t1).as_secs_f64() * 1e3);
        all_ms.push((t2 - t0).as_secs_f64() * 1e3);
    }

    // Same again, but with a chunk of host work between reps, the way the bench loop's
    // pairing verify sits between proofs. If the GPU is dropping to a low power state
    // between submissions, this is where it shows.
    let cpu = CpuBackend::default();
    let pk2 = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let ccpu = cpu.prepare(pk2).unwrap();
    let mut gapped = Vec::new();
    for _ in 0..40 {
        let mut tg = StageTimings::default();
        // ~1-3 ms of pure host work, no GPU touched.
        let _ = ccpu.compute_h(&w, &mut tg).unwrap();
        let mut t = StageTimings::default();
        let t0 = Instant::now();
        let h = c.compute_h(&w, &mut t).unwrap();
        let _o = c.msms(&w, &h, &mut t).unwrap();
        gapped.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    println!(
        "MSM-FLOOR {name:<14} domain {domain:>7} n_vars {n_vars:>7}  \
         compute_h {:6.2}  msms {:6.2}  total {:6.2}  |  with host gap {:6.2} ms",
        median(h_ms),
        median(m_ms),
        median(all_ms),
        median(gapped),
    );
}

#[test]
fn where_the_fixed_msm_cost_goes() {
    for name in [
        "tiny_mul",
        "js_1x1_d8",
        "js_2x2_d16",
        "js_2x2_d32",
        "js_8x8_d32",
        "js_16x16_d32",
    ] {
        bench_one(name, dir_for(name));
    }
}
