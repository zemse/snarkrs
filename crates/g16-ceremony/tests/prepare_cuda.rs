//! `--backend cpu` against `--backend cuda` for `ptau prepare`.
//!
//! The CUDA twin of `tests/prepare_metal.rs`, and the same bar: byte identity of the
//! whole output file. The CPU path is already byte-identical to snarkjs 0.7.6 (pinned in
//! `tests/prepare.rs`), so cpu-against-cuda equality carries the snarkjs claim across
//! without running snarkjs at all.
//!
//! Three settings instead of that file's four. The CUDA host has no ladder budget to
//! shrink, because the budget exists to bound what a macOS interactivity kill throws
//! away and there is no such kill on a headless card, so the `split` pass has nothing to
//! split and does not exist here. `device` at `min_block` 1 and `host fallback` at
//! `usize::MAX` are exactly as there.
//!
//! Every test skips loudly when there is no NVIDIA device, the same way the CUDA prover
//! suites do: this file is compiled by a feature, and the feature says "test the CUDA
//! path", not "this machine has a GPU".

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use g16_ceremony::{phase1, prepare, CpuGroupFft, CpuKeyScale};
use g16_cuda::CudaGroupFft;
use g16_field::{AffineRepr, G1Affine};

/// `01 02 .. 20`, the beacon the rest of the ceremony suite uses.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-prepare-cuda-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bench_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .join(rel)
}

/// The CUDA backend at a given crossover, or `None` on a machine with no device.
fn cuda(min_block: Option<usize>) -> Option<CudaGroupFft> {
    match CudaGroupFft::new() {
        Ok(k) => Some(match min_block {
            Some(n) => k.with_min_block(n),
            None => k,
        }),
        Err(e) => {
            eprintln!("SKIPPED: no CUDA backend: {e}");
            None
        }
    }
}

/// The shipped crossover, one that forces even a two-point block onto the device, and
/// one that sends every block home, the only case that runs `prepare::ifft_block`'s
/// fallback arm (prepare.rs:352) on a backend that is not `CpuGroupFft`.
const CROSSOVERS: [(Option<usize>, &str); 3] = [
    (None, "shipped"),
    (Some(1), "device"),
    (Some(usize::MAX), "host fallback"),
];

fn assert_same_bytes(what: &str, cpu: &Path, gpu: &Path) {
    let a = std::fs::read(cpu).unwrap();
    let b = std::fs::read(gpu).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: cpu wrote {} bytes, cuda wrote {}",
        a.len(),
        b.len()
    );
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{what}: first difference at byte {i}, cpu 0x{:02x} against cuda 0x{:02x}",
            a[i], b[i]
        );
    }
}

/// Run both backends over one file and diff the results, printing both wall clocks so a
/// run of this suite is also the cheapest speedup measurement there is.
fn compare(dir: &Path, name: &str, label: &str, ptau: &Path, fft: &CudaGroupFft) {
    let cpu = dir.join(format!("{name}.{label}.cpu.ptau"));
    let gpu = dir.join(format!("{name}.{label}.cuda.ptau"));

    let t = Instant::now();
    prepare::prepare_phase2(ptau, &cpu, &CpuGroupFft).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    prepare::prepare_phase2(ptau, &gpu, fft).unwrap();
    let gpu_s = t.elapsed().as_secs_f64();

    eprintln!(
        "prepare {name} ({label}): cpu {cpu_s:.2}s, cuda {gpu_s:.2}s, {:.2}x",
        cpu_s / gpu_s
    );
    assert_same_bytes(&format!("prepare {name} ({label})"), &cpu, &gpu);
}

/// A power-10 file straight out of `ptau new`, one with a real tau, and snarkjs' own
/// power-13 file where the checkout carries it. Same three inputs and the same reasoning
/// as the Metal twin: `ptau new` writes `tau = 1`, so the fresh file is the densest
/// point-at-infinity coverage any input in this repo gives, and the beacon output is the
/// opposite.
fn inputs(dir: &Path) -> Vec<(String, PathBuf)> {
    let fresh = dir.join("new_10.ptau");
    phase1::ptau_new(10, &fresh).unwrap();
    let beaconed = dir.join("beacon_10.ptau");
    phase1::beacon(&fresh, &beaconed, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();

    let mut out = vec![
        ("new_10".to_owned(), fresh),
        ("beacon_10".to_owned(), beaconed),
    ];
    let local = bench_path("ptau/local_13.ptau");
    if local.is_file() {
        out.push(("local_13".to_owned(), local));
    }
    out
}

#[test]
fn prepare_is_byte_identical_across_backends() {
    let dir = tmp_dir("prepare");
    let inputs = inputs(&dir);
    for (min_block, label) in CROSSOVERS {
        let Some(fft) = cuda(min_block) else {
            return;
        };
        for (name, ptau) in &inputs {
            compare(&dir, name, label, ptau, &fft);
        }
    }
}

/// The powers that matter, off by default because each is a full CPU prepare. Run with
/// `cargo test --release -p g16-ceremony --features cuda -- --ignored` on a box that has
/// `bench/ptau/ppot_0080_15.ptau` and `_16` staged.
#[test]
#[ignore = "minutes: two full CPU prepares at powers 15 and 16"]
fn prepare_is_byte_identical_at_the_powers_that_matter() {
    let dir = tmp_dir("prepare-big");
    let Some(fft) = cuda(None) else {
        return;
    };
    let mut ran = 0;
    for power in [15u32, 16] {
        let ptau = bench_path(&format!("ptau/ppot_0080_{power}.ptau"));
        if !ptau.is_file() {
            eprintln!("SKIPPED power {power}: {} is not there", ptau.display());
            continue;
        }
        compare(&dir, &format!("ppot_{power}"), "shipped", &ptau, &fft);
        ran += 1;
    }
    assert!(ran > 0, "no ppot fixtures under bench/ptau");
}

/// Section 12's `power+1` block, whose last input slot is the point at infinity
/// (`powersoftau_preparephase2.js:78-81`, mirrored at prepare.rs:562), through the seam
/// rather than through a whole file, so a failure names the block instead of a byte
/// offset in a 1.2 GB diff.
#[test]
fn the_identity_padding_survives_the_device_transform() {
    let Some(fft) = cuda(Some(1)) else {
        return;
    };
    for bits in [1u32, 4, 12, 13] {
        let n = 1usize << bits;
        let mut points: Vec<G1Affine> = Vec::with_capacity(n);
        let mut k = g16_field::Fr::from(7u64);
        for _ in 0..n - 1 {
            k *= g16_field::Fr::from(11u64);
            points.push((G1Affine::generator() * k).into());
        }
        // The padding snarkjs writes into the last slot, and the only place in the file
        // where an input point is the identity.
        points.push(G1Affine::identity());

        let want = prepare::lagrange_evaluations_g1(&points, &CpuGroupFft).unwrap();
        let got = prepare::lagrange_evaluations_g1(&points, &fft).unwrap();
        assert_eq!(got, want, "2^{bits} block with identity padding");
    }
}
