//! `--backend cpu` against `--backend metal` for `ptau prepare`.
//!
//! The bar is byte identity of the whole output file, not "the proof verifies" and not
//! "snarkjs accepts it". The CPU path is already byte-identical to snarkjs 0.7.6 (pinned
//! in `tests/prepare.rs`), so cpu-against-metal equality carries the snarkjs claim across
//! without running snarkjs at all; the `snarkjs powersoftau verify` test at the bottom is
//! a second opinion on top of that, and it is the only check here that pairs sections 12
//! to 15 back against section 2 rather than trusting anything either backend computed.
//!
//! Nothing has to be pinned to make the two runs comparable. `prepare` is a pure function
//! of its input file: no entropy, no beacon, no RNG. So every test below is a plain diff
//! of two outputs.
//!
//! Four settings, and only one of them is the shipped one. `MetalGroupFft::min_block` is
//! 4,096, so at the shipped setting a power-10 file, whose largest block is 2^11, never
//! reaches the device and the test would be comparing the CPU against itself; hence the
//! `device` pass at `min_block` 1. `host fallback` at `usize::MAX` is not redundant
//! either, being the only case that runs `prepare::ifft_block`'s fallback arm
//! (prepare.rs:352) on a backend that is not `CpuGroupFft`.
//!
//! The fourth crosses the command-buffer boundary. The backend runs one pass per command
//! buffer and splits a pass larger than its ladder budget across several with a `gid_off`;
//! the shipped budget is 2^19 ladders on G1, which only section 12's top block at power 20
//! and above reaches, so on any file a test can afford `gid_off` would always be zero. The
//! `split` pass shrinks the budget to 300 and makes a power-10 file cross it hundreds of
//! times instead.
//!
//! Block sizes are the other boundary and a power-13 input crosses the crossover on its
//! own: it holds every block from 2^0 to 2^14, three of them above the shipped threshold
//! and eleven below.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use g16_ceremony::{phase1, prepare, CpuGroupFft, CpuKeyScale};
use g16_field::{AffineRepr, G1Affine};
use g16_metal::MetalGroupFft;

/// `01 02 .. 20`, the beacon the rest of the ceremony suite uses.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-prepare-metal-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bench_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .join(rel)
}

/// The Metal backend at a given crossover, or `None` on a machine with no device.
///
/// A skip rather than a failure: this file is compiled by a feature, and the feature says
/// "test the Metal path", not "this machine has a GPU". Every caller prints why.
fn metal(min_block: Option<usize>, budget: Option<usize>) -> Option<MetalGroupFft> {
    match MetalGroupFft::new() {
        Ok(k) => {
            let k = match min_block {
                Some(n) => k.with_min_block(n),
                None => k,
            };
            Some(match budget {
                Some(b) => k.with_budget(b),
                None => k,
            })
        }
        Err(e) => {
            eprintln!("SKIPPED: no Metal backend: {e}");
            None
        }
    }
}

/// The shipped crossover, one that forces even a two-point block onto the device, one that
/// sends every block home, and one that splits every pass across command buffers.
const CROSSOVERS: [(Option<usize>, Option<usize>, &str); 4] = [
    (None, None, "shipped"),
    (Some(1), None, "device"),
    (Some(usize::MAX), None, "host fallback"),
    (Some(1), Some(300), "split"),
];

fn assert_same_bytes(what: &str, cpu: &Path, gpu: &Path) {
    let a = std::fs::read(cpu).unwrap();
    let b = std::fs::read(gpu).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: cpu wrote {} bytes, metal wrote {}",
        a.len(),
        b.len()
    );
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{what}: first difference at byte {i}, cpu 0x{:02x} against metal 0x{:02x}",
            a[i], b[i]
        );
    }
}

/// Run both backends over one file and diff the results, printing both wall clocks so a
/// run of this suite is also the cheapest speedup measurement there is.
fn compare(dir: &Path, name: &str, label: &str, ptau: &Path, fft: &MetalGroupFft) {
    let cpu = dir.join(format!("{name}.{label}.cpu.ptau"));
    let gpu = dir.join(format!("{name}.{label}.metal.ptau"));

    let t = Instant::now();
    prepare::prepare_phase2(ptau, &cpu, &CpuGroupFft).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    prepare::prepare_phase2(ptau, &gpu, fft).unwrap();
    let gpu_s = t.elapsed().as_secs_f64();

    eprintln!(
        "prepare {name} ({label}): cpu {cpu_s:.2}s, metal {gpu_s:.2}s, {:.2}x",
        cpu_s / gpu_s
    );
    assert_same_bytes(&format!("prepare {name} ({label})"), &cpu, &gpu);
}

/// A power-10 file straight out of `ptau new`, one with a real tau, and snarkjs' own
/// power-13 file.
///
/// The first is not redundant with the second and is the more interesting of the two.
/// `ptau new` writes `tau = 1`, so every stored point is the generator and the transform
/// produces the point at infinity for most outputs: it is the densest infinity coverage
/// any input in this repo gives, and infinity is a live path here, not a defensive one.
/// The beacon output is the opposite, a file where essentially nothing is the identity, so
/// a ladder that is exact on infinity and wrong on a real point fails on it. `local_13`
/// then adds real prior contributions and the block sizes that straddle the crossover.
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
    for (min_block, budget, label) in CROSSOVERS {
        let Some(fft) = metal(min_block, budget) else {
            return;
        };
        for (name, ptau) in &inputs {
            // `local_13` at a budget of 300 is 200,000 command buffers, which is a
            // different test from the one this pass is for: the split arithmetic is the
            // same at every block size and the power-10 files already cross it.
            if budget.is_some() && name == "local_13" {
                continue;
            }
            compare(&dir, name, label, ptau, &fft);
        }
    }
}

/// The two powers the milestone is actually about, and the reason they are not in the
/// suite by default: a power-16 `prepare` is 48 s of CPU on a quiet machine and 85 s on a
/// busy one, so this pair is minutes rather than seconds. Run it with
/// `cargo test --release -p g16-ceremony --features metal -- --ignored`.
///
/// Power 16 is the size worth the wait. It exercises every block from 2^0 to 2^17, both
/// groups, and section 12's identity padding, on a file that is a real ceremony's output
/// rather than anything this repo generated.
#[test]
#[ignore = "minutes: two full CPU prepares at powers 15 and 16"]
fn prepare_is_byte_identical_at_the_powers_that_matter() {
    let dir = tmp_dir("prepare-big");
    let Some(fft) = metal(None, None) else {
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
/// rather than through a whole file.
///
/// The file-level tests above already cover it, but only as one block in eighty-five: if
/// this is the thing that breaks, a diff of a 1.2 GB file says byte 4,981,232 differs and
/// nothing about why. This says which block.
#[test]
fn the_identity_padding_survives_the_device_transform() {
    let Some(fft) = metal(Some(1), None) else {
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

// --- snarkjs, the second opinion ---

fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    Command::new(&bin).arg("--help").output().ok().map(|_| bin)
}

/// `powersoftau verify` on a Metal-prepared file.
///
/// This is the only check in the file that is not a diff. It pairs the Lagrange sections
/// back against section 2 and against the contribution transcript, so it would catch a
/// transform that both backends got wrong in the same way, which no cpu-against-metal
/// comparison can.
#[test]
fn snarkjs_verifies_a_metal_prepared_ptau() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED snarkjs_verifies_a_metal_prepared_ptau: no snarkjs on PATH");
        return;
    };
    let Some(fft) = metal(Some(1), None) else {
        return;
    };
    let ptau = bench_path("ptau/local_13.ptau");
    if !ptau.is_file() {
        eprintln!("SKIPPED snarkjs_verifies_a_metal_prepared_ptau: no local_13.ptau");
        return;
    }
    let dir = tmp_dir("snarkjs");
    let out = dir.join("local_13.metal.ptau");
    prepare::prepare_phase2(&ptau, &out, &fft).unwrap();

    let res = Command::new(&bin)
        .args(["powersoftau", "verify", out.to_str().unwrap()])
        .output()
        .unwrap_or_else(|e| panic!("running snarkjs powersoftau verify: {e}"));
    let mut log = String::from_utf8_lossy(&res.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&res.stderr));
    assert!(
        res.status.success(),
        "snarkjs powersoftau verify failed:\n{log}"
    );
}
