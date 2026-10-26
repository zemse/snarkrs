//! CUDA `ptau prepare` through injected wrong stores, held against the CPU file, and its
//! warm wall time.
//!
//! CUDA has no completion token: a stream that drained is all a readback has, and a store
//! the device did not make looks exactly like one it did. `injected_*` puts
//! `snarkrs_cuda::inject`'s one fault, a wrong store, behind every readback of the group
//! FFT in turn over a whole `prepare phase2`: every `Ok` must be byte identical to the CPU
//! output for the same input, and a persistent fault must fail the command with the
//! existing destination byte identical and no stage behind. An `Ok` with other bytes
//! fails the test. What refuses the wrong store is the value checks on every readback
//! (`crates/cuda/src/check.rs`): the probe of one thread per pass and the scan of every
//! finished block. The injector is per thread and `prepare` submits a section's device
//! blocks from the thread that called it, so the test arms on its own thread. It needs no
//! quiet machine, only an NVIDIA device, so it runs with the rest of the `cuda` feature's
//! tests, and skips loudly without one.
//!
//! The group FFT is the one CUDA ceremony seam; the KeyScale and MSM operations the Metal
//! soak (`recovery_metal.rs`) covers have no CUDA kernels. `G16_RECOVERY_POWER` sets the
//! ptau power (default 10, at most 16).

#![cfg(feature = "cuda")]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use common::beacon_bytes;
use snarkrs_ceremony::{phase1, prepare, CeremonyError, CpuGroupFft, CpuKeyScale};
use snarkrs_cuda::inject::{self, Fault};
use snarkrs_cuda::CudaGroupFft;

const SENTINEL: &[u8] = b"existing destination";

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v} is not a count")),
        Err(_) => default,
    }
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-recovery-cuda-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The CUDA backend at a given crossover, or `None` on a machine with no device.
fn cuda(min_block: Option<usize>) -> Option<CudaGroupFft> {
    match CudaGroupFft::new() {
        Ok(k) => Some(match min_block {
            Some(n) => k.with_min_block(n),
            None => k,
        }),
        Err(e) => {
            resources::skip(format_args!("no CUDA backend: {e}"));
            None
        }
    }
}

/// A beaconed power-`power` file under `dir`: `ptau new` writes `tau = 1`, so the beacon
/// is what gives every section points other than the generator.
fn beaconed_input(dir: &Path, power: u32) -> PathBuf {
    let raw = dir.join("raw.ptau");
    phase1::ptau_new(power, &raw).unwrap();
    let input = dir.join("beacon.ptau");
    phase1::beacon(&raw, &input, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
    input
}

#[test]
fn injected_cuda_wrong_stores_match_cpu_or_preserve_destination() {
    let power = env_usize("G16_RECOVERY_POWER", 10) as u32;
    assert!((4..=16).contains(&power), "G16_RECOVERY_POWER={power}");
    // Crossover 1, so every block of every section is the device's and every readback
    // is one of the FFT's.
    let Some(fft) = cuda(Some(1)) else { return };
    let dir = tmp_dir("injected");
    let input = beaconed_input(&dir, power);
    let cpu = dir.join("cpu.ptau");
    prepare::prepare_phase2(&input, &cpu, &CpuGroupFft).unwrap();
    let want = std::fs::read(&cpu).unwrap();
    let known = ["raw.ptau", "beacon.ptau", "cpu.ptau", "out.ptau"];

    let out = dir.join("out.ptau");
    let run = || {
        std::fs::write(&out, SENTINEL).unwrap();
        prepare::prepare_phase2(&input, &out, &fft)
    };

    inject::arm(None);
    run().unwrap();
    let readbacks = inject::calls();
    assert!(readbacks > 0, "no readback reached the device");
    assert_eq!(std::fs::read(&out).unwrap(), want, "unfaulted");

    // Once-wrong: refused, re-run, the CPU's bytes.
    let (mut retried, mut escaped) = (0u32, 0u32);
    for at in 0..readbacks {
        inject::arm_with(Some((at, Fault::WrongStore)), false);
        let result = run();
        assert!(inject::fired(), "readback {at} was not reached");
        match result {
            Ok(()) if std::fs::read(&out).unwrap() == want => retried += 1,
            Ok(()) => {
                escaped += 1;
                eprintln!("readback {at}: a wrong store reached the file");
            }
            Err(e) => panic!("readback {at}: a once-wrong store failed the command: {e}"),
        }
    }
    inject::arm(None);

    // Wrong on every attempt: the command fails, the destination and the directory are
    // as they were.
    let persistent = [0, readbacks / 2, readbacks - 1];
    for at in persistent {
        inject::arm_with(Some((at, Fault::WrongStore)), true);
        let err = run().expect_err("a persistent wrong store was accepted");
        assert!(inject::fired(), "readback {at} was not reached");
        assert!(
            matches!(err, CeremonyError::Accel(_)),
            "readback {at}: {err}"
        );
        assert!(
            err.to_string()
                .contains("a value read back is not one the GPU wrote"),
            "readback {at}: {err}"
        );
        assert_eq!(
            std::fs::read(&out).unwrap(),
            SENTINEL,
            "readback {at}: the destination was touched"
        );
        let stray: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| !known.contains(&n.as_str()))
            .collect();
        assert!(stray.is_empty(), "readback {at}: left behind: {stray:?}");
    }
    inject::arm(None);

    eprintln!(
        "power {power}: {readbacks} readbacks, WrongStore {retried} retried, {escaped} escaped, \
         {} persistent failed with the destination untouched",
        persistent.len()
    );
    assert_eq!(escaped, 0, "{escaped} wrong stores reached the file");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Warm wall time of `prepare phase2` on CUDA at the shipped crossover, for pricing a
/// change to the backend; no check of its own. `G16_RECOVERY_ROUNDS` rounds (default 5)
/// over a beaconed file at `G16_RECOVERY_POWER` (default 16), one line per run, every
/// round printed so the first, which compiles the kernels or warms their cache, can be
/// dropped. With `G16_TIMING_DIR` set the input is built there once and reused by the next
/// process, which is how two builds of the backend are run turn and turn about on the same
/// file: the test binary of each, alternated by a shell loop.
#[test]
#[ignore = "timing, not a check: see its doc comment"]
fn time_cuda_prepare() {
    let rounds = env_usize("G16_RECOVERY_ROUNDS", 5);
    let power = env_usize("G16_RECOVERY_POWER", 16) as u32;
    assert!((4..=20).contains(&power), "G16_RECOVERY_POWER={power}");
    let Some(fft) = cuda(None) else { return };
    let keep = std::env::var_os("G16_TIMING_DIR");
    let dir = match &keep {
        Some(d) => PathBuf::from(d).join(format!("power-{power}")),
        None => tmp_dir("timing"),
    };
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("beacon.ptau");
    if !input.is_file() {
        beaconed_input(&dir, power);
    }
    let out = dir.join(format!("out-{}.ptau", std::process::id()));
    for round in 0..rounds {
        let _ = std::fs::remove_file(&out);
        let t = Instant::now();
        prepare::prepare_phase2(&input, &out, &fft).unwrap();
        eprintln!(
            "timing power {power} round {round}: prepare phase2: {:.3} s",
            t.elapsed().as_secs_f64()
        );
    }
    let _ = std::fs::remove_file(&out);
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
