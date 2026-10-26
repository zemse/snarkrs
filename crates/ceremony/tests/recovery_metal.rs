//! Metal ceremony outputs through GPU interruptions, held against the CPU file.
//!
//! A completion token says a command buffer ran its last dispatch; it does not say the
//! stores before it landed. Metal has been seen to report buffers `Completed` with their
//! token after a hang and recovery while the values were wrong, so a retried or
//! recovered submission is only checked here by what it wrote: every successful Metal
//! output must be byte identical to the CPU output for the same inputs and pinned RNG,
//! every failed one must leave the existing destination byte identical with no stage
//! behind, and `zkey verify` through the Metal MSM must accept the CPU-made final key.
//!
//! The test causes no fault itself. A round that sees no interruption checks only the
//! unfaulted path, so the run prints how many operations failed and fails under
//! `G16_REQUIRE_TESTS=1` if none did. Interruptions come from GPU recovery episodes,
//! which a second process proving at 2^22 in a loop has produced (BUG-28); retries
//! inside the backend succeed silently and are what the byte comparison is for.
//!
//! `#[ignore]` because it is a soak that wants the GPU to itself apart from the
//! deliberate load. `G16_RECOVERY_ROUNDS` sets the round count (default 50) and
//! `G16_RECOVERY_POWER` the ptau power (default 10, at most 16).

#![cfg(all(feature = "metal", target_os = "macos"))]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

mod common;

use std::path::{Path, PathBuf};

use snarkrs_ceremony::write::BinFileWriter;
use snarkrs_ceremony::{
    bellman, challenge, contribute, phase1, prepare, setup, transcript, CeremonyError,
    ContributionParams, CpuGroupFft, CpuKeyScale,
};
use snarkrs_field::{Fr, PrimeField};
use snarkrs_metal::{MetalGroupFft, MetalKeyScale, MetalMsmBackend};
use snarkrs_msm::{CpuMsm, GroupFft, KeyScale, MsmBackend};

const SENTINEL: &[u8] = b"existing destination";

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v} is not a count")),
        Err(_) => default,
    }
}

/// `x * 1 = y` repeated `m` times, so every slot of the key has `m` terms and setup's
/// MSMs go to the backend rather than the small-slot path.
fn write_r1cs(path: &Path, m: u32) {
    let mut w = BinFileWriter::create(path, b"r1cs", 1, 3).unwrap();
    w.start_section(1).unwrap();
    w.write_u32(32).unwrap();
    for limb in Fr::MODULUS.as_ref() {
        w.write_u64(*limb).unwrap();
    }
    for value in [3, 1, 0, 1] {
        w.write_u32(value).unwrap();
    }
    w.write_u64(3).unwrap();
    w.write_u32(m).unwrap();
    w.end_section().unwrap();
    w.start_section(2).unwrap();
    for _ in 0..m {
        for signal in [2, 0, 1] {
            w.write_u32(1).unwrap();
            w.write_u32(signal).unwrap();
            w.write_fr_plain(&Fr::from(1u64)).unwrap();
        }
    }
    w.end_section().unwrap();
    w.start_section(3).unwrap();
    for label in 0..3 {
        w.write_u64(label).unwrap();
    }
    w.end_section().unwrap();
    w.finish().unwrap();
}

fn rng(tag: &str) -> transcript::CeremonyRng {
    transcript::rng_from_entropy_with(&[5; 64], tag)
}

struct Backends<'a> {
    key: &'a dyn KeyScale,
    fft: &'a dyn GroupFft,
    msm: &'a dyn MsmBackend,
}

/// Inputs every operation reads, all made on the CPU once.
struct Inputs {
    dir: PathBuf,
    r1cs: PathBuf,
    raw: PathBuf,
    contributed: PathBuf,
    prepared: PathBuf,
    init: PathBuf,
    challenge: PathBuf,
    bellman: PathBuf,
}

impl Inputs {
    fn new(test: &str, power: u32) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "snarkrs-recovery-metal-{test}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let r1cs = dir.join("circuit.r1cs");
        // With its two public signals the circuit's domain is then 2^(power - 1), inside
        // the ptau, and every slot is over setup's MSM crossover from power 10 up.
        write_r1cs(&r1cs, (1 << (power - 1)) - 2);
        let raw = dir.join("raw.ptau");
        let contributed = dir.join("contributed.ptau");
        let prepared = dir.join("prepared.ptau");
        let init = dir.join("init.zkey");
        let challenge = dir.join("challenge");
        let bellman = dir.join("bellman");
        phase1::ptau_new(power, &raw).unwrap();
        phase1::contribute_with(
            &raw,
            &contributed,
            ContributionParams::default(),
            rng("inputs phase1"),
            &CpuKeyScale,
        )
        .unwrap();
        prepare::prepare_phase2(&contributed, &prepared, &CpuGroupFft).unwrap();
        setup::setup(&r1cs, &prepared, &init, &CpuMsm::new()).unwrap();
        challenge::export_challenge(&contributed, &challenge).unwrap();
        bellman::export_bellman(&init, &bellman).unwrap();
        Self {
            dir,
            r1cs,
            raw,
            contributed,
            prepared,
            init,
            challenge,
            bellman,
        }
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

const OPERATIONS: [&str; 8] = [
    "ptau contribute",
    "ptau beacon",
    "prepare phase2",
    "zkey setup",
    "zkey contribute",
    "zkey beacon",
    "challenge contribute",
    "bellman contribute",
];

fn run(op: usize, i: &Inputs, out: &Path, b: &Backends) -> Result<(), CeremonyError> {
    let beacon = common::beacon_bytes();
    let params = ContributionParams::default;
    match op {
        0 => phase1::contribute_with(&i.contributed, out, params(), rng("ptau"), b.key).map(|_| ()),
        1 => phase1::beacon(&i.raw, out, None, &beacon, 10, b.key).map(|_| ()),
        2 => prepare::prepare_phase2(&i.contributed, out, b.fft),
        3 => setup::setup(&i.r1cs, &i.prepared, out, b.msm).map(|_| ()),
        4 => contribute::contribute_with(&i.init, out, None, rng("zkey"), b.key).map(|_| ()),
        5 => contribute::beacon(&i.init, out, None, &beacon, 10, b.key).map(|_| ()),
        6 => challenge::challenge_contribute_with(&i.challenge, out, rng("challenge"), b.key)
            .map(|_| ()),
        7 => bellman::bellman_contribute_with(&i.bellman, out, rng("bellman"), b.key).map(|_| ()),
        _ => unreachable!(),
    }
}

fn entries(dir: &Path) -> Vec<std::ffi::OsString> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    entries.sort();
    entries
}

/// The soak's reference is one CPU run per operation, which is only a reference if a
/// second run with the same pinned RNG gives the same bytes. CPU only.
#[test]
fn cpu_reference_outputs_are_reproducible() {
    let inputs = Inputs::new("reproducible", 10);
    let cpu_msm = CpuMsm::new();
    let cpu = Backends {
        key: &CpuKeyScale,
        fft: &CpuGroupFft,
        msm: &cpu_msm,
    };
    for (op, name) in OPERATIONS.iter().enumerate() {
        let [a, b] = ["a", "b"].map(|run_tag| {
            let out = inputs.dir.join(format!("{run_tag}-{op}"));
            run(op, &inputs, &out, &cpu).unwrap();
            std::fs::read(out).unwrap()
        });
        assert_eq!(a, b, "{name}");
    }
}

#[test]
#[ignore = "GPU recovery soak: run beside a 2^22 proof loop, see the module docs"]
fn interrupted_metal_ceremony_outputs_match_cpu_or_preserve_destination() {
    let rounds = env_usize("G16_RECOVERY_ROUNDS", 50);
    let power = env_usize("G16_RECOVERY_POWER", 10) as u32;
    assert!((4..=16).contains(&power), "G16_RECOVERY_POWER={power}");
    let (key, fft, msm) = match (
        MetalKeyScale::new(),
        MetalGroupFft::new(),
        MetalMsmBackend::new(),
    ) {
        (Ok(key), Ok(fft), Ok(msm)) => (key, fft, msm),
        _ => {
            resources::skip("no Metal device");
            return;
        }
    };
    // At the shipped crossovers inputs this small would stay on the host.
    let metal = Backends {
        key: &key.with_min_points(1),
        fft: &fft.with_min_block(1),
        msm: &msm,
    };
    let cpu_msm = CpuMsm::new();
    let cpu = Backends {
        key: &CpuKeyScale,
        fft: &CpuGroupFft,
        msm: &cpu_msm,
    };

    let inputs = Inputs::new("soak", power);
    let expected: Vec<Vec<u8>> = (0..OPERATIONS.len())
        .map(|op| {
            let out = inputs.dir.join(format!("cpu-{op}"));
            run(op, &inputs, &out, &cpu).unwrap();
            std::fs::read(out).unwrap()
        })
        .collect();
    let final_key = inputs.dir.join("cpu-4");

    let out_dir = inputs.dir.join("metal");
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut failed = vec![0usize; OPERATIONS.len() + 1];
    let mut problems = Vec::new();
    for round in 0..rounds {
        for (op, name) in OPERATIONS.iter().enumerate() {
            let out = out_dir.join(format!("out-{op}"));
            std::fs::write(&out, SENTINEL).unwrap();
            let before = entries(&out_dir);
            match run(op, &inputs, &out, &metal) {
                Ok(()) => {
                    if std::fs::read(&out).unwrap() != expected[op] {
                        problems.push(format!("round {round} {name}: Ok but differs from CPU"));
                    }
                }
                Err(CeremonyError::Accel(e)) => {
                    failed[op] += 1;
                    eprintln!("round {round} {name}: {e}");
                    if std::fs::read(&out).unwrap() != SENTINEL {
                        problems.push(format!("round {round} {name}: failed and replaced"));
                    }
                }
                Err(e) => problems.push(format!("round {round} {name}: non-device error {e}")),
            }
            if entries(&out_dir) != before {
                problems.push(format!("round {round} {name}: directory changed"));
            }
        }
        match contribute::verify_from_init(&inputs.init, &inputs.prepared, &final_key, metal.msm) {
            Ok(_) => {}
            Err(CeremonyError::Accel(e)) => {
                failed[OPERATIONS.len()] += 1;
                eprintln!("round {round} zkey verify: {e}");
            }
            Err(e) => problems.push(format!(
                "round {round} zkey verify: rejected a good key: {e}"
            )),
        }
    }

    let total: usize = failed.iter().sum();
    eprintln!(
        "{rounds} rounds at power {power}: {} device failures ({})",
        total,
        OPERATIONS
            .iter()
            .chain(["zkey verify"].iter())
            .zip(&failed)
            .map(|(name, n)| format!("{name} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    if total == 0 {
        resources::skip("no device failure observed, so no interrupted submission was checked");
    }
}
