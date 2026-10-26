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
//! Two soaks. `interrupted_*` causes no fault itself. A round that sees no interruption
//! checks only the unfaulted path, so the run prints how many operations failed and
//! fails under `G16_REQUIRE_TESTS=1` if none did. Interruptions come from GPU recovery
//! episodes, which a second process proving at 2^22 in a loop has produced (BUG-28);
//! retries inside the backend succeed silently and are what the byte comparison is for.
//!
//! `injected_*` puts each of `snarkrs_metal::inject`'s faults behind one submission or
//! dispatch of every operation in turn: a refused status, an ignored queue, a stale and
//! a missing token, a dispatch cut short with the rest of its buffer or of its attempt
//! dropped, and a wrong store under a good token. All seven are retried inside the
//! backend and the output must be the CPU bytes, or the operation fails with the
//! destination untouched; an `Ok` with other bytes fails the test. The wrong store is
//! the BUG-28 mechanism no token sees, and what refuses it is the value checks on every
//! ceremony readback (`snarkrs_metal::check`): the MSM's on-curve check for `zkey setup`
//! and `zkey verify`, the ladder, prefix and finish checks for the KeyScale operations,
//! the probe and the final scan for the group FFT. The injector is per thread and `zkey
//! setup` sends its slot MSMs from the rayon pool, so the whole run is installed on a
//! one-thread pool. It needs no quiet machine, only a Metal device, so it runs with the
//! rest of the `metal` feature's tests; about two minutes at power 10.
//!
//! `interrupted_*` is `#[ignore]`: a soak that wants the GPU to itself apart from the
//! deliberate load. `G16_RECOVERY_ROUNDS` sets the round count (default 50 interrupted,
//! 4 injected, where round `r` arms index `r` of each operation's submissions and
//! dispatches) and `G16_RECOVERY_POWER` the ptau power (default 10, at most 16).

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
use snarkrs_metal::inject::{self, Fault, Rest};
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
    /// Left on disk when dropped, for the next process to reuse (`Inputs::reuse`).
    keep: bool,
}

impl Inputs {
    fn new(test: &str, power: u32) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "snarkrs-recovery-metal-{test}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        Self::build(dir, power, false)
    }

    /// The inputs under `dir`, built there if `dir` does not hold them yet and kept
    /// afterwards, so two test binaries can time the same files turn and turn about.
    fn reuse(dir: PathBuf, power: u32) -> Self {
        if dir.join("bellman").is_file() {
            Self::paths(dir, true)
        } else {
            Self::build(dir, power, true)
        }
    }

    fn paths(dir: PathBuf, keep: bool) -> Self {
        Self {
            r1cs: dir.join("circuit.r1cs"),
            raw: dir.join("raw.ptau"),
            contributed: dir.join("contributed.ptau"),
            prepared: dir.join("prepared.ptau"),
            init: dir.join("init.zkey"),
            challenge: dir.join("challenge"),
            bellman: dir.join("bellman"),
            dir,
            keep,
        }
    }

    fn build(dir: PathBuf, power: u32, keep: bool) -> Self {
        std::fs::create_dir_all(&dir).unwrap();
        let i = Self::paths(dir, keep);
        // With its two public signals the circuit's domain is then 2^(power - 1), inside
        // the ptau, and every slot is over setup's MSM crossover from power 10 up.
        write_r1cs(&i.r1cs, (1 << (power - 1)) - 2);
        phase1::ptau_new(power, &i.raw).unwrap();
        phase1::contribute_with(
            &i.raw,
            &i.contributed,
            ContributionParams::default(),
            rng("inputs phase1"),
            &CpuKeyScale,
        )
        .unwrap();
        prepare::prepare_phase2(&i.contributed, &i.prepared, &CpuGroupFft).unwrap();
        setup::setup(&i.r1cs, &i.prepared, &i.init, &CpuMsm::new()).unwrap();
        challenge::export_challenge(&i.contributed, &i.challenge).unwrap();
        bellman::export_bellman(&i.init, &i.bellman).unwrap();
        i
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
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

/// Warm wall time of the eight operations on Metal, for pricing a change to the backend;
/// no check of its own. `G16_RECOVERY_ROUNDS` rounds (default 5) over inputs at
/// `G16_RECOVERY_POWER` (default 16), one line per run, every round printed so the first,
/// which warms the caches, can be dropped. With `G16_TIMING_DIR` set the inputs are built
/// there once and reused by the next process, which is how two builds of the backend are
/// run turn and turn about on the same files: the test binary of each, alternated by a
/// shell loop, with the load average read beside every run.
#[test]
#[ignore = "timing, not a check: see its doc comment"]
fn time_metal_operations() {
    let rounds = env_usize("G16_RECOVERY_ROUNDS", 5);
    let power = env_usize("G16_RECOVERY_POWER", 16) as u32;
    assert!((4..=20).contains(&power), "G16_RECOVERY_POWER={power}");
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
    let metal = Backends {
        key: &key,
        fft: &fft,
        msm: &msm,
    };
    let inputs = match std::env::var_os("G16_TIMING_DIR") {
        Some(dir) => Inputs::reuse(PathBuf::from(dir).join(format!("power-{power}")), power),
        None => Inputs::new("timing", power),
    };
    let out_dir = inputs.dir.join(format!("timing-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).unwrap();
    for round in 0..rounds {
        for (op, name) in OPERATIONS.iter().enumerate() {
            let out = out_dir.join(format!("out-{op}"));
            let _ = std::fs::remove_file(&out);
            let t = std::time::Instant::now();
            run(op, &inputs, &out, &metal).unwrap();
            eprintln!(
                "timing power {power} round {round}: {name}: {:.3} s",
                t.elapsed().as_secs_f64()
            );
        }
    }
    let _ = std::fs::remove_dir_all(&out_dir);
}

/// One of the injector's faults: behind a submission's wait, or at a dispatch.
#[derive(Clone, Copy)]
enum Arm {
    Wait(Fault),
    Cut(Rest),
}

const ARMS: [Arm; 7] = [
    Arm::Wait(Fault::Status),
    Arm::Wait(Fault::Ignored),
    Arm::Wait(Fault::Stale),
    Arm::Wait(Fault::MissingToken),
    Arm::Cut(Rest::Buffer),
    Arm::Cut(Rest::Attempt),
    Arm::Wait(Fault::WrongStore),
];

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Wait(f) => format!("{f:?}"),
            Arm::Cut(r) => format!("Cut({r:?})"),
        }
    }

    /// Arms this fault at `round` of `submissions` waits or `dispatches` dispatches.
    fn arm(self, round: usize, submissions: u32, dispatches: u32) -> u32 {
        match self {
            Arm::Wait(f) => {
                let at = round as u32 % submissions;
                inject::arm_with(Some((at, f)), false);
                at
            }
            Arm::Cut(r) => {
                let at = round as u32 % dispatches;
                inject::arm_cut(at, r);
                at
            }
        }
    }
}

/// What one faulted run did, counted per operation and arm.
#[derive(Default)]
struct Tally {
    /// `Ok` and the CPU bytes: the backend retried.
    retried: usize,
    /// `Ok` and other bytes: the fault reached the file.
    escaped: usize,
    /// A device error with the destination and directory untouched.
    failed: usize,
}

#[test]
fn injected_metal_faults_match_cpu_or_preserve_destination() {
    let rounds = env_usize("G16_RECOVERY_ROUNDS", 4);
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

    // Everything below runs on the pool's one worker, the arming included, so the
    // submissions `zkey setup` makes from inside `par_iter` are this thread's.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let inputs = Inputs::new("inject", power);
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
        let mut problems = Vec::new();
        for (op, name) in OPERATIONS.iter().enumerate() {
            let out = out_dir.join(format!("out-{op}"));
            inject::arm(None);
            run(op, &inputs, &out, &metal).unwrap();
            assert_eq!(std::fs::read(&out).unwrap(), expected[op], "{name}: unfaulted");
            let (submissions, dispatches) = (inject::calls(), inject::dispatches());
            assert!(
                submissions > 0 && dispatches > 0,
                "{name}: nothing was submitted to Metal"
            );

            let mut tallies: Vec<Tally> = ARMS.iter().map(|_| Tally::default()).collect();
            for round in 0..rounds {
                for (arm, tally) in ARMS.iter().zip(&mut tallies) {
                    let at = arm.arm(round, submissions, dispatches);
                    let what = format!("{name}, {} at {at}", arm.label());
                    std::fs::write(&out, SENTINEL).unwrap();
                    let before = entries(&out_dir);
                    let result = run(op, &inputs, &out, &metal);
                    let fired = inject::fired();
                    inject::arm(None);
                    if !fired {
                        problems.push(format!("{what}: never reached"));
                    }
                    match result {
                        Ok(()) if std::fs::read(&out).unwrap() == expected[op] => {
                            tally.retried += 1;
                        }
                        Ok(()) => {
                            tally.escaped += 1;
                            problems.push(format!("{what}: Ok but differs from CPU"));
                        }
                        Err(CeremonyError::Accel(e)) => {
                            tally.failed += 1;
                            eprintln!("{what}: {e}");
                            if std::fs::read(&out).unwrap() != SENTINEL {
                                problems.push(format!("{what}: failed and replaced"));
                            }
                        }
                        Err(e) => problems.push(format!("{what}: non-device error {e}")),
                    }
                    if entries(&out_dir) != before {
                        problems.push(format!("{what}: directory changed"));
                    }
                }
            }
            eprintln!(
                "{name}: {submissions} submissions, {dispatches} dispatches; {}",
                ARMS.iter()
                    .zip(&tallies)
                    .map(|(arm, t)| format!(
                        "{} {} retried, {} escaped, {} failed",
                        arm.label(),
                        t.retried,
                        t.escaped,
                        t.failed
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }

        inject::arm(None);
        contribute::verify_from_init(&inputs.init, &inputs.prepared, &final_key, metal.msm).unwrap();
        let (submissions, dispatches) = (inject::calls(), inject::dispatches());
        let mut tally = Tally::default();
        for round in 0..rounds {
            for arm in ARMS {
                let at = arm.arm(round, submissions, dispatches);
                let what = format!("zkey verify, {} at {at}", arm.label());
                let result =
                    contribute::verify_from_init(&inputs.init, &inputs.prepared, &final_key, metal.msm);
                let fired = inject::fired();
                inject::arm(None);
                if !fired {
                    problems.push(format!("{what}: never reached"));
                }
                match result {
                    Ok(_) => tally.retried += 1,
                    Err(CeremonyError::Accel(e)) => {
                        tally.failed += 1;
                        eprintln!("{what}: {e}");
                    }
                    Err(e) => problems.push(format!("{what}: rejected a good key: {e}")),
                }
            }
        }
        eprintln!(
            "zkey verify: {submissions} submissions, {dispatches} dispatches; {} accepted, {} failed",
            tally.retried, tally.failed
        );
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}
