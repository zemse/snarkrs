//! `snarkrs groth16 setup --backend cpu` against `--backend metal`, whole file, byte for byte.
//!
//! That comparison is the whole milestone. `tests/setup.rs` already pins the CPU path to
//! snarkjs' own bytes, so a Metal zkey equal to a CPU zkey is a Metal zkey equal to
//! snarkjs, and it is a `cmp` rather than a signature check: a verifying proof does not
//! notice a wrong point in section 8, and neither does `snarkjs zkey verify`, which
//! recomputes sections 3 to 7 but takes 8 and 9 on the transcript's word.
//!
//! Byte equality is not luck. Group addition is exact and associative, so the summation
//! order a backend picks cannot move the answer, and the exit is canonical affine. What a
//! `cmp` catches is the things that do move it: a mishandled point at infinity, a dropped
//! term, an `Fq2` miscompiled under register pressure.
//!
//! # Why this drives the binary instead of calling `setup()`
//!
//! `MetalMsmBackend` lives in `snarkrs-metal`, which dev-depends on this crate to test its
//! kernels against `prepare::point_times_fr` and `CpuKeyScale`. Depending back on it, even
//! for a test, closes that loop. Driving `target/release/snarkrs` costs a process spawn and
//! tests the thing the correctness bar actually names: the shipped command, its
//! `--backend` flag and its file output.
//!
//! So the binary has to exist. Build it first:
//!
//! ```text
//! cargo build --release --features metal
//! cargo test --release -p snarkrs-ceremony --test setup_metal -- --nocapture
//! ```
//!
//! Without it, or without the `metal` feature in it, every test here skips loudly rather
//! than reporting a green it did not earn. `G16_BIN` overrides the path.
//!
//! The four production circuits the density table in `setup::setup` is drawn from are not
//! in this repo, so they are opt in through `G16_PRODUCTION_BUILDS`, pointed at the
//! `production/builds` directory. They are the cases that matter most and the ones
//! nobody else can run: `--O2` is where three quarters of the terms reach the multiexp,
//! and it is the only place a Metal MSM does enough work to be wrong.
//!
//! The timings each case prints are a convenience, not a benchmark, and with
//! `G16_PRODUCTION_BUILDS` set they need `--test-threads=1` to mean anything: cargo runs the two
//! tests concurrently, and a 2^20 setup against a 2^18 one is two twelve-thread runs and
//! several GB of resident ptau on one machine. `metal-profile/setup-metal-timings.md` holds
//! the numbers taken standalone.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// One circuit and the ptau whose power it is an exact fit for.
///
/// The fit is not a free choice even here, where nothing is compared against snarkjs:
/// `hash_h_points` reads one point past the end of ptau section 2 when
/// `cirPower == power`, so a different ptau is a different `csHash` and the two backends
/// would still agree, on a different file. Keeping the fit means these runs produce the
/// same bytes as the ones in `setup-metal-timings.md`.
struct Case {
    /// Directory under `bench/artifacts`, or under `G16_PRODUCTION_BUILDS` for a production one.
    dir: &'static str,
    r1cs: &'static str,
    ptau: &'static str,
    /// Share of terms in slots of 32 or more, `setup::MULTIEXP_MIN_TERMS`, measured. This
    /// is the number that decides how much of the run the Metal backend touches at all, so
    /// a case with a low one is testing the CPU loop and the wiring, not the kernel.
    msm_share: &'static str,
}

/// In this repo, ordered cheapest first. `tiny_mul` is here for the shape rather than the
/// density: at `cirPower` 3 every slot is far below the multiexp threshold, so it proves
/// the flag is plumbed and the identity slots survive, and it costs a second.
#[rustfmt::skip]
const LOCAL: &[Case] = &[
    Case { dir: "tiny_mul",     r1cs: "circuit.r1cs", ptau: "local_13.ptau",     msm_share: "0%" },
    Case { dir: "tornado",      r1cs: "circuit.r1cs", ptau: "ppot_0080_15.ptau", msm_share: "27.5%" },
    Case { dir: "js_2x2_d32",   r1cs: "circuit.r1cs", ptau: "local_19.ptau",     msm_share: "51.7%" },
];

/// The four production builds, under `G16_PRODUCTION_BUILDS`. The `--O1` and `--O2` compiles of
/// the same two circuits sit at opposite ends of the density table, which is the point of
/// running all four: `--O1` exercises the small-slot loop and the wiring, `--O2` puts three
/// quarters of the terms through the device.
#[rustfmt::skip]
const PRODUCTION: &[Case] = &[
    Case { dir: "prod_small_O1",      r1cs: "prod_small.r1cs",      ptau: "ppot_0080_17.ptau", msm_share: "3.8%" },
    Case { dir: "prod_small_O2",      r1cs: "prod_small.r1cs",      ptau: "ppot_0080_16.ptau", msm_share: "75.8%" },
    Case { dir: "prod_large_O1", r1cs: "prod_large.r1cs", ptau: "ppot_0080_20.ptau", msm_share: "4.8%" },
    Case { dir: "prod_large_O2", r1cs: "prod_large.r1cs", ptau: "ppot_0080_18.ptau", msm_share: "74.2%" },
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-ceremony-setup-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `snarkrs` binary, or `None` when it was never built.
fn g16_bin() -> Option<PathBuf> {
    let bin = match std::env::var("G16_BIN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => root().join("target/release/snarkrs"),
    };
    bin.exists().then_some(bin)
}

/// The r1cs and ptau of a case, or `None` when either was never generated.
fn inputs(case: &Case, builds: &Path) -> Option<(PathBuf, PathBuf)> {
    let r1cs = builds.join(case.dir).join(case.r1cs);
    let ptau = root().join("bench/ptau").join(case.ptau);
    (r1cs.exists() && ptau.exists()).then_some((r1cs, ptau))
}

/// `snarkrs groth16 setup`, timed. `Err` carries stderr so the caller can tell a missing feature from
/// a real failure: those two must not both look like a skip.
fn setup(
    bin: &Path,
    r1cs: &Path,
    ptau: &Path,
    out: &Path,
    backend: &str,
) -> Result<Duration, String> {
    let start = Instant::now();
    let res = Command::new(bin)
        .args(["groth16", "setup"])
        .arg(r1cs)
        .arg(ptau)
        .arg(out)
        .args(["--backend", backend])
        .output()
        .expect("snarkrs");
    if res.status.success() {
        Ok(start.elapsed())
    } else {
        Err(String::from_utf8_lossy(&res.stderr).into_owned())
    }
}

/// Whether this binary has a Metal backend, asked of the binary itself rather than of
/// `cfg!`: the test process and the binary are separate builds and only one of them was
/// given the feature.
fn has_metal(bin: &Path) -> bool {
    // Every path through `--backend metal` opens the device before it reads a file, so a
    // nonexistent r1cs still reaches the backend error first.
    let missing = root().join("bench/artifacts/does-not-exist.r1cs");
    let out = std::env::temp_dir().join("snarkrs-metal-probe.zkey");
    match setup(bin, &missing, &missing, &out, "metal") {
        Ok(_) => panic!("`snarkrs groth16 setup` succeeded on a nonexistent r1cs"),
        Err(e) => !e.contains("WITHOUT the `metal` feature"),
    }
}

/// The comparison, over whichever table the caller passes.
///
/// Deletes each pair as it goes: `prod_large_O1` alone writes 295 MB twice,
/// and the four production cases together would leave 1.6 GB in the temp directory.
fn compare(cases: &[Case], builds: &Path, label: &str) -> usize {
    let Some(bin) = g16_bin() else {
        eprintln!("skipping: no `snarkrs` binary; run `cargo build --release --features metal`");
        return 0;
    };
    if !has_metal(&bin) {
        eprintln!(
            "skipping: {} was built without the `metal` feature",
            bin.display()
        );
        return 0;
    }
    let dir = tmp_dir(label);
    let mut ran = 0;
    for case in cases {
        let Some((r1cs, ptau)) = inputs(case, builds) else {
            eprintln!("skipping {}: artifact missing", case.dir);
            continue;
        };
        let cpu = dir.join(format!("{}-cpu.zkey", case.dir));
        let metal = dir.join(format!("{}-metal.zkey", case.dir));

        let t_cpu = setup(&bin, &r1cs, &ptau, &cpu, "cpu")
            .unwrap_or_else(|e| panic!("{}: cpu setup failed:\n{e}", case.dir));
        let t_metal = setup(&bin, &r1cs, &ptau, &metal, "metal")
            .unwrap_or_else(|e| panic!("{}: metal setup failed:\n{e}", case.dir));

        let a = std::fs::read(&cpu).unwrap();
        let b = std::fs::read(&metal).unwrap();
        assert_eq!(a.len(), b.len(), "{}: file length", case.dir);
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        assert!(
            first.is_none(),
            "{}: cpu and metal zkeys differ at byte {}",
            case.dir,
            first.unwrap()
        );

        eprintln!(
            "{:<32} {:>8} of terms in msm slots   cpu {:>8.2?}   metal {:>8.2?}   {:.2}x",
            case.dir,
            case.msm_share,
            t_cpu,
            t_metal,
            t_cpu.as_secs_f64() / t_metal.as_secs_f64(),
        );
        ran += 1;
        let _ = std::fs::remove_file(&cpu);
        let _ = std::fs::remove_file(&metal);
    }
    let _ = std::fs::remove_dir_all(&dir);
    ran
}

/// The three circuits this repo carries, spanning nothing to half the terms in a multiexp.
#[test]
fn cpu_and_metal_agree_on_the_bundled_circuits() {
    let ran = compare(LOCAL, &root().join("bench/artifacts"), "local");
    if g16_bin().is_some_and(|b| has_metal(&b)) {
        assert!(ran > 0, "no artifacts present, nothing was compared");
    }
}

/// The four production builds, including the two `--O2` compiles where three quarters of
/// the terms reach the device.
///
/// Opt in with `G16_PRODUCTION_BUILDS=/path/to/production/builds`. Half an hour at 2^18 and
/// 2^20, most of it the Metal runs, which are the slow half.
#[test]
fn cpu_and_metal_agree_on_the_production_circuits() {
    let Ok(builds) = std::env::var("G16_PRODUCTION_BUILDS") else {
        eprintln!("skipping: set G16_PRODUCTION_BUILDS to a production/builds directory");
        return;
    };
    let ran = compare(PRODUCTION, Path::new(&builds), "production");
    if g16_bin().is_some_and(|b| has_metal(&b)) {
        assert!(ran > 0, "G16_PRODUCTION_BUILDS holds none of the four circuits");
    }
}
