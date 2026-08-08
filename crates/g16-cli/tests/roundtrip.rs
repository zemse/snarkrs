//! End-to-end tests that drive the real `g16` binary.
//!
//! These exist because unit tests on the JSON module can only prove that our reader and
//! our writer agree with each other. The thing that actually has to hold is that
//! `snarkjs groth16 verify` accepts what we wrote, so the encoding test here shells out
//! to snarkjs when it is installed and says so loudly when it is not.
//!
//! Everything runs against the checked-in artifacts and skips with a message when they
//! are absent, so a fresh clone without `gen-artifacts.sh` does not report false green.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn g16() -> &'static str {
    env!("CARGO_BIN_EXE_g16")
}

fn artifacts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

struct Variant {
    name: String,
    dir: PathBuf,
}

fn variants() -> Vec<Variant> {
    let Ok(entries) = std::fs::read_dir(artifacts_root()) else {
        return Vec::new();
    };
    let mut out: Vec<Variant> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .map(|dir| Variant {
            name: dir.file_name().unwrap().to_string_lossy().into_owned(),
            dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// A scratch directory per test, so two tests running concurrently cannot overwrite each
/// other's proof and turn a real failure into a confusing one.
fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-cli-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(cmd: &mut Command) -> Output {
    cmd.output()
        .unwrap_or_else(|e| panic!("failed to run {cmd:?}: {e}"))
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn prove(v: &Variant, out: &Path) -> (PathBuf, PathBuf) {
    let proof = out.join("proof.json");
    let public = out.join("public.json");
    let o = run(Command::new(g16()).args([
        "prove",
        "--zkey",
        v.dir.join("circuit.zkey").to_str().unwrap(),
        "--witness",
        v.dir.join("circuit.wtns").to_str().unwrap(),
        "--proof",
        proof.to_str().unwrap(),
        "--public",
        public.to_str().unwrap(),
        "--backend",
        "cpu",
        "--stage-timings",
    ]));
    assert!(o.status.success(), "{}: prove failed: {}", v.name, text(&o));
    (proof, public)
}

fn each(test: &str, f: impl Fn(&Variant, &Path)) {
    let found = variants();
    if found.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
        return;
    }
    let out = scratch(test);
    for v in &found {
        eprintln!("{test}: {}", v.name);
        f(v, &out);
    }
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn prove_then_verify_round_trips() {
    each("prove_then_verify", |v, out| {
        let (proof, public) = prove(v, out);
        let o = run(Command::new(g16()).args([
            "verify",
            "--vkey",
            v.dir.join("vkey.json").to_str().unwrap(),
            "--proof",
            proof.to_str().unwrap(),
            "--public",
            public.to_str().unwrap(),
        ]));
        assert!(
            o.status.success(),
            "{}: verify failed: {}",
            v.name,
            text(&o)
        );
        assert!(text(&o).contains("OK"), "{}", text(&o));

        // Our public.json must be the same signals snarkjs published for this witness,
        // otherwise the proof verifies only against our own idea of the statement.
        let ours: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&public).unwrap()).unwrap();
        let theirs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(v.dir.join("public.json")).unwrap())
                .unwrap();
        assert_eq!(ours, theirs, "{}: public signals differ", v.name);
    });
}

/// The encoding oracle. A proof that our verifier accepts but snarkjs rejects means the
/// JSON is wrong, which is a different bug from the pairing check being wrong, and it is
/// the one that only shows up at the ecosystem boundary.
#[test]
fn snarkjs_accepts_our_proof() {
    if Command::new("snarkjs").arg("--version").output().is_err() {
        eprintln!("SKIPPED snarkjs_accepts_our_proof: snarkjs is not on PATH");
        return;
    }
    each("snarkjs", |v, out| {
        let (proof, public) = prove(v, out);
        let o = run(Command::new("snarkjs").args([
            "groth16",
            "verify",
            v.dir.join("vkey.json").to_str().unwrap(),
            public.to_str().unwrap(),
            proof.to_str().unwrap(),
        ]));
        let said = text(&o);
        assert!(
            said.contains("OK!"),
            "{}: snarkjs rejected our proof: {said}",
            v.name
        );
    });
}

#[test]
fn verify_rejects_a_tampered_public_input() {
    each("tampered", |v, out| {
        let (proof, public) = prove(v, out);
        let mut signals: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(&public).unwrap()).unwrap();
        signals[0] = "1".to_string();
        let tampered = out.join("public-tampered.json");
        std::fs::write(&tampered, serde_json::to_string(&signals).unwrap()).unwrap();

        let o = run(Command::new(g16()).args([
            "verify",
            "--vkey",
            v.dir.join("vkey.json").to_str().unwrap(),
            "--proof",
            proof.to_str().unwrap(),
            "--public",
            tampered.to_str().unwrap(),
        ]));
        assert!(
            !o.status.success(),
            "{}: verify accepted a tampered public input",
            v.name
        );
    });
}

/// `bench` is what the project is measured by, so its output shape is a test, not a
/// convention: a missing column silently drops a stage from every chart built on it.
#[test]
fn bench_writes_a_verified_csv() {
    let found = variants();
    let Some(v) = found.first() else {
        eprintln!("SKIPPED bench_writes_a_verified_csv: no artifacts");
        return;
    };
    let out = scratch("bench");
    let csv = out.join("bench.csv");
    let o = run(Command::new(g16()).args([
        "bench",
        "--artifacts",
        artifacts_root().to_str().unwrap(),
        "--variant",
        &v.name,
        "--reps",
        "2",
        "--backend",
        "cpu",
        "--mode",
        "both",
        "--csv",
        csv.to_str().unwrap(),
    ]));
    assert!(o.status.success(), "bench failed: {}", text(&o));

    let body = std::fs::read_to_string(&csv).unwrap();
    let mut lines = body.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(
        header,
        [
            "host",
            "os",
            "arch",
            "cores",
            "variant",
            "constraints",
            "prover",
            "backend",
            "mode",
            "rep",
            "ms",
            "prepare_ms",
            "gather_us",
            "ntt_us",
            "pointwise_us",
            "msm_us",
            "assemble_us",
            "verified"
        ]
    );
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    assert_eq!(rows.len(), 4, "2 reps in each of 2 modes");
    let col =
        |r: &Vec<&str>, name: &str| r[header.iter().position(|h| *h == name).unwrap()].to_string();
    for r in &rows {
        assert_eq!(r.len(), header.len());
        assert_eq!(col(r, "verified"), "yes");
        assert_eq!(col(r, "prover"), "ours");
        assert_eq!(col(r, "backend"), "cpu");
        assert!(col(r, "ms").parse::<f64>().unwrap() > 0.0);
        assert!(col(r, "prepare_ms").parse::<f64>().unwrap() > 0.0);
    }
    let modes: Vec<String> = rows.iter().map(|r| col(r, "mode")).collect();
    assert_eq!(modes, ["cold", "cold", "warm", "warm"]);

    // Cold has to be slower than warm by construction: it pays the same prepare inside
    // the timed region. Compared on the minimum of each, because a loaded machine can
    // make any single rep arbitrarily slow but cannot make one faster than the floor.
    let best = |mode: &str| {
        rows.iter()
            .filter(|r| col(r, "mode") == mode)
            .map(|r| col(r, "ms").parse::<f64>().unwrap())
            .fold(f64::INFINITY, f64::min)
    };
    let (cold, warm) = (best("cold"), best("warm"));
    assert!(
        cold > warm,
        "cold {cold} ms was not slower than warm {warm} ms"
    );

    // Warm reports setup out of band; cold folds it into `ms`.
    for r in rows.iter().filter(|r| col(r, "mode") == "cold") {
        assert!(
            col(r, "ms").parse::<f64>().unwrap() > col(r, "prepare_ms").parse::<f64>().unwrap()
        );
    }
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn bench_rejects_an_unknown_variant() {
    let o = run(Command::new(g16()).args([
        "bench",
        "--artifacts",
        artifacts_root().to_str().unwrap(),
        "--variant",
        "no-such-circuit",
        "--reps",
        "1",
    ]));
    assert!(!o.status.success());
    assert!(text(&o).contains("no-such-circuit"), "{}", text(&o));
}

#[cfg(not(feature = "metal"))]
#[test]
fn metal_without_the_feature_fails_with_a_clear_message() {
    let found = variants();
    let Some(v) = found.first() else {
        eprintln!("SKIPPED metal_without_the_feature: no artifacts");
        return;
    };
    let out = scratch("metal");
    let o = run(Command::new(g16()).args([
        "prove",
        "--zkey",
        v.dir.join("circuit.zkey").to_str().unwrap(),
        "--witness",
        v.dir.join("circuit.wtns").to_str().unwrap(),
        "--proof",
        out.join("proof.json").to_str().unwrap(),
        "--public",
        out.join("public.json").to_str().unwrap(),
        "--backend",
        "metal",
    ]));
    assert!(!o.status.success());
    let said = text(&o);
    assert!(said.contains("--features metal"), "{said}");
    assert!(
        !out.join("proof.json").exists(),
        "a failed prove wrote a proof"
    );
    std::fs::remove_dir_all(&out).ok();
}
