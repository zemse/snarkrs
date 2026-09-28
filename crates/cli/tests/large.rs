//! The production-scale artifacts (BUG-23): every backend this binary was built with proves
//! each one, and both our verifier and snarkjs have to accept the proof.
//!
//! Everything else in the tree runs at domain 2^18 or below, so an index that wraps, a
//! launch geometry that degenerates or a chunking path that only switches on once the
//! working set stops fitting has never run. These artifacts live in `bench/artifacts/large/`,
//! one level below where every other test scans, and come from
//! `LARGE=1 bench/scripts/gen-artifacts.sh`.
//!
//! The proof is written with `--self-verify false`, so what the verifiers see is what the
//! device computed: with self-verify on, a wrong GPU proof is retried and then replaced by a
//! CPU one, and this test would pass on exactly the defect it exists to catch.
//!
//!     cargo test --release -p snarkrs-cli --features metal --test large -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use snarkrs_cli::artifacts::variants;

fn snarkrs() -> &'static str {
    env!("CARGO_BIN_EXE_snarkrs")
}

fn large_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/large")
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

/// Every backend this binary was built with, or the ones named in `G16_LARGE_BACKENDS`
/// (comma separated), so one backend can be rerun without paying for the others.
fn backends() -> Vec<String> {
    if let Ok(list) = std::env::var("G16_LARGE_BACKENDS") {
        return list.split(',').map(|s| s.trim().to_string()).collect();
    }
    let mut out = vec!["cpu"];
    if cfg!(feature = "wgpu") {
        out.push("wgpu");
    }
    if cfg!(feature = "metal") {
        out.push("metal");
    }
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    out.into_iter().map(String::from).collect()
}

#[test]
#[ignore = "long: a domain 2^22 circuit on every backend, minutes per proof on the cpu"]
fn every_backend_proves_the_large_artifacts() {
    let found = variants(large_root());
    if found.is_empty() {
        eprintln!(
            "SKIPPED every_backend_proves_the_large_artifacts: nothing under bench/artifacts/large, \
             run LARGE=1 bench/scripts/gen-artifacts.sh"
        );
        return;
    }
    let snarkjs = Command::new("snarkjs").arg("--version").output().is_ok();
    assert!(
        snarkjs,
        "snarkjs is not on PATH, and it is the independent verifier here"
    );

    let out = std::env::temp_dir().join(format!("snarkrs-large-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let mut failures = Vec::new();
    for v in &found {
        let theirs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(v.dir.join("public.json")).unwrap())
                .unwrap();
        for backend in backends() {
            let tag = format!("{} on {backend}", v.name);
            let proof = out.join(format!("{}-{backend}-proof.json", v.name));
            let public = out.join(format!("{}-{backend}-public.json", v.name));
            // No `G16_WGPU_LIMITS`: wgpu proves this at its default floor profile, which is
            // the point. The 567 MB of CSR values, the 256 MiB H query and the 431 MB G2
            // query are all over the floor's 128 MiB binding, and the backend chunks them.
            let o = run(Command::new(snarkrs()).args([
                "groth16",
                "prove",
                v.zkey().to_str().unwrap(),
                v.wtns().to_str().unwrap(),
                proof.to_str().unwrap(),
                public.to_str().unwrap(),
                "--backend",
                &backend,
                "--self-verify",
                "false",
                "--stage-timings",
            ]));
            eprintln!("{tag}:\n{}", text(&o));
            if !o.status.success() {
                failures.push(format!("{tag}: prove failed: {}", text(&o)));
                continue;
            }

            let ours: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&public).unwrap()).unwrap();
            if ours != theirs {
                failures.push(format!(
                    "{tag}: public signals differ from snarkjs' public.json"
                ));
            }

            let o = run(Command::new(snarkrs()).args([
                "groth16",
                "verify",
                v.vkey().to_str().unwrap(),
                public.to_str().unwrap(),
                proof.to_str().unwrap(),
            ]));
            if !o.status.success() || !text(&o).contains("OK") {
                failures.push(format!("{tag}: our verifier rejected it: {}", text(&o)));
            }

            let o = run(Command::new("snarkjs").args([
                "groth16",
                "verify",
                v.vkey().to_str().unwrap(),
                public.to_str().unwrap(),
                proof.to_str().unwrap(),
            ]));
            if !text(&o).contains("OK!") {
                failures.push(format!("{tag}: snarkjs rejected it: {}", text(&o)));
            }
        }
    }
    std::fs::remove_dir_all(&out).ok();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
