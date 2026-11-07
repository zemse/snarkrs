//! What the conversion tests share: the gnark toy fixture, the bench artifacts, and the two
//! outside verifiers, gnark's `groth16.Verify` through `test-support/gnark` and `snarkjs
//! groth16 verify`. Each outside half skips with a message when its tool is missing, or
//! fails under `G16_REQUIRE_TESTS=1`.

#![allow(dead_code)]

pub mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

pub fn harness() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-support/gnark")
}

pub fn toy(name: &str) -> PathBuf {
    harness().join("toy").join(name)
}

/// A bench artifact directory with every file named, or `None` after a skip message.
pub fn artifact(name: &str, files: &[&str]) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .join(name);
    resources::complete_files(&dir, files, name).then_some(dir)
}

pub fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-gnark-convert-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The harness built once per test binary, or `None` without Go.
fn gnark_binary() -> Option<&'static Path> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        if Command::new("go").arg("version").output().is_err() {
            return None;
        }
        let bin = scratch("harness").join("gnark-fixture");
        let out = Command::new("go")
            .current_dir(harness())
            .args(["build", "-o"])
            .arg(&bin)
            .arg(".")
            .output()
            .expect("go build");
        assert!(
            out.status.success(),
            "go build: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(bin)
    })
    .as_deref()
}

/// gnark's `groth16.Verify` on the three files: `Some(true)` when it accepts, `None` without
/// Go.
pub fn gnark_verify(vk: &Path, public: &Path, proof: &Path) -> Option<bool> {
    let Some(bin) = gnark_binary() else {
        resources::skip("gnark verify: no `go` on PATH, so gnark's own verifier was not run");
        return None;
    };
    let out = Command::new(bin)
        .arg("verify")
        .args([vk, public, proof])
        .output()
        .expect("gnark harness");
    if !out.status.success() {
        eprintln!("gnark: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Some(out.status.success())
}

/// `SNARKJS` or `snarkjs` on `PATH`, if it runs.
fn snarkjs() -> Option<String> {
    let exe = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".into());
    Command::new(&exe).arg("--help").output().ok().map(|_| exe)
}

/// `snarkjs <args>`: `Some(true)` when it exits 0 and prints no error, `None` without it.
pub fn snarkjs_run(args: &[&std::ffi::OsStr]) -> Option<bool> {
    let Some(exe) = snarkjs() else {
        resources::skip("snarkjs: not on PATH and SNARKJS unset, so snarkjs was not run");
        return None;
    };
    let out = Command::new(exe).args(args).output().expect("snarkjs");
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    // snarkjs 0.7 exits 0 on a failed verify on some paths; the log line is what it means.
    let ok = out.status.success() && !text.contains("ERROR") && !text.contains("Invalid proof");
    if !ok {
        eprintln!("snarkjs {args:?}: {}", text.trim());
    }
    Some(ok)
}

pub fn snarkjs_verify(vk: &Path, public: &Path, proof: &Path) -> Option<bool> {
    snarkjs_run(&[
        "groth16".as_ref(),
        "verify".as_ref(),
        vk.as_os_str(),
        public.as_os_str(),
        proof.as_os_str(),
    ])
}
