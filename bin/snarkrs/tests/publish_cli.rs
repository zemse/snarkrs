//! Every file the CLI writes itself goes through a stage: a write that fails leaves the old
//! destination byte for byte and no stage beside it, and one that succeeds replaces it.
//! Driven through the binary, because the exit status is part of the contract.

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const OLD: &[u8] = b"existing destination";

fn tiny_mul() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul");
    let files = [
        "circuit_js/circuit.wasm",
        "input.json",
        "circuit.wtns",
        "circuit.zkey",
        "vkey.json",
    ];
    resources::complete_files(&dir, &files, "publish_cli").then_some(dir)
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("snarkrs-publish-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn snarkrs(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_snarkrs"))
        .args(args)
        .current_dir(dir)
        .env("SNARKRS_NO_TIPS", "1")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[track_caller]
fn expect(o: &Output, code: i32) {
    assert_eq!(
        o.status.code(),
        Some(code),
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// A copy of the fixture's inputs in a fresh directory.
fn setup(tiny: &Path, test: &str) -> PathBuf {
    let dir = scratch(test);
    std::fs::copy(
        tiny.join("circuit_js/circuit.wasm"),
        dir.join("circuit.wasm"),
    )
    .unwrap();
    for f in ["input.json", "circuit.wtns", "circuit.zkey"] {
        std::fs::copy(tiny.join(f), dir.join(f)).unwrap();
    }
    dir
}

/// `args` with `out` a directory fails with `code`, leaving the directory, its neighbours
/// and the rest of `dir` as they were. Then, with `out` an existing file, it succeeds and
/// replaces it, and nothing else appears.
fn check(dir: &Path, out: &str, args: &[&str], code: i32) {
    std::fs::create_dir(dir.join(out)).unwrap();
    let before = entries(dir);
    expect(&snarkrs(dir, args), code);
    assert!(dir.join(out).is_dir());
    assert_eq!(entries(dir), before, "orphan stage");
    std::fs::remove_dir(dir.join(out)).unwrap();

    std::fs::write(dir.join(out), OLD).unwrap();
    let before = entries(dir);
    expect(&snarkrs(dir, args), 0);
    assert_ne!(
        std::fs::read(dir.join(out)).unwrap(),
        OLD,
        "{out} not replaced"
    );
    assert_eq!(entries(dir), before, "orphan stage");
}

#[test]
fn solidity_verifier_and_trace_publish_through_a_stage() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = setup(&tiny, "export");
    check(
        &dir,
        "verifier.sol",
        &[
            "zkey",
            "export",
            "solidityverifier",
            "circuit.zkey",
            "verifier.sol",
        ],
        1,
    );
    check(
        &dir,
        "trace.txt",
        &[
            "trace",
            "--zkey",
            "circuit.zkey",
            "--witness",
            "circuit.wtns",
            "--out",
            "trace.txt",
        ],
        1,
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wtns_calculate_publishes_through_a_stage() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = setup(&tiny, "wtns");
    let args = [
        "wtns",
        "calculate",
        "circuit.wasm",
        "input.json",
        "out.wtns",
    ];
    check(&dir, "out.wtns", &args, 1);
    assert_eq!(
        std::fs::read(dir.join("out.wtns")).unwrap(),
        std::fs::read(tiny.join("circuit.wtns")).unwrap()
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// public.json a directory fails the set before either file is staged, so the existing
/// proof.json is not replaced by a proof whose signals were never written.
#[test]
fn prove_replaces_proof_and_public_together_or_not_at_all() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = setup(&tiny, "prove");
    let args = [
        "groth16",
        "prove",
        "circuit.zkey",
        "circuit.wtns",
        "proof.json",
        "public.json",
        "--backend",
        "cpu",
    ];
    std::fs::write(dir.join("proof.json"), OLD).unwrap();
    std::fs::create_dir(dir.join("public.json")).unwrap();
    let before = entries(&dir);
    expect(&snarkrs(&dir, &args), 1);
    assert_eq!(std::fs::read(dir.join("proof.json")).unwrap(), OLD);
    assert!(dir.join("public.json").is_dir());
    assert_eq!(entries(&dir), before, "orphan stage");
    std::fs::remove_dir(dir.join("public.json")).unwrap();

    std::fs::write(dir.join("public.json"), OLD).unwrap();
    expect(&snarkrs(&dir, &args), 0);
    for f in ["proof.json", "public.json"] {
        assert_ne!(std::fs::read(dir.join(f)).unwrap(), OLD, "{f} not replaced");
    }
    assert_eq!(entries(&dir), before, "orphan stage");
    let vkey = tiny.join("vkey.json");
    let vkey = vkey.to_str().unwrap();
    let verify = ["groth16", "verify", vkey, "public.json", "proof.json"];
    expect(&snarkrs(&dir, &verify), 0);
    std::fs::remove_dir_all(&dir).ok();
}

/// `args` in `dir` with the umask at 022, as a login shell usually has it.
#[cfg(unix)]
fn snarkrs_umask_022(dir: &Path, args: &[&str]) -> Output {
    use std::os::unix::process::CommandExt;
    let mut c = Command::new(env!("CARGO_BIN_EXE_snarkrs"));
    c.args(args)
        .current_dir(dir)
        .env("SNARKRS_NO_TIPS", "1")
        .stdin(Stdio::null());
    // SAFETY: umask is async-signal-safe and touches nothing but the child's own mask.
    unsafe {
        c.pre_exec(|| {
            libc::umask(0o022);
            Ok(())
        });
    }
    c.output().unwrap()
}

/// A new output is as readable as a plain write makes it, 0644 under umask 022, and an
/// existing one keeps its own mode.
#[cfg(unix)]
#[test]
fn new_outputs_follow_the_umask_and_existing_ones_keep_their_mode() {
    use std::os::unix::fs::PermissionsExt;
    let Some(tiny) = tiny_mul() else { return };
    let dir = setup(&tiny, "mode");
    let mode = |f: &str| std::fs::metadata(dir.join(f)).unwrap().permissions().mode() & 0o777;
    let runs: [&[&str]; 4] = [
        &[
            "zkey",
            "export",
            "solidityverifier",
            "circuit.zkey",
            "verifier.sol",
        ],
        &[
            "wtns",
            "calculate",
            "circuit.wasm",
            "input.json",
            "out.wtns",
        ],
        &[
            "trace",
            "--zkey",
            "circuit.zkey",
            "--witness",
            "circuit.wtns",
            "--out",
            "trace.txt",
        ],
        &[
            "groth16",
            "prove",
            "circuit.zkey",
            "circuit.wtns",
            "proof.json",
            "public.json",
            "--backend",
            "cpu",
        ],
    ];
    let outputs = [
        "verifier.sol",
        "out.wtns",
        "trace.txt",
        "proof.json",
        "public.json",
    ];
    for args in runs {
        expect(&snarkrs_umask_022(&dir, args), 0);
    }
    for f in outputs {
        assert_eq!(mode(f), 0o644, "new {f}");
        std::fs::set_permissions(dir.join(f), std::fs::Permissions::from_mode(0o640)).unwrap();
    }
    for args in runs {
        expect(&snarkrs_umask_022(&dir, args), 0);
    }
    for f in outputs {
        assert_eq!(mode(f), 0o640, "existing {f}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
