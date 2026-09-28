//! The native witness binary's subprocess handling, against small shell scripts that stand
//! in for circom's C++ program. The real one does not build on arm64 macOS (see the
//! README), and what is under test here is how its exits, signals and files are read, not
//! the circuit.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use snarkrs_field::Fr;
use snarkrs_witness::{native, wtns_bytes};

struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "snarkrs-witness-native-{test}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("input.json"), r#"{"a":"3"}"#).unwrap();
        Scratch(dir)
    }

    /// A stand-in binary: `body` is POSIX sh with `$1` the input and `$2` the output. It
    /// records the output path it was given, and `ls -l` of it once written, in `argv.log`.
    fn bin(&self, name: &str, body: &str, dat: bool) -> PathBuf {
        let bin = self.0.join(name);
        let log = self.0.join("argv.log");
        std::fs::write(
            &bin,
            format!("#!/bin/sh\necho \"$2\" > '{}'\n{body}\n", log.display()),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        if dat {
            std::fs::write(self.0.join(format!("{name}.dat")), b"").unwrap();
        }
        bin
    }

    fn logged_output(&self) -> Option<PathBuf> {
        let s = std::fs::read_to_string(self.0.join("argv.log")).ok()?;
        Some(PathBuf::from(s.lines().next()?))
    }

    fn input(&self) -> PathBuf {
        self.0.join("input.json")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn witness(n: u64) -> Vec<Fr> {
    (0..n).map(|i| Fr::from(i * i + 1)).collect()
}

/// A good witness at `$2`, from a fixture the script copies.
fn writes(s: &Scratch, w: &[Fr]) -> String {
    let fixture = s.0.join("fixture.wtns");
    std::fs::write(&fixture, wtns_bytes(w)).unwrap();
    format!(
        "cp '{}' \"$2\"\nls -l \"$2\" >> '{}'",
        fixture.display(),
        s.0.join("argv.log").display()
    )
}

fn err(r: Result<impl std::fmt::Debug, snarkrs_witness::WitnessError>) -> String {
    r.unwrap_err().to_string()
}

#[test]
fn success_reads_the_witness_and_leaves_nothing_behind() {
    let s = Scratch::new("ok");
    let w = witness(5);
    let bin = s.bin("circuit", &writes(&s, &w), true);
    assert_eq!(native::calculate(&bin, &s.input(), Some(5)).unwrap(), w);
    let out = s.logged_output().unwrap();
    assert!(out.is_absolute(), "{}", out.display());
    assert!(out.starts_with(std::env::temp_dir()), "{}", out.display());
    assert!(!out.exists(), "the temp witness is still there");
    // Created 0600 before the binary opened it, and still 0600 once written.
    let log = std::fs::read_to_string(s.0.join("argv.log")).unwrap();
    assert!(log.contains("-rw-------"), "{log}");

    // To a file: renamed into place.
    let dest = s.0.join("out.wtns");
    native::calculate_to_file(&bin, &s.input(), &dest).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), wtns_bytes(&w));
    let tmp = s.logged_output().unwrap();
    assert_ne!(tmp, dest);
    assert!(!tmp.exists());
}

#[test]
fn exit_zero_without_a_witness_is_a_failure() {
    let s = Scratch::new("usage");
    let bin = s.bin(
        "circuit",
        "echo \"Usage: $0 <input.json> <output.wtns>\"",
        true,
    );
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("exited 0 but wrote no witness"), "{e}");
    // A stale file where the witness should go is neither read nor replaced.
    let dest = s.0.join("witness.wtns");
    std::fs::write(&dest, wtns_bytes(&witness(3))).unwrap();
    let e = err(native::calculate_to_file(&bin, &s.input(), &dest));
    assert!(e.contains("usage error"), "{e}");
    assert_eq!(std::fs::read(&dest).unwrap(), wtns_bytes(&witness(3)));
}

#[test]
fn a_nonzero_exit_names_the_status_and_the_causes() {
    let s = Scratch::new("exit");
    let bin = s.bin(
        "circuit",
        "echo 'Not all inputs have been set' >&2\nexit 3",
        true,
    );
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("exited with status 3"), "{e}");
    assert!(e.contains("circuit assert"), "{e}");
    assert!(!s.logged_output().unwrap().exists());
}

#[test]
fn sigabrt_is_a_failure() {
    let s = Scratch::new("abrt");
    // What `assert(false)` does, after writing half a file.
    let bin = s.bin("circuit", "echo partial > \"$2\"\nkill -ABRT $$", true);
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("killed by SIGABRT"), "{e}");
    assert!(!s.logged_output().unwrap().exists());
    let dest = s.0.join("witness.wtns");
    let e = err(native::calculate_to_file(&bin, &s.input(), &dest));
    assert!(e.contains("SIGABRT"), "{e}");
    assert!(!dest.exists());
}

#[test]
fn a_missing_dat_is_reported_before_anything_runs() {
    let s = Scratch::new("dat");
    let bin = s.bin("circuit", "exit 0", false);
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("circuit.dat is missing"), "{e}");
    assert!(s.logged_output().is_none(), "the binary ran");
}

#[test]
fn a_witness_of_the_wrong_length_or_shape_is_refused() {
    let s = Scratch::new("nvars");
    let bin = s.bin("circuit", &writes(&s, &witness(5)), true);
    let e = err(native::calculate(&bin, &s.input(), Some(6)));
    assert!(e.contains("5 wires and the zkey has 6"), "{e}");
    assert!(!s.logged_output().unwrap().exists());

    // w[0] must be the constant one.
    let mut bad = witness(4);
    bad[0] = Fr::from(2u64);
    let bin = s.bin("circuit", &writes(&s, &bad), true);
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("not a BN254 witness"), "{e}");
    let bin = s.bin("circuit", "echo garbage > \"$2\"", true);
    let e = err(native::calculate(&bin, &s.input(), None));
    assert!(e.contains("not a BN254 witness"), "{e}");
}

#[test]
fn detect_goes_by_content() {
    let s = Scratch::new("detect");
    let wasm = s.0.join("circuit.bin");
    std::fs::write(&wasm, b"\0asm\x01\0\0\0").unwrap();
    assert_eq!(native::detect(&wasm).unwrap(), native::Kind::Wasm);
    let bin = s.bin("circuit.wasm", "exit 0", false);
    assert_eq!(native::detect(&bin).unwrap(), native::Kind::Native);
    let e = err(native::detect(&s.input()));
    assert!(e.contains("neither a wasm module"), "{e}");
    let e = err(native::detect(Path::new("no/such/file")));
    assert!(e.contains("ENOENT"), "{e}");
}
