//! snarkjs command lines, verbatim, against the `snarkrs` binary.
//!
//! The normaliser's unit tests in `src/cli.rs` cover the rewriting. These cover what a
//! script sees: exit codes, the log lines it greps, the default file names, and a whole
//! ceremony and proof driven by the lines snarkjs' own README uses. Phase 1 needs nothing
//! on disk; phase 2 needs `tiny_mul` from `bench/artifacts` and skips loudly without it.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("snarkrs-cli-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One snarkjs line, split on spaces the way a shell would split it, run in `dir`.
fn snarkrs(dir: &Path, line: &str) -> Output {
    snarkrs_with_stdin(dir, line, None)
}

fn snarkrs_with_stdin(dir: &Path, line: &str, stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_snarkrs"))
        .args(line.split_whitespace())
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Dropping the handle closes stdin, which is what a script with no terminal gives it.
    let mut pipe = child.stdin.take().unwrap();
    if let Some(input) = stdin {
        pipe.write_all(input.as_bytes()).unwrap();
    }
    drop(pipe);
    child.wait_with_output().unwrap()
}

fn said(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[track_caller]
fn expect(o: &Output, code: i32, line: &str) {
    assert_eq!(o.status.code(), Some(code), "`{line}`: {}", said(o));
}

/// 0 for success, 1 for a failure, 99 for a line snarkjs could not use, with the command
/// list printed for the last.
#[test]
fn exit_codes_follow_snarkjs() {
    let dir = scratch("exit");
    for (line, code, needle) in [
        ("", 99, "Usage:"),
        ("-v", 99, "Usage:"),
        ("nosuchcommand", 99, "Invalid command"),
        ("groth17 prove", 99, "Invalid command"),
        ("g16v a b c d", 99, "unexpected argument"),
        ("ptn bn128", 99, "required"),
        ("g16p --nosuchflag", 99, "nosuchflag"),
        ("--help", 0, "Full Command"),
        ("-h", 0, "Full Command"),
        ("g16v -h", 0, "Usage"),
        ("groth16 verify --help", 0, "Usage"),
        ("--version", 0, "snarkrs"),
        ("pks", 1, "Groth16 on BN254 only"),
        ("zkey export json", 1, "does not implement yet"),
        ("ptn bls12381 4", 1, "not supported"),
        ("ptn BLS12-381 4", 1, "not supported"),
        ("ptn bn128 29", 1, "Power must be between 1 and 28"),
        ("fi", 1, "needs a file"),
        ("fi x.json", 1, "Extension json is not allowed."),
    ] {
        let o = snarkrs(&dir, line);
        expect(&o, code, line);
        assert!(said(&o).contains(needle), "`{line}` said: {}", said(&o));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A missing file is an error line in snarkjs' form and exit 1, not a panic and not 99.
#[test]
fn a_missing_file_is_an_error_line() {
    let dir = scratch("missing");
    let o = snarkrs(&dir, "groth16 verify");
    expect(&o, 1, "groth16 verify");
    assert!(
        stdout(&o).starts_with("[ERROR] snarkJS: Error: loading verification_key.json"),
        "{}",
        said(&o)
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn tiny_mul() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul");
    (dir.join("circuit.r1cs").is_file() && dir.join("circuit.wtns").is_file()).then_some(dir)
}

/// The ceremony and proof from snarkjs' README, line for line in its own spellings, with
/// every optional file name left to its default.
#[test]
fn a_ceremony_and_a_proof_run_on_snarkjs_lines() {
    let dir = scratch("ceremony");
    let run = |line: &str| {
        let o = snarkrs(&dir, line);
        expect(&o, 0, line);
        stdout(&o)
    };

    // Phase 1. `ptn` names its output after the power when no name is given.
    let out = run("powersoftau new bn128 4 -v");
    assert!(
        out.starts_with("[INFO]  snarkJS: First Contribution Hash:\n\t\t"),
        "{out}"
    );
    assert!(dir.join("powersOfTau4_0000.ptau").is_file());

    // No `-e`: snarkjs asks on stdin, and so does this.
    let line = "ptc powersOfTau4_0000.ptau pot_0001.ptau --name=First";
    let o = snarkrs_with_stdin(&dir, line, Some("some typed entropy\n"));
    expect(&o, 0, line);
    assert!(
        stdout(&o).starts_with("Enter a random text. (Entropy): "),
        "{}",
        said(&o)
    );
    assert!(stdout(&o).contains("[INFO]  snarkJS: Contribution Response Hash imported: "));
    // With stdin closed there is nobody to ask: exit 1, not a hang.
    let line = "ptc powersOfTau4_0000.ptau never.ptau";
    let o = snarkrs_with_stdin(&dir, line, None);
    expect(&o, 1, line);
    assert!(!dir.join("never.ptau").exists());

    let out = run("powersOfTau CONTRIBUTE pot_0001.ptau pot_0002.ptau -e=more -n=Second");
    assert!(
        out.contains("[INFO]  snarkJS: Next Challenge Hash: \n\t\t"),
        "{out}"
    );
    run("ptb pot_0002.ptau pot_beacon.ptau 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 10 -n=Final");
    run("powersoftau prepare phase2 pot_beacon.ptau powersoftau.ptau");
    let out = run("ptv powersoftau.ptau");
    assert!(
        out.contains("[INFO]  snarkJS: Powers Of tau file OK!"),
        "{out}"
    );
    assert!(
        out.ends_with("[INFO]  snarkJS: Powers of Tau Ok!\n"),
        "{out}"
    );
    assert_eq!(out.matches("Response Hash:").count(), 3, "{out}");
    let out = run("fi powersoftau.ptau");
    assert!(out.contains("prepared       true"), "{out}");

    let Some(tiny) = tiny_mul() else {
        eprintln!("SKIPPED phase 2 of a_ceremony_and_a_proof_run_on_snarkjs_lines: no tiny_mul");
        std::fs::remove_dir_all(&dir).ok();
        return;
    };
    std::fs::copy(tiny.join("circuit.r1cs"), dir.join("circuit.r1cs")).unwrap();
    std::fs::copy(tiny.join("circuit.wtns"), dir.join("witness.wtns")).unwrap();

    // Phase 2, on defaults: circuit.r1cs, powersoftau.ptau, circuit_0000.zkey,
    // circuit_final.zkey, circuit_vk.json, witness.wtns, proof.json and public.json.
    let out = run("groth16 setup");
    assert!(
        out.starts_with("[INFO]  snarkJS: Circuit hash: \n"),
        "{out}"
    );
    assert!(dir.join("circuit_0000.zkey").is_file());
    run("zkc circuit_0000.zkey circuit_0001.zkey -e=phase2 -n=One");
    let out = run("zkb circuit_0001.zkey circuit_final.zkey 0102 10");
    assert!(
        out.starts_with("[INFO]  snarkJS: Contribution Hash: \n"),
        "{out}"
    );
    for line in [
        "zkey verify",
        "zkv",
        "zkvr",
        "zkey verify r1cs circuit.r1cs powersoftau.ptau circuit_final.zkey",
        "zkvi",
        "zkey verify init circuit_0000.zkey powersoftau.ptau circuit_final.zkey",
    ] {
        let out = run(line);
        assert!(
            out.ends_with("[INFO]  snarkJS: ZKey Ok!\n"),
            "`{line}`: {out}"
        );
    }
    // A chain that does not start at the given initial key is a verdict: exit 1.
    let line = "zkvi circuit_0001.zkey powersoftau.ptau circuit_0001.zkey";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    assert!(stdout(&o).starts_with("[ERROR] snarkJS: "), "{}", said(&o));

    let out = run("zkev");
    assert!(out.contains("EXPORT VERIFICATION KEY FINISHED"), "{out}");
    assert!(dir.join("circuit_vk.json").is_file());

    run("g16p");
    assert!(dir.join("proof.json").is_file() && dir.join("public.json").is_file());
    let out = run("g16v circuit_vk.json");
    assert_eq!(out, "[INFO]  snarkJS: OK!\n");
    let out = run("groth16 verify circuit_vk.json public.json proof.json");
    assert_eq!(out, "[INFO]  snarkJS: OK!\n");

    // The first public signal plus one.
    let public: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("public.json")).unwrap()).unwrap();
    let mut tampered = public.clone();
    tampered[0] = (tampered[0].parse::<u128>().unwrap() + 1).to_string();
    std::fs::write(
        dir.join("tampered.json"),
        serde_json::to_string(&tampered).unwrap(),
    )
    .unwrap();
    let line = "g16v circuit_vk.json tampered.json proof.json";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    assert_eq!(stdout(&o), "[ERROR] snarkJS: Invalid proof\n");

    std::fs::remove_dir_all(&dir).ok();
}
