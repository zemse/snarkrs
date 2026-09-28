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

/// snarkjs 0.7.6, named by `SNARKJS` because the one on PATH may be another version.
fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    let Ok(out) = Command::new(&bin).arg("--help").output() else {
        eprintln!("SKIPPED: snarkjs is not installed");
        return None;
    };
    if !String::from_utf8_lossy(&out.stdout).contains("snarkjs@0.7.6") {
        eprintln!("SKIPPED: {bin} is not snarkjs 0.7.6; set SNARKJS");
        return None;
    }
    Some(bin)
}

/// One line through snarkjs in `dir`: its exit code and its stdout with logplease's colours
/// taken out. Stdout goes through a file, because snarkjs calls `process.exit` before a
/// pipe has drained and a long log comes back cut short.
fn snarkjs(bin: &str, dir: &Path, line: &str) -> (i32, String) {
    let log = dir.join(".snarkjs.log");
    let status = Command::new(bin)
        .args(line.split_whitespace())
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&log).unwrap())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| panic!("running snarkjs {line}: {e}"));
    let out = std::fs::read_to_string(&log).unwrap();
    (status.code().unwrap_or(-1), strip_colour(&out))
}

fn strip_colour(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[track_caller]
fn same_bytes(dir: &Path, ours: &str, theirs: &str) {
    let got = std::fs::read(dir.join(ours)).unwrap();
    let want = std::fs::read(dir.join(theirs)).unwrap();
    assert_eq!(got.len(), want.len(), "{ours} vs {theirs}: lengths");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{ours} vs {theirs}: first difference at byte {i}");
    }
}

/// The first `n` log lines, hash rows included: the deterministic head of a log whose tail
/// depends on the entropy.
fn head(log: &str, n: usize) -> String {
    let mut seen = 0;
    log.lines()
        .take_while(|l| {
            if l.starts_with('[') {
                seen += 1;
            }
            seen <= n
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A contribution made away from the ptau: export the challenge, contribute to it, import
/// the response. Export and import are deterministic and are held to snarkjs' bytes and
/// log; the contribution is random, so snarkjs has to accept what it produced.
#[test]
fn challenge_and_response_match_snarkjs() {
    let Some(js) = snarkjs_bin() else {
        return;
    };
    let dir = scratch("challenge");
    let run = |line: &str| {
        let o = snarkrs(&dir, line);
        expect(&o, 0, line);
        stdout(&o)
    };
    let js_run = |line: &str| {
        let (code, out) = snarkjs(&js, &dir, line);
        assert_eq!(code, 0, "snarkjs {line}: {out}");
        out
    };
    run("ptn bn128 4 pot_0000.ptau");
    run("ptc pot_0000.ptau pot_0001.ptau -e=one -n=First");

    // `challenge` is the default name.
    let ours = run("powersoftau export challenge pot_0001.ptau");
    let theirs = js_run("powersoftau export challenge pot_0001.ptau challenge_js");
    same_bytes(&dir, "challenge", "challenge_js");
    assert_eq!(ours, theirs);
    let ours = run("ptec pot_0000.ptau fresh.bin");
    let theirs = js_run("ptec pot_0000.ptau fresh_js");
    same_bytes(&dir, "fresh.bin", "fresh_js");
    assert_eq!(ours, theirs);

    // The response is named after the challenge by default. What it claims and what it
    // answers are fixed by the challenge, so those two hashes match snarkjs'.
    let ours = run("powersoftau challenge contribute bn128 challenge -e=two");
    assert!(dir.join("challenge.response").is_file());
    let theirs = js_run("ptcc bn128 challenge_js challenge_js.response -e=two");
    assert_eq!(head(&ours, 2), head(&theirs, 2));
    assert!(
        ours.contains("[INFO]  snarkJS: Contribution Response Hash: \n\t\t"),
        "{ours}"
    );
    run("ptcc bn128 fresh.bin -e=three");
    assert!(dir.join("fresh.response").is_file());

    // Import is deterministic given the response, with and without the points.
    for flags in ["-n=Second", "-n=Second -nopoints -nocheck"] {
        let tag = if flags.contains("nopoints") {
            "_np"
        } else {
            ""
        };
        let ours = run(&format!(
            "powersoftau import response pot_0001.ptau challenge.response pot_0002{tag}.ptau {flags}"
        ));
        let theirs = js_run(&format!(
            "ptir pot_0001.ptau challenge.response js_0002{tag}.ptau {flags}"
        ));
        same_bytes(
            &dir,
            &format!("pot_0002{tag}.ptau"),
            &format!("js_0002{tag}.ptau"),
        );
        assert_eq!(ours, theirs, "{flags}");
    }
    let out = js_run("ptv pot_0002.ptau");
    assert!(out.contains("Powers of Tau Ok!"), "{out}");
    // snarkjs' response imports here too, and snarkjs accepts the result.
    run("ptir pot_0001.ptau challenge_js.response pot_0002b.ptau");
    js_run("ptv pot_0002b.ptau");

    // A response to another file's challenge is refused.
    let line = "ptir pot_0000.ptau challenge.response never.ptau";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    assert!(
        stdout(&o).starts_with("[ERROR] snarkJS: Error: "),
        "{}",
        said(&o)
    );
    let line = "ptcc bls12381 challenge -e=x";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    assert!(said(&o).contains("not supported"), "{}", said(&o));

    std::fs::remove_dir_all(&dir).ok();
}

/// A zkey contribution made in bellman's format, and the Solidity verifier. Export and
/// import are deterministic and held to snarkjs' bytes and log; the contribution is random,
/// so snarkjs' `zkvi` has to accept the key it ends in. The verifier is our own contract,
/// so it is held to the library's output and only its log to snarkjs'.
#[test]
fn bellman_and_solidity_match_snarkjs() {
    let Some(js) = snarkjs_bin() else {
        return;
    };
    let Some(tiny) = tiny_mul() else {
        eprintln!("SKIPPED bellman_and_solidity_match_snarkjs: no tiny_mul");
        return;
    };
    let dir = scratch("bellman");
    let run = |line: &str| {
        let o = snarkrs(&dir, line);
        expect(&o, 0, line);
        stdout(&o)
    };
    let js_run = |line: &str| {
        let (code, out) = snarkjs(&js, &dir, line);
        assert_eq!(code, 0, "snarkjs {line}: {out}");
        out
    };
    std::fs::copy(tiny.join("circuit.r1cs"), dir.join("circuit.r1cs")).unwrap();
    run("ptn bn128 4 pot_0000.ptau");
    run("ptc pot_0000.ptau pot_0001.ptau -e=one");
    run("pt2 pot_0001.ptau powersoftau.ptau");
    run("g16s");
    run("zkc circuit_0000.zkey circuit_0001.zkey -e=two -n=First");

    // `circuit.mpcparams` is the default name.
    let ours = run("zkey export bellman circuit_0001.zkey");
    let theirs = js_run("zkeb circuit_0001.zkey js.mpcparams");
    same_bytes(&dir, "circuit.mpcparams", "js.mpcparams");
    assert_eq!(ours, theirs);

    let out =
        run("zkey bellman contribute bn128 circuit.mpcparams circuit_response.mpcparams -e=three");
    assert!(
        out.starts_with("[INFO]  snarkJS: Contribution Hash: \n\t\t"),
        "{out}"
    );
    js_run("zkbc bn128 circuit.mpcparams js_response.mpcparams -e=four");

    // Import is deterministic given the response, whichever tool contributed.
    for (resp, tag) in [("circuit_response", ""), ("js_response", "b")] {
        let ours = run(&format!(
            "zkey import bellman circuit_0001.zkey {resp}.mpcparams circuit_0002{tag}.zkey -n=Bell"
        ));
        let theirs = js_run(&format!(
            "zkib circuit_0001.zkey {resp}.mpcparams js_0002{tag}.zkey -n=Bell"
        ));
        same_bytes(
            &dir,
            &format!("circuit_0002{tag}.zkey"),
            &format!("js_0002{tag}.zkey"),
        );
        assert_eq!(ours, theirs, "{resp}");
        let out = js_run(&format!(
            "zkvi circuit_0000.zkey powersoftau.ptau circuit_0002{tag}.zkey"
        ));
        assert!(out.contains("ZKey Ok!"), "{out}");
    }

    // A response whose chain the zkey does not continue is a verdict, worded as snarkjs'.
    let line = "zkib circuit_0002.zkey js_response.mpcparams never.zkey";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    let (code, theirs) = snarkjs(&js, &dir, line);
    assert_eq!(code, 1, "{theirs}");
    assert_eq!(stdout(&o), theirs);
    let line = "zkbc bls12381 circuit.mpcparams x.mpcparams -e=x";
    let o = snarkrs(&dir, line);
    expect(&o, 1, line);
    assert!(said(&o).contains("not supported"), "{}", said(&o));

    // `circuit_final.zkey` and `verifier.sol` are the defaults.
    std::fs::copy(
        dir.join("circuit_0002.zkey"),
        dir.join("circuit_final.zkey"),
    )
    .unwrap();
    let ours = run("zkey export solidityverifier");
    let theirs = js_run("zkesv circuit_final.zkey js_verifier.sol");
    assert_eq!(ours, theirs);
    let want =
        snarkrs_ceremony::solidity::solidity_verifier(&dir.join("circuit_final.zkey")).unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.join("verifier.sol")).unwrap(),
        want
    );
    run("generateverifier");

    std::fs::remove_dir_all(&dir).ok();
}
