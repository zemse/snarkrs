//! `wtns calculate`, `wtns debug` and `groth16 fullprove` through the binary, against
//! snarkjs 0.7.6 where it is installed (`SNARKJS`), circom where a test needs a circuit of
//! its own, and `bench/artifacts/tiny_mul`. Each piece that is missing skips loudly.
//!
//! The byte-for-byte check over every artifact's wasm is `snarkrs-witness`'s own test; this
//! one is about what a script sees: exit codes, stdout, stderr, and which files appear.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("snarkrs-wtns-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One line through snarkrs in `dir`, with `envs` added and tips on unless asked.
fn snarkrs_env(dir: &Path, line: &str, envs: &[(&str, &Path)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_snarkrs"));
    c.args(line.split_whitespace())
        .current_dir(dir)
        .env_remove("SNARKRS_NO_TIPS")
        .stdin(Stdio::null());
    for (k, v) in envs {
        c.env(k, v);
    }
    c.output().unwrap()
}

fn snarkrs(dir: &Path, line: &str) -> Output {
    snarkrs_env(dir, line, &[])
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// stderr without the tip, which snarkjs does not print.
fn err_less_tip(o: &Output) -> String {
    let e = err(o);
    match e.find("tip: ") {
        Some(i) => e[..i].to_string(),
        None => e,
    }
}

#[track_caller]
fn expect(o: &Output, code: i32, line: &str) {
    assert_eq!(
        o.status.code(),
        Some(code),
        "`{line}`:\n{}{}",
        out(o),
        err(o)
    );
}

fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    let Ok(o) = Command::new(&bin).arg("--help").output() else {
        eprintln!("SKIPPED: snarkjs is not installed");
        return None;
    };
    if !String::from_utf8_lossy(&o.stdout).contains("snarkjs@0.7.6") {
        eprintln!("SKIPPED: {bin} is not snarkjs 0.7.6; set SNARKJS");
        return None;
    }
    Some(bin)
}

/// snarkjs' exit code, stdout and stderr. Both streams go through files, because snarkjs
/// calls `process.exit` before a pipe has drained. Colours and the stack trace of a thrown
/// error are taken out: the trace names node_modules paths, and is not what a script
/// matches on.
fn snarkjs(bin: &str, dir: &Path, line: &str) -> (i32, String, String) {
    let (o, e) = (dir.join(".js.out"), dir.join(".js.err"));
    let status = Command::new(bin)
        .args(line.split_whitespace())
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&o).unwrap())
        .stderr(std::fs::File::create(&e).unwrap())
        .status()
        .unwrap();
    let read = |p: &Path| {
        strip_colour(&std::fs::read_to_string(p).unwrap())
            .lines()
            .filter(|l| !l.starts_with("    at "))
            .map(|l| format!("{l}\n"))
            .collect::<String>()
    };
    (status.code().unwrap_or(-1), read(&o), read(&e))
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

fn tiny_mul() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul");
    let ok = [
        "circuit_js/circuit.wasm",
        "input.json",
        "circuit.wtns",
        "circuit.zkey",
        "vkey.json",
    ]
    .iter()
    .all(|f| dir.join(f).is_file());
    if !ok {
        eprintln!("SKIPPED: no bench/artifacts/tiny_mul");
    }
    ok.then_some(dir)
}

/// A circuit with array inputs, `log()` and an assert, compiled by circom into `dir`.
fn logging_circuit(dir: &Path) -> Option<PathBuf> {
    std::fs::write(
        dir.join("t.circom"),
        r#"pragma circom 2.0.0;
template T() {
    signal input a;
    signal input x[2][3];
    signal output b;
    log("hello", a, x[1][2]);
    log(a);
    assert(a < 100);
    b <== a * x[0][0];
}
component main {public [a]} = T();
"#,
    )
    .unwrap();
    let o = Command::new("circom")
        .args(["t.circom", "--wasm", "--sym", "-o", "."])
        .current_dir(dir)
        .output();
    match o {
        Ok(o) if o.status.success() => Some(dir.join("t_js/t.wasm")),
        _ => {
            eprintln!("SKIPPED: circom is not installed or failed");
            None
        }
    }
}

const TIP: &str = "tip: for faster witness generation";

/// Defaults, bytes, and the tip: on stderr, silenced by the variable, stdout unchanged.
#[test]
fn wtns_calculate_writes_snarkjs_bytes_and_tips_on_stderr() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = scratch("calc");
    std::fs::copy(
        tiny.join("circuit_js/circuit.wasm"),
        dir.join("circuit.wasm"),
    )
    .unwrap();
    std::fs::copy(tiny.join("input.json"), dir.join("input.json")).unwrap();
    let want = std::fs::read(tiny.join("circuit.wtns")).unwrap();

    // circuit.wasm, input.json and witness.wtns are the defaults.
    let o = snarkrs(&dir, "wtns calculate");
    expect(&o, 0, "wtns calculate");
    assert_eq!(std::fs::read(dir.join("witness.wtns")).unwrap(), want);
    assert_eq!(out(&o), "", "stdout is snarkjs': nothing");
    assert!(err(&o).starts_with(TIP), "{}", err(&o));
    assert!(err(&o).contains("witness from memory"), "{}", err(&o));
    assert!(err(&o).lines().count() <= 3, "{}", err(&o));

    let o = snarkrs_env(
        &dir,
        "wc circuit.wasm input.json quiet.wtns",
        &[("SNARKRS_NO_TIPS", Path::new("1"))],
    );
    expect(&o, 0, "wc");
    assert_eq!((out(&o).as_str(), err(&o).as_str()), ("", ""));
    assert_eq!(std::fs::read(dir.join("quiet.wtns")).unwrap(), want);

    // `wtns debug` writes the same bytes, and needs the .sym.
    let o = snarkrs(&dir, "wd circuit.wasm input.json debug.wtns");
    expect(&o, 1, "wd without a sym");
    assert_eq!(
        out(&o),
        "[ERROR] snarkJS: Error: ENOENT: no such file or directory, open 'circuit.sym'\n"
    );
    std::fs::write(dir.join("circuit.sym"), "1,1,0,main.out\n").unwrap();
    let o = snarkrs(
        &dir,
        "wtns debug circuit.wasm input.json debug.wtns -g -s -t",
    );
    expect(&o, 0, "wd");
    assert_eq!(std::fs::read(dir.join("debug.wtns")).unwrap(), want);

    std::fs::remove_dir_all(&dir).ok();
}

/// A circuit's `log()` lines and its failures, line for line against snarkjs.
#[test]
fn witness_errors_and_logs_match_snarkjs() {
    let Some(js) = snarkjs_bin() else { return };
    let dir = scratch("errors");
    let Some(_) = logging_circuit(&dir) else {
        std::fs::remove_dir_all(&dir).ok();
        return;
    };
    for (what, input) in [
        ("good", r#"{"a":"3","x":[[1,2,3],[4,5,6]]}"#),
        ("negative", r#"{"a":"-1","x":[[1,2,"0x3"],[4,5,6]]}"#),
        ("missing input", r#"{"x":[[1,2,3],[4,5,6]]}"#),
        ("too few values", r#"{"a":"3","x":[[1,2,3],[4,5]]}"#),
        ("extra value", r#"{"a":"3","x":[[1,2,3],[4,5,6,7]]}"#),
        (
            "unknown signal",
            r#"{"a":"3","x":[[1,2,3],[4,5,6]],"zz":1}"#,
        ),
        ("assert", r#"{"a":"300","x":[[1,2,3],[4,5,6]]}"#),
        ("not an integer", r#"{"a":1.5,"x":[[1,2,3],[4,5,6]]}"#),
        ("not a number", r#"{"x":[[1,2,3],[4,5,6]],"a":"abc"}"#),
        ("null", r#"{"a":null,"x":[[1,2,3],[4,5,6]]}"#),
    ] {
        std::fs::write(dir.join("in.json"), input).unwrap();
        let o = snarkrs_env(
            &dir,
            "wc t_js/t.wasm in.json ours.wtns",
            &[("SNARKRS_NO_TIPS", Path::new("1"))],
        );
        let (code, js_out, js_err) = snarkjs(&js, &dir, "wc t_js/t.wasm in.json js.wtns");
        assert_eq!(
            o.status.code(),
            Some(code),
            "{what}: {}{}",
            out(&o),
            err(&o)
        );
        assert_eq!(out(&o), js_out, "{what}: stdout");
        assert_eq!(err(&o), js_err, "{what}: stderr");
        if code == 0 {
            assert_eq!(
                std::fs::read(dir.join("ours.wtns")).unwrap(),
                std::fs::read(dir.join("js.wtns")).unwrap(),
                "{what}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The proof from `fullprove` verifies under snarkjs, and with the wasm no witness file
/// is written anywhere: not in the working directory, not in the temp directory.
#[test]
fn fullprove_from_wasm_verifies_and_writes_no_witness() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = scratch("fullprove");
    let tmp = dir.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let wasm = tiny.join("circuit_js/circuit.wasm");
    let line = format!(
        "g16f {} {} {} proof.json public.json",
        tiny.join("input.json").display(),
        wasm.display(),
        tiny.join("circuit.zkey").display()
    );
    let o = snarkrs_env(&dir, &line, &[("TMPDIR", &tmp)]);
    expect(&o, 0, &line);
    assert_eq!(out(&o), "");
    assert!(err(&o).starts_with(TIP), "{}", err(&o));
    assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0, "temp dir");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["proof.json", "public.json", "tmp"]);
    let public = |p: &Path| -> Vec<String> {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    assert_eq!(
        public(&dir.join("public.json")),
        public(&tiny.join("public.json"))
    );

    std::fs::copy(tiny.join("vkey.json"), dir.join("vkey.json")).unwrap();
    let o = snarkrs(&dir, "g16v vkey.json public.json proof.json");
    assert_eq!(out(&o), "[INFO]  snarkJS: OK!\n");
    if let Some(js) = snarkjs_bin() {
        let (code, js_out, _) =
            snarkjs(&js, &dir, "groth16 verify vkey.json public.json proof.json");
        assert_eq!((code, js_out.as_str()), (0, "[INFO]  snarkJS: OK!\n"));
    }

    // An input the circuit rejects is snarkjs' error line and exit 1, and nothing is
    // written.
    std::fs::write(dir.join("bad.json"), r#"{"a":"3"}"#).unwrap();
    let line = format!(
        "g16f bad.json {} {} p2.json s2.json",
        wasm.display(),
        tiny.join("circuit.zkey").display()
    );
    let o = snarkrs(&dir, &line);
    expect(&o, 1, &line);
    assert!(
        out(&o).starts_with("[ERROR] snarkJS: Error: Not all inputs have been set."),
        "{}",
        out(&o)
    );
    assert!(!err(&o).contains(TIP));
    assert!(!dir.join("p2.json").exists());
    std::fs::remove_dir_all(&dir).ok();
}

/// A stand-in for circom's native binary that copies `fixture` to its output.
#[cfg(unix)]
fn stand_in(dir: &Path, name: &str, fixture: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = dir.join(name);
    std::fs::write(
        &bin,
        format!("#!/bin/sh\ncp '{}' \"$2\"\n", fixture.display()),
    )
    .unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(dir.join(format!("{name}.dat")), b"").unwrap();
    bin
}

/// The same commands with a native binary where the wasm was: detected by content, not
/// by name.
#[cfg(unix)]
#[test]
fn native_binaries_work_where_the_wasm_did() {
    let Some(tiny) = tiny_mul() else { return };
    let dir = scratch("native");
    let tmp = dir.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let bin = stand_in(&dir, "circuit", &tiny.join("circuit.wtns"));
    let input = tiny.join("input.json");
    let zkey = tiny.join("circuit.zkey");

    let line = format!("wc circuit {} w.wtns", input.display());
    let o = snarkrs(&dir, &line);
    expect(&o, 0, &line);
    assert!(err(&o).starts_with(TIP), "{}", err(&o));
    assert_eq!(
        std::fs::read(dir.join("w.wtns")).unwrap(),
        std::fs::read(tiny.join("circuit.wtns")).unwrap()
    );

    let line = format!(
        "groth16 fullprove {} {} {} proof.json public.json",
        input.display(),
        bin.display(),
        zkey.display()
    );
    let o = snarkrs_env(&dir, &line, &[("TMPDIR", &tmp)]);
    expect(&o, 0, &line);
    assert!(err(&o).starts_with(TIP), "{}", err(&o));
    assert_eq!(
        std::fs::read_dir(&tmp).unwrap().count(),
        0,
        "temp witness left"
    );
    std::fs::copy(tiny.join("vkey.json"), dir.join("vkey.json")).unwrap();
    let o = snarkrs(&dir, "g16v vkey.json public.json proof.json");
    assert_eq!(out(&o), "[INFO]  snarkJS: OK!\n");

    // A binary from another circuit: the witness has the wrong number of wires.
    let other = dir.join("other.wtns");
    let mut w = snarkrs_formats::wtns::Witness::load(&tiny.join("circuit.wtns"))
        .unwrap()
        .0;
    w.push(w[1]);
    std::fs::write(&other, snarkrs_witness::wtns_bytes(&w)).unwrap();
    let wrong = stand_in(&dir, "wrong", &other);
    let line = format!(
        "g16f {} {} {} p2.json s2.json",
        input.display(),
        wrong.display(),
        zkey.display()
    );
    let o = snarkrs_env(&dir, &line, &[("TMPDIR", &tmp)]);
    expect(&o, 1, &line);
    assert!(out(&o).contains("not the same circuit"), "{}", out(&o));
    assert!(!dir.join("p2.json").exists());
    assert_eq!(
        std::fs::read_dir(&tmp).unwrap().count(),
        0,
        "temp witness left"
    );

    // `wtns debug` is the wasm's alone.
    let line = format!("wd circuit {} d.wtns", input.display());
    let o = snarkrs(&dir, &line);
    expect(&o, 1, &line);
    assert!(
        out(&o).contains("wasm witness calculator only"),
        "{}",
        out(&o)
    );

    // Neither wasm nor executable.
    let line = format!("wc {} {} w.wtns", input.display(), input.display());
    let o = snarkrs(&dir, &line);
    expect(&o, 1, &line);
    assert!(out(&o).contains("neither a wasm module"), "{}", out(&o));
    std::fs::remove_dir_all(&dir).ok();
}

/// stderr of a failing native binary reaches the user, before our own error line.
#[cfg(unix)]
#[test]
fn a_native_failure_forwards_its_stderr() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("native-fail");
    std::fs::write(dir.join("input.json"), "{}").unwrap();
    let bin = dir.join("circuit");
    std::fs::write(
        &bin,
        "#!/bin/sh\necho 'Not all inputs have been set. Only 0 out of 2' >&2\nkill -ABRT $$\n",
    )
    .unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(dir.join("circuit.dat"), b"").unwrap();
    let o = snarkrs(&dir, "wc circuit input.json w.wtns");
    expect(&o, 1, "wc");
    assert!(err_less_tip(&o).contains("Only 0 out of 2"), "{}", err(&o));
    assert!(out(&o).contains("killed by SIGABRT"), "{}", out(&o));
    assert!(!err(&o).contains(TIP));
    assert!(!dir.join("w.wtns").exists());
    std::fs::remove_dir_all(&dir).ok();
}
