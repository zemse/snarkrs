//! The r1cs, witness and ptau commands this crate backs, diffed byte for byte against
//! snarkjs itself.
//!
//! Every test runs snarkjs and our function on the same input and compares the whole
//! output: the JSON or ptau file, or the text printed to stdout. Logged lines are framed
//! with [`Logplease`], so the comparison covers snarkjs' colour codes too.
//!
//! snarkjs is `$SNARKJS` when set, either an executable or a `cli.js` to run under node,
//! and `snarkjs` on `PATH` otherwise. The reference is 0.7.6, and each test prints the
//! version it ran against. Everything skips with a message when snarkjs or the artifacts
//! under `bench/artifacts` are absent. The circom-compiled cases additionally need
//! `circom` on `PATH`, since the artifacts carry no `.sym` and no custom gates.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use g16_ceremony::r1cs::R1cs;
use g16_ceremony::r1cs_export::{r1cs_export_json, r1cs_info, r1cs_print};
use g16_ceremony::snarkjs_log::Logplease;
use g16_ceremony::sym::Syms;
use g16_ceremony::wtns_check::wtns_check;
use g16_ceremony::CeremonyError;

/// Small enough that snarkjs finishes each in a second or two.
const VARIANTS: [&str; 2] = ["tiny_mul", "js_1x1_d8"];

fn artifacts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

fn snarkjs() -> Command {
    match std::env::var("SNARKJS") {
        Ok(p) if p.ends_with(".js") => {
            let mut c = Command::new("node");
            c.arg(p);
            c
        }
        Ok(p) => Command::new(p),
        Err(_) => Command::new("snarkjs"),
    }
}

/// Whether snarkjs runs at all, printing its version or the skip.
fn have_snarkjs(test: &str) -> bool {
    match snarkjs().arg("--version").output() {
        Ok(o) => {
            let banner = String::from_utf8_lossy(&o.stdout);
            eprintln!("{test}: {}", banner.lines().next().unwrap_or("snarkjs"));
            true
        }
        Err(_) => {
            eprintln!("SKIPPED {test}: snarkjs not found");
            false
        }
    }
}

fn variants(test: &str) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = VARIANTS
        .iter()
        .map(|v| artifacts_root().join(v))
        .filter(|d| d.join("circuit.r1cs").is_file())
        .collect();
    if dirs.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
    }
    dirs
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-ceremony-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run with stdout and stderr sent to files rather than pipes. snarkjs ends with
/// `process.exit`, and Node writes to a pipe asynchronously, so a pipe loses the tail of a
/// large print (`r1cs print` on js_1x1_d8 came back 460 KB short); writes to a file are
/// synchronous.
fn run(cmd: &mut Command) -> Output {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("snarkjs-out-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (out, err) = (dir.join("stdout"), dir.join("stderr"));
    let status = cmd
        .stdout(std::fs::File::create(&out).unwrap())
        .stderr(std::fs::File::create(&err).unwrap())
        .status()
        .unwrap_or_else(|e| panic!("failed to run {cmd:?}: {e}"));
    let o = Output {
        status,
        stdout: std::fs::read(&out).unwrap(),
        stderr: std::fs::read(&err).unwrap(),
    };
    let _ = std::fs::remove_dir_all(&dir);
    o
}

/// Run a snarkjs command that must succeed and return its stdout.
fn snarkjs_ok(args: &[&std::ffi::OsStr]) -> Vec<u8> {
    let o = run(snarkjs().args(args));
    assert!(o.status.success(), "snarkjs {args:?} failed: {o:?}");
    o.stdout
}

/// First differing byte, with context, so a mismatch in a megabyte of output is readable.
fn assert_same(what: &str, ours: &[u8], theirs: &[u8]) {
    if ours == theirs {
        return;
    }
    let at = ours
        .iter()
        .zip(theirs)
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(theirs.len()));
    let ctx = |b: &[u8]| {
        String::from_utf8_lossy(&b[at.saturating_sub(80).min(b.len())..(at + 80).min(b.len())])
            .into_owned()
    };
    panic!(
        "{what}: differs at byte {at} ({} vs {} bytes)\n--- ours\n{}\n--- snarkjs\n{}",
        ours.len(),
        theirs.len(),
        ctx(ours),
        ctx(theirs)
    );
}

/// A circuit with one custom gate, so sections 4 and 5 exist. circom accepts it for R1CS
/// output; snarkjs' setup would ignore the gate, but the exporters do not.
const CUSTOM_GATE_CIRCOM: &str = r#"pragma circom 2.0.6;
pragma custom_templates;

template custom Gate(k) {
    signal input a;
    signal input b;
    signal output c;
    c <-- a * b + k;
}

template Main() {
    signal input a;
    signal input b;
    signal output c;
    signal t;
    t <== a * b;
    component g = Gate(7);
    g.a <== t;
    g.b <== b;
    c <== g.c;
}
component main {public [a]} = Main();
"#;

/// `(r1cs, sym)` pairs circom compiles into `out`: the tiny_mul source and the custom-gate
/// circuit. Empty, with a message, when circom is not installed.
fn circom_circuits(test: &str, out: &Path) -> Vec<(PathBuf, PathBuf)> {
    if Command::new("circom").arg("--version").output().is_err() {
        eprintln!("{test}: circom not found, skipping the compiled circuits");
        return Vec::new();
    }
    let mut sources = vec![("custom_gate".to_string(), CUSTOM_GATE_CIRCOM.to_string())];
    let tiny = artifacts_root().join("tiny_mul/circuit.circom");
    if let Ok(src) = std::fs::read_to_string(tiny) {
        sources.push(("tiny_mul".to_string(), src));
    }
    sources
        .into_iter()
        .map(|(name, src)| {
            let file = out.join(format!("{name}.circom"));
            std::fs::write(&file, src).unwrap();
            let o = run(Command::new("circom")
                .arg(&file)
                .args(["--r1cs", "--sym", "-o"])
                .arg(out));
            assert!(o.status.success(), "circom {name} failed: {o:?}");
            (
                out.join(format!("{name}.r1cs")),
                out.join(format!("{name}.sym")),
            )
        })
        .collect()
}

#[test]
fn r1cs_info_matches_snarkjs() {
    if !have_snarkjs("r1cs_info_matches_snarkjs") {
        return;
    }
    for dir in variants("r1cs_info_matches_snarkjs") {
        let path = dir.join("circuit.r1cs");
        let theirs = snarkjs_ok(&["r1cs".as_ref(), "info".as_ref(), path.as_os_str()]);
        let mut log = Logplease::new(Vec::new());
        r1cs_info(&R1cs::open(&path).unwrap(), &mut log).unwrap();
        assert_same(&format!("{} r1cs info", path.display()), &log.out, &theirs);
    }
}

#[test]
fn r1cs_export_json_matches_snarkjs() {
    if !have_snarkjs("r1cs_export_json_matches_snarkjs") {
        return;
    }
    let out = scratch("r1cs-json");
    let mut r1cs_files: Vec<PathBuf> = variants("r1cs_export_json_matches_snarkjs")
        .into_iter()
        .map(|d| d.join("circuit.r1cs"))
        .collect();
    r1cs_files.extend(
        circom_circuits("r1cs_export_json_matches_snarkjs", &out)
            .into_iter()
            .map(|(r1cs, _)| r1cs),
    );
    for path in r1cs_files {
        let theirs = out.join("snarkjs.json");
        snarkjs_ok(&[
            "r1cs".as_ref(),
            "export".as_ref(),
            "json".as_ref(),
            path.as_os_str(),
            theirs.as_os_str(),
        ]);
        let mut ours = Vec::new();
        r1cs_export_json(&R1cs::open(&path).unwrap(), &mut ours).unwrap();
        assert_same(
            &format!("{} r1cs export json", path.display()),
            &ours,
            &std::fs::read(&theirs).unwrap(),
        );
    }
}

/// A `.sym` for the artifact circuits that exercises every quirk of `loadsyms.js` the
/// printer can see: a CRLF line, an alias appended with `|`, a line for wire 0, a name
/// with a comma (dropped), a padded index (never looked up), an empty name replaced by a
/// later one, and wires left unnamed so they print as `undefined`.
const QUIRKY_SYM: &str = "\u{feff}1,1,0,main.out\n2,2,0,main.a\r\n3,2,0,main.alias\n\
    4,-1,0,main.gone\n5,03,0,main.padded\n6,3,0,has,comma\n7,4,0,\n8,4,0,main.late\n\
    9,0,0,main.const\n";

#[test]
fn r1cs_print_matches_snarkjs() {
    if !have_snarkjs("r1cs_print_matches_snarkjs") {
        return;
    }
    let out = scratch("r1cs-print");
    let quirky = out.join("quirky.sym");
    std::fs::write(&quirky, QUIRKY_SYM).unwrap();
    let mut cases: Vec<(PathBuf, PathBuf)> = variants("r1cs_print_matches_snarkjs")
        .into_iter()
        .map(|d| (d.join("circuit.r1cs"), quirky.clone()))
        .collect();
    cases.extend(circom_circuits("r1cs_print_matches_snarkjs", &out));
    for (r1cs, sym) in cases {
        let theirs = snarkjs_ok(&[
            "r1cs".as_ref(),
            "print".as_ref(),
            r1cs.as_os_str(),
            sym.as_os_str(),
        ]);
        let mut log = Logplease::new(Vec::new());
        r1cs_print(
            &R1cs::open(&r1cs).unwrap(),
            &Syms::load(&sym).unwrap(),
            &mut log,
        )
        .unwrap();
        assert_same(
            &format!("{} with {} r1cs print", r1cs.display(), sym.display()),
            &log.out,
            &theirs,
        );
    }
}

/// The witness bytes with value `k` of section 2 rewritten by `f`, which gets its 32 bytes.
fn edit_witness(wtns: &[u8], k: usize, f: impl FnOnce(&mut [u8])) -> Vec<u8> {
    let file = g16_zkey::binfile::BinFile::from_bytes(wtns.to_vec(), b"wtns", 2).unwrap();
    let start = file.sections().iter().find(|s| s.id == 2).unwrap().start;
    let mut out = wtns.to_vec();
    f(&mut out[start + 32 * k..start + 32 * (k + 1)]);
    out
}

/// `v + r` in place, which fits in 256 bits for any `v < r` and is the same field element.
fn add_modulus(v: &mut [u8]) {
    let r = g16_ceremony::r_le();
    let mut carry = 0u16;
    for (b, m) in v.iter_mut().zip(r) {
        let s = *b as u16 + m as u16 + carry;
        *b = s as u8;
        carry = s >> 8;
    }
    assert_eq!(carry, 0);
}

/// A good witness passes, one with a wire nudged by one fails at the constraint snarkjs
/// names, and one holding `v + r` for a wire passes, because both sides reduce it.
#[test]
fn wtns_check_matches_snarkjs() {
    if !have_snarkjs("wtns_check_matches_snarkjs") {
        return;
    }
    let out = scratch("wtns-check");
    for dir in variants("wtns_check_matches_snarkjs") {
        let r1cs = dir.join("circuit.r1cs");
        let good = std::fs::read(dir.join("circuit.wtns")).unwrap();
        let n = (good.len() - 76) / 32;
        let cases = [
            ("good", good.clone(), true),
            (
                "nudged",
                edit_witness(&good, n - 1, |v| v[0] = v[0].wrapping_add(1)),
                false,
            ),
            ("plus_r", edit_witness(&good, n - 1, add_modulus), true),
        ];
        for (name, bytes, want) in cases {
            let wtns = out.join(format!("{name}.wtns"));
            std::fs::write(&wtns, bytes).unwrap();
            let o = run(snarkjs().args([
                "wtns".as_ref(),
                "check".as_ref(),
                r1cs.as_os_str(),
                wtns.as_os_str(),
            ]));
            assert_eq!(o.status.success(), want, "{name}: snarkjs said {o:?}");
            let mut log = Logplease::new(Vec::new());
            let got = wtns_check(&r1cs, &wtns, &mut log).unwrap();
            assert_eq!(got, want, "{} {name}", dir.display());
            assert_same(
                &format!("{} {name} wtns check", dir.display()),
                &log.out,
                &o.stdout,
            );
        }
    }
}

/// A witness for another field is refused before any constraint is read, with the lines
/// snarkjs logs up to its throw and its message.
#[test]
fn wtns_check_refuses_another_curve_like_snarkjs() {
    if !have_snarkjs("wtns_check_refuses_another_curve_like_snarkjs") {
        return;
    }
    let Some(dir) = variants("wtns_check_refuses_another_curve_like_snarkjs")
        .into_iter()
        .next()
    else {
        return;
    };
    let out = scratch("wtns-curve");
    let r1cs = dir.join("circuit.r1cs");
    let mut bytes = std::fs::read(dir.join("circuit.wtns")).unwrap();
    // Section 1's payload starts at 24: `n8`, then the prime.
    bytes[28] ^= 1;
    let wtns = out.join("other.wtns");
    std::fs::write(&wtns, bytes).unwrap();
    let o = run(snarkjs().args([
        "wtns".as_ref(),
        "check".as_ref(),
        r1cs.as_os_str(),
        wtns.as_os_str(),
    ]));
    assert!(!o.status.success());
    let mut log = Logplease::new(Vec::new());
    let err = wtns_check(&r1cs, &wtns, &mut log).unwrap_err();
    assert!(matches!(err, CeremonyError::WitnessCurveMismatch), "{err}");
    let theirs = String::from_utf8_lossy(&o.stdout);
    let ours = String::from_utf8(log.out).unwrap();
    assert!(
        theirs.starts_with(&ours),
        "ours:\n{ours}\nsnarkjs:\n{theirs}"
    );
    assert!(
        theirs.contains(&format!("Error: {err}")),
        "snarkjs:\n{theirs}"
    );
}
