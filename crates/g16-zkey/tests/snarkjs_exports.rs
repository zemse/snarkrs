//! The export commands this crate backs, diffed byte for byte against snarkjs itself.
//!
//! Every test runs snarkjs and our function on the same input and compares the whole
//! output: the JSON file, or the text printed to stdout and stderr. A format has no
//! partial credit, since whatever consumes the file compares it the same way.
//!
//! snarkjs is `$SNARKJS` when set, either an executable or a `cli.js` to run under node,
//! and `snarkjs` on `PATH` otherwise. The reference is 0.7.6, and any other version skips,
//! as does everything when snarkjs or the artifacts under `bench/artifacts` are absent, so
//! a fresh clone does not report false green.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use g16_zkey::calldata::groth16_solidity_calldata;
use g16_zkey::export_json::{wtns_export_json, zkey_export_json};
use g16_zkey::file_info::file_info;

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

/// Whether the reference snarkjs runs, printing the skip when it does not. Only 0.7.6
/// counts: the outputs are pinned to it, and 0.7.2 already differs (`wtns check` there
/// logs "Ouputs:").
fn have_snarkjs(test: &str) -> bool {
    match snarkjs().arg("--version").output() {
        Ok(o) if String::from_utf8_lossy(&o.stdout).starts_with("snarkjs@0.7.6") => true,
        Ok(o) => {
            let banner = String::from_utf8_lossy(&o.stdout);
            eprintln!(
                "SKIPPED {test}: found {}, need snarkjs@0.7.6 (point $SNARKJS at its cli.js)",
                banner.lines().next().unwrap_or("an unknown snarkjs")
            );
            false
        }
        Err(_) => {
            eprintln!("SKIPPED {test}: snarkjs not found");
            false
        }
    }
}

/// The variants present, or a skip message and nothing.
fn variants(test: &str) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = VARIANTS
        .iter()
        .map(|v| artifacts_root().join(v))
        .filter(|d| d.join("circuit.zkey").is_file())
        .collect();
    if dirs.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
    }
    dirs
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-zkey-{test}-{}", std::process::id()));
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

/// First differing byte, with context, so a mismatch in a megabyte of JSON is readable.
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
        String::from_utf8_lossy(&b[at.saturating_sub(80)..(at + 80).min(b.len())]).into_owned()
    };
    panic!(
        "{what}: differs at byte {at} ({} vs {} bytes)\n--- ours\n{}\n--- snarkjs\n{}",
        ours.len(),
        theirs.len(),
        ctx(ours),
        ctx(theirs)
    );
}

#[test]
fn zkey_export_json_matches_snarkjs() {
    if !have_snarkjs("zkey_export_json_matches_snarkjs") {
        return;
    }
    let out = scratch("zkey-json");
    for dir in variants("zkey_export_json_matches_snarkjs") {
        let theirs = out.join("snarkjs.json");
        let o = run(snarkjs().args([
            "zkey".as_ref(),
            "export".as_ref(),
            "json".as_ref(),
            dir.join("circuit.zkey").as_os_str(),
            theirs.as_os_str(),
        ]));
        assert!(o.status.success(), "snarkjs failed: {o:?}");
        let mut ours = Vec::new();
        zkey_export_json(&dir.join("circuit.zkey"), &mut ours).unwrap();
        assert_same(
            &format!("{} zkey export json", dir.display()),
            &ours,
            &std::fs::read(&theirs).unwrap(),
        );
    }
}

#[test]
fn wtns_export_json_matches_snarkjs() {
    if !have_snarkjs("wtns_export_json_matches_snarkjs") {
        return;
    }
    let out = scratch("wtns-json");
    for dir in variants("wtns_export_json_matches_snarkjs") {
        let theirs = out.join("snarkjs.json");
        let o = run(snarkjs().args([
            "wtns".as_ref(),
            "export".as_ref(),
            "json".as_ref(),
            dir.join("circuit.wtns").as_os_str(),
            theirs.as_os_str(),
        ]));
        assert!(o.status.success(), "snarkjs failed: {o:?}");
        let mut ours = Vec::new();
        wtns_export_json(&dir.join("circuit.wtns"), &mut ours).unwrap();
        assert_same(
            &format!("{} wtns export json", dir.display()),
            &ours,
            &std::fs::read(&theirs).unwrap(),
        );
    }
}

#[test]
fn soliditycalldata_matches_snarkjs() {
    if !have_snarkjs("soliditycalldata_matches_snarkjs") {
        return;
    }
    for dir in variants("soliditycalldata_matches_snarkjs") {
        let (public, proof) = (dir.join("public.json"), dir.join("proof.json"));
        let o = run(snarkjs().args([
            "zkey".as_ref(),
            "export".as_ref(),
            "soliditycalldata".as_ref(),
            public.as_os_str(),
            proof.as_os_str(),
        ]));
        assert!(o.status.success(), "snarkjs failed: {o:?}");
        let read = |p: &Path| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let ours = groth16_solidity_calldata(&read(&proof), &read(&public)).unwrap() + "\n";
        assert_same(
            &format!("{} soliditycalldata", dir.display()),
            ours.as_bytes(),
            &o.stdout,
        );
    }
}

/// `file info` on every kind of binfile, and on the broken shapes it exists to describe:
/// a payload running off the end, a header cut mid-length, a version cut mid-word, a
/// duplicated and an empty section, the wrong magic, a missing file, a bad extension.
#[test]
fn file_info_matches_snarkjs() {
    if !have_snarkjs("file_info_matches_snarkjs") {
        return;
    }
    let dirs = variants("file_info_matches_snarkjs");
    let Some(dir) = dirs.first() else { return };
    let out = scratch("file-info");
    let zkey = std::fs::read(dir.join("circuit.zkey")).unwrap();

    let mut cases: Vec<PathBuf> = ["circuit.zkey", "circuit.r1cs", "circuit.wtns"]
        .iter()
        .map(|f| dir.join(f))
        .collect();
    let mut write = |name: &str, bytes: &[u8]| {
        let p = out.join(name);
        std::fs::write(&p, bytes).unwrap();
        cases.push(p);
    };
    write("payload_cut.zkey", &zkey[..1000]);
    write("length_cut.zkey", &zkey[..20]);
    write("version_cut.zkey", &zkey[..6]);
    write("empty.zkey", &[]);
    write(
        "magic.zkey",
        &std::fs::read(dir.join("circuit.wtns")).unwrap(),
    );
    // Section 1 twice, then a zero-length section 5.
    let mut dup = b"zkey".to_vec();
    dup.extend_from_slice(&1u32.to_le_bytes());
    dup.extend_from_slice(&3u32.to_le_bytes());
    for (id, body) in [(1u32, &[1u8, 0, 0, 0][..]), (1, &[2, 0, 0, 0]), (5, &[])] {
        dup.extend_from_slice(&id.to_le_bytes());
        dup.extend_from_slice(&(body.len() as u64).to_le_bytes());
        dup.extend_from_slice(body);
    }
    write("dup.zkey", &dup);
    cases.push(out.join("missing.zkey"));
    cases.push(out.join("notes.txt"));

    for case in cases {
        let name = case.to_str().unwrap();
        let o = run(snarkjs().args(["file", "info", name]));
        assert!(o.status.success(), "snarkjs failed: {o:?}");
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        file_info(name, &mut stdout, &mut stderr).unwrap();
        assert_same(&format!("{name} stdout"), &stdout, &o.stdout);
        assert_same(&format!("{name} stderr"), &stderr, &o.stderr);
    }
}
