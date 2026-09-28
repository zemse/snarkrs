//! bellman `MPCParameters` against snarkjs 0.7.6.
//!
//! `zkey export bellman` and `zkey import bellman` are deterministic and are held to
//! snarkjs' bytes. `zkey bellman contribute` is too once `getRandomRng`'s 64 OS bytes are
//! pinned, which is done by running snarkjs' module with `crypto.randomFillSync` stubbed,
//! as `tests/challenge.rs` does for phase 1. Then, with real randomness, each side imports
//! the other's response and both `zkey verify`s accept the key and name its contributions.
//!
//! Needs snarkjs 0.7.6 (`SNARKJS` names the binary), the tiny_mul and js_1x1_d8
//! artifacts, and `bench/ptau/local_13.ptau` for the verify half; skips with a message
//! when any is absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use snarkrs_ceremony::bellman;
use snarkrs_ceremony::contribute;
use snarkrs_ceremony::transcript::rng_from_entropy_with;
use snarkrs_ceremony::CpuKeyScale;
use snarkrs_msm::CpuMsm;

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench")
}

fn os_bytes(seed: u8) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, b) in out.iter_mut().enumerate() {
        *b = (i as u8).wrapping_add(seed);
    }
    out
}

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

fn snarkjs_pkg(bin: &str) -> Option<PathBuf> {
    let path = if bin.contains('/') {
        PathBuf::from(bin)
    } else {
        let out = Command::new("which").arg(bin).output().ok()?;
        PathBuf::from(String::from_utf8_lossy(&out.stdout).trim())
    };
    let pkg = std::fs::canonicalize(path)
        .ok()?
        .parent()?
        .parent()?
        .to_path_buf();
    pkg.join("src/zkey_bellman_contribute.js")
        .is_file()
        .then_some(pkg)
}

fn snarkjs(bin: &str, args: &[&str]) -> String {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running snarkjs {args:?}: {e}"));
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "snarkjs {args:?} failed:\n{log}");
    log
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-ceremony-bellman-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn assert_same_bytes(what: &str, ours: &Path, theirs: &Path) {
    let got = std::fs::read(ours).unwrap();
    let want = std::fs::read(theirs).unwrap();
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: {} bytes written, snarkjs wrote {}",
        got.len(),
        want.len()
    );
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what}: first difference at byte {i}, 0x{:02x} vs 0x{:02x}",
            got[i], want[i]
        );
    }
}

/// An init key for tiny_mul over local_13 and one named `zkey contribute` on top, so the
/// exported chain has a record whose name import must carry forward.
fn fixtures(dir: &Path) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let r1cs = bench().join("artifacts/tiny_mul/circuit.r1cs");
    let ptau = bench().join("ptau/local_13.ptau");
    if !r1cs.is_file() || !ptau.is_file() {
        eprintln!("SKIPPED: no tiny_mul r1cs or local_13 ptau");
        return None;
    }
    let init = dir.join("init.zkey");
    let z1 = dir.join("z1.zkey");
    snarkrs_ceremony::setup::setup(&r1cs, &ptau, &init, &CpuMsm::new()).unwrap();
    contribute::contribute_with(
        &init,
        &z1,
        Some("first"),
        rng_from_entropy_with(&os_bytes(0), "fixture"),
        &CpuKeyScale,
    )
    .unwrap();
    Some((ptau, init, z1))
}

fn assert_verifies(bin: &str, ptau: &Path, init: &Path, zkey: &Path, names: &[&str]) {
    let report = contribute::verify_from_init(init, ptau, zkey, &CpuMsm::new()).unwrap();
    assert_eq!(
        report.contribution_hashes.len(),
        names.len(),
        "ours on {zkey:?}"
    );
    let log = snarkjs(bin, &["zkvi", s(init), s(ptau), s(zkey)]);
    assert!(log.contains("ZKey Ok!"), "snarkjs on {zkey:?}:\n{log}");
    for name in names {
        assert!(log.contains(name), "snarkjs does not list {name}:\n{log}");
    }
}

#[test]
fn export_bellman_is_byte_identical_to_snarkjs() {
    let Some(bin) = snarkjs_bin() else { return };
    let inputs: Vec<PathBuf> = [
        "artifacts/tiny_mul/c0.zkey",
        "artifacts/tiny_mul/circuit.zkey",
        "artifacts/js_1x1_d8/circuit.zkey",
    ]
    .iter()
    .map(|p| bench().join(p))
    .filter(|p| p.is_file())
    .collect();
    if inputs.is_empty() {
        eprintln!("SKIPPED: no artifacts");
        return;
    }
    let dir = tmp_dir("export");
    for (i, zkey) in inputs.iter().enumerate() {
        let ours = dir.join(format!("{i}.ours.mpcparams"));
        let theirs = dir.join(format!("{i}.snarkjs.mpcparams"));
        bellman::export_bellman(zkey, &ours).unwrap();
        snarkjs(&bin, &["zkey", "export", "bellman", s(zkey), s(&theirs)]);
        assert_same_bytes(&format!("export bellman {zkey:?}"), &ours, &theirs);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// With snarkjs' OS bytes pinned our response is its response, and importing it gives
/// the same zkey, which both verifiers accept.
#[test]
fn contribute_and_import_are_byte_identical_to_snarkjs() {
    let Some(bin) = snarkjs_bin() else { return };
    let dir = tmp_dir("pinned");
    let Some((ptau, init, z1)) = fixtures(&dir) else {
        return;
    };
    let mp = dir.join("z1.mpcparams");
    bellman::export_bellman(&z1, &mp).unwrap();

    let resp = dir.join("resp.ours.mpcparams");
    bellman::bellman_contribute_with(
        &mp,
        &resp,
        rng_from_entropy_with(&os_bytes(5), "bellman entropy"),
        &CpuKeyScale,
    )
    .unwrap();

    match snarkjs_pkg(&bin) {
        None => eprintln!("snarkjs package directory not found: response bytes not compared"),
        Some(pkg) => {
            let theirs = dir.join("resp.snarkjs.mpcparams");
            let script = dir.join("contribute.mjs");
            std::fs::write(
                &script,
                format!(
                    r#"import crypto from "crypto";
const OS = new Uint8Array(64); for (let i = 0; i < 64; i++) OS[i] = (i + 5) & 0xff;
crypto.randomFillSync = (a) => {{ a.set(OS.subarray(0, a.length)); return a; }};
const {{ getCurveFromName }} = await import("{pkg}/src/curves.js");
const bc = (await import("{pkg}/src/zkey_bellman_contribute.js")).default;
const curve = await getCurveFromName("bn128");
await bc(curve, "{mp}", "{out}", "bellman entropy");
await curve.terminate();
"#,
                    pkg = pkg.display(),
                    mp = mp.display(),
                    out = theirs.display(),
                ),
            )
            .unwrap();
            let out = Command::new("node").arg(&script).output().unwrap();
            assert!(
                out.status.success(),
                "node: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert_same_bytes("bellman contribute", &resp, &theirs);
        }
    }

    let ours = dir.join("z2.ours.zkey");
    let theirs = dir.join("z2.snarkjs.zkey");
    let rep = bellman::import_bellman(&z1, &resp, &ours, Some("via bellman")).unwrap();
    assert_eq!((rep.n_prior, rep.contribution_hashes.len()), (1, 2));
    snarkjs(
        &bin,
        &[
            "zkey",
            "import",
            "bellman",
            s(&z1),
            s(&resp),
            s(&theirs),
            "-n=via bellman",
        ],
    );
    assert_same_bytes("import bellman", &ours, &theirs);
    assert_verifies(&bin, &ptau, &init, &ours, &["first", "via bellman"]);

    std::fs::remove_dir_all(&dir).ok();
}

/// Real randomness: each side imports the other's response, and both verifiers accept.
#[test]
fn responses_cross_import_between_snarkjs_and_ours() {
    let Some(bin) = snarkjs_bin() else { return };
    let dir = tmp_dir("cross");
    let Some((ptau, init, z1)) = fixtures(&dir) else {
        return;
    };
    let mp = dir.join("z1.mpcparams");
    bellman::export_bellman(&z1, &mp).unwrap();

    let r_theirs = dir.join("theirs.mpcparams");
    snarkjs(
        &bin,
        &[
            "zkey",
            "bellman",
            "contribute",
            "bn128",
            s(&mp),
            s(&r_theirs),
            "-e=snarkjs entropy",
        ],
    );
    let z_ours = dir.join("imported_by_us.zkey");
    bellman::import_bellman(&z1, &r_theirs, &z_ours, Some("from snarkjs")).unwrap();
    assert_verifies(&bin, &ptau, &init, &z_ours, &["first", "from snarkjs"]);

    let r_ours = dir.join("ours.mpcparams");
    bellman::bellman_contribute(&mp, &r_ours, "our entropy", &CpuKeyScale).unwrap();
    let z_theirs = dir.join("imported_by_snarkjs.zkey");
    snarkjs(
        &bin,
        &[
            "zkey",
            "import",
            "bellman",
            s(&z1),
            s(&r_ours),
            s(&z_theirs),
            "-n=from us",
        ],
    );
    assert_verifies(&bin, &ptau, &init, &z_theirs, &["first", "from us"]);

    // A response built on a different chain is refused: this sibling of z1 has a
    // different first contribution.
    let sibling = dir.join("sibling.zkey");
    contribute::contribute_with(
        &init,
        &sibling,
        Some("first"),
        rng_from_entropy_with(&os_bytes(1), "fixture"),
        &CpuKeyScale,
    )
    .unwrap();
    let err = bellman::import_bellman(&sibling, &r_ours, &dir.join("x.zkey"), None).unwrap_err();
    assert!(err.to_string().contains("previous contribution 0"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}
