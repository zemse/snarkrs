//! The challenge/response exchange against snarkjs 0.7.6.
//!
//! `export challenge` and `import response` are deterministic, so they are held to
//! snarkjs' bytes directly. `challenge contribute` is deterministic once the 64 OS-random
//! bytes `getRandomRng` mixes in are pinned, which snarkjs' CLI cannot do, so its side is
//! run through the module entry point with `crypto.randomFillSync` stubbed, the same way
//! the `tests/phase1.rs` oracles were produced. With real randomness on both sides, each
//! implementation then imports the other's response and both `powersoftau verify`s accept
//! the result.
//!
//! Every test needs snarkjs 0.7.6 (`SNARKJS` names the binary) and skips with a message
//! without it. The pinned-RNG test also needs the package directory, found by resolving
//! the binary's symlink to `snarkjs/build/cli.cjs`.

use std::path::{Path, PathBuf};
use std::process::Command;

use g16_ceremony::challenge::{self, NO_HASH};
use g16_ceremony::phase1;
use g16_ceremony::ptau::Ptau;
use g16_ceremony::transcript::rng_from_entropy_with;
use g16_ceremony::{ContributionParams, CpuKeyScale};

const POWER: u32 = 3;

fn os_bytes(seed: u8) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, b) in out.iter_mut().enumerate() {
        *b = (i as u8).wrapping_add(seed);
    }
    out
}

/// snarkjs 0.7.6, or `None` with the reason printed.
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

/// The package directory behind the binary, for importing `src/*.js` directly.
fn snarkjs_pkg(bin: &str) -> Option<PathBuf> {
    let path = if bin.contains('/') {
        PathBuf::from(bin)
    } else {
        let out = Command::new("which").arg(bin).output().ok()?;
        PathBuf::from(String::from_utf8_lossy(&out.stdout).trim())
    };
    let real = std::fs::canonicalize(path).ok()?;
    let pkg = real.parent()?.parent()?.to_path_buf();
    pkg.join("src/powersoftau_challenge_contribute.js")
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
        "g16-ceremony-challenge-{test}-{}",
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

/// A fresh ptau and one ordinary contribution on top of it, both ours; both are already
/// pinned byte-identical to snarkjs by `tests/phase1.rs`.
fn fixtures(dir: &Path) -> (PathBuf, PathBuf) {
    let p0 = dir.join("p0.ptau");
    let p1 = dir.join("p1.ptau");
    phase1::ptau_new(POWER, &p0).unwrap();
    phase1::contribute_with(
        &p0,
        &p1,
        ContributionParams {
            name: Some("first".into()),
            ..Default::default()
        },
        rng_from_entropy_with(&os_bytes(0), "fixture"),
        &CpuKeyScale,
    )
    .unwrap();
    (p0, p1)
}

/// Both implementations' `powersoftau verify` accept `ptau`, and snarkjs names `names`.
fn assert_verifies(bin: &str, ptau: &Path, names: &[&str]) {
    let report = phase1::verify(ptau).unwrap();
    assert_eq!(
        report.contribution_hashes.len(),
        names.len(),
        "ours on {ptau:?}"
    );
    let log = snarkjs(bin, &["powersoftau", "verify", s(ptau)]);
    assert!(
        log.contains("Powers of Tau Ok!"),
        "snarkjs on {ptau:?}:\n{log}"
    );
    for name in names {
        assert!(log.contains(name), "snarkjs does not list {name}:\n{log}");
    }
}

#[test]
fn export_challenge_is_byte_identical_to_snarkjs() {
    let Some(bin) = snarkjs_bin() else { return };
    let dir = tmp_dir("export");
    let (p0, p1) = fixtures(&dir);
    for (what, ptau) in [("fresh", &p0), ("contributed", &p1)] {
        let ours = dir.join(format!("{what}.ours.challenge"));
        let theirs = dir.join(format!("{what}.snarkjs.challenge"));
        let got = challenge::export_challenge(ptau, &ours).unwrap();
        snarkjs(
            &bin,
            &["powersoftau", "export", "challenge", s(ptau), s(&theirs)],
        );
        assert_same_bytes(what, &ours, &theirs);
        assert_eq!(
            got.challenge_hash,
            Ptau::open(ptau).unwrap().last_challenge().unwrap()
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Our response to a challenge equals snarkjs' for the same OS bytes and entropy, and
/// importing it (with points, with `-nopoints`, and a second import onto the `-nopoints`
/// file, which patches the previous record) is byte-identical in every case.
#[test]
fn contribute_and_import_are_byte_identical_to_snarkjs() {
    let Some(bin) = snarkjs_bin() else { return };
    let dir = tmp_dir("pinned");
    let (_, p1) = fixtures(&dir);

    let c1 = dir.join("c1.challenge");
    challenge::export_challenge(&p1, &c1).unwrap();
    let r1 = dir.join("r1.ours.response");
    let resp = challenge::challenge_contribute_with(
        &c1,
        &r1,
        rng_from_entropy_with(&os_bytes(7), "pinned entropy"),
        &CpuKeyScale,
    )
    .unwrap();
    assert_eq!(resp.power, POWER);

    match snarkjs_pkg(&bin) {
        None => eprintln!("snarkjs package directory not found: response bytes not compared"),
        Some(pkg) => {
            let r1_theirs = dir.join("r1.snarkjs.response");
            let script = dir.join("contribute.mjs");
            std::fs::write(
                &script,
                format!(
                    r#"import crypto from "crypto";
const OS = new Uint8Array(64); for (let i = 0; i < 64; i++) OS[i] = (i + 7) & 0xff;
crypto.randomFillSync = (a) => {{ a.set(OS.subarray(0, a.length)); return a; }};
const {{ getCurveFromName }} = await import("{pkg}/src/curves.js");
const cc = (await import("{pkg}/src/powersoftau_challenge_contribute.js")).default;
const curve = await getCurveFromName("bn128");
await cc(curve, "{c1}", "{r1}", "pinned entropy");
await curve.terminate();
"#,
                    pkg = pkg.display(),
                    c1 = c1.display(),
                    r1 = r1_theirs.display(),
                ),
            )
            .unwrap();
            let out = Command::new("node").arg(&script).output().unwrap();
            assert!(
                out.status.success(),
                "node: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert_same_bytes("challenge contribute", &r1, &r1_theirs);
        }
    }

    // With points, and named.
    let ours = dir.join("p2.ours.ptau");
    let theirs = dir.join("p2.snarkjs.ptau");
    let rep = challenge::import_response(&p1, &r1, &ours, Some("second"), true).unwrap();
    assert_eq!(rep.response_hash, resp.response_hash);
    snarkjs(
        &bin,
        &[
            "powersoftau",
            "import",
            "response",
            s(&p1),
            s(&r1),
            s(&theirs),
            "-n=second",
        ],
    );
    assert_same_bytes("import response", &ours, &theirs);
    assert_verifies(&bin, &ours, &["first", "second"]);

    // `-nopoints`, unnamed.
    let np_ours = dir.join("p2np.ours.ptau");
    let np_theirs = dir.join("p2np.snarkjs.ptau");
    let rep = challenge::import_response(&p1, &r1, &np_ours, None, false).unwrap();
    assert_eq!(rep.next_challenge, NO_HASH);
    snarkjs(
        &bin,
        &[
            "powersoftau",
            "import",
            "response",
            s(&p1),
            s(&r1),
            s(&np_theirs),
            "-nopoints",
        ],
    );
    assert_same_bytes("import response -nopoints", &np_ours, &np_theirs);

    // The next round comes from the full file and is imported onto the `-nopoints` one.
    let c2 = dir.join("c2.challenge");
    challenge::export_challenge(&ours, &c2).unwrap();
    let r2 = dir.join("r2.response");
    challenge::challenge_contribute_with(
        &c2,
        &r2,
        rng_from_entropy_with(&os_bytes(9), "round two"),
        &CpuKeyScale,
    )
    .unwrap();
    let np3_ours = dir.join("p3np.ours.ptau");
    let np3_theirs = dir.join("p3np.snarkjs.ptau");
    challenge::import_response(&np_ours, &r2, &np3_ours, Some("third"), false).unwrap();
    snarkjs(
        &bin,
        &[
            "powersoftau",
            "import",
            "response",
            s(&np_theirs),
            s(&r2),
            s(&np3_theirs),
            "-nopoints",
            "-n=third",
        ],
    );
    assert_same_bytes("import onto a -nopoints file", &np3_ours, &np3_theirs);

    std::fs::remove_dir_all(&dir).ok();
}

/// Real randomness on both sides: each imports the other's response, and both verifiers
/// accept both results and list every contribution.
#[test]
fn responses_cross_import_between_snarkjs_and_ours() {
    let Some(bin) = snarkjs_bin() else { return };
    let dir = tmp_dir("cross");
    let (p0, _) = fixtures(&dir);
    let c0 = dir.join("c0.challenge");
    challenge::export_challenge(&p0, &c0).unwrap();

    // snarkjs responds, we import.
    let r_theirs = dir.join("theirs.response");
    snarkjs(
        &bin,
        &[
            "powersoftau",
            "challenge",
            "contribute",
            "bn128",
            s(&c0),
            s(&r_theirs),
            "-e=snarkjs entropy",
        ],
    );
    let p_ours = dir.join("imported_by_us.ptau");
    challenge::import_response(&p0, &r_theirs, &p_ours, Some("from snarkjs"), true).unwrap();
    assert_verifies(&bin, &p_ours, &["from snarkjs"]);

    // We respond, snarkjs imports.
    let r_ours = dir.join("ours.response");
    challenge::challenge_contribute(&c0, &r_ours, "our entropy", &CpuKeyScale).unwrap();
    let p_theirs = dir.join("imported_by_snarkjs.ptau");
    snarkjs(
        &bin,
        &[
            "powersoftau",
            "import",
            "response",
            s(&p0),
            s(&r_ours),
            s(&p_theirs),
            "-n=from us",
        ],
    );
    assert_verifies(&bin, &p_theirs, &["from us"]);

    // A response to an older challenge is refused.
    let p1 = dir.join("p1.ptau");
    let err =
        challenge::import_response(&p1, &r_ours, &dir.join("x.ptau"), None, true).unwrap_err();
    assert!(
        err.to_string().contains("not based on the previous hash"),
        "{err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
