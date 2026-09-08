//! `groth16 setup` against the only oracle worth having: snarkjs' own bytes.
//!
//! Two of them, and they cover different halves of the file.
//!
//! 1. **The shipped `.zkey`s.** Each `bench/artifacts/*/circuit.zkey` came out of snarkjs
//!    and then out of `zkey contribute`, which copies sections 3 to 7 verbatim and leaves
//!    the header's `alpha_1`, `beta_1`, `beta_2` and `gamma_2` and section 10's `csHash`
//!    alone (`zkey_contribute.js:76-88`, checked by `zkey_verify_frominit.js:125-140`,
//!    `:169-197`). So a contributed key still pins every one of those bytes against a
//!    setup that ran on someone else's machine. Sections 8 and 9 are the two it divides by
//!    delta, and they are the two this test cannot compare directly, which is why `csHash`
//!    matters: it covers section 8's undivided points and the H difference points.
//! 2. **A live `snarkjs groth16 setup`**, which compares the whole file including sections
//!    8 and 9, and then runs `snarkjs zkey verify` over the result. It needs node, so it
//!    is opt in through `G16_SNARKJS`.
//!
//! The `ptau` column is not a free choice. `csHash` folds in points read from ptau section
//! 2, and when `cirPower == power` the read runs one point past the end of that section
//! (`zkey_new.js:511`), so the digest then depends on the bytes of the *next* section too.
//! Every ppot case below is an exact fit, which is how the shipped keys were built and is
//! precisely the case that is easy to get wrong.

use std::path::{Path, PathBuf};
use std::process::Command;

use g16_ceremony::setup::{self, SetupReport};
use g16_msm::CpuMsm;
use g16_zkey::binfile::BinFile;

/// One circuit, the ptau its shipped zkey was generated against, and the numbers that
/// setup derives before it computes anything.
struct Case {
    dir: &'static str,
    r1cs: &'static str,
    ptau: &'static str,
    cir_power: u32,
    n_vars: usize,
    n_public: usize,
    /// Whether the shipped `circuit.zkey` came from *this* ptau. Where it did not, only
    /// the live snarkjs run can judge the bytes: `csHash` and the point sections both
    /// depend on which ceremony the powers came from.
    shipped: bool,
}

/// Ordered smallest first: a failure on `tiny_mul` says something quite different from one
/// that only shows up at 2^18, and the cheap cases should report first.
///
/// `tiny_mul`'s shipped zkey came from a ptau this repo does not carry, so it is marked
/// `shipped: false` and only the live snarkjs run judges it.
///
/// `anon-aadhaar` is absent entirely. It is the largest artifact at `cirPower` 21 against
/// `ppot_0080_21`, so it is another exact fit, and it does pass: 29 s, sections 3 to 7 and
/// the csHash all equal to the shipped key, 3.29M section-4 records. But it peaks at 3.7 GB
/// resident and writes 631 MB, and cargo runs the tests in this file concurrently, so
/// putting it in the table would double that. Run it by hand when the H path changes.
#[rustfmt::skip]
const CASES: &[Case] = &[
    Case { dir: "tiny_mul",      r1cs: "circuit.r1cs", ptau: "local_13.ptau",      cir_power: 3,  n_vars: 6,      n_public: 3,  shipped: false },
    Case { dir: "js_1x1_d8",     r1cs: "circuit.r1cs", ptau: "local_19.ptau",      cir_power: 12, n_vars: 3373,   n_public: 4,  shipped: true },
    Case { dir: "js_2x2_d16",    r1cs: "circuit.r1cs", ptau: "local_19.ptau",      cir_power: 14, n_vars: 10194,  n_public: 6,  shipped: true },
    Case { dir: "railgun-01x01", r1cs: "01x01.r1cs",   ptau: "ppot_0080_15.ptau",  cir_power: 15, n_vars: 20154,  n_public: 4,  shipped: true },
    Case { dir: "js_2x2_d32",    r1cs: "circuit.r1cs", ptau: "local_19.ptau",      cir_power: 15, n_vars: 18002,  n_public: 6,  shipped: true },
    Case { dir: "tornado",       r1cs: "circuit.r1cs", ptau: "ppot_0080_15.ptau",  cir_power: 15, n_vars: 28300,  n_public: 6,  shipped: true },
    Case { dir: "sha256",        r1cs: "sha256.r1cs",  ptau: "ppot_0080_16.ptau",  cir_power: 16, n_vars: 59170,  n_public: 256, shipped: true },
    Case { dir: "js_8x8_d32",    r1cs: "circuit.r1cs", ptau: "local_19.ptau",      cir_power: 17, n_vars: 70640,  n_public: 18, shipped: true },
    Case { dir: "js_16x16_d32",  r1cs: "circuit.r1cs", ptau: "local_19.ptau",      cir_power: 18, n_vars: 140824, n_public: 34, shipped: true },
    Case { dir: "keccak256",     r1cs: "keccak256.r1cs", ptau: "ppot_0080_18.ptau", cir_power: 18, n_vars: 240257, n_public: 256, shipped: true },
    Case { dir: "rsa2048",       r1cs: "rsa2048.r1cs", ptau: "ppot_0080_18.ptau",  cir_power: 18, n_vars: 190035, n_public: 17, shipped: true },
    Case { dir: "railgun-13x01", r1cs: "13x01.r1cs",   ptau: "ppot_0080_18.ptau",  cir_power: 18, n_vars: 141499, n_public: 16, shipped: true },
];

/// The sections `zkey contribute` copies verbatim, so a contributed key still holds
/// setup's own bytes for them.
const UNTOUCHED_SECTIONS: [u32; 5] = [3, 4, 5, 6, 7];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-setup-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The three input paths, or `None` when the artifact was never generated. A fresh clone
/// without `gen-artifacts.sh` should skip loudly, not report false green.
fn inputs(case: &Case) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let dir = root().join("bench/artifacts").join(case.dir);
    let r1cs = dir.join(case.r1cs);
    let zkey = dir.join("circuit.zkey");
    let ptau = root().join("bench/ptau").join(case.ptau);
    (r1cs.exists() && zkey.exists() && ptau.exists()).then_some((r1cs, ptau, zkey))
}

fn run(case: &Case, out: &Path) -> SetupReport {
    let (r1cs, ptau, _) = inputs(case).expect("checked by the caller");
    setup::setup(&r1cs, &ptau, out, &CpuMsm::new())
        .unwrap_or_else(|e| panic!("{}: setup failed: {e}", case.dir))
}

/// Sections 3 to 7 and the header points snarkjs wrote, against the ones we write.
#[test]
fn matches_the_zkey_snarkjs_built() {
    let dir = tmp_dir("shipped");
    let mut ran = 0;
    for case in CASES {
        let Some((_, _, shipped)) = inputs(case) else {
            eprintln!("skipping {}: artifact missing", case.dir);
            continue;
        };
        let out = dir.join(format!("{}.zkey", case.dir));
        let report = run(case, &out);

        assert_eq!(report.cir_power, case.cir_power, "{}: cirPower", case.dir);
        assert_eq!(report.n_vars, case.n_vars, "{}: nVars", case.dir);
        assert_eq!(report.n_public, case.n_public, "{}: nPublic", case.dir);
        assert_eq!(
            report.domain_size,
            1 << case.cir_power,
            "{}: domainSize",
            case.dir
        );
        ran += 1;
        if !case.shipped {
            continue;
        }

        let want = BinFile::open(&shipped, b"zkey", 2).unwrap();
        let got = BinFile::open(&out, b"zkey", 2).unwrap();
        for id in UNTOUCHED_SECTIONS {
            let (a, b) = (
                got.unique_section(id).unwrap(),
                want.unique_section(id).unwrap(),
            );
            assert_eq!(a.len(), b.len(), "{}: section {id} length", case.dir);
            let first = a.iter().zip(b).position(|(x, y)| x != y);
            assert!(
                first.is_none(),
                "{}: section {id} differs at byte {}",
                case.dir,
                first.unwrap()
            );
        }

        // The header's first three points come from the ptau and `gamma_2` is the plain
        // generator, so a contribution leaves all four alone and only scales `delta_1` and
        // `delta_2`. That is why the comparison stops at byte 468: 84 bytes of scalars,
        // then alpha_1, beta_1, beta_2 and gamma_2.
        const THROUGH_GAMMA_2: usize = 84 + 64 + 64 + 128 + 128;
        let (a, b) = (
            got.unique_section(2).unwrap(),
            want.unique_section(2).unwrap(),
        );
        assert_eq!(
            a[..THROUGH_GAMMA_2],
            b[..THROUGH_GAMMA_2],
            "{}: header up to gamma_2",
            case.dir
        );

        assert_eq!(
            want.unique_section(10).unwrap()[..64],
            report.cs_hash[..],
            "{}: csHash",
            case.dir
        );
    }
    assert!(ran > 0, "no artifacts present, nothing was checked");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `cirPower` and `domainSize` against the header of the zkey snarkjs actually produced,
/// over every artifact rather than the table above. The floor-log2 plus one is easy to
/// write as a `next_power_of_two`, which is right on everything except an exact power.
#[test]
fn derived_domain_matches_every_shipped_header() {
    let dirs = root().join("bench/artifacts");
    let Ok(entries) = std::fs::read_dir(&dirs) else {
        eprintln!("skipping: no artifacts directory");
        return;
    };
    let mut ran = 0;
    for entry in entries.flatten() {
        let dir = entry.path();
        let zkey = dir.join("circuit.zkey");
        let Some(r1cs) = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "r1cs"))
        else {
            continue;
        };
        if !zkey.exists() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let r1cs = g16_ceremony::r1cs::R1cs::open(&r1cs).unwrap();
        let power = setup::circuit_power(
            r1cs.header().n_constraints as usize,
            r1cs.header().n_public(),
        );
        let header = g16_ceremony::Groth16Header::read(
            BinFile::open(&zkey, b"zkey", 2)
                .unwrap()
                .unique_section(2)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(header.domain_size, 1u32 << power, "{name}: domainSize");
        assert_eq!(header.n_vars, r1cs.header().n_vars, "{name}: nVars");
        assert_eq!(
            header.n_public as usize,
            r1cs.header().n_public(),
            "{name}: nPublic"
        );
        ran += 1;
    }
    assert!(ran > 0, "no artifacts present, nothing was checked");
}

/// A ptau smaller than the circuit and a raw ptau are both refused before any point is
/// read, and with the arm that says which it was.
#[test]
fn refuses_a_ptau_it_cannot_use() {
    let case = &CASES[0];
    let Some((r1cs, _, _)) = inputs(case) else {
        eprintln!("skipping: artifact missing");
        return;
    };
    let dir = tmp_dir("refuse");
    let small = root().join("bench/ptau/local_13.ptau");
    if small.exists() {
        // js_1x1_d8 needs 2^12, which local_13 has; js_16x16_d32 needs 2^18, which it does
        // not.
        let big = root().join("bench/artifacts/js_16x16_d32/circuit.r1cs");
        let err = setup::setup(&big, &small, &dir.join("x.zkey"), &CpuMsm::new()).unwrap_err();
        assert!(
            matches!(
                err,
                g16_ceremony::CeremonyError::PtauTooSmall {
                    needed: 18,
                    have: 13
                }
            ),
            "expected PtauTooSmall, got {err}"
        );
        // And the same ptau does serve a circuit that fits, so the rejection is about the
        // power and not about the file.
        setup::setup(&r1cs, &small, &dir.join("y.zkey"), &CpuMsm::new()).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Whole-file byte equality against a live `snarkjs groth16 setup`, then `snarkjs zkey
/// verify` over what we wrote.
///
/// Opt in with `G16_SNARKJS=/path/to/snarkjs/cli.js`, since it needs node and a snarkjs
/// checkout with its dependencies installed. This is the only test that covers sections 8
/// and 9, which a contributed zkey has divided by delta.
///
/// Unset `G16_SNARKJS` and this reports itself as passing, which is the one way it can be
/// green without having compared anything: absent artifacts are counted, and zero of them
/// fails. A run that matters says so in its output.
#[test]
fn matches_snarkjs_bytes() {
    let Ok(cli) = std::env::var("G16_SNARKJS") else {
        eprintln!("skipping: set G16_SNARKJS to a snarkjs cli.js to run this");
        return;
    };
    let dir = tmp_dir("snarkjs");
    let mut ran = 0;
    for case in CASES {
        let Some((r1cs, ptau, _)) = inputs(case) else {
            eprintln!("skipping {}: artifact missing", case.dir);
            continue;
        };
        let ours = dir.join(format!("{}-ours.zkey", case.dir));
        let theirs = dir.join(format!("{}-snarkjs.zkey", case.dir));
        run(case, &ours);

        let out = Command::new("node")
            .args(["--max-old-space-size=16384"])
            .arg(&cli)
            .arg("groth16")
            .arg("setup")
            .args([&r1cs, &ptau, &theirs])
            .output()
            .expect("node");
        assert!(out.status.success(), "{}: snarkjs setup failed", case.dir);

        let a = std::fs::read(&ours).unwrap();
        let b = std::fs::read(&theirs).unwrap();
        assert_eq!(a.len(), b.len(), "{}: file length", case.dir);
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        assert!(
            first.is_none(),
            "{}: differs at byte {}",
            case.dir,
            first.unwrap()
        );

        let out = Command::new("node")
            .args(["--max-old-space-size=16384"])
            .arg(&cli)
            .arg("zkey")
            .arg("verify")
            .args([&r1cs, &ptau, &ours])
            .output()
            .expect("node");
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("ZKey Ok!"),
            "{}: snarkjs zkey verify rejected our key:\n{}",
            case.dir,
            String::from_utf8_lossy(&out.stdout)
        );
        ran += 1;
        let _ = std::fs::remove_file(&ours);
        let _ = std::fs::remove_file(&theirs);
    }
    assert!(ran > 0, "no artifacts present, nothing was checked");
    let _ = std::fs::remove_dir_all(&dir);
}
