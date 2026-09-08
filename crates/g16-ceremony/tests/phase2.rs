//! Phase 2 against snarkjs itself.
//!
//! Two oracles, and neither is a round trip through our own code:
//!
//! 1. Every checked-in artifact already ships the `verification_key.json` snarkjs wrote
//!    for it, so the export is a byte comparison against a file nobody in this workspace
//!    produced.
//! 2. A beacon is a pure function of its inputs, so `zkey beacon` is byte comparable too.
//!    A `zkey contribute` is not, because it mixes OS entropy by design, but a contribute
//!    followed by the same beacon is: the beacon reads the file the contribution wrote,
//!    and every byte of the beacon's output that differs is a byte the contribution got
//!    wrong. That is the test that covers `contribute` end to end.
//!
//! The snarkjs half needs `snarkjs` on `PATH`; without it the comparison tests skip with a
//! message rather than reporting green.

use std::path::{Path, PathBuf};
use std::process::Command;

use g16_ceremony::contribute::{self, MpcParams};
use g16_ceremony::vkey;
use g16_ceremony::{CpuKeyScale, Groth16Header};
use g16_msm::CpuMsm;

fn artifacts() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize();
    let Ok(root) = root else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("vkey.json").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-phase2-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn exported_vkey_is_byte_identical_to_snarkjs() {
    let artifacts = artifacts();
    if artifacts.is_empty() {
        eprintln!("no artifacts with a vkey.json, skipping");
        return;
    }
    let dir = tmp_dir("vkey");
    for (name, path) in &artifacts {
        let out = dir.join(format!("{name}.json"));
        vkey::export_verification_key(&path.join("circuit.zkey"), &out).unwrap();
        let got = std::fs::read(&out).unwrap();
        let want = std::fs::read(path.join("vkey.json")).unwrap();
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: {} bytes written, snarkjs wrote {}",
            got.len(),
            want.len()
        );
        if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
            let from = i.saturating_sub(40);
            panic!(
                "{name}: first difference at byte {i}\n  ours:    {:?}\n  snarkjs: {:?}",
                String::from_utf8_lossy(&got[from..(i + 40).min(got.len())]),
                String::from_utf8_lossy(&want[from..(i + 40).min(want.len())]),
            );
        }
    }
    eprintln!("{} verification keys byte-identical", artifacts.len());
}

/// The beacon every comparison here uses. Any hex string would do; what matters is that
/// both implementations get the same one, and that it is not the all-zero hash, which
/// would hide a byte-order mistake in the SHA-256 chain.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
const BEACON_EXP: u8 = 10;

fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

/// snarkjs itself, when it is installed. `SNARKJS` overrides the binary name; a spawn is
/// the only honest probe, since a shim on `PATH` can still fail to start.
fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    Command::new(&bin).arg("--help").output().ok().map(|_| bin)
}

/// Run snarkjs and return whether it exited zero. It reports a failed check on stderr and
/// exits 1, so the status is the whole result; the output is returned for the panic
/// message.
fn snarkjs(bin: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running snarkjs {args:?}: {e}"));
    let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), log)
}

fn expect_snarkjs(bin: &str, args: &[&str]) {
    let (ok, log) = snarkjs(bin, args);
    assert!(ok, "snarkjs {args:?} failed:\n{log}");
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

/// Every zkey small enough that running snarkjs over it twice is not the whole test time.
/// `tiny_mul` carries the only checked-in initial key, so it is the one input whose
/// section 10 has zero contributions to accumulate.
fn small_inputs() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts");
    [
        ("tiny_mul-init", "tiny_mul/c0.zkey"),
        ("tiny_mul", "tiny_mul/circuit.zkey"),
        ("js_1x1_d8", "js_1x1_d8/circuit.zkey"),
    ]
    .iter()
    .map(|(name, rel)| ((*name).to_owned(), root.join(rel)))
    .filter(|(_, p)| p.is_file())
    .collect()
}

/// The contribution hash snarkjs 0.7.6 printed for `zkey beacon` over
/// `tiny_mul/c0.zkey` with [`BEACON_HEX`] and 2^10 iterations. Pinned because it is the
/// one number a contributor publishes, and it is derived from the record rather than read
/// back out of the file, so a byte-identical file with a wrong hash would still pass every
/// other assertion here.
const TINY_MUL_BEACON_HASH: &str = "eaf0a5fa0f5ebf1e0fc2db7aff0f3fc47c602b5344f2ed6fc4ca5ba4d84923c7\
                                    523a7e0fb143efffbba8162cb8e72fba8999165e61a93527f49f79fb3c4c3777";

#[test]
fn beacon_output_is_byte_identical_to_snarkjs() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED beacon_output_is_byte_identical_to_snarkjs: no snarkjs on PATH");
        return;
    };
    let inputs = small_inputs();
    assert!(!inputs.is_empty(), "no artifacts to beacon");
    let dir = tmp_dir("beacon");

    // Named and unnamed are different params blobs, and the name is the only field whose
    // length is not fixed, so both go through.
    for (name, input) in &inputs {
        for label in [None, Some("bench beacon")] {
            let tag = if label.is_some() { "named" } else { "anon" };
            let theirs = dir.join(format!("{name}-{tag}-snarkjs.zkey"));
            let ours = dir.join(format!("{name}-{tag}-ours.zkey"));
            let mut args = vec![
                "zkey",
                "beacon",
                input.to_str().unwrap(),
                theirs.to_str().unwrap(),
                BEACON_HEX,
                "10",
            ];
            let named;
            if let Some(l) = label {
                named = format!("-n={l}");
                args.push(&named);
            }
            expect_snarkjs(&bin, &args);

            let report = contribute::beacon(
                input,
                &ours,
                label,
                &beacon_bytes(),
                BEACON_EXP,
                &CpuKeyScale,
            )
            .unwrap();
            assert_same_bytes(&format!("{name} ({tag})"), &ours, &theirs);

            if name == "tiny_mul-init" && label.is_none() {
                assert_eq!(hex(&report.hash), TINY_MUL_BEACON_HASH);
            }
        }
    }
    eprintln!("{} zkeys beaconed byte-identically", inputs.len() * 2);
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// A `zkey contribute` cannot be byte-compared against snarkjs, because both sides mix
/// fresh OS entropy on purpose. A beacon over the file it wrote can be: the beacon reads
/// the header, the whole of section 10 and both rescaled sections, and every byte of its
/// output that differs is a byte the contribution before it got wrong.
///
/// Both directions are run. Ours-then-theirs proves our output is what snarkjs reads back;
/// theirs-then-ours proves we read a chain snarkjs wrote, which is the case a fresh file of
/// our own never exercises.
#[test]
fn a_contribution_is_pinned_by_the_beacon_that_follows_it() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED a_contribution_is_pinned_by_the_beacon_that_follows_it: no snarkjs");
        return;
    };
    let inputs = small_inputs();
    assert!(!inputs.is_empty(), "no artifacts to contribute to");
    let dir = tmp_dir("contribute");

    for (name, input) in &inputs {
        for source in ["ours", "snarkjs"] {
            let c1 = dir.join(format!("{name}-{source}-c1.zkey"));
            if source == "ours" {
                contribute::contribute(
                    input,
                    &c1,
                    Some("alice"),
                    "phase2 test entropy",
                    &CpuKeyScale,
                )
                .unwrap();
            } else {
                expect_snarkjs(
                    &bin,
                    &[
                        "zkey",
                        "contribute",
                        input.to_str().unwrap(),
                        c1.to_str().unwrap(),
                        "-n=alice",
                        "-e=phase2 test entropy",
                    ],
                );
            }

            let theirs = dir.join(format!("{name}-{source}-c2-snarkjs.zkey"));
            let ours = dir.join(format!("{name}-{source}-c2-ours.zkey"));
            expect_snarkjs(
                &bin,
                &[
                    "zkey",
                    "beacon",
                    c1.to_str().unwrap(),
                    theirs.to_str().unwrap(),
                    BEACON_HEX,
                    "10",
                    "-n=final",
                ],
            );
            contribute::beacon(
                &c1,
                &ours,
                Some("final"),
                &beacon_bytes(),
                BEACON_EXP,
                &CpuKeyScale,
            )
            .unwrap();
            assert_same_bytes(
                &format!("{name} after a {source} contribution"),
                &ours,
                &theirs,
            );

            // The record count is the cheapest thing to get wrong while every byte still
            // matches, because a beacon rewrites the whole section either way.
            let file = g16_zkey::binfile::BinFile::open(&ours, b"zkey", 2).unwrap();
            let mpc = MpcParams::read(file.unique_section(10).unwrap()).unwrap();
            let before = g16_zkey::binfile::BinFile::open(input, b"zkey", 2).unwrap();
            let n_before = MpcParams::read(before.unique_section(10).unwrap())
                .unwrap()
                .contributions
                .len();
            assert_eq!(mpc.contributions.len(), n_before + 2, "{name}");
        }
    }
}

/// The circuit the verifier tests run on: small enough that snarkjs' own `groth16 setup`
/// and `zkey verify r1cs` (which runs a whole setup again) are seconds rather than minutes,
/// and built from a ptau this repo actually ships, which `tiny_mul` is not.
fn verify_fixture() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let r1cs = root.join("bench/artifacts/js_1x1_d8/circuit.r1cs");
    let ptau = root.join("bench/ptau/local_19.ptau");
    (r1cs.is_file() && ptau.is_file()).then_some((r1cs, ptau))
}

/// Swap two points of a section in place. A flipped bit would give a coordinate pair off
/// the curve, which is a different rejection from the one under test; a swap keeps every
/// point valid and changes only the order, so what fails is the check and not the decoder.
fn swap_two_points(path: &Path, id: u32) {
    let file = g16_zkey::binfile::BinFile::open(path, b"zkey", 2).unwrap();
    let s = file
        .sections()
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("section {id} is missing"));
    let (start, len) = (s.start, s.len);
    assert!(len >= 2 * 64, "section {id} has fewer than two points");
    drop(file);
    let mut bytes = std::fs::read(path).unwrap();
    for i in 0..64 {
        bytes.swap(start + i, start + 64 + i);
    }
    std::fs::write(path, bytes).unwrap();
}

/// `zkey verify` from both sides, over three chains: ours end to end, snarkjs end to end,
/// and one of each. The mixed chain is the one that matters, because a transcript that is
/// self-consistent within one implementation and wrong against the other passes the
/// single-implementation chains and fails this one.
#[test]
fn both_verifiers_accept_every_chain_and_reject_a_tampered_one() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED both_verifiers_accept_every_chain_...: no snarkjs on PATH");
        return;
    };
    let Some((r1cs, ptau)) = verify_fixture() else {
        eprintln!("SKIPPED both_verifiers_accept_every_chain_...: no js_1x1_d8 or local_19.ptau");
        return;
    };
    let dir = tmp_dir("verify");
    let msm = CpuMsm::new();

    let init = dir.join("init.zkey");
    expect_snarkjs(
        &bin,
        &[
            "groth16",
            "setup",
            r1cs.to_str().unwrap(),
            ptau.to_str().unwrap(),
            init.to_str().unwrap(),
        ],
    );

    for (chain, by_us) in [
        ("ours", [true, true]),
        ("mixed", [true, false]),
        ("theirs", [false, false]),
    ] {
        let c1 = dir.join(format!("{chain}-c1.zkey"));
        if by_us[0] {
            contribute::contribute(
                &init,
                &c1,
                Some("alice"),
                "phase2 verify entropy",
                &CpuKeyScale,
            )
            .unwrap();
        } else {
            expect_snarkjs(
                &bin,
                &[
                    "zkey",
                    "contribute",
                    init.to_str().unwrap(),
                    c1.to_str().unwrap(),
                    "-n=alice",
                    "-e=phase2 verify entropy",
                ],
            );
        }

        let c2 = dir.join(format!("{chain}-c2.zkey"));
        if by_us[1] {
            contribute::beacon(
                &c1,
                &c2,
                Some("final"),
                &beacon_bytes(),
                BEACON_EXP,
                &CpuKeyScale,
            )
            .unwrap();
        } else {
            expect_snarkjs(
                &bin,
                &[
                    "zkey",
                    "beacon",
                    c1.to_str().unwrap(),
                    c2.to_str().unwrap(),
                    BEACON_HEX,
                    "10",
                    "-n=final",
                ],
            );
        }

        let report = contribute::verify_from_init(&init, &ptau, &c2, &msm)
            .unwrap_or_else(|e| panic!("{chain}: our verify rejected the chain: {e}"));
        assert_eq!(report.contribution_hashes.len(), 2, "{chain}");
        let init_header = Groth16Header::read(
            g16_zkey::binfile::BinFile::open(&init, b"zkey", 2)
                .unwrap()
                .unique_section(2)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            (report.n_vars, report.n_public, report.domain_size),
            (
                init_header.n_vars as usize,
                init_header.n_public as usize,
                init_header.domain_size as usize
            ),
            "{chain}"
        );

        // `zkvi`, not `zkey verify init`: the `zkey verify` alias is claimed by the r1cs
        // form (`cli.js:220`), so the long spelling of the init form never matches.
        expect_snarkjs(
            &bin,
            &[
                "zkvi",
                init.to_str().unwrap(),
                ptau.to_str().unwrap(),
                c2.to_str().unwrap(),
            ],
        );
    }

    // The strongest statement available: snarkjs regenerates the initial key from the
    // circuit itself and still accepts the two contributions we put on top of it.
    expect_snarkjs(
        &bin,
        &[
            "zkv",
            r1cs.to_str().unwrap(),
            ptau.to_str().unwrap(),
            dir.join("ours-c2.zkey").to_str().unwrap(),
        ],
    );

    // Neither rejection is one a passing chain would have produced, so the checks above are
    // not vacuous: section 5 is a byte comparison against the init, section 9 is the only
    // check in the command that is neither a memcmp nor a single pairing.
    for (id, what) in [(5u32, "A"), (9, "H")] {
        let broken = dir.join(format!("broken-{id}.zkey"));
        std::fs::copy(dir.join("ours-c2.zkey"), &broken).unwrap();
        swap_two_points(&broken, id);
        let err = contribute::verify_from_init(&init, &ptau, &broken, &msm).expect_err(&format!(
            "a swapped pair in section {id} ({what}) was accepted"
        ));
        eprintln!("section {id} tampered: {err}");
    }
}

/// The contribution hash snarkjs prints for `js_1x1_d8`'s own record, taken from its
/// `zkey verify r1cs` output ("contribution #1 bench").
///
/// This is the one assertion here that needs no snarkjs at run time, and it is the only
/// one that pins [`contribute::contribution_hash`] against a chain nobody in this workspace
/// wrote. The file is a `zkey contribute` from `gen-artifacts.sh:58`, so the record it
/// checks was produced years before this crate existed.
const JS_1X1_D8_CONTRIBUTION_HASH: &str =
    "d16dedad2f5c4e7b9b99c7c3b6c33eb939da1bc952ca57d282995856e869f8b4\
     236cb5e2fdef93de041cbbe56eb2b895dd20aa9fe27ff3adf3b62bb6cfd5bfee";

#[test]
fn a_checked_in_contribution_hashes_to_what_snarkjs_printed() {
    let zkey =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/js_1x1_d8/circuit.zkey");
    if !zkey.is_file() {
        eprintln!("SKIPPED a_checked_in_contribution_hashes_...: no js_1x1_d8");
        return;
    }
    let file = g16_zkey::binfile::BinFile::open(&zkey, b"zkey", 2).unwrap();
    let mpc = MpcParams::read(file.unique_section(10).unwrap()).unwrap();
    assert_eq!(mpc.contributions.len(), 1);
    assert_eq!(
        hex(&contribute::contribution_hash(&mpc.contributions[0])),
        JS_1X1_D8_CONTRIBUTION_HASH.replace(' ', "")
    );
    assert_eq!(mpc.contributions[0].params.name.as_deref(), Some("bench"));
}
