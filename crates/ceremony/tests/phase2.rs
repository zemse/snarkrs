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

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

mod common;

/// The beacon every comparison here uses. Any hex string would do; what matters is that
/// both implementations get the same one, and that it is not the all-zero hash, which
/// would hide a byte-order mistake in the SHA-256 chain.
use common::{beacon_bytes, BEACON_HEX};

use std::path::{Path, PathBuf};
use std::process::Command;

use snarkrs_ceremony::contribute::{self, Check, MpcParams};
use snarkrs_ceremony::vkey;
use snarkrs_ceremony::write::g2_lem;
use snarkrs_ceremony::{CeremonyError, CpuKeyScale, Groth16Header};
use snarkrs_field::G2Affine;
use snarkrs_msm::CpuMsm;

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
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-ceremony-phase2-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn exported_vkey_is_byte_identical_to_snarkjs() {
    let artifacts = artifacts();
    if artifacts.is_empty() {
        resources::skip("no artifacts with a vkey.json");
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

const BEACON_EXP: u8 = 10;

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
    .filter(|(_, p)| {
        p.is_file() || {
            resources::skip(format_args!("missing {}", p.display()));
            false
        }
    })
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
        resources::skip("beacon_output_is_byte_identical_to_snarkjs: no snarkjs on PATH");
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
        resources::skip("a_contribution_is_pinned_by_the_beacon_that_follows_it: no snarkjs");
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
            let file = snarkrs_formats::binfile::BinFile::open(&ours, b"zkey", 2).unwrap();
            let mpc = MpcParams::read(file.unique_section(10).unwrap()).unwrap();
            let before = snarkrs_formats::binfile::BinFile::open(input, b"zkey", 2).unwrap();
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
    let file = snarkrs_formats::binfile::BinFile::open(path, b"zkey", 2).unwrap();
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
        resources::skip("both_verifiers_accept_every_chain_...: no snarkjs on PATH");
        return;
    };
    let Some((r1cs, ptau)) = verify_fixture() else {
        resources::skip("both_verifiers_accept_every_chain_...: no js_1x1_d8 or local_19.ptau");
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
            snarkrs_formats::binfile::BinFile::open(&init, b"zkey", 2)
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
        resources::skip("a_checked_in_contribution_hashes_...: no js_1x1_d8");
        return;
    }
    let file = snarkrs_formats::binfile::BinFile::open(&zkey, b"zkey", 2).unwrap();
    let mpc = MpcParams::read(file.unique_section(10).unwrap()).unwrap();
    assert_eq!(mpc.contributions.len(), 1);
    assert_eq!(
        hex(&contribute::contribution_hash(&mpc.contributions[0])),
        JS_1X1_D8_CONTRIBUTION_HASH.replace(' ', "")
    );
    assert_eq!(mpc.contributions[0].params.name.as_deref(), Some("bench"));
}

/// `verify_from_init` computes `SG1 * (n_vars - n_public - 1)` for the L section length
/// and `coeffs[domain_size - 1]` in the H check, on three fields that arrive straight out
/// of a file. Debug panics on the subtraction and release wraps to `usize::MAX` and panics
/// on the index, so an unvalidated header aborts the one command that owes its caller an
/// `Err` on hostile input, in both profiles.
#[test]
fn a_header_whose_counts_cannot_be_computed_with_is_refused() {
    let zkey =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul/circuit.zkey");
    if !zkey.is_file() {
        resources::skip("a_header_whose_counts_...: no tiny_mul");
        return;
    }
    let file = snarkrs_formats::binfile::BinFile::open(&zkey, b"zkey", 2).unwrap();
    let body = file.unique_section(2).unwrap().to_vec();
    Groth16Header::read(&body).expect("the unedited header still parses");
    // Two primes of `u32 n8` and 32 bytes come first, so `nVars` is at 72, `nPublic` at 76
    // and `domainSize` at 80.
    for (at, value) in [(72usize, 0u32), (76, u32::MAX), (80, 0), (80, 3)] {
        let mut edited = body.clone();
        edited[at..at + 4].copy_from_slice(&value.to_le_bytes());
        assert!(
            matches!(
                Groth16Header::read(&edited),
                Err(CeremonyError::Malformed { section: 2, .. })
            ),
            "byte {at} set to {value} was accepted"
        );
    }
}

/// A key with an empty chain is a key whose delta is 1, and anyone can forge a proof under
/// one. snarkjs reports it as verified; phase 1 already refuses the analogue, and this is
/// the only place the two phases disagreed about it.
#[test]
fn an_uncontributed_key_does_not_verify_against_itself() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench");
    let (r1cs, ptau) = (
        dir.join("artifacts/tiny_mul/circuit.r1cs"),
        dir.join("ptau/local_13.ptau"),
    );
    if !r1cs.is_file() || !ptau.is_file() {
        resources::skip("an_uncontributed_key_...: no tiny_mul or local_13");
        return;
    }
    let msm = CpuMsm::new();
    let init = tmp_dir("uncontributed").join("init.zkey");
    snarkrs_ceremony::setup::setup(&r1cs, &ptau, &init, &msm).unwrap();
    let err = contribute::verify_from_init(&init, &ptau, &init, &msm)
        .expect_err("an init key verified as a finished one");
    assert!(
        matches!(&err, CeremonyError::Verification(s) if s.contains("no contribution")),
        "{err}"
    );
}

/// A point that is off the curve, or on it but outside the prime-order subgroup, still
/// decodes: `binfile` builds every point with `new_unchecked`. Arkworks documents
/// arithmetic on one of those as meaningless, and each of these six goes straight into a
/// `same_ratio` pairing, so a `verify` that never looks makes no statement about any curve
/// point. snarkjs never writes such a header, so the gate costs nothing real.
#[test]
fn a_header_point_off_the_curve_or_off_the_subgroup_is_refused() {
    let zkey =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul/circuit.zkey");
    if !zkey.is_file() {
        resources::skip("a_header_point_off_...: no tiny_mul");
        return;
    }
    let file = snarkrs_formats::binfile::BinFile::open(&zkey, b"zkey", 2).unwrap();
    let body = file.unique_section(2).unwrap().to_vec();

    // `alpha_1` is the first point of the header, at 84. One flipped bit in its `x` moves
    // it off the curve without moving the limbs out of range.
    let mut edited = body.clone();
    edited[84] ^= 1;
    assert!(
        !snarkrs_formats::binfile::g1(&edited[84..84 + 64])
            .unwrap()
            .is_on_curve(),
        "the edit has to move alpha_1 off the curve for this to test anything"
    );
    assert!(
        matches!(
            Groth16Header::read(&edited),
            Err(CeremonyError::Malformed { section: 2, .. })
        ),
        "an off-curve alpha_1 was accepted"
    );

    // `gamma_2` is at 340: 84 plus two G1 and one G2. G1 has cofactor 1 on BN254, so the
    // subgroup half of the check only ever does work on a G2 point.
    let mut edited = body.clone();
    edited[340..340 + 128].copy_from_slice(&g2_lem(&off_subgroup_g2()));
    assert!(
        matches!(
            Groth16Header::read(&edited),
            Err(CeremonyError::Malformed { section: 2, .. })
        ),
        "an off-subgroup gamma_2 was accepted"
    );
}

/// A point of `E'(Fq2)` outside the order-`r` subgroup, found by taking the smallest `x`
/// whose curve equation has a root. The compressed decoder is the shortest way to solve
/// for `y`, and like every decoder here it does not check what it returns.
fn off_subgroup_g2() -> G2Affine {
    for i in 1u64..256 {
        let mut x = [0u8; 64];
        x[56..].copy_from_slice(&i.to_be_bytes());
        let Ok(p) = snarkrs_ceremony::transcript::g2_from_compressed(&x) else {
            continue;
        };
        if p.is_on_curve() && !p.is_in_correct_subgroup_assuming_on_curve() {
            return p;
        }
    }
    panic!("no off-subgroup G2 point under x = 256");
}

/// The bound is checked before the input is opened, so a path that does not exist is the
/// cheapest proof that nothing downstream of it was reached: an accepted exponent fails
/// with an `Io` error instead.
#[test]
fn a_beacon_exponent_outside_the_bounds_is_refused_before_the_file_is_opened() {
    let dir = tmp_dir("beacon-bounds");
    for exp in [0u8, 9, 64, 100, 128, 255] {
        let err = contribute::beacon(
            &dir.join("no-such-input.zkey"),
            &dir.join("out.zkey"),
            None,
            &beacon_bytes(),
            exp,
            &CpuKeyScale,
        )
        .expect_err("an exponent outside 10..=63 was accepted");
        assert!(
            matches!(&err, CeremonyError::BadParams(s) if s.contains("numIterationsExp")),
            "{exp}: {err}"
        );
    }
}

/// `numIterationsExp` reaches the verifier as a raw byte off disk, and
/// `rng_from_beacon_params` turns it into `2^exp` SHA-256 rounds. Unchecked, 100 is a
/// `zkey verify` that never returns, on the one command whose whole job is to survive a
/// file someone else wrote.
#[test]
fn a_beacon_record_with_an_unbounded_exponent_is_refused_by_the_verifier() {
    let bench = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench");
    let (r1cs, ptau) = (
        bench.join("artifacts/tiny_mul/circuit.r1cs"),
        bench.join("ptau/local_13.ptau"),
    );
    if !r1cs.is_file() || !ptau.is_file() {
        resources::skip("a_beacon_record_with_an_unbounded_exponent_...: no tiny_mul or ptau");
        return;
    }
    let dir = tmp_dir("hostile-beacon");
    let init = dir.join("init.zkey");
    snarkrs_ceremony::setup::setup(&r1cs, &ptau, &init, &CpuMsm::new()).unwrap();
    let beaconed = dir.join("beacon.zkey");
    contribute::beacon(
        &init,
        &beaconed,
        None,
        &beacon_bytes(),
        BEACON_EXP,
        &CpuKeyScale,
    )
    .unwrap();
    contribute::verify_from_init(&init, &ptau, &beaconed, &CpuMsm::new())
        .expect("the unedited beacon verifies");

    // The beacon is the only record and carries no name, so its params blob is the last
    // 36 bytes of section 10: `2, exp, 3, 32`, then the hash.
    let file = snarkrs_formats::binfile::BinFile::open(&beaconed, b"zkey", 2).unwrap();
    let section = file.sections().iter().find(|s| s.id == 10).unwrap();
    let at = section.start + section.len - 35;
    drop(file);
    let mut bytes = std::fs::read(&beaconed).unwrap();
    assert_eq!(
        (bytes[at - 1], bytes[at], bytes[at + 1], bytes[at + 2]),
        (2, BEACON_EXP, 3, 32),
        "the params blob is not where this test edits"
    );
    bytes[at] = 100;
    std::fs::write(&beaconed, bytes).unwrap();

    let err = contribute::verify_from_init(&init, &ptau, &beaconed, &CpuMsm::new())
        .expect_err("a numIterationsExp of 100 was accepted");
    assert!(
        matches!(&err, CeremonyError::Verification(s) if s.contains("numIterationsExp")),
        "{err}"
    );
}

/// `(id, start, len)` of each section of a zkey, in file order.
type Table = [(u32, usize, usize)];

/// One edit of a contributed key, and the message both forms of `zkey verify` must then
/// reject it with, under the exact check and under the random one.
struct Tamper {
    what: &'static str,
    edit: fn(&mut Vec<u8>, &Table),
    expect: &'static str,
    random: &'static str,
}

/// `(id, start, len)` of every section, in file order.
fn section_table(path: &Path) -> Vec<(u32, usize, usize)> {
    let file = snarkrs_formats::binfile::BinFile::open(path, b"zkey", 2).unwrap();
    file.sections()
        .iter()
        .map(|s| (s.id, s.start, s.len))
        .collect()
}

fn body(table: &Table, id: u32) -> (usize, usize) {
    let &(_, start, len) = table.iter().find(|s| s.0 == id).unwrap();
    (start, len)
}

/// Swap point `at` of section `id` with the first later point that differs from it.
fn swap_from(bytes: &mut [u8], table: &Table, id: u32, stride: usize, at: usize) {
    let (start, len) = body(table, id);
    let a = start + at * stride;
    let b = (a + stride..start + len)
        .step_by(stride)
        .find(|&b| bytes[a..a + stride] != bytes[b..b + stride])
        .unwrap_or_else(|| panic!("section {id} has no point after {at} that differs from it"));
    for i in 0..stride {
        bytes.swap(a + i, b + i);
    }
}

/// Swap the last two points of section `id`, which in a section of more than 2^20 points
/// is inside the last chunk setup produces.
fn swap_last_two(bytes: &mut [u8], table: &Table, id: u32, stride: usize) {
    let (start, len) = body(table, id);
    let last = start + len - stride;
    assert_ne!(
        bytes[last - stride..last],
        bytes[last..last + stride],
        "section {id} ends in two equal points, so swapping them changes nothing"
    );
    for i in 0..stride {
        bytes.swap(last - stride + i, last + i);
    }
}

/// Rewrite the file with `id`'s body cut or zero-padded to `new_len`, every section
/// header kept in order.
fn resize(bytes: &mut Vec<u8>, table: &Table, id: u32, new_len: usize) {
    let mut out = bytes[..12].to_vec();
    for &(sid, start, len) in table {
        let mut b = bytes[start..start + len].to_vec();
        if sid == id {
            b.resize(new_len, 0);
        }
        out.extend_from_slice(&sid.to_le_bytes());
        out.extend_from_slice(&(b.len() as u64).to_le_bytes());
        out.extend_from_slice(&b);
    }
    *bytes = out;
}

const TAMPERS: &[Tamper] = &[
    Tamper {
        what: "IC",
        edit: |b, t| swap_last_two(b, t, 3, 64),
        expect: "IC section is not identical",
        random: "IC section is not identical",
    },
    Tamper {
        what: "Coeffs, last record",
        edit: |b, t| {
            let (start, len) = body(t, 4);
            b[start + len - 1] ^= 1;
        },
        expect: "Coeffs section is not identical",
        random: "Coeffs section is not identical",
    },
    Tamper {
        what: "A, last chunk",
        edit: |b, t| swap_last_two(b, t, 5, 64),
        expect: "A section is not identical",
        random: "A section does not match",
    },
    Tamper {
        what: "B1, one bit of the last point",
        edit: |b, t| {
            let (start, len) = body(t, 6);
            b[start + len - 1] ^= 1;
        },
        expect: "B1 section is not identical",
        random: "B1 section does not match",
    },
    Tamper {
        what: "B2, last chunk",
        edit: |b, t| swap_last_two(b, t, 7, 128),
        expect: "B2 section is not identical",
        random: "B2 section does not match",
    },
    Tamper {
        what: "L, last chunk",
        edit: |b, t| swap_last_two(b, t, 8, 64),
        expect: "L section does not match",
        random: "L section does not match",
    },
    Tamper {
        what: "H",
        edit: |b, t| swap_last_two(b, t, 9, 64),
        expect: "H section does not match",
        random: "H section does not match",
    },
    Tamper {
        what: "csHash",
        edit: |b, t| b[body(t, 10).0] ^= 1,
        expect: "inconsistent transcript",
        random: "inconsistent transcript",
    },
    Tamper {
        // `alpha_1` and `beta_1` sit at 84 and 148 of section 2. Both are curve points,
        // so the swap passes the header's own checks and only the comparison catches it.
        what: "header alpha1",
        edit: |b, t| {
            let start = body(t, 2).0;
            for i in 0..64 {
                b.swap(start + 84 + i, start + 148 + i);
            }
        },
        expect: "invalid alpha1",
        random: "invalid alpha1",
    },
    Tamper {
        what: "A, one point longer",
        edit: |b, t| {
            let len = body(t, 5).1;
            resize(b, t, 5, len + 64);
        },
        expect: "A section is not identical",
        random: "A section is not identical",
    },
    Tamper {
        what: "B2, one point shorter",
        edit: |b, t| {
            let len = body(t, 7).1;
            resize(b, t, 7, len - 128);
        },
        expect: "B2 section is not identical",
        random: "B2 section is not identical",
    },
    Tamper {
        what: "L, one point shorter",
        edit: |b, t| {
            let len = body(t, 8).1;
            resize(b, t, 8, len - 64);
        },
        expect: "invalid L section size",
        random: "invalid L section size",
    },
    Tamper {
        what: "H, one point longer",
        edit: |b, t| {
            let len = body(t, 9).1;
            resize(b, t, 9, len + 64);
        },
        expect: "invalid H section size",
        random: "invalid H section size",
    },
];

/// The first point of section `id` that is not the point at infinity.
fn first_finite(bytes: &[u8], table: &Table, id: u32, stride: usize) -> usize {
    let (start, len) = body(table, id);
    (0..len / stride)
        .find(|i| bytes[start + i * stride..start + (i + 1) * stride] != vec![0u8; stride][..])
        .unwrap_or_else(|| panic!("section {id} is all infinity"))
}

/// Add `q` to the first coordinate of the first finite point of section `id`: the same
/// field element, encoded out of range, which the exact check sees only as other bytes.
fn non_canonical(bytes: &mut [u8], table: &Table, id: u32, stride: usize) {
    use snarkrs_field::PrimeField;
    let at = body(table, id).0 + first_finite(bytes, table, id, stride) * stride;
    let q = snarkrs_field::Fq::MODULUS.0;
    let mut carry = 0u128;
    for (i, limb) in q.iter().enumerate() {
        let b = &mut bytes[at + i * 8..at + i * 8 + 8];
        let sum = u128::from(u64::from_le_bytes(b.try_into().unwrap())) + u128::from(*limb) + carry;
        b.copy_from_slice(&(sum as u64).to_le_bytes());
        carry = sum >> 64;
    }
    assert_eq!(carry, 0, "x + q does not fit 256 bits");
}

/// Flip the low bit of the first finite point's `y` (`y.c0` in G2): still below `q`, and
/// off the curve.
fn off_curve(bytes: &mut [u8], table: &Table, id: u32, stride: usize) {
    let at = body(table, id).0 + first_finite(bytes, table, id, stride) * stride;
    bytes[at + stride / 2] ^= 1;
}

/// A point of order 10069 on the G2 twist, the smallest prime factor of its cofactor
/// `h`: a twist point of no particular order, times `r * h / 10069`.
fn order_10069_point() -> snarkrs_field::G2Projective {
    use snarkrs_field::{AffineRepr, Fq, Fq2, PrimeField};
    // h / 10069, little-endian.
    const H_OVER: [u64; 4] = [
        0x6c3cd334915f1659,
        0x207142f7671af448,
        0x9e28bcf65b5681da,
        0x0001_3af7a58fce69,
    ];
    for x in 1u64.. {
        let Some(p) =
            G2Affine::get_point_from_x_unchecked(Fq2::new(Fq::from(x), Fq::from(1u64)), false)
        else {
            continue;
        };
        let t = p.mul_bigint(snarkrs_field::Fr::MODULUS.0);
        let t = AffineRepr::mul_bigint(&snarkrs_field::CurveGroup::into_affine(t), H_OVER);
        if !snarkrs_field::Zero::is_zero(&t) {
            let t = snarkrs_field::CurveGroup::into_affine(t);
            assert!(t.is_on_curve() && !t.is_in_correct_subgroup_assuming_on_curve());
            assert!(snarkrs_field::Zero::is_zero(&t.mul_bigint([10069u64])));
            return t.into_group();
        }
    }
    unreachable!()
}

/// Add a point of order 10069 to the first finite B2 point. The result is on the curve,
/// canonically encoded and outside the order-r subgroup, which only a subgroup check
/// rejects for certain.
fn small_subgroup_b2(bytes: &mut [u8], table: &Table) {
    let at = body(table, 7).0 + first_finite(bytes, table, 7, 128) * 128;
    let p = snarkrs_formats::binfile::g2(&bytes[at..at + 128]).unwrap();
    let moved = snarkrs_field::CurveGroup::into_affine(p + order_10069_point());
    bytes[at..at + 128].copy_from_slice(&g2_lem(&moved));
}

/// Add G to the first finite point of section 5 and subtract it from the next, so the
/// section's plain sum is unchanged and only unequal scalars on the two points catch it.
fn cancel_pair_a(bytes: &mut [u8], table: &Table) {
    use snarkrs_field::{AffineRepr, CurveGroup};
    let first = first_finite(bytes, table, 5, 64);
    let start = body(table, 5).0;
    let g = snarkrs_field::G1Affine::generator();
    for (i, sign) in [(first, true), (first + 1, false)] {
        let at = start + i * 64;
        let p = snarkrs_formats::binfile::g1(&bytes[at..at + 64]).unwrap();
        let moved = if sign { p + g } else { p - g }.into_affine();
        bytes[at..at + 64].copy_from_slice(&snarkrs_ceremony::write::g1_lem(&moved));
    }
}

/// Edits only the random check has its own reason to reject, in a point that is otherwise
/// the right one: the exact check sees different bytes.
const RANDOM_TAMPERS: &[Tamper] = &[
    Tamper {
        what: "A, +G and -G on two points",
        edit: |b, t| cancel_pair_a(b, t),
        expect: "A section is not identical",
        random: "A section does not match",
    },
    Tamper {
        what: "A, x + q",
        edit: |b, t| non_canonical(b, t, 5, 64),
        expect: "A section is not identical",
        random: "has a coordinate at or above q",
    },
    Tamper {
        what: "B1, x + q",
        edit: |b, t| non_canonical(b, t, 6, 64),
        expect: "B1 section is not identical",
        random: "has a coordinate at or above q",
    },
    Tamper {
        what: "B2, x.c0 + q",
        edit: |b, t| non_canonical(b, t, 7, 128),
        expect: "B2 section is not identical",
        random: "has a coordinate at or above q",
    },
    Tamper {
        what: "A, off the curve",
        edit: |b, t| off_curve(b, t, 5, 64),
        expect: "A section is not identical",
        random: "A section does not match: point",
    },
    Tamper {
        what: "B1, off the curve",
        edit: |b, t| off_curve(b, t, 6, 64),
        expect: "B1 section is not identical",
        random: "B1 section does not match: point",
    },
    Tamper {
        what: "B2, off the curve",
        edit: |b, t| off_curve(b, t, 7, 128),
        expect: "B2 section is not identical",
        random: "B2 section does not match: point",
    },
    Tamper {
        what: "B2, plus a point of order 10069",
        edit: |b, t| small_subgroup_b2(b, t),
        expect: "B2 section is not identical",
        random: "B2 section does not match: a point is outside the order-r subgroup",
    },
];

/// Every form and check of `zkey verify` on one key: `(init exact, r1cs exact, init random,
/// r1cs random)`.
type Verdicts = [Result<contribute::ZkeyVerifyReport, CeremonyError>; 4];

fn verify_all(r1cs: &Path, ptau: &Path, init: &Path, key: &Path) -> Verdicts {
    let msm = CpuMsm::new();
    [
        contribute::verify_from_init_with(init, ptau, key, &msm, Check::Exact),
        contribute::verify_from_r1cs_with(r1cs, ptau, key, &msm, &msm, Check::Exact),
        contribute::verify_from_init_with(init, ptau, key, &msm, Check::Random),
        contribute::verify_from_r1cs_with(r1cs, ptau, key, &msm, &msm, Check::Random),
    ]
}

/// Both forms reject `what` with the same message under each check, containing `expect`
/// under the exact one and `random` under the random one.
#[track_caller]
fn assert_rejected(verdicts: Verdicts, what: &str, expect: &str, random: &str) {
    let [init, r1cs, init_random, r1cs_random] = verdicts.map(|v| v.err().map(|e| e.to_string()));
    for (form, err, want) in [
        ("init", &init, expect),
        ("r1cs", &r1cs, expect),
        ("init --check random", &init_random, random),
        ("r1cs --check random", &r1cs_random, random),
    ] {
        let err = err
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: the {form} form accepted it"));
        assert!(err.contains(want), "{what}: {form}: {err}");
    }
    assert_eq!(r1cs, init, "{what}");
    assert_eq!(r1cs_random, init_random, "{what}");
}

/// `zkey verify r1cs` compares setup's output as it is produced instead of through an init
/// key on disk, so every rejection has to come out as `zkey verify init` against that init
/// key words it, which also pins the order the checks run in. anon-aadhaar has 1.1 million
/// signals, so its A, B1, B2 and C sections span two of setup's chunks and the edits
/// marked "last chunk" land in the second.
///
/// The random check has to reject every one of them too, and the edits only it has a
/// reason of its own to reject: an out-of-range coordinate, a point off the curve, and a
/// B2 point with a component of small order.
#[test]
fn the_r1cs_form_rejects_what_the_init_form_rejects() {
    let bench = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench");
    let msm = CpuMsm::new();
    for (circuit, power) in [("js_2x2_d32", 17), ("anon-aadhaar", 21)] {
        let r1cs = bench.join(format!("artifacts/{circuit}/circuit.r1cs"));
        let ptau = bench.join(format!("ptau/ppot_0080_{power}.ptau"));
        if !r1cs.is_file() || !ptau.is_file() {
            resources::skip(format_args!("the_r1cs_form_rejects_...: no {circuit}"));
            continue;
        }
        let dir = tmp_dir(&format!("r1cs-form-{circuit}"));
        let init = dir.join("init.zkey");
        let good = dir.join("good.zkey");
        snarkrs_ceremony::setup::setup(&r1cs, &ptau, &init, &msm).unwrap();
        contribute::contribute(&init, &good, None, "r1cs form", &CpuKeyScale).unwrap();
        for (verdict, check) in verify_all(&r1cs, &ptau, &init, &good).into_iter().zip([
            Check::Exact,
            Check::Exact,
            Check::Random,
            Check::Random,
        ]) {
            let report = verdict.expect("the untouched key verifies");
            assert_eq!(report.contribution_hashes.len(), 1);
            assert_eq!(report.check, check);
        }
        let report = contribute::verify_from_r1cs(&r1cs, &ptau, &good, &msm, &msm).unwrap();
        assert_eq!(report.setup_backend, Some("cpu"));
        assert_eq!(report.check, Check::Exact);

        let table = section_table(&good);
        let pristine = std::fs::read(&good).unwrap();
        let bad = dir.join("bad.zkey");
        for t in TAMPERS.iter().chain(RANDOM_TAMPERS) {
            let mut bytes = pristine.clone();
            (t.edit)(&mut bytes, &table);
            std::fs::write(&bad, &bytes).unwrap();
            assert_rejected(
                verify_all(&r1cs, &ptau, &init, &bad),
                &format!("{circuit}: {}", t.what),
                t.expect,
                t.random,
            );
        }

        // The first chunk, and the last point of the first chunk, which only a section of
        // more than 2^20 points has.
        for (id, stride, expect, random) in [
            (
                5,
                64,
                "A section is not identical",
                "A section does not match",
            ),
            (
                8,
                64,
                "L section does not match",
                "L section does not match",
            ),
        ] {
            let points = body(&table, id).1 / stride;
            for at in [0, (1 << 20) - 1].into_iter().filter(|&at| at + 1 < points) {
                let mut bytes = pristine.clone();
                swap_from(&mut bytes, &table, id, stride, at);
                std::fs::write(&bad, &bytes).unwrap();
                assert_rejected(
                    verify_all(&r1cs, &ptau, &init, &bad),
                    &format!("{circuit}: section {id}, point {at}"),
                    expect,
                    random,
                );
            }
        }

        // The same shape and one coefficient different. Section 4 differs too, and
        // `csHash` is the one snarkjs checks first.
        let parsed = snarkrs_ceremony::r1cs::R1cs::open(&r1cs).unwrap();
        let term = parsed.constraints().unwrap()[0].a[0];
        let constraints = parsed
            .file()
            .sections()
            .iter()
            .find(|s| s.id == 2)
            .unwrap()
            .start;
        drop(parsed);
        // Coefficients are plain little-endian and nonzero, so one less is still canonical.
        let mut other = std::fs::read(&r1cs).unwrap();
        let at = constraints + term.coef_ptr as usize;
        for b in &mut other[at..at + 32] {
            let (v, borrow) = b.overflowing_sub(1);
            *b = v;
            if !borrow {
                break;
            }
        }
        let other_r1cs = dir.join("other.r1cs");
        std::fs::write(&other_r1cs, other).unwrap();
        let err = contribute::verify_from_r1cs(&other_r1cs, &ptau, &good, &msm, &msm)
            .expect_err("a key verified against a different circuit");
        assert!(err.to_string().contains("circuit does not match"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A key of another shape fails on the header, before any section is recomputed.
    let (small, big) = (
        bench.join("artifacts/js_2x2_d32/circuit.r1cs"),
        bench.join("artifacts/anon-aadhaar/circuit.r1cs"),
    );
    let ptau = bench.join("ptau/ppot_0080_21.ptau");
    if !small.is_file() || !big.is_file() || !ptau.is_file() {
        resources::skip("the_r1cs_form_rejects_...: no js_2x2_d32 or anon-aadhaar");
        return;
    }
    let dir = tmp_dir("r1cs-form-shape");
    let init = dir.join("init.zkey");
    let key = dir.join("key.zkey");
    snarkrs_ceremony::setup::setup(&small, &ptau, &init, &msm).unwrap();
    contribute::contribute(&init, &key, None, "shape", &CpuKeyScale).unwrap();
    let err = contribute::verify_from_r1cs(&big, &ptau, &key, &msm, &msm)
        .expect_err("a key verified against a circuit of another shape");
    assert!(
        err.to_string().contains("different circuit parameters"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
