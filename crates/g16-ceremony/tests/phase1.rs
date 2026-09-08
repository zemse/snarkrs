//! Phase 1 against snarkjs 0.7.6's own bytes.
//!
//! All three commands are deterministic given their inputs, so every assertion here is a
//! byte comparison rather than a property:
//!
//! * `ptau new` takes only `power`;
//! * `ptau beacon` takes the input file, the beacon hash and the iteration exponent;
//! * `ptau contribute` takes the input file, the entropy string and the 64 OS-random bytes
//!   it mixes in, and [`g16_ceremony::transcript::rng_from_entropy_with`] pins the last of
//!   those.
//!
//! `.ptau` is gitignored repo-wide, so the oracles cannot be checked in and are pinned as
//! SHA-256 instead. They were produced by snarkjs 0.7.6 through its own module entry
//! points, with `crypto.randomFillSync` stubbed to hand back the bytes 0..63:
//!
//! ```js
//! import crypto from "crypto";
//! const OS = new Uint8Array(64); for (let i=0;i<64;i++) OS[i] = i;
//! crypto.randomFillSync = (a) => { a.set(OS.subarray(0, a.length)); return a; };
//! const { getCurveFromName } = await import("ffjavascript");
//! const curve = await getCurveFromName("bn128");
//! const nw = (await import("snarkjs/src/powersoftau_new.js")).default;
//! const co = (await import("snarkjs/src/powersoftau_contribute.js")).default;
//! const be = (await import("snarkjs/src/powersoftau_beacon.js")).default;
//! await nw(curve, 8, "new_8.ptau");
//! await co("new_8.ptau", "c_8.ptau", "phase1 test", "some fixed entropy");
//! await be("c_8.ptau", "cb_8.ptau", null, BEACON_HEX, 10);
//! ```
//!
//! Each case is asserted at three levels, so a failure says which layer moved: the
//! response hash covers the previous challenge, the compressed points and the pubkey; the
//! next challenge covers the uncompressed read-back of what was written; and the file
//! digest covers everything else, meaning the container framing, the LEM encoding, the
//! contribution record and `partialHash`. A response hash that matches while the file
//! digest does not is an encoding or framing bug; a response hash that does not match is
//! the point arithmetic or the RNG.

use std::path::{Path, PathBuf};

use g16_ceremony::phase1;
use g16_ceremony::ptau::Ptau;
use g16_ceremony::transcript::{blake2b512, rng_from_entropy_with, Digest};
use g16_ceremony::{CeremonyError, ContributionKind, ContributionParams, CpuKeyScale};
use g16_field::AffineRepr;
use sha2::{Digest as _, Sha256};

/// The beacon every case below uses, `01 02 .. 20`.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

/// The 64 bytes `powersoftau contribute` would have taken from the OS, pinned so the
/// command becomes reproducible. `misc.js:186-190` reads them through
/// `crypto.randomFillSync` and hashes them before the entropy string.
fn os_bytes() -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, b) in out.iter_mut().enumerate() {
        *b = i as u8;
    }
    out
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-phase1-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

fn digest_of(path: &Path) -> String {
    hex(&Sha256::digest(std::fs::read(path).unwrap()))
}

fn beacon_bytes() -> Vec<u8> {
    unhex(BEACON_HEX)
}

/// The three hashes that identify one produced file, checked outermost last.
struct Oracle {
    /// snarkjs' "Contribution Response Hash".
    response: &'static str,
    /// The `nextChallenge` stored in the record.
    next: &'static str,
    /// SHA-256 of the whole `.ptau`.
    file: &'static str,
}

fn assert_matches(what: &str, report: &phase1::Phase1Report, path: &Path, oracle: &Oracle) {
    assert_eq!(
        hex(&report.response_hash),
        oracle.response,
        "{what}: response hash differs, so the compressed points, the pubkey or the previous challenge did"
    );
    assert_eq!(
        hex(&report.next_challenge),
        oracle.next,
        "{what}: next challenge differs, so the uncompressed read-back of the new sections did"
    );
    assert_eq!(
        digest_of(path),
        oracle.file,
        "{what}: file digest differs while both hashes match, so this is framing, LEM encoding, the record or partialHash"
    );
}

// -------------------------------------------------------------------- ptau new

/// `powersoftau new bn128 <power>` at 8, 10 and 12, byte for byte, plus the first
/// challenge hash it returns.
///
/// A fresh file is nothing but the generator repeated, so this pins the header (both
/// `power` fields, the plain little-endian `q`), the four element counts as a function of
/// `power`, the LEM generator constants and the empty section 7 in one comparison.
#[test]
fn ptau_new_reproduces_snarkjs_byte_for_byte() {
    let cases: [(u32, &str, &str); 3] = [
        (
            8,
            "199d173eb7abadfbe82650a9390813f641f9e5a27dd894dd5721304ac016da4f",
            "219cd1f3eab9d2a70ebec1e89ce41ade8d761eb39fd6702acda5776283026ac8\
             81746beac81c214b887e9102e84c8341824fd983f4e7df844d150ddf5fd2fe48",
        ),
        (
            10,
            "a28944526c62a4feb017bf42ccbac359d5503fbe878a9a6f8ddda58fea5d2d6c",
            "95f0b4499e50f8da383b0d74c174c1698bdffe1b35066754005889a147849bbf\
             8d64ff6c989bd89a4736b569a99a1c83a50dc181e9fe1d4d23d1888a98b3157e",
        ),
        (
            12,
            "18dd67751dd0659bcd6f58d961ef478d855f1695325ad9db9cd68e30e411e24a",
            "9e63a5f62b96538daaed2372481920d1a40b91959ea38ef9f5f6a3033b886516\
             0710d067c09d09615f928ea517bcdf49ad75abd2c8340b400e3b18e968b4ffef",
        ),
    ];
    let dir = tmp_dir("new");
    for (power, file_digest, challenge) in cases {
        let out = dir.join(format!("new_{power}.ptau"));
        let got = phase1::ptau_new(power, &out).unwrap();
        assert_eq!(
            hex(&got),
            challenge.replace(' ', ""),
            "power {power}: first challenge hash"
        );
        assert_eq!(digest_of(&out), file_digest, "power {power}: file bytes");
        // `384 * 2^power + 208` for a fresh file (`s7 = 4`), the size formula the ptau
        // spec derives from the section table.
        assert_eq!(
            std::fs::metadata(&out).unwrap().len(),
            384 * (1u64 << power) + 208,
            "power {power}: file length"
        );
    }
}

/// The first challenge hash of a fresh file is what `Ptau::last_challenge` has to return
/// for it, since there is no record to read it from
/// (`powersoftau_contribute.js:54-58`).
#[test]
fn a_fresh_file_reports_the_first_challenge_as_its_last() {
    let dir = tmp_dir("first-challenge");
    let out = dir.join("new_8.ptau");
    let challenge = phase1::ptau_new(8, &out).unwrap();
    let ptau = Ptau::open(&out).unwrap();
    assert_eq!(ptau.last_challenge().unwrap(), challenge);
    assert!(ptau.contributions().unwrap().is_empty());
}

/// `cli.js:701` rejects anything outside `1..=28`, and 28 is also the two-adicity of
/// `Fr`, so a bigger accumulator could never be prepared for phase 2.
#[test]
fn ptau_new_rejects_a_power_snarkjs_would_reject() {
    let dir = tmp_dir("new-bounds");
    for power in [0u32, 29, 64] {
        let out = dir.join(format!("bad_{power}.ptau"));
        assert!(
            matches!(
                phase1::ptau_new(power, &out),
                Err(CeremonyError::BadParams(_))
            ),
            "power {power} should be refused"
        );
    }
}

// -------------------------------------------------------------------- beacon

/// `powersoftau beacon` straight onto a fresh file, at powers 8 and 10.
///
/// This is the case that exercises the first-challenge seed path: the key is derived from
/// `calculateFirstChallengeHash(power)` rather than from a stored record, so a wrong
/// element count in that hash produces a different key and a different file, not merely a
/// different digest.
#[test]
fn beacon_on_a_fresh_file_reproduces_snarkjs() {
    let cases: [(u32, Oracle); 2] = [
        (
            8,
            Oracle {
                response: "3e21215e698c5fcb23400a4dc05c2b47ead98c2bb80e7e25ea56fe19115f8e92505b11a2cf5d17b92bebb7d14650d3e2694a54da66544f5a846545204a3f24ae",
                next: "91f869f77924551d534b3b27d6599d7d7a09d25dea1b17a46db2e08d927eb5361c46353adfad7092a86e22e2045736306a460149afce7377c5e1976ae5fe171a",
                file: "6b828df149840d77a11ff318aa3634a5ed8840e5eaac9b08af7d1df079f4e13c",
            },
        ),
        (
            10,
            Oracle {
                response: "b242269e39738005d1f2b55c5fec43dd323c9c9f0ff18036eb664c4572762f7bd24a078f30bc04f9761fa0359541130d156aef4deb7d31fab35548fc9bcf4d28",
                next: "e9b488bf847e2bf3bd0a4c15bd4d830cd9038c71c360cd2684c39ce34f5b57214849575a0bcab68b52224ff60ed904ccafae66e598a0fcffa65812e6ea6689fe",
                file: "3061f935928c520a54092580068867f70360f24ee9bdbda917e7546d9a48ef86",
            },
        ),
    ];
    let dir = tmp_dir("beacon-fresh");
    for (power, oracle) in cases {
        let fresh = dir.join(format!("new_{power}.ptau"));
        let out = dir.join(format!("beacon_{power}.ptau"));
        phase1::ptau_new(power, &fresh).unwrap();
        let report = phase1::beacon(
            &fresh,
            &out,
            Some("final beacon"),
            &beacon_bytes(),
            12,
            &CpuKeyScale,
        )
        .unwrap();
        assert_eq!(report.index, 1);
        assert_eq!(report.kind, ContributionKind::Beacon);
        assert_matches(&format!("beacon power {power}"), &report, &out, &oracle);
    }
}

// -------------------------------------------------------------------- contribute

/// `powersoftau contribute` with both of its entropy sources pinned, at powers 8 and 10.
///
/// If this one fails while [`beacon_on_a_fresh_file_reproduces_snarkjs`] passes, the point
/// arithmetic and the file writer are fine and the divergence is upstream in
/// `transcript::rng_from_entropy_with`: the beacon path seeds ChaCha from an iterated
/// SHA-256, this one seeds it from BLAKE2b over the OS bytes and the entropy string.
#[test]
fn contribute_with_pinned_entropy_reproduces_snarkjs() {
    let cases: [(u32, Oracle); 2] = [
        (
            8,
            Oracle {
                response: "92d70de375137f43a969d061850ecb53ce2be7de72f5b1506222ff0ddbf1abb1181fc104ec59e1ec05988eb90b8992de0106fa2ff25a56106bae7daf8dd98b84",
                next: "75cb618a1af584e14ffe77d00452a0068bc3f0abef97824749e7935ccc7873299d8e24f7c2c349ca7c9e98054fa6e67711ca922efcffb1a1104130bca5361828",
                file: "e306334b684505868f327239cdafd27ef3962273463c7e9929a819ad4cbf03fb",
            },
        ),
        (
            10,
            Oracle {
                response: "33a0ea64b14f101197d643cdf854de82aa55e415204267f5951137a5750857a778d54804f502e5ded68f1d7dfdafbfe9b9e9dd6728419040478f28121f76c2ca",
                next: "c08ab9c9dc9ded328f72b65d387181951c98292644bd1bf0bf0805ecff1b2015346dea22a69756d0901874526d5955f8652df4a77759e7af490a02d99cc97020",
                file: "07f821c18696297155ba3566a74dead56d518b60a684c32556436b30f902ed36",
            },
        ),
    ];
    let dir = tmp_dir("contribute");
    for (power, oracle) in cases {
        let fresh = dir.join(format!("new_{power}.ptau"));
        let out = dir.join(format!("contrib_{power}.ptau"));
        phase1::ptau_new(power, &fresh).unwrap();
        let params = ContributionParams {
            name: Some("phase1 test".into()),
            ..Default::default()
        };
        let report = phase1::contribute_with(
            &fresh,
            &out,
            params,
            rng_from_entropy_with(&os_bytes(), "some fixed entropy"),
            &CpuKeyScale,
        )
        .unwrap();
        assert_eq!(report.index, 1);
        assert_eq!(report.kind, ContributionKind::Contribute);
        assert_matches(&format!("contribute power {power}"), &report, &out, &oracle);
    }
}

// -------------------------------------------------------------------- the chain

/// A two-record chain: contribute, then beacon on top of it.
///
/// The second record is the one worth having. Its key is seeded from the first record's
/// stored `nextChallenge`, its `partialHash` resumes a stream that started from that same
/// digest, and section 7 has to carry the earlier record forward unchanged, so a file
/// digest match here says the chain is joined the way snarkjs joins it and not merely that
/// one contribution is right.
#[test]
fn a_contribution_then_a_beacon_reproduces_snarkjs() {
    let dir = tmp_dir("chain");
    let fresh = dir.join("new_8.ptau");
    let first = dir.join("c_8.ptau");
    let second = dir.join("cb_8.ptau");
    phase1::ptau_new(8, &fresh).unwrap();

    let params = ContributionParams {
        name: Some("phase1 test".into()),
        ..Default::default()
    };
    phase1::contribute_with(
        &fresh,
        &first,
        params,
        rng_from_entropy_with(&os_bytes(), "some fixed entropy"),
        &CpuKeyScale,
    )
    .unwrap();

    // snarkjs was called with `null` for the name here, so the beacon record's params are
    // just the two beacon fields and there is no type-1 entry.
    let report = phase1::beacon(&first, &second, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
    assert_eq!(report.index, 2);
    assert_matches(
        "chained beacon",
        &report,
        &second,
        &Oracle {
            response: "745133b3975ecf62c7006219f4d809d4ef3a8ceeff02290535641bfde25dbb0236b37cdd75f0b3c159ddd6f1ba51342be6204c06f0ab658c0078d567e2094d58",
            next: "76159af96841373913b9affdb439e40302f62fa1ad50ceeae60970771d7c2d7ce3cebae96245ce63f65e13df00d08d9308f0093ae3cb9942d1b2c7e28e0e2033",
            file: "12ead6233d2989fe1988d34003558e542868556295f03026c5a9221bdf60a99c",
        },
    );

    // Read the chain back through our own reader: both records decode, the stored
    // `partialHash` reconstructs the response hash snarkjs printed, and record 1's
    // `nextChallenge` is what record 2 was keyed from.
    let ptau = Ptau::open(&second).unwrap();
    let contributions = ptau.contributions().unwrap();
    assert_eq!(contributions.len(), 2);
    assert_eq!(contributions[0].kind, ContributionKind::Contribute);
    assert_eq!(contributions[0].params.name.as_deref(), Some("phase1 test"));
    assert_eq!(contributions[1].kind, ContributionKind::Beacon);
    assert_eq!(contributions[1].params.name, None);
    assert_eq!(contributions[1].params.num_iterations_exp, Some(10));
    assert_eq!(contributions[1].params.beacon_hash, Some(beacon_bytes()));
    assert_eq!(
        hex(&contributions[0].response_hash().unwrap()),
        "92d70de375137f43a969d061850ecb53ce2be7de72f5b1506222ff0ddbf1abb1181fc104ec59e1ec05988eb90b8992de0106fa2ff25a56106bae7daf8dd98b84",
    );
    assert_eq!(
        contributions[1].response_hash().unwrap(),
        report.response_hash
    );
    assert_eq!(ptau.last_challenge().unwrap(), report.next_challenge);
}

/// Each contribution's stored `tauG1` is the second element of section 2 and its
/// `alphaG1` the first of section 4 (`powersoftau_contribute.js:76-84`). Nothing else in
/// the file states that, so a swapped `firstPoints` index would only surface here or in
/// `snarkjs powersoftau verify`.
#[test]
fn the_record_holds_the_points_the_verifier_looks_for() {
    let dir = tmp_dir("record-points");
    let fresh = dir.join("new_8.ptau");
    let out = dir.join("beacon_8.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    phase1::beacon(&fresh, &out, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();

    let ptau = Ptau::open(&out).unwrap();
    let c = &ptau.contributions().unwrap()[0];
    assert_eq!(c.tau_g1, ptau.g1_points(2, 1, 1).unwrap()[0]);
    assert_eq!(c.tau_g2, ptau.g2_points(3, 1, 1).unwrap()[0]);
    assert_eq!(c.alpha_g1, ptau.g1_points(4, 0, 1).unwrap()[0]);
    assert_eq!(c.beta_g1, ptau.g1_points(5, 0, 1).unwrap()[0]);
    assert_eq!(c.beta_g2, ptau.g2_points(6, 0, 1).unwrap()[0]);
    // Element 0 of sections 2 and 3 stays the generator: only the exponent moves
    // (`powersoftau_verify.js:184-187`).
    assert!(ptau.g1_points(2, 0, 1).unwrap()[0].is_on_curve());
    assert_eq!(
        ptau.g1_points(2, 0, 1).unwrap()[0],
        g16_field::G1Affine::generator()
    );
    assert_eq!(
        ptau.g2_points(3, 0, 1).unwrap()[0],
        g16_field::G2Affine::generator()
    );
}

/// Reusing the same entropy for a second contribution reuses the same three private
/// scalars, because `getRandomRng` seeds only from the OS bytes and the entropy string:
/// the challenge reaches `createPTauKey` but only `getG2sp` consumes it
/// (`keypair.js:61-73`). The accumulator still moves, since tau is applied twice, and the
/// two records still differ in every stored point. Nothing in the format detects the
/// reuse, which is the argument for a beacon at the end of a ceremony rather than a
/// second contribution from the same machine.
#[test]
fn the_same_entropy_twice_reuses_the_key_but_still_moves_the_accumulator() {
    let dir = tmp_dir("same-entropy");
    let fresh = dir.join("new_8.ptau");
    let one = dir.join("one.ptau");
    let two = dir.join("two.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    let a = phase1::contribute_with(
        &fresh,
        &one,
        ContributionParams::default(),
        rng_from_entropy_with(&os_bytes(), "same"),
        &CpuKeyScale,
    )
    .unwrap();
    let b = phase1::contribute_with(
        &one,
        &two,
        ContributionParams::default(),
        rng_from_entropy_with(&os_bytes(), "same"),
        &CpuKeyScale,
    )
    .unwrap();
    assert_ne!(a.response_hash, b.response_hash);
    let ptau = Ptau::open(&two).unwrap();
    let c = ptau.contributions().unwrap();
    assert_ne!(c[0].tau_g1, c[1].tau_g1);
    // Same `g1_s`, because it came off the same stream at the same position.
    assert_eq!(c[0].pubkeys.tau.g1_s, c[1].pubkeys.tau.g1_s);
    // Different `g2_sp`, because that one is derived from the challenge.
    assert_ne!(c[0].pubkeys.tau.g2_sp, c[1].pubkeys.tau.g2_sp);
    assert_eq!(ptau.last_challenge().unwrap(), b.next_challenge);
}

/// The same chain at power 14, where every hashed section spans more than one chunk.
///
/// `processSection` hashes `floor((1<<20)/sG)` points per `update`, so section 2 takes two
/// chunks at 32767 G1 points and section 3 takes two at 16384 G2. Three things only start
/// mattering here. The scalar has to be carried across a chunk as `t *= inc^n` rather than
/// restarted. `partialHash` is a serialised BLAKE2b state whose bytes past `pos` are
/// whatever the last zero-copy compression left behind, so a different chunk size writes
/// different bytes into the file even though the digest is unchanged. And the multi-chunk
/// read-back of the new file for the uncompressed pass has to seek to the right offsets.
///
/// A power-8 file exercises none of that: every section there is a single chunk.
#[test]
fn multi_chunk_sections_reproduce_snarkjs() {
    let dir = tmp_dir("multi-chunk");
    let fresh = dir.join("new_14.ptau");
    let first = dir.join("c_14.ptau");
    let second = dir.join("cb_14.ptau");

    phase1::ptau_new(14, &fresh).unwrap();
    assert_eq!(
        digest_of(&fresh),
        "6dcf0303754e8c8a2b8fbd40f808d0308db1f52c29447caa16e08488f0b8d84e"
    );

    let params = ContributionParams {
        name: Some("phase1 test".into()),
        ..Default::default()
    };
    let report = phase1::contribute_with(
        &fresh,
        &first,
        params,
        rng_from_entropy_with(&os_bytes(), "some fixed entropy"),
        &CpuKeyScale,
    )
    .unwrap();
    assert_matches(
        "power 14 contribute",
        &report,
        &first,
        &Oracle {
            response: "f4490932a9e0cc560a07fdb77d5fa8334b42b61f5b2d04d609b65adbd0713eecbaecf3a4cd4a9a1b107a06bf02d30d39d2bb2924da51f712185081437d01ae70",
            next: "1fec08a3b7115afa75108659c6af9ce0d9290faf11012ce641899badb1a08ad24433834122d30603a4ca74e20b6976c87e12daa7b2605e492868a0740df9bd42",
            file: "2bbd2300aafbbe5ae9c9d7d5ec146e5fe790f21b3f06eb2762d710f5e60b0523",
        },
    );

    let report = phase1::beacon(&first, &second, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
    assert_matches(
        "power 14 beacon",
        &report,
        &second,
        &Oracle {
            response: "ae6f4c5ab901959edb9888b3f8ed94ff110bc34d1d79cebe4675923a1b033796973c4169f36747efa51e9e6d7dcb68cbc451cc7176f9018af56719f883928c6d",
            next: "80ae912f94dfc64e95c400e192599ac2adcf15c831674ac6132d6e73eea825fb1d3e2aa7eb4146e9714070153a3dec1cef3422cef4258ca43592d0d52942dae7",
            file: "7937ccb96cd3153afed624cc958a73ccf93c192c5650fa7aaa37409cb668d88f",
        },
    );
}

// -------------------------------------------------------------------- verify

/// Build a two-record chain at power 8 and hand it to our own verifier.
///
/// The chain is the one `a_contribution_then_a_beacon_reproduces_snarkjs` pins to
/// snarkjs' bytes, so "our verifier accepts it" and "snarkjs' verifier accepts it" are the
/// same statement about the same file.
fn a_chain(dir: &Path) -> PathBuf {
    let fresh = dir.join("new_8.ptau");
    let first = dir.join("c_8.ptau");
    let second = dir.join("cb_8.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    phase1::contribute_with(
        &fresh,
        &first,
        ContributionParams {
            name: Some("phase1 test".into()),
            ..Default::default()
        },
        rng_from_entropy_with(&os_bytes(), "some fixed entropy"),
        &CpuKeyScale,
    )
    .unwrap();
    phase1::beacon(&first, &second, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
    second
}

#[test]
fn verify_accepts_a_chain_we_built() {
    let dir = tmp_dir("verify-ok");
    let chain = a_chain(&dir);
    let report = phase1::verify(&chain).unwrap();
    assert_eq!(report.power, 8);
    assert_eq!(report.ceremony_power, 8);
    assert!(!report.prepared);
    assert!(report.next_challenge_checked);
    assert_eq!(report.contribution_hashes.len(), 2);
    assert_eq!(
        hex(&report.contribution_hashes[1]),
        "745133b3975ecf62c7006219f4d809d4ef3a8ceeff02290535641bfde25dbb0236b37cdd75f0b3c159ddd6f1ba51342be6204c06f0ab658c0078d567e2094d58"
    );
}

/// A fresh accumulator is well-formed and worthless: snarkjs refuses it outright rather
/// than reporting it valid (`powersoftau_verify.js:151-154`).
#[test]
fn verify_refuses_a_file_with_no_contribution() {
    let dir = tmp_dir("verify-empty");
    let fresh = dir.join("new_8.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    assert!(matches!(
        phase1::verify(&fresh),
        Err(CeremonyError::Verification(_))
    ));
}

/// Overwrite `len` bytes at `at` and return the path to the damaged copy.
fn tampered(src: &Path, dst: &Path, at: usize, bytes: &[u8]) -> PathBuf {
    let mut data = std::fs::read(src).unwrap();
    data[at..at + bytes.len()].copy_from_slice(bytes);
    std::fs::write(dst, data).unwrap();
    dst.to_path_buf()
}

/// A single replaced point inside section 2 is what the random linear combination exists
/// to catch: it costs one pairing rather than `2n-1` of them, and nothing else in the file
/// notices.
#[test]
fn verify_catches_a_replaced_point_in_the_powers() {
    let dir = tmp_dir("verify-points");
    let chain = a_chain(&dir);
    // Section 2's payload starts after the 12-byte preamble, the 12-byte header entry plus
    // its 44 bytes, and its own 12-byte entry: element 5 is 5 * 64 bytes into that.
    let at = 12 + 12 + 44 + 12 + 5 * 64;
    let broken = tampered(&chain, &dir.join("broken.ptau"), at, &[0u8; 64]);
    let err = phase1::verify(&broken).unwrap_err();
    assert!(
        matches!(&err, CeremonyError::Verification(m) if m.contains("tauG1")),
        "expected a tauG1 powers failure, got {err}"
    );
}

/// The contribution record is what a verifier trusts about the chain, so a record whose
/// `tauG1` no longer matches the section it claims to describe has to fail even though
/// every point in the file is still a valid group element and the container still parses.
#[test]
fn verify_catches_a_rewritten_contribution_record() {
    let dir = tmp_dir("verify-record");
    let chain = a_chain(&dir);
    let data = std::fs::read(&chain).unwrap();

    // Everything before section 7 is fixed by power 8: the 12-byte preamble, then each
    // section's 12-byte entry header and its payload.
    let n = 256usize;
    let s7 = 12
        + (12 + 44)
        + (12 + (2 * n - 1) * 64)
        + (12 + n * 128)
        + 2 * (12 + n * 64)
        + (12 + 128)
        + 12;
    assert_eq!(
        u32::from_le_bytes(data[s7..s7 + 4].try_into().unwrap()),
        2,
        "section 7 is not where the layout says it is"
    );
    // `paramLength` is the last four bytes of the 1504-byte fixed prefix.
    let first_params =
        u32::from_le_bytes(data[s7 + 4 + 1500..s7 + 4 + 1504].try_into().unwrap()) as usize;
    let first_tau_g1 = data[s7 + 4..s7 + 4 + 64].to_vec();
    let second = s7 + 4 + 1504 + first_params;
    assert_ne!(
        first_tau_g1,
        data[second..second + 64],
        "record layout moved"
    );

    // Swap the beacon's `tauG1` for the earlier contribution's, which is a real point from
    // the same ceremony rather than random bytes.
    let broken = tampered(&chain, &dir.join("record.ptau"), second, &first_tau_g1);
    let err = phase1::verify(&broken).unwrap_err();
    assert!(
        matches!(&err, CeremonyError::Verification(_)),
        "expected a verification failure, got {err}"
    );
}

/// A prepared file adds four Lagrange sections, and the check that they belong to this
/// accumulator moves the transform onto the scalars: `<raw, r>` against
/// `<lagrange, fft(r)>`. `bench/ptau/local_13.ptau` is snarkjs' own
/// `powersoftau preparephase2` output, so this also pins that our forward NTT agrees with
/// ffjavascript's root-of-unity convention, which no `.ptau` header states.
#[test]
fn verify_checks_the_lagrange_sections_of_a_prepared_file() {
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/ptau/local_13.ptau");
    if !local.exists() {
        eprintln!("skipping: {} is not present", local.display());
        return;
    }
    let report = phase1::verify(&local).unwrap();
    assert_eq!(report.power, 13);
    assert!(report.prepared);
    assert!(report.next_challenge_checked);
    assert_eq!(report.contribution_hashes.len(), 1);
}

// -------------------------------------------------------------------- refusals

/// `power != ceremonyPower` marks a `powersoftau truncate` output, whose points are a
/// prefix of the ones the chain hashed. Contributing into it could never reproduce the
/// next challenge, so both commands refuse outright
/// (`powersoftau_contribute.js:37-40`, `powersoftau_beacon.js:47-50`).
#[test]
fn a_truncated_ceremony_refuses_both_commands() {
    let ppot = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/ptau/ppot_0080_15.ptau");
    if !ppot.exists() {
        eprintln!("skipping: {} is not present", ppot.display());
        return;
    }
    let dir = tmp_dir("truncated");
    let out = dir.join("nope.ptau");
    assert!(matches!(
        phase1::beacon(&ppot, &out, None, &beacon_bytes(), 10, &CpuKeyScale),
        Err(CeremonyError::TruncatedCeremony {
            power: 15,
            ceremony_power: 28
        })
    ));
    assert!(matches!(
        phase1::contribute_with(
            &ppot,
            &out,
            ContributionParams::default(),
            rng_from_entropy_with(&os_bytes(), "x"),
            &CpuKeyScale,
        ),
        Err(CeremonyError::TruncatedCeremony { .. })
    ));
}

/// The bounds `powersoftau_beacon.js:26-42` checks before it opens a file. The accepted
/// hex set is exactly "non-empty, even length, all hex": snarkjs scans `[\da-f]{2}/gi` and
/// then asserts `bytes*2 == str.length`, and it strips a `0x` prefix before the scan but
/// not before that assertion, so `0x...` fails there too.
#[test]
fn beacon_arguments_are_bounded_the_way_snarkjs_bounds_them() {
    assert!(phase1::parse_beacon_args(BEACON_HEX, "10").is_ok());
    assert!(phase1::parse_beacon_args(BEACON_HEX, "63").is_ok());
    assert_eq!(
        phase1::parse_beacon_args("0aFF", "10").unwrap().0,
        vec![0x0a, 0xff],
        "uppercase hex is accepted, the scan is case-insensitive"
    );
    for bad in ["", "0102030", "0x0102", "zz", "01 02"] {
        assert!(
            phase1::parse_beacon_args(bad, "10").is_err(),
            "beacon hash {bad:?} should be refused"
        );
    }
    for bad in ["9", "64", "255", "", "12abc", "-1"] {
        assert!(
            phase1::parse_beacon_args(BEACON_HEX, bad).is_err(),
            "numIterationsExp {bad:?} should be refused"
        );
    }
    // 255 bytes is the cap: the length is written into a single `u8` in the params TLV.
    let long = "ab".repeat(255);
    assert!(phase1::parse_beacon_args(&long, "10").is_ok());
    let too_long = "ab".repeat(256);
    assert!(phase1::parse_beacon_args(&too_long, "10").is_err());
}

/// The same bounds again on the typed entry point, which the library exposes separately
/// from the string parser and which a caller could reach without going through the CLI.
#[test]
fn the_beacon_entry_point_rechecks_its_own_bounds() {
    let dir = tmp_dir("beacon-bounds");
    let fresh = dir.join("new_8.ptau");
    let out = dir.join("nope.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    for (hash, exp) in [
        (vec![], 10u8),
        (vec![1u8; 256], 10),
        (vec![1u8], 9),
        (vec![1u8], 64),
    ] {
        assert!(
            matches!(
                phase1::beacon(&fresh, &out, None, &hash, exp, &CpuKeyScale),
                Err(CeremonyError::BadParams(_))
            ),
            "{} bytes / exponent {exp} should be refused",
            hash.len()
        );
    }
}

/// A contributor name is truncated to the first 64 **UTF-16 code units** and only then
/// encoded (`powersoftau_utils.js:261`), so the byte length of a stored name is not the
/// character count and a multi-byte name keeps fewer characters than a `u8` would suggest.
///
/// The `u8` length field cannot actually overflow from a name: 64 UTF-16 units is at most
/// 192 UTF-8 bytes, since a three-byte character is one unit and a four-byte one is two.
/// [`ContributionParams::encode`] still guards it, because the same TLV carries a beacon
/// hash, which is bounded separately and by a different caller.
#[test]
fn a_long_name_is_truncated_the_way_snarkjs_truncates_it() {
    let dir = tmp_dir("long-name");
    let fresh = dir.join("new_8.ptau");
    let out = dir.join("named.ptau");
    phase1::ptau_new(8, &fresh).unwrap();
    let name: String = std::iter::repeat_n('\u{1F600}', 64).collect();
    let params = ContributionParams {
        name: Some(name),
        ..Default::default()
    };
    phase1::contribute_with(
        &fresh,
        &out,
        params,
        rng_from_entropy_with(&os_bytes(), "x"),
        &CpuKeyScale,
    )
    .unwrap();
    let ptau = Ptau::open(&out).unwrap();
    let stored = ptau.contributions().unwrap()[0]
        .params
        .name
        .clone()
        .unwrap();
    assert_eq!(
        stored.chars().count(),
        32,
        "64 UTF-16 units of a four-byte character is 32 characters"
    );
    assert_eq!(stored.len(), 128, "and 128 bytes in the params TLV");
}

/// `blake2b512("")` is the "Blank Contribution Hash" the first challenge starts from
/// (`powersoftau_utils.js:322`), and it is the one input to `first_challenge_hash` that is
/// not derived from `power`.
#[test]
fn the_first_challenge_starts_from_the_blank_hash() {
    assert_eq!(
        hex(&blake2b512(b"")),
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
         d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
    );
    // Distinct per power, so a file cannot be replayed at a different size.
    let hashes: Vec<Digest> = (8..=12).map(phase1::first_challenge_hash).collect();
    for (i, a) in hashes.iter().enumerate() {
        for b in hashes.iter().skip(i + 1) {
            assert_ne!(a, b);
        }
    }
}
