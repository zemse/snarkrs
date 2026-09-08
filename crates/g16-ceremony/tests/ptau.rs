//! Read every `.ptau` in `bench/ptau` and check the parse against the file itself.
//!
//! These are real ceremony files, not fixtures: `local_13` and `local_19` came out of
//! `snarkjs powersoftau` locally, and `ppot_0080_*` are the PSE ceremony's perpetual
//! powers of tau, prepared and then truncated to each power. That makes them the sharpest
//! oracle available for the section table, because every one of the size formulas is
//! overdetermined: `expected_bytes` rebuilds the whole file length from `power` plus
//! section 7's declared length alone, so a single wrong element count misses the real byte
//! total and the assertion names the file it missed on.
//!
//! `Ptau::contributions` is covered only by `recomputed_g2_sp_matches_the_stored_pubkey`,
//! since `last_challenge` walks the same 62 records without paying for a pubkey. That one
//! test is not optional: `g2_sp` is the single field in a record that is recomputed rather
//! than read, so nothing else here would notice it being derived off the wrong challenge.

use std::path::{Path, PathBuf};

use g16_ceremony::ptau::*;
use g16_ceremony::transcript::same_ratio;
use g16_field::{AffineRepr, G1Affine, G2Affine};

/// power, ceremony power, contributions, and the byte total the formulas have to land on.
struct Expected {
    name: &'static str,
    power: u32,
    ceremony_power: u32,
    n_contributions: usize,
    bytes: u64,
}

/// Every file the repo ships, with the numbers read out of it independently. The ppot
/// files all carry `ceremonyPower = 28` because they are `powersoftau truncate` output of
/// the same 2^28 ceremony, and all 62 contributions with them.
const FILES: &[Expected] = &[
    Expected {
        name: "local_13.ptau",
        power: 13,
        ceremony_power: 13,
        n_contributions: 1,
        bytes: 9_438_631,
    },
    Expected {
        name: "local_19.ptau",
        power: 19,
        ceremony_power: 19,
        n_contributions: 1,
        bytes: 603_981_223,
    },
    Expected {
        name: "ppot_0080_15.ptau",
        power: 15,
        ceremony_power: 28,
        n_contributions: 62,
        bytes: 37_842_066,
    },
    Expected {
        name: "ppot_0080_16.ptau",
        power: 16,
        ceremony_power: 28,
        n_contributions: 62,
        bytes: 75_590_802,
    },
    Expected {
        name: "ppot_0080_17.ptau",
        power: 17,
        ceremony_power: 28,
        n_contributions: 62,
        bytes: 151_088_274,
    },
    Expected {
        name: "ppot_0080_18.ptau",
        power: 18,
        ceremony_power: 28,
        n_contributions: 62,
        bytes: 302_083_218,
    },
    Expected {
        name: "ppot_0080_19.ptau",
        power: 19,
        ceremony_power: 28,
        n_contributions: 62,
        bytes: 604_073_106,
    },
];

fn ptau_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/ptau")
}

/// The files are large and not in git, so a checkout without them skips rather than fails.
fn present() -> Vec<(&'static Expected, PathBuf)> {
    FILES
        .iter()
        .map(|e| (e, ptau_dir().join(e.name)))
        .filter(|(_, p)| p.is_file())
        .collect()
}

#[test]
fn every_shipped_file_is_complete_and_prepared() {
    let files = present();
    assert!(!files.is_empty(), "no .ptau in {}", ptau_dir().display());
    for (want, path) in files {
        let file = Ptau::open(&path).unwrap_or_else(|e| panic!("{}: {e}", want.name));
        let info = file.info();
        assert_eq!(info.power, want.power, "{}", want.name);
        assert_eq!(info.ceremony_power, want.ceremony_power, "{}", want.name);
        assert_eq!(
            file.header().is_truncated_ceremony(),
            want.power != want.ceremony_power,
            "{}",
            want.name
        );
        assert!(
            info.prepared,
            "{} should carry sections 12 to 15",
            want.name
        );
        assert!(file.is_prepared(), "{}", want.name);
        assert_eq!(info.declared_sections, 11, "{}", want.name);
        assert_eq!(info.sections.len(), 11, "{}", want.name);
        assert!(
            info.is_complete(),
            "{} reports incomplete: {info:?}",
            want.name
        );
        assert!(info.truncation.is_none(), "{}", want.name);
        assert_eq!(info.file_bytes, want.bytes, "{}", want.name);
        assert_eq!(info.chain_end, want.bytes, "{}", want.name);
        // The whole point of the formulas: the size is rebuilt from `power` and section
        // 7's length, and has to land on the byte total of the real file.
        assert_eq!(info.expected_bytes, Some(want.bytes), "{}", want.name);
        assert_eq!(info.n_contributions, want.n_contributions, "{}", want.name);
        for report in &info.sections {
            assert!(
                !report.is_corrupt(),
                "{} section {}: {report:?}",
                want.name,
                report.id
            );
            assert!(
                report.is_complete(),
                "{} section {}: {report:?}",
                want.name,
                report.id
            );
        }
    }
}

/// The ids and lengths of a prepared file, spelled out once so a change to
/// `expected_section_bytes` cannot quietly agree with itself.
#[test]
fn section_table_matches_the_formulas() {
    let path = ptau_dir().join("local_13.ptau");
    if !path.is_file() {
        return;
    }
    let file = Ptau::open(&path).expect("local_13 opens");
    let n = 1u64 << 13;
    let want: &[(u32, u64)] = &[
        (1, 44),
        (2, (2 * n - 1) * 64),
        (3, n * 128),
        (4, n * 64),
        (5, n * 64),
        (6, 128),
        (7, 1515),
        (12, (4 * n - 1) * 64),
        (13, (2 * n - 1) * 128),
        (14, (2 * n - 1) * 64),
        (15, (2 * n - 1) * 64),
    ];
    let got: Vec<(u32, u64)> = file
        .file()
        .sections()
        .iter()
        .map(|s| (s.id, s.len as u64))
        .collect();
    assert_eq!(got, want);
    for &(id, len) in want {
        // Section 7 is the one length that is not a function of `power`.
        let expected = expected_section_bytes(id, 13);
        if id == 7 {
            assert_eq!(expected, None);
        } else {
            assert_eq!(expected, Some(len), "section {id}");
        }
    }
    assert_eq!(expected_section_elements(2, 13), Some(2 * n - 1));
    assert_eq!(expected_section_elements(12, 13), Some(4 * n - 1));
    assert_eq!(expected_section_elements(13, 13), Some(2 * n - 1));
    assert_eq!(expected_section_elements(1, 13), None);
    assert_eq!(expected_section_elements(7, 13), None);
}

/// Element 0 of sections 2 and 3 is the generator, which is what `powersoftau_verify.js:
/// 184-204` asserts, and the cheapest end-to-end check that the LEM decode is right.
#[test]
fn point_sections_start_at_the_generator() {
    let path = ptau_dir().join("local_13.ptau");
    if !path.is_file() {
        return;
    }
    let file = Ptau::open(&path).expect("local_13 opens");
    assert_eq!(file.g1_points(2, 0, 1).unwrap()[0], G1Affine::generator());
    assert_eq!(file.g2_points(3, 0, 1).unwrap()[0], G2Affine::generator());
    // alphaTauG1[0] is alpha*G1 and betaTauG1[0] is beta*G1, so neither is the generator,
    // but both must still decode to a curve point rather than to garbage.
    assert!(file.g1_points(4, 0, 1).unwrap()[0].is_on_curve());
    assert!(file.g1_points(5, 0, 1).unwrap()[0].is_on_curve());
    assert!(file.g2_points(6, 0, 1).unwrap()[0].is_on_curve());
}

/// Section 12 carries one block more than the others, `p = power+1`, so its top block is
/// `2n` points at element `2n - 1` and the section ends exactly there. That block is the
/// one snarkjs feeds `2n` inputs while section 2 only holds `2n - 1`, padding the last
/// *input* slot with the point at infinity before the transform
/// (`powersoftau_preparephase2.js:78-81`); the block on disk is the output of an inverse
/// group FFT over that input, so nothing in it is infinity.
#[test]
fn section_12_carries_one_block_more_than_the_rest() {
    let path = ptau_dir().join("local_13.ptau");
    if !path.is_file() {
        return;
    }
    let file = Ptau::open(&path).expect("local_13 opens");
    let block = file.lagrange_block_g1(12, 1 << 14).expect("power+1 block");
    assert_eq!(block.len(), 1 << 14);
    assert!(block.iter().all(|p| p.is_on_curve() && !p.is_zero()));
    // The block ends on the section, which is what makes `4n - 1` the element count: one
    // more point is not there.
    assert!(file.g1_points(12, (1 << 14) - 1, (1 << 14) + 1).is_err());
    // The other three stop a block earlier, at `2n - 1`.
    assert_eq!(file.lagrange_block_g1(12, 1 << 13).unwrap().len(), 1 << 13);
    assert_eq!(file.lagrange_block_g2(1 << 13).unwrap().len(), 1 << 13);
    assert_eq!(file.lagrange_block_g1(14, 1 << 13).unwrap().len(), 1 << 13);
    assert_eq!(file.lagrange_block_g1(15, 1 << 13).unwrap().len(), 1 << 13);
}

#[test]
fn reads_past_a_section_are_refused() {
    let path = ptau_dir().join("local_13.ptau");
    if !path.is_file() {
        return;
    }
    let file = Ptau::open(&path).expect("local_13 opens");
    // Section 3 holds exactly 2^13 points.
    assert!(file.g2_points(3, 8191, 1).is_ok());
    assert!(file.g2_points(3, 8192, 1).is_err());
    assert!(file.g1_points(2, 0, usize::MAX).is_err());
    // Section 13's last block is 2^13 long, so 2^14 runs off the end even though the same
    // request is legal against section 12.
    assert!(file.lagrange_block_g2(1 << 14).is_err());
    // A non-power-of-two would straddle two blocks and silently mix two evaluations.
    assert!(file.lagrange_block_g1(12, 3000).is_err());
}

/// The record walk, over 1 and over 62 records. `nextChallenge` is the last field before
/// `type`, so landing on the right bytes means every `paramLength` in the chain was read
/// correctly.
#[test]
fn last_challenge_walks_the_whole_chain() {
    let cases: &[(&str, &str)] = &[
        ("local_13.ptau", "c75298e4f01915a53437731cf72cd7fb0ccbc42f4739abad5f55b521b7fc94c775761e08223f72545fa5c025b168268cea454e89602890e54483c6e7abcc5a36"),
        ("local_19.ptau", "28e10d9b66bc19e1dab0f1195a1949e9a27b280ae2fec4a1ca512a8ca6cf9f3670326ece0b75af24687252372b01ab5b73ed4891ed9bc0d1a365a78759362da0"),
        ("ppot_0080_15.ptau", "3381a48f5d99da6901fc5a56220c168248babbd8306e4e41d3ad2424ec8a436ca43e17bd260305eb32ebb722eea44451e51baa81933af03f263a112f044f5e5a"),
        ("ppot_0080_19.ptau", "3381a48f5d99da6901fc5a56220c168248babbd8306e4e41d3ad2424ec8a436ca43e17bd260305eb32ebb722eea44451e51baa81933af03f263a112f044f5e5a"),
    ];
    for (name, want) in cases {
        let path = ptau_dir().join(name);
        if !path.is_file() {
            continue;
        }
        let file = Ptau::open(&path).expect("opens");
        let got = file.last_challenge().expect("chain walks");
        assert_eq!(hex(&got), *want, "{name}");
    }
}

/// Names and beacon flags out of section 7, without decoding a pubkey.
#[test]
fn contribution_summaries_carry_names_and_kinds() {
    let path = ptau_dir().join("local_13.ptau");
    if path.is_file() {
        let info = Ptau::open(&path).expect("opens").info();
        assert_eq!(info.summaries.len(), 1);
        assert_eq!(info.summaries[0].index, 1);
        assert_eq!(info.summaries[0].name.as_deref(), Some("quick"));
        assert_eq!(
            info.summaries[0].kind,
            g16_ceremony::ContributionKind::Contribute
        );
        assert_eq!(info.summaries[0].param_bytes, 7);
    }

    let path = ptau_dir().join("ppot_0080_15.ptau");
    if path.is_file() {
        let info = Ptau::open(&path).expect("opens").info();
        assert_eq!(info.summaries.len(), 62);
        // The imported ppot responses carry no params at all, which is why `paramLength`
        // is zero for 58 of the 62.
        assert!(info.summaries[..58]
            .iter()
            .all(|c| c.param_bytes == 0 && c.name.is_none()));
        let last = info.summaries.last().unwrap();
        assert_eq!(last.index, 62);
        assert_eq!(last.kind, g16_ceremony::ContributionKind::Beacon);
        assert_eq!(last.param_bytes, 100);
        assert_eq!(last.name, None);
    }
}

/// A prefix of a real file, which is what an interrupted download leaves behind. Strict
/// open has to refuse it and lenient open has to describe it, down to the byte total it
/// would have had.
#[test]
fn a_truncated_download_is_reported_not_rejected() {
    let path = ptau_dir().join("local_13.ptau");
    if !path.is_file() {
        return;
    }
    let full = std::fs::read(&path).expect("read local_13");
    let cut = 5_000_000usize;
    let dir = std::env::temp_dir().join("g16-ptau-truncation");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let short = dir.join("local_13_short.ptau");
    std::fs::write(&short, &full[..cut]).expect("write prefix");

    assert!(
        Ptau::open(&short).is_err(),
        "a strict open must refuse a short file"
    );
    let file = Ptau::open_lenient(&short).expect("a lenient open must describe it");
    let info = file.info();

    assert_eq!(info.power, 13);
    assert_eq!(info.file_bytes, cut as u64);
    assert!(!info.is_complete());
    // 12 bytes of preamble, then sections 1 to 7 whole, puts section 12's payload at
    // 3_147_459; everything after the cut is missing.
    let t = info.truncation.expect("truncation is reported");
    assert_eq!(t.section, Some(12));
    assert_eq!(t.declared, (4 * (1u64 << 13) - 1) * 64);
    assert_eq!(t.present, cut as u64 - 3_147_459);
    // Sections 1 to 7 arrived, 12 is short, 13 to 15 were never reached.
    assert_eq!(info.sections.len(), 8);
    assert_eq!(info.declared_sections, 11);
    assert!(info.sections[..7].iter().all(|s| s.is_complete()));
    assert!(!info.sections[7].is_complete());
    assert!(
        !info.sections[7].is_corrupt(),
        "short is not the same as wrong"
    );
    // 13, 14 and 15 are absent, so the file is not prepared however much of 12 arrived.
    assert!(!info.prepared);
    // The report still names the contributor, and still knows how big the file should be.
    assert_eq!(info.n_contributions, 1);
    assert_eq!(info.summaries[0].name.as_deref(), Some("quick"));
    assert_eq!(info.expected_bytes, Some(full.len() as u64));

    std::fs::remove_file(&short).ok();
}

/// `ptau pick`: smallest sufficient prepared file, by `zkey_new.js:62-69`'s rule and
/// nothing else.
#[test]
fn pick_names_the_smallest_sufficient_file() {
    let dir = ptau_dir();
    if !dir.is_dir() {
        return;
    }
    let candidates = scan_dir(&dir).expect("scan");
    assert!(!candidates.is_empty());
    assert!(
        candidates.windows(2).all(|w| w[0].power <= w[1].power),
        "sorted by power"
    );
    assert!(candidates.iter().all(|c| c.prepared && c.complete));

    let smallest = candidates[0].power;
    let largest = candidates.last().unwrap().power;
    for want in smallest..=largest {
        let choice = pick(&candidates, want).unwrap_or_else(|| panic!("nothing covers 2^{want}"));
        assert!(want <= choice.power);
        assert!(choice.prepared);
        // Nothing smaller would also have done.
        assert!(!candidates
            .iter()
            .any(|c| c.prepared && want <= c.power && c.power < choice.power));
    }
    // A circuit past the biggest file has no answer, and that is not an error.
    assert!(pick(&candidates, largest + 1).is_none());
    assert_eq!(
        pick_in_dir(&dir, smallest).expect("pick").map(|c| c.power),
        Some(smallest)
    );
}

/// `g2_sp` is not stored (`powersoftau_utils.js:81-99` reads six G1 and three G2, and none
/// of them is it), so `contributions` rebuilds it from the challenge each record was made
/// against. `e(g1_s, g2_spx) == e(g1_sx, g2_sp)` holds only when that challenge is right,
/// which makes it a direct test of the seed.
///
/// Record 0 is the whole point. Every later record seeds from the stored `next_challenge`,
/// so a wrong seed shows up once per file and only where `power != ceremonyPower`: seeding
/// from `power` leaves all three of record 0's keys failing this on the ppot files and
/// passing on the local ones.
#[test]
fn recomputed_g2_sp_matches_the_stored_pubkey() {
    let files = present();
    assert!(!files.is_empty(), "no .ptau in {}", ptau_dir().display());
    let mut saw_truncated = false;
    for (want, path) in files {
        // 62 pubkeys is 186 pairings, so one truncated file and one untruncated is the
        // whole matrix; the rest are the same 62 records at another power.
        if want.name != "ppot_0080_15.ptau" && want.name != "local_13.ptau" {
            continue;
        }
        saw_truncated |= want.power != want.ceremony_power;
        let file = Ptau::open(&path).unwrap_or_else(|e| panic!("{}: {e}", want.name));
        let contributions = file.contributions().expect("section 7 decodes");
        assert_eq!(contributions.len(), want.n_contributions, "{}", want.name);
        for (i, c) in contributions.iter().enumerate() {
            for (which, k) in [
                ("tau", &c.pubkeys.tau),
                ("alpha", &c.pubkeys.alpha),
                ("beta", &c.pubkeys.beta),
            ] {
                assert!(
                    same_ratio(&k.g1_s, &k.g1_sx, &k.g2_sp, &k.g2_spx),
                    "{}: record {i} {which} g2_sp was derived off the wrong challenge",
                    want.name
                );
            }
        }
    }
    assert!(
        saw_truncated,
        "the truncated case is the one that regressed"
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
