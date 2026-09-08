//! `prepare phase2` against snarkjs 0.7.6's own bytes.
//!
//! The command takes no entropy and no clock: its output is a pure function of the input
//! file. So every assertion here is a byte comparison, and the only question is how to get
//! an input both implementations agree on when `.ptau` is gitignored repo-wide and nothing
//! can be checked in.
//!
//! Two answers, and they cover different halves:
//!
//! 1. **A chain this crate builds.** `ptau new` followed by `ptau beacon` is deterministic
//!    end to end, and [`phase1`] already pins those files against snarkjs by SHA-256 at
//!    powers 8 and 10, so the input is snarkjs' file by construction rather than by
//!    assumption. What is pinned below is the digest of `powersoftau prepare phase2` run
//!    over it. A beacon input matters: a *fresh* accumulator is the generator repeated, so
//!    its Lagrange evaluations are `G` followed by 2^p - 1 points at infinity, and a
//!    transform with every twiddle wrong would still produce them.
//! 2. **`bench/ptau/local_13.ptau`**, a real prepared file from a real contribution.
//!    Stripping it back to its seven raw sections and re-preparing has to reproduce the
//!    whole file, framing included, at a power none of the pinned cases reach.
//!
//! The unit tests in `src/prepare.rs` cover the one thing neither oracle can reach: the
//! two-coset split at `bits == Fr::TWO_ADICITY + 1`, which only a power-28 file triggers.

use std::path::{Path, PathBuf};
use std::time::Instant;

use g16_ceremony::prepare::{
    group_ifft_g1, lagrange_evaluations_g1, prepare_phase2, prepared_element_count,
};
use g16_ceremony::ptau::{Ptau, LAGRANGE_SECTIONS, PTAU_MAGIC, S_CONTRIBUTIONS};
use g16_ceremony::write::BinFileWriter;
use g16_ceremony::{phase1, ptau, CeremonyError, CpuGroupFft, CpuKeyScale};
use g16_field::{CurveGroup, FftField, Fr, G1Projective, One, PrimeField, PrimeGroup};
use sha2::{Digest as _, Sha256};

/// The beacon `tests/phase1.rs` pins its files against, `01 02 .. 20` at 2^12 iterations
/// under the name "final beacon". Reused verbatim so the input to every case below is a
/// file that test has already compared to snarkjs byte for byte.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
const BEACON_ITERATIONS: u8 = 12;
const BEACON_NAME: &str = "final beacon";

/// power, SHA-256 of the beacon input, SHA-256 of `powersoftau prepare phase2` over it.
///
/// The input digests at 8 and 10 are the ones `tests/phase1.rs` already asserts against
/// snarkjs; the third is this crate's own and is here so a phase-1 change that moves the
/// input announces itself as an input mismatch rather than as a prepare mismatch.
const CASES: [(u32, &str, &str); 3] = [
    (
        8,
        "6b828df149840d77a11ff318aa3634a5ed8840e5eaac9b08af7d1df079f4e13c",
        "37f6a4d59570d9a8de687f1f1aa2ec90dfce589df1a366a2e4dde57970a5c301",
    ),
    (
        10,
        "3061f935928c520a54092580068867f70360f24ee9bdbda917e7546d9a48ef86",
        "6593600af97edaeb4b676fc16d9deb89329d13fb30091e47adeb08fb68050cd3",
    ),
    (
        12,
        "2862d2582084c27a5a74784dcdec25a5f639016897bccd8d0d2be47b86aaba33",
        "f4dfc09e35bb9228bd31b2ac859d61c60bf4af862e53a02474e6bef946210cf0",
    ),
];

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "g16-ceremony-prepare-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest_of(path: &Path) -> String {
    hex(&Sha256::digest(std::fs::read(path).unwrap()))
}

fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

fn bench_ptau(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/ptau")
        .join(name);
    p.is_file().then(|| p.canonicalize().unwrap())
}

/// Byte comparison that names the offset rather than dumping two multi-megabyte vectors.
fn assert_same_bytes(label: &str, got: &[u8], want: &[u8]) {
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{label}: first difference at byte {i}, 0x{:02x} vs 0x{:02x}",
            got[i], want[i]
        );
    }
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {} bytes against {}",
        got.len(),
        want.len()
    );
}

/// The three pinned cases, from `ptau new` through `ptau beacon` to `ptau prepare`.
///
/// Powers 8, 10 and 12 span the shapes that matter: at 8 the largest transform is 2^9
/// points and every block fits one rayon task, and by 12 the section-12 `power+1` block is
/// a 2^13-point transform running fourteen passes. `snarkjs powersoftau verify` accepts
/// all three outputs, which is the check that reads sections 12 to 15 back and pairs them
/// against section 2.
#[test]
fn the_pinned_chain_reproduces_snarkjs_byte_for_byte() {
    let dir = tmp_dir("pinned");
    for (power, input_digest, prepared_digest) in CASES {
        let fresh = dir.join(format!("new_{power}.ptau"));
        let input = dir.join(format!("in_{power}.ptau"));
        let out = dir.join(format!("prepared_{power}.ptau"));

        phase1::ptau_new(power, &fresh).unwrap();
        phase1::beacon(
            &fresh,
            &input,
            Some(BEACON_NAME),
            &beacon_bytes(),
            BEACON_ITERATIONS,
            &CpuKeyScale,
        )
        .unwrap();
        assert_eq!(
            digest_of(&input),
            input_digest,
            "power {power}: the phase-1 input moved, so this case is judging the wrong file"
        );

        let started = Instant::now();
        prepare_phase2(&input, &out, &CpuGroupFft).unwrap();
        eprintln!("power {power} prepare: {:.2?}", started.elapsed());

        assert_eq!(
            digest_of(&out),
            prepared_digest,
            "power {power}: prepared file differs from snarkjs'"
        );

        // The header is the one field the command does not merely copy, and it is
        // deliberately lossy: `ceremonyPower` is re-derived from `power`.
        let prepared = Ptau::open(&out).unwrap();
        assert!(prepared.is_prepared());
        assert_eq!(prepared.header().power, power);
        assert_eq!(prepared.header().ceremony_power, power);
        assert_eq!(
            prepared.contributions().unwrap().len(),
            1,
            "power {power}: section 7 must survive the copy"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A real prepared file, taken apart and rebuilt.
///
/// `local_13.ptau` came out of `snarkjs powersoftau contribute` and then `prepare phase2`
/// on someone's machine, so unlike the pinned cases nothing here was produced by this
/// crate at all. It is also the only case where the section-12 `power+1` block is a
/// 2^14-point transform.
#[test]
fn local_13_prepare_reproduces_the_snarkjs_file_byte_for_byte() {
    let Some(prepared) = bench_ptau("local_13.ptau") else {
        eprintln!("skipping: bench/ptau/local_13.ptau is absent");
        return;
    };
    let dir = tmp_dir("local13");
    let raw = dir.join("raw.ptau");
    let mine = dir.join("prepared.ptau");

    let src = Ptau::open(&prepared).unwrap();
    assert!(src.is_prepared(), "local_13.ptau is the prepared oracle");
    assert_eq!(src.header().power, 13);

    // Back to the seven sections `powersoftau contribute` would have left behind.
    let mut w = BinFileWriter::create(&raw, PTAU_MAGIC, 1, 7).unwrap();
    w.write_section_verbatim(ptau::S_HEADER, src.section(ptau::S_HEADER).unwrap())
        .unwrap();
    for id in ptau::POINT_SECTIONS.iter().chain(&[S_CONTRIBUTIONS]) {
        w.write_section_verbatim(*id, src.section(*id).unwrap())
            .unwrap();
    }
    w.finish().unwrap();

    let started = Instant::now();
    prepare_phase2(&raw, &mine, &CpuGroupFft).unwrap();
    eprintln!("power 13 prepare: {:.2?}", started.elapsed());

    assert_same_bytes(
        "local_13.ptau",
        &std::fs::read(&mine).unwrap(),
        &std::fs::read(&prepared).unwrap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every prepared file the repo ships, on section lengths alone. No transform runs, so
/// this reaches powers 15 to 21 that nothing else here can afford, and a wrong block count
/// at any of them shows up as a byte total that misses.
#[test]
fn prepared_element_counts_match_the_shipped_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/ptau");
    let Ok(entries) = std::fs::read_dir(&root) else {
        eprintln!("skipping: bench/ptau is absent");
        return;
    };
    let mut seen = 0;
    for path in entries.flatten().map(|e| e.path()) {
        if path.extension().is_none_or(|e| e != "ptau") {
            continue;
        }
        let Ok(file) = Ptau::open(&path) else {
            continue;
        };
        if !file.is_prepared() {
            continue;
        }
        let power = file.header().power;
        for id in LAGRANGE_SECTIONS {
            let stride = if id == ptau::S_LAGRANGE_TAU_G2 {
                g16_ceremony::SG2
            } else {
                g16_ceremony::SG1
            };
            let want = prepared_element_count(id, power).unwrap();
            assert_eq!(
                file.section(id).unwrap().len(),
                want * stride,
                "{path:?} section {id}"
            );
            seen += 1;
        }
        // Only the four Lagrange sections have a prepared count; section 2 has an element
        // count and it is not one of these.
        assert!(prepared_element_count(ptau::S_TAU_G1, power).is_none());
    }
    assert!(seen > 0, "no prepared ptau in bench/ptau");
}

/// arkworks' two-adic root is ffjavascript's `w[maxBits]`, the value the whole `ROOTs`
/// table is squared down from (`build_fft.js:44-52`). Both are `nqr^rem` for the smallest
/// non-residue, so they agree by definition rather than by luck, and this pins it.
#[test]
fn the_two_adic_root_is_ffjavascripts() {
    assert_eq!(Fr::TWO_ADICITY, 28);
    assert_eq!(
        Fr::TWO_ADIC_ROOT_OF_UNITY.into_bigint().to_string(),
        "19103219067921713944291392827692070036145651957329286315305642004821462161904"
    );
}

/// `group_ifft_g1` is the same transform as `lagrange_evaluations_g1` with a projective
/// entry and exit spliced on, and those two conversions are the only place in the module
/// where a coordinate system can be misread.
#[test]
fn the_projective_entry_point_agrees_with_the_affine_one() {
    let tau = Fr::from(4_242_424_242u64);
    let mut acc = Fr::one();
    let affine: Vec<_> = (0..64)
        .map(|_| {
            let p = (G1Projective::generator() * acc).into_affine();
            acc *= tau;
            p
        })
        .collect();

    let mut projective: Vec<G1Projective> = affine.iter().map(|p| (*p).into()).collect();
    group_ifft_g1(&mut projective, &CpuGroupFft).unwrap();

    let want = lagrange_evaluations_g1(&affine, &CpuGroupFft).unwrap();
    for (i, (got, want)) in projective.iter().zip(&want).enumerate() {
        assert_eq!(got.into_affine(), *want, "element {i}");
    }
}

/// A block whose length is not a power of two has no domain, and snarkjs' own check is a
/// byte-length comparison that would let a caller past it into a wrong answer
/// (`engine_fft.js:493-496`).
#[test]
fn a_non_power_of_two_block_is_rejected() {
    let p = G1Projective::generator().into_affine();
    assert!(matches!(
        lagrange_evaluations_g1(&[p, p, p], &CpuGroupFft),
        Err(CeremonyError::BadParams(_))
    ));
}
