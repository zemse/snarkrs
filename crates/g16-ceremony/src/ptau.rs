//! `.ptau`, the phase-1 output, and the only file both `groth16 setup` and
//! `powersoftau contribute` read.
//!
//! Same container as `.zkey`, magic `ptau`, version 1. Sections 2 to 6 are the raw powers,
//! 7 is the contribution chain, and 12 to 15 are the Lagrange evaluations that
//! `powersoftau prepare phase2` adds. Everything is uncompressed affine **LEM**: little
//! endian, single Montgomery, `x` then `y`, with the point at infinity written as all
//! zero bytes and no flag bit anywhere (`wasm_curve.js:264-272`).
//!
//! Two numbers are worth stating because getting either wrong is silent:
//!
//! * **Section 2 holds `2n - 1` points, not `2n`** (`powersoftau_new.js:85`, confirmed
//!   four more times in contribute, beacon, export-challenge and verify). The consequence
//!   lands in section 12, whose `power+1` block wants `2n` inputs and pads the last slot
//!   with the point at infinity (`powersoftau_preparephase2.js:78-81`).
//! * **Sections 12 to 15 are a concatenation of per-power blocks**, `2^p` points for
//!   `p = 0..=power` (and one more block for section 12), so block `2^k` starts at element
//!   `2^k - 1`. That is the whole derivation of the `(domainSize-1)*sG` offset setup seeks
//!   to (`zkey_new.js:145-151`).
//!
//! `power != ceremonyPower` marks a file `powersoftau truncate` produced. Setup works
//! against one; `contribute` and `beacon` refuse it outright
//! (`powersoftau_contribute.js:37-40`), and `verify` skips the final next-challenge
//! comparison for it (`powersoftau_verify.js:247-252`).
//!
//! Prepared or not is decided by the presence of sections 12, 13, 14 **and** 15
//! (`powersoftau_verify.js:268-272`), never by `nSections`: a truncated or converted file
//! also declares 11.

use std::path::Path;

use g16_field::{G1Affine, G2Affine};
use g16_zkey::binfile::BinFile;

use crate::transcript::{Digest, PtauPubKeys, PARTIAL_HASH_BYTES};
use crate::{CeremonyError, ContributionKind, ContributionParams};

/// `.ptau` magic (`powersoftau_new.js:75`).
pub const PTAU_MAGIC: &[u8; 4] = b"ptau";
/// Highest container version this reader accepts. Every writer emits 1.
pub const PTAU_MAX_VERSION: u32 = 1;

/// Section ids, named because the numbers appear in every loop bound in this crate.
pub const S_HEADER: u32 = 1;
pub const S_TAU_G1: u32 = 2;
pub const S_TAU_G2: u32 = 3;
pub const S_ALPHA_TAU_G1: u32 = 4;
pub const S_BETA_TAU_G1: u32 = 5;
pub const S_BETA_G2: u32 = 6;
pub const S_CONTRIBUTIONS: u32 = 7;
pub const S_LAGRANGE_TAU_G1: u32 = 12;
pub const S_LAGRANGE_TAU_G2: u32 = 13;
pub const S_LAGRANGE_ALPHA_TAU_G1: u32 = 14;
pub const S_LAGRANGE_BETA_TAU_G1: u32 = 15;

/// The five point sections, in the order every hash absorbs them
/// (`powersoftau_contribute.js:75-84`). Swapping two of these is undetectable until
/// `snarkjs powersoftau verify` rejects the file.
pub const POINT_SECTIONS: [u32; 5] = [S_TAU_G1, S_TAU_G2, S_ALPHA_TAU_G1, S_BETA_TAU_G1, S_BETA_G2];

/// The four sections `prepare phase2` adds, and the exact set whose presence means
/// "prepared" (`powersoftau_verify.js:268-272`).
pub const LAGRANGE_SECTIONS: [u32; 4] = [
    S_LAGRANGE_TAU_G1,
    S_LAGRANGE_TAU_G2,
    S_LAGRANGE_ALPHA_TAU_G1,
    S_LAGRANGE_BETA_TAU_G1,
];

/// Fixed bytes of one contribution record before its TLV params, on BN254:
/// `64 + 128 + 64 + 64 + 128 + 768 + 216 + 64 + 4 + 4`.
pub const CONTRIBUTION_PREFIX_BYTES: usize = 1504;

/// Section 1. `n8` and `q` are checked against BN254 at open, so they are not kept: the
/// curve is identified by `q` alone and there is no curve-name field anywhere in the
/// format (`powersoftau_utils.js:59-63`).
#[derive(Clone, Copy, Debug)]
pub struct PtauHeader {
    pub power: u32,
    /// The power the ceremony actually ran at. `powersoftau new` passes 0 and
    /// `writePTauHeader` turns that into `power` (`powersoftau_utils.js:30`), so a genuine
    /// ceremony power of 0 cannot be expressed. A value different from `power` means the
    /// file was truncated.
    pub ceremony_power: u32,
}

impl PtauHeader {
    /// Bytes of section 1 on BN254. `readPTauHeader` asserts the consumed bytes equal the
    /// declared size (`powersoftau_utils.js:66-68`), so a 40-byte three-field header is
    /// rejected rather than tolerated.
    pub const BYTES: usize = 12 + crate::N8;

    /// True for a file `powersoftau truncate` produced.
    pub fn is_truncated_ceremony(&self) -> bool {
        self.power != self.ceremony_power
    }
}

/// One record of section 7.
///
/// `tau_g1` is the *second* element of section 2 and `tau_g2` the second of section 3,
/// while `alpha_g1`, `beta_g1` and `beta_g2` are the *first* elements of sections 4, 5 and
/// 6 (`powersoftau_contribute.js:76-84`, asserted the same way at
/// `powersoftau_verify.js:188-234`).
#[derive(Clone, Debug)]
pub struct PtauContribution {
    pub tau_g1: G1Affine,
    pub tau_g2: G2Affine,
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    /// The nine stored points plus the three `g2_sp` recomputed from the previous
    /// challenge, since the file does not carry them.
    pub pubkeys: PtauPubKeys,
    /// A serialised mid-stream BLAKE2b state, 216 bytes. It exists only so the response
    /// hash (which is never stored) can be reconstructed by resuming and re-absorbing the
    /// pubkey (`powersoftau_utils.js:176-181`).
    pub partial_hash: [u8; PARTIAL_HASH_BYTES],
    pub next_challenge: Digest,
    pub kind: ContributionKind,
    pub params: ContributionParams,
}

impl PtauContribution {
    /// Parse one record. `prev_challenge` is the previous contribution's `next_challenge`,
    /// or the first-challenge hash for the first record; it is needed because `g2_sp` is
    /// not stored and has to be recomputed with `getG2sp`.
    pub fn read(data: &mut &[u8], prev_challenge: &Digest) -> Result<Self, CeremonyError> {
        let _ = (data, prev_challenge);
        todo!("PtauContribution::read")
    }

    /// The record as written, params included. The pubkey goes out in **Montgomery** LEM
    /// here (`powersoftau_utils.js:253`), even though the same nine points are hashed
    /// non-Montgomery.
    pub fn encode(&self) -> Vec<u8> {
        todo!("PtauContribution::encode")
    }

    /// Resume from [`PtauContribution::partial_hash`], absorb the 576-byte uncompressed
    /// pubkey blob and digest. This is the "Contribution Response Hash" snarkjs prints and
    /// the value the next challenge hash is seeded with.
    pub fn response_hash(&self) -> Result<Digest, CeremonyError> {
        todo!("PtauContribution::response_hash")
    }
}

/// One row of [`PtauInfo`]: what a section claims against what the header implies.
#[derive(Clone, Copy, Debug)]
pub struct SectionReport {
    pub id: u32,
    /// Length from the section chain.
    pub declared: u64,
    /// Bytes actually present, which differs from `declared` only on a truncated file.
    pub present: u64,
    /// What the `power` in section 1 says this section should be, or `None` for section 7,
    /// whose length is genuinely variable and self-checking.
    pub expected: Option<u64>,
}

impl SectionReport {
    /// A length that is not a whole number of points, or that disagrees with the formula,
    /// is corruption rather than truncation and deserves a different word in the report.
    pub fn is_corrupt(&self) -> bool {
        matches!(self.expected, Some(e) if e != self.declared)
    }
}

/// What `g16 ptau info` prints. Built even for a file that does not fully parse, which is
/// the whole reason [`Ptau::open_lenient`] exists.
#[derive(Clone, Debug)]
pub struct PtauInfo {
    pub power: u32,
    pub ceremony_power: u32,
    pub prepared: bool,
    /// `nSections` as declared, which a truncated file will not have reached.
    pub declared_sections: usize,
    pub sections: Vec<SectionReport>,
    pub n_contributions: usize,
    /// Total file bytes, and what the chain says they should be. A chain that ends before
    /// EOF is as much a problem as one that runs past it.
    pub file_bytes: u64,
    pub chain_end: u64,
    /// Where the section chain ran out of file, from the lenient scan.
    pub truncation: Option<g16_zkey::binfile::Truncation>,
}

/// A mapped `.ptau`, header parsed, everything else left on disk.
pub struct Ptau {
    file: BinFile,
    header: PtauHeader,
}

impl Ptau {
    /// Map, check the magic, version and `q`, and parse section 1. A truncated file is an
    /// error here; [`Ptau::open_lenient`] is the one that reports instead.
    pub fn open(path: &Path) -> Result<Self, CeremonyError> {
        let _ = path;
        todo!("Ptau::open")
    }

    /// [`Ptau::open`] that survives a short file, so `g16 ptau info` can say how short.
    /// Every accessor below still bounds-checks, so a lenient handle is safe to read from;
    /// it just runs out of section earlier.
    pub fn open_lenient(path: &Path) -> Result<Self, CeremonyError> {
        let _ = path;
        todo!("Ptau::open_lenient")
    }

    /// [`Ptau::open`] for a file already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, CeremonyError> {
        let _ = bytes;
        todo!("Ptau::from_bytes")
    }

    pub fn header(&self) -> &PtauHeader {
        &self.header
    }

    pub fn power(&self) -> u32 {
        self.header.power
    }

    /// `2^power`, the number of points in section 3, 4 and 5.
    pub fn domain(&self) -> usize {
        1usize << self.header.power
    }

    pub fn file(&self) -> &BinFile {
        &self.file
    }

    /// Sections 12 to 15 all present. This is snarkjs' own test, and the only correct one.
    pub fn is_prepared(&self) -> bool {
        todo!("Ptau::is_prepared")
    }

    /// A whole section, by id.
    pub fn section(&self, id: u32) -> Result<&[u8], CeremonyError> {
        let _ = id;
        todo!("Ptau::section")
    }

    /// `n` elements of `stride` bytes starting at element `offset` of section `id`. Bounds
    /// checked, matching `readSection` rather than the raw `fd.read` snarkjs uses in
    /// `hashHPointsChunk` (`binfileutils.js:117-135`).
    pub fn section_elements(
        &self,
        id: u32,
        stride: usize,
        offset: usize,
        n: usize,
    ) -> Result<&[u8], CeremonyError> {
        let _ = (id, stride, offset, n);
        todo!("Ptau::section_elements")
    }

    /// `n` G1 points from element `offset` of section `id`.
    pub fn g1_points(
        &self,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<G1Affine>, CeremonyError> {
        let _ = (id, offset, n);
        todo!("Ptau::g1_points")
    }

    /// `n` G2 points from element `offset` of section `id`.
    pub fn g2_points(
        &self,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<G2Affine>, CeremonyError> {
        let _ = (id, offset, n);
        todo!("Ptau::g2_points")
    }

    /// The `domain_size`-point Lagrange block of section `id`, which starts at element
    /// `domain_size - 1`. This is the read `groth16 setup` makes four times, and the
    /// offset is not a magic number: see the module doc.
    pub fn lagrange_block_g1(
        &self,
        id: u32,
        domain_size: usize,
    ) -> Result<Vec<G1Affine>, CeremonyError> {
        let _ = (id, domain_size);
        todo!("Ptau::lagrange_block_g1")
    }

    /// [`Ptau::lagrange_block_g1`] for section 13, the only G2 Lagrange section.
    pub fn lagrange_block_g2(&self, domain_size: usize) -> Result<Vec<G2Affine>, CeremonyError> {
        let _ = domain_size;
        todo!("Ptau::lagrange_block_g2")
    }

    /// Section 7, or an empty vector when it is absent. Unlike snarkjs we reject a
    /// duplicate section 7: its guard reads `sections[7][0].length`, which is `undefined`
    /// on the `{p, size}` object, so the check never fires
    /// (`powersoftau_utils.js:228`).
    pub fn contributions(&self) -> Result<Vec<PtauContribution>, CeremonyError> {
        todo!("Ptau::contributions")
    }

    /// The challenge hash the next contribution must build on: the last contribution's
    /// `next_challenge`, or [`crate::phase1::first_challenge_hash`] when there are none
    /// (`powersoftau_contribute.js:54-58`).
    pub fn last_challenge(&self) -> Result<Digest, CeremonyError> {
        todo!("Ptau::last_challenge")
    }

    /// Everything `g16 ptau info` prints, including per-section "declared vs expected" and
    /// "present vs declared". Never fails: a file too broken to describe is described as
    /// broken.
    pub fn info(&self) -> PtauInfo {
        todo!("Ptau::info")
    }
}

/// Expected byte length of a point section for a given `power`, or `None` for section 7.
/// The table is §2 of the format spec: `(2n-1)*sG1` for 2, `n*sG2` for 3, `n*sG1` for 4
/// and 5, `sG2` for 6, `(4n-1)*sG1` for 12, `(2n-1)*sG2` for 13, `(2n-1)*sG1` for 14 and
/// 15.
pub fn expected_section_bytes(id: u32, power: u32) -> Option<u64> {
    let _ = (id, power);
    todo!("expected_section_bytes")
}
