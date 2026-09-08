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
//!   lands in section 12, whose `power+1` block wants `2n` inputs and pads the last
//!   *input* slot with the point at infinity before the transform runs
//!   (`powersoftau_preparephase2.js:78-81`). The block stored on disk is that transform's
//!   output, so no point in it is infinity.
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
//!
//! # Describing a file that does not fully parse
//!
//! Telling a complete power-19 from a half-finished download is the entire job of
//! `g16 ptau info`, so this module has a second door. [`Ptau::open_lenient`] indexes with
//! [`Scan::Lenient`], which clamps the section whose declared length runs past EOF instead
//! of rejecting the file, and [`Ptau::info`] then reports declared, present and expected
//! side by side. The three are different failures: present below declared is a truncated
//! transfer, declared away from expected is a corrupt or non-BN254 file, and a chain that
//! ends before EOF is trailing garbage. [`PtauInfo::expected_bytes`] reconstructs the full
//! size the file should have had, which the section chain alone cannot say once it has run
//! off the end, by taking `nSections` at its word about the layout: every snarkjs writer
//! emits either [`NEW_LAYOUT`] or [`PREPARED_LAYOUT`], in that order, and only section 7's
//! length is not a function of `power`.

use std::path::{Path, PathBuf};

use g16_field::{G1Affine, G2Affine};
use g16_zkey::binfile::{self, BinFile, Cursor, Scan};
use g16_zkey::ZkeyError;
use rayon::prelude::*;

use crate::transcript::{
    read_ptau_pubkey, write_ptau_pubkey, Digest, PtauPubKeys, Transcript, PARTIAL_HASH_BYTES,
    PTAU_PUBKEY_BYTES,
};
use crate::{check_modulus, q_le, CeremonyError, ContributionKind, ContributionParams, SG1, SG2};

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

/// Section ids in the order `powersoftau new` writes them, which is the layout a file
/// declaring 7 sections has (`powersoftau_new.js:75-132`).
pub const NEW_LAYOUT: [u32; 7] = [
    S_HEADER,
    S_TAU_G1,
    S_TAU_G2,
    S_ALPHA_TAU_G1,
    S_BETA_TAU_G1,
    S_BETA_G2,
    S_CONTRIBUTIONS,
];

/// The layout of a file declaring 11 sections: `prepare phase2`, `convert` and `truncate`
/// all write these ids in this order (`powersoftau_preparephase2.js:29-42`,
/// `powersoftau_truncate.js:46-58`).
pub const PREPARED_LAYOUT: [u32; 11] = [
    S_HEADER,
    S_TAU_G1,
    S_TAU_G2,
    S_ALPHA_TAU_G1,
    S_BETA_TAU_G1,
    S_BETA_G2,
    S_CONTRIBUTIONS,
    S_LAGRANGE_TAU_G1,
    S_LAGRANGE_TAU_G2,
    S_LAGRANGE_ALPHA_TAU_G1,
    S_LAGRANGE_BETA_TAU_G1,
];

/// Fixed bytes of one contribution record before its TLV params, on BN254:
/// `64 + 128 + 64 + 64 + 128 + 768 + 216 + 64 + 4 + 4`.
pub const CONTRIBUTION_PREFIX_BYTES: usize = 1504;

/// Offset of the `type` u32 inside a record, and of the `paramLength` u32 after it. Both
/// are read without decoding a single point, which is what lets [`Ptau::info`] and
/// [`Ptau::last_challenge`] walk 62 records for free.
const KIND_OFFSET: usize = CONTRIBUTION_PREFIX_BYTES - 8;
const PARAM_LEN_OFFSET: usize = CONTRIBUTION_PREFIX_BYTES - 4;

/// Bytes of a BLAKE2b-512 digest, the width of `nextChallenge`.
const DIGEST_BYTES: usize = 64;

/// The 12 bytes of framing every section costs: `u32 id`, `u64 length`
/// (`binfileutils.js:23-32`). The file preamble is the same width by coincidence: 4 magic
/// bytes, `u32 version`, `u32 nSections`.
const SECTION_FRAMING: u64 = 12;
const PREAMBLE_BYTES: u64 = 12;

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
        let head = take(data, CONTRIBUTION_PREFIX_BYTES)?;
        let mut cur = Cursor::new(head, S_CONTRIBUTIONS);
        let tau_g1 = binfile::g1(cur.take(SG1)?);
        let tau_g2 = binfile::g2(cur.take(SG2)?);
        let alpha_g1 = binfile::g1(cur.take(SG1)?);
        let beta_g1 = binfile::g1(cur.take(SG1)?);
        let beta_g2 = binfile::g2(cur.take(SG2)?);
        let pubkeys = read_ptau_pubkey(cur.take(PTAU_PUBKEY_BYTES)?, prev_challenge)?;
        let mut partial_hash = [0u8; PARTIAL_HASH_BYTES];
        partial_hash.copy_from_slice(cur.take(PARTIAL_HASH_BYTES)?);
        let mut next_challenge = [0u8; DIGEST_BYTES];
        next_challenge.copy_from_slice(cur.take(DIGEST_BYTES)?);
        let kind = ContributionKind::from_u32(cur.u32()?)?;
        let param_len = cur.u32()? as usize;
        let params = ContributionParams::decode(take(data, param_len)?)?;
        Ok(Self {
            tau_g1,
            tau_g2,
            alpha_g1,
            beta_g1,
            beta_g2,
            pubkeys,
            partial_hash,
            next_challenge,
            kind,
            params,
        })
    }

    /// The record as written, params included. The pubkey goes out in **Montgomery** LEM
    /// here (`powersoftau_utils.js:253`), even though the same nine points are hashed
    /// non-Montgomery.
    ///
    /// This is not a byte-preserving round trip of a record read from a file: a name is
    /// truncated to 64 UTF-16 code units on the way out (`powersoftau_utils.js:261`), and
    /// an unrecognised param type would have failed the read anyway. Anything copying an
    /// existing chain forward should copy section 7 verbatim rather than re-encode it.
    pub fn encode(&self) -> Vec<u8> {
        // The params were either decoded from a file, which caps every field at the `u8`
        // length it was read through, or built by `phase1`, which rejects an over-long
        // name with `BadParams` before a record exists. An encode failure here is a bug in
        // this crate, not bad input.
        let params = self
            .params
            .encode()
            .expect("contribution params are validated before a record is built");
        let mut out = Vec::with_capacity(CONTRIBUTION_PREFIX_BYTES + params.len());
        out.extend_from_slice(&crate::write::g1_lem(&self.tau_g1));
        out.extend_from_slice(&crate::write::g2_lem(&self.tau_g2));
        out.extend_from_slice(&crate::write::g1_lem(&self.alpha_g1));
        out.extend_from_slice(&crate::write::g1_lem(&self.beta_g1));
        out.extend_from_slice(&crate::write::g2_lem(&self.beta_g2));
        out.extend_from_slice(&write_ptau_pubkey(&self.pubkeys, true));
        out.extend_from_slice(&self.partial_hash);
        out.extend_from_slice(&self.next_challenge);
        out.extend_from_slice(&self.kind.as_u32().to_le_bytes());
        out.extend_from_slice(&(params.len() as u32).to_le_bytes());
        out.extend_from_slice(&params);
        out
    }

    /// Resume from [`PtauContribution::partial_hash`], absorb the 768-byte uncompressed
    /// pubkey blob and digest. This is the "Contribution Response Hash" snarkjs prints and
    /// the value the next challenge hash is seeded with.
    pub fn response_hash(&self) -> Result<Digest, CeremonyError> {
        let mut hasher = Transcript::from_partial_hash(&self.partial_hash)?;
        hasher.update(&write_ptau_pubkey(&self.pubkeys, false));
        Ok(hasher.finalize())
    }
}

/// What one contribution says about itself without decoding a point.
///
/// [`Ptau::info`] builds these rather than [`PtauContribution`]s because naming the 62
/// contributors of a ppot file must not cost 186 `getG2sp` derivations, and because a
/// pubkey that fails to decode is no reason to stop reporting the file.
#[derive(Clone, Debug)]
pub struct ContributionSummary {
    /// One-based, the way snarkjs numbers them (`powersoftau_utils.js:236`).
    pub index: usize,
    pub kind: ContributionKind,
    pub name: Option<String>,
    /// `paramLength`. Zero for the imported ppot responses, which carry no name at all.
    pub param_bytes: usize,
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

    /// The section chain reached this section's end. False for the one section a lenient
    /// scan had to clamp.
    pub fn is_complete(&self) -> bool {
        self.present == self.declared
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
    /// Who contributed, in chain order. Empty for a file whose section 7 is absent or too
    /// damaged to walk, which is not the same as a file with no contributions: that one
    /// has a section 7 holding a single `u32 0`.
    pub summaries: Vec<ContributionSummary>,
    /// Total file bytes, and what the chain says they should be. A chain that ends before
    /// EOF is as much a problem as one that runs past it.
    pub file_bytes: u64,
    pub chain_end: u64,
    /// The size a complete file with this `power`, `nSections` and section 7 would have.
    /// `None` when `nSections` is not one of the two layouts snarkjs writes, or when the
    /// scan died before reaching section 7 and its variable length is therefore unknown.
    pub expected_bytes: Option<u64>,
    /// Where the section chain ran out of file, from the lenient scan.
    pub truncation: Option<g16_zkey::binfile::Truncation>,
}

impl PtauInfo {
    /// COMPLETE: every declared section is present in full, no length disagrees with
    /// `power`, and the chain lands exactly on EOF.
    pub fn is_complete(&self) -> bool {
        self.truncation.is_none()
            && self.sections.len() == self.declared_sections
            && self.chain_end == self.file_bytes
            && !self.sections.iter().any(|s| s.is_corrupt())
    }
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
        Self::from_file(BinFile::open_with(
            path,
            PTAU_MAGIC,
            PTAU_MAX_VERSION,
            Scan::Strict,
        )?)
    }

    /// [`Ptau::open`] that survives a short file, so `g16 ptau info` can say how short.
    /// Every accessor below still bounds-checks, so a lenient handle is safe to read from;
    /// it just runs out of section earlier.
    pub fn open_lenient(path: &Path) -> Result<Self, CeremonyError> {
        Self::from_file(BinFile::open_with(
            path,
            PTAU_MAGIC,
            PTAU_MAX_VERSION,
            Scan::Lenient,
        )?)
    }

    /// [`Ptau::open`] for a file already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, CeremonyError> {
        Self::from_file(BinFile::from_bytes(bytes, PTAU_MAGIC, PTAU_MAX_VERSION)?)
    }

    fn from_file(file: BinFile) -> Result<Self, CeremonyError> {
        let header = read_header(&file)?;
        Ok(Self { file, header })
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
        LAGRANGE_SECTIONS.iter().all(|&id| self.has_section(id))
    }

    fn has_section(&self, id: u32) -> bool {
        self.file.sections().iter().any(|s| s.id == id)
    }

    /// A whole section, by id.
    pub fn section(&self, id: u32) -> Result<&[u8], CeremonyError> {
        match self.file.unique_section(id) {
            Ok(data) => Ok(data),
            // Raised as this crate's own arm rather than passed through, so a caller can
            // tell "no such section" from "the section is there and wrong" without
            // reaching into `ZkeyError`.
            Err(ZkeyError::MissingSection(id)) => Err(CeremonyError::MissingSection(id)),
            Err(e) => Err(e.into()),
        }
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
        let data = self.section(id)?;
        let range = offset
            .checked_mul(stride)
            .zip(offset.checked_add(n).and_then(|e| e.checked_mul(stride)))
            .filter(|&(_, end)| end <= data.len());
        match range {
            Some((start, end)) => Ok(&data[start..end]),
            None => Err(CeremonyError::malformed(
                id,
                format!(
                    "elements {offset}..{} of {stride} bytes run past the {}-byte section",
                    offset.saturating_add(n),
                    data.len()
                ),
            )),
        }
    }

    /// `n` G1 points from element `offset` of section `id`.
    pub fn g1_points(
        &self,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<G1Affine>, CeremonyError> {
        let data = self.section_elements(id, SG1, offset, n)?;
        Ok(data.par_chunks_exact(SG1).map(binfile::g1).collect())
    }

    /// `n` G2 points from element `offset` of section `id`.
    pub fn g2_points(
        &self,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<G2Affine>, CeremonyError> {
        let data = self.section_elements(id, SG2, offset, n)?;
        Ok(data.par_chunks_exact(SG2).map(binfile::g2).collect())
    }

    /// The `domain_size`-point Lagrange block of section `id`, which starts at element
    /// `domain_size - 1`. This is the read `groth16 setup` makes four times, and the
    /// offset is not a magic number: see the module doc.
    pub fn lagrange_block_g1(
        &self,
        id: u32,
        domain_size: usize,
    ) -> Result<Vec<G1Affine>, CeremonyError> {
        self.check_lagrange(id, domain_size)?;
        self.g1_points(id, domain_size - 1, domain_size)
    }

    /// [`Ptau::lagrange_block_g1`] for section 13, the only G2 Lagrange section.
    pub fn lagrange_block_g2(&self, domain_size: usize) -> Result<Vec<G2Affine>, CeremonyError> {
        self.check_lagrange(S_LAGRANGE_TAU_G2, domain_size)?;
        self.g2_points(S_LAGRANGE_TAU_G2, domain_size - 1, domain_size)
    }

    /// The two things that go wrong before the bounds check does: asking a raw file for a
    /// block it never had, and a `domain_size` that is not a power of two, which would
    /// silently read across a block boundary and return points from two different
    /// evaluations.
    fn check_lagrange(&self, id: u32, domain_size: usize) -> Result<(), CeremonyError> {
        if !self.has_section(id) {
            return Err(CeremonyError::NotPrepared);
        }
        if !domain_size.is_power_of_two() {
            return Err(CeremonyError::malformed(
                id,
                format!("domain size {domain_size} is not a power of two"),
            ));
        }
        Ok(())
    }

    /// Section 7, or an empty vector when it is absent. Unlike snarkjs we reject a
    /// duplicate section 7: its guard reads `sections[7][0].length`, which is `undefined`
    /// on the `{p, size}` object, so the check never fires
    /// (`powersoftau_utils.js:228`).
    pub fn contributions(&self) -> Result<Vec<PtauContribution>, CeremonyError> {
        let section = match self.section(S_CONTRIBUTIONS) {
            Ok(data) => data,
            Err(CeremonyError::MissingSection(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut data = section;
        let n = read_u32(&mut data)? as usize;
        // A lying count would otherwise reserve 2^32 records before the first read.
        let mut out = Vec::with_capacity(n.min(data.len() / CONTRIBUTION_PREFIX_BYTES));
        let mut prev = crate::phase1::first_challenge_hash(self.header.power);
        for _ in 0..n {
            let c = PtauContribution::read(&mut data, &prev)?;
            prev = c.next_challenge;
            out.push(c);
        }
        // "Invalid contribution section size" (`powersoftau_utils.js:239`). The records
        // are the only thing in section 7, so landing short of the end means a length
        // field lied and every earlier record is suspect too.
        if !data.is_empty() {
            return Err(CeremonyError::malformed(
                S_CONTRIBUTIONS,
                format!("{} bytes left after {n} contributions", data.len()),
            ));
        }
        Ok(out)
    }

    /// The challenge hash the next contribution must build on: the last contribution's
    /// `next_challenge`, or [`crate::phase1::first_challenge_hash`] when there are none
    /// (`powersoftau_contribute.js:54-58`).
    ///
    /// Walks the records by length instead of decoding them. Every pubkey in between would
    /// otherwise cost three `getG2sp` derivations, and none of them are part of the answer.
    pub fn last_challenge(&self) -> Result<Digest, CeremonyError> {
        let section = match self.section(S_CONTRIBUTIONS) {
            Ok(data) => data,
            Err(CeremonyError::MissingSection(_)) => {
                return Ok(crate::phase1::first_challenge_hash(self.header.power))
            }
            Err(e) => return Err(e),
        };
        let mut data = section;
        let n = read_u32(&mut data)? as usize;
        let mut last = None;
        for _ in 0..n {
            let head = take(&mut data, CONTRIBUTION_PREFIX_BYTES)?;
            let mut digest = [0u8; DIGEST_BYTES];
            digest.copy_from_slice(&head[KIND_OFFSET - DIGEST_BYTES..KIND_OFFSET]);
            last = Some(digest);
            take(&mut data, param_len(head))?;
        }
        if !data.is_empty() {
            return Err(CeremonyError::malformed(
                S_CONTRIBUTIONS,
                format!("{} bytes left after {n} contributions", data.len()),
            ));
        }
        match last {
            Some(d) => Ok(d),
            None => Ok(crate::phase1::first_challenge_hash(self.header.power)),
        }
    }

    /// Everything `g16 ptau info` prints, including per-section "declared vs expected" and
    /// "present vs declared". Never fails: a file too broken to describe is described as
    /// broken.
    pub fn info(&self) -> PtauInfo {
        let truncation = self.file.truncation();
        let entries = self.file.sections();
        let mut sections = Vec::with_capacity(entries.len());
        for (i, s) in entries.iter().enumerate() {
            // A lenient scan clamps `len` to the bytes it found and keeps the real length
            // in `truncation`, and it only ever does that to the last entry it pushed.
            let clamped =
                i + 1 == entries.len() && matches!(truncation, Some(t) if t.section == Some(s.id));
            sections.push(SectionReport {
                id: s.id,
                declared: match truncation {
                    Some(t) if clamped => t.declared,
                    _ => s.len as u64,
                },
                present: s.len as u64,
                expected: expected_section_bytes(s.id, self.header.power),
            });
        }
        let summaries = self
            .section(S_CONTRIBUTIONS)
            .map(contribution_summaries)
            .unwrap_or_default();
        PtauInfo {
            power: self.header.power,
            ceremony_power: self.header.ceremony_power,
            prepared: self.is_prepared(),
            declared_sections: self.file.declared_sections(),
            n_contributions: summaries.len(),
            summaries,
            file_bytes: self.file.total_len() as u64,
            chain_end: entries
                .last()
                .map(|s| (s.start + s.len) as u64)
                .unwrap_or(PREAMBLE_BYTES),
            expected_bytes: self.expected_bytes(&sections),
            sections,
            truncation,
        }
    }

    /// The size this file would have if nothing were missing. Section 7 is the only
    /// length that is not a function of `power`, so it is taken from the chain; the ids
    /// past the truncation point come from `nSections`, which every snarkjs writer sets to
    /// one of the two fixed layouts.
    fn expected_bytes(&self, reports: &[SectionReport]) -> Option<u64> {
        let layout: &[u32] = match self.file.declared_sections() {
            7 => &NEW_LAYOUT,
            11 => &PREPARED_LAYOUT,
            _ => return None,
        };
        let mut total = PREAMBLE_BYTES;
        for &id in layout {
            let len = match id {
                S_CONTRIBUTIONS => reports.iter().find(|r| r.id == id)?.declared,
                _ => expected_section_bytes(id, self.header.power)?,
            };
            total = total.checked_add(SECTION_FRAMING)?.checked_add(len)?;
        }
        Some(total)
    }
}

/// Section 1, with the same "consumed bytes must equal the declared size" assertion
/// `readPTauHeader` makes (`powersoftau_utils.js:66-68`).
fn read_header(file: &BinFile) -> Result<PtauHeader, CeremonyError> {
    let section = match file.unique_section(S_HEADER) {
        Ok(data) => data,
        Err(ZkeyError::MissingSection(id)) => return Err(CeremonyError::MissingSection(id)),
        Err(e) => return Err(e.into()),
    };
    let mut cur = Cursor::new(section, S_HEADER);
    check_modulus(&mut cur, &q_le())?;
    let power = cur.u32()?;
    let ceremony_power = cur.u32()?;
    if cur.remaining() != 0 {
        return Err(CeremonyError::malformed(
            S_HEADER,
            format!("{} bytes left after the header", cur.remaining()),
        ));
    }
    Ok(PtauHeader {
        power,
        ceremony_power,
    })
}

/// Split `n` bytes off the front, the way every record field is read.
fn take<'a>(data: &mut &'a [u8], n: usize) -> Result<&'a [u8], CeremonyError> {
    if data.len() < n {
        return Err(CeremonyError::malformed(
            S_CONTRIBUTIONS,
            format!("wanted {n} bytes, only {} left", data.len()),
        ));
    }
    let (head, rest) = data.split_at(n);
    *data = rest;
    Ok(head)
}

fn read_u32(data: &mut &[u8]) -> Result<u32, CeremonyError> {
    let b = take(data, 4)?;
    Ok(u32::from_le_bytes(b.try_into().expect("slice is 4 bytes")))
}

/// `paramLength` out of a record prefix already known to be [`CONTRIBUTION_PREFIX_BYTES`]
/// long.
fn param_len(head: &[u8]) -> usize {
    u32::from_le_bytes(
        head[PARAM_LEN_OFFSET..CONTRIBUTION_PREFIX_BYTES]
            .try_into()
            .expect("slice is 4 bytes"),
    ) as usize
}

/// Walk section 7 for the report, stopping at the first record that does not parse rather
/// than discarding the ones before it. A file whose chain is cut mid-record still knows
/// who the earlier contributors were, and that is exactly what someone running
/// `g16 ptau info` on a half-finished download wants to see.
fn contribution_summaries(section: &[u8]) -> Vec<ContributionSummary> {
    let mut out = Vec::new();
    let mut data = section;
    let n = match read_u32(&mut data) {
        Ok(n) => n as usize,
        Err(_) => return out,
    };
    for index in 1..=n {
        let head = match take(&mut data, CONTRIBUTION_PREFIX_BYTES) {
            Ok(head) => head,
            Err(_) => break,
        };
        let kind = u32::from_le_bytes(
            head[KIND_OFFSET..PARAM_LEN_OFFSET]
                .try_into()
                .expect("slice is 4 bytes"),
        );
        let kind = match ContributionKind::from_u32(kind) {
            Ok(kind) => kind,
            Err(_) => break,
        };
        let param_bytes = param_len(head);
        let params = match take(&mut data, param_bytes) {
            Ok(params) => params,
            Err(_) => break,
        };
        out.push(ContributionSummary {
            index,
            kind,
            name: ContributionParams::decode(params).ok().and_then(|p| p.name),
            param_bytes,
        });
    }
    out
}

/// Elements a section holds for a given `power`, or `None` for sections 1 and 7, whose
/// bodies are not point arrays.
///
/// `n = 2^power`. Sections 2 to 6 are the writer loops in `powersoftau_new.js:85,:95,:105,
/// :115,:124`. Sections 12 to 15 are a concatenation of `2^p` points for `p = 0..=power`,
/// so `2n - 1`, with section 12 taking one more block for `p = power+1` and therefore
/// `4n - 1` (`powersoftau_preparephase2.js:56-62`). `powersoftau_truncate.js:55-58` states
/// the same four counts a second way, as the prefix length it copies.
pub fn expected_section_elements(id: u32, power: u32) -> Option<u64> {
    // 2^(power+2) has to fit, and nothing real goes past 2^28 anyway: `Fr`'s two-adicity
    // is what caps a usable ceremony.
    if power > 60 {
        return None;
    }
    let n = 1u64 << power;
    Some(match id {
        S_TAU_G1 => 2 * n - 1,
        S_TAU_G2 | S_ALPHA_TAU_G1 | S_BETA_TAU_G1 => n,
        S_BETA_G2 => 1,
        S_LAGRANGE_TAU_G1 => 4 * n - 1,
        S_LAGRANGE_TAU_G2 | S_LAGRANGE_ALPHA_TAU_G1 | S_LAGRANGE_BETA_TAU_G1 => 2 * n - 1,
        _ => return None,
    })
}

/// Expected byte length of a section for a given `power`, or `None` for section 7, whose
/// length is genuinely variable and self-checking. Section 1 is a fixed 44 bytes on BN254;
/// everything else is [`expected_section_elements`] times the point size.
pub fn expected_section_bytes(id: u32, power: u32) -> Option<u64> {
    if id == S_HEADER {
        return Some(PtauHeader::BYTES as u64);
    }
    let stride = match id {
        S_TAU_G2 | S_BETA_G2 | S_LAGRANGE_TAU_G2 => SG2 as u64,
        _ => SG1 as u64,
    };
    expected_section_elements(id, power)?.checked_mul(stride)
}

/// A `.ptau` on disk, described by the facts `groth16 setup` selects on.
#[derive(Clone, Debug)]
pub struct PtauChoice {
    pub path: PathBuf,
    pub power: u32,
    pub ceremony_power: u32,
    /// Sections 12 to 15 present, which is what `zkey_new.js:66-69` requires.
    pub prepared: bool,
    /// Informational: [`scan_dir`] already drops anything that fails a strict open, and a
    /// transfer cut mid-section is exactly that, so a candidate reaching this struct with
    /// `complete == false` was cut on a section boundary and is missing whole sections.
    pub complete: bool,
}

/// Every `.ptau` in `dir` that opens, sorted by power and then by path so the pick is not
/// a function of directory order.
///
/// A file that does not parse is skipped rather than reported: the directory is a pile of
/// downloads, not a manifest, and a truncated transfer is precisely what a strict open
/// rejects. Use [`Ptau::open_lenient`] on a specific path to find out why one was skipped.
pub fn scan_dir(dir: &Path) -> Result<Vec<PtauChoice>, CeremonyError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ptau") {
            continue;
        }
        let Ok(file) = Ptau::open(&path) else {
            continue;
        };
        let info = file.info();
        out.push(PtauChoice {
            path,
            power: info.power,
            ceremony_power: info.ceremony_power,
            prepared: info.prepared,
            complete: info.is_complete(),
        });
    }
    out.sort_by(|a, b| a.power.cmp(&b.power).then_with(|| a.path.cmp(&b.path)));
    Ok(out)
}

/// The smallest candidate a circuit of this `cir_power` can be set up against.
///
/// The rule is `zkey_new.js:62-69` and nothing more: `cirPower <= ptau.power`, and section
/// 12 present. Smallest wins because the Lagrange block setup reads is `domainSize` points
/// either way, while every other byte of a larger file is one the reader has to seek past.
pub fn pick(candidates: &[PtauChoice], cir_power: u32) -> Option<&PtauChoice> {
    candidates
        .iter()
        .filter(|c| c.prepared && cir_power <= c.power)
        .min_by(|a, b| a.power.cmp(&b.power).then_with(|| a.path.cmp(&b.path)))
}

/// [`scan_dir`] then [`pick`], the whole of `g16 ptau pick`.
pub fn pick_in_dir(dir: &Path, cir_power: u32) -> Result<Option<PtauChoice>, CeremonyError> {
    let candidates = scan_dir(dir)?;
    Ok(pick(&candidates, cir_power).cloned())
}
