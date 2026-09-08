//! Phase 2: `zkey contribute`, `zkey beacon` and `zkey verify`.
//!
//! A contribution is cheap by construction. Sections 3 to 7 are copied **verbatim**
//! (`zkey_contribute.js:76-88`) and only sections 8 and 9 are rescaled, by a single
//! `invDelta` applied uniformly (`mpc_applykey.js:29-51` starts at `first = invDelta` with
//! `inc = 1`, so the `Fr.exp(inc, n)` in its loop is a no-op). The header's `vk_delta_1`
//! and `vk_delta_2` are multiplied by `prvKey`, not divided.
//!
//! `zkey verify` then requires those five sections to be byte-for-byte equal between the
//! init file and the final one (`zkey_verify_frominit.js:169-197`), which is why
//! [`crate::write::BinFileWriter::write_section_verbatim`] exists and why re-encoding a
//! point on the copy path fails a check whose message says nothing about encoding.
//!
//! What `verify` does **not** do is worth stating plainly, because the command name
//! oversells it: it never checks section 4 against the r1cs, and never checks that
//! sections 3 and 5 to 7 are the correct MSMs of any circuit. Those properties come only
//! from regenerating the init file, which is exactly what the r1cs form does
//! (`zkey_verify_fromr1cs.js:26-29` runs a whole `newZKey` first). So
//! [`verify_from_init`] verifies the contribution chain and that the final key is a
//! consistent rescaling of the given init key. It does not verify the circuit.

use std::path::Path;

use g16_field::{G1Affine, G2Affine};
use g16_msm::MsmBackend;

use crate::transcript::{Digest, Transcript};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionKind, ContributionParams};

/// One record of zkey section 10.
///
/// `g2_sp` is not stored: verification recomputes it as `hashToG2(transcript)`
/// (`zkey_verify_frominit.js:61`), and that recomputation is what binds the published key
/// to the transcript rather than to the contributor's word.
#[derive(Clone, Debug)]
pub struct ZkeyContribution {
    pub delta_after: G1Affine,
    pub g1_s: G1Affine,
    pub g1_sx: G1Affine,
    pub g2_spx: G2Affine,
    pub transcript: Digest,
    pub kind: ContributionKind,
    pub params: ContributionParams,
}

impl ZkeyContribution {
    /// Fixed bytes before the params: `3*sG1 + sG2 + 64 + 4 + 4`.
    pub const PREFIX_BYTES: usize = 3 * crate::SG1 + crate::SG2 + 72;
}

/// zkey section 10.
///
/// A freshly generated zkey has exactly 68 bytes here: the 64-byte `csHash` then a u32
/// zero (`zkey_new.js:168-171`).
#[derive(Clone, Debug)]
pub struct MpcParams {
    pub cs_hash: Digest,
    pub contributions: Vec<ZkeyContribution>,
}

impl MpcParams {
    pub fn read(section: &[u8]) -> Result<Self, CeremonyError> {
        let _ = section;
        todo!("MpcParams::read")
    }

    /// Write section 10 into an already-started section.
    pub fn write(&self, w: &mut BinFileWriter) -> Result<(), CeremonyError> {
        let _ = w;
        todo!("MpcParams::write")
    }
}

/// `hashPubKey` (`zkey_utils.js:558-564`): `delta_after`, `g1_s` and `g1_sx` uncompressed
/// G1, then `g2_spx` uncompressed G2, then the raw 64-byte transcript. 320 bytes, and the
/// unit the accumulated transcript is built from.
pub fn hash_pubkey(t: &mut Transcript, c: &ZkeyContribution) {
    let _ = (t, c);
    todo!("hash_pubkey")
}

/// This contribution's own hash, `blake2b512(hashPubKey(c))`, the value snarkjs prints for
/// a contributor to publish (`zkey_contribute.js:99-102`).
pub fn contribution_hash(c: &ZkeyContribution) -> Digest {
    let _ = c;
    todo!("contribution_hash")
}

/// The transcript a new contribution starts from: `csHash`, then every prior
/// contribution's 320-byte pubkey block, in file order.
pub fn accumulated_transcript(params: &MpcParams) -> Transcript {
    let _ = params;
    todo!("accumulated_transcript")
}

/// What a contribution produced, for the CLI to print.
#[derive(Clone, Debug)]
pub struct ContributionReport {
    pub index: usize,
    pub kind: ContributionKind,
    /// `blake2b512(hashPubKey(c))` for the contribution just made.
    pub hash: Digest,
    pub delta_after: G1Affine,
}

/// `zkey contribute`: draw a delta from the contributor's entropy, rescale sections 8 and
/// 9 by its inverse, and append a record.
///
/// `entropy` is mixed with 64 OS-random bytes, so this is non-deterministic by design and
/// two runs with the same string produce different keys. That is the point of the command.
pub fn contribute(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    entropy: &str,
    msm: &dyn MsmBackend,
) -> Result<ContributionReport, CeremonyError> {
    let _ = (zkey_in, zkey_out, name, entropy, msm);
    todo!("contribute")
}

/// `zkey beacon`: the same contribution with a reproducible key, derived by iterating
/// SHA-256 `2^num_iterations_exp` times over `beacon_hash`.
///
/// `beacon_hash` is raw bytes, not hex: the CLI parses the hex and enforces the bounds
/// snarkjs does (`10 <= num_iterations_exp <= 63`, and a hash of 1 to 255 bytes).
pub fn beacon(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    beacon_hash: &[u8],
    num_iterations_exp: u8,
    msm: &dyn MsmBackend,
) -> Result<ContributionReport, CeremonyError> {
    let _ = (
        zkey_in,
        zkey_out,
        name,
        beacon_hash,
        num_iterations_exp,
        msm,
    );
    todo!("beacon")
}

/// What `zkey verify` checked, in the order it checked it.
#[derive(Clone, Debug)]
pub struct ZkeyVerifyReport {
    /// One hash per contribution, oldest first. snarkjs prints them reversed.
    pub contribution_hashes: Vec<Digest>,
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
}

/// `zkey verify init` (`zkey_verify_frominit.js:32-231`), in six parts: the contribution
/// chain, init-vs-final header agreement, the shared `csHash` and section sizes, byte
/// identity of sections 3 to 7, and a same-ratio check on each of the two rescaled
/// sections.
///
/// The ptau is needed for the last of those: the H check MSMs the ptau differences
/// `tauG1[i + domainSize] - tauG1[i]` against a random scalar vector whose last entry is
/// forced to zero, because the quotient polynomial has degree `domainSize - 2`.
pub fn verify_from_init(
    init_zkey: &Path,
    ptau: &Path,
    final_zkey: &Path,
    msm: &dyn MsmBackend,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    let _ = (init_zkey, ptau, final_zkey, msm);
    todo!("verify_from_init")
}

/// `zkey verify r1cs`: [`crate::setup::setup`] into a temporary init key, then
/// [`verify_from_init`] against it. This is the only form that checks the key against a
/// circuit, and it costs a full setup, every MSM included.
pub fn verify_from_r1cs(
    r1cs: &Path,
    ptau: &Path,
    final_zkey: &Path,
    msm: &dyn MsmBackend,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    let _ = (r1cs, ptau, final_zkey, msm);
    todo!("verify_from_r1cs")
}
