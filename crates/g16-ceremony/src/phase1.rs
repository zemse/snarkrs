//! Phase 1: `powersoftau new`, `contribute`, `beacon` and `verify`.
//!
//! A fresh file is fully determined by `power`. `newAccumulator` writes the generator
//! repeated `2n-1`, `n`, `n`, `n` and once into sections 2 to 6, and a u32 zero into
//! section 7 (`powersoftau_new.js:73-143`). No randomness, no timestamps, and nothing
//! derived from the first challenge hash it returns.
//!
//! Each contribution then hashes the same points **twice, in two different encodings**,
//! and swapping them is undetectable until `snarkjs powersoftau verify` rejects the file:
//!
//! * the **response** hash absorbs the previous challenge and then the new points
//!   **compressed**, in section order 2, 3, 4, 5, 6; the BLAKE2b state is snapshotted at
//!   that point into `partialHash`, and only then is the 576-byte uncompressed pubkey
//!   absorbed (`powersoftau_contribute.js:66-93`);
//! * the **next challenge** hash absorbs that response digest and then the same points
//!   **uncompressed** (`:97-106`).
//!
//! The response hash is never stored. `partialHash` exists precisely so it can be
//! reconstructed on read by resuming and re-absorbing the pubkey.
//!
//! `contribute` and `beacon` both refuse a file where `power != ceremonyPower`
//! (`powersoftau_contribute.js:37-40`), and both warn when section 12 is present, since
//! any contribution invalidates the prepared sections and phase 2 has to be re-prepared.

use std::path::Path;

use crate::transcript::Digest;
use crate::CeremonyError;

/// `nSections` a fresh file declares (`powersoftau_new.js:75`). A prepared, converted or
/// truncated file declares 11, which is why the count is not a test for "prepared".
pub const PTAU_NEW_SECTIONS: u32 = 7;
/// `nSections` `prepare phase2`, `convert` and `truncate` declare.
pub const PTAU_PREPARED_SECTIONS: u32 = 11;

/// snarkjs' bounds on a beacon (`powersoftau_beacon.js:26-42`, `zkey_beacon.js:38-42`).
/// Both phases enforce the same ones.
pub const BEACON_ITERATIONS_MIN: u8 = 10;
pub const BEACON_ITERATIONS_MAX: u8 = 63;
pub const BEACON_HASH_MAX_BYTES: usize = 255;

/// `calculateFirstChallengeHash` (`powersoftau_utils.js:312-358`), the challenge a file
/// with no contributions starts from.
///
/// It absorbs `blake2b512("")` and then the **uncompressed generators repeated**:
/// `2^power * 2 - 1` G1, `2^power` G2, `2^power` G1, `2^power` G1, one G2. Note it hashes
/// the generator repeated rather than the file's contents, which is correct only because a
/// fresh file is exactly that. The 341000-point blocking in the original is a performance
/// detail with no framing implication, since BLAKE2b is a stream.
pub fn first_challenge_hash(power: u32) -> Digest {
    let _ = power;
    todo!("first_challenge_hash")
}

/// `powersoftau new`: write a fresh accumulator at `power` and return its first challenge
/// hash.
pub fn ptau_new(power: u32, out: &Path) -> Result<Digest, CeremonyError> {
    let _ = (power, out);
    todo!("ptau_new")
}

/// What a phase-1 contribution produced.
#[derive(Clone, Debug)]
pub struct Phase1Report {
    pub index: usize,
    pub kind: crate::ContributionKind,
    /// The reconstructed response hash, which snarkjs prints as the "Contribution Response
    /// Hash" for the contributor to publish.
    pub response_hash: Digest,
    /// The value actually stored in the record.
    pub next_challenge: Digest,
}

/// `powersoftau contribute`: draw tau, alpha and beta from the contributor's entropy,
/// rescale every point section, and append a record.
///
/// The RNG draw order is load-bearing and is `Fr, Fr, Fr, G1, G1, G1`: the three private
/// scalars in tau/alpha/beta order, then one `g1_s` per key in the same order
/// (`keypair.js:61-74`).
pub fn contribute(
    ptau_in: &Path,
    ptau_out: &Path,
    name: Option<&str>,
    entropy: &str,
) -> Result<Phase1Report, CeremonyError> {
    let _ = (ptau_in, ptau_out, name, entropy);
    todo!("phase1::contribute")
}

/// `powersoftau beacon`: the same contribution with a key derived from a public beacon,
/// and the only part of a ceremony anyone can reproduce.
pub fn beacon(
    ptau_in: &Path,
    ptau_out: &Path,
    name: Option<&str>,
    beacon_hash: &[u8],
    num_iterations_exp: u8,
) -> Result<Phase1Report, CeremonyError> {
    let _ = (ptau_in, ptau_out, name, beacon_hash, num_iterations_exp);
    todo!("phase1::beacon")
}

/// What `powersoftau verify` checked.
#[derive(Clone, Debug)]
pub struct PtauVerifyReport {
    pub power: u32,
    pub ceremony_power: u32,
    pub prepared: bool,
    /// One response hash per contribution, oldest first.
    pub contribution_hashes: Vec<Digest>,
    /// False when `power != ceremonyPower`, where the final next-challenge comparison is
    /// skipped because the points in the file are a prefix of the ones that were hashed
    /// (`powersoftau_verify.js:247-252`).
    pub next_challenge_checked: bool,
}

/// `powersoftau verify`: walk the contribution chain, check each key's same-ratio pairs
/// against the recomputed `g2_sp`, check the stored generators at the head of each
/// section, and recompute the final next-challenge hash.
pub fn verify(ptau: &Path) -> Result<PtauVerifyReport, CeremonyError> {
    let _ = ptau;
    todo!("phase1::verify")
}

/// Parse and bounds-check the two beacon arguments the way snarkjs does before it touches
/// a file: a non-empty, even-length hex string of at most 255 bytes, and an exponent in
/// `10..=63`.
pub fn parse_beacon_args(
    beacon_hash_hex: &str,
    num_iterations_exp: &str,
) -> Result<(Vec<u8>, u8), CeremonyError> {
    let _ = (beacon_hash_hex, num_iterations_exp);
    todo!("parse_beacon_args")
}
