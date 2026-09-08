//! `groth16 setup`: an `.r1cs` and a prepared `.ptau` in, an initial `.zkey` out.
//!
//! `zkey_new.js:36-171`. Four things about this command surprise people who have only
//! read the paper:
//!
//! * **The initial key has `gamma = delta = 1` and setup performs no division.**
//!   `zkey_new.js:127-129` writes the plain generators as `gamma2`, `delta1` and `delta2`,
//!   so sections 3 and 8 are undivided. `zkey contribute` is what introduces a delta at
//!   all (`zkey_contribute.js:63-64, :90-92`).
//! * **This is not one big MSM.** `composeAndWritePoints` produces one output point per
//!   signal, dispatching a single scalar multiplication when a signal appears once and a
//!   multiexp when it appears more than once, then one batch-to-affine over the chunk
//!   (`zkey_new.js:459-491`). It is MSM-bound overall, which is why the backend is a
//!   parameter, but the shape is per-signal.
//! * **Section 9 involves no arithmetic at all.** `writeHs` is a strided gather: read the
//!   `2*domainSize` Lagrange block of ptau section 12 and take every odd element
//!   (`zkey_new.js:186-191`). The `cirPower == Fr::TWO_ADICITY` case is different because
//!   that block came out of the coset branch and is laid out as two contiguous halves
//!   rather than interleaved, so it reads the second half in one go (`:192-194`).
//! * **The section write order is `1, 2, 4, 3, 9, 8, 5, 6, 7, 10`**, not id order
//!   (`zkey_new.js:76-168`). It changes both the file layout and `csHash`, so getting it
//!   wrong is not cosmetic.
//!
//! The one shipped bug that has to be reproduced rather than fixed lives in
//! [`hash_h_points`].

use std::path::Path;

use g16_msm::MsmBackend;

use crate::transcript::Digest;
use crate::CeremonyError;

/// zkey section ids, in id order rather than write order.
pub const S_PROTOCOL: u32 = 1;
pub const S_HEADER: u32 = 2;
pub const S_IC: u32 = 3;
pub const S_COEFFS: u32 = 4;
pub const S_A: u32 = 5;
pub const S_B1: u32 = 6;
pub const S_B2: u32 = 7;
pub const S_C: u32 = 8;
pub const S_H: u32 = 9;
pub const S_MPC_PARAMS: u32 = 10;

/// Sections a fresh zkey holds, and the number `createBinFile` declares
/// (`zkey_new.js:49`).
pub const ZKEY_SECTIONS: u32 = 10;

/// `.zkey` magic, and the version this crate writes.
pub const ZKEY_MAGIC: &[u8; 4] = b"zkey";
pub const ZKEY_VERSION: u32 = 1;
/// Both readers in `zkey_verify_frominit.js` pass a max version of 2 while every writer
/// emits 1, so 1 and 2 are both acceptable on read.
pub const ZKEY_MAX_VERSION: u32 = 2;

/// Section 4 record stride on BN254: three u32 indices then one 32-byte scalar.
pub const COEF_RECORD_BYTES: usize = 12 + crate::N8;

/// What a setup produced, for the CLI to print and a test to assert on.
#[derive(Clone, Debug)]
pub struct SetupReport {
    pub n_vars: usize,
    /// `nOutputs + nPubInputs`, the ONE signal excluded.
    pub n_public: usize,
    pub n_constraints: usize,
    pub domain_size: usize,
    pub cir_power: u32,
    /// Section 4's record count.
    pub n_coefs: usize,
    /// The circuit hash written into section 10, and the value every later contribution
    /// transcript is seeded with.
    pub cs_hash: Digest,
}

/// `zkey_new.js:59` with the floor-log2 of `misc.js:53`:
/// `floor(log2(nConstraints + nPubInputs + nOutputs)) + 1`.
///
/// Note the consequence, which is not an off-by-one: for a total that is already a power
/// of two the domain still doubles, so `2^cirPower` is always strictly greater than
/// `nConstraints + nPublic`. That is precisely what makes the synthetic `nConstraints + s`
/// row indices land inside the `domainSize`-element Lagrange window.
pub fn circuit_power(n_constraints: usize, n_public: usize) -> u32 {
    crate::floor_log2((n_constraints + n_public) as u64) + 1
}

/// Run the whole setup.
///
/// `msm` is threaded through rather than hardcoded because this command is MSM-bound and
/// is the first thing in the ceremony worth putting on a GPU. Nothing else about it wants
/// a device.
pub fn setup(
    r1cs_path: &Path,
    ptau_path: &Path,
    out_path: &Path,
    msm: &dyn MsmBackend,
) -> Result<SetupReport, CeremonyError> {
    let _ = (r1cs_path, ptau_path, out_path, msm);
    todo!("setup")
}

/// Which of the four ptau Lagrange sections a term's base comes from
/// (`zkey_new.js:38-41`, dispatched through `sBuffs` at `:415-420`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BaseTag {
    /// Section 12, `L_j(tau) * G1`.
    TauG1,
    /// Section 13, `L_j(tau) * G2`.
    TauG2,
    /// Section 14, `alpha * L_j(tau) * G1`.
    AlphaTauG1,
    /// Section 15, `beta * L_j(tau) * G1`.
    BetaTauG1,
}

/// One term of one output point: which Lagrange buffer, which element of it, and where
/// the scalar lives.
///
/// `coef_ptr` of `None` is snarkjs' `-1` sentinel and means "the scalar is 1"
/// (`zkey_new.js:436-446`). It appears only in the public-signal tail, which appends a
/// synthetic constraint row per public signal with `a = 1, b = 0, c = 0` so that `A_s(x)`
/// is nonzero for every public signal.
#[derive(Clone, Copy, Debug)]
pub struct PointTerm {
    pub tag: BaseTag,
    /// Element index into the tagged Lagrange block, which is the constraint index.
    pub index: u32,
    pub coef_ptr: Option<u32>,
}

/// The five accumulator lists `processConstraints` builds, before any point is computed.
///
/// Lengths are fixed at construction and never grow: `A`, `B1` and `B2` are `nVars` long,
/// `C` is `nVars - nPublic - 1`, and `IC` is `nPublic + 1`. That matters beyond bookkeeping
/// because each length is hashed into `csHash` as a big-endian u32 before its points. A
/// signal that appears in no constraint leaves an empty slot, and an empty slot writes the
/// group identity, which serialises as all-zero bytes.
pub struct Accumulators {
    pub ic: Vec<Vec<PointTerm>>,
    pub a: Vec<Vec<PointTerm>>,
    pub b1: Vec<Vec<PointTerm>>,
    pub b2: Vec<Vec<PointTerm>>,
    pub c: Vec<Vec<PointTerm>>,
    /// Section 4 in push order: `(matrix, constraint, signal, coef_ptr)`, matrix 0 for A
    /// and 1 for B. C terms are never recorded. Do not sort this: the order is execution
    /// order and the file stores it as such.
    pub coefs: Vec<CoefRecord>,
}

/// One section-4 record.
#[derive(Clone, Copy, Debug)]
pub struct CoefRecord {
    /// 0 for A, 1 for B. Never 2.
    pub matrix: u32,
    pub constraint: u32,
    pub signal: u32,
    /// `None` is the scalar 1, as in [`PointTerm`].
    pub coef_ptr: Option<u32>,
}

/// Walk the r1cs once and build every accumulator, in the three-pass order
/// `zkey_new.js:222-300` uses: A terms, then B terms, then C terms per constraint, then
/// the public-signal tail.
pub fn process_constraints(
    r1cs: &crate::r1cs::R1cs,
    domain_size: usize,
) -> Result<Accumulators, CeremonyError> {
    let _ = (r1cs, domain_size);
    todo!("process_constraints")
}

/// One output point per accumulator slot: identity for an empty slot, a single scalar
/// multiplication for a slot with one term, a multiexp otherwise.
pub fn compose_points_g1(
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &crate::r1cs::R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<g16_field::G1Affine>, CeremonyError> {
    let _ = (slots, bases, r1cs, msm);
    todo!("compose_points_g1")
}

/// [`compose_points_g1`] for section 7, the only G2 output.
pub fn compose_points_g2(
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &crate::r1cs::R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<g16_field::G2Affine>, CeremonyError> {
    let _ = (slots, bases, r1cs, msm);
    todo!("compose_points_g2")
}

/// The four `domainSize`-point Lagrange blocks setup reads out of the ptau, one per
/// [`BaseTag`], each from element `domainSize - 1` of its section.
pub struct LagrangeBlocks {
    pub tau_g1: Vec<g16_field::G1Affine>,
    pub tau_g2: Vec<g16_field::G2Affine>,
    pub alpha_tau_g1: Vec<g16_field::G1Affine>,
    pub beta_tau_g1: Vec<g16_field::G1Affine>,
}

impl LagrangeBlocks {
    pub fn read(ptau: &crate::ptau::Ptau, domain_size: usize) -> Result<Self, CeremonyError> {
        let _ = (ptau, domain_size);
        todo!("LagrangeBlocks::read")
    }
}

/// Section 9: every odd element of the `2*domainSize` Lagrange block of ptau section 12.
/// No arithmetic. Only section 12 carries the `power+1` block, which is why this read is
/// possible at all when `cirPower == power`.
pub fn write_hs(
    ptau: &crate::ptau::Ptau,
    domain_size: usize,
) -> Result<Vec<g16_field::G1Affine>, CeremonyError> {
    let _ = (ptau, domain_size);
    todo!("write_hs")
}

/// Fold the H difference points into `csHash`, bug included.
///
/// The points are `tauG1[i + domainSize] - tauG1[i]` from ptau **section 2**, the raw
/// monomial powers, not the Lagrange section 12 (`zkey_new.js:517-518`).
///
/// `zkey_new.js:511` reads `Math.min(domainSize-1, CHUNK_SIZE)` and forgot the `- i`, so
/// `n` does not depend on the chunk offset. With `CHUNK_SIZE = 1<<14`, a `domainSize` of
/// 2^15 or more hashes a full 16384 points per chunk over `2^(k-14)` chunks, that is
/// `domainSize` points after announcing `domainSize - 1`. When additionally
/// `cirPower == power` the last chunk reads 64 bytes past the end of section 2, through a
/// raw `fd.read` that bounds-checks nothing, and folds section 3's header and the first 52
/// bytes of `tauG2[0]` into the digest.
///
/// This is the shipped behaviour and `csHash` is defined by it, so reproduce it. It is
/// also the highest-risk line in the whole setup path: confirm it against a real
/// `snarkjs groth16 setup` with `cirPower == power >= 15` before trusting our digest.
/// The verifier is immune because `sameRatioH` multiplies index `domainSize - 1` by zero.
pub fn hash_h_points(
    ptau: &crate::ptau::Ptau,
    domain_size: usize,
    transcript: &mut crate::transcript::Transcript,
) -> Result<(), CeremonyError> {
    let _ = (ptau, domain_size, transcript);
    todo!("hash_h_points")
}
