//! snarkjs `.zkey` (Groth16, BN254) and `.wtns` parsing.
//!
//! Section layout, from the snarkjs binfile format:
//!   1 header (protocol id)          2 groth16 header (curve, nVars, nPublic, domainSize, alpha/beta/delta/...)
//!   3 IC (verifier)                 4 coefficients  (m, c, s, coef) records
//!   5 A  bases G1  [u_j(tau)]_1     6 B1 bases G1   [v_j(tau)]_1
//!   7 B2 bases G2  [v_j(tau)]_2     8 C  bases G1   (private wires only)
//!   9 H  bases G1                   10 contributions
//!
//! Points are stored in **Montgomery form, little-endian**, which is NOT what
//! `ark-serialize` expects; converting out of Montgomery is required and is the single
//! most common source of "my prover produces garbage" bugs. Field elements in section 4
//! are likewise Montgomery.

pub mod wtns;

use g16_field::*;

/// Everything needed to prove, as parsed from a `.zkey`.
pub struct ProvingKey {
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,
    /// Section 5: bases for the A MSM. Length `n_vars`.
    pub a_query: Vec<G1Affine>,
    /// Section 6: bases for the B-in-G1 MSM. Length `n_vars`.
    pub b_g1_query: Vec<G1Affine>,
    /// Section 7: bases for the B-in-G2 MSM. Length `n_vars`.
    pub b_g2_query: Vec<G2Affine>,
    /// Section 8: bases for the L MSM. Length `n_vars - n_public - 1`.
    pub l_query: Vec<G1Affine>,
    /// Section 9: bases for the H MSM. Length `domain_size` (snarkjs writes `domain_size`
    /// entries; only the first `domain_size - 1` are ever used).
    pub h_query: Vec<G1Affine>,
    /// Section 4, already sorted into CSR by constraint index. See [`Coefficients`].
    pub coeffs: Coefficients,
    pub vk: VerifyingKey,
}

/// Section 4 in CSR form: `A[c] = sum over row c of coef * w[s]`.
///
/// snarkjs stores an unordered `(matrix, constraint, signal, coef)` list, which on CPU
/// is consumed as a *scatter* under striped mutexes and on GPU cannot be consumed that
/// way at all (there is no 32-byte atomic). Sorting once at key load turns it into a
/// race-free gather that both backends share. This is a one-time cost paid in `prepare`,
/// never per proof.
pub struct Coefficients {
    /// `row_ptr[m][c]..row_ptr[m][c+1]` indexes into `signal`/`value`, for matrix m in {A, B}.
    pub row_ptr: [Vec<u32>; 2],
    pub signal: [Vec<u32>; 2],
    pub value: [Vec<Fr>; 2],
}

pub struct VerifyingKey {
    pub alpha_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    pub delta_g2: G2Affine,
    /// `gamma^-1 * L_i(tau) * g1` for the public inputs, length `n_public + 1`.
    pub ic: Vec<G1Affine>,
}

#[derive(Debug, thiserror::Error)]
pub enum ZkeyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a zkey file (bad magic {0:?})")]
    BadMagic([u8; 4]),
    #[error("unsupported protocol id {0} (only groth16 = 1)")]
    UnsupportedProtocol(u32),
    #[error("unsupported curve: expected BN254")]
    UnsupportedCurve,
    #[error("missing section {0}")]
    MissingSection(u32),
    #[error("malformed section {section}: {reason}")]
    Malformed { section: u32, reason: String },
}

impl ProvingKey {
    /// Parse a `.zkey` by mmap. Must not copy the point sections more than once.
    pub fn load(path: &std::path::Path) -> Result<Self, ZkeyError> {
        todo!("g16-zkey: implement ProvingKey::load")
    }
}

impl VerifyingKey {
    /// Parse snarkjs' `verification_key.json`, so we can verify against the same key
    /// snarkjs uses without trusting our own zkey reader.
    pub fn from_json(path: &std::path::Path) -> Result<Self, ZkeyError> {
        todo!("g16-zkey: implement VerifyingKey::from_json")
    }
}
