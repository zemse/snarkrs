//! Groth16 BN254 keys and proofs between gnark v0.14 and snarkjs.
//!
//! * [`proof`]: proofs, public inputs and verifying keys, gnark's binary forms (raw and
//!   compressed) to snarkjs' `proof.json`, `public.json` and `verification_key.json`, and
//!   back. No arithmetic: both libraries verify the same equation over the same points.

pub mod proof;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConvertError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gnark(#[from] snarkrs_gnark::GnarkError),
    #[error(transparent)]
    Zkey(#[from] snarkrs_formats::ZkeyError),
    #[error(transparent)]
    Ceremony(#[from] snarkrs_ceremony::CeremonyError),
    #[error(transparent)]
    R1cs(#[from] snarkrs_gnark_solver::ReadError),
    #[error(transparent)]
    Accel(#[from] snarkrs_msm::AccelError),
    #[error(transparent)]
    Json(#[from] snarkrs_groth16::json::JsonError),
    /// The inputs do not describe one circuit and one key, or a key this crate cannot
    /// carry over.
    #[error("{what}: {reason}")]
    Mismatch { what: &'static str, reason: String },
}

pub(crate) fn mismatch(what: &'static str, reason: impl Into<String>) -> ConvertError {
    ConvertError::Mismatch {
        what,
        reason: reason.into(),
    }
}
