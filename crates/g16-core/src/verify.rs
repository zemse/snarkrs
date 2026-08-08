//! Groth16 verification.
//!
//! `e(A, B) = e(alpha, beta) * e(L_bar, gamma) * e(C, delta)`, rearranged into a single
//! 4-pair multi-Miller loop with `A` negated so the check is `product == 1` and only one
//! final exponentiation is needed.

use crate::Proof;
use g16_field::*;
use g16_zkey::VerifyingKey;

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("expected {want} public inputs, got {got}")]
    PublicInputCount { got: usize, want: usize },
    #[error("pairing check failed")]
    PairingFailed,
}

pub fn verify(vk: &VerifyingKey, public: &[Fr], proof: &Proof) -> Result<(), VerifyError> {
    todo!("g16-core::verify: implement verify")
}

/// Aggregate the public inputs: `L_bar = sum_i w_i * IC[i]`. An MSM over `n_public + 1`
/// points, small enough that a plain double-and-add loop is the right choice.
pub fn aggregate_public(vk: &VerifyingKey, public: &[Fr]) -> Result<G1Projective, VerifyError> {
    todo!("g16-core::verify: implement aggregate_public")
}
