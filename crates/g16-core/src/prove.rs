//! Stages 10-11: blinding and assembly, plus the top-level `prove` that drives a backend.

use crate::{PreparedCircuit, Proof, ProveError, StageTimings};
use g16_field::*;

/// Full proof: stages 0-11. `rng` supplies the zero-knowledge blinders `r` and `s`.
pub fn prove<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    rng: &mut R,
    timings: &mut StageTimings,
) -> Result<Proof, ProveError> {
    todo!("g16-core::prove: implement prove")
}

/// Deterministic variant with caller-supplied `r`, `s`. Used only by tests, so a proof
/// can be compared against a reference implementation bit for bit. Never use in
/// production: reusing `r`/`s` across proofs of different witnesses leaks the witness.
pub fn prove_with_blinders(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    r: Fr,
    s: Fr,
    timings: &mut StageTimings,
) -> Result<Proof, ProveError> {
    todo!("g16-core::prove: implement prove_with_blinders")
}
