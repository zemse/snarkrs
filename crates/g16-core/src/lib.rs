//! Groth16 proving and verification, and the backend contract every accelerator implements.
//!
//! Stage numbering follows Bloemen's write-up (<https://xn--2-umb.com/22/groth16/>),
//! expanded into implementable steps:
//!
//! ```text
//!  -1  witness generation          CPU only, forever. Data-dependent control flow.
//!   0  coefficient gather          A,B,C evals = w . (A,B,C)      | backend
//!   1  3 x iNTT                    onto coefficient form          | backend
//!   2  3 x coset shift             x[i] *= g^i                    |  compute_h()
//!   3  3 x NTT on the coset        evaluations on y               |
//!   4  H = A*B - C                 elementwise                    |
//!   5  MSM A   -> G1  (zkey s5)    scalars = full witness         |
//!   6  MSM B   -> G2  (zkey s7)    scalars = full witness         | backend
//!   7  MSM B   -> G1  (zkey s6)    scalars = full witness         |  msms()
//!   8  MSM L   -> G1  (zkey s8)    scalars = private witness      |
//!   9  MSM H   -> G1  (zkey s9)    scalars = H evaluations        |
//!  10  sample r, s                 CPU only. CSPRNG, trust boundary.
//!  11  blind and assemble          CPU. ~10 group ops, O(1).
//! ```
//!
//! The backend boundary is drawn at stages 0-4 and 5-9 rather than at individual
//! primitives, because a GPU backend must keep the three domain vectors resident across
//! all six transforms and must be free to run the five MSMs concurrently on its own
//! queues. A per-primitive trait would force a host round trip between every step and
//! measure the bus instead of the arithmetic.

pub mod cpu;
pub mod prove;
pub mod verify;

use g16_field::*;
use g16_zkey::ProvingKey;

#[derive(Debug, thiserror::Error)]
pub enum ProveError {
    #[error("witness has {got} entries, proving key expects {want}")]
    WitnessLength { got: usize, want: usize },
    #[error("backend {backend}: {reason}")]
    Backend {
        backend: &'static str,
        reason: String,
    },
}

/// A Groth16 proof. Serialises to snarkjs' `proof.json` shape so `snarkjs groth16 verify`
/// can be used as an independent oracle against our own verifier.
#[derive(Clone, Debug)]
pub struct Proof {
    pub a: G1Affine,
    pub b: G2Affine,
    pub c: G1Affine,
}

/// The five MSM results, stages 5-9.
pub struct MsmOutputs {
    pub a_g1: G1Projective,
    pub b_g2: G2Projective,
    pub b_g1: G1Projective,
    pub l_g1: G1Projective,
    pub h_g1: G1Projective,
}

/// Wall-clock for each stage group, filled in by the backend so `bench` can report the
/// split without re-running. All in microseconds.
#[derive(Default, Clone, Copy, Debug)]
pub struct StageTimings {
    pub gather_us: u64,
    pub ntt_us: u64,
    pub pointwise_us: u64,
    pub msm_us: u64,
    pub assemble_us: u64,
}

/// Creates [`PreparedCircuit`]s. One per accelerator.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Witness-independent work: sort section 4 into CSR, build twiddles, and for a GPU
    /// backend upload every base vector and keep it resident. This is the whole of the
    /// "warm vs cold" distinction, and it is why a prover that calls this per proof
    /// measures its own key upload instead of its arithmetic.
    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError>;
}

/// A circuit whose witness-independent data is loaded and (on GPU) device-resident.
/// Must be safe to call from multiple threads against the same instance.
pub trait PreparedCircuit: Send + Sync {
    fn backend_name(&self) -> &'static str;
    fn n_vars(&self) -> usize;
    fn n_public(&self) -> usize;
    fn domain_size(&self) -> usize;
    /// Read-only access to the key, for the stages that stay on the host (11).
    fn key(&self) -> &ProvingKey;

    /// Stages 0-4. Returns `H` evaluated on the coset, length `domain_size`.
    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<Vec<Fr>, ProveError>;

    /// Stages 5-9. `h` is the output of [`Self::compute_h`]. A GPU backend is expected to
    /// take `h` from its own device buffer rather than the slice when it can.
    fn msms(
        &self,
        witness: &[Fr],
        h: &[Fr],
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError>;
}
