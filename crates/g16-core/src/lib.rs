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

/// Wraps a region of the proving path in a hotpath annotation, or expands to the region
/// unchanged when the `hotpath` feature is off.
///
/// `StageTimings` already reports the five stage groups, and a sampling profile already
/// reports which functions the CPU is in. Neither can answer "of the six transforms in
/// `compute_h`, which one", because all six are the same function on the same data and a
/// sampler folds them into one line. That is what these annotations are for, and it is why
/// they sit at region boundaries rather than on functions.
///
/// Two things to keep in mind when reading the report they produce:
///
/// * Several of these regions run concurrently under `rayon::join`, so their durations
///   overlap and do not sum to the proof. The five MSM annotations in particular are
///   wall-clock windows on five things running at once.
/// * The annotation itself costs something. It is off by default for that reason, and any
///   number quoted as a timing should come from a build without it.
#[cfg(feature = "hotpath")]
macro_rules! stage {
    ($label:expr, $body:expr) => {
        hotpath::measure_block!($label, $body)
    };
}

#[cfg(not(feature = "hotpath"))]
macro_rules! stage {
    ($label:expr, $body:expr) => {
        $body
    };
}

pub mod cpu;
/// snarkjs' `proof.json` and `public.json` encoding, which is the interop contract the whole
/// benchmark rests on. Here rather than in `g16-cli` because the browser prover has no
/// filesystem and still has to hand `snarkjs.groth16.verify` exactly these bytes.
pub mod json;
pub mod prove;
/// A deterministic execution trace, for diffing one machine's intermediate values against
/// another's. Debugging only: it pins stage 10's blinders, which no proving path may do.
pub mod trace;
pub mod verify;

use g16_field::*;
use g16_zkey::ProvingKey;

#[derive(thiserror::Error)]
pub enum ProveError {
    #[error("witness has {got} entries, proving key expects {want}")]
    WitnessLength { got: usize, want: usize },
    #[error("backend {backend}: {reason}")]
    Backend {
        backend: &'static str,
        reason: String,
    },
    /// Raised by [`prove::prove_with_blinders`] when the crate was built at opt-level 0 or 1.
    ///
    /// Not a safety rail on the arithmetic, a rail on wall clock. `cargo test` defaults to
    /// the dev profile, this workspace declares no `[profile.dev]`, and so the default is
    /// opt-level 0 with `overflow-checks` branching on every limb of the CIOS multiply. The
    /// campaign sweep that `campaign.rs` measures at under a minute took 19 minutes and
    /// 187 CPU-minutes that way before it was killed, with twelve variants still to go.
    #[error(
        "refusing to prove at opt-level 0 or 1: overflow checks on every limb of the CIOS \
         multiply make this tens of times slower than the numbers in README.md, which is how \
         a bare `cargo test` comes to saturate every core for the better part of an hour.\n\
         \n  release:  cargo test --release --workspace\
         \n  filtered: cargo test --workspace -- --skip campaign --skip roundtrip\
         \n  override: G16_ALLOW_UNOPTIMIZED_PROVING=1 (no effect on wasm32, which has no env)"
    )]
    Unoptimized,
}

/// `Debug` delegates to `Display`, because `Debug` is what `.expect()` and `.unwrap()` print
/// and the derived form throws the message away. `campaign.rs:279` reaches `prove` through an
/// `.expect`, and with the derive in place a variant carrying three lines of instructions
/// printed as the single word `Unoptimized`.
impl core::fmt::Debug for ProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

/// A Groth16 proof. Serialises to snarkjs' `proof.json` shape so `snarkjs groth16 verify`
/// can be used as an independent oracle against our own verifier.
///
/// # A proof is not an identity
///
/// Groth16 proofs are malleable, and this is a property of the scheme rather than a defect
/// in this implementation. Anyone holding a valid `(A, B, C)` can produce a different,
/// equally valid proof of the same statement without knowing the witness: pick any nonzero
/// `z` in `Fr` and take `(A * z^-1, B * z, C)`. The pairing check is
/// `e(A, B) = e(alpha, beta) e(L_bar, gamma) e(C, delta)` and `e(A z^-1, B z) = e(A, B)` by
/// bilinearity, so the left side is unchanged and the right side never mentioned `A` or `B`.
/// There are roughly `r` such proofs for every proof, and this crate cannot prevent it.
///
/// The consequence is for the caller, not for the verifier. **Never use proof bytes as a
/// nullifier, a replay key, a deduplication key, or any kind of identity.** An attacker who
/// observes a proof can mint unlimited distinct encodings of it that all verify. Key on
/// `(verifying key, public inputs)` instead, which is what the proof actually attests to.
///
/// This is pinned by a regression test rather than left as folklore, because it is the sort
/// of property that reads as a bug and gets "fixed" by someone adding a uniqueness check
/// that does not work.
#[derive(Clone, Debug)]
pub struct Proof {
    pub a: G1Affine,
    pub b: G2Affine,
    pub c: G1Affine,
}

/// The result of stages 0-4, carried from [`PreparedCircuit::compute_h`] into
/// [`PreparedCircuit::msms`] without forcing it through host memory.
///
/// The CPU backend has the coefficients on the host already and stores them inline. A GPU
/// backend leaves them in device memory and returns [`HPoly::Device`], carrying a handle
/// only it knows how to interpret plus the buffer length. Stage 9's MSM then reads its
/// scalars from the buffer stage 4 wrote, which is the entire point of grouping the stages
/// this way.
///
/// The handle travels through the value rather than being stashed on the circuit, because
/// `PreparedCircuit` is `&self` and explicitly safe to prove with concurrently. A circuit
/// holding "the last H buffer" would race between two in-flight proofs, and the race would
/// produce a proof that simply fails to verify, with nothing else to go on.
pub enum HPoly {
    /// Coefficients in host memory.
    Host(Vec<Fr>),
    /// Coefficients in backend-owned memory. `tag` identifies the owning backend so a
    /// handle cannot be handed to a backend that would misread it, and `data` is that
    /// backend's own handle: a buffer index, a wrapped device pointer, whatever it needs.
    Device {
        tag: &'static str,
        len: usize,
        data: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    },
}

impl HPoly {
    pub fn len(&self) -> usize {
        match self {
            HPoly::Host(v) => v.len(),
            HPoly::Device { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Host-visible coefficients, copying down from the device if that is where they live.
    /// Tests and cross-checks want this; the proving path should not, since forcing the
    /// copy is exactly the round trip the enum exists to avoid.
    pub fn to_host(&self) -> Option<&[Fr]> {
        match self {
            HPoly::Host(v) => Some(v),
            HPoly::Device { .. } => None,
        }
    }

    /// The backend's own handle, if `self` came from that backend. Returns `None` for a
    /// host vector or for another backend's handle, so a backend can fall back rather
    /// than misinterpret someone else's pointer.
    pub fn device_handle<T: std::any::Any + Send + Sync>(&self, want: &'static str) -> Option<&T> {
        match self {
            HPoly::Device { tag, data, .. } if *tag == want => data.downcast_ref::<T>(),
            _ => None,
        }
    }
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
    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError>;

    /// Stages 5-9. `h` must be the value [`Self::compute_h`] returned on this same
    /// circuit; passing one from a different circuit is a caller error and backends are
    /// entitled to reject it.
    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError>;

    /// Stages 0-9 in one call, which is the shape [`prove::prove`] drives.
    ///
    /// Only stage 9's MSM reads `H`; stages 5-8 read the witness, which exists before
    /// `compute_h` starts. Whether starting them early buys anything is a property of
    /// the backend, not of the proof: on the CPU backend an overlapped schedule measured
    /// even with this sequential one, because work stealing already absorbs the witness
    /// MSMs into stage 9's idle capacity either way. So the default runs the two stage
    /// groups in sequence, and a backend that has somewhere concurrent to put stages 5-8
    /// overrides this; the Metal backend does, on its second command queue.
    fn h_and_msms(&self, witness: &[Fr], t: &mut StageTimings) -> Result<MsmOutputs, ProveError> {
        let h = stage!("stages 0-4 compute_h", self.compute_h(witness, t))?;
        stage!("stages 5-9 msms", self.msms(witness, &h, t))
    }

    /// `H` in host memory, copying it down if that is where it is not.
    ///
    /// **Debugging only**, and the default is the honest answer for a backend that has not
    /// implemented it: `None`, rather than a silent empty vector that would read as "H is
    /// fine" in a trace. Forcing the copy is the round trip [`HPoly`] exists to avoid, and
    /// [`crate::trace`] is the only caller.
    fn h_to_host(&self, h: &HPoly) -> Option<Vec<Fr>> {
        h.to_host().map(<[Fr]>::to_vec)
    }
}
