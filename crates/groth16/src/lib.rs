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
/// reports which functions the CPU is in. Neither can answer "of the three domain vectors,
/// which one", because all three go through the same function on the same shape of data
/// and a sampler folds them into one line. That is what these annotations are for, and it
/// is why they sit at region boundaries rather than on functions.
///
/// They cannot go finer than that. The three stage-1-3 regions separate the three vectors
/// from each other and from stages 0 and 4; the six transforms cannot be split here at
/// all, because `intt_coset_ntt` fuses them per cache block.
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
/// benchmark rests on. Here rather than in the CLI because the browser prover has no
/// filesystem and still has to hand `snarkjs.groth16.verify` exactly these bytes.
pub mod json;
/// Turning off macOS malloc's large cache, which otherwise keeps freed buffers in the peak.
pub mod malloc;
pub mod prove;
/// A deterministic execution trace, for diffing one machine's intermediate values against
/// another's. Debugging only: it pins stage 10's blinders, which no proving path may do.
pub mod trace;
pub mod verify;

use snarkrs_field::*;
use snarkrs_formats::ProvingKey;

#[derive(thiserror::Error)]
#[non_exhaustive]
pub enum ProveError {
    #[error("witness has {got} entries, proving key expects {want}")]
    WitnessLength { got: usize, want: usize },
    #[error("backend {backend}: {reason}")]
    Backend {
        backend: &'static str,
        reason: String,
    },
    /// Work the backend accepted did not complete: a Metal command buffer macOS killed, a
    /// lost wgpu device. Unlike [`Self::Backend`] it says nothing about the witness or the
    /// key, so the same proof may succeed on a second attempt.
    #[error("backend {backend}: device fault: {reason}")]
    Device {
        backend: &'static str,
        reason: String,
    },
    #[error("w[0] is the constant-one wire and must be 1")]
    ConstantWire,
    #[error("the RNG produced a zero blinder, which a CSPRNG does with probability 2^-254")]
    ZeroBlinder,
    /// Raised by [`prove::prove`] when the proof it just produced does not verify against
    /// the key it was proved with.
    #[error(
        "the proof does not verify against the key it was proved with ({0}). Either the \
         witness does not satisfy the circuit, the key is not what it claims to be, or the \
         prover or its accelerator computed something wrong"
    )]
    SelfVerify(verify::VerifyError),
    /// Raised by every proving entry point in [`prove`] when the crate was built at opt-level 0
    /// or 1.
    ///
    /// Not a safety rail on the arithmetic, a rail on wall clock. `cargo test` defaults to
    /// the dev profile, this workspace declares no `[profile.dev]`, and so the default is
    /// opt-level 0 with `overflow-checks` branching on every limb of the CIOS multiply. The
    /// campaign sweep that `campaign.rs` measures at under a minute took 19 minutes and
    /// 187 CPU-minutes that way before it was killed, with twelve variants still to go.
    ///
    /// The same holds for a library user's debug build, whose profile is the one our crates
    /// are compiled with, so the message is written for them: what happened, and the
    /// `Cargo.toml` lines that fix it without giving up a debug build of their own code.
    #[error(
        "refusing to prove: snarkrs was compiled without optimisation (opt-level {level}), \
         which makes proving tens of times slower, minutes where an optimised build takes \
         seconds. This is usually a debug build (`cargo run`, `cargo test`).\n\
         \nFix it in one of two ways:\n\
         \n  1. Optimise dependencies in debug builds. Your own code stays quick to compile\
         \n     and debug. Add this to the Cargo.toml at your workspace root:\n\
         \n         [profile.dev.package.\"*\"]\
         \n         opt-level = 3\n\
         \n  2. Build in release mode: `cargo run --release`, `cargo test --release`.\n\
         \nTo prove unoptimised anyway, for example in a test that has to run in a debug\
         \nbuild, set G16_ALLOW_UNOPTIMIZED_PROVING=1. That works on native targets only:\
         \nwasm32 has no environment, so there use 1 or 2.",
        level = env!("SNARKRS_OPT_LEVEL")
    )]
    Unoptimized,
}

impl ProveError {
    /// Whether the device failed, rather than anything the caller passed. Every other variant
    /// fails the same way on a second attempt and on the CPU.
    pub fn is_device_fault(&self) -> bool {
        matches!(self, Self::Device { .. })
    }
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// `H` is a linear image of the witness, so a host copy is zeroed when it goes rather than
/// left in the heap for the next allocation. A device copy is the owning backend's to scrub.
impl Drop for HPoly {
    fn drop(&mut self) {
        if let HPoly::Host(v) = self {
            scrub(v);
        }
    }
}

/// Zero a witness-derived vector so the stores survive: volatile writes the optimiser may
/// not drop as dead, then one fence.
///
/// Not `zeroize`'s slice impl, which is correct and costs a fence per element on one
/// thread: 4-7% of a warm CPU proof on js_16x16_d32 and keccak256 for H plus stage 0's B
/// and C. This is the same writes spread over the pool, with the fence paid once.
pub fn scrub(v: &mut [Fr]) {
    use rayon::prelude::*;
    v.par_chunks_mut(1 << 12).for_each(|chunk| {
        for x in chunk {
            for limb in x.0 .0.iter_mut() {
                // SAFETY: `limb` is a valid, aligned, exclusive `&mut u64`.
                unsafe { core::ptr::write_volatile(limb, 0) };
            }
        }
    });
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// The check of the result in [`prove::prove`]. Zero from [`prove::prove_unchecked`].
    pub verify_us: u64,
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
    /// Read-only key data for verification and host-side blinding/assembly (stage 11).
    ///
    /// Preserves `n_vars`, `n_public`, `domain_size`, the full `vk` (including `ic`), and
    /// `alpha_g1`, `beta_g1`, `beta_g2`, `delta_g1`, `delta_g2` from the prepared key.
    /// The dimensions agree with this circuit's corresponding accessors.
    ///
    /// Bulk storage (`coeffs` and the five query vectors) is backend-dependent and may
    /// be released after upload. This is not guaranteed to be a complete proving key
    /// reusable to prepare another backend; retain or reload the original key for that.
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
