//! Groth16 on BN254, byte-compatible with snarkjs: load a `.zkey`, prove on the CPU or a
//! GPU, verify, and write the `proof.json` and `public.json` snarkjs reads.
//!
//! The prover, the verifier, the key and witness formats and the CPU backend are always
//! compiled. Everything else is a feature, off by default:
//!
//! | feature | adds |
//! | --- | --- |
//! | `metal` | [`metal`], Apple GPUs |
//! | `cuda` | [`cuda`], NVIDIA GPUs through the driver API, no toolkit at build time |
//! | `wgpu` | [`wgpu`], WebGPU on Metal, Vulkan, DX12 or a browser |
//! | `ceremony` | [`ceremony`], powers of tau, phase 2 setup and contributions |
//! | `witness` | [`witness`], circom's native witness binary as a subprocess |
//! | `witness-wasm` | `witness` plus circom's `circuit.wasm` on wasmtime |
//!
//! Proving from memory, with the witness already a `Vec<Fr>`:
//!
//! ```no_run
//! use snarkrs::{prove, verify, Backend, CpuBackend, ProvingKey, StageTimings, Witness};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse and prepare once, prove many times: on a GPU backend `prepare` is the upload.
//! let pk = ProvingKey::load("circuit.zkey".as_ref())?;
//! let n_public = pk.n_public;
//! let circuit = CpuBackend::new().prepare(pk)?;
//!
//! let w = Witness::load("circuit.wtns".as_ref())?.0;
//! let mut t = StageTimings::default();
//! let proof = prove(circuit.as_ref(), &w, &mut snarkrs::rand::thread_rng(), &mut t)?;
//!
//! let public = &w[1..=n_public];
//! verify(&circuit.key().vk, public, &proof)?;
//! snarkrs::write_proof("proof.json".as_ref(), &proof)?;
//! snarkrs::write_public("public.json".as_ref(), public)?;
//! # Ok(())
//! # }
//! ```
//!
//! Swap [`CpuBackend`] for `snarkrs::metal::MetalBackend`, `snarkrs::cuda::CudaBackend` or
//! `snarkrs::wgpu::WgpuProver` to change where it runs.

pub mod json;

pub use snarkrs_field as field;
pub use snarkrs_formats as formats;
pub use snarkrs_groth16 as groth16;
pub use snarkrs_msm as msm;
pub use snarkrs_ntt as ntt;

#[cfg(feature = "ceremony")]
pub use snarkrs_ceremony as ceremony;
#[cfg(feature = "cuda")]
pub use snarkrs_cuda as cuda;
#[cfg(feature = "metal")]
pub use snarkrs_metal as metal;
#[cfg(feature = "wgpu")]
pub use snarkrs_wgpu as wgpu;
#[cfg(feature = "witness")]
pub use snarkrs_witness as witness;

/// The RNG `prove` draws its blinders from. [`rand::thread_rng`] is seeded by the OS.
pub use ark_std::rand;

pub use json::{read_proof, read_public, write_proof, write_public};
pub use snarkrs_field::Fr;
pub use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey, ZkeyError};
pub use snarkrs_groth16::{
    cpu::CpuBackend,
    prove::prove,
    verify::{verify, VerifyError},
    Backend, PreparedCircuit, Proof, ProveError, StageTimings,
};
