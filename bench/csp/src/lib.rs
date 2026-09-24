//! The ethproofs client-side-proving benchmark, run against this prover.
//!
//! <https://ethproofs.org/csp-benchmarks> publishes six numbers per circuit: proving time,
//! verification time, peak memory, proof size, preprocessing size and constraint count.
//! The published `circom` row is circom + witnesscalc + rapidsnark. This crate keeps the
//! first two and swaps the third for `g16-core`, so the difference between our row and
//! theirs is the prover and nothing else.
//!
//! Three things about the upstream protocol are easy to get wrong and change the answer
//! by more than the prover does, so they are reproduced here deliberately:
//!
//! * **Witness generation is inside the timed region.** Upstream's `prove` spawns the
//!   witnesscalc thread itself and the criterion closure joins it. Timing only the
//!   prover would report a number 10-20% below what the page compares against.
//! * **So is the zkey read.** `groth16_prover_zkey_file_wrapper` takes a *path*, so every
//!   timed iteration re-reads the whole key. For keccak_2048 that is 1.1 GB per
//!   iteration. This is the cold mode of `g16 bench`, not the warm one.
//! * **Verification re-reads the zkey too**, and derives the verifying key from it rather
//!   than loading a `verification_key.json`. That is why the published verify times run
//!   to hundreds of milliseconds for a pairing check that costs about one.
//!
//! What is deliberately *not* reproduced is criterion's sampling ramp; see [`run`].

use std::path::{Path, PathBuf};

pub mod circuits;
pub mod inputs;
pub mod mem;
pub mod metrics;
pub mod run;

/// A benchmark family. Upstream also has `blake3`, `poseidon2` and `ecdsa`; the first two
/// have no circom circuit at all and `ecdsa` is tracked separately because its witness
/// generator has to be compiled from source rather than shipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    Sha256,
    Keccak,
    Poseidon,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Sha256 => "sha256",
            Target::Keccak => "keccak",
            Target::Poseidon => "poseidon",
        }
    }

    /// The sizes upstream's `BENCH_INPUT_PROFILE=full` iterates. Bytes for the hashes,
    /// field elements for poseidon, which is why the two lists differ.
    pub fn input_sizes(self) -> &'static [usize] {
        match self {
            Target::Sha256 | Target::Keccak => &[128, 256, 512, 1024, 2048],
            Target::Poseidon => &[2, 4, 8, 12, 16],
        }
    }

    pub const ALL: [Target; 3] = [Target::Sha256, Target::Keccak, Target::Poseidon];
}

/// One circuit: a target at one input size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub target: Target,
    pub input_size: usize,
}

impl Variant {
    pub fn name(&self) -> String {
        format!("{}_{}", self.target.as_str(), self.input_size)
    }

    /// Every variant of every target, in the order upstream runs them.
    pub fn all() -> Vec<Variant> {
        Target::ALL
            .iter()
            .flat_map(|&target| {
                target
                    .input_sizes()
                    .iter()
                    .map(move |&input_size| Variant { target, input_size })
            })
            .collect()
    }

    pub fn zkey(&self, artifacts: &Path) -> PathBuf {
        artifacts.join(self.name()).join("circuit.zkey")
    }

    /// `zkey + <name>.cpp + <name>.dat`, which is the set upstream's
    /// `sum_file_sizes_in_the_dir` happens to sum: those three files are the whole of the
    /// directory the zkey sits in over there. Spelled out rather than walking our own
    /// artifact directory, which also holds inputs, a vkey and a cached witness.
    pub fn preprocessing_size(&self, artifacts: &Path, witness_src: &Path) -> anyhow::Result<u64> {
        let name = self.name();
        let family = self.target.as_str();
        let mut total = std::fs::metadata(self.zkey(artifacts))?.len();
        for ext in ["cpp", "dat"] {
            let p = witness_src
                .join(family)
                .join(&name)
                .join(format!("{name}.{ext}"));
            total += std::fs::metadata(&p)
                .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
                .len();
        }
        Ok(total)
    }
}

/// Which prover to measure. `wgpu` is absent on purpose: it is the browser backend and
/// this benchmark is a native one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
        }
    }
}

/// What a row measures. The point of the second variant is that the published `circom`
/// row is rapidsnark, on hardware we do not have, against circuits that have since been
/// replaced. Running rapidsnark here, over the same zkeys, from the same witness, under
/// the same loop, is the only comparison with nothing left to argue about.
#[derive(Clone, Debug)]
pub enum Prover {
    G16(Backend),
    /// The `rapidsnark` CLI: `prover <zkey> <wtns> <proof.json> <public.json>`.
    Rapidsnark(std::path::PathBuf),
}

impl Prover {
    /// The `name` column.
    pub fn system(&self) -> &'static str {
        match self {
            Prover::G16(_) => "g16",
            Prover::Rapidsnark(_) => "rapidsnark",
        }
    }

    /// The `feat` column: which backend, for us; nothing for rapidsnark, which has one.
    pub fn feat(&self) -> Option<String> {
        match self {
            Prover::G16(b) => Some(b.as_str().to_string()),
            Prover::Rapidsnark(_) => None,
        }
    }

    pub fn label(&self) -> String {
        match self.feat() {
            Some(f) => format!("{}/{f}", self.system()),
            None => self.system().to_string(),
        }
    }
}

/// Refuses rather than falling back. A row that silently measured the CPU while labelled
/// `metal` is worse than no row.
pub fn make_backend(kind: Backend) -> anyhow::Result<Box<dyn g16_core::Backend>> {
    match kind {
        Backend::Cpu => Ok(Box::new(g16_core::cpu::CpuBackend::new())),
        #[cfg(feature = "metal")]
        Backend::Metal => Ok(Box::new(g16_metal::MetalBackend::new()?)),
        #[cfg(not(feature = "metal"))]
        Backend::Metal => anyhow::bail!("built without the `metal` feature"),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Box::new(g16_cuda::CudaBackend::new()?)),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!("built without the `cuda` feature"),
    }
}
