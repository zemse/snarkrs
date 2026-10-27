//! The ethproofs client-side-proving benchmark, run against this prover.
//!
//! <https://ethproofs.org/csp-benchmarks> publishes six numbers per circuit: proving time,
//! verification time, peak memory, proof size, preprocessing size and constraint count.
//! The published `circom` row is circom + witnesscalc + rapidsnark. This crate keeps the
//! first two and swaps the third for `snarkrs-groth16`, so the difference between our row and
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
//!   iteration. This is the cold mode of `snarkrs bench`, not the warm one.
//! * **Verification re-reads the zkey too**, and derives the verifying key from it rather
//!   than loading a `verification_key.json`. That is why the published verify times run
//!   to hundreds of milliseconds for a pairing check that costs about one.
//!
//! What is deliberately *not* reproduced is criterion's sampling ramp; see [`run`].

use anyhow::Context;
use snarkrs_field::Fr;
use snarkrs_formats::{ProvingKey, VerifyingKey};
use snarkrs_groth16::{PreparedCircuit, Proof, StageTimings};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub mod circuits;
pub mod inputs;
pub mod mem;
pub mod metrics;
pub mod run;

/// A benchmark family. Upstream also has `blake3` and `poseidon2`, which have no circom
/// circuit at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    Sha256,
    Keccak,
    Poseidon,
    Ecdsa,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Sha256 => "sha256",
            Target::Keccak => "keccak",
            Target::Poseidon => "poseidon",
            Target::Ecdsa => "ecdsa",
        }
    }

    /// The sizes upstream's `BENCH_INPUT_PROFILE=full` iterates. Bytes for the hashes,
    /// field elements for poseidon, which is why the two lists differ. ECDSA has one
    /// circuit and no size to vary: the 32 is the digest it signs, and it is in the name
    /// because a witness generator is addressed as `family_size`.
    pub fn input_sizes(self) -> &'static [usize] {
        match self {
            Target::Sha256 | Target::Keccak => &[128, 256, 512, 1024, 2048],
            Target::Poseidon => &[2, 4, 8, 12, 16],
            Target::Ecdsa => &[32],
        }
    }

    pub const ALL: [Target; 4] = [
        Target::Sha256,
        Target::Keccak,
        Target::Poseidon,
        Target::Ecdsa,
    ];
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
///
/// `Auto` is a row policy, not a backend: per variant it runs the CPU below
/// [`AUTO_METAL_FROM`] constraints and Metal at or above, and each cell names the one it
/// ran. `Cpu` and `Metal` stay exactly what they say.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
    Auto,
}

/// `auto` runs Metal from this many constraints up and the CPU below.
///
/// From the M2 Max runs in `bench/results/csp/README.md`: the CPU was faster at 53,048
/// constraints (`sha256_128`, 47.36 ms against 49.73) and at all five poseidon sizes,
/// 517 to 2,092, and Metal was faster from 89,224 (`sha256_256`, 65.63 ms against 87.02)
/// on every larger variant. Nothing between those two was measured. A straight line
/// through them crosses near 56,700; 2^16 sits above that, so a circuit in the gap goes
/// to the CPU, which has no fixed per-proof cost to lose. Another machine has another
/// crossover.
pub const AUTO_METAL_FROM: usize = 1 << 16;

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Auto => "auto",
        }
    }

    /// The backend that proves a circuit of `n_constraints`: `Auto` picks by
    /// [`AUTO_METAL_FROM`], any other backend is itself.
    pub fn resolve(self, n_constraints: usize) -> Backend {
        match self {
            Backend::Auto if n_constraints >= AUTO_METAL_FROM => Backend::Metal,
            Backend::Auto => Backend::Cpu,
            b => b,
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

    /// The `feat` column for a circuit of `n_constraints`: which backend, for us; nothing
    /// for rapidsnark, which has one. Under `auto` it is `auto-` and the backend that ran,
    /// so each cell names its own. A key loaded under [`KeyPolicy::Unblinded`] adds
    /// `unblinded`, so the row, its file name and its breakdown all carry it.
    pub fn feat(&self, n_constraints: usize, key: KeyPolicy) -> Option<String> {
        let mut parts = Vec::new();
        if let Prover::G16(b) = self {
            if *b == Backend::Auto {
                parts.push(b.as_str());
            }
            parts.push(b.resolve(n_constraints).as_str());
        }
        if key == KeyPolicy::Unblinded {
            parts.push("unblinded");
        }
        (!parts.is_empty()).then(|| parts.join("-"))
    }

    pub fn label(&self, n_constraints: usize, key: KeyPolicy) -> String {
        match self.feat(n_constraints, key) {
            Some(f) => format!("{}/{f}", self.system()),
            None => self.system().to_string(),
        }
    }
}

/// How a variant's key was loaded.
///
/// `Unblinded` is the benchmark exception for the published poseidon keys, whose
/// `gamma_g2` equals their `delta_g2`: no phase-2 contribution, so anyone can forge a
/// proof against them, and `ProvingKey::load` and the checked `prove` both refuse them.
/// Only `--allow-unblinded-zkey` reaches it, only for a key that has that shape, and it
/// drops that one refusal: every other load, prove and verify check still runs. A row
/// measured this way says so in `feat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyPolicy {
    Checked,
    Unblinded,
}

/// `ProvingKey::load`, or with `allow_unblinded` its benchmark twin. On a key with
/// `gamma_g2 != delta_g2` the two run the same checks, so such a key comes back
/// `Checked` either way.
pub fn load_key(zkey: &Path, allow_unblinded: bool) -> anyhow::Result<(ProvingKey, KeyPolicy)> {
    let loaded = if allow_unblinded {
        ProvingKey::load_allowing_unblinded(zkey)
    } else {
        ProvingKey::load(zkey)
    };
    let pk = loaded.with_context(|| format!("loading {}", zkey.display()))?;
    let key = if allow_unblinded && pk.vk.gamma_g2 == pk.vk.delta_g2 {
        KeyPolicy::Unblinded
    } else {
        KeyPolicy::Checked
    };
    Ok((pk, key))
}

impl KeyPolicy {
    /// The checked `prove`, or the twin that skips only the `gamma_g2 != delta_g2` refusal
    /// in its self-verify.
    pub fn prove(
        self,
        circuit: &dyn PreparedCircuit,
        witness: &[Fr],
        timings: &mut StageTimings,
    ) -> Result<Proof, snarkrs_groth16::ProveError> {
        let rng = &mut ark_std::rand::thread_rng();
        match self {
            KeyPolicy::Checked => snarkrs_groth16::prove::prove(circuit, witness, rng, timings),
            KeyPolicy::Unblinded => {
                snarkrs_groth16::prove::prove_allowing_unblinded(circuit, witness, rng, timings)
            }
        }
    }

    /// The checked `verify`, or its twin as [`KeyPolicy::prove`] picks.
    pub fn verify(
        self,
        vk: &VerifyingKey,
        public: &[Fr],
        proof: &Proof,
    ) -> Result<(), snarkrs_groth16::verify::VerifyError> {
        match self {
            KeyPolicy::Checked => snarkrs_groth16::verify::verify(vk, public, proof),
            KeyPolicy::Unblinded => {
                snarkrs_groth16::verify::verify_allowing_unblinded(vk, public, proof)
            }
        }
    }
}

/// A key loaded and prepared, and what each half cost.
pub struct Prepared {
    pub circuit: Box<dyn PreparedCircuit>,
    pub key: KeyPolicy,
    pub load: Duration,
    pub prepare: Duration,
}

/// The key at `zkey` loaded and prepared on `backend`, with every check [`load_key`] runs.
///
/// Metal decodes the key straight into its device buffers with `MetalBackend::load`, so
/// its upload is timed as part of the load, and its prepare is making the backend. Every
/// other backend is [`load_key`] then [`make_backend`] and `prepare`, timed as such.
pub fn load_prepared(
    backend: Backend,
    zkey: &Path,
    allow_unblinded: bool,
) -> anyhow::Result<Prepared> {
    #[cfg(feature = "metal")]
    if backend == Backend::Metal {
        let t = Instant::now();
        let metal = snarkrs_metal::MetalBackend::new()?;
        let prepare = t.elapsed();
        let t = Instant::now();
        let loaded = if allow_unblinded {
            metal.load_allowing_unblinded(zkey)
        } else {
            metal.load(zkey)
        };
        let circuit = loaded.with_context(|| format!("loading {}", zkey.display()))?;
        let load = t.elapsed();
        let vk = &circuit.key().vk;
        return Ok(Prepared {
            key: if allow_unblinded && vk.gamma_g2 == vk.delta_g2 {
                KeyPolicy::Unblinded
            } else {
                KeyPolicy::Checked
            },
            circuit: Box::new(circuit),
            load,
            prepare,
        });
    }
    let t = Instant::now();
    let (pk, key) = load_key(zkey, allow_unblinded)?;
    let load = t.elapsed();
    let t = Instant::now();
    let circuit = make_backend(backend)?.prepare(pk)?;
    Ok(Prepared {
        circuit,
        key,
        load,
        prepare: t.elapsed(),
    })
}

/// Refuses rather than falling back. A row that silently measured the CPU while labelled
/// `metal` is worse than no row. So `auto` that resolves to Metal in a build without it
/// is an error too, not a CPU cell.
pub fn make_backend(kind: Backend) -> anyhow::Result<Box<dyn snarkrs_groth16::Backend>> {
    match kind {
        Backend::Auto => anyhow::bail!("`auto` is resolved per circuit before a backend is made"),
        Backend::Cpu => Ok(Box::new(snarkrs_groth16::cpu::CpuBackend::new())),
        #[cfg(feature = "metal")]
        Backend::Metal => Ok(Box::new(snarkrs_metal::MetalBackend::new()?)),
        #[cfg(not(feature = "metal"))]
        Backend::Metal => anyhow::bail!("built without the `metal` feature"),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Box::new(snarkrs_cuda::CudaBackend::new()?)),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!("built without the `cuda` feature"),
    }
}
