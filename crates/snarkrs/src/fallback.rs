//! What `snarkrs groth16 prove` does when an accelerator hands back a proof that does not
//! verify, or fails to finish one.
//!
//! A logic bug fails the same way on every run and dies in CI. A GPU that returns a wrong
//! proof in production is far more likely to be transient: the DSN 2026 silent data
//! corruption study (arXiv 2605.04213) found the corruption warp aligned and structured, which
//! is exactly the shape that gives a wrong bucket sum and a failing pairing rather than a
//! crash, and `icicle-snark#18` is a CUDA prover producing three different invalid proofs in
//! three runs. So a failed self-verify on a GPU is retried once there, then proved on the CPU,
//! and each step is reported so the failure *rate* is what an operator alarms on.
//!
//! A device fault takes the same path: a Metal command buffer macOS kept killing past the
//! backend's own retries, a lost wgpu device. A device that stays lost fails the retry quickly
//! and the proof goes to the CPU.
//!
//! Only those two are retried ([`ProveError::SelfVerify`] and
//! [`ProveError::is_device_fault`]). Every other error is deterministic, and a witness that
//! does not satisfy the circuit fails on the CPU too, which is how the last message tells the
//! two apart.

use anyhow::Result;
use g16_core::{prove::prove, PreparedCircuit, Proof, ProveError, StageTimings};
use g16_field::Fr;

/// Why [`prove_with_fallback`] did not take a backend's answer.
#[derive(Debug, PartialEq, Eq)]
pub enum Cause {
    /// The proof failed its self-verify.
    SelfVerify,
    /// The device did not finish the proof. Carries the error text.
    Device(String),
}

impl Cause {
    fn of(attempt: &Result<Proof, ProveError>) -> Option<Self> {
        match attempt {
            Err(ProveError::SelfVerify(_)) => Some(Self::SelfVerify),
            Err(e) if e.is_device_fault() => Some(Self::Device(e.to_string())),
            _ => None,
        }
    }
}

impl std::fmt::Display for Cause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfVerify => f.write_str("failed its self-verify"),
            Self::Device(e) => write!(f, "hit a device fault ({e})"),
        }
    }
}

/// A step [`prove_with_fallback`] took, for the caller to report.
#[derive(Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The first proof on `backend` failed; proving again there.
    Retry { backend: &'static str, cause: Cause },
    /// The second one failed too; proving on the CPU.
    Cpu { backend: &'static str, cause: Cause },
    /// The CPU proof verified, so `backend` failed the same proof twice.
    DeviceSuspect { backend: &'static str },
}

/// [`prove`], retried once on the same backend and then on the CPU when the proof fails its
/// self-verify or the device faults. `cpu` builds the CPU circuit only if it is needed: the
/// key has usually moved into the accelerator at prepare, so the caller reloads it.
/// `timings` describe the attempt that produced the proof.
pub fn prove_with_fallback<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    rng: &mut R,
    timings: &mut StageTimings,
    cpu: impl FnOnce() -> Result<Box<dyn PreparedCircuit>>,
    report: &mut dyn FnMut(Fallback),
) -> Result<Proof> {
    let backend = circuit.backend_name();
    let attempt = prove(circuit, witness, rng, timings);
    let cause = match Cause::of(&attempt) {
        Some(cause) if backend != "cpu" => cause,
        _ => return Ok(attempt?),
    };

    report(Fallback::Retry { backend, cause });
    *timings = StageTimings::default();
    let attempt = prove(circuit, witness, rng, timings);
    let Some(cause) = Cause::of(&attempt) else {
        return Ok(attempt?);
    };

    report(Fallback::Cpu { backend, cause });
    let cpu = cpu()?;
    *timings = StageTimings::default();
    let proof = prove(cpu.as_ref(), witness, rng, timings)?;
    report(Fallback::DeviceSuspect { backend });
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use g16_core::{cpu::CpuBackend, Backend, HPoly, MsmOutputs};
    use g16_field::{G1Projective, PrimeGroup};
    use g16_zkey::{wtns::Witness, ProvingKey};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// What [`Faulty`] does to its first few MSMs.
    #[derive(Clone, Copy)]
    enum Fault {
        /// `A` comes back off by one generator, which is what a dropped kernel error or a
        /// flipped bucket looks like from outside: a well-formed proof that fails the pairing.
        Wrong,
        /// A command buffer the OS killed, past the backend's own retries.
        Device,
        /// A key the backend cannot use, which no retry changes.
        Key,
    }

    /// The CPU backend, except that its first `faults` MSMs fail as `fault` says.
    struct Faulty {
        inner: Box<dyn PreparedCircuit>,
        fault: Fault,
        faults: AtomicUsize,
    }

    impl PreparedCircuit for Faulty {
        fn backend_name(&self) -> &'static str {
            "faulty"
        }
        fn n_vars(&self) -> usize {
            self.inner.n_vars()
        }
        fn n_public(&self) -> usize {
            self.inner.n_public()
        }
        fn domain_size(&self) -> usize {
            self.inner.domain_size()
        }
        fn key(&self) -> &ProvingKey {
            self.inner.key()
        }
        fn compute_h(&self, w: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
            self.inner.compute_h(w, t)
        }
        fn msms(
            &self,
            w: &[Fr],
            h: &HPoly,
            t: &mut StageTimings,
        ) -> Result<MsmOutputs, ProveError> {
            let mut m = self.inner.msms(w, h, t)?;
            let left = self.faults.load(Ordering::SeqCst);
            if left == 0 {
                return Ok(m);
            }
            self.faults.store(left - 1, Ordering::SeqCst);
            match self.fault {
                Fault::Wrong => m.a_g1 += G1Projective::generator(),
                Fault::Device => {
                    return Err(ProveError::Device {
                        backend: "faulty",
                        reason: "msm: command buffer did not complete".to_string(),
                    })
                }
                Fault::Key => {
                    return Err(ProveError::Backend {
                        backend: "faulty",
                        reason: "h_query has 3 bases, domain size is 4".to_string(),
                    })
                }
            }
            Ok(m)
        }
    }

    fn tiny() -> Option<PathBuf> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul");
        dir.join("circuit.zkey").is_file().then_some(dir)
    }

    fn cpu(dir: &Path) -> Box<dyn PreparedCircuit> {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        CpuBackend::new().prepare(pk).unwrap()
    }

    /// The outcome, the steps reported, and whether the CPU circuit was built.
    fn run(
        fault: Fault,
        faults: usize,
        witness: &[Fr],
        dir: &Path,
    ) -> (Result<Proof>, Vec<Fallback>, bool) {
        let circuit = Faulty {
            inner: cpu(dir),
            fault,
            faults: AtomicUsize::new(faults),
        };
        let mut steps = Vec::new();
        let mut built = false;
        let out = prove_with_fallback(
            &circuit,
            witness,
            &mut StdRng::from_seed([3; 32]),
            &mut StageTimings::default(),
            || {
                built = true;
                Ok(cpu(dir))
            },
            &mut |f| steps.push(f),
        );
        (out, steps, built)
    }

    fn device() -> Cause {
        Cause::Device(
            "backend faulty: device fault: msm: command buffer did not complete".to_string(),
        )
    }

    #[test]
    fn one_fault_is_absorbed_by_the_retry_and_two_by_the_cpu() {
        let Some(dir) = tiny() else {
            eprintln!("SKIPPED fallback: no tiny_mul");
            return;
        };
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let backend = "faulty";

        let cases: [(Fault, fn() -> Cause); 2] = [
            (Fault::Wrong, || Cause::SelfVerify),
            (Fault::Device, device),
        ];
        for (fault, cause) in cases {
            let (out, steps, _) = run(fault, 0, &w, &dir);
            assert!(out.is_ok());
            assert!(steps.is_empty());

            let (out, steps, built) = run(fault, 1, &w, &dir);
            assert!(out.is_ok());
            assert!(!built);
            assert_eq!(
                steps,
                [Fallback::Retry {
                    backend,
                    cause: cause()
                }]
            );

            let (out, steps, _) = run(fault, 2, &w, &dir);
            assert!(out.is_ok());
            assert_eq!(
                steps,
                [
                    Fallback::Retry {
                        backend,
                        cause: cause()
                    },
                    Fallback::Cpu {
                        backend,
                        cause: cause()
                    },
                    Fallback::DeviceSuspect { backend }
                ]
            );
        }
    }

    /// A key the backend refuses, or a witness whose constant wire is not one, is refused
    /// again on a retry, so it fails at once and the CPU is never built.
    #[test]
    fn a_deterministic_error_is_not_retried() {
        let Some(dir) = tiny() else {
            eprintln!("SKIPPED fallback: no tiny_mul");
            return;
        };
        let mut w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let (out, steps, built) = run(Fault::Key, 1, &w, &dir);
        let e = out.unwrap_err();
        assert!(e.to_string().contains("h_query has 3 bases"), "{e}");
        assert!(steps.is_empty(), "{steps:?}");
        assert!(!built);

        w[0] = Fr::from(2u64);
        let (out, steps, built) = run(Fault::Wrong, 0, &w, &dir);
        let e = out.unwrap_err();
        assert!(e.to_string().contains("constant-one wire"), "{e}");
        assert!(steps.is_empty(), "{steps:?}");
        assert!(!built);
    }

    /// A witness that does not satisfy the circuit fails everywhere, so the CPU attempt fails
    /// too and the device is not blamed.
    #[test]
    fn an_unsatisfying_witness_fails_on_the_cpu_too_and_blames_no_device() {
        let Some(dir) = tiny() else {
            eprintln!("SKIPPED fallback: no tiny_mul");
            return;
        };
        let mut w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let last = w.len() - 1;
        w[last] += Fr::from(1u64);
        let (out, steps, _) = run(Fault::Wrong, 0, &w, &dir);
        assert!(out.is_err());
        assert!(!steps
            .iter()
            .any(|s| matches!(s, Fallback::DeviceSuspect { .. })));
    }
}
