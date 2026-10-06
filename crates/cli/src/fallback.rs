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

use anyhow::{ensure, Result};
use snarkrs_field::Fr;
use snarkrs_groth16::{prove::prove, PreparedCircuit, Proof, ProveError, StageTimings};

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
    ensure!(
        cpu.backend_name() == "cpu",
        "fallback factory did not return a CPU backend"
    );
    ensure!(
        (cpu.n_vars(), cpu.n_public(), cpu.domain_size())
            == (circuit.n_vars(), circuit.n_public(), circuit.domain_size())
            && (cpu.key().n_vars, cpu.key().n_public, cpu.key().domain_size)
                == (circuit.n_vars(), circuit.n_public(), circuit.domain_size()),
        "CPU fallback dimensions differ from the original circuit"
    );
    let original = &circuit.key().vk;
    let replacement = &cpu.key().vk;
    ensure!(
        replacement.alpha_g1 == original.alpha_g1
            && replacement.beta_g2 == original.beta_g2
            && replacement.gamma_g2 == original.gamma_g2
            && replacement.delta_g2 == original.delta_g2
            && replacement.ic == original.ic
            && cpu.key().alpha_g1 == circuit.key().alpha_g1
            && cpu.key().beta_g1 == circuit.key().beta_g1
            && cpu.key().beta_g2 == circuit.key().beta_g2
            && cpu.key().delta_g1 == circuit.key().delta_g1
            && cpu.key().delta_g2 == circuit.key().delta_g2,
        "CPU fallback verification key differs from the original statement"
    );
    *timings = StageTimings::default();
    let proof = prove(cpu.as_ref(), witness, rng, timings)?;
    report(Fallback::DeviceSuspect { backend });
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use snarkrs_field::{G1Projective, PrimeGroup};
    use snarkrs_formats::{wtns::Witness, ProvingKey};
    use snarkrs_groth16::{cpu::CpuBackend, Backend, HPoly, MsmOutputs};
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
        /// A device fault followed by a deterministic error.
        DeviceThenKey,
    }

    /// The CPU backend, except that its first `faults` MSMs fail as `fault` says.
    struct Faulty {
        inner: Box<dyn PreparedCircuit>,
        fault: Fault,
        faults: AtomicUsize,
        backend: &'static str,
        metadata: Option<ProvingKey>,
        dimensions: Option<(usize, usize, usize)>,
    }

    impl PreparedCircuit for Faulty {
        fn backend_name(&self) -> &'static str {
            self.backend
        }
        fn n_vars(&self) -> usize {
            self.dimensions.map_or(self.inner.n_vars(), |d| d.0)
        }
        fn n_public(&self) -> usize {
            self.dimensions.map_or(self.inner.n_public(), |d| d.1)
        }
        fn domain_size(&self) -> usize {
            self.dimensions.map_or(self.inner.domain_size(), |d| d.2)
        }
        fn key(&self) -> &ProvingKey {
            self.metadata.as_ref().unwrap_or_else(|| self.inner.key())
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
            t.msm_us += 1 << 60;
            match self.fault {
                Fault::Wrong => m.a_g1 += G1Projective::generator(),
                Fault::Device | Fault::DeviceThenKey
                    if left > 1 || matches!(self.fault, Fault::Device) =>
                {
                    return Err(ProveError::Device {
                        backend: "faulty",
                        reason: "msm: command buffer did not complete".to_string(),
                    })
                }
                Fault::Key | Fault::DeviceThenKey | Fault::Device => {
                    return Err(ProveError::Backend {
                        backend: "faulty",
                        reason: "h_query has 3 bases, domain size is 4".to_string(),
                    })
                }
            }
            Ok(m)
        }
    }

    fn tiny() -> PathBuf {
        let dir = std::env::var_os("G16_ARTIFACTS")
            .map(|root| PathBuf::from(root).join("tiny_mul"))
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts/tiny_mul")
            });
        assert!(
            dir.join("circuit.zkey").is_file(),
            "tiny_mul fixture required: {}",
            dir.display()
        );
        dir
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
            backend: "faulty",
            metadata: None,
            dimensions: None,
        };
        let mut steps = Vec::new();
        let mut built = false;
        let mut timings = StageTimings::default();
        let out = prove_with_fallback(
            &circuit,
            witness,
            &mut StdRng::from_seed([3; 32]),
            &mut timings,
            || {
                built = true;
                Ok(cpu(dir))
            },
            &mut |f| steps.push(f),
        );
        if out.is_ok() {
            assert!(timings.msm_us < 1 << 60, "failed attempt timings leaked");
        }
        (out, steps, built)
    }

    fn faulty(dir: &Path, backend: &'static str, faults: usize) -> Faulty {
        Faulty {
            inner: cpu(dir),
            fault: Fault::Device,
            faults: AtomicUsize::new(faults),
            backend,
            metadata: None,
            dimensions: None,
        }
    }

    #[test]
    fn replacement_must_be_cpu_and_bind_the_original_statement() {
        let dir = tiny();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        for case in 0..17 {
            let circuit = faulty(&dir, "faulty", 2);
            let mut replacement = faulty(&dir, "cpu", 0);
            let mut key = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
            match case {
                0 => replacement.backend = "another-gpu",
                1 => replacement.dimensions = Some((key.n_vars + 1, key.n_public, key.domain_size)),
                2 => replacement.dimensions = Some((key.n_vars, key.n_public + 1, key.domain_size)),
                3 => replacement.dimensions = Some((key.n_vars, key.n_public, key.domain_size * 2)),
                4 => key.n_vars += 1,
                5 => key.n_public += 1,
                6 => key.domain_size *= 2,
                7 => key.vk.alpha_g1 = Default::default(),
                8 => key.vk.beta_g2 = Default::default(),
                9 => key.vk.gamma_g2 = Default::default(),
                10 => key.vk.delta_g2 = Default::default(),
                11 => key.vk.ic[0] = Default::default(),
                12 => key.alpha_g1 = Default::default(),
                13 => key.beta_g1 = Default::default(),
                14 => key.beta_g2 = Default::default(),
                15 => key.delta_g1 = Default::default(),
                16 => key.delta_g2 = Default::default(),
                _ => unreachable!(),
            }
            replacement.metadata = Some(key);
            let mut steps = Vec::new();
            let out = prove_with_fallback(
                &circuit,
                &w,
                &mut StdRng::from_seed([3; 32]),
                &mut StageTimings::default(),
                || Ok(Box::new(replacement)),
                &mut |f| steps.push(f),
            );
            let message = out.unwrap_err().to_string();
            assert!(
                message.contains(if case == 0 {
                    "CPU backend"
                } else if case <= 6 {
                    "dimensions"
                } else {
                    "verification key"
                }),
                "case {case}: {message}"
            );
            assert_eq!(steps.len(), 2, "case {case}: {steps:?}");
            assert!(matches!(steps[1], Fallback::Cpu { .. }));
        }
    }

    #[test]
    fn factory_and_cpu_proof_errors_never_report_success_or_retry_cpu() {
        let dir = tiny();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        for case in 0..3 {
            let circuit = faulty(&dir, "faulty", 2);
            let mut steps = Vec::new();
            let out = prove_with_fallback(
                &circuit,
                &w,
                &mut StdRng::from_seed([3; 32]),
                &mut StageTimings::default(),
                || {
                    if case == 0 {
                        anyhow::bail!("factory failed");
                    }
                    let mut replacement = faulty(&dir, "cpu", 1);
                    if case == 2 {
                        replacement.fault = Fault::Wrong;
                    }
                    Ok(Box::new(replacement))
                },
                &mut |f| steps.push(f),
            );
            assert!(out.is_err());
            assert_eq!(steps.len(), 2);
            assert!(matches!(steps[1], Fallback::Cpu { .. }));
        }
        let circuit = faulty(&dir, "cpu", 1);
        let mut steps = Vec::new();
        assert!(prove_with_fallback(
            &circuit,
            &w,
            &mut StdRng::from_seed([3; 32]),
            &mut StageTimings::default(),
            || panic!("CPU must not fall back"),
            &mut |f| steps.push(f),
        )
        .is_err());
        assert!(steps.is_empty());
    }

    fn device() -> Cause {
        Cause::Device(
            "backend faulty: device fault: msm: command buffer did not complete".to_string(),
        )
    }

    #[test]
    fn one_fault_is_absorbed_by_the_retry_and_two_by_the_cpu() {
        let dir = tiny();
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
        let dir = tiny();
        let mut w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let (out, steps, built) = run(Fault::Key, 1, &w, &dir);
        let e = out.unwrap_err();
        assert!(e.to_string().contains("h_query has 3 bases"), "{e}");
        assert!(steps.is_empty(), "{steps:?}");
        assert!(!built);

        let (out, steps, built) = run(Fault::DeviceThenKey, 2, &w, &dir);
        assert!(out.unwrap_err().to_string().contains("h_query has 3 bases"));
        assert_eq!(
            steps,
            [Fallback::Retry {
                backend: "faulty",
                cause: device()
            }]
        );
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
        let dir = tiny();
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
