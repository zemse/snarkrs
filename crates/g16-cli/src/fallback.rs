//! What `g16 prove` does when an accelerator hands back a proof that does not verify.
//!
//! A logic bug fails the same way on every run and dies in CI. A GPU that returns a wrong
//! proof in production is far more likely to be transient: the DSN 2026 silent data
//! corruption study (arXiv 2605.04213) found the corruption warp aligned and structured, which
//! is exactly the shape that gives a wrong bucket sum and a failing pairing rather than a
//! crash, and `icicle-snark#18` is a CUDA prover producing three different invalid proofs in
//! three runs. So a failed self-verify on a GPU is retried once there, then proved on the CPU,
//! and each step is reported so the failure *rate* is what an operator alarms on.
//!
//! Only [`ProveError::SelfVerify`] is retried. Every other error is deterministic, and a
//! witness that does not satisfy the circuit fails on the CPU too, which is how the last
//! message tells the two apart.

use anyhow::Result;
use g16_core::{prove::prove, PreparedCircuit, Proof, ProveError, StageTimings};
use g16_field::Fr;

/// A step [`prove_with_fallback`] took, for the caller to report.
#[derive(Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The first proof on `backend` failed its self-verify; proving again there.
    Retry { backend: &'static str },
    /// The second one failed too; proving on the CPU.
    Cpu { backend: &'static str },
    /// The CPU proof verified, so `backend` computed a wrong proof twice.
    DeviceSuspect { backend: &'static str },
}

/// [`prove`], retried once on the same backend and then on the CPU when the proof fails its
/// self-verify. `cpu` builds the CPU circuit only if it is needed: the key has usually moved
/// into the accelerator at prepare, so the caller reloads it. `timings` describe the attempt
/// that produced the proof.
pub fn prove_with_fallback<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    rng: &mut R,
    timings: &mut StageTimings,
    cpu: impl FnOnce() -> Result<Box<dyn PreparedCircuit>>,
    report: &mut dyn FnMut(Fallback),
) -> Result<Proof> {
    let backend = circuit.backend_name();
    let mut attempt = prove(circuit, witness, rng, timings);
    if backend == "cpu" || !matches!(attempt, Err(ProveError::SelfVerify(_))) {
        return Ok(attempt?);
    }

    report(Fallback::Retry { backend });
    *timings = StageTimings::default();
    attempt = prove(circuit, witness, rng, timings);
    if !matches!(attempt, Err(ProveError::SelfVerify(_))) {
        return Ok(attempt?);
    }

    report(Fallback::Cpu { backend });
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

    /// The CPU backend, except that its first `faults` MSM outputs come back with `A` off by
    /// one generator, which is what a dropped kernel error or a flipped bucket looks like from
    /// outside: a well-formed proof that fails the pairing.
    struct Faulty {
        inner: Box<dyn PreparedCircuit>,
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
            if left > 0 {
                self.faults.store(left - 1, Ordering::SeqCst);
                m.a_g1 += G1Projective::generator();
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

    fn run(faults: usize, witness: &[Fr], dir: &Path) -> (Result<Proof>, Vec<Fallback>) {
        let circuit = Faulty {
            inner: cpu(dir),
            faults: AtomicUsize::new(faults),
        };
        let mut steps = Vec::new();
        let out = prove_with_fallback(
            &circuit,
            witness,
            &mut StdRng::from_seed([3; 32]),
            &mut StageTimings::default(),
            || Ok(cpu(dir)),
            &mut |f| steps.push(f),
        );
        (out, steps)
    }

    #[test]
    fn one_fault_is_absorbed_by_the_retry_and_two_by_the_cpu() {
        let Some(dir) = tiny() else {
            eprintln!("SKIPPED fallback: no tiny_mul");
            return;
        };
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let backend = "faulty";

        let (out, steps) = run(0, &w, &dir);
        assert!(out.is_ok());
        assert!(steps.is_empty());

        let (out, steps) = run(1, &w, &dir);
        assert!(out.is_ok());
        assert_eq!(steps, [Fallback::Retry { backend }]);

        let (out, steps) = run(2, &w, &dir);
        assert!(out.is_ok());
        assert_eq!(
            steps,
            [
                Fallback::Retry { backend },
                Fallback::Cpu { backend },
                Fallback::DeviceSuspect { backend }
            ]
        );
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
        let (out, steps) = run(0, &w, &dir);
        assert!(out.is_err());
        assert!(!steps
            .iter()
            .any(|s| matches!(s, Fallback::DeviceSuspect { .. })));
    }
}
