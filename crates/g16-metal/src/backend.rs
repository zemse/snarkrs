//! Metal implementation of `g16_core::Backend`.
//!
//! This module is only wiring. Stages 0 to 4 live in [`crate::stages`] and stages 5 to 9
//! in [`crate::msm`]; what happens here is the split between witness-independent work
//! (which must all be hoisted into [`MetalBackend::prepare`]) and per-proof work (which
//! must touch nothing shared and mutable).
//!
//! # What `prepare` costs, and where it goes
//!
//! On unified memory there is no PCIe hop, so "upload" here means one `memcpy` into a
//! `StorageModeShared` buffer, at whatever rate a cold page walk allows. The cost that
//! actually dominates is not the copy, it is the two runtime MSL compiles: the field
//! prelude plus the stage kernels, and the field prelude again plus the MSM kernels.
//! Both are paid once per process in [`MetalBackend::new`], never per key and never per
//! proof. Set `G16_METAL_PREPARE=1` to have the breakdown printed to stderr rather than
//! guessed at.
//!
//! # Why there is no CPU fallback path
//!
//! Both kernel sets are complete and both are checked against the CPU backend element by
//! element, so no stage falls back and [`MetalCircuit::backend_name`] can honestly say
//! "metal" for the whole proof. There is deliberately no size gate either: the GPU does
//! lose to the CPU below roughly 3,400 constraints, but a gate that quietly ran the CPU
//! would mean `g16 prove --backend metal` reported a number that belongs to the other
//! backend, which is the exact failure this crate is meant to avoid. The crossover is a
//! measurement to report, not a thing to hide.

use std::sync::Arc;
use std::time::Instant;

use g16_core::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::Fr;
use g16_zkey::ProvingKey;
use metal::Device;

use crate::msm::{G1Bases, G2Bases, Job, JobG1, JobG2, MetalMsm, ScalarBuf};
use crate::stages::{HHandle, HResident, HStages};

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "metal",
        reason: reason.into(),
    }
}

/// Where the time in [`MetalBackend::prepare`] went, in microseconds.
///
/// Kept per circuit rather than printed, so a caller can report it without re-running
/// the preparation to time it.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrepareCost {
    /// [`HStages::prepare`]: both CSR matrices repacked, both twiddle tables, the coset
    /// power table.
    pub stages_us: u64,
    /// The five base vectors repacked from ark's 72/136-byte affines and uploaded.
    pub bases_us: u64,
    pub total_us: u64,
}

/// Device, command queues and every compiled pipeline state, built once per process.
///
/// Cloning a `MetalBackend` is not offered: the compiles are the expensive part and
/// [`Self::prepare`] hands the already-compiled state to the circuit through an `Arc`.
pub struct MetalBackend {
    stages: Arc<HStages>,
    msm: Arc<MetalMsm>,
}

impl MetalBackend {
    /// Picks the system default device and compiles every kernel from source.
    ///
    /// Fails on a machine with no Metal device rather than silently falling back to CPU:
    /// a benchmark that quietly measures the wrong backend is worse than an error.
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default().ok_or_else(|| {
            bad("no Metal device on this machine, so the metal backend cannot run")
        })?;
        Self::with_device(device)
    }

    /// Same, on a caller-supplied device.
    ///
    /// The two kernel modules share this one `MTLDevice`, so every buffer either of them
    /// allocates is usable by the other. That is what lets stage 9 read the buffer stage
    /// 4 wrote instead of round-tripping H through the host.
    ///
    /// They do not yet share one `MTLLibrary`: [`HStages::with_device`] and
    /// [`MetalMsm::with_device`] each compile their own, so the field prelude is
    /// compiled twice. Merging them needs an MSM entry point that accepts a prebuilt
    /// library, which `crate::msm` does not expose.
    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        let stages = HStages::with_device(device.clone())?;
        let msm = MetalMsm::with_device(device)?;
        Ok(Self {
            stages: Arc::new(stages),
            msm: Arc::new(msm),
        })
    }

    pub fn device(&self) -> &Device {
        self.stages.device()
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        Ok(Box::new(MetalCircuit::new(
            self.stages.clone(),
            self.msm.clone(),
            pk,
        )?))
    }
}

/// A key whose witness-independent data is device resident: both CSR matrices, both
/// twiddle tables, the coset power table and all five base vectors.
///
/// # Concurrency
///
/// Nothing here is mutated by a proof. Per-proof scratch is **pooled behind a mutex**,
/// in two separate pools that each kernel module owns: `crate::stages`' pool hands out
/// the witness, A, B, C, transform temporary and the two H buffers, and takes them back
/// when the [`HHandle`] inside the [`HPoly`] is dropped; `crate::msm`'s pool hands out
/// the counting-sort and bucket scratch and takes it back after `wait_until_completed`.
/// Neither pool is reachable from this struct, so two concurrent `prove` calls against
/// one `MetalCircuit` cannot be handed the same buffer. Command buffers and encoders,
/// the two Metal objects that are not thread safe, are created and destroyed inside a
/// single call and never stored.
pub struct MetalCircuit {
    pk: ProvingKey,
    stages: Arc<HStages>,
    msm: Arc<MetalMsm>,
    resident: HResident,
    a_bases: G1Bases,
    b_g1_bases: G1Bases,
    b_g2_bases: G2Bases,
    l_bases: G1Bases,
    h_bases: G1Bases,
    cost: PrepareCost,
}

impl MetalCircuit {
    fn new(stages: Arc<HStages>, msm: Arc<MetalMsm>, pk: ProvingKey) -> Result<Self, ProveError> {
        // Shape checks first. `crate::msm` would reject an out-of-range job later, but by
        // then the message names buffer offsets rather than the section of the zkey that
        // is the wrong length, and the mismatch is a property of the key, not the proof.
        let n_vars = pk.n_vars;
        for (name, got) in [
            ("a_query", pk.a_query.len()),
            ("b_g1_query", pk.b_g1_query.len()),
            ("b_g2_query", pk.b_g2_query.len()),
        ] {
            if got != n_vars {
                return Err(bad(format!("{name} has {got} bases, n_vars is {n_vars}")));
            }
        }
        if pk.h_query.len() != pk.domain_size {
            return Err(bad(format!(
                "h_query has {} bases, domain size is {}",
                pk.h_query.len(),
                pk.domain_size
            )));
        }
        let want_l = n_vars
            .checked_sub(pk.n_public + 1)
            .ok_or_else(|| bad(format!("n_public {} exceeds n_vars {n_vars}", pk.n_public)))?;
        if pk.l_query.len() != want_l {
            return Err(bad(format!(
                "l_query has {} bases, private witness is {want_l} long",
                pk.l_query.len()
            )));
        }

        let t_all = Instant::now();
        // Stages 0 to 4. This also re-runs the CPU backend's key validation (power-of-two
        // domain, CSR row count, every signal index below n_vars), so a bad key becomes
        // an error here instead of an out-of-bounds GPU read later.
        let t0 = Instant::now();
        let resident = stages.prepare(&pk)?;
        let stages_us = t0.elapsed().as_micros() as u64;

        // Stages 5 to 9. Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this
        // arkworks and `G2Affine` is 136, neither is `repr(C)`, and both carry an
        // infinity flag that the packed layout encodes as all-zero coordinates instead.
        let t1 = Instant::now();
        let a_bases = msm.upload_g1_bases(&pk.a_query);
        let b_g1_bases = msm.upload_g1_bases(&pk.b_g1_query);
        let b_g2_bases = msm.upload_g2_bases(&pk.b_g2_query);
        let l_bases = msm.upload_g1_bases(&pk.l_query);
        let h_bases = msm.upload_g1_bases(&pk.h_query);
        let bases_us = t1.elapsed().as_micros() as u64;

        let cost = PrepareCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        if std::env::var_os("G16_METAL_PREPARE").is_some() {
            eprintln!(
                "metal prepare: stages {} us, bases {} us, total {} us (domain {}, n_vars {n_vars})",
                cost.stages_us, cost.bases_us, cost.total_us, pk.domain_size,
            );
        }

        Ok(Self {
            pk,
            stages,
            msm,
            resident,
            a_bases,
            b_g1_bases,
            b_g2_bases,
            l_bases,
            h_bases,
            cost,
        })
    }

    /// What [`MetalBackend::prepare`] cost for this key.
    pub fn prepare_cost(&self) -> PrepareCost {
        self.cost
    }

    fn check_witness(&self, witness: &[Fr]) -> Result<(), ProveError> {
        if witness.len() != self.pk.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.pk.n_vars,
            });
        }
        Ok(())
    }

    /// Stages 5-8: the four MSM jobs whose scalars are the witness. One upload of the
    /// witness serves all four. Passing the same `ScalarBuf` over the same range is what
    /// makes the first three share a single counting sort inside `msm_batch`; a second
    /// upload would silently cost two more digit pipelines as well as the copy.
    fn witness_jobs<'a>(&'a self, w: &'a ScalarBuf) -> [Job<'a>; 4] {
        [
            Job::G1(JobG1 {
                bases: &self.a_bases,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: self.pk.n_vars,
            }),
            Job::G2(JobG2 {
                bases: &self.b_g2_bases,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: self.pk.n_vars,
            }),
            Job::G1(JobG1 {
                bases: &self.b_g1_bases,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: self.pk.n_vars,
            }),
            Job::G1(JobG1 {
                bases: &self.l_bases,
                base_off: 0,
                scalars: w,
                // Section 8 covers the private wires only: witness[0] is the constant 1
                // and witness[1..=n_public] are the public inputs, which the verifier
                // folds in through IC instead.
                scalar_off: self.pk.n_public + 1,
                n: self.l_bases.len(),
            }),
        ]
    }

    /// Stage 9's scalars. The normal path is the device buffer stage 4 wrote, which
    /// never touches the host. `HPoly::device_handle` returns `None` for a handle from
    /// some other backend, and in that case the only thing left is a host vector, so
    /// a CPU `compute_h` can still be finished on the GPU rather than rejected.
    ///
    /// The returned `ScalarBuf` reads the buffer `h` owns, so `h` (and with it the
    /// scratch behind the handle) must outlive the `msm_batch` that consumes it; that
    /// is the lifetime `scalars_from_device_std` requires.
    fn h_scalars(&self, h: &HPoly) -> Result<ScalarBuf, ProveError> {
        match h.device_handle::<HHandle>(crate::stages::TAG) {
            Some(handle) => {
                if handle.len() != self.pk.domain_size {
                    return Err(bad(format!(
                        "device h holds {} entries, domain size is {}",
                        handle.len(),
                        self.pk.domain_size
                    )));
                }
                // Stage 4 wrote H in both Montgomery and standard form; the MSM reads
                // the standard copy directly. Re-deriving it from `h_mont` through
                // `scalars_from_device_mont` was the old path here, and it cost one
                // extra command buffer plus a full-domain Montgomery reduction that the
                // pointwise kernel had already performed.
                Ok(self
                    .msm
                    .scalars_from_device_std(handle.h_std(), handle.len()))
            }
            None => {
                let host = h.to_host().ok_or_else(|| {
                    bad("compute_h output is neither a metal handle nor a host vector")
                })?;
                Ok(self.msm.upload_scalars(host))
            }
        }
    }

    /// Stage 9's MSM job over `h_scalars`.
    fn h_job<'a>(&'a self, h_scalars: &'a ScalarBuf) -> Job<'a> {
        Job::G1(JobG1 {
            bases: &self.h_bases,
            base_off: 0,
            scalars: h_scalars,
            scalar_off: 0,
            n: self.pk.domain_size,
        })
    }
}

impl PreparedCircuit for MetalCircuit {
    /// Always "metal", and that is a claim about what ran: no stage in this backend has a
    /// CPU fallback, so the name cannot be describing a proof the CPU did.
    fn backend_name(&self) -> &'static str {
        "metal"
    }
    fn n_vars(&self) -> usize {
        self.pk.n_vars
    }
    fn n_public(&self) -> usize {
        self.pk.n_public
    }
    fn domain_size(&self) -> usize {
        self.pk.domain_size
    }
    fn key(&self) -> &ProvingKey {
        &self.pk
    }

    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        self.resident.compute_h(&self.stages, witness, t)
    }

    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        if h.len() != self.pk.domain_size {
            return Err(bad(format!(
                "h has {} entries, domain size is {}",
                h.len(),
                self.pk.domain_size
            )));
        }

        let start = Instant::now();

        let w = self.msm.upload_scalars(witness);
        let h_scalars = self.h_scalars(h)?;

        // One command buffer for all five. Order matches the stage numbering, and
        // `msm_batch` returns results in job order.
        let [j5, j6, j7, j8] = self.witness_jobs(&w);
        let jobs = [j5, j6, j7, j8, self.h_job(&h_scalars)];
        let out = self.msm.msm_batch(&jobs)?;
        if out.len() != jobs.len() {
            return Err(bad(format!(
                "msm_batch returned {} results for {} jobs",
                out.len(),
                jobs.len()
            )));
        }

        let a_g1 = out[0].g1()?;
        let b_g2 = out[1].g2()?;
        let b_g1 = out[2].g1()?;
        let l_g1 = out[3].g1()?;
        let h_g1 = out[4].g1()?;
        t.msm_us += start.elapsed().as_micros() as u64;

        Ok(MsmOutputs {
            a_g1,
            b_g2,
            b_g1,
            l_g1,
            h_g1,
        })
    }

    /// Stages 0-9, with stages 5-8 started before `H` exists.
    ///
    /// Only stage 9 reads the buffer stage 4 writes; the other four MSMs read the
    /// witness, which is in hand before `compute_h` starts. [`HStages`] and [`MetalMsm`]
    /// own separate command queues, so the witness batch is committed from a second
    /// thread while the compute_h command buffers run, and the device interleaves the
    /// two. The H MSM then goes out as its own batch once `compute_h` has returned.
    ///
    /// Splitting the five-job batch is not free: the witness jobs lose their seat in the
    /// concurrent encoder beside H's accumulation, and a second submission is paid.
    /// Measured against those costs the overlap still wins about 1.2 ms at 2^16 and
    /// 1.1 ms at 2^17 (csp warm medians and minima alike), because the four witness
    /// MSMs' marginal GPU time fits inside the 2.5 to 3.9 ms the transforms take. The
    /// CPU backend keeps the sequential default: there the same overlap measured even,
    /// since work stealing already absorbs the witness MSMs either way.
    fn h_and_msms(&self, witness: &[Fr], t: &mut StageTimings) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        let start = Instant::now();

        let mut compute_h_us = 0u64;
        let (h_out, wit_out) = std::thread::scope(|s| {
            let wit = s.spawn(|| {
                let w = self.msm.upload_scalars(witness);
                self.msm.msm_batch(&self.witness_jobs(&w))
            });
            let h_out = (|| {
                let t0 = Instant::now();
                let h = self.compute_h(witness, t)?;
                compute_h_us = t0.elapsed().as_micros() as u64;
                let h_scalars = self.h_scalars(&h)?;
                // `h` stays alive across the batch: the scratch behind its handle owns
                // the buffer `h_scalars` reads.
                self.msm.msm_batch(&[self.h_job(&h_scalars)])
            })();
            (h_out, wit.join())
        });
        let wit_out = wit_out.unwrap_or_else(|e| std::panic::resume_unwind(e))?;
        let h_out = h_out?;
        if wit_out.len() != 4 || h_out.len() != 1 {
            return Err(bad(format!(
                "msm_batch returned {} witness results and {} h results",
                wit_out.len(),
                h_out.len()
            )));
        }

        let a_g1 = wit_out[0].g1()?;
        let b_g2 = wit_out[1].g2()?;
        let b_g1 = wit_out[2].g1()?;
        let l_g1 = wit_out[3].g1()?;
        let h_g1 = h_out[0].g1()?;
        // The witness batch overlaps stages 0-4 here, so the MSMs no longer have a wall
        // window of their own. `msm_us` takes what they add beyond `compute_h`, which
        // keeps the stage fields summing to the proof instead of double-counting the
        // overlap. `compute_h` filled `gather_us` and `ntt_us` itself, above.
        t.msm_us += (start.elapsed().as_micros() as u64).saturating_sub(compute_h_us);

        Ok(MsmOutputs {
            a_g1,
            b_g2,
            b_g1,
            l_g1,
            h_g1,
        })
    }
}

/// `Backend` and `PreparedCircuit` both require `Send + Sync`, and the whole reason
/// `prepare` is a separate call is that one resident key is proved against from many
/// threads. If a `!Sync` field ever lands on either struct this stops compiling here
/// rather than at the `Box<dyn PreparedCircuit>` coercion, where the error names the
/// trait object instead of the field.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MetalBackend>();
    assert_send_sync::<MetalCircuit>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use g16_core::prove::prove_with_blinders;
    use g16_core::verify::verify;
    use g16_zkey::{wtns::Witness, VerifyingKey};
    use std::path::{Path, PathBuf};

    fn artifacts() -> Vec<(String, PathBuf)> {
        let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts")
            .canonicalize()
        else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(&root) else {
            return Vec::new();
        };
        let mut out: Vec<(String, PathBuf)> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|d| {
                ["circuit.zkey", "circuit.wtns", "vkey.json"]
                    .iter()
                    .all(|f| d.join(f).is_file())
            })
            .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
            .collect();
        out.sort();
        out
    }

    /// An empty artifact directory reports a skip on stderr rather than passing
    /// silently, so a missing fixture cannot be mistaken for a green backend.
    fn for_each(test: &str, f: impl Fn(&str, &Path)) {
        let found = artifacts();
        if found.is_empty() {
            eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
            return;
        }
        for (name, dir) in &found {
            eprintln!("{test}: {name}");
            f(name, dir);
        }
    }

    /// The public signals are the witness prefix, which is what snarkjs publishes. Taken
    /// from the witness rather than parsed out of `public.json` so this crate needs no
    /// JSON dependency to test the whole pipeline.
    fn load(dir: &Path) -> (Box<dyn PreparedCircuit>, Vec<Fr>, VerifyingKey) {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let circuit = MetalBackend::new().unwrap().prepare(pk).unwrap();
        (circuit, witness, vk)
    }

    /// The test this backend exists to pass: every stage on the GPU, our own verifier as
    /// the oracle. Blinders are fixed rather than random so a failure is reproducible;
    /// the randomised path is exercised through the CLI.
    #[test]
    fn a_metal_proof_verifies_on_every_artifact() {
        for_each("a_metal_proof_verifies_on_every_artifact", |name, dir| {
            let (circuit, witness, vk) = load(dir);
            assert_eq!(circuit.backend_name(), "metal", "{name}");
            let public = witness[1..=circuit.n_public()].to_vec();
            let mut t = StageTimings::default();
            let proof = prove_with_blinders(
                circuit.as_ref(),
                &witness,
                Fr::from(31337u64),
                Fr::from(4242u64),
                &mut t,
            )
            .unwrap();
            verify(&vk, &public, &proof).unwrap_or_else(|e| panic!("{name}: {e}"));
            // A backend that reported nothing would make `bench` print a free prover.
            assert!(t.msm_us > 0, "{name}: msms reported no time");
        });
    }

    /// Zero blinders remove the masking, so the H evaluations are the only thing between
    /// the MSMs and the pairing check. This is the case that fails loudly if stage 4's
    /// buffer, the Montgomery convention or the coset were wrong.
    #[test]
    fn zero_blinders_still_verify() {
        for_each("zero_blinders_still_verify", |name, dir| {
            let (circuit, witness, vk) = load(dir);
            let public = witness[1..=circuit.n_public()].to_vec();
            let mut t = StageTimings::default();
            let proof = prove_with_blinders(
                circuit.as_ref(),
                &witness,
                Fr::from(0u64),
                Fr::from(0u64),
                &mut t,
            )
            .unwrap();
            verify(&vk, &public, &proof).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// `PreparedCircuit` promises one instance is safe to prove with from several
    /// threads, and that promise is the whole reason the per-proof scratch in both kernel
    /// modules is pooled behind a mutex rather than stored on the circuit. A circuit that
    /// handed two in-flight proofs the same H buffer would not crash: it would produce a
    /// proof that simply fails to verify, which is why the check is a verification and
    /// not just a join.
    #[test]
    fn one_metal_circuit_proves_concurrently() {
        for_each("one_metal_circuit_proves_concurrently", |name, dir| {
            let (circuit, witness, vk) = load(dir);
            let public = witness[1..=circuit.n_public()].to_vec();
            let circuit = circuit.as_ref();
            let (witness, vk, public) = (&witness, &vk, &public);

            std::thread::scope(|scope| {
                let threads: Vec<_> = (1..=4u64)
                    .map(|i| {
                        scope.spawn(move || {
                            let mut t = StageTimings::default();
                            let proof = prove_with_blinders(
                                circuit,
                                witness,
                                Fr::from(i * 7),
                                Fr::from(i * 11),
                                &mut t,
                            )
                            .unwrap();
                            verify(vk, public, &proof).unwrap_or_else(|e| panic!("{name}: {e}"));
                        })
                    })
                    .collect();
                for t in threads {
                    t.join().unwrap();
                }
            });
        });
    }

    #[test]
    fn a_short_witness_is_rejected_before_any_dispatch() {
        for_each("a_short_witness_is_rejected", |name, dir| {
            let (circuit, witness, _) = load(dir);
            let mut t = StageTimings::default();
            let short = &witness[..witness.len() - 1];
            assert!(
                matches!(
                    circuit.compute_h(short, &mut t),
                    Err(ProveError::WitnessLength { .. })
                ),
                "{name}: compute_h accepted a short witness"
            );
            let h = circuit.compute_h(&witness, &mut t).unwrap();
            assert!(
                matches!(
                    circuit.msms(short, &h, &mut t),
                    Err(ProveError::WitnessLength { .. })
                ),
                "{name}: msms accepted a short witness"
            );
            // And the circuit still works afterwards, so the rejection did not leave a
            // buffer checked out of the pool.
            circuit.msms(&witness, &h, &mut t).unwrap();
        });
    }

    /// What `prepare` actually costs and where it goes, printed rather than asserted:
    /// the numbers are hardware, and a threshold here would be a flaky test.
    ///
    /// Run with
    /// `cargo test -p g16-metal --release --lib measure_prepare -- --ignored --nocapture`.
    ///
    /// The two `MetalBackend::new` calls are the point of the first half. The first one
    /// pays the runtime MSL compile; the second compiles the identical source again in
    /// the same process, and the gap between them is the OS shader cache, not our code.
    /// That is why a benchmark that constructs the backend per proof can look almost
    /// free after the first rep and still be measuring the Metal compiler on rep one.
    #[test]
    #[ignore = "measurement, not a check"]
    fn measure_prepare() {
        let t = Instant::now();
        let backend = MetalBackend::new().unwrap();
        println!("MetalBackend::new  first  {:>8.2} ms", ms(t));
        let t = Instant::now();
        let _second = MetalBackend::new().unwrap();
        println!("MetalBackend::new  second {:>8.2} ms", ms(t));

        println!(
            "{:<16} {:>10} {:>10} {:>10} {:>10}",
            "artifact", "zkey ms", "stages ms", "bases ms", "prepare ms"
        );
        for (name, dir) in artifacts() {
            let t = Instant::now();
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
            let zkey_ms = ms(t);
            let t = Instant::now();
            let circuit =
                MetalCircuit::new(backend.stages.clone(), backend.msm.clone(), pk).unwrap();
            let total = ms(t);
            let c = circuit.prepare_cost();
            println!(
                "{name:<16} {zkey_ms:>10.2} {:>10.2} {:>10.2} {total:>10.2}",
                c.stages_us as f64 / 1000.0,
                c.bases_us as f64 / 1000.0,
            );
        }
    }

    fn ms(t: Instant) -> f64 {
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// A device handle carrying someone else's tag must not be dereferenced as an
    /// `HHandle`. Since there is no host copy behind it either, the only correct answer
    /// is an error.
    #[test]
    fn an_h_from_another_backend_is_refused() {
        for_each("an_h_from_another_backend_is_refused", |name, dir| {
            let (circuit, witness, _) = load(dir);
            let mut t = StageTimings::default();
            let foreign = HPoly::Device {
                tag: "cuda",
                len: circuit.domain_size(),
                data: std::sync::Arc::new(0u32),
            };
            assert!(
                matches!(
                    circuit.msms(&witness, &foreign, &mut t),
                    Err(ProveError::Backend { .. })
                ),
                "{name}: msms accepted a foreign device handle"
            );
            // A host vector of the wrong length is a different error and must not be
            // conflated with it.
            assert!(matches!(
                circuit.msms(&witness, &HPoly::Host(vec![Fr::from(1u64); 3]), &mut t),
                Err(ProveError::Backend { .. })
            ));
        });
    }
}
