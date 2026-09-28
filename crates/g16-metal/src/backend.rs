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
//! would mean `snarkrs groth16 prove --backend metal` reported a number that belongs to the
//! other backend, which is the exact failure this crate is meant to avoid. The crossover is
//! a measurement to report, not a thing to hide.

use std::sync::Arc;
use std::time::Instant;

use g16_core::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::Fr;
use g16_zkey::{Coefficients, ProvingKey};
use metal::Device;

use crate::msm::{G1Bases, G2Bases, Job, JobG1, JobG2, MetalMsm, ScalarBuf, Work};
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
    work: Work,
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

    /// [`Self::new`] with MSMs whose cost follows the key and not the witness, so proving
    /// time, buffer sizes and dispatch geometry do not reveal how many witness entries
    /// are zero or one; see [`Work::Constant`].
    ///
    /// What stays witness dependent, on the device: which bucket a digit lands in, so
    /// the atomics' contention in the digit pipeline, the run count of the accumulation
    /// and of the fold (which points each thread stores, not how many additions it
    /// makes), and the identity shortcuts in the accumulation's mixed additions. The
    /// merge is a fixed tree (`msm_fold_*`) and it and the reduce add with the complete
    /// formulas, so neither moves with the bucket contents: at 2^18 a witness of zeros,
    /// one of bits and a dense one read 3.15, 3.16 and 3.17 ms for the fold in G1 and
    /// 13.0, 13.0 and 13.3 in G2, and 1.81, 1.84 and 1.84 for the reduce in G1, 8.26,
    /// 8.23 and 8.29 in G2, where the shortcut versions read 0.77, 0.96 and 1.03 (fold,
    /// G1), 2.60, 3.78 and 3.86 (G2), 0.25, 0.25 and 1.47 (reduce, G1) and 0.83, 0.82
    /// and 9.78 (G2). On the host: the identity tests inside the combine's few hundred
    /// curve additions, as on the CPU backend. Stages 0 to 4 were already fixed-shape;
    /// their one value test, the gather's skip of the multiply for a 0 or 1 witness
    /// value, is off (keccak256's gather 0.45 ms to 0.71, what a dense witness costs;
    /// js_16x16_d32 unchanged at 1.56).
    ///
    /// Priced warm on the M2 Max, 15 reps, alternating rounds under the GPU lock,
    /// medians against the variable path: js_16x16_d32 115 ms against 91 (+26%),
    /// keccak256 156 against 27 (5.7x), rsa2048 130 against 35 (3.7x). The complete
    /// formulas are 19%, 11% and 11% of that over the shortcut fold and reduce. The
    /// bit-heavy circuits pay more than on the CPU because the variable witness plans
    /// there were a few hundred scalars over a few windows, and each is now an MSM the
    /// size of H's, one of them in G2.
    pub fn constant_work() -> Result<Self, ProveError> {
        let mut backend = Self::new()?;
        backend.work = Work::Constant;
        Ok(backend)
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
            work: Work::Variable,
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
            self.work,
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
///
/// # What stays on the host
///
/// Only the header and the verifying key. The five query sections and the coefficients
/// are emptied as each one is uploaded, so [`PreparedCircuit::key`] returns a key whose
/// vectors are empty. Nothing on the host reads them after `prepare`.
///
/// They go through [`crate::alloc::release`] rather than a plain drop, because macOS
/// malloc keeps freed large blocks resident in its large cache, and on js_384x384_d32
/// that held 2.4 GB of the 9.1 GB peak.
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
    work: Work,
}

impl MetalCircuit {
    fn new(
        stages: Arc<HStages>,
        msm: Arc<MetalMsm>,
        mut pk: ProvingKey,
        work: Work,
    ) -> Result<Self, ProveError> {
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
        let Coefficients {
            row_ptr,
            signal,
            value,
        } = std::mem::replace(
            &mut pk.coeffs,
            Coefficients {
                row_ptr: Default::default(),
                signal: Default::default(),
                value: Default::default(),
            },
        );
        row_ptr
            .into_iter()
            .chain(signal)
            .for_each(crate::alloc::release);
        value.into_iter().for_each(crate::alloc::release);
        let stages_us = t0.elapsed().as_micros() as u64;

        // Stages 5 to 9. Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this
        // arkworks and `G2Affine` is 136, neither is `repr(C)`, and both carry an
        // infinity flag that the packed layout encodes as all-zero coordinates instead.
        // Each host vector is taken and released as soon as it is uploaded.
        let t1 = Instant::now();
        let g1 = |v: Vec<_>| {
            let bases = msm.upload_g1_bases(&v);
            crate::alloc::release(v);
            bases
        };
        let a_bases = g1(std::mem::take(&mut pk.a_query))?;
        let b_g1_bases = g1(std::mem::take(&mut pk.b_g1_query))?;
        let b_g2_bases = {
            let v = std::mem::take(&mut pk.b_g2_query);
            let bases = msm.upload_g2_bases(&v);
            crate::alloc::release(v);
            bases?
        };
        let l_bases = g1(std::mem::take(&mut pk.l_query))?;
        let h_bases = g1(std::mem::take(&mut pk.h_query))?;
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
            work,
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
    /// witness serves all four. Passing the same `ScalarBuf`, over the same range or a
    /// short suffix of it, is what makes all four share a single counting sort inside
    /// `msm_batch`; a second upload would silently cost more digit pipelines as well as
    /// the copy.
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
                // Stage 4 wrote H in standard form, which the MSM reads directly.
                // Converting a Montgomery copy through `scalars_from_device_mont` was
                // the old path here, and it cost one extra command buffer plus a
                // full-domain Montgomery reduction that the pointwise kernel had
                // already performed.
                //
                // The plan's digit entries go into the domain vectors stage 4 is done
                // with, rather than a buffer of their own from the MSM pool.
                Ok(self
                    .msm
                    .scalars_from_device_std_with(handle.h_std(), handle.len(), self.work)
                    .with_lent(handle.lend_entries()))
            }
            None => {
                let host = h.to_host().ok_or_else(|| {
                    bad("compute_h output is neither a metal handle nor a host vector")
                })?;
                self.msm.upload_scalars_with(host, self.work)
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

/// The smallest domain where [`MetalCircuit::h_and_msms`] splits the five-job batch to
/// overlap the witness MSMs with the transforms. See [`overlap_pays`].
const OVERLAP_MIN_DOMAIN: usize = 1 << 17;

/// Whether the two-queue overlap in [`MetalCircuit::h_and_msms`] pays at this size.
///
/// The split has no ordering control: the witness batch and the compute_h buffers sit on
/// different queues, and whether the device hides one inside the other or serialises
/// them is its arbitration, not ours. Both outcomes are real, and which one a session
/// gets is machine state that can flip between rounds and then stick. At 2^16 the same
/// binary measured csp minima of 12.0 to 12.3 ms for two interleaved rounds and then
/// 13.0 to 16.6 for the next six, against an unsplit base flat at 12.5 to 12.7; a
/// full-ladder sweep an hour later, forced both ways, read 10.6 against 12.0 in every
/// round, and an independent session reproduced the losing mode at 15.1 to 16.9 against
/// a settled 12.3. A schedule that wins 1.4 ms in one arbitration state and loses 3 in
/// the other is priced above its best case, so below the crossover the sequential path
/// stands. At 2^17 and up the witness work is large enough that even the lost race
/// nets, and no observed round in any session has lost there: the ladder sweep's minima
/// read -1.0 ms at 2^17 (js_8x8_d32), -1.1 (sha256_256), -1.9 (keccak256), -3.1
/// (rsa2048) and -6.9 (anon-aadhaar, 2^21), with the eight-round interleaved csp
/// campaign and the independent session agreeing on the sign. Sweep it again with
/// `G16_METAL_OVERLAP=0` or `=1`, which forces the path regardless of size.
fn overlap_pays(domain_size: usize) -> bool {
    match std::env::var("G16_METAL_OVERLAP")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
    {
        Some(0) => false,
        Some(_) => true,
        None => domain_size >= OVERLAP_MIN_DOMAIN,
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
        self.resident
            .compute_h_with(&self.stages, witness, self.work, t)
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

        let w = self.msm.upload_scalars_with(witness, self.work)?;
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

    /// Without this the trait default reports "no H" for every device handle, so
    /// `snarkrs trace` on this backend would print the H rows empty. Debugging seam only:
    /// it costs a full-domain readback, which is the round trip [`HPoly`] exists to avoid.
    fn h_to_host(&self, h: &HPoly) -> Option<Vec<Fr>> {
        match h.device_handle::<HHandle>(crate::stages::TAG) {
            Some(handle) => handle.to_host(),
            None => h.to_host().map(<[Fr]>::to_vec),
        }
    }

    /// Stages 0-9, with stages 5-8 started before `H` exists.
    ///
    /// Only stage 9 reads the buffer stage 4 writes; the other four MSMs read the
    /// witness, which is in hand before `compute_h` starts. [`HStages`] and [`MetalMsm`]
    /// own separate command queues, so the witness batch is committed from a second
    /// thread while the compute_h command buffers run, and the device interleaves the
    /// two. The H MSM then goes out as its own batch once `compute_h` has returned.
    ///
    /// Splitting the five-job batch is not free, and worse, its price is not fixed: the
    /// witness jobs lose their seat in the concurrent encoder beside H's accumulation,
    /// a second submission is paid, and the device arbitrates the two queues however it
    /// likes. See [`overlap_pays`] for the measured consequences and for why the split
    /// only happens at [`OVERLAP_MIN_DOMAIN`] and above; below it this method takes the
    /// trait's own sequence, bit for bit. The CPU backend keeps the sequential default
    /// everywhere: there the same overlap measured even, since work stealing already
    /// absorbs the witness MSMs either way.
    fn h_and_msms(&self, witness: &[Fr], t: &mut StageTimings) -> Result<MsmOutputs, ProveError> {
        if !overlap_pays(self.pk.domain_size) {
            let h = self.compute_h(witness, t)?;
            return self.msms(witness, &h, t);
        }
        self.check_witness(witness)?;
        let start = Instant::now();

        let mut compute_h_us = 0u64;
        let (h_out, wit_out) = std::thread::scope(|s| {
            let wit = s.spawn(|| {
                let w = self.msm.upload_scalars_with(witness, self.work)?;
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
        load_on(dir, MetalBackend::new().unwrap())
    }

    fn load_on(
        dir: &Path,
        backend: MetalBackend,
    ) -> (Box<dyn PreparedCircuit>, Vec<Fr>, VerifyingKey) {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let circuit = backend.prepare(pk).unwrap();
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

    /// The constant-work backend proves every artifact, and at pinned blinders its proof
    /// is the variable one bit for bit: the two paths compute the same five points by
    /// different routes, so any drift between them is a wrong MSM, not a different proof.
    #[test]
    fn a_constant_work_metal_proof_is_the_variable_one() {
        for_each(
            "a_constant_work_metal_proof_is_the_variable_one",
            |name, dir| {
                let (constant, witness, vk) = load_on(dir, MetalBackend::constant_work().unwrap());
                let (variable, _, _) = load(dir);
                let public = witness[1..=constant.n_public()].to_vec();
                let mut t = StageTimings::default();
                let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
                let proof = prove_with_blinders(constant.as_ref(), &witness, r, s, &mut t).unwrap();
                verify(&vk, &public, &proof).unwrap_or_else(|e| panic!("{name}: {e}"));
                let want = prove_with_blinders(variable.as_ref(), &witness, r, s, &mut t).unwrap();
                assert_eq!(proof.a, want.a, "{name}");
                assert_eq!(proof.b, want.b, "{name}");
                assert_eq!(proof.c, want.c, "{name}");
            },
        );
    }

    /// `g16_core::trace` is the only caller and it takes the trait default unless a
    /// backend overrides it; the default cannot see through a device handle, so without
    /// the override every Metal trace recorded "no H".
    #[test]
    fn h_to_host_reads_the_device_handle_back() {
        for_each("h_to_host_reads_the_device_handle_back", |name, dir| {
            let (circuit, witness, _) = load(dir);
            let mut t = StageTimings::default();
            let h = circuit.compute_h(&witness, &mut t).unwrap();
            assert!(
                h.to_host().is_none(),
                "{name}: H did not stay on the device"
            );
            let host = circuit
                .h_to_host(&h)
                .unwrap_or_else(|| panic!("{name}: no H"));
            assert_eq!(host.len(), circuit.domain_size(), "{name}");
            let cpu = g16_core::cpu::CpuBackend::new()
                .prepare(ProvingKey::load(&dir.join("circuit.zkey")).unwrap())
                .unwrap();
            let want = cpu.compute_h(&witness, &mut t).unwrap();
            assert_eq!(host.as_slice(), want.to_host().expect("cpu h"), "{name}");
        });
    }

    /// The domain vectors go to one MSM at a time. Two MSMs over the same H at once would
    /// scatter their digit entries into one buffer, so the second has to get `None` and
    /// take its entries from the pool.
    #[test]
    fn h_lends_its_domain_vectors_to_one_msm_at_a_time() {
        for_each(
            "h_lends_its_domain_vectors_to_one_msm_at_a_time",
            |name, dir| {
                let (circuit, witness, _) = load(dir);
                let mut t = StageTimings::default();
                let h = circuit.compute_h(&witness, &mut t).unwrap();
                let handle = h.device_handle::<HHandle>(crate::stages::TAG).unwrap();
                let first = handle.lend_entries();
                assert!(first.is_some(), "{name}: nothing lent");
                assert!(handle.lend_entries().is_none(), "{name}: lent twice");
                drop(first);
                assert!(handle.lend_entries().is_some(), "{name}: never given back");
            },
        );
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
    ///
    /// Four proofs sharing the device is also what gets a command buffer killed for
    /// `ImpactingInteractivity`, which failed this test in 10 of 20 runs before the
    /// proving path retried it. A failure names the artifact, the proof and the error.
    fn prove_concurrently(test: &str, backend: fn() -> Result<MetalBackend, ProveError>) {
        for_each(test, |name, dir| {
            let (circuit, witness, vk) = load_on(dir, backend().unwrap());
            let public = witness[1..=circuit.n_public()].to_vec();
            let circuit = circuit.as_ref();
            let (witness, vk, public) = (&witness, &vk, &public);

            std::thread::scope(|scope| {
                let threads: Vec<_> = (1..=4u64)
                    .map(|i| {
                        let thread = std::thread::Builder::new().name(format!("{name} proof {i}"));
                        thread
                            .spawn_scoped(scope, move || {
                                let mut t = StageTimings::default();
                                let proof = prove_with_blinders(
                                    circuit,
                                    witness,
                                    Fr::from(i * 7),
                                    Fr::from(i * 11),
                                    &mut t,
                                )
                                .unwrap_or_else(|e| panic!("{name}: proof {i}: {e}"));
                                verify(vk, public, &proof)
                                    .unwrap_or_else(|e| panic!("{name}: proof {i} verifies: {e}"));
                            })
                            .unwrap()
                    })
                    .collect();
                for t in threads {
                    t.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
                }
            });
        });
    }

    #[test]
    fn one_metal_circuit_proves_concurrently() {
        prove_concurrently("one_metal_circuit_proves_concurrently", MetalBackend::new);
    }

    #[test]
    fn one_constant_work_circuit_proves_concurrently() {
        prove_concurrently(
            "one_constant_work_circuit_proves_concurrently",
            MetalBackend::constant_work,
        );
    }

    /// Every submission of a proof failed in turn, once, after the GPU had finished it,
    /// and the retry must give the unfaulted proof bit for bit. A retry that re-ran an
    /// in-place transform on its own output, or read counters a previous attempt had
    /// already advanced, gives a different proof here rather than one in a hundred runs
    /// on a busy machine. Small domains only, to keep it quick and because from
    /// `OVERLAP_MIN_DOMAIN` up the witness batch is submitted from a second thread, which
    /// the per-thread injection misses. With the bound lifted and `G16_METAL_OVERLAP=0`
    /// both work modes passed on every artifact, anon-aadhaar included, in 467 s.
    fn a_retried_submission_is_exact(
        test: &str,
        backend: fn() -> Result<MetalBackend, ProveError>,
    ) {
        use crate::cb::inject;
        for_each(test, |name, dir| {
            let (circuit, witness, _) = load_on(dir, backend().unwrap());
            if circuit.domain_size() > 1 << 14 {
                return;
            }
            let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
            let mut t = StageTimings::default();
            inject::arm(None);
            let want = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let submissions = inject::calls();
            assert!(submissions > 0, "{name}: no submission was waited on");
            // A status fault and a stale token at every wait in turn: the second is the
            // buffer that says `Completed` without having run, and it must take the
            // same retry as the first.
            for fault in [inject::Fault::Status, inject::Fault::Stale] {
                for at in 0..submissions {
                    inject::arm_with(Some((at, fault)), false);
                    let got = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t)
                        .unwrap_or_else(|e| panic!("{name}: {fault:?} at wait {at}: {e}"));
                    assert!(
                        inject::fired(),
                        "{name}: wait {at} never failed ({fault:?})"
                    );
                    assert_eq!(got.a, want.a, "{name}: {fault:?} at wait {at}");
                    assert_eq!(got.b, want.b, "{name}: {fault:?} at wait {at}");
                    assert_eq!(got.c, want.c, "{name}: {fault:?} at wait {at}");
                }
            }
            inject::arm(None);
        });
    }

    #[test]
    fn a_retried_submission_gives_the_same_proof() {
        a_retried_submission_is_exact(
            "a_retried_submission_gives_the_same_proof",
            MetalBackend::new,
        );
    }

    #[test]
    fn a_retried_constant_work_submission_gives_the_same_proof() {
        a_retried_submission_is_exact(
            "a_retried_constant_work_submission_gives_the_same_proof",
            MetalBackend::constant_work,
        );
    }

    /// Every dispatch of a proof cut short in turn: the attempt runs whole up to it, half
    /// of it, and nothing after it in its command buffer (or in the attempt), so the
    /// retry runs over a gather that wrote half of A, B and C, an NTT batch stopped
    /// partway through its in-place passes, counters half incremented, a merge half
    /// folded into its buckets. The retried proof must still be the unfaulted one bit
    /// for bit, which holds only if every retried unit starts from inputs no dispatch in
    /// it writes. The fault injector before this one failed a buffer after it had run to
    /// its end, which never tests that. Small domains only, as the wait test above.
    fn a_submission_cut_short_is_retried_exactly(
        test: &str,
        backend: fn() -> Result<MetalBackend, ProveError>,
    ) {
        use crate::cb::inject;
        for_each(test, |name, dir| {
            let (circuit, witness, _) = load_on(dir, backend().unwrap());
            if circuit.domain_size() > 1 << 14 {
                return;
            }
            let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
            let mut t = StageTimings::default();
            inject::arm(None);
            let want = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let dispatches = inject::dispatches();
            assert!(dispatches > 0, "{name}: no dispatch was encoded");
            for rest in [inject::Rest::Buffer, inject::Rest::Attempt] {
                for at in 0..dispatches {
                    inject::arm_cut(at, rest);
                    let got = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t)
                        .unwrap_or_else(|e| panic!("{name}: cut at dispatch {at} ({rest:?}): {e}"));
                    assert!(
                        inject::fired(),
                        "{name}: dispatch {at} was never encoded ({rest:?})"
                    );
                    assert_eq!(got.a, want.a, "{name}: cut at dispatch {at} ({rest:?})");
                    assert_eq!(got.b, want.b, "{name}: cut at dispatch {at} ({rest:?})");
                    assert_eq!(got.c, want.c, "{name}: cut at dispatch {at} ({rest:?})");
                }
            }
            eprintln!("{name}: {dispatches} dispatches, each cut twice, retried exactly");
            inject::arm(None);
        });
    }

    #[test]
    fn a_submission_cut_short_gives_the_same_proof() {
        a_submission_cut_short_is_retried_exactly(
            "a_submission_cut_short_gives_the_same_proof",
            MetalBackend::new,
        );
    }

    #[test]
    fn a_constant_work_submission_cut_short_gives_the_same_proof() {
        a_submission_cut_short_is_retried_exactly(
            "a_constant_work_submission_cut_short_gives_the_same_proof",
            MetalBackend::constant_work,
        );
    }

    /// Every attempt's token is stale from the chosen wait on, so no retry can pass,
    /// and the proof must be refused as a device fault rather than returned: the seal
    /// is a check, not a hint. One wait per stage group (the gather, the MSM batch)
    /// rather than every one, since each case costs `RETRIES` proofs with backoff.
    fn a_stale_token_is_refused(test: &str, backend: fn() -> Result<MetalBackend, ProveError>) {
        use crate::cb::inject;
        for_each(test, |name, dir| {
            let (circuit, witness, _) = load_on(dir, backend().unwrap());
            if circuit.domain_size() > 1 << 12 {
                return;
            }
            let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
            let mut t = StageTimings::default();
            inject::arm(None);
            prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let submissions = inject::calls();
            for at in [0, submissions - 1] {
                inject::arm_with(Some((at, inject::Fault::Stale)), true);
                let err = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t)
                    .err()
                    .unwrap_or_else(|| panic!("{name}: a stale token from wait {at} proved"));
                assert!(err.is_device_fault(), "{name}: wait {at}: {err}");
                assert!(
                    err.to_string().contains("completion token"),
                    "{name}: wait {at}: {err}"
                );
            }
            inject::arm(None);
        });
    }

    #[test]
    fn a_stale_token_is_refused_once_the_retries_are_spent() {
        a_stale_token_is_refused(
            "a_stale_token_is_refused_once_the_retries_are_spent",
            MetalBackend::new,
        );
    }

    #[test]
    fn a_stale_constant_work_token_is_refused_once_the_retries_are_spent() {
        a_stale_token_is_refused(
            "a_stale_constant_work_token_is_refused_once_the_retries_are_spent",
            MetalBackend::constant_work,
        );
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
            let circuit = MetalCircuit::new(
                backend.stages.clone(),
                backend.msm.clone(),
                pk,
                Work::Variable,
            )
            .unwrap();
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
