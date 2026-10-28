//! Metal implementation of `snarkrs_groth16::Backend`.
//!
//! This module is only wiring. Stages 0 to 4 live in [`crate::stages`] and stages 5 to 9
//! in [`crate::msm`]; what happens here is the split between witness-independent work
//! (which must all be hoisted into [`MetalBackend::prepare`]) and per-proof work (which
//! must touch nothing shared and mutable).
//!
//! # What `prepare` costs, and where it goes
//!
//! On unified memory there is no PCIe hop, so "upload" here means one pack straight into
//! a `StorageModeShared` buffer, at whatever rate a cold page walk allows. The cost that
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
//!
//! The one stage the CPU does take is stage 6, B in G2, and not as a fallback: it runs
//! on the rayon pool at the same time as the GPU runs the other four, because the pool
//! is otherwise idle for the whole proof and G2 is the MSM this GPU is worst at. The
//! GPU still runs every other stage, so the proof's time is still the GPU's. See
//! [`g2_on_cpu`].

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use snarkrs_field::{Fr, G1Projective, G2Affine, G2Projective};
use snarkrs_formats::{Coefficients, ProvingKey, ZkeyError};
use snarkrs_groth16::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use snarkrs_msm::{CpuMsm, MsmBackend};

use crate::load::DeviceSink;
use crate::msm::{G1Bases, G2Bases, Job, JobG1, JobG2, MetalMsm, ScalarBuf, Work};
use crate::mtl::{self, Device};
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
        let device = mtl::system_default().ok_or_else(|| {
            bad("no Metal device on this machine, so the metal backend cannot run")
        })?;
        Self::with_device(device)
    }

    /// [`Self::new`] with MSMs whose cost follows the key and not the witness, so proving
    /// time, buffer sizes and dispatch geometry do not reveal how many witness entries
    /// are zero or one; see [`Work::Constant`].
    ///
    /// What stays witness dependent, on the device: which bucket a digit lands in. That
    /// is the digit pipeline's atomics, which contend by occupancy (at 2^18 the count,
    /// scan and scatter read 0.56 ms on a witness of zeros, 0.68 on bits, 0.84 dense and
    /// 1.9 when every scalar is one dense value, so every entry of a window hits one
    /// row), and the addresses the accumulation's and the fold's stores land on. Every
    /// other decision in the accumulation (`msm_segmented_constant_*`), the fold
    /// (`msm_fold_*`) and the reduce is a select on the entry rather than a branch, and
    /// every addition is a complete one (`Proj` in msm.metal), so the instruction
    /// stream, the loads, the stores and the field products of every thread are the
    /// key's: see `constant_work_phase_flatness` in msm_gpu_probes.rs for the per-phase
    /// medians across zeros, ones, bits, dense, one-value and small witnesses. On the
    /// host: the identity tests inside the combine's few hundred curve additions, as on
    /// the CPU backend. Stages 0 to 4 were already fixed-shape; their one value test,
    /// the gather's skip of the multiply for a 0 or 1 witness value, is off (keccak256's
    /// gather 0.45 ms to 0.71, what a dense witness costs; js_16x16_d32 unchanged at
    /// 1.56).
    ///
    /// Priced warm on the M2 Max, interleaved rounds against the variable path and
    /// against the constant-work kernels before the select-only accumulation (load 2.9
    /// to 3.3 at the start of each batch): keccak256, 10 reps x 4 rounds, 118 ms (round
    /// medians 115-120) against 28 variable (4.2x) and 108 (105-109) before, the MSM
    /// stage 99 against 18 and 86; anon-aadhaar, 4 reps x 3 rounds, 474-545 against
    /// 172-190 variable and 430-482 before, the MSM stage 398-447 against 106-117 and
    /// 340-385. The variable path itself is unchanged (keccak256 28.1 and 28.0 ms).
    /// js_16x16_d32 and rsa2048 were last priced with the shortcut accumulation, at
    /// +26% and 3.7x. The bit-heavy circuits pay more than on the CPU because the
    /// variable witness plans there were a few hundred scalars over a few windows, and
    /// each is now an MSM the size of H's, one of them in G2.
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

    /// [`ProvingKey::load`] and [`Backend::prepare`] as one pass: the G1 query sections and
    /// both CSR matrices are decoded and validated straight into their shared buffers, so
    /// none of them is ever an arkworks vector. Every check `load` makes still runs, the
    /// per-point curve checks and the `gamma_g2 != delta_g2` refusal included, and the
    /// circuit is the one `prepare` would build from the same file.
    ///
    /// On keccak_2048 (1.5M constraints, a 1.1 GB key) the two calls took 219 and 113 ms;
    /// what this skips is the vectors' writes to fresh pages and the serial repack of
    /// each into its buffer.
    pub fn load(&self, path: &Path) -> Result<MetalCircuit, LoadError> {
        self.load_with(path, false)
    }

    /// [`Self::load`] through [`ProvingKey::load_allowing_unblinded`]. Benchmark only, for
    /// the reason given there.
    pub fn load_allowing_unblinded(&self, path: &Path) -> Result<MetalCircuit, LoadError> {
        self.load_with(path, true)
    }

    fn load_with(&self, path: &Path, allow_unblinded: bool) -> Result<MetalCircuit, LoadError> {
        let mut sink = DeviceSink::new(self.device());
        let loaded = if allow_unblinded {
            ProvingKey::load_allowing_unblinded_into(path, &mut sink)
        } else {
            ProvingKey::load_into(path, &mut sink)
        };
        let pk = match loaded {
            Ok(pk) => pk,
            Err(e) => {
                return Err(sink
                    .failed
                    .take()
                    .map_or(LoadError::Key(e), LoadError::Prove))
            }
        };
        Ok(MetalCircuit::from_sink(
            self.stages.clone(),
            self.msm.clone(),
            pk,
            sink,
            self.work,
        )?)
    }
}

/// Why [`MetalBackend::load`] failed: the file is not a key it accepts, or the device
/// could not take it.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Key(#[from] ZkeyError),
    #[error(transparent)]
    Prove(#[from] ProveError),
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
/// the counting-sort and bucket scratch and takes it back after `waitUntilCompleted`.
/// Neither pool is reachable from this struct, so two concurrent `prove` calls against
/// one `MetalCircuit` cannot be handed the same buffer. Command buffers and encoders,
/// the two Metal objects that are not thread safe, are created and destroyed inside a
/// single call and never stored.
///
/// # What stays on the host
///
/// The dimensions, all five header points used for blinding/assembly, and the full
/// verifying key (including `vk.ic`). The five query sections and all coefficient vectors
/// are emptied as each one is uploaded. [`PreparedCircuit::key`] permits releasing this
/// bulk storage; its result is not a complete key for preparing another backend.
///
/// They go through [`snarkrs_groth16::malloc::release`] rather than a plain drop, because macOS
/// malloc keeps freed large blocks resident in its large cache, and on js_384x384_d32
/// that held 2.4 GB of the 9.1 GB peak.
pub struct MetalCircuit {
    pk: ProvingKey,
    stages: Arc<HStages>,
    msm: Arc<MetalMsm>,
    resident: HResident,
    a_bases: G1Bases,
    b_g1_bases: G1Bases,
    b_g2: BG2,
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
            value_id,
            values,
        } = std::mem::replace(
            &mut pk.coeffs,
            Coefficients {
                row_ptr: Default::default(),
                signal: Default::default(),
                value_id: Default::default(),
                values: Vec::new(),
            },
        );
        row_ptr
            .into_iter()
            .chain(signal)
            .chain(value_id)
            .for_each(snarkrs_groth16::malloc::release);
        snarkrs_groth16::malloc::release(values);
        let stages_us = t0.elapsed().as_micros() as u64;

        // Stages 5 to 9. Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this
        // arkworks and `G2Affine` is 136, neither is `repr(C)`, and both carry an
        // infinity flag that the packed layout encodes as all-zero coordinates instead.
        // Each host vector is taken and released as soon as it is uploaded.
        let t1 = Instant::now();
        let g1 = |v: Vec<_>| {
            let bases = msm.upload_g1_bases(&v);
            snarkrs_groth16::malloc::release(v);
            bases
        };
        let a_bases = g1(std::mem::take(&mut pk.a_query))?;
        let b_g1_bases = g1(std::mem::take(&mut pk.b_g1_query))?;
        let b_g2 = BG2::new(
            &msm,
            std::mem::take(&mut pk.b_g2_query),
            pk.domain_size,
            work,
        )?;
        let l_bases = g1(std::mem::take(&mut pk.l_query))?;
        let h_bases = g1(std::mem::take(&mut pk.h_query))?;
        let bases_us = t1.elapsed().as_micros() as u64;

        Ok(Self::finish(
            pk,
            stages,
            msm,
            resident,
            [a_bases, b_g1_bases, l_bases, h_bases],
            b_g2,
            (stages_us, bases_us, t_all),
            work,
        ))
    }

    /// [`Self::new`] for a key [`MetalBackend::load`] decoded into `sink`: the G1 bases and
    /// both CSR matrices are already in their buffers, so what is left is the twiddle and
    /// coset tables, the G2 bases, and the same shape checks.
    fn from_sink(
        stages: Arc<HStages>,
        msm: Arc<MetalMsm>,
        mut pk: ProvingKey,
        sink: DeviceSink<'_>,
        work: Work,
    ) -> Result<Self, ProveError> {
        let n_vars = pk.n_vars;
        let [Some(a), Some(b1), Some(l), Some(h)] = sink.g1 else {
            return Err(bad(
                "the loader did not hand every G1 query section to the sink",
            ));
        };
        let csr = sink
            .csr
            .ok_or_else(|| bad("the loader did not hand section 4 to the sink"))?;
        let want_l = n_vars
            .checked_sub(pk.n_public + 1)
            .ok_or_else(|| bad(format!("n_public {} exceeds n_vars {n_vars}", pk.n_public)))?;
        for (name, got, want) in [
            ("a_query", a.len(), n_vars),
            ("b_g1_query", b1.len(), n_vars),
            ("b_g2_query", pk.b_g2_query.len(), n_vars),
            ("l_query", l.len(), want_l),
            ("h_query", h.len(), pk.domain_size),
        ] {
            if got != want {
                return Err(bad(format!("{name} has {got} bases, want {want}")));
            }
        }

        let t_all = Instant::now();
        let resident = stages.prepare_csr(pk.domain_size, n_vars, csr)?;
        let stages_us = t_all.elapsed().as_micros() as u64;
        let t1 = Instant::now();
        let b_g2 = BG2::new(
            &msm,
            std::mem::take(&mut pk.b_g2_query),
            pk.domain_size,
            work,
        )?;
        let bases_us = t1.elapsed().as_micros() as u64;
        Ok(Self::finish(
            pk,
            stages,
            msm,
            resident,
            [a, b1, l, h],
            b_g2,
            (stages_us, bases_us, t_all),
            work,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        pk: ProvingKey,
        stages: Arc<HStages>,
        msm: Arc<MetalMsm>,
        resident: HResident,
        [a_bases, b_g1_bases, l_bases, h_bases]: [G1Bases; 4],
        b_g2: BG2,
        (stages_us, bases_us, t_all): (u64, u64, Instant),
        work: Work,
    ) -> Self {
        let cost = PrepareCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        if std::env::var_os("G16_METAL_PREPARE").is_some() {
            eprintln!(
                "metal prepare: stages {} us, bases {} us, total {} us (domain {}, n_vars {})",
                cost.stages_us, cost.bases_us, cost.total_us, pk.domain_size, pk.n_vars,
            );
        }

        Self {
            pk,
            stages,
            msm,
            resident,
            a_bases,
            b_g1_bases,
            b_g2,
            l_bases,
            h_bases,
            cost,
            work,
        }
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
    fn witness_jobs<'a>(&'a self, w: &'a ScalarBuf) -> Vec<Job<'a>> {
        let mut jobs = vec![Job::G1(JobG1 {
            bases: &self.a_bases,
            base_off: 0,
            scalars: w,
            scalar_off: 0,
            n: self.pk.n_vars,
        })];
        if let BG2::Gpu(bases) = &self.b_g2 {
            jobs.push(Job::G2(JobG2 {
                bases,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n: self.pk.n_vars,
            }));
        }
        jobs.push(Job::G1(JobG1 {
            bases: &self.b_g1_bases,
            base_off: 0,
            scalars: w,
            scalar_off: 0,
            n: self.pk.n_vars,
        }));
        jobs.push(Job::G1(JobG1 {
            bases: &self.l_bases,
            base_off: 0,
            scalars: w,
            // Section 8 covers the private wires only: witness[0] is the constant 1
            // and witness[1..=n_public] are the public inputs, which the verifier
            // folds in through IC instead.
            scalar_off: self.pk.n_public + 1,
            n: self.l_bases.len(),
        }));
        jobs
    }

    /// Stage 6 on the CPU pool, started from `s` so it runs while the GPU has the other
    /// four; `None` when B in G2 is one of the [`Self::witness_jobs`].
    fn b_g2_on_cpu<'s>(
        &'s self,
        s: &'s std::thread::Scope<'s, '_>,
        witness: &'s [Fr],
    ) -> Option<std::thread::ScopedJoinHandle<'s, G2Projective>> {
        let BG2::Cpu { bases, pool } = &self.b_g2 else {
            return None;
        };
        Some(s.spawn(move || {
            pool.install(|| {
                let cpu = match self.work {
                    Work::Constant => CpuMsm::constant_work(),
                    Work::Variable => CpuMsm::new(),
                };
                cpu.msm_g2(bases, witness)
            })
        }))
    }

    /// A, B in G2, B in G1 and L, from the [`Self::witness_jobs`] results in job order
    /// and stage 6's CPU result when it ran there.
    fn witness_results(
        &self,
        out: Vec<crate::msm::MsmResult>,
        cpu_b_g2: Option<std::thread::ScopedJoinHandle<'_, G2Projective>>,
    ) -> Result<(G1Projective, G2Projective, G1Projective, G1Projective), ProveError> {
        let want = 3 + usize::from(cpu_b_g2.is_none());
        if out.len() != want {
            return Err(bad(format!(
                "msm_batch returned {} witness results, want {want}",
                out.len()
            )));
        }
        let mut out = out.into_iter();
        let mut next = || out.next().expect("length checked above");
        let a_g1 = next().g1()?;
        let b_g2 = match cpu_b_g2 {
            Some(h) => h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)),
            None => next().g2()?,
        };
        let b_g1 = next().g1()?;
        let l_g1 = next().g1()?;
        Ok((a_g1, b_g2, b_g1, l_g1))
    }

    /// Stage 9's scalars. The normal path is the device buffer stage 4 wrote, which
    /// never touches the host. `HPoly::require_handle` refuses a handle from some other
    /// backend and returns `None` for a host vector, so a CPU `compute_h` can still be
    /// finished on the GPU rather than rejected.
    ///
    /// The returned `ScalarBuf` reads the buffer `h` owns, so `h` (and with it the
    /// scratch behind the handle) must outlive the `msm_batch` that consumes it; that
    /// is the lifetime `scalars_from_device_std` requires.
    fn h_scalars(&self, h: &HPoly) -> Result<ScalarBuf, ProveError> {
        match h.require_handle::<HHandle>(crate::stages::TAG)? {
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
                let HPoly::Host(host) = h else {
                    unreachable!("require_handle returns None only for a host vector")
                };
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

/// Where stage 6, B in G2, runs for one key.
enum BG2 {
    Gpu(G2Bases),
    /// The key's host bases, for [`CpuMsm`] on a pool of its own.
    Cpu {
        bases: Vec<G2Affine>,
        pool: rayon::ThreadPool,
    },
}

impl BG2 {
    /// Where stage 6 runs for this key, see [`g2_on_cpu`], with the bases put there.
    fn new(
        msm: &MetalMsm,
        v: Vec<G2Affine>,
        domain_size: usize,
        work: Work,
    ) -> Result<Self, ProveError> {
        if g2_on_cpu(&v, domain_size, work) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(rayon::current_num_threads())
                .thread_name(|i| format!("metal-g2-{i}"))
                .build()
                .map_err(|e| bad(format!("stage 6 thread pool: {e}")))?;
            Ok(BG2::Cpu { bases: v, pool })
        } else {
            let bases = msm.upload_g2_bases(&v);
            snarkrs_groth16::malloc::release(v);
            Ok(BG2::Gpu(bases?))
        }
    }
}

/// Whether stage 6 runs on the CPU, on a pool of its own, while the GPU runs stages 5, 7,
/// 8 and 9.
///
/// The G2 MSM is the one this GPU is worst at (three Fq products per Fq2 product and a
/// 256-byte reduce entry), and the CPU pool is otherwise idle for the whole proof. The
/// pool is its own rather than the global one because stages 0 to 4 and the MSM's
/// scalar classify and combine run on the global pool, and behind a busy G2 MSM they
/// waited: rsa2048 constant-work's ntt went from 19 ms to 132.
///
/// Variable work moves it everywhere. Warm on the M2 Max, interleaved medians against the
/// GPU-only batch: js_16x16_d32 90.8 ms to 67.6, anon-aadhaar 186.1 to 175.3,
/// js_384x384_d32 1436 to 1125, railgun-13x01 90.2 to 52.4, js_8x8_d32 54.9 to 39.0,
/// js_2x2_d16 14.8 to 8.9, rsa2048 35.1 to 31.6, sha256 even. keccak256 is the one that
/// does not gain: 27.3 to 27.9 at the median, 26.8 to 26.1 at the minimum. Its B query is
/// half infinity and its witness mostly bits, so the GPU's G2 job was nearly free, and
/// what is left is the pool's contention with stages 0 to 4 (ntt 9.1 to 9.8 ms).
///
/// Constant work makes the CPU pay for every finite base, so the decision is the key's
/// finite G2 bases against the domain, both fixed by the key and not the witness. It won
/// at 0.23 of the domain (anon-aadhaar, 636 ms to 416), 0.35 (js_16x16_d32, 115 to 88),
/// 0.47 (keccak256, 154 to 111), 0.55 (js_1x1_d8, 18.7 to 11.6) and 0.57 (tornado, 40.9
/// to 33.7), and lost at 0.71 (rsa2048, 129 to 164), so the line sits at 5/8.
///
/// `G16_METAL_G2_CPU=0` or `=1` forces it either way, for re-measuring.
fn g2_on_cpu(bases: &[G2Affine], domain_size: usize, work: Work) -> bool {
    match std::env::var("G16_METAL_G2_CPU")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
    {
        Some(0) => false,
        Some(_) => true,
        None => match work {
            Work::Variable => true,
            Work::Constant => {
                let finite = bases.iter().filter(|p| !p.infinity).count();
                finite * 8 <= domain_size * 5
            }
        },
    }
}

impl PreparedCircuit for MetalCircuit {
    /// Always [`crate::stages::TAG`] ("metal"), and that is a claim about what ran: no stage
    /// in this backend has a CPU fallback, so the name cannot be describing a proof the CPU
    /// did. Stage 6 on the pool is concurrent with the GPU's four, not instead of them.
    fn backend_name(&self) -> &'static str {
        crate::stages::TAG
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

        let (a_g1, b_g2, b_g1, l_g1, h_g1) = std::thread::scope(|s| {
            let cpu_b_g2 = self.b_g2_on_cpu(s, witness);
            let w = self.msm.upload_scalars_with(witness, self.work)?;
            let h_scalars = self.h_scalars(h)?;

            // One command buffer for all of them. Order matches the stage numbering,
            // and `msm_batch` returns results in job order.
            let mut jobs = self.witness_jobs(&w);
            jobs.push(self.h_job(&h_scalars));
            let mut out = self.msm.msm_batch(&jobs)?;
            let h_g1 = out
                .pop()
                .ok_or_else(|| bad("msm_batch returned no results"))?
                .g1()?;
            let (a_g1, b_g2, b_g1, l_g1) = self.witness_results(out, cpu_b_g2)?;
            Ok::<_, ProveError>((a_g1, b_g2, b_g1, l_g1, h_g1))
        })?;
        // One command buffer for the GPU's share, so `msm_split` stays `None`.
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
        match h.require_handle::<HHandle>(crate::stages::TAG) {
            Ok(Some(handle)) => handle.to_host(),
            Ok(None) => h.to_host().map(<[Fr]>::to_vec),
            Err(_) => None,
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
            let cpu_b_g2 = self.b_g2_on_cpu(s, witness);
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
            let wit_out = wit
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))
                .and_then(|out| self.witness_results(out, cpu_b_g2));
            (h_out, wit_out)
        });
        let (a_g1, b_g2, b_g1, l_g1) = wit_out?;
        let h_out = h_out?;
        if h_out.len() != 1 {
            return Err(bad(format!("msm_batch returned {} h results", h_out.len())));
        }
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
    mod resources {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-support/resources.rs"
        ));
    }

    use super::*;
    use snarkrs_formats::{wtns::Witness, Matrix, VerifyingKey};
    use snarkrs_groth16::prove::prove_with_blinders;
    use snarkrs_groth16::verify::verify;
    use std::path::{Path, PathBuf};

    #[test]
    #[ignore = "requires a Metal device and bench/artifacts/tiny_mul/circuit.zkey"]
    fn key_retention_releases_only_metal_bulk_storage() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts/tiny_mul/circuit.zkey");
        let expected = ProvingKey::load(&path).expect("tiny_mul key");
        assert!(!expected.a_query.is_empty());
        assert!(!expected.vk.ic.is_empty());
        let circuit = MetalBackend::new()
            .expect("Metal device")
            .prepare(ProvingKey::load(&path).unwrap())
            .unwrap();
        let key = circuit.key();
        let dimensions = (expected.n_vars, expected.n_public, expected.domain_size);
        assert_eq!((key.n_vars, key.n_public, key.domain_size), dimensions);
        assert_eq!(
            (circuit.n_vars(), circuit.n_public(), circuit.domain_size()),
            dimensions
        );
        assert_eq!(
            (
                key.alpha_g1,
                key.beta_g1,
                key.beta_g2,
                key.delta_g1,
                key.delta_g2
            ),
            (
                expected.alpha_g1,
                expected.beta_g1,
                expected.beta_g2,
                expected.delta_g1,
                expected.delta_g2
            )
        );
        assert_eq!(
            (
                key.vk.alpha_g1,
                key.vk.beta_g2,
                key.vk.gamma_g2,
                key.vk.delta_g2,
                &key.vk.ic
            ),
            (
                expected.vk.alpha_g1,
                expected.vk.beta_g2,
                expected.vk.gamma_g2,
                expected.vk.delta_g2,
                &expected.vk.ic
            )
        );
        assert_eq!(
            [
                key.a_query.len(),
                key.b_g1_query.len(),
                key.b_g2_query.len(),
                key.l_query.len(),
                key.h_query.len()
            ],
            [0; 5]
        );
        assert_eq!(
            [
                key.a_query.capacity(),
                key.b_g1_query.capacity(),
                key.b_g2_query.capacity(),
                key.l_query.capacity(),
                key.h_query.capacity()
            ],
            [0; 5]
        );
        for m in Matrix::BOTH.map(Matrix::index) {
            assert_eq!(
                [
                    key.coeffs.row_ptr[m].len(),
                    key.coeffs.signal[m].len(),
                    key.coeffs.value_id[m].len()
                ],
                [0; 3]
            );
            assert_eq!(
                [
                    key.coeffs.row_ptr[m].capacity(),
                    key.coeffs.signal[m].capacity(),
                    key.coeffs.value_id[m].capacity()
                ],
                [0; 3]
            );
        }
        assert_eq!(key.coeffs.values.capacity(), 0);
    }

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
            resources::skip(format_args!("{test}: no artifacts under bench/artifacts"));
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

    /// `snarkrs_groth16::trace` is the only caller and it takes the trait default unless a
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
            let cpu = snarkrs_groth16::cpu::CpuBackend::new()
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

    /// One store of each submission in turn is wrong with status and token in place,
    /// the fault no completion check sees (BUG-28). The backend returns one of two
    /// things and never a third: the unfaulted proof, when the wrong word was a window
    /// sum and the on-curve check sent the batch round again, or a proof that does not
    /// verify, when it was a word of H, which stays in the field and reaches the proof.
    /// `prove` wraps this in `verify` and refuses the second at the exit; a caller of
    /// `prove_unchecked` is handed it.
    fn a_wrong_store_is_retried_or_reaches_the_proof(
        test: &str,
        backend: fn() -> Result<MetalBackend, ProveError>,
    ) {
        use crate::cb::inject;
        for_each(test, |name, dir| {
            let (circuit, witness, vk) = load_on(dir, backend().unwrap());
            if circuit.domain_size() > 1 << 14 {
                return;
            }
            let public = witness[1..=circuit.n_public()].to_vec();
            let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
            let mut t = StageTimings::default();
            inject::arm(None);
            let want = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let submissions = inject::calls();
            let (mut retried, mut escaped) = (0, 0);
            for at in 0..submissions {
                inject::arm_with(Some((at, inject::Fault::WrongStore)), false);
                let got = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t)
                    .unwrap_or_else(|e| panic!("{name}: WrongStore after wait {at}: {e}"));
                assert!(inject::fired(), "{name}: no readback after wait {at}");
                if (got.a, got.b, got.c) == (want.a, want.b, want.c) {
                    retried += 1;
                    assert!(
                        inject::calls() > submissions,
                        "{name}: wait {at}: the unfaulted proof with no retry"
                    );
                } else {
                    escaped += 1;
                    assert!(
                        verify(&vk, &public, &got).is_err(),
                        "{name}: wait {at}: a proof from a wrong store verifies"
                    );
                }
            }
            eprintln!("{test}: {name}: {retried} retried, {escaped} reached the proof");
            assert!(retried > 0, "{name}: no wrong window sum was caught");
            assert!(escaped > 0, "{name}: no wrong H word reached the proof");
            inject::arm(None);
        });
    }

    #[test]
    fn a_wrong_store_is_retried_or_fails_the_proof() {
        a_wrong_store_is_retried_or_reaches_the_proof(
            "a_wrong_store_is_retried_or_fails_the_proof",
            MetalBackend::new,
        );
    }

    #[test]
    fn a_constant_work_wrong_store_is_retried_or_fails_the_proof() {
        a_wrong_store_is_retried_or_reaches_the_proof(
            "a_constant_work_wrong_store_is_retried_or_fails_the_proof",
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
    /// `cargo test -p snarkrs-metal --release --lib measure_prepare -- --ignored --nocapture`.
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
