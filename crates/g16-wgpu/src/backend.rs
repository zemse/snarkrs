//! The WebGPU implementation of `g16_core::Backend`: the file that turns nine stages of
//! kernels into a proof.
//!
//! Everything here is wiring. Stages 0 to 4 are [`crate::stages`], stages 5 to 9 are
//! [`crate::batch`], and what happens in this file is the split between witness-independent
//! work, which must all be hoisted into `WgpuProver::prepare`, and per-proof work, which
//! must touch nothing shared and mutable.
//!
//! # One circuit, two ways of waiting for it
//!
//! `g16_core::PreparedCircuit` is a synchronous trait and every wgpu readback is
//! fundamentally asynchronous, so something has to block. On native, `pollster::block_on`
//! blocks a thread while `device.poll` drives the queue, which is exactly what the readback
//! is documented to need. In a browser there is no thread to block: `poll` is a no-op, the
//! only thing that fires a `mapAsync` callback is a yield to the event loop, and
//! `pollster::block_on` compiles for wasm32 and then hangs the tab.
//!
//! So the *async* half of this file compiles on both targets and the blocking half does not.
//! [`WgpuCircuit::compute_h_async`] and [`WgpuCircuit::msms_async`] are the real
//! implementations; the `PreparedCircuit` impl is four `pollster::block_on` calls over them
//! and is `cfg`-gated off wasm32, and `crate::wasm` awaits the same two functions from the
//! browser's dedicated Web Worker. U13 did it this way after starting to copy the file: the
//! only thing that differs between a native proof and a browser proof is who waits, and
//! duplicating 200 lines of job wiring to express that is how the two backends drift.
//!
//! # What `prepare` hoists, and what it cannot
//!
//! The trait's own words: "Witness-independent work: sort section 4 into CSR, build twiddles,
//! and for a GPU backend upload every base vector and keep it resident."  All of that happens
//! in [`WgpuCircuit::new`]. At `js_16x16_d32` it is about 26 MB of CSR, 24 MB of twiddle and
//! coset tables and 62 MB of base vectors.
//!
//! One thing that cannot be hoisted to the *backend* the way `g16-metal` hoists it: the NTT
//! pipelines. WGSL has no runtime-sized `var<workgroup>`, so the tile size is a compile-time
//! constant and the six transforms are compiled for one specific `log_n`. `HStages` therefore
//! lives on the circuit and two keys of different domain sizes compile it twice.
//! [`crate::batch::MsmBatch`] does not have that problem, so the three MSM modules and their
//! fifteen pipelines are built once per process and shared through an `Arc`.
//!
//! # There is no CPU fallback, and no size gate
//!
//! Every stage runs on the device and every one is checked against the CPU backend element by
//! element, so `WgpuCircuit::backend_name` can honestly say "wgpu" for the whole proof.
//! There is deliberately no "small circuits go to the CPU" gate either, for the reason
//! `g16-metal` gives: a gate that quietly ran the other backend would make
//! `g16 prove --backend wgpu` report a number that belongs to somebody else. The crossover is
//! a measurement to publish, and `tests/proof.rs` publishes it.

use std::sync::Arc;
// Not `std::time`: `Instant::now()` panics at run time on wasm32-unknown-unknown, and every
// timing below is taken on the browser's path too. web-time is a plain re-export of
// `std::time` on native, so no number on this machine changes.
use web_time::Instant;

use g16_core::{HPoly, MsmOutputs, ProveError, StageTimings};
use g16_field::{Fr, One, Zero};
use g16_zkey::ProvingKey;

use crate::batch::{G1Bases, G2Bases, Group, Job, MontConvert, MsmBatch, Source};
use crate::device::{bad, WgpuBackend};
use crate::msm::Work;
use crate::readback::Seal;
use crate::stages::{HStages, WgpuHandle};

#[cfg(not(target_arch = "wasm32"))]
use crate::readback::is_aborted;

// `Backend` and `PreparedCircuit` are the synchronous traits, `Stage4` is only reachable
// through the synchronous `compute_h_with`, and `LimitsProfile::from_env` reads an
// environment a browser does not have. All four are native only, and importing them
// unconditionally is an unused-import warning on wasm32 rather than an error, which is
// exactly the kind of warning that gets ignored until it hides a real one.
#[cfg(not(target_arch = "wasm32"))]
use crate::device::LimitsProfile;
#[cfg(not(target_arch = "wasm32"))]
use crate::stages::Stage4;
#[cfg(not(target_arch = "wasm32"))]
use g16_core::{Backend, PreparedCircuit};

/// Where the time in [`WgpuProver::prepare`] went, in microseconds.
///
/// Kept per circuit rather than printed, so a caller can report it without re-running the
/// preparation to time it. `G16_WGPU_PREPARE=1` prints it as well.
#[derive(Clone, Copy, Debug, Default)]
pub struct CircuitCost {
    /// [`HStages::new`]: three shader modules, the CSR, both twiddle tables and the coset
    /// powers.
    pub stages_us: u64,
    /// The five base vectors repacked from ark's 72 and 136 byte affines and uploaded.
    pub bases_us: u64,
    pub total_us: u64,
}

/// One device, one queue, and the three MSM shader modules, built once per process.
///
/// Cloning is not offered: what makes this expensive is the pipeline compiles hanging off it,
/// and [`Self::prepare`] hands them to every circuit through an `Arc` rather than duplicating
/// anything.
pub struct WgpuProver {
    device: Arc<WgpuBackend>,
    msm: Arc<MsmBatch>,
    work: Work,
}

impl WgpuProver {
    /// Opens an adapter and a device at the profile named by `G16_WGPU_LIMITS`, which
    /// defaults to the **browser floor** and not this adapter's own limits.
    ///
    /// That default is the point of the crate and it costs throughput: `Raised` on this M2
    /// Max is 4 GiB of storage binding against 128 MiB and 32 KiB of workgroup storage
    /// against 16 KiB. A kernel that only fits the raised profile is a kernel that fails in a
    /// stock browser, and native `cargo test` will not notice, so the number this backend
    /// reports by default is the one a browser could reproduce.
    ///
    /// Fails rather than falling back to anything. A benchmark that quietly measures a
    /// different backend is worse than no number at all.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, ProveError> {
        Self::with_profile(LimitsProfile::from_env()?)
    }

    /// [`Self::new`] with MSMs whose cost follows the key and not the witness, so proving
    /// time, buffer sizes and dispatch geometry do not reveal how many witness entries are
    /// zero or one; see [`Work::Constant`].
    ///
    /// What stays witness dependent, on the device: which bucket a digit lands in, so the
    /// atomics' contention and the run count of the accumulation's slices. The merge is a
    /// fixed tree (`msm_fold_*`) and it and the reduce (`msm_reduce_constant_*`) add with
    /// the complete formulas, so an empty bucket or slot costs what a full one does:
    /// `tests/constant_work.rs::constant_work_phase_occupancy` at 2^18 reads the G1 fold
    /// at 24.0, 26.9 and 25.7 ms and the G1 reduce at 32.2 on a witness of zeros, of bits
    /// and a dense one, and in G2 103, 103 and 104 against 186 on all three (at fold
    /// length 16, the run that saw no abort), where the shortcut kernels read 1.75
    /// against 2.62 and 1.56 against 11.6 in G1. On the host: the identity tests inside
    /// the combine's few dozen curve additions, as on the CPU backend. Stages 0 to 4 were
    /// fixed-shape already: the gather multiplies every term and the field prelude
    /// reduces with `select`.
    ///
    /// Priced warm on the M2 Max at the floor profile, 15 reps, alternating rounds under
    /// the GPU lock, medians against the variable path: js_16x16_d32 1097 ms against 708
    /// (+55%), keccak256 1301 against 133 (9.8x), rsa2048 1099 against 316 (3.5x),
    /// anon-aadhaar 5363 against 1087 (4.9x, 3 reps). Steeper than Metal's +26% and 5.7x
    /// because the constant-work kernels are all curve arithmetic, where WGSL is 2.3x
    /// behind MSL, and the bit-heavy circuits' G2 job goes from a few hundred scalars to
    /// the size of H's.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn constant_work() -> Result<Self, ProveError> {
        let mut prover = Self::new()?;
        prover.work = Work::Constant;
        Ok(prover)
    }

    /// Native only: opening a device is async and blocking a browser's only thread on it
    /// hangs the tab. The browser opens the device with `.await` in `crate::wasm`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_profile(profile: LimitsProfile) -> Result<Self, ProveError> {
        let device = pollster::block_on(WgpuBackend::with_profile(profile))?;
        Self::with_device(Arc::new(device))
    }

    /// Same, on a device the caller already opened. `tests/proof.rs` uses it to share one
    /// adapter across every artifact in a binary, which is 5 fewer device creations.
    pub fn with_device(device: Arc<WgpuBackend>) -> Result<Self, ProveError> {
        Self::with_device_and_work(device, Work::Variable)
    }

    /// Same, with the work mode chosen. The three MSM modules carry both modes, so a
    /// constant-work prover and a variable one on the same device compile nothing twice.
    pub fn with_device_and_work(device: Arc<WgpuBackend>, work: Work) -> Result<Self, ProveError> {
        let start = Instant::now();
        let msm = MsmBatch::new(&device)?;
        let cost = msm.cost();
        if std::env::var_os("G16_WGPU_PREPARE").is_some() {
            eprintln!(
                "wgpu backend: {} MSM modules, {} pipelines, {} us of naga and {} us of \
                 pipeline creation, {} us wall",
                cost.modules,
                cost.pipelines,
                cost.module_us,
                cost.pipeline_us,
                start.elapsed().as_micros(),
            );
        }
        Ok(Self {
            device,
            msm: Arc::new(msm),
            work,
        })
    }

    pub fn device(&self) -> &Arc<WgpuBackend> {
        &self.device
    }

    pub fn work(&self) -> Work {
        self.work
    }

    /// The shared MSM pipelines, for a report. `tests/proof.rs` reads
    /// [`MsmBatch::last_readback_bytes`] through it, which is the only way to hold this
    /// backend to design §3's 64 KiB per-proof readback budget from outside.
    pub fn msm(&self) -> &MsmBatch {
        &self.msm
    }
}

/// Native only, because `prepare` hands back a `Box<dyn PreparedCircuit>` and that trait is
/// synchronous. The browser builds a [`WgpuCircuit`] directly; see `crate::wasm`.
#[cfg(not(target_arch = "wasm32"))]
impl Backend for WgpuProver {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        let _gpu = self.device.exclusive();
        let mut circuit = WgpuCircuit::with_work(
            Arc::clone(&self.device),
            Arc::clone(&self.msm),
            pk,
            self.work,
        )?;
        // The key lands in a sealed submission of its own, and one the GPU cut short is
        // uploaded again: see `WgpuCircuit::key_landed`.
        let mut first = true;
        retry_aborted("the key upload", &mut StageTimings::default(), |_| {
            if !first {
                circuit.reupload()?;
            }
            first = false;
            pollster::block_on(circuit.key_landed())
        })?;
        Ok(Box::new(circuit))
    }
}

/// A key whose witness-independent data is device resident: the CSR, both twiddle tables, the
/// coset powers and all five base vectors.
///
/// # Concurrency
///
/// Nothing here is mutated by a proof. The per-proof scratch is pooled behind two mutexes,
/// one owned by [`HStages`] and one by [`MsmBatch`], so two concurrent proofs cannot be
/// handed the same buffer. What they *would* share is the device's single uncaptured-error
/// slot and the device-wide `onSubmittedWorkDone` fence, so both `compute_h` and `msms` hold
/// [`WgpuBackend::exclusive`] for their whole GPU section. Concurrent proofs against one
/// circuit are therefore correct and serialised rather than parallel, which is what that
/// method's doc comment argues is the only honest option on one queue.
pub struct WgpuCircuit {
    pk: ProvingKey,
    device: Arc<WgpuBackend>,
    msm: Arc<MsmBatch>,
    stages: HStages,
    a_bases: G1Bases,
    b_g1_bases: G1Bases,
    b_g2_bases: G2Bases,
    l_bases: G1Bases,
    h_bases: G1Bases,
    cost: CircuitCost,
    /// What [`PreparedCircuit::msms`] proves under; the browser chooses per call instead.
    work: Work,
}

impl WgpuCircuit {
    /// Uploads everything witness independent. `pub` because `crate::wasm` builds a circuit
    /// without going through [`Backend::prepare`], whose return type is a boxed
    /// `PreparedCircuit` and therefore native only.
    pub fn new(
        device: Arc<WgpuBackend>,
        msm: Arc<MsmBatch>,
        pk: ProvingKey,
    ) -> Result<Self, ProveError> {
        Self::with_work(device, msm, pk, Work::Variable)
    }

    /// Same, with the work mode [`PreparedCircuit::msms`] proves under.
    pub fn with_work(
        device: Arc<WgpuBackend>,
        msm: Arc<MsmBatch>,
        pk: ProvingKey,
        work: Work,
    ) -> Result<Self, ProveError> {
        // Shape checks first. `crate::batch` would reject an out-of-range job later, but by
        // then the message names buffer offsets rather than the section of the zkey that is
        // the wrong length, and the mismatch is a property of the key and not of the proof.
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
                "l_query has {} bases, the private witness is {want_l} long",
                pk.l_query.len()
            )));
        }

        let (stages, bases, cost) = Self::upload(&device, &pk)?;
        Ok(Self {
            pk,
            device,
            msm,
            stages,
            a_bases: bases.0,
            b_g1_bases: bases.1,
            b_g2_bases: bases.2,
            l_bases: bases.3,
            h_bases: bases.4,
            cost,
            work,
        })
    }

    /// The device-resident half of a circuit: stages 0 to 4's modules and tables, then the
    /// five base vectors. The uploads are queued here and land with the next submission;
    /// [`Self::key_landed`] is that submission.
    #[allow(clippy::type_complexity)]
    fn upload(
        device: &WgpuBackend,
        pk: &ProvingKey,
    ) -> Result<
        (
            HStages,
            (G1Bases, G1Bases, G2Bases, G1Bases, G1Bases),
            CircuitCost,
        ),
        ProveError,
    > {
        let t_all = Instant::now();
        // Stages 0 to 4, including the three shader modules. This also validates the key the
        // way the CPU backend does (power-of-two domain, CSR row count, every signal index
        // below n_vars), so a bad key is an error here rather than an out-of-range GPU read
        // later.
        let t0 = Instant::now();
        let stages = HStages::new(device, pk)?;
        let stages_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let a_bases = G1Bases::upload(device, &pk.a_query)?;
        let b_g1_bases = G1Bases::upload(device, &pk.b_g1_query)?;
        let b_g2_bases = G2Bases::upload(device, &pk.b_g2_query)?;
        let l_bases = G1Bases::upload(device, &pk.l_query)?;
        let h_bases = G1Bases::upload(device, &pk.h_query)?;
        let bases_us = t1.elapsed().as_micros() as u64;

        let cost = CircuitCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        if std::env::var_os("G16_WGPU_PREPARE").is_some() {
            eprintln!(
                "wgpu prepare: stages {} us, bases {} us ({} MB), total {} us (domain {}, \
                 n_vars {})",
                cost.stages_us,
                cost.bases_us,
                (a_bases.bytes()
                    + b_g1_bases.bytes()
                    + b_g2_bases.bytes()
                    + l_bases.bytes()
                    + h_bases.bytes())
                    / 1_000_000,
                cost.total_us,
                pk.domain_size,
                pk.n_vars,
            );
        }
        Ok((
            stages,
            (a_bases, b_g1_bases, b_g2_bases, l_bases, h_bases),
            cost,
        ))
    }

    /// Submits the key's queued uploads under a [`Seal`] and refuses a submission the GPU
    /// cut short, as `ProveError::Device` with [`crate::readback::is_aborted`] true.
    ///
    /// # Why the key needs a submission of its own
    ///
    /// `queue.write_buffer` only queues a copy, and wgpu flushes every queued copy at the
    /// head of the next submission. Without this, the next submission is the first proof's
    /// stages 0 to 4, so that submission carries 112 MB of key (`railgun-13x01`) in front
    /// of its pass, and a kill anywhere in it leaves the token missing and the retry
    /// re-uploading the witness and the parameters and nothing else: the key stays as the
    /// cut copy left it, every later token is present, and the proof is wrong. In about
    /// 900 `g16 prove` calls under load, each a fresh process, all 3 whose first submission
    /// was refused proved wrong (BUG-33), while 30 stage 0-4 refusals in one process with
    /// the key long resident retried exactly; `tests/abort_probe.rs` over 1,200 fresh
    /// circuits under the same load had 14 first submissions refused, and the 4 that then
    /// proved wrong had `H` zero at every entry and all five MSMs wrong, which is a gather
    /// over a CSR and MSMs over bases the cut copy never delivered. The epoch's copy is
    /// the last one queued, so a blit encoder cut anywhere before it leaves the token
    /// missing and this refuses it; [`WgpuProver::prepare`] then uploads the key again.
    pub async fn key_landed(&self) -> Result<(), ProveError> {
        let seal = Seal::new(&self.device, "g16 key upload")?;
        let mut enc =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("g16 key upload"),
                });
        let sealed = seal.close(&self.device, &mut enc);
        self.device.submit([enc.finish()]);
        self.device.wait_for_submitted_work().await?;
        if let Some(e) = self.device.take_error() {
            return Err(self
                .device
                .fault(format!("device error uploading the key: {e}")));
        }
        seal.verify(sealed).await
    }

    /// Queues the key's uploads again, into fresh buffers, after [`Self::key_landed`]
    /// refused them.
    pub fn reupload(&mut self) -> Result<(), ProveError> {
        let (stages, bases, cost) = Self::upload(&self.device, &self.pk)?;
        self.stages = stages;
        self.a_bases = bases.0;
        self.b_g1_bases = bases.1;
        self.b_g2_bases = bases.2;
        self.l_bases = bases.3;
        self.h_bases = bases.4;
        self.cost = cost;
        Ok(())
    }

    /// What [`WgpuProver::prepare`] cost for this key.
    pub fn prepare_cost(&self) -> CircuitCost {
        self.cost
    }

    /// Device bytes this key holds resident: the five base vectors.
    pub fn base_bytes(&self) -> u64 {
        self.a_bases.bytes()
            + self.b_g1_bases.bytes()
            + self.b_g2_bases.bytes()
            + self.l_bases.bytes()
            + self.h_bases.bytes()
    }

    pub fn stages(&self) -> &HStages {
        &self.stages
    }

    /// The key this circuit was prepared from. Inherent as well as on `PreparedCircuit`,
    /// because that trait is native only and stage 11 in the browser needs it.
    pub fn key(&self) -> &ProvingKey {
        &self.pk
    }

    pub fn n_vars(&self) -> usize {
        self.pk.n_vars
    }

    pub fn n_public(&self) -> usize {
        self.pk.n_public
    }

    pub fn domain_size(&self) -> usize {
        self.pk.domain_size
    }

    pub fn msm(&self) -> &MsmBatch {
        &self.msm
    }

    pub fn work(&self) -> Work {
        self.work
    }

    /// Stages 0 to 4 with stage 4's implementation chosen, for the test that runs both.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_h_with(
        &self,
        witness: &[Fr],
        mode: Stage4,
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        let _gpu = self.device.exclusive();
        pollster::block_on(self.stages.compute_h_with(&self.device, witness, mode, t))
    }

    /// Stages 0 to 4, awaited. The real implementation behind `PreparedCircuit::compute_h`.
    ///
    /// **Takes no lock, and the caller must.** [`crate::device::WgpuBackend::exclusive`] is
    /// what makes two concurrent proofs on one device produce right answers, and its own doc
    /// comment requires it to be held at the synchronous boundary rather than inside an
    /// `async fn`, so that no `MutexGuard` is live across an `.await` and these futures stay
    /// `Send`. The `PreparedCircuit` impl below takes it; the browser worker in
    /// `crate::wasm` runs one proof at a time by protocol and refuses a second instead.
    pub async fn compute_h_async(
        &self,
        witness: &[Fr],
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        self.stages.compute_h(&self.device, witness, t).await
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
}

/// Scalars that are neither 0 nor 1, over the whole witness and over the private tail.
///
/// One pass rather than two, and it is the only host work `msms` does on the witness: the
/// limbs are already on the device from stage 0. `crate::msm::DigitPlan` sizes both the
/// window width and `cap` from this count, and `cap` bounds a storage write that WebGPU drops
/// in silence when it goes out of range, so **an undercount loses entries with no error
/// anywhere**. Counting rather than estimating is not optional. At 140,824 variables this is
/// about 0.2 ms.
fn general_counts(witness: &[Fr], private_from: usize) -> (u32, u32) {
    let mut all = 0u32;
    let mut private = 0u32;
    for (i, x) in witness.iter().enumerate() {
        if !(x.is_zero() || x.is_one()) {
            all += 1;
            if i >= private_from {
                private += 1;
            }
        }
    }
    (all, private)
}

#[cfg(not(target_arch = "wasm32"))]
impl PreparedCircuit for WgpuCircuit {
    /// Always "wgpu", and that is a claim about what ran: no stage in this backend has a CPU
    /// fallback, so the name cannot be describing a proof the CPU did.
    fn backend_name(&self) -> &'static str {
        "wgpu"
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
        let _gpu = self.device.exclusive();
        retry_aborted("stages 0 to 4", t, |t| {
            pollster::block_on(self.compute_h_async(witness, t))
        })
    }

    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        // The guard first, then the clock, and the order is the whole point. The reason is
        // the long comment in `msms_async`, which is where the clock starts.
        // No `retry_aborted` here: the batch retries each of its own submissions, which is
        // shorter than the stage (`crate::batch`).
        let _gpu = self.device.exclusive();
        pollster::block_on(self.msms_async(witness, h, t))
    }

    fn h_to_host(&self, h: &HPoly) -> Option<Vec<Fr>> {
        let _gpu = self.device.exclusive();
        pollster::block_on(self.h_to_host_async(h))
    }
}

/// Attempts at stages 0 to 4 when the GPU abandoned their submission, counting the first.
/// The same count and backoff as `g16-metal`'s ceremony FFT and as `crate::batch` gives
/// each of the MSM submissions.
#[cfg(not(target_arch = "wasm32"))]
const ATTEMPTS: u32 = 4;

/// Runs `f`, and runs it again after a backoff when the GPU abandoned its submission.
///
/// An abandoned submission (`crate::readback::Seal`) is a scheduling event and not an
/// arithmetic one: macOS took the GPU back for the compositor. Stages 0 to 4 re-encode from
/// the witness upload, so a retry reads nothing half-written. `t` is restored between
/// attempts so a stage timing is the attempt that produced the answer. Native only, which
/// is why it sits on the synchronous `PreparedCircuit` impl and not on the `async fn`s the
/// browser awaits: there is no thread to sleep there, and the error reaches the page as it
/// is. Stages 5 to 9 are not wrapped in it: their pass was the one the kill landed on 75
/// times in 76, so `crate::batch` cuts it into short submissions and retries each on its
/// own, and a whole-stage retry on top would only multiply the attempts.
#[cfg(not(target_arch = "wasm32"))]
fn retry_aborted<T>(
    what: &str,
    t: &mut StageTimings,
    mut f: impl FnMut(&mut StageTimings) -> Result<T, ProveError>,
) -> Result<T, ProveError> {
    let before = *t;
    let mut attempt = 1;
    loop {
        match f(t) {
            Err(e) if is_aborted(&e) && attempt < ATTEMPTS => {
                let wait = std::time::Duration::from_millis(200 << attempt);
                eprintln!(
                    "wgpu: {what}: {e}; attempt {attempt} of {ATTEMPTS}, retrying in {} ms",
                    wait.as_millis()
                );
                *t = before;
                std::thread::sleep(wait);
                attempt += 1;
            }
            r => return r,
        }
    }
}

impl WgpuCircuit {
    /// Stages 5 to 9, awaited, under the circuit's own [`Work`]. The real implementation;
    /// the `PreparedCircuit::msms` above is `pollster::block_on` over this.
    pub async fn msms_async(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.msms_async_with(witness, h, self.work, t).await
    }

    /// Same, with the work mode chosen per call, which is how the browser worker asks for
    /// it: one circuit serves both modes there.
    ///
    /// Takes no lock, for the reason in [`Self::compute_h_async`]. The clock below still
    /// starts after the caller's guard is acquired, which is the ordering the long comment
    /// inside argues for.
    pub async fn msms_async_with(
        &self,
        witness: &[Fr],
        h: &HPoly,
        work: Work,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        if h.len() != self.pk.domain_size {
            return Err(bad(format!(
                "h has {} entries, the domain size is {}",
                h.len(),
                self.pk.domain_size
            )));
        }
        // The guard first, then the clock, and the order is the whole point.
        //
        // `compute_h` starts its clock inside `HStages::compute_h_with`, which is after the
        // guard, so it reports this proof's own work. With the clock started first here,
        // `msms` reported this proof's work **plus** however long it waited behind another
        // proof: at `js_16x16_d32` that is 900 ms of somebody else's MSM landing in a number
        // this repo publishes as `msm_us`, and the two halves of one proof were measuring
        // different things. `WgpuBackend::exclusive` exists partly so that a stage timing
        // never contains another proof's GPU time, and taking the timestamp outside it gave
        // exactly that back. The wait is real and it is a cost of the guard, but it is not
        // MSM time and this field is what `bench/results/history.csv` records.
        //
        // Pinned by `tests/proof.rs::the_msm_stage_timing_is_not_charged_for_another_proofs
        // _gpu_section`, which holds the guard for 500 ms and requires the reported number
        // not to grow by it. Single-threaded proving, which is every benchmark in this repo,
        // is unaffected: the guard is uncontended and the two orderings differ by the lock
        // acquisition.
        //
        // The guard itself is the caller's, one stack frame up, so that it is not held across
        // an `.await`. Acquiring it there and starting the clock here keeps the ordering.
        let start = Instant::now();

        let private_from = self.pk.n_public + 1;
        // Under constant work nothing on the host looks at a scalar's value either: the
        // plans take `n`, and `None` is what says so.
        let (g_all, g_private) = match work {
            Work::Variable => {
                let (all, private) = general_counts(witness, private_from);
                (Some(all), Some(private))
            }
            Work::Constant => (None, None),
        };
        let n_vars = self.pk.n_vars as u32;
        let l_len = self.l_bases.len() as u32;
        let domain = self.pk.domain_size as u32;

        // Four G1 jobs and one G2, in stage order, so the results come back as
        // [A, B-G2, B-G1, L, H] and `MsmOutputs` is filled positionally.
        let witness_jobs = [
            Job::G1 {
                bases: &self.a_bases,
                base_off: 0,
            },
            Job::G2 {
                bases: &self.b_g2_bases,
                base_off: 0,
            },
            Job::G1 {
                bases: &self.b_g1_bases,
                base_off: 0,
            },
        ];
        let l_jobs = [Job::G1 {
            bases: &self.l_bases,
            base_off: 0,
        }];
        let h_jobs = [Job::G1 {
            bases: &self.h_bases,
            base_off: 0,
        }];

        // The normal path: `H` is the buffer stage 4 wrote and the witness is the one stage 0
        // uploaded, so nothing crosses the host boundary and stage 9's scalars never exist in
        // host memory at all. `HPoly::device_handle` returns `None` for a handle from another
        // backend, so a Metal handle reaching here is a fallback rather than a reinterpreted
        // pointer.
        let handle = h.device_handle::<WgpuHandle>(crate::stages::TAG);
        let host_h = h.to_host();
        let (mont, groups) = match handle {
            Some(handle) => {
                if handle.len() != self.pk.domain_size {
                    return Err(bad(format!(
                        "device h holds {} entries, the domain size is {}",
                        handle.len(),
                        self.pk.domain_size
                    )));
                }
                let w = handle.witness_std();
                (
                    Some(MontConvert {
                        src: handle.witness_mont(),
                        dst: w,
                        n: n_vars,
                    }),
                    vec![
                        Group {
                            scalars: Source::Device {
                                buf: w,
                                general: g_all,
                            },
                            scalar_off: 0,
                            n: n_vars,
                            jobs: &witness_jobs,
                            work,
                        },
                        Group {
                            scalars: Source::Device {
                                buf: w,
                                general: g_private,
                            },
                            // Section 8 covers the private wires only: witness[0] is the
                            // constant 1 and witness[1..=n_public] are the public inputs,
                            // which the verifier folds in through IC instead.
                            scalar_off: private_from as u32,
                            n: l_len,
                            jobs: &l_jobs,
                            work,
                        },
                        Group {
                            scalars: Source::Device {
                                buf: handle.h_std(),
                                // Nobody on the host has seen these values, so the only safe
                                // answer is "all of them are general". Overestimating costs a
                                // wider window and a larger entry array; underestimating
                                // loses entries in silence. See `crate::batch::Source`.
                                general: None,
                            },
                            scalar_off: 0,
                            n: domain,
                            jobs: &h_jobs,
                            work,
                        },
                    ],
                )
            }
            None => {
                // A `compute_h` from another backend. Not the proving path, and the only
                // thing it is really for is letting a test hold stages 5 to 9 to the CPU's
                // stages 0 to 4 without the device's own H in the way.
                let host_h = host_h.ok_or_else(|| {
                    bad("compute_h output is neither a wgpu handle nor a host vector")
                })?;
                (
                    None,
                    vec![
                        Group {
                            scalars: Source::Host(witness),
                            scalar_off: 0,
                            n: n_vars,
                            jobs: &witness_jobs,
                            work,
                        },
                        Group {
                            scalars: Source::Host(witness),
                            scalar_off: private_from as u32,
                            n: l_len,
                            jobs: &l_jobs,
                            work,
                        },
                        Group {
                            scalars: Source::Host(host_h),
                            scalar_off: 0,
                            n: domain,
                            jobs: &h_jobs,
                            work,
                        },
                    ],
                )
            }
        };

        let out = self.msm.run(&self.device, mont, &groups).await?;
        if out.len() != 5 {
            return Err(bad(format!(
                "the MSM batch returned {} results for 5 jobs",
                out.len()
            )));
        }
        let outputs = MsmOutputs {
            a_g1: out[0].g1()?,
            b_g2: out[1].g2()?,
            b_g1: out[2].g1()?,
            l_g1: out[3].g1()?,
            h_g1: out[4].g1()?,
        };
        t.msm_us += start.elapsed().as_micros() as u64;
        Ok(outputs)
    }

    /// `H` copied down from the device, awaited. **Debugging only**: see
    /// [`g16_core::PreparedCircuit::h_to_host`], whose native impl is `block_on` over this,
    /// and which the browser cannot use because every readback here is asynchronous.
    ///
    /// `None` only for another backend's device handle, rather than a guess at how to read
    /// somebody else's pointer. A host `HPoly` is copied straight out: [`Self::msms_async`]
    /// accepts one, so it reaches here whenever stages 0 to 4 ran elsewhere.
    pub async fn h_to_host_async(&self, h: &HPoly) -> Option<Vec<Fr>> {
        if let Some(v) = h.to_host() {
            return Some(v.to_vec());
        }
        let handle = h.device_handle::<WgpuHandle>(crate::stages::TAG)?;
        handle.to_host(&self.device).await.ok()
    }

    /// Stages 5 to 9, one MSM at a time, for bisecting a device loss. Not a proving path.
    ///
    /// `msms_async` submits all five jobs in three groups, so when the device dies there is
    /// nothing to say which of them killed it. This runs exactly one and reports how long it
    /// took. It builds its own group rather than filtering that function's, so the proving
    /// path keeps no branch it does not need and cannot be broken by a change made for a
    /// bisection.
    ///
    /// Device-resident `H` only, because the browser is the only caller and the whole point
    /// is to reproduce what the browser does. The result is discarded: a single MSM is not a
    /// proof and this returns no proof, only whether the GPU survived it.
    pub async fn msm_probe_async(
        &self,
        witness: &[Fr],
        h: &HPoly,
        which: &str,
    ) -> Result<u64, ProveError> {
        self.check_witness(witness)?;
        let handle = h
            .device_handle::<WgpuHandle>(crate::stages::TAG)
            .ok_or_else(|| bad("msm_probe needs an H that is still on the device"))?;

        let private_from = self.pk.n_public + 1;
        let (g_all, g_private) = general_counts(witness, private_from);
        let n_vars = self.pk.n_vars as u32;
        let w = handle.witness_std();
        let mont = Some(MontConvert {
            src: handle.witness_mont(),
            dst: w,
            n: n_vars,
        });

        let jobs = [match which {
            "a-g1" => Job::G1 {
                bases: &self.a_bases,
                base_off: 0,
            },
            "b-g2" => Job::G2 {
                bases: &self.b_g2_bases,
                base_off: 0,
            },
            "b-g1" => Job::G1 {
                bases: &self.b_g1_bases,
                base_off: 0,
            },
            "l-g1" => Job::G1 {
                bases: &self.l_bases,
                base_off: 0,
            },
            "h-g1" => Job::G1 {
                bases: &self.h_bases,
                base_off: 0,
            },
            _ => {
                return Err(bad(format!(
                    "unknown msm probe {which:?}; expected one of a-g1, b-g2, b-g1, l-g1, h-g1"
                )))
            }
        }];

        // Each job reads a different range of a different scalar source, and getting that
        // wrong would measure a different circuit rather than a smaller one.
        let work = self.work;
        let group = match which {
            "l-g1" => Group {
                scalars: Source::Device {
                    buf: w,
                    general: Some(g_private),
                },
                scalar_off: private_from as u32,
                n: self.l_bases.len() as u32,
                jobs: &jobs,
                work,
            },
            "h-g1" => Group {
                scalars: Source::Device {
                    buf: handle.h_std(),
                    general: None,
                },
                scalar_off: 0,
                n: self.pk.domain_size as u32,
                jobs: &jobs,
                work,
            },
            _ => Group {
                scalars: Source::Device {
                    buf: w,
                    general: Some(g_all),
                },
                scalar_off: 0,
                n: n_vars,
                jobs: &jobs,
                work,
            },
        };

        let start = Instant::now();
        let out = self.msm.run(&self.device, mont, &[group]).await?;
        if out.len() != 1 {
            return Err(bad(format!(
                "the MSM batch returned {} results for 1 job",
                out.len()
            )));
        }
        Ok(start.elapsed().as_micros() as u64)
    }
}

/// `Backend` and `PreparedCircuit` both require `Send + Sync`, and the whole reason `prepare`
/// is a separate call is that one resident key is proved against from many threads. If a
/// `!Sync` field ever lands on either struct this stops compiling here rather than at the
/// `Box<dyn PreparedCircuit>` coercion, where the error names the trait object instead of the
/// field.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WgpuProver>();
    assert_send_sync::<WgpuCircuit>();
};
