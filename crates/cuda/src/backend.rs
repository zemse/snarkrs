//! CUDA implementation of `snarkrs_groth16::Backend`.
//!
//! Wiring only. Stages 0 to 4 live in [`crate::stages`] and stages 5 to 9 in
//! [`crate::msm`]; what this module decides is the split between witness-independent work
//! (all of it hoisted out of the proof) and per-proof work (which must touch nothing
//! shared and mutable). Same split as `snarkrs-metal`'s `backend.rs`, on purpose: the two
//! backends exist to be compared, and a comparison in which one of them hoists a cost the
//! other pays per proof is measuring the harness.
//!
//! # Where the preparation cost goes, and why it is split in two here
//!
//! On Metal the expensive half of preparation is two runtime MSL compiles, and the uploads
//! are `memcpy`s into unified memory. On CUDA both halves are expensive and they belong in
//! different places.
//!
//! * **Compiling.** NVRTC emits PTX and the driver runs ptxas again at `cuModuleLoad`, so
//!   a translation unit is compiled twice over before a single kernel runs. The MSM unit is
//!   where that lands, essentially all of it in the fully inlined `Fq2` curve arithmetic.
//!   Measured through [`Self::compile_us`] on the T4: **275.5 s with the driver's JIT cache
//!   disabled, 1.95 s with it warm**, against the 4m50s the kernel lane measured for an
//!   offline `nvcc -cubin` over the same source. The two numbers agree, and the gap between
//!   them is not our code: the driver keeps a PTX-to-cubin disk cache per user (`CUDA_CACHE_
//!   DISABLE=1` turns it off, which is how the cold figure above was taken), so the *first*
//!   process on a fresh machine or after a driver upgrade pays four and a half minutes and
//!   every process after it pays two seconds. A benchmark run must not be that first
//!   process, and a CI job with no persistent home directory always is.
//!
//!   Either way the cost depends on the *device*, not on the key, so it is paid once in
//!   [`CudaBackend::new`] and the resulting modules are handed to every circuit
//!   [`CudaBackend::prepare`] builds. A prover that constructs the backend per key is
//!   measuring ptxas; one that constructs it inside a benchmark's timed region is measuring
//!   nothing at all.
//! * **Uploading.** Five base vectors, two CSR matrices, two twiddle tables and the coset
//!   power table, all across PCIe. That cost depends on the key, so it is paid once per
//!   [`CudaBackend::prepare`] and never per proof. This is the half Metal barely has, and
//!   it is why the warm/cold distinction matters more on a discrete card.
//!
//! Set `G16_CUDA_PREPARE=1` to have both breakdowns printed to stderr rather than guessed
//! at, or read [`CudaCircuit::prepare_cost`] for the same numbers programmatically.
//!
//! # Why there is no CPU fallback path
//!
//! Every stage runs on the device. There is no size gate, and there is deliberately not
//! going to be one. The GPU does lose to the CPU at small domains, and on a discrete card
//! it loses by more than on unified memory, because the witness has to cross PCIe before
//! stage 0 can start. A gate that quietly ran the CPU below some threshold would mean
//! `g16 prove --backend cuda` reported a number belonging to a different backend, which is
//! precisely the failure this crate exists to avoid. [`CudaCircuit::backend_name`] returns
//! `"cuda"`, and that is a claim about what actually executed. The crossover is a
//! measurement to report, not a thing to hide.
//!
//! The one exception is not a fallback: `msms` accepts an `HPoly::Host`, so a `compute_h`
//! run on the CPU backend can still be finished on the GPU. That path exists for
//! cross-checks and is reached only when a caller deliberately mixes backends.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::CudaModule;
use snarkrs_field::{Field, Fr};
use snarkrs_formats::{Coefficients, HBasis, ProvingKey};
use snarkrs_groth16::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};

use crate::context::{bad, plan_lock, Cuda};
use crate::msm::{device_budget, too_big, CudaMsm, Work};
use crate::stages::{CudaStages, HHandle, TAG};

/// Where the time in [`CudaBackend::prepare`] went, in microseconds.
///
/// Kept on the circuit rather than printed, so a caller can report it without re-running
/// the preparation in order to time it.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrepareCost {
    /// [`CudaStages::from_module`]: both CSR matrices repacked and uploaded, both twiddle
    /// tables, the coset power table.
    pub stages_us: u64,
    /// The five base vectors repacked from ark's 72 and 136 byte affines and uploaded.
    /// This is the PCIe half, and at `js_16x16_d32` it is hundreds of megabytes.
    pub bases_us: u64,
    pub total_us: u64,
}

/// One device, one context, one stream, and both translation units already compiled.
///
/// Built once per process. Deliberately not `Clone`: the compiles are what make it
/// expensive, and [`Self::prepare`] shares them by handing each circuit an `Arc` of the
/// same module rather than by duplicating anything.
pub struct CudaBackend {
    work: Work,
    cuda: Cuda,
    stages_module: Arc<CudaModule>,
    /// The MSM unit for each [`Work`], variable then constant. The constructor compiles
    /// the one its work needs and [`Self::prepare`] the other on first use: only the
    /// constant-work unit carries the constant-work kernels (`kernels::unit_msm_with`).
    msm_modules: Mutex<[Option<Arc<CudaModule>>; 2]>,
    compile_us: u64,
    /// [`Self::with_vram_limit`], or `G16_CUDA_VRAM_LIMIT`.
    vram_limit: Option<u64>,
}

impl CudaBackend {
    /// Opt-in fixed-work proving, not constant time. See [`Work::Constant`].
    pub fn constant_work() -> Result<Self, ProveError> {
        Self::open(env_ordinal()?, Work::Constant)
    }

    /// Select constant work for circuits prepared after this call. The first such
    /// [`Self::prepare`] compiles the constant-work MSM unit, which [`Self::new`] does not.
    pub fn with_constant_work(mut self) -> Self {
        self.work = Work::Constant;
        self
    }

    /// Cap the device memory a circuit prepared after this call may plan against, below
    /// what the driver reports free: its resident key and one proof's stage scratch at
    /// prepare, a further proof's scratch, and each MSM batch (see
    /// [`CudaMsm::with_vram_limit`]). A key over it is refused at prepare and a proof whose
    /// MSMs do not fit at once runs them in shorter runs, each refusal a
    /// [`ProveError::DeviceMemory`] before anything is allocated. `G16_CUDA_VRAM_LIMIT`
    /// (bytes) sets the same for a backend that is opened without this call.
    pub fn with_vram_limit(mut self, bytes: u64) -> Self {
        self.vram_limit = Some(bytes);
        self
    }

    /// Opens device 0, or the ordinal in `G16_CUDA_DEVICE`, and compiles every kernel
    /// variable work launches.
    ///
    /// Fails on a machine with no usable NVIDIA device rather than falling back to the
    /// CPU, for the reason in the module docs.
    pub fn new() -> Result<Self, ProveError> {
        Self::with_ordinal(env_ordinal()?)
    }

    /// Same, on a caller-chosen device ordinal.
    ///
    /// Both kernel modules load into the one [`Cuda`] context, and every launch in either
    /// of them is issued on its one default stream. That is not incidental: a stream
    /// executes in issue order, and that is the entire mechanism ordering stage 4's write
    /// of `H` before stage 9's read of it. Two contexts, or two streams, and the handoff
    /// would need an event and would be a silent race without one.
    pub fn with_ordinal(ordinal: usize) -> Result<Self, ProveError> {
        Self::open(ordinal, Work::Variable)
    }

    fn open(ordinal: usize, work: Work) -> Result<Self, ProveError> {
        let cuda = open_device(ordinal)?;

        let t = Instant::now();
        // Two units rather than one. They share only the `Fr` prelude, and the MSM unit is
        // by far the larger; keeping them apart means a change to the NTT kernels does not
        // re-trigger the minutes-long ptxas pass over the G2 curve arithmetic, at the cost
        // of one extra NVRTC invocation.
        let stages_module = CudaStages::compile(&cuda)?;
        let msm_module = CudaMsm::compile_with(&cuda, work)?;
        let compile_us = t.elapsed().as_micros() as u64;

        if std::env::var_os("G16_CUDA_PREPARE").is_some() {
            let (major, minor) = cuda.compute_capability();
            eprintln!(
                "cuda backend: {} (sm_{major}{minor}, {} SMs), compile {:.2} ms",
                cuda.device_name(),
                cuda.sm_count(),
                compile_us as f64 / 1000.0,
            );
        }

        let mut msm_modules = [None, None];
        msm_modules[work as usize] = Some(msm_module);
        Ok(Self {
            work,
            cuda,
            stages_module,
            msm_modules: Mutex::new(msm_modules),
            compile_us,
            vram_limit: env_vram_limit()?,
        })
    }

    /// The MSM module for this backend's work, compiled here if the constructor compiled
    /// the other one, and shared by every circuit prepared after.
    fn msm_module(&self) -> Result<Arc<CudaModule>, ProveError> {
        let mut modules = self.msm_modules.lock().unwrap_or_else(|e| e.into_inner());
        let slot = &mut modules[self.work as usize];
        if let Some(m) = slot.as_ref() {
            return Ok(m.clone());
        }
        let m = CudaMsm::compile_with(&self.cuda, self.work)?;
        *slot = Some(m.clone());
        Ok(m)
    }

    /// The open context, for a caller that wants to allocate against the same device.
    pub fn cuda(&self) -> &Cuda {
        &self.cuda
    }

    pub fn device_name(&self) -> &str {
        self.cuda.device_name()
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.cuda.compute_capability()
    }

    pub fn sm_count(&self) -> i32 {
        self.cuda.sm_count()
    }

    /// What [`Self::with_ordinal`] spent on NVRTC plus the driver's ptxas, in microseconds.
    ///
    /// Paid once per process and never again, which is the only reason a number this large
    /// is acceptable. Expect about 2 s on a T4 with the driver's JIT cache warm and about
    /// 275 s without it; a benchmark harness that reports a cold number as if it were the
    /// steady state is reporting the CUDA installation, not this backend.
    pub fn compile_us(&self) -> u64 {
        self.compile_us
    }

    /// [`Backend::prepare`] without the box, for a caller that wants what only the CUDA
    /// circuit reports: [`CudaCircuit::prepare_cost`], [`CudaCircuit::last_msm_runs`] and
    /// the rest.
    pub fn prepare_circuit(&self, pk: ProvingKey) -> Result<CudaCircuit, ProveError> {
        CudaCircuit::new(self, pk)
    }
}

/// Device 0, or the ordinal in `G16_CUDA_DEVICE`.
fn env_ordinal() -> Result<usize, ProveError> {
    match std::env::var("G16_CUDA_DEVICE") {
        Ok(v) => v.trim().parse::<usize>().map_err(|e| {
            bad(format!(
                "G16_CUDA_DEVICE={v:?} is not a device ordinal: {e}"
            ))
        }),
        Err(_) => Ok(0),
    }
}

/// `G16_CUDA_VRAM_LIMIT`, in bytes, when it is set.
fn env_vram_limit() -> Result<Option<u64>, ProveError> {
    match std::env::var("G16_CUDA_VRAM_LIMIT") {
        Ok(v) => v.trim().parse::<u64>().map(Some).map_err(|e| {
            bad(format!(
                "G16_CUDA_VRAM_LIMIT={v:?} is not a byte count: {e}"
            ))
        }),
        Err(_) => Ok(None),
    }
}

/// Opens a device, or says why it could not.
///
/// A thin wrapper over [`Cuda::new`], which is where the interesting part lives: `cudarc`'s
/// dynamic loader panics rather than erroring when `libcuda` is absent, and `Cuda::new`
/// catches that so the signature means what it says. Without it there would be no way for
/// this backend to report "no device" and no way for `tests/pipeline_gpu.rs` to skip, since
/// the process would already be gone.
fn open_device(ordinal: usize) -> Result<Cuda, ProveError> {
    Cuda::new(ordinal).map_err(|e| bad(format!("no usable CUDA device {ordinal}: {e}")))
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        Ok(Box::new(self.prepare_circuit(pk)?))
    }
}

/// A key whose witness-independent data is device resident: both CSR matrices, both
/// twiddle tables, the coset power table and all five base vectors.
///
/// # Concurrency
///
/// `PreparedCircuit` promises one instance is safe to prove with from several threads, and
/// nothing here is mutated by a proof. The per-proof scratch stages 0 to 4 need is pooled
/// behind a mutex inside [`CudaStages`] and handed back when the handle inside the
/// [`HPoly`] drops, so two in-flight proofs cannot be given the same `A`, `B`, `C` or `H`
/// buffer. Stages 5 to 9 allocate their scratch per call out of the driver's own
/// stream-ordered pool and free it at the end, so there is nothing to share; but each call
/// plans that scratch against the free device memory, so the calls of concurrent proofs run
/// one at a time under the context's plan lock ([`Cuda::plan_memory`]), and a new stage
/// scratch and a prepare take the same lock.
///
/// What concurrent proofs do share is the stream. That is safe, because a stream executes
/// in issue order and the two proofs own disjoint buffers, so neither can observe the
/// other's writes. It does make the CUDA-event timings meaningless, though: an event
/// measures the stream rather than the call, so each proof's events also span whatever the
/// other queued in between. Concurrency is correct; concurrency plus trustworthy
/// `StageTimings` is not, and the benchmark runs proofs serially for that reason.
///
/// # What stays on the host
///
/// The dimensions, the five header points blinding and assembly use, and the full
/// verifying key (including `vk.ic`). The coefficients are released once stages 0 to 4
/// hold them on the device, and the five queries once the bases are resident, so
/// [`PreparedCircuit::key`] is not a complete key for preparing another backend.
pub struct CudaCircuit {
    pk: ProvingKey,
    stages: CudaStages,
    msm: CudaMsm,
    cost: PrepareCost,
    /// Points an MSM run of [`PreparedCircuit::h_and_msms_from_rows`] covers at most, see
    /// [`CudaMsm`]'s `msms_resident`.
    msm_chunk: usize,
}

/// [`CudaCircuit`]'s MSM run on the rows path: 2^22 points, so the largest batch's scratch
/// is 3 GB whatever the key, and a key below it runs each MSM whole.
pub const MSM_CHUNK: usize = 1 << 22;

impl CudaCircuit {
    fn new(backend: &CudaBackend, mut pk: ProvingKey) -> Result<Self, ProveError> {
        // Shape checks first. `crate::msm` would reject an out-of-range job later, but by
        // then the message names buffer offsets rather than the section of the zkey that is
        // the wrong length, and the mismatch is a property of the key, not of the proof.
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
        // A gnark key's H query is `domain_size - 1` bases for H's bit-reversed coefficients,
        // which only `h_and_msms_from_rows` computes.
        let want_h = pk.h_basis.bases(pk.domain_size);
        if pk.h_query.len() != want_h {
            let (got, n) = (pk.h_query.len(), pk.domain_size);
            return Err(bad(match pk.h_basis {
                HBasis::SnarkjsOddCoset => format!("h_query has {got} bases, domain size is {n}"),
                HBasis::GnarkBitReversed => format!(
                    "h_query has {got} bases, a gnark key on a domain of {n} needs {want_h}"
                ),
            }));
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

        // Everything this key keeps on the device and one proof's stage scratch, refused
        // before the first upload when it does not fit: a key too large for the card is
        // then a `DeviceMemory` refusal in milliseconds, not the driver's out-of-memory
        // after minutes of upload. The MSMs fit themselves to what is left, see
        // `CudaMsm::msms`. The plan lock is held through the uploads, so a proof on another
        // circuit of this backend does not plan against memory this key is about to take.
        let memory = backend.cuda.plan_memory();
        let _plan = plan_lock(&memory);
        let need = resident_bytes(&pk);
        let budget = device_budget(
            backend.cuda.context(),
            backend.cuda.stream(),
            backend.vram_limit,
        )?;
        if need > budget {
            return Err(too_big(
                need,
                budget,
                "this key's resident data and one proof's stage scratch",
            ));
        }

        let t_all = Instant::now();

        // Stages 0 to 4. This also validates the key the way the CPU backend does (domain a
        // power of two, CSR row count, every signal index below `n_vars`), so a malformed
        // key becomes an error here instead of an out-of-bounds device read later. On a GPU
        // that read is not a panic: it is either whatever the allocator left next to the
        // witness, or an illegal address that kills the context and takes every other
        // in-flight proof down with it.
        let t0 = Instant::now();
        let stages = CudaStages::from_module(&backend.cuda, backend.stages_module.clone(), &pk)?;
        let stages = if backend.work == Work::Constant {
            stages.constant_work()
        } else {
            stages
        };
        let stages = match backend.vram_limit {
            Some(bytes) => stages.with_vram_limit(bytes),
            None => stages,
        };
        release_coeffs(&mut pk);
        let stages_us = t0.elapsed().as_micros() as u64;

        // Stages 5 to 9. Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this
        // arkworks and `G2Affine` is 136, neither is `repr(C)`, and both carry an infinity
        // flag that the packed layout encodes as all-zero coordinates instead.
        let t1 = Instant::now();
        let msm = CudaMsm::from_module(&backend.cuda, backend.msm_module()?)?.with_key(&pk)?;
        release_queries(&mut pk);
        let msm = if backend.work == Work::Constant {
            msm.constant_work()
        } else {
            msm
        };
        let msm = match backend.vram_limit {
            Some(bytes) => msm.with_vram_limit(bytes),
            None => msm,
        };
        let bases_us = t1.elapsed().as_micros() as u64;

        let cost = PrepareCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        if std::env::var_os("G16_CUDA_PREPARE").is_some() {
            eprintln!(
                "cuda prepare: stages {} us, bases {} us, total {} us (domain {}, n_vars {n_vars})",
                cost.stages_us, cost.bases_us, cost.total_us, pk.domain_size,
            );
        }

        Ok(Self {
            pk,
            stages,
            msm,
            cost,
            msm_chunk: MSM_CHUNK,
        })
    }

    /// A circuit whose stages and bases the caller made resident, as [`crate::GnarkProver`]
    /// streams them from a dump. `pk` is the host half only: its queries may be empty,
    /// which [`PreparedCircuit::key`] allows.
    pub(crate) fn from_parts(
        pk: ProvingKey,
        stages: CudaStages,
        msm: CudaMsm,
        cost: PrepareCost,
        msm_chunk: usize,
    ) -> Self {
        Self {
            pk,
            stages,
            msm,
            cost,
            msm_chunk,
        }
    }

    /// What [`CudaBackend::prepare`] cost for this key. Excludes the compile, which is a
    /// property of the process and lives on [`CudaBackend::compile_us`].
    pub fn prepare_cost(&self) -> PrepareCost {
        self.cost
    }

    /// Microseconds the most recent witness upload spent crossing PCIe, device measured.
    ///
    /// Already counted inside `StageTimings::gather_us`; broken out because it is the one
    /// cost the unified-memory Metal backend does not pay at all, and putting the two
    /// backends' `gather_us` side by side without saying so would be comparing different
    /// things. Only meaningful when proofs are serial.
    pub fn last_witness_upload_us(&self) -> u64 {
        self.stages.last_witness_upload_us()
    }

    /// Device time of the last `msms` call, from a CUDA event pair. Excludes the host
    /// scalar pack, the upload and the readback, all of which `StageTimings::msm_us`
    /// includes.
    pub fn last_msm_device_us(&self) -> u64 {
        self.msm.last_device_us()
    }

    /// Batches the last MSM call ran, one when all five fit the device memory at once; see
    /// [`CudaBackend::with_vram_limit`].
    pub fn last_msm_runs(&self) -> u64 {
        self.msm.last_runs()
    }

    /// [`CudaBackend::with_vram_limit`] for this circuit's MSMs from here on, in place of
    /// the backend's.
    pub fn with_vram_limit(self, bytes: u64) -> Self {
        Self {
            msm: self.msm.with_vram_limit(bytes),
            ..self
        }
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

impl PreparedCircuit for CudaCircuit {
    /// Always [`TAG`] ("cuda"), and that is a claim about what ran: no stage in this backend
    /// has a CPU fallback, so the name cannot be describing a proof the CPU did.
    fn backend_name(&self) -> &'static str {
        TAG
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
        self.stages.compute_h(witness, t)
    }

    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        if self.pk.h_basis != HBasis::SnarkjsOddCoset {
            return Err(bad(
                "a gnark key's H basis is its bit-reversed coefficients, not the odd coset; \
                 prove it from A.w and B.w with `h_and_msms_from_rows`",
            ));
        }
        // Checked here rather than left to `msm_batch`'s bounds check, which only rejects
        // an `H` that is too *long*. A short one is in bounds against `h_query` and would
        // simply run: the missing tail would contribute nothing and the proof would fail
        // verification with no other symptom. Converting that into a message is the whole
        // point of the check.
        if h.len() != self.pk.domain_size {
            return Err(bad(format!(
                "h has {} entries, domain size is {}",
                h.len(),
                self.pk.domain_size
            )));
        }
        self.msm.msms(witness, h, t)
    }

    /// Stages 1 to 9 from the caller's rows, every stage on the device: the rows, the
    /// witness and nothing else cross PCIe, and `H` never leaves. The only way a gnark key
    /// proves; a zkey proves this way too, with `H` left on the odd coset.
    ///
    /// For a gnark key the H MSM runs over the first `n - 1` coefficients against `Z` and
    /// its result is multiplied by `-1/2`, which is `1 / t` on the odd coset; see
    /// `kernels/rows.cu`. The MSMs run in runs of at most [`MSM_CHUNK`] points.
    fn h_and_msms_from_rows(
        &self,
        witness: &[Fr],
        a_rows: &[Fr],
        b_rows: &[Fr],
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        let gnark = self.pk.h_basis == HBasis::GnarkBitReversed;
        let handle = self
            .stages
            .compute_h_from_rows(witness, a_rows, b_rows, gnark, t)?;
        let h_len = self.pk.h_basis.bases(self.pk.domain_size);
        let start = Instant::now();
        let mut out =
            self.msm
                .msms_resident(handle.witness_std(), handle.h_std(), h_len, self.msm_chunk)?;
        t.msm_us += start.elapsed().as_micros() as u64;
        if gnark {
            out.h_g1 *= -Fr::from(2u64).inverse().expect("2 is invertible mod r");
        }
        Ok(out)
    }

    /// The debugging seam only: a device `H` costs a full PCIe readback of the whole
    /// domain, which is the round trip [`HPoly`] exists to avoid.
    ///
    /// `None` only for another backend's device handle. A host `HPoly` is copied straight
    /// out, as the trait default does, because [`Self::msms`] accepts one and so it reaches
    /// here whenever stages 0 to 4 ran elsewhere.
    fn h_to_host(&self, h: &HPoly) -> Option<Vec<Fr>> {
        match h.require_handle::<HHandle>(TAG) {
            Ok(Some(handle)) => handle.to_host().ok(),
            Ok(None) => h.to_host().map(<[Fr]>::to_vec),
            Err(_) => None,
        }
    }
}

/// Device bytes a zkey's [`CudaCircuit`] takes before its first MSM: what
/// [`CudaStages::from_module`] and [`CudaMsm::with_key`] upload, and the scratch of one proof,
/// each buffer rounded up to the 2 MiB pages the driver maps it in.
fn resident_bytes(pk: &ProvingKey) -> u64 {
    const PAGE: u64 = 2 << 20;
    CudaStages::key_buffers(pk)
        .into_iter()
        .chain(CudaMsm::key_buffers(pk))
        .chain(CudaStages::scratch_buffers(pk.domain_size, pk.n_vars))
        .map(|b| b.div_ceil(PAGE) * PAGE)
        .sum()
}

/// Empties `pk`'s coefficients, which [`CudaStages`] has made resident and nothing reads
/// from the host again. Through [`snarkrs_groth16::malloc::release`], for the reason it
/// gives.
fn release_coeffs(pk: &mut ProvingKey) {
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
}

/// Empties `pk`'s five queries once [`CudaMsm::with_key`] has made them resident.
fn release_queries(pk: &mut ProvingKey) {
    snarkrs_groth16::malloc::release(std::mem::take(&mut pk.a_query));
    snarkrs_groth16::malloc::release(std::mem::take(&mut pk.b_g1_query));
    snarkrs_groth16::malloc::release(std::mem::take(&mut pk.b_g2_query));
    snarkrs_groth16::malloc::release(std::mem::take(&mut pk.l_query));
    snarkrs_groth16::malloc::release(std::mem::take(&mut pk.h_query));
}

/// `Backend` and `PreparedCircuit` both require `Send + Sync`, and the whole reason
/// `prepare` is a separate call is that one resident key is proved against from many
/// threads. If a `!Sync` field ever lands on either struct this stops compiling here
/// rather than at the `Box<dyn PreparedCircuit>` coercion, where the error names the trait
/// object instead of the field.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaBackend>();
    assert_send_sync::<CudaCircuit>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_formats::Matrix;

    /// `G16_ARTIFACTS` first: CARGO_MANIFEST_DIR is baked in at compile time and does not
    /// exist on the GPU box a cross-built test binary runs on.
    fn tiny_mul_zkey() -> std::path::PathBuf {
        std::env::var_os("G16_ARTIFACTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
            })
            .join("tiny_mul/circuit.zkey")
    }

    /// Free device memory once the stream has drained, which is when freed pages count.
    fn free_now(cuda: &Cuda) -> usize {
        cuda.stream().synchronize().unwrap();
        cuda.context().mem_get_info().unwrap().0
    }

    /// A key over the budget is refused as `DeviceMemory` with nothing allocated, a key at
    /// it prepares and proves, and a proof's first scratch over the stages' own budget is
    /// refused the same way. The budget is `with_vram_limit`, so this stands in for a key
    /// too large for the card (2^25 on a T4) at the cost of tiny_mul.
    #[test]
    #[ignore = "requires a CUDA device and tiny_mul/circuit.zkey under G16_ARTIFACTS or bench/artifacts"]
    fn a_key_over_the_vram_budget_is_refused_before_anything_is_allocated() {
        let path = tiny_mul_zkey();
        let pk = ProvingKey::load(&path).expect("tiny_mul key");
        let need = resident_bytes(&pk);
        let backend = CudaBackend::new()
            .expect("CUDA device")
            .with_vram_limit(need - 1);
        let before = free_now(backend.cuda());
        match backend.prepare_circuit(pk) {
            Err(ProveError::DeviceMemory { reason, .. }) => {
                assert!(reason.contains(&need.to_string()), "{reason}")
            }
            Err(e) => panic!("refused as {e}, not DeviceMemory"),
            Ok(_) => panic!(
                "prepared a key of {need} bytes under a limit of {}",
                need - 1
            ),
        }
        assert_eq!(free_now(backend.cuda()), before, "the refusal allocated");

        let backend = backend.with_vram_limit(need);
        let pk = ProvingKey::load(&path).unwrap();
        let witness: Vec<Fr> = (0..pk.n_vars as u64).map(Fr::from).collect();
        let circuit = backend
            .prepare_circuit(pk)
            .expect("a key that fits its budget");
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).unwrap();
        circuit.msms(&witness, &h, &mut t).unwrap();
        drop(h);

        let pk = ProvingKey::load(&path).unwrap();
        let scratch: u64 = CudaStages::scratch_buffers(pk.domain_size, pk.n_vars)
            .iter()
            .sum();
        let stages = CudaStages::from_module(&backend.cuda, backend.stages_module.clone(), &pk)
            .unwrap()
            .with_vram_limit(scratch - 1);
        let before = free_now(backend.cuda());
        match stages.compute_h(&witness, &mut t) {
            Err(ProveError::DeviceMemory { reason, .. }) => {
                assert!(reason.contains("stage scratch"), "{reason}")
            }
            Err(e) => panic!("refused as {e}, not DeviceMemory"),
            Ok(_) => panic!("allocated {scratch} bytes of scratch under a limit one less"),
        }
        assert_eq!(free_now(backend.cuda()), before, "the refusal allocated");
    }

    #[test]
    #[ignore = "requires a CUDA device and tiny_mul/circuit.zkey under G16_ARTIFACTS or bench/artifacts"]
    fn key_retention_releases_only_cuda_bulk_storage() {
        let path = tiny_mul_zkey();
        let expected = ProvingKey::load(&path).expect("tiny_mul key");
        assert!(!expected.a_query.is_empty());
        assert!(!expected.vk.ic.is_empty());
        let circuit = CudaBackend::new()
            .expect("CUDA device")
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

    /// The panic-to-error conversion is what makes "no GPU here" a skip rather than an
    /// abort, and it runs on every developer machine that has no NVIDIA card. On a machine
    /// that does have one this asserts the opposite: the device opens and has a name.
    ///
    /// [`open_device`] rather than [`CudaBackend::new`] on purpose. Opening a context is
    /// microseconds; compiling the MSM unit behind it is minutes on a T4, and this test
    /// has nothing to say about the compile. `tests/pipeline_gpu.rs` pays that cost once
    /// for the whole suite.
    ///
    /// Either outcome passes. What must never happen is the process dying, which is what
    /// an uncaught `dlopen` unwrap inside `cudarc` does.
    #[test]
    fn opening_a_device_never_panics() {
        match open_device(0) {
            Ok(cuda) => {
                let (major, minor) = cuda.compute_capability();
                assert!(!cuda.device_name().is_empty());
                eprintln!(
                    "cuda device 0: {} (sm_{major}{minor}, {} SMs)",
                    cuda.device_name(),
                    cuda.sm_count(),
                );
            }
            Err(e) => eprintln!("no cuda device on this machine: {e}"),
        }
    }

    /// An ordinal past the end of the device list must be an error, on a box with a GPU and
    /// on one without. Never a panic, and never a silent fall back to device 0, which would
    /// make `G16_CUDA_DEVICE=3` on a two-card box report the wrong card's numbers.
    #[test]
    fn an_ordinal_past_the_last_device_is_refused() {
        match open_device(4096) {
            Ok(_) => panic!("opened device ordinal 4096"),
            Err(e) => assert!(
                matches!(
                    e,
                    ProveError::Backend {
                        backend: "cuda",
                        ..
                    }
                ),
                "wrong error kind for a bad ordinal: {e}"
            ),
        }
    }
}
