//! CUDA implementation of `g16_core::Backend`.
//!
//! Wiring only. Stages 0 to 4 live in [`crate::stages`] and stages 5 to 9 in
//! [`crate::msm`]; what this module decides is the split between witness-independent work
//! (all of it hoisted out of the proof) and per-proof work (which must touch nothing
//! shared and mutable). Same split as `g16-metal`'s `backend.rs`, on purpose: the two
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

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaModule;
use g16_core::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::Fr;
use g16_zkey::ProvingKey;

use crate::context::Cuda;
use crate::msm::CudaMsm;
use crate::stages::CudaStages;

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "cuda",
        reason: reason.into(),
    }
}

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
    cuda: Cuda,
    stages_module: Arc<CudaModule>,
    msm_module: Arc<CudaModule>,
    compile_us: u64,
}

impl CudaBackend {
    /// Opens device 0, or the ordinal in `G16_CUDA_DEVICE`, and compiles every kernel.
    ///
    /// Fails on a machine with no usable NVIDIA device rather than falling back to the
    /// CPU, for the reason in the module docs.
    pub fn new() -> Result<Self, ProveError> {
        let ordinal = match std::env::var("G16_CUDA_DEVICE") {
            Ok(v) => v.trim().parse::<usize>().map_err(|e| {
                bad(format!(
                    "G16_CUDA_DEVICE={v:?} is not a device ordinal: {e}"
                ))
            })?,
            Err(_) => 0,
        };
        Self::with_ordinal(ordinal)
    }

    /// Same, on a caller-chosen device ordinal.
    ///
    /// Both kernel modules load into the one [`Cuda`] context, and every launch in either
    /// of them is issued on its one default stream. That is not incidental: a stream
    /// executes in issue order, and that is the entire mechanism ordering stage 4's write
    /// of `H` before stage 9's read of it. Two contexts, or two streams, and the handoff
    /// would need an event and would be a silent race without one.
    pub fn with_ordinal(ordinal: usize) -> Result<Self, ProveError> {
        let cuda = open_device(ordinal)?;

        let t = Instant::now();
        // Two units rather than one. They share only the `Fr` prelude, and the MSM unit is
        // by far the larger; keeping them apart means a change to the NTT kernels does not
        // re-trigger the minutes-long ptxas pass over the G2 curve arithmetic, at the cost
        // of one extra NVRTC invocation.
        let stages_module = CudaStages::compile(&cuda)?;
        let msm_module = CudaMsm::compile(&cuda)?;
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

        Ok(Self {
            cuda,
            stages_module,
            msm_module,
            compile_us,
        })
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
        Ok(Box::new(CudaCircuit::new(self, pk)?))
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
/// stream-ordered pool and free it at the end, so there is nothing to share.
///
/// What concurrent proofs do share is the stream. That is safe, because a stream executes
/// in issue order and the two proofs own disjoint buffers, so neither can observe the
/// other's writes. It does make the CUDA-event timings meaningless, though: an event
/// measures the stream rather than the call, so each proof's events also span whatever the
/// other queued in between. Concurrency is correct; concurrency plus trustworthy
/// `StageTimings` is not, and the benchmark runs proofs serially for that reason.
pub struct CudaCircuit {
    pk: ProvingKey,
    stages: CudaStages,
    msm: CudaMsm,
    cost: PrepareCost,
}

impl CudaCircuit {
    fn new(backend: &CudaBackend, pk: ProvingKey) -> Result<Self, ProveError> {
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

        // Stages 0 to 4. This also validates the key the way the CPU backend does (domain a
        // power of two, CSR row count, every signal index below `n_vars`), so a malformed
        // key becomes an error here instead of an out-of-bounds device read later. On a GPU
        // that read is not a panic: it is either whatever the allocator left next to the
        // witness, or an illegal address that kills the context and takes every other
        // in-flight proof down with it.
        let t0 = Instant::now();
        let stages = CudaStages::from_module(&backend.cuda, backend.stages_module.clone(), &pk)?;
        let stages_us = t0.elapsed().as_micros() as u64;

        // Stages 5 to 9. Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this
        // arkworks and `G2Affine` is 136, neither is `repr(C)`, and both carry an infinity
        // flag that the packed layout encodes as all-zero coordinates instead.
        let t1 = Instant::now();
        let msm = CudaMsm::from_module(&backend.cuda, backend.msm_module.clone())?.with_key(&pk)?;
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
        })
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
    /// Always "cuda", and that is a claim about what ran: no stage in this backend has a
    /// CPU fallback, so the name cannot be describing a proof the CPU did.
    fn backend_name(&self) -> &'static str {
        "cuda"
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
