//! Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
//! `H = A*B - C`, with the three domain vectors resident in device memory throughout.
//!
//! Host twin of `crates/g16-metal/src/stages.rs`, running the identical schedule against
//! `kernels/gather.cu`, `kernels/ntt.cu` and `kernels/pointwise.cu`. Same pass split, same
//! fusion, same argument block, so a Metal-versus-CUDA number compares two implementations
//! of one algorithm rather than two algorithms.
//!
//! # What this module is for
//!
//! [`g16_core::PreparedCircuit::compute_h`] is a stage *group*, not a primitive. A trait
//! with an `ntt()` method would force the three vectors back through host memory between
//! every one of the six transforms, and on a discrete card that is not a cache miss, it is
//! a PCIe round trip in both directions. Everything below runs as one queue of launches on
//! one stream with a single synchronize at the end, and the result never leaves the
//! device: the value handed back is [`g16_core::HPoly::Device`] carrying an [`HHandle`],
//! and stage 9's MSM reads the buffer stage 4 wrote.
//!
//! # Launch budget, and why it is a smaller worry here than on Metal
//!
//! The Metal backend batches stages 0 to 4 into one command buffer because a commit and
//! wait costs 0.15 to 0.19 ms on the M2 Max and does not shrink with the work, fifty to a
//! hundred times more than encoding one more dispatch. CUDA has no equivalent object: a
//! launch on an already-warm stream is a few microseconds of host time and is
//! asynchronous, so the fourteen launches of a 2^18 proof queue up without the host
//! blocking, and the single `synchronize` at the end is the only wait. The thing to avoid
//! on this side is not a submission but a *synchronize* in the middle, which is why the
//! per-stage timings below come from CUDA events rather than from a wall clock wrapped
//! around each group.
//!
//! # Where the host still does work per proof
//!
//! Packing the witness, and then shipping it. `ark_ff::Fp` is not `repr(C)`, so nothing
//! can be byte-cast (see `g16_gpu_layout`), and the witness is the one per-proof vector
//! that has to be repacked. On Apple silicon the repack writes straight into the
//! `MTLBuffer` and that write *is* the upload. Here it is only the host half: the packed
//! bytes still have to cross PCIe. Both halves are measured and both are charged to
//! `gather_us`; see [`CudaStages::compute_h`].
//!
//! Everything else the kernels read (the CSR rows, both twiddle tables, the coset power
//! table) is witness independent, is uploaded once in [`CudaStages::new`], and stays
//! resident. A prover that rebuilt this per proof would be measuring its own key upload.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use g16_core::{HPoly, ProveError, StageTimings};
use g16_field::{Domain, Field, Fr};
use g16_gpu_layout::{PackedFr, PackedScalar};
use g16_zkey::ProvingKey;

use crate::{as_words, from_words, kernels, Cuda};

/// The tag on [`g16_core::HPoly::Device`] values produced here. A handle carrying any
/// other tag came from a different backend and must not be dereferenced as an [`HHandle`].
pub const TAG: &str = "cuda";

/// Threads per block, clamped down by the device limit and by how much work there is.
///
/// 256 for the same reason the Metal twin prefers 256: it is eight warps, enough to hide
/// the dependent load in the gather and to keep several NTT butterflies in flight, without
/// being so wide that a small batch leaves most of the block idle.
///
/// Unlike Metal there is no per-kernel maximum to clamp against. Metal reports
/// `maxTotalThreadsPerThreadgroup` for each compiled pipeline and rejects a larger
/// threadgroup; CUDA's equivalent limit is the register budget, and at 256 threads it
/// cannot bind. A block of 256 threads using the hardware maximum of 255 registers per
/// thread needs 65280 of the 65536 registers a block may have, so 256 is a legal launch
/// for these kernels whatever the compiler decides to allocate.
const PREFERRED_BLOCK: u32 = 256;

/// Hard cap on NTT passes fused into one launch, independent of the memory budget.
///
/// 2^10 elements at 32 bytes is 32768 bytes. The real cap is recomputed from the device's
/// own shared-memory-per-block attribute in [`CudaStages::new`]; this only bounds it. See
/// the budget section at the top of `kernels/ntt.cu` for the derivation, which is done for
/// CUDA's 48 KB default rather than copied from Metal's 32 KB threadgroup limit and lands
/// on the same number.
const MAX_FUSED_PASSES: u32 = 10;

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "cuda",
        reason: reason.into(),
    }
}

/// Wraps a driver or NVRTC failure with the operation that produced it. The driver's own
/// message is `CUDA_ERROR_ILLEGAL_ADDRESS` and nothing else, so without the `what` there is
/// no way to tell which of fourteen launches died.
fn drv(what: &str, e: impl std::fmt::Display) -> ProveError {
    bad(format!("{what}: {e}"))
}

// ---------------------------------------------------------------------------
// The kernel argument block
// ---------------------------------------------------------------------------

/// Mirrors `struct NttParams` in `kernels/ntt.cu`. 52 bytes, no padding on either side.
/// The CUDA side carries a `static_assert` on its size and this carries a `const`
/// assertion, so the two cannot drift into a silently misread argument block.
///
/// Passed **by value**. The PTX shows `.param .align 4 .b8 [52]`, meaning the driver copies
/// these 52 bytes into parameter space verbatim, so the Rust layout has to match byte for
/// byte and a pointer would be read as five `u32` of address bits.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NttParams {
    log_n: u32,
    s0: u32,
    k: u32,
    scale_mode: u32,
    store_mode: u32,
    kscale: PackedFr,
}

const _: () = assert!(core::mem::size_of::<NttParams>() == 52);

// SAFETY: `DeviceRepr` marks a type that may be handed to a kernel as a by-value argument.
// `NttParams` is `repr(C)`, is five `u32` followed by a `PackedFr` (itself `repr(C)`, eight
// `u32`, no padding, every bit pattern valid), and its size is asserted above to be exactly
// the 52 bytes the kernel's parameter slot expects.
unsafe impl DeviceRepr for NttParams {}

const SCALE_NONE: u32 = 0;
const SCALE_CONST: u32 = 1;
const SCALE_TABLE: u32 = 2;
const STORE_PLAIN: u32 = 0;
const STORE_JOIN: u32 = 1;

/// One batch of NTT passes: `k` consecutive passes starting at pass `s0`.
#[derive(Clone, Copy, Debug)]
struct Batch {
    s0: u32,
    k: u32,
}

/// Splits `log_n` passes into as few batches as the shared-memory budget allows, sized as
/// evenly as possible.
///
/// Evenly, not greedily, and the argument is the one the Metal backend makes. A greedy fill
/// at 2^18 with a cap of 10 gives 10 + 8, where the second launch has only 2^8 blocks doing
/// 2^17 butterflies; an even 9 + 9 gives both launches 2^9 blocks. The pathological version
/// is what profiling of bellperson caught at 2^26, where the leftover kernel launched 16
/// million blocks of two threads each. On CUDA there is a second reason to prefer the even
/// split: a 9-pass batch is 16 KB of shared memory and lets four blocks sit on a Turing
/// SM's 64 KB, where a 10-pass batch is 32 KB and caps residency at two.
fn split_passes(log_n: u32, max_fused: u32) -> Vec<Batch> {
    if log_n == 0 {
        // A one-point domain has no butterflies at all, but the head still runs so the
        // load-time scale and the store epilogue happen.
        return vec![Batch { s0: 0, k: 0 }];
    }
    let batches = log_n.div_ceil(max_fused.max(1));
    let base = log_n / batches;
    let rem = log_n % batches;
    let mut out = Vec::with_capacity(batches as usize);
    let mut s0 = 0;
    for i in 0..batches {
        let k = base + u32::from(i < rem);
        out.push(Batch { s0, k });
        s0 += k;
    }
    out
}

// ---------------------------------------------------------------------------
// Per-proof scratch
// ---------------------------------------------------------------------------

/// The five events that time one proof. Created with the scratch and re-recorded on every
/// proof rather than allocated per proof: `cuEventCreate` is cheap but not free, and a
/// timing apparatus that shows up in the timing is worthless.
///
/// `CU_EVENT_DEFAULT`, not `new_event(None)`, whose default is `CU_EVENT_DISABLE_TIMING`.
/// An event created with timing disabled records fine and then fails at `elapsed_ms`, which
/// would turn a measurement into an error at the end of an otherwise correct proof.
struct Events {
    start: CudaEvent,
    upload: CudaEvent,
    gather: CudaEvent,
    ntt: CudaEvent,
    join: CudaEvent,
}

impl Events {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self, ProveError> {
        let ev = || {
            ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| drv("cuEventCreate", e))
        };
        Ok(Self {
            start: ev()?,
            upload: ev()?,
            gather: ev()?,
            ntt: ev()?,
            join: ev()?,
        })
    }
}

/// Per-proof scratch: the pinned staging buffer for the witness, its device copy, the three
/// domain vectors, one transform temporary, the two H outputs and the timing events.
///
/// Pooled rather than allocated per proof, for two reasons that are both stronger here than
/// on Metal. `cuMemAlloc` is a synchronizing call on most drivers, so a fresh allocation
/// inside a proof drains the stream; and `cuMemAllocHost` for the pinned staging has to
/// page-lock and register the memory, which is milliseconds for the megabytes this needs.
/// Pooling also keeps `compute_h` safe to call concurrently on one circuit, which the
/// `PreparedCircuit` contract requires: each in-flight proof holds its own set and no two
/// proofs ever share a buffer.
///
/// Every device buffer is `CudaSlice<u32>`. `cudarc`'s `DeviceRepr` is a foreign trait and
/// `PackedFr` a foreign type, so packed data crosses as words via `as_words`/`from_words`;
/// the kernels' `Fr*` parameters are the same 32 bytes either way, since a device pointer
/// carries no element type by the time it reaches the launch.
struct Scratch {
    /// Page-locked host staging for the witness. Written by the packer, DMA'd from.
    host_witness: PinnedHostSlice<u32>,
    witness: CudaSlice<u32>,
    a: CudaSlice<u32>,
    b: CudaSlice<u32>,
    c: CudaSlice<u32>,
    t: CudaSlice<u32>,
    h_mont: CudaSlice<u32>,
    h_std: CudaSlice<u32>,
    /// Bound to the kernel parameters the current mode never dereferences. See
    /// [`CudaStages::launch_head`].
    unused: CudaSlice<u32>,
    ev: Events,
}

type Pool = Arc<Mutex<Vec<Scratch>>>;

// ---------------------------------------------------------------------------
// The resident, witness-independent state
// ---------------------------------------------------------------------------

/// Everything stages 0 to 4 need, compiled and uploaded once per proving key.
///
/// Holds the module and the four kernel handles, the two CSR matrices, both twiddle tables,
/// the coset power table, and the scratch pool. Constructing this is the whole of the
/// warm/cold distinction: NVRTC compiling the stages unit and the key upload are both
/// prepare-time costs and neither may be paid again per proof.
pub struct CudaStages {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Kept alive explicitly. `CudaFunction` holds an `Arc<CudaModule>` internally so this
    /// is not strictly load bearing, but a caller that wants another kernel out of the same
    /// unit should not have to recompile it.
    module: Arc<CudaModule>,
    gather: CudaFunction,
    head: CudaFunction,
    tail: CudaFunction,
    h_join: CudaFunction,

    /// Largest number of NTT passes one launch can fuse, from the device's own shared
    /// memory attribute rather than hardcoded.
    max_fused: u32,
    /// Threads per block: [`PREFERRED_BLOCK`] clamped by the device limit.
    block: u32,

    domain: Domain,
    n_vars: usize,
    /// snarkjs' `inc`, a primitive 2n-th root of unity. Not `Domain::coset_gen`; see
    /// `g16_core::cpu::CpuCircuit::new` for the argument, which is a contract with the
    /// section 9 bases in the zkey and not a free choice.
    coset_shift: Fr,
    batches: Vec<Batch>,

    row_ptr: [CudaSlice<u32>; 2],
    signal: [CudaSlice<u32>; 2],
    value: [CudaSlice<u32>; 2],
    tw_fwd: CudaSlice<u32>,
    tw_inv: CudaSlice<u32>,
    coset_pows: CudaSlice<u32>,

    pool: Pool,
    /// Microseconds the most recent witness upload spent on the wire. Broken out because
    /// `StageTimings` has no field for it and folding it into `gather_us` without saying so
    /// would hide the one cost the unified-memory Metal backend does not pay at all.
    last_upload_us: AtomicU64,
}

impl CudaStages {
    /// Compiles `kernels::unit_stages()` and uploads everything witness independent.
    ///
    /// The compile is the expensive half: NVRTC over the gather plus NTT plus pointwise
    /// unit is tens of milliseconds, comparable to a whole proof at the small domains. It
    /// belongs here and nowhere else.
    pub fn new(cuda: &Cuda, pk: &ProvingKey) -> Result<Self, ProveError> {
        Self::from_module(cuda, Self::compile(cuda)?, pk)
    }

    /// Compile the stages translation unit on its own, so a process that prepares several
    /// keys pays NVRTC and the driver's ptxas once. Twin of [`crate::msm::CudaMsm::compile`],
    /// and the same argument, though the numbers here are far smaller: the stages unit has
    /// no `Fq2` curve arithmetic to inline.
    pub fn compile(cuda: &Cuda) -> Result<Arc<CudaModule>, ProveError> {
        cuda.compile("stages", &kernels::unit_stages())
            .map_err(|e| bad(e.to_string()))
    }

    /// Upload one key against an already-compiled module.
    ///
    /// `module` must have come from [`Self::compile`] against this same [`Cuda`]; see
    /// [`crate::msm::CudaMsm::from_module`] for why mixing contexts fails late rather than
    /// here.
    pub fn from_module(
        cuda: &Cuda,
        module: Arc<CudaModule>,
        pk: &ProvingKey,
    ) -> Result<Self, ProveError> {
        let func = |name: &str| -> Result<CudaFunction, ProveError> {
            module
                .load_function(name)
                .map_err(|e| bad(format!("kernel {name} missing: {e}")))
        };
        let gather = func("g16_gather_abc")?;
        let head = func("g16_ntt_head")?;
        let tail = func("g16_ntt_tail")?;
        let h_join = func("g16_h_join")?;

        let ctx = cuda.context().clone();
        let stream = cuda.stream().clone();

        // Elements of 32 bytes that fit in one block's shared memory, as a power of two.
        //
        // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, and deliberately NOT
        // ..._MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. The first is the 48 KB every architecture
        // from Volta up gives a block for free; the second is the 64 KB (Turing) to 163 KB
        // (A100) ceiling that needs a per-function cudaFuncSetAttribute opt in. Taking that
        // opt in would buy one extra fused pass, which at 2^18 turns an 18 = 9 + 9 split
        // into 18 = 9 + 9, in exchange for an architecture-dependent limit the host would
        // have to get right on every card. 48 KB holds 1536 Fr, and the largest power of
        // two at or below that is 1024, so k <= 10. Querying beats hardcoding 49152, which
        // is a device property written down as a constant.
        let budget = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
            .map_err(|e| drv("query shared memory per block", e))? as usize;
        let elems = budget / core::mem::size_of::<PackedFr>();
        if elems < 2 {
            return Err(bad(format!(
                "shared memory per block is {budget} bytes, which holds fewer than two Fr"
            )));
        }
        let max_fused = (usize::BITS - 1 - elems.leading_zeros() as u32).min(MAX_FUSED_PASSES);

        let max_threads = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|e| drv("query max threads per block", e))? as u32;
        let block = PREFERRED_BLOCK.min(max_threads).max(1);

        let domain = Domain::new(pk.domain_size).map_err(|e| bad(e.to_string()))?;
        if domain.size != pk.domain_size {
            return Err(bad(format!(
                "domain size {} is not a power of two",
                pk.domain_size
            )));
        }
        // Every kernel indexes with a 32-bit `gid` and a grid dimension is a u32. A domain
        // that did not fit would not fail, it would wrap and gather the wrong rows.
        if u32::try_from(domain.size).is_err() || u32::try_from(pk.n_vars).is_err() {
            return Err(bad(format!(
                "domain {} / n_vars {} exceeds the u32 indexing the kernels use",
                domain.size, pk.n_vars
            )));
        }

        // Same derivation as the CPU backend, and it must stay the same: the coset is fixed
        // by the zkey's section 9 bases, so any other shift pairs the evaluations against
        // the wrong Lagrange polynomials.
        let coset_shift = Domain::new(
            domain
                .size
                .checked_mul(2)
                .ok_or_else(|| bad("domain size overflows"))?,
        )
        .map_err(|e| bad(format!("no 2n-th root of unity: {e}")))?
        .group_gen;

        for (m, name) in [(0usize, "A"), (1usize, "B")] {
            if pk.coeffs.row_ptr[m].len() != domain.size + 1 {
                return Err(bad(format!(
                    "matrix {name} has {} rows, domain size is {}",
                    pk.coeffs.row_ptr[m].len().saturating_sub(1),
                    domain.size
                )));
            }
            // Checked once per key rather than per proof. On the GPU an out-of-range signal
            // index is not a panic: it is either a read of whatever the allocator left after
            // the witness, or an illegal address that kills the context and takes every
            // other in-flight proof with it. This check is the only thing standing between a
            // malformed key and one of those two.
            if pk.coeffs.signal[m].iter().any(|&s| s as usize >= pk.n_vars) {
                return Err(bad(format!(
                    "matrix {name} references a signal beyond n_vars {}",
                    pk.n_vars
                )));
            }
        }

        // shift^j for j in [0, n). Not the twiddles: the twiddles are powers of the domain's
        // own root of unity and this is a primitive 2n-th root, so no table g16-field builds
        // can be reused. A running product, one multiply per entry.
        let mut pows = Vec::with_capacity(domain.size);
        let mut acc = Fr::ONE;
        for _ in 0..domain.size {
            pows.push(acc);
            acc *= coset_shift;
        }

        // Bound to locals before the struct literal, because the closures borrow `stream`
        // and the struct moves it.
        let up_u32 = |data: &[u32]| upload_u32(&stream, data);
        let up_fr = |data: &[Fr]| upload_u32(&stream, as_words(&PackedFr::pack_slice(data)));
        let row_ptr = [
            up_u32(&pk.coeffs.row_ptr[0])?,
            up_u32(&pk.coeffs.row_ptr[1])?,
        ];
        let signal = [up_u32(&pk.coeffs.signal[0])?, up_u32(&pk.coeffs.signal[1])?];
        let value = [up_fr(&pk.coeffs.value[0])?, up_fr(&pk.coeffs.value[1])?];
        let tw_fwd = up_fr(&domain.twiddles())?;
        let tw_inv = up_fr(&domain.twiddles_inv())?;
        let coset_pows = up_fr(&pows)?;

        Ok(Self {
            row_ptr,
            signal,
            value,
            tw_fwd,
            tw_inv,
            coset_pows,

            n_vars: pk.n_vars,
            batches: split_passes(domain.log_size, max_fused),
            coset_shift,
            domain,

            ctx,
            stream,
            module,
            gather,
            head,
            tail,
            h_join,
            max_fused,
            block,
            pool: Arc::new(Mutex::new(Vec::new())),
            last_upload_us: AtomicU64::new(0),
        })
    }

    pub fn domain_size(&self) -> usize {
        self.domain.size
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn coset_shift(&self) -> Fr {
        self.coset_shift
    }

    /// The stream stages 0 to 4 run on. The MSM stage takes this so H needs no ordering
    /// beyond the synchronize `compute_h` already does.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The compiled stages unit, for a caller that wants another kernel out of it.
    pub fn module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    /// Passes fused into one launch. 10 on every card from Volta up.
    pub fn max_fused_passes(&self) -> u32 {
        self.max_fused
    }

    /// Kernel launches issued by one call to [`Self::compute_h`], for reporting.
    pub fn launch_count(&self) -> usize {
        1 + 6 * self.batches.len()
    }

    /// The `(s0, k)` pass split, for reporting.
    pub fn pass_batches(&self) -> Vec<(u32, u32)> {
        self.batches.iter().map(|b| (b.s0, b.k)).collect()
    }

    /// Microseconds the most recent witness upload spent crossing PCIe, GPU measured.
    ///
    /// Also included in `StageTimings::gather_us`; this is the same number broken out, so a
    /// benchmark can say how much of the CUDA backend's stage 0 is a bus cost. Only
    /// meaningful when proofs are serial, since concurrent proofs on one circuit overwrite
    /// each other's value.
    pub fn last_witness_upload_us(&self) -> u64 {
        self.last_upload_us.load(Ordering::Relaxed)
    }

    /// Stages 0 to 4. Returns `H` on the coset, length `domain_size`, still on the device.
    ///
    /// # Timing attribution
    ///
    /// CUDA events, not a wall clock. An event is recorded into the stream between launches
    /// and costs nothing to place, so the three stage groups get separate device-side
    /// measurements without a single mid-pipeline synchronize. The Metal backend cannot do
    /// this: its only measurement is a CPU clock around a command buffer commit and wait,
    /// so a split there costs two extra 0.15 ms round trips and the loss of the stage 4
    /// fusion, which is why it is opt in there and always on here.
    ///
    /// * `gather_us` is the host repack, plus the PCIe upload, plus the gather kernel. The
    ///   upload is stage 0's input arriving and is charged here rather than dropped;
    ///   [`Self::last_witness_upload_us`] breaks it out.
    /// * `ntt_us` is the twelve transform launches, and in the default fused path it also
    ///   contains stage 4, which rides out on the store epilogue of the last one.
    /// * `pointwise_us` is therefore **0** in the default path. Stage 4 is not a launch and
    ///   not a separable region of one, so splitting it out would mean inventing a number.
    ///   Setting `G16_CUDA_UNFUSED=1` disables the fusion and runs the standalone
    ///   `g16_h_join` kernel, which gives `pointwise_us` a real value and costs a full write
    ///   of C plus a full read of A, B and C. That mode exists to check the fused path
    ///   against the unfused reference, not to make the table look complete.
    ///
    /// One caveat that comes with events rather than with a wall clock: an event measures
    /// the *stream*, not this call. Two proofs in flight on one circuit share the default
    /// stream, so each one's events also span whatever the other queued in between. The
    /// results stay correct (the two proofs own disjoint scratch), but the timings are only
    /// meaningful when proofs are run serially, which is how the benchmark runs them.
    pub fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size;
        let mut sc = self.take_scratch()?;

        // The repack, straight into page-locked staging. On Metal this write is the whole
        // upload; here it is only the host half of it.
        let start = Instant::now();
        {
            let words = sc
                .host_witness
                .as_mut_slice()
                .map_err(|e| drv("map pinned witness staging", e))?;
            // SAFETY: `host_witness` was allocated with exactly `n_vars * 8` u32, the driver
            // returns a page-aligned pointer where `PackedFr` needs only 4-byte alignment,
            // `PackedFr` is eight `u32` with no padding and no invalid bit patterns, and
            // nothing else holds a reference to this buffer (the scratch is out of the pool,
            // and a scratch only returns to the pool after the stream was synchronized).
            let dst = unsafe {
                core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<PackedFr>(), self.n_vars)
            };
            PackedFr::pack_into(witness, dst);
        }
        let pack_us = start.elapsed().as_micros() as u64;

        let fused = std::env::var_os("G16_CUDA_UNFUSED").is_none();

        sc.ev
            .start
            .record(&self.stream)
            .map_err(|e| drv("record start event", e))?;
        // Asynchronous precisely because the source is pinned. From pageable memory the
        // driver stages through a buffer of its own and blocks the host, and the event pair
        // around it would then be measuring a host stall rather than the wire.
        self.stream
            .memcpy_htod(&sc.host_witness, &mut sc.witness)
            .map_err(|e| drv("upload witness", e))?;
        sc.ev
            .upload
            .record(&self.stream)
            .map_err(|e| drv("record upload event", e))?;

        self.launch_gather(&sc, n)?;
        sc.ev
            .gather
            .record(&self.stream)
            .map_err(|e| drv("record gather event", e))?;

        self.launch_transforms(&sc, n, fused)?;
        sc.ev
            .ntt
            .record(&self.stream)
            .map_err(|e| drv("record ntt event", e))?;

        if !fused {
            self.launch_h_join(&sc, n)?;
            sc.ev
                .join
                .record(&self.stream)
                .map_err(|e| drv("record join event", e))?;
        }

        // The one wait. Everything above was queued without the host blocking.
        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize stages 0-4", e))?;

        let upload_us = elapsed_us(&sc.ev.start, &sc.ev.upload)?;
        self.last_upload_us.store(upload_us, Ordering::Relaxed);
        t.gather_us += pack_us + upload_us + elapsed_us(&sc.ev.upload, &sc.ev.gather)?;
        t.ntt_us += elapsed_us(&sc.ev.gather, &sc.ev.ntt)?;
        t.pointwise_us += if fused {
            0
        } else {
            elapsed_us(&sc.ev.ntt, &sc.ev.join)?
        };

        Ok(HPoly::Device {
            tag: TAG,
            len: n,
            data: Arc::new(HHandle {
                scratch: Some(sc),
                pool: Arc::clone(&self.pool),
                stream: Arc::clone(&self.stream),
                len: n,
            }),
        })
    }

    // -----------------------------------------------------------------------
    // Launches
    // -----------------------------------------------------------------------

    /// Stage 0. One thread per output row; see the mapping argument at the top of
    /// `kernels/gather.cu`.
    fn launch_gather(&self, sc: &Scratch, n: usize) -> Result<(), ProveError> {
        let nn = n as u32;
        let block = self.block.min(nn.max(1));
        let cfg = LaunchConfig {
            grid_dim: (nn.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = self.stream.launch_builder(&self.gather);
        lb.arg(&self.row_ptr[0])
            .arg(&self.signal[0])
            .arg(&self.value[0])
            .arg(&self.row_ptr[1])
            .arg(&self.signal[1])
            .arg(&self.value[1])
            .arg(&sc.witness)
            .arg(&sc.a)
            .arg(&sc.b)
            .arg(&sc.c)
            .arg(&nn);
        // SAFETY: the kernel's eleven parameters are bound in order and with matching types;
        // `row_ptr` holds domain_size + 1 entries and every `signal` entry was checked
        // against `n_vars` in `new`, so no thread reads out of bounds; the kernel's
        // `gid >= n` guard covers the tail block.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_gather_abc", e))?;
        Ok(())
    }

    /// The six transforms.
    ///
    /// `fuse_h` folds stage 4 into the store epilogue of the last batch of C's forward
    /// transform, which is legal because A and B are already fully transformed by then and
    /// their coset evaluations are exactly what `H = A*B - C` needs. It saves a whole write
    /// of C and a whole read of A, B and C.
    fn launch_transforms(&self, sc: &Scratch, n: usize, fuse_h: bool) -> Result<(), ProveError> {
        let log_n = self.domain.log_size;
        let last = self.batches.len() - 1;
        let size_inv = PackedFr::from_fr(&self.domain.size_inv);

        for (vi, v) in [&sc.a, &sc.b, &sc.c].into_iter().enumerate() {
            // Stage 1, the iNTT. Out of place v -> t for the head, then in place on t. The
            // 1/n normalisation rides in on the load.
            for (bi, b) in self.batches.iter().enumerate() {
                let p = NttParams {
                    log_n,
                    s0: b.s0,
                    k: b.k,
                    scale_mode: if bi == 0 { SCALE_CONST } else { SCALE_NONE },
                    store_mode: STORE_PLAIN,
                    kscale: size_inv,
                };
                if bi == 0 {
                    self.launch_head(&p, v, &sc.t, &self.tw_inv, sc, n)?;
                } else {
                    self.launch_tail(&p, &sc.t, &self.tw_inv, sc, n)?;
                }
            }

            // Stages 2 and 3: the coset shift, applied on load, then the forward transform
            // back into v. Stage 4 rides out on the last store of the last vector.
            for (bi, b) in self.batches.iter().enumerate() {
                let join = fuse_h && vi == 2 && bi == last;
                let p = NttParams {
                    log_n,
                    s0: b.s0,
                    k: b.k,
                    scale_mode: if bi == 0 { SCALE_TABLE } else { SCALE_NONE },
                    store_mode: if join { STORE_JOIN } else { STORE_PLAIN },
                    kscale: size_inv,
                };
                if bi == 0 {
                    self.launch_head(&p, &sc.t, v, &self.tw_fwd, sc, n)?;
                } else {
                    self.launch_tail(&p, v, &self.tw_fwd, sc, n)?;
                }
            }
        }
        Ok(())
    }

    /// Batch 0 of a transform: out of place, absorbs the bit-reverse and the load-time
    /// scale, and carries the fused stage 4 when the mode says so.
    ///
    /// Every declared parameter is bound even on a mode that never dereferences it. CUDA,
    /// unlike Metal's validation layer, would accept a null pointer there, but binding a
    /// real allocation keeps the two hosts symmetric and turns a stray read into a fault
    /// rather than into whatever address zero holds. Outside `STORE_JOIN` the join operands
    /// get `sc.unused`, a one-element buffer, so no launch ever binds the same buffer twice.
    ///
    /// The bindings are shared references even where the kernel writes. `cudarc`'s
    /// `&mut CudaSlice` argument form exists only to record read/write events, and those
    /// are recorded only when the context is in multi-stream mode; everything here is issued
    /// in order on one stream, which is what actually orders it. The Metal encoder binds the
    /// same way (`set_buffer` takes `&Buffer` for both directions), and taking `&mut` here
    /// would make the aliasing that the JOIN mode needs inexpressible without a second dummy
    /// buffer per mutable slot.
    #[allow(clippy::too_many_arguments)]
    fn launch_head(
        &self,
        p: &NttParams,
        src: &CudaSlice<u32>,
        dst: &CudaSlice<u32>,
        tw: &CudaSlice<u32>,
        sc: &Scratch,
        n: usize,
    ) -> Result<(), ProveError> {
        let joining = p.store_mode == STORE_JOIN;
        let (join_a, join_b) = if joining {
            (&sc.a, &sc.b)
        } else {
            (&sc.unused, &sc.unused)
        };
        let (h_mont, h_std) = if joining {
            (&sc.h_mont, &sc.h_std)
        } else {
            (&sc.unused, &sc.unused)
        };
        let cfg = self.batch_config(p, n);
        let mut lb = self.stream.launch_builder(&self.head);
        lb.arg(src)
            .arg(dst)
            .arg(tw)
            .arg(&self.coset_pows)
            .arg(join_a)
            .arg(join_b)
            .arg(h_mont)
            .arg(h_std)
            .arg(p);
        // SAFETY: the nine parameters match the kernel in order and type, `p` is the 52-byte
        // by-value block it expects, the launch's shared memory is exactly the 2^k * 32
        // bytes it indexes, and every buffer this mode dereferences is domain sized.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_ntt_head", e))?;
        Ok(())
    }

    /// Every batch after the first: in place, slice strided by 2^s0, never scaled.
    fn launch_tail(
        &self,
        p: &NttParams,
        a: &CudaSlice<u32>,
        tw: &CudaSlice<u32>,
        sc: &Scratch,
        n: usize,
    ) -> Result<(), ProveError> {
        let joining = p.store_mode == STORE_JOIN;
        let (join_a, join_b) = if joining {
            (&sc.a, &sc.b)
        } else {
            (&sc.unused, &sc.unused)
        };
        let (h_mont, h_std) = if joining {
            (&sc.h_mont, &sc.h_std)
        } else {
            (&sc.unused, &sc.unused)
        };
        let cfg = self.batch_config(p, n);
        let mut lb = self.stream.launch_builder(&self.tail);
        lb.arg(a)
            .arg(tw)
            .arg(join_a)
            .arg(join_b)
            .arg(h_mont)
            .arg(h_std)
            .arg(p);
        // SAFETY: as for the head. The tail is never launched with k = 0: a zero-width batch
        // only occurs as the single batch of a size-1 domain, and `split_passes` gives that
        // domain exactly one batch, which the head serves.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_ntt_tail", e))?;
        Ok(())
    }

    /// Standalone stage 4, only on the `G16_CUDA_UNFUSED` path.
    fn launch_h_join(&self, sc: &Scratch, n: usize) -> Result<(), ProveError> {
        let nn = n as u32;
        let block = self.block.min(nn.max(1));
        let cfg = LaunchConfig {
            grid_dim: (nn.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = self.stream.launch_builder(&self.h_join);
        lb.arg(&sc.a)
            .arg(&sc.b)
            .arg(&sc.c)
            .arg(&sc.h_mont)
            .arg(&sc.h_std)
            .arg(&nn);
        // SAFETY: six parameters, all domain sized, and the kernel guards its tail block.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_h_join", e))?;
        Ok(())
    }

    /// Grid, block and shared memory for one NTT batch.
    ///
    /// One block owns 2^k elements and runs all k passes over them in shared memory, so the
    /// grid is n / 2^k blocks; the `max(1)` covers the size-1 domain, where k = 0 and the
    /// single block does a load and a store and no butterfly. The block width is free (both
    /// kernel loops are strided) so it is the preference clamped by the slice, and
    /// `shared_mem_bytes` is the slice and nothing else, since neither kernel declares any
    /// static `__shared__`.
    fn batch_config(&self, p: &NttParams, n: usize) -> LaunchConfig {
        let blk = 1u32 << p.k;
        let grid = ((n as u32) >> p.k).max(1);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (self.block.min(blk).max(1), 1, 1),
            shared_mem_bytes: blk * core::mem::size_of::<PackedFr>() as u32,
        }
    }

    // -----------------------------------------------------------------------
    // Scratch pool
    // -----------------------------------------------------------------------

    fn take_scratch(&self) -> Result<Scratch, ProveError> {
        if let Some(s) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(s);
        }
        let n = self.domain.size;
        let words = |elems: usize| -> Result<CudaSlice<u32>, ProveError> {
            // alloc_zeros, not alloc. Nothing in this stage group reads a buffer before
            // writing it: the gather writes every row of A, B and C, batch 0 of every
            // transform writes every element of its destination, and the join epilogue
            // writes every element of both H buffers. So the zeroing is not load bearing the
            // way it is for the MSM's bucket accumulators. It is taken anyway because it
            // happens once per pooled scratch rather than once per proof, and because
            // `cudaMalloc` hands back whatever the last tenant left: an uninitialised limb
            // array is a perfectly plausible looking field element, so a future kernel that
            // skipped an element would produce a wrong proof rather than an obvious one.
            self.stream
                .alloc_zeros::<u32>(elems.max(1) * (core::mem::size_of::<PackedFr>() / 4))
                .map_err(|e| drv("allocate scratch", e))
        };
        // SAFETY: the pinned buffer is unset on return, which is fine because the packer
        // writes all `n_vars` elements before anything reads it, and the only reader is the
        // DMA that the same thread issues after that write.
        let host_witness = unsafe { self.ctx.alloc_pinned::<u32>(self.n_vars.max(1) * 8) }
            .map_err(|e| drv("allocate pinned witness staging", e))?;
        Ok(Scratch {
            host_witness,
            witness: words(self.n_vars)?,
            a: words(n)?,
            b: words(n)?,
            c: words(n)?,
            t: words(n)?,
            h_mont: words(n)?,
            h_std: words(n)?,
            unused: words(1)?,
            ev: Events::new(&self.ctx)?,
        })
    }
}

/// Uploads `data`, padding an empty slice to one word, and waits.
///
/// Two things to know. `cuMemAlloc` with a size of zero is documented to fail with
/// `CUDA_ERROR_INVALID_VALUE`, and two real inputs reach here empty: an all-zero CSR matrix,
/// and the twiddle table of a size-1 domain (n/2 = 0 entries). One padding word keeps the
/// allocation legal, and no kernel reads it because every loop that could is bounded by a
/// row pointer or by a domain size that is itself zero.
///
/// And the copy is issued from pageable host memory, which `cudarc` does not synchronize
/// after (`SyncOnDrop::Sync(None)` for a plain slice). The callers pass temporaries that are
/// dropped at the end of their statement, so the wait has to happen here, not once at the
/// end of `new`. This runs at prepare time only, so the cost of waiting is irrelevant.
fn upload_u32(stream: &Arc<CudaStream>, data: &[u32]) -> Result<CudaSlice<u32>, ProveError> {
    let src: &[u32] = if data.is_empty() { &[0u32] } else { data };
    let buf = stream
        .clone_htod(src)
        .map_err(|e| drv("upload to device", e))?;
    stream
        .synchronize()
        .map_err(|e| drv("synchronize upload", e))?;
    Ok(buf)
}

/// Event-pair elapsed time in microseconds. `elapsed_ms` is GPU measured with about half a
/// microsecond of resolution, so rounding to whole microseconds loses nothing real.
fn elapsed_us(start: &CudaEvent, end: &CudaEvent) -> Result<u64, ProveError> {
    let ms = start
        .elapsed_ms(end)
        .map_err(|e| drv("read event elapsed time", e))?;
    Ok((f64::from(ms) * 1000.0).round().max(0.0) as u64)
}

// ---------------------------------------------------------------------------
// The device-resident result
// ---------------------------------------------------------------------------

/// What [`g16_core::HPoly::Device`] carries out of stage 4, under the tag [`TAG`].
///
/// Recover it with `h.device_handle::<HHandle>(g16_cuda::stages::TAG)`. It owns the whole
/// scratch set for the proof, which is what makes the buffers safe to read until the caller
/// drops the `HPoly`: the scratch goes back into the pool at that point and not before, so a
/// second concurrent proof cannot be handed buffers stage 9 is still reading.
///
/// The work that produced these buffers was synchronized before `compute_h` returned, so a
/// consumer on any stream may read them with no further ordering.
pub struct HHandle {
    scratch: Option<Scratch>,
    pool: Pool,
    stream: Arc<CudaStream>,
    len: usize,
}

impl HHandle {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn sc(&self) -> &Scratch {
        // Only `drop` clears the option, and it cannot run while a `&self` borrow lives.
        self.scratch.as_ref().expect("handle used after drop")
    }

    /// H in **standard form**, the integer in `[0, r)`, as `len * 8` words.
    ///
    /// This is what stage 9's Pippenger MSM must read: a window digit of a Montgomery
    /// representative is a digit of `a*R mod r`, which is a different number.
    pub fn h_std(&self) -> &CudaSlice<u32> {
        &self.sc().h_std
    }

    /// H in **Montgomery form**, matching `PackedFr` and arkworks' internal representation.
    /// For further field arithmetic and for host comparisons.
    pub fn h_mont(&self) -> &CudaSlice<u32> {
        &self.sc().h_mont
    }

    /// Copies H down to the host. Tests and cross-checks want this; the proving path must
    /// not, because forcing the copy is exactly the PCIe round trip that
    /// [`g16_core::HPoly`] exists to avoid.
    pub fn to_host(&self) -> Result<Vec<Fr>, ProveError> {
        let words = self.download(self.h_mont())?;
        let packed = from_words::<PackedFr>(&words)
            .ok_or_else(|| bad("H readback is not a whole number of field elements"))?;
        Ok(PackedFr::unpack_slice(&packed))
    }

    /// The standard-form copy, read back and validated. `None` at any index whose limbs are
    /// not a canonical residue, which would mean the reduction in `fr_from_mont` is wrong; a
    /// silent modular reduction here would hide exactly that bug.
    pub fn to_host_std(&self) -> Result<Option<Vec<Fr>>, ProveError> {
        let words = self.download(self.h_std())?;
        let packed = from_words::<PackedScalar>(&words)
            .ok_or_else(|| bad("H readback is not a whole number of field elements"))?;
        Ok(packed.iter().map(PackedScalar::to_fr).collect())
    }

    /// Device to host, then wait. The wait is not optional: `cudarc` issues the copy into a
    /// plain `Vec` asynchronously and does not synchronize afterwards, so reading the `Vec`
    /// without this returns whatever was in the uninitialised allocation.
    fn download(&self, buf: &CudaSlice<u32>) -> Result<Vec<u32>, ProveError> {
        let v = self
            .stream
            .clone_dtoh(buf)
            .map_err(|e| drv("download H", e))?;
        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize H download", e))?;
        Ok(v)
    }
}

impl Drop for HHandle {
    fn drop(&mut self) {
        if let Some(s) = self.scratch.take() {
            self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split must cover every pass exactly once, in order, with no batch wider than the
    /// shared-memory budget. Everything downstream indexes twiddles off `s0`, so an overlap
    /// or a gap here is a wrong transform, not a slow one.
    #[test]
    fn pass_batches_partition_the_transform() {
        for max_fused in 1..=10u32 {
            for log_n in 0..=24u32 {
                let bs = split_passes(log_n, max_fused);
                assert!(!bs.is_empty());
                let mut at = 0;
                for b in &bs {
                    assert_eq!(b.s0, at, "log_n {log_n} max {max_fused}: gap or overlap");
                    assert!(
                        b.k <= max_fused,
                        "log_n {log_n}: batch wider than the budget"
                    );
                    at += b.k;
                }
                assert_eq!(at, log_n, "log_n {log_n}: passes lost");
                // As few batches as the budget allows.
                assert_eq!(bs.len() as u32, log_n.div_ceil(max_fused).max(1));
                // And as even as possible: no two batches differ by more than one pass.
                let lo = bs.iter().map(|b| b.k).min().unwrap();
                let hi = bs.iter().map(|b| b.k).max().unwrap();
                assert!(hi - lo <= 1, "log_n {log_n}: uneven split {bs:?}");
            }
        }
    }

    /// The shapes the artifacts actually use. These are the numbers the Metal backend's twin
    /// test asserts, which is the point: both GPUs run one schedule, so a change to the
    /// splitting rule on either side shows up as a diff here rather than only in a timing.
    #[test]
    fn the_artifact_domains_split_as_documented() {
        assert_eq!(split_passes(3, 10).len(), 1); // tiny_mul, domain 8
        assert_eq!(
            split_passes(12, 10).iter().map(|b| b.k).collect::<Vec<_>>(),
            vec![6, 6] // js_1x1_d8, domain 4096
        );
        assert_eq!(
            split_passes(18, 10).iter().map(|b| b.k).collect::<Vec<_>>(),
            vec![9, 9] // js_16x16_d32, domain 2^18
        );
        assert_eq!(
            split_passes(22, 10).iter().map(|b| b.k).collect::<Vec<_>>(),
            vec![8, 7, 7]
        );
    }

    /// The argument block is a wire format shared with `kernels/ntt.cu`, which carries the
    /// matching `static_assert`. A field added on one side only would be read at the wrong
    /// offset and would produce a wrong transform rather than an error.
    #[test]
    fn ntt_params_is_the_size_the_kernel_asserts() {
        assert_eq!(core::mem::size_of::<NttParams>(), 52);
        assert!(kernels::NTT_CU.contains("static_assert(sizeof(NttParams) == 52"));
    }

    /// The five mode tags are compared by value inside the kernel, so they are part of the
    /// same wire format as the struct around them.
    #[test]
    fn the_mode_tags_match_the_kernel() {
        for (name, want) in [
            ("G16_SCALE_NONE", SCALE_NONE),
            ("G16_SCALE_CONST", SCALE_CONST),
            ("G16_SCALE_TABLE", SCALE_TABLE),
            ("G16_STORE_PLAIN", STORE_PLAIN),
            ("G16_STORE_JOIN", STORE_JOIN),
        ] {
            let spaced = format!("{name}  = {want}u");
            let tight = format!("{name} = {want}u");
            assert!(
                kernels::NTT_CU.contains(&spaced) || kernels::NTT_CU.contains(&tight),
                "kernels/ntt.cu does not define {name} as {want}"
            );
        }
    }
}
