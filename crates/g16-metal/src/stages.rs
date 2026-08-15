//! Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
//! `H = A*B - C`, with the three domain vectors resident in device memory throughout.
//!
//! # What this module is for
//!
//! [`g16_core::PreparedCircuit::compute_h`] is a stage *group*, not a primitive, and the
//! reason is here. A trait with an `ntt()` method would force the three vectors back
//! through host memory between every one of the six transforms, so a GPU backend built
//! against it would measure the bus. Instead everything below runs inside a single
//! command buffer with a single wait, and the result never leaves the device: the value
//! handed back is [`g16_core::HPoly::Device`] carrying an [`HHandle`], and stage 9's MSM
//! reads the buffer stage 4 wrote.
//!
//! # Dispatch budget, which is what actually decides whether this is worth doing
//!
//! Measured on this M2 Max during scouting: a command buffer costs 0.15 to 0.19 ms to
//! commit and wait for, and that floor does not shrink with the work. An extra dispatch
//! encoded into an already-open command buffer costs 2 to 3 microseconds, fifty to a
//! hundred times less. 128 trivial dispatches took 0.29 ms batched into one command
//! buffer against 14.6 ms as 128 separate commit/wait pairs.
//!
//! So the whole of stages 0 to 4 is one command buffer. At the 2^18 domain of the largest
//! artifact that is fourteen dispatches: one gather, then two per transform for six
//! transforms, with stage 4 fused into the last one. The reference Metal implementations
//! do the opposite (zkonduit commits and blocks four times per MSM, zkmopro uses one
//! command buffer per dispatch) and at our domain sizes that alone would decide the
//! result before any arithmetic ran.
//!
//! # Where the host still does work per proof
//!
//! Packing the witness. `ark_ff::Fp` is not `repr(C)` and `ark_ec::G1Affine` is 72 bytes
//! rather than 64, so nothing can be byte-cast; see `crate::layout`. The witness is the
//! one per-proof vector that has to be repacked, and it is repacked straight into the
//! `MTLBuffer` contents pointer, so on unified memory that write *is* the upload and
//! there is no second copy. Everything else the kernels read (the CSR rows, both twiddle
//! tables, the coset power table) is witness independent and is packed once in
//! [`HStages::prepare`].

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use g16_core::{HPoly, ProveError, StageTimings};
use g16_field::{Domain, Field, Fr};
use g16_zkey::ProvingKey;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

use crate::kernels::{FR_MSL, GATHER_MSL, NTT_MSL, POINTWISE_MSL};
use crate::layout::{as_bytes, PackedFr, PackedScalar};

/// The tag on [`g16_core::HPoly::Device`] values produced here. A handle carrying any
/// other tag came from a different backend and must not be dereferenced as an [`HHandle`].
pub const TAG: &str = "metal";

/// Threadgroup size preference, clamped down by each pipeline's own reported maximum.
///
/// Never a hardcoded 1024. The device advertises 1024 threads per threadgroup, but a
/// kernel carrying a CIOS Montgomery multiply reports less because of register pressure:
/// 896 was measured for a bare multiply kernel on this machine and 512 for a kernel doing
/// point arithmetic. Metal rejects a threadgroup larger than the pipeline's limit, so the
/// preference is always `min`ed against `max_total_threads_per_threadgroup`.
const PREFERRED_THREADS: u64 = 256;

/// Hard cap on passes fused into one dispatch, independent of the memory budget.
///
/// 2^10 elements at 32 bytes is exactly the 32768-byte threadgroup allocation this device
/// permits, so no device can support more than this for BN254 `Fr` and the constant is a
/// statement about the field, not about the hardware. The actual cap is recomputed from
/// `maxThreadgroupMemoryLength` in [`HStages::new_with_library`]; this only bounds it.
const MAX_FUSED_PASSES: u32 = 10;

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "metal",
        reason: reason.into(),
    }
}

// ---------------------------------------------------------------------------
// The kernel argument block
// ---------------------------------------------------------------------------

/// Mirrors `struct NttParams` in `shaders/ntt.metal`. 52 bytes, no padding on either
/// side. The MSL carries a `static_assert` on its size and this carries a `const`
/// assertion, so the two cannot drift into a silently misread argument block.
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

const SCALE_NONE: u32 = 0;
const SCALE_CONST: u32 = 1;
const SCALE_TABLE: u32 = 2;
const STORE_PLAIN: u32 = 0;
const STORE_JOIN: u32 = 1;

// ---------------------------------------------------------------------------
// Compiled kernels
// ---------------------------------------------------------------------------

/// Device, queue and the four compiled pipeline states for stages 0 to 4.
///
/// Built once and shared. Compiling this MSL through `newLibraryWithSource` was measured
/// at about 54 ms for the field prelude alone, which is a tenth of the CPU backend's
/// whole 596 ms proof for the largest artifact, so it must never sit inside a timed
/// region or a per-proof path.
pub struct HStages {
    device: Device,
    queue: CommandQueue,
    #[allow(dead_code)]
    library: Library,
    gather: ComputePipelineState,
    head: ComputePipelineState,
    tail: ComputePipelineState,
    h_join: ComputePipelineState,
    /// Largest number of NTT passes one dispatch can fuse, derived from the device's
    /// threadgroup memory limit rather than hardcoded.
    max_fused: u32,
}

// SAFETY: `MTLDevice`, `MTLCommandQueue`, `MTLLibrary` and `MTLComputePipelineState` are
// documented as safe to use from multiple threads concurrently. The types that are *not*
// thread safe, `MTLCommandBuffer` and `MTLComputeCommandEncoder`, are created inside
// `compute_h` on the calling thread and never escape it.
unsafe impl Send for HStages {}
unsafe impl Sync for HStages {}

impl HStages {
    /// The MSL for stages 0 to 4, in dependency order.
    ///
    /// Exposed so a backend that also wants the MSM kernels can compile one library
    /// instead of two and pay the runtime compile once. `pointwise.metal` must precede
    /// `ntt.metal`: the fused NTT epilogue calls `g16_store_h`, which is defined there.
    pub fn source() -> String {
        format!("{FR_MSL}\n{GATHER_MSL}\n{POINTWISE_MSL}\n{NTT_MSL}\n")
    }

    /// Picks the system default device and compiles [`Self::source`].
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default().ok_or_else(|| bad("no Metal device"))?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        let library = device
            .new_library_with_source(&Self::source(), &CompileOptions::new())
            // The compiler diagnostic names the offending line of MSL and is the only
            // thing that does, so it is propagated verbatim rather than summarised.
            .map_err(|e| bad(format!("MSL compile failed:\n{e}")))?;
        let queue = device.new_command_queue();
        Self::new_with_library(device, queue, library)
    }

    /// Builds from an already-compiled library, for a backend that compiles the whole
    /// prover (stages 0 to 4 plus the MSM kernels) as a single translation unit.
    pub fn new_with_library(
        device: Device,
        queue: CommandQueue,
        library: Library,
    ) -> Result<Self, ProveError> {
        let pso = |name: &str| -> Result<ComputePipelineState, ProveError> {
            let f = library
                .get_function(name, None)
                .map_err(|e| bad(format!("kernel {name} missing: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| bad(format!("pipeline {name}: {e}")))
        };
        let gather = pso("g16_gather_abc")?;
        let head = pso("g16_ntt_head")?;
        let tail = pso("g16_ntt_tail")?;
        let h_join = pso("g16_h_join")?;

        // Elements of 32 bytes that fit in threadgroup memory, as a power of two. On this
        // M2 Max maxThreadgroupMemoryLength is 32768, so 1024 elements and 10 passes.
        // Querying costs nothing and hardcoding 32768, as lambdaworks does, is a device
        // property written down as a constant.
        let budget = device.max_threadgroup_memory_length() as usize;
        let elems = budget / core::mem::size_of::<PackedFr>();
        if elems < 2 {
            return Err(bad(format!(
                "threadgroup memory {budget} bytes holds fewer than two Fr"
            )));
        }
        let max_fused = (usize::BITS - 1 - elems.leading_zeros() as u32).min(MAX_FUSED_PASSES);

        Ok(Self {
            device,
            queue,
            library,
            gather,
            head,
            tail,
            h_join,
            max_fused,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Passes fused into one dispatch. 10 on this device.
    pub fn max_fused_passes(&self) -> u32 {
        self.max_fused
    }

    /// Witness-independent upload: the two CSR matrices, both twiddle tables and the
    /// coset power table, packed once and left resident.
    pub fn prepare(&self, pk: &ProvingKey) -> Result<HResident, ProveError> {
        HResident::new(self, pk)
    }

    fn buf<T: crate::layout::Packed>(&self, data: &[T]) -> Buffer {
        // A zero-length MTLBuffer is not a valid allocation, and an empty CSR matrix or a
        // size-1 domain's empty twiddle table both reach here. One padding element keeps
        // the binding legal; no kernel reads it, because every loop that could is bounded
        // by a row pointer or a domain size that is zero.
        if data.is_empty() {
            return self.device.new_buffer(
                core::mem::size_of::<T>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
        }
        let bytes = as_bytes(data);
        self.device.new_buffer_with_data(
            bytes.as_ptr() as *const c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn buf_u32(&self, data: &[u32]) -> Buffer {
        if data.is_empty() {
            return self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn empty(&self, elems: usize) -> Buffer {
        let bytes = (elems.max(1) * core::mem::size_of::<PackedFr>()) as u64;
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Threadgroup width for a pipeline: the preference, clamped by what the pipeline
    /// will actually accept and by how much work there is.
    fn threads(&self, pso: &ComputePipelineState, work: u64) -> u64 {
        pso.max_total_threads_per_threadgroup()
            .min(PREFERRED_THREADS)
            .min(work.max(1))
            .max(1)
    }
}

// ---------------------------------------------------------------------------
// Resident, witness-independent state
// ---------------------------------------------------------------------------

/// One batch of NTT passes: `k` consecutive passes starting at pass `s0`.
#[derive(Clone, Copy, Debug)]
struct Batch {
    s0: u32,
    k: u32,
}

/// Splits `log_n` passes into as few batches as the threadgroup budget allows, sized as
/// evenly as possible.
///
/// Evenly, not greedily. A greedy fill at 2^18 with a cap of 10 gives 10 + 8, where the
/// second dispatch has only 2^8 threadgroups doing 2^17 butterflies; an even 9 + 9 gives
/// both dispatches 2^9 groups. The pathological version of the greedy split is what
/// profiling of bellperson caught at 2^26, where the leftover kernel launched 16 million
/// threadgroups of two threads each.
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

/// Per-proof scratch: the three domain vectors, one transform temporary, the packed
/// witness, and the two H outputs.
///
/// Pooled rather than allocated per proof. A freshly allocated shared buffer is faulted
/// in on first touch, measured at 15.2 GB/s against 54.8 GB/s once warm, so a cold
/// allocation of the 48 MB this needs at 2^18 would cost about 3 ms of page faults on
/// every proof. Pooling also keeps `compute_h` safe to call concurrently on one circuit,
/// which the `PreparedCircuit` contract requires: each in-flight proof holds its own set.
struct Scratch {
    witness: Buffer,
    a: Buffer,
    b: Buffer,
    c: Buffer,
    t: Buffer,
    h_mont: Buffer,
    h_std: Buffer,
}

// SAFETY: `MTLBuffer` is thread safe; a `Scratch` is owned by exactly one in-flight proof
// at a time by construction (it is removed from the pool while in use and returned only
// when the `HHandle` that owns it is dropped).
unsafe impl Send for Scratch {}
unsafe impl Sync for Scratch {}

type Pool = Arc<Mutex<Vec<Scratch>>>;

/// Everything uploaded once per key.
pub struct HResident {
    domain: Domain,
    n_vars: usize,
    /// snarkjs' `inc`, a primitive 2n-th root of unity. Not `Domain::coset_gen`; see
    /// `g16_core::cpu::CpuCircuit::new` for the argument, which is a contract with the
    /// section 9 bases in the zkey and not a free choice.
    coset_shift: Fr,
    batches: Vec<Batch>,

    row_ptr: [Buffer; 2],
    signal: [Buffer; 2],
    value: [Buffer; 2],
    tw_fwd: Buffer,
    tw_inv: Buffer,
    coset_pows: Buffer,

    pool: Pool,
}

// SAFETY: as for `Scratch`. Every field is either plain data or an `MTLBuffer`, which is
// only read by kernels after this point, never mutated.
unsafe impl Send for HResident {}
unsafe impl Sync for HResident {}

impl HResident {
    fn new(st: &HStages, pk: &ProvingKey) -> Result<Self, ProveError> {
        let domain = Domain::new(pk.domain_size).map_err(|e| bad(e.to_string()))?;
        if domain.size != pk.domain_size {
            return Err(bad(format!(
                "domain size {} is not a power of two",
                pk.domain_size
            )));
        }

        // Same derivation as the CPU backend, and it must stay the same: the coset is
        // fixed by the zkey's section 9 bases, so any other shift pairs the evaluations
        // against the wrong Lagrange polynomials.
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
            // Checked once per key rather than per proof. On the GPU an out-of-range
            // signal index is not a panic, it is a silent out-of-bounds read of whatever
            // follows the witness buffer, so this check is the only thing standing
            // between a malformed key and a proof built on garbage.
            if pk.coeffs.signal[m].iter().any(|&s| s as usize >= pk.n_vars) {
                return Err(bad(format!(
                    "matrix {name} references a signal beyond n_vars {}",
                    pk.n_vars
                )));
            }
        }

        // shift^j for j in [0, n). Not the twiddles: the twiddles are powers of the
        // domain's own root of unity and this is a primitive 2n-th root, so no table
        // g16-field builds can be reused. A running product, one multiply per entry.
        let mut pows = Vec::with_capacity(domain.size);
        let mut acc = Fr::ONE;
        for _ in 0..domain.size {
            pows.push(acc);
            acc *= coset_shift;
        }

        Ok(Self {
            n_vars: pk.n_vars,
            batches: split_passes(domain.log_size, st.max_fused),
            row_ptr: [
                st.buf_u32(&pk.coeffs.row_ptr[0]),
                st.buf_u32(&pk.coeffs.row_ptr[1]),
            ],
            signal: [
                st.buf_u32(&pk.coeffs.signal[0]),
                st.buf_u32(&pk.coeffs.signal[1]),
            ],
            value: [
                st.buf(&PackedFr::pack_slice(&pk.coeffs.value[0])),
                st.buf(&PackedFr::pack_slice(&pk.coeffs.value[1])),
            ],
            tw_fwd: st.buf(&PackedFr::pack_slice(&domain.twiddles())),
            tw_inv: st.buf(&PackedFr::pack_slice(&domain.twiddles_inv())),
            coset_pows: st.buf(&PackedFr::pack_slice(&pows)),
            coset_shift,
            domain,
            pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn domain_size(&self) -> usize {
        self.domain.size
    }

    pub fn coset_shift(&self) -> Fr {
        self.coset_shift
    }

    /// Dispatches encoded for one call to [`Self::compute_h`], for reporting.
    pub fn dispatch_count(&self) -> usize {
        1 + 6 * self.batches.len()
    }

    /// The `(s0, k)` pass split, for reporting.
    pub fn pass_batches(&self) -> Vec<(u32, u32)> {
        self.batches.iter().map(|b| (b.s0, b.k)).collect()
    }

    fn take_scratch(&self, st: &HStages) -> Scratch {
        if let Some(s) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return s;
        }
        let n = self.domain.size;
        Scratch {
            witness: st.empty(self.n_vars),
            a: st.empty(n),
            b: st.empty(n),
            c: st.empty(n),
            t: st.empty(n),
            h_mont: st.empty(n),
            h_std: st.empty(n),
        }
    }

    /// Stages 0 to 4. Returns `H` on the coset, length `domain_size`, still on the device.
    ///
    /// # Timing attribution
    ///
    /// In the default single-command-buffer path there is exactly one GPU measurement to
    /// make, because there is exactly one submission: `gather_us` carries the host-side
    /// witness pack, `ntt_us` carries the whole GPU wall time for stages 0 to 4, and
    /// `pointwise_us` is zero. Splitting it further would mean inventing a number.
    /// Setting `G16_METAL_PROFILE=1` instead submits three command buffers, so each field
    /// gets a real measurement, at the cost of two extra 0.15 ms round trips and of
    /// disabling the stage 4 fusion. That is exactly why it is opt in.
    pub fn compute_h(
        &self,
        st: &HStages,
        witness: &[Fr],
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size;
        let sc = self.take_scratch(st);

        // The pack writes straight into the buffer's contents pointer. On unified memory
        // that write is the upload; there is no staging copy and no blit.
        let start = Instant::now();
        // SAFETY: `sc.witness` was allocated with exactly `n_vars * size_of::<PackedFr>()`
        // bytes in shared storage, no GPU work is in flight against this scratch (it was
        // just taken from the pool, and a pooled scratch is only returned once the
        // command buffer that used it has completed), and `PackedFr` has no invalid bit
        // patterns.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(sc.witness.contents() as *mut PackedFr, self.n_vars)
        };
        PackedFr::pack_into(witness, dst);
        let pack_us = start.elapsed().as_micros() as u64;

        let profile = std::env::var_os("G16_METAL_PROFILE").is_some();
        if profile {
            self.run_profiled(st, &sc, n, t, pack_us)?;
        } else {
            t.gather_us += pack_us;
            let start = Instant::now();
            let cb = st.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_gather(st, enc, &sc, n);
            self.encode_transforms(st, enc, &sc, n, true);
            enc.end_encoding();
            cb.commit();
            crate::cb::wait_ok(cb, "stages 0-4 (gather and transforms)")?;
            t.ntt_us += start.elapsed().as_micros() as u64;
        }

        Ok(HPoly::Device {
            tag: TAG,
            len: n,
            data: Arc::new(HHandle {
                scratch: Some(sc),
                pool: Arc::clone(&self.pool),
                len: n,
            }),
        })
    }

    /// Three command buffers so the three stage groups get honest separate numbers. Costs
    /// two extra round trips and gives up the stage 4 fusion, hence opt in.
    fn run_profiled(
        &self,
        st: &HStages,
        sc: &Scratch,
        n: usize,
        t: &mut StageTimings,
        pack_us: u64,
    ) -> Result<(), ProveError> {
        let start = Instant::now();
        let cb = st.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_gather(st, enc, sc, n);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "stage 0-1 gather (profiled)")?;
        t.gather_us += pack_us + start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let cb = st.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_transforms(st, enc, sc, n, false);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "stages 2-3 transforms (profiled)")?;
        t.ntt_us += start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let cb = st.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&st.h_join);
        enc.set_buffer(0, Some(&sc.a), 0);
        enc.set_buffer(1, Some(&sc.b), 0);
        enc.set_buffer(2, Some(&sc.c), 0);
        enc.set_buffer(3, Some(&sc.h_mont), 0);
        enc.set_buffer(4, Some(&sc.h_std), 0);
        let nn = n as u32;
        enc.set_bytes(5, 4, &nn as *const u32 as *const c_void);
        let tg = st.threads(&st.h_join, n as u64);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "stage 4 h_join (profiled)")?;
        t.pointwise_us += start.elapsed().as_micros() as u64;
        Ok(())
    }

    fn encode_gather(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        sc: &Scratch,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&st.gather);
        enc.set_buffer(0, Some(&self.row_ptr[0]), 0);
        enc.set_buffer(1, Some(&self.signal[0]), 0);
        enc.set_buffer(2, Some(&self.value[0]), 0);
        enc.set_buffer(3, Some(&self.row_ptr[1]), 0);
        enc.set_buffer(4, Some(&self.signal[1]), 0);
        enc.set_buffer(5, Some(&self.value[1]), 0);
        enc.set_buffer(6, Some(&sc.witness), 0);
        enc.set_buffer(7, Some(&sc.a), 0);
        enc.set_buffer(8, Some(&sc.b), 0);
        enc.set_buffer(9, Some(&sc.c), 0);
        let nn = n as u32;
        enc.set_bytes(10, 4, &nn as *const u32 as *const c_void);
        let tg = st.threads(&st.gather, n as u64);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// The six transforms. `fuse_h` folds stage 4 into the store epilogue of the last
    /// batch of C's forward transform, which is legal because A and B are already fully
    /// transformed by then and their coset evaluations are exactly what `H = A*B - C`
    /// needs. It saves a whole write of C and a whole read of A, B and C.
    fn encode_transforms(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        sc: &Scratch,
        n: usize,
        fuse_h: bool,
    ) {
        let log_n = self.domain.log_size;
        let last = self.batches.len() - 1;
        let size_inv = PackedFr::from_fr(&self.domain.size_inv);

        for (vi, v) in [&sc.a, &sc.b, &sc.c].into_iter().enumerate() {
            // Stage 1, the iNTT. Out of place v -> t for the head, then in place on t.
            // The 1/n normalisation rides in on the load.
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
                    self.encode_head(st, enc, &p, v, &sc.t, &self.tw_inv, sc, n);
                } else {
                    self.encode_tail(st, enc, &p, &sc.t, &self.tw_inv, sc, n);
                }
            }

            // Stages 2 and 3: the coset shift, applied on load, then the forward
            // transform back into v. Stage 4 rides out on the last store of the last
            // vector.
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
                    self.encode_head(st, enc, &p, &sc.t, v, &self.tw_fwd, sc, n);
                } else {
                    self.encode_tail(st, enc, &p, v, &self.tw_fwd, sc, n);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_head(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        p: &NttParams,
        src: &Buffer,
        dst: &Buffer,
        tw: &Buffer,
        sc: &Scratch,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&st.head);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(tw), 0);
        enc.set_buffer(3, Some(&self.coset_pows), 0);
        // Every declared argument is bound even when the dispatch-uniform mode means the
        // kernel never reads it: Metal's validation layer objects to a null binding for a
        // declared buffer, whether or not the shader touches it on this path.
        enc.set_buffer(4, Some(&sc.a), 0);
        enc.set_buffer(5, Some(&sc.b), 0);
        enc.set_buffer(6, Some(&sc.h_mont), 0);
        enc.set_buffer(7, Some(&sc.h_std), 0);
        enc.set_bytes(
            8,
            core::mem::size_of::<NttParams>() as u64,
            p as *const NttParams as *const c_void,
        );
        self.dispatch_batch(st, enc, &st.head, p, n);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_tail(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        p: &NttParams,
        a: &Buffer,
        tw: &Buffer,
        sc: &Scratch,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&st.tail);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(2, Some(tw), 0);
        enc.set_buffer(4, Some(&sc.a), 0);
        enc.set_buffer(5, Some(&sc.b), 0);
        enc.set_buffer(6, Some(&sc.h_mont), 0);
        enc.set_buffer(7, Some(&sc.h_std), 0);
        enc.set_bytes(
            8,
            core::mem::size_of::<NttParams>() as u64,
            p as *const NttParams as *const c_void,
        );
        self.dispatch_batch(st, enc, &st.tail, p, n);
    }

    fn dispatch_batch(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        p: &NttParams,
        n: usize,
    ) {
        let blk = 1usize << p.k;
        let groups = (n / blk).max(1) as u64;
        // The slice, and nothing else. Held to the device's own reported limit rather
        // than to a constant.
        enc.set_threadgroup_memory_length(0, (blk * core::mem::size_of::<PackedFr>()) as u64);
        let tg = st.threads(pso, blk as u64);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }
}

// ---------------------------------------------------------------------------
// The device-resident result
// ---------------------------------------------------------------------------

/// What [`g16_core::HPoly::Device`] carries out of stage 4, under the tag [`TAG`].
///
/// Recover it with `h.device_handle::<HHandle>(g16_metal::stages::TAG)`. It owns the
/// whole scratch set for the proof, which is what makes the buffers safe to read until
/// the caller drops the `HPoly`: the scratch goes back into the pool at that point and
/// not before, so a second concurrent proof cannot be handed buffers stage 9 is still
/// reading.
pub struct HHandle {
    scratch: Option<Scratch>,
    pool: Pool,
    len: usize,
}

// SAFETY: `MTLBuffer` is thread safe, and the scratch inside is owned exclusively by this
// handle from the moment `compute_h` returns until this handle is dropped.
unsafe impl Send for HHandle {}
unsafe impl Sync for HHandle {}

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

    /// H in **standard form**, the integer in `[0, r)`. This is what stage 9's Pippenger
    /// MSM must read: a window digit of a Montgomery representative is a digit of
    /// `a*R mod r`, which is a different number.
    pub fn h_std(&self) -> &Buffer {
        &self.sc().h_std
    }

    /// H in **Montgomery form**, matching `layout::PackedFr` and arkworks' internal
    /// representation. For further field arithmetic and for host comparisons.
    pub fn h_mont(&self) -> &Buffer {
        &self.sc().h_mont
    }

    /// Copies H down to the host. Tests and cross-checks want this; the proving path
    /// must not, because forcing the copy is exactly the round trip that
    /// [`g16_core::HPoly`] exists to avoid.
    pub fn to_host(&self) -> Vec<Fr> {
        // SAFETY: the buffer holds exactly `len` `PackedFr` in shared storage and the
        // command buffer that wrote it was waited on before `compute_h` returned.
        let s = unsafe {
            core::slice::from_raw_parts(self.sc().h_mont.contents() as *const PackedFr, self.len)
        };
        PackedFr::unpack_slice(s)
    }

    /// The standard-form copy, read back and validated. `None` at any index whose limbs
    /// are not a canonical residue, which would mean the reduction in `fr_from_mont` is
    /// wrong; a silent modular reduction here would hide exactly that bug.
    pub fn to_host_std(&self) -> Option<Vec<Fr>> {
        // SAFETY: as above.
        let s = unsafe {
            core::slice::from_raw_parts(self.sc().h_std.contents() as *const PackedScalar, self.len)
        };
        s.iter().map(PackedScalar::to_fr).collect()
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

    /// The split must cover every pass exactly once, in order, with no batch wider than
    /// the threadgroup budget. Everything downstream indexes twiddles off `s0`, so an
    /// overlap or a gap here is a wrong transform, not a slow one.
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

    /// The shapes the artifacts actually use, spelled out so a change to the splitting
    /// rule shows up as a diff in the dispatch count rather than only in a timing.
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
}
