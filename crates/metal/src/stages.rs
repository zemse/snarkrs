//! Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
//! `H = A*B - C`, with the three domain vectors resident in device memory throughout.
//!
//! # What this module is for
//!
//! [`snarkrs_groth16::PreparedCircuit::compute_h`] is a stage *group*, not a primitive, and the
//! reason is here. A trait with an `ntt()` method would force the three vectors back
//! through host memory between every one of the six transforms, so a GPU backend built
//! against it would measure the bus. Instead everything below runs inside a single
//! command buffer with a single wait, and the result never leaves the device: the value
//! handed back is [`snarkrs_groth16::HPoly::Device`] carrying an [`HHandle`], and stage 9's MSM
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
//! So stages 0 to 4 are encoded as few command buffers as the display will tolerate:
//! four, one for the gather and one per domain vector, committed back to back and waited
//! on once. See the note in [`HResident::compute_h`] for why it is four and not one. At
//! the 2^18 domain of the largest `js_*` artifact that is thirteen dispatches: one
//! gather, then two per transform for six transforms, with stage 4 fused into the last
//! one. The reference Metal implementations do the opposite (zkonduit commits and blocks
//! four times per MSM, zkmopro uses one command buffer per dispatch) and at our domain
//! sizes that alone would decide the result before any arithmetic ran.
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
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use metal::objc::rc::autoreleasepool;
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, Library, MTLSize};
use snarkrs_field::{Domain, Field, Fr};
use snarkrs_formats::ProvingKey;
use snarkrs_groth16::{HPoly, ProveError, StageTimings};

use crate::kernels::{FR_MSL, GATHER_MSL, NTT_MSL, POINTWISE_MSL, SEAL_MSL};
use crate::layout::{PackedFr, PackedScalar, FR_MODULUS};
use crate::msm::Work;

/// The tag on [`snarkrs_groth16::HPoly::Device`] values produced here. A handle carrying any
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

/// A, B and C: the three witness-evaluation vectors that go round the transform pipeline
/// together. Named rather than written as a bare 3 because it is also the number of
/// command buffers `compute_h` splits the transforms across, and the two must agree.
const N_DOMAIN_VECTORS: usize = 3;

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
/// whole 596 ms proof for the largest `js_*` artifact, so it must never sit inside a timed
/// region or a per-proof path.
pub struct HStages {
    device: Device,
    queue: crate::cb::Queue,
    gather: ComputePipelineState,
    head: ComputePipelineState,
    tail: ComputePipelineState,
    h_join: ComputePipelineState,
    /// The completion token every command buffer here ends with.
    seal: crate::cb::Seal,
    /// Largest number of NTT passes one dispatch can fuse, derived from the device's
    /// threadgroup memory limit rather than hardcoded.
    max_fused: u32,
}

impl HStages {
    /// The MSL for stages 0 to 4, in dependency order. `pointwise.metal` must precede
    /// `ntt.metal`: the fused NTT epilogue calls `g16_store_h`, which is defined there.
    pub(crate) fn source() -> String {
        format!("{FR_MSL}\n{GATHER_MSL}\n{POINTWISE_MSL}\n{NTT_MSL}\n{SEAL_MSL}\n")
    }

    /// Picks the system default device and compiles the MSL for stages 0 to 4.
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
        let queue = crate::cb::Queue::new(&device);
        Self::new_with_library(device, queue, library)
    }

    /// Builds from an already-compiled library, for a backend that compiles the whole
    /// prover (stages 0 to 4 plus the MSM kernels) as a single translation unit.
    pub(crate) fn new_with_library(
        device: Device,
        queue: crate::cb::Queue,
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
        let seal = crate::cb::Seal::new(&device, &library)?;

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
            gather,
            head,
            tail,
            h_join,
            seal,
            max_fused,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
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

    /// `data` packed into Montgomery form straight into a shared buffer, with no host
    /// `Vec` of packed values in between.
    fn buf(&self, data: &[Fr]) -> Result<Buffer, ProveError> {
        // A zero-length MTLBuffer is not a valid allocation, and an empty CSR matrix or a
        // size-1 domain's empty twiddle table both reach here. One padding element keeps
        // the binding legal; no kernel reads it, because every loop that could is bounded
        // by a row pointer or a domain size that is zero.
        let buf = self.empty(data.len())?;
        // SAFETY: the buffer was just allocated with room for `data.len()` packed values
        // and nothing has been encoded against it, so no dispatch can be reading it.
        let dst =
            unsafe { core::slice::from_raw_parts_mut(buf.contents() as *mut PackedFr, data.len()) };
        PackedFr::pack_into(data, dst);
        Ok(buf)
    }

    fn buf_u32(&self, data: &[u32]) -> Result<Buffer, ProveError> {
        if data.is_empty() {
            return crate::alloc::shared(&self.device, 4);
        }
        crate::alloc::shared_with_data(&self.device, data.as_ptr() as *const c_void, data.len() * 4)
    }

    fn empty(&self, elems: usize) -> Result<Buffer, ProveError> {
        crate::alloc::shared(
            &self.device,
            elems.max(1) * core::mem::size_of::<PackedFr>(),
        )
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
/// witness, and H.
///
/// A, B, C and the temporary share `vectors`, one domain apart, so that once
/// `compute_h` has returned the whole buffer can be lent to stage 9's MSM for its digit
/// entries: nothing reads the four vectors after that. It is sized for whichever of the
/// two needs more.
///
/// Pooled rather than allocated per proof. A freshly allocated shared buffer is faulted
/// in on first touch, measured at 15.2 GB/s against 54.8 GB/s once warm, so a cold
/// allocation of the 48 MB this needs at 2^18 would cost about 3 ms of page faults on
/// every proof. Pooling also keeps `compute_h` safe to call concurrently on one circuit,
/// which the `PreparedCircuit` contract requires: each in-flight proof holds its own set.
struct Scratch {
    witness: Buffer,
    vectors: Buffer,
    /// Bytes between consecutive vectors in `vectors`.
    stride: u64,
    h_std: Buffer,
}

/// A domain vector: its buffer and the byte offset it starts at.
type Slot<'a> = (&'a Buffer, u64);

impl Scratch {
    fn a(&self) -> Slot<'_> {
        (&self.vectors, 0)
    }
    fn b(&self) -> Slot<'_> {
        (&self.vectors, self.stride)
    }
    fn c(&self) -> Slot<'_> {
        (&self.vectors, 2 * self.stride)
    }
    /// The transform temporary.
    fn t(&self) -> Slot<'_> {
        (&self.vectors, 3 * self.stride)
    }
}

fn bind(enc: &metal::ComputeCommandEncoderRef, index: u64, (buf, off): Slot<'_>) {
    enc.set_buffer(index, Some(buf), off);
}

/// How many of the `n` words of `h_std` lie outside `[0, r)`, and the index of the first,
/// or `None` when every one is a canonical residue.
///
/// Stage 4's store is `fr_from_mont`, which ends in a conditional subtraction whatever
/// its input, so no dispatch of this module can leave a word outside the field. One that
/// is there was not written by the transforms: the buffer's completion token says the
/// GPU ran the buffer to its end, this says whether what it wrote is what the host
/// reads. Priced in the commit that added it.
fn h_off_field(buf: &Buffer, n: usize) -> Option<(usize, usize)> {
    use rayon::prelude::*;
    const CHUNK: usize = 1 << 16;
    // SAFETY: the buffer holds `n` `PackedScalar` in shared storage and every command
    // buffer that wrote it has completed.
    let words = unsafe { core::slice::from_raw_parts(buf.contents() as *const PackedScalar, n) };
    words
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut count = 0;
            let mut first = None;
            for (i, w) in chunk.iter().enumerate() {
                if !canonical(&w.v) {
                    count += 1;
                    first.get_or_insert(ci * CHUNK + i);
                }
            }
            first.map(|first| (count, first))
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (Some((ca, fa)), Some((cb, fb))) => Some((ca + cb, fa.min(fb))),
                (a, None) => a,
                (None, b) => b,
            },
        )
}

/// Whether the little-endian limbs `v` are below the field modulus.
fn canonical(v: &[u32; 8]) -> bool {
    for i in (0..8).rev() {
        if v[i] != FR_MODULUS[i] {
            return v[i] < FR_MODULUS[i];
        }
    }
    false
}

fn off_field_error(context: &str, count: usize, n: usize, first: usize) -> ProveError {
    ProveError::Device {
        backend: "metal",
        reason: format!(
            "{context}: {count} of {n} H words are outside the field, the first at index \
             {first}: the GPU reported the buffer complete but its writes are not there"
        ),
    }
}

type Pool = Arc<Mutex<Vec<Scratch>>>;

/// Scratch sets [`HHandle`] keeps for reuse. More proofs than this in flight still work;
/// the extra sets are freed rather than kept.
const MAX_POOLED_SCRATCH: usize = 4;

/// Everything uploaded once per key.
pub struct HResident {
    domain: Domain,
    n_vars: usize,
    batches: Vec<Batch>,

    row_ptr: [Buffer; 2],
    signal: [Buffer; 2],
    value: [Buffer; 2],
    tw_fwd: Buffer,
    tw_inv: Buffer,
    coset_pows: Buffer,

    pool: Pool,
}

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
            let row_ptr = &pk.coeffs.row_ptr[m];
            if row_ptr.len() != domain.size + 1 {
                return Err(bad(format!(
                    "matrix {name} has {} rows, domain size is {}",
                    row_ptr.len().saturating_sub(1),
                    domain.size
                )));
            }
            // `gather.metal` walks `for (k = lo; k < hi; k++)`, so a decreasing pair is a
            // row that silently accumulates zero and a total past the signal array is an
            // out-of-bounds device read. Same two loops as `snarkrs_groth16::cpu`, the oracle.
            for c in 1..row_ptr.len() {
                if row_ptr[c] < row_ptr[c - 1] {
                    return Err(bad(format!(
                        "matrix {name} row_ptr is not monotone at row {}: {} then {}",
                        c - 1,
                        row_ptr[c - 1],
                        row_ptr[c]
                    )));
                }
            }
            let total = row_ptr[row_ptr.len() - 1] as usize;
            if total != pk.coeffs.signal[m].len() || total != pk.coeffs.value[m].len() {
                return Err(bad(format!(
                    "matrix {name} row_ptr ends at {total} but has {} signals and {} values",
                    pk.coeffs.signal[m].len(),
                    pk.coeffs.value[m].len()
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
        // snarkrs-field builds can be reused. A running product, one multiply per entry.
        let mut pows = Vec::with_capacity(domain.size);
        let mut acc = Fr::ONE;
        for _ in 0..domain.size {
            pows.push(acc);
            acc *= coset_shift;
        }
        // Host tables, released once packed so they do not sit in malloc's large cache.
        let table = |data: Vec<Fr>| {
            let buf = st.buf(&data);
            crate::alloc::release(data);
            buf
        };
        let tw_fwd = table(domain.twiddles())?;
        let tw_inv = table(domain.twiddles_inv())?;
        let coset_pows = table(pows)?;

        Ok(Self {
            n_vars: pk.n_vars,
            batches: split_passes(domain.log_size, st.max_fused),
            row_ptr: [
                st.buf_u32(&pk.coeffs.row_ptr[0])?,
                st.buf_u32(&pk.coeffs.row_ptr[1])?,
            ],
            signal: [
                st.buf_u32(&pk.coeffs.signal[0])?,
                st.buf_u32(&pk.coeffs.signal[1])?,
            ],
            value: [st.buf(&pk.coeffs.value[0])?, st.buf(&pk.coeffs.value[1])?],
            tw_fwd,
            tw_inv,
            coset_pows,
            domain,
            pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn domain_size(&self) -> usize {
        self.domain.size
    }

    /// Dispatches encoded for one call to [`Self::compute_h`], for reporting.
    pub fn dispatch_count(&self) -> usize {
        1 + 6 * self.batches.len()
    }

    /// The `(s0, k)` pass split, for reporting.
    pub fn pass_batches(&self) -> Vec<(u32, u32)> {
        self.batches.iter().map(|b| (b.s0, b.k)).collect()
    }

    fn take_scratch(&self, st: &HStages) -> Result<Scratch, ProveError> {
        if let Some(s) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(s);
        }
        let n = self.domain.size;
        let stride = n * core::mem::size_of::<PackedFr>();
        let bytes = (4 * stride).max(crate::msm::dense_entries_bytes(n));
        Ok(Scratch {
            witness: st.empty(self.n_vars)?,
            vectors: crate::alloc::shared(&st.device, bytes)?,
            stride: stride as u64,
            h_std: st.empty(n)?,
        })
    }

    /// Stages 0 to 4. Returns `H` on the coset, length `domain_size`, still on the device.
    ///
    /// # Timing attribution
    ///
    /// In the default single-command-buffer path there is exactly one GPU measurement to
    /// make, because there is exactly one submission: `gather_us` carries the host-side
    /// witness pack, `ntt_us` carries the whole GPU wall time for stages 0 to 4, and
    /// `pointwise_us` is zero. Splitting it further would mean inventing a number.
    /// Setting `G16_METAL_PROFILE=1` instead waits on each stage group separately, so each
    /// field gets a real measurement, at the cost of two extra 0.15 ms round trips and of
    /// disabling the stage 4 fusion. That is exactly why it is opt in.
    pub fn compute_h(
        &self,
        st: &HStages,
        witness: &[Fr],
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        self.compute_h_with(st, witness, Work::Variable, t)
    }

    /// [`Self::compute_h`] with the work mode chosen. [`Work::Constant`] stops the gather
    /// skipping the multiply for a 0 or 1 witness value; nothing else in stages 0 to 4
    /// looks at a value, and the result is the same either way.
    pub fn compute_h_with(
        &self,
        st: &HStages,
        witness: &[Fr],
        work: Work,
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size;
        let sc = self.take_scratch(st)?;

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
        // One autorelease pool per proof, around every submission below.
        //
        // `commandBuffer` and `computeCommandEncoder` are both `+0` autoreleased, and
        // metal-rs hands back a borrowed `&CommandBufferRef` that neither retains nor
        // releases. A Rust binary has no ambient pool and no run loop, so with none here
        // the objc runtime installs a page nothing ever drains and every submission leaks
        // its command buffer and its encoder for the life of the process;
        // `OBJC_DEBUG_MISSING_POOLS=YES` names both classes. The pool closes after the
        // waits rather than after each commit, because the refs the waits read are its.
        autoreleasepool(|| -> Result<(), ProveError> {
            if profile {
                return self.run_profiled(st, &sc, n, work, t, pack_us);
            }
            t.gather_us += pack_us;
            let start = Instant::now();

            // One command buffer per stage group, not one for all of them.
            //
            // An Apple GPU can be preempted between command buffers but not between
            // dispatches inside a single encoder. Gather plus all six transforms in one
            // buffer is 10 to 15 ms of back-to-back work at 2^18, which is longer than a
            // 120 Hz frame, so macOS aborts it with
            // kIOGPUCommandBufferCallbackErrorImpactingInteractivity rather than let it
            // hold the display. That is not hypothetical: it killed roughly one proof in
            // three on the larger circuits, and because `wait_ok` refuses to read the
            // output of a faulted buffer it surfaced as a failed proof rather than a
            // wrong one.
            //
            // Splitting per domain vector gives the compositor four places to get in.
            // Nothing is serialised on the CPU to buy that: all four are committed back
            // to back and only then waited on, and a queue runs its buffers in submission
            // order, so the cost is four submissions instead of one and not four round
            // trips. The stage 4 fusion survives because it rides out on the last store
            // of vector 2, which is inside that vector's own buffer.
            //
            // A kill is retried whole, from the gather: the transforms run in place, so a
            // vector whose buffer died half way is no longer an input anything can be
            // re-run from, while the gather rewrites A, B and C from the witness buffer,
            // which nothing on the device writes.
            //
            // Each buffer ends with its completion token, and the wait checks it: after
            // a GPU hang and recovery the driver reported buffers as `Completed` with no
            // error although neither of their encoders had run (3 of 300 in the seal
            // probe), which the status alone would have read as a finished proof.
            crate::cb::with_retry(&st.queue, |queue| {
                let mut cbs = Vec::with_capacity(1 + N_DOMAIN_VECTORS);

                let cb = crate::cb::command_buffer(queue);
                let enc = cb.new_compute_command_encoder();
                enc.set_label("stages 0-1 (gather)");
                self.encode_gather(st, enc, &sc, n, work);
                let token = st.seal.encode(enc);
                enc.end_encoding();
                cb.commit();
                cbs.push(("stages 0-1 (gather)", cb, token));

                for vi in 0..N_DOMAIN_VECTORS {
                    let cb = crate::cb::command_buffer(queue);
                    let enc = cb.new_compute_command_encoder();
                    enc.set_label("stages 2-4 (transforms)");
                    self.encode_transforms_vector(st, enc, &sc, n, vi, true);
                    let token = st.seal.encode(enc);
                    enc.end_encoding();
                    cb.commit();
                    cbs.push(("stages 2-4 (transforms)", cb, token));
                }

                // Every buffer is waited on before any failure is returned, so no
                // dispatch of this attempt is still writing the scratch when the next
                // one is encoded. The error kept is the first in submission order, which
                // is the first that happened.
                let mut first = Ok(());
                for (context, cb, token) in cbs {
                    let r = st.seal.wait(cb, token, context);
                    if first.is_ok() {
                        first = r;
                    }
                }
                first?;
                // Every buffer ran to its end. Whether H is there is a separate question
                // (BUG-28: words outside the field under a GPU recovery, with every
                // token present), and a wrong answer to it is retried like a kill.
                match h_off_field(&sc.h_std, n) {
                    Some((count, at)) => {
                        Err(off_field_error("stages 2-4 (transforms)", count, n, at))
                    }
                    None => Ok(()),
                }
            })?;
            t.ntt_us += start.elapsed().as_micros() as u64;
            Ok(())
        })?;

        Ok(HPoly::Device {
            tag: TAG,
            len: n,
            data: Arc::new(HHandle {
                scratch: Some(sc),
                pool: Arc::clone(&self.pool),
                len: n,
                lent: Arc::new(AtomicBool::new(false)),
            }),
        })
    }

    /// A wait per stage group, so the three get honest separate numbers. Costs two extra
    /// round trips and gives up the stage 4 fusion, hence opt in.
    ///
    /// Five command buffers, not three: the transforms are split per domain vector here
    /// exactly as they are on the fast path, because the preemption fault that split is
    /// for does not care which path encoded the work.
    fn run_profiled(
        &self,
        st: &HStages,
        sc: &Scratch,
        n: usize,
        work: Work,
        t: &mut StageTimings,
        pack_us: u64,
    ) -> Result<(), ProveError> {
        let queue = st.queue.get();
        let start = Instant::now();
        let cb = crate::cb::command_buffer(&queue);
        let enc = cb.new_compute_command_encoder();
        enc.set_label("stage 0-1 gather (profiled)");
        self.encode_gather(st, enc, sc, n, work);
        let token = st.seal.encode(enc);
        enc.end_encoding();
        cb.commit();
        st.seal.wait(cb, token, "stage 0-1 gather (profiled)")?;
        t.gather_us += pack_us + start.elapsed().as_micros() as u64;

        // Split per domain vector here too, and for the reason `compute_h` gives: all six
        // transforms in one encoder is the preemption fault, and a diagnostic path that
        // dies on the domains worth profiling diagnoses nothing. `ntt_us` is the wall time
        // around the whole group, the same number the fast path reports.
        let start = Instant::now();
        let mut cbs = Vec::with_capacity(N_DOMAIN_VECTORS);
        for vi in 0..N_DOMAIN_VECTORS {
            let cb = crate::cb::command_buffer(&queue);
            let enc = cb.new_compute_command_encoder();
            enc.set_label("stages 2-3 transforms (profiled)");
            self.encode_transforms_vector(st, enc, sc, n, vi, false);
            let token = st.seal.encode(enc);
            enc.end_encoding();
            cb.commit();
            cbs.push((cb, token));
        }
        for (cb, token) in cbs {
            st.seal
                .wait(cb, token, "stages 2-3 transforms (profiled)")?;
        }
        t.ntt_us += start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let cb = crate::cb::command_buffer(&queue);
        let enc = cb.new_compute_command_encoder();
        enc.set_label("stage 4 h_join (profiled)");
        enc.set_compute_pipeline_state(&st.h_join);
        bind(enc, 0, sc.a());
        bind(enc, 1, sc.b());
        bind(enc, 2, sc.c());
        enc.set_buffer(3, Some(&sc.h_std), 0);
        let nn = n as u32;
        enc.set_bytes(4, 4, &nn as *const u32 as *const c_void);
        let tg = st.threads(&st.h_join, n as u64);
        crate::cb::dispatch_threads(enc, MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        let token = st.seal.encode(enc);
        enc.end_encoding();
        cb.commit();
        st.seal.wait(cb, token, "stage 4 h_join (profiled)")?;
        if let Some((count, at)) = h_off_field(&sc.h_std, n) {
            return Err(off_field_error("stage 4 h_join (profiled)", count, n, at));
        }
        t.pointwise_us += start.elapsed().as_micros() as u64;
        Ok(())
    }

    fn encode_gather(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        sc: &Scratch,
        n: usize,
        work: Work,
    ) {
        enc.set_compute_pipeline_state(&st.gather);
        enc.set_buffer(0, Some(&self.row_ptr[0]), 0);
        enc.set_buffer(1, Some(&self.signal[0]), 0);
        enc.set_buffer(2, Some(&self.value[0]), 0);
        enc.set_buffer(3, Some(&self.row_ptr[1]), 0);
        enc.set_buffer(4, Some(&self.signal[1]), 0);
        enc.set_buffer(5, Some(&self.value[1]), 0);
        enc.set_buffer(6, Some(&sc.witness), 0);
        bind(enc, 7, sc.a());
        bind(enc, 8, sc.b());
        bind(enc, 9, sc.c());
        let nn = n as u32;
        enc.set_bytes(10, 4, &nn as *const u32 as *const c_void);
        let constant = u32::from(work == Work::Constant);
        enc.set_bytes(11, 4, &constant as *const u32 as *const c_void);
        let tg = st.threads(&st.gather, n as u64);
        crate::cb::dispatch_threads(enc, MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// The iNTT, the coset shift and the forward NTT for one of the three domain vectors,
    /// one vector per command buffer. See the comment in `compute_h` for why that matters.
    ///
    /// `fuse_h` folds stage 4 into the store epilogue of the last batch of C's forward
    /// transform, which is legal because A and B are already fully transformed by then and
    /// their coset evaluations are exactly what `H = A*B - C` needs. It saves a whole write
    /// of C and a whole read of A, B and C.
    fn encode_transforms_vector(
        &self,
        st: &HStages,
        enc: &metal::ComputeCommandEncoderRef,
        sc: &Scratch,
        n: usize,
        vi: usize,
        fuse_h: bool,
    ) {
        let log_n = self.domain.log_size;
        let last = self.batches.len() - 1;
        let size_inv = PackedFr::from_fr(&self.domain.size_inv);

        {
            let v = [sc.a(), sc.b(), sc.c()][vi];
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
                    self.encode_head(st, enc, &p, v, sc.t(), &self.tw_inv, sc, n);
                } else {
                    self.encode_tail(st, enc, &p, sc.t(), &self.tw_inv, sc, n);
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
                    self.encode_head(st, enc, &p, sc.t(), v, &self.tw_fwd, sc, n);
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
        src: Slot<'_>,
        dst: Slot<'_>,
        tw: &Buffer,
        sc: &Scratch,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&st.head);
        bind(enc, 0, src);
        bind(enc, 1, dst);
        enc.set_buffer(2, Some(tw), 0);
        enc.set_buffer(3, Some(&self.coset_pows), 0);
        // Every declared argument is bound even when the dispatch-uniform mode means the
        // kernel never reads it: Metal's validation layer objects to a null binding for a
        // declared buffer, whether or not the shader touches it on this path.
        bind(enc, 4, sc.a());
        bind(enc, 5, sc.b());
        enc.set_buffer(6, Some(&sc.h_std), 0);
        enc.set_bytes(
            7,
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
        a: Slot<'_>,
        tw: &Buffer,
        sc: &Scratch,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&st.tail);
        bind(enc, 0, a);
        enc.set_buffer(2, Some(tw), 0);
        bind(enc, 4, sc.a());
        bind(enc, 5, sc.b());
        enc.set_buffer(6, Some(&sc.h_std), 0);
        enc.set_bytes(
            7,
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
        crate::cb::dispatch_thread_groups(enc, MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }
}

// ---------------------------------------------------------------------------
// The device-resident result
// ---------------------------------------------------------------------------

/// What [`snarkrs_groth16::HPoly::Device`] carries out of stage 4, under the tag [`TAG`].
///
/// Recover it with `h.device_handle::<HHandle>(snarkrs_metal::stages::TAG)`. It owns the
/// whole scratch set for the proof, which is what makes the buffers safe to read until
/// the caller drops the `HPoly`: the scratch goes back into the pool at that point and
/// not before, so a second concurrent proof cannot be handed buffers stage 9 is still
/// reading.
pub struct HHandle {
    scratch: Option<Scratch>,
    pool: Pool,
    len: usize,
    /// Set while a [`crate::msm::Lent`] over the domain vectors is alive.
    lent: Arc<AtomicBool>,
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

    /// H in **standard form**, the integer in `[0, r)`. This is what stage 9's Pippenger
    /// MSM must read: a window digit of a Montgomery representative is a digit of
    /// `a*R mod r`, which is a different number.
    pub fn h_std(&self) -> &Buffer {
        &self.sc().h_std
    }

    /// The domain vectors, lent to the MSM over `h_std` for its digit entries. `None`
    /// while an earlier loan is still out.
    ///
    /// The handle must outlive the batch the loan goes into, which it already has to for
    /// `h_std`. A retried batch re-encodes from its zeroing dispatches and rewrites every
    /// entry, and a retried `compute_h` happens before this can be called, so no retry
    /// reads what the other wrote.
    pub(crate) fn lend_entries(&self) -> Option<crate::msm::Lent> {
        crate::msm::Lent::claim(&self.sc().vectors, &self.lent)
    }

    /// Copies H down to the host, validated: `None` if any value is not a canonical
    /// residue, which would mean the reduction in `fr_from_mont` is wrong; a silent
    /// modular reduction here would hide exactly that bug. Tests and cross-checks want
    /// this; the proving path must not, because forcing the copy is exactly the round
    /// trip that [`snarkrs_groth16::HPoly`] exists to avoid.
    pub fn to_host(&self) -> Option<Vec<Fr>> {
        // SAFETY: the buffer holds exactly `len` `PackedScalar` in shared storage and the
        // command buffer that wrote it was waited on before `compute_h` returned.
        let s = unsafe {
            core::slice::from_raw_parts(self.sc().h_std.contents() as *const PackedScalar, self.len)
        };
        s.iter().map(PackedScalar::to_fr).collect()
    }
}

impl Drop for HHandle {
    fn drop(&mut self) {
        if let Some(s) = self.scratch.take() {
            // The witness, its three domain images (and H's digit entries, if they were
            // lent) and H. The handle is only dropped once the MSMs that
            // read `h` have completed.
            for b in [&s.witness, &s.vectors, &s.h_std] {
                crate::alloc::scrub(b);
            }
            // One scratch set per proof in flight is all the pool is for. Uncapped, a burst
            // of concurrent proofs left that many full sets (six domain vectors' worth
            // each) resident for the life of the circuit.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < MAX_POOLED_SCRATCH {
                pool.push(s);
            }
        }
    }
}

/// Everything a `PreparedCircuit` carries from this module has to be `Send + Sync`, and
/// all four of these are, by derivation rather than by assertion: `metal-rs` declares
/// every handle it defines `Sync + Send`, and MTLDevice, MTLCommandQueue, MTLLibrary,
/// MTLComputePipelineState and MTLBuffer are documented thread-safe. The two Metal types
/// that are *not*, `MTLCommandBuffer` and `MTLComputeCommandEncoder`, are created inside
/// `compute_h` on the calling thread and never stored.
///
/// Checked here rather than asserted with a hand-written `unsafe impl`, which these four
/// used to carry. An `unsafe impl` overrides the auto-trait analysis instead of relying
/// on it, so a `Cell` or an `Rc` added to any of them later would have compiled, and the
/// guard in `backend.rs` that exists to catch exactly that would have passed anyway.
/// Same mechanism and same reasoning as `msm::thread_safety`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HStages>();
    assert_send_sync::<Scratch>();
    assert_send_sync::<HResident>();
    assert_send_sync::<HHandle>();
};

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

    fn tiny_key() -> Option<ProvingKey> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts/tiny_mul/circuit.zkey");
        if !path.is_file() {
            eprintln!("SKIPPED: no tiny_mul artifact");
            return None;
        }
        Some(ProvingKey::load(&path).unwrap())
    }

    /// `gather.metal` reads a decreasing pair as an empty range, so without this check the
    /// key would have produced a silently zeroed row. Mirrors `snarkrs_groth16::cpu`'s test.
    #[test]
    fn prepare_rejects_a_non_monotone_row_ptr() {
        let Some(mut pk) = tiny_key() else { return };
        let rp = &mut pk.coeffs.row_ptr[0];
        // Lift the start of a nonempty row above its end, which leaves the last entry
        // alone so this fails the monotonicity check and not the nonzero-count one.
        let c = (1..rp.len())
            .find(|&c| rp[c] > rp[c - 1])
            .expect("matrix A has no nonzeros");
        rp[c - 1] = rp[c] + 1;
        let st = HStages::new().expect("Metal device");
        let Err(err) = st.prepare(&pk) else {
            panic!("accepted a non-monotone row_ptr");
        };
        assert!(err.to_string().contains("not monotone"), "{err}");
    }

    /// Past the end of the signal array `gather.metal` reads whatever follows the buffer.
    #[test]
    fn prepare_rejects_a_row_ptr_past_the_signal_total() {
        let Some(mut pk) = tiny_key() else { return };
        let rp = &mut pk.coeffs.row_ptr[0];
        *rp.last_mut().expect("row_ptr is nonempty") += 1;
        let st = HStages::new().expect("Metal device");
        let Err(err) = st.prepare(&pk) else {
            panic!("accepted a row_ptr past the signal total");
        };
        assert!(err.to_string().contains("row_ptr ends at"), "{err}");
    }

    /// The production gather binding on ragged rows, including empty rows and repeated
    /// signals, against a CPU dot product. A test-only multiply-site probe counts terms;
    /// it checks the work shape, not the compiler's instructions or elapsed time.
    #[test]
    fn constant_work_gather_matches_cpu_and_visits_every_term() {
        use ark_ec::AdditiveGroup;
        let st = HStages::new().expect("Metal device");
        let source = HStages::source();
        let multiply = "fr_mul(value[k], w)";
        assert_eq!(source.matches(multiply).count(), 1);
        let probe_source = source.replace(multiply, "fr_one()");
        let library = st
            .device
            .new_library_with_source(&probe_source, &CompileOptions::new())
            .expect("multiply-site probe MSL");
        let probe = HStages::new_with_library(
            st.device.clone(),
            crate::cb::Queue::new(&st.device),
            library,
        )
        .expect("multiply-site probe pipelines");
        let n = 8;
        let rows = [
            vec![0, 0, 1, 5, 5, 7, 8, 8, 11],
            vec![0, 2, 2, 3, 7, 7, 8, 10, 11],
        ];
        let signals = [
            vec![0, 1, 2, 1, 3, 4, 5, 6, 7, 0, 2],
            vec![7, 0, 1, 2, 3, 2, 4, 5, 6, 7, 0],
        ];
        let values: [Vec<Fr>; 2] = std::array::from_fn(|m| {
            (0..11)
                .map(|i| match i % 4 {
                    0 => Fr::ZERO,
                    1 => Fr::ONE,
                    2 => -Fr::ONE,
                    _ => -Fr::from((i + m + 2) as u64),
                })
                .collect()
        });
        let domain = Domain::new(n).unwrap();
        let res = HResident {
            n_vars: n,
            batches: split_passes(domain.log_size, st.max_fused),
            row_ptr: std::array::from_fn(|m| st.buf_u32(&rows[m]).unwrap()),
            signal: std::array::from_fn(|m| st.buf_u32(&signals[m]).unwrap()),
            value: std::array::from_fn(|m| st.buf(&values[m]).unwrap()),
            tw_fwd: st.buf(&domain.twiddles()).unwrap(),
            tw_inv: st.buf(&domain.twiddles_inv()).unwrap(),
            coset_pows: st.buf(&vec![Fr::ONE; n]).unwrap(),
            domain,
            pool: Arc::new(Mutex::new(Vec::new())),
        };
        let sc = res.take_scratch(&st).unwrap();
        let cases = [
            ("zeros", vec![Fr::ZERO; n]),
            ("ones", vec![Fr::ONE; n]),
            (
                "mixed",
                (0..n)
                    .map(|i| match i % 3 {
                        0 => Fr::ZERO,
                        1 => Fr::ONE,
                        _ => -Fr::from(i as u64 + 2),
                    })
                    .collect(),
            ),
            ("dense", (0..n).map(|i| -Fr::from(i as u64 + 2)).collect()),
        ];
        for (label, witness) in cases {
            // SAFETY: scratch owns n packed witness elements and has no submission in flight.
            let dst = unsafe {
                core::slice::from_raw_parts_mut(sc.witness.contents() as *mut PackedFr, n)
            };
            PackedFr::pack_into(&witness, dst);
            for work in [Work::Variable, Work::Constant] {
                for (driver, counting) in [(&st, false), (&probe, true)] {
                    let queue = driver.queue.get();
                    let cb = crate::cb::command_buffer(&queue);
                    let enc = cb.new_compute_command_encoder();
                    res.encode_gather(driver, enc, &sc, n, work);
                    let token = driver.seal.encode(enc);
                    enc.end_encoding();
                    cb.commit();
                    driver.seal.wait(cb, token, "gather oracle").unwrap();
                    let expected: [Vec<Fr>; 2] = std::array::from_fn(|m| {
                        (0..n)
                            .map(|r| {
                                (rows[m][r]..rows[m][r + 1])
                                    .map(|k| {
                                        let k = k as usize;
                                        let w = witness[signals[m][k] as usize];
                                        if !counting {
                                            values[m][k] * w
                                        } else if work == Work::Constant {
                                            Fr::ONE
                                        } else if w == Fr::ZERO {
                                            Fr::ZERO
                                        } else if w == Fr::ONE {
                                            values[m][k]
                                        } else {
                                            Fr::ONE
                                        }
                                    })
                                    .sum()
                            })
                            .collect()
                    });
                    for (slot, want) in [
                        (sc.a(), expected[0].clone()),
                        (sc.b(), expected[1].clone()),
                        (
                            sc.c(),
                            expected[0]
                                .iter()
                                .zip(&expected[1])
                                .map(|(a, b)| *a * b)
                                .collect(),
                        ),
                    ] {
                        // SAFETY: each slot holds n PackedFr and the sealed submission completed.
                        let got = unsafe {
                            core::slice::from_raw_parts(
                                (slot.0.contents() as *const u8).add(slot.1 as usize)
                                    as *const PackedFr,
                                n,
                            )
                        };
                        assert_eq!(
                            PackedFr::unpack_slice(got),
                            want,
                            "{label} {work:?} probe={counting}"
                        );
                    }
                }
            }
        }
    }

    /// The gather alone, on one key, over witnesses of zeros, the circuit's own and dense
    /// random values, in both work modes: under `Work::Variable` the time follows how many
    /// witness values are 0 or 1, under `Work::Constant` it should not. `G16_PROBE_CIRCUIT`
    /// names the artifact, keccak256 by default.
    #[test]
    #[ignore = "GPU measurement; run explicitly on the measurement machine"]
    fn gather_time_by_witness() {
        use snarkrs_field::PrimeField;
        let name = std::env::var("G16_PROBE_CIRCUIT").unwrap_or_else(|_| "keccak256".into());
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts")
            .join(&name);
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let own = snarkrs_formats::wtns::Witness::load(&dir.join("circuit.wtns"))
            .unwrap()
            .0;
        let st = HStages::new().expect("Metal device");
        let res = st.prepare(&pk).unwrap();
        let n = res.domain.size;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let dense: Vec<Fr> = (0..own.len())
            .map(|_| {
                let mut bytes = [0u8; 32];
                for b in &mut bytes {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *b = seed as u8;
                }
                Fr::from_le_bytes_mod_order(&bytes)
            })
            .collect();
        let zeros = vec![Fr::from(0u64); own.len()];
        let sc = res.take_scratch(&st).unwrap();
        for work in [Work::Variable, Work::Constant] {
            for (label, w) in [("zeros", &zeros), ("own", &own), ("dense", &dense)] {
                // SAFETY: as in `compute_h`; nothing is in flight against this scratch.
                let dst = unsafe {
                    core::slice::from_raw_parts_mut(
                        sc.witness.contents() as *mut PackedFr,
                        res.n_vars,
                    )
                };
                PackedFr::pack_into(w, dst);
                let mut samples = Vec::new();
                let queue = st.queue.get();
                for rep in 0..25 {
                    let t = Instant::now();
                    let cb = crate::cb::command_buffer(&queue);
                    let enc = cb.new_compute_command_encoder();
                    res.encode_gather(&st, enc, &sc, n, work);
                    let token = st.seal.encode(enc);
                    enc.end_encoding();
                    cb.commit();
                    st.seal.wait(cb, token, "gather probe").unwrap();
                    if rep >= 5 {
                        samples.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                samples.sort_by(f64::total_cmp);
                println!(
                    "gather {name} {work:?} {label:<5} median {:.3} ms min {:.3}",
                    samples[samples.len() / 2],
                    samples[0]
                );
            }
        }
    }

    /// The check refuses a word at the modulus and above and passes one below it, and
    /// counts and locates them.
    #[test]
    fn off_field_words_are_counted_and_located() {
        let st = HStages::new().expect("Metal device");
        let n = 3 * (1 << 16) + 5;
        let buf = st.empty(n).unwrap();
        assert_eq!(h_off_field(&buf, n), None, "zeros are canonical");
        // SAFETY: shared storage, `n` words, nothing in flight.
        let words =
            unsafe { core::slice::from_raw_parts_mut(buf.contents() as *mut PackedScalar, n) };
        let mut below = FR_MODULUS;
        below[0] -= 1;
        words[7] = PackedScalar { v: below };
        assert_eq!(h_off_field(&buf, n), None, "r - 1 is canonical");
        words[1 << 16] = PackedScalar { v: FR_MODULUS };
        assert_eq!(h_off_field(&buf, n), Some((1, 1 << 16)), "r is not");
        words[n - 1] = PackedScalar { v: [u32::MAX; 8] };
        words[3] = PackedScalar {
            v: [0, 0, 0, 0, 0, 0, 0, 0x4000_0000],
        };
        assert_eq!(h_off_field(&buf, n), Some((3, 3)));
    }

    /// What the check adds to a proof, printed: a scan of H at the domain sizes the
    /// artifacts use. Run with
    /// `cargo test -p snarkrs-metal --release off_field_check_costs -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn off_field_check_costs_this_much() {
        let st = HStages::new().expect("Metal device");
        for log_n in [14u32, 18, 21, 22] {
            let n = 1usize << log_n;
            let buf = st.empty(n).unwrap();
            // SAFETY: as above.
            let words =
                unsafe { core::slice::from_raw_parts_mut(buf.contents() as *mut PackedScalar, n) };
            for (i, w) in words.iter_mut().enumerate() {
                *w = PackedScalar::from_fr(&Fr::from(i as u64 + 1));
            }
            let mut samples = Vec::new();
            for _ in 0..20 {
                let t = Instant::now();
                assert_eq!(h_off_field(&buf, n), None);
                samples.push(t.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "h_off_field 2^{log_n}: median {:.3} ms, min {:.3} ms",
                samples[samples.len() / 2],
                samples[0]
            );
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
