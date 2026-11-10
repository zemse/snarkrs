//! Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
//! `H = A*B - C`, with the three domain vectors resident in device memory throughout.
//!
//! Host twin of `crates/metal/src/stages.rs`, running the identical schedule against
//! `kernels/gather.cu`, `kernels/ntt.cu` and `kernels/pointwise.cu`. Same pass split, same
//! fusion, same argument block, so a Metal-versus-CUDA number compares two implementations
//! of one algorithm rather than two algorithms.
//!
//! # What this module is for
//!
//! [`snarkrs_groth16::PreparedCircuit::compute_h`] is a stage *group*, not a primitive. A trait
//! with an `ntt()` method would force the three vectors back through host memory between
//! every one of the six transforms, and on a discrete card that is not a cache miss, it is
//! a PCIe round trip in both directions. Everything below runs as one queue of launches on
//! one stream with a single synchronize at the end, and the result never leaves the
//! device: the value handed back is [`snarkrs_groth16::HPoly::Device`] carrying an [`HHandle`],
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
//! can be byte-cast (see `snarkrs_gpu_layout`), and the witness is the one per-proof vector
//! that has to be repacked. On Apple silicon the repack writes straight into the
//! `MTLBuffer` and that write *is* the upload. Here it is only the host half: the packed
//! bytes still have to cross PCIe. Both halves are measured and both are charged to
//! `gather_us`; see [`CudaStages::compute_h`].
//!
//! Everything else the kernels read (the CSR rows, both twiddle tables, the coset power
//! table) is witness independent, is uploaded once in [`CudaStages::new`], and stays
//! resident. A prover that rebuilt this per proof would be measuring its own key upload.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use snarkrs_field::{Domain, Field, Fr};
use snarkrs_formats::{HBasis, Matrix, ProvingKey};
use snarkrs_gpu_layout::{Packed, PackedFr, PackedScalar};
use snarkrs_groth16::{HPoly, ProveError, StageTimings};

use crate::context::{bad, download, drv, elapsed_us, upload_words};
use crate::{from_words, kernels, Cuda};

/// The tag on [`snarkrs_groth16::HPoly::Device`] values produced here. A handle carrying any
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
    /// After the gnark tail, [`CudaStages::compute_h_from_rows`] only.
    tail: CudaEvent,
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
            tail: ev()?,
        })
    }
}

/// Words in each half of a [`Staging`] pair, 32 MiB. A whole number of `Fr` and of both
/// packed point sizes, so a chunk never splits an element.
const STAGING_WORDS: usize = 8 << 20;

/// Words in each half of the pair [`Staging::for_key`] gives a prepare, 4 MiB: large enough
/// that each copy runs at the bus's rate, small enough that the pair is a rounding error
/// beside the key it uploads. Also a whole number of every element size.
const KEY_STAGING_WORDS: usize = 1 << 20;

/// The largest element a staging chunk carries, a packed G2 point, in words.
const ELEMENT_WORDS: usize = 32;

/// Two page-locked buffers the host fills in turn while the other is on the wire, for a
/// vector too large to stage whole: 64 MiB of pinned memory however long the vector is.
/// The rows path stages the witness and both row vectors through one, and a key whose
/// bases are streamed from disk goes through one too. A zkey's prepare packs its resident
/// vectors through a smaller one, see [`Self::for_key`].
pub(crate) struct Staging {
    bufs: [PinnedHostSlice<u32>; 2],
    /// Words in each buffer, a whole number of [`ELEMENT_WORDS`].
    words: usize,
    /// Recorded after each buffer's copy is queued, and waited on before the host writes
    /// that buffer again. Waiting on one never recorded returns at once.
    done: [CudaEvent; 2],
}

impl Staging {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, ProveError> {
        Self::with_words(ctx, STAGING_WORDS)
    }

    /// A pair for a prepare whose longest vector is `longest` words: [`KEY_STAGING_WORDS`]
    /// each, or less when the whole vector is shorter. The packed vector is never built
    /// on the host, so this pair is all the host memory an upload adds to the key.
    pub(crate) fn for_key(ctx: &Arc<CudaContext>, longest: usize) -> Result<Self, ProveError> {
        let words = longest.max(1).next_multiple_of(ELEMENT_WORDS);
        Self::with_words(ctx, words.min(KEY_STAGING_WORDS))
    }

    fn with_words(ctx: &Arc<CudaContext>, words: usize) -> Result<Self, ProveError> {
        let buf = || {
            // SAFETY: unset on return, and only ever read by a copy of the words `fill`
            // has just written.
            unsafe { ctx.alloc_pinned::<u32>(words) }.map_err(|e| drv("allocate pinned staging", e))
        };
        let ev = || {
            ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(|e| drv("cuEventCreate", e))
        };
        Ok(Self {
            bufs: [buf()?, buf()?],
            words,
            done: [ev()?, ev()?],
        })
    }

    /// `dst[off..off + len]` from `fill`, a chunk at a time: `fill(start, out)` writes words
    /// `start..start + out.len()` of the source into `out`. Each copy is queued on `stream`
    /// as soon as its chunk is written, so the host packs one chunk while the previous one
    /// crosses the bus; the last may still be in flight on return, ordered before anything
    /// queued after it.
    pub(crate) fn upload(
        &mut self,
        stream: &Arc<CudaStream>,
        dst: &mut CudaSlice<u32>,
        off: usize,
        len: usize,
        mut fill: impl FnMut(usize, &mut [u32]) -> Result<(), ProveError>,
    ) -> Result<(), ProveError> {
        if off.checked_add(len).is_none_or(|end| end > dst.len()) {
            return Err(bad(format!(
                "staged upload of {len} words at {off} into a buffer of {}",
                dst.len()
            )));
        }
        for (i, start) in (0..len).step_by(self.words).enumerate() {
            let m = self.words.min(len - start);
            let k = i % 2;
            self.done[k]
                .synchronize()
                .map_err(|e| drv("wait for a staging buffer", e))?;
            let chunk = &mut self.bufs[k]
                .as_mut_slice()
                .map_err(|e| drv("map pinned staging", e))?[..m];
            fill(start, chunk)?;
            // A plain slice, not the `PinnedHostSlice`, so the copy is exactly `m` words;
            // the memory under it is page locked either way, so the copy is asynchronous.
            stream
                .memcpy_htod(&*chunk, &mut dst.slice_mut(off + start..off + start + m))
                .map_err(|e| drv("staged upload", e))?;
            self.done[k]
                .record(stream)
                .map_err(|e| drv("record staging event", e))?;
        }
        Ok(())
    }

    /// `src` from its first word, as [`PackedFr`] words (Montgomery limbs, as on the host),
    /// packed on the pool.
    pub(crate) fn upload_fr(
        &mut self,
        stream: &Arc<CudaStream>,
        dst: &mut CudaSlice<u32>,
        src: &[Fr],
    ) -> Result<(), ProveError> {
        self.upload(stream, dst, 0, src.len() * 8, |start, out| {
            pack_fr(&src[start / 8..(start + out.len()) / 8], out);
            Ok(())
        })
    }

    /// `src` into a new device buffer, each element packed by `pack` straight into the
    /// staging on the pool, so no packed copy of the vector exists on the host. `P` is at
    /// most [`ELEMENT_WORDS`] words and divides it.
    pub(crate) fn upload_packed<S: Sync, P: Packed + Send>(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &[S],
        pack: impl Fn(&[S], &mut [P]) + Sync,
    ) -> Result<CudaSlice<u32>, ProveError> {
        use rayon::prelude::*;
        const PIECE: usize = 1 << 12;
        let per = core::mem::size_of::<P>() / 4;
        debug_assert!(ELEMENT_WORDS.is_multiple_of(per));
        let words = src.len() * per;
        let mut dst = stream
            .alloc_zeros::<u32>(words.max(1))
            .map_err(|e| drv("allocate a resident vector", e))?;
        self.upload(stream, &mut dst, 0, words, |start, out| {
            // SAFETY: `Packed` promises plain `u32` words with no padding and every bit
            // pattern valid, and a chunk is a whole number of `P` because the buffer is a
            // whole number of `ELEMENT_WORDS`; 4-byte aligned like any `u32` slice.
            let out = unsafe {
                core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<P>(), out.len() / per)
            };
            let src = &src[start / per..][..out.len()];
            out.par_chunks_mut(PIECE)
                .zip(src.par_chunks(PIECE))
                .for_each(|(d, s)| pack(s, d));
            Ok(())
        })?;
        Ok(dst)
    }
}

impl Drop for Staging {
    /// A copy may still be reading either buffer, and freeing pinned memory under a DMA is
    /// undefined, so both are waited on first.
    fn drop(&mut self) {
        for e in &self.done {
            let _ = e.synchronize();
        }
    }
}

/// `src` into `out` as [`PackedFr`] words, on the pool for a chunk large enough to be
/// worth it.
fn pack_fr(src: &[Fr], out: &mut [u32]) {
    use rayon::prelude::*;
    const PIECE: usize = 1 << 14;
    // SAFETY: `PackedFr` is eight `u32` with no padding and no invalid bit patterns, and
    // `out` holds exactly `src.len() * 8` of them, 4-byte aligned like any `u32` slice.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<PackedFr>(), out.len() / 8)
    };
    dst.par_chunks_mut(PIECE)
        .zip(src.par_chunks(PIECE))
        .for_each(|(d, s)| PackedFr::pack_into(s, d));
}

/// `count` powers of `base` from `base^0`, packed and uploaded through `staging` a chunk at
/// a time, so a 2^24 domain's twiddle and coset tables never exist whole on the host. The
/// values are `snarkrs_field::powers`', which `Domain::twiddles` builds, element for element:
/// each piece starts from `base^start` and runs the product, and a field element has one
/// representation however it was reached.
fn upload_powers(
    stream: &Arc<CudaStream>,
    staging: &mut Staging,
    base: Fr,
    count: usize,
) -> Result<CudaSlice<u32>, ProveError> {
    let mut dst = stream
        .alloc_zeros::<u32>(count.max(1) * 8)
        .map_err(|e| drv("allocate power table", e))?;
    staging.upload(stream, &mut dst, 0, count * 8, |start, out| {
        // SAFETY: as in `pack_fr`.
        let packed = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<PackedFr>(), out.len() / 8)
        };
        fill_powers(base, start / 8, packed);
        Ok(())
    })?;
    Ok(dst)
}

/// `out[i] = base^(first + i)`, on the pool.
fn fill_powers(base: Fr, first: usize, out: &mut [PackedFr]) {
    use rayon::prelude::*;
    const PIECE: usize = 1 << 14;
    out.par_chunks_mut(PIECE)
        .enumerate()
        .for_each(|(i, piece)| {
            let mut acc = base.pow([(first + i * PIECE) as u64]);
            for p in piece {
                *p = PackedFr::from_fr(&acc);
                acc *= base;
            }
        });
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
    /// The same witness in standard (non-Montgomery) limbs, written on device by
    /// `g16_mont_to_std` during `compute_h` and consumed by the MSM stage through
    /// [`HHandle::witness_std_for`], so the witness crosses PCIe exactly once per proof.
    witness_std: CudaSlice<u32>,
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
    /// The witness and row staging of [`CudaStages::compute_h_from_rows`], made there on
    /// its first use of this scratch, or by [`CudaStages::reserve_scratch`].
    staging: Option<Staging>,
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
    work: crate::msm::Work,
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
    mont_to_std: CudaFunction,
    rows_abc: CudaFunction,
    gnark_coeffs: CudaFunction,
    /// A key with no constraint matrices, a gnark one: stage 0 is the caller's rows, through
    /// [`Self::compute_h_from_rows`], and [`Self::compute_h`] refuses.
    rows_only: bool,

    /// Largest number of NTT passes one launch can fuse, from the device's own shared
    /// memory attribute rather than hardcoded.
    max_fused: u32,
    /// Threads per block: [`PREFERRED_BLOCK`] clamped by the device limit.
    block: u32,

    domain: Domain,
    n_vars: usize,
    /// snarkjs' `inc`, from `Domain::proving_coset_shift`.
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
    /// Kernel launches issued so far, which a device test holds [`Self::launch_count`] to.
    launches: AtomicUsize,
    /// A cap on the device memory a new scratch may take, below what the driver says is
    /// free; see [`Self::with_vram_limit`].
    vram_limit: Option<u64>,
}

impl CudaStages {
    /// Disable witness classification while preserving the reuse fingerprint.
    pub fn constant_work(mut self) -> Self {
        self.work = crate::msm::Work::Constant;
        self
    }

    /// Allocate a proof's scratch only when it fits in at most `bytes`, or in what the
    /// driver reports free less a headroom when that is less, and refuse the proof before
    /// allocating otherwise. A pooled scratch is reused without asking.
    pub fn with_vram_limit(mut self, bytes: u64) -> Self {
        self.vram_limit = Some(bytes);
        self
    }

    /// Compiles `kernels::unit_stages()` and uploads everything witness independent.
    ///
    /// The compile is the expensive half: NVRTC over the gather plus NTT plus pointwise
    /// unit is 1.3 s on a T4 ([`crate::context::Cuda::compile`] has the measured table),
    /// hundreds of times a whole proof at the small domains. It belongs here and nowhere
    /// else.
    pub fn new(cuda: &Cuda, pk: &ProvingKey) -> Result<Self, ProveError> {
        Self::from_module(cuda, Self::compile(cuda)?, pk)
    }

    /// Compile the stages translation unit on its own, so a process that prepares several
    /// keys pays NVRTC and the driver's ptxas once. Twin of [`crate::msm::CudaMsm::compile`],
    /// and the same argument, though the numbers here are far smaller: the stages unit has
    /// no `Fq2` curve arithmetic to inline.
    pub fn compile(cuda: &Cuda) -> Result<Arc<CudaModule>, ProveError> {
        cuda.compile("stages", &kernels::unit_stages())
            .map_err(ProveError::from)
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
                .map_err(|e| drv(&format!("kernel {name} missing"), e))
        };
        let gather = func("g16_gather_abc")?;
        let head = func("g16_ntt_head")?;
        let tail = func("g16_ntt_tail")?;
        let h_join = func("g16_h_join")?;
        let mont_to_std = func("g16_mont_to_std")?;
        let rows_abc = func("g16_rows_abc")?;
        let gnark_coeffs = func("g16_gnark_coeffs")?;

        let ctx = cuda.context().clone();
        let stream = cuda.stream().clone();

        // Elements of 32 bytes that fit in one block's shared memory, as a power of two.
        //
        // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, and deliberately NOT
        // ..._MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. The first is the 48 KB every architecture
        // from Volta up gives a block for free; the second is the 64 KB (Turing) to 163 KB
        // (A100) ceiling that needs a per-function cudaFuncSetAttribute opt in. Taking that
        // opt in would buy one extra fused pass, which at 2^18 turns an 18 = 9 + 9 split
        // into 18 = 9 + 9 again (two batches either way), so it changes nothing at the sizes
        // this prover actually runs, in exchange for an architecture-dependent limit the
        // host would have to get right on every card. 48 KB holds 1536 Fr, and the largest power of
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
        if domain.size() != pk.domain_size {
            return Err(bad(format!(
                "domain size {} is not a power of two",
                pk.domain_size
            )));
        }
        // Every kernel indexes with a 32-bit `gid` and a grid dimension is a u32. A domain
        // that did not fit would not fail, it would wrap and gather the wrong rows.
        if u32::try_from(domain.size()).is_err() || u32::try_from(pk.n_vars).is_err() {
            return Err(bad(format!(
                "domain {} / n_vars {} exceeds the u32 indexing the kernels use",
                domain.size(),
                pk.n_vars
            )));
        }

        // See `Domain::proving_coset_shift` for why this shift and not another.
        let coset_shift = domain
            .proving_coset_shift()
            .map_err(|e| bad(format!("no 2n-th root of unity: {e}")))?;

        // A gnark key's matrices stay in gnark's R1CS. Its stage 0 is the caller's rows, so
        // there is nothing to gather from, and the three tables go up a chunk at a time:
        // at 2^24 they are 1 GB, and the host this key is proved on may not have it spare.
        let rows_only = pk.h_basis != HBasis::SnarkjsOddCoset;
        if !rows_only {
            pk.coeffs
                .check_structure(domain.size(), pk.n_vars)
                .map_err(|e| bad(e.to_string()))?;
        }
        // The tables and the matrices' values are packed straight into the staging: none
        // of them exists on the host as a whole vector. A gnark key has no values here.
        let n = domain.size();
        let (a, b) = (Matrix::A.index(), Matrix::B.index());
        let nnz = pk.coeffs.value_id[a].len().max(pk.coeffs.value_id[b].len());
        let mut staging = Staging::for_key(&ctx, n.max(nnz) * 8)?;
        let tw_fwd = upload_powers(&stream, &mut staging, domain.group_gen(), n / 2)?;
        let tw_inv = upload_powers(&stream, &mut staging, domain.group_gen_inv(), n / 2)?;
        // shift^j for j in [0, n). Not the twiddles: the twiddles are powers of the domain's
        // own root of unity and this is a primitive 2n-th root, so no table snarkrs-field
        // builds can be reused.
        let coset_pows = upload_powers(&stream, &mut staging, coset_shift, n)?;
        if rows_only {
            drop(staging);
            let empty = || upload_words(&stream, &[]);
            return Ok(Self {
                work: crate::msm::Work::Variable,
                row_ptr: [empty()?, empty()?],
                signal: [empty()?, empty()?],
                value: [empty()?, empty()?],
                tw_fwd,
                tw_inv,
                coset_pows,

                n_vars: pk.n_vars,
                batches: split_passes(domain.log_size(), max_fused),
                coset_shift,
                domain,

                ctx,
                stream,
                module,
                gather,
                head,
                tail,
                h_join,
                mont_to_std,
                rows_abc,
                gnark_coeffs,
                rows_only,
                max_fused,
                block,
                pool: Arc::new(Mutex::new(Vec::new())),
                last_upload_us: AtomicU64::new(0),
                launches: AtomicUsize::new(0),
                vram_limit: None,
            });
        }

        // Each nonzero's interned value, looked up as it is packed.
        let values = &pk.coeffs.values;
        let mut up_values = |m: usize| {
            staging.upload_packed(&stream, &pk.coeffs.value_id[m], |ids, out| {
                for (d, &id) in out.iter_mut().zip(ids) {
                    *d = PackedFr::from_fr(&values[id as usize]);
                }
            })
        };
        let value = [up_values(a)?, up_values(b)?];
        drop(staging);
        // Already plain words, so the driver copies them from where they are.
        let up_u32 = |data: &[u32]| upload_words(&stream, data);
        let row_ptr = [
            up_u32(&pk.coeffs.row_ptr[a])?,
            up_u32(&pk.coeffs.row_ptr[b])?,
        ];
        let signal = [up_u32(&pk.coeffs.signal[a])?, up_u32(&pk.coeffs.signal[b])?];

        Ok(Self {
            work: crate::msm::Work::Variable,
            row_ptr,
            signal,
            value,
            tw_fwd,
            tw_inv,
            coset_pows,

            n_vars: pk.n_vars,
            batches: split_passes(domain.log_size(), max_fused),
            coset_shift,
            domain,

            ctx,
            stream,
            module,
            gather,
            head,
            tail,
            h_join,
            mont_to_std,
            rows_abc,
            gnark_coeffs,
            rows_only,
            max_fused,
            block,
            pool: Arc::new(Mutex::new(Vec::new())),
            last_upload_us: AtomicU64::new(0),
            launches: AtomicUsize::new(0),
            vram_limit: None,
        })
    }

    /// Allocate one proof's scratch now, rows staging included, and pool it, so the first
    /// proof allocates nothing: what a prover that reserves its device memory up front calls
    /// once after construction.
    pub(crate) fn reserve_scratch(&self) -> Result<(), ProveError> {
        let mut sc = self.take_scratch()?;
        if sc.staging.is_none() {
            sc.staging = Some(Staging::new(&self.ctx)?);
        }
        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize scratch reservation", e))?;
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(sc);
        Ok(())
    }

    /// The device buffers [`Self::reserve_scratch`] allocates for a domain of `n` and
    /// `n_vars` wires, in bytes, as `take_scratch` sizes them: the witness in both forms,
    /// six domain vectors and the one-element placeholder. The pinned staging is host
    /// memory and not counted.
    pub(crate) fn scratch_buffers(n: usize, n_vars: usize) -> Vec<u64> {
        let fr = |elems: usize| elems.max(1) as u64 * core::mem::size_of::<PackedFr>() as u64;
        let mut out = vec![fr(n_vars), fr(n_vars)];
        out.extend([fr(n); 6]);
        out.push(fr(1));
        out
    }

    /// The device buffers [`Self::from_module`] keeps for `pk`, in bytes: both twiddle
    /// tables, the coset powers, and each matrix's values, row pointers and signals. A
    /// gnark key's are [`Self::rows_table_buffers`].
    pub(crate) fn key_buffers(pk: &ProvingKey) -> Vec<u64> {
        let n = pk.domain_size;
        if pk.h_basis != HBasis::SnarkjsOddCoset {
            return Self::rows_table_buffers(n);
        }
        let fr = |elems: usize| elems.max(1) as u64 * core::mem::size_of::<PackedFr>() as u64;
        let u32s = |words: usize| words.max(1) as u64 * 4;
        let mut out = vec![fr(n / 2), fr(n / 2), fr(n)];
        for m in Matrix::BOTH.map(Matrix::index) {
            out.push(fr(pk.coeffs.value_id[m].len()));
            out.push(u32s(pk.coeffs.row_ptr[m].len()));
            out.push(u32s(pk.coeffs.signal[m].len()));
        }
        out
    }

    /// The device buffers a gnark key's [`Self::from_module`] keeps, in bytes: both twiddle
    /// tables, the coset powers, and the six one-word placeholders where a zkey's matrices
    /// go.
    pub(crate) fn rows_table_buffers(n: usize) -> Vec<u64> {
        let fr = |elems: usize| elems.max(1) as u64 * core::mem::size_of::<PackedFr>() as u64;
        let mut out = vec![fr(n / 2), fr(n / 2), fr(n)];
        out.extend([4; 6]);
        out
    }

    pub fn domain_size(&self) -> usize {
        self.domain.size()
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
    ///
    /// The fused path: the witness mont-to-std, the gather, and one launch per batch for
    /// each of the six transforms. `G16_CUDA_UNFUSED` adds the standalone `g16_h_join`, and
    /// that is read from the environment inside `compute_h` rather than here. The device
    /// test `compute_h_issues_launch_count_launches` counts the launches themselves.
    pub fn launch_count(&self) -> usize {
        2 + 6 * self.batches.len()
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
        if self.rows_only {
            return Err(bad(
                "a gnark key carries no constraint matrices; prove it from A.w and B.w with \
                 `h_and_msms_from_rows`",
            ));
        }
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size();
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
        // Fingerprint, and classify only in variable work. The prefix is what
        // lets the MSM stage size its windows for the general scalars only; the fold is
        // what lets it prove the witness it was handed is the one this scratch holds.
        let (witness_prefix, witness_fold) = witness_metadata(witness, self.work);
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

        // Standard-form witness for stage 5-8, written on device so the MSM never
        // re-packs or re-uploads the witness. Costs one elementwise kernel over n_vars;
        // its time lands in the gather bucket, which is where the witness plumbing is
        // accounted anyway.
        self.launch_mont_to_std(&sc, self.n_vars)?;

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
                witness_prefix,
                witness_fold,
                n_vars: self.n_vars,
                reuse: true,
            }),
        })
    }

    /// Stages 0 to 4 with stage 0 from the caller: `a_rows[i]` and `b_rows[i]` are
    /// constraint `i`'s A.w and B.w, at most `domain_size` of each, and every row past them
    /// is zero. With `gnark`, `H` then goes on to the basis a gnark key's `Z` is laid out
    /// for (see `g16_gnark_coeffs` in `kernels/rows.cu`): coefficients times `-2`, bit
    /// reversed, coefficient `n - 1` checked zero here. Otherwise it stays on the odd coset,
    /// as [`Self::compute_h`] leaves it.
    ///
    /// The witness and both row vectors cross through the scratch's [`Staging`] pair, so
    /// beyond the caller's own buffers this holds 64 MiB of pinned host memory, not a copy
    /// of either. The handle carries the standard-form witness for the MSMs, which read it
    /// through [`HHandle::witness_std`]; [`HHandle::witness_std_for`] refuses it, since no
    /// fingerprint was taken.
    ///
    /// Timings as [`Self::compute_h`]: `gather_us` is the staged uploads, overlapped with the
    /// host's packing, plus the witness conversion and the row kernel; `ntt_us` is the six
    /// transforms and, with `gnark`, the inverse one and the reordering after them.
    pub fn compute_h_from_rows(
        &self,
        witness: &[Fr],
        a_rows: &[Fr],
        b_rows: &[Fr],
        gnark: bool,
        t: &mut StageTimings,
    ) -> Result<HHandle, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size();
        let rows = a_rows.len();
        if b_rows.len() != rows || rows > n {
            return Err(bad(format!(
                "A.w has {rows} rows and B.w has {}; both must be one per constraint, at most \
                 the domain size {n}",
                b_rows.len()
            )));
        }
        // Owned by the handle from here, so the scratch goes back to the pool on every
        // return below, not only on success.
        let mut handle = HHandle {
            scratch: Some(self.take_scratch()?),
            pool: Arc::clone(&self.pool),
            stream: Arc::clone(&self.stream),
            len: n,
            witness_prefix: Vec::new(),
            witness_fold: 0,
            n_vars: self.n_vars,
            reuse: false,
        };
        let sc = handle.scratch.as_mut().expect("just taken");
        if sc.staging.is_none() {
            sc.staging = Some(Staging::new(&self.ctx)?);
        }
        let fused = std::env::var_os("G16_CUDA_UNFUSED").is_none();

        sc.ev
            .start
            .record(&self.stream)
            .map_err(|e| drv("record start event", e))?;
        {
            let Scratch {
                staging,
                witness: w,
                a,
                b,
                ..
            } = &mut *sc;
            let staging = staging.as_mut().expect("made above");
            staging.upload_fr(&self.stream, w, witness)?;
            staging.upload_fr(&self.stream, a, a_rows)?;
            staging.upload_fr(&self.stream, b, b_rows)?;
        }
        sc.ev
            .upload
            .record(&self.stream)
            .map_err(|e| drv("record upload event", e))?;

        self.launch_mont_to_std(sc, self.n_vars)?;
        self.launch_rows_abc(sc, rows, n)?;
        sc.ev
            .gather
            .record(&self.stream)
            .map_err(|e| drv("record gather event", e))?;

        self.launch_transforms(sc, n, fused)?;
        sc.ev
            .ntt
            .record(&self.stream)
            .map_err(|e| drv("record ntt event", e))?;
        if !fused {
            self.launch_h_join(sc, n)?;
            sc.ev
                .join
                .record(&self.stream)
                .map_err(|e| drv("record join event", e))?;
        }
        if gnark {
            self.launch_gnark_tail(sc, n)?;
            sc.ev
                .tail
                .record(&self.stream)
                .map_err(|e| drv("record tail event", e))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize stages 0-4 from rows", e))?;

        let upload_us = elapsed_us(&sc.ev.start, &sc.ev.upload)?;
        self.last_upload_us.store(upload_us, Ordering::Relaxed);
        t.gather_us += upload_us + elapsed_us(&sc.ev.upload, &sc.ev.gather)?;
        t.ntt_us += elapsed_us(&sc.ev.gather, &sc.ev.ntt)?;
        let before_tail = if fused {
            &sc.ev.ntt
        } else {
            t.pointwise_us += elapsed_us(&sc.ev.ntt, &sc.ev.join)?;
            &sc.ev.join
        };
        if gnark {
            t.ntt_us += elapsed_us(before_tail, &sc.ev.tail)?;
            // `bitrev(n - 1) = n - 1`, so the last slot is coefficient `n - 1`. `A*B - C`
            // has degree at most `2n - 2` and vanishes on the domain, so its quotient by `t`
            // has degree at most `n - 2`, for any rows at all, since C is A*B on them by
            // construction. A nonzero here is the arithmetic above gone wrong.
            let last = self
                .stream
                .clone_dtoh(&sc.h_mont.slice((n - 1) * 8..n * 8))
                .map_err(|e| drv("download H coefficient n - 1", e))?;
            self.stream
                .synchronize()
                .map_err(|e| drv("synchronize H coefficient n - 1", e))?;
            if last.iter().any(|&w| w != 0) {
                return Err(bad(format!(
                    "H has a nonzero coefficient {}, so the quotient by the vanishing \
                     polynomial is not exact",
                    n - 1
                )));
            }
        }
        Ok(handle)
    }

    // -----------------------------------------------------------------------
    // Launches
    // -----------------------------------------------------------------------

    /// Stage 0 from rows: C = A*B on the first `rows`, all three zero past them. One thread
    /// per element; see `g16_rows_abc` in rows.cu.
    fn launch_rows_abc(&self, sc: &Scratch, rows: usize, n: usize) -> Result<(), ProveError> {
        let (rr, nn) = (rows as u32, n as u32);
        let block = self.block.min(nn.max(1));
        let cfg = LaunchConfig {
            grid_dim: (nn.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = self.stream.launch_builder(&self.rows_abc);
        lb.arg(&sc.a).arg(&sc.b).arg(&sc.c).arg(&rr).arg(&nn);
        // SAFETY: five parameters bound in order with matching types; all three buffers are
        // domain sized, `rows <= n` was checked by the caller, and the kernel guards its
        // tail block.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_rows_abc", e))?;
        self.launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// `H` on the odd coset to a gnark key's basis: the inverse transform of `h_mont` into
    /// `t`, the 1/n riding in on the load as in stage 1, then `g16_gnark_coeffs` back into
    /// both H buffers.
    fn launch_gnark_tail(&self, sc: &Scratch, n: usize) -> Result<(), ProveError> {
        let log_n = self.domain.log_size();
        let size_inv = PackedFr::from_fr(&self.domain.size_inv());
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
                self.launch_head(&p, &sc.h_mont, &sc.t, &self.tw_inv, sc, n)?;
            } else {
                self.launch_tail(&p, &sc.t, &self.tw_inv, sc, n)?;
            }
        }
        let nn = n as u32;
        let block = self.block.min(nn.max(1));
        let cfg = LaunchConfig {
            grid_dim: (nn.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = self.stream.launch_builder(&self.gnark_coeffs);
        lb.arg(&sc.t)
            .arg(&self.coset_pows)
            .arg(&sc.h_mont)
            .arg(&sc.h_std)
            .arg(&log_n);
        // SAFETY: five parameters bound in order with matching types; `t`, `coset_pows` and
        // both H buffers are domain sized, every index the kernel forms (`bitrev(p)` and
        // `n - j` for `j > 0`) is below `n`, and it guards its tail block. `t` and the H
        // buffers are distinct allocations, so no thread reads what another writes.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_gnark_coeffs", e))?;
        self.launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Stage 0.5: the on-device Montgomery-to-standard conversion of the witness. One
    /// thread per element, no sharing; see `g16_mont_to_std` in pointwise.cu.
    fn launch_mont_to_std(&self, sc: &Scratch, n: usize) -> Result<(), ProveError> {
        let nn = n as u32;
        let block = self.block.min(nn.max(1));
        let cfg = LaunchConfig {
            grid_dim: (nn.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = self.stream.launch_builder(&self.mont_to_std);
        lb.arg(&sc.witness).arg(&sc.witness_std).arg(&nn);
        // SAFETY: three parameters bound in order with matching types; both buffers hold
        // exactly `n_vars` packed elements and the kernel's `gid >= n` guard covers the
        // tail block.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch g16_mont_to_std", e))?;
        self.launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

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
        self.launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The six transforms.
    ///
    /// `fuse_h` folds stage 4 into the store epilogue of the last batch of C's forward
    /// transform, which is legal because A and B are already fully transformed by then and
    /// their coset evaluations are exactly what `H = A*B - C` needs. It saves a whole write
    /// of C and a whole read of A, B and C.
    fn launch_transforms(&self, sc: &Scratch, n: usize, fuse_h: bool) -> Result<(), ProveError> {
        let log_n = self.domain.log_size();
        let last = self.batches.len() - 1;
        let size_inv = PackedFr::from_fr(&self.domain.size_inv());

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
        self.launches.fetch_add(1, Ordering::Relaxed);
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
        self.launches.fetch_add(1, Ordering::Relaxed);
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
        self.launches.fetch_add(1, Ordering::Relaxed);
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
        let n = self.domain.size();
        // Refused before the first allocation, as the MSM refuses a batch: a scratch the
        // card cannot hold would otherwise be the driver's out-of-memory partway through.
        let need = Self::scratch_buffers(n, self.n_vars).iter().sum::<u64>();
        let budget = crate::msm::device_budget(&self.ctx, &self.stream, self.vram_limit)?;
        if need > budget {
            return Err(crate::msm::too_big(
                need,
                budget,
                "one proof's stage scratch",
            ));
        }
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
        // DMA that the same thread issues after that write. A rows-only key stages its
        // witness through `Staging` instead and never packs into this, so it gets one word.
        let host_words = if self.rows_only {
            1
        } else {
            self.n_vars.max(1) * 8
        };
        let host_witness = unsafe { self.ctx.alloc_pinned::<u32>(host_words) }
            .map_err(|e| drv("allocate pinned witness staging", e))?;
        Ok(Scratch {
            host_witness,
            witness: words(self.n_vars)?,
            witness_std: words(self.n_vars)?,
            a: words(n)?,
            b: words(n)?,
            c: words(n)?,
            t: words(n)?,
            h_mont: words(n)?,
            h_std: words(n)?,
            unused: words(1)?,
            ev: Events::new(&self.ctx)?,
            staging: None,
        })
    }
}

// ---------------------------------------------------------------------------
// The device-resident result
// ---------------------------------------------------------------------------

/// An order-dependent fold of the raw limbs, without scalar classification.
///
/// The fold is an integrity tag, not a cryptographic hash: it exists to catch the caller
/// handing `msms` a different witness than the one `compute_h` converted (an API misuse
/// that would otherwise silently prove the wrong statement), and an accidental collision
/// under random data is a 2^-64 event. An adversary who controls the witness controls
/// both sides of the comparison anyway, so nothing is entrusted to it.
fn witness_fingerprint(witness: &[Fr]) -> u64 {
    let mut fold = 0xcbf29ce484222325u64;
    for s in witness {
        for &l in &s.0 .0 {
            fold = (fold.rotate_left(5) ^ l).wrapping_mul(0x100000001b3);
        }
    }
    fold
}

fn witness_metadata(witness: &[Fr], work: crate::msm::Work) -> (Vec<u32>, u64) {
    // Preserve witness_std_for's slice API. These bounds also safely size a
    // variable MSM, whose ones pass scans independently of the prefix.
    let prefix = crate::msm::scalar_prefix(witness, work)
        .unwrap_or_else(|| (0..=witness.len() as u32).collect());
    (prefix, witness_fingerprint(witness))
}

/// What [`snarkrs_groth16::HPoly::Device`] carries out of stage 4, under the tag [`TAG`].
///
/// Recover it with `h.device_handle::<HHandle>(snarkrs_cuda::stages::TAG)`. It owns the whole
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
    /// `witness_prefix[i]` = how many of the first `i` witness scalars are neither 0 nor
    /// 1, or the conservative bound `i` under constant work. Used by [`Self::witness_std_for`].
    witness_prefix: Vec<u32>,
    /// Order-dependent fold of the witness limbs, the cheap identity check behind
    /// [`Self::witness_std_for`].
    witness_fold: u64,
    n_vars: usize,
    /// Whether [`Self::witness_std_for`] may vouch for the witness at all. False from
    /// [`CudaStages::compute_h_from_rows`], which takes no fingerprint.
    reuse: bool,
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

    /// The standard-form witness this handle's `compute_h` left on the device, plus its
    /// general-scalar prefix, IF `witness` is the same vector it was computed from.
    ///
    /// `None` when the length or the limb fold disagrees, in which case the caller must
    /// fall back to uploading `witness` itself; using the buffer anyway would prove a
    /// different statement than the caller asked for. `G16_CUDA_WITNESS_REUSE=0` forces
    /// the fallback, as the A/B lever for measuring what the reuse is worth.
    pub fn witness_std_for(
        &self,
        witness: &[snarkrs_field::Fr],
    ) -> Option<(&CudaSlice<u32>, &[u32])> {
        if std::env::var("G16_CUDA_WITNESS_REUSE").as_deref() == Ok("0") {
            return None;
        }
        if !self.reuse || witness.len() != self.n_vars {
            return None;
        }
        let fold = witness_fingerprint(witness);
        if fold != self.witness_fold {
            return None;
        }
        Some((&self.sc().witness_std, &self.witness_prefix))
    }

    /// The standard-form witness this handle's stages left on the device, `n_vars * 8`
    /// words, with nothing checked: for the caller that ran those stages itself, as the
    /// rows path does stages 0 to 9 in one call.
    pub(crate) fn witness_std(&self) -> &CudaSlice<u32> {
        &self.sc().witness_std
    }

    /// Copies H down to the host. Tests and cross-checks want this; the proving path must
    /// not, because forcing the copy is exactly the PCIe round trip that
    /// [`snarkrs_groth16::HPoly`] exists to avoid.
    pub fn to_host(&self) -> Result<Vec<Fr>, ProveError> {
        let words = download(&self.stream, self.h_mont(), "H")?;
        let packed = from_words::<PackedFr>(&words)
            .ok_or_else(|| bad("H readback is not a whole number of field elements"))?;
        Ok(PackedFr::unpack_slice(&packed))
    }

    /// The standard-form copy, read back and validated. `None` at any index whose limbs are
    /// not a canonical residue, which would mean the reduction in `fr_from_mont` is wrong; a
    /// silent modular reduction here would hide exactly that bug.
    pub fn to_host_std(&self) -> Result<Option<Vec<Fr>>, ProveError> {
        let words = download(&self.stream, self.h_std(), "H")?;
        let packed = from_words::<PackedScalar>(&words)
            .ok_or_else(|| bad("H readback is not a whole number of field elements"))?;
        Ok(packed.iter().map(PackedScalar::to_fr).collect())
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
    use crate::as_words;

    #[test]
    fn constant_metadata_preserves_fingerprint_without_classification() {
        use crate::msm::Work;
        let witness = [
            Fr::from(0u64),
            Fr::from(1u64),
            Fr::from(2u64),
            -Fr::from(1u64),
        ];
        let variable = witness_metadata(&witness, Work::Variable);
        let constant = witness_metadata(&witness, Work::Constant);
        assert_eq!(variable.0, [0, 0, 0, 1, 2]);
        assert_eq!(constant.0, [0, 1, 2, 3, 4]);
        assert!(crate::msm::scalar_prefix(&witness, Work::Constant).is_none());
        let old_fold = witness
            .iter()
            .flat_map(|s| s.0 .0)
            .fold(0xcbf29ce484222325u64, |fold, limb| {
                (fold.rotate_left(5) ^ limb).wrapping_mul(0x100000001b3)
            });
        assert_eq!(variable.1, old_fold);
        assert_eq!(constant.1, old_fold);
        assert_eq!(witness_fingerprint(&[]), 0xcbf29ce484222325u64);
        let mut changed = witness;
        changed.swap(0, 1);
        assert_ne!(witness_fingerprint(&changed), old_fold);
        changed[3] += Fr::from(1u64);
        assert_ne!(witness_fingerprint(&changed), old_fold);
        // A conservative prefix never underallocates a variable job at an offset.
        for lo in 0..=witness.len() {
            for hi in lo..=witness.len() {
                assert!(constant.0[hi] - constant.0[lo] >= variable.0[hi] - variable.0[lo]);
            }
        }
    }

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

    /// `g16_gnark_coeffs` on the host, over the natural-order output of the inverse
    /// transform `launch_gnark_tail` runs first.
    fn gnark_coeffs_model(g: &[Fr], pows: &[Fr], log_n: u32) -> Vec<Fr> {
        let n = 1usize << log_n;
        (0..n)
            .map(|p| {
                let j = if log_n == 0 {
                    0
                } else {
                    ((p as u32).reverse_bits() >> (32 - log_n)) as usize
                };
                if j == 0 {
                    g[j]
                } else {
                    -(g[j] * pows[n - j])
                }
            })
            .collect()
    }

    /// The gnark tail the kernels run is the CPU backend's `to_gnark_coefficients`, element
    /// for element: the natural inverse transform plus `g16_gnark_coeffs`' reorder and
    /// `-shift^(n - j)` unscale against the DIF inverse plus the inverse coset table. Any
    /// vector does, as both are linear; at `n = 1` there is no reversal and no unscale.
    #[test]
    fn the_gnark_tail_is_the_cpu_backends() {
        use snarkrs_ntt::{CpuNtt, Direction};
        let ntt = CpuNtt::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for log_n in [0u32, 1, 2, 5, 10, 13] {
            let domain = Domain::new(1 << log_n).unwrap();
            let shift = domain.proving_coset_shift().unwrap();
            let n = domain.size();
            let v: Vec<Fr> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Fr::from(seed) * Fr::from(seed.rotate_left(17))
                })
                .collect();

            let mut want = v.clone();
            ntt.intt_to_bitrev(&domain, &mut want);
            ntt.coset_scale_bitrev(&domain, &mut want, shift.inverse().unwrap());

            let mut g = v.clone();
            ntt.ntt(&domain, &mut g, Direction::Inverse);
            let mut pows = vec![PackedFr::default(); n];
            fill_powers(shift, 0, &mut pows);
            let pows = PackedFr::unpack_slice(&pows);
            assert_eq!(gnark_coeffs_model(&g, &pows, log_n), want, "n = {n}");
        }
    }

    /// The chunked tables are `Domain`'s, whichever piece and chunk boundary a power falls on.
    #[test]
    fn staged_power_tables_are_the_domains() {
        let domain = Domain::new(1 << 16).unwrap();
        let shift = domain.proving_coset_shift().unwrap();
        for (base, want) in [
            (domain.group_gen(), domain.twiddles()),
            (domain.group_gen_inv(), domain.twiddles_inv()),
            (shift, snarkrs_field::powers(shift, domain.size())),
        ] {
            // Two chunks that split a piece, as a staged upload's would.
            let cut = want.len() / 2 + 77;
            let mut got = vec![PackedFr::default(); want.len()];
            let (lo, hi) = got.split_at_mut(cut);
            fill_powers(base, 0, lo);
            fill_powers(base, cut, hi);
            assert_eq!(PackedFr::unpack_slice(&got), want);
        }
        let src: Vec<Fr> = (0..40_000u64).map(Fr::from).collect();
        let mut words = vec![0u32; src.len() * 8];
        pack_fr(&src, &mut words);
        assert_eq!(words, as_words(&PackedFr::pack_slice(&src)));
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

    /// [`CudaStages::launch_count`] is what [`CudaStages::compute_h`] actually issues, on a
    /// one-batch and a two-batch domain, so a launch added to the pipeline without the count
    /// fails here rather than only in a report.
    #[test]
    #[ignore = "requires a CUDA device and tiny_mul and sha256 under G16_ARTIFACTS or bench/artifacts"]
    fn compute_h_issues_launch_count_launches() {
        let root = std::env::var_os("G16_ARTIFACTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
            });
        let cuda = Cuda::new(0).expect("CUDA device");
        let unfused = std::env::var_os("G16_CUDA_UNFUSED").is_some();
        for name in ["tiny_mul", "sha256"] {
            let dir = root.join(name);
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("proving key");
            let w = snarkrs_formats::wtns::Witness::load(&dir.join("circuit.wtns"))
                .expect("witness")
                .0;
            let stages = CudaStages::new(&cuda, &pk).unwrap();
            for _ in 0..2 {
                let before = stages.launches.load(Ordering::Relaxed);
                stages.compute_h(&w, &mut StageTimings::default()).unwrap();
                let issued = stages.launches.load(Ordering::Relaxed) - before;
                eprintln!(
                    "{name}: {issued} launches, batches {:?}",
                    stages.pass_batches()
                );
                assert_eq!(issued, stages.launch_count() + unfused as usize, "{name}");
            }
        }
    }
}
