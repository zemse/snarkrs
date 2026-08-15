//! Stages 5 to 9: the five Pippenger multi-scalar multiplications, on CUDA.
//!
//! Host twin of `crates/g16-metal/src/msm.rs`, driving `kernels/msm.cu`. The algorithm and
//! every decision behind it are documented at the top of that kernel file, and it is
//! deliberately identical to the Metal one: a counting sort by bucket index built from
//! 32-bit atomics on plain counters, so no bucket is ever written by two threads; a
//! segmented accumulation that gives every thread a fixed-length slice of the sorted
//! entries, so per-thread work is uniform even when one bucket holds a tenth of the input;
//! and a block-level reduction that hands the host one point per window. Two backends
//! running two different MSMs would measure nothing, which is why nothing here is
//! "improved".
//!
//! # What this file owns, and where it differs from the Metal twin
//!
//! The Metal host has the shape it does because a Metal command buffer costs 0.16 ms to
//! commit and wait on and does not get cheaper with less work in it, so all five MSMs and
//! their 37 dispatches go into ONE command buffer with one wait. CUDA has no such object.
//! A launch on a warm stream is a few microseconds of host time and is asynchronous, so the
//! same 37 launches queue up without the host blocking and the only wait is the single
//! `synchronize` before the readback. The thing to avoid here is not a submission, it is a
//! *synchronize* in the middle, and there is none.
//!
//! Two consequences follow, and both are simplifications rather than losses:
//!
//! * There is no scratch pool. The Metal one exists because a `memcpy` into a freshly
//!   allocated shared buffer runs at 15.2 GB/s against 54.8 GB/s into one already touched,
//!   so the first-touch page faults had to be paid once rather than per proof. `cudarc`'s
//!   `CudaStream::alloc` calls `cuMemAllocAsync` where the driver supports it, which is the
//!   driver's own stream-ordered memory pool: the second proof's allocation is served out
//!   of the first proof's freed blocks with no trip to the OS. Reimplementing that in Rust
//!   would be slower and would add a way to hand a live buffer to a second proof.
//! * `MsmParams` reaches the kernel by value through the parameter space, so there is no
//!   buffer index to keep in sync with the kernel's `[[buffer(n)]]` attributes and none of
//!   that class of mistake is expressible.
//!
//! # cudaMalloc does not zero, and this is where that bites
//!
//! On Apple silicon a fresh `MTLBuffer` is zero-filled and the Metal MSM leans on it: the
//! all-zero encoding is the point at infinity, so a fresh bucket array is already an array
//! of identities and a fresh counter array is already zero. `cudaMalloc` hands back
//! whatever the last tenant left. An uninitialised `ZZ` limb array is a perfectly plausible
//! looking field element, so the failure mode is a proof that does not verify, with no
//! crash and nothing to point at, on some witnesses and not others.
//!
//! Three things guard that here. Every allocation goes through [`CudaMsm::zeros`], which is
//! `alloc_zeros`, a stream-ordered `cuMemsetD8Async` over the whole block. The count
//! histogram is additionally cleared by `zero_u32` before `msm_count` accumulates onto it.
//! Every bucket array is additionally run through `msm_clear_g1`/`msm_clear_g2` before an
//! accumulation writes into it. The last two are redundant with `alloc_zeros` as this file
//! stands, and are kept anyway: they are the steps the Metal schedule has, and the moment
//! anyone reuses a buffer across two MSMs they stop being redundant and become the only
//! thing between this backend and a silent wrong answer.
//!
//! # Timings
//!
//! [`StageTimings::msm_us`] is filled from a CUDA event pair spanning every launch, plus
//! the host-side scalar pack and upload and the readback and Horner tail, both on a wall
//! clock. Host time spent *issuing* launches is deliberately not added: the launches are
//! asynchronous, so that time overlaps the device work the event pair already counted.
//! [`CudaMsm::last_device_us`] exposes the device half on its own, which is a number the
//! Metal backend cannot produce at all.
//!
//! An event measures the *stream*, not the call. Two proofs in flight on one circuit share
//! the stream, so each one's events also span whatever the other queued in between. The
//! results stay correct, because launches on a stream execute in issue order and the two
//! proofs own disjoint allocations, but the timings are only meaningful when proofs are run
//! serially. Same caveat and same reason as `stages::CudaStages::compute_h`.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ark_ff::{AdditiveGroup, One, Zero};
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PushKernelArg,
};

use g16_core::{HPoly, MsmOutputs, ProveError, StageTimings};
use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use g16_gpu_layout::{Packed, PackedFq, PackedFq2, PackedG1Affine, PackedG2Affine, PackedScalar};
use g16_zkey::ProvingKey;

use crate::stages::{HHandle, TAG};
use crate::{as_words, from_words, kernels, Cuda};

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "cuda",
        reason: reason.into(),
    }
}

/// Wraps a driver failure with the operation that produced it. The driver's own message is
/// `CUDA_ERROR_ILLEGAL_ADDRESS` and nothing else, so without the `what` there is no way to
/// tell which of thirty-seven launches died.
fn drv(what: &str, e: impl std::fmt::Display) -> ProveError {
    bad(format!("{what}: {e}"))
}

// ---------------------------------------------------------------------------
// Window sizing
// ---------------------------------------------------------------------------

/// Bits the signed recoding is laid out over: 254 for BN254's `Fr`, plus one so the carry
/// out of the top window is provably zero. Must match `g16_msm::RECODE_BITS` and the
/// arithmetic in `sc_signed_digit`.
const RECODE_BITS: usize = 255;

/// Caps the bucket array at 2^15 points per window.
const MAX_WINDOW: u32 = 16;

/// Threads per block in the reduction and the ones kernel. Must be **at most** `REDUCE_TG`
/// in `kernels/msm.cu`, which is the size of the `__shared__` point array those kernels
/// index by `threadIdx.x`. A wider block writes past the end of shared memory: no compile
/// error, no launch error, just wrong points and a proof that fails to verify. The Metal
/// twin proved that experimentally, by setting the host constant to 128 with the shader
/// left at 64 and watching `H` stay bit-exact while all five MSM outputs went wrong.
/// Guarded by [`tests::the_kernel_declares_the_same_block_sizes`].
const REDUCE_TG: u32 = 64;

/// Shared array size of the prefix-sum kernel. Same contract as [`REDUCE_TG`].
const SCAN_TG: u32 = 256;

/// Entries one thread of the segmented accumulation owns.
///
/// Accumulation costs `SLICE_LEN` mixed additions per thread and the merge costs
/// `max_bucket_count / SLICE_LEN` full additions for the fattest bucket, so the balance
/// point is near the square root of the worst occupancy, which is a few hundred on these
/// artifacts. 64 sits under that on purpose: it also keeps the thread count high enough to
/// fill the machine at the smaller domains, where there are only a few thousand slices to
/// begin with. That argument is about the algorithm rather than about Apple silicon, which
/// is why the constant carries over unchanged. Sweep it with `G16_CUDA_MSM_L`.
const SLICE_LEN: usize = 64;

fn slice_len() -> usize {
    std::env::var("G16_CUDA_MSM_L")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(SLICE_LEN)
}

/// `G16_CUDA_MSM_LEGACY_ACC=1` swaps the segmented accumulation for the simple
/// one-thread-per-bucket kernel. Kept only so the load-imbalance claim at the top of
/// `kernels/msm.cu` can be reproduced on NVIDIA rather than taken on trust from an M2 Max.
fn legacy_accumulate() -> bool {
    std::env::var("G16_CUDA_MSM_LEGACY_ACC").as_deref() == Ok("1")
}

/// Meaningful bits of the top window at width `c`.
///
/// The digits are laid out over [`RECODE_BITS`] bits in `W = ceil(255 / c)` windows, and
/// `W * c` overshoots 255 by up to `c - 1`, so the top window has only
/// `255 - (W - 1) * c` meaningful bits and its digits crowd into `2^(top_bits - 1)` buckets
/// instead of `2^(c-1)`.
fn top_bits(c: u32) -> u32 {
    let w = RECODE_BITS.div_ceil(c as usize);
    (RECODE_BITS - (w - 1) * c as usize).min(c as usize) as u32
}

/// Window width for `m` scalars that actually reach the buckets.
///
/// `m` is the general-scalar count, not the input length. That distinction matters
/// enormously: a 140k-long witness with 2k general scalars is a 2k problem, and sizing the
/// window for 140k would allocate 16384 buckets per window and spend half a million point
/// additions reducing buckets that 34k additions filled.
///
/// # This default is a placeholder and has NOT been measured on NVIDIA
///
/// The Metal twin picks `c` from a three-term cost model whose coefficients (`MADD_US`,
/// `ROW_US`, `MERGE_US`) were fitted to an M2 Max by forcing five window widths on two
/// artifacts. Those are Apple silicon microseconds, they are meaningless on a T4, and none
/// of them are copied here. What is used instead is the textbook Pippenger heuristic,
/// `c ~ log2(m) - 3`, clamped, with one structural correction described below.
///
/// **Whoever first runs this on a real card should sweep `G16_CUDA_MSM_C` over 8..=16 on
/// each of the five MSMs and replace this function with the measured answer.** Do not
/// assume the optimum is shallow. On the M2 Max at 140k constraints the sweep gave c=8
/// 139.5 ms, c=10 152.4, c=12 272.4, c=13 127.6, c=14 308.7, c=16 336.1: a 2.4x cliff one
/// step either side of the optimum. Whatever the constants turn out to be on NVIDIA, the
/// shape of that curve is a property of the algorithm and will still be there.
///
/// # The one correction, which is structural rather than fitted
///
/// Those cliffs are not noise and they are not Apple specific. At c=12 and c=14 the top
/// window has three meaningful bits, so its digits crowd into four live buckets instead of
/// 2048 or 8192 and each of the four holds a quarter of the scalars. On the CPU that costs
/// nothing, because a serial bucket loop only cares about the total. On either GPU it is
/// the critical path: such a bucket's run spans thousands of slices, and `msm_merge_*`
/// walks a bucket's slice range in ONE thread while the rest of the machine waits.
///
/// So the heuristic's pick is stepped *down* (never up, which would only inflate the bucket
/// array) by at most three, until the top window keeps at least half its bits. At 140k
/// general scalars that turns 14 into 13, which is where the M2 Max sweep puts the optimum.
/// That agreement is a sanity check, not a measurement.
///
/// Override with `G16_CUDA_MSM_C` to sweep it.
pub fn window_size(m: usize) -> u32 {
    if let Ok(v) = std::env::var("G16_CUDA_MSM_C") {
        if let Ok(c) = v.parse::<u32>() {
            return c.clamp(2, MAX_WINDOW);
        }
    }
    // floor(log2(m)) + 1 for m >= 1, so `bits - 4` is the usual `log2(m) - 3`.
    let bits = usize::BITS - m.max(1).leading_zeros();
    let base = bits.saturating_sub(4).clamp(3, MAX_WINDOW);
    // `2 * top_bits >= c` reads as "the top window keeps at least half its bits", so its
    // fattest bucket is at most about sqrt(2^c) times fatter than a uniform one rather than
    // 2^(c-3) times.
    for c in (base.saturating_sub(3).max(3)..=base).rev() {
        if 2 * top_bits(c) >= c {
            return c;
        }
    }
    base
}

/// Blocks in the ones kernel. Enough that a 2^18-long witness gives each thread about sixty
/// scalars to scan, few enough that the host adds a few dozen points. Same formula as the
/// Metal twin: it trades device parallelism against a serial host tail, and changing it
/// changes what the two backends are comparing.
fn ones_groups_for(n: usize) -> usize {
    n.div_ceil(REDUCE_TG as usize * 64).clamp(1, 64)
}

// ---------------------------------------------------------------------------
// Device-side structs
// ---------------------------------------------------------------------------

/// Mirrors `struct MsmParams` in `kernels/msm.cu`: ten `u32`, 40 bytes, no padding on
/// either side.
///
/// Passed **by value**. The driver copies these 40 bytes into the kernel's parameter space
/// verbatim, so the Rust layout has to match byte for byte; a field inserted on one side
/// only is read as some other field's value, not as a type error.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct MsmParams {
    n: u32,
    c: u32,
    n_windows: u32,
    n_buckets: u32,
    cap: u32,
    scalar_off: u32,
    base_off: u32,
    ones_groups: u32,
    slice_len: u32,
    slices: u32,
}

// SAFETY: `DeviceRepr` marks a type that may be handed to a kernel as a by-value argument.
// `MsmParams` is `repr(C)`, is ten `u32` with no padding and no invalid bit patterns, and
// its size is asserted below to be the 40 bytes the kernel's parameter slot expects.
unsafe impl DeviceRepr for MsmParams {}

/// Mirrors `struct MsmEntry` in `kernels/msm.cu`, which is MSL's `uint2` in the twin. The
/// host never reads or writes one; this exists so the entry array is sized from the
/// kernel's own record layout rather than from a hardcoded 8.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct MsmEntry {
    row: u32,
    point_and_sign: u32,
}

/// A G1 point in extended Jacobian coordinates, as the kernels write it: 128 bytes,
/// `x`, `y`, `zz`, `zzz`, with `x/zz` and `y/zzz` the affine coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PackedXyzzG1 {
    pub x: PackedFq,
    pub y: PackedFq,
    pub zz: PackedFq,
    pub zzz: PackedFq,
}

/// The G2 twin, 256 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PackedXyzzG2 {
    pub x: PackedFq2,
    pub y: PackedFq2,
    pub zz: PackedFq2,
    pub zzz: PackedFq2,
}

const _: () = {
    assert!(core::mem::size_of::<MsmParams>() == 40);
    assert!(core::mem::size_of::<MsmEntry>() == 8);
    assert!(core::mem::size_of::<PackedXyzzG1>() == 128);
    assert!(core::mem::size_of::<PackedXyzzG2>() == 256);
};

// SAFETY: both are `repr(C)` aggregates of `PackedFq`, itself eight `u32` with no padding
// and every bit pattern valid, so each is a whole number of `u32` and any word sequence off
// the device is a legal value.
unsafe impl Packed for PackedXyzzG1 {}
unsafe impl Packed for PackedXyzzG2 {}

/// XYZZ to arkworks' Jacobian, with no field inversion.
///
/// XYZZ carries the invariant `ZZ^3 = ZZZ^2`, so setting the Jacobian `Z = ZZZ` gives
/// `Z^2 = ZZ^3` and the point `(X * ZZ^2, Y * ZZ^3, ZZZ)` has `x = X*ZZ^2 / ZZ^3 = X/ZZ`
/// and `y = Y*ZZ^3 / ZZZ^3 = Y/ZZZ`, which is exactly the XYZZ point. Three
/// multiplications, versus two inversions if this went through affine, and there are
/// hundreds of these per proof.
// `&self` rather than `self`, against clippy's `wrong_self_convention`, for two reasons:
// these are 128 and 256 byte values, so a reference is what you want at the call site
// anyway, and the Metal twin declares the identical signature, which keeps the two files a
// readable diff.
#[allow(clippy::wrong_self_convention)]
impl PackedXyzzG1 {
    pub fn to_projective(&self) -> G1Projective {
        let zz = self.zz.to_fq();
        if zz.is_zero() {
            return G1Projective::zero();
        }
        let x = self.x.to_fq();
        let y = self.y.to_fq();
        let zzz = self.zzz.to_fq();
        let zz2 = zz * zz;
        G1Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz)
    }
}

#[allow(clippy::wrong_self_convention)]
impl PackedXyzzG2 {
    pub fn to_projective(&self) -> G2Projective {
        let zz = self.zz.to_fq2();
        if zz.is_zero() {
            return G2Projective::zero();
        }
        let x = self.x.to_fq2();
        let y = self.y.to_fq2();
        let zzz = self.zzz.to_fq2();
        let zz2 = zz * zz;
        G2Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz)
    }
}

const G1_WORDS: usize = core::mem::size_of::<PackedXyzzG1>() / 4;
const G2_WORDS: usize = core::mem::size_of::<PackedXyzzG2>() / 4;
const ENTRY_WORDS: usize = core::mem::size_of::<MsmEntry>() / 4;
const SCALAR_WORDS: usize = core::mem::size_of::<PackedScalar>() / 4;

// ---------------------------------------------------------------------------
// Resident inputs
// ---------------------------------------------------------------------------

/// A G1 base vector, repacked and resident on the device.
///
/// Built once per key in `prepare` and never per proof. The repack is not optional: on this
/// arkworks `G1Affine` is 72 bytes and carries an infinity flag, so a byte cast would hand
/// the kernel a 72-byte stride where it reads 64 and every point after the first would be
/// garbage. [`PackedG1Affine::from_affine`] reads the flag and maps infinity onto the
/// all-zero encoding that `aff_is_inf` tests for; snarkjs zkeys really do contain points at
/// infinity in the query vectors, so that is a live path and not a defensive one.
pub struct G1Bases {
    buf: CudaSlice<u32>,
    len: usize,
}

/// The G2 twin. `G2Affine` is 136 bytes against the 128 the kernel reads, same argument.
pub struct G2Bases {
    buf: CudaSlice<u32>,
    len: usize,
}

impl G1Bases {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl G2Bases {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Scalars in **standard** form (see `g16_gpu_layout`'s module docs for why not
/// Montgomery), owned and resident on the device.
///
/// `general_prefix[i]` is how many of the first `i` scalars are neither 0 nor 1. It is
/// built during the pack, which already walks every scalar, so it is free, and it is what
/// lets the batch size the window and the entry array for the work that actually reaches
/// the buckets rather than for the input length. A witness is typically over 99% zeros and
/// ones, so the difference is two orders of magnitude.
pub struct ScalarBuf {
    buf: CudaSlice<u32>,
    len: usize,
    general_prefix: Option<Vec<u32>>,
}

impl ScalarBuf {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// A borrowed view, which is what a [`Job`] takes.
    pub fn as_scalars(&self) -> Scalars<'_> {
        Scalars {
            buf: &self.buf,
            len: self.len,
            general_prefix: self.general_prefix.as_deref(),
        }
    }
}

/// Standard-form scalars on the device, borrowed rather than owned.
///
/// The point of the borrow is stage 9. `H` comes out of the stages lane already on the
/// device and already converted to standard form (`stages::HHandle::h_std`), so copying it
/// down to the host to pack and upload it again would be a PCIe round trip in both
/// directions for a buffer that is already exactly right. [`HPoly`] exists to avoid that,
/// and this type is what consumes it.
#[derive(Clone, Copy)]
pub struct Scalars<'a> {
    buf: &'a CudaSlice<u32>,
    len: usize,
    general_prefix: Option<&'a [u32]>,
}

impl<'a> Scalars<'a> {
    /// Wraps a device buffer that already holds `len` standard-form scalars.
    ///
    /// Errors rather than truncates on a buffer too short for `len`. A kernel reading past
    /// the end of a `CudaSlice` is undefined and, because the allocation is very likely
    /// still mapped, would show up as a wrong proof rather than as a fault.
    ///
    /// No host classification is possible without reading the values back, so a buffer
    /// wrapped here reports every scalar as general. That is exactly right for `H`, whose
    /// evaluations are dense; for anything else it only oversizes the entry array and the
    /// window, which is safe in both directions.
    pub fn device_std(buf: &'a CudaSlice<u32>, len: usize) -> Result<Self, ProveError> {
        if buf.len() < len * SCALAR_WORDS {
            return Err(bad(format!(
                "device scalar buffer holds {} words; {len} scalars need {}",
                buf.len(),
                len * SCALAR_WORDS
            )));
        }
        Ok(Self {
            buf,
            len,
            general_prefix: None,
        })
    }

    /// Like [`Self::device_std`], but with the host-side zero/one classification the
    /// producer built while it still had the scalars in hand. This is the witness-reuse
    /// entry point: `compute_h` uploads the witness once, converts it to standard form on
    /// device, and hands both the buffer and the prefix here through `stages::HHandle`.
    pub fn device_std_with_prefix(
        buf: &'a CudaSlice<u32>,
        len: usize,
        prefix: &'a [u32],
    ) -> Result<Self, ProveError> {
        if prefix.len() != len + 1 {
            return Err(bad(format!(
                "general prefix holds {} entries; {len} scalars need {}",
                prefix.len(),
                len + 1
            )));
        }
        let mut s = Self::device_std(buf, len)?;
        s.general_prefix = Some(prefix);
        Ok(s)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// General scalars in `range`, or the whole range's length when the buffer came from the
    /// device and was never classified on the host. Overestimating is safe.
    fn general_in(&self, range: &Range<usize>) -> usize {
        match self.general_prefix {
            Some(p) => (p[range.end] - p[range.start]) as usize,
            None => range.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// One MSM. `scalar_off` and `base_off` exist so the L MSM, whose scalars are the private
/// suffix of the witness, can share the witness buffer with the A and B MSMs instead of
/// uploading a second copy of 140k scalars across PCIe.
pub struct JobG1<'a> {
    pub bases: &'a G1Bases,
    pub base_off: usize,
    pub scalars: Scalars<'a>,
    pub scalar_off: usize,
    pub n: usize,
}

pub struct JobG2<'a> {
    pub bases: &'a G2Bases,
    pub base_off: usize,
    pub scalars: Scalars<'a>,
    pub scalar_off: usize,
    pub n: usize,
}

pub enum Job<'a> {
    G1(JobG1<'a>),
    G2(JobG2<'a>),
}

#[derive(Clone, Copy, Debug)]
pub enum MsmResult {
    G1(G1Projective),
    G2(G2Projective),
}

impl MsmResult {
    pub fn g1(self) -> Result<G1Projective, ProveError> {
        match self {
            MsmResult::G1(p) => Ok(p),
            MsmResult::G2(_) => Err(bad("expected a G1 MSM result, got G2")),
        }
    }
    pub fn g2(self) -> Result<G2Projective, ProveError> {
        match self {
            MsmResult::G2(p) => Ok(p),
            MsmResult::G1(_) => Err(bad("expected a G2 MSM result, got G1")),
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// A loaded kernel plus the largest block the driver will accept for it.
///
/// The CUDA analogue of Metal's `maxTotalThreadsPerThreadgroup`, queried for the same
/// reason: never hardcode 1024. A launch whose `blockDim.x * registers_per_thread` exceeds
/// the SM's 65536-register file is rejected at launch with
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, and the G2 kernels here sit at the 255-register
/// ceiling, which puts their limit at 256 threads. That is above every block size this file
/// asks for, so the clamp does not bind today; it is here so that it still does not bind on
/// a card or a CUDA version where the compiler spends registers differently.
struct Kernel {
    f: CudaFunction,
    max_block: u32,
}

impl Kernel {
    fn load(module: &Arc<CudaModule>, name: &'static str) -> Result<Self, ProveError> {
        let f = module
            .load_function(name)
            .map_err(|e| bad(format!("kernel {name} missing: {e}")))?;
        let max_block = f
            .max_threads_per_block()
            .map_err(|e| drv(&format!("query max block size of {name}"), e))?
            .max(1) as u32;
        Ok(Self { f, max_block })
    }

    /// One thread per item, `prefer` threads a block, clamped by what the kernel accepts.
    ///
    /// The grid is rounded up and floored at one block. Every kernel this is used with
    /// guards on its own `gid`, so the surplus threads of the last block, and the whole
    /// block launched for an empty MSM, return without touching memory. That is cheaper
    /// than special-casing `items == 0` at five call sites, and a `grid_dim` of zero is a
    /// launch error rather than a no-op.
    fn cfg_1d(&self, items: usize, prefer: u32) -> LaunchConfig {
        let block = prefer.min(self.max_block).max(1);
        let grid = (items as u64).div_ceil(block as u64).max(1) as u32;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// One block per group, for the three kernels that reduce inside a block. `threads` must
    /// not exceed the kernel's own `__shared__` array length; see [`REDUCE_TG`].
    fn cfg_blocks(&self, groups: usize, threads: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (groups.max(1) as u32, 1, 1),
            block_dim: (threads.min(self.max_block).max(1), 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

struct Kernels {
    zero_u32: Kernel,
    mont_to_std: Kernel,
    count: Kernel,
    scan: Kernel,
    scatter: Kernel,
    clear_g1: Kernel,
    clear_g2: Kernel,
    accumulate_g1: Kernel,
    accumulate_g2: Kernel,
    segmented_g1: Kernel,
    segmented_g2: Kernel,
    merge_g1: Kernel,
    merge_g2: Kernel,
    reduce_g1: Kernel,
    reduce_g2: Kernel,
    ones_g1: Kernel,
    ones_g2: Kernel,
}

/// Threads a block for the trivial kernels. `zero_u32` and the two bucket clears take six
/// registers each and are pure stores, so all that matters is having enough warps in flight
/// to saturate the memory system.
const WIDE_BLOCK: u32 = 256;

/// Threads a block for the point kernels.
///
/// Measured on a T4 at `-O3` by the kernel lane: `msm_count` 28 registers, `msm_scatter` 30,
/// the G1 point kernels 122 to 217, and every G2 one pinned at the 255 ceiling with a small
/// stack spill. At 255 registers an SM holds 8 warps out of 32 however the blocks are
/// shaped, so this number cannot buy occupancy back. 128 is picked because four warps a
/// block gives the scheduler more independent blocks to interleave across the tail of a
/// ragged dispatch than 256 would, at no cost. **Not swept on a real card**; it is the
/// second thing to try after `G16_CUDA_MSM_C`.
const POINT_BLOCK: u32 = 128;

// ---------------------------------------------------------------------------
// The backend object
// ---------------------------------------------------------------------------

/// Stages 5 to 9, with every base vector of one proving key resident on the device.
///
/// Built once, in `prepare`. That is not a style preference. NVRTC only emits PTX, and the
/// driver runs the same ptxas at `cuModuleLoad`, so the JIT cost lands inside this call:
/// the kernel lane measured about 4m50s on a T4, essentially all of it in the fully inlined
/// `Fq2` curve arithmetic (`msm_reduce_g2` 85.6 s, `msm_merge_g2` 24.8 s, `msm_ones_g2`
/// 16.6 s, against 4.7 ms for `msm_scan`). A prover that constructs this per proof is
/// measuring ptxas; one that constructs it inside a benchmark's timed region is measuring
/// nothing at all. The header of `kernels/msm.cu` records the two candidate fixes, both of
/// which are benchmarks rather than obvious wins.
pub struct CudaMsm {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Kept alive explicitly. `CudaFunction` holds an `Arc<CudaModule>` internally, so this
    /// field documents the ownership rather than being what keeps the module loaded.
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    k: Kernels,

    /// The five base vectors, stages 5 to 9 in order.
    a: G1Bases,
    b_g2: G2Bases,
    b_g1: G1Bases,
    l: G1Bases,
    h: G1Bases,

    n_vars: usize,
    n_public: usize,

    last_device_us: AtomicU64,
    last_combine_us: AtomicU64,
}

impl CudaMsm {
    /// Compiles `kernels::unit_msm()` and uploads all five repacked base vectors.
    pub fn new(cuda: &Cuda, pk: &ProvingKey) -> Result<Self, ProveError> {
        Self::without_key(cuda)?.with_key(pk)
    }

    /// The same object with no key attached: the kernels are compiled and uploads work, but
    /// the five resident vectors are empty, so [`Self::msms`] has nothing to run against.
    ///
    /// This is what a caller with its own base vectors uses, and what the tests below use so
    /// that the point arithmetic can be checked against a naive sum without a hundred
    /// megabytes of zkey. It still pays the full ptxas cost; see the type's docs.
    pub fn without_key(cuda: &Cuda) -> Result<Self, ProveError> {
        Self::from_module(cuda, Self::compile(cuda)?)
    }

    /// Compile the MSM translation unit on its own.
    ///
    /// Split out of the constructor so a process that prepares more than one key pays
    /// NVRTC and ptxas once rather than once per key. This is not a micro-optimisation:
    /// the type's own docs record minutes, not milliseconds, for the G2 curve arithmetic,
    /// and [`crate::backend::CudaBackend`] compiles here exactly once and hands the same
    /// module to every circuit it prepares.
    pub fn compile(cuda: &Cuda) -> Result<Arc<CudaModule>, ProveError> {
        cuda.compile("msm", &kernels::unit_msm())
            .map_err(|e| bad(e.to_string()))
    }

    /// Bind the seventeen kernel handles out of an already-compiled module. No bases yet.
    ///
    /// `module` must have come from [`Self::compile`] against this same [`Cuda`]. A module
    /// belongs to the context that loaded it, and a function pulled out of one context and
    /// launched on another context's stream is an invalid-handle failure at launch, not at
    /// bind, which is a long way from the mistake.
    pub fn from_module(cuda: &Cuda, module: Arc<CudaModule>) -> Result<Self, ProveError> {
        let k = Kernels {
            zero_u32: Kernel::load(&module, "zero_u32")?,
            mont_to_std: Kernel::load(&module, "fr_mont_to_std")?,
            count: Kernel::load(&module, "msm_count")?,
            scan: Kernel::load(&module, "msm_scan")?,
            scatter: Kernel::load(&module, "msm_scatter")?,
            clear_g1: Kernel::load(&module, "msm_clear_g1")?,
            clear_g2: Kernel::load(&module, "msm_clear_g2")?,
            accumulate_g1: Kernel::load(&module, "msm_accumulate_g1")?,
            accumulate_g2: Kernel::load(&module, "msm_accumulate_g2")?,
            segmented_g1: Kernel::load(&module, "msm_segmented_g1")?,
            segmented_g2: Kernel::load(&module, "msm_segmented_g2")?,
            merge_g1: Kernel::load(&module, "msm_merge_g1")?,
            merge_g2: Kernel::load(&module, "msm_merge_g2")?,
            reduce_g1: Kernel::load(&module, "msm_reduce_g1")?,
            reduce_g2: Kernel::load(&module, "msm_reduce_g2")?,
            ones_g1: Kernel::load(&module, "msm_ones_g1")?,
            ones_g2: Kernel::load(&module, "msm_ones_g2")?,
        };

        let ctx = cuda.context().clone();
        let stream = cuda.stream().clone();

        Ok(Self {
            a: upload_g1(&stream, &[])?,
            b_g2: upload_g2(&stream, &[])?,
            b_g1: upload_g1(&stream, &[])?,
            l: upload_g1(&stream, &[])?,
            h: upload_g1(&stream, &[])?,
            ctx,
            stream,
            module,
            k,
            n_vars: 0,
            n_public: 0,
            last_device_us: AtomicU64::new(0),
            last_combine_us: AtomicU64::new(0),
        })
    }

    /// Make one key resident. Consumes and returns `self` so a half-uploaded object is not
    /// reachable: an error here drops every vector that did make it across.
    ///
    /// Sections 5 to 9 of the zkey. Four `n_vars`-long G1 vectors at 64 bytes a point plus
    /// one G2 vector at 128, so tens to hundreds of megabytes across PCIe. Once, at prepare
    /// time, never per proof.
    pub fn with_key(mut self, pk: &ProvingKey) -> Result<Self, ProveError> {
        self.a = upload_g1(&self.stream, &pk.a_query)?;
        self.b_g2 = upload_g2(&self.stream, &pk.b_g2_query)?;
        self.b_g1 = upload_g1(&self.stream, &pk.b_g1_query)?;
        self.l = upload_g1(&self.stream, &pk.l_query)?;
        self.h = upload_g1(&self.stream, &pk.h_query)?;
        self.n_vars = pk.n_vars;
        self.n_public = pk.n_public;
        Ok(self)
    }

    /// The stream every launch here is issued on. The stages lane uses the same one, which
    /// is what orders stage 4's write of `H` before stage 9's read of it with no event.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Stage 5's bases, and so on. Public so a caller can build its own [`Job`] against a
    /// resident vector instead of re-uploading one.
    pub fn a_bases(&self) -> &G1Bases {
        &self.a
    }
    pub fn b_g2_bases(&self) -> &G2Bases {
        &self.b_g2
    }
    pub fn b_g1_bases(&self) -> &G1Bases {
        &self.b_g1
    }
    pub fn l_bases(&self) -> &G1Bases {
        &self.l
    }
    pub fn h_bases(&self) -> &G1Bases {
        &self.h
    }

    /// GPU time, from a CUDA event pair, spanning every launch of the last
    /// [`Self::msm_batch`]. Excludes the host scalar pack, the upload and the readback.
    pub fn last_device_us(&self) -> u64 {
        self.last_device_us.load(Ordering::Relaxed)
    }

    /// Wall clock of the last batch's readback plus its Horner tail.
    pub fn last_combine_us(&self) -> u64 {
        self.last_combine_us.load(Ordering::Relaxed)
    }

    // -----------------------------------------------------------------------
    // Uploads
    // -----------------------------------------------------------------------

    /// Repacks and uploads a G1 base vector. See [`G1Bases`] for why the repack is not
    /// optional.
    pub fn upload_g1_bases(&self, bases: &[G1Affine]) -> Result<G1Bases, ProveError> {
        upload_g1(&self.stream, bases)
    }

    pub fn upload_g2_bases(&self, bases: &[G2Affine]) -> Result<G2Bases, ProveError> {
        upload_g2(&self.stream, bases)
    }

    /// Packs scalars into standard form and uploads them, classifying as it goes.
    ///
    /// One pass over the witness, and the classification it produces is what keeps the zero
    /// and one scalars out of Pippenger entirely.
    pub fn upload_scalars(&self, scalars: &[Fr]) -> Result<ScalarBuf, ProveError> {
        let mut packed = Vec::with_capacity(scalars.len());
        let mut prefix = Vec::with_capacity(scalars.len() + 1);
        let mut general = 0u32;
        prefix.push(0);
        for s in scalars {
            if !(s.is_zero() || s.is_one()) {
                general += 1;
            }
            prefix.push(general);
            packed.push(PackedScalar::from_fr(s));
        }
        Ok(ScalarBuf {
            buf: upload_words(&self.stream, as_words(&packed))?,
            len: scalars.len(),
            general_prefix: Some(prefix),
        })
    }

    /// Converts a device-resident Montgomery `Fr` buffer into standard-form scalars.
    ///
    /// The stages lane already produces both forms, so the proving path should read
    /// `HHandle::h_std` through [`Scalars::device_std`] rather than call this. It exists for
    /// a caller that only has the Montgomery buffer, and to make the asymmetry impossible to
    /// miss: feeding Montgomery limbs to the window decomposition yields a proof wrong by a
    /// factor of R, which fails verification with no other symptom.
    pub fn scalars_from_device_mont(
        &self,
        mont: &CudaSlice<u32>,
        len: usize,
    ) -> Result<ScalarBuf, ProveError> {
        if mont.len() < len * SCALAR_WORDS {
            return Err(bad(format!(
                "Montgomery buffer holds {} words; {len} elements need {}",
                mont.len(),
                len * SCALAR_WORDS
            )));
        }
        let out = self.zeros(len.max(1) * SCALAR_WORDS)?;
        let n = len as u32;
        let cfg = self.k.mont_to_std.cfg_1d(len, WIDE_BLOCK);
        let mut lb = self.stream.launch_builder(&self.k.mont_to_std.f);
        lb.arg(mont).arg(&out).arg(&n);
        // SAFETY: three parameters bound in order and with matching types; `mont` was
        // checked above to hold at least `len` elements and `out` was allocated with exactly
        // that many; the kernel's `gid < len` guard covers the tail block.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch fr_mont_to_std", e))?;
        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize fr_mont_to_std", e))?;
        Ok(ScalarBuf {
            buf: out,
            len,
            general_prefix: None,
        })
    }

    /// Every allocation in this module goes through here.
    ///
    /// `alloc_zeros` rather than `alloc`, unconditionally, including for buffers a kernel
    /// fully overwrites. `cudaMalloc` hands back the previous tenant's bytes, an
    /// uninitialised limb array is a plausible looking field element, and the resulting
    /// failure is a proof that does not verify rather than a fault. A stream-ordered
    /// `cuMemsetD8Async` over the block is the cheapest possible way to make that class of
    /// bug impossible. The `max(1)` is because `cuMemAlloc` of zero bytes is documented to
    /// fail with `CUDA_ERROR_INVALID_VALUE`, and an empty MSM reaches here.
    fn zeros(&self, words: usize) -> Result<CudaSlice<u32>, ProveError> {
        self.stream
            .alloc_zeros::<u32>(words.max(1))
            .map_err(|e| drv("allocate MSM scratch", e))
    }

    // -----------------------------------------------------------------------
    // Running MSMs
    // -----------------------------------------------------------------------

    /// One MSM. Convenience wrapper over [`Self::msm_batch`]; the proving path should use
    /// the batch, which shares one digit pipeline across the three MSMs that read the whole
    /// witness.
    pub fn msm(&self, job: Job<'_>) -> Result<MsmResult, ProveError> {
        let out = self.msm_batch(std::slice::from_ref(&job))?;
        out.into_iter()
            .next()
            .ok_or_else(|| bad("msm_batch returned no result for one job"))
    }

    /// Every MSM in one stream, one synchronize, one readback.
    ///
    /// Jobs that share a scalar buffer, offset and length share their digit pipeline: the A,
    /// B-in-G2 and B-in-G1 MSMs all run over the whole witness, so the counting sort runs
    /// once for the three of them and only the point stages repeat. That removes six of the
    /// fifteen digit launches and, more to the point, two thirds of the scatter's memory
    /// traffic.
    pub fn msm_batch(&self, jobs: &[Job<'_>]) -> Result<Vec<MsmResult>, ProveError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        // The start event is recorded before anything is allocated, because `zeros()`
        // queues a `cuMemsetD8Async` over every buffer it hands out and that memset is real
        // per-proof device work (tens of megabytes of bucket array at 2^18). Leaving it
        // outside the span would make the MSM look faster than it is.
        //
        // The events are created per call rather than kept on `self`. Two concurrent proofs
        // sharing one `CudaMsm` would otherwise record into the same pair and read each
        // other's numbers; `cuEventCreate` is far cheaper than a single MSM, so the
        // allocation is not worth pooling at this granularity. `CU_EVENT_DEFAULT`, not
        // `new_event(None)`, whose default is `CU_EVENT_DISABLE_TIMING`: such an event
        // records fine and then fails at `elapsed_ms`, turning a measurement into an error
        // at the end of an otherwise correct proof.
        let ev = |what: &str| {
            self.ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| drv(what, e))
        };
        let start = ev("create MSM start event")?;
        let end = ev("create MSM end event")?;

        start
            .record(&self.stream)
            .map_err(|e| drv("record MSM start event", e))?;

        // ---- plan ----
        let mut plans: Vec<Plan<'_>> = Vec::new();
        let mut by_key: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut job_plan = Vec::with_capacity(jobs.len());

        for job in jobs {
            let (scalars, soff, n, bases_len, boff) = match job {
                Job::G1(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
                Job::G2(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
            };
            if soff + n > scalars.len {
                return Err(bad(format!(
                    "scalar range {}..{} exceeds the {} scalars available",
                    soff,
                    soff + n,
                    scalars.len
                )));
            }
            if boff + n > bases_len {
                return Err(bad(format!(
                    "base range {}..{} exceeds the {bases_len} bases uploaded",
                    boff,
                    boff + n
                )));
            }
            // Identity of the device buffer, not of its contents: two jobs share a digit
            // pipeline exactly when they read the same scalars over the same range.
            let key = (scalars.buf as *const CudaSlice<u32> as usize, soff, n);
            let idx = match by_key.get(&key) {
                Some(&i) => i,
                None => {
                    let i = plans.len();
                    plans.push(Plan::new(self, scalars, soff, n)?);
                    by_key.insert(key, i);
                    i
                }
            };
            job_plan.push(idx);
        }

        let mut outs: Vec<Outputs> = Vec::with_capacity(jobs.len());
        for (i, job) in jobs.iter().enumerate() {
            outs.push(Outputs::new(self, job, &plans[job_plan[i]])?);
        }

        // ---- launch ----
        for p in &plans {
            self.launch_digits(p)?;
        }
        for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
            self.launch_points(job, out, &plans[job_plan[i]])?;
        }
        end.record(&self.stream)
            .map_err(|e| drv("record MSM end event", e))?;

        // The one wait. Everything above was queued without the host blocking.
        self.stream
            .synchronize()
            .map_err(|e| drv("synchronize stages 5-9", e))?;
        self.last_device_us
            .store(elapsed_us(&start, &end)?, Ordering::Relaxed);

        // ---- combine ----
        let t0 = Instant::now();
        let mut results = Vec::with_capacity(jobs.len());
        for (i, out) in outs.iter().enumerate() {
            results.push(out.combine(self, &plans[job_plan[i]])?);
        }
        self.last_combine_us
            .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        Ok(results)
    }

    /// Stages 5 to 9 for one proof, in the order [`MsmOutputs`] declares them.
    ///
    /// `h` must be what this circuit's `compute_h` returned. A [`HPoly::Device`] tagged for
    /// this backend is read in place, which is the whole reason the stage boundary carries a
    /// handle instead of a `Vec<Fr>`: `H` is `domain_size` scalars, up to 8 MB at 2^18, and
    /// pulling it down only to push it back up is two PCIe crossings for nothing. A handle
    /// tagged for a different backend is an error rather than a panic, because the caller
    /// mixing two backends is a recoverable mistake and dereferencing another backend's
    /// pointer is not.
    pub fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }

        let t0 = Instant::now();
        // The witness the gather already uploaded is reused when `h` is our own device
        // handle and can vouch (length + limb fold) that it was computed from this very
        // witness; then the standard-form copy `compute_h` converted on device serves
        // stages 5-8 with no second PCIe transfer and no host-side re-pack. Any doubt
        // falls back to the plain upload. `G16_CUDA_WITNESS_REUSE=0` forces the fallback.
        // `w_owned` exists only to give the fallback upload an owner that outlives the
        // batch; the reuse arm borrows straight from the handle inside `h`.
        let reused = h
            .device_handle::<HHandle>(TAG)
            .and_then(|hd| hd.witness_std_for(witness));
        let w_owned: Option<ScalarBuf> = if reused.is_none() {
            Some(self.upload_scalars(witness)?)
        } else {
            None
        };
        let w = match reused {
            Some((buf, prefix)) => Scalars::device_std_with_prefix(buf, witness.len(), prefix)?,
            None => w_owned.as_ref().expect("uploaded above").as_scalars(),
        };
        // Held only so the borrow in `h_scalars` outlives the batch.
        let h_owned: Option<ScalarBuf>;
        let h_scalars = match h {
            HPoly::Host(v) => {
                h_owned = Some(self.upload_scalars(v)?);
                h_owned.as_ref().expect("just assigned Some").as_scalars()
            }
            HPoly::Device { tag, len, .. } => {
                if *tag != TAG {
                    return Err(bad(format!(
                        "H is a device handle tagged \"{tag}\", which this backend cannot \
                         read; it came from another backend's compute_h"
                    )));
                }
                let handle = h
                    .device_handle::<HHandle>(TAG)
                    .ok_or_else(|| bad("H carries the \"cuda\" tag but not a stages::HHandle"))?;
                // `h_std`, never `h_mont`. A window digit of a Montgomery representative is
                // a digit of `a*R mod r`, a different number, and the proof would be wrong
                // by a factor of R with nothing else to go on.
                Scalars::device_std(handle.h_std(), *len)?
            }
        };
        let upload_us = t0.elapsed().as_micros() as u64;

        let n = witness.len();
        let l_off = self.n_public + 1;
        let l_n = self.l.len();
        // Not clamped to `h_query.len()`. A domain size that disagrees with section 9 is a
        // key or a stage-4 bug, and `msm_batch`'s bounds check names both lengths; silently
        // truncating would drop the tail of `H` and produce a proof that only fails at
        // verification.
        let h_n = h_scalars.len();
        let jobs = vec![
            Job::G1(JobG1 {
                bases: &self.a,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n,
            }),
            Job::G2(JobG2 {
                bases: &self.b_g2,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n,
            }),
            Job::G1(JobG1 {
                bases: &self.b_g1,
                base_off: 0,
                scalars: w,
                scalar_off: 0,
                n,
            }),
            Job::G1(JobG1 {
                bases: &self.l,
                base_off: 0,
                scalars: w,
                scalar_off: l_off,
                n: l_n,
            }),
            Job::G1(JobG1 {
                bases: &self.h,
                base_off: 0,
                scalars: h_scalars,
                scalar_off: 0,
                n: h_n,
            }),
        ];
        let out = self.msm_batch(&jobs)?;

        // Host launch time is not added: the launches are asynchronous, so it overlaps the
        // device work the event pair already counted. What is added is the two things that
        // genuinely happen with the device idle.
        t.msm_us += upload_us + self.last_device_us() + self.last_combine_us();

        Ok(MsmOutputs {
            a_g1: out[0].g1()?,
            b_g2: out[1].g2()?,
            b_g1: out[2].g1()?,
            l_g1: out[3].g1()?,
            h_g1: out[4].g1()?,
        })
    }

    // -----------------------------------------------------------------------
    // Launches
    // -----------------------------------------------------------------------

    /// Everything that depends only on the scalars: clear the histogram, count, scan,
    /// scatter. Shared by every job that reads the same scalar range.
    ///
    /// Buffers the kernels *write* are bound as shared references, not `&mut`. `cudarc`'s
    /// `&mut CudaSlice` argument form exists only to record read/write events, and those are
    /// recorded only when the context is in multi-stream mode; everything here is issued in
    /// order on one stream, which is what actually orders it. The Metal encoder binds the
    /// same way (`set_buffer` takes `&Buffer` for both directions), and taking `&mut` would
    /// make the plan sharing inexpressible, since three jobs read one plan's `cursor` while
    /// their own kernels write their own buckets.
    fn launch_digits(&self, p: &Plan<'_>) -> Result<(), ProveError> {
        let params = p.params();
        let rows = p.n_windows * p.n_buckets;

        // `msm_count` accumulates onto the histogram with `atomicAdd`, so it must start at
        // zero. `zeros()` already memset the allocation; this is the explicit step the
        // Metal schedule has, kept so that the requirement is visible at the call site
        // rather than buried in an allocator's behaviour.
        let len = rows as u32;
        let cfg = self.k.zero_u32.cfg_1d(rows, WIDE_BLOCK);
        let mut lb = self.stream.launch_builder(&self.k.zero_u32.f);
        lb.arg(&p.counts).arg(&len);
        // SAFETY: two parameters in order; `counts` was allocated with exactly `rows` words
        // and the kernel guards `gid < len`.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch zero_u32", e))?;

        let cfg = self.k.count.cfg_1d(p.n, POINT_BLOCK);
        let mut lb = self.stream.launch_builder(&self.k.count.f);
        lb.arg(p.scalars).arg(&p.counts).arg(&params);
        // SAFETY: three parameters in order; the scalar range was bounds checked in
        // `msm_batch`, every row the kernel can index is `w * n_buckets + (mag - 1)` with
        // `mag <= n_buckets`, and `counts` holds `n_windows * n_buckets` words.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_count", e))?;

        // One block per window, and at most `SCAN_TG` threads because that is the length of
        // the kernel's shared array.
        let cfg = self.k.scan.cfg_blocks(p.n_windows, SCAN_TG);
        let mut lb = self.stream.launch_builder(&self.k.scan.f);
        lb.arg(&p.counts).arg(&p.cursor).arg(&params);
        // SAFETY: three parameters in order; the grid is exactly `n_windows` blocks, which
        // is what the kernel's `blockIdx.x` indexes, and both arrays hold
        // `n_windows * n_buckets` words.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_scan", e))?;

        let cfg = self.k.scatter.cfg_1d(p.n, POINT_BLOCK);
        let mut lb = self.stream.launch_builder(&self.k.scatter.f);
        lb.arg(p.scalars)
            .arg(&p.cursor)
            .arg(&p.entries)
            .arg(&params);
        // SAFETY: four parameters in order. `entries` holds `n_windows * cap` records and
        // `cap` is an exact upper bound on the entries one window can emit: only general
        // scalars emit at all, and each emits at most one per window.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_scatter", e))?;

        Ok(())
    }

    /// The point stages of one job: clear, accumulate, merge, reduce, ones.
    fn launch_points(&self, job: &Job<'_>, out: &Outputs, p: &Plan<'_>) -> Result<(), ProveError> {
        let mut params = p.params();
        params.base_off = out.base_off as u32;
        params.ones_groups = out.ones_groups as u32;

        let (clear, acc, seg, merge, reduce, ones, bases) = match job {
            Job::G1(j) => (
                &self.k.clear_g1,
                &self.k.accumulate_g1,
                &self.k.segmented_g1,
                &self.k.merge_g1,
                &self.k.reduce_g1,
                &self.k.ones_g1,
                &j.bases.buf,
            ),
            Job::G2(j) => (
                &self.k.clear_g2,
                &self.k.accumulate_g2,
                &self.k.segmented_g2,
                &self.k.merge_g2,
                &self.k.reduce_g2,
                &self.k.ones_g2,
                &j.bases.buf,
            ),
        };

        let rows = p.n_windows * p.n_buckets;
        if legacy_accumulate() {
            let cfg = acc.cfg_1d(rows, POINT_BLOCK);
            let mut lb = self.stream.launch_builder(&acc.f);
            lb.arg(&p.entries)
                .arg(bases)
                .arg(&p.counts)
                .arg(&p.cursor)
                .arg(&out.buckets)
                .arg(&params);
            // SAFETY: six parameters in order; one thread owns one bucket row exclusively,
            // and the kernel guards `row < n_windows * n_buckets`.
            unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_accumulate", e))?;
        } else {
            // A bucket that no slice writes directly still has to read as the identity,
            // which on this encoding means `zz == 0`. `zeros()` already guarantees that for
            // a fresh allocation; this is the algorithm's own step and the thing that would
            // still be correct if the allocation were ever recycled.
            let cfg = clear.cfg_1d(rows, WIDE_BLOCK);
            let mut lb = self.stream.launch_builder(&clear.f);
            lb.arg(&out.buckets).arg(&params);
            // SAFETY: two parameters in order; `buckets` holds `rows` points and the kernel
            // guards `gid < rows`.
            unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_clear", e))?;

            // One thread per slice, `gid = w * slices + k`. The surplus threads of the last
            // block self-guard on `w >= n_windows`.
            let threads = p.n_windows * p.slices;
            let cfg = seg.cfg_1d(threads, POINT_BLOCK);
            let mut lb = self.stream.launch_builder(&seg.f);
            lb.arg(&p.entries)
                .arg(bases)
                .arg(&p.cursor)
                .arg(&out.buckets)
                .arg(&out.spill_pts)
                .arg(&out.spill_rows)
                .arg(&params);
            // SAFETY: seven parameters in order; the spill arrays hold two slots per slice,
            // which is the maximum one thread writes (at most one run continues backwards
            // and at most one forwards), and both are fully written before any early return
            // that could leave the merge reading them.
            unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_segmented", e))?;

            let cfg = merge.cfg_1d(rows, POINT_BLOCK);
            let mut lb = self.stream.launch_builder(&merge.f);
            lb.arg(&out.buckets)
                .arg(&out.spill_pts)
                .arg(&out.spill_rows)
                .arg(&p.counts)
                .arg(&p.cursor)
                .arg(&params);
            // SAFETY: six parameters in order; one thread owns one bucket row and only reads
            // the spill slots of the slices its own run overlaps, which are inside the
            // `2 * n_windows * slices` allocated.
            unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_merge", e))?;
        }

        // One block per window, at most `REDUCE_TG` threads. See `REDUCE_TG`: a wider block
        // indexes past the end of the kernel's shared array with no diagnostic at all.
        let cfg = reduce.cfg_blocks(p.n_windows, REDUCE_TG);
        let mut lb = self.stream.launch_builder(&reduce.f);
        lb.arg(&out.buckets).arg(&out.window_sums).arg(&params);
        // SAFETY: three parameters in order; the grid is exactly `n_windows` blocks, which
        // is what `blockIdx.x` indexes into `window_sums`.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_reduce", e))?;

        // Not optional. `msm_count` and `msm_scatter` deliberately drop every scalar equal
        // to 1, so without this kernel those terms are simply missing from the proof.
        let cfg = ones.cfg_blocks(out.ones_groups, REDUCE_TG);
        let mut lb = self.stream.launch_builder(&ones.f);
        lb.arg(p.scalars).arg(bases).arg(&out.ones).arg(&params);
        // SAFETY: four parameters in order; the grid is exactly `ones_groups` blocks, which
        // is what `blockIdx.x` indexes into `ones`, and the grid-stride loop is bounded by
        // `p.n`, itself bounds checked against both the scalar and the base vector.
        unsafe { lb.launch(cfg) }.map_err(|e| drv("launch msm_ones", e))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// A digit pipeline: everything that depends only on the scalars.
// ---------------------------------------------------------------------------

struct Plan<'a> {
    n: usize,
    scalar_off: usize,
    c: u32,
    n_windows: usize,
    n_buckets: usize,
    cap: usize,
    slice_len: usize,
    slices: usize,
    scalars: &'a CudaSlice<u32>,
    counts: CudaSlice<u32>,
    cursor: CudaSlice<u32>,
    entries: CudaSlice<u32>,
}

impl<'a> Plan<'a> {
    fn new(
        msm: &CudaMsm,
        scalars: Scalars<'a>,
        scalar_off: usize,
        n: usize,
    ) -> Result<Self, ProveError> {
        let range = scalar_off..scalar_off + n;
        let general = scalars.general_in(&range);
        let c = window_size(general);
        let n_windows = RECODE_BITS.div_ceil(c as usize);
        let n_buckets = 1usize << (c - 1);
        // Only general scalars emit entries, and each emits at most one per window, so this
        // bounds the scatter exactly rather than conservatively.
        let cap = general.max(1);
        let slice_len = slice_len();
        let rows = n_windows * n_buckets;
        Ok(Self {
            n,
            scalar_off,
            c,
            n_windows,
            n_buckets,
            cap,
            slice_len,
            slices: cap.div_ceil(slice_len).max(1),
            scalars: scalars.buf,
            counts: msm.zeros(rows)?,
            cursor: msm.zeros(rows)?,
            // 8 bytes an entry: the bucket row travels with the point index so the segmented
            // accumulation can find run boundaries without recomputing digits.
            entries: msm.zeros(n_windows * cap * ENTRY_WORDS)?,
        })
    }

    fn params(&self) -> MsmParams {
        MsmParams {
            n: self.n as u32,
            c: self.c,
            n_windows: self.n_windows as u32,
            n_buckets: self.n_buckets as u32,
            cap: self.cap as u32,
            scalar_off: self.scalar_off as u32,
            base_off: 0,
            ones_groups: 0,
            slice_len: self.slice_len as u32,
            slices: self.slices as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// A single MSM's point stages and its readback.
// ---------------------------------------------------------------------------

struct Outputs {
    buckets: CudaSlice<u32>,
    spill_pts: CudaSlice<u32>,
    spill_rows: CudaSlice<u32>,
    window_sums: CudaSlice<u32>,
    ones: CudaSlice<u32>,
    ones_groups: usize,
    base_off: usize,
    is_g2: bool,
}

impl Outputs {
    fn new(msm: &CudaMsm, job: &Job<'_>, plan: &Plan<'_>) -> Result<Self, ProveError> {
        let (is_g2, base_off, point_words) = match job {
            Job::G1(j) => (false, j.base_off, G1_WORDS),
            Job::G2(j) => (true, j.base_off, G2_WORDS),
        };
        let ones_groups = ones_groups_for(plan.n);
        let rows = plan.n_windows * plan.n_buckets;
        // Two spill slots per slice: at most one run of a slice continues backwards and at
        // most one continues forwards.
        let spill_slots = 2 * plan.n_windows * plan.slices;
        Ok(Self {
            buckets: msm.zeros(rows * point_words)?,
            spill_pts: msm.zeros(spill_slots * point_words)?,
            spill_rows: msm.zeros(spill_slots)?,
            window_sums: msm.zeros(plan.n_windows * point_words)?,
            ones: msm.zeros(ones_groups * point_words)?,
            ones_groups,
            base_off,
            is_g2,
        })
    }

    /// Read back `n_windows + ones_groups` points and finish on the host.
    ///
    /// Horner over the windows, high to low, `c` doublings between each. Same order as the
    /// CPU backend, so the two agree bit for bit and not just up to the group law. A few
    /// hundred curve additions per proof, against tens of millions on the device: moving
    /// this to the GPU would cost more in launches than it saves, which is the same place
    /// the Metal twin and zkonduit's Metal MSM both stop.
    fn combine(&self, msm: &CudaMsm, plan: &Plan<'_>) -> Result<MsmResult, ProveError> {
        let w_words = download(&msm.stream, &self.window_sums)?;
        let o_words = download(&msm.stream, &self.ones)?;
        if self.is_g2 {
            let w = from_words::<PackedXyzzG2>(&w_words)
                .ok_or_else(|| bad("G2 window sums are not a whole number of points"))?;
            let o = from_words::<PackedXyzzG2>(&o_words)
                .ok_or_else(|| bad("G2 ones partials are not a whole number of points"))?;
            let mut acc = w[plan.n_windows - 1].to_projective();
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += w[k].to_projective();
            }
            for x in &o {
                acc += x.to_projective();
            }
            Ok(MsmResult::G2(acc))
        } else {
            let w = from_words::<PackedXyzzG1>(&w_words)
                .ok_or_else(|| bad("G1 window sums are not a whole number of points"))?;
            let o = from_words::<PackedXyzzG1>(&o_words)
                .ok_or_else(|| bad("G1 ones partials are not a whole number of points"))?;
            let mut acc = w[plan.n_windows - 1].to_projective();
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += w[k].to_projective();
            }
            for x in &o {
                acc += x.to_projective();
            }
            Ok(MsmResult::G1(acc))
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer helpers
// ---------------------------------------------------------------------------

/// Uploads `data`, padding an empty slice to one word, and waits.
///
/// Two things to know. `cuMemAlloc` of zero bytes is documented to fail with
/// `CUDA_ERROR_INVALID_VALUE`, and real inputs reach here empty: `l_query` is empty for a
/// circuit whose every signal is public. And the copy is issued from pageable host memory,
/// which `cudarc` does not synchronize after, so the wait has to happen before the source
/// `Vec` is dropped at the end of the caller's statement. That is a real per-proof cost for
/// the witness upload and the honest place to pay it; making it asynchronous means staging
/// through page-locked memory, which is what `stages::CudaStages` does for the witness and
/// what this file would copy if the upload ever shows up in a profile.
fn upload_words(stream: &Arc<CudaStream>, data: &[u32]) -> Result<CudaSlice<u32>, ProveError> {
    let src: &[u32] = if data.is_empty() { &[0u32] } else { data };
    let buf = stream
        .clone_htod(src)
        .map_err(|e| drv("upload to device", e))?;
    stream
        .synchronize()
        .map_err(|e| drv("synchronize upload", e))?;
    Ok(buf)
}

fn upload_g1(stream: &Arc<CudaStream>, bases: &[G1Affine]) -> Result<G1Bases, ProveError> {
    let packed = PackedG1Affine::pack_slice(bases);
    Ok(G1Bases {
        buf: upload_words(stream, as_words(&packed))?,
        len: bases.len(),
    })
}

fn upload_g2(stream: &Arc<CudaStream>, bases: &[G2Affine]) -> Result<G2Bases, ProveError> {
    let packed = PackedG2Affine::pack_slice(bases);
    Ok(G2Bases {
        buf: upload_words(stream, as_words(&packed))?,
        len: bases.len(),
    })
}

/// Device to host, then wait. The wait is not optional: `cudarc` issues the copy into a
/// plain `Vec` asynchronously and does not synchronize afterwards, so reading the `Vec`
/// without this returns whatever was in the uninitialised allocation.
fn download(stream: &Arc<CudaStream>, buf: &CudaSlice<u32>) -> Result<Vec<u32>, ProveError> {
    let v = stream
        .clone_dtoh(buf)
        .map_err(|e| drv("download MSM points", e))?;
    stream
        .synchronize()
        .map_err(|e| drv("synchronize MSM readback", e))?;
    Ok(v)
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
// Thread safety
// ---------------------------------------------------------------------------

/// `PreparedCircuit` must be `Send + Sync`, so anything a CUDA backend stores on one has to
/// be too. Checked rather than assumed. `cudarc` declares `CudaContext`, `CudaStream`,
/// `CudaSlice` and `CudaEvent` `Send + Sync`, and the driver API is documented thread safe;
/// what actually makes two concurrent proofs correct here is that a stream executes its
/// launches in issue order and that the two proofs own disjoint allocations.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaMsm>();
    assert_send_sync::<G1Bases>();
    assert_send_sync::<G2Bases>();
    assert_send_sync::<ScalarBuf>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use g16_gpu_layout::{FQ_MODULUS, FQ_N0};

    /// The drift guard for the two block sizes, which is the failure this file can cause
    /// most easily and diagnose least easily.
    ///
    /// `REDUCE_TG` and `SCAN_TG` are the lengths of the `__shared__` scratch arrays in
    /// `msm_reduce_*`, `msm_ones_*` and `msm_scan`, and the host launches that many threads
    /// a block. Raising either on the Rust side alone makes every thread above the kernel's
    /// array length index out of bounds in shared memory: no compile error, no launch error,
    /// just wrong points. Measured on the Metal twin rather than assumed: setting the host
    /// constant to 128 with the shader left at 64 leaves `H` bit-exact and turns all five
    /// MSM outputs wrong, so the proof stops verifying with nothing to point at.
    #[test]
    fn the_kernel_declares_the_same_block_sizes() {
        for (name, want) in [("REDUCE_TG", REDUCE_TG), ("SCAN_TG", SCAN_TG)] {
            let line = format!("#define {name} {want}");
            assert!(
                kernels::MSM_CU.contains(&line),
                "kernels/msm.cu does not contain the line:\n{line}\n\
                 The host launches {want} threads a block and the kernel sizes its shared \
                 array from its own #define; if they disagree the reduction writes out of \
                 bounds and the MSM answers are silently wrong."
            );
        }
    }

    /// The `Fq` twin of `crate::tests::cuda_declares_the_same_constants`: the base field
    /// modulus and `N0` live in two languages and must be edited together. The kernel's own
    /// header says a Rust test greps for these exact lines; this is that test.
    #[test]
    fn the_kernel_declares_the_same_fq_constants() {
        let want_n = format!(
            "__constant__ u32 FQ_N[8] = {{ {} }};",
            FQ_MODULUS
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            kernels::MSM_CU.contains(&want_n),
            "kernels/msm.cu does not contain the line:\n{want_n}"
        );
        let want_n0 = format!("__constant__ u32 FQ_N0 = 0x{FQ_N0:08x}u;");
        assert!(
            kernels::MSM_CU.contains(&want_n0),
            "kernels/msm.cu does not contain the line:\n{want_n0}"
        );
    }

    /// Every entry point this host loads must be present in the source, and `extern "C"`.
    /// Without the `extern "C"` NVRTC mangles the name and `load_function` fails at run
    /// time, on a GPU box, minutes into a ptxas run.
    #[test]
    fn every_kernel_this_host_loads_is_declared_extern_c() {
        for name in [
            "zero_u32",
            "fr_mont_to_std",
            "msm_count",
            "msm_scan",
            "msm_scatter",
            "msm_clear_g1",
            "msm_clear_g2",
            "msm_accumulate_g1",
            "msm_accumulate_g2",
            "msm_segmented_g1",
            "msm_segmented_g2",
            "msm_merge_g1",
            "msm_merge_g2",
            "msm_reduce_g1",
            "msm_reduce_g2",
            "msm_ones_g1",
            "msm_ones_g2",
        ] {
            let want = format!("extern \"C\" __global__ void {name}(");
            assert!(
                kernels::MSM_CU.contains(&want),
                "kernels/msm.cu has no entry point declared as:\n{want}"
            );
        }
    }

    /// The recoding uses `2^(c-1)` buckets, not `2^(c-1) + 1`. That extra bucket is a real
    /// bug already found and fixed once in the CPU code: never written, but still walked by
    /// the running-sum reduction, which shifts every coefficient by one.
    #[test]
    fn every_window_width_keeps_its_digits_inside_its_buckets() {
        for c in 3..=MAX_WINDOW {
            let w = RECODE_BITS.div_ceil(c as usize);
            assert!(
                (w - 1) * (c as usize) < RECODE_BITS,
                "c = {c} wastes a window"
            );
            assert!(w * c as usize >= RECODE_BITS, "c = {c} loses the top bits");
            assert!(top_bits(c) >= 1 && top_bits(c) <= c, "c = {c}");
        }
    }

    /// The window sizes this ships with, written down so a change to [`window_size`] shows
    /// up as a diff here rather than as a silent 2x in a benchmark nobody reran.
    ///
    /// The values themselves are the unmeasured heuristic described on [`window_size`], so
    /// this test pins behaviour, not optimality.
    #[test]
    fn the_default_window_sizes_are_what_we_think_they_are() {
        // The degenerate top windows the correction is there to avoid.
        assert_eq!(top_bits(12), 3);
        assert_eq!(top_bits(14), 3);
        assert_eq!(top_bits(13), 8);
        assert_eq!(top_bits(15), 15);

        // A 140k witness with about 140k general scalars: the heuristic says 14, whose top
        // window holds four live buckets, so the correction steps down to 13.
        assert_eq!(window_size(140_261), 13);
        // A 2^18 H vector, dense by construction.
        assert_eq!(window_size(1 << 18), 15);
        // Small inputs clamp rather than underflow.
        assert_eq!(window_size(0), 3);
        assert_eq!(window_size(1), 3);
        assert_eq!(window_size(64), 3);
        // And nothing ever exceeds the bucket-array cap.
        for m in [1usize, 7, 1000, 1 << 20, 1 << 30] {
            let c = window_size(m);
            assert!((3..=MAX_WINDOW).contains(&c), "m = {m} gave c = {c}");
        }
    }

    /// `slices` and `cap` must bound what the kernels index, for every window width. The
    /// segmented accumulation launches `n_windows * slices` threads and each owns
    /// `slice_len` entries, so `slices * slice_len >= cap` or the tail of a window's entry
    /// run is never accumulated and the proof is quietly short a few points.
    #[test]
    fn the_slice_plan_covers_every_entry() {
        for general in [0usize, 1, 63, 64, 65, 4095, 140_261, 1 << 18] {
            let cap = general.max(1);
            let slices = cap.div_ceil(SLICE_LEN).max(1);
            assert!(
                slices * SLICE_LEN >= cap,
                "general = {general} leaves {} entries unaccumulated",
                cap - slices * SLICE_LEN
            );
        }
    }

    /// `ones_groups` is what both the kernel's grid and the host's tail sum are sized from.
    #[test]
    fn ones_groups_stay_in_range() {
        for n in [0usize, 1, 1000, 1 << 18, 1 << 24] {
            let g = ones_groups_for(n);
            assert!((1..=64).contains(&g), "n = {n} gave {g} groups");
        }
    }

    // -----------------------------------------------------------------------
    // On-device checks. These skip, loudly, when there is no NVIDIA card, because the rest
    // of this workspace is developed on an M2 Max; a silent pass is not something a
    // correctness suite for a proof system should be able to do.
    //
    // They are `#[ignore]` as well as guarded, because `Cuda::new` does not merely return an
    // error on a machine with no `libcuda`: `cudarc`'s dynamic loader panics inside it, so
    // the guard below cannot catch that case and a plain `cargo test` on the development Mac
    // would go red for a reason that has nothing to do with this code.
    //
    // Each of these compiles the MSM unit, which is minutes of ptxas on a T4. Run them on
    // the GPU box with
    // `cargo test -p g16-cuda --features cuda --lib msm:: -- --ignored --test-threads=1 --nocapture`
    // and expect the first one to sit there for a while.
    // -----------------------------------------------------------------------

    fn device() -> Option<Cuda> {
        match Cuda::new(0) {
            Ok(c) => {
                let (major, minor) = c.compute_capability();
                eprintln!(
                    "cuda device: {} (sm_{major}{minor}, {} SMs)",
                    c.device_name(),
                    c.sm_count()
                );
                Some(c)
            }
            Err(e) => {
                eprintln!("SKIPPING cuda MSM test, no usable device: {e}");
                None
            }
        }
    }

    fn splitmix(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A handful of scalars against a naive sum, before anything the size of a real circuit.
    ///
    /// This is the test that isolates the point arithmetic and the launch geometry: if the
    /// digit recoding, `pt_madd`, the block sizes or the zeroing are wrong, it fails here
    /// rather than 140,000 points later inside a proof that merely does not verify. The
    /// scalars are deliberately dense in 0 and 1, which is what a witness looks like and
    /// what the kernels take a different path for.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn small_g1_msm_matches_a_naive_sum() {
        use g16_field::{CurveGroup, PrimeField, PrimeGroup};
        let Some(cuda) = device() else { return };
        let m = CudaMsm::without_key(&cuda).expect("compile MSM unit");
        let mut seed = 0x1234_5678_9abc_def0u64;
        for n in [1usize, 2, 3, 31, 32, 33, 257, 1000] {
            let mut bases = Vec::with_capacity(n);
            let mut scalars = Vec::with_capacity(n);
            let mut cur = G1Projective::generator();
            for i in 0..n {
                cur += G1Projective::generator();
                bases.push(cur.into_affine());
                scalars.push(match i % 4 {
                    0 => Fr::zero(),
                    1 => Fr::one(),
                    _ => {
                        let mut b = [0u8; 32];
                        for c in b.chunks_mut(8) {
                            c.copy_from_slice(&splitmix(&mut seed).to_le_bytes());
                        }
                        Fr::from_le_bytes_mod_order(&b)
                    }
                });
            }
            let want = bases
                .iter()
                .zip(&scalars)
                .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
            let db = m.upload_g1_bases(&bases).expect("upload bases");
            let ds = m.upload_scalars(&scalars).expect("upload scalars");
            let got = m
                .msm(Job::G1(JobG1 {
                    bases: &db,
                    base_off: 0,
                    scalars: ds.as_scalars(),
                    scalar_off: 0,
                    n,
                }))
                .expect("msm")
                .g1()
                .expect("G1 result");
            assert_eq!(got.into_affine(), want.into_affine(), "n = {n}");
        }
    }

    /// The same for G2, which is a different field, different kernels, and the ones that sit
    /// at the 255-register ceiling.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn small_g2_msm_matches_a_naive_sum() {
        use g16_field::{CurveGroup, PrimeGroup};
        let Some(cuda) = device() else { return };
        let m = CudaMsm::without_key(&cuda).expect("compile MSM unit");
        let n = 200usize;
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut cur = G2Projective::generator();
        for i in 0..n {
            cur += G2Projective::generator();
            bases.push(cur.into_affine());
            scalars.push(Fr::from((i as u64) * 7919 + 3));
        }
        scalars[0] = Fr::zero();
        scalars[1] = Fr::one();
        let want = bases
            .iter()
            .zip(&scalars)
            .fold(G2Projective::zero(), |a, (b, s)| a + *b * s);
        let db = m.upload_g2_bases(&bases).expect("upload bases");
        let ds = m.upload_scalars(&scalars).expect("upload scalars");
        let got = m
            .msm(Job::G2(JobG2 {
                bases: &db,
                base_off: 0,
                scalars: ds.as_scalars(),
                scalar_off: 0,
                n,
            }))
            .expect("msm")
            .g2()
            .expect("G2 result");
        assert_eq!(got.into_affine(), want.into_affine());
    }

    /// The path stage 9 actually takes: scalars that are already a device buffer, never
    /// copied to the host. Checked against the host-packed path on the same values, because
    /// the Montgomery-versus-standard asymmetry silently costs a factor of R if it is got
    /// wrong and there is no other symptom.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn device_resident_scalars_agree_with_host_packing() {
        use g16_field::{CurveGroup, PrimeGroup};
        let Some(cuda) = device() else { return };
        let m = CudaMsm::without_key(&cuda).expect("compile MSM unit");
        let n = 512usize;
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut cur = G1Projective::generator();
        for i in 0..n {
            cur += G1Projective::generator();
            bases.push(cur.into_affine());
            scalars.push(Fr::from((i as u64) * 104_729 + 17));
        }
        let want = bases
            .iter()
            .zip(&scalars)
            .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
        let db = m.upload_g1_bases(&bases).expect("upload bases");

        let host = m.upload_scalars(&scalars).expect("upload scalars");
        let run = |sc: Scalars<'_>| {
            m.msm(Job::G1(JobG1 {
                bases: &db,
                base_off: 0,
                scalars: sc,
                scalar_off: 0,
                n,
            }))
            .expect("msm")
            .g1()
            .expect("G1 result")
        };
        assert_eq!(run(host.as_scalars()).into_affine(), want.into_affine());

        // Now the same values as a Montgomery buffer on the device, converted in place, the
        // way `stages` leaves H behind.
        let packed = g16_gpu_layout::PackedFr::pack_slice(&scalars);
        let mont = upload_words(m.stream(), as_words(&packed)).expect("upload Montgomery H");
        let std = m
            .scalars_from_device_mont(&mont, n)
            .expect("convert to standard form");
        assert_eq!(run(std.as_scalars()).into_affine(), want.into_affine());
    }

    /// A device handle from another backend must come back as an error, not as a panic and
    /// certainly not as a dereferenced foreign pointer.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn an_h_handle_from_another_backend_is_refused() {
        let Some(cuda) = device() else { return };
        let m = CudaMsm::without_key(&cuda).expect("compile MSM unit");
        let h = HPoly::Device {
            tag: "metal",
            len: 8,
            data: std::sync::Arc::new(()),
        };
        // `MsmOutputs` has no `Debug`, so this cannot use `expect_err`.
        let msg = match m.msms(&[], &h, &mut StageTimings::default()) {
            Ok(_) => panic!("a foreign device handle must be refused, not read"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("metal"), "unhelpful message: {msg}");
    }
}
