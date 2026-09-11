//! Stages 5-9: the five Pippenger multi-scalar multiplications, on the GPU.
//!
//! The algorithm and every decision behind it are documented at the top of
//! `src/shaders/msm.metal`, which is the half of this module that does the work. In
//! summary: a counting sort by bucket index built from 32-bit atomics on plain counters,
//! so no bucket is ever written by two threads; a segmented accumulation that gives every
//! thread a fixed-length slice of the sorted entries, so per-thread work is uniform even
//! when one bucket holds a tenth of the input; and a threadgroup reduction that hands the
//! host one point per window.
//!
//! What this file owns is the shape of the submission, and that is where the Apple
//! silicon result actually lives. Measured on this M2 Max during the scouting phase, a
//! command buffer costs 0.16 ms to commit and wait on, and does not get cheaper with
//! less work in it, while an extra dispatch encoded into an already-open command buffer
//! costs 2 to 3 us. That is a factor of about 60. Both Metal MSM implementations we read
//! submit one command buffer per stage, four or more per MSM; at this prover's domain
//! sizes that alone would cost more than the arithmetic. [`MetalMsm::msm_batch`]
//! therefore encodes all five MSMs, 37 dispatches, into ONE command buffer with ONE
//! `wait_until_completed`, and reads back a few hundred points at the end.
//!
//! # What is on the GPU and what is not
//!
//! On the GPU: the scalar classification and signed-digit recoding, the counting sort,
//! bucket accumulation for both G1 and G2, the bucket merge, the per-window reduction,
//! and the sum of the bases whose scalar is 1. G2 is not a CPU fallback; `Fq` and `Fq2`
//! are implemented in MSL in this crate because the field prelude only covers `Fr`.
//!
//! On the host: packing the scalars into standard form, choosing the window width, the
//! Horner combination of `n_windows` points per MSM, and the sum of the per-threadgroup
//! partials. That is a few hundred curve additions per proof against tens of millions on
//! the device, and it is the same place zkonduit's Metal MSM leaves its window tail, for
//! the same reason: the serial part is small enough that moving it costs more in
//! dispatches than it saves.
//!
//! # Measured, on this M2 Max
//!
//! Stages 5-9 only, warm, medians of nine. Bases upload is `prepare` work and is outside
//! both timed regions; host scalar packing is per-proof work and is inside the GPU's.
//!
//! | artifact | constraints | CPU ms | GPU ms | speedup |
//! |---|---|---|---|---|
//! | `tiny_mul` | 2 | 0.26 | 2.5 | **0.10x** |
//! | `js_1x1_d8` | 3,359 | 17.9 | 17.5 | 1.02x |
//! | `js_2x2_d16` | 10,153 | 48.8 | 25.4 | 1.9x |
//! | `js_2x2_d32` | 17,929 | 83.4 | 30.4 | 2.7x |
//! | `js_8x8_d32` | 70,357 | 284 | 76 | 3.7x |
//! | `js_16x16_d32` | 140,261 | 533 | 127 | 4.2x |
//!
//! The crossover sits at roughly 3,400 constraints, and below it the GPU loses badly:
//! at `tiny_mul` the whole batch is dispatch and submission cost, ten times what the
//! CPU needs to do the arithmetic. That is a real result and the reason
//! [`MetalMsm::msm_batch`] exists in the shape it does rather than as five calls.
//!
//! Two caveats on the speedups, both structural. Stages 5-9 are 76 to 82% of CPU proving
//! time, so 4.2x here is not 4.2x on a proof; and the CPU side of the comparison is our
//! own Pippenger, which is itself 11 to 19% behind rapidsnark.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

use ark_ff::{AdditiveGroup, One, Zero};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLDispatchType, MTLResourceOptions, MTLSize,
};

use g16_core::ProveError;
use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

use crate::kernels::{FR_MSL, MSM_MSL};
use crate::layout::{
    as_bytes, Packed, PackedFq, PackedFq2, PackedG1Affine, PackedG2Affine, PackedScalar,
};

fn err(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "metal",
        reason: reason.into(),
    }
}

// ---------------------------------------------------------------------------
// Window sizing
// ---------------------------------------------------------------------------

/// Bits the signed recoding is laid out over: 254 for BN254's `Fr`, plus one so the
/// carry out of the top window is provably zero. Must match `g16_msm::RECODE_BITS`.
const RECODE_BITS: usize = 255;

/// Caps the bucket array at 2^15 points per window.
const MAX_WINDOW: u32 = 16;

/// Threads per threadgroup in the reduction and the ones kernel. Must equal `REDUCE_TG`
/// in `msm.metal`, which sizes the threadgroup array; the dispatch may use fewer if the
/// pipeline reports a lower maximum, and the kernel reads the real count at runtime.
const REDUCE_TG: usize = 64;

/// Threadgroup array size of the prefix-sum kernel. Same contract as [`REDUCE_TG`].
const SCAN_TG: usize = 256;

/// Entries one thread of the segmented accumulation owns.
///
/// Accumulation costs `SLICE_LEN` mixed additions per thread and the merge costs
/// `max_bucket_count / SLICE_LEN` full additions for the fattest bucket, so the balance
/// point is near the square root of the worst occupancy, which is a few hundred on these
/// artifacts. 64 sits under that on purpose: it also keeps the thread count high enough
/// to fill the machine at the smaller domains, where there are only a few thousand
/// slices to begin with. Override with `G16_METAL_MSM_L` to sweep it.
const SLICE_LEN: usize = 64;

fn slice_len() -> usize {
    std::env::var("G16_METAL_MSM_L")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(SLICE_LEN)
}

/// Slice length the segmented accumulation actually dispatches with, per plan.
///
/// [`SLICE_LEN`] balances accumulation against merge for a plan big enough to fill the
/// machine, and it is the constant the window cost model was fitted with, so
/// [`window_size`] keeps reading it. But a witness plan has a few hundred general
/// scalars: at 64 entries a slice that is ~500 threads, each a *dependent* chain of 64
/// mixed additions, on a device that wants thousands of threads before it can hide any
/// latency. Occupancy binds long before the merge does, so small plans trade slice
/// length for threads until the dispatch reaches [`SEG_TARGET_THREADS`]. Measured on
/// the csp artifacts: the witness plans' bucket half drops 2x in G1 and G2 both, and a
/// plan at H's size keeps slice 64 and does not move.
const SEG_TARGET_THREADS: usize = 4096;

fn slice_len_for(n_windows: usize, cap: usize) -> usize {
    let forced = std::env::var("G16_METAL_MSM_L")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    if let Some(l) = forced {
        return l;
    }
    let mut l = SLICE_LEN;
    while l > 8 && n_windows * cap.div_ceil(l) < SEG_TARGET_THREADS {
        l /= 2;
    }
    l
}

/// Threadgroups per window in the reduce, chosen the same way [`slice_len_for`] chooses
/// its slice length: by occupancy, not by work.
///
/// One threadgroup per window was fine at c=8, where 32 windows of 128 buckets keep the
/// segments short. At c=13 it is 20 threadgroups of [`REDUCE_TG`] threads on a device
/// with more cores than threadgroups, each thread a dependent chain of 128 full
/// additions, and it measured 3.96 ms *flat between 2^16 and 2^17*: cost that does not
/// scale with the input is latency, not work. Splitting each window across groups costs
/// one `pt_mul_small` per thread (the segment identity already carries the window-global
/// bucket offset) plus `reduce_groups - 1` host-side additions per window, both noise.
/// Groups double until the dispatch reaches [`REDUCE_TARGET_THREADS`], floored so every
/// thread keeps at least [`REDUCE_TG`] buckets per group. The target is higher than the
/// accumulation's because a reduce thread pays a fixed `pt_mul_small` on top of its
/// segment, so past the sweet spot more threads mean more of that tax: at c=13 on the
/// csp artifacts the sweep read 2.52 ms at 4 groups, 2.06 at 8, 2.48 at 16, 3.68 at 32.
/// Override with `G16_METAL_MSM_RG` to sweep it.
const REDUCE_TARGET_THREADS: usize = 8192;

fn reduce_groups_for(n_windows: usize, n_buckets: usize) -> usize {
    let forced = std::env::var("G16_METAL_MSM_RG")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    if let Some(g) = forced {
        return g.min(n_buckets.max(1));
    }
    let mut g = 1;
    while n_buckets / (2 * g) >= REDUCE_TG && n_windows * g * REDUCE_TG < REDUCE_TARGET_THREADS {
        g *= 2;
    }
    g
}

/// `G16_METAL_MSM_LEGACY_ACC=1` swaps the segmented accumulation for the simple
/// one-thread-per-bucket kernel. Kept only so the load-imbalance claim in
/// `shaders/msm.metal` can be reproduced rather than taken on trust.
fn legacy_accumulate() -> bool {
    std::env::var("G16_METAL_MSM_LEGACY_ACC").as_deref() == Ok("1")
}

/// Window width for `m` scalars that actually reach the buckets.
///
/// `m` is the general-scalar count, not the input length. That distinction matters
/// enormously: a 140k-long witness with 2k general scalars is a 2k problem, and sizing
/// the window for 140k would allocate 16384 buckets per window and spend half a million
/// point additions reducing buckets that 34k additions filled.
///
/// # Why this is not the CPU's cost model
///
/// The CPU minimises `W * (m + 3 * 2^(c-1))`: windows times bucket fills plus the
/// running-sum reduction. On the GPU that model picks the wrong `c`, and the measurements
/// say so loudly. On `js_8x8_d32`'s A MSM, forcing each width in turn gave
/// c=11 23.9 ms, c=12 16.3, **c=13 9.5**, c=14 21.4, c=15 20.0. That is not a smooth
/// curve with a shallow optimum, it is a 2.3x swing between neighbours, and the CPU
/// model cannot see it because it has no term for the thing causing it.
///
/// The cause is the top window. The digits are laid out over `RECODE_BITS` bits in `W`
/// windows of `c`, and `W * c` overshoots `RECODE_BITS` by up to `c - 1` bits, so the
/// top window has only `RECODE_BITS - (W-1)*c` meaningful bits and its digits crowd into
/// `2^(top_bits - 1)` buckets instead of `2^(c-1)`. At c=11 that is TWO buckets holding
/// half the scalars each. On the CPU this costs nothing, because a serial bucket loop
/// only cares about the total. On the GPU it is the critical path: those buckets' runs
/// span thousands of slices, and `msm_merge_*` walks a bucket's slice range in one
/// thread while the rest of the machine waits.
///
/// # The three terms, and where their constants come from
///
/// Fitted to this M2 Max by measuring five window widths on two artifacts and solving
/// for the coefficients, then checked against the eight measurements not used in the fit
/// (largest error 8%):
///
/// * `MADD_US` per bucket fill: `W * m` mixed additions.
/// * `ROW_US` per bucket: `W * 2^(c-1)` buckets, each cleared, merged, and passed twice
///   through the running-sum reduction.
/// * `MERGE_US` per merge iteration on the fattest bucket, which is the serial term. It
///   is 8300x the cost of a bucket fill because it happens in one lane with the rest of
///   the device idle, which is exactly why it dominates when the top window degenerates.
///
/// `ROW_US` was refitted after [`reduce_groups_for`] landed, because most of what it
/// priced was the old one-threadgroup-per-window reduce sitting idle: 2.06 ms over
/// 81,920 rows is 0.025 us, down from 0.058. That refit is what moves H from c=8 to
/// c=13 at 2^16, which does 37% fewer bucket fills and measured 6.16 ms against 7.36.
/// Checked against a c sweep of {8, 10, 12, 13, 14, 16} on both csp artifacts: the
/// model ranks c=13 first at both sizes, as the sweep does, and its largest absolute
/// error (c=16, where the reduce's per-bucket cost keeps falling with bucket count) is
/// on a width 78% off the winner.
///
/// Override with `G16_METAL_MSM_C` to sweep it.
pub fn window_size(m: usize) -> u32 {
    if let Ok(v) = std::env::var("G16_METAL_MSM_C") {
        if let Ok(c) = v.parse::<u32>() {
            return c.clamp(2, MAX_WINDOW);
        }
    }
    /// One mixed addition in the segmented accumulation, microseconds.
    const MADD_US: f64 = 0.00324;
    /// One bucket's share of the clear, merge and reduction kernels, microseconds.
    const ROW_US: f64 = 0.025;
    /// One iteration of the merge loop on the busiest bucket, microseconds. Serial.
    const MERGE_US: f64 = 27.0;

    let slice = slice_len() as f64;
    let mut best = 3;
    let mut best_cost = f64::MAX;
    for c in 3..=MAX_WINDOW {
        let cu = c as usize;
        let w = RECODE_BITS.div_ceil(cu);
        // Bits the top window actually reaches, hence how many of its buckets are live.
        let top_bits = RECODE_BITS - (w - 1) * cu;
        let top_buckets = (1u64 << (top_bits.min(cu) - 1)) as f64;
        let fattest = m as f64 / top_buckets;

        let cost = MADD_US * (w * m) as f64
            + ROW_US * (w << (cu - 1)) as f64
            + MERGE_US * (fattest / slice);
        if cost < best_cost {
            best_cost = cost;
            best = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Device-side structs
// ---------------------------------------------------------------------------

/// Mirrors `struct MsmParams` in `msm.metal`. All counts, no pointers, passed by
/// `setBytes` so it never needs a buffer.
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
    reduce_groups: u32,
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
    assert!(core::mem::size_of::<MsmParams>() == 44);
    assert!(core::mem::size_of::<PackedXyzzG1>() == 128);
    assert!(core::mem::size_of::<PackedXyzzG2>() == 256);
};

unsafe impl Packed for PackedXyzzG1 {}
unsafe impl Packed for PackedXyzzG2 {}

/// XYZZ to arkworks' Jacobian, with no field inversion.
///
/// XYZZ carries the invariant `ZZ^3 = ZZZ^2`, so setting the Jacobian `Z = ZZZ` gives
/// `Z^2 = ZZ^3` and the point `(X * ZZ^2, Y * ZZ^3, ZZZ)` has
/// `x = X*ZZ^2 / ZZ^3 = X/ZZ` and `y = Y*ZZ^3 / ZZZ^3 = Y/ZZZ`, which is exactly the
/// XYZZ point. Three multiplications, versus two inversions if this went through affine,
/// and there are hundreds of these per proof.
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

// ---------------------------------------------------------------------------
// Resident inputs
// ---------------------------------------------------------------------------

/// A base vector, repacked and resident on the device. Built once per key in
/// `prepare`; the whole point of the `Backend` contract's stage grouping is that this
/// never happens per proof.
///
/// `inf` remembers which bases are the point at infinity. A zkey is full of them: the B
/// queries of the csp keys are 61% infinity, because a wire that appears in no B
/// constraint still owns a slot. An infinity base contributes nothing whatever its
/// scalar, so the ones gather (see [`Outputs::alloc`]) drops those indices on the host
/// instead of paying a device load and a dead branch per point per proof.
pub struct G1Bases {
    buf: Buffer,
    len: usize,
    inf: Vec<bool>,
}

pub struct G2Bases {
    buf: Buffer,
    len: usize,
    inf: Vec<bool>,
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

/// Scalars in **standard** form (see `layout`'s module docs for why not Montgomery),
/// resident on the device.
///
/// `general_prefix[i]` is how many of the first `i` scalars are neither 0 nor 1. It is
/// built during the pack, which already walks every scalar, so it is free, and it is
/// what lets [`MetalMsm::msm_batch`] size the window and the entry array for the work
/// that actually reaches the buckets rather than for the input length. A witness is
/// typically over 99% zeros and ones, so the difference is two orders of magnitude.
pub struct ScalarBuf {
    buf: Buffer,
    len: usize,
    general_prefix: Option<Vec<u32>>,
    /// Ascending indices of the scalars that are exactly 1, from the same
    /// classification pass. Host-only: what reaches the device is the per-job gather
    /// list [`Outputs::alloc`] filters from it, since which of these indices matter
    /// also depends on the job's bases. `None` when the buffer was never classified.
    ones_idx: Option<Vec<u32>>,
}

impl ScalarBuf {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// General scalars in `range`, or the whole range's length when the buffer came from
    /// the device and was never classified on the host. Overestimating is safe: it only
    /// oversizes the entry array and the window.
    fn general_in(&self, range: &Range<usize>) -> usize {
        match &self.general_prefix {
            Some(p) => (p[range.end] - p[range.start]) as usize,
            None => range.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// One MSM. `scalar_off` and `base_off` exist so the L MSM, whose scalars are the
/// private suffix of the witness, can share the witness buffer with the A and B MSMs
/// instead of uploading a second copy.
pub struct JobG1<'a> {
    pub bases: &'a G1Bases,
    pub base_off: usize,
    pub scalars: &'a ScalarBuf,
    pub scalar_off: usize,
    pub n: usize,
}

pub struct JobG2<'a> {
    pub bases: &'a G2Bases,
    pub base_off: usize,
    pub scalars: &'a ScalarBuf,
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
            MsmResult::G2(_) => Err(err("expected a G1 MSM result, got G2")),
        }
    }
    pub fn g2(self) -> Result<G2Projective, ProveError> {
        match self {
            MsmResult::G2(p) => Ok(p),
            MsmResult::G1(_) => Err(err("expected a G2 MSM result, got G1")),
        }
    }
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

struct Pipelines {
    zero_u32: ComputePipelineState,
    mont_to_std: ComputePipelineState,
    count: ComputePipelineState,
    scan: ComputePipelineState,
    scatter: ComputePipelineState,
    clear_g1: ComputePipelineState,
    clear_g2: ComputePipelineState,
    accumulate_g1: ComputePipelineState,
    accumulate_g2: ComputePipelineState,
    segmented_g1: ComputePipelineState,
    segmented_g2: ComputePipelineState,
    merge_g1: ComputePipelineState,
    merge_g2: ComputePipelineState,
    reduce_g1: ComputePipelineState,
    reduce_g2: ComputePipelineState,
    ones_g1: ComputePipelineState,
    ones_g2: ComputePipelineState,
    ones_idx_g1: ComputePipelineState,
    ones_idx_g2: ComputePipelineState,
}

/// Scratch buffers, kept between calls.
///
/// Two measured facts make this worth the code. A `memcpy` into a freshly allocated
/// shared buffer runs at 15.2 GB/s against 54.8 GB/s into one that has already been
/// touched, because the first write to every page faults it in; and the same first-touch
/// cost applies to a kernel's first write. Allocating the counting-sort scratch on every
/// proof would pay that every time, so buffers go back to the pool instead.
struct Pool {
    device: Device,
    free: Mutex<Vec<Buffer>>,
}

impl Pool {
    fn take(&self, bytes: usize) -> Buffer {
        let bytes = bytes.max(4);
        let mut free = self.free.lock().expect("scratch pool poisoned");
        // Smallest buffer that fits, so a single huge allocation cannot be handed out
        // for every small request and then be unavailable for the one that needs it.
        let mut best: Option<usize> = None;
        for (i, b) in free.iter().enumerate() {
            if (b.length() as usize) >= bytes {
                match best {
                    Some(j) if free[j].length() <= b.length() => {}
                    _ => best = Some(i),
                }
            }
        }
        if let Some(i) = best {
            return free.swap_remove(i);
        }
        drop(free);
        self.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    fn give(&self, bufs: Vec<Buffer>) {
        let mut free = self.free.lock().expect("scratch pool poisoned");
        free.extend(bufs);
        // Unbounded reuse would keep every shape any circuit ever asked for. This is a
        // cache, not an arena.
        if free.len() > 48 {
            free.sort_by_key(|b| std::cmp::Reverse(b.length()));
            free.truncate(48);
        }
    }
}

// ---------------------------------------------------------------------------
// The backend object
// ---------------------------------------------------------------------------

pub struct MetalMsm {
    device: Device,
    queue: CommandQueue,
    pipelines: Pipelines,
    pool: Pool,
}

impl MetalMsm {
    /// Picks the system default device and compiles every MSM kernel from source.
    ///
    /// This is the expensive call: runtime MSL compilation of this library measured
    /// around 60 ms during scouting, plus pipeline construction. It must happen once, in
    /// `prepare`, and never on a proving path. A prover that constructs this per proof
    /// is measuring the Metal compiler.
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default()
            .ok_or_else(|| err("no Metal device; this machine cannot run the metal backend"))?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        // One translation unit: the field prelude followed by this file. There is no
        // include path for a runtime-compiled source string, so the concatenation is the
        // include, and the prelude's header guard is what makes it safe.
        let source = format!("{FR_MSL}\n{MSM_MSL}\n");
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &opts)
            .map_err(|e| err(format!("MSL compilation failed: {e}")))?;

        let pso = |name: &str| -> Result<ComputePipelineState, ProveError> {
            let f = library
                .get_function(name, None)
                .map_err(|e| err(format!("kernel {name} not found: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| err(format!("pipeline {name}: {e}")))
        };

        let pipelines = Pipelines {
            zero_u32: pso("zero_u32")?,
            mont_to_std: pso("fr_mont_to_std")?,
            count: pso("msm_count")?,
            scan: pso("msm_scan")?,
            scatter: pso("msm_scatter")?,
            clear_g1: pso("msm_clear_g1")?,
            clear_g2: pso("msm_clear_g2")?,
            accumulate_g1: pso("msm_accumulate_g1")?,
            accumulate_g2: pso("msm_accumulate_g2")?,
            segmented_g1: pso("msm_segmented_g1")?,
            segmented_g2: pso("msm_segmented_g2")?,
            merge_g1: pso("msm_merge_g1")?,
            merge_g2: pso("msm_merge_g2")?,
            reduce_g1: pso("msm_reduce_g1")?,
            reduce_g2: pso("msm_reduce_g2")?,
            ones_g1: pso("msm_ones_g1")?,
            ones_g2: pso("msm_ones_g2")?,
            ones_idx_g1: pso("msm_ones_idx_g1")?,
            ones_idx_g2: pso("msm_ones_idx_g2")?,
        };

        let queue = device.new_command_queue();
        Ok(Self {
            pool: Pool {
                device: device.clone(),
                free: Mutex::new(Vec::new()),
            },
            device,
            queue,
            pipelines,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn shared_buffer<T: Packed>(&self, items: &[T]) -> Buffer {
        let bytes = as_bytes(items);
        if bytes.is_empty() {
            return self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }
        self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Repacks and uploads a G1 base vector. `ark_ec::G1Affine` is 72 bytes on this
    /// arkworks and carries an infinity flag, so a byte cast would hand the GPU a 72-byte
    /// stride where the shader reads 64. [`PackedG1Affine::from_affine`] reads the flag
    /// and maps infinity onto the all-zero encoding the kernels test for.
    pub fn upload_g1_bases(&self, bases: &[G1Affine]) -> G1Bases {
        let packed = PackedG1Affine::pack_slice(bases);
        G1Bases {
            buf: self.shared_buffer(&packed),
            len: bases.len(),
            inf: bases.iter().map(|b| b.infinity).collect(),
        }
    }

    pub fn upload_g2_bases(&self, bases: &[G2Affine]) -> G2Bases {
        let packed = PackedG2Affine::pack_slice(bases);
        G2Bases {
            buf: self.shared_buffer(&packed),
            len: bases.len(),
            inf: bases.iter().map(|b| b.infinity).collect(),
        }
    }

    /// Packs scalars into standard form and uploads them, classifying as it goes.
    ///
    /// One pass, and the classification it produces is what keeps the zero and one
    /// scalars out of Pippenger entirely.
    ///
    /// This is per-proof work on the MSM stage's critical path, and `from_fr` is a full
    /// Montgomery reduction per scalar, so the pass is chunked over the thread pool and
    /// writes straight into the Metal buffer rather than through an intermediate `Vec`.
    /// Each chunk counts its own generals and builds its stretch of the prefix locally;
    /// a serial fix-up then shifts every stretch by the chunks before it, which is one
    /// add per scalar against the reduction the parallel pass just paid.
    pub fn upload_scalars(&self, scalars: &[Fr]) -> ScalarBuf {
        use rayon::prelude::*;
        const CHUNK: usize = 4096;

        let n = scalars.len();
        let bytes = (n.max(1) * core::mem::size_of::<PackedScalar>()) as u64;
        let buf = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let mut prefix = vec![0u32; n + 1];
        let mut ones_idx = Vec::new();
        if n > 0 {
            // SAFETY: the buffer was just allocated with room for `n` packed scalars
            // and nothing has been encoded against it, so no dispatch can be reading it.
            let slots: &mut [PackedScalar] =
                unsafe { core::slice::from_raw_parts_mut(buf.contents().cast(), n) };
            let parts: Vec<(u32, Vec<u32>)> = slots
                .par_chunks_mut(CHUNK)
                .zip_eq(scalars.par_chunks(CHUNK))
                .zip_eq(prefix[1..].par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(ci, ((dst, src), pre))| {
                    let mut general = 0u32;
                    let mut ones = Vec::new();
                    for (i, ((d, s), p)) in dst.iter_mut().zip(src).zip(pre.iter_mut()).enumerate()
                    {
                        if s.is_one() {
                            ones.push((ci * CHUNK + i) as u32);
                        } else if !s.is_zero() {
                            general += 1;
                        }
                        *p = general;
                        *d = PackedScalar::from_fr(s);
                    }
                    (general, ones)
                })
                .collect();
            let mut off = 0u32;
            for (chunk, (total, ones)) in prefix[1..].chunks_mut(CHUNK).zip(&parts) {
                if off > 0 {
                    for p in chunk {
                        *p += off;
                    }
                }
                off += total;
                ones_idx.extend_from_slice(ones);
            }
        }
        ScalarBuf {
            buf,
            len: n,
            general_prefix: Some(prefix),
            ones_idx: Some(ones_idx),
        }
    }

    /// Converts a device-resident Montgomery `Fr` buffer (what the NTT leaves behind for
    /// stage 9) into standard-form scalars, in its own command buffer.
    ///
    /// No host classification is possible without reading the values back, so this
    /// buffer reports every scalar as general. That is exactly right for `H`, whose
    /// evaluations are dense, and it is why this is a separate entry point rather than
    /// the default.
    pub fn scalars_from_device_mont(
        &self,
        mont: &Buffer,
        len: usize,
    ) -> Result<ScalarBuf, ProveError> {
        let out = self.device.new_buffer(
            (len.max(1) * 32) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.mont_to_std);
        enc.set_buffer(0, Some(mont), 0);
        enc.set_buffer(1, Some(&out), 0);
        let n = len as u32;
        enc.set_bytes(2, 4, (&n as *const u32).cast());
        dispatch_1d(enc, &self.pipelines.mont_to_std, len, 64);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "stage 9 scalar conversion (mont_to_std)")?;
        Ok(ScalarBuf {
            buf: out,
            len,
            general_prefix: None,
            ones_idx: None,
        })
    }

    /// Wraps a device-resident buffer that already holds **standard-form** scalars
    /// (`layout::PackedScalar`), without dispatching anything.
    ///
    /// Stage 4 writes H in both forms precisely so stage 9 can read the standard copy
    /// directly; going through [`Self::scalars_from_device_mont`] instead re-runs
    /// `fr_mont_to_std` over the whole domain in an extra command buffer, duplicating
    /// work the pointwise kernel already did.
    ///
    /// The clone retains the `MTLBuffer`, but the *contents* stay owned by whoever
    /// allocated them (for H, the pooled stage scratch). The caller must keep the
    /// producing handle alive until every MSM reading this buffer has completed, or a
    /// recycled scratch could overwrite the scalars mid-flight. The proving path does:
    /// `MetalCircuit::msms` borrows the `HPoly` for its whole duration.
    ///
    /// Like the Montgomery entry point, no host classification is possible without a
    /// readback, so every scalar reports as general. Right for H, whose evaluations are
    /// dense.
    pub fn scalars_from_device_std(&self, std: &Buffer, len: usize) -> ScalarBuf {
        ScalarBuf {
            buf: std.clone(),
            len,
            general_prefix: None,
            ones_idx: None,
        }
    }

    /// One MSM on its own command buffer. Convenience for tests; the proving path should
    /// call [`Self::msm_batch`], because five separate command buffers cost five times
    /// the 0.16 ms submission floor.
    pub fn msm_g1(&self, bases: &G1Bases, scalars: &ScalarBuf) -> Result<G1Projective, ProveError> {
        let n = bases.len.min(scalars.len);
        let out = self.msm_batch(&[Job::G1(JobG1 {
            bases,
            base_off: 0,
            scalars,
            scalar_off: 0,
            n,
        })])?;
        out[0].g1()
    }

    pub fn msm_g2(&self, bases: &G2Bases, scalars: &ScalarBuf) -> Result<G2Projective, ProveError> {
        let n = bases.len.min(scalars.len);
        let out = self.msm_batch(&[Job::G2(JobG2 {
            bases,
            base_off: 0,
            scalars,
            scalar_off: 0,
            n,
        })])?;
        out[0].g2()
    }

    /// Every MSM in one command buffer, one `wait_until_completed`, one readback.
    ///
    /// Jobs that share a scalar buffer, offset and length share their digit pipeline:
    /// the A, B-in-G2 and B-in-G1 MSMs all run over the whole witness, so the counting
    /// sort runs once for the three of them and only the point stages are repeated. That
    /// removes six of the fifteen digit dispatches and, more to the point, two thirds of
    /// the scatter's memory traffic.
    pub fn msm_batch<'a>(&self, jobs: &[Job<'a>]) -> Result<Vec<MsmResult>, ProveError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        // ---- plan ----
        let mut plans: Vec<Plan<'a>> = Vec::new();
        let mut by_key: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut job_plan = Vec::with_capacity(jobs.len());

        for job in jobs {
            let (sbuf, soff, n, bases_len, boff) = match job {
                Job::G1(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
                Job::G2(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
            };
            if soff + n > sbuf.len {
                return Err(err(format!(
                    "scalar range {}..{} exceeds the {} scalars uploaded",
                    soff,
                    soff + n,
                    sbuf.len
                )));
            }
            if boff + n > bases_len {
                return Err(err(format!(
                    "base range {}..{} exceeds the {} bases uploaded",
                    boff,
                    boff + n,
                    bases_len
                )));
            }
            // Identity of the uploaded vector, not of its contents: two jobs share a
            // digit pipeline exactly when they read the same scalars over the same range.
            let key = (sbuf as *const ScalarBuf as usize, soff, n);
            let idx = match by_key.get(&key) {
                Some(&i) => i,
                None => {
                    let i = plans.len();
                    plans.push(Plan::new(sbuf, soff, n));
                    by_key.insert(key, i);
                    i
                }
            };
            job_plan.push(idx);
        }

        // ---- allocate ----
        let mut scratch: Vec<Buffer> = Vec::new();
        for p in &mut plans {
            p.alloc(&self.pool, &mut scratch);
        }
        let mut outs: Vec<Outputs> = Vec::with_capacity(jobs.len());
        for job in jobs {
            let plan = &plans[job_plan[outs.len()]];
            outs.push(Outputs::alloc(&self.pool, &mut scratch, job, plan));
        }

        // ---- encode, once ----
        //
        // `G16_METAL_MSM_PHASES=1` splits the single command buffer into one per digit
        // pipeline and one per point pipeline, waiting on each, and prints wall times to
        // stderr. Strictly a measurement aid: it adds one ~0.15 ms submission floor per
        // piece, so the sum reads slightly worse than the production path it explains.
        if std::env::var_os("G16_METAL_MSM_PHASES").is_some() {
            let run = |label: String,
                       f: &mut dyn FnMut(&ComputeCommandEncoderRef)|
             -> Result<(), ProveError> {
                let t = std::time::Instant::now();
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                f(enc);
                enc.end_encoding();
                cb.commit();
                // The production path below checks status; this one must too. A timing
                // read off a faulted command buffer is not a slow result, it is a
                // measurement of how long the GPU took to fail, reported as if it were
                // work. This path exists to explain timings, so a wrong one is worse here
                // than anywhere.
                crate::cb::wait_ok(cb, "MSM phase")?;
                eprintln!(
                    "[msm-phase] {label}: {:.2} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                Ok(())
            };
            for (pi, p) in plans.iter().enumerate() {
                run(
                    format!(
                        "digits plan{pi} n={} cap={} c={} w={}",
                        p.n, p.cap, p.c, p.n_windows
                    ),
                    &mut |enc| p.encode(self, enc),
                )?;
            }
            // `G16_METAL_MSM_PHASES=2` goes one level finer and times the four bucket
            // stages one command buffer each. Same caveat, four more submission floors.
            let fine = std::env::var("G16_METAL_MSM_PHASES").as_deref() == Ok("2");
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                let p = &plans[job_plan[i]];
                let kind = if out.is_g2 { "g2" } else { "g1" };
                if fine {
                    run(
                        format!("clear   job{i} {kind} rows={}", p.n_windows * p.n_buckets),
                        &mut |enc| out.encode_clear(self, enc, job, p),
                    )?;
                    run(
                        format!("accum   job{i} {kind} slices={}", p.n_windows * p.slices),
                        &mut |enc| out.encode_accumulate(self, enc, job, p),
                    )?;
                    run(
                        format!("merge   job{i} {kind} rows={}", p.n_windows * p.n_buckets),
                        &mut |enc| out.encode_merge(self, enc, job, p),
                    )?;
                    run(
                        format!("reduce  job{i} {kind} w={}", p.n_windows),
                        &mut |enc| out.encode_reduce(self, enc, job, p),
                    )?;
                } else {
                    run(
                        format!("buckets job{i} {kind} n={} cap={} c={}", p.n, p.cap, p.c),
                        &mut |enc| out.encode_buckets(self, enc, job, p),
                    )?;
                }
                run(
                    format!("ones    job{i} {kind} n={} groups={}", p.n, out.ones_groups),
                    &mut |enc| out.encode_ones(self, enc, job, p),
                )?;
            }
        } else {
            // A concurrent encoder, staged. The default serial encoder barriers every
            // dispatch against the previous one, so the 37 dispatches ran strictly one
            // at a time and the per-phase GPU times summed exactly to the batch total:
            // zero overlap. But most of these dispatches are independent, and the small
            // ones are latency chains that leave nearly the whole device idle: the four
            // witness MSMs' accumulations and ones scans are a few thousand threads
            // each, while H's accumulation is throughput-bound and can absorb them.
            //
            // Grouping by stage keeps the hazards trivial to state: every dispatch of
            // one phase is independent of every other dispatch of that phase (disjoint
            // outputs; shared inputs are read-only), and each phase reads only what
            // earlier phases wrote, so one full barrier between phases is both
            // necessary and sufficient. metal-rs 0.29 does not bind
            // memoryBarrierWithScope:, so it is called the way `cb` reads the
            // unbound timing properties.
            //
            // The ones scans have no dependency at all (scalars and bases in, own
            // buffer out) and are encoded into the accumulation phase, the widest one.
            let cb = self.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            let barrier = |enc: &ComputeCommandEncoderRef| {
                // SAFETY: `ComputeCommandEncoderRef` is `objc::Message`, and
                // memoryBarrierWithScope: is a documented MTLComputeCommandEncoder
                // method taking MTLBarrierScope; MTLBarrierScopeBuffers is 1 << 0.
                unsafe {
                    let () = msg_send![enc, memoryBarrierWithScope: 1u64];
                }
            };
            for p in &plans {
                p.encode_zero(self, enc);
            }
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                out.encode_clear(self, enc, job, &plans[job_plan[i]]);
            }
            barrier(enc);
            for p in &plans {
                p.encode_count(self, enc);
            }
            barrier(enc);
            for p in &plans {
                p.encode_scan(self, enc);
            }
            barrier(enc);
            for p in &plans {
                p.encode_scatter(self, enc);
            }
            barrier(enc);
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                out.encode_accumulate(self, enc, job, &plans[job_plan[i]]);
                out.encode_ones(self, enc, job, &plans[job_plan[i]]);
            }
            barrier(enc);
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                out.encode_merge(self, enc, job, &plans[job_plan[i]]);
            }
            barrier(enc);
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                out.encode_reduce(self, enc, job, &plans[job_plan[i]]);
            }
            enc.end_encoding();
            cb.commit();
            // Before this check the next statement reinterpreted pooled buffers regardless
            // of whether the GPU had actually written them, so a fault returned the
            // previous proof's window sums with an Ok.
            crate::cb::wait_ok(cb, "MSM batch")?;
        }

        // ---- combine ----
        //
        // Independent per job, and each one is a serial Horner plus a partial sum, so
        // the five jobs split over the thread pool rather than queueing behind B_g2's
        // G2 arithmetic.
        let results: Vec<MsmResult> = {
            use rayon::prelude::*;
            outs.par_iter()
                .enumerate()
                .map(|(i, out)| out.combine(&plans[job_plan[i]]))
                .collect()
        };

        // Buffers go home. Anything the plan and the outputs still reference is dead
        // now: `wait_until_completed` returned, so the GPU is finished with all of it.
        drop(plans);
        drop(outs);
        self.pool.give(scratch);

        Ok(results)
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
    reduce_groups: usize,
    scalars: &'a Buffer,
    counts: Option<Buffer>,
    cursor: Option<Buffer>,
    entries: Option<Buffer>,
}

impl<'a> Plan<'a> {
    fn new(scalars: &'a ScalarBuf, scalar_off: usize, n: usize) -> Self {
        let range = scalar_off..scalar_off + n;
        let general = scalars.general_in(&range);
        let c = window_size(general);
        let n_windows = RECODE_BITS.div_ceil(c as usize);
        let n_buckets = 1usize << (c - 1);
        let cap = general.max(1);
        let slice_len = slice_len_for(n_windows, cap);
        Self {
            n,
            scalar_off,
            c,
            n_windows,
            n_buckets,
            // Only general scalars emit entries, and each emits at most one per window,
            // so this bounds the scatter exactly. `max(1)` because a zero-length Metal
            // buffer is not a thing.
            cap,
            slice_len,
            slices: cap.div_ceil(slice_len).max(1),
            reduce_groups: reduce_groups_for(n_windows, n_buckets),
            scalars: &scalars.buf,
            counts: None,
            cursor: None,
            entries: None,
        }
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
            reduce_groups: self.reduce_groups as u32,
        }
    }

    fn alloc(&mut self, pool: &Pool, keep: &mut Vec<Buffer>) {
        let rows = self.n_windows * self.n_buckets;
        let counts = pool.take(rows * 4);
        let cursor = pool.take(rows * 4);
        // 8 bytes an entry: the bucket row travels with the point index so the
        // segmented accumulation can find run boundaries without recomputing digits.
        let entries = pool.take(self.n_windows * self.cap * 8);
        keep.push(counts.clone());
        keep.push(cursor.clone());
        keep.push(entries.clone());
        self.counts = Some(counts);
        self.cursor = Some(cursor);
        self.entries = Some(entries);
    }

    fn encode(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        self.encode_zero(msm, enc);
        self.encode_count(msm, enc);
        self.encode_scan(msm, enc);
        self.encode_scatter(msm, enc);
    }

    /// The counters must start at zero, and unlike the bucket array (whose kernel
    /// writes rather than accumulates) they cannot rely on a fresh allocation, since
    /// the pool hands back used buffers.
    fn encode_zero(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let rows = self.n_windows * self.n_buckets;
        enc.set_compute_pipeline_state(&msm.pipelines.zero_u32);
        enc.set_buffer(
            0,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        let len = rows as u32;
        enc.set_bytes(1, 4, (&len as *const u32).cast());
        dispatch_1d(enc, &msm.pipelines.zero_u32, rows, 256);
    }

    fn encode_count(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.count);
        enc.set_buffer(0, Some(self.scalars), 0);
        enc.set_buffer(
            1,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 2, &p);
        dispatch_1d(enc, &msm.pipelines.count, self.n, 64);
    }

    fn encode_scan(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.scan);
        enc.set_buffer(
            0,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        enc.set_buffer(
            1,
            Some(self.cursor.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 2, &p);
        let scan_tg = SCAN_TG.min(msm.pipelines.scan.max_total_threads_per_threadgroup() as usize);
        enc.dispatch_thread_groups(
            MTLSize::new(self.n_windows as u64, 1, 1),
            MTLSize::new(scan_tg as u64, 1, 1),
        );
    }

    fn encode_scatter(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.scatter);
        enc.set_buffer(0, Some(self.scalars), 0);
        enc.set_buffer(
            1,
            Some(self.cursor.as_ref().expect("plan not allocated")),
            0,
        );
        enc.set_buffer(
            2,
            Some(self.entries.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 3, &p);
        dispatch_1d(enc, &msm.pipelines.scatter, self.n, 64);
    }
}

// ---------------------------------------------------------------------------
// A single MSM's point stages and its readback.
// ---------------------------------------------------------------------------

struct Outputs {
    buckets: Buffer,
    spill_pts: Buffer,
    spill_rows: Buffer,
    window_sums: Buffer,
    ones: Buffer,
    ones_groups: usize,
    /// Base indices of the live one-scalar contributions, when the host classified the
    /// scalars: `msm_ones_idx_*` gathers exactly these. `None` falls back to the
    /// `msm_ones_*` scan over all `n` scalars, which is the only option for a
    /// device-resident buffer.
    ones_idx: Option<(Buffer, usize)>,
    base_off: usize,
    is_g2: bool,
}

/// Threadgroups in the ones kernel.
///
/// The scan is a *dependent* chain: each thread's accumulator waits on its previous
/// mixed addition, and about every other scalar of a bit-decomposition witness is a
/// one, so a thread's chain is as long as its stretch of the input. The first shape of
/// this function handed each thread 64 scalars regardless of `n`, which is 832 threads
/// at 2^16 on a device with 4,864 ALUs: the four witness MSMs' ones scans were flat in
/// `n` because they were latency chains at a sixth of occupancy. Eight scalars per
/// thread puts thousands of threads in flight; the price is that the host sums up to
/// 256 partials per MSM instead of 64, which is still under 0.1 ms. Swept via
/// `G16_METAL_MSM_ONES_SPT`: 4 and 8 tie, 16 and up climb back toward the old shape.
fn ones_groups_for(n: usize) -> usize {
    let spt = std::env::var("G16_METAL_MSM_ONES_SPT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(8);
    n.div_ceil(REDUCE_TG * spt).clamp(1, 256)
}

impl Outputs {
    fn alloc(pool: &Pool, keep: &mut Vec<Buffer>, job: &Job<'_>, plan: &Plan<'_>) -> Self {
        let (is_g2, base_off, point_bytes, scalar_off, sbuf, inf) = match job {
            Job::G1(j) => (
                false,
                j.base_off,
                core::mem::size_of::<PackedXyzzG1>(),
                j.scalar_off,
                j.scalars,
                &j.bases.inf,
            ),
            Job::G2(j) => (
                true,
                j.base_off,
                core::mem::size_of::<PackedXyzzG2>(),
                j.scalar_off,
                j.scalars,
                &j.bases.inf,
            ),
        };

        // A classified scalar buffer turns the ones scan into a gather. The buffer's
        // one-indices are sorted, so the job's range is a subrange; mapping each index
        // to its base and dropping the bases at infinity happens here, once per proof,
        // because it is exactly what the kernel would otherwise discover point by
        // point. On the csp B queries, 61% of the bases are infinity, and this is
        // where their scalars stop costing anything.
        let ones_idx = sbuf.ones_idx.as_ref().map(|idx| {
            let lo = idx.partition_point(|&e| (e as usize) < scalar_off);
            let hi = idx.partition_point(|&e| (e as usize) < scalar_off + plan.n);
            let gather: Vec<u32> = idx[lo..hi]
                .iter()
                .map(|&e| (base_off + (e as usize - scalar_off)) as u32)
                .filter(|&b| !inf[b as usize])
                .collect();
            let buf = pool.take(gather.len() * 4);
            if !gather.is_empty() {
                // SAFETY: the pooled buffer holds at least `gather.len()` u32s and the
                // batch that could read it has not been committed yet.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        gather.as_ptr(),
                        buf.contents().cast::<u32>(),
                        gather.len(),
                    );
                }
            }
            (buf, gather.len())
        });
        let ones_groups = match &ones_idx {
            Some((_, count)) => ones_groups_for(*count),
            None => ones_groups_for(plan.n),
        };
        let buckets = pool.take(plan.n_windows * plan.n_buckets * point_bytes);
        // Two spill slots per slice: at most one run of a slice continues backwards and
        // at most one continues forwards.
        let spill_slots = 2 * plan.n_windows * plan.slices;
        let spill_pts = pool.take(spill_slots * point_bytes);
        let spill_rows = pool.take(spill_slots * 4);
        let window_sums = pool.take(plan.n_windows * plan.reduce_groups * point_bytes);
        let ones = pool.take(ones_groups * point_bytes);
        keep.push(buckets.clone());
        keep.push(spill_pts.clone());
        keep.push(spill_rows.clone());
        keep.push(window_sums.clone());
        keep.push(ones.clone());
        if let Some((buf, _)) = &ones_idx {
            keep.push(buf.clone());
        }
        Self {
            buckets,
            spill_pts,
            spill_rows,
            window_sums,
            ones,
            ones_groups,
            ones_idx,
            base_off,
            is_g2,
        }
    }

    /// The Pippenger half: clear, segmented accumulation, merge, reduce. Split from
    /// [`Self::encode_ones`] so phase mode can time the bucket machinery and the ones
    /// scan separately. The production path in [`MetalMsm::msm_batch`] does not call
    /// this: it encodes the individual stages itself, grouped across jobs, so that
    /// independent dispatches share a concurrent phase.
    fn encode_buckets(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        self.encode_clear(msm, enc, job, plan);
        self.encode_accumulate(msm, enc, job, plan);
        self.encode_merge(msm, enc, job, plan);
        self.encode_reduce(msm, enc, job, plan);
    }

    fn params_for(&self, plan: &Plan<'_>) -> MsmParams {
        let mut p = plan.params();
        p.base_off = self.base_off as u32;
        p.ones_groups = self.ones_groups as u32;
        p
    }

    /// A pooled bucket array holds the previous proof's points, and a bucket that no
    /// slice writes directly has to read as the identity. The legacy accumulation
    /// writes every bucket, so it needs no clear.
    fn encode_clear(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        if legacy_accumulate() {
            return;
        }
        let p = self.params_for(plan);
        let clear_pso = match job {
            Job::G1(_) => &msm.pipelines.clear_g1,
            Job::G2(_) => &msm.pipelines.clear_g2,
        };
        enc.set_compute_pipeline_state(clear_pso);
        enc.set_buffer(0, Some(&self.buckets), 0);
        set_params(enc, 1, &p);
        dispatch_1d(enc, clear_pso, plan.n_windows * plan.n_buckets, 256);
    }

    fn encode_accumulate(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        let p = self.params_for(plan);
        let rows = plan.n_windows * plan.n_buckets;
        if legacy_accumulate() {
            let (acc_pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.accumulate_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.accumulate_g2, &j.bases.buf),
            };
            enc.set_compute_pipeline_state(acc_pso);
            enc.set_buffer(0, Some(plan.entries.as_ref().unwrap()), 0);
            enc.set_buffer(1, Some(bases), 0);
            enc.set_buffer(2, Some(plan.counts.as_ref().unwrap()), 0);
            enc.set_buffer(3, Some(plan.cursor.as_ref().unwrap()), 0);
            enc.set_buffer(4, Some(&self.buckets), 0);
            set_params(enc, 5, &p);
            dispatch_1d(enc, acc_pso, rows, 64);
        } else {
            let (seg_pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.segmented_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.segmented_g2, &j.bases.buf),
            };
            enc.set_compute_pipeline_state(seg_pso);
            enc.set_buffer(0, Some(plan.entries.as_ref().unwrap()), 0);
            enc.set_buffer(1, Some(bases), 0);
            enc.set_buffer(2, Some(plan.cursor.as_ref().unwrap()), 0);
            enc.set_buffer(3, Some(&self.buckets), 0);
            enc.set_buffer(4, Some(&self.spill_pts), 0);
            enc.set_buffer(5, Some(&self.spill_rows), 0);
            set_params(enc, 6, &p);
            dispatch_1d(enc, seg_pso, plan.n_windows * plan.slices, 64);
        }
    }

    fn encode_merge(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        if legacy_accumulate() {
            return;
        }
        let p = self.params_for(plan);
        let merge_pso = match job {
            Job::G1(_) => &msm.pipelines.merge_g1,
            Job::G2(_) => &msm.pipelines.merge_g2,
        };
        enc.set_compute_pipeline_state(merge_pso);
        enc.set_buffer(0, Some(&self.buckets), 0);
        enc.set_buffer(1, Some(&self.spill_pts), 0);
        enc.set_buffer(2, Some(&self.spill_rows), 0);
        enc.set_buffer(3, Some(plan.counts.as_ref().unwrap()), 0);
        enc.set_buffer(4, Some(plan.cursor.as_ref().unwrap()), 0);
        set_params(enc, 5, &p);
        dispatch_1d(enc, merge_pso, plan.n_windows * plan.n_buckets, 64);
    }

    fn encode_reduce(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        let p = self.params_for(plan);
        let red_pso = match job {
            Job::G1(_) => &msm.pipelines.reduce_g1,
            Job::G2(_) => &msm.pipelines.reduce_g2,
        };
        enc.set_compute_pipeline_state(red_pso);
        enc.set_buffer(0, Some(&self.buckets), 0);
        enc.set_buffer(1, Some(&self.window_sums), 0);
        set_params(enc, 2, &p);
        let tg = REDUCE_TG.min(red_pso.max_total_threads_per_threadgroup() as usize);
        enc.dispatch_thread_groups(
            MTLSize::new((plan.n_windows * plan.reduce_groups) as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
    }

    /// The scalar-of-1 half: one dispatch. Independent of the digit pipeline and of
    /// every bucket stage. With a gather list it is `msm_ones_idx_*` over exactly the
    /// live contributions; without one it is the `msm_ones_*` scan over all `n`
    /// scalars.
    fn encode_ones(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        let mut p = self.params_for(plan);
        if let Some((idx, count)) = &self.ones_idx {
            let (pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.ones_idx_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.ones_idx_g2, &j.bases.buf),
            };
            // The gather kernel reads its length from `n`; the plan's other counts do
            // not apply to it.
            p.n = *count as u32;
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(idx), 0);
            enc.set_buffer(1, Some(bases), 0);
            enc.set_buffer(2, Some(&self.ones), 0);
            set_params(enc, 3, &p);
            let tg = REDUCE_TG.min(pso.max_total_threads_per_threadgroup() as usize);
            enc.dispatch_thread_groups(
                MTLSize::new(self.ones_groups as u64, 1, 1),
                MTLSize::new(tg as u64, 1, 1),
            );
            return;
        }
        let (ones_pso, bases) = match job {
            Job::G1(j) => (&msm.pipelines.ones_g1, &j.bases.buf),
            Job::G2(j) => (&msm.pipelines.ones_g2, &j.bases.buf),
        };
        enc.set_compute_pipeline_state(ones_pso);
        enc.set_buffer(0, Some(plan.scalars), 0);
        enc.set_buffer(1, Some(bases), 0);
        enc.set_buffer(2, Some(&self.ones), 0);
        set_params(enc, 3, &p);
        let tg = REDUCE_TG.min(ones_pso.max_total_threads_per_threadgroup() as usize);
        enc.dispatch_thread_groups(
            MTLSize::new(self.ones_groups as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
    }

    fn combine(&self, plan: &Plan<'_>) -> MsmResult {
        let rg = plan.reduce_groups;
        if self.is_g2 {
            let w: &[PackedXyzzG2] = unsafe { read_back(&self.window_sums, plan.n_windows * rg) };
            let o: &[PackedXyzzG2] = unsafe { read_back(&self.ones, self.ones_groups) };
            // Each window's `reduce_groups` partials fold first, then the Horner.
            let sum_w = |k: usize| {
                let mut s = w[k * rg].to_projective();
                for x in &w[k * rg + 1..(k + 1) * rg] {
                    s += x.to_projective();
                }
                s
            };
            let mut acc = sum_w(plan.n_windows - 1);
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += sum_w(k);
            }
            for x in o {
                acc += x.to_projective();
            }
            MsmResult::G2(acc)
        } else {
            let w: &[PackedXyzzG1] = unsafe { read_back(&self.window_sums, plan.n_windows * rg) };
            let o: &[PackedXyzzG1] = unsafe { read_back(&self.ones, self.ones_groups) };
            // Horner over the windows, high to low, `c` doublings between each, each
            // window's `reduce_groups` partials folded first. Same order as the CPU
            // backend, so the two agree in the group and the audit can compare affine.
            let sum_w = |k: usize| {
                let mut s = w[k * rg].to_projective();
                for x in &w[k * rg + 1..(k + 1) * rg] {
                    s += x.to_projective();
                }
                s
            };
            let mut acc = sum_w(plan.n_windows - 1);
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += sum_w(k);
            }
            for x in o {
                acc += x.to_projective();
            }
            MsmResult::G1(acc)
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn set_params(enc: &ComputeCommandEncoderRef, index: u64, p: &MsmParams) {
    enc.set_bytes(
        index,
        core::mem::size_of::<MsmParams>() as u64,
        (p as *const MsmParams).cast(),
    );
}

/// Threadgroup size is a preference clamped by what the pipeline will actually accept.
///
/// Never hardcode 1024: a kernel carrying one CIOS Montgomery multiply already reports
/// 896 on this device from register pressure alone, and the XYZZ mixed-addition kernel
/// measured 512. Both reference implementations query the pipeline; so does this.
fn dispatch_1d(
    enc: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    n: usize,
    prefer: usize,
) {
    if n == 0 {
        return;
    }
    let tg = prefer
        .min(pso.max_total_threads_per_threadgroup() as usize)
        .max(1);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
}

/// # Safety
///
/// The buffer must hold at least `len` `T`s written by a completed command buffer, and
/// `T` must be a `Packed` type, so every bit pattern is valid.
unsafe fn read_back<T: Packed>(buf: &Buffer, len: usize) -> &[T] {
    core::slice::from_raw_parts(buf.contents().cast::<T>(), len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g16_core::{cpu::CpuBackend, Backend, StageTimings};
    use g16_field::CurveGroup;
    use g16_zkey::{wtns::Witness, ProvingKey};
    use std::path::PathBuf;

    fn artifact(name: &str) -> Option<PathBuf> {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts")
            .join(name);
        if d.join("circuit.zkey").exists() && d.join("circuit.wtns").exists() {
            Some(d)
        } else {
            None
        }
    }

    /// The same drift guard for the two threadgroup sizes, which had none.
    ///
    /// `REDUCE_TG` and `SCAN_TG` are the *array* sizes of the `threadgroup` scratch in
    /// `msm_reduce_*`, `msm_ones_*` and `msm_scan`, and the host dispatches that many
    /// threads. Raising either one on the Rust side alone makes every thread above the
    /// MSL array's length index out of bounds in threadgroup memory: no compile error,
    /// no Metal validation error, just wrong points. Measured, not assumed. Setting
    /// `REDUCE_TG` to 128 with the MSL left at 64 leaves H bit-exact and turns all five
    /// MSM outputs wrong, so the proof stops verifying with nothing to point at.
    #[test]
    fn msl_declares_the_same_threadgroup_sizes() {
        for (name, want) in [("REDUCE_TG", REDUCE_TG), ("SCAN_TG", SCAN_TG)] {
            let line = format!("#define {name} {want}");
            assert!(
                MSM_MSL.contains(&line),
                "shaders/msm.metal does not contain the line:\n{line}\n\
                 The host dispatches {want} threads and the MSL sizes its threadgroup \
                 array from its own #define; if they disagree the reduction writes out \
                 of bounds and the MSM answers are silently wrong."
            );
        }
    }

    /// The drift guard, same shape as `layout::tests::msl_declares_the_same_constants`:
    /// the Fq modulus and `N0` live in two languages and must be edited together.
    #[test]
    fn msl_declares_the_same_fq_constants() {
        let want_n = format!(
            "constant uint FQ_N[8] = {{ {} }};",
            crate::layout::FQ_MODULUS
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            MSM_MSL.contains(&want_n),
            "shaders/msm.metal does not contain the line:\n{want_n}"
        );
        let want_n0 = format!("constant uint FQ_N0 = 0x{:08x}u;", crate::layout::FQ_N0);
        assert!(
            MSM_MSL.contains(&want_n0),
            "shaders/msm.metal does not contain the line:\n{want_n0}"
        );
    }

    /// The MSL must compile and every kernel must produce a pipeline. Separate from the
    /// arithmetic tests so a compile error is reported as a compile error.
    #[test]
    fn the_library_compiles_and_every_kernel_has_a_pipeline() {
        let m = MetalMsm::new().expect("Metal device");
        // Occupancy is a measured property, not an assumption. Print it so a change in
        // register pressure shows up in the test log rather than as a mystery slowdown.
        eprintln!(
            "max threads/threadgroup: count {} scatter {} accumulate_g1 {} accumulate_g2 {} \
             reduce_g1 {} reduce_g2 {} (simd width {})",
            m.pipelines.count.max_total_threads_per_threadgroup(),
            m.pipelines.scatter.max_total_threads_per_threadgroup(),
            m.pipelines
                .accumulate_g1
                .max_total_threads_per_threadgroup(),
            m.pipelines
                .accumulate_g2
                .max_total_threads_per_threadgroup(),
            m.pipelines.reduce_g1.max_total_threads_per_threadgroup(),
            m.pipelines.reduce_g2.max_total_threads_per_threadgroup(),
            m.pipelines.count.thread_execution_width(),
        );
        assert!(m.pipelines.reduce_g2.max_total_threads_per_threadgroup() >= 1);
    }

    /// A handful of scalars against a naive sum, before anything the size of a real
    /// circuit. This is the test that isolates the point arithmetic: if the digit
    /// recoding or `madd` is wrong, it fails here rather than 140,000 points later.
    #[test]
    fn small_g1_msm_matches_a_naive_sum() {
        use g16_field::PrimeGroup;
        let m = MetalMsm::new().expect("Metal device");
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for n in [1usize, 2, 3, 31, 32, 33, 257, 1000] {
            let mut bases = Vec::with_capacity(n);
            let mut scalars = Vec::with_capacity(n);
            let mut cur = G1Projective::generator();
            for i in 0..n {
                cur += G1Projective::generator();
                bases.push(cur.into_affine());
                // Deliberately dense in the two special cases, which is what a witness
                // looks like and what the kernels take a different path for.
                scalars.push(match i % 4 {
                    0 => Fr::zero(),
                    1 => Fr::one(),
                    _ => {
                        let mut b = [0u8; 32];
                        for c in b.chunks_mut(8) {
                            c.copy_from_slice(&next().to_le_bytes());
                        }
                        <Fr as g16_field::PrimeField>::from_le_bytes_mod_order(&b)
                    }
                });
            }
            let want = bases
                .iter()
                .zip(&scalars)
                .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
            let db = m.upload_g1_bases(&bases);
            let ds = m.upload_scalars(&scalars);
            let got = m.msm_g1(&db, &ds).unwrap();
            assert_eq!(got.into_affine(), want.into_affine(), "n = {n}");
        }
    }

    #[test]
    fn small_g2_msm_matches_a_naive_sum() {
        use g16_field::PrimeGroup;
        let m = MetalMsm::new().expect("Metal device");
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
        let db = m.upload_g2_bases(&bases);
        let ds = m.upload_scalars(&scalars);
        let got = m.msm_g2(&db, &ds).unwrap();
        assert_eq!(got.into_affine(), want.into_affine());
    }

    /// The path stage 9 will actually take once the NTT keeps `H` on the device: a
    /// Montgomery `Fr` buffer converted in place rather than packed on the host.
    ///
    /// This is the asymmetry that silently costs a factor of R if it is got wrong, so it
    /// is checked against the host-packed path on the same values.
    #[test]
    fn device_resident_montgomery_scalars_agree_with_host_packing() {
        use g16_field::PrimeGroup;
        let m = MetalMsm::new().expect("Metal device");
        let n = 512usize;
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut cur = G1Projective::generator();
        for i in 0..n {
            cur += G1Projective::generator();
            bases.push(cur.into_affine());
            scalars.push(Fr::from((i as u64) * 104_729 + 17));
        }
        let packed = crate::layout::PackedFr::pack_slice(&scalars);
        let bytes = as_bytes(&packed);
        let mont = m.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let db = m.upload_g1_bases(&bases);
        let from_host = m.msm_g1(&db, &m.upload_scalars(&scalars)).unwrap();
        let from_device = m
            .msm_g1(&db, &m.scalars_from_device_mont(&mont, n).unwrap())
            .unwrap();
        let want = bases
            .iter()
            .zip(&scalars)
            .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
        assert_eq!(from_host.into_affine(), want.into_affine());
        assert_eq!(from_device.into_affine(), want.into_affine());
    }

    /// The real test: all five MSMs of an actual circuit, GPU against the CPU backend,
    /// on identical inputs. Points must be exactly equal after normalisation.
    fn five_msms_against_cpu(name: &str) {
        let Some(dir) = artifact(name) else {
            eprintln!("{name}: artifact missing, skipped");
            return;
        };
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;

        let circuit = CpuBackend::new().prepare(pk).expect("prepare");
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
        let cpu = circuit.msms(&witness, &h, &mut t).expect("cpu msms");
        let h_scalars = h.to_host().expect("cpu H is on the host");
        let pk = circuit.key();
        let n_public = circuit.n_public();

        let m = MetalMsm::new().expect("Metal device");
        let a_bases = m.upload_g1_bases(&pk.a_query);
        let b1_bases = m.upload_g1_bases(&pk.b_g1_query);
        let b2_bases = m.upload_g2_bases(&pk.b_g2_query);
        let l_bases = m.upload_g1_bases(&pk.l_query);
        let h_bases = m.upload_g1_bases(&pk.h_query);
        let w_scalars = m.upload_scalars(&witness);
        let h_dev = m.upload_scalars(h_scalars);

        let n = witness.len();
        let l_off = n_public + 1;
        let jobs = vec![
            Job::G1(JobG1 {
                bases: &a_bases,
                base_off: 0,
                scalars: &w_scalars,
                scalar_off: 0,
                n,
            }),
            Job::G2(JobG2 {
                bases: &b2_bases,
                base_off: 0,
                scalars: &w_scalars,
                scalar_off: 0,
                n,
            }),
            Job::G1(JobG1 {
                bases: &b1_bases,
                base_off: 0,
                scalars: &w_scalars,
                scalar_off: 0,
                n,
            }),
            Job::G1(JobG1 {
                bases: &l_bases,
                base_off: 0,
                scalars: &w_scalars,
                scalar_off: l_off,
                n: pk.l_query.len(),
            }),
            Job::G1(JobG1 {
                bases: &h_bases,
                base_off: 0,
                scalars: &h_dev,
                scalar_off: 0,
                n: h_scalars.len(),
            }),
        ];
        let got = m.msm_batch(&jobs).expect("gpu msms");

        assert_eq!(
            got[0].g1().unwrap().into_affine(),
            cpu.a_g1.into_affine(),
            "{name}: stage 5, MSM A -> G1"
        );
        assert_eq!(
            got[1].g2().unwrap().into_affine(),
            cpu.b_g2.into_affine(),
            "{name}: stage 6, MSM B -> G2"
        );
        assert_eq!(
            got[2].g1().unwrap().into_affine(),
            cpu.b_g1.into_affine(),
            "{name}: stage 7, MSM B -> G1"
        );
        assert_eq!(
            got[3].g1().unwrap().into_affine(),
            cpu.l_g1.into_affine(),
            "{name}: stage 8, MSM L -> G1"
        );
        assert_eq!(
            got[4].g1().unwrap().into_affine(),
            cpu.h_g1.into_affine(),
            "{name}: stage 9, MSM H -> G1"
        );
    }

    #[test]
    fn five_msms_match_cpu_on_tiny_mul() {
        five_msms_against_cpu("tiny_mul");
    }

    #[test]
    fn five_msms_match_cpu_on_js_1x1_d8() {
        five_msms_against_cpu("js_1x1_d8");
    }

    /// GPU against CPU on wall clock, warm, for the same five MSMs.
    ///
    /// Not a check, a measurement. Bases upload is `prepare` work and sits outside the
    /// timed region on both sides; scalar packing and upload is per-proof work and sits
    /// inside the GPU's. Run with
    /// `cargo test -p g16-metal --release measure_against_cpu -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn measure_against_cpu() {
        use std::time::Instant;
        let reps = 9;
        let t_start = Instant::now();
        let m = MetalMsm::new().expect("Metal device");
        let compile_ms = t_start.elapsed().as_secs_f64() * 1e3;
        println!("MSL compile + all pipelines: {compile_ms:.1} ms (once per process)");
        println!(
            "{:<14} {:>8} {:>8} {:>7} {:>8} {:>9} {:>9} {:>9} {:>8}",
            "artifact",
            "n_vars",
            "domain",
            "gen%",
            "c(wit/H)",
            "cpu ms",
            "gpu ms",
            "gpu-nosc",
            "upload"
        );

        for name in [
            "tiny_mul",
            "js_1x1_d8",
            "js_2x2_d16",
            "js_2x2_d32",
            "js_8x8_d32",
            "js_16x16_d32",
        ] {
            let Some(dir) = artifact(name) else {
                continue;
            };
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
            let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
            let circuit = CpuBackend::new().prepare(pk).expect("prepare");
            let mut t = StageTimings::default();
            let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
            let h_scalars = h.to_host().unwrap().to_vec();
            let pk = circuit.key();
            let n = witness.len();
            let l_off = circuit.n_public() + 1;

            let t0 = Instant::now();
            let a_bases = m.upload_g1_bases(&pk.a_query);
            let b1_bases = m.upload_g1_bases(&pk.b_g1_query);
            let b2_bases = m.upload_g2_bases(&pk.b_g2_query);
            let l_bases = m.upload_g1_bases(&pk.l_query);
            let h_bases = m.upload_g1_bases(&pk.h_query);
            let upload_ms = t0.elapsed().as_secs_f64() * 1e3;

            let general = {
                let s = m.upload_scalars(&witness);
                s.general_in(&(0..n))
            };

            let mut cpu = Vec::new();
            let mut gpu = Vec::new();
            let mut gpu_nosc = Vec::new();
            for r in 0..reps + 1 {
                let t0 = Instant::now();
                let mut tt = StageTimings::default();
                let _ = circuit.msms(&witness, &h, &mut tt).unwrap();
                let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

                let t0 = Instant::now();
                let w_scalars = m.upload_scalars(&witness);
                let h_dev = m.upload_scalars(&h_scalars);
                let pack_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                let jobs = vec![
                    Job::G1(JobG1 {
                        bases: &a_bases,
                        base_off: 0,
                        scalars: &w_scalars,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G2(JobG2 {
                        bases: &b2_bases,
                        base_off: 0,
                        scalars: &w_scalars,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &b1_bases,
                        base_off: 0,
                        scalars: &w_scalars,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &l_bases,
                        base_off: 0,
                        scalars: &w_scalars,
                        scalar_off: l_off,
                        n: pk.l_query.len(),
                    }),
                    Job::G1(JobG1 {
                        bases: &h_bases,
                        base_off: 0,
                        scalars: &h_dev,
                        scalar_off: 0,
                        n: h_scalars.len(),
                    }),
                ];
                let _ = m.msm_batch(&jobs).unwrap();
                let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
                // Drop the first round: it pays first-touch page faults on every scratch
                // buffer, which is exactly the cold cost the pool exists to pay once.
                if r > 0 {
                    cpu.push(cpu_ms);
                    gpu.push(pack_ms + dev_ms);
                    gpu_nosc.push(dev_ms);
                }
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let c_wit = window_size(general);
            let c_h = window_size(h_scalars.len());
            println!(
                "{:<14} {:>8} {:>8} {:>6.1}% {:>4}/{:<3} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
                name,
                n,
                circuit.domain_size(),
                100.0 * general as f64 / n as f64,
                c_wit,
                c_h,
                med(&mut cpu),
                med(&mut gpu),
                med(&mut gpu_nosc),
                upload_ms,
            );
        }
    }

    /// Per-MSM wall clock plus the bucket-occupancy histogram that explains it.
    ///
    /// One thread owns one bucket, so the accumulate dispatch finishes when the fattest
    /// bucket does. This test prints that number next to the average, because a ratio
    /// far from 1 is the difference between a dispatch that uses the machine and one
    /// that is a single serial loop with a quarter of a million idle threads beside it.
    #[test]
    #[ignore = "measurement, not a check"]
    fn measure_per_msm() {
        use std::time::Instant;
        let m = MetalMsm::new().expect("Metal device");
        for name in ["js_2x2_d16", "js_2x2_d32", "js_8x8_d32", "js_16x16_d32"] {
            let Some(dir) = artifact(name) else { continue };
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
            let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
            let circuit = CpuBackend::new().prepare(pk).expect("prepare");
            let mut t = StageTimings::default();
            let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
            let h_scalars = h.to_host().unwrap().to_vec();
            let pk = circuit.key();
            let n = witness.len();
            let l_off = circuit.n_public() + 1;

            let a_bases = m.upload_g1_bases(&pk.a_query);
            let b2_bases = m.upload_g2_bases(&pk.b_g2_query);
            let l_bases = m.upload_g1_bases(&pk.l_query);
            let h_bases = m.upload_g1_bases(&pk.h_query);
            let w_scalars = m.upload_scalars(&witness);
            let h_dev = m.upload_scalars(&h_scalars);

            println!("--- {name}");
            for (label, scalars_slice, job) in [
                ("A  G1", &witness[..] as &[Fr], 0usize),
                ("B  G2", &witness[..], 1),
                ("L  G1", &witness[l_off..], 2),
                ("H  G1", &h_scalars[..], 3),
            ] {
                let (nn, soff) = match job {
                    2 => (pk.l_query.len(), l_off),
                    3 => (h_scalars.len(), 0),
                    _ => (n, 0),
                };
                let sbuf = if job == 3 { &h_dev } else { &w_scalars };
                let jobs = match job {
                    1 => vec![Job::G2(JobG2 {
                        bases: &b2_bases,
                        base_off: 0,
                        scalars: sbuf,
                        scalar_off: soff,
                        n: nn,
                    })],
                    2 => vec![Job::G1(JobG1 {
                        bases: &l_bases,
                        base_off: 0,
                        scalars: sbuf,
                        scalar_off: soff,
                        n: nn,
                    })],
                    3 => vec![Job::G1(JobG1 {
                        bases: &h_bases,
                        base_off: 0,
                        scalars: sbuf,
                        scalar_off: soff,
                        n: nn,
                    })],
                    _ => vec![Job::G1(JobG1 {
                        bases: &a_bases,
                        base_off: 0,
                        scalars: sbuf,
                        scalar_off: soff,
                        n: nn,
                    })],
                };
                let _ = m.msm_batch(&jobs).unwrap();
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let _ = m.msm_batch(&jobs).unwrap();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }

                // Host-side replica of the kernel's recoding, for the histogram only.
                let general = scalars_slice
                    .iter()
                    .filter(|s| !(s.is_zero() || s.is_one()))
                    .count();
                let c = window_size(general);
                let nw = RECODE_BITS.div_ceil(c as usize);
                let nb = 1usize << (c - 1);
                let mut counts = vec![0u32; nw * nb];
                for s in scalars_slice {
                    if s.is_zero() || s.is_one() {
                        continue;
                    }
                    let big = <Fr as g16_field::PrimeField>::into_bigint(*s);
                    let limbs: &[u64] = big.as_ref();
                    for w in 0..nw {
                        let (mag, _) = host_signed_digit(limbs, w, c);
                        if mag != 0 {
                            counts[w * nb + (mag - 1) as usize] += 1;
                        }
                    }
                }
                let maxb = counts.iter().copied().max().unwrap_or(0);
                let total: u64 = counts.iter().map(|&x| x as u64).sum();
                let avg = total as f64 / (nw * nb) as f64;
                println!(
                    "{label}  n {:>7}  gen {:>7}  c {:>2}  W {:>2}  buckets {:>6}  \
                     entries {:>9}  avg/bucket {:>7.2}  max {:>7}  ratio {:>6.0}x  {:>8.2} ms",
                    nn,
                    general,
                    c,
                    nw,
                    nb,
                    total,
                    avg,
                    maxb,
                    maxb as f64 / avg.max(1e-9),
                    best
                );
            }
        }
    }

    /// The CPU-side twin of `sc_signed_digit`, used only by the histogram above.
    fn host_signed_digit(limbs: &[u64], i: usize, c: u32) -> (u32, bool) {
        let read = |off: usize, width: u32| -> u64 {
            let idx = off / 64;
            if idx >= limbs.len() {
                return 0;
            }
            let sh = off % 64;
            let mut buf = limbs[idx] >> sh;
            if sh + width as usize > 64 && idx + 1 < limbs.len() {
                buf |= limbs[idx + 1] << (64 - sh);
            }
            buf & ((1u64 << width) - 1)
        };
        let off = i * c as usize;
        let b = read(off, c);
        let carry = if off == 0 { 0 } else { read(off - 1, 1) };
        if (b >> (c - 1)) & 1 == 1 {
            let mag = (1u64 << c) - b - carry;
            (mag as u32, mag != 0)
        } else {
            ((b + carry) as u32, false)
        }
    }

    /// The larger artifacts. Slow in a debug build, so they are opt-in; run with
    /// `cargo test -p g16-metal --release -- --ignored`.
    #[test]
    #[ignore = "minutes on the CPU oracle side"]
    fn five_msms_match_cpu_on_every_artifact() {
        for name in ["js_2x2_d16", "js_2x2_d32", "js_8x8_d32", "js_16x16_d32"] {
            five_msms_against_cpu(name);
        }
    }
}

#[cfg(test)]
mod thread_safety {
    /// `PreparedCircuit` must be `Send + Sync`, so anything the Metal backend stores on
    /// one has to be too. Checked rather than assumed: `metal-rs` declares its handles
    /// `Send + Sync`, and MTLDevice, MTLBuffer, MTLComputePipelineState and
    /// MTLCommandQueue are documented thread-safe. Command buffers and encoders are not,
    /// and none of them outlive a `msm_batch` call.
    const fn assert_send_sync<T: Send + Sync>() {}
    const _: () = {
        assert_send_sync::<super::MetalMsm>();
        assert_send_sync::<super::G1Bases>();
        assert_send_sync::<super::G2Bases>();
        assert_send_sync::<super::ScalarBuf>();
    };
}
