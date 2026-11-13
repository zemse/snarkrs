//! Stages 5 to 9, host side: the digit pipeline, which is everything that depends only on
//! the scalars.
//!
//! [`crate::gen::msm`] explains the algorithm and every decision behind it. This file is the
//! other half: the window width, the parameter block, the five pipelines, the three scratch
//! buffers and the four dispatches that fill them. [`crate::points`] is the other end of it,
//! and [`crate::batch`] packs a whole proof's dispatches into a few short submissions.
//!
//! # What a digit pipeline is, and why there are three of them per proof rather than five
//!
//! Everything in here is a function of `(scalar buffer, range)` and nothing else, so jobs
//! that read the same scalars over the same range share it. The A, B-in-G2 and B-in-G1 MSMs
//! all run over the whole witness, so the counting sort runs once for the three of them; L
//! has its own range (`scalar_off = n_public + 1`) and H has its own buffer. That is
//! 3 pipelines and 12 dispatches per proof instead of 5 and 20, and more to the point it is
//! two thirds less scatter traffic. A key with enough bases at infinity gives A and B a
//! sort each instead, over the positions that are not ([`LiveRun`]).
//!
//! # The window width is chosen from `m`, not from `n`
//!
//! `m` is the count of scalars that are neither 0 nor 1. A 140k-long witness with 2k general
//! scalars is a 2k problem, and sizing the window for 140k would allocate 16384 buckets per
//! window and spend half a million point additions reducing buckets that 34k additions
//! filled. [`DigitPlan::new`] takes that count from the caller, because the caller is the
//! host packer which already walked every scalar.
//!
//! Which is also why the shape follows the witness. Under [`Work::Constant`] the plan takes
//! `m = n` whatever the scalars hold, the kernels skip nothing and a zero digit lands in one
//! of [`DUMMY_ROWS`] rows per window behind the real buckets, so every window's entry region
//! is exactly full and every dispatch is sized from the key.

use bytemuck::{Pod, Zeroable};
use snarkrs_gpu_layout::LIMBS;
use snarkrs_groth16::ProveError;

use crate::device::{bad, WgpuBackend};
use crate::gather::storage_entry;
use crate::gen::field::Variant;
use crate::gen::msm as wgsl;
use crate::params::ParamRing;
use crate::pipelines::Kernels;

/// Bytes one field element occupies on the device, in either encoding.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// Bytes one scattered entry occupies: `vec2<u32>`, the bucket row and the signed point
/// index.
pub const ENTRY_BYTES: u64 = 8;

// ---------------------------------------------------------------------------
// Window sizing
// ---------------------------------------------------------------------------

/// Bits the signed recoding is laid out over: 254 for BN254's `Fr`, plus one so the carry out
/// of the top window is provably zero. Must match `snarkrs_msm::RECODE_BITS` and
/// `snarkrs_metal::msm::RECODE_BITS`, and [`crate::gen::msm`]'s module docs carry the proof.
///
/// `snarkrs-msm` is a dev-dependency here and not a runtime one, so the comparison against it is
/// a `const _` in `tests/msm_digits.rs` rather than in this file.
pub const RECODE_BITS: u32 = 255;

/// Caps the bucket array at 2^15 rows per window.
///
/// Also the width [`crate::gen::msm`]'s `sc_bits` is correct up to: it masks with
/// `(1u << width) - 1u`, and a width of 32 would be a shift by the full word width, which
/// WGSL leaves indeterminate.
pub const MAX_WINDOW: u32 = 16;

/// Whether an MSM's cost may depend on the scalar values. The wgpu twin of `snarkrs_msm`'s
/// and `snarkrs_metal::msm::Work`, decided per [`crate::batch::Group`] and inherited by the
/// plans built over it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Work {
    /// No scalar is classified or skipped: every one of the `n` scalars emits one entry per
    /// window, a zero digit into one of [`DUMMY_ROWS`] rows behind the real buckets that
    /// nothing reads, and the window is priced for `n` scalars. The ones pass is off and
    /// the merge is the fixed tree of `msm_fold_*` rather than a walk per bucket. The
    /// window width, every buffer, every dispatch and the number of mixed additions then
    /// follow the key (`n`, and which bases are at infinity) and not the witness.
    ///
    /// That closes the channel of "Remote Side-Channel Attacks on Anonymous Transactions"
    /// (USENIX Security 2020), which recovered Zcash witness sparsity from proving time.
    /// The accumulation is `msm_segmented_constant_*`, which decides nothing by branching
    /// on an entry (a run boundary, a padded iteration, a base at infinity and a digit's
    /// sign are all selects) and adds with the complete formulas
    /// (`gen::points::proj_ops`); the merge is the fixed tree of `msm_fold_*`, written the
    /// same way; and the reduce adds with the complete formulas too. So every thread's
    /// instruction stream, loads, stores and field products are the key's, and an empty
    /// bucket or slot costs what a full one does. It is not constant time in the strict
    /// sense: which bucket a digit names still follows its value, so the digit pipeline's
    /// atomics contend by occupancy and the accumulation's and fold's stores land on
    /// witness-chosen addresses, as in every Pippenger over a bucket layout; see
    /// `crate::backend::WgpuProver::constant_work` for what that leaves and measures.
    /// Stages 0 to 4 were fixed-shape already: the gather multiplies every term and the
    /// field prelude reduces with `select`.
    ///
    /// What it costs is what the zeros and ones were saving, so it is priced per circuit
    /// on `crate::backend::WgpuProver::constant_work`.
    Constant,
    /// Zero scalars are dropped, one scalars go to the ones pass or, where the host counted
    /// few of them, to the buckets ([`DigitPlan::ones_in_buckets`]), the window is sized for
    /// the rest. Running time, buffer sizes and the dispatch geometry then follow how many
    /// witness entries are zero or one.
    #[default]
    Variable,
}

/// Dummy rows per window under [`Work::Constant`]. A power of two, so a scalar's row is
/// its index masked; several rather than one only so the zero digits' atomics spread over
/// that many addresses instead of serialising on one. `snarkrs-metal` uses the same 64.
pub const DUMMY_ROWS: u32 = 64;
const _: () = assert!(DUMMY_ROWS.is_power_of_two());

/// Forces the window width, or 0 to compute it. The browser's counterpart to
/// `G16_WGPU_MSM_C`, which cannot work there: `std::env::var` on wasm always fails, so a
/// page has no way to reach the environment variable at all. Set through `set_msm_c` in the
/// wasm bindings, and read by [`window_size_for`] below.
pub static WINDOW_OVERRIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The MSMs one counting sort feeds, by group: its window width is theirs to share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jobs {
    pub g1: u32,
    pub g2: u32,
}

impl Jobs {
    /// One MSM over G1: `L`, `H`, and what a caller who does not say gets.
    pub const G1: Self = Self { g1: 1, g2: 0 };
}

/// [`window_size_for`] for one MSM over G1.
pub fn window_size(m: usize) -> u32 {
    window_size_for(m, Jobs::G1)
}

/// Window width for `m` scalars that actually reach the buckets of `jobs`.
///
/// The width that minimises [`crate::points::stage_us`] summed over the jobs, which prices
/// each job's accumulation at the slice length it will run at, its merge and its reduce in
/// serial chain steps of its own curve. A G2 step is 3.5 G1 steps, so a witness plan, whose
/// one sort feeds A and B in G1 and B in G2, is chosen mostly by its G2 job.
///
/// ```text
/// general scalars     one G1 job    A, B in G2 and B in G1
/// 256 to 1,024             5                 5
/// 2,048 to 32,768          8                 8
/// 49,152 to 262,144       13                13
/// 393,216                 13                16
/// 524,288 and up          16                16
/// ```
///
/// The whole stage in Chrome 154 on an M2 Max with every plan forced to one width, each
/// kernel in a timed pass of its own, milliseconds, the best of slices 16 to 128 at each
/// width and this function's pick for the witness plan starred:
///
/// ```text
/// circuit (witness, H)                c = 5      8     13     16
/// js_1x1_d8    (3,234, 4,096)          38.3   34.4*  65.0  162.5
/// js_2x2_d16   (9,906, 16,384)         68.8   49.2*  74.7  180.9
/// js_2x2_d32   (17,682, 32,768)               69.4*  87.1
/// js_8x8_d32   (69,372, 131,072)      303.9  189.5  163.5* 253.1
/// js_16x16_d32 (138,292, 262,144)            335.7  249.8* 322.5
/// ```
///
/// # What it replaced
///
/// A least-squares fit to a whole-G1-MSM sweep in native wgpu: 6.61 ns a mixed addition,
/// 0.16 us a bucket row, 164 us a merge iteration on the busiest bucket at slice 128. It
/// priced the accumulation by its entries alone and every bucket row alike, where the
/// reduce is 6.5 ms over the 81,920 rows of c = 13 and 18.9 over the 524,288 of c = 16, so
/// it moved from c = 8 to 13 at 139,700 general scalars: `js_8x8_d32` ran at 8, and
/// `js_16x16_d32`'s witness (138,292) sat just under the line with a G2 accumulation of
/// 129 ms at 8. It also weighed every plan as one G1 job.
///
/// # Why this is not the CPU's cost model
///
/// The CPU minimises `W * (m + 3 * 2^(c-1))`. On a GPU that picks the wrong `c` and the
/// Metal measurements say so loudly: on `js_8x8_d32`'s A MSM, forcing each width in turn gave
/// c=11 23.9 ms, c=12 16.3, c=13 9.5, c=14 21.4, c=15 20.0. A 2.3x swing between neighbours
/// is not a smooth curve with a shallow optimum, and the CPU model has no term for the thing
/// causing it.
///
/// The cause is the top window. `W * c` overshoots [`RECODE_BITS`] by up to `c - 1` bits, so
/// the top window has only `RECODE_BITS - (W-1)*c` meaningful bits and its digits crowd into
/// `2^(top_bits - 1)` buckets instead of `2^(c-1)`. At c=11 that is two buckets holding half
/// the scalars each. On the CPU a serial bucket loop only cares about the total; on the GPU
/// those runs span thousands of slices and the merge walks one in a single lane.
///
/// It is still a fit to one device, and `G16_WGPU_MSM_C` forces a width.
pub fn window_size_for(m: usize, jobs: Jobs) -> u32 {
    let forced = WINDOW_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if forced > 0 {
        return forced.clamp(2, MAX_WINDOW);
    }
    if let Ok(v) = std::env::var("G16_WGPU_MSM_C") {
        if let Ok(c) = v.parse::<u32>() {
            return c.clamp(2, MAX_WINDOW);
        }
    }
    let mut best = 3;
    let mut best_cost = f64::MAX;
    for c in 3..=MAX_WINDOW {
        let cost = f64::from(jobs.g1) * crate::points::stage_us(crate::gen::points::G1, m, c)
            + f64::from(jobs.g2) * crate::points::stage_us(crate::gen::points::G2, m, c);
        if cost < best_cost {
            best_cost = cost;
            best = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// The parameter block
// ---------------------------------------------------------------------------

/// Mirrors `struct MsmParams` in [`crate::gen::msm`]. 80 bytes, a multiple of the 16 WGSL
/// rounds every uniform struct up to, so there is no invisible tail padding for the two
/// declarations to disagree about.
///
/// The nine fields before the last three are the point stages', and nothing in this file
/// reads them but `dummy_rows`. They live here rather than in a second struct because
/// `snarkrs-metal` has one `MsmParams` covering both halves of an MSM, because design §3 puts
/// every parameter block for a proof in one uniform ring, and because a second struct is a
/// second host mirror to keep in step. See [`crate::points::PointPlan`], which fills them.
///
/// `ParamRing::push` cannot check the correspondence and nothing else will either: a
/// mismatch reads plausible garbage with no validation error anywhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MsmParams {
    /// Elements this kernel's domain holds.
    pub n: u32,
    /// Window width in bits.
    pub c: u32,
    /// `ceil(255 / c)`.
    pub n_windows: u32,
    /// `2^(c-1)`.
    pub n_buckets: u32,
    /// Entries reserved per window.
    pub cap: u32,
    /// Element offset into the scalar buffer.
    pub scalar_off: u32,
    /// First element of this dispatch.
    pub lo: u32,
    /// Element offset into the base vector. Zero for every digit kernel.
    pub base_off: u32,
    /// Workgroups in `msm_ones_*`. `msm_count` and `msm_scatter` read only whether it is
    /// zero, which is where a scalar of 1 stays in the sort: a [`DigitPlan`] writes 1 for a
    /// plan with a ones pass.
    pub ones_groups: u32,
    /// Entries one thread of `msm_segmented_*` owns. Zero for every digit kernel.
    pub slice_len: u32,
    /// `ceil(cap / slice_len)`. Zero for every digit kernel.
    pub slices: u32,
    /// [`DUMMY_ROWS`] under [`Work::Constant`], 0 otherwise. Read by the digit kernels,
    /// which skip nothing when it is set, and by the segmented pass.
    pub dummy_rows: u32,
    /// The first dummy row: the real rows of every window, `n_windows * n_buckets` over
    /// the plan's own window count. Carried rather than recomputed because a window slab
    /// lowers `n_windows` to its end and the segmented pass reads this inside one.
    pub dummy_base: u32,
    /// Spill slots per window this `msm_fold_*` level reads. Zero for every other kernel.
    pub fold_in: u32,
    /// Groups per window this level writes; 1 on the last level, which writes every run
    /// to its bucket.
    pub fold_groups: u32,
    /// Slots one group folds.
    pub fold_len: u32,
    /// Workgroups `msm_reduce_*` runs per window, each over a contiguous chunk of its
    /// buckets, writing one partial per group. Zero for every other kernel.
    pub reduce_groups: u32,
    /// 1 when the sort reads its scalars through a live index, 0 otherwise. Read by
    /// `msm_count` and `msm_scatter` only, like the two below; see [`LiveRun`].
    pub live: u32,
    /// The first word of this sort's run in the index.
    pub live_off: u32,
    /// The position this sort's range starts at, which the index counts from the group's.
    pub live_lo: u32,
}

const _: () = assert!(core::mem::size_of::<MsmParams>() == 80);

impl MsmParams {
    /// The block for a kernel that reads only `n` and `lo` (`zero_u32`, `fr_mont_to_std`),
    /// so no plan is needed to fill the rest.
    pub fn range(n: u32, lo: u32) -> Self {
        Self {
            n,
            lo,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// One digit pipeline's shape
// ---------------------------------------------------------------------------

/// The sizes one counting sort runs at: how many scalars, how wide the windows, how much
/// room the entry array needs.
///
/// Derived once and then read by every dispatch and every allocation, so the shader's view
/// and the host's cannot disagree about `cap` in particular, which is the one number whose
/// being too small corrupts silently.
/// The run of a live index one counting sort reads its scalars through, in place of every
/// position of its range: thread `i` of `msm_count` and `msm_scatter` takes the scalar at
/// the position word `off + i` of the index names. See `crate::batch::Live`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRun {
    /// The run's first word in the index.
    pub off: u32,
    /// The position the sort's range starts at, counted like the index's own positions from
    /// the group's first scalar.
    pub lo: u32,
    /// Positions the range holds, live or not, which the ones pass and every bounds check
    /// still walk.
    pub span: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigitPlan {
    n: u32,
    scalar_off: u32,
    c: u32,
    n_windows: u32,
    n_buckets: u32,
    cap: u32,
    /// [`DUMMY_ROWS`] under [`Work::Constant`], 0 otherwise. The dummy rows follow the real
    /// bucket rows in the counters, the cursors and the bucket array, and only the digit
    /// kernels and the accumulation ever address them.
    dummy_rows: u32,
    /// The live index run the sort reads through, and then `n` is its length.
    live: Option<LiveRun>,
    /// Whether a scalar of 1 leaves the sort for `msm_ones_*`. Never under
    /// [`Work::Constant`]; see [`Self::ones_in_buckets`].
    ones_pass: bool,
}

impl DigitPlan {
    /// Plans a counting sort over `n` scalars starting at `scalar_off`.
    ///
    /// `general` is how many of those `n` are neither 0 nor 1. `None` means the caller could
    /// not classify them, which is the case for a device-resident buffer such as stage 4's
    /// `h_std`, and then every scalar is assumed general. **Overestimating is safe and
    /// underestimating is not**: `cap` bounds the scatter's write into each window's slice of
    /// the entry array, and WebGPU drops an out-of-range storage write in silence, so a `cap`
    /// below the true general count loses entries with no error anywhere. That is why this
    /// takes a count rather than a hint, and why `None` means `n` rather than a guess.
    pub fn new(n: u32, scalar_off: u32, general: Option<u32>) -> Result<Self, ProveError> {
        Self::with_work(n, scalar_off, general, Work::Variable)
    }

    /// Same, at a forced window width. For the acceptance sweep over several `c`, and for
    /// tests that need a specific one without going through `G16_WGPU_MSM_C`.
    pub fn with_c(
        n: u32,
        scalar_off: u32,
        general: Option<u32>,
        c: u32,
    ) -> Result<Self, ProveError> {
        Self::with_work_c(n, scalar_off, general, c, Work::Variable)
    }

    /// [`Self::new`] with the work mode chosen. Under [`Work::Constant`] `general` is
    /// ignored: the plan prices `n` scalars whatever the caller knows about them, so
    /// nothing in its shape depends on a value.
    pub fn with_work(
        n: u32,
        scalar_off: u32,
        general: Option<u32>,
        work: Work,
    ) -> Result<Self, ProveError> {
        Self::for_jobs(n, scalar_off, general, work, Jobs::G1)
    }

    /// [`Self::with_work`] for a sort that feeds `jobs`, which is what the window width is
    /// chosen for; see [`window_size_for`].
    pub fn for_jobs(
        n: u32,
        scalar_off: u32,
        general: Option<u32>,
        work: Work,
        jobs: Jobs,
    ) -> Result<Self, ProveError> {
        let general = Self::classified(n, general, work)?;
        let c = window_size_for(general as usize, jobs);
        Self::at(n, scalar_off, general, c, work)
    }

    /// [`Self::with_c`] with the work mode chosen.
    pub fn with_work_c(
        n: u32,
        scalar_off: u32,
        general: Option<u32>,
        c: u32,
        work: Work,
    ) -> Result<Self, ProveError> {
        Self::at(n, scalar_off, Self::classified(n, general, work)?, c, work)
    }

    fn classified(n: u32, general: Option<u32>, work: Work) -> Result<u32, ProveError> {
        match general {
            Some(g) if g > n => Err(bad(format!(
                "digit plan over {n} scalars was told {g} of them are general"
            ))),
            Some(g) if work == Work::Variable => Ok(g),
            _ => Ok(n),
        }
    }

    /// The one place the shape is derived from `c`.
    ///
    /// Both constructors funnel through here rather than each computing `n_windows` and
    /// `n_buckets` for itself. That is not tidiness: the first version of this file did
    /// duplicate the two lines, and a mutation test that replaced `div_ceil` with `/` in one
    /// copy went **undetected**, because every test that fixes `c` went through the other
    /// copy. A derivation written twice is a derivation that can be wrong once.
    fn at(n: u32, scalar_off: u32, general: u32, c: u32, work: Work) -> Result<Self, ProveError> {
        if !(2..=MAX_WINDOW).contains(&c) {
            return Err(bad(format!(
                "window width {c} is outside 2..={MAX_WINDOW}; sc_bits masks with \
                 (1 << c) - 1 and a shift by the word width is indeterminate in WGSL"
            )));
        }
        let n_windows = RECODE_BITS.div_ceil(c);
        // Only general scalars emit entries, and each emits at most one per window, so this
        // bounds the scatter exactly. At least 1, because a zero-length WebGPU buffer is not
        // a thing.
        let cap = general.max(1);
        // `n_windows * cap` is the entry array's length and it is what the scatter's `slot`
        // indexes, so it has to fit in a u32 before anything downstream can check it against
        // the storage binding limit. At c = 2 that is 128 windows, and 128 * 2^25 already
        // overflows. Nothing reaches it today because the cost model never picks a small `c`
        // for a large `m`, but `with_c` lets a caller force one.
        if u64::from(n_windows) * u64::from(cap) > u64::from(u32::MAX) {
            return Err(bad(format!(
                "{n_windows} windows x cap {cap} overflows a u32 entry index; c = {c} is too \
                 narrow for {general} general scalars"
            )));
        }
        Ok(Self {
            n,
            scalar_off,
            c,
            n_windows,
            n_buckets: 1u32 << (c - 1),
            cap,
            dummy_rows: match work {
                Work::Constant => DUMMY_ROWS,
                Work::Variable => 0,
            },
            live: None,
            ones_pass: work == Work::Variable,
        })
    }

    /// The same sort with a scalar of 1 recoded like any other, into bucket (0, 0), and no
    /// ones pass over any job of it. The caller's `general` count must then have held the
    /// ones as well, because each takes an entry of window 0.
    ///
    /// For a range whose ones are few: the pass costs a job its workgroup tree whatever it
    /// finds, 0.7 ms over G1 and 1.9 over G2 on `js_2x2_d32`'s 30 ones, and the bucket
    /// they share costs the merge little until it is well past the plan's fattest; see
    /// [`Self::ones_fit`].
    pub fn ones_in_buckets(mut self) -> Self {
        self.ones_pass = false;
        self
    }

    /// Whether `ones` scalars of 1 are few enough to share bucket (0, 0): no more than one
    /// and a half times what the plan's fattest bucket holds. That is a bucket of the top
    /// window, whose `RECODE_BITS - (W - 1) * c` bits spread the plan's entries over
    /// `2^(top_bits - 1)` buckets, and the merge walks it in one lane
    /// (`crate::points::stage_us`); `cap` here counts the ones.
    ///
    /// Under it the ones lengthen no chain. The half over is two merge steps a slice
    /// against the eight of the pass they replace, four to nine steps at the slice
    /// lengths `crate::points::slice_len_for` gives, so about even at the line.
    /// `railgun-13x01`'s B sort (335 a bucket and 439 ones) is the one artifact
    /// between the two: 188.4 and 189.5 ms with its ones sorted against 192.0 and 196.3
    /// with the pass, lower quartiles 186.5 and 187.3 against 189.7 and 189.5.
    pub fn ones_fit(&self, ones: u32) -> bool {
        let top_bits = RECODE_BITS - (self.n_windows - 1) * self.c;
        let top_buckets = 1u64 << (top_bits.min(self.c) - 1);
        2 * u64::from(ones) * top_buckets <= 3 * u64::from(self.cap)
    }

    /// The same sort over `run` of a live index: `n` is the run's length, `scalar_off` the
    /// first position of the range it lies in.
    pub fn over(mut self, run: LiveRun) -> Self {
        self.live = Some(run);
        self
    }

    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn live(&self) -> Option<LiveRun> {
        self.live
    }
    /// Positions of the scalar and base vectors the sort's range holds: `n`, or the live
    /// run's span.
    pub fn span(&self) -> u32 {
        self.live.map_or(self.n, |run| run.span)
    }
    pub fn scalar_off(&self) -> u32 {
        self.scalar_off
    }
    pub fn c(&self) -> u32 {
        self.c
    }
    pub fn n_windows(&self) -> u32 {
        self.n_windows
    }
    pub fn n_buckets(&self) -> u32 {
        self.n_buckets
    }
    pub fn cap(&self) -> u32 {
        self.cap
    }
    pub fn dummy_rows(&self) -> u32 {
        self.dummy_rows
    }
    /// Whether the jobs over this sort run `msm_ones_*`.
    pub fn ones_pass(&self) -> bool {
        self.ones_pass
    }
    pub fn work(&self) -> Work {
        if self.dummy_rows == 0 {
            Work::Variable
        } else {
            Work::Constant
        }
    }

    /// Counters, cursors and bucket rows: the real buckets of every window, then every
    /// window's dummy rows. `n_windows * n_buckets` under [`Work::Variable`].
    pub fn rows(&self) -> u32 {
        self.n_windows * (self.n_buckets + self.dummy_rows)
    }

    /// Slots in the entry array: `n_windows * cap`.
    pub fn entries(&self) -> u32 {
        self.n_windows * self.cap
    }

    /// The largest storage binding [`DigitBuffers::new`] would make for this plan, in bytes,
    /// which is what decides whether a sub-MSM of this shape fits a device at all. The entry
    /// array, at every shape any key reaches: 20 windows of 8-byte entries per general
    /// scalar at c = 13, so 4.19 million of them fill a 128 MiB binding twice over.
    pub fn largest_binding(&self) -> u64 {
        (u64::from(self.rows()) * 4).max(u64::from(self.entries()) * ENTRY_BYTES)
    }

    /// The parameter block the digit kernels read, with the point-stage fields left at
    /// zero but `ones_groups`, which is 1 where there is a ones pass.
    /// [`crate::points::PointPlan::params`] is the one that fills them.
    pub fn params(&self, n: u32, lo: u32) -> MsmParams {
        MsmParams {
            n,
            c: self.c,
            n_windows: self.n_windows,
            n_buckets: self.n_buckets,
            cap: self.cap,
            scalar_off: self.scalar_off,
            lo,
            base_off: 0,
            ones_groups: u32::from(self.ones_pass),
            slice_len: 0,
            slices: 0,
            dummy_rows: self.dummy_rows,
            dummy_base: self.n_windows * self.n_buckets,
            fold_in: 0,
            fold_groups: 0,
            fold_len: 0,
            reduce_groups: 0,
            live: u32::from(self.live.is_some()),
            live_off: self.live.map_or(0, |run| run.off),
            live_lo: self.live.map_or(0, |run| run.lo),
        }
    }
}

/// The three scratch buffers a digit pipeline fills.
///
/// Allocated together because their sizes all come from one [`DigitPlan`] and getting one of
/// the three wrong is not a validation error, it is a silently truncated sort.
pub struct DigitBuffers {
    pub counts: wgpu::Buffer,
    pub cursor: wgpu::Buffer,
    pub entries: wgpu::Buffer,
}

impl DigitBuffers {
    /// Allocates for `plan`, plus `slack` extra elements on each buffer.
    ///
    /// `slack` exists for one reason and it is a test: every output here is exactly the size
    /// the kernel should write, so a kernel that writes one element too far lands out of
    /// range and WebGPU drops it in silence. U7 shipped two such bugs into a review because
    /// of exactly that. `tests/msm_digits.rs` allocates slack and checks it survives; the
    /// prover passes 0.
    pub fn new(backend: &WgpuBackend, plan: &DigitPlan, slack: u32) -> Result<Self, ProveError> {
        let rows = u64::from(plan.rows() + slack);
        let entries = u64::from(plan.entries() + slack);
        Ok(Self {
            counts: storage_buffer(backend, "g16 msm counts", rows * 4)?,
            cursor: storage_buffer(backend, "g16 msm cursor", rows * 4)?,
            entries: storage_buffer(backend, "g16 msm entries", entries * ENTRY_BYTES)?,
        })
    }

    /// Total device bytes, for a report.
    pub fn bytes(&self) -> u64 {
        self.counts.size() + self.cursor.size() + self.entries.size()
    }
}

pub(crate) fn storage_buffer(
    backend: &WgpuBackend,
    label: &str,
    bytes: u64,
) -> Result<wgpu::Buffer, ProveError> {
    let limits = backend.granted_limits();
    if bytes == 0 {
        return Err(bad(format!("{label}: a zero length buffer is not legal")));
    }
    if bytes > limits.max_storage_buffer_binding_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes, over the {} byte storage binding limit. Lower the \
             window width or cut the input range; `crate::batch` cuts it before it gets here.",
            limits.max_storage_buffer_binding_size
        )));
    }
    Ok(backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

// ---------------------------------------------------------------------------
// The pipelines
// ---------------------------------------------------------------------------

/// How the five entry points are split across shader modules.
///
/// # This is a measurement, and it inverts what the design assumes
///
/// The design and `crate::pipelines`' module docs both commit to **many small modules**,
/// because an earlier measurement recorded 129 s of pipeline creation for
/// one monolithic WGSL shader. On this platform, for these five kernels, the opposite is
/// true. Cold means Metal's on-disk function cache misses, which needs a genuinely new MSL
/// function name and not a nonce comment, because naga strips comments before Metal ever sees
/// the source; `tests/msm_digits.rs::the_digit_modules_are_fused_for_a_measured_reason`
/// renames every entry point per run to force that. Three runs, M2 Max, naga to MSL,
/// milliseconds:
///
/// ```text
/// shape                          KiB   naga  pipelines  total
/// cold: digits, no prelude      10.5    0.3       83.2   83.5
/// cold: fr_mont_to_std          30.2    2.3      119.6  121.9
/// cold: the two as Split        40.7    2.7      202.8  205.4
/// cold: all five as Fused       39.7    2.5      169.6  172.2
/// warm: either shape            40.7    2.5        0.9    3.4
/// ```
///
/// **[`ModuleShape::Fused`] is 17% cheaper cold, by about 34 ms, and identical warm.** Two
/// things follow and both are worth carrying forward.
///
/// First, `zero_u32`, `msm_count`, `msm_scan` and `msm_scatter` contain no field arithmetic
/// at all, and they still cost 83 ms of Metal pipeline creation between them, about 21 ms
/// each. That is a per-entry-point floor, not a function of module size, and it is most of
/// what a cold `prepare` pays.
///
/// Second, and this settles a `TASKS.local.md` item U7 filed: Metal compiles a pipeline per entry
/// point and dead-strips the module around it, so **carrying a prelude a kernel never calls
/// costs naga time and no pipeline time**. U7's note that stage 0's module carries 30 KiB of
/// `Fq` and `Fq2` it never calls, with an "unmeasured saving", has now been measured
/// indirectly here: the saving is about 2 ms of naga per module and nothing else.
///
/// Both shapes ship. `the_digit_modules_are_fused_for_a_measured_reason` compiles the two
/// arrangements through [`crate::gen::msm`] and asserts the default is the faster of them
/// rather than pinning today's answer, so if Tint reverses it at U14 it fails loudly;
/// `the_split_module_shape_runs_the_same_sort_and_the_same_conversion` is what runs
/// [`ModuleShape::Split`] through [`MsmDigits`] itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModuleShape {
    /// All five entry points behind one copy of the `Fr` prelude. The default, on the table
    /// above.
    #[default]
    Fused,
    /// The four scalar-only kernels in one module and `fr_mont_to_std` behind the prelude in
    /// another. What design §4's "one module per group of kernels" asks for, kept selectable
    /// because it is the thing the default is measured against.
    Split,
}

/// The five digit-pipeline entry points, their bind group layouts and their pipelines.
///
/// One shader module by default, five pipeline layouts, one bind group layout per entry
/// point. The module count is [`ModuleShape`], which carries the measurement; the layout
/// count is not negotiable, because giving all five the union of their bindings would make
/// the storage-buffer count meaningless as a check against the browser floor's 8.
pub struct MsmDigits {
    /// One [`Kernels`] under [`ModuleShape::Fused`], two under [`ModuleShape::Split`]. Looked
    /// up in order by [`Self::pipeline`], so the rest of the file does not branch on the
    /// shape at all.
    modules: Vec<Kernels>,
    shape: ModuleShape,
    bgl_zero: wgpu::BindGroupLayout,
    bgl_count: wgpu::BindGroupLayout,
    bgl_scan: wgpu::BindGroupLayout,
    bgl_scatter: wgpu::BindGroupLayout,
    bgl_mont: wgpu::BindGroupLayout,
    _layouts: Vec<wgpu::PipelineLayout>,
    wg: wgsl::Workgroups,
    workgroups_per_dispatch: u32,
    source_len: usize,
    /// One word bound as the live index of a sort that reads none, because a binding the
    /// kernel names has to be bound.
    no_live: wgpu::Buffer,
}

impl MsmDigits {
    /// Compiles at the measured shape and this device's workgroup-per-dimension limit.
    pub fn new(backend: &WgpuBackend) -> Result<Self, ProveError> {
        let max_wg = backend
            .granted_limits()
            .max_compute_workgroups_per_dimension;
        Self::with_shape(
            backend,
            wgsl::Workgroups::default(),
            wgsl::LimbPick::default(),
            max_wg,
        )
    }

    /// Same, with the workgroup sizes, the limb-pick strategy and the per-dispatch workgroup
    /// cap forced, at the default [`ModuleShape`].
    ///
    /// Public for the tests that cannot be written otherwise: the workgroup sweep and the
    /// limb-pick sweep that made those constants measured numbers, and the multi-dispatch
    /// path, which no artifact reaches because a single dispatch covers 8.4M scalars at 128
    /// threads. A code path no test can reach is a code path that ships untested.
    pub fn with_shape(
        backend: &WgpuBackend,
        wg: wgsl::Workgroups,
        pick: wgsl::LimbPick,
        workgroups_per_dispatch: u32,
    ) -> Result<Self, ProveError> {
        Self::with_module_shape(
            backend,
            wg,
            pick,
            workgroups_per_dispatch,
            ModuleShape::default(),
        )
    }

    /// Same, choosing how the entry points are split across modules.
    pub fn with_module_shape(
        backend: &WgpuBackend,
        wg: wgsl::Workgroups,
        pick: wgsl::LimbPick,
        workgroups_per_dispatch: u32,
        shape: ModuleShape,
    ) -> Result<Self, ProveError> {
        let limits = backend.granted_limits();
        // The browser floor, not the granted limit; see `WgpuBackend::ceiling_invocations`.
        let max_inv = backend.ceiling_invocations();
        for (name, n) in [
            ("zero", wg.zero),
            ("mont", wg.mont),
            ("count", wg.count),
            ("scan", wg.scan),
            ("scatter", wg.scatter),
        ] {
            if n == 0 || n > max_inv {
                return Err(bad(format!(
                    "{name} workgroup size {n} is outside 1..={max_inv}"
                )));
            }
        }
        let max_wg = limits.max_compute_workgroups_per_dimension;
        if workgroups_per_dispatch == 0 || workgroups_per_dispatch > max_wg {
            return Err(bad(format!(
                "workgroups_per_dispatch {workgroups_per_dispatch} is outside 1..={max_wg}"
            )));
        }
        // msm_scan's workgroup array is `array<u32, scan>`; the floor allows 16384 bytes and
        // 256 threads, so this cannot bind. The check is here rather than left to a pipeline
        // creation error that names bytes instead of the constant that set them.
        let scan_bytes = u64::from(wg.scan) * 4;
        if scan_bytes > backend.ceiling_workgroup_bytes() {
            return Err(bad(format!(
                "msm_scan's workgroup array is {scan_bytes} bytes at {} threads, over the {} \
                 byte ceiling (the smaller of this device's {} and the browser floor's {})",
                wg.scan,
                backend.ceiling_workgroup_bytes(),
                limits.max_compute_workgroup_storage_size,
                crate::gen::FLOOR_WORKGROUP_BYTES,
            )));
        }

        let device = backend.device();
        let mk_bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let bgl_zero = mk_bgl("g16 zero_u32", &Self::zero_entries());
        let bgl_count = mk_bgl("g16 msm_count", &Self::count_entries());
        let bgl_scan = mk_bgl("g16 msm_scan", &Self::scan_entries());
        let bgl_scatter = mk_bgl("g16 msm_scatter", &Self::scatter_entries());
        let bgl_mont = mk_bgl("g16 fr_mont_to_std", &Self::mont_entries());

        let mk_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        // One pipeline layout per entry point, exactly as `gen::ntt`'s split-by-mode does it.
        // Giving all five the union of their bindings would put every kernel at the widest
        // one's set and make the storage-buffer count meaningless as a check against the
        // floor's 8.
        let l_zero = mk_layout("g16 zero_u32", &bgl_zero);
        let l_count = mk_layout("g16 msm_count", &bgl_count);
        let l_scan = mk_layout("g16 msm_scan", &bgl_scan);
        let l_scatter = mk_layout("g16 msm_scatter", &bgl_scatter);
        let l_mont = mk_layout("g16 fr_mont_to_std", &bgl_mont);

        let v = Variant::default();
        let (modules, source_len) = match shape {
            ModuleShape::Fused => {
                let src = wgsl::fused_module_at(v, wg, pick);
                let len = src.len();
                let k = Kernels::build_with_layouts(
                    backend,
                    "msm_digits",
                    &src,
                    &[
                        (wgsl::ENTRY_ZERO, &l_zero),
                        (wgsl::ENTRY_MONT, &l_mont),
                        (wgsl::ENTRY_COUNT, &l_count),
                        (wgsl::ENTRY_SCAN, &l_scan),
                        (wgsl::ENTRY_SCATTER, &l_scatter),
                    ],
                )?;
                (vec![k], len)
            }
            ModuleShape::Split => {
                let dsrc = wgsl::digits_module_at(wg, pick);
                let msrc = wgsl::mont_module_at(v, wg.mont);
                let len = dsrc.len() + msrc.len();
                let d = Kernels::build_with_layouts(
                    backend,
                    "msm_digits",
                    &dsrc,
                    &[
                        (wgsl::ENTRY_ZERO, &l_zero),
                        (wgsl::ENTRY_COUNT, &l_count),
                        (wgsl::ENTRY_SCAN, &l_scan),
                        (wgsl::ENTRY_SCATTER, &l_scatter),
                    ],
                )?;
                let m = Kernels::build(
                    backend,
                    "fr_mont_to_std",
                    &msrc,
                    &l_mont,
                    &[wgsl::ENTRY_MONT],
                )?;
                (vec![d, m], len)
            }
        };

        Ok(Self {
            modules,
            shape,
            bgl_zero,
            bgl_count,
            bgl_scan,
            bgl_scatter,
            bgl_mont,
            _layouts: vec![l_zero, l_count, l_scan, l_scatter, l_mont],
            wg,
            workgroups_per_dispatch,
            source_len,
            no_live: storage_buffer(backend, "g16 msm no live index", 4)?,
        })
    }

    /// The pipeline for one entry point, wherever it landed.
    fn pipeline(&self, entry: &str) -> Result<&wgpu::ComputePipeline, ProveError> {
        for m in &self.modules {
            if let Ok(p) = m.get(entry) {
                return Ok(p);
            }
        }
        Err(bad(format!(
            "no module in this MsmDigits has an entry point {entry:?}"
        )))
    }

    // ---- bind group layouts, as data, so a test can count what the device enforces ----

    pub fn zero_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_ZERO_BUF, false),
        ]
    }
    pub fn count_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_SCALARS, true),
            storage_entry(wgsl::BIND_COUNTS_ATOMIC, false),
            storage_entry(wgsl::BIND_LIVE, true),
        ]
    }
    pub fn scan_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_COUNTS_READ, true),
            storage_entry(wgsl::BIND_CURSOR_WRITE, false),
        ]
    }
    pub fn scatter_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_SCALARS, true),
            storage_entry(wgsl::BIND_CURSOR_ATOMIC, false),
            storage_entry(wgsl::BIND_ENTRIES, false),
            storage_entry(wgsl::BIND_LIVE, true),
        ]
    }
    pub fn mont_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_MONT_SRC, true),
            storage_entry(wgsl::BIND_MONT_DST, false),
        ]
    }

    /// Storage buffers each entry point's pipeline layout declares, counted from the lists
    /// above rather than from a duplicate. All five must be at most 8, which is the floor.
    pub fn storage_buffer_counts() -> [(&'static str, u32); 5] {
        fn count(entries: &[wgpu::BindGroupLayoutEntry]) -> u32 {
            entries
                .iter()
                .filter(|e| {
                    matches!(
                        e.ty,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { .. },
                            ..
                        }
                    )
                })
                .count() as u32
        }
        [
            (wgsl::ENTRY_ZERO, count(&Self::zero_entries())),
            (wgsl::ENTRY_MONT, count(&Self::mont_entries())),
            (wgsl::ENTRY_COUNT, count(&Self::count_entries())),
            (wgsl::ENTRY_SCAN, count(&Self::scan_entries())),
            (wgsl::ENTRY_SCATTER, count(&Self::scatter_entries())),
        ]
    }

    // ---- bind groups ----

    fn bind(
        &self,
        backend: &WgpuBackend,
        label: &str,
        bgl: &wgpu::BindGroupLayout,
        ring: &ParamRing,
        bufs: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let mut entries = Vec::with_capacity(bufs.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: wgsl::BIND_PARAMS,
            resource: wgpu::BindingResource::Buffer(ring.binding()),
        });
        for &(binding, buf) in bufs {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            });
        }
        backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bgl,
                entries: &entries,
            })
    }

    /// `zero_u32` over any `u32` buffer.
    pub fn bind_zero(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.bind(
            backend,
            "g16 zero_u32",
            &self.bgl_zero,
            ring,
            &[(wgsl::BIND_ZERO_BUF, buf)],
        )
    }

    /// `fr_mont_to_std`, checked for length because WebGPU derives a runtime-sized array's
    /// length from the binding size and then returns zero for an out-of-range read and drops
    /// an out-of-range write, so a short destination is a truncated scalar vector and no
    /// error anywhere.
    pub fn bind_mont(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        n: u32,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup, ProveError> {
        let want = u64::from(n) * FR_BYTES;
        for (name, buf) in [("src", src), ("dst", dst)] {
            if buf.size() < want {
                return Err(bad(format!(
                    "fr_mont_to_std {name} is {} bytes, {n} elements need {want}",
                    buf.size()
                )));
            }
        }
        Ok(self.bind(
            backend,
            "g16 fr_mont_to_std",
            &self.bgl_mont,
            ring,
            &[(wgsl::BIND_MONT_SRC, src), (wgsl::BIND_MONT_DST, dst)],
        ))
    }

    /// The three bind groups the counting sort needs, all length checked against `plan`.
    pub fn bind_sort(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        plan: &DigitPlan,
        scalars: &wgpu::Buffer,
        bufs: &DigitBuffers,
    ) -> Result<SortBinds, ProveError> {
        self.bind_sort_over(backend, ring, plan, scalars, None, bufs)
    }

    /// [`Self::bind_sort`] with the live index a plan [`DigitPlan::over`] a run reads
    /// through, which it must then be given.
    pub fn bind_sort_over(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        plan: &DigitPlan,
        scalars: &wgpu::Buffer,
        live: Option<&wgpu::Buffer>,
        bufs: &DigitBuffers,
    ) -> Result<SortBinds, ProveError> {
        let want_scalars = u64::from(plan.scalar_off + plan.span()) * FR_BYTES;
        if scalars.size() < want_scalars {
            return Err(bad(format!(
                "the scalar buffer is {} bytes, {}..{} needs {want_scalars}",
                scalars.size(),
                plan.scalar_off,
                plan.scalar_off + plan.span()
            )));
        }
        let live = match (plan.live, live) {
            (None, None) => &self.no_live,
            (Some(run), Some(buf)) => {
                let want = u64::from(run.off + plan.n) * 4;
                if buf.size() < want {
                    return Err(bad(format!(
                        "the live index is {} bytes, words {}..{} need {want}",
                        buf.size(),
                        run.off,
                        run.off + plan.n
                    )));
                }
                buf
            }
            (plan_has, _) => {
                return Err(bad(format!(
                    "a sort planned {} a live index was bound {} one",
                    if plan_has.is_some() {
                        "over"
                    } else {
                        "without"
                    },
                    if plan_has.is_some() {
                        "without"
                    } else {
                        "with"
                    },
                )))
            }
        };
        let want_rows = u64::from(plan.rows()) * 4;
        for (name, buf) in [("counts", &bufs.counts), ("cursor", &bufs.cursor)] {
            if buf.size() < want_rows {
                return Err(bad(format!(
                    "{name} is {} bytes, {} windows x {} buckets and {} dummy rows need \
                     {want_rows}",
                    buf.size(),
                    plan.n_windows,
                    plan.n_buckets,
                    plan.dummy_rows
                )));
            }
        }
        let want_entries = u64::from(plan.entries()) * ENTRY_BYTES;
        if bufs.entries.size() < want_entries {
            return Err(bad(format!(
                "the entry array is {} bytes, {} windows x cap {} need {want_entries}",
                bufs.entries.size(),
                plan.n_windows,
                plan.cap
            )));
        }
        Ok(SortBinds {
            zero: self.bind_zero(backend, ring, &bufs.counts),
            count: self.bind(
                backend,
                "g16 msm_count",
                &self.bgl_count,
                ring,
                &[
                    (wgsl::BIND_SCALARS, scalars),
                    (wgsl::BIND_COUNTS_ATOMIC, &bufs.counts),
                    (wgsl::BIND_LIVE, live),
                ],
            ),
            scan: self.bind(
                backend,
                "g16 msm_scan",
                &self.bgl_scan,
                ring,
                &[
                    (wgsl::BIND_COUNTS_READ, &bufs.counts),
                    (wgsl::BIND_CURSOR_WRITE, &bufs.cursor),
                ],
            ),
            scatter: self.bind(
                backend,
                "g16 msm_scatter",
                &self.bgl_scatter,
                ring,
                &[
                    (wgsl::BIND_SCALARS, scalars),
                    (wgsl::BIND_CURSOR_ATOMIC, &bufs.cursor),
                    (wgsl::BIND_ENTRIES, &bufs.entries),
                    (wgsl::BIND_LIVE, live),
                ],
            ),
        })
    }

    // ---- parameter blocks ----

    /// Dispatches a 1D kernel at `wg` threads needs to cover `n` elements.
    fn dispatches(&self, n: u32, wg: u32) -> u32 {
        n.div_ceil(wg * self.workgroups_per_dispatch).max(1)
    }

    /// Elements one dispatch of a `wg`-thread kernel covers.
    fn span(&self, wg: u32) -> u32 {
        wg * self.workgroups_per_dispatch
    }

    /// Parameter blocks for the whole counting sort, in dispatch order: `zero_u32` over the
    /// counter rows, `msm_count`, `msm_scan`, `msm_scatter`.
    ///
    /// Separate from [`Self::encode_sort`] because design §3 writes every parameter block for
    /// the whole proof in one `write_buffer` before encoding starts, so the pushes have to
    /// happen before the compute pass exists.
    pub fn plan_sort(
        &self,
        plan: &DigitPlan,
        ring: &mut ParamRing,
    ) -> Result<SortOffsets, ProveError> {
        Ok(SortOffsets {
            zero: self.push_range(ring, plan.rows(), self.wg.zero, |lo| {
                plan.params(plan.rows(), lo)
            })?,
            count: self.push_range(ring, plan.n, self.wg.count, |lo| plan.params(plan.n, lo))?,
            // One workgroup per window, always one dispatch: n_windows is at most 128 at
            // c = 2 and the limit is 65535.
            scan: vec![ring.push(&plan.params(plan.n, 0))?],
            scatter: self
                .push_range(ring, plan.n, self.wg.scatter, |lo| plan.params(plan.n, lo))?,
        })
    }

    /// Parameter blocks for `fr_mont_to_std` over `n` elements.
    pub fn plan_mont(&self, n: u32, ring: &mut ParamRing) -> Result<Vec<u32>, ProveError> {
        self.push_range(ring, n, self.wg.mont, |lo| MsmParams::range(n, lo))
    }

    /// Parameter blocks for `zero_u32` over `n` words of any buffer.
    pub fn plan_zero(&self, n: u32, ring: &mut ParamRing) -> Result<Vec<u32>, ProveError> {
        self.push_range(ring, n, self.wg.zero, |lo| MsmParams::range(n, lo))
    }

    fn push_range(
        &self,
        ring: &mut ParamRing,
        n: u32,
        wg: u32,
        params: impl Fn(u32) -> MsmParams,
    ) -> Result<Vec<u32>, ProveError> {
        let span = self.span(wg);
        let mut offsets = Vec::with_capacity(self.dispatches(n, wg) as usize);
        let mut lo = 0u32;
        loop {
            offsets.push(ring.push(&params(lo))?);
            lo = lo.saturating_add(span);
            if lo >= n {
                break;
            }
        }
        Ok(offsets)
    }

    // ---- encoding ----

    /// The whole counting sort: four dispatch groups into one pass, in order.
    ///
    /// WebGPU orders dispatches inside a compute pass and makes each one's writes visible to
    /// the next with no explicit barrier, which is what lets `zero`, `count`, `scan` and
    /// `scatter` share a pass. Metal needs an explicit `memoryBarrier` for the same thing;
    /// this is one of the few places WGSL is the simpler of the two.
    pub fn encode_sort(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        plan: &DigitPlan,
        binds: &SortBinds,
        offsets: &SortOffsets,
    ) -> Result<(), ProveError> {
        self.encode_zero_rows(pass, plan, binds, offsets)?;
        self.encode_count(pass, plan, binds, offsets)?;
        self.encode_scan(pass, plan, binds, offsets)?;
        self.encode_scatter(pass, plan, binds, offsets)
    }

    /// The four stages separately, so a test can run three of them and look at what the
    /// third wrote.
    ///
    /// `msm_scan` turns the counters into run starts and `msm_scatter` turns those starts
    /// into run ends, so running the whole sort and reading the cursor cannot tell a correct
    /// scan from a wrong one that the scatter happens to undo.
    pub fn encode_zero_rows(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        plan: &DigitPlan,
        binds: &SortBinds,
        offsets: &SortOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            self.pipeline(wgsl::ENTRY_ZERO)?,
            &binds.zero,
            plan.rows(),
            self.wg.zero,
            &offsets.zero,
            wgsl::ENTRY_ZERO,
        )
    }

    pub fn encode_count(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        plan: &DigitPlan,
        binds: &SortBinds,
        offsets: &SortOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            self.pipeline(wgsl::ENTRY_COUNT)?,
            &binds.count,
            plan.n,
            self.wg.count,
            &offsets.count,
            wgsl::ENTRY_COUNT,
        )
    }

    /// One workgroup per window, always one dispatch: `n_windows` is at most 128 at `c = 2`
    /// and `maxComputeWorkgroupsPerDimension` is 65535.
    pub fn encode_scan(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        plan: &DigitPlan,
        binds: &SortBinds,
        offsets: &SortOffsets,
    ) -> Result<(), ProveError> {
        if offsets.scan.len() != 1 {
            return Err(bad(format!(
                "msm_scan takes one parameter block, got {}",
                offsets.scan.len()
            )));
        }
        pass.set_pipeline(self.pipeline(wgsl::ENTRY_SCAN)?);
        pass.set_bind_group(0, &binds.scan, &[offsets.scan[0]]);
        crate::readback::dispatch(pass, plan.n_windows);
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        plan: &DigitPlan,
        binds: &SortBinds,
        offsets: &SortOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            self.pipeline(wgsl::ENTRY_SCATTER)?,
            &binds.scatter,
            plan.n,
            self.wg.scatter,
            &offsets.scatter,
            wgsl::ENTRY_SCATTER,
        )
    }

    /// `fr_mont_to_std` over `n` elements.
    pub fn encode_mont(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let pipeline = self.pipeline(wgsl::ENTRY_MONT)?;
        self.encode_range(
            pass,
            pipeline,
            bind,
            n,
            self.wg.mont,
            offsets,
            "fr_mont_to_std",
        )
    }

    /// `zero_u32` over `n` words.
    pub fn encode_zero(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let pipeline = self.pipeline(wgsl::ENTRY_ZERO)?;
        self.encode_range(pass, pipeline, bind, n, self.wg.zero, offsets, "zero_u32")
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_range(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        n: u32,
        wg: u32,
        offsets: &[u32],
        what: &str,
    ) -> Result<(), ProveError> {
        let want = self.dispatches(n, wg) as usize;
        if offsets.len() != want {
            return Err(bad(format!(
                "{what} over {n} elements needs {want} dispatches, got {} parameter offsets",
                offsets.len()
            )));
        }
        pass.set_pipeline(pipeline);
        let span = self.span(wg);
        let mut lo = 0u32;
        for &off in offsets {
            let hi = lo.saturating_add(span).min(n);
            pass.set_bind_group(0, bind, &[off]);
            crate::readback::dispatch(pass, hi.saturating_sub(lo).div_ceil(wg).max(1));
            lo = hi;
        }
        Ok(())
    }

    // ---- reporting ----

    /// Ring slots `fr_mont_to_std` over `n` elements consumes, which is also its dispatch
    /// count. One at every domain any artifact reaches, because a single dispatch covers
    /// `128 * 65535` elements.
    ///
    /// Separate from [`Self::plan_mont`] because a ring has to be *sized* before anything is
    /// pushed into it, and [`crate::batch::MsmBatch`] sizes one ring for a whole proof.
    pub fn mont_slots(&self, n: u32) -> u32 {
        self.dispatches(n, self.wg.mont)
    }

    /// Parameter ring slots the whole counting sort consumes for `plan`, which is also its
    /// dispatch count because every dispatch takes its own block. Four at every shape any
    /// artifact reaches.
    pub fn sort_slots(&self, plan: &DigitPlan) -> u32 {
        self.dispatches(plan.rows(), self.wg.zero)
            + self.dispatches(plan.n, self.wg.count)
            + 1
            + self.dispatches(plan.n, self.wg.scatter)
    }

    pub fn workgroups(&self) -> wgsl::Workgroups {
        self.wg
    }

    /// Every shader module this kernel set compiled: one under [`ModuleShape::Fused`], two
    /// under [`ModuleShape::Split`]. Public so `crate::selftest::module_diagnostics` can ask
    /// the browser what it thought of them.
    pub fn modules(&self) -> &[Kernels] {
        &self.modules
    }

    pub fn module_shape(&self) -> ModuleShape {
        self.shape
    }

    /// Generated WGSL, in bytes, summed over however many modules this shape uses.
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    /// What every module cost to build, summed.
    pub fn cost(&self) -> crate::pipelines::PrepareCost {
        let mut c = crate::pipelines::PrepareCost::default();
        for m in &self.modules {
            c.add(m.cost());
        }
        c
    }
}

/// The three bind groups one counting sort dispatches through.
pub struct SortBinds {
    pub zero: wgpu::BindGroup,
    pub count: wgpu::BindGroup,
    pub scan: wgpu::BindGroup,
    pub scatter: wgpu::BindGroup,
}

/// Dynamic offsets for one counting sort, one group per kernel.
#[derive(Clone, Debug, Default)]
pub struct SortOffsets {
    pub zero: Vec<u32>,
    pub count: Vec<u32>,
    pub scan: Vec<u32>,
    pub scatter: Vec<u32>,
}

impl SortOffsets {
    pub fn total(&self) -> usize {
        self.zero.len() + self.count.len() + self.scan.len() + self.scatter.len()
    }
}

// ---------------------------------------------------------------------------
// Host-side scalar packing
// ---------------------------------------------------------------------------

/// Standard-form limbs of `xs`, and how many of them are neither 0 nor 1.
///
/// This is the host half of design §3's "the witness is uploaded in Montgomery form only".
/// It is **not** what the prover does with the witness: `snarkrs-metal` calls `into_bigint()` per
/// scalar, which is one Montgomery reduction each, and in the browser rayon-core falls back
/// to a sequential pool so 140,000 of those would run on one thread. The prover uploads
/// Montgomery limbs (a pure limb split) and derives standard form with one `fr_mont_to_std`
/// dispatch.
///
/// This function exists for the fixed inputs, the tests, and any caller that already holds
/// `Fr` values and does not care. It says so rather than being quietly on the hot path.
pub fn pack_scalars(xs: &[snarkrs_field::Fr]) -> (Vec<u32>, u32) {
    use snarkrs_field::{One, Zero};
    use snarkrs_gpu_layout::PackedScalar;
    let mut out = Vec::with_capacity(xs.len() * LIMBS);
    let mut general = 0u32;
    for x in xs {
        if !(x.is_zero() || x.is_one()) {
            general += 1;
        }
        out.extend_from_slice(&PackedScalar::from_fr(x).v);
    }
    (out, general)
}
