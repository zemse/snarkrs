//! Window sizing: the cost model that picks `c`, and the tuning constants and
//! `G16_METAL_MSM_*` overrides that shape every other dispatch.

/// Bits the signed recoding is laid out over: 254 for BN254's `Fr`, plus one so the
/// carry out of the top window is provably zero.
pub(super) const RECODE_BITS: usize = 255;
const _: () = assert!(RECODE_BITS == snarkrs_msm::RECODE_BITS);

/// Caps the bucket array at 2^15 points per window.
const MAX_WINDOW: u32 = 16;

/// Whether an MSM's cost may depend on the scalar values. The Metal twin of
/// `snarkrs_msm`'s, decided per scalar upload so a plan built over the buffer inherits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// No scalar is classified or skipped: every one of the `n` scalars emits one entry
    /// per window, a zero digit into one of [`DUMMY_ROWS`] rows behind the real buckets
    /// that nothing reads, and the window is priced for `n` scalars over the full 255
    /// bits. The window width, every buffer, every dispatch and the number of mixed
    /// additions then follow the key (`n`, and which bases are at infinity) and not the
    /// witness.
    ///
    /// That closes the channel of "Remote Side-Channel Attacks on Anonymous
    /// Transactions" (USENIX Security 2020), which recovered Zcash witness sparsity from
    /// proving time. The merge is the fixed tree of `msm_fold_*` rather than a walk per
    /// bucket, and the fold and the reduce add with the complete formulas (`Proj` in
    /// msm.metal), so an empty bucket or slot costs what a full one does. It is not
    /// constant time in the strict sense: bucket occupancy still follows the digit
    /// values, so the atomics' contention and the accumulation's run count stay data
    /// dependent, as in every Pippenger; see
    /// [`MetalBackend::constant_work`](crate::MetalBackend::constant_work) for what that
    /// leaves.
    ///
    /// What it costs is what the zeros and ones were saving, so it is priced per
    /// circuit in `MetalBackend::constant_work`: about a fifth on a dense witness, 4-6x
    /// on a bit-heavy one.
    Constant,
    /// Zero scalars are dropped, one scalars go to the ones gather, the window is sized
    /// for the rest over the bits they actually use. Running time, buffer sizes and the
    /// dispatch geometry then follow how many witness entries are zero or one.
    Variable,
}

/// Dummy rows per window under [`Work::Constant`]. A power of two, so a thread's row is
/// its index masked; several rather than one only so the zero digits' atomics spread
/// over that many addresses instead of serialising on one.
pub(super) const DUMMY_ROWS: usize = 64;
const _: () = assert!(DUMMY_ROWS.is_power_of_two());

/// Mixed additions, G1-weighted, that one command buffer of a split batch may hold in
/// its accumulation piece: about 14 ms at `MADD_US`, the size of a transform buffer.
/// See [`Submission`](super::Submission).
pub(super) const SPLIT_BUDGET: usize = 1 << 22;

/// Threads per threadgroup in the reduction and the ones kernel. Must equal `REDUCE_TG`
/// in `msm.metal`, which sizes the threadgroup array; the dispatch may use fewer if the
/// pipeline reports a lower maximum, and the kernel reads the real count at runtime.
pub(super) const REDUCE_TG: usize = 64;

/// Threadgroup array size of the prefix-sum kernel. Same contract as [`REDUCE_TG`].
pub(super) const SCAN_TG: usize = 256;

/// Threads per threadgroup in the wide merge. Same contract as [`REDUCE_TG`].
pub(super) const MERGE_TG: usize = 64;

/// The spill slot sentinel. Must equal `MSM_NO_ROW` in `msm.metal`: a slot holding it
/// spilled nothing, and the host-tail combine skips it the way `msm_merge_*` does.
pub(super) const MSM_NO_ROW: u32 = u32::MAX;

/// Entries one thread of the segmented accumulation owns.
///
/// Accumulation costs `SLICE_LEN` mixed additions per thread and the merge costs
/// `max_bucket_count / SLICE_LEN` full additions for the fattest bucket, so the balance
/// point is near the square root of the worst occupancy, which is a few hundred on these
/// artifacts. 64 sits under that on purpose: it also keeps the thread count high enough
/// to fill the machine at the smaller domains, where there are only a few thousand
/// slices to begin with. Override with `G16_METAL_MSM_L` to sweep it.
const SLICE_LEN: usize = 64;

/// The `G16_METAL_MSM_L` override: unparsable or zero is ignored, and anything past
/// `u32::MAX` is clamped, because the shader takes the slice length as a `u32` and a
/// truncation to zero would leave every bucket at the identity.
pub(super) fn parse_slice_override(v: &str) -> Option<usize> {
    v.parse::<usize>()
        .ok()
        .filter(|v| *v > 0)
        .map(|v| v.min(u32::MAX as usize))
}

fn forced_slice_len() -> Option<usize> {
    std::env::var("G16_METAL_MSM_L")
        .ok()
        .and_then(|v| parse_slice_override(&v))
}

fn slice_len() -> usize {
    forced_slice_len().unwrap_or(SLICE_LEN)
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

pub(super) fn slice_len_for(n_windows: usize, cap: usize) -> usize {
    if let Some(l) = forced_slice_len() {
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
/// thread keeps at least two buckets per group. One bucket per thread is where the split
/// stops paying: at c=8 it halves 128 buckets into a lane each, and the fixed
/// `pt_mul_small` every lane owes then outweighs the segment it saves. Counting the
/// shader's segment loop, `pt_mul_small` and the threadgroup tree for a full window gives
/// 8,251 G1 field products at one group against 13,809 at two, and the mid-ladder
/// circuits measure the difference: `js_1x1_d8` 7.24 ms against 7.57, `js_2x2_d16` 12.88
/// against 13.16, `railgun-01x01` 18.46 against 18.81, `tornado` 24.85 against 25.07. The target is higher than the
/// accumulation's because a reduce thread pays a fixed `pt_mul_small` on top of its
/// segment, so past the sweet spot more threads mean more of that tax: at c=13 on the
/// csp artifacts the sweep read 2.52 ms at 4 groups, 2.06 at 8, 2.48 at 16, 3.68 at 32.
/// Override with `G16_METAL_MSM_RG` to sweep it.
const REDUCE_TARGET_THREADS: usize = 8192;

pub(super) fn reduce_groups_for(n_windows: usize, n_buckets: usize) -> usize {
    let forced = std::env::var("G16_METAL_MSM_RG")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    if let Some(g) = forced {
        return g.min(n_buckets.max(1));
    }
    let mut g = 1;
    while n_buckets / (2 * g) >= 2 * REDUCE_TG && n_windows * g * REDUCE_TG < REDUCE_TARGET_THREADS
    {
        g *= 2;
    }
    g
}

/// Buckets at or below which a one-window plan skips the GPU merge and reduce and
/// finishes in [`Outputs::combine`](super::Outputs::combine) instead.
///
/// A one-window witness plan on the csp keccak circuits has 256 buckets holding a few
/// hundred entries, and its reduce dispatches `reduce_groups * REDUCE_TG` threads, 128
/// on a device with 4,864 ALUs, each a dependent chain of point additions capped by a
/// `pt_mul_small` and a six-level threadgroup tree. That is latency, not work: the G2
/// reduce measured 2.96 ms against a 0.014 ms arithmetic floor, and sweeping
/// `G16_METAL_MSM_RG` moved it only to 2.20, because no group count fixes a dispatch
/// two threadgroups wide. The host does the same 2 * 256 additions plus the spill fold
/// in well under half a millisecond inside the rayon combine, where the module already
/// sums the per-threadgroup partials and runs the Horner. 256 is the largest shape
/// measured; a one-window plan can in principle carry up to 2^15 buckets, and 2 * 2^15
/// host additions would cost more than the dispatch it replaces.
///
/// Measured in production at keccak_128: the witness batch's device time drops from
/// 8.33 ms to 2.58 on the driver clock, and the warm minima move by -0.3 there, -0.1
/// at keccak_256 and -0.5 at keccak_512, with sha256 and the whole ladder flat. The
/// wall win is a tenth of the device win because the removed dispatches were latency,
/// not throughput: the second queue's arbitration was already filling their idle lanes
/// with transform and H work, so most of what they cost was borne off the critical
/// path. What the change buys beyond the milliseconds is headroom: those 5.7 ms stop
/// competing with the transforms the moment anything else on the device gets faster.
pub(super) const HOST_TAIL_MAX_BUCKETS: usize = 256;

/// Expected slice span of a top-window bucket at or above which its merge moves to
/// `msm_merge_wide_*`, one threadgroup per bucket instead of one thread.
///
/// `msm_merge_*` walks a bucket's slice range serially, so a top-window bucket whose
/// run spans dozens of slices holds the whole merge dispatch open in a single lane.
/// The wide kernel strides the range across [`MERGE_TG`] threads and folds a tree.
/// Below the threshold the serial walk is already two or three dependent additions and
/// a threadgroup would only add barriers.
///
/// Measured on the csp keccak circuits with overlap off, driver clock: the H merge
/// phase drops from 0.85 to 0.65 ms at 2^17 (span 16) and from 1.57 to 0.78 ms at 2^18
/// (span 32); at 2^19 the top window is full (c=15 spends all 15 top bits), the span
/// estimate is below one and the wide path stays off. Warm medians move by the same
/// amounts, -0.35 and -0.85 ms, because the H MSM is the critical path. Override with
/// `G16_METAL_MSM_WIDE` to sweep it; 0 disables the wide path.
const MERGE_WIDE_MIN_SPAN: usize = 4;

pub(super) fn merge_wide_min_span() -> usize {
    std::env::var("G16_METAL_MSM_WIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(MERGE_WIDE_MIN_SPAN)
}

/// Spill slots one thread of `msm_fold_*` takes per level: the constant-work merge,
/// whose shape follows the key rather than which buckets the digits landed in.
///
/// Swept at 2^18 (c=13, 8192 slices a window), merge phase on a dense witness, G1/G2:
/// 4 slots 0.96/3.60 ms over seven levels, 8 0.98/3.88 over five, 16 1.17/4.66, 32
/// 1.57/7.03; the longer the group the longer each thread's dependent chain. 8 ties 4
/// with two fewer barriers. Override with `G16_METAL_MSM_FOLD` (at least 4, so every
/// level shrinks the slot count).
const FOLD_LEN: usize = 8;

pub(super) fn fold_len() -> usize {
    std::env::var("G16_METAL_MSM_FOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 4)
        .unwrap_or(FOLD_LEN)
}

/// Bucket rows at or above which the scatter's windows split into two serial
/// dispatches.
///
/// Every scatter store lands on the cache line holding its bucket run's tail, so the
/// kernel keeps one line per row live: 128 bytes times the row count. The c=13 H plans
/// hold 81,920 rows, 10.5 MB, and their scattered stores cost 0.07 ms over 2.6 M
/// entries; the c=15 plan holds 278,528 rows, 35.6 MB against a cache shared with the
/// scalar stream, and pays 0.58 ms over 8.9 M entries, 2.4x more per entry. Halving
/// the live set by dispatching the windows in two serial halves measured 0.88 to
/// 0.62 ms at that shape (three halves tied, four lost), while the c=13 shapes lost
/// 0.01 ms to the second scalar read, so the split starts where they end. Override
/// with `G16_METAL_MSM_SCATTER_SPLIT`; 0 keeps every plan in one dispatch.
pub(super) const SCATTER_SPLIT_ROWS: usize = 131072;

pub(super) fn scatter_split_rows() -> usize {
    match std::env::var("G16_METAL_MSM_SCATTER_SPLIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) => usize::MAX,
        Some(rows) => rows,
        None => SCATTER_SPLIT_ROWS,
    }
}

/// `G16_METAL_MSM_LEGACY_ACC=1` swaps the segmented accumulation for the simple
/// one-thread-per-bucket kernel. Kept only so the load-imbalance claim in
/// `shaders/msm.metal` can be reproduced rather than taken on trust.
pub(super) fn legacy_accumulate() -> bool {
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
    window_size_for(m, RECODE_BITS)
}

/// [`window_size`] with the recoding laid out over `recode_bits` instead of the full
/// 255. A plan whose scalars are provably short
/// (see [`ScalarBuf::bits_in`](super::ScalarBuf::bits_in)) prices its windows over the
/// bits it will actually dispatch, so the `windows * m` term stops charging for digit
/// rows that hold nothing: at sha256_128 the four witness plans use 8 of the 51 windows
/// a full-width layout would emit at c = 5.
///
/// A layout is at least one bit wide; 0 is read as 1.
pub fn window_size_for(m: usize, recode_bits: usize) -> u32 {
    if let Ok(v) = std::env::var("G16_METAL_MSM_C") {
        if let Ok(c) = v.parse::<u32>() {
            return c.clamp(2, MAX_WINDOW);
        }
    }
    // `w - 1` in the loop below underflows at zero bits. `Plan::new` cannot get there,
    // since it prices `bits_in + 1`, but this is public and its callers pick their own
    // width.
    let recode_bits = recode_bits.max(1);
    /// One mixed addition in the segmented accumulation, microseconds.
    const MADD_US: f64 = 0.00324;
    /// One bucket's share of the clear, merge and reduction kernels, microseconds.
    ///
    /// Refit from 0.025 after the wide merge and the scan reduce cut what a row
    /// costs, against production warm medians rather than isolated phases: the
    /// anchors are c=10 winning tornado's plans at m about 33k (24.5 ms against 26.4
    /// at the old c=8 and 30.2 at c=13), c=12 winning sha256's H at m=65,536 (its
    /// minima read 9.9-10.0 against 10.2 at c=13, and c=10 loses at 10.4-10.7),
    /// c=12 winning js_8x8_d32's witness plans at m about 69k by 2.5 ms, and c=13
    /// and c=15 keeping the csp H jobs at 2^17/2^18 and 2^19. 0.022 holds every one
    /// of those with at least 70 us of modelled margin; 0.025 tips the 65k H to
    /// c=10, which is measured slower there.
    const ROW_US: f64 = 0.022;
    /// One iteration of the merge loop on the busiest bucket, microseconds. Serial.
    const MERGE_US: f64 = 27.0;
    /// The wide merge's floor, microseconds: its dispatch plus one threadgroup's
    /// stride-and-tree. Bracketed rather than fitted: subtracting the spill half of
    /// the measured merge phase leaves 60 to 160 us across the two csp sizes that
    /// engage it, and no ranking above or below flips anywhere in that range. With
    /// the wide path priced, forcing the once-pathological widths confirms the
    /// model: at 2^17 c=12/13/14 measure 10.0/9.1/10.0 ms on the H job (they read
    /// 27.5/11.2/27.8 before the wide merge), and at 2^19 c=15 keeps winning at
    /// 27.3 against 29.5/29.8, so the blow-ups the serial term guarded against are
    /// gone rather than merely repriced.
    const WIDE_US: f64 = 100.0;

    let slice = slice_len() as f64;
    let mut best = 3;
    let mut best_cost = f64::MAX;
    for c in 3..=MAX_WINDOW {
        let cu = c as usize;
        let w = recode_bits.div_ceil(cu);
        // Bits the top window actually reaches, hence how many of its buckets are live.
        let top_bits = recode_bits - (w - 1) * cu;
        let top_buckets = (1u64 << (top_bits.min(cu) - 1)) as f64;
        let fattest = m as f64 / top_buckets;

        // The serial walk only prices spans the wide merge leaves alone; past the
        // threshold the fattest bucket costs one threadgroup's strided pass instead.
        let span = fattest / slice;
        let fat = if span >= MERGE_WIDE_MIN_SPAN as f64 {
            WIDE_US + MERGE_US * span / MERGE_TG as f64
        } else {
            MERGE_US * span
        };
        let cost = MADD_US * (w * m) as f64 + ROW_US * (w << (cu - 1)) as f64 + fat;
        if cost < best_cost {
            best_cost = cost;
            best = c;
        }
    }
    best
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
pub(super) fn ones_groups_for(n: usize) -> usize {
    let spt = std::env::var("G16_METAL_MSM_ONES_SPT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(8);
    n.div_ceil(REDUCE_TG * spt).clamp(1, 256)
}
