//! The back half of stages 5 to 9 in WGSL: BN254 curve arithmetic and the five point
//! kernels that turn a sorted entry array into one point per window, emitted once per curve.
//!
//! A port of `g16-metal/src/shaders/msm.metal:316-1158`, the half of that file the digit
//! pipeline in [`crate::gen::msm`] leaves untouched. Everything here is written in terms of
//! [`Curve::f`], the coordinate field's function prefix, on top of the `fq_*` and `fq2_*`
//! routines [`crate::gen::field`] already emits, so there is no second copy of the base field
//! anywhere in this crate and no second copy of the curve arithmetic either.
//!
//! # One generator, two groups
//!
//! [`G1`] and [`G2`] are the same source at two sets of constants. That was U10's claim
//! when it wrote this file for G2 alone and U9 checked it by instantiating G1: **four things
//! turned out to be G2-specific and are now parameters**, and none of them was the
//! arithmetic. The entry point names were `const &'static str` with `_g2` baked in rather
//! than derived from the suffix; the affine struct's doc comment named `PackedG2Affine` and
//! the G2 curve equation; the `Fq2` prelude was emitted unconditionally, which is 1.3 KiB a
//! G1 module never calls; and the workgroup sizes were one `Default` rather than one
//! per curve, which matters because an `Xyzz<Fq>` is 128 bytes against an `Xyzz<Fq2>`'s 256
//! and the occupancy arithmetic is therefore different. The `a = 0` short Weierstrass
//! assumption is the one thing genuinely shared, and it holds on both BN254 groups.
//!
//! Four of the five MSMs in a Groth16 proof are G1 (A, B-G1, L, H) and one is G2, so [`G1`]
//! is the hot path here and [`G2`] is the one that is tight against the limits.
//!
//! # Why XYZZ, and why the bases stay affine
//!
//! Extended Jacobian `(X, Y, ZZ, ZZZ)` with `x = X/ZZ`, `y = Y/ZZZ` and the invariant
//! `ZZ^3 = ZZZ^2`. Mixed addition (`madd-2008-s`) is 7M + 2S against 7M + 4S for Jacobian
//! `madd-2007-bl`, and the general addition (`add-2008-s`) is 12M + 2S against 11M + 5S with
//! roughly a third of the field additions. Bucket accumulation is essentially all mixed
//! additions, so that is the operation that decides the kernel. ICME's WGSL MSM lifts every
//! affine base to `z = R` and pays the full general addition in the hottest loop, which is a
//! 35% waste before anything else it does.
//!
//! `ZZ == 0` is the identity, and `(0, 0)` is the affine point at infinity. The second is
//! unambiguous rather than a convention on either group: BN254's G1 is `y^2 = x^3 + 3` and
//! its G2 is `y^2 = x^3 + 3/(9 + u)`, and neither constant term is zero, so `(0, 0)` is off
//! curve and can never be a real point. snarkjs zkeys really do contain points at infinity in
//! the A, B and C query vectors, so this is a case that occurs and not a defensive one.
//!
//! # What `Fq2` costs, and what that does to the shapes below
//!
//! An `Fq2` multiply is 3 `Fq` multiplies through Karatsuba and an `Fq2` square is 2, so
//! every point routine here is about 3x its G1 twin in arithmetic and exactly 2x in bytes. An
//! `Xyzz<Fq2>` is **256 bytes**, four `Fq2` of 64. Two consequences run through this whole
//! file:
//!
//! * `array<PtG2, 64>` is 16384 bytes, which is exactly `maxComputeWorkgroupStorageSize` at
//!   the browser floor. It passes validation and leaves one workgroup resident per core.
//!   [`Workgroups::tg`] carries the measurement of what that is worth.
//! * The bucket array is `n_windows * 2^(c-1) * 256` bytes, so the largest window width the
//!   128 MiB storage binding floor allows is **15**, not the 16 [`crate::msm::MAX_WINDOW`]
//!   permits: `c = 16` is 16 windows x 32768 buckets x 256 B = 134.2 MB. G1 has twice the
//!   headroom for the same `c`. [`crate::points::PointBuffers::new`] rejects it by name
//!   rather than letting `create_buffer` fail with a byte count.
//!
//! # Every branch in `pt_madd` is reachable, which is not obvious
//!
//! The `pp == 0` arm handles two bases in one bucket that are equal or opposite. The signed
//! recoding puts a point and its negation in *different* buckets, so it is tempting to call
//! this dead. It is not: two *different* base indices in one bucket can still hold equal or
//! opposite points, and nothing in a zkey guarantees the query vectors have distinct entries.
//! `tests/msm_g1.rs` and `tests/msm_g2.rs` build both cases on purpose rather than hoping a
//! random draw hits them, and a mutation that deleted the doubling arm passed every other
//! test in the G1 suite.
//!
//! # Uniformity
//!
//! Every `workgroupBarrier()` in this file sits outside every conditional, and the two loops
//! that contain one have a literal bound, because the workgroup size is generated. WGSL makes
//! a barrier in non-uniform control flow a hard shader-creation error where MSL merely makes
//! it undefined, and `tests/wgsl_static.rs::every_barrier_sits_in_uniform_control_flow`
//! checks the emitted text rather than trusting that naga compiled it.

use std::fmt::Write as _;

use crate::gen::field::{Variant, FQ, FQ2_OPS, MUL64};
use crate::gen::msm::params_struct;

// ---------------------------------------------------------------------------
// Which curve
// ---------------------------------------------------------------------------

/// One curve's names, strides and measured shape, so the arithmetic below is written once.
///
/// Every routine this module emits is written in terms of [`Self::f`] and [`Self::fty`], so
/// swapping `fq2` for `fq` and 256 bytes for 128 really is most of the difference between the
/// two groups. The rest of the fields are the things that turned out **not** to follow from
/// those two: the packed layout the base vector arrives in, the curve equation the infinity
/// sentinel argument rests on, whether the `Fq2` prelude is needed at all, and the workgroup
/// sizes, which are measured per curve because a 128-byte accumulator and a 256-byte one do
/// not have the same occupancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Curve {
    /// Entry-point and function suffix: `msm_reduce_g2`, `pt_madd_g2`.
    pub suffix: &'static str,
    /// Field function prefix: `fq2_mul`, `fq2_add`.
    pub f: &'static str,
    /// WGSL type of one coordinate.
    pub fty: &'static str,
    /// WGSL type of an affine base.
    pub aff: &'static str,
    /// WGSL type of an XYZZ accumulator.
    pub pt: &'static str,
    /// The `g16_gpu_layout` type the base vector is packed as, by name. Emitted into the
    /// generated source so the wire format has one name on both sides of the boundary.
    pub packed: &'static str,
    /// This group's curve equation, for the comment that justifies the `(0, 0)` sentinel.
    pub curve_eq: &'static str,
    /// Whether the coordinate field needs [`crate::gen::field::FQ2_OPS`] in front of it.
    /// False for G1, which is 1,286 bytes of WGSL naga would otherwise parse and dead-strip
    /// (a 47,067 byte module against G2's 48,353). Small, and free.
    pub needs_fq2: bool,
    /// Device bytes per affine base. Must equal the `packed` type's `size_of`.
    pub base_bytes: u64,
    /// Device bytes per XYZZ accumulator, four coordinates.
    pub point_bytes: u64,
    /// Entries one thread of `msm_segmented_*` owns. Not baked into the WGSL (it is a
    /// uniform field), but measured per curve, so it lives with the other shape constants
    /// rather than as a bare `const` a plan could take from the wrong curve.
    ///
    /// Accumulation costs `slice_len` mixed additions per thread and the merge costs
    /// `max_bucket_count / slice_len` full additions for the fattest bucket, so the balance
    /// point is near the square root of the worst occupancy. `g16-metal` chose 64 and heliax
    /// reached the same constant independently.
    ///
    /// **Swept per curve rather than inherited, and both land on 128, not Metal's 64.**
    /// `the_g1_slice_length_is_measured` and `the_slice_length_is_measured`, 32,768 general
    /// scalars at `c = 12`, `clear + segmented + merge`, medians of three release runs,
    /// microseconds:
    ///
    /// ```text
    /// slice_len        16        32        64       128       256       512      1024
    /// G1 us      195290.6  106322.1   62398.2   51856.2   69339.5  118077.7  226310.7
    /// G2 us      664100.3  371389.7  240642.5  164629.4  214537.1  361292.0  693739.6
    /// ```
    ///
    /// A clean V with the minimum at 128 on both curves, 16% better than 64 over G1 and 32%
    /// better over G2, and 3.7x worse at either end of the swept range. This is the sharpest
    /// of the three shape constants and the only one where the design's inherited value is
    /// badly wrong rather than marginally so. The two arms are the two costs: below 128 the
    /// spill count doubles with every halving and the merge walks proportionally more slots,
    /// above it each thread's serial run grows and the load balance the segmentation exists
    /// to provide goes away.
    ///
    /// The curves agree, and the G1 column is 3.1x faster, which is the `Fq2` multiply ratio
    /// again.
    pub slice_len: u32,
    /// The measured workgroup sizes for this curve's five entry points.
    pub wg: Workgroups,
    /// What a caller whose bucket array is over the storage binding limit should be told
    /// beyond the byte counts. G2 has a real alternative and G1 does not.
    pub headroom: &'static str,
}

/// BN254's G1, over `Fq`. Four of a proof's five MSMs.
pub const G1: Curve = Curve {
    suffix: "g1",
    f: "fq",
    fty: "Fq",
    aff: "AffG1",
    pt: "PtG1",
    packed: "PackedG1Affine",
    curve_eq: "y^2 = x^3 + 3",
    needs_fq2: false,
    base_bytes: 64,
    point_bytes: 128,
    slice_len: 128,
    wg: Workgroups {
        clear: 256,
        segmented: 128,
        merge: 256,
        tg: 128,
    },
    headroom: "no BN254 group has a narrower accumulator than this one, so the only way \
               down is a narrower window",
};

/// BN254's G2, over `Fq2 = Fq[u]/(u^2 + 1)`.
pub const G2: Curve = Curve {
    suffix: "g2",
    f: "fq2",
    fty: "Fq2",
    aff: "AffG2",
    pt: "PtG2",
    packed: "PackedG2Affine",
    curve_eq: "y^2 = x^3 + 3/(9 + u)",
    needs_fq2: true,
    base_bytes: 128,
    point_bytes: 256,
    slice_len: 128,
    wg: Workgroups {
        clear: 256,
        segmented: 128,
        merge: 256,
        tg: 64,
    },
    headroom: "G1 has twice the headroom at the same c",
};

/// The two, in the order a proof spends its time on them.
pub const CURVES: [Curve; 2] = [G1, G2];

impl Curve {
    /// Bytes one reduction's workgroup array occupies at `tg` threads.
    pub const fn workgroup_bytes(&self, tg: u32) -> u64 {
        self.point_bytes * tg as u64
    }

    /// The largest `tg` whose `array<Pt, tg>` still fits [`FLOOR_WORKGROUP_BYTES`]. 64 for G2,
    /// 128 for G1, and the difference is the whole reason the reduction is swept per curve.
    pub const fn max_tg(&self) -> u32 {
        (FLOOR_WORKGROUP_BYTES / self.point_bytes) as u32
    }

    /// This curve's shipped workgroup sizes with `tg` forced, for the reduction sweep.
    pub const fn with_tg(&self, tg: u32) -> Workgroups {
        Workgroups { tg, ..self.wg }
    }

    // ---- entry point names ----
    //
    // Derived from the suffix rather than written out, because U10 wrote them out as five
    // `const &'static str` with `_g2` in the text and that was one of the four things that
    // stopped this file being generic.

    /// Reset a pooled bucket array to the identity.
    pub fn entry_clear(&self) -> String {
        format!("msm_clear_{}", self.suffix)
    }
    /// Fixed-length-slice bucket accumulation, the kernel that does the work.
    pub fn entry_segmented(&self) -> String {
        format!("msm_segmented_{}", self.suffix)
    }
    /// Fold each bucket's spilled partials into it.
    pub fn entry_merge(&self) -> String {
        format!("msm_merge_{}", self.suffix)
    }
    /// One window's `2^(c-1)` buckets to one point.
    pub fn entry_reduce(&self) -> String {
        format!("msm_reduce_{}", self.suffix)
    }
    /// Sum the bases whose scalar is exactly 1.
    pub fn entry_ones(&self) -> String {
        format!("msm_ones_{}", self.suffix)
    }

    /// The five, in the order a proof dispatches them.
    pub fn entries(&self) -> [String; 5] {
        [
            self.entry_clear(),
            self.entry_segmented(),
            self.entry_merge(),
            self.entry_reduce(),
            self.entry_ones(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Bindings, group 0
// ---------------------------------------------------------------------------

/// The uniform block. Same number and same struct as [`crate::gen::msm`], so U11 can push one
/// parameter block and dispatch both halves of an MSM from it.
pub const BIND_PARAMS: u32 = 0;
/// The bucket array, `n_windows * 2^(c-1)` accumulators. Written by clear and segmented,
/// read and written by merge, read by reduce.
pub const BIND_BUCKETS: u32 = 1;
/// `(row, point << 1 | sign)` per emitted digit, what `msm_scatter` produced.
pub const BIND_ENTRIES: u32 = 2;
/// The affine base vector.
pub const BIND_BASES: u32 = 3;
/// Run ends, what `msm_scatter` left behind.
pub const BIND_CURSOR: u32 = 4;
/// Two spilled partials per slice.
pub const BIND_SPILL_PTS: u32 = 5;
/// The bucket row each spill slot belongs to, or [`NO_ROW`].
pub const BIND_SPILL_ROWS: u32 = 6;
/// Entries per bucket, what `msm_count` produced.
pub const BIND_COUNTS: u32 = 7;
/// One point per window, the reduction's output.
pub const BIND_WSUMS: u32 = 8;
/// Standard-form scalars, for the ones pass only.
pub const BIND_SCALARS: u32 = 9;
/// One point per ones group.
pub const BIND_ONES: u32 = 10;

/// Storage buffers each entry point's pipeline layout declares. The browser floor allows 8
/// and this adapter reports 9 under strict compliance, so a kernel at 9 passes here and
/// fails in Chrome. `tests/msm_g1.rs` and `tests/msm_g2.rs` assert these against the layouts
/// the host builds and `tests/wgsl_static.rs` asserts them against the emitted text, which
/// are different questions: the first is what the device enforces, the second is what a
/// browser would. Curve-independent: the same eleven resources over either group.
pub const STORAGE_CLEAR: u32 = 1;
pub const STORAGE_SEGMENTED: u32 = 6;
pub const STORAGE_MERGE: u32 = 5;
pub const STORAGE_REDUCE: u32 = 2;
pub const STORAGE_ONES: u32 = 3;

/// The spill-slot tag meaning "this slot holds nothing". Not 0, which is bucket row 0.
pub const NO_ROW: u32 = 0xffff_ffff;

// ---------------------------------------------------------------------------
// Workgroup sizes
// ---------------------------------------------------------------------------

/// Threads per workgroup for the five entry points, emitted as literals, per curve.
///
/// **Design §4 assigns 256 to the clear and 64 to the other four, and it is wrong about at
/// least two of them on both curves.** §4 carried `g16-metal`'s threadgroup sizes over on the
/// assumption that MSL's occupancy reasoning transfers. It has now failed for `gather_abc`
/// (24%), the NTT (7.8x), four of the five digit kernels (2 to 7%) and survived once, for
/// `h_join`.
///
/// Measured by `the_g1_workgroup_sizes_are_measured` and `the_g2_workgroup_sizes_are_measured`
/// in `tests/msm_g1.rs` and `tests/msm_g2.rs`: 32,768 general scalars at `c = 12`, one
/// kernel's size varied at a time with the others at the shipped value, **each kernel timed
/// alone**, each cell the median of five **release** runs, GPU wall microseconds per
/// repetition. M2 Max, wgpu 30 through naga to MSL.
///
/// ```text
/// kernel                  16       32       64      128      256    §4  ships
/// msm_clear_g1           5.6      4.1      4.7      4.8      4.9   256    256
/// msm_segmented_g1   29120.6  28939.4  29340.1  28698.5  28706.9    64    128
/// msm_merge_g1       25721.5  24316.3  24346.0  24214.7  24191.6    64    256
///
/// msm_clear_g2          19.6     17.4     17.6     19.2     20.2   256    256
/// msm_segmented_g2   89110.2  88808.3  89218.7  87704.8  87891.6    64    128
/// msm_merge_g2       85287.5  81665.4  81135.8  81687.8  80457.4    64    256
/// ```
///
/// Three things in that table are worth carrying forward, and one of them is a warning.
///
/// **The two curves agree on every 1D size, and the G1 column is 3.05x faster than the G2
/// column.** That ratio is what says the measurement is not an artifact of the harness: an
/// `Fq2` multiply is 3 `Fq` multiplies through Karatsuba, so 3.05x is the arithmetic and
/// nothing else. It also means the G1 sweep was not a formality; it simply agreed, which is
/// the first time in this crate that an inherited shape survived a re-measurement on a second
/// input rather than moving.
///
/// **`msm_segmented_*` wants 128 and `msm_merge_*` wants 256, and Metal's 64 is wrong for
/// both**, though only by 2.1% and 1.6% over G1. The spread across the whole row is under 3%
/// on both curves, which is a real result and a small one: this kernel is memory-bound on the
/// base vector and the entry array, not occupancy-bound, so the workgroup size barely moves
/// it. The design's reasoning (register pressure decides the accumulation) predicts a large
/// effect and there is not one.
///
/// **The clear row is noise and is not a decision.** Its whole spread is 1.5 microseconds
/// over G1 and 2.8 over G2, on a kernel that writes one `Fq`-worth of zeros per bucket and is
/// bandwidth-bound: 45,056 rows at `c = 12` is 1.4 MB over G1, which is about 5 microseconds
/// at this machine's bandwidth, and that is what it costs at every size. Two consecutive runs
/// of the G1 sweep named 64 and then 32 as the winner, which settles it. 256 ships on both
/// curves because it is the fewest dispatched workgroups, not because it won. The tests
/// assert an absolute 25-microsecond slack on this row for exactly that reason; a 3% relative
/// bound on a 5-microsecond row would be a coin flip in CI.
///
/// Release only, and re-run under Tint at U14. The debug harness inverted `h_join`'s answer
/// once already, because differencing cancels the fixed cost of a submit and not the
/// per-encode host cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workgroups {
    pub clear: u32,
    pub segmented: u32,
    pub merge: u32,
    /// Threads in `msm_reduce_*` and `msm_ones_*`, which is also the length of the
    /// `array<Pt, tg>` both hold in workgroup storage. **This is the one size the two curves
    /// do not agree on, and the byte budget is why.**
    ///
    /// An `Xyzz<Fq>` is 128 bytes and an `Xyzz<Fq2>` is 256, so the widest reduction
    /// `maxComputeWorkgroupStorageSize`'s 16384-byte floor allows is 128 threads over G1 and
    /// 64 over G2. Both curves want the widest one they can have.
    ///
    /// `the_g1_reduction_threadgroup_is_measured` and `the_reduction_threadgroup_is_measured`,
    /// 32,768 general scalars at `c = 12`, medians of five release runs, microseconds for
    /// `msm_reduce_*` plus `msm_ones_*`:
    ///
    /// ```text
    /// tg     G1 shared B    G1 us      G2 shared B     G2 us
    ///  8            1024  139054.7            2048  466285.4
    /// 16            2048   71507.1            4096  239764.4
    /// 32            4096   45057.1            8192  140261.5
    /// 64            8192   25286.5           16384   81319.6
    /// 128          16384   17891.5               -         -
    /// ```
    ///
    /// **Every doubling of `tg` is worth 40% to 50%, all the way to the limit.** Design §4
    /// argues the other way, that `tg = 32` keeps two workgroups resident per core where the
    /// widest keeps one, and that occupancy is worth more than the tree; the measurement says
    /// it is worth 2.5x less. The reason is that `tg` is not only the tree width, it is also
    /// the divisor on each thread's serial segment: at half the threads every thread reduces
    /// twice as many buckets one after another, and that term dominates everything else in
    /// the kernel. This is the third time the design's workgroup-storage reasoning has been
    /// measured in this crate and the third time it did not survive.
    ///
    /// Both curves therefore ship a reduction that sits at **exactly** the floor's whole
    /// workgroup allocation with zero headroom, which is uncomfortable and is the trade the
    /// numbers force: one more byte of workgroup storage in either reduction, on any platform
    /// that charges for anything the generator does not count, and the pipeline fails to
    /// create in a browser rather than running slowly. The alternative costs 41% over G1 and
    /// 63% over G2 of a kernel that is a quarter of the point stage. `tests/wgsl_static.rs`
    /// audits both modules at this width on every run, and U14 has to re-measure in Chrome,
    /// where the array is charged by Tint and not by naga.
    pub tg: u32,
}

// ---------------------------------------------------------------------------
// Limits the generator refuses to cross
// ---------------------------------------------------------------------------

/// `maxComputeInvocationsPerWorkgroup` at the browser floor. This adapter allows 1024 and no
/// browser does, so the check is against the floor.
pub const FLOOR_INVOCATIONS: u32 = 256;
/// `maxComputeWorkgroupStorageSize` at the browser floor, in bytes.
pub const FLOOR_WORKGROUP_BYTES: u64 = 16384;

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// One curve's five point kernels behind one copy of the coordinate field's prelude, at that
/// curve's measured shape.
pub fn points_module(v: Variant, c: Curve) -> String {
    points_module_at(v, c, c.wg)
}

/// Same, at chosen workgroup sizes.
///
/// One module and not five, because U8 measured the split: five digit entry points cost
/// 205 ms cold as two modules against 172 ms as one, 17% more, and identical warm. Metal
/// compiles a pipeline per entry point and dead-strips the module around it, so the shared
/// prelude is paid once by naga and never by the platform compiler. The prelude here is `Fq`
/// and `Fq2` only: nothing in this module touches `Fr`, so the 28 KiB of scalar-field
/// multiply the digit module needs is left out.
///
/// # Panics
///
/// On a workgroup size outside `1..=`[`FLOOR_INVOCATIONS`], on a `tg` whose workgroup array
/// would exceed [`FLOOR_WORKGROUP_BYTES`], and on [`Variant::NoCarry13x20`], whose `R = 2^260`
/// wire format is not what the host packers produce.
pub fn points_module_at(v: Variant, c: Curve, wg: Workgroups) -> String {
    assert_eq!(
        v,
        Variant::Cios32Unrolled,
        "the point kernels read g16_gpu_layout::{}, which is 8 limbs of 32 bits; variant \
         {v:?} has a different wire format and no host packer",
        c.packed
    );
    for (name, n) in [
        ("clear", wg.clear),
        ("segmented", wg.segmented),
        ("merge", wg.merge),
        ("tg", wg.tg),
    ] {
        assert!(
            n > 0 && n <= FLOOR_INVOCATIONS,
            "{name} workgroup size {n} is outside 1..={FLOOR_INVOCATIONS}, the floor's \
             maxComputeInvocationsPerWorkgroup"
        );
    }
    let shared = c.workgroup_bytes(wg.tg);
    assert!(
        shared <= FLOOR_WORKGROUP_BYTES,
        "msm_reduce_{} holds array<{}, {}> = {shared} bytes of workgroup storage, over the \
         {FLOOR_WORKGROUP_BYTES} byte browser floor",
        c.suffix,
        c.pt,
        wg.tg
    );

    let mut s = format!(
        "// Generated by g16-wgpu::gen::points, variant {v:?}, curve {}, workgroups clear {} \
         segmented {} merge {} tg {} ({shared} B shared).\n// Do not edit by hand.\n",
        c.suffix, wg.clear, wg.segmented, wg.merge, wg.tg
    );
    s.push_str(MUL64);
    s.push_str(&FQ.ops(v));
    // Only when the coordinate field is Fq2. Carrying a prelude a kernel never calls costs
    // naga time and no pipeline time (U8 measured about 2 ms per module), so this is 1,286
    // bytes off a G1 module rather than anything structural. It is here because a G1 module
    // declaring Fq2 would be a claim about the curve that is not true, not because of the
    // bytes.
    if c.needs_fq2 {
        s.push_str(FQ2_OPS);
    }
    s.push_str(&point_types(c));
    s.push_str(&point_ops(c));
    s.push_str(&params_struct());
    s.push_str(&bindings(c));
    s.push_str(&entry_clear(c, wg.clear));
    s.push_str(&entry_segmented(c, wg.segmented));
    s.push_str(&entry_merge(c, wg.merge));
    s.push_str(&entry_reduce(c, wg.tg));
    s.push_str(&entry_ones(c, wg.tg));
    s
}

/// The affine and accumulator structs, and nothing else: the coordinate type comes from the
/// field prelude.
fn point_types(c: Curve) -> String {
    let (aff, pt, fty, packed, eq) = (c.aff, c.pt, c.fty, c.packed, c.curve_eq);
    format!(
        "
// ---------------------------------------------------------------------------
// Points. {aff} is {base} bytes and matches g16_gpu_layout::{packed} exactly; {pt} is
// {point} bytes. (0, 0) is the affine point at infinity, which is unambiguous rather than a
// convention: this group is {eq}, whose constant term is nonzero, so (0, 0) is off curve.
//
// {pt} is the accumulator: x = X/ZZ, y = Y/ZZZ, with the invariant ZZ^3 = ZZZ^2, and ZZ == 0
// is the identity. That last one is why the clear kernel writes only zz.
// ---------------------------------------------------------------------------

struct {aff} {{ x: {fty}, y: {fty} }}
struct {pt} {{ x: {fty}, y: {fty}, zz: {fty}, zzz: {fty} }}
",
        base = c.base_bytes,
        point = c.point_bytes,
    )
}

/// Every curve routine, written once in terms of `c.f`.
fn point_ops(c: Curve) -> String {
    let (f, aff, pt, sfx) = (c.f, c.aff, c.pt, c.suffix);
    let mut s = String::new();
    let _ = write!(
        s,
        "
fn pt_zero_{sfx}() -> {pt} {{
    return {pt}({f}_zero(), {f}_zero(), {f}_zero(), {f}_zero());
}}

fn pt_is_zero_{sfx}(p: {pt}) -> bool {{ return {f}_is_zero(p.zz); }}

fn aff_is_inf_{sfx}(p: {aff}) -> bool {{ return {f}_is_zero(p.x) && {f}_is_zero(p.y); }}

fn pt_from_affine_{sfx}(p: {aff}) -> {pt} {{
    return {pt}(p.x, p.y, {f}_one(), {f}_one());
}}

// mdbl-2008-s: doubling an affine point straight into XYZZ. BN254 has a = 0 on both G1 and
// G2, so the a*ZZ^2 term of the general doubling disappears.
fn pt_dbl_affine_{sfx}(p: {aff}) -> {pt} {{
    if ({f}_is_zero(p.y)) {{
        // Only reachable for a 2-torsion point, which BN254's odd-order groups do not
        // contain. The identity is the mathematically correct answer anyway.
        return pt_zero_{sfx}();
    }}
    let u = {f}_add(p.y, p.y);
    let v = {f}_sqr(u);
    let w = {f}_mul(u, v);
    let s = {f}_mul(p.x, v);
    let xx = {f}_sqr(p.x);
    let m = {f}_add({f}_add(xx, xx), xx);
    let rx = {f}_sub({f}_sqr(m), {f}_add(s, s));
    let ry = {f}_sub({f}_mul(m, {f}_sub(s, rx)), {f}_mul(w, p.y));
    return {pt}(rx, ry, v, w);
}}

// dbl-2008-s-1 with a = 0.
fn pt_dbl_{sfx}(p: {pt}) -> {pt} {{
    if (pt_is_zero_{sfx}(p) || {f}_is_zero(p.y)) {{
        return pt_zero_{sfx}();
    }}
    let u = {f}_add(p.y, p.y);
    let v = {f}_sqr(u);
    let w = {f}_mul(u, v);
    let s = {f}_mul(p.x, v);
    let xx = {f}_sqr(p.x);
    let m = {f}_add({f}_add(xx, xx), xx);
    let rx = {f}_sub({f}_sqr(m), {f}_add(s, s));
    let ry = {f}_sub({f}_mul(m, {f}_sub(s, rx)), {f}_mul(w, p.y));
    return {pt}(rx, ry, {f}_mul(v, p.zz), {f}_mul(w, p.zzz));
}}

// madd-2008-s: XYZZ += affine, 7M + 2S. The inner loop of bucket accumulation and therefore
// the single hottest routine in this backend.
fn pt_madd_{sfx}(acc: {pt}, p: {aff}) -> {pt} {{
    if (aff_is_inf_{sfx}(p)) {{ return acc; }}
    if (pt_is_zero_{sfx}(acc)) {{ return pt_from_affine_{sfx}(p); }}
    let u2 = {f}_mul(p.x, acc.zz);
    let s2 = {f}_mul(p.y, acc.zzz);
    let pd = {f}_sub(u2, acc.x);
    let rr = {f}_sub(s2, acc.y);
    if ({f}_is_zero(pd)) {{
        // Same x. Either the same point, which doubles, or its negative, which cancels. Both
        // are reachable: the signed recoding puts a point and its negation in different
        // buckets, but two different base indices in one bucket can still be equal or
        // opposite and a zkey does not guarantee distinct bases.
        if ({f}_is_zero(rr)) {{ return pt_dbl_affine_{sfx}(p); }}
        return pt_zero_{sfx}();
    }}
    let pp = {f}_sqr(pd);
    let ppp = {f}_mul(pd, pp);
    let q = {f}_mul(acc.x, pp);
    let rx = {f}_sub({f}_sub({f}_sqr(rr), ppp), {f}_add(q, q));
    let ry = {f}_sub({f}_mul(rr, {f}_sub(q, rx)), {f}_mul(acc.y, ppp));
    return {pt}(rx, ry, {f}_mul(acc.zz, pp), {f}_mul(acc.zzz, ppp));
}}

// add-2008-s: XYZZ + XYZZ, 12M + 2S. Used by the merge and the window reduction, which run
// 2 * 2^(c-1) times per window against n mixed additions in the accumulation.
fn pt_add_{sfx}(a: {pt}, b: {pt}) -> {pt} {{
    if (pt_is_zero_{sfx}(a)) {{ return b; }}
    if (pt_is_zero_{sfx}(b)) {{ return a; }}
    let u1 = {f}_mul(a.x, b.zz);
    let u2 = {f}_mul(b.x, a.zz);
    let s1 = {f}_mul(a.y, b.zzz);
    let s2 = {f}_mul(b.y, a.zzz);
    let pd = {f}_sub(u2, u1);
    let rr = {f}_sub(s2, s1);
    if ({f}_is_zero(pd)) {{
        if ({f}_is_zero(rr)) {{ return pt_dbl_{sfx}(a); }}
        return pt_zero_{sfx}();
    }}
    let pp = {f}_sqr(pd);
    let ppp = {f}_mul(pd, pp);
    let q = {f}_mul(u1, pp);
    let rx = {f}_sub({f}_sub({f}_sqr(rr), ppp), {f}_add(q, q));
    let ry = {f}_sub({f}_mul(rr, {f}_sub(q, rx)), {f}_mul(s1, ppp));
    return {pt}(rx, ry, {f}_mul({f}_mul(a.zz, b.zz), pp), {f}_mul({f}_mul(a.zzz, b.zzz), ppp));
}}

// k * P for a small k: here k is a bucket index inside one window, so it is below 2^15. Plain
// MSB-first double-and-add. It runs once per reduce thread against a whole segment of bucket
// additions, so a windowed ladder would not pay for itself.
fn pt_mul_small_{sfx}(p: {pt}, k: u32) -> {pt} {{
    var acc = pt_zero_{sfx}();
    if (k == 0u || pt_is_zero_{sfx}(p)) {{ return acc; }}
    let top = 31u - countLeadingZeros(k);
    for (var i: i32 = i32(top); i >= 0; i = i - 1) {{
        acc = pt_dbl_{sfx}(acc);
        if (((k >> u32(i)) & 1u) == 1u) {{
            acc = pt_add_{sfx}(acc, p);
        }}
    }}
    return acc;
}}
"
    );
    s
}

/// The eleven module-scope resources. No entry point reaches more than six.
///
/// Every resource gets its own binding number even though no entry point uses them all, so
/// the module is legal under the strictest reading of WGSL's "two resource variables in one
/// entry point's resource interface must not share a group and binding", and each entry point
/// gets a pipeline layout declaring only what it reads. A pipeline layout is allowed to
/// declare bindings the shader ignores, so this costs nothing.
///
/// `SPILL_PTS` and `SPILL_ROWS` are `read_write` even though `msm_merge_g2` only reads them,
/// because one declaration serves both entry points and WGSL has no way to vary the access
/// mode per entry point. The bind group layouts follow, so merge declares them writable and
/// writes nothing.
fn bindings(c: Curve) -> String {
    let pt = c.pt;
    let aff = c.aff;
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Standard form, NOT Montgomery, exactly as msm_count and msm_scatter read it. Only the ones
// pass needs it, to test a scalar for equality with 1.
alias Scalar = array<u32, 8>;

// A spill slot holding nothing. Not 0, which is a real bucket row.
const NO_ROW: u32 = {NO_ROW}u;

@group(0) @binding({BIND_BUCKETS}) var<storage, read_write> BUCKETS: array<{pt}>;
@group(0) @binding({BIND_ENTRIES}) var<storage, read> ENTRIES: array<vec2<u32>>;
@group(0) @binding({BIND_BASES}) var<storage, read> BASES: array<{aff}>;
@group(0) @binding({BIND_CURSOR}) var<storage, read> CURSOR: array<u32>;
@group(0) @binding({BIND_SPILL_PTS}) var<storage, read_write> SPILL_PTS: array<{pt}>;
@group(0) @binding({BIND_SPILL_ROWS}) var<storage, read_write> SPILL_ROWS: array<u32>;
@group(0) @binding({BIND_COUNTS}) var<storage, read> COUNTS: array<u32>;
@group(0) @binding({BIND_WSUMS}) var<storage, read_write> WSUMS: array<{pt}>;
@group(0) @binding({BIND_SCALARS}) var<storage, read> SCALARS: array<Scalar>;
@group(0) @binding({BIND_ONES}) var<storage, read_write> ONES: array<{pt}>;
"
    );
    s
}

fn entry_clear(c: Curve, wg: u32) -> String {
    let f = c.f;
    let entry = c.entry_clear();
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Every bucket has to start at the identity, and unlike a fresh WebGPU allocation (which is
// zero) a pooled buffer holds the previous proof's points.
//
// Only zz is written. pt_add, pt_madd and the host conversion all test zz alone, and a bucket
// the segmented pass direct-writes is overwritten in full anyway, so clearing the other three
// coordinates would be {bytes} bytes of pure memory traffic per bucket instead of {quarter}.
@compute @workgroup_size({wg})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    if (i >= P.n_windows * P.n_buckets) {{ return; }}
    BUCKETS[i].zz = {f}_zero();
}}
",
        bytes = c.point_bytes,
        quarter = c.point_bytes / 4,
    );
    s
}

fn entry_segmented(c: Curve, wg: u32) -> String {
    let (f, sfx) = (c.f, c.suffix);
    let entry = c.entry_segmented();
    let mut s = String::new();
    let _ = write!(
        s,
        "
// The load-balanced bucket accumulation, and the kernel that is 94% of the point stage.
//
// THE PROBLEM IT SOLVES, measured rather than assumed. One thread per bucket makes per-thread
// work proportional to bucket occupancy, and occupancy is wildly non-uniform: a witness
// contains repeated values and every copy of one value lands in the same bucket of every
// window. g16-metal measured, on js_2x2_d32 at c = 11, a busiest bucket of 11,758 entries
// against a mean of 17.02, a 691x imbalance. That single thread ran 11,758 serial mixed
// additions while 24,575 others finished in about 17 and waited, and it cost 201 ms of a
// 253 ms MSM. End to end over five MSMs: 455.9 ms against 126.8 ms segmented.
//
// THE FIX, zkmopro's segmented SMVP adapted to this layout. Slice the entry array into fixed
// runs of slice_len and give one thread each slice, so per-thread work is uniform BY
// CONSTRUCTION rather than by hoping the digits spread. Within its slice a thread finds
// bucket boundaries by watching entry.x change.
//
//   * A run that neither starts at the slice's first entry nor ends at its last is wholly
//     contained here, so no other thread will ever touch that bucket and it is written
//     straight to BUCKETS[row] with no synchronisation of any kind.
//   * The first and last runs may continue into the neighbouring slices, so they go to this
//     slice's two spill slots tagged with their row. At most two spills per thread.
//
// The run that ends at the slice boundary always spills, whether or not it actually
// continues. Spilling one run that did not need to costs the merge one addition; failing to
// spill one that did would lose it silently.
@compute @workgroup_size({wg})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = P.lo + gid.x;
    if (t >= P.n_windows * P.slices) {{ return; }}
    let w = t / P.slices;
    let k = t - w * P.slices;
    let base = w * P.cap;
    // The scatter left every cursor at its run's end, so the last bucket's cursor is the end
    // of this window's whole region. Slices past it are empty.
    let used = CURSOR[w * P.n_buckets + P.n_buckets - 1u] - base;

    let head_slot = 2u * t;
    let tail_slot = head_slot + 1u;
    // Both slots are tagged empty before the early return, because the merge reads every slot
    // in a bucket's slice range and a pooled spill_rows buffer holds the last proof's tags.
    SPILL_ROWS[head_slot] = NO_ROW;
    SPILL_ROWS[tail_slot] = NO_ROW;

    let lo = k * P.slice_len;
    if (lo >= used) {{ return; }}
    let hi = min(lo + P.slice_len, used);

    var cur_row = ENTRIES[base + lo].x;
    var acc = pt_zero_{sfx}();
    var first_run = true;

    for (var i = lo; i < hi; i = i + 1u) {{
        let e = ENTRIES[base + i];
        if (e.x != cur_row) {{
            if (first_run) {{
                SPILL_ROWS[head_slot] = cur_row;
                SPILL_PTS[head_slot] = acc;
                first_run = false;
            }} else {{
                BUCKETS[cur_row] = acc;
            }}
            acc = pt_zero_{sfx}();
            cur_row = e.x;
        }}
        var b = BASES[P.base_off + (e.y >> 1u)];
        // The signed digit's sign travels in the entry's low bit, so a negated base costs one
        // field negation here and no second bucket.
        if ((e.y & 1u) != 0u) {{ b.y = {f}_neg(b.y); }}
        acc = pt_madd_{sfx}(acc, b);
    }}

    if (first_run) {{
        SPILL_ROWS[head_slot] = cur_row;
        SPILL_PTS[head_slot] = acc;
    }} else {{
        SPILL_ROWS[tail_slot] = cur_row;
        SPILL_PTS[tail_slot] = acc;
    }}
}}
"
    );
    s
}

fn entry_merge(c: Curve, wg: u32) -> String {
    let sfx = c.suffix;
    let entry = c.entry_merge();
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Fold each bucket's spilled partials into it. One thread per bucket, looking only at the
// slices its own run overlaps, which it computes from the run's start and end, so there is no
// search and no atomic. The worst case is count / slice_len additions for the fattest bucket,
// which turns g16-metal's 11,758-step serial loop into about 180.
@compute @workgroup_size({wg})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = P.lo + gid.x;
    if (row >= P.n_windows * P.n_buckets) {{ return; }}
    let cnt = COUNTS[row];
    if (cnt == 0u) {{ return; }}
    let w = row / P.n_buckets;
    let base = w * P.cap;
    let start = CURSOR[row] - cnt - base;
    let end = CURSOR[row] - base;
    let k_lo = start / P.slice_len;
    // The last slice this bucket's run reaches. `cnt > 0` was checked above so `end >= 1`
    // and the subtraction cannot wrap; `end` is one past the run, so `end - 1` is its last
    // entry. Setting this to `k_lo` looks harmless when a run fits inside one slice and
    // silently drops every spill past the first boundary when it does not, which is correct
    // for every n up to slice_len and wrong for every n above it.
    let k_hi = (end - 1u) / P.slice_len;

    // Spills first, into a local identity, and the bucket is touched only if there were any.
    //
    // The obvious spelling is `var acc = BUCKETS[row]` then add the spills then store, which
    // is what the MSL original does. It costs a 256-byte read and a 256-byte write on EVERY
    // nonzero bucket, and most buckets have no spill at all: there are two spill slots per
    // slice and `2^(c-1)` buckets per window, so at c = 12 with slice_len 128 that is 512
    // slots against 2048 buckets and at least three quarters of the threads would move 512
    // bytes to add nothing. Measured on 32,768 general scalars at c = 12, M2 Max, release:
    // **84.8 ms the MSL way against 6.1 ms this way, 13.9x**, and it takes the whole G2 MSM
    // from 249 ms to 170 ms. The 84.8 ms was 23 MB of bucket traffic at 0.27 GB/s, which is
    // what says the cost was never the arithmetic.
    //
    // Addition is commutative and associative, so folding the spills together first and
    // adding that to the bucket once is the same point.
    var acc = pt_zero_{sfx}();
    var any = false;
    for (var k = k_lo; k <= k_hi; k = k + 1u) {{
        let slot = 2u * (w * P.slices + k);
        if (SPILL_ROWS[slot] == row) {{
            acc = pt_add_{sfx}(acc, SPILL_PTS[slot]);
            any = true;
        }}
        if (SPILL_ROWS[slot + 1u] == row) {{
            acc = pt_add_{sfx}(acc, SPILL_PTS[slot + 1u]);
            any = true;
        }}
    }}
    if (any) {{
        BUCKETS[row] = pt_add_{sfx}(BUCKETS[row], acc);
    }}
}}
"
    );
    s
}

fn entry_reduce(c: Curve, tg: u32) -> String {
    let (pt, sfx) = (c.pt, c.suffix);
    let entry = c.entry_reduce();
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Collapse one window's 2^(c-1) buckets to one point. One workgroup per window.
//
// The window sum is sum_j (j+1) B_j. Split the buckets into one segment per thread at
// [lo, hi). Inside a segment the reverse running sum gives P = sum_j (j - lo + 1) B_j and
// Q = sum_j B_j in two additions per bucket, and the segment contributes P + lo * Q. The
// per-thread results are then tree-reduced in workgroup memory, so the host reads back one
// point per window and does nothing but the Horner combination.
//
// {tg} threads is {shared} bytes of workgroup storage. See gen::points::Workgroups::tg for
// what the alternatives measured and why this one ships.
var<workgroup> SHARED: array<{pt}, {tg}>;

@compute @workgroup_size({tg})
fn {entry}(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) tid: u32) {{
    let w = wid.x;
    let seg_len = (P.n_buckets + {tg}u - 1u) / {tg}u;
    let lo = tid * seg_len;
    let hi = min(lo + seg_len, P.n_buckets);

    var mine = pt_zero_{sfx}();
    if (lo < hi) {{
        var run = pt_zero_{sfx}();
        var tot = pt_zero_{sfx}();
        for (var j = hi; j > lo; j = j - 1u) {{
            run = pt_add_{sfx}(run, BUCKETS[w * P.n_buckets + (j - 1u)]);
            tot = pt_add_{sfx}(tot, run);
        }}
        mine = pt_add_{sfx}(tot, pt_mul_small_{sfx}(run, lo));
    }}
    SHARED[tid] = mine;

    // The barrier sits outside the conditional and the loop bound is the literal workgroup
    // size, so every invocation reaches every barrier the same number of times. WGSL makes
    // that a hard shader-creation error to get wrong; MSL merely makes it undefined.
    for (var s = 1u; s < {tg}u; s = s << 1u) {{
        workgroupBarrier();
        if ((tid & ((s << 1u) - 1u)) == 0u && tid + s < {tg}u) {{
            SHARED[tid] = pt_add_{sfx}(SHARED[tid], SHARED[tid + s]);
        }}
    }}
    workgroupBarrier();
    if (tid == 0u) {{ WSUMS[w] = SHARED[0]; }}
}}
",
        shared = c.workgroup_bytes(tg),
    );
    s
}

fn entry_ones(c: Curve, tg: u32) -> String {
    let sfx = c.suffix;
    let entry = c.entry_ones();
    let mut s = String::new();
    let _ = write!(
        s,
        "
// The scalar-of-1 path. Four of the five MSMs take the witness as scalars and a bit-heavy
// circuit is over 99% zeros and ones, and msm_count and msm_scatter route both classes out of
// Pippenger entirely, so something has to add the one-scalars back. One mixed addition each.
//
// Strided so consecutive lanes read consecutive scalars, then the same tree as the reduction,
// so the host adds only ones_groups points. It shares SHARED with msm_reduce_{sfx}: they never
// run at the same time and a second array would double this module's workgroup allocation for
// nothing.
@compute @workgroup_size({tg})
fn {entry}(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) tid: u32) {{
    let g = wid.x;
    let stride = P.ones_groups * {tg}u;
    var acc = pt_zero_{sfx}();
    for (var i = g * {tg}u + tid; i < P.n; i = i + stride) {{
        let sc = SCALARS[P.scalar_off + i];
        let hi = sc[1] | sc[2] | sc[3] | sc[4] | sc[5] | sc[6] | sc[7];
        if (!(hi == 0u && sc[0] == 1u)) {{ continue; }}
        // i, not scalar_off + i: base_off is this MSM's own offset into the base vector, and
        // the two ranges are independent. The L MSM reads scalars from n_public + 1 and bases
        // from 0.
        acc = pt_madd_{sfx}(acc, BASES[P.base_off + i]);
    }}
    SHARED[tid] = acc;

    for (var s = 1u; s < {tg}u; s = s << 1u) {{
        workgroupBarrier();
        if ((tid & ((s << 1u) - 1u)) == 0u && tid + s < {tg}u) {{
            SHARED[tid] = pt_add_{sfx}(SHARED[tid], SHARED[tid + s]);
        }}
    }}
    workgroupBarrier();
    if (tid == 0u) {{ ONES[g] = SHARED[0]; }}
}}
"
    );
    s
}
