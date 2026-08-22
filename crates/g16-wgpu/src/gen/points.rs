//! The back half of stages 5 to 9 in WGSL: BN254 curve arithmetic and the five point
//! kernels that turn a sorted entry array into one point per window.
//!
//! A port of `g16-metal/src/shaders/msm.metal:316-1158`, the half of that file the digit
//! pipeline in [`crate::gen::msm`] leaves untouched. Everything here is `Fq2` arithmetic on
//! top of the `fq_*` and `fq2_*` routines [`crate::gen::field`] already emits, so there is no
//! second copy of the base field anywhere in this crate.
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
//! unambiguous rather than a convention: BN254's G2 is `y^2 = x^3 + 3/(9 + u)`, whose
//! constant term is nonzero, so `(0, 0)` is off curve and can never be a real point. snarkjs
//! zkeys really do contain points at infinity in the query vectors, so this is a case that
//! occurs and not a defensive one.
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
//! `tests/msm_g2.rs` builds both cases on purpose rather than hoping a random draw hits them.
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

/// One curve's names and strides, so the arithmetic below is written once.
///
/// Only [`G2`] exists today. G1 is design §8's U9 and is one more `const` of this shape plus
/// the host wiring: every routine emitted here is written in terms of `c.f` and `c.fty`, so
/// swapping `fq2` for `fq` and 256 bytes for 128 is the whole of the difference. Nothing in
/// this file is G2-specific except those two constants and the `a = 0` short Weierstrass
/// assumption, which holds on both BN254 groups.
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
    /// Device bytes per affine base. Must equal `g16_gpu_layout::PackedG2Affine`.
    pub base_bytes: u64,
    /// Device bytes per XYZZ accumulator.
    pub point_bytes: u64,
}

/// BN254's G2, over `Fq2 = Fq[u]/(u^2 + 1)`.
pub const G2: Curve = Curve {
    suffix: "g2",
    f: "fq2",
    fty: "Fq2",
    aff: "AffG2",
    pt: "PtG2",
    base_bytes: 128,
    point_bytes: 256,
};

impl Curve {
    /// Bytes one reduction's workgroup array occupies at `tg` threads.
    pub const fn workgroup_bytes(&self, tg: u32) -> u64 {
        self.point_bytes * tg as u64
    }
}

// ---------------------------------------------------------------------------
// Entry point names
// ---------------------------------------------------------------------------

/// Reset a pooled bucket array to the identity.
pub const ENTRY_CLEAR: &str = "msm_clear_g2";
/// Fixed-length-slice bucket accumulation, the kernel that does the work.
pub const ENTRY_SEGMENTED: &str = "msm_segmented_g2";
/// Fold each bucket's spilled partials into it.
pub const ENTRY_MERGE: &str = "msm_merge_g2";
/// One window's `2^(c-1)` buckets to one point.
pub const ENTRY_REDUCE: &str = "msm_reduce_g2";
/// Sum the bases whose scalar is exactly 1.
pub const ENTRY_ONES: &str = "msm_ones_g2";

/// The five, in the order a proof dispatches them.
pub const ENTRIES: [&str; 5] = [
    ENTRY_CLEAR,
    ENTRY_SEGMENTED,
    ENTRY_MERGE,
    ENTRY_REDUCE,
    ENTRY_ONES,
];

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
/// fails in Chrome. `tests/msm_g2.rs` asserts these against the layouts the host builds and
/// `tests/wgsl_static.rs` asserts them against the emitted text, which are different
/// questions: the first is what the device enforces, the second is what a browser would.
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

/// Threads per workgroup for the five entry points, emitted as literals.
///
/// **Design §4 assigns 256 to `msm_clear_g2` and 64 to the other four, and three of those
/// four are wrong.** §4 carried `g16-metal`'s threadgroup sizes over on the assumption that
/// MSL's occupancy reasoning transfers. It has now failed for `gather_abc` (24%), the NTT
/// (7.8x), four of the five digit kernels (2 to 7%) and survived once, for `h_join`.
///
/// Measured by `tests/msm_g2.rs::the_g2_workgroup_sizes_are_measured`, 32,768 general scalars
/// at `c = 12`, one kernel's size varied at a time with the others at the shipped value, each
/// cell the median of five **release** runs of the whole five-kernel point stage, GPU wall
/// microseconds:
///
/// ```text
/// kernel                 16      32      64     128     256    §4  ships
/// msm_clear_g2        176.6   132.3   115.8   112.4   115.1   256    128
/// msm_segmented_g2   6216.5  5449.1  5507.4  5691.4  5806.9    64     32
/// msm_merge_g2        438.5   322.4   288.3   281.5   287.6    64    128
/// ```
///
/// `msm_segmented_g2` is 94% of the point stage and it wants **32**, which is 1.1% better
/// than Metal's 64 and 14% better than 16. It is also the one kernel here whose register
/// pressure is extreme: one thread holds an `Xyzz<Fq2>` accumulator, an `Aff<Fq2>` base and
/// the CIOS accumulator of an `Fq` multiply, so a wide workgroup runs out of registers and
/// the occupancy that a wide workgroup is supposed to buy never arrives. The same kernel over
/// G1 would hold half as much and may well want a different size, which is exactly why this
/// is a parameter.
///
/// `msm_merge_g2` inverts: it wants 128, 2.4% over Metal's 64. It walks a bucket's slice range
/// in one lane and does almost nothing per lane on a bucket with no spills, so it is a
/// scheduling problem rather than a register one and the two kernels disagree for the same
/// reason `msm_count` and `msm_scatter` disagree.
///
/// Release only, and re-run under Tint at U14. The debug harness inverted `h_join`'s answer
/// once already, because differencing cancels the fixed cost of a submit and not the
/// per-encode host cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workgroups {
    pub clear: u32,
    pub segmented: u32,
    pub merge: u32,
    /// Threads in `msm_reduce_g2` and `msm_ones_g2`, which is also the length of the
    /// `array<PtG2, tg>` both hold in workgroup storage.
    ///
    /// **Design §4 says 32 and the measurement says 32, for a reason the design does not
    /// give.** §4's argument is occupancy: an `Xyzz<Fq2>` is 256 bytes, so `tg = 64` is 16384
    /// bytes, exactly the floor's whole workgroup allocation, and one workgroup is resident
    /// per core where `tg = 32` keeps two. That is the same "fill the budget" reasoning U6
    /// measured to be 5.1x **wrong** on the NTT, so it was swept rather than inherited.
    ///
    /// `tests/msm_g2.rs::the_reduction_threadgroup_is_measured`, 32,768 general scalars at
    /// `c = 12`, medians of five release runs, microseconds for `msm_reduce_g2` and
    /// `msm_ones_g2` alone:
    ///
    /// ```text
    /// tg    shared B   reduce us   ones us   sum us
    ///  8        2048       990.7     153.8   1144.5
    /// 16        4096       566.6     149.3    715.9
    /// 32        8192       412.4     147.8    560.2
    /// 64       16384       398.7     148.5    547.2
    /// ```
    ///
    /// **64 wins, by 2.3% on the pair.** The design's occupancy story is real but it is worth
    /// less than the halved tree depth and the halved per-thread segment: at `tg = 32` each
    /// thread reduces twice as many buckets serially, and that term dominates. So this is the
    /// second time the design's workgroup-storage reasoning has been measured and the second
    /// time it did not hold, in the opposite direction from the NTT.
    ///
    /// **32 ships anyway, and the reason is not the 2.3%.** At `tg = 64` the workgroup array
    /// is 16384 bytes, which is exactly `maxComputeWorkgroupStorageSize` at the floor with
    /// **zero** headroom: one more byte in either reduction, on any future platform that
    /// charges for anything else, and the pipeline fails to create in a browser rather than
    /// running slowly. 2.3% of 0.55 ms is 13 microseconds against a proof this backend is
    /// trying to bring under 200 ms, which is 0.007%, and it is not worth standing on a limit
    /// with nothing to spare. `tg = 64` is selectable, tested, and what a `Raised` profile
    /// should take; U14 should re-measure both in Chrome, where the array is charged by Tint
    /// and not by naga.
    pub tg: u32,
}

impl Default for Workgroups {
    fn default() -> Self {
        Self {
            clear: 256,
            segmented: 128,
            merge: 256,
            tg: 64,
        }
    }
}

impl Workgroups {
    /// Every 1D kernel at the same size, the reduction left at its shipped `tg`. For the
    /// sweep in `tests/msm_g2.rs`.
    pub fn uniform(n: u32) -> Self {
        Self {
            clear: n,
            segmented: n,
            merge: n,
            tg: Self::default().tg,
        }
    }

    /// The shipped sizes with `tg` forced, for the reduction sweep.
    pub fn with_tg(tg: u32) -> Self {
        Self {
            tg,
            ..Self::default()
        }
    }
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

/// The five G2 point kernels behind one copy of the `Fq` and `Fq2` prelude, at the measured
/// shape.
pub fn points_module(v: Variant) -> String {
    points_module_at(v, G2, Workgroups::default())
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
/// wire format is not `g16_gpu_layout::PackedG2Affine` and which no host packer produces.
pub fn points_module_at(v: Variant, c: Curve, wg: Workgroups) -> String {
    assert_eq!(
        v,
        Variant::Cios32Unrolled,
        "the point kernels read g16_gpu_layout::PackedG2Affine, which is 8 limbs of 32 bits; \
         variant {v:?} has a different wire format and no host packer"
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
    s.push_str(FQ2_OPS);
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

/// `AffG2` and `PtG2`, and nothing else: the coordinate type comes from the field prelude.
fn point_types(c: Curve) -> String {
    let (aff, pt, fty) = (c.aff, c.pt, c.fty);
    format!(
        "
// ---------------------------------------------------------------------------
// Points. {aff} is {base} bytes and matches g16_gpu_layout::PackedG2Affine exactly; {pt} is
// {point} bytes. (0, 0) is the affine point at infinity, which is unambiguous rather than a
// convention: BN254's G2 is y^2 = x^3 + 3/(9 + u), so (0, 0) is off curve.
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
fn {ENTRY_CLEAR}(@builtin(global_invocation_id) gid: vec3<u32>) {{
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
fn {ENTRY_SEGMENTED}(@builtin(global_invocation_id) gid: vec3<u32>) {{
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
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Fold each bucket's spilled partials into it. One thread per bucket, looking only at the
// slices its own run overlaps, which it computes from the run's start and end, so there is no
// search and no atomic. The worst case is count / slice_len additions for the fattest bucket,
// which turns g16-metal's 11,758-step serial loop into about 180.
@compute @workgroup_size({wg})
fn {ENTRY_MERGE}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = P.lo + gid.x;
    if (row >= P.n_windows * P.n_buckets) {{ return; }}
    let cnt = COUNTS[row];
    if (cnt == 0u) {{ return; }}
    let w = row / P.n_buckets;
    let base = w * P.cap;
    let start = CURSOR[row] - cnt - base;
    let end = CURSOR[row] - base;
    let k_lo = start / P.slice_len;
    let k_hi = k_lo; // PROBE

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
fn {ENTRY_REDUCE}(@builtin(workgroup_id) wid: vec3<u32>,
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
fn {ENTRY_ONES}(@builtin(workgroup_id) wid: vec3<u32>,
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
