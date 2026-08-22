//! Stages 5 to 9, back half: the five point kernels of one curve, their buffers, and the
//! host tail.
//!
//! [`crate::gen::points`] is the WGSL and carries the algorithm. This file owns the shapes:
//! how big a bucket array is, where the spill slots live, how many `ones` groups to run, and
//! the Horner combination the device deliberately does not do.
//!
//! # One implementation, two groups
//!
//! [`MsmPointsG1`] and [`MsmPointsG2`] are two instantiations of [`MsmPoints`], not two
//! copies of it. The only things that differ are the WGSL the generator emits, the byte
//! strides that come with it, and the arkworks projective type the readback decodes into,
//! which is what [`PointCurve`] carries. Four of a proof's five MSMs are G1 (A, B-G1, L and
//! H) and one is G2, so a bug fixed in one place is a bug fixed for all five.
//!
//! # What the device does and what the host does
//!
//! On the device: bucket accumulation over fixed-length slices, the spill merge, the
//! per-window reduction and the sum of the bases whose scalar is 1. On the host: the Horner
//! combination of `n_windows` points and the sum of at most 64 `ones` partials. That is at
//! most 84 curve additions per MSM against tens of millions on the device, and it is where
//! ZPrize, heliax and `g16-metal` all independently left the window tail, for the same
//! reason: compiling a shader containing `add_points` at the tail costs more than the
//! readback plus the additions.
//!
//! # One readback, and G2's 256-byte point makes it free
//!
//! `msm_reduce_*` writes `n_windows` points and `msm_ones_*` writes `ones_groups` points, and
//! both go into **one** buffer, so the whole result of an MSM is one `copy_buffer_to_buffer`
//! and one `mapAsync`. Design §3 puts the per-proof readback ceiling at 64 KiB across all
//! five MSMs and this is 20 + 8 points here, 7 KiB over G2 and 3.5 KiB over G1.
//!
//! The two bindings are windows into that one buffer at different offsets, and a storage
//! binding offset must be a multiple of `minStorageBufferOffsetAlignment`, which is **256** in
//! every browser and never improves. An `Xyzz<Fq2>` is exactly 256 bytes, so
//! `n_windows * 256` is aligned for free. It is not free for G1, whose point is 128 bytes, so
//! [`PointBuffers::new`] rounds the offset up rather than relying on the coincidence.
//!
//! # Nothing here trusts a buffer to be the right size
//!
//! WebGPU returns zero for an out-of-range storage read and drops an out-of-range storage
//! write, both silently, so a short bucket array is a wrong point and not an error. Every
//! bind function checks every buffer against the plan it was built from, and
//! `tests/msm_g1.rs` and `tests/msm_g2.rs` pre-fill every output with a sentinel and assert
//! the slack survives.

use std::marker::PhantomData;

use ark_ff::AdditiveGroup;
use g16_core::ProveError;
use g16_field::{Fq, Fq2, G1Projective, G2Projective, Zero};
use g16_gpu_layout::{PackedFq, PackedFq2, LIMBS};

use crate::device::{bad, WgpuBackend};
use crate::gather::storage_entry;
use crate::gen::field::Variant;
use crate::gen::msm::BIND_PARAMS;
use crate::gen::points as wgsl;
use crate::msm::{storage_buffer, DigitPlan, MsmParams};
use crate::params::ParamRing;
use crate::pipelines::Kernels;

/// `u32` words in one `Xyzz<Fq>`: four `Fq` of eight limbs, 128 bytes.
const XYZZ_G1_WORDS: usize = 4 * LIMBS;
/// `u32` words in one `Xyzz<Fq2>`: four `Fq2` of two `Fq` of eight limbs, 256 bytes.
const XYZZ_G2_WORDS: usize = 4 * 2 * LIMBS;

/// A storage binding offset must be a multiple of this, in every browser, at every tier,
/// forever. The design calls it physics.
const BINDING_ALIGN: u64 = 256;

// ---------------------------------------------------------------------------
// Slice length and ones groups
// ---------------------------------------------------------------------------

// The slice length lives on `gen::points::Curve`, because it is measured per curve and a
// plan that took one curve's constant against another curve's module would be a silently
// wrong point rather than an error. `Curve::slice_len` carries both sweeps.

/// Workgroups in `msm_ones_*`: enough that a 2^18-long witness gives each thread about sixty
/// scalars to scan, few enough that the host adds a few dozen points.
///
/// Capped at 64 because every group is a point in the readback and a point the host adds
/// serially, and floored at 1 because a dispatch of zero workgroups writes nothing and the
/// host would then read a stale slot.
fn ones_groups_for(n: u32, tg: u32) -> u32 {
    n.div_ceil(tg * 64).clamp(1, 64)
}

// ---------------------------------------------------------------------------
// The shape of one MSM's point stages
// ---------------------------------------------------------------------------

/// Everything the point kernels need that the [`DigitPlan`] does not already say.
///
/// Derived once from the digit plan and the base offset, so the five kernels, the five
/// allocations and the host tail cannot disagree about `slices` in particular, which appears
/// in the segmented pass's thread count, the spill array's length and the merge's slot
/// arithmetic. Getting it wrong in one of the three is a silently wrong point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointPlan {
    base_off: u32,
    slice_len: u32,
    slices: u32,
    ones_groups: u32,
    tg: u32,
}

impl PointPlan {
    /// Plans the point stages for `digits`, reading bases from `base_off`, at `curve`'s
    /// measured slice length.
    ///
    /// `tg` is the reduction workgroup size, which sets `ones_groups`. It must be the `tg` the
    /// module was generated at; [`MsmPoints::plan_points`] passes its own and its own curve,
    /// so a caller who goes through the module cannot get either pair wrong.
    pub fn new(
        digits: &DigitPlan,
        base_off: u32,
        tg: u32,
        curve: wgsl::Curve,
    ) -> Result<Self, ProveError> {
        Self::with_slice_len(digits, base_off, tg, curve.slice_len)
    }

    /// Same, at a forced slice length. For the sweeps in `tests/msm_g1.rs`.
    pub fn with_slice_len(
        digits: &DigitPlan,
        base_off: u32,
        tg: u32,
        slice_len: u32,
    ) -> Result<Self, ProveError> {
        if slice_len == 0 {
            return Err(bad(
                "slice_len 0 would give every thread no work and no bound",
            ));
        }
        if tg == 0 {
            return Err(bad("a reduction workgroup of 0 threads is not a thing"));
        }
        // At least one slice per window even when the window is empty: the segmented pass
        // tags its spill slots before it tests for an empty window, and a window with no
        // slices would leave the merge reading whatever the pool last put there.
        let slices = digits.cap().div_ceil(slice_len).max(1);
        // The spill array is indexed by `2 * (w * slices + k)`, so this product has to fit a
        // u32 before anything downstream can size a buffer from it.
        let slots = u64::from(digits.n_windows()) * u64::from(slices) * 2;
        if slots > u64::from(u32::MAX) {
            return Err(bad(format!(
                "{} windows x {slices} slices x 2 spill slots overflows a u32 index; \
                 slice_len {slice_len} is too small for a cap of {}",
                digits.n_windows(),
                digits.cap()
            )));
        }
        Ok(Self {
            base_off,
            slice_len,
            slices,
            ones_groups: ones_groups_for(digits.n(), tg),
            tg,
        })
    }

    pub fn base_off(&self) -> u32 {
        self.base_off
    }
    pub fn slice_len(&self) -> u32 {
        self.slice_len
    }
    pub fn slices(&self) -> u32 {
        self.slices
    }
    pub fn ones_groups(&self) -> u32 {
        self.ones_groups
    }
    pub fn tg(&self) -> u32 {
        self.tg
    }

    /// Threads the segmented pass dispatches: one per slice per window.
    pub fn seg_threads(&self, digits: &DigitPlan) -> u32 {
        digits.n_windows() * self.slices
    }

    /// Spill slots: two per slice per window, because at most one run of a slice continues
    /// backwards and at most one continues forwards.
    pub fn spill_slots(&self, digits: &DigitPlan) -> u32 {
        2 * self.seg_threads(digits)
    }

    /// The full parameter block, digit fields and point fields together.
    ///
    /// `n` is the element count the *dispatching* kernel's guard uses, and `lo` is where this
    /// dispatch starts. Only `msm_ones_*` reads `n`; the other four compute their own domain
    /// from `n_windows`, `n_buckets` and `slices`, exactly as the Metal originals do.
    pub fn params(&self, digits: &DigitPlan, lo: u32) -> MsmParams {
        MsmParams {
            base_off: self.base_off,
            ones_groups: self.ones_groups,
            slice_len: self.slice_len,
            slices: self.slices,
            ..digits.params(digits.n(), lo)
        }
    }
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// A debug label carrying the curve, so a validation error or a capture names the right one
/// of the two MSM families.
fn label(curve: wgsl::Curve, what: &str) -> String {
    format!("g16 msm {what} {}", curve.suffix)
}

/// The four device allocations one MSM's point stages need.
///
/// `results` is one buffer holding both outputs, because that makes the whole readback one
/// copy. See the module docs for why the offset is aligned rather than assumed aligned.
pub struct PointBuffers {
    pub buckets: wgpu::Buffer,
    pub spill_pts: wgpu::Buffer,
    pub spill_rows: wgpu::Buffer,
    pub results: wgpu::Buffer,
    /// Byte offset of the `ones` partials inside [`Self::results`].
    ones_off: u64,
    ones_groups: u32,
}

impl PointBuffers {
    /// Allocates for `(digits, points)`, plus `slack` extra elements on every output.
    ///
    /// `slack` exists for one reason and it is a test: every output here is exactly the size
    /// the kernel should write, and WebGPU drops an out-of-range storage write in silence, so
    /// without slack an over-run is *unobservable*. `tests/msm_g1.rs` and `tests/msm_g2.rs`
    /// allocate it and check it survives; the prover passes 0.
    pub fn new(
        backend: &WgpuBackend,
        digits: &DigitPlan,
        points: &PointPlan,
        slack: u32,
        curve: wgsl::Curve,
    ) -> Result<Self, ProveError> {
        let pt = curve.point_bytes;
        let rows = u64::from(digits.rows() + slack);
        let slots = u64::from(points.spill_slots(digits) + slack);
        let ones_groups = points.ones_groups;

        // The one error a caller can hit that is about the curve rather than about being
        // greedy: G2's 256-byte accumulators halve the window width the floor's storage
        // binding allows. c = 16 is 16 x 32768 x 256 = 134.2 MB against a 128 MiB binding,
        // where the same c over G1 is 67.1 MB and fits with room to spare.
        let bucket_bytes = rows * pt;
        let limit = backend.granted_limits().max_storage_buffer_binding_size;
        if bucket_bytes > limit {
            return Err(bad(format!(
                "a {} window x {} bucket array of {pt}-byte {} accumulators is {bucket_bytes} \
                 bytes, over the {limit} byte storage binding limit. c = {} is too wide for \
                 this curve; {}.",
                digits.n_windows(),
                digits.n_buckets(),
                curve.pt,
                digits.c(),
                curve.headroom
            )));
        }

        let ones_off = u64::from(digits.n_windows()) * pt;
        let ones_off = ones_off.div_ceil(BINDING_ALIGN) * BINDING_ALIGN;
        let results_bytes = ones_off + u64::from(ones_groups + slack) * pt;

        Ok(Self {
            buckets: storage_buffer(backend, &label(curve, "buckets"), bucket_bytes)?,
            spill_pts: storage_buffer(backend, &label(curve, "spill points"), slots * pt)?,
            spill_rows: storage_buffer(backend, &label(curve, "spill rows"), slots * 4)?,
            results: storage_buffer(backend, &label(curve, "results"), results_bytes)?,
            ones_off,
            ones_groups,
        })
    }

    /// Total device bytes, for a report.
    pub fn bytes(&self) -> u64 {
        self.buckets.size() + self.spill_pts.size() + self.spill_rows.size() + self.results.size()
    }

    /// Byte offset of the `ones` partials inside [`Self::results`].
    pub fn ones_offset(&self) -> u64 {
        self.ones_off
    }

    /// Bytes the host has to read back: both outputs, in one range starting at zero.
    pub fn results_bytes(&self, curve: wgsl::Curve) -> u64 {
        self.ones_off + u64::from(self.ones_groups) * curve.point_bytes
    }
}

// ---------------------------------------------------------------------------
// The pipelines
// ---------------------------------------------------------------------------

/// Which BN254 group an [`MsmPoints`] is over: the WGSL shape, and the arkworks type the
/// readback decodes into.
///
/// A trait rather than a runtime enum because [`MsmPoints::combine`] has to *return* the
/// right projective type. A runtime curve field would make that `Result<Either<..>>` at every
/// call site in the prover, and the whole point of one implementation is that the prover's
/// four G1 MSMs and its one G2 MSM read identically.
pub trait PointCurve {
    /// The generated WGSL's shape and this curve's measured constants.
    const WGSL: wgsl::Curve;
    /// The arkworks projective type one window sum decodes into.
    type Projective: AdditiveGroup;
    /// One `Xyzz<F>` as the kernel wrote it to a projective point, with no field inversion.
    fn from_xyzz_bytes(raw: &[u8]) -> Result<Self::Projective, ProveError>;
}

/// BN254's G1: the A, B-G1, L and H MSMs, four of a proof's five.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G1Curve {}

/// BN254's G2: the B-G2 MSM, one of a proof's five and the expensive one per point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G2Curve {}

impl PointCurve for G1Curve {
    const WGSL: wgsl::Curve = wgsl::G1;
    type Projective = G1Projective;
    fn from_xyzz_bytes(raw: &[u8]) -> Result<G1Projective, ProveError> {
        xyzz_g1_from_bytes(raw)
    }
}

impl PointCurve for G2Curve {
    const WGSL: wgsl::Curve = wgsl::G2;
    type Projective = G2Projective;
    fn from_xyzz_bytes(raw: &[u8]) -> Result<G2Projective, ProveError> {
        xyzz_g2_from_bytes(raw)
    }
}

/// One curve's five point entry points, their bind group layouts and their pipelines.
///
/// One shader module, five pipeline layouts, one bind group layout per entry point. One
/// module because U8 measured the split and it costs 17% cold for five entry points of the
/// same shape; one layout per entry point because giving all five the union of their bindings
/// would put every kernel at the widest one's set of six and make the count meaningless as a
/// check against the browser floor's eight.
pub struct MsmPoints<C: PointCurve> {
    kernels: Kernels,
    curve: wgsl::Curve,
    /// The five entry point names, `msm_clear_g1` and so on, built once from the suffix.
    /// Owned rather than `&'static str` because they are derived from the curve and not
    /// written out, which is what stopped the generator being G2-only.
    names: [String; 5],
    wg: wgsl::Workgroups,
    workgroups_per_dispatch: u32,
    bgl_clear: wgpu::BindGroupLayout,
    bgl_segmented: wgpu::BindGroupLayout,
    bgl_merge: wgpu::BindGroupLayout,
    bgl_reduce: wgpu::BindGroupLayout,
    bgl_ones: wgpu::BindGroupLayout,
    _layouts: Vec<wgpu::PipelineLayout>,
    source_len: usize,
    _curve: PhantomData<C>,
}

/// The G1 point stages: the A, B-G1, L and H MSMs.
pub type MsmPointsG1 = MsmPoints<G1Curve>;
/// The G2 point stages: the B-G2 MSM.
pub type MsmPointsG2 = MsmPoints<G2Curve>;

/// Indices into [`MsmPoints::names`], in dispatch order.
const CLEAR: usize = 0;
const SEGMENTED: usize = 1;
const MERGE: usize = 2;
const REDUCE: usize = 3;
const ONES: usize = 4;

impl<C: PointCurve> MsmPoints<C> {
    /// Compiles at this curve's measured shape and this device's workgroup-per-dimension
    /// limit.
    pub fn new(backend: &WgpuBackend) -> Result<Self, ProveError> {
        let max_wg = backend
            .granted_limits()
            .max_compute_workgroups_per_dimension;
        Self::with_shape(backend, C::WGSL.wg, max_wg)
    }

    /// Same, with the workgroup sizes and the per-dispatch workgroup cap forced.
    ///
    /// Public for the tests that cannot be written otherwise: the workgroup sweep, the `tg`
    /// sweep that made [`wgsl::Workgroups::tg`]'s table, and the multi-dispatch path, which no
    /// artifact reaches because one dispatch covers 8.4M rows at 128 threads. A code path no
    /// test can reach is a code path that ships untested.
    pub fn with_shape(
        backend: &WgpuBackend,
        wg: wgsl::Workgroups,
        workgroups_per_dispatch: u32,
    ) -> Result<Self, ProveError> {
        let curve = C::WGSL;
        let limits = backend.granted_limits();
        let max_inv = limits.max_compute_invocations_per_workgroup;
        for (name, n) in [
            ("clear", wg.clear),
            ("segmented", wg.segmented),
            ("merge", wg.merge),
            ("tg", wg.tg),
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
        // The check the accumulator width makes interesting: `array<Pt, tg>` at 256 bytes a
        // point over G2 and 128 over G1, so the same tg costs the two curves different
        // budgets and G1 can afford twice the reduction width. Reported here against the
        // *granted* limit and by the generator against the floor, which are 32768 and 16384
        // on this adapter, so a `Raised` device would otherwise pass this and then panic
        // inside the generator.
        let shared = curve.workgroup_bytes(wg.tg);
        let ceiling =
            u64::from(limits.max_compute_workgroup_storage_size).min(wgsl::FLOOR_WORKGROUP_BYTES);
        if shared > ceiling {
            return Err(bad(format!(
                "{} at {} threads holds {shared} bytes of workgroup storage, over the \
                 {ceiling} byte ceiling (the smaller of this device's {} and the browser \
                 floor's {})",
                curve.entry_reduce(),
                wg.tg,
                limits.max_compute_workgroup_storage_size,
                wgsl::FLOOR_WORKGROUP_BYTES
            )));
        }

        let device = backend.device();
        let mk_bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let names = curve.entries();
        let bgl_clear = mk_bgl(&names[CLEAR], &Self::clear_entries());
        let bgl_segmented = mk_bgl(&names[SEGMENTED], &Self::segmented_entries());
        let bgl_merge = mk_bgl(&names[MERGE], &Self::merge_entries());
        let bgl_reduce = mk_bgl(&names[REDUCE], &Self::reduce_entries());
        let bgl_ones = mk_bgl(&names[ONES], &Self::ones_entries());

        let mk_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let l_clear = mk_layout(&names[CLEAR], &bgl_clear);
        let l_segmented = mk_layout(&names[SEGMENTED], &bgl_segmented);
        let l_merge = mk_layout(&names[MERGE], &bgl_merge);
        let l_reduce = mk_layout(&names[REDUCE], &bgl_reduce);
        let l_ones = mk_layout(&names[ONES], &bgl_ones);

        let src = wgsl::points_module_at(Variant::default(), curve, wg);
        let source_len = src.len();
        let kernels = Kernels::build_with_layouts(
            backend,
            &format!("msm_points_{}", curve.suffix),
            &src,
            &[
                (names[CLEAR].as_str(), &l_clear),
                (names[SEGMENTED].as_str(), &l_segmented),
                (names[MERGE].as_str(), &l_merge),
                (names[REDUCE].as_str(), &l_reduce),
                (names[ONES].as_str(), &l_ones),
            ],
        )?;

        Ok(Self {
            kernels,
            curve,
            names,
            wg,
            workgroups_per_dispatch,
            bgl_clear,
            bgl_segmented,
            bgl_merge,
            bgl_reduce,
            bgl_ones,
            _layouts: vec![l_clear, l_segmented, l_merge, l_reduce, l_ones],
            source_len,
            _curve: PhantomData,
        })
    }

    // ---- bind group layouts, as data, so a test can count what the device enforces ----

    pub fn clear_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            ParamRing::layout_entry(BIND_PARAMS),
            storage_entry(wgsl::BIND_BUCKETS, false),
        ]
    }
    pub fn segmented_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
        [
            ParamRing::layout_entry(BIND_PARAMS),
            storage_entry(wgsl::BIND_ENTRIES, true),
            storage_entry(wgsl::BIND_BASES, true),
            storage_entry(wgsl::BIND_CURSOR, true),
            storage_entry(wgsl::BIND_BUCKETS, false),
            storage_entry(wgsl::BIND_SPILL_PTS, false),
            storage_entry(wgsl::BIND_SPILL_ROWS, false),
        ]
    }
    /// Five, and the two spill bindings are declared writable although the kernel only reads
    /// them: one WGSL declaration serves both entry points and its access mode is fixed at
    /// module scope, so the layout has to match the declaration and not the use.
    pub fn merge_entries() -> [wgpu::BindGroupLayoutEntry; 6] {
        [
            ParamRing::layout_entry(BIND_PARAMS),
            storage_entry(wgsl::BIND_BUCKETS, false),
            storage_entry(wgsl::BIND_SPILL_PTS, false),
            storage_entry(wgsl::BIND_SPILL_ROWS, false),
            storage_entry(wgsl::BIND_COUNTS, true),
            storage_entry(wgsl::BIND_CURSOR, true),
        ]
    }
    pub fn reduce_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
        [
            ParamRing::layout_entry(BIND_PARAMS),
            storage_entry(wgsl::BIND_BUCKETS, false),
            storage_entry(wgsl::BIND_WSUMS, false),
        ]
    }
    pub fn ones_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
        [
            ParamRing::layout_entry(BIND_PARAMS),
            storage_entry(wgsl::BIND_SCALARS, true),
            storage_entry(wgsl::BIND_BASES, true),
            storage_entry(wgsl::BIND_ONES, false),
        ]
    }

    /// Storage buffers each entry point's pipeline layout declares, counted from the lists
    /// above rather than from a duplicate. All five must be at most 8, the browser floor.
    ///
    /// Curve-independent by construction: the bindings are the same eleven resources over
    /// either group, only their element type changes. It is still reported per instantiation
    /// so a test names the entry point it is talking about.
    pub fn storage_buffer_counts() -> [(String, u32); 5] {
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
            (C::WGSL.entry_clear(), count(&Self::clear_entries())),
            (C::WGSL.entry_segmented(), count(&Self::segmented_entries())),
            (C::WGSL.entry_merge(), count(&Self::merge_entries())),
            (C::WGSL.entry_reduce(), count(&Self::reduce_entries())),
            (C::WGSL.entry_ones(), count(&Self::ones_entries())),
        ]
    }

    // ---- bind groups ----

    fn bind(
        &self,
        backend: &WgpuBackend,
        label: &str,
        bgl: &wgpu::BindGroupLayout,
        ring: &ParamRing,
        res: Vec<(u32, wgpu::BindingResource<'_>)>,
    ) -> wgpu::BindGroup {
        let mut entries = Vec::with_capacity(res.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: BIND_PARAMS,
            resource: wgpu::BindingResource::Buffer(ring.binding()),
        });
        for (binding, resource) in res {
            entries.push(wgpu::BindGroupEntry { binding, resource });
        }
        backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bgl,
                entries: &entries,
            })
    }

    /// The five bind groups one MSM dispatches through, every buffer length checked.
    ///
    /// One function rather than five, because the checks are the interesting part and a
    /// caller that can build four of five bind groups is a caller that can forget the fifth.
    /// That is also why it takes nine arguments: splitting it to satisfy the lint would put
    /// the length checks somewhere a caller can skip.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_all(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        digits: &DigitPlan,
        points: &PointPlan,
        scalars: &wgpu::Buffer,
        bases: &wgpu::Buffer,
        sort: &crate::msm::DigitBuffers,
        bufs: &PointBuffers,
    ) -> Result<PointBinds, ProveError> {
        if points.tg != self.wg.tg {
            return Err(bad(format!(
                "this plan was built for a {}-thread reduction and the module was generated \
                 at {}; ones_groups would not match the stride the kernel walks",
                points.tg, self.wg.tg
            )));
        }
        let pt = self.curve.point_bytes;
        let need = |what: &str, buf: &wgpu::Buffer, want: u64| -> Result<(), ProveError> {
            if buf.size() < want {
                return Err(bad(format!(
                    "{what} is {} bytes, this plan needs {want}",
                    buf.size()
                )));
            }
            Ok(())
        };
        let rows = u64::from(digits.rows());
        let slots = u64::from(points.spill_slots(digits));
        need("the bucket array", &bufs.buckets, rows * pt)?;
        need("the spill point array", &bufs.spill_pts, slots * pt)?;
        need("the spill row array", &bufs.spill_rows, slots * 4)?;
        need(
            "the entry array",
            &sort.entries,
            u64::from(digits.entries()) * 8,
        )?;
        need("the counts array", &sort.counts, rows * 4)?;
        need("the cursor array", &sort.cursor, rows * 4)?;
        need(
            "the results buffer",
            &bufs.results,
            bufs.results_bytes(self.curve),
        )?;
        // The base range this MSM reads. `msm_ones_g2` walks base_off .. base_off + n and the
        // segmented pass reads base_off + (entry >> 1), whose largest value is base_off + n - 1
        // because msm_scatter stores the index within this MSM.
        let base_end = u64::from(points.base_off) + u64::from(digits.n());
        need("the base vector", bases, base_end * self.curve.base_bytes)?;
        need(
            "the scalar buffer",
            scalars,
            u64::from(digits.scalar_off() + digits.n()) * (LIMBS as u64) * 4,
        )?;

        fn whole(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
            b.as_entire_binding()
        }
        fn window(b: &wgpu::Buffer, off: u64, len: u64) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: b,
                offset: off,
                size: std::num::NonZeroU64::new(len),
            })
        }

        Ok(PointBinds {
            clear: self.bind(
                backend,
                &self.names[CLEAR],
                &self.bgl_clear,
                ring,
                vec![(wgsl::BIND_BUCKETS, whole(&bufs.buckets))],
            ),
            segmented: self.bind(
                backend,
                &self.names[SEGMENTED],
                &self.bgl_segmented,
                ring,
                vec![
                    (wgsl::BIND_ENTRIES, whole(&sort.entries)),
                    (wgsl::BIND_BASES, whole(bases)),
                    (wgsl::BIND_CURSOR, whole(&sort.cursor)),
                    (wgsl::BIND_BUCKETS, whole(&bufs.buckets)),
                    (wgsl::BIND_SPILL_PTS, whole(&bufs.spill_pts)),
                    (wgsl::BIND_SPILL_ROWS, whole(&bufs.spill_rows)),
                ],
            ),
            merge: self.bind(
                backend,
                &self.names[MERGE],
                &self.bgl_merge,
                ring,
                vec![
                    (wgsl::BIND_BUCKETS, whole(&bufs.buckets)),
                    (wgsl::BIND_SPILL_PTS, whole(&bufs.spill_pts)),
                    (wgsl::BIND_SPILL_ROWS, whole(&bufs.spill_rows)),
                    (wgsl::BIND_COUNTS, whole(&sort.counts)),
                    (wgsl::BIND_CURSOR, whole(&sort.cursor)),
                ],
            ),
            // Two windows into one buffer. The kernel indexes each from zero, so the offsets
            // are what keep the two outputs apart, and they are 256-byte aligned by
            // construction. See the module docs.
            reduce: self.bind(
                backend,
                &self.names[REDUCE],
                &self.bgl_reduce,
                ring,
                vec![
                    (wgsl::BIND_BUCKETS, whole(&bufs.buckets)),
                    (
                        wgsl::BIND_WSUMS,
                        window(&bufs.results, 0, u64::from(digits.n_windows()) * pt),
                    ),
                ],
            ),
            ones: self.bind(
                backend,
                &self.names[ONES],
                &self.bgl_ones,
                ring,
                vec![
                    (wgsl::BIND_SCALARS, whole(scalars)),
                    (wgsl::BIND_BASES, whole(bases)),
                    (
                        wgsl::BIND_ONES,
                        window(
                            &bufs.results,
                            bufs.ones_off,
                            u64::from(points.ones_groups) * pt,
                        ),
                    ),
                ],
            ),
        })
    }

    // ---- parameter blocks ----

    /// The point plan for this module: this curve's measured slice length and this module's
    /// reduction width, so the two things [`PointPlan`] cannot check for itself come from the
    /// module that will run it.
    pub fn plan_points(&self, digits: &DigitPlan, base_off: u32) -> Result<PointPlan, ProveError> {
        PointPlan::new(digits, base_off, self.wg.tg, self.curve)
    }

    fn dispatches(&self, n: u32, wg: u32) -> u32 {
        n.div_ceil(wg * self.workgroups_per_dispatch).max(1)
    }

    fn span(&self, wg: u32) -> u32 {
        wg * self.workgroups_per_dispatch
    }

    /// Parameter blocks for the whole point stage, in dispatch order.
    ///
    /// Separate from [`Self::encode`] because design §3 writes every parameter block for the
    /// whole proof in one `write_buffer` before encoding starts, so the pushes have to happen
    /// before the compute pass exists.
    pub fn plan(
        &self,
        digits: &DigitPlan,
        points: &PointPlan,
        ring: &mut ParamRing,
    ) -> Result<PointOffsets, ProveError> {
        let rows = digits.rows();
        Ok(PointOffsets {
            clear: self.push_range(ring, digits, points, rows, self.wg.clear)?,
            segmented: self.push_range(
                ring,
                digits,
                points,
                points.seg_threads(digits),
                self.wg.segmented,
            )?,
            merge: self.push_range(ring, digits, points, rows, self.wg.merge)?,
            // One workgroup per window and per ones group, always one dispatch: n_windows is
            // at most 128 and ones_groups at most 64, against a 65535 limit.
            reduce: ring.push(&points.params(digits, 0))?,
            ones: ring.push(&points.params(digits, 0))?,
        })
    }

    fn push_range(
        &self,
        ring: &mut ParamRing,
        digits: &DigitPlan,
        points: &PointPlan,
        n: u32,
        wg: u32,
    ) -> Result<Vec<u32>, ProveError> {
        let span = self.span(wg);
        let mut offsets = Vec::with_capacity(self.dispatches(n, wg) as usize);
        let mut lo = 0u32;
        loop {
            offsets.push(ring.push(&points.params(digits, lo))?);
            lo = lo.saturating_add(span);
            if lo >= n {
                break;
            }
        }
        Ok(offsets)
    }

    /// Ring slots the point stage consumes for this plan, which is also its dispatch count.
    /// Five at every shape any artifact reaches.
    pub fn slots(&self, digits: &DigitPlan, points: &PointPlan) -> u32 {
        self.dispatches(digits.rows(), self.wg.clear)
            + self.dispatches(points.seg_threads(digits), self.wg.segmented)
            + self.dispatches(digits.rows(), self.wg.merge)
            + 2
    }

    // ---- encoding ----

    /// All five dispatches into one pass, in order.
    ///
    /// WebGPU orders dispatches inside a compute pass and makes each one's writes visible to
    /// the next with no explicit barrier, which is what lets clear, segmented, merge, reduce
    /// and ones share a pass with the counting sort that feeds them. Metal needs an explicit
    /// `memoryBarrier` for the same thing.
    pub fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        digits: &DigitPlan,
        points: &PointPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        self.encode_clear(pass, digits, binds, offsets)?;
        self.encode_segmented(pass, digits, points, binds, offsets)?;
        self.encode_merge(pass, digits, binds, offsets)?;
        self.encode_reduce(pass, digits, binds, offsets)?;
        self.encode_ones(pass, points, binds, offsets)
    }

    /// The five separately, so a test can run four of them and look at what the fourth wrote.
    /// Reading the bucket array after the merge is the only way to tell a correct segmented
    /// pass from a wrong one whose spills the merge happens to undo.
    pub fn encode_clear(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        digits: &DigitPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            &self.names[CLEAR],
            &binds.clear,
            digits.rows(),
            self.wg.clear,
            &offsets.clear,
        )
    }

    pub fn encode_segmented(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        digits: &DigitPlan,
        points: &PointPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            &self.names[SEGMENTED],
            &binds.segmented,
            points.seg_threads(digits),
            self.wg.segmented,
            &offsets.segmented,
        )
    }

    pub fn encode_merge(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        digits: &DigitPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        self.encode_range(
            pass,
            &self.names[MERGE],
            &binds.merge,
            digits.rows(),
            self.wg.merge,
            &offsets.merge,
        )
    }

    /// One workgroup per window.
    pub fn encode_reduce(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        digits: &DigitPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        pass.set_pipeline(self.kernels.get(&self.names[REDUCE])?);
        pass.set_bind_group(0, &binds.reduce, &[offsets.reduce]);
        pass.dispatch_workgroups(digits.n_windows(), 1, 1);
        Ok(())
    }

    /// One workgroup per ones group.
    pub fn encode_ones(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        points: &PointPlan,
        binds: &PointBinds,
        offsets: &PointOffsets,
    ) -> Result<(), ProveError> {
        pass.set_pipeline(self.kernels.get(&self.names[ONES])?);
        pass.set_bind_group(0, &binds.ones, &[offsets.ones]);
        pass.dispatch_workgroups(points.ones_groups, 1, 1);
        Ok(())
    }

    fn encode_range(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        what: &str,
        bind: &wgpu::BindGroup,
        n: u32,
        wg: u32,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let want = self.dispatches(n, wg) as usize;
        if offsets.len() != want {
            return Err(bad(format!(
                "{what} over {n} elements needs {want} dispatches, got {} parameter offsets",
                offsets.len()
            )));
        }
        pass.set_pipeline(self.kernels.get(what)?);
        let span = self.span(wg);
        let mut lo = 0u32;
        for &off in offsets {
            let hi = lo.saturating_add(span).min(n);
            pass.set_bind_group(0, bind, &[off]);
            pass.dispatch_workgroups(hi.saturating_sub(lo).div_ceil(wg).max(1), 1, 1);
            lo = hi;
        }
        Ok(())
    }

    // ---- the host tail ----

    /// The Horner combination of the window sums plus the `ones` partials.
    ///
    /// `raw` is [`PointBuffers::results`] read back whole, so this is `n_windows` points
    /// followed by padding followed by `ones_groups` points. High window to low, `c` doublings
    /// between each, which is the same order `g16-msm` uses on the CPU, so the two agree bit
    /// for bit and not merely up to the group law.
    pub fn combine(
        &self,
        raw: &[u8],
        digits: &DigitPlan,
        points: &PointPlan,
        bufs: &PointBuffers,
    ) -> Result<C::Projective, ProveError> {
        let pt = self.curve.point_bytes as usize;
        let want = bufs.results_bytes(self.curve) as usize;
        if raw.len() < want {
            return Err(bad(format!(
                "the readback is {} bytes, {} windows plus {} ones groups need {want}",
                raw.len(),
                digits.n_windows(),
                points.ones_groups
            )));
        }
        let at =
            |i: usize| -> Result<C::Projective, ProveError> { C::from_xyzz_bytes(&raw[i..i + pt]) };
        let last = digits.n_windows() as usize - 1;
        let mut acc = at(last * pt)?;
        for k in (0..last).rev() {
            for _ in 0..digits.c() {
                acc.double_in_place();
            }
            acc += at(k * pt)?;
        }
        let ones = bufs.ones_off as usize;
        for g in 0..points.ones_groups as usize {
            acc += at(ones + g * pt)?;
        }
        Ok(acc)
    }

    // ---- reporting ----

    pub fn workgroups(&self) -> wgsl::Workgroups {
        self.wg
    }

    pub fn curve(&self) -> wgsl::Curve {
        self.curve
    }

    /// Generated WGSL, in bytes.
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    pub fn cost(&self) -> crate::pipelines::PrepareCost {
        self.kernels.cost()
    }

    pub fn summary(&self) -> String {
        self.kernels.summary()
    }
}

/// The five bind groups one MSM's point stages dispatch through.
pub struct PointBinds {
    pub clear: wgpu::BindGroup,
    pub segmented: wgpu::BindGroup,
    pub merge: wgpu::BindGroup,
    pub reduce: wgpu::BindGroup,
    pub ones: wgpu::BindGroup,
}

/// Dynamic offsets for one point stage, one group per kernel.
#[derive(Clone, Debug, Default)]
pub struct PointOffsets {
    pub clear: Vec<u32>,
    pub segmented: Vec<u32>,
    pub merge: Vec<u32>,
    pub reduce: u32,
    pub ones: u32,
}

impl PointOffsets {
    pub fn total(&self) -> usize {
        self.clear.len() + self.segmented.len() + self.merge.len() + 2
    }
}

// ---------------------------------------------------------------------------
// XYZZ to arkworks, with no field inversion
// ---------------------------------------------------------------------------

/// Limb group `j` of a flattened point: eight little-endian `u32` starting at word `8 * j`.
///
/// Exactly `g16_gpu_layout::PackedFq`, which is what makes these decoders the inverse of the
/// packing the bases went out in rather than a second opinion about the wire format.
fn limb_at(raw: &[u8], j: usize) -> PackedFq {
    let mut v = [0u32; LIMBS];
    for (k, word) in v.iter_mut().enumerate() {
        let at = (j * LIMBS + k) * 4;
        *word = u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]);
    }
    PackedFq { v }
}

/// One `Xyzz<Fq>` as the kernel wrote it, 128 bytes, to a projective point.
///
/// # Why there is no inversion
///
/// XYZZ carries the invariant `ZZ^3 = ZZZ^2`, so setting the Jacobian `Z = ZZZ` gives
/// `Z^2 = ZZ^3` and the point `(X * ZZ^2, Y * ZZ^3, ZZZ)` has `x = X*ZZ^2 / ZZ^3 = X/ZZ` and
/// `y = Y*ZZ^3 / ZZZ^3 = Y/ZZZ`, which is exactly the XYZZ point. Three multiplications
/// against two inversions through affine, and there are up to 84 of these per MSM.
///
/// `new_unchecked` is deliberate. These come back from a kernel that was handed on-curve
/// inputs, and re-checking the curve equation on every window sum would cost a subgroup check
/// per point for no information: a bug in the kernel shows up as a wrong MSM, which every
/// test here compares against the CPU Pippenger.
pub fn xyzz_g1_from_bytes(raw: &[u8]) -> Result<G1Projective, ProveError> {
    if raw.len() < XYZZ_G1_WORDS * 4 {
        return Err(bad(format!(
            "an Xyzz<Fq> is {} bytes and this slice is {}",
            XYZZ_G1_WORDS * 4,
            raw.len()
        )));
    }
    // x, y, zz, zzz, each eight little-endian u32 of Montgomery Fq.
    let coord = |i: usize| -> Fq { limb_at(raw, i).to_fq() };
    let (x, y, zz, zzz) = (coord(0), coord(1), coord(2), coord(3));
    if zz.is_zero() {
        return Ok(G1Projective::zero());
    }
    let zz2 = zz * zz;
    Ok(G1Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz))
}

/// One `Xyzz<Fq2>` as the kernel wrote it, 256 bytes, to a projective point.
///
/// See [`xyzz_g1_from_bytes`] for why there is no inversion here and no curve check; the only
/// difference is that each coordinate is two `Fq` rather than one.
pub fn xyzz_g2_from_bytes(raw: &[u8]) -> Result<G2Projective, ProveError> {
    if raw.len() < XYZZ_G2_WORDS * 4 {
        return Err(bad(format!(
            "an Xyzz<Fq2> is {} bytes and this slice is {}",
            XYZZ_G2_WORDS * 4,
            raw.len()
        )));
    }
    // The flattened 32-word point is x.c0, x.c1, y.c0, y.c1, zz.c0, zz.c1, zzz.c0, zzz.c1.
    let coord = |i: usize| -> Fq2 {
        PackedFq2 {
            c0: limb_at(raw, i * 2),
            c1: limb_at(raw, i * 2 + 1),
        }
        .to_fq2()
    };
    let (x, y, zz, zzz) = (coord(0), coord(1), coord(2), coord(3));
    if zz.is_zero() {
        return Ok(G2Projective::zero());
    }
    let zz2 = zz * zz;
    Ok(G2Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz))
}
