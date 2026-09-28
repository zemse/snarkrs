//! WGSL source generation.
//!
//! Every kernel this backend runs is emitted from Rust at run time, not shipped as a
//! `.wgsl` file. [`field`] explains why in detail; the short version is that
//! the limb-width sweep measured a 3.8x penalty for indexing a
//! function-scope `array<u32, N>` by a loop variable, so every limb loop has to be
//! straight-line code with literal indices, and nobody is hand-writing 8 rounds x 8 limbs
//! of that twice (once for `Fr`, once for `Fq`) and keeping it right.
//!
//! [`gather`] is stage 0's generator, [`ntt`] is stages 1 to 3's, [`pointwise`] is
//! stage 4's standalone reference kernel, the one the fused NTT epilogue is checked
//! against, and [`msm`] is the front half of stages 5 to 9, the counting sort that decides
//! which bucket every point lands in, and [`points`] is the back half: the XYZZ curve
//! arithmetic and the five point kernels that turn a sorted entry array into one point per
//! window, emitted once per curve from [`points::Curve`]. U10 wrote it for G2 and U9
//! instantiated [`points::G1`] from the same lines.
//! They all put a field prelude in front
//! of their own entry points: [`field::field_module`] when they need `Fq` and `Fq2`, or just `FR.ops` when,
//! like the NTT, they only ever touch the scalar field.

/// `maxComputeInvocationsPerWorkgroup` at the browser floor.
///
/// Every generator in this module asserts against this rather than against what the adapter
/// grants, because the source it emits has to compile in a stock browser and this M2 Max
/// grants 1024. It lives here, once, so that the host validators in `crate::gather`,
/// `crate::ntt`, `crate::pointwise`, `crate::msm` and `crate::points` can read the same
/// number through [`crate::device::WgpuBackend::ceiling_invocations`]. They used to compare
/// against the granted limit instead, which meant a `with_shape` call under
/// `G16_WGPU_LIMITS=raised` could pass its own check and then panic inside the generator
/// rather than returning a `ProveError`.
pub const FLOOR_INVOCATIONS: u32 = 256;

/// `maxComputeWorkgroupStorageSize` at the browser floor, in bytes. Same reasoning as
/// [`FLOOR_INVOCATIONS`]: this adapter grants 32768 and no browser does.
pub const FLOOR_WORKGROUP_BYTES: u64 = 16384;

pub mod field;
pub mod gather;
pub mod msm;
pub mod ntt;
pub mod points;
pub mod pointwise;

pub use field::{field_module, Field, Variant, FQ, FQ2_OPS, FR, MUL64};
pub use gather::gather_module;
pub use msm::{digits_module, fused_module_at, mont_module, LimbPick, Workgroups};
pub use ntt::{ntt_module, Mode};
pub use points::{points_module, Curve, G1, G2};
pub use pointwise::h_join_module;
