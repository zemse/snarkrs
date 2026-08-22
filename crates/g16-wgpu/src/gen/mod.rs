//! WGSL source generation.
//!
//! Every kernel this backend runs is emitted from Rust at run time, not shipped as a
//! `.wgsl` file. [`field`] explains why in detail; the short version is that
//! the limb-width sweep measured a 3.8x penalty for indexing a
//! function-scope `array<u32, N>` by a loop variable, so every limb loop has to be
//! straight-line code with literal indices, and nobody is hand-writing 8 rounds x 8 limbs
//! of that twice (once for `Fr`, once for `Fq`) and keeping it right.
//!
//! [`gather`] is stage 0's generator, [`ntt`] is stages 1 to 3's, and [`pointwise`] is
//! stage 4's standalone reference kernel, the one the fused NTT epilogue is checked
//! against. Later units add `gen::msm` next to them. They all put a field prelude in front
//! of their own entry points: [`field::field_module`] when they need `Fq` and `Fq2`, or just `FR.ops` when,
//! like the NTT, they only ever touch the scalar field.

pub mod field;
pub mod gather;
pub mod ntt;
pub mod pointwise;

pub use field::{field_module, Field, Variant, FQ, FQ2_OPS, FR, MUL64};
pub use gather::gather_module;
pub use ntt::{ntt_module, Mode};
pub use pointwise::h_join_module;
