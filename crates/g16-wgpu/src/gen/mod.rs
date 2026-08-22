//! WGSL source generation.
//!
//! Every kernel this backend runs is emitted from Rust at run time, not shipped as a
//! `.wgsl` file. [`field`] explains why in detail; the short version is that
//! the limb-width sweep measured a 3.8x penalty for indexing a
//! function-scope `array<u32, N>` by a loop variable, so every limb loop has to be
//! straight-line code with literal indices, and nobody is hand-writing 8 rounds x 8 limbs
//! of that twice (once for `Fr`, once for `Fq`) and keeping it right.
//!
//! Later units add `gen::ntt`, `gen::msm` and `gen::gather` next to [`field`]; they all
//! concatenate [`field::field_module`] in front of their own entry points.

pub mod field;

pub use field::{field_module, Field, Variant, FQ, FQ2_OPS, FR, MUL64};
