//! Kernel sources, compiled at run time by NVRTC.
//!
//! NVRTC has no filesystem, so a translation unit is assembled here by concatenating the
//! headers a kernel needs with the kernel itself. That is why the `.cu` files carry no
//! `#include` of their own: the include graph lives in this file, in Rust, where it is
//! checked by the compiler rather than by a preprocessor search path that only exists on
//! the machine that happened to build it.

/// BN254 scalar field. Twin of `g16_gpu_layout`'s `PackedFr`, guarded by
/// [`crate::tests::cuda_declares_the_same_constants`].
pub const FR_CUH: &str = include_str!("kernels/bn254_fr.cuh");

/// Field correctness probe. Test-only, but compiled the same way as everything else so
/// that a change which breaks compilation cannot hide behind a `cfg(test)`.
pub const FIELD_PROBE_CU: &str = include_str!("kernels/field_probe.cu");

/// The `fr_probe` / `fr_constants` translation unit.
pub fn unit_field_probe() -> String {
    format!("{FR_CUH}\n{FIELD_PROBE_CU}")
}
