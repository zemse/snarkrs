//! The CUDA C kernel sources, and the include graph that assembles them.
//!
//! Neither NVRTC nor hipRTC has a filesystem, so a translation unit is assembled by
//! concatenating the headers a kernel needs with the kernel itself. That is why the `.cu`
//! files carry no `#include` of their own: the include graph lives here, in Rust, where it
//! is checked by the compiler rather than by a preprocessor search path that only exists on
//! the machine that happened to build it.
//!
//! Both vendors compile these same six files. The arithmetic is not vendor-specific: the
//! CIOS Montgomery multiply, the NTT butterflies and the Pippenger buckets are the same
//! code on NVIDIA and on AMD, and a second copy of them under a `hip/` directory would be
//! two implementations of one algorithm that only a proof-verification failure could tell
//! apart. One copy, one place to fix a carry.
//!
//! What differs per backend is the preprocessor prelude, and that stays the caller's
//! business: [`unit_stages`] and its two twins take it as a string and paste it at the top.
//! `g16-cuda` splices its NVIDIA-only inline-PTX switch in there, which is a define no AMD
//! compile should ever see.

/// BN254 scalar field. Twin of `g16_gpu_layout`'s `PackedFr`, guarded by `g16-cuda`'s
/// `tests::cuda_declares_the_same_constants`.
pub const FR_CUH: &str = include_str!("kernels/bn254_fr.cuh");

/// Stage 0: gather the A and B coefficient columns out of the CSR proving key.
pub const GATHER_CU: &str = include_str!("kernels/gather.cu");

/// Stages 1 to 3: the six transforms, iNTT then coset shift then forward NTT.
pub const NTT_CU: &str = include_str!("kernels/ntt.cu");

/// Stage 4: `H = A*B - C`, fused into the tail of the last transform where possible.
pub const POINTWISE_CU: &str = include_str!("kernels/pointwise.cu");

/// Stages 5 to 9: Fq, Fq2, the curve, and the CSR Pippenger MSM.
pub const MSM_CU: &str = include_str!("kernels/msm.cu");

/// Field correctness probe. Test-only, but compiled the same way as everything else so
/// that a change which breaks compilation cannot hide behind a `cfg(test)`.
pub const FIELD_PROBE_CU: &str = include_str!("kernels/field_probe.cu");

/// Stages 0 to 4, one translation unit. They share `Fr` and nothing else, and compiling
/// them together means one runtime-compiler invocation instead of three.
///
/// The order gather / ntt / pointwise is a correctness contract, not a preference:
/// `ntt.cu` forward-declares `g16_store_h` and `pointwise.cu` defines it, so pointwise
/// must come last. Both files say so in prose.
pub fn unit_stages(defines: &str) -> String {
    format!("{defines}{FR_CUH}\n{GATHER_CU}\n{NTT_CU}\n{POINTWISE_CU}")
}

/// Stages 5 to 9. Separate from the stages unit because it is by far the largest and
/// compiling it is most of the prepare cost.
pub fn unit_msm(defines: &str) -> String {
    format!("{defines}{FR_CUH}\n{MSM_CU}")
}

/// The `fr_probe` / `fr_constants` translation unit.
pub fn unit_field_probe(defines: &str) -> String {
    format!("{defines}{FR_CUH}\n{FIELD_PROBE_CU}")
}
