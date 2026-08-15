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

/// Source-level feature defines, prepended to every unit.
///
/// `G16_CUDA_FF_PTX=1` in the environment defines `G16_FF_PTX`, which swaps the portable
/// CIOS Montgomery multiply for the inline-PTX carry-chain version in `bn254_fr.cuh` /
/// `msm.cu`. OFF by default: the PTX path has been validated against a host simulation
/// of its exact chain shape but has never executed on an NVIDIA card. First run on real
/// hardware: check `fr_probe` passes with the flag on, then compare `bench_chain` with
/// the flag off/on; only then is it a candidate for the default.
///
/// Going through the source text rather than an NVRTC `-D` option is deliberate: the PTX
/// cache key in `context.rs` hashes the source, so a define spliced into the source can
/// never collide with a cached portable build.
fn defines() -> String {
    if std::env::var("G16_CUDA_FF_PTX").as_deref() == Ok("1") {
        "#define G16_FF_PTX 1\n".to_string()
    } else {
        String::new()
    }
}

/// Stages 0 to 4, one translation unit. They share `Fr` and nothing else, and compiling
/// them together means one NVRTC invocation instead of three.
pub fn unit_stages() -> String {
    format!("{}{FR_CUH}\n{GATHER_CU}\n{NTT_CU}\n{POINTWISE_CU}", defines())
}

/// Stages 5 to 9. Separate from the stages unit because it is by far the largest and
/// compiling it is most of the prepare cost.
pub fn unit_msm() -> String {
    format!("{}{FR_CUH}\n{MSM_CU}", defines())
}

/// The `fr_probe` / `fr_constants` translation unit.
pub fn unit_field_probe() -> String {
    format!("{}{FR_CUH}\n{FIELD_PROBE_CU}", defines())
}
