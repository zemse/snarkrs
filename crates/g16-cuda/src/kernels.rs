//! Kernel sources, compiled at run time by NVRTC.
//!
//! The sources and the include graph live in `g16-gpu-kernels`, because hipRTC compiles the
//! same six files and a second copy of the CIOS Montgomery multiply would be two
//! implementations of one algorithm that only a failing proof could tell apart. What is left
//! here is the one NVIDIA-only knob: the preprocessor prelude each translation unit is
//! prefixed with.

pub use g16_gpu_kernels::{
    CURVE_CUH, FFT_CU, FIELD_PROBE_CU, FR_CUH, GATHER_CU, MSM_CU, NTT_CU, POINTWISE_CU,
};

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
    g16_gpu_kernels::unit_stages(&defines())
}

/// Stages 5 to 9. Separate from the stages unit because it is by far the largest and
/// compiling it is most of the prepare cost.
pub fn unit_msm() -> String {
    g16_gpu_kernels::unit_msm(&defines())
}

/// The ceremony group FFT unit. See `g16_gpu_kernels::unit_fft` for why it is not part
/// of the MSM unit.
pub fn unit_fft() -> String {
    g16_gpu_kernels::unit_fft(&defines())
}

/// The `fr_probe` / `fr_constants` translation unit.
pub fn unit_field_probe() -> String {
    g16_gpu_kernels::unit_field_probe(&defines())
}
