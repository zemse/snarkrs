//! Kernel sources, compiled at run time by NVRTC.
//!
//! NVRTC has no filesystem, so a translation unit is assembled by concatenating the headers a
//! kernel needs with the kernel itself. That is why the `.cu` files carry no `#include` of
//! their own: the include graph lives here, in Rust, where it is checked by the compiler
//! rather than by a preprocessor search path that only exists on the machine that happened to
//! build it. Each unit is prefixed with `defines`, the preprocessor prelude.

/// BN254 scalar field. Twin of `snarkrs_gpu_layout`'s `PackedFr`, guarded by
/// `tests::cuda_declares_the_same_constants`.
pub const FR_CUH: &str = include_str!("kernels/bn254_fr.cuh");

/// Stage 0: gather the A and B coefficient columns out of the CSR proving key.
pub const GATHER_CU: &str = include_str!("kernels/gather.cu");

/// Stages 1 to 3: the six transforms, iNTT then coset shift then forward NTT.
pub const NTT_CU: &str = include_str!("kernels/ntt.cu");

/// Stage 4: `H = A*B - C`, fused into the tail of the last transform where possible.
pub const POINTWISE_CU: &str = include_str!("kernels/pointwise.cu");

/// BN254 Fq, Fq2 and the G1/G2 point arithmetic, shared by the MSM and FFT units. Split
/// out of `msm.cu` so the FFT unit does not recompile the seventeen MSM entry points;
/// everything in it is a template or a `__forceinline__` helper, so a unit pays only for
/// what its own kernels instantiate.
pub const CURVE_CUH: &str = include_str!("kernels/bn254_curve.cuh");

/// Stages 5 to 9: the CSR Pippenger MSM kernels.
pub const MSM_CU: &str = include_str!("kernels/msm.cu");

/// The group inverse FFT behind `ptau prepare`: the GLV ladder and the two mix kernels.
pub const FFT_CU: &str = include_str!("kernels/fft.cu");

/// Field correctness probe. Test-only, but compiled the same way as everything else so
/// that a change which breaks compilation cannot hide behind a `cfg(test)`.
pub const FIELD_PROBE_CU: &str = include_str!("kernels/field_probe.cu");

/// Source-level feature defines, prepended to every unit.
///
/// `G16_CUDA_FF_PTX=1` in the environment defines `G16_FF_PTX`, which swaps the portable
/// CIOS Montgomery multiply for the inline-PTX carry-chain version in `bn254_fr.cuh` /
/// `msm.cu`. OFF by default: the PTX path has been validated against a host simulation
/// of its exact chain shape but has never executed on an NVIDIA card. First run on real
/// hardware: check `fr_probe` passes with the flag on, then compare `g16 ptau fft-bench`
/// with the flag off/on; it prints which way the flag was set. Only then is it a candidate
/// for the default.
///
/// `G16_CUDA_FFT_VARIANTS=1` defines `G16_FFT_VARIANTS`, which instantiates the FFT
/// experiment entry points (`kernels/fft.cu`, bottom) alongside the shipped pair. One
/// define covering the whole candidate set is the point: the sweep costs one compile,
/// and a machine that never sweeps never pays for it.
///
/// Going through the source text rather than an NVRTC `-D` option is deliberate: the PTX
/// cache key in `context.rs` hashes the source, so a define spliced into the source can
/// never collide with a cached portable build.
fn defines() -> String {
    let mut d = String::new();
    if std::env::var("G16_CUDA_FF_PTX").as_deref() == Ok("1") {
        d.push_str("#define G16_FF_PTX 1\n");
    }
    if fft_variants_enabled() {
        d.push_str("#define G16_FFT_VARIANTS 1\n");
    }
    d
}

/// Whether the FFT unit carries the experiment entry points. `fft.rs` consults this too,
/// so the host never tries to load a kernel the unit was assembled without.
pub(crate) fn fft_variants_enabled() -> bool {
    std::env::var("G16_CUDA_FFT_VARIANTS").as_deref() == Ok("1")
}

/// Stages 0 to 4, one translation unit. They share `Fr` and nothing else, and compiling
/// them together means one NVRTC invocation instead of three.
///
/// The order gather / ntt / pointwise is a correctness contract, not a preference:
/// `ntt.cu` forward-declares `g16_store_h` and `pointwise.cu` defines it, so pointwise
/// must come last. Both files say so in prose.
pub fn unit_stages() -> String {
    format!(
        "{}{FR_CUH}\n{GATHER_CU}\n{NTT_CU}\n{POINTWISE_CU}",
        defines()
    )
}

/// Stages 5 to 9. Separate from the stages unit because it is by far the largest and
/// compiling it is most of the prepare cost.
pub fn unit_msm() -> String {
    format!("{}{FR_CUH}\n{CURVE_CUH}\n{MSM_CU}", defines())
}

/// The ceremony group FFT. Its own unit rather than a rider on the MSM one: a unit
/// compiles every `extern "C" __global__` it contains whether or not the host loads it,
/// the MSM unit already costs 113.6 s of NVRTC on a cold T4, and `ptau prepare` never
/// launches an MSM kernel. The price is that the shared curve arithmetic is compiled
/// twice on a box that runs both the prover and the ceremony, once per unit, which the
/// on-disk PTX cache pays only on the first run of each.
pub fn unit_fft() -> String {
    format!("{}{FR_CUH}\n{CURVE_CUH}\n{FFT_CU}", defines())
}

/// The `fr_probe` / `fr_constants` translation unit.
pub fn unit_field_probe() -> String {
    format!("{}{FR_CUH}\n{FIELD_PROBE_CU}", defines())
}
