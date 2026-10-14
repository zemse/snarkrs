//! Everything `snarkrs` does besides dispatching commands, and the snarkjs command line itself.
//!
//! Split out of `main.rs` so the JSON encoding and the benchmark can be tested directly
//! instead of only through the process boundary.

pub mod artifacts;
pub mod bench;
pub mod cli;
pub mod fallback;
#[cfg(feature = "cuda")]
pub mod fftbench;
pub mod json;
pub mod log;

use anyhow::Result;
use snarkrs_ceremony::{CpuGroupFft, CpuKeyScale};
use snarkrs_groth16::{cpu::CpuBackend, Backend};
use snarkrs_msm::{AccelError, CheckedGroupFft, CpuMsm, GroupFft, KeyScale, MsmBackend};

/// Whether CPUID reports BMI2 (leaf 7 EBX bit 8) and ADX (bit 19), given the highest
/// basic leaf and leaf 7 subleaf 0's EBX.
pub fn adx_bmi2(max_leaf: u32, leaf7_ebx: u32) -> bool {
    max_leaf >= 7 && leaf7_ebx & (1 << 8) != 0 && leaf7_ebx & (1 << 19) != 0
}

/// Whether this CPU has ADX and BMI2. Raw CPUID, because `is_x86_feature_detected!` folds to
/// `true` when the feature is enabled at compile time, which is exactly the build that asks.
#[cfg(target_arch = "x86_64")]
// Older toolchains declare the cpuid intrinsics unsafe and newer ones do not.
#[allow(unused_unsafe)]
pub fn cpu_has_adx_bmi2() -> bool {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    // SAFETY: every x86_64 CPU has CPUID, and leaf 7 is read only when leaf 0 lists it.
    let (max_leaf, leaf7_ebx) = unsafe {
        let max_leaf = __cpuid(0).eax;
        let ebx = if max_leaf >= 7 {
            __cpuid_count(7, 0).ebx
        } else {
            0
        };
        (max_leaf, ebx)
    };
    adx_bmi2(max_leaf, leaf7_ebx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Metal,
    Cuda,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Wgpu => "wgpu",
            BackendKind::Metal => "metal",
            BackendKind::Cuda => "cuda",
        }
    }
}

/// A compiled backend that failed to open its device or build its kernels.
#[cfg(any(
    feature = "wgpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn unavailable(kind: BackendKind, e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("backend `{}` is unavailable: {e}", kind.as_str())
}

/// The backend's feature is off. Each backend's feature has the backend's own name.
fn not_built(kind: BackendKind) -> anyhow::Error {
    let k = kind.as_str();
    anyhow::anyhow!(
        "backend `{k}` is unavailable: this binary was built WITHOUT the `{k}` \
         feature. Rebuild with `cargo build --release --features {k}`."
    )
}

/// Construct a backend, or explain precisely why it cannot exist in this binary.
///
/// Three failure modes, and they need three different fixes, so they get three different
/// messages: the feature is off (rebuild), the feature is on but the target is not macOS
/// (wrong machine), and the feature is on and the machine has no Metal device (this
/// backend cannot run here). The last one is an error rather than a quiet fall back to
/// the CPU backend, because a benchmark that silently measures the other backend is
/// worse than no number at all.
pub fn make_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::new())),
        BackendKind::Wgpu => wgpu_backend(),
        BackendKind::Metal => metal_backend(),
        BackendKind::Cuda => cuda_backend(),
    }
}

/// [`make_backend`] with fixed-work MSM scheduling behind `prove --constant-work`.
/// This is not a constant-time guarantee; see each backend's work-mode contract.
pub fn make_constant_work_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::constant_work())),
        BackendKind::Wgpu => wgpu_constant_work_backend(),
        BackendKind::Metal => metal_constant_work_backend(),
        BackendKind::Cuda => cuda_constant_work_backend(),
    }
}

#[cfg(feature = "cuda")]
fn cuda_constant_work_backend() -> Result<Box<dyn Backend>> {
    Ok(Box::new(
        snarkrs_cuda::CudaBackend::constant_work()
            .map_err(|e| unavailable(BackendKind::Cuda, e))?,
    ))
}

#[cfg(not(feature = "cuda"))]
fn cuda_constant_work_backend() -> Result<Box<dyn Backend>> {
    cuda_backend()
}

#[cfg(feature = "wgpu")]
fn wgpu_constant_work_backend() -> Result<Box<dyn Backend>> {
    Ok(Box::new(
        snarkrs_wgpu::WgpuProver::constant_work().map_err(|e| unavailable(BackendKind::Wgpu, e))?,
    ))
}

/// Same two reasons as [`wgpu_backend`], which is where they are spelled out.
#[cfg(not(feature = "wgpu"))]
fn wgpu_constant_work_backend() -> Result<Box<dyn Backend>> {
    wgpu_backend()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_constant_work_backend() -> Result<Box<dyn Backend>> {
    Ok(Box::new(
        snarkrs_metal::MetalBackend::constant_work()
            .map_err(|e| unavailable(BackendKind::Metal, e))?,
    ))
}

/// Same three reasons as [`metal_backend`], which is where they are spelled out.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_constant_work_backend() -> Result<Box<dyn Backend>> {
    metal_backend()
}

/// WebGPU has the same two failure modes as CUDA and for a similar reason: `wgpu` compiles on
/// every target this workspace supports and picks a native backend (Metal, Vulkan, DX12) at
/// run time, so there is no "wrong operating system" arm. Either the feature is off, or the
/// feature is on and the machine either has an adapter or does not.
///
/// A missing adapter is an error rather than a quiet fall back to the CPU backend, for the
/// reason the other two give: a benchmark that silently measures a different backend is worse
/// than no number at all.
///
/// The expensive constructor is here, once, and never on a proving path: it opens the device
/// and compiles the three MSM shader modules. The per-key work, which is the NTT pipelines and
/// every base vector, is in `prepare`.
#[cfg(feature = "wgpu")]
fn wgpu_backend() -> Result<Box<dyn Backend>> {
    Ok(Box::new(
        snarkrs_wgpu::WgpuProver::new().map_err(|e| unavailable(BackendKind::Wgpu, e))?,
    ))
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_backend() -> Result<Box<dyn Backend>> {
    Err(not_built(BackendKind::Wgpu))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    // `MetalBackend::new` is the expensive constructor: it compiles the MSL and builds
    // every pipeline state, so it belongs here, once, and never on a proving path.
    Ok(Box::new(
        snarkrs_metal::MetalBackend::new().map_err(|e| unavailable(BackendKind::Metal, e))?,
    ))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    Err(no_metal())
}

/// CUDA has one fewer failure mode than Metal, and it is worth saying why.
///
/// `snarkrs-cuda` uses `cudarc` with `dynamic-loading`, so it compiles on any target and
/// resolves `libcuda` and `libnvrtc` with `dlopen` at run time. There is therefore no
/// "wrong operating system" arm here: either the feature is off, or the feature is on and
/// the machine either has a usable device or does not. As with Metal, a missing device is
/// an error rather than a quiet fall back to the CPU backend, because a benchmark that
/// silently measures a different backend is worse than no number at all.
#[cfg(feature = "cuda")]
fn cuda_backend() -> Result<Box<dyn Backend>> {
    // The expensive constructor: NVRTC compiles every kernel here, once, never on a
    // proving path.
    Ok(Box::new(
        snarkrs_cuda::CudaBackend::new().map_err(|e| unavailable(BackendKind::Cuda, e))?,
    ))
}

#[cfg(not(feature = "cuda"))]
fn cuda_backend() -> Result<Box<dyn Backend>> {
    Err(not_built(BackendKind::Cuda))
}

/// The feature is off, or it is on and the target is not macOS. Two different fixes, so
/// two different messages, for [`make_backend`] and the ceremony constructors alike.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn no_metal() -> anyhow::Error {
    if cfg!(feature = "metal") {
        anyhow::anyhow!(
            "backend `metal` is unavailable: the `metal` feature is enabled but this target \
             is not macOS, so snarkrs-metal compiled to nothing"
        )
    } else {
        not_built(BackendKind::Metal)
    }
}

/// The primitive a ceremony command needs, which decides which backends can serve it.
#[derive(Clone, Copy, Debug)]
pub enum CeremonySeam {
    Msm,
    GroupFft,
    KeyScale,
}

/// Selection without opening a device. A compiled backend can still fail to initialise.
pub fn ceremony_backend_kind(kind: BackendKind, seam: CeremonySeam) -> Result<BackendKind> {
    match kind {
        BackendKind::Cpu => Ok(kind),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Cuda if matches!(seam, CeremonySeam::GroupFft) && cfg!(feature = "cuda") => {
            Ok(kind)
        }
        BackendKind::Cuda => Err(no_cuda()),
        BackendKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Ok(kind)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(no_metal())
            }
        }
    }
}

/// The MSM behind `setup` and the two same-ratio checks.
///
/// The three constructors below are `make_backend` for the ceremony: same three failure
/// modes, same three messages, same refusal to fall back to the CPU when the asked-for
/// backend is not there. A benchmark that silently measures the other backend is worse than
/// no number, and a `.zkey` that silently came from the other backend is worse than no key.
pub fn msm_backend(kind: BackendKind) -> Result<Box<dyn MsmBackend>> {
    match ceremony_backend_kind(kind, CeremonySeam::Msm)? {
        BackendKind::Cpu => Ok(Box::new(CpuMsm::new())),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Metal => metal_msm(),
        BackendKind::Cuda => Err(no_cuda()),
    }
}

/// The group inverse FFT behind `ptau prepare`.
pub fn fft_backend(kind: BackendKind) -> Result<Box<dyn GroupFft>> {
    let fft: Box<dyn GroupFft> = match ceremony_backend_kind(kind, CeremonySeam::GroupFft)? {
        BackendKind::Cpu => Box::new(CpuGroupFft),
        BackendKind::Wgpu => return Err(no_wgpu()),
        BackendKind::Metal => metal_fft()?,
        BackendKind::Cuda => cuda_fft()?,
    };
    Ok(Box::new(CheckedGroupFft(fft)))
}

/// The batch apply-key behind every contribute and beacon command.
pub fn key_backend(kind: BackendKind) -> Result<Box<dyn KeyScale>> {
    match ceremony_backend_kind(kind, CeremonySeam::KeyScale)? {
        BackendKind::Cpu => Ok(Box::new(CpuKeyScale)),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Metal => metal_key(),
        BackendKind::Cuda => Err(no_cuda()),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_msm() -> Result<Box<dyn MsmBackend>> {
    Ok(Box::new(
        snarkrs_metal::MetalMsmBackend::new().map_err(|e| unavailable(BackendKind::Metal, e))?,
    ))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_fft() -> Result<Box<dyn GroupFft>> {
    Ok(Box::new(
        snarkrs_metal::MetalGroupFft::new().map_err(|e| unavailable(BackendKind::Metal, e))?,
    ))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_key() -> Result<Box<dyn KeyScale>> {
    Ok(Box::new(
        snarkrs_metal::MetalKeyScale::new().map_err(|e| unavailable(BackendKind::Metal, e))?,
    ))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_msm() -> Result<Box<dyn MsmBackend>> {
    Err(no_metal())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_fft() -> Result<Box<dyn GroupFft>> {
    Err(no_metal())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_key() -> Result<Box<dyn KeyScale>> {
    Err(no_metal())
}

#[cfg(feature = "cuda")]
fn cuda_fft() -> Result<Box<dyn GroupFft>> {
    Ok(Box::new(
        snarkrs_cuda::CudaGroupFft::new().map_err(|e| unavailable(BackendKind::Cuda, e))?,
    ))
}

#[cfg(not(feature = "cuda"))]
fn cuda_fft() -> Result<Box<dyn GroupFft>> {
    Err(no_cuda())
}

/// CUDA has kernels for the prover and for `ptau prepare`'s group FFT, and none for the
/// two ceremony seams this message now covers. Saying "not built in" when the feature
/// *is* on would send someone to rebuild a binary that already has everything it is
/// going to get.
fn no_cuda() -> anyhow::Error {
    if cfg!(feature = "cuda") {
        AccelError::Unimplemented {
            backend: "cuda",
            reason: "snarkrs-cuda implements the prover and the `ptau prepare` group FFT; \
                     the ceremony MSM and apply-key have no CUDA kernels yet",
        }
        .into()
    } else {
        not_built(BackendKind::Cuda)
    }
}

/// Unlike the other two this is not a "yet". A ceremony kernel is a point scalar
/// multiplication, which inlines the point operations three or more times over a 256-byte
/// `Xyzz<Fq2>`, and that is exactly the shape WebKit 323560 miscompiles on iOS.
fn no_wgpu() -> anyhow::Error {
    anyhow::anyhow!(
        "backend `wgpu` is unavailable for the ceremony: these kernels are Metal and CUDA \
         only, because a scalar-multiplication ladder is the shape WebKit 323560 \
         miscompiles. Use `--backend cpu` or `--backend metal`."
    )
}

/// Machine identification for the benchmark CSV.
///
/// `os` and `arch` are reported the way Python's `platform` module reports them, not the
/// way Rust's `std::env::consts` does, so rows written here and rows written by
/// `bench/scripts/run-comparison.py` land in the same CSV without two spellings of the
/// same machine.
pub struct HostInfo {
    pub host: String,
    pub os: &'static str,
    pub arch: &'static str,
    pub cores: usize,
}

impl HostInfo {
    pub fn detect() -> Self {
        Self {
            host: std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".to_string()),
            os: match std::env::consts::OS {
                "macos" => "Darwin",
                "linux" => "Linux",
                "windows" => "Windows",
                other => other,
            },
            arch: match std::env::consts::ARCH {
                "aarch64" => "arm64",
                other => other,
            },
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_always_available() {
        assert_eq!(make_backend(BackendKind::Cpu).unwrap().name(), "cpu");
    }

    #[test]
    fn adx_bmi2_needs_both_bits_and_leaf_7() {
        let both = (1 << 8) | (1 << 19);
        assert!(adx_bmi2(7, both));
        assert!(adx_bmi2(0x20, both | 0xffff_0000));
        assert!(!adx_bmi2(6, both), "leaf 7 is garbage below max leaf 7");
        assert!(!adx_bmi2(7, 1 << 8), "bmi2 alone (x86-64-v3)");
        assert!(!adx_bmi2(7, 1 << 19), "adx alone");
        assert!(!adx_bmi2(7, !both));
    }

    /// The machine running the tests built this binary, so a `+adx,+bmi2` build must see both.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "adx",
        target_feature = "bmi2"
    ))]
    #[test]
    fn adx_bmi2_build_runs_on_a_cpu_with_both() {
        assert!(cpu_has_adx_bmi2());
    }

    /// Every backend has a spelling in the CSV and on the command line, and they have to be
    /// the same string: `bench/results/history.csv` is keyed on it and a row written as
    /// "webgpu" would never join a row written as "wgpu".
    #[test]
    fn every_backend_kind_has_a_stable_name() {
        for (kind, want) in [
            (BackendKind::Cpu, "cpu"),
            (BackendKind::Wgpu, "wgpu"),
            (BackendKind::Metal, "metal"),
            (BackendKind::Cuda, "cuda"),
        ] {
            assert_eq!(kind.as_str(), want);
        }
    }

    /// Without the feature the error must name the feature, so the fix is obvious from the
    /// message alone. With it, the backend either builds or explains that the machine has no
    /// adapter; both are acceptable on a headless box and neither may be a silent CPU proof.
    #[test]
    fn wgpu_either_builds_or_says_why_not() {
        match make_backend(BackendKind::Wgpu) {
            Ok(b) => assert_eq!(b.name(), "wgpu"),
            Err(e) => {
                let e = e.to_string();
                assert!(e.contains("wgpu"), "{e}");
                assert!(
                    !e.contains("cpu"),
                    "a wgpu failure must not mention a fallback: {e}"
                );
            }
        }
    }

    /// Without the feature the error must name the feature, so the fix is obvious from
    /// the message alone.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_feature_says_so() {
        match make_backend(BackendKind::Cuda) {
            Ok(_) => panic!("built a cuda backend without the cuda feature"),
            Err(e) => {
                let e = e.to_string();
                assert!(e.contains("cuda"), "{e}");
                assert!(e.contains("--features cuda"), "{e}");
            }
        }
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn metal_without_the_feature_says_so() {
        let e = match make_backend(BackendKind::Metal) {
            Ok(_) => panic!("built a metal backend without the metal feature"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("metal"), "{e}");
        assert!(e.contains("--features metal"), "{e}");
    }

    /// Every seam needs a CPU implementation under the same spelling, because
    /// `--backend cpu` is the file every accelerated run is `cmp`'d against and the only
    /// one already held against snarkjs byte for byte.
    #[test]
    fn every_ceremony_seam_has_a_cpu_backend() {
        assert_eq!(msm_backend(BackendKind::Cpu).unwrap().name(), "cpu");
        assert_eq!(fft_backend(BackendKind::Cpu).unwrap().name(), "cpu");
        assert_eq!(key_backend(BackendKind::Cpu).unwrap().name(), "cpu");
    }

    fn backend_error<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("selected an unavailable ceremony backend"),
            Err(e) => e.to_string(),
        }
    }

    /// All three real constructors must reject WebGPU, even when its feature is on.
    #[test]
    fn wgpu_rejects_every_ceremony_seam() {
        for e in [
            backend_error(msm_backend(BackendKind::Wgpu)),
            backend_error(fft_backend(BackendKind::Wgpu)),
            backend_error(key_backend(BackendKind::Wgpu)),
        ] {
            assert!(
                e.contains("backend `wgpu` is unavailable for the ceremony"),
                "{e}"
            );
            assert!(e.contains("--backend cpu"), "{e}");
        }
    }

    #[test]
    fn cuda_rejects_ceremony_msm_and_key_scaling() {
        for e in [
            backend_error(msm_backend(BackendKind::Cuda)),
            backend_error(key_backend(BackendKind::Cuda)),
        ] {
            assert!(e.contains("backend `cuda` is unavailable"), "{e}");
            if cfg!(feature = "cuda") {
                assert!(e.contains("MSM and apply-key"), "{e}");
                assert!(e.contains("no CUDA kernels"), "{e}");
                assert!(!e.contains("WITHOUT"), "{e}");
            } else {
                assert!(e.contains("WITHOUT the `cuda` feature"), "{e}");
                assert!(e.contains("--features cuda"), "{e}");
            }
        }
    }

    #[test]
    fn cuda_group_fft_selection_respects_the_feature() {
        if cfg!(feature = "cuda") {
            // Selection only: this test must not load a CUDA driver.
            assert!(matches!(
                ceremony_backend_kind(BackendKind::Cuda, CeremonySeam::GroupFft),
                Ok(BackendKind::Cuda)
            ));
        } else {
            let e = backend_error(fft_backend(BackendKind::Cuda));
            assert!(e.contains("WITHOUT the `cuda` feature"), "{e}");
            assert!(e.contains("--features cuda"), "{e}");
        }
    }

    #[test]
    fn supported_ceremony_selection_keeps_the_requested_backend() {
        for seam in [
            CeremonySeam::Msm,
            CeremonySeam::GroupFft,
            CeremonySeam::KeyScale,
        ] {
            assert!(matches!(
                ceremony_backend_kind(BackendKind::Cpu, seam),
                Ok(BackendKind::Cpu)
            ));
            if cfg!(all(feature = "metal", target_os = "macos")) {
                assert!(matches!(
                    ceremony_backend_kind(BackendKind::Metal, seam),
                    Ok(BackendKind::Metal)
                ));
            } else {
                assert!(ceremony_backend_kind(BackendKind::Metal, seam).is_err());
            }
        }
    }

    /// Feature and target failures need different fixes, and both must name Metal.
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn unavailable_metal_reports_the_feature_or_target() {
        for e in [
            backend_error(msm_backend(BackendKind::Metal)),
            backend_error(fft_backend(BackendKind::Metal)),
            backend_error(key_backend(BackendKind::Metal)),
        ] {
            assert!(e.contains("backend `metal` is unavailable"), "{e}");
            if cfg!(feature = "metal") {
                assert!(e.contains("not macOS"), "{e}");
            } else {
                assert!(e.contains("WITHOUT the `metal` feature"), "{e}");
                assert!(e.contains("--features metal"), "{e}");
            }
        }
    }
    #[test]
    fn host_info_is_populated() {
        let h = HostInfo::detect();
        assert!(!h.host.is_empty());
        assert!(h.cores >= 1);
        assert!(!h.os.is_empty() && !h.arch.is_empty());
    }
}
