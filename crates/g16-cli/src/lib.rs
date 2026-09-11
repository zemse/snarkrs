//! Everything `g16` does that is not argument parsing.
//!
//! Split out of `main.rs` so the JSON encoding and the benchmark can be tested directly
//! instead of only through the process boundary.

pub mod artifacts;
pub mod bench;
#[cfg(feature = "cuda")]
pub mod fftbench;
pub mod json;

use anyhow::Result;
use g16_core::{cpu::CpuBackend, Backend};

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
    Ok(Box::new(g16_wgpu::WgpuProver::new().map_err(|e| {
        anyhow::anyhow!("backend `wgpu` is unavailable: {e}")
    })?))
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_backend() -> Result<Box<dyn Backend>> {
    anyhow::bail!(
        "backend `wgpu` is unavailable: this binary was built WITHOUT the `wgpu` \
         feature. Rebuild with `cargo build --release --features wgpu`."
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    // `MetalBackend::new` is the expensive constructor: it compiles the MSL and builds
    // every pipeline state, so it belongs here, once, and never on a proving path.
    Ok(Box::new(g16_metal::MetalBackend::new().map_err(|e| {
        anyhow::anyhow!("backend `metal` is unavailable: {e}")
    })?))
}

#[cfg(all(feature = "metal", not(target_os = "macos")))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    anyhow::bail!(
        "backend `metal` is unavailable: the `metal` feature is enabled but this target \
         is not macOS, so g16-metal compiled to nothing"
    )
}

#[cfg(not(feature = "metal"))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    anyhow::bail!(
        "backend `metal` is unavailable: this binary was built WITHOUT the `metal` \
         feature. Rebuild with `cargo build --release --features metal`."
    )
}

/// CUDA has one fewer failure mode than Metal, and it is worth saying why.
///
/// `g16-cuda` uses `cudarc` with `dynamic-loading`, so it compiles on any target and
/// resolves `libcuda` and `libnvrtc` with `dlopen` at run time. There is therefore no
/// "wrong operating system" arm here: either the feature is off, or the feature is on and
/// the machine either has a usable device or does not. As with Metal, a missing device is
/// an error rather than a quiet fall back to the CPU backend, because a benchmark that
/// silently measures a different backend is worse than no number at all.
#[cfg(feature = "cuda")]
fn cuda_backend() -> Result<Box<dyn Backend>> {
    // The expensive constructor: NVRTC compiles every kernel here, once, never on a
    // proving path.
    Ok(Box::new(g16_cuda::CudaBackend::new().map_err(|e| {
        anyhow::anyhow!("backend `cuda` is unavailable: {e}")
    })?))
}

#[cfg(not(feature = "cuda"))]
fn cuda_backend() -> Result<Box<dyn Backend>> {
    anyhow::bail!(
        "backend `cuda` is unavailable: this binary was built WITHOUT the `cuda` \
         feature. Rebuild with `cargo build --release --features cuda`."
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

    #[test]
    fn host_info_is_populated() {
        let h = HostInfo::detect();
        assert!(!h.host.is_empty());
        assert!(h.cores >= 1);
        assert!(!h.os.is_empty() && !h.arch.is_empty());
    }
}
