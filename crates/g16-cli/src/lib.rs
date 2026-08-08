//! Everything `g16` does that is not argument parsing.
//!
//! Split out of `main.rs` so the JSON encoding and the benchmark can be tested directly
//! instead of only through the process boundary.

pub mod artifacts;
pub mod bench;
pub mod json;

use anyhow::Result;
use g16_core::{cpu::CpuBackend, Backend};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    Cpu,
    Metal,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
        }
    }
}

/// Construct a backend, or explain precisely why it cannot exist in this binary.
///
/// The two failure modes are worth distinguishing in the message, because they need
/// opposite fixes: "not compiled in" is a rebuild, "not implemented" is more code. The
/// second one is a `todo!()` inside `g16-metal`, and since the release profile is
/// `panic = "abort"` it cannot be caught here, so the warning is printed before the call
/// rather than translated after it.
pub fn make_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::new())),
        BackendKind::Metal => metal_backend(),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend() -> Result<Box<dyn Backend>> {
    eprintln!(
        "note: the metal backend is compiled in but NOT implemented yet; the next call \
         is expected to abort on todo!() in g16-metal/src/backend.rs"
    );
    Ok(Box::new(g16_metal::MetalBackend::new()?))
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
         feature. Rebuild with `cargo build --release --features metal`. (The backend \
         itself is also still unimplemented: with the feature on it aborts on a todo!().)"
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

    /// Without the feature the error must name the feature, so the fix is obvious from
    /// the message alone.
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
