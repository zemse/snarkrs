//! Device handle and runtime kernel compilation.
//!
//! Mirrors `g16-metal`'s use of `newLibraryWithSource`: the kernel sources are `include_str!`
//! into the binary and compiled by NVRTC for the GPU that is actually present. The
//! alternative, compiling `.cu` files with `nvcc` in a `build.rs`, would mean the build host
//! needs a CUDA toolkit and the resulting binary carries a fixed set of architectures. This
//! way `cargo build --features cuda` works on a laptop with no CUDA at all, and the binary
//! runs on whatever card it finds.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("no CUDA device: {0}")]
    NoDevice(String),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NVRTC failed to compile {unit}: {log}")]
    Compile { unit: &'static str, log: String },
    #[error("unsupported compute capability {0}.{1}; this backend needs at least 7.0 (Volta)")]
    UnsupportedArch(i32, i32),
    #[error("{0}")]
    Other(String),
}

/// An open CUDA context plus its default stream.
pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    arch: &'static str,
    name: String,
    cc: (i32, i32),
    sm_count: i32,
}

impl Cuda {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| CudaError::NoDevice(e.to_string()))?;
        let stream = ctx.default_stream();
        let cc = ctx.compute_capability()?;
        let arch = arch_flag(cc)?;
        let name = ctx.name()?;
        let sm_count = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?;
        Ok(Self {
            ctx,
            stream,
            arch,
            name,
            cc,
            sm_count,
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    /// Streaming multiprocessor count. The CUDA analogue of the Metal backend's core
    /// count, and the number a dispatch has to fill before the card is busy at all.
    pub fn sm_count(&self) -> i32 {
        self.sm_count
    }

    /// Compile one translation unit and load it. `unit` is only used in error messages.
    ///
    /// `--use_fast_math` is deliberately NOT passed. There is not a single floating point
    /// operation in any of these kernels, so it could only change integer codegen for the
    /// worse, and a flag that relaxes numerical guarantees has no business anywhere near
    /// a proof system.
    pub fn compile(&self, unit: &'static str, src: &str) -> Result<Arc<CudaModule>, CudaError> {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(self.arch),
            // Treat warnings as the signal they are. NVRTC is quiet on clean code, so
            // anything it prints is worth reading rather than scrolling past.
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts).map_err(|e| {
            CudaError::Compile {
                unit,
                log: e.to_string(),
            }
        })?;
        Ok(self.ctx.load_module(ptx)?)
    }

    /// Compile and pull out a named set of kernels in one go.
    pub fn functions(
        &self,
        unit: &'static str,
        src: &str,
        names: &[&str],
    ) -> Result<Vec<CudaFunction>, CudaError> {
        let module = self.compile(unit, src)?;
        names
            .iter()
            .map(|n| module.load_function(n).map_err(CudaError::from))
            .collect()
    }
}

/// Map a compute capability to the NVRTC `--gpu-architecture` value.
///
/// `compute_XY` rather than `sm_XY` on purpose: NVRTC emits PTX, and the driver JIT
/// specialises it for the exact chip at load time. Asking for `sm_XY` would produce a
/// cubin pinned to one architecture for no benefit here.
///
/// Unknown newer capabilities fall back to the highest entry this table knows rather than
/// failing, because PTX for an older architecture is forward compatible through the JIT.
/// The floor is 7.0: below that there is no independent thread scheduling, and the MSM
/// kernels' warp-level reductions would need rewriting.
fn arch_flag(cc: (i32, i32)) -> Result<&'static str, CudaError> {
    Ok(match cc {
        (7, 0) => "compute_70",
        (7, 2) => "compute_72",
        (7, 5) => "compute_75", // Turing, the T4 this was developed against
        (8, 0) => "compute_80", // Ampere, A100
        (8, 6) => "compute_86", // Ampere, A10G / RTX 30xx
        (8, 7) => "compute_87",
        (8, 9) => "compute_89", // Ada, L4 / L40S / RTX 40xx
        (9, 0) => "compute_90", // Hopper
        (major, minor) if major > 9 || (major == 9 && minor > 0) => "compute_90",
        (major, minor) => return Err(CudaError::UnsupportedArch(major, minor)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_flags_cover_the_cards_this_will_meet() {
        assert_eq!(arch_flag((7, 5)).unwrap(), "compute_75");
        assert_eq!(arch_flag((8, 6)).unwrap(), "compute_86");
        assert_eq!(arch_flag((8, 9)).unwrap(), "compute_89");
        // Forward compatibility: a card newer than this table still gets usable PTX.
        assert_eq!(arch_flag((10, 0)).unwrap(), "compute_90");
        assert_eq!(arch_flag((12, 3)).unwrap(), "compute_90");
        // And anything older than Volta is refused rather than silently miscompiled.
        assert!(arch_flag((6, 1)).is_err());
        assert!(arch_flag((5, 0)).is_err());
    }
}
