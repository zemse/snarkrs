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
    /// Opens a context on `ordinal`, or says why it could not.
    ///
    /// # This returns `Err` where `cudarc` panics
    ///
    /// With the `dynamic-loading` feature, `cudarc` resolves `libcuda` lazily on first use
    /// and **unwraps** the `dlopen`. On a machine with no NVIDIA driver at all, which is
    /// every macOS host in this workspace, `CudaContext::new` therefore aborts the process
    /// instead of returning the error its signature promises. That defeats the whole
    /// skip-if-no-device pattern the test suites are written around, so the unwind is
    /// caught here and turned back into a [`CudaError`].
    ///
    /// Only the context creation is wrapped. Everything after it has a real driver behind
    /// it and a real error channel, and swallowing panics from arbitrary later code would
    /// hide genuine bugs.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        // The default hook prints a panic message and a backtrace for something that is
        // not a crash. Silenced for the duration of this one call and restored straight
        // after, so a panic anywhere else still prints normally.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let opened = std::panic::catch_unwind(|| CudaContext::new(ordinal));
        std::panic::set_hook(hook);

        let ctx = match opened {
            Ok(r) => r.map_err(|e| CudaError::NoDevice(e.to_string()))?,
            Err(payload) => {
                let what = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with a non-string payload".to_string());
                return Err(CudaError::NoDevice(format!(
                    "the CUDA driver library could not be loaded: {what}"
                )));
            }
        };
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

    /// Compile one translation unit and load it, going through the on-disk PTX cache.
    ///
    /// `--use_fast_math` is deliberately NOT passed. There is not a single floating point
    /// operation in any of these kernels, so it could only change integer codegen for the
    /// worse, and a flag that relaxes numerical guarantees has no business anywhere near
    /// a proof system.
    ///
    /// # Why there is a cache at all, with the numbers
    ///
    /// Getting the MSM kernels onto a T4 for the first time costs roughly **270 to 300
    /// seconds**. It is worth being precise about where that goes, because the obvious
    /// summary is wrong in both directions.
    ///
    /// There are two expensive stages, not one:
    ///
    /// | stage | cold | warm |
    /// |---|---:|---:|
    /// | NVRTC, source to PTX | 113.6 s | 0.18 s |
    /// | driver JIT, PTX to SASS, inside `cuModuleLoad` | about 175 s | 0.12 s |
    ///
    /// Both figures were measured on the same T4, and the NVRTC one is corroborated
    /// offline: `nvcc -arch=compute_75 -ptx` on the identical source takes 108 s and emits
    /// byte-identical PTX. The stages unit, for contrast, is 1.3 s.
    ///
    /// **`~/.nv/ComputeCache` caches both stages, not just the JIT.** Deleting it and
    /// recompiling reproduces the 113.6 s NVRTC figure; the next run is 0.18 s. That is the
    /// single most important fact here, and it is why a cold-start number for any CUDA
    /// prover taken on a machine that has run it before is measuring a cache hit. On a
    /// fresh CI runner with no persistent `$HOME`, every run is cold.
    ///
    /// **So what does this cache buy, honestly.** It covers the NVRTC stage only. With
    /// `~/.nv` warm it saves nothing measurable: 0.30 s either way, verified by running
    /// with `G16_CUDA_NO_CACHE=1`. With `~/.nv` cold it removes 113 s of the 283, and the
    /// remaining 175 s of driver JIT is not something a process can cache for itself
    /// through this API. That is a real but partial win, and it is the reason to keep it
    /// rather than a reason to have built it: the driver's cache is a fixed-size LRU shared
    /// by every CUDA process on the machine (`CUDA_CACHE_MAXSIZE`, one gigabyte by
    /// default), a 30 MB entry in it is evictable by unrelated work, and an entry that
    /// survives here is one that cannot be evicted by someone else's job.
    ///
    /// An earlier version of this comment claimed the cache meant "compile once ever, then
    /// pay 120 ms of driver JIT". The 120 ms was a warm-`~/.nv` measurement quoted as if it
    /// held on a cold machine, which it does not. Corrected above.
    ///
    /// Two ways to shrink the generated code were measured and neither is worth taking:
    ///
    /// * Removing every `#pragma unroll` changes nothing: 109.7 s against 108.0 s.
    /// * Demoting `__forceinline__` to `__inline__` cuts NVRTC to 25.7 s and the PTX from
    ///   787,873 lines to 235,000, and is a **trap**. The smaller PTX leaves 4 functions out
    ///   of line with 350 call sites, and the driver JIT then takes **29.6 s** on a warm
    ///   `~/.nv` where the fully inlined version takes **120 ms**. Inlined PTX is
    ///   straight-line code the JIT merely assembles; PTX with calls makes it redo the
    ///   interprocedural work. Total cost is worse and the generated code is worse, so
    ///   `__forceinline__` stays.
    ///
    /// A cache miss is never an error. A corrupt, truncated or unreadable entry falls
    /// through to a real compile, and a cache directory that cannot be written is ignored.
    /// `G16_CUDA_NO_CACHE=1` bypasses only *this* cache; measuring a genuine cold compile
    /// also needs `~/.nv` cleared, and `CUDA_CACHE_DISABLE=1` disables only the driver's.
    pub fn compile(&self, unit: &'static str, src: &str) -> Result<Arc<CudaModule>, CudaError> {
        let key = cache_key(src, self.arch);
        let path = cache_path(unit, &key);

        if let Some(p) = path.as_ref().filter(|_| !cache_disabled()) {
            if let Ok(cached) = std::fs::read_to_string(p) {
                // Only a non-empty entry that ends the way NVRTC ends one is trusted. The
                // write below is atomic, so a torn file should be impossible; this is here
                // because "should be impossible" and "is impossible" differ on a machine
                // that lost power mid-write, and the failure mode of a truncated PTX is a
                // confusing driver error rather than a clean miss.
                if cached.len() > 64 && cached.contains(".visible .entry") {
                    if let Ok(m) = self.ctx.load_module(cudarc::nvrtc::Ptx::from_src(cached)) {
                        return Ok(m);
                    }
                }
            }
        }

        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(self.arch),
            // Treat warnings as the signal they are. NVRTC is quiet on clean code, so
            // anything it prints is worth reading rather than scrolling past.
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        let ptx =
            cudarc::nvrtc::compile_ptx_with_opts(src, opts).map_err(|e| CudaError::Compile {
                unit,
                log: e.to_string(),
            })?;

        if let Some(p) = path.as_ref().filter(|_| !cache_disabled()) {
            let text = ptx.to_src();
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
                // Write to a unique temporary and rename, so two provers racing on the same
                // cache entry cannot leave a half-written file for a third to read. Rename
                // is atomic within a filesystem, and both paths are in the same directory.
                let tmp = dir.join(format!(
                    "{}.{}.tmp",
                    p.file_name().unwrap().to_string_lossy(),
                    std::process::id()
                ));
                if std::fs::write(&tmp, &text).is_ok() && std::fs::rename(&tmp, p).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }

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

fn cache_disabled() -> bool {
    std::env::var_os("G16_CUDA_NO_CACHE").is_some_and(|v| v != "0")
}

/// FNV-1a over the source and the architecture.
///
/// Written out rather than using `DefaultHasher` because that is explicitly not stable
/// across Rust releases, and a hash that changes under the reader's feet turns the cache
/// into a directory that only ever grows. FNV is not cryptographic and does not need to be:
/// the cache is per user, and the worst case for a collision is loading kernels that do not
/// match the source, which the `.visible .entry` check does not catch. That risk is
/// accepted because the input is this crate's own `include_str!` sources, not attacker
/// input, and a 128-bit FNV over them will not collide by accident.
fn cache_key(src: &str, arch: &str) -> String {
    let mut lo: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hi: u64 = 0x9e37_79b9_7f4a_7c15;
    for b in src.as_bytes().iter().chain(b"|").chain(arch.as_bytes()) {
        lo ^= u64::from(*b);
        lo = lo.wrapping_mul(0x0000_0100_0000_01b3);
        hi = hi.rotate_left(7) ^ lo;
        hi = hi.wrapping_mul(0x8864_3f65_ef0d_9b1d);
    }
    format!("{lo:016x}{hi:016x}")
}

/// `$G16_CUDA_CACHE`, else `$XDG_CACHE_HOME/g16-cuda`, else `$HOME/.cache/g16-cuda`.
/// `None` when none of those can be determined, which disables caching rather than
/// guessing at a writable directory.
fn cache_path(unit: &str, key: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("G16_CUDA_CACHE")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME").map(|c| std::path::PathBuf::from(c).join("g16-cuda"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache/g16-cuda"))
        })?;
    Some(dir.join(format!("{unit}-{key}.ptx")))
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
