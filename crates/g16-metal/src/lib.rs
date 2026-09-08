//! Apple Metal backend for BN254 Groth16, stages 0-9.
//!
//! Three facts about this platform shape the whole design, all verified on the M2 Max
//! this is developed against:
//!
//! 1. **MSL compiles at runtime.** `MTLDevice::newLibraryWithSource` uses the Metal
//!    runtime compiler that ships with the OS, so no offline `metal` toolchain download
//!    is needed. Kernels are embedded as source strings and compiled once at `prepare`.
//! 2. **Memory is unified.** `newBufferWithBytesNoCopy` wraps an mmap'd zkey with zero
//!    copies, but it needs a page-aligned pointer and page-multiple length at Apple's
//!    16 KB pages. Wrap the whole mapping once and pass per-section offsets as kernel
//!    arguments; wrapping sections individually will fail alignment. There is no PCIe
//!    hop to amortise here, so the CPU-vs-GPU "transfer cost" story does not apply.
//! 3. **Threadgroup memory is 32 KB, SIMD width 32, max 1024 threads/threadgroup.**
//!    A BN254 `Fr` is 32 bytes, so at most 1024 field elements fit in threadgroup
//!    memory: an NTT can keep only 2^10 points local per pass and must go back to
//!    device memory between passes.
//!
//! Apple's integer multiply is roughly 4x more expensive than NVIDIA's relative to
//! everything else, so the honest ceiling here is 2-3x over the CPU sharing the same
//! silicon, degrading toward 1.5-2x once witness generation is counted. This backend is
//! for client-side proving, not server throughput. Measure, do not hope.

/// The packed structs that cross into MSL. Host-side only, so it builds everywhere;
/// its definitions are mirrored by `src/shaders/bn254_fr.metal` and the two must be
/// changed together.
pub(crate) mod cb;
pub mod layout;

#[cfg(target_os = "macos")]
pub mod backend;
/// The three ceremony seams: `setup`'s MSM, `ptau prepare`'s group FFT, and the batch
/// apply-key behind the four contribute and beacon commands.
#[cfg(target_os = "macos")]
pub mod ceremony;
/// The group inverse FFT behind `ptau prepare`: `ceremony::MetalGroupFft` is its seam.
#[cfg(target_os = "macos")]
pub mod fft;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod msm;
/// Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
/// `H = A*B - C`, all in one command buffer with the domain vectors kept resident.
#[cfg(target_os = "macos")]
pub mod stages;

#[cfg(target_os = "macos")]
pub use backend::{MetalBackend, MetalCircuit, PrepareCost};
#[cfg(target_os = "macos")]
pub use ceremony::{MetalGroupFft, MetalKeyScale, MetalMsmBackend};
