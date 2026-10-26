//! Apple Metal backend for BN254 Groth16, stages 0-9.
//!
//! Three facts about this platform shape the whole design, all verified on the M2 Max
//! this is developed against:
//!
//! 1. **MSL compiles at runtime.** `MTLDevice::newLibraryWithSource` uses the Metal
//!    runtime compiler that ships with the OS, so no offline `metal` toolchain download
//!    is needed. Kernels are embedded as source strings and compiled once at `prepare`.
//! 2. **Memory is unified.** A `StorageModeShared` buffer is host memory the GPU reads
//!    directly, so `prepare` packs each resident vector straight into its buffer's
//!    contents: one write per element, no host staging copy and no PCIe hop. The zkey is
//!    not wrapped in place. `newBufferWithBytesNoCopy` would need a page-aligned pointer
//!    and a page-multiple length at Apple's 16 KB pages, so only a wrap of the whole
//!    mapping with per-section offsets could work, and nothing here does that yet.
//! 3. **Threadgroup memory is 32 KB, SIMD width 32, max 1024 threads/threadgroup.**
//!    A BN254 `Fr` is 32 bytes, so at most 1024 field elements fit in threadgroup
//!    memory: an NTT can keep only 2^10 points local per pass and must go back to
//!    device memory between passes.
//!
//! Apple's integer multiply is roughly 4x more expensive than NVIDIA's relative to
//! everything else, so the honest ceiling here is 2-3x over the CPU sharing the same
//! silicon, degrading toward 1.5-2x once witness generation is counted. This backend is
//! for client-side proving, not server throughput. Measure, do not hope.

#[cfg(target_os = "macos")]
pub(crate) mod alloc;
#[cfg(target_os = "macos")]
pub(crate) mod cb;
/// The value checks on every ceremony readback: on the curve, and one point recomputed.
#[cfg(target_os = "macos")]
pub(crate) mod check;
#[cfg(target_os = "macos")]
pub(crate) mod enc;
/// The packed structs that cross into MSL. Host-side only, so it builds everywhere;
/// its definitions are mirrored by `src/shaders/bn254_fr.metal` and the two must be
/// changed together.
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
#[cfg(target_os = "macos")]
pub mod mtl;
/// Stages 0 to 4 on the GPU: the CSR gather, the six NTTs, the coset shift and
/// `H = A*B - C`, thirteen dispatches across four command buffers with the domain
/// vectors kept resident.
#[cfg(target_os = "macos")]
pub mod stages;

#[cfg(target_os = "macos")]
pub use backend::{MetalBackend, MetalCircuit, PrepareCost};
/// The per-thread fault injector behind every submission, for a test in another crate
/// that fails, cuts or corrupts a chosen one. Only with the `fault-inject` feature, which
/// no default build or shipping binary enables.
#[cfg(all(target_os = "macos", feature = "fault-inject"))]
pub use cb::inject;
#[cfg(target_os = "macos")]
pub use ceremony::{MetalGroupFft, MetalKeyScale, MetalMsmBackend};
/// The Metal bindings [`mtl`]'s handles come from, for a caller that builds or inspects one.
#[cfg(target_os = "macos")]
pub use objc2_metal;
