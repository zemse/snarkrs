//! A WebGPU backend for the BN254 Groth16 prover, through `wgpu`, on native and in a browser.
//!
//! # State: the field layer, the device layer, stage 0 and the NTT
//!
//! [`gen::field`] emits the whole BN254
//! field prelude (`Fr`, `Fq`, `Fq2`) as WGSL text. [`device`], [`pipelines`], [`params`] and
//! [`readback`] are the four pieces every kernel unit after this one is built on: a device at
//! a chosen limits profile, shader modules compiled once and timed, the uniform parameter
//! ring, and the one readback that works on both targets. [`gather`] and [`gen::gather`] are
//! stage 0, the CSR gather, restructured onto 7 storage buffers because the Metal original
//! binds 10 and the browser floor allows 8. [`ntt`] and [`gen::ntt`] are stages 1 to 3, the
//! six transforms with the bit-reverse, the `1/n` normalisation and the coset shift fused
//! into their loads, plus the stage 4 epilogue U7 fuses into the last of them.
//!
//! There is still no `Backend` implementation and no proof. Stage 4 is U7, the
//! MSM is U8 to U10, and U11 is where they become a backend.
//!
//! # Why a fourth backend exists at all
//!
//! Not to beat `g16-metal`. It cannot: the limb-width sweep measured the
//! WGSL Montgomery multiply at 1.964 G mul/s against MSL's 4.51 on the same M2 Max, 44%, and
//! that gap is structural. WGSL has no 64-bit integer and no widening multiply, gpuweb#1565
//! asking for them has been open since March 2021, and no amount of kernel tuning recovers
//! it. On a Mac, use the Metal backend.
//!
//! This backend exists so that one prover runs in a browser, on Windows and on Linux without
//! three more kernel languages. The number that matters is not Metal, it is snarkjs.
//!
//! # The two rules the whole crate is shaped around
//!
//! **Kernels are generated, not written.** A WGSL function-scope `array<u32, N>` indexed by a
//! loop variable does not stay in registers, and unrolling the limb loops is worth 2.6x to
//! 3.8x, more than any choice of limb width. The generator takes the field constants from
//! `g16-gpu-layout` at run time rather than from a second copy in shader source. See
//! [`gen::field`].
//!
//! **Nothing inherited from `g16-metal` is trusted without a measurement.** Three of the
//! three shape constants ported from MSL so far have been wrong here: `gather_abc`'s
//! workgroup size by up to 24%, the NTT's by up to 7.8x, and the NTT's "fuse as many passes
//! as the workgroup budget allows" by 5x. MSL and WGSL are different compilers targeting the
//! same silicon, and none of Metal's occupancy reasoning survives the trip. Every such
//! constant in this crate carries the table it was measured from and re-runs as a test.
//!
//! **Everything is sized for the browser floor.** The default device profile is
//! [`device::LimitsProfile::Floor`], the WebGPU spec defaults, not what this Mac's adapter
//! offers. A kernel that only fits native Metal's headroom is a kernel that fails in a stock
//! browser, and native `cargo test` will not tell you.

pub mod device;
pub mod gather;
pub mod gen;
pub mod ntt;
pub mod params;
pub mod pipelines;
pub mod readback;

pub use device::{LimitsProfile, WgpuBackend};
pub use gather::{CsrHost, CsrTables, GatherAbc, GatherParams};
pub use ntt::{
    split_passes, Batch, Direction, Epilogue, Ntt, NttParams, NttTables, Planned, Scale, Transform,
};
pub use params::ParamRing;
pub use pipelines::{Kernels, PrepareCost};
pub use readback::Readback;
