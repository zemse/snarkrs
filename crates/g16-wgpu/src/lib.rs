//! A WebGPU backend for the BN254 Groth16 prover, through `wgpu`, on native and in a browser.
//!
//! # State: stages 0 to 4, and the front half of the MSM
//!
//! [`gen::field`] emits the whole BN254
//! field prelude (`Fr`, `Fq`, `Fq2`) as WGSL text. [`device`], [`pipelines`], [`params`] and
//! [`readback`] are the four pieces every kernel unit after this one is built on: a device at
//! a chosen limits profile, shader modules compiled once and timed, the uniform parameter
//! ring, and the one readback that works on both targets. [`gather`] and [`gen::gather`] are
//! stage 0, the CSR gather, restructured onto 7 storage buffers because the Metal original
//! binds 10 and the browser floor allows 8. [`ntt`] and [`gen::ntt`] are stages 1 to 3, the
//! six transforms with the bit-reverse, the `1/n` normalisation and the coset shift fused
//! into their loads. [`stages`] is U7: it drives all of stages 0 to 4 into **one submit**
//! and hands back [`g16_core::HPoly::Device`] carrying a [`stages::WgpuHandle`], so `H`
//! never touches the host. [`pointwise`] and [`gen::pointwise`] are stage 4, and they are
//! what the prover dispatches: the fused NTT epilogue the design specifies is implemented
//! and tested here too, and it measured slower. See [`stages::Stage4`]. [`msm`] and
//! [`gen::msm`] are U8, the front half of stages 5 to 9: the signed-digit recoding, the 0/1
//! classification and the counting sort that makes one thread own one bucket by construction,
//! so no 64-bit atomic is ever needed. It touches no curve arithmetic.
//!
//! [`points`] and [`gen::points`] are the back half, U10 and U9: the XYZZ curve arithmetic
//! and the five point kernels, emitted once per curve and instantiated for both BN254 groups.
//! [`MsmPointsG1`] is four of a proof's five MSMs (A, B-G1, L, H) and [`MsmPointsG2`] is the
//! fifth.
//!
//! There is still no `Backend` implementation and no proof. U11 is where the two halves of
//! an MSM become one.
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
//! **Nothing inherited from `g16-metal` is trusted without a measurement.** Five of the six
//! shape decisions ported from MSL so far have been wrong here: `gather_abc`'s workgroup size
//! by up to 24%, the NTT's by up to 7.8x, the NTT's "fuse as many passes as the workgroup
//! budget allows" by 5x, stage 4's fusion into the NTT store by 5% to 9%, and four of the five
//! digit-pipeline workgroup sizes, though only by 2% to 7% ([`gen::msm::Workgroups`]). MSL and
//! WGSL are different compilers targeting the same silicon, and almost none of Metal's
//! occupancy reasoning survives the trip. Every such constant in this crate carries the table
//! it was measured from and re-runs as a test.
//!
//! **The design is not trusted either, and it has now been wrong about a mitigation rather
//! than just a constant.** §4's "one shader module per group of kernels" is 17% *slower* cold
//! than putting all five MSM digit kernels in one module, because Metal compiles a pipeline
//! per entry point and dead-strips the module around it. See [`msm::ModuleShape`].
//!
//! **Everything is sized for the browser floor.** The default device profile is
//! [`device::LimitsProfile::Floor`], the WebGPU spec defaults, not what this Mac's adapter
//! offers. A kernel that only fits native Metal's headroom is a kernel that fails in a stock
//! browser, and native `cargo test` will not tell you.

pub mod device;
pub mod gather;
pub mod gen;
pub mod msm;
pub mod ntt;
pub mod params;
pub mod pipelines;
pub mod points;
pub mod pointwise;
pub mod readback;
pub mod stages;

pub use device::{LimitsProfile, WgpuBackend};
pub use gather::{CsrHost, CsrTables, GatherAbc, GatherParams};
pub use msm::{window_size, DigitBuffers, DigitPlan, MsmDigits, MsmParams, SortBinds, SortOffsets};
pub use ntt::{
    split_passes, Batch, Direction, Epilogue, Ntt, NttParams, NttTables, Planned, Scale, Transform,
};
pub use params::ParamRing;
pub use pipelines::{Kernels, PrepareCost};
pub use points::{
    xyzz_g1_from_bytes, xyzz_g2_from_bytes, G1Curve, G2Curve, MsmPoints, MsmPointsG1, MsmPointsG2,
    PointBinds, PointBuffers, PointCurve, PointOffsets, PointPlan,
};
pub use pointwise::{HJoin, HJoinParams};
pub use readback::Readback;
pub use stages::{HStages, Stage4, WgpuHandle};
