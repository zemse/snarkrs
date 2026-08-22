//! A WebGPU backend for the BN254 Groth16 prover, through `wgpu`, on native and in a browser.
//!
//! # State: the field layer only
//!
//! This crate currently contains the WGSL generator and the crate skeleton. There is no
//! device, no pipeline, no kernel and no `Backend` implementation yet. Those come later.
//! What is here is [`gen::field`], which
//! emits the whole BN254 field prelude (`Fr`, `Fq`, `Fq2`) as WGSL text, plus a test that
//! runs the emitted multiply on a real GPU and checks it against `ark-ff` bit for bit.
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
//! # The one rule the whole crate is shaped around
//!
//! A WGSL function-scope `array<u32, N>` indexed by a loop variable does not stay in
//! registers, and unrolling the limb loops is worth 2.6x to 3.8x, more than any choice of
//! limb width. So the kernels are **generated**, and the generator takes the field constants
//! from `g16-gpu-layout` at run time rather than from a second copy in shader source. See
//! [`gen::field`].

pub mod gen;
