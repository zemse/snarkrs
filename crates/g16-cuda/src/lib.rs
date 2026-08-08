//! NVIDIA CUDA backend. Deliberately a stub.
//!
//! The plan, when this gets built: do not write the MSM. sppark's Pippenger is
//! Apache-2.0, already skips zero digits and routes +/-1 digits around the bucket
//! machinery, and is roughly 8,000 lines of the hardest CUDA in the problem. Wrap it,
//! write the NTT and gather kernels, and keep this crate's surface identical to
//! `g16-metal`'s so `g16-core::Backend` stays the only contract.
//!
//! Not testable on this machine: an M2 Max has no NVIDIA GPU. This crate exists so the
//! feature-flag shape is settled while the CPU and Metal backends are being built, not
//! because it works.

#[cfg(feature = "cuda")]
compile_error!("the CUDA backend is not implemented yet");
