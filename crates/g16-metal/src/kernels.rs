//! MSL kernel sources, compiled at runtime.
//!
//! Everything below `bn254_fr.metal` is field arithmetic we must write ourselves: MSL has
//! no 256-bit integers, so `Fr` is 4 x `uint64` (or 8 x `uint32`, which is often faster on
//! Apple GPUs because the 64-bit multiply is emulated) with CIOS Montgomery multiplication.

/// Shared prelude: `Fr` as 8 x u32 limbs, add/sub/Montgomery-mul, BN254 scalar modulus.
pub const FR_MSL: &str = include_str!("shaders/bn254_fr.metal");
/// Stage 0: CSR gather.
pub const GATHER_MSL: &str = include_str!("shaders/gather.metal");
/// Stages 1-3: radix-2 NTT passes and the coset shift.
pub const NTT_MSL: &str = include_str!("shaders/ntt.metal");
/// Stage 4: `H = A*B - C`.
pub const POINTWISE_MSL: &str = include_str!("shaders/pointwise.metal");
/// Stages 5-9: G1/G2 point arithmetic and Pippenger buckets.
pub const MSM_MSL: &str = include_str!("shaders/msm.metal");
/// The ceremony primitives that are not a multiexp: a general point scalar
/// multiplication, batch projective-to-affine, and the batch apply-key. Reuses the point
/// arithmetic in `msm.metal`, so it must be concatenated after it.
pub const CEREMONY_MSL: &str = include_str!("shaders/ceremony.metal");
