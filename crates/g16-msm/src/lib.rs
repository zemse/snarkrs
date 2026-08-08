//! Pippenger multi-scalar multiplication over BN254 G1 and G2.
//!
//! Five MSMs dominate the proof (~53% of a fast CPU prover's wall clock at 151K
//! constraints, ~70% of a slow one's), so this is the crate that decides whether the
//! project is worth anything.
//!
//! Two properties matter beyond raw speed, both because the witness is the scalar vector
//! for four of the five MSMs:
//!   * a scalar of 0 must cost nothing beyond the digit scan,
//!   * a scalar of 1 must cost exactly one mixed addition, not a full bucket round trip.
//! In bit-decomposition-heavy circuits over 99% of witness scalars are 0 or 1, so these
//! two paths are not micro-optimisations, they are most of the work.

use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

pub trait MsmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective;
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective;
}

/// Window size for `n` points. Pippenger's cost is minimised near `ln(n)`; sppark uses
/// `min(floor(lg2(1.5n)) - 8, 18)` floored at 10 on GPU, but the CPU optimum is smaller
/// because there is no bucket-sort machinery to amortise.
pub fn window_size(n: usize) -> u32 {
    todo!("g16-msm: implement window_size")
}

pub struct CpuMsm {
    pub threads: usize,
}

impl CpuMsm {
    pub fn new() -> Self {
        todo!("g16-msm: implement CpuMsm::new")
    }
}

impl MsmBackend for CpuMsm {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn msm_g1(&self, _bases: &[G1Affine], _scalars: &[Fr]) -> G1Projective {
        todo!("g16-msm: implement CpuMsm::msm_g1")
    }
    fn msm_g2(&self, _bases: &[G2Affine], _scalars: &[Fr]) -> G2Projective {
        todo!("g16-msm: implement CpuMsm::msm_g2")
    }
}
