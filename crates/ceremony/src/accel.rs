//! The CPU side of [`GroupFft`] and [`KeyScale`], and the reference every other
//! implementation of them is compared against.
//!
//! Both are the code the commands used to call directly, moved behind the trait: the
//! ceremony's whole claim is that its output is byte-identical to snarkjs 0.7.6, so the CPU
//! path a `--backend cpu` run takes has to produce the same points it did before there was
//! a flag. Every output is affine, which is canonical, so how a point is computed is free.

use ark_ec::short_weierstrass::Affine;
use rayon::prelude::*;
use snarkrs_field::raw::{RawFq, RawFq2};
use snarkrs_field::{Field, Fr, G1Affine, G2Affine};
use snarkrs_msm::glv;
use snarkrs_msm::xyzz::Xyzz;
use snarkrs_msm::{AccelError, GroupFft, KeyScale};

use crate::prepare::PrepareCurve;

/// Points per rayon task inside one applied-key chunk, which is also the batch
/// [`glv::mul_batch`] shares its inversions across.
///
/// The scalar sequence `first * inc^i` is a serial recurrence, so each task recovers its
/// own head with a single `inc^(k*SUBCHUNK)` exponentiation and then carries the product
/// forward locally. Per-point exponentiation would also parallelise but costs a 254-bit
/// `Fr` power per point next to one point multiplication, which is a few percent of the
/// work for nothing. The batch wants to be large: an addition step involves the sixth of
/// it whose digit is nonzero, and one inversion is spread over those. Phase 1 calls with
/// sixteen response chunks (262144 G1 or 131072 G2 points), zkey contribute with 65536.
const KEY_SUBCHUNK: usize = 2048;

/// The group FFT on the CPU: [`crate::prepare::ifft`] behind the trait.
pub struct CpuGroupFft;

impl GroupFft for CpuGroupFft {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
        crate::prepare::ifft(a);
        Ok(())
    }

    fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
        crate::prepare::ifft(a);
        Ok(())
    }
}

/// `batchApplyKey` on the CPU, for both phases.
pub struct CpuKeyScale;

impl KeyScale for CpuKeyScale {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn apply_key_g1(&self, points: &mut [G1Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        apply_key(points, first, inc);
        Ok(())
    }

    fn apply_key_g2(&self, points: &mut [G2Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        apply_key(points, first, inc);
        Ok(())
    }
}

/// `P_i *= first * inc^i` over one chunk, in place.
///
/// One `Fr` exponentiation per task and then a running product, so the recurrence
/// parallelises without changing which scalar any point gets. The products are
/// [`glv::mul_batch`]: GLV on raw limbs, every point of the task in lockstep in affine
/// coordinates with one shared inversion a step, where arkworks' affine multiply is plain
/// double-and-add over all 254 bits. Affine is canonical, so the bytes are the same.
fn apply_key<P: PrepareCurve>(points: &mut [Affine<P>], first: Fr, inc: Fr) {
    points
        .par_chunks_mut(KEY_SUBCHUNK)
        .enumerate()
        .for_each(|(k, chunk)| {
            let mut t = first * inc.pow([(k * KEY_SUBCHUNK) as u64]);
            let scalars: Vec<Fr> = (0..chunk.len())
                .map(|_| {
                    let s = t;
                    t *= inc;
                    s
                })
                .collect();
            let input = chunk.to_vec();
            glv::mul_batch(&input, &scalars, chunk);
        });
}
