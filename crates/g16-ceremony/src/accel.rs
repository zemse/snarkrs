//! The CPU side of [`GroupFft`] and [`KeyScale`], and the reference every other
//! implementation of them is compared against.
//!
//! Both are the code the commands used to call directly, moved behind the trait and not
//! otherwise touched: the ceremony's whole claim is that its output is byte-identical to
//! snarkjs 0.7.6, so the CPU path a `--backend cpu` run takes has to be the same arithmetic
//! in the same order it was before there was a flag.

use g16_field::raw::{RawFq, RawFq2};
use g16_field::{AffineRepr, CurveGroup, Field, Fr, G1Affine, G2Affine};
use g16_msm::xyzz::Xyzz;
use g16_msm::{AccelError, GroupFft, KeyScale};
use rayon::prelude::*;

/// Points per rayon task inside one applied-key chunk.
///
/// The scalar sequence `first * inc^i` is a serial recurrence, so each task recovers its
/// own head with a single `inc^(k*SUBCHUNK)` exponentiation and then carries the product
/// forward locally. Per-point exponentiation would also parallelise but costs a 254-bit
/// `Fr` power per point next to one point multiplication, which is a few percent of the
/// work for nothing.
const KEY_SUBCHUNK: usize = 1024;

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
/// parallelises without changing which scalar any point gets.
fn apply_key<C>(points: &mut [C], first: Fr, inc: Fr)
where
    C: AffineRepr<ScalarField = Fr> + Send + Sync,
{
    points
        .par_chunks_mut(KEY_SUBCHUNK)
        .enumerate()
        .for_each(|(k, chunk)| {
            let mut t = first * inc.pow([(k * KEY_SUBCHUNK) as u64]);
            let scaled: Vec<C::Group> = chunk
                .iter()
                .map(|p| {
                    let q = *p * t;
                    t *= inc;
                    q
                })
                .collect();
            // One inversion per task instead of one per point, which is what makes the
            // affine output cheaper than the multiplications that produced it.
            chunk.copy_from_slice(&C::Group::normalize_batch(&scaled));
        });
}
