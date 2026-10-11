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

use crate::prepare::{batch_to_affine_serial, PrepareCurve};

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
/// parallelises without changing which scalar any point gets. Each product is
/// [`glv::mul_affine`] on raw limbs rather than arkworks' affine multiply, which is plain
/// double-and-add over all 254 bits; affine is canonical, so the bytes are the same.
fn apply_key<P: PrepareCurve>(points: &mut [Affine<P>], first: Fr, inc: Fr) {
    points
        .par_chunks_mut(KEY_SUBCHUNK)
        .enumerate()
        .for_each(|(k, chunk)| {
            let mut t = first * inc.pow([(k * KEY_SUBCHUNK) as u64]);
            let scaled: Vec<Xyzz<P::RF>> = chunk
                .iter()
                .map(|p| {
                    let q = glv::mul_affine(p, &t);
                    t *= inc;
                    q
                })
                .collect();
            // One inversion per task instead of one per point, which is what makes the
            // affine output cheaper than the multiplications that produced it.
            batch_to_affine_serial::<P>(&scaled, chunk);
        });
}
