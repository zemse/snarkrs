//! The H query's change of basis between gnark's `G1.Z` and snarkjs' section 9, both ways,
//! as one inverse FFT over G1 and `n` scalar multiplications. See the crate docs for the
//! identity both directions rest on.
//!
//! The transform is the ceremony's: a [`GroupFft`] runs `out_i = 1/n sum_j omega^-ij a_j`
//! over XYZZ points, every twiddle a GLV scalar multiplication, so `n/2 log n` of those plus
//! the `n` here. [`snarkrs_ceremony::CpuGroupFft`] is the CPU one; a device backend that
//! implements the trait for `powersoftau prepare phase2` takes the same call.

use crate::{bitrev, mismatch, ConvertError};
use rayon::prelude::*;
use snarkrs_ceremony::prepare::batch_to_affine;
use snarkrs_field::raw::RawFq;
use snarkrs_field::*;
use snarkrs_msm::glv;
use snarkrs_msm::xyzz::{RawCurve, Xyzz};
use snarkrs_msm::GroupFft;

/// Points one task scales, from one exponentiation of its first factor.
const SCALE_CHUNK: usize = 1 << 12;

/// gnark's `G1.Z`, `n - 1` points in its bit-reversed order, to snarkjs' section 9 for the
/// same key: `n` points, `h_i = -1/(2n) sum_{j < n-1} (s omega^i)^-j Z_j`.
pub fn zkey_h_from_gnark_z(
    z: &[G1Affine],
    fft: &dyn GroupFft,
) -> Result<Vec<G1Affine>, ConvertError> {
    let n = z.len() + 1;
    let (bits, shift) = domain(n)?;
    // v_j = -1/2 s^-j Z_j, and v_{n-1} = 0 for the Z_{n-1} gnark does not keep.
    let factor = -Fr::from(2u64).inverse().expect("2 is a unit mod r");
    let step = shift.inverse().expect("a root of unity is nonzero");
    let mut v = vec![Xyzz::<RawFq>::ZERO; n];
    scale(&mut v[..n - 1], factor, step, |j| xyzz(&z[bitrev(j, bits)]));
    ifft(fft, &mut v)?;
    Ok(batch_to_affine::<g1::Config>(&v))
}

/// snarkjs' section 9, `n` points, to gnark's `G1.Z` for the same key: `n - 1` points in
/// gnark's bit-reversed order, `Z_j = -2 sum_i (s omega^i)^j h_i`.
pub fn gnark_z_from_zkey_h(
    h: &[G1Affine],
    fft: &dyn GroupFft,
) -> Result<Vec<G1Affine>, ConvertError> {
    let n = h.len();
    let (bits, shift) = domain(n)?;
    let mut v: Vec<Xyzz<RawFq>> = h.par_iter().map(xyzz).collect();
    ifft(fft, &mut v)?;
    // The forward DFT is the inverse one read backwards and times n:
    // sum_i omega^ij h_i = n v_{-j}. So Z_j = -2n s^j v_{(n - j) mod n}.
    let factor = -Fr::from(2 * n as u64);
    let mut natural = vec![Xyzz::<RawFq>::ZERO; n - 1];
    scale(&mut natural, factor, shift, |j| v[(n - j) % n]);
    drop(v);
    let natural = batch_to_affine::<g1::Config>(&natural);
    // `bitrev(n - 1) = n - 1`, the slot gnark drops, so every other slot is in range.
    Ok((0..n - 1)
        .into_par_iter()
        .map(|k| natural[bitrev(k, bits)])
        .collect())
}

/// The transform on `fft`, or on the CPU below the block size `fft` says it loses on.
fn ifft(fft: &dyn GroupFft, a: &mut [Xyzz<RawFq>]) -> Result<(), ConvertError> {
    if a.len() >= fft.min_block() {
        fft.ifft_g1(a)?;
    } else {
        snarkrs_ceremony::CpuGroupFft.ifft_g1(a)?;
    }
    Ok(())
}

/// `log2(n)` and the coset shift, for a domain both libraries can hold a key on.
fn domain(n: usize) -> Result<(u32, Fr), ConvertError> {
    // Two points at least: on a one-point domain H is zero, and a zkey whose section 9 is
    // all infinity is refused as zeroed. 2^27 at most, for the 2n-th root.
    if n < 2 || !n.is_power_of_two() || n.trailing_zeros() >= Fr::TWO_ADICITY {
        return Err(mismatch(
            "domain",
            format!(
                "{n} is not a power of two from 2 to 2^{}",
                Fr::TWO_ADICITY - 1
            ),
        ));
    }
    let d = Domain::new(n).map_err(|e| mismatch("domain", e.to_string()))?;
    let shift = d
        .proving_coset_shift()
        .map_err(|e| mismatch("domain", e.to_string()))?;
    Ok((n.trailing_zeros(), shift))
}

/// `out[j] = [factor * step^j] point(j)`, in parallel, each task starting its run of
/// factors with one exponentiation.
fn scale(
    out: &mut [Xyzz<RawFq>],
    factor: Fr,
    step: Fr,
    point: impl Fn(usize) -> Xyzz<RawFq> + Sync,
) {
    out.par_chunks_mut(SCALE_CHUNK)
        .enumerate()
        .for_each(|(c, out)| {
            let first = c * SCALE_CHUNK;
            let mut f = factor * step.pow([first as u64]);
            for (j, o) in (first..).zip(out.iter_mut()) {
                *o = glv::mul(&point(j), &f);
                f *= step;
            }
        });
}

fn xyzz(p: &G1Affine) -> Xyzz<RawFq> {
    if p.infinity {
        return Xyzz::ZERO;
    }
    let (x, y) = g1::Config::raw_xy(p);
    Xyzz {
        x,
        y,
        zz: RawFq::ONE,
        zzz: RawFq::ONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_ceremony::CpuGroupFft;

    /// Both bases from one tau and delta, the way each setup computes them: section 9 from
    /// the odd Lagrange polynomials of the 2n-th roots, `G1.Z` from powers of tau.
    fn bases(n: usize, tau: Fr, delta: Fr) -> (Vec<G1Affine>, Vec<G1Affine>) {
        let g = G1Affine::generator();
        let d_inv = delta.inverse().unwrap();
        let s = Domain::new(n).unwrap().proving_coset_shift().unwrap();
        let t = tau.pow([n as u64]) - Fr::ONE;
        // L_k(x) = (x^2n - 1) / 2n * w^k / (x - w^k) on the 2n roots w = s.
        let scale = (tau.pow([2 * n as u64]) - Fr::ONE) / Fr::from(2 * n as u64);
        let h = (0..n)
            .map(|i| {
                let w = s.pow([2 * i as u64 + 1]);
                (g * (scale * w / (tau - w) * d_inv)).into_affine()
            })
            .collect();
        let bits = n.trailing_zeros();
        let mut z = vec![G1Affine::identity(); n - 1];
        for (k, slot) in z.iter_mut().enumerate() {
            let j = bitrev(k, bits);
            *slot = (g * (tau.pow([j as u64]) * t * d_inv)).into_affine();
        }
        (h, z)
    }

    #[test]
    fn zkey_h_to_gnark_z_is_exact() {
        for n in [2, 4, 16, 256] {
            let (h, z) = bases(n, Fr::from(123456789u64), Fr::from(987654321u64));
            assert_eq!(gnark_z_from_zkey_h(&h, &CpuGroupFft).unwrap(), z, "n = {n}");
        }
    }

    /// Not the original section 9, which needs the `Z_{n-1}` gnark drops, but one that pairs
    /// every quotient of degree `n - 2` or less with the same point.
    #[test]
    fn gnark_z_to_zkey_h_proves_what_the_original_proves() {
        for n in [2, 4, 16, 256] {
            let (h, z) = bases(n, Fr::from(31337u64), Fr::from(271828u64));
            let converted = zkey_h_from_gnark_z(&z, &CpuGroupFft).unwrap();
            assert_eq!(converted.len(), n);
            let s = Domain::new(n).unwrap().proving_coset_shift().unwrap();
            let omega = Domain::new(n).unwrap().group_gen();
            // A quotient of the top degree, evaluated where a prover pairs it with section 9:
            // P(s omega^i) = H(s omega^i) t(s omega^i) = -2 H(s omega^i).
            let coeffs: Vec<Fr> = (0..n - 1).map(|k| Fr::from(3 * k as u64 + 5)).collect();
            let p: Vec<Fr> = (0..n)
                .map(|i| {
                    let x = s * omega.pow([i as u64]);
                    -Fr::from(2u64) * coeffs.iter().rev().fold(Fr::zero(), |acc, c| acc * x + c)
                })
                .collect();
            let msm = |bases: &[G1Affine]| -> G1Projective {
                bases.iter().zip(&p).map(|(b, e)| *b * e).sum()
            };
            assert_eq!(msm(&converted), msm(&h), "n = {n}");
            // And gnark's side of the same proof term, sum_j H_j Z_j.
            let gnark: G1Projective = (0..n - 1)
                .map(|j| z[bitrev(j, n.trailing_zeros())] * coeffs[j])
                .sum();
            assert_eq!(msm(&converted), gnark, "n = {n}");
            // Round trip: the forward transform recovers gnark's points exactly.
            assert_eq!(gnark_z_from_zkey_h(&converted, &CpuGroupFft).unwrap(), z);
        }
    }

    #[test]
    fn refuses_a_domain_it_cannot_hold() {
        assert!(zkey_h_from_gnark_z(&[], &CpuGroupFft).is_err());
        assert!(gnark_z_from_zkey_h(&[G1Affine::generator(); 3], &CpuGroupFft).is_err());
    }
}
