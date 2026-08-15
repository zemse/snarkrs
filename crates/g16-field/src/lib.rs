//! BN254 types and the evaluation domain, shared by every crate and every backend.
//!
//! Field and curve arithmetic is arkworks'. Everything above it is ours. This crate
//! exists so the rest of the workspace never names `ark_bn254` directly, which keeps
//! a future from-scratch field backend a one-file swap.

pub use ark_bn254::{Bn254, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
pub use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, PrimeGroup};
pub use ark_ff::{BigInteger, FftField, Field, One, PrimeField, UniformRand, Zero};

/// A radix-2 evaluation domain over `Fr`.
///
/// The prover needs the `size`-th roots of unity for the NTT and their inverses for the
/// iNTT. It does NOT get a coset generator from here, on purpose: the snarkjs-compatible
/// H coset is the odd half of the `2*size`-th roots of unity, so each backend derives its
/// shift as a primitive `2*size`-th root whose square is `group_gen` (see
/// `CpuCircuit::new` in g16-core for the full argument). An earlier version of this type
/// exposed `coset_gen = Fr::GENERATOR` and `vanishing_on_coset()`, which describe a
/// mathematically valid but snarkjs-INCOMPATIBLE coset: nothing on the proving path used
/// them, and pairing them with section-9 H bases would produce proofs that fail to
/// verify, so the footgun was removed rather than documented around.
#[derive(Clone, Debug)]
pub struct Domain {
    pub size: usize,
    pub log_size: u32,
    /// Primitive `size`-th root of unity.
    pub group_gen: Fr,
    pub group_gen_inv: Fr,
    /// `Fr::from(size).inverse()`, the iNTT normalisation factor.
    pub size_inv: Fr,
}

#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("domain size {0} exceeds the 2-adicity of the BN254 scalar field")]
    TooLarge(usize),
}

impl Domain {
    /// Smallest power-of-two domain with at least `min_size` points.
    pub fn new(min_size: usize) -> Result<Self, DomainError> {
        // A zero-length constraint system still needs one evaluation point, and
        // `checked_next_power_of_two` is the only rounding that cannot panic on a
        // bogus caller-supplied length.
        let size = min_size
            .max(1)
            .checked_next_power_of_two()
            .ok_or(DomainError::TooLarge(min_size))?;
        let log_size = size.trailing_zeros();

        // BN254's Fr has 2-adicity 28, so 2^28 is the largest radix-2 domain that
        // exists at all. Anything bigger has no primitive root to build from.
        if log_size > Fr::TWO_ADICITY {
            return Err(DomainError::TooLarge(size));
        }

        // TWO_ADIC_ROOT_OF_UNITY has order exactly 2^28. Each squaring halves the
        // order, so 28 - log_size squarings land on a primitive `size`-th root.
        let mut group_gen = Fr::TWO_ADIC_ROOT_OF_UNITY;
        for _ in log_size..Fr::TWO_ADICITY {
            group_gen.square_in_place();
        }

        Ok(Self {
            size,
            log_size,
            group_gen,
            // Both inverses are of nonzero elements: a root of unity is never zero,
            // and `size` is a power of two far below the modulus.
            group_gen_inv: group_gen.inverse().expect("root of unity is nonzero"),
            size_inv: Fr::from(size as u64)
                .inverse()
                .expect("domain size is nonzero mod r"),
        })
    }

    /// Forward twiddles: `[1, g, g^2, ..., g^(size/2 - 1)]`.
    pub fn twiddles(&self) -> Vec<Fr> {
        powers(self.group_gen, self.size / 2)
    }

    /// Inverse twiddles, for the iNTT.
    pub fn twiddles_inv(&self) -> Vec<Fr> {
        powers(self.group_gen_inv, self.size / 2)
    }
}

/// Running product rather than `pow` per index: one multiplication per twiddle instead
/// of a full square-and-multiply ladder, which matters at 2^24 points.
fn powers(base: Fr, n: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(n);
    let mut acc = Fr::ONE;
    for _ in 0..n {
        out.push(acc);
        acc *= base;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domains() -> Vec<Domain> {
        [1usize, 2, 8, 64, 1024]
            .iter()
            .map(|&n| Domain::new(n).unwrap())
            .collect()
    }

    #[test]
    fn rounds_up_to_a_power_of_two() {
        for (min, want_size, want_log) in [
            (0usize, 1usize, 0u32),
            (1, 1, 0),
            (3, 4, 2),
            (5, 8, 3),
            (1024, 1024, 10),
            (1025, 2048, 11),
        ] {
            let d = Domain::new(min).unwrap();
            assert_eq!(
                (d.size, d.log_size),
                (want_size, want_log),
                "min_size {min}"
            );
        }
    }

    #[test]
    fn rejects_sizes_past_the_two_adicity() {
        let max = 1usize << Fr::TWO_ADICITY;
        assert!(Domain::new(max).is_ok());
        assert!(matches!(
            Domain::new(max + 1),
            Err(DomainError::TooLarge(_))
        ));
    }

    #[test]
    fn group_gen_has_exact_order_size() {
        for d in domains() {
            assert_eq!(d.group_gen.pow([d.size as u64]), Fr::ONE, "size {}", d.size);
            if d.size > 1 {
                // Order is a divisor of size, so a primitive root is the one that
                // survives the single largest proper divisor.
                assert_ne!(
                    d.group_gen.pow([(d.size / 2) as u64]),
                    Fr::ONE,
                    "size {}",
                    d.size
                );
            }
            assert_eq!(d.group_gen * d.group_gen_inv, Fr::ONE);
        }
    }

    #[test]
    fn twiddles_are_powers_of_group_gen() {
        for d in domains() {
            let tw = d.twiddles();
            let tw_inv = d.twiddles_inv();
            assert_eq!(tw.len(), d.size / 2);
            assert_eq!(tw_inv.len(), d.size / 2);
            for (i, (&w, &w_inv)) in tw.iter().zip(tw_inv.iter()).enumerate() {
                assert_eq!(w, d.group_gen.pow([i as u64]), "size {} index {i}", d.size);
                assert_eq!(w * w_inv, Fr::ONE, "size {} index {i}", d.size);
            }
        }
    }

    #[test]
    fn size_inv_normalises() {
        for d in domains() {
            assert_eq!(
                d.size_inv * Fr::from(d.size as u64),
                Fr::ONE,
                "size {}",
                d.size
            );
        }
    }

    #[test]
    fn subgroup_has_size_distinct_elements() {
        for d in domains() {
            let subgroup: Vec<Fr> = powers(d.group_gen, d.size);
            // `size` distinct elements is what "group of order size" means; a root of
            // too-small order would fold the subgroup onto itself and every NTT built
            // from it would silently alias.
            for i in 0..subgroup.len() {
                for j in (i + 1)..subgroup.len() {
                    assert_ne!(
                        subgroup[i], subgroup[j],
                        "size {} collision {i},{j}",
                        d.size
                    );
                }
            }
        }
    }

    #[test]
    fn vanishing_is_zero_on_the_domain_itself() {
        let d = Domain::new(16).unwrap();
        for h in powers(d.group_gen, d.size) {
            assert!((h.pow([d.size as u64]) - Fr::ONE).is_zero());
        }
    }
}
