//! BN254 types and the evaluation domain, shared by every crate and every backend.
//!
//! Field and curve arithmetic is arkworks'. Everything above it is ours. This crate
//! exists so the rest of the workspace never names `ark_bn254` directly, which keeps
//! a future from-scratch field backend a one-file swap.

pub mod raw;

pub use ark_bn254::{g1, g2, Bn254, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
pub use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, PrimeGroup};
pub use ark_ff::{BigInteger, FftField, Field, One, PrimeField, UniformRand, Zero};

/// Whether this is an x86_64 build with `+adx,+bmi2`. That buys LLVM's mulx, not ark-ff's
/// assembly multiply, which BN254 never reaches (see this crate's Cargo.toml).
pub const ADX_BMI2: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2"
));

/// A radix-2 evaluation domain over `Fr`.
///
/// The prover needs the `size`-th roots of unity for the NTT and their inverses for the
/// iNTT. The coset shift comes from [`Domain::proving_coset_shift`], not from a coset
/// generator such as `Fr::GENERATOR`: that is a valid coset which snarkjs rejects.
///
/// The fields are private so every `Domain` comes out of [`Domain::new`]: `size` is a
/// power of two, `log_size` is its log, and the three field elements are the ones derived
/// from it. Consumers rely on that instead of re-checking it.
#[derive(Clone, Debug)]
pub struct Domain {
    size: usize,
    log_size: u32,
    group_gen: Fr,
    group_gen_inv: Fr,
    size_inv: Fr,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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

    /// Number of points, a power of two.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// `log2(size)`.
    #[inline]
    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Primitive `size`-th root of unity.
    #[inline]
    pub fn group_gen(&self) -> Fr {
        self.group_gen
    }

    /// Inverse of [`Self::group_gen`].
    #[inline]
    pub fn group_gen_inv(&self) -> Fr {
        self.group_gen_inv
    }

    /// `Fr::from(size).inverse()`, the iNTT normalisation factor.
    #[inline]
    pub fn size_inv(&self) -> Fr {
        self.size_inv
    }

    /// The shift of the coset the prover evaluates A, B and C on: snarkjs' `inc`, a primitive
    /// `2*size`-th root of unity whose square is [`Self::group_gen`].
    ///
    /// snarkjs evaluates on the odd points of the `2*size`-th roots of unity (`inc =
    /// Fr.w[power+1]` in groth16_prove.js), and section 9 of the zkey holds the Lagrange
    /// basis of that same `2*size` domain restricted to its odd indices (`writeHs` in
    /// zkey_new.js takes `sTauG1[2i+1]`). Evaluation index `i` must land on `inc^(2i+1)`,
    /// which needs `Fr.w[power] == inc^2`. A shift built from `Fr::GENERATOR` is a valid
    /// coset but pairs the evaluations with the wrong Lagrange polynomials, so the proof
    /// fails to verify.
    pub fn proving_coset_shift(&self) -> Result<Fr, DomainError> {
        let doubled = self
            .size
            .checked_mul(2)
            .ok_or(DomainError::TooLarge(self.size))?;
        Ok(Domain::new(doubled)?.group_gen)
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
    fn proving_coset_shift_is_the_2n_th_root_and_squares_to_group_gen() {
        for d in domains() {
            let shift = d.proving_coset_shift().unwrap();
            let doubled = Domain::new(2 * d.size()).unwrap();
            assert_eq!(shift, doubled.group_gen(), "n = {}", d.size());
            assert_eq!(shift * shift, d.group_gen(), "n = {}", d.size());
        }
    }

    #[test]
    fn proving_coset_shift_fails_past_the_two_adicity() {
        let d = Domain::new(1 << Fr::TWO_ADICITY).unwrap();
        assert!(matches!(
            d.proving_coset_shift(),
            Err(DomainError::TooLarge(_))
        ));
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
