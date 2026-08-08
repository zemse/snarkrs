//! BN254 types and the evaluation domain, shared by every crate and every backend.
//!
//! Field and curve arithmetic is arkworks'. Everything above it is ours. This crate
//! exists so the rest of the workspace never names `ark_bn254` directly, which keeps
//! a future from-scratch field backend a one-file swap.

pub use ark_bn254::{Bn254, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
pub use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, pairing::Pairing};
pub use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};

/// A radix-2 evaluation domain over `Fr`, plus the coset generator used by Jordi's trick.
///
/// The prover needs three things from a domain: the `size`-th roots of unity for the
/// NTT, their inverses for the iNTT, and a generator of a *disjoint* coset so that
/// `H(X)*Z(X)` can be evaluated where `Z` is nonzero. See the article's Proving section.
#[derive(Clone, Debug)]
pub struct Domain {
    pub size: usize,
    pub log_size: u32,
    /// Primitive `size`-th root of unity.
    pub group_gen: Fr,
    pub group_gen_inv: Fr,
    /// `Fr::from(size).inverse()`, the iNTT normalisation factor.
    pub size_inv: Fr,
    /// Multiplicative generator of `Fr*`, used to shift onto a disjoint coset.
    pub coset_gen: Fr,
}

#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("domain size {0} exceeds the 2-adicity of the BN254 scalar field")]
    TooLarge(usize),
}

impl Domain {
    /// Smallest power-of-two domain with at least `min_size` points.
    pub fn new(min_size: usize) -> Result<Self, DomainError> {
        todo!("g16-field: implement Domain::new")
    }

    /// Forward twiddles: `[1, g, g^2, ..., g^(size/2 - 1)]`.
    pub fn twiddles(&self) -> Vec<Fr> {
        todo!("g16-field: implement Domain::twiddles")
    }

    /// Inverse twiddles, for the iNTT.
    pub fn twiddles_inv(&self) -> Vec<Fr> {
        todo!("g16-field: implement Domain::twiddles_inv")
    }

    /// `Z(coset_gen) = coset_gen^size - 1`, the vanishing polynomial evaluated on the
    /// coset. Constant across the whole coset, which is exactly why the trick works.
    pub fn vanishing_on_coset(&self) -> Fr {
        todo!("g16-field: implement Domain::vanishing_on_coset")
    }
}
