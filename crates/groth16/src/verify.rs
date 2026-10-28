//! Groth16 verification.
//!
//! `e(A, B) = e(alpha, beta) * e(L_bar, gamma) * e(C, delta)`, rearranged into a single
//! 4-pair multi-Miller loop with `A` negated so the check is `product == 1` and only one
//! final exponentiation is needed.

use crate::Proof;
use ark_ec::pairing::MillerLoopOutput;
use rayon::prelude::*;
use snarkrs_field::raw::RawFq;
use snarkrs_field::*;
use snarkrs_formats::VerifyingKey;
use snarkrs_msm::glv;
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    #[error("expected {want} public inputs, got {got}")]
    PublicInputCount { got: usize, want: usize },
    #[error("verifying key has an empty IC; it must carry at least the constant-wire point")]
    EmptyVerifyingKey,
    #[error("pairing check failed")]
    PairingFailed,
    /// A proof element that no honest prover emits: off the curve, outside the prime-order
    /// subgroup, or the point at infinity.
    #[error("proof element {0}")]
    InvalidProof(&'static str),
    #[error(transparent)]
    UnsafeKey(#[from] snarkrs_formats::ZkeyError),
}

/// Check a proof against a verifying key and public inputs, validating both first.
///
/// `A` and `C` must be on the curve, `B` on the curve and in the prime-order subgroup,
/// and none of the three the point at infinity. The key must pass
/// [`VerifyingKey::check_structure`]. Together these cost about one G2 subgroup check,
/// small against the pairing, and they are what make the result a guarantee of this
/// function rather than of whichever loader the caller happened to use: `Proof`'s and
/// `VerifyingKey`'s fields are public, so a value built by hand or over FFI would
/// otherwise reach the Miller loop unchecked. The Groth16 soundness argument stops
/// applying the moment an off-subgroup element does.
///
/// Infinity is refused outright rather than left to the pairing, because arkworks drops a
/// pair with a zero element from the product, which turns the 4-pair check into a
/// 3-pair one instead of failing it.
///
/// A proof that verifies is malleable. See [`Proof`] for why that matters to the caller.
pub fn verify(vk: &VerifyingKey, public: &[Fr], proof: &Proof) -> Result<(), VerifyError> {
    vk.check_structure()?;
    checked_pairing(vk, public, proof)
}

/// [`check_proof`], then the pairing check, with `B`'s subgroup check run beside the Miller
/// loops rather than before them. A proof that fails the points check is refused with that
/// error whatever the pairing says.
fn checked_pairing(vk: &VerifyingKey, public: &[Fr], proof: &Proof) -> Result<(), VerifyError> {
    let (points, pairing) = rayon::join(
        || check_proof(proof),
        || verify_unchecked(vk, public, proof),
    );
    points?;
    pairing
}

/// [`verify`] for a key with no phase-2 contribution: every check but the `gamma_g2 !=
/// delta_g2` refusal, see [`VerifyingKey::check_structure_allowing_unblinded`]. Benchmark
/// only. Passing it proves nothing, since anyone can forge a proof against such a key; it
/// still catches a proof the prover got wrong.
pub fn verify_allowing_unblinded(
    vk: &VerifyingKey,
    public: &[Fr],
    proof: &Proof,
) -> Result<(), VerifyError> {
    vk.check_structure_allowing_unblinded()?;
    checked_pairing(vk, public, proof)
}

/// The three proof elements, checked as [`verify`] documents.
pub fn check_proof(proof: &Proof) -> Result<(), VerifyError> {
    let bad = |what| Err(VerifyError::InvalidProof(what));
    if proof.a.infinity {
        return bad("A is the point at infinity");
    }
    if proof.b.infinity {
        return bad("B is the point at infinity");
    }
    if proof.c.infinity {
        return bad("C is the point at infinity");
    }
    // BN254 G1 has cofactor 1, so on the curve is in the subgroup there. The subgroup call
    // stays on A and C anyway so the check survives a change of curve.
    if !(proof.a.is_on_curve() && proof.a.is_in_correct_subgroup_assuming_on_curve()) {
        return bad("A is not a valid G1 point");
    }
    if !(proof.c.is_on_curve() && proof.c.is_in_correct_subgroup_assuming_on_curve()) {
        return bad("C is not a valid G1 point");
    }
    if !(proof.b.is_on_curve() && proof.b.is_in_correct_subgroup_assuming_on_curve()) {
        return bad("B is not a valid G2 point");
    }
    Ok(())
}

/// [`verify`] without validating the key or the proof: the bare pairing check.
///
/// For a caller that has already validated both, for instance through [`crate::json`]
/// and [`VerifyingKey::from_json`], and wants the microseconds back. Feeding it an
/// off-subgroup `B` returns `false` today rather than accepting, but that is a property
/// of BN254 and of arkworks' arithmetic, not something this signature promises. A key
/// with every pair degenerate makes it accept everything.
pub fn verify_unchecked(
    vk: &VerifyingKey,
    public: &[Fr],
    proof: &Proof,
) -> Result<(), VerifyError> {
    let l_bar = aggregate_public(vk, public)?;

    // One Miller loop per pair and a single final exponentiation over their product, so
    // the four checks cost one exponentiation rather than four. Not `multi_pairing`, which
    // unwraps the exponentiation: it is `None` when the loop's output is zero, which only
    // degenerate input reaches, and here that input is the caller's. The output group is
    // written additively, so "the product is 1" is "the sum is 0".
    let mlo = miller_loop([
        (-proof.a, proof.b),
        (vk.alpha_g1, vk.beta_g2),
        (l_bar.into_affine(), vk.gamma_g2),
        (proof.c, vk.delta_g2),
    ]);
    match Bn254::final_exponentiation(mlo) {
        Some(result) if result.is_zero() => Ok(()),
        _ => Err(VerifyError::PairingFailed),
    }
}

/// The Miller loop over `pairs`, one pair per thread. Its output is a product over the
/// pairs, so the per-pair loops multiplied together are the element `multi_miller_loop`
/// returns for all four: 0.34 ms on the M2 Max where the 4-pair loop is 0.75.
fn miller_loop(pairs: [(G1Affine, G2Affine); 4]) -> MillerLoopOutput<Bn254> {
    MillerLoopOutput(
        pairs
            .into_par_iter()
            .map(|(p, q)| Bn254::multi_miller_loop([p], [q]).0)
            .reduce(<Bn254 as Pairing>::TargetField::one, |x, y| x * y),
    )
}

/// Aggregate the public inputs: `L_bar = sum_i w_i * IC[i]`. An MSM over `n_public + 1`
/// points, small enough that one scalar multiplication per point is the right choice.
pub fn aggregate_public(vk: &VerifyingKey, public: &[Fr]) -> Result<G1Projective, VerifyError> {
    // `VerifyingKey`'s fields are public, so a key reaching here need not have come through
    // `from_json`: it can be built by hand or over FFI. An empty `ic` has no constant-wire
    // point, and `saturating_sub` would quietly turn that into "expects zero public inputs"
    // and then index `vk.ic[0]`.
    if vk.ic.is_empty() {
        return Err(VerifyError::EmptyVerifyingKey);
    }
    // IC[0] is the coefficient of the constant wire, which is fixed at 1 and therefore
    // never transmitted, so the caller supplies one fewer scalar than there are points.
    let want = vk.ic.len() - 1;
    if public.len() != want {
        return Err(VerifyError::PublicInputCount {
            got: public.len(),
            want,
        });
    }

    // GLV on raw limbs: arkworks' affine multiply is double-and-add over all 254 bits.
    let mut acc = Xyzz::<RawFq>::ZERO;
    if !vk.ic[0].infinity {
        let (x, y) = <g1::Config as RawCurve>::raw_xy(&vk.ic[0]);
        acc.madd(x, y);
    }
    for (point, scalar) in vk.ic[1..].iter().zip(public) {
        acc.add_assign(&glv::mul_affine(point, scalar));
    }
    Ok(to_projective::<g1::Config>(&acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::{G1Affine, G2Affine};

    /// `VerifyingKey`'s fields are public, so this crate cannot assume a key came through
    /// `snarkrs-formats`'s JSON reader. A hand-built key with an empty `ic` used to reach
    /// `vk.ic[0]` by way of `saturating_sub(1)`, and the release profile's `panic = "abort"`
    /// turned that bounds check into a SIGABRT. It must be an error instead.
    #[test]
    fn an_empty_ic_is_an_error_and_not_a_panic() {
        let vk = VerifyingKey {
            alpha_g1: G1Affine::identity(),
            beta_g2: G2Affine::identity(),
            gamma_g2: G2Affine::identity(),
            delta_g2: G2Affine::identity(),
            ic: Vec::new(),
        };
        // The empty public vector is the case that used to slip through: `want` computed as
        // 0, the length check passed, and the indexing killed the process.
        assert!(matches!(
            aggregate_public(&vk, &[]),
            Err(VerifyError::EmptyVerifyingKey)
        ));
        assert!(matches!(
            aggregate_public(&vk, &[Fr::from(1u64)]),
            Err(VerifyError::EmptyVerifyingKey)
        ));
    }

    /// The per-pair loops multiply to the same element as one 4-pair loop, a zero pair
    /// included, which `multi_miller_loop` drops in both.
    #[test]
    fn the_split_miller_loop_is_the_4_pair_loop() {
        use snarkrs_field::{AffineRepr, CurveGroup};
        let g1 = |k: u64| (G1Affine::generator() * Fr::from(k)).into_affine();
        let g2 = |k: u64| (G2Affine::generator() * Fr::from(k)).into_affine();
        for pairs in [
            [
                (g1(3), g2(5)),
                (g1(7), g2(11)),
                (g1(13), g2(17)),
                (g1(19), g2(23)),
            ],
            [
                (g1(3), g2(5)),
                (G1Affine::identity(), g2(11)),
                (g1(13), g2(17)),
                (g1(19), G2Affine::identity()),
            ],
        ] {
            let (a, b): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            assert_eq!(miller_loop(pairs).0, Bn254::multi_miller_loop(a, b).0);
        }
    }

    /// Why `verify_allowing_unblinded` is benchmark only. With `gamma_g2 == delta_g2`, the
    /// proof `A = alpha`, `B = beta`, `C = -L_bar` passes the pairing for any public input.
    /// `verify` refuses the key; the opt-in accepts the forgery, and still refuses a key or
    /// a proof that fails any other check.
    #[test]
    fn the_unblinded_opt_in_accepts_a_forgery_and_nothing_else_it_should_not() {
        use snarkrs_field::{AffineRepr, CurveGroup};
        let g1 = |k: u64| (G1Affine::generator() * Fr::from(k)).into_affine();
        let g2 = |k: u64| (G2Affine::generator() * Fr::from(k)).into_affine();
        let key = |alpha_g1| VerifyingKey {
            alpha_g1,
            beta_g2: g2(5),
            gamma_g2: g2(7),
            delta_g2: g2(7),
            ic: vec![g1(11), g1(13)],
        };
        let vk = key(g1(3));
        let public = [Fr::from(42u64)];
        let forged = Proof {
            a: vk.alpha_g1,
            b: vk.beta_g2,
            c: (-aggregate_public(&vk, &public).unwrap()).into_affine(),
        };

        assert!(matches!(
            verify(&vk, &public, &forged),
            Err(VerifyError::UnsafeKey(_))
        ));
        verify_allowing_unblinded(&vk, &public, &forged).unwrap();

        assert!(matches!(
            verify_allowing_unblinded(&key(G1Affine::identity()), &public, &forged),
            Err(VerifyError::UnsafeKey(_))
        ));
        let mut bad_proof = forged.clone();
        bad_proof.a = G1Affine::identity();
        assert!(matches!(
            verify_allowing_unblinded(&vk, &public, &bad_proof),
            Err(VerifyError::InvalidProof(_))
        ));
        assert!(matches!(
            verify_allowing_unblinded(&vk, &[Fr::from(43u64)], &forged),
            Err(VerifyError::PairingFailed)
        ));
    }
}
