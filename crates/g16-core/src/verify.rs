//! Groth16 verification.
//!
//! `e(A, B) = e(alpha, beta) * e(L_bar, gamma) * e(C, delta)`, rearranged into a single
//! 4-pair multi-Miller loop with `A` negated so the check is `product == 1` and only one
//! final exponentiation is needed.

use crate::Proof;
use g16_field::*;
use g16_zkey::VerifyingKey;

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("expected {want} public inputs, got {got}")]
    PublicInputCount { got: usize, want: usize },
    #[error("verifying key has an empty IC; it must carry at least the constant-wire point")]
    EmptyVerifyingKey,
    #[error("pairing check failed")]
    PairingFailed,
}

/// Check a proof against a verifying key and public inputs.
///
/// # What this does not check
///
/// The points in `proof` are taken as given. This function does no on-curve, subgroup or
/// canonical-encoding validation, because `Proof`'s fields are public and a `Proof` that
/// reached here through this crate's own deserializer has already been validated by it:
/// `g16-cli`'s `json.rs` checks on-curve and prime-order-subgroup membership for `pi_b`,
/// which is the one that matters (BN254 G1 has cofactor 1, so on-curve implies in-subgroup
/// there; G2 does not).
///
/// A caller that constructs a `Proof` by hand, over FFI, or with `ark-serialize` and
/// `Validate::No` gets no such guarantee and must validate before calling. Feeding an
/// off-subgroup `B` here returns `false` today rather than accepting, but that is a
/// property of BN254 and of arkworks' arithmetic, not something this signature promises:
/// the Groth16 soundness argument stops applying the moment an off-subgroup element
/// reaches the Miller loop.
///
/// A proof that verifies is malleable. See [`Proof`] for why that matters to the caller.
pub fn verify(vk: &VerifyingKey, public: &[Fr], proof: &Proof) -> Result<(), VerifyError> {
    let l_bar = aggregate_public(vk, public)?;

    // `multi_pairing` runs one Miller loop per pair and a single final exponentiation
    // over their product, so the four checks cost one exponentiation rather than four.
    // The output group is written additively, so "the product is 1" is "the sum is 0".
    let result = Bn254::multi_pairing(
        [-proof.a, vk.alpha_g1, l_bar.into_affine(), proof.c],
        [proof.b, vk.beta_g2, vk.gamma_g2, vk.delta_g2],
    );

    if result.is_zero() {
        Ok(())
    } else {
        Err(VerifyError::PairingFailed)
    }
}

/// Aggregate the public inputs: `L_bar = sum_i w_i * IC[i]`. An MSM over `n_public + 1`
/// points, small enough that a plain double-and-add loop is the right choice.
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

    let mut acc = vk.ic[0].into_group();
    for (point, scalar) in vk.ic[1..].iter().zip(public) {
        acc += *point * *scalar;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g16_field::{G1Affine, G2Affine};

    /// `VerifyingKey`'s fields are public, so this crate cannot assume a key came through
    /// `g16-zkey`'s JSON reader. A hand-built key with an empty `ic` used to reach
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
}
