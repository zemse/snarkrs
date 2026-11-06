//! gnark's proof and public-input files, the interchange with gnark's own verifier.
//!
//! A proof is `Proof.WriteRawTo` (backend/groth16/bn254/marshal.go): raw encoder points
//! `Ar`, `Bs`, `Krs`, then `Commitments` as a `u32` BE count and points, then
//! `CommitmentPok`. Without commitments that is 324 bytes ending in a zero count and an
//! all-zero `CommitmentPok`, which is the point at infinity. snarkrs' `(A, B, C)` is gnark's
//! `(Ar, Bs, Krs)`. Public inputs are 32 bytes each, big-endian and canonical,
//! `fr.Element.Bytes()`, the constant wire left out.

use crate::{raw_g1, raw_g2, GnarkError, Reader, G1_UNCOMPRESSED, G2_UNCOMPRESSED};
use snarkrs_field::*;
use snarkrs_groth16::Proof;

/// Bytes of a `WriteRawTo` proof without commitments.
pub const PROOF_BYTES: usize = 2 * G1_UNCOMPRESSED + G2_UNCOMPRESSED + 4 + G1_UNCOMPRESSED;

/// `proof` as gnark's `Proof.WriteRawTo` writes it, which gnark's `ReadFrom` reads.
pub fn write_proof(proof: &Proof) -> Vec<u8> {
    let mut out = Vec::with_capacity(PROOF_BYTES);
    out.extend_from_slice(&raw_g1(&proof.a));
    out.extend_from_slice(&raw_g2(&proof.b));
    out.extend_from_slice(&raw_g1(&proof.c));
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&raw_g1(&G1Affine::identity()));
    out
}

/// A gnark proof, raw or compressed. Refused if it carries commitments. The points are
/// checked on the curve, and B in the subgroup; `snarkrs_groth16::verify` checks the rest.
pub fn read_proof(data: &[u8]) -> Result<Proof, GnarkError> {
    let mut r = Reader::new(data, "proof");
    let a = r.g1("Ar")?;
    let b = r.g2("Bs")?;
    let c = r.g1("Krs")?;
    let commitments = r.u32_be()?;
    if commitments != 0 {
        return Err(GnarkError::Commitments {
            part: "proof",
            count: commitments.into(),
        });
    }
    if !r.g1("CommitmentPok")?.infinity {
        return Err(r.bad("CommitmentPok is not infinity on a proof with no commitments".into()));
    }
    r.finish()?;
    Ok(Proof { a, b, c })
}

/// Public inputs as 32-byte big-endian canonical scalars, the layout the harness in
/// `test-support/gnark` reads.
pub fn write_public(public: &[Fr]) -> Vec<u8> {
    public
        .iter()
        .flat_map(|x| x.into_bigint().to_bytes_be())
        .collect()
}

/// [`write_public`]'s layout back, each scalar refused at or above the modulus.
pub fn read_public(data: &[u8]) -> Result<Vec<Fr>, GnarkError> {
    if !data.len().is_multiple_of(32) {
        return Err(GnarkError::Malformed {
            part: "public inputs",
            reason: format!("{} bytes is not a whole number of scalars", data.len()),
        });
    }
    data.chunks(32)
        .enumerate()
        .map(|(i, b)| {
            crate::fr_be(b).ok_or_else(|| GnarkError::Malformed {
                part: "public inputs",
                reason: format!("input {i} is not below the scalar modulus"),
            })
        })
        .collect()
}
