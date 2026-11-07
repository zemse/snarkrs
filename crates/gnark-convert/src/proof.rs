//! Proofs, public inputs and verifying keys between gnark's binary forms and snarkjs' JSON.
//!
//! snarkrs' `(A, B, C)` is snarkjs' `(pi_a, pi_b, pi_c)` and gnark's `(Ar, Bs, Krs)`, and
//! snarkjs' IC is gnark's `G1.K`, so every conversion here is an encoding. The readers are
//! [`snarkrs_gnark`]'s and [`snarkrs_groth16::json`]'s, which take both of gnark's forms and
//! check every point; this module adds the writers neither has.
//!
//! gnark's compressed form ([`compressed_g1`], [`compressed_g2`]) is the big-endian `X`,
//! for G2 `A1` before `A0`, with the top two bits of the first byte `10` when `Y` is the
//! lexicographically smaller root and `11` when it is the larger, and `01` and nothing
//! else for infinity. `VerifyingKey.WriteTo` and `Proof.WriteTo` write it, `WriteRawTo`
//! the raw form, and `ReadFrom` takes either.
//!
//! A snarkjs `verification_key.json` has no `vk_beta_1` or `vk_delta_1`, which gnark's key
//! carries for compatibility and its verifier never reads. [`gnark_vk_from_snarkjs`] fills
//! them from the zkey when there is one, and with the point at infinity when there is not:
//! gnark's `ReadFrom` and `groth16.Verify` take such a key, and
//! [`snarkrs_gnark::GnarkVerifyingKey::read`] refuses it, as it refuses any header point at
//! infinity.

use crate::ConvertError;
use snarkrs_field::*;
use snarkrs_formats::VerifyingKey;
use snarkrs_gnark::proof::PROOF_BYTES;
use snarkrs_gnark::{raw_g1, raw_g2, GnarkVerifyingKey, G1_COMPRESSED, G2_COMPRESSED};
use snarkrs_groth16::Proof;

/// Bytes of a `WriteTo` proof without commitments: three compressed points, a zero count
/// and a compressed-infinity `CommitmentPok`.
pub const COMPRESSED_PROOF_BYTES: usize = 2 * G1_COMPRESSED + G2_COMPRESSED + 4 + G1_COMPRESSED;

const COMPRESSED_INFINITY: u8 = 0b01 << 6;
const COMPRESSED_SMALLEST: u8 = 0b10 << 6;
const COMPRESSED_LARGEST: u8 = 0b11 << 6;

fn fq_be(x: &Fq) -> [u8; 32] {
    x.into_bigint()
        .to_bytes_be()
        .try_into()
        .expect("a BN254 base-field element is 32 bytes")
}

/// gnark's `LexicographicallyLargest`: above `(q - 1) / 2` as an integer.
fn fq_largest(y: &Fq) -> bool {
    y.into_bigint() > Fq::MODULUS_MINUS_ONE_DIV_TWO
}

/// gnark's `E2.LexicographicallyLargest`: decided by `A1`, or by `A0` when `A1` is zero.
fn fq2_largest(y: &Fq2) -> bool {
    if y.c1.is_zero() {
        fq_largest(&y.c0)
    } else {
        fq_largest(&y.c1)
    }
}

/// `p` as gnark's `G1Affine.Bytes`.
pub fn compressed_g1(p: &G1Affine) -> [u8; G1_COMPRESSED] {
    let mut out = [0u8; G1_COMPRESSED];
    if p.infinity {
        out[0] = COMPRESSED_INFINITY;
        return out;
    }
    out.copy_from_slice(&fq_be(&p.x));
    out[0] |= if fq_largest(&p.y) {
        COMPRESSED_LARGEST
    } else {
        COMPRESSED_SMALLEST
    };
    out
}

/// `p` as gnark's `G2Affine.Bytes`: `X.A1` then `X.A0`.
pub fn compressed_g2(p: &G2Affine) -> [u8; G2_COMPRESSED] {
    let mut out = [0u8; G2_COMPRESSED];
    if p.infinity {
        out[0] = COMPRESSED_INFINITY;
        return out;
    }
    out[..32].copy_from_slice(&fq_be(&p.x.c1));
    out[32..].copy_from_slice(&fq_be(&p.x.c0));
    out[0] |= if fq2_largest(&p.y) {
        COMPRESSED_LARGEST
    } else {
        COMPRESSED_SMALLEST
    };
    out
}

/// One encoder point of each group, raw or compressed.
fn put_g1(out: &mut Vec<u8>, p: &G1Affine, compressed: bool) {
    if compressed {
        out.extend_from_slice(&compressed_g1(p));
    } else {
        out.extend_from_slice(&raw_g1(p));
    }
}

fn put_g2(out: &mut Vec<u8>, p: &G2Affine, compressed: bool) {
    if compressed {
        out.extend_from_slice(&compressed_g2(p));
    } else {
        out.extend_from_slice(&raw_g2(p));
    }
}

/// `proof` as gnark's `Proof.WriteTo` (`compressed`) or `WriteRawTo` writes it.
pub fn gnark_proof(proof: &Proof, compressed: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(if compressed {
        COMPRESSED_PROOF_BYTES
    } else {
        PROOF_BYTES
    });
    put_g1(&mut out, &proof.a, compressed);
    put_g2(&mut out, &proof.b, compressed);
    put_g1(&mut out, &proof.c, compressed);
    out.extend_from_slice(&0u32.to_be_bytes());
    put_g1(&mut out, &G1Affine::identity(), compressed);
    out
}

/// `vk` as gnark's `VerifyingKey.WriteTo` (`compressed`) or `WriteRawTo` writes it, with no
/// commitments: the six points, `G1.K` as a `u32` BE count and its points, then an empty
/// `PublicAndCommitmentCommitted` and a zero commitment-key count, a `u32` BE each.
pub fn gnark_vk(vk: &GnarkVerifyingKey, compressed: bool) -> Vec<u8> {
    let mut out = Vec::new();
    put_g1(&mut out, &vk.alpha_g1, compressed);
    put_g1(&mut out, &vk.beta_g1, compressed);
    put_g2(&mut out, &vk.beta_g2, compressed);
    put_g2(&mut out, &vk.gamma_g2, compressed);
    put_g1(&mut out, &vk.delta_g1, compressed);
    put_g2(&mut out, &vk.delta_g2, compressed);
    out.extend_from_slice(&(vk.k.len() as u32).to_be_bytes());
    for p in &vk.k {
        put_g1(&mut out, p, compressed);
    }
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out
}

/// A snarkjs verifying key as gnark's, `beta_g1` and `delta_g1` from the zkey's header when
/// the caller has it and the point at infinity when not. See the module docs.
pub fn gnark_vk_from_snarkjs(
    vk: &VerifyingKey,
    beta_g1: Option<G1Affine>,
    delta_g1: Option<G1Affine>,
) -> GnarkVerifyingKey {
    GnarkVerifyingKey {
        alpha_g1: vk.alpha_g1,
        beta_g1: beta_g1.unwrap_or_else(G1Affine::identity),
        beta_g2: vk.beta_g2,
        gamma_g2: vk.gamma_g2,
        delta_g1: delta_g1.unwrap_or_else(G1Affine::identity),
        delta_g2: vk.delta_g2,
        k: vk.ic.clone(),
    }
}

/// gnark's verifying key as snarkjs' `verification_key.json`, byte for byte what `snarkjs
/// zkey export verificationkey` writes for a zkey holding the same key.
pub fn snarkjs_vk(vk: &GnarkVerifyingKey) -> String {
    snarkrs_ceremony::vkey::verification_key_text(&vk.to_verifying_key())
}

/// Public inputs as gnark's `witness.MarshalBinary` writes a public witness: `u32` BE
/// public and secret counts (the secret one zero), then the vector as a `u32` BE length and
/// 32-byte big-endian canonical scalars. The constant wire is not in it.
pub fn gnark_public_witness(public: &[Fr]) -> Vec<u8> {
    let n = public.len() as u32;
    let mut out = Vec::with_capacity(12 + 32 * public.len());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&snarkrs_gnark::proof::write_public(public));
    out
}

/// [`gnark_public_witness`]'s layout back, refused unless it is a public witness of exactly
/// the length it claims, every scalar below the modulus.
pub fn read_gnark_public_witness(data: &[u8]) -> Result<Vec<Fr>, ConvertError> {
    let bad = |reason: String| crate::mismatch("public witness", reason);
    if data.len() < 12 {
        return Err(bad(format!(
            "{} bytes is shorter than the header",
            data.len()
        )));
    }
    let word = |i: usize| u32::from_be_bytes(data[4 * i..4 * i + 4].try_into().expect("4 bytes"));
    let (n_public, n_secret, len) = (word(0), word(1), word(2));
    if n_secret != 0 || len != n_public {
        return Err(bad(format!(
            "{n_public} public and {n_secret} secret values in a vector of {len}; a public \
             witness has no secret ones"
        )));
    }
    if (data.len() - 12) as u64 != u64::from(len) * 32 {
        return Err(bad(format!(
            "{len} values do not fill the {} bytes after the header",
            data.len() - 12
        )));
    }
    Ok(snarkrs_gnark::proof::read_public(&data[12..])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::AdditiveGroup;
    use snarkrs_gnark::{encoded_g1, encoded_g2};

    /// Every flag a compressed point can carry, decoded back by snarkrs-gnark's reader,
    /// which the toy fixture holds to gnark's own bytes.
    #[test]
    fn compressed_points_decode_to_themselves() {
        let mut g1 = G1Projective::from(G1Affine::generator());
        let mut g2 = G2Projective::from(G2Affine::generator());
        let (mut large, mut small) = (0, 0);
        for _ in 0..64 {
            let (p, q) = (g1.into_affine(), g2.into_affine());
            assert_eq!(encoded_g1(&compressed_g1(&p)).unwrap(), p);
            assert_eq!(encoded_g1(&compressed_g1(&-p)).unwrap(), -p);
            assert_eq!(encoded_g2(&compressed_g2(&q)).unwrap(), q);
            assert_eq!(encoded_g2(&compressed_g2(&-q)).unwrap(), -q);
            if fq_largest(&p.y) {
                large += 1;
            } else {
                small += 1;
            }
            g1 = g1.double() + G1Affine::generator();
            g2 = g2.double() + G2Affine::generator();
        }
        assert!(large > 0 && small > 0);
        assert_eq!(
            encoded_g1(&compressed_g1(&G1Affine::identity())).unwrap(),
            G1Affine::identity()
        );
        assert_eq!(
            encoded_g2(&compressed_g2(&G2Affine::identity())).unwrap(),
            G2Affine::identity()
        );
    }

    #[test]
    fn public_witness_round_trips_and_refuses_a_bad_header() {
        let public = [Fr::from(5u64), -Fr::from(1u64)];
        let bytes = gnark_public_witness(&public);
        assert_eq!(read_gnark_public_witness(&bytes).unwrap(), public);
        assert!(read_gnark_public_witness(&bytes[..bytes.len() - 1]).is_err());
        let mut secret = bytes.clone();
        secret[7] = 1;
        assert!(read_gnark_public_witness(&secret).is_err());
    }
}
