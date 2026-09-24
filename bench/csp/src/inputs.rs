//! The benchmark inputs, reproduced from the upstream generator's definition.
//!
//! Every hash target draws its message from `StdRng::seed_from_u64(input_size)` and hashes
//! it with the matching host-side hash, so a run on any machine sees the same bytes. The
//! digest is an input to the circuit, not only an output: these circuits take the message
//! and the expected digest and constrain their own output to equal it. ECDSA is the one
//! target that signs rather than hashes, and its key comes from a seed of its own.
//!
//! Nothing here affects proving time - a Groth16 prover does the same work whatever the
//! witness holds - but it does decide whether the witness satisfies the R1CS at all, and
//! a proof over a witness that does not is the one benchmark result that means nothing.

use rand::{RngCore, SeedableRng};
use sha2::{Digest, Sha256};
use sha3::Keccak256;

fn message(input_size: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; input_size];
    rand::rngs::StdRng::seed_from_u64(input_size as u64).fill_bytes(&mut bytes);
    bytes
}

/// `(message, sha256(message))`, both as byte vectors.
pub fn sha256(input_size: usize) -> (Vec<u8>, Vec<u8>) {
    let m = message(input_size);
    let d = Sha256::digest(&m).to_vec();
    (m, d)
}

/// `(message, keccak256(message))`, both as byte vectors.
pub fn keccak(input_size: usize) -> (Vec<u8>, Vec<u8>) {
    let m = message(input_size);
    let d = Keccak256::digest(&m).to_vec();
    (m, d)
}

/// `input_size` field elements as decimal strings.
///
/// The top byte is masked to 5 bits before the little-endian read, which is what keeps
/// every draw below the BN254 scalar modulus without a rejection loop. A rejection loop
/// would consume a variable number of RNG words and the sequence would stop being
/// reproducible from the seed alone.
pub fn poseidon(input_size: usize) -> Vec<String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(input_size as u64);
    (0..input_size)
        .map(|_| {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            bytes[31] &= 0x1f;
            num_bigint::BigUint::from_bytes_le(&bytes).to_string()
        })
        .collect()
}

/// The public half of one secp256k1 signature, every value 32 big-endian bytes. Nothing
/// else reaches the circuit: it witnesses `sinv`, `R` and the lattice hint itself.
pub struct Signature {
    pub r: Vec<u8>,
    pub s: Vec<u8>,
    pub msghash: Vec<u8>,
    pub pubkey_x: Vec<u8>,
    pub pubkey_y: Vec<u8>,
}

/// One signature over the sha256 target's 128 byte digest, under a key of its own seed.
///
/// `normalize_s` is not something the circuit asks for - it accepts either representative
/// of s - but the upstream input carries it, and s and n - s send the verifier down
/// different witnesses.
pub fn ecdsa() -> Signature {
    use k256::ecdsa::signature::hazmat::PrehashSigner;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xecd5a);
    let key = k256::ecdsa::SigningKey::random(&mut rng);
    let point = key.verifying_key().to_encoded_point(false);
    let coordinate = |b: Option<&k256::FieldBytes>| b.expect("uncompressed point").to_vec();

    let (_, msghash) = sha256(128);
    let sig: k256::ecdsa::Signature = key.sign_prehash(&msghash).expect("a 32 byte prehash");
    let sig = sig.normalize_s().unwrap_or(sig);
    let bytes = sig.to_bytes();
    let (r, s) = bytes.split_at(32);

    Signature {
        r: r.to_vec(),
        s: s.to_vec(),
        msghash,
        pubkey_x: coordinate(point.x()),
        pubkey_y: coordinate(point.y()),
    }
}

/// 32 big-endian bytes as the four 64-bit limbs the bigint templates read, least
/// significant first, as decimal strings.
fn limbs(be: &[u8]) -> Vec<String> {
    be.rchunks(8)
        .map(|w| {
            let w: [u8; 8] = w.try_into().expect("a multiple of eight bytes");
            u64::from_be_bytes(w).to_string()
        })
        .collect()
}

/// The circom input object for one variant, as the JSON its witness generator expects.
pub fn json(target: super::Target, input_size: usize) -> String {
    fn bytes_as_decimals(b: &[u8]) -> Vec<String> {
        b.iter().map(|n| n.to_string()).collect()
    }
    let value = match target {
        super::Target::Sha256 => {
            let (m, d) = sha256(input_size);
            serde_json::json!({ "in": bytes_as_decimals(&m), "hash": bytes_as_decimals(&d) })
        }
        super::Target::Keccak => {
            let (m, d) = keccak(input_size);
            serde_json::json!({ "in": bytes_as_decimals(&m), "hash": bytes_as_decimals(&d) })
        }
        super::Target::Poseidon => serde_json::json!({ "inputs": poseidon(input_size) }),
        super::Target::Ecdsa => {
            let sig = ecdsa();
            serde_json::json!({
                "r": limbs(&sig.r),
                "s": limbs(&sig.s),
                "msghash": limbs(&sig.msghash),
                "pubkey": [limbs(&sig.pubkey_x), limbs(&sig.pubkey_y)],
            })
        }
    };
    serde_json::to_string(&value).expect("serde_json::Value always serialises")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed is the input size, so two sizes never share a prefix and the same size
    /// always replays. Both halves matter: the first is why `sha256_256` is not
    /// `sha256_128` twice, the second is why a rerun compares to the run before it.
    #[test]
    fn messages_replay_and_differ_by_size() {
        assert_eq!(sha256(128).0, sha256(128).0);
        assert_ne!(sha256(128).0, sha256(256).0[..128].to_vec());
    }

    #[test]
    fn digests_are_of_the_message() {
        let (m, d) = sha256(512);
        assert_eq!(d, Sha256::digest(&m).to_vec());
        let (m, d) = keccak(512);
        assert_eq!(d, Keccak256::digest(&m).to_vec());
    }

    /// The circuit witnesses `sinv` and `R` itself and checks them, so an input that does
    /// not satisfy the ECDSA relation fails witness generation rather than proving
    /// something false. That failure does not say which half is wrong; this does.
    #[test]
    fn the_signature_verifies_under_its_own_key() {
        use k256::ecdsa::signature::hazmat::PrehashVerifier;
        use k256::elliptic_curve::sec1::FromEncodedPoint;

        let signature = ecdsa();
        let point = k256::EncodedPoint::from_affine_coordinates(
            k256::FieldBytes::from_slice(&signature.pubkey_x),
            k256::FieldBytes::from_slice(&signature.pubkey_y),
            false,
        );
        let q = k256::AffinePoint::from_encoded_point(&point).unwrap();
        let key = k256::ecdsa::VerifyingKey::from_affine(q).unwrap();

        let mut bytes = signature.r.clone();
        bytes.extend_from_slice(&signature.s);
        let sig = k256::ecdsa::Signature::from_slice(&bytes).unwrap();
        assert!(key.verify_prehash(&signature.msghash, &sig).is_ok());
        assert_eq!(sig.to_bytes(), sig.normalize_s().unwrap_or(sig).to_bytes());
    }

    /// 2^64 + 5 is `[5, 1, 0, 0]`, not `[0, 0, 1, 5]`: the bigint templates read the low
    /// limb first, and a reversed input is a different number that still range checks.
    #[test]
    fn limbs_are_least_significant_first() {
        let mut be = [0u8; 32];
        be[31] = 5;
        be[23] = 1;
        assert_eq!(limbs(&be), ["5", "1", "0", "0"]);
    }

    /// Below the modulus by construction: 5 bits masked off the top byte caps a draw at
    /// 2^253 - 1, and the BN254 scalar field is a little over 2^253.
    #[test]
    fn poseidon_draws_are_in_field() {
        let modulus = num_bigint::BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap();
        let drawn = poseidon(16);
        assert_eq!(drawn.len(), 16);
        for s in drawn {
            assert!(s.parse::<num_bigint::BigUint>().unwrap() < modulus);
        }
    }
}
