//! The path-taking half of snarkjs' `proof.json` and `public.json` interop.
//!
//! The encoding itself lives in [`g16_core::json`], which is where the browser prover can
//! reach it: a wasm build has no filesystem and still has to hand `snarkjs.groth16.verify`
//! exactly the same bytes this CLI writes. Everything about `c0`/`c1` ordering, normal-form
//! decimals and the projective third component is documented there, in one copy.

use anyhow::{Context, Result};
use g16_field::*;
use std::path::Path;

pub use g16_core::json::{
    dec, parse_field, proof_from_str, proof_from_value, proof_to_string, public_from_str,
    public_from_value, public_to_string, read_g1, read_g2, JsonError,
};
use g16_core::Proof;

pub fn write_proof(path: &Path, p: &Proof) -> Result<()> {
    std::fs::write(path, proof_to_string(p))
        .with_context(|| format!("writing proof to {}", path.display()))
}

pub fn write_public(path: &Path, public: &[Fr]) -> Result<()> {
    std::fs::write(path, public_to_string(public))
        .with_context(|| format!("writing public signals to {}", path.display()))
}

pub fn read_proof(path: &Path) -> Result<Proof> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading proof from {}", path.display()))?;
    proof_from_str(&text).with_context(|| format!("in {}", path.display()))
}

pub fn read_public(path: &Path) -> Result<Vec<Fr>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading public signals from {}", path.display()))?;
    public_from_str(&text).with_context(|| format!("in {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::variants;
    use num_bigint::BigUint;
    use serde_json::Value;

    fn value(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    /// The one test that catches a swapped `c0`/`c1` without running a pairing: read
    /// snarkjs' own proof, write it back, and require the numbers to come out identical.
    /// If the reader and the writer were both wrong in the same direction this would
    /// still pass, which is why `snarkjs_accepts_our_proof` in tests/ exists as well.
    #[test]
    fn reencoding_snarkjs_own_proof_is_a_fixed_point() {
        let found = variants(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts"));
        if found.is_empty() {
            eprintln!("SKIPPED: no artifacts under bench/artifacts");
            return;
        }
        for v in &found {
            let path = v.dir.join("proof.json");
            if !path.is_file() {
                continue;
            }
            eprintln!("reencoding_snarkjs_own_proof_is_a_fixed_point: {}", v.name);
            let original: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let ours = value(&proof_to_string(&proof_from_value(&original).unwrap()));
            for k in ["pi_a", "pi_b", "pi_c", "protocol", "curve"] {
                assert_eq!(ours[k], original[k], "{} field {k}", v.name);
            }

            let pub_path = v.dir.join("public.json");
            let public = read_public(&pub_path).unwrap();
            assert_eq!(
                value(&public_to_string(&public)),
                value(&std::fs::read_to_string(&pub_path).unwrap()),
                "{} public.json",
                v.name
            );
        }
    }

    #[test]
    fn rejects_a_coordinate_at_or_above_the_modulus() {
        let p = BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le());
        let v = Value::String(p.to_string());
        assert!(parse_field::<Fq>(&v, "x").is_err());
        assert!(parse_field::<Fq>(&Value::String((&p - 1u32).to_string()), "x").is_ok());
        assert!(parse_field::<Fq>(&Value::String("-1".into()), "x").is_err());
        assert!(parse_field::<Fq>(&Value::Number(1.into()), "x").is_err());
    }

    #[test]
    fn rejects_a_point_off_the_curve() {
        // y^2 = x^3 + 3, so (1, 2) is on the curve (it is the G1 generator) and (1, 3)
        // is not. Both must be handled, otherwise this test could pass because the
        // reader rejects everything.
        assert!(read_g1(&value(r#"["1","3","1"]"#), "pi_a").is_err());
        assert_eq!(
            read_g1(&value(r#"["1","2","1"]"#), "pi_a").unwrap(),
            G1Affine::generator()
        );
    }

    #[test]
    fn rejects_the_wrong_shape() {
        assert!(read_g1(&value(r#"["1","2"]"#), "pi_a").is_err());
        assert!(proof_from_value(&value(r#"{"protocol":"plonk"}"#)).is_err());
        assert!(proof_from_value(&value(r#"{"pi_a":["0","1","0"]}"#)).is_err());
    }

    /// The identity round-trips through the zero encoding rather than through (0, 0),
    /// which is not a curve point and would fail the reader's own check.
    #[test]
    fn identity_round_trips() {
        let p = Proof {
            a: G1Affine::identity(),
            b: G2Affine::identity(),
            c: G1Affine::identity(),
        };
        let back = proof_from_value(&value(&proof_to_string(&p))).unwrap();
        assert!(back.a.infinity && back.b.infinity && back.c.infinity);
    }

    #[test]
    fn public_signals_round_trip() {
        let xs = vec![Fr::zero(), Fr::one(), Fr::from(u64::MAX), -Fr::one()];
        let path = std::env::temp_dir().join(format!("g16-public-{}.json", std::process::id()));
        write_public(&path, &xs).unwrap();
        assert_eq!(read_public(&path).unwrap(), xs);
        std::fs::remove_file(&path).ok();
    }
}
