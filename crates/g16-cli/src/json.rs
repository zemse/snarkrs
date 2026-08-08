//! snarkjs-compatible JSON for `proof.json` and `public.json`.
//!
//! This encoding is the interop contract the whole benchmark rests on: `snarkjs groth16
//! verify` is one of the three oracles, and it reads exactly these two files. Three
//! things have to be right or the proof is rejected at the very last step, after the
//! expensive part already ran:
//!
//! * every coordinate is a decimal string of the **normal** (non-Montgomery) integer;
//! * `pi_b`'s `Fq2` pairs are written `c0` first, `c1` second, which is ffjavascript's
//!   `toObject` order and the opposite of how the extension is usually printed;
//! * the third component is the projective/Jacobian `z`, written `"1"` and `["1","0"]`
//!   because snarkjs always normalises before writing.
//!
//! The reader is deliberately stricter than the writer: it rejects coordinates that are
//! not less than the modulus, points off the curve, and points outside the prime-order
//! subgroup. A verifier that reads untrusted JSON and calls `Affine::new` instead would
//! panic, or worse, accept a small-order point.

use anyhow::{anyhow, bail, Context, Result};
use g16_core::Proof;
use g16_field::*;
use num_bigint::BigUint;
use serde_json::Value;
use std::path::Path;

/// Normal-form decimal, the only representation snarkjs understands.
fn dec<F: PrimeField>(x: F) -> String {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le()).to_string()
}

fn parse_field<F: PrimeField>(v: &Value, what: &str) -> Result<F> {
    let s = v
        .as_str()
        .ok_or_else(|| anyhow!("{what}: expected a decimal string, got {v}"))?;
    let n: BigUint = s
        .parse()
        .with_context(|| format!("{what}: {s:?} is not a decimal integer"))?;
    // Reducing silently would turn a malformed proof into a different, valid-looking
    // one, which is precisely the class of bug a verifier must not have.
    if n >= BigUint::from_bytes_le(&F::MODULUS.to_bytes_le()) {
        bail!("{what}: {s} is not less than the field modulus");
    }
    Ok(F::from_le_bytes_mod_order(&n.to_bytes_le()))
}

fn elements<'a>(v: &'a Value, n: usize, what: &str) -> Result<&'a [Value]> {
    let a = v
        .as_array()
        .ok_or_else(|| anyhow!("{what}: expected an array of {n}"))?;
    if a.len() != n {
        bail!("{what}: expected {n} entries, got {}", a.len());
    }
    Ok(a)
}

fn g1_string(p: &G1Affine) -> String {
    if p.infinity {
        // ffjavascript's zero is [0, 1, 0]. A real proof never contains it, but writing
        // the affine (0, 0) here would be a point that is not on the curve.
        r#"["0","1","0"]"#.to_string()
    } else {
        format!("[\"{}\",\"{}\",\"1\"]", dec(p.x), dec(p.y))
    }
}

fn fq2_string(x: &Fq2) -> String {
    format!("[\"{}\",\"{}\"]", dec(x.c0), dec(x.c1))
}

fn g2_string(p: &G2Affine) -> String {
    if p.infinity {
        r#"[["0","0"],["1","0"],["0","0"]]"#.to_string()
    } else {
        format!("[{},{},[\"1\",\"0\"]]", fq2_string(&p.x), fq2_string(&p.y))
    }
}

/// `proof.json` in snarkjs' key order. Written by hand rather than through `serde_json`
/// because `serde_json`'s default map is sorted, which would emit `curve` first and make
/// a diff against snarkjs' own output unreadable. The bytes are still ordinary JSON.
pub fn proof_to_string(p: &Proof) -> String {
    format!(
        "{{\n \"pi_a\": {},\n \"pi_b\": {},\n \"pi_c\": {},\n \"protocol\": \"groth16\",\n \"curve\": \"bn128\"\n}}\n",
        g1_string(&p.a),
        g2_string(&p.b),
        g1_string(&p.c),
    )
}

/// `public.json`: a flat array of decimal strings, one per public signal.
pub fn public_to_string(public: &[Fr]) -> String {
    let items: Vec<String> = public.iter().map(|x| format!("\"{}\"", dec(*x))).collect();
    format!("[\n {}\n]\n", items.join(",\n "))
}

pub fn write_proof(path: &Path, p: &Proof) -> Result<()> {
    std::fs::write(path, proof_to_string(p))
        .with_context(|| format!("writing proof to {}", path.display()))
}

pub fn write_public(path: &Path, public: &[Fr]) -> Result<()> {
    std::fs::write(path, public_to_string(public))
        .with_context(|| format!("writing public signals to {}", path.display()))
}

fn checked_g1(x: Fq, y: Fq, what: &str) -> Result<G1Affine> {
    let p = G1Affine::new_unchecked(x, y);
    if !p.is_on_curve() {
        bail!("{what}: point is not on the curve");
    }
    if !p.is_in_correct_subgroup_assuming_on_curve() {
        bail!("{what}: point is not in the prime-order subgroup");
    }
    Ok(p)
}

fn checked_g2(x: Fq2, y: Fq2, what: &str) -> Result<G2Affine> {
    let p = G2Affine::new_unchecked(x, y);
    if !p.is_on_curve() {
        bail!("{what}: point is not on the curve");
    }
    // G2 over BN254 has a large cofactor, so this check is doing real work here: an
    // attacker-supplied off-subgroup B would otherwise reach the Miller loop.
    if !p.is_in_correct_subgroup_assuming_on_curve() {
        bail!("{what}: point is not in the prime-order subgroup");
    }
    Ok(p)
}

fn read_g1(v: &Value, what: &str) -> Result<G1Affine> {
    let a = elements(v, 3, what)?;
    let z: Fq = parse_field(&a[2], what)?;
    if z.is_zero() {
        return Ok(G1Affine::identity());
    }
    // ffjavascript's curve arithmetic is Jacobian, so the affine point is (x/z^2, y/z^3).
    // snarkjs always writes z = 1, where every convention agrees; this branch only
    // matters for hand-written input.
    let zi = z.inverse().expect("z is nonzero here");
    let zi2 = zi.square();
    checked_g1(
        parse_field::<Fq>(&a[0], what)? * zi2,
        parse_field::<Fq>(&a[1], what)? * zi2 * zi,
        what,
    )
}

fn read_fq2(v: &Value, what: &str) -> Result<Fq2> {
    let a = elements(v, 2, what)?;
    Ok(Fq2::new(
        parse_field(&a[0], what)?,
        parse_field(&a[1], what)?,
    ))
}

fn read_g2(v: &Value, what: &str) -> Result<G2Affine> {
    let a = elements(v, 3, what)?;
    let z = read_fq2(&a[2], what)?;
    if z.is_zero() {
        return Ok(G2Affine::identity());
    }
    let zi = z.inverse().expect("z is nonzero here");
    let zi2 = zi.square();
    checked_g2(
        read_fq2(&a[0], what)? * zi2,
        read_fq2(&a[1], what)? * zi2 * zi,
        what,
    )
}

pub fn proof_from_value(v: &Value) -> Result<Proof> {
    // The protocol/curve tags are cheap to check and turn "the pairing failed" into
    // "you handed me a plonk proof", which is a different bug entirely.
    match v.get("protocol").and_then(Value::as_str) {
        Some("groth16") | None => {}
        Some(other) => bail!("proof protocol is {other:?}, expected groth16"),
    }
    match v.get("curve").and_then(Value::as_str) {
        Some("bn128") | None => {}
        Some(other) => bail!("proof curve is {other:?}, expected bn128"),
    }
    let field = |k: &str| v.get(k).ok_or_else(|| anyhow!("proof has no {k}"));
    Ok(Proof {
        a: read_g1(field("pi_a")?, "pi_a")?,
        b: read_g2(field("pi_b")?, "pi_b")?,
        c: read_g1(field("pi_c")?, "pi_c")?,
    })
}

pub fn read_proof(path: &Path) -> Result<Proof> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading proof from {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    proof_from_value(&v).with_context(|| format!("in {}", path.display()))
}

pub fn read_public(path: &Path) -> Result<Vec<Fr>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading public signals from {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let a = v
        .as_array()
        .ok_or_else(|| anyhow!("{}: expected a flat JSON array", path.display()))?;
    a.iter()
        .enumerate()
        .map(|(i, x)| parse_field::<Fr>(x, &format!("public signal {i}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::variants;

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
