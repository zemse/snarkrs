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
//!
//! # Why this is in `g16-core` and not next to the CLI
//!
//! It lived in `g16-cli` until U13, which is fine while a proof only ever leaves this
//! workspace through a file. The browser prover has no filesystem and hands its proof to
//! `snarkjs.groth16.verify` across the JS boundary as a string, so it needs exactly these
//! three rules and nothing else in `g16-cli`. Two copies of a `c0`/`c1` ordering is how one
//! of them ends up wrong, and the copy that would have been wrong is the one no `cargo test`
//! on this machine ever exercises. `g16-cli`'s `json` module is now the path-taking half:
//! `read_proof`, `read_public`, `write_proof`, `write_public`, and the tests.
//!
//! `anyhow` does not appear here. A library that returns `anyhow::Error` forces the choice on
//! everything downstream, and `g16-wgpu`'s wasm entry point turns errors into `JsError`.

use crate::Proof;
use g16_field::*;
use num_bigint::BigUint;
use serde_json::Value;

/// Anything wrong with a `proof.json` or a `public.json`.
///
/// One variant carrying a sentence, because every caller either prints it (the CLI) or
/// converts it to a `JsError` (the browser), and nobody branches on the reason.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct JsonError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, JsonError> {
    Err(JsonError(msg.into()))
}

/// Normal-form decimal, the only representation snarkjs understands.
pub fn dec<F: PrimeField>(x: F) -> String {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le()).to_string()
}

pub fn parse_field<F: PrimeField>(v: &Value, what: &str) -> Result<F, JsonError> {
    let Some(s) = v.as_str() else {
        return err(format!("{what}: expected a decimal string, got {v}"));
    };
    let Ok(n) = s.parse::<BigUint>() else {
        return err(format!("{what}: {s:?} is not a decimal integer"));
    };
    // Reducing silently would turn a malformed proof into a different, valid-looking
    // one, which is precisely the class of bug a verifier must not have.
    if n >= BigUint::from_bytes_le(&F::MODULUS.to_bytes_le()) {
        return err(format!("{what}: {s} is not less than the field modulus"));
    }
    Ok(F::from_le_bytes_mod_order(&n.to_bytes_le()))
}

fn elements<'a>(v: &'a Value, n: usize, what: &str) -> Result<&'a [Value], JsonError> {
    let Some(a) = v.as_array() else {
        return err(format!("{what}: expected an array of {n}"));
    };
    if a.len() != n {
        return err(format!("{what}: expected {n} entries, got {}", a.len()));
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

fn checked_g1(x: Fq, y: Fq, what: &str) -> Result<G1Affine, JsonError> {
    let p = G1Affine::new_unchecked(x, y);
    if !p.is_on_curve() {
        return err(format!("{what}: point is not on the curve"));
    }
    if !p.is_in_correct_subgroup_assuming_on_curve() {
        return err(format!("{what}: point is not in the prime-order subgroup"));
    }
    Ok(p)
}

fn checked_g2(x: Fq2, y: Fq2, what: &str) -> Result<G2Affine, JsonError> {
    let p = G2Affine::new_unchecked(x, y);
    if !p.is_on_curve() {
        return err(format!("{what}: point is not on the curve"));
    }
    // G2 over BN254 has a large cofactor, so this check is doing real work here: an
    // attacker-supplied off-subgroup B would otherwise reach the Miller loop.
    if !p.is_in_correct_subgroup_assuming_on_curve() {
        return err(format!("{what}: point is not in the prime-order subgroup"));
    }
    Ok(p)
}

pub fn read_g1(v: &Value, what: &str) -> Result<G1Affine, JsonError> {
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

fn read_fq2(v: &Value, what: &str) -> Result<Fq2, JsonError> {
    let a = elements(v, 2, what)?;
    Ok(Fq2::new(
        parse_field(&a[0], what)?,
        parse_field(&a[1], what)?,
    ))
}

pub fn read_g2(v: &Value, what: &str) -> Result<G2Affine, JsonError> {
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

pub fn proof_from_value(v: &Value) -> Result<Proof, JsonError> {
    // The protocol/curve tags are cheap to check and turn "the pairing failed" into
    // "you handed me a plonk proof", which is a different bug entirely.
    match v.get("protocol").and_then(Value::as_str) {
        Some("groth16") | None => {}
        Some(other) => return err(format!("proof protocol is {other:?}, expected groth16")),
    }
    match v.get("curve").and_then(Value::as_str) {
        Some("bn128") | None => {}
        Some(other) => return err(format!("proof curve is {other:?}, expected bn128")),
    }
    let field = |k: &str| {
        v.get(k)
            .ok_or_else(|| JsonError(format!("proof has no {k}")))
    };
    Ok(Proof {
        a: read_g1(field("pi_a")?, "pi_a")?,
        b: read_g2(field("pi_b")?, "pi_b")?,
        c: read_g1(field("pi_c")?, "pi_c")?,
    })
}

/// `public.json` as a flat array of decimal strings.
pub fn public_from_value(v: &Value) -> Result<Vec<Fr>, JsonError> {
    let Some(a) = v.as_array() else {
        return err("expected a flat JSON array of public signals");
    };
    a.iter()
        .enumerate()
        .map(|(i, x)| parse_field::<Fr>(x, &format!("public signal {i}")))
        .collect()
}

/// Parse a whole `proof.json` document.
pub fn proof_from_str(text: &str) -> Result<Proof, JsonError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| JsonError(format!("proof is not valid JSON: {e}")))?;
    proof_from_value(&v)
}

/// Parse a whole `public.json` document.
pub fn public_from_str(text: &str) -> Result<Vec<Fr>, JsonError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| JsonError(format!("public signals are not valid JSON: {e}")))?;
    public_from_value(&v)
}
