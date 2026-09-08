//! `zkey export verificationkey`: the four verifier points, the precomputed pairing, and
//! IC, as snarkjs' `verification_key.json`.
//!
//! `zkey_export_verificationkey.js:55-89` builds the object and `cli.js:589-601` writes it
//! with one space of indentation per level. Key order is JS object insertion order, so
//! `IC` is last because `vKey.IC` is assigned after the literal. `vk_beta_1` and
//! `vk_delta_1` are **not** exported: only the four points a verifier needs, plus IC.
//!
//! Three encoding details, none of them the same as anything on disk:
//!
//! * Every coordinate is a **decimal string** in normal, non-Montgomery form, and every
//!   point is in projective triple form, so G1 is `[x, y, "1"]` and G2 is
//!   `[[x0, x1], [y0, y1], ["1", "0"]]` (`wasm_curve.js:338-356`).
//! * G2 nests `c0` first, the opposite of the `c1`-first order the same points take when
//!   they enter a transcript hash.
//! * `nPublic` stays a JSON **number**. `stringifyBigInts` returns non-bigint values
//!   unchanged, so it is the one field that is not a string.
//!
//! `vk_alphabeta_12` is `e(alpha_1, beta_2)` in `F12`, nested 2 x 3 x 2 for twelve decimal
//! strings (`engine.js:30-33`). It is redundant, a precomputed pairing verifiers may use.
//!
//! This crate already reads a `verification_key.json` on the proving side
//! ([`g16_zkey::VerifyingKey::from_json`]), so the two must agree on field names. The file
//! written here is byte-identical to snarkjs' for all thirteen artifacts, whitespace
//! included: `bfj` at `{space: 1}` is `JSON.stringify(v, null, 1)` and ends without a
//! trailing newline.

use std::path::Path;

use g16_field::{Bn254, Fq, Fq2, G1Affine, G2Affine, Pairing};
use g16_zkey::binfile::BinFile;
use serde_json::Value;

use crate::setup::{S_HEADER, S_IC};
use crate::{CeremonyError, Groth16Header, SG1};

/// A coordinate as snarkjs writes it: the normal-form integer in decimal.
///
/// arkworks' `Display` for a field element is `into_bigint().to_string()`, which is
/// exactly that, so this is the same value `g16_core::json::dec` produces. That helper is
/// not reachable from here (`g16-core` is not a dependency of this crate, and the manifest
/// is fixed), and one `to_string` is not worth a dependency edge.
fn dec<F: std::fmt::Display>(x: F) -> Value {
    Value::String(x.to_string())
}

/// `[x, y, "1"]`, ffjavascript's projective triple with `z` normalised.
///
/// The identity is `[0, 1, 0]` and not the affine `(0, 0)` on disk: the affine zero is not
/// a point on the curve, and a verifier that reads it back would reject it. An `IC` entry
/// is the identity when a public signal appears in no constraint, which is rare but not
/// impossible.
fn g1_json(p: &G1Affine) -> Value {
    if p.infinity {
        Value::Array(vec![
            dec(Fq::from(0u64)),
            dec(Fq::from(1u64)),
            dec(Fq::from(0u64)),
        ])
    } else {
        Value::Array(vec![dec(p.x), dec(p.y), dec(Fq::from(1u64))])
    }
}

/// `[c0, c1]`, the opposite of the `c1`-first order the same coordinate takes in a
/// transcript hash ([`crate::transcript::g2_uncompressed`]).
fn fq2_json(x: &Fq2) -> Value {
    Value::Array(vec![dec(x.c0), dec(x.c1)])
}

fn g2_json(p: &G2Affine) -> Value {
    if p.infinity {
        Value::Array(vec![
            fq2_json(&Fq2::new(Fq::from(0u64), Fq::from(0u64))),
            fq2_json(&Fq2::new(Fq::from(0u64), Fq::from(0u64))),
            fq2_json(&Fq2::new(Fq::from(0u64), Fq::from(0u64))),
        ])
    } else {
        Value::Array(vec![
            fq2_json(&p.x),
            fq2_json(&p.y),
            fq2_json(&Fq2::new(Fq::from(1u64), Fq::from(0u64))),
        ])
    }
}

/// The keys in snarkjs' insertion order, which is the order they appear in the file.
///
/// `serde_json::Value`'s map is a `BTreeMap` here (the `preserve_order` feature is off in
/// this workspace), so the order cannot live in the `Value` and has to be carried
/// alongside it. [`verification_key_json`] hands back the `Value` for callers that only
/// read fields; [`export_verification_key`] formats from this list so the file matches.
fn fields(zkey: &Path) -> Result<Vec<(&'static str, Value)>, CeremonyError> {
    let file = BinFile::open(zkey, b"zkey", crate::setup::ZKEY_MAX_VERSION)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;

    let ic_bytes = file.unique_section(S_IC)?;
    let n_ic = header.n_public as usize + 1;
    g16_zkey::binfile::expect_records(ic_bytes, n_ic, SG1, S_IC)?;
    let ic: Vec<Value> = ic_bytes
        .chunks_exact(SG1)
        .map(|b| g1_json(&g16_zkey::binfile::g1(b)))
        .collect();

    // `e(alpha_1, beta_2)` in F12, nested 2 x 3 x 2 by the tower Fq12 = Fq2(Fq6(Fq2)).
    // Redundant, and only ever a precomputation a verifier may skip the pairing with.
    let ab = Bn254::pairing(header.alpha_g1, header.beta_g2).0;
    let alphabeta = Value::Array(
        [ab.c0, ab.c1]
            .iter()
            .map(|c| Value::Array([c.c0, c.c1, c.c2].iter().map(fq2_json).collect()))
            .collect(),
    );

    Ok(vec![
        ("protocol", Value::String("groth16".into())),
        ("curve", Value::String("bn128".into())),
        // The one field that is not a string: `stringifyBigInts` passes a plain number
        // through untouched (`ffjavascript/src/utils_bigint.js:23-25`).
        ("nPublic", Value::from(header.n_public)),
        ("vk_alpha_1", g1_json(&header.alpha_g1)),
        ("vk_beta_2", g2_json(&header.beta_g2)),
        ("vk_gamma_2", g2_json(&header.gamma_g2)),
        ("vk_delta_2", g2_json(&header.delta_g2)),
        ("vk_alphabeta_12", alphabeta),
        ("IC", Value::Array(ic)),
    ])
}

/// Build the JSON object from a zkey's section 2 and section 3.
///
/// The returned map is sorted by key, not in snarkjs' order. Use
/// [`export_verification_key`] for a file that matches byte for byte.
pub fn verification_key_json(zkey: &Path) -> Result<Value, CeremonyError> {
    Ok(Value::Object(
        fields(zkey)?
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    ))
}

/// [`verification_key_json`] written to `out`, byte-identical to `snarkjs zkey export
/// verificationkey`.
///
/// `cli.js:594` writes with `bfj` at `{space: 1}`, which is `JSON.stringify(v, null, 1)`
/// with no trailing newline.
pub fn export_verification_key(zkey: &Path, out: &Path) -> Result<(), CeremonyError> {
    let fields = fields(zkey)?;
    let mut text = String::from("{\n");
    for (i, (key, value)) in fields.iter().enumerate() {
        text.push_str(&format!(" \"{key}\": "));
        render(value, 1, &mut text);
        text.push_str(if i + 1 == fields.len() { "\n" } else { ",\n" });
    }
    text.push('}');
    std::fs::write(out, text)?;
    Ok(())
}

/// `JSON.stringify` at one space of indent, for the string / number / array tree
/// [`fields`] builds.
///
/// `serde_json::to_string_pretty` is fixed at two spaces, and reaching its formatter needs
/// `serde::Serialize` in scope, which is a transitive dependency here rather than a
/// declared one. Nine lines is cheaper than the dependency edge.
fn render(v: &Value, depth: usize, out: &mut String) {
    match v {
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                out.push_str(&" ".repeat(depth + 1));
                render(item, depth + 1, out);
                out.push_str(if i + 1 == items.len() { "\n" } else { ",\n" });
            }
            out.push_str(&" ".repeat(depth));
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}
