//! `zkey export soliditycalldata` (alias `generatecall`) for a groth16 proof: the argument
//! list a Solidity `verifyProof` call takes, as the one line snarkjs 0.7.6 prints.
//!
//! `groth16_exportsoliditycalldata.js:22-44`, which is string assembly over
//! `unstringifyBigInts(proof)` and `unstringifyBigInts(public)`:
//!
//! * **Every number is `"0x"` plus 64 lowercase hex digits**, left padded with zeros
//!   (`p256`). A value wider than 256 bits is not truncated; the padding loop only ever
//!   adds.
//! * **The G2 coordinates are swapped**, `[x1, x0]` then `[y1, y0]`, because the EVM
//!   precompile wants `Fq2` imaginary part first. `pi_b[2]`, the projective `z`, is
//!   dropped along with `pi_a[2]` and `pi_c[2]`.
//! * **Separators are not uniform.** The proof points are joined with `", "` and `","`
//!   where the template happens to have them, and the public inputs with a bare `","`.
//!   This is what makes the output impossible to produce by serialising a JSON array.
//!
//! `unstringifyBigInts` only converts strings that are all decimal digits or `0x` hex
//! (`ffjavascript/src/utils.js:23-27`). Any other string survives as itself and `p256`
//! pads it as text, and a JSON number goes through `Number.prototype.toString(16)`. Both
//! are reproduced for the cases that have one meaning; a value snarkjs would crash on (a
//! missing coordinate, `null`) or print as a float in base 16 is an error here instead.
//!
//! Which exporter runs is decided by the caller on `proof.protocol` (`cli.js:678-686`);
//! this is the `"groth16"` arm.

use num_bigint::BigUint;
use serde_json::Value;

use crate::ZkeyError;

/// `snarkjs zkey export soliditycalldata <public.json> <proof.json>` for a groth16 proof:
/// the text it prints, without the trailing newline `console.log` adds.
pub fn groth16_solidity_calldata(proof: &Value, public: &Value) -> Result<String, ZkeyError> {
    let at = |path: &[usize], key: &str| -> Result<String, ZkeyError> {
        let mut v = proof
            .get(key)
            .ok_or_else(|| ZkeyError::BadJson(format!("proof has no {key}")))?;
        for &i in path {
            v = v
                .get(i)
                .ok_or_else(|| ZkeyError::BadJson(format!("proof {key} has no element {i}")))?;
        }
        p256(v)
    };
    let inputs = public
        .as_array()
        .ok_or_else(|| ZkeyError::BadJson("public signals are not an array".into()))?
        .iter()
        .map(p256)
        .collect::<Result<Vec<_>, _>>()?
        .join(",");
    Ok(format!(
        "[{}, {}],[[{}, {}],[{}, {}]],[{}, {}],[{}]",
        at(&[0], "pi_a")?,
        at(&[1], "pi_a")?,
        at(&[0, 1], "pi_b")?,
        at(&[0, 0], "pi_b")?,
        at(&[1, 1], "pi_b")?,
        at(&[1, 0], "pi_b")?,
        at(&[0], "pi_c")?,
        at(&[1], "pi_c")?,
        inputs,
    ))
}

/// `p256` over what `unstringifyBigInts` left behind, quoted.
fn p256(v: &Value) -> Result<String, ZkeyError> {
    let hex = match v {
        Value::String(s) => {
            let parsed = if let Some(h) = s.strip_prefix("0x") {
                is_all(h, |c| c.is_ascii_hexdigit()).then(|| BigUint::parse_bytes(h.as_bytes(), 16))
            } else {
                is_all(s, |c| c.is_ascii_digit()).then(|| BigUint::parse_bytes(s.as_bytes(), 10))
            };
            match parsed.flatten() {
                Some(n) => n.to_str_radix(16),
                // Not a number to ffjavascript, so it is still a string, and
                // `String.prototype.toString(16)` ignores the radix.
                None => s.clone(),
            }
        }
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                js_int_hex(u as f64, u)
            } else if let Some(i) = n.as_i64() {
                format!("-{}", js_int_hex(i.unsigned_abs() as f64, i.unsigned_abs()))
            } else {
                return Err(ZkeyError::BadJson(format!(
                    "{n} is not an integer; snarkjs would print it as a base-16 float"
                )));
            }
        }
        other => {
            return Err(ZkeyError::BadJson(format!(
                "{other} is not a number snarkjs can print as calldata"
            )))
        }
    };
    let pad = 64usize.saturating_sub(hex.chars().count());
    Ok(format!("\"0x{}{hex}\"", "0".repeat(pad)))
}

fn is_all(s: &str, f: impl Fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(f)
}

/// `Number.prototype.toString(16)` of an integer that `JSON.parse` has already rounded to
/// a double. Exact below 2^53; above it the double is the value snarkjs sees.
fn js_int_hex(as_double: f64, exact: u64) -> String {
    if exact < (1u64 << 53) {
        format!("{exact:x}")
    } else {
        format!("{:x}", as_double as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p256_follows_unstringify() {
        let s = |v: &str| p256(&Value::String(v.into())).unwrap();
        assert_eq!(s("255"), format!("\"0x{}ff\"", "0".repeat(62)));
        assert_eq!(s("0xFF"), format!("\"0x{}ff\"", "0".repeat(62)));
        assert_eq!(s("007"), format!("\"0x{}7\"", "0".repeat(63)));
        // Left alone by `unstringifyBigInts`, then padded as text.
        assert_eq!(s("-5"), format!("\"0x{}-5\"", "0".repeat(62)));
        assert_eq!(
            p256(&serde_json::json!(-5)).unwrap(),
            format!("\"0x{}-5\"", "0".repeat(62))
        );
        assert_eq!(
            p256(&serde_json::json!(16)).unwrap(),
            format!("\"0x{}10\"", "0".repeat(62))
        );
        assert!(p256(&Value::Null).is_err());
        assert!(p256(&serde_json::json!(1.5)).is_err());
    }
}
