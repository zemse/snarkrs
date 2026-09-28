//! A circuit's input JSON, read the way `snarkjs wtns calculate` reads it.
//!
//! snarkjs runs `JSON.parse`, then ffjavascript's `unstringifyBigInts`, then circom_runtime's
//! `flatArray` and `normalize` (`BigInt(v) % p`, lifted to `[0, p)`). Every step leaves a
//! trace in what a given file means, so each is reproduced rather than approximated:
//!
//! - A JSON number is a double, so `12345678901234567890` is the integer nearest to it,
//!   not that integer, and `1.5` is a RangeError.
//! - A string goes through `BigInt(string)`: surrounding whitespace is dropped, the empty
//!   string is 0, `0x`, `0o` and `0b` prefixes work without a sign, decimal takes one.
//! - `true` is 1, `false` 0, `null` a TypeError and an object a SyntaxError.
//! - Nested arrays flatten depth first.
//! - Keys keep `Object.keys` order: array-index keys ascending, then the rest in the order
//!   the file first names them, and a repeated key keeps its first place and its last
//!   value. The order decides which error a bad file reports, and whether the circuit ran
//!   (and logged) before it.

use num_bigint::{BigInt, Sign};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::{JsKind, WitnessError};

/// A parsed JSON value, keeping what `serde_json::Value` would lose: object key order and
/// the fact that every number is a double.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// The circuit's inputs: signal names in the order they are set, each with its values.
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub(crate) signals: Vec<(String, Json)>,
}

impl Input {
    /// `JSON.parse` of the file's text, keys in `Object.keys` order.
    pub fn from_json_str(text: &str) -> Result<Self, WitnessError> {
        let json: Json = serde_json::from_str(text).map_err(|e| WitnessError::Js {
            kind: JsKind::SyntaxError,
            message: format!("{e} is not valid JSON"),
        })?;
        Self::from_json(json)
    }

    /// From a `serde_json::Value`. Its objects are sorted by key unless serde_json's
    /// `preserve_order` is on, so the signals are set in that order; for a well-formed
    /// input the witness is the same either way.
    pub fn from_value(value: &serde_json::Value) -> Result<Self, WitnessError> {
        Self::from_json(Json::from(value))
    }

    fn from_json(json: Json) -> Result<Self, WitnessError> {
        let signals = match json {
            Json::Object(o) => o,
            Json::Array(a) => a
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v))
                .collect(),
            Json::String(s) => s
                .chars()
                .enumerate()
                .map(|(i, c)| (i.to_string(), Json::String(c.to_string())))
                .collect(),
            Json::Null => {
                return Err(WitnessError::Js {
                    kind: JsKind::TypeError,
                    message: "Cannot convert undefined or null to object".into(),
                })
            }
            Json::Bool(_) | Json::Number(_) => Vec::new(),
        };
        Ok(Self { signals })
    }

    /// The signal names, in the order they will be set.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.signals.iter().map(|(k, _)| k.as_str())
    }
}

impl TryFrom<&serde_json::Value> for Input {
    type Error = WitnessError;
    fn try_from(v: &serde_json::Value) -> Result<Self, WitnessError> {
        Self::from_value(v)
    }
}

impl From<&serde_json::Value> for Json {
    fn from(v: &serde_json::Value) -> Self {
        use serde_json::Value as V;
        match v {
            V::Null => Json::Null,
            V::Bool(b) => Json::Bool(*b),
            V::Number(n) => Json::Number(n.as_f64().unwrap_or(f64::NAN)),
            V::String(s) => Json::String(s.clone()),
            V::Array(a) => Json::Array(a.iter().map(Json::from).collect()),
            V::Object(o) => Json::Object(order_keys(
                o.iter().map(|(k, v)| (k.clone(), Json::from(v))).collect(),
            )),
        }
    }
}

/// `OrdinaryOwnPropertyKeys`: array indices (canonical decimal, below 2^32 - 1) ascending,
/// then every other key in insertion order. `entries` has no duplicates.
fn order_keys(entries: Vec<(String, Json)>) -> Vec<(String, Json)> {
    let (mut idx, rest): (Vec<_>, Vec<_>) = entries
        .into_iter()
        .partition(|(k, _)| array_index(k).is_some());
    idx.sort_by_key(|(k, _)| array_index(k));
    idx.extend(rest);
    idx
}

fn array_index(k: &str) -> Option<u32> {
    let n: u32 = k.parse().ok()?;
    (n != u32::MAX && n.to_string() == k).then_some(n)
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
                Ok(Json::Bool(b))
            }
            // Rounded to the nearest double, which is what JSON.parse does with the digits.
            fn visit_u64<E>(self, n: u64) -> Result<Json, E> {
                Ok(Json::Number(n as f64))
            }
            fn visit_i64<E>(self, n: i64) -> Result<Json, E> {
                Ok(Json::Number(n as f64))
            }
            fn visit_f64<E>(self, n: f64) -> Result<Json, E> {
                Ok(Json::Number(n))
            }
            fn visit_str<E>(self, s: &str) -> Result<Json, E> {
                Ok(Json::String(s.to_owned()))
            }
            fn visit_string<E>(self, s: String) -> Result<Json, E> {
                Ok(Json::String(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = seq.next_element()? {
                    v.push(x);
                }
                Ok(Json::Array(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut v: Vec<(String, Json)> = Vec::new();
                while let Some((k, x)) = map.next_entry::<String, Json>()? {
                    // JSON.parse: a repeated key keeps its first position and its last value.
                    match v.iter_mut().find(|(e, _)| *e == k) {
                        Some(slot) => slot.1 = x,
                        None => v.push((k, x)),
                    }
                }
                Ok(Json::Object(order_keys(v)))
            }
        }
        d.deserialize_any(V)
    }
}

/// circom_runtime's `flatArray`: arrays flatten depth first, anything else is a leaf.
pub(crate) fn flatten(v: &Json) -> Vec<&Json> {
    fn fill<'a>(out: &mut Vec<&'a Json>, v: &'a Json) {
        match v {
            Json::Array(a) => a.iter().for_each(|x| fill(out, x)),
            other => out.push(other),
        }
    }
    let mut out = Vec::new();
    fill(&mut out, v);
    out
}

/// `BigInt(v)`, with V8's messages for what it refuses.
pub(crate) fn to_bigint(v: &Json) -> Result<BigInt, WitnessError> {
    let err = |kind, message: String| WitnessError::Js { kind, message };
    match v {
        Json::Bool(b) => Ok(BigInt::from(*b as u8)),
        Json::Null => Err(err(
            JsKind::TypeError,
            "Cannot convert null to a BigInt".into(),
        )),
        Json::Number(n) => f64_to_bigint(*n).ok_or_else(|| {
            err(
                JsKind::RangeError,
                format!(
                    "The number {} cannot be converted to a BigInt because it is not an integer",
                    js_number(*n)
                ),
            )
        }),
        Json::String(s) => string_to_bigint(s).ok_or_else(|| {
            err(
                JsKind::SyntaxError,
                format!("Cannot convert {s} to a BigInt"),
            )
        }),
        Json::Object(_) => Err(err(
            JsKind::SyntaxError,
            "Cannot convert [object Object] to a BigInt".into(),
        )),
        // `flatten` never yields one.
        Json::Array(_) => unreachable!("arrays are flattened before conversion"),
    }
}

/// The exact integer value of an integral double.
fn f64_to_bigint(n: f64) -> Option<BigInt> {
    if !n.is_finite() || n.fract() != 0.0 {
        return None;
    }
    let bits = n.to_bits();
    let negative = bits >> 63 == 1;
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1 << 52) - 1);
    if exp == 0 {
        // Zero or subnormal, and a nonzero subnormal is not integral.
        return Some(BigInt::from(0));
    }
    let mantissa = BigInt::from(frac | (1 << 52));
    let shift = exp - 1075;
    let mag = if shift >= 0 {
        mantissa << shift as usize
    } else {
        mantissa >> (-shift) as usize
    };
    Some(if negative { -mag } else { mag })
}

/// `Number.prototype.toString` for the doubles that reach an error: non-integers, which
/// are always below 1e21, so only the small end switches to exponent form.
fn js_number(n: f64) -> String {
    if n != 0.0 && n.abs() < 1e-6 {
        format!("{n:e}")
    } else {
        format!("{n}")
    }
}

/// `StringToBigInt`. None where V8 throws a SyntaxError.
fn string_to_bigint(s: &str) -> Option<BigInt> {
    let t = s.trim_matches(is_js_space);
    if t.is_empty() {
        return Some(BigInt::from(0));
    }
    let b = t.as_bytes();
    if b.len() > 2 && b[0] == b'0' {
        let radix = match b[1] {
            b'x' | b'X' => 16,
            b'o' | b'O' => 8,
            b'b' | b'B' => 2,
            _ => 0,
        };
        if radix != 0 {
            return parse_digits(&t[2..], radix).map(|m| BigInt::from_biguint(Sign::Plus, m));
        }
    }
    let (sign, digits) = match b[0] {
        b'-' => (Sign::Minus, &t[1..]),
        b'+' => (Sign::Plus, &t[1..]),
        _ => (Sign::Plus, t),
    };
    parse_digits(digits, 10).map(|m| BigInt::from_biguint(sign, m))
}

fn parse_digits(d: &str, radix: u32) -> Option<num_bigint::BigUint> {
    if d.is_empty() || !d.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    num_bigint::BigUint::parse_bytes(d.as_bytes(), radix)
}

/// ECMAScript's WhiteSpace and LineTerminator.
fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\x0b' | '\x0c' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// circom_runtime's `fnvHash`: 64-bit FNV-1a over the name's UTF-16 code units, split into
/// the two 32-bit halves the module takes, high first.
pub(crate) fn fnv_hash(name: &str) -> (u32, u32) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for u in name.encode_utf16() {
        h ^= u as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    ((h >> 32) as u32, h as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(v: &Json) -> String {
        to_bigint(v).unwrap().to_string()
    }

    #[test]
    fn keys_follow_object_keys_order() {
        let i = Input::from_json_str(r#"{"b":1,"a":2,"10":3,"2":4,"b":5,"01":6}"#).unwrap();
        assert_eq!(i.names().collect::<Vec<_>>(), ["2", "10", "b", "a", "01"]);
        assert_eq!(i.signals[2].1, Json::Number(5.0));
    }

    #[test]
    fn numbers_are_doubles() {
        let i = Input::from_json_str(r#"{"a":12345678901234567890}"#).unwrap();
        assert_eq!(big(&i.signals[0].1), "12345678901234567168");
        assert_eq!(big(&Json::Number(-0.0)), "0");
        assert_eq!(big(&Json::Number(1e21)), "1000000000000000000000");
        assert_eq!(big(&Json::Number(-3.0)), "-3");
        let e = to_bigint(&Json::Number(1.5)).unwrap_err();
        assert_eq!(
            e.to_string(),
            "The number 1.5 cannot be converted to a BigInt because it is not an integer"
        );
        assert_eq!(e.js_name(), "RangeError");
        let e = to_bigint(&Json::Number(1.5e-7)).unwrap_err();
        assert!(e.to_string().starts_with("The number 1.5e-7 "), "{e}");
    }

    #[test]
    fn strings_follow_string_to_bigint() {
        for (s, want) in [
            ("12", "12"),
            (" 12\n", "12"),
            ("", "0"),
            ("  ", "0"),
            ("-5", "-5"),
            ("+5", "5"),
            ("0x1F", "31"),
            ("0X1f", "31"),
            ("0o17", "15"),
            ("0b101", "5"),
            ("007", "7"),
        ] {
            assert_eq!(big(&Json::String(s.into())), want, "{s:?}");
        }
        for s in ["abc", "-0x5", "0x", "1e3", "1_000", "1.0", "- 5", "0xg"] {
            let e = to_bigint(&Json::String(s.into())).unwrap_err();
            assert_eq!(e.to_string(), format!("Cannot convert {s} to a BigInt"));
            assert_eq!(e.js_name(), "SyntaxError");
        }
    }

    #[test]
    fn other_leaves() {
        assert_eq!(big(&Json::Bool(true)), "1");
        assert_eq!(big(&Json::Bool(false)), "0");
        let e = to_bigint(&Json::Null).unwrap_err();
        assert_eq!(
            (e.js_name(), e.to_string().as_str()),
            ("TypeError", "Cannot convert null to a BigInt")
        );
        let e = to_bigint(&Json::Object(vec![])).unwrap_err();
        assert_eq!(e.to_string(), "Cannot convert [object Object] to a BigInt");
    }

    #[test]
    fn arrays_flatten_depth_first() {
        let v: Json = serde_json::from_str(r#"[[1,[2,3]],[],4,[[5]]]"#).unwrap();
        let flat: Vec<String> = flatten(&v).into_iter().map(big).collect();
        assert_eq!(flat, ["1", "2", "3", "4", "5"]);
    }

    /// The reference values from circom_runtime's `fnvHash`, computed with node.
    #[test]
    fn fnv_matches_circom_runtime() {
        assert_eq!(fnv_hash("a"), (0xaf63dc4c, 0x8601ec8c));
        assert_eq!(fnv_hash(""), (0xcbf29ce4, 0x84222325));
    }

    #[test]
    fn top_level_shapes() {
        assert!(Input::from_json_str("5").unwrap().signals.is_empty());
        let e = Input::from_json_str("null").unwrap_err();
        assert_eq!(e.js_name(), "TypeError");
        let e = Input::from_json_str("not json").unwrap_err();
        assert_eq!(e.js_name(), "SyntaxError");
        let v = serde_json::json!({"b": "1", "a": ["2"]});
        let i = Input::from_value(&v).unwrap();
        assert_eq!(i.names().collect::<Vec<_>>(), ["a", "b"]);
    }
}
