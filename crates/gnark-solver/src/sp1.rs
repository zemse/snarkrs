//! SP1's Groth16 wrap circuit: its four KoalaBear hints and its witness JSON.
//!
//! The hints are `go/sp1/koalabear/koalabear.go` in sp1-recursion-gnark-ffi 6.10.0, which
//! calls back into Rust (`src/koalabear.rs`, Plonky3's KoalaBear and its degree 4 binomial
//! extension) for the two inverses. KoalaBear is p = 2^31 - 2^24 + 1 and the extension is
//! F_p[X] / (X^4 - 3), as `MulE` there multiplies. An inverse is unique, so computing it
//! another way gives SP1's canonical u32 exactly. Where SP1's Rust panics (the inverse of
//! zero, which aborts the Go process through the FFI) the hint returns an error instead.

use crate::Solver;
use snarkrs_field::{BigInteger, Fr, PrimeField};

/// KoalaBear's modulus.
pub const P: u64 = 2130706433;

pub const REDUCE: &str = "github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.ReduceHint";
pub const SPLIT_LIMBS: &str =
    "github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.SplitLimbsHint";
pub const INV_F: &str = "github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.InvFHint";
pub const INV_E: &str = "github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.InvEHint";

impl Solver {
    /// A solver for SP1's circuit: gnark's standard hints and SP1's four.
    pub fn sp1() -> Self {
        let mut s = Self::new();
        register(&mut s);
        s
    }
}

/// Register SP1's four KoalaBear hints.
pub fn register(s: &mut Solver) {
    s.register(REDUCE, reduce);
    s.register(SPLIT_LIMBS, split_limbs);
    s.register(INV_F, inv_f);
    s.register(INV_E, inv_e);
}

/// `x` as an integer, divided by `d`: quotient and remainder.
fn div_rem(x: &Fr, d: u64) -> (Fr, u64) {
    let mut limbs = x.into_bigint().0;
    let mut rem = 0u128;
    for limb in limbs.iter_mut().rev() {
        let cur = rem << 64 | u128::from(*limb);
        *limb = (cur / u128::from(d)) as u64;
        rem = cur % u128::from(d);
    }
    // The quotient is at most `x`, so below r.
    (
        Fr::from_bigint(ark_ff::BigInt::new(limbs)).unwrap(),
        rem as u64,
    )
}

fn input(inputs: &[Fr], n: usize) -> Result<(), String> {
    if inputs.len() != n {
        return Err(format!(
            "expects {n} input operand(s), got {}",
            inputs.len()
        ));
    }
    Ok(())
}

/// `ReduceHint`: the input over p, quotient then remainder.
pub fn reduce(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    input(inputs, 1)?;
    let (q, r) = div_rem(&inputs[0], P);
    outputs[0] = q;
    outputs[1] = Fr::from(r);
    Ok(())
}

/// `SplitLimbsHint`: a KoalaBear element as its low 24 bits and its high 7.
pub fn split_limbs(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    input(inputs, 1)?;
    let v = inputs[0].into_bigint();
    if v.0[1..].iter().any(|&l| l != 0) || v.0[0] >= P {
        return Err("input is not in the field".into());
    }
    outputs[0] = Fr::from(v.0[0] & ((1 << 24) - 1));
    outputs[1] = Fr::from(v.0[0] >> 24);
    Ok(())
}

/// `InvFHint`: the inverse of the input mod p.
pub fn inv_f(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    input(inputs, 1)?;
    let a = div_rem(&inputs[0], P).1;
    if a == 0 {
        return Err("inverse of zero".into());
    }
    outputs[0] = Fr::from(pow(a, P - 2));
    Ok(())
}

/// `InvEHint`: the inverse of `a + bX + cX^2 + dX^3`, each input reduced mod p.
pub fn inv_e(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    input(inputs, 4)?;
    let x: [u64; 4] = std::array::from_fn(|i| div_rem(&inputs[i], P).1);
    if x == [0; 4] {
        return Err("inverse of zero".into());
    }
    // x^(p^4 - 2), the inverse in the multiplicative group of order p^4 - 1.
    let mut e = u128::from(P).pow(4) - 2;
    let (mut base, mut acc) = (x, [1, 0, 0, 0]);
    while e > 0 {
        if e & 1 == 1 {
            acc = ext_mul(&acc, &base);
        }
        base = ext_mul(&base, &base);
        e >>= 1;
    }
    for (o, v) in outputs.iter_mut().zip(acc) {
        *o = Fr::from(v);
    }
    Ok(())
}

fn pow(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % P;
        }
        b = b * b % P;
        e >>= 1;
    }
    r
}

fn ext_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut r = [0u64; 4];
    for i in 0..4 {
        for j in 0..4 {
            let t = a[i] * b[j] % P;
            if i + j >= 4 {
                r[i + j - 4] = (r[i + j - 4] + 3 * t) % P;
            } else {
                r[i + j] = (r[i + j] + t) % P;
            }
        }
    }
    r
}

/// The witness JSON SP1 writes (`WitnessInput` in `go/sp1/sp1.go`).
#[derive(serde::Deserialize)]
pub struct Witness {
    pub vars: Vec<String>,
    pub felts: Vec<String>,
    pub exts: Vec<Vec<String>>,
    pub vkey_hash: String,
    pub committed_values_digest: String,
    pub exit_code: String,
    pub vk_root: String,
    pub proof_nonce: String,
}

impl Witness {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("SP1 witness JSON: {e}"))
    }

    /// The assignment `frontend.NewWitness` makes of `NewCircuit(witnessInput)`: the five
    /// public inputs in field order, then `Vars`, `Felts` and each of `Exts`' four values,
    /// a KoalaBear variable being one wire (its `Value`; `UpperBound` is no wire).
    pub fn assignment(&self) -> Result<(Vec<Fr>, Vec<Fr>), String> {
        let public = [
            &self.vkey_hash,
            &self.committed_values_digest,
            &self.exit_code,
            &self.vk_root,
            &self.proof_nonce,
        ]
        .into_iter()
        .map(|s| parse_go_int(s))
        .collect::<Result<Vec<_>, _>>()?;
        let mut secret =
            Vec::with_capacity(self.vars.len() + self.felts.len() + 4 * self.exts.len());
        for s in self.vars.iter().chain(&self.felts) {
            secret.push(parse_go_int(s)?);
        }
        for e in &self.exts {
            if e.len() != 4 {
                return Err(format!("an extension element of {} values", e.len()));
            }
            for s in e {
                secret.push(parse_go_int(s)?);
            }
        }
        Ok((public, secret))
    }
}

/// A string as `fr.Element.SetString` reads one: `big.Int.SetString(s, 0)`, so an optional
/// sign then a 0x, 0o, 0b or bare 0 (octal) prefix, reduced mod r. Underscores, which Go
/// accepts between digits after a prefix, are refused.
pub fn parse_go_int(s: &str) -> Result<Fr, String> {
    let bad = || format!("{s:?} is not an integer");
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let lower = digits.to_ascii_lowercase();
    let (radix, body) = if let Some(b) = lower.strip_prefix("0x") {
        (16, b)
    } else if let Some(b) = lower.strip_prefix("0b") {
        (2, b)
    } else if let Some(b) = lower.strip_prefix("0o") {
        (8, b)
    } else if lower.len() > 1 && lower.starts_with('0') {
        (8, &lower[1..])
    } else {
        (10, lower.as_str())
    };
    // num-bigint takes underscores anywhere, Go only after a prefix.
    if body.is_empty() || !body.bytes().all(|c| c.is_ascii_alphanumeric()) {
        return Err(bad());
    }
    let v = num_bigint::BigUint::parse_bytes(body.as_bytes(), radix).ok_or_else(bad)?;
    let r = num_bigint::BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le());
    let v = Fr::from_le_bytes_mod_order(&(v % r).to_bytes_le());
    Ok(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::Zero;

    #[test]
    fn ids_are_sp1s() {
        // Read off the SP1 v6.1.0 circuit's MHintsDependencies.
        assert_eq!(crate::hint_id(REDUCE), 342246697);
        assert_eq!(crate::hint_id(SPLIT_LIMBS), 3038391954);
        assert_eq!(crate::hint_id(INV_F), 3795648912);
        assert_eq!(crate::hint_id(INV_E), 3205574893);
    }

    #[test]
    fn inverses_invert() {
        let x = [5u64, P - 1, 0, 123456789];
        let mut out = [Fr::zero(); 4];
        inv_e(&x.map(Fr::from), &mut out).unwrap();
        let y: [u64; 4] = std::array::from_fn(|i| div_rem(&out[i], P).1);
        assert_eq!(ext_mul(&x, &y), [1, 0, 0, 0]);
        let mut one = [Fr::zero()];
        inv_f(&[Fr::from(P + 7)], &mut one).unwrap();
        assert_eq!(div_rem(&one[0], P).1 * 7 % P, 1);
        assert!(inv_f(&[Fr::from(P)], &mut one).is_err());
        assert!(inv_e(&[Fr::from(P); 4], &mut out).is_err());
    }

    #[test]
    fn reduce_and_split() {
        let x = -Fr::from(5u64);
        let mut out = [Fr::zero(); 2];
        reduce(&[x], &mut out).unwrap();
        assert_eq!(out[0] * Fr::from(P) + out[1], x);
        split_limbs(&[Fr::from(P - 1)], &mut out).unwrap();
        assert_eq!(out, [Fr::from(0u64), Fr::from(127u64)]);
        assert!(split_limbs(&[Fr::from(P)], &mut out).is_err());
    }

    #[test]
    fn go_integers() {
        assert_eq!(parse_go_int("255").unwrap(), Fr::from(255u64));
        assert_eq!(parse_go_int("0xff").unwrap(), Fr::from(255u64));
        assert_eq!(parse_go_int("0377").unwrap(), Fr::from(255u64));
        assert_eq!(parse_go_int("-1").unwrap(), -Fr::from(1u64));
        assert_eq!(parse_go_int("0").unwrap(), Fr::zero());
        assert!(parse_go_int("1_0").is_err());
        assert!(parse_go_int("").is_err());
    }
}
