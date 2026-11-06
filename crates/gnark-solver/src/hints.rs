//! Hints, and the gnark standard hints the solver ships.
//!
//! A gnark hint is `func(q *big.Int, inputs, outputs []*big.Int) error`, called with each
//! input linear expression evaluated and reduced, and each output reduced mod r once it
//! returns. Here the field is always BN254's, so a hint takes and gives `Fr` directly; one
//! that wants the integer calls `into_bigint`.

use snarkrs_field::{BigInteger, Field, Fr, PrimeField, Zero};

/// A hint: the evaluated inputs in, one slot per output wire out (each 0 going in, as
/// gnark's are), or a message for the error the solve then stops with.
pub type Hint = dyn Fn(&[Fr], &mut [Fr]) -> Result<(), String> + Send + Sync;

/// gnark's hint ID for a hint function's fully qualified name: its 32-bit FNV-1a
/// (`solver.GetHintID`).
pub const fn hint_id(name: &str) -> u32 {
    let bytes = name.as_bytes();
    let mut h: u32 = 0x811c9dc5;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u32;
        h = h.wrapping_mul(0x01000193);
        i += 1;
    }
    h
}

pub const N_BITS: &str = "github.com/consensys/gnark/std/math/bits.nBits";
pub const ITH_BIT: &str = "github.com/consensys/gnark/std/math/bits.ithBit";
pub const N_TRITS: &str = "github.com/consensys/gnark/std/math/bits.nTrits";
pub const INV_ZERO: &str = "github.com/consensys/gnark/constraint/solver.InvZeroHint";

pub(crate) fn register_std(s: &mut crate::Solver) {
    s.register(N_BITS, n_bits);
    s.register(ITH_BIT, ith_bit);
    s.register(N_TRITS, n_trits);
    s.register(INV_ZERO, inv_zero);
}

fn one_input(inputs: &[Fr]) -> Result<Fr, String> {
    match inputs {
        [x] => Ok(*x),
        _ => Err(format!("expects 1 input, got {}", inputs.len())),
    }
}

/// `bits.nBits`: the low bits of the input, one per output.
pub fn n_bits(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    let n = one_input(inputs)?.into_bigint();
    for (i, o) in outputs.iter_mut().enumerate() {
        *o = Fr::from(u64::from(i < 256 && n.get_bit(i)));
    }
    Ok(())
}

/// `bits.ithBit`: bit `inputs[1]` of `inputs[0]`, 0 when the index is past a u64.
pub fn ith_bit(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    let [n, i] = inputs else {
        return Err(format!("expects 2 inputs, got {}", inputs.len()));
    };
    let (n, i) = (n.into_bigint(), i.into_bigint());
    let bit = i.0[1..].iter().all(|&l| l == 0) && i.0[0] < 256 && n.get_bit(i.0[0] as usize);
    if let Some(o) = outputs.first_mut() {
        *o = Fr::from(u64::from(bit));
    }
    Ok(())
}

/// `bits.nTrits`: the low base-3 digits of the input, one per output.
pub fn n_trits(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    let mut n = num_bigint::BigUint::from_bytes_le(&one_input(inputs)?.into_bigint().to_bytes_le());
    let three = num_bigint::BigUint::from(3u32);
    for o in outputs.iter_mut() {
        let digit = &n % &three;
        *o = Fr::from(digit.iter_u64_digits().next().unwrap_or(0));
        n /= &three;
    }
    Ok(())
}

/// `solver.InvZeroHint`: `1 / x`, or 0 for 0.
pub fn inv_zero(inputs: &[Fr], outputs: &mut [Fr]) -> Result<(), String> {
    let x = one_input(inputs)?;
    if let Some(o) = outputs.first_mut() {
        *o = x.inverse().unwrap_or_else(Fr::zero);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_gnarks() {
        // Read off the SP1 v6.1.0 circuit's MHintsDependencies.
        assert_eq!(hint_id(N_BITS), 4115454955);
        assert_eq!(hint_id(INV_ZERO), 2534161455);
        assert_eq!(hint_id(ITH_BIT), 3428870437);
        assert_eq!(hint_id(N_TRITS), 2140186541);
    }

    #[test]
    fn std_hints() {
        let mut out = [Fr::zero(); 5];
        n_bits(&[Fr::from(0b10110u64)], &mut out).unwrap();
        assert_eq!(out, [0, 1, 1, 0, 1].map(|b: u64| Fr::from(b)));
        n_trits(&[Fr::from(5u64 + 2 * 9 + 81)], &mut out).unwrap();
        assert_eq!(out, [2, 1, 2, 0, 1].map(|b: u64| Fr::from(b)));
        let mut one = [Fr::zero()];
        ith_bit(&[Fr::from(8u64), Fr::from(3u64)], &mut one).unwrap();
        assert_eq!(one[0], Fr::from(1u64));
        ith_bit(&[Fr::from(8u64), -Fr::from(1u64)], &mut one).unwrap();
        assert_eq!(one[0], Fr::zero());
        inv_zero(&[Fr::zero()], &mut one).unwrap();
        assert_eq!(one[0], Fr::zero());
        inv_zero(&[Fr::from(7u64)], &mut one).unwrap();
        assert_eq!(one[0] * Fr::from(7u64), Fr::from(1u64));
    }
}
