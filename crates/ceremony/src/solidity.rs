//! `zkey export solidityverifier`: a Groth16 verifier contract for the EVM, generated from
//! a zkey's verification key.
//!
//! snarkjs renders `templates/verifier_groth16.sol.ejs`, which is GPL-3.0; this workspace is
//! MIT OR Apache-2.0, so the contract here is written from the verification equation and
//! the precompile specs rather than from that template. What it keeps from snarkjs is only
//! the external interface, because that is what `zkey export soliditycalldata` output is
//! fed to and what deployed callers link against:
//!
//! * contract `Groth16Verifier`, `pragma solidity >=0.7.0 <0.9.0`;
//! * `verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC,
//!   uint[N] calldata _pubSignals) public view returns (bool)`, `N = nPublic`.
//!
//! The check is `e(-A, B) e(alpha, beta) e(vk_x, gamma) e(C, delta) == 1` with
//! `vk_x = IC_0 + sum s_i IC_i`, built from the three BN254 precompiles: `0x06` ecAdd and
//! `0x07` ecMul (EIP-196) for `vk_x`, and `0x08` the pairing check (EIP-197). Every failure
//! returns `false` rather than reverting, which is what a caller of snarkjs' contract sees:
//!
//! * a public signal `>= r` is refused explicitly. `0x07` reduces its scalar mod `r`, so
//!   without the check `s` and `s + r` would both verify and a nullifier could be spent
//!   twice;
//! * `A.y >= q` is refused explicitly, because `A` is negated in the contract and `q - y`
//!   of an unreduced `y` wraps to a different value instead of failing;
//! * every other coordinate goes to a precompile untouched, and those fail on a coordinate
//!   `>= q`, a point off the curve, or (for `0x08`) a G2 point outside the subgroup.
//!
//! G2 coordinates are passed imaginary part first, EIP-197's order, which is also the
//! order `soliditycalldata` writes `_pB` in. The verification key's own G2 points are
//! emitted in that order, the reverse of the `[c0, c1]` of `verification_key.json`.
//!
//! With `nPublic == 0` the signature above would carry a `uint[0]`, which solc rejects, so
//! the parameter is dropped and the contract takes the three proof points alone.
//!
//! Constants are read from inline assembly, which solc accepts from 0.6.x on, so the
//! pragma's lower bound holds; the output is checked to compile on 0.6.12 and 0.8.20.

use std::fmt::Write as _;
use std::path::Path;

use snarkrs_field::{Fq, Fr, G1Affine, G2Affine, PrimeField};
use snarkrs_formats::binfile::BinFile;

use crate::setup::{S_HEADER, S_IC, ZKEY_MAX_VERSION};
use crate::{CeremonyError, Groth16Header, SG1};

/// The verifier contract's source for the zkey at `zkey`. Backs `snarkjs zkey export
/// solidityverifier`.
pub fn solidity_verifier(zkey: &Path) -> Result<String, CeremonyError> {
    let file = BinFile::open(zkey, b"zkey", ZKEY_MAX_VERSION)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let ic_bytes = file.unique_section(S_IC)?;
    let n_ic = header.n_public as usize + 1;
    snarkrs_formats::binfile::expect_records(ic_bytes, n_ic, SG1, S_IC)?;
    let ic = ic_bytes
        .chunks_exact(SG1)
        .map(snarkrs_formats::binfile::g1)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(render(&header, &ic))
}

/// [`solidity_verifier`] written to `out`. Backs `snarkjs zkey export solidityverifier`.
pub fn export_solidity_verifier(zkey: &Path, out: &Path) -> Result<(), CeremonyError> {
    std::fs::write(out, solidity_verifier(zkey)?)?;
    Ok(())
}

/// The identity is `(0, 0)` to the precompiles, which is also arkworks' affine zero.
fn g1_words(p: &G1Affine) -> [String; 2] {
    if p.infinity {
        ["0".into(), "0".into()]
    } else {
        [p.x.to_string(), p.y.to_string()]
    }
}

/// `x_im, x_re, y_im, y_re`, EIP-197's word order.
fn g2_words(p: &G2Affine) -> [String; 4] {
    if p.infinity {
        ["0".into(), "0".into(), "0".into(), "0".into()]
    } else {
        [
            p.x.c1.to_string(),
            p.x.c0.to_string(),
            p.y.c1.to_string(),
            p.y.c0.to_string(),
        ]
    }
}

fn modulus<F: PrimeField>() -> String {
    // `MODULUS` is a `BigInt`, whose `Display` is decimal.
    F::MODULUS.to_string()
}

fn render(h: &Groth16Header, ic: &[G1Affine]) -> String {
    let n = h.n_public as usize;
    let mut s = String::new();
    let w = &mut s;

    // Every `writeln!` below targets a `String`, which cannot fail.
    macro_rules! line {
        ($($t:tt)*) => { writeln!(w, $($t)*).unwrap() };
    }

    line!("// SPDX-License-Identifier: MIT");
    line!("//");
    line!("// Groth16 verifier over BN254, generated from a zkey by snarkrs. It checks");
    line!("// e(-A, B) e(alpha, beta) e(vk_x, gamma) e(C, delta) == 1 with the EIP-196 and");
    line!("// EIP-197 precompiles, where vk_x = IC[0] + sum pubSignals[i] IC[i + 1].");
    line!();
    line!("pragma solidity >=0.7.0 <0.9.0;");
    line!();
    line!("contract Groth16Verifier {{");
    line!("    // Scalar field order; every public signal must be below it.");
    line!("    uint256 constant R = {};", modulus::<Fr>());
    line!("    // Base field order.");
    line!("    uint256 constant Q = {};", modulus::<Fq>());
    line!();
    line!("    // G1 points are (x, y). G2 points are (x_im, x_re, y_im, y_re), EIP-197 order.");
    let [ax, ay] = g1_words(&h.alpha_g1);
    line!("    uint256 constant ALPHA_X = {ax};");
    line!("    uint256 constant ALPHA_Y = {ay};");
    for (name, p) in [
        ("BETA", &h.beta_g2),
        ("GAMMA", &h.gamma_g2),
        ("DELTA", &h.delta_g2),
    ] {
        let [x1, x0, y1, y0] = g2_words(p);
        line!("    uint256 constant {name}_X1 = {x1};");
        line!("    uint256 constant {name}_X0 = {x0};");
        line!("    uint256 constant {name}_Y1 = {y1};");
        line!("    uint256 constant {name}_Y0 = {y0};");
    }
    line!();
    for (i, p) in ic.iter().enumerate() {
        let [x, y] = g1_words(p);
        line!("    uint256 constant IC{i}_X = {x};");
        line!("    uint256 constant IC{i}_Y = {y};");
    }
    line!();

    let pub_param = if n == 0 {
        String::new()
    } else {
        format!(", uint[{n}] calldata _pubSignals")
    };
    line!("    function verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC{pub_param}) public view returns (bool valid) {{");
    line!("        assembly {{");
    line!("            // Scratch past the free memory pointer; nothing is allocated after it.");
    line!("            let p := mload(0x40)");
    line!("            let ok := 1");
    line!();
    line!("            // vk_x accumulates at p. Each term is ecMul'd at p + 0x40 and then");
    line!("            // ecAdd'ed onto (p, p + 0x20), so the two calls share one buffer.");
    line!("            mstore(p, IC0_X)");
    line!("            mstore(add(p, 0x20), IC0_Y)");
    for i in 0..n {
        line!("            {{");
        line!(
            "                let s := calldataload(add(_pubSignals, {:#x}))",
            32 * i
        );
        line!("                ok := and(ok, lt(s, R))");
        line!("                mstore(add(p, 0x40), IC{}_X)", i + 1);
        line!("                mstore(add(p, 0x60), IC{}_Y)", i + 1);
        line!("                mstore(add(p, 0x80), s)");
        line!("                ok := and(ok, staticcall(gas(), 0x07, add(p, 0x40), 0x60, add(p, 0x40), 0x40))");
        line!("                ok := and(ok, staticcall(gas(), 0x06, p, 0x80, p, 0x40))");
        line!("            }}");
    }
    line!();
    line!("            // Four (G1, G2) pairs of 0xc0 bytes each, starting with (vk_x, gamma)");
    line!("            // since vk_x is already at p.");
    line!("            mstore(add(p, 0x40), GAMMA_X1)");
    line!("            mstore(add(p, 0x60), GAMMA_X0)");
    line!("            mstore(add(p, 0x80), GAMMA_Y1)");
    line!("            mstore(add(p, 0xa0), GAMMA_Y0)");
    line!();
    line!("            // (-A, B). Negation is q - y, so y itself must be reduced; the");
    line!("            // identity (0, 0) stays (0, 0).");
    line!("            let ay := calldataload(add(_pA, 0x20))");
    line!("            ok := and(ok, lt(ay, Q))");
    line!("            mstore(add(p, 0xc0), calldataload(_pA))");
    line!("            mstore(add(p, 0xe0), mod(sub(Q, ay), Q))");
    line!("            calldatacopy(add(p, 0x100), _pB, 0x80)");
    line!();
    line!("            // (alpha, beta)");
    line!("            mstore(add(p, 0x180), ALPHA_X)");
    line!("            mstore(add(p, 0x1a0), ALPHA_Y)");
    line!("            mstore(add(p, 0x1c0), BETA_X1)");
    line!("            mstore(add(p, 0x1e0), BETA_X0)");
    line!("            mstore(add(p, 0x200), BETA_Y1)");
    line!("            mstore(add(p, 0x220), BETA_Y0)");
    line!();
    line!("            // (C, delta)");
    line!("            calldatacopy(add(p, 0x240), _pC, 0x40)");
    line!("            mstore(add(p, 0x280), DELTA_X1)");
    line!("            mstore(add(p, 0x2a0), DELTA_X0)");
    line!("            mstore(add(p, 0x2c0), DELTA_Y1)");
    line!("            mstore(add(p, 0x2e0), DELTA_Y0)");
    line!();
    line!("            ok := and(ok, staticcall(gas(), 0x08, p, 0x300, p, 0x20))");
    line!("            valid := and(ok, mload(p))");
    line!("        }}");
    line!("    }}");
    line!("}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::AffineRepr;

    /// The moduli are what the range checks compare against, so they are pinned to the
    /// published BN254 values rather than trusted to `Display`.
    #[test]
    fn the_moduli_are_bn254s_in_decimal() {
        assert_eq!(
            modulus::<Fr>(),
            "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        );
        assert_eq!(
            modulus::<Fq>(),
            "21888242871839275222246405745257275088696311157297823662689037894645226208583"
        );
    }

    /// EIP-197 takes the imaginary half first, the reverse of arkworks' `(c0, c1)`.
    #[test]
    fn g2_words_put_the_imaginary_half_first() {
        let g = G2Affine::generator();
        assert_eq!(
            g2_words(&g),
            [
                g.x.c1.to_string(),
                g.x.c0.to_string(),
                g.y.c1.to_string(),
                g.y.c0.to_string()
            ]
        );
    }

    /// `uint[0]` does not compile, so a key with no public signals drops the parameter,
    /// and its IC is the single constant term.
    #[test]
    fn no_public_signals_drops_the_parameter() {
        let g1 = G1Affine::generator();
        let g2 = G2Affine::generator();
        let h = Groth16Header {
            n_vars: 1,
            n_public: 0,
            domain_size: 1,
            alpha_g1: g1,
            beta_g1: g1,
            beta_g2: g2,
            gamma_g2: g2,
            delta_g1: g1,
            delta_g2: g2,
        };
        let src = render(&h, &[g1]);
        assert!(src.contains("uint[2] calldata _pC) public view returns (bool valid)"));
        assert!(!src.contains("_pubSignals"));
        assert!(!src.contains("IC1_X"));
    }
}
