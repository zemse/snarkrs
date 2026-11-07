//! Groth16 BN254 keys and proofs between gnark v0.14 and snarkjs.
//!
//! * [`proof`]: proofs, public inputs and verifying keys, gnark's binary forms (raw and
//!   compressed) to snarkjs' `proof.json`, `public.json` and `verification_key.json`, and
//!   back. No arithmetic: both libraries verify the same equation over the same points.
//! * [`to_zkey`]: a gnark `WriteDump` proving key, its verifying key and its R1CS to a
//!   `.zkey` that snarkjs' prover, rapidsnark and snarkrs' zkey path prove with.
//!
//! The two keys hold the same toxic waste in the same groups. Everything but the H query
//! carries over point for point, because both libraries number wires the same way (the
//! constant wire, the public wires, the rest) and put constraint `c` on `omega^c` of the
//! same domain, the 2^28-th root of unity squared down. The H query is a change of basis:
//!
//! * snarkjs' section 9 is `h_i = [L_{2i+1}(tau) / delta]_1` for `i < n`, the Lagrange
//!   polynomials of the `2n`-th roots of unity at their odd points `s omega^i`, with `s`
//!   the primitive `2n`-th root whose square is `omega`.
//! * gnark's `G1.Z` is `Z_j = [tau^j t(tau) / delta]_1` for `j < n - 1`, `t = x^n - 1`,
//!   stored bit-reversed.
//!
//! `x^j t(x)` has degree below `2n` and vanishes on the even points, and `t = -2` on the
//! odd ones, so interpolating it on the `2n` roots gives `Z_j = -2 sum_i (s omega^i)^j h_i`
//! for every `j < n`: a forward DFT of `h` scaled by `-2 s^j`, exact. Inverted, that is
//! `h_i = -1/(2n) sum_j (s omega^i)^-j Z_j`, which needs `Z_{n-1}` and gnark does not keep
//! it. [`basis`] puts zero in its place. The `h'` that gives differs from the true `h` by a
//! multiple of `(s omega^i)^(1-n)`, and against the evaluations `P(s omega^i) = -2 H(s
//! omega^i)` a prover pairs with section 9 that difference contributes a multiple of the
//! sum of `x H(x)` over the coset, which is zero whenever `H` has degree at most `n - 2`
//! (every power `x^1` to `x^(n-1)` sums to zero over it). It always has: every prover
//! builds C as `A * B` on the constraint rows, so `A * B - C` is divisible by `t` with a
//! quotient of degree `n - 2` at most. So the converted section 9 proves exactly what the
//! original key proves, and is still not the section 9 a setup from a ptau would write:
//! `snarkjs zkey verify` against any ptau refuses the converted zkey, and that is expected.

pub mod basis;
pub mod proof;
pub mod to_zkey;

use snarkrs_field::Fr;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConvertError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gnark(#[from] snarkrs_gnark::GnarkError),
    #[error(transparent)]
    Zkey(#[from] snarkrs_formats::ZkeyError),
    #[error(transparent)]
    Ceremony(#[from] snarkrs_ceremony::CeremonyError),
    #[error(transparent)]
    R1cs(#[from] snarkrs_gnark_solver::ReadError),
    #[error(transparent)]
    Accel(#[from] snarkrs_msm::AccelError),
    #[error(transparent)]
    Json(#[from] snarkrs_groth16::json::JsonError),
    /// The inputs do not describe one circuit and one key, or a key this crate cannot
    /// carry over.
    #[error("{what}: {reason}")]
    Mismatch { what: &'static str, reason: String },
}

pub(crate) fn mismatch(what: &'static str, reason: impl Into<String>) -> ConvertError {
    ConvertError::Mismatch {
        what,
        reason: reason.into(),
    }
}

/// `j` with its low `bits` bits reversed: where gnark's bit-reversed `G1.Z` keeps `Z_j`.
pub(crate) fn bitrev(j: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        j.reverse_bits() >> (usize::BITS - bits)
    }
}

/// `A.w` and `B.w` for every row of a zkey's domain from its section 4, the two vectors a
/// gnark-style prover takes in place of the matrices. Rows past the constraints are zero.
pub fn zkey_rows(pk: &snarkrs_formats::ProvingKey, witness: &[Fr]) -> [Vec<Fr>; 2] {
    use rayon::prelude::*;
    use snarkrs_formats::Matrix;
    Matrix::BOTH.map(|m| {
        let m = m.index();
        let c = &pk.coeffs;
        (0..pk.domain_size)
            .into_par_iter()
            .map(|row| {
                (c.row_ptr[m][row] as usize..c.row_ptr[m][row + 1] as usize)
                    .map(|k| c.value(m, k) * witness[c.signal[m][k] as usize])
                    .sum()
            })
            .collect()
    })
}
