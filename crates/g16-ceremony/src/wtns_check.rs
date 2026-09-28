//! `wtns check`: does a witness satisfy every constraint of a circuit, `A.w * B.w = C.w`,
//! with the log lines and the verdict snarkjs 0.7.6 gives (`wtns_check.js:26-149`).
//!
//! The check is only as strict as snarkjs' and the reading is only as loose, because a
//! verdict that differs from snarkjs' on the same two files is a bug in a drop-in:
//!
//! * **Constraints are read as `readLC` reads them** ([`for_each_constraint`]): a signal
//!   repeated inside one combination keeps its last coefficient, and coefficients are
//!   reduced mod r.
//! * **Witness values are reduced mod r too**, since `Fr.fromRprLE` is a Montgomery
//!   multiplication. A `.wtns` holding `v + r` for some wire passes where `v` does.
//! * **The witness is sliced by the curve's 32 bytes**, not by the `n8` its own header
//!   declares, and neither `nWitness` against `nVars` nor `w[0] == 1` is checked. A wrong
//!   constant wire fails only through a constraint that uses it, as it does in snarkjs.
//! * **The primes are compared as integers**: the r1cs `r` against the witness header's
//!   field, so a witness for another curve is refused before any constraint is read, with
//!   snarkjs' message ("proving key" included, though no key is involved).
//! * **The first failing constraint stops the walk** and is the one named.
//!
//! One divergence. A constraint that names a signal past the end of the witness makes
//! snarkjs read whatever sits in its wasm memory after a short slice; that is an error
//! here, since there is no value to reproduce.

use std::path::Path;

use g16_field::{Fr, PrimeField, Zero};
use g16_zkey::binfile::{BinFile, Cursor};

use crate::r1cs::{R1cs, S_CUSTOM_GATES_LIST, S_CUSTOM_GATES_USES};
use crate::r1cs_export::{for_each_constraint, LinearCombination};
use crate::snarkjs_log::SnarkjsLog;
use crate::{CeremonyError, N8};

/// `readBinFile(wtnsFilename, "wtns", 2, ...)` (`wtns_check.js:43`).
const WTNS_MAX_VERSION: u32 = 2;

/// `··· processing` is logged every this many constraints (`wtns_check.js:78`).
const PROGRESS_EVERY: u32 = 500_000;

/// `snarkjs wtns check <circuit.r1cs> <witness.wtns>`: log what snarkjs logs and return
/// whether the witness satisfies the circuit. snarkjs exits 0 on `true`, 1 on `false`.
pub fn wtns_check<L: SnarkjsLog + ?Sized>(
    r1cs_path: &Path,
    wtns_path: &Path,
    log: &mut L,
) -> Result<bool, CeremonyError> {
    log.info("WITNESS CHECKING STARTED")?;
    log.info("> Reading r1cs file")?;
    let r1cs = R1cs::open(r1cs_path)?;
    log.info("> Reading witness file")?;
    let wtns = BinFile::open(wtns_path, b"wtns", WTNS_MAX_VERSION)?;
    let mut header = Cursor::new(wtns.unique_section(1)?, 1);
    let n8 = header.u32()? as usize;
    let q = header.take(n8)?;
    header.u32()?;
    // `readHeader` closes the section with `endReadSection`, which refuses leftovers.
    if header.remaining() != 0 {
        return Err(CeremonyError::malformed(
            1,
            format!("{} bytes left after the witness header", header.remaining()),
        ));
    }
    if !same_integer(q, &crate::r_le()) {
        return Err(CeremonyError::WitnessCurveMismatch);
    }
    let witness = wtns.unique_section(2)?;

    let h = *r1cs.header();
    let has = |id: u32| r1cs.file().sections().iter().any(|s| s.id == id);
    let custom_gates = has(S_CUSTOM_GATES_LIST) && has(S_CUSTOM_GATES_USES);
    for line in [
        "----------------------------".to_string(),
        "  WITNESS CHECK".to_string(),
        "  Curve:          bn128".to_string(),
        format!("  Vars (wires):   {}", h.n_vars),
        format!("  Outputs:        {}", h.n_outputs),
        format!("  Public Inputs:  {}", h.n_pub_inputs),
        format!("  Private Inputs: {}", h.n_prv_inputs),
        format!(
            "  Labels:         {}",
            g16_zkey::json_out::js_number(h.n_labels)
        ),
        format!("  Constraints:    {}", h.n_constraints),
        format!("  Custom Gates:   {custom_gates}"),
        "----------------------------".to_string(),
    ] {
        log.info(&line)?;
    }
    log.info("> Checking witness correctness")?;

    let value = |signal: u32| -> Result<Fr, CeremonyError> {
        let at = signal as usize * N8;
        witness
            .get(at..at + N8)
            .map(Fr::from_le_bytes_mod_order)
            .ok_or_else(|| {
                CeremonyError::malformed(
                    2,
                    format!(
                        "the witness has {} values, a constraint reads signal {signal}",
                        witness.len() / N8
                    ),
                )
            })
    };
    let eval = |lc: &LinearCombination| -> Result<Fr, CeremonyError> {
        let mut acc = Fr::zero();
        for (signal, coef) in lc {
            acc += value(*signal)? * coef;
        }
        Ok(acc)
    };

    let mut ok = true;
    for_each_constraint(&r1cs, |i, [a, b, c]| {
        if i != 0 && i % PROGRESS_EVERY == 0 {
            log.info(&format!(
                "··· processing r1cs constraints {i}/{}",
                h.n_constraints
            ))?;
        }
        if eval(&a)? * eval(&b)? - eval(&c)? != Fr::zero() {
            log.warn(&format!("··· aborting checking process at constraint {i}"))?;
            ok = false;
            return Ok(false);
        }
        Ok(true)
    })?;

    if ok {
        log.info("WITNESS IS CORRECT")?;
        log.info("WITNESS CHECKING FINISHED SUCCESSFULLY")?;
    } else {
        log.warn("WITNESS IS NOT CORRECT")?;
        log.warn("WITNESS CHECKING FINISHED UNSUCCESSFULLY")?;
    }
    Ok(ok)
}

/// Two little-endian integers of possibly different widths, compared by value, as
/// `Scalar.eq` compares what `readBigInt` returns.
fn same_integer(a: &[u8], b: &[u8]) -> bool {
    let trim = |x: &[u8]| -> usize { x.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1) };
    a[..trim(a)] == b[..trim(b)]
}
