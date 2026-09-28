//! `r1cs info`, `r1cs print` and `r1cs export json`: the three commands that show a
//! `.r1cs` to a person, reproduced byte for byte against snarkjs 0.7.6.
//!
//! All three see a constraint the way r1csfile's `readConstraints` builds it
//! (`r1csfile.js:62-117`), not the way [`R1cs::constraints`] does, and the difference is
//! visible in the output:
//!
//! * **A linear combination is a JS object keyed by signal.** `lc[idx] = val` means a
//!   signal repeated inside one combination keeps only its **last** coefficient, and
//!   `Object.keys` then yields integer keys in **ascending** order, whatever order the
//!   file had them in. [`LinearCombination`] is that object.
//! * **Coefficients are reduced mod r.** `F.fromRprLE` is a Montgomery multiplication,
//!   which reduces a stored value at or above `r` rather than rejecting it, so these
//!   views decode with `from_le_bytes_mod_order` where [`R1cs::coef`] refuses.
//! * **Numbers print through `Fr.toString`**, the wasm field's, which is plain decimal of
//!   the canonical value. `-1` is `r - 1` spelled out: the `"-1"` branch in
//!   `r1cs_print.js:34` belongs to the non-wasm `F1Field` and never fires on BN254.
//!
//! `r1cs print` then assembles each line by string concatenation (`r1cs_print.js:25-42`):
//! a coefficient of exactly 1 is dropped, a term after the first gets `" +"` in front
//! unless the coefficient starts with `-`, and the name comes from the `.sym` file, with
//! `one` shown as `1` and a wire the file does not name shown as `undefined`. "After the
//! first" means after the accumulated string stops being empty, so a leading term with
//! an empty name and a coefficient of 1 leaves the next term unprefixed. That is kept.
//!
//! `r1cs export json` is `readR1cs(name, true, true, true)` with `curve` and `F` deleted
//! (`r1cs_export_json.js:24-31`), which fixes both the key order and the two extra
//! members, `map` from section 3 and the custom-gate lists from sections 4 and 5, empty
//! arrays when the file has none. Its progress lines ("undefined: Loading constraints")
//! are the logger's, not the file's, and are not reproduced.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};

use snarkrs_field::{Fr, PrimeField};
use snarkrs_formats::binfile::Cursor;
use snarkrs_formats::json_out::{js_number, JsonWriter};

use crate::r1cs::{R1cs, S_CONSTRAINTS, S_CUSTOM_GATES_LIST, S_CUSTOM_GATES_USES};
use crate::snarkjs_log::SnarkjsLog;
use crate::sym::Syms;
use crate::{CeremonyError, N8};

/// One linear combination as r1csfile's `readLC` leaves it: signal to coefficient, last
/// write wins, iterated in ascending signal order.
pub type LinearCombination = BTreeMap<u32, Fr>;

/// `r1cs info <circuit.r1cs>`: the seven lines `r1cs_info.js:24-44` logs.
pub fn r1cs_info<L: SnarkjsLog + ?Sized>(r1cs: &R1cs, log: &mut L) -> Result<(), CeremonyError> {
    let h = r1cs.header();
    // `R1cs::open` has already refused any prime but BN254's `r`.
    log.info("Curve: bn-128")?;
    log.info(&format!("# of Wires: {}", h.n_vars))?;
    log.info(&format!("# of Constraints: {}", h.n_constraints))?;
    log.info(&format!("# of Private Inputs: {}", h.n_prv_inputs))?;
    log.info(&format!("# of Public Inputs: {}", h.n_pub_inputs))?;
    log.info(&format!("# of Labels: {}", js_number(h.n_labels)))?;
    log.info(&format!("# of Outputs: {}", h.n_outputs))?;
    Ok(())
}

/// Walk section 2 the way `readConstraints` does, handing `f` each constraint as three
/// [`LinearCombination`]s, A then B then C, in file order, until `f` returns `false`.
///
/// Streams, where [`R1cs::constraints`] collects, so `wtns check` can stop at the first
/// failure and neither printer holds the circuit. Like `readLC` it does not check a
/// signal against `nVars`; a combination running past the section is still an error,
/// where snarkjs would read zeros.
pub fn for_each_constraint<F>(r1cs: &R1cs, mut f: F) -> Result<(), CeremonyError>
where
    F: FnMut(u32, [LinearCombination; 3]) -> Result<bool, CeremonyError>,
{
    let mut cur = Cursor::new(r1cs.constraint_bytes()?, S_CONSTRAINTS);
    let mut read_lc = || -> Result<LinearCombination, CeremonyError> {
        let n = cur.u32()?;
        let mut lc = LinearCombination::new();
        for _ in 0..n {
            let signal = cur.u32()?;
            lc.insert(signal, Fr::from_le_bytes_mod_order(cur.take(N8)?));
        }
        Ok(lc)
    };
    for i in 0..r1cs.header().n_constraints {
        let lcs = [read_lc()?, read_lc()?, read_lc()?];
        if !f(i, lcs)? {
            break;
        }
    }
    Ok(())
}

/// `r1cs print <circuit.r1cs> <circuit.sym>`: one `[ A ] * [ B ] - [ C ] = 0` line per
/// constraint, logged at info.
pub fn r1cs_print<L: SnarkjsLog + ?Sized>(
    r1cs: &R1cs,
    syms: &Syms,
    log: &mut L,
) -> Result<(), CeremonyError> {
    for_each_constraint(r1cs, |_, [a, b, c]| {
        let line = format!(
            "[ {} ] * [ {} ] - [ {} ] = 0",
            lc_to_string(&a, syms),
            lc_to_string(&b, syms),
            lc_to_string(&c, syms)
        );
        log.info(&line)?;
        Ok(true)
    })
}

/// `lc2str` (`r1cs_print.js:25-40`).
fn lc_to_string(lc: &LinearCombination, syms: &Syms) -> String {
    let mut s = String::new();
    for (signal, coef) in lc {
        let name = match syms.var_name(*signal) {
            Some("one") => "1",
            Some(n) => n,
            None => "undefined",
        };
        let mut vs = fr_dec(coef);
        if vs == "1" {
            vs.clear();
        }
        if !s.is_empty() && !vs.starts_with('-') {
            vs.insert(0, '+');
        }
        if !s.is_empty() {
            vs.insert(0, ' ');
        }
        s.push_str(&vs);
        s.push_str(name);
    }
    s
}

pub(crate) fn fr_dec(v: &Fr) -> String {
    v.into_bigint().to_string()
}

/// `r1cs export json <circuit.r1cs> <circuit.json>`: the circuit as the JSON
/// `r1cs_export_json.js` writes. Streams constraints one at a time.
pub fn r1cs_export_json<W: Write>(r1cs: &R1cs, out: W) -> Result<(), CeremonyError> {
    let h = *r1cs.header();
    let file = r1cs.file();
    let has = |id: u32| file.sections().iter().any(|s| s.id == id);
    // `readR1csHeader` (`r1csfile.js:54-55`): both sections, or custom gates are off.
    let use_custom_gates = has(S_CUSTOM_GATES_LIST) && has(S_CUSTOM_GATES_USES);
    // Read before anything is written, so a bad section 3 fails without a half file.
    let map = r1cs.label_map()?;

    let mut w = JsonWriter::new(BufWriter::new(out));
    w.begin_object()?;
    w.field_raw("n8", &N8.to_string())?;
    w.field_string("prime", &Fr::MODULUS.to_string())?;
    w.field_raw("nVars", &h.n_vars.to_string())?;
    w.field_raw("nOutputs", &h.n_outputs.to_string())?;
    w.field_raw("nPubInputs", &h.n_pub_inputs.to_string())?;
    w.field_raw("nPrvInputs", &h.n_prv_inputs.to_string())?;
    w.field_raw("nLabels", &js_number(h.n_labels))?;
    w.field_raw("nConstraints", &h.n_constraints.to_string())?;
    w.field_raw(
        "useCustomGates",
        if use_custom_gates { "true" } else { "false" },
    )?;

    w.key("constraints")?;
    w.begin_array()?;
    for_each_constraint(r1cs, |_, constraint| {
        w.begin_array()?;
        for lc in &constraint {
            w.begin_object()?;
            for (signal, coef) in lc {
                w.field_string(&signal.to_string(), &fr_dec(coef))?;
            }
            w.end()?;
        }
        w.end()?;
        Ok(true)
    })?;
    w.end()?;

    w.key("map")?;
    w.begin_array()?;
    for label in map {
        w.raw(&js_number(label))?;
    }
    w.end()?;

    w.key("customGates")?;
    w.begin_array()?;
    if use_custom_gates {
        write_custom_gates(r1cs, &mut w)?;
    }
    w.end()?;
    w.key("customGatesUses")?;
    w.begin_array()?;
    if use_custom_gates {
        write_custom_gate_uses(r1cs, &mut w)?;
    }
    w.end()?;
    w.end()?;
    w.finish()?;
    Ok(())
}

/// Section 4, `readCustomGatesListSection` (`r1csfile.js:229-251`): a count, then per gate
/// a NUL-terminated template name, a parameter count and that many field elements.
fn write_custom_gates<W: Write>(r1cs: &R1cs, w: &mut JsonWriter<W>) -> Result<(), CeremonyError> {
    let data = r1cs.file().unique_section(S_CUSTOM_GATES_LIST)?;
    let mut cur = Cursor::new(data, S_CUSTOM_GATES_LIST);
    let n = cur.u32()?;
    for _ in 0..n {
        // `fd.readString()`: bytes up to the first NUL, decoded with `TextDecoder`.
        let rest = &data[data.len() - cur.remaining()..];
        let len = rest.iter().position(|&b| b == 0).ok_or_else(|| {
            CeremonyError::malformed(S_CUSTOM_GATES_LIST, "template name is not NUL-terminated")
        })?;
        let name = String::from_utf8_lossy(cur.take(len)?).into_owned();
        cur.take(1)?;
        let n_params = cur.u32()?;
        w.begin_object()?;
        w.field_string("templateName", &name)?;
        w.key("parameters")?;
        w.begin_array()?;
        for _ in 0..n_params {
            w.string(&fr_dec(&Fr::from_le_bytes_mod_order(cur.take(N8)?)))?;
        }
        w.end()?;
        w.end()?;
    }
    if cur.remaining() != 0 {
        return Err(CeremonyError::malformed(
            S_CUSTOM_GATES_LIST,
            format!("{} bytes left after the gate list", cur.remaining()),
        ));
    }
    Ok(())
}

/// Section 5, `readCustomGatesUsesSection` (`r1csfile.js:253-278`): a u32 count, then per
/// use a gate id, a signal count and that many u64 signals.
fn write_custom_gate_uses<W: Write>(
    r1cs: &R1cs,
    w: &mut JsonWriter<W>,
) -> Result<(), CeremonyError> {
    let data = r1cs.file().unique_section(S_CUSTOM_GATES_USES)?;
    let mut cur = Cursor::new(data, S_CUSTOM_GATES_USES);
    let n = cur.u32()?;
    for _ in 0..n {
        let id = cur.u32()?;
        let n_signals = cur.u32()?;
        w.begin_object()?;
        w.field_raw("id", &id.to_string())?;
        w.key("signals")?;
        w.begin_array()?;
        for _ in 0..n_signals {
            w.raw(&js_number(cur.u64()?))?;
        }
        w.end()?;
        w.end()?;
    }
    Ok(())
}
