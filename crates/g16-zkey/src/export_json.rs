//! `zkey export json` and `wtns export json`: the two binfiles this crate already reads,
//! written back out as the JSON snarkjs 0.7.6 produces for them, byte for byte.
//!
//! Neither command goes through [`crate::ProvingKey`] or [`crate::wtns::Witness`]. Both of
//! those decode into the shapes the prover wants, sort section 4 into CSR and drop
//! everything the prover does not need, while the export is a straight transcription in
//! file order. They also validate more than snarkjs does (a witness whose first element is
//! not 1, a key with no phase-2 contribution), and an export tool that refuses to show you
//! the file you are trying to debug is the wrong tool.
//!
//! # zkey
//!
//! `zkey_export_json.js:22-28` is `readZKey(name, true)`, then `delete curve`, then
//! `stringifyBigInts`. So the JSON is the object `readZKey` builds (`zkey_utils.js:341-441`),
//! in its insertion order, with every point turned into `toObject` form:
//!
//! * **Points are projective triples** of decimal strings, `[x, y, "1"]` in G1 and
//!   `[[x0, x1], [y0, y1], ["1", "0"]]` in G2, coordinates taken out of Montgomery. The
//!   point at infinity, stored as all zero bytes, becomes `["0", "1", "0"]` (and the
//!   `Fq2` equivalent), not `(0, 0)` (`wasm_curve.js:338-355`).
//! * **`C` starts with `nPublic + 1` nulls.** `readZKey` fills it from index `nPublic + 1`
//!   (`zkey_utils.js:414-420`), so the array has holes where the public signals would be
//!   and bfj writes each hole as `null`.
//! * **`ccoefs[].value` is the plain coefficient**: the stored `v*R^2` times `R^-2`
//!   (`readFr2`, `zkey_utils.js:443-446`), which is [`crate::binfile::fr_double_montgomery`]
//!   without its range check, because `F1Field.mul` reduces whatever it is handed.
//! * **`power` is `log2(domainSize)`**, snarkjs' bit-trick floor log (`misc.js:53-56`).
//!
//! The container is opened with max version 1, not the 2 the prover accepts: `readZKey`
//! passes 1 to `readBinFile` (`zkey_utils.js:342`). Sections 1 to 9 are read and 10, the
//! contributions, is not.
//!
//! # wtns
//!
//! `wtns_utils.js:read` reads `nWitness` integers of `n8` bytes each and nothing else, and
//! `cli.js:476` stringifies them. No reduction, no check of the prime, no check that
//! `w[0] == 1`. Values are decoded as plain little-endian integers of whatever width the
//! header says, so this is the one reader in the crate that does not care which curve the
//! file is for.

use std::io::{BufWriter, Write};
use std::path::Path;

use g16_field::*;
use num_bigint::BigUint;

use crate::binfile::{self, BinFile, Cursor, FQ_BYTES, G1_BYTES, G2_BYTES};
use crate::json_out::JsonWriter;
use crate::ZkeyError;

/// `readZKey` opens with `maxVersion = 1` (`zkey_utils.js:342`).
const ZKEY_EXPORT_MAX_VERSION: u32 = 1;
/// `wtns_utils.js:read` opens with `maxVersion = 2`.
const WTNS_MAX_VERSION: u32 = 2;

/// `snarkjs zkey export json <zkey> <json>`: write the groth16 key as `zkey_export_json.js`
/// does. Streams; nothing is held beyond the mapped file and one point.
pub fn zkey_export_json<W: Write>(zkey: &Path, out: W) -> Result<(), ZkeyError> {
    let file = BinFile::open(zkey, b"zkey", ZKEY_EXPORT_MAX_VERSION)?;
    let mut w = JsonWriter::new(BufWriter::new(out));
    write_zkey(&file, &mut w)?;
    w.finish()?;
    Ok(())
}

fn write_zkey<W: Write>(file: &BinFile, w: &mut JsonWriter<W>) -> Result<(), ZkeyError> {
    let mut s1 = Cursor::new(file.unique_section(1)?, 1);
    let protocol = s1.u32()?;
    if protocol != 1 {
        // plonk (2) and fflonk (10) have their own headers and their own export shape.
        return Err(ZkeyError::UnsupportedProtocol(protocol));
    }
    trailing(&s1, 1)?;

    let mut s2 = Cursor::new(file.unique_section(2)?, 2);
    let n8q = s2.u32()?;
    let q = BigUint::from_bytes_le(s2.take(n8q as usize)?);
    let n8r = s2.u32()?;
    let r = BigUint::from_bytes_le(s2.take(n8r as usize)?);
    // Every coordinate below is decoded out of BN254's Montgomery form, which only means
    // anything if the file is on BN254. snarkjs would pick another curve here.
    if n8q as usize != FQ_BYTES
        || n8r as usize != FQ_BYTES
        || q != BigUint::from(Fq::MODULUS)
        || r != BigUint::from(Fr::MODULUS)
    {
        return Err(ZkeyError::UnsupportedCurve);
    }
    let n_vars = s2.u32()?;
    let n_public = s2.u32()?;
    let domain_size = s2.u32()?;

    w.begin_object()?;
    w.field_string("protocol", "groth16")?;
    w.field_raw("n8q", &n8q.to_string())?;
    w.field_string("q", &q.to_string())?;
    w.field_raw("n8r", &n8r.to_string())?;
    w.field_string("r", &r.to_string())?;
    w.field_raw("nVars", &n_vars.to_string())?;
    w.field_raw("nPublic", &n_public.to_string())?;
    w.field_raw("domainSize", &domain_size.to_string())?;
    w.field_raw("power", &js_log2(domain_size).to_string())?;
    for (name, g2) in [
        ("vk_alpha_1", false),
        ("vk_beta_1", false),
        ("vk_beta_2", true),
        ("vk_gamma_2", true),
        ("vk_delta_1", false),
        ("vk_delta_2", true),
    ] {
        w.key(name)?;
        if g2 {
            g2_object(w, s2.take(G2_BYTES)?)?;
        } else {
            g1_object(w, s2.take(G1_BYTES)?)?;
        }
    }
    trailing(&s2, 2)?;

    let n_ic = n_public as usize + 1;
    w.key("IC")?;
    g1_section(w, file, 3, 0, n_ic)?;

    let s4 = file.unique_section(4)?;
    let mut c4 = Cursor::new(s4, 4);
    let n_coefs = c4.u32()? as usize;
    binfile::expect_records(&s4[4..], n_coefs, 12 + n8r as usize, 4)?;
    let r_inv2 = binfile::r_inv().square();
    w.key("ccoefs")?;
    w.begin_array()?;
    for _ in 0..n_coefs {
        let matrix = c4.u32()?;
        let constraint = c4.u32()?;
        let signal = c4.u32()?;
        let value = Fr::from_le_bytes_mod_order(c4.take(n8r as usize)?) * r_inv2;
        w.begin_object()?;
        w.field_raw("matrix", &matrix.to_string())?;
        w.field_raw("constraint", &constraint.to_string())?;
        w.field_raw("signal", &signal.to_string())?;
        w.field_string("value", &value.into_bigint().to_string())?;
        w.end()?;
    }
    w.end()?;

    let n_vars = n_vars as usize;
    w.key("A")?;
    g1_section(w, file, 5, 0, n_vars)?;
    w.key("B1")?;
    g1_section(w, file, 6, 0, n_vars)?;
    w.key("B2")?;
    let s7 = file.unique_section(7)?;
    binfile::expect_records(s7, n_vars, G2_BYTES, 7)?;
    w.begin_array()?;
    for p in s7.chunks_exact(G2_BYTES) {
        g2_object(w, p)?;
    }
    w.end()?;
    // `for (i = nPublic+1; i < nVars; i++) C[i] = ...`: the leading holes are real array
    // slots, and a key with `nVars <= nPublic` gets an empty array, not a negative count.
    w.key("C")?;
    g1_section(w, file, 8, n_ic, n_vars.saturating_sub(n_ic))?;
    w.key("hExps")?;
    g1_section(w, file, 9, 0, domain_size as usize)?;
    w.end()?;
    Ok(())
}

/// `snarkjs wtns export json <wtns> <json>`: every witness value as a decimal string.
pub fn wtns_export_json<W: Write>(wtns: &Path, out: W) -> Result<(), ZkeyError> {
    let mut file = BinFile::open(wtns, b"wtns", WTNS_MAX_VERSION)?;
    let mut w = JsonWriter::new(BufWriter::new(out));
    let written = write_wtns(&file, &mut w);
    // The witness is secret; see `Witness::parse`. A mapped file has no copy to scrub, so
    // this only matters to a caller that handed over bytes, but it costs nothing.
    file.scrub();
    written?;
    w.finish()?;
    Ok(())
}

fn write_wtns<W: Write>(file: &BinFile, w: &mut JsonWriter<W>) -> Result<(), ZkeyError> {
    let mut header = Cursor::new(file.unique_section(1)?, 1);
    let n8 = header.u32()? as usize;
    header.take(n8)?;
    let n_witness = header.u32()? as usize;
    trailing(&header, 1)?;
    let data = file.unique_section(2)?;
    // `endReadSection` refuses a section whose length is not exactly what was read.
    binfile::expect_records(data, n_witness, n8, 2)?;
    w.begin_array()?;
    if n8 > 0 {
        for v in data.chunks_exact(n8) {
            w.string(&BigUint::from_bytes_le(v).to_string())?;
        }
    } else {
        // `n8 == 0` reads `nWitness` empty buffers, each of which is the integer 0.
        for _ in 0..n_witness {
            w.string("0")?;
        }
    }
    w.end()?;
    Ok(())
}

/// `endReadSection` (`binfileutils.js:81-87`): a section is read to its exact end or the
/// read fails.
fn trailing(cur: &Cursor, section: u32) -> Result<(), ZkeyError> {
    if cur.remaining() != 0 {
        return Err(ZkeyError::Malformed {
            section,
            reason: format!("{} bytes left after reading the section", cur.remaining()),
        });
    }
    Ok(())
}

/// `n` G1 points of section `id` as an array, preceded by `holes` nulls.
fn g1_section<W: Write>(
    w: &mut JsonWriter<W>,
    file: &BinFile,
    id: u32,
    holes: usize,
    n: usize,
) -> Result<(), ZkeyError> {
    let data = file.unique_section(id)?;
    binfile::expect_records(data, n, G1_BYTES, id)?;
    w.begin_array()?;
    for _ in 0..holes {
        w.raw("null")?;
    }
    for p in data.chunks_exact(G1_BYTES) {
        g1_object(w, p)?;
    }
    w.end()?;
    Ok(())
}

/// One coordinate out of Montgomery, as a decimal string.
fn fq_dec(b: &[u8]) -> Result<String, ZkeyError> {
    Ok(binfile::fq(b)?.into_bigint().to_string())
}

/// `G1.toObject` (`wasm_curve.js:338-355`) of a stored LEM point.
pub(crate) fn g1_object<W: Write>(w: &mut JsonWriter<W>, b: &[u8]) -> Result<(), ZkeyError> {
    w.begin_array()?;
    if b[..G1_BYTES].iter().all(|&x| x == 0) {
        for c in ["0", "1", "0"] {
            w.string(c)?;
        }
    } else {
        w.string(&fq_dec(&b[..FQ_BYTES])?)?;
        w.string(&fq_dec(&b[FQ_BYTES..G1_BYTES])?)?;
        w.string("1")?;
    }
    w.end()?;
    Ok(())
}

/// `G2.toObject` of a stored LEM point: three `Fq2` pairs, `c0` first.
pub(crate) fn g2_object<W: Write>(w: &mut JsonWriter<W>, b: &[u8]) -> Result<(), ZkeyError> {
    w.begin_array()?;
    if b[..G2_BYTES].iter().all(|&x| x == 0) {
        for pair in [["0", "0"], ["1", "0"], ["0", "0"]] {
            w.begin_array()?;
            for c in pair {
                w.string(c)?;
            }
            w.end()?;
        }
    } else {
        for coord in 0..2 {
            w.begin_array()?;
            for half in 0..2 {
                let at = (coord * 2 + half) * FQ_BYTES;
                w.string(&fq_dec(&b[at..at + FQ_BYTES])?)?;
            }
            w.end()?;
        }
        w.begin_array()?;
        w.string("1")?;
        w.string("0")?;
        w.end()?;
    }
    w.end()?;
    Ok(())
}

/// `misc.js:53-56`, a floor log2 over 32 bits that returns 0 for 0.
fn js_log2(v: u32) -> u32 {
    v.checked_ilog2().unwrap_or(0)
}
