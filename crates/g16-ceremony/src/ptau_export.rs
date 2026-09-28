//! `powersoftau export json`, `powersoftau truncate` and `powersoftau convert`: the three
//! ptau commands that reshape a finished file rather than contribute to it, reproduced
//! byte for byte against snarkjs 0.7.6.
//!
//! # export json writes numbers that are not the points
//!
//! `powersoftau_export_json.js:22-101` reads every point as the raw 64 or 128 bytes
//! `fromRprLEM` returns and hands the whole object to `stringifyBigIntsWithField(curve.Fr,
//! ...)`, which prints any `Uint8Array` with `Fr.toString`. That is the **scalar** field's
//! `toString` applied to a base-field point: wasm's `fromMontgomery` over the first 32
//! bytes, read as an `Fr` in Montgomery form. So every "point" in the file is the single
//! number
//!
//! ```text
//!   (first 32 stored bytes, as a little-endian integer) * R^-1  mod r
//! ```
//!
//! which is neither coordinate and cannot be turned back into one. The same holds for the
//! hashes in each contribution. The output is reproduced anyway, because it is what the
//! command writes, and [`fr_to_string`] is that formula with a note on why.
//!
//! It gets stranger for a buffer shorter than 32 bytes, which only a beacon hash can be.
//! `op1` copies the buffer over `pOp1` and reduces the 32 bytes there
//! (`wasm_field1.js:89-93`), so the bytes past the hash are whatever the previous
//! conversion left: the start of that contribution's `responseHash`. [`Pop1`] carries
//! those 32 bytes through the walk in the order `stringifyBigIntsWithField` visits them.
//!
//! The Lagrange sections are exported one block per power `0 ..= power`, so section 12's
//! extra `power+1` block never appears, and the command needs a prepared file: an
//! unprepared one fails on the missing section 12. The progress lines go to stdout
//! through `console.log`, unframed, and name two sections `lAlphaTauG2` and `lBetaTauG2`
//! although both are G1.
//!
//! # truncate
//!
//! One file per power `1 .. power`, each a prefix of every point section, section 7
//! whole, and the header keeping the source's `ceremonyPower` so the result is marked as
//! truncated (`powersoftau_truncate.js:22-60`). The name is the CLI's template, the input
//! name up to its last dot plus `_`, then the power as two digits: [`truncate_template`].
//!
//! # convert
//!
//! `powersoftau_convert.js:23-66` upgrades a file prepared by a snarkjs that stopped
//! section 12 at `power`: it copies everything, and appends to section 12 one more block,
//! the Lagrange evaluations at `power+1`. It appends unconditionally. Converting a file
//! that already has the block, which is every file a current snarkjs prepares, doubles it
//! up and leaves section 12 one block longer than any reader expects. That is reproduced.
//! Like `prepare phase2` it writes the header without a `ceremonyPower`, so a truncated
//! file comes out unmarked. The block is computed by [`crate::prepare`]'s transform, the
//! same one `prepare phase2` runs.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use g16_field::{FftField, Fr, G1Affine, PrimeField};
use g16_msm::GroupFft;
use g16_zkey::binfile::r_inv;
use g16_zkey::json_out::JsonWriter;

use crate::ptau::{
    expected_section_bytes, Ptau, CONTRIBUTION_PREFIX_BYTES, PTAU_MAGIC, S_ALPHA_TAU_G1, S_BETA_G2,
    S_BETA_TAU_G1, S_CONTRIBUTIONS, S_HEADER, S_LAGRANGE_ALPHA_TAU_G1, S_LAGRANGE_BETA_TAU_G1,
    S_LAGRANGE_TAU_G1, S_LAGRANGE_TAU_G2, S_TAU_G1, S_TAU_G2,
};
use crate::snarkjs_log::SnarkjsLog;
use crate::transcript::{PARTIAL_HASH_BYTES, PTAU_PUBKEY_BYTES};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionParams, N8, SG1, SG2};

/// Every writer here emits container version 1 (`createBinFile(..., "ptau", 1, 11)`).
const PTAU_VERSION: u32 = 1;
/// Both commands write the prepared layout's 11 sections.
const PTAU_SECTIONS: u32 = crate::phase1::PTAU_PREPARED_SECTIONS;

/// `exportSection` logs every this many points (`powersoftau_export_json.js:47`).
const PROGRESS_EVERY: usize = 10_000;

/// The 32 bytes at wasm's `pOp1`, which is where `Fr.toString` reads its input from.
///
/// Every conversion in an export is at least 32 bytes wide except a short beacon hash, so
/// the state is fully overwritten before it is ever read partially: the first value in
/// traversal order is a 64-byte point. Zero is therefore a safe start.
struct Pop1([u8; N8]);

impl Pop1 {
    /// `Fr.toString(buf)` on the wasm field.
    fn stringify(&mut self, buf: &[u8]) -> String {
        let n = buf.len().min(N8);
        self.0[..n].copy_from_slice(&buf[..n]);
        fr_to_string(&self.0)
    }
}

/// `Fr.toString` of 32 bytes as ffjavascript's wasm field does it: treat them as a
/// Montgomery-form `Fr`, take them out of Montgomery (`value * R^-1 mod r`, fully reduced
/// even for an input above `r`, because the input is below `r * R`), print in decimal.
pub fn fr_to_string(b: &[u8; N8]) -> String {
    (Fr::from_le_bytes_mod_order(b) * r_inv())
        .into_bigint()
        .to_string()
}

/// `snarkjs powersoftau export json <ptau> <json>`: write the JSON, and the progress lines
/// snarkjs prints to stdout on `progress`.
pub fn ptau_export_json<W: Write, P: Write>(
    ptau: &Ptau,
    out: W,
    progress: &mut P,
) -> Result<(), CeremonyError> {
    let power = ptau.power();
    let n = 1usize << power;
    let mut pop = Pop1([0; N8]);
    let mut w = JsonWriter::new(BufWriter::new(out));
    w.begin_object()?;
    w.field_string("q", &g16_field::Fq::MODULUS.to_string())?;
    w.field_raw("power", &power.to_string())?;
    w.key("contributions")?;
    write_contributions(ptau, &mut w, &mut pop)?;

    for (key, id, count) in [
        ("tauG1", S_TAU_G1, 2 * n - 1),
        ("tauG2", S_TAU_G2, n),
        ("alphaTauG1", S_ALPHA_TAU_G1, n),
        ("betaTauG1", S_BETA_TAU_G1, n),
        ("betaG2", S_BETA_G2, 1),
    ] {
        let stride = if matches!(id, S_TAU_G2 | S_BETA_G2) {
            SG2
        } else {
            SG1
        };
        let data = ptau.section(id)?;
        // `endReadSection` refuses a section that is not exactly what was read.
        if data.len() != count * stride {
            return Err(CeremonyError::malformed(
                id,
                format!("{} bytes, expected {count} points", data.len()),
            ));
        }
        w.key(key)?;
        w.begin_array()?;
        for (i, p) in data.chunks_exact(stride).enumerate() {
            if i != 0 && i % PROGRESS_EVERY == 0 {
                writeln!(progress, "{key}: {i}")?;
            }
            w.string(&pop.stringify(p))?;
        }
        w.end()?;
    }

    for (key, log_name, id) in [
        ("lTauG1", "lTauG1", S_LAGRANGE_TAU_G1),
        ("lTauG2", "lTauG2", S_LAGRANGE_TAU_G2),
        ("lAlphaTauG1", "lAlphaTauG2", S_LAGRANGE_ALPHA_TAU_G1),
        ("lBetaTauG1", "lBetaTauG2", S_LAGRANGE_BETA_TAU_G1),
    ] {
        let stride = if id == S_LAGRANGE_TAU_G2 { SG2 } else { SG1 };
        let data = ptau.section(id)?;
        let mut points = data.chunks_exact(stride);
        w.key(key)?;
        w.begin_array()?;
        for p in 0..=power {
            writeln!(progress, "{log_name}: Power: {p}")?;
            let n_points = 1usize << p;
            w.begin_array()?;
            for i in 0..n_points {
                if i != 0 && i % PROGRESS_EVERY == 0 {
                    writeln!(progress, "{log_name}: {i}/{n_points}")?;
                }
                let point = points.next().ok_or_else(|| {
                    CeremonyError::malformed(id, format!("ends inside the 2^{p} block"))
                })?;
                w.string(&pop.stringify(point))?;
            }
            w.end()?;
        }
        w.end()?;
    }
    w.end()?;
    w.finish()?;
    Ok(())
}

/// Section 7 as `readContributions` returns it, in its key order
/// (`powersoftau_utils.js:163-240`).
fn write_contributions<W: Write>(
    ptau: &Ptau,
    w: &mut JsonWriter<W>,
    pop: &mut Pop1,
) -> Result<(), CeremonyError> {
    // Decoded once for what the raw bytes do not hold: the response hash is recomputed
    // from the partial hash and the pubkey, and the params are parsed.
    let parsed = ptau.contributions()?;
    let mut data = &ptau.section(S_CONTRIBUTIONS)?[4..];
    w.begin_array()?;
    for (i, c) in parsed.iter().enumerate() {
        let head = &data[..CONTRIBUTION_PREFIX_BYTES];
        let param_len = u32::from_le_bytes(
            head[CONTRIBUTION_PREFIX_BYTES - 4..]
                .try_into()
                .expect("4 bytes"),
        ) as usize;
        data = &data[CONTRIBUTION_PREFIX_BYTES + param_len..];

        let mut at = 0usize;
        let mut take = |n: usize| {
            let s = &head[at..at + n];
            at += n;
            s
        };
        w.begin_object()?;
        for (key, len) in [
            ("tauG1", SG1),
            ("tauG2", SG2),
            ("alphaG1", SG1),
            ("betaG1", SG1),
            ("betaG2", SG2),
        ] {
            w.field_string(key, &pop.stringify(take(len)))?;
        }
        let key = take(PTAU_PUBKEY_BYTES);
        let partial_hash = take(PARTIAL_HASH_BYTES);
        let next_challenge = take(64);

        // `fromPtauPubKeyRpr` reads six G1 points then three G2, but builds the object as
        // tau, alpha, beta, each `g1_s, g1_sx, g2_spx`; stringification follows the object.
        w.key("key")?;
        w.begin_object()?;
        for (k, name) in ["tau", "alpha", "beta"].iter().enumerate() {
            w.key(name)?;
            w.begin_object()?;
            w.field_string("g1_s", &pop.stringify(&key[2 * k * SG1..]))?;
            w.field_string("g1_sx", &pop.stringify(&key[(2 * k + 1) * SG1..]))?;
            w.field_string("g2_spx", &pop.stringify(&key[6 * SG1 + k * SG2..]))?;
            w.end()?;
        }
        w.end()?;

        w.field_string("partialHash", &pop.stringify(partial_hash))?;
        w.field_string("nextChallenge", &pop.stringify(next_challenge))?;
        w.field_raw("type", &c.kind.as_u32().to_string())?;
        w.field_string("responseHash", &pop.stringify(&c.response_hash()?))?;
        let ContributionParams {
            name,
            num_iterations_exp,
            beacon_hash,
        } = &c.params;
        if let Some(name) = name {
            w.field_string("name", name)?;
        }
        if let Some(exp) = num_iterations_exp {
            w.field_raw("numIterationsExp", &exp.to_string())?;
        }
        if let Some(hash) = beacon_hash {
            w.field_string("beaconHash", &pop.stringify(hash))?;
        }
        w.field_raw("id", &(i + 1).to_string())?;
        w.end()?;
    }
    w.end()?;
    Ok(())
}

/// The name template `cli.js:866-882` builds for `powersoftau truncate`: everything
/// before the last `.` of the input name, then `_`. A name without a dot is consumed
/// whole by that loop, leaving just `_`.
pub fn truncate_template(ptau_name: &str) -> String {
    match ptau_name.rfind('.') {
        Some(dot) => format!("{}_", &ptau_name[..dot]),
        None => "_".to_string(),
    }
}

/// `snarkjs powersoftau truncate <ptau>`: write `<template><pp>.ptau` for every power
/// `1 .. power`, `pp` at least two digits, and return the paths in order.
pub fn ptau_truncate<L: SnarkjsLog + ?Sized>(
    ptau: &Ptau,
    template: &str,
    log: &mut L,
) -> Result<Vec<PathBuf>, CeremonyError> {
    let h = *ptau.header();
    let mut written = Vec::new();
    for p in 1..h.power {
        let sp = format!("{p:02}");
        log.debug(&format!("Writing Power: {sp}"))?;
        let path = PathBuf::from(format!("{template}{sp}.ptau"));
        let mut out = BinFileWriter::create(&path, PTAU_MAGIC, PTAU_VERSION, PTAU_SECTIONS)?;
        write_header(&mut out, p, h.ceremony_power)?;
        for id in [
            S_TAU_G1,
            S_TAU_G2,
            S_ALPHA_TAU_G1,
            S_BETA_TAU_G1,
            S_BETA_G2,
            S_CONTRIBUTIONS,
            S_LAGRANGE_TAU_G1,
            S_LAGRANGE_TAU_G2,
            S_LAGRANGE_ALPHA_TAU_G1,
            S_LAGRANGE_BETA_TAU_G1,
        ] {
            let body = ptau.section(id)?;
            // Section 7 goes whole; every other section's prefix is exactly the full size
            // of that section at power `p` (`powersoftau_truncate.js:48-57`).
            let len = match expected_section_bytes(id, p) {
                Some(len) => len as usize,
                None => body.len(),
            };
            let prefix = body.get(..len).ok_or_else(|| {
                CeremonyError::malformed(
                    id,
                    format!("{} bytes, too short for power {p}", body.len()),
                )
            })?;
            out.write_section_verbatim(id, prefix)?;
        }
        out.finish()?;
        written.push(path);
    }
    Ok(written)
}

/// `snarkjs powersoftau convert <old.ptau> <new.ptau>`: copy the file and append the
/// `power+1` Lagrange block to section 12, with `fft` doing the transform.
pub fn ptau_convert<L: SnarkjsLog + ?Sized>(
    ptau: &Ptau,
    new_ptau: &Path,
    fft: &dyn GroupFft,
    log: &mut L,
) -> Result<(), CeremonyError> {
    let power = ptau.power();
    // The appended block is `2^(power+1)` points, and the transform stops one past the
    // two-adicity, so this is `prepare phase2`'s own ceiling.
    if power > Fr::TWO_ADICITY {
        return Err(CeremonyError::CircuitTooBig(power));
    }
    let mut out = BinFileWriter::create(new_ptau, PTAU_MAGIC, PTAU_VERSION, PTAU_SECTIONS)?;
    write_header(&mut out, power, power)?;
    for id in [
        S_TAU_G1,
        S_TAU_G2,
        S_ALPHA_TAU_G1,
        S_BETA_TAU_G1,
        S_BETA_G2,
        S_CONTRIBUTIONS,
    ] {
        out.write_section_verbatim(id, ptau.section(id)?)?;
    }

    log.debug("Starting section: tauG1")?;
    out.start_section(S_LAGRANGE_TAU_G1)?;
    out.write_bytes(ptau.section(S_LAGRANGE_TAU_G1)?)?;
    // Section 2 holds `2n - 1` points and the block wants `2n`, so the last input is the
    // point at infinity, exactly as in `prepare phase2`.
    let m = 1usize << (power + 1);
    let mut input = ptau.g1_points(S_TAU_G1, 0, m - 1)?;
    input.push(G1Affine::identity());
    let block = crate::prepare::lagrange_evaluations_g1(&input, fft)?;
    out.write_g1_slice(&block)?;
    out.end_section()?;

    for id in [
        S_LAGRANGE_TAU_G2,
        S_LAGRANGE_ALPHA_TAU_G1,
        S_LAGRANGE_BETA_TAU_G1,
    ] {
        out.write_section_verbatim(id, ptau.section(id)?)?;
    }
    out.finish()
}

/// `writePTauHeader` (`powersoftau_utils.js:26-50`), including its `ceremonyPower || power`.
fn write_header(
    out: &mut BinFileWriter,
    power: u32,
    ceremony_power: u32,
) -> Result<(), CeremonyError> {
    out.start_section(S_HEADER)?;
    out.write_prime(&crate::q_le())?;
    out.write_u32(power)?;
    out.write_u32(if ceremony_power == 0 {
        power
    } else {
        ceremony_power
    })?;
    out.end_section()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator's `x` is stored as `R mod q`, and snarkjs 0.7.6 exports it as this
    /// number: checked against the first `tauG1` entry of a fresh power-3 file.
    #[test]
    fn fr_to_string_is_scalar_from_montgomery() {
        let x = crate::write::fq_lem(&g16_field::Fq::from(1u64));
        assert_eq!(
            fr_to_string(&x),
            "19251647757787308156787743079029887954678750029819827393622431040979082323348"
        );
    }

    #[test]
    fn template_follows_the_cli_loop() {
        assert_eq!(
            truncate_template("pot/ppot_0080_20.ptau"),
            "pot/ppot_0080_20_"
        );
        assert_eq!(truncate_template("a.b.ptau"), "a.b_");
        assert_eq!(truncate_template("noext"), "_");
    }
}
