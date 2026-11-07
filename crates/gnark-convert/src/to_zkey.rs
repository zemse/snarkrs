//! gnark's `ProvingKey.WriteDump`, `VerifyingKey.WriteTo` and R1CS to a `.zkey`.
//!
//! Section by section, in the order `zkey new` writes them (1, 2, 4, 3, 9, 8, 5, 6, 7, 10):
//!
//! * 2: `nVars` is gnark's `nbWires`, `nPublic` its public wires less the constant one, the
//!   domain gnark's; the six points from the two keys, which must agree.
//! * 4: every R1C's `L` terms as matrix A and `R` terms as matrix B, constraint by
//!   constraint, double Montgomery. gnark's `O` reaches a prover only through the C query.
//! * 3: the verifying key's `G1.K`. 8: the proving key's `G1.K`, the private wires.
//! * 5, 6, 7: `G1.A`, `G1.B`, `G2.B`, back to one point per wire.
//! * 9: [`crate::basis::zkey_h_from_gnark_z`], see the crate docs for why it proves exactly.
//! * 10: a zero `csHash` and no contributions. No transcript produced this key.
//!
//! What a zkey from `zkey new` has and this one does not: the `nPublic + 1` rows snarkjs
//! appends after the constraints, `A = w_s` for the constant wire and each public signal,
//! with their `L_{nConstraints + s}(tau)` terms in the A query and IC. gnark adds no such
//! rows, and its A query and `G1.K` do not include them. Section 4 here holds gnark's
//! constraints and nothing else, which is what this A query is the commitment to, so the
//! key is consistent. Neither prover needs the rows: snarkjs' `groth16 prove`, rapidsnark
//! and snarkrs all evaluate section 4 as it is and build C as `A * B` per row, and none of
//! them reads `nConstraints`, which the zkey does not even store. The rows are snarkjs'
//! setup's own guard that every public input moves the A query, and a gnark key's security
//! rests on gnark's setup, as it did before the conversion.

use crate::{mismatch, ConvertError};
use rayon::prelude::*;
use snarkrs_ceremony::setup::{
    S_A, S_B1, S_B2, S_C, S_COEFFS, S_H, S_HEADER, S_IC, S_MPC_PARAMS, S_PROTOCOL, ZKEY_MAGIC,
    ZKEY_SECTIONS, ZKEY_VERSION,
};
use snarkrs_ceremony::write::{fr_double_montgomery, BinFileWriter};
use snarkrs_ceremony::Groth16Header;
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey};
use snarkrs_gnark_solver::{GnarkR1cs, Term};
use snarkrs_msm::GroupFft;
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};

/// Section 4's stride: three `u32` indices and a scalar.
const RECORD_BYTES: usize = 12 + 32;

/// Instructions one task encodes. Section 4 is written in this many-instruction runs, a
/// pool's worth at a time, so the whole section is never held at once.
const OPS_PER_TASK: usize = 1 << 14;

/// What [`write_zkey`] wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkeyReport {
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
    pub n_constraints: usize,
    /// Section 4 records of matrix A, then of B.
    pub n_coefficients: [usize; 2],
}

/// Write the zkey for gnark's `pk` and `vk` to `out`, with `r1cs` the circuit `pk` was set
/// up for. Staged beside `out` and renamed over it, so an error leaves `out` as it was.
///
/// The three must describe one circuit: the keys' common points agree, the R1CS has the
/// keys' wires, public wires and domain, and a wire with no `L` (`R`) term is exactly a
/// wire whose `G1.A` (`G1.B`) point is infinity. The last is the check that catches an
/// R1CS from another circuit of the same shape, since a key's points are otherwise
/// random-looking group elements nothing here can test against the matrices.
pub fn write_zkey(
    pk: &GnarkProvingKey,
    vk: &GnarkVerifyingKey,
    r1cs: &GnarkR1cs,
    out: &Path,
    fft: &dyn GroupFft,
) -> Result<ZkeyReport, ConvertError> {
    snarkrs_formats::stage::staged_all(&[out], |stages| {
        write_zkey_unstaged(pk, vk, r1cs, &stages[0], fft)
    })
}

/// [`write_zkey`] straight into `out`, which must be seekable, with no stage: an error
/// leaves whatever was written. For `/dev/null`, when only the cost of a conversion is
/// wanted, or a caller with staging of its own.
pub fn write_zkey_unstaged(
    pk: &GnarkProvingKey,
    vk: &GnarkVerifyingKey,
    r1cs: &GnarkR1cs,
    out: &Path,
    fft: &dyn GroupFft,
) -> Result<ZkeyReport, ConvertError> {
    let report = check(pk, vk, r1cs)?;
    // Before the file exists: it is the one step that can fail on arithmetic grounds.
    let h = crate::basis::zkey_h_from_gnark_z(&pk.z, fft)?;
    let header = Groth16Header {
        n_vars: report.n_vars as u32,
        n_public: report.n_public as u32,
        domain_size: report.domain_size as u32,
        alpha_g1: pk.alpha_g1,
        beta_g1: pk.beta_g1,
        beta_g2: pk.beta_g2,
        gamma_g2: vk.gamma_g2,
        delta_g1: pk.delta_g1,
        delta_g2: pk.delta_g2,
    };
    let mut w = BinFileWriter::create(out, ZKEY_MAGIC, ZKEY_VERSION, ZKEY_SECTIONS)?;
    w.start_section(S_PROTOCOL)?;
    w.write_u32(snarkrs_ceremony::PROTOCOL_GROTH16)?;
    w.end_section()?;
    w.start_section(S_HEADER)?;
    header.write(&mut w)?;
    w.end_section()?;
    write_coefficients(&mut w, r1cs, report.n_coefficients)?;
    for (id, points) in [
        (S_IC, &vk.k),
        (S_H, &h),
        (S_C, &pk.k),
        (S_A, &pk.a),
        (S_B1, &pk.b_g1),
    ] {
        w.start_section(id)?;
        w.write_g1_slice(points)?;
        w.end_section()?;
    }
    w.start_section(S_B2)?;
    w.write_g2_slice(&pk.b_g2)?;
    w.end_section()?;
    w.start_section(S_MPC_PARAMS)?;
    w.write_bytes(&[0u8; 64])?;
    w.write_u32(0)?;
    w.end_section()?;
    w.finish()?;
    Ok(report)
}

/// Every refusal [`write_zkey`] makes, and the section 4 record counts.
fn check(
    pk: &GnarkProvingKey,
    vk: &GnarkVerifyingKey,
    r1cs: &GnarkR1cs,
) -> Result<ZkeyReport, ConvertError> {
    for (same, what) in [
        (vk.alpha_g1 == pk.alpha_g1, "G1.Alpha"),
        (vk.beta_g1 == pk.beta_g1, "G1.Beta"),
        (vk.delta_g1 == pk.delta_g1, "G1.Delta"),
        (vk.beta_g2 == pk.beta_g2, "G2.Beta"),
        (vk.delta_g2 == pk.delta_g2, "G2.Delta"),
    ] {
        if !same {
            return Err(mismatch(
                "verifying key",
                format!("{what} differs from the proving key's"),
            ));
        }
    }
    let n_wires = pk.header.n_wires;
    let public_wires = n_wires - pk.k.len();
    if vk.k.len() != public_wires || r1cs.nb_public() != public_wires {
        return Err(mismatch(
            "public wires",
            format!(
                "the proving key leaves {public_wires}, the verifying key has {} and the R1CS {}",
                vk.k.len(),
                r1cs.nb_public()
            ),
        ));
    }
    if r1cs.nb_wires() != n_wires {
        return Err(mismatch(
            "R1CS",
            format!("{} wires, the key has {n_wires}", r1cs.nb_wires()),
        ));
    }
    // gnark sets up on `fft.NewDomain(nbConstraints)`, the next power of two.
    let n = pk.header.domain_size;
    let nc = r1cs.nb_constraints();
    if nc.max(1).next_power_of_two() != n {
        return Err(mismatch(
            "R1CS",
            format!("{nc} constraints, but the key's domain is {n}"),
        ));
    }

    // Which wires each matrix touches with a nonzero coefficient, and how many records.
    // Bit `m` of `used[w]` is set when wire `w` has a term in matrix `m`.
    let used: Vec<AtomicU8> = (0..n_wires)
        .into_par_iter()
        .map(|_| AtomicU8::new(0))
        .collect();
    let counts = (0..r1cs.nb_instructions())
        .into_par_iter()
        .step_by(OPS_PER_TASK)
        .map(|start| {
            let mut counts = [0usize; 2];
            let end = (start + OPS_PER_TASK).min(r1cs.nb_instructions());
            r1cs.for_each_r1c(start..end, |_, l, r, _| {
                for (m, terms) in [l, r].into_iter().enumerate() {
                    for t in terms.iter().filter(|t| t.coefficient != 0) {
                        used[t.wire as usize].fetch_or(1 << m, Ordering::Relaxed);
                        counts[m] += 1;
                    }
                }
            });
            counts
        })
        .reduce(|| [0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    for (m, (query, name)) in [(&pk.a, "G1.A"), (&pk.b_g1, "G1.B")]
        .into_iter()
        .enumerate()
    {
        let uses = |i: usize| used[i].load(Ordering::Relaxed) & (1 << m) != 0;
        if let Some(i) = (0..n_wires)
            .into_par_iter()
            .find_first(|&i| uses(i) == query[i].infinity)
        {
            return Err(mismatch(
                "R1CS",
                format!(
                    "wire {i} {} in matrix {} but its {name} point is {}infinity; the R1CS is \
                     not the one this key was set up for",
                    if uses(i) {
                        "appears"
                    } else {
                        "does not appear"
                    },
                    ["A", "B"][m],
                    if query[i].infinity { "" } else { "not " },
                ),
            ));
        }
    }
    if counts[0] + counts[1] > u32::MAX as usize {
        return Err(mismatch(
            "R1CS",
            format!(
                "{} terms do not fit section 4's u32 count",
                counts[0] + counts[1]
            ),
        ));
    }
    Ok(ZkeyReport {
        n_vars: n_wires,
        n_public: public_wires - 1,
        domain_size: n,
        n_constraints: nc,
        n_coefficients: counts,
    })
}

/// Section 4: the record count, then each R1C's A records and its B records, a run of
/// instructions per task and a pool's worth of runs per write.
fn write_coefficients(
    w: &mut BinFileWriter,
    r1cs: &GnarkR1cs,
    counts: [usize; 2],
) -> Result<(), ConvertError> {
    let values: Vec<[u8; 32]> = r1cs
        .coefficients()
        .par_iter()
        .map(fr_double_montgomery)
        .collect();
    let record = |out: &mut Vec<u8>, m: u32, c: usize, t: &Term| {
        out.extend_from_slice(&m.to_le_bytes());
        out.extend_from_slice(&(c as u32).to_le_bytes());
        out.extend_from_slice(&t.wire.to_le_bytes());
        out.extend_from_slice(&values[t.coefficient as usize]);
    };
    w.start_section(S_COEFFS)?;
    w.write_u32((counts[0] + counts[1]) as u32)?;
    let n_ops = r1cs.nb_instructions();
    let per_write = OPS_PER_TASK * 4 * rayon::current_num_threads();
    let mut written = 0usize;
    for at in (0..n_ops).step_by(per_write) {
        let end = (at + per_write).min(n_ops);
        let runs: Vec<Vec<u8>> = (at..end)
            .into_par_iter()
            .step_by(OPS_PER_TASK)
            .map(|start| {
                let mut out = Vec::new();
                r1cs.for_each_r1c(start..(start + OPS_PER_TASK).min(end), |c, l, r, _| {
                    for (m, terms) in [l, r].into_iter().enumerate() {
                        for t in terms.iter().filter(|t| t.coefficient != 0) {
                            record(&mut out, m as u32, c, t);
                        }
                    }
                });
                out
            })
            .collect();
        for run in &runs {
            written += run.len() / RECORD_BYTES;
            w.write_bytes(run)?;
        }
    }
    // The two passes walk the same instructions, so this holds unless the R1CS changed
    // under them.
    assert_eq!(
        written,
        counts[0] + counts[1],
        "section 4 records match the count"
    );
    w.end_section()?;
    Ok(())
}
