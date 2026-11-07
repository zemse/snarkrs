//! A `.zkey` to gnark's `ProvingKey.WriteDump` and `VerifyingKey.WriteTo`.
//!
//! Section by section: 5, 6 and 7 are `G1.A`, `G1.B` and `G2.B`, less the points at
//! infinity, which gnark leaves out and flags in `InfinityA` and `InfinityB`; 8 is `G1.K`,
//! the private wires; 3 is the verifying key's `G1.K`; 9 becomes `G1.Z` through
//! [`crate::basis::gnark_z_from_zkey_h`], exactly. Section 4 has no place in gnark's key,
//! which holds no constraints: a prover of the result takes A.w and B.w per row, which
//! [`crate::zkey_rows`] computes from the zkey and a `.wtns`. A gnark R1CS, which gnark's
//! own prover would need to solve the circuit, is not written.
//!
//! The rows a zkey adds after the constraints, one `A = w_s` row per public signal and the
//! constant wire, come along unchanged: they are rows of section 4 like any other, so
//! [`crate::zkey_rows`] includes them and the A query already pays for them.

use crate::{mismatch, ConvertError};
use rayon::prelude::*;
use snarkrs_ceremony::write::{g1_lem, g2_lem};
use snarkrs_field::*;
use snarkrs_formats::{HBasis, ProvingKey};
use snarkrs_gnark::pk::DumpHeader;
use snarkrs_gnark::{raw_g1, raw_g2, GnarkProvingKey, GnarkVerifyingKey, DUMP_MARKER};
use snarkrs_msm::GroupFft;
use std::io::Write;

/// gnark-crypto's `GeneratorFullMultiplicativeGroup` for BN254's Fr, the
/// `FrMultiplicativeGen` every domain records.
const FR_MULTIPLICATIVE_GEN: u64 = 5;

/// Points encoded per buffered write of a dumped slice.
const WRITE_BATCH: usize = 1 << 16;

/// gnark's proving and verifying keys for the key `pk` holds.
///
/// Refused: a key whose H query is not snarkjs' (one read from gnark already has gnark's),
/// a domain gnark cannot hold, and a wire whose section 6 and section 7 points disagree on
/// being infinity, which no setup produces and gnark's one `InfinityB` cannot express.
pub fn gnark_keys(
    pk: &ProvingKey,
    fft: &dyn GroupFft,
) -> Result<(GnarkProvingKey, GnarkVerifyingKey), ConvertError> {
    if pk.h_basis != HBasis::SnarkjsOddCoset {
        return Err(mismatch(
            "zkey",
            "the H query is already gnark's; this key came from gnark",
        ));
    }
    if let Some(i) = (0..pk.n_vars)
        .into_par_iter()
        .find_first(|&i| pk.b_g1_query[i].infinity != pk.b_g2_query[i].infinity)
    {
        return Err(mismatch(
            "zkey",
            format!("wire {i} is infinity in one of sections 6 and 7 and not the other"),
        ));
    }
    let z = crate::basis::gnark_z_from_zkey_h(&pk.h_query, fft)?;
    let count = |q: &[G1Affine]| q.par_iter().filter(|p| p.infinity).count();
    let header = DumpHeader {
        domain_size: pk.domain_size,
        n_wires: pk.n_vars,
        nb_infinity_a: count(&pk.a_query),
        nb_infinity_b: count(&pk.b_g1_query),
        g1_a: pk.n_vars - count(&pk.a_query),
        g1_b: pk.n_vars - count(&pk.b_g1_query),
        g1_z: z.len(),
        g1_k: pk.l_query.len(),
        g2_b: pk.n_vars - count(&pk.b_g1_query),
    };
    let gpk = GnarkProvingKey {
        header,
        alpha_g1: pk.alpha_g1,
        beta_g1: pk.beta_g1,
        delta_g1: pk.delta_g1,
        beta_g2: pk.beta_g2,
        delta_g2: pk.delta_g2,
        a: pk.a_query.clone(),
        b_g1: pk.b_g1_query.clone(),
        b_g2: pk.b_g2_query.clone(),
        z,
        k: pk.l_query.clone(),
    };
    let gvk = GnarkVerifyingKey {
        alpha_g1: pk.alpha_g1,
        beta_g1: pk.beta_g1,
        beta_g2: pk.beta_g2,
        gamma_g2: pk.vk.gamma_g2,
        delta_g1: pk.delta_g1,
        delta_g2: pk.delta_g2,
        k: pk.vk.ic.clone(),
    };
    Ok((gpk, gvk))
}

/// `pk` as `ProvingKey.WriteDump` writes it, which gnark's `ReadDump` and
/// [`GnarkProvingKey::read_dump`] read. A wire is flagged in `InfinityA` (`InfinityB`)
/// exactly when its `G1.A` (`G1.B`) point is infinity, which is how gnark's setup flags
/// them: a zero scalar, and no other scalar, gives that point.
pub fn write_dump(pk: &GnarkProvingKey, mut out: impl Write) -> Result<(), ConvertError> {
    let n = pk.header.domain_size;
    let domain = Domain::new(n).map_err(|e| mismatch("domain", e.to_string()))?;
    if domain.size() != n || pk.z.len() + 1 != n {
        return Err(mismatch(
            "proving key",
            format!("{} Z points on a domain of {n}", pk.z.len()),
        ));
    }
    let n_wires = pk.a.len();
    if pk.b_g1.len() != n_wires || pk.b_g2.len() != n_wires {
        return Err(mismatch(
            "proving key",
            "A and B are not one point per wire",
        ));
    }
    out.write_all(&DUMP_MARKER)?;

    // Domain.WriteTo: the cardinality, five elements big-endian, and withPrecompute, which
    // gnark's setup leaves on.
    out.write_all(&(n as u64).to_be_bytes())?;
    let gen = Fr::from(FR_MULTIPLICATIVE_GEN);
    for x in [
        domain.size_inv(),
        domain.group_gen(),
        domain.group_gen_inv(),
        gen,
        gen.inverse().expect("5 is a unit mod r"),
    ] {
        out.write_all(&x.into_bigint().to_bytes_be())?;
    }
    out.write_all(&[1])?;

    for p in [&pk.alpha_g1, &pk.beta_g1, &pk.delta_g1] {
        out.write_all(&raw_g1(p))?;
    }
    for p in [&pk.beta_g2, &pk.delta_g2] {
        out.write_all(&raw_g2(p))?;
    }
    let flags_a: Vec<u8> = pk.a.par_iter().map(|p| u8::from(p.infinity)).collect();
    let flags_b: Vec<u8> = pk.b_g1.par_iter().map(|p| u8::from(p.infinity)).collect();
    if pk
        .b_g2
        .par_iter()
        .zip(&flags_b)
        .any(|(p, &f)| p.infinity != (f == 1))
    {
        return Err(mismatch(
            "proving key",
            "G1.B and G2.B disagree on which wires are infinity",
        ));
    }
    let ones = |f: &[u8]| f.par_iter().filter(|&&f| f == 1).count() as u64;
    out.write_all(&(n_wires as u64).to_be_bytes())?;
    out.write_all(&ones(&flags_a).to_be_bytes())?;
    out.write_all(&ones(&flags_b).to_be_bytes())?;
    out.write_all(&flags_a)?;
    out.write_all(&flags_b)?;
    // No commitment keys.
    out.write_all(&0u32.to_be_bytes())?;

    let finite_a: Vec<&G1Affine> = pk.a.iter().filter(|p| !p.infinity).collect();
    let finite_b: Vec<&G1Affine> = pk.b_g1.iter().filter(|p| !p.infinity).collect();
    let finite_b2: Vec<&G2Affine> = pk.b_g2.iter().filter(|p| !p.infinity).collect();
    dump_slice(&mut out, &finite_a, g1_lem)?;
    dump_slice(&mut out, &finite_b, g1_lem)?;
    let z: Vec<&G1Affine> = pk.z.iter().collect();
    dump_slice(&mut out, &z, g1_lem)?;
    let k: Vec<&G1Affine> = pk.k.iter().collect();
    dump_slice(&mut out, &k, g1_lem)?;
    dump_slice(&mut out, &finite_b2, g2_lem)?;
    out.flush()?;
    Ok(())
}

/// `unsafe.WriteSlice`: a `u64` LE count and the points' memory, which for gnark-crypto's
/// affine points is the Montgomery limbs little-endian, `A0` before `A1`: a zkey's point
/// layout exactly, infinity `(0, 0)` in both.
fn dump_slice<T: Sync, const N: usize>(
    out: &mut impl Write,
    points: &[&T],
    enc: impl Fn(&T) -> [u8; N] + Sync,
) -> Result<(), ConvertError> {
    out.write_all(&(points.len() as u64).to_le_bytes())?;
    for batch in points.chunks(WRITE_BATCH) {
        let bytes: Vec<u8> = batch.par_iter().flat_map_iter(|p| enc(p)).collect();
        out.write_all(&bytes)?;
    }
    Ok(())
}
