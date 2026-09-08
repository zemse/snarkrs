//! `groth16 setup`: an `.r1cs` and a prepared `.ptau` in, an initial `.zkey` out.
//!
//! `zkey_new.js:36-171`. Four things about this command surprise people who have only
//! read the paper:
//!
//! * **The initial key has `gamma = delta = 1` and setup performs no division.**
//!   `zkey_new.js:127-129` writes the plain generators as `gamma2`, `delta1` and `delta2`,
//!   so sections 3 and 8 are undivided. `zkey contribute` is what introduces a delta at
//!   all (`zkey_contribute.js:63-64, :90-92`).
//! * **This is not one big MSM.** `composeAndWritePoints` produces one output point per
//!   signal, dispatching a single scalar multiplication when a signal appears once and a
//!   multiexp when it appears more than once, then one batch-to-affine over the chunk
//!   (`zkey_new.js:459-491`). It is MSM-bound overall, which is why the backend is a
//!   parameter, but the shape is per-signal.
//! * **Section 9 involves no arithmetic at all.** `writeHs` is a strided gather: read the
//!   `2*domainSize` Lagrange block of ptau section 12 and take every odd element
//!   (`zkey_new.js:186-191`). The `cirPower == Fr::TWO_ADICITY` case is different because
//!   that block came out of the coset branch and is laid out as two contiguous halves
//!   rather than interleaved, so it reads the second half in one go (`:192-194`).
//! * **The section write order is `1, 2, 4, 3, 9, 8, 5, 6, 7, 10`**, not id order
//!   (`zkey_new.js:76-168`). It changes both the file layout and `csHash`, so getting it
//!   wrong is not cosmetic.
//!
//! The one shipped bug that has to be reproduced rather than fixed lives in
//! [`hash_h_points`].

use std::ops::Mul;
use std::path::Path;

use g16_field::{
    AffineRepr, CurveGroup, FftField, Fq, Fr, G1Affine, G1Projective, G2Affine, G2Projective, One,
    PrimeField, Zero,
};
use g16_msm::MsmBackend;
use g16_zkey::binfile;
use rayon::prelude::*;

use crate::ptau::{self, Ptau};
use crate::r1cs::{Matrix, R1cs};
use crate::transcript::{Digest, Transcript};
use crate::write::BinFileWriter;
use crate::{CeremonyError, Groth16Header, N8, SG1};

/// zkey section ids, in id order rather than write order.
pub const S_PROTOCOL: u32 = 1;
pub const S_HEADER: u32 = 2;
pub const S_IC: u32 = 3;
pub const S_COEFFS: u32 = 4;
pub const S_A: u32 = 5;
pub const S_B1: u32 = 6;
pub const S_B2: u32 = 7;
pub const S_C: u32 = 8;
pub const S_H: u32 = 9;
pub const S_MPC_PARAMS: u32 = 10;

/// Sections a fresh zkey holds, and the number `createBinFile` declares
/// (`zkey_new.js:49`).
pub const ZKEY_SECTIONS: u32 = 10;

/// `.zkey` magic, and the version this crate writes.
pub const ZKEY_MAGIC: &[u8; 4] = b"zkey";
pub const ZKEY_VERSION: u32 = 1;
/// Both readers in `zkey_verify_frominit.js` pass a max version of 2 while every writer
/// emits 1, so 1 and 2 are both acceptable on read.
pub const ZKEY_MAX_VERSION: u32 = 2;

/// Section 4 record stride on BN254: three u32 indices then one 32-byte scalar.
pub const COEF_RECORD_BYTES: usize = 12 + crate::N8;

/// Section 4 records encoded per `write_bytes` call. Only a buffering choice: the bytes
/// are identical whatever the split, and 8192 records is 352 KB.
const COEF_BATCH: usize = 8192;

/// Points per `hashHPoints` chunk, snarkjs' `CHUNK_SIZE` at `zkey_new.js:505`. Unlike the
/// coefficient batch this one IS load bearing, because the missing `- i` at `:511` makes
/// the number of points hashed a function of it.
const H_CHUNK: usize = 1 << 14;

/// Terms in one accumulator slot below which a Pippenger multiexp is not worth its bucket
/// array, and the slot is summed with plain scalar multiplications instead.
///
/// snarkjs splits at 2 (`zkey_new.js:467-481`), but only because its alternative is a
/// round trip into a wasm worker either way. The sum is the same however it is computed,
/// and slots here are tiny: on `anon-aadhaar` the mean A slot holds three terms, so
/// dispatching a full MSM per slot would be several million bucket allocations.
const MULTIEXP_MIN_TERMS: usize = 32;

/// What a setup produced, for the CLI to print and a test to assert on.
#[derive(Clone, Debug)]
pub struct SetupReport {
    pub n_vars: usize,
    /// `nOutputs + nPubInputs`, the ONE signal excluded.
    pub n_public: usize,
    pub n_constraints: usize,
    pub domain_size: usize,
    pub cir_power: u32,
    /// Section 4's record count.
    pub n_coefs: usize,
    /// The circuit hash written into section 10, and the value every later contribution
    /// transcript is seeded with.
    pub cs_hash: Digest,
}

/// `zkey_new.js:59` with the floor-log2 of `misc.js:53`:
/// `floor(log2(nConstraints + nPubInputs + nOutputs)) + 1`.
///
/// Note the consequence, which is not an off-by-one: for a total that is already a power
/// of two the domain still doubles, so `2^cirPower` is always strictly greater than
/// `nConstraints + nPublic`. That is precisely what makes the synthetic `nConstraints + s`
/// row indices land inside the `domainSize`-element Lagrange window.
pub fn circuit_power(n_constraints: usize, n_public: usize) -> u32 {
    crate::floor_log2((n_constraints + n_public) as u64) + 1
}

/// Run the whole setup.
///
/// `msm` is threaded through rather than hardcoded because this command is MSM-bound and
/// is the first thing in the ceremony worth putting on a GPU. Nothing else about it wants
/// a device.
pub fn setup(
    r1cs_path: &Path,
    ptau_path: &Path,
    out_path: &Path,
    msm: &dyn MsmBackend,
) -> Result<SetupReport, CeremonyError> {
    let ptau = Ptau::open(ptau_path)?;
    let r1cs = R1cs::open(r1cs_path)?;

    let n_vars = r1cs.header().n_vars as usize;
    let n_public = r1cs.header().n_public();
    let n_constraints = r1cs.header().n_constraints as usize;
    let cir_power = circuit_power(n_constraints, n_public);
    let domain_size = 1usize << cir_power;

    // Order matters only for the message a caller sees: snarkjs tests the power first
    // (`zkey_new.js:61`) and only then section 12 (`:66`).
    if cir_power > ptau.power() {
        return Err(CeremonyError::PtauTooSmall {
            needed: cir_power,
            have: ptau.power(),
        });
    }
    if !ptau.is_prepared() {
        return Err(CeremonyError::NotPrepared);
    }

    // `vk_alpha_1` and `vk_beta_1` are element 0 of the alpha and beta ptau sections, that
    // is `alpha*tau^0*G1` and `beta*tau^0*G1` (`zkey_new.js:101,107,113`). Reading them
    // from the Lagrange sections instead would give `alpha*L_0(tau)*G1`, which for a
    // one-element domain is the same point and for every real domain is not.
    let alpha_g1 = ptau.g1_points(ptau::S_ALPHA_TAU_G1, 0, 1)?[0];
    let beta_g1 = ptau.g1_points(ptau::S_BETA_TAU_G1, 0, 1)?[0];
    let beta_g2 = ptau.g2_points(ptau::S_BETA_G2, 0, 1)?[0];

    let header = Groth16Header {
        n_vars: n_vars as u32,
        n_public: n_public as u32,
        domain_size: domain_size as u32,
        alpha_g1,
        beta_g1,
        beta_g2,
        gamma_g2: G2Affine::generator(),
        delta_g1: G1Affine::generator(),
        delta_g2: G2Affine::generator(),
    };

    let mut cs = Transcript::new();
    let mut w = BinFileWriter::create(out_path, ZKEY_MAGIC, ZKEY_VERSION, ZKEY_SECTIONS)?;

    w.start_section(S_PROTOCOL)?;
    w.write_u32(crate::PROTOCOL_GROTH16)?;
    w.end_section()?;

    w.start_section(S_HEADER)?;
    header.write(&mut w)?;
    w.end_section()?;
    // The six header points, uncompressed, in write order (`zkey_new.js:104,110,116,
    // :130-132`). The scalars beside them on disk are never hashed.
    cs.update_g1(&header.alpha_g1);
    cs.update_g1(&header.beta_g1);
    cs.update_g2(&header.beta_g2);
    cs.update_g2(&header.gamma_g2);
    cs.update_g1(&header.delta_g1);
    cs.update_g2(&header.delta_g2);

    let acc = process_constraints(&r1cs, domain_size)?;
    let n_coefs = acc.coefs.len();
    write_coefficients(&mut w, &r1cs, &acc.coefs)?;

    let bases = LagrangeBlocks::read(&ptau, domain_size)?;

    write_points_g1(&mut w, &mut cs, S_IC, &acc.ic, &bases, &r1cs, msm)?;

    w.start_section(S_H)?;
    w.write_g1_slice(&write_hs(&ptau, domain_size)?)?;
    w.end_section()?;

    hash_h_points(&ptau, domain_size, &mut cs)?;

    write_points_g1(&mut w, &mut cs, S_C, &acc.c, &bases, &r1cs, msm)?;
    write_points_g1(&mut w, &mut cs, S_A, &acc.a, &bases, &r1cs, msm)?;
    write_points_g1(&mut w, &mut cs, S_B1, &acc.b1, &bases, &r1cs, msm)?;
    write_points_g2(&mut w, &mut cs, S_B2, &acc.b2, &bases, &r1cs, msm)?;

    let cs_hash = cs.finalize();
    w.start_section(S_MPC_PARAMS)?;
    w.write_bytes(&cs_hash)?;
    w.write_u32(0)?;
    w.end_section()?;
    w.finish()?;

    Ok(SetupReport {
        n_vars,
        n_public,
        n_constraints,
        domain_size,
        cir_power,
        n_coefs,
        cs_hash,
    })
}

/// Which of the four ptau Lagrange sections a term's base comes from
/// (`zkey_new.js:38-41`, dispatched through `sBuffs` at `:415-420`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BaseTag {
    /// Section 12, `L_j(tau) * G1`.
    TauG1,
    /// Section 13, `L_j(tau) * G2`.
    TauG2,
    /// Section 14, `alpha * L_j(tau) * G1`.
    AlphaTauG1,
    /// Section 15, `beta * L_j(tau) * G1`.
    BetaTauG1,
}

/// One term of one output point: which Lagrange buffer, which element of it, and where
/// the scalar lives.
///
/// `coef_ptr` of `None` is snarkjs' `-1` sentinel and means "the scalar is 1"
/// (`zkey_new.js:436-446`). It appears only in the public-signal tail, which appends a
/// synthetic constraint row per public signal with `a = 1, b = 0, c = 0` so that `A_s(x)`
/// is nonzero for every public signal.
#[derive(Clone, Copy, Debug)]
pub struct PointTerm {
    pub tag: BaseTag,
    /// Element index into the tagged Lagrange block, which is the constraint index.
    pub index: u32,
    pub coef_ptr: Option<u32>,
}

/// The five accumulator lists `processConstraints` builds, before any point is computed.
///
/// Lengths are fixed at construction and never grow: `A`, `B1` and `B2` are `nVars` long,
/// `C` is `nVars - nPublic - 1`, and `IC` is `nPublic + 1`. That matters beyond bookkeeping
/// because each length is hashed into `csHash` as a big-endian u32 before its points. A
/// signal that appears in no constraint leaves an empty slot, and an empty slot writes the
/// group identity, which serialises as all-zero bytes.
pub struct Accumulators {
    pub ic: Vec<Vec<PointTerm>>,
    pub a: Vec<Vec<PointTerm>>,
    pub b1: Vec<Vec<PointTerm>>,
    pub b2: Vec<Vec<PointTerm>>,
    pub c: Vec<Vec<PointTerm>>,
    /// Section 4 in push order: `(matrix, constraint, signal, coef_ptr)`, matrix 0 for A
    /// and 1 for B. C terms are never recorded. Do not sort this: the order is execution
    /// order and the file stores it as such.
    pub coefs: Vec<CoefRecord>,
}

/// One section-4 record.
#[derive(Clone, Copy, Debug)]
pub struct CoefRecord {
    /// 0 for A, 1 for B. Never 2.
    pub matrix: u32,
    pub constraint: u32,
    pub signal: u32,
    /// `None` is the scalar 1, as in [`PointTerm`].
    pub coef_ptr: Option<u32>,
}

/// Walk the r1cs once and build every accumulator, in the three-pass order
/// `zkey_new.js:222-300` uses: A terms, then B terms, then C terms per constraint, then
/// the public-signal tail.
pub fn process_constraints(r1cs: &R1cs, domain_size: usize) -> Result<Accumulators, CeremonyError> {
    let n_vars = r1cs.header().n_vars as usize;
    let n_public = r1cs.header().n_public();
    let n_constraints = r1cs.header().n_constraints as usize;

    // Signal 0 is the constant ONE and signals `1 ..= nPublic` are the public ones, so a
    // circuit with no private signal at all still needs `nVars > nPublic`. snarkjs would
    // instead build a `BigArray` of negative length and only fail later, in the C write.
    if n_vars <= n_public {
        return Err(CeremonyError::BadParams(format!(
            "{n_vars} signals cannot hold the ONE signal plus {n_public} public ones"
        )));
    }
    // The tail rows sit at Lagrange element `nConstraints + s`, and the largest of them
    // has to stay inside the `domainSize`-element block. `circuit_power` guarantees it;
    // this catches a caller that computed `domain_size` some other way, before the bound
    // turns into an out-of-range base index a few million points later.
    if n_constraints + n_public >= domain_size {
        return Err(CeremonyError::BadParams(format!(
            "domain size {domain_size} does not cover {n_constraints} constraints plus \
             {n_public} public signals"
        )));
    }

    let mut ic = vec![Vec::new(); n_public + 1];
    let mut a = vec![Vec::new(); n_vars];
    let mut b1 = vec![Vec::new(); n_vars];
    let mut b2 = vec![Vec::new(); n_vars];
    let mut c = vec![Vec::new(); n_vars - n_public - 1];
    let nnz = r1cs.nonzeros()?;
    let mut coefs = Vec::with_capacity(nnz.a + nnz.b + n_public + 1);

    // `IC` and `C` partition the signals, so one closure covers both destinations. The
    // tag differs per pass, which is the only thing that distinguishes the three.
    macro_rules! push_ic_or_c {
        ($signal:expr, $term:expr) => {{
            let s = $signal as usize;
            if s <= n_public {
                ic[s].push($term);
            } else {
                c[s - n_public - 1].push($term);
            }
        }};
    }

    for (constraint, cons) in r1cs.constraints()?.iter().enumerate() {
        let constraint = constraint as u32;
        for term in &cons.a {
            let coef_ptr = Some(term.coef_ptr);
            a[term.signal as usize].push(PointTerm {
                tag: BaseTag::TauG1,
                index: constraint,
                coef_ptr,
            });
            push_ic_or_c!(
                term.signal,
                PointTerm {
                    tag: BaseTag::BetaTauG1,
                    index: constraint,
                    coef_ptr,
                }
            );
            coefs.push(CoefRecord {
                matrix: Matrix::A.as_u32(),
                constraint,
                signal: term.signal,
                coef_ptr,
            });
        }
        for term in &cons.b {
            let coef_ptr = Some(term.coef_ptr);
            b1[term.signal as usize].push(PointTerm {
                tag: BaseTag::TauG1,
                index: constraint,
                coef_ptr,
            });
            b2[term.signal as usize].push(PointTerm {
                tag: BaseTag::TauG2,
                index: constraint,
                coef_ptr,
            });
            push_ic_or_c!(
                term.signal,
                PointTerm {
                    tag: BaseTag::AlphaTauG1,
                    index: constraint,
                    coef_ptr,
                }
            );
            coefs.push(CoefRecord {
                matrix: Matrix::B.as_u32(),
                constraint,
                signal: term.signal,
                coef_ptr,
            });
        }
        // The C pass touches no coefficient record and no A/B accumulator: the C matrix
        // reaches the key only through `IC` and `C` (`zkey_new.js:272-287`).
        for term in &cons.c {
            push_ic_or_c!(
                term.signal,
                PointTerm {
                    tag: BaseTag::TauG1,
                    index: constraint,
                    coef_ptr: Some(term.coef_ptr),
                }
            );
        }
    }

    // One synthetic row per public signal, `a = 1` and nothing else, so `A_s(x)` is
    // nonzero for every public signal (`zkey_new.js:290-300`).
    for s in 0..=n_public {
        let index = (n_constraints + s) as u32;
        a[s].push(PointTerm {
            tag: BaseTag::TauG1,
            index,
            coef_ptr: None,
        });
        ic[s].push(PointTerm {
            tag: BaseTag::BetaTauG1,
            index,
            coef_ptr: None,
        });
        coefs.push(CoefRecord {
            matrix: Matrix::A.as_u32(),
            constraint: index,
            signal: s as u32,
            coef_ptr: None,
        });
    }

    Ok(Accumulators {
        ic,
        a,
        b1,
        b2,
        c,
        coefs,
    })
}

/// Section 4: a `u32` count then one 44-byte record each, the coefficient re-encoded from
/// the r1cs' plain little-endian into double Montgomery (`zkey_new.js:303-334`).
fn write_coefficients(
    w: &mut BinFileWriter,
    r1cs: &R1cs,
    coefs: &[CoefRecord],
) -> Result<(), CeremonyError> {
    w.start_section(S_COEFFS)?;
    w.write_u32(coefs.len() as u32)?;
    for batch in coefs.chunks(COEF_BATCH) {
        let encoded: Vec<[u8; COEF_RECORD_BYTES]> = batch
            .par_iter()
            .map(|rec| {
                let value = coefficient(r1cs, rec.coef_ptr)?;
                let mut out = [0u8; COEF_RECORD_BYTES];
                out[..4].copy_from_slice(&rec.matrix.to_le_bytes());
                out[4..8].copy_from_slice(&rec.constraint.to_le_bytes());
                out[8..12].copy_from_slice(&rec.signal.to_le_bytes());
                out[12..].copy_from_slice(&crate::write::fr_double_montgomery(&value));
                Ok(out)
            })
            .collect::<Result<_, CeremonyError>>()?;
        for record in &encoded {
            w.write_bytes(record)?;
        }
    }
    w.end_section()
}

/// The scalar a term contributes: the raw r1cs coefficient, or 1 for the `-1` sentinel.
fn coefficient(r1cs: &R1cs, coef_ptr: Option<u32>) -> Result<Fr, CeremonyError> {
    match coef_ptr {
        Some(ptr) => r1cs.coef(ptr),
        None => Ok(Fr::one()),
    }
}

/// Hash the slot count, compose the points, write the section, hash the points.
///
/// snarkjs interleaves the writing and the hashing chunk by chunk (`zkey_new.js:369-377`)
/// but nothing else touches the hasher in between, so the byte stream is the same.
fn write_points_g1(
    w: &mut BinFileWriter,
    cs: &mut Transcript,
    id: u32,
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<(), CeremonyError> {
    cs.update_u32_be(slots.len() as u32);
    let points = compose_points_g1(slots, bases, r1cs, msm)?;
    w.start_section(id)?;
    w.write_g1_slice(&points)?;
    w.end_section()?;
    for p in &points {
        cs.update_g1(p);
    }
    Ok(())
}

/// [`write_points_g1`] for section 7.
fn write_points_g2(
    w: &mut BinFileWriter,
    cs: &mut Transcript,
    id: u32,
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<(), CeremonyError> {
    cs.update_u32_be(slots.len() as u32);
    let points = compose_points_g2(slots, bases, r1cs, msm)?;
    w.start_section(id)?;
    w.write_g2_slice(&points)?;
    w.end_section()?;
    for p in &points {
        cs.update_g2(p);
    }
    Ok(())
}

/// One output point per accumulator slot: identity for an empty slot, a single scalar
/// multiplication for a slot with one term, a multiexp otherwise.
pub fn compose_points_g1(
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<G1Affine>, CeremonyError> {
    let acc: Vec<G1Projective> = slots
        .par_iter()
        .map(|slot| {
            let (points, scalars) = gather(slot, r1cs, |term| bases.g1(term))?;
            Ok(combine(&points, &scalars, |b, s| msm.msm_g1(b, s)))
        })
        .collect::<Result<_, CeremonyError>>()?;
    Ok(G1Projective::normalize_batch(&acc))
}

/// [`compose_points_g1`] for section 7, the only G2 output.
pub fn compose_points_g2(
    slots: &[Vec<PointTerm>],
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<G2Affine>, CeremonyError> {
    let acc: Vec<G2Projective> = slots
        .par_iter()
        .map(|slot| {
            let (points, scalars) = gather(slot, r1cs, |term| bases.g2(term))?;
            Ok(combine(&points, &scalars, |b, s| msm.msm_g2(b, s)))
        })
        .collect::<Result<_, CeremonyError>>()?;
    Ok(G2Projective::normalize_batch(&acc))
}

/// The bases and scalars of one slot, in term order.
fn gather<P, F>(
    slot: &[PointTerm],
    r1cs: &R1cs,
    base: F,
) -> Result<(Vec<P>, Vec<Fr>), CeremonyError>
where
    F: Fn(&PointTerm) -> Result<P, CeremonyError>,
{
    let mut points = Vec::with_capacity(slot.len());
    let mut scalars = Vec::with_capacity(slot.len());
    for term in slot {
        points.push(base(term)?);
        scalars.push(coefficient(r1cs, term.coef_ptr)?);
    }
    Ok((points, scalars))
}

/// snarkjs' three-way dispatch (`zkey_new.js:459-486`): identity, one scalar
/// multiplication, or a multiexp. See [`MULTIEXP_MIN_TERMS`] for the fourth case, which is
/// a small-slot shortcut and not a fourth behaviour.
fn combine<A, P, F>(points: &[A], scalars: &[Fr], msm: F) -> P
where
    A: Copy + Mul<Fr, Output = P>,
    P: Zero + Copy + std::ops::AddAssign<P>,
    F: Fn(&[A], &[Fr]) -> P,
{
    if points.len() >= MULTIEXP_MIN_TERMS {
        return msm(points, scalars);
    }
    let mut acc = P::zero();
    for (p, s) in points.iter().zip(scalars) {
        acc += *p * *s;
    }
    acc
}

/// The four `domainSize`-point Lagrange blocks setup reads out of the ptau, one per
/// [`BaseTag`], each from element `domainSize - 1` of its section.
pub struct LagrangeBlocks {
    pub tau_g1: Vec<G1Affine>,
    pub tau_g2: Vec<G2Affine>,
    pub alpha_tau_g1: Vec<G1Affine>,
    pub beta_tau_g1: Vec<G1Affine>,
}

impl LagrangeBlocks {
    pub fn read(ptau: &Ptau, domain_size: usize) -> Result<Self, CeremonyError> {
        Ok(Self {
            tau_g1: ptau.lagrange_block_g1(ptau::S_LAGRANGE_TAU_G1, domain_size)?,
            tau_g2: ptau.lagrange_block_g2(domain_size)?,
            alpha_tau_g1: ptau.lagrange_block_g1(ptau::S_LAGRANGE_ALPHA_TAU_G1, domain_size)?,
            beta_tau_g1: ptau.lagrange_block_g1(ptau::S_LAGRANGE_BETA_TAU_G1, domain_size)?,
        })
    }

    fn g1(&self, term: &PointTerm) -> Result<G1Affine, CeremonyError> {
        let block = match term.tag {
            BaseTag::TauG1 => &self.tau_g1,
            BaseTag::AlphaTauG1 => &self.alpha_tau_g1,
            BaseTag::BetaTauG1 => &self.beta_tau_g1,
            BaseTag::TauG2 => return Err(wrong_group(term)),
        };
        block.get(term.index as usize).copied().ok_or_else(|| {
            CeremonyError::BadParams(format!(
                "Lagrange element {} is past the {}-point block",
                term.index,
                block.len()
            ))
        })
    }

    fn g2(&self, term: &PointTerm) -> Result<G2Affine, CeremonyError> {
        if term.tag != BaseTag::TauG2 {
            return Err(wrong_group(term));
        }
        self.tau_g2
            .get(term.index as usize)
            .copied()
            .ok_or_else(|| {
                CeremonyError::BadParams(format!(
                    "Lagrange element {} is past the {}-point block",
                    term.index,
                    self.tau_g2.len()
                ))
            })
    }
}

fn wrong_group(term: &PointTerm) -> CeremonyError {
    CeremonyError::BadParams(format!("{:?} is not a base of this group", term.tag))
}

/// Section 9: every odd element of the `2*domainSize` Lagrange block of ptau section 12.
/// No arithmetic. Only section 12 carries the `power+1` block, which is why this read is
/// possible at all when `cirPower == power`.
pub fn write_hs(ptau: &Ptau, domain_size: usize) -> Result<Vec<G1Affine>, CeremonyError> {
    if !domain_size.is_power_of_two() {
        return Err(CeremonyError::BadParams(format!(
            "domain size {domain_size} is not a power of two"
        )));
    }
    let cir_power = domain_size.trailing_zeros();
    // `2*domainSize` points from the block that starts at element `2*domainSize - 1`.
    let block = 2 * domain_size;
    if cir_power < Fr::TWO_ADICITY {
        let data = ptau.section_elements(ptau::S_LAGRANGE_TAU_G1, SG1, block - 1, block)?;
        Ok(data
            .par_chunks_exact(SG1)
            .skip(1)
            .step_by(2)
            .map(binfile::g1)
            .collect())
    } else if cir_power == Fr::TWO_ADICITY {
        // At the two-adicity there is no `2*domainSize`-th root of unity, so
        // `lagrangeEvaluations` took its coset branch and wrote the even and odd halves
        // contiguously instead of interleaved. Every odd element is then just the second
        // half (`zkey_new.js:192-194`).
        ptau.g1_points(
            ptau::S_LAGRANGE_TAU_G1,
            block - 1 + domain_size,
            domain_size,
        )
    } else {
        Err(CeremonyError::CircuitTooBig(cir_power))
    }
}

/// Fold the H difference points into `csHash`, bug included.
///
/// The points are `tauG1[i + domainSize] - tauG1[i]` from ptau **section 2**, the raw
/// monomial powers, not the Lagrange section 12 (`zkey_new.js:517-518`).
///
/// `zkey_new.js:511` reads `Math.min(domainSize-1, CHUNK_SIZE)` and forgot the `- i`, so
/// `n` does not depend on the chunk offset. With `CHUNK_SIZE = 1<<14`, a `domainSize` of
/// 2^15 or more hashes a full 16384 points per chunk over `2^(k-14)` chunks, that is
/// `domainSize` points after announcing `domainSize - 1`. When additionally
/// `cirPower == power` the last chunk reads 64 bytes past the end of section 2, through a
/// raw `fd.read` that bounds-checks nothing, and folds section 3's header and the first 52
/// bytes of `tauG2[0]` into the digest.
///
/// This is the shipped behaviour and `csHash` is defined by it, so reproduce it. Which is
/// why the reads below go through the whole-file buffer and the section's start offset
/// rather than [`Ptau::section_elements`]: a bounds-checked read would refuse the last
/// chunk of exactly the case that matters. Confirmed against snarkjs on `tornado` at
/// `cirPower == power == 15` and `sha256` at 16, both byte-identical.
/// The verifier is immune because `sameRatioH` multiplies index `domainSize - 1` by zero.
pub fn hash_h_points(
    ptau: &Ptau,
    domain_size: usize,
    transcript: &mut Transcript,
) -> Result<(), CeremonyError> {
    transcript.update_u32_be((domain_size - 1) as u32);

    let file = ptau.file();
    let start = file
        .sections()
        .iter()
        .find(|s| s.id == ptau::S_TAU_G1)
        .ok_or(CeremonyError::MissingSection(ptau::S_TAU_G1))?
        .start;
    let bytes = file.bytes();

    let mut i = 0usize;
    while i < domain_size - 1 {
        let n = std::cmp::min(domain_size - 1, H_CHUNK);
        let read = |element: usize| -> Result<&[u8], CeremonyError> {
            let from = start + element * SG1;
            bytes.get(from..from + n * SG1).ok_or_else(|| {
                CeremonyError::malformed(
                    ptau::S_TAU_G1,
                    format!(
                        "H hash reads {n} points at element {element}, past the end of the file"
                    ),
                )
            })
        };
        let hi = read(i + domain_size)?;
        let lo = read(i)?;
        let diff: Vec<G1Projective> = hi
            .par_chunks_exact(SG1)
            .zip(lo.par_chunks_exact(SG1))
            .map(|(a, b)| h_difference(a, b))
            .collect();
        for p in G1Projective::normalize_batch(&diff) {
            transcript.update_g1(&p);
        }
        i += H_CHUNK;
    }
    Ok(())
}

/// One `g1m_subAffine`, on whichever implementation the input calls for.
///
/// Every real ptau point takes the arkworks path. The bytes read past section 2 do not: a
/// coordinate at or above `q` makes wasmcurves' wrapping arithmetic observable, and the
/// digest is defined by it, so those go through [`noncanonical`].
fn h_difference(hi: &[u8], lo: &[u8]) -> G1Projective {
    let canonical = [&hi[..N8], &hi[N8..], &lo[..N8], &lo[N8..]]
        .iter()
        .all(|b| noncanonical::in_range(&noncanonical::load(b)));
    if canonical {
        return G1Projective::from(binfile::g1(hi)) - binfile::g1(lo);
    }
    let (x, y, z) = noncanonical::sub_affine(hi, lo);
    if noncanonical::is_zero(&z) {
        return G1Projective::zero();
    }
    G1Projective::new_unchecked(
        montgomery_limbs(&x),
        montgomery_limbs(&y),
        montgomery_limbs(&z),
    )
}

/// Limbs wasmcurves left above `q` become the arkworks element with the same value: the
/// remaining steps, a batch inversion and two multiplications, are congruence preserving on
/// both sides, so only the representative had to be settled here.
fn montgomery_limbs(limbs: &noncanonical::U256) -> Fq {
    let mut le = [0u8; N8];
    for (i, limb) in limbs.iter().enumerate() {
        le[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
    }
    Fq::new_unchecked(Fq::from_le_bytes_mod_order(&le).into_bigint())
}

/// wasmcurves' `Fq` and its Jacobian point addition, transcribed limb for limb.
///
/// It exists for one input in the whole pipeline: the 64 bytes [`hash_h_points`]
/// deliberately reads past ptau section 2, which are a section header and part of the next
/// point rather than a curve point. Both coordinates come out at or above `q`, and there
/// arkworks and wasmcurves stop agreeing. `f1m_add` (`build_f1m.js:69-85`) reacts to a
/// 256-bit carry by subtracting `q` once from the *wrapped* sum, which is short by exactly
/// `2^256`, so the result is not even congruent mod `q`. Reducing the inputs first does not
/// recover it, and neither does any canonical implementation: the shipped digest is defined
/// by these exact wrapping semantics.
///
/// Nothing else may use this. Every in-range input goes through arkworks, which is both
/// faster and, on those, identical.
mod noncanonical {
    /// Eight little-endian 32-bit limbs, the layout wasm linear memory holds and the width
    /// `f1m_mul`'s inner loop works in (`build_f1m.js:236-243`).
    pub type U256 = [u32; 8];

    pub const ZERO: U256 = [0; 8];
    const Q: U256 = [
        0xd87c_fd47,
        0x3c20_8c16,
        0x6871_ca8d,
        0x9781_6a91,
        0x8181_585d,
        0xb850_45b6,
        0xe131_a029,
        0x3064_4e72,
    ];
    /// `R mod q`, which is what `f1m_one` writes (`build_f1m.js:41`).
    const ONE: U256 = [
        0xc58f_0d9d,
        0xd35d_438d,
        0xf5c7_0b3d,
        0x0a78_eb28,
        0x7879_462c,
        0x666e_a36f,
        0x9a07_df2f,
        0x0e0a_77c1,
    ];
    /// `2^32 - (q^-1 mod 2^32)`, the Montgomery constant the 32-bit CIOS loop multiplies by.
    const NP32: u64 = 0xe486_6389;

    pub fn load(bytes: &[u8]) -> U256 {
        let mut out = ZERO;
        for (i, limb) in out.iter_mut().enumerate() {
            *limb = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        }
        out
    }

    /// True when the limbs are a canonical field element, so arkworks and wasmcurves agree
    /// on every operation that follows and this module is not needed.
    pub fn in_range(x: &U256) -> bool {
        !int_gte(x, &Q)
    }

    pub fn is_zero(x: &U256) -> bool {
        x == &ZERO
    }

    /// `int_add`: the wrapped sum and the carry out (`build_int.js:186-230`).
    fn int_add(x: &U256, y: &U256) -> (U256, bool) {
        let mut out = ZERO;
        let mut carry = 0u64;
        for i in 0..8 {
            let acc = u64::from(x[i]) + u64::from(y[i]) + carry;
            out[i] = acc as u32;
            carry = acc >> 32;
        }
        (out, carry != 0)
    }

    /// `int_sub`: the wrapped difference and the borrow out (`build_int.js:232-278`).
    fn int_sub(x: &U256, y: &U256) -> (U256, bool) {
        let mut out = ZERO;
        let mut borrow = 0u64;
        for i in 0..8 {
            let acc = u64::from(x[i])
                .wrapping_sub(u64::from(y[i]))
                .wrapping_sub(borrow);
            out[i] = acc as u32;
            borrow = (acc >> 32) & 1;
        }
        (out, borrow != 0)
    }

    /// `int_gte`, an unsigned compare from the top limb down (`build_int.js:148-180`).
    fn int_gte(x: &U256, y: &U256) -> bool {
        for i in (0..8).rev() {
            if x[i] != y[i] {
                return x[i] > y[i];
            }
        }
        true
    }

    /// `f1m_add` (`build_f1m.js:69-85`). The carry branch subtracts `q` from a sum that has
    /// already lost `2^256`; that is the whole reason this module exists.
    pub fn add(x: &U256, y: &U256) -> U256 {
        let (r, carry) = int_add(x, y);
        if carry || int_gte(&r, &Q) {
            int_sub(&r, &Q).0
        } else {
            r
        }
    }

    /// `f1m_sub` (`build_f1m.js:87-102`).
    pub fn sub(x: &U256, y: &U256) -> U256 {
        let (r, borrow) = int_sub(x, y);
        if borrow {
            int_add(&r, &Q).0
        } else {
            r
        }
    }

    pub fn neg(x: &U256) -> U256 {
        sub(&ZERO, x)
    }

    /// `f1m_mul` (`build_f1m.js:235-435`): CIOS over 32-bit limbs with a two-word
    /// accumulator, then the same carry-or-gte reduction `add` uses.
    ///
    /// The `[c0, c1] = [c1, c0]` at `:407` is a rename in the code generator, so at run time
    /// it is a value swap followed by `c1 = c0 >> 32`, in that order.
    pub fn mul(x: &U256, y: &U256) -> U256 {
        let mut r = ZERO;
        let mut m = [0u64; 8];
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for k in 0..15usize {
            for i in k.saturating_sub(7)..=k.min(7) {
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(x[i]) * u64::from(y[k - i]);
                c1 += c0 >> 32;
            }
            for i in k.saturating_sub(7).max(1)..=k.min(7) {
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(Q[i]) * m[k - i];
                c1 += c0 >> 32;
            }
            if k < 8 {
                m[k] = ((c0 & 0xFFFF_FFFF) * NP32) & 0xFFFF_FFFF;
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(Q[0]) * m[k];
                c1 += c0 >> 32;
            }
            if k >= 8 {
                r[k - 8] = c0 as u32;
            }
            std::mem::swap(&mut c0, &mut c1);
            c1 = c0 >> 32;
        }
        r[7] = c0 as u32;
        if c1 as u32 != 0 || int_gte(&r, &Q) {
            int_sub(&r, &Q).0
        } else {
            r
        }
    }

    /// `g1m_subAffine` (`build_curve_jacobian_a0.js:919-932`): negate the second point, then
    /// `addAffine`, which is mmadd-2007-bl with both `Z` implicitly one (`:760-843`).
    ///
    /// Returns the Jacobian triple `batchToAffine` would be handed.
    pub fn sub_affine(p1: &[u8], p2: &[u8]) -> (U256, U256, U256) {
        let (x1, y1) = (load(&p1[..32]), load(&p1[32..64]));
        let (x2, y2) = (load(&p2[..32]), neg(&load(&p2[32..64])));

        // `isZeroAffine` is an integer test on both coordinates, so it fires on the raw
        // limbs and not on the value.
        if is_zero(&x1) && is_zero(&y1) {
            return (x2, y2, ONE);
        }
        if is_zero(&x2) && is_zero(&y2) {
            return (x1, y1, ONE);
        }
        // Equality is on limbs too, so two representatives of the same value would miss the
        // doubling branch. That is the shipped behaviour.
        if x1 == x2 && y1 == y2 {
            return double_affine(&x2, &y2);
        }

        let h = sub(&x2, &x1);
        let y2_minus_y1 = sub(&y2, &y1);
        let hh = square(&h);
        let i = add(&hh, &hh);
        let i = add(&i, &i);
        let j = mul(&h, &i);
        let r = add(&y2_minus_y1, &y2_minus_y1);
        let v = mul(&x1, &i);
        let r2 = square(&r);
        let v2 = add(&v, &v);

        let x3 = sub(&r2, &j);
        let x3 = sub(&x3, &v2);

        let y1_j2 = mul(&y1, &j);
        let y1_j2 = add(&y1_j2, &y1_j2);

        let y3 = sub(&v, &x3);
        let y3 = mul(&y3, &r);
        let y3 = sub(&y3, &y1_j2);

        (x3, y3, add(&h, &h))
    }

    /// `doubleAffine`, dbl-2009-l (`build_curve_jacobian_a0.js:358-424`). Reachable from
    /// [`sub_affine`] only when the two limb patterns coincide.
    fn double_affine(x: &U256, y: &U256) -> (U256, U256, U256) {
        if is_zero(x) && is_zero(y) {
            return (ZERO, ZERO, ZERO);
        }
        let xx = square(x);
        let yy = square(y);
        let yyyy = square(&yy);
        let s = add(x, &yy);
        let s = square(&s);
        let s = sub(&s, &xx);
        let s = sub(&s, &yyyy);
        let s = add(&s, &s);
        let m = add(&xx, &xx);
        let m = add(&m, &xx);
        let z3 = add(y, y);
        let x3 = square(&m);
        let x3 = sub(&x3, &s);
        let x3 = sub(&x3, &s);
        let eight_yyyy = add(&yyyy, &yyyy);
        let eight_yyyy = add(&eight_yyyy, &eight_yyyy);
        let eight_yyyy = add(&eight_yyyy, &eight_yyyy);
        let y3 = sub(&s, &x3);
        let y3 = mul(&y3, &m);
        let y3 = sub(&y3, &eight_yyyy);
        (x3, y3, z3)
    }

    /// `f1m_square` is a separate function from `f1m_mul` (`build_f1m.js:437`), but only as
    /// a schedule: it accumulates the same partial products and ends in the same carry-or-gte
    /// reduction. Verified to agree with `mul(x, x)` on the out-of-range point this module
    /// exists for; see `tests/setup.rs`.
    fn square(x: &U256) -> U256 {
        mul(x, x)
    }
}
