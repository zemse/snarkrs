//! Phase 1: `powersoftau new`, `contribute`, `beacon` and `verify`.
//!
//! A fresh file is fully determined by `power`. `newAccumulator` writes the generator
//! repeated `2n-1`, `n`, `n`, `n` and once into sections 2 to 6, and a u32 zero into
//! section 7 (`powersoftau_new.js:73-143`). No randomness, no timestamps, and nothing
//! derived from the first challenge hash it returns.
//!
//! Each contribution then hashes the same points **twice, in two different encodings**,
//! and swapping them is undetectable until `snarkjs powersoftau verify` rejects the file:
//!
//! * the **response** hash absorbs the previous challenge and then the new points
//!   **compressed**, in section order 2, 3, 4, 5, 6; the BLAKE2b state is snapshotted at
//!   that point into `partialHash`, and only then is the 768-byte uncompressed pubkey
//!   absorbed (`powersoftau_contribute.js:66-93`);
//! * the **next challenge** hash absorbs that response digest and then the same points
//!   **uncompressed** (`:97-106`).
//!
//! The response hash is never stored. `partialHash` exists precisely so it can be
//! reconstructed on read by resuming and re-absorbing the pubkey.
//!
//! `contribute` and `beacon` both refuse a file where `power != ceremonyPower`
//! (`powersoftau_contribute.js:37-40`), and both warn when section 12 is present, since
//! any contribution invalidates the prepared sections and phase 2 has to be re-prepared.
//! We drop sections 12 to 15 the same way snarkjs does, by declaring seven sections and
//! never writing them; the caller is expected to re-run `g16 ptau prepare`.
//!
//! The points are transformed once and hashed twice in two encodings, which is why the
//! new file is **read back off disk** for the uncompressed pass rather than kept in RAM.
//! snarkjs does the same (`hashSection` seeks into `fdNew`, `:161-182`) and for the same
//! reason: the next challenge cannot start until the response digest exists, and holding
//! a second copy of a 2^28 accumulator is 12 GB.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use ark_ec::short_weierstrass::Affine;
use blake2::{Blake2b512, Digest as _};
use g16_field::{g1, g2, AffineRepr, CurveGroup, Domain, Field, Fr, G1Affine, G2Affine, Zero};
use g16_msm::{AccelError, CpuMsm, KeyScale, MsmBackend};
use g16_ntt::{CpuNtt, Direction, NttBackend};
use g16_zkey::binfile;
use rayon::prelude::*;

use crate::ptau::{self, Ptau, PtauContribution, PtauHeader};
use crate::transcript::{
    g1_compressed, g1_uncompressed, g2_compressed, g2_uncompressed, get_g2_sp,
    rng_from_beacon_params, rng_from_entropy, same_ratio, write_ptau_pubkey, CeremonyRng, Digest,
    PtauKey, PtauPubKey, Transcript, PARTIAL_HASH_BYTES, PERSONALIZATION_ALPHA,
    PERSONALIZATION_BETA, PERSONALIZATION_TAU,
};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionKind, ContributionParams, SCG1, SCG2, SG1, SG2};

/// `nSections` a fresh file declares (`powersoftau_new.js:75`). A prepared, converted or
/// truncated file declares 11, which is why the count is not a test for "prepared".
pub const PTAU_NEW_SECTIONS: u32 = 7;
/// `nSections` `prepare phase2`, `convert` and `truncate` declare.
pub const PTAU_PREPARED_SECTIONS: u32 = 11;

/// snarkjs' bounds on a beacon (`powersoftau_beacon.js:26-42`, `zkey_beacon.js:38-42`).
/// Both phases enforce the same ones.
pub const BEACON_ITERATIONS_MIN: u8 = 10;
pub const BEACON_ITERATIONS_MAX: u8 = 63;
pub const BEACON_HASH_MAX_BYTES: usize = 255;

/// `powersoftau new` refuses anything outside this (`cli.js:701`). 28 is also the
/// two-adicity of `Fr`, so a larger accumulator could never be prepared for phase 2.
pub const POWER_MIN: u32 = 1;
pub const POWER_MAX: u32 = 28;

/// Points hashed into the response per `update`, `floor((1<<20)/sG)`: 16384 G1 or 8192
/// G2 (`powersoftau_contribute.js:137`, `powersoftau_beacon.js:143`).
///
/// This number is **not** free to change. `partialHash` is a serialised mid-stream
/// BLAKE2b state, and the bytes past its `pos` are whatever the last zero-copy compression
/// left in the block buffer, so a different chunking writes different bytes into the
/// `.ptau` (see [`crate::transcript::Transcript::update`]). It is also not
/// `mpc_applykey.js`'s `MAX_CHUNK_SIZE`: phase 1 has its own `processSection` and uses
/// this instead.
const fn response_chunk(sg: usize) -> usize {
    (1 << 20) / sg
}

/// Points read back per `update` of the next-challenge hash, `floor((1<<24)/sG)`
/// (`powersoftau_contribute.js:165`). Only the digest of that hasher is ever used, so
/// unlike [`response_chunk`] this one is a pure buffering choice.
const fn challenge_chunk(sg: usize) -> usize {
    (1 << 24) / sg
}

/// The two point groups a `.ptau` section can hold, with the three encodings each one is
/// written or hashed in. Sections 2, 4 and 5 are G1, sections 3 and 6 are G2, and every
/// step below differs only in these constants.
trait PtauGroup: AffineRepr<ScalarField = Fr> + Send + Sync {
    /// `sG`, bytes on disk (LEM and uncompressed are both this wide).
    const SG: usize;
    /// `scG`, bytes compressed.
    const SCG: usize;
    fn from_lem(bytes: &[u8]) -> Self;
    fn write_lem(&self, out: &mut [u8]);
    fn write_compressed(&self, out: &mut [u8]);
    fn write_uncompressed(&self, out: &mut [u8]);
    /// The one call `verify` makes that the group cannot express generically: the backend
    /// trait has a separate entry point per group.
    fn msm(backend: &dyn MsmBackend, bases: &[Self], scalars: &[Fr]) -> Self::Group;
    /// [`Self::msm`] for the contribution's own primitive.
    fn apply_key(
        scale: &dyn KeyScale,
        points: &mut [Self],
        first: Fr,
        inc: Fr,
    ) -> Result<(), AccelError>;
    fn read_points(
        file: &Ptau,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<Self>, CeremonyError>;
}

// Written against the short-Weierstrass types directly rather than the `G1Affine` /
// `G2Affine` aliases: those are projections through `BnConfig`, and coherence declines to
// normalise a projection in an impl header, so the two impls read as one.
impl PtauGroup for Affine<g1::Config> {
    const SG: usize = SG1;
    const SCG: usize = SCG1;

    fn from_lem(bytes: &[u8]) -> Self {
        binfile::g1(bytes)
    }

    fn write_lem(&self, out: &mut [u8]) {
        out.copy_from_slice(&crate::write::g1_lem(self));
    }

    fn write_compressed(&self, out: &mut [u8]) {
        out.copy_from_slice(&g1_compressed(self));
    }

    fn write_uncompressed(&self, out: &mut [u8]) {
        out.copy_from_slice(&g1_uncompressed(self));
    }

    fn msm(backend: &dyn MsmBackend, bases: &[Self], scalars: &[Fr]) -> Self::Group {
        backend.msm_g1(bases, scalars)
    }

    fn apply_key(
        scale: &dyn KeyScale,
        points: &mut [Self],
        first: Fr,
        inc: Fr,
    ) -> Result<(), AccelError> {
        scale.apply_key_g1(points, first, inc)
    }

    fn read_points(
        file: &Ptau,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<Self>, CeremonyError> {
        file.g1_points(id, offset, n)
    }
}

impl PtauGroup for Affine<g2::Config> {
    const SG: usize = SG2;
    const SCG: usize = SCG2;

    fn from_lem(bytes: &[u8]) -> Self {
        binfile::g2(bytes)
    }

    fn write_lem(&self, out: &mut [u8]) {
        out.copy_from_slice(&crate::write::g2_lem(self));
    }

    fn write_compressed(&self, out: &mut [u8]) {
        out.copy_from_slice(&g2_compressed(self));
    }

    fn write_uncompressed(&self, out: &mut [u8]) {
        out.copy_from_slice(&g2_uncompressed(self));
    }

    fn msm(backend: &dyn MsmBackend, bases: &[Self], scalars: &[Fr]) -> Self::Group {
        backend.msm_g2(bases, scalars)
    }

    fn apply_key(
        scale: &dyn KeyScale,
        points: &mut [Self],
        first: Fr,
        inc: Fr,
    ) -> Result<(), AccelError> {
        scale.apply_key_g2(points, first, inc)
    }

    fn read_points(
        file: &Ptau,
        id: u32,
        offset: usize,
        n: usize,
    ) -> Result<Vec<Self>, CeremonyError> {
        file.g2_points(id, offset, n)
    }
}

/// `calculateFirstChallengeHash` (`powersoftau_utils.js:312-358`), the challenge a file
/// with no contributions starts from.
///
/// It absorbs `blake2b512("")` and then the **uncompressed generators repeated**:
/// `2^power * 2 - 1` G1, `2^power` G2, `2^power` G1, `2^power` G1, one G2. Note it hashes
/// the generator repeated rather than the file's contents, which is correct only because a
/// fresh file is exactly that. The 341000-point blocking in the original is a performance
/// detail with no framing implication, since BLAKE2b is a stream.
///
/// Uses the `blake2` crate rather than [`Transcript`]: nothing here snapshots a partial
/// hash, so the only thing that matters is the digest, and at power 28 this absorbs 34 GB
/// of repeated generator that our hand-driven compression loop has no reason to be slow
/// over.
pub fn first_challenge_hash(power: u32) -> Digest {
    let mut h = Blake2b512::new();
    h.update(crate::transcript::blake2b512(b""));
    let n = 1u64 << power;
    let g1 = g1_uncompressed(&G1Affine::generator());
    let g2 = g2_uncompressed(&G2Affine::generator());
    let mut repeat = |point: &[u8], count: u64| {
        // 8192 points is 512 KB of G1 or 1 MB of G2, enough that the per-call overhead
        // disappears and small enough to stay in cache.
        let batch = 8192u64.min(count.max(1));
        let block: Vec<u8> = point.repeat(batch as usize);
        let mut left = count;
        while left >= batch {
            h.update(&block);
            left -= batch;
        }
        for _ in 0..left {
            h.update(point);
        }
    };
    repeat(&g1, n * 2 - 1);
    repeat(&g2, n);
    repeat(&g1, n);
    repeat(&g1, n);
    h.update(g2);
    h.finalize().into()
}

/// Write section 1, the 44-byte header (`powersoftau_utils.js:26-50`).
///
/// `ceremonyPower` is not optional in 0.7.6: the reader consumes it and then asserts the
/// section size, so a 40-byte three-field header is rejected. `writePTauHeader` treats a
/// falsy `ceremonyPower` as "same as power" (`:30`), which is how `powersoftau new` ends
/// up writing `power` twice.
fn write_header(
    w: &mut BinFileWriter,
    power: u32,
    ceremony_power: u32,
) -> Result<(), CeremonyError> {
    w.start_section(ptau::S_HEADER)?;
    w.write_prime(&crate::q_le())?;
    w.write_u32(power)?;
    w.write_u32(ceremony_power)?;
    w.end_section()
}

fn check_power(power: u32) -> Result<(), CeremonyError> {
    if !(POWER_MIN..=POWER_MAX).contains(&power) {
        return Err(CeremonyError::BadParams(format!(
            "power must be between {POWER_MIN} and {POWER_MAX}, got {power}"
        )));
    }
    Ok(())
}

/// `powersoftau new`: write a fresh accumulator at `power` and return its first challenge
/// hash.
pub fn ptau_new(power: u32, out: &Path) -> Result<Digest, CeremonyError> {
    check_power(power)?;
    let n = 1usize << power;
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();

    let mut w = BinFileWriter::create(
        out,
        ptau::PTAU_MAGIC,
        ptau::PTAU_MAX_VERSION,
        PTAU_NEW_SECTIONS,
    )?;
    write_header(&mut w, power, power)?;

    // tauG1 is `2n - 1`, not `2n` (`powersoftau_new.js:85`). Everything downstream that
    // reads this section, `prepare phase2` most of all, is built around the missing top
    // element.
    w.start_section(ptau::S_TAU_G1)?;
    w.write_g1_repeated(&g1, n * 2 - 1)?;
    w.end_section()?;

    w.start_section(ptau::S_TAU_G2)?;
    w.write_g2_repeated(&g2, n)?;
    w.end_section()?;

    w.start_section(ptau::S_ALPHA_TAU_G1)?;
    w.write_g1_repeated(&g1, n)?;
    w.end_section()?;

    w.start_section(ptau::S_BETA_TAU_G1)?;
    w.write_g1_repeated(&g1, n)?;
    w.end_section()?;

    w.start_section(ptau::S_BETA_G2)?;
    w.write_g2(&g2)?;
    w.end_section()?;

    // Section 7 exists even with nothing in it, and the file is unusable without it:
    // `readContributions` throws on a missing section 7 (`powersoftau_utils.js:227`).
    w.start_section(ptau::S_CONTRIBUTIONS)?;
    w.write_u32(0)?;
    w.end_section()?;

    w.finish()?;
    Ok(first_challenge_hash(power))
}

/// What a phase-1 contribution produced.
#[derive(Clone, Debug)]
pub struct Phase1Report {
    pub index: usize,
    pub kind: crate::ContributionKind,
    /// The reconstructed response hash, which snarkjs prints as the "Contribution Response
    /// Hash" for the contributor to publish.
    pub response_hash: Digest,
    /// The value actually stored in the record.
    pub next_challenge: Digest,
}

/// One of the five point sections, with the geometric scalar sequence applied to it.
///
/// `first` and `inc` are `processSection`'s two scalar arguments
/// (`powersoftau_contribute.js:75-84`): tau alone scales sections 2 and 3, while 4, 5 and
/// 6 start at alpha or beta and then step by tau.
struct SectionPlan {
    id: u32,
    n_points: usize,
    first: Fr,
    inc: Fr,
    /// Which transformed point the contribution record keeps: element 1 for sections 2 and
    /// 3 (`tau*G1`, `tau*G2`), element 0 for 4, 5 and 6.
    keep: usize,
}

/// Transform one section into the new file, hashing the compressed form into the response
/// as it goes, and return the point the contribution record keeps.
///
/// Reads and writes one `response_chunk` at a time, so peak memory is a megabyte of input
/// plus the transformed copy, independent of `power`.
fn process_section<C: PtauGroup>(
    ptau: &Ptau,
    w: &mut BinFileWriter,
    plan: &SectionPlan,
    response: &mut Transcript,
    scale: &dyn KeyScale,
) -> Result<C, CeremonyError> {
    let chunk = response_chunk(C::SG).min(plan.n_points.max(1));
    let mut lem = vec![0u8; chunk * C::SG];
    let mut comp = vec![0u8; chunk * C::SCG];
    let mut kept = None;
    let mut t = plan.first;

    w.start_section(plan.id)?;
    let mut done = 0usize;
    while done < plan.n_points {
        let n = (plan.n_points - done).min(chunk);
        let raw = ptau.section_elements(plan.id, C::SG, done, n)?;
        let mut points: Vec<C> = raw.par_chunks_exact(C::SG).map(C::from_lem).collect();
        C::apply_key(scale, &mut points, t, plan.inc)?;

        let lem_out = &mut lem[..n * C::SG];
        let comp_out = &mut comp[..n * C::SCG];
        points
            .par_iter()
            .zip(lem_out.par_chunks_exact_mut(C::SG))
            .zip(comp_out.par_chunks_exact_mut(C::SCG))
            .for_each(|((p, l), c)| {
                p.write_lem(l);
                p.write_compressed(c);
            });
        w.write_bytes(lem_out)?;
        response.update(comp_out);

        if plan.keep >= done && plan.keep < done + n {
            kept = Some(points[plan.keep - done]);
        }
        done += n;
        // `t = Fr.mul(t, Fr.exp(inc, n))`, so the next chunk starts where this one ended.
        t *= plan.inc.pow([n as u64]);
    }
    w.end_section()?;

    kept.ok_or_else(|| {
        CeremonyError::malformed(
            plan.id,
            format!(
                "section holds {} points, the record needs element {}",
                plan.n_points, plan.keep
            ),
        )
    })
}

/// Absorb one written section's **uncompressed** form into the next-challenge hash,
/// reading it back off the new file (`hashSection`, `powersoftau_contribute.js:161-182`).
fn hash_section_u<C: PtauGroup>(
    file: &mut File,
    offset: u64,
    n_points: usize,
    hasher: &mut Transcript,
) -> Result<(), CeremonyError> {
    let chunk = challenge_chunk(C::SG).min(n_points.max(1));
    let mut lem = vec![0u8; chunk * C::SG];
    let mut u = vec![0u8; chunk * C::SG];
    file.seek(SeekFrom::Start(offset))?;
    let mut done = 0usize;
    while done < n_points {
        let n = (n_points - done).min(chunk);
        file.read_exact(&mut lem[..n * C::SG])?;
        lem[..n * C::SG]
            .par_chunks_exact(C::SG)
            .zip(u[..n * C::SG].par_chunks_exact_mut(C::SG))
            .for_each(|(src, dst)| C::from_lem(src).write_uncompressed(dst));
        hasher.update(&u[..n * C::SG]);
        done += n;
    }
    Ok(())
}

/// The body of both `contribute` and `beacon`: they differ only in where the key comes
/// from and what lands in the params TLV (`powersoftau_contribute.js:33-118` against
/// `powersoftau_beacon.js:44-124`, which are otherwise the same file twice).
fn apply_contribution(
    ptau_in: &Path,
    ptau_out: &Path,
    kind: ContributionKind,
    params: ContributionParams,
    make_key: impl FnOnce(&Digest) -> PtauKey,
    scale: &dyn KeyScale,
) -> Result<Phase1Report, CeremonyError> {
    let ptau = Ptau::open(ptau_in)?;
    let header = *ptau.header();
    // A truncated file's points are a prefix of the ones the chain hashed, so a
    // contribution on top of it could never reproduce the next challenge
    // (`powersoftau_contribute.js:37-40`).
    if header.is_truncated_ceremony() {
        return Err(CeremonyError::TruncatedCeremony {
            power: header.power,
            ceremony_power: header.ceremony_power,
        });
    }
    check_power(header.power)?;
    // Fail on an unencodable name before a 600 MB file exists, not inside `encode`.
    params.encode()?;

    let power = header.power;
    let n = 1usize << power;
    let last_challenge = ptau.last_challenge()?;
    let key = make_key(&last_challenge);

    // Section 7 is copied forward verbatim rather than re-encoded. snarkjs rebuilds every
    // record from its parsed form (`writeContributions`), which silently re-truncates a
    // name and would drop a param type a future version added; the bytes we were handed
    // are the ones that were hashed.
    let prior = ptau.section(ptau::S_CONTRIBUTIONS)?;
    let n_prior = u32::from_le_bytes(
        prior
            .get(..4)
            .ok_or_else(|| CeremonyError::malformed(ptau::S_CONTRIBUTIONS, "section is empty"))?
            .try_into()
            .expect("four bytes"),
    );
    let prior_records = &prior[4..];

    let plans = [
        SectionPlan {
            id: ptau::S_TAU_G1,
            n_points: n * 2 - 1,
            first: Fr::ONE,
            inc: key.tau.prv_key,
            keep: 1,
        },
        SectionPlan {
            id: ptau::S_TAU_G2,
            n_points: n,
            first: Fr::ONE,
            inc: key.tau.prv_key,
            keep: 1,
        },
        SectionPlan {
            id: ptau::S_ALPHA_TAU_G1,
            n_points: n,
            first: key.alpha.prv_key,
            inc: key.tau.prv_key,
            keep: 0,
        },
        SectionPlan {
            id: ptau::S_BETA_TAU_G1,
            n_points: n,
            first: key.beta.prv_key,
            inc: key.tau.prv_key,
            keep: 0,
        },
        SectionPlan {
            id: ptau::S_BETA_G2,
            n_points: 1,
            first: key.beta.prv_key,
            inc: key.tau.prv_key,
            keep: 0,
        },
    ];

    let mut response = Transcript::new();
    response.update(&last_challenge);

    let mut w = BinFileWriter::create(
        ptau_out,
        ptau::PTAU_MAGIC,
        ptau::PTAU_MAX_VERSION,
        PTAU_NEW_SECTIONS,
    )?;
    write_header(&mut w, power, power)?;

    // Payload offsets in the file being written, so the uncompressed pass can seek to
    // them. Every length up to section 6 is fixed by `power`, and 12 bytes of entry header
    // precede each payload (`binfileutils.js:53-58`).
    let mut at = 12 + 12 + PtauHeader::BYTES as u64;
    let mut starts = [0u64; 5];
    for (i, plan) in plans.iter().enumerate() {
        at += 12;
        starts[i] = at;
        at += plan.n_points as u64
            * if plan.id == ptau::S_TAU_G2 || plan.id == ptau::S_BETA_G2 {
                SG2 as u64
            } else {
                SG1 as u64
            };
    }

    let tau_g1 = process_section::<G1Affine>(&ptau, &mut w, &plans[0], &mut response, scale)?;
    let tau_g2 = process_section::<G2Affine>(&ptau, &mut w, &plans[1], &mut response, scale)?;
    let alpha_g1 = process_section::<G1Affine>(&ptau, &mut w, &plans[2], &mut response, scale)?;
    let beta_g1 = process_section::<G1Affine>(&ptau, &mut w, &plans[3], &mut response, scale)?;
    let beta_g2 = process_section::<G2Affine>(&ptau, &mut w, &plans[4], &mut response, scale)?;

    // Snapshot before the pubkey, which is the only reason `partialHash` is stored
    // (`powersoftau_contribute.js:86`).
    let partial_hash: [u8; PARTIAL_HASH_BYTES] = response.partial_hash();
    let pubkeys = key.pubkeys();
    response.update(&write_ptau_pubkey(&pubkeys, false));
    let response_hash = response.finalize();

    let mut next = Transcript::new();
    next.update(&response_hash);
    {
        // The writer flushes on every `end_section` seek, so the payloads of sections 1
        // to 6 are already on disk; this handle only ever reads behind the write cursor.
        let mut back = File::open(ptau_out)?;
        hash_section_u::<G1Affine>(&mut back, starts[0], plans[0].n_points, &mut next)?;
        hash_section_u::<G2Affine>(&mut back, starts[1], plans[1].n_points, &mut next)?;
        hash_section_u::<G1Affine>(&mut back, starts[2], plans[2].n_points, &mut next)?;
        hash_section_u::<G1Affine>(&mut back, starts[3], plans[3].n_points, &mut next)?;
        hash_section_u::<G2Affine>(&mut back, starts[4], plans[4].n_points, &mut next)?;
    }
    let next_challenge = next.finalize();

    let record = PtauContribution {
        tau_g1,
        tau_g2,
        alpha_g1,
        beta_g1,
        beta_g2,
        pubkeys,
        partial_hash,
        next_challenge,
        kind,
        params,
    };
    w.start_section(ptau::S_CONTRIBUTIONS)?;
    w.write_u32(n_prior + 1)?;
    w.write_bytes(prior_records)?;
    w.write_bytes(&record.encode())?;
    w.end_section()?;
    w.finish()?;

    Ok(Phase1Report {
        index: n_prior as usize + 1,
        kind,
        response_hash,
        next_challenge,
    })
}

/// `powersoftau contribute`: draw tau, alpha and beta from the contributor's entropy,
/// rescale every point section, and append a record.
///
/// The RNG draw order is load-bearing and is `Fr, Fr, Fr, G1, G1, G1`: the three private
/// scalars in tau/alpha/beta order, then one `g1_s` per key in the same order
/// (`keypair.js:61-74`).
pub fn contribute(
    ptau_in: &Path,
    ptau_out: &Path,
    name: Option<&str>,
    entropy: &str,
    key: &dyn KeyScale,
) -> Result<Phase1Report, CeremonyError> {
    let params = ContributionParams {
        name: name.map(str::to_owned),
        ..Default::default()
    };
    contribute_with(ptau_in, ptau_out, params, rng_from_entropy(entropy), key)
}

/// [`contribute`] with the RNG supplied, so a run can be held against snarkjs' bytes.
/// `powersoftau contribute` mixes 64 OS-random bytes into its seed and is deliberately not
/// reproducible otherwise; [`crate::transcript::rng_from_entropy_with`] is the hook.
pub fn contribute_with(
    ptau_in: &Path,
    ptau_out: &Path,
    params: ContributionParams,
    mut rng: CeremonyRng,
    key: &dyn KeyScale,
) -> Result<Phase1Report, CeremonyError> {
    apply_contribution(
        ptau_in,
        ptau_out,
        ContributionKind::Contribute,
        params,
        |challenge| crate::transcript::create_ptau_key(&mut rng, challenge),
        key,
    )
}

/// `powersoftau beacon`: the same contribution with a key derived from a public beacon,
/// and the only part of a ceremony anyone can reproduce.
pub fn beacon(
    ptau_in: &Path,
    ptau_out: &Path,
    name: Option<&str>,
    beacon_hash: &[u8],
    num_iterations_exp: u8,
    key: &dyn KeyScale,
) -> Result<Phase1Report, CeremonyError> {
    check_beacon(beacon_hash, num_iterations_exp)?;
    let params = ContributionParams {
        name: name.map(str::to_owned),
        num_iterations_exp: Some(num_iterations_exp),
        beacon_hash: Some(beacon_hash.to_vec()),
    };
    let mut rng = rng_from_beacon_params(beacon_hash, num_iterations_exp);
    apply_contribution(
        ptau_in,
        ptau_out,
        ContributionKind::Beacon,
        params,
        |challenge| crate::transcript::create_ptau_key(&mut rng, challenge),
        key,
    )
}

/// What `powersoftau verify` checked.
#[derive(Clone, Debug)]
pub struct PtauVerifyReport {
    pub power: u32,
    pub ceremony_power: u32,
    pub prepared: bool,
    /// One response hash per contribution, oldest first.
    pub contribution_hashes: Vec<Digest>,
    /// False when `power != ceremonyPower`, where the final next-challenge comparison is
    /// skipped because the points in the file are a prefix of the ones that were hashed
    /// (`powersoftau_verify.js:247-252`).
    pub next_challenge_checked: bool,
}

/// Points per chunk in every `verify` loop, `MAX_CHUNK_SIZE` (`powersoftau_verify.js:339`).
/// Nothing depends on it: no partial hash is snapshotted here.
const VERIFY_CHUNK: usize = 1 << 16;

/// One link of the chain, as the checks see it. Contribution #1's predecessor is the
/// unmodified accumulator, whose five points are the generators and whose "next
/// challenge" is the first challenge hash (`powersoftau_verify.js:139-148`).
struct Predecessor {
    tau_g1: G1Affine,
    tau_g2: G2Affine,
    alpha_g1: G1Affine,
    beta_g1: G1Affine,
    beta_g2: G2Affine,
    next_challenge: Digest,
}

impl Predecessor {
    /// `initialContribution`. Note it seeds from `ceremonyPower`, not `power`: a truncated
    /// file's chain was built over the full ceremony
    /// (`powersoftau_verify.js:146`).
    fn initial(ceremony_power: u32) -> Self {
        Self {
            tau_g1: G1Affine::generator(),
            tau_g2: G2Affine::generator(),
            alpha_g1: G1Affine::generator(),
            beta_g1: G1Affine::generator(),
            beta_g2: G2Affine::generator(),
            next_challenge: first_challenge_hash(ceremony_power),
        }
    }

    fn of(c: &PtauContribution) -> Self {
        Self {
            tau_g1: c.tau_g1,
            tau_g2: c.tau_g2,
            alpha_g1: c.alpha_g1,
            beta_g1: c.beta_g1,
            beta_g2: c.beta_g2,
            next_challenge: c.next_challenge,
        }
    }
}

fn failed(what: impl Into<String>) -> CeremonyError {
    CeremonyError::Verification(what.into())
}

/// `verifyContribution` (`powersoftau_verify.js:29-127`): a beacon's key has to be the one
/// its own parameters produce, every key has to satisfy its own same-ratio pair, and each
/// of the five stored points has to be the predecessor's scaled by the committed exponent.
///
/// `g2_sp` is recomputed here rather than taken from the record, because it is not stored:
/// it is `getG2sp(personalization, prevChallenge, g1_s, g1_sx)`, which is the only thing
/// binding a contribution to its predecessor's challenge (`:73-75`).
fn verify_contribution(
    index: usize,
    cur: &PtauContribution,
    prev: &Predecessor,
) -> Result<(), CeremonyError> {
    let named = |what: &str| format!("{what} in contribution #{index}");

    if cur.kind == ContributionKind::Beacon {
        let exp = cur
            .params
            .num_iterations_exp
            .ok_or_else(|| failed(named("beacon record has no numIterationsExp")))?;
        let hash = cur
            .params
            .beacon_hash
            .as_deref()
            .ok_or_else(|| failed(named("beacon record has no beaconHash")))?;
        check_beacon(hash, exp)?;
        let mut rng = rng_from_beacon_params(hash, exp);
        let expected = crate::transcript::create_ptau_key(&mut rng, &prev.next_challenge);
        for (name, got, want) in [
            ("tau", cur.pubkeys.tau, expected.tau.pubkey),
            ("alpha", cur.pubkeys.alpha, expected.alpha.pubkey),
            ("beta", cur.pubkeys.beta, expected.beta.pubkey),
        ] {
            if got.g1_s != want.g1_s || got.g1_sx != want.g1_sx || got.g2_spx != want.g2_spx {
                return Err(failed(named(&format!(
                    "BEACON key ({name}) is not generated correctly"
                ))));
            }
        }
    }

    // The stored `g2_sp` came from whatever challenge the reader guessed; this is the one
    // the chain actually commits to.
    let bound = |k: &PtauPubKey, personalization: u8| PtauPubKey {
        g2_sp: get_g2_sp(personalization, &prev.next_challenge, &k.g1_s, &k.g1_sx),
        ..*k
    };
    let tau = bound(&cur.pubkeys.tau, PERSONALIZATION_TAU);
    let alpha = bound(&cur.pubkeys.alpha, PERSONALIZATION_ALPHA);
    let beta = bound(&cur.pubkeys.beta, PERSONALIZATION_BETA);

    for (name, k) in [("tau", &tau), ("alpha", &alpha), ("beta", &beta)] {
        if !same_ratio(&k.g1_s, &k.g1_sx, &k.g2_sp, &k.g2_spx) {
            return Err(failed(named(&format!("INVALID key ({name})"))));
        }
    }

    // Each of these says "the same exponent that the key commits to was applied to the
    // accumulator". The two `sameRatio` argument orders are not interchangeable: the G1
    // pair carries the ratio for tauG1, alphaG1 and betaG1, and the G2 pair carries it for
    // tauG2 and betaG2 (`powersoftau_verify.js:95-123`).
    let checks: [(&str, bool); 5] = [
        (
            "INVALID tau*G1: it does not follow the previous contribution",
            same_ratio(&prev.tau_g1, &cur.tau_g1, &tau.g2_sp, &tau.g2_spx),
        ),
        (
            "INVALID tau*G2: it does not follow the previous contribution",
            same_ratio(&tau.g1_s, &tau.g1_sx, &prev.tau_g2, &cur.tau_g2),
        ),
        (
            "INVALID alpha*G1: it does not follow the previous contribution",
            same_ratio(&prev.alpha_g1, &cur.alpha_g1, &alpha.g2_sp, &alpha.g2_spx),
        ),
        (
            "INVALID beta*G1: it does not follow the previous contribution",
            same_ratio(&prev.beta_g1, &cur.beta_g1, &beta.g2_sp, &beta.g2_spx),
        ),
        (
            "INVALID beta*G2: it does not follow the previous contribution",
            same_ratio(&beta.g1_s, &beta.g1_sx, &prev.beta_g2, &cur.beta_g2),
        ),
    ];
    for (message, ok) in checks {
        if !ok {
            return Err(failed(named(message)));
        }
    }
    Ok(())
}

/// What one point section's random linear combination produced.
struct PowersCheck<C: PtauGroup> {
    /// `sum r_i * P_i` over `i < n-1`.
    r1: C::Group,
    /// The same scalars against `P_{i+1}`, so `sameRatio(R1, R2, ...)` proves the whole
    /// section is a geometric sequence in one pairing rather than `n`.
    r2: C::Group,
    /// The points at the indexes the contribution record is supposed to have kept.
    singular: Vec<C>,
}

/// `processSection` (`powersoftau_verify.js:338-395`): absorb the section's uncompressed
/// form into the next-challenge hash and build the two-sided random combination that
/// proves the powers step by a single exponent.
///
/// The cross-chunk link is not decoration. `r1`/`r2` inside a chunk only relate that
/// chunk's own points, so a fresh scalar ties the last point of one chunk to the first of
/// the next; without it a file could restart the sequence at every chunk boundary.
fn verify_powers<C: PtauGroup>(
    file: &Ptau,
    id: u32,
    n_points: usize,
    singular_indexes: &[usize],
    rng: &mut CeremonyRng,
    hasher: &mut Transcript,
    msm: &dyn MsmBackend,
) -> Result<PowersCheck<C>, CeremonyError> {
    let mut r1 = C::Group::zero();
    let mut r2 = C::Group::zero();
    let mut singular = Vec::new();
    let mut last_base: Option<C> = None;
    let mut u = vec![0u8; VERIFY_CHUNK.min(n_points.max(1)) * C::SG];

    let mut done = 0usize;
    while done < n_points {
        let n = (n_points - done).min(VERIFY_CHUNK);
        let bases = C::read_points(file, id, done, n)?;

        let u_out = &mut u[..n * C::SG];
        bases
            .par_iter()
            .zip(u_out.par_chunks_exact_mut(C::SG))
            .for_each(|(p, dst)| p.write_uncompressed(dst));
        hasher.update(u_out);

        // 32-bit scalars, the width `misc.getRandomBytes(4*(n-1))` gives multiExpAffine.
        let scalars: Vec<Fr> = (0..n.saturating_sub(1))
            .map(|_| Fr::from(rng.next_u32()))
            .collect();
        if let Some(last) = last_base {
            let r = Fr::from(rng.next_u32());
            r1 += last * r;
            r2 += bases[0] * r;
        }
        if n > 1 {
            r1 += C::msm(msm, &bases[..n - 1], &scalars);
            r2 += C::msm(msm, &bases[1..], &scalars);
        }
        last_base = Some(bases[n - 1]);

        for &sp in singular_indexes {
            if sp >= done && sp < done + n {
                singular.push(bases[sp - done]);
            }
        }
        done += n;
    }
    Ok(PowersCheck { r1, r2, singular })
}

/// `verifyLagrangeEvaluations` (`powersoftau_verify.js:398-489`): the prepared section is
/// the group iFFT of the raw one, so for a random `r`, `<raw, r>` must equal
/// `<lagrange, fft(r)>`.
///
/// That identity holds for any consistent root of unity, since the DFT matrix is symmetric
/// and `iFFT^T * fft` is the identity, and it needs no group FFT on the verifier's side:
/// the transform moves onto the scalars, where the existing `g16-ntt` can do it.
///
/// The `power+1` block of section 12 is the exception the whole section table turns on.
/// tauG1 holds `2n-1` points, so the top block's last input is the point at infinity and
/// its scalar is forced to zero (`:428-433`, `:444-446`); reading `2n` points from a
/// `2n-1` point section would run past the end.
fn verify_lagrange<C: PtauGroup>(
    file: &Ptau,
    tau_section: u32,
    lagrange_section: u32,
    power: u32,
    seed: [u32; 8],
    msm: &dyn MsmBackend,
    ntt: &CpuNtt,
) -> Result<(), CeremonyError> {
    let top = if tau_section == ptau::S_TAU_G1 {
        power + 1
    } else {
        power
    };
    for p in 0..=top {
        let n = 1usize << p;
        let padded = tau_section == ptau::S_TAU_G1 && p == power + 1;

        let mut scalars: Vec<Fr> = {
            let mut rng = CeremonyRng::from_seed_words(seed);
            (0..n)
                .map(|i| {
                    if padded && i == n - 1 {
                        Fr::zero()
                    } else {
                        Fr::from(rng.next_u32())
                    }
                })
                .collect()
        };

        let mut bases = C::read_points(file, tau_section, 0, if padded { n - 1 } else { n })?;
        if padded {
            bases.push(C::zero());
        }
        let raw = C::msm(msm, &bases, &scalars);

        // `Fr.fft` over the same random vector. snarkjs re-seeds the RNG to rebuild it
        // rather than keeping it, which is why the zero at the top of the padded block has
        // to be reproduced in both copies.
        let domain = Domain::new(n)
            .map_err(|e| CeremonyError::malformed(lagrange_section, e.to_string()))?;
        ntt.ntt(&domain, &mut scalars, Direction::Forward);

        // Block `p` starts at element `2^p - 1`, the same seek `powersoftau_verify.js:478`
        // makes.
        let block = C::read_points(file, lagrange_section, n - 1, n)?;
        let transformed = C::msm(msm, &block, &scalars);

        if raw != transformed {
            return Err(failed(format!(
                "phase2 calculation of section {lagrange_section} does not match the powers of tau at 2^{p}"
            )));
        }
    }
    Ok(())
}

/// `powersoftau verify`: walk the contribution chain, check each key's same-ratio pairs
/// against the recomputed `g2_sp`, check the stored generators at the head of each
/// section, and recompute the final next-challenge hash.
///
/// The order is snarkjs': the last contribution first, because everything else in the file
/// is checked against it, then the powers, then the earlier records
/// (`powersoftau_verify.js:129-285`).
pub fn verify(ptau: &Path) -> Result<PtauVerifyReport, CeremonyError> {
    let file = Ptau::open(ptau)?;
    let header = *file.header();
    let contributions = file.contributions()?;
    if contributions.is_empty() {
        return Err(failed(
            "this file has no contribution, it cannot be used in production",
        ));
    }
    let n = 1usize << header.power;
    let msm = CpuMsm::new();
    let ntt = CpuNtt::new();
    let last = contributions.len() - 1;

    let predecessor = |i: usize| match i {
        0 => Predecessor::initial(header.ceremony_power),
        _ => Predecessor::of(&contributions[i - 1]),
    };

    verify_contribution(last + 1, &contributions[last], &predecessor(last))?;

    // The next-challenge hash is rebuilt from the file's own bytes, seeded with the last
    // contribution's reconstructed response hash.
    let cur = &contributions[last];
    let mut hasher = Transcript::new();
    hasher.update(&cur.response_hash()?);
    let mut rng = rng_from_entropy("g16 ptau verify");

    let tau1 = verify_powers::<G1Affine>(
        &file,
        ptau::S_TAU_G1,
        n * 2 - 1,
        &[0, 1],
        &mut rng,
        &mut hasher,
        &msm,
    )?;
    if !same_ratio(
        &tau1.r1.into_affine(),
        &tau1.r2.into_affine(),
        &G2Affine::generator(),
        &cur.tau_g2,
    ) {
        return Err(failed("tauG1 section: powers do not match"));
    }
    if tau1.singular[0] != G1Affine::generator() {
        return Err(failed(
            "first element of the tau*G1 section must be the generator",
        ));
    }
    if tau1.singular[1] != cur.tau_g1 {
        return Err(failed(
            "second element of the tau*G1 section does not match the contribution section",
        ));
    }

    let tau2 = verify_powers::<G2Affine>(
        &file,
        ptau::S_TAU_G2,
        n,
        &[0, 1],
        &mut rng,
        &mut hasher,
        &msm,
    )?;
    if !same_ratio(
        &G1Affine::generator(),
        &cur.tau_g1,
        &tau2.r1.into_affine(),
        &tau2.r2.into_affine(),
    ) {
        return Err(failed("tauG2 section: powers do not match"));
    }
    if tau2.singular[0] != G2Affine::generator() {
        return Err(failed(
            "first element of the tau*G2 section must be the generator",
        ));
    }
    if tau2.singular[1] != cur.tau_g2 {
        return Err(failed(
            "second element of the tau*G2 section does not match the contribution section",
        ));
    }

    for (id, head, what) in [
        (ptau::S_ALPHA_TAU_G1, cur.alpha_g1, "alpha*tau*G1"),
        (ptau::S_BETA_TAU_G1, cur.beta_g1, "beta*tau*G1"),
    ] {
        let r = verify_powers::<G1Affine>(&file, id, n, &[0], &mut rng, &mut hasher, &msm)?;
        if !same_ratio(
            &r.r1.into_affine(),
            &r.r2.into_affine(),
            &G2Affine::generator(),
            &cur.tau_g2,
        ) {
            return Err(failed(format!("{what} section: powers do not match")));
        }
        if r.singular[0] != head {
            return Err(failed(format!(
                "first element of the {what} section does not match the contribution section"
            )));
        }
    }

    // Section 6 is one point, so it gets no combination, only the hash and the comparison.
    let beta_g2 = file.g2_points(ptau::S_BETA_G2, 0, 1)?[0];
    hasher.update(&g2_uncompressed(&beta_g2));
    if beta_g2 != cur.beta_g2 {
        return Err(failed(
            "the betaG2 section does not match the contribution section",
        ));
    }

    // Skipped on a truncated file, whose points are a prefix of the ones that were hashed
    // (`powersoftau_verify.js:247-252`).
    let next_challenge_checked = !header.is_truncated_ceremony();
    if next_challenge_checked && hasher.finalize() != cur.next_challenge {
        return Err(failed(
            "hash of the values does not match the next challenge of the last contributor",
        ));
    }

    for i in (0..last).rev() {
        verify_contribution(i + 1, &contributions[i], &predecessor(i))?;
    }

    let prepared = file.is_prepared();
    if prepared {
        // One seed for all four sections and every power, the way snarkjs draws it once
        // outside the loop (`powersoftau_verify.js:404-407`).
        let mut seed = [0u32; 8];
        for w in seed.iter_mut() {
            *w = rng.next_u32();
        }
        verify_lagrange::<G1Affine>(
            &file,
            ptau::S_TAU_G1,
            ptau::S_LAGRANGE_TAU_G1,
            header.power,
            seed,
            &msm,
            &ntt,
        )?;
        verify_lagrange::<G2Affine>(
            &file,
            ptau::S_TAU_G2,
            ptau::S_LAGRANGE_TAU_G2,
            header.power,
            seed,
            &msm,
            &ntt,
        )?;
        verify_lagrange::<G1Affine>(
            &file,
            ptau::S_ALPHA_TAU_G1,
            ptau::S_LAGRANGE_ALPHA_TAU_G1,
            header.power,
            seed,
            &msm,
            &ntt,
        )?;
        verify_lagrange::<G1Affine>(
            &file,
            ptau::S_BETA_TAU_G1,
            ptau::S_LAGRANGE_BETA_TAU_G1,
            header.power,
            seed,
            &msm,
            &ntt,
        )?;
    }

    let contribution_hashes = contributions
        .iter()
        .map(|c| c.response_hash())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PtauVerifyReport {
        power: header.power,
        ceremony_power: header.ceremony_power,
        prepared,
        contribution_hashes,
        next_challenge_checked,
    })
}

/// The two bounds snarkjs enforces on a beacon before it opens a file
/// (`powersoftau_beacon.js:26-42`).
fn check_beacon(beacon_hash: &[u8], num_iterations_exp: u8) -> Result<(), CeremonyError> {
    if beacon_hash.is_empty() {
        return Err(CeremonyError::BadParams(
            "invalid beacon hash: it must be a valid hexadecimal sequence".into(),
        ));
    }
    if beacon_hash.len() > BEACON_HASH_MAX_BYTES {
        return Err(CeremonyError::BadParams(format!(
            "maximum length of beacon hash is {BEACON_HASH_MAX_BYTES} bytes, got {}",
            beacon_hash.len()
        )));
    }
    if !(BEACON_ITERATIONS_MIN..=BEACON_ITERATIONS_MAX).contains(&num_iterations_exp) {
        return Err(CeremonyError::BadParams(format!(
            "invalid numIterationsExp: must be between {BEACON_ITERATIONS_MIN} and {BEACON_ITERATIONS_MAX}, got {num_iterations_exp}"
        )));
    }
    Ok(())
}

/// Parse and bounds-check the two beacon arguments the way snarkjs does before it touches
/// a file: a non-empty, even-length hex string of at most 255 bytes, and an exponent in
/// `10..=63`.
///
/// snarkjs decodes with a `[\da-f]{2}/gi` scan and then asserts `bytes*2 == str.length`
/// (`powersoftau_beacon.js:27-33`), which makes the accepted set exactly "even-length,
/// all hex". A `0x` prefix is stripped before the scan but not before the length check
/// (`misc.js:230-236`), so `0x1234` is **rejected**; we reject it too rather than accept a
/// beacon snarkjs' own verifier would then be asked to reproduce from different bytes.
pub fn parse_beacon_args(
    beacon_hash_hex: &str,
    num_iterations_exp: &str,
) -> Result<(Vec<u8>, u8), CeremonyError> {
    let bad = || {
        CeremonyError::BadParams(format!(
            "invalid beacon hash `{beacon_hash_hex}`: it must be a valid hexadecimal sequence"
        ))
    };
    if beacon_hash_hex.is_empty() || !beacon_hash_hex.len().is_multiple_of(2) {
        return Err(bad());
    }
    let mut bytes = Vec::with_capacity(beacon_hash_hex.len() / 2);
    for pair in beacon_hash_hex.as_bytes().chunks_exact(2) {
        let s = std::str::from_utf8(pair).map_err(|_| bad())?;
        bytes.push(u8::from_str_radix(s, 16).map_err(|_| bad())?);
    }
    // `parseInt` would take the leading digits of "12abc"; a silently different exponent
    // is a silently different beacon, so this is strict where snarkjs is lax.
    let exp: u8 = num_iterations_exp.parse().map_err(|_| {
        CeremonyError::BadParams(format!(
            "invalid numIterationsExp `{num_iterations_exp}`: must be between {BEACON_ITERATIONS_MIN} and {BEACON_ITERATIONS_MAX}"
        ))
    })?;
    check_beacon(&bytes, exp)?;
    Ok((bytes, exp))
}
