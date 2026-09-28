//! Phase 1 without a `.ptau` in the contributor's hands: `powersoftau export challenge`,
//! `powersoftau challenge contribute` and `powersoftau import response`.
//!
//! This is the Zcash-style exchange. The coordinator exports a challenge, a contributor
//! who never sees the `.ptau` turns it into a response, and the coordinator imports that
//! response as a new contribution record. Both files are headerless concatenations in the
//! same section order as the `.ptau` (2, 3, 4, 5, 6), and they use the two encodings the
//! ptau body never does:
//!
//! * a **challenge** is the previous response hash, then every point **uncompressed**
//!   (non-Montgomery big-endian, G2 `c1` first), so its BLAKE2b-512 digest is by
//!   construction the `nextChallenge` the ptau already stores
//!   (`powersoftau_export_challenge.js:37-57`);
//! * a **response** is the hash of the whole challenge file, then every point
//!   **compressed**, then the 768-byte pubkey uncompressed
//!   (`powersoftau_challenge_contribute.js:93-109`). Its digest is the contribution's
//!   response hash.
//!
//! The contribution arithmetic, the key and the RNG are [`crate::phase1`]'s and
//! [`crate::transcript`]'s; what is new here is framing and three details of `import`:
//!
//! * the response hash absorbs the compressed points in `import`'s chunking, not
//!   `contribute`'s: `floor(2^24 / sG)` points per update when importing the points and
//!   `floor(2^24 / scG)` with `-nopoints` (`powersoftau_import.js:148`, `:184`). The
//!   digest does not care, but `partialHash` stores the block buffer's stale tail, so the
//!   `.ptau` bytes do (see [`crate::transcript::Transcript::update`]);
//! * `-nopoints` writes only sections 1 and 7 and stores a `nextChallenge` of 64 `0xff`
//!   bytes. The next import onto such a file adopts whatever previous hash the response
//!   claims and **rewrites the previous record's `nextChallenge`** with it
//!   (`powersoftau_import.js:68-71`), so section 7 is patched rather than copied verbatim;
//! * the header is rewritten with `ceremonyPower = power` (`writePTauHeader` with no third
//!   argument, `:64`), so importing onto a truncated file loses its ceremony power, as it
//!   does in snarkjs.
//!
//! `-nocheck` changes nothing in 0.7.6: `cli.js:782-785` consults it only after an import
//! that returned a falsy challenge, which never happens, and then reaches a `// TODO
//! Verify`. There is no parameter for it here.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use ark_ec::short_weierstrass::Affine;
use blake2::{Blake2b512, Digest as _};
use rayon::prelude::*;
use snarkrs_field::{g1, g2, Field, Fr, G1Affine, G2Affine};
use snarkrs_msm::KeyScale;

use crate::phase1::{self, Phase1Report, PtauGroup, PTAU_NEW_SECTIONS};
use crate::ptau::{self, Ptau, PtauContribution, PtauHeader, CONTRIBUTION_PREFIX_BYTES};
use crate::transcript::{
    self, g1_from_compressed, g1_from_uncompressed, g2_from_compressed, g2_from_uncompressed,
    get_g2_sp, CeremonyRng, Digest, PtauPubKey, PtauPubKeys, Transcript, PERSONALIZATION_ALPHA,
    PERSONALIZATION_BETA, PERSONALIZATION_TAU, PTAU_PUBKEY_BYTES,
};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionKind, ContributionParams, SCG1, SCG2, SG1, SG2};

/// The `nextChallenge` a `-nopoints` import stores, since without the points there is
/// nothing to hash (`powersoftau_import.js:28-29`).
pub const NO_HASH: Digest = [0xff; 64];

/// `nSections` of a `-nopoints` import: the header and the contributions, nothing else
/// (`powersoftau_import.js:63`).
pub const PTAU_NOPOINTS_SECTIONS: u32 = 2;

/// Bytes of the digest that opens both files.
const HASH_BYTES: usize = 64;

/// Offset of `nextChallenge` inside a stored record: it sits just before `type` and
/// `paramLength`, the record's last two fixed u32s.
const NEXT_CHALLENGE_OFFSET: usize = CONTRIBUTION_PREFIX_BYTES - 8 - HASH_BYTES;

/// Points per chunk when streaming a challenge or response. Only digests are taken over
/// these, so the size is a buffering choice; it is `floor(2^20 / sG)`, a megabyte of
/// uncompressed points.
const fn stream_chunk(sg: usize) -> usize {
    (1 << 20) / sg
}

/// The two parsers a response and a challenge need on top of [`PtauGroup`]'s encoders.
trait WireGroup: PtauGroup {
    fn from_uncompressed(bytes: &[u8]) -> Result<Self, CeremonyError>;
    fn from_compressed(bytes: &[u8]) -> Result<Self, CeremonyError>;
}

// `Affine<Config>` rather than the aliases, for the coherence reason given at
// `PtauGroup`'s impls in `phase1`.
impl WireGroup for Affine<g1::Config> {
    fn from_uncompressed(bytes: &[u8]) -> Result<Self, CeremonyError> {
        g1_from_uncompressed(bytes)
    }
    fn from_compressed(bytes: &[u8]) -> Result<Self, CeremonyError> {
        g1_from_compressed(bytes)
    }
}

impl WireGroup for Affine<g2::Config> {
    fn from_uncompressed(bytes: &[u8]) -> Result<Self, CeremonyError> {
        g2_from_uncompressed(bytes)
    }
    fn from_compressed(bytes: &[u8]) -> Result<Self, CeremonyError> {
        g2_from_compressed(bytes)
    }
}

/// What `export challenge` read out of the ptau.
#[derive(Clone, Debug)]
pub struct ExportedChallenge {
    /// The first 64 bytes of the challenge: the last contribution's response hash, or
    /// `blake2b512("")` for a fresh file. snarkjs prints it as "Last Response Hash".
    pub last_response_hash: Digest,
    /// The challenge file's digest, equal to the ptau's `nextChallenge`. snarkjs prints it
    /// as "New Challenge Hash".
    pub challenge_hash: Digest,
}

/// Write the challenge for the next contribution to `ptau`. Backs `snarkjs powersoftau
/// export challenge`.
///
/// Fails, after writing the file as snarkjs does, when the digest of what was written is
/// not the `nextChallenge` the ptau declares: the points were changed after the record
/// that hashed them.
pub fn export_challenge(ptau: &Path, out: &Path) -> Result<ExportedChallenge, CeremonyError> {
    let file = Ptau::open(ptau)?;
    let n = file.domain();
    let contributions = file.contributions()?;
    let (last_response_hash, declared) = match contributions.last() {
        None => (
            transcript::blake2b512(b""),
            phase1::first_challenge_hash(file.power()),
        ),
        Some(c) => (c.response_hash()?, c.next_challenge),
    };

    let mut w = BufWriter::new(File::create(out)?);
    let mut h = Blake2b512::new();
    w.write_all(&last_response_hash)?;
    h.update(last_response_hash);
    export_section::<G1Affine>(&file, ptau::S_TAU_G1, n * 2 - 1, &mut w, &mut h)?;
    export_section::<G2Affine>(&file, ptau::S_TAU_G2, n, &mut w, &mut h)?;
    export_section::<G1Affine>(&file, ptau::S_ALPHA_TAU_G1, n, &mut w, &mut h)?;
    export_section::<G1Affine>(&file, ptau::S_BETA_TAU_G1, n, &mut w, &mut h)?;
    export_section::<G2Affine>(&file, ptau::S_BETA_G2, 1, &mut w, &mut h)?;
    w.flush()?;
    let challenge_hash: Digest = h.finalize().into();

    if challenge_hash != declared {
        return Err(CeremonyError::Verification(
            "ptau file is corrupted: the calculated challenge hash does not match the declared one"
                .into(),
        ));
    }
    Ok(ExportedChallenge {
        last_response_hash,
        challenge_hash,
    })
}

/// One ptau section, LEM on disk, written and hashed uncompressed.
fn export_section<C: WireGroup>(
    file: &Ptau,
    id: u32,
    n_points: usize,
    w: &mut impl Write,
    h: &mut Blake2b512,
) -> Result<(), CeremonyError> {
    let chunk = stream_chunk(C::SG).min(n_points.max(1));
    let mut u = vec![0u8; chunk * C::SG];
    let mut done = 0usize;
    while done < n_points {
        let n = (n_points - done).min(chunk);
        let lem = file.section_elements(id, C::SG, done, n)?;
        let out = &mut u[..n * C::SG];
        lem.par_chunks_exact(C::SG)
            .zip(out.par_chunks_exact_mut(C::SG))
            .try_for_each(|(src, dst)| -> Result<(), CeremonyError> {
                C::from_lem(src)?.write_uncompressed(dst);
                Ok(())
            })?;
        w.write_all(out)?;
        h.update(&*out);
        done += n;
    }
    Ok(())
}

/// What `challenge contribute` produced.
#[derive(Clone, Debug)]
pub struct ChallengeResponse {
    pub power: u32,
    /// The challenge's first 64 bytes, "Claimed Previous Response Hash".
    pub claimed_previous_response: Digest,
    /// The whole challenge file's digest, "Current Challenge Hash", and the first 64
    /// bytes of the response.
    pub challenge_hash: Digest,
    /// "Contribution Response Hash", the value the contributor publishes.
    pub response_hash: Digest,
}

/// The power a challenge of `size` bytes was exported at, or `None` when no power gives
/// that size: `64 + (2n-1) sG1 + n sG2 + 2n sG1 + sG2`, solved for `n`
/// (`powersoftau_challenge_contribute.js:49-59`).
pub fn challenge_power(size: u64) -> Option<u32> {
    let per = (4 * SG1 + SG2) as u64;
    let num = (size + SG1 as u64).checked_sub((HASH_BYTES + SG2) as u64)?;
    if num % per != 0 {
        return None;
    }
    let n = num / per;
    n.is_power_of_two().then(|| n.trailing_zeros())
}

/// Bytes of a response at `power`, the size `import response` insists on
/// (`powersoftau_import.js:45-53`).
pub fn response_bytes(power: u32) -> u64 {
    let n = 1u64 << power;
    (HASH_BYTES + PTAU_PUBKEY_BYTES + SCG2) as u64
        + (2 * n - 1) * SCG1 as u64
        + n * SCG2 as u64
        + 2 * n * SCG1 as u64
}

/// Contribute to the challenge at `challenge` with the contributor's entropy, writing the
/// response to `response`. Backs `snarkjs powersoftau challenge contribute`.
pub fn challenge_contribute(
    challenge: &Path,
    response: &Path,
    entropy: &str,
    scale: &dyn KeyScale,
) -> Result<ChallengeResponse, CeremonyError> {
    challenge_contribute_with(
        challenge,
        response,
        transcript::rng_from_entropy(entropy)?,
        scale,
    )
}

/// [`challenge_contribute`] with the RNG supplied, so a run can be held against snarkjs'
/// bytes. Backs `snarkjs powersoftau challenge contribute`.
pub fn challenge_contribute_with(
    challenge: &Path,
    response: &Path,
    mut rng: CeremonyRng,
    scale: &dyn KeyScale,
) -> Result<ChallengeResponse, CeremonyError> {
    let mut input = File::open(challenge)?;
    let size = input.metadata()?.len();
    let power = challenge_power(size)
        .ok_or_else(|| CeremonyError::BadParams("invalid challenge file size".into()))?;
    phase1::check_power(power)?;
    let n = 1usize << power;

    // The key is bound to the digest of the whole file, so it takes a pass of its own
    // before the first point can be transformed.
    let mut h = Blake2b512::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let got = input.read(&mut buf)?;
        if got == 0 {
            break;
        }
        h.update(&buf[..got]);
    }
    let challenge_hash: Digest = h.finalize().into();
    input.seek(SeekFrom::Start(0))?;
    let mut claimed_previous_response = [0u8; HASH_BYTES];
    input.read_exact(&mut claimed_previous_response)?;

    let key = transcript::create_ptau_key(&mut rng, &challenge_hash);
    let tau = *key.tau.prv_key.expose();
    let alpha = *key.alpha.prv_key.expose();
    let beta = *key.beta.prv_key.expose();

    let mut w = BufWriter::new(File::create(response)?);
    let mut rh = Blake2b512::new();
    w.write_all(&challenge_hash)?;
    rh.update(challenge_hash);
    let mut sections = || -> Result<(), CeremonyError> {
        let io = (&mut input, &mut w, &mut rh);
        apply_key_section::<G1Affine>(io.0, io.1, io.2, n * 2 - 1, Fr::ONE, tau, scale)?;
        apply_key_section::<G2Affine>(io.0, io.1, io.2, n, Fr::ONE, tau, scale)?;
        apply_key_section::<G1Affine>(io.0, io.1, io.2, n, alpha, tau, scale)?;
        apply_key_section::<G1Affine>(io.0, io.1, io.2, n, beta, tau, scale)?;
        apply_key_section::<G2Affine>(io.0, io.1, io.2, 1, beta, tau, scale)?;
        Ok(())
    };
    let result = sections();
    // The copies of the private scalars go whether or not the sections were written.
    for mut s in [tau, alpha, beta] {
        zeroize::Zeroize::zeroize(&mut s);
    }
    result?;

    let pubkey = transcript::write_ptau_pubkey(&key.pubkeys(), false);
    w.write_all(&pubkey)?;
    rh.update(pubkey);
    w.flush()?;
    Ok(ChallengeResponse {
        power,
        claimed_previous_response,
        challenge_hash,
        response_hash: rh.finalize().into(),
    })
}

/// `applyKeyToChallengeSection` with `COMPRESSED` output (`mpc_applykey.js:55-77`): read
/// uncompressed, scale point `i` by `first * inc^i`, write and hash compressed.
fn apply_key_section<C: WireGroup>(
    input: &mut File,
    w: &mut impl Write,
    h: &mut Blake2b512,
    n_points: usize,
    first: Fr,
    inc: Fr,
    scale: &dyn KeyScale,
) -> Result<(), CeremonyError> {
    let chunk = stream_chunk(C::SG).min(n_points.max(1));
    let mut u = vec![0u8; chunk * C::SG];
    let mut c = vec![0u8; chunk * C::SCG];
    let mut t = first;
    let mut done = 0usize;
    while done < n_points {
        let n = (n_points - done).min(chunk);
        let raw = &mut u[..n * C::SG];
        input.read_exact(raw)?;
        let mut points = raw
            .par_chunks_exact(C::SG)
            .map(C::from_uncompressed)
            .collect::<Result<Vec<C>, _>>()?;
        C::apply_key(scale, &mut points, t, inc)?;
        let out = &mut c[..n * C::SCG];
        points
            .par_iter()
            .zip(out.par_chunks_exact_mut(C::SCG))
            .for_each(|(p, dst)| p.write_compressed(dst));
        w.write_all(out)?;
        h.update(&*out);
        done += n;
        t *= inc.pow([n as u64]);
    }
    zeroize::Zeroize::zeroize(&mut t);
    Ok(())
}

/// Points per response-hash update in `import`, which is what fixes `partialHash`'s stale
/// tail: `floor(2^24 / sG)` when the points are imported, `floor(2^24 / scG)` when they
/// are not (`powersoftau_import.js:148`, `:184`).
const fn import_chunk(sg: usize, scg: usize, import_points: bool) -> usize {
    if import_points {
        (1 << 24) / sg
    } else {
        (1 << 24) / scg
    }
}

/// Import the response at `response` onto `ptau_in`, writing `ptau_out` with a new
/// contribution record. Backs `snarkjs powersoftau import response`; `import_points =
/// false` is its `-nopoints`.
pub fn import_response(
    ptau_in: &Path,
    response: &Path,
    ptau_out: &Path,
    name: Option<&str>,
    import_points: bool,
) -> Result<Phase1Report, CeremonyError> {
    let old = Ptau::open(ptau_in)?;
    let power = old.power();
    phase1::check_power(power)?;
    let n = 1usize << power;
    // `if (name)`: an empty name is no name.
    let params = ContributionParams {
        name: name.filter(|s| !s.is_empty()).map(str::to_owned),
        ..Default::default()
    };
    params.encode()?;

    let mut resp = File::open(response)?;
    if resp.metadata()?.len() != response_bytes(power) {
        return Err(CeremonyError::BadParams(
            "size of the contribution is invalid".into(),
        ));
    }

    let (n_prior, mut prior_records, last_record) = prior_records(&old)?;
    let mut last_challenge = old.last_challenge()?;
    let mut previous = [0u8; HASH_BYTES];
    resp.read_exact(&mut previous)?;
    if last_challenge == NO_HASH {
        // Only a `-nopoints` import stores this, and only on the record it appended, so
        // `last_record` is that record.
        last_challenge = previous;
        if let Some(at) = last_record {
            prior_records[at + NEXT_CHALLENGE_OFFSET..at + NEXT_CHALLENGE_OFFSET + HASH_BYTES]
                .copy_from_slice(&previous);
        }
    }
    if previous != last_challenge {
        return Err(CeremonyError::Verification(
            "wrong contribution: this contribution is not based on the previous hash".into(),
        ));
    }

    let mut w = BinFileWriter::create(
        ptau_out,
        ptau::PTAU_MAGIC,
        ptau::PTAU_MAX_VERSION,
        if import_points {
            PTAU_NEW_SECTIONS
        } else {
            PTAU_NOPOINTS_SECTIONS
        },
    )?;
    phase1::write_header(&mut w, power, power)?;

    let mut hasher = Transcript::new();
    hasher.update(&previous);
    let mut io = ImportIo {
        resp: &mut resp,
        w: &mut w,
        hasher: &mut hasher,
        import_points,
    };
    let tau_g1 = io.section::<G1Affine>(ptau::S_TAU_G1, n * 2 - 1, 1)?;
    let tau_g2 = io.section::<G2Affine>(ptau::S_TAU_G2, n, 1)?;
    let alpha_g1 = io.section::<G1Affine>(ptau::S_ALPHA_TAU_G1, n, 0)?;
    let beta_g1 = io.section::<G1Affine>(ptau::S_BETA_TAU_G1, n, 0)?;
    let beta_g2 = io.section::<G2Affine>(ptau::S_BETA_G2, 1, 0)?;

    let partial_hash = hasher.partial_hash();
    let mut key_bytes = [0u8; PTAU_PUBKEY_BYTES];
    resp.read_exact(&mut key_bytes)?;
    hasher.update(&key_bytes);
    let response_hash = hasher.finalize();
    let pubkeys = read_uncompressed_pubkey(&key_bytes, &last_challenge)?;

    let next_challenge = if import_points {
        let mut next = Transcript::new();
        next.update(&response_hash);
        let mut at = ptau::PREAMBLE_BYTES + ptau::SECTION_FRAMING + PtauHeader::BYTES as u64;
        let mut back = File::open(ptau_out)?;
        for (sg, count) in [(SG1, n * 2 - 1), (SG2, n), (SG1, n), (SG1, n), (SG2, 1)] {
            at += ptau::SECTION_FRAMING;
            if sg == SG1 {
                phase1::hash_section_u::<G1Affine>(&mut back, at, count, &mut next)?;
            } else {
                phase1::hash_section_u::<G2Affine>(&mut back, at, count, &mut next)?;
            }
            at += (count * sg) as u64;
        }
        next.finalize()
    } else {
        NO_HASH
    };

    let record = PtauContribution {
        tau_g1,
        tau_g2,
        alpha_g1,
        beta_g1,
        beta_g2,
        pubkeys,
        partial_hash,
        next_challenge,
        kind: ContributionKind::Contribute,
        params,
    };
    w.start_section(ptau::S_CONTRIBUTIONS)?;
    w.write_u32(n_prior + 1)?;
    w.write_bytes(&prior_records)?;
    w.write_bytes(&record.encode())?;
    w.end_section()?;
    w.finish()?;

    Ok(Phase1Report {
        index: n_prior as usize + 1,
        kind: ContributionKind::Contribute,
        response_hash,
        next_challenge,
    })
}

/// Section 7's record count, its records as stored, and the offset of the last record
/// within them. Records are carried forward byte for byte, as `contribute` does, except
/// for the one `nextChallenge` a `-nopoints` predecessor forces [`import_response`] to
/// rewrite.
fn prior_records(old: &Ptau) -> Result<(u32, Vec<u8>, Option<usize>), CeremonyError> {
    let section = old.section(ptau::S_CONTRIBUTIONS)?;
    let bad = |why: &str| CeremonyError::malformed(ptau::S_CONTRIBUTIONS, why.to_owned());
    let count = u32::from_le_bytes(
        section
            .get(..4)
            .ok_or_else(|| bad("section is empty"))?
            .try_into()
            .expect("four bytes"),
    );
    let records = &section[4..];
    let mut at = 0usize;
    let mut last = None;
    for _ in 0..count {
        let head = records
            .get(at..at + CONTRIBUTION_PREFIX_BYTES)
            .ok_or_else(|| bad("record runs past the section"))?;
        let param_len = u32::from_le_bytes(
            head[CONTRIBUTION_PREFIX_BYTES - 4..]
                .try_into()
                .expect("four bytes"),
        ) as usize;
        last = Some(at);
        at += CONTRIBUTION_PREFIX_BYTES + param_len;
    }
    if at != records.len() {
        return Err(bad("records do not fill the section"));
    }
    Ok((count, records.to_vec(), last))
}

/// `fromPtauPubKeyRpr(buff, 0, curve, false)`: the response's nine points, uncompressed.
/// The three `g2_sp` are not in the file and are derived from the challenge the response
/// answers, as a read of the stored record would.
fn read_uncompressed_pubkey(
    bytes: &[u8; PTAU_PUBKEY_BYTES],
    challenge: &Digest,
) -> Result<PtauPubKeys, CeremonyError> {
    let g1 = |i: usize| g1_from_uncompressed(&bytes[i * SG1..(i + 1) * SG1]);
    let g2 = |i: usize| g2_from_uncompressed(&bytes[6 * SG1 + i * SG2..6 * SG1 + (i + 1) * SG2]);
    let key = |g1_s: G1Affine, g1_sx: G1Affine, g2_spx: G2Affine, personalization: u8| PtauPubKey {
        g1_s,
        g1_sx,
        g2_sp: get_g2_sp(personalization, challenge, &g1_s, &g1_sx),
        g2_spx,
    };
    Ok(PtauPubKeys {
        tau: key(g1(0)?, g1(1)?, g2(0)?, PERSONALIZATION_TAU),
        alpha: key(g1(2)?, g1(3)?, g2(1)?, PERSONALIZATION_ALPHA),
        beta: key(g1(4)?, g1(5)?, g2(2)?, PERSONALIZATION_BETA),
    })
}

/// The response being read and the ptau being written, for [`ImportIo::section`].
struct ImportIo<'a> {
    resp: &'a mut File,
    w: &'a mut BinFileWriter,
    hasher: &'a mut Transcript,
    import_points: bool,
}

impl ImportIo<'_> {
    /// `processSection` (`powersoftau_import.js:131-203`): hash the compressed points in
    /// import's chunking, write them LEM when importing points, and return element `keep`
    /// for the record.
    fn section<C: WireGroup>(
        &mut self,
        id: u32,
        n_points: usize,
        keep: usize,
    ) -> Result<C, CeremonyError> {
        let chunk = import_chunk(C::SG, C::SCG, self.import_points);
        let mut comp = vec![0u8; chunk.min(n_points) * C::SCG];
        let mut lem = vec![
            0u8;
            if self.import_points {
                comp.len() * 2
            } else {
                0
            }
        ];
        let mut kept = None;
        if self.import_points {
            self.w.start_section(id)?;
        }
        let mut done = 0usize;
        while done < n_points {
            let n = (n_points - done).min(chunk);
            let buf = &mut comp[..n * C::SCG];
            self.resp.read_exact(buf)?;
            self.hasher.update(buf);
            if self.import_points {
                let points = buf
                    .par_chunks_exact(C::SCG)
                    .map(C::from_compressed)
                    .collect::<Result<Vec<C>, _>>()?;
                let out = &mut lem[..n * C::SG];
                points
                    .par_iter()
                    .zip(out.par_chunks_exact_mut(C::SG))
                    .for_each(|(p, dst)| p.write_lem(dst));
                self.w.write_bytes(out)?;
                if keep >= done && keep < done + n {
                    kept = Some(points[keep - done]);
                }
            } else if keep >= done && keep < done + n {
                let at = (keep - done) * C::SCG;
                kept = Some(C::from_compressed(&buf[at..at + C::SCG])?);
            }
            done += n;
        }
        if self.import_points {
            self.w.end_section()?;
        }
        kept.ok_or_else(|| {
            CeremonyError::malformed(
                id,
                format!("section holds {n_points} points, the record needs element {keep}"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size formula and its inverse agree, and a size between two powers is refused.
    #[test]
    fn challenge_sizes_round_trip_through_the_power() {
        for power in 1..=12 {
            let n = 1u64 << power;
            let size = HASH_BYTES as u64
                + (2 * n - 1) * SG1 as u64
                + n * SG2 as u64
                + 2 * n * SG1 as u64
                + SG2 as u64;
            assert_eq!(challenge_power(size), Some(power));
            assert_eq!(challenge_power(size + 1), None);
        }
        assert_eq!(challenge_power(0), None);
    }

    #[test]
    fn nopoints_marker_sits_where_the_record_keeps_next_challenge() {
        // tauG1, tauG2, alphaG1, betaG1, betaG2, pubkey, partialHash precede it.
        assert_eq!(
            NEXT_CHALLENGE_OFFSET,
            SG1 + SG2 + SG1 + SG1 + SG2 + PTAU_PUBKEY_BYTES + transcript::PARTIAL_HASH_BYTES
        );
    }
}
