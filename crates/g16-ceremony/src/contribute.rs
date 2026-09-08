//! Phase 2: `zkey contribute`, `zkey beacon` and `zkey verify`.
//!
//! A contribution is cheap by construction. Sections 3 to 7 are copied **verbatim**
//! (`zkey_contribute.js:76-88`) and only sections 8 and 9 are rescaled, by a single
//! `invDelta` applied uniformly (`mpc_applykey.js:29-51` starts at `first = invDelta` with
//! `inc = 1`, so the `Fr.exp(inc, n)` in its loop is a no-op). The header's `vk_delta_1`
//! and `vk_delta_2` are multiplied by `prvKey`, not divided.
//!
//! `zkey verify` then requires those five sections to be byte-for-byte equal between the
//! init file and the final one (`zkey_verify_frominit.js:169-197`), which is why
//! [`crate::write::BinFileWriter::write_section_verbatim`] exists and why re-encoding a
//! point on the copy path fails a check whose message says nothing about encoding.
//!
//! What `verify` does **not** do is worth stating plainly, because the command name
//! oversells it: it never checks section 4 against the r1cs, and never checks that
//! sections 3 and 5 to 7 are the correct MSMs of any circuit. Those properties come only
//! from regenerating the init file, which is exactly what the r1cs form does
//! (`zkey_verify_fromr1cs.js:26-29` runs a whole `newZKey` first). So
//! [`verify_from_init`] verifies the contribution chain and that the final key is a
//! consistent rescaling of the given init key. It does not verify the circuit.

use std::path::Path;

use g16_field::{
    AffineRepr, CurveGroup, Domain, Field, Fr, G1Affine, G1Projective, G2Affine, Zero,
};
use g16_msm::{KeyScale, MsmBackend};
use g16_ntt::{CpuNtt, Direction, NttBackend};
use rayon::prelude::*;

use crate::ptau::Ptau;
use crate::setup::{
    S_A, S_B1, S_B2, S_C, S_COEFFS, S_H, S_HEADER, S_IC, S_MPC_PARAMS, S_PROTOCOL, ZKEY_MAGIC,
    ZKEY_MAX_VERSION, ZKEY_SECTIONS, ZKEY_VERSION,
};
use crate::transcript::{
    self, create_delta_key, hash_to_g2, same_ratio, CeremonyRng, Digest, Transcript,
};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionKind, ContributionParams, Groth16Header, SG1, SG2};

use g16_zkey::binfile::{self, BinFile, Cursor};

/// Points per pass over a rescaled or checked section, snarkjs' `MAX_CHUNK_SIZE`
/// (`mpc_applykey.js:30`). It bounds the working set at 4 MB of G1 on the way in and the
/// same on the way out, and it is the unit a GPU kernel would take: one launch per chunk,
/// with [`scale_g1`] as the body.
const CHUNK_POINTS: usize = 1 << 16;

/// One record of zkey section 10.
///
/// `g2_sp` is not stored: verification recomputes it as `hashToG2(transcript)`
/// (`zkey_verify_frominit.js:61`), and that recomputation is what binds the published key
/// to the transcript rather than to the contributor's word.
#[derive(Clone, Debug)]
pub struct ZkeyContribution {
    pub delta_after: G1Affine,
    pub g1_s: G1Affine,
    pub g1_sx: G1Affine,
    pub g2_spx: G2Affine,
    pub transcript: Digest,
    pub kind: ContributionKind,
    pub params: ContributionParams,
}

impl ZkeyContribution {
    /// Fixed bytes before the params: `3*sG1 + sG2 + 64 + 4 + 4`.
    pub const PREFIX_BYTES: usize = 3 * crate::SG1 + crate::SG2 + 72;
}

/// zkey section 10.
///
/// A freshly generated zkey has exactly 68 bytes here: the 64-byte `csHash` then a u32
/// zero (`zkey_new.js:168-171`).
#[derive(Clone, Debug)]
pub struct MpcParams {
    pub cs_hash: Digest,
    pub contributions: Vec<ZkeyContribution>,
}

impl MpcParams {
    pub fn read(section: &[u8]) -> Result<Self, CeremonyError> {
        let mut cur = Cursor::new(section, S_MPC_PARAMS);
        let mut cs_hash = [0u8; 64];
        cs_hash.copy_from_slice(cur.take(64)?);
        let n = cur.u32()? as usize;
        // `readContribution` runs off the end of a truncated section rather than checking
        // the count first (`zkey_utils.js:452-486`); the cursor's own bounds are what
        // makes a short section an error here instead of a panic.
        let mut contributions = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let delta_after = binfile::g1(cur.take(SG1)?);
            let g1_s = binfile::g1(cur.take(SG1)?);
            let g1_sx = binfile::g1(cur.take(SG1)?);
            let g2_spx = binfile::g2(cur.take(SG2)?);
            let mut t = [0u8; 64];
            t.copy_from_slice(cur.take(64)?);
            let kind = ContributionKind::from_u32(cur.u32()?)?;
            let param_len = cur.u32()? as usize;
            let params = ContributionParams::decode(cur.take(param_len)?)?;
            contributions.push(ZkeyContribution {
                delta_after,
                g1_s,
                g1_sx,
                g2_spx,
                transcript: t,
                kind,
                params,
            });
        }
        if cur.remaining() != 0 {
            return Err(CeremonyError::malformed(
                S_MPC_PARAMS,
                format!("{} bytes left after {n} contributions", cur.remaining()),
            ));
        }
        Ok(Self {
            cs_hash,
            contributions,
        })
    }

    /// Write section 10 into an already-started section.
    pub fn write(&self, w: &mut BinFileWriter) -> Result<(), CeremonyError> {
        w.write_bytes(&self.cs_hash)?;
        w.write_u32(self.contributions.len() as u32)?;
        for c in &self.contributions {
            w.write_g1(&c.delta_after)?;
            w.write_g1(&c.g1_s)?;
            w.write_g1(&c.g1_sx)?;
            w.write_g2(&c.g2_spx)?;
            w.write_bytes(&c.transcript)?;
            w.write_u32(c.kind.as_u32())?;
            let params = c.params.encode()?;
            w.write_u32(params.len() as u32)?;
            w.write_bytes(&params)?;
        }
        Ok(())
    }
}

/// `hashPubKey` (`zkey_utils.js:558-564`): `delta_after`, `g1_s` and `g1_sx` uncompressed
/// G1, then `g2_spx` uncompressed G2, then the raw 64-byte transcript. 320 bytes, and the
/// unit the accumulated transcript is built from.
pub fn hash_pubkey(t: &mut Transcript, c: &ZkeyContribution) {
    t.update_g1(&c.delta_after);
    t.update_g1(&c.g1_s);
    t.update_g1(&c.g1_sx);
    t.update_g2(&c.g2_spx);
    t.update(&c.transcript);
}

/// This contribution's own hash, `blake2b512(hashPubKey(c))`, the value snarkjs prints for
/// a contributor to publish (`zkey_contribute.js:99-102`).
pub fn contribution_hash(c: &ZkeyContribution) -> Digest {
    let mut t = Transcript::new();
    hash_pubkey(&mut t, c);
    t.finalize()
}

/// The transcript a new contribution starts from: `csHash`, then every prior
/// contribution's 320-byte pubkey block, in file order.
pub fn accumulated_transcript(params: &MpcParams) -> Transcript {
    let mut t = Transcript::new();
    t.update(&params.cs_hash);
    for c in &params.contributions {
        hash_pubkey(&mut t, c);
    }
    t
}

/// What a contribution produced, for the CLI to print.
#[derive(Clone, Debug)]
pub struct ContributionReport {
    pub index: usize,
    pub kind: ContributionKind,
    /// `blake2b512(hashPubKey(c))` for the contribution just made.
    pub hash: Digest,
    pub delta_after: G1Affine,
}

/// `zkey contribute`: draw a delta from the contributor's entropy, rescale sections 8 and
/// 9 by its inverse, and append a record.
///
/// `entropy` is mixed with 64 OS-random bytes, so this is non-deterministic by design and
/// two runs with the same string produce different keys. That is the point of the command.
pub fn contribute(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    entropy: &str,
    key: &dyn KeyScale,
) -> Result<ContributionReport, CeremonyError> {
    let mut rng = transcript::rng_from_entropy(entropy);
    let params = ContributionParams {
        name: name.map(str::to_owned),
        num_iterations_exp: None,
        beacon_hash: None,
    };
    apply(
        zkey_in,
        zkey_out,
        ContributionKind::Contribute,
        params,
        &mut rng,
        key,
    )
}

/// `zkey beacon`: the same contribution with a reproducible key, derived by iterating
/// SHA-256 `2^num_iterations_exp` times over `beacon_hash`.
///
/// `beacon_hash` is raw bytes, not hex: the CLI parses the hex and enforces the bounds
/// snarkjs does (`10 <= num_iterations_exp <= 63`, and a hash of 1 to 255 bytes).
pub fn beacon(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    beacon_hash: &[u8],
    num_iterations_exp: u8,
    key: &dyn KeyScale,
) -> Result<ContributionReport, CeremonyError> {
    let mut rng = transcript::rng_from_beacon_params(beacon_hash, num_iterations_exp);
    let params = ContributionParams {
        name: name.map(str::to_owned),
        num_iterations_exp: Some(num_iterations_exp),
        beacon_hash: Some(beacon_hash.to_vec()),
    };
    apply(
        zkey_in,
        zkey_out,
        ContributionKind::Beacon,
        params,
        &mut rng,
        key,
    )
}

/// The body both commands share. They differ only in where `rng` came from and what goes
/// in the params blob (`zkey_contribute.js:29-107` and `zkey_beacon.js:52-127` are
/// otherwise the same file twice).
fn apply(
    zkey_in: &Path,
    zkey_out: &Path,
    kind: ContributionKind,
    params: ContributionParams,
    rng: &mut CeremonyRng,
    key: &dyn KeyScale,
) -> Result<ContributionReport, CeremonyError> {
    let file = BinFile::open(zkey_in, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let mut header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let mut mpc = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;

    let (delta, digest) = create_delta_key(rng, accumulated_transcript(&mpc));

    // The header points are multiplied by the key, and only sections 8 and 9 by its
    // inverse. Scaling the header by `invDelta` instead produces a file that still parses
    // and fails every ratio check in `zkey verify`.
    header.delta_g1 = (header.delta_g1 * delta.prv_key).into_affine();
    header.delta_g2 = (header.delta_g2 * delta.prv_key).into_affine();

    let contribution = ZkeyContribution {
        delta_after: header.delta_g1,
        g1_s: delta.g1_s,
        g1_sx: delta.g1_sx,
        g2_spx: delta.g2_spx,
        transcript: digest,
        kind,
        params,
    };
    mpc.contributions.push(contribution.clone());

    let inv_delta = delta
        .prv_key
        .inverse()
        .ok_or_else(|| CeremonyError::BadParams("the contribution key drew zero".into()))?;

    // Unlike a fresh zkey, whose sections go out `1, 2, 4, 3, 9, 8, 5, 6, 7, 10`, a
    // contribution writes them in id order: `writeHeader` then five `copySection` calls
    // then two `applyKeyToSection` calls then `writeMPCParams`.
    let mut w = BinFileWriter::create(zkey_out, ZKEY_MAGIC, ZKEY_VERSION, ZKEY_SECTIONS)?;
    w.start_section(S_PROTOCOL)?;
    w.write_u32(crate::PROTOCOL_GROTH16)?;
    w.end_section()?;
    w.start_section(S_HEADER)?;
    header.write(&mut w)?;
    w.end_section()?;
    for id in [S_IC, S_COEFFS, S_A, S_B1, S_B2] {
        w.write_section_verbatim(id, file.unique_section(id)?)?;
    }
    for id in [S_C, S_H] {
        apply_key_to_section(&mut w, file.unique_section(id)?, id, inv_delta, key)?;
    }
    w.start_section(S_MPC_PARAMS)?;
    mpc.write(&mut w)?;
    w.end_section()?;
    w.finish()?;

    Ok(ContributionReport {
        index: mpc.contributions.len(),
        kind,
        hash: contribution_hash(&contribution),
        delta_after: contribution.delta_after,
    })
}

/// Section 1 holds one u32 and nothing else. Checked before the header, because a plonk
/// or fflonk zkey has a different section 2 that would otherwise parse far enough to give
/// a confusing error.
fn check_protocol(file: &BinFile) -> Result<(), CeremonyError> {
    let mut cur = Cursor::new(file.unique_section(S_PROTOCOL)?, S_PROTOCOL);
    let id = cur.u32()?;
    if id != crate::PROTOCOL_GROTH16 {
        return Err(CeremonyError::UnsupportedProtocol(id));
    }
    Ok(())
}

/// `applyKeyToSection` with `inc = 1` (`mpc_applykey.js:29-51`): one constant scalar over
/// the whole array, so the `Fr.exp(inc, n)` at the end of its loop is a no-op and there is
/// no geometric sequence to carry across chunks. Reading it as a running power is the
/// mistake that produces a file whose first point is right and whose second is not.
///
/// It goes through the same [`KeyScale`] seam phase 1 uses, with `inc = 1`, because it is
/// the same loop: 97% of this command is here, and a backend that has the geometric case
/// has this one for free.
fn apply_key_to_section(
    w: &mut BinFileWriter,
    body: &[u8],
    id: u32,
    k: Fr,
    key: &dyn KeyScale,
) -> Result<(), CeremonyError> {
    if !body.len().is_multiple_of(SG1) {
        return Err(CeremonyError::malformed(
            id,
            format!("{} bytes is not a whole number of G1 points", body.len()),
        ));
    }
    w.start_section(id)?;
    for chunk in body.chunks(CHUNK_POINTS * SG1) {
        let mut points: Vec<G1Affine> = chunk.par_chunks_exact(SG1).map(binfile::g1).collect();
        key.apply_key_g1(&mut points, k, Fr::ONE)?;
        w.write_g1_slice(&points)?;
    }
    w.end_section()
}

/// What `zkey verify` checked, in the order it checked it.
#[derive(Clone, Debug)]
pub struct ZkeyVerifyReport {
    /// One hash per contribution, oldest first. snarkjs prints them reversed.
    pub contribution_hashes: Vec<Digest>,
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
}

/// `zkey verify init` (`zkey_verify_frominit.js:32-231`), in six parts: the contribution
/// chain, init-vs-final header agreement, the shared `csHash` and section sizes, byte
/// identity of sections 3 to 7, and a same-ratio check on each of the two rescaled
/// sections.
///
/// The ptau is needed for the last of those: the H check MSMs the ptau differences
/// `tauG1[i + domainSize] - tauG1[i]` against a random scalar vector whose last entry is
/// forced to zero, because the quotient polynomial has degree `domainSize - 2`.
pub fn verify_from_init(
    init_zkey: &Path,
    ptau: &Path,
    final_zkey: &Path,
    msm: &dyn MsmBackend,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    let file = BinFile::open(final_zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let mpc = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;

    let init = BinFile::open(init_zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&init)?;
    let init_header = Groth16Header::read(init.unique_section(S_HEADER)?)?;
    let init_mpc = MpcParams::read(init.unique_section(S_MPC_PARAMS)?)?;

    let contribution_hashes = verify_chain(&mpc, &header)?;

    if (
        init_header.n_vars,
        init_header.n_public,
        init_header.domain_size,
    ) != (header.n_vars, header.n_public, header.domain_size)
    {
        return Err(fail("different circuit parameters"));
    }
    for (ours, theirs, what) in [
        (header.alpha_g1, init_header.alpha_g1, "alpha1"),
        (header.beta_g1, init_header.beta_g1, "beta1"),
    ] {
        if ours != theirs {
            return Err(fail(format!("invalid {what}")));
        }
    }
    for (ours, theirs, what) in [
        (header.beta_g2, init_header.beta_g2, "beta2"),
        (header.gamma_g2, init_header.gamma_g2, "gamma2"),
    ] {
        if ours != theirs {
            return Err(fail(format!("invalid {what}")));
        }
    }
    // `delta_1` is checked against the chain and `delta_2` only against `delta_1`, so a
    // final key whose `delta_2` was left at the generator fails here and not earlier.
    if !same_ratio(
        &G1Affine::generator(),
        &header.delta_g1,
        &G2Affine::generator(),
        &header.delta_g2,
    ) {
        return Err(fail("invalid delta2"));
    }
    if mpc.cs_hash != init_mpc.cs_hash {
        return Err(fail("circuit does not match"));
    }

    let n_vars = header.n_vars as usize;
    let n_public = header.n_public as usize;
    let domain_size = header.domain_size as usize;
    let l_bytes = file.unique_section(S_C)?;
    if l_bytes.len() != SG1 * (n_vars - n_public - 1) {
        return Err(fail("invalid L section size"));
    }
    let h_bytes = file.unique_section(S_H)?;
    if h_bytes.len() != SG1 * domain_size {
        return Err(fail("invalid H section size"));
    }

    for (id, what) in [
        (S_IC, "IC"),
        (S_COEFFS, "Coeffs"),
        (S_A, "A"),
        (S_B1, "B1"),
        (S_B2, "B2"),
    ] {
        if file.unique_section(id)? != init.unique_section(id)? {
            return Err(fail(format!("{what} section is not identical")));
        }
    }

    let mut rng = transcript::rng_from_entropy("");
    check_l_section(
        init.unique_section(S_C)?,
        l_bytes,
        &header.delta_g2,
        &init_header.delta_g2,
        &mut rng,
        msm,
    )?;
    check_h_section(
        ptau,
        h_bytes,
        domain_size,
        &header.delta_g2,
        &init_header.delta_g2,
        &mut rng,
        msm,
    )?;

    Ok(ZkeyVerifyReport {
        contribution_hashes,
        n_vars,
        n_public,
        domain_size,
    })
}

/// `zkey verify r1cs`: [`crate::setup::setup`] into a temporary init key, then
/// [`verify_from_init`] against it. This is the only form that checks the key against a
/// circuit, and it costs a full setup, every MSM included.
pub fn verify_from_r1cs(
    r1cs: &Path,
    ptau: &Path,
    final_zkey: &Path,
    msm: &dyn MsmBackend,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    // snarkjs keeps the init key in memory (`zkey_verify_fromr1cs.js:26`, a `bigMem`
    // handle). At anon-aadhaar that is 631 MB, so this writes it beside the output
    // instead and removes it afterwards.
    let init = final_zkey.with_extension("verify-init.zkey");
    let out = crate::setup::setup(r1cs, ptau, &init, msm)
        .and_then(|_| verify_from_init(&init, ptau, final_zkey, msm));
    let _ = std::fs::remove_file(&init);
    out
}

fn fail(what: impl Into<String>) -> CeremonyError {
    CeremonyError::Verification(what.into())
}

/// The contribution chain: each record's transcript has to be the accumulated hash the
/// record itself claims, its pubkey has to be self-consistent, and its `deltaAfter` has to
/// follow the previous one by that same key.
///
/// A beacon record is checked harder than a contribution: its key is a pure function of
/// the beacon hash and the iteration count it stores, so both points are recomputed rather
/// than trusted (`zkey_verify_frominit.js:75-88`).
fn verify_chain(mpc: &MpcParams, header: &Groth16Header) -> Result<Vec<Digest>, CeremonyError> {
    let mut accumulated = Transcript::new();
    accumulated.update(&mpc.cs_hash);
    let mut cur_delta = G1Affine::generator();
    let mut hashes = Vec::with_capacity(mpc.contributions.len());
    for (i, c) in mpc.contributions.iter().enumerate() {
        let mut ours = accumulated.clone();
        ours.update_g1(&c.g1_s);
        ours.update_g1(&c.g1_sx);
        if ours.finalize() != c.transcript {
            return Err(fail(format!("INVALID({i}): inconsistent transcript")));
        }
        let g2_sp = hash_to_g2(&c.transcript);
        if !same_ratio(&c.g1_s, &c.g1_sx, &g2_sp, &c.g2_spx) {
            return Err(fail(format!(
                "INVALID({i}): public key G1 and G2 do not have the same ratio"
            )));
        }
        if !same_ratio(&cur_delta, &c.delta_after, &g2_sp, &c.g2_spx) {
            return Err(fail(format!(
                "INVALID({i}): deltaAfter does not follow the public key"
            )));
        }
        if c.kind == ContributionKind::Beacon {
            let (hash, exp) = match (&c.params.beacon_hash, c.params.num_iterations_exp) {
                (Some(hash), Some(exp)) => (hash, exp),
                _ => {
                    return Err(fail(format!(
                        "INVALID({i}): beacon record has no parameters"
                    )))
                }
            };
            let mut rng = transcript::rng_from_beacon_params(hash, exp);
            let prv_key = transcript::fr_from_rng(&mut rng);
            let g1_s = transcript::g1_from_rng(&mut rng);
            if g1_s != c.g1_s || (g1_s * prv_key).into_affine() != c.g1_sx {
                return Err(fail(format!(
                    "INVALID({i}): key of the beacon does not match"
                )));
            }
        }
        hash_pubkey(&mut accumulated, c);
        hashes.push(contribution_hash(c));
        cur_delta = c.delta_after;
    }
    if header.delta_g1 != cur_delta {
        return Err(fail("invalid delta1"));
    }
    Ok(hashes)
}

/// Both rescaled sections reduce to one G1 point each by a random linear combination, and
/// the pair then has to have the same ratio as the two `delta_2`s
/// (`zkey_verify_frominit.js:234-269`). One combination over both files with the *same*
/// scalars is what makes this a check rather than two unrelated sums.
fn check_l_section(
    init_body: &[u8],
    final_body: &[u8],
    delta_g2: &G2Affine,
    init_delta_g2: &G2Affine,
    rng: &mut CeremonyRng,
    msm: &dyn MsmBackend,
) -> Result<(), CeremonyError> {
    if init_body.len() != final_body.len() {
        return Err(fail("L section does not match"));
    }
    let n = final_body.len() / SG1;
    if n == 0 {
        return Ok(());
    }
    // 32-bit scalars, as snarkjs uses (`getRandomBytes(4*n)`). A forged section would have
    // to survive a random combination, and 32 bits of soundness per check is snarkjs'
    // choice, not something the port gets to tighten without diverging.
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::from(rng.next_u32())).collect();
    let r1 = msm.msm_g1(&decode_g1(init_body), &scalars);
    let r2 = msm.msm_g1(&decode_g1(final_body), &scalars);
    if !same_ratio(
        &r1.into_affine(),
        &r2.into_affine(),
        delta_g2,
        init_delta_g2,
    ) {
        return Err(fail("L section does not match"));
    }
    Ok(())
}

/// The H check, the one part of `zkey verify` that is not a byte comparison or a single
/// ratio.
///
/// Section 9 holds `L_i(tau) * (tau^n - 1) / delta` on the odd half of the `2n`-th roots
/// of unity, so it cannot be compared against the ptau directly. snarkjs instead evaluates
/// a random polynomial two ways: once against `tauG1[i+n] - tauG1[i]` in the monomial
/// basis, and once against section 9 after pushing the same coefficients through the same
/// coset shift and forward transform (`zkey_verify_frominit.js:271-350`).
///
/// The last coefficient is forced to zero because the quotient polynomial has degree
/// `n - 2`, and the shift is `first = -2`, `inc = w_{2n}`, which is exactly the coset the
/// prover's H lands on.
///
/// That forced zero is also the only reason snarkjs works here at all. Its `tauG1[i + n]`
/// read runs to element `2n - 1` of a section holding `2n - 1` elements
/// (`zkey_verify_frominit.js:296`, a raw `fdPTau.read` at a computed offset), so on a key
/// whose `domainSize` equals the ptau's own it reads a G1 point out of the next section's
/// entry header. The point it decodes there is multiplied by the zero coefficient and
/// vanishes. We drop the term instead of reproducing the read: the sum is the same, and
/// unlike `hashHPoints` no hash is defined by the bytes.
#[allow(clippy::too_many_arguments)]
fn check_h_section(
    ptau_path: &Path,
    h_body: &[u8],
    domain_size: usize,
    delta_g2: &G2Affine,
    init_delta_g2: &G2Affine,
    rng: &mut CeremonyRng,
    msm: &dyn MsmBackend,
) -> Result<(), CeremonyError> {
    let ptau = Ptau::open(ptau_path)?;
    if ptau.domain() < domain_size {
        return Err(CeremonyError::PtauTooSmall {
            needed: crate::floor_log2(domain_size as u64),
            have: ptau.power(),
        });
    }
    let mut coeffs: Vec<Fr> = (0..domain_size)
        .map(|_| transcript::fr_from_rng(rng))
        .collect();
    coeffs[domain_size - 1] = Fr::zero();

    let n = domain_size - 1;
    let hi = ptau.g1_points(crate::ptau::S_TAU_G1, domain_size, n)?;
    let lo = ptau.g1_points(crate::ptau::S_TAU_G1, 0, n)?;
    let diff: Vec<G1Affine> = G1Projective::normalize_batch(
        &hi.par_iter()
            .zip(&lo)
            .map(|(a, b)| *a - *b)
            .collect::<Vec<_>>(),
    );
    let r1 = msm.msm_g1(&diff, &coeffs[..n]);

    // `Fr.w[power+1]`: a primitive `2n`-th root, so `inc^i` walks the odd coset the H
    // bases live on. The `power >= Fr.s` branch snarkjs also has is unreachable here,
    // because a domain above 2^28 has no `Domain` to build in the first place.
    let domain = Domain::new(domain_size)
        .map_err(|_| CeremonyError::CircuitTooBig(crate::floor_log2(domain_size as u64)))?;
    let double = Domain::new(domain_size * 2)
        .map_err(|_| CeremonyError::CircuitTooBig(domain.log_size + 1))?;
    let mut shifted = coeffs;
    let first = -Fr::from(2u64);
    let mut t = first;
    for c in shifted.iter_mut() {
        *c *= t;
        t *= double.group_gen;
    }
    let ntt = CpuNtt::new();
    ntt.ntt(&domain, &mut shifted, Direction::Forward);

    let r2 = msm.msm_g1(&decode_g1(h_body), &shifted);
    if !same_ratio(
        &r1.into_affine(),
        &r2.into_affine(),
        delta_g2,
        init_delta_g2,
    ) {
        return Err(fail("H section does not match"));
    }
    Ok(())
}

fn decode_g1(body: &[u8]) -> Vec<G1Affine> {
    body.par_chunks_exact(SG1).map(binfile::g1).collect()
}
