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
use std::time::{Duration, Instant};

use ark_ec::short_weierstrass::{Affine, Projective};
use rayon::prelude::*;
use snarkrs_field::{
    g1, g2, AffineRepr, CurveGroup, Domain, Field, Fr, G1Affine, G1Projective, G2Affine, Zero,
};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};
use snarkrs_msm::{AccelError, KeyScale, MsmBackend};
use snarkrs_ntt::{CpuNtt, Direction};

use crate::phase1::check_beacon;
use crate::ptau::Ptau;
use crate::setup::{
    Combined, Prepared, SetupSink, S_A, S_B1, S_B2, S_C, S_COEFFS, S_H, S_HEADER, S_IC,
    S_MPC_PARAMS, S_PROTOCOL, ZKEY_MAGIC, ZKEY_MAX_VERSION, ZKEY_SECTIONS, ZKEY_VERSION,
};
use crate::transcript::{
    self, create_delta_key, hash_to_g2, same_ratio, CeremonyRng, Digest, Transcript,
};
use crate::write::{g1_lem, g2_lem, BinFileWriter};
use crate::{
    CeremonyError, ContainerError, ContributionKind, ContributionParams, Groth16Header, SG1, SG2,
};

use snarkrs_formats::binfile::{self, BinFile, Cursor};

/// Points per pass over a rescaled section. snarkjs' `MAX_CHUNK_SIZE` is 2^16
/// (`mpc_applykey.js:30`); the chunking changes no byte, and a larger one gives the pool
/// more tasks between the joins of [`apply_key_to_section`]. 16 MB of G1 on the way in and
/// the same on the way out, and still the unit a GPU kernel would take: one launch per
/// chunk, with [`KeyScale::apply_key_g1`] as the body.
const CHUNK_POINTS: usize = 1 << 18;

/// Points per MSM in the L and H checks. Each check is a sum of products, so it is taken a
/// chunk at a time and never holds a whole section decoded: 2^21 G1 points is 151 MB,
/// where a 2^25 circuit's H section is 2.4 GB.
const CHECK_CHUNK: usize = 1 << 21;

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
            let delta_after = binfile::g1(cur.take(SG1)?)?;
            let g1_s = binfile::g1(cur.take(SG1)?)?;
            let g1_sx = binfile::g1(cur.take(SG1)?)?;
            let g2_spx = binfile::g2(cur.take(SG2)?)?;
            for (p, what) in [
                (&delta_after, "delta_after"),
                (&g1_s, "g1_s"),
                (&g1_sx, "g1_sx"),
            ] {
                crate::check_g1(p, S_MPC_PARAMS, what)?;
            }
            crate::check_g2(&g2_spx, S_MPC_PARAMS, "g2_spx")?;
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
    let mut rng = transcript::rng_from_entropy(entropy)?;
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

/// [`contribute`] with the RNG supplied, so a run can be held against another one.
///
/// `zkey contribute` mixes 64 OS-random bytes into its seed and is deliberately not
/// reproducible otherwise; [`crate::transcript::rng_from_entropy_with`] is the hook, and
/// this is phase 2's twin of [`crate::phase1::contribute_with`]. Made public for the
/// cpu-against-metal comparison, which needs both runs to draw the same delta: without it
/// the only byte-comparable phase 2 command is `beacon`.
pub fn contribute_with(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    mut rng: CeremonyRng,
    key: &dyn KeyScale,
) -> Result<ContributionReport, CeremonyError> {
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
/// `beacon_hash` is raw bytes, not hex; [`crate::phase1::parse_beacon_args`] is the
/// parser the CLI hands them through. The bounds snarkjs enforces
/// (`10 <= num_iterations_exp <= 63`, and a hash of 1 to 255 bytes) are checked here, so
/// they hold for a caller that is not a command.
pub fn beacon(
    zkey_in: &Path,
    zkey_out: &Path,
    name: Option<&str>,
    beacon_hash: &[u8],
    num_iterations_exp: u8,
    key: &dyn KeyScale,
) -> Result<ContributionReport, CeremonyError> {
    let iterations = check_beacon(beacon_hash, num_iterations_exp)?;
    let mut rng = transcript::rng_from_beacon_params(beacon_hash, iterations);
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
    crate::write::staged(zkey_out, |stage| {
        apply_inner(zkey_in, stage, kind, params, rng, key)
    })
}

fn apply_inner(
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
    header.delta_g1 = (header.delta_g1 * delta.prv_key.expose()).into_affine();
    header.delta_g2 = (header.delta_g2 * delta.prv_key.expose()).into_affine();

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

    // As secret as delta itself: sections 8 and 9 are scaled by it, and zeroed on drop.
    let inv_delta = zeroize::Zeroizing::new(
        delta
            .prv_key
            .expose()
            .inverse()
            .ok_or_else(|| CeremonyError::BadParams("the contribution key drew zero".into()))?,
    );

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
        apply_key_to_section(&mut w, file.unique_section(id)?, id, *inv_delta, key)?;
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
pub(crate) fn check_protocol(file: &BinFile) -> Result<(), CeremonyError> {
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
    let scale = |chunk: &[u8]| -> Result<Vec<G1Affine>, CeremonyError> {
        let mut points =
            binfile::decode_records(chunk, SG1, G1Affine::identity(), |_, b| binfile::g1(b))?;
        key.apply_key_g1(&mut points, k, Fr::ONE)?;
        Ok(points)
    };
    // Chunk `i` is written on this thread while the pool decodes and scales chunk `i + 1`,
    // so the serial encode and write is off the critical path. The order of writes is the
    // order of the chunks either way.
    let mut chunks = body.chunks(CHUNK_POINTS * SG1);
    let mut done = chunks.next().map(scale).transpose()?;
    while let Some(points) = done.take() {
        let next = chunks.next();
        let (written, scaled) =
            rayon::join(|| w.write_g1_slice(&points), || next.map(scale).transpose());
        written?;
        done = scaled?;
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
    /// The backend that recomputed the initial key, for the r1cs form.
    pub setup_backend: Option<&'static str>,
    /// How sections 5 to 7 were checked.
    pub check: Check,
    /// Wall time per stage, in run order.
    pub phases: Vec<(&'static str, Duration)>,
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
    verify_from_init_with(init_zkey, ptau, final_zkey, msm, Check::Exact)
}

/// [`verify_from_init`] with sections 5 to 7 checked as `check` says. The byte comparison
/// [`Check::Exact`] makes here is already cheaper than any combination, so
/// [`Check::Random`] buys this form nothing but the same check the r1cs form makes.
pub fn verify_from_init_with(
    init_zkey: &Path,
    ptau: &Path,
    final_zkey: &Path,
    msm: &dyn MsmBackend,
    check: Check,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    let file = BinFile::open(final_zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let mpc = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;

    let init = BinFile::open(init_zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&init)?;
    let init_header = Groth16Header::read(init.unique_section(S_HEADER)?)?;
    let init_mpc = MpcParams::read(init.unique_section(S_MPC_PARAMS)?)?;

    let contribution_hashes = check_headers(&mpc, &header, &init_header)?;
    if mpc.cs_hash != init_mpc.cs_hash {
        return Err(fail("circuit does not match"));
    }

    let n_vars = header.n_vars as usize;
    let n_public = header.n_public as usize;
    let domain_size = header.domain_size as usize;
    let l_bytes = file.unique_section(S_C)?;
    if l_points(n_vars, n_public).map(|n| SG1 * n) != Some(l_bytes.len()) {
        return Err(fail("invalid L section size"));
    }
    let h_bytes = file.unique_section(S_H)?;
    if h_bytes.len() != SG1 * domain_size {
        return Err(fail("invalid H section size"));
    }

    // Drawn with both keys already open, so no scalar exists before the points it weighs.
    let mut rng = transcript::rng_from_entropy("g16 zkey verify")?;
    for (id, what) in COMPARED {
        let (fin, ini) = (file.unique_section(id)?, init.unique_section(id)?);
        match (check, id) {
            (Check::Random, S_A | S_B1) => {
                random_against_init::<g1::Config>(ini, fin, what, &mut rng, msm)?
            }
            (Check::Random, S_B2) => {
                random_against_init::<g2::Config>(ini, fin, what, &mut rng, msm)?
            }
            _ if fin != ini => return Err(fail(format!("{what} section is not identical"))),
            _ => {}
        }
    }

    let at = Instant::now();
    check_l_section(
        init.unique_section(S_C)?,
        l_bytes,
        &header.delta_g2,
        &init_header.delta_g2,
        &mut rng,
        msm,
    )?;
    let l_time = at.elapsed();
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
        setup_backend: None,
        check,
        phases: vec![("L check", l_time), ("H check", at.elapsed() - l_time)],
    })
}

/// The sections `zkey verify` requires to be byte-identical between the init key and the
/// final one, in the order it compares them (`zkey_verify_frominit.js:169-197`).
const COMPARED: [(u32, &str); 5] = [
    (S_IC, "IC"),
    (S_COEFFS, "Coeffs"),
    (S_A, "A"),
    (S_B1, "B1"),
    (S_B2, "B2"),
];

/// The checks both forms of `zkey verify` make before `csHash`: the contribution chain,
/// then init-vs-final header agreement.
fn check_headers(
    mpc: &MpcParams,
    header: &Groth16Header,
    init_header: &Groth16Header,
) -> Result<Vec<Digest>, CeremonyError> {
    let contribution_hashes = verify_chain(mpc, header)?;

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
    Ok(contribution_hashes)
}

/// `zkey verify r1cs`: [`verify_from_init`] against the initial key a setup of `r1cs` would
/// write, recomputed and compared as it is produced. This is the only form that checks
/// the key against a circuit, and it costs a full setup, every MSM included.
///
/// `setup_msm` computes that initial key and `check_msm` the L and H checks. They are two
/// parameters because they fail differently: a wrong recomputed point fails a byte
/// comparison, so a faulty `setup_msm` can only reject a good key, while a faulty
/// `check_msm` that returns equal sums on both sides would accept a bad one.
///
/// snarkjs holds the whole initial key in memory (`zkey_verify_fromr1cs.js:26`, a `bigMem`
/// handle). Here no copy of it exists, in memory or on disk: every check on it is either a
/// byte comparison, made a chunk at a time against the final key's mapped bytes, or a sum
/// the L check takes a chunk at a time. So the memory this takes is setup's own term lists
/// plus a few chunks, whatever the size of the key.
///
/// The header checks run before the recompute rather than after it, because they need only
/// the header a setup would write. A key that fails one fails in seconds instead of after
/// a full setup, with the same message. Everything else is reported in
/// [`verify_from_init`]'s order once the recompute is done.
pub fn verify_from_r1cs(
    r1cs: &Path,
    ptau: &Path,
    final_zkey: &Path,
    setup_msm: &dyn MsmBackend,
    check_msm: &dyn MsmBackend,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    verify_from_r1cs_with(r1cs, ptau, final_zkey, setup_msm, check_msm, Check::Exact)
}

/// [`verify_from_r1cs`] with sections 5 to 7 checked as `check` says.
///
/// [`Check::Random`] composes none of their points. Setup instead sums the Lagrange bases
/// under the scalars each section's combination comes to ([`Prepared::run`]), on
/// `setup_msm`, and the final key's points are decoded, checked and combined on
/// `check_msm`. As in the exact check, setup's output is only ever compared, so a faulty
/// `setup_msm` can only reject.
pub fn verify_from_r1cs_with(
    r1cs: &Path,
    ptau: &Path,
    final_zkey: &Path,
    setup_msm: &dyn MsmBackend,
    check_msm: &dyn MsmBackend,
    check: Check,
) -> Result<ZkeyVerifyReport, CeremonyError> {
    let setup = Prepared::open(r1cs, ptau)?;
    let file = BinFile::open(final_zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let mpc = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;
    let init_header = setup.header();
    let contribution_hashes = check_headers(&mpc, &header, init_header)?;

    let n_vars = header.n_vars as usize;
    let n_public = header.n_public as usize;
    let domain_size = header.domain_size as usize;
    let l_len = SG1 * l_points(n_vars, n_public).ok_or_else(|| fail("invalid L section size"))?;
    // Drawn with the final key already open, so no scalar exists before the points it
    // weighs.
    let mut rng = transcript::rng_from_entropy("g16 zkey verify")?;
    let combine = (check == Check::Random).then(|| {
        let r = random_scalars(&mut rng, init_header.n_vars as usize);
        (r, sweep_seed(&mut rng))
    });
    let mut sink = Recompute::new(&file, l_len, &mut rng, check_msm);
    sink.random = combine.as_ref().map(|(r, seed)| Random {
        r,
        seed: *seed,
        msm: check_msm,
        a: None,
        b1: None,
        b2: None,
        combined: None,
    });
    let report = setup.run(&mut sink, setup_msm, combine.as_ref().map(|(r, _)| &r[..]))?;
    let Recompute {
        identical,
        l,
        random,
        ..
    } = sink;
    let mut verdicts = random.map(Random::finish).unwrap_or_default();

    // In the random check csHash is over the final key's own sections 5 to 7, since none
    // of setup's were composed (`Random`). Those bytes are the true composition only if
    // their check passes, so a failed one makes the comparison meaningless; it is skipped
    // then, and the failure is raised below in its place in the order.
    if verdicts.iter().all(|(_, v)| v.is_ok()) && mpc.cs_hash != report.cs_hash {
        return Err(fail("circuit does not match"));
    }
    if file.unique_section(S_C)?.len() != l_len {
        return Err(fail("invalid L section size"));
    }
    let h_bytes = file.unique_section(S_H)?;
    if h_bytes.len() != SG1 * domain_size {
        return Err(fail("invalid H section size"));
    }
    for (id, what) in COMPARED {
        file.unique_section(id)?;
        if let Some((_, verdict)) = verdicts.iter_mut().find(|(v, _)| *v == id) {
            std::mem::replace(verdict, Ok(()))?;
        } else if !identical.contains(&(id, true)) {
            return Err(fail(format!("{what} section is not identical")));
        }
    }
    // Every verdict has to have been raised by the loop above, which is what makes
    // skipping csHash on a failed one safe.
    for (_, verdict) in verdicts {
        verdict?;
    }
    l.finish()?.check(&header.delta_g2, &init_header.delta_g2)?;
    let at = Instant::now();
    check_h_section(
        ptau,
        h_bytes,
        domain_size,
        &header.delta_g2,
        &init_header.delta_g2,
        &mut rng,
        check_msm,
    )?;

    let mut phases = report.phases;
    phases.push(("H check", at.elapsed()));
    Ok(ZkeyVerifyReport {
        contribution_hashes,
        n_vars,
        n_public,
        domain_size,
        setup_backend: Some(setup_msm.name()),
        check,
        phases,
    })
}

/// [`verify_from_init`]'s comparisons, made against setup's sections as they are produced
/// instead of against an init key on disk.
///
/// Each compared section is matched a chunk at a time against the final key's mapped
/// bytes, and section 8, which the contributions rescaled, goes straight into the L
/// check's random combination. What stays is a verdict per section, for
/// [`verify_from_r1cs`] to report in snarkjs' order once `csHash` is known.
struct Recompute<'a> {
    file: &'a BinFile,
    /// The section being produced, and the comparison of it if it is one of [`COMPARED`].
    open: Option<(u32, Option<Compare<'a>>)>,
    /// `(id, byte-identical)` for every compared section produced.
    identical: Vec<(u32, bool)>,
    l: LStream<'a>,
    /// Sections 5 to 7 under [`Check::Random`], which setup supplies rather than produces.
    random: Option<Random<'a>>,
}

impl<'a> Recompute<'a> {
    /// `l_len` is the byte length the final key's section 8 must have; one of any other
    /// length fails its size check, so the L sums skip it.
    fn new(
        file: &'a BinFile,
        l_len: usize,
        rng: &'a mut CeremonyRng,
        msm: &'a dyn MsmBackend,
    ) -> Self {
        let body = file
            .unique_section(S_C)
            .ok()
            .filter(|body| body.len() == l_len);
        Self {
            file,
            open: None,
            identical: Vec::new(),
            l: LStream {
                body,
                at: 0,
                sums: RatioSums::default(),
                failed: None,
                rng,
                msm,
            },
            random: None,
        }
    }

    fn compare(&mut self) -> Option<&mut Compare<'a>> {
        self.open.as_mut().and_then(|(_, c)| c.as_mut())
    }
}

impl SetupSink for Recompute<'_> {
    /// Section 9 is rescaled by every contribution, and the H check reads the final key's
    /// copy against the ptau, so the recomputed one would be compared against nothing.
    fn wants(&self, id: u32) -> bool {
        id != S_H
    }

    fn header(&mut self, _: &Groth16Header) -> Result<(), CeremonyError> {
        Ok(())
    }

    fn start(&mut self, id: u32) -> Result<(), CeremonyError> {
        let compared = COMPARED.iter().any(|&(c, _)| c == id);
        let compare = compared.then(|| {
            let body = self.file.unique_section(id).ok();
            Compare {
                body,
                at: 0,
                same: body.is_some(),
            }
        });
        self.open = Some((id, compare));
        Ok(())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CeremonyError> {
        if let Some(c) = self.compare() {
            c.next(bytes.len(), |fin| fin == bytes);
        }
        Ok(())
    }

    fn g1(&mut self, points: &[G1Affine]) -> Result<(), CeremonyError> {
        if matches!(self.open, Some((S_C, _))) {
            self.l.add(points);
        } else if let Some(c) = self.compare() {
            c.next(points.len() * SG1, |fin| encodes_as(points, fin, g1_lem));
        }
        Ok(())
    }

    fn g2(&mut self, points: &[G2Affine]) -> Result<(), CeremonyError> {
        if let Some(c) = self.compare() {
            c.next(points.len() * SG2, |fin| encodes_as(points, fin, g2_lem));
        }
        Ok(())
    }

    fn end(&mut self) -> Result<(), CeremonyError> {
        if let Some((id, Some(c))) = self.open.take() {
            let whole = c.body.is_some_and(|body| c.at == body.len());
            self.identical.push((id, c.same && whole));
        }
        Ok(())
    }

    fn finish(&mut self, _: &Digest) -> Result<(), CeremonyError> {
        Ok(())
    }

    fn supply(&mut self, id: u32, hash: &(dyn Fn(Vec<u8>) + Sync)) -> Result<(), CeremonyError> {
        let random = self.random.as_mut().ok_or_else(|| {
            CeremonyError::BadParams(format!("section {id} supplied to an exact check"))
        })?;
        // A missing section is raised in order later, as the exact check raises it.
        let body = self.file.unique_section(id).unwrap_or(&[]);
        let (r, seed, msm) = (random.r, random.seed, random.msm);
        match id {
            S_A => random.a = Some(reduce(body, r, seed, msm, hash)),
            S_B1 => random.b1 = Some(reduce(body, r, seed, msm, hash)),
            S_B2 => random.b2 = Some(reduce(body, r, seed, msm, hash)),
            _ => {
                return Err(CeremonyError::BadParams(format!(
                    "section {id} is not one the random check combines"
                )))
            }
        }
        Ok(())
    }

    fn combined(&mut self, sums: Combined) -> Result<(), CeremonyError> {
        match &mut self.random {
            Some(random) => {
                random.combined = Some(sums);
                Ok(())
            }
            None => Err(CeremonyError::BadParams(
                "combined sections handed to an exact check".into(),
            )),
        }
    }
}

/// One section of the final key being matched against its recomputation.
struct Compare<'a> {
    /// `None` when the final key has no such section; that error is raised in order later.
    body: Option<&'a [u8]>,
    /// Bytes of `body` matched so far.
    at: usize,
    same: bool,
}

impl Compare<'_> {
    /// Match the next `len` bytes of the final key's section with `matches`. Once a
    /// mismatch is found the rest is only counted.
    fn next(&mut self, len: usize, matches: impl FnOnce(&[u8]) -> bool) {
        let fin = self.body.and_then(|b| b.get(self.at..self.at + len));
        self.at += len;
        self.same = self.same && fin.is_some_and(matches);
    }
}

/// True when `fin` is exactly `points`, each encoded by `enc`.
fn encodes_as<T: Sync, const N: usize>(points: &[T], fin: &[u8], enc: fn(&T) -> [u8; N]) -> bool {
    fin.len() == points.len() * N
        && points
            .par_iter()
            .zip(fin.par_chunks_exact(N))
            .with_min_len(4096)
            .all(|(p, b)| enc(p) == *b)
}

/// The L check fed by setup's section 8 as it is produced: each chunk of the recomputed C
/// and the same points of the final key go into [`RatioSums`] under the same scalars, so
/// neither section is ever held whole.
struct LStream<'a> {
    /// The final key's section 8, when its length is the one the check needs.
    body: Option<&'a [u8]>,
    at: usize,
    sums: RatioSums,
    /// The first error, raised when the L check is reached, as `check_l_section` would.
    failed: Option<CeremonyError>,
    rng: &'a mut CeremonyRng,
    msm: &'a dyn MsmBackend,
}

impl LStream<'_> {
    fn add(&mut self, init: &[G1Affine]) {
        let Some(body) = self.body else { return };
        if self.failed.is_some() {
            return;
        }
        let len = init.len() * SG1;
        let Some(fin) = body.get(self.at..self.at + len) else {
            self.failed = Some(fail("L section does not match"));
            return;
        };
        self.at += len;
        let scalars = l_scalars(self.rng, init.len());
        let added = decode_g1(fin)
            .map_err(CeremonyError::from)
            .and_then(|fin| self.sums.add(init, &fin, &scalars, self.msm));
        if let Err(e) = added {
            self.failed = Some(e);
        }
    }

    fn finish(self) -> Result<RatioSums, CeremonyError> {
        match self.failed {
            Some(e) => Err(e),
            // Setup has to have produced every point of the final key's section 8.
            None if self.body.is_some_and(|body| self.at != body.len()) => {
                Err(fail("L section does not match"))
            }
            None => Ok(self.sums),
        }
    }
}

/// How `zkey verify` checks sections 5 to 7, the A, B1 and B2 points. IC, Coeffs and C are
/// checked the same way under both: IC and Coeffs byte for byte, and C composed and fed to
/// the L check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Check {
    /// Byte for byte, as snarkjs checks them. In the r1cs form that composes every point,
    /// one multiexp per signal, which is most of the command's time.
    #[default]
    Exact,
    /// One random linear combination per section, held against the same combination of
    /// what the section has to hold, after every point of it is decoded and checked. See
    /// [`reduce`] for what that proves and with what error.
    Random,
}

impl Check {
    pub fn name(self) -> &'static str {
        match self {
            Check::Exact => "exact",
            Check::Random => "random",
        }
    }
}

/// Points per pass of [`reduce`], as in setup's own sections.
const RANDOM_CHUNK: usize = 1 << 20;

/// Combinations [`sweep`] takes of a G2 section, and the bits of each of their scalars.
const SWEEPS: usize = 10;
const SWEEP_BITS: u32 = 13;

/// Points per bucket pass of [`sweep`]. Each pass ends in a running sum over its 2^13
/// buckets, about a tenth of the pass at this size.
const SWEEP_TASK: usize = 1 << 18;

/// A group a section of the random check is in.
trait Section: RawCurve {
    /// Bytes per point on disk.
    const STRIDE: usize;
    /// [`sweep`] combinations a section of this group needs: none in G1, whose curve has
    /// prime order, so a point on it is in the group.
    const SWEEPS: usize;
    fn decode(b: &[u8]) -> Result<Affine<Self>, ContainerError>;
    fn uncompressed(points: &[Affine<Self>]) -> Vec<u8>;
    fn combine(
        msm: &dyn MsmBackend,
        bases: &[Affine<Self>],
        scalars: &[Fr],
    ) -> Result<Projective<Self>, AccelError>;
}

impl Section for g1::Config {
    const STRIDE: usize = SG1;
    const SWEEPS: usize = 0;
    fn decode(b: &[u8]) -> Result<G1Affine, ContainerError> {
        binfile::g1(b)
    }
    fn uncompressed(points: &[G1Affine]) -> Vec<u8> {
        transcript::g1_uncompressed_all(points)
    }
    fn combine(
        msm: &dyn MsmBackend,
        bases: &[G1Affine],
        scalars: &[Fr],
    ) -> Result<G1Projective, AccelError> {
        msm.try_msm_g1(bases, scalars)
    }
}

impl Section for g2::Config {
    const STRIDE: usize = SG2;
    const SWEEPS: usize = SWEEPS;
    fn decode(b: &[u8]) -> Result<G2Affine, ContainerError> {
        binfile::g2(b)
    }
    fn uncompressed(points: &[G2Affine]) -> Vec<u8> {
        transcript::g2_uncompressed_all(points)
    }
    fn combine(
        msm: &dyn MsmBackend,
        bases: &[G2Affine],
        scalars: &[Fr],
    ) -> Result<Projective<g2::Config>, AccelError> {
        msm.try_msm_g2(bases, scalars)
    }
}

/// One section of a key reduced by [`reduce`].
struct Reduced<P: Section> {
    /// `sum_i r_i * P_i`.
    sum: Projective<P>,
    /// [`sweep`]'s combinations, [`Section::SWEEPS`] of them.
    sweeps: Vec<Projective<P>>,
}

impl<P: Section> Reduced<P> {
    /// Every sweep in the order-r subgroup, which with high probability means every point
    /// is. Checked per point here, on a handful of points.
    fn in_subgroup(&self) -> bool {
        self.sweeps.iter().all(|s| {
            let s = s.into_affine();
            s.is_on_curve() && s.is_in_correct_subgroup_assuming_on_curve()
        })
    }
}

/// Why [`reduce`] gave up on a section.
enum Failure {
    /// Not `r.len()` points long.
    Size,
    /// Point `at` failed to decode or is off the curve.
    Point { at: usize, why: &'static str },
    /// Not a verdict on the key: the MSM failed.
    Other(CeremonyError),
}

impl Failure {
    /// `whose` names the key when it is not the final one.
    fn into_error(self, what: &str, whose: &str) -> CeremonyError {
        match self {
            // The exact check's message, since its byte comparison fails on the same keys.
            Failure::Size => fail(format!("{what} section is not identical")),
            Failure::Point { at, why } => fail(format!(
                "{what} section does not match: point {at}{whose} {why}"
            )),
            Failure::Other(e) => e,
        }
    }
}

/// The random check's view of one point section: decode and check every point, take
/// `sum_i r_i * P_i` on `msm`, sweep a G2 section, and hand `hash` the uncompressed
/// encoding of each chunk in order.
///
/// What a pass proves, with each `r_i` uniform below 2^128 and drawn after the key is
/// fixed:
///
/// * Each point decodes canonically, every coordinate below `q` and the all-zero encoding
///   alone read as infinity, as snarkjs writes it, and lies on its curve.
/// * G1 has prime order `r`, so a G1 point on the curve is in the group. G2's twist does
///   not: `E'(Fq2)` has order `r * h`, `h` having 10069 as its smallest prime factor. A
///   point with a component of order `d | h` survives a random combination with
///   probability about `1/d`, so [`sweep`] takes [`SWEEPS`] more combinations with 13-bit
///   scalars and the caller checks each lands in the order-r subgroup. Each one misses a
///   nonzero component with probability at most `2^-13` (a component of order above
///   2^13 is hit by at most one scalar below 2^13), so all ten miss it with at most
///   `2^-130`.
/// * With every point in a group of prime order `r > 2^128`, two sections that differ at
///   point `k` give equal sums with probability at most `2^-128`: fixing every other
///   scalar, at most one `r_k` below 2^128 cancels the difference.
///
/// So a section the random check accepts differs from the one it should hold with
/// probability at most `2^-128 + 2^-130`, and over A, B1 and B2 under `2^-126`.
fn reduce<P: Section>(
    body: &[u8],
    r: &[Fr],
    sweep_seed: [u32; 8],
    msm: &dyn MsmBackend,
    mut hash: impl FnMut(Vec<u8>),
) -> Result<Reduced<P>, Failure> {
    if Some(body.len()) != r.len().checked_mul(P::STRIDE) {
        return Err(Failure::Size);
    }
    let mut rng = CeremonyRng::from_seed_words(sweep_seed);
    let mut out = Reduced {
        sum: Projective::zero(),
        sweeps: vec![Projective::zero(); P::SWEEPS],
    };
    for (k, (bytes, r)) in body
        .chunks(RANDOM_CHUNK * P::STRIDE)
        .zip(r.chunks(RANDOM_CHUNK))
        .enumerate()
    {
        let points = binfile::decode_records(bytes, P::STRIDE, Affine::identity(), |i, b| {
            let p = P::decode(b).map_err(|_| (i, "has a coordinate at or above q"))?;
            if !p.is_on_curve() {
                return Err((i, "is not on the curve"));
            }
            Ok(p)
        })
        .map_err(|(i, why)| Failure::Point {
            at: k * RANDOM_CHUNK + i,
            why,
        })?;
        out.sum += P::combine(msm, &points, r).map_err(|e| Failure::Other(e.into()))?;
        sweep(&points, &mut rng, &mut out.sweeps);
        hash(P::uncompressed(&points));
    }
    Ok(out)
}

/// Add `sum_i c_i * P_i` over `points` to each of `sums`, every `c_i` fresh and uniform
/// below 2^13, by bucket: each point is one addition into the bucket of its scalar, and a
/// running sum over the buckets weighs each by its index.
///
/// The arithmetic is plain additions on the curve, with no endomorphism, so the sums are
/// the true ones even for a point outside the subgroup, which is the point of taking them.
fn sweep<P: RawCurve>(points: &[Affine<P>], rng: &mut CeremonyRng, sums: &mut [Projective<P>]) {
    if sums.is_empty() {
        return;
    }
    let n = points.len();
    let mask = (1u32 << SWEEP_BITS) - 1;
    let mut scalars = vec![0u16; sums.len() * n];
    for pair in scalars.chunks_mut(2) {
        let w = rng.next_u32();
        pair[0] = (w & mask) as u16;
        if let Some(c) = pair.get_mut(1) {
            *c = (w >> 16 & mask) as u16;
        }
    }
    let jobs: Vec<(usize, usize)> = (0..sums.len())
        .flat_map(|k| (0..n).step_by(SWEEP_TASK).map(move |from| (k, from)))
        .collect();
    let parts: Vec<(usize, Projective<P>)> = jobs
        .into_par_iter()
        .map(|(k, from)| {
            let to = (from + SWEEP_TASK).min(n);
            // On the heap: 2^13 G2 buckets are 2 MB, a rayon thread's whole stack.
            #[allow(clippy::useless_vec)]
            let mut buckets = vec![Xyzz::<P::RF>::ZERO; 1 << SWEEP_BITS];
            for (p, &c) in points[from..to]
                .iter()
                .zip(&scalars[k * n + from..k * n + to])
            {
                if c != 0 && !p.infinity {
                    let (x, y) = P::raw_xy(p);
                    buckets[c as usize].madd(x, y);
                }
            }
            let (mut running, mut acc) = (Xyzz::ZERO, Xyzz::ZERO);
            for b in buckets[1..].iter().rev() {
                running.add_assign(b);
                acc.add_assign(&running);
            }
            (k, to_projective::<P>(&acc))
        })
        .collect();
    for (k, s) in parts {
        sums[k] += s;
    }
}

/// `n` scalars of the random check, each uniform below 2^128.
fn random_scalars(rng: &mut CeremonyRng, n: usize) -> Vec<Fr> {
    let raw: Vec<u128> = (0..n)
        .map(|_| u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()))
        .collect();
    raw.par_iter().map(|&v| Fr::from(v)).collect()
}

/// The seed of [`sweep`]'s scalars, drawn from the check's own RNG.
fn sweep_seed(rng: &mut CeremonyRng) -> [u32; 8] {
    std::array::from_fn(|_| rng.next_u32())
}

/// [`Recompute`]'s state under [`Check::Random`]: sections 5 to 7 of the final key reduced
/// as setup reaches them, and what setup says they should combine to.
///
/// csHash is a hash over the composed points, and none are composed here, so setup hashes
/// the final key's own points in their place, which [`reduce`] supplies. That stands for
/// the hash of the composition only once the random check has shown those points are the
/// composition, which is why [`verify_from_r1cs_with`] compares it only then.
struct Random<'a> {
    r: &'a [Fr],
    seed: [u32; 8],
    msm: &'a dyn MsmBackend,
    a: Option<Result<Reduced<g1::Config>, Failure>>,
    b1: Option<Result<Reduced<g1::Config>, Failure>>,
    b2: Option<Result<Reduced<g2::Config>, Failure>>,
    combined: Option<Combined>,
}

impl Random<'_> {
    /// The verdict on each section, by id.
    fn finish(self) -> Vec<(u32, Result<(), CeremonyError>)> {
        let Some(want) = self.combined else {
            let none = |id| {
                (
                    id,
                    Err(CeremonyError::BadParams("setup combined nothing".into())),
                )
            };
            return vec![none(S_A), none(S_B1), none(S_B2)];
        };
        vec![
            (S_A, judge(self.a, "A", want.a)),
            (S_B1, judge(self.b1, "B1", want.b1)),
            (S_B2, judge(self.b2, "B2", want.b2)),
        ]
    }
}

/// The random check's verdict on one section of the final key, given what it should sum
/// to. `None` is a section setup never supplied.
fn judge<P: Section>(
    found: Option<Result<Reduced<P>, Failure>>,
    what: &str,
    want: Projective<P>,
) -> Result<(), CeremonyError> {
    let found = found
        .ok_or_else(|| CeremonyError::BadParams(format!("setup never supplied {what}")))?
        .map_err(|f| f.into_error(what, ""))?;
    if !found.in_subgroup() {
        return Err(fail(format!(
            "{what} section does not match: a point is outside the order-r subgroup"
        )));
    }
    if found.sum != want {
        return Err(fail(format!("{what} section does not match")));
    }
    Ok(())
}

/// [`verify_from_init_with`]'s random check of one section: the final key's against the
/// initial key's, under the same scalars, both decoded and checked.
///
/// Holding every sweep equal on both sides as well makes this an equality check on the
/// whole curve, not only the subgroup: the sweeps' 13-bit scalars catch a difference of
/// any order with probability at least `1 - 2^-13` each.
fn random_against_init<P: Section>(
    init: &[u8],
    fin: &[u8],
    what: &str,
    rng: &mut CeremonyRng,
    msm: &dyn MsmBackend,
) -> Result<(), CeremonyError> {
    if init.len() != fin.len() || !fin.len().is_multiple_of(P::STRIDE) {
        return Err(fail(format!("{what} section is not identical")));
    }
    let r = random_scalars(rng, fin.len() / P::STRIDE);
    let seed = sweep_seed(rng);
    let ini = reduce::<P>(init, &r, seed, msm, drop)
        .map_err(|f| f.into_error(what, " of the initial key"))?;
    match reduce::<P>(fin, &r, seed, msm, drop) {
        Ok(f) if f.in_subgroup() && f.sweeps != ini.sweeps => {
            Err(fail(format!("{what} section does not match")))
        }
        f => judge(Some(f), what, ini.sum),
    }
}

/// The private signals past ONE, which is the length of section 8 in points.
fn l_points(n_vars: usize, n_public: usize) -> Option<usize> {
    n_vars.checked_sub(n_public)?.checked_sub(1)
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
    // snarkjs has no guard here (`ys` in the bundle), so `zkey verify` on an untouched init
    // key reports success: the loop below leaves `cur_delta` at the generator, which is
    // what `header.delta_g1` still is, `same_ratio` on the two generators passes, the byte
    // comparisons against the init file are trivial when it is the same file, and both
    // rescale checks pass because `invDelta == 1`. That is a clean verification for a key
    // whose delta is 1, under which anyone can forge a proof. Phase 1 already makes the
    // opposite choice (`phase1::verify`), and no key carrying a contribution can trip this.
    if mpc.contributions.is_empty() {
        return Err(fail(
            "this key has no contribution, it cannot be used in production",
        ));
    }
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
            let iterations =
                check_beacon(hash, exp).map_err(|e| fail(format!("INVALID({i}): {e}")))?;
            let mut rng = transcript::rng_from_beacon_params(hash, iterations);
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
    let mut sums = RatioSums::default();
    for (init, fin) in init_body
        .chunks(CHECK_CHUNK * SG1)
        .zip(final_body.chunks(CHECK_CHUNK * SG1))
    {
        let scalars = l_scalars(rng, init.len() / SG1);
        sums.add(&decode_g1(init)?, &decode_g1(fin)?, &scalars, msm)?;
    }
    sums.check(delta_g2, init_delta_g2)
}

/// `n` of the L check's scalars. 32 bits each, as snarkjs uses (`getRandomBytes(4*n)`):
/// a forged section would have to survive a random combination, and 32 bits of soundness
/// per check is snarkjs' choice, not something the port gets to tighten without
/// diverging.
fn l_scalars(rng: &mut CeremonyRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::from(rng.next_u32())).collect()
}

/// The L check's two random combinations, one per file, summed a chunk at a time with
/// the same scalars on both sides.
#[derive(Default)]
struct RatioSums {
    init: G1Projective,
    fin: G1Projective,
    n: usize,
}

impl RatioSums {
    fn add(
        &mut self,
        init: &[G1Affine],
        fin: &[G1Affine],
        scalars: &[Fr],
        msm: &dyn MsmBackend,
    ) -> Result<(), CeremonyError> {
        self.init += msm.try_msm_g1(init, scalars)?;
        self.fin += msm.try_msm_g1(fin, scalars)?;
        self.n += scalars.len();
        Ok(())
    }

    /// An empty section has nothing to compare.
    fn check(&self, delta_g2: &G2Affine, init_delta_g2: &G2Affine) -> Result<(), CeremonyError> {
        if self.n > 0
            && !same_ratio(
                &self.init.into_affine(),
                &self.fin.into_affine(),
                delta_g2,
                init_delta_g2,
            )
        {
            return Err(fail("L section does not match"));
        }
        Ok(())
    }
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
    let hi = ptau.section_elements(crate::ptau::S_TAU_G1, SG1, domain_size, n)?;
    let lo = ptau.section_elements(crate::ptau::S_TAU_G1, SG1, 0, n)?;
    let mut r1 = G1Projective::zero();
    for ((hi, lo), coeffs) in hi
        .chunks(CHECK_CHUNK * SG1)
        .zip(lo.chunks(CHECK_CHUNK * SG1))
        .zip(coeffs[..n].chunks(CHECK_CHUNK))
    {
        let (hi, lo) = (decode_g1(hi)?, decode_g1(lo)?);
        let diff: Vec<G1Projective> = hi.par_iter().zip(&lo).map(|(a, b)| *a - *b).collect();
        r1 += msm.try_msm_g1(&crate::setup::to_affine(&diff), coeffs)?;
    }

    // `Fr.w[power+1]`: a primitive `2n`-th root, so `inc^i` walks the odd coset the H
    // bases live on. The `power >= Fr.s` branch snarkjs also has is unreachable here,
    // because a domain above 2^28 has no `Domain` to build in the first place.
    let domain = Domain::new(domain_size)
        .map_err(|_| CeremonyError::CircuitTooBig(crate::floor_log2(domain_size as u64)))?;
    let double = Domain::new(domain_size * 2)
        .map_err(|_| CeremonyError::CircuitTooBig(domain.log_size() + 1))?;
    let mut shifted = coeffs;
    let first = -Fr::from(2u64);
    let mut t = first;
    for c in shifted.iter_mut() {
        *c *= t;
        t *= double.group_gen();
    }
    let ntt = CpuNtt::new();
    ntt.ntt(&domain, &mut shifted, Direction::Forward);

    if h_body.len() != SG1 * domain_size {
        return Err(fail("invalid H section size"));
    }
    let mut r2 = G1Projective::zero();
    for (h, s) in h_body
        .chunks(CHECK_CHUNK * SG1)
        .zip(shifted.chunks(CHECK_CHUNK))
    {
        r2 += msm.try_msm_g1(&decode_g1(h)?, s)?;
    }
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

fn decode_g1(body: &[u8]) -> Result<Vec<G1Affine>, ContainerError> {
    binfile::decode_records(body, SG1, G1Affine::identity(), |_, b| binfile::g1(b))
}
