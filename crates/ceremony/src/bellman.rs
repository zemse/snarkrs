//! Phase 2 through bellman's `MPCParameters`: `zkey export bellman`, `zkey bellman
//! contribute` and `zkey import bellman`.
//!
//! The `.mpcparams` file is the one Zcash's `phase2` crate reads and writes, so a zkey
//! ceremony can take contributions from that tool. It carries the same key as the zkey in
//! bellman's shapes and encodings, every point **uncompressed** (non-Montgomery
//! big-endian, G2 `c1` first) and every count a **big-endian** u32
//! (`zkey_export_bellman.js`):
//!
//! ```text
//! alpha1 beta1 beta2 gamma2 delta1 delta2
//! u32 nIC   IC      (zkey section 3)
//! u32 nH    H       (domainSize - 1 points, derived from section 9, see below)
//! u32 nL    L       (section 8)
//! u32 nA    A       (section 5)
//! u32 nB1   B1      (section 6)
//! u32 nB2   B2      (section 7, G2)
//! csHash, u32 nContributions, then per contribution
//!     deltaAfter g1_s g1_sx (G1)  g2_spx (G2)  transcript (64 bytes)
//! ```
//!
//! Section 9 does not hold bellman's `H` query. snarkjs stores the odd-coset Lagrange
//! basis there, and bellman wants `tau^i Z(tau) / delta` for `i < domainSize - 1`. Export
//! converts with a forward group FFT, then scales point `i` by `-2 w^i` with `w` the
//! primitive `2^(power+1)`-th root, then drops the last point (`zkey_export_bellman.js:
//! 48-54`). Import inverts it: pad with the identity, scale by `-(1/2) w^-i`, inverse FFT
//! (`zkey_import_bellman.js:128-138`). ffjavascript's forward FFT is the plain DFT
//! `out[k] = sum_j in[j] w_n^(jk)` and its inverse is [`crate::prepare::group_ifft_g1`],
//! so the forward one is computed here as `n * ifft(in)[-k mod n]`, folding the `n` into
//! the export scale rather than adding a second transform to the crate.
//!
//! A contribution rescales `delta` and the `H` and `L` queries exactly as `zkey
//! contribute` does, from the same RNG draws in the same order (`Fr` for delta, then one
//! `G1` for `g1_s`) and the same transcript (`zkey_bellman_contribute.js:35-138`).
//!
//! Import takes only `delta` and the `H` and `L` queries from the file; every other section
//! is copied from the old zkey and the counts in the file are only checked. It writes a
//! **version 1** zkey in the section order `1, 2, 3, 4, 9, 8, 5, 6, 7, 10`
//! (`zkey_import_bellman.js:98-167`). A mismatched `csHash`, a chain shorter than the old
//! one, or a previous contribution that differs makes snarkjs log an error and return
//! `false` without writing anything; each is a [`CeremonyError::Verification`] here.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;
use snarkrs_field::{
    AffineRepr, CurveGroup, FftField, Field, Fr, G1Affine, G1Projective, G2Affine,
};
use snarkrs_formats::binfile::{self, BinFile};
use snarkrs_msm::KeyScale;

use crate::accel::{CpuGroupFft, CpuKeyScale};
use crate::contribute::{
    accumulated_transcript, check_protocol, contribution_hash, MpcParams, ZkeyContribution,
};
use crate::setup::{
    S_A, S_B1, S_B2, S_C, S_COEFFS, S_H, S_HEADER, S_IC, S_MPC_PARAMS, S_PROTOCOL, ZKEY_MAGIC,
    ZKEY_MAX_VERSION, ZKEY_SECTIONS, ZKEY_VERSION,
};
use crate::transcript::{
    self, create_delta_key, g1_from_uncompressed, g1_uncompressed, g2_from_uncompressed,
    g2_uncompressed, CeremonyRng, Digest,
};
use crate::write::BinFileWriter;
use crate::{CeremonyError, ContributionKind, ContributionParams, Groth16Header, SG1, SG2};

/// Points per read or write while streaming a query. A buffering choice only: nothing
/// hashes these bytes.
const CHUNK_POINTS: usize = 1 << 16;

/// Bytes of the six verification-key points that open the file.
const VK_BYTES: usize = 3 * SG1 + 3 * SG2;

/// `Fr.w[power + 1]`, the primitive `2^(power+1)`-th root of unity. arkworks derives it by
/// squaring the two-adic root down, which is how ffjavascript builds its table
/// (`build_fft.js:44-63`).
fn coset_root(power: u32) -> Result<Fr, CeremonyError> {
    if power + 1 > Fr::TWO_ADICITY {
        return Err(CeremonyError::CircuitTooBig(power));
    }
    Ok(Fr::get_root_of_unity(1u64 << (power + 1)).expect("2^(power+1) divides r - 1"))
}

fn log2_domain(header: &Groth16Header) -> Result<u32, CeremonyError> {
    let n = header.domain_size;
    if !n.is_power_of_two() {
        return Err(CeremonyError::malformed(
            S_HEADER,
            format!("domain size {n} is not a power of two"),
        ));
    }
    Ok(n.trailing_zeros())
}

fn decode_g1(body: &[u8], id: u32) -> Result<Vec<G1Affine>, CeremonyError> {
    if !body.len().is_multiple_of(SG1) {
        return Err(CeremonyError::malformed(
            id,
            format!("{} bytes is not a whole number of G1 points", body.len()),
        ));
    }
    Ok(binfile::decode_records(
        body,
        SG1,
        G1Affine::identity(),
        |_, b| binfile::g1(b),
    )?)
}

fn write_count(w: &mut impl Write, n: usize) -> Result<(), CeremonyError> {
    w.write_all(&(n as u32).to_be_bytes())?;
    Ok(())
}

/// A LEM section as a counted array of uncompressed points (`writePointArray`).
fn export_array(w: &mut impl Write, body: &[u8], sg: usize, id: u32) -> Result<(), CeremonyError> {
    if !body.len().is_multiple_of(sg) {
        return Err(CeremonyError::malformed(
            id,
            format!("{} bytes is not a whole number of points", body.len()),
        ));
    }
    write_count(w, body.len() / sg)?;
    let mut out = vec![0u8; body.len().min(CHUNK_POINTS * sg)];
    for chunk in body.chunks(CHUNK_POINTS * sg) {
        let dst = &mut out[..chunk.len()];
        chunk
            .par_chunks_exact(sg)
            .zip(dst.par_chunks_exact_mut(sg))
            .try_for_each(|(src, dst)| -> Result<(), CeremonyError> {
                if sg == SG1 {
                    dst.copy_from_slice(&g1_uncompressed(&binfile::g1(src)?));
                } else {
                    dst.copy_from_slice(&g2_uncompressed(&binfile::g2(src)?));
                }
                Ok(())
            })?;
        w.write_all(dst)?;
    }
    Ok(())
}

/// Section 9 in bellman's form: `DFT(H)[i] * -2 w^i` for `i < n - 1`.
fn export_h(section: &[u8], power: u32) -> Result<Vec<G1Affine>, CeremonyError> {
    let points = decode_g1(section, S_H)?;
    let n = points.len();
    if n != 1usize << power {
        return Err(CeremonyError::malformed(
            S_H,
            format!("{n} points, the domain is 2^{power}"),
        ));
    }
    let mut a: Vec<G1Projective> = points.par_iter().map(|p| p.into_group()).collect();
    crate::prepare::group_ifft_g1(&mut a, &CpuGroupFft)?;
    // `ifft` leaves `(1/n) DFT[-k]` at `k`; undo the reversal, and put `n` in the scale.
    a[1..].reverse();
    let mut affine = G1Projective::normalize_batch(&a);
    let first = -Fr::from(2u64) * Fr::from(n as u64);
    CpuKeyScale.apply_key_g1(&mut affine, first, coset_root(power)?)?;
    affine.pop();
    Ok(affine)
}

/// Write the zkey at `zkey` as a bellman `MPCParameters` file. Backs `snarkjs zkey export
/// bellman`.
pub fn export_bellman(zkey: &Path, out: &Path) -> Result<(), CeremonyError> {
    let file = BinFile::open(zkey, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let mpc = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;
    let power = log2_domain(&header)?;
    let h = export_h(file.unique_section(S_H)?, power)?;

    let mut w = BufWriter::new(File::create(out)?);
    w.write_all(&g1_uncompressed(&header.alpha_g1))?;
    w.write_all(&g1_uncompressed(&header.beta_g1))?;
    w.write_all(&g2_uncompressed(&header.beta_g2))?;
    w.write_all(&g2_uncompressed(&header.gamma_g2))?;
    w.write_all(&g1_uncompressed(&header.delta_g1))?;
    w.write_all(&g2_uncompressed(&header.delta_g2))?;
    export_array(&mut w, file.unique_section(S_IC)?, SG1, S_IC)?;
    write_count(&mut w, h.len())?;
    for p in &h {
        w.write_all(&g1_uncompressed(p))?;
    }
    export_array(&mut w, file.unique_section(S_C)?, SG1, S_C)?;
    export_array(&mut w, file.unique_section(S_A)?, SG1, S_A)?;
    export_array(&mut w, file.unique_section(S_B1)?, SG1, S_B1)?;
    export_array(&mut w, file.unique_section(S_B2)?, SG2, S_B2)?;
    w.write_all(&mpc.cs_hash)?;
    write_count(&mut w, mpc.contributions.len())?;
    for c in &mpc.contributions {
        write_contribution(&mut w, c)?;
    }
    w.flush()?;
    Ok(())
}

fn write_contribution(w: &mut impl Write, c: &ZkeyContribution) -> Result<(), CeremonyError> {
    w.write_all(&g1_uncompressed(&c.delta_after))?;
    w.write_all(&g1_uncompressed(&c.g1_s))?;
    w.write_all(&g1_uncompressed(&c.g1_sx))?;
    w.write_all(&g2_uncompressed(&c.g2_spx))?;
    w.write_all(&c.transcript)?;
    Ok(())
}

/// A sequential reader over an `.mpcparams` file.
struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, CeremonyError> {
        let mut out = vec![0u8; n];
        self.inner.read_exact(&mut out)?;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, CeremonyError> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }
    fn g1(&mut self) -> Result<G1Affine, CeremonyError> {
        g1_from_uncompressed(&self.bytes(SG1)?)
    }
    fn g2(&mut self) -> Result<G2Affine, CeremonyError> {
        g2_from_uncompressed(&self.bytes(SG2)?)
    }
    fn digest(&mut self) -> Result<Digest, CeremonyError> {
        let mut d = [0u8; 64];
        self.inner.read_exact(&mut d)?;
        Ok(d)
    }
    /// The contribution records, as `readMPCParams`' bellman twin reads them. Type and
    /// params are not in the file.
    fn contributions(&mut self) -> Result<(Digest, Vec<ZkeyContribution>), CeremonyError> {
        let cs_hash = self.digest()?;
        let n = self.u32()?;
        let mut out = Vec::with_capacity((n as usize).min(1024));
        for _ in 0..n {
            out.push(ZkeyContribution {
                delta_after: self.g1()?,
                g1_s: self.g1()?,
                g1_sx: self.g1()?,
                g2_spx: self.g2()?,
                transcript: self.digest()?,
                kind: ContributionKind::Contribute,
                params: ContributionParams::default(),
            });
        }
        Ok((cs_hash, out))
    }
}

/// Contribute to the `.mpcparams` at `input` with the contributor's entropy, writing the
/// result to `output` and returning the contribution hash. Backs `snarkjs zkey bellman
/// contribute`.
pub fn bellman_contribute(
    input: &Path,
    output: &Path,
    entropy: &str,
    scale: &dyn KeyScale,
) -> Result<Digest, CeremonyError> {
    bellman_contribute_with(input, output, transcript::rng_from_entropy(entropy)?, scale)
}

/// [`bellman_contribute`] with the RNG supplied, so a run can be held against snarkjs'
/// bytes. Backs `snarkjs zkey bellman contribute`.
pub fn bellman_contribute_with(
    input: &Path,
    output: &Path,
    mut rng: CeremonyRng,
    scale: &dyn KeyScale,
) -> Result<Digest, CeremonyError> {
    let mut r = Reader {
        inner: BufReader::new(File::open(input)?),
    };
    let mut w = BufWriter::new(File::create(output)?);

    // The chain sits after every query and the key's transcript absorbs it, so the whole
    // file is read before the key is drawn. `zkey contribute` streams instead, but it
    // reads the chain from section 10 first; this file offers no such order.
    let head = r.bytes(VK_BYTES)?;
    let alpha_beta_gamma = &head[..2 * SG1 + 2 * SG2];
    let delta1 = g1_from_uncompressed(&head[2 * SG1 + 2 * SG2..3 * SG1 + 2 * SG2])?;
    let delta2 = g2_from_uncompressed(&head[3 * SG1 + 2 * SG2..])?;
    let mut arrays = Vec::with_capacity(6);
    for sg in [SG1, SG1, SG1, SG1, SG1, SG2] {
        let n = r.u32()? as usize;
        arrays.push((n, r.bytes(n * sg)?));
    }
    let (cs_hash, prior) = r.contributions()?;
    let mut mpc = MpcParams {
        cs_hash,
        contributions: prior,
    };

    let (delta, digest) = create_delta_key(&mut rng, accumulated_transcript(&mpc));
    let delta1 = (delta1 * delta.prv_key.expose()).into_affine();
    let delta2 = (delta2 * delta.prv_key.expose()).into_affine();
    let inv_delta = zeroize::Zeroizing::new(
        delta
            .prv_key
            .expose()
            .inverse()
            .ok_or_else(|| CeremonyError::BadParams("the contribution key drew zero".into()))?,
    );
    let current = ZkeyContribution {
        delta_after: delta1,
        g1_s: delta.g1_s,
        g1_sx: delta.g1_sx,
        g2_spx: delta.g2_spx,
        transcript: digest,
        kind: ContributionKind::Contribute,
        params: ContributionParams::default(),
    };
    mpc.contributions.push(current.clone());

    w.write_all(alpha_beta_gamma)?;
    w.write_all(&g1_uncompressed(&delta1))?;
    w.write_all(&g2_uncompressed(&delta2))?;
    for (i, (n, body)) in arrays.iter().enumerate() {
        write_count(&mut w, *n)?;
        // IC, A, B1 and B2 are copied; H (1) and L (2) are divided by delta
        // (`applyKeyToChallengeSection` with `first = invDelta`, `inc = 1`).
        if i == 1 || i == 2 {
            for chunk in body.chunks(CHUNK_POINTS * SG1) {
                let mut points = chunk
                    .par_chunks_exact(SG1)
                    .map(g1_from_uncompressed)
                    .collect::<Result<Vec<_>, _>>()?;
                scale.apply_key_g1(&mut points, *inv_delta, Fr::ONE)?;
                for p in &points {
                    w.write_all(&g1_uncompressed(p))?;
                }
            }
        } else {
            w.write_all(body)?;
        }
    }
    w.write_all(&mpc.cs_hash)?;
    write_count(&mut w, mpc.contributions.len())?;
    for c in &mpc.contributions {
        write_contribution(&mut w, c)?;
    }
    w.flush()?;
    Ok(contribution_hash(&current))
}

/// What `import bellman` wrote.
#[derive(Clone, Debug)]
pub struct BellmanImport {
    /// Contributions the old zkey already had.
    pub n_prior: usize,
    /// The whole new chain, with a hash per contribution as `zkey verify` prints them.
    pub contribution_hashes: Vec<Digest>,
}

fn import_failed(what: &str) -> CeremonyError {
    CeremonyError::Verification(what.to_owned())
}

/// Import the `.mpcparams` at `mpcparams` onto `zkey_old`, writing `zkey_new`. `name`
/// labels every contribution the file adds. Backs `snarkjs zkey import bellman`.
pub fn import_bellman(
    zkey_old: &Path,
    mpcparams: &Path,
    zkey_new: &Path,
    name: Option<&str>,
) -> Result<BellmanImport, CeremonyError> {
    let file = BinFile::open(zkey_old, ZKEY_MAGIC, ZKEY_MAX_VERSION)?;
    check_protocol(&file)?;
    let mut header = Groth16Header::read(file.unique_section(S_HEADER)?)?;
    let old = MpcParams::read(file.unique_section(S_MPC_PARAMS)?)?;
    let power = log2_domain(&header)?;
    let n_vars = header.n_vars as usize;
    let n_public = header.n_public as usize;
    let domain = header.domain_size as usize;

    let mut r = Reader {
        inner: BufReader::new(File::open(mpcparams)?),
    };
    // The chain first, at the offset the old header's counts imply
    // (`zkey_import_bellman.js:42-48`), so a mismatch is refused before anything is
    // decoded or written.
    let skip = VK_BYTES
        + 8
        + SG1 * n_vars
        + 4
        + SG1 * (domain - 1)
        + 4
        + SG1 * n_vars
        + 4
        + SG1 * n_vars
        + 4
        + SG2 * n_vars;
    std::io::copy(&mut (&mut r.inner).take(skip as u64), &mut std::io::sink())?;
    let (cs_hash, mut contributions) = r.contributions()?;
    if cs_hash != old.cs_hash {
        return Err(import_failed(
            "hash of the original circuit does not match with the MPC one",
        ));
    }
    if old.contributions.len() > contributions.len() {
        return Err(import_failed(
            "the imported file does not include new contributions",
        ));
    }
    for (i, (was, now)) in old.contributions.iter().zip(&contributions).enumerate() {
        let same = was.delta_after == now.delta_after
            && was.g1_s == now.g1_s
            && was.g1_sx == now.g1_sx
            && was.g2_spx == now.g2_spx
            && was.transcript == now.transcript;
        if !same {
            return Err(import_failed(&format!(
                "previous contribution {i} does not match"
            )));
        }
    }
    // Old records keep their type, beacon params and name; new ones are type 0, named
    // `name` when one is given (`:55-68`, `:90-94`).
    for (was, now) in old.contributions.iter().zip(contributions.iter_mut()) {
        now.kind = was.kind;
        now.params = was.params.clone();
    }
    let params = ContributionParams {
        name: name.filter(|s| !s.is_empty()).map(str::to_owned),
        ..Default::default()
    };
    params.encode()?;
    for now in contributions.iter_mut().skip(old.contributions.len()) {
        now.params = params.clone();
    }
    let new_mpc = MpcParams {
        cs_hash,
        contributions,
    };

    // Now the points, from the start.
    let mut r = Reader {
        inner: BufReader::new(File::open(mpcparams)?),
    };
    r.bytes(2 * SG1 + 2 * SG2)?;
    header.delta_g1 = r.g1()?;
    header.delta_g2 = r.g2()?;
    let expect = |got: u32, want: usize, what: &str| {
        if got as usize != want {
            return Err(import_failed(&format!(
                "invalid number of points in {what}"
            )));
        }
        Ok(())
    };

    let mut w = BinFileWriter::create(zkey_new, ZKEY_MAGIC, ZKEY_VERSION, ZKEY_SECTIONS)?;
    w.start_section(S_PROTOCOL)?;
    w.write_u32(crate::PROTOCOL_GROTH16)?;
    w.end_section()?;
    w.start_section(S_HEADER)?;
    header.write(&mut w)?;
    w.end_section()?;

    expect(r.u32()?, n_public + 1, "IC")?;
    r.bytes(SG1 * (n_public + 1))?;
    w.write_section_verbatim(S_IC, file.unique_section(S_IC)?)?;
    w.write_section_verbatim(S_COEFFS, file.unique_section(S_COEFFS)?)?;

    expect(r.u32()?, domain - 1, "H")?;
    let mut h = Vec::with_capacity(domain);
    for chunk in r.bytes(SG1 * (domain - 1))?.chunks_exact(SG1) {
        h.push(g1_from_uncompressed(chunk)?);
    }
    h.push(G1Affine::identity());
    let w_inv = coset_root(power)?
        .inverse()
        .expect("a root of unity is a unit");
    let half = Fr::from(2u64).inverse().expect("2 is a unit mod r");
    CpuKeyScale.apply_key_g1(&mut h, -half, w_inv)?;
    let mut a: Vec<G1Projective> = h.par_iter().map(|p| p.into_group()).collect();
    crate::prepare::group_ifft_g1(&mut a, &CpuGroupFft)?;
    w.start_section(S_H)?;
    w.write_g1_slice(&G1Projective::normalize_batch(&a))?;
    w.end_section()?;

    let n_l = n_vars - n_public - 1;
    expect(r.u32()?, n_l, "L")?;
    let l = r.bytes(SG1 * n_l)?;
    w.start_section(S_C)?;
    for chunk in l.chunks(CHUNK_POINTS * SG1) {
        let points = chunk
            .par_chunks_exact(SG1)
            .map(g1_from_uncompressed)
            .collect::<Result<Vec<_>, _>>()?;
        w.write_g1_slice(&points)?;
    }
    w.end_section()?;

    for (id, sg, what) in [(S_A, SG1, "A"), (S_B1, SG1, "B1"), (S_B2, SG2, "B2")] {
        expect(r.u32()?, n_vars, what)?;
        r.bytes(sg * n_vars)?;
        w.write_section_verbatim(id, file.unique_section(id)?)?;
    }

    w.start_section(S_MPC_PARAMS)?;
    new_mpc.write(&mut w)?;
    w.end_section()?;
    w.finish()?;

    Ok(BellmanImport {
        n_prior: old.contributions.len(),
        contribution_hashes: new_mpc
            .contributions
            .iter()
            .map(contribution_hash)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::UniformRand;

    /// Export's H transform followed by import's is the identity on any vector whose DFT
    /// ends in zero, which is the shape the dropped last point assumes. Checked here on
    /// a random vector with that property built directly, so the two directions are held
    /// against each other without a zkey.
    #[test]
    fn the_h_transforms_invert_each_other() {
        let mut rng = seeded_rng();
        let power = 3u32;
        let n = 1usize << power;
        // Build DFT-space values with the last one zero, then go back with the inverse.
        let mut dft: Vec<G1Projective> = (0..n)
            .map(|i| {
                if i + 1 == n {
                    G1Projective::default()
                } else {
                    G1Projective::rand(&mut rng)
                }
            })
            .collect();
        // ifft(dft)[m] = (1/n) sum dft_j w^-jm, and the forward DFT of that is dft again.
        crate::prepare::group_ifft_g1(&mut dft, &CpuGroupFft).unwrap();
        let h = G1Projective::normalize_batch(&dft);
        let mut section = Vec::new();
        for p in &h {
            section.extend_from_slice(&crate::write::g1_lem(p));
        }
        let exported = export_h(&section, power).unwrap();
        assert_eq!(exported.len(), n - 1);

        let mut back = exported.clone();
        back.push(G1Affine::identity());
        let w_inv = coset_root(power).unwrap().inverse().unwrap();
        let half = Fr::from(2u64).inverse().unwrap();
        CpuKeyScale.apply_key_g1(&mut back, -half, w_inv).unwrap();
        let mut a: Vec<G1Projective> = back.iter().map(|p| p.into_group()).collect();
        crate::prepare::group_ifft_g1(&mut a, &CpuGroupFft).unwrap();
        assert_eq!(G1Projective::normalize_batch(&a), h);
    }

    fn seeded_rng() -> impl rand_chacha::rand_core::RngCore {
        use rand_chacha::rand_core::SeedableRng;
        rand_chacha::ChaCha20Rng::seed_from_u64(7)
    }
}
