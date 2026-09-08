//! The snarkjs ceremony, byte for byte: `powersoftau` (phase 1) and `groth16 setup` plus
//! `zkey contribute` (phase 2), producing files snarkjs 0.7.6 itself accepts.
//!
//! The proving path in `g16-zkey` and `g16-core` only ever *reads* a finished `.zkey`.
//! This crate is the other half: it writes `.ptau` and `.zkey`, which means it has to
//! reproduce snarkjs' conventions exactly rather than merely understand them. Three of
//! those conventions are not choices we would make, and all three are load-bearing:
//!
//! * **Four encodings of a field element live in these files.** Curve coordinates are
//!   single Montgomery little-endian (`toRprLEM`); zkey section 4 coefficients are
//!   *double* Montgomery (`v*R^2`); r1cs coefficients and the header primes are plain
//!   little-endian; and every point that enters a transcript hash is non-Montgomery
//!   **big-endian**, with the two halves of a G2 coordinate swapped. The first three are
//!   decoded in [`g16_zkey::binfile`], which cites the snarkjs line each was read from;
//!   the fourth lives in [`transcript`], deliberately in different functions from the
//!   on-disk codecs in [`write`].
//! * **Two shipped snarkjs bugs are part of the format.** `hashHPoints` forgets to
//!   subtract the chunk offset (`zkey_new.js:511`) and so hashes `domainSize` points
//!   after announcing `domainSize - 1`, reading one G1 point past the end of ptau section
//!   2 when `cirPower == power >= 15`. `readContributions` guards duplicate section 7s
//!   with `sections[7][0].length`, which is `undefined`, so the check never fires
//!   (`powersoftau_utils.js:228`). We reproduce the first, because `csHash` is defined by
//!   the shipped behaviour, and we do not reproduce the second, because it only ever
//!   accepts files it should reject.
//! * **Order is not id order.** A fresh zkey's sections go out `1, 2, 4, 3, 9, 8, 5, 6,
//!   7, 10` (`zkey_new.js:76-168`), and both the file layout and `csHash` depend on it.
//!
//! Module map: [`r1cs`] and [`ptau`] read the two inputs, [`write`] is the container
//! writer every output goes through, [`transcript`] is the hash-and-RNG layer shared by
//! both phases, and [`phase1`], [`prepare`], [`setup`], [`contribute`] and [`vkey`] are
//! the commands.

pub mod contribute;
pub mod phase1;
pub mod prepare;
pub mod ptau;
pub mod r1cs;
pub mod setup;
pub mod transcript;
pub mod vkey;
pub mod write;

use g16_field::{BigInteger, Fq, Fr, G1Affine, G2Affine, PrimeField};
use g16_zkey::binfile::{Cursor, G1_BYTES, G2_BYTES};
use g16_zkey::ZkeyError;

use write::BinFileWriter;

/// Bytes per uncompressed G1 point on disk, snarkjs' `sG1`. BN254 only: the whole
/// workspace is BN254, and the ceremony is where a second curve would first show up, so
/// the constants are named after snarkjs' rather than derived from a generic `n8`.
pub const SG1: usize = G1_BYTES;
/// Bytes per uncompressed G2 point on disk, snarkjs' `sG2`.
pub const SG2: usize = G2_BYTES;
/// Bytes per compressed G1 point, snarkjs' `scG1`. Only phase 1's response hash and the
/// response file use it; nothing on disk in a `.ptau` body is compressed.
pub const SCG1: usize = 32;
/// Bytes per compressed G2 point, snarkjs' `scG2`.
pub const SCG2: usize = 64;
/// `n8` for BN254: one field element is 32 bytes in every encoding.
pub const N8: usize = 32;

/// snarkjs' groth16 protocol id, zkey section 1 (`zkey_constants.js:3`).
pub const PROTOCOL_GROTH16: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum CeremonyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Everything the shared container reader already reports: bad magic, a version past
    /// the supported one, a missing or duplicated section, a short section body.
    #[error(transparent)]
    Zkey(#[from] g16_zkey::ZkeyError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("malformed section {section}: {reason}")]
    Malformed { section: u32, reason: String },
    #[error("missing section {0}")]
    MissingSection(u32),
    /// The section chain does not reach the end of the file, or a declared length runs
    /// past it. Only [`ptau::Ptau::open_lenient`] produces this instead of failing.
    #[error("truncated: section {section:?} declares {declared} bytes, {present} present")]
    Truncated {
        section: Option<u32>,
        declared: u64,
        present: u64,
    },
    /// A `.ptau` or `.r1cs` header names a prime this workspace cannot prove over.
    #[error("unsupported curve: expected BN254")]
    UnsupportedCurve,
    #[error("unsupported protocol id {0} (only groth16 = 1)")]
    UnsupportedProtocol(u32),
    /// `zkey_new.js:61-64`. Not a warning: the Lagrange block the setup needs does not
    /// exist in a smaller ptau.
    #[error("circuit needs 2^{needed} powers of tau, this file has 2^{have}")]
    PtauTooSmall { needed: u32, have: u32 },
    /// Sections 12 to 15 are absent, so `powersoftau prepare phase2` has not been run
    /// (`zkey_new.js:66-69`).
    #[error("powers of tau is not prepared: run `g16 ptau prepare` first")]
    NotPrepared,
    /// `powersoftau_contribute.js:37-40`. A truncated ptau (`power != ceremonyPower`) can
    /// still be set up against, but it can never be contributed to again.
    #[error("this file is truncated (power {power}, ceremony power {ceremony_power}) and cannot be contributed to")]
    TruncatedCeremony { power: u32, ceremony_power: u32 },
    /// `Fr` has 2-adicity 28, so `cirPower` above that has no domain to evaluate on
    /// (`zkey_new.js:195-197`, "Circuit too big for this curve").
    #[error(
        "circuit too big for this curve: 2^{0} exceeds the two-adicity of BN254's scalar field"
    )]
    CircuitTooBig(u32),
    /// A CLI argument snarkjs would have rejected before touching a file: an odd-length
    /// beacon hash, `numIterationsExp` outside `10..=63`, an over-long contributor name.
    #[error("{0}")]
    BadParams(String),
    /// A check inside `ptau verify` or `zkey verify` failed. The string names the check,
    /// not the fix, in the shape snarkjs prints.
    #[error("verification failed: {0}")]
    Verification(String),
}

impl CeremonyError {
    /// Shorthand for the `Malformed` arm, which every reader in this crate raises.
    pub fn malformed(section: u32, reason: impl Into<String>) -> Self {
        Self::Malformed {
            section,
            reason: reason.into(),
        }
    }
}

/// What produced a contribution record. The u32 written to disk, in both phase 1
/// (`powersoftau_utils.js:174`) and phase 2 (`zkey_utils.js:474`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContributionKind {
    /// Entropy from the contributor, `type = 0`.
    Contribute,
    /// A public beacon, `type = 1`, and the only reproducible half of a ceremony.
    Beacon,
}

impl ContributionKind {
    pub fn from_u32(v: u32) -> Result<Self, CeremonyError> {
        match v {
            0 => Ok(Self::Contribute),
            1 => Ok(Self::Beacon),
            _ => Err(CeremonyError::malformed(
                0,
                format!("contribution type {v}"),
            )),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::Contribute => 0,
            Self::Beacon => 1,
        }
    }
}

/// The TLV blob tailing every contribution record. Byte-identical in phase 1
/// (`powersoftau_utils.js:186-206`, `:258-279`) and phase 2 (`zkey_utils.js:459-486`,
/// `:512-525`), which is why it lives here rather than twice.
///
/// Types must be **strictly ascending** on read, so the writer emits 1, then 2, then 3,
/// and an unrecognised type is a hard error rather than something to skip over.
#[derive(Clone, Default, Debug)]
pub struct ContributionParams {
    /// Type 1. snarkjs truncates to the first 64 UTF-16 code units and *then* encodes, so
    /// a name of 64 emoji produces more than 64 bytes; the real cap is the `u8` length
    /// field it is written into.
    pub name: Option<String>,
    /// Type 2, beacon only.
    pub num_iterations_exp: Option<u8>,
    /// Type 3, beacon only. Raw bytes, not the hex the CLI takes.
    pub beacon_hash: Option<Vec<u8>>,
}

impl ContributionParams {
    /// Parse the blob. `data` must be exactly `paramLength` bytes: a run that stops short
    /// of the end is "Parameters do not match", the same rejection snarkjs makes.
    pub fn decode(data: &[u8]) -> Result<Self, CeremonyError> {
        let mut out = Self::default();
        let mut pos = 0usize;
        let mut last_type = 0u8;
        // A `u8` length always follows types 1 and 3, so every read below is bounds
        // checked against `data` rather than trusted.
        let byte = |pos: usize| -> Result<u8, CeremonyError> {
            data.get(pos)
                .copied()
                .ok_or_else(|| CeremonyError::BadParams("parameters end mid-field".into()))
        };
        while pos < data.len() {
            let ty = byte(pos)?;
            pos += 1;
            if ty <= last_type {
                return Err(CeremonyError::BadParams(
                    "parameters in the contribution must be sorted".into(),
                ));
            }
            last_type = ty;
            match ty {
                1 => {
                    let len = byte(pos)? as usize;
                    pos += 1;
                    let raw = data
                        .get(pos..pos + len)
                        .ok_or_else(|| CeremonyError::BadParams("name runs past end".into()))?;
                    // snarkjs decodes with `TextDecoder`, which substitutes rather than
                    // rejects, so a name with a broken sequence must not fail the parse.
                    out.name = Some(String::from_utf8_lossy(raw).into_owned());
                    pos += len;
                }
                2 => {
                    out.num_iterations_exp = Some(byte(pos)?);
                    pos += 1;
                }
                3 => {
                    let len = byte(pos)? as usize;
                    pos += 1;
                    let raw = data.get(pos..pos + len).ok_or_else(|| {
                        CeremonyError::BadParams("beacon hash runs past end".into())
                    })?;
                    out.beacon_hash = Some(raw.to_vec());
                    pos += len;
                }
                other => {
                    return Err(CeremonyError::BadParams(format!(
                        "parameter type {other} not recognized"
                    )))
                }
            }
        }
        if pos != data.len() {
            return Err(CeremonyError::BadParams("parameters do not match".into()));
        }
        Ok(out)
    }

    /// The blob as written, empty when nothing is set. `paramLength` is its length.
    ///
    /// snarkjs truncates a name to 64 UTF-16 code units and only then encodes it, so a
    /// name of 64 emoji produces more than 255 bytes and silently overflows the `u8`
    /// length field (`zkey_utils.js:514`). We truncate the same way and then refuse the
    /// overflow, because the alternative is a file whose next field starts in the wrong
    /// place.
    pub fn encode(&self) -> Result<Vec<u8>, CeremonyError> {
        let mut out = Vec::new();
        if let Some(name) = &self.name {
            let units: Vec<u16> = name.encode_utf16().take(64).collect();
            let bytes = String::from_utf16_lossy(&units).into_bytes();
            if bytes.len() > u8::MAX as usize {
                return Err(CeremonyError::BadParams(format!(
                    "contributor name encodes to {} bytes, over the 255-byte field",
                    bytes.len()
                )));
            }
            out.push(1);
            out.push(bytes.len() as u8);
            out.extend_from_slice(&bytes);
        }
        if let Some(exp) = self.num_iterations_exp {
            out.push(2);
            out.push(exp);
        }
        if let Some(hash) = &self.beacon_hash {
            if hash.len() > u8::MAX as usize {
                return Err(CeremonyError::BadParams(format!(
                    "beacon hash is {} bytes, over the 255-byte field",
                    hash.len()
                )));
            }
            out.push(3);
            out.push(hash.len() as u8);
            out.extend_from_slice(hash);
        }
        Ok(out)
    }
}

/// zkey section 2, the groth16 header.
///
/// Point order is `alpha1, beta1, beta2, gamma2, delta1, delta2`. The comment block at
/// `zkey_utils.js:24-37` lists `alpha1, beta1, delta1, beta2, gamma2, delta2` and is
/// simply wrong; both writers (`zkey_new.js:102-129`, `zkey_utils.js:82-87`) and the
/// reader (`zkey_utils.js:249-254`) agree with the order here.
///
/// In a freshly generated zkey `gamma_g2`, `delta_g1` and `delta_g2` are the plain
/// generators: setup performs no division, and `zkey contribute` is what later scales
/// them (`zkey_contribute.js:63-64`).
#[derive(Clone, Debug)]
pub struct Groth16Header {
    pub n_vars: u32,
    /// `nOutputs + nPubInputs`, so the constant ONE signal is excluded.
    pub n_public: u32,
    pub domain_size: u32,
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,
}

impl Groth16Header {
    /// Bytes of section 2 on BN254: `4 + 32 + 4 + 32 + 12` of scalars, then `3*sG1 +
    /// 3*sG2` of points.
    pub const BYTES: usize = 84 + 3 * SG1 + 3 * SG2;

    /// Parse section 2, rejecting a `q` or `r` that is not BN254's.
    pub fn read(section: &[u8]) -> Result<Self, CeremonyError> {
        let mut cur = Cursor::new(section, 2);
        check_modulus(&mut cur, &q_le())?;
        check_modulus(&mut cur, &r_le())?;
        let n_vars = cur.u32()?;
        let n_public = cur.u32()?;
        let domain_size = cur.u32()?;
        let alpha_g1 = g16_zkey::binfile::g1(cur.take(SG1)?);
        let beta_g1 = g16_zkey::binfile::g1(cur.take(SG1)?);
        let beta_g2 = g16_zkey::binfile::g2(cur.take(SG2)?);
        let gamma_g2 = g16_zkey::binfile::g2(cur.take(SG2)?);
        let delta_g1 = g16_zkey::binfile::g1(cur.take(SG1)?);
        let delta_g2 = g16_zkey::binfile::g2(cur.take(SG2)?);
        if cur.remaining() != 0 {
            return Err(CeremonyError::malformed(
                2,
                format!("{} bytes left after the header", cur.remaining()),
            ));
        }
        Ok(Self {
            n_vars,
            n_public,
            domain_size,
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
        })
    }

    /// Write section 2 into an already-started section.
    pub fn write(&self, w: &mut BinFileWriter) -> Result<(), CeremonyError> {
        w.write_prime(&q_le())?;
        w.write_prime(&r_le())?;
        w.write_u32(self.n_vars)?;
        w.write_u32(self.n_public)?;
        w.write_u32(self.domain_size)?;
        w.write_g1(&self.alpha_g1)?;
        w.write_g1(&self.beta_g1)?;
        w.write_g2(&self.beta_g2)?;
        w.write_g2(&self.gamma_g2)?;
        w.write_g1(&self.delta_g1)?;
        w.write_g2(&self.delta_g2)
    }
}

/// `u32 n8` then the modulus as a plain little-endian integer, the shape both the ptau
/// header (`powersoftau_utils.js:35-39`) and the zkey header (`zkey_new.js:85-90`) use.
/// Rejects anything that is not the expected prime, since the curve is identified only by
/// this field: there is no curve-name anywhere in either format.
pub fn check_modulus(cur: &mut Cursor, expected_le: &[u8]) -> Result<(), CeremonyError> {
    let n8 = cur.u32()? as usize;
    if n8 != expected_le.len() || cur.take(n8)? != expected_le {
        return Err(CeremonyError::UnsupportedCurve);
    }
    Ok(())
}

/// BN254's base field modulus, plain little-endian, as both headers store it.
pub fn q_le() -> [u8; N8] {
    let mut out = [0u8; N8];
    out.copy_from_slice(&Fq::MODULUS.to_bytes_le());
    out
}

/// BN254's scalar field modulus, plain little-endian.
pub fn r_le() -> [u8; N8] {
    let mut out = [0u8; N8];
    out.copy_from_slice(&Fr::MODULUS.to_bytes_le());
    out
}

/// `misc.js:53-56`: a floor-log2 that returns 0 for 0, which is what makes
/// [`setup::circuit_power`] behave the way `zkey_new.js:59` does.
pub fn floor_log2(n: u64) -> u32 {
    if n == 0 {
        0
    } else {
        63 - n.leading_zeros()
    }
}

/// Re-exported so the ceremony's own error type can absorb a container failure without
/// every module importing `g16_zkey` for the one name.
pub type ContainerError = ZkeyError;
