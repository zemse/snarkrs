//! The binfile writer, and the on-disk encodings every section body uses.
//!
//! `nSections` is declared before any section exists (`binfileutils.js:46`) and is never
//! revised, so [`BinFileWriter::create`] takes the true count and [`BinFileWriter::finish`]
//! refuses a file that emitted a different number. A writer that lies here produces a file
//! whose reader either stops early or runs off the end, with no error from either side.
//!
//! Section lengths are written as zero and backfilled: `startWriteSection` reserves the u64
//! (`binfileutils.js:57`) and `endWriteSection` seeks back to fill it in (`:63-67`). A file
//! whose write was interrupted therefore has a `0` length in the middle of its chain, which
//! a scanner reads as an empty section and then misparses everything after. That is one of
//! the two truncation shapes [`crate::ptau::Ptau::open_lenient`] looks for.
//!
//! Three of the encoders below carry the same numbers in different forms, and this is the
//! module where they must not be confused:
//!
//! * [`fq_lem`] and the point writers store `x*R mod q` little-endian, which is what
//!   ffjavascript's `toRprLEM` dumps: the internal Montgomery limbs, verbatim.
//! * [`fr_plain`] strips Montgomery; it is what a `.r1cs` coefficient and both header
//!   primes use.
//! * [`fr_double_montgomery`] writes `v * R^2`, which is what and only what zkey section 4
//!   wants. snarkjs' own reader confirms it by multiplying back by `R^-2`
//!   (`readFr2`, `zkey_utils.js:443-446`).
//!
//! Nothing here writes the big-endian non-Montgomery form. That encoding never reaches a
//! file body, only a hash or a challenge file, and it lives in [`crate::transcript`] on
//! purpose.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::OnceLock;

use ark_ff::BigInt;
use g16_field::{AffineRepr, Field, Fq, Fr, G1Affine, G2Affine, PrimeField};

use crate::{CeremonyError, N8, SG1, SG2};

/// Points per buffered write in the slice and repeat helpers. 8192 G2 points is a 1 MB
/// staging buffer, which is small enough to stay resident and large enough that the
/// syscall count stops mattering on a 2^28 section.
const POINT_BATCH: usize = 8192;

/// A binfile under construction.
///
/// The handle is private on purpose: the length backfill seeks behind the caller's back,
/// so anything that could reach the file could also leave a section header claiming a
/// length that is not there.
pub struct BinFileWriter {
    out: BufWriter<File>,
    /// What `create` promised in the preamble.
    declared: u32,
    /// How many sections have actually been closed.
    written: u32,
    /// File offset of the reserved length u64 of the open section, if one is open.
    open_at: Option<u64>,
    /// Payload bytes written into the open section so far.
    section_len: u64,
}

impl BinFileWriter {
    /// Create the file and write `magic`, `version` and `n_sections`. `n_sections` is a
    /// promise: [`BinFileWriter::finish`] checks it.
    pub fn create(
        path: &Path,
        magic: &[u8; 4],
        version: u32,
        n_sections: u32,
    ) -> Result<Self, CeremonyError> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(magic)?;
        out.write_all(&version.to_le_bytes())?;
        out.write_all(&n_sections.to_le_bytes())?;
        Ok(Self {
            out,
            declared: n_sections,
            written: 0,
            open_at: None,
            section_len: 0,
        })
    }

    /// Write the entry header for `id` and reserve its length. One section may be open at
    /// a time; opening a second is a programming error, not a format one.
    pub fn start_section(&mut self, id: u32) -> Result<(), CeremonyError> {
        if self.open_at.is_some() {
            return Err(CeremonyError::malformed(
                id,
                "a section is already open; call end_section first",
            ));
        }
        self.out.write_all(&id.to_le_bytes())?;
        let at = self.out.stream_position()?;
        self.out.write_all(&0u64.to_le_bytes())?;
        self.open_at = Some(at);
        self.section_len = 0;
        Ok(())
    }

    /// Seek back over the payload, write the real length, and return to the end.
    pub fn end_section(&mut self) -> Result<(), CeremonyError> {
        let at = self
            .open_at
            .take()
            .ok_or_else(|| CeremonyError::malformed(0, "end_section without start_section"))?;
        let end = self.out.stream_position()?;
        self.out.seek(SeekFrom::Start(at))?;
        self.out.write_all(&self.section_len.to_le_bytes())?;
        self.out.seek(SeekFrom::Start(end))?;
        self.written += 1;
        self.section_len = 0;
        Ok(())
    }

    /// Flush, and check the section count matches what `create` declared.
    pub fn finish(mut self) -> Result<(), CeremonyError> {
        if self.open_at.is_some() {
            return Err(CeremonyError::malformed(
                0,
                "finish with a section still open",
            ));
        }
        if self.written != self.declared {
            return Err(CeremonyError::malformed(
                0,
                format!(
                    "declared {} sections, wrote {}",
                    self.declared, self.written
                ),
            ));
        }
        self.out.flush()?;
        Ok(())
    }

    /// Payload bytes written into the open section so far. Both contribution writers need
    /// it, because `paramLength` is a count they emit before the params themselves.
    pub fn section_len(&self) -> u64 {
        self.section_len
    }

    /// Every payload byte goes through here, which is why the "is a section open" check
    /// lives here and not in each typed writer. Writing outside a section would land bytes
    /// between two entry headers, where the scanner reads them as the next entry's id and
    /// length: the file then parses, into sections that are not the ones written.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), CeremonyError> {
        if self.open_at.is_none() {
            return Err(CeremonyError::malformed(
                0,
                "write outside a section; call start_section first",
            ));
        }
        self.out.write_all(bytes)?;
        self.section_len += bytes.len() as u64;
        Ok(())
    }

    pub fn write_u8(&mut self, v: u8) -> Result<(), CeremonyError> {
        self.write_bytes(&[v])
    }

    /// Little-endian, like every integer in these formats except `hashU32`, which is a
    /// hash input and not a file field. See
    /// [`crate::transcript::Transcript::update_u32_be`].
    pub fn write_u32(&mut self, v: u32) -> Result<(), CeremonyError> {
        self.write_bytes(&v.to_le_bytes())
    }

    pub fn write_u64(&mut self, v: u64) -> Result<(), CeremonyError> {
        self.write_bytes(&v.to_le_bytes())
    }

    /// `u32 n8` then the modulus as a plain little-endian integer. Both headers store the
    /// prime this way and neither stores a curve name, so this field is the only thing
    /// identifying the curve.
    pub fn write_prime(&mut self, modulus_le: &[u8; N8]) -> Result<(), CeremonyError> {
        self.write_u32(N8 as u32)?;
        self.write_bytes(modulus_le)
    }

    /// A base-field coordinate in the stored Montgomery form.
    pub fn write_fq(&mut self, v: &Fq) -> Result<(), CeremonyError> {
        self.write_bytes(&fq_lem(v))
    }

    /// A scalar in ordinary form, which is what a `.r1cs` coefficient uses.
    pub fn write_fr_plain(&mut self, v: &Fr) -> Result<(), CeremonyError> {
        self.write_bytes(&fr_plain(v))
    }

    /// A scalar as `v * R^2`. zkey section 4 and nothing else.
    pub fn write_fr_double_montgomery(&mut self, v: &Fr) -> Result<(), CeremonyError> {
        self.write_bytes(&fr_double_montgomery(v))
    }

    pub fn write_g1(&mut self, p: &G1Affine) -> Result<(), CeremonyError> {
        self.write_bytes(&g1_lem(p))
    }

    pub fn write_g2(&mut self, p: &G2Affine) -> Result<(), CeremonyError> {
        self.write_bytes(&g2_lem(p))
    }

    /// A run of G1 points through one staging buffer. The point sections are the bulk of
    /// both file formats, so they are not written a `write_g1` at a time.
    pub fn write_g1_slice(&mut self, points: &[G1Affine]) -> Result<(), CeremonyError> {
        let mut buf = Vec::with_capacity(POINT_BATCH.min(points.len().max(1)) * SG1);
        for batch in points.chunks(POINT_BATCH) {
            buf.clear();
            for p in batch {
                buf.extend_from_slice(&g1_lem(p));
            }
            self.write_bytes(&buf)?;
        }
        Ok(())
    }

    pub fn write_g2_slice(&mut self, points: &[G2Affine]) -> Result<(), CeremonyError> {
        let mut buf = Vec::with_capacity(POINT_BATCH.min(points.len().max(1)) * SG2);
        for batch in points.chunks(POINT_BATCH) {
            buf.clear();
            for p in batch {
                buf.extend_from_slice(&g2_lem(p));
            }
            self.write_bytes(&buf)?;
        }
        Ok(())
    }

    /// The same point `n` times, which is exactly what `powersoftau new` writes into
    /// sections 2 to 6 (`powersoftau_new.js:84-126`). A fresh file is fully determined by
    /// `power`: no randomness, no timestamps, nothing derived from the challenge hash it
    /// returns.
    pub fn write_g1_repeated(&mut self, p: &G1Affine, n: usize) -> Result<(), CeremonyError> {
        let one = g1_lem(p);
        let mut buf = Vec::with_capacity(POINT_BATCH.min(n.max(1)) * SG1);
        for _ in 0..POINT_BATCH.min(n.max(1)) {
            buf.extend_from_slice(&one);
        }
        let mut left = n;
        while left > 0 {
            let take = left.min(POINT_BATCH);
            self.write_bytes(&buf[..take * SG1])?;
            left -= take;
        }
        Ok(())
    }

    pub fn write_g2_repeated(&mut self, p: &G2Affine, n: usize) -> Result<(), CeremonyError> {
        let one = g2_lem(p);
        let mut buf = Vec::with_capacity(POINT_BATCH.min(n.max(1)) * SG2);
        for _ in 0..POINT_BATCH.min(n.max(1)) {
            buf.extend_from_slice(&one);
        }
        let mut left = n;
        while left > 0 {
            let take = left.min(POINT_BATCH);
            self.write_bytes(&buf[..take * SG2])?;
            left -= take;
        }
        Ok(())
    }

    /// Copy a section body straight from another file. `zkey contribute` copies sections
    /// 3 to 7 verbatim (`zkey_contribute.js:76-88`), and `zkey verify` then requires them
    /// to be byte-identical (`zkey_verify_frominit.js:169-197`), so a re-encode here would
    /// fail a check whose error message says nothing about re-encoding.
    pub fn write_section_verbatim(&mut self, id: u32, body: &[u8]) -> Result<(), CeremonyError> {
        self.start_section(id)?;
        self.write_bytes(body)?;
        self.end_section()
    }
}

/// The element whose *value* is `R mod r`, the inverse of the one
/// [`g16_zkey::binfile::r_inv`] names. One inversion, once, so the section-4 encoder is a
/// single multiply per coefficient.
fn r_value() -> &'static Fr {
    static R: OnceLock<Fr> = OnceLock::new();
    R.get_or_init(|| {
        g16_zkey::binfile::r_inv()
            .inverse()
            .expect("R is a unit mod r")
    })
}

/// Four u64 limbs as 32 little-endian bytes.
fn limbs_le(b: &BigInt<4>) -> [u8; N8] {
    let mut out = [0u8; N8];
    for (i, limb) in b.0.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

/// `x * R mod q`, little-endian: the Montgomery limbs as ffjavascript stores them, and the
/// exact inverse of [`g16_zkey::binfile::fq`].
pub fn fq_lem(v: &Fq) -> [u8; N8] {
    limbs_le(&v.0)
}

/// The ordinary value, little-endian. No Montgomery.
pub fn fr_plain(v: &Fr) -> [u8; N8] {
    limbs_le(&v.into_bigint())
}

/// `v * R^2`, little-endian. zkey section 4 only.
///
/// The stored integer is the Montgomery representation of the element whose *value* is
/// `v * R`, since a Montgomery representation is `value * R`. So multiply by `R` and take
/// the limbs rather than the value. Round-trips against
/// [`g16_zkey::binfile::fr_double_montgomery`], which reads the limbs back as Montgomery
/// (landing on `v * R`) and multiplies by `R^-1`.
pub fn fr_double_montgomery(v: &Fr) -> [u8; N8] {
    limbs_le(&(*v * r_value()).0)
}

/// Affine `x` then `y`, each [`fq_lem`]. The point at infinity is all zero bytes, which is
/// not a curve point and carries no flag bit: `g1m_isZeroAffine` tests `x == 0 && y == 0`
/// (`build_curve_jacobian_a0.js:50-71`), and `0 * R = 0` so Montgomery form does not
/// change it.
pub fn g1_lem(p: &G1Affine) -> [u8; SG1] {
    let mut out = [0u8; SG1];
    if p.is_zero() {
        return out;
    }
    out[..N8].copy_from_slice(&fq_lem(&p.x));
    out[N8..].copy_from_slice(&fq_lem(&p.y));
    out
}

/// `x.c0, x.c1, y.c0, y.c1`, each [`fq_lem`]. Note `c0` first: the big-endian hashing form
/// in [`crate::transcript`] puts `c1` first, and the two orders are one function apart.
pub fn g2_lem(p: &G2Affine) -> [u8; SG2] {
    let mut out = [0u8; SG2];
    if p.is_zero() {
        return out;
    }
    out[..N8].copy_from_slice(&fq_lem(&p.x.c0));
    out[N8..2 * N8].copy_from_slice(&fq_lem(&p.x.c1));
    out[2 * N8..3 * N8].copy_from_slice(&fq_lem(&p.y.c0));
    out[3 * N8..].copy_from_slice(&fq_lem(&p.y.c1));
    out
}
