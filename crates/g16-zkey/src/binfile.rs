//! The iden3 "binfile" container shared by `.zkey` and `.wtns`, plus the field and point
//! decoders every section needs.
//!
//! Layout (`@iden3/binfileutils`): 4 magic bytes, u32 version, u32 nSections, then a flat
//! run of `u32 sectionId, u64 sectionLength, payload`. Ids are neither ordered nor unique,
//! so the only safe read is to scan the whole chain once and index it.
//!
//! # Montgomery, the part that silently corrupts proofs
//!
//! snarkjs mixes three different encodings of a field element in one file, and picking
//! the wrong one produces a proof that fails to verify with no other symptom:
//!
//! * **Curve coordinates** (sections 2, 3, 5-9) go through ffjavascript's `toRprLEM`,
//!   which dumps the internal Montgomery limbs verbatim. Stored integer is `x*R mod q`.
//!   Decoded with [`fq`], which installs the limbs as-is via `new_unchecked`.
//! * **Section 4 coefficients** are written as `toRprLE(Fr.mul(v, R2))`, that is
//!   `fromMontgomery(v*R * R^2*R) = v*R^2`. **Double** Montgomery, not single. Decoded
//!   with [`fr_double_montgomery`]. snarkjs' own reader confirms this by multiplying by
//!   `R^-2` (`readFr2` in `zkey_utils.js`), and its prover confirms it a second way: it
//!   feeds the raw bytes to a Montgomery `mul` against a *normal-form* witness, so the
//!   two conversions cancel to one.
//! * **Witness values** (`.wtns` section 2) are plain little-endian integers. snarkjs
//!   reads them straight into `publicSignals` with `Scalar.fromRprLE` and hands the same
//!   bytes to a multiexp that consumes normal-form scalars. Decoded with [`fr_normal`].

use ark_ff::BigInt;
use g16_field::*;
use memmap2::Mmap;

use crate::ZkeyError;

pub const FQ_BYTES: usize = 32;
pub const G1_BYTES: usize = FQ_BYTES * 2;
pub const G2_BYTES: usize = FQ_BYTES * 4;
pub const FR_BYTES: usize = 32;

/// A memory-mapped binfile with its section chain indexed.
pub struct BinFile {
    map: Mmap,
    /// `(id, start, len)` in file order. A handful of entries, so a linear scan beats a
    /// map and keeps duplicate ids addressable.
    sections: Vec<(u32, usize, usize)>,
}

impl BinFile {
    pub fn open(
        path: &std::path::Path,
        magic: &[u8; 4],
        max_version: u32,
    ) -> Result<Self, ZkeyError> {
        let file = std::fs::File::open(path)?;
        // Safety: we only ever hand out shared slices of the mapping, and the mapping
        // outlives them. A concurrent truncation of the file would be UB, which is the
        // standard and unavoidable caveat of mmap on any key file.
        let map = unsafe { Mmap::map(&file)? };

        if map.len() < 12 {
            return Err(ZkeyError::BadMagic([0; 4]));
        }
        let got: [u8; 4] = map[0..4].try_into().expect("slice is 4 bytes");
        if &got != magic {
            return Err(ZkeyError::BadMagic(got));
        }
        let version = u32_at(&map, 4);
        if version > max_version {
            return Err(ZkeyError::Malformed {
                section: 0,
                reason: format!("version {version} exceeds supported {max_version}"),
            });
        }
        let n_sections = u32_at(&map, 8) as usize;

        let mut sections = Vec::with_capacity(n_sections);
        let mut pos = 12usize;
        for i in 0..n_sections {
            if pos + 12 > map.len() {
                return Err(ZkeyError::Malformed {
                    section: 0,
                    reason: format!("section header {i} runs past end of file"),
                });
            }
            let id = u32_at(&map, pos);
            let len = u64_at(&map, pos + 4) as usize;
            pos += 12;
            let end = pos.checked_add(len).ok_or_else(|| ZkeyError::Malformed {
                section: id,
                reason: "section length overflows".into(),
            })?;
            if end > map.len() {
                return Err(ZkeyError::Malformed {
                    section: id,
                    reason: format!("length {len} runs past end of file"),
                });
            }
            sections.push((id, pos, len));
            pos = end;
        }

        Ok(Self { map, sections })
    }

    /// The one section with this id. Duplicates are a format error for every section we
    /// read, so rejecting them here beats silently taking the first.
    pub fn unique_section(&self, id: u32) -> Result<&[u8], ZkeyError> {
        let mut found = None;
        for &(sid, start, len) in &self.sections {
            if sid == id {
                if found.is_some() {
                    return Err(ZkeyError::Malformed {
                        section: id,
                        reason: "duplicated section".into(),
                    });
                }
                found = Some(&self.map[start..start + len]);
            }
        }
        found.ok_or(ZkeyError::MissingSection(id))
    }
}

/// A forward-only reader over one section, so the header parse reads like the spec.
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    section: u32,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8], section: u32) -> Self {
        Self {
            data,
            pos: 0,
            section,
        }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], ZkeyError> {
        let end = self.pos.checked_add(n).ok_or_else(|| self.short(n))?;
        if end > self.data.len() {
            return Err(self.short(n));
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn u32(&mut self) -> Result<u32, ZkeyError> {
        Ok(u32_at(self.take(4)?, 0))
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn short(&self, want: usize) -> ZkeyError {
        ZkeyError::Malformed {
            section: self.section,
            reason: format!(
                "wanted {want} bytes at offset {}, only {} left",
                self.pos,
                self.data.len().saturating_sub(self.pos)
            ),
        }
    }
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("slice is 4 bytes"))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("slice is 8 bytes"))
}

/// 32 little-endian bytes as a 4-limb bigint, no reduction and no Montgomery conversion.
pub fn bigint(b: &[u8]) -> BigInt<4> {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u64_at(b, i * 8);
    }
    BigInt::new(limbs)
}

/// A base-field coordinate as snarkjs stores it: the Montgomery limbs verbatim.
pub fn fq(b: &[u8]) -> Fq {
    Fq::new_unchecked(bigint(b))
}

/// A scalar stored in ordinary (non-Montgomery) form. Rejects a value at or above the
/// modulus rather than reducing it, because a witness that needs reducing is a bug
/// upstream, not something to paper over.
pub fn fr_normal(b: &[u8], section: u32) -> Result<Fr, ZkeyError> {
    Fr::from_bigint(bigint(b)).ok_or_else(|| ZkeyError::Malformed {
        section,
        reason: "field element is not below the scalar modulus".into(),
    })
}

/// A section-4 coefficient: the stored integer is `v * R^2`, so one `new_unchecked`
/// (which divides by `R` by reinterpreting the limbs) leaves `v * R`, and multiplying by
/// the element whose value is `R^-1` finishes the job.
pub fn fr_double_montgomery(b: &[u8], r_inv: &Fr) -> Fr {
    Fr::new_unchecked(bigint(b)) * r_inv
}

/// The element whose *value* is `R^-1 mod r`: limbs `1` reinterpreted as Montgomery.
pub fn r_inv() -> Fr {
    Fr::new_unchecked(BigInt::new([1, 0, 0, 0]))
}

/// snarkjs writes the point at infinity as the affine pair `(0, 0)`, which is not on the
/// curve. Every other decoder in the pipeline would then reject or, worse, silently
/// mangle it, so the mapping has to happen here.
pub fn g1(b: &[u8]) -> G1Affine {
    let x = fq(&b[..FQ_BYTES]);
    let y = fq(&b[FQ_BYTES..G1_BYTES]);
    if x.is_zero() && y.is_zero() {
        G1Affine::identity()
    } else {
        G1Affine::new_unchecked(x, y)
    }
}

/// `Fq2` components are stored `c0` then `c1`, matching the `[[x0, x1], ...]` order in
/// `verification_key.json`.
pub fn g2(b: &[u8]) -> G2Affine {
    let x = Fq2::new(fq(&b[..FQ_BYTES]), fq(&b[FQ_BYTES..2 * FQ_BYTES]));
    let y = Fq2::new(
        fq(&b[2 * FQ_BYTES..3 * FQ_BYTES]),
        fq(&b[3 * FQ_BYTES..G2_BYTES]),
    );
    if x.is_zero() && y.is_zero() {
        G2Affine::identity()
    } else {
        G2Affine::new_unchecked(x, y)
    }
}

/// Checks a section holds exactly `n` records of `stride` bytes. Getting this wrong is
/// how an off-by-one in one section quietly shifts every later one.
pub fn expect_records(data: &[u8], n: usize, stride: usize, section: u32) -> Result<(), ZkeyError> {
    let want = n * stride;
    if data.len() != want {
        return Err(ZkeyError::Malformed {
            section,
            reason: format!(
                "expected {n} records of {stride} bytes ({want}), got {}",
                data.len()
            ),
        });
    }
    Ok(())
}
