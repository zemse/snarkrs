//! The iden3 "binfile" container shared by `.zkey`, `.wtns`, `.ptau` and `.r1cs`, plus the
//! field and point decoders every section needs. Only the 4-byte magic and the version
//! differ between the four, which is why [`BinFile::open`] takes the magic as a parameter
//! rather than hardcoding `zkey`.
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

use crate::ZkeyError;

pub const FQ_BYTES: usize = 32;
pub const G1_BYTES: usize = FQ_BYTES * 2;
pub const G2_BYTES: usize = FQ_BYTES * 4;
pub const FR_BYTES: usize = 32;

/// Where a binfile's bytes live. Both variants deref to `&[u8]`, so nothing below this
/// type, and no decoder in `lib.rs` or `wtns.rs`, knows which one it got.
///
/// The split exists because the browser has no filesystem to map. `wasm32-unknown-unknown`
/// has no `open(2)`: a zkey arrives over `fetch` and is written into linear memory, so an
/// owned `Vec<u8>` is the only backing that can exist there.
///
/// This is not a build fix, and it would be dishonest to sell it as one. Checked before
/// the change: memmap2 0.9.11 compiles for wasm32 and `cargo build --target
/// wasm32-unknown-unknown -p g16-zkey` already succeeded. What it fixes is an API that
/// links and then cannot work, because `File::open` on that target fails at runtime for
/// every path there is.
///
/// `Mapped` stays the native default and is not a micro-optimisation: `js_16x16_d32`'s
/// zkey is 94.4 MB (`bench/artifacts/manifest.csv`) and mmap keeps it out of the process
/// entirely, paged in on demand by the sections we actually read.
pub enum Backing {
    #[cfg(not(target_family = "wasm"))]
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl core::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// One entry of the section chain. `len` is what the file will hand out, which for an
/// intact file is exactly the declared length; see [`Scan::Lenient`] for the other case.
#[derive(Clone, Copy, Debug)]
pub struct Section {
    pub id: u32,
    /// Offset of the payload, one byte past the 12-byte entry header.
    pub start: usize,
    pub len: usize,
}

/// How far a short file gets to go before the scan gives up.
///
/// `Strict` is what every proving-path reader wants and is the only behaviour that existed
/// before: a declared length that runs past EOF is a hard error, because a zkey whose tail
/// is missing cannot prove anything and failing at open beats failing inside an MSM.
///
/// `Lenient` exists for `g16 ptau info`, which has to *report* a truncated download rather
/// than refuse to look at it. It clamps the offending section to the bytes that are
/// actually there, records the shortfall in [`BinFile::truncation`] and stops. binfileutils
/// has no such mode; this one is ours.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scan {
    Strict,
    Lenient,
}

/// Where a lenient scan ran out of file.
#[derive(Clone, Copy, Debug)]
pub struct Truncation {
    /// The section whose payload is short, or `None` when the chain was cut inside a
    /// 12-byte entry header and there is no id to name.
    pub section: Option<u32>,
    pub declared: u64,
    pub present: u64,
}

/// A binfile with its section chain indexed.
pub struct BinFile {
    data: Backing,
    /// Entries in file order. A handful of them, so a linear scan beats a map and keeps
    /// duplicate ids addressable.
    sections: Vec<Section>,
    /// `nSections` as the header claims, which `Lenient` may not have reached.
    declared_sections: usize,
    truncation: Option<Truncation>,
}

impl BinFile {
    /// Map the file and index it. Native only, because there is no filesystem to map on
    /// wasm; the browser path is [`BinFile::from_bytes`].
    #[cfg(not(target_family = "wasm"))]
    pub fn open(
        path: &std::path::Path,
        magic: &[u8; 4],
        max_version: u32,
    ) -> Result<Self, ZkeyError> {
        Self::open_with(path, magic, max_version, Scan::Strict)
    }

    /// [`BinFile::open`] with the scan mode spelled out.
    #[cfg(not(target_family = "wasm"))]
    pub fn open_with(
        path: &std::path::Path,
        magic: &[u8; 4],
        max_version: u32,
        scan: Scan,
    ) -> Result<Self, ZkeyError> {
        let file = std::fs::File::open(path)?;
        // Safety: we only ever hand out shared slices of the mapping, and the mapping
        // outlives them. A concurrent truncation of the file would be UB, which is the
        // standard and unavoidable caveat of mmap on any key file.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Self::index(Backing::Mapped(map), magic, max_version, scan)
    }

    /// Index a binfile already in memory. Takes the `Vec` by value rather than a slice
    /// because the wasm caller has just written a 94 MB zkey into linear memory and a
    /// borrow would force it to keep a second owner alive for the life of the key; the
    /// whole point of the byte path is that the file exists exactly once.
    pub fn from_bytes(
        bytes: Vec<u8>,
        magic: &[u8; 4],
        max_version: u32,
    ) -> Result<Self, ZkeyError> {
        Self::index(Backing::Owned(bytes), magic, max_version, Scan::Strict)
    }

    /// [`BinFile::from_bytes`] with the scan mode spelled out.
    pub fn from_bytes_with(
        bytes: Vec<u8>,
        magic: &[u8; 4],
        max_version: u32,
        scan: Scan,
    ) -> Result<Self, ZkeyError> {
        Self::index(Backing::Owned(bytes), magic, max_version, scan)
    }

    /// Header check plus the one scan of the section chain, shared by both constructors so
    /// the mmap path and the byte path cannot drift apart on a bounds check. Every one of
    /// them is load-bearing: the header is attacker-controlled, and the offsets recorded
    /// here are the only thing standing between a lying section length and a panic inside
    /// `unique_section`.
    fn index(
        data: Backing,
        magic: &[u8; 4],
        max_version: u32,
        scan: Scan,
    ) -> Result<Self, ZkeyError> {
        if data.len() < 12 {
            return Err(ZkeyError::BadMagic([0; 4]));
        }
        let got: [u8; 4] = data[0..4].try_into().expect("slice is 4 bytes");
        if &got != magic {
            return Err(ZkeyError::BadMagic(got));
        }
        let version = u32_at(&data, 4);
        if version > max_version {
            return Err(ZkeyError::Malformed {
                section: 0,
                reason: format!("version {version} exceeds supported {max_version}"),
            });
        }
        let n_sections = u32_at(&data, 8) as usize;

        // Bounded by the file: a lying `nSections` of 2^32-1 would otherwise reserve 96 GB
        // before the loop reads a byte. Twelve is the smallest an entry can be.
        let mut sections = Vec::with_capacity(n_sections.min(data.len() / 12));
        let mut truncation = None;
        let mut pos = 12usize;
        for i in 0..n_sections {
            if pos + 12 > data.len() {
                if scan == Scan::Lenient {
                    truncation = Some(Truncation {
                        section: None,
                        declared: 12,
                        present: (data.len() - pos.min(data.len())) as u64,
                    });
                    break;
                }
                return Err(ZkeyError::Malformed {
                    section: 0,
                    reason: format!("section header {i} runs past end of file"),
                });
            }
            let id = u32_at(&data, pos);
            let len = u64_at(&data, pos + 4) as usize;
            pos += 12;
            let end = pos.checked_add(len).ok_or_else(|| ZkeyError::Malformed {
                section: id,
                reason: "section length overflows".into(),
            })?;
            if end > data.len() {
                if scan == Scan::Lenient {
                    let present = data.len() - pos;
                    sections.push(Section {
                        id,
                        start: pos,
                        len: present,
                    });
                    truncation = Some(Truncation {
                        section: Some(id),
                        declared: len as u64,
                        present: present as u64,
                    });
                    break;
                }
                return Err(ZkeyError::Malformed {
                    section: id,
                    reason: format!("length {len} runs past end of file"),
                });
            }
            sections.push(Section {
                id,
                start: pos,
                len,
            });
            pos = end;
        }

        Ok(Self {
            data,
            sections,
            declared_sections: n_sections,
            truncation,
        })
    }

    /// Every entry of the section chain, in file order, duplicates included.
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// `nSections` as the header declares it. `createBinFile` writes this before any
    /// section exists and never revises it (`binfileutils.js:46`), so it is a claim about
    /// the file rather than a fact derived from it, and `sections().len()` can be smaller.
    pub fn declared_sections(&self) -> usize {
        self.declared_sections
    }

    /// Where a [`Scan::Lenient`] index ran out of file, `None` when the chain is intact.
    pub fn truncation(&self) -> Option<Truncation> {
        self.truncation
    }

    /// Total bytes of the file, so a caller can check the chain ends exactly at EOF.
    pub fn total_len(&self) -> usize {
        self.data.len()
    }

    /// The whole file. Needed by readers that must address bytes outside any section,
    /// which is not a hypothetical: `zkey_new.js:517` reads one G1 point past the end of
    /// ptau section 2 and folds it into `csHash`.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// The one section with this id. Duplicates are a format error for every section we
    /// read, so rejecting them here beats silently taking the first.
    pub fn unique_section(&self, id: u32) -> Result<&[u8], ZkeyError> {
        let mut found = None;
        for &Section {
            id: sid,
            start,
            len,
        } in &self.sections
        {
            if sid == id {
                if found.is_some() {
                    return Err(ZkeyError::Malformed {
                        section: id,
                        reason: "duplicated section".into(),
                    });
                }
                found = Some(&self.data[start..start + len]);
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

    pub fn u8(&mut self) -> Result<u8, ZkeyError> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32, ZkeyError> {
        Ok(u32_at(self.take(4)?, 0))
    }

    pub fn u64(&mut self) -> Result<u64, ZkeyError> {
        Ok(u64_at(self.take(8)?, 0))
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Bytes consumed so far. The ceremony readers need it because both param blobs
    /// self-check by requiring the cursor to land exactly on the section end
    /// (`powersoftau_utils.js:239`, `zkey_utils.js:481-483`).
    pub fn pos(&self) -> usize {
        self.pos
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
