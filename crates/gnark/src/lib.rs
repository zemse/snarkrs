//! gnark v0.14.0 Groth16 BN254 files: the proving key `WriteDump`, the verifying key
//! `WriteTo`, and the witness dump agreed for SP1 (`unsafe.WriteMarker`, then
//! `unsafe.WriteSlice` of W, A.w and B.w).
//!
//! gnark writes two unrelated encodings, and every reader here has to know which one it is
//! looking at:
//!
//! * **Encoder points** (`curve.Encoder`, the dump's five header points, the verifying key,
//!   the proof) are canonical big-endian coordinates, `X` then `Y`, and for G2 `A1` before
//!   `A0`. The top two bits of the first byte are a flag: `00` uncompressed, `01`
//!   compressed infinity, `10` and `11` compressed with the lexicographically smaller and
//!   larger `Y`. Decoded with [`encoded_g1`] and [`encoded_g2`].
//! * **Dumped slices** (`unsafe.WriteSlice`) are a `u64` little-endian length and then the
//!   slice's memory verbatim: `fp.Element` and `fr.Element` are four Montgomery limbs,
//!   little-endian, and a G2 coordinate is `A0` before `A1`. That is the layout snarkjs
//!   writes for zkey points, so [`snarkrs_formats::binfile::g1`] and `g2` decode it
//!   unchanged, the point at infinity being `(0, 0)` in both.
//!
//! Every length in either encoding is bounded by the bytes actually present before
//! anything is allocated from it. gnark's own `unsafe.ReadSlice` allocates whatever the
//! length says, which is the CVE-2024-50354 class the zkey loader guards against too.

pub mod pk;
pub mod proof;
pub mod vk;
pub mod witness;

pub use pk::{GnarkDump, GnarkProvingKey, Query};
pub use vk::GnarkVerifyingKey;
pub use witness::GnarkWitness;

use ark_ec::short_weierstrass::SWCurveConfig;
use snarkrs_field::*;
use snarkrs_formats::binfile::bigint;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GnarkError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The first eight bytes of a dump. gnark writes `0xdeadbeef` as a native `u64`, so a
    /// mismatch is either not a dump or one written on a big-endian machine.
    #[error("not a little-endian gnark dump (marker {0:02x?})")]
    BadMarker([u8; 8]),
    #[error("malformed {part}: {reason}")]
    Malformed { part: &'static str, reason: String },
    /// BSB22 commitments (`api.Commit`) add a Pedersen key and a proof element this crate
    /// has no prover for.
    #[error("{part} has {count} commitment keys; gnark's BSB22 commitments are not supported")]
    Commitments { part: &'static str, count: u64 },
    #[error("{part}: {reason}")]
    Mismatch { part: &'static str, reason: String },
}

/// The marker `unsafe.WriteMarker` writes: `0xdeadbeef` as a little-endian `u64`.
pub const DUMP_MARKER: [u8; 8] = 0xdead_beef_u64.to_le_bytes();

/// Bytes of one dumped `fr.Element`.
pub const FR_BYTES: usize = 32;

/// A forward-only reader over a gnark file, so each parse reads like gnark's own.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    part: &'static str,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8], part: &'static str) -> Self {
        Self { data, pos: 0, part }
    }

    /// Name what the bytes that follow belong to, for the error messages.
    pub(crate) fn part(&mut self, part: &'static str) {
        self.part = part;
    }

    pub(crate) fn bad(&self, reason: String) -> GnarkError {
        GnarkError::Malformed {
            part: self.part,
            reason,
        }
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], GnarkError> {
        let left = self.remaining();
        if n > left {
            return Err(self.bad(format!(
                "wanted {n} bytes at offset {}, only {left} left",
                self.pos
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub(crate) fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GnarkError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32_be(&mut self) -> Result<u32, GnarkError> {
        Ok(u32::from_be_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    pub(crate) fn u64_be(&mut self) -> Result<u64, GnarkError> {
        Ok(u64::from_be_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub(crate) fn u64_le(&mut self) -> Result<u64, GnarkError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// `n` records of `stride` bytes, refused unless every one of them is present. The
    /// length is the file's to claim, so the product is checked rather than trusted.
    pub(crate) fn records(&mut self, n: u64, stride: usize) -> Result<&'a [u8], GnarkError> {
        let bytes = usize::try_from(n)
            .ok()
            .and_then(|n| n.checked_mul(stride))
            .filter(|&b| b <= self.remaining())
            .ok_or_else(|| {
                self.bad(format!(
                    "{n} records of {stride} bytes, only {} bytes left",
                    self.remaining()
                ))
            })?;
        self.take(bytes)
    }

    /// An `unsafe.WriteSlice`: a little-endian `u64` count, then `count * stride` bytes.
    pub(crate) fn dumped_slice(&mut self, stride: usize) -> Result<&'a [u8], GnarkError> {
        let n = self.u64_le()?;
        self.records(n, stride)
    }

    pub(crate) fn finish(self) -> Result<(), GnarkError> {
        if self.remaining() != 0 {
            return Err(self.bad(format!("{} trailing bytes", self.remaining())));
        }
        Ok(())
    }
}

/// Top two bits of an encoder point's first byte.
const FLAG_MASK: u8 = 0b11 << 6;
const UNCOMPRESSED: u8 = 0b00 << 6;
const COMPRESSED_INFINITY: u8 = 0b01 << 6;
const COMPRESSED_SMALLEST: u8 = 0b10 << 6;
const COMPRESSED_LARGEST: u8 = 0b11 << 6;

/// Bytes of a compressed and an uncompressed encoder G1 point. G2 is twice each.
pub const G1_COMPRESSED: usize = 32;
pub const G1_UNCOMPRESSED: usize = 64;
pub const G2_COMPRESSED: usize = 64;
pub const G2_UNCOMPRESSED: usize = 128;

/// A canonical big-endian base-field element, refused at or above the modulus as gnark's
/// `SetBytesCanonical` refuses it.
fn fq_be(b: &[u8]) -> Option<Fq> {
    let mut le = [0u8; 32];
    le.copy_from_slice(&b[..32]);
    le.reverse();
    Fq::from_bigint(bigint(&le))
}

/// A canonical big-endian scalar.
pub(crate) fn fr_be(b: &[u8]) -> Option<Fr> {
    let mut le = [0u8; 32];
    le.copy_from_slice(&b[..32]);
    le.reverse();
    Fr::from_bigint(bigint(&le))
}

/// `x` as gnark writes a coordinate, 32 bytes big-endian.
pub(crate) fn fq_to_be(x: &Fq) -> [u8; 32] {
    x.into_bigint()
        .to_bytes_be()
        .try_into()
        .expect("a BN254 base-field element is 32 bytes")
}

/// gnark's `LexicographicallyLargest`: above `(q - 1) / 2` as an integer.
fn fq_largest(y: &Fq) -> bool {
    y.into_bigint() > Fq::MODULUS_MINUS_ONE_DIV_TWO
}

/// gnark's `E2.LexicographicallyLargest`: decided by `A1`, or by `A0` when `A1` is zero.
fn fq2_largest(y: &Fq2) -> bool {
    if y.c1.is_zero() {
        fq_largest(&y.c0)
    } else {
        fq_largest(&y.c1)
    }
}

/// The flag of the encoder point starting with `first`, and how many bytes it takes.
pub(crate) fn encoded_len(first: u8, compressed: usize) -> usize {
    if first & FLAG_MASK == UNCOMPRESSED {
        2 * compressed
    } else {
        compressed
    }
}

/// One encoder G1 point, compressed or not, checked on the curve. BN254 G1 has cofactor 1,
/// so on the curve is in the subgroup. `b` must be exactly the length [`encoded_len`] gives.
pub fn encoded_g1(b: &[u8]) -> Result<G1Affine, String> {
    let flag = b[0] & FLAG_MASK;
    let mut x_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&b[..32]);
    x_bytes[0] &= !FLAG_MASK;
    match flag {
        UNCOMPRESSED => {
            if b.iter().all(|&v| v == 0) {
                return Ok(G1Affine::identity());
            }
            let (x, y) = (fq_be(&x_bytes), fq_be(&b[32..64]));
            let (Some(x), Some(y)) = (x, y) else {
                return Err("coordinate is not below the base-field modulus".into());
            };
            let p = G1Affine::new_unchecked(x, y);
            if !p.is_on_curve() {
                return Err("point is not on the curve".into());
            }
            Ok(p)
        }
        COMPRESSED_INFINITY => {
            if x_bytes.iter().any(|&v| v != 0) {
                return Err("compressed infinity with nonzero bytes".into());
            }
            Ok(G1Affine::identity())
        }
        COMPRESSED_SMALLEST | COMPRESSED_LARGEST => {
            let x = fq_be(&x_bytes).ok_or("x is not below the base-field modulus")?;
            let y2 = x.square() * x + g1::Config::COEFF_B;
            let mut y = y2.sqrt().ok_or("x is not the abscissa of a curve point")?;
            if fq_largest(&y) != (flag == COMPRESSED_LARGEST) {
                y = -y;
            }
            Ok(G1Affine::new_unchecked(x, y))
        }
        _ => unreachable!("a two-bit flag has four values"),
    }
}

/// One encoder G2 point, compressed or not, checked on the curve and in the prime-order
/// subgroup, which gnark's own decoder checks too.
pub fn encoded_g2(b: &[u8]) -> Result<G2Affine, String> {
    let flag = b[0] & FLAG_MASK;
    let mut x1_bytes = [0u8; 32];
    x1_bytes.copy_from_slice(&b[..32]);
    x1_bytes[0] &= !FLAG_MASK;
    let p = match flag {
        UNCOMPRESSED => {
            if b.iter().all(|&v| v == 0) {
                return Ok(G2Affine::identity());
            }
            let coords = [&x1_bytes[..], &b[32..64], &b[64..96], &b[96..128]].map(fq_be);
            let [Some(x1), Some(x0), Some(y1), Some(y0)] = coords else {
                return Err("coordinate is not below the base-field modulus".into());
            };
            let p = G2Affine::new_unchecked(Fq2::new(x0, x1), Fq2::new(y0, y1));
            if !p.is_on_curve() {
                return Err("point is not on the curve".into());
            }
            p
        }
        COMPRESSED_INFINITY => {
            if x1_bytes.iter().chain(&b[32..64]).any(|&v| v != 0) {
                return Err("compressed infinity with nonzero bytes".into());
            }
            return Ok(G2Affine::identity());
        }
        COMPRESSED_SMALLEST | COMPRESSED_LARGEST => {
            let (x1, x0) = (fq_be(&x1_bytes), fq_be(&b[32..64]));
            let (Some(x1), Some(x0)) = (x1, x0) else {
                return Err("x is not below the base-field modulus".into());
            };
            let x = Fq2::new(x0, x1);
            let y2 = x.square() * x + g2::Config::COEFF_B;
            let mut y = y2.sqrt().ok_or("x is not the abscissa of a curve point")?;
            if fq2_largest(&y) != (flag == COMPRESSED_LARGEST) {
                y = -y;
            }
            G2Affine::new_unchecked(x, y)
        }
        _ => unreachable!("a two-bit flag has four values"),
    };
    if !p.is_in_correct_subgroup_assuming_on_curve() {
        return Err("point is not in the prime-order subgroup".into());
    }
    Ok(p)
}

/// `p` as gnark's `RawBytes`: uncompressed, all zero for infinity.
pub fn raw_g1(p: &G1Affine) -> [u8; G1_UNCOMPRESSED] {
    let mut out = [0u8; G1_UNCOMPRESSED];
    if !p.infinity {
        out[..32].copy_from_slice(&fq_to_be(&p.x));
        out[32..].copy_from_slice(&fq_to_be(&p.y));
    }
    out
}

/// `p` as gnark's `RawBytes`: `X.A1, X.A0, Y.A1, Y.A0`, all zero for infinity.
pub fn raw_g2(p: &G2Affine) -> [u8; G2_UNCOMPRESSED] {
    let mut out = [0u8; G2_UNCOMPRESSED];
    if !p.infinity {
        for (i, c) in [p.x.c1, p.x.c0, p.y.c1, p.y.c0].iter().enumerate() {
            out[32 * i..32 * (i + 1)].copy_from_slice(&fq_to_be(c));
        }
    }
    out
}

impl Reader<'_> {
    /// One encoder G1 point, whichever form its flag says.
    pub(crate) fn g1(&mut self, what: &str) -> Result<G1Affine, GnarkError> {
        let first = self
            .peek()
            .ok_or_else(|| self.bad(format!("{what}: no bytes left")))?;
        let b = self.take(encoded_len(first, G1_COMPRESSED))?;
        encoded_g1(b).map_err(|e| self.bad(format!("{what}: {e}")))
    }

    /// One encoder G2 point, whichever form its flag says.
    pub(crate) fn g2(&mut self, what: &str) -> Result<G2Affine, GnarkError> {
        let first = self
            .peek()
            .ok_or_else(|| self.bad(format!("{what}: no bytes left")))?;
        let b = self.take(encoded_len(first, G2_COMPRESSED))?;
        encoded_g2(b).map_err(|e| self.bad(format!("{what}: {e}")))
    }
}

/// A header point that every honest setup makes finite and valid. The zkey loader refuses
/// the same six for the same reason: a zeroed `beta` or `delta` switches zero knowledge
/// off without any verifier noticing. See `snarkrs_formats::ProvingKey::load`.
pub(crate) fn check_header_g1(
    p: &G1Affine,
    part: &'static str,
    what: &str,
) -> Result<(), GnarkError> {
    if p.infinity {
        return Err(GnarkError::Malformed {
            part,
            reason: format!(
                "{what} is the point at infinity, which no honest setup produces; \
                 this key cannot provide zero knowledge"
            ),
        });
    }
    if !(p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()) {
        return Err(GnarkError::Malformed {
            part,
            reason: format!("{what} is not a valid G1 point"),
        });
    }
    Ok(())
}

/// [`check_header_g1`] for G2.
pub(crate) fn check_header_g2(
    p: &G2Affine,
    part: &'static str,
    what: &str,
) -> Result<(), GnarkError> {
    if p.infinity {
        return Err(GnarkError::Malformed {
            part,
            reason: format!(
                "{what} is the point at infinity, which no honest setup produces; \
                 this key cannot provide zero knowledge"
            ),
        });
    }
    if !(p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()) {
        return Err(GnarkError::Malformed {
            part,
            reason: format!("{what} is not a valid G2 point"),
        });
    }
    Ok(())
}

/// Prove `witness` against a circuit prepared from [`GnarkProvingKey::into_proving_key`],
/// and check the proof against the circuit's verifying key before returning it, as
/// `snarkrs_groth16::prove::prove` does. [`proof::write_proof`] writes it for gnark.
///
/// Only the CPU backend proves a gnark key today; every other backend refuses it.
pub fn prove<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
    circuit: &dyn snarkrs_groth16::PreparedCircuit,
    witness: &GnarkWitness<'_>,
    rng: &mut R,
    timings: &mut snarkrs_groth16::StageTimings,
) -> Result<snarkrs_groth16::Proof, snarkrs_groth16::ProveError> {
    snarkrs_groth16::prove::prove_from_rows(
        circuit,
        witness.w(),
        witness.aw(),
        witness.bw(),
        rng,
        timings,
    )
}

/// The whole of a file, mapped. Native only, like every path-taking loader in snarkrs.
pub(crate) fn map(path: &std::path::Path) -> Result<memmap2::Mmap, GnarkError> {
    let file = std::fs::File::open(path)?;
    // Safety: only shared slices of the mapping are handed out, and none outlives it. A
    // concurrent truncation of the file would be UB, the standard caveat of mmap on any
    // key file, as in `snarkrs_formats::binfile::BinFile::open`.
    Ok(unsafe { memmap2::Mmap::map(&file)? })
}

#[cfg(test)]
mod tests;
