//! `.wtns` witness file parsing.
//!
//! Unlike the zkey, section 2 holds **plain** little-endian integers, not Montgomery
//! limbs. snarkjs proves it twice over: it lifts the public signals straight out of these
//! bytes with `Scalar.fromRprLE`, and it hands the same bytes to a multiexp whose scalars
//! are read as ordinary integers.

use crate::binfile::{self, BinFile, Cursor, FR_BYTES};
use g16_field::*;
use rayon::prelude::*;

/// The full witness vector `w = (1, public..., private...)`, length `n_vars`.
pub struct Witness(pub Vec<Fr>);

impl Witness {
    /// Native only; the browser has no filesystem, so it uses [`Witness::from_bytes`].
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: &std::path::Path) -> Result<Self, super::ZkeyError> {
        Self::parse(BinFile::open(path, b"wtns", 2)?)
    }

    /// A `.wtns` already in memory. The witness is small next to the zkey (4.5 MB at
    /// 140,261 constraints against 94.4 MB), but it is regenerated per proof in the
    /// browser, so it needs the same byte path.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, super::ZkeyError> {
        Self::parse(BinFile::from_bytes(bytes, b"wtns", 2)?)
    }

    fn parse(file: BinFile) -> Result<Self, super::ZkeyError> {
        let mut header = Cursor::new(file.unique_section(1)?, 1);
        let n8 = header.u32()? as usize;
        if n8 != FR_BYTES {
            return Err(super::ZkeyError::UnsupportedCurve);
        }
        if header.take(n8)? != Fr::MODULUS.to_bytes_le() {
            return Err(super::ZkeyError::UnsupportedCurve);
        }
        let n_witness = header.u32()? as usize;

        let data = file.unique_section(2)?;
        binfile::expect_records(data, n_witness, FR_BYTES, 2)?;
        let values = data
            .par_chunks_exact(FR_BYTES)
            .map(|b| binfile::fr_normal(b, 2))
            .collect::<Result<Vec<Fr>, _>>()?;

        // w[0] is the constant one wire. Every QAP row and the public-input part of the
        // verifier equation assume it, so a witness that fails here is unusable and the
        // failure is far cheaper to see now than as a proof that does not verify.
        if values.first() != Some(&Fr::ONE) {
            return Err(super::ZkeyError::Malformed {
                section: 2,
                reason: "witness[0] is not 1".into(),
            });
        }

        Ok(Self(values))
    }
}
