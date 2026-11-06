//! The solved gnark witness, as the dump agreed with the SP1 exporter:
//!
//! ```text
//!   unsafe.WriteMarker
//!   unsafe.WriteSlice W   []fr.Element, every wire in gnark's order, W[0] = 1
//!   unsafe.WriteSlice Aw  []fr.Element, A.w for each constraint
//!   unsafe.WriteSlice Bw  []fr.Element, B.w for each constraint
//! ```
//!
//! Each `fr.Element` is four Montgomery limbs, little-endian, which is how arkworks holds an
//! `Fr` too: both use `R = 2^256`. C is not in the file. On the constraint rows it is
//! `Aw * Bw`, which is exactly how snarkrs builds C from a zkey's two matrices.

use crate::{GnarkError, Reader, DUMP_MARKER, FR_BYTES};
use snarkrs_field::*;
use snarkrs_formats::binfile::{bigint, decode_records};

/// A solved witness: the wires and the two constraint-row vectors the prover needs.
/// Scrubbed on drop, like every other witness image in snarkrs.
pub struct GnarkWitness {
    w: Vec<Fr>,
    aw: Vec<Fr>,
    bw: Vec<Fr>,
}

impl GnarkWitness {
    /// Map a witness dump and decode it. 1.5 GB for SP1's circuit, read once and decoded in
    /// parallel straight into the three vectors.
    pub fn read_dump(path: &std::path::Path) -> Result<Self, GnarkError> {
        Self::from_bytes(&crate::map(path)?)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, GnarkError> {
        let mut r = Reader::new(data, "marker");
        let marker: [u8; 8] = r.take(8)?.try_into().expect("8 bytes");
        if marker != DUMP_MARKER {
            return Err(GnarkError::BadMarker(marker));
        }
        // All three lengths are checked against the file before any vector exists.
        r.part("W");
        let w = r.dumped_slice(FR_BYTES)?;
        r.part("Aw");
        let aw = r.dumped_slice(FR_BYTES)?;
        r.part("Bw");
        let bw = r.dumped_slice(FR_BYTES)?;
        r.part("trailer");
        r.finish()?;
        let (w, (aw, bw)) = rayon::join(
            || decode(w, "W"),
            || rayon::join(|| decode(aw, "Aw"), || decode(bw, "Bw")),
        );
        Self::from_parts(w?, aw?, bw?)
    }

    /// A witness already in memory. `aw` and `bw` must have one entry per constraint, and
    /// `w[0]` must be the constant one.
    pub fn from_parts(w: Vec<Fr>, aw: Vec<Fr>, bw: Vec<Fr>) -> Result<Self, GnarkError> {
        let witness = Self { w, aw, bw };
        if witness.aw.len() != witness.bw.len() {
            return Err(GnarkError::Malformed {
                part: "witness",
                reason: format!(
                    "Aw has {} rows and Bw has {}",
                    witness.aw.len(),
                    witness.bw.len()
                ),
            });
        }
        // Every QAP row and the public-input half of the verifier equation assume it, as in
        // the `.wtns` loader.
        if witness.w.first() != Some(&Fr::ONE) {
            return Err(GnarkError::Malformed {
                part: "W",
                reason: "W[0] is not 1".into(),
            });
        }
        Ok(witness)
    }

    /// Every wire, constant first.
    pub fn w(&self) -> &[Fr] {
        &self.w
    }

    /// `A.w`, one entry per constraint.
    pub fn aw(&self) -> &[Fr] {
        &self.aw
    }

    /// `B.w`, one entry per constraint.
    pub fn bw(&self) -> &[Fr] {
        &self.bw
    }

    /// Constraint rows.
    pub fn n_constraints(&self) -> usize {
        self.aw.len()
    }

    /// The `n_public` inputs after the constant wire, which is what a verifier takes.
    pub fn public(&self, n_public: usize) -> Option<&[Fr]> {
        self.w.get(1..n_public.checked_add(1)?)
    }
}

impl Drop for GnarkWitness {
    fn drop(&mut self) {
        snarkrs_groth16::scrub(&mut self.w);
        snarkrs_groth16::scrub(&mut self.aw);
        snarkrs_groth16::scrub(&mut self.bw);
    }
}

/// One dumped `[]fr.Element`: Montgomery limbs, refused at or above the modulus for the
/// reason `snarkrs_formats::binfile::fq` gives.
fn decode(data: &[u8], part: &'static str) -> Result<Vec<Fr>, GnarkError> {
    decode_records(data, FR_BYTES, Fr::zero(), |i, b| {
        let limbs = bigint(b);
        if limbs >= Fr::MODULUS {
            return Err(GnarkError::Malformed {
                part,
                reason: format!("element {i} is not below the scalar modulus"),
            });
        }
        Ok(Fr::new_unchecked(limbs))
    })
}
