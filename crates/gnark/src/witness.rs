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
use rayon::prelude::*;
use snarkrs_field::*;
use snarkrs_formats::binfile::{bigint, decode_records};
use std::borrow::Cow;

/// A solved witness: the wires and the two constraint-row vectors the prover needs.
///
/// Owned when it came from [`GnarkWitness::read_dump`], and then scrubbed on drop like every
/// other witness image in snarkrs. Borrowed when it came from [`GnarkWitness::from_parts`],
/// which copies nothing: the buffers stay the caller's, to scrub when it is done with them.
pub struct GnarkWitness<'a> {
    w: Cow<'a, [Fr]>,
    aw: Cow<'a, [Fr]>,
    bw: Cow<'a, [Fr]>,
}

impl GnarkWitness<'static> {
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
        let witness = Self {
            w: Cow::Owned(w?),
            aw: Cow::Owned(aw?),
            bw: Cow::Owned(bw?),
        };
        witness.check_shape()?;
        Ok(witness)
    }
}

impl<'a> GnarkWitness<'a> {
    /// A witness in the caller's memory, borrowed and not copied. `aw` and `bw` must have
    /// one entry per constraint, and `w[0]` must be the constant one.
    ///
    /// A buffer gnark's solver wrote is already in this form: `fr.Element` and arkworks'
    /// `Fr` are both four Montgomery limbs with `R = 2^256`, see [`fr_slice`]. Every element
    /// is checked below the modulus, because that is the one thing a reinterpreted buffer
    /// can get wrong and arkworks' arithmetic assumes it.
    pub fn from_parts(w: &'a [Fr], aw: &'a [Fr], bw: &'a [Fr]) -> Result<Self, GnarkError> {
        for (part, v) in [("W", w), ("Aw", aw), ("Bw", bw)] {
            if let Some(i) = v.par_iter().position_first(|x| x.0 >= Fr::MODULUS) {
                return Err(GnarkError::Malformed {
                    part,
                    reason: format!("element {i} is not below the scalar modulus"),
                });
            }
        }
        let witness = Self {
            w: Cow::Borrowed(w),
            aw: Cow::Borrowed(aw),
            bw: Cow::Borrowed(bw),
        };
        witness.check_shape()?;
        Ok(witness)
    }

    fn check_shape(&self) -> Result<(), GnarkError> {
        if self.aw.len() != self.bw.len() {
            return Err(GnarkError::Malformed {
                part: "witness",
                reason: format!("Aw has {} rows and Bw has {}", self.aw.len(), self.bw.len()),
            });
        }
        // Every QAP row and the public-input half of the verifier equation assume it, as in
        // the `.wtns` loader.
        if self.w.first() != Some(&Fr::ONE) {
            return Err(GnarkError::Malformed {
                part: "W",
                reason: "W[0] is not 1".into(),
            });
        }
        Ok(())
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

/// Only what this witness owns is scrubbed. A borrowed buffer is the caller's.
impl Drop for GnarkWitness<'_> {
    fn drop(&mut self) {
        for v in [&mut self.w, &mut self.aw, &mut self.bw] {
            if let Cow::Owned(v) = v {
                snarkrs_groth16::scrub(v);
            }
        }
    }
}

/// gnark's `[]fr.Element` memory, as `Fr`, without a copy. `None` unless every element is
/// below the modulus.
///
/// Sound because the two layouts are one: `fr.Element` is `[4]uint64` and arkworks' `Fr` is
/// a `BigInt<4>`, a `[u64; 4]`, beside a zero-sized marker. The assertions below pin size
/// and alignment, and a 32-byte struct whose one non-zero-sized field is 32 bytes has that
/// field at offset 0. The values agree too, since both hold `x * 2^256 mod r`.
pub fn fr_slice(limbs: &[[u64; 4]]) -> Option<&[Fr]> {
    const _: () = assert!(std::mem::size_of::<Fr>() == std::mem::size_of::<[u64; 4]>());
    const _: () = assert!(std::mem::align_of::<Fr>() == std::mem::align_of::<[u64; 4]>());
    if limbs
        .par_iter()
        .any(|l| ark_ff::BigInt::new(*l) >= Fr::MODULUS)
    {
        return None;
    }
    // SAFETY: same size and alignment as asserted above, every limb set is canonical, and
    // the lifetime and length carry over from `limbs`.
    Some(unsafe { std::slice::from_raw_parts(limbs.as_ptr().cast::<Fr>(), limbs.len()) })
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
