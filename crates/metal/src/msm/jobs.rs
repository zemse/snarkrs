//! What a batch is asked for and what it hands back: one job per MSM, one result each.

use snarkrs_field::{G1Projective, G2Projective};
use snarkrs_groth16::ProveError;

use super::err;
use super::inputs::{G1Bases, G2Bases, ScalarBuf};

/// One MSM. `scalar_off` and `base_off` exist so the L MSM, whose scalars are the
/// private suffix of the witness, can share the witness buffer with the A and B MSMs
/// instead of uploading a second copy.
pub struct JobG1<'a> {
    pub bases: &'a G1Bases,
    pub base_off: usize,
    pub scalars: &'a ScalarBuf,
    pub scalar_off: usize,
    pub n: usize,
}

pub struct JobG2<'a> {
    pub bases: &'a G2Bases,
    pub base_off: usize,
    pub scalars: &'a ScalarBuf,
    pub scalar_off: usize,
    pub n: usize,
}

pub enum Job<'a> {
    G1(JobG1<'a>),
    G2(JobG2<'a>),
}

pub(super) fn validate_job_ranges(
    soff: usize,
    n: usize,
    scalars_len: usize,
    boff: usize,
    bases_len: usize,
) -> Result<(), ProveError> {
    let scalar_end = soff.checked_add(n).ok_or_else(|| {
        err(format!(
            "scalar range offset {soff} + length {n} overflows usize"
        ))
    })?;
    if scalar_end > scalars_len {
        return Err(err(format!(
            "scalar range {soff}..{scalar_end} exceeds the {scalars_len} scalars uploaded"
        )));
    }
    let base_end = boff.checked_add(n).ok_or_else(|| {
        err(format!(
            "base range offset {boff} + length {n} overflows usize"
        ))
    })?;
    if base_end > bases_len {
        return Err(err(format!(
            "base range {boff}..{base_end} exceeds the {bases_len} bases uploaded"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum MsmResult {
    G1(G1Projective),
    G2(G2Projective),
}

impl MsmResult {
    pub fn g1(self) -> Result<G1Projective, ProveError> {
        match self {
            MsmResult::G1(p) => Ok(p),
            MsmResult::G2(_) => Err(err("expected a G1 MSM result, got G2")),
        }
    }
    pub fn g2(self) -> Result<G2Projective, ProveError> {
        match self {
            MsmResult::G2(p) => Ok(p),
            MsmResult::G1(_) => Err(err("expected a G2 MSM result, got G1")),
        }
    }

    /// Whether the point satisfies the curve equation. The partial sums come back as
    /// XYZZ words that `to_projective` takes on trust, and the addition formulas are
    /// rational maps, so one partial sum that is not a curve point (a buffer the GPU
    /// reported written but did not write, BUG-28) leaves the whole sum off the curve
    /// with all but negligible probability. One inversion per job.
    pub(super) fn on_curve(&self) -> bool {
        use snarkrs_field::CurveGroup;
        match self {
            MsmResult::G1(p) => p.into_affine().is_on_curve(),
            MsmResult::G2(p) => p.into_affine().is_on_curve(),
        }
    }
}
