//! `powersoftau prepare phase2`: sections 12 to 15 from sections 2 to 5.
//!
//! This is the expensive command in the whole ceremony, and the reason is one line of
//! ffjavascript. `G.lagrangeEvaluations` (`powersoftau_preparephase2.js:87`) routes to
//! `_fft` with `fnFFTMix = "g1m_fftMix"`, and `build_bn128.js:76` wires that FFT with
//! `opGtimesF = "g1m_timesFr"`. So this is an **inverse FFT over the group**, not over the
//! scalar field: every butterfly twiddle is a full 254-bit point scalar multiplication.
//! [`g16_ntt`] operates on `&mut [Fr]` and cannot express it, which is why the transform
//! lives here rather than there.
//!
//! The output layout is a concatenation of per-power blocks in ascending `p`, each holding
//! `2^p` points, so block `2^k` starts at element `2^k - 1`
//! (`powersoftau_preparephase2.js:56-62`). Section 12 gets one extra block at `power+1`
//! and sections 13 to 15 stop at `power`, which is exactly why `writeHs` can ask section
//! 12 for `2*domainSize` points when `cirPower == power` and the other three are only ever
//! read at the `domainSize` block.
//!
//! Two details that are easy to lose:
//!
//! * **The `power+1` block of section 12 is padded with the point at infinity.** Section 2
//!   holds `2n - 1` points but the block wants `2n`, so snarkjs reads `(nPoints-1)` points
//!   and writes `G1.zeroAffine` into the last slot
//!   (`powersoftau_preparephase2.js:78-81`).
//! * **At `bits == Fr::TWO_ADICITY + 1` the transform splits into two cosets**
//!   (`engine_fft.js:483-497`) and the block comes out as two contiguous halves rather
//!   than interleaved. `writeHs` knows this and reads the second half directly instead of
//!   striding (`zkey_new.js:192-194`), so the two must agree.

use std::path::Path;

use g16_field::{G1Affine, G2Affine};

use crate::CeremonyError;

/// Run the whole command: read sections 2 to 5, write 12 to 15, and copy everything else
/// through. The output declares 11 sections.
pub fn prepare_phase2(ptau_in: &Path, ptau_out: &Path) -> Result<(), CeremonyError> {
    let _ = (ptau_in, ptau_out);
    todo!("prepare_phase2")
}

/// The Lagrange evaluations of one block of G1 points: an inverse FFT over the group,
/// including the two-coset split at `bits == Fr::TWO_ADICITY + 1`.
///
/// `points.len()` must be a power of two. Above `2^(TWO_ADICITY + 1)` there is no root of
/// unity to build from, and that is [`CeremonyError::CircuitTooBig`].
pub fn lagrange_evaluations_g1(points: &[G1Affine]) -> Result<Vec<G1Affine>, CeremonyError> {
    let _ = points;
    todo!("lagrange_evaluations_g1")
}

/// [`lagrange_evaluations_g1`] over G2, for section 13.
pub fn lagrange_evaluations_g2(points: &[G2Affine]) -> Result<Vec<G2Affine>, CeremonyError> {
    let _ = points;
    todo!("lagrange_evaluations_g2")
}

/// In-place radix-2 inverse FFT over G1, twiddles applied as scalar multiplications.
///
/// Projective in and out because the butterflies are additions and the caller batches back
/// to affine once, not `2^p` times.
pub fn group_ifft_g1(a: &mut [g16_field::G1Projective]) -> Result<(), CeremonyError> {
    let _ = a;
    todo!("group_ifft_g1")
}

/// [`group_ifft_g1`] over G2.
pub fn group_ifft_g2(a: &mut [g16_field::G2Projective]) -> Result<(), CeremonyError> {
    let _ = a;
    todo!("group_ifft_g2")
}

/// Element count of a prepared section, given `power`: `2^(power+2) - 1` for section 12,
/// which carries the extra `power+1` block, and `2^(power+1) - 1` for 13, 14 and 15.
pub fn prepared_element_count(section: u32, power: u32) -> Option<usize> {
    let _ = (section, power);
    todo!("prepared_element_count")
}
