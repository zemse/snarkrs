//! The two ceremony primitives that are not a multiexp, beside [`crate::MsmBackend`].
//!
//! `setup` is the only ceremony command an MSM accelerates. The other two expensive ones
//! are shaped differently and each needs its own seam:
//!
//! * `ptau prepare` is an inverse FFT **over the group**, where every butterfly twiddle is
//!   a full 254-bit point scalar multiplication. Measured at power 16, `point_times_fr` is
//!   99.2% of the command and the transform's `log(n)` passes are strictly sequential, so
//!   the unit a device wants is one whole block, not one butterfly. [`GroupFft`] is cut at
//!   `ifft`, which owns every pass and none of the delicate work: the two-coset split, the
//!   batch inversion back to affine and the file layout stay on the host.
//! * `ptau contribute`, `ptau beacon`, `zkey contribute` and `zkey beacon` are all one
//!   loop, `P_i *= first * inc^i`. Measured on a 2^20 zkey it is 97% of the command and
//!   over 99% of that one loop. [`KeyScale`] is that loop, N points in and N points out.
//!
//! Both live here rather than in `g16-ceremony` so that a backend crate can implement them
//! without depending on the ceremony's file readers, and both take host slices rather than
//! device handles: on unified memory that is a `copy_from_slice`, and on a discrete card
//! the transfer is a rounding error against `n log n` point multiplications.
//!
//! Neither trait is fallible in the sense the CPU is. A failure here is a lost device or a
//! kernel that would not build, never a numeric one, so the error carries which backend and
//! which primitive rather than anything about the input.

use g16_field::raw::{RawFq, RawFq2};
use g16_field::{Fr, G1Affine, G2Affine};

use crate::xyzz::Xyzz;

/// What an accelerated ceremony primitive can fail with.
///
/// Deliberately not `CeremonyError`: that type is about file formats and lives a crate
/// away, and a backend has nothing to say about either.
#[derive(Debug, thiserror::Error)]
pub enum AccelError {
    /// The backend exists on this machine but has no kernel for this primitive yet. The
    /// command has to stop rather than fall back, for the reason `make_backend` gives:
    /// silently running the other backend is worse than no answer.
    #[error("backend `{backend}` has no {op} implementation yet")]
    Unimplemented {
        backend: &'static str,
        op: &'static str,
    },
    /// The device rejected the work, lost it, or handed back something unreadable.
    #[error("backend `{backend}` failed during {op}: {reason}")]
    Device {
        backend: &'static str,
        op: &'static str,
        reason: String,
    },
    /// The caller passed a length or a group this backend cannot take. A bug in the
    /// caller, not in the device.
    #[error("{op}: {reason}")]
    Shape { op: &'static str, reason: String },
}

impl AccelError {
    /// Shorthand for the `Device` arm, which is what every backend raises when a queue
    /// call comes back with an error string.
    pub fn device(backend: &'static str, op: &'static str, reason: impl Into<String>) -> Self {
        Self::Device {
            backend,
            op,
            reason: reason.into(),
        }
    }
}

/// The inverse FFT over a curve group, in place, over the XYZZ form the transform runs in.
///
/// `a.len()` is a power of two and the caller has already bit-reversed nothing: the whole
/// transform is the implementation's, including the `1/n` scaling pass and the reversal of
/// `a[1..]` that the inverse reassembly folds in. See `g16_ceremony::prepare` for what that
/// tail is and why reading half of it produces an answer that is one rotation from correct.
///
/// Points arriving as [`Xyzz::ZERO`] are a live path, not a defensive one: the `power+1`
/// block of ptau section 12 is padded with the point at infinity.
pub trait GroupFft: Send + Sync {
    fn name(&self) -> &'static str;

    /// The smallest block worth sending to this backend. Blocks shorter than this go to
    /// the CPU transform instead.
    ///
    /// Default 1, meaning "everything", which is what the CPU implementation wants. A
    /// device wants a real number here: nearly all of a section's work sits in its top two
    /// blocks, and the ones under any sane crossover are the same short handful at every
    /// power, so a crossover that is roughly right costs nothing and a missing one ships a
    /// path that loses to the CPU on most of the blocks in the section.
    fn min_block(&self) -> usize {
        1
    }

    fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError>;
    fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError>;

    /// [`Self::ifft_g1`] over several independent blocks in one call.
    ///
    /// The blocks are separate transforms and the implementation may run them in any
    /// order, or together. The default runs them one at a time, which is all the CPU
    /// wants; a device overrides it to co-schedule their passes, because a small block's
    /// passes cannot fill it alone and a ptau section is mostly blocks like that.
    fn ifft_g1_many(&self, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), AccelError> {
        for a in blocks.iter_mut() {
            self.ifft_g1(a)?;
        }
        Ok(())
    }

    fn ifft_g2_many(&self, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), AccelError> {
        for a in blocks.iter_mut() {
            self.ifft_g2(a)?;
        }
        Ok(())
    }
}

/// `batchApplyKey` (`build_curve_jacobian_a0.js:1289-1310`): replace `P_i` by
/// `P_i * (first * inc^i)` in place.
///
/// `inc == 1` is the constant-scalar case both zkey commands use, and it is not a separate
/// entry point because it is not a separate kernel: `Fr::one().pow(k)` is exactly one, so
/// the geometric walk degenerates on its own.
///
/// The multiplier is the scalar's mathematical value, not its Montgomery residue:
/// `g1m_timesFr` de-Montgomeries before multiplying (`build_bn128.js:54-72`).
///
/// Output is affine, which is the whole reason this is one call and not N: a per-point
/// `into_affine` is a modular inversion each, and at 2^20 points that inversion is the
/// command. Montgomery's trick needs exactly one inversion per batch, so a backend with no
/// device-side inverse can legitimately take the single accumulated product back to the
/// host for it.
pub trait KeyScale: Send + Sync {
    fn name(&self) -> &'static str;
    fn apply_key_g1(&self, points: &mut [G1Affine], first: Fr, inc: Fr) -> Result<(), AccelError>;
    fn apply_key_g2(&self, points: &mut [G2Affine], first: Fr, inc: Fr) -> Result<(), AccelError>;
}
