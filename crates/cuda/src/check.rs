//! Host-side checks of what a transform wrote, for the ceremony seam.
//!
//! A download that returns says the stream drained; it says nothing about the values the
//! kernels stored before it. The Metal side has reported buffers complete with values it
//! did not store (BUG-28) and holds every ceremony readback to these checks since then.
//! The group FFT's outputs go straight into a file, so every readback of `fft.rs` runs
//! them inside the attempt that made it: a failure is a [`ProveError::Device`], and the
//! transform is re-run from the caller's unchanged slice, so a transient wrong store is
//! re-run and a persistent one fails the operation with nothing written.
//!
//! The checks themselves are shared with Metal: the range check is
//! `snarkrs_gpu_layout::Reduced`, the curve equation, the exact comparison and the spot
//! position are `snarkrs_msm::check`. One thread per pass is the spot recomputed on the
//! host. What stays here is `Reduced` for this crate's XYZZ wire structs and the error a
//! failure is refused with.

use snarkrs_groth16::ProveError;

pub(crate) use snarkrs_gpu_layout::Reduced;
pub(crate) use snarkrs_msm::check::{same_point, spot, xyzz_on_curve, CurveField};

use crate::msm::{PackedXyzzG1, PackedXyzzG2};

impl Reduced for PackedXyzzG1 {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced() && self.zz.reduced() && self.zzz.reduced()
    }
}

impl Reduced for PackedXyzzG2 {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced() && self.zz.reduced() && self.zzz.reduced()
    }
}

/// The error a readback that fails a check is refused with. A device fault rather than a
/// backend one, the variant `g16 prove --fallback` retries; the reason names the
/// operation and what was wrong, in the spelling the Metal checks use.
pub(crate) fn wrong(context: &str, what: impl std::fmt::Display) -> ProveError {
    ProveError::Device {
        backend: "cuda",
        reason: format!("{context}: {what}: a value read back is not one the GPU wrote"),
    }
}
