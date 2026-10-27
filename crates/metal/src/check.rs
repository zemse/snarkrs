//! Host-side checks of what a submission wrote, for the ceremony seams.
//!
//! A completion token says the last dispatch of a command buffer ran; it says nothing
//! about the stores before it. Metal has reported buffers `Completed`, token in place,
//! with values it did not store (BUG-28), and the MSM already refuses a window sum off
//! the curve for that reason (`msm.rs`, `combine`). The ceremony kernels' outputs go
//! straight into a file, so every readback of theirs is checked here the same way, inside
//! the `cb::with_retry` closure that made it: a failure is a [`ProveError::Device`] with
//! no `kIOGPU` name in it, which `with_retry` retries blind from the unchanged inputs, so
//! a transient fault is re-run and a persistent one fails the operation with nothing
//! written.
//!
//! The checks themselves are shared with CUDA: the range check is
//! `snarkrs_gpu_layout::Reduced`, the curve equation, the exact comparison and the spot
//! position are `snarkrs_msm::check`. What stays here is `Reduced` for this crate's XYZZ
//! wire structs and the error a failure is refused with.

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
/// backend one, so [`crate::cb::with_retry`] re-runs the submission; the reason names the
/// submission and what was wrong, in the spelling the MSM's check uses.
pub(crate) fn wrong(context: &str, what: impl std::fmt::Display) -> ProveError {
    ProveError::Device {
        backend: "metal",
        reason: format!("{context}: {what}: a value read back is not one the GPU wrote"),
    }
}
