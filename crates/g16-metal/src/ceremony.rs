//! The Metal side of the three ceremony seams: [`MsmBackend`], [`GroupFft`] and
//! [`KeyScale`].
//!
//! Separate from [`crate::backend`], which implements `g16_core::Backend`, the whole
//! prover. A ceremony command wants one primitive at a time and holds no proving key, so
//! it selects one of these instead. All three take host slices: the ceremony's inputs are
//! mmap'd sections of a multi-gigabyte file that no `prepare` step ever made resident, and
//! on unified memory the upload is a `copy_from_slice` anyway.
//!
//! Nothing here may be shared into `g16-wgpu`. A point scalar multiplication inlines the
//! point operations three or more times over a 256-byte `Xyzz<Fq2>`, which is exactly the
//! shape WebKit 323560 (filed by this project) miscompiles on iOS.

use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use g16_msm::xyzz::Xyzz;
use g16_msm::{AccelError, GroupFft, KeyScale, MsmBackend};
use metal::Device;

use crate::msm::MetalMsm;

/// The name every error and every `name()` on this side reports, and the spelling
/// `--backend metal` uses.
const BACKEND: &str = "metal";

/// [`MetalMsm`] behind the ceremony's MSM trait.
///
/// A wrapper rather than an `impl` on `MetalMsm` itself, because `MetalMsm`'s own
/// `msm_g1` takes device handles and the trait's takes host slices. Two methods of the
/// same name and different argument types on one type resolve silently in favour of the
/// inherent one, and a reader cannot see which was called.
///
/// Every call uploads its own bases. That is right for `setup`, whose slots are gathered
/// per output point and never repeat a base vector, and it is the reason the slot loop
/// wants `MetalMsm::msm_batch` rather than this trait once a kernel is worth its dispatch:
/// 97,648 slots at 0.149 ms of commit-and-wait each is 14.5 s of round trip on a 45.6 s
/// command.
pub struct MetalMsmBackend {
    msm: MetalMsm,
}

impl MetalMsmBackend {
    /// Compiles the MSM library. Expensive (about 60 ms of runtime MSL compilation plus
    /// pipeline construction), so it belongs once at the top of a command.
    pub fn new() -> Result<Self, ProveError> {
        Ok(Self {
            msm: MetalMsm::new()?,
        })
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        Ok(Self {
            msm: MetalMsm::with_device(device)?,
        })
    }

    pub fn msm(&self) -> &MetalMsm {
        &self.msm
    }
}

/// The trait is infallible, so a device failure has nowhere to go but a panic.
///
/// That is the right end for it. Every failure `MetalMsm` reports here is a lost device or
/// a pipeline that would not build, never a numeric one, and the alternative to stopping is
/// returning a point: an identity, or whatever a half-run command buffer left behind. Both
/// are silently wrong bytes in a `.zkey` that people will prove against for years. The
/// panic message names the backend so it is not mistaken for an arithmetic bug.
impl MsmBackend for MetalMsmBackend {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        let bases = self.msm.upload_g1_bases(bases);
        let scalars = self.msm.upload_scalars(scalars);
        self.msm
            .msm_g1(&bases, &scalars)
            .unwrap_or_else(|e| panic!("metal msm_g1 failed, no zkey was written: {e}"))
    }

    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        let bases = self.msm.upload_g2_bases(bases);
        let scalars = self.msm.upload_scalars(scalars);
        self.msm
            .msm_g2(&bases, &scalars)
            .unwrap_or_else(|e| panic!("metal msm_g2 failed, no zkey was written: {e}"))
    }
}

/// The group inverse FFT behind `ptau prepare`, the slowest command in the project.
///
/// Not implemented yet. The device is opened here anyway, so `--backend metal` on a
/// machine with no Metal device fails for that reason and not this one.
pub struct MetalGroupFft {
    device: Device,
}

impl MetalGroupFft {
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default().ok_or_else(|| ProveError::Backend {
            backend: "metal",
            reason: "no Metal device; this machine cannot run the metal backend".into(),
        })?;
        Ok(Self::with_device(device))
    }

    pub fn with_device(device: Device) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl GroupFft for MetalGroupFft {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn ifft_g1(&self, _a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
        Err(AccelError::Unimplemented {
            backend: BACKEND,
            op: "group ifft over G1",
        })
    }

    fn ifft_g2(&self, _a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
        Err(AccelError::Unimplemented {
            backend: BACKEND,
            op: "group ifft over G2",
        })
    }
}

/// `batchApplyKey` behind the four contribute and beacon commands.
///
/// Not implemented yet, on the same terms as [`MetalGroupFft`].
pub struct MetalKeyScale {
    device: Device,
}

impl MetalKeyScale {
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default().ok_or_else(|| ProveError::Backend {
            backend: "metal",
            reason: "no Metal device; this machine cannot run the metal backend".into(),
        })?;
        Ok(Self::with_device(device))
    }

    pub fn with_device(device: Device) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl KeyScale for MetalKeyScale {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn apply_key_g1(
        &self,
        _points: &mut [G1Affine],
        _first: Fr,
        _inc: Fr,
    ) -> Result<(), AccelError> {
        Err(AccelError::Unimplemented {
            backend: BACKEND,
            op: "batch apply key over G1",
        })
    }

    fn apply_key_g2(
        &self,
        _points: &mut [G2Affine],
        _first: Fr,
        _inc: Fr,
    ) -> Result<(), AccelError> {
        Err(AccelError::Unimplemented {
            backend: BACKEND,
            op: "batch apply key over G2",
        })
    }
}
