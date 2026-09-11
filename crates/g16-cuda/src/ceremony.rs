//! The CUDA side of the ceremony seams: [`GroupFft`] behind `ptau prepare`.
//!
//! The Metal twin (`g16-metal/src/ceremony.rs`) implements all three seams; this file
//! implements the one whose kernels exist. `setup`'s [`g16_msm::MsmBackend`] and the
//! apply-key behind the four contribute and beacon commands still have no CUDA kernels,
//! and the CLI keeps saying so per command rather than falling back.
//!
//! Separate from [`crate::backend`], which implements `g16_core::Backend`, the whole
//! prover. A ceremony command wants one primitive at a time and holds no proving key, so
//! it opens its own context here instead of borrowing the prover's.

use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_msm::xyzz::Xyzz;
use g16_msm::{AccelError, GroupFft};

use crate::fft::FftKernels;
use crate::msm::bad;
use crate::{Cuda, CudaError};

/// The name every error on this side reports, and the spelling `--backend cuda` uses.
const BACKEND: &str = "cuda";

/// The group inverse FFT behind `ptau prepare`, the slowest command in the project.
///
/// A wrapper over [`FftKernels`], which owns the kernels, the twiddle tables and the
/// submission. What the trait adds on top is the error mapping and the crossover:
/// `min_block` is the backend's own number, so `prepare::ifft_block` (prepare.rs:352)
/// carries no backend-shaped branch and the threshold can move without touching the
/// ceremony crate.
pub struct CudaGroupFft {
    fft: FftKernels,
}

impl CudaGroupFft {
    /// Opens device 0, or the ordinal in `G16_CUDA_DEVICE`, and compiles the FFT unit.
    /// Same selection rule as `CudaBackend::new`, and the same refusal to fall back:
    /// a machine with no usable device is an error, not a CPU run.
    pub fn new() -> Result<Self, ProveError> {
        let ordinal = match std::env::var("G16_CUDA_DEVICE") {
            Ok(v) => v.trim().parse::<usize>().map_err(|e| {
                bad(format!(
                    "G16_CUDA_DEVICE={v:?} is not a device ordinal: {e}"
                ))
            })?,
            Err(_) => 0,
        };
        Self::with_ordinal(ordinal)
    }

    pub fn with_ordinal(ordinal: usize) -> Result<Self, ProveError> {
        let cuda = Cuda::new(ordinal)
            .map_err(|e: CudaError| bad(format!("no usable CUDA device {ordinal}: {e}")))?;
        Self::with_cuda(&cuda)
    }

    /// The kernels on an already-open context, for a caller that holds one.
    pub fn with_cuda(cuda: &Cuda) -> Result<Self, ProveError> {
        Ok(Self {
            fft: FftKernels::new(cuda)?,
        })
    }

    /// The kernel layer, for a measurement that wants to drive the transform without
    /// going through `ptau prepare`.
    pub fn kernels(&self) -> &FftKernels {
        &self.fft
    }

    /// [`FftKernels::with_min_block`], so a test can put a file the shipped crossover
    /// would route home onto the device instead.
    pub fn with_min_block(mut self, n: usize) -> Self {
        self.fft = self.fft.with_min_block(n);
        self
    }
}

impl GroupFft for CudaGroupFft {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn min_block(&self) -> usize {
        self.fft.min_block()
    }

    fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
        self.fft
            .ifft_g1(a)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G1", e.to_string()))
    }

    fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
        self.fft
            .ifft_g2(a)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G2", e.to_string()))
    }

    fn ifft_g1_many(&self, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), AccelError> {
        self.fft
            .ifft_g1_many(blocks)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G1", e.to_string()))
    }

    fn ifft_g2_many(&self, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), AccelError> {
        self.fft
            .ifft_g2_many(blocks)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G2", e.to_string()))
    }
}

/// `prepare` hands the transform to worker threads; checked rather than assumed, the
/// same way `msm.rs` checks `CudaMsm`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaGroupFft>();
};
