//! The [`KeySink`] behind [`MetalBackend::load`](crate::MetalBackend::load): the zkey's
//! G1 query sections and CSR matrices decoded straight into shared buffers.
//!
//! [`MetalBackend::prepare`](snarkrs_groth16::Backend::prepare) takes a parsed key, so
//! each of those sections is written twice: once into an arkworks vector by the parse,
//! then again into its shared buffer, with the vector dropped right after. On keccak_2048
//! that is 6.6 million 72-byte affines and 11.9 million coefficients written to fresh
//! pages for nothing. Through this sink the parse's one write is the upload.

use snarkrs_field::{Fr, G1Affine};
use snarkrs_formats::{CsrInto, KeySink, PointsInto, ZkeyError};
use snarkrs_groth16::ProveError;

use crate::layout::{PackedFr, PackedG1Affine};
use crate::msm::G1Bases;
use crate::mtl::DeviceRef;
use crate::stages::Csr;

/// The sections sent here, in the order the loader asks for them.
const SECTIONS: [u32; 4] = [5, 6, 8, 9];

pub(crate) struct DeviceSink<'d> {
    device: &'d DeviceRef,
    /// Sections 5, 6, 8 and 9, the A, B-in-G1, L and H bases.
    pub(crate) g1: [Option<G1Bases>; 4],
    pub(crate) csr: Option<[Csr; 2]>,
    /// The allocation that failed, if one did. The loader only carries a [`ZkeyError`], so
    /// the device's own error waits here for the caller.
    pub(crate) failed: Option<ProveError>,
}

impl<'d> DeviceSink<'d> {
    pub(crate) fn new(device: &'d DeviceRef) -> Self {
        Self {
            device,
            g1: [None, None, None, None],
            csr: None,
            failed: None,
        }
    }

    fn fail(&mut self, e: ProveError) -> ZkeyError {
        let text = e.to_string();
        self.failed = Some(e);
        ZkeyError::Io(std::io::Error::new(std::io::ErrorKind::OutOfMemory, text))
    }
}

impl KeySink for DeviceSink<'_> {
    type G1 = PackedG1Affine;
    type Fr = PackedFr;

    fn g1(p: &G1Affine) -> PackedG1Affine {
        PackedG1Affine::from_affine(p)
    }

    fn fr(v: &Fr) -> PackedFr {
        PackedFr::from_fr(v)
    }

    fn g1_section(
        &mut self,
        section: u32,
        n: usize,
    ) -> Result<Option<PointsInto<'_, PackedG1Affine>>, ZkeyError> {
        let Some(at) = SECTIONS.iter().position(|&s| s == section) else {
            return Ok(None);
        };
        let bases = match G1Bases::zeroed(self.device, n) {
            Ok(b) => b,
            Err(e) => return Err(self.fail(e)),
        };
        let bases = self.g1[at].insert(bases);
        // SAFETY: just allocated, so no command buffer holds it.
        let (points, infinity) = unsafe { bases.host_mut() };
        Ok(Some(PointsInto { points, infinity }))
    }

    fn coefficients(
        &mut self,
        rows: usize,
        nnz: [usize; 2],
    ) -> Result<Option<[CsrInto<'_, PackedFr>; 2]>, ZkeyError> {
        let made = Csr::zeroed(self.device, rows, nnz[0])
            .and_then(|a| Ok([a, Csr::zeroed(self.device, rows, nnz[1])?]));
        let csr = match made {
            Ok(c) => c,
            Err(e) => return Err(self.fail(e)),
        };
        let [a, b] = self.csr.insert(csr);
        Ok(Some([csr_into(a), csr_into(b)]))
    }
}

fn csr_into(c: &mut Csr) -> CsrInto<'_, PackedFr> {
    // SAFETY: only reached from `coefficients`, on a matrix it just allocated, so no
    // command buffer holds any of its buffers.
    let (row_ptr, signal, value) = unsafe { c.host_mut() };
    CsrInto {
        row_ptr,
        signal,
        value,
    }
}
