//! NVIDIA CUDA backend.
//!
//! Same shape as `g16-metal`, same stage boundaries, same packed wire format (both read
//! [`g16_gpu_layout`], so the two GPUs are fed bit-identical bytes and a comparison
//! between them means something).
//!
//! # How this differs from the Metal backend, and why
//!
//! * Kernels are compiled by NVRTC at run time instead of `newLibraryWithSource`, for the
//!   same reason: one binary, no offline toolchain, correct code for whatever card is
//!   present. See [`context::Cuda::compile`].
//! * There is no unified memory. On Apple silicon a packed host buffer *is* the device
//!   buffer and the "upload" is a `memcpy` into `MTLBuffer::contents()`. On a discrete
//!   NVIDIA card every input crosses PCIe, so the transfer is real work that shows up in
//!   the measurements, and the stage boundary that keeps `H` resident on the device
//!   (`g16_core::HPoly::Device`) matters more here than it does on Metal.
//! * Bucket accumulation cannot lean on Metal's zero-filled fresh allocations.
//!   `cudaMalloc` does not zero, so every accumulator is explicitly zeroed. Relying on
//!   the allocator would be a silent wrong-answer bug rather than a crash.
//!
//! # Building without a GPU
//!
//! `--features cuda` compiles on any host. `cudarc` is used with `dynamic-loading`, which
//! resolves `libcuda` and `libnvrtc` with `dlopen` at run time rather than at link time,
//! so the whole backend can be written and typechecked on a machine that has no NVIDIA
//! hardware and no CUDA toolkit. Only running the tests needs a card.

#[cfg(feature = "cuda")]
mod context;
#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub use context::{Cuda, CudaError};

#[cfg(feature = "cuda")]
mod words {
    use g16_gpu_layout::Packed;

    /// Reinterpret a packed slice as the `u32` words the driver copies.
    ///
    /// `cudarc`'s `DeviceRepr` is a foreign trait and `PackedFr` is a foreign type, so it
    /// cannot be implemented for them here. Transferring as `u32` sidesteps the orphan
    /// rule without a newtype wrapper at every call site, and is exact rather than a
    /// workaround: every `Packed` type is by definition a whole number of `u32`s with no
    /// padding.
    pub fn as_words<T: Packed>(items: &[T]) -> &[u32] {
        // SAFETY: `Packed` promises `repr(C)`, no padding, nothing but `u32` inside, and
        // every bit pattern valid. `size_of::<T>()` is therefore a multiple of 4 and the
        // alignment of 4 is satisfied, so the reinterpretation is in bounds and aligned.
        unsafe {
            core::slice::from_raw_parts(
                items.as_ptr().cast::<u32>(),
                items.len() * (core::mem::size_of::<T>() / 4),
            )
        }
    }

    /// Inverse of [`as_words`], for values coming back off the device.
    ///
    /// Returns `None` on a length that is not a whole number of `T`, which would mean a
    /// kernel wrote a different number of elements than the host expected. Silently
    /// truncating there would turn a launch-geometry bug into a wrong proof.
    pub fn from_words<T: Packed + Default>(w: &[u32]) -> Option<Vec<T>> {
        let per = core::mem::size_of::<T>() / 4;
        if w.len() % per != 0 {
            return None;
        }
        let mut out = vec![T::default(); w.len() / per];
        // SAFETY: same contract as `as_words`, in the other direction. The destination is
        // exactly `w.len()` words long by construction.
        unsafe {
            core::ptr::copy_nonoverlapping(w.as_ptr(), out.as_mut_ptr().cast::<u32>(), w.len());
        }
        Some(out)
    }
}

#[cfg(feature = "cuda")]
pub use words::{as_words, from_words};

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use g16_gpu_layout::{FR_MODULUS, FR_N0};

    /// The drift guard, the CUDA twin of the Metal backend's
    /// `msl_declares_the_same_constants`. The kernel carries these constants as literal
    /// text; if either side is edited alone this fails instead of shipping proofs that are
    /// wrong by a factor of R.
    #[test]
    fn cuda_declares_the_same_constants() {
        let src = crate::kernels::FR_CUH;
        let want_n = format!(
            "__constant__ u32 FR_N[8] = {{ {} }};",
            FR_MODULUS
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            src.contains(&want_n),
            "kernels/bn254_fr.cuh does not contain the line:\n{want_n}"
        );
        let want_n0 = format!("__constant__ u32 FR_N0 = 0x{FR_N0:08x}u;");
        assert!(
            src.contains(&want_n0),
            "kernels/bn254_fr.cuh does not contain the line:\n{want_n0}"
        );
        assert!(
            src.contains("struct Fr {\n    u32 v[8];\n};"),
            "the CUDA Fr struct no longer mirrors PackedFr"
        );
    }

    /// A `--use_fast_math` here would be meaningless at best. Assert it never appears.
    #[test]
    fn no_fast_math() {
        assert!(!crate::kernels::FR_CUH.contains("fast_math"));
    }
}
