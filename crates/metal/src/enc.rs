//! The encoder and buffer helpers the MSM, the ceremony and the group FFT all bind
//! through.

use objc2_metal::{MTLBuffer, MTLComputePipelineState};
use snarkrs_groth16::ProveError;

use crate::layout::{as_bytes, Packed};
use crate::mtl::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, DeviceRef};

/// `setBytes` for a params struct. Every caller's struct is `repr(C)` and all-`u32`, so
/// its bytes are exactly what the shader's `constant` reference reads.
pub(crate) fn set_params<T: Copy>(enc: &ComputeCommandEncoderRef, index: usize, p: &T) {
    // SAFETY: `p` is a live `T`, readable for `size_of::<T>()` bytes.
    unsafe {
        crate::mtl::set_bytes(
            enc,
            index,
            core::mem::size_of::<T>(),
            (p as *const T).cast(),
        )
    };
}

/// Threadgroup size is a preference clamped by what the pipeline will actually accept.
///
/// Never hardcode 1024: a kernel carrying one CIOS Montgomery multiply already reports
/// 896 on this device from register pressure alone, and the XYZZ mixed-addition kernel
/// measured 512. Both reference implementations query the pipeline; so does this. The
/// kernel bounds-checks its own index because `dispatch_threads` rounds the grid up.
pub(crate) fn dispatch_1d(
    enc: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    n: usize,
    prefer: usize,
) {
    if n == 0 {
        return;
    }
    let tg = prefer.min(pso.maxTotalThreadsPerThreadgroup()).max(1);
    crate::cb::dispatch_threads(enc, crate::mtl::size(n), crate::mtl::size(tg));
}

/// # Safety
///
/// The buffer must hold at least `len` `T`s written by a completed command buffer, and
/// `T` must be a `Packed` type, so every bit pattern is valid.
pub(crate) unsafe fn read_back<T: Packed>(buf: &Buffer, len: usize) -> &[T] {
    core::slice::from_raw_parts(buf.contents().cast::<T>().as_ptr(), len)
}

/// A shared buffer holding a copy of `items`. A zero-length `MTLBuffer` is not a valid
/// allocation, so an empty slice gets 4 bytes that no kernel reads.
pub(crate) fn buffer<T: Packed>(device: &DeviceRef, items: &[T]) -> Result<Buffer, ProveError> {
    let bytes = as_bytes(items);
    if bytes.is_empty() {
        return crate::alloc::shared(device, 4);
    }
    crate::alloc::shared_with_data(device, bytes.as_ptr().cast(), bytes.len())
}

/// `items` packed by `pack` straight into a shared buffer. Packing into a `Vec` and
/// copying that in held a third copy of the section at the peak of `prepare`. Empty
/// becomes 4 bytes, as in [`buffer`].
pub(crate) fn packed_buffer<S, T: Packed>(
    device: &DeviceRef,
    items: &[S],
    pack: fn(&[S], &mut [T]),
) -> Result<Buffer, ProveError> {
    if items.is_empty() {
        return crate::alloc::shared(device, 4);
    }
    let buf = crate::alloc::shared(device, items.len() * core::mem::size_of::<T>())?;
    // SAFETY: the buffer was just allocated with room for `items.len()` packed values
    // and nothing has been encoded against it, so no dispatch can be reading it.
    let dst =
        unsafe { core::slice::from_raw_parts_mut(buf.contents().cast().as_ptr(), items.len()) };
    pack(items, dst);
    Ok(buf)
}
