//! Allocating a shared buffer, and finding out whether it worked.
//!
//! `newBufferWithLength:` returns nil when the allocation fails. Every allocation the
//! prover makes goes through this module, so a failure is an error here rather than a
//! nil handle further on: `contents()` on one answers null, and the next
//! `from_raw_parts_mut` over it writes through address 0, or an encoder binds nil and the
//! command buffer faults, which [`crate::cb::wait_ok`] can only report as an opaque "did
//! not complete".
//!
//! The sizes make that a real condition rather than a theoretical one. An H plan at a
//! 2^20 domain picks c = 15, so its entry array alone is 17 * 2^20 * 8 bytes = 143 MB,
//! with another 36 MB of spill points and 36 MB of buckets beside it: a fifth of a
//! gigabyte of scratch for one job, on a machine that may have 8 GB for everything.
//!
//! A zero-byte request is not a valid `MTLBuffer` either, so a buffer whose `length` is 0
//! is refused the same way. A request above the device's `maxBufferLength` is refused
//! before it reaches the driver, so the error names the limit rather than only the size.
//! A size is the product of its `factors` through [`bytes`], so a product that wraps is
//! refused the same way rather than slipping under that check as a small request.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use snarkrs_groth16::ProveError;

use crate::mtl::{Buffer, DeviceRef};

/// The product of `factors`, the byte count of a buffer of that shape. One that does not
/// fit in a `usize` is refused with the error an oversized request gets.
pub(crate) fn bytes(factors: &[usize]) -> Result<usize, ProveError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .ok_or_else(|| ProveError::Backend {
            backend: "metal",
            reason: format!("a shared buffer of {factors:?} bytes overflows a usize"),
        })
}

/// `bytes` of shared storage, zeroed by the driver.
pub(crate) fn shared(device: &DeviceRef, bytes: usize) -> Result<Buffer, ProveError> {
    fits(device, bytes)?;
    checked(
        crate::mtl::buffer(device, bytes, MTLResourceOptions::StorageModeShared),
        bytes,
    )
}

/// `bytes` of shared storage holding a copy of what `src` points at. Same contract as
/// `newBufferWithBytes:length:options:`: `src` must be readable for `bytes`.
pub(crate) fn shared_with_data(
    device: &DeviceRef,
    src: *const c_void,
    bytes: usize,
) -> Result<Buffer, ProveError> {
    fits(device, bytes)?;
    let src = NonNull::new(src.cast_mut()).expect("a buffer copied from a null pointer");
    // SAFETY: the caller's contract above.
    let buf = unsafe {
        crate::mtl::buffer_with_data(device, src, bytes, MTLResourceOptions::StorageModeShared)
    };
    checked(buf, bytes)
}

/// Zero a shared buffer from the host. For scratch that held witness-derived data and is
/// going back to a pool, where it would otherwise sit readable until a same-sized proof
/// overwrote it. The caller guarantees the GPU is done with it: every pooled buffer comes
/// back only after its command buffer completed.
pub(crate) fn scrub(buf: &Buffer) {
    let ptr = buf.contents().cast::<u8>().as_ptr();
    let len = buf.length();
    // Split over the pool: one thread's memset left 8 ms on an anon-aadhaar proof's
    // critical path, which is hundreds of MB of scratch. The address travels as a `usize`
    // because a raw pointer is not `Send`.
    const CHUNK: usize = 1 << 20;
    let at = ptr as usize;
    use rayon::prelude::*;
    (0..len.div_ceil(CHUNK)).into_par_iter().for_each(|i| {
        let lo = i * CHUNK;
        let n = CHUNK.min(len - lo);
        // SAFETY: `contents` of a shared buffer is host-visible and `len` bytes long, no
        // command buffer holds it (see above), and the chunks are disjoint.
        unsafe { core::ptr::write_bytes((at + lo) as *mut u8, 0, n) };
    });
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

fn fits(device: &DeviceRef, bytes: usize) -> Result<(), ProveError> {
    let max = device.maxBufferLength();
    if bytes > max {
        return Err(ProveError::Backend {
            backend: "metal",
            reason: format!("a {bytes}-byte shared buffer is over the device's {max}-byte limit"),
        });
    }
    Ok(())
}

fn checked(buf: Option<Buffer>, bytes: usize) -> Result<Buffer, ProveError> {
    match buf {
        Some(buf) if buf.length() != 0 => Ok(buf),
        _ => Err(ProveError::Backend {
            backend: "metal",
            reason: format!("could not allocate a {bytes}-byte shared buffer"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_over_max_buffer_length_is_refused() {
        let device = crate::mtl::system_default().expect("no Metal device");
        let max = device.maxBufferLength();
        for bytes in [max + 1, usize::MAX] {
            let err = shared(&device, bytes).err().expect("refused");
            assert!(
                matches!(
                    err,
                    ProveError::Backend {
                        backend: "metal",
                        ..
                    }
                ),
                "{err}"
            );
            // Never read: the size check refuses first.
            let src = [0u8; 4];
            let err = shared_with_data(&device, src.as_ptr().cast(), bytes)
                .err()
                .expect("refused");
            assert!(
                matches!(
                    err,
                    ProveError::Backend {
                        backend: "metal",
                        ..
                    }
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn a_wrapping_byte_count_is_refused() {
        assert_eq!(bytes(&[3, 32, 8]).unwrap(), 768);
        assert_eq!(bytes(&[]).unwrap(), 1);
        assert_eq!(bytes(&[usize::MAX, 0]).unwrap(), 0);
        // 2^60 x 32 wraps to 0, and 2^62 + 1 x 4 to 4: both would pass the size check
        // as small requests if multiplied with `*`.
        for factors in [[1usize << 60, 32], [(1 << 62) + 1, 4], [usize::MAX, 2]] {
            let err = bytes(&factors).expect_err("refused");
            assert!(
                matches!(
                    err,
                    ProveError::Backend {
                        backend: "metal",
                        ..
                    }
                ),
                "{err}"
            );
        }
        assert_eq!(bytes(&[1 << 30, 1 << 30, 1 << 3]).unwrap(), 1 << 63);
        assert!(bytes(&[1 << 30, 1 << 30, 1 << 3, 2]).is_err());
    }

    #[test]
    fn an_ordinary_request_is_allocated() {
        let device = crate::mtl::system_default().expect("no Metal device");
        assert_eq!(shared(&device, 4096).unwrap().length(), 4096);
        let src = [7u8; 64];
        let buf = shared_with_data(&device, src.as_ptr().cast(), src.len()).unwrap();
        assert_eq!(buf.length(), 64);
        // SAFETY: a fresh shared buffer of 64 bytes, which no command buffer holds.
        let got = unsafe { core::slice::from_raw_parts(buf.contents().cast::<u8>().as_ptr(), 64) };
        assert_eq!(got, src);
    }
}
