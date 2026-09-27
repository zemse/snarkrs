//! Allocating a shared buffer, and finding out whether it worked.
//!
//! `newBufferWithLength:` returns nil when the allocation fails, and `metal-rs` 0.29
//! forwards the `msg_send!` result straight into a `Buffer` handle without looking at
//! it. Nothing on the Rust side is fallible, so the failure surfaces much later and
//! wearing someone else's clothes: `contents()` on the nil handle answers null, and the
//! next `from_raw_parts_mut` over it writes through address 0, or an encoder binds nil
//! and the command buffer faults, which [`crate::cb::wait_ok`] can only report as an
//! opaque "did not complete".
//!
//! The sizes make that a real condition rather than a theoretical one. An H plan at a
//! 2^20 domain picks c = 15, so its entry array alone is 17 * 2^20 * 8 bytes = 143 MB,
//! with another 71 MB of spill points and 36 MB of buckets beside it: a quarter of a
//! gigabyte of scratch for one job, on a machine that may have 8 GB for everything.
//!
//! So every allocation the prover makes goes through this module. A nil handle answers 0
//! to `length`, because objc_msgSend returns zero for an integer-returning selector sent
//! to nil, and a zero-byte request is not a valid `MTLBuffer` either, so one test
//! catches both.

use std::ffi::c_void;

use g16_core::ProveError;
use metal::{Buffer, Device, MTLResourceOptions};

/// `bytes` of shared storage, zeroed by the driver.
pub(crate) fn shared(device: &Device, bytes: usize) -> Result<Buffer, ProveError> {
    checked(
        device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
        bytes,
    )
}

/// `bytes` of shared storage holding a copy of what `src` points at. Same contract as
/// `Device::new_buffer_with_data`: `src` must be readable for `bytes`.
pub(crate) fn shared_with_data(
    device: &Device,
    src: *const c_void,
    bytes: usize,
) -> Result<Buffer, ProveError> {
    checked(
        device.new_buffer_with_data(src, bytes as u64, MTLResourceOptions::StorageModeShared),
        bytes,
    )
}

/// Zero a shared buffer from the host. For scratch that held witness-derived data and is
/// going back to a pool, where it would otherwise sit readable until a same-sized proof
/// overwrote it. The caller guarantees the GPU is done with it: every pooled buffer comes
/// back only after its command buffer completed.
pub(crate) fn scrub(buf: &Buffer) {
    let ptr = buf.contents().cast::<u8>();
    if ptr.is_null() {
        return;
    }
    let len = buf.length() as usize;
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

fn checked(buf: Buffer, bytes: usize) -> Result<Buffer, ProveError> {
    if buf.length() == 0 {
        return Err(ProveError::Backend {
            backend: "metal",
            reason: format!("could not allocate a {bytes}-byte shared buffer"),
        });
    }
    Ok(buf)
}
