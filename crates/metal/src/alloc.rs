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
//! with another 36 MB of spill points and 36 MB of buckets beside it: a fifth of a
//! gigabyte of scratch for one job, on a machine that may have 8 GB for everything.
//!
//! So every allocation the prover makes goes through this module. A nil handle answers 0
//! to `length`, because objc_msgSend returns zero for an integer-returning selector sent
//! to nil, and a zero-byte request is not a valid `MTLBuffer` either, so one test
//! catches both.

use std::ffi::c_void;

use metal::{Buffer, Device, MTLResourceOptions};
use snarkrs_groth16::ProveError;

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

// libSystem, which every process links. Declared here rather than taken from `libc` so
// this crate's dependency set does not change for two calls.
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
        -> *mut c_void;
}
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const MAP_PRIVATE: i32 = 0x2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANON: i32 = 0x1000;
/// Apple silicon's page. Also a multiple of a 4 KB page, so the rounding stays valid on
/// a machine that has those.
const PAGE: usize = 16 << 10;

/// Frees `v` with its pages handed back to the OS now, not whenever malloc reuses them.
///
/// Dropping is not enough for a key section. macOS malloc keeps a freed large block in
/// its large cache, still resident and still counted in the footprint, and
/// `malloc_zone_pressure_relief` returns 0 without emptying it. On js_384x384_d32 that
/// was 2.4 GB of dropped zkey sections held for the rest of the process. Mapping fresh
/// anonymous pages over the block's whole pages releases the old ones at once and leaves
/// malloc owning an address range that costs nothing until it is touched again.
pub(crate) fn release<T: Copy>(mut v: Vec<T>) {
    let start = v.as_mut_ptr() as usize;
    let lo = start.next_multiple_of(PAGE);
    let hi = (start + v.capacity() * core::mem::size_of::<T>()) & !(PAGE - 1);
    if hi > lo {
        // SAFETY: `lo..hi` lies inside `v`'s allocation, which nothing else references,
        // and `T: Copy` means the drop below reads none of it. The new pages are private,
        // readable and writable, as malloc's were, so it can reuse or unmap the block as
        // it would have. XNU puts the old mapping back if a fixed mapping fails, so a
        // failure costs only the saving; nothing is checked because nothing changes.
        unsafe {
            mmap(
                lo as *mut c_void,
                hi - lo,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_FIXED | MAP_ANON,
                -1,
                0,
            );
        }
    }
    drop(v);
}
