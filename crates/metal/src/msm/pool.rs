//! The scratch buffer pool every batch takes its digit, bucket and partial-sum
//! buffers from, and returns them to scrubbed.

use std::sync::Mutex;

use metal::{Buffer, Device};
use snarkrs_groth16::ProveError;

/// Scratch buffers, kept between calls.
///
/// Two measured facts make this worth the code. A `memcpy` into a freshly allocated
/// shared buffer runs at 15.2 GB/s against 54.8 GB/s into one that has already been
/// touched, because the first write to every page faults it in; and the same first-touch
/// cost applies to a kernel's first write. Allocating the counting-sort scratch on every
/// proof would pay that every time, so buffers go back to the pool instead.
pub(super) struct Pool {
    pub(super) device: Device,
    pub(super) free: Mutex<Vec<Buffer>>,
}

impl Pool {
    /// The lock is taken back from a poisoned mutex rather than panicked on, the same as
    /// `stages::HResident`'s pool: the vector holds plain buffer handles, so a panic
    /// while it was held leaves nothing half-updated for the next caller to trip over.
    pub(super) fn take(&self, bytes: usize) -> Result<Buffer, ProveError> {
        let bytes = bytes.max(4);
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        // Smallest buffer that fits, so a single huge allocation cannot be handed out
        // for every small request and then be unavailable for the one that needs it.
        let mut best: Option<usize> = None;
        for (i, b) in free.iter().enumerate() {
            if (b.length() as usize) >= bytes {
                match best {
                    Some(j) if free[j].length() <= b.length() => {}
                    _ => best = Some(i),
                }
            }
        }
        if let Some(i) = best {
            return Ok(free.swap_remove(i));
        }
        drop(free);
        crate::alloc::shared(&self.device, bytes)
    }

    pub(super) fn give(&self, bufs: Vec<Buffer>) {
        // Every one of these held the witness's digits, buckets or partial sums. Zeroed
        // here, outside the lock, rather than on `take`, so what sits in the pool between
        // proofs is zeros. A zeroed page stays resident, so the first-touch cost the pool
        // exists to avoid is not paid again.
        for b in &bufs {
            crate::alloc::scrub(b);
        }
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        free.extend(bufs);
        // Unbounded reuse would keep every shape any circuit ever asked for. This is a
        // cache, not an arena.
        if free.len() > 48 {
            free.sort_by_key(|b| std::cmp::Reverse(b.length()));
            free.truncate(48);
        }
    }
}
