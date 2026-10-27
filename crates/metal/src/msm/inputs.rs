//! What an MSM reads, resident on the device: the base vectors uploaded once per key,
//! and the scalars with the classification that sizes a plan for them.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use snarkrs_groth16::ProveError;

use crate::layout::PackedG1Affine;
use crate::mtl::{Buffer, DeviceRef};
use objc2_metal::MTLBuffer;

use super::window::{window_size_for, Work, RECODE_BITS};

/// A base vector, repacked and resident on the device. Built once per key in
/// `prepare`; the whole point of the `Backend` contract's stage grouping is that this
/// never happens per proof.
///
/// `inf` remembers which bases are the point at infinity. A zkey is full of them: the B
/// queries of the csp keys are 61% infinity, because a wire that appears in no B
/// constraint still owns a slot. An infinity base contributes nothing whatever its
/// scalar, so the ones gather (see [`Outputs::alloc`](super::Outputs::alloc)) drops those
/// indices on the host instead of paying a device load and a dead branch per point per proof.
pub struct G1Bases {
    pub(super) buf: Buffer,
    pub(super) len: usize,
    pub(super) inf: Vec<bool>,
}

pub struct G2Bases {
    pub(super) buf: Buffer,
    pub(super) len: usize,
    pub(super) inf: Vec<bool>,
}

impl G1Bases {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Room for `n` bases, all at infinity, for a caller that fills it in place through
    /// [`Self::host_mut`]. Empty still gets one base's worth, since a zero-length
    /// `MTLBuffer` is not a valid allocation.
    pub(crate) fn zeroed(device: &DeviceRef, n: usize) -> Result<Self, ProveError> {
        let bytes = crate::alloc::bytes(&[n.max(1), core::mem::size_of::<PackedG1Affine>()])?;
        Ok(Self {
            buf: crate::alloc::shared(device, bytes)?,
            len: n,
            inf: vec![true; n],
        })
    }

    /// The bases and their infinity flags, writable from the host.
    ///
    /// # Safety
    ///
    /// No command buffer may hold the buffer while the slice lives. A fresh one from
    /// [`Self::zeroed`] that has not been handed to an MSM satisfies that.
    pub(crate) unsafe fn host_mut(&mut self) -> (&mut [PackedG1Affine], &mut [bool]) {
        let points = core::slice::from_raw_parts_mut(
            self.buf.contents().cast::<PackedG1Affine>().as_ptr(),
            self.len,
        );
        (points, &mut self.inf)
    }
}

impl G2Bases {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Chunk width of the classification pass in
/// [`MetalMsm::upload_scalars`](super::MetalMsm::upload_scalars), and hence
/// the granularity of [`ScalarBuf::bits_in`].
pub(super) const CLASSIFY_CHUNK: usize = 4096;

/// Bit length of a packed standard-form scalar: one past the highest set bit, 0 for 0.
pub(super) fn scalar_bits(v: &[u32; 8]) -> u32 {
    match v.iter().rposition(|w| *w != 0) {
        Some(i) => i as u32 * 32 + 32 - v[i].leading_zeros(),
        None => 0,
    }
}

/// Scalars in **standard** form (see `layout`'s module docs for why not Montgomery),
/// resident on the device.
///
/// `general_prefix[i]` is how many of the first `i` scalars are neither 0 nor 1. It is
/// built during the pack, which already walks every scalar, so it is free, and it is
/// what lets [`MetalMsm::msm_batch`](super::MetalMsm::msm_batch) size the window and the
/// entry array for the work that actually reaches the buckets rather than for the input
/// length. A witness is typically over 99% zeros and ones, so the difference is two
/// orders of magnitude.
pub struct ScalarBuf {
    pub(super) buf: Buffer,
    pub(super) len: usize,
    pub(super) general_prefix: Option<Vec<u32>>,
    /// Ascending indices of the scalars that are exactly 1, from the same
    /// classification pass. Host-only: what reaches the device is the per-job gather
    /// list [`Outputs::alloc`](super::Outputs::alloc) filters from it, since which of these
    /// indices matter also depends on the job's bases. `None` when the buffer was never
    /// classified.
    pub(super) ones_idx: Option<Vec<u32>>,
    /// Per-[`CLASSIFY_CHUNK`] maximum scalar bit length, from the same pass. This is
    /// what lets a plan drop its empty high windows: a scalar of `b` bits recodes
    /// exactly in `ceil((b + 1) / c)` signed windows, and on the csp circuits the
    /// witness values are mostly 32-bit words, so 43 of the 51 windows a full-width
    /// layout emits at c = 5 hold nothing. `None` when the buffer was never classified.
    pub(super) chunk_bits: Option<Vec<u32>>,
    /// Under [`Work::Constant`] a plan over this buffer ignores the three fields above
    /// even if they are set, and prices itself for `n` scalars over the full width.
    pub(super) work: Work,
    /// Somewhere a plan over the whole buffer may put its digit entries instead of the
    /// pool. See [`Lent`].
    pub(super) lent: Option<Lent>,
}

impl ScalarBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    /// This buffer with `lent` offered to the plan that covers all of it.
    pub(crate) fn with_lent(mut self, lent: Option<Lent>) -> Self {
        self.lent = lent;
        self
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// General scalars in `range`, or the whole range's length when the buffer came from
    /// the device and was never classified on the host. Overestimating is safe: it only
    /// oversizes the entry array and the window.
    pub(super) fn general_in(&self, range: &Range<usize>) -> usize {
        match &self.general_prefix {
            Some(p) => (p[range.end] - p[range.start]) as usize,
            None => range.len(),
        }
    }

    /// Maximum scalar bit length in `range`, rounded out to classification chunks, or
    /// the field's full width when the buffer came from the device and was never
    /// classified. Overestimating is safe for the same reason as [`Self::general_in`]:
    /// the extra windows' digits are all zero, so they only cost empty bucket rows.
    pub(super) fn bits_in(&self, range: &Range<usize>) -> usize {
        match &self.chunk_bits {
            Some(b) => {
                let lo = range.start / CLASSIFY_CHUNK;
                let hi = range.end.div_ceil(CLASSIFY_CHUNK).min(b.len());
                b[lo..hi].iter().copied().max().unwrap_or(0) as usize
            }
            None => RECODE_BITS - 1,
        }
    }
}

/// Device memory its owner lends a plan for the digit entries, and gets back when this is
/// dropped.
///
/// Stage 4's scratch lends H's plan the four domain vectors, which nothing reads once
/// `compute_h` has returned. At 2^22 that is the 537 MB entry array the pool would
/// otherwise hold beside them. The owner zeroes it with the rest of its scratch; nothing
/// here goes into the pool.
pub(crate) struct Lent {
    pub(super) buf: Buffer,
    busy: Arc<AtomicBool>,
}

impl Lent {
    /// `buf`, unless `busy` says another `Lent` over it is still alive: two MSMs over one
    /// H at once must not scatter into the same entries.
    pub(crate) fn claim(buf: &Buffer, busy: &Arc<AtomicBool>) -> Option<Self> {
        if busy.swap(true, Ordering::Acquire) {
            return None;
        }
        Some(Self {
            buf: buf.clone(),
            busy: Arc::clone(busy),
        })
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
    }
}

/// Bytes of digit entries a plan over `n` unclassified scalars takes, which is what
/// [`Plan::alloc`](super::Plan::alloc) asks for over a device-resident H. Stage 4 sizes the
/// buffer it lends with this; a lent buffer that turns out smaller is not used, so a
/// disagreement costs the saving and nothing else.
pub(crate) fn dense_entries_bytes(n: usize) -> Result<usize, ProveError> {
    let c = window_size_for(n, RECODE_BITS);
    crate::alloc::bytes(&[RECODE_BITS.div_ceil(c as usize), n.max(1), 8])
}
