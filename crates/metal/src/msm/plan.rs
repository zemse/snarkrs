//! A digit pipeline: everything about an MSM that depends only on its scalars, and
//! which jobs of a batch can share one.

use std::collections::HashMap;

use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState};
use snarkrs_groth16::ProveError;

use crate::enc::{dispatch_1d, set_params};
use crate::mtl::{self, Buffer, ComputeCommandEncoderRef};

use super::inputs::ScalarBuf;
use super::pool::Pool;
use super::window::{
    fold_len, legacy_accumulate, merge_wide_min_span, reduce_groups_for, scatter_split_rows,
    slice_len_for, window_size_for, Work, DUMMY_ROWS, HOST_TAIL_MAX_BUCKETS, RECODE_BITS, SCAN_TG,
};
use super::wire::MsmParams;
use super::MetalMsm;

/// A job whose scalar range is a suffix of another job's in the same batch rides that
/// job's digit pipeline when the prefix it skips is at most `1 / SHARE_MAX_SKIP` of the
/// wider range.
///
/// The L MSM reads the witness past its public inputs, a few entries short of what A and
/// B read, and its own plan was a second counting sort over the same digits: at 2^22 an
/// entry array of about 420 MB and a count, scan and scatter. Riding the wider plan, its
/// accumulation steps over the skipped points' entries, a load and a branch each, and its
/// window, buckets and spill follow the wider range. Both costs grow with the prefix,
/// hence the bound; the proving path skips `n_public + 1` of `n_vars`. Which entries
/// are skipped follows the digits of the constant wire and the public inputs, so under
/// [`Work::Constant`] the branch reveals nothing the verifier does not hold.
pub(super) const SHARE_MAX_SKIP: usize = 8;

/// Where the plan of each `(buffer, scalar_off, n)` job starts. Two jobs share a digit
/// pipeline when they read the same scalars over the same range, or when one's range is
/// a suffix of the other's that skips at most `1 / SHARE_MAX_SKIP` of it.
pub(super) fn plan_starts(jobs: &[(usize, usize, usize)]) -> Vec<usize> {
    // The lowest start of every range that ends at the same scalar of the same buffer.
    let mut lowest: HashMap<(usize, usize), usize> = HashMap::new();
    for &(id, soff, n) in jobs {
        let lo = lowest.entry((id, soff + n)).or_insert(soff);
        *lo = (*lo).min(soff);
    }
    jobs.iter()
        .map(|&(id, soff, n)| {
            let end = soff + n;
            let lo = lowest[&(id, end)];
            if (soff - lo) * SHARE_MAX_SKIP <= end - lo {
                lo
            } else {
                soff
            }
        })
        .collect()
}

pub(super) struct Plan<'a> {
    pub(super) n: usize,
    pub(super) scalar_off: usize,
    pub(super) c: u32,
    pub(super) n_windows: usize,
    pub(super) n_buckets: usize,
    pub(super) cap: usize,
    pub(super) slice_len: usize,
    pub(super) slices: usize,
    pub(super) reduce_groups: usize,
    /// Top-window rows the wide merge owns; 0 keeps everything on `msm_merge_*`.
    pub(super) merge_wide_rows: usize,
    /// Serial window-range dispatches of the scatter; 2 caps the live tail-line set
    /// on row-heavy plans (see [`SCATTER_SPLIT_ROWS`](super::window::SCATTER_SPLIT_ROWS)),
    /// 1 everywhere else.
    pub(super) scatter_parts: usize,
    pub(super) host_tail: bool,
    /// Classified inputs have an exact ones gather. Unclassified inputs instead
    /// recode 1 into bucket zero, so an H plan needs no separate full-buffer scan.
    pub(super) separate_ones: bool,
    /// [`DUMMY_ROWS`] under [`Work::Constant`], 0 otherwise. The dummy rows follow the
    /// real bucket rows in `counts`, `cursor` and the bucket array, and only the digit
    /// pipeline and the accumulation ever address them.
    pub(super) dummy_rows: usize,
    pub(super) scalars: &'a Buffer,
    /// Where the entries go if it is big enough; the pool otherwise.
    pub(super) lent: Option<&'a Buffer>,
    pub(super) counts: Option<Buffer>,
    pub(super) cursor: Option<Buffer>,
    pub(super) entries: Option<Buffer>,
}

impl<'a> Plan<'a> {
    pub(super) fn new(scalars: &'a ScalarBuf, scalar_off: usize, n: usize) -> Self {
        let range = scalar_off..scalar_off + n;
        let constant = scalars.work == Work::Constant;
        // Constant work plans for the range's length whatever the buffer knows about
        // its contents: every scalar reaches the buckets, in every window.
        let general = if constant {
            n
        } else {
            scalars.general_in(&range)
        };
        // One bit past the range's longest scalar, for the signed carry: a scalar of
        // `b` bits recodes exactly in `ceil((b + 1) / c)` windows, with zero carry out
        // of the top one. Windows above that hold no digit on any scalar in the range,
        // so planning them would only clear, merge and reduce empty rows. H's scalars
        // are device-resident and unclassified, so `bits_in` keeps its full bound.
        let recode_bits = if constant {
            RECODE_BITS
        } else {
            (scalars.bits_in(&range) + 1).min(RECODE_BITS)
        };
        let c = window_size_for(general, recode_bits);
        let n_windows = recode_bits.div_ceil(c as usize);
        let n_buckets = 1usize << (c - 1);
        let cap = general.max(1);
        let slice_len = slice_len_for(n_windows, cap, scalars.work);
        // Live buckets of the top window, whose short bit range concentrates the
        // digits. When their expected run span clears the wide threshold, their merge
        // moves to `msm_merge_wide_*` and `msm_merge_*` skips them.
        let top_bits = recode_bits - (n_windows - 1) * c as usize;
        let top_buckets = 1usize << (top_bits.min(c as usize) - 1);
        let wide_span = merge_wide_min_span();
        let merge_wide_rows = if wide_span > 0 && general / top_buckets / slice_len >= wide_span {
            top_buckets
        } else {
            0
        };
        Self {
            n,
            scalar_off,
            c,
            n_windows,
            n_buckets,
            // Classified inputs emit only general scalars. Unclassified inputs use n
            // as their bound, including any ones they recode into buckets. Each emits
            // at most one entry per window. `max(1)` avoids a zero-length Metal buffer.
            cap,
            slice_len,
            slices: cap.div_ceil(slice_len).max(1),
            reduce_groups: reduce_groups_for(n_windows, n_buckets),
            merge_wide_rows,
            scatter_parts: if n_windows > 1 && n_windows * n_buckets >= scatter_split_rows() {
                2
            } else {
                1
            },
            // A constant plan spans 255 bits, so it never has one window; stated so the
            // host-tail fold, which indexes the real buckets by spill row, cannot meet a
            // dummy row.
            host_tail: !constant && n_windows == 1 && n_buckets <= HOST_TAIL_MAX_BUCKETS,
            separate_ones: !constant
                && (scalars.ones_idx.is_some()
                    || std::env::var("G16_METAL_MSM_ROUTE_ONES").as_deref() == Ok("0")),
            dummy_rows: if constant { DUMMY_ROWS } else { 0 },
            scalars: &scalars.buf,
            // Whole-buffer plans only. Two jobs over the same range share one plan, so
            // this is at most one plan per buffer.
            lent: scalars
                .lent
                .as_ref()
                .filter(|_| scalar_off == 0 && n == scalars.len)
                .map(|l| &l.buf),
            counts: None,
            cursor: None,
            entries: None,
        }
    }

    /// The constant-work merge's levels, as (spill slots per window in, groups per
    /// window), from the accumulation's two slots per slice down to one group; empty on
    /// every other plan, which merges with `msm_merge_*`.
    pub(super) fn fold_levels(&self) -> Vec<(usize, usize)> {
        if self.dummy_rows == 0 || legacy_accumulate() {
            return Vec::new();
        }
        let len = fold_len();
        let mut levels = Vec::new();
        let mut m = 2 * self.slices;
        loop {
            let groups = m.div_ceil(len);
            levels.push((m, groups));
            if groups == 1 {
                return levels;
            }
            m = 2 * groups;
        }
    }

    /// Spill slots the segmented accumulation writes per slice. One holds the run a slice
    /// starts with, which may continue an earlier slice's bucket; every later run starts
    /// in the slice and goes straight to its bucket. The constant-work fold overwrites
    /// the buckets it writes, so there the slice's last run takes a second slot instead.
    ///
    /// A second slot on every plan was 67 MB of the H plan's scratch at 2^21 (c=16,
    /// 32,768 slices a window, 128 bytes a point).
    pub(super) fn spill_per_slice(&self) -> usize {
        if self.dummy_rows > 0 {
            2
        } else {
            1
        }
    }

    /// Rows in `counts`, `cursor` and the bucket array: the real buckets of every window,
    /// then every window's dummy rows.
    pub(super) fn rows(&self) -> usize {
        self.n_windows * (self.n_buckets + self.dummy_rows)
    }

    pub(super) fn params(&self) -> MsmParams {
        MsmParams {
            n: self.n as u32,
            c: self.c,
            n_windows: self.n_windows as u32,
            n_buckets: self.n_buckets as u32,
            cap: self.cap as u32,
            scalar_off: self.scalar_off as u32,
            base_off: 0,
            ones_groups: 0,
            slice_len: self.slice_len as u32,
            slices: self.slices as u32,
            reduce_groups: self.reduce_groups as u32,
            merge_wide_base: ((self.n_windows - 1) * self.n_buckets) as u32,
            merge_wide_rows: self.merge_wide_rows as u32,
            separate_ones: self.separate_ones as u32,
            dummy_rows: self.dummy_rows as u32,
            seg_first: 0,
            skip: 0,
        }
    }

    pub(super) fn alloc(&mut self, pool: &Pool, keep: &mut Vec<Buffer>) -> Result<(), ProveError> {
        let rows = self.rows();
        let counts = pool.take(&[rows, 4])?;
        let cursor = pool.take(&[rows, 4])?;
        // 8 bytes an entry: the bucket row travels with the point index so the
        // segmented accumulation can find run boundaries without recomputing digits.
        let bytes = crate::alloc::bytes(&[self.n_windows, self.cap, 8])?;
        let entries = match self.lent.filter(|b| b.length() >= bytes) {
            // Not kept: it goes back to its owner, not into the pool.
            Some(b) => b.clone(),
            None => {
                let e = pool.take(&[bytes])?;
                keep.push(e.clone());
                e
            }
        };
        keep.push(counts.clone());
        keep.push(cursor.clone());
        self.counts = Some(counts);
        self.cursor = Some(cursor);
        self.entries = Some(entries);
        Ok(())
    }

    pub(super) fn encode(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        self.encode_zero(msm, enc);
        self.encode_count(msm, enc);
        self.encode_scan(msm, enc);
        self.encode_scatter(msm, enc);
    }

    /// The counters must start at zero, and unlike the bucket array (whose kernel
    /// writes rather than accumulates) they cannot rely on a fresh allocation, since
    /// the pool hands back used buffers.
    pub(super) fn encode_zero(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let rows = self.rows();
        enc.setComputePipelineState(&msm.pipelines.zero_u32);
        mtl::set_buffer(enc, 0, self.counts.as_ref().expect("plan not allocated"), 0);
        let len = rows as u32;
        set_params(enc, 1, &len);
        dispatch_1d(enc, &msm.pipelines.zero_u32, rows, 256);
    }

    pub(super) fn encode_count(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.setComputePipelineState(&msm.pipelines.count);
        mtl::set_buffer(enc, 0, self.scalars, 0);
        mtl::set_buffer(enc, 1, self.counts.as_ref().expect("plan not allocated"), 0);
        set_params(enc, 2, &p);
        dispatch_1d(enc, &msm.pipelines.count, self.n, 64);
    }

    pub(super) fn encode_scan(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.setComputePipelineState(&msm.pipelines.scan);
        mtl::set_buffer(enc, 0, self.counts.as_ref().expect("plan not allocated"), 0);
        mtl::set_buffer(enc, 1, self.cursor.as_ref().expect("plan not allocated"), 0);
        set_params(enc, 2, &p);
        let scan_tg = SCAN_TG.min(msm.pipelines.scan.maxTotalThreadsPerThreadgroup());
        crate::cb::dispatch_thread_groups(enc, mtl::size(self.n_windows), mtl::size(scan_tg));
    }

    /// The scatter, or the `part`-th of its window-range dispatches. The parts write
    /// disjoint rows, so no barrier is *required* between them; the caller inserts one
    /// anyway, because running them serially is the entire point of the split.
    pub(super) fn encode_scatter_part(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        part: usize,
    ) {
        if part >= self.scatter_parts {
            return;
        }
        let step = self.n_windows.div_ceil(self.scatter_parts);
        let lo = part * step;
        let hi = (lo + step).min(self.n_windows);
        let p = self.params();
        enc.setComputePipelineState(&msm.pipelines.scatter);
        mtl::set_buffer(enc, 0, self.scalars, 0);
        mtl::set_buffer(enc, 1, self.cursor.as_ref().expect("plan not allocated"), 0);
        mtl::set_buffer(
            enc,
            2,
            self.entries.as_ref().expect("plan not allocated"),
            0,
        );
        set_params(enc, 3, &p);
        let span = [lo as u32, hi as u32];
        set_params(enc, 4, &span);
        dispatch_1d(enc, &msm.pipelines.scatter, self.n, 64);
    }

    pub(super) fn encode_scatter(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        for part in 0..self.scatter_parts {
            self.encode_scatter_part(msm, enc, part);
        }
    }
}
