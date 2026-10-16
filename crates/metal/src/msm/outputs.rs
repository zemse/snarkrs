//! One MSM's point stages: its buckets, spill and partial-sum buffers, the
//! dispatches that fill them, and the host combine that reads them back.

use ark_ff::{AdditiveGroup, Zero};
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState};
use snarkrs_field::{G1Projective, G2Projective};
use snarkrs_groth16::ProveError;

use crate::enc::{dispatch_1d, read_back, set_params};
use crate::mtl::{self, Buffer, ComputeCommandEncoderRef};

use super::jobs::{Job, MsmResult};
use super::submit::barrier;
use super::window::{
    fold_len, legacy_accumulate, ones_groups_for, MERGE_TG, MSM_NO_ROW, REDUCE_TG,
};
use super::wire::{MsmParams, PackedXyzzG1, PackedXyzzG2};
use super::{MetalMsm, Plan};

pub(super) struct Outputs {
    pub(super) buckets: Buffer,
    pub(super) spill_pts: Buffer,
    pub(super) spill_rows: Buffer,
    pub(super) window_sums: Buffer,
    pub(super) ones: Buffer,
    pub(super) ones_groups: usize,
    /// Base indices of the live one-scalar contributions, when the host classified the
    /// scalars: `msm_ones_idx_*` gathers exactly these. `None` falls back to the
    /// `msm_ones_*` scan over all `n` scalars, which is the only option for a
    /// device-resident buffer.
    pub(super) ones_idx: Option<(Buffer, usize)>,
    /// The second half of the constant-work merge's ping-pong, sized for its first
    /// level's output; the spill buffers are the other half. `None` off that path.
    pub(super) fold: Option<(Buffer, Buffer)>,
    pub(super) base_off: usize,
    /// Leading scalars of the plan's range this job does not cover, when it rides a wider
    /// job's plan (see [`SHARE_MAX_SKIP`](super::plan::SHARE_MAX_SKIP)); 0 otherwise.
    pub(super) skip: usize,
    pub(super) is_g2: bool,
}

impl Outputs {
    pub(super) fn alloc(
        msm: &MetalMsm,
        keep: &mut Vec<Buffer>,
        job: &Job<'_>,
        plan: &Plan<'_>,
        skip: usize,
    ) -> Result<Self, ProveError> {
        let pool = &msm.pool;
        let (is_g2, base_off, point_bytes, scalar_off, n, sbuf, inf) = match job {
            Job::G1(j) => (
                false,
                j.base_off,
                core::mem::size_of::<PackedXyzzG1>(),
                j.scalar_off,
                j.n,
                j.scalars,
                &j.bases.inf,
            ),
            Job::G2(j) => (
                true,
                j.base_off,
                core::mem::size_of::<PackedXyzzG2>(),
                j.scalar_off,
                j.n,
                j.scalars,
                &j.bases.inf,
            ),
        };

        // A classified scalar buffer turns the ones scan into a gather. The buffer's
        // one-indices are sorted, so the job's range is a subrange; mapping each index
        // to its base and dropping the bases at infinity happens here, once per proof,
        // because it is exactly what the kernel would otherwise discover point by
        // point. On the csp B queries, 61% of the bases are infinity, and this is
        // where their scalars stop costing anything. A constant-work plan has no ones
        // path at all, whatever the buffer knows.
        let ones_idx = match sbuf.ones_idx.as_ref().filter(|_| plan.dummy_rows == 0) {
            Some(idx) => {
                let lo = idx.partition_point(|&e| (e as usize) < scalar_off);
                let hi = idx.partition_point(|&e| (e as usize) < scalar_off + n);
                let gather: Vec<u32> = idx[lo..hi]
                    .iter()
                    .map(|&e| (base_off + (e as usize - scalar_off)) as u32)
                    .filter(|&b| !inf[b as usize])
                    .collect();
                let buf = pool.take(gather.len() * 4)?;
                if !gather.is_empty() {
                    // SAFETY: the pooled buffer holds at least `gather.len()` u32s and
                    // the batch that could read it has not been committed yet.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            gather.as_ptr(),
                            buf.contents().cast::<u32>().as_ptr(),
                            gather.len(),
                        );
                    }
                }
                Some((buf, gather.len()))
            }
            None => None,
        };
        let ones_groups = match &ones_idx {
            Some((_, count)) => ones_groups_for(*count),
            None if plan.separate_ones => ones_groups_for(n),
            None => 0,
        };
        // The dummy rows are written by the accumulation like any interior run, and
        // never read; the clear covers the real rows only.
        let buckets = pool.take(plan.rows() * point_bytes)?;
        // At most one run of a slice continues backwards, into a bucket an earlier slice
        // owns. See [`Plan::spill_per_slice`] for the second slot.
        let spill_slots = plan.spill_per_slice() * plan.n_windows * plan.slices;
        let spill_pts = pool.take(spill_slots * point_bytes)?;
        let spill_rows = pool.take(spill_slots * 4)?;
        let fold = match plan.fold_levels().first() {
            Some(&(_, groups)) => {
                let slots = 2 * plan.n_windows * groups;
                let pts = pool.take(slots * point_bytes)?;
                let rows = pool.take(slots * 4)?;
                keep.push(pts.clone());
                keep.push(rows.clone());
                Some((pts, rows))
            }
            None => None,
        };
        // The G1 reduce writes a pair per simdgroup. Query this pipeline's width:
        // at width 16 a 64-thread group writes eight points, not four.
        let sums_per_group = if is_g2 {
            1
        } else {
            2 * msm.reduce_layout(plan).simdgroups
        };
        let window_sums =
            pool.take(plan.n_windows * plan.reduce_groups * sums_per_group * point_bytes)?;
        let ones = pool.take(ones_groups * point_bytes)?;
        keep.push(buckets.clone());
        keep.push(spill_pts.clone());
        keep.push(spill_rows.clone());
        keep.push(window_sums.clone());
        keep.push(ones.clone());
        if let Some((buf, _)) = &ones_idx {
            keep.push(buf.clone());
        }
        Ok(Self {
            buckets,
            spill_pts,
            spill_rows,
            window_sums,
            ones,
            ones_groups,
            ones_idx,
            fold,
            base_off,
            skip,
            is_g2,
        })
    }

    /// The Pippenger half: clear, segmented accumulation, merge, reduce. Split from
    /// [`Self::encode_ones`] so phase mode can time the bucket machinery and the ones
    /// scan separately. The production path in [`MetalMsm::msm_batch`] does not call
    /// this: it encodes the individual stages itself, grouped across jobs, so that
    /// independent dispatches share a concurrent phase.
    pub(super) fn encode_buckets(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        self.encode_clear(msm, enc, job, plan);
        self.encode_accumulate(msm, enc, job, plan);
        self.encode_merge(msm, enc, job, plan);
        self.encode_reduce(msm, enc, job, plan);
    }

    pub(super) fn params_for(&self, plan: &Plan<'_>) -> MsmParams {
        let mut p = plan.params();
        p.base_off = self.base_off as u32;
        p.ones_groups = self.ones_groups as u32;
        p.skip = self.skip as u32;
        p
    }

    /// A bucket that no slice writes directly has to read as the identity. Dispatched on
    /// the legacy accumulation too, although that one writes every row: skipping it there
    /// made the legacy path correct only while that kernel stays unconditional, and the
    /// clear is about 0.1 ms.
    pub(super) fn encode_clear(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        let p = self.params_for(plan);
        let clear_pso = match job {
            Job::G1(_) => &msm.pipelines.clear_g1,
            Job::G2(_) => &msm.pipelines.clear_g2,
        };
        enc.setComputePipelineState(clear_pso);
        mtl::set_buffer(enc, 0, &self.buckets, 0);
        set_params(enc, 1, &p);
        dispatch_1d(enc, clear_pso, plan.n_windows * plan.n_buckets, 256);
    }

    pub(super) fn encode_accumulate(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        self.encode_accumulate_slices(msm, enc, job, plan, 0, plan.n_windows * plan.slices);
    }

    /// The accumulation over `count` slices from global slice `first`, so a split
    /// submission can hand out whole windows a command buffer at a time. The legacy
    /// kernel owns rows, not slices, and ignores the range.
    pub(super) fn encode_accumulate_slices(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
        first: usize,
        count: usize,
    ) {
        let mut p = self.params_for(plan);
        p.seg_first = first as u32;
        let rows = plan.n_windows * plan.n_buckets;
        if legacy_accumulate() {
            let (acc_pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.accumulate_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.accumulate_g2, &j.bases.buf),
            };
            enc.setComputePipelineState(acc_pso);
            mtl::set_buffer(enc, 0, plan.entries.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 1, bases, 0);
            mtl::set_buffer(enc, 2, plan.counts.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 3, plan.cursor.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 4, &self.buckets, 0);
            set_params(enc, 5, &p);
            dispatch_1d(enc, acc_pso, rows, 64);
        } else {
            let (seg_pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.segmented_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.segmented_g2, &j.bases.buf),
            };
            enc.setComputePipelineState(seg_pso);
            mtl::set_buffer(enc, 0, plan.entries.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 1, bases, 0);
            mtl::set_buffer(enc, 2, plan.cursor.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 3, &self.buckets, 0);
            mtl::set_buffer(enc, 4, &self.spill_pts, 0);
            mtl::set_buffer(enc, 5, &self.spill_rows, 0);
            set_params(enc, 6, &p);
            dispatch_1d(enc, seg_pso, count, 64);
        }
    }

    /// Every level of the merge, with a barrier between levels. The production path
    /// calls [`Self::encode_merge_level`] instead, one level of every job at a time.
    pub(super) fn encode_merge(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        let mut level = 0;
        while self.encode_merge_level(msm, enc, job, plan, level) {
            barrier(enc);
            level += 1;
        }
    }

    /// Level `level` of the merge; true when another follows and reads what this one
    /// wrote. A constant-work plan folds its spill slots through
    /// [`Plan::fold_levels`], ping-ponging between the spill buffers and `fold`;
    /// every other plan merges in one level.
    pub(super) fn encode_merge_level(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
        level: usize,
    ) -> bool {
        let levels = plan.fold_levels();
        if let Some(&(m_in, groups)) = levels.get(level) {
            let (fold_pts, fold_rows) = self.fold.as_ref().expect("a fold plan has its buffers");
            let spill = (&self.spill_pts, &self.spill_rows);
            let (src, dst) = if level.is_multiple_of(2) {
                (spill, (fold_pts, fold_rows))
            } else {
                ((fold_pts, fold_rows), spill)
            };
            let last = level + 1 == levels.len();
            let f: [u32; 4] = [m_in as u32, groups as u32, fold_len() as u32, last as u32];
            let pso = match job {
                Job::G1(_) => &msm.pipelines.fold_g1,
                Job::G2(_) => &msm.pipelines.fold_g2,
            };
            enc.setComputePipelineState(pso);
            mtl::set_buffer(enc, 0, src.0, 0);
            mtl::set_buffer(enc, 1, src.1, 0);
            mtl::set_buffer(enc, 2, dst.0, 0);
            mtl::set_buffer(enc, 3, dst.1, 0);
            mtl::set_buffer(enc, 4, &self.buckets, 0);
            set_params(enc, 5, &self.params_for(plan));
            set_params(enc, 6, &f);
            dispatch_1d(enc, pso, plan.n_windows * groups, 64);
            return !last;
        }
        if level > 0 || legacy_accumulate() || plan.host_tail {
            return false;
        }
        let p = self.params_for(plan);
        let merge_pso = match job {
            Job::G1(_) => &msm.pipelines.merge_g1,
            Job::G2(_) => &msm.pipelines.merge_g2,
        };
        enc.setComputePipelineState(merge_pso);
        mtl::set_buffer(enc, 0, &self.buckets, 0);
        mtl::set_buffer(enc, 1, &self.spill_pts, 0);
        mtl::set_buffer(enc, 2, &self.spill_rows, 0);
        mtl::set_buffer(enc, 3, plan.counts.as_ref().unwrap(), 0);
        mtl::set_buffer(enc, 4, plan.cursor.as_ref().unwrap(), 0);
        set_params(enc, 5, &p);
        dispatch_1d(enc, merge_pso, plan.n_windows * plan.n_buckets, 64);

        // The top window's fat buckets, one threadgroup each. Disjoint rows from the
        // dispatch above and the same read-only inputs, so the two share a phase.
        if plan.merge_wide_rows > 0 {
            let wide_pso = match job {
                Job::G1(_) => &msm.pipelines.merge_wide_g1,
                Job::G2(_) => &msm.pipelines.merge_wide_g2,
            };
            enc.setComputePipelineState(wide_pso);
            mtl::set_buffer(enc, 0, &self.buckets, 0);
            mtl::set_buffer(enc, 1, &self.spill_pts, 0);
            mtl::set_buffer(enc, 2, &self.spill_rows, 0);
            mtl::set_buffer(enc, 3, plan.counts.as_ref().unwrap(), 0);
            mtl::set_buffer(enc, 4, plan.cursor.as_ref().unwrap(), 0);
            set_params(enc, 5, &p);
            let tg = MERGE_TG.min(wide_pso.maxTotalThreadsPerThreadgroup());
            crate::cb::dispatch_thread_groups(enc, mtl::size(plan.merge_wide_rows), mtl::size(tg));
        }
        false
    }

    pub(super) fn encode_reduce(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        if plan.host_tail {
            return;
        }
        let p = self.params_for(plan);
        let is_g2 = matches!(job, Job::G2(_));
        enc.setComputePipelineState(msm.reduce_pso(is_g2, plan));
        mtl::set_buffer(enc, 0, &self.buckets, 0);
        mtl::set_buffer(enc, 1, &self.window_sums, 0);
        set_params(enc, 2, &p);
        let tg = msm.reduce_tg(is_g2, plan);
        crate::cb::dispatch_thread_groups(
            enc,
            mtl::size(plan.n_windows * plan.reduce_groups),
            mtl::size(tg),
        );
    }

    /// The scalar-of-1 half: one dispatch. Independent of the digit pipeline and of
    /// every bucket stage. With a gather list it is `msm_ones_idx_*` over exactly the
    /// live contributions. Unclassified scalars normally route ones into the buckets;
    /// the measurement override restores the `msm_ones_*` scan over all n scalars.
    pub(super) fn encode_ones(
        &self,
        msm: &MetalMsm,
        enc: &ComputeCommandEncoderRef,
        job: &Job<'_>,
        plan: &Plan<'_>,
    ) {
        // H's empty ones scan measured 0.23/0.22/0.27 ms at 2^17/2^18/2^19,
        // including the isolated submission floor. Recode ones into the buckets
        // when there is no host classification; the entry capacity already bounds n.
        // G16_METAL_MSM_ROUTE_ONES=0 restores the scan for an end-to-end comparison.
        if !plan.separate_ones {
            return;
        }
        let mut p = self.params_for(plan);
        if let Some((idx, count)) = &self.ones_idx {
            let (pso, bases) = match job {
                Job::G1(j) => (&msm.pipelines.ones_idx_g1, &j.bases.buf),
                Job::G2(j) => (&msm.pipelines.ones_idx_g2, &j.bases.buf),
            };
            // The gather kernel reads its length from `n`; the plan's other counts do
            // not apply to it.
            p.n = *count as u32;
            enc.setComputePipelineState(pso);
            mtl::set_buffer(enc, 0, idx, 0);
            mtl::set_buffer(enc, 1, bases, 0);
            mtl::set_buffer(enc, 2, &self.ones, 0);
            set_params(enc, 3, &p);
            let tg = REDUCE_TG.min(pso.maxTotalThreadsPerThreadgroup());
            crate::cb::dispatch_thread_groups(enc, mtl::size(self.ones_groups), mtl::size(tg));
            return;
        }
        let (ones_pso, bases) = match job {
            Job::G1(j) => (&msm.pipelines.ones_g1, &j.bases.buf),
            Job::G2(j) => (&msm.pipelines.ones_g2, &j.bases.buf),
        };
        // The scan walks the job's own range, not the plan's.
        p.n = (plan.n - self.skip) as u32;
        p.scalar_off = (plan.scalar_off + self.skip) as u32;
        enc.setComputePipelineState(ones_pso);
        mtl::set_buffer(enc, 0, plan.scalars, 0);
        mtl::set_buffer(enc, 1, bases, 0);
        mtl::set_buffer(enc, 2, &self.ones, 0);
        set_params(enc, 3, &p);
        let tg = REDUCE_TG.min(ones_pso.maxTotalThreadsPerThreadgroup());
        crate::cb::dispatch_thread_groups(enc, mtl::size(self.ones_groups), mtl::size(tg));
    }

    pub(super) fn combine(&self, msm: &MetalMsm, plan: &Plan<'_>) -> MsmResult {
        let rg = plan.reduce_groups;
        if self.is_g2 {
            // SAFETY: `alloc` sized `ones` at `ones_groups` points and the ones scan
            // wrote one per group. Every read in this function is off the batch
            // `msm_batch` has already waited on, so the writes have landed.
            let o: &[PackedXyzzG2] = unsafe { read_back(&self.ones, self.ones_groups) };
            if plan.host_tail {
                // The GPU stopped after the accumulation; see [`HOST_TAIL_MAX_BUCKETS`].
                // Fold the spilled runs into the buckets, then the reverse running sum
                // gives sum_j (j+1) B_j, the whole window in a one-window plan.
                //
                // SAFETY: the bucket array holds `n_windows * n_buckets` points, and
                // `host_tail` implies a single window, so `n_buckets` is all of it.
                let b: &[PackedXyzzG2] = unsafe { read_back(&self.buckets, plan.n_buckets) };
                let mut buckets: Vec<G2Projective> = b.iter().map(|x| x.to_projective()).collect();
                if !legacy_accumulate() {
                    let slots = plan.spill_per_slice() * plan.slices;
                    // SAFETY: both spill arrays hold `spill_per_slice * n_windows *
                    // slices` slots and the single window again makes `slots` the whole
                    // of each. The accumulation writes every slot, sentinel row included.
                    let rows: &[u32] = unsafe { read_back(&self.spill_rows, slots) };
                    let pts: &[PackedXyzzG2] = unsafe { read_back(&self.spill_pts, slots) };
                    for (r, pt) in rows.iter().zip(pts) {
                        if *r != MSM_NO_ROW {
                            buckets[*r as usize] += pt.to_projective();
                        }
                    }
                }
                let mut run = G2Projective::zero();
                let mut acc = G2Projective::zero();
                for b in buckets.iter().rev() {
                    run += b;
                    if !run.is_zero() {
                        acc += &run;
                    }
                }
                for x in o {
                    acc += x.to_projective();
                }
                return MsmResult::G2(acc);
            }
            // SAFETY: the G2 reduce writes one point per (window, group), which is the
            // `sums_per_group = 1` `alloc` sized `window_sums` with.
            let w: &[PackedXyzzG2] = unsafe { read_back(&self.window_sums, plan.n_windows * rg) };
            // Each window's `reduce_groups` partials fold first, then the Horner.
            let sum_w = |k: usize| {
                let mut s = w[k * rg].to_projective();
                for x in &w[k * rg + 1..(k + 1) * rg] {
                    s += x.to_projective();
                }
                s
            };
            let mut acc = sum_w(plan.n_windows - 1);
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += sum_w(k);
            }
            for x in o {
                acc += x.to_projective();
            }
            MsmResult::G2(acc)
        } else {
            // SAFETY: as in the G2 arm — `ones_groups` points, written by the completed
            // batch.
            let o: &[PackedXyzzG1] = unsafe { read_back(&self.ones, self.ones_groups) };
            if plan.host_tail {
                // Same shape as the G2 arm above, and the same lengths hold for the same
                // reason: `host_tail` implies one window.
                // SAFETY: `n_buckets` is the whole one-window bucket array.
                let b: &[PackedXyzzG1] = unsafe { read_back(&self.buckets, plan.n_buckets) };
                let mut buckets: Vec<G1Projective> = b.iter().map(|x| x.to_projective()).collect();
                if !legacy_accumulate() {
                    let slots = plan.spill_per_slice() * plan.slices;
                    // SAFETY: `slots` is the whole of each one-window spill array.
                    let rows: &[u32] = unsafe { read_back(&self.spill_rows, slots) };
                    let pts: &[PackedXyzzG1] = unsafe { read_back(&self.spill_pts, slots) };
                    for (r, pt) in rows.iter().zip(pts) {
                        if *r != MSM_NO_ROW {
                            buckets[*r as usize] += pt.to_projective();
                        }
                    }
                }
                let mut run = G1Projective::zero();
                let mut acc = G1Projective::zero();
                for b in buckets.iter().rev() {
                    run += b;
                    if !run.is_zero() {
                        acc += &run;
                    }
                }
                for x in o {
                    acc += x.to_projective();
                }
                return MsmResult::G1(acc);
            }
            // The G1 reduce hands back a (partial, weight) PAIR per simdgroup; slot s
            // of a window carries the buckets based at `(s/sgs)*chunk + (s%sgs)*stride`
            // (see msm_reduce_scan_impl). The bases step uniformly whenever the chunk
            // splits evenly over the threadgroup, as the large unforced shapes do, and
            // then the weighted term folds to suffix additions plus one small multiple
            // per window. Small chunks and overrides that break the stride pay one
            // multiple per slot instead. The fold tripled in additions against the old
            // one-point-per-group readback, so the windows fold on the thread pool
            // before the serial Horner, which keeps the same window order as the CPU
            // backend so the audit can compare affine.
            let layout = msm.reduce_layout(plan);
            let slots = layout.slots;
            // SAFETY: `alloc` sized `window_sums` at `n_windows * reduce_groups *
            // sums_per_group` points with `sums_per_group = 2 * simdgroups`, from this
            // same `ReduceLayout`; `slots` is `reduce_groups * simdgroups`, so
            // `n_windows * slots * 2` is exactly what it allocated and the reduce wrote.
            let w: &[PackedXyzzG1] =
                unsafe { read_back(&self.window_sums, plan.n_windows * slots * 2) };
            let sum_w = |k: usize| layout.fold(&w[k * slots * 2..(k + 1) * slots * 2]);
            let sums: Vec<G1Projective> = {
                use rayon::prelude::*;
                (0..plan.n_windows).into_par_iter().map(sum_w).collect()
            };
            let mut acc = sums[plan.n_windows - 1];
            for k in (0..plan.n_windows - 1).rev() {
                for _ in 0..plan.c {
                    acc.double_in_place();
                }
                acc += sums[k];
            }
            for x in o {
                acc += x.to_projective();
            }
            MsmResult::G1(acc)
        }
    }
}

/// The G1 shader and readback must use the same SIMD width, including partial groups.
/// Kept independent of the device so the weighting identity can be checked on the CPU.
pub(super) struct ReduceLayout {
    pub(super) simdgroups: usize,
    pub(super) slots: usize,
    pub(super) chunk: usize,
    pub(super) stride: usize,
}

impl ReduceLayout {
    pub(super) fn new(buckets: usize, groups: usize, threads: usize, width: usize) -> Self {
        let simdgroups = threads.div_ceil(width);
        let chunk = buckets.div_ceil(groups);
        Self {
            simdgroups,
            slots: groups * simdgroups,
            chunk,
            stride: width * chunk.div_ceil(threads),
        }
    }

    pub(super) fn fold(&self, pairs: &[PackedXyzzG1]) -> G1Projective {
        assert_eq!(pairs.len(), 2 * self.slots);
        if self.chunk == self.simdgroups * self.stride {
            let mut c = G1Projective::zero();
            let mut suff = G1Projective::zero();
            let mut t = G1Projective::zero();
            for s in (0..self.slots).rev() {
                c += pairs[2 * s].to_projective();
                suff += pairs[2 * s + 1].to_projective();
                if s >= 1 {
                    t += &suff;
                }
            }
            c + mul_u64(t, self.stride as u64)
        } else {
            let mut c = G1Projective::zero();
            for s in 0..self.slots {
                let base = (s / self.simdgroups) * self.chunk + (s % self.simdgroups) * self.stride;
                c += pairs[2 * s].to_projective();
                c += mul_u64(pairs[2 * s + 1].to_projective(), base as u64);
            }
            c
        }
    }
}

/// `k * p` by MSB-first double-and-add. `k` is a bucket base, a few thousand at most,
/// and this runs a handful of times per window inside the combine.
pub(super) fn mul_u64(p: G1Projective, k: u64) -> G1Projective {
    let mut acc = G1Projective::zero();
    if k == 0 || p.is_zero() {
        return acc;
    }
    for i in (0..64 - k.leading_zeros()).rev() {
        acc.double_in_place();
        if (k >> i) & 1 == 1 {
            acc += &p;
        }
    }
    acc
}
