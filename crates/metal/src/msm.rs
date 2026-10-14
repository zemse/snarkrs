//! Stages 5-9: the five Pippenger multi-scalar multiplications, on the GPU.
//!
//! The algorithm and every decision behind it are documented at the top of
//! `src/shaders/msm.metal`, which is the half of this module that does the work. In
//! summary: a counting sort by bucket index built from 32-bit atomics on plain counters,
//! so no bucket is ever written by two threads; a segmented accumulation that gives every
//! thread a fixed-length slice of the sorted entries, so per-thread work is uniform even
//! when one bucket holds a tenth of the input; and a threadgroup reduction that hands the
//! host one point per window.
//!
//! What this file owns is the shape of the submission, and that is where the Apple
//! silicon result actually lives. Measured on this M2 Max during the scouting phase, a
//! command buffer costs 0.16 ms to commit and wait on, and does not get cheaper with
//! less work in it, while an extra dispatch encoded into an already-open command buffer
//! costs 2 to 3 us. That is a factor of about 60. Both Metal MSM implementations we read
//! submit one command buffer per stage, four or more per MSM; at this prover's domain
//! sizes that alone would cost more than the arithmetic. [`MetalMsm::msm_batch`]
//! therefore encodes every job it is handed into ONE command buffer with ONE
//! `wait_until_completed`, and reads back a few hundred points at the end. The proving
//! path hands it all five MSMs below a 2^17 domain; at and above it `MetalCircuit`
//! splits them into two batches so the witness four can run while `compute_h` is still
//! on the other queue (see `backend::overlap_pays`).
//!
//! # What is on the GPU and what is not
//!
//! On the GPU: the scalar classification and signed-digit recoding, the counting sort,
//! bucket accumulation for both G1 and G2, the bucket merge, the per-window reduction,
//! and the sum of the bases whose scalar is 1. G2 is not a CPU fallback; `Fq` and `Fq2`
//! are implemented in MSL in this crate because the field prelude only covers `Fr`.
//!
//! On the host: packing the scalars into standard form, choosing the window width, the
//! Horner combination of `n_windows` points per MSM, and the sum of the per-threadgroup
//! partials. That is a few hundred curve additions per proof against tens of millions on
//! the device, and it is the same place zkonduit's Metal MSM leaves its window tail, for
//! the same reason: the serial part is small enough that moving it costs more in
//! dispatches than it saves.
//!
//! # Measured, on this M2 Max
//!
//! Stages 5-9 only, warm, medians of nine. Bases upload is `prepare` work and is outside
//! both timed regions; host scalar packing is per-proof work and is inside the GPU's.
//!
//! | artifact | constraints | CPU ms | GPU ms | speedup |
//! |---|---|---|---|---|
//! | `tiny_mul` | 2 | 0.26 | 2.5 | **0.10x** |
//! | `js_1x1_d8` | 3,359 | 17.9 | 17.5 | 1.02x |
//! | `js_2x2_d16` | 10,153 | 48.8 | 25.4 | 1.9x |
//! | `js_2x2_d32` | 17,929 | 83.4 | 30.4 | 2.7x |
//! | `js_8x8_d32` | 70,357 | 284 | 76 | 3.7x |
//! | `js_16x16_d32` | 140,261 | 533 | 127 | 4.2x |
//!
//! The crossover sits at roughly 3,400 constraints, and below it the GPU loses badly:
//! at `tiny_mul` the whole batch is dispatch and submission cost, ten times what the
//! CPU needs to do the arithmetic. That is a real result and the reason
//! [`MetalMsm::msm_batch`] exists in the shape it does rather than as five calls.
//!
//! Two caveats on the speedups, both structural. Stages 5-9 are 76 to 82% of CPU proving
//! time, so 4.2x here is not 4.2x on a proof; and the CPU side of the comparison is our
//! own Pippenger, which is itself 11 to 19% behind rapidsnark.

use std::collections::HashMap;
use std::sync::Mutex;

use ark_ff::{One, Zero};
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

use snarkrs_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use snarkrs_groth16::ProveError;

use crate::enc::{dispatch_1d, packed_buffer, set_params};
use crate::kernels::{FR_MSL, MSM_MSL, SEAL_MSL};
use crate::layout::{PackedG1Affine, PackedG2Affine, PackedScalar};

mod inputs;
mod jobs;
mod outputs;
mod pool;
mod submit;
mod window;
mod wire;

pub(crate) use inputs::{dense_entries_bytes, Lent};
use inputs::{scalar_bits, CLASSIFY_CHUNK};
pub use inputs::{G1Bases, G2Bases, ScalarBuf};
use jobs::validate_job_ranges;
pub use jobs::{Job, JobG1, JobG2, MsmResult};
use outputs::{Outputs, ReduceLayout};
use pool::Pool;
use submit::{barrier, Submission};
use window::{
    fold_len, legacy_accumulate, merge_wide_min_span, reduce_groups_for, scatter_split_rows,
    slice_len_for, DUMMY_ROWS, HOST_TAIL_MAX_BUCKETS, RECODE_BITS, REDUCE_TG, SCAN_TG,
    SPLIT_BUDGET,
};
pub use window::{window_size, window_size_for, Work};
use wire::MsmParams;
pub use wire::{PackedXyzzG1, PackedXyzzG2};

// Read by msm_cpu_tests.rs and msm_gpu_probes.rs through `super::*`.
#[cfg(test)]
use crate::enc::read_back;
#[cfg(test)]
use crate::layout::{PackedFq, PackedFq2};

fn err(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "metal",
        reason: reason.into(),
    }
}

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
const SHARE_MAX_SKIP: usize = 8;

/// Where the plan of each `(buffer, scalar_off, n)` job starts. Two jobs share a digit
/// pipeline when they read the same scalars over the same range, or when one's range is
/// a suffix of the other's that skips at most `1 / SHARE_MAX_SKIP` of it.
fn plan_starts(jobs: &[(usize, usize, usize)]) -> Vec<usize> {
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

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

struct Pipelines {
    zero_u32: ComputePipelineState,
    mont_to_std: ComputePipelineState,
    count: ComputePipelineState,
    scan: ComputePipelineState,
    scatter: ComputePipelineState,
    clear_g1: ComputePipelineState,
    clear_g2: ComputePipelineState,
    accumulate_g1: ComputePipelineState,
    accumulate_g2: ComputePipelineState,
    segmented_g1: ComputePipelineState,
    segmented_g2: ComputePipelineState,
    merge_g1: ComputePipelineState,
    merge_g2: ComputePipelineState,
    merge_wide_g1: ComputePipelineState,
    merge_wide_g2: ComputePipelineState,
    fold_g1: ComputePipelineState,
    fold_g2: ComputePipelineState,
    reduce_g1: ComputePipelineState,
    reduce_g2: ComputePipelineState,
    /// The reduce on the constant-work point operations, for plans with dummy rows;
    /// see `pt_add_constant` in msm.metal.
    reduce_constant_g1: ComputePipelineState,
    reduce_constant_g2: ComputePipelineState,
    ones_g1: ComputePipelineState,
    ones_g2: ComputePipelineState,
    ones_idx_g1: ComputePipelineState,
    ones_idx_g2: ComputePipelineState,
}

// ---------------------------------------------------------------------------
// The backend object
// ---------------------------------------------------------------------------

pub struct MetalMsm {
    device: Device,
    queue: crate::cb::Queue,
    pipelines: Pipelines,
    /// The completion token every command buffer here ends with.
    seal: crate::cb::Seal,
    pool: Pool,
}

impl MetalMsm {
    /// Picks the system default device and compiles every MSM kernel from source.
    ///
    /// This is the expensive call: runtime MSL compilation of this library measured
    /// around 60 ms during scouting, plus pipeline construction. It must happen once, in
    /// `prepare`, and never on a proving path. A prover that constructs this per proof
    /// is measuring the Metal compiler.
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default()
            .ok_or_else(|| err("no Metal device; this machine cannot run the metal backend"))?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        // One translation unit: the field prelude followed by this file. There is no
        // include path for a runtime-compiled source string, so the concatenation is the
        // include, and the prelude's header guard is what makes it safe.
        let source = format!("{FR_MSL}\n{MSM_MSL}\n{SEAL_MSL}\n");
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &opts)
            .map_err(|e| err(format!("MSL compilation failed: {e}")))?;

        let pso = |name: &str| -> Result<ComputePipelineState, ProveError> {
            let f = library
                .get_function(name, None)
                .map_err(|e| err(format!("kernel {name} not found: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| err(format!("pipeline {name}: {e}")))
        };

        let pipelines = Pipelines {
            zero_u32: pso("zero_u32")?,
            mont_to_std: pso("fr_mont_to_std")?,
            count: pso("msm_count")?,
            scan: pso("msm_scan")?,
            scatter: pso("msm_scatter")?,
            clear_g1: pso("msm_clear_g1")?,
            clear_g2: pso("msm_clear_g2")?,
            accumulate_g1: pso("msm_accumulate_g1")?,
            accumulate_g2: pso("msm_accumulate_g2")?,
            segmented_g1: pso("msm_segmented_g1")?,
            segmented_g2: pso("msm_segmented_g2")?,
            merge_g1: pso("msm_merge_g1")?,
            merge_g2: pso("msm_merge_g2")?,
            merge_wide_g1: pso("msm_merge_wide_g1")?,
            merge_wide_g2: pso("msm_merge_wide_g2")?,
            fold_g1: pso("msm_fold_g1")?,
            fold_g2: pso("msm_fold_g2")?,
            reduce_g1: pso("msm_reduce_g1")?,
            reduce_g2: pso("msm_reduce_g2")?,
            reduce_constant_g1: pso("msm_reduce_constant_g1")?,
            reduce_constant_g2: pso("msm_reduce_constant_g2")?,
            ones_g1: pso("msm_ones_g1")?,
            ones_g2: pso("msm_ones_g2")?,
            ones_idx_g1: pso("msm_ones_idx_g1")?,
            ones_idx_g2: pso("msm_ones_idx_g2")?,
        };
        let seal = crate::cb::Seal::new(&device, &library)?;

        let queue = crate::cb::Queue::new(&device);
        Ok(Self {
            pool: Pool {
                device: device.clone(),
                free: Mutex::new(Vec::new()),
            },
            device,
            queue,
            pipelines,
            seal,
        })
    }

    /// The reduce a plan dispatches: the constant-work kernel for a plan with dummy
    /// rows, the plain one otherwise.
    fn reduce_pso(&self, is_g2: bool, plan: &Plan<'_>) -> &ComputePipelineState {
        match (is_g2, plan.dummy_rows > 0) {
            (false, false) => &self.pipelines.reduce_g1,
            (false, true) => &self.pipelines.reduce_constant_g1,
            (true, false) => &self.pipelines.reduce_g2,
            (true, true) => &self.pipelines.reduce_constant_g2,
        }
    }

    /// Threads per threadgroup the reduce actually dispatches with. [`Outputs::combine`]
    /// recomputes the G1 kernel's simdgroup slots from it, so the two must agree.
    fn reduce_tg(&self, is_g2: bool, plan: &Plan<'_>) -> usize {
        REDUCE_TG
            .min(
                self.reduce_pso(is_g2, plan)
                    .max_total_threads_per_threadgroup() as usize,
            )
            .max(1)
    }

    fn reduce_layout(&self, plan: &Plan<'_>) -> ReduceLayout {
        ReduceLayout::new(
            plan.n_buckets,
            plan.reduce_groups,
            self.reduce_tg(false, plan),
            self.reduce_pso(false, plan).thread_execution_width() as usize,
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn upload_g1_bases(&self, bases: &[G1Affine]) -> Result<G1Bases, ProveError> {
        Ok(G1Bases {
            buf: packed_buffer(&self.device, bases, PackedG1Affine::pack_into)?,
            len: bases.len(),
            inf: bases.iter().map(|b| b.infinity).collect(),
        })
    }

    pub fn upload_g2_bases(&self, bases: &[G2Affine]) -> Result<G2Bases, ProveError> {
        Ok(G2Bases {
            buf: packed_buffer(&self.device, bases, PackedG2Affine::pack_into)?,
            len: bases.len(),
            inf: bases.iter().map(|b| b.infinity).collect(),
        })
    }

    /// Packs scalars into standard form and uploads them, classifying as it goes.
    ///
    /// One pass, and the classification it produces is what keeps the zero and one
    /// scalars out of Pippenger entirely.
    ///
    /// This is per-proof work on the MSM stage's critical path, and `from_fr` is a full
    /// Montgomery reduction per scalar, so the pass is chunked over the thread pool and
    /// writes straight into the Metal buffer rather than through an intermediate `Vec`.
    /// Each chunk counts its own generals and builds its stretch of the prefix locally;
    /// a serial fix-up then shifts every stretch by the chunks before it, which is one
    /// add per scalar against the reduction the parallel pass just paid.
    pub fn upload_scalars(&self, scalars: &[Fr]) -> Result<ScalarBuf, ProveError> {
        self.upload_scalars_with(scalars, Work::Variable)
    }

    /// [`Self::upload_scalars`] with the work mode chosen. [`Work::Constant`] packs and
    /// nothing else: no prefix, no ones list, no bit bound, so nothing on the host
    /// depends on a scalar's value and the plan built over the buffer cannot either.
    pub fn upload_scalars_with(&self, scalars: &[Fr], work: Work) -> Result<ScalarBuf, ProveError> {
        use rayon::prelude::*;

        let n = scalars.len();
        let bytes = n.max(1) * core::mem::size_of::<PackedScalar>();
        let buf = crate::alloc::shared(&self.device, bytes)?;
        if n > 0 && work == Work::Constant {
            // SAFETY: as below.
            let slots: &mut [PackedScalar] =
                unsafe { core::slice::from_raw_parts_mut(buf.contents().cast(), n) };
            slots
                .par_chunks_mut(CLASSIFY_CHUNK)
                .zip_eq(scalars.par_chunks(CLASSIFY_CHUNK))
                .for_each(|(dst, src)| {
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = PackedScalar::from_fr(s);
                    }
                });
        }
        if work == Work::Constant {
            return Ok(ScalarBuf {
                buf,
                len: n,
                general_prefix: None,
                ones_idx: None,
                chunk_bits: None,
                work,
                lent: None,
            });
        }
        let mut prefix = vec![0u32; n + 1];
        let mut ones_idx = Vec::new();
        let mut chunk_bits = Vec::new();
        if n > 0 {
            // SAFETY: the buffer was just allocated with room for `n` packed scalars
            // and nothing has been encoded against it, so no dispatch can be reading it.
            let slots: &mut [PackedScalar] =
                unsafe { core::slice::from_raw_parts_mut(buf.contents().cast(), n) };
            let parts: Vec<(u32, Vec<u32>, u32)> = slots
                .par_chunks_mut(CLASSIFY_CHUNK)
                .zip_eq(scalars.par_chunks(CLASSIFY_CHUNK))
                .zip_eq(prefix[1..].par_chunks_mut(CLASSIFY_CHUNK))
                .enumerate()
                .map(|(ci, ((dst, src), pre))| {
                    let mut general = 0u32;
                    let mut ones = Vec::new();
                    let mut bits = 0u32;
                    for (i, ((d, s), p)) in dst.iter_mut().zip(src).zip(pre.iter_mut()).enumerate()
                    {
                        if s.is_one() {
                            ones.push((ci * CLASSIFY_CHUNK + i) as u32);
                        } else if !s.is_zero() {
                            general += 1;
                        }
                        *p = general;
                        *d = PackedScalar::from_fr(s);
                        bits = bits.max(scalar_bits(&d.v));
                    }
                    (general, ones, bits)
                })
                .collect();
            let mut off = 0u32;
            for (chunk, (total, ones, bits)) in prefix[1..].chunks_mut(CLASSIFY_CHUNK).zip(&parts) {
                if off > 0 {
                    for p in chunk {
                        *p += off;
                    }
                }
                off += total;
                ones_idx.extend_from_slice(ones);
                chunk_bits.push(*bits);
            }
        }
        Ok(ScalarBuf {
            buf,
            len: n,
            general_prefix: Some(prefix),
            ones_idx: Some(ones_idx),
            chunk_bits: Some(chunk_bits),
            work,
            lent: None,
        })
    }

    /// Converts a device-resident Montgomery `Fr` buffer (what the NTT leaves behind for
    /// stage 9) into standard-form scalars, in its own command buffer.
    ///
    /// No host classification is possible without reading the values back, so this
    /// buffer reports every scalar as general. That is exactly right for `H`, whose
    /// evaluations are dense, and it is why this is a separate entry point rather than
    /// the default.
    pub fn scalars_from_device_mont(
        &self,
        mont: &Buffer,
        len: usize,
    ) -> Result<ScalarBuf, ProveError> {
        let out = crate::alloc::shared(&self.device, len.max(1) * 32)?;
        // Reads `mont`, which nothing here writes, so a killed attempt re-runs whole.
        crate::cb::with_retry(&self.queue, |queue| {
            let context = "stage 9 scalar conversion (mont_to_std)";
            let cb = crate::cb::command_buffer(queue);
            let enc = cb.new_compute_command_encoder();
            enc.set_label(context);
            enc.set_compute_pipeline_state(&self.pipelines.mont_to_std);
            enc.set_buffer(0, Some(mont), 0);
            enc.set_buffer(1, Some(&out), 0);
            let n = len as u32;
            enc.set_bytes(2, 4, (&n as *const u32).cast());
            dispatch_1d(enc, &self.pipelines.mont_to_std, len, 64);
            let token = self.seal.encode(enc);
            enc.end_encoding();
            cb.commit();
            self.seal.wait(cb, token, context)
        })?;
        Ok(ScalarBuf {
            buf: out,
            len,
            general_prefix: None,
            ones_idx: None,
            chunk_bits: None,
            work: Work::Variable,
            lent: None,
        })
    }

    /// Wraps a device-resident buffer that already holds **standard-form** scalars
    /// (`layout::PackedScalar`), without dispatching anything.
    ///
    /// Stage 4 writes H in both forms precisely so stage 9 can read the standard copy
    /// directly; going through [`Self::scalars_from_device_mont`] instead re-runs
    /// `fr_mont_to_std` over the whole domain in an extra command buffer, duplicating
    /// work the pointwise kernel already did.
    ///
    /// The clone retains the `MTLBuffer`, but the *contents* stay owned by whoever
    /// allocated them (for H, the pooled stage scratch). The caller must keep the
    /// producing handle alive until every MSM reading this buffer has completed, or a
    /// recycled scratch could overwrite the scalars mid-flight. The proving path does:
    /// `MetalCircuit::msms` borrows the `HPoly` for its whole duration.
    ///
    /// Like the Montgomery entry point, no host classification is possible without a
    /// readback, so every scalar reports as general. Right for H, whose evaluations are
    /// dense.
    pub fn scalars_from_device_std(&self, std: &Buffer, len: usize) -> ScalarBuf {
        self.scalars_from_device_std_with(std, len, Work::Variable)
    }

    /// [`Self::scalars_from_device_std`] with the work mode chosen. Unclassified either
    /// way; what [`Work::Constant`] changes for H is that its kernels stop skipping the
    /// zero scalars and zero digits.
    pub fn scalars_from_device_std_with(&self, std: &Buffer, len: usize, work: Work) -> ScalarBuf {
        ScalarBuf {
            buf: std.clone(),
            len,
            general_prefix: None,
            ones_idx: None,
            chunk_bits: None,
            work,
            lent: None,
        }
    }

    /// One MSM on its own command buffer. Convenience for tests; the proving path should
    /// call [`Self::msm_batch`], because five separate command buffers cost five times
    /// the 0.16 ms submission floor.
    ///
    /// A length mismatch is an error, not a shorter MSM: `snarkrs_msm::MsmBackend` requires a
    /// backend to treat it as a malformed key, because the prefix is a valid-looking point
    /// for the wrong proof.
    pub fn msm_g1(&self, bases: &G1Bases, scalars: &ScalarBuf) -> Result<G1Projective, ProveError> {
        if bases.len != scalars.len {
            return Err(err(format!(
                "msm over G1: {} bases against {} scalars",
                bases.len, scalars.len
            )));
        }
        let n = bases.len;
        let out = self.msm_batch(&[Job::G1(JobG1 {
            bases,
            base_off: 0,
            scalars,
            scalar_off: 0,
            n,
        })])?;
        out[0].g1()
    }

    pub fn msm_g2(&self, bases: &G2Bases, scalars: &ScalarBuf) -> Result<G2Projective, ProveError> {
        if bases.len != scalars.len {
            return Err(err(format!(
                "msm over G2: {} bases against {} scalars",
                bases.len, scalars.len
            )));
        }
        let n = bases.len;
        let out = self.msm_batch(&[Job::G2(JobG2 {
            bases,
            base_off: 0,
            scalars,
            scalar_off: 0,
            n,
        })])?;
        out[0].g2()
    }

    /// Every MSM in one command buffer, one `wait_until_completed`, one readback.
    ///
    /// Jobs that share a scalar buffer, offset and length share their digit pipeline:
    /// the A, B-in-G2 and B-in-G1 MSMs all run over the whole witness, so the counting
    /// sort runs once for the three of them and only the point stages are repeated. L
    /// reads the witness past its public prefix and rides the same plan (see
    /// [`SHARE_MAX_SKIP`]). A plan costs four dispatches (`zero`, `count`, `scan`,
    /// `scatter`), so collapsing the five proving jobs to two plans removes twelve of
    /// the twenty and, more to the point, three of the four witness scatters' memory
    /// traffic. A bucket array past `SCATTER_SPLIT_ROWS` splits the scatter in two,
    /// which scales both counts.
    pub fn msm_batch<'a>(&self, jobs: &[Job<'a>]) -> Result<Vec<MsmResult>, ProveError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        // ---- plan ----
        let mut plans: Vec<Plan<'a>> = Vec::new();
        let mut by_key: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut job_plan = Vec::with_capacity(jobs.len());
        let mut job_skip = Vec::with_capacity(jobs.len());

        let mut ranges = Vec::with_capacity(jobs.len());
        for job in jobs {
            let (sbuf, soff, n, bases_len, boff) = match job {
                Job::G1(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
                Job::G2(j) => (j.scalars, j.scalar_off, j.n, j.bases.len, j.base_off),
            };
            validate_job_ranges(soff, n, sbuf.len, boff, bases_len)?;
            // Identity of the uploaded vector, not of its contents.
            ranges.push((sbuf, sbuf as *const ScalarBuf as usize, soff, n));
        }
        let keys: Vec<_> = ranges
            .iter()
            .map(|&(_, id, soff, n)| (id, soff, n))
            .collect();
        for (&(sbuf, id, soff, n), start) in ranges.iter().zip(plan_starts(&keys)) {
            let key = (id, start, soff + n - start);
            let idx = match by_key.get(&key) {
                Some(&i) => i,
                None => {
                    let i = plans.len();
                    plans.push(Plan::new(sbuf, start, soff + n - start));
                    by_key.insert(key, i);
                    i
                }
            };
            job_plan.push(idx);
            job_skip.push(soff - start);
        }

        // ---- allocate ----
        let mut scratch: Vec<Buffer> = Vec::new();
        for p in &mut plans {
            p.alloc(&self.pool, &mut scratch)?;
        }
        let mut outs: Vec<Outputs> = Vec::with_capacity(jobs.len());
        for job in jobs {
            let i = outs.len();
            outs.push(Outputs::alloc(
                self,
                &mut scratch,
                job,
                &plans[job_plan[i]],
                job_skip[i],
            )?);
        }

        // ---- combine ----
        //
        // Independent per job, and each one is a serial Horner plus a partial sum, so
        // the five jobs split over the thread pool rather than queueing behind B_g2's
        // G2 arithmetic. A result off the curve is a device fault, not an answer: the
        // batch that produced it is re-run whole by the retry below, as a killed one is.
        let combine = || -> Result<Vec<MsmResult>, ProveError> {
            use rayon::prelude::*;
            let results: Vec<MsmResult> = outs
                .par_iter()
                .enumerate()
                .map(|(i, out)| out.combine(self, &plans[job_plan[i]]))
                .collect();
            if let Some(i) = results.iter().position(|r| !r.on_curve()) {
                return Err(ProveError::Device {
                    backend: "metal",
                    reason: format!(
                        "MSM batch: job {i}'s sum is not a point on the curve: a partial \
                         sum read back is not one the GPU wrote"
                    ),
                });
            }
            Ok(results)
        };

        // ---- encode, once ----
        //
        // `G16_METAL_MSM_PHASES=1` splits the single command buffer into one per digit
        // pipeline and one per point pipeline, waiting on each, and prints wall times to
        // stderr. Strictly a measurement aid: it adds one ~0.15 ms submission floor per
        // piece, so the sum reads slightly worse than the production path it explains.
        let results = if std::env::var_os("G16_METAL_MSM_PHASES").is_some() {
            let detail = std::env::var("G16_METAL_MSM_PHASES")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(1);
            let queue = self.queue.get();
            let run = |label: String,
                       f: &mut dyn FnMut(&ComputeCommandEncoderRef)|
             -> Result<(), ProveError> {
                let t = std::time::Instant::now();
                let cb = crate::cb::command_buffer(&queue);
                let enc = cb.new_compute_command_encoder();
                enc.set_label(&label);
                f(enc);
                let token = self.seal.encode(enc);
                enc.end_encoding();
                cb.commit();
                // The production path below checks status and token; this one must too.
                // A timing read off a faulted command buffer is not a slow result, it is
                // a measurement of how long the GPU took to fail, reported as if it were
                // work. This path exists to explain timings, so a wrong one is worse here
                // than anywhere.
                self.seal.wait(cb, token, "MSM phase")?;
                eprintln!(
                    "[msm-phase] {label}: {:.2} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                Ok(())
            };
            for (pi, p) in plans.iter().enumerate() {
                let shape = format!(
                    "plan{pi} n={} cap={} c={} w={}",
                    p.n, p.cap, p.c, p.n_windows
                );
                if detail >= 3 {
                    // Separate the contended atomics from the prefix scan's serial
                    // barrier chain. Every line pays its own submission floor.
                    run(format!("zero    {shape}"), &mut |enc| {
                        p.encode_zero(self, enc)
                    })?;
                    run(format!("count   {shape}"), &mut |enc| {
                        p.encode_count(self, enc)
                    })?;
                    run(format!("scan    {shape}"), &mut |enc| {
                        p.encode_scan(self, enc)
                    })?;
                    run(format!("scatter {shape}"), &mut |enc| {
                        p.encode_scatter(self, enc)
                    })?;
                } else {
                    run(format!("digits {shape}"), &mut |enc| p.encode(self, enc))?;
                }
            }
            // `G16_METAL_MSM_PHASES=2` goes one level finer and times the four bucket
            // stages one command buffer each. Same caveat, four more submission floors.
            let fine = detail >= 2;
            for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                let p = &plans[job_plan[i]];
                let kind = if out.is_g2 { "g2" } else { "g1" };
                if fine {
                    run(
                        format!("clear   job{i} {kind} rows={}", p.n_windows * p.n_buckets),
                        &mut |enc| out.encode_clear(self, enc, job, p),
                    )?;
                    run(
                        format!("accum   job{i} {kind} slices={}", p.n_windows * p.slices),
                        &mut |enc| out.encode_accumulate(self, enc, job, p),
                    )?;
                    run(
                        format!("merge   job{i} {kind} rows={}", p.n_windows * p.n_buckets),
                        &mut |enc| out.encode_merge(self, enc, job, p),
                    )?;
                    run(
                        format!("reduce  job{i} {kind} w={}", p.n_windows),
                        &mut |enc| out.encode_reduce(self, enc, job, p),
                    )?;
                } else {
                    run(
                        format!("buckets job{i} {kind} n={} cap={} c={}", p.n, p.cap, p.c),
                        &mut |enc| out.encode_buckets(self, enc, job, p),
                    )?;
                }
                run(
                    format!("ones    job{i} {kind} n={} groups={}", p.n, out.ones_groups),
                    &mut |enc| out.encode_ones(self, enc, job, p),
                )?;
            }
            combine()?
        } else {
            // A concurrent encoder, staged. The default serial encoder barriers every
            // dispatch against the previous one, so the 37 dispatches ran strictly one
            // at a time and the per-phase GPU times summed exactly to the batch total:
            // zero overlap. But most of these dispatches are independent, and the small
            // ones are latency chains that leave nearly the whole device idle: the four
            // witness MSMs' accumulations and ones scans are a few thousand threads
            // each, while H's accumulation is throughput-bound and can absorb them.
            //
            // Grouping by stage keeps the hazards trivial to state: every dispatch of
            // one phase is independent of every other dispatch of that phase (disjoint
            // outputs; shared inputs are read-only), and each phase reads only what
            // earlier phases wrote, so one full barrier between phases is both
            // necessary and sufficient. metal-rs 0.29 does not bind
            // memoryBarrierWithScope:, so it is called the way `cb` reads the
            // unbound timing properties.
            //
            // The ones scans have no dependency at all (scalars and bases in, own
            // buffer out) and are encoded into the accumulation phase, the widest one.
            //
            // A constant-work batch goes out split (see `Submission`): the same
            // phases, one command buffer each, and the accumulation in pieces.
            let split = plans.iter().any(|p| p.dummy_rows > 0);
            let encode = |sub: &mut Submission<'_>| -> Result<(), ProveError> {
                for p in &plans {
                    p.encode_zero(self, sub.enc());
                }
                for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                    out.encode_clear(self, sub.enc(), job, &plans[job_plan[i]]);
                }
                sub.phase()?;
                for p in &plans {
                    p.encode_count(self, sub.enc());
                }
                sub.phase()?;
                for p in &plans {
                    p.encode_scan(self, sub.enc());
                }
                sub.phase()?;
                // A split plan's second part must not overlap its first inside this
                // concurrent encoder, so it goes behind a barrier of its own. The witness
                // plans never split and ride entirely in the first wave.
                for p in &plans {
                    p.encode_scatter_part(self, sub.enc(), 0);
                }
                if plans.iter().any(|p| p.scatter_parts > 1) {
                    sub.phase()?;
                    for p in &plans {
                        p.encode_scatter_part(self, sub.enc(), 1);
                    }
                }
                sub.phase()?;
                for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                    let plan = &plans[job_plan[i]];
                    out.encode_ones(self, sub.enc(), job, plan);
                    if !split {
                        out.encode_accumulate(self, sub.enc(), job, plan);
                        continue;
                    }
                    // Whole windows per piece, as many as the budget holds, each in a
                    // command buffer of its own.
                    let per_window = plan.cap * if out.is_g2 { 3 } else { 1 };
                    let step = (SPLIT_BUDGET / per_window.max(1)).clamp(1, plan.n_windows);
                    for lo in (0..plan.n_windows).step_by(step) {
                        let hi = (lo + step).min(plan.n_windows);
                        out.encode_accumulate_slices(
                            self,
                            sub.enc(),
                            job,
                            plan,
                            lo * plan.slices,
                            (hi - lo) * plan.slices,
                        );
                        sub.close()?;
                    }
                }
                sub.phase()?;
                // One merge level of every job at a time, a barrier between levels. The
                // constant-work fold of a G2 job is 3.9 ms at 2^18, all levels, so they
                // share one command buffer even when the batch is split.
                for level in 0.. {
                    let mut more = false;
                    for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                        more |= out.encode_merge_level(
                            self,
                            sub.enc(),
                            job,
                            &plans[job_plan[i]],
                            level,
                        );
                    }
                    if !more {
                        break;
                    }
                    barrier(sub.enc());
                }
                sub.phase()?;
                for (i, (job, out)) in jobs.iter().zip(&outs).enumerate() {
                    out.encode_reduce(self, sub.enc(), job, &plans[job_plan[i]]);
                }
                sub.wait("MSM batch")
            };
            // Before this check the combine reinterpreted pooled buffers regardless of
            // whether the GPU had actually written them, so a fault returned the
            // previous proof's window sums with an Ok. Split or not, the whole batch is
            // re-encoded from its zeroing dispatches, so a retry reads nothing
            // half-written.
            crate::cb::with_retry(&self.queue, |queue| {
                encode(&mut Submission::new(self, queue, split))?;
                combine()
            })?
        };

        // Buffers go home. Anything the plan and the outputs still reference is dead
        // now: `wait_until_completed` returned, so the GPU is finished with all of it.
        drop(plans);
        drop(outs);
        self.pool.give(scratch);

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// A digit pipeline: everything that depends only on the scalars.
// ---------------------------------------------------------------------------

struct Plan<'a> {
    n: usize,
    scalar_off: usize,
    c: u32,
    n_windows: usize,
    n_buckets: usize,
    cap: usize,
    slice_len: usize,
    slices: usize,
    reduce_groups: usize,
    /// Top-window rows the wide merge owns; 0 keeps everything on `msm_merge_*`.
    merge_wide_rows: usize,
    /// Serial window-range dispatches of the scatter; 2 caps the live tail-line set
    /// on row-heavy plans (see [`SCATTER_SPLIT_ROWS`]), 1 everywhere else.
    scatter_parts: usize,
    host_tail: bool,
    /// Classified inputs have an exact ones gather. Unclassified inputs instead
    /// recode 1 into bucket zero, so an H plan needs no separate full-buffer scan.
    separate_ones: bool,
    /// [`DUMMY_ROWS`] under [`Work::Constant`], 0 otherwise. The dummy rows follow the
    /// real bucket rows in `counts`, `cursor` and the bucket array, and only the digit
    /// pipeline and the accumulation ever address them.
    dummy_rows: usize,
    scalars: &'a Buffer,
    /// Where the entries go if it is big enough; the pool otherwise.
    lent: Option<&'a Buffer>,
    counts: Option<Buffer>,
    cursor: Option<Buffer>,
    entries: Option<Buffer>,
}

impl<'a> Plan<'a> {
    fn new(scalars: &'a ScalarBuf, scalar_off: usize, n: usize) -> Self {
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
        let slice_len = slice_len_for(n_windows, cap);
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
    fn fold_levels(&self) -> Vec<(usize, usize)> {
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
    fn spill_per_slice(&self) -> usize {
        if self.dummy_rows > 0 {
            2
        } else {
            1
        }
    }

    /// Rows in `counts`, `cursor` and the bucket array: the real buckets of every window,
    /// then every window's dummy rows.
    fn rows(&self) -> usize {
        self.n_windows * (self.n_buckets + self.dummy_rows)
    }

    fn params(&self) -> MsmParams {
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

    fn alloc(&mut self, pool: &Pool, keep: &mut Vec<Buffer>) -> Result<(), ProveError> {
        let rows = self.rows();
        let counts = pool.take(rows * 4)?;
        let cursor = pool.take(rows * 4)?;
        // 8 bytes an entry: the bucket row travels with the point index so the
        // segmented accumulation can find run boundaries without recomputing digits.
        let bytes = self.n_windows * self.cap * 8;
        let entries = match self.lent.filter(|b| b.length() as usize >= bytes) {
            // Not kept: it goes back to its owner, not into the pool.
            Some(b) => b.clone(),
            None => {
                let e = pool.take(bytes)?;
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

    fn encode(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        self.encode_zero(msm, enc);
        self.encode_count(msm, enc);
        self.encode_scan(msm, enc);
        self.encode_scatter(msm, enc);
    }

    /// The counters must start at zero, and unlike the bucket array (whose kernel
    /// writes rather than accumulates) they cannot rely on a fresh allocation, since
    /// the pool hands back used buffers.
    fn encode_zero(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let rows = self.rows();
        enc.set_compute_pipeline_state(&msm.pipelines.zero_u32);
        enc.set_buffer(
            0,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        let len = rows as u32;
        enc.set_bytes(1, 4, (&len as *const u32).cast());
        dispatch_1d(enc, &msm.pipelines.zero_u32, rows, 256);
    }

    fn encode_count(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.count);
        enc.set_buffer(0, Some(self.scalars), 0);
        enc.set_buffer(
            1,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 2, &p);
        dispatch_1d(enc, &msm.pipelines.count, self.n, 64);
    }

    fn encode_scan(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.scan);
        enc.set_buffer(
            0,
            Some(self.counts.as_ref().expect("plan not allocated")),
            0,
        );
        enc.set_buffer(
            1,
            Some(self.cursor.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 2, &p);
        let scan_tg = SCAN_TG.min(msm.pipelines.scan.max_total_threads_per_threadgroup() as usize);
        crate::cb::dispatch_thread_groups(
            enc,
            MTLSize::new(self.n_windows as u64, 1, 1),
            MTLSize::new(scan_tg as u64, 1, 1),
        );
    }

    /// The scatter, or the `part`-th of its window-range dispatches. The parts write
    /// disjoint rows, so no barrier is *required* between them; the caller inserts one
    /// anyway, because running them serially is the entire point of the split.
    fn encode_scatter_part(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef, part: usize) {
        if part >= self.scatter_parts {
            return;
        }
        let step = self.n_windows.div_ceil(self.scatter_parts);
        let lo = part * step;
        let hi = (lo + step).min(self.n_windows);
        let p = self.params();
        enc.set_compute_pipeline_state(&msm.pipelines.scatter);
        enc.set_buffer(0, Some(self.scalars), 0);
        enc.set_buffer(
            1,
            Some(self.cursor.as_ref().expect("plan not allocated")),
            0,
        );
        enc.set_buffer(
            2,
            Some(self.entries.as_ref().expect("plan not allocated")),
            0,
        );
        set_params(enc, 3, &p);
        let span = [lo as u32, hi as u32];
        enc.set_bytes(4, 8, span.as_ptr().cast());
        dispatch_1d(enc, &msm.pipelines.scatter, self.n, 64);
    }

    fn encode_scatter(&self, msm: &MetalMsm, enc: &ComputeCommandEncoderRef) {
        for part in 0..self.scatter_parts {
            self.encode_scatter_part(msm, enc, part);
        }
    }
}

#[cfg(test)]
#[path = "msm_cpu_tests.rs"]
mod cpu_tests;

#[cfg(test)]
#[path = "msm_gpu_probes.rs"]
mod gpu_probes;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod thread_safety {
    /// `PreparedCircuit` must be `Send + Sync`, so anything the Metal backend stores on
    /// one has to be too. Checked rather than assumed: `metal-rs` declares its handles
    /// `Send + Sync`, and MTLDevice, MTLBuffer, MTLComputePipelineState and
    /// MTLCommandQueue are documented thread-safe. Command buffers and encoders are not,
    /// and none of them outlive a `msm_batch` call.
    const fn assert_send_sync<T: Send + Sync>() {}
    const _: () = {
        assert_send_sync::<super::MetalMsm>();
        assert_send_sync::<super::G1Bases>();
        assert_send_sync::<super::G2Bases>();
        assert_send_sync::<super::ScalarBuf>();
    };
}
