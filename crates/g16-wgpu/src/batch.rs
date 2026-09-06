//! Stages 5 to 9 as **one submit**: every MSM in a proof, sharing what can be shared.
//!
//! [`crate::msm`] is the digit half of an MSM and [`crate::points`] is the curve half. This
//! is the file that makes a proof out of them: five MSMs, three counting sorts, one command
//! encoder, one submission and one readback.
//!
//! # Why one submit, and what the alternative costs
//!
//! `g16-metal`'s `msm.rs` puts all five MSMs in one command buffer because committing one and
//! waiting on it costs 0.16 ms and does not get cheaper with less in it, where an extra
//! dispatch into an already open encoder costs 2 to 3 microseconds. The same ratio holds here
//! and is if anything wider: [`crate::stages`] measured an empty submit plus its fence at
//! **22 to 24 microseconds in release** on this M2 Max, against 2 to 3 for a dispatch. A
//! proof's MSMs are 38 dispatches at the shapes the artifacts reach; five submissions would
//! be five fences and five `mapAsync` round trips at about 0.3 ms each, which is 1.6 ms of
//! pure latency on a stage that is trying to get under 100.
//!
//! So: one [`wgpu::CommandEncoder`], one compute pass, every dispatch of every MSM in it,
//! then five `copy_buffer_to_buffer` calls into five windows of a **single** staging buffer,
//! then one submit and one map. Design §3 caps the whole per-proof readback at 64 KiB and
//! [`MsmBatch::readback_bytes`] reports what a given proof actually asks for, so the claim is
//! checkable rather than asserted.
//!
//! # Why the caller groups the jobs instead of this file deducing the grouping
//!
//! Everything in [`crate::msm`] is a function of `(scalar buffer, range)` and nothing else,
//! so two MSMs over the same scalars can share one counting sort. In a Groth16 proof the A,
//! B-in-G2 and B-in-G1 MSMs all run over the whole witness, so three of the five share one
//! sort: 3 digit pipelines and 12 dispatches per proof instead of 5 and 20, and two thirds
//! less scatter traffic.
//!
//! That sharing is expressed by the caller putting three [`Job`]s into one [`Group`], not by
//! this file comparing buffer handles. wgpu offers no buffer identity comparison, so the
//! deduction would have to be pointer equality on `&wgpu::Buffer`, which is true for the
//! shape the prover happens to build today and silently false the first time someone passes
//! two references to the same buffer through different paths. A wrong answer there is not an
//! error: it is one extra sort, or worse, one MSM reading another's entry array. Making the
//! caller say it costs one line in `crate::backend` and makes the mistake unrepresentable.
//!
//! # The witness reaches the device once, in Montgomery form
//!
//! Stage 0 already uploaded the witness as `PackedFr` ([`crate::stages`]), and every digit
//! kernel reads standard form. So [`MsmBatch::run`] takes an optional [`MontConvert`] and
//! heads its encoder with one `fr_mont_to_std` dispatch rather than making the host pack
//! 140,824 scalars a second time and upload another 4.5 MB. `g16-metal` does the same
//! conversion for `H` alone and pays a whole extra command buffer for it, which its own
//! `audit_metal_vs_cpu.rs` measures; here it is one more dispatch in an encoder that is open
//! anyway.
//!
//! [`Source::Host`] exists for the case where there is no device witness to convert, which is
//! a CPU `compute_h` finished on the GPU. It is not the proving path and it says so.
//!
//! # Scratch is pooled on the plan, not on the size
//!
//! A pooled buffer is reused only when the [`DigitPlan`] or `(DigitPlan, PointPlan)` it was
//! allocated for is *equal* to the one being run, not merely large enough. The window width
//! `c` is chosen from the count of scalars that are neither 0 nor 1, so it is a property of
//! the witness and not of the circuit, and two witnesses of one circuit can want different
//! shapes. Reusing on "big enough" would mean `PointBuffers::ones_offset` came from the plan
//! that allocated the buffer while `PointPlan::ones_groups` came from the plan being run, and
//! the two disagreeing is a readback that decodes the wrong bytes with no error anywhere. In
//! steady state the plans are equal on every proof of one circuit and the pool never
//! allocates.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use g16_core::ProveError;
use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective, Zero};
use g16_gpu_layout::{as_bytes, Packed, PackedG1Affine, PackedG2Affine, LIMBS};

use crate::device::{bad, WgpuBackend};
use crate::gen::points as wgsl;
use crate::msm::{pack_scalars, storage_buffer, DigitBuffers, DigitPlan, MsmDigits};
use crate::params::ParamRing;
use crate::points::{MsmPointsG1, MsmPointsG2, PointBuffers, PointPlan};
use crate::readback::Readback;

/// Bytes one scalar occupies on the device, in either encoding.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// Every job's slice of the readback starts on a multiple of this.
///
/// `copy_buffer_to_buffer` only needs 4, and 256 is free: a G1 window sum is 128 bytes and a
/// G2 one is 256, and `PointBuffers` already rounds its own internal `ones` offset to 256 for
/// the storage-binding alignment. Keeping the whole readback on the same grid means a dump of
/// it lines up with the buffers it came from.
const SLICE_ALIGN: u64 = 256;

// ---------------------------------------------------------------------------
// Resident base vectors
// ---------------------------------------------------------------------------

fn upload_packed<T: Packed>(
    backend: &WgpuBackend,
    label: &str,
    packed: &[T],
) -> Result<wgpu::Buffer, ProveError> {
    // `max(1)`: a zero length WebGPU buffer is not a thing, and an empty query section is
    // legal (a circuit whose private witness is empty has no L bases). One zeroed element is
    // the point at infinity in this wire format, so the pad is also the right value.
    let bytes = (packed.len().max(1) * size_of::<T>()) as u64;
    let buf = storage_buffer(backend, label, bytes)?;
    if !packed.is_empty() {
        backend.queue().write_buffer(&buf, 0, as_bytes(packed));
    }
    Ok(buf)
}

/// One G1 base vector, resident on the device for the life of the key.
///
/// This is the "warm versus cold" cost of stages 5 to 9 and it is the larger half of a
/// `prepare`: `js_16x16_d32` has 140,824 A bases, 140,824 B-G1 bases, 140,823 L bases and
/// 262,144 H bases, 43 MB of G1 at 64 bytes a point, plus 18 MB of G2. A prover that uploaded
/// them per proof would measure its own key upload.
///
/// Repacking, not casting: `ark_ec::G1Affine` is 72 bytes on this arkworks, is not `repr(C)`,
/// and carries an infinity flag that [`PackedG1Affine`] encodes as all-zero coordinates
/// instead. The A query really does contain points at infinity, 1,187 of `js_16x16_d32`'s
/// 140,824, so this is a live case and not a theoretical one.
pub struct G1Bases {
    buf: wgpu::Buffer,
    len: usize,
}

impl G1Bases {
    pub fn upload(backend: &WgpuBackend, ps: &[G1Affine]) -> Result<Self, ProveError> {
        Ok(Self {
            buf: upload_packed(backend, "g16 msm g1 bases", &PackedG1Affine::pack_slice(ps))?,
            len: ps.len(),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Device bytes, for a report.
    pub fn bytes(&self) -> u64 {
        self.buf.size()
    }
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

/// One G2 base vector. 128 bytes a point, so the B-G2 query is twice the bytes of a G1 one of
/// the same length and its MSM is 3.05x the arithmetic. See [`G1Bases`].
pub struct G2Bases {
    buf: wgpu::Buffer,
    len: usize,
}

impl G2Bases {
    pub fn upload(backend: &WgpuBackend, ps: &[G2Affine]) -> Result<Self, ProveError> {
        Ok(Self {
            buf: upload_packed(backend, "g16 msm g2 bases", &PackedG2Affine::pack_slice(ps))?,
            len: ps.len(),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> u64 {
        self.buf.size()
    }
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

// ---------------------------------------------------------------------------
// What a caller asks for
// ---------------------------------------------------------------------------

/// One MSM: which base vector, and where in it this job starts.
///
/// The scalars are not here. They are on the [`Group`], because the grouping *is* the
/// statement that several jobs share one counting sort.
pub enum Job<'a> {
    G1 { bases: &'a G1Bases, base_off: u32 },
    G2 { bases: &'a G2Bases, base_off: u32 },
}

impl Job<'_> {
    fn bases(&self) -> &wgpu::Buffer {
        match self {
            Job::G1 { bases, .. } => &bases.buf,
            Job::G2 { bases, .. } => &bases.buf,
        }
    }
    fn base_off(&self) -> u32 {
        match self {
            Job::G1 { base_off, .. } | Job::G2 { base_off, .. } => *base_off,
        }
    }
    fn curve(&self) -> wgsl::Curve {
        match self {
            Job::G1 { .. } => wgsl::G1,
            Job::G2 { .. } => wgsl::G2,
        }
    }
    fn base_len(&self) -> usize {
        match self {
            Job::G1 { bases, .. } => bases.len,
            Job::G2 { bases, .. } => bases.len,
        }
    }
}

/// Where a group's scalars come from.
pub enum Source<'a> {
    /// Standard-form scalars already on the device. The proving path: the witness
    /// [`crate::stages::WgpuHandle::witness_std`] and `H`
    /// [`crate::stages::WgpuHandle::h_std`].
    ///
    /// `general` is how many of the `n` scalars in range are neither 0 nor 1, and it sizes
    /// both the window width and `cap`, the bound on the scatter's write into each window's
    /// slice of the entry array. **Overestimating is safe and underestimating is not**:
    /// WebGPU drops an out-of-range storage write in silence, so a `cap` below the true count
    /// loses entries with no error anywhere. `None` therefore means `n`, not a guess, and is
    /// the only honest answer for a buffer the host has never seen. That is the case for `H`,
    /// whose values exist only on the device.
    Device {
        buf: &'a wgpu::Buffer,
        general: Option<u32>,
    },
    /// Host scalars, packed and uploaded here.
    ///
    /// **Not the proving path.** It exists so a CPU `compute_h` can be finished on the GPU,
    /// which is what makes "the MSM half is right" a statement a test can make without the
    /// NTT half being right too. It costs one Montgomery reduction per scalar on the host
    /// plus the upload, and in a browser rayon falls back to a sequential pool, so 140,000 of
    /// them run on one thread.
    Host(&'a [Fr]),
}

/// Several MSMs over one scalar range, therefore over one counting sort.
pub struct Group<'a> {
    pub scalars: Source<'a>,
    /// Element offset into the scalar buffer. `n_public + 1` for the L query, which covers
    /// the private wires only; zero for everything else.
    pub scalar_off: u32,
    /// Scalars in range, which is also the base count each job reads.
    pub n: u32,
    pub jobs: &'a [Job<'a>],
}

/// One `fr_mont_to_std` dispatch, run before anything else in the encoder.
pub struct MontConvert<'a> {
    pub src: &'a wgpu::Buffer,
    pub dst: &'a wgpu::Buffer,
    pub n: u32,
}

/// One MSM's answer, in the group the job named.
#[derive(Clone, Copy, Debug)]
pub enum MsmResult {
    G1(G1Projective),
    G2(G2Projective),
}

impl MsmResult {
    /// The G1 point, or an error naming the mismatch rather than a panic. A caller that
    /// asked for the wrong one has its job list out of step with how it reads the results,
    /// which is exactly the bug that produces a proof that does not verify and says nothing.
    pub fn g1(&self) -> Result<G1Projective, ProveError> {
        match self {
            MsmResult::G1(p) => Ok(*p),
            MsmResult::G2(_) => Err(bad("asked a G2 MSM result for a G1 point")),
        }
    }
    pub fn g2(&self) -> Result<G2Projective, ProveError> {
        match self {
            MsmResult::G2(p) => Ok(*p),
            MsmResult::G1(_) => Err(bad("asked a G1 MSM result for a G2 point")),
        }
    }
}

// ---------------------------------------------------------------------------
// Pooled scratch
// ---------------------------------------------------------------------------

/// One counting sort's buffers plus the plan they were allocated for. See the module docs for
/// why the plan is stored rather than the sizes.
struct DigitSlot {
    plan: DigitPlan,
    bufs: DigitBuffers,
}

/// One MSM's point buffers plus the pair of plans they were allocated for, and the curve,
/// because a G1 and a G2 job of the same shape need different byte counts and the plans alone
/// do not say which.
struct PointSlot {
    plan: (DigitPlan, PointPlan),
    curve: &'static str,
    bufs: PointBuffers,
}

/// One in-flight proof's stage 5 to 9 device memory.
///
/// Checked out of a pool for the duration of one [`MsmBatch::run`] and returned on the way
/// out, for the same reason [`crate::stages::HStages`] pools its own: `PreparedCircuit` takes
/// `&self` and two proofs must not be handed the same bucket array. Nothing here is reachable
/// from [`MsmBatch`] except through the mutex.
#[derive(Default)]
struct Scratch {
    digits: Vec<DigitSlot>,
    points: Vec<PointSlot>,
    /// Host scalars uploaded by [`Source::Host`], one per group that uses it. Kept in the
    /// pool because the fallback path is also the path the tests hammer.
    uploads: Vec<wgpu::Buffer>,
    ring: Option<ParamRing>,
    readback: Option<Readback>,
}

// ---------------------------------------------------------------------------
// The batch
// ---------------------------------------------------------------------------

/// The three pipeline sets a proof's MSMs need, and the scratch pool behind them.
///
/// Built once per process: the digit module and the two curve modules are three shader
/// modules and fifteen pipelines, and `crate::msm::ModuleShape`'s table measures Metal at
/// about 21 ms of pipeline creation per entry point cold.
pub struct MsmBatch {
    digits: MsmDigits,
    g1: MsmPointsG1,
    g2: MsmPointsG2,
    pool: Mutex<Vec<Scratch>>,
    /// Bytes the most recent [`Self::run`] copied down. See [`Self::last_readback_bytes`].
    last_readback: AtomicU64,
}

impl MsmBatch {
    /// Compiles all three modules at their measured shapes.
    pub fn new(backend: &WgpuBackend) -> Result<Self, ProveError> {
        Ok(Self {
            digits: MsmDigits::new(backend)?,
            g1: MsmPointsG1::new(backend)?,
            g2: MsmPointsG2::new(backend)?,
            pool: Mutex::new(Vec::new()),
            last_readback: AtomicU64::new(0),
        })
    }

    pub fn digits(&self) -> &MsmDigits {
        &self.digits
    }
    pub fn g1(&self) -> &MsmPointsG1 {
        &self.g1
    }
    pub fn g2(&self) -> &MsmPointsG2 {
        &self.g2
    }

    /// What every module cost to build, summed over the three.
    pub fn cost(&self) -> crate::pipelines::PrepareCost {
        let mut c = self.digits.cost();
        c.add(self.g1.cost());
        c.add(self.g2.cost());
        c
    }

    /// Bytes the most recent [`Self::run`] copied from the device.
    ///
    /// Design §3 caps the whole per-proof readback at 64 KiB, and that is a claim about this
    /// code rather than about the design, so it is reported and checked rather than asserted
    /// in a comment. It is what actually happened and not a prediction, which is the useful
    /// direction: a plan that got `ones_groups` wrong would show up here.
    ///
    /// One number for the whole batch, so it is only attributable while one proof at a time
    /// is running, which `crate::backend` arranges with `WgpuBackend::exclusive`.
    pub fn last_readback_bytes(&self) -> u64 {
        self.last_readback.load(Ordering::Relaxed)
    }

    /// Scratch sets sitting idle in the pool. Public so a test can assert that two concurrent
    /// proofs hold two different sets rather than racing on one, which is not observable from
    /// outside otherwise.
    pub fn pooled(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Every MSM in `groups`, in one submit, in job order.
    ///
    /// The returned vector is flat: group order, then job order within each group. A job with
    /// `n == 0` contributes the identity and dispatches nothing, which is what an empty L
    /// query needs.
    ///
    /// `mont`, if present, is one `fr_mont_to_std` dispatch encoded before everything else.
    /// The compute pass orders dispatches and makes each one's writes visible to the next, so
    /// the sorts that read `mont.dst` see the converted values with no explicit barrier.
    pub async fn run(
        &self,
        backend: &WgpuBackend,
        mont: Option<MontConvert<'_>>,
        groups: &[Group<'_>],
    ) -> Result<Vec<MsmResult>, ProveError> {
        // ---- plan, on the host, before anything is allocated or encoded ----
        let mut dplans: Vec<DigitPlan> = Vec::with_capacity(groups.len());
        // (group index, job index within the group, point plan).
        let mut pplans: Vec<(usize, usize, PointPlan)> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            let general = match &g.scalars {
                Source::Device { general, .. } => *general,
                // The packer walked every scalar to build the limbs, so the count is free
                // here and there is no reason to fall back to `n`.
                Source::Host(_) => None,
            };
            let dplan = DigitPlan::new(g.n.max(1), g.scalar_off, general)?;
            for (ji, job) in g.jobs.iter().enumerate() {
                if g.n == 0 {
                    continue;
                }
                if job.base_len() < (job.base_off() + g.n) as usize {
                    return Err(bad(format!(
                        "group {gi} job {ji} reads bases {}..{} of a {} point vector",
                        job.base_off(),
                        job.base_off() + g.n,
                        job.base_len()
                    )));
                }
                let pplan = match job {
                    Job::G1 { .. } => self.g1.plan_points(&dplan, job.base_off())?,
                    Job::G2 { .. } => self.g2.plan_points(&dplan, job.base_off())?,
                };
                pplans.push((gi, ji, pplan));
            }
            dplans.push(dplan);
        }

        // ---- check out and grow the scratch ----
        let mut sc = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();

        sc.digits.truncate(dplans.len());
        for (i, plan) in dplans.iter().enumerate() {
            let stale = sc.digits.get(i).map(|s| s.plan != *plan).unwrap_or(true);
            if stale {
                let bufs = DigitBuffers::new(backend, plan, 0)?;
                let slot = DigitSlot { plan: *plan, bufs };
                match sc.digits.get_mut(i) {
                    Some(s) => *s = slot,
                    None => sc.digits.push(slot),
                }
            }
        }

        sc.points.truncate(pplans.len());
        for (i, (gi, _, pplan)) in pplans.iter().enumerate() {
            let dplan = dplans[*gi];
            let curve = groups[*gi].jobs[pplans[i].1].curve();
            let key = (dplan, *pplan);
            let stale = sc
                .points
                .get(i)
                .map(|s| s.plan != key || s.curve != curve.suffix)
                .unwrap_or(true);
            if stale {
                let bufs = PointBuffers::new(backend, &dplan, pplan, 0, curve)?;
                let slot = PointSlot {
                    plan: key,
                    curve: curve.suffix,
                    bufs,
                };
                match sc.points.get_mut(i) {
                    Some(s) => *s = slot,
                    None => sc.points.push(slot),
                }
            }
        }

        // Host uploads, one buffer per `Source::Host` group, in group order.
        let mut upload_at: Vec<Option<usize>> = vec![None; groups.len()];
        let mut n_uploads = 0usize;
        for (gi, g) in groups.iter().enumerate() {
            let Source::Host(xs) = &g.scalars else {
                continue;
            };
            let want = (g.scalar_off as usize) + (g.n as usize);
            if xs.len() < want {
                return Err(bad(format!(
                    "group {gi} reads scalars {}..{want} of a {} element host vector",
                    g.scalar_off,
                    xs.len()
                )));
            }
            let bytes = (xs.len().max(1) as u64) * FR_BYTES;
            let stale = sc
                .uploads
                .get(n_uploads)
                .map(|b| b.size() < bytes)
                .unwrap_or(true);
            if stale {
                let buf = storage_buffer(backend, "g16 msm host scalars", bytes)?;
                match sc.uploads.get_mut(n_uploads) {
                    Some(b) => *b = buf,
                    None => sc.uploads.push(buf),
                }
            }
            let (words, _) = pack_scalars(xs);
            backend
                .queue()
                .write_buffer(&sc.uploads[n_uploads], 0, bytemuck::cast_slice(&words));
            upload_at[gi] = Some(n_uploads);
            n_uploads += 1;
        }
        sc.uploads.truncate(n_uploads);

        // ---- the parameter ring, sized before anything is pushed ----
        let mut slots = mont.as_ref().map_or(0, |m| self.digits.mont_slots(m.n));
        for (gi, plan) in dplans.iter().enumerate() {
            if groups[gi].n == 0 {
                continue;
            }
            slots += self.digits.sort_slots(plan);
        }
        for (i, (gi, _, pplan)) in pplans.iter().enumerate() {
            let dplan = dplans[*gi];
            slots += match groups[*gi].jobs[pplans[i].1] {
                Job::G1 { .. } => self.g1.slots(&dplan, pplan),
                Job::G2 { .. } => self.g2.slots(&dplan, pplan),
            };
        }
        let slots = slots.max(1);
        if sc.ring.as_ref().map(|r| r.slots() < slots).unwrap_or(true) {
            sc.ring = Some(ParamRing::new(backend, "g16 msm batch params", slots)?);
        }
        let ring = sc.ring.as_mut().expect("just built");
        ring.reset();

        let mont_off = match &mont {
            Some(m) => self.digits.plan_mont(m.n, ring)?,
            None => Vec::new(),
        };
        let mut sort_off = Vec::with_capacity(dplans.len());
        for (gi, plan) in dplans.iter().enumerate() {
            sort_off.push(if groups[gi].n == 0 {
                None
            } else {
                Some(self.digits.plan_sort(plan, ring)?)
            });
        }
        let mut point_off = Vec::with_capacity(pplans.len());
        for (i, (gi, _, pplan)) in pplans.iter().enumerate() {
            let dplan = dplans[*gi];
            point_off.push(match groups[*gi].jobs[pplans[i].1] {
                Job::G1 { .. } => self.g1.plan(&dplan, pplan, ring)?,
                Job::G2 { .. } => self.g2.plan(&dplan, pplan, ring)?,
            });
        }
        ring.flush(backend);

        // ---- the readback, one staging buffer for every job's window sums ----
        let mut slices: Vec<(u64, u64)> = Vec::with_capacity(pplans.len());
        let mut total = 0u64;
        for (i, (gi, _, _)) in pplans.iter().enumerate() {
            let curve = groups[*gi].jobs[pplans[i].1].curve();
            let bytes = sc.points[i].bufs.results_bytes(curve);
            slices.push((total, bytes));
            total += bytes.div_ceil(SLICE_ALIGN) * SLICE_ALIGN;
        }
        let total = total.max(4);
        if sc
            .readback
            .as_ref()
            .map(|r| r.capacity() < total)
            .unwrap_or(true)
        {
            sc.readback = Some(Readback::new(backend, "g16 msm batch results", total)?);
        }

        // ---- bind ----
        let ring = sc.ring.as_ref().expect("just built");
        let mut mont_bind = None;
        if let Some(m) = &mont {
            mont_bind = Some(self.digits.bind_mont(backend, ring, m.n, m.src, m.dst)?);
        }
        let scalar_buf = |gi: usize| -> &wgpu::Buffer {
            match &groups[gi].scalars {
                Source::Device { buf, .. } => buf,
                Source::Host(_) => &sc.uploads[upload_at[gi].expect("planned above")],
            }
        };
        let mut sort_binds = Vec::with_capacity(dplans.len());
        for (gi, plan) in dplans.iter().enumerate() {
            sort_binds.push(if groups[gi].n == 0 {
                None
            } else {
                Some(self.digits.bind_sort(
                    backend,
                    ring,
                    plan,
                    scalar_buf(gi),
                    &sc.digits[gi].bufs,
                )?)
            });
        }
        let mut point_binds = Vec::with_capacity(pplans.len());
        for (i, (gi, ji, pplan)) in pplans.iter().enumerate() {
            let dplan = dplans[*gi];
            let job = &groups[*gi].jobs[*ji];
            let sort = &sc.digits[*gi].bufs;
            let pts = &sc.points[i].bufs;
            point_binds.push(match job {
                Job::G1 { .. } => self.g1.bind_all(
                    backend,
                    ring,
                    &dplan,
                    pplan,
                    scalar_buf(*gi),
                    job.bases(),
                    sort,
                    pts,
                )?,
                Job::G2 { .. } => self.g2.bind_all(
                    backend,
                    ring,
                    &dplan,
                    pplan,
                    scalar_buf(*gi),
                    job.bases(),
                    sort,
                    pts,
                )?,
            });
        }

        // ---- encode: one encoder, one compute pass, every dispatch ----
        let mut enc = backend
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("g16 stages 5-9"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g16 stages 5-9"),
                timestamp_writes: None,
            });
            if let (Some(m), Some(bind)) = (&mont, &mont_bind) {
                self.digits.encode_mont(&mut pass, bind, m.n, &mont_off)?;
            }
            for (gi, plan) in dplans.iter().enumerate() {
                if let (Some(bind), Some(off)) = (&sort_binds[gi], &sort_off[gi]) {
                    self.digits.encode_sort(&mut pass, plan, bind, off)?;
                }
            }
            for (i, (gi, ji, pplan)) in pplans.iter().enumerate() {
                let dplan = dplans[*gi];
                match groups[*gi].jobs[*ji] {
                    Job::G1 { .. } => {
                        self.g1
                            .encode(&mut pass, &dplan, pplan, &point_binds[i], &point_off[i])?
                    }
                    Job::G2 { .. } => {
                        self.g2
                            .encode(&mut pass, &dplan, pplan, &point_binds[i], &point_off[i])?
                    }
                }
            }
        }
        let rb = sc.readback.as_ref().expect("just built");
        for (i, (at, bytes)) in slices.iter().enumerate() {
            rb.copy_from_at(&mut enc, &sc.points[i].bufs.results, 0, *at, *bytes)?;
        }

        // One submit and one map, for the whole of stages 5 to 9.
        self.last_readback.store(total, Ordering::Relaxed);
        let raw = rb.submit_and_read(backend, enc, total).await?;

        // ---- the host tail: Horner over each job's window sums ----
        //
        // `out` is indexed by the flat job number, `first_job[gi] + ji`, so a group whose
        // jobs were all skipped for `n == 0` still occupies its own slots and the caller's
        // job list and the result list stay in step. Getting that wrong is a proof that does
        // not verify with nothing else to go on, which is why it is a prefix sum and not a
        // `push` inside the loop that skips.
        let mut first_job = Vec::with_capacity(groups.len() + 1);
        let mut n_jobs = 0usize;
        for g in groups {
            first_job.push(n_jobs);
            n_jobs += g.jobs.len();
        }
        let mut out: Vec<Option<MsmResult>> = vec![None; n_jobs];
        for (i, (gi, ji, pplan)) in pplans.iter().enumerate() {
            let dplan = dplans[*gi];
            let (at, bytes) = slices[i];
            let window = &raw[at as usize..(at + bytes) as usize];
            let pts = &sc.points[i].bufs;
            out[first_job[*gi] + *ji] = Some(match groups[*gi].jobs[*ji] {
                Job::G1 { .. } => MsmResult::G1(self.g1.combine(window, &dplan, pplan, pts)?),
                Job::G2 { .. } => MsmResult::G2(self.g2.combine(window, &dplan, pplan, pts)?),
            });
            // The trace sees only the five final points. Under the knob, record what
            // `combine` just folded, so a wrong final point names a window instead of a run.
            // See `WINDOW_DEBUG` for the iPhone this is for. `i` is the flat job index, in
            // the deterministic order the caller listed the jobs, so the labels line up
            // between two machines.
            if crate::points::WINDOW_DEBUG.load(Ordering::Relaxed) != 0 {
                crate::points::window_log(&match groups[*gi].jobs[*ji] {
                    Job::G1 { .. } => {
                        self.g1
                            .debug_windows(&format!("j{i}_g1"), window, &dplan, pplan, pts)?
                    }
                    Job::G2 { .. } => {
                        self.g2
                            .debug_windows(&format!("j{i}_g2"), window, &dplan, pplan, pts)?
                    }
                });
            }
        }
        // Every job in a group with `n == 0` was skipped above, and its answer is the empty
        // sum. An empty L query is the real case: a circuit whose wires are all public.
        for (gi, g) in groups.iter().enumerate() {
            for (ji, job) in g.jobs.iter().enumerate() {
                let at = first_job[gi] + ji;
                if out[at].is_none() {
                    out[at] = Some(match job {
                        Job::G1 { .. } => MsmResult::G1(G1Projective::zero()),
                        Job::G2 { .. } => MsmResult::G2(G2Projective::zero()),
                    });
                }
            }
        }

        // The bind groups borrow the scratch, so they have to go before it is handed back.
        drop(point_binds);
        drop(sort_binds);
        drop(mont_bind);
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(sc);

        Ok(out.into_iter().map(|x| x.expect("filled above")).collect())
    }
}
