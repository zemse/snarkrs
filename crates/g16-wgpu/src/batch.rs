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
//! [`MsmBatch::last_readback_bytes`] reports what a given proof actually asks for, so the
//! claim is checkable rather than asserted.
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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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

/// Points per base buffer on this device: the largest power of two whose G2 chunk, at 128
/// bytes a point, fits one storage binding. 2^20 at the floor, 2^25 under `auto` here.
///
/// One size for both groups rather than twice as many G1 points per buffer, because the
/// sub-MSMs of a group are cut at every job's buffer boundaries (see [`MsmBatch::run`]) and
/// a G1 job sharing a sort with a G2 one would be cut at the G2 boundaries anyway.
pub fn base_chunk_elems(backend: &WgpuBackend) -> u32 {
    let limits = backend.granted_limits();
    let bytes = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let elems = bytes / wgsl::G2.base_bytes;
    u32::try_from(elems.max(1)).map_or(1u32 << 31, |e| {
        if e.is_power_of_two() {
            e
        } else {
            1 << (31 - e.leading_zeros())
        }
    })
}

/// One base vector in buffers of `chunk` points each, the last one shorter.
///
/// Several buffers and not one bound in ranges: a range binding needs a 256-byte aligned
/// offset, which a job's `base_off` does not promise, and it does nothing for a vector over
/// `maxBufferSize`, which the H query already is at a 2^22 domain (4M points at 64 bytes is
/// exactly 256 MiB, the floor). A vector that fits is one buffer, bound whole as before.
struct BaseChunks {
    bufs: Vec<wgpu::Buffer>,
    len: usize,
    chunk: u32,
}

impl BaseChunks {
    fn upload<T: Packed, P>(
        backend: &WgpuBackend,
        label: &str,
        ps: &[P],
        chunk: u32,
        pack: impl Fn(&[P]) -> Vec<T>,
    ) -> Result<Self, ProveError> {
        if chunk == 0 {
            return Err(bad(format!(
                "{label}: a base chunk of zero points holds nothing"
            )));
        }
        let mut bufs = Vec::with_capacity(ps.len().div_ceil(chunk as usize).max(1));
        // `max(1)`: a zero length WebGPU buffer is not a thing, and an empty query section
        // is legal (a circuit whose private witness is empty has no L bases). One zeroed
        // element is the point at infinity in this wire format, so the pad is also the right
        // value. Packed a chunk at a time, so the host never holds a second copy of the whole
        // vector: 431 MB of G2 at `js_384x384_d32`.
        if ps.is_empty() {
            bufs.push(storage_buffer(backend, label, size_of::<T>() as u64)?);
        }
        for part in ps.chunks(chunk as usize) {
            let packed = pack(part);
            let buf = storage_buffer(backend, label, as_bytes(&packed).len() as u64)?;
            backend.queue().write_buffer(&buf, 0, as_bytes(&packed));
            bufs.push(buf);
        }
        Ok(Self {
            bufs,
            len: ps.len(),
            chunk,
        })
    }

    fn bytes(&self) -> u64 {
        self.bufs.iter().map(|b| b.size()).sum()
    }
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
///
/// Held in buffers of [`base_chunk_elems`] points each, so a vector over the storage binding
/// limit (the H query at a 2^22 domain is 256 MiB) is several buffers and a job over it is
/// several sub-MSMs. A vector that fits is one buffer.
pub struct G1Bases {
    chunks: BaseChunks,
}

impl G1Bases {
    pub fn upload(backend: &WgpuBackend, ps: &[G1Affine]) -> Result<Self, ProveError> {
        Self::upload_chunked(backend, ps, base_chunk_elems(backend))
    }

    /// Same, at `chunk_elems` points per buffer. Public so a test can force the chunked
    /// path on a key that would otherwise fit in one.
    pub fn upload_chunked(
        backend: &WgpuBackend,
        ps: &[G1Affine],
        chunk_elems: u32,
    ) -> Result<Self, ProveError> {
        Ok(Self {
            chunks: BaseChunks::upload(
                backend,
                "g16 msm g1 bases",
                ps,
                chunk_elems,
                PackedG1Affine::pack_slice,
            )?,
        })
    }

    pub fn len(&self) -> usize {
        self.chunks.len
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.len == 0
    }
    /// Device bytes, for a report.
    pub fn bytes(&self) -> u64 {
        self.chunks.bytes()
    }
    /// Points per buffer.
    pub fn chunk_elems(&self) -> u32 {
        self.chunks.chunk
    }
    /// Buffers this vector occupies. One unless it is over the binding limit.
    pub fn chunks(&self) -> usize {
        self.chunks.bufs.len()
    }
    /// Buffer `i`, holding points `i * chunk_elems..`.
    pub fn chunk(&self, i: usize) -> &wgpu::Buffer {
        &self.chunks.bufs[i]
    }
}

/// One G2 base vector. 128 bytes a point, so the B-G2 query is twice the bytes of a G1 one of
/// the same length and its MSM is 3.05x the arithmetic. See [`G1Bases`].
pub struct G2Bases {
    chunks: BaseChunks,
}

impl G2Bases {
    pub fn upload(backend: &WgpuBackend, ps: &[G2Affine]) -> Result<Self, ProveError> {
        Self::upload_chunked(backend, ps, base_chunk_elems(backend))
    }

    pub fn upload_chunked(
        backend: &WgpuBackend,
        ps: &[G2Affine],
        chunk_elems: u32,
    ) -> Result<Self, ProveError> {
        Ok(Self {
            chunks: BaseChunks::upload(
                backend,
                "g16 msm g2 bases",
                ps,
                chunk_elems,
                PackedG2Affine::pack_slice,
            )?,
        })
    }

    pub fn len(&self) -> usize {
        self.chunks.len
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.len == 0
    }
    pub fn bytes(&self) -> u64 {
        self.chunks.bytes()
    }
    pub fn chunk_elems(&self) -> u32 {
        self.chunks.chunk
    }
    pub fn chunks(&self) -> usize {
        self.chunks.bufs.len()
    }
    pub fn chunk(&self, i: usize) -> &wgpu::Buffer {
        &self.chunks.bufs[i]
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
    fn chunks(&self) -> &BaseChunks {
        match self {
            Job::G1 { bases, .. } => &bases.chunks,
            Job::G2 { bases, .. } => &bases.chunks,
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
        self.chunks().len
    }
    /// Which base buffer holds this job's point for scalar `i`, and where in it.
    fn locate(&self, i: u32) -> (usize, u32) {
        let at = u64::from(self.base_off()) + u64::from(i);
        let chunk = u64::from(self.chunks().chunk);
        ((at / chunk) as usize, (at % chunk) as u32)
    }
    /// The first scalar index past `i` whose point is in a different base buffer.
    fn next_boundary(&self, i: u32) -> u32 {
        let at = u64::from(self.base_off()) + u64::from(i);
        let chunk = u64::from(self.chunks().chunk);
        let next = (at / chunk + 1) * chunk - u64::from(self.base_off());
        u32::try_from(next).unwrap_or(u32::MAX)
    }
}

/// One sub-MSM's scalar range: a slice of one group that fits every scratch buffer this
/// device can bind and lies inside one base buffer of every job. See [`MsmBatch::run`].
struct Piece {
    gi: usize,
    lo: u32,
    dplan: DigitPlan,
}

/// One job over one piece: the base buffer it reads and the point plan inside it.
struct PieceJob {
    pi: usize,
    ji: usize,
    chunk: usize,
    pplan: PointPlan,
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
    /// Sub-MSMs the most recent [`Self::run`] dispatched, G1 and G2. See
    /// [`Self::last_sub_msms`].
    last_sub_msms: [AtomicU32; 2],
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
            last_sub_msms: [AtomicU32::new(0), AtomicU32::new(0)],
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

    /// Sub-MSMs the most recent [`Self::run`] dispatched, `(G1, G2)`: one per job on a key
    /// whose scratch and bases fit the binding limit, more on one that had to be cut. Same
    /// attribution caveat as [`Self::last_readback_bytes`].
    pub fn last_sub_msms(&self) -> (u32, u32) {
        (
            self.last_sub_msms[0].load(Ordering::Relaxed),
            self.last_sub_msms[1].load(Ordering::Relaxed),
        )
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
    ///
    /// # A group over the binding limit runs as several sub-MSMs
    ///
    /// Everything a sub-MSM allocates grows with its scalar count: the entry array is
    /// `n_windows * general * 8` bytes, 20 windows at c = 13, so 4.19 million general scalars
    /// want 671 MB of it against a 128 MiB binding at the floor, and the spill points and the
    /// base buffers follow. So each group's range is cut into [`Piece`]s, the largest power
    /// of two whose plan fits every binding, and cut again wherever a job's base vector moves
    /// to its next buffer. Every piece is a whole counting sort plus a whole point stage for
    /// each job, encoded into the same pass, and the host sums a job's pieces after the
    /// readback, one addition per extra piece. A group that fits is one piece and encodes
    /// exactly what it did before this existed; `js_384x384_d32`'s H query at the floor is
    /// eight. The pieces have separate scratch, so a key that has to be cut holds one bucket
    /// and spill set per piece rather than per job.
    pub async fn run(
        &self,
        backend: &WgpuBackend,
        mont: Option<MontConvert<'_>>,
        groups: &[Group<'_>],
    ) -> Result<Vec<MsmResult>, ProveError> {
        let limit = backend.granted_limits().max_storage_buffer_binding_size;
        self.run_with_binding_limit(backend, mont, groups, limit)
            .await
    }

    /// Same, cutting the pieces for a binding limit of the caller's choosing rather than the
    /// device's. Public so a test can run the cut path on a key that fits: the pieces are
    /// planned against `limit` and allocated against the device, so a limit above the
    /// device's is refused at allocation as it would be without this.
    pub async fn run_with_binding_limit(
        &self,
        backend: &WgpuBackend,
        mont: Option<MontConvert<'_>>,
        groups: &[Group<'_>],
        limit: u64,
    ) -> Result<Vec<MsmResult>, ProveError> {
        // ---- plan, on the host, before anything is allocated or encoded ----
        let mut pieces: Vec<Piece> = Vec::with_capacity(groups.len());
        let mut pjobs: Vec<PieceJob> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            for (ji, job) in g.jobs.iter().enumerate() {
                if g.n == 0 {
                    continue;
                }
                // Widened before the add, like the scalar bound below. `base_off` is a
                // public field of a public `Job`, so it is caller-supplied: a `u32` sum near
                // the top of the range wraps in release, the comparison passes, and the point
                // kernels index past the base vector, which WebGPU drops in silence.
                let end = (job.base_off() as u64) + (g.n as u64);
                if (job.base_len() as u64) < end {
                    return Err(bad(format!(
                        "group {gi} job {ji} reads bases {}..{end} of a {} point vector",
                        job.base_off(),
                        job.base_len()
                    )));
                }
            }
            for p in self.pieces(gi, g, limit)? {
                let pi = pieces.len();
                for (ji, job) in g.jobs.iter().enumerate() {
                    let (chunk, local) = job.locate(p.lo);
                    let pplan = match job {
                        Job::G1 { .. } => self.g1.plan_points(&p.dplan, local)?,
                        Job::G2 { .. } => self.g2.plan_points(&p.dplan, local)?,
                    };
                    pjobs.push(PieceJob {
                        pi,
                        ji,
                        chunk,
                        pplan,
                    });
                }
                pieces.push(p);
            }
        }
        let job_of = |pj: &PieceJob| &groups[pieces[pj.pi].gi].jobs[pj.ji];

        // ---- check out and grow the scratch ----
        let mut sc = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();

        sc.digits.truncate(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            let stale = sc.digits.get(i).map(|s| s.plan != p.dplan).unwrap_or(true);
            if stale {
                let bufs = DigitBuffers::new(backend, &p.dplan, 0)?;
                let slot = DigitSlot {
                    plan: p.dplan,
                    bufs,
                };
                match sc.digits.get_mut(i) {
                    Some(s) => *s = slot,
                    None => sc.digits.push(slot),
                }
            }
        }

        sc.points.truncate(pjobs.len());
        for (i, pj) in pjobs.iter().enumerate() {
            let dplan = pieces[pj.pi].dplan;
            let curve = job_of(pj).curve();
            let key = (dplan, pj.pplan);
            let stale = sc
                .points
                .get(i)
                .map(|s| s.plan != key || s.curve != curve.suffix)
                .unwrap_or(true);
            if stale {
                let bufs = PointBuffers::new(backend, &dplan, &pj.pplan, 0, curve)?;
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
        for p in &pieces {
            slots += self.digits.sort_slots(&p.dplan);
        }
        for pj in &pjobs {
            let dplan = pieces[pj.pi].dplan;
            slots += match job_of(pj) {
                Job::G1 { .. } => self.g1.slots(&dplan, &pj.pplan),
                Job::G2 { .. } => self.g2.slots(&dplan, &pj.pplan),
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
        let mut sort_off = Vec::with_capacity(pieces.len());
        for p in &pieces {
            sort_off.push(self.digits.plan_sort(&p.dplan, ring)?);
        }
        let mut point_off = Vec::with_capacity(pjobs.len());
        for pj in &pjobs {
            let dplan = pieces[pj.pi].dplan;
            point_off.push(match job_of(pj) {
                Job::G1 { .. } => self.g1.plan(&dplan, &pj.pplan, ring)?,
                Job::G2 { .. } => self.g2.plan(&dplan, &pj.pplan, ring)?,
            });
        }
        ring.flush(backend);

        // ---- the readback, one staging buffer for every sub-MSM's window sums ----
        let mut slices: Vec<(u64, u64)> = Vec::with_capacity(pjobs.len());
        let mut total = 0u64;
        for (i, pj) in pjobs.iter().enumerate() {
            let bytes = sc.points[i].bufs.results_bytes(job_of(pj).curve());
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
        let mut sort_binds = Vec::with_capacity(pieces.len());
        for (pi, p) in pieces.iter().enumerate() {
            sort_binds.push(self.digits.bind_sort(
                backend,
                ring,
                &p.dplan,
                scalar_buf(p.gi),
                &sc.digits[pi].bufs,
            )?);
        }
        let mut point_binds = Vec::with_capacity(pjobs.len());
        for (i, pj) in pjobs.iter().enumerate() {
            let p = &pieces[pj.pi];
            let job = job_of(pj);
            let sort = &sc.digits[pj.pi].bufs;
            let pts = &sc.points[i].bufs;
            let bases = &job.chunks().bufs[pj.chunk];
            point_binds.push(match job {
                Job::G1 { .. } => self.g1.bind_all(
                    backend,
                    ring,
                    &p.dplan,
                    &pj.pplan,
                    scalar_buf(p.gi),
                    bases,
                    sort,
                    pts,
                )?,
                Job::G2 { .. } => self.g2.bind_all(
                    backend,
                    ring,
                    &p.dplan,
                    &pj.pplan,
                    scalar_buf(p.gi),
                    bases,
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
            for (pi, p) in pieces.iter().enumerate() {
                self.digits
                    .encode_sort(&mut pass, &p.dplan, &sort_binds[pi], &sort_off[pi])?;
            }
            for (i, pj) in pjobs.iter().enumerate() {
                let dplan = pieces[pj.pi].dplan;
                match job_of(pj) {
                    Job::G1 { .. } => self.g1.encode(
                        &mut pass,
                        &dplan,
                        &pj.pplan,
                        &point_binds[i],
                        &point_off[i],
                    )?,
                    Job::G2 { .. } => self.g2.encode(
                        &mut pass,
                        &dplan,
                        &pj.pplan,
                        &point_binds[i],
                        &point_off[i],
                    )?,
                }
            }
            // Last, so the token is present only if every MSM's last dispatch ran.
            sc.readback
                .as_ref()
                .expect("just built")
                .seal()
                .dispatch(backend, &mut pass);
        }
        let rb = sc.readback.as_ref().expect("just built");
        for (i, (at, bytes)) in slices.iter().enumerate() {
            rb.copy_from_at(&mut enc, &sc.points[i].bufs.results, 0, *at, *bytes)?;
        }

        // One submit and one map, for the whole of stages 5 to 9.
        self.last_readback.store(total, Ordering::Relaxed);
        let (mut n_g1, mut n_g2) = (0u32, 0u32);
        for pj in &pjobs {
            match job_of(pj) {
                Job::G1 { .. } => n_g1 += 1,
                Job::G2 { .. } => n_g2 += 1,
            }
        }
        self.last_sub_msms[0].store(n_g1, Ordering::Relaxed);
        self.last_sub_msms[1].store(n_g2, Ordering::Relaxed);
        let raw = rb.submit_and_read(backend, enc, total).await?;

        // ---- the host tail: Horner over each sub-MSM's window sums, summed per job ----
        //
        // `out` is indexed by the flat job number, `first_job[gi] + ji`, so a group whose
        // jobs were all skipped for `n == 0` still occupies its own slots and the caller's
        // job list and the result list stay in step. Getting that wrong is a proof that does
        // not verify with nothing else to go on, which is why it is a prefix sum and not a
        // `push` inside the loop that skips. A job cut into several pieces lands in the same
        // slot several times and the pieces add.
        let mut first_job = Vec::with_capacity(groups.len() + 1);
        let mut n_jobs = 0usize;
        for g in groups {
            first_job.push(n_jobs);
            n_jobs += g.jobs.len();
        }
        let mut out: Vec<Option<MsmResult>> = vec![None; n_jobs];
        for (i, pj) in pjobs.iter().enumerate() {
            let p = &pieces[pj.pi];
            let (at, bytes) = slices[i];
            let window = &raw[at as usize..(at + bytes) as usize];
            let pts = &sc.points[i].bufs;
            let part = match job_of(pj) {
                Job::G1 { .. } => MsmResult::G1(self.g1.combine(window, &p.dplan, &pj.pplan, pts)?),
                Job::G2 { .. } => MsmResult::G2(self.g2.combine(window, &p.dplan, &pj.pplan, pts)?),
            };
            let slot = &mut out[first_job[p.gi] + pj.ji];
            *slot = Some(match (slot.take(), part) {
                (None, part) => part,
                (Some(MsmResult::G1(a)), MsmResult::G1(b)) => MsmResult::G1(a + b),
                (Some(MsmResult::G2(a)), MsmResult::G2(b)) => MsmResult::G2(a + b),
                _ => return Err(bad("two pieces of one job are in different groups")),
            });
            // The trace sees only the five final points. Under the knob, record what
            // `combine` just folded, so a wrong final point names a window instead of a run.
            // See `WINDOW_DEBUG` for the iPhone this is for. `i` is the flat sub-MSM index,
            // in the deterministic order the caller listed the jobs, so the labels line up
            // between two machines.
            if crate::points::WINDOW_DEBUG.load(Ordering::Relaxed) != 0 {
                crate::points::window_log(&match job_of(pj) {
                    Job::G1 { .. } => self.g1.debug_windows(
                        &format!("j{i}_g1"),
                        window,
                        &p.dplan,
                        &pj.pplan,
                        pts,
                    )?,
                    Job::G2 { .. } => self.g2.debug_windows(
                        &format!("j{i}_g2"),
                        window,
                        &p.dplan,
                        &pj.pplan,
                        pts,
                    )?,
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

    /// Cuts one group's range into sub-MSMs. See [`Self::run`].
    ///
    /// The cap starts at the whole range and halves to the next power of two until every
    /// piece's plan fits `limit`. Checked piece by piece rather than once at the cap, because
    /// the window width is chosen per piece from its general count and a narrower window has
    /// more windows: 65,536 general scalars at c = 8 take 32 windows of entries where 100,000
    /// at c = 13 take 20, so a shorter piece can want a larger entry array.
    fn pieces(&self, gi: usize, g: &Group<'_>, limit: u64) -> Result<Vec<Piece>, ProveError> {
        if g.n == 0 {
            return Ok(Vec::new());
        }
        let general = match &g.scalars {
            Source::Device { general, .. } => *general,
            // Not available at plan time: this runs before the upload, so `pack_scalars`
            // has not walked these scalars yet. `None` means `n`, which is the overestimate
            // `Source::Device`'s `general` documents as the safe direction.
            Source::Host(_) => None,
        };
        let mut cap = g.n;
        loop {
            let mut pieces = Vec::new();
            let mut lo = 0u32;
            let mut fits = true;
            while lo < g.n {
                let mut hi = lo.saturating_add(cap).min(g.n);
                for job in g.jobs {
                    hi = hi.min(job.next_boundary(lo));
                }
                let n = hi - lo;
                // A piece never has more general scalars than the whole range, and never
                // more than its own length.
                let dplan = DigitPlan::new(n, g.scalar_off + lo, general.map(|x| x.min(n)))?;
                if !self.fits(&dplan, g.jobs, limit)? {
                    fits = false;
                    break;
                }
                pieces.push(Piece { gi, lo, dplan });
                lo = hi;
            }
            if fits {
                return Ok(pieces);
            }
            if cap == 1 {
                return Err(bad(format!(
                    "group {gi}: no sub-MSM of its {} scalars fits the {limit} byte storage \
                     binding limit, down to one scalar; the bucket array alone is over it at \
                     this window width",
                    g.n
                )));
            }
            cap = if cap.is_power_of_two() {
                cap / 2
            } else {
                1 << (31 - cap.leading_zeros())
            };
        }
    }

    /// Whether every buffer one sub-MSM at `dplan` allocates, for the sort and for each of
    /// `jobs`, is at most `limit` bytes.
    fn fits(&self, dplan: &DigitPlan, jobs: &[Job<'_>], limit: u64) -> Result<bool, ProveError> {
        if dplan.largest_binding() > limit {
            return Ok(false);
        }
        for job in jobs {
            let largest = match job {
                Job::G1 { .. } => self
                    .g1
                    .plan_points(dplan, 0)?
                    .largest_binding(dplan, wgsl::G1),
                Job::G2 { .. } => self
                    .g2
                    .plan_points(dplan, 0)?
                    .largest_binding(dplan, wgsl::G2),
            };
            if largest > limit {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
