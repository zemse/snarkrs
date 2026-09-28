//! Stages 5 to 9 as a few short submissions: every MSM in a proof, sharing what can be shared.
//!
//! [`crate::msm`] is the digit half of an MSM and [`crate::points`] is the curve half. This
//! is the file that makes a proof out of them: five MSMs, three counting sorts, one parameter
//! ring, and a run of command encoders each holding about 300 ms of GPU time.
//!
//! # Why not one submit, which is what the design asked for
//!
//! `g16-metal`'s `msm.rs` puts all five MSMs in one command buffer because committing one and
//! waiting on it costs 0.16 ms and does not get cheaper with less in it, where an extra
//! dispatch into an already open encoder costs 2 to 3 microseconds. The same ratio holds here:
//! [`crate::stages`] measured an empty submit plus its fence at **22 to 24 microseconds in
//! release** on this M2 Max, and a `mapAsync` round trip at about 0.3 ms. Design §3 therefore
//! put the whole of stages 5 to 9 in one submission, and that is what this file did until
//! BUG-32.
//!
//! What the one submission cost is the whole stage every time macOS took the GPU back. The
//! kill in [`crate::readback::Seal`]'s docs ends the compute pass that is running, and under
//! another process's load it ended the 500 to 700 ms MSM pass in 75 of 76 aborts against
//! once for the 20 ms stage 0 to 4 pass; g16-metal saw 350 ms buffers killed and 200 ms ones
//! survive. Four retries of the whole pass were not enough (16 of 40 CLI proofs gave up), and
//! each retry threw away everything that had run.
//!
//! So the batch is cut into submissions of about [`SUBMISSION_US`] of estimated GPU time,
//! each sealed and checked on its own, and a submission the GPU cut short is the only thing
//! run again, up to [`ATTEMPTS`] times. The cut is by **window slabs** of the point stage,
//! not by sub-MSMs: a G2 MSM over `js_16x16_d32` is 520 ms of the stage's 700, and cutting
//! it into five sub-MSMs measured 693 ms, into nine 1,138 ms, because every sub-MSM pays a
//! bucket clear, merge and reduction of its own and under-occupies the GPU while it runs. A
//! window slab ([`crate::points::MsmPoints::plan_slabs`]) is the same dispatches
//! distributed over more encoders: its clear, accumulation and merge touch only its own
//! windows' rows, so a slab is complete from its clear and independent of every other, and
//! the reduction runs once after the last.
//!
//! # What the kill turned out to be, measured, and what that set the two numbers to
//!
//! Sixty `g16 prove` calls per configuration on railgun-13x01 under two other proof loops,
//! rotated so the load drifted over every configuration alike:
//!
//! ```text
//! configuration                            gave up   aborted attempts   wrong
//! main (one 700 ms pass, 4 attempts)        12/60          77             1
//! one pass, 8 attempts                       2/60         102             0
//! 100 ms slabs, 4 attempts                   1/60         174             0
//! 100 ms slabs, 8 attempts                   1/60         185             0
//! 300 ms slabs, 8 attempts                   0/60         108             0
//! ```
//!
//! Two things follow. The kill is mostly per command buffer and only weakly per
//! millisecond: a 100 ms slab was killed about a quarter of the time where the 700 ms pass
//! was killed half of it, so seven times the buffers is 3.5 times the kills, and cutting
//! finer than the GPU's occupancy allows (below) costs more than it saves. And the kills
//! come in bursts: a submission whose previous attempt was killed was killed again 36 to
//! 61% of the time against 25% for one that was not, so the fourth attempt at 1.6 s often
//! sat inside the burst the first one hit, and the attempt count is the lever that moved
//! the give-up rate, from 12 to 2 in 60 on its own. The split is what makes an attempt
//! cheap enough to have eight of. The wrong proof is the stage 0 to 4 case in
//! `security/README.md` finding 7, which this file does not touch.
//!
//! The cost is one round trip per submission, about 0.4 ms each, and the occupancy a slab
//! gives up: the G2 accumulation at `js_16x16_d32` is 270 workgroups of 128 threads over
//! 38 GPU cores, so a slab of a sixth of it runs the GPU a sixth full. Warm, against one
//! submission: +3 to 5% of MSM time at 300 ms on `js_16x16_d32` and none on `anon-aadhaar`,
//! +12 to 17% at 100 ms on both.
//!
//! The readback is still one staging buffer, one window of it per sub-MSM, and design §3's
//! 64 KiB cap on it still applies: [`MsmBatch::last_readback_bytes`] reports what a given
//! proof actually asked for and [`MsmBatch::last_submits`] how many submissions it took, so
//! both claims are checkable rather than asserted.
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
use crate::readback::{is_aborted, Readback};

/// Bytes one scalar occupies on the device, in either encoding.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// Every job's slice of the readback starts on a multiple of this.
///
/// `copy_buffer_to_buffer` only needs 4, and 256 is free: a G1 window sum is 128 bytes and a
/// G2 one is 256, and `PointBuffers` already rounds its own internal `ones` offset to 256 for
/// the storage-binding alignment. Keeping the whole readback on the same grid means a dump of
/// it lines up with the buffers it came from.
const SLICE_ALIGN: u64 = 256;

/// GPU time one submission of the batch may hold, in microseconds of [`slab_us`]'s
/// estimate, unless a single window is over it (see [`MsmBatch::pieces`]).
/// `G16_WGPU_SUBMIT_US` overrides it.
///
/// 300 ms and not the 100 the task asked for, from the measurement in the module docs: the
/// kill is mostly per command buffer, so a 100 ms slab is killed a quarter of the time
/// where the 700 ms pass was killed half of it, and under two proof loops 100 ms slabs gave
/// up 1 proof in 60 against 0 in 60 at 300 ms, while costing 12 to 17% of warm MSM time
/// against 3 to 5%: a G2 accumulation at `js_16x16_d32` is 270 workgroups, and a slab of a
/// sixth of it leaves most of the GPU idle.
pub const SUBMISSION_US: f64 = 300_000.0;

/// Attempts at one submission the GPU cut short, counting the first, with `g16-metal`'s
/// backoff between them, capped at 1.6 s. Native only: the browser has no thread to sleep
/// on, and the error reaches the page as it is. `G16_WGPU_ATTEMPTS` overrides it.
///
/// Twice `g16-metal`'s four, because a submission here is a fraction of the batch and a
/// lost attempt costs that much less, and because the kills come in bursts (the table in
/// the module docs), so the fourth attempt at 1.6 s is often still inside the burst the
/// first one hit.
#[cfg(not(target_arch = "wasm32"))]
const ATTEMPTS: u32 = 8;
#[cfg(target_arch = "wasm32")]
const ATTEMPTS: u32 = 1;

/// [`ATTEMPTS`], or the override.
#[cfg(not(target_arch = "wasm32"))]
fn attempts() -> u32 {
    std::env::var("G16_WGPU_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(ATTEMPTS)
}
#[cfg(target_arch = "wasm32")]
fn attempts() -> u32 {
    ATTEMPTS
}

/// Nanoseconds one (window, general scalar) entry costs in the segmented accumulation,
/// per curve. Measured warm on this M2 Max: G1 5.8 ns at c = 13 (`js_384x384_d32`) to
/// 8.8 ns at c = 8 (`js_16x16_d32`), G2 93 to 118 ns at the same two shapes. A budget, not
/// a benchmark: it only has to put a slab within a factor of two of its measured time, and
/// the 13x between the curves is what matters, not the 3x their arithmetic would predict.
fn entry_ns(curve: wgsl::Curve) -> f64 {
    if curve.suffix == wgsl::G2.suffix {
        105.0
    } else {
        7.0
    }
}

/// Microseconds one bucket row costs in the clear, the merge and the reduction together.
/// Measured as the fixed cost of a G1 sub-MSM at c = 13, 25 ms over 81,920 rows.
const ROW_US: f64 = 0.3;

/// Nanoseconds one (window, scalar) costs in the counting sort. Measured on
/// `anon-aadhaar`, whose 1.1M scalars are 92% zeros and ones: 39 ms of sort over 35M
/// digits.
const SORT_NS: f64 = 1.0;

/// Estimated GPU microseconds of `windows` windows of one sub-MSM's point stage.
fn slab_us(curve: wgsl::Curve, dplan: &DigitPlan, windows: u32) -> f64 {
    f64::from(windows)
        * (entry_ns(curve) * f64::from(dplan.cap()) / 1000.0
            + ROW_US * f64::from(dplan.n_buckets()))
}

/// Estimated GPU microseconds of one piece's counting sort.
fn sort_us(dplan: &DigitPlan) -> f64 {
    SORT_NS * f64::from(dplan.n()) * f64::from(dplan.n_windows()) / 1000.0
}

/// [`SUBMISSION_US`], or the override.
fn submission_us() -> f64 {
    std::env::var("G16_WGPU_SUBMIT_US")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|us| *us > 0.0)
        .unwrap_or(SUBMISSION_US)
}

/// Window slabs for a point stage of `us` estimated microseconds: the fewest of near equal
/// size that each fit `budget`, and never more than one per window.
fn slabs_for(n_windows: u32, us: f64, budget: f64) -> Vec<std::ops::Range<u32>> {
    let k = ((us / budget).ceil().max(1.0) as u32).min(n_windows.max(1));
    (0..k)
        .map(|i| {
            let lo = (u64::from(i) * u64::from(n_windows) / u64::from(k)) as u32;
            let hi = (u64::from(i + 1) * u64::from(n_windows) / u64::from(k)) as u32;
            lo..hi
        })
        .filter(|r| r.end > r.start)
        .collect()
}

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

/// One job over one piece: the base buffer it reads, the point plan inside it, and the
/// window slabs its point stage is encoded in.
struct PieceJob {
    pi: usize,
    ji: usize,
    chunk: usize,
    pplan: PointPlan,
    slabs: Vec<std::ops::Range<u32>>,
}

/// One slab of one sub-MSM in encode order, with the sort it depends on when it is the
/// first of its piece, and what it is estimated to cost. What a submission is packed from.
struct Unit {
    pj: usize,
    slab: usize,
    sort: Option<usize>,
    us: f64,
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
    /// Submissions the most recent [`Self::run`] made. See [`Self::last_submits`].
    last_submits: AtomicU32,
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
            last_submits: AtomicU32::new(0),
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

    /// Submissions the most recent [`Self::run`] made, counting every attempt at one the
    /// GPU cut short: one for a batch under [`SUBMISSION_US`], more for one that was cut.
    /// `tests/proof.rs` holds a proof's submit count to one plus this. Same attribution
    /// caveat as [`Self::last_readback_bytes`].
    pub fn last_submits(&self) -> u32 {
        self.last_submits.load(Ordering::Relaxed)
    }

    /// Scratch sets sitting idle in the pool. Public so a test can assert that two concurrent
    /// proofs hold two different sets rather than racing on one, which is not observable from
    /// outside otherwise.
    pub fn pooled(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Every MSM in `groups`, in job order, over as few submissions as [`SUBMISSION_US`]
    /// allows.
    ///
    /// The returned vector is flat: group order, then job order within each group. A job with
    /// `n == 0` contributes the identity and dispatches nothing, which is what an empty L
    /// query needs.
    ///
    /// `mont`, if present, is one `fr_mont_to_std` dispatch encoded before everything else.
    /// The compute pass orders dispatches and makes each one's writes visible to the next, so
    /// the sorts that read `mont.dst` see the converted values with no explicit barrier; a
    /// later submission reads what an earlier one wrote because the queue runs them in order
    /// and each is waited on before the next is encoded.
    ///
    /// # A submission the GPU cuts short is run again, alone
    ///
    /// Every submission ends with [`crate::readback::Seal`]'s token and is refused without
    /// it. On native it is then re-encoded and submitted again, up to [`ATTEMPTS`] times with
    /// a backoff, and only it: everything before it was verified complete and nothing it
    /// writes is read by anything before it. It can be run again because each of its parts
    /// is complete from its own first write: the conversion is a pure map, a sort zeroes its
    /// counters first, a window slab clears its buckets first, and the reduction and the
    /// ones pass only read what the slabs left. The uploads the refused submission carried
    /// are queued again first, because a kill has been seen to take them with it (BUG-33),
    /// and the first submission's are the parameter ring every later one reads through.
    /// Once the attempts are spent the error is a
    /// `ProveError::Device`, which `g16 prove`'s fallback proves past. The stage timing the
    /// caller keeps includes the attempts that were lost.
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
        self.run_with_limits(backend, mont, groups, limit, submission_us())
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
        self.run_with_limits(backend, mont, groups, limit, submission_us())
            .await
    }

    /// Same, with the submission budget of the caller's choosing as well. Public so a test
    /// can force a key that fits one submission into one per window.
    pub async fn run_with_limits(
        &self,
        backend: &WgpuBackend,
        mont: Option<MontConvert<'_>>,
        groups: &[Group<'_>],
        limit: u64,
        budget_us: f64,
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
                    let us = slab_us(job.curve(), &p.dplan, p.dplan.n_windows());
                    pjobs.push(PieceJob {
                        pi,
                        ji,
                        chunk,
                        pplan,
                        slabs: slabs_for(p.dplan.n_windows(), us, budget_us),
                    });
                }
                pieces.push(p);
            }
        }
        let job_of = |pj: &PieceJob| &groups[pieces[pj.pi].gi].jobs[pj.ji];

        // The slabs in encode order, each with the sort it needs run first, packed into
        // submissions of at most `budget_us` each unless a single slab is over it.
        let mut units: Vec<Unit> = Vec::new();
        let mut sorted: Vec<bool> = vec![false; pieces.len()];
        for (i, pj) in pjobs.iter().enumerate() {
            let p = &pieces[pj.pi];
            let curve = job_of(pj).curve();
            for (k, s) in pj.slabs.iter().enumerate() {
                let mut us = slab_us(curve, &p.dplan, s.end - s.start);
                let sort = (!sorted[pj.pi]).then_some(pj.pi);
                if sort.is_some() {
                    sorted[pj.pi] = true;
                    us += sort_us(&p.dplan);
                }
                if k + 1 == pj.slabs.len() {
                    // The ones pass walks every scalar once.
                    us += SORT_NS * f64::from(p.dplan.n()) / 1000.0;
                }
                units.push(Unit {
                    pj: i,
                    slab: k,
                    sort,
                    us,
                });
            }
        }
        let mut subs: Vec<std::ops::Range<usize>> = Vec::new();
        let mut start = 0usize;
        let mut acc = 0.0f64;
        for (ui, u) in units.iter().enumerate() {
            if ui > start && acc + u.us > budget_us {
                subs.push(start..ui);
                start = ui;
                acc = 0.0;
            }
            acc += u.us;
        }
        // One submission even with nothing to dispatch, so a `mont` alone still runs.
        subs.push(start..units.len());

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
                Job::G1 { .. } => self.g1.slots_slabs(&dplan, &pj.pplan, &pj.slabs),
                Job::G2 { .. } => self.g2.slots_slabs(&dplan, &pj.pplan, &pj.slabs),
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
                Job::G1 { .. } => self.g1.plan_slabs(&dplan, &pj.pplan, ring, &pj.slabs)?,
                Job::G2 { .. } => self.g2.plan_slabs(&dplan, &pj.pplan, ring, &pj.slabs)?,
            });
        }
        ring.flush(backend);

        // ---- the readback, one staging buffer with a window per sub-MSM ----
        //
        // Sized for every sub-MSM at once and filled a submission at a time: the sub-MSMs a
        // submission finishes are copied into it from offset 0 and read back before the next
        // submission is encoded, so no submission's slices can be more than all of them.
        let mut total = 0u64;
        for (i, pj) in pjobs.iter().enumerate() {
            let bytes = sc.points[i].bufs.results_bytes(job_of(pj).curve());
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

        // ---- encode, submit and read, one submission at a time ----
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
        let (mut n_g1, mut n_g2) = (0u32, 0u32);
        for pj in &pjobs {
            match job_of(pj) {
                Job::G1 { .. } => n_g1 += 1,
                Job::G2 { .. } => n_g2 += 1,
            }
        }
        self.last_sub_msms[0].store(n_g1, Ordering::Relaxed);
        self.last_sub_msms[1].store(n_g2, Ordering::Relaxed);
        let rb = sc.readback.as_ref().expect("just built");
        let mut read_bytes = 0u64;
        let mut submits = 0u32;
        for (si, sub) in subs.iter().enumerate() {
            // The sub-MSMs this submission finishes, and their windows of the staging buffer.
            let done: Vec<usize> = units[sub.clone()]
                .iter()
                .filter(|u| u.slab + 1 == pjobs[u.pj].slabs.len())
                .map(|u| u.pj)
                .collect();
            let mut slices: Vec<(u64, u64)> = Vec::with_capacity(done.len());
            let mut bytes_here = 0u64;
            for &i in &done {
                let bytes = sc.points[i].bufs.results_bytes(job_of(&pjobs[i]).curve());
                slices.push((bytes_here, bytes));
                bytes_here += bytes.div_ceil(SLICE_ALIGN) * SLICE_ALIGN;
            }
            let bytes_here = bytes_here.max(4);
            read_bytes += bytes_here;

            let mut attempt = 1u32;
            let attempts = attempts();
            let raw = loop {
                let mut enc =
                    backend
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("g16 stages 5-9"),
                        });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("g16 stages 5-9"),
                        timestamp_writes: None,
                    });
                    if si == 0 {
                        if let (Some(m), Some(bind)) = (&mont, &mont_bind) {
                            self.digits.encode_mont(&mut pass, bind, m.n, &mont_off)?;
                        }
                    }
                    for u in &units[sub.clone()] {
                        let pj = &pjobs[u.pj];
                        let dplan = &pieces[pj.pi].dplan;
                        if let Some(pi) = u.sort {
                            self.digits.encode_sort(
                                &mut pass,
                                dplan,
                                &sort_binds[pi],
                                &sort_off[pi],
                            )?;
                        }
                        let last = u.slab + 1 == pj.slabs.len();
                        match job_of(pj) {
                            Job::G1 { .. } => {
                                let (b, o) = (&point_binds[u.pj], &point_off[u.pj]);
                                self.g1
                                    .encode_slab(&mut pass, dplan, &pj.pplan, b, o, u.slab)?;
                                if last {
                                    self.g1.encode_tail(&mut pass, dplan, &pj.pplan, b, o)?;
                                }
                            }
                            Job::G2 { .. } => {
                                let (b, o) = (&point_binds[u.pj], &point_off[u.pj]);
                                self.g2
                                    .encode_slab(&mut pass, dplan, &pj.pplan, b, o, u.slab)?;
                                if last {
                                    self.g2.encode_tail(&mut pass, dplan, &pj.pplan, b, o)?;
                                }
                            }
                        }
                    }
                    // Last, so the token is present only if every dispatch above ran.
                    rb.seal().dispatch(backend, &mut pass);
                }
                for (k, &i) in done.iter().enumerate() {
                    let (at, bytes) = slices[k];
                    rb.copy_from_at(&mut enc, &sc.points[i].bufs.results, 0, at, bytes)?;
                }
                submits += 1;
                match rb.submit_and_read(backend, enc, bytes_here).await {
                    Err(e) if is_aborted(&e) && attempt < attempts => {
                        wait_before_retry(si + 1, subs.len(), &e, attempt, attempts);
                        attempt += 1;
                        // A kill can take the refused submission's queued uploads with
                        // it, and the first submission's carry the whole parameter ring
                        // and every host scalar vector (BUG-33: a first submission of a
                        // fresh circuit lost its key that way, 4 times in 13). Queued again
                        // so the retry reads what it was planned over; the epoch is
                        // re-queued by `Seal::dispatch` above.
                        ring.flush(backend);
                        for (gi, g) in groups.iter().enumerate() {
                            if let (Source::Host(xs), Some(k)) = (&g.scalars, upload_at[gi]) {
                                let (words, _) = pack_scalars(xs);
                                backend.queue().write_buffer(
                                    &sc.uploads[k],
                                    0,
                                    bytemuck::cast_slice(&words),
                                );
                            }
                        }
                    }
                    r => break r?,
                }
            };

            // ---- the host tail: Horner over each sub-MSM's window sums, summed per job ----
            for (k, &i) in done.iter().enumerate() {
                let pj = &pjobs[i];
                let p = &pieces[pj.pi];
                let (at, bytes) = slices[k];
                let window = &raw[at as usize..(at + bytes) as usize];
                let pts = &sc.points[i].bufs;
                let part = match job_of(pj) {
                    Job::G1 { .. } => {
                        MsmResult::G1(self.g1.combine(window, &p.dplan, &pj.pplan, pts)?)
                    }
                    Job::G2 { .. } => {
                        MsmResult::G2(self.g2.combine(window, &p.dplan, &pj.pplan, pts)?)
                    }
                };
                let slot = &mut out[first_job[p.gi] + pj.ji];
                *slot = Some(match (slot.take(), part) {
                    (None, part) => part,
                    (Some(MsmResult::G1(a)), MsmResult::G1(b)) => MsmResult::G1(a + b),
                    (Some(MsmResult::G2(a)), MsmResult::G2(b)) => MsmResult::G2(a + b),
                    _ => return Err(bad("two pieces of one job are in different groups")),
                });
                // The trace sees only the five final points. Under the knob, record what
                // `combine` just folded, so a wrong final point names a window instead of a
                // run. See `WINDOW_DEBUG` for the iPhone this is for. `i` is the flat
                // sub-MSM index, in the deterministic order the caller listed the jobs, so
                // the labels line up between two machines.
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
        }
        self.last_readback.store(read_bytes, Ordering::Relaxed);
        self.last_submits.store(submits, Ordering::Relaxed);
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
    ///
    /// Never cut for [`SUBMISSION_US`]: a window is the finest slab there is, and a job whose
    /// single window is over the budget (a dense 2^22 key's G2 job under `auto` limits, at
    /// about 310 ms a window) runs one window per submission rather than paying a second
    /// sort, clear, merge and reduction. Cutting it into two sub-MSMs instead measured 9.7 s
    /// of MSM against 7.6 s in one.
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

/// Reports a submission the GPU cut short and sleeps `g16-metal`'s backoff before it is run
/// again: 400, 800, then 1600 ms from the third attempt on. Native only; see [`ATTEMPTS`].
#[cfg(not(target_arch = "wasm32"))]
fn wait_before_retry(which: usize, of: usize, e: &ProveError, attempt: u32, of_attempts: u32) {
    let wait = std::time::Duration::from_millis(200 << attempt.min(3));
    eprintln!(
        "wgpu: stages 5 to 9, submission {which} of {of}: {e}; attempt {attempt} of \
         {of_attempts}, retrying in {} ms",
        wait.as_millis()
    );
    std::thread::sleep(wait);
}

/// Never reached: [`ATTEMPTS`] is 1 here, so no submission is run twice.
#[cfg(target_arch = "wasm32")]
fn wait_before_retry(_which: usize, _of: usize, _e: &ProveError, _attempt: u32, _of_attempts: u32) {
}
