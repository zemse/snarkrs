//! The Metal side of the three ceremony seams: [`MsmBackend`], [`GroupFft`] and
//! [`KeyScale`].
//!
//! Separate from [`crate::backend`], which implements `g16_core::Backend`, the whole
//! prover. A ceremony command wants one primitive at a time and holds no proving key, so
//! it selects one of these instead. All three take host slices: the ceremony's inputs are
//! mmap'd sections of a multi-gigabyte file that no `prepare` step ever made resident, and
//! on unified memory the upload is a `copy_from_slice` anyway.
//!
//! Nothing here may be shared into `g16-wgpu`. A point scalar multiplication inlines the
//! point operations three or more times over a 256-byte `Xyzz<Fq2>`, which is exactly the
//! shape WebKit 323560 (filed by this project) miscompiles on iOS.

use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{
    AffineRepr, CurveGroup, Field, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective,
};
use g16_msm::xyzz::Xyzz;
use g16_msm::{AccelError, GroupFft, KeyScale, MsmBackend};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

use crate::fft::FftKernels;
use crate::kernels::{CEREMONY_MSL, FR_MSL, MSM_MSL};
use crate::layout::{
    as_bytes, Packed, PackedFq, PackedFq2, PackedFr, PackedG1Affine, PackedG2Affine, PackedScalar,
};
use crate::msm::{MetalMsm, PackedXyzzG1, PackedXyzzG2};

/// The name every error and every `name()` on this side reports, and the spelling
/// `--backend metal` uses.
const BACKEND: &str = "metal";

/// [`MetalMsm`] behind the ceremony's MSM trait.
///
/// A wrapper rather than an `impl` on `MetalMsm` itself, because `MetalMsm`'s own
/// `msm_g1` takes device handles and the trait's takes host slices. Two methods of the
/// same name and different argument types on one type resolve silently in favour of the
/// inherent one, and a reader cannot see which was called.
///
/// Every call uploads its own bases. That is right for `setup`, whose slots are gathered
/// per output point and never repeat a base vector, and it is the reason the slot loop
/// wants `MetalMsm::msm_batch` rather than this trait once a kernel is worth its dispatch:
/// 97,648 slots at 0.149 ms of commit-and-wait each is 14.5 s of round trip on a 45.6 s
/// command.
pub struct MetalMsmBackend {
    msm: MetalMsm,
}

impl MetalMsmBackend {
    /// Compiles the MSM library. Expensive (about 60 ms of runtime MSL compilation plus
    /// pipeline construction), so it belongs once at the top of a command.
    pub fn new() -> Result<Self, ProveError> {
        Ok(Self {
            msm: MetalMsm::new()?,
        })
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        Ok(Self {
            msm: MetalMsm::with_device(device)?,
        })
    }

    pub fn msm(&self) -> &MetalMsm {
        &self.msm
    }
}

/// The trait is infallible, so a device failure has nowhere to go but a panic.
///
/// That is the right end for it. Every failure `MetalMsm` reports here is a lost device or
/// a pipeline that would not build, never a numeric one, and the alternative to stopping is
/// returning a point: an identity, or whatever a half-run command buffer left behind. Both
/// are silently wrong bytes in a `.zkey` that people will prove against for years. The
/// panic message names the backend so it is not mistaken for an arithmetic bug.
impl MsmBackend for MetalMsmBackend {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        let bases = self.msm.upload_g1_bases(bases);
        let scalars = self.msm.upload_scalars(scalars);
        self.msm
            .msm_g1(&bases, &scalars)
            .unwrap_or_else(|e| panic!("metal msm_g1 failed, no zkey was written: {e}"))
    }

    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        let bases = self.msm.upload_g2_bases(bases);
        let scalars = self.msm.upload_scalars(scalars);
        self.msm
            .msm_g2(&bases, &scalars)
            .unwrap_or_else(|e| panic!("metal msm_g2 failed, no zkey was written: {e}"))
    }
}

/// The group inverse FFT behind `ptau prepare`, the slowest command in the project.
///
/// A wrapper over [`FftKernels`], which owns the kernels, the twiddle table and the
/// single-command-buffer submission. What the trait adds on top is the error mapping and
/// the crossover: `min_block` is the backend's own number, so `prepare::ifft_block`
/// (prepare.rs:352) carries no backend-shaped branch and the threshold can move without
/// touching the ceremony crate.
pub struct MetalGroupFft {
    fft: FftKernels,
}

impl MetalGroupFft {
    /// Compiles the FFT library. Expensive (runtime MSL compilation plus 16 pipelines),
    /// so it belongs once at the top of a command, the same as [`MetalMsmBackend::new`].
    pub fn new() -> Result<Self, ProveError> {
        Ok(Self {
            fft: FftKernels::new()?,
        })
    }

    /// Fallible, where the seam that stubbed this type had it infallible: the pipelines
    /// are built here now, and a pipeline that will not build has to be reported.
    /// [`MetalMsmBackend::with_device`] already had this signature for the same reason.
    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        Ok(Self {
            fft: FftKernels::with_device(device)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.fft.device()
    }

    /// The kernel layer, for a measurement that wants to set a window or a block size
    /// without going through `ptau prepare`.
    pub fn kernels(&self) -> &FftKernels {
        &self.fft
    }

    /// [`FftKernels::with_min_block`], so a test can put a file the shipped crossover
    /// would route home onto the device instead.
    pub fn with_min_block(mut self, n: usize) -> Self {
        self.fft = self.fft.with_min_block(n);
        self
    }
}

impl GroupFft for MetalGroupFft {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn min_block(&self) -> usize {
        self.fft.min_block()
    }

    fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
        self.fft
            .ifft_g1(a)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G1", e.to_string()))
    }

    fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
        self.fft
            .ifft_g2(a)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G2", e.to_string()))
    }
}

/// `batchApplyKey` behind the four contribute and beacon commands.
///
/// Holds one [`CeremonyKernels`] for its whole life. Compiling the library is ~60 ms and
/// the four commands call this trait once per file chunk: `zkey contribute` at 2^20 hands
/// over 1,664,401 points in 26 calls, and `ptau contribute` hands over one
/// `response_chunk` at a time, so a per-call compile would cost more than the arithmetic.
pub struct MetalKeyScale {
    kernels: CeremonyKernels,
    min_points: usize,
}

/// Below this many points in one call the CPU wins and the device is not asked.
///
/// Measured on an M2 Max over the whole call, host pack and unpack included, against
/// `CpuKeyScale` on the same vector (`g16-ceremony/tests/contribute_metal.rs`,
/// `crossover_sweep`), milliseconds:
///
/// ```text
///     n      G1 cpu   G1 gpu    G2 cpu   G2 gpu
///    64       2.09     5.83      5.94    25.60
///   128       6.83     7.09     19.20    27.88
///   256      16.64     6.78     45.55    28.74
///   512      36.27     6.79    101.17    28.68
/// ```
///
/// Both groups cross between 128 and 256, so one number serves both. Note the CPU side is
/// single-threaded below `KEY_SUBCHUNK` (1024) points, which is not a flaw in the
/// comparison: a short call is exactly what a section's last partial chunk is, and the
/// CPU really does run it on one thread.
///
/// The device's cost is nearly flat to 4096 points, so anything from 128 to 1024 costs at
/// most a few milliseconds a call; 256 is the crossover rather than a tuned optimum. It
/// is a live path either way: a power-8 `.ptau` has 511 points in its largest section.
const KEY_MIN_POINTS: usize = 256;

impl MetalKeyScale {
    pub fn new() -> Result<Self, ProveError> {
        Ok(Self::from_kernels(CeremonyKernels::new()?))
    }

    /// Returns a `Result` where the stub it replaced returned `Self`: this now compiles
    /// the MSL library, which is the call that can fail.
    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        Ok(Self::from_kernels(CeremonyKernels::with_device(device)?))
    }

    fn from_kernels(kernels: CeremonyKernels) -> Self {
        Self {
            kernels,
            min_points: env_usize("G16_METAL_KEY_MIN", KEY_MIN_POINTS),
        }
    }

    pub fn device(&self) -> &Device {
        self.kernels.device()
    }

    pub fn kernels(&self) -> &CeremonyKernels {
        &self.kernels
    }

    /// The crossover this instance uses. See [`KEY_MIN_POINTS`].
    pub fn min_points(&self) -> usize {
        self.min_points
    }

    /// The crossover, for a sweep and for the tests that need every call on the device
    /// however short it is. Zero and one both mean "everything".
    pub fn with_min_points(mut self, n: usize) -> Self {
        self.min_points = n;
        self
    }

    /// Points per command buffer, forwarded to [`CeremonyKernels::with_chunk`]. Only the
    /// tests move it.
    pub fn with_chunk(mut self, chunk: usize) -> Self {
        self.kernels = self.kernels.with_chunk(chunk);
        self
    }
}

impl KeyScale for MetalKeyScale {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn apply_key_g1(&self, points: &mut [G1Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        if points.len() < self.min_points {
            apply_key_on_host(points, first, inc);
            return Ok(());
        }
        self.kernels
            .apply_key_g1(points, first, inc)
            .map_err(|e| AccelError::device(BACKEND, "batch apply key over G1", e.to_string()))
    }

    fn apply_key_g2(&self, points: &mut [G2Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        if points.len() < self.min_points {
            apply_key_on_host(points, first, inc);
            return Ok(());
        }
        self.kernels
            .apply_key_g2(points, first, inc)
            .map_err(|e| AccelError::device(BACKEND, "batch apply key over G2", e.to_string()))
    }
}

/// `P_i *= first * inc^i` on this thread, for a call under [`KEY_MIN_POINTS`].
///
/// The same arithmetic as `g16_ceremony::CpuKeyScale` without the rayon split, written out
/// here because g16-ceremony is not a dependency of this crate and the accel traits live
/// in g16-msm precisely so that it need not be. Duplicating it cannot cost byte identity:
/// the output is affine and affine is canonical, so a correct implementation has no room
/// to differ. One inversion for the batch, as there.
fn apply_key_on_host<C: AffineRepr<ScalarField = Fr>>(points: &mut [C], first: Fr, inc: Fr) {
    let mut t = first;
    let scaled: Vec<C::Group> = points
        .iter()
        .map(|p| {
            let q = *p * t;
            t *= inc;
            q
        })
        .collect();
    points.copy_from_slice(&C::Group::normalize_batch(&scaled));
}

// ---------------------------------------------------------------------------
// The kernel layer
//
// `shaders/ceremony.metal` and this half are one unit and must be changed together, the
// same contract `layout.rs` states for the wire structs. What lives here is the shape of
// the submission and the one field inversion Montgomery's trick needs, which the shaders
// deliberately do not have.
// ---------------------------------------------------------------------------

/// Window widths compiled into `shaders/ceremony.metal`.
///
/// The pipeline names are built from this list (`cer_point_mul_g1_c4` and so on), so a
/// width added here without a matching `CER_LADDER_KERNELS` line fails at construction,
/// where the message names the missing kernel, rather than at dispatch.
pub(crate) const CER_WINDOWS: [u32; 4] = [2, 3, 4, 5];

/// Ladder window for G1. Measured, not derived: see [`WINDOW_G2`].
const WINDOW_G1: u32 = 4;

/// Ladder window for G2, and the same width G1 takes.
///
/// The multiply count says wider is always better, because the doublings do not shrink
/// and the additions do. What the count cannot see is that the table is indexed by a
/// digit, so it cannot live in registers at all: `Xyzz<Fq>` is 128 bytes and `Xyzz<Fq2>`
/// is 256, and `2^(c-1)` multiples is 1 KB per thread at c=4 on G1 and at c=3 on G2.
///
/// The sweep is `tests::window_sweep`, 2^16 full-width scalars, medians of four on this
/// M2 Max:
///
/// | c | G1 ms | G2 ms |
/// |---|---:|---:|
/// | 2 | 41.2 | 126.1 |
/// | 3 | **30.9** | 109.5 |
/// | 4 | **30.9** | **103.9** |
/// | 5 | 32.5 | 105.1 |
///
/// Two results worth keeping. G1 at c=3 and c=4 is a dead tie, so 4 is taken for having
/// the smaller multiply count and therefore the more headroom; the choice is worth about
/// 2% either way and is not worth re-tuning. And G2 does NOT want a narrower window than
/// G1, which is what the register-pressure argument predicted: 8 entries of 256 bytes is
/// 2 KB a thread and it still beats c=3 by 5%. The spill is real, the ladder is simply
/// not latency bound on it, because the doublings between two table reads are enough work
/// to cover the load.
const WINDOW_G2: u32 = 4;

/// Points one thread owns in each of the two batch-to-affine passes.
///
/// Both passes are serial within a segment, so this trades thread count against the host
/// scan: `n / SEG_LEN` threads and `n / SEG_LEN` host multiplications. At 32 a 2^18-point
/// chunk is 8,192 threads, which fills this machine, and 8,192 host `Fq` multiplies, which
/// is under a millisecond against the ladder dispatch that produced the points.
const SEG_LEN: usize = 32;

/// Points per command buffer.
///
/// Bounds the scratch, which is the binding constraint rather than the dispatch cost: a
/// chunk holds the input, the XYZZ intermediate, the per-point prefix and the affine
/// output at once, so on G2 it is 2^18 * (128 + 256 + 64 + 128) = 151 MB. Two command
/// buffers per chunk at 0.149 ms each is 1.2 ms on a 2^20 section, which is not a cost
/// worth trading memory for.
const DEFAULT_CHUNK: usize = 1 << 18;

pub(crate) fn cer_err(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: BACKEND,
        reason: reason.into(),
    }
}

/// Mirrors `struct CerParams` in `shaders/ceremony.metal`. Passed by `setBytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CerParams {
    n: u32,
    seg_len: u32,
    segments: u32,
    index_off: u32,
    has_inc: u32,
    first: PackedFr,
    inc: PackedFr,
}

const _: () = {
    assert!(core::mem::size_of::<CerParams>() == 84);
    assert!(core::mem::align_of::<CerParams>() == 4);
};

/// The pipelines for one group. `point_mul` and `apply_key` are parallel to
/// [`CER_WINDOWS`].
struct GroupPipelines {
    point_mul: Vec<ComputePipelineState>,
    apply_key: Vec<ComputePipelineState>,
    affine_prefix: ComputePipelineState,
    affine_finish: ComputePipelineState,
}

/// Position of `c` in [`CER_WINDOWS`], clamped to the compiled range rather than
/// rejected: an out-of-range width can only come from the env override, and a sweep that
/// silently ran the wrong width would be reported as a measurement.
pub(crate) fn window_index(c: u32) -> usize {
    CER_WINDOWS
        .iter()
        .position(|w| *w == c)
        .unwrap_or(CER_WINDOWS.len() - 1)
}

/// What the two ladder kernels and the two batch-to-affine kernels need to know about a
/// group, so the submission code is written once.
trait CerGroup {
    type Raw: Copy;
    type Field: Field;
    type PackedField: Packed + Default;
    type PackedPoint: Packed + Default;
    type PackedAffine: Packed + Default;
    type Affine;

    /// Names this group in an error, in the spelling [`AccelError`] uses.
    const OP: &'static str;

    fn pack_point(p: &Xyzz<Self::Raw>) -> Self::PackedPoint;
    fn unpack_point(p: &Self::PackedPoint) -> Xyzz<Self::Raw>;
    fn pack_affine(p: &Self::Affine) -> Self::PackedAffine;
    fn unpack_affine(p: &Self::PackedAffine) -> Self::Affine;
    fn field(p: &Self::PackedField) -> Self::Field;
    fn pack_field(f: &Self::Field) -> Self::PackedField;
    fn pipelines(k: &CeremonyKernels) -> &GroupPipelines;
    fn window(k: &CeremonyKernels) -> u32;
}

struct CerG1;
struct CerG2;

impl CerGroup for CerG1 {
    type Raw = RawFq;
    type Field = Fq;
    type PackedField = PackedFq;
    type PackedPoint = PackedXyzzG1;
    type PackedAffine = PackedG1Affine;
    type Affine = G1Affine;

    const OP: &'static str = "G1";

    fn pack_point(p: &Xyzz<RawFq>) -> PackedXyzzG1 {
        PackedXyzzG1 {
            x: PackedFq::from_fq(&p.x.to_fq()),
            y: PackedFq::from_fq(&p.y.to_fq()),
            zz: PackedFq::from_fq(&p.zz.to_fq()),
            zzz: PackedFq::from_fq(&p.zzz.to_fq()),
        }
    }

    fn unpack_point(p: &PackedXyzzG1) -> Xyzz<RawFq> {
        Xyzz {
            x: RawFq::from_fq(&p.x.to_fq()),
            y: RawFq::from_fq(&p.y.to_fq()),
            zz: RawFq::from_fq(&p.zz.to_fq()),
            zzz: RawFq::from_fq(&p.zzz.to_fq()),
        }
    }

    fn pack_affine(p: &G1Affine) -> PackedG1Affine {
        PackedG1Affine::from_affine(p)
    }

    fn unpack_affine(p: &PackedG1Affine) -> G1Affine {
        p.to_affine()
    }

    fn field(p: &PackedFq) -> Fq {
        p.to_fq()
    }

    fn pack_field(f: &Fq) -> PackedFq {
        PackedFq::from_fq(f)
    }

    fn pipelines(k: &CeremonyKernels) -> &GroupPipelines {
        &k.g1
    }

    fn window(k: &CeremonyKernels) -> u32 {
        k.window_g1
    }
}

impl CerGroup for CerG2 {
    type Raw = RawFq2;
    type Field = Fq2;
    type PackedField = PackedFq2;
    type PackedPoint = PackedXyzzG2;
    type PackedAffine = PackedG2Affine;
    type Affine = G2Affine;

    const OP: &'static str = "G2";

    fn pack_point(p: &Xyzz<RawFq2>) -> PackedXyzzG2 {
        PackedXyzzG2 {
            x: PackedFq2::from_fq2(&p.x.to_fq2()),
            y: PackedFq2::from_fq2(&p.y.to_fq2()),
            zz: PackedFq2::from_fq2(&p.zz.to_fq2()),
            zzz: PackedFq2::from_fq2(&p.zzz.to_fq2()),
        }
    }

    fn unpack_point(p: &PackedXyzzG2) -> Xyzz<RawFq2> {
        Xyzz {
            x: RawFq2::from_fq2(&p.x.to_fq2()),
            y: RawFq2::from_fq2(&p.y.to_fq2()),
            zz: RawFq2::from_fq2(&p.zz.to_fq2()),
            zzz: RawFq2::from_fq2(&p.zzz.to_fq2()),
        }
    }

    fn pack_affine(p: &G2Affine) -> PackedG2Affine {
        PackedG2Affine::from_affine(p)
    }

    fn unpack_affine(p: &PackedG2Affine) -> G2Affine {
        p.to_affine()
    }

    fn field(p: &PackedFq2) -> Fq2 {
        p.to_fq2()
    }

    fn pack_field(f: &Fq2) -> PackedFq2 {
        PackedFq2::from_fq2(f)
    }

    fn pipelines(k: &CeremonyKernels) -> &GroupPipelines {
        &k.g2
    }

    fn window(k: &CeremonyKernels) -> u32 {
        k.window_g2
    }
}

/// The three kernels behind the ceremony seams: a general point scalar multiplication,
/// batch projective-to-affine, and the batch apply-key.
///
/// Compiling the library is the expensive call and it belongs once at the top of a
/// command, exactly as [`crate::msm::MetalMsm::new`] does. Nothing here holds per-command
/// state, so one instance serves a whole run.
pub struct CeremonyKernels {
    device: Device,
    queue: CommandQueue,
    g1: GroupPipelines,
    g2: GroupPipelines,
    window_g1: u32,
    window_g2: u32,
    chunk: usize,
}

impl CeremonyKernels {
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default()
            .ok_or_else(|| cer_err("no Metal device; this machine cannot run the metal backend"))?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        // One translation unit, in dependency order: the `Fr` prelude, then the point
        // arithmetic, then this file, which adds no curve math of its own. Each has its
        // own header guard, so the concatenation is the include.
        let source = format!("{FR_MSL}\n{MSM_MSL}\n{CEREMONY_MSL}\n");
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &opts)
            .map_err(|e| cer_err(format!("MSL compilation failed: {e}")))?;

        let pso = |name: &str| -> Result<ComputePipelineState, ProveError> {
            let f = library
                .get_function(name, None)
                .map_err(|e| cer_err(format!("kernel {name} not found: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| cer_err(format!("pipeline {name}: {e}")))
        };

        let group = |g: &str| -> Result<GroupPipelines, ProveError> {
            let mut point_mul = Vec::with_capacity(CER_WINDOWS.len());
            let mut apply_key = Vec::with_capacity(CER_WINDOWS.len());
            for c in CER_WINDOWS {
                point_mul.push(pso(&format!("cer_point_mul_{g}_c{c}"))?);
                apply_key.push(pso(&format!("cer_apply_key_{g}_c{c}"))?);
            }
            Ok(GroupPipelines {
                point_mul,
                apply_key,
                affine_prefix: pso(&format!("cer_affine_prefix_{g}"))?,
                affine_finish: pso(&format!("cer_affine_finish_{g}"))?,
            })
        };

        let g1 = group("g1")?;
        let g2 = group("g2")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            g1,
            g2,
            window_g1: env_window("G16_METAL_CER_C_G1", WINDOW_G1),
            window_g2: env_window("G16_METAL_CER_C_G2", WINDOW_G2),
            chunk: DEFAULT_CHUNK,
        })
    }

    /// Points per command buffer. Only the tests move this: they need a chunk small
    /// enough that a cheap vector still crosses it, because the chunk boundary is where a
    /// wrong `index_off` or a segment seed off by one shows up.
    pub fn with_chunk(mut self, chunk: usize) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The ladder window this instance uses for G1. See [`WINDOW_G1`] for the measurement.
    pub fn window_g1(&self) -> u32 {
        self.window_g1
    }

    pub fn window_g2(&self) -> u32 {
        self.window_g2
    }

    /// `out[i] = [scalars[i]] points[i]`, the primitive `point_times_fr` (prepare.rs:149)
    /// is on the CPU.
    ///
    /// The result agrees with the CPU as a curve point, not limb for limb: XYZZ is
    /// projective and the two ladders reach the same point through different
    /// representatives. Every ceremony path out of here ends in
    /// [`Self::batch_to_affine_g1`], and affine is canonical.
    pub fn point_mul_g1(
        &self,
        points: &[Xyzz<RawFq>],
        scalars: &[Fr],
    ) -> Result<Vec<Xyzz<RawFq>>, ProveError> {
        self.point_mul::<CerG1>(points, scalars)
    }

    pub fn point_mul_g2(
        &self,
        points: &[Xyzz<RawFq2>],
        scalars: &[Fr],
    ) -> Result<Vec<Xyzz<RawFq2>>, ProveError> {
        self.point_mul::<CerG2>(points, scalars)
    }

    /// Montgomery's trick over the whole vector, one inversion per chunk and that one on
    /// the host. The CPU twin is `batch_to_affine` (prepare.rs:198).
    pub fn batch_to_affine_g1(&self, points: &[Xyzz<RawFq>]) -> Result<Vec<G1Affine>, ProveError> {
        self.batch_to_affine::<CerG1>(points)
    }

    pub fn batch_to_affine_g2(&self, points: &[Xyzz<RawFq2>]) -> Result<Vec<G2Affine>, ProveError> {
        self.batch_to_affine::<CerG2>(points)
    }

    /// `points[i] = [first * inc^i] points[i]`, in place and affine out.
    ///
    /// `inc == Fr::ONE` is the constant-scalar case both zkey commands use and it is not a
    /// separate kernel, only a flag that skips the per-thread exponentiation.
    pub fn apply_key_g1(
        &self,
        points: &mut [G1Affine],
        first: Fr,
        inc: Fr,
    ) -> Result<(), ProveError> {
        self.apply_key::<CerG1>(points, first, inc)
    }

    pub fn apply_key_g2(
        &self,
        points: &mut [G2Affine],
        first: Fr,
        inc: Fr,
    ) -> Result<(), ProveError> {
        self.apply_key::<CerG2>(points, first, inc)
    }

    // -- the shared submission code --

    fn buffer<T: Packed>(&self, items: &[T]) -> Buffer {
        let bytes = as_bytes(items);
        if bytes.is_empty() {
            return self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }
        self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn scratch<T>(&self, len: usize) -> Buffer {
        let bytes = (len.max(1) * core::mem::size_of::<T>()) as u64;
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    fn point_mul<G: CerGroup>(
        &self,
        points: &[Xyzz<G::Raw>],
        scalars: &[Fr],
    ) -> Result<Vec<Xyzz<G::Raw>>, ProveError> {
        if points.len() != scalars.len() {
            return Err(cer_err(format!(
                "point_mul over {}: {} points against {} scalars",
                G::OP,
                points.len(),
                scalars.len()
            )));
        }
        let pipelines = G::pipelines(self);
        let pso = &pipelines.point_mul[window_index(G::window(self))];
        let mut out = Vec::with_capacity(points.len());

        for (pts, scs) in points.chunks(self.chunk).zip(scalars.chunks(self.chunk)) {
            let n = pts.len();
            let packed: Vec<G::PackedPoint> = pts.iter().map(G::pack_point).collect();
            let in_buf = self.buffer(&packed);
            let sc_buf = self.buffer(&PackedScalar::pack_slice(scs));
            let out_buf = self.scratch::<G::PackedPoint>(n);
            let p = CerParams {
                n: n as u32,
                ..Default::default()
            };

            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(&in_buf), 0);
            enc.set_buffer(1, Some(&sc_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            set_params(enc, 3, &p);
            dispatch_1d(enc, pso, n, 64);
            enc.end_encoding();
            cb.commit();
            crate::cb::wait_ok(cb, "ceremony point scalar multiplication")?;

            // SAFETY: the command buffer completed, and `n` points of this type is
            // exactly what the kernel wrote.
            let got: &[G::PackedPoint] = unsafe { cer_read_back(&out_buf, n) };
            out.extend(got.iter().map(G::unpack_point));
        }
        Ok(out)
    }

    fn batch_to_affine<G: CerGroup>(
        &self,
        points: &[Xyzz<G::Raw>],
    ) -> Result<Vec<G::Affine>, ProveError> {
        let mut out = Vec::with_capacity(points.len());
        for pts in points.chunks(self.chunk) {
            let n = pts.len();
            let packed: Vec<G::PackedPoint> = pts.iter().map(G::pack_point).collect();
            let in_buf = self.buffer(&packed);
            let aff = self.scratch::<G::PackedAffine>(n);
            self.affine_from_device::<G>(&in_buf, &aff, n, None)?;
            // SAFETY: as above, and `PackedAffine` is `Packed`, so every bit pattern is
            // a valid value.
            let got: &[G::PackedAffine] = unsafe { cer_read_back(&aff, n) };
            out.extend(got.iter().map(G::unpack_affine));
        }
        Ok(out)
    }

    fn apply_key<G: CerGroup>(
        &self,
        points: &mut [G::Affine],
        first: Fr,
        inc: Fr,
    ) -> Result<(), ProveError> {
        let pipelines = G::pipelines(self);
        let pso = &pipelines.apply_key[window_index(G::window(self))];
        let chunk = self.chunk;

        for (ci, block) in points.chunks_mut(chunk).enumerate() {
            let n = block.len();
            let packed: Vec<G::PackedAffine> = block.iter().map(G::pack_affine).collect();
            let in_buf = self.buffer(&packed);
            let xyzz = self.scratch::<G::PackedPoint>(n);
            let aff = self.scratch::<G::PackedAffine>(n);
            let p = CerParams {
                n: n as u32,
                index_off: (ci * chunk) as u32,
                has_inc: u32::from(inc != Fr::ONE),
                first: PackedFr::from_fr(&first),
                inc: PackedFr::from_fr(&inc),
                ..Default::default()
            };

            // The ladder and the first batch-to-affine pass go into one command buffer:
            // dispatches inside a serial compute encoder are ordered and coherent, so the
            // prefix pass reads what the ladder wrote with no barrier and no second
            // submission.
            let ladder = |enc: &ComputeCommandEncoderRef| {
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&in_buf), 0);
                enc.set_buffer(1, Some(&xyzz), 0);
                set_params(enc, 2, &p);
                dispatch_1d(enc, pso, n, 64);
            };
            self.affine_from_device::<G>(&xyzz, &aff, n, Some(&ladder))?;

            // SAFETY: the command buffer completed and `n` affine points is what the
            // finish pass wrote.
            let got: &[G::PackedAffine] = unsafe { cer_read_back(&aff, n) };
            for (dst, src) in block.iter_mut().zip(got) {
                *dst = G::unpack_affine(src);
            }
        }
        Ok(())
    }

    /// The two batch-to-affine passes over a device-resident XYZZ buffer, with the one
    /// host inversion between them.
    ///
    /// `pre` is encoded into the first command buffer ahead of the prefix pass, so a
    /// caller that produced `pts` on the device pays one submission instead of two.
    fn affine_from_device<G: CerGroup>(
        &self,
        pts: &Buffer,
        out: &Buffer,
        n: usize,
        pre: Option<&dyn Fn(&ComputeCommandEncoderRef)>,
    ) -> Result<(), ProveError> {
        if n == 0 {
            return Ok(());
        }
        let pipelines = G::pipelines(self);
        let segments = n.div_ceil(SEG_LEN);
        let prefix = self.scratch::<G::PackedField>(n);
        let segprod = self.scratch::<G::PackedField>(segments);
        let p = CerParams {
            n: n as u32,
            seg_len: SEG_LEN as u32,
            segments: segments as u32,
            ..Default::default()
        };

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        if let Some(pre) = pre {
            pre(enc);
        }
        enc.set_compute_pipeline_state(&pipelines.affine_prefix);
        enc.set_buffer(0, Some(pts), 0);
        enc.set_buffer(1, Some(&prefix), 0);
        enc.set_buffer(2, Some(&segprod), 0);
        set_params(enc, 3, &p);
        dispatch_1d(enc, &pipelines.affine_prefix, segments, 64);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "ceremony batch-to-affine, prefix pass")?;

        // SAFETY: the command buffer completed and the prefix pass wrote one field
        // element per segment.
        let got: &[G::PackedField] = unsafe { cer_read_back(&segprod, segments) };
        let seeds = self.buffer(&build_seeds::<G>(got)?);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipelines.affine_finish);
        enc.set_buffer(0, Some(pts), 0);
        enc.set_buffer(1, Some(&prefix), 0);
        enc.set_buffer(2, Some(&seeds), 0);
        enc.set_buffer(3, Some(out), 0);
        set_params(enc, 4, &p);
        dispatch_1d(enc, &pipelines.affine_finish, segments, 64);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "ceremony batch-to-affine, finish pass")?;
        Ok(())
    }
}

/// The inverse of each segment's own accumulator, from ONE inversion.
///
/// Montgomery's trick again, one level up. `cer_affine_prefix_*` writes a SEGMENT-LOCAL
/// running product, so the value its backward pass must start from is that segment's own
/// product inverted, not the inverse of everything before it: seeding with the global
/// prefix leaves every segment after the first wrong by the product of the ones ahead of
/// it, which is the shape of bug this comment exists to stop coming back.
///
/// So the host runs the same prefix, one inversion, suffix walk that
/// `batch_to_affine_serial` (prepare.rs:198) runs over points, over `segments` values
/// instead. That is why no shader in this crate needs an `f_inv`.
fn build_seeds<G: CerGroup>(segprod: &[G::PackedField]) -> Result<Vec<G::PackedField>, ProveError> {
    let mut prefix = Vec::with_capacity(segprod.len());
    let mut acc = G::Field::ONE;
    for s in segprod {
        prefix.push(acc);
        acc *= G::field(s);
    }
    // Every factor is a product of nonzero `ZZ*ZZZ`, or ONE for a segment that was all
    // points at infinity, so this cannot fail on well-formed input. It can fail on an
    // XYZZ point with `ZZ != 0` and `ZZZ == 0`, which is off the curve; refusing beats
    // writing an affine point that is silently zero.
    let mut inv = acc.inverse().ok_or_else(|| {
        cer_err(format!(
            "batch-to-affine over {}: the product of the segment accumulators is zero, \
             so an input point had ZZ != 0 with ZZ*ZZZ == 0",
            G::OP
        ))
    })?;
    let mut seeds = vec![G::PackedField::default(); segprod.len()];
    for j in (0..segprod.len()).rev() {
        seeds[j] = G::pack_field(&(inv * prefix[j]));
        inv *= G::field(&segprod[j]);
    }
    Ok(seeds)
}

/// `G16_METAL_CER_C_G1` / `_G2` override the measured window, for a sweep. An
/// unparseable or uncompiled value falls back to the default rather than failing, because
/// the only caller is a measurement and a typo that silently picked a different width
/// would be reported as a result.
pub(crate) fn env_window(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|c| CER_WINDOWS.contains(c))
        .unwrap_or(default)
}

/// `G16_METAL_KEY_MIN` overrides the apply-key crossover, for the same sweep and on the
/// same terms as [`env_window`].
fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// `setBytes` for a params struct. `CerParams` is `repr(C)` and all-`u32`, so its bytes
/// are exactly what the shader's `constant CerParams&` reads.
fn set_params(enc: &ComputeCommandEncoderRef, index: u64, p: &CerParams) {
    enc.set_bytes(
        index,
        core::mem::size_of::<CerParams>() as u64,
        (p as *const CerParams).cast(),
    );
}

/// A copy of `msm::dispatch_1d`, which is private to that module. Same contract: the
/// preferred threadgroup size is clamped to what the pipeline reports, and the kernel
/// bounds-checks its own index because `dispatch_threads` rounds the grid up.
pub(crate) fn dispatch_1d(
    enc: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    n: usize,
    prefer: usize,
) {
    if n == 0 {
        return;
    }
    let tg = prefer
        .min(pso.max_total_threads_per_threadgroup() as usize)
        .max(1);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
}

/// # Safety
///
/// The buffer must hold at least `len` `T`s written by a completed command buffer, and
/// `T` must be a `Packed` type, so every bit pattern is valid.
pub(crate) unsafe fn cer_read_back<T: Packed>(buf: &Buffer, len: usize) -> &[T] {
    core::slice::from_raw_parts(buf.contents().cast::<T>(), len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g16_ceremony::prepare::{batch_to_affine, point_times_fr};
    use g16_ceremony::CpuKeyScale;

    /// Bits the recoding covers. Same value and same reason as `msm::RECODE_BITS`, which
    /// is private to that module: `g16_msm::RECODE_BITS` is private to its crate too, so
    /// the number lives in three places and every one of them is guarded by a test.
    const RECODE_BITS: usize = 255;

    /// The ladder decomposes a scalar with `sc_signed_digit` from `msm.metal`, whose digit
    /// count comes from this constant. If the shader's copy drifts below the CPU's the top
    /// window is dropped, which is invisible for every scalar under `2^((nw-1)*c)` and
    /// wrong for every scalar above it, so a random test with small scalars passes.
    #[test]
    fn msl_declares_the_same_recode_bits() {
        let line = format!("#define CER_RECODE_BITS {RECODE_BITS}");
        assert!(
            CEREMONY_MSL.contains(&line),
            "shaders/ceremony.metal does not contain the line:\n{line}"
        );
    }

    /// Every width in [`CER_WINDOWS`] has a kernel pair, and the two defaults are among
    /// them. Construction would catch a missing kernel, but only on a machine with a
    /// device; this runs anywhere.
    #[test]
    fn every_compiled_window_has_a_kernel_pair() {
        for c in CER_WINDOWS {
            for g in ["g1", "g2"] {
                assert!(
                    CEREMONY_MSL.contains(&format!("CER_LADDER_KERNELS({g}_c{c},")),
                    "shaders/ceremony.metal has no CER_LADDER_KERNELS line for {g} c={c}, \
                     so cer_point_mul_{g}_c{c} does not exist and the pipeline will not build"
                );
            }
        }
        assert!(CER_WINDOWS.contains(&WINDOW_G1));
        assert!(CER_WINDOWS.contains(&WINDOW_G2));
    }

    /// Deterministic full-width `Fr`, so a failure is reproducible without a seed to
    /// record. Three 64-bit draws combined multiplicatively, which reaches the top of the
    /// field rather than the bottom 64 bits `Fr::from(u64)` alone would give.
    struct Lcg(u64);

    impl Lcg {
        fn word(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn fr(&mut self) -> Fr {
            let a = Fr::from(self.word());
            let b = Fr::from(self.word());
            let c = Fr::from(self.word());
            a * b + c
        }
    }

    /// An XYZZ representative of `[k] G` with a deliberately non-trivial `ZZ`.
    ///
    /// `ZZ = u^2`, `ZZZ = u^3` keeps the invariant `ZZ^3 = ZZZ^2`, so the kernels see the
    /// same spread of representatives the group FFT hands them rather than a vector that
    /// is entirely `ZZ == 1`. A ladder that reads `zz` where it means `zzz` is exact on
    /// the latter and wrong on the former.
    fn scaled_g1(k: Fr, u: u64) -> Xyzz<RawFq> {
        let p = (G1Affine::generator() * k).into_affine();
        let u = RawFq::from_fq(&Fq::from(u | 1));
        let zz = u.sqr();
        let zzz = zz.mul(u);
        Xyzz {
            x: RawFq::from_fq(&p.x).mul(zz),
            y: RawFq::from_fq(&p.y).mul(zzz),
            zz,
            zzz,
        }
    }

    fn scaled_g2(k: Fr, u: u64) -> Xyzz<RawFq2> {
        let p = (G2Affine::generator() * k).into_affine();
        let u = RawFq2 {
            c0: RawFq::from_fq(&Fq::from(u | 1)),
            c1: RawFq::ONE,
        };
        let zz = u.sqr();
        let zzz = zz.mul(u);
        Xyzz {
            x: RawFq2::from_fq2(&p.x).mul(zz),
            y: RawFq2::from_fq2(&p.y).mul(zzz),
            zz,
            zzz,
        }
    }

    /// The scalars every kernel test runs, corners first.
    ///
    /// 0 and 1 are not decoration: the recoding sends 1 to digit +1 of window 0 and 0 to
    /// no digit at all, and `r - 1` is the only scalar that reaches the top window's
    /// highest digit. `2` and `4` land on the even table entries, which are built by a
    /// doubling rather than an addition.
    fn corner_scalars(rng: &mut Lcg, extra: usize) -> Vec<Fr> {
        let mut v = vec![
            Fr::from(0u64),
            Fr::ONE,
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(0u64) - Fr::ONE,
            Fr::from(0u64) - Fr::from(2u64),
        ];
        for _ in 0..extra {
            v.push(rng.fr());
        }
        v
    }

    fn kernels() -> Option<CeremonyKernels> {
        // No device on a headless builder is a skip, not a failure, the same way the
        // artifact-gated tests in `msm.rs` skip.
        Device::system_default()?;
        Some(CeremonyKernels::new().expect("ceremony kernels"))
    }

    #[test]
    fn point_mul_g1_matches_the_cpu_ladder() {
        let Some(k) = kernels() else { return };
        // Small enough that the whole vector is one chunk, so this test is about the
        // ladder alone; `point_mul_crosses_the_chunk_boundary` covers the other half.
        let mut rng = Lcg(0x5eed_0001);
        let scalars = corner_scalars(&mut rng, 200);
        let mut pts: Vec<Xyzz<RawFq>> = Vec::with_capacity(scalars.len());
        for (i, _) in scalars.iter().enumerate() {
            // Every third point is the identity, which is a live input: ptau section 12's
            // `power+1` block is padded with it.
            pts.push(if i % 3 == 0 {
                Xyzz::ZERO
            } else {
                scaled_g1(rng.fr(), rng.word())
            });
        }

        let want: Vec<Xyzz<RawFq>> = pts
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        let got = k.point_mul_g1(&pts, &scalars).expect("point_mul_g1");

        // Affine is where the two representations have to agree, and it is the form every
        // ceremony path exits through, so this is the comparison the byte-identity claim
        // actually rests on.
        assert_eq!(
            batch_to_affine::<g16_field::g1::Config>(&got),
            batch_to_affine::<g16_field::g1::Config>(&want)
        );
    }

    #[test]
    fn point_mul_g2_matches_the_cpu_ladder() {
        let Some(k) = kernels() else { return };
        let mut rng = Lcg(0x5eed_0002);
        let scalars = corner_scalars(&mut rng, 120);
        let mut pts: Vec<Xyzz<RawFq2>> = Vec::with_capacity(scalars.len());
        for (i, _) in scalars.iter().enumerate() {
            pts.push(if i % 3 == 0 {
                Xyzz::ZERO
            } else {
                scaled_g2(rng.fr(), rng.word())
            });
        }

        let want: Vec<Xyzz<RawFq2>> = pts
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        let got = k.point_mul_g2(&pts, &scalars).expect("point_mul_g2");

        assert_eq!(
            batch_to_affine::<g16_field::g2::Config>(&got),
            batch_to_affine::<g16_field::g2::Config>(&want)
        );
    }

    /// Every compiled width answers the same, which is what makes the sweep in
    /// [`window_sweep`] a timing question rather than a correctness one.
    #[test]
    fn every_window_gives_the_same_points() {
        let Some(base) = kernels() else { return };
        let mut rng = Lcg(0x5eed_0003);
        let scalars = corner_scalars(&mut rng, 60);
        let pts: Vec<Xyzz<RawFq>> = scalars
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i % 5 == 0 {
                    Xyzz::ZERO
                } else {
                    scaled_g1(rng.fr(), rng.word())
                }
            })
            .collect();
        let want: Vec<Xyzz<RawFq>> = pts
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        let want = batch_to_affine::<g16_field::g1::Config>(&want);

        for c in CER_WINDOWS {
            let mut k = base_clone(&base);
            k.window_g1 = c;
            k.window_g2 = c;
            let got = k.point_mul_g1(&pts, &scalars).expect("point_mul_g1");
            assert_eq!(
                batch_to_affine::<g16_field::g1::Config>(&got),
                want,
                "G1 ladder disagrees with the CPU at window width {c}"
            );
        }
    }

    /// A second handle on the same device, so a test can vary the window without
    /// recompiling the library for every width.
    fn base_clone(k: &CeremonyKernels) -> CeremonyKernels {
        CeremonyKernels::with_device(k.device().clone()).expect("ceremony kernels")
    }

    #[test]
    fn batch_to_affine_matches_the_cpu() {
        let Some(k) = kernels() else { return };
        let mut rng = Lcg(0x5eed_0004);
        // 1000 points against a 97-point chunk and a 32-point segment: the tail is ragged
        // at both levels, which is where a seed built from the wrong suffix product or a
        // segment that runs past `n` shows up.
        let k = k.with_chunk(97);
        let mut g1: Vec<Xyzz<RawFq>> = Vec::new();
        let mut g2: Vec<Xyzz<RawFq2>> = Vec::new();
        for i in 0..1000 {
            // Runs of identities, including one long enough to fill a whole segment, so a
            // segment whose accumulator never leaves ONE is exercised.
            let inf = (32..80).contains(&i) || i % 7 == 0;
            g1.push(if inf {
                Xyzz::ZERO
            } else {
                scaled_g1(rng.fr(), rng.word())
            });
            g2.push(if inf {
                Xyzz::ZERO
            } else {
                scaled_g2(rng.fr(), rng.word())
            });
        }

        assert_eq!(
            k.batch_to_affine_g1(&g1).expect("batch_to_affine_g1"),
            batch_to_affine::<g16_field::g1::Config>(&g1)
        );
        assert_eq!(
            k.batch_to_affine_g2(&g2).expect("batch_to_affine_g2"),
            batch_to_affine::<g16_field::g2::Config>(&g2)
        );
    }

    /// `inc == 1`, the constant-scalar case both zkey commands use, and the geometric one
    /// phase 1 uses, over a vector that crosses the chunk boundary several times.
    ///
    /// The chunk boundary is the whole point: the geometric key is `first * inc^i` in the
    /// index of the WHOLE array, and a chunk that recomputes from its own local index
    /// gives the right answer for chunk 0 and the wrong one for every chunk after it.
    #[test]
    fn apply_key_matches_the_cpu() {
        let Some(k) = kernels() else { return };
        let k = k.with_chunk(97);
        let mut rng = Lcg(0x5eed_0005);
        let first = rng.fr();
        let inc = rng.fr();

        let mut g1: Vec<G1Affine> = Vec::new();
        let mut g2: Vec<G2Affine> = Vec::new();
        for i in 0..500 {
            if i % 11 == 0 {
                g1.push(G1Affine::identity());
                g2.push(G2Affine::identity());
            } else {
                g1.push((G1Affine::generator() * rng.fr()).into_affine());
                g2.push((G2Affine::generator() * rng.fr()).into_affine());
            }
        }

        for (first, inc, what) in [
            (first, inc, "geometric"),
            (first, Fr::ONE, "constant"),
            (Fr::ONE, Fr::ONE, "identity key"),
            (Fr::from(0u64), inc, "zero key"),
            (Fr::from(0u64) - Fr::ONE, inc, "maximal first"),
        ] {
            let mut want1 = g1.clone();
            let mut got1 = g1.clone();
            CpuKeyScale
                .apply_key_g1(&mut want1, first, inc)
                .expect("cpu apply_key_g1");
            k.apply_key_g1(&mut got1, first, inc).expect("apply_key_g1");
            assert_eq!(got1, want1, "G1 apply_key disagrees, {what}");

            let mut want2 = g2.clone();
            let mut got2 = g2.clone();
            CpuKeyScale
                .apply_key_g2(&mut want2, first, inc)
                .expect("cpu apply_key_g2");
            k.apply_key_g2(&mut got2, first, inc).expect("apply_key_g2");
            assert_eq!(got2, want2, "G2 apply_key disagrees, {what}");
        }
    }

    /// The ladder over a vector long enough to cross the chunk boundary, with the corner
    /// scalars deliberately placed in the second and third chunks rather than the first.
    #[test]
    fn point_mul_crosses_the_chunk_boundary() {
        let Some(k) = kernels() else { return };
        let k = k.with_chunk(97);
        let mut rng = Lcg(0x5eed_0006);
        let corners = corner_scalars(&mut rng, 0);
        let mut scalars: Vec<Fr> = (0..400).map(|_| rng.fr()).collect();
        for (i, s) in corners.iter().enumerate() {
            scalars[150 + i] = *s;
            scalars[250 + i] = *s;
        }
        let pts: Vec<Xyzz<RawFq>> = (0..scalars.len())
            .map(|i| {
                if i % 13 == 0 {
                    Xyzz::ZERO
                } else {
                    scaled_g1(rng.fr(), rng.word())
                }
            })
            .collect();

        let want: Vec<Xyzz<RawFq>> = pts
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        let got = k.point_mul_g1(&pts, &scalars).expect("point_mul_g1");
        assert_eq!(
            batch_to_affine::<g16_field::g1::Config>(&got),
            batch_to_affine::<g16_field::g1::Config>(&want)
        );
    }

    /// The empty and single-point vectors, which is where a `div_ceil` on zero or a
    /// segment seed for no segments would panic rather than answer.
    #[test]
    fn degenerate_lengths() {
        let Some(k) = kernels() else { return };
        assert!(k
            .point_mul_g1(&[], &[])
            .expect("empty point_mul")
            .is_empty());
        assert!(k
            .batch_to_affine_g1(&[])
            .expect("empty to_affine")
            .is_empty());
        let mut none: Vec<G1Affine> = Vec::new();
        k.apply_key_g1(&mut none, Fr::ONE, Fr::ONE)
            .expect("empty apply_key");

        let one = vec![Xyzz::<RawFq>::ZERO];
        assert_eq!(
            k.batch_to_affine_g1(&one).expect("one to_affine"),
            batch_to_affine::<g16_field::g1::Config>(&one)
        );
    }

    /// The window sweep behind [`WINDOW_G1`] and [`WINDOW_G2`]. Not a correctness test:
    /// run it with `--ignored --nocapture` and read the numbers.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn window_sweep() {
        let Some(base) = kernels() else { return };
        let mut rng = Lcg(0xbeef);
        const N: usize = 1 << 16;
        let scalars: Vec<Fr> = (0..N).map(|_| rng.fr()).collect();
        let g1: Vec<Xyzz<RawFq>> = (0..N).map(|_| scaled_g1(rng.fr(), rng.word())).collect();
        let g2: Vec<Xyzz<RawFq2>> = (0..N).map(|_| scaled_g2(rng.fr(), rng.word())).collect();

        // Fq multiplies per ladder call, from the shader's own operation counts: a G1
        // doubling is 9 and a general addition 14, and an Fq2 multiply is 3 Fq ones
        // (Karatsuba) with a square 2, so the G2 stanzas cost 24 and 40.
        let muls = |c: u32, dbl: u64, add: u64| -> u64 {
            let nw = (RECODE_BITS as u64).div_ceil(c as u64);
            let tbl = 1u64 << (c - 1);
            nw * c as u64 * dbl + nw * add + (tbl / 2) * dbl + (tbl / 2).saturating_sub(1) * add
        };

        for c in CER_WINDOWS {
            let mut k = base_clone(&base);
            k.window_g1 = c;
            k.window_g2 = c;
            let t = std::time::Instant::now();
            k.point_mul_g1(&g1, &scalars).expect("g1");
            let e1 = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            k.point_mul_g2(&g2, &scalars).expect("g2");
            let e2 = t.elapsed().as_secs_f64();
            let r1 = muls(c, 9, 14) * N as u64;
            let r2 = muls(c, 24, 40) * N as u64;
            println!(
                "c={c}  G1 {:7.1} ms  {:5.2} G mul/s   |   G2 {:7.1} ms  {:5.2} G mul/s",
                e1 * 1e3,
                r1 as f64 / e1 / 1e9,
                e2 * 1e3,
                r2 as f64 / e2 / 1e9,
            );
        }

        // The same vectors through the CPU ladder, single threaded, for the ratio. The
        // twelve-thread figure the brief quotes is this over 12, which is generous to the
        // CPU: `point_times_fr` is called from a rayon loop that also walks the twiddles.
        let t = std::time::Instant::now();
        let _: Vec<Xyzz<RawFq>> = g1
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        println!("cpu G1 serial {:7.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let t = std::time::Instant::now();
        let _: Vec<Xyzz<RawFq2>> = g2
            .iter()
            .zip(&scalars)
            .map(|(p, s)| point_times_fr(p, s))
            .collect();
        println!("cpu G2 serial {:7.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
