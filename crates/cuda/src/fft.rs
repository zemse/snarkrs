//! The group inverse FFT behind `ptau prepare`, on CUDA.
//!
//! Host twin of `crates/metal/src/fft.rs`, driving `kernels/fft.cu`. The transform,
//! the twiddle tables and the two permutations that are deliberately not kernels are the
//! same; what this file owns is the CUDA shape of the submission, and it is simpler than
//! the Metal one for reasons that are that backend's, not this one's:
//!
//! * **No ladder budget, and no retry of a driver failure.** Both exist on Metal to
//!   survive macOS killing a busy command buffer with
//!   `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, a display-contention
//!   behaviour with no analogue on a headless compute card. A pass here is one launch
//!   however large it is, `gid_off` stays zero, and a driver failure is an error rather
//!   than something to sleep and retry: the codes that are device faults leave the
//!   context unusable (`context::is_device_fault`). What is re-run is a readback that
//!   fails the value checks below, the one fault the driver does not report.
//! * **One synchronize, at the read back.** The Metal host batches passes into command
//!   buffers and waits on each; launches on one CUDA stream queue asynchronously and
//!   execute in issue order, which is exactly the pass-after-pass dependency the
//!   transform needs, so the rounds are ordered for free and the host never blocks until
//!   the download. Same argument as the header of `msm.rs`: the thing to avoid is a
//!   synchronize, and there is none in the loop.
//! * **Every readback is checked, as on Metal (`check.rs`), with no wait per pass.**
//!   Metal probes one thread of every piece after the piece's wait; here nothing waits
//!   before the download, so as each pass is launched the probed thread's source and
//!   destination points are copied on the same stream into a small per-block log,
//!   behind the launch and ahead of the pass that overwrites the source. The host then
//!   probes every pass from the one download at the end and scans every finished block
//!   on the curve. A refused readback re-runs the whole transform from the caller's
//!   slice, which nothing has written yet, up to [`RETRIES`] times; a persistent one
//!   fails the call with the caller's points as they were.
//! * **The pack lands in a host `Vec` and crosses PCIe.** Metal writes the bit-reversed
//!   points straight into shared memory. Here the copy is real but small next to the
//!   arithmetic: a pass moves `n` points each way once per transform, against ~3,000 Fq
//!   multiplies per ladder per pass.
//!
//! Co-dispatch note: `ifft_many` keeps the Metal lockstep rounds, and by default the
//! launches within a round still serialize on the one stream, so a round of small
//! blocks does not fill the device the way Metal's `Concurrent` encoder does.
//! `G16_CUDA_FFT_STREAMS=N` (or [`FftKernels::with_streams`]) round-robins the blocks
//! over N streams instead; 1 stays the default until a measurement on a real card says
//! what the fill is worth.
//!
//! The experiment knobs, all run-time and none costing a recompile:
//! `G16_CUDA_FFT_VARIANT` picks a kernel variant ([`VARIANTS`]), `G16_CUDA_FFT_BLOCK`
//! the threads per block, `G16_CUDA_FFT_STREAMS` the stream count, and
//! `G16_CUDA_FFT_TIME=1` wraps every launch in CUDA events and prints per-pass device
//! times. The one thing that does cost a compile is `G16_CUDA_FFT_VARIANTS=1`, which
//! instantiates the whole variant set into the unit, once.
//!
//! The result agrees with the CPU as a curve point, not limb for limb, and the
//! ceremony's byte identity survives because `lagrange_evaluations` (prepare.rs:414)
//! ends every path in `batch_to_affine`, and affine is canonical. So `cmp` of a
//! `--backend cpu` output against a `--backend cuda` one is a total check on this file.

use std::sync::Arc;

use ark_ec::scalar_mul::glv::GLVConfig;
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaModule, CudaSlice, CudaStream, DeviceRepr, PushKernelArg,
};
use rayon::prelude::*;
use snarkrs_field::raw::{RawField, RawFq, RawFq2};
use snarkrs_field::{FftField, Field, Fr};
use snarkrs_gpu_layout::glv::twiddle_table;
use snarkrs_gpu_layout::{Packed, PackedFq, PackedFq2};
use snarkrs_groth16::ProveError;
use snarkrs_msm::glv;
use snarkrs_msm::xyzz::Xyzz;

use crate::check::{self, CurveField, Reduced};
use crate::msm::{bad, download, drv, upload_words, Kernel, PackedXyzzG1, PackedXyzzG2};
use crate::{as_words, from_words, kernels, Cuda};

/// Ladder window of the shipped variant, for both groups.
///
/// 5 is the Metal sweep's answer for both groups over a whole `ppot_0080_16.ptau`
/// prepare (`crates/metal/src/fft.rs`, `FFT_WINDOW_G1`), carried over rather than re-swept:
/// there is no NVIDIA sweep yet. The sweep exists as [`VARIANTS`] behind
/// `G16_CUDA_FFT_VARIANTS=1`, one compile for the whole set; this constant names the
/// default until that sweep says otherwise.
const FFT_WINDOW: u32 = 5;

/// Blocks shorter than this go back to the CPU.
///
/// Metal's number (`crates/metal/src/fft.rs`, `MIN_BLOCK`), carried over unswept. The
/// overhead it hides from is smaller here: Metal's 0.149 ms commit-and-wait floor is a
/// 0.006 ms launch on CUDA (`bench/results/device-microbench.md`), so the true crossover
/// sits lower and only a sweep on a real card can say where. `G16_CUDA_FFT_MIN_BLOCK`
/// moves it without a rebuild, and the threshold's real job is unchanged: the blocks
/// under it are the same short handful at every power, and `process_section`
/// (prepare.rs:673) runs them on their own worker thread concurrent with the device.
const MIN_BLOCK: usize = 1 << 12;

/// Threads per block. The ladder is one long dependent chain per thread, the same shape
/// as the MSM point kernels, so the same choice as `msm.rs`'s `POINT_BLOCK` and the same
/// caveat: not swept on a real card. `G16_CUDA_FFT_BLOCK` overrides it without a
/// rebuild; the launch clamps to the kernel's own maximum, which matters for the
/// `__launch_bounds__` variants.
const FFT_BLOCK: u32 = 128;

/// The kernel variants `kernels/fft.cu` can instantiate, in the order they appear
/// there. Row 0 is the shipped configuration and always exists; the `gated` rows exist
/// only when the unit was assembled with `G16_FFT_VARIANTS`
/// (`G16_CUDA_FFT_VARIANTS=1`), which is what makes one NVRTC compile carry the whole
/// sweep. `G16_CUDA_FFT_VARIANT=<name>` selects one per run; the hypothesis each row
/// tests is documented at the instantiation site in `fft.cu`.
struct VariantDef {
    name: &'static str,
    mix_g1: &'static str,
    scale_g1: &'static str,
    mix_g2: &'static str,
    scale_g2: &'static str,
    gated: bool,
}

const VARIANTS: [VariantDef; 4] = [
    VariantDef {
        name: "c5",
        mix_g1: "fft_mix_g1_c5",
        scale_g1: "fft_mix_scale_g1_c5",
        mix_g2: "fft_mix_g2_c5",
        scale_g2: "fft_mix_scale_g2_c5",
        gated: false,
    },
    VariantDef {
        name: "c4",
        mix_g1: "fft_mix_g1_c4",
        scale_g1: "fft_mix_scale_g1_c4",
        mix_g2: "fft_mix_g2_c4",
        scale_g2: "fft_mix_scale_g2_c4",
        gated: true,
    },
    VariantDef {
        name: "c3r",
        mix_g1: "fft_mix_g1_c3r",
        scale_g1: "fft_mix_scale_g1_c3r",
        mix_g2: "fft_mix_g2_c3r",
        scale_g2: "fft_mix_scale_g2_c3r",
        gated: true,
    },
    VariantDef {
        name: "c5r128",
        mix_g1: "fft_mix_g1_c5r128",
        scale_g1: "fft_mix_scale_g1_c5r128",
        mix_g2: "fft_mix_g2_c5r128",
        scale_g2: "fft_mix_scale_g2_c5r128",
        gated: true,
    },
];

/// Mirrors `struct FftParams` in `kernels/fft.cu`: five `u32`, 20 bytes, no padding on
/// either side. Passed by value through the parameter space, like `MsmParams`.
///
/// `gid_off` is always zero here. It exists in the kernel because the Metal host splits
/// a pass across command buffers to bound what a macOS kill throws away; this host has
/// nothing to bound, and dropping the field would fork the kernel source for nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FftParams {
    n: u32,
    span: u32,
    log_span: u32,
    tw_shift: u32,
    gid_off: u32,
}

const _: () = {
    assert!(core::mem::size_of::<FftParams>() == 20);
    assert!(core::mem::align_of::<FftParams>() == 4);
};

// SAFETY: `repr(C)`, five `u32`, no padding, no invalid bit patterns, and the size is
// asserted above to be the 20 bytes the kernel's parameter slot expects.
unsafe impl DeviceRepr for FftParams {}

/// One block's live state across the lockstep rounds of [`FftKernels::ifft_many`]: its
/// ping-pong pair, its own `1/n`-scaled twiddle table, the log of its probed threads,
/// the stream its passes run on, and where in the caller's slice of blocks it came from.
struct Block {
    idx: usize,
    n: usize,
    bits: u32,
    size_inv: Fr,
    src: CudaSlice<u32>,
    dst: CudaSlice<u32>,
    stw: CudaSlice<u32>,
    /// [`LOG_POINTS`] slots per pass, filled by [`log_probe`] and read by [`probe`].
    log: CudaSlice<u32>,
    probes: Vec<Probe>,
    stream: Arc<CudaStream>,
}

/// One butterfly of a pass's probed thread, held on the host to the pass's own source;
/// see [`probe`].
struct Probe {
    /// The point it reads and writes at the low index, and at the high one.
    lo: usize,
    hi: usize,
    /// The twiddle on the high input, `None` for the `j == 0` butterflies.
    w: Option<Fr>,
    /// `1/n`, which the block's fused pass puts on both inputs.
    scale: Option<Fr>,
    /// Where its four points sit in the block's log, in points: `dst[lo]`, `dst[hi]`,
    /// `src[lo]`, `src[hi]`.
    slot: usize,
}

/// Points the log holds per pass: two butterflies of four, the most a thread has.
const LOG_POINTS: usize = 8;

/// Attempts at the transform before giving up, counting the first, and the backoff
/// before each re-run: Metal's (`crates/metal/src/cb.rs`, `RETRIES` and `backoff`),
/// carried over rather than re-derived. Nothing on a headless card is known to need the
/// wait; a wrong store from a fault the driver did not report is given the same time to
/// clear, 9.2 s over the eight attempts.
const RETRIES: u32 = 8;

fn backoff(attempt: u32) -> std::time::Duration {
    std::time::Duration::from_millis(200 << attempt.min(3))
}

const GROUP_IFFT: &str = "ceremony group inverse fft";

/// How an attempt at the transform ended short of its points.
enum Attempt {
    /// A readback failed a check: the transform is re-run from the unchanged inputs.
    Refused(ProveError),
    /// The driver failed, or a readback had the wrong shape: no re-run.
    Failed(ProveError),
}

impl From<ProveError> for Attempt {
    fn from(e: ProveError) -> Self {
        Attempt::Failed(e)
    }
}

/// Blocks the host until every stream of a multi-stream round has drained, however the
/// round ends.
///
/// The happy path does not need this: each block's `download` synchronizes its own stream
/// before anything is freed. The error paths do. `ifft_many` has `?` exits after launches
/// have been issued, and one of the buffers those launches read is the shared `tw`, which
/// lives on `self.stream` rather than on any pool stream. cudarc frees a `CudaSlice` with
/// `cuMemFreeAsync` on the stream it was allocated against, and that only orders against
/// work already queued on *that* stream, so an early return would free the twiddle table
/// out from under mix passes still queued elsewhere. A device-side use-after-free reads as
/// corrupt points or as an illegal address that takes the context down, either of which is
/// worse than the error being reported.
struct DrainOnExit<'a>(&'a [Arc<CudaStream>]);

impl Drop for DrainOnExit<'_> {
    fn drop(&mut self) {
        for s in self.0 {
            let _ = s.synchronize();
        }
    }
}

/// The two kernels for one group of one variant.
struct FftPair {
    mix: Kernel,
    mix_scale: Kernel,
}

/// One loaded variant: both groups' pairs under the name the environment selects by.
struct Variant {
    name: &'static str,
    g1: FftPair,
    g2: FftPair,
}

/// An event pair around one launch, read back after the stream drains. Only allocated
/// when timing is on; the launch path stays event-free otherwise.
struct PassTime {
    idx: usize,
    bits: u32,
    exp: u32,
    fused: bool,
    threads: usize,
    start: CudaEvent,
    end: CudaEvent,
}

/// What the two kernels need to know about a group, so the submission code is written
/// once. The smaller CUDA twin of `crates/metal/src/fft.rs`'s trait of the same name.
trait FftGroup {
    type Raw: CurveField;
    type PackedPoint: Packed + Default + Reduced + Send + Sync;
    /// The curve whose GLV lattice a twiddle of this group decomposes against. The two
    /// groups share one `beta` and take different eigenvalues, so they take different
    /// lattices too; see `snarkrs_gpu_layout::glv`.
    type Cfg: GLVConfig<ScalarField = Fr>;

    /// Names this group in an error, in the spelling `snarkrs_msm::AccelError` uses.
    const OP: &'static str;

    fn pack(p: &Xyzz<Self::Raw>) -> Self::PackedPoint;
    fn unpack(p: &Self::PackedPoint) -> Xyzz<Self::Raw>;
    fn kernels(k: &FftKernels) -> &FftPair;
}

struct FftG1;
struct FftG2;

impl FftGroup for FftG1 {
    type Raw = RawFq;
    type PackedPoint = PackedXyzzG1;
    type Cfg = snarkrs_field::g1::Config;

    const OP: &'static str = "G1";

    fn pack(p: &Xyzz<RawFq>) -> PackedXyzzG1 {
        PackedXyzzG1 {
            x: PackedFq::from_fq(&p.x.to_fq()),
            y: PackedFq::from_fq(&p.y.to_fq()),
            zz: PackedFq::from_fq(&p.zz.to_fq()),
            zzz: PackedFq::from_fq(&p.zzz.to_fq()),
        }
    }

    fn unpack(p: &PackedXyzzG1) -> Xyzz<RawFq> {
        Xyzz {
            x: RawFq::from_fq(&p.x.to_fq()),
            y: RawFq::from_fq(&p.y.to_fq()),
            zz: RawFq::from_fq(&p.zz.to_fq()),
            zzz: RawFq::from_fq(&p.zzz.to_fq()),
        }
    }

    fn kernels(k: &FftKernels) -> &FftPair {
        &k.variants[k.active].g1
    }
}

impl FftGroup for FftG2 {
    type Raw = RawFq2;
    type PackedPoint = PackedXyzzG2;
    type Cfg = snarkrs_field::g2::Config;

    const OP: &'static str = "G2";

    fn pack(p: &Xyzz<RawFq2>) -> PackedXyzzG2 {
        PackedXyzzG2 {
            x: PackedFq2::from_fq2(&p.x.to_fq2()),
            y: PackedFq2::from_fq2(&p.y.to_fq2()),
            zz: PackedFq2::from_fq2(&p.zz.to_fq2()),
            zzz: PackedFq2::from_fq2(&p.zzz.to_fq2()),
        }
    }

    fn unpack(p: &PackedXyzzG2) -> Xyzz<RawFq2> {
        Xyzz {
            x: RawFq2::from_fq2(&p.x.to_fq2()),
            y: RawFq2::from_fq2(&p.y.to_fq2()),
            zz: RawFq2::from_fq2(&p.zz.to_fq2()),
            zzz: RawFq2::from_fq2(&p.zzz.to_fq2()),
        }
    }

    fn kernels(k: &FftKernels) -> &FftPair {
        &k.variants[k.active].g2
    }
}

/// The shipped variant's four kernels, plus the gated experiment set when the unit carries
/// it, compiled once.
///
/// Compiling the unit is the expensive call (NVRTC plus the driver's ptxas on a miss of
/// both caches; see `context.rs` for the measured cliff) and it belongs once at the top
/// of a command, exactly as `CudaMsm::compile` does for the prover.
pub struct FftKernels {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Kept alive explicitly. `CudaFunction` holds an `Arc<CudaModule>` internally, so
    /// this field documents the ownership rather than being what keeps the module loaded.
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    variants: Vec<Variant>,
    /// Whether the gated rows were loaded, so a refused name can say why.
    gated: bool,
    active: usize,
    min_block: usize,
    block: u32,
    streams: usize,
    timing: bool,
}

impl FftKernels {
    /// Compiles `kernels::unit_fft()` and binds the shipped variant's four kernel handles,
    /// plus the gated experiment set when the unit carries it.
    pub fn new(cuda: &Cuda) -> Result<Self, ProveError> {
        let module = Self::compile(cuda)?;
        Self::from_module(cuda, module)
    }

    /// Compile the FFT translation unit on its own, so a caller holding several
    /// commands' worth of work pays NVRTC once. Same split as `CudaMsm::compile`.
    pub fn compile(cuda: &Cuda) -> Result<Arc<CudaModule>, ProveError> {
        Self::compile_with(cuda, kernels::fft_variants_enabled())
    }

    /// [`Self::compile`] with the gated experiment set chosen by the caller instead of
    /// `G16_CUDA_FFT_VARIANTS`. Bind the result with [`Self::from_module_with`] and the
    /// same `variants`.
    pub fn compile_with(cuda: &Cuda, variants: bool) -> Result<Arc<CudaModule>, ProveError> {
        cuda.compile("fft", &kernels::unit_fft_with(variants))
            .map_err(ProveError::from)
    }

    /// Bind the kernel handles out of an already-compiled module. `module` must have
    /// come from [`Self::compile`] against this same [`Cuda`], for the reason
    /// `CudaMsm::from_module` gives.
    ///
    /// Loads every variant the unit was assembled with: the shipped row always, the
    /// gated rows exactly when `G16_CUDA_FFT_VARIANTS=1`, because that same env decided
    /// whether their entry points exist in the module at all (`kernels::defines`).
    pub fn from_module(cuda: &Cuda, module: Arc<CudaModule>) -> Result<Self, ProveError> {
        Self::from_module_with(cuda, module, kernels::fft_variants_enabled())
    }

    /// [`Self::from_module`] for a module from [`Self::compile_with`]: `gated` must be the
    /// `variants` it was compiled with.
    pub fn from_module_with(
        cuda: &Cuda,
        module: Arc<CudaModule>,
        gated: bool,
    ) -> Result<Self, ProveError> {
        let mut variants = Vec::new();
        for def in VARIANTS.iter().filter(|d| !d.gated || gated) {
            variants.push(Variant {
                name: def.name,
                g1: FftPair {
                    mix: Kernel::load(&module, def.mix_g1)?,
                    mix_scale: Kernel::load(&module, def.scale_g1)?,
                },
                g2: FftPair {
                    mix: Kernel::load(&module, def.mix_g2)?,
                    mix_scale: Kernel::load(&module, def.scale_g2)?,
                },
            });
        }
        let k = Self {
            ctx: cuda.context().clone(),
            stream: cuda.stream().clone(),
            module,
            variants,
            gated,
            active: 0,
            min_block: env_min_block(),
            block: env_block(),
            streams: env_streams(),
            timing: env_timing(),
        };
        match std::env::var("G16_CUDA_FFT_VARIANT") {
            Ok(v) => k.with_variant(&v),
            Err(_) => Ok(k),
        }
    }

    /// Selects the kernel variant every later transform launches. See [`VARIANTS`].
    pub fn with_variant(mut self, name: &str) -> Result<Self, ProveError> {
        match self.variants.iter().position(|v| v.name == name) {
            Some(i) => {
                self.active = i;
                Ok(self)
            }
            None => Err(bad(format!(
                "fft variant {name:?} is not loaded; available: [{}]{}",
                self.variants
                    .iter()
                    .map(|v| v.name)
                    .collect::<Vec<_>>()
                    .join(", "),
                if self.gated {
                    ""
                } else {
                    "; the experiment set needs G16_CUDA_FFT_VARIANTS=1 at compile"
                }
            ))),
        }
    }

    /// The active variant's name.
    pub fn variant(&self) -> &'static str {
        self.variants[self.active].name
    }

    /// Every variant the compiled unit carries, in [`VARIANTS`] order.
    pub fn variant_names(&self) -> Vec<&'static str> {
        self.variants.iter().map(|v| v.name).collect()
    }

    /// Threads per block for every launch. See [`FFT_BLOCK`].
    pub fn with_block(mut self, threads: u32) -> Self {
        self.block = threads.max(1);
        self
    }

    /// Streams the independent blocks of one call spread over. See the module docs'
    /// co-dispatch note: 1 keeps the proven serial shape.
    pub fn with_streams(mut self, n: usize) -> Self {
        self.streams = n.max(1);
        self
    }

    /// Per-pass device timing to stderr, the same switch as `G16_CUDA_FFT_TIME=1`.
    pub fn with_timing(mut self, on: bool) -> Self {
        self.timing = on;
        self
    }

    /// Shortest block this instance will run on the device. See [`MIN_BLOCK`].
    pub fn min_block(&self) -> usize {
        self.min_block
    }

    /// The ladder window, the same for both groups and for every instance. See
    /// [`FFT_WINDOW`].
    pub fn window(&self) -> u32 {
        FFT_WINDOW
    }

    /// Moves the crossover. A test passes 1 so that a power-10 file, whose largest block
    /// is 2^11, reaches the device at all; a measurement passes a sweep.
    pub fn with_min_block(mut self, n: usize) -> Self {
        self.min_block = n;
        self
    }

    /// In-place inverse FFT over G1, the whole of what `prepare::ifft` (prepare.rs:320)
    /// does and in the same order.
    pub fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), ProveError> {
        self.ifft_many::<FftG1>(&mut [a])
    }

    pub fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), ProveError> {
        self.ifft_many::<FftG2>(&mut [a])
    }

    /// [`Self::ifft_g1`] over several independent blocks. See the module docs: the
    /// rounds are kept, the concurrency Metal gets from them is not, yet.
    pub fn ifft_g1_many(&self, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), ProveError> {
        self.ifft_many::<FftG1>(blocks)
    }

    pub fn ifft_g2_many(&self, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), ProveError> {
        self.ifft_many::<FftG2>(blocks)
    }

    /// The transform over one or more independent blocks, in lockstep rounds: round
    /// `exp` is pass `exp` of every block still deep enough to have one, and a block's
    /// own depth `exp == bits` is its fused mix+scale pass, so each block retires in its
    /// own last round and no scaling round follows.
    ///
    /// The rounds, the downloads and the checks are one attempt; see [`RETRIES`] for
    /// what a refused readback does, and [`probe`] and [`scan`] for the checks. The
    /// caller's points are written only after every block has passed both.
    fn ifft_many<G: FftGroup>(&self, blocks: &mut [&mut [Xyzz<G::Raw>]]) -> Result<(), ProveError> {
        for a in blocks.iter() {
            check_block_len(G::OP, a.len())?;
        }

        let pair = G::kernels(self);
        let point_words = core::mem::size_of::<G::PackedPoint>() / 4;

        // The stream each block's passes run on. One stream is the proven serial shape;
        // `G16_CUDA_FFT_STREAMS=N` round-robins the independent blocks over N streams so
        // a round of small blocks can fill the device instead of queuing behind each
        // other. No cross-stream ordering is needed: blocks are independent transforms
        // and each one's uploads, passes, log copies and download stay on its own
        // stream, while the shared `tw` upload below completes (host-blocking) before
        // any launch is issued anywhere. When `tw` is *freed* is a separate question,
        // and the one this reasoning used to skip; [`DrainOnExit`] answers it.
        let pool: Vec<Arc<CudaStream>> = if self.streams > 1 {
            (0..self.streams.min(blocks.len().max(1)))
                .map(|_| {
                    self.ctx
                        .new_stream()
                        .map_err(|e| drv("create fft stream", e))
                })
                .collect::<Result<_, _>>()?
        } else {
            vec![self.stream.clone()]
        };

        // Per-block ping-pong. `src` is filled from the caller's slice by [`fill`] at
        // the start of every attempt and is never written by the pass that reads it;
        // `dst` starts zeroed (msm.rs's allocation policy: an uninitialised limb array
        // is a plausible field element) and every pass overwrites all `n` of its slots.
        // The log holds the probed thread of every pass.
        let mut live: Vec<Block> = Vec::with_capacity(blocks.len());
        for (bi, a) in blocks.iter().enumerate() {
            let n = a.len();
            if n <= 1 {
                continue;
            }
            let bits = n.trailing_zeros();
            let stream = pool[live.len() % pool.len()].clone();
            let alloc = |words: usize, what: &str| {
                stream.alloc_zeros::<u32>(words).map_err(|e| drv(what, e))
            };
            let src = alloc(n * point_words, "allocate fft input")?;
            let dst = alloc(n * point_words, "allocate fft scratch")?;
            let log = alloc(
                LOG_POINTS * bits as usize * point_words,
                "allocate fft probe log",
            )?;

            let size_inv = Fr::from(n as u64)
                .inverse()
                .expect("a power of two is a unit mod r");
            // The `1/n` scaling rides the block's last mix pass (`fft_mix_scale_impl`),
            // which reads a second table with `s = 1/n` folded into every entry,
            // `stw[0]` doubling as the plain `[s]` the `lo` side needs. `s` differs per
            // block, so unlike `tw` this table cannot be shared across a round.
            let stw = upload_words(&stream, as_words(&twiddle_table::<G::Cfg>(bits, size_inv)))?;
            live.push(Block {
                idx: bi,
                n,
                bits,
                size_inv,
                src,
                dst,
                stw,
                log,
                probes: Vec::new(),
                stream,
            });
        }
        let Some(max_bits) = live.iter().map(|b| b.bits).max() else {
            return Ok(());
        };

        // One plain table serves every mix pass of every block: pass `exp` wants
        // `roots[exp]^j`, and `tw_shift = max_bits - exp` reads the deepest block's
        // table at the right stride whatever the block's own depth is.
        let tw = upload_words(
            &self.stream,
            as_words(&twiddle_table::<G::Cfg>(max_bits, Fr::ONE)),
        )?;
        // Declared after `tw` so it drops before it: locals drop in reverse order, so
        // every launch below is finished before the buffers those launches read go away.
        // See [`DrainOnExit`].
        let _drain = DrainOnExit(&pool);
        let curve_b = G::Raw::b();

        // The attempts. A refused readback re-runs the rounds from the caller's slice,
        // which nothing has written yet, after the backoff; the driver's own failures
        // come straight out, as the module docs say.
        let mut times: Vec<PassTime> = Vec::new();
        let mut got: Vec<Vec<G::PackedPoint>> = Vec::new();
        for attempt in 0..RETRIES {
            times.clear();
            for b in &mut live {
                fill::<G>(b, &*blocks[b.idx])?;
            }
            self.run_rounds::<G>(pair, &mut live, &tw, max_bits, &mut times)?;
            match read_back_checked::<G>(&live, curve_b) {
                Ok(points) => {
                    got = points;
                    break;
                }
                Err(Attempt::Refused(e)) if attempt + 1 < RETRIES => {
                    let thread = std::thread::current();
                    eprintln!(
                        "cuda: attempt {} of {RETRIES} failed on thread {}: {e}; re-running the transform",
                        attempt + 1,
                        thread.name().unwrap_or("unnamed"),
                    );
                    std::thread::sleep(backoff(attempt + 1));
                }
                Err(Attempt::Refused(e) | Attempt::Failed(e)) => return Err(e),
            }
        }

        for (b, got) in live.iter().zip(&got) {
            // The `a[1..].reverse()` that finishes the inverse (prepare.rs:340), folded
            // into the read back: `ifft(a)[0] = X[0]/n` and `ifft(a)[i] = X[n-i]/n`.
            // Splitting this off from the scaling gives an answer that is a rotation
            // away from correct and still looks plausible, which is why the module doc
            // there says so twice.
            let a = &mut *blocks[b.idx];
            a[0] = G::unpack(&got[0]);
            let n = b.n;
            a[1..].par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = G::unpack(&got[n - 1 - i]);
            });
        }

        // Every stream is drained by its block's download above, so the event pairs are
        // complete and reading them costs nothing on the device.
        if self.timing {
            let mut total = 0f64;
            for t in &times {
                let ms = f64::from(
                    t.start
                        .elapsed_ms(&t.end)
                        .map_err(|e| drv("read fft event pair", e))?,
                );
                total += ms;
                eprintln!(
                    "snarkrs-cuda fft-time: {} block {} n=2^{} pass {:>2}{} threads {:>8} {:>10.3} ms",
                    G::OP,
                    t.idx,
                    t.bits,
                    t.exp,
                    if t.fused { " fused" } else { "      " },
                    t.threads,
                    ms
                );
            }
            // A sum of per-launch times, so under more than one stream it can exceed
            // the wall the launches actually took together.
            eprintln!(
                "snarkrs-cuda fft-time: {} device sum {:.3} ms over {} launches \
                 (variant {}, block {}, streams {})",
                G::OP,
                total,
                times.len(),
                self.variant(),
                self.block,
                pool.len()
            );
        }
        Ok(())
    }

    /// The rounds of one attempt: every pass of every block, launched in lockstep and
    /// each followed on its stream by the copies that log its probed thread.
    fn run_rounds<G: FftGroup>(
        &self,
        pair: &FftPair,
        live: &mut [Block],
        tw: &CudaSlice<u32>,
        max_bits: u32,
        times: &mut Vec<PassTime>,
    ) -> Result<(), ProveError> {
        let point_words = core::mem::size_of::<G::PackedPoint>() / 4;
        for exp in 1..=max_bits {
            for b in live.iter_mut().filter(|b| b.bits >= exp) {
                let (kernel, third, params, threads) = if b.bits == exp {
                    // The block's last pass: the mix with `1/n` fused in, dense over all
                    // `n/2` butterflies at two ladders each, on the block's own scaled
                    // table.
                    let params = FftParams {
                        n: b.n as u32,
                        ..Default::default()
                    };
                    (&pair.mix_scale, &b.stw, params, b.n / 2)
                } else {
                    // The mix grid is dense over the `j != 0` butterflies (see
                    // `fft_mix_impl`): `groups * (span - 1)` ladder threads, except at
                    // `exp == 1`, where every butterfly is `j == 0` and the pass is all
                    // of them.
                    let threads = if exp == 1 {
                        b.n / 2
                    } else {
                        b.n / 2 - (b.n >> exp)
                    };
                    let params = FftParams {
                        n: b.n as u32,
                        span: 1 << (exp - 1),
                        log_span: exp - 1,
                        tw_shift: max_bits - exp,
                        gid_off: 0,
                    };
                    (&pair.mix, tw, params, threads)
                };
                let cfg = kernel.cfg_1d(threads, self.block);
                let start = self.timed_event(&b.stream)?;
                {
                    let mut lb = b.stream.launch_builder(&kernel.f);
                    lb.arg(&b.src).arg(&b.dst).arg(third).arg(&params);
                    // SAFETY: four parameters bound in order and with matching types;
                    // `src` and `dst` each hold `n` points of this group, every
                    // in-bounds index the kernel touches is under `n`, the twiddle
                    // table holds at least `(n/2) * GLV_WORDS` words by construction,
                    // and the surplus threads of the last CUDA block exit on the
                    // kernel's own bounds guard.
                    unsafe { lb.launch(cfg) }.map_err(|e| drv("launch fft mix", e))?;
                }
                if let Some(start) = start {
                    let end = self.timed_event(&b.stream)?.expect("timing is on");
                    times.push(PassTime {
                        idx: b.idx,
                        bits: b.bits,
                        exp,
                        fused: b.bits == exp,
                        threads,
                        start,
                        end,
                    });
                }
                log_probe::<G>(b, exp, threads, &params, point_words)?;
            }
            // Launches on one stream execute in issue order, so pass `exp + 1` of a
            // block reads what its pass `exp` wrote with no event and no host wait; the
            // swap is pure host bookkeeping.
            for b in live.iter_mut().filter(|b| b.bits >= exp) {
                core::mem::swap(&mut b.src, &mut b.dst);
            }
        }
        Ok(())
    }

    /// An event recorded on `stream` now, or `None` with timing off. `CU_EVENT_DEFAULT`
    /// and not the cudarc default, which is `CU_EVENT_DISABLE_TIMING`: that one records
    /// fine and then fails at `elapsed_ms`, the same trap `msm.rs` documents.
    fn timed_event(&self, stream: &CudaStream) -> Result<Option<CudaEvent>, ProveError> {
        if !self.timing {
            return Ok(None);
        }
        let e = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| drv("create fft timing event", e))?;
        e.record(stream)
            .map_err(|e| drv("record fft timing event", e))?;
        Ok(Some(e))
    }
}

/// The primitive `2^bits`-th root of unity, the one `twiddle_table` raises to its powers.
fn primitive_root(bits: u32) -> Fr {
    let mut root = Fr::TWO_ADIC_ROOT_OF_UNITY;
    for _ in bits..Fr::TWO_ADICITY {
        root.square_in_place();
    }
    root
}

/// The caller's block into `src`, for one attempt.
///
/// `bit_reverse` (prepare.rs:256) folded into the pack, in gather form so the loop
/// splits over the pool: the permutation is an involution, so `src[rev(i)] = a[i]` and
/// `src[j] = a[rev(j)]` are the same map. Only this direction is applied; the rotation at
/// the read back is the one that is NOT an involution. Unlike Metal the pack cannot land
/// in the device buffer directly, so it lands in a `Vec` and one `memcpy_htod` follows,
/// waited on because the source is pageable and goes away at the end of this function.
fn fill<G: FftGroup>(b: &mut Block, a: &[Xyzz<G::Raw>]) -> Result<(), ProveError> {
    let mut packed = vec![G::PackedPoint::default(); b.n];
    packed
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, s)| *s = G::pack(&a[bit_reverse_index(j, b.bits)]));
    b.stream
        .memcpy_htod(as_words(&packed), &mut b.src)
        .map_err(|e| drv("upload fft input", e))?;
    b.stream
        .synchronize()
        .map_err(|e| drv("synchronize upload", e))?;
    b.probes.clear();
    Ok(())
}

/// The probed thread of the pass just launched: its two or four points, copied on the
/// block's stream from `dst` and `src` into the log, behind the launch and ahead of the
/// pass that overwrites `src`, with what the host will hold them to.
///
/// The thread is [`check::spot`] of the pass's parameters, so every pass of a run probes
/// a fixed position and consecutive passes probe different ones. Its butterflies are the
/// ones `fft_mix_impl` or `fft_mix_scale_impl` gives that thread, index for index.
fn log_probe<G: FftGroup>(
    b: &mut Block,
    exp: u32,
    threads: usize,
    p: &FftParams,
    point_words: usize,
) -> Result<(), ProveError> {
    let half = b.n / 2;
    let salt = (u64::from(p.n) << 40) ^ (u64::from(p.span) << 32);
    let d = check::spot(threads, salt);

    // `(lo, hi, twiddle on hi, scale on both)`, in the kernels' index arithmetic.
    let mut butterflies: Vec<(usize, usize, Option<Fr>, Option<Fr>)> = Vec::with_capacity(2);
    if b.bits == exp {
        let s = b.size_inv;
        let w = s * primitive_root(b.bits).pow([d as u64]);
        butterflies.push((d, d + half, Some(w), Some(s)));
    } else if exp == 1 {
        butterflies.push((2 * d, 2 * d + 1, None, None));
    } else {
        let span = 1usize << (exp - 1);
        let groups = half >> (exp - 1);
        let g = d / (span - 1);
        let j = 1 + (d - g * (span - 1));
        let lo = (g << exp) + j;
        let w = primitive_root(exp).pow([j as u64]);
        butterflies.push((lo, lo + span, Some(w), None));
        if d < groups {
            let lo0 = d << exp;
            butterflies.push((lo0, lo0 + span, None, None));
        }
    }

    let Block {
        src,
        dst,
        log,
        probes,
        stream,
        ..
    } = &mut *b;
    for (lo, hi, w, scale) in butterflies {
        let slot = probes.len() * 4;
        for (k, (from, i)) in [(&*dst, lo), (&*dst, hi), (&*src, lo), (&*src, hi)]
            .into_iter()
            .enumerate()
        {
            let at = (slot + k) * point_words;
            stream
                .memcpy_dtod(
                    &from.slice(i * point_words..(i + 1) * point_words),
                    &mut log.slice_mut(at..at + point_words),
                )
                .map_err(|e| drv("log fft probe", e))?;
        }
        probes.push(Probe {
            lo,
            hi,
            w,
            scale,
            slot,
        });
    }
    Ok(())
}

/// A block's buffer off the device as points: `download`, the injector's hook on the
/// words, then the shape check.
fn read_back<P: Packed + Default>(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<u32>,
) -> Result<Vec<P>, ProveError> {
    let mut words = download(stream, buf)?;
    crate::fault::stored(&mut words);
    from_words::<P>(&words).ok_or_else(|| bad("fft read back is not a whole number of points"))
}

/// Every block's result and log, downloaded and held to the checks in block order,
/// stopping at the first refusal. Nothing of the caller's is written here.
fn read_back_checked<G: FftGroup>(
    live: &[Block],
    curve_b: G::Raw,
) -> Result<Vec<Vec<G::PackedPoint>>, Attempt> {
    let mut out = Vec::with_capacity(live.len());
    for b in live {
        // The one wait per block: `download` synchronizes the block's stream before the
        // `Vec` is read, which also orders it after every launch and copy of the rounds.
        // The log follows on the drained stream.
        let got = read_back::<G::PackedPoint>(&b.stream, &b.src)?;
        let log = read_back::<G::PackedPoint>(&b.stream, &b.log)?;
        probe::<G>(b, &log, curve_b)?;
        scan::<G>(b, &got, curve_b)?;
        out.push(got);
    }
    Ok(out)
}

/// One thread of every pass, recomputed on the host and held to what the pass wrote.
///
/// Each probed butterfly is computed from the pass's own source with the host ladder and
/// compared as a point, since the two ladders reach different representatives. What this
/// catches that a curve check cannot: a destination the device left holding the previous
/// pass's points, which are on the curve.
///
/// An input off the curve is an earlier pass's wrong store, not this one's; the probe
/// stands aside and the scan over the finished block refuses it.
fn probe<G: FftGroup>(b: &Block, log: &[G::PackedPoint], curve_b: G::Raw) -> Result<(), Attempt> {
    for p in &b.probes {
        let (dlo, dhi, slo, shi) = (
            &log[p.slot],
            &log[p.slot + 1],
            &log[p.slot + 2],
            &log[p.slot + 3],
        );
        if !slo.reduced() || !shi.reduced() {
            continue;
        }
        let (u, t) = (G::unpack(slo), G::unpack(shi));
        if !check::xyzz_on_curve(&u, curve_b) || !check::xyzz_on_curve(&t, curve_b) {
            continue;
        }
        let t = match p.w {
            Some(w) => glv::mul(&t, &w),
            None => t,
        };
        let u = match p.scale {
            Some(s) => glv::mul(&u, &s),
            None => u,
        };
        let mut sum = u;
        sum.add_assign(&t);
        let mut diff = u;
        diff.add_assign(&Xyzz { y: t.y.neg(), ..t });
        for (i, got, want) in [(p.lo, dlo, sum), (p.hi, dhi, diff)] {
            if !got.reduced() || !check::same_point(&G::unpack(got), &want) {
                return Err(Attempt::Refused(check::wrong(
                    GROUP_IFFT,
                    format!(
                        "point {i} of a {}-point block is not the host butterfly's",
                        b.n
                    ),
                )));
            }
        }
    }
    Ok(())
}

/// Every point of a finished block, held to the curve.
///
/// Once, at the end, not per pass: a pass over a 2^20-point G2 block is a million
/// six-multiplication checks against about a second of GPU, so twenty of them would cost
/// a good part of the transform while the pool is busy with the sub-crossover blocks, and
/// once loses nothing, because a wrong word makes a point off the curve and the
/// butterflies are rational maps, so it reaches at least one output. What the scan cannot
/// see, a destination left holding the previous pass's points, [`probe`] catches from the
/// log.
fn scan<G: FftGroup>(b: &Block, got: &[G::PackedPoint], curve_b: G::Raw) -> Result<(), Attempt> {
    let bad = got
        .par_iter()
        .position_any(|p| !p.reduced() || !check::xyzz_on_curve(&G::unpack(p), curve_b));
    if let Some(i) = bad {
        return Err(Attempt::Refused(check::wrong(
            GROUP_IFFT,
            format!("point {i} of a {}-point block is not on the curve", b.n),
        )));
    }
    Ok(())
}

/// Refuses a block length the transform has no subgroup for, before anything touches the
/// device. 0 and 1 pass: those blocks are their own transform and are skipped.
fn check_block_len(op: &str, n: usize) -> Result<(), ProveError> {
    if n <= 1 {
        return Ok(());
    }
    if !n.is_power_of_two() {
        return Err(bad(format!(
            "group ifft over {op}: {n} points is not a power of two"
        )));
    }
    if n.trailing_zeros() > Fr::TWO_ADICITY {
        return Err(bad(format!(
            "group ifft over {op}: 2^{} is past the {}-bit two-adic subgroup",
            n.trailing_zeros(),
            Fr::TWO_ADICITY
        )));
    }
    Ok(())
}

/// `i` with its low `bits` bits reversed, the permutation `bit_reverse` (prepare.rs:256)
/// applies by swapping in place.
#[inline]
fn bit_reverse_index(i: usize, bits: u32) -> usize {
    ((i as u32).reverse_bits() >> (u32::BITS - bits)) as usize
}

/// `G16_CUDA_FFT_MIN_BLOCK` overrides [`MIN_BLOCK`], which is how the crossover gets
/// swept on a real card without a rebuild and how a test forces a small file onto the
/// device.
fn env_min_block() -> usize {
    std::env::var("G16_CUDA_FFT_MIN_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(MIN_BLOCK)
}

/// `G16_CUDA_FFT_BLOCK` overrides [`FFT_BLOCK`], the threads-per-block sweep knob.
fn env_block() -> u32 {
    std::env::var("G16_CUDA_FFT_BLOCK")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(FFT_BLOCK)
}

/// `G16_CUDA_FFT_STREAMS` spreads independent blocks over that many streams; 1, the
/// default, is the proven serial shape.
fn env_streams() -> usize {
    std::env::var("G16_CUDA_FFT_STREAMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1)
}

/// `G16_CUDA_FFT_TIME=1` wraps every launch in a CUDA event pair and prints per-pass
/// device times to stderr after the read back.
fn env_timing() -> bool {
    std::env::var("G16_CUDA_FFT_TIME").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    mod resources {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-support/resources.rs"
        ));
    }

    use super::*;
    use ark_ec::short_weierstrass::{Affine, Projective};
    use snarkrs_field::raw::RawField;
    use snarkrs_field::{AffineRepr, CurveGroup, Fq, Fq2, G1Affine, G2Affine, Zero};
    use snarkrs_gpu_layout::PackedGlv;
    use snarkrs_msm::xyzz::{to_projective, RawCurve};

    type G1Cfg = snarkrs_field::g1::Config;
    type G2Cfg = snarkrs_field::g2::Config;

    /// Every row of [`VARIANTS`] has an instantiation line in the source, and the gated
    /// rows sit behind the `G16_FFT_VARIANTS` guard their loading is keyed on.
    /// Construction would catch a missing kernel, but only on a machine with a device
    /// and minutes into a compile; this runs anywhere.
    #[test]
    fn every_variant_has_an_instantiation_line() {
        assert!(kernels::FFT_CU.contains("#ifdef G16_FFT_VARIANTS"));
        for def in VARIANTS {
            for suffix in [
                def.mix_g1.trim_start_matches("fft_mix_"),
                def.mix_g2.trim_start_matches("fft_mix_"),
            ] {
                // Matches both macro forms: `FFT_KERNELS(g1_c5,` and
                // `FFT_KERNELS_LB(g1_c5r128,`.
                assert!(
                    kernels::FFT_CU.contains(&format!("({suffix},")),
                    "kernels/fft.cu has no FFT_KERNELS line for {suffix}, so \
                     fft_mix_{suffix} does not exist and load_function will fail"
                );
            }
            assert_eq!(
                def.scale_g1,
                format!(
                    "fft_mix_scale_{}",
                    def.mix_g1.trim_start_matches("fft_mix_")
                )
            );
            assert_eq!(
                def.scale_g2,
                format!(
                    "fft_mix_scale_{}",
                    def.mix_g2.trim_start_matches("fft_mix_")
                )
            );
        }
        // The shipped row is ungated and named after the shipped window.
        assert!(!VARIANTS[0].gated);
        assert_eq!(VARIANTS[0].name, format!("c{FFT_WINDOW}"));
        // The gated rows are inside the guard: everything after `#ifdef` and before the
        // closing `#endif // G16_FFT_VARIANTS` is the experiment block, and each gated
        // instantiation must appear after the `#ifdef`.
        let guard = kernels::FFT_CU.find("#ifdef G16_FFT_VARIANTS").unwrap();
        for def in VARIANTS.iter().filter(|d| d.gated) {
            let suffix = def.mix_g1.trim_start_matches("fft_mix_");
            let at = kernels::FFT_CU.find(&format!("({suffix},")).unwrap();
            assert!(
                at > guard,
                "{suffix} is instantiated outside the G16_FFT_VARIANTS guard, so every \
                 user pays its compile time"
            );
        }
    }

    /// Without `extern "C"` NVRTC mangles the entry names and `load_function` fails at
    /// run time, on a GPU box, minutes into a ptxas run. The entry points come out of
    /// the `FFT_KERNELS` macro, so the check is on the macro body.
    #[test]
    fn the_entry_points_are_declared_extern_c() {
        for want in [
            "extern \"C\" __global__ void fft_mix_##SUF(",
            "extern \"C\" __global__ void fft_mix_scale_##SUF(",
            "extern \"C\" __global__ void __launch_bounds__(MAXT, MINB)",
        ] {
            assert!(
                kernels::FFT_CU.contains(want),
                "kernels/fft.cu has no entry point declared as:\n{want}"
            );
        }
    }

    /// The GLV constants live in two languages. `beta` is the one that can be wrong
    /// silently: a table entry decomposed against arkworks' lattice and a kernel that
    /// multiplies X by some other cube root of one still produces a point on the curve,
    /// in the right subgroup, and wrong. The CUDA twin of the Metal backend's
    /// `msl_declares_the_same_glv_constants`.
    #[test]
    fn cuda_declares_the_same_glv_constants() {
        let beta = <snarkrs_field::g1::Config as GLVConfig>::ENDO_COEFFS[0];
        let want = format!(
            "__constant__ u32 FQ_BETA[8] = {{ {} }};",
            PackedFq::from_fq(&beta)
                .v
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            kernels::FFT_CU.contains(&want),
            "kernels/fft.cu does not contain the line:\n{want}"
        );
        for (name, want) in [
            ("GLV_WORDS", core::mem::size_of::<PackedGlv>() / 4),
            ("GLV_RECODE_BITS", 128),
        ] {
            let line = format!("#define {name} {want}");
            assert!(
                kernels::FFT_CU.contains(&line),
                "kernels/fft.cu does not contain the line:\n{line}"
            );
        }
    }

    /// The FFT unit must never carry the MSM entry points, which is the whole reason it
    /// exists as a unit; see `kernels::unit_fft`.
    #[test]
    fn the_fft_unit_compiles_no_msm_kernels() {
        let unit = kernels::unit_fft();
        assert!(!unit.contains("msm_count"), "the FFT unit grew the MSM");
        assert!(unit.contains("fft_mix_impl"));
        assert!(
            unit.contains("jac_dbl"),
            "the FFT unit is missing the Jac family"
        );
    }

    /// The host-side arms of `ifft_many`, which need no device: 0 and 1 points pass
    /// through, anything else must be a power of two inside Fr's two-adic subgroup.
    #[test]
    fn a_bad_block_length_is_refused() {
        for n in [0, 1, 2, 4, 1 << Fr::TWO_ADICITY] {
            assert!(check_block_len("G1", n).is_ok(), "{n} points was refused");
        }
        for n in [3, 6, 12, (1 << 20) + 1] {
            let Err(ProveError::Backend { backend, reason }) = check_block_len("G1", n) else {
                panic!("{n} points was accepted");
            };
            assert_eq!(backend, "cuda");
            assert!(reason.contains("not a power of two"), "{reason}");
        }
        let n = 1usize << (Fr::TWO_ADICITY + 1);
        let Err(ProveError::Backend { reason, .. }) = check_block_len("G2", n) else {
            panic!("2^{} points was accepted", Fr::TWO_ADICITY + 1);
        };
        assert!(reason.contains("G2"), "{reason}");
        assert!(reason.contains("two-adic subgroup"), "{reason}");
    }

    /// `f_beta_mul(Fq2)` multiplies both halves by the one G1 `beta`, which is the G2
    /// endomorphism only because ark's G2 coefficient is that same `beta` with `c1 == 0`.
    /// The kernel comment asserts it; this pins it, so a change to either lattice shows up
    /// here and not as a G2 block that disagrees.
    #[test]
    fn the_g2_endomorphism_is_the_g1_beta_in_c0() {
        let g1 = <G1Cfg as GLVConfig>::ENDO_COEFFS;
        let g2 = <G2Cfg as GLVConfig>::ENDO_COEFFS;
        assert_eq!(g1.len(), 1);
        assert_eq!(g2.len(), 1);
        assert_eq!(g2[0], Fq2::new(g1[0], Fq::zero()));
        for line in ["r.c0 = fq_mul(x.c0, b);", "r.c1 = fq_mul(x.c1, b);"] {
            assert!(
                kernels::FFT_CU.contains(line),
                "kernels/fft.cu's f_beta_mul(Fq2) no longer contains:\n{line}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // On-device checks. Ported from `crates/metal/src/fft.rs`; the oracle differs
    // because snarkrs-ceremony depends on this crate, so `CpuGroupFft` cannot be a
    // dev-dependency here without a cycle. `dft_ref` below is independent of every
    // table and formula the kernels share with the CPU transform instead.
    // -----------------------------------------------------------------------

    /// Deterministic draws, so a failure is reproducible without a seed to record. Same
    /// generator and same reason as the Metal twin's.
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

    /// `[k] G` in XYZZ with a deliberately non-trivial `ZZ`, so the butterflies see the
    /// spread of representatives a real transform hands them rather than a vector that is
    /// entirely `ZZ == 1`. `ZZ = u^2`, `ZZZ = u^3` keeps the `ZZ^3 == ZZZ^2` invariant.
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

    /// `n` points with every fourth one the point at infinity.
    ///
    /// Infinity is a live input, not a defensive one: ptau section 12's `power+1` block is
    /// padded with it (prepare.rs:562), and after one butterfly it is also what a
    /// cancellation leaves behind, so it has to survive the ladder, the addition and the
    /// scaling pass.
    fn block_g1(seed: u64, n: usize) -> Vec<Xyzz<RawFq>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    Xyzz::ZERO
                } else {
                    scaled_g1(rng.fr(), rng.word())
                }
            })
            .collect()
    }

    fn block_g2(seed: u64, n: usize) -> Vec<Xyzz<RawFq2>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    Xyzz::ZERO
                } else {
                    scaled_g2(rng.fr(), rng.word())
                }
            })
            .collect()
    }

    /// `n` points with none at infinity, for the test that corrupts the first point of a
    /// readback: a wrong word beside a zero `ZZ` is still the identity, so on `block_g1`'s
    /// inputs, where point 0 of a pass is often one, the fault would be harmless and
    /// nothing would need to notice it.
    fn dense_g1(seed: u64, n: usize) -> Vec<Xyzz<RawFq>> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| scaled_g1(rng.fr(), rng.word())).collect()
    }

    /// Element for element the same points, whatever the representatives.
    fn same_g1(a: &[Xyzz<RawFq>], b: &[Xyzz<RawFq>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| check::same_point(p, q))
    }

    /// `n` points drawn from a pool of three, laid out so the first pass meets equal and
    /// opposite pairs: pass 1 pairs `a[i]` with `a[i + n/2]`, which here is `a[i]` again
    /// for even `i` and `-a[i]` for odd `i`. Half of that pass takes `pt_add`'s doubling
    /// branch and the other half cancels to infinity, which every later pass then ladders
    /// and adds. The same-x branches and a zero `ZZ` are where a point formula goes wrong
    /// silently, and a block of distinct random points never reaches them.
    fn repeated<F: RawField>(pool: &[Xyzz<F>], n: usize) -> Vec<Xyzz<F>> {
        let half = (n / 2).max(1);
        (0..n)
            .map(|i| {
                let p = pool[(i % half) % pool.len()];
                if i >= half && i % 2 == 1 {
                    Xyzz { y: p.y.neg(), ..p }
                } else {
                    p
                }
            })
            .collect()
    }

    fn pool_g1(seed: u64) -> Vec<Xyzz<RawFq>> {
        let mut rng = Lcg(seed);
        (0..3).map(|_| scaled_g1(rng.fr(), rng.word())).collect()
    }

    fn pool_g2(seed: u64) -> Vec<Xyzz<RawFq2>> {
        let mut rng = Lcg(seed);
        (0..3).map(|_| scaled_g2(rng.fr(), rng.word())).collect()
    }

    /// The comparison the byte-identity claim rests on: affine, which is canonical, and
    /// which is what `lagrange_evaluations` (prepare.rs:414) writes to the file.
    fn affine<P: RawCurve>(a: &[Xyzz<P::RF>]) -> Vec<Affine<P>> {
        Projective::normalize_batch(&a.iter().map(to_projective::<P>).collect::<Vec<_>>())
    }

    /// The transform the device is measured against, sharing nothing with the kernels:
    /// ark's projective arithmetic, a plain scalar multiplication per twiddle, and the
    /// primitive root `prepare::root_table` builds, which is ark's `get_root_of_unity`
    /// (`TWO_ADIC_ROOT_OF_UNITY` squared down to the block). `inverse` runs the same
    /// butterflies over `w^-1` and scales by `1/n`, the textbook definition, so the
    /// `a[1..].reverse()` fold, the GLV lattice, `f_beta_mul` and both twiddle tables are
    /// all on the other side of the comparison. `prepare::ifft` agrees with this by
    /// construction: `fft` over `w` then `a[1..].reverse()` is `sum a[j] w^(-ij)`.
    fn dft_ref<P: RawCurve<ScalarField = Fr>>(a: &[Xyzz<P::RF>], inverse: bool) -> Vec<Affine<P>> {
        let n = a.len();
        let mut v: Vec<Projective<P>> = a.iter().map(to_projective::<P>).collect();
        if n > 1 {
            let bits = n.trailing_zeros();
            for i in 0..n {
                let r = ((i as u32).reverse_bits() >> (u32::BITS - bits)) as usize;
                if i < r {
                    v.swap(i, r);
                }
            }
            let mut w = Fr::get_root_of_unity(n as u64).expect("2^bits divides r - 1");
            if inverse {
                w = w.inverse().expect("a root of unity is a unit");
            }
            let mut len = 2;
            while len <= n {
                let step = w.pow([(n / len) as u64]);
                let mut tw = Vec::with_capacity(len / 2);
                let mut t = Fr::ONE;
                for _ in 0..len / 2 {
                    tw.push(t);
                    t *= step;
                }
                v.par_chunks_mut(len).for_each(|chunk| {
                    let (lo, hi) = chunk.split_at_mut(len / 2);
                    for (j, (lo, hi)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                        let t = *hi * tw[j];
                        *hi = *lo - t;
                        *lo += t;
                    }
                });
                len <<= 1;
            }
            if inverse {
                let size_inv = Fr::from(n as u64)
                    .inverse()
                    .expect("a power of two is a unit mod r");
                v.par_iter_mut().for_each(|p| *p *= size_inv);
            }
        }
        Projective::normalize_batch(&v)
    }

    /// One block through the device and back: the result is the reference inverse, and
    /// the reference forward transform of it is the input again. The round trip is what
    /// pins the convention rather than the arithmetic: an answer off by the rotation the
    /// module doc warns about is a valid transform of something, just not of `input`.
    fn check_ifft<G: FftGroup>(k: &FftKernels, input: &[Xyzz<G::Raw>], what: &str)
    where
        G::Cfg: RawCurve<RF = G::Raw>,
    {
        let mut got = input.to_vec();
        k.ifft_many::<G>(&mut [&mut got])
            .unwrap_or_else(|e| panic!("{} {what}: device ifft failed: {e}", G::OP));
        assert!(
            affine::<G::Cfg>(&got) == dft_ref::<G::Cfg>(input, true),
            "{} {what}: the device ifft disagrees with the reference",
            G::OP
        );
        assert!(
            dft_ref::<G::Cfg>(&got, false) == affine::<G::Cfg>(input),
            "{} {what}: the forward transform of the device ifft is not the input",
            G::OP
        );
    }

    fn device() -> Option<Cuda> {
        match Cuda::new(0) {
            Ok(c) => {
                let (major, minor) = c.compute_capability();
                eprintln!(
                    "cuda device: {} (sm_{major}{minor}, {} SMs)",
                    c.device_name(),
                    c.sm_count()
                );
                Some(c)
            }
            Err(e) => {
                resources::skip(format_args!("cuda FFT test, no usable device: {e}"));
                None
            }
        }
    }

    /// The shipped variant, compiled from the FFT unit. A second process on the same
    /// box hits the PTX cache (`context.rs`), so only the first test of a cold box pays
    /// NVRTC.
    fn kernels() -> Option<FftKernels> {
        let cuda = device()?;
        Some(FftKernels::new(&cuda).expect("fft kernels"))
    }

    /// Every block size from 2^0 to 2^12, every size a ptau file below power 12 contains
    /// and the shape of every pass count from zero to thirteen, then 2^15 for a grid
    /// wider than any one CUDA block. 2^0 is the host pass-through: nothing is launched
    /// and the point comes back untouched.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn ifft_g1_matches_the_reference_at_every_block_size() {
        let Some(k) = kernels() else { return };
        for bits in (0..=12u32).chain([15]) {
            let n = 1usize << bits;
            let input = block_g1(0x5eed_1000 + u64::from(bits), n);
            check_ifft::<FftG1>(&k, &input, &format!("block of 2^{bits}"));
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn ifft_g2_matches_the_reference_at_every_block_size() {
        let Some(k) = kernels() else { return };
        for bits in (0..=10u32).chain([12]) {
            let n = 1usize << bits;
            let input = block_g2(0x5eed_2000 + u64::from(bits), n);
            check_ifft::<FftG2>(&k, &input, &format!("block of 2^{bits}"));
        }
    }

    /// A block that is entirely the point at infinity, which is what `ptau new` produces
    /// before any contribution: `tau == 1` makes every input the generator and most
    /// outputs the identity. A ladder that starts its accumulator anywhere but infinity,
    /// or a butterfly that reads a zero `ZZ` as a real coordinate, fails here and nowhere
    /// else.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn an_all_infinity_block_stays_infinity() {
        let Some(k) = kernels() else { return };
        let n = 256;
        let mut g1 = vec![Xyzz::<RawFq>::ZERO; n];
        k.ifft_g1(&mut g1).expect("ifft_g1");
        assert!(
            g1.iter().all(|p| p.is_zero()),
            "G1 infinity did not survive"
        );
        let mut g2 = vec![Xyzz::<RawFq2>::ZERO; n];
        k.ifft_g2(&mut g2).expect("ifft_g2");
        assert!(
            g2.iter().all(|p| p.is_zero()),
            "G2 infinity did not survive"
        );
    }

    /// Blocks whose butterflies meet the same point twice and a point against its
    /// negation (see [`repeated`]), at the one-pass size and three deeper ones, plus a
    /// constant block, whose transform is its single point at index 0 and infinity
    /// everywhere else: `(1/n) sum_j P w^(-ij)` is `P` at `i == 0` and a geometric sum
    /// of the roots, which is zero, at every other `i`.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn repeated_and_cancelling_points_agree() {
        let Some(k) = kernels() else { return };
        let pool_g1 = pool_g1(0x5eed_7001);
        let pool_g2 = pool_g2(0x5eed_7002);
        for n in [2usize, 8, 64, 512] {
            check_ifft::<FftG1>(
                &k,
                &repeated(&pool_g1, n),
                &format!("repeated block of {n}"),
            );
            check_ifft::<FftG2>(
                &k,
                &repeated(&pool_g2, n),
                &format!("repeated block of {n}"),
            );
        }

        let n = 128;
        let mut g1 = vec![pool_g1[0]; n];
        k.ifft_g1(&mut g1).expect("ifft_g1");
        let g1 = affine::<G1Cfg>(&g1);
        assert_eq!(g1[0], affine::<G1Cfg>(&pool_g1[..1])[0]);
        assert!(g1[1..].iter().all(|p| p.is_zero()), "G1 constant block");
        let mut g2 = vec![pool_g2[1]; n];
        k.ifft_g2(&mut g2).expect("ifft_g2");
        let g2 = affine::<G2Cfg>(&g2);
        assert_eq!(g2[0], affine::<G2Cfg>(&pool_g2[1..2])[0]);
        assert!(g2[1..].iter().all(|p| p.is_zero()), "G2 constant block");
    }

    /// A wrong store under a drained stream is refused and the transform re-run: by the
    /// scan when the result readback carries it, by the probe when the log does, each a
    /// readback of its own, so every readback index of a clean run is tried. Either way
    /// the answer is the reference's and the run cost one re-run. Wrong on every attempt,
    /// the call fails and the caller's points are as they were. No point at infinity in
    /// the inputs (see `dense_g1`); `an_all_infinity_block_stays_infinity` covers those.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn a_wrong_store_is_retried_by_the_fft() {
        use crate::fault::inject::{self, Fault};
        let Some(k) = kernels() else { return };
        let base: Vec<Vec<_>> = [4usize, 8, 16]
            .into_iter()
            .map(|n| dense_g1(0x5eed_2930 + n as u64, n))
            .collect();
        let want: Vec<_> = base.iter().map(|b| dft_ref::<G1Cfg>(b, true)).collect();
        let run = |got: &mut Vec<Vec<Xyzz<RawFq>>>| {
            k.ifft_g1_many(&mut got.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>())
        };
        inject::arm(None);
        let mut got = base.clone();
        run(&mut got).unwrap();
        let calls = inject::calls();
        assert_eq!(calls, 2 * base.len() as u32, "a result and a log per block");

        for at in 0..calls {
            inject::arm_with(Some((at, Fault::WrongStore)), false);
            let mut got = base.clone();
            run(&mut got).unwrap();
            assert!(inject::fired(), "readback {at}");
            assert_eq!(inject::calls(), 2 * calls, "readback {at}: one re-run");
            for (g, w) in got.iter().zip(&want) {
                assert!(affine::<G1Cfg>(g) == *w, "readback {at}");
            }
        }
        eprintln!("{calls} readbacks, each corrupted once and refused");

        inject::arm_with(Some((0, Fault::WrongStore)), true);
        let mut got = base.clone();
        assert!(matches!(run(&mut got), Err(ProveError::Device { .. })));
        assert!(inject::fired());
        assert_eq!(inject::calls(), RETRIES, "one refused readback an attempt");
        for (g, b) in got.iter().zip(&base) {
            assert!(same_g1(g, b), "a failed call wrote its points");
        }
        inject::arm(None);
        let mut got = base.clone();
        run(&mut got).unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!(affine::<G1Cfg>(g) == *w);
        }
    }

    /// Blocks of every depth from 0 to 2^10 in one `ifft_many` call, which is the shape
    /// `process_section` sends: one `tw` table at the deepest block's `max_bits` serves
    /// every shallower block through `tw_shift`, each block's own `exp == bits` round is
    /// its fused pass, and the 0- and 1-point blocks are skipped rather than launched.
    /// Once on the one stream and once round-robined over three, with the timing events
    /// on for the second so that path is exercised too.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn blocks_of_every_depth_agree_in_one_call() {
        let Some(k) = kernels() else { return };
        let sizes: Vec<usize> = std::iter::once(0).chain((0..=10).map(|b| 1 << b)).collect();
        let g1: Vec<Vec<Xyzz<RawFq>>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| block_g1(0x5eed_6100 + i as u64, n))
            .collect();
        let g2: Vec<Vec<Xyzz<RawFq2>>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| block_g2(0x5eed_6200 + i as u64, n))
            .collect();
        let want_g1: Vec<_> = g1.iter().map(|a| dft_ref::<G1Cfg>(a, true)).collect();
        let want_g2: Vec<_> = g2.iter().map(|a| dft_ref::<G2Cfg>(a, true)).collect();

        let mut k = k;
        for streams in [1usize, 3] {
            k = k.with_streams(streams).with_timing(streams > 1);
            let mut got_g1 = g1.clone();
            k.ifft_g1_many(&mut got_g1.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>())
                .expect("ifft_g1_many");
            let mut got_g2 = g2.clone();
            k.ifft_g2_many(&mut got_g2.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>())
                .expect("ifft_g2_many");
            for (i, &n) in sizes.iter().enumerate() {
                assert!(
                    affine::<G1Cfg>(&got_g1[i]) == want_g1[i],
                    "G1 block of {n} points disagrees under {streams} streams"
                );
                assert!(
                    affine::<G2Cfg>(&got_g2[i]) == want_g2[i],
                    "G2 block of {n} points disagrees under {streams} streams"
                );
            }
        }
    }

    /// The launch geometry is not the answer: every thread count from one per block to
    /// past the kernel's own maximum, which `cfg_1d` clamps, reaches the same points.
    /// 7 divides neither a pass's thread count nor a warp, so the last block is partial
    /// in every pass and the kernel's bounds guard is what keeps the surplus threads out.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn the_thread_count_does_not_change_the_points() {
        let Some(k) = kernels() else { return };
        let g1 = block_g1(0x5eed_8001, 1 << 9);
        let g2 = block_g2(0x5eed_8002, 1 << 7);
        let mut k = k;
        for threads in [1u32, 7, 32, 128, 1024] {
            k = k.with_block(threads);
            check_ifft::<FftG1>(&k, &g1, &format!("{threads} threads per block"));
            check_ifft::<FftG2>(&k, &g2, &format!("{threads} threads per block"));
        }
    }

    /// Every variant the unit can carry reaches the same points, which is what makes a
    /// `G16_CUDA_FFT_VARIANT` sweep a measurement rather than a correctness gamble. The
    /// gated rows are compiled here on purpose: this is the one test that pays for them,
    /// and the PTX cache keeps a rerun free. Run it last and alone: the shipped two pairs
    /// are already a 60 MB PTX and about nine minutes of NVRTC on a 4 vCPU g4dn.xlarge
    /// (every ladder inlines the CIOS multiply, see the `fft.cu` banner), and this set is
    /// four times that, so with `--ignored` it would hold every other gate behind it.
    #[test]
    #[ignore = "needs an NVIDIA device and compiles the gated variant set, tens of minutes cold"]
    fn every_variant_gives_the_same_points() {
        let Some(cuda) = device() else { return };
        let module = FftKernels::compile_with(&cuda, true).expect("compile the gated fft unit");
        let mut k = FftKernels::from_module_with(&cuda, module, true).expect("fft kernels");
        assert_eq!(
            k.variant_names(),
            VARIANTS.iter().map(|d| d.name).collect::<Vec<_>>()
        );
        let g1 = block_g1(0x5eed_3001, 512);
        let g2 = block_g2(0x5eed_3002, 512);
        for def in &VARIANTS {
            k = k.with_variant(def.name).expect("a loaded variant");
            assert_eq!(k.variant(), def.name);
            check_ifft::<FftG1>(&k, &g1, &format!("variant {}", def.name));
            check_ifft::<FftG2>(&k, &g2, &format!("variant {}", def.name));
        }
        assert!(
            k.with_variant("c9").is_err(),
            "an unknown variant was accepted"
        );
    }
}
