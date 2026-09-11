//! The group inverse FFT behind `ptau prepare`, on CUDA.
//!
//! Host twin of `crates/g16-metal/src/fft.rs`, driving `kernels/fft.cu`. The transform,
//! the twiddle tables and the two permutations that are deliberately not kernels are the
//! same; what this file owns is the CUDA shape of the submission, and it is simpler than
//! the Metal one for reasons that are that backend's, not this one's:
//!
//! * **No retry loop and no ladder budget.** Both exist to survive macOS killing a busy
//!   command buffer with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, a
//!   display-contention behaviour with no analogue on a headless compute card. A pass
//!   here is one launch however large it is, `gid_off` stays zero, and a driver failure
//!   is an error rather than something to sleep and retry.
//! * **One synchronize, at the read back.** The Metal host batches passes into command
//!   buffers and waits on each; launches on one CUDA stream queue asynchronously and
//!   execute in issue order, which is exactly the pass-after-pass dependency the
//!   transform needs, so the rounds are ordered for free and the host never blocks until
//!   the download. Same argument as the header of `msm.rs`: the thing to avoid is a
//!   synchronize, and there is none in the loop.
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
use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{FftField, Field, Fr};
use g16_gpu_layout::glv::twiddle_table;
use g16_gpu_layout::{Packed, PackedFq, PackedFq2};
use g16_msm::xyzz::Xyzz;
use rayon::prelude::*;

use crate::msm::{bad, download, drv, upload_words, Kernel, PackedXyzzG1, PackedXyzzG2};
use crate::{as_words, from_words, kernels, Cuda};

/// Ladder window of the shipped variant, for both groups.
///
/// 5 is the Metal sweep's answer for both groups over a whole `ppot_0080_16.ptau`
/// prepare (`g16-metal/src/fft.rs`, `FFT_WINDOW_G1`), carried over rather than re-swept:
/// there is no NVIDIA sweep yet. The sweep exists as [`VARIANTS`] behind
/// `G16_CUDA_FFT_VARIANTS=1`, one compile for the whole set; this constant names the
/// default until that sweep says otherwise.
const FFT_WINDOW: u32 = 5;

/// Blocks shorter than this go back to the CPU.
///
/// Metal's number (`g16-metal/src/fft.rs`, `MIN_BLOCK`), carried over unswept. The
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
/// ping-pong pair, its own `1/n`-scaled twiddle table, the stream its passes run on,
/// and where in the caller's slice of blocks it came from.
struct Block {
    idx: usize,
    n: usize,
    bits: u32,
    src: CudaSlice<u32>,
    dst: CudaSlice<u32>,
    stw: CudaSlice<u32>,
    stream: Arc<CudaStream>,
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
/// once. The smaller CUDA twin of `g16-metal/src/fft.rs`'s trait of the same name.
trait FftGroup {
    type Raw: Copy + Send + Sync;
    type PackedPoint: Packed + Default + Send + Sync;
    /// The curve whose GLV lattice a twiddle of this group decomposes against. The two
    /// groups share one `beta` and take different eigenvalues, so they take different
    /// lattices too; see `g16_gpu_layout::glv`.
    type Cfg: GLVConfig<ScalarField = Fr>;

    /// Names this group in an error, in the spelling `g16_msm::AccelError` uses.
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
    type Cfg = g16_field::g1::Config;

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
    type Cfg = g16_field::g2::Config;

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

/// The four FFT kernels, compiled once.
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
    active: usize,
    min_block: usize,
    block: u32,
    streams: usize,
    timing: bool,
}

impl FftKernels {
    /// Compiles `kernels::unit_fft()` and binds the four kernel handles.
    pub fn new(cuda: &Cuda) -> Result<Self, ProveError> {
        let module = Self::compile(cuda)?;
        Self::from_module(cuda, module)
    }

    /// Compile the FFT translation unit on its own, so a caller holding several
    /// commands' worth of work pays NVRTC once. Same split as `CudaMsm::compile`.
    pub fn compile(cuda: &Cuda) -> Result<Arc<CudaModule>, ProveError> {
        cuda.compile("fft", &kernels::unit_fft())
            .map_err(|e| bad(e.to_string()))
    }

    /// Bind the kernel handles out of an already-compiled module. `module` must have
    /// come from [`Self::compile`] against this same [`Cuda`], for the reason
    /// `CudaMsm::from_module` gives.
    ///
    /// Loads every variant the unit was assembled with: the shipped row always, the
    /// gated rows exactly when `G16_CUDA_FFT_VARIANTS=1`, because that same env decided
    /// whether their entry points exist in the module at all (`kernels::defines`).
    pub fn from_module(cuda: &Cuda, module: Arc<CudaModule>) -> Result<Self, ProveError> {
        let mut variants = Vec::new();
        for def in VARIANTS
            .iter()
            .filter(|d| !d.gated || kernels::fft_variants_enabled())
        {
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
                if kernels::fft_variants_enabled() {
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
    fn ifft_many<G: FftGroup>(&self, blocks: &mut [&mut [Xyzz<G::Raw>]]) -> Result<(), ProveError> {
        for a in blocks.iter() {
            let n = a.len();
            if n <= 1 {
                continue;
            }
            if !n.is_power_of_two() {
                return Err(bad(format!(
                    "group ifft over {}: {n} points is not a power of two",
                    G::OP
                )));
            }
            if n.trailing_zeros() > Fr::TWO_ADICITY {
                return Err(bad(format!(
                    "group ifft over {}: 2^{} is past the {}-bit two-adic subgroup",
                    G::OP,
                    n.trailing_zeros(),
                    Fr::TWO_ADICITY
                )));
            }
        }

        let pair = G::kernels(self);
        let point_words = core::mem::size_of::<G::PackedPoint>() / 4;

        // The stream each block's passes run on. One stream is the proven serial shape;
        // `G16_CUDA_FFT_STREAMS=N` round-robins the independent blocks over N streams so
        // a round of small blocks can fill the device instead of queuing behind each
        // other. No cross-stream ordering is needed: blocks are independent transforms
        // and each one's uploads, passes and download stay on its own stream, while the
        // shared `tw` upload below completes (host-blocking) before any launch is
        // issued anywhere.
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

        // Per-block ping-pong. `src` holds the input of the pass about to run and is
        // never written by it; `dst` starts zeroed (msm.rs's allocation policy: an
        // uninitialised limb array is a plausible field element) and every pass
        // overwrites all `n` of its slots.
        let mut live: Vec<Block> = Vec::with_capacity(blocks.len());
        for (bi, a) in blocks.iter().enumerate() {
            let n = a.len();
            if n <= 1 {
                continue;
            }
            let bits = n.trailing_zeros();
            let stream = pool[live.len() % pool.len()].clone();

            // `bit_reverse` (prepare.rs:256) folded into the pack, in gather form so the
            // loop splits over the pool: the permutation is an involution, so
            // `src[rev(i)] = a[i]` and `src[j] = a[rev(j)]` are the same map. Only this
            // direction is applied; the rotation at the bottom is the one that is NOT an
            // involution. Unlike Metal the pack cannot land in the device buffer
            // directly, so it lands in a `Vec` and one `memcpy_htod` follows.
            let mut packed = vec![G::PackedPoint::default(); n];
            packed
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, s)| *s = G::pack(&a[bit_reverse_index(j, bits)]));
            let src = upload_words(&stream, as_words(&packed))?;
            let dst = stream
                .alloc_zeros::<u32>(n * point_words)
                .map_err(|e| drv("allocate fft scratch", e))?;

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
                src,
                dst,
                stw,
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

        let mut times: Vec<PassTime> = Vec::new();
        for exp in 1..=max_bits {
            for b in live.iter().filter(|b| b.bits >= exp) {
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
                    (&pair.mix, &tw, params, threads)
                };
                let cfg = kernel.cfg_1d(threads, self.block);
                let start = self.timed_event(&b.stream)?;
                let mut lb = b.stream.launch_builder(&kernel.f);
                lb.arg(&b.src).arg(&b.dst).arg(third).arg(&params);
                // SAFETY: four parameters bound in order and with matching types; `src`
                // and `dst` each hold `n` points of this group, every in-bounds index
                // the kernel touches is under `n`, the twiddle table holds at least
                // `(n/2) * GLV_WORDS` words by construction, and the surplus threads of
                // the last CUDA block exit on the kernel's own bounds guard.
                unsafe { lb.launch(cfg) }.map_err(|e| drv("launch fft mix", e))?;
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
            }
            // Launches on one stream execute in issue order, so pass `exp + 1` of a
            // block reads what its pass `exp` wrote with no event and no host wait; the
            // swap is pure host bookkeeping.
            for b in live.iter_mut().filter(|b| b.bits >= exp) {
                core::mem::swap(&mut b.src, &mut b.dst);
            }
        }

        for b in &live {
            // The one wait per block: `download` synchronizes the block's stream before
            // the `Vec` is read, which also orders it after every launch above.
            let words = download(&b.stream, &b.src)?;
            let got = from_words::<G::PackedPoint>(&words)
                .ok_or_else(|| bad("fft read back is not a whole number of points"))?;

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
                    "g16-cuda fft-time: {} block {} n=2^{} pass {:>2}{} threads {:>8} {:>10.3} ms",
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
                "g16-cuda fft-time: {} device sum {:.3} ms over {} launches \
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
    use super::*;
    use g16_gpu_layout::PackedGlv;

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
        let beta = <g16_field::g1::Config as GLVConfig>::ENDO_COEFFS[0];
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
    /// exists as a unit; see `g16_gpu_kernels::unit_fft`.
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
}
