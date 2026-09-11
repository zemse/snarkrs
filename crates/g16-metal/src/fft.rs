//! The group inverse FFT behind `ptau prepare`, the slowest command in the project.
//!
//! `shaders/fft.metal` and this file are one unit and must be changed together, the same
//! contract `layout.rs` states for the wire structs. What lives here is the shape of the
//! submission, the twiddle table, and the two permutations that a kernel deliberately does
//! not do.
//!
//! The CPU twin is `g16_ceremony::prepare::ifft` (prepare.rs:320), and the reason only that
//! one function is ported is that a profile of the command put 99.2% of it inside
//! `point_times_fr` (prepare.rs:149). `batch_to_affine` is 0.07% and the file I/O is
//! 0.05%, so both stay on the host permanently; moving `batch_to_affine` would buy 0.07%
//! and force a device field inversion into the design for it.
//!
//! The result agrees with the CPU as a curve point, not limb for limb, because the ladder
//! here is GLV over a fixed signed window and the ladder there is wNAF over the whole
//! 254-bit scalar, and XYZZ is projective on top of that. The
//! ceremony's byte-identity survives that for one reason: `lagrange_evaluations`
//! (prepare.rs:414) ends every path in `batch_to_affine`, and affine is canonical. So
//! `cmp` of a `--backend cpu` output against a `--backend metal` one is a total check on
//! this file and there is nothing to keep in step.

use ark_ec::scalar_mul::glv::GLVConfig;
use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{FftField, Field, Fr};
use g16_gpu_layout::glv::twiddle_table;
use g16_msm::xyzz::Xyzz;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLDispatchType, MTLResourceOptions,
};
use rayon::prelude::*;

use crate::ceremony::{cer_err, cer_read_back, dispatch_1d, env_window, window_index, CER_WINDOWS};
use crate::kernels::{CEREMONY_MSL, FFT_MSL, FR_MSL, MSM_MSL};
use crate::layout::{as_bytes, Packed, PackedFq, PackedFq2};
use crate::msm::{PackedXyzzG1, PackedXyzzG2};

/// Ladder window for G1, and one wider than the ceremony ladders take.
///
/// `ceremony::WINDOW_G1` is 4, off a sweep of isolated multiplications that `WINDOW_G2`
/// there says cannot be quoted. This one is swept through a whole `ptau prepare` on
/// `ppot_0080_16.ptau`, which is what the constant is for, and it was re-swept when the
/// ladder became GLV, because that is a different shape: half the doublings, two additions
/// a window instead of one, and one table serving both halves. Medians of four warm runs
/// on this M2 Max, round-robin over the widths so drift hits each of them equally:
///
/// | c | wall s |
/// |---|---:|
/// | 2 | 6.97 |
/// | 3 | 5.16 |
/// | 4 | 4.54 |
/// | 5 | **4.37** |
///
/// Crossing the two widths separately, medians of three: (g1, g2) of (4,5) is 4.416,
/// (5,4) 4.429, (5,5) 4.335. Both groups want 5 and neither is carrying the other, and
/// that reads more cleanly than the pre-GLV sweep did: the crossed pair separates from
/// (5,5) by about 2% against a 0.7% spread within (5,5).
///
/// GLV not moving the optimum is the useful part of this. The table is `2^(c-1)` multiples
/// either way, since the endomorphism side reads the same one, so what a width costs in
/// registers is unchanged and only the ratio of doublings to additions moved. c=6 is still
/// not compiled: `CER_WINDOWS` stops at 5, and 32 entries of a 192-byte `Jac<Fq2>` is 6 KB
/// a thread on G2.
const FFT_WINDOW_G1: u32 = 5;
/// See [`FFT_WINDOW_G1`]; G2 wants the same width and gains about as much from it.
const FFT_WINDOW_G2: u32 = 5;

/// Blocks shorter than this go back to the CPU.
///
/// Not a crossover measurement, and deliberately not tuned to one. On paper the GPU is
/// still ahead of the CPU at 2^10, and what the paper leaves out is the buffer
/// allocation, the pipeline binding and the repack, each tens of microseconds, plus a
/// 2^9-thread dispatch not filling 38 cores. The reason to stop caring at 2^12 is the
/// distribution rather than the crossover: a section's work is concentrated in its top
/// two blocks, the blocks under 2^12 are the same twelve at every power, and since
/// `process_section` (prepare.rs:673) gives them their own worker thread they no longer
/// serialize with the device at all. The threshold's only real job is to keep the GPU
/// path from being embarrassing on a power-8 test file. [`FftKernels::with_min_block`]
/// moves it, which is how the test suite forces the device path onto blocks a real ptau
/// would route home, and `G16_METAL_FFT_MIN_BLOCK` moves it from outside the process,
/// which is how the crossover is swept from the CLI without a rebuild.
const MIN_BLOCK: usize = 1 << 12;

/// Threads per threadgroup asked for, clamped by [`dispatch_1d`] to what the pipeline
/// reports. Same preference the ceremony ladders use: the kernel is one long dependent
/// chain per thread, so occupancy comes from the grid, not the group.
const THREADGROUP: usize = 64;

/// Attempts at one command buffer before giving up. See
/// [`FftKernels::dispatch_with_retry`]: a macOS interactivity kill is a scheduling event,
/// not an arithmetic one, and a power-20 `ptau prepare` is seventeen minutes of work to
/// throw away over one.
const RETRIES: u32 = 4;

/// Mirrors `struct FftParams` in `shaders/fft.metal`. Passed by `setBytes`, which copies
/// at encode time, so a re-submitted command buffer carries its own values rather than
/// whatever the next piece of the pass overwrote them with.
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

/// One block's pass in a round: what [`FftKernels::run_round`] batches into command
/// buffers, split across several if the round is bigger than the budget. `third` is the
/// round's shared twiddle table for a mix pass and the block's own `1/n`-scaled table
/// for the fused final pass. `ladders_per_thread` is the pass's weight against the
/// ladder budget: 1 for a mix pass, 2 for the fused one, so a command buffer's duration
/// does not double where every thread runs two ladders.
struct Pass<'a> {
    pso: &'a ComputePipelineState,
    src: &'a Buffer,
    dst: &'a Buffer,
    third: &'a Buffer,
    params: FftParams,
    threads: usize,
    ladders_per_thread: usize,
}

/// One block's live state across the lockstep rounds of [`FftKernels::ifft_many`]: its
/// ping-pong pair, its own `1/n`-scaled twiddle table, and where in the caller's slice
/// of blocks it came from.
struct Block {
    idx: usize,
    n: usize,
    bits: u32,
    src: Buffer,
    dst: Buffer,
    stw: Buffer,
}

/// The pipelines for one group, parallel to [`CER_WINDOWS`].
struct FftPipelines {
    mix: Vec<ComputePipelineState>,
    mix_scale: Vec<ComputePipelineState>,
}

/// What the two kernels need to know about a group, so the submission code is written
/// once.
///
/// A second, smaller twin of `ceremony::CerGroup` rather than a use of it. The two overlap
/// only in the packing, and keeping this module's marker types here is what lets the whole
/// FFT be added without touching the ladder and apply-key code it shares a crate with.
trait FftGroup {
    type Raw: Copy + Send + Sync;
    type PackedPoint: Packed + Default + Send + Sync;
    /// The curve whose GLV lattice a twiddle of this group decomposes against. The two
    /// groups share one `beta` and take different eigenvalues, so they take different
    /// lattices too; see item 4 of `shaders/fft.metal`.
    type Cfg: GLVConfig<ScalarField = Fr>;

    /// Names this group in an error, in the spelling [`g16_msm::AccelError`] uses.
    const OP: &'static str;
    /// Ladder invocations one command buffer may hold, summed across its pieces. See
    /// [`FftKernels::run_round`].
    ///
    /// Set by the duration it buys. `GPUStartTime` to `GPUEndTime` on every buffer of a
    /// `ppot_0080_17.ptau` prepare on this M2 Max gives 0.559 us a G1 ladder and 1.994 us
    /// a G2 ladder, both flat to within 5% from the 10th to the 90th percentile over the
    /// buffers carrying at least 32k ladders, so a budget's duration is just the budget
    /// times its group's rate and that rate is the whole of the difference between the two
    /// numbers below.
    ///
    /// Those are the GLV ladder's rates, 32% and 30% under the ones the pair below was set
    /// against, which were measured before it. The pair still stands: what a budget costs
    /// is a function of its THREAD count, per the sweep below, and GLV changed how long a
    /// thread takes and not how many threads a budget buys. So the same numbers now buy
    /// shorter command buffers at the same price, which is the direction that helps.
    ///
    /// What a budget COSTS, though, does not depend on the group. `G16_METAL_FFT_BUDGET_G1`
    /// and `_G2` swept one at a time over `ppot_0080_17.ptau`, each column that group's own
    /// GPU-busy total against the same total at a budget loose enough not to bind there at
    /// all (2^18 for G1, 2^17 for G2), medians of 3 round-robin:
    ///
    /// | budget | G1 | G2 |
    /// |---|---:|---:|
    /// | 2^16 | +5.6% | +4.7% |
    /// | 2^15 | +17.8% | +16.8% |
    /// | 2^14 | +34.4% | +36.2% |
    ///
    /// The two columns agree because the loss is the ramp and drain of a dispatch that no
    /// longer fills the device, which follows the thread count, and `ladders_per_thread` is
    /// 1 or 2 in either group. So the price of a budget is its own value, not the group's,
    /// and the 4:1 ratio the two constants keep buys equal duration per buffer at very
    /// unequal cost: nearly all of what the shipped pair costs is G2's.
    const BUDGET: usize;
    /// Bytes one packed point takes, for the ping-pong scratch allocation.
    const SCRATCH: usize;

    fn pack(p: &Xyzz<Self::Raw>) -> Self::PackedPoint;
    fn unpack(p: &Self::PackedPoint) -> Xyzz<Self::Raw>;
    fn pipelines(k: &FftKernels) -> &FftPipelines;
    fn window(k: &FftKernels) -> u32;
    fn budget(k: &FftKernels) -> Option<usize>;
}

struct FftG1;
struct FftG2;

impl FftGroup for FftG1 {
    type Raw = RawFq;
    type PackedPoint = PackedXyzzG1;
    type Cfg = g16_field::g1::Config;

    const OP: &'static str = "G1";
    // 2^19 ladders was 428 ms a command buffer. Measured at power 19, where a 2^20-point
    // block is the first to reach it: 132 buffers, 208 ms median, 422 ms at the top. 2^18
    // caps the top at 216 ms, which is the median it already had, for +0.4% of G1's
    // GPU-busy time there. Those are the pre-GLV ladder's; on this one 2^18 measured 74 ms
    // median and 148 ms at the top over the 52 G1 buffers of a power-17 run.
    const BUDGET: usize = 1 << 18;
    const SCRATCH: usize = core::mem::size_of::<PackedXyzzG1>();

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

    fn pipelines(k: &FftKernels) -> &FftPipelines {
        &k.g1
    }

    fn window(k: &FftKernels) -> u32 {
        k.window_g1
    }

    fn budget(k: &FftKernels) -> Option<usize> {
        k.budget_g1
    }
}

impl FftGroup for FftG2 {
    type Raw = RawFq2;
    type PackedPoint = PackedXyzzG2;
    type Cfg = g16_field::g2::Config;

    const OP: &'static str = "G2";
    // A quarter of G1's, because a G2 ladder is 3.5 times a G1 one and the two are meant to
    // produce the same buffer; see `FftG1::BUDGET`. 2^17 was 350 ms median and 390 ms at
    // the top at power 19, and that is the size macOS objects to: at power 17 on a machine
    // also driving a display it lost 5 of 6 runs to `ImpactingInteractivity` after all four
    // retries, where every budget at or below 2^16 finished 24 of 24 in the same window.
    // 2^16 is 179 ms median, 210 ms at the top, for +3.7% of G2's GPU-busy time. Those are
    // the pre-GLV ladder's too; on this one 2^16 measured 129 ms median and 141 ms at the
    // top over the 33 G2 buffers of a power-17 run, and the 3.5x ratio that sets the pair
    // is unchanged at 3.57.
    const BUDGET: usize = 1 << 16;
    const SCRATCH: usize = core::mem::size_of::<PackedXyzzG2>();

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

    fn pipelines(k: &FftKernels) -> &FftPipelines {
        &k.g2
    }

    fn window(k: &FftKernels) -> u32 {
        k.window_g2
    }

    fn budget(k: &FftKernels) -> Option<usize> {
        k.budget_g2
    }
}

/// The two FFT kernels, compiled once.
///
/// Compiling the library is the expensive call (runtime MSL compilation plus 16 pipelines)
/// and it belongs once at the top of a command, exactly as [`crate::msm::MetalMsm::new`]
/// and [`crate::ceremony::CeremonyKernels::new`] do. Nothing here holds per-block state,
/// so one instance serves a whole `ptau prepare`.
pub struct FftKernels {
    device: Device,
    queue: CommandQueue,
    g1: FftPipelines,
    g2: FftPipelines,
    window_g1: u32,
    window_g2: u32,
    min_block: usize,
    budget_g1: Option<usize>,
    budget_g2: Option<usize>,
}

impl FftKernels {
    pub fn new() -> Result<Self, ProveError> {
        let device = Device::system_default()
            .ok_or_else(|| cer_err("no Metal device; this machine cannot run the metal backend"))?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, ProveError> {
        // One translation unit, in dependency order: the `Fr` prelude, the point
        // arithmetic, the ladder, then this file. Each has its own header guard, so the
        // concatenation is the include, and `fft.metal` adds no curve math of its own.
        let source = format!("{FR_MSL}\n{MSM_MSL}\n{CEREMONY_MSL}\n{FFT_MSL}\n");
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

        let group = |g: &str| -> Result<FftPipelines, ProveError> {
            let mut mix = Vec::with_capacity(CER_WINDOWS.len());
            let mut mix_scale = Vec::with_capacity(CER_WINDOWS.len());
            for c in CER_WINDOWS {
                mix.push(pso(&format!("fft_mix_{g}_c{c}"))?);
                mix_scale.push(pso(&format!("fft_mix_scale_{g}_c{c}"))?);
            }
            Ok(FftPipelines { mix, mix_scale })
        };

        let g1 = group("g1")?;
        let g2 = group("g2")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            g1,
            g2,
            window_g1: env_window("G16_METAL_FFT_C_G1", FFT_WINDOW_G1),
            window_g2: env_window("G16_METAL_FFT_C_G2", FFT_WINDOW_G2),
            min_block: env_min_block(),
            budget_g1: env_budget("G16_METAL_FFT_BUDGET_G1"),
            budget_g2: env_budget("G16_METAL_FFT_BUDGET_G2"),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Shortest block this instance will run on the device. See [`MIN_BLOCK`].
    pub fn min_block(&self) -> usize {
        self.min_block
    }

    /// Moves the crossover. A test passes 1 so that a power-10 file, whose largest block
    /// is 2^11, reaches the device at all; a measurement passes a sweep. Consuming rather
    /// than a setter because the pipelines are already built and nothing else about the
    /// instance changes.
    pub fn with_min_block(mut self, n: usize) -> Self {
        self.min_block = n;
        self
    }

    /// Overrides [`FftGroup::BUDGET`], the ladders one command buffer may hold.
    ///
    /// Both groups at once, which is what a test wants and a sweep does not; a sweep uses
    /// the two [`env_budget`] variables. The tests set it small because the shipped budget
    /// is 2^18 ladders on G1, so a block big enough to split a pass across command buffers
    /// on its own is a block too big to put in a unit test, and the `gid_off` arithmetic
    /// would go untested at every size the suite can afford to run.
    pub fn with_budget(mut self, ladders: usize) -> Self {
        self.budget_g1 = Some(ladders.max(1));
        self.budget_g2 = Some(ladders.max(1));
        self
    }

    /// The ladder window this instance uses for G1. See [`FFT_WINDOW_G1`].
    pub fn window_g1(&self) -> u32 {
        self.window_g1
    }

    pub fn window_g2(&self) -> u32 {
        self.window_g2
    }

    /// In-place inverse FFT over G1, the whole of what `prepare::ifft` (prepare.rs:320)
    /// does and in the same order.
    pub fn ifft_g1(&self, a: &mut [Xyzz<RawFq>]) -> Result<(), ProveError> {
        self.ifft_many::<FftG1>(&mut [a])
    }

    /// [`Self::ifft_g1`] over G2, which is 44% of `ptau prepare` on 20% of its scalar
    /// multiplications.
    pub fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), ProveError> {
        self.ifft_many::<FftG2>(&mut [a])
    }

    /// [`Self::ifft_g1`] over several independent blocks, their passes co-dispatched.
    pub fn ifft_g1_many(&self, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), ProveError> {
        self.ifft_many::<FftG1>(blocks)
    }

    pub fn ifft_g2_many(&self, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), ProveError> {
        self.ifft_many::<FftG2>(blocks)
    }

    /// The transform over one or more independent blocks, in lockstep rounds: round `exp`
    /// is pass `exp` of every block still deep enough to have one, and a block's own
    /// depth `exp == bits` is its fused mix+scale pass, so each block retires in its own
    /// last round and no scaling round follows.
    ///
    /// Co-dispatching is where the rounds pay. A ptau section is one block per power, so
    /// no pass of a power-15 section dispatches more than 33k threads and most dispatch
    /// far fewer, on a device that does not near peak until several times that are in
    /// flight; one block at a time leaves it mostly idle at exactly the powers where the
    /// CPU fallback does not already cover the loss. A round holds one pass of every
    /// block, and co-dispatch alone (before the dense grid and the fused pass) was a
    /// whole ppot_0080_15 prepare in 3.61 s against 4.78 s (medians of 3, M2 Max).
    /// Blocks of different depths co-exist because the rounds align on the pass
    /// exponent: a block joins every round up to its own depth and then waits, finished,
    /// in its `src` buffer while the deeper blocks run on.
    fn ifft_many<G: FftGroup>(&self, blocks: &mut [&mut [Xyzz<G::Raw>]]) -> Result<(), ProveError> {
        for a in blocks.iter() {
            let n = a.len();
            if n <= 1 {
                continue;
            }
            if !n.is_power_of_two() {
                return Err(cer_err(format!(
                    "group ifft over {}: {n} points is not a power of two",
                    G::OP
                )));
            }
            if n.trailing_zeros() > Fr::TWO_ADICITY {
                return Err(cer_err(format!(
                    "group ifft over {}: 2^{} is past the {}-bit two-adic subgroup",
                    G::OP,
                    n.trailing_zeros(),
                    Fr::TWO_ADICITY
                )));
            }
        }

        let pipelines = G::pipelines(self);
        let idx = window_index(G::window(self));
        let mix = &pipelines.mix[idx];
        let mix_scale = &pipelines.mix_scale[idx];

        // Per-block ping-pong. `src` holds the input of the pass about to run and is
        // never written by it, which is the whole reason a killed command buffer can
        // simply be re-run; `dst` is scratch and its contents after a failure are not
        // looked at.
        let mut live: Vec<Block> = Vec::with_capacity(blocks.len());
        for (bi, a) in blocks.iter().enumerate() {
            let n = a.len();
            if n <= 1 {
                continue;
            }
            let bits = n.trailing_zeros();
            let src = self.scratch(n * G::SCRATCH);
            let dst = self.scratch(n * G::SCRATCH);

            // The input goes straight into the buffer rather than into a `Vec` that is
            // then copied in. Metal buffers here are `StorageModeShared` (msm.rs:460), so
            // the two are the same memory and the copy would be a second 268 MB of
            // traffic per block at power 20 for nothing.
            //
            // `bit_reverse` (prepare.rs:256) is folded into that write, in gather form
            // so the loop splits over the pool: the permutation is an involution, so
            // `src[rev(i)] = a[i]` and `src[j] = a[rev(j)]` are the same map. Only this
            // direction is applied, and the rotation at the bottom is the one that is
            // NOT an involution.
            //
            // SAFETY: the buffer was just allocated with room for `n` of this type and
            // nothing has been encoded against it, so no dispatch can be reading it.
            let slots: &mut [G::PackedPoint] =
                unsafe { core::slice::from_raw_parts_mut(src.contents().cast(), n) };
            slots
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, s)| *s = G::pack(&a[bit_reverse_index(j, bits)]));

            let size_inv = Fr::from(n as u64)
                .inverse()
                .expect("a power of two is a unit mod r");
            // The `1/n` scaling rides the block's last mix pass (see
            // `fft_mix_scale_impl`), which reads a second table with `s = 1/n` folded
            // into every entry, `stw[0]` doubling as the plain `[s]` the `lo` side
            // needs. `s` differs per block, so unlike `tw` this table cannot be shared
            // across a round; it costs 16 bytes a point next to the ping-pong pair's
            // 256 or 512.
            let stw = self.buffer(&twiddle_table::<G::Cfg>(bits, size_inv));
            live.push(Block {
                idx: bi,
                n,
                bits,
                src,
                dst,
                stw,
            });
        }
        let Some(max_bits) = live.iter().map(|b| b.bits).max() else {
            return Ok(());
        };

        // One plain table serves every mix pass of every block. Pass `exp` wants
        // `roots[exp]^j`, and `roots[exp] == W^(2^(max_bits - exp))` for `W` the
        // primitive `2^max_bits`-th root whatever the block's own depth is, so
        // `tw_shift = max_bits - exp` reads the deepest block's table at the right
        // stride for all of them and no block carries its own. Only the fused final
        // passes read their per-block `stw` instead.
        let tw = self.buffer(&twiddle_table::<G::Cfg>(max_bits, Fr::ONE));

        for exp in 1..=max_bits {
            let round: Vec<Pass> = live
                .iter()
                .filter(|b| b.bits >= exp)
                .map(|b| {
                    if b.bits == exp {
                        // The block's last pass: the mix with `1/n` fused in (see
                        // `fft_mix_scale_impl`), dense over all `n/2` butterflies at
                        // two ladders each, on the block's own scaled table.
                        Pass {
                            pso: mix_scale,
                            src: &b.src,
                            dst: &b.dst,
                            third: &b.stw,
                            params: FftParams {
                                n: b.n as u32,
                                ..Default::default()
                            },
                            threads: b.n / 2,
                            ladders_per_thread: 2,
                        }
                    } else {
                        // The mix grid is dense over the `j != 0` butterflies (see
                        // `fft_mix_impl`): `groups * (span - 1)` ladder threads, except
                        // at `exp == 1`, where every butterfly is `j == 0` and the pass
                        // is all of them.
                        let threads = if exp == 1 {
                            b.n / 2
                        } else {
                            b.n / 2 - (b.n >> exp)
                        };
                        Pass {
                            pso: mix,
                            src: &b.src,
                            dst: &b.dst,
                            third: &tw,
                            params: FftParams {
                                n: b.n as u32,
                                span: 1 << (exp - 1),
                                log_span: exp - 1,
                                tw_shift: max_bits - exp,
                                gid_off: 0,
                            },
                            threads,
                            ladders_per_thread: 1,
                        }
                    }
                })
                .collect();
            self.run_round::<G>(&round)?;
            for b in live.iter_mut().filter(|b| b.bits >= exp) {
                core::mem::swap(&mut b.src, &mut b.dst);
            }
        }

        for b in &live {
            // SAFETY: the command buffer completed, `n` points of this type is exactly
            // what the last pass wrote into what is now `src`, and `PackedPoint` is
            // `Packed`, so every bit pattern is valid.
            let got: &[G::PackedPoint] = unsafe { cer_read_back(&b.src, b.n) };

            // The `a[1..].reverse()` that finishes the inverse (prepare.rs:340), folded
            // into the read back: `ifft(a)[0] = X[0]/n` and `ifft(a)[i] = X[n-i]/n`.
            // Splitting this off from the scaling gives an answer that is a rotation away
            // from correct and still looks plausible, which is why the module doc there
            // says so twice.
            let a = &mut *blocks[b.idx];
            a[0] = G::unpack(&got[0]);
            let n = b.n;
            a[1..].par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = G::unpack(&got[n - 1 - i]);
            });
        }
        Ok(())
    }

    /// Run one round, each pass `src` to `dst`, and retry what macOS kills.
    ///
    /// The obvious design was one command buffer for a whole block: consecutive
    /// dispatches on a serial compute encoder are already ordered with an implicit
    /// barrier, so one commit-and-wait per block rather than one per budget piece was
    /// there for the taking. macOS took it back. A submission that keeps the GPU busy
    /// while the GPU is also driving a display returns
    /// `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, and it is NOT a duration
    /// limit that a smaller buffer stays under. Measured on this M2 Max: the same
    /// 2^19-point G2 block was killed 6.8 s into a run, and finished in 16.6 s with
    /// exactly the same command-buffer split when the machine was quiet; a power-20 run
    /// died on buffers an order of magnitude smaller still. The trigger is contention with
    /// whatever else wants the GPU, so this survives the kill instead of trying to avoid
    /// it.
    ///
    /// Surviving it is what fixes the round boundary in place. Every `src` in the round
    /// is read-only for the whole round and every `dst` is scratch, so a killed command
    /// buffer has damaged only destinations and re-running it is exact. Two passes of one
    /// block in one command buffer would not be, because the second has already
    /// overwritten the first one's input; one pass each of several blocks is, because the
    /// pairs are disjoint.
    ///
    /// [`FftGroup::BUDGET`] splits a round into command buffers of at most that many
    /// summed ladders, a fused piece counting two a thread, a pass bigger than the
    /// budget splitting on `gid_off`. Smaller buffers do not avoid the kill, so that is a
    /// throughput and blast-radius knob rather than a safety one: what it decides is how
    /// much work one kill throws away, against a floor with a measured reason. At 1,024
    /// ladders a command buffer a 2^19-point G2 block takes 138 s against 16.6 s, because a
    /// dispatch that small does not fill the device.
    ///
    /// Every commit goes through `cb::wait_ok` (cb.rs:57). A faulted buffer that went
    /// unnoticed would leave the previous pass's points in a `dst`, which is a wrong
    /// point in a file that is otherwise perfectly formed.
    fn run_round<G: FftGroup>(&self, round: &[Pass]) -> Result<(), ProveError> {
        let budget = G::budget(self).unwrap_or(G::BUDGET);
        let mut batch: Vec<(&Pass, FftParams, usize)> = Vec::new();
        let mut used = 0usize;
        for pass in round {
            let mut done = 0usize;
            while done < pass.threads {
                // Room in threads at this pass's weight, so a fused piece spends the
                // budget twice as fast as a mix piece and a command buffer's duration
                // stays flat across the mixture.
                let take = ((budget - used) / pass.ladders_per_thread).min(pass.threads - done);
                if take == 0 && !batch.is_empty() {
                    self.dispatch_with_retry(&batch)?;
                    batch.clear();
                    used = 0;
                    continue;
                }
                // Floored at 1 so a test budget below one fused thread still advances.
                let take = take.max(1);
                let p = FftParams {
                    gid_off: done as u32,
                    ..pass.params
                };
                batch.push((pass, p, take));
                used += take * pass.ladders_per_thread;
                done += take;
                if used >= budget {
                    self.dispatch_with_retry(&batch)?;
                    batch.clear();
                    used = 0;
                }
            }
        }
        if !batch.is_empty() {
            self.dispatch_with_retry(&batch)?;
        }
        Ok(())
    }

    /// One command buffer holding one round's worth of pieces, re-submitted unchanged if
    /// it comes back anything but `Completed`.
    ///
    /// Retrying blind rather than on the interactivity error specifically: the error text
    /// is not an API, and a fault that is genuinely the kernel's (an out-of-range index,
    /// a lost device) fails all four attempts and is reported with the last message. The
    /// backoff exists because the failure means something else wants the GPU, and coming
    /// straight back with the same work is how a `ptau prepare` at power 20 loses a
    /// seventeen-minute run to a window being dragged.
    fn dispatch_with_retry(&self, batch: &[(&Pass, FftParams, usize)]) -> Result<(), ProveError> {
        let mut err = None;
        for attempt in 0..RETRIES {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(200 << attempt));
            }
            let cb = self.queue.new_command_buffer();
            // Concurrent rather than the serial default: the pieces touch disjoint
            // buffer pairs, or disjoint index ranges of one pair, so there is no hazard
            // for the implicit serial barrier to protect, and with the barrier in place
            // the co-dispatched blocks would run one after another, which is exactly the
            // underfill co-dispatching exists to remove.
            let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            for (pass, p, threads) in batch {
                enc.set_compute_pipeline_state(pass.pso);
                enc.set_buffer(0, Some(pass.src), 0);
                enc.set_buffer(1, Some(pass.dst), 0);
                enc.set_buffer(2, Some(pass.third), 0);
                set_params(enc, 3, p);
                dispatch_1d(enc, pass.pso, *threads, THREADGROUP);
            }
            enc.end_encoding();
            cb.commit();
            match crate::cb::wait_ok(cb, "ceremony group inverse fft") {
                Ok(()) => return Ok(()),
                Err(e) => err = Some(e),
            }
        }
        Err(err.expect("RETRIES is at least 1"))
    }

    fn scratch(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
    }

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
}

/// `i` with its low `bits` bits reversed, the permutation `bit_reverse` (prepare.rs:256)
/// applies by swapping in place.
#[inline]
fn bit_reverse_index(i: usize, bits: u32) -> usize {
    ((i as u32).reverse_bits() >> (u32::BITS - bits)) as usize
}

/// `G16_METAL_FFT_MIN_BLOCK` overrides [`MIN_BLOCK`]. Only the tests set it, and they set
/// it to 0 so that a block small enough to run in a second still takes the device path;
/// with the default in force a power-13 file would be compared against itself.
fn env_min_block() -> usize {
    env_usize("G16_METAL_FFT_MIN_BLOCK").unwrap_or(MIN_BLOCK)
}

/// `G16_METAL_FFT_BUDGET_G1` and `_G2` override [`FftGroup::BUDGET`] for one group and
/// `G16_METAL_FFT_BUDGET` overrides both, which is how the two were crossed from the CLI
/// without a rebuild. Split the same way and for the same reason as `G16_METAL_FFT_C_G1`:
/// the two groups do not have to move together, and the sweep in [`FftGroup::BUDGET`]
/// holds one of them still while the other moves. Floored at 1, since `run_round` advances
/// by at most the budget and a budget of 0 would hang the command.
fn env_budget(var: &str) -> Option<usize> {
    env_usize(var)
        .or_else(|| env_usize("G16_METAL_FFT_BUDGET"))
        .map(|n| n.max(1))
}

fn env_usize(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// `setBytes` for the params struct. `FftParams` is `repr(C)` and all-`u32`, so its bytes
/// are exactly what the shader's `constant FftParams&` reads.
fn set_params(enc: &ComputeCommandEncoderRef, index: u64, p: &FftParams) {
    enc.set_bytes(
        index,
        core::mem::size_of::<FftParams>() as u64,
        (p as *const FftParams).cast(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PackedGlv;
    use g16_ceremony::prepare::batch_to_affine;
    use g16_ceremony::CpuGroupFft;
    use g16_field::{AffineRepr, CurveGroup, Fq, G1Affine, G2Affine};
    use g16_msm::GroupFft;

    /// Every width in [`CER_WINDOWS`] has a kernel pair, and both defaults are among them.
    /// Construction would catch a missing kernel, but only on a machine with a device;
    /// this runs anywhere.
    #[test]
    fn every_compiled_window_has_a_kernel_pair() {
        for c in CER_WINDOWS {
            for g in ["g1", "g2"] {
                assert!(
                    FFT_MSL.contains(&format!("FFT_KERNELS({g}_c{c},")),
                    "shaders/fft.metal has no FFT_KERNELS line for {g} c={c}, so \
                     fft_mix_{g}_c{c} does not exist and the pipeline will not build"
                );
            }
        }
        assert!(CER_WINDOWS.contains(&FFT_WINDOW_G1));
        assert!(CER_WINDOWS.contains(&FFT_WINDOW_G2));
    }

    /// The GLV constants live in two languages. `beta` is the one that can be wrong
    /// silently: a table entry decomposed against arkworks' lattice and a kernel that
    /// multiplies X by some other cube root of one still produces a point on the curve,
    /// in the right subgroup, and wrong.
    #[test]
    fn msl_declares_the_same_glv_constants() {
        let beta = <g16_field::g1::Config as GLVConfig>::ENDO_COEFFS[0];
        let want = format!(
            "constant uint FQ_BETA[8] = {{ {} }};",
            PackedFq::from_fq(&beta)
                .v
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            FFT_MSL.contains(&want),
            "shaders/fft.metal does not contain the line:\n{want}"
        );
        for (name, want) in [
            ("GLV_WORDS", core::mem::size_of::<PackedGlv>() / 4),
            ("GLV_RECODE_BITS", 128),
        ] {
            let line = format!("#define {name} {want}");
            assert!(
                FFT_MSL.contains(&line),
                "shaders/fft.metal does not contain the line:\n{line}"
            );
        }
    }

    /// Deterministic draws, so a failure is reproducible without a seed to record. Same
    /// generator and same reason as `ceremony::tests::Lcg`.
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

    fn kernels() -> Option<FftKernels> {
        // No device on a headless builder is a skip, not a failure, the same way the
        // artifact-gated tests in `msm.rs` skip.
        Device::system_default()?;
        Some(FftKernels::new().expect("fft kernels"))
    }

    /// The comparison the byte-identity claim rests on: affine, which is canonical, and
    /// which is what `lagrange_evaluations` (prepare.rs:414) writes to the file.
    fn same_g1(got: &[Xyzz<RawFq>], want: &[Xyzz<RawFq>]) -> bool {
        batch_to_affine::<g16_field::g1::Config>(got)
            == batch_to_affine::<g16_field::g1::Config>(want)
    }

    fn same_g2(got: &[Xyzz<RawFq2>], want: &[Xyzz<RawFq2>]) -> bool {
        batch_to_affine::<g16_field::g2::Config>(got)
            == batch_to_affine::<g16_field::g2::Config>(want)
    }

    /// Every block size from 2^0 to 2^12, which is every size a ptau file below power 12
    /// contains and the shape of every pass count from zero to thirteen.
    #[test]
    fn ifft_g1_matches_the_cpu_at_every_block_size() {
        let Some(k) = kernels() else { return };
        for bits in 0..=12u32 {
            let n = 1usize << bits;
            let mut got = block_g1(0x5eed_1000 + u64::from(bits), n);
            let mut want = got.clone();
            k.ifft_g1(&mut got).expect("ifft_g1");
            CpuGroupFft.ifft_g1(&mut want).expect("cpu ifft_g1");
            assert!(same_g1(&got, &want), "G1 block of 2^{bits} disagrees");
        }
    }

    #[test]
    fn ifft_g2_matches_the_cpu_at_every_block_size() {
        let Some(k) = kernels() else { return };
        for bits in 0..=10u32 {
            let n = 1usize << bits;
            let mut got = block_g2(0x5eed_2000 + u64::from(bits), n);
            let mut want = got.clone();
            k.ifft_g2(&mut got).expect("ifft_g2");
            CpuGroupFft.ifft_g2(&mut want).expect("cpu ifft_g2");
            assert!(same_g2(&got, &want), "G2 block of 2^{bits} disagrees");
        }
    }

    /// A block that is entirely the point at infinity, which is what `ptau new` produces
    /// before any contribution: `tau == 1` makes every input the generator and most
    /// outputs the identity. A ladder that starts its accumulator anywhere but infinity,
    /// or a butterfly that reads a zero `ZZ` as a real coordinate, fails here and nowhere
    /// else.
    #[test]
    fn an_all_infinity_block_stays_infinity() {
        let Some(k) = kernels() else { return };
        let n = 256;
        let mut got = vec![Xyzz::<RawFq>::ZERO; n];
        k.ifft_g1(&mut got).expect("ifft_g1");
        assert!(got.iter().all(|p| p.is_zero()), "infinity did not survive");
    }

    /// Every compiled window reaches the same points. The ladder is free to use any
    /// window because the exit is affine and canonical, and this is the assertion that
    /// says so rather than assuming it, which is also what makes a sweep of
    /// `G16_METAL_FFT_C_G1` a measurement rather than a correctness gamble.
    #[test]
    fn every_window_gives_the_same_points() {
        let Some(_) = kernels() else { return };
        let device = Device::system_default().expect("device");
        let n = 512;
        let base_g1 = block_g1(0x5eed_3001, n);
        let base_g2 = block_g2(0x5eed_3002, n);

        let mut want_g1 = base_g1.clone();
        let mut want_g2 = base_g2.clone();
        CpuGroupFft.ifft_g1(&mut want_g1).expect("cpu ifft_g1");
        CpuGroupFft.ifft_g2(&mut want_g2).expect("cpu ifft_g2");

        for c in CER_WINDOWS {
            let mut k = FftKernels::with_device(device.clone()).expect("fft kernels");
            k.window_g1 = c;
            k.window_g2 = c;
            let mut got_g1 = base_g1.clone();
            let mut got_g2 = base_g2.clone();
            k.ifft_g1(&mut got_g1).expect("ifft_g1");
            k.ifft_g2(&mut got_g2).expect("ifft_g2");
            assert!(same_g1(&got_g1, &want_g1), "G1 disagrees at c={c}");
            assert!(same_g2(&got_g2, &want_g2), "G2 disagrees at c={c}");
        }
    }

    /// The `gid_off` split, which no block a unit test can afford to run would reach on
    /// its own: the shipped budget is 2^18 ladders and the largest block here is 2^12
    /// points, so without [`FftKernels::with_budget`] every pass fits in one command
    /// buffer and the offset is always zero.
    ///
    /// 7 and 100 are deliberately not divisors of anything: they split every pass at a
    /// different point inside its groups, so a `gid_off` applied to the wrong side of the
    /// group/position split, or a piece that silently re-runs the butterflies the previous
    /// one already did, comes out as a wrong point rather than as luck.
    #[test]
    fn a_pass_split_across_command_buffers_agrees() {
        let Some(_) = kernels() else { return };
        let device = Device::system_default().expect("device");
        // 2^10 rather than the 2^12 the size sweep uses, because a budget of 1 means one
        // command buffer per butterfly and 2^12 would be 30,000 of them.
        let n = 1 << 10;
        let base = block_g1(0x5eed_5000, n);
        let mut want = base.clone();
        CpuGroupFft.ifft_g1(&mut want).expect("cpu ifft_g1");

        for budget in [1usize, 7, 100, 1 << 9] {
            let k = FftKernels::with_device(device.clone())
                .expect("fft kernels")
                .with_budget(budget);
            let mut got = base.clone();
            k.ifft_g1(&mut got).expect("ifft_g1");
            assert!(same_g1(&got, &want), "budget {budget} disagrees");
        }
    }

    /// A block that is not a power of two, and one past the two-adic subgroup, are refused
    /// rather than transformed into something plausible. `lagrange_evaluations` checks both
    /// before it gets here, so this is the backend refusing to trust its caller.
    #[test]
    fn a_bad_block_length_is_refused() {
        let Some(k) = kernels() else { return };
        let mut a = block_g1(0x5eed_4000, 3);
        assert!(k.ifft_g1(&mut a).is_err(), "3 points is not a power of two");
    }
}
