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
//! Co-dispatch note: `ifft_many` keeps the Metal lockstep rounds, but launches within a
//! round still serialize on the one stream, so a round of small blocks does not fill the
//! device the way Metal's `Concurrent` encoder does. Fixing that needs one fat launch
//! with a per-block parameter array, or a stream per block; neither is taken for the
//! first port, whose bar is byte identity, not underfill.
//!
//! The result agrees with the CPU as a curve point, not limb for limb, and the
//! ceremony's byte identity survives because `lagrange_evaluations` (prepare.rs:414)
//! ends every path in `batch_to_affine`, and affine is canonical. So `cmp` of a
//! `--backend cpu` output against a `--backend cuda` one is a total check on this file.

use std::sync::Arc;

use ark_ec::scalar_mul::glv::GLVConfig;
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, DeviceRepr, PushKernelArg};
use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{FftField, Field, Fr};
use g16_gpu_layout::glv::twiddle_table;
use g16_gpu_layout::{Packed, PackedFq, PackedFq2};
use g16_msm::xyzz::Xyzz;
use rayon::prelude::*;

use crate::msm::{bad, download, drv, upload_words, Kernel, PackedXyzzG1, PackedXyzzG2};
use crate::{as_words, from_words, kernels, Cuda};

/// Ladder window for both groups, and the only width `kernels/fft.cu` instantiates.
///
/// 5 is the Metal sweep's answer for both groups over a whole `ppot_0080_16.ptau`
/// prepare (`g16-metal/src/fft.rs`, `FFT_WINDOW_G1`), carried over rather than re-swept:
/// there is no NVIDIA sweep yet, and compiling the other widths to enable one would cost
/// every fresh machine NVRTC time for a knob the shipped configuration does not use
/// (`fft.cu`'s banner has the numbers). To sweep on NVIDIA, add `FFT_KERNELS` lines to
/// `fft.cu` and widths here, and take the compile hit once.
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
/// caveat: not swept on a real card.
const FFT_BLOCK: u32 = 128;

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
/// ping-pong pair, its own `1/n`-scaled twiddle table, and where in the caller's slice
/// of blocks it came from.
struct Block {
    idx: usize,
    n: usize,
    bits: u32,
    src: CudaSlice<u32>,
    dst: CudaSlice<u32>,
    stw: CudaSlice<u32>,
}

/// The two kernels for one group at the shipped width.
struct FftPair {
    mix: Kernel,
    mix_scale: Kernel,
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
        &k.g1
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
        &k.g2
    }
}

/// The four FFT kernels, compiled once.
///
/// Compiling the unit is the expensive call (NVRTC plus the driver's ptxas on a miss of
/// both caches; see `context.rs` for the measured cliff) and it belongs once at the top
/// of a command, exactly as `CudaMsm::compile` does for the prover.
pub struct FftKernels {
    stream: Arc<CudaStream>,
    /// Kept alive explicitly. `CudaFunction` holds an `Arc<CudaModule>` internally, so
    /// this field documents the ownership rather than being what keeps the module loaded.
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    g1: FftPair,
    g2: FftPair,
    min_block: usize,
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
    pub fn from_module(cuda: &Cuda, module: Arc<CudaModule>) -> Result<Self, ProveError> {
        // The names are `FFT_WINDOW` spelled out, because `Kernel::load` wants 'static
        // names; a width added to `FFT_WINDOWS` needs its pair added here, and the
        // grep test below fails on a width whose kernels do not exist in the source.
        let g1 = FftPair {
            mix: Kernel::load(&module, "fft_mix_g1_c5")?,
            mix_scale: Kernel::load(&module, "fft_mix_scale_g1_c5")?,
        };
        let g2 = FftPair {
            mix: Kernel::load(&module, "fft_mix_g2_c5")?,
            mix_scale: Kernel::load(&module, "fft_mix_scale_g2_c5")?,
        };
        Ok(Self {
            stream: cuda.stream().clone(),
            module,
            g1,
            g2,
            min_block: env_min_block(),
        })
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
            let src = upload_words(&self.stream, as_words(&packed))?;
            let dst = self
                .stream
                .alloc_zeros::<u32>(n * point_words)
                .map_err(|e| drv("allocate fft scratch", e))?;

            let size_inv = Fr::from(n as u64)
                .inverse()
                .expect("a power of two is a unit mod r");
            // The `1/n` scaling rides the block's last mix pass (`fft_mix_scale_impl`),
            // which reads a second table with `s = 1/n` folded into every entry,
            // `stw[0]` doubling as the plain `[s]` the `lo` side needs. `s` differs per
            // block, so unlike `tw` this table cannot be shared across a round.
            let stw = upload_words(
                &self.stream,
                as_words(&twiddle_table::<G::Cfg>(bits, size_inv)),
            )?;
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

        // One plain table serves every mix pass of every block: pass `exp` wants
        // `roots[exp]^j`, and `tw_shift = max_bits - exp` reads the deepest block's
        // table at the right stride whatever the block's own depth is.
        let tw = upload_words(
            &self.stream,
            as_words(&twiddle_table::<G::Cfg>(max_bits, Fr::ONE)),
        )?;

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
                let cfg = kernel.cfg_1d(threads, FFT_BLOCK);
                let mut lb = self.stream.launch_builder(&kernel.f);
                lb.arg(&b.src).arg(&b.dst).arg(third).arg(&params);
                // SAFETY: four parameters bound in order and with matching types; `src`
                // and `dst` each hold `n` points of this group, every in-bounds index
                // the kernel touches is under `n`, the twiddle table holds at least
                // `(n/2) * GLV_WORDS` words by construction, and the surplus threads of
                // the last CUDA block exit on the kernel's own bounds guard.
                unsafe { lb.launch(cfg) }.map_err(|e| drv("launch fft mix", e))?;
            }
            // Launches on the one stream execute in issue order, so pass `exp + 1` reads
            // what pass `exp` wrote with no event and no host wait; the swap is pure
            // host bookkeeping.
            for b in live.iter_mut().filter(|b| b.bits >= exp) {
                core::mem::swap(&mut b.src, &mut b.dst);
            }
        }

        for b in &live {
            // The one wait per block: `download` synchronizes the stream before the
            // `Vec` is read, which also orders it after every launch above.
            let words = download(&self.stream, &b.src)?;
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
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use g16_gpu_layout::PackedGlv;

    /// Widths with a kernel pair per group in `kernels/fft.cu`, parallel to the
    /// `FFT_KERNELS` lines there.
    const FFT_WINDOWS: [u32; 1] = [FFT_WINDOW];

    /// Every width in [`FFT_WINDOWS`] has a kernel pair in the source, and the shipped
    /// width is among them. Construction would catch a missing kernel, but only on a
    /// machine with a device; this runs anywhere.
    #[test]
    fn every_compiled_window_has_a_kernel_pair() {
        for c in FFT_WINDOWS {
            for g in ["g1", "g2"] {
                assert!(
                    kernels::FFT_CU.contains(&format!("FFT_KERNELS({g}_c{c},")),
                    "kernels/fft.cu has no FFT_KERNELS line for {g} c={c}, so \
                     fft_mix_{g}_c{c} does not exist and load_function will fail"
                );
            }
        }
        assert!(FFT_WINDOWS.contains(&FFT_WINDOW));
    }

    /// Without `extern "C"` NVRTC mangles the entry names and `load_function` fails at
    /// run time, on a GPU box, minutes into a ptxas run. The entry points come out of
    /// the `FFT_KERNELS` macro, so the check is on the macro body.
    #[test]
    fn the_entry_points_are_declared_extern_c() {
        for want in [
            "extern \"C\" __global__ void fft_mix_##SUF(",
            "extern \"C\" __global__ void fft_mix_scale_##SUF(",
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
