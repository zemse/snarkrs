//! The group inverse FFT behind `ptau prepare`, the slowest command in the project.
//!
//! `shaders/fft.metal` and this file are one unit and must be changed together, the same
//! contract `layout.rs` states for the wire structs. What lives here is the shape of the
//! submission, the twiddle table, and the two permutations that a kernel deliberately does
//! not do.
//!
//! The CPU twin is `g16_ceremony::prepare::ifft` (prepare.rs:320) and the measurement that
//! decided every constant below is in
//! `bench/results/` alongside the prepare profile: 99.2% of `ptau prepare` is inside
//! `point_times_fr` (prepare.rs:149), so there is one primitive to move and nothing else
//! in the command worth touching.
//!
//! The result agrees with the CPU as a curve point, not limb for limb, because the ladder
//! here is a fixed signed window and the ladder there is wNAF, and XYZZ is projective. The
//! ceremony's byte-identity survives that for one reason: `lagrange_evaluations`
//! (prepare.rs:414) ends every path in `batch_to_affine`, and affine is canonical. So
//! `cmp` of a `--backend cpu` output against a `--backend metal` one is a total check on
//! this file and there is nothing to keep in step.

use g16_core::ProveError;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{FftField, Field, Fr};
use g16_msm::xyzz::Xyzz;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions,
};

use crate::ceremony::{cer_err, cer_read_back, dispatch_1d, env_window, window_index, CER_WINDOWS};
use crate::kernels::{CEREMONY_MSL, FFT_MSL, FR_MSL, MSM_MSL};
use crate::layout::{as_bytes, Packed, PackedFq, PackedFq2, PackedScalar};
use crate::msm::{PackedXyzzG1, PackedXyzzG2};

/// Ladder window for both groups, the same widths and the same measurement as
/// `ceremony::WINDOW_G1`: a sweep of 2^16 full-width scalars put c=4 fastest on G1 and on
/// G2, and the register-pressure argument that predicted G2 would want a narrower window
/// was wrong. `tests::window_sweep` in `ceremony.rs` is the measurement; `fft_window_sweep`
/// below re-runs it through a whole transform in case the FFT's memory traffic moves it,
/// and it does not.
const FFT_WINDOW_G1: u32 = 4;
/// See [`FFT_WINDOW_G1`].
const FFT_WINDOW_G2: u32 = 4;

/// Blocks shorter than this go back to the CPU.
///
/// Not a crossover measurement, and deliberately not tuned to one. At power 20 the GPU is
/// still ahead of the CPU at 2^10 on paper (8.7 ms of arithmetic against 36 ms), and what
/// the paper leaves out is the buffer allocation, the pipeline binding and the repack,
/// each tens of microseconds, plus a 2^9-thread dispatch not filling 38 cores. The reason
/// to stop caring at 2^12 is the distribution rather than the crossover: every block below
/// 2^12 together is 0.18% of a power-20 run and the top two blocks are 78.4% of it, so the
/// threshold's only real job is to keep the GPU path from being embarrassing on a power-8
/// test file. [`FftKernels::with_min_block`] moves it, which is how the test suite forces
/// the device path onto blocks a real ptau would route home, and
/// `G16_METAL_FFT_MIN_BLOCK` moves it from outside the process, which is how the
/// crossover is swept from the CLI without a rebuild.
const MIN_BLOCK: usize = 1 << 12;

/// Threads per threadgroup asked for, clamped by [`dispatch_1d`] to what the pipeline
/// reports. Same preference the ceremony ladders use: the kernel is one long dependent
/// chain per thread, so occupancy comes from the grid, not the group.
const THREADGROUP: usize = 64;

/// Mirrors `struct FftParams` in `shaders/fft.metal`. Passed by `setBytes`, which copies
/// at encode time, so one encoder can carry a different pass in every dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FftParams {
    n: u32,
    span: u32,
    log_span: u32,
    tw_shift: u32,
}

const _: () = {
    assert!(core::mem::size_of::<FftParams>() == 16);
    assert!(core::mem::align_of::<FftParams>() == 4);
};

/// The pipelines for one group, parallel to [`CER_WINDOWS`].
struct FftPipelines {
    mix: Vec<ComputePipelineState>,
    scale: Vec<ComputePipelineState>,
}

/// What the two kernels need to know about a group, so the submission code is written
/// once.
///
/// A second, smaller twin of `ceremony::CerGroup` rather than a use of it. The two overlap
/// only in the packing, and keeping this module's marker types here is what lets the whole
/// FFT be added without touching the ladder and apply-key code it shares a crate with.
trait FftGroup {
    type Raw: Copy;
    type PackedPoint: Packed + Default;

    /// Names this group in an error, in the spelling [`g16_msm::AccelError`] uses.
    const OP: &'static str;

    fn pack(p: &Xyzz<Self::Raw>) -> Self::PackedPoint;
    fn unpack(p: &Self::PackedPoint) -> Xyzz<Self::Raw>;
    fn pipelines(k: &FftKernels) -> &FftPipelines;
    fn window(k: &FftKernels) -> u32;
}

struct FftG1;
struct FftG2;

impl FftGroup for FftG1 {
    type Raw = RawFq;
    type PackedPoint = PackedXyzzG1;

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

    fn pipelines(k: &FftKernels) -> &FftPipelines {
        &k.g1
    }

    fn window(k: &FftKernels) -> u32 {
        k.window_g1
    }
}

impl FftGroup for FftG2 {
    type Raw = RawFq2;
    type PackedPoint = PackedXyzzG2;

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

    fn pipelines(k: &FftKernels) -> &FftPipelines {
        &k.g2
    }

    fn window(k: &FftKernels) -> u32 {
        k.window_g2
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
            let mut scale = Vec::with_capacity(CER_WINDOWS.len());
            for c in CER_WINDOWS {
                mix.push(pso(&format!("fft_mix_{g}_c{c}"))?);
                scale.push(pso(&format!("fft_scale_{g}_c{c}"))?);
            }
            Ok(FftPipelines { mix, scale })
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
        self.ifft::<FftG1>(a)
    }

    /// [`Self::ifft_g1`] over G2, which is 44% of `ptau prepare` on 20% of its scalar
    /// multiplications.
    pub fn ifft_g2(&self, a: &mut [Xyzz<RawFq2>]) -> Result<(), ProveError> {
        self.ifft::<FftG2>(a)
    }

    fn ifft<G: FftGroup>(&self, a: &mut [Xyzz<G::Raw>]) -> Result<(), ProveError> {
        let n = a.len();
        if n <= 1 {
            return Ok(());
        }
        if !n.is_power_of_two() {
            return Err(cer_err(format!(
                "group ifft over {}: {n} points is not a power of two",
                G::OP
            )));
        }
        let bits = n.trailing_zeros();
        if bits > Fr::TWO_ADICITY {
            return Err(cer_err(format!(
                "group ifft over {}: 2^{bits} is past the {}-bit two-adic subgroup",
                G::OP,
                Fr::TWO_ADICITY
            )));
        }

        let pipelines = G::pipelines(self);
        let idx = window_index(G::window(self));
        let mix = &pipelines.mix[idx];
        let scale = &pipelines.scale[idx];

        // `bit_reverse` (prepare.rs:256) folded into the upload. The permutation is an
        // involution, so writing `packed[rev(i)] = a[i]` and reading `packed[rev(i)]` back
        // are the same map; only this direction is applied, and the inverse rotation below
        // is the one that is NOT an involution.
        let mut packed = vec![G::PackedPoint::default(); n];
        for (i, p) in a.iter().enumerate() {
            packed[bit_reverse_index(i, bits)] = G::pack(p);
        }
        let work = self.buffer(&packed);
        drop(packed);

        let tw = self.buffer(&twiddle_table(bits));
        let size_inv = Fr::from(n as u64)
            .inverse()
            .expect("a power of two is a unit mod r");
        let inv = self.buffer(&[PackedScalar::from_fr(&size_inv)]);

        // Every pass and the scaling go into ONE serial compute encoder and one command
        // buffer. Metal orders consecutive dispatches on a serial encoder with an implicit
        // barrier, which is the dependency the transform needs and the only one it needs,
        // so the vector never leaves the device between passes. `Plan::encode`
        // (msm.rs:917) relies on the same property.
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for exp in 1..=bits {
            let p = FftParams {
                n: n as u32,
                span: 1 << (exp - 1),
                log_span: exp - 1,
                tw_shift: bits - exp,
            };
            enc.set_compute_pipeline_state(mix);
            enc.set_buffer(0, Some(&work), 0);
            enc.set_buffer(1, Some(&tw), 0);
            set_params(enc, 2, &p);
            dispatch_1d(enc, mix, n / 2, THREADGROUP);
        }
        let p = FftParams {
            n: n as u32,
            ..Default::default()
        };
        enc.set_compute_pipeline_state(scale);
        enc.set_buffer(0, Some(&work), 0);
        enc.set_buffer(1, Some(&inv), 0);
        set_params(enc, 2, &p);
        dispatch_1d(enc, scale, n, THREADGROUP);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "ceremony group inverse fft")?;

        // SAFETY: the command buffer completed, `n` points of this type is exactly what
        // the passes wrote, and `PackedPoint` is `Packed`, so every bit pattern is valid.
        let got: &[G::PackedPoint] = unsafe { cer_read_back(&work, n) };

        // The `a[1..].reverse()` that finishes the inverse (prepare.rs:340), folded into
        // the read back: `ifft(a)[0] = X[0]/n` and `ifft(a)[i] = X[n-i]/n`. Splitting this
        // off from the scaling gives an answer that is a rotation away from correct and
        // still looks plausible, which is why the module doc there says so twice.
        a[0] = G::unpack(&got[0]);
        for (i, out) in a.iter_mut().enumerate().skip(1) {
            *out = G::unpack(&got[n - i]);
        }
        Ok(())
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

/// `W^i` for `i < 2^(bits-1)`, with `W` the primitive `2^bits`-th root, in STANDARD form.
///
/// One table serves every pass of the block. snarkjs uses a different root per pass,
/// `roots[exp]` the primitive `2^exp`-th root (`build_fft.js:44-63`, mirrored by
/// `prepare::root_table`), and `roots[exp] == W^(2^(bits-exp))`, so pass `exp`'s twiddle
/// `roots[exp]^j` is this table at `j << (bits - exp)`. The largest index a pass reads is
/// `(2^(exp-1) - 1) << (bits - exp)`, which is under `2^(bits-1)`, so the table is exactly
/// half the block and no pass runs off it.
///
/// The de-Montgomery step is here and not in a kernel. `g1m_timesFr` is
/// `frm_fromMontgomery` then a variable-base multiplication (`build_bn128.js:60-76`), and a
/// window digit of a Montgomery representative is a digit of `a*R mod r`, which is a
/// different number: getting it backwards produces points that are wrong by a factor of R
/// and a file that verifies against nothing.
fn twiddle_table(bits: u32) -> Vec<PackedScalar> {
    let half = 1usize << (bits - 1);
    let mut root = Fr::TWO_ADIC_ROOT_OF_UNITY;
    for _ in bits..Fr::TWO_ADICITY {
        root.square_in_place();
    }
    let mut w = Fr::ONE;
    let mut out = Vec::with_capacity(half);
    for _ in 0..half {
        out.push(PackedScalar::from_fr(&w));
        w *= root;
    }
    out
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
    std::env::var("G16_METAL_FFT_MIN_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(MIN_BLOCK)
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
