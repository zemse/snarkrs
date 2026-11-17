//! The CUDA side of the ceremony seams: [`MsmBackend`] behind `setup` and `zkey verify`'s
//! recompute, and [`GroupFft`] behind `ptau prepare`.
//!
//! The Metal twin (`crates/metal/src/ceremony.rs`) implements all three seams; this file
//! implements the two whose kernels exist. The apply-key behind the four contribute and
//! beacon commands still has no CUDA kernel, and the CLI keeps saying so per command
//! rather than falling back.
//!
//! Separate from [`crate::backend`], which implements `snarkrs_groth16::Backend`, the whole
//! prover. A ceremony command wants one primitive at a time and holds no proving key, so
//! it opens its own context here instead of borrowing the prover's.

use std::ops::{AddAssign, Range};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ark_ec::CurveGroup;
use cudarc::driver::CudaSlice;
use snarkrs_field::raw::{RawFq, RawFq2};
use snarkrs_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use snarkrs_gpu_layout::{Packed, PackedG1Affine, PackedG2Affine};
use snarkrs_groth16::ProveError;
use snarkrs_msm::xyzz::Xyzz;
use snarkrs_msm::{AccelError, CpuMsm, GroupFft, MsmBackend};

use crate::backend::{env_ordinal, env_vram_limit};
use crate::context::{bad, time_enabled, upload_words, BACKEND};
use crate::fft::{backoff, FftKernels, RETRIES};
use crate::msm::{CudaMsm, G1Bases, G2Bases, Job, JobG1, JobG2, MIN_RUN};
use crate::{as_words, check, Cuda, CudaError};

/// Points below which a call goes to [`CpuMsm`] rather than the device.
///
/// Not measured on a T4. Every device call pays a fixed cost the CPU does not: the bases
/// and scalars cross PCIe, the memory plan synchronizes the stream and queries the pool,
/// and the digit and point stages are a dozen or more launches with a readback behind
/// them, all before any arithmetic. `setup` sends a slot here only once it holds
/// `MULTIEXP_MIN_TERMS` (256) terms, and most such slots are a few hundred, where a
/// Pippenger on the host is a few milliseconds. 2^12 keeps those on the CPU and sends the
/// large slots, and the `--check random` combination of any circuit past 2^12 constraints
/// (one MSM over its Lagrange bases), to the device. `G16_CUDA_CER_MSM_MIN` overrides it, for a sweep and for a run that must put
/// every call on the device.
const MSM_MIN_POINTS: usize = 1 << 12;

/// The longest run one device call takes. `setup`'s combination hands over 2^21 bases a
/// call (`COMBINE_CHUNK`), which in G2 is 256 MiB of packed bases before any scratch, so
/// this keeps a call there to one run. A run the memory plan refuses is halved, down to
/// `MIN_RUN`, and the runs' sums are added on the host.
const MSM_MAX_RUN: usize = 1 << 21;

/// [`CudaMsm`] behind the ceremony's MSM trait.
///
/// A wrapper rather than an `impl` on `CudaMsm`, for the reason `MetalMsmBackend` gives:
/// `CudaMsm::msm` takes device handles and the trait takes host slices. Every call uploads
/// its own bases, as on Metal, because `setup` never repeats a base vector.
///
/// Held to the check the Metal MSM holds its batches to: a sum that is not a point of the
/// curve is a device fault, and the run that produced it is re-run from the unchanged host
/// slices, `RETRIES` attempts in all, before the call fails. An MSM has no per-point
/// readback to spot check against the host, so a wrong sum that is still on the curve
/// passes here. The callers fail closed on it: `zkey verify` compares what this computes
/// with the CPU's own reading of the key, and a key `setup` wrote from a wrong sum fails
/// that verify.
pub struct CudaMsmBackend {
    msm: CudaMsm,
    cpu: CpuMsm,
    min_points: usize,
    max_run: usize,
    /// One call on the device at a time. `setup` calls from every rayon worker, and the
    /// memory plan reads free memory once per batch: an upload racing it would allocate
    /// against bytes the plan already counted. Nothing under it may use rayon; see
    /// [`Self::run_g1`].
    one: Mutex<()>,
    runs: AtomicU64,
    tally: Tally,
}

/// Where a command's calls went, printed when the backend drops under `G16_CUDA_TIME=1`:
/// the evidence that a run reached the device at all, and the numbers a sweep of
/// `G16_CUDA_CER_MSM_MIN` reads.
#[derive(Default)]
struct Tally {
    device_calls: AtomicU64,
    device_points: AtomicU64,
    device_us: AtomicU64,
    host_calls: AtomicU64,
    host_points: AtomicU64,
}

impl Drop for Tally {
    fn drop(&mut self) {
        if !time_enabled() {
            return;
        }
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        eprintln!(
            "cuda ceremony msm: {} calls of {} points on the device in {:.3} s, \
             {} calls of {} points on the host",
            get(&self.device_calls),
            get(&self.device_points),
            get(&self.device_us) as f64 / 1e6,
            get(&self.host_calls),
            get(&self.host_points),
        );
    }
}

impl CudaMsmBackend {
    /// Opens device 0, or the ordinal in `G16_CUDA_DEVICE`, and compiles the MSM unit. Same
    /// selection rule and the same refusal to fall back as [`CudaGroupFft::new`]. The unit
    /// is the prover's, so a box that has proved on CUDA loads it from the caches, and a
    /// cold one pays NVRTC and the driver JIT here (see [`CudaMsm`]).
    pub fn new() -> Result<Self, ProveError> {
        let ordinal = env_ordinal()?;
        let cuda = Cuda::new(ordinal)
            .map_err(|e: CudaError| bad(format!("no usable CUDA device {ordinal}: {e}")))?;
        Self::with_cuda(&cuda)
    }

    /// The MSM on an already-open context. `G16_CUDA_VRAM_LIMIT` caps its memory plan as
    /// it caps the prover's.
    pub fn with_cuda(cuda: &Cuda) -> Result<Self, ProveError> {
        let mut msm = CudaMsm::without_key(cuda)?;
        if let Some(limit) = env_vram_limit()? {
            msm = msm.with_vram_limit(limit);
        }
        let min_points = match std::env::var("G16_CUDA_CER_MSM_MIN") {
            Ok(v) => v.trim().parse::<usize>().map_err(|e| {
                bad(format!(
                    "G16_CUDA_CER_MSM_MIN={v:?} is not a point count: {e}"
                ))
            })?,
            Err(_) => MSM_MIN_POINTS,
        };
        Ok(Self {
            msm,
            cpu: CpuMsm::new(),
            min_points,
            max_run: MSM_MAX_RUN,
            one: Mutex::new(()),
            runs: AtomicU64::new(0),
            tally: Tally::default(),
        })
    }

    pub fn msm(&self) -> &CudaMsm {
        &self.msm
    }

    /// The crossover this instance uses. See `MSM_MIN_POINTS`.
    pub fn min_points(&self) -> usize {
        self.min_points
    }

    /// The crossover, for the tests that need every call on the device. Zero and one both
    /// mean "everything".
    pub fn with_min_points(mut self, n: usize) -> Self {
        self.min_points = n;
        self
    }

    /// The longest run of one device call, at least `MIN_RUN`. Only the tests move it.
    pub fn with_max_run(mut self, n: usize) -> Self {
        self.max_run = n.max(MIN_RUN);
        self
    }

    /// [`CudaMsm::with_vram_limit`], for a test that wants a run halved or refused.
    pub fn with_vram_limit(mut self, bytes: u64) -> Self {
        self.msm = self.msm.with_vram_limit(bytes);
        self
    }

    /// Device runs the last call took: zero when it went to the CPU.
    pub fn last_runs(&self) -> u64 {
        self.runs.load(Ordering::Relaxed)
    }

    /// `n` points through the device in runs of at most `max_run`, summed on the host. A
    /// run the memory plan refuses is halved and tried again; any other failure is the
    /// call's.
    fn runs<P: Default + AddAssign>(
        &self,
        n: usize,
        op: &'static str,
        run: impl Fn(Range<usize>) -> Result<P, ProveError>,
        on_curve: impl Fn(&P) -> bool,
    ) -> Result<P, AccelError> {
        let _one = self.one.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = Instant::now();
        let mut sum = P::default();
        let (mut at, mut len, mut runs) = (0, self.max_run, 0u64);
        while at < n {
            let r = at..(at + len).min(n);
            match checked(op, || run(r.clone()), &on_curve) {
                Ok(p) => {
                    sum += p;
                    at = r.end;
                    runs += 1;
                }
                Err(ProveError::DeviceMemory { .. }) if r.len() > MIN_RUN => {
                    len = (r.len() / 2).max(MIN_RUN);
                }
                Err(e) => return Err(AccelError::device(BACKEND, op, e.to_string())),
            }
        }
        self.runs.store(runs, Ordering::Relaxed);
        let t = &self.tally;
        t.device_calls.fetch_add(1, Ordering::Relaxed);
        t.device_points.fetch_add(n as u64, Ordering::Relaxed);
        t.device_us
            .fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        Ok(sum)
    }

    /// A call under the crossover, or an empty one, on the host.
    fn on_host<P>(&self, n: usize, msm: impl FnOnce(&CpuMsm) -> P) -> P {
        self.runs.store(0, Ordering::Relaxed);
        self.tally.host_calls.fetch_add(1, Ordering::Relaxed);
        self.tally
            .host_points
            .fetch_add(n as u64, Ordering::Relaxed);
        msm(&self.cpu)
    }

    /// One device MSM over bases [`pack`] already packed. Nothing here or below it touches
    /// rayon, which is what makes holding `one` across it safe: `setup` calls from rayon
    /// workers, and a worker that held the lock and waited on the pool could be handed a
    /// task that blocks on the same lock.
    fn run_g1(&self, bases: &[PackedG1Affine], scalars: &[Fr]) -> Result<G1Projective, ProveError> {
        let bases = G1Bases::from_device(self.upload(bases)?, bases.len())?;
        let scalars = self.msm.upload_scalars(scalars)?;
        let n = bases.len();
        self.msm
            .msm(Job::G1(JobG1 {
                bases: &bases,
                base_off: 0,
                scalars: scalars.as_scalars(),
                scalar_off: 0,
                n,
            }))?
            .g1()
    }

    fn run_g2(&self, bases: &[PackedG2Affine], scalars: &[Fr]) -> Result<G2Projective, ProveError> {
        let bases = G2Bases::from_device(self.upload(bases)?, bases.len())?;
        let scalars = self.msm.upload_scalars(scalars)?;
        let n = bases.len();
        self.msm
            .msm(Job::G2(JobG2 {
                bases: &bases,
                base_off: 0,
                scalars: scalars.as_scalars(),
                scalar_off: 0,
                n,
            }))?
            .g2()
    }

    /// One pageable copy. `CudaMsm`'s own base upload stages through a pair of page-locked
    /// buffers it allocates per call, which suits a key prepared once and is a fixed cost
    /// too many for a ceremony that calls this per slot.
    fn upload<P: Packed>(&self, packed: &[P]) -> Result<CudaSlice<u32>, ProveError> {
        upload_words(self.msm.stream(), as_words(packed))
    }
}

/// `bases` in the device's layout, packed on the pool before the device lock is taken.
fn pack<A: Sync, P: Packed + Default + Send>(
    bases: &[A],
    pack: impl Fn(&[A], &mut [P]) + Sync,
) -> Vec<P> {
    use rayon::prelude::*;
    const PIECE: usize = 1 << 12;
    let mut packed = vec![P::default(); bases.len()];
    packed
        .par_chunks_mut(PIECE)
        .zip(bases.par_chunks(PIECE))
        .for_each(|(d, s)| pack(s, d));
    packed
}

/// One run, re-run while its sum is off the curve, `RETRIES` attempts in all. A driver
/// failure and a memory refusal come straight out, as in the group FFT.
fn checked<P>(
    op: &str,
    run: impl Fn() -> Result<P, ProveError>,
    on_curve: impl Fn(&P) -> bool,
) -> Result<P, ProveError> {
    let mut attempt = 0;
    loop {
        let p = run()?;
        if on_curve(&p) {
            return Ok(p);
        }
        attempt += 1;
        let e = check::wrong(
            &format!("ceremony {op}"),
            "the sum is not a point on the curve",
        );
        if attempt >= RETRIES {
            return Err(e);
        }
        eprintln!("cuda: attempt {attempt} of {RETRIES} failed: {e}; re-running the MSM");
        std::thread::sleep(backoff(attempt));
    }
}

/// Ceremony callers use the fallible methods. The legacy methods panic on device failure
/// rather than return a silently wrong point, as on Metal.
impl MsmBackend for CudaMsmBackend {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        self.try_msm_g1(bases, scalars)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        self.try_msm_g2(bases, scalars)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn try_msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> Result<G1Projective, AccelError> {
        if bases.len() != scalars.len() {
            return Err(AccelError::Shape {
                op: "msm_g1",
                reason: format!("{} bases but {} scalars", bases.len(), scalars.len()),
            });
        }
        if bases.is_empty() || bases.len() < self.min_points {
            return Ok(self.on_host(bases.len(), |c| c.msm_g1(bases, scalars)));
        }
        let bases = pack(bases, PackedG1Affine::pack_into);
        self.runs(
            bases.len(),
            "msm_g1",
            |r| self.run_g1(&bases[r.clone()], &scalars[r]),
            |p| p.into_affine().is_on_curve(),
        )
    }

    fn try_msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> Result<G2Projective, AccelError> {
        if bases.len() != scalars.len() {
            return Err(AccelError::Shape {
                op: "msm_g2",
                reason: format!("{} bases but {} scalars", bases.len(), scalars.len()),
            });
        }
        if bases.is_empty() || bases.len() < self.min_points {
            return Ok(self.on_host(bases.len(), |c| c.msm_g2(bases, scalars)));
        }
        let bases = pack(bases, PackedG2Affine::pack_into);
        self.runs(
            bases.len(),
            "msm_g2",
            |r| self.run_g2(&bases[r.clone()], &scalars[r]),
            |p| p.into_affine().is_on_curve(),
        )
    }
}

/// The group inverse FFT behind `ptau prepare`, the slowest command in the project.
///
/// A wrapper over [`FftKernels`], which owns the kernels, the twiddle tables and the
/// submission. What the trait adds on top is the error mapping and the crossover:
/// `min_block` is the backend's own number, so `prepare::ifft_block` (prepare.rs:352)
/// carries no backend-shaped branch and the threshold can move without touching the
/// ceremony crate.
pub struct CudaGroupFft {
    fft: FftKernels,
}

impl CudaGroupFft {
    /// Opens device 0, or the ordinal in `G16_CUDA_DEVICE`, and compiles the FFT unit.
    /// Same selection rule as `CudaBackend::new`, and the same refusal to fall back:
    /// a machine with no usable device is an error, not a CPU run.
    pub fn new() -> Result<Self, ProveError> {
        let ordinal = match std::env::var("G16_CUDA_DEVICE") {
            Ok(v) => v.trim().parse::<usize>().map_err(|e| {
                bad(format!(
                    "G16_CUDA_DEVICE={v:?} is not a device ordinal: {e}"
                ))
            })?,
            Err(_) => 0,
        };
        Self::with_ordinal(ordinal)
    }

    pub fn with_ordinal(ordinal: usize) -> Result<Self, ProveError> {
        let cuda = Cuda::new(ordinal)
            .map_err(|e: CudaError| bad(format!("no usable CUDA device {ordinal}: {e}")))?;
        Self::with_cuda(&cuda)
    }

    /// The kernels on an already-open context, for a caller that holds one.
    pub fn with_cuda(cuda: &Cuda) -> Result<Self, ProveError> {
        Ok(Self {
            fft: FftKernels::new(cuda)?,
        })
    }

    /// The kernel layer, for a measurement that wants to drive the transform without
    /// going through `ptau prepare`.
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

impl GroupFft for CudaGroupFft {
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

    fn ifft_g1_many(&self, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), AccelError> {
        self.fft
            .ifft_g1_many(blocks)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G1", e.to_string()))
    }

    fn ifft_g2_many(&self, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), AccelError> {
        self.fft
            .ifft_g2_many(blocks)
            .map_err(|e| AccelError::device(BACKEND, "group ifft over G2", e.to_string()))
    }
}

/// `prepare` hands the transform to worker threads and `setup` calls the MSM from every
/// rayon worker; checked rather than assumed, the same way `msm.rs` checks `CudaMsm`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaGroupFft>();
    assert_send_sync::<CudaMsmBackend>();
};

#[cfg(test)]
mod tests {
    mod resources {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-support/resources.rs"
        ));
    }

    use super::*;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::UniformRand;
    use snarkrs_field::Zero;

    /// The adapter on device 0 with every call on the device, or a loud skip.
    fn backend() -> Option<CudaMsmBackend> {
        match Cuda::new(0) {
            Ok(cuda) => {
                eprintln!("cuda device: {}", cuda.device_name());
                Some(
                    CudaMsmBackend::with_cuda(&cuda)
                        .expect("the MSM unit")
                        .with_min_points(0),
                )
            }
            Err(e) => {
                resources::skip(format_args!(
                    "cuda ceremony MSM test, no usable device: {e}"
                ));
                None
            }
        }
    }

    fn scalars(seed: u64, n: usize) -> Vec<Fr> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| Fr::rand(&mut rng)).collect()
    }

    /// `n` distinct bases from one random start, one addition apart, so a vector past a
    /// million points costs a second rather than a million ladders.
    fn bases_g1(seed: u64, n: usize) -> Vec<G1Affine> {
        let mut rng = StdRng::seed_from_u64(seed);
        let step = G1Projective::rand(&mut rng);
        let mut cur = G1Projective::rand(&mut rng);
        let pts: Vec<_> = (0..n)
            .map(|_| {
                cur += step;
                cur
            })
            .collect();
        G1Projective::normalize_batch(&pts)
    }

    fn bases_g2(seed: u64, n: usize) -> Vec<G2Affine> {
        let mut rng = StdRng::seed_from_u64(seed);
        let step = G2Projective::rand(&mut rng);
        let mut cur = G2Projective::rand(&mut rng);
        let pts: Vec<_> = (0..n)
            .map(|_| {
                cur += step;
                cur
            })
            .collect();
        G2Projective::normalize_batch(&pts)
    }

    fn agree_g1(b: &CudaMsmBackend, bases: &[G1Affine], scalars: &[Fr], what: &str) {
        let got = b
            .try_msm_g1(bases, scalars)
            .unwrap_or_else(|e| panic!("G1 {what}: {e}"));
        assert_eq!(got, CpuMsm::new().msm_g1(bases, scalars), "G1 {what}");
    }

    fn agree_g2(b: &CudaMsmBackend, bases: &[G2Affine], scalars: &[Fr], what: &str) {
        let got = b
            .try_msm_g2(bases, scalars)
            .unwrap_or_else(|e| panic!("G2 {what}: {e}"));
        assert_eq!(got, CpuMsm::new().msm_g2(bases, scalars), "G2 {what}");
    }

    /// Both groups against `CpuMsm` on random bases and full-width scalars, from one point
    /// to past the slice and window boundaries the kernels have.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn msm_matches_the_cpu_on_random_inputs() {
        let Some(b) = backend() else { return };
        for (i, n) in [1usize, 2, 255, 256, 1024, 4097, 70_001]
            .into_iter()
            .enumerate()
        {
            let s = scalars(0xce00 + i as u64, n);
            agree_g1(&b, &bases_g1(0xc100 + i as u64, n), &s, &format!("n = {n}"));
            assert_eq!(b.last_runs(), 1, "G1 n = {n} took one device run");
            agree_g2(&b, &bases_g2(0xc200 + i as u64, n), &s, &format!("n = {n}"));
            assert_eq!(b.last_runs(), 1, "G2 n = {n} took one device run");
        }
    }

    /// An empty input is the identity, a length mismatch is a shape error in either
    /// direction, and the shipped crossover keeps a short call on the host.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn empty_input_mismatch_and_crossover() {
        let Some(b) = backend() else { return };
        assert_eq!(b.try_msm_g1(&[], &[]).unwrap(), G1Projective::default());
        assert_eq!(b.last_runs(), 0);
        assert_eq!(b.try_msm_g2(&[], &[]).unwrap(), G2Projective::default());
        assert_eq!(b.last_runs(), 0);
        let g1 = bases_g1(1, 3);
        let g2 = bases_g2(2, 3);
        let s = scalars(3, 3);
        for (nb, ns) in [(0, 1), (1, 0), (2, 3), (3, 2)] {
            assert!(matches!(
                b.try_msm_g1(&g1[..nb], &s[..ns]),
                Err(AccelError::Shape { op: "msm_g1", .. })
            ));
            assert!(matches!(
                b.try_msm_g2(&g2[..nb], &s[..ns]),
                Err(AccelError::Shape { op: "msm_g2", .. })
            ));
        }
        let b = b.with_min_points(MSM_MIN_POINTS);
        agree_g1(&b, &g1, &s, "below the crossover");
        assert_eq!(
            b.last_runs(),
            0,
            "a call below the crossover reached the device"
        );
        agree_g2(&b, &g2, &s, "below the crossover");
        assert_eq!(
            b.last_runs(),
            0,
            "a call below the crossover reached the device"
        );
    }

    /// Bases at infinity contribute nothing whatever their scalar: one in five of them, and
    /// then every one, which must come back as the identity rather than garbage.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn infinity_bases_contribute_nothing() {
        let Some(b) = backend() else { return };
        let n = 2049;
        let s = scalars(0x1f, n);
        let mut g1 = bases_g1(0x11, n);
        let mut g2 = bases_g2(0x12, n);
        for i in (0..n).step_by(5) {
            g1[i] = G1Affine::identity();
            g2[i] = G2Affine::identity();
        }
        agree_g1(&b, &g1, &s, "every fifth base at infinity");
        agree_g2(&b, &g2, &s, "every fifth base at infinity");
        let g1 = vec![G1Affine::identity(); n];
        let g2 = vec![G2Affine::identity(); n];
        assert!(b.try_msm_g1(&g1, &s).unwrap().is_zero());
        assert!(b.try_msm_g2(&g2, &s).unwrap().is_zero());
    }

    /// Past one `setup` chunk (2^20 points) in one run, and the same vector cut into runs
    /// whose sums the host adds, both against the CPU. G2 is cut at a smaller size: a 2^20
    /// G2 reference on the gate box's four cores is minutes for no extra coverage.
    #[test]
    #[ignore = "a 2^20 MSM twice and its CPU reference; about a minute on a T4 box"]
    fn a_call_past_one_chunk_matches_the_cpu() {
        let Some(b) = backend() else { return };
        let n = (1 << 20) + 3;
        let (g1, s) = (bases_g1(0x21, n), scalars(0x22, n));
        let want = CpuMsm::new().msm_g1(&g1, &s);
        assert_eq!(
            b.try_msm_g1(&g1, &s).unwrap(),
            want,
            "G1 2^20 + 3 in one run"
        );
        assert_eq!(b.last_runs(), 1);
        let b = b.with_max_run(1 << 18);
        assert_eq!(b.try_msm_g1(&g1, &s).unwrap(), want, "G1 2^20 + 3 in runs");
        assert_eq!(b.last_runs(), 5);
        drop(g1);
        let n = (1 << 18) + 5;
        let b = b.with_max_run(1 << 16);
        agree_g2(&b, &bases_g2(0x23, n), &s[..n], "2^18 + 5 in runs");
        assert_eq!(b.last_runs(), 5);
    }

    /// A memory limit that holds a run of 2^11 points and not 2^13 halves a 2^13 call
    /// until it fits and still sums to the CPU's answer; a limit that holds nothing is a
    /// clean device error, not a panic and not a CPU answer.
    #[test]
    #[ignore = "needs an NVIDIA device"]
    fn a_vram_limit_halves_the_runs_or_refuses() {
        let Some(b) = backend() else { return };
        let n = 1 << 13;
        let (g1, s) = (bases_g1(0x31, n), scalars(0x32, n));
        let need = |len: usize| {
            let bases = b.msm().upload_g1_bases(&g1[..len]).unwrap();
            let sc = b.msm().upload_scalars(&s[..len]).unwrap();
            let job = Job::G1(JobG1 {
                bases: &bases,
                base_off: 0,
                scalars: sc.as_scalars(),
                scalar_off: 0,
                n: len,
            });
            b.msm().batch_bytes(&[job]).unwrap()
        };
        let (small, large) = (need(1 << 11), need(n));
        assert!(
            small < large,
            "a 2^13 batch plans {large} bytes, 2^11 {small}"
        );
        let limited = b.with_vram_limit(small);
        agree_g1(&limited, &g1, &s, "under a limit that holds 2^11");
        assert!(limited.last_runs() >= 4, "{} runs", limited.last_runs());

        let refused = limited.with_vram_limit(1);
        let g2 = bases_g2(0x33, MIN_RUN);
        for e in [
            refused.try_msm_g1(&g1, &s).unwrap_err(),
            refused.try_msm_g2(&g2, &s[..MIN_RUN]).unwrap_err(),
        ] {
            assert!(
                matches!(
                    e,
                    AccelError::Device {
                        backend: "cuda",
                        ..
                    }
                ),
                "{e}"
            );
            assert!(e.to_string().contains("device memory"), "{e}");
        }
    }
}
