//! [`GnarkProver`]: one gnark Groth16 key resident on one CUDA device, for a process that
//! proves gnark witnesses while something else holds most of the same card and most of the
//! host's memory. SP1's GPU server is the case it is shaped for: about 18 GB of a 46 GB
//! L40S and 15 GB of a 30 GB host are its, permanently.
//!
//! # What construction does, and why all of it happens there
//!
//! * **The key never exists decoded on the host.** [`GnarkDump`] maps the `WriteDump` file
//!   and hands out one 32 MiB run of points at a time, checked and re-expanded, straight
//!   into the pinned staging buffer that crosses to the device; the pages it read are
//!   dropped as it goes. The dumped point layout, Montgomery little-endian `x` then `y` and
//!   the identity as zero bytes, is [`snarkrs_gpu_layout`]'s packed layout byte for byte
//!   (`tests::a_dumped_point_is_its_packed_layout` holds that), so nothing is repacked. A
//!   6 GB key goes up through 64 MiB of host memory.
//! * **Every device byte a proof needs is reserved.** The bases, the transform tables, one
//!   proof's stage scratch and the MSM arena are allocated here and kept; a proof allocates
//!   nothing on the device, so it cannot run out of memory halfway. The total is
//!   [`required_device_bytes`], computable from the dump's header before anything is
//!   allocated, and construction refuses up front when the device has less free.
//! * **Its own stream.** The device's primary context is the one the CUDA runtime in the
//!   same process uses, and its default stream serializes against every blocking stream in
//!   it; this prover issues everything on a non-blocking stream of its own (see
//!   [`Cuda::new_private`]), and puts back whatever context the calling thread had current.
//!
//! # What a proof does
//!
//! [`GnarkProver::prove`] takes the solver's three vectors borrowed ([`GnarkWitness`]),
//! stages them through the same 64 MiB pair, and runs stages 1 to 9 on the device
//! (`h_and_msms_from_rows`), then blinds, assembles and checks the proof against gnark's
//! verifying key on the host. Stages 1 to 9 are serialized: the reservation holds one
//! proof's scratch.
//!
//! A `GnarkProver` is a [`PreparedCircuit`], so a caller that proves through
//! `Box<dyn PreparedCircuit>` and `snarkrs_gnark::prove` takes [`prepare`] where it took
//! `CpuBackend::new().prepare(pk)`; only the key's header and five points stay on the host.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use cudarc::driver::{sys, CudaSlice, CudaStream};
use snarkrs_field::Fr;
use snarkrs_formats::ProvingKey;
use snarkrs_gnark::Query;
use snarkrs_gnark::{pk::DumpHeader, GnarkDump, GnarkError, GnarkVerifyingKey, GnarkWitness};
use snarkrs_groth16::{HPoly, MsmOutputs, PreparedCircuit, Proof, ProveError, StageTimings};

use crate::backend::{CudaCircuit, PrepareCost, MSM_CHUNK};
use crate::msm::{resident_scratch, CudaMsm, G1Bases, G2Bases};
use crate::stages::{CudaStages, Staging};
use crate::Cuda;

fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "cuda",
        reason: reason.into(),
    }
}

fn gnark(e: GnarkError) -> ProveError {
    bad(format!("gnark key: {e}"))
}

/// Allocation granularity assumed for every device buffer when summing a reservation:
/// the driver maps large allocations in 2 MiB pages, so each buffer is counted rounded up.
const PAGE: u64 = 2 << 20;

/// Device memory counted on top of the buffers: the two kernel modules, the local memory
/// their spilling kernels get, and the driver's own per-stream state. Not measured; the
/// MSM module's PTX is 31 MB, and the rest is margin.
pub const DEVICE_MARGIN: u64 = 512 << 20;

/// How a [`GnarkProver`] is built.
#[derive(Clone, Copy, Debug)]
pub struct GnarkProverConfig {
    /// CUDA device ordinal.
    pub device: usize,
    /// Points an MSM run covers at most, [`MSM_CHUNK`] by default. The MSM arena grows with
    /// it, by about 0.8 GB per 2^20 points; a larger run is fewer, larger batches.
    pub msm_chunk: usize,
    /// Run stages 1 to 9 once at construction over a zero witness, so a kernel's first
    /// launch, and what the driver allocates for it then, happens before the first job.
    /// On by default; it costs about one proof's time and an `n_vars` host vector.
    pub warm_up: bool,
}

impl Default for GnarkProverConfig {
    fn default() -> Self {
        Self {
            device: 0,
            msm_chunk: MSM_CHUNK,
            warm_up: true,
        }
    }
}

/// Every device buffer a [`GnarkProver`] for `header` reserves, in bytes, by what holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceBudget {
    /// The five query vectors: `G1.A`, `G1.B`, `G1.K`, `G1.Z` at 64 bytes a point, `G2.B`
    /// at 128.
    pub bases: u64,
    /// Both twiddle tables and the coset powers.
    pub tables: u64,
    /// One proof's stage scratch: the witness in both forms and six domain vectors.
    pub scratch: u64,
    /// The MSM arena, sized for the largest batch of a run of `msm_chunk` points.
    pub msm: u64,
    /// The above with every buffer rounded up to [`PAGE`], plus [`DEVICE_MARGIN`]: what
    /// construction checks against the device's free memory.
    pub total: u64,
}

/// What a [`GnarkProver`] for `header` takes on the device, from the dump's header alone, so
/// a caller can check it against `cudaMemGetInfo` before committing to a construction.
/// [`DumpHeader::read`] sizes a 6 GB dump in milliseconds.
pub fn required_device_bytes(
    header: &DumpHeader,
    msm_chunk: usize,
) -> Result<DeviceBudget, ProveError> {
    let n = header.domain_size;
    let w = header.n_wires;
    let g1 = |points: usize| points.max(1) as u64 * 64;
    let bases = vec![
        g1(w),
        g1(w),
        g1(header.g1_k),
        g1(header.g1_z),
        w.max(1) as u64 * 128,
    ];
    let tables = CudaStages::rows_table_buffers(n);
    let scratch = CudaStages::scratch_buffers(n, w);
    let msm: Vec<u64> = resident_scratch(w, header.g1_k, header.g1_z, msm_chunk, None)?
        .into_iter()
        .map(|words| words.max(1) as u64 * 4)
        .collect();
    let paged = |b: &u64| b.div_ceil(PAGE) * PAGE;
    let total = [&bases, &tables, &scratch, &msm]
        .iter()
        .flat_map(|v| v.iter())
        .map(paged)
        .sum::<u64>()
        + DEVICE_MARGIN;
    Ok(DeviceBudget {
        bases: bases.iter().sum(),
        tables: tables.iter().sum(),
        scratch: scratch.iter().sum(),
        msm: msm.iter().sum(),
        total,
    })
}

/// Where construction's time went, in microseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct GnarkProverCost {
    /// Mapping the dump and checking its header, the verifying key against it.
    pub open_us: u64,
    /// NVRTC and the driver's JIT for both units, or their caches.
    pub compile_us: u64,
    /// The transform tables and one proof's scratch.
    pub stages_us: u64,
    /// The five queries read, checked and streamed to the device.
    pub bases_us: u64,
    /// The MSM arena.
    pub reserve_us: u64,
    /// [`GnarkProverConfig::warm_up`]'s pass, zero without it.
    pub warm_up_us: u64,
    pub total_us: u64,
}

/// A gnark key resident on one device, every byte a proof needs reserved; see the module
/// docs.
pub struct GnarkProver {
    circuit: CudaCircuit,
    cuda: Cuda,
    budget: DeviceBudget,
    cost: GnarkProverCost,
    /// One proof at a time: the reservation holds one proof's scratch.
    one: Mutex<()>,
}

impl GnarkProver {
    /// [`Self::open_with`] on device 0 with runs of [`MSM_CHUNK`].
    pub fn open(pk_dump: &Path, vk: &GnarkVerifyingKey) -> Result<Self, ProveError> {
        Self::open_with(pk_dump, vk, GnarkProverConfig::default())
    }

    /// Read `pk_dump`, a gnark `WriteDump` proving key, onto `config.device`, checked against
    /// `vk` the way `GnarkProvingKey::into_proving_key` checks it, and reserve everything a
    /// proof needs. Refused before anything is allocated if the device has less free memory
    /// than [`required_device_bytes`] says; any later allocation failure is an error too,
    /// with everything allocated so far released.
    pub fn open_with(
        pk_dump: &Path,
        vk: &GnarkVerifyingKey,
        config: GnarkProverConfig,
    ) -> Result<Self, ProveError> {
        let t_all = Instant::now();
        let dump = GnarkDump::open(pk_dump).map_err(gnark)?;
        let header = dump.header().clone();
        let pk = dump.proving_key(vk).map_err(gnark)?;
        let budget = required_device_bytes(&header, config.msm_chunk)?;
        let open_us = t_all.elapsed().as_micros() as u64;

        // Probed first: without `libcuda` the driver call that saves the context panics.
        let no_device = |e| bad(format!("no usable CUDA device {}: {e}", config.device));
        crate::context::libcuda_loadable().map_err(no_device)?;
        let _restore = CurrentContext::save();
        let cuda = Cuda::new_private(config.device).map_err(no_device)?;
        let (free, total) = cuda
            .context()
            .mem_get_info()
            .map_err(|e| crate::context::fault(format!("query device memory: {e}"), e))?;
        if (free as u64) < budget.total {
            return Err(bad(format!(
                "this key needs {} bytes of device memory ({budget:?}); device {} ({}) has {free} \
                 of {total} free",
                budget.total,
                config.device,
                cuda.device_name()
            )));
        }

        let t = Instant::now();
        let stages_module = CudaStages::compile(&cuda)?;
        let msm_module = CudaMsm::compile(&cuda)?;
        let compile_us = t.elapsed().as_micros() as u64;

        let t = Instant::now();
        let stages = CudaStages::from_module(&cuda, stages_module, &pk)?;
        stages.reserve_scratch()?;
        let stages_us = t.elapsed().as_micros() as u64;

        let t = Instant::now();
        let stream = cuda.stream();
        let mut staging = Staging::new(cuda.context())?;
        let mut up = |q| upload_query(&dump, q, stream, &mut staging);
        let (a, b_g1, k, z, b_g2) = (
            up(Query::G1A)?,
            up(Query::G1B)?,
            up(Query::G1K)?,
            up(Query::G1Z)?,
            up(Query::G2B)?,
        );
        drop(staging);
        stream
            .synchronize()
            .map_err(|e| crate::context::fault(format!("synchronize key upload: {e}"), e))?;
        let bases = (
            G1Bases::from_device(a, dump.len(Query::G1A))?,
            G2Bases::from_device(b_g2, dump.len(Query::G2B))?,
            G1Bases::from_device(b_g1, dump.len(Query::G1B))?,
            G1Bases::from_device(k, dump.len(Query::G1K))?,
            G1Bases::from_device(z, dump.len(Query::G1Z))?,
        );
        drop(dump);
        let bases_us = t.elapsed().as_micros() as u64;

        let t = Instant::now();
        let arena = resident_scratch(
            header.n_wires,
            header.g1_k,
            header.g1_z,
            config.msm_chunk,
            None,
        )?;
        let msm = CudaMsm::from_module(&cuda, msm_module)?
            .with_bases(bases, pk.n_vars, pk.n_public)?
            .with_arena(&arena)?;
        let reserve_us = t.elapsed().as_micros() as u64;

        let (n_vars, domain) = (pk.n_vars, pk.domain_size);
        let prepare = PrepareCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        let circuit = CudaCircuit::from_parts(pk, stages, msm, prepare, config.msm_chunk);

        // Stages 1 to 9 once over the all-zero witness and no rows: every kernel a proof
        // launches runs here first, so whatever the driver sets up on a kernel's first
        // launch (its local memory, above all, for the MSM kernels that spill) is taken now
        // and not by the first job. Any rows will do, as C = A*B on them keeps H exact.
        let t = Instant::now();
        if config.warm_up {
            let zero = vec![snarkrs_field::Fr::from(0u64); n_vars];
            let mut t = StageTimings::default();
            snarkrs_groth16::PreparedCircuit::h_and_msms_from_rows(
                &circuit,
                &zero,
                &[],
                &[],
                &mut t,
            )
            .map_err(|e| bad(format!("warm-up pass on a domain of {domain}: {e}")))?;
        }
        let warm_up_us = t.elapsed().as_micros() as u64;

        let cost = GnarkProverCost {
            open_us,
            compile_us,
            stages_us,
            bases_us,
            reserve_us,
            warm_up_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        Ok(Self {
            circuit,
            cuda,
            budget,
            cost,
            one: Mutex::new(()),
        })
    }

    /// Prove `witness`, as gnark's solver left it, and check the proof against the
    /// verifying key before returning it, as `snarkrs_gnark::prove` does. Stage times go to
    /// `timings`; `msm_us` there is wall clock over every MSM batch.
    pub fn prove<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
        &self,
        witness: &GnarkWitness<'_>,
        rng: &mut R,
        timings: &mut StageTimings,
    ) -> Result<Proof, ProveError> {
        snarkrs_gnark::prove(self, witness, rng, timings)
    }

    /// The prepared circuit, for its per-proof breakdowns
    /// ([`CudaCircuit::last_witness_upload_us`], [`CudaCircuit::last_msm_device_us`]).
    pub fn circuit(&self) -> &CudaCircuit {
        &self.circuit
    }

    /// What construction reserved, by buffer kind.
    pub fn budget(&self) -> DeviceBudget {
        self.budget
    }

    /// Where construction's time went.
    pub fn cost(&self) -> GnarkProverCost {
        self.cost
    }

    pub fn device_name(&self) -> &str {
        self.cuda.device_name()
    }

    /// Free and total device memory, bytes, as the driver reports them now.
    pub fn device_memory(&self) -> Result<(usize, usize), ProveError> {
        let _restore = CurrentContext::save();
        self.cuda
            .context()
            .mem_get_info()
            .map_err(|e| crate::context::fault(format!("query device memory: {e}"), e))
    }
}

/// The CUDA backend's `prepare` for a gnark key on disk: [`GnarkProver::open_with`] on
/// `device` with the default runs and warm-up, boxed as the circuit `snarkrs_gnark::prove`
/// takes. Where a caller would write `CpuBackend::new().prepare(pk)` after
/// `GnarkProvingKey::read_dump(path)?.into_proving_key(&vk)?`, this reads `path` itself
/// and never builds that host key.
pub fn prepare(
    device: usize,
    pk_dump: &Path,
    vk: &GnarkVerifyingKey,
) -> Result<Box<dyn PreparedCircuit>, ProveError> {
    let config = GnarkProverConfig {
        device,
        ..GnarkProverConfig::default()
    };
    Ok(Box::new(GnarkProver::open_with(pk_dump, vk, config)?))
}

/// The circuit behind [`GnarkProver::prove`], for a caller that holds a
/// `Box<dyn PreparedCircuit>`. Stages 1 to 9 run under the prover's one-proof lock with the
/// calling thread's CUDA context put back after; the blinding, assembly and check around
/// them are the caller's thread's and run unlocked. [`PreparedCircuit::key`] is the host
/// half only, every query empty.
impl PreparedCircuit for GnarkProver {
    fn backend_name(&self) -> &'static str {
        self.circuit.backend_name()
    }
    fn n_vars(&self) -> usize {
        self.circuit.n_vars()
    }
    fn n_public(&self) -> usize {
        self.circuit.n_public()
    }
    fn domain_size(&self) -> usize {
        self.circuit.domain_size()
    }
    fn key(&self) -> &ProvingKey {
        self.circuit.key()
    }

    /// Refused, as for any gnark key: it has no constraint matrices.
    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        self.circuit.compute_h(witness, t)
    }

    /// Refused, as for any gnark key: its H basis is not the odd coset.
    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.circuit.msms(witness, h, t)
    }

    fn h_and_msms_from_rows(
        &self,
        witness: &[Fr],
        a_rows: &[Fr],
        b_rows: &[Fr],
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        let _one = self.one.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = CurrentContext::save();
        self.circuit
            .h_and_msms_from_rows(witness, a_rows, b_rows, t)
    }
}

/// One query read from the dump into a device buffer of its own, a staging chunk at a
/// time, and refused if it is all identity as `read_dump` refuses it.
fn upload_query(
    dump: &GnarkDump,
    q: Query,
    stream: &std::sync::Arc<CudaStream>,
    staging: &mut Staging,
) -> Result<CudaSlice<u32>, ProveError> {
    let stride = q.stride();
    let words = dump.len(q) * stride / 4;
    let mut buf = stream
        .alloc_zeros::<u32>(words.max(1))
        .map_err(|e| crate::context::fault(format!("allocate the {q:?} bases: {e}"), e))?;
    let mut live = 0;
    staging.upload(stream, &mut buf, 0, words, |start, out| {
        // SAFETY: any `u32` slice is a valid byte slice four times as long, and nothing else
        // holds this one while `read_points` writes it.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4)
        };
        // A staging chunk is a whole number of both point sizes, so `start` is on a point.
        live += dump
            .read_points(q, start * 4 / stride, bytes)
            .map_err(gnark)?;
        Ok(())
    })?;
    dump.check_not_all_infinity(q, live).map_err(gnark)?;
    Ok(buf)
}

/// A `Box<dyn PreparedCircuit>` must be `Send + Sync`; checked here rather than at the
/// coercion, where the error would name the trait instead of the field.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GnarkProver>();
};

/// The calling thread's current CUDA context, put back on drop. `cudarc` makes the
/// device's primary context current before every call it makes, and a caller with a
/// context of its own current on this thread should find it there afterwards.
struct CurrentContext(Option<sys::CUcontext>);

impl CurrentContext {
    fn save() -> Self {
        Self(cudarc::driver::result::ctx::get_current().ok().flatten())
    }
}

impl Drop for CurrentContext {
    fn drop(&mut self) {
        if let Some(ctx) = self.0 {
            // SAFETY: a context handle the driver returned as current on this thread, put
            // back as it was. A failure leaves cudarc's context current, which is no worse
            // than not restoring at all.
            let _ = unsafe { cudarc::driver::result::ctx::set_current(ctx) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_gnark::GnarkProvingKey;
    use snarkrs_gpu_layout::{as_bytes, PackedG1Affine, PackedG2Affine};
    use std::path::PathBuf;

    fn toy(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-support/gnark/toy")
            .join(name)
    }

    /// The bytes [`GnarkDump::read_points`] writes for each query are what
    /// `PackedG1Affine::pack_slice` and `PackedG2Affine::pack_slice` make of the decoded key:
    /// the dumped layout is the device layout, identity included, so construction copies it
    /// without repacking.
    #[test]
    fn a_dumped_point_is_its_packed_layout() {
        let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
        let dump = GnarkDump::open(&toy("pk.dump")).unwrap();
        for (q, packed) in [
            (
                Query::G1A,
                as_bytes(&PackedG1Affine::pack_slice(&pk.a)).to_vec(),
            ),
            (
                Query::G1B,
                as_bytes(&PackedG1Affine::pack_slice(&pk.b_g1)).to_vec(),
            ),
            (
                Query::G1K,
                as_bytes(&PackedG1Affine::pack_slice(&pk.k)).to_vec(),
            ),
            (
                Query::G1Z,
                as_bytes(&PackedG1Affine::pack_slice(&pk.z)).to_vec(),
            ),
            (
                Query::G2B,
                as_bytes(&PackedG2Affine::pack_slice(&pk.b_g2)).to_vec(),
            ),
        ] {
            let mut raw = vec![0u8; dump.len(q) * q.stride()];
            dump.read_points(q, 0, &mut raw).unwrap();
            assert_eq!(raw, packed, "{q:?}");
        }
        // The identity really is in there, so the zero encoding was compared and not
        // assumed.
        assert!(pk.a.iter().any(|p| p.infinity) && pk.b_g2.iter().any(|p| p.infinity));
    }

    /// The SP1 v6.1.0 key's shape, from its header (`DumpHeader::read` on the 5.86 GB dump).
    fn sp1() -> DumpHeader {
        DumpHeader {
            domain_size: 1 << 24,
            n_wires: 15_747_144,
            nb_infinity_a: 2_978_799,
            nb_infinity_b: 476_595,
            g1_a: 15_747_144 - 2_978_799,
            g1_b: 15_747_144 - 476_595,
            g1_z: (1 << 24) - 1,
            g1_k: 15_747_138,
            g2_b: 15_747_144 - 476_595,
        }
    }

    /// The reservation for SP1's key in runs of 2^22 fits the 27 GB the target card has
    /// beside the SP1 server, under the 24 GB aimed for. The figures are printed for the
    /// record; the arithmetic they come from is `required_device_bytes`'.
    #[test]
    fn the_sp1_reservation_fits_beside_the_sp1_server() {
        // `G16_GNARK_SP1_PK=<groth16_pk.bin>` checks the shape against the key itself.
        if let Some(path) = std::env::var_os("G16_GNARK_SP1_PK") {
            assert_eq!(DumpHeader::read(Path::new(&path)).unwrap(), sp1());
        }
        let b = required_device_bytes(&sp1(), MSM_CHUNK).unwrap();
        eprintln!("SP1 v6.1.0, runs of 2^22: {b:?}");
        assert_eq!(
            b.bases,
            (2 * 15_747_144 + 15_747_138 + ((1 << 24) - 1)) * 64 + 15_747_144 * 128
        );
        assert_eq!(b.tables, (1u64 << 25) * 32 + 24);
        assert_eq!(b.scratch, (2 * 15_747_144 + 6 * (1u64 << 24) + 1) * 32);
        assert!(b.total < 24_000_000_000, "{b:?}");
        assert!(b.total > b.bases + b.tables + b.scratch + b.msm);
        for chunk in [1 << 21, 1 << 23] {
            let c = required_device_bytes(&sp1(), chunk).unwrap();
            eprintln!("SP1 v6.1.0, runs of 2^{}: {c:?}", chunk.trailing_zeros());
        }
    }

    /// The arena is each batch's buffers, largest first, maxed rank for rank, so every
    /// batch of the schedule fits it rank for rank: the condition the arena's smallest-fit
    /// draw needs to never miss. Includes runs short enough to pick a narrower window.
    #[test]
    fn every_resident_batch_fits_the_reservation() {
        use crate::msm::{batch_words, resident_schedule, Resident, Work};
        for (n_vars, l, h, chunk) in [
            (3072, 3067, 4095, MSM_CHUNK),
            (15_747_144, 15_747_138, (1 << 24) - 1, MSM_CHUNK),
            (5000, 4990, 8191, 2048),
            (4097, 4000, 4097, 4096),
            (10, 3, 15, 4),
        ] {
            let arena = resident_scratch(n_vars, l, h, chunk, None).unwrap();
            for (stage, _, n) in resident_schedule(n_vars, l, h, chunk).unwrap() {
                let g2: &[bool] = match stage {
                    Resident::Witness => &[false, true, false],
                    _ => &[false],
                };
                let mut words = batch_words(n, g2, None, Work::Variable).unwrap();
                words.sort_unstable_by(|a, b| b.cmp(a));
                assert!(words.len() <= arena.len());
                for (w, r) in words.iter().zip(&arena) {
                    assert!(w <= r, "{stage:?} run of {n}: {words:?} against {arena:?}");
                }
            }
        }
    }
}
