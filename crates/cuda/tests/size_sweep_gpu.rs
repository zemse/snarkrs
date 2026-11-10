//! Where a card stops proving: one power-of-two domain per process, on a synthetic key.
//!
//! Driven by `bench/scripts/cuda-size-sweep.sh`, which runs [`size_sweep_point`] once per
//! domain and phase, so each process's `VmHWM` is that phase's alone and a host OOM kill
//! takes down one point rather than the sweep. Every knob is an environment variable:
//!
//! * `G16_SWEEP_LOG`: the domain is `2^log` (required).
//! * `G16_SWEEP_PHASE`: `cuda` (default) proves on the device and writes the proof, `cpu`
//!   rebuilds the same key, proves it on the CPU backend and compares.
//! * `G16_SWEEP_OUT`: where both phases write (default `.`).
//! * `G16_SWEEP_VARS`: `n_vars` as a fraction of the domain (default 0.75, the onion
//!   ladder's wires over its domain).
//! * `G16_SWEEP_BISECT=0` skips the search for the planned batch size below.
//!
//! The key is random, so the proof verifies against nothing; the CPU backend with the same
//! blinders is the oracle, and a proof that differs from it fails the `cpu` phase. A
//! device that cannot hold the key must refuse it as `ProveError::DeviceMemory` before
//! allocating, or cut the MSMs into runs; any other error fails the `cuda` phase.
//!
//! The `cuda` phase also measures what `VRAM_HEADROOM` guards. A thread polls the free
//! device memory through each MSM call, so the batch's peak is the free memory before it
//! less the least seen during it. The bytes the planner sized that batch at are found
//! from outside by bisecting `with_vram_limit`: one batch exactly when the limit is at
//! least the plan. Peak less plan is what the driver took beyond the planned buffers.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use snarkrs_cuda::msm::CudaMsm;
use snarkrs_cuda::stages::CudaStages;
use snarkrs_cuda::Cuda;
use snarkrs_field::{CurveGroup, Fr, G1Projective, G2Projective, PrimeField, PrimeGroup};
use snarkrs_formats::{Coefficients, HBasis, ProvingKey, VerifyingKey};
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::prove::prove_with_blinders;
use snarkrs_groth16::{
    Backend, HPoly, MsmOutputs, PreparedCircuit, Proof, ProveError, StageTimings,
};

/// `CudaCircuit`'s two halves, held here so the MSM half can be rebuilt with a limit and
/// read for its run count; the proving path is the same `compute_h` then `msms`.
struct Circuit {
    pk: ProvingKey,
    stages: CudaStages,
    msm: Option<CudaMsm>,
}

impl Circuit {
    fn msm(&self) -> &CudaMsm {
        self.msm.as_ref().expect("an MSM half")
    }
    fn limit(&mut self, bytes: u64) {
        let m = self.msm.take().expect("an MSM half");
        self.msm = Some(m.with_vram_limit(bytes));
    }
}

impl PreparedCircuit for Circuit {
    fn backend_name(&self) -> &'static str {
        "cuda"
    }
    fn n_vars(&self) -> usize {
        self.pk.n_vars
    }
    fn n_public(&self) -> usize {
        self.pk.n_public
    }
    fn domain_size(&self) -> usize {
        self.pk.domain_size
    }
    fn key(&self) -> &ProvingKey {
        &self.pk
    }
    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        self.stages.compute_h(witness, t)
    }
    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.msm().msms(witness, h, t)
    }
}

/// The least free device memory seen since the last [`Sampler::reset`].
struct Sampler {
    min_free: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    fn start(cuda: &Cuda) -> Self {
        let ctx = cuda.context().clone();
        let min_free = Arc::new(AtomicU64::new(u64::MAX));
        let stop = Arc::new(AtomicBool::new(false));
        let (m, s) = (min_free.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Ok((free, _)) = ctx.mem_get_info() {
                    m.fetch_min(free as u64, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        });
        Self {
            min_free,
            stop,
            thread: Some(thread),
        }
    }
    fn reset(&self) {
        self.min_free.store(u64::MAX, Ordering::Relaxed);
    }
    fn min_free(&self) -> u64 {
        self.min_free.load(Ordering::Relaxed)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v:?} does not parse")),
        Err(_) => default,
    }
}

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_fr(seed: &mut u64) -> Fr {
    let mut b = [0u8; 32];
    for c in b.chunks_mut(8) {
        c.copy_from_slice(&splitmix(seed).to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&b)
}

/// `(first + i)` times the generator for `i` in `0..n`, on every core, a chunk at a time so
/// no projective copy of the whole vector is ever held. The same points for any thread count.
fn multiples<P: CurveGroup + Send + Sync>(n: usize, first: u64, g: P) -> Vec<P::Affine> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let mut out = vec![P::Affine::default(); n];
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, part) in out.chunks_mut(per).enumerate() {
            scope.spawn(move || {
                let mut cur = g * P::ScalarField::from(first + (t * per) as u64);
                for block in part.chunks_mut(1 << 14) {
                    let proj: Vec<P> = block
                        .iter()
                        .map(|_| {
                            let p = cur;
                            cur += g;
                            p
                        })
                        .collect();
                    block.copy_from_slice(&P::normalize_batch(&proj));
                }
            });
        }
    });
    out
}

/// A key on a domain of `2^log` with `n_vars` signals and two random CSR matrices of about
/// one and a half entries a row. The bases are distinct multiples of the generator, a
/// different run per query, so no MSM's answer is the identity. Deterministic in `log` and
/// `n_vars`, so the `cpu` phase rebuilds the key the `cuda` phase proved.
fn synthetic_key(log: u32, n_vars: usize) -> ProvingKey {
    let domain = 1usize << log;
    let g1 = G1Projective::generator();
    let g2 = G2Projective::generator();
    let mat = |seed: u64| {
        let mut seed = seed;
        let mut row_ptr = Vec::with_capacity(domain + 1);
        let mut signal = Vec::with_capacity(domain * 3 / 2);
        let mut value = Vec::with_capacity(domain * 3 / 2);
        row_ptr.push(0u32);
        for _ in 0..domain {
            for _ in 0..splitmix(&mut seed) % 4 {
                signal.push((splitmix(&mut seed) % n_vars as u64) as u32);
                value.push(rand_fr(&mut seed));
            }
            row_ptr.push(signal.len() as u32);
        }
        (row_ptr, signal, value)
    };
    let base = 0x5133_0000_0000_0000u64 | u64::from(log);
    let ((rp0, sg0, v0), (rp1, sg1, v1)) = std::thread::scope(|s| {
        let a = s.spawn(|| mat(base));
        let b = s.spawn(|| mat(base ^ 0xB));
        (a.join().unwrap(), b.join().unwrap())
    });
    let n_public = 1usize;
    let one = g1.into_affine();
    let one2 = g2.into_affine();
    ProvingKey {
        n_vars,
        n_public,
        domain_size: domain,
        alpha_g1: one,
        beta_g1: one,
        beta_g2: one2,
        delta_g1: one,
        delta_g2: one2,
        a_query: multiples(n_vars, 2, g1),
        b_g1_query: multiples(n_vars, 3 << 32, g1),
        b_g2_query: multiples::<G2Projective>(n_vars, 2, g2),
        l_query: multiples(n_vars - n_public - 1, 5 << 32, g1),
        h_query: multiples(domain, 7 << 32, g1),
        h_basis: HBasis::SnarkjsOddCoset,
        coeffs: Coefficients::from_values([rp0, rp1], [sg0, sg1], [v0, v1]),
        vk: VerifyingKey {
            alpha_g1: one,
            beta_g2: one2,
            gamma_g2: one2,
            delta_g2: one2,
            ic: vec![one; n_public + 1],
        },
    }
}

fn witness(log: u32, n_vars: usize) -> Vec<Fr> {
    let mut seed = 0x0057_EE90_0000_0000u64 | u64::from(log);
    // Wire 0 is the constant one, which the prover checks.
    let mut w: Vec<Fr> = (0..n_vars).map(|_| rand_fr(&mut seed)).collect();
    w[0] = Fr::from(1u64);
    w
}

fn proof_text(p: &Proof) -> String {
    format!("{:?}\n{:?}\n{:?}\n", p.a, p.b, p.c)
}

/// `VmHWM` and `VmRSS` of this process, KiB.
fn host_kib() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.split_whitespace().next()?.parse().ok())
            .unwrap_or(0)
    };
    (field("VmHWM:"), field("VmRSS:"))
}

/// How a device failure reads: the planner's own refusal, or the driver running out.
fn failure(e: &ProveError) -> &'static str {
    let s = e.to_string();
    if matches!(e, ProveError::DeviceMemory { .. }) {
        "refused"
    } else if s.contains("OUT_OF_MEMORY") {
        "driver_oom"
    } else {
        "other"
    }
}

const MIB: f64 = (1u64 << 20) as f64;

/// One record, written as one JSON object per phase and printed after `SWEEP `.
struct Record(Vec<(String, String)>);

impl Record {
    fn num(&mut self, k: &str, v: impl std::fmt::Display) {
        self.0.push((k.into(), v.to_string()));
    }
    fn text(&mut self, k: &str, v: &str) {
        self.0.push((k.into(), format!("{:?}", v)));
    }
    fn json(&self) -> String {
        let body: Vec<String> = self.0.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        format!("{{{}}}", body.join(","))
    }
}

#[test]
#[ignore = "requires NVIDIA; one sweep point per process, see bench/scripts/cuda-size-sweep.sh"]
fn size_sweep_point() {
    let log: u32 = env_or("G16_SWEEP_LOG", u32::MAX);
    assert!(
        log <= 28,
        "set G16_SWEEP_LOG to the log2 domain, at most 28"
    );
    let phase: String = env_or("G16_SWEEP_PHASE", "cuda".to_string());
    let out = PathBuf::from(env_or("G16_SWEEP_OUT", ".".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let frac: f64 = env_or("G16_SWEEP_VARS", 0.75);
    let n_vars = (((1usize << log) as f64 * frac) as usize).max(4);

    let mut rec = Record(Vec::new());
    rec.num("log", log);
    rec.text("phase", &phase);
    rec.num("n_vars", n_vars);
    let t = Instant::now();
    let pk = synthetic_key(log, n_vars);
    let w = witness(log, n_vars);
    rec.num("keygen_s", format!("{:.1}", t.elapsed().as_secs_f64()));
    let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
    let proof_path = out.join(format!("2^{log}.proof"));

    let result = match phase.as_str() {
        "cuda" => cuda_phase(pk, &w, r, s, &proof_path, &mut rec).map(|_| ()),
        "cpu" => {
            let t = Instant::now();
            let cpu = CpuBackend::new().prepare(pk).expect("cpu prepare");
            let mut st = StageTimings::default();
            let p = prove_with_blinders(cpu.as_ref(), &w, r, s, &mut st).expect("cpu prove");
            rec.num("cpu_prove_s", format!("{:.1}", t.elapsed().as_secs_f64()));
            match std::fs::read_to_string(&proof_path) {
                Ok(got) => {
                    let same = got == proof_text(&p);
                    rec.num("matches_cuda", same);
                    if same {
                        Ok(())
                    } else {
                        Err(format!("2^{log}: the cuda proof differs from the cpu's"))
                    }
                }
                Err(_) => {
                    rec.num("matches_cuda", "null");
                    Ok(())
                }
            }
        }
        p => panic!("G16_SWEEP_PHASE={p:?}: want cuda or cpu"),
    };
    let (hwm, _) = host_kib();
    rec.num("host_hwm_mib", format!("{:.0}", hwm as f64 / 1024.0));
    let line = rec.json();
    println!("SWEEP {line}");
    std::fs::write(out.join(format!("2^{log}-{phase}.json")), line + "\n").unwrap();
    if let Err(e) = result {
        panic!("{e}");
    }
}

/// Prepare, one cold MSM call, a proof, one warm MSM call, then the planned-batch search.
/// `Ok` for a proof or a clean refusal, `Err` for anything else.
fn cuda_phase(
    pk: ProvingKey,
    w: &[Fr],
    r: Fr,
    s: Fr,
    proof_path: &std::path::Path,
    rec: &mut Record,
) -> Result<Option<Proof>, String> {
    let cuda = Cuda::new(0).expect("a usable CUDA device");
    let (free0, total) = cuda.context().mem_get_info().unwrap();
    rec.num("device_total_mib", format!("{:.0}", total as f64 / MIB));
    rec.num("free_at_open_mib", format!("{:.0}", free0 as f64 / MIB));
    let t = Instant::now();
    let stages_module = CudaStages::compile(&cuda).expect("NVRTC stages");
    let msm_module = CudaMsm::compile(&cuda).expect("NVRTC msm");
    rec.num("compile_s", format!("{:.1}", t.elapsed().as_secs_f64()));
    let sampler = Sampler::start(&cuda);
    let mib = |b: u64| format!("{:.1}", b as f64 / MIB);
    let free = || cuda.context().mem_get_info().unwrap().0 as u64;

    let t = Instant::now();
    let prepared = CudaStages::from_module(&cuda, stages_module, &pk).and_then(|stages| {
        let msm = CudaMsm::from_module(&cuda, msm_module)?.with_key(&pk)?;
        Ok((stages, msm))
    });
    let (stages, msm) = match prepared {
        Ok(p) => p,
        Err(e) => {
            rec.text("outcome", &format!("prepare_{}", failure(&e)));
            rec.text("error", &e.to_string());
            return clean(&e);
        }
    };
    let mut c = Circuit {
        pk,
        stages,
        msm: Some(msm),
    };
    rec.num("prepare_s", format!("{:.1}", t.elapsed().as_secs_f64()));
    rec.num("free_after_prepare_mib", mib(free()));

    // The first MSM call on this context: whatever the driver reserves at a kernel's first
    // launch lands inside this batch's peak.
    let mut st = StageTimings::default();
    let h = match c.compute_h(w, &mut st) {
        Ok(h) => h,
        Err(e) => {
            rec.text("outcome", &format!("compute_h_{}", failure(&e)));
            rec.text("error", &e.to_string());
            return clean(&e);
        }
    };
    let batch = |c: &Circuit, h: &HPoly, st: &mut StageTimings| {
        let before = free();
        sampler.reset();
        let t = Instant::now();
        let got = c.msms(w, h, st);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (got, before, before.saturating_sub(sampler.min_free()), ms)
    };
    let (got, before, cold_peak, cold_ms) = batch(&c, &h, &mut st);
    if let Err(e) = got {
        rec.text("outcome", &format!("msm_{}", failure(&e)));
        rec.text("error", &e.to_string());
        return clean(&e);
    }
    rec.num("runs", c.msm().last_runs());
    rec.num("free_before_msm_mib", mib(before));
    rec.num("msm_cold_peak_mib", mib(cold_peak));
    rec.num("msm_cold_ms", format!("{cold_ms:.0}"));
    drop(h);

    sampler.reset();
    let free_before_prove = free();
    let mut st = StageTimings::default();
    let t = Instant::now();
    let proof = match prove_with_blinders(&c, w, r, s, &mut st) {
        Ok(p) => p,
        Err(e) => {
            rec.text("outcome", &format!("prove_{}", failure(&e)));
            rec.text("error", &e.to_string());
            return clean(&e);
        }
    };
    rec.num(
        "prove_ms",
        format!("{:.0}", t.elapsed().as_secs_f64() * 1e3),
    );
    rec.num(
        "prove_peak_mib",
        mib(free_before_prove.saturating_sub(sampler.min_free())),
    );
    rec.num("min_free_in_prove_mib", mib(sampler.min_free()));
    rec.num("msm_ms", format!("{:.0}", st.msm_us as f64 / 1e3));
    rec.num("runs_prove", c.msm().last_runs());
    std::fs::write(proof_path, proof_text(&proof)).unwrap();

    let mut st = StageTimings::default();
    let h = c.compute_h(w, &mut st).map_err(|e| e.to_string())?;
    let (got, _, warm_peak, warm_ms) = batch(&c, &h, &mut st);
    got.map_err(|e| e.to_string())?;
    rec.num("msm_warm_peak_mib", mib(warm_peak));
    rec.num("msm_warm_ms", format!("{warm_ms:.0}"));
    rec.num("runs_warm", c.msm().last_runs());
    rec.num("free_after_warm_mib", mib(free()));
    rec.num("min_free_in_warm_msm_mib", mib(sampler.min_free()));

    // The planned bytes of the one batch: the least limit that still runs one batch,
    // to 64 KiB. Only when unlimited is one batch, since a cut proof plans per run.
    if c.msm().last_runs() == 1 && env_or("G16_SWEEP_BISECT", 1u32) == 1 {
        let one = |c: &mut Circuit, limit: u64| {
            c.limit(limit);
            let mut st = StageTimings::default();
            c.msms(w, &h, &mut st).is_ok() && c.msm().last_runs() == 1
        };
        let (mut lo, mut hi) = (0u64, warm_peak + (64 << 20));
        while !one(&mut c, hi) {
            (lo, hi) = (hi, hi * 2);
        }
        while hi - lo > 64 << 10 {
            let mid = lo + (hi - lo) / 2;
            if one(&mut c, mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        rec.num("planned_mib", mib(hi));
        rec.num("cold_over_plan_mib", mib(cold_peak.saturating_sub(hi)));
        rec.num("warm_over_plan_mib", mib(warm_peak.saturating_sub(hi)));
    }
    rec.text("outcome", "proved");
    Ok(Some(proof))
}

/// The planner's `DeviceMemory` refusal is clean; anything else fails the point.
fn clean(e: &ProveError) -> Result<Option<Proof>, String> {
    if matches!(e, ProveError::DeviceMemory { .. }) {
        Ok(None)
    } else {
        Err(e.to_string())
    }
}
