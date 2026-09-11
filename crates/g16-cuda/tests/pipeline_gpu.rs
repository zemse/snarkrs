//! The whole pipeline on a real NVIDIA device: stages 0 to 11, on the checked-in
//! artifacts, verified.
//!
//! This is the deliverable for the CUDA backend. Everything below either produces a proof
//! that our own verifier accepts, or compares the GPU's intermediates against the CPU
//! backend element by element. The CPU backend is the oracle throughout: it already agrees
//! with snarkjs' own `buildABC1 -> ifft -> batchApplyKey -> fft -> joinABC` index by
//! index, so agreeing with it pins the CSR gather, the transform ordering, the iNTT
//! normalisation, the coset shift, the Montgomery convention and the window decomposition
//! all at once.
//!
//! This is exact integer arithmetic in a prime field. Anything short of every element
//! matching is a failure, not a tolerance.
//!
//! # Skipping
//!
//! Every test here skips, loudly, when there is no NVIDIA device or no artifacts, because
//! the workspace is developed on an M2 Max where neither is guaranteed. A skip prints the
//! reason to stderr, so a silent pass on a box that *does* have a card cannot be mistaken
//! for a real one. `CudaBackend::new` is what makes that possible at all: `cudarc`'s
//! dynamic loader panics when `libcuda` is absent, and the backend converts that into an
//! error precisely so this file can print a line instead of aborting the process.
//!
//! # Why everything shares one backend and one set of prepared circuits
//!
//! NVRTC plus the driver's ptxas over the MSM unit costs 275 s on a T4 the first time it
//! is ever run and 2 s on every run after that, because the driver caches PTX-to-cubin on
//! disk; a prepared circuit is on top of that hundreds of megabytes of base vectors across
//! PCIe. Both are process-lifetime costs by design, so they are paid once here in a
//! `OnceLock` and shared by every test. Building a backend per test would not be a slower
//! test suite, it would be a test suite that never finishes, and the eight tests below
//! would each be paying for the same eight kernels.
//!
//! The first run on a fresh box therefore takes about five minutes before the first
//! assertion. That is the CUDA installation warming up, not a hang.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use g16_core::cpu::CpuBackend;
use g16_core::prove::prove_with_blinders;
use g16_core::verify::verify;
use g16_core::{Backend, HPoly, PreparedCircuit, ProveError, StageTimings};
use g16_cuda::stages::{HHandle, TAG};
use g16_cuda::CudaBackend;
use g16_field::Fr;
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One artifact, prepared on both backends, with its witness and verifying key.
///
/// The CPU circuit rides along because it is the oracle for the two comparison tests, and
/// preparing it is cheap next to parsing the zkey, which has already been paid by then.
struct Fixture {
    name: String,
    cuda: Box<dyn PreparedCircuit>,
    cpu: Box<dyn PreparedCircuit>,
    witness: Vec<Fr>,
    vk: VerifyingKey,
}

impl Fixture {
    fn public(&self) -> Vec<Fr> {
        // The public signals are the witness prefix, which is what snarkjs publishes.
        // Taken from the witness rather than parsed out of `public.json` so this crate
        // needs no JSON dependency to test the whole pipeline.
        self.witness[1..=self.cuda.n_public()].to_vec()
    }
}

fn artifact_dirs() -> Vec<(String, PathBuf)> {
    // `G16_ARTIFACTS` first, because CARGO_MANIFEST_DIR is baked in at compile time and
    // this binary is cross-compiled on a Mac to be run on a rented GPU box, where that
    // path does not exist. Without the override the canonicalize below fails, this
    // returns nothing, and every test that iterates it passes having checked nothing.
    // A GPU correctness suite that silently tests zero artifacts is worse than one that
    // does not run, because it reports success.
    let root = match std::env::var_os("G16_ARTIFACTS") {
        Some(p) => PathBuf::from(p)
            .canonicalize()
            .unwrap_or_else(|e| panic!("G16_ARTIFACTS is set but unusable: {e}")),
        None => match Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts")
            .canonicalize()
        {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        },
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            ["circuit.zkey", "circuit.wtns", "vkey.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    // Sorted so the small circuits run first: if the backend is broken at all, `tiny_mul`
    // says so in a second rather than after a 2^18 domain has been proved.
    out.sort_by_key(|(_, d)| {
        std::fs::metadata(d.join("circuit.zkey"))
            .map(|m| m.len())
            .unwrap_or(u64::MAX)
    });
    out
}

/// `None` plus a printed reason when there is no device or no artifacts. Every test starts
/// with this and returns early, so the suite is honest about what it did not check.
fn fixtures() -> Option<&'static [Fixture]> {
    static ONCE: OnceLock<Vec<Fixture>> = OnceLock::new();
    let built = ONCE.get_or_init(|| {
        let backend = match CudaBackend::new() {
            Ok(b) => {
                let (major, minor) = b.compute_capability();
                eprintln!(
                    "cuda device: {} (sm_{major}{minor}, {} SMs), kernels compiled in {:.2} s",
                    b.device_name(),
                    b.sm_count(),
                    b.compile_us() as f64 / 1e6,
                );
                b
            }
            Err(e) => {
                eprintln!("SKIPPING every cuda pipeline test, no usable device: {e}");
                return Vec::new();
            }
        };
        let cpu = CpuBackend::new();

        let dirs = artifact_dirs();
        if dirs.is_empty() {
            eprintln!("SKIPPING every cuda pipeline test: no artifacts under bench/artifacts");
            return Vec::new();
        }

        let mut out = Vec::new();
        for (name, dir) in dirs {
            let t = Instant::now();
            let pk = ProvingKey::load(&dir.join("circuit.zkey"))
                .unwrap_or_else(|e| panic!("{name}: loading circuit.zkey: {e}"));
            let witness = Witness::load(&dir.join("circuit.wtns"))
                .unwrap_or_else(|e| panic!("{name}: loading circuit.wtns: {e}"))
                .0;
            let vk = VerifyingKey::from_json(&dir.join("vkey.json"))
                .unwrap_or_else(|e| panic!("{name}: loading vkey.json: {e}"));
            let load_ms = t.elapsed().as_secs_f64() * 1000.0;

            let t = Instant::now();
            let cuda_circuit = backend
                .prepare(pk)
                .unwrap_or_else(|e| panic!("{name}: cuda prepare: {e}"));
            let prep_ms = t.elapsed().as_secs_f64() * 1000.0;
            // Parsed a second time rather than cloned: `ProvingKey` is not `Clone`, and
            // deriving it would put a hundred-megabyte copy one keystroke away in the
            // proving path. The reparse costs a test a few seconds and costs the library
            // nothing.
            let cpu_circuit = cpu
                .prepare(
                    ProvingKey::load(&dir.join("circuit.zkey"))
                        .unwrap_or_else(|e| panic!("{name}: reloading circuit.zkey: {e}")),
                )
                .unwrap_or_else(|e| panic!("{name}: cpu prepare: {e}"));

            eprintln!(
                "{name}: n_vars {}, domain {}, load {load_ms:.0} ms, cuda prepare {prep_ms:.0} ms",
                cuda_circuit.n_vars(),
                cuda_circuit.domain_size(),
            );
            out.push(Fixture {
                name,
                cuda: cuda_circuit,
                cpu: cpu_circuit,
                witness,
                vk,
            });
        }
        // The backend itself is dropped here and the circuits outlive it on purpose: each
        // one holds an `Arc` of the compiled modules and its own clone of the context and
        // stream, so the modules stay loaded for exactly as long as something can launch
        // out of them. If that were wrong, the first launch below would fail with an
        // invalid handle rather than silently misbehave.
        out
    });
    (!built.is_empty()).then_some(built.as_slice())
}

/// Runs `f` on every fixture, or prints one skip line and returns.
///
/// One test at a time, whatever `--test-threads` says. Two of these running at once would
/// be correct (see `CudaCircuit`'s concurrency notes) but they would have several proofs'
/// worth of bucket arrays live on a 15 GB card at the same time, and an out-of-memory
/// failure here would read as a backend bug. Serialising also keeps the CUDA-event
/// timings printed below meaningful, since an event measures the stream rather than the
/// call. `one_cuda_circuit_proves_concurrently` still runs its four threads concurrently:
/// the lock is held by the test, not by each proof.
fn for_each(test: &str, f: impl Fn(&Fixture)) {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A test that panicked poisoned the lock. That is exactly when the remaining tests
    // still need to run, so the poison is stepped over rather than propagated.
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let Some(all) = fixtures() else {
        eprintln!("SKIPPED {test}");
        return;
    };
    for fx in all {
        eprintln!("{test}: {}", fx.name);
        f(fx);
    }
}

// ---------------------------------------------------------------------------
// The proof
// ---------------------------------------------------------------------------

/// The test this backend exists to pass: every stage on the GPU, our own verifier as the
/// oracle. Blinders are fixed rather than random so a failure is reproducible; the
/// randomised path is exercised through the CLI.
#[test]
fn a_cuda_proof_verifies_on_every_artifact() {
    for_each("a_cuda_proof_verifies_on_every_artifact", |fx| {
        assert_eq!(fx.cuda.backend_name(), "cuda", "{}", fx.name);
        let public = fx.public();
        let mut t = StageTimings::default();
        let t0 = Instant::now();
        let proof = prove_with_blinders(
            fx.cuda.as_ref(),
            &fx.witness,
            Fr::from(31337u64),
            Fr::from(4242u64),
            &mut t,
        )
        .unwrap_or_else(|e| panic!("{}: prove: {e}", fx.name));
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        verify(&fx.vk, &public, &proof).unwrap_or_else(|e| panic!("{}: verify: {e}", fx.name));
        eprintln!(
            "  proved in {wall:.1} ms  (gather {:.1}, ntt {:.1}, pointwise {:.1}, msm {:.1}, \
             assemble {:.1} ms)",
            t.gather_us as f64 / 1000.0,
            t.ntt_us as f64 / 1000.0,
            t.pointwise_us as f64 / 1000.0,
            t.msm_us as f64 / 1000.0,
            t.assemble_us as f64 / 1000.0,
        );
        // A backend that reported nothing would make `bench` print a free prover.
        assert!(t.msm_us > 0, "{}: msms reported no time", fx.name);
        assert!(t.gather_us > 0, "{}: stage 0 reported no time", fx.name);
    });
}

/// Zero blinders remove the masking, so the H evaluations are the only thing between the
/// MSMs and the pairing check. This is the case that fails loudly if stage 4's buffer, the
/// Montgomery convention or the coset were wrong.
#[test]
fn zero_blinders_still_verify() {
    for_each("zero_blinders_still_verify", |fx| {
        let public = fx.public();
        let mut t = StageTimings::default();
        let proof = prove_with_blinders(
            fx.cuda.as_ref(),
            &fx.witness,
            Fr::from(0u64),
            Fr::from(0u64),
            &mut t,
        )
        .unwrap_or_else(|e| panic!("{}: prove: {e}", fx.name));
        verify(&fx.vk, &public, &proof).unwrap_or_else(|e| panic!("{}: verify: {e}", fx.name));
    });
}

/// One prepared circuit, several threads. The promise `PreparedCircuit` makes, and the
/// reason the stage scratch is pooled behind a mutex rather than stored on the circuit.
///
/// A circuit that handed two in-flight proofs the same H buffer would not crash: it would
/// produce a proof that simply fails to verify. That is why the check here is a
/// verification and not just a join.
#[test]
fn one_cuda_circuit_proves_concurrently() {
    for_each("one_cuda_circuit_proves_concurrently", |fx| {
        let public = fx.public();
        std::thread::scope(|scope| {
            let threads: Vec<_> = (1..=4u64)
                .map(|i| {
                    let public = &public;
                    scope.spawn(move || {
                        let mut t = StageTimings::default();
                        let proof = prove_with_blinders(
                            fx.cuda.as_ref(),
                            &fx.witness,
                            Fr::from(i * 7),
                            Fr::from(i * 11),
                            &mut t,
                        )
                        .unwrap_or_else(|e| panic!("{}: prove: {e}", fx.name));
                        verify(&fx.vk, public, &proof)
                            .unwrap_or_else(|e| panic!("{}: verify: {e}", fx.name));
                    })
                })
                .collect();
            for th in threads {
                th.join().unwrap();
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Bisecting a wrong proof: stages 0-4, then stages 5-9
// ---------------------------------------------------------------------------

/// Stages 0 to 4 against the CPU backend, element by element, in both encodings.
///
/// The first discriminator when a proof fails to verify. `h_mont` is the Montgomery form
/// the transforms produce and `h_std` is the standard-form copy stage 9 reads its window
/// digits out of; checking both is what separates "the NTT is wrong" from "the mont-to-std
/// conversion is wrong", and only the second of those would leave the proof wrong by a
/// factor of R.
#[test]
fn cuda_h_matches_the_cpu_backend() {
    for_each("cuda_h_matches_the_cpu_backend", |fx| {
        let mut t = StageTimings::default();
        let want = fx
            .cpu
            .compute_h(&fx.witness, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu compute_h: {e}", fx.name));
        let want = want
            .to_host()
            .expect("the cpu backend returns a host vector")
            .to_vec();

        let got = fx
            .cuda
            .compute_h(&fx.witness, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda compute_h: {e}", fx.name));
        let handle = got
            .device_handle::<HHandle>(TAG)
            .expect("cuda compute_h returned something other than a cuda device handle");

        let mont = handle
            .to_host()
            .unwrap_or_else(|e| panic!("{}: reading h_mont back: {e}", fx.name));
        assert_eq!(mont.len(), want.len(), "{}: h_mont length", fx.name);
        for (i, (g, w)) in mont.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{}: h_mont mismatch at index {i}", fx.name);
        }

        // `to_host_std` validates on the way out that every limb array is a canonical
        // residue below r, so a broken `fr_from_mont` cannot hide behind a silent
        // reduction on the host.
        let std_form = handle
            .to_host_std()
            .unwrap_or_else(|e| panic!("{}: reading h_std back: {e}", fx.name))
            .unwrap_or_else(|| panic!("{}: h_std holds a value that is not below r", fx.name));
        assert_eq!(std_form.len(), want.len(), "{}: h_std length", fx.name);
        for (i, (g, w)) in std_form.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{}: h_std mismatch at index {i}", fx.name);
        }
        eprintln!("  {} coefficients agree in both encodings", want.len());
    });
}

/// Stages 5 to 9 against the CPU backend, point by point, on the *same* `H`.
///
/// The second discriminator. Both backends are handed the CPU's host `H`, so a mismatch
/// here is in the MSM and nowhere else: the window decomposition, the counting sort, the
/// bucket accumulation or the Horner tail. Feeding the CUDA backend a host `HPoly` also
/// exercises the upload path that a mixed-backend cross-check uses, which the proving path
/// never takes.
#[test]
fn cuda_msms_match_the_cpu_backend() {
    for_each("cuda_msms_match_the_cpu_backend", |fx| {
        let mut t = StageTimings::default();
        let h = fx
            .cpu
            .compute_h(&fx.witness, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu compute_h: {e}", fx.name));

        let want = fx
            .cpu
            .msms(&fx.witness, &h, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu msms: {e}", fx.name));
        let got = fx
            .cuda
            .msms(&fx.witness, &h, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda msms: {e}", fx.name));

        assert_eq!(got.a_g1, want.a_g1, "{}: stage 5, A in G1", fx.name);
        assert_eq!(got.b_g2, want.b_g2, "{}: stage 6, B in G2", fx.name);
        assert_eq!(got.b_g1, want.b_g1, "{}: stage 7, B in G1", fx.name);
        assert_eq!(got.l_g1, want.l_g1, "{}: stage 8, L in G1", fx.name);
        assert_eq!(got.h_g1, want.h_g1, "{}: stage 9, H in G1", fx.name);
    });
}

/// And the same five points when `H` stays on the device, which is the path the proving
/// path actually takes. Separate from the test above so that a failure here with that one
/// passing points straight at the handoff (`h_std` versus `h_mont`, or the stream ordering
/// between stage 4's write and stage 9's read) rather than at the MSM.
#[test]
fn the_resident_h_gives_the_same_msms_as_a_host_one() {
    for_each("the_resident_h_gives_the_same_msms_as_a_host_one", |fx| {
        let mut t = StageTimings::default();
        let host_h = fx
            .cpu
            .compute_h(&fx.witness, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu compute_h: {e}", fx.name));
        let want = fx
            .cuda
            .msms(&fx.witness, &host_h, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda msms from host h: {e}", fx.name));

        let device_h = fx
            .cuda
            .compute_h(&fx.witness, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda compute_h: {e}", fx.name));
        let got = fx
            .cuda
            .msms(&fx.witness, &device_h, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda msms from device h: {e}", fx.name));

        assert_eq!(got.a_g1, want.a_g1, "{}: stage 5", fx.name);
        assert_eq!(got.b_g2, want.b_g2, "{}: stage 6", fx.name);
        assert_eq!(got.b_g1, want.b_g1, "{}: stage 7", fx.name);
        assert_eq!(got.l_g1, want.l_g1, "{}: stage 8", fx.name);
        assert_eq!(got.h_g1, want.h_g1, "{}: stage 9", fx.name);
    });
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

/// A short witness must be refused before anything is dispatched, and refusing it must not
/// leave a scratch buffer checked out of the stage pool.
#[test]
fn a_short_witness_is_rejected_before_any_dispatch() {
    for_each("a_short_witness_is_rejected_before_any_dispatch", |fx| {
        let mut t = StageTimings::default();
        let short = &fx.witness[..fx.witness.len() - 1];
        assert!(
            matches!(
                fx.cuda.compute_h(short, &mut t),
                Err(ProveError::WitnessLength { .. })
            ),
            "{}: compute_h accepted a short witness",
            fx.name
        );
        let h = fx.cuda.compute_h(&fx.witness, &mut t).unwrap();
        assert!(
            matches!(
                fx.cuda.msms(short, &h, &mut t),
                Err(ProveError::WitnessLength { .. })
            ),
            "{}: msms accepted a short witness",
            fx.name
        );
        // And the circuit still works afterwards, which is what says the rejection did not
        // strand a buffer in the pool.
        fx.cuda.msms(&fx.witness, &h, &mut t).unwrap();
    });
}

/// A device handle carrying another backend's tag must not be dereferenced as an
/// `HHandle`, and there is no host copy behind it, so the only correct answer is an error.
/// An `HPoly::Host` of the wrong length is a different mistake and must produce a
/// different, equally loud error rather than a proof that silently drops H's tail.
#[test]
fn an_h_from_another_backend_is_refused() {
    for_each("an_h_from_another_backend_is_refused", |fx| {
        let mut t = StageTimings::default();
        let foreign = HPoly::Device {
            tag: "metal",
            len: fx.cuda.domain_size(),
            data: std::sync::Arc::new(0u32),
        };
        assert!(
            matches!(
                fx.cuda.msms(&fx.witness, &foreign, &mut t),
                Err(ProveError::Backend { .. })
            ),
            "{}: msms accepted a foreign device handle",
            fx.name
        );
        assert!(
            matches!(
                fx.cuda
                    .msms(&fx.witness, &HPoly::Host(vec![Fr::from(1u64); 3]), &mut t),
                Err(ProveError::Backend { .. })
            ),
            "{}: msms accepted an H of the wrong length",
            fx.name
        );
    });
}
