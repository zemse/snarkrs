//! Adversarial validation of the CUDA backend on a real device.
//!
//! `tests/pipeline_gpu.rs` asks "does this produce a proof our verifier accepts, and does
//! it agree with the CPU backend on the checked-in artifacts". That is the right first
//! question and it is not the last one. Everything here exists because a Groth16 prover
//! has failure modes that answer "yes" to that question and are still wrong:
//!
//! * **Uninitialised device memory.** `cudaMalloc` does not zero. A bucket array read
//!   before it is written is the previous tenant's bytes, and on a *fresh* context those
//!   bytes are very often already zero, which is the correct value. So the bug hides
//!   until the driver's pool starts recycling blocks, at which point the same witness
//!   starts producing a proof that does not verify, non-deterministically, with no crash
//!   and nothing in a log. [`dirty_device_memory_changes_no_proof`] fills the pool with
//!   `0xFF` and frees it before proving, which is the cheapest way to make that class of
//!   bug reproducible instead of occasional.
//! * **Points at infinity in the base vectors.** snarkjs zkeys really do contain them, the
//!   packed layout encodes them as all-zero coordinates, and the existing on-device MSM
//!   tests build their bases from multiples of the generator, so none of them is ever the
//!   identity. [`infinity_bases_are_treated_as_the_identity`] puts them in deliberately,
//!   in both groups, including the case where *every* base is infinite.
//! * **The top of the scalar field.** The signed recoding borrows `2^c` out of a window
//!   and pays it back through the next window's carry bit, and the argument that the top
//!   window never carries out rests on `r < 2^254`. Scalars near `r - 1` are the ones that
//!   exercise it, and no artifact witness contains any.
//! * **Determinism.** The proof is a deterministic function of the witness and the
//!   blinders. Atomics make the *order* of bucket additions arbitrary, which is fine for
//!   the group law but is exactly the shape of thing that turns into a flaky answer if a
//!   race is real rather than benign.
//!
//! Every test skips loudly with no device and no artifacts, same as the pipeline suite,
//! and shares one backend for the same reason: the compile and the base uploads are
//! process-lifetime costs.

#![cfg(feature = "cuda")]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use snarkrs_cuda::msm::{CudaMsm, Job, JobG1, JobG2, Work};
use snarkrs_cuda::{Cuda, CudaBackend};
use snarkrs_field::{
    CurveGroup, Field, Fr, G1Affine, G1Projective, G2Affine, G2Projective, One, PrimeField,
    PrimeGroup, Zero,
};
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::prove::prove_with_blinders;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, PreparedCircuit, Proof, StageTimings};

/// NVIDIA gate for the source port. An explicitly requested run must not silently skip.
#[test]
#[ignore = "requires NVIDIA NVRTC/PTX and device arithmetic validation"]
fn constant_work_matches_cpu_and_variable_for_both_groups() {
    use snarkrs_cuda::as_words;
    use snarkrs_cuda::msm::Scalars;
    use snarkrs_gpu_layout::{PackedFr, PackedScalar};
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cuda = Cuda::new(0).expect("NVIDIA gate requires a usable device");
    let variable =
        CudaMsm::from_module(&cuda, CudaMsm::compile(&cuda).expect("NVRTC/PTX")).unwrap();
    let module = CudaMsm::compile_with(&cuda, Work::Constant).expect("NVRTC/PTX compilation");
    let constant = CudaMsm::from_module(&cuda, module).unwrap().constant_work();
    assert!(dirty_the_pool(&cuda) > 0);
    let mut seed = 0x921bu64;
    for n in [0, 1, 63, 64, 65, 4097] {
        let g1 = G1Projective::generator().into_affine();
        let g2 = G2Projective::generator().into_affine();
        let b1: Vec<_> = (0..n + 3)
            .map(|i| match i % 5 {
                0 => G1Affine::identity(),
                1 => -g1,
                _ => g1,
            })
            .collect();
        let b2: Vec<_> = (0..n + 3)
            .map(|i| match i % 5 {
                0 => G2Affine::identity(),
                1 => -g2,
                _ => g2,
            })
            .collect();
        let db1 = constant.upload_g1_bases(&b1).unwrap();
        let db2 = constant.upload_g2_bases(&b2).unwrap();
        let patterns = [
            vec![Fr::zero(); n],
            vec![Fr::one(); n],
            (0..n).map(|i| Fr::from((i % 2) as u64)).collect(),
            (0..n)
                .map(|i| if i % 64 == 0 { -Fr::one() } else { Fr::zero() })
                .collect(),
            (0..n).map(|_| rand_fr(&mut seed)).collect(),
            vec![-Fr::one(); n],
        ];
        for values in patterns {
            let want1 = naive_g1(&b1[3..], &values);
            let want2 = naive_g2(&b2[3..], &values);
            let mut padded = vec![Fr::from(19u64), Fr::from(23u64)];
            padded.extend_from_slice(&values);
            let classified = variable.upload_scalars(&padded).unwrap();
            let unclassified = constant.upload_scalars(&padded).unwrap();
            let std_words = PackedScalar::pack_slice(&padded);
            let std = cuda.stream().clone_htod(as_words(&std_words)).unwrap();
            let mont_words = PackedFr::pack_slice(&padded);
            let mont = cuda.stream().clone_htod(as_words(&mont_words)).unwrap();
            cuda.stream().synchronize().unwrap();
            let converted = constant
                .scalars_from_device_mont(&mont, padded.len())
                .unwrap();
            let stale_prefix = vec![0; padded.len() + 1];
            for scalars in [
                classified.as_scalars(),
                unclassified.as_scalars(),
                Scalars::device_std(&std, padded.len()).unwrap(),
                converted.as_scalars(),
                Scalars::device_std_with_prefix(&std, padded.len(), &stale_prefix).unwrap(),
            ] {
                let half = n / 2;
                // The first three jobs share a plan across groups. The suffix is a
                // distinct offset/length plan over the very same scalar handle.
                let jobs = [
                    Job::G1(JobG1 {
                        bases: &db1,
                        base_off: 3,
                        scalars,
                        scalar_off: 2,
                        n,
                    }),
                    Job::G2(JobG2 {
                        bases: &db2,
                        base_off: 3,
                        scalars,
                        scalar_off: 2,
                        n,
                    }),
                    Job::G1(JobG1 {
                        bases: &db1,
                        base_off: 3,
                        scalars,
                        scalar_off: 2,
                        n,
                    }),
                    Job::G2(JobG2 {
                        bases: &db2,
                        base_off: 3 + half,
                        scalars,
                        scalar_off: 2 + half,
                        n: n - half,
                    }),
                ];
                let got = constant.msm_batch(&jobs).unwrap();
                assert_eq!(got[0].g1().unwrap(), want1);
                assert_eq!(got[1].g2().unwrap(), want2);
                assert_eq!(got[2].g1().unwrap(), want1);
                assert_eq!(
                    got[3].g2().unwrap(),
                    naive_g2(&b2[3 + half..], &values[half..])
                );
            }
            let scalars = classified.as_scalars();
            assert_eq!(
                variable
                    .msm(Job::G1(JobG1 {
                        bases: &db1,
                        base_off: 3,
                        scalars,
                        scalar_off: 2,
                        n
                    }))
                    .unwrap()
                    .g1()
                    .unwrap(),
                want1
            );
            assert_eq!(
                variable
                    .msm(Job::G2(JobG2 {
                        bases: &db2,
                        base_off: 3,
                        scalars,
                        scalar_off: 2,
                        n
                    }))
                    .unwrap()
                    .g2()
                    .unwrap(),
                want2
            );
        }
    }
    for scalars in [vec![Fr::one(); 65], vec![-Fr::one(); 65]] {
        assert_eq!(
            run_g1(&constant, &vec![G1Affine::identity(); 65], &scalars),
            G1Projective::zero()
        );
        assert_eq!(
            run_g2(&constant, &vec![G2Affine::identity(); 65], &scalars),
            G2Projective::zero()
        );
    }
    let g1 = G1Projective::generator().into_affine();
    let g2 = G2Projective::generator().into_affine();
    assert_eq!(
        run_g1(&constant, &[g1, -g1], &[Fr::one(); 2]),
        G1Projective::zero()
    );
    assert_eq!(
        run_g2(&constant, &[g2, -g2], &[Fr::one(); 2]),
        G2Projective::zero()
    );
}

/// Seven points cycled, every fifth negated and every eleventh at infinity, so one
/// bucket meets the same base twice (a doubling), a base and its negation (a
/// cancellation) and the identity: the cases `pt_madd` had shortcuts for and the
/// complete formulas must get right without them.
fn colliding_bases(n: usize) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let p1: Vec<G1Affine> = (1..=7u64)
        .map(|k| (G1Projective::generator() * Fr::from(k)).into_affine())
        .collect();
    let p2: Vec<G2Affine> = (1..=7u64)
        .map(|k| (G2Projective::generator() * Fr::from(k)).into_affine())
        .collect();
    let g1 = (0..n)
        .map(|i| match (i % 11, i % 5) {
            (4, _) => G1Affine::identity(),
            (_, 2) => -p1[i % 7],
            _ => p1[i % 7],
        })
        .collect();
    let g2 = (0..n)
        .map(|i| match (i % 11, i % 5) {
            (4, _) => G2Affine::identity(),
            (_, 2) => -p2[i % 7],
            _ => p2[i % 7],
        })
        .collect();
    (g1, g2)
}

/// Scalar vectors chosen for where their digits land rather than how many are zero:
/// one dense value everywhere puts a window's every entry in one bucket (one run the
/// length of the region, through every slice's head and tail and the whole fold); a
/// value and its negation, or two values, alternate between two buckets; single bits
/// give each scalar one live window; small values live in window zero and send
/// everything else to the dummy rows.
fn adversarial_scalars(n: usize) -> Vec<(&'static str, Vec<Fr>)> {
    let a = -Fr::from(0x7f3a_19c5_d02b_6e41u64);
    let b = Fr::from(0x1234_5678_9abc_def1u64) * Fr::from(0x0fed_cba9_8765_4321u64);
    vec![
        ("one value", vec![a; n]),
        (
            "a value and its negation",
            (0..n).map(|i| if i % 2 == 0 { a } else { -a }).collect(),
        ),
        (
            "two values",
            (0..n).map(|i| if i % 3 == 0 { a } else { b }).collect(),
        ),
        (
            "single bits",
            (0..n)
                .map(|i| Fr::from(2u64).pow([(i % 254) as u64]))
                .collect(),
        ),
        ("small", (0..n).map(|i| Fr::from((i % 9) as u64)).collect()),
        ("zeros", vec![Fr::zero(); n]),
        ("ones", vec![Fr::one(); n]),
    ]
}

/// Constant work against the CPU MSM on the bucket collisions above, in both groups, at
/// sizes from one short slice to many windows of many slices, plus an offset job at each
/// size the way the L MSM reads the witness. The CUDA twin of the Metal test of the same
/// name; the two groups share one plan, as the A and B MSMs do in a proof.
#[test]
fn constant_work_matches_the_cpu_on_bucket_collisions() {
    use snarkrs_msm::MsmBackend;
    on_device(
        "constant_work_matches_the_cpu_on_bucket_collisions",
        |cuda| {
            let module = CudaMsm::compile_with(cuda, Work::Constant).expect("compile the MSM unit");
            let m = CudaMsm::from_module(cuda, module)
                .expect("bind the MSM kernels")
                .constant_work();
            let cpu = snarkrs_msm::CpuMsm::new();
            for n in [3usize, 70, 1500, 20_011] {
                let (g1, g2) = colliding_bases(n);
                let b1 = m.upload_g1_bases(&g1).expect("upload G1 bases");
                let b2 = m.upload_g2_bases(&g2).expect("upload G2 bases");
                let off = n.min(5);
                for (label, scalars) in adversarial_scalars(n) {
                    let s = m.upload_scalars(&scalars).expect("upload scalars");
                    let got = m
                        .msm_batch(&[
                            Job::G1(JobG1 {
                                bases: &b1,
                                base_off: 0,
                                scalars: s.as_scalars(),
                                scalar_off: 0,
                                n,
                            }),
                            Job::G2(JobG2 {
                                bases: &b2,
                                base_off: 0,
                                scalars: s.as_scalars(),
                                scalar_off: 0,
                                n,
                            }),
                        ])
                        .unwrap_or_else(|e| panic!("{label}, n = {n}: {e}"));
                    assert_eq!(
                        got[0].g1().expect("G1 result").into_affine(),
                        cpu.msm_g1(&g1, &scalars).into_affine(),
                        "{label}, n = {n}: G1"
                    );
                    assert_eq!(
                        got[1].g2().expect("G2 result").into_affine(),
                        cpu.msm_g2(&g2, &scalars).into_affine(),
                        "{label}, n = {n}: G2"
                    );
                    if off == n {
                        continue;
                    }
                    let got = m
                        .msm(Job::G1(JobG1 {
                            bases: &b1,
                            base_off: off,
                            scalars: s.as_scalars(),
                            scalar_off: off,
                            n: n - off,
                        }))
                        .unwrap_or_else(|e| panic!("{label}, n = {n}, offset {off}: {e}"));
                    assert_eq!(
                        got.g1().expect("G1 result").into_affine(),
                        cpu.msm_g1(&g1[off..], &scalars[off..]).into_affine(),
                        "{label}, n = {n}: G1 over an offset range"
                    );
                }
            }
        },
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One artifact on both backends. Unlike the pipeline suite's fixture this keeps the
/// `CudaBackend` alive, because [`dirty_device_memory_changes_no_proof`] needs the same
/// context and the same stream-ordered allocator the proofs run on. Allocating out of a
/// second context would dirty a different pool and the test would prove nothing.
struct World {
    backend: CudaBackend,
    fixtures: Vec<Fixture>,
}

struct Fixture {
    name: String,
    cuda: Box<dyn PreparedCircuit>,
    cpu: Box<dyn PreparedCircuit>,
    witness: Vec<Fr>,
    vk: VerifyingKey,
}

impl Fixture {
    fn public(&self) -> Vec<Fr> {
        self.witness[1..=self.cuda.n_public()].to_vec()
    }

    fn prove(&self, r: u64, s: u64) -> Proof {
        let mut t = StageTimings::default();
        prove_with_blinders(
            self.cuda.as_ref(),
            &self.witness,
            Fr::from(r),
            Fr::from(s),
            &mut t,
        )
        .unwrap_or_else(|e| panic!("{}: cuda prove: {e}", self.name))
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
    out.sort_by_key(|(_, d)| {
        std::fs::metadata(d.join("circuit.zkey"))
            .map(|m| m.len())
            .unwrap_or(u64::MAX)
    });
    out
}

fn world() -> Option<&'static World> {
    static ONCE: OnceLock<Option<World>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("SKIPPING every adversarial cuda test, no usable device: {e}");
                return None;
            }
        };
        let dirs = artifact_dirs();
        if dirs.is_empty() {
            eprintln!("SKIPPING every adversarial cuda test: no artifacts under bench/artifacts");
            return None;
        }
        let cpu = CpuBackend::new();
        let mut fixtures = Vec::new();
        for (name, dir) in dirs {
            let pk = ProvingKey::load(&dir.join("circuit.zkey"))
                .unwrap_or_else(|e| panic!("{name}: loading circuit.zkey: {e}"));
            let witness = Witness::load(&dir.join("circuit.wtns"))
                .unwrap_or_else(|e| panic!("{name}: loading circuit.wtns: {e}"))
                .0;
            let vk = VerifyingKey::from_json(&dir.join("vkey.json"))
                .unwrap_or_else(|e| panic!("{name}: loading vkey.json: {e}"));
            let cuda = backend
                .prepare(pk)
                .unwrap_or_else(|e| panic!("{name}: cuda prepare: {e}"));
            let cpu = cpu
                .prepare(
                    ProvingKey::load(&dir.join("circuit.zkey"))
                        .unwrap_or_else(|e| panic!("{name}: reloading circuit.zkey: {e}")),
                )
                .unwrap_or_else(|e| panic!("{name}: cpu prepare: {e}"));
            fixtures.push(Fixture {
                name,
                cuda,
                cpu,
                witness,
                vk,
            });
        }
        Some(World { backend, fixtures })
    })
    .as_ref()
}

/// One test at a time on the device, whatever `--test-threads` says. A test that
/// panicked poisoned this, and that is exactly when the remaining tests still need to
/// run, so the poison is stepped over rather than propagated. Both entry points below
/// take the *same* lock: `on_device` opens its own context and compiles its own copy of
/// the MSM unit, and doing that while a 2^18 proof holds several hundred megabytes of
/// bucket array would read as a backend bug when it is an out-of-memory.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn for_each(test: &str, f: impl Fn(&World, &Fixture)) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(w) = world() else {
        resources::skip(format_args!(
            "{test}: no usable device or no artifacts, see the line above"
        ));
        return;
    };
    for fx in &w.fixtures {
        eprintln!("{test}: {}", fx.name);
        f(w, fx);
    }
}

/// A device-only test that wants no artifacts. Shares the same lock so it cannot run
/// alongside a 2^18 proof.
fn on_device(test: &str, f: impl Fn(&Cuda)) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    match Cuda::new(0) {
        Ok(cuda) => f(&cuda),
        Err(e) => resources::skip(format_args!("{test}, no usable device: {e}")),
    }
}

fn same_proof(what: &str, name: &str, a: &Proof, b: &Proof) {
    assert_eq!(a.a, b.a, "{name}: {what}: pi_a differs");
    assert_eq!(a.b, b.b, "{name}: {what}: pi_b differs");
    assert_eq!(a.c, b.c, "{name}: {what}: pi_c differs");
}

// ---------------------------------------------------------------------------
// 1. The two backends must compute the same function, not merely two valid proofs
// ---------------------------------------------------------------------------

/// Same witness, same `r`, same `s`, both backends, compared as three curve points.
///
/// The pipeline suite compares `H` and the five MSM outputs, which is a stronger
/// statement about the *stages*. It is a weaker statement about the *backend*, because
/// nothing there runs stage 11 on the GPU's own MSM outputs. A proof is malleable: there
/// are about `r` valid encodings of every provable statement, so "both verify" is not
/// evidence that the two backends agree. Bit equality is.
#[test]
fn cuda_and_cpu_produce_the_same_proof() {
    for_each("cuda_and_cpu_produce_the_same_proof", |_, fx| {
        let mut t = StageTimings::default();
        let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
        let want = prove_with_blinders(fx.cpu.as_ref(), &fx.witness, r, s, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu prove: {e}", fx.name));
        let got = prove_with_blinders(fx.cuda.as_ref(), &fx.witness, r, s, &mut t)
            .unwrap_or_else(|e| panic!("{}: cuda prove: {e}", fx.name));
        same_proof("cuda vs cpu", &fx.name, &got, &want);
        verify(&fx.vk, &fx.public(), &got).unwrap_or_else(|e| panic!("{}: verify: {e}", fx.name));
    });
}

// ---------------------------------------------------------------------------
// 2. Uninitialised memory
// ---------------------------------------------------------------------------

/// Fills the driver's stream-ordered pool with `0xFF` and frees it, so the next
/// allocation of that size is served a block full of ones rather than a fresh zero page.
///
/// Returns the bytes actually dirtied. Stops at the first allocation failure rather than
/// erroring: how much of a 15 GB card is free depends on what else is resident, and the
/// test wants "as much as fits", not a fixed number.
fn dirty_the_pool(cuda: &Cuda) -> usize {
    const CHUNK: usize = 64 << 20; // words, so 256 MB a chunk
    let stream = cuda.stream();
    let ones = vec![0xFFFF_FFFFu32; CHUNK];
    let mut held = Vec::new();
    let mut bytes = 0usize;
    // 24 chunks is 6 GB, comfortably more than the largest artifact's scratch and well
    // under the card, so the loop normally runs to completion rather than to an OOM.
    for _ in 0..24 {
        match stream.clone_htod(&ones) {
            Ok(buf) => {
                bytes += CHUNK * 4;
                held.push(buf);
            }
            Err(_) => break,
        }
    }
    stream.synchronize().expect("synchronize dirty fill");
    // Freed here. `cuMemFreeAsync` returns the blocks to the driver's pool without
    // handing them back to the OS, which is precisely the state that makes a missing
    // zeroing observable: the very next `alloc` gets these bytes.
    drop(held);
    bytes
}

/// The highest-risk defect in this port, made reproducible.
///
/// Order matters: prove once on a fresh context to get the reference, then dirty, then
/// prove again. If any accumulator, bucket array, counter or spill slot is read before it
/// is written, the second proof differs from the first and from the CPU's. On a fresh
/// context alone the bug would be invisible, because untouched device pages read as zero
/// and zero happens to be the identity in this encoding and the right start for a counter.
#[test]
fn dirty_device_memory_changes_no_proof() {
    for_each("dirty_device_memory_changes_no_proof", |w, fx| {
        let mut t = StageTimings::default();
        let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
        let want = prove_with_blinders(fx.cpu.as_ref(), &fx.witness, r, s, &mut t)
            .unwrap_or_else(|e| panic!("{}: cpu prove: {e}", fx.name));
        let clean = fx.prove(31337, 4242);
        same_proof("clean cuda vs cpu", &fx.name, &clean, &want);

        // Three rounds, because the pool's free lists differ after a proof has run and
        // returned its own blocks: the first round dirties a pool shaped by the clean
        // proof, later ones a pool shaped by a dirty one.
        for round in 0..3 {
            let bytes = dirty_the_pool(w.backend.cuda());
            let got = fx.prove(31337, 4242);
            same_proof(
                &format!("round {round} after dirtying {} MB", bytes >> 20),
                &fx.name,
                &got,
                &want,
            );
            verify(&fx.vk, &fx.public(), &got)
                .unwrap_or_else(|e| panic!("{}: verify after dirtying: {e}", fx.name));
        }
    });
}

// ---------------------------------------------------------------------------
// 3. Determinism
// ---------------------------------------------------------------------------

/// The same witness and blinders must give a bit-identical proof, every time, in one
/// process.
///
/// Atomics make the order in which points land in a bucket arbitrary from run to run.
/// Addition in a group is associative and commutative so that is benign, and this test is
/// what says so rather than assuming it. It is also the test that catches a genuinely
/// racy write: a benign reordering and a real race look identical in a single run.
#[test]
fn repeated_proofs_are_bit_identical() {
    for_each("repeated_proofs_are_bit_identical", |_, fx| {
        let first = fx.prove(31337, 4242);
        for i in 1..16 {
            let again = fx.prove(31337, 4242);
            same_proof(&format!("repetition {i}"), &fx.name, &again, &first);
        }
        // And a different `(r, s)` must actually move the proof, otherwise the loop above
        // would pass just as well on a backend that cached its answer.
        let other = fx.prove(31338, 4242);
        assert_ne!(other.a, first.a, "{}: r did not affect pi_a", fx.name);
    });
}

// ---------------------------------------------------------------------------
// 4. Points at infinity and degenerate scalars
// ---------------------------------------------------------------------------

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

fn naive_g1(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    bases
        .iter()
        .zip(scalars)
        .fold(G1Projective::zero(), |a, (b, s)| a + *b * s)
}

fn naive_g2(bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
    bases
        .iter()
        .zip(scalars)
        .fold(G2Projective::zero(), |a, (b, s)| a + *b * s)
}

fn run_g1(m: &CudaMsm, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    let db = m.upload_g1_bases(bases).expect("upload G1 bases");
    let ds = m.upload_scalars(scalars).expect("upload scalars");
    m.msm(Job::G1(JobG1 {
        bases: &db,
        base_off: 0,
        scalars: ds.as_scalars(),
        scalar_off: 0,
        n: bases.len(),
    }))
    .expect("G1 msm")
    .g1()
    .expect("G1 result")
}

fn run_g2(m: &CudaMsm, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
    let db = m.upload_g2_bases(bases).expect("upload G2 bases");
    let ds = m.upload_scalars(scalars).expect("upload scalars");
    m.msm(Job::G2(JobG2 {
        bases: &db,
        base_off: 0,
        scalars: ds.as_scalars(),
        scalar_off: 0,
        n: bases.len(),
    }))
    .expect("G2 msm")
    .g2()
    .expect("G2 result")
}

/// Infinity in the base vector, in every position that matters, in both groups.
///
/// snarkjs zkeys contain points at infinity in the query vectors, the packed layout
/// encodes them as all-zero coordinates, and `pt_madd` has an `aff_is_inf` branch for
/// them. Nothing else in this workspace's on-device tests ever builds an identity base:
/// they are all multiples of the generator. So this is the only place the branch is
/// exercised on a device, and it is exercised against a general scalar (the bucket path),
/// a one (the `msm_ones_*` path) and a zero (dropped entirely), because those are three
/// different kernels.
#[test]
fn infinity_bases_are_treated_as_the_identity() {
    on_device("infinity_bases_are_treated_as_the_identity", |cuda| {
        let m = CudaMsm::without_key(cuda).expect("compile the MSM unit");
        let mut seed = 0xF00D_BEEF_1234_5678u64;

        for n in [1usize, 2, 5, 64, 65, 1000] {
            let mut bases = Vec::with_capacity(n);
            let mut scalars = Vec::with_capacity(n);
            let mut cur = G1Projective::generator();
            for i in 0..n {
                cur += G1Projective::generator();
                // Every fifth base is infinity, and index 0 always is, so the very first
                // entry of the vector is the degenerate one.
                bases.push(if i % 5 == 0 {
                    G1Affine::identity()
                } else {
                    cur.into_affine()
                });
                scalars.push(match i % 3 {
                    0 => rand_fr(&mut seed),
                    1 => Fr::one(),
                    _ => Fr::zero(),
                });
            }
            assert_eq!(
                run_g1(&m, &bases, &scalars).into_affine(),
                naive_g1(&bases, &scalars).into_affine(),
                "G1 with infinity bases, n = {n}"
            );

            // And the pathological case: every base is infinity, so the answer is the
            // identity and every kernel has to produce it rather than a stray zz.
            let all_inf = vec![G1Affine::identity(); n];
            assert!(
                run_g1(&m, &all_inf, &scalars).is_zero(),
                "G1 with every base at infinity did not give the identity, n = {n}"
            );
        }

        // G2 is a different field and different kernels, so it gets the same treatment.
        for n in [1usize, 3, 200] {
            let mut bases = Vec::with_capacity(n);
            let mut scalars = Vec::with_capacity(n);
            let mut cur = G2Projective::generator();
            for i in 0..n {
                cur += G2Projective::generator();
                bases.push(if i % 4 == 0 {
                    G2Affine::identity()
                } else {
                    cur.into_affine()
                });
                scalars.push(match i % 3 {
                    0 => rand_fr(&mut seed),
                    1 => Fr::one(),
                    _ => Fr::zero(),
                });
            }
            assert_eq!(
                run_g2(&m, &bases, &scalars).into_affine(),
                naive_g2(&bases, &scalars).into_affine(),
                "G2 with infinity bases, n = {n}"
            );
            let all_inf = vec![G2Affine::identity(); n];
            assert!(
                run_g2(&m, &all_inf, &scalars).is_zero(),
                "G2 with every base at infinity did not give the identity, n = {n}"
            );
        }
    });
}

/// Scalar vectors that are entirely one class.
///
/// `msm_count` and `msm_scatter` drop every 0 and every 1, and `msm_ones_*` picks the ones
/// back up. An all-zero vector therefore reaches the bucket stages with nothing at all in
/// them, and an all-one vector reaches them with an empty entry array while the whole
/// answer comes out of the ones kernel. Both are inputs a real circuit produces: the `L`
/// MSM of a circuit whose private witness is all zeros is the first one, and the `A` MSM
/// of a witness of flags is close to the second.
#[test]
fn degenerate_scalar_vectors_still_agree() {
    on_device("degenerate_scalar_vectors_still_agree", |cuda| {
        let m = CudaMsm::without_key(cuda).expect("compile the MSM unit");
        for n in [1usize, 2, 63, 64, 65, 4097] {
            let mut bases = Vec::with_capacity(n);
            let mut cur = G1Projective::generator();
            for _ in 0..n {
                cur += G1Projective::generator();
                bases.push(cur.into_affine());
            }
            for (what, scalars) in [
                ("all zero", vec![Fr::zero(); n]),
                ("all one", vec![Fr::one(); n]),
                // r - 1, the largest scalar there is. Its top window is the one the
                // "no carry out of the top" argument is about.
                ("all r-1", vec![-Fr::one(); n]),
                // 2^253, which sets the highest bit any canonical residue can set and so
                // puts a digit in the top window of every width.
                ("all 2^253", vec![Fr::from(2u64).pow([253u64]); n]),
            ] {
                assert_eq!(
                    run_g1(&m, &bases, &scalars).into_affine(),
                    naive_g1(&bases, &scalars).into_affine(),
                    "G1 {what}, n = {n}"
                );
            }
        }
    });
}

/// Scalars at and around the top of the field, mixed rather than uniform.
///
/// The signed recoding borrows `2^c` from a window whose raw value is in the top half and
/// pays it back through the next window's carry bit. The claim that the *top* window never
/// carries out rests on every scalar being below `2^254`, which is true for canonical
/// residues and is exactly what a scalar near `r - 1` tests. If the recoding were wrong
/// there, the artifacts would not show it: a witness is mostly small integers and zeros.
#[test]
fn scalars_at_the_top_of_the_field_agree() {
    on_device("scalars_at_the_top_of_the_field_agree", |cuda| {
        let m = CudaMsm::without_key(cuda).expect("compile the MSM unit");
        let mut seed = 0x5EED_0BAD_C0DE_1111u64;
        let edge = [
            -Fr::one(),
            -Fr::from(2u64),
            -Fr::from(3u64),
            Fr::from(2u64).pow([253u64]),
            Fr::from(2u64).pow([253u64]) - Fr::one(),
            Fr::from(2u64).pow([252u64]),
            Fr::from(2u64).pow([128u64]),
            Fr::from(2u64).pow([128u64]) - Fr::one(),
            Fr::from(u64::MAX),
            Fr::from(2u64),
            Fr::one(),
            Fr::zero(),
        ];
        let n = 3000usize;
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut cur = G1Projective::generator();
        for i in 0..n {
            cur += G1Projective::generator();
            bases.push(cur.into_affine());
            // Mostly edge values so the fat buckets are the interesting ones, with random
            // scalars mixed in so the bucket occupancy is not uniform either.
            scalars.push(if i % 7 == 0 {
                rand_fr(&mut seed)
            } else {
                edge[i % edge.len()]
            });
        }
        assert_eq!(
            run_g1(&m, &bases, &scalars).into_affine(),
            naive_g1(&bases, &scalars).into_affine(),
            "G1 with field-edge scalars"
        );
    });
}

/// `base_off` and `scalar_off` are how the `L` MSM reads the private suffix of the witness
/// against its own base vector without a second upload. An off-by-one there is a proof
/// that fails to verify and nothing else, and no test in this workspace varies them.
#[test]
fn offsets_select_the_right_sub_msm() {
    on_device("offsets_select_the_right_sub_msm", |cuda| {
        let m = CudaMsm::without_key(cuda).expect("compile the MSM unit");
        let mut seed = 0x0FF5_E700_1234_9999u64;
        let total = 777usize;
        let mut bases = Vec::with_capacity(total);
        let mut scalars = Vec::with_capacity(total);
        let mut cur = G1Projective::generator();
        for i in 0..total {
            cur += G1Projective::generator();
            bases.push(cur.into_affine());
            scalars.push(match i % 4 {
                0 => Fr::zero(),
                1 => Fr::one(),
                _ => rand_fr(&mut seed),
            });
        }
        let db = m.upload_g1_bases(&bases).expect("upload bases");
        let ds = m.upload_scalars(&scalars).expect("upload scalars");

        for (base_off, scalar_off, n) in [
            (0usize, 0usize, total),
            (1, 1, total - 1),
            (0, 5, total - 5),
            (5, 0, total - 5),
            (13, 130, 400),
            (total - 1, total - 1, 1),
            (total, total, 0),
        ] {
            let got = m
                .msm(Job::G1(JobG1 {
                    bases: &db,
                    base_off,
                    scalars: ds.as_scalars(),
                    scalar_off,
                    n,
                }))
                .unwrap_or_else(|e| panic!("msm at ({base_off}, {scalar_off}, {n}): {e}"))
                .g1()
                .expect("G1 result");
            let want = naive_g1(
                &bases[base_off..base_off + n],
                &scalars[scalar_off..scalar_off + n],
            );
            assert_eq!(
                got.into_affine(),
                want.into_affine(),
                "base_off {base_off}, scalar_off {scalar_off}, n {n}"
            );
        }
    });
}

/// Every window width the kernels can address, in both groups and under both schedules,
/// against a naive sum.
///
/// `window_size` picks 3..=8 and nothing else in this workspace ever ran another width, so
/// 2 and 9..=16 existed only as `G16_CUDA_MSM_C` values no test had validated. The widths
/// are not interchangeable in the kernels. `top_bits(12)` and `top_bits(14)` are 3, so at
/// those widths the top window has four live buckets, each holding a quarter of the
/// scalars, and such a bucket's run spans thousands of slices: that is the one-thread
/// `msm_merge_*` walk and, under constant work, the deepest fold. The signed digit's
/// `(1 << c) - b - carry` and its `mag <= 2^(c-1)` bound move with `c`, and at 16 a window
/// is 32768 buckets, the reduce kernel's longest serial walk. A wrong digit at any of them
/// is a proof that fails verification with nothing else to go on, so this is the gate for
/// sweeping `G16_CUDA_MSM_C` at all.
///
/// Forced through [`CudaMsm::with_window`] rather than the environment, which is
/// process-global and would race the rest of this binary. Four scalar vectors a length:
/// all zero and all one, so the bucket stages see nothing and the whole answer comes out
/// of `msm_ones_*` at every width; all `r - 1`, a digit in the top window of every width
/// and the edge of the "no carry out of the top" argument; and a mix of random, 0, 1,
/// `r - 1` and `2^253`. The references are computed once per vector and reused across
/// every width and schedule, which is most of what keeps this to minutes.
#[test]
#[ignore = "sweeps every window width in both groups and both schedules; minutes on a T4"]
fn every_window_width_agrees_with_a_naive_sum() {
    on_device("every_window_width_agrees_with_a_naive_sum", |cuda| {
        let variable_module = CudaMsm::compile(cuda).expect("compile the MSM unit");
        let constant_module =
            CudaMsm::compile_with(cuda, Work::Constant).expect("compile the MSM unit");
        let mut seed = 0x5EED_C0DE_0000_0C16u64;
        // Around the slice length, and a few thousand that are not powers of two so the
        // last slice of a window is partial and the fat top buckets at c = 12 and 14 span
        // many slices.
        for n in [1usize, 63, 64, 65, 2500, 4095, 6007] {
            let mut b1 = Vec::with_capacity(n);
            let mut b2 = Vec::with_capacity(n);
            let mut cur1 = G1Projective::generator();
            let mut cur2 = G2Projective::generator();
            for _ in 0..n {
                cur1 += G1Projective::generator();
                cur2 += G2Projective::generator();
                b1.push(cur1.into_affine());
                b2.push(cur2.into_affine());
            }
            let top = Fr::from(2u64).pow([253u64]);
            let vectors = [
                ("all zero", vec![Fr::zero(); n]),
                ("all one", vec![Fr::one(); n]),
                ("all r-1", vec![-Fr::one(); n]),
                (
                    "mixed",
                    (0..n)
                        .map(|i| match i % 5 {
                            0 => rand_fr(&mut seed),
                            1 => Fr::one(),
                            2 => Fr::zero(),
                            3 => -Fr::one(),
                            _ => top,
                        })
                        .collect(),
                ),
            ];
            let want: Vec<(G1Affine, G2Affine)> = vectors
                .iter()
                .map(|(_, s)| {
                    (
                        naive_g1(&b1, s).into_affine(),
                        naive_g2(&b2, s).into_affine(),
                    )
                })
                .collect();

            // 2..=16 is the range the plan accepts: `MAX_WINDOW` in msm.rs, checked below.
            for c in 2u32..=16 {
                for constant in [false, true] {
                    let schedule = if constant { "constant" } else { "variable" };
                    eprintln!("  n = {n}, c = {c}, {schedule} work");
                    let module = if constant {
                        &constant_module
                    } else {
                        &variable_module
                    };
                    let m = CudaMsm::from_module(cuda, module.clone())
                        .expect("bind the MSM kernels")
                        .with_window(c);
                    let m = if constant { m.constant_work() } else { m };
                    let db1 = m.upload_g1_bases(&b1).expect("upload G1 bases");
                    let db2 = m.upload_g2_bases(&b2).expect("upload G2 bases");
                    for ((what, scalars), (want1, want2)) in vectors.iter().zip(&want) {
                        let ds = m.upload_scalars(scalars).expect("upload scalars");
                        // One batch, so the two groups share the digit plan the way the
                        // A and B MSMs do in a proof.
                        let got = m
                            .msm_batch(&[
                                Job::G1(JobG1 {
                                    bases: &db1,
                                    base_off: 0,
                                    scalars: ds.as_scalars(),
                                    scalar_off: 0,
                                    n,
                                }),
                                Job::G2(JobG2 {
                                    bases: &db2,
                                    base_off: 0,
                                    scalars: ds.as_scalars(),
                                    scalar_off: 0,
                                    n,
                                }),
                            ])
                            .unwrap_or_else(|e| {
                                panic!("c = {c}, n = {n}, {what}, {schedule} work: {e}")
                            });
                        assert_eq!(
                            got[0].g1().expect("G1 result").into_affine(),
                            *want1,
                            "G1 c = {c}, n = {n}, {what}, {schedule} work"
                        );
                        assert_eq!(
                            got[1].g2().expect("G2 result").into_affine(),
                            *want2,
                            "G2 c = {c}, n = {n}, {what}, {schedule} work"
                        );
                    }
                }
            }
        }

        // The hook reaches the plan: a width the kernels cannot address is refused there,
        // rather than quietly replaced by the heuristic's pick, which is what would make
        // every assertion above vacuous.
        let g = G1Projective::generator().into_affine();
        for c in [1u32, 17] {
            let m = CudaMsm::from_module(cuda, variable_module.clone())
                .expect("bind the MSM kernels")
                .with_window(c);
            let db = m.upload_g1_bases(&[g]).expect("upload G1 base");
            let ds = m.upload_scalars(&[Fr::from(5u64)]).expect("upload scalar");
            let err = m
                .msm(Job::G1(JobG1 {
                    bases: &db,
                    base_off: 0,
                    scalars: ds.as_scalars(),
                    scalar_off: 0,
                    n: 1,
                }))
                .err()
                .unwrap_or_else(|| panic!("c = {c} was accepted"))
                .to_string();
            assert!(err.contains("window"), "c = {c}: unhelpful message: {err}");
        }
    });
}

// ---------------------------------------------------------------------------
// 5. Concurrency, with inputs that can actually tell two proofs apart
// ---------------------------------------------------------------------------

/// Several threads on one circuit, each with its *own* witness, each checked against the
/// CPU's answer for that same witness.
///
/// `pipeline_gpu.rs`'s `one_cuda_circuit_proves_concurrently` runs four threads too, but
/// all four prove the *same* witness. That cannot detect the failure it is aimed at: if
/// two in-flight proofs were handed the same scratch buffers they would write identical
/// bytes into them, so the race is benign by construction and the test passes whether or
/// not the pool works. Distinct witnesses remove that coincidence. If `take_scratch` ever
/// hands one `Scratch` to two threads, one of them reads the other's `A`, `B`, `C` or `H`
/// and this fails on the first run.
///
/// `compute_h` rather than `prove` because a witness only has to be the right length here.
/// It need not satisfy the constraint system, which is what lets every thread have a
/// different one; stages 0 to 4 are linear algebra and three transforms, and none of them
/// cares whether the witness is a solution.
#[test]
fn concurrent_proofs_with_distinct_witnesses_do_not_mix() {
    for_each(
        "concurrent_proofs_with_distinct_witnesses_do_not_mix",
        |_, fx| {
            const THREADS: usize = 8;
            let n_vars = fx.cuda.n_vars();
            let mut seed = 0xC0FF_EE00_5EED_0001u64;
            let witnesses: Vec<Vec<Fr>> = (0..THREADS)
                .map(|_| (0..n_vars).map(|_| rand_fr(&mut seed)).collect())
                .collect();

            // The oracle, computed serially on the CPU first so nothing about it can be
            // affected by what the GPU threads do.
            let want: Vec<Vec<Fr>> = witnesses
                .iter()
                .map(|w| {
                    let mut t = StageTimings::default();
                    fx.cpu
                        .compute_h(w, &mut t)
                        .unwrap_or_else(|e| panic!("{}: cpu compute_h: {e}", fx.name))
                        .to_host()
                        .expect("the cpu backend returns a host vector")
                        .to_vec()
                })
                .collect();

            std::thread::scope(|scope| {
                let handles: Vec<_> = witnesses
                    .iter()
                    .zip(&want)
                    .enumerate()
                    .map(|(i, (w, expect))| {
                        scope.spawn(move || {
                            // Twice per thread, so a scratch set that has been through the pool
                            // once is exercised as well as a freshly allocated one.
                            for round in 0..2 {
                                let mut t = StageTimings::default();
                                let h = fx
                                    .cuda
                                    .compute_h(w, &mut t)
                                    .unwrap_or_else(|e| panic!("{}: cuda compute_h: {e}", fx.name));
                                let got = h
                                    .device_handle::<snarkrs_cuda::stages::HHandle>(
                                        snarkrs_cuda::stages::TAG,
                                    )
                                    .expect("a cuda device handle")
                                    .to_host()
                                    .unwrap_or_else(|e| panic!("{}: reading H back: {e}", fx.name));
                                assert_eq!(
                                    got.len(),
                                    expect.len(),
                                    "{}: thread {i} round {round}: H length",
                                    fx.name
                                );
                                for (k, (g, e)) in got.iter().zip(expect).enumerate() {
                                    assert_eq!(
                                        g, e,
                                        "{}: thread {i} round {round}: H differs at index {k}. \
                                     Two concurrent proofs were handed the same scratch.",
                                        fx.name
                                    );
                                }
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        },
    );
}

/// The same argument for stages 5 to 9: several threads, each with its own witness and its
/// own `H`, each checked against the CPU.
///
/// The MSM lane allocates its scratch per call out of the driver's stream-ordered pool
/// rather than out of a pool of its own, so what this exercises is that the pool really
/// does hand two concurrent callers disjoint blocks, and that sharing one stream between
/// them cannot let one job's `msm_scatter` land in another job's entry array.
#[test]
fn concurrent_msms_with_distinct_inputs_do_not_mix() {
    for_each(
        "concurrent_msms_with_distinct_inputs_do_not_mix",
        |_, fx| {
            const THREADS: usize = 6;
            let n_vars = fx.cuda.n_vars();
            let domain = fx.cuda.domain_size();
            let mut seed = 0xBEEF_1234_ABCD_0002u64;
            let inputs: Vec<(Vec<Fr>, snarkrs_groth16::HPoly)> = (0..THREADS)
                .map(|_| {
                    (
                        (0..n_vars).map(|_| rand_fr(&mut seed)).collect(),
                        snarkrs_groth16::HPoly::Host(
                            (0..domain).map(|_| rand_fr(&mut seed)).collect(),
                        ),
                    )
                })
                .collect();

            let want: Vec<_> = inputs
                .iter()
                .map(|(w, h)| {
                    let mut t = StageTimings::default();
                    fx.cpu
                        .msms(w, h, &mut t)
                        .unwrap_or_else(|e| panic!("{}: cpu msms: {e}", fx.name))
                })
                .collect();

            std::thread::scope(|scope| {
                let handles: Vec<_> = inputs
                    .iter()
                    .zip(&want)
                    .enumerate()
                    .map(|(i, ((w, h), expect))| {
                        scope.spawn(move || {
                            let mut t = StageTimings::default();
                            let got = fx
                                .cuda
                                .msms(w, h, &mut t)
                                .unwrap_or_else(|e| panic!("{}: cuda msms: {e}", fx.name));
                            assert_eq!(got.a_g1, expect.a_g1, "{}: thread {i}: stage 5", fx.name);
                            assert_eq!(got.b_g2, expect.b_g2, "{}: thread {i}: stage 6", fx.name);
                            assert_eq!(got.b_g1, expect.b_g1, "{}: thread {i}: stage 7", fx.name);
                            assert_eq!(got.l_g1, expect.l_g1, "{}: thread {i}: stage 8", fx.name);
                            assert_eq!(got.h_g1, expect.h_g1, "{}: thread {i}: stage 9", fx.name);
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        },
    );
}

/// Stages 0 to 4 on a witness that is not a solution to anything, against the CPU.
///
/// Every `compute_h` check in this workspace feeds the artifact's own witness, which is a
/// vector of small integers, mostly 0 and 1, with a lot of structure. The transforms do not
/// care, but the *gather* does: it walks the CSR rows of sections 4 and 5, and a row whose
/// coefficient happens to be 1 in every fixture is a row whose multiply is never tested.
/// Uniform random field elements remove all of that structure.
#[test]
fn a_random_witness_gives_the_same_h_as_the_cpu() {
    for_each("a_random_witness_gives_the_same_h_as_the_cpu", |_, fx| {
        let mut seed = 0xDEAD_BEEF_0BAD_0003u64;
        for trial in 0..3 {
            let w: Vec<Fr> = (0..fx.cuda.n_vars()).map(|_| rand_fr(&mut seed)).collect();
            let mut t = StageTimings::default();
            let want = fx
                .cpu
                .compute_h(&w, &mut t)
                .unwrap_or_else(|e| panic!("{}: cpu compute_h: {e}", fx.name));
            let want = want.to_host().expect("host vector").to_vec();
            let got = fx
                .cuda
                .compute_h(&w, &mut t)
                .unwrap_or_else(|e| panic!("{}: cuda compute_h: {e}", fx.name));
            let got = got
                .device_handle::<snarkrs_cuda::stages::HHandle>(snarkrs_cuda::stages::TAG)
                .expect("a cuda device handle")
                .to_host()
                .unwrap_or_else(|e| panic!("{}: reading H back: {e}", fx.name));
            assert_eq!(
                got.len(),
                want.len(),
                "{}: trial {trial}: H length",
                fx.name
            );
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, e, "{}: trial {trial}: H differs at index {i}", fx.name);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 6. Domain sizes no artifact has
// ---------------------------------------------------------------------------

/// A proving key with a chosen domain size and a random CSR, so the transforms can be run
/// at sizes the checked-in circuits do not produce.
///
/// The six artifacts cover domains 2^3, 2^12, 2^14, 2^15, 2^17 and 2^18 and nothing else.
/// The NTT batches its passes (`MAX_FUSED_PASSES` in `stages.rs`), so how a transform is
/// split into kernel launches is a function of `log2(n)` alone, and a split that is wrong
/// at one `log2(n)` is invisible at every other. 2^13 and 2^16 in particular are exactly
/// the sizes a batching off-by-one would land on, and nothing in this workspace has ever
/// run them on a device.
///
/// The bases are real points rather than identities: an identity base makes every MSM
/// return the identity, which would agree with the CPU whether or not the MSM worked.
/// `n_vars` stays small so building the G2 vector stays cheap; the domain is what varies.
fn synthetic_key(domain_size: usize, n_vars: usize, seed: &mut u64) -> ProvingKey {
    use snarkrs_formats::Coefficients;

    let g1 = |n: usize| -> Vec<G1Affine> {
        let mut cur = G1Projective::generator();
        (0..n)
            .map(|_| {
                cur += G1Projective::generator();
                cur.into_affine()
            })
            .collect()
    };
    let mut cur2 = G2Projective::generator();
    let g2: Vec<G2Affine> = (0..n_vars)
        .map(|_| {
            cur2 += G2Projective::generator();
            cur2.into_affine()
        })
        .collect();

    // Two CSR matrices with about three entries a row, every signal index in range.
    let mut mat = || {
        let mut row_ptr = Vec::with_capacity(domain_size + 1);
        let mut signal = Vec::new();
        let mut value = Vec::new();
        row_ptr.push(0u32);
        for _ in 0..domain_size {
            let k = (splitmix(seed) % 4) as usize;
            for _ in 0..k {
                signal.push((splitmix(seed) % n_vars as u64) as u32);
                value.push(rand_fr(seed));
            }
            row_ptr.push(signal.len() as u32);
        }
        (row_ptr, signal, value)
    };
    let (rp0, sg0, v0) = mat();
    let (rp1, sg1, v1) = mat();

    let n_public = 1usize;
    let one = G1Projective::generator().into_affine();
    let one2 = G2Projective::generator().into_affine();
    ProvingKey {
        n_vars,
        n_public,
        domain_size,
        alpha_g1: one,
        beta_g1: one,
        beta_g2: one2,
        delta_g1: one,
        delta_g2: one2,
        a_query: g1(n_vars),
        b_g1_query: g1(n_vars),
        b_g2_query: g2,
        l_query: g1(n_vars - n_public - 1),
        h_query: g1(domain_size),
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

/// Every power-of-two domain from 1 up to 2^16, plus 2^19, against the CPU backend.
///
/// Both stage groups, on the same synthetic key, with a random witness: `compute_h` pins
/// the gather, the six transforms and the pass batching at each `log2(n)`, and `msms` pins
/// the window plan at an `H` length no artifact produces. This is the coverage gap the
/// artifact suite structurally cannot close, because the domain size is a property of the
/// circuit and there are only six circuits.
///
/// The key is built twice from the same seed rather than cloned, because `ProvingKey` is
/// deliberately not `Clone` and should stay that way: a derive would put a hundred-megabyte
/// copy one keystroke away from the proving path.
#[test]
fn every_domain_size_agrees_with_the_cpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            resources::skip(format_args!(
                "every_domain_size_agrees_with_the_cpu, no usable device: {e}"
            ));
            return;
        }
    };
    let cpu = CpuBackend::new();
    for log in [
        0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19,
    ] {
        let domain = 1usize << log;
        let n_vars = domain.clamp(4, 512);
        let key_seed = 0x0D0D_0000_0000_0000u64 | u64::from(log);
        let mut s1 = key_seed;
        let mut s2 = key_seed;
        let cuda_circuit = backend
            .prepare(synthetic_key(domain, n_vars, &mut s1))
            .unwrap_or_else(|e| panic!("2^{log}: cuda prepare: {e}"));
        let cpu_circuit = cpu
            .prepare(synthetic_key(domain, n_vars, &mut s2))
            .unwrap_or_else(|e| panic!("2^{log}: cpu prepare: {e}"));
        assert_eq!(s1, s2, "2^{log}: the two key builds diverged");

        let mut wseed = key_seed ^ 0xABCD_EF01_2345_6789;
        let w: Vec<Fr> = (0..n_vars).map(|_| rand_fr(&mut wseed)).collect();

        let mut t = StageTimings::default();
        let want = cpu_circuit
            .compute_h(&w, &mut t)
            .unwrap_or_else(|e| panic!("2^{log}: cpu compute_h: {e}"));
        let want_h = want.to_host().expect("host vector").to_vec();
        let got = cuda_circuit
            .compute_h(&w, &mut t)
            .unwrap_or_else(|e| panic!("2^{log}: cuda compute_h: {e}"));
        let got_h = got
            .device_handle::<snarkrs_cuda::stages::HHandle>(snarkrs_cuda::stages::TAG)
            .expect("a cuda device handle")
            .to_host()
            .unwrap_or_else(|e| panic!("2^{log}: reading H back: {e}"));
        assert_eq!(got_h.len(), want_h.len(), "2^{log}: H length");
        for (i, (g, e)) in got_h.iter().zip(&want_h).enumerate() {
            assert_eq!(g, e, "domain 2^{log}: H differs at index {i}");
        }

        let want_m = cpu_circuit
            .msms(&w, &want, &mut t)
            .unwrap_or_else(|e| panic!("2^{log}: cpu msms: {e}"));
        let got_m = cuda_circuit
            .msms(&w, &got, &mut t)
            .unwrap_or_else(|e| panic!("2^{log}: cuda msms: {e}"));
        assert_eq!(got_m.a_g1, want_m.a_g1, "domain 2^{log}: stage 5");
        assert_eq!(got_m.b_g2, want_m.b_g2, "domain 2^{log}: stage 6");
        assert_eq!(got_m.b_g1, want_m.b_g1, "domain 2^{log}: stage 7");
        assert_eq!(got_m.l_g1, want_m.l_g1, "domain 2^{log}: stage 8");
        assert_eq!(got_m.h_g1, want_m.h_g1, "domain 2^{log}: stage 9");
        eprintln!("  domain 2^{log} ({domain}) agrees on H and all five MSMs");
    }
}
