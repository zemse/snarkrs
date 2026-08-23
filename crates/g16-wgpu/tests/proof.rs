//! U11's acceptance tests: whole proofs, on a real GPU, against the CPU backend and against
//! our own verifier.
//!
//! Everything before this unit checked a stage. This file checks the thing the project is
//! for. Native only, for the reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ark_std::rand::{rngs::StdRng, SeedableRng};
use g16_core::cpu::CpuBackend;
use g16_core::prove::{prove, prove_with_blinders};
use g16_core::verify::verify;
use g16_core::{Backend, HPoly, ProveError, StageTimings};
use g16_field::{Fr, One, Zero};
use g16_wgpu::backend::WgpuProver;
use g16_wgpu::{G1Bases, G2Bases, Group, Job, MontConvert, MsmResult, Source};
use g16_wgpu::{LimitsProfile, WgpuBackend};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

#[path = "gpulock/mod.rs"]
mod gpulock;

/// Serialises the tests in this binary against each other.
///
/// # This is not tidiness, and the reason is a measurement nobody expected
///
/// Every test here proves on the one device [`device`] hands out, and
/// `WgpuBackend::exclusive` already serialises the GPU section of a proof, so a lock here
/// buys nothing for correctness *on one device*. What it buys is that
/// `a_whole_proof_is_two_submits_and_the_readback_is_bounded` can read a device-global submit
/// counter and get the number this binary's own proof made rather than the number four
/// concurrent tests made: the first draft of that test read 18 where the answer is 2.
///
/// The obvious fix was to give that one test a **second** `WgpuBackend`, so its counter was
/// private. That is what the second draft did, and it made three unrelated tests fail with
/// `PairingFailed` about one run in two. Bisected:
///
/// ```text
/// one device, 4 threads proving          12 runs, 0 wrong   <- see below
/// two processes, one device each         2 runs,  0 wrong
/// two devices in one process, tiny load  3 runs,  0 wrong
/// two devices in one process, serialised 24 reps, 0 wrong
/// two devices in one process, concurrent 36 reps, 7 wrong
/// ```
///
/// **Two `wgpu::Device`s in one process, both under load, silently produce wrong results on
/// this machine.** The same two workloads in two *processes* are fine, so it is neither GPU
/// contention nor total memory. No uncaptured error is raised and the device-lost callback
/// (added in `crate::device` while chasing this) never fires. The wrong values are always
/// whole stage outputs: H identical to the previous proof's, or four of the five MSMs wrong
/// with the first one right. M2 Max, wgpu 30.0.1, Metal, `STRICT_WEBGPU_COMPLIANCE` on,
/// `LimitsProfile::Floor`.
///
/// **The first row of that table does not mean what it was written to mean, and U11's
/// mutation round is what said so.** Deleting `WgpuBackend::exclusive` from `compute_h` and
/// `msms` and running `two_proofs_in_parallel_on_one_prepared_circuit_both_verify` alone,
/// which is four threads on *one* device, gave `verify: PairingFailed` on **run 2 of 3**.
/// So one device driven from four unguarded threads is not fine either; 12 runs was a small
/// sample of a roughly one-in-three event and it came up empty. Whatever else is going on
/// with two devices, the guard `crate::backend` takes is load bearing for correctness on one,
/// and not only for attributing an uncaptured error or a timing.
///
/// Nothing this crate ships opens two devices, so the product is not affected: a
/// `WgpuProver` owns one, and every proof through it holds the guard. It is written up in
/// `TASKS.md` because it is worth reporting upstream and because the next person to reach
/// for a private device in a test needs to know. Here, the answer is one device and this
/// lock.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One device for the whole binary. Opening an adapter is not free and every test here wants
/// the same one; sharing it is also what makes the submit counter meaningful across tests.
fn device() -> &'static Arc<WgpuBackend> {
    static B: OnceLock<Arc<WgpuBackend>> = OnceLock::new();
    B.get_or_init(|| {
        Arc::new(
            pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
                .expect("no wgpu device at the Floor profile"),
        )
    })
}

/// One `WgpuProver`, therefore one set of MSM pipelines, for the whole binary.
fn prover() -> &'static WgpuProver {
    static P: OnceLock<WgpuProver> = OnceLock::new();
    P.get_or_init(|| {
        WgpuProver::with_device(Arc::clone(device())).expect("the wgpu backend did not build")
    })
}

struct Artifact {
    name: String,
    dir: PathBuf,
}

impl Artifact {
    fn key(&self) -> ProvingKey {
        ProvingKey::load(&self.dir.join("circuit.zkey")).expect("circuit.zkey")
    }
    fn witness(&self) -> Vec<Fr> {
        Witness::load(&self.dir.join("circuit.wtns"))
            .expect("circuit.wtns")
            .0
    }
    fn vkey(&self) -> VerifyingKey {
        VerifyingKey::from_json(&self.dir.join("vkey.json")).expect("vkey.json")
    }
    /// Constraint count from snarkjs' `r1cs-info.txt`, the same field
    /// `g16_cli::artifacts::Variant::constraints` reads, so the crossover this file publishes
    /// is quoted in the same units as `bench/results/history.csv`. -1 when it is missing.
    fn constraints(&self) -> i64 {
        let Ok(text) = std::fs::read_to_string(self.dir.join("r1cs-info.txt")) else {
            return -1;
        };
        text.lines()
            .find(|l| l.contains("# of Constraints"))
            .and_then(|l| l.split_whitespace().last())
            .and_then(|t| t.parse().ok())
            .unwrap_or(-1)
    }
}

fn artifacts() -> Vec<Artifact> {
    let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
    else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<Artifact> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            ["circuit.zkey", "circuit.wtns", "vkey.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .map(|dir| Artifact {
            name: dir.file_name().unwrap().to_string_lossy().into_owned(),
            dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Runs `f` over every artifact. An empty artifact directory reports a skip on stderr rather
/// than passing silently, because `bench/artifacts` is a gitignored symlink and a green test
/// over nothing is the failure mode that hides.
fn for_each_artifact(test: &str, f: impl Fn(&Artifact)) {
    let found = artifacts();
    if found.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
        return;
    }
    for a in &found {
        eprintln!("{test}: {}", a.name);
        f(a);
    }
}

fn public_of(witness: &[Fr], n_public: usize) -> Vec<Fr> {
    witness[1..=n_public].to_vec()
}

// ---------------------------------------------------------------------------
// 1. The acceptance: a proof, on every artifact, through our own verifier
// ---------------------------------------------------------------------------

/// The test the whole backend exists for.
#[test]
fn every_artifact_proves_and_verifies_on_the_device() {
    let _one_at_a_time = exclusive();
    for_each_artifact("every_artifact_proves_and_verifies_on_the_device", |a| {
        let pk = a.key();
        let n_public = pk.n_public;
        let circuit = prover().prepare(pk).expect("prepare");
        let witness = a.witness();
        let public = public_of(&witness, n_public);
        let vk = a.vkey();

        let mut t = StageTimings::default();
        let mut rng = StdRng::from_seed([7u8; 32]);
        let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).expect("prove");
        verify(&vk, &public, &proof).expect("our verifier rejected our own proof");

        assert_eq!(circuit.backend_name(), "wgpu");
        // A prover that reported no time would make `bench` claim a free proof.
        assert!(
            t.gather_us > 0 && t.ntt_us > 0 && t.msm_us > 0,
            "{}: stage timings are {t:?}",
            a.name
        );
        eprintln!(
            "  {} gather {} us, ntt {} us, msm {} us, assemble {} us",
            a.name, t.gather_us, t.ntt_us, t.msm_us, t.assemble_us
        );
    });
}

/// The claim `README.md` already makes for CPU against Metal, made for WGSL or not made.
///
/// With `r` and `s` pinned the proof is a deterministic function of the witness, so the two
/// backends have to agree on all three points *exactly* and not merely up to the group law.
/// Every MSM in both combines its windows high to low with `c` doublings between, and stage 4
/// is exact field arithmetic either way, so a single differing bit is a bug and never an
/// association difference.
#[test]
fn cpu_and_wgpu_proofs_are_bit_identical_at_pinned_blinders() {
    let _one_at_a_time = exclusive();
    for_each_artifact(
        "cpu_and_wgpu_proofs_are_bit_identical_at_pinned_blinders",
        |a| {
            let witness = a.witness();
            let (r, s) = (Fr::from(12345u64), Fr::from(67890u64));

            let mut t = StageTimings::default();
            let cpu = CpuBackend::new().prepare(a.key()).expect("cpu prepare");
            let want =
                prove_with_blinders(cpu.as_ref(), &witness, r, s, &mut t).expect("cpu prove");

            let gpu = prover().prepare(a.key()).expect("wgpu prepare");
            let got =
                prove_with_blinders(gpu.as_ref(), &witness, r, s, &mut t).expect("wgpu prove");

            assert_eq!(got.a, want.a, "{}: pi_a differs", a.name);
            assert_eq!(got.b, want.b, "{}: pi_b differs", a.name);
            assert_eq!(got.c, want.c, "{}: pi_c differs", a.name);

            // And it is not vacuously true because both are the identity.
            assert!(!got.a.infinity, "{}: pi_a is infinity", a.name);
            assert!(!got.b.infinity, "{}: pi_b is infinity", a.name);
            assert!(!got.c.infinity, "{}: pi_c is infinity", a.name);

            verify(&a.vkey(), &public_of(&witness, cpu.n_public()), &got).expect("verify");
        },
    );
}

/// Two proofs of the same witness on one circuit must be identical, and so must the H
/// buffer's contribution to them. Pooled scratch that leaked state between proofs shows here
/// and nowhere else.
#[test]
fn proving_the_same_witness_twice_is_stable() {
    let _one_at_a_time = exclusive();
    for_each_artifact("proving_the_same_witness_twice_is_stable", |a| {
        let circuit = prover().prepare(a.key()).expect("prepare");
        let witness = a.witness();
        let (r, s) = (Fr::from(3u64), Fr::from(5u64));
        let mut t = StageTimings::default();
        let p1 = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).expect("1");
        let p2 = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).expect("2");
        assert_eq!((p1.a, p1.b, p1.c), (p2.a, p2.b, p2.c), "{}", a.name);
    });
}

// ---------------------------------------------------------------------------
// 2. The contract: `&self`, from several threads
// ---------------------------------------------------------------------------

/// `PreparedCircuit` is documented as safe to call from several threads against one instance,
/// and U11 had to make that true rather than leave it as a claim.
///
/// The piece that was not safe is not a buffer: it is the device's single uncaptured-error
/// slot, which `take_error` empties, so with two proofs in flight one proof's dropped
/// dispatch was reported against the other and the proof that caused it returned `Ok` over
/// buffers whose contents nobody could describe. `WgpuBackend::exclusive` now serialises the
/// GPU section of a proof, so concurrent proofs are correct and serialised rather than
/// parallel.
///
/// This asserts the outcome: four threads, one circuit, four proofs that verify. It is also
/// the test that showed the guard is not merely tidiness. Run with both
/// `let _gpu = self.device.exclusive();` lines deleted from `crate::backend` it fails with
/// `verify: PairingFailed` about one run in three, which is written up on [`exclusive`].
/// `the_gpu_section_of_a_proof_is_behind_the_device_guard` asserts the mechanism instead,
/// deterministically, because one run in three is not a gate anyone should rely on.
#[test]
fn two_proofs_in_parallel_on_one_prepared_circuit_both_verify() {
    let _one_at_a_time = exclusive();
    for_each_artifact("two_proofs_in_parallel_on_one_prepared_circuit", |a| {
        let pk = a.key();
        let n_public = pk.n_public;
        let circuit = prover().prepare(pk).expect("prepare");
        let witness = a.witness();
        let public = public_of(&witness, n_public);
        let vk = a.vkey();
        let circuit = circuit.as_ref();
        let (witness, vk, public) = (&witness, &vk, &public);

        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..4u8)
                .map(|seed| {
                    scope.spawn(move || {
                        let mut t = StageTimings::default();
                        let mut rng = StdRng::from_seed([seed; 32]);
                        let proof = prove(circuit, witness, &mut rng, &mut t).expect("prove");
                        verify(vk, public, &proof).expect("verify");
                    })
                })
                .collect();
            for t in threads {
                t.join().expect("a proving thread panicked");
            }
        });
    });
}

/// The other half of the same contract: the GPU section of a proof really is behind
/// [`g16_wgpu::WgpuBackend::exclusive`], and not merely documented as being.
///
/// `two_proofs_in_parallel_on_one_prepared_circuit_both_verify` cannot see this. Four threads
/// on **one** device were measured correct with the lock and correct without it (12 runs, 0
/// wrong; the bisect is on [`exclusive`]), because the thing the lock defends against is the
/// device's single uncaptured-error slot being emptied by the wrong proof, and that needs a
/// device error to happen at all. A test that waits for one is a test that never fails on a
/// working machine.
///
/// So this asserts the mechanism instead of the symptom. Hold the backend guard on this
/// thread, call `compute_h` on another, and it must **not** finish; release, and it must.
/// That is deterministic in both directions: a guard that is taken blocks whatever the GPU is
/// doing, and a guard that is not taken lets the smallest artifact through in about 20 ms.
///
/// Both entry points are checked. Deleting the guard from `compute_h` and leaving it in
/// `msms` is exactly the shape a careless edit takes, and one assertion would pass it.
///
/// The smallest artifact, on purpose: this waits 300 ms for a call that must not complete,
/// and `js_16x16_d32`'s `msms` takes 900 ms all by itself, so the largest artifact would
/// "pass" with no guard at all.
#[test]
fn the_gpu_section_of_a_proof_is_behind_the_device_guard() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let _one_at_a_time = exclusive();
    /// Long enough that the smallest artifact's `compute_h` and `msms` both finish inside it
    /// with no guard (20 ms and 21 ms warm), short enough to pay twice.
    const BLOCKED_FOR: std::time::Duration = std::time::Duration::from_millis(300);

    let found = artifacts();
    let Some(a) = found.iter().min_by_key(|a| a.witness().len()) else {
        eprintln!("SKIPPED the_gpu_section_of_a_proof_is_behind_the_device_guard: no artifacts");
        return;
    };
    eprintln!(
        "the_gpu_section_of_a_proof_is_behind_the_device_guard: {}",
        a.name
    );

    let circuit = prover().prepare(a.key()).expect("prepare");
    let witness = a.witness();
    let mut t = StageTimings::default();
    // Warm the pools and take an H to hand `msms`, both while nothing holds the guard.
    let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
    circuit.msms(&witness, &h, &mut t).expect("msms");

    let circuit = circuit.as_ref();
    let (witness, h) = (&witness, &h);
    for (what, call) in [("compute_h", 0u8), ("msms", 1u8)] {
        let done = AtomicBool::new(false);
        let done = &done;
        let guard = device().exclusive();
        std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                let mut t = StageTimings::default();
                match call {
                    0 => {
                        circuit.compute_h(witness, &mut t).expect("compute_h");
                    }
                    _ => {
                        circuit.msms(witness, h, &mut t).expect("msms");
                    }
                }
                done.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(BLOCKED_FOR);
            assert!(
                !done.load(Ordering::SeqCst),
                "{what} finished while another thread held WgpuBackend::exclusive, so the \
                 GPU section of a proof is not serialised and one proof can take another's \
                 uncaptured error"
            );
            drop(guard);
            worker.join().expect("the blocked call panicked");
        });
        assert!(done.load(Ordering::SeqCst), "{what} never finished");
    }
}

// ---------------------------------------------------------------------------
// 3. Rejections
// ---------------------------------------------------------------------------

#[test]
fn wrong_witness_lengths_are_all_rejected() {
    let _one_at_a_time = exclusive();
    for_each_artifact("wrong_witness_lengths_are_all_rejected", |a| {
        let circuit = prover().prepare(a.key()).expect("prepare");
        let witness = a.witness();
        let n = witness.len();
        let mut t = StageTimings::default();

        for bad in [0usize, 1, n - 1, n + 1] {
            let mut w = witness.clone();
            w.resize(bad, Fr::from(1u64));
            assert!(
                matches!(
                    circuit.compute_h(&w, &mut t),
                    Err(ProveError::WitnessLength { .. })
                ),
                "{}: compute_h accepted a {bad} element witness where {n} was wanted",
                a.name
            );
        }

        // `msms` has its own check, and it has to: a caller can reach it directly with an H
        // from a correct `compute_h` and a witness from somewhere else.
        let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
        let short = &witness[..n - 1];
        assert!(matches!(
            circuit.msms(short, &h, &mut t),
            Err(ProveError::WitnessLength { .. })
        ));
        // And an H of the wrong length is a backend error rather than a wrong proof.
        let wrong = HPoly::Host(vec![Fr::from(1u64); 3]);
        assert!(matches!(
            circuit.msms(&witness, &wrong, &mut t),
            Err(ProveError::Backend { .. })
        ));
    });
}

/// Stages 5 to 9 alone, against stages 0 to 4 from the CPU backend.
///
/// This is what `crate::batch::Source::Host` is for, and it is the only way to say "the MSM
/// half is right" without the transform half being right too. It also exercises the fallback
/// itself, which the proving path never touches and which would otherwise ship untested.
#[test]
fn the_device_msms_finish_a_cpu_computed_h() {
    let _one_at_a_time = exclusive();
    for_each_artifact("the_device_msms_finish_a_cpu_computed_h", |a| {
        let witness = a.witness();
        let mut t = StageTimings::default();

        let cpu = CpuBackend::new().prepare(a.key()).expect("cpu prepare");
        let h = cpu.compute_h(&witness, &mut t).expect("cpu compute_h");
        assert!(h.to_host().is_some(), "the CPU backend stopped being Host");

        let want = cpu.msms(&witness, &h, &mut t).expect("cpu msms");
        let gpu = prover().prepare(a.key()).expect("wgpu prepare");
        let got = gpu.msms(&witness, &h, &mut t).expect("wgpu msms");

        assert_eq!(got.a_g1, want.a_g1, "{}: A", a.name);
        assert_eq!(got.b_g2, want.b_g2, "{}: B in G2", a.name);
        assert_eq!(got.b_g1, want.b_g1, "{}: B in G1", a.name);
        assert_eq!(got.l_g1, want.l_g1, "{}: L", a.name);
        assert_eq!(got.h_g1, want.h_g1, "{}: H", a.name);
    });
}

/// A [`Group`] with `n == 0` dispatches nothing and still owns its own result slots.
///
/// `MsmBatch::run` returns one result per job, flat, in group then job order, but it only
/// *computes* the jobs of non-empty groups. So the two loops that fill the output disagree
/// about indexing on purpose: the computed jobs are placed at `first_job[group] + job`, a
/// prefix sum, and a second pass fills the identity into whatever is left. Indexing the first
/// pass by the running count of computed jobs instead would be correct on every artifact we
/// have and wrong the first time a group is empty, and the symptom would be a proof that does
/// not verify with nothing else to go on.
///
/// The case is real and is the reason the code is written that way: an empty L query is a
/// circuit whose wires are all public. No artifact under `bench/artifacts` has one, and no
/// test had one either until this, which is how the mutation that collapses the two indexings
/// survived U11's first mutation round.
///
/// The oracle is the same batch without the empty group, so this asserts the placement and
/// nothing about the curve arithmetic that four other tests already cover. Both arms of the
/// identity fill are exercised, G1 and G2, because they are two separate match arms.
#[test]
fn an_empty_group_dispatches_nothing_and_keeps_its_result_slots() {
    let _one_at_a_time = exclusive();
    for_each_artifact(
        "an_empty_group_dispatches_nothing_and_keeps_its_slots",
        |a| {
            let pk = a.key();
            let witness = a.witness();
            let n_vars = pk.n_vars as u32;
            let b = device();
            let batch = prover().msm();
            let a_bases = G1Bases::upload(b, &pk.a_query).expect("a bases");
            let l_bases = G1Bases::upload(b, &pk.l_query).expect("l bases");
            let g2_bases = G2Bases::upload(b, &pk.b_g2_query).expect("b_g2 bases");

            let first = [Job::G1 {
                bases: &a_bases,
                base_off: 0,
            }];
            // Two jobs and two curves, so both arms of the identity fill are covered.
            let empty = [
                Job::G1 {
                    bases: &l_bases,
                    base_off: 0,
                },
                Job::G2 {
                    bases: &g2_bases,
                    base_off: 0,
                },
            ];
            let last = [Job::G2 {
                bases: &g2_bases,
                base_off: 0,
            }];

            let g_first = || Group {
                scalars: Source::Host(&witness),
                scalar_off: 0,
                n: n_vars,
                jobs: &first,
            };
            let g_last = || Group {
                scalars: Source::Host(&witness),
                scalar_off: 0,
                n: n_vars,
                jobs: &last,
            };

            let want = pollster::block_on(batch.run(b, None, &[g_first(), g_last()])).expect("two");
            assert_eq!(want.len(), 2, "{}", a.name);

            let got = pollster::block_on(batch.run(
                b,
                None,
                &[
                    g_first(),
                    Group {
                        scalars: Source::Host(&witness),
                        scalar_off: 0,
                        n: 0,
                        jobs: &empty,
                    },
                    g_last(),
                ],
            ))
            .expect("three");

            assert_eq!(
                got.len(),
                4,
                "{}: one result per job, empty group included",
                a.name
            );
            let g1 = |r: &MsmResult| r.g1().expect("a G1 result");
            let g2 = |r: &MsmResult| r.g2().expect("a G2 result");
            assert_eq!(
                g1(&got[0]),
                g1(&want[0]),
                "{}: the job before the empty group",
                a.name
            );
            assert!(
                g1(&got[1]).is_zero(),
                "{}: the empty group's G1 job is not the identity",
                a.name
            );
            assert!(
                g2(&got[2]).is_zero(),
                "{}: the empty group's G2 job is not the identity",
                a.name
            );
            assert_eq!(
                g2(&got[3]),
                g2(&want[1]),
                "{}: the job after the empty group",
                a.name
            );
            // And the oracle is not vacuous.
            assert!(
                !g1(&want[0]).is_zero() && !g2(&want[1]).is_zero(),
                "{}",
                a.name
            );
        },
    );
}

// ---------------------------------------------------------------------------
// 4. The budgets the design writes down
// ---------------------------------------------------------------------------

/// Design §3 puts a whole proof in **two** submits, one for `compute_h` and one for `msms`.
/// It also caps the whole per-proof readback at 64 KiB, and that half is wrong; see below.
///
/// # Why the submit count needs [`exclusive`] and not a second device
///
/// `WgpuBackend::submits` is a counter on the device that every test in this binary shares,
/// and tests in a binary run in parallel by default, so reading it either side of a proof
/// measures whatever else was proving at the same time: the first draft of this test read 18
/// where the answer is 2. The obvious fix, a private device for this test, corrupts other
/// tests' results; see [`exclusive`], which carries the bisect.
///
/// # Design §3's 64 KiB readback budget does not hold, and the reason is not a bug
///
/// The readback is `n_windows` points per MSM plus up to `ones_groups` partials, and
/// `n_windows` is `ceil(255 / c)`. Design §3 quotes "20 window sums plus up to 64
/// `ones_groups` partials per MSM, four G1 and one G2", which is `c = 13`. But `c` is chosen
/// from the count of scalars that are neither 0 nor 1, and a **small** circuit gets a
/// **small** `c`: the bucket clear and reduction dominate its cost model, so
/// `msm::window_size` picks 3 and the proof reads back 85 window sums per MSM instead of 20.
/// Measured over the artifacts:
///
/// ```text
/// artifact       constraints   readback B
/// tiny_mul                 8        67072   <- over 65536
/// js_1x1_d8             3359        25856
/// js_2x2_d16           10153        26368
/// js_2x2_d32           17929        27136
/// js_8x8_d32           70357        25856
/// js_16x16_d32        140261        35328
/// ```
///
/// So the budget holds on every real circuit and is exceeded by 2% on an 8-constraint one,
/// which is the shape nobody will ever prove in a browser and which reads back more bytes
/// than its witness contains. It costs nothing measurable either way, because the cost of a
/// readback is one `mapAsync` round trip at about 0.3 ms and not the byte count, and the
/// wasm ceiling that number is really defending (`get_mapped_range` copies the whole mapped
/// `ArrayBuffer` into linear memory, 33.8 ms for 64 MiB) is three orders of magnitude away.
///
/// This asserts a bound that is a theorem about the code rather than the design's constant:
/// `ceil(255/c_min)` windows plus `ones_groups` partials, over four G1 jobs and one G2, with
/// `c_min = 3` because `msm::window_size` searches `3..=16`. That is loose, because a `c` low
/// enough to give 85 windows only happens for an `n` far too small to give 64 ones groups,
/// and it is still the honest ceiling.
#[test]
fn a_whole_proof_is_two_submits_and_the_readback_is_bounded() {
    let _one_at_a_time = exclusive();
    /// The narrowest window `msm::window_size` will return.
    const C_MIN: u32 = 3;
    /// `ones_groups_for` clamps to this.
    const MAX_ONES_GROUPS: u64 = 64;
    /// Design §3's figure, printed against the measurement rather than asserted.
    const DESIGN_BUDGET: u64 = 64 * 1024;

    let windows = u64::from(g16_wgpu::msm::RECODE_BITS.div_ceil(C_MIN));
    let per_job = |pt: u64| (windows * pt).div_ceil(256) * 256 + MAX_ONES_GROUPS * pt;
    let ceiling = 4 * per_job(g16_wgpu::gen::points::G1.point_bytes)
        + per_job(g16_wgpu::gen::points::G2.point_bytes);

    let found = artifacts();
    if found.is_empty() {
        eprintln!("SKIPPED a_whole_proof_is_two_submits: no artifacts under bench/artifacts");
        return;
    }
    let device = device();
    let prover = prover();

    println!("artifact       constraints  submits   readback B  over design 64 KiB");
    for a in &found {
        let circuit = prover.prepare(a.key()).expect("prepare");
        let witness = a.witness();
        let mut t = StageTimings::default();
        let mut rng = StdRng::from_seed([1u8; 32]);
        // One warm proof first. The pools allocate on the first proof, and a future version
        // that cleared a buffer with a dispatch of its own would otherwise be counted here.
        prove(circuit.as_ref(), &witness, &mut rng, &mut t).expect("warm");

        let before = device.submits();
        let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).expect("prove");
        let after = device.submits();
        assert_eq!(
            after - before,
            2,
            "{}: a proof took {} submits, design §3 budgets 2 (compute_h, then msms)",
            a.name,
            after - before
        );

        let read = prover.msm().last_readback_bytes();
        assert!(
            read > 0 && read <= ceiling,
            "{}: the MSM readback is {read} bytes, over the {ceiling} byte ceiling that \
             {windows} windows at c = {C_MIN} and {MAX_ONES_GROUPS} ones groups allow",
            a.name
        );
        verify(&a.vkey(), &public_of(&witness, circuit.n_public()), &proof).expect("verify");
        println!(
            "{:14} {:>11}  {:>7}   {:>10}  {}",
            a.name,
            a.constraints(),
            after - before,
            read,
            if read > DESIGN_BUDGET { "yes" } else { "no" }
        );
    }
    println!("ceiling from the constants: {ceiling} bytes");
}

// ---------------------------------------------------------------------------
// 5. The denominator: where a proof's time actually goes, and against what
// ---------------------------------------------------------------------------

/// Which of the five MSMs owns the 907 ms.
///
/// `msms` is 98% of a warm proof at 2^18, so "the MSM is slow" is not a useful statement and
/// the split is. This runs the three digit groups one at a time as their own batches, which
/// costs three fences instead of one (about 70 microseconds against 900 ms, so it changes
/// nothing), and then all three together to confirm the parts add up.
///
/// It builds the base vectors itself rather than reaching into `WgpuCircuit`, so nothing had
/// to become public for a report to exist.
///
/// # What it says, measured on an M2 Max at the Floor profile, best of three
///
/// ```text
/// artifact       n_vars   general   domain   A+B2+B1     L        H      sum   batched
/// tiny_mul            6         5        8     12.71   3.31     5.79    21.82    21.14
/// js_1x1_d8        3373      3234     4096    189.78  38.62    36.75   265.15   264.12
/// js_2x2_d16      10194      9906    16384    192.52  40.47    38.86   271.85   269.43
/// js_2x2_d32      18002     17682    32768    194.20  39.68    45.53   279.41   278.69
/// js_8x8_d32      70640     69372   131072    485.99  69.42    97.41   652.82   645.77
/// js_16x16_d32   140824    138292   262144    666.34 100.09   156.90   923.32   957.56
/// ```
///
/// Two things, and the second is the one that matters.
///
/// **The batch costs what the parts cost.** Three submissions and one submission agree to
/// within 1% at every artifact except the largest, where the single batch is 3.7% *slower*
/// and the run-to-run spread on a 900 ms number is wider than that. So `msm_batch` adds no
/// arithmetic and hides none: what it saves is two fences and two map round trips, which is
/// about 0.7 ms and invisible next to this.
///
/// **The cost barely moves with `n`.** From 3,373 variables to 18,002, a 5.3x increase, the
/// three witness MSMs go from 189.8 ms to 194.2, 2.3%. That is U9's finding reproduced at
/// whole-proof scale and it is worth doing the division: 189.8 ms for A, B-G2 and B-G1 is
/// 38 ms per G1-equivalent MSM (B-G2 costs 3.05x a G1 one, so the three are five units),
/// against the 38.3 ms `tests/msm_g1.rs::what_one_g1_msm_costs_against_the_cpu` measured for
/// **one** G1 MSM at n = 4096 standing alone. The two agree to the last figure. Whatever is
/// wrong is not in this file: batching five MSMs is free, and each of the five costs exactly
/// what U9 measured it costing on its own. The fix is the occupancy item in `TASKS.md`.
#[test]
fn where_the_msm_time_goes() {
    let _one_at_a_time = exclusive();
    let _lock = gpulock::exclusive_gpu();
    for_each_artifact("where_the_msm_time_goes", |a| {
        let pk = a.key();
        let witness = a.witness();
        let n_vars = pk.n_vars as u32;
        let n_public = pk.n_public;
        let domain = pk.domain_size as u32;
        let l_len = pk.l_query.len() as u32;
        let private_from = (n_public + 1) as u32;

        let b = device();
        let batch = prover().msm();
        let a_bases = G1Bases::upload(b, &pk.a_query).expect("a");
        let b_g1_bases = G1Bases::upload(b, &pk.b_g1_query).expect("b_g1");
        let b_g2_bases = G2Bases::upload(b, &pk.b_g2_query).expect("b_g2");
        let l_bases = G1Bases::upload(b, &pk.l_query).expect("l");
        let h_bases = G1Bases::upload(b, &pk.h_query).expect("h");

        // Stage 4's own output, so the H group reads the buffer the prover would.
        let circuit = prover().prepare(a.key()).expect("prepare");
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
        let handle = h
            .device_handle::<g16_wgpu::stages::WgpuHandle>(g16_wgpu::stages::TAG)
            .expect("a wgpu handle");

        let (g_all, g_priv) = {
            let mut all = 0u32;
            let mut private = 0u32;
            for (i, x) in witness.iter().enumerate() {
                if !(x.is_zero() || x.is_one()) {
                    all += 1;
                    if i >= private_from as usize {
                        private += 1;
                    }
                }
            }
            (all, private)
        };

        let w_jobs = [
            Job::G1 {
                bases: &a_bases,
                base_off: 0,
            },
            Job::G2 {
                bases: &b_g2_bases,
                base_off: 0,
            },
            Job::G1 {
                bases: &b_g1_bases,
                base_off: 0,
            },
        ];
        let l_jobs = [Job::G1 {
            bases: &l_bases,
            base_off: 0,
        }];
        let h_jobs = [Job::G1 {
            bases: &h_bases,
            base_off: 0,
        }];
        let ws = handle.witness_std();
        let mont = || MontConvert {
            src: handle.witness_mont(),
            dst: ws,
            n: n_vars,
        };
        let w_group = || Group {
            scalars: Source::Device {
                buf: ws,
                general: Some(g_all),
            },
            scalar_off: 0,
            n: n_vars,
            jobs: &w_jobs,
        };
        let l_group = || Group {
            scalars: Source::Device {
                buf: ws,
                general: Some(g_priv),
            },
            scalar_off: private_from,
            n: l_len,
            jobs: &l_jobs,
        };
        let h_group = || Group {
            scalars: Source::Device {
                buf: handle.h_std(),
                general: None,
            },
            scalar_off: 0,
            n: domain,
            jobs: &h_jobs,
        };

        // Warm the pools and the pipelines before anything is timed.
        let all = vec![w_group(), l_group(), h_group()];
        pollster::block_on(batch.run(b, Some(mont()), &all)).expect("warm");

        let time = |groups: Vec<Group<'_>>, mont: Option<MontConvert<'_>>| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                pollster::block_on(batch.run(
                    b,
                    mont.as_ref().map(|m| MontConvert {
                        src: m.src,
                        dst: m.dst,
                        n: m.n,
                    }),
                    &groups,
                ))
                .expect("run");
                best = best.min(start.elapsed().as_secs_f64() * 1000.0);
            }
            best
        };

        let t_w = time(vec![w_group()], Some(mont()));
        let t_l = time(vec![l_group()], None);
        let t_h = time(vec![h_group()], None);
        let t_all = time(vec![w_group(), l_group(), h_group()], Some(mont()));

        println!(
            "{:14} n_vars {:>7} general {:>7} domain {:>7} | A+B2+B1 {:8.2} ms  L {:8.2}  \
             H {:8.2}  sum {:8.2}  batched {:8.2}",
            a.name,
            n_vars,
            g_all,
            domain,
            t_w,
            t_l,
            t_h,
            t_w + t_l + t_h,
            t_all
        );
        // The batch cannot be slower than running the same work in three submissions, which
        // is the claim `msm_batch` is built on. Allowed 10% of slack for the medians being
        // three-sample minima on a machine that is not idle.
        assert!(
            t_all <= 1.10 * (t_w + t_l + t_h),
            "{}: one batch took {t_all:.2} ms where three took {:.2}",
            a.name,
            t_w + t_l + t_h
        );
    });
}
