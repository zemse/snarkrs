//! `Work::Constant` on the device: the MSMs still agree with the CPU on every sparsity, a
//! whole proof is the variable one bit for bit, the geometry a witness of bits and a dense
//! one produce is the same, and the retry paths hold in this mode too. Native only, for the
//! reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use g16_core::prove::prove_with_blinders;
use g16_core::verify::verify;
use g16_core::{Backend, StageTimings};
use g16_field::{
    CurveGroup, Fr, G1Affine, G1Projective, G2Affine, G2Projective, One, PrimeField, PrimeGroup,
    Zero,
};
use g16_msm::{CpuMsm, MsmBackend};
use g16_wgpu::backend::WgpuProver;
use g16_wgpu::msm::pack_scalars;
use g16_wgpu::{
    DigitBuffers, DigitPlan, G1Bases, G2Bases, Group, Job, LimitsProfile, MsmResult, ParamRing,
    PointBuffers, Source, WgpuBackend, Work, DUMMY_ROWS,
};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

#[path = "msmcommon/mod.rs"]
mod common;
#[path = "gpulock/mod.rs"]
mod gpulock;

use common::{exclusive, median, read_words, storage_words, submit_sealed};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One device for the whole binary; see `tests/proof.rs::exclusive` for why not two.
fn device() -> &'static Arc<WgpuBackend> {
    static B: OnceLock<Arc<WgpuBackend>> = OnceLock::new();
    B.get_or_init(|| {
        Arc::new(
            pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
                .expect("no wgpu device at the Floor profile"),
        )
    })
}

/// The variable prover, whose MSM modules the constant one shares: both modes live in the
/// same three modules, so this is one compile for the binary.
fn prover() -> &'static WgpuProver {
    static P: OnceLock<WgpuProver> = OnceLock::new();
    P.get_or_init(|| {
        WgpuProver::with_device(Arc::clone(device())).expect("the wgpu backend did not build")
    })
}

fn constant_prover() -> &'static WgpuProver {
    static P: OnceLock<WgpuProver> = OnceLock::new();
    P.get_or_init(|| {
        WgpuProver::with_device_and_work(Arc::clone(device()), Work::Constant)
            .expect("the wgpu backend did not build")
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

/// Runs `f` over every artifact, and says so rather than passing over none: the artifact
/// directory is a gitignored symlink.
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

/// Bases with every seventh at infinity: the finite mask is the one thing the key is
/// allowed to decide. An arithmetic walk rather than `n` scalar multiplications, as
/// `tests/msm_g1.rs::walk_bases` argues: structure in the bases cannot help an MSM.
fn bases_with_infinity(n: usize) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let mut p1 = G1Projective::generator();
    let mut p2 = G2Projective::generator();
    let (mut g1, mut g2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        p1 += G1Projective::generator();
        p2 += G2Projective::generator();
        g1.push(p1);
        g2.push(p2);
    }
    let mut g1 = G1Projective::normalize_batch(&g1);
    let mut g2 = G2Projective::normalize_batch(&g2);
    for i in (3..n).step_by(7) {
        g1[i] = G1Affine::identity();
        g2[i] = G2Affine::identity();
    }
    (g1, g2)
}

fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn random_scalars(n: usize, seed: &mut u64) -> Vec<Fr> {
    (0..n)
        .map(|_| {
            let mut b = [0u8; 32];
            for c in b.chunks_mut(8) {
                c.copy_from_slice(&xorshift(seed).to_le_bytes());
            }
            Fr::from_le_bytes_mod_order(&b)
        })
        .collect()
}

/// Scalar vectors of every sparsity the witness path meets: dense random, all zeros, all
/// ones, bits with a few generals, and small values.
fn scalars_of_every_sparsity(n: usize) -> Vec<(&'static str, Vec<Fr>)> {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    vec![
        ("random", random_scalars(n, &mut seed)),
        ("zeros", vec![Fr::zero(); n]),
        ("ones", vec![Fr::one(); n]),
        (
            "bits and a few generals",
            (0..n)
                .map(|i| {
                    if i % 97 == 5 {
                        -Fr::from(i as u64 * 7919 + 3)
                    } else {
                        Fr::from((i % 2) as u64)
                    }
                })
                .collect(),
        ),
        ("small", (0..n).map(|i| Fr::from((i % 5) as u64)).collect()),
    ]
}

fn g1_of(r: &MsmResult) -> G1Projective {
    r.g1().expect("a G1 result")
}
fn g2_of(r: &MsmResult) -> G2Projective {
    r.g2().expect("a G2 result")
}

// ---------------------------------------------------------------------------
// 1. The MSMs against the CPU
// ---------------------------------------------------------------------------

/// `Work::Constant` against the CPU MSM, in both groups, on every sparsity, over a range
/// that crosses slices and fold levels, plus a job over an offset subrange the way the L
/// MSM reads one. Then the same again with the group cut into sub-MSMs at a tiny binding
/// limit over chunked base vectors, and with every window in a submission of its own, the
/// two shapes `crate::batch` reaches on a large key and under a kill.
#[test]
fn constant_work_msms_match_the_cpu_on_every_sparsity() {
    let _one_at_a_time = exclusive();
    /// Crosses the 128-entry slices and gives the fold five levels at c = 8.
    const N: usize = 3 * 4096 + 7;
    /// Small enough to cut the entry array (32 windows x 12,295 entries x 8 bytes) twice.
    const LIMIT: u64 = 2 << 20;
    let b = device();
    let batch = prover().msm();
    let (g1, g2) = bases_with_infinity(N);
    let b1 = G1Bases::upload(b, &g1).expect("g1 bases");
    let b2 = G2Bases::upload(b, &g2).expect("g2 bases");
    let chunk = ((N as u32) / 7).max(3) | 1;
    let c1 = G1Bases::upload_chunked(b, &g1, chunk).expect("g1 chunks");
    let c2 = G2Bases::upload_chunked(b, &g2, chunk).expect("g2 chunks");
    assert!(c1.chunks() >= 2);
    let cpu = CpuMsm::new();
    let off = 5usize;
    let n = N as u32;

    for (label, scalars) in scalars_of_every_sparsity(N) {
        let want1 = cpu.msm_g1(&g1, &scalars);
        let want2 = cpu.msm_g2(&g2, &scalars);
        let want_off = cpu.msm_g1(&g1[off..], &scalars[off..]);
        let check = |what: &str, got: &[MsmResult]| {
            assert_eq!(got.len(), 3, "{label}, {what}");
            assert_eq!(g1_of(&got[0]), want1, "{label}, {what}: G1");
            assert_eq!(g2_of(&got[1]), want2, "{label}, {what}: G2");
            assert_eq!(
                g1_of(&got[2]),
                want_off,
                "{label}, {what}: G1 over an offset range"
            );
        };
        // Whole, one piece, the shape every artifact under 2^20 variables takes.
        let whole = [
            Job::G1 {
                bases: &b1,
                base_off: 0,
            },
            Job::G2 {
                bases: &b2,
                base_off: 0,
            },
        ];
        let off_job = [Job::G1 {
            bases: &b1,
            base_off: off as u32,
        }];
        let gs = [
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: 0,
                n,
                jobs: &whole,
                work: Work::Constant,
            },
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: off as u32,
                n: n - off as u32,
                jobs: &off_job,
                work: Work::Constant,
            },
        ];
        let got = pollster::block_on(batch.run(b, None, &gs)).expect("run");
        check("whole", &got);
        assert_eq!(batch.last_sub_msms(), (2, 1), "{label}: cut for nothing");

        // Cut into sub-MSMs on the entry array and at every base buffer boundary.
        let cut = [
            Job::G1 {
                bases: &c1,
                base_off: 0,
            },
            Job::G2 {
                bases: &c2,
                base_off: 0,
            },
        ];
        let cut_off = [Job::G1 {
            bases: &c1,
            base_off: off as u32,
        }];
        let gs = [
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: 0,
                n,
                jobs: &cut,
                work: Work::Constant,
            },
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: off as u32,
                n: n - off as u32,
                jobs: &cut_off,
                work: Work::Constant,
            },
        ];
        let got =
            pollster::block_on(batch.run_with_binding_limit(b, None, &gs, LIMIT)).expect("run");
        check("cut into sub-MSMs", &got);
        let (s1, s2) = batch.last_sub_msms();
        assert!(
            s1 >= 4 && s2 >= 2,
            "{label}: {s1} G1 and {s2} G2 sub-MSMs, nothing was cut"
        );

        // One window per submission.
        let limit = b.granted_limits().max_storage_buffer_binding_size;
        let gs = [
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: 0,
                n,
                jobs: &whole,
                work: Work::Constant,
            },
            Group {
                scalars: Source::Host(&scalars),
                scalar_off: off as u32,
                n: n - off as u32,
                jobs: &off_job,
                work: Work::Constant,
            },
        ];
        let got = pollster::block_on(batch.run_with_limits(b, None, &gs, limit, 1.0)).expect("run");
        check("one window per submission", &got);
        let fewest = 3 * g16_wgpu::msm::RECODE_BITS.div_ceil(g16_wgpu::msm::MAX_WINDOW);
        assert!(
            batch.last_submits() >= fewest,
            "{label}: {} submits is not one per window",
            batch.last_submits()
        );
        eprintln!(
            "  {label:<24} whole, {s1}+{s2} sub-MSMs and {} submissions all match the CPU",
            batch.last_submits()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. A whole proof
// ---------------------------------------------------------------------------

/// The constant-work prover proves every artifact, and at pinned blinders its proof is the
/// variable one bit for bit: the two paths compute the same five points by different
/// routes, so any drift between them is a wrong MSM, not a different proof.
#[test]
fn a_constant_work_wgpu_proof_is_the_variable_one() {
    let _one_at_a_time = exclusive();
    for_each_artifact("a_constant_work_wgpu_proof_is_the_variable_one", |a| {
        let witness = a.witness();
        let (r, s) = (Fr::from(31337u64), Fr::from(4242u64));
        let mut t = StageTimings::default();
        let constant = constant_prover()
            .prepare(a.key())
            .expect("constant prepare");
        let proof =
            prove_with_blinders(constant.as_ref(), &witness, r, s, &mut t).expect("constant");
        verify(&a.vkey(), &public_of(&witness, constant.n_public()), &proof)
            .unwrap_or_else(|e| panic!("{}: {e}", a.name));
        let variable = prover().prepare(a.key()).expect("variable prepare");
        let want =
            prove_with_blinders(variable.as_ref(), &witness, r, s, &mut t).expect("variable");
        assert_eq!(proof.a, want.a, "{}: pi_a", a.name);
        assert_eq!(proof.b, want.b, "{}: pi_b", a.name);
        assert_eq!(proof.c, want.c, "{}: pi_c", a.name);
        assert!(!proof.a.infinity && !proof.b.infinity && !proof.c.infinity);
    });
}

// ---------------------------------------------------------------------------
// 3. Geometry
// ---------------------------------------------------------------------------

/// `Work::Constant`'s promise, checked where it is decided and where it is spent: a plan
/// over a witness of bits and a plan over a dense one have the same shape and the same
/// fold tree, the digit pipeline fills every window's entry region exactly, and a whole
/// batch over the two makes the same submissions, dispatches, workgroups, sub-MSMs and
/// readback. The variable path over the same two witnesses is shown to differ, so the
/// check is known to bite.
#[test]
fn constant_work_geometry_ignores_the_witness() {
    use std::sync::atomic::Ordering::Relaxed;
    let _one_at_a_time = exclusive();
    const N: usize = 2 * 4096 + 5;
    let b = device();
    let batch = prover().msm();
    let d = batch.digits();
    let (g1, g2) = bases_with_infinity(N);
    let b1 = G1Bases::upload(b, &g1).expect("g1 bases");
    let b2 = G2Bases::upload(b, &g2).expect("g2 bases");
    let n = N as u32;
    let sparse: Vec<Fr> = (0..N).map(|i| Fr::from((i % 2) as u64)).collect();
    let dense: Vec<Fr> = (0..N).map(|i| -Fr::from(i as u64 * 7919 + 3)).collect();

    let mut seen = Vec::new();
    for scalars in [&sparse, &dense] {
        // The plans, and everything sized from them.
        let dplan = DigitPlan::with_work(n, 0, None, Work::Constant).expect("digit plan");
        let p1 = batch.g1().plan_points(&dplan, 0).expect("g1 plan");
        let p2 = batch.g2().plan_points(&dplan, 0).expect("g2 plan");
        assert_eq!(dplan.dummy_rows(), DUMMY_ROWS);
        assert_eq!(p1.ones_groups(), 0);
        assert!(p1.fold_levels().len() >= 2, "{:?}", p1.fold_levels());
        let sort = DigitBuffers::new(b, &dplan, 0).expect("sort buffers");
        let pts1 = PointBuffers::new(b, &dplan, &p1, 0, batch.g1().curve()).expect("g1 buffers");
        let pts2 = PointBuffers::new(b, &dplan, &p2, 0, batch.g2().curve()).expect("g2 buffers");
        let shape = (
            dplan,
            p1,
            p2,
            p1.fold_levels(),
            sort.bytes(),
            pts1.bytes(),
            pts2.bytes(),
            dplan.largest_binding(),
        );

        // The digit pipeline, then the histogram it built: every scalar is in every window
        // exactly once, in a real row or a dummy one, and the region is full.
        let (words, _) = pack_scalars(scalars);
        let sbuf = storage_words(b, "constant scalars", &words);
        let mut ring = ParamRing::new(b, "constant ring", d.sort_slots(&dplan)).expect("ring");
        let soff = d.plan_sort(&dplan, &mut ring).expect("plan sort");
        ring.flush(b);
        let sbind = d
            .bind_sort(b, &ring, &dplan, &sbuf, &sort)
            .expect("bind sort");
        submit_sealed(b, "the constant-work sort", |pass| {
            d.encode_sort(pass, &dplan, &sbind, &soff).expect("encode");
        });
        let counts = read_words(b, &sort.counts);
        let cursor = read_words(b, &sort.cursor);
        let (nw, nb) = (dplan.n_windows() as usize, dplan.n_buckets() as usize);
        let real = nw * nb;
        let dr = DUMMY_ROWS as usize;
        let mut in_real = 0usize;
        for w in 0..nw {
            let real_w: u32 = counts[w * nb..(w + 1) * nb].iter().sum();
            let dummy_w: u32 = counts[real + w * dr..real + (w + 1) * dr].iter().sum();
            assert_eq!(real_w as usize + dummy_w as usize, N, "window {w}");
            assert_eq!(
                cursor[real + (w + 1) * dr - 1] as usize,
                (w + 1) * N,
                "window {w}'s region is not full"
            );
            in_real += real_w as usize;
        }

        // The whole batch, counted every way the device and the batch report it.
        let jobs = [
            Job::G1 {
                bases: &b1,
                base_off: 0,
            },
            Job::G2 {
                bases: &b2,
                base_off: 0,
            },
        ];
        let gs = [Group {
            scalars: Source::Host(scalars),
            scalar_off: 0,
            n,
            jobs: &jobs,
            work: Work::Constant,
        }];
        // Once to allocate the pools, so the counted run is a proof's own.
        pollster::block_on(batch.run(b, None, &gs)).expect("warm");
        let before = (
            b.submits(),
            g16_wgpu::DISPATCHES.load(Relaxed),
            g16_wgpu::WORKGROUPS.load(Relaxed),
        );
        pollster::block_on(batch.run(b, None, &gs)).expect("run");
        let counted = (
            b.submits() - before.0,
            g16_wgpu::DISPATCHES.load(Relaxed) - before.1,
            g16_wgpu::WORKGROUPS.load(Relaxed) - before.2,
            batch.last_submits(),
            batch.last_sub_msms(),
            batch.last_readback_bytes(),
        );
        seen.push((shape, counted, in_real));
    }
    assert_eq!(seen[0].0, seen[1].0, "the plan shape follows the witness");
    assert_eq!(
        seen[0].1, seen[1].1,
        "the dispatch geometry follows the witness"
    );
    // What did change is where the entries went, which is the one thing Pippenger cannot
    // hide.
    assert_ne!(seen[0].2, seen[1].2);
    eprintln!(
        "  {:?}: {} submits, {} dispatches, {} workgroups over {:?} sub-MSMs, fold levels {:?}",
        seen[0].0 .0, seen[0].1 .0, seen[0].1 .1, seen[0].1 .2, seen[0].1 .4, seen[0].0 .3
    );

    let general = |xs: &[Fr]| xs.iter().filter(|x| !(x.is_zero() || x.is_one())).count() as u32;
    let sparse_v = DigitPlan::new(n, 0, Some(general(&sparse))).expect("plan");
    let dense_v = DigitPlan::new(n, 0, Some(general(&dense))).expect("plan");
    assert_ne!(
        sparse_v, dense_v,
        "the variable path should size its plan from the witness"
    );
}

// ---------------------------------------------------------------------------
// 4. The retry, in this mode
// ---------------------------------------------------------------------------

/// `tests/proof.rs::a_submission_cut_short_at_every_dispatch_is_retried_exactly` in
/// constant-work mode, on one small artifact: the fold levels and the dummy rows are new
/// dispatches and new buffers a half-run pass can leave behind, and a retry over them has
/// to give the unfaulted proof bit for bit. `js_1x1_d8` because its fold has four levels
/// where `tiny_mul`'s has one; the smallest artifact otherwise. Under the cross-process
/// lock, so a real kill is not a retry this test did not plan.
#[test]
fn a_constant_work_submission_cut_short_at_every_dispatch_is_retried_exactly() {
    use std::sync::atomic::Ordering::Relaxed;
    let _one_at_a_time = exclusive();
    let _lock = gpulock::exclusive_gpu();
    let found = artifacts();
    let Some(a) = found
        .iter()
        .find(|a| a.name == "js_1x1_d8")
        .or_else(|| found.iter().min_by_key(|a| a.witness().len()))
    else {
        eprintln!("SKIPPED a_constant_work_submission_cut_short_at_every_dispatch: no artifacts");
        return;
    };
    eprintln!(
        "a_constant_work_submission_cut_short_at_every_dispatch_is_retried_exactly: {}",
        a.name
    );
    let circuit = constant_prover().prepare(a.key()).expect("prepare");
    let witness = a.witness();
    let vkey = a.vkey();
    let public = public_of(&witness, circuit.n_public());
    let (r, s) = (Fr::from(7u64), Fr::from(11u64));
    let mut t = StageTimings::default();
    prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).expect("warm");
    let want = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).expect("clean");
    let msm = constant_prover().msm().last_submits();

    let mut points = 0u32;
    for sub in 0..=msm {
        let mut k = 0u32;
        loop {
            let before = g16_wgpu::CUTS.load(Relaxed);
            g16_wgpu::CUT_NEXT.store((u64::from(sub) << 32) | u64::from(k + 1), Relaxed);
            let got = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t);
            let left = g16_wgpu::CUT_NEXT.swap(0, Relaxed);
            assert_eq!(left, 0, "{}: submission {sub} never closed", a.name);
            let cuts = g16_wgpu::CUTS.load(Relaxed) - before;
            if cuts == 0 {
                break;
            }
            assert_eq!(cuts, 1, "{}: one cut asked for, {cuts} made", a.name);
            let got = got.unwrap_or_else(|e| {
                panic!(
                    "{}: submission {sub} cut after {k} dispatches did not recover: {e}",
                    a.name
                )
            });
            assert_eq!(
                (got.a, got.b, got.c),
                (want.a, want.b, want.c),
                "{}: submission {sub} cut after {k} dispatches, the retry proved something else",
                a.name
            );
            verify(&vkey, &public, &got).expect("verify");
            k += 1;
        }
        assert!(k >= 2, "{}: submission {sub} was cut at {k} points", a.name);
        points += k;
    }
    eprintln!(
        "  {}: {} submissions cut at each of their {points} points, retried exactly",
        a.name,
        msm + 1
    );
}

// ---------------------------------------------------------------------------
// 5. What the mode leaves to occupancy, phase by phase
// ---------------------------------------------------------------------------

/// What `Work::Constant` leaves to bucket occupancy, phase by phase: one constant-work MSM
/// in each group over the same bases, scalars all zero, bits, and dense random, each
/// phase in its own sealed submission. `G16_REVIEW_N` sets the length (2^18 by default,
/// H's length on keccak256 and js_16x16_d32) and `G16_WGPU_MSM_FOLD` the fold length.
///
/// The counterpart of `g16-metal`'s `constant_work_phase_occupancy`. Its numbers on this
/// M2 Max are in the commit that added the mode.
#[test]
#[ignore = "GPU measurement; run explicitly on the measurement machine"]
fn constant_work_phase_occupancy() {
    let _one_at_a_time = exclusive();
    let _lock = gpulock::exclusive_gpu();
    let n: usize = std::env::var("G16_REVIEW_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let b = device();
    let batch = prover().msm();
    let d = batch.digits();
    let (g1, g2) = bases_with_infinity(n);
    let b1 = G1Bases::upload(b, &g1).expect("g1 bases");
    let b2 = G2Bases::upload(b, &g2).expect("g2 bases");
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let bits: Vec<Fr> = (0..n).map(|_| Fr::from(xorshift(&mut seed) & 1)).collect();
    let dense = random_scalars(n, &mut seed);
    let sets = [
        ("zeros", vec![Fr::zero(); n]),
        ("bits", bits),
        ("dense", dense),
    ];

    /// One sealed submission of `encode`, run again if the GPU cut it short, in ms.
    fn timed(b: &WgpuBackend, encode: impl Fn(&mut wgpu::ComputePass<'_>)) -> f64 {
        let t = std::time::Instant::now();
        submit_sealed(b, "occupancy probe", encode);
        t.elapsed().as_secs_f64() * 1e3
    }

    for group in ["g1", "g2"] {
        for (label, scalars) in &sets {
            let dplan = DigitPlan::with_work(n as u32, 0, None, Work::Constant).expect("plan");
            let (words, _) = pack_scalars(scalars);
            let sbuf = storage_words(b, "probe scalars", &words);
            let sort = DigitBuffers::new(b, &dplan, 0).expect("sort buffers");
            // The same five phases through either curve's module, so the loop below is
            // written once.
            macro_rules! probe {
                ($p:expr, $bases:expr) => {{
                    let p = $p;
                    let pplan = p.plan_points(&dplan, 0).expect("point plan");
                    let pts = PointBuffers::new(b, &dplan, &pplan, 0, p.curve()).expect("buffers");
                    let slots = d.sort_slots(&dplan) + p.slots(&dplan, &pplan);
                    let mut ring = ParamRing::new(b, "probe ring", slots).expect("ring");
                    let soff = d.plan_sort(&dplan, &mut ring).expect("plan sort");
                    let poff = p.plan(&dplan, &pplan, &mut ring).expect("plan points");
                    ring.flush(b);
                    let sbind = d.bind_sort(b, &ring, &dplan, &sbuf, &sort).expect("bind sort");
                    let pbind = p
                        .bind_all(b, &ring, &dplan, &pplan, &sbuf, $bases.chunk(0), &sort, &pts)
                        .expect("bind points");
                    let mut t: [Vec<f64>; 5] = Default::default();
                    for rep in 0..12 {
                        let x = [
                            timed(b, |pass| d.encode_sort(pass, &dplan, &sbind, &soff).unwrap()),
                            timed(b, |pass| p.encode_clear(pass, &dplan, &pbind, &poff).unwrap()),
                            timed(b, |pass| {
                                p.encode_segmented(pass, &pplan, &pbind, &poff).unwrap()
                            }),
                            timed(b, |pass| p.encode_merge(pass, &dplan, &pbind, &poff).unwrap()),
                            timed(b, |pass| p.encode_reduce(pass, &dplan, &pbind, &poff).unwrap()),
                        ];
                        if rep >= 3 {
                            for (v, x) in t.iter_mut().zip(x) {
                                v.push(x);
                            }
                        }
                    }
                    let [dg, c, a, g, r] = t.map(median);
                    println!(
                        "occupancy {group} {label:<5} n={n} c={} w={} fold {} levels of {} | \
                         digits {dg:.2} clear {c:.2} accum {a:.2} merge {g:.2} reduce {r:.2} ms",
                        dplan.c(),
                        dplan.n_windows(),
                        pplan.fold_levels().len(),
                        pplan.fold_len(),
                    );
                }};
            }
            if group == "g1" {
                probe!(batch.g1(), b1);
            } else {
                probe!(batch.g2(), b2);
            }
        }
    }
}
