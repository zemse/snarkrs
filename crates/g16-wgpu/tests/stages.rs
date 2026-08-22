//! U7's acceptance tests: stages 0 to 4 end to end on a real GPU, against
//! `g16_core::cpu::CpuCircuit::compute_h`.
//!
//! # What is being pinned
//!
//! 1. **`compute_h` matches the CPU backend elementwise on every artifact, through both the
//!    fused epilogue and the standalone `h_join` kernel.** Two paths because the fused one
//!    is inside the NTT store and cannot be checked against itself; `h_join` shares the `Fr`
//!    prelude with it and nothing else.
//! 2. **Stages 0 to 4 are exactly one submit.** Counted through
//!    `WgpuBackend::submit`, not asserted in a comment. Design §3 measures an empty submit
//!    plus its fence at 0.1 to 0.3 ms against 2 to 3 microseconds for an extra dispatch, so
//!    19 submits instead of one would be 1.9 to 5.7 ms of pure latency at 2^18.
//! 3. **`h_std` really is standard form and `h_mont` really is Montgomery.** Each is
//!    compared against its own host encoder, `PackedScalar::from_fr` and `PackedFr::from_fr`,
//!    and the test additionally checks that a swap would have been caught: the two encodings
//!    have to actually differ at most indices, which they do by a factor of R.
//! 4. **The pooled scratch is checked out for the life of the handle and returns on drop**,
//!    which is what makes `compute_h` safe to call concurrently on one circuit. That claim
//!    is in `PreparedCircuit`'s own doc comment, so it is tested rather than assumed.
//!
//! # Rules the inputs follow
//!
//! **Never symmetric.** The synthetic key's A and B differ in row length, signal and value,
//! and the witness entries are all distinct, so an operand swap between A and B, or a
//! `h = a*a - c`, has to change a number. The last round's Fq2 bug survived because every
//! test input had `a == b`.
//!
//! **Never trivially zero.** Every test asserts the oracle's `H` is nowhere zero before
//! comparing anything. `H` on the coset is generically nonzero at every index, unlike the
//! gather, where half the rows are legitimately zero and a kernel that wrote nothing would
//! agree with the oracle on them. That property is what makes a freshly created (zeroed)
//! output buffer a usable sentinel here, so no separate sentinel fill is needed.
//!
//! **Fixed seeds**, so a failure reproduces.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use g16_core::cpu::CpuCircuit;
use g16_core::{HPoly, PreparedCircuit as _, StageTimings};
use g16_field::{AffineRepr as _, Fr, G1Affine, G2Affine, Zero as _};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::{PackedScalar, LIMBS};
use g16_wgpu::gather::{fr_words, upload_fr};
use g16_wgpu::gen::pointwise as wgsl;
use g16_wgpu::stages::TAG;
use g16_wgpu::{
    HJoin, HStages, LimitsProfile, ParamRing, Readback, Stage4, WgpuBackend, WgpuHandle,
};
use g16_zkey::{wtns::Witness, Coefficients, ProvingKey, VerifyingKey};

// ---------------------------------------------------------------------------
// Device, built once for the whole binary
// ---------------------------------------------------------------------------

fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

/// Serialises every test in this binary against the one device they share.
///
/// Three tests here assert on `WgpuBackend::submits()`, and that counter lives on the
/// backend, not on the call. `floor()` is a process-wide `OnceLock`, so under the default
/// (parallel) `cargo test` another test's proof is counted into theirs and
/// `compute_h_matches_the_cpu_backend_on_every_artifact`,
/// `compute_h_matches_the_cpu_backend_at_every_small_domain` and
/// `stages_zero_to_four_are_exactly_one_submit` fail outright: `cargo test -p g16-wgpu` was
/// red and only `--test-threads=1` was green, which nothing in the file said and nothing
/// enforced. This is what enforces it.
///
/// A poisoned lock is taken anyway. The test that panicked has already failed; making every
/// later test fail as well would bury the first message.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn artifacts() -> Vec<(String, PathBuf)> {
    let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
    else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Running the device path
// ---------------------------------------------------------------------------

/// One `compute_h`, read back in both encodings, with the submit count it cost.
struct Run {
    mont: Vec<u32>,
    std: Vec<u32>,
    /// A's coset evaluations, read only when a caller asks. The one case that needs them is
    /// a domain where `H` is structurally zero and every other comparison is therefore
    /// vacuous; see the 2^0 branch below.
    coset_a: Option<Vec<u32>>,
    submits: u64,
    timings: StageTimings,
}

fn run(stages: &HStages, witness: &[Fr], mode: Stage4) -> Run {
    run_inner(stages, witness, mode, false)
}

fn run_inner(stages: &HStages, witness: &[Fr], mode: Stage4, want_coset: bool) -> Run {
    let b = floor();
    let mut timings = StageTimings::default();
    let before = b.submits();
    let h = pollster::block_on(stages.compute_h_with(b, witness, mode, &mut timings))
        .expect("compute_h failed");
    // Counted before any readback, because a readback is a submit of its own and would
    // otherwise be charged to stages 0 to 4.
    let submits = b.submits() - before;

    assert_eq!(h.len(), stages.domain_size(), "H is the wrong length");
    assert!(
        h.to_host().is_none(),
        "compute_h returned a host vector; the whole point of HPoly::Device is that H \
         never crosses the bus"
    );
    let handle: &WgpuHandle = h
        .device_handle(TAG)
        .expect("the handle is not a WgpuHandle under the wgpu tag");

    let mont = pollster::block_on(handle.h_mont_words(b)).expect("h_mont readback");
    let std = pollster::block_on(handle.h_std_words(b)).expect("h_std readback");
    let coset_a = want_coset.then(|| {
        let (a, _, _) = handle.coset();
        let bytes = stages.domain_size() as u64 * (LIMBS * 4) as u64;
        let rb = Readback::new(b, "coset a", bytes).unwrap();
        let mut enc = b.device().create_command_encoder(&Default::default());
        rb.copy_from(&mut enc, a, 0, bytes).unwrap();
        let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).unwrap();
        bytemuck::cast_slice::<u8, u32>(&raw).to_vec()
    });
    Run {
        mont,
        std,
        coset_a,
        submits,
        timings,
    }
}

/// Elementwise, in device words, naming the element that first disagreed.
fn compare(what: &str, got: &[u32], want: &[u32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: {} words back, oracle has {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got
        .chunks_exact(LIMBS)
        .zip(want.chunks_exact(LIMBS))
        .enumerate()
    {
        assert_eq!(
            g, w,
            "{what}: element {i} is {g:08x?} on the device, {w:08x?} on the CPU"
        );
    }
}

/// Both encodings against their own host encoder, plus the checks that make a swap of the
/// two visible.
fn check_both_encodings(what: &str, run: &Run, want: &[Fr], expect_nonzero: bool) {
    assert!(!want.is_empty(), "{what}: nothing to compare");
    if expect_nonzero {
        assert!(
            want.iter().all(|x| !x.is_zero()),
            "{what}: the oracle's H has a zero in it, so a kernel that wrote nothing would \
             agree with it somewhere and this comparison is weaker than it looks"
        );
    }

    let want_mont = fr_words(want);
    let want_std: Vec<u32> = want
        .iter()
        .flat_map(|x| PackedScalar::from_fr(x).v)
        .collect();
    compare(&format!("{what} h_mont"), &run.mont, &want_mont);
    compare(&format!("{what} h_std"), &run.std, &want_std);

    // The two encodings differ by a factor of R, so they agree only where H is 0 or where
    // R is 1, neither of which happens here. Without this the two comparisons above would
    // both pass on an implementation that wrote the same thing into both buffers, and a
    // proof from that is wrong by R and fails verification with nothing else to go on.
    let differ = want_mont
        .chunks_exact(LIMBS)
        .zip(want_std.chunks_exact(LIMBS))
        .filter(|(m, s)| m != s)
        .count();
    assert_eq!(
        differ,
        if expect_nonzero { want.len() } else { 0 },
        "{what}: the Montgomery and standard encodings coincide at {} of {} indices, so \
         this artifact cannot distinguish them",
        want.len() - differ,
        want.len()
    );
}

// ---------------------------------------------------------------------------
// 1. Every artifact, both stage 4 paths, against the CPU backend
// ---------------------------------------------------------------------------

#[test]
fn compute_h_matches_the_cpu_backend_on_every_artifact() {
    let _gpu = exclusive();
    let b = floor();
    let found = artifacts();
    assert!(
        !found.is_empty(),
        "no artifacts under bench/artifacts; the symlink into the main worktree is missing \
         and this test would otherwise pass by doing nothing"
    );

    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let domain_size = pk.domain_size;
        let n_vars = pk.n_vars;

        let stages = HStages::new(b, &pk).expect("stages 0 to 4 did not build");
        let fused = run(&stages, &w, Stage4::Fused);
        let standalone = run(&stages, &w, Stage4::Standalone);

        let cpu = CpuCircuit::new(pk).expect("cpu circuit");
        let mut cpu_t = StageTimings::default();
        let want = match cpu.compute_h(&w, &mut cpu_t).expect("cpu compute_h") {
            HPoly::Host(v) => v,
            HPoly::Device { .. } => unreachable!("the cpu backend has nowhere to put a device H"),
        };

        check_both_encodings(&format!("{name} fused"), &fused, &want, true);
        check_both_encodings(&format!("{name} standalone"), &standalone, &want, true);

        // The two device paths share the NTT and nothing else about stage 4, so byte
        // equality between them is a statement about stage 4 rather than about the
        // transform.
        assert_eq!(
            fused.mont, standalone.mont,
            "{name}: the fused epilogue and the standalone h_join disagree on h_mont"
        );
        assert_eq!(fused.std, standalone.std, "{name}: they disagree on h_std");

        // Acceptance: exactly one submit for stages 0 to 4, on both paths.
        assert_eq!(
            fused.submits, 1,
            "{name}: the fused path took {} submits, design §3 budgets one",
            fused.submits
        );
        assert_eq!(
            standalone.submits, 1,
            "{name}: the standalone path took {} submits",
            standalone.submits
        );

        println!(
            "{name}: n_vars {n_vars} domain 2^{} batches {:?}  \
             fused {} dispatches, standalone {}  \
             pack {} us, gpu {} us (fused) / {} us (standalone), cpu {} us  \
             scratch {:.1} MiB",
            domain_size.trailing_zeros(),
            stages.pass_batches(),
            stages.dispatches(Stage4::Fused),
            stages.dispatches(Stage4::Standalone),
            fused.timings.gather_us,
            fused.timings.ntt_us,
            standalone.timings.ntt_us,
            cpu_t.gather_us + cpu_t.ntt_us + cpu_t.pointwise_us,
            stages.scratch_bytes() as f64 / (1024.0 * 1024.0),
        );
    }
}

// ---------------------------------------------------------------------------
// 2. A synthetic key: small domains, an asymmetric CSR, and the paths artifacts never reach
// ---------------------------------------------------------------------------

/// A CSR shaped like a real key's but deliberately asymmetric between A and B.
///
/// Row lengths follow different patterns per matrix, including empty rows and singletons,
/// and values and signals come from one stream so no value in A appears in B. A swap of the
/// two matrices has to change the answer.
fn synthetic_coeffs(rng: &mut SplitMix64, n_rows: usize, n_vars: usize) -> Coefficients {
    let mut row_ptr = [
        Vec::with_capacity(n_rows + 1),
        Vec::with_capacity(n_rows + 1),
    ];
    let mut signal = [Vec::new(), Vec::new()];
    let mut value: [Vec<Fr>; 2] = [Vec::new(), Vec::new()];

    for m in 0..2 {
        row_ptr[m].push(0u32);
        for c in 0..n_rows {
            // Different length patterns per matrix, both including zero-length rows further
            // in. Row 0 is forced nonempty in both, and to different lengths, so that a
            // one-row domain still has a nonzero H to compare and still cannot survive a
            // swap of the two matrices.
            let len = match (m, c) {
                (0, 0) => 2,
                (1, 0) => 3,
                (0, _) => c % 4,
                (_, _) => (c % 3) + (c % 5) / 4,
            };
            for _ in 0..len {
                signal[m].push((rng.0.wrapping_mul(0x9e37_79b9) as usize % n_vars) as u32);
                rng.0 = rng.0.wrapping_add(0x1234_5678_9abc_def1);
                value[m].push(rng.next_fr());
            }
            row_ptr[m].push(signal[m].len() as u32);
        }
    }
    Coefficients {
        row_ptr,
        signal,
        value,
    }
}

/// A proving key carrying only what stages 0 to 4 read.
///
/// The five base vectors are left empty: nothing in `HStages` or in
/// `CpuCircuit::compute_h` touches them, and filling them with 2^18 dummy points would
/// cost more than everything else this test does. `msms` would reject this key, which is
/// correct, because this key cannot produce a proof.
fn synthetic_key(rng: &mut SplitMix64, domain_size: usize, n_vars: usize) -> ProvingKey {
    ProvingKey {
        n_vars,
        n_public: 1.min(n_vars.saturating_sub(1)),
        domain_size,
        alpha_g1: G1Affine::zero(),
        beta_g1: G1Affine::zero(),
        beta_g2: G2Affine::zero(),
        delta_g1: G1Affine::zero(),
        delta_g2: G2Affine::zero(),
        a_query: Vec::new(),
        b_g1_query: Vec::new(),
        b_g2_query: Vec::new(),
        l_query: Vec::new(),
        h_query: Vec::new(),
        coeffs: synthetic_coeffs(rng, domain_size, n_vars),
        vk: VerifyingKey {
            alpha_g1: G1Affine::zero(),
            beta_g2: G2Affine::zero(),
            gamma_g2: G2Affine::zero(),
            delta_g2: G2Affine::zero(),
            ic: Vec::new(),
        },
    }
}

/// A witness whose entries are all distinct, so an index bug cannot alias.
fn witness(rng: &mut SplitMix64, n: usize) -> Vec<Fr> {
    let w: Vec<Fr> = (0..n).map(|_| rng.next_fr()).collect();
    for i in 1..w.len() {
        assert_ne!(w[i], w[i - 1], "witness entries collided");
    }
    w
}

#[test]
fn compute_h_matches_the_cpu_backend_at_every_small_domain() {
    let _gpu = exclusive();
    let b = floor();
    // From 2^0 up. A one-point domain is `split_passes(0, _) == [(s0 0, k 0)]`, so it is
    // the only thing that ever compiles and dispatches the `k = 0` entry points, which U6
    // generated and left with no caller. 2^9 is the first domain that needs two batches at
    // the shipped tile of 8, so the joined tail runs too.
    for log_n in [0u32, 1, 2, 3, 4, 7, 8, 9, 12] {
        let n = 1usize << log_n;
        let n_vars = (n * 2 + 7).max(3);
        let mut rng = SplitMix64(0x0007_0000 ^ u64::from(log_n));
        let pk = synthetic_key(&mut rng, n, n_vars);
        let w = witness(&mut rng, n_vars);

        // Asymmetry, restated from the data rather than trusted from the generator.
        let nnz = |m: usize| *pk.coeffs.row_ptr[m].last().unwrap();
        if log_n >= 2 {
            assert_ne!(nnz(0), nnz(1), "2^{log_n}: A and B have the same nnz");
        }

        // At 2^0 the coset is one point and A, B and C are constants, so
        // `H = A(x)B(x) - C(x)` is identically zero for any witness: C is exactly A*B by
        // construction and a degree-0 product is still degree 0. Every H comparison is
        // therefore vacuous there, and the thing worth asserting instead is that the
        // pipeline ran at all, which the coset vector says. From 2^1 up, A*B has degree 2
        // where C has degree 1, so H is generically nonzero and the normal checks bite.
        let degenerate = log_n == 0;

        let stages = HStages::new(b, &pk).expect("stages 0 to 4 did not build");
        let fused = run_inner(&stages, &w, Stage4::Fused, degenerate);
        let standalone = run(&stages, &w, Stage4::Standalone);

        let cpu = CpuCircuit::new(pk).expect("cpu circuit");
        let mut t = StageTimings::default();
        let want = match cpu.compute_h(&w, &mut t).unwrap() {
            HPoly::Host(v) => v,
            HPoly::Device { .. } => unreachable!(),
        };
        assert_eq!(
            want.iter().all(|x| x.is_zero()),
            degenerate,
            "2^{log_n}: the oracle's H is not the shape this test expects"
        );

        check_both_encodings(&format!("2^{log_n} fused"), &fused, &want, !degenerate);
        check_both_encodings(
            &format!("2^{log_n} standalone"),
            &standalone,
            &want,
            !degenerate,
        );
        if degenerate {
            let a = fused.coset_a.as_ref().expect("coset A was not read");
            assert!(
                a.iter().any(|&x| x != 0),
                "2^{log_n}: A's coset evaluation is all zero, so nothing ran and the \
                 all-zero H above proves nothing"
            );
        }
        assert_eq!(fused.mont, standalone.mont, "2^{log_n}: paths disagree");
        assert_eq!(fused.submits, 1);
        assert_eq!(standalone.submits, 1);
        println!(
            "2^{log_n}: {n} rows, {n_vars} vars, batches {:?}, {} dispatches fused",
            stages.pass_batches(),
            stages.dispatches(Stage4::Fused)
        );
    }
}

/// Keys with no coefficients: B empty, then A and B both empty.
///
/// **The second case is why `CsrTables::upload` pads an `Fr` buffer to 32 bytes.** A 2^0
/// synthetic key with no nonzeros anywhere used to fail bind-group creation outright, "the
/// buffer bound at binding index 3 is bound with size 4 where the shader expects 32",
/// because WebGPU sizes a runtime-sized array binding by its element stride and
/// `storage_u32` pads an empty vector to one *word*. Every dispatch in the encoder was
/// dropped.
///
/// The first case does **not** reach it, and that is worth writing down because the first
/// draft of this test only had that case and the mutation reverting the fix passed. A and B
/// are concatenated into one `value` array, so it is empty only when *both* matrices are,
/// and "the B matrix is empty" leaves A's coefficients in the buffer.
///
/// These are binding tests, not arithmetic ones. With B zero, C is zero and H is zero, so
/// comparing zero against zero says nothing about the field. What each case asserts is what
/// it can: case one that A's coset evaluation is not zero, so the dispatches really ran; case
/// two that `compute_h` returns `Ok` at all, since the failure it guards against is a
/// validation error that `WgpuBackend::take_error` turns into an `Err`.
#[test]
fn a_key_with_no_coefficients_still_binds() {
    let _gpu = exclusive();
    let b = floor();
    for empty_a in [false, true] {
        let mut rng = SplitMix64(0x00e3_0000 ^ u64::from(empty_a));
        let (n, n_vars) = (64usize, 40usize);
        let mut pk = synthetic_key(&mut rng, n, n_vars);
        pk.coeffs.row_ptr[1] = vec![0u32; n + 1];
        pk.coeffs.signal[1].clear();
        pk.coeffs.value[1].clear();
        if empty_a {
            pk.coeffs.row_ptr[0] = vec![0u32; n + 1];
            pk.coeffs.signal[0].clear();
            pk.coeffs.value[0].clear();
        }
        let nnz_a = pk.coeffs.value[0].len();
        assert_eq!(
            nnz_a == 0,
            empty_a,
            "the key is not the shape this case wants"
        );
        let w = witness(&mut rng, n_vars);

        let stages = HStages::new(b, &pk).expect("an empty coefficient section refused to build");
        // `Ok` here *is* the assertion for the both-empty case: without the 32-byte pad this
        // is `Err("device error in stages 0 to 4: Validation Error ... bound with size 4
        // where the shader expects 32")`.
        let run = run_inner(&stages, &w, Stage4::Fused, true);

        let cpu = CpuCircuit::new(pk).expect("cpu circuit");
        let want = match cpu
            .compute_h(&w, &mut StageTimings::default())
            .expect("cpu compute_h")
        {
            HPoly::Host(v) => v,
            HPoly::Device { .. } => unreachable!(),
        };
        assert!(want.iter().all(|x| x.is_zero()), "H should be zero here");
        check_both_encodings(
            if empty_a { "both empty" } else { "empty B" },
            &run,
            &want,
            false,
        );

        let a = run.coset_a.as_ref().unwrap();
        assert_eq!(
            a.iter().all(|&x| x == 0),
            empty_a,
            "A's coset evaluation is {} and the key says it should be the other way",
            if empty_a { "nonzero" } else { "all zero" }
        );
        println!(
            "{}: {n} rows, A has {nnz_a} nonzeros, one submit and no device error",
            if empty_a { "A and B empty" } else { "B empty" }
        );
    }
}

// ---------------------------------------------------------------------------
// 3. One submit, on its own, and what a rejected witness costs
// ---------------------------------------------------------------------------

#[test]
fn stages_zero_to_four_are_exactly_one_submit() {
    let _gpu = exclusive();
    let b = floor();
    let mut rng = SplitMix64(0x0517_0000);
    // 2^10 needs two NTT batches at the shipped tile, so this counts the multi-dispatch
    // shape and not the degenerate single-batch one.
    let (n, n_vars) = (1024usize, 700usize);
    let pk = synthetic_key(&mut rng, n, n_vars);
    let w = witness(&mut rng, n_vars);
    let stages = HStages::new(b, &pk).expect("stages");

    assert!(
        stages.dispatches(Stage4::Fused) > 1,
        "this test is meant to count a multi-dispatch encode"
    );

    for mode in [Stage4::Fused, Stage4::Standalone] {
        let mut t = StageTimings::default();
        let before = b.submits();
        let h = pollster::block_on(stages.compute_h_with(b, &w, mode, &mut t)).expect("compute_h");
        let after = b.submits();
        assert_eq!(
            after - before,
            1,
            "{mode:?}: {} dispatches went out in {} submits, design §3 budgets one",
            stages.dispatches(mode),
            after - before
        );
        drop(h);
    }

    // A rejected witness must not reach the queue at all: the length check is the one thing
    // standing between a short witness and an out-of-range read that WGSL answers with zero.
    let before = b.submits();
    let err = match pollster::block_on(stages.compute_h(
        b,
        &w[..n_vars - 1],
        &mut StageTimings::default(),
    )) {
        Ok(_) => panic!("a short witness was accepted"),
        Err(e) => e,
    };
    assert_eq!(b.submits(), before, "a rejected witness still submitted");
    assert!(
        format!("{err}").contains("witness"),
        "the error does not name the witness: {err}"
    );
    println!(
        "2^10: {} dispatches fused / {} standalone, one submit each; a short witness \
         submits nothing",
        stages.dispatches(Stage4::Fused),
        stages.dispatches(Stage4::Standalone)
    );
}

// ---------------------------------------------------------------------------
// 4. The pool, which is what makes concurrent proving safe
// ---------------------------------------------------------------------------

#[test]
fn the_pooled_scratch_is_exclusive_per_handle_and_returns_on_drop() {
    let _gpu = exclusive();
    let b = floor();
    let mut rng = SplitMix64(0x9001);
    let (n, n_vars) = (256usize, 200usize);
    let pk = synthetic_key(&mut rng, n, n_vars);
    let w1 = witness(&mut rng, n_vars);
    let w2 = witness(&mut rng, n_vars);
    assert!(
        w1.iter().zip(&w2).all(|(x, y)| x != y),
        "witnesses collided"
    );

    let cpu = CpuCircuit::new(synthetic_key(&mut SplitMix64(0x9001), n, n_vars)).unwrap();
    let host = |w: &[Fr]| match cpu.compute_h(w, &mut StageTimings::default()).unwrap() {
        HPoly::Host(v) => v,
        HPoly::Device { .. } => unreachable!(),
    };
    let want1 = host(&w1);
    let want2 = host(&w2);

    let stages = HStages::new(b, &pk).expect("stages");
    assert_eq!(stages.pooled(), 0, "the pool starts empty");

    // Two handles alive at once: they must not be handed the same scratch, or the second
    // proof would overwrite the first's H while the first still owns it.
    let mut t = StageTimings::default();
    let h1 = pollster::block_on(stages.compute_h(b, &w1, &mut t)).unwrap();
    assert_eq!(stages.pooled(), 0, "the scratch went back while in use");
    let h2 = pollster::block_on(stages.compute_h(b, &w2, &mut t)).unwrap();
    assert_eq!(stages.pooled(), 0, "two live handles, nothing pooled");

    // And they hold different data, which is the observable form of "different scratch".
    let g1 = h1.device_handle::<WgpuHandle>(TAG).unwrap();
    let g2 = h2.device_handle::<WgpuHandle>(TAG).unwrap();
    let got1 = pollster::block_on(g1.h_mont_words(b)).unwrap();
    let got2 = pollster::block_on(g2.h_mont_words(b)).unwrap();
    compare("first handle", &got1, &fr_words(&want1));
    compare("second handle", &got2, &fr_words(&want2));

    drop(h1);
    assert_eq!(
        stages.pooled(),
        1,
        "dropping a handle did not return its scratch"
    );
    drop(h2);
    assert_eq!(stages.pooled(), 2);

    // A third proof reuses a pooled set. Everything the kernels write is written in full,
    // so the previous proof's contents must not survive into this one.
    let h3 = pollster::block_on(stages.compute_h(b, &w1, &mut t)).unwrap();
    assert_eq!(
        stages.pooled(),
        1,
        "the third proof allocated instead of reusing"
    );
    let g3 = h3.device_handle::<WgpuHandle>(TAG).unwrap();
    compare(
        "reused scratch",
        &pollster::block_on(g3.h_mont_words(b)).unwrap(),
        &fr_words(&want1),
    );
    drop(h3);
    assert_eq!(stages.pooled(), 2);
    println!("pool: 0 -> 0 (two live) -> 2 (both dropped) -> reused, {n} points");
}

#[test]
fn one_circuit_computes_h_concurrently() {
    let _gpu = exclusive();
    let b = floor();
    let mut rng = SplitMix64(0x00c0_11de);
    let (n, n_vars) = (512usize, 400usize);
    let pk = synthetic_key(&mut rng, n, n_vars);
    let ws: Vec<Vec<Fr>> = (0..4).map(|_| witness(&mut rng, n_vars)).collect();

    let cpu = CpuCircuit::new(synthetic_key(&mut SplitMix64(0x00c0_11de), n, n_vars)).unwrap();
    let want: Vec<Vec<u32>> = ws
        .iter()
        .map(
            |w| match cpu.compute_h(w, &mut StageTimings::default()).unwrap() {
                HPoly::Host(v) => fr_words(&v),
                HPoly::Device { .. } => unreachable!(),
            },
        )
        .collect();

    let stages = &HStages::new(b, &pk).expect("stages");
    // Four threads against one `HStages`, which is exactly the claim `PreparedCircuit`
    // makes about itself. If the parameter ring or the scratch were shared, each proof
    // would dispatch with another's row ranges and the results would be wrong rather than
    // merely slow.
    std::thread::scope(|s| {
        for (w, want) in ws.iter().zip(&want) {
            s.spawn(move || {
                let mut t = StageTimings::default();
                let h = pollster::block_on(stages.compute_h(b, w, &mut t)).expect("compute_h");
                let g = h.device_handle::<WgpuHandle>(TAG).unwrap();
                let got = pollster::block_on(g.h_mont_words(b)).unwrap();
                compare("concurrent", &got, want);
            });
        }
    });
    println!(
        "4 concurrent proofs on one circuit over {n} points, {} scratch sets pooled after",
        stages.pooled()
    );
}

// ---------------------------------------------------------------------------
// 4b. Does the fusion actually pay? It does not, and that is design §4's claim
// ---------------------------------------------------------------------------

/// One `compute_h`, returning (GPU wall time, host time) in microseconds.
fn once_us(stages: &HStages, w: &[Fr], mode: Stage4) -> (u64, u64) {
    let b = floor();
    let mut t = StageTimings::default();
    drop(pollster::block_on(stages.compute_h_with(b, w, mode, &mut t)).expect("compute_h"));
    (t.ntt_us, t.gather_us)
}

/// Median GPU and host microseconds for each mode, measured **interleaved**.
///
/// Two things this shape is defending against, both of which bit while it was written.
/// A freshly allocated scratch set is 52 MiB at 2^18 and its first touch is a page fault,
/// which is the cost the pool exists to pay once, so the first call of one mode against the
/// second call of the other measures the allocator: hence the warm-up. And a block of one
/// mode followed by a block of the other attributes any drift over the run (thermal, another
/// test contending for the GPU) entirely to the second block: hence the interleave.
fn compare_modes(stages: &HStages, w: &[Fr], reps: usize) -> [(u64, u64); 2] {
    for _ in 0..2 {
        once_us(stages, w, Stage4::Fused);
        once_us(stages, w, Stage4::Standalone);
    }
    let mut got: [Vec<(u64, u64)>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..reps {
        got[0].push(once_us(stages, w, Stage4::Fused));
        got[1].push(once_us(stages, w, Stage4::Standalone));
    }
    std::array::from_fn(|i| {
        let mut gpu: Vec<u64> = got[i].iter().map(|r| r.0).collect();
        let mut host: Vec<u64> = got[i].iter().map(|r| r.1).collect();
        gpu.sort_unstable();
        host.sort_unstable();
        (gpu[reps / 2], host[reps / 2])
    })
}

/// **Design §4 and §8 both say to fuse stage 4 into the last NTT batch, and on this machine
/// the fusion is a pessimisation.** This test is the measurement and the reason
/// [`Stage4::default`] is what it is.
///
/// The design's argument is that the fusion saves a whole write of C and a whole read of A,
/// B and C. Counting the streams the last batch of C's forward transform touches, in units
/// of one domain vector:
///
/// ```text
/// standalone   plain tail:  read DST, write DST                        2
///              h_join:      read A, B, C, write h_mont, h_std          5   total 7
/// fused        join tail:   read DST, A, B, write h_mont, h_std        5   total 5
/// ```
///
/// So the fusion moves 29% less data and is measured slower. The reason is the one thing
/// that count leaves out: **on any domain that needs more than one batch the last batch is
/// strided.** Its element index is `base + (m << s0)`, so with `s0 = 12` at 2^18 consecutive
/// threads touch addresses 128 KiB apart, and the four streams the join adds are all
/// uncoalesced where `h_join` reads and writes them contiguously. At 32 useful bytes per
/// 128-byte line that is a 4x amplification, which turns 5 strided streams (20) against
/// 2 strided plus 5 contiguous (13) and puts the fused path ahead in bytes moved and behind
/// in bytes fetched.
///
/// That is a hypothesis with the right sign and the right order of magnitude, not a proof:
/// the measured penalty on the joined batch alone is about 2.3x where the line-fetch count
/// predicts 1.5x, so something else (register pressure from inlining two more unrolled CIOS
/// multiplies into a kernel that already holds a workgroup tile) is probably also in it.
/// Nobody has isolated the two and this comment does not pretend otherwise.
///
/// **`g16-metal` has the same shape and, as far as this repo records, has never measured
/// it**: its 2^18 split is 9 + 9, so its joined batch also runs at `s0 = 9`. Filed in
/// `TASKS.md`.
#[test]
fn the_fusion_is_measured_and_not_assumed() {
    let _gpu = exclusive();
    let b = floor();
    let found = artifacts();
    assert!(!found.is_empty(), "no artifacts under bench/artifacts");

    let mut ratios: Vec<f64> = Vec::new();
    println!("artifact          domain  batches  fused gpu  alone gpu  ratio   host us  fence us");
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let log_n = pk.domain_size.trailing_zeros();
        let stages = HStages::new(b, &pk).expect("stages");
        let nb = stages.pass_batches().len();

        let [(fused, host_f), (alone, host_a)] = compare_modes(&stages, &w, 9);
        let ratio = fused as f64 / alone as f64;
        ratios.push(ratio);
        println!(
            "{name:<18}2^{log_n:<6}{nb:>7}{fused:>11}{alone:>11}{ratio:>7.3}{:>10}{:>10}",
            host_f.max(host_a),
            empty_submit_us()
        );
    }
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];

    // Self-correcting rather than a snapshot of today's answer: it asserts that the mode
    // this crate *defaults to* is the faster one, whichever that turns out to be. If Tint
    // or a later kernel change flips the ranking, this fails and the default has to move,
    // rather than the crate quietly shipping the slower path. 5% of slack, because at the
    // two smallest artifacts the whole GPU number is mostly the fence.
    let (name, ok) = match Stage4::default() {
        Stage4::Standalone => ("Standalone", median >= 0.95),
        Stage4::Fused => ("Fused", median <= 1.05),
    };
    println!(
        "median fused/standalone {median:.3}; the shipped default is {name}, \
         empty submit plus fence {} us",
        empty_submit_us()
    );
    assert!(
        ok,
        "Stage4::default() is {name} and the median fused/standalone ratio is {median:.3}, \
         so the crate defaults to the slower of the two"
    );
}

/// Microseconds for an empty submit plus its fence, so the small-domain rows above can be
/// read for what they are.
///
/// Design §3 puts this at 0.1 to 0.3 ms from an earlier scouting measurement. It is measured
/// here rather than quoted, because at 2^3 it *is* the whole number.
fn empty_submit_us() -> u64 {
    let b = floor();
    let mut us: Vec<u64> = (0..7)
        .map(|_| {
            let enc = b.device().create_command_encoder(&Default::default());
            let t0 = std::time::Instant::now();
            b.submit([enc.finish()]);
            pollster::block_on(b.wait_for_submitted_work()).unwrap();
            t0.elapsed().as_micros() as u64
        })
        .collect();
    us.sort_unstable();
    us[3]
}

// ---------------------------------------------------------------------------
// 5. The standalone kernel's own shape
// ---------------------------------------------------------------------------

#[test]
fn the_h_join_pipeline_layout_declares_at_most_eight_storage_buffers() {
    let _gpu = exclusive();
    let b = floor();
    let entries = HJoin::bind_group_layout_entries();
    let storage = HJoin::storage_buffer_count();
    let uniform = entries
        .iter()
        .filter(|e| {
            matches!(
                e.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                }
            )
        })
        .count() as u32;

    // The Floor, not this adapter's 9. A kernel sized for the ninth passes natively and
    // fails in a stock browser, which research file 01 records heliax shipping twice.
    let spec = wgpu::Limits::default();
    assert_eq!(spec.max_storage_buffers_per_shader_stage, 8);
    assert!(storage <= spec.max_storage_buffers_per_shader_stage);
    assert_eq!(storage, wgsl::STORAGE_BUFFERS);
    assert_eq!(uniform, 1);

    // Independently counted from the generated source, so a binding added to the shader
    // without a matching layout entry (or the reverse) fails here rather than at dispatch.
    let src = wgsl::h_join_module(Default::default());
    assert_eq!(src.matches("var<storage").count() as u32, storage);
    assert_eq!(src.matches("var<uniform").count() as u32, uniform);
    assert_eq!(src.matches("@group(0)").count() as u32, storage + uniform);
    // Only the Fr prelude. Fq2 in here would mean the curve field came along for the ride.
    assert!(!src.contains("fn fq2_mul("), "h_join dragged in Fq2");
    assert!(!src.contains("fn fq_mul("), "h_join dragged in Fq");

    let j = HJoin::new(b).expect("h_join does not build at the Floor");
    assert_eq!(b.granted_limits().max_storage_buffers_per_shader_stage, 8);
    println!(
        "h_join: {storage} storage + {uniform} uniform in 1 bind group, {} bytes of WGSL, {}",
        j.source_len(),
        j.kernels().summary()
    );
    println!(
        "metal counterpart binds 5 storage buffers plus a setBytes scalar \
         (pointwise.metal g16_h_join); the floor is {}",
        spec.max_storage_buffers_per_shader_stage
    );
}

/// The sentinel an oversized output buffer is filled with. All-ones is above the modulus, so
/// it is not a value any correct stage 4 can produce, and it is not the zero a freshly
/// created buffer holds either.
const SENTINEL: u32 = 0xffff_ffff;

/// `h_join` over a range, into buffers with one element of slack past the end.
///
/// **This test exists because a mutation got through without it.** Changing the kernel's
/// guard from `i >= P.hi` to `i > P.hi` passed the entire rest of this file. The reason is
/// that `HStages` allocates `h_mont` and `h_std` at exactly the domain size, so the extra
/// write lands out of range and WGSL drops it silently: an over-run is unobservable unless
/// something is bound past the end for it to land in. `crates/g16-wgpu/tests/gather.rs` and
/// `tests/ntt.rs` both learned this already and both keep a slack element; stage 4 did not
/// and had to.
///
/// 1000 elements, not 1024, so the last workgroup is partial at every workgroup size swept.
/// A round element count is what makes a bounds guard never get asked a question.
#[test]
fn the_standalone_join_writes_exactly_the_range_it_was_given() {
    let _gpu = exclusive();
    let b = floor();
    const N: u32 = 1000;
    let mut rng = SplitMix64(0x5_1ac);
    // Three different vectors. `a == b` is how the last round's Fq2 bug survived.
    // N + 1 long: the inputs have to cover the slack element too, or `HJoin::bind`'s own
    // length check refuses the oversized output binding before the kernel ever runs. Only
    // the first N are compared; the extra one is there so an over-run has real data to
    // compute a real (and wrong) value from rather than reading zeros.
    let xs: Vec<Vec<Fr>> = (0..3).map(|_| witness(&mut rng, N as usize + 1)).collect();
    assert!(
        xs[0].iter().zip(&xs[1]).all(|(p, q)| p != q)
            && xs[1].iter().zip(&xs[2]).all(|(p, q)| p != q),
        "the three inputs are not distinct"
    );
    let want: Vec<Fr> = (0..N as usize)
        .map(|i| xs[0][i] * xs[1][i] - xs[2][i])
        .collect();
    assert!(
        want.iter().all(|x| !x.is_zero()),
        "the oracle has a zero in it"
    );
    let want_mont = fr_words(&want);
    let want_std: Vec<u32> = want
        .iter()
        .flat_map(|x| PackedScalar::from_fr(x).v)
        .collect();

    let a = upload_fr(b, "join a", &xs[0]).unwrap();
    let bb = upload_fr(b, "join b", &xs[1]).unwrap();
    let c = upload_fr(b, "join c", &xs[2]).unwrap();

    // One dispatch, then two forced chunkings. The chunked path is what a 2^24 domain would
    // take and no artifact reaches it, so without this it ships untested.
    // The caps are multiples of the workgroup size, because `HJoin::with_shape` rounds a
    // cap down to a whole number of workgroups (and refuses one under a single workgroup),
    // so a cap of 1000 would quietly become 768 and a cap of 128 would be an error.
    for (cap, want_dispatches) in [(1024u32, 1u32), (512, 2), (256, 4)] {
        let j = HJoin::with_shape(b, cap, wgsl::WORKGROUP).expect("h_join pipeline");
        assert_eq!(
            j.dispatches(N),
            want_dispatches,
            "cap {cap}: the chunking is not what this test assumed"
        );

        let fill = vec![SENTINEL; (N as usize + 1) * LIMBS];
        let outs: Vec<wgpu::Buffer> = ["h_mont", "h_std"]
            .iter()
            .map(|label| {
                let buf = g16_wgpu::gather::fr_buffer(b, label, N + 1).unwrap();
                b.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&fill));
                buf
            })
            .collect();

        let mut ring = ParamRing::new(b, "join range", j.dispatches(N)).unwrap();
        let offsets = j.plan(N, &mut ring).unwrap();
        ring.flush(b);
        // Bound over `N + 1`, so the guard has somewhere to be wrong into. `HJoin::bind`
        // only requires the buffers to be at least as long as the count it is given.
        let bind = j
            .bind(b, &ring, N + 1, &a, &bb, &c, &outs[0], &outs[1])
            .expect("bind");

        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            j.encode(&mut pass, &bind, N, &offsets).expect("encode");
        }
        b.submit([enc.finish()]);
        pollster::block_on(b.wait_for_submitted_work()).unwrap();
        if let Some(e) = b.take_error() {
            panic!("cap {cap}: device error: {e}");
        }

        for (name, buf, want) in [
            ("h_mont", &outs[0], &want_mont),
            ("h_std", &outs[1], &want_std),
        ] {
            let bytes = (N as u64 + 1) * (LIMBS * 4) as u64;
            let rb = Readback::new(b, "join range", bytes).unwrap();
            let mut enc = b.device().create_command_encoder(&Default::default());
            rb.copy_from(&mut enc, buf, 0, bytes).unwrap();
            let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).unwrap();
            let words = bytemuck::cast_slice::<u8, u32>(&raw);
            let cut = N as usize * LIMBS;
            compare(&format!("cap {cap} {name}"), &words[..cut], want);
            assert!(
                words[cut..].iter().all(|&x| x == SENTINEL),
                "cap {cap} {name}: element {N} was written, so the `i >= hi` guard is off \
                 by one (slack is {:08x?})",
                &words[cut..]
            );
        }
    }
    println!("h_join over {N} elements in 1, 2 and 4 dispatches, slack untouched in all three");
}

/// Runs `h_join` alone over `n` random elements and returns the microseconds one pass costs,
/// with submit overhead differenced out.
///
/// Legitimate to repeat in one submit: the kernel reads A, B and C, which it never writes,
/// so every pass computes the same values from the same inputs.
///
/// `None` when the many-pass submit came back no slower than the one-pass submit, which
/// means another test was contending for the device and the difference is noise. Reporting
/// junk as a number is the failure `bench/scripts` was fixed for; this reports it as absent.
fn join_us(
    n: u32,
    workgroup: u32,
    a: &wgpu::Buffer,
    bb: &wgpu::Buffer,
    c: &wgpu::Buffer,
) -> Option<f64> {
    let b = floor();
    let j = HJoin::with_shape(b, n.max(workgroup), workgroup).expect("h_join pipeline");
    let h_mont = g16_wgpu::gather::fr_buffer(b, "h_mont", n).unwrap();
    let h_std = g16_wgpu::gather::fr_buffer(b, "h_std", n).unwrap();
    let mut ring = ParamRing::new(b, "join sweep", j.dispatches(n).max(1)).unwrap();
    let offsets = j.plan(n, &mut ring).unwrap();
    ring.flush(b);
    let bind = j.bind(b, &ring, n, a, bb, c, &h_mont, &h_std).unwrap();

    const REPEATS: u32 = 40;
    let once = |passes: u32| -> u128 {
        let t0 = std::time::Instant::now();
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..passes {
                j.encode(&mut pass, &bind, n, &offsets).unwrap();
            }
        }
        b.queue().submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        t0.elapsed().as_micros()
    };
    for _ in 0..3 {
        once(1);
        once(1 + REPEATS);
    }
    let mut one: Vec<u128> = (0..5).map(|_| once(1)).collect();
    let mut many: Vec<u128> = (0..5).map(|_| once(1 + REPEATS)).collect();
    one.sort_unstable();
    many.sort_unstable();
    (many[2] > one[2]).then(|| (many[2] - one[2]) as f64 / REPEATS as f64)
}

#[test]
fn the_standalone_join_workgroup_size_is_measured() {
    let _gpu = exclusive();
    let b = floor();
    const SIZES: [u32; 4] = [32, 64, 128, 256];
    let mut totals = [0.0f64; SIZES.len()];
    let mut rows = String::new();

    for log_n in [12u32, 14, 15, 17, 18] {
        let n = 1u32 << log_n;
        let mut rng = SplitMix64(0x304 ^ u64::from(log_n));
        // Three different vectors, because a kernel that read A twice would be invisible
        // against a == b and that is exactly the bug the last round's verifier found.
        let a = upload_fr(b, "sweep a", &witness(&mut rng, n as usize)).unwrap();
        let bb = upload_fr(b, "sweep b", &witness(&mut rng, n as usize)).unwrap();
        let c = upload_fr(b, "sweep c", &witness(&mut rng, n as usize)).unwrap();

        let mut cells = Vec::new();
        for (i, &wg) in SIZES.iter().enumerate() {
            let us = join_us(n, wg, &a, &bb, &c);
            cells.push(match us {
                Some(v) => format!("{v:7.0}"),
                None => "      ?".to_string(),
            });
            totals[i] += us.unwrap_or(f64::NAN);
        }
        rows.push_str(&format!("2^{log_n:<5}{}\n", cells.join(" ")));
    }

    println!(
        "domain  {}\n{}total  {}",
        SIZES
            .iter()
            .map(|w| format!("{w:>7}"))
            .collect::<Vec<_>>()
            .join(" "),
        rows,
        totals
            .iter()
            .map(|t| format!("{t:7.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // The shipped constant has to be one of the sizes swept, so the table above is about the
    // number that ships and not about a neighbour of it.
    assert!(SIZES.contains(&wgsl::WORKGROUP));
    // No hard timing bar. Under default parallel `cargo test` this contends with the
    // artifact test on the same GPU and a bar tight enough to be meaningful was flaky about
    // one run in three in U5's experience, and a flaky test gets deleted rather than fixed.
    // The numbers are printed and the constant carries the table.
    //
    // **Read this table only from a `--release` run.** The differencing below cancels the
    // fixed cost of a submit and a fence, not the per-encode host cost, and that scales with
    // the pass count exactly as the kernel does. In debug the same 2^12 cell reads 58 to 80
    // microseconds against 15 in release, and the ranking inverts. The doc comment on
    // `gen::pointwise::WORKGROUP` has both tables and the argument.
    #[cfg(debug_assertions)]
    println!(
        "NOTE: debug build. The numbers above are dominated by wgpu's host-side encode cost \
         and the ranking is not the kernel's; re-run with --release."
    );
}

// ---------------------------------------------------------------------------
// 6. The tag, which is what stops one backend reading another's handle
// ---------------------------------------------------------------------------

#[test]
fn a_handle_under_another_tag_is_not_a_wgpu_handle() {
    let _gpu = exclusive();
    let b = floor();
    let mut rng = SplitMix64(0x7a6);
    let (n, n_vars) = (64usize, 40usize);
    let pk = synthetic_key(&mut rng, n, n_vars);
    let w = witness(&mut rng, n_vars);
    let stages = HStages::new(b, &pk).expect("stages");
    let h = pollster::block_on(stages.compute_h(b, &w, &mut StageTimings::default())).unwrap();

    // The two decoding helpers, exercised here rather than left as untested API. They are
    // the ones a debugging session reaches for, and `to_host_std` returning `None` is the
    // only signal that `fr_from_mont` produced a non-canonical residue.
    let g = h.device_handle::<WgpuHandle>(TAG).expect("handle");
    let cpu = CpuCircuit::new(synthetic_key(&mut SplitMix64(0x7a6), n, n_vars)).unwrap();
    let want = match cpu.compute_h(&w, &mut StageTimings::default()).unwrap() {
        HPoly::Host(v) => v,
        HPoly::Device { .. } => unreachable!(),
    };
    assert_eq!(pollster::block_on(g.to_host(b)).unwrap(), want, "to_host");
    assert_eq!(
        pollster::block_on(g.to_host_std(b)).unwrap().as_deref(),
        Some(&want[..]),
        "to_host_std: a None here means h_std holds limbs that are not a residue below r"
    );

    assert!(h.device_handle::<WgpuHandle>(TAG).is_some());
    assert!(
        h.device_handle::<WgpuHandle>("metal").is_none(),
        "a wgpu handle answered to the metal tag"
    );
    // And the reverse: a foreign handle under our own tag is still not a WgpuHandle,
    // because the downcast is on the type as well as on the tag.
    let foreign = HPoly::Device {
        tag: TAG,
        len: n,
        data: std::sync::Arc::new(vec![0u8; 4]),
    };
    assert!(foreign.device_handle::<WgpuHandle>(TAG).is_none());
    assert_eq!(TAG, "wgpu");
    println!("tag {TAG}: a foreign handle downcasts to None rather than to a pointer");
}
