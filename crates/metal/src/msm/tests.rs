//! The MSM module's unit tests: the host-side helpers, then the GPU against the CPU.

#[test]
fn slice_override_never_zero_or_past_u32() {
    assert_eq!(parse_slice_override("128"), Some(128));
    assert_eq!(parse_slice_override("0"), None);
    assert_eq!(parse_slice_override("x"), None);
    assert_eq!(parse_slice_override(""), None);
    assert_eq!(parse_slice_override("-4"), None);
    // low 32 bits zero: would truncate to 0 under `as u32`
    assert_eq!(parse_slice_override("4294967296"), Some(u32::MAX as usize));
    assert_eq!(
        parse_slice_override("18446744073709551615"),
        Some(u32::MAX as usize)
    );
}

use super::window::{parse_slice_override, DUMMY_ROWS, MERGE_TG, MSM_NO_ROW, RECODE_BITS, SCAN_TG};
use super::*;
use crate::layout::as_bytes;
use snarkrs_field::CurveGroup;
use snarkrs_formats::{wtns::Witness, ProvingKey};
use snarkrs_groth16::{cpu::CpuBackend, Backend, StageTimings};
use std::path::PathBuf;

#[test]
fn job_ranges_reject_scalar_overflow() {
    for (off, n) in [(usize::MAX, 1), (1, usize::MAX), (usize::MAX, 2)] {
        assert!(matches!(
            validate_job_ranges(off, n, usize::MAX, 0, usize::MAX),
            Err(ProveError::Backend {
                backend: "metal",
                ..
            })
        ));
    }
}

#[test]
fn job_ranges_reject_base_overflow() {
    for (off, n) in [(usize::MAX, 1), (1, usize::MAX), (usize::MAX, 2)] {
        assert!(matches!(
            validate_job_ranges(0, n, usize::MAX, off, usize::MAX),
            Err(ProveError::Backend {
                backend: "metal",
                ..
            })
        ));
    }
}

#[test]
fn job_ranges_reject_out_of_bounds() {
    for (soff, n, slen, boff, blen) in [
        (3, 2, 4, 0, 4),
        (0, 2, 4, 3, 4),
        (5, 0, 4, 0, 4),
        (0, 0, 4, 5, 4),
        (0, 1, 0, 0, 1),
        (0, 1, 1, 0, 0),
    ] {
        assert!(matches!(
            validate_job_ranges(soff, n, slen, boff, blen),
            Err(ProveError::Backend {
                backend: "metal",
                ..
            })
        ));
    }
}

#[test]
fn job_ranges_accept_empty_and_exact_end() {
    for (soff, n, slen, boff, blen) in [
        (0, 0, 0, 0, 0),
        (0, 0, 4, 0, 4),
        (4, 0, 4, 7, 7),
        (1, 3, 4, 2, 5),
        (0, 4, 4, 0, 4),
        (usize::MAX, 0, usize::MAX, usize::MAX, usize::MAX),
        (usize::MAX - 1, 1, usize::MAX, usize::MAX - 1, usize::MAX),
    ] {
        assert!(validate_job_ranges(soff, n, slen, boff, blen).is_ok());
    }
}

/// `name` is relative to `bench/artifacts`, so the grouped sets are reached as
/// `csp/keccak_128` rather than being unreachable from a flat name.
fn artifact(name: &str) -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .join(name);
    if d.join("circuit.zkey").exists() && d.join("circuit.wtns").exists() {
        Some(d)
    } else {
        None
    }
}

/// The same drift guard for the two threadgroup sizes, which had none.
///
/// `REDUCE_TG` and `SCAN_TG` are the *array* sizes of the `threadgroup` scratch in
/// `msm_reduce_*`, `msm_ones_*` and `msm_scan`, and the host dispatches that many
/// threads. Raising either one on the Rust side alone makes every thread above the
/// MSL array's length index out of bounds in threadgroup memory: no compile error,
/// no Metal validation error, just wrong points. Measured, not assumed. Setting
/// `REDUCE_TG` to 128 with the MSL left at 64 leaves H bit-exact and turns all five
/// MSM outputs wrong, so the proof stops verifying with nothing to point at.
#[test]
fn msl_declares_the_same_threadgroup_sizes() {
    for (name, want) in [
        ("REDUCE_TG", REDUCE_TG),
        ("SCAN_TG", SCAN_TG),
        ("MERGE_TG", MERGE_TG),
    ] {
        let line = format!("#define {name} {want}");
        assert!(
            MSM_MSL.contains(&line),
            "shaders/msm.metal does not contain the line:\n{line}\n\
             The host dispatches {want} threads and the MSL sizes its threadgroup \
             array from its own #define; if they disagree the reduction writes out \
             of bounds and the MSM answers are silently wrong."
        );
    }
}

/// The spill sentinel lives in two languages too. If the MSL value moved, the
/// host-tail combine would fold garbage slots or drop real ones, silently.
#[test]
fn msl_declares_the_same_spill_sentinel() {
    assert!(
        MSM_MSL.contains("constant uint MSM_NO_ROW = 0xffffffffu;"),
        "shaders/msm.metal no longer defines MSM_NO_ROW as 0xffffffff; \
         `Outputs::combine`'s host-tail spill fold tests against the same value \
         and must be changed with it."
    );
    assert_eq!(MSM_NO_ROW, u32::MAX);
}

/// Stage 4 sizes the buffer it lends H's plan with `dense_entries_bytes`. If that
/// drifted below what `Plan::alloc` asks for, the loan would be refused on every proof
/// and the entries would come from the pool again, which nothing else would notice.
#[test]
fn a_lent_buffer_fits_the_dense_plan_it_is_lent_to() {
    let m = MetalMsm::new().expect("Metal device");
    let std = crate::alloc::shared(&m.device, 4).unwrap();
    for log_n in 0..=26 {
        let n = 1usize << log_n;
        for work in [Work::Variable, Work::Constant] {
            let h = m.scalars_from_device_std_with(&std, n, work);
            let plan = Plan::new(&h, 0, n);
            assert_eq!(
                plan.n_windows * plan.cap * 8,
                dense_entries_bytes(n),
                "2^{log_n} {work:?}"
            );
        }
    }
}

/// The drift guard, same shape as `layout::tests::msl_declares_the_same_constants`:
/// the Fq modulus and `N0` live in two languages and must be edited together.
#[test]
fn msl_declares_the_same_fq_constants() {
    let want_n = format!(
        "constant uint FQ_N[8] = {{ {} }};",
        crate::layout::FQ_MODULUS
            .iter()
            .map(|l| format!("0x{l:08x}u"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        MSM_MSL.contains(&want_n),
        "shaders/msm.metal does not contain the line:\n{want_n}"
    );
    let want_n0 = format!("constant uint FQ_N0 = 0x{:08x}u;", crate::layout::FQ_N0);
    assert!(
        MSM_MSL.contains(&want_n0),
        "shaders/msm.metal does not contain the line:\n{want_n0}"
    );
    // 3b of the G2 twist, the constant of the complete formulas' constant-work
    // reduce, in Montgomery form as the shader holds every Fq.
    use ark_ec::short_weierstrass::SWCurveConfig;
    let b3 = <snarkrs_field::g2::Config as SWCurveConfig>::COEFF_B * snarkrs_field::Fq2::from(3u64);
    let packed = PackedFq2::from_fq2(&b3);
    for (name, limbs) in [("C0", packed.c0.v), ("C1", packed.c1.v)] {
        let want = format!(
            "constant uint FQ2_G2_B3_{name}[8] = {{ {} }};",
            limbs
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            MSM_MSL.contains(&want),
            "shaders/msm.metal does not contain the line:\n{want}"
        );
    }
}

/// The MSL must compile and every kernel must produce a pipeline. Separate from the
/// arithmetic tests so a compile error is reported as a compile error.
#[test]
fn the_library_compiles_and_every_kernel_has_a_pipeline() {
    let m = MetalMsm::new().expect("Metal device");
    // Occupancy is a measured property, not an assumption. Print it so a change in
    // register pressure shows up in the test log rather than as a mystery slowdown.
    eprintln!(
        "max threads/threadgroup: count {} scatter {} accumulate_g1 {} accumulate_g2 {} \
         reduce_g1 {} reduce_g2 {} (simd width {})",
        m.pipelines.count.max_total_threads_per_threadgroup(),
        m.pipelines.scatter.max_total_threads_per_threadgroup(),
        m.pipelines
            .accumulate_g1
            .max_total_threads_per_threadgroup(),
        m.pipelines
            .accumulate_g2
            .max_total_threads_per_threadgroup(),
        m.pipelines.reduce_g1.max_total_threads_per_threadgroup(),
        m.pipelines.reduce_g2.max_total_threads_per_threadgroup(),
        m.pipelines.count.thread_execution_width(),
    );
    assert!(m.pipelines.reduce_g2.max_total_threads_per_threadgroup() >= 1);
}

/// A handful of scalars against a naive sum, before anything the size of a real
/// circuit. This is the test that isolates the point arithmetic: if the digit
/// recoding or `madd` is wrong, it fails here rather than 140,000 points later.
#[test]
fn small_g1_msm_matches_a_naive_sum() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for n in [1usize, 2, 3, 31, 32, 33, 257, 1000] {
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut cur = G1Projective::generator();
        for i in 0..n {
            cur += G1Projective::generator();
            bases.push(cur.into_affine());
            // Deliberately dense in the two special cases, which is what a witness
            // looks like and what the kernels take a different path for.
            scalars.push(match i % 4 {
                0 => Fr::zero(),
                1 => Fr::one(),
                _ => {
                    let mut b = [0u8; 32];
                    for c in b.chunks_mut(8) {
                        c.copy_from_slice(&next().to_le_bytes());
                    }
                    <Fr as snarkrs_field::PrimeField>::from_le_bytes_mod_order(&b)
                }
            });
        }
        let want = bases
            .iter()
            .zip(&scalars)
            .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
        let db = m.upload_g1_bases(&bases).expect("upload bases");
        let ds = m.upload_scalars(&scalars).expect("upload scalars");
        let got = m.msm_g1(&db, &ds).unwrap();
        assert_eq!(got.into_affine(), want.into_affine(), "n = {n}");
    }
}

/// The three exceptional cases in `pt_madd`, reached on purpose rather than by the luck
/// of the scatter order.
///
/// One base with one scalar, repeated, puts every copy in the same bucket of every
/// window, so the second addition into a bucket is `P + P` and has to double. Its
/// negation with the same scalar lands in the same buckets and has to cancel to the
/// identity, and the next copy has to restart from it. Bases at infinity with nonzero
/// scalars ride along and have to be skipped. The runs are long enough to cross
/// accumulation slices, so the spill and merge path sees the same cases.
#[test]
fn doubling_cancellation_and_infinity_inside_one_bucket_match_a_naive_sum() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let s = Fr::from(0x1234_5678_9abc_def1u64) * Fr::from(0x0fed_cba9_8765_4321u64);
    for n in [2usize, 3, 64, 3001] {
        let p1 = (G1Projective::generator() * Fr::from(7u64)).into_affine();
        let p2 = (G2Projective::generator() * Fr::from(7u64)).into_affine();
        let pattern = |i: usize| i % 5;
        let g1: Vec<G1Affine> = (0..n)
            .map(|i| match pattern(i) {
                0 | 1 => p1,
                2 => -p1,
                3 => G1Affine::identity(),
                _ => p1,
            })
            .collect();
        let g2: Vec<G2Affine> = (0..n)
            .map(|i| match pattern(i) {
                0 | 1 => p2,
                2 => -p2,
                3 => G2Affine::identity(),
                _ => p2,
            })
            .collect();
        let scalars = vec![s; n];

        let want1 = g1.iter().fold(G1Projective::zero(), |a, b| a + *b * s);
        let want2 = g2.iter().fold(G2Projective::zero(), |a, b| a + *b * s);
        let ds = m.upload_scalars(&scalars).expect("upload scalars");
        let got1 = m.msm_g1(&m.upload_g1_bases(&g1).unwrap(), &ds).unwrap();
        let got2 = m.msm_g2(&m.upload_g2_bases(&g2).unwrap(), &ds).unwrap();
        assert_eq!(got1.into_affine(), want1.into_affine(), "G1, n = {n}");
        assert_eq!(got2.into_affine(), want2.into_affine(), "G2, n = {n}");
    }
}

/// `snarkrs_msm::MsmBackend` forbids reducing to the shorter side: the prefix sum is a
/// valid group element for a key that is wrong, and this is the `setup` path that
/// writes `.zkey` files. `MetalMsmBackend` panics through this error.
#[test]
fn a_length_mismatch_is_an_error_not_a_shorter_msm() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let bases: Vec<G1Affine> = (0..8)
        .map(|i| (G1Projective::generator() * Fr::from(i as u64 + 1)).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..5).map(|i| Fr::from(i as u64 + 1)).collect();
    let db = m.upload_g1_bases(&bases).expect("upload bases");
    let ds = m.upload_scalars(&scalars).expect("upload scalars");
    let e = m.msm_g1(&db, &ds).expect_err("8 bases against 5 scalars");
    let text = e.to_string();
    assert!(text.contains('8') && text.contains('5'), "{text}");
}

#[test]
fn a_length_mismatch_is_an_error_not_a_shorter_msm_g2() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let bases: Vec<G2Affine> = (0..4)
        .map(|i| (G2Projective::generator() * Fr::from(i as u64 + 1)).into_affine())
        .collect();
    let scalars: Vec<Fr> = (0..9).map(|i| Fr::from(i as u64 + 1)).collect();
    let db = m.upload_g2_bases(&bases).expect("upload bases");
    let ds = m.upload_scalars(&scalars).expect("upload scalars");
    let e = m.msm_g2(&db, &ds).expect_err("4 bases against 9 scalars");
    let text = e.to_string();
    assert!(text.contains('4') && text.contains('9'), "{text}");
}

#[test]
fn small_g2_msm_matches_a_naive_sum() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let n = 200usize;
    let mut bases = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    let mut cur = G2Projective::generator();
    for i in 0..n {
        cur += G2Projective::generator();
        bases.push(cur.into_affine());
        scalars.push(Fr::from((i as u64) * 7919 + 3));
    }
    scalars[0] = Fr::zero();
    scalars[1] = Fr::one();
    let want = bases
        .iter()
        .zip(&scalars)
        .fold(G2Projective::zero(), |a, (b, s)| a + *b * s);
    let db = m.upload_g2_bases(&bases).expect("upload bases");
    let ds = m.upload_scalars(&scalars).expect("upload scalars");
    let got = m.msm_g2(&db, &ds).unwrap();
    assert_eq!(got.into_affine(), want.into_affine());
}

/// The host tail, which nothing else in the default suite reaches.
///
/// Past [`HOST_TAIL_MAX_BUCKETS`] a one-window plan stops after the accumulation and
/// [`Outputs::combine`] folds the spilled runs into the buckets itself, then walks a
/// reverse running sum. That is a different final answer, not a different schedule,
/// and it needs scalars short enough to recode in a single window, which none of the
/// js artifacts produce. The plan shape is asserted as well as the answer, because a
/// refit of the window model that moved `c` off 9 here would otherwise leave this
/// quietly testing the GPU reduce instead.
#[test]
fn a_one_window_plan_finishes_on_the_host_and_matches_a_naive_sum() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let n = 1000usize;
    let mut g1 = Vec::with_capacity(n);
    let mut g2 = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    let mut c1 = G1Projective::generator();
    let mut c2 = G2Projective::generator();
    for i in 0..n {
        c1 += G1Projective::generator();
        c2 += G2Projective::generator();
        g1.push(c1.into_affine());
        g2.push(c2.into_affine());
        // Under 2^8, so the plan lays the recoding over nine bits and spends them in
        // one window. The cycle also carries zeros and ones, which keeps the ones
        // gather and the spill fold on the path together.
        scalars.push(Fr::from((i % 200) as u64));
    }
    let ds = m.upload_scalars(&scalars).expect("upload scalars");
    let plan = Plan::new(&ds, 0, n);
    assert!(
        plan.host_tail,
        "c = {} over {} windows of {} buckets is not the host-tail shape",
        plan.c, plan.n_windows, plan.n_buckets
    );

    let b1 = m.upload_g1_bases(&g1).expect("upload bases");
    let b2 = m.upload_g2_bases(&g2).expect("upload bases");
    let want1 = g1
        .iter()
        .zip(&scalars)
        .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
    let want2 = g2
        .iter()
        .zip(&scalars)
        .fold(G2Projective::zero(), |a, (b, s)| a + *b * s);
    let got1 = m.msm_g1(&b1, &ds).unwrap();
    let got2 = m.msm_g2(&b2, &ds).unwrap();
    assert_eq!(got1.into_affine(), want1.into_affine());
    assert_eq!(got2.into_affine(), want2.into_affine());
}

/// The device entry point cannot assume H never contains a one. Exercise both
/// groups and reuse the scratch across zero, one, and mixed scalar vectors.
#[test]
fn unclassified_ones_and_zeros_match_both_cpu_groups() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let n = 65;
    let g1: Vec<_> = (0..n)
        .map(|i| (G1Projective::generator() * Fr::from((i + 1) as u64)).into_affine())
        .collect();
    let g2: Vec<_> = (0..n)
        .map(|i| (G2Projective::generator() * Fr::from((i + 1) as u64)).into_affine())
        .collect();
    let b1 = m.upload_g1_bases(&g1).expect("upload bases");
    let b2 = m.upload_g2_bases(&g2).expect("upload bases");
    for kind in [0, 1, 2, 0] {
        let scalars: Vec<_> = (0..n)
            .map(|i| match kind {
                0 => Fr::zero(),
                1 => Fr::one(),
                _ => Fr::from((i % 5) as u64),
            })
            .collect();
        let uploaded = m.upload_scalars(&scalars).expect("upload scalars");
        let device = m.scalars_from_device_std(&uploaded.buf, n);
        let want1 = g1
            .iter()
            .zip(&scalars)
            .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
        let want2 = g2
            .iter()
            .zip(&scalars)
            .fold(G2Projective::zero(), |a, (b, s)| a + *b * s);
        assert_eq!(
            m.msm_g1(&b1, &device).unwrap().into_affine(),
            want1.into_affine()
        );
        assert_eq!(
            m.msm_g2(&b2, &device).unwrap().into_affine(),
            want2.into_affine()
        );
    }
}

/// Bases with every seventh at infinity, for the constant-work tests: the finite
/// mask is the one thing the key is allowed to decide.
fn bases_with_infinity(n: usize) -> (Vec<G1Affine>, Vec<G2Affine>) {
    use snarkrs_field::PrimeGroup;
    let g1 = (0..n)
        .map(|i| {
            if i % 7 == 3 {
                G1Affine::identity()
            } else {
                (G1Projective::generator() * Fr::from(i as u64 + 1)).into_affine()
            }
        })
        .collect();
    let g2 = (0..n)
        .map(|i| {
            if i % 7 == 3 {
                G2Affine::identity()
            } else {
                (G2Projective::generator() * Fr::from(i as u64 + 1)).into_affine()
            }
        })
        .collect();
    (g1, g2)
}

/// Scalar vectors of every sparsity the witness path meets: dense random, all
/// zeros, all ones, bits with a few generals, and small values.
fn scalars_of_every_sparsity(n: usize) -> Vec<(&'static str, Vec<Fr>)> {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let random: Vec<Fr> = (0..n)
        .map(|_| {
            let mut b = [0u8; 32];
            for c in b.chunks_mut(8) {
                c.copy_from_slice(&next().to_le_bytes());
            }
            <Fr as snarkrs_field::PrimeField>::from_le_bytes_mod_order(&b)
        })
        .collect();
    vec![
        ("random", random),
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

/// `Work::Constant` against the CPU MSM and the variable path, in both groups, on
/// every sparsity, over a range that crosses classification chunks and slices,
/// plus a job that reads an offset subrange the way the L MSM does.
#[test]
fn constant_work_matches_the_cpu_on_every_sparsity() {
    use snarkrs_msm::MsmBackend;
    let m = MetalMsm::new().expect("Metal device");
    let cpu = snarkrs_msm::CpuMsm::new();
    let n = 3 * CLASSIFY_CHUNK + 7;
    let (g1, g2) = bases_with_infinity(n);
    let b1 = m.upload_g1_bases(&g1).expect("upload bases");
    let b2 = m.upload_g2_bases(&g2).expect("upload bases");
    let off = 5;
    for (label, scalars) in scalars_of_every_sparsity(n) {
        let constant = m
            .upload_scalars_with(&scalars, Work::Constant)
            .expect("upload scalars");
        let variable = m.upload_scalars(&scalars).expect("upload scalars");
        let want1 = cpu.msm_g1(&g1, &scalars).into_affine();
        let want2 = cpu.msm_g2(&g2, &scalars).into_affine();
        assert_eq!(
            m.msm_g1(&b1, &constant).unwrap().into_affine(),
            want1,
            "{label}: G1"
        );
        assert_eq!(
            m.msm_g2(&b2, &constant).unwrap().into_affine(),
            want2,
            "{label}: G2"
        );
        assert_eq!(
            m.msm_g1(&b1, &variable).unwrap().into_affine(),
            want1,
            "{label}: G1, variable"
        );
        let got = m
            .msm_batch(&[Job::G1(JobG1 {
                bases: &b1,
                base_off: off,
                scalars: &constant,
                scalar_off: off,
                n: n - off,
            })])
            .unwrap();
        assert_eq!(
            got[0].g1().unwrap().into_affine(),
            cpu.msm_g1(&g1[off..], &scalars[off..]).into_affine(),
            "{label}: G1 over an offset range"
        );
    }
}

/// A suffix job riding a wider job's plan, the way L rides A and B's, in both work
/// modes and both groups, on every sparsity: listed before the job it rides, over
/// bases of its own and over the wider job's bases at an offset.
#[test]
fn a_suffix_sharing_the_wider_plan_matches_the_cpu() {
    use snarkrs_msm::MsmBackend;
    let m = MetalMsm::new().expect("Metal device");
    let cpu = snarkrs_msm::CpuMsm::new();
    let n = 3 * CLASSIFY_CHUNK + 7;
    let off = 5;
    let (g1, g2) = bases_with_infinity(n);
    let b1 = m.upload_g1_bases(&g1).expect("upload bases");
    let b2 = m.upload_g2_bases(&g2).expect("upload bases");
    let l1 = m.upload_g1_bases(&g1[off..]).expect("upload bases");
    for (label, scalars) in scalars_of_every_sparsity(n) {
        let want_full = cpu.msm_g1(&g1, &scalars).into_affine();
        let want1 = cpu.msm_g1(&g1[off..], &scalars[off..]).into_affine();
        let want2 = cpu.msm_g2(&g2[off..], &scalars[off..]).into_affine();
        for work in [Work::Constant, Work::Variable] {
            let s = m
                .upload_scalars_with(&scalars, work)
                .expect("upload scalars");
            let got = m
                .msm_batch(&[
                    Job::G1(JobG1 {
                        bases: &l1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: off,
                        n: n - off,
                    }),
                    Job::G1(JobG1 {
                        bases: &b1,
                        base_off: 0,
                        scalars: &s,
                        scalar_off: 0,
                        n,
                    }),
                    Job::G2(JobG2 {
                        bases: &b2,
                        base_off: off,
                        scalars: &s,
                        scalar_off: off,
                        n: n - off,
                    }),
                ])
                .unwrap();
            assert_eq!(
                got[0].g1().unwrap().into_affine(),
                want1,
                "{label} {work:?}: L"
            );
            assert_eq!(
                got[1].g1().unwrap().into_affine(),
                want_full,
                "{label} {work:?}: A"
            );
            assert_eq!(
                got[2].g2().unwrap().into_affine(),
                want2,
                "{label} {work:?}: G2"
            );
        }
    }
}

/// `Work::Constant`'s promise, checked where it is decided: plans over witnesses
/// of every sparsity have the same shape, allocate the same bytes,
/// dispatch the same threads, and fill every window's entry region exactly. The
/// variable path over the same two witnesses is shown to differ, so the check is
/// known to bite.
#[test]
fn constant_work_geometry_ignores_the_witness() {
    let m = MetalMsm::new().expect("Metal device");
    let n = 2 * CLASSIFY_CHUNK + 5;
    let (g1, _) = bases_with_infinity(n);
    let b1 = m.upload_g1_bases(&g1).expect("upload bases");
    let sparse: Vec<Fr> = (0..n).map(|i| Fr::from((i % 2) as u64)).collect();
    let dense: Vec<Fr> = (0..n).map(|i| -Fr::from(i as u64 * 7919 + 3)).collect();

    // Everything a dispatch or an allocation is sized from, the merge tree included.
    let shape = |p: &Plan<'_>| {
        vec![
            p.n,
            p.c as usize,
            p.n_windows,
            p.n_buckets,
            p.cap,
            p.slice_len,
            p.slices,
            p.reduce_groups,
            p.merge_wide_rows,
            p.scatter_parts,
            p.host_tail as usize,
            p.separate_ones as usize,
            p.dummy_rows,
            p.rows(),
        ]
        .into_iter()
        .chain(p.fold_levels().into_iter().flat_map(|(m, g)| [m, g]))
        .collect::<Vec<_>>()
    };
    let mut seen = Vec::new();
    for (_, scalars) in scalars_of_every_sparsity(n) {
        let s = m
            .upload_scalars_with(&scalars, Work::Constant)
            .expect("upload scalars");
        let mut plan = Plan::new(&s, 0, n);
        let mut keep = Vec::new();
        plan.alloc(&m.pool, &mut keep).expect("plan scratch");
        let job = Job::G1(JobG1 {
            bases: &b1,
            base_off: 0,
            scalars: &s,
            scalar_off: 0,
            n,
        });
        let out = Outputs::alloc(&m, &mut keep, &job, &plan, 0).expect("outputs");
        assert_eq!(out.ones_groups, 0);
        assert!(out.ones_idx.is_none());
        let bytes: Vec<u64> = keep.iter().map(|b| b.length()).collect();

        // The digit pipeline, then the histogram it built: every scalar is in
        // every window exactly once, in a real row or a dummy one.
        let queue = m.queue.get();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        plan.encode(&m, enc);
        enc.end_encoding();
        cb.commit();
        crate::cb::wait_ok(cb, "constant-work digits").unwrap();
        // SAFETY: `alloc` sized both at `rows()` u32s and the batch has completed.
        let counts: &[u32] = unsafe { read_back(plan.counts.as_ref().unwrap(), plan.rows()) };
        let cursor: &[u32] = unsafe { read_back(plan.cursor.as_ref().unwrap(), plan.rows()) };
        let real = plan.n_windows * plan.n_buckets;
        let mut in_real = 0usize;
        for w in 0..plan.n_windows {
            let real_w: u32 = counts[w * plan.n_buckets..(w + 1) * plan.n_buckets]
                .iter()
                .sum();
            let dummy_w: u32 = counts[real + w * DUMMY_ROWS..real + (w + 1) * DUMMY_ROWS]
                .iter()
                .sum();
            assert_eq!(real_w as usize + dummy_w as usize, n, "window {w}");
            assert_eq!(
                cursor[real + (w + 1) * DUMMY_ROWS - 1] as usize,
                (w + 1) * n,
                "window {w}'s region is not full"
            );
            in_real += real_w as usize;
        }
        seen.push((shape(&plan), bytes, in_real));
        drop(plan);
        drop(out);
        m.pool.give(keep);
    }
    for case in &seen[1..] {
        assert_eq!(seen[0].0, case.0, "plan shape follows the witness");
        assert_eq!(seen[0].1, case.1, "allocation follows the witness");
    }
    // What did change is where the entries went, which is the one thing Pippenger
    // cannot hide.
    assert_ne!(seen[0].2, seen[1].2);

    let sparse_v = m.upload_scalars(&sparse).expect("upload scalars");
    let dense_v = m.upload_scalars(&dense).expect("upload scalars");
    assert_ne!(
        shape(&Plan::new(&sparse_v, 0, n)),
        shape(&Plan::new(&dense_v, 0, n)),
        "the variable path should size its plan from the witness"
    );
}

/// The path stage 9 will actually take once the NTT keeps `H` on the device: a
/// Montgomery `Fr` buffer converted in place rather than packed on the host.
///
/// This is the asymmetry that silently costs a factor of R if it is got wrong, so it
/// is checked against the host-packed path on the same values.
#[test]
fn device_resident_montgomery_scalars_agree_with_host_packing() {
    use snarkrs_field::PrimeGroup;
    let m = MetalMsm::new().expect("Metal device");
    let n = 512usize;
    let mut bases = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    let mut cur = G1Projective::generator();
    for i in 0..n {
        cur += G1Projective::generator();
        bases.push(cur.into_affine());
        scalars.push(Fr::from((i as u64) * 104_729 + 17));
    }
    let packed = crate::layout::PackedFr::pack_slice(&scalars);
    let bytes = as_bytes(&packed);
    let mont =
        crate::alloc::shared_with_data(&m.device, bytes.as_ptr().cast(), bytes.len()).unwrap();

    let db = m.upload_g1_bases(&bases).expect("upload bases");
    let from_host = m
        .msm_g1(&db, &m.upload_scalars(&scalars).expect("upload scalars"))
        .unwrap();
    let from_device = m
        .msm_g1(&db, &m.scalars_from_device_mont(&mont, n).unwrap())
        .unwrap();
    let want = bases
        .iter()
        .zip(&scalars)
        .fold(G1Projective::zero(), |a, (b, s)| a + *b * s);
    assert_eq!(from_host.into_affine(), want.into_affine());
    assert_eq!(from_device.into_affine(), want.into_affine());
}

/// The real test: all five MSMs of an actual circuit, GPU against the CPU backend,
/// on identical inputs. Points must be exactly equal after normalisation.
fn five_msms_against_cpu(name: &str) {
    let Some(dir) = artifact(name) else {
        eprintln!("{name}: artifact missing, skipped");
        return;
    };
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
    let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;

    let circuit = CpuBackend::new().prepare(pk).expect("prepare");
    let mut t = StageTimings::default();
    let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
    let cpu = circuit.msms(&witness, &h, &mut t).expect("cpu msms");
    let h_scalars = h.to_host().expect("cpu H is on the host");
    let pk = circuit.key();
    let n_public = circuit.n_public();

    let m = MetalMsm::new().expect("Metal device");
    let a_bases = m.upload_g1_bases(&pk.a_query).expect("upload bases");
    let b1_bases = m.upload_g1_bases(&pk.b_g1_query).expect("upload bases");
    let b2_bases = m.upload_g2_bases(&pk.b_g2_query).expect("upload bases");
    let l_bases = m.upload_g1_bases(&pk.l_query).expect("upload bases");
    let h_bases = m.upload_g1_bases(&pk.h_query).expect("upload bases");
    let w_scalars = m.upload_scalars(&witness).expect("upload scalars");
    let h_dev = m.upload_scalars(h_scalars).expect("upload scalars");

    let n = witness.len();
    let l_off = n_public + 1;
    let jobs = vec![
        Job::G1(JobG1 {
            bases: &a_bases,
            base_off: 0,
            scalars: &w_scalars,
            scalar_off: 0,
            n,
        }),
        Job::G2(JobG2 {
            bases: &b2_bases,
            base_off: 0,
            scalars: &w_scalars,
            scalar_off: 0,
            n,
        }),
        Job::G1(JobG1 {
            bases: &b1_bases,
            base_off: 0,
            scalars: &w_scalars,
            scalar_off: 0,
            n,
        }),
        Job::G1(JobG1 {
            bases: &l_bases,
            base_off: 0,
            scalars: &w_scalars,
            scalar_off: l_off,
            n: pk.l_query.len(),
        }),
        Job::G1(JobG1 {
            bases: &h_bases,
            base_off: 0,
            scalars: &h_dev,
            scalar_off: 0,
            n: h_scalars.len(),
        }),
    ];
    let got = m.msm_batch(&jobs).expect("gpu msms");

    assert_eq!(
        got[0].g1().unwrap().into_affine(),
        cpu.a_g1.into_affine(),
        "{name}: stage 5, MSM A -> G1"
    );
    assert_eq!(
        got[1].g2().unwrap().into_affine(),
        cpu.b_g2.into_affine(),
        "{name}: stage 6, MSM B -> G2"
    );
    assert_eq!(
        got[2].g1().unwrap().into_affine(),
        cpu.b_g1.into_affine(),
        "{name}: stage 7, MSM B -> G1"
    );
    assert_eq!(
        got[3].g1().unwrap().into_affine(),
        cpu.l_g1.into_affine(),
        "{name}: stage 8, MSM L -> G1"
    );
    assert_eq!(
        got[4].g1().unwrap().into_affine(),
        cpu.h_g1.into_affine(),
        "{name}: stage 9, MSM H -> G1"
    );
}

#[test]
fn five_msms_match_cpu_on_tiny_mul() {
    five_msms_against_cpu("tiny_mul");
}

#[test]
fn five_msms_match_cpu_on_js_1x1_d8() {
    five_msms_against_cpu("js_1x1_d8");
}

/// GPU against CPU on wall clock, warm, for the same five MSMs.
///
/// Not a check, a measurement. Bases upload is `prepare` work and sits outside the
/// timed region on both sides; scalar packing and upload is per-proof work and sits
/// inside the GPU's. Run with
/// `cargo test -p snarkrs-metal --release measure_against_cpu -- --ignored --nocapture`.
#[test]
#[ignore = "measurement, not a check"]
fn measure_against_cpu() {
    use std::time::Instant;
    let reps = 9;
    let t_start = Instant::now();
    let m = MetalMsm::new().expect("Metal device");
    let compile_ms = t_start.elapsed().as_secs_f64() * 1e3;
    println!("MSL compile + all pipelines: {compile_ms:.1} ms (once per process)");
    println!(
        "{:<14} {:>8} {:>8} {:>7} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "artifact",
        "n_vars",
        "domain",
        "gen%",
        "c(wit/H)",
        "cpu ms",
        "gpu ms",
        "gpu-nosc",
        "upload"
    );

    for name in [
        "tiny_mul",
        "js_1x1_d8",
        "js_2x2_d16",
        "js_2x2_d32",
        "js_8x8_d32",
        "js_16x16_d32",
    ] {
        let Some(dir) = artifact(name) else {
            continue;
        };
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let circuit = CpuBackend::new().prepare(pk).expect("prepare");
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
        let h_scalars = h.to_host().unwrap().to_vec();
        let pk = circuit.key();
        let n = witness.len();
        let l_off = circuit.n_public() + 1;

        let t0 = Instant::now();
        let a_bases = m.upload_g1_bases(&pk.a_query).expect("upload bases");
        let b1_bases = m.upload_g1_bases(&pk.b_g1_query).expect("upload bases");
        let b2_bases = m.upload_g2_bases(&pk.b_g2_query).expect("upload bases");
        let l_bases = m.upload_g1_bases(&pk.l_query).expect("upload bases");
        let h_bases = m.upload_g1_bases(&pk.h_query).expect("upload bases");
        let upload_ms = t0.elapsed().as_secs_f64() * 1e3;

        let general = {
            let s = m.upload_scalars(&witness).expect("upload scalars");
            s.general_in(&(0..n))
        };

        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        let mut gpu_nosc = Vec::new();
        for r in 0..reps + 1 {
            let t0 = Instant::now();
            let mut tt = StageTimings::default();
            let _ = circuit.msms(&witness, &h, &mut tt).unwrap();
            let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t0 = Instant::now();
            let w_scalars = m.upload_scalars(&witness).expect("upload scalars");
            let h_dev = m.upload_scalars(&h_scalars).expect("upload scalars");
            let pack_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let jobs = vec![
                Job::G1(JobG1 {
                    bases: &a_bases,
                    base_off: 0,
                    scalars: &w_scalars,
                    scalar_off: 0,
                    n,
                }),
                Job::G2(JobG2 {
                    bases: &b2_bases,
                    base_off: 0,
                    scalars: &w_scalars,
                    scalar_off: 0,
                    n,
                }),
                Job::G1(JobG1 {
                    bases: &b1_bases,
                    base_off: 0,
                    scalars: &w_scalars,
                    scalar_off: 0,
                    n,
                }),
                Job::G1(JobG1 {
                    bases: &l_bases,
                    base_off: 0,
                    scalars: &w_scalars,
                    scalar_off: l_off,
                    n: pk.l_query.len(),
                }),
                Job::G1(JobG1 {
                    bases: &h_bases,
                    base_off: 0,
                    scalars: &h_dev,
                    scalar_off: 0,
                    n: h_scalars.len(),
                }),
            ];
            let _ = m.msm_batch(&jobs).unwrap();
            let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
            // Drop the first round: it pays first-touch page faults on every scratch
            // buffer, which is exactly the cold cost the pool exists to pay once.
            if r > 0 {
                cpu.push(cpu_ms);
                gpu.push(pack_ms + dev_ms);
                gpu_nosc.push(dev_ms);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let c_wit = window_size(general);
        let c_h = window_size(h_scalars.len());
        println!(
            "{:<14} {:>8} {:>8} {:>6.1}% {:>4}/{:<3} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
            name,
            n,
            circuit.domain_size(),
            100.0 * general as f64 / n as f64,
            c_wit,
            c_h,
            med(&mut cpu),
            med(&mut gpu),
            med(&mut gpu_nosc),
            upload_ms,
        );
    }
}

/// Per-MSM wall clock plus the bucket-occupancy histogram behind the kernel choice.
///
/// The shipped accumulation hands every thread a fixed-length slice of the sorted
/// entries, so per-thread work is uniform whatever the histogram says, and the spill
/// slots plus `msm_merge_*` exist to stitch back together the runs a fat bucket
/// spreads over several slices. The max-to-average ratio printed here is what that
/// machinery is paid for: `G16_METAL_MSM_LEGACY_ACC=1` gives one thread one bucket
/// instead, and then the dispatch finishes when the fattest bucket does.
#[test]
#[ignore = "measurement, not a check"]
fn measure_per_msm() {
    use std::time::Instant;
    let m = MetalMsm::new().expect("Metal device");
    for name in ["js_2x2_d16", "js_2x2_d32", "js_8x8_d32", "js_16x16_d32"] {
        let Some(dir) = artifact(name) else { continue };
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let witness = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let circuit = CpuBackend::new().prepare(pk).expect("prepare");
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
        let h_scalars = h.to_host().unwrap().to_vec();
        let pk = circuit.key();
        let n = witness.len();
        let l_off = circuit.n_public() + 1;

        let a_bases = m.upload_g1_bases(&pk.a_query).expect("upload bases");
        let b2_bases = m.upload_g2_bases(&pk.b_g2_query).expect("upload bases");
        let l_bases = m.upload_g1_bases(&pk.l_query).expect("upload bases");
        let h_bases = m.upload_g1_bases(&pk.h_query).expect("upload bases");
        let w_scalars = m.upload_scalars(&witness).expect("upload scalars");
        let h_dev = m.upload_scalars(&h_scalars).expect("upload scalars");

        println!("--- {name}");
        for (label, scalars_slice, job) in [
            ("A  G1", &witness[..] as &[Fr], 0usize),
            ("B  G2", &witness[..], 1),
            ("L  G1", &witness[l_off..], 2),
            ("H  G1", &h_scalars[..], 3),
        ] {
            let (nn, soff) = match job {
                2 => (pk.l_query.len(), l_off),
                3 => (h_scalars.len(), 0),
                _ => (n, 0),
            };
            let sbuf = if job == 3 { &h_dev } else { &w_scalars };
            let jobs = match job {
                1 => vec![Job::G2(JobG2 {
                    bases: &b2_bases,
                    base_off: 0,
                    scalars: sbuf,
                    scalar_off: soff,
                    n: nn,
                })],
                2 => vec![Job::G1(JobG1 {
                    bases: &l_bases,
                    base_off: 0,
                    scalars: sbuf,
                    scalar_off: soff,
                    n: nn,
                })],
                3 => vec![Job::G1(JobG1 {
                    bases: &h_bases,
                    base_off: 0,
                    scalars: sbuf,
                    scalar_off: soff,
                    n: nn,
                })],
                _ => vec![Job::G1(JobG1 {
                    bases: &a_bases,
                    base_off: 0,
                    scalars: sbuf,
                    scalar_off: soff,
                    n: nn,
                })],
            };
            let _ = m.msm_batch(&jobs).unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = m.msm_batch(&jobs).unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }

            // Host-side replica of the kernel's recoding, for the histogram only.
            let general = scalars_slice
                .iter()
                .filter(|s| !(s.is_zero() || s.is_one()))
                .count();
            let c = window_size(general);
            let nw = RECODE_BITS.div_ceil(c as usize);
            let nb = 1usize << (c - 1);
            let mut counts = vec![0u32; nw * nb];
            for s in scalars_slice {
                if s.is_zero() || s.is_one() {
                    continue;
                }
                let big = <Fr as snarkrs_field::PrimeField>::into_bigint(*s);
                let limbs: &[u64] = big.as_ref();
                for w in 0..nw {
                    let (mag, _) = host_signed_digit(limbs, w, c);
                    if mag != 0 {
                        counts[w * nb + (mag - 1) as usize] += 1;
                    }
                }
            }
            let maxb = counts.iter().copied().max().unwrap_or(0);
            let total: u64 = counts.iter().map(|&x| x as u64).sum();
            let avg = total as f64 / (nw * nb) as f64;
            println!(
                "{label}  n {:>7}  gen {:>7}  c {:>2}  W {:>2}  buckets {:>6}  \
                 entries {:>9}  avg/bucket {:>7.2}  max {:>7}  ratio {:>6.0}x  {:>8.2} ms",
                nn,
                general,
                c,
                nw,
                nb,
                total,
                avg,
                maxb,
                maxb as f64 / avg.max(1e-9),
                best
            );
        }
    }
}

/// The CPU-side twin of `sc_signed_digit`, used only by the histogram above.
fn host_signed_digit(limbs: &[u64], i: usize, c: u32) -> (u32, bool) {
    let read = |off: usize, width: u32| -> u64 {
        let idx = off / 64;
        if idx >= limbs.len() {
            return 0;
        }
        let sh = off % 64;
        let mut buf = limbs[idx] >> sh;
        if sh + width as usize > 64 && idx + 1 < limbs.len() {
            buf |= limbs[idx + 1] << (64 - sh);
        }
        buf & ((1u64 << width) - 1)
    };
    let off = i * c as usize;
    let b = read(off, c);
    let carry = if off == 0 { 0 } else { read(off - 1, 1) };
    if (b >> (c - 1)) & 1 == 1 {
        let mag = (1u64 << c) - b - carry;
        (mag as u32, mag != 0)
    } else {
        ((b + carry) as u32, false)
    }
}

/// The larger artifacts. Slow in a debug build, so they are opt-in; run with
/// `cargo test -p snarkrs-metal --release -- --ignored`.
#[test]
#[ignore = "minutes on the CPU oracle side"]
fn five_msms_match_cpu_on_every_artifact() {
    for name in [
        "js_2x2_d16",
        "js_2x2_d32",
        "js_8x8_d32",
        "js_16x16_d32",
        // The csp set, a level down and therefore missed by every flat name. It is
        // where the witness plans are one window wide and the H plans reach c = 15,
        // so it is the only set that covers the host tail and the split scatter.
        "csp/sha256_128",
        "csp/sha256_256",
        "csp/keccak_128",
        "csp/keccak_256",
        "csp/keccak_512",
    ] {
        five_msms_against_cpu(name);
    }
}
