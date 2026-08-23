//! U9's acceptance tests: the whole G1 MSM on a real GPU, against `g16-msm`'s CPU Pippenger.
//!
//! Four of the five MSMs in a Groth16 proof are G1 (A, B-G1, L and H) and one is G2, so this
//! is the hot path and `tests/msm_g2.rs` is the sideshow. The two files ask the same questions
//! and share [`msmcommon`] for everything that does not name a curve.
//!
//! # What is being pinned
//!
//! 1. **`msm_g1` equals the CPU Pippenger exactly**, at n = 1, 2, 3, 63, 64, 65, 127, 128,
//!    129, 130, 200, 256, 257, 1000 and 2^16, at every window width from 2 to 16, with a
//!    nonzero `scalar_off` and a nonzero `base_off`; on all-zero scalars, on all-one scalars,
//!    on a witness-shaped mixture, on an input containing points at infinity, and on **real
//!    witnesses and real base vectors from `bench/artifacts`**, which carry a distribution no
//!    generated input here reproduces. Exactly, not up to the group law: both sides combine
//!    the windows high to low with `c` doublings between, so a disagreement is a bug and
//!    never an association difference.
//!
//!    128 and 129 are named on purpose. The G2 merge shipped with `k_hi = k_lo` left behind
//!    by a debugging session, which is correct for every n up to `slice_len` and drops spills
//!    for every n above it; the bisect that found it landed on exactly this pair.
//! 2. **The three branches of `pt_madd_g1` that a random draw will not hit.** Two equal bases
//!    in one bucket take the doubling arm, two opposite bases take the cancelling arm, and an
//!    infinity base takes the skip. All three are built on purpose.
//! 3. **Nothing writes outside its range.** Every output buffer is allocated with slack and
//!    pre-filled with a sentinel, because WebGPU drops an out-of-range storage write in
//!    silence, so without a sentinel an over-run is unobservable and a kernel that writes
//!    nothing agrees with a zeroed oracle.
//! 4. **Every entry point fits the browser floor**: at most 8 storage buffers per pipeline
//!    layout, counted from the layouts the host actually builds, and at most 16384 bytes of
//!    workgroup storage, computed from the generator at every `tg` it will accept.
//! 5. **The G1 shape constants are measured separately from G2's.** An `Xyzz<Fq>` is 128
//!    bytes against an `Xyzz<Fq2>`'s 256 and an `Fq` multiply is a third of an `Fq2` one, so
//!    neither the occupancy arithmetic nor the ratio between accumulation and merge carries
//!    over. Every one of the three sweeps here is run again rather than inherited.
//! 6. **The two curves' offsets do not coincide.** G1's `n_windows * 128` is not always a
//!    multiple of the 256-byte storage binding alignment, where G2's `n_windows * 256`
//!    always is, so the `ones` window inside the results buffer is rounded up and the
//!    rounding is checked at an odd `n_windows`.
//!
//! # Rules the inputs follow
//!
//! **Never degenerate by accident.** Every random scalar vector is asserted to produce a
//! non-identity result before anything is compared, so a test cannot pass by both sides
//! computing the point at infinity. The degenerate vectors are deliberate and are their own
//! test.
//!
//! **Fixed seeds**, `ark_std::test_rng`, so a failure reproduces.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ark_std::{rand::Rng, test_rng, UniformRand};
use g16_field::{CurveGroup, Fr, G1Affine, G1Projective, One, Zero};
use g16_gpu_layout::{PackedG1Affine, LIMBS};
use g16_msm::{CpuMsm, MsmBackend};
use g16_wgpu::gen::points as wgsl;
use g16_wgpu::msm::{pack_scalars, DigitBuffers, DigitPlan, MsmDigits};
use g16_wgpu::points::{MsmPointsG1, PointBuffers, PointPlan};
use g16_wgpu::{LimitsProfile, ParamRing, WgpuBackend};
use g16_zkey::{wtns::Witness, ProvingKey};

#[path = "msmcommon/mod.rs"]
mod common;
#[path = "gpulock/mod.rs"]
mod gpulock;

use common::{
    argmin, exclusive, fill, general_count, general_scalars, median, read_bytes, read_words,
    storage_words, witness_shaped, SENTINEL,
};

// ---------------------------------------------------------------------------
// Device and pipelines, built once for the whole binary
// ---------------------------------------------------------------------------

fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

fn digits() -> &'static MsmDigits {
    static D: OnceLock<MsmDigits> = OnceLock::new();
    D.get_or_init(|| MsmDigits::new(floor()).expect("digit pipelines"))
}

fn points() -> &'static MsmPointsG1 {
    static P: OnceLock<MsmPointsG1> = OnceLock::new();
    P.get_or_init(|| MsmPointsG1::new(floor()).expect("G1 point pipelines"))
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Random points, one scalar multiplication each. Fine up to a few thousand.
fn rand_bases(n: usize, rng: &mut impl Rng) -> Vec<G1Affine> {
    let proj: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(rng)).collect();
    G1Projective::normalize_batch(&proj)
}

/// Bases at 2^16 without paying for 65,536 scalar multiplications: an arithmetic progression
/// of points. Structure in the bases cannot help or hurt an MSM, which only ever adds them,
/// and the scalars are still random.
fn walk_bases(n: usize, rng: &mut impl Rng) -> Vec<G1Affine> {
    let step = G1Projective::rand(rng);
    let mut cur = G1Projective::rand(rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    G1Projective::normalize_batch(&proj)
}

/// The base vector as the device sees it: `PackedG1Affine`, 64 bytes, infinity mapped onto
/// the all-zero encoding by reading arkworks' flag rather than by hoping `x` and `y` are zero.
fn base_words(bases: &[G1Affine]) -> Vec<u32> {
    let packed = PackedG1Affine::pack_slice(bases);
    let mut out = Vec::with_capacity(packed.len() * 16);
    for p in &packed {
        out.extend_from_slice(&p.x.v);
        out.extend_from_slice(&p.y.v);
    }
    out
}

/// Every artifact under `bench/artifacts` that has both a key and a witness.
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
// Device plumbing
// ---------------------------------------------------------------------------

/// What one full-device MSM produced, plus enough of its scratch to check the over-runs.
struct Run {
    result: G1Projective,
    buckets: Vec<u32>,
    spill_rows: Vec<u32>,
    results_bytes: usize,
    ones_off: u64,
    rows: u32,
    spill_slots: u32,
    ones_groups: u32,
    n_windows: u32,
    c: u32,
}

/// The scalar and base vectors, packed and resident on the device.
///
/// Split out of [`run_msm`] because packing 65,536 `G1Affine` into `PackedG1Affine` and
/// writing 4 MB of them is host work a proof does **once**, at key load, and it is larger
/// than the MSM. Timing it as part of the MSM is what `what_one_g1_msm_costs_against_the_cpu`
/// used to do and it made the device look four times worse than it is.
struct Uploaded {
    scalars: wgpu::Buffer,
    bases: wgpu::Buffer,
}

fn upload(bases: &[G1Affine], all_scalars: &[Fr]) -> Uploaded {
    let b = floor();
    let (words, _) = pack_scalars(all_scalars);
    Uploaded {
        scalars: storage_words(b, "u9 scalars", &words),
        bases: storage_words(b, "u9 bases", &base_words(bases)),
    }
}

/// The whole MSM: counting sort then the five point kernels, one encoder, one submit.
///
/// `slack` extra elements are allocated on every output the kernels index from zero and
/// pre-filled with [`SENTINEL`]. The two reduction outputs are *windows* into `results` with
/// an explicit binding size, so an over-run there is dropped by WebGPU rather than landing in
/// the slack; the binding size is the guard for those two and the sentinel is the guard for
/// the bucket array and the spill arrays.
fn run_msm(
    bases: &[G1Affine],
    all_scalars: &[Fr],
    scalar_off: u32,
    base_off: u32,
    n: u32,
    c: Option<u32>,
    slack: u32,
) -> Run {
    let up = upload(bases, all_scalars);
    run_msm_on(&up, all_scalars, scalar_off, base_off, n, c, slack)
}

/// Same, on a base and scalar vector that is already on the device.
#[allow(clippy::too_many_arguments)]
fn run_msm_on(
    up: &Uploaded,
    all_scalars: &[Fr],
    scalar_off: u32,
    base_off: u32,
    n: u32,
    c: Option<u32>,
    slack: u32,
) -> Run {
    let b = floor();
    let d = digits();
    let p = points();

    let general = general_count(&all_scalars[scalar_off as usize..(scalar_off + n) as usize]);

    let dplan = match c {
        Some(c) => DigitPlan::with_c(n, scalar_off, Some(general), c),
        None => DigitPlan::new(n, scalar_off, Some(general)),
    }
    .expect("digit plan");
    let pplan = p.plan_points(&dplan, base_off).expect("point plan");

    let scalars = &up.scalars;
    let bases_buf = &up.bases;
    let sort = DigitBuffers::new(b, &dplan, slack).expect("digit buffers");
    let pts = PointBuffers::new(b, &dplan, &pplan, slack, p.curve()).expect("point buffers");
    for buf in [&sort.counts, &sort.cursor, &sort.entries] {
        fill(b, buf, SENTINEL);
    }
    for buf in [&pts.buckets, &pts.spill_pts, &pts.spill_rows, &pts.results] {
        fill(b, buf, SENTINEL);
    }

    let slots = d.sort_slots(&dplan) + p.slots(&dplan, &pplan) + 4;
    let mut ring = ParamRing::new(b, "u9 ring", slots).expect("ring");
    let soff = d.plan_sort(&dplan, &mut ring).expect("plan sort");
    let poff = p.plan(&dplan, &pplan, &mut ring).expect("plan points");
    ring.flush(b);

    let sbind = d
        .bind_sort(b, &ring, &dplan, scalars, &sort)
        .expect("bind sort");
    let pbind = p
        .bind_all(b, &ring, &dplan, &pplan, scalars, bases_buf, &sort, &pts)
        .expect("bind points");

    let mut enc = b.device().create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        d.encode_sort(&mut pass, &dplan, &sbind, &soff)
            .expect("encode sort");
        p.encode(&mut pass, &dplan, &pplan, &pbind, &poff)
            .expect("encode points");
    }
    b.submit([enc.finish()]);
    b.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    assert!(b.take_error().is_none(), "device error during the G1 MSM");

    let raw = read_bytes(b, &pts.results, pts.results.size());
    let result = p
        .combine(&raw, &dplan, &pplan, &pts)
        .expect("combine the window sums");

    Run {
        result,
        buckets: read_words(b, &pts.buckets),
        spill_rows: read_words(b, &pts.spill_rows),
        results_bytes: raw.len(),
        ones_off: pts.ones_offset(),
        rows: dplan.rows(),
        spill_slots: pplan.spill_slots(&dplan),
        ones_groups: pplan.ones_groups(),
        n_windows: dplan.n_windows(),
        c: dplan.c(),
    }
}

impl Run {
    /// Every element past the range a kernel owns still holds the sentinel.
    fn assert_no_overrun(&self, slack: u32, what: &str) {
        if slack == 0 {
            return;
        }
        let words_per_point = (wgsl::G1.point_bytes / 4) as usize;
        for i in self.rows as usize..(self.rows + slack) as usize {
            let at = i * words_per_point;
            assert_eq!(
                &self.buckets[at..at + words_per_point],
                &vec![SENTINEL; words_per_point][..],
                "{what}: msm_clear_g1 or msm_segmented_g1 wrote bucket {i}, which is past the \
                 {} rows this plan has",
                self.rows
            );
        }
        for i in self.spill_slots as usize..(self.spill_slots + slack) as usize {
            assert_eq!(
                self.spill_rows[i], SENTINEL,
                "{what}: msm_segmented_g1 tagged spill slot {i}, which is past the {} slots \
                 this plan has",
                self.spill_slots
            );
        }
    }
}

/// The oracle, and the assertion that the comparison is not vacuous.
fn assert_matches_cpu(run: &Run, bases: &[G1Affine], scalars: &[Fr], what: &str) {
    let want = CpuMsm::new().msm_g1(bases, scalars);
    assert_eq!(
        run.result, want,
        "{what}: the device MSM at c = {} disagrees with g16-msm's CPU Pippenger",
        run.c
    );
}

// ---------------------------------------------------------------------------
// 1. The acceptance: every length, every window width
// ---------------------------------------------------------------------------

#[test]
fn the_g1_msm_matches_cpu_pippenger_at_every_length() {
    let mut rng = test_rng();
    let mut checked = 0;
    // 128 and 129 straddle slice_len, which is where the G2 merge's dropped-spill bug lived
    // and where the bisect that found it stopped. 63/64/65 and 256/257 straddle the two
    // plausible workgroup sizes for the same reason.
    for n in [
        1usize, 2, 3, 63, 64, 65, 127, 128, 129, 130, 200, 256, 257, 1000,
    ] {
        let bases = rand_bases(n, &mut rng);
        let scalars = general_scalars(n, &mut rng);
        let run = run_msm(&bases, &scalars, 0, 0, n as u32, None, 3);
        assert!(
            !run.result.is_zero(),
            "n = {n}: the whole MSM came out as the identity, so this comparison proves \
             nothing"
        );
        assert_matches_cpu(&run, &bases, &scalars, &format!("n = {n}"));
        run.assert_no_overrun(3, &format!("n = {n}"));
        checked += 1;
    }
    println!("{checked} lengths matched the CPU Pippenger exactly");
}

#[test]
fn the_g1_msm_matches_cpu_pippenger_at_every_window_width() {
    let mut rng = test_rng();
    let n = 700usize;
    let bases = rand_bases(n, &mut rng);
    let scalars = general_scalars(n, &mut rng);
    // Every c the digit plan accepts, 16 included. G2 has to stop at 15 because a 16-window
    // by 32768-bucket array of 256-byte accumulators is 134.2 MB against a 128 MiB storage
    // binding floor; the same array over G1 is 67.1 MB, so this is the one shape where the
    // two curves genuinely differ rather than merely costing different amounts.
    for c in [2u32, 3, 5, 8, 11, 12, 13, 14, 15, 16] {
        let run = run_msm(&bases, &scalars, 0, 0, n as u32, Some(c), 2);
        assert!(!run.result.is_zero(), "c = {c}: identity result");
        assert_matches_cpu(&run, &bases, &scalars, &format!("c = {c}"));
        run.assert_no_overrun(2, &format!("c = {c}"));
    }
    println!("10 window widths from c = 2 to c = 16 matched the CPU Pippenger exactly");
}

#[test]
fn the_widest_window_g2_cannot_hold_fits_over_g1() {
    let b = floor();
    let p = points();
    // The exact allocation `tests/msm_g2.rs` asserts is refused: 16 windows x 32768 buckets.
    // Over G1 that is 524,288 rows of 128 bytes = 67,108,864, half the 128 MiB floor, so it
    // is allocated rather than refused. The point is that the check is a byte count and not
    // a hard-coded curve rule.
    let dplan = DigitPlan::with_c(1 << 16, 0, Some(1 << 16), 16).expect("plan");
    let pplan = p.plan_points(&dplan, 0).expect("point plan");
    let bufs = PointBuffers::new(b, &dplan, &pplan, 1, p.curve()).expect("a G1 c = 16 array");
    assert_eq!(
        u64::from(dplan.rows()) * wgsl::G1.point_bytes,
        67_108_864,
        "the arithmetic this test rests on has moved"
    );
    assert!(bufs.buckets.size() >= 67_108_864);
    println!(
        "c = 16 over G1: {} bytes of buckets, allocated, where G2 is refused at {} bytes",
        bufs.buckets.size(),
        u64::from(dplan.rows()) * wgsl::G2.point_bytes
    );
}

// ---------------------------------------------------------------------------
// 2. The scalar classes a witness actually contains
// ---------------------------------------------------------------------------

#[test]
fn the_degenerate_scalar_vectors_a_real_witness_contains() {
    let mut rng = test_rng();
    let n = 500usize;
    let bases = rand_bases(n, &mut rng);

    // All zero. Every kernel runs, every bucket stays at the identity, and the answer is the
    // identity. This is the one case where a kernel that wrote nothing would also be right,
    // which is why the sentinel check below matters more here than anywhere else.
    let zeros = vec![Fr::zero(); n];
    let run = run_msm(&bases, &zeros, 0, 0, n as u32, None, 2);
    assert!(
        run.result.is_zero(),
        "all-zero scalars gave a nonzero point"
    );
    assert_matches_cpu(&run, &bases, &zeros, "all zero");
    run.assert_no_overrun(2, "all zero");

    // All one. Nothing reaches a bucket at all and the entire answer comes from msm_ones_g1,
    // which is the path a bit-heavy circuit spends most of its bases on.
    let ones = vec![Fr::one(); n];
    let run = run_msm(&bases, &ones, 0, 0, n as u32, None, 2);
    let want: G1Projective = bases.iter().map(|b| G1Projective::from(*b)).sum();
    assert_eq!(
        run.result, want,
        "all-one scalars are not the sum of the bases"
    );
    assert_matches_cpu(&run, &bases, &ones, "all one");
    run.assert_no_overrun(2, "all one");

    // The real shape: the bucket path and the ones path both run, and their two
    // contributions have to be added exactly once each.
    for ppm in [10_000u32, 100_000, 500_000] {
        let mixed = witness_shaped(n, ppm, &mut rng);
        let run = run_msm(&bases, &mixed, 0, 0, n as u32, None, 2);
        assert!(!run.result.is_zero(), "ppm {ppm}: identity result");
        assert_matches_cpu(&run, &bases, &mixed, &format!("witness shaped, {ppm} ppm"));
        run.assert_no_overrun(2, &format!("witness shaped, {ppm} ppm"));
    }
    println!("all-zero, all-one and three witness-shaped mixtures matched");
}

// ---------------------------------------------------------------------------
// 3. The branches a random draw will not reach
// ---------------------------------------------------------------------------

#[test]
fn a_base_at_infinity_is_skipped_and_does_not_poison_its_bucket() {
    let mut rng = test_rng();
    let n = 200usize;
    let mut bases = rand_bases(n, &mut rng);
    // snarkjs zkeys really do contain these in the A, B and C query vectors. ICME's WGSL MSM
    // turns (0, 0) into a live point and poisons whichever bucket it lands in.
    for i in (0..n).step_by(7) {
        bases[i] = G1Affine::identity();
    }
    let scalars = general_scalars(n, &mut rng);
    let run = run_msm(&bases, &scalars, 0, 0, n as u32, None, 2);
    assert!(!run.result.is_zero());
    assert_matches_cpu(&run, &bases, &scalars, "with infinity bases");

    // And the same input with the infinities dropped from both sides must give the same
    // point, which is the property that says they were skipped rather than merely tolerated.
    let (kept_b, kept_s): (Vec<G1Affine>, Vec<Fr>) = bases
        .iter()
        .zip(&scalars)
        .filter(|(b, _)| !b.infinity)
        .map(|(b, s)| (*b, *s))
        .unzip();
    assert_eq!(kept_b.len(), n - n.div_ceil(7));
    assert_eq!(
        run.result,
        CpuMsm::new().msm_g1(&kept_b, &kept_s),
        "dropping the infinity bases changed the answer, so they were not skipped"
    );

    // Every base at infinity: the answer is the identity however general the scalars are,
    // and a kernel that lifted (0, 0) to a live point would produce something else.
    let all_inf = vec![G1Affine::identity(); n];
    let run = run_msm(&all_inf, &scalars, 0, 0, n as u32, None, 2);
    assert!(
        run.result.is_zero(),
        "an all-infinity base vector gave a nonzero point"
    );
    println!("29 of 200 bases at infinity skipped exactly, and 200 of 200 as well");
}

#[test]
fn two_equal_and_two_opposite_bases_in_one_bucket_take_the_branches_they_should() {
    let mut rng = test_rng();
    let p = G1Affine::rand(&mut rng);
    let k = Fr::rand(&mut rng);

    // Equal bases with equal scalars: every window puts both copies in the same bucket, so
    // pt_madd_g1's `pp == 0 && rr == 0` arm (mdbl-2008-s on the affine point) runs in every
    // nonzero window. Nothing about a real zkey guarantees distinct query entries.
    let bases = vec![p, p];
    let scalars = vec![k, k];
    let run = run_msm(&bases, &scalars, 0, 0, 2, None, 2);
    assert_eq!(run.result, G1Projective::from(p) * k * Fr::from(2u64));
    assert_matches_cpu(&run, &bases, &scalars, "two equal bases");

    // Opposite bases with equal scalars: same bucket, `pp == 0 && rr != 0`, so the pair
    // cancels to the identity and the whole MSM is the identity.
    let bases = vec![p, -p];
    let scalars = vec![k, k];
    let run = run_msm(&bases, &scalars, 0, 0, 2, None, 2);
    assert!(
        run.result.is_zero(),
        "P and -P with the same scalar did not cancel"
    );
    assert_matches_cpu(&run, &bases, &scalars, "two opposite bases");

    // And a run of the same point longer than one slice, which forces the doubling arm and
    // then the general pt_add_g1 in the merge and the reduction rather than only in the
    // accumulation. 300 is over slice_len, so the run really does span slices.
    let n = 300usize;
    assert!(n as u32 > wgsl::G1.slice_len);
    let bases = vec![p; n];
    let scalars = general_scalars(n, &mut rng);
    let run = run_msm(&bases, &scalars, 0, 0, n as u32, None, 2);
    let sum: Fr = scalars.iter().sum();
    assert_eq!(run.result, G1Projective::from(p) * sum);
    println!("the equal, opposite and repeated-base arms of pt_madd_g1 all agree");
}

// ---------------------------------------------------------------------------
// 4. Offsets, which are how the five MSMs of a proof share one buffer
// ---------------------------------------------------------------------------

#[test]
fn a_nonzero_scalar_offset_and_base_offset_read_the_right_windows() {
    let mut rng = test_rng();
    let total = 400usize;
    let n = 150u32;
    let scalar_off = 37u32;
    let base_off = 91u32;
    // Deliberately different offsets: the L MSM reads scalars from n_public + 1 and bases
    // from 0, so a kernel that used one offset for both would pass a test where they agree.
    let bases = rand_bases(total, &mut rng);
    let scalars = general_scalars(total, &mut rng);
    let run = run_msm(&bases, &scalars, scalar_off, base_off, n, None, 2);
    let want = CpuMsm::new().msm_g1(
        &bases[base_off as usize..(base_off + n) as usize],
        &scalars[scalar_off as usize..(scalar_off + n) as usize],
    );
    assert!(!run.result.is_zero());
    assert_eq!(run.result, want, "the offsets select the wrong sub-vectors");

    // The same lengths with the offsets swapped must give a different answer, or the test
    // above would pass with the two offsets confused.
    let other = CpuMsm::new().msm_g1(
        &bases[scalar_off as usize..(scalar_off + n) as usize],
        &scalars[base_off as usize..(base_off + n) as usize],
    );
    assert_ne!(
        want, other,
        "the two offsets happened to select the same thing"
    );

    // And again with scalars that are sometimes exactly 1, which is the case the general
    // vector above cannot reach: `msm_ones_*` is the only kernel that reads the scalar
    // buffer and the base vector in the same statement, so it is the only place the two
    // offsets can be confused, and with no 1-scalars it contributes nothing and the
    // confusion is invisible. A mutation that made the ones pass read `BASES[scalar_off + i]`
    // passed every test in this file except the artifact one until this block existed.
    let mixed = witness_shaped(total, 200_000, &mut rng);
    let ones_here = mixed[scalar_off as usize..(scalar_off + n) as usize]
        .iter()
        .filter(|x| x.is_one())
        .count();
    assert!(
        ones_here > 10,
        "only {ones_here} of {n} scalars are exactly 1, so the ones path is barely running"
    );
    let run = run_msm(&bases, &mixed, scalar_off, base_off, n, None, 2);
    assert_eq!(
        run.result,
        CpuMsm::new().msm_g1(
            &bases[base_off as usize..(base_off + n) as usize],
            &mixed[scalar_off as usize..(scalar_off + n) as usize],
        ),
        "the offsets select the wrong sub-vectors once the ones path carries part of the sum"
    );
    println!(
        "scalar_off {scalar_off} and base_off {base_off} over {n} of {total}, \
         {ones_here} of them through msm_ones_g1"
    );
}

// ---------------------------------------------------------------------------
// 5. The 128-byte point's binding alignment, which G2 gets for free
// ---------------------------------------------------------------------------

#[test]
fn the_ones_window_is_aligned_even_when_n_windows_is_odd() {
    // A storage binding offset must be a multiple of 256 in every browser. G2's point is
    // exactly 256 bytes so `n_windows * 256` is aligned whatever `n_windows` is; G1's is 128,
    // so an odd `n_windows` lands on a 128-byte boundary and the offset has to be rounded up.
    // c = 14 gives ceil(255 / 14) = 19 windows, which is odd, and 19 * 128 = 2432 is not a
    // multiple of 256.
    let b = floor();
    let p = points();
    let mut odd = 0;
    for c in 2u32..=16 {
        let dplan = DigitPlan::with_c(64, 0, Some(64), c).expect("plan");
        let pplan = p.plan_points(&dplan, 0).expect("point plan");
        let bufs = PointBuffers::new(b, &dplan, &pplan, 0, p.curve()).expect("buffers");
        let raw = u64::from(dplan.n_windows()) * wgsl::G1.point_bytes;
        let off = bufs.ones_offset();
        assert_eq!(
            off % 256,
            0,
            "c = {c}: the ones offset {off} is not aligned"
        );
        assert!(
            off >= raw,
            "c = {c}: the ones window at {off} overlaps the {raw} bytes of window sums"
        );
        if !raw.is_multiple_of(256) {
            odd += 1;
            assert_eq!(off, raw + 128, "c = {c}: rounded further than one point");
        }
    }
    // If this ever hits zero the test above has stopped exercising the rounding at all.
    assert!(
        odd > 0,
        "no window count in 2..=16 produced an unaligned offset, so the rounding is untested"
    );
    println!("{odd} of 15 window widths need the ones offset rounded up over G1");
}

// ---------------------------------------------------------------------------
// 6. The size the acceptance names, and real witnesses
// ---------------------------------------------------------------------------

#[test]
fn the_g1_msm_matches_cpu_pippenger_at_2_16() {
    let _gpu = exclusive();
    let mut rng = test_rng();
    let n = 1usize << 16;
    let bases = walk_bases(n, &mut rng);
    // 5% general, which is a realistic witness and still 3,277 scalars in the buckets, so
    // both paths carry real work. A fully general 2^16 is the sweep's input, not this one.
    let scalars = witness_shaped(n, 50_000, &mut rng);
    let run = run_msm(&bases, &scalars, 0, 0, n as u32, None, 1);
    assert!(!run.result.is_zero());
    assert_matches_cpu(&run, &bases, &scalars, "2^16");
    run.assert_no_overrun(1, "2^16");
    println!("2^16 bases at c = {} matched the CPU Pippenger", run.c);
}

/// The A and L MSMs of every artifact, on their real bases and their real witness.
///
/// This is the case no generated input reaches, and it reaches it in a different direction
/// from the one the design expects. Two things it pins:
///
/// **The base vectors really do contain points at infinity.** 1,187 of `js_16x16_d32`'s
/// 140,824 A-query bases are the identity, and the count is about 0.85% on every artifact.
/// The prior art (ICME's WGSL MSM) lifts `(0, 0)` to a live projective point and poisons
/// whichever bucket it lands in, which on this input would be wrong on every proof rather
/// than on an unlucky one. Nothing generated in this file puts the identity where a real key
/// puts it.
///
/// **And these witnesses are not sparse, which the design assumes they are.**
/// `gen::points`' `msm_ones_*` comment says "a bit-heavy circuit is over 99% zeros and ones",
/// and the printed table below says 95.9% to 98.2% of the entries are *general* on all six
/// artifacts, which is the opposite. The one-scalar path is therefore carrying almost nothing
/// on this repo's circuits, and the window model in `crate::msm::window_size`, which sizes
/// from the general count, is being handed nearly the whole witness. That does not make the
/// kernel wrong (a witness that is 98% general is exactly what the bucket path is for) and it
/// is not this unit's to fix, but the justification written next to the kernel is not what
/// these artifacts contain. U12 owns the cost model and should know.
///
/// The two MSMs also exercise the two offset shapes a proof actually uses: A reads scalars
/// and bases both from 0, L reads scalars from `n_public + 1` and bases from 0.
#[test]
fn real_witnesses_from_the_artifacts_match_the_cpu_pippenger() {
    let found = artifacts();
    assert!(
        !found.is_empty(),
        "no artifacts under bench/artifacts; the symlink into the main worktree is missing \
         and this test would otherwise pass by doing nothing"
    );
    let cpu = CpuMsm::new();
    // Which witness shapes the corpus actually covers. Asserted across the corpus at the end
    // rather than per artifact, because both shapes are real and neither is the normal one.
    let mut saw_mostly_general = None::<String>;
    let mut saw_mostly_bits = None::<String>;
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        assert_eq!(pk.a_query.len(), pk.n_vars);
        assert!(w.len() >= pk.n_vars);
        let w = &w[..pk.n_vars];

        let general = general_count(w);
        let inf = pk.a_query.iter().filter(|p| p.infinity).count();

        // A: scalars and bases both from 0, over the whole witness.
        let n = pk.n_vars as u32;
        let run = run_msm(&pk.a_query, w, 0, 0, n, None, 0);
        assert!(!run.result.is_zero(), "{name}: the A MSM is the identity");
        assert_eq!(
            run.result,
            cpu.msm_g1(&pk.a_query, w),
            "{name}: the A MSM disagrees with the CPU Pippenger at c = {}",
            run.c
        );

        // L: scalars from n_public + 1, bases from 0, and the two ranges have different
        // lengths from the A MSM's, which is what a proof actually dispatches.
        let skip = pk.n_public + 1;
        let ln = pk.l_query.len() as u32;
        assert_eq!(pk.l_query.len(), pk.n_vars - skip);
        let run_l = run_msm(&pk.l_query, w, skip as u32, 0, ln, None, 0);
        assert_eq!(
            run_l.result,
            cpu.msm_g1(&pk.l_query, &w[skip..]),
            "{name}: the L MSM disagrees with the CPU Pippenger at c = {}",
            run_l.c
        );

        println!(
            "{name:16} n_vars {n:7} general {general:7} ({:.1}%) infinity bases {inf:5}  \
             A at c = {}, L over {ln} at c = {}",
            100.0 * general as f64 / n as f64,
            run.c,
            run_l.c
        );
        // The two properties the paragraph above rests on, asserted rather than eyeballed, so
        // a future artifact set that loses either one says so instead of quietly making this
        // test a duplicate of the random one. `tiny_mul` is six variables and one infinity
        // base, which is too small to say anything about proportions.
        if pk.n_vars > 1000 {
            assert!(
                inf > 0,
                "{name}: no base at infinity in the A query, so the skip path is untested here"
            );
            // Both shapes exist in the wild and the two take different paths through the MSM:
            // a mostly-general witness drives the bucket machinery and hands `window_size`
            // nearly the whole witness, while a mostly-0/1 one drives `msm_ones` and leaves
            // the buckets nearly empty. The joinsplit artifacts are 96% to 98% general; the
            // anon-aadhaar key is 7.9%, which is 92% zeros and ones over 1.1M variables. An
            // earlier version of this test asserted every artifact was mostly general, which
            // was true of the six circuits that existed when it was written and stopped being
            // true the moment a real-world key landed in the tree. The property worth holding
            // is that the corpus covers both, not that every member looks the same.
            if general * 10 > n * 9 {
                saw_mostly_general.get_or_insert_with(|| name.clone());
            }
            if general * 10 < n {
                saw_mostly_bits.get_or_insert_with(|| name.clone());
            }
        }
    }
    assert!(
        saw_mostly_general.is_some(),
        "no artifact has a mostly-general witness, so the bucket path is only covered by the \
         random tests and this one has become a duplicate of them"
    );
    assert!(
        saw_mostly_bits.is_some(),
        "no artifact has a mostly-zeros-and-ones witness, so the msm_ones path is only \
         covered by synthetic scalars here"
    );
    println!(
        "corpus covers both witness shapes: mostly general via {}, mostly 0/1 via {}",
        saw_mostly_general.unwrap(),
        saw_mostly_bits.unwrap()
    );
}

// ---------------------------------------------------------------------------
// 7. The browser floor, from the host layouts and from the generator
// ---------------------------------------------------------------------------

#[test]
fn every_g1_pipeline_layout_declares_at_most_eight_storage_buffers() {
    // 8 is `maxStorageBuffersPerShaderStage` at the spec floor. This adapter reports 9 under
    // strict compliance and 29 without it, so a kernel at 9 passes every device test here and
    // fails in a browser. Counted from the entry lists the layouts are actually built from,
    // not from a second copy.
    let want = [
        (wgsl::G1.entry_clear(), wgsl::STORAGE_CLEAR),
        (wgsl::G1.entry_segmented(), wgsl::STORAGE_SEGMENTED),
        (wgsl::G1.entry_merge(), wgsl::STORAGE_MERGE),
        (wgsl::G1.entry_reduce(), wgsl::STORAGE_REDUCE),
        (wgsl::G1.entry_ones(), wgsl::STORAGE_ONES),
    ];
    let got = MsmPointsG1::storage_buffer_counts();
    assert_eq!(got.len(), want.len());
    for ((gname, gn), (wname, wn)) in got.iter().zip(want) {
        assert_eq!(*gname, wname);
        assert_eq!(
            *gn, wn,
            "{gname} declares {gn} storage buffers and gen::points says {wn}"
        );
        assert!(
            *gn <= 8,
            "{gname} declares {gn} storage buffers, the browser floor allows 8"
        );
        println!("{gname:20} {gn} storage buffers");
    }
}

#[test]
fn every_g1_entry_point_fits_the_workgroup_storage_floor() {
    // Computed from the generator rather than from the device, because this adapter grants
    // 32768 bytes and a browser grants 16384. `tg = 128` is 16384 exactly over G1, which
    // passes with nothing to spare, and `tg = 256` is 32768, which passes on this Mac under
    // `Raised` and fails in every browser.
    let curve = wgsl::G1;
    let mut widest = 0u64;
    for tg in [1u32, 2, 4, 8, 16, 32, 64, 128] {
        let src = wgsl::points_module_at(
            g16_wgpu::gen::field::Variant::default(),
            curve,
            curve.with_tg(tg),
        );
        let bytes = curve.workgroup_bytes(tg);
        assert!(
            bytes <= wgsl::FLOOR_WORKGROUP_BYTES,
            "tg = {tg} is {bytes} bytes of workgroup storage"
        );
        // And the number in the doc header is the number the declaration produces, so a
        // future edit cannot make the two disagree.
        assert!(
            src.contains(&format!("array<{}, {tg}>", curve.pt)),
            "tg = {tg}: the module does not declare array<{}, {tg}>",
            curve.pt
        );
        widest = widest.max(bytes);
        println!("tg {tg:3}  array<{}, {tg}> = {bytes:5} B", curve.pt);
    }
    assert_eq!(
        widest,
        wgsl::FLOOR_WORKGROUP_BYTES,
        "tg = 128 should be exactly the floor over G1"
    );
    assert_eq!(curve.max_tg(), 128);

    // And past the floor the generator refuses rather than emitting a module that compiles
    // here and fails in Chrome.
    let err = std::panic::catch_unwind(|| {
        wgsl::points_module_at(
            g16_wgpu::gen::field::Variant::default(),
            curve,
            curve.with_tg(256),
        )
    })
    .unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "not a string panic".into());
    assert!(
        msg.contains("32768") && msg.contains("browser floor"),
        "tg = 256 should be refused by byte count, and the panic says: {msg}"
    );
    println!("tg = 256 refused: {msg}");
}

#[test]
fn the_g1_module_carries_no_fq2_it_never_calls() {
    // An Fq2 prelude in a G1 module is 6.6 KiB naga parses, validates and dead-strips. U8
    // measured what that is worth end to end (about 2 ms of module time per module and no
    // pipeline time), so this is small; it is asserted rather than left to drift because the
    // check costs nothing and "the generator emits what this curve needs" is the property
    // that makes one file serve two curves.
    let v = g16_wgpu::gen::field::Variant::default();
    let g1 = wgsl::points_module(v, wgsl::G1);
    let g2 = wgsl::points_module(v, wgsl::G2);
    assert!(
        !g1.contains("fn fq2_mul("),
        "the G1 module declares fq2_mul"
    );
    assert!(!g1.contains("struct Fq2"), "the G1 module declares Fq2");
    assert!(g2.contains("fn fq2_mul("), "the G2 module lost fq2_mul");
    // And every routine the G1 kernels do call is there under the right prefix.
    for f in [
        "fn fq_mul(",
        "fn fq_sqr(",
        "fn fq_neg(",
        "fn pt_madd_g1(",
        "fn pt_add_g1(",
        "fn pt_dbl_g1(",
        "fn pt_dbl_affine_g1(",
        "fn pt_mul_small_g1(",
    ] {
        assert!(g1.contains(f), "the G1 module is missing {f}");
    }
    for e in wgsl::G1.entries() {
        assert!(g1.contains(&format!("fn {e}(")), "no entry point {e}");
        assert!(!g2.contains(&format!("fn {e}(")), "{e} leaked into G2");
    }
    println!(
        "G1 module {} KiB, G2 module {} KiB, {} KiB of that the Fq2 prelude",
        g1.len() / 1024,
        g2.len() / 1024,
        (g2.len() - g1.len()) / 1024
    );
}

#[test]
fn the_g1_module_compiles_and_reports_what_it_cost() {
    let p = points();
    let cost = p.cost();
    println!("{}", p.summary());
    println!(
        "  {} KiB of WGSL, module {:.1} ms, pipelines {:.1} ms",
        p.source_len() / 1024,
        cost.module_us as f64 / 1e3,
        cost.pipeline_us as f64 / 1e3
    );
    // Design §9 risk 2 puts the kill line at 10 s of MSM pipeline creation in Chrome, and
    // research file 03 recorded 129 s natively for one monolithic WGSL shader with an
    // unrolled multiply inlined at a dozen sites. This module inlines `fq_mul` at about
    // sixty. If that number were going to bite anywhere it would bite here, so it is asserted
    // rather than printed.
    assert!(
        cost.compile_us < 5_000_000,
        "the G1 point module took {:.1} s to build",
        cost.compile_us as f64 / 1e6
    );
    assert_eq!(cost.pipelines, 5);
    assert_eq!(cost.modules, 1);
}

// ---------------------------------------------------------------------------
// 8. The three constants design §4 assigns and this unit measures, for G1
// ---------------------------------------------------------------------------

/// One timed point stage: `reps` repetitions of the five kernels in one submit, differenced
/// against one repetition so the submit and the fence cancel.
///
/// The five-kernel sequence is idempotent, which is what makes repeating it legal: `clear`
/// resets every bucket's `zz`, `segmented` rewrites the direct buckets and both spill slots
/// of every slice, and `merge` re-reads the buckets it wrote. Running it twice gives the same
/// bucket array, so the second repetition is the same work and not a different one.
struct Bench {
    scalars: wgpu::Buffer,
    bases: wgpu::Buffer,
    sort: DigitBuffers,
    dplan: DigitPlan,
}

impl Bench {
    fn new(n: u32, c: u32) -> Self {
        let b = floor();
        let d = digits();
        let mut rng = test_rng();
        let bases = walk_bases(n as usize, &mut rng);
        let scalars = general_scalars(n as usize, &mut rng);
        let (words, general) = pack_scalars(&scalars);
        let dplan = DigitPlan::with_c(n, 0, Some(general), c).expect("plan");
        let sbuf = storage_words(b, "u9 bench scalars", &words);
        let sort = DigitBuffers::new(b, &dplan, 0).expect("digit buffers");

        // The counting sort runs once and its output is reused by every timed repetition,
        // because none of the five point kernels writes any of it.
        let mut ring =
            ParamRing::new(b, "u9 bench sort ring", d.sort_slots(&dplan) + 4).expect("ring");
        let soff = d.plan_sort(&dplan, &mut ring).expect("plan");
        ring.flush(b);
        let sbind = d
            .bind_sort(b, &ring, &dplan, &sbuf, &sort)
            .expect("bind sort");
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            d.encode_sort(&mut pass, &dplan, &sbind, &soff)
                .expect("encode");
        }
        b.submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        Self {
            scalars: sbuf,
            bases: storage_words(b, "u9 bench bases", &base_words(&bases)),
            sort,
            dplan,
        }
    }

    /// Microseconds per repetition of `what`, submit cost differenced out.
    fn time(&self, p: &MsmPointsG1, slice_len: u32, reps: u32, what: Stage) -> f64 {
        let b = floor();
        let pplan =
            PointPlan::with_slice_len(&self.dplan, 0, p.workgroups().tg, slice_len).expect("plan");
        let pts = PointBuffers::new(b, &self.dplan, &pplan, 0, p.curve()).expect("buffers");
        let mut ring =
            ParamRing::new(b, "u9 bench ring", p.slots(&self.dplan, &pplan) + 4).expect("ring");
        let poff = p.plan(&self.dplan, &pplan, &mut ring).expect("plan");
        ring.flush(b);
        let bind = p
            .bind_all(
                b,
                &ring,
                &self.dplan,
                &pplan,
                &self.scalars,
                &self.bases,
                &self.sort,
                &pts,
            )
            .expect("bind");

        let once = |n: u32| -> f64 {
            let mut enc = b.device().create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                // Setup, and only as much of it as the timed kernel needs. Giving every case
                // the whole five-kernel stage and differencing a 4-microsecond kernel against
                // it reports zero, which is what the first version of the G2 harness did to
                // `msm_clear_g2`.
                match what {
                    Stage::All | Stage::Accumulate | Stage::Clear | Stage::Segmented => {}
                    Stage::Merge => {
                        p.encode_clear(&mut pass, &self.dplan, &bind, &poff)
                            .expect("encode");
                        p.encode_segmented(&mut pass, &self.dplan, &pplan, &bind, &poff)
                            .expect("encode");
                    }
                    Stage::Reduction => p
                        .encode(&mut pass, &self.dplan, &pplan, &bind, &poff)
                        .expect("encode"),
                }
                for _ in 0..n {
                    match what {
                        Stage::All => p
                            .encode(&mut pass, &self.dplan, &pplan, &bind, &poff)
                            .expect("encode"),
                        Stage::Accumulate => {
                            p.encode_clear(&mut pass, &self.dplan, &bind, &poff)
                                .expect("encode");
                            p.encode_segmented(&mut pass, &self.dplan, &pplan, &bind, &poff)
                                .expect("encode");
                            p.encode_merge(&mut pass, &self.dplan, &bind, &poff)
                                .expect("encode");
                        }
                        Stage::Clear => p
                            .encode_clear(&mut pass, &self.dplan, &bind, &poff)
                            .expect("encode"),
                        Stage::Segmented => p
                            .encode_segmented(&mut pass, &self.dplan, &pplan, &bind, &poff)
                            .expect("encode"),
                        Stage::Merge => p
                            .encode_merge(&mut pass, &self.dplan, &bind, &poff)
                            .expect("encode"),
                        Stage::Reduction => {
                            p.encode_reduce(&mut pass, &self.dplan, &bind, &poff)
                                .expect("encode");
                            p.encode_ones(&mut pass, &pplan, &bind, &poff)
                                .expect("encode");
                        }
                    }
                }
            }
            let t = std::time::Instant::now();
            b.submit([enc.finish()]);
            b.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            t.elapsed().as_secs_f64() * 1e6
        };
        // Warm, then difference `reps` extra repetitions against zero extra.
        let _ = once(0);
        let base = once(0);
        let full = once(reps);
        ((full - base) / reps as f64).max(0.0)
    }
}

/// Which kernels a timed repetition contains.
///
/// Splitting the point stage in two is what makes the sweeps readable: the reduction and the
/// accumulation differ by an order of magnitude, so timing all five together buries whichever
/// one the sweep is varying under the other.
#[derive(Clone, Copy)]
enum Stage {
    /// All five, for a whole-stage figure.
    #[allow(dead_code)]
    All,
    /// clear, segmented, merge. What `slice_len` and the three 1D workgroup sizes move.
    Accumulate,
    /// reduce, ones. What `tg` moves.
    Reduction,
    /// One kernel on its own, for the workgroup sweep.
    ///
    /// Repeating `msm_merge_g1` alone adds every bucket's spills again, so after the second
    /// repetition the bucket array is wrong. That is deliberate and it is safe here: the work
    /// per repetition is identical, nothing reads the result, and every timed run is thrown
    /// away. No correctness test uses this path.
    Clear,
    Segmented,
    Merge,
}

fn shipped_size(wg: &wgsl::Workgroups, entry: &str) -> u32 {
    if entry == wgsl::G1.entry_clear() {
        wg.clear
    } else if entry == wgsl::G1.entry_segmented() {
        wg.segmented
    } else {
        wg.merge
    }
}

#[test]
fn the_g1_workgroup_sizes_are_measured() {
    // Timing. Takes the cross-process lock, because cargo runs the test binaries in parallel
    // and a sibling binary saturating the GPU makes every number here fiction.
    let _gpu_timing = gpulock::exclusive_gpu();
    let _gpu = exclusive();
    let bench = Bench::new(32_768, 12);
    let sizes = [16u32, 32, 64, 128, 256];
    let shipped = wgsl::G1.wg;

    // Each kernel timed ALONE, at repetition counts that put every cell in the milliseconds.
    // `msm_segmented_g1` is the overwhelming majority of the accumulation and `msm_clear_g1`
    // is a fraction of a percent of it, so a shared repetition count would report the clear's
    // whole row as noise.
    let cases: [(String, Stage, u32); 3] = [
        (wgsl::G1.entry_clear(), Stage::Clear, 2000),
        (wgsl::G1.entry_segmented(), Stage::Segmented, 8),
        (wgsl::G1.entry_merge(), Stage::Merge, 20),
    ];

    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, stage, reps) in cases {
        let mut row = Vec::new();
        for &sz in &sizes {
            let mut wg = shipped;
            match stage {
                Stage::Clear => wg.clear = sz,
                Stage::Segmented => wg.segmented = sz,
                _ => wg.merge = sz,
            }
            let p = MsmPointsG1::with_shape(floor(), wg, 65535).expect("pipelines");
            row.push(median(
                (0..5)
                    .map(|_| bench.time(&p, wgsl::G1.slice_len, reps, stage))
                    .collect(),
            ));
        }
        rows.push((name, row));
    }

    println!(
        "32,768 general scalars, c = 12, slice_len {}, each kernel alone, medians of five, us",
        wgsl::G1.slice_len
    );
    print!("{:20}", "kernel");
    for s in sizes {
        print!("{s:>9}");
    }
    println!("{:>7}{:>7}", "ships", "best");
    for (name, row) in &rows {
        let ships = shipped_size(&shipped, name);
        let best = argmin(row);
        print!("{name:20}");
        for v in row {
            print!("{v:9.1}");
        }
        println!("{ships:>7}{:>7}", sizes[best]);
    }

    // The assertion is that the shipped size is the best of the five, not that today's
    // numbers are pinned, so if Tint reverses the ranking at U14 this fails loudly rather
    // than silently keeping a size that stopped being right.
    for (name, row) in &rows {
        let ships = shipped_size(&shipped, name);
        let best = argmin(row);
        let at_shipped = row[sizes.iter().position(|&s| s == ships).unwrap()];
        let regret = at_shipped / row[best] - 1.0;
        println!(
            "{name}: ships {ships}, best {} ({:+.1}%)",
            sizes[best],
            regret * 100.0
        );
        // Relative for the two kernels that cost milliseconds, absolute for `msm_clear_g1`,
        // whose whole row is a few tens of microseconds and moves by 2x between runs. A 3%
        // bound on a 10-microsecond row would be a coin flip in CI.
        assert!(
            regret < 0.03 || at_shipped - row[best] < 25.0,
            "{name} ships {ships} at {at_shipped:.1} us and {} is {:.1} us ({:.1}%) faster; \
             gen::points::G1's workgroup sizes are stale",
            sizes[best],
            at_shipped - row[best],
            regret * 100.0
        );
    }
}

#[test]
fn the_g1_reduction_threadgroup_is_measured() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let _gpu = exclusive();
    let bench = Bench::new(32_768, 12);
    let curve = wgsl::G1;

    // Up to 128, where G2 stops at 64: `array<PtG1, 128>` is 16384 bytes, exactly the floor.
    println!("32,768 general scalars, c = 12, reduce + ones only, medians of five, us");
    println!("{:>4} {:>10} {:>10}", "tg", "shared B", "us");
    let mut best = (f64::MAX, 0u32);
    let mut shipped_us = 0.0;
    for tg in [8u32, 16, 32, 64, 128] {
        let p = MsmPointsG1::with_shape(floor(), curve.with_tg(tg), 65535).expect("pipelines");
        let us = median(
            (0..5)
                .map(|_| bench.time(&p, curve.slice_len, 5, Stage::Reduction))
                .collect(),
        );
        println!("{tg:>4} {:>10} {us:>10.1}", curve.workgroup_bytes(tg));
        if us < best.0 {
            best = (us, tg);
        }
        if tg == curve.wg.tg {
            shipped_us = us;
        }
    }
    let regret = shipped_us / best.0 - 1.0;
    println!(
        "ships tg = {} at {shipped_us:.1} us, best tg = {} at {:.1} us ({:+.1}%)",
        curve.wg.tg,
        best.1,
        best.0,
        regret * 100.0
    );
    assert_eq!(
        best.1, curve.wg.tg,
        "the shipped tg is not the measured winner; gen::points::G1.wg.tg is stale"
    );
    assert!(regret < 0.01, "shipped tg regret {:.1}%", regret * 100.0);
    assert_eq!(
        curve.workgroup_bytes(128),
        wgsl::FLOOR_WORKGROUP_BYTES,
        "G1's widest reduction should be exactly the floor"
    );
}

#[test]
fn the_g1_slice_length_is_measured() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let _gpu = exclusive();
    let bench = Bench::new(32_768, 12);
    let p = points();
    println!("32,768 general scalars, c = 12, clear + segmented + merge, medians of three, us");
    let lens = [16u32, 32, 64, 128, 256, 512, 1024];
    let mut row = Vec::new();
    for &l in &lens {
        row.push(median(
            (0..3)
                .map(|_| bench.time(p, l, 3, Stage::Accumulate))
                .collect(),
        ));
    }
    print!("{:12}", "slice_len");
    for l in lens {
        print!("{l:>9}");
    }
    println!();
    print!("{:12}", "us");
    for v in &row {
        print!("{v:9.1}");
    }
    println!();
    let best = argmin(&row);
    let at_shipped = row[lens
        .iter()
        .position(|&l| l == wgsl::G1.slice_len)
        .expect("the shipped slice_len is not in the swept set")];
    let regret = at_shipped / row[best] - 1.0;
    println!(
        "ships {}, best {} ({:+.1}%)",
        wgsl::G1.slice_len,
        lens[best],
        regret * 100.0
    );
    assert_eq!(
        lens[best],
        wgsl::G1.slice_len,
        "the shipped slice_len is not the measured winner; gen::points::G1.slice_len is stale"
    );
    assert!(
        regret < 0.01,
        "shipped slice_len regret {:.1}%",
        regret * 100.0
    );
}

// ---------------------------------------------------------------------------
// 9. The plan arithmetic, on the host, with no device
// ---------------------------------------------------------------------------

#[test]
fn the_point_plan_derives_its_shape_once() {
    let d = DigitPlan::with_c(1000, 0, Some(1000), 12).expect("plan");
    let p = PointPlan::new(&d, 0, 32, wgsl::G1).expect("point plan");
    // Derived from the constant, never retyped: the G2 suite shipped two stale expectations
    // written as literals and both had to be fixed when a constant moved under them.
    assert_eq!(p.slice_len(), wgsl::G1.slice_len);
    assert_eq!(p.slices(), 1000u32.div_ceil(wgsl::G1.slice_len));
    assert_eq!(p.seg_threads(&d), d.n_windows() * p.slices());
    assert_eq!(p.spill_slots(&d), 2 * p.seg_threads(&d));
    // ceil(1000 / (32 * 64)) = 1, and it is never zero: a dispatch of zero workgroups writes
    // nothing and the host would then add a stale slot.
    assert_eq!(p.ones_groups(), 1000u32.div_ceil(32 * 64).clamp(1, 64));

    // An empty window still gets one slice, because the segmented pass tags its spill slots
    // before it tests for an empty window.
    let empty = DigitPlan::with_c(500, 0, Some(0), 12).expect("plan");
    assert_eq!(empty.cap(), 1);
    assert_eq!(PointPlan::new(&empty, 0, 32, wgsl::G1).unwrap().slices(), 1);

    // The parameter block carries both halves, and the digit half is untouched.
    let params = p.params(&d, 7);
    assert_eq!(params.c, d.c());
    assert_eq!(params.n_windows, d.n_windows());
    assert_eq!(params.cap, d.cap());
    assert_eq!(params.lo, 7);
    assert_eq!(params.slice_len, p.slice_len());
    assert_eq!(params.slices, p.slices());
    assert_eq!(params.ones_groups, p.ones_groups());
    assert_eq!(std::mem::size_of_val(&params), 48);

    assert!(PointPlan::with_slice_len(&d, 0, 32, 0).is_err());
    assert!(PointPlan::new(&d, 0, 0, wgsl::G1).is_err());
    println!("the point plan's five derived numbers agree with their definitions");
}

#[test]
fn a_plan_built_for_the_wrong_reduction_width_is_refused() {
    let b = floor();
    let p = points();
    let d = DigitPlan::with_c(64, 0, Some(64), 8).expect("plan");
    let wrong = p.workgroups().tg / 2;
    let pplan = PointPlan::new(&d, 0, wrong, wgsl::G1).expect("point plan");
    let sort = DigitBuffers::new(b, &d, 0).expect("digit buffers");
    let pts = PointBuffers::new(b, &d, &pplan, 0, p.curve()).expect("point buffers");
    let ring = ParamRing::new(b, "u9 ring", 8).expect("ring");
    let scalars = storage_words(b, "u9 scalars", &vec![0u32; 64 * LIMBS]);
    let bases = storage_words(b, "u9 bases", &vec![0u32; 64 * 16]);
    let err = match p.bind_all(b, &ring, &d, &pplan, &scalars, &bases, &sort, &pts) {
        Err(e) => e,
        Ok(_) => panic!(
            "a plan built for tg = {wrong} bound against a tg = {} module",
            p.workgroups().tg
        ),
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("reduction") && msg.contains("ones_groups"),
        "the error should name the mismatch, and it says: {msg}"
    );
    println!("tg mismatch refused: {msg}");
}

#[test]
fn a_short_buffer_is_refused_rather_than_read_as_zero() {
    // WebGPU returns zero for an out-of-range storage read and drops an out-of-range write,
    // both silently, so a base vector one point short is a wrong MSM and not an error. Every
    // length check in `bind_all` is here because nothing downstream would notice.
    let b = floor();
    let p = points();
    let d = DigitPlan::with_c(64, 0, Some(64), 8).expect("plan");
    let pplan = p.plan_points(&d, 0).expect("point plan");
    let sort = DigitBuffers::new(b, &d, 0).expect("digit buffers");
    let pts = PointBuffers::new(b, &d, &pplan, 0, p.curve()).expect("point buffers");
    let ring = ParamRing::new(b, "u9 ring", 8).expect("ring");
    let full_scalars = storage_words(b, "u9 scalars", &vec![0u32; 64 * LIMBS]);
    let full_bases = storage_words(b, "u9 bases", &vec![0u32; 64 * 16]);

    // One point short, and one scalar short.
    let short_bases = storage_words(b, "u9 short bases", &vec![0u32; 63 * 16]);
    let short_scalars = storage_words(b, "u9 short scalars", &vec![0u32; 63 * LIMBS]);
    for (what, sc, ba) in [
        ("the base vector", &full_scalars, &short_bases),
        ("the scalar buffer", &short_scalars, &full_bases),
    ] {
        let err = match p.bind_all(b, &ring, &d, &pplan, sc, ba, &sort, &pts) {
            Err(e) => format!("{e}"),
            Ok(_) => panic!("{what} one element short was accepted"),
        };
        assert!(err.contains(what), "the error should name {what}: {err}");
        println!("{err}");
    }
}

// ---------------------------------------------------------------------------
// 10. What it actually costs, against the CPU it has to beat
// ---------------------------------------------------------------------------

/// The number this unit exists to produce, published whether or not it is flattering.
///
/// **It is not flattering. One G1 MSM on the device is slower than the same MSM on twelve CPU
/// threads, at every size this test runs.** The ratio column below is under 1 everywhere.
///
/// The reason is occupancy, and the sweeps in this file are what say so rather than a guess.
/// At n = 32,768 and c = 12 the three kernels that cost anything run these thread counts:
///
/// * `msm_segmented_g1` gets one thread per slice per window, which is
///   `n_windows * ceil(cap / slice_len)` = 22 * 256 = **5,632 threads**, each doing 128 mixed
///   additions in a row. That is 6.5M `fq_mul` in 28.7 ms, about 226 M/s against the 1,964
///   M/s this crate measured for the multiply itself: **12% of the field throughput**.
/// * `msm_reduce_g1` gets one workgroup per window, 22 * 128 = **2,816 threads**, on a
///   machine with 38 cores. 2M `fq_mul` in 17.9 ms is 110 M/s, **5.6% of peak**.
///
/// Both kernels are latency-bound on a machine that is nowhere near full, and both get better
/// with n: the artifacts are 140k variables and the design targets 2^18, which is 8x the work
/// per thread count here. Whether that is enough is U11's number to take, on a whole proof,
/// and the fix if it is not is structural rather than a constant: give each window several
/// workgroups and reduce their partials, which is a kernel change and not a sweep. This unit
/// owes U11 and U12 the measurement and the diagnosis, and there is no assertion on the ratio
/// because a threshold here would just be a number someone picked.
#[test]
fn what_one_g1_msm_costs_against_the_cpu() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let _gpu = exclusive();
    let p = points();
    let cpu = CpuMsm::new();
    println!(
        "one G1 MSM, M2 Max, wgpu through naga to MSL, release. slice_len {}, tg {}.",
        wgsl::G1.slice_len,
        p.workgroups().tg
    );
    println!(
        "{:>8} {:>9} {:>4} {:>10} {:>10} {:>10} {:>8}",
        "n", "general", "c", "upload ms", "device ms", "cpu ms", "ratio"
    );
    for (n, ppm) in [
        (4_096usize, 1_000_000u32),
        (32_768, 1_000_000),
        (65_536, 50_000),
    ] {
        let mut rng = test_rng();
        let bases = walk_bases(n, &mut rng);
        let scalars = if ppm == 1_000_000 {
            general_scalars(n, &mut rng)
        } else {
            witness_shaped(n, ppm, &mut rng)
        };
        let general = general_count(&scalars);

        // Packing and uploading is prepare work, so it is done and then not timed. It is not
        // small: 65,536 bases is 4 MB of `PackedG1Affine` built one Montgomery limb split at
        // a time, and including it made the 65,536 row read 57.7 ms instead of the figure
        // below.
        let t = std::time::Instant::now();
        let up = upload(&bases, &scalars);
        let upload_ms = t.elapsed().as_secs_f64() * 1e3;

        // Warm: the first run of a shape pays for allocating the bucket array and the spill
        // arrays, which a prover pools across proofs.
        let _ = run_msm_on(&up, &scalars, 0, 0, n as u32, None, 0);
        let t = std::time::Instant::now();
        let run = run_msm_on(&up, &scalars, 0, 0, n as u32, None, 0);
        let device_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let want = cpu.msm_g1(&bases, &scalars);
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            run.result, want,
            "n = {n}: the timed run disagreed with the CPU"
        );
        // The readback is one copy of both outputs, and design §3 caps it at 64 KiB across
        // all five MSMs of a proof.
        assert!(
            run.results_bytes < 65_536,
            "one G1 MSM read back {} bytes: {} windows plus {} ones groups from {}",
            run.results_bytes,
            run.n_windows,
            run.ones_groups,
            run.ones_off
        );

        println!(
            "{n:>8} {general:>9} {:>4} {upload_ms:>10.1} {device_ms:>10.1} {cpu_ms:>10.1} \
             {:>7.2}x",
            run.c,
            cpu_ms / device_ms
        );
    }
    println!(
        "The device figure is one warm run: the counting sort, the five point kernels, one \
         submit, one readback and the host Horner tail, plus allocating this shape's scratch. \
         It excludes packing and uploading the bases and the scalars, which is prepare work \
         and is the upload column. The CPU figure is g16-msm's rayon Pippenger on {} threads, \
         which is not the comparison that matters (design §8 U12 owns that) but is the only \
         oracle this unit has.",
        CpuMsm::new().threads
    );
}
