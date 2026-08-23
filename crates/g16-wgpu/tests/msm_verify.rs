//! An independent audit of the two device MSMs, written by someone who wrote none of them.
//!
//! # Why this file exists next to `tests/msm_g1.rs` and `tests/msm_g2.rs`
//!
//! Those two suites are thorough and they share one blind spot: their oracle is
//! `g16_msm::CpuMsm`, a Pippenger over **the same signed recoding** the WGSL `sc_digit`
//! reimplements, and `gen::msm` says so in as many words ("identical digit for digit to
//! `g16_msm::signed_digit`"). A recoding that is wrong in the same way on both sides is
//! invisible to them. Everything here compares against `sum_i s_i * P_i` computed one
//! `mul_bigint` at a time, which shares no code with either MSM, no window width, no bucket
//! and no digit.
//!
//! # The second thing, which is a shape and not an oracle
//!
//! `msm_merge_*` folds a bucket's spills over the slices `[k_lo, k_hi]` its run reaches, and
//! `k_hi` is the line that shipped wrong once already. The bug that shipped, `k_hi = k_lo`,
//! drops every slice past the first; the neighbouring mistake, `k_hi = k_lo + 1u`, drops
//! every slice past the second and needs a bucket whose run crosses **two** boundaries to
//! show at all.
//!
//! **The existing suites do catch both, and the claim that they do not was checked before it
//! was believed.** Applying `k_hi = k_lo + 1u` and running only `tests/msm_g1.rs`,
//! `tests/msm_g2.rs` and `tests/proof.rs` fails `the_g1_msm_matches_cpu_pippenger_at_every_
//! window_width` and `real_witnesses_from_the_artifacts_match_the_cpu_pippenger`. The reason
//! is not the length sweep, which never puts 129 entries in one bucket: it is `c = 2`, where
//! 700 entries fall into two buckets and every run is 350 long, and it is the real witnesses,
//! whose repeated values are the imbalance the segmented pass exists for (`gen::points`
//! quotes a busiest bucket of 11,758 against a mean of 17 on `js_2x2_d32`).
//!
//! What is missing is not the coverage but any statement of it. Both sources are incidental:
//! one is the far end of a window-width sweep and the other is a property of six artifacts on
//! a gitignored symlink, and nothing anywhere asserts that a bucket run ever crosses a slice
//! boundary. Change `slice_len`, drop `c = 2` from the sweep, or run without artifacts, and
//! the multi-slice merge stops being exercised with no test going red.
//!
//! So the tests here build the case on purpose and assert the shape they built. Repeating one
//! scalar reproduces the distribution exactly, because equal scalars have equal digits and
//! therefore share a bucket in **every** window;
//! [`one_bucket_holding_a_run_longer_than_several_slices_is_merged_whole`] does that from
//! `slice_len - 1` to `5 * slice_len + 7` and asserts the observed span, recomputed from the
//! counts and cursors the kernels themselves wrote.
//!
//! Native only, for the reason in `tests/device.rs`. Fixed seeds throughout.

use std::sync::OnceLock;

use ark_std::{rand::Rng, test_rng, UniformRand};
use g16_field::{
    CurveGroup, Fr, G1Affine, G1Projective, G2Affine, G2Projective, PrimeField, PrimeGroup, Zero,
};
use g16_gpu_layout::{PackedG1Affine, PackedG2Affine};
use g16_wgpu::gen::points as wgsl;
use g16_wgpu::msm::{pack_scalars, DigitBuffers, DigitPlan, MsmDigits};
use g16_wgpu::points::{MsmPointsG1, MsmPointsG2, PointBuffers};
use g16_wgpu::{LimitsProfile, ParamRing, WgpuBackend};

#[path = "msmcommon/mod.rs"]
mod common;
#[path = "gpulock/mod.rs"]
mod gpulock;

use common::{fill, general_scalars, read_bytes, read_words, storage_words, SENTINEL};

// ---------------------------------------------------------------------------
// Device, built once for the binary
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

fn g1() -> &'static MsmPointsG1 {
    static P: OnceLock<MsmPointsG1> = OnceLock::new();
    P.get_or_init(|| MsmPointsG1::new(floor()).expect("G1 point pipelines"))
}

fn g2() -> &'static MsmPointsG2 {
    static P: OnceLock<MsmPointsG2> = OnceLock::new();
    P.get_or_init(|| MsmPointsG2::new(floor()).expect("G2 point pipelines"))
}

/// Serialises this binary's tests. They all dispatch on the one device above, and the
/// sentinel assertions read back scratch that a concurrent test would be writing.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// The oracle: no Pippenger, no windows, no recoding
// ---------------------------------------------------------------------------

/// `sum_i s_i * P_i`, one double-and-add ladder per term.
///
/// Deliberately the slowest possible implementation. It shares nothing with either MSM
/// beyond arkworks' group law, which is what makes it able to disagree with both of them at
/// once. `mul_bigint` rather than `Mul<Fr>` so nothing here depends on how a scalar is
/// represented either.
fn naive_g1(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    assert_eq!(bases.len(), scalars.len());
    let mut acc = G1Projective::zero();
    for (b, s) in bases.iter().zip(scalars) {
        acc += G1Projective::from(*b).mul_bigint(s.into_bigint());
    }
    acc
}

fn naive_g2(bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
    assert_eq!(bases.len(), scalars.len());
    let mut acc = G2Projective::zero();
    for (b, s) in bases.iter().zip(scalars) {
        acc += G2Projective::from(*b).mul_bigint(s.into_bigint());
    }
    acc
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

fn rand_g1(n: usize, rng: &mut impl Rng) -> Vec<G1Affine> {
    let proj: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(rng)).collect();
    G1Projective::normalize_batch(&proj)
}

fn rand_g2(n: usize, rng: &mut impl Rng) -> Vec<G2Affine> {
    let proj: Vec<G2Projective> = (0..n).map(|_| G2Projective::rand(rng)).collect();
    G2Projective::normalize_batch(&proj)
}

fn g1_words(bases: &[G1Affine]) -> Vec<u32> {
    let packed = PackedG1Affine::pack_slice(bases);
    let mut out = Vec::with_capacity(packed.len() * 16);
    for p in &packed {
        out.extend_from_slice(&p.x.v);
        out.extend_from_slice(&p.y.v);
    }
    out
}

fn g2_words(bases: &[G2Affine]) -> Vec<u32> {
    let packed = PackedG2Affine::pack_slice(bases);
    let mut out = Vec::with_capacity(packed.len() * 32);
    for p in &packed {
        out.extend_from_slice(&p.x.c0.v);
        out.extend_from_slice(&p.x.c1.v);
        out.extend_from_slice(&p.y.c0.v);
        out.extend_from_slice(&p.y.c1.v);
    }
    out
}

// ---------------------------------------------------------------------------
// One MSM, with every output buffer poisoned first
// ---------------------------------------------------------------------------

/// What one device MSM produced, plus the scratch the over-run assertions need.
struct Run<P> {
    result: P,
    buckets: Vec<u32>,
    spill_rows: Vec<u32>,
    /// `(row, spill slots this bucket's run reaches)` for the fattest bucket of window 0.
    /// The point of the fat-bucket tests is that this exceeds 1, and a test that asserts on
    /// the answer alone cannot say whether it did.
    widest_run_slices: u32,
    rows: u32,
    spill_slots: u32,
}

impl<P> Run<P> {
    fn assert_no_overrun(&self, point_bytes: u64, slack: u32, what: &str) {
        let words = (point_bytes / 4) as usize;
        for i in self.rows as usize..(self.rows + slack) as usize {
            let at = i * words;
            assert_eq!(
                &self.buckets[at..at + words],
                &vec![SENTINEL; words][..],
                "{what}: a point kernel wrote bucket {i}, past the {} rows this plan has",
                self.rows
            );
        }
        for i in self.spill_slots as usize..(self.spill_slots + slack) as usize {
            assert_eq!(
                self.spill_rows[i], SENTINEL,
                "{what}: msm_segmented wrote spill slot {i}, past the {} this plan has",
                self.spill_slots
            );
        }
    }
}

/// The five point kernels behind one counting sort, on whichever curve `run` is asked for.
///
/// A macro rather than a trait: the two `MsmPoints` instantiations differ only in the
/// projective type they return and the packed base layout, and threading that through a
/// trait object would add a layer to read past for two call sites.
macro_rules! run_msm {
    ($points:expr, $curve:expr, $bases_words:expr, $scalars:expr, $n:expr, $c:expr, $slack:expr) => {{
        let (sw, general) = pack_scalars($scalars);
        run_msm_words!(
            digits(),
            $points,
            $curve,
            $bases_words,
            &sw,
            general,
            $n,
            $c,
            $slack,
            None
        )
    }};
}

/// The same, on chosen pipelines, a scalar buffer that is already `u32` words, and an
/// optional poison written over the live part of `spill_rows` before the dispatch.
///
/// Each extra exists for one test and none of the three can be written without it:
/// `pack_scalars` only produces canonical field elements, the multi-dispatch path is a
/// property of the `MsmDigits`/`MsmPoints` instance rather than of the plan, and the
/// spill-tag poison is the only way to reproduce a pooled buffer that still holds the
/// previous proof's tags.
macro_rules! run_msm_words {
    ($digits:expr, $points:expr, $curve:expr, $bases_words:expr, $words:expr, $general:expr,
     $n:expr, $c:expr, $slack:expr, $poison:expr) => {{
        let b = floor();
        let d = $digits;
        let p = $points;
        let n: u32 = $n;
        let slack: u32 = $slack;

        let general: u32 = $general;
        let dplan = match $c {
            Some(c) => DigitPlan::with_c(n, 0, Some(general), c),
            None => DigitPlan::new(n, 0, Some(general)),
        }
        .expect("digit plan");
        let pplan = p.plan_points(&dplan, 0).expect("point plan");

        let scalars_buf = storage_words(b, "verify scalars", $words);
        let bases_buf = storage_words(b, "verify bases", $bases_words);

        let sort = DigitBuffers::new(b, &dplan, slack).expect("digit buffers");
        let pts = PointBuffers::new(b, &dplan, &pplan, slack, $curve).expect("point buffers");
        // Every output is poisoned, so a kernel that writes nothing cannot agree with a
        // zeroed oracle and an out-of-range write, which WebGPU drops in silence, is visible
        // as a surviving sentinel rather than as nothing at all.
        for buf in [&sort.counts, &sort.cursor, &sort.entries] {
            fill(b, buf, SENTINEL);
        }
        for buf in [&pts.buckets, &pts.spill_pts, &pts.spill_rows, &pts.results] {
            fill(b, buf, SENTINEL);
        }
        // The sentinel is neither a row number nor NO_ROW, so a merge that read an untagged
        // slot would ignore it, which is exactly why a sentinel fill cannot see a missing
        // tag. `$poison` writes real row numbers over the live part of the array instead,
        // leaving the slack at the sentinel so the over-run assertion still works.
        if let Some(rows) = $poison {
            let rows: &[u32] = rows;
            assert_eq!(
                rows.len(),
                pplan.spill_slots(&dplan) as usize,
                "the poison is not the length of the spill array"
            );
            b.queue()
                .write_buffer(&pts.spill_rows, 0, bytemuck::cast_slice(rows));
        }

        let slots = d.sort_slots(&dplan) + p.slots(&dplan, &pplan) + 4;
        let mut ring = ParamRing::new(b, "verify ring", slots).expect("ring");
        let soff = d.plan_sort(&dplan, &mut ring).expect("plan sort");
        let poff = p.plan(&dplan, &pplan, &mut ring).expect("plan points");
        ring.flush(b);

        let sbind = d
            .bind_sort(b, &ring, &dplan, &scalars_buf, &sort)
            .expect("bind sort");
        let pbind = p
            .bind_all(
                b,
                &ring,
                &dplan,
                &pplan,
                &scalars_buf,
                &bases_buf,
                &sort,
                &pts,
            )
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
        assert!(b.take_error().is_none(), "device error during the MSM");

        let raw = read_bytes(b, &pts.results, pts.results.size());
        let result = p
            .combine(&raw, &dplan, &pplan, &pts)
            .expect("combine the window sums");

        // How many slices the fattest bucket spans, which is `k_hi - k_lo + 1` in
        // `msm_merge_*`. Recomputed from the counts and cursors the kernels wrote, not from
        // the plan, so it reports what the device actually built.
        //
        // Over **every** window, not window 0. A repeated scalar occupies one bucket per
        // window, but the window whose digit is zero has no bucket at all, and for a random
        // scalar at c = 5 that is one window in sixteen. Scanning window 0 alone therefore
        // made the premise a property of the draw; scanning all of them makes it a property
        // of the scalar, and a nonzero scalar has a nonzero digit somewhere by definition.
        let counts = read_words(b, &sort.counts);
        let cursor = read_words(b, &sort.cursor);
        let mut widest = 0u32;
        for w in 0..dplan.n_windows() {
            let base = w * dplan.cap();
            for j in 0..dplan.n_buckets() {
                let row = (w * dplan.n_buckets() + j) as usize;
                let cnt = counts[row];
                if cnt == 0 || cnt == SENTINEL {
                    continue;
                }
                let end = cursor[row] - base;
                let start = end - cnt;
                let span = (end - 1) / pplan.slice_len() - start / pplan.slice_len() + 1;
                widest = widest.max(span);
            }
        }

        Run {
            result,
            buckets: read_words(b, &pts.buckets),
            spill_rows: read_words(b, &pts.spill_rows),
            widest_run_slices: widest,
            rows: dplan.rows(),
            spill_slots: pplan.spill_slots(&dplan),
        }
    }};
}

// ---------------------------------------------------------------------------
// 1. The whole G1 MSM against a naive sum, over a grid of lengths and widths
// ---------------------------------------------------------------------------

/// Every length that straddles a structural boundary, at seven window widths, against an
/// oracle that shares no code with the device.
///
/// The lengths are the two workgroup sizes the point kernels ship (128 and 256), `slice_len`
/// and its multiples, and the odd numbers around each, because every one of those is a place
/// a `<=` can be a `<`. The widths span the whole range `DigitPlan` accepts: `c = 2` gives 2
/// buckets and 128 windows, `c = 16` gives 32,768 buckets and 16, and the reduction's
/// segment arithmetic (`ceil(n_buckets / tg)` with `tg = 128`) is a different shape at each
/// end.
#[test]
fn the_g1_msm_equals_a_naive_sum_of_scalar_multiples() {
    let _one = exclusive();
    let mut rng = test_rng();
    let lengths = [
        1usize, 2, 3, 5, 7, 63, 64, 65, 127, 128, 129, 130, 255, 256, 257, 383, 384, 385, 511, 512,
        513,
    ];
    let widths = [2u32, 4, 8, 11, 12, 13, 16];
    let mut cases = 0usize;
    for n in lengths {
        let bases = rand_g1(n, &mut rng);
        let scalars = general_scalars(n, &mut rng);
        let want = naive_g1(&bases, &scalars);
        assert!(
            !want.is_zero(),
            "n = {n}: the oracle itself is the identity, so nothing below proves anything"
        );
        let words = g1_words(&bases);
        for c in widths {
            let run = run_msm!(g1(), wgsl::G1, &words, &scalars, n as u32, Some(c), 2);
            assert_eq!(
                run.result, want,
                "n = {n}, c = {c}: the device MSM disagrees with sum_i s_i * P_i"
            );
            run.assert_no_overrun(wgsl::G1.point_bytes, 2, &format!("n = {n}, c = {c}"));
            cases += 1;
        }
    }
    println!("{cases} (length, window width) pairs matched a naive scalar-multiplication sum");
}

// ---------------------------------------------------------------------------
// 2. One bucket whose run crosses more than one slice boundary, on purpose
// ---------------------------------------------------------------------------

/// `m` copies of one scalar over `m` distinct bases, so every window has exactly one occupied
/// bucket holding all `m` entries.
///
/// Each slice of the run sees no bucket change, so every one of them spills its whole
/// partial to its head slot and `BUCKETS[row]` is left at the identity by `msm_clear_*`; the answer is
/// therefore **entirely** the merge's fold, and a merge that stops early returns a proper
/// prefix of it. At `m = 5 * slice_len + 7` that is 6 slices, so `k_hi = k_lo` returns about a
/// sixth of the right point and `k_hi = k_lo + 1u` about a third.
///
/// The span is asserted, not assumed. `c = 2` in the sweep above and the real witnesses in
/// `tests/msm_g1.rs` both happen to produce runs this long, and neither says so, so both can
/// stop doing it without a test going red. This one cannot: it recomputes the fattest
/// bucket's span from the counts and cursors the kernels wrote and fails if it is not
/// `ceil(m / slice_len)`.
#[test]
fn one_bucket_holding_a_run_longer_than_several_slices_is_merged_whole() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G1.slice_len;
    let mut cases = 0usize;
    for m in [
        sl - 1,
        sl,
        sl + 1,
        2 * sl,
        2 * sl + 1,
        3 * sl + 1,
        4 * sl,
        5 * sl + 7,
    ] {
        let n = m as usize;
        let bases = rand_g1(n, &mut rng);
        let k = general_scalars(1, &mut rng)[0];
        let scalars = vec![k; n];
        let want = naive_g1(&bases, &scalars);
        assert!(!want.is_zero(), "m = {m}: degenerate oracle");

        let run = run_msm!(g1(), wgsl::G1, &g1_words(&bases), &scalars, m, None, 2);
        // The claim this test rests on, asserted rather than assumed: the fattest bucket's
        // run really does reach this many slices. Without it a change to `slice_len` or to
        // the digit plan could quietly make every case here a one-slice run again.
        let want_slices = m.div_ceil(sl);
        // The answer first and the premise second, deliberately. A run whose dispatches did
        // not all take effect shows up in both, and the useful message is the one that says
        // the MSM is wrong rather than the one that says a count came back small.
        assert_eq!(
            run.result, want,
            "m = {m} copies of one scalar: the device MSM disagrees with the naive sum, over \
             a bucket run that should span {want_slices} slices"
        );
        assert_eq!(
            run.widest_run_slices, want_slices,
            "m = {m}: the fattest bucket spans {} slices, not the {want_slices} this test \
             needs to exercise the merge's slice loop",
            run.widest_run_slices
        );
        run.assert_no_overrun(wgsl::G1.point_bytes, 2, &format!("m = {m}"));
        cases += 1;
    }
    println!(
        "{cases} single-bucket runs of up to {} slices merged whole",
        (5 * sl + 7).div_ceil(sl)
    );
}

/// The same fat run with sparse buckets around it, so the segmented pass takes all three of
/// its paths in one dispatch: a head spill, a tail spill, and a run wholly inside one slice
/// that is written straight to `BUCKETS[row]` with no spill at all.
///
/// The direct-write path is the one that is invisible in the pure fat-bucket case above and
/// the multi-slice merge is the one that is invisible in a purely random draw. Both at once
/// is the only arrangement where a merge that folded the spills into a bucket some other
/// thread had already direct-written would double-count.
#[test]
fn a_fat_bucket_beside_sparse_ones_merges_both_kinds() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G1.slice_len as usize;
    for (fat, sparse) in [(3 * sl + 5, 200usize), (2 * sl + 1, 900), (7 * sl, 64)] {
        let n = fat + sparse;
        let bases = rand_g1(n, &mut rng);
        let k = general_scalars(1, &mut rng)[0];
        let mut scalars = vec![k; fat];
        scalars.extend(general_scalars(sparse, &mut rng));
        // Interleave, so the fat bucket's entries are not also contiguous in base order and
        // the scatter has to place them by digit rather than by index.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }
        let scalars: Vec<Fr> = order.iter().map(|&i| scalars[i]).collect();

        let want = naive_g1(&bases, &scalars);
        assert!(!want.is_zero(), "fat = {fat}: degenerate oracle");
        let run = run_msm!(
            g1(),
            wgsl::G1,
            &g1_words(&bases),
            &scalars,
            n as u32,
            None,
            2
        );
        // A run of `fat` entries starting anywhere touches at least `(fat - 1) / sl + 1`
        // slices, and one more if it happens to start mid-slice. Deriving the floor rather
        // than typing a number, because the number changes with `slice_len`.
        let least = (fat - 1) / sl + 1;
        assert!(
            least >= 3,
            "this case was chosen to force at least three slices and its floor is {least}"
        );
        assert_eq!(
            run.result, want,
            "fat = {fat} repeated scalars beside {sparse} random ones: device MSM disagrees"
        );
        assert!(
            run.widest_run_slices as usize >= least,
            "fat = {fat}, sparse = {sparse}: the fattest bucket spans {} slices, fewer than \
             the {least} a run of {fat} must touch",
            run.widest_run_slices
        );
        run.assert_no_overrun(wgsl::G1.point_bytes, 2, &format!("fat = {fat}"));
    }
    println!("a multi-slice bucket and single-slice buckets in one dispatch agree with the sum");
}

/// The same shape on G2, at one size, because the merge kernel is emitted from the same
/// generator and a curve-specific `k_hi` is not a thing that can happen, but the *plan* is
/// per curve and `slice_len` is read off `wgsl::G2`.
#[test]
fn a_multi_slice_bucket_run_over_g2_is_merged_whole() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G2.slice_len;
    let m = 3 * sl + 5;
    let n = m as usize;
    let bases = rand_g2(n, &mut rng);
    let k = general_scalars(1, &mut rng)[0];
    let scalars = vec![k; n];
    let want = naive_g2(&bases, &scalars);
    assert!(!want.is_zero(), "degenerate oracle");

    let run = run_msm!(g2(), wgsl::G2, &g2_words(&bases), &scalars, m, None, 2);
    assert_eq!(
        run.result, want,
        "m = {m} copies of one scalar over G2: the device MSM disagrees with the naive sum"
    );
    assert_eq!(run.widest_run_slices, m.div_ceil(sl));
    run.assert_no_overrun(wgsl::G2.point_bytes, 2, "g2 fat bucket");
    println!("a {}-slice G2 bucket run merged whole", m.div_ceil(sl));
}

// ---------------------------------------------------------------------------
// 3. The three arms of pt_madd, inside a run long enough to span slices
// ---------------------------------------------------------------------------

/// `P` repeated with one scalar over several slices, and `P, -P` alternating over several
/// slices.
///
/// `tests/msm_g1.rs` builds both arms at n = 2, where the whole run is one entry pair inside
/// one slice. Here each arm runs hundreds of times and the partial results cross slice
/// boundaries, so the doubling arm's output has to survive a spill, a merge and a reduction
/// rather than only the accumulator it was computed in. The alternating case is the sharper
/// of the two: every second `pt_madd` returns the identity, so a slice can spill the identity
/// and the merge has to treat that as a real contribution rather than as an empty slot.
#[test]
fn the_equal_and_opposite_arms_survive_a_run_that_spans_slices() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G1.slice_len as usize;
    let p = G1Affine::rand(&mut rng);
    let k = general_scalars(1, &mut rng)[0];

    // 3 slices and a bit, every entry the same point: the doubling arm on every madd after
    // the first of each slice.
    let n = 3 * sl + 9;
    let bases = vec![p; n];
    let scalars = vec![k; n];
    let run = run_msm!(
        g1(),
        wgsl::G1,
        &g1_words(&bases),
        &scalars,
        n as u32,
        None,
        2
    );
    assert_eq!(run.widest_run_slices, n.div_ceil(sl) as u32);
    assert_eq!(
        run.result,
        naive_g1(&bases, &scalars),
        "{n} copies of one point and one scalar disagreed with the naive sum"
    );

    // P, -P, P, -P ... with one scalar: the cancelling arm on every second madd, and an even
    // count so the whole MSM is the identity. Compared against the naive sum as well, which
    // is the identity for the same reason and by a different route.
    let n = 4 * sl;
    let bases: Vec<G1Affine> = (0..n).map(|i| if i % 2 == 0 { p } else { -p }).collect();
    let scalars = vec![k; n];
    let want = naive_g1(&bases, &scalars);
    assert!(want.is_zero(), "the alternating oracle is not the identity");
    let run = run_msm!(
        g1(),
        wgsl::G1,
        &g1_words(&bases),
        &scalars,
        n as u32,
        None,
        2
    );
    assert_eq!(run.widest_run_slices, (n / sl) as u32);
    assert!(
        run.result.is_zero(),
        "{n} alternating P and -P did not cancel; the cancelling arm loses points across a \
         slice boundary"
    );

    // And an odd count of the same alternation, where the answer is one copy of k*P rather
    // than the identity, so the test above cannot be passed by a kernel that returns zero.
    let n = 4 * sl + 1;
    let bases: Vec<G1Affine> = (0..n).map(|i| if i % 2 == 0 { p } else { -p }).collect();
    let scalars = vec![k; n];
    let want = naive_g1(&bases, &scalars);
    assert!(!want.is_zero());
    let run = run_msm!(
        g1(),
        wgsl::G1,
        &g1_words(&bases),
        &scalars,
        n as u32,
        None,
        2
    );
    assert_eq!(
        run.result, want,
        "an odd alternation of P and -P did not leave exactly one k*P"
    );
    println!("the doubling and cancelling arms agree with the naive sum across 4 slices");
}

/// Bases at infinity **inside** a multi-slice run, which is the arrangement a zkey produces:
/// `js_16x16_d32`'s A query holds 1,187 infinities in 140,824 bases, and repeated witness
/// values put several of them in one bucket.
///
/// A kernel that lifted `(0, 0)` to a live point would poison the fat bucket, and because the
/// fat bucket is the whole answer here the damage is not diluted by 900 correct ones.
#[test]
fn infinity_bases_inside_a_fat_bucket_are_skipped() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G1.slice_len as usize;
    let n = 3 * sl + 11;
    let mut bases = rand_g1(n, &mut rng);
    // Every fifth, which lands infinities at both ends of every slice and in the middle.
    for i in (0..n).step_by(5) {
        bases[i] = G1Affine::identity();
    }
    let k = general_scalars(1, &mut rng)[0];
    let scalars = vec![k; n];
    let want = naive_g1(&bases, &scalars);
    assert!(!want.is_zero());

    let run = run_msm!(
        g1(),
        wgsl::G1,
        &g1_words(&bases),
        &scalars,
        n as u32,
        None,
        2
    );
    assert_eq!(run.widest_run_slices, n.div_ceil(sl) as u32);
    assert_eq!(
        run.result,
        want,
        "{} infinity bases inside a {}-slice run changed the answer",
        n.div_ceil(5),
        n.div_ceil(sl)
    );

    // A whole slice of nothing but infinities, so one thread's entire partial is the
    // identity and it still has to spill a slot the merge reads.
    let mut bases = rand_g1(n, &mut rng);
    for b in bases.iter_mut().take(2 * sl).skip(sl) {
        *b = G1Affine::identity();
    }
    let want = naive_g1(&bases, &scalars);
    assert!(!want.is_zero());
    let run = run_msm!(
        g1(),
        wgsl::G1,
        &g1_words(&bases),
        &scalars,
        n as u32,
        None,
        2
    );
    assert_eq!(
        run.result, want,
        "a slice whose every base is at infinity changed the answer"
    );
    println!("infinity bases inside and filling a slice of a fat run are both skipped exactly");
}

// ---------------------------------------------------------------------------
// 4. Scalars a `Fr` cannot hold
// ---------------------------------------------------------------------------

/// A scalar buffer whose words are not canonical field elements.
///
/// `security/README.md` reads the zkey as trusted, but the scalars are the *witness*, and
/// nothing between `Witness::load` and `PackedScalar::from_fr` re-checks that a limb group is
/// below `r`. On the CPU that is moot: `Fr` is canonical by construction. On the device the
/// eight limbs are read raw by `sc_bits`, and a digit that came out too large would index
/// `COUNTS[w * n_buckets + d - 1]` past its window, which WebGPU turns into a dropped write
/// that this crate would never hear about.
///
/// Two claims, and they are different claims.
///
/// **Below 2^254 the answer is still right.** The recoding is exact for any integer whose
/// **bit 254 is clear**, and 2^254 rather than 2^255 is the bound, which is worth spelling
/// out because the obvious reading of `RECODE_BITS = 255` gives the wrong one and this test
/// was written with the wrong one first. `sc_digit` borrows `2^c` whenever a window's top bit
/// is set and the next window pays it back through its carry; the highest window has no next
/// window, so its top bit, which is bit `W*c - 1 = 254`, must be clear or the borrow is never
/// repaid and the sum comes out `2^255` short. `gen::msm` says exactly that and rests it on
/// every BN254 scalar being below `r < 2^254`. So `u + r` for `u < 2^254 - r` is a
/// non-canonical representative the recoding still handles exactly, and it must give the same
/// point as `u`: not because the device reduces anything, but because `u + r` and `u` are the
/// same multiple of a point of order `r`.
///
/// **Above that the answer is wrong by construction and the writes still have to stay inside
/// their buffers.** All ones is the worst case: every window is at its maximum and bit 254 is
/// set, so the top borrow is exactly the one the proof excludes. The magnitude is bounded by
/// `2^(c-1)` whatever the input, so the row index is in range however wrong the value is, and
/// this asserts that construction rather than trusting it.
#[test]
fn scalars_that_are_not_canonical_field_elements_stay_inside_their_buffers() {
    let _one = exclusive();
    let mut rng = test_rng();
    let n = 300usize;
    let bases = rand_g1(n, &mut rng);
    let words = g1_words(&bases);

    // `u + r` with `u` below 2^251, which is comfortably below `2^254 - r` (about 2^252.0),
    // so every sum has bit 254 clear and is a value the recoding covers exactly. Built by
    // masking a random field element's top limb rather than by rejection sampling, so the
    // yield is 300 of 300 and the test never quietly shrinks.
    let r_limbs: [u64; 4] = Fr::MODULUS.0;
    let mut raw: Vec<u32> = Vec::with_capacity(n * 8);
    let mut canonical: Vec<Fr> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut u = Fr::rand(&mut rng).into_bigint();
        u.0[3] &= (1u64 << 59) - 1;
        let small = Fr::from_bigint(u).expect("below 2^251 is below r");
        if small.is_zero() || small == Fr::from(1u64) {
            continue;
        }
        let mut sum = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let t = u128::from(u.0[i]) + u128::from(r_limbs[i]) + carry;
            sum[i] = t as u64;
            carry = t >> 64;
        }
        assert_eq!(carry, 0);
        assert_eq!(sum[3] >> 62, 0, "u + r reached bit 254");
        canonical.push(small);
        for l in sum {
            raw.push(l as u32);
            raw.push((l >> 32) as u32);
        }
    }
    let m = canonical.len();
    assert!(m >= n - 2, "{} of {n} draws were 0 or 1", n - m);
    let kept_bases = &bases[..m];
    let want = naive_g1(kept_bases, &canonical);
    assert!(!want.is_zero(), "degenerate oracle");
    let kept_words = g1_words(kept_bases);

    let run = run_msm_words!(
        digits(),
        g1(),
        wgsl::G1,
        &kept_words,
        &raw,
        m as u32,
        m as u32,
        None,
        2,
        None
    );
    assert_eq!(
        run.result, want,
        "u + r and u are the same multiple of a point of order r, and the device disagreed \
         over {m} of them"
    );
    run.assert_no_overrun(wgsl::G1.point_bytes, 2, "u + r");

    // All ones: 256 bits set, so bit 255 is set and the recoding's top-carry proof does not
    // cover this input. The answer is meaningless and is deliberately not asserted. What is
    // asserted is that nothing wrote outside a buffer.
    let ones = vec![0xffff_ffffu32; n * 8];
    let run = run_msm_words!(
        digits(),
        g1(),
        wgsl::G1,
        &words,
        &ones,
        n as u32,
        n as u32,
        None,
        4,
        None
    );
    run.assert_no_overrun(wgsl::G1.point_bytes, 4, "all-ones scalars");
    println!(
        "{m} non-canonical scalars below 2^254 gave the canonical answer, and {n} all-ones \
         scalars wrote nothing outside their buffers"
    );
}

// ---------------------------------------------------------------------------
// 5. The dispatch chunking, which no artifact reaches
// ---------------------------------------------------------------------------

/// The whole MSM with `workgroups_per_dispatch` forced to 2, so `clear`, `segmented` and
/// `merge` are each encoded as many dispatches over `P.lo`.
///
/// `MsmDigits::with_shape` and `MsmPoints::with_shape` are both public and both say, in the
/// same words, that one reason is "the multi-dispatch path, which no artifact reaches ... a
/// code path no test can reach is a code path that ships untested". **Every existing caller
/// of either passes 65535**, so the path was reachable and still untested. `tests/gather.rs`
/// does force the chunking on stage 0 (three workgroups per dispatch, with a short last
/// chunk); nothing did it for stages 5 to 9.
///
/// The chunking is where a `lo` that is not carried, a last chunk that is rounded up rather
/// than clamped, or a parameter ring slot count that disagrees with the dispatch count all
/// live, and none of the three is visible in a single-dispatch encode. `n = 2000` at the
/// default window width makes all three kernels take several chunks; the answer must not
/// move.
#[test]
fn the_msm_is_the_same_point_when_every_kernel_is_split_across_dispatches() {
    let _one = exclusive();
    let mut rng = test_rng();
    let n = 2000usize;
    let bases = rand_g1(n, &mut rng);
    let words = g1_words(&bases);
    let scalars = general_scalars(n, &mut rng);
    let want = naive_g1(&bases, &scalars);
    assert!(!want.is_zero(), "degenerate oracle");
    let (sw, general) = pack_scalars(&scalars);

    let whole = run_msm_words!(
        digits(),
        g1(),
        wgsl::G1,
        &words,
        &sw,
        general,
        n as u32,
        None,
        2,
        None
    );
    assert_eq!(
        whole.result, want,
        "the single-dispatch encode is already wrong"
    );

    // Two workgroups per dispatch: at the shipped sizes that is 512 rows a chunk for clear
    // and merge and 256 threads a chunk for the segmented pass.
    let d = MsmDigits::with_shape(
        floor(),
        g16_wgpu::gen::msm::Workgroups::default(),
        g16_wgpu::gen::msm::LimbPick::default(),
        2,
    )
    .expect("chunked digit pipelines");
    let p = MsmPointsG1::with_shape(floor(), wgsl::G1.wg, 2).expect("chunked point pipelines");
    let split = run_msm_words!(
        &d,
        &p,
        wgsl::G1,
        &words,
        &sw,
        general,
        n as u32,
        None,
        2,
        None
    );

    assert_eq!(
        split.result, want,
        "splitting clear, segmented and merge across dispatches changed the answer"
    );
    split.assert_no_overrun(wgsl::G1.point_bytes, 2, "chunked");
    // And it really was split, which is the premise: without this the test passes if
    // `with_shape` quietly ignored the cap.
    let dplan = DigitPlan::new(n as u32, 0, Some(general)).expect("plan");
    let pplan = p.plan_points(&dplan, 0).expect("point plan");
    let dispatches = p.slots(&dplan, &pplan);
    assert!(
        dispatches > 5,
        "the chunked run encoded {dispatches} point dispatches, so nothing was chunked and \
         this test compares two identical encodes"
    );
    println!("{dispatches} point dispatches at 2 workgroups each, same point as one dispatch");
}

// ---------------------------------------------------------------------------
// 6. A pooled spill array that still holds the last proof's tags
// ---------------------------------------------------------------------------

/// `msm_segmented_*` tags both of its spill slots `NO_ROW` **before** the early return that
/// skips an empty slice, and its comment says why: "the merge reads every slot in a bucket's
/// slice range and a pooled spill_rows buffer holds the last proof's tags". Nothing tested
/// it. Deleting both lines leaves the whole existing suite green, `tests/msm_g1.rs`,
/// `tests/msm_g2.rs`, `tests/proof.rs` and the five tests above included, because every one
/// of them pre-fills the array with a sentinel that is neither a row number nor `NO_ROW`, so
/// a stale slot is ignored by the merge whether it was re-tagged or not.
///
/// The slot that matters is the **tail**. A slice whose entries are all one run writes its
/// head slot and leaves the tail untouched after the pre-tagging, and that is the common case
/// inside a fat bucket, which is the case the segmented pass was built for. If the tail still
/// held a row from the previous proof, the merge would fold that slice's leftover point into
/// whichever bucket the tag named.
///
/// So this reproduces the pool: run once, take the tags the run produced, and run the same
/// MSM again with **the head tag copied into the tail slot** of every slice. That is not an
/// arbitrary poison. It is the one value guaranteed to name a bucket whose own slice range
/// contains this slice, so a merge that trusts it will read it. `SPILL_PTS` is left at the
/// sentinel, whose `zz` is nonzero and therefore a live point to `pt_add_g1`.
///
/// The answer must not move.
#[test]
fn a_pooled_spill_array_holding_the_previous_tags_changes_nothing() {
    let _one = exclusive();
    let mut rng = test_rng();
    let sl = wgsl::G1.slice_len as usize;
    // A fat bucket, so most slices are a single run and most tail slots go untouched.
    let n = 4 * sl + 17;
    let bases = rand_g1(n, &mut rng);
    let k = general_scalars(1, &mut rng)[0];
    let scalars = vec![k; n];
    let want = naive_g1(&bases, &scalars);
    assert!(!want.is_zero(), "degenerate oracle");
    let words = g1_words(&bases);
    let (sw, general) = pack_scalars(&scalars);

    let first = run_msm_words!(
        digits(),
        g1(),
        wgsl::G1,
        &words,
        &sw,
        general,
        n as u32,
        None,
        2,
        None
    );
    assert_eq!(first.result, want, "the unpoisoned run is already wrong");

    // Head tag into the tail slot, everywhere. Slices whose head is NO_ROW were empty and
    // stay empty, which is the other half of what the pre-tagging defends.
    let slots = first.spill_slots as usize;
    let mut poison: Vec<u32> = first.spill_rows[..slots].to_vec();
    let mut carried = 0usize;
    for slice in 0..slots / 2 {
        if poison[2 * slice] != wgsl::NO_ROW {
            poison[2 * slice + 1] = poison[2 * slice];
            carried += 1;
        }
    }
    assert!(
        carried > 4,
        "only {carried} slices had a head tag to carry, so this poison names no bucket"
    );

    let second = run_msm_words!(
        digits(),
        g1(),
        wgsl::G1,
        &words,
        &sw,
        general,
        n as u32,
        None,
        2,
        Some(&poison[..])
    );
    assert_eq!(
        second.result, want,
        "a spill array pre-loaded with {carried} stale tail tags changed the answer, so \
         msm_segmented_g1 is not re-tagging the slots it does not write"
    );
    second.assert_no_overrun(wgsl::G1.point_bytes, 2, "poisoned spill tags");
    println!("{carried} stale tail tags folded into nothing");
}

/// Bases at 2^16 without paying for 65,536 scalar multiplications: an arithmetic progression.
/// Structure in the bases cannot help or hurt an MSM, which only ever adds them.
fn walk_g1(n: usize, rng: &mut impl Rng) -> Vec<G1Affine> {
    let step = G1Projective::rand(rng);
    let mut cur = G1Projective::rand(rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    G1Projective::normalize_batch(&proj)
}

// ---------------------------------------------------------------------------
// 7. The window width the cost model picks
// ---------------------------------------------------------------------------

/// `crate::msm::window_size` against a measured sweep, at the two `m` where its most
/// suspicious constant changes the answer.
///
/// # Why these two sizes and not a grid
///
/// The model's `SLICE_LEN` is 64 and its comment says "Duplicated from Metal's `SLICE_LEN`
/// because the kernel that uses it does not exist yet; U9 owns the real one". U9 landed, swept
/// it on both curves, and shipped **128** (`gen::points::Curve::slice_len`). The model was not
/// updated, so it now disagrees with the kernel it is modelling, and the comment reads as an
/// invitation to go and tidy that up.
///
/// Doing so would make the prover slower. `SLICE_LEN` appears only in the merge term, so
/// doubling it halves that term, and the pick changes at exactly two of the sizes this repo
/// proves: `m = 18002` goes 8 to 10 and `m = 65536` goes 13 to 10. Measured, medians of
/// three, whole G1 MSM including the upload, M2 Max, release, milliseconds:
///
/// ```text
/// m         c=8    c=9   c=10   c=11   c=12   c=13   c=14   model  best
/// 18002    41.5   53.4   45.6   69.9   64.1   64.6  100.6       8     8
/// 65536    76.0  117.1   82.7  143.0  104.7   75.1  140.5      13    13
/// ```
///
/// The stale constant picks the measured winner at both, and the value it would be "corrected"
/// to picks a width that is 10.0% worse at `m = 18002` and 10.1% worse at `m = 65536`. So the
/// 64 stays, the comment in `crate::msm::window_size` now says why, and this test is the thing
/// that will argue back the next time someone changes it.
///
/// The assertion is 10% and not "is the best", because at `m = 65536` the second-placed width
/// is 1.2% away and that is inside the noise of a three-sample median. 10% still fails on
/// either of the two substitutions above and on anything larger.
#[test]
fn the_window_width_the_cost_model_picks_is_within_ten_percent_of_the_measured_best() {
    use std::time::Instant;
    let _one = exclusive();
    let _gpu = gpulock::exclusive_gpu();
    let mut rng = test_rng();
    for m in [18002usize, 65536usize] {
        let bases = walk_g1(m, &mut rng);
        let words = g1_words(&bases);
        let scalars = general_scalars(m, &mut rng);
        let (sw, general) = pack_scalars(&scalars);
        let picked = g16_wgpu::window_size(m);
        let mut line = String::new();
        let mut best = (f64::MAX, 0u32);
        let mut at_pick = f64::MAX;
        for c in 8u32..=14 {
            let mut ts = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                let run = run_msm_words!(
                    digits(),
                    g1(),
                    wgsl::G1,
                    &words,
                    &sw,
                    general,
                    m as u32,
                    Some(c),
                    0,
                    None
                );
                let _ = std::hint::black_box(run.result);
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ts[1];
            line.push_str(&format!("  c={c} {med:.1}"));
            if med < best.0 {
                best = (med, c);
            }
            if c == picked {
                at_pick = med;
            }
        }
        println!(
            "m={m} model picks c={picked} ({at_pick:.1} ms), best c={} ({:.1} ms) |{line}",
            best.1, best.0
        );
        assert!(
            at_pick <= 1.10 * best.0,
            "m = {m}: window_size picked c = {picked} at {at_pick:.1} ms, {:.1}% over the \
             measured best c = {} at {:.1} ms",
            100.0 * (at_pick / best.0 - 1.0),
            best.1,
            best.0
        );
    }
}
