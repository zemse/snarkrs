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
//! # The second gap, which is a shape and not an oracle
//!
//! `msm_merge_*` folds a bucket's spills over the slices `[k_lo, k_hi]` its run reaches. The
//! historical bug (`k_hi = k_lo`, shipped in the G2 merge) drops every slice past the first,
//! and the G1 suite catches it at n = 129 because 1,000 random entries spread over 128-entry
//! slices leave a few buckets straddling **one** boundary. Nothing in either suite makes a
//! single bucket's run cross **two**, because a random draw at those sizes never puts 129
//! entries in one bucket. So `k_hi = k_lo + 1u`, a plausible off-by-one in exactly the line
//! that was already wrong once, passes the whole existing suite.
//!
//! The witness distribution the segmented pass was built for is the one that produces those
//! runs: `gen::points` quotes a busiest bucket of 11,758 entries against a mean of 17 on
//! `js_2x2_d32`, from repeated witness values. Repeating one scalar reproduces it exactly,
//! because equal scalars have equal digits and therefore share a bucket in **every** window.
//! [`one_bucket_holding_a_run_longer_than_several_slices_is_merged_whole`] does that at
//! `slice_len - 1` through `5 * slice_len + 7`, and it is the only test in this crate where
//! `k_hi - k_lo` exceeds 1.
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

use common::{
    fill, general_count, general_scalars, read_bytes, read_words, storage_words, SENTINEL,
};

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
        let b = floor();
        let d = digits();
        let p = $points;
        let n: u32 = $n;
        let slack: u32 = $slack;

        let general = general_count(&$scalars[..n as usize]);
        let dplan = match $c {
            Some(c) => DigitPlan::with_c(n, 0, Some(general), c),
            None => DigitPlan::new(n, 0, Some(general)),
        }
        .expect("digit plan");
        let pplan = p.plan_points(&dplan, 0).expect("point plan");

        let (sw, _) = pack_scalars($scalars);
        let scalars_buf = storage_words(b, "verify scalars", &sw);
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

        // How many slices the fattest bucket of window 0 spans, which is `k_hi - k_lo + 1`
        // in `msm_merge_*`. Recomputed here from the counts and cursors the kernels wrote,
        // not from the plan, so it reports what the device actually built.
        let counts = read_words(b, &sort.counts);
        let cursor = read_words(b, &sort.cursor);
        let mut widest = 0u32;
        for row in 0..dplan.n_buckets() as usize {
            let cnt = counts[row];
            if cnt == 0 || cnt == SENTINEL {
                continue;
            }
            let end = cursor[row];
            let start = end - cnt;
            let span = (end - 1) / pplan.slice_len() - start / pplan.slice_len() + 1;
            widest = widest.max(span);
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
// 2. The gap: one bucket whose run crosses more than one slice boundary
// ---------------------------------------------------------------------------

/// `m` copies of one scalar over `m` distinct bases, so every window has exactly one occupied
/// bucket holding all `m` entries.
///
/// This is the only shape in this crate where `msm_merge_*` iterates its slice loop more than
/// twice. Each slice of the run sees no bucket change, so every one of them spills its whole
/// partial to its head slot and `BUCKETS[row]` is left at the identity by `msm_clear_*`; the
/// answer is therefore **entirely** the merge's fold, and a merge that stops early returns a
/// proper prefix of it. At `m = 5 * slice_len + 7` that is 6 slices, so `k_hi = k_lo` returns
/// about a sixth of the right point and `k_hi = k_lo + 1u` about a third.
///
/// It is also the distribution the segmented pass exists for: `gen::points` cites 11,758
/// entries in one bucket against a mean of 17 on `js_2x2_d32`, which comes from repeated
/// witness values, and repeated values have identical digits in every window.
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
        assert_eq!(
            run.widest_run_slices, want_slices,
            "m = {m}: the fattest bucket spans {} slices, not the {want_slices} this test \
             needs to exercise the merge's slice loop",
            run.widest_run_slices
        );
        assert_eq!(
            run.result, want,
            "m = {m} copies of one scalar: the device MSM disagrees with the naive sum, and \
             this bucket's run spans {want_slices} slices"
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
        assert!(
            run.widest_run_slices as usize >= least,
            "fat = {fat}, sparse = {sparse}: the fattest bucket spans {} slices, fewer than \
             the {least} a run of {fat} must touch",
            run.widest_run_slices
        );
        assert_eq!(
            run.result, want,
            "fat = {fat} repeated scalars beside {sparse} random ones: device MSM disagrees"
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
    assert_eq!(run.widest_run_slices, m.div_ceil(sl));
    assert_eq!(
        run.result, want,
        "m = {m} copies of one scalar over G2: the device MSM disagrees with the naive sum"
    );
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
