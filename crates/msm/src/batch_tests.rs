//! Deliberate collisions, checked against ark arithmetic and the scheduler's state.
//! Random MSMs do not pin exceptions at the ends of a shared inversion round.

use super::*;
use ark_ec::CurveGroup;

fn insert<P: RawCurve>(fill: &mut BatchFill<P>, b: usize, p: Affine<P>) {
    assert!(!p.infinity);
    let (x, y) = P::raw_xy(&p);
    fill.insert(b, x, y);
}

fn check_buckets<P: RawCurve>(fill: &BatchFill<P>, want: &[Projective<P>]) {
    assert!(fill.pending.is_empty());
    assert!(fill.retry.is_empty());
    assert!(fill.busy.iter().all(|b| !b));
    for (b, expected) in want.iter().enumerate() {
        let mut got = Projective::<P>::zero();
        if fill.occupied[b] {
            got += Affine::<P>::new_unchecked(
                P::field_back(fill.xy[b][0]),
                P::field_back(fill.xy[b][1]),
            );
        }
        if let Some(side) = &fill.side {
            got += to_projective::<P>(&side[b]);
        }
        assert_eq!(got, *expected, "bucket {b}");
    }
}

fn check_finish<P: RawCurve<ScalarField = Fr>>(fill: BatchFill<P>, want: &[Projective<P>]) {
    let expected = want
        .iter()
        .enumerate()
        .fold(Projective::<P>::zero(), |acc, (b, p)| {
            acc + *p * Fr::from((b + 1) as u64)
        });
    assert_eq!(fill.finish(), expected);
}

fn denominator_positions<P: RawCurve<ScalarField = Fr>>() {
    let p = P::GENERATOR;
    let q = (Projective::<P>::from(p) + p).into_affine();
    // Move a doubling or cancellation through every position, including both ends.
    // All other denominators are ordinary x differences.
    for exceptional in 0..17 {
        for cancel in [false, true] {
            let mut fill = BatchFill::<P>::new(17, MIN_BATCH);
            let mut want = vec![Projective::<P>::zero(); 17];
            for (b, expected) in want.iter_mut().enumerate() {
                insert(&mut fill, b, p);
                let rhs = if b != exceptional {
                    q
                } else if cancel {
                    -p
                } else {
                    p
                };
                insert(&mut fill, b, rhs);
                *expected = Projective::<P>::from(p) + rhs;
            }
            assert_eq!(fill.pending.len(), 17);
            fill.flush();
            assert!(fill.dens.iter().all(|d| !d.is_zero()));
            if cancel {
                assert!(fill.dens[exceptional] == P::RF::ONE);
                assert!(!fill.occupied[exceptional]);
            } else {
                assert!(fill.dens[exceptional] == P::raw_xy(&p).1.double());
            }
            check_buckets(&fill, &want);
            check_finish(fill, &want);
        }
    }

    // An entirely cancelling batch has product ONE, even at the automatic flush
    // boundary of either end of the batch range. Empty and singleton rounds must also
    // leave the scheduler reusable.
    for batch in [MIN_BATCH, BATCH] {
        for n in [0, 1, batch - 1, batch, batch + 1] {
            let mut fill = BatchFill::<P>::new(n, batch);
            for b in 0..n {
                insert(&mut fill, b, p);
                insert(&mut fill, b, -p);
            }
            fill.flush();
            assert!(fill.occupied.iter().all(|b| !b));
            assert!(fill.dens.iter().all(|d| *d == P::RF::ONE));
            let mut want = vec![Projective::<P>::zero(); n];
            check_buckets(&fill, &want);
            for (b, expected) in want.iter_mut().enumerate() {
                insert(&mut fill, b, q);
                *expected = q.into();
            }
            check_buckets(&fill, &want);
            check_finish(fill, &want);
        }
    }
}

#[test]
fn batch_denominator_positions_g1() {
    denominator_positions::<snarkrs_field::g1::Config>();
}

#[test]
fn batch_denominator_positions_g2() {
    denominator_positions::<snarkrs_field::g2::Config>();
}

fn retries_and_side<P: RawCurve<ScalarField = Fr>>() {
    for batch in [MIN_BATCH, BATCH] {
        retries_and_side_at::<P>(batch);
    }
}

fn retries_and_side_at<P: RawCurve<ScalarField = Fr>>(batch: usize) {
    let p = P::GENERATOR;
    let q = (Projective::<P>::from(p) + p).into_affine();
    let b = 7;
    for cancel_first in [false, true] {
        let mut fill = BatchFill::<P>::new(8, batch);
        let mut want = vec![Projective::<P>::zero(); 8];
        let rhs = if cancel_first { -p } else { q };
        for point in [p, rhs, p, -p, p, p] {
            insert(&mut fill, b, point);
            want[b] += point;
        }
        assert_eq!(fill.pending.len(), 1);
        assert_eq!(fill.retry.len(), 4);
        assert!(fill.side.is_none());
        fill.flush();
        assert_eq!(fill.occupied[b], !cancel_first);
        fill.reschedule();
        assert!(fill.retry.is_empty());
        assert_eq!(fill.pending.len(), 1);
        assert!(fill.busy[b]);
        // LIFO rescheduling sends p, -p, p to the side if the bucket stayed
        // occupied, and only p, -p if a cancellation freed the first retry.
        let side = fill.side.as_ref().expect("third conflict must reach side");
        let expected_side = if cancel_first {
            Projective::<P>::zero()
        } else {
            p.into()
        };
        assert_eq!(to_projective::<P>(&side[b]), expected_side);
        fill.flush();
        check_buckets(&fill, &want);
        check_finish(fill, &want);
    }

    // The retry trigger must bound a hot bucket without an explicit flush. The
    // side stays live when the affine half later cancels, and after it refills.
    for refill in [false, true] {
        let mut fill = BatchFill::<P>::new(8, batch);
        let mut want = vec![Projective::<P>::zero(); 8];
        for _ in 0..batch + 2 {
            insert(&mut fill, b, p);
            want[b] += p;
        }
        assert!(fill.retry.is_empty());
        assert_eq!(fill.pending.len(), 1);
        assert_eq!(
            to_projective::<P>(&fill.side.as_ref().expect("retry trigger")[b]),
            Projective::<P>::from(p) * Fr::from((batch - 1) as u64)
        );
        fill.flush();
        let affine =
            Affine::<P>::new_unchecked(P::field_back(fill.xy[b][0]), P::field_back(fill.xy[b][1]));
        insert(&mut fill, b, -affine);
        want[b] -= affine;
        fill.flush();
        assert!(!fill.occupied[b]);
        check_buckets(&fill, &want);
        if refill {
            insert(&mut fill, b, q);
            want[b] += q;
            check_buckets(&fill, &want);
        }
        check_finish(fill, &want);
    }

    let mut fill = BatchFill::<P>::new(8, batch);
    let mut want = vec![Projective::<P>::zero(); 8];
    for i in 0..6 * batch + 7 {
        let point = if i % 3 == 0 { -p } else { p };
        insert(&mut fill, b, point);
        want[b] += point;
        assert!(fill.pending.len() <= batch);
        assert!(fill.retry.len() < batch);
    }
    assert!(fill.side.is_some());
    assert!(!fill.retry.is_empty());
    check_finish(fill, &want);

    // The other trigger: `batch` independent pending entries with retries already
    // queued, over the bucket count that batch is sized for. A normal pending flush
    // must reschedule those retries exactly once.
    let n_buckets = 2 * batch;
    assert_eq!(batch_size(n_buckets), batch);
    let mut fill = BatchFill::<P>::new(n_buckets, batch);
    let mut want = vec![Projective::<P>::zero(); n_buckets];
    for point in [p, q, p, p] {
        insert(&mut fill, 0, point);
        want[0] += point;
    }
    for (b, expected) in want.iter_mut().enumerate().take(batch).skip(1) {
        insert(&mut fill, b, p);
        insert(&mut fill, b, q);
        *expected = Projective::<P>::from(p) + q;
    }
    assert!(fill.retry.is_empty());
    assert_eq!(fill.pending.len(), 1);
    assert!(fill.side.is_some());
    // finish must drain that last pending entry itself.
    check_finish(fill, &want);
}

#[test]
fn batch_retries_and_side_g1() {
    retries_and_side::<snarkrs_field::g1::Config>();
}

#[test]
fn batch_retries_and_side_g2() {
    retries_and_side::<snarkrs_field::g2::Config>();
}

#[test]
fn batch_size_stays_below_the_bucket_count() {
    for c in 3..=MAX_WINDOW {
        let n_buckets = 1usize << (c - 1);
        let batch = batch_size(n_buckets);
        assert!(batch < n_buckets, "c = {c}");
        assert_eq!(batch >= MIN_BATCH, c >= 9, "c = {c}");
        assert_eq!(batch == BATCH, c >= 12, "c = {c}");
    }
}

fn threshold_and_infinity<P: FastFill<ScalarField = Fr>>() {
    // The exact first prescanned lengths choosing c = 9 (the batch fill at MIN_BATCH)
    // and c = 12 (at the full BATCH), not just nearby powers of two.
    for (n, c) in [(3_328, 9), (30_720, 12)] {
        threshold_and_infinity_at::<P>(n, c);
    }
    let p = P::GENERATOR;

    // Ones can double, cancel, and refill an XYZZ accumulator across scan chunks.
    for n in [17, 4 * SCAN_CHUNK, 4 * SCAN_CHUNK + 1] {
        let bases: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 | 2 => p,
                1 => -p,
                _ => Affine::<P>::identity(),
            })
            .collect();
        let want = bases.iter().fold(Projective::<P>::zero(), |acc, p| acc + p);
        assert_eq!(
            pippenger(&bases, &vec![Fr::one(); n], 12, Work::Variable),
            want
        );
        assert_eq!(
            pippenger(&bases, &vec![Fr::one(); n], 12, Work::Constant),
            want
        );
    }
}

fn threshold_and_infinity_at<P: FastFill<ScalarField = Fr>>(n: usize, c: u32) {
    // Split one scalar to keep the mathematical MSM unchanged across the threshold.
    assert_eq!(window_size(n), c - 1);
    assert_eq!(window_size(n + 1), c);
    let p = P::GENERATOR;
    let mut bases = vec![p; n];
    let mut scalars = vec![-Fr::one(); n];
    let want = Projective::<P>::from(p) * -Fr::from(n as u64);
    // Identity as the right operand never enters BatchFill's raw coordinate API.
    // Test every scalar class, across parallel prescan chunks when n spans them.
    for (i, s) in [
        (0, Fr::zero()),
        (SCAN_CHUNK.min(n / 3), Fr::one()),
        ((2 * SCAN_CHUNK).min(2 * n / 3), Fr::from(2u64)),
        (n, -Fr::one()),
    ] {
        bases.insert(i, Affine::<P>::identity());
        scalars.insert(i, s);
    }
    assert_eq!(
        prescan(&bases, &scalars, bases.len(), Work::Constant)
            .idx
            .len(),
        n
    );
    assert_eq!(pippenger(&bases, &scalars, 1, Work::Constant), want);
    assert_eq!(pippenger(&bases, &scalars, 1, Work::Variable), want);
    let last = scalars.len() - 1;
    scalars[last] -= Fr::from(2u64);
    bases.push(p);
    scalars.push(Fr::from(2u64));
    assert_eq!(
        prescan(&bases, &scalars, bases.len(), Work::Constant)
            .idx
            .len(),
        n + 1
    );
    assert_eq!(pippenger(&bases, &scalars, 12, Work::Constant), want);
    assert_eq!(pippenger(&bases, &scalars, 12, Work::Variable), want);
}

/// `Work::Constant`'s promise, checked where it is decided: what reaches the bucket loop is
/// fixed by which bases are at infinity, not by the scalars. A witness that is all zeros and
/// ones and a dense one of the same length give the same general count, so the same window
/// width and the same allocations, and the same result as the variable-time path.
fn constant_work_ignores_the_scalar_values<P: FastFill<ScalarField = Fr>>() {
    let n = 3 * SCAN_CHUNK + 5;
    let p = P::GENERATOR;
    let bases: Vec<_> = (0..n)
        .map(|i| {
            if i % 7 == 3 {
                Affine::<P>::identity()
            } else {
                (Projective::<P>::from(p) * Fr::from(i as u64 + 1)).into()
            }
        })
        .collect();
    let finite = bases.iter().filter(|b| !b.infinity).count();
    let sparse: Vec<Fr> = (0..n).map(|i| Fr::from((i % 2) as u64)).collect();
    let dense: Vec<Fr> = (0..n).map(|i| -Fr::from(i as u64 * 7919 + 3)).collect();
    for s in [&sparse, &dense] {
        assert_eq!(prescan(&bases, s, n, Work::Constant).idx.len(), finite);
        assert_eq!(
            pippenger(&bases, s, 12, Work::Constant),
            pippenger(&bases, s, 12, Work::Variable)
        );
    }
    assert!(prescan(&bases, &sparse, n, Work::Variable).idx.is_empty());
}

#[test]
fn constant_work_ignores_the_scalar_values_g1() {
    constant_work_ignores_the_scalar_values::<snarkrs_field::g1::Config>();
}

#[test]
fn constant_work_ignores_the_scalar_values_g2() {
    constant_work_ignores_the_scalar_values::<snarkrs_field::g2::Config>();
}

#[test]
fn batch_threshold_and_infinity_g1() {
    threshold_and_infinity::<snarkrs_field::g1::Config>();
}

#[test]
fn batch_threshold_and_infinity_g2() {
    threshold_and_infinity::<snarkrs_field::g2::Config>();
}

/// Every reduce over the same buckets, against an independently weighted sum. A side
/// entry is the points its XYZZ bucket takes, in order.
fn check_reduce<P: RawCurve<ScalarField = Fr>>(
    buckets: &[Projective<P>],
    side: &[(usize, Vec<Projective<P>>)],
) {
    let n = buckets.len();
    let mut xy = vec![[P::RF::ZERO; 2]; n];
    let mut occupied = vec![false; n];
    let mut want = Projective::<P>::zero();
    for (j, b) in buckets.iter().enumerate() {
        want += *b * Fr::from((j + 1) as u64);
        let a = b.into_affine();
        if !a.infinity {
            let (x, y) = P::raw_xy(&a);
            xy[j] = [x, y];
            occupied[j] = true;
        }
    }
    let mut side_buckets = SideBuckets::new(n);
    for (j, points) in side {
        for p in points {
            let (x, y) = P::raw_xy(&p.into_affine());
            side_buckets.entry(*j).madd(x, y);
            want += *p * Fr::from((j + 1) as u64);
        }
    }
    let side = (!side.is_empty()).then_some(&side_buckets);
    let got = affine_running_sum::<P>(&mut xy.clone(), &mut occupied.clone(), side);
    assert_eq!(to_projective::<P>(&got), want, "n = {n}");

    if let Some(side) = side {
        fold_side::<P>(&mut xy, &mut occupied, side);
    }
    assert_eq!(
        to_projective::<P>(&xyzz_running_sum(&xy, &occupied)),
        want,
        "n = {n}, xyzz"
    );
    // From one bucket a segment to one segment longer than the range.
    let mut seg = 1;
    while seg <= 2 * n.max(1) {
        let got = segmented_running_sum_at::<P>(&xy, &occupied, seg);
        assert_eq!(to_projective::<P>(&got), want, "n = {n}, seg = {seg}");
        // The same lengths as chunks, each a task with its own segments.
        let got = chunked_running_sum_at::<P>(&xy, &occupied, seg);
        assert_eq!(to_projective::<P>(&got), want, "n = {n}, chunk = {seg}");
        seg *= 2;
    }
}

/// `i * G` for `i` in `-m..=m`, index `i + m`.
fn small_multiples<P: RawCurve<ScalarField = Fr>>(m: i64) -> Vec<Projective<P>> {
    let g = Projective::<P>::from(P::GENERATOR);
    (-m..=m)
        .map(|i| {
            let p = g * Fr::from(i.unsigned_abs());
            if i < 0 {
                -p
            } else {
                p
            }
        })
        .collect()
}

fn reduce_shapes<P: RawCurve<ScalarField = Fr>>() {
    use ark_std::rand::Rng;
    let g = Projective::<P>::from(P::GENERATOR);
    let zero = Projective::<P>::zero();
    // Nothing at all, on both sides of the XYZZ threshold.
    for n in [0, 1, 100, REDUCE_XYZZ, REDUCE_XYZZ + 1] {
        check_reduce::<P>(&vec![zero; n], &[]);
    }
    // One bucket, at every position of a range that is not a whole number of segments.
    for j in 0..77 {
        let mut buckets = vec![zero; 77];
        buckets[j] = g;
        check_reduce::<P>(&buckets, &[]);
    }
    // Every bucket the same point: each segment's second step doubles its running sum
    // and its total, and the segments' running sums are all equal in the recursion.
    for n in [1, 2, 64, REDUCE_XYZZ + 1, 257] {
        check_reduce::<P>(&vec![g; n], &[]);
    }
    // A point and its negative in turn: every running sum cancels every other step, and
    // whole segments sum to nothing.
    for n in [64, REDUCE_XYZZ + 1, 257] {
        let buckets: Vec<_> = (0..n).map(|j| if j % 2 == 0 { g } else { -g }).collect();
        check_reduce::<P>(&buckets, &[]);
    }
    // Small multiples of one point, so running sums and totals keep meeting as equal
    // points and as negatives, within a segment and across them. `gaps` thins the range.
    let small = small_multiples::<P>(3);
    let mut rng = ark_std::test_rng();
    for n in [33, 64, 100, REDUCE_XYZZ + 1, 256, 1000] {
        for gaps in [1, 2, 8] {
            for _ in 0..6 {
                let buckets: Vec<_> = (0..n)
                    .map(|_| {
                        if rng.gen_range(0..gaps) == 0 {
                            small[rng.gen_range(0..small.len())]
                        } else {
                            zero
                        }
                    })
                    .collect();
                check_reduce::<P>(&buckets, &[]);
            }
        }
    }
    // Unrelated points at a production bucket count.
    let mut cur = g * Fr::from(0x9e37_79b9_7f4a_7c15u64);
    let step = g * Fr::from(0xd1b5_4a32_d192_ed03u64);
    let buckets: Vec<_> = (0..4096)
        .map(|_| {
            cur += step;
            cur
        })
        .collect();
    check_reduce::<P>(&buckets, &[]);
}

#[test]
fn reduce_shapes_g1() {
    reduce_shapes::<snarkrs_field::g1::Config>();
}

#[test]
fn reduce_shapes_g2() {
    reduce_shapes::<snarkrs_field::g2::Config>();
}

fn reduce_folds_side<P: RawCurve<ScalarField = Fr>>() {
    use ark_std::rand::Rng;
    let small = small_multiples::<P>(9);
    let m = |i: i64| small[(i + 9) as usize];
    let mut rng = ark_std::test_rng();
    for n in [8, 80, 300] {
        let mut buckets: Vec<_> = (0..n).map(|_| m(rng.gen_range(-9..=9))).collect();
        // A side bucket of 3G, with ZZ != 1, over an affine bucket that differs, equals
        // it, negates it and is empty; one that cancelled to the identity but kept its
        // slot; and one whose own two points doubled.
        buckets[0] = m(5);
        buckets[1] = m(3);
        buckets[2] = m(-3);
        buckets[n / 2] = m(0);
        buckets[n - 2] = m(4);
        buckets[n - 1] = m(7);
        let side = vec![
            (0, vec![m(1), m(2)]),
            (1, vec![m(1), m(2)]),
            (2, vec![m(1), m(2)]),
            (n / 2, vec![m(1), m(2)]),
            (n - 2, vec![m(1), m(-1)]),
            (n - 1, vec![m(1), m(1)]),
        ];
        check_reduce::<P>(&buckets, &side);
        // The same side buckets over an otherwise empty range.
        let mut sparse = vec![Projective::<P>::zero(); n];
        sparse[2] = m(-3);
        check_reduce::<P>(&sparse, &side);
    }
}

#[test]
fn reduce_folds_side_g1() {
    reduce_folds_side::<snarkrs_field::g1::Config>();
}

#[test]
fn reduce_folds_side_g2() {
    reduce_folds_side::<snarkrs_field::g2::Config>();
}

#[test]
fn reduce_segment_is_a_power_of_two_below_the_range() {
    for n in REDUCE_XYZZ + 1..=1 << MAX_WINDOW {
        let seg = reduce_segment(n);
        assert!(seg.is_power_of_two() && seg >= 2 && seg < n, "n = {n}");
    }
    assert_eq!(reduce_segment(4096), 32);
}

/// Two fills of one window, each with its own points per bucket, merged while both still
/// hold unsettled rounds, against ark's sum per bucket.
fn check_merge<P: RawCurve<ScalarField = Fr>>(a: &[Vec<Affine<P>>], b: &[Vec<Affine<P>>]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    for batch in [MIN_BATCH, BATCH] {
        let mut fills = [BatchFill::<P>::new(n, batch), BatchFill::<P>::new(n, batch)];
        let mut want = vec![Projective::<P>::zero(); n];
        for (fill, points) in fills.iter_mut().zip([a, b]) {
            for (j, bucket) in points.iter().enumerate() {
                for &point in bucket {
                    insert(fill, j, point);
                    want[j] += point;
                }
            }
        }
        merge_fills(&mut fills);
        check_buckets(&fills[0], &want);
        let weighted = want
            .iter()
            .enumerate()
            .fold(Projective::<P>::zero(), |acc, (b, p)| {
                acc + *p * Fr::from((b + 1) as u64)
            });
        let got = chunked_running_sum::<P>(&fills[0].xy, &fills[0].occupied);
        assert_eq!(to_projective::<P>(&got), weighted);
        let [merged, _] = fills;
        check_finish(merged, &want);
    }
}

fn merge_shapes<P: RawCurve<ScalarField = Fr>>() {
    use ark_std::rand::Rng;
    let small = small_multiples::<P>(9);
    let m = |i: i64| small[(i + 9) as usize].into_affine();
    let none = || vec![Vec::new(); 9];
    // Nothing on either side, nothing on one, and a single point on one.
    check_merge::<P>(&none(), &none());
    let mut one = none();
    one[8] = vec![m(4)];
    check_merge::<P>(&one, &none());
    check_merge::<P>(&none(), &one);

    let mut a = none();
    let mut b = none();
    // The two fills' buckets meet as equal points and as opposite ones.
    (a[0], b[0]) = (vec![m(1)], vec![m(1)]);
    (a[1], b[1]) = (vec![m(1)], vec![m(-1)]);
    // Opposite only once a pending addition lands.
    (a[2], b[2]) = (vec![m(1), m(2)], vec![m(-3)]);
    // One fill arrives with retries, which settle into its side bucket, and the bucket
    // then differs from, equals and negates the other's.
    (a[3], b[3]) = (vec![m(2)], vec![m(1); 6]);
    (a[4], b[4]) = (vec![m(6)], vec![m(1); 6]);
    (a[5], b[5]) = (vec![m(-6)], vec![m(1); 6]);
    // Retries on both sides that sum to nothing.
    (a[6], b[6]) = (vec![m(1); 6], vec![m(-2); 3]);
    // A bucket that emptied inside one fill, over a live one and an empty one.
    (a[7], b[7]) = (vec![m(5)], vec![m(1), m(-1)]);
    b[8] = vec![m(2), m(1), m(-3)];
    check_merge::<P>(&a, &b);
    check_merge::<P>(&b, &a);

    // A hot bucket past the retry trigger, cancelled by the other fill's one point.
    let hot = MIN_BATCH + 2;
    let total = (Projective::<P>::from(m(1)) * Fr::from(hot as u64)).into_affine();
    let mut a = none();
    let mut b = none();
    (a[3], b[3]) = (vec![-total], vec![m(1); hot]);
    (a[4], b[4]) = (vec![total], vec![m(1); hot]);
    check_merge::<P>(&a, &b);
    check_merge::<P>(&b, &a);

    // More buckets than one task of the merge takes.
    let mut rng = ark_std::test_rng();
    let n = 2 * REDUCE_CHUNK + 37;
    let mut side = || -> Vec<Vec<Affine<P>>> {
        (0..n)
            .map(|_| {
                (0..rng.gen_range(0..3))
                    .map(|_| rng.gen_range(-9i64..=9))
                    .filter(|&i| i != 0)
                    .map(m)
                    .collect()
            })
            .collect()
    };
    let (a, b) = (side(), side());
    check_merge::<P>(&a, &b);
}

#[test]
fn merge_shapes_g1() {
    merge_shapes::<snarkrs_field::g1::Config>();
}

#[test]
fn merge_shapes_g2() {
    merge_shapes::<snarkrs_field::g2::Config>();
}

/// Every window of a small MSM through [`ganged_windows`], in pieces from one point up to
/// the whole range and on one thread and twelve, against the digits times the bases in
/// ark arithmetic. Repeated and negated bases under one scalar meet across the pieces,
/// in whichever fills the pool happened to give them.
fn ganged_pieces<P: RawCurve<ScalarField = Fr>>()
where
    P::BaseField: Send + Sync,
{
    let small = small_multiples::<P>(9);
    let n = 700;
    let mut bases: Vec<Affine<P>> = (0..n).map(|i| small[10 + i % 9].into_affine()).collect();
    let mut scalars: Vec<Fr> = (0..n).map(|i| -Fr::from(i as u64 * 7919 + 3)).collect();
    for i in 0..n / 2 {
        // One scalar over a base and its negative, alternating: every bucket they
        // touch sees doublings and cancellations, most of them across pieces.
        bases[i] = if i % 3 == 0 { -bases[0] } else { bases[0] };
        scalars[i] = scalars[0];
    }
    scalars[n - 1] = Fr::zero();
    for len in [0, 1, n] {
        let scan = Prescan::<P> {
            idx: (0..len as u32).collect(),
            bigints: scalars[..len].iter().map(|s| s.into_bigint()).collect(),
            ones_sum: Projective::zero(),
        };
        for c in [5u32, 9] {
            let want: Vec<_> = (0..RECODE_BITS.div_ceil(c as usize))
                .map(|w| {
                    (0..len).fold(Projective::<P>::zero(), |acc, k| {
                        let d = signed_digit(scan.bigints[k].as_ref(), w, c);
                        let term = Projective::<P>::from(bases[k]) * Fr::from(d.unsigned_abs());
                        if d < 0 {
                            acc - term
                        } else {
                            acc + term
                        }
                    })
                })
                .collect();
            for grain in [1, 2, 7, 64, n / 2, n] {
                for threads in [1, 12] {
                    for work in [Work::Variable, Work::Constant] {
                        assert_eq!(
                            ganged_windows(&bases, &scan, c, work, threads, grain),
                            want,
                            "{len} points at c = {c}, pieces of {grain}, {threads} threads"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ganged_pieces_g1() {
    ganged_pieces::<snarkrs_field::g1::Config>();
}

#[test]
fn ganged_pieces_g2() {
    ganged_pieces::<snarkrs_field::g2::Config>();
}

/// A window closes once, after its last piece, however the threads come and go: every
/// thread that claimed a piece hands in one fill, and one that came too late hands in
/// none.
#[test]
fn a_gang_closes_with_every_fill() {
    type P = snarkrs_field::g1::Config;
    let fill = || Some(BatchFill::<P>::new(16, 8));
    let gang = Gang::<P>::new();
    let pieces = 3;
    assert!(gang.enter() && gang.enter() && gang.enter());
    gang.next.store(2, Ordering::Relaxed);
    // A piece is still unclaimed, then two threads are still inside.
    assert!(gang.leave(fill(), pieces).is_none());
    gang.next.store(4, Ordering::Relaxed);
    assert!(gang.leave(None, pieces).is_none());
    assert!(gang.enter());
    assert!(gang.leave(fill(), pieces).is_none());
    assert_eq!(gang.leave(fill(), pieces).map(|fills| fills.len()), Some(3));
    assert!(!gang.enter());
    assert_eq!(gang.inside.load(Ordering::Relaxed), 0);

    for c in 3..=MAX_WINDOW {
        let n_buckets = 1usize << (c - 1);
        let grain = split_grain(n_buckets);
        assert!(grain >= MIN_GRAIN && grain >= n_buckets / GRAIN_BUCKET_RATIO);
    }
}

/// The XYZZ running sum against the segmented one at every segment length, per bucket,
/// on one thread and then on every thread of the pool at once, which is how the window
/// tasks run it.
fn time_reduce<P: RawCurve<ScalarField = Fr>>(name: &str) {
    use std::time::Instant;
    let g = Projective::<P>::from(P::GENERATOR);
    let step = g * Fr::from(0xd1b5_4a32_d192_ed03u64);
    let threads = rayon::current_num_threads();
    for log in [6u32, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let n = 1usize << log;
        let mut cur = g * Fr::from(0x9e37_79b9_7f4a_7c15u64);
        let proj: Vec<_> = (0..n)
            .map(|_| {
                cur += step;
                cur
            })
            .collect();
        let xy: Vec<[P::RF; 2]> = Projective::normalize_batch(&proj)
            .iter()
            .map(|a| {
                let (x, y) = P::raw_xy(a);
                [x, y]
            })
            .collect();
        let occupied = vec![true; n];
        let want = to_projective::<P>(&xyzz_running_sum(&xy, &occupied));
        let reps = (1 << 18) / n + 3;
        let time = |f: &(dyn Fn() -> Xyzz<P::RF> + Sync)| {
            let one = || {
                let mut best = f64::MAX;
                for _ in 0..reps {
                    let t = Instant::now();
                    let got = f();
                    best = best.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
                    assert_eq!(to_projective::<P>(&got), want);
                }
                best
            };
            let single = one();
            let mut all = vec![0.0; threads];
            std::thread::scope(|s| {
                for slot in all.iter_mut() {
                    s.spawn(|| *slot = one());
                }
            });
            (single, all.iter().sum::<f64>() / threads as f64)
        };
        let (s, a) = time(&|| xyzz_running_sum(&xy, &occupied));
        println!("{name} 2^{log:2} xyzz           {s:7.1} ns a bucket alone, {a:7.1} on {threads} threads");
        let mut seg = 2;
        while seg <= n / 4 {
            let (s, a) = time(&|| segmented_running_sum_at::<P>(&xy, &occupied, seg));
            println!(
                "{name} 2^{log:2} seg {seg:5}{}     {s:7.1} ns a bucket alone, {a:7.1} on {threads} threads",
                if seg == reduce_segment(n) { "*" } else { " " },
            );
            seg *= 2;
        }
    }
}

#[test]
#[ignore = "benchmark, meaningless in a debug build"]
fn time_reduce_g1() {
    time_reduce::<snarkrs_field::g1::Config>("G1");
}

#[test]
#[ignore = "benchmark, meaningless in a debug build"]
fn time_reduce_g2() {
    time_reduce::<snarkrs_field::g2::Config>("G2");
}
