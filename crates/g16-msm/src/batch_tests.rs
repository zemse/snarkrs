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
            got += Affine::<P>::new_unchecked(P::field_back(fill.bx[b]), P::field_back(fill.by[b]));
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
            let mut fill = BatchFill::<P>::new(17);
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
    // boundary. Empty and singleton rounds must also leave the scheduler reusable.
    for n in [0, 1, BATCH - 1, BATCH, BATCH + 1] {
        let mut fill = BatchFill::<P>::new(n);
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

#[test]
fn batch_denominator_positions_g1() {
    denominator_positions::<g16_field::g1::Config>();
}

#[test]
fn batch_denominator_positions_g2() {
    denominator_positions::<g16_field::g2::Config>();
}

fn retries_and_side<P: RawCurve<ScalarField = Fr>>() {
    let p = P::GENERATOR;
    let q = (Projective::<P>::from(p) + p).into_affine();
    let b = 7;
    for cancel_first in [false, true] {
        let mut fill = BatchFill::<P>::new(8);
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
        let mut fill = BatchFill::<P>::new(8);
        let mut want = vec![Projective::<P>::zero(); 8];
        for _ in 0..BATCH + 2 {
            insert(&mut fill, b, p);
            want[b] += p;
        }
        assert!(fill.retry.is_empty());
        assert_eq!(fill.pending.len(), 1);
        assert_eq!(
            to_projective::<P>(&fill.side.as_ref().expect("retry trigger")[b]),
            Projective::<P>::from(p) * Fr::from((BATCH - 1) as u64)
        );
        fill.flush();
        let affine =
            Affine::<P>::new_unchecked(P::field_back(fill.bx[b]), P::field_back(fill.by[b]));
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

    let mut fill = BatchFill::<P>::new(8);
    let mut want = vec![Projective::<P>::zero(); 8];
    for i in 0..6 * BATCH + 7 {
        let point = if i % 3 == 0 { -p } else { p };
        insert(&mut fill, b, point);
        want[b] += point;
        assert!(fill.pending.len() <= BATCH);
        assert!(fill.retry.len() < BATCH);
    }
    assert!(fill.side.is_some());
    assert!(!fill.retry.is_empty());
    check_finish(fill, &want);

    // The other trigger: BATCH independent pending entries with retries already
    // queued. A normal pending flush must reschedule those retries exactly once.
    let mut fill = BatchFill::<P>::new(BATCH_MIN_BUCKETS);
    let mut want = vec![Projective::<P>::zero(); BATCH_MIN_BUCKETS];
    for point in [p, q, p, p] {
        insert(&mut fill, 0, point);
        want[0] += point;
    }
    for (b, expected) in want.iter_mut().enumerate().take(BATCH).skip(1) {
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
    retries_and_side::<g16_field::g1::Config>();
}

#[test]
fn batch_retries_and_side_g2() {
    retries_and_side::<g16_field::g2::Config>();
}

fn threshold_and_infinity<P: RawCurve<ScalarField = Fr>>() {
    assert_eq!(BATCH_MIN_BUCKETS, 2048);
    // The exact first prescanned length choosing c = 12, not just a nearby power
    // of two. Split one scalar to keep the mathematical MSM unchanged across it.
    let n = 30_720;
    assert_eq!(window_size(n), 11);
    assert_eq!(window_size(n + 1), 12);
    let p = P::GENERATOR;
    let mut bases = vec![p; n];
    let mut scalars = vec![-Fr::one(); n];
    let want = Projective::<P>::from(p) * -Fr::from(n as u64);
    // Identity as the right operand never enters BatchFill's raw coordinate API.
    // Test every scalar class across parallel prescan chunks.
    for (i, s) in [
        (0, Fr::zero()),
        (SCAN_CHUNK, Fr::one()),
        (2 * SCAN_CHUNK, Fr::from(2u64)),
        (n, -Fr::one()),
    ] {
        bases.insert(i, Affine::<P>::identity());
        scalars.insert(i, s);
    }
    assert_eq!(prescan(&bases, &scalars, bases.len()).idx.len(), n);
    assert_eq!(pippenger(&bases, &scalars, 1), want);
    let last = scalars.len() - 1;
    scalars[last] -= Fr::from(2u64);
    bases.push(p);
    scalars.push(Fr::from(2u64));
    assert_eq!(prescan(&bases, &scalars, bases.len()).idx.len(), n + 1);
    assert_eq!(pippenger(&bases, &scalars, 12), want);

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
        assert_eq!(pippenger(&bases, &vec![Fr::one(); n], 12), want);
    }
}

#[test]
fn batch_threshold_and_infinity_g1() {
    threshold_and_infinity::<g16_field::g1::Config>();
}

#[test]
fn batch_threshold_and_infinity_g2() {
    threshold_and_infinity::<g16_field::g2::Config>();
}
