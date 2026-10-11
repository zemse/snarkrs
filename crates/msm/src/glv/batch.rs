//! Many independent `[k_i] P_i` at once, in affine coordinates, with each step's
//! inversions shared across the batch (Montgomery's trick).
//!
//! The ladder is [`super::mul`]'s, run in lockstep: every point doubles at the same step and
//! adds its own table entry where its digit is nonzero. In affine form a doubling is
//! `3M + 2S` and an addition `2M + S` plus one share of a batched inversion, which is three
//! multiplications, against XYZZ's `6M + 3S` and `12M + 2S`. Across the 127 doublings and
//! about 49 additions of a GLV multiplication that is roughly two thirds of the work.
//!
//! Affine addition has no answer when both inputs share an `x`: the doubling and the
//! cancellation of the projective formulas. In the ladder that needs a partial sum equal to
//! plus or minus a table entry, which a short GLV half rarely reaches but an adversarial
//! scalar can. Such a point leaves the batch and is redone with [`super::mul`], so the
//! result is exact for every input.

use ark_ec::short_weierstrass::Affine;
use snarkrs_field::raw::RawField;
use snarkrs_field::Fr;

use super::{decompose, wnaf, GlvField, DIGITS, W};
use crate::xyzz::{RawCurve, Xyzz};

const TABLE: usize = 1 << (W - 2);

#[derive(Clone, Copy)]
struct Pt<F> {
    x: F,
    y: F,
}

/// Per-point ladder state.
struct Lane<F> {
    d1: [i8; DIGITS],
    d2: [i8; DIGITS],
    neg1: bool,
    neg2: bool,
    t1: [Pt<F>; TABLE],
    /// Out of the batch: zero input, or an exceptional addition; see the module doc.
    out: bool,
}

/// `out[i] = [k_i] P_i` for every `i`, affine in and out.
pub fn mul_batch<P: RawCurve>(points: &[Affine<P>], scalars: &[Fr], out: &mut [Affine<P>])
where
    P::RF: GlvField,
{
    assert_eq!(points.len(), scalars.len(), "one scalar per point");
    assert_eq!(points.len(), out.len(), "one output per point");
    let zero = Pt {
        x: P::RF::ZERO,
        y: P::RF::ZERO,
    };
    let mut lanes: Vec<Lane<P::RF>> = Vec::with_capacity(points.len());
    let mut steps = 0;
    for (p, k) in points.iter().zip(scalars) {
        let mut lane = Lane {
            d1: [0; DIGITS],
            d2: [0; DIGITS],
            neg1: false,
            neg2: false,
            t1: [zero; TABLE],
            out: true,
        };
        if !p.infinity {
            let (k1, k2) = decompose(k);
            let n =
                wnaf(k1.unsigned_abs(), &mut lane.d1).max(wnaf(k2.unsigned_abs(), &mut lane.d2));
            if n > 0 {
                let (x, y) = P::raw_xy(p);
                lane.t1[0] = Pt { x, y };
                lane.neg1 = k1 < 0;
                lane.neg2 = k2 < 0;
                lane.out = false;
                steps = steps.max(n);
            }
        }
        lanes.push(lane);
    }

    let mut scratch = Scratch {
        den: Vec::new(),
        pre: Vec::new(),
        who: Vec::new(),
    };
    // Odd multiples: 2P, then 3P = P + 2P and so on, one batched addition per entry.
    let mut two: Vec<Pt<P::RF>> = lanes.iter().map(|l| l.t1[0]).collect();
    let live: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].out).collect();
    {
        let mut ok = vec![true; lanes.len()];
        batch_dbl::<P>(&mut two, &live, &mut ok, &mut scratch);
        for j in 1..TABLE {
            let mut next: Vec<Pt<P::RF>> = lanes.iter().map(|l| l.t1[j - 1]).collect();
            let adds: Vec<(usize, Pt<P::RF>)> = live.iter().map(|&i| (i, two[i])).collect();
            let mut nz = vec![false; lanes.len()];
            batch_add::<P>(&mut next, &mut nz, &adds, &mut ok, &mut scratch);
            for &i in &live {
                lanes[i].t1[j] = next[i];
            }
        }
        // Odd multiples of a point of prime order never collide; this is belt and braces.
        for &i in &live {
            if !ok[i] {
                lanes[i].out = true;
            }
        }
    }

    let mut acc: Vec<Pt<P::RF>> = vec![zero; lanes.len()];
    let mut acc_zero = vec![true; lanes.len()];
    let mut ok = vec![true; lanes.len()];
    let mut idx = Vec::with_capacity(lanes.len());
    let mut adds = Vec::with_capacity(lanes.len());
    for s in (0..steps).rev() {
        idx.clear();
        idx.extend((0..lanes.len()).filter(|&i| !lanes[i].out && ok[i] && !acc_zero[i]));
        batch_dbl::<P>(&mut acc, &idx, &mut ok, &mut scratch);
        for half in 0..2 {
            adds.clear();
            for (i, l) in lanes.iter().enumerate() {
                if l.out || !ok[i] {
                    continue;
                }
                let (d, neg) = if half == 0 {
                    (l.d1[s], l.neg1)
                } else {
                    (l.d2[s], l.neg2)
                };
                if d == 0 {
                    continue;
                }
                let mut e = l.t1[(d.unsigned_abs() >> 1) as usize];
                if half == 1 {
                    e.x = P::RF::endo_x(e.x);
                }
                if (d < 0) != neg {
                    e.y = e.y.neg();
                }
                adds.push((i, e));
            }
            batch_add::<P>(&mut acc, &mut acc_zero, &adds, &mut ok, &mut scratch);
        }
    }

    for (i, l) in lanes.iter().enumerate() {
        out[i] = if l.out && (points[i].infinity || scalars[i] == Fr::from(0u64)) {
            Affine::identity()
        } else if l.out || !ok[i] {
            to_affine::<P>(&super::mul_affine(&points[i], &scalars[i]))
        } else if acc_zero[i] {
            Affine::identity()
        } else {
            Affine::new_unchecked(P::field_back(acc[i].x), P::field_back(acc[i].y))
        };
    }
}

fn to_affine<P: RawCurve>(p: &Xyzz<P::RF>) -> Affine<P> {
    if p.is_zero() {
        return Affine::identity();
    }
    let t = P::raw_inv(p.zz.mul(p.zzz));
    let x = p.x.mul(p.zzz).mul(t);
    let y = p.y.mul(p.zz).mul(t);
    Affine::new_unchecked(P::field_back(x), P::field_back(y))
}

struct Scratch<F> {
    den: Vec<F>,
    pre: Vec<F>,
    who: Vec<usize>,
}

/// `a[i] = 2 a[i]` for `i` in `idx`. A point with `y = 0` has order two, which neither
/// BN254 group contains; one is marked not `ok` rather than divided by.
fn batch_dbl<P: RawCurve>(
    a: &mut [Pt<P::RF>],
    idx: &[usize],
    ok: &mut [bool],
    s: &mut Scratch<P::RF>,
) {
    s.den.clear();
    s.who.clear();
    for &i in idx {
        let d = a[i].y.double();
        if d.is_zero() {
            ok[i] = false;
            continue;
        }
        s.den.push(d);
        s.who.push(i);
    }
    invert_all::<P>(s);
    for (k, &i) in s.who.iter().enumerate() {
        let p = a[i];
        let xx = p.x.sqr();
        let l = xx.double().add(xx).mul(s.den[k]);
        let x3 = l.sqr().sub(p.x.double());
        let y3 = l.mul(p.x.sub(x3)).sub(p.y);
        a[i] = Pt { x: x3, y: y3 };
    }
}

/// `a[i] += q` for each `(i, q)` in `adds` (at most one per `i`). Into the identity is a
/// copy; a shared `x` is the exceptional case and marks `i` not `ok`.
fn batch_add<P: RawCurve>(
    a: &mut [Pt<P::RF>],
    zero: &mut [bool],
    adds: &[(usize, Pt<P::RF>)],
    ok: &mut [bool],
    s: &mut Scratch<P::RF>,
) {
    s.den.clear();
    s.who.clear();
    for (k, &(i, q)) in adds.iter().enumerate() {
        if zero[i] {
            a[i] = q;
            zero[i] = false;
            continue;
        }
        let d = q.x.sub(a[i].x);
        if d.is_zero() {
            ok[i] = false;
            continue;
        }
        s.den.push(d);
        s.who.push(k);
    }
    invert_all::<P>(s);
    for (j, &k) in s.who.iter().enumerate() {
        let (i, q) = adds[k];
        let p = a[i];
        let l = q.y.sub(p.y).mul(s.den[j]);
        let x3 = l.sqr().sub(p.x).sub(q.x);
        let y3 = l.mul(p.x.sub(x3)).sub(p.y);
        a[i] = Pt { x: x3, y: y3 };
    }
}

/// Replace every `s.den[k]` by its inverse: one field inversion for the lot.
fn invert_all<P: RawCurve>(s: &mut Scratch<P::RF>) {
    let n = s.den.len();
    if n == 0 {
        return;
    }
    s.pre.clear();
    let mut acc = P::RF::ONE;
    for d in &s.den {
        s.pre.push(acc);
        acc = acc.mul(*d);
    }
    let mut inv = P::raw_inv(acc);
    for k in (0..n).rev() {
        let d = s.den[k];
        s.den[k] = inv.mul(s.pre[k]);
        inv = inv.mul(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
    use snarkrs_field::{g1, G1Projective};

    type F = snarkrs_field::raw::RawFq;

    /// The two exceptional additions, doubling and cancellation, are refused rather than
    /// divided by zero, and the rest of the batch is untouched.
    #[test]
    fn a_shared_x_leaves_the_batch() {
        let p = (G1Projective::generator() * Fr::from(5u64)).into_affine();
        let r = (G1Projective::generator() * Fr::from(7u64)).into_affine();
        let (x, y) = <g1::Config as RawCurve>::raw_xy(&p);
        let (rx, ry) = <g1::Config as RawCurve>::raw_xy(&r);
        let q = Pt { x, y };
        let other = Pt { x: rx, y: ry };
        let mut s = Scratch {
            den: Vec::new(),
            pre: Vec::new(),
            who: Vec::new(),
        };
        let mut a: Vec<Pt<F>> = vec![q, q, q];
        let mut zero = vec![false; 3];
        let mut ok = vec![true; 3];
        let neg = Pt { x, y: y.neg() };
        batch_add::<g1::Config>(
            &mut a,
            &mut zero,
            &[(0, q), (1, neg), (2, other)],
            &mut ok,
            &mut s,
        );
        assert_eq!(ok, vec![false, false, true]);
        let want = (p.into_group() + r).into_affine();
        assert_eq!(
            (
                <g1::Config as RawCurve>::field_back(a[2].x),
                <g1::Config as RawCurve>::field_back(a[2].y)
            ),
            (want.x, want.y)
        );
    }
}
