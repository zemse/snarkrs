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

use super::{wnaf, GlvField, DIGITS, W};
use crate::xyzz::{RawCurve, Xyzz};

const TABLE: usize = 1 << (W - 2);

#[derive(Clone, Copy)]
struct Pt<F> {
    x: F,
    y: F,
}

/// `out[i] = [k_i] P_i` for every `i`, affine in and out.
pub fn mul_batch<P: RawCurve>(points: &[Affine<P>], scalars: &[Fr], out: &mut [Affine<P>])
where
    P::RF: GlvField,
{
    mul_batch_in(&mut BatchState::default(), points, scalars, out)
}

/// [`mul_batch`] with caller-kept working memory; see [`BatchState`].
pub fn mul_batch_in<P: RawCurve>(
    st: &mut BatchState<P::RF>,
    points: &[Affine<P>],
    scalars: &[Fr],
    out: &mut [Affine<P>],
) where
    P::RF: GlvField,
{
    assert_eq!(points.len(), out.len(), "one output per point");
    let xyzz: Vec<Xyzz<P::RF>> = points
        .iter()
        .map(|p| {
            if p.infinity {
                return Xyzz::ZERO;
            }
            let (x, y) = P::raw_xy(p);
            Xyzz {
                x,
                y,
                zz: P::RF::ONE,
                zzz: P::RF::ONE,
            }
        })
        .collect();
    let mut res = vec![Xyzz::ZERO; points.len()];
    mul_batch_xyzz_in(st, &xyzz, scalars, &mut res);
    for (o, r) in out.iter_mut().zip(&res) {
        *o = if r.is_zero() {
            Affine::identity()
        } else {
            Affine::new_unchecked(P::field_back(r.x), P::field_back(r.y))
        };
    }
}

/// `out[i] = [k_i] P_i` for every `i`. The inputs may be any XYZZ points; every output is
/// either the identity or affine, `ZZ = ZZZ = 1`.
///
/// On G2 the scalar is cut in four 64-bit parts rather than two 127-bit ones, with `psi`
/// beside `phi` ([`GlvField::split`]): half the doublings again, for two more tables.
pub fn mul_batch_xyzz<F: GlvField>(points: &[Xyzz<F>], scalars: &[Fr], out: &mut [Xyzz<F>]) {
    mul_batch_xyzz_in(&mut BatchState::default(), points, scalars, out)
}

/// The working memory of [`mul_batch_xyzz_in`], kept between calls. A batch of 1024 G2
/// points holds 4.6 MB of tables; allocating that afresh per call, from every thread of a
/// pass at once, costs page faults the multiplications then wait on.
pub struct BatchState<F> {
    dig: Vec<i8>,
    neg: Vec<bool>,
    tab: Vec<Pt<F>>,
    gone: Vec<bool>,
    two: Vec<Pt<F>>,
    next: Vec<Pt<F>>,
    acc: Vec<Pt<F>>,
    acc_zero: Vec<bool>,
    ok: Vec<bool>,
    idx: Vec<usize>,
    adds: Vec<(usize, Pt<F>)>,
    s: Scratch<F>,
}

impl<F> Default for BatchState<F> {
    fn default() -> Self {
        BatchState {
            dig: Vec::new(),
            neg: Vec::new(),
            tab: Vec::new(),
            gone: Vec::new(),
            two: Vec::new(),
            next: Vec::new(),
            acc: Vec::new(),
            acc_zero: Vec::new(),
            ok: Vec::new(),
            idx: Vec::new(),
            adds: Vec::new(),
            s: Scratch {
                den: Vec::new(),
                pre: Vec::new(),
                who: Vec::new(),
            },
        }
    }
}

/// `v` as `n` copies of `x`, reusing its allocation.
fn fill<T: Copy>(v: &mut Vec<T>, n: usize, x: T) {
    v.clear();
    v.resize(n, x);
}

/// [`mul_batch_xyzz`] with caller-kept working memory.
pub fn mul_batch_xyzz_in<F: GlvField>(
    st: &mut BatchState<F>,
    points: &[Xyzz<F>],
    scalars: &[Fr],
    out: &mut [Xyzz<F>],
) {
    assert_eq!(points.len(), scalars.len(), "one scalar per point");
    assert_eq!(points.len(), out.len(), "one output per point");
    let dims = F::DIMS;
    let base = to_affine_all(points, &mut st.s);

    let zero = Pt {
        x: F::ZERO,
        y: F::ZERO,
    };
    // Per point: up to four parts of the scalar (two on G1, four on G2; see
    // [`GlvField::split`]), each with its wNAF digits, sign and table of odd multiples.
    // Digits are stored step-major, `[s][j][i]`, so a step's scan for nonzero digits reads
    // one contiguous row per part rather than touching every point's whole state.
    let n = points.len();
    let row = |s: usize, j: usize| (s * dims + j) * n;
    let at = |i: usize, j: usize, e: usize| (i * dims + j) * TABLE + e;
    fill(&mut st.dig, DIGITS * dims * n, 0);
    fill(&mut st.neg, dims * n, false);
    // Every live entry is written before it is read, so the tables are only grown.
    if st.tab.len() < dims * TABLE * n {
        st.tab.resize(dims * TABLE * n, zero);
    }
    // Out of the batch: zero input, or an exceptional addition; see the module doc.
    fill(&mut st.gone, n, true);
    let mut steps = 0;
    let mut tmp = [0i8; DIGITS];
    for (i, (p, k)) in base.iter().zip(scalars).enumerate() {
        let Some(p) = p else { continue };
        let parts = F::split(k);
        let mut m = 0;
        for (j, part) in parts.iter().enumerate().take(dims) {
            let len = wnaf(part.unsigned_abs(), &mut tmp);
            for (s, &d) in tmp[..len].iter().enumerate() {
                st.dig[row(s, j) + i] = d;
            }
            st.neg[i * dims + j] = *part < 0;
            m = m.max(len);
        }
        if m > 0 {
            st.tab[at(i, 0, 0)] = *p;
            st.gone[i] = false;
            steps = steps.max(m);
        }
    }

    // Odd multiples: 2P, then 3P = P + 2P and so on, one batched addition per entry; then
    // each endomorphism's table is its image of the first, entry by entry.
    st.idx.clear();
    st.idx.extend((0..n).filter(|&i| !st.gone[i]));
    {
        st.two.clear();
        st.two.extend((0..n).map(|i| st.tab[at(i, 0, 0)]));
        fill(&mut st.ok, n, true);
        batch_dbl(&mut st.two, &st.idx, &mut st.ok, &mut st.s);
        fill(&mut st.next, n, zero);
        fill(&mut st.acc_zero, n, false);
        for e in 1..TABLE {
            st.adds.clear();
            for &i in &st.idx {
                st.next[i] = st.tab[at(i, 0, e - 1)];
                st.adds.push((i, st.two[i]));
            }
            batch_add(
                &mut st.next,
                &mut st.acc_zero,
                &st.adds,
                &mut st.ok,
                &mut st.s,
            );
            for &i in &st.idx {
                st.tab[at(i, 0, e)] = st.next[i];
            }
        }
        // Odd multiples of a point of prime order never collide; this is belt and braces.
        for &i in &st.idx {
            if !st.ok[i] {
                st.gone[i] = true;
            }
            for j in 1..dims {
                for e in 0..TABLE {
                    let q = st.tab[at(i, 0, e)];
                    let (x, y) = F::endo_j(j, q.x, q.y);
                    st.tab[at(i, j, e)] = Pt { x, y };
                }
            }
        }
    }

    fill(&mut st.acc, n, zero);
    fill(&mut st.acc_zero, n, true);
    fill(&mut st.ok, n, true);
    for s in (0..steps).rev() {
        st.idx.clear();
        st.idx
            .extend((0..n).filter(|&i| !st.gone[i] && st.ok[i] && !st.acc_zero[i]));
        batch_dbl(&mut st.acc, &st.idx, &mut st.ok, &mut st.s);
        for j in 0..dims {
            st.adds.clear();
            let r = &st.dig[row(s, j)..row(s, j) + n];
            for (i, &d) in r.iter().enumerate() {
                if d == 0 || st.gone[i] || !st.ok[i] {
                    continue;
                }
                let mut e = st.tab[at(i, j, (d.unsigned_abs() >> 1) as usize)];
                if (d < 0) != st.neg[i * dims + j] {
                    e.y = e.y.neg();
                }
                st.adds.push((i, e));
            }
            batch_add(
                &mut st.acc,
                &mut st.acc_zero,
                &st.adds,
                &mut st.ok,
                &mut st.s,
            );
        }
    }

    for i in 0..n {
        out[i] = if base[i].is_none() || scalars[i] == Fr::from(0u64) {
            Xyzz::ZERO
        } else if st.gone[i] || !st.ok[i] {
            let r = super::mul(&points[i], &scalars[i]);
            to_affine_all(&[r], &mut st.s)[0].map_or(Xyzz::ZERO, |p| affine_xyzz(p))
        } else if st.acc_zero[i] {
            Xyzz::ZERO
        } else {
            affine_xyzz(st.acc[i])
        };
    }
}

fn affine_xyzz<F: RawField>(p: Pt<F>) -> Xyzz<F> {
    Xyzz {
        x: p.x,
        y: p.y,
        zz: F::ONE,
        zzz: F::ONE,
    }
}

/// XYZZ to affine with one inversion for the lot: `x = X / ZZ`, `y = Y / ZZZ`, both from
/// the inverse of `ZZ ZZZ`. The identity comes back as `None`.
fn to_affine_all<F: GlvField>(points: &[Xyzz<F>], s: &mut Scratch<F>) -> Vec<Option<Pt<F>>> {
    s.den.clear();
    s.who.clear();
    for (i, p) in points.iter().enumerate() {
        if !p.is_zero() && !(p.zz == F::ONE && p.zzz == F::ONE) {
            s.den.push(p.zz.mul(p.zzz));
            s.who.push(i);
        }
    }
    invert_all(s);
    let mut out: Vec<Option<Pt<F>>> = points
        .iter()
        .map(|p| (!p.is_zero()).then_some(Pt { x: p.x, y: p.y }))
        .collect();
    for (k, &i) in s.who.iter().enumerate() {
        let p = &points[i];
        let t = s.den[k];
        out[i] = Some(Pt {
            x: p.x.mul(p.zzz).mul(t),
            y: p.y.mul(p.zz).mul(t),
        });
    }
    out
}

struct Scratch<F> {
    den: Vec<F>,
    pre: Vec<F>,
    who: Vec<usize>,
}

/// `a[i] = 2 a[i]` for `i` in `idx`. A point with `y = 0` has order two, which neither
/// BN254 group contains; one is marked not `ok` rather than divided by.
fn batch_dbl<F: GlvField>(a: &mut [Pt<F>], idx: &[usize], ok: &mut [bool], s: &mut Scratch<F>) {
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
    invert_all(s);
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
fn batch_add<F: GlvField>(
    a: &mut [Pt<F>],
    zero: &mut [bool],
    adds: &[(usize, Pt<F>)],
    ok: &mut [bool],
    s: &mut Scratch<F>,
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
    invert_all(s);
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
fn invert_all<F: GlvField>(s: &mut Scratch<F>) {
    let n = s.den.len();
    if n == 0 {
        return;
    }
    s.pre.clear();
    let mut acc = F::ONE;
    for d in &s.den {
        s.pre.push(acc);
        acc = acc.mul(*d);
    }
    let mut inv = acc.inv();
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
        batch_add(
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
