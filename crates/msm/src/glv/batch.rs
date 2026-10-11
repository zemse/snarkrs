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
    need: Vec<usize>,
    owner: Vec<usize>,
    ends: Vec<usize>,
    count: Vec<u32>,
    touched: Vec<usize>,
    rounds: Vec<Vec<(usize, Pt<F>)>>,
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
            need: Vec::new(),
            owner: Vec::new(),
            ends: Vec::new(),
            count: Vec::new(),
            touched: Vec::new(),
            rounds: Vec::new(),
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
    assert_eq!(points.len(), out.len(), "one output per point");
    mul_sum_batch_xyzz_in(st, points, scalars, 1, out)
}

/// `out[i] = sum_t [k_(i terms + t)] P_(i terms + t)` for every `i`: sums of `terms`
/// products, each sum one ladder. The terms of a sum share its doublings (Shamir's trick
/// over the GLV parts of every term), so a sum of two costs the doublings of one product
/// and the additions of two. Inputs and outputs as in [`mul_batch_xyzz`].
pub fn mul_sum_batch_xyzz_in<F: GlvField>(
    st: &mut BatchState<F>,
    points: &[Xyzz<F>],
    scalars: &[Fr],
    terms: usize,
    out: &mut [Xyzz<F>],
) {
    assert!(terms > 0, "a sum of at least one term");
    assert_eq!(points.len(), terms * out.len(), "terms points per output");
    let mut ends = std::mem::take(&mut st.ends);
    ends.clear();
    ends.extend((1..=out.len()).map(|i| i * terms));
    mul_sums_xyzz_in(st, points, scalars, &ends, out);
    st.ends = ends;
}

/// [`mul_sum_batch_xyzz_in`] with a term count of its own per sum: `out[i]` sums the
/// terms `ends[i - 1]..ends[i]` (from 0 for `i = 0`), so `ends` is nondecreasing and ends
/// at `points.len()`. An empty range is the identity.
///
/// Each ladder step is a doubling round and then addition rounds, each sum taking at most
/// one addition a round (its `r`-th nonzero digit of the step goes in round `r`), so the
/// step costs as many shared inversions as its busiest sum has nonzero digits. A term's
/// table holds only the odd multiples its digits use: a small scalar, `1` or `-1` say,
/// builds none.
pub fn mul_sums_xyzz_in<F: GlvField>(
    st: &mut BatchState<F>,
    points: &[Xyzz<F>],
    scalars: &[Fr],
    ends: &[usize],
    out: &mut [Xyzz<F>],
) {
    assert_eq!(points.len(), scalars.len(), "one scalar per point");
    assert_eq!(ends.len(), out.len(), "one end per output");
    assert!(
        ends.windows(2).all(|w| w[0] <= w[1]) && ends.last().map_or(0, |&e| e) == points.len(),
        "ends run up to the term count"
    );
    let dims = F::DIMS;
    let base = to_affine_all(points, &mut st.s);

    let zero = Pt {
        x: F::ZERO,
        y: F::ZERO,
    };
    // Per term `u` and part `d` of its scalar (two on G1, four on G2; see
    // [`GlvField::split`]): wNAF digits, a sign and a table of odd multiples. Digits are
    // stored step-major, `[s][d][u]`, so a step's scan for nonzero digits reads one
    // contiguous row per part rather than touching every term's whole state.
    let n = out.len();
    let m = points.len();
    st.owner.clear();
    for (i, &e) in ends.iter().enumerate() {
        st.owner.resize(e, i);
    }
    let row = |s: usize, d: usize| (s * dims + d) * m;
    let at = |u: usize, d: usize, e: usize| (u * dims + d) * TABLE + e;
    fill(&mut st.dig, DIGITS * dims * m, 0);
    fill(&mut st.neg, dims * m, false);
    // Every live entry is written before it is read, so the tables are only grown.
    if st.tab.len() < dims * TABLE * m {
        st.tab.resize(dims * TABLE * m, zero);
    }
    // Out of the batch: every term zero, or an exceptional addition; see the module doc.
    fill(&mut st.gone, n, true);
    // Per term: how many odd multiples its digits use, 0 for a term with none.
    fill(&mut st.need, m, 0);
    let mut steps = 0;
    let mut tmp = [0i8; DIGITS];
    for (u, (p, k)) in base.iter().zip(scalars).enumerate() {
        let Some(p) = p else { continue };
        let split = F::split(k);
        let (mut len_max, mut top) = (0, 0);
        for (d, part) in split.iter().enumerate().take(dims) {
            let len = wnaf(part.unsigned_abs(), &mut tmp);
            for (s, &x) in tmp[..len].iter().enumerate() {
                st.dig[row(s, d) + u] = x;
                top = top.max(x.unsigned_abs());
            }
            st.neg[u * dims + d] = *part < 0;
            len_max = len_max.max(len);
        }
        if len_max > 0 {
            st.tab[at(u, 0, 0)] = *p;
            st.need[u] = top as usize / 2 + 1;
            st.gone[st.owner[u]] = false;
            steps = steps.max(len_max);
        }
    }

    // Odd multiples: 2P, then 3P = P + 2P and so on, one batched addition per entry; then
    // each endomorphism's table is its image of the first, entry by entry.
    {
        st.idx.clear();
        st.idx.extend((0..m).filter(|&u| st.need[u] > 1));
        st.two.clear();
        st.two.extend((0..m).map(|u| st.tab[at(u, 0, 0)]));
        fill(&mut st.ok, m, true);
        batch_dbl(&mut st.two, &st.idx, &mut st.ok, &mut st.s);
        fill(&mut st.next, m, zero);
        fill(&mut st.acc_zero, m, false);
        for e in 1..TABLE {
            st.idx.retain(|&u| st.need[u] > e);
            st.adds.clear();
            for &u in &st.idx {
                st.next[u] = st.tab[at(u, 0, e - 1)];
                st.adds.push((u, st.two[u]));
            }
            batch_add(
                &mut st.next,
                &mut st.acc_zero,
                &st.adds,
                &mut st.ok,
                &mut st.s,
            );
            for &u in &st.idx {
                st.tab[at(u, 0, e)] = st.next[u];
            }
        }
        // Odd multiples of a point of prime order never collide; this is belt and braces.
        for u in (0..m).filter(|&u| st.need[u] > 0) {
            if !st.ok[u] {
                st.gone[st.owner[u]] = true;
            }
            // An odd part's table is `phi` of the one before it, so `phi psi` costs a `phi`
            // rather than a `psi` and a `phi`.
            for d in 1..dims {
                let (src, j) = if d % 2 == 1 { (d - 1, 1) } else { (0, d) };
                for e in 0..st.need[u] {
                    let q = st.tab[at(u, src, e)];
                    let (x, y) = F::endo_j(j, q.x, q.y);
                    st.tab[at(u, d, e)] = Pt { x, y };
                }
            }
        }
    }

    fill(&mut st.acc, n, zero);
    fill(&mut st.acc_zero, n, true);
    fill(&mut st.ok, n, true);
    fill(&mut st.count, n, 0);
    for s in (0..steps).rev() {
        st.idx.clear();
        st.idx
            .extend((0..n).filter(|&i| !st.gone[i] && st.ok[i] && !st.acc_zero[i]));
        batch_dbl(&mut st.acc, &st.idx, &mut st.ok, &mut st.s);
        // Round `r` gets each sum's `r`-th nonzero digit of the step.
        let mut used = 0;
        st.touched.clear();
        for d in 0..dims {
            let r = &st.dig[row(s, d)..row(s, d) + m];
            for (u, &x) in r.iter().enumerate() {
                if x == 0 {
                    continue;
                }
                let i = st.owner[u];
                if st.gone[i] || !st.ok[i] {
                    continue;
                }
                let mut e = st.tab[at(u, d, (x.unsigned_abs() >> 1) as usize)];
                if (x < 0) != st.neg[u * dims + d] {
                    e.y = e.y.neg();
                }
                let k = st.count[i] as usize;
                if k == 0 {
                    st.touched.push(i);
                }
                st.count[i] += 1;
                if k == st.rounds.len() {
                    st.rounds.push(Vec::new());
                }
                if k == used {
                    st.rounds[k].clear();
                    used += 1;
                }
                st.rounds[k].push((i, e));
            }
        }
        for &i in &st.touched {
            st.count[i] = 0;
        }
        for adds in &st.rounds[..used] {
            batch_add(&mut st.acc, &mut st.acc_zero, adds, &mut st.ok, &mut st.s);
        }
    }

    let mut from = 0;
    for (i, &to) in ends.iter().enumerate() {
        let us = from..to;
        from = to;
        out[i] = if us
            .clone()
            .all(|u| base[u].is_none() || scalars[u] == Fr::from(0u64))
        {
            Xyzz::ZERO
        } else if st.gone[i] || !st.ok[i] {
            let mut r = Xyzz::ZERO;
            for u in us {
                r.add_assign(&super::mul(&points[u], &scalars[u]));
            }
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
    let mut out: Vec<Option<Pt<F>>> = points
        .iter()
        .map(|p| (!p.is_zero()).then_some(Pt { x: p.x, y: p.y }))
        .collect();
    let who = &s.who;
    invert_each(&s.den, &mut s.pre, |k, t| {
        let p = &points[who[k]];
        out[who[k]] = Some(Pt {
            x: p.x.mul(p.zzz).mul(t),
            y: p.y.mul(p.zz).mul(t),
        });
    });
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
    let who = &s.who;
    invert_pairs(&s.den, &mut s.pre, |k, inv, other| {
        let i = who[k];
        let p = a[i];
        let Some((k2, inv2)) = other else {
            let xx = p.x.sqr();
            let l = xx.double().add(xx).mul(inv);
            let x3 = l.sqr().sub(p.x.double());
            let y3 = l.mul(p.x.sub(x3)).sub(p.y);
            a[i] = Pt { x: x3, y: y3 };
            return;
        };
        // Two at once, step by step; see [`invert_pairs`].
        let j = who[k2];
        let q = a[j];
        let (xx, xx2) = (p.x.sqr(), q.x.sqr());
        let l = xx.double().add(xx).mul(inv);
        let l2 = xx2.double().add(xx2).mul(inv2);
        let x3 = l.sqr().sub(p.x.double());
        let x32 = l2.sqr().sub(q.x.double());
        let y3 = l.mul(p.x.sub(x3)).sub(p.y);
        let y32 = l2.mul(q.x.sub(x32)).sub(q.y);
        a[i] = Pt { x: x3, y: y3 };
        a[j] = Pt { x: x32, y: y32 };
    });
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
    let who = &s.who;
    invert_pairs(&s.den, &mut s.pre, |j, inv, other| {
        let (i, q) = adds[who[j]];
        let p = a[i];
        let Some((j2, inv2)) = other else {
            let l = q.y.sub(p.y).mul(inv);
            let x3 = l.sqr().sub(p.x).sub(q.x);
            let y3 = l.mul(p.x.sub(x3)).sub(p.y);
            a[i] = Pt { x: x3, y: y3 };
            return;
        };
        // Two at once, step by step; see [`invert_pairs`].
        let (i2, q2) = adds[who[j2]];
        let p2 = a[i2];
        let l = q.y.sub(p.y).mul(inv);
        let l2 = q2.y.sub(p2.y).mul(inv2);
        let x3 = l.sqr().sub(p.x).sub(q.x);
        let x32 = l2.sqr().sub(p2.x).sub(q2.x);
        let y3 = l.mul(p.x.sub(x3)).sub(p.y);
        let y32 = l2.mul(p2.x.sub(x32)).sub(p2.y);
        a[i] = Pt { x: x3, y: y3 };
        a[i2] = Pt { x: x32, y: y32 };
    });
}

/// `f(k, 1 / den[k])` for every `k`, from one field inversion.
#[inline(always)]
fn invert_each<F: GlvField>(den: &[F], pre: &mut Vec<F>, mut f: impl FnMut(usize, F)) {
    invert_pairs(den, pre, |k, i, other| {
        f(k, i);
        if let Some((k2, i2)) = other {
            f(k2, i2);
        }
    });
}

/// [`invert_each`] two at a time: `f(k, 1 / den[k], Some((k2, 1 / den[k2])))`, or `None`
/// for the one left over.
///
/// The running products are two chains, one per half, walked side by side, and their two
/// totals share the inversion. Each pair of inverses goes to `f` as it comes out of the
/// backward walk, one from each chain, so the caller's formula runs in that loop rather
/// than in a pass of its own, and for two independent points at once. That is the point:
/// a doubling or an addition is a chain of dependent multiplies, each a long asm block,
/// longer than what the core looks ahead across, so one point at a time leaves the
/// multiplier waiting; two interleaved step by step keep it busy.
#[inline(always)]
fn invert_pairs<F: GlvField>(
    den: &[F],
    pre: &mut Vec<F>,
    mut f: impl FnMut(usize, F, Option<(usize, F)>),
) {
    let n = den.len();
    if n == 0 {
        return;
    }
    let h = n / 2;
    pre.clear();
    pre.resize(n, F::ONE);
    let (mut a, mut b) = (F::ONE, F::ONE);
    for k in 0..h {
        pre[k] = a;
        pre[h + k] = b;
        a = a.mul(den[k]);
        b = b.mul(den[h + k]);
    }
    // An odd one out goes on the end of the second chain.
    if n % 2 == 1 {
        pre[n - 1] = b;
        b = b.mul(den[n - 1]);
    }
    let ab = a.mul(b).inv();
    let (mut ia, mut ib) = (ab.mul(b), ab.mul(a));
    if n % 2 == 1 {
        f(n - 1, ib.mul(pre[n - 1]), None);
        ib = ib.mul(den[n - 1]);
    }
    for k in (0..h).rev() {
        f(k, ia.mul(pre[k]), Some((h + k, ib.mul(pre[h + k]))));
        ia = ia.mul(den[k]);
        ib = ib.mul(den[h + k]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
    use snarkrs_field::{g1, G1Projective};

    type F = snarkrs_field::raw::RawFq;

    /// Both chains of the shared inversion, at every length up to a few past two, odd ones
    /// included, against one inversion per element.
    #[test]
    fn invert_each_matches_one_at_a_time() {
        let mut pre = Vec::new();
        invert_each::<F>(&[], &mut pre, |_, _| panic!("nothing to invert"));
        let mut x = F::ONE.add(F::ONE);
        for n in 1..=9 {
            let den: Vec<F> = (0..n)
                .map(|_| {
                    x = x.sqr().add(F::ONE);
                    x
                })
                .collect();
            let mut seen = vec![false; n];
            invert_each(&den, &mut pre, |k, i| {
                assert_eq!(den[k].mul(i), F::ONE, "n = {n}, k = {k}");
                seen[k] = true;
            });
            assert!(seen.iter().all(|&s| s), "n = {n}");
        }
    }

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
