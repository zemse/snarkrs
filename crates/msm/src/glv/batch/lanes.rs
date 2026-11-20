//! The ladder of [`super::mul_sums_xyzz_in`] on AVX-512 IFMA, eight points a vector.
//!
//! Same algorithm, same rounds, same exceptional cases: only the arithmetic moves. The digits
//! and the tables are built by the portable code exactly as before; the tables are then taken
//! into the radix-2^52 domain of [`crate::ifma`] once, the accumulators live there for the
//! whole ladder, and only the result comes back. Every output is affine and canonical, so
//! the bytes are the portable path's.
//!
//! Layout. The accumulators are blocked: block `b` holds points `8b..8b+8`, one vector per
//! limb of each coordinate, so a doubling (which every live point takes every step) is plain
//! vector loads and stores. An addition round touches the few points with a nonzero digit,
//! which are copied into lanes eight at a time, limb by limb; the tables are rows of `x` then
//! `y` limbs, which is what that copy reads. A batch inversion runs four interleaved chains of
//! lane vectors, so 32 running products, and one scalar inversion per round.
//!
//! On this lane domain see `crate::ifma`: values are lazy in `[0, 2p)`, and `is_zero` tests
//! for a multiple of `p`.

use core::any::TypeId;
use core::arch::x86_64::*;

use snarkrs_field::raw::{RawFq, RawFq2};

use super::{BatchState, Pt, TABLE};
use crate::glv::GlvField;
use crate::ifma::{self, Lanes, V, V2};

/// Chains a batch inversion interleaves. A product's latency is a few times its
/// throughput, and a chain step has nothing else to overlap with.
const CHAINS: usize = 4;

/// Lane vectors in the blocked layout, and the two extra operations the ladder needs.
pub(crate) trait Blk: Lanes {
    /// `LIMBS * 8` u64, limb-major, lane-minor.
    unsafe fn load(p: *const u64) -> Self;
    unsafe fn store(&self, p: *mut u64);
    unsafe fn sqr(&self) -> Self;
    unsafe fn neg(&self) -> Self;
    /// The Frobenius: the identity on `Fq`, conjugation on `Fq2`.
    unsafe fn conj(&self) -> Self;
}

impl Blk for V {
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn load(p: *const u64) -> Self {
        let mut v = [_mm512_setzero_si512(); 5];
        for (j, vj) in v.iter_mut().enumerate() {
            *vj = _mm512_loadu_si512(p.add(8 * j) as *const _);
        }
        v
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn store(&self, p: *mut u64) {
        for (j, vj) in self.iter().enumerate() {
            _mm512_storeu_si512(p.add(8 * j) as *mut _, *vj);
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn sqr(&self) -> Self {
        ifma::mul(self, self)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn neg(&self) -> Self {
        ifma::sub(&[_mm512_setzero_si512(); 5], self)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn conj(&self) -> Self {
        *self
    }
}

impl Blk for V2 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn load(p: *const u64) -> Self {
        V2(V::load(p), V::load(p.add(40)))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn store(&self, p: *mut u64) {
        self.0.store(p);
        self.1.store(p.add(40));
    }
    /// `(a + b)(a - b) + 2ab u`, two products.
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn sqr(&self) -> Self {
        let c0 = ifma::mul(&ifma::add(&self.0, &self.1), &ifma::sub(&self.0, &self.1));
        let ab = ifma::mul(&self.0, &self.1);
        V2(c0, ifma::add(&ab, &ab))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn neg(&self) -> Self {
        V2(self.0.neg(), self.1.neg())
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn conj(&self) -> Self {
        V2(self.0, self.1.neg())
    }
}

/// Whether [`run`] takes batches of `F` on this machine.
pub(super) fn available<F: GlvField>() -> bool {
    let t = TypeId::of::<F>();
    (t == TypeId::of::<RawFq>() || t == TypeId::of::<RawFq2>()) && ifma::available()
}

/// The tables and the ladder here; the caller has checked [`available`]. Leaves `gone`,
/// the accumulators, `acc_zero` and `ok` of `st` as the portable tables and ladder do.
pub(super) fn run<F: GlvField>(
    st: &mut BatchState<F>,
    base: &[Option<Pt<F>>],
    n: usize,
    m: usize,
    steps: usize,
) {
    if TypeId::of::<F>() == TypeId::of::<RawFq>() {
        // SAFETY: `F` is `RawFq`, so these are the same types.
        let st = unsafe { &mut *(st as *mut BatchState<F> as *mut BatchState<RawFq>) };
        let base = unsafe { &*(base as *const [Option<Pt<F>>] as *const [Option<Pt<RawFq>>]) };
        // SAFETY: `available` checked the target features.
        unsafe { ladder::<V>(st, base, n, m, steps) };
    } else if TypeId::of::<F>() == TypeId::of::<RawFq2>() {
        // SAFETY: as above.
        let st = unsafe { &mut *(st as *mut BatchState<F> as *mut BatchState<RawFq2>) };
        let base = unsafe { &*(base as *const [Option<Pt<F>>] as *const [Option<Pt<RawFq2>>]) };
        // SAFETY: as above.
        unsafe { ladder::<V2>(st, base, n, m, steps) };
    } else {
        unreachable!("run without available");
    }
}

/// The working memory of one ladder, kept in [`BatchState`] between calls.
#[derive(Default)]
pub(super) struct LaneState {
    ax: Vec<u64>,
    ay: Vec<u64>,
    tab: Vec<u64>,
    zero: Vec<u8>,
    bad: Vec<u8>,
    /// Per item of a batch inversion: its denominator, then its running product.
    den: Vec<u64>,
    pre: Vec<u64>,
    /// Per addition group: `a.x, a.y, q.x, q.y`.
    grp: Vec<u64>,
    blocks: Vec<(u32, u8)>,
    groups: Vec<([u32; 8], u8)>,
    rounds: Vec<Vec<(u32, u32, bool)>>,
    used: Vec<u32>,
    /// Table building, per block of eight terms: the entry just made and `2P`.
    cx: Vec<u64>,
    cy: Vec<u64>,
    tx: Vec<u64>,
    ty: Vec<u64>,
}

#[inline(always)]
fn bit(m: &[u8], i: usize) -> bool {
    m[i >> 3] >> (i & 7) & 1 == 1
}

/// Element `(i, j)` of a blocked array of `lim`-limb lanes.
#[inline(always)]
fn blk_at(lim: usize, i: usize, j: usize) -> usize {
    ((i >> 3) * lim + j) * 8 + (i & 7)
}

/// `f(k, 1 / den_k)` for every `k < count`, the denominators at `den[k * w..]`, through one
/// scalar inversion of the lane products. Every lane of every denominator must be nonzero.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn batch_inv<L: Blk>(
    den: &[u64],
    pre: &mut Vec<u64>,
    count: usize,
    mut f: impl FnMut(usize, L),
) {
    if count == 0 {
        return;
    }
    let w = L::LIMBS * 8;
    pre.clear();
    pre.resize(count * w, 0);
    let mut acc = [L::one(); CHAINS];
    for k in 0..count {
        let c = k % CHAINS;
        acc[c].store(pre.as_mut_ptr().add(k * w));
        acc[c] = acc[c].mul(&L::load(den.as_ptr().add(k * w)));
    }
    // inv[c] = 1 / acc[c], from the inverse of the product of all four.
    let p01 = acc[0].mul(&acc[1]);
    let p23 = acc[2].mul(&acc[3]);
    let t = p01.mul(&p23).invert();
    let i01 = t.mul(&p23);
    let i23 = t.mul(&p01);
    let mut inv = [
        i01.mul(&acc[1]),
        i01.mul(&acc[0]),
        i23.mul(&acc[3]),
        i23.mul(&acc[2]),
    ];
    for k in (0..count).rev() {
        let c = k % CHAINS;
        let ik = inv[c].mul(&L::load(pre.as_ptr().add(k * w)));
        inv[c] = inv[c].mul(&L::load(den.as_ptr().add(k * w)));
        f(k, ik);
    }
}

/// Copy limbs of eight points out of a blocked array into one lane block.
#[inline(always)]
unsafe fn pick(src: &[u64], lim: usize, idx: &[u32; 8], mask: u8, out: *mut u64) {
    for (l, &i) in idx.iter().enumerate() {
        if mask >> l & 1 == 1 {
            for j in 0..lim {
                *out.add(j * 8 + l) = *src.get_unchecked(blk_at(lim, i as usize, j));
            }
        }
    }
}

/// The inverse of [`pick`].
#[inline(always)]
unsafe fn put(dst: &mut [u64], lim: usize, idx: &[u32; 8], mask: u8, from: *const u64) {
    for (l, &i) in idx.iter().enumerate() {
        if mask >> l & 1 == 1 {
            for j in 0..lim {
                *dst.get_unchecked_mut(blk_at(lim, i as usize, j)) = *from.add(j * 8 + l);
            }
        }
    }
}

/// Table rows of eight entries into two lane blocks, `x` and `y`.
#[inline(always)]
unsafe fn pick_rows(tab: &[u64], lim: usize, rows: &[u32; 8], mask: u8, x: *mut u64, y: *mut u64) {
    for (l, &r) in rows.iter().enumerate() {
        if mask >> l & 1 == 1 {
            let row = tab.as_ptr().add(r as usize * 2 * lim);
            for j in 0..lim {
                *x.add(j * 8 + l) = *row.add(j);
                *y.add(j * 8 + l) = *row.add(lim + j);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn ladder<L: Blk>(
    st: &mut BatchState<L::Raw>,
    base: &[Option<Pt<L::Raw>>],
    n: usize,
    m: usize,
    steps: usize,
) where
    L::Raw: GlvField,
{
    let dims = <L::Raw as GlvField>::DIMS;
    let lim = L::LIMBS;
    let w = lim * 8;
    let nb = n.div_ceil(8);
    let mut ls = std::mem::take(&mut st.lanes);
    let LaneState {
        ax,
        ay,
        tab,
        zero,
        bad,
        den,
        pre,
        grp,
        blocks,
        groups,
        rounds,
        used,
        cx,
        cy,
        tx,
        ty,
    } = &mut ls;

    // The tables first, since a failure there takes its sum out of the batch.
    let at = |u: usize, d: usize, e: usize| (u * dims + d) * TABLE + e;
    if tab.len() < m * dims * TABLE * 2 * lim {
        tab.resize(m * dims * TABLE * 2 * lim, 0);
    }
    build_tables::<L>(st, base, m, tab, den, pre, blocks, [cx, cy, tx, ty]);

    ax.clear();
    ax.resize(nb * w, 0);
    ay.clear();
    ay.resize(nb * w, 0);
    zero.clear();
    zero.resize(nb, 0);
    bad.clear();
    bad.resize(nb, 0);
    for i in 0..nb * 8 {
        if i < n {
            zero[i >> 3] |= 1 << (i & 7);
        }
        // Padding lanes never act.
        if i >= n || st.gone[i] {
            bad[i >> 3] |= 1 << (i & 7);
        }
    }

    let one = L::one();
    let row = |s: usize, d: usize| (s * dims + d) * m;
    let mut bx = [0u64; 80];
    let mut by = [0u64; 80];

    for s in (0..steps).rev() {
        // Doubling, every live block.
        blocks.clear();
        for b in 0..nb {
            let live = !(zero[b] | bad[b]);
            if live != 0 {
                blocks.push((b as u32, live));
            }
        }
        den.clear();
        den.resize(blocks.len() * w, 0);
        for (k, (b, live)) in blocks.iter_mut().enumerate() {
            let y = L::load(ay.as_ptr().add(*b as usize * w));
            let d = y.add(&y);
            // y = 0 has order two, which neither group contains; refused, not divided by.
            let z = d.is_zero() & *live;
            if z != 0 {
                bad[*b as usize] |= z;
                *live &= !z;
            }
            L::select(*live, &d, &one).store(den.as_mut_ptr().add(k * w));
        }
        batch_inv::<L>(den, pre, blocks.len(), |k, inv| {
            let (b, live) = blocks[k];
            let px = ax.as_mut_ptr().add(b as usize * w);
            let py = ay.as_mut_ptr().add(b as usize * w);
            let x = L::load(px);
            let y = L::load(py);
            let xx = x.sqr();
            let l = xx.add(&xx).add(&xx).mul(&inv);
            let x3 = l.sqr().sub(&x.add(&x));
            let y3 = l.mul(&x.sub(&x3)).sub(&y);
            L::select(live, &x3, &x).store(px);
            L::select(live, &y3, &y).store(py);
        });

        // Round `r` gets each sum's `r`-th nonzero digit of the step.
        let mut nr = 0;
        used.clear();
        used.resize(n, 0);
        for d in 0..dims {
            let r = &st.dig[row(s, d)..row(s, d) + m];
            for (u, &x) in r.iter().enumerate() {
                if x == 0 {
                    continue;
                }
                let i = st.owner[u];
                if bit(bad, i) {
                    continue;
                }
                let e = at(u, d, (x.unsigned_abs() >> 1) as usize);
                let negate = (x < 0) != st.neg[u * dims + d];
                let k = used[i] as usize;
                used[i] += 1;
                if k == rounds.len() {
                    rounds.push(Vec::new());
                }
                if k == nr {
                    rounds[k].clear();
                    nr += 1;
                }
                rounds[k].push((i as u32, e as u32, negate));
            }
        }

        for adds in &rounds[..nr] {
            // Into the identity is a copy, the table entry with its sign.
            groups.clear();
            let mut cur = ([0u32; 8], 0u8, [0u32; 8], 0u8);
            for &(i, e, negate) in adds.iter() {
                if !bit(zero, i as usize) {
                    continue;
                }
                let l = cur.1.count_ones() as usize;
                cur.0[l] = i;
                cur.2[l] = e;
                cur.1 |= 1 << l;
                cur.3 |= (negate as u8) << l;
                if l == 7 {
                    copy_group::<L>(ax, ay, tab, lim, &cur);
                    cur = ([0u32; 8], 0, [0u32; 8], 0);
                }
            }
            if cur.1 != 0 {
                copy_group::<L>(ax, ay, tab, lim, &cur);
            }

            // The rest: eight additions a group.
            grp.clear();
            den.clear();
            let mut g = ([0u32; 8], 0u8);
            let mut rows = [0u32; 8];
            let mut negs = 0u8;
            let flush = |g: &mut ([u32; 8], u8),
                         rows: &[u32; 8],
                         negs: u8,
                         groups: &mut Vec<([u32; 8], u8)>,
                         grp: &mut Vec<u64>,
                         den: &mut Vec<u64>,
                         bad: &mut [u8]| {
                let base = grp.len();
                grp.resize(base + 4 * w, 0);
                let gp = grp.as_mut_ptr().add(base);
                pick(ax, lim, &g.0, g.1, gp);
                pick(ay, lim, &g.0, g.1, gp.add(w));
                pick_rows(tab, lim, rows, g.1, gp.add(2 * w), gp.add(3 * w));
                let qy = L::load(gp.add(3 * w));
                L::select(negs, &qy.neg(), &qy).store(gp.add(3 * w));
                let dd = L::load(gp.add(2 * w)).sub(&L::load(gp));
                // A shared x: the doubling or the cancellation, out of the batch.
                let z = dd.is_zero() & g.1;
                if z != 0 {
                    for l in 0..8 {
                        if z >> l & 1 == 1 {
                            let i = g.0[l] as usize;
                            bad[i >> 3] |= 1 << (i & 7);
                        }
                    }
                    g.1 &= !z;
                }
                let k = den.len();
                den.resize(k + w, 0);
                L::select(g.1, &dd, &one).store(den.as_mut_ptr().add(k));
                groups.push(*g);
            };
            for &(i, e, negate) in adds.iter() {
                if bit(zero, i as usize) {
                    // Copied above.
                    continue;
                }
                let l = g.1.count_ones() as usize;
                g.0[l] = i;
                g.1 |= 1 << l;
                rows[l] = e;
                negs |= (negate as u8) << l;
                if l == 7 {
                    flush(&mut g, &rows, negs, groups, grp, den, bad);
                    g = ([0u32; 8], 0);
                    negs = 0;
                }
            }
            if g.1 != 0 {
                flush(&mut g, &rows, negs, groups, grp, den, bad);
            }
            // Copies clear the identity only now, so a point has its copy or its addition
            // in a round, never both (each point is in a round at most once).
            for &(i, _, _) in adds.iter() {
                zero[i as usize >> 3] &= !(1 << (i & 7));
            }
            let ng = groups.len();
            batch_inv::<L>(den, pre, ng, |k, inv| {
                let (idx, mask) = groups[k];
                let gp = grp.as_ptr().add(k * 4 * w);
                let (a_x, a_y) = (L::load(gp), L::load(gp.add(w)));
                let (q_x, q_y) = (L::load(gp.add(2 * w)), L::load(gp.add(3 * w)));
                let l = q_y.sub(&a_y).mul(&inv);
                let x3 = l.sqr().sub(&a_x).sub(&q_x);
                let y3 = l.mul(&a_x.sub(&x3)).sub(&a_y);
                x3.store(bx.as_mut_ptr());
                y3.store(by.as_mut_ptr());
                put(ax, lim, &idx, mask, bx.as_ptr());
                put(ay, lim, &idx, mask, by.as_ptr());
            });
        }
    }

    // Back out, canonical.
    for b in 0..nb {
        let xs = L::load(ax.as_ptr().add(b * w)).leave();
        let ys = L::load(ay.as_ptr().add(b * w)).leave();
        for l in 0..8 {
            let i = b * 8 + l;
            if i < n {
                st.acc[i].x = xs[l];
                st.acc[i].y = ys[l];
                st.acc_zero[i] = bit(zero, i);
                st.ok[i] = !bit(bad, i) || st.gone[i];
            }
        }
    }
    st.lanes = ls;
}

/// [`ladder`]'s copies into the identity: eight table entries, signed, into the accumulators.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn copy_group<L: Blk>(
    ax: &mut [u64],
    ay: &mut [u64],
    tab: &[u64],
    lim: usize,
    g: &([u32; 8], u8, [u32; 8], u8),
) {
    let mut bx = [0u64; 80];
    let mut by = [0u64; 80];
    pick_rows(tab, lim, &g.2, g.1, bx.as_mut_ptr(), by.as_mut_ptr());
    let y = L::load(by.as_ptr());
    L::select(g.3, &y.neg(), &y).store(by.as_mut_ptr());
    put(ax, lim, &g.0, g.1, bx.as_ptr());
    put(ay, lim, &g.0, g.1, by.as_ptr());
}

/// The odd-multiple tables of every term, `P, 3P, .., (2 need - 1) P` for part 0 and their
/// images under the endomorphisms for the other parts, as rows of `tab` (`x` limbs then
/// `y` limbs, lane domain). Blocks of eight consecutive terms: `2P` is one batched
/// doubling, each further entry one batched addition of `2P`, and every part past the
/// first is computed from the vectors at hand, so nothing is gathered. A term whose
/// doubling or addition meets the exceptional case (which a point of prime order never
/// does) takes its sum out of the batch, as the portable tables do.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn build_tables<L: Blk>(
    st: &mut BatchState<L::Raw>,
    base: &[Option<Pt<L::Raw>>],
    m: usize,
    tab: &mut [u64],
    den: &mut Vec<u64>,
    pre: &mut Vec<u64>,
    blocks: &mut Vec<(u32, u8)>,
    [cx, cy, tx, ty]: [&mut Vec<u64>; 4],
) where
    L::Raw: GlvField,
{
    use snarkrs_field::raw::RawField;
    let dims = <L::Raw as GlvField>::DIMS;
    let lim = L::LIMBS;
    let w = lim * 8;
    let nbt = m.div_ceil(8);
    let at = |u: usize, d: usize, e: usize| (u * dims + d) * TABLE + e;
    let need = |u: usize| if u < m { st.need[u] } else { 0 };
    // Lanes of block `b` whose terms have more than `e` entries.
    let mask = |b: usize, e: usize| -> u8 {
        let mut k = 0u8;
        for l in 0..8 {
            k |= ((need(8 * b + l) > e) as u8) << l;
        }
        k
    };
    // `endo_j(j, x, y) = (conj^c(x) kx_j, conj^c(y) ky_j)`, conjugated from `psi` on.
    let one_raw = <L::Raw as RawField>::ONE;
    let mut kx = [L::one(); 4];
    let mut ky = [L::one(); 4];
    for j in 1..dims {
        let (x, y) = <L::Raw as GlvField>::endo_j(j, one_raw, one_raw);
        kx[j] = L::enter(&[x; 8]);
        ky[j] = L::enter(&[y; 8]);
    }
    let one = L::one();
    let mut bad = vec![0u8; nbt];
    let mut bx = [0u64; 80];
    let mut by = [0u64; 80];
    // Entry `e` of every part, for the lanes of `k`, from part 0's `(x, y)`.
    let mut store = |b: usize, k: u8, e: usize, x: &L, y: &L, tab: &mut [u64]| {
        let mut emit = |d: usize, x: &L, y: &L| {
            x.store(bx.as_mut_ptr());
            y.store(by.as_mut_ptr());
            for l in 0..8 {
                if k >> l & 1 == 1 {
                    let r = at(8 * b + l, d, e) * 2 * lim;
                    for j in 0..lim {
                        *tab.get_unchecked_mut(r + j) = bx[j * 8 + l];
                        *tab.get_unchecked_mut(r + lim + j) = by[j * 8 + l];
                    }
                }
            }
        };
        emit(0, x, y);
        for d in 1..dims {
            // Part `d` is `endo_j(d)` of part 0, `phi psi` included: the constants of
            // `endo_j(3)` are those of `phi` after `psi`.
            let (xd, yd) = if d >= 2 {
                (x.conj().mul(&kx[d]), y.conj().mul(&ky[d]))
            } else {
                (x.mul(&kx[d]), *y)
            };
            emit(d, &xd, &yd);
        }
    };

    cx.clear();
    cx.resize(nbt * w, 0);
    cy.clear();
    cy.resize(nbt * w, 0);
    tx.clear();
    tx.resize(nbt * w, 0);
    ty.clear();
    ty.resize(nbt * w, 0);
    let zero_raw = <L::Raw as RawField>::ZERO;
    for b in 0..nbt {
        let k = mask(b, 0);
        if k == 0 {
            continue;
        }
        let mut xs = [zero_raw; 8];
        let mut ys = [zero_raw; 8];
        for l in 0..8 {
            if k >> l & 1 == 1 {
                let p = base[8 * b + l]
                    .as_ref()
                    .expect("a term with digits has a point");
                xs[l] = p.x;
                ys[l] = p.y;
            }
        }
        let (x, y) = (L::enter(&xs), L::enter(&ys));
        x.store(cx.as_mut_ptr().add(b * w));
        y.store(cy.as_mut_ptr().add(b * w));
        store(b, k, 0, &x, &y, tab);
    }

    // 2P where a second entry is needed.
    blocks.clear();
    den.clear();
    for b in 0..nbt {
        let mut k = mask(b, 1);
        if k == 0 {
            continue;
        }
        let y = L::load(cy.as_ptr().add(b * w));
        let d = y.add(&y);
        let z = d.is_zero() & k;
        bad[b] |= z;
        k &= !z;
        let at_k = den.len();
        den.resize(at_k + w, 0);
        L::select(k, &d, &one).store(den.as_mut_ptr().add(at_k));
        blocks.push((b as u32, k));
    }
    batch_inv::<L>(den, pre, blocks.len(), |i, inv| {
        let b = blocks[i].0 as usize;
        let x = L::load(cx.as_ptr().add(b * w));
        let y = L::load(cy.as_ptr().add(b * w));
        let xx = x.sqr();
        let l = xx.add(&xx).add(&xx).mul(&inv);
        let x3 = l.sqr().sub(&x.add(&x));
        let y3 = l.mul(&x.sub(&x3)).sub(&y);
        x3.store(tx.as_mut_ptr().add(b * w));
        y3.store(ty.as_mut_ptr().add(b * w));
    });

    for e in 1..TABLE {
        blocks.clear();
        den.clear();
        for b in 0..nbt {
            let mut k = mask(b, e) & !bad[b];
            if k == 0 {
                continue;
            }
            let d = L::load(tx.as_ptr().add(b * w)).sub(&L::load(cx.as_ptr().add(b * w)));
            let z = d.is_zero() & k;
            bad[b] |= z;
            k &= !z;
            let at_k = den.len();
            den.resize(at_k + w, 0);
            L::select(k, &d, &one).store(den.as_mut_ptr().add(at_k));
            blocks.push((b as u32, k));
        }
        if blocks.is_empty() {
            break;
        }
        batch_inv::<L>(den, pre, blocks.len(), |i, inv| {
            let (b, k) = blocks[i];
            let b = b as usize;
            let (px, py) = (cx.as_mut_ptr().add(b * w), cy.as_mut_ptr().add(b * w));
            let (x, y) = (L::load(px), L::load(py));
            let qx = L::load(tx.as_ptr().add(b * w));
            let qy = L::load(ty.as_ptr().add(b * w));
            let l = qy.sub(&y).mul(&inv);
            let x3 = l.sqr().sub(&x).sub(&qx);
            let y3 = l.mul(&x.sub(&x3)).sub(&y);
            x3.store(px);
            y3.store(py);
            store(b, k, e, &x3, &y3, tab);
        });
    }

    for b in 0..nbt {
        for l in 0..8 {
            if bad[b] >> l & 1 == 1 {
                st.gone[st.owner[8 * b + l]] = true;
            }
        }
    }
}
