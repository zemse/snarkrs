//! The G1 batch-affine fill on AVX-512 IFMA, eight entries per vector.
//!
//! Chosen at run time by [`prepare`], so a generic x86_64 build carries it and a machine
//! without the extension never executes an instruction of it. The scheduling (busy map,
//! retries, side buckets) is [`BatchFill`](crate::BatchFill)'s, line for line; only the
//! arithmetic and the bucket storage differ.
//!
//! The field is BN254 Fq in radix 2^52, five limbs per element, one element per 64-bit
//! lane, Montgomery with R = 2^260 against the scalar path's 2^256. Values are lazy, in
//! `[0, 2p)`, and every product takes inputs in that range, which keeps its output there
//! because `4p < 2^260`. Everything the fill touches per entry stays in this domain: the
//! bases are converted once per MSM in scan order ([`Bases`]), the buckets live as
//! radix-2^52 rows for the whole chunk, and only `finish` converts them back, one bulk pass
//! before the scalar running sum. The radix change costs one product per element each way.
//!
//! A flush walks the pending entries in groups of eight, lane `l` of group `g` holding
//! entry `8g + l`, so the forward prefix product is eight interleaved chains, one per lane.
//! The eight chain totals are inverted together (one scalar inversion) and the backward
//! pass recovers each entry's inverse lane-parallel. Doublings and cancellations are lane
//! masks: a doubling swaps the numerator and denominator (the result formulas are shared,
//! since `px == bx` there), a cancellation contributes ONE and is not written back.

use core::arch::x86_64::*;

use ark_ec::short_weierstrass::Projective;
use ark_ff::{AdditiveGroup, BigInt, Field};
use rayon::prelude::*;
use snarkrs_field::raw::{RawFq, FQ_MOD};
use snarkrs_field::{Fq, G1Affine};

use crate::xyzz::{to_projective, RawCurve, Xyzz};
use crate::{affine_running_sum, batch_size, signed_digit, FastFill, Prescan, SideBuckets, Work};

type G1 = snarkrs_field::g1::Config;
type V = [__m512i; 5];

const MASK: u64 = (1 << 52) - 1;
/// Limbs per row: x then y, five each.
const ROW: usize = 10;
/// Set on a scan index in the pending list when the point enters negated.
const NEG: u32 = 1 << 31;

const fn to_52(x: [u64; 4]) -> [u64; 5] {
    [
        x[0] & MASK,
        (x[0] >> 52 | x[1] << 12) & MASK,
        (x[1] >> 40 | x[2] << 24) & MASK,
        (x[2] >> 28 | x[3] << 36) & MASK,
        x[3] >> 16,
    ]
}

/// Inverse of [`to_52`] for normalised limbs of a value below 2^256.
const fn from_52(l: [u64; 5]) -> [u64; 4] {
    [
        l[0] | l[1] << 52,
        l[1] >> 12 | l[2] << 40,
        l[2] >> 24 | l[3] << 28,
        l[3] >> 36 | l[4] << 16,
    ]
}

/// `2^e mod p` by repeated doubling; compile time only.
const fn pow2_mod_p(e: u32) -> [u64; 4] {
    let mut x = [1u64, 0, 0, 0];
    let mut i = 0;
    while i < e {
        // x < p < 2^254, so the doubling cannot carry out of limb 3.
        let y = [
            x[0] << 1,
            x[1] << 1 | x[0] >> 63,
            x[2] << 1 | x[1] >> 63,
            x[3] << 1 | x[2] >> 63,
        ];
        let mut ge = true;
        let mut j = 4;
        while j > 0 {
            j -= 1;
            if y[j] != FQ_MOD[j] {
                ge = y[j] > FQ_MOD[j];
                break;
            }
        }
        x = if ge { sub_4(y, FQ_MOD) } else { y };
        i += 1;
    }
    x
}

const fn sub_4(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut r = [0u64; 4];
    let mut borrow = 0u64;
    let mut j = 0;
    while j < 4 {
        let t = (a[j] as u128)
            .wrapping_sub(b[j] as u128)
            .wrapping_sub(borrow as u128);
        r[j] = t as u64;
        borrow = (t >> 127) as u64;
        j += 1;
    }
    r
}

const fn twice_p() -> [u64; 4] {
    let p = FQ_MOD;
    [
        p[0] << 1,
        p[1] << 1 | p[0] >> 63,
        p[2] << 1 | p[1] >> 63,
        p[3] << 1 | p[2] >> 63,
    ]
}

/// `-p^{-1} mod 2^52`, by Newton's iteration on the 2-adic inverse.
const fn k0() -> u64 {
    let p0 = FQ_MOD[0];
    let mut inv: u64 = 1;
    let mut i = 0;
    while i < 7 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg() & MASK
}

const P: [u64; 5] = to_52(FQ_MOD);
const P2: [u64; 5] = to_52(twice_p());
const K0: u64 = k0();
/// 1 in the 2^260 domain.
const ONE: [u64; 5] = to_52(pow2_mod_p(260));
/// Multiplying a 2^256-domain element by this lands it in the 2^260 domain.
const INTO: [u64; 5] = to_52(pow2_mod_p(264));
/// And this takes a 2^260-domain element back to the 2^256 domain.
const OUT_OF: [u64; 5] = to_52(pow2_mod_p(256));

/// Whether this machine runs the kernels. `is_x86_feature_detected!` caches the probe.
fn available() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512ifma")
}

// ---------------------------------------------------------------------------
// Lane-parallel field arithmetic. Every function carries the target features so
// the intrinsics inline; callers must have checked [`available`].
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn splat(c: &[u64; 5]) -> V {
    [
        _mm512_set1_epi64(c[0] as i64),
        _mm512_set1_epi64(c[1] as i64),
        _mm512_set1_epi64(c[2] as i64),
        _mm512_set1_epi64(c[3] as i64),
        _mm512_set1_epi64(c[4] as i64),
    ]
}

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn select(m: __mmask8, if_set: &V, otherwise: &V) -> V {
    let mut r = *otherwise;
    for j in 0..5 {
        r[j] = _mm512_mask_blend_epi64(m, otherwise[j], if_set[j]);
    }
    r
}

/// Montgomery product `a b / 2^260`. Inputs and output in `[0, 2p)`, limbs normalised.
/// The probe this is taken from checked it against a bignum reference on 221,920
/// products, edge values and 2000-step lazy chains included.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn mul(a: &V, b: &V) -> V {
    let z = _mm512_setzero_si512();
    let mask = _mm512_set1_epi64(MASK as i64);
    let k0 = _mm512_set1_epi64(K0 as i64);
    let p = splat(&P);
    let mut t = [z; 6];
    for i in 0..5 {
        let bi = b[i];
        for j in 0..5 {
            t[j] = _mm512_madd52lo_epu64(t[j], a[j], bi);
        }
        for j in 0..5 {
            t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], a[j], bi);
        }
        let m = _mm512_madd52lo_epu64(z, t[0], k0);
        for j in 0..5 {
            t[j] = _mm512_madd52lo_epu64(t[j], m, p[j]);
        }
        for j in 0..5 {
            t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], m, p[j]);
        }
        t[1] = _mm512_add_epi64(t[1], _mm512_srli_epi64::<52>(t[0]));
        t = [t[1], t[2], t[3], t[4], t[5], z];
    }
    for j in 0..4 {
        t[j + 1] = _mm512_add_epi64(t[j + 1], _mm512_srli_epi64::<52>(t[j]));
        t[j] = _mm512_and_si512(t[j], mask);
    }
    [t[0], t[1], t[2], t[3], t[4]]
}

/// `x - m` where that does not borrow, else `x`. Normalised limbs in and out.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn cond_sub(x: &V, m: &[u64; 5]) -> V {
    let mask = _mm512_set1_epi64(MASK as i64);
    let mut d = *x;
    let mut borrow = _mm512_setzero_si512();
    for j in 0..5 {
        let t = _mm512_sub_epi64(
            _mm512_sub_epi64(x[j], _mm512_set1_epi64(m[j] as i64)),
            borrow,
        );
        borrow = _mm512_srli_epi64::<63>(t);
        d[j] = _mm512_and_si512(t, mask);
    }
    let keep = _mm512_test_epi64_mask(borrow, borrow);
    select(keep, x, &d)
}

/// `a + b`, inputs and output in `[0, 2p)`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn add(a: &V, b: &V) -> V {
    let mask = _mm512_set1_epi64(MASK as i64);
    let mut s = [_mm512_setzero_si512(); 5];
    let mut carry = _mm512_setzero_si512();
    for j in 0..5 {
        let t = _mm512_add_epi64(_mm512_add_epi64(a[j], b[j]), carry);
        carry = _mm512_srli_epi64::<52>(t);
        s[j] = if j < 4 { _mm512_and_si512(t, mask) } else { t };
    }
    cond_sub(&s, &P2)
}

/// `a - b` as `a + 2p - b`, inputs and output in `[0, 2p)`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub(a: &V, b: &V) -> V {
    let mask = _mm512_set1_epi64(MASK as i64);
    let mut s = [_mm512_setzero_si512(); 5];
    let mut carry = _mm512_setzero_si512();
    for j in 0..5 {
        let t = _mm512_add_epi64(
            _mm512_sub_epi64(
                _mm512_add_epi64(a[j], _mm512_set1_epi64(P2[j] as i64)),
                b[j],
            ),
            carry,
        );
        // Arithmetic shift: a limb may go negative before the carry settles it.
        carry = _mm512_srai_epi64::<52>(t);
        s[j] = if j < 4 { _mm512_and_si512(t, mask) } else { t };
    }
    cond_sub(&s, &P2)
}

/// Lanes holding a multiple of `p`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn is_zero(x: &V) -> __mmask8 {
    let c = cond_sub(x, &P);
    let or = _mm512_or_si512(
        _mm512_or_si512(c[0], c[1]),
        _mm512_or_si512(_mm512_or_si512(c[2], c[3]), c[4]),
    );
    _mm512_cmpeq_epi64_mask(or, _mm512_setzero_si512())
}

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn load_lanes(cols: &[[u64; 8]; 5]) -> V {
    let mut v = [_mm512_setzero_si512(); 5];
    for j in 0..5 {
        // SAFETY: 8 u64 is 64 bytes, and the load is unaligned.
        v[j] = unsafe { _mm512_loadu_si512(cols[j].as_ptr() as *const _) };
    }
    v
}

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn store_lanes(v: &V) -> [[u64; 8]; 5] {
    let mut cols = [[0u64; 8]; 5];
    for j in 0..5 {
        // SAFETY: as in `load_lanes`.
        unsafe { _mm512_storeu_si512(cols[j].as_mut_ptr() as *mut _, v[j]) };
    }
    cols
}

/// Limbs `j..j+5` of eight rows of `base`, lane `l` at element offset `off[l]`.
///
/// # Safety
/// Every unmasked lane's `off + j + 4` must be in bounds of the allocation `base` points into.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn gather(base: *const u64, off: __m512i, m: __mmask8, j: usize) -> V {
    let z = _mm512_setzero_si512();
    let p = base.add(j) as *const i64;
    [
        _mm512_mask_i64gather_epi64::<8>(z, m, off, p),
        _mm512_mask_i64gather_epi64::<8>(z, m, off, p.add(1)),
        _mm512_mask_i64gather_epi64::<8>(z, m, off, p.add(2)),
        _mm512_mask_i64gather_epi64::<8>(z, m, off, p.add(3)),
        _mm512_mask_i64gather_epi64::<8>(z, m, off, p.add(4)),
    ]
}

/// # Safety
/// As [`gather`], for writes; unmasked lanes must not repeat an offset.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn scatter(base: *mut u64, off: __m512i, m: __mmask8, j: usize, v: &V) {
    let p = base.add(j) as *mut i64;
    for (l, vl) in v.iter().enumerate() {
        _mm512_mask_i64scatter_epi64::<8>(p.add(l), m, off, *vl);
    }
}

/// Eight `u32` indices times [`ROW`], as 64-bit element offsets.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn row_offsets(idx: &[u32; 8]) -> __m512i {
    // SAFETY: 8 u32 is 32 bytes, and the load is unaligned.
    let i = _mm512_cvtepu32_epi64(unsafe { _mm256_loadu_si256(idx.as_ptr() as *const _) });
    // 10 i = 8 i + 2 i; vpmullq would need AVX-512DQ for nothing.
    _mm512_add_epi64(_mm512_slli_epi64::<3>(i), _mm512_slli_epi64::<1>(i))
}

// ---------------------------------------------------------------------------
// Per-MSM base conversion
// ---------------------------------------------------------------------------

/// The scanned bases in fill order, `ROW` limbs each: `rows[10k..10k + 10]` is
/// `bases[scan.idx[k]]` in the 2^260 domain. Built once per MSM and read by every window,
/// sequentially within a chunk.
pub(crate) struct Bases {
    rows: Vec<u64>,
}

/// Converts `idx.len()` bases into `out`.
#[target_feature(enable = "avx512f,avx512ifma")]
fn convert_bases(bases: &[G1Affine], idx: &[u32], out: &mut [u64]) {
    let into = splat(&INTO);
    for (g, group) in idx.chunks(8).enumerate() {
        let mut xs = [[0u64; 8]; 5];
        let mut ys = [[0u64; 8]; 5];
        for (l, &i) in group.iter().enumerate() {
            let (x, y) = G1::raw_xy(&bases[i as usize]);
            let (x, y) = (to_52(x.0), to_52(y.0));
            for j in 0..5 {
                xs[j][l] = x[j];
                ys[j][l] = y[j];
            }
        }
        let xs = store_lanes(&mul(&load_lanes(&xs), &into));
        let ys = store_lanes(&mul(&load_lanes(&ys), &into));
        for l in 0..group.len() {
            let row = &mut out[(8 * g + l) * ROW..][..ROW];
            for j in 0..5 {
                row[j] = xs[j][l];
                row[5 + j] = ys[j][l];
            }
        }
    }
}

/// Entries per parallel conversion task.
const CONVERT_CHUNK: usize = 1 << 12;

impl FastFill for G1 {
    type Prepared = Bases;

    fn prepare(bases: &[G1Affine], scan: &Prescan<G1>) -> Option<Bases> {
        if !available() {
            return None;
        }
        // The pending list keeps the sign in bit 31 of the scan index.
        assert!(scan.idx.len() < NEG as usize);
        let mut rows = vec![0u64; scan.idx.len() * ROW];
        rows.par_chunks_mut(CONVERT_CHUNK * ROW)
            .zip(scan.idx.par_chunks(CONVERT_CHUNK))
            // SAFETY: `available` checked the features above.
            .for_each(|(out, idx)| unsafe { convert_bases(bases, idx, out) });
        Some(Bases { rows })
    }

    fn window_chunk(
        prep: &Bases,
        bases: &[G1Affine],
        scan: &Prescan<G1>,
        range: core::ops::Range<usize>,
        window: usize,
        c: u32,
        work: Work,
    ) -> Projective<G1> {
        let n_buckets = 1usize << (c - 1);
        let mut fill = Fill::new(bases, &scan.idx, &prep.rows, n_buckets);
        // See `window_chunk_batch`.
        let mut dummy = Xyzz::<RawFq>::ZERO;
        for k in range {
            let d = signed_digit(scan.bigints[k].as_ref(), window, c);
            if d == 0 {
                if work == Work::Constant {
                    let (x, y) = G1::raw_xy(&bases[scan.idx[k] as usize]);
                    dummy.madd(x, y);
                }
                continue;
            }
            if d > 0 {
                fill.insert((d - 1) as usize, k as u32);
            } else {
                fill.insert((-d - 1) as usize, k as u32 | NEG);
            }
        }
        core::hint::black_box(&dummy);
        fill.finish()
    }
}

// ---------------------------------------------------------------------------
// The fill
// ---------------------------------------------------------------------------

/// [`BatchFill`](crate::BatchFill) for G1 with radix-2^52 buckets. Only built from a
/// [`Bases`], which only [`prepare`](FastFill::prepare) makes, after checking the CPU; that
/// is what makes the `unsafe` calls into the target-feature functions sound.
struct Fill<'a> {
    bases: &'a [G1Affine],
    idx: &'a [u32],
    rows: &'a [u64],
    /// `ROW` limbs per bucket, meaningful while `occupied`.
    buckets: Vec<u64>,
    occupied: Vec<bool>,
    busy: Vec<bool>,
    /// The current round: bucket, and scan index with [`NEG`] for a negated point.
    pend_b: Vec<u32>,
    pend_k: Vec<u32>,
    retry: Vec<(u32, u32)>,
    side: Option<SideBuckets<RawFq>>,
    /// Per group of eight: what the backward pass needs, [`SAVED`] vectors.
    saved: Vec<V>,
    cancels: Vec<u8>,
    batch: usize,
}

/// Prefix product, denominator, numerator, `bx + px`, `bx`, `by`.
const SAVED: usize = 6;

impl<'a> Fill<'a> {
    fn new(bases: &'a [G1Affine], idx: &'a [u32], rows: &'a [u64], n_buckets: usize) -> Self {
        let batch = batch_size(n_buckets);
        let cap = batch + batch / 2;
        Fill {
            bases,
            idx,
            rows,
            buckets: vec![0; n_buckets * ROW],
            occupied: vec![false; n_buckets],
            busy: vec![false; n_buckets],
            pend_b: Vec::with_capacity(cap),
            pend_k: Vec::with_capacity(cap),
            retry: Vec::new(),
            side: None,
            saved: Vec::with_capacity(cap.div_ceil(8) * SAVED),
            cancels: Vec::with_capacity(cap.div_ceil(8)),
            batch,
        }
    }

    /// An empty bucket takes the point as is.
    fn set(&mut self, b: usize, k: u32) {
        let src = &self.rows[(k & !NEG) as usize * ROW..][..ROW];
        let dst = &mut self.buckets[b * ROW..][..ROW];
        dst[..5].copy_from_slice(&src[..5]);
        if k & NEG == 0 {
            dst[5..].copy_from_slice(&src[5..]);
        } else {
            // 2p - y: y is in (0, 2p), never a multiple of p on a curve without
            // 2-torsion, so the result is in (0, 2p) as well.
            let mut borrow = 0i64;
            for j in 0..5 {
                let t = P2[j] as i64 - src[5 + j] as i64 - borrow;
                borrow = (t < 0) as i64;
                dst[5 + j] = t as u64 & MASK;
            }
        }
        self.occupied[b] = true;
    }

    /// See [`BatchFill::insert`](crate::BatchFill).
    fn insert(&mut self, b: usize, k: u32) {
        if !self.occupied[b] {
            self.set(b, k);
            return;
        }
        if self.busy[b] {
            self.retry.push((b as u32, k));
            if self.retry.len() >= self.batch {
                self.flush();
                self.reschedule();
            }
            return;
        }
        self.busy[b] = true;
        self.pend_b.push(b as u32);
        self.pend_k.push(k);
        if self.pend_b.len() >= self.batch {
            self.flush();
            self.reschedule();
        }
    }

    fn reschedule(&mut self) {
        while let Some((b, k)) = self.retry.pop() {
            let b = b as usize;
            if !self.occupied[b] {
                self.set(b, k);
            } else if self.busy[b] {
                let (x, y) = G1::raw_xy(&self.bases[self.idx[(k & !NEG) as usize] as usize]);
                let y = if k & NEG == 0 { y } else { y.neg() };
                self.side
                    .get_or_insert_with(|| SideBuckets::new(self.occupied.len()))
                    .entry(b)
                    .madd(x, y);
            } else {
                self.busy[b] = true;
                self.pend_b.push(b as u32);
                self.pend_k.push(k);
            }
        }
    }

    fn flush(&mut self) {
        // SAFETY: see the type's doc.
        unsafe { self.flush_ifma() }
    }

    /// [`BatchFill::flush`](crate::BatchFill), eight entries a vector.
    #[target_feature(enable = "avx512f,avx512ifma")]
    fn flush_ifma(&mut self) {
        let n = self.pend_b.len();
        if n == 0 {
            return;
        }
        let groups = n.div_ceil(8);
        self.saved.clear();
        self.cancels.clear();
        let one = splat(&ONE);
        let zero = [_mm512_setzero_si512(); 5];
        let mut acc = one;
        for g in 0..groups {
            let (b_off, k_off, valid, neg) = self.group(g);
            // SAFETY: `group` offsets address whole rows of `buckets` and `rows`, and
            // masks off the lanes past the end of the round.
            let (bx, by, px, py) = unsafe {
                let bk = self.buckets.as_ptr();
                let pt = self.rows.as_ptr();
                (
                    gather(bk, b_off, valid, 0),
                    gather(bk, b_off, valid, 5),
                    gather(pt, k_off, valid, 0),
                    gather(pt, k_off, valid, 5),
                )
            };
            let py = select(neg, &sub(&zero, &py), &py);
            let dx = sub(&px, &bx);
            let mut num = sub(&py, &by);
            let mut den = select(valid, &dx, &one);
            let same_x = is_zero(&dx) & valid;
            let mut cancel = 0;
            if same_x != 0 {
                // Rare: the scalar path's `Op::Dbl` and `Op::Cancel`, as lane masks.
                let dbl = same_x & is_zero(&num);
                cancel = same_x & !dbl;
                if dbl != 0 {
                    let xx = mul(&bx, &bx);
                    num = select(dbl, &add(&add(&xx, &xx), &xx), &num);
                    den = select(dbl, &add(&by, &by), &den);
                }
                den = select(cancel, &one, &den);
            }
            self.saved
                .extend_from_slice(&[acc, den, num, add(&bx, &px), bx, by]);
            self.cancels.push(cancel);
            acc = mul(&acc, &den);
        }

        let mut inv = invert_lanes(&acc);

        for g in (0..groups).rev() {
            let s = &self.saved[g * SAVED..][..SAVED];
            let d_inv = mul(&inv, &s[0]);
            inv = mul(&inv, &s[1]);
            let lambda = mul(&s[2], &d_inv);
            let x3 = sub(&mul(&lambda, &lambda), &s[3]);
            let y3 = sub(&mul(&lambda, &sub(&s[4], &x3)), &s[5]);
            let (b_off, _, valid, _) = self.group(g);
            let cancel = self.cancels[g];
            // SAFETY: as for the gathers; a round holds each bucket at most once, so the
            // unmasked offsets are distinct.
            unsafe {
                let bk = self.buckets.as_mut_ptr();
                scatter(bk, b_off, valid & !cancel, 0, &x3);
                scatter(bk, b_off, valid & !cancel, 5, &y3);
            }
            let lo = 8 * g;
            for l in 0..(n - lo).min(8) {
                let b = self.pend_b[lo + l] as usize;
                self.busy[b] = false;
                if cancel >> l & 1 != 0 {
                    self.occupied[b] = false;
                }
            }
        }
        self.pend_b.clear();
        self.pend_k.clear();
    }

    /// Bucket and point row offsets, live lanes and negated lanes of group `g`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    fn group(&self, g: usize) -> (__m512i, __m512i, __mmask8, __mmask8) {
        let lo = 8 * g;
        let len = (self.pend_b.len() - lo).min(8);
        let mut b = [0u32; 8];
        let mut k = [0u32; 8];
        let mut neg = 0u8;
        for l in 0..len {
            b[l] = self.pend_b[lo + l];
            let kl = self.pend_k[lo + l];
            k[l] = kl & !NEG;
            neg |= ((kl >> 31) as u8) << l;
        }
        let valid = (((1u16 << len) - 1) & 0xff) as u8;
        (row_offsets(&b), row_offsets(&k), valid, neg)
    }

    /// Flush the stragglers, convert the buckets back in one pass, then the scalar
    /// running sum.
    fn finish(mut self) -> Projective<G1> {
        self.flush();
        self.reschedule();
        self.flush();
        let n_buckets = self.occupied.len();
        let mut bx = vec![RawFq::ZERO; n_buckets];
        let mut by = vec![RawFq::ZERO; n_buckets];
        // SAFETY: see the type's doc.
        unsafe { export(&self.buckets, &mut bx, &mut by) };
        to_projective(&affine_running_sum(
            &bx,
            &by,
            &self.occupied,
            self.side.as_ref(),
        ))
    }
}

/// `x^{-1}` per lane, all lanes nonzero, through one scalar inversion.
#[target_feature(enable = "avx512f,avx512ifma")]
fn invert_lanes(x: &V) -> V {
    let cols = store_lanes(&cond_sub(x, &P));
    // A 2^260-domain integer `a = v 2^260` read as an ark element is `v 2^4`. Montgomery's
    // trick over the eight, then a factor 2^8 turns `v^-1 2^-4` into `v^-1 2^4`, which
    // is `v^-1` in the 2^260 domain.
    let mut f = [Fq::ZERO; 8];
    for (l, fl) in f.iter_mut().enumerate() {
        let limbs = [cols[0][l], cols[1][l], cols[2][l], cols[3][l], cols[4][l]];
        *fl = Fq::new_unchecked(BigInt::new(from_52(limbs)));
    }
    let mut prefix = [Fq::ONE; 8];
    let mut acc = Fq::ONE;
    for l in 0..8 {
        prefix[l] = acc;
        acc *= f[l];
    }
    let mut inv = acc.inverse().expect("inverse of zero") * Fq::from(256u64);
    let mut out = [[0u64; 8]; 5];
    for l in (0..8).rev() {
        let il = inv * prefix[l];
        inv *= f[l];
        let limbs = to_52(il.0 .0);
        for j in 0..5 {
            out[j][l] = limbs[j];
        }
    }
    load_lanes(&out)
}

/// Every bucket row back to canonical 2^256-domain `RawFq`, eight buckets a vector.
#[target_feature(enable = "avx512f,avx512ifma")]
fn export(buckets: &[u64], bx: &mut [RawFq], by: &mut [RawFq]) {
    let out_of = splat(&OUT_OF);
    let n = bx.len();
    for lo in (0..n).step_by(8) {
        let len = (n - lo).min(8);
        let mut idx = [0u32; 8];
        for (l, i) in idx.iter_mut().enumerate().take(len) {
            *i = (lo + l) as u32;
        }
        let off = row_offsets(&idx);
        let valid = (((1u16 << len) - 1) & 0xff) as u8;
        // SAFETY: lanes below `len` address rows of `buckets`, the rest are masked.
        let (x, y) = unsafe {
            (
                gather(buckets.as_ptr(), off, valid, 0),
                gather(buckets.as_ptr(), off, valid, 5),
            )
        };
        let x = store_lanes(&cond_sub(&mul(&x, &out_of), &P));
        let y = store_lanes(&cond_sub(&mul(&y, &out_of), &P));
        for l in 0..len {
            bx[lo + l] = RawFq(from_52([x[0][l], x[1][l], x[2][l], x[3][l], x[4][l]]));
            by[lo + l] = RawFq(from_52([y[0][l], y[1][l], y[2][l], y[3][l], y[4][l]]));
        }
    }
}

#[cfg(test)]
mod tests;
