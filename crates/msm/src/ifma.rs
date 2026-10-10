//! The batch-affine fill on AVX-512 IFMA, eight entries per vector, for both groups.
//!
//! Chosen at run time by [`prepare`], so a generic x86_64 build carries it and a machine
//! without the extension never executes an instruction of it. The scheduling (busy map,
//! retries, side buckets) is [`BatchFill`](crate::BatchFill)'s, line for line; only the
//! arithmetic and the bucket storage differ.
//!
//! The base field is BN254 Fq in radix 2^52, five limbs per element, one element per
//! 64-bit lane (G2's Fq2 is two such vectors, Karatsuba as in `RawFq2`), Montgomery with R = 2^260 against the scalar path's 2^256. Values are lazy, in
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

use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ff::{AdditiveGroup, BigInt, Field};
use rayon::prelude::*;
use snarkrs_field::raw::{RawField, RawFq, RawFq2, FQ_MOD};
use snarkrs_field::{Fq, Fq2};

use crate::xyzz::{to_projective, RawCurve, Xyzz};
use crate::{affine_running_sum, batch_size, signed_digit, FastFill, Prescan, SideBuckets, Work};

type G1 = snarkrs_field::g1::Config;
type G2 = snarkrs_field::g2::Config;
type V = [__m512i; 5];

const MASK: u64 = (1 << 52) - 1;
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

/// Eight `u32` indices times `row`, as 64-bit element offsets.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn row_offsets(idx: &[u32; 8], row: usize) -> __m512i {
    let mut off = [0u64; 8];
    for (o, &i) in off.iter_mut().zip(idx) {
        *o = i as u64 * row as u64;
    }
    // SAFETY: 8 u64 is 64 bytes, and the load is unaligned.
    unsafe { _mm512_loadu_si512(off.as_ptr() as *const _) }
}

// ---------------------------------------------------------------------------
// The two coordinate fields as lane vectors
// ---------------------------------------------------------------------------

/// Fq2 as two lane vectors, `c0 + c1 u` with `u^2 = -1`.
#[derive(Clone, Copy)]
struct V2(V, V);

/// The coordinate field of a group, eight elements a vector. Every method needs the
/// target features: callers must have checked [`available`].
trait Lanes: Copy {
    /// u64 limbs per element, 5 for Fq and 10 for Fq2.
    const LIMBS: usize;
    type Raw: RawField;
    unsafe fn one() -> Self;
    unsafe fn mul(&self, o: &Self) -> Self;
    unsafe fn add(&self, o: &Self) -> Self;
    unsafe fn sub(&self, o: &Self) -> Self;
    unsafe fn is_zero(&self) -> __mmask8;
    unsafe fn select(m: __mmask8, if_set: &Self, otherwise: &Self) -> Self;
    /// Limbs `j..j + LIMBS` of eight rows, see [`gather`].
    unsafe fn gather(base: *const u64, off: __m512i, m: __mmask8, j: usize) -> Self;
    unsafe fn scatter(&self, base: *mut u64, off: __m512i, m: __mmask8, j: usize);
    /// Eight canonical 2^256-domain elements into the 2^260 domain.
    unsafe fn enter(xs: &[Self::Raw; 8]) -> Self;
    /// And back out, canonical.
    unsafe fn leave(&self) -> [Self::Raw; 8];
    /// Every lane's inverse, all lanes nonzero, through one scalar inversion.
    unsafe fn invert(&self) -> Self;
}

/// Reads a 2^260-domain integer `a = v 2^260` as an ark element, which is `v 2^4`.
fn lane_fq(cols: &[[u64; 8]; 5], l: usize) -> Fq {
    let limbs = [cols[0][l], cols[1][l], cols[2][l], cols[3][l], cols[4][l]];
    Fq::new_unchecked(BigInt::new(from_52(limbs)))
}

/// The inverse of [`lane_fq`]: a canonical ark element's limbs, as a lane.
fn put_lane(cols: &mut [[u64; 8]; 5], l: usize, x: &Fq) {
    let limbs = to_52(x.0 .0);
    for j in 0..5 {
        cols[j][l] = limbs[j];
    }
}

/// Montgomery's trick over eight ark elements of `v 2^4`, then a factor 2^8 turns
/// `v^-1 2^-4` into `v^-1 2^4`, which is `v^-1` in the 2^260 domain.
fn invert8<F: Field + From<u64>>(f: &[F; 8]) -> [F; 8] {
    let mut prefix = [F::ONE; 8];
    let mut acc = F::ONE;
    for l in 0..8 {
        prefix[l] = acc;
        acc *= f[l];
    }
    let mut inv = acc.inverse().expect("inverse of zero") * F::from(256u64);
    let mut out = [F::ONE; 8];
    for l in (0..8).rev() {
        out[l] = inv * prefix[l];
        inv *= f[l];
    }
    out
}

impl Lanes for V {
    const LIMBS: usize = 5;
    type Raw = RawFq;
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn one() -> Self {
        splat(&ONE)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn mul(&self, o: &Self) -> Self {
        mul(self, o)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn add(&self, o: &Self) -> Self {
        add(self, o)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn sub(&self, o: &Self) -> Self {
        sub(self, o)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn is_zero(&self) -> __mmask8 {
        is_zero(self)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn select(m: __mmask8, if_set: &Self, otherwise: &Self) -> Self {
        select(m, if_set, otherwise)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn gather(base: *const u64, off: __m512i, m: __mmask8, j: usize) -> Self {
        gather(base, off, m, j)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn scatter(&self, base: *mut u64, off: __m512i, m: __mmask8, j: usize) {
        scatter(base, off, m, j, self)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn enter(xs: &[RawFq; 8]) -> Self {
        let mut cols = [[0u64; 8]; 5];
        for (l, x) in xs.iter().enumerate() {
            let limbs = to_52(x.0);
            for j in 0..5 {
                cols[j][l] = limbs[j];
            }
        }
        mul(&load_lanes(&cols), &splat(&INTO))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn leave(&self) -> [RawFq; 8] {
        let cols = store_lanes(&cond_sub(&mul(self, &splat(&OUT_OF)), &P));
        let mut out = [RawFq::ZERO; 8];
        for (l, o) in out.iter_mut().enumerate() {
            *o = RawFq(from_52([
                cols[0][l], cols[1][l], cols[2][l], cols[3][l], cols[4][l],
            ]));
        }
        out
    }
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn invert(&self) -> Self {
        let cols = store_lanes(&cond_sub(self, &P));
        let mut f = [Fq::ZERO; 8];
        for (l, fl) in f.iter_mut().enumerate() {
            *fl = lane_fq(&cols, l);
        }
        let inv = invert8(&f);
        let mut out = [[0u64; 8]; 5];
        for (l, il) in inv.iter().enumerate() {
            put_lane(&mut out, l, il);
        }
        load_lanes(&out)
    }
}

impl Lanes for V2 {
    const LIMBS: usize = 10;
    type Raw = RawFq2;
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn one() -> Self {
        V2(splat(&ONE), [_mm512_setzero_si512(); 5])
    }
    /// Karatsuba, three products, as `RawFq2::mul`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn mul(&self, o: &Self) -> Self {
        let v0 = mul(&self.0, &o.0);
        let v1 = mul(&self.1, &o.1);
        let s = mul(&add(&self.0, &self.1), &add(&o.0, &o.1));
        V2(sub(&v0, &v1), sub(&sub(&s, &v0), &v1))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn add(&self, o: &Self) -> Self {
        V2(add(&self.0, &o.0), add(&self.1, &o.1))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn sub(&self, o: &Self) -> Self {
        V2(sub(&self.0, &o.0), sub(&self.1, &o.1))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn is_zero(&self) -> __mmask8 {
        is_zero(&self.0) & is_zero(&self.1)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn select(m: __mmask8, if_set: &Self, otherwise: &Self) -> Self {
        V2(
            select(m, &if_set.0, &otherwise.0),
            select(m, &if_set.1, &otherwise.1),
        )
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn gather(base: *const u64, off: __m512i, m: __mmask8, j: usize) -> Self {
        V2(gather(base, off, m, j), gather(base, off, m, j + 5))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn scatter(&self, base: *mut u64, off: __m512i, m: __mmask8, j: usize) {
        scatter(base, off, m, j, &self.0);
        scatter(base, off, m, j + 5, &self.1);
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn enter(xs: &[RawFq2; 8]) -> Self {
        V2(V::enter(&xs.map(|x| x.c0)), V::enter(&xs.map(|x| x.c1)))
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn leave(&self) -> [RawFq2; 8] {
        let (c0, c1) = (self.0.leave(), self.1.leave());
        let mut out = [RawFq2::ZERO; 8];
        for (l, o) in out.iter_mut().enumerate() {
            *o = RawFq2 {
                c0: c0[l],
                c1: c1[l],
            };
        }
        out
    }
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn invert(&self) -> Self {
        let (c0, c1) = (
            store_lanes(&cond_sub(&self.0, &P)),
            store_lanes(&cond_sub(&self.1, &P)),
        );
        // Both components carry the same 2^4, so the element is `v 2^4` as for Fq.
        let mut f = [Fq2::ZERO; 8];
        for (l, fl) in f.iter_mut().enumerate() {
            *fl = Fq2::new(lane_fq(&c0, l), lane_fq(&c1, l));
        }
        let inv = invert8(&f);
        let (mut o0, mut o1) = ([[0u64; 8]; 5], [[0u64; 8]; 5]);
        for (l, il) in inv.iter().enumerate() {
            put_lane(&mut o0, l, &il.c0);
            put_lane(&mut o1, l, &il.c1);
        }
        V2(load_lanes(&o0), load_lanes(&o1))
    }
}

/// The group-side glue: which lane field a group's coordinates live in.
trait IfmaCurve: RawCurve {
    type L: Lanes<Raw = Self::RF>;
}

impl IfmaCurve for G1 {
    type L = V;
}

impl IfmaCurve for G2 {
    type L = V2;
}

// ---------------------------------------------------------------------------
// Per-MSM base conversion
// ---------------------------------------------------------------------------

/// The scanned bases in fill order, `2 LIMBS` u64 each: row `k` is `bases[scan.idx[k]]`
/// in the 2^260 domain, x then y. Built once per MSM and read by every window,
/// sequentially within a chunk.
pub(crate) struct Bases {
    rows: Vec<u64>,
}

/// Converts `idx.len()` bases into `out`.
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn convert_bases<P: IfmaCurve>(bases: &[Affine<P>], idx: &[u32], out: &mut [u64]) {
    let w = P::L::LIMBS;
    let rows = out.as_mut_ptr();
    for (g, group) in idx.chunks(8).enumerate() {
        let mut xs = [P::RF::ZERO; 8];
        let mut ys = [P::RF::ZERO; 8];
        let mut lanes = [0u32; 8];
        for (l, &i) in group.iter().enumerate() {
            (xs[l], ys[l]) = P::raw_xy(&bases[i as usize]);
            lanes[l] = (8 * g + l) as u32;
        }
        let valid = (((1u16 << group.len()) - 1) & 0xff) as u8;
        let off = row_offsets(&lanes, 2 * w);
        P::L::enter(&xs).scatter(rows, off, valid, 0);
        P::L::enter(&ys).scatter(rows, off, valid, w);
    }
}

/// Entries per parallel conversion task.
const CONVERT_CHUNK: usize = 1 << 12;

fn prepare<P: IfmaCurve>(bases: &[Affine<P>], scan: &Prescan<P>) -> Option<Bases>
where
    P::BaseField: Send + Sync,
{
    if !available() {
        return None;
    }
    // The pending list keeps the sign in bit 31 of the scan index.
    assert!(scan.idx.len() < NEG as usize);
    let row = 2 * P::L::LIMBS;
    let mut rows = vec![0u64; scan.idx.len() * row];
    rows.par_chunks_mut(CONVERT_CHUNK * row)
        .zip(scan.idx.par_chunks(CONVERT_CHUNK))
        // SAFETY: `available` checked the features above, and `out` holds a row per index.
        .for_each(|(out, idx)| unsafe { convert_bases::<P>(bases, idx, out) });
    Some(Bases { rows })
}

fn window_chunk<P: IfmaCurve>(
    prep: &Bases,
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    range: core::ops::Range<usize>,
    window: usize,
    c: u32,
    work: Work,
) -> Projective<P> {
    let n_buckets = 1usize << (c - 1);
    let mut fill = Fill::<P>::new(bases, &scan.idx, &prep.rows, n_buckets);
    // See `window_chunk_batch`.
    let mut dummy = Xyzz::<P::RF>::ZERO;
    for k in range {
        let d = signed_digit(scan.bigints[k].as_ref(), window, c);
        if d == 0 {
            if work == Work::Constant {
                let (x, y) = P::raw_xy(&bases[scan.idx[k] as usize]);
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

macro_rules! ifma_fast_fill {
    ($cfg:ty) => {
        impl FastFill for $cfg {
            type Prepared = Bases;

            fn prepare(bases: &[Affine<Self>], scan: &Prescan<Self>) -> Option<Bases> {
                prepare(bases, scan)
            }

            fn window_chunk(
                prep: &Bases,
                bases: &[Affine<Self>],
                scan: &Prescan<Self>,
                range: core::ops::Range<usize>,
                window: usize,
                c: u32,
                work: Work,
            ) -> Projective<Self> {
                window_chunk(prep, bases, scan, range, window, c, work)
            }
        }
    };
}

ifma_fast_fill!(G1);
ifma_fast_fill!(G2);

// ---------------------------------------------------------------------------
// The fill
// ---------------------------------------------------------------------------

/// [`BatchFill`](crate::BatchFill) with radix-2^52 buckets. Only built from a [`Bases`],
/// which only [`prepare`] makes, after checking the CPU; that is what makes the `unsafe`
/// calls into the target-feature functions sound.
struct Fill<'a, P: IfmaCurve> {
    bases: &'a [Affine<P>],
    idx: &'a [u32],
    rows: &'a [u64],
    /// `2 LIMBS` u64 per bucket, meaningful while `occupied`.
    buckets: Vec<u64>,
    occupied: Vec<bool>,
    busy: Vec<bool>,
    /// The current round: bucket, and scan index with [`NEG`] for a negated point.
    pend_b: Vec<u32>,
    pend_k: Vec<u32>,
    retry: Vec<(u32, u32)>,
    side: Option<SideBuckets<P::RF>>,
    /// Per group of eight: what the backward pass needs, [`SAVED`] vectors.
    saved: Vec<P::L>,
    cancels: Vec<u8>,
    batch: usize,
}

/// Prefix product, denominator, numerator, `bx + px`, `bx`, `by`.
const SAVED: usize = 6;

impl<'a, P: IfmaCurve> Fill<'a, P> {
    const W: usize = P::L::LIMBS;

    fn new(bases: &'a [Affine<P>], idx: &'a [u32], rows: &'a [u64], n_buckets: usize) -> Self {
        let batch = batch_size(n_buckets);
        let cap = batch + batch / 2;
        Fill {
            bases,
            idx,
            rows,
            buckets: vec![0; n_buckets * 2 * Self::W],
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
        let w = Self::W;
        let src = &self.rows[(k & !NEG) as usize * 2 * w..][..2 * w];
        let dst = &mut self.buckets[b * 2 * w..][..2 * w];
        dst[..w].copy_from_slice(&src[..w]);
        if k & NEG == 0 {
            dst[w..].copy_from_slice(&src[w..]);
        } else {
            // 2p - y per Fq component, and 0 for a zero component (an Fq2 y can have
            // one), so the result stays in [0, 2p).
            for (d, s) in dst[w..].chunks_exact_mut(5).zip(src[w..].chunks_exact(5)) {
                if s.iter().all(|&l| l == 0) {
                    d.fill(0);
                    continue;
                }
                let mut borrow = 0i64;
                for j in 0..5 {
                    let t = P2[j] as i64 - s[j] as i64 - borrow;
                    borrow = (t < 0) as i64;
                    d[j] = t as u64 & MASK;
                }
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
                let (x, y) = P::raw_xy(&self.bases[self.idx[(k & !NEG) as usize] as usize]);
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
    unsafe fn flush_ifma(&mut self) {
        let n = self.pend_b.len();
        if n == 0 {
            return;
        }
        let w = Self::W;
        let groups = n.div_ceil(8);
        self.saved.clear();
        self.cancels.clear();
        let one = P::L::one();
        let mut acc = one;
        for g in 0..groups {
            let (b_off, k_off, valid, neg) = self.group(g);
            // `group` offsets address whole rows of `buckets` and `rows`, and masks off the
            // lanes past the end of the round.
            let bk = self.buckets.as_ptr();
            let pt = self.rows.as_ptr();
            let bx = P::L::gather(bk, b_off, valid, 0);
            let by = P::L::gather(bk, b_off, valid, w);
            let px = P::L::gather(pt, k_off, valid, 0);
            let py = P::L::gather(pt, k_off, valid, w);
            let zero = one.sub(&one);
            let py = P::L::select(neg, &zero.sub(&py), &py);
            let dx = px.sub(&bx);
            let mut num = py.sub(&by);
            let mut den = P::L::select(valid, &dx, &one);
            let same_x = dx.is_zero() & valid;
            let mut cancel = 0;
            if same_x != 0 {
                // Rare: the scalar path's `Op::Dbl` and `Op::Cancel`, as lane masks.
                let dbl = same_x & num.is_zero();
                cancel = same_x & !dbl;
                if dbl != 0 {
                    let xx = bx.mul(&bx);
                    num = P::L::select(dbl, &xx.add(&xx).add(&xx), &num);
                    den = P::L::select(dbl, &by.add(&by), &den);
                }
                den = P::L::select(cancel, &one, &den);
            }
            self.saved
                .extend_from_slice(&[acc, den, num, bx.add(&px), bx, by]);
            self.cancels.push(cancel);
            acc = acc.mul(&den);
        }

        let mut inv = acc.invert();

        for g in (0..groups).rev() {
            let s = &self.saved[g * SAVED..][..SAVED];
            let d_inv = inv.mul(&s[0]);
            inv = inv.mul(&s[1]);
            let lambda = s[2].mul(&d_inv);
            let x3 = lambda.mul(&lambda).sub(&s[3]);
            let y3 = lambda.mul(&s[4].sub(&x3)).sub(&s[5]);
            let (b_off, _, valid, _) = self.group(g);
            let cancel = self.cancels[g];
            // A round holds each bucket at most once, so the unmasked offsets are distinct.
            let bk = self.buckets.as_mut_ptr();
            x3.scatter(bk, b_off, valid & !cancel, 0);
            y3.scatter(bk, b_off, valid & !cancel, w);
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
        let row = 2 * Self::W;
        (row_offsets(&b, row), row_offsets(&k, row), valid, neg)
    }

    /// Flush the stragglers, convert the buckets back in one pass, then the scalar
    /// running sum.
    fn finish(mut self) -> Projective<P> {
        self.flush();
        self.reschedule();
        self.flush();
        let n_buckets = self.occupied.len();
        let mut bx = vec![P::RF::ZERO; n_buckets];
        let mut by = vec![P::RF::ZERO; n_buckets];
        // SAFETY: see the type's doc.
        unsafe { export::<P::L>(&self.buckets, &mut bx, &mut by) };
        to_projective(&affine_running_sum(
            &bx,
            &by,
            &self.occupied,
            self.side.as_ref(),
        ))
    }
}

/// Every bucket row back to canonical 2^256-domain raw limbs, eight buckets a vector.
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn export<L: Lanes>(buckets: &[u64], bx: &mut [L::Raw], by: &mut [L::Raw]) {
    let w = L::LIMBS;
    let n = bx.len();
    for lo in (0..n).step_by(8) {
        let len = (n - lo).min(8);
        let mut idx = [0u32; 8];
        for (l, i) in idx.iter_mut().enumerate().take(len) {
            *i = (lo + l) as u32;
        }
        let off = row_offsets(&idx, 2 * w);
        let valid = (((1u16 << len) - 1) & 0xff) as u8;
        // Lanes below `len` address rows of `buckets`, the rest are masked.
        let x = L::gather(buckets.as_ptr(), off, valid, 0).leave();
        let y = L::gather(buckets.as_ptr(), off, valid, w).leave();
        bx[lo..lo + len].copy_from_slice(&x[..len]);
        by[lo..lo + len].copy_from_slice(&y[..len]);
    }
}

#[cfg(test)]
mod tests;
