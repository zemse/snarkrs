//! The NTT butterflies on AVX-512 IFMA, eight `Fr` elements per vector.
//!
//! Chosen at run time ([`available`]), so a generic x86_64 build carries it and a machine
//! without the extension never executes an instruction of it. Every kernel here is the
//! vector twin of a scalar kernel in `lib.rs` and computes the same canonical result to
//! the bit: field arithmetic is exact, so only the representation in between differs.
//!
//! The representation is `snarkrs-msm`'s IFMA one moved to the scalar field: radix 2^52,
//! five limbs an element, one element a 64-bit lane, values lazy in `[0, 2r)` (`4r <
//! 2^260`). The data never changes Montgomery domain. An `Fr` holds `x 2^256 mod r`, and
//! every multiply here has a twiddle (or coset factor) on one side, stored as `w 2^260`, so
//! a product with `R' = 2^260` is `x w 2^256`: still an ordinary `Fr`. Only the limb split
//! changes on the way in and out, which is shifts and permutes, plus one conditional
//! subtraction on the way out to land back in `[0, r)`.
//!
//! Two layouts are used for a cache block held in registers' worth of scratch:
//!
//!   * consecutive: vector `g` holds elements `8g .. 8g + 8`, so a pass at `half >= 8`
//!     pairs whole vectors and is lane parallel;
//!   * transposed, per 64-element run `G`: vector `i` holds elements `64G + 8l + i` in lane
//!     `l`, so the passes at `half` 1, 2 and 4 pair whole vectors too.
//!
//! An 8x8 transpose per limb moves between the two, once per block.

#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::*;

use snarkrs_field::raw::FR_MOD;
use snarkrs_field::Fr;

/// Five radix-2^52 limbs, eight lanes.
pub(crate) type V = [__m512i; 5];

const MASK: u64 = (1 << 52) - 1;

const fn to_52(x: [u64; 4]) -> [u64; 5] {
    [
        x[0] & MASK,
        (x[0] >> 52 | x[1] << 12) & MASK,
        (x[1] >> 40 | x[2] << 24) & MASK,
        (x[2] >> 28 | x[3] << 36) & MASK,
        x[3] >> 16,
    ]
}

const fn twice(p: [u64; 4]) -> [u64; 4] {
    [
        p[0] << 1,
        p[1] << 1 | p[0] >> 63,
        p[2] << 1 | p[1] >> 63,
        p[3] << 1 | p[2] >> 63,
    ]
}

/// `-r^{-1} mod 2^52`, by Newton's iteration on the 2-adic inverse.
const fn k0() -> u64 {
    let p0 = FR_MOD[0];
    let mut inv: u64 = 1;
    let mut i = 0;
    while i < 7 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg() & MASK
}

const P: [u64; 5] = to_52(FR_MOD);
const P2: [u64; 5] = to_52(twice(FR_MOD));
const K0: u64 = k0();

/// Whether this machine runs the kernels. `is_x86_feature_detected!` caches the probe.
/// `G16_NTT_IFMA=0` keeps the scalar butterflies, for A/B timing in one binary.
pub(crate) fn available() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512ifma")
        && std::env::var_os("G16_NTT_IFMA").is_none_or(|v| v != "0")
}

/// An `Fr` as the `R' = 2^260` twiddle operand: `w 2^256 * 16 = w 2^260`, canonical.
fn twiddle_limbs(w: &Fr) -> [u64; 5] {
    to_52((*w * Fr::from(16u64)).0 .0)
}

/// Eight lanes of five limbs, laid out to load as a [`V`].
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct Lanes(pub [[u64; 8]; 5]);

/// A twiddle table in plain structure-of-arrays form, one `Vec` per limb, so any eight
/// consecutive entries load with one unaligned load per limb. Entry `i` is `w_m^i` for
/// `i` in `0 ..= m/2`, the same table as the scalar side's for a size-`m` transform.
pub(crate) struct Table {
    limbs: [Vec<u64>; 5],
    /// `m / 2`, the index of the last entry.
    top: usize,
}

impl Table {
    /// From `entries[i] = w_m^i`, `i` in `0 ..= m/2`.
    pub(crate) fn new(entries: impl ExactSizeIterator<Item = Fr>) -> Self {
        let len = entries.len();
        let mut limbs: [Vec<u64>; 5] = std::array::from_fn(|_| Vec::with_capacity(len));
        for w in entries {
            let l = twiddle_limbs(&w);
            for j in 0..5 {
                limbs[j].push(l[j]);
            }
        }
        Table {
            limbs,
            top: len - 1,
        }
    }

    /// `tw::<INV>(i + t)` in lane `t`.
    ///
    /// # Safety
    /// IFMA available; `i + 8 <= top` forward, `i + 7 < top` inverse.
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn load<const INV: bool>(&self, i: usize) -> V {
        let mut v = [_mm512_setzero_si512(); 5];
        if INV {
            debug_assert!(i + 7 < self.top);
            let b = self.top - i - 7;
            let rev = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            for j in 0..5 {
                let x = _mm512_loadu_si512(self.limbs[j].as_ptr().add(b) as *const _);
                v[j] = _mm512_permutexvar_epi64(rev, x);
            }
        } else {
            debug_assert!(i + 8 <= self.top + 1);
            for j in 0..5 {
                v[j] = _mm512_loadu_si512(self.limbs[j].as_ptr().add(i) as *const _);
            }
        }
        v
    }

    /// `tw::<INV>(i + 2t)` in lane `t`.
    ///
    /// # Safety
    /// IFMA available; `i + 16 <= top` forward, `i + 15 < top` inverse.
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn load_even<const INV: bool>(&self, i: usize) -> V {
        let mut v = [_mm512_setzero_si512(); 5];
        if INV {
            debug_assert!(i + 15 < self.top);
            let b = self.top - i - 15;
            let idx = _mm512_set_epi64(1, 3, 5, 7, 9, 11, 13, 15);
            for j in 0..5 {
                let p = self.limbs[j].as_ptr().add(b);
                let lo = _mm512_loadu_si512(p as *const _);
                let hi = _mm512_loadu_si512(p.add(8) as *const _);
                v[j] = _mm512_permutex2var_epi64(lo, idx, hi);
            }
        } else {
            debug_assert!(i + 16 <= self.top + 1);
            let idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
            for j in 0..5 {
                let p = self.limbs[j].as_ptr().add(i);
                let lo = _mm512_loadu_si512(p as *const _);
                let hi = _mm512_loadu_si512(p.add(8) as *const _);
                v[j] = _mm512_permutex2var_epi64(lo, idx, hi);
            }
        }
        v
    }
}

/// The in-block twiddles, packed per pass so the block kernels read them in order.
pub(crate) struct BlockTables {
    /// `levels[INV][k - 3][q]`: lanes `tw::<INV>(local, (8q + t) * B / 2^(k+1))` for the
    /// pass at `half = 2^k`, `k` in `3 .. log2(B)`.
    levels: [Vec<Vec<Lanes>>; 2],
    /// `tw::<INV>(local, s8 * {1, 2, 3})` for the radix-8 head and tail, splatted.
    eighth: [[Lanes; 3]; 2],
}

impl BlockTables {
    /// From the scalar side's size-`block` table `local`.
    pub(crate) fn new(local: &[Fr], block: usize) -> Self {
        let tw = |inv: bool, i: usize| -> [u64; 5] {
            twiddle_limbs(&local[if inv { local.len() - 1 - i } else { i }])
        };
        let lanes = |f: &dyn Fn(usize) -> [u64; 5]| {
            let mut out = Lanes([[0; 8]; 5]);
            for t in 0..8 {
                let l = f(t);
                for j in 0..5 {
                    out.0[j][t] = l[j];
                }
            }
            out
        };
        let log = block.trailing_zeros() as usize;
        let levels = [false, true].map(|inv| {
            (3..log)
                .map(|k| {
                    let h = 1usize << k;
                    let stride = block / (2 * h);
                    (0..h / 8)
                        .map(|q| lanes(&|t| tw(inv, (8 * q + t) * stride)))
                        .collect()
                })
                .collect()
        });
        let s8 = block / 8;
        let eighth = [false, true].map(|inv| [1, 2, 3].map(|m| lanes(&|_| tw(inv, m * s8))));
        BlockTables { levels, eighth }
    }
}

// ---------------------------------------------------------------------------
// Lane-parallel field arithmetic, `snarkrs-msm`'s IFMA kernels on the scalar field.
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn splat(c: &[u64; 5]) -> V {
    c.map(|x| _mm512_set1_epi64(x as i64))
}

#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn load_lanes(l: &Lanes) -> V {
    // SAFETY: `Lanes` is 64-byte aligned and each row is 64 bytes.
    std::array::from_fn(|j| unsafe { _mm512_load_si512(l.0[j].as_ptr() as *const _) })
}

/// Montgomery product `a b / 2^260`. Inputs and output in `[0, 2r)`, limbs normalised.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn mul(a: &V, b: &V) -> V {
    let z = _mm512_setzero_si512();
    let mask = _mm512_set1_epi64(MASK as i64);
    let k0 = _mm512_set1_epi64(K0 as i64);
    let p = splat(&P);
    let mut t = [z; 6];
    for &bi in b {
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
    let mut r = d;
    for j in 0..5 {
        r[j] = _mm512_mask_blend_epi64(keep, d[j], x[j]);
    }
    r
}

/// `a + b`, inputs and output in `[0, 2r)`.
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

/// `a - b` as `a + 2r - b`, inputs and output in `[0, 2r)`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub_raw(a: &V, b: &V) -> V {
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

/// The scalar side's `add_sub::<INV>`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn add_sub<const INV: bool>(a: &V, t: &V) -> (V, V) {
    if INV {
        (sub_raw(a, t), add(a, t))
    } else {
        (add(a, t), sub_raw(a, t))
    }
}

/// The scalar side's `sub::<INV>`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub<const INV: bool>(a: &V, b: &V) -> V {
    if INV {
        sub_raw(b, a)
    } else {
        sub_raw(a, b)
    }
}

// ---------------------------------------------------------------------------
// Moving between `[Fr]` and lanes.
// ---------------------------------------------------------------------------

/// Four 64-bit limb vectors (lane `t` = element `t`'s limb) into radix 2^52.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn split_52(l: [__m512i; 4]) -> V {
    let m = _mm512_set1_epi64(MASK as i64);
    [
        _mm512_and_si512(l[0], m),
        _mm512_and_si512(
            _mm512_or_si512(_mm512_srli_epi64::<52>(l[0]), _mm512_slli_epi64::<12>(l[1])),
            m,
        ),
        _mm512_and_si512(
            _mm512_or_si512(_mm512_srli_epi64::<40>(l[1]), _mm512_slli_epi64::<24>(l[2])),
            m,
        ),
        _mm512_and_si512(
            _mm512_or_si512(_mm512_srli_epi64::<28>(l[2]), _mm512_slli_epi64::<36>(l[3])),
            m,
        ),
        _mm512_srli_epi64::<16>(l[3]),
    ]
}

/// The inverse of [`split_52`], after reducing to `[0, r)`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn join_64(v: &V) -> [__m512i; 4] {
    let v = cond_sub(v, &P);
    [
        _mm512_or_si512(v[0], _mm512_slli_epi64::<52>(v[1])),
        _mm512_or_si512(_mm512_srli_epi64::<12>(v[1]), _mm512_slli_epi64::<40>(v[2])),
        _mm512_or_si512(_mm512_srli_epi64::<24>(v[2]), _mm512_slli_epi64::<28>(v[3])),
        _mm512_or_si512(_mm512_srli_epi64::<36>(v[3]), _mm512_slli_epi64::<16>(v[4])),
    ]
}

/// Four registers of two elements each (`r[k]` = elements `2k, 2k + 1`, limbs in order)
/// into four limb vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn deinterleave(r: [__m512i; 4]) -> [__m512i; 4] {
    let ia = _mm512_set_epi64(13, 9, 5, 1, 12, 8, 4, 0);
    let ib = _mm512_set_epi64(15, 11, 7, 3, 14, 10, 6, 2);
    let lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
    let hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
    let t0 = _mm512_permutex2var_epi64(r[0], ia, r[1]);
    let t1 = _mm512_permutex2var_epi64(r[0], ib, r[1]);
    let t2 = _mm512_permutex2var_epi64(r[2], ia, r[3]);
    let t3 = _mm512_permutex2var_epi64(r[2], ib, r[3]);
    [
        _mm512_permutex2var_epi64(t0, lo, t2),
        _mm512_permutex2var_epi64(t0, hi, t2),
        _mm512_permutex2var_epi64(t1, lo, t3),
        _mm512_permutex2var_epi64(t1, hi, t3),
    ]
}

/// The inverse of [`deinterleave`].
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn interleave(l: [__m512i; 4]) -> [__m512i; 4] {
    let ia = _mm512_set_epi64(13, 9, 5, 1, 12, 8, 4, 0);
    let ib = _mm512_set_epi64(15, 11, 7, 3, 14, 10, 6, 2);
    let lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
    let hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
    let t0 = _mm512_permutex2var_epi64(l[0], lo, l[1]);
    let t2 = _mm512_permutex2var_epi64(l[0], hi, l[1]);
    let t1 = _mm512_permutex2var_epi64(l[2], lo, l[3]);
    let t3 = _mm512_permutex2var_epi64(l[2], hi, l[3]);
    [
        _mm512_permutex2var_epi64(t0, ia, t1),
        _mm512_permutex2var_epi64(t0, ib, t1),
        _mm512_permutex2var_epi64(t2, ia, t3),
        _mm512_permutex2var_epi64(t2, ib, t3),
    ]
}

/// Eight consecutive elements at `p`.
///
/// # Safety
/// IFMA available, `p .. p + 8` readable, every element canonical.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn load8(p: *const Fr) -> V {
    let q = p as *const __m512i;
    split_52(deinterleave([
        _mm512_loadu_si512(q as *const _),
        _mm512_loadu_si512(q.add(1) as *const _),
        _mm512_loadu_si512(q.add(2) as *const _),
        _mm512_loadu_si512(q.add(3) as *const _),
    ]))
}

/// # Safety
/// IFMA available, `p .. p + 8` writable.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn store8(p: *mut Fr, v: &V) {
    let r = interleave(join_64(v));
    let q = p as *mut __m512i;
    for (k, x) in r.iter().enumerate() {
        _mm512_storeu_si512(q.add(k) as *mut _, *x);
    }
}

/// Elements `p[8t]`, `t` in `0..8`.
///
/// # Safety
/// IFMA available, `p[0 .. 57]` readable, every element canonical.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn load8_stride8(p: *const Fr) -> V {
    let e = |t: usize| _mm256_loadu_si256(p.add(8 * t) as *const __m256i);
    let pair = |t: usize| _mm512_inserti64x4::<1>(_mm512_castsi256_si512(e(t)), e(t + 1));
    split_52(deinterleave([pair(0), pair(2), pair(4), pair(6)]))
}

/// # Safety
/// IFMA available, `p[0 .. 57]` writable.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn store8_stride8(p: *mut Fr, v: &V) {
    let r = interleave(join_64(v));
    for (k, x) in r.iter().enumerate() {
        _mm256_storeu_si256(p.add(8 * 2 * k) as *mut __m256i, _mm512_castsi512_si256(*x));
        _mm256_storeu_si256(
            p.add(8 * (2 * k + 1)) as *mut __m256i,
            _mm512_extracti64x4_epi64::<1>(*x),
        );
    }
}

/// `out[j]` lane `i` = `a[i]` lane `j`, limb by limb.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn transpose8(a: &mut [V]) {
    debug_assert_eq!(a.len(), 8);
    let s_lo = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    let s_hi = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
    let q_lo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
    let q_hi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
    let o_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
    let o_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
    for j in 0..5 {
        let r: [__m512i; 8] = std::array::from_fn(|i| a[i][j]);
        // 2x2 blocks: s[2k] = [r_{2k}[0] r_{2k+1}[0] r_{2k}[2] r_{2k+1}[2] ...].
        let mut s = [_mm512_setzero_si512(); 8];
        for k in 0..4 {
            s[2 * k] = _mm512_permutex2var_epi64(r[2 * k], s_lo, r[2 * k + 1]);
            s[2 * k + 1] = _mm512_permutex2var_epi64(r[2 * k], s_hi, r[2 * k + 1]);
        }
        // 4x4: q[4h + c] holds column c (mod 4 pairs) for rows 4h .. 4h + 4.
        let mut q = [_mm512_setzero_si512(); 8];
        for h in 0..2 {
            let (e, o) = (s[4 * h], s[4 * h + 2]);
            let (e1, o1) = (s[4 * h + 1], s[4 * h + 3]);
            q[4 * h] = _mm512_permutex2var_epi64(e, q_lo, o); // cols 0, 4
            q[4 * h + 2] = _mm512_permutex2var_epi64(e, q_hi, o); // cols 2, 6
            q[4 * h + 1] = _mm512_permutex2var_epi64(e1, q_lo, o1); // cols 1, 5
            q[4 * h + 3] = _mm512_permutex2var_epi64(e1, q_hi, o1); // cols 3, 7
        }
        for c in 0..4 {
            a[c][j] = _mm512_permutex2var_epi64(q[c], o_lo, q[4 + c]);
            a[c + 4][j] = _mm512_permutex2var_epi64(q[c], o_hi, q[4 + c]);
        }
    }
}

// ---------------------------------------------------------------------------
// Block kernels: a CACHE_BLOCK of `Fr` through scratch lanes.
// ---------------------------------------------------------------------------

/// Scratch for one block, `block / 8` vectors.
pub(crate) fn scratch(block: usize) -> Vec<V> {
    // SAFETY: an all-zero bit pattern is a valid __m512i.
    vec![unsafe { core::mem::zeroed() }; block / 8]
}

/// Passes `half = 8 .. block/2` of a DIT, consecutive layout.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn dit_body<const INV: bool>(s: &mut [V], t: &BlockTables) {
    let levels = &t.levels[INV as usize];
    for (k, lv) in levels.iter().enumerate() {
        let hv = 1usize << k; // half / 8
        for b in s.chunks_exact_mut(2 * hv) {
            let (lo, hi) = b.split_at_mut(hv);
            for q in 0..hv {
                let w = load_lanes(&lv[q]);
                let (x, y) = add_sub::<INV>(&lo[q], &mul(&hi[q], &w));
                lo[q] = x;
                hi[q] = y;
            }
        }
    }
}

/// Passes `half = block/2 .. 8` of a DIF, consecutive layout.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn dif_body<const INV: bool>(s: &mut [V], t: &BlockTables) {
    let levels = &t.levels[INV as usize];
    for (k, lv) in levels.iter().enumerate().rev() {
        let hv = 1usize << k;
        for b in s.chunks_exact_mut(2 * hv) {
            let (lo, hi) = b.split_at_mut(hv);
            for q in 0..hv {
                let w = load_lanes(&lv[q]);
                let (a, c) = (lo[q], hi[q]);
                lo[q] = add(&a, &c);
                hi[q] = mul(&sub::<INV>(&a, &c), &w);
            }
        }
    }
}

/// The scalar `dit_head` on eight transposed vectors: run `l` in lane `l`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn head<const INV: bool>(x: &mut [V], t: &BlockTables) {
    let e = &t.eighth[INV as usize];
    let (w1, w2, w3) = (load_lanes(&e[0]), load_lanes(&e[1]), load_lanes(&e[2]));
    let (a0, a1) = (add(&x[0], &x[1]), sub_raw(&x[0], &x[1]));
    let (a2, a3) = (add(&x[2], &x[3]), sub_raw(&x[2], &x[3]));
    let (a4, a5) = (add(&x[4], &x[5]), sub_raw(&x[4], &x[5]));
    let (a6, a7) = (add(&x[6], &x[7]), sub_raw(&x[6], &x[7]));
    let (b0, b2) = (add(&a0, &a2), sub_raw(&a0, &a2));
    let (b1, b3) = add_sub::<INV>(&a1, &mul(&a3, &w2));
    let (b4, b6) = (add(&a4, &a6), sub_raw(&a4, &a6));
    let (b5, b7) = add_sub::<INV>(&a5, &mul(&a7, &w2));
    x[0] = add(&b0, &b4);
    x[4] = sub_raw(&b0, &b4);
    let (s, d) = add_sub::<INV>(&b1, &mul(&b5, &w1));
    x[1] = s;
    x[5] = d;
    let (s, d) = add_sub::<INV>(&b2, &mul(&b6, &w2));
    x[2] = s;
    x[6] = d;
    let (s, d) = add_sub::<INV>(&b3, &mul(&b7, &w3));
    x[3] = s;
    x[7] = d;
}

/// The scalar `dif_tail` on eight transposed vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn tail<const INV: bool>(x: &mut [V], t: &BlockTables) {
    let e = &t.eighth[INV as usize];
    let (w1, w2, w3) = (load_lanes(&e[0]), load_lanes(&e[1]), load_lanes(&e[2]));
    let (u0, u4) = (add(&x[0], &x[4]), sub_raw(&x[0], &x[4]));
    let (u1, u5) = (add(&x[1], &x[5]), mul(&sub::<INV>(&x[1], &x[5]), &w1));
    let (u2, u6) = (add(&x[2], &x[6]), mul(&sub::<INV>(&x[2], &x[6]), &w2));
    let (u3, u7) = (add(&x[3], &x[7]), mul(&sub::<INV>(&x[3], &x[7]), &w3));
    let (v0, v2) = (add(&u0, &u2), sub_raw(&u0, &u2));
    let (v1, v3) = (add(&u1, &u3), mul(&sub::<INV>(&u1, &u3), &w2));
    let (v4, v6) = (add(&u4, &u6), sub_raw(&u4, &u6));
    let (v5, v7) = (add(&u5, &u7), mul(&sub::<INV>(&u5, &u7), &w2));
    x[0] = add(&v0, &v1);
    x[1] = sub_raw(&v0, &v1);
    x[2] = add(&v2, &v3);
    x[3] = sub_raw(&v2, &v3);
    x[4] = add(&v4, &v5);
    x[5] = sub_raw(&v4, &v5);
    x[6] = add(&v6, &v7);
    x[7] = sub_raw(&v6, &v7);
}

/// Every DIT pass of one block, the scalar `block_dit`.
///
/// # Safety
/// IFMA available; `block.len() == 8 * s.len()`, a multiple of 64.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn block_dit<const INV: bool>(block: &mut [Fr], s: &mut [V], t: &BlockTables) {
    let p = block.as_mut_ptr();
    for (g, run) in s.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        for (i, v) in run.iter_mut().enumerate() {
            *v = load8_stride8(p.add(64 * g + i));
        }
        head::<INV>(run, t);
        transpose8(run);
    }
    dit_body::<INV>(s, t);
    for (g, v) in s.iter().enumerate() {
        store8(p.add(8 * g), v);
    }
}

/// Every DIF pass of one block, the scalar `block_dif`.
///
/// # Safety
/// As [`block_dit`].
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn block_dif<const INV: bool>(block: &mut [Fr], s: &mut [V], t: &BlockTables) {
    let p = block.as_mut_ptr();
    for (g, v) in s.iter_mut().enumerate() {
        *v = load8(p.add(8 * g));
    }
    dif_body::<INV>(s, t);
    for (g, run) in s.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        transpose8(run);
        tail::<INV>(run, t);
        for (i, v) in run.iter().enumerate() {
            store8_stride8(p.add(64 * g + i), v);
        }
    }
}

/// The fused middle of `intt_coset_ntt` on one block: inverse DIF, the coset factors
/// `coset16` (already in the `2^260` twiddle form, as `Fr` times 16), forward DIT.
///
/// # Safety
/// As [`block_dit`], and `coset16.len() == block.len()`.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn block_fused(block: &mut [Fr], coset16: &[Fr], s: &mut [V], t: &BlockTables) {
    let p = block.as_mut_ptr();
    let c = coset16.as_ptr();
    for (g, v) in s.iter_mut().enumerate() {
        *v = load8(p.add(8 * g));
    }
    dif_body::<true>(s, t);
    for (g, run) in s.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        transpose8(run);
        tail::<true>(run, t);
        for (i, v) in run.iter_mut().enumerate() {
            *v = mul(v, &load8_stride8(c.add(64 * g + i)));
        }
        head::<false>(run, t);
        transpose8(run);
    }
    dit_body::<false>(s, t);
    for (g, v) in s.iter().enumerate() {
        store8(p.add(8 * g), v);
    }
}

// ---------------------------------------------------------------------------
// Outer passes, straight on `[Fr]`, eight butterflies a step.
// ---------------------------------------------------------------------------

/// The scalar `butterflies` over `lo.len() / 8 * 8` pairs; returns how many it did.
/// `tab` is the size-`2 half` table, so the twiddle of pair `k` is entry `j0 + k`.
///
/// # Safety
/// IFMA available; `lo.len() == hi.len()`, `j0 + lo.len() <= tab.top`.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn butterflies<const DIF: bool, const INV: bool>(
    lo: &mut [Fr],
    hi: &mut [Fr],
    tab: &Table,
    j0: usize,
) -> usize {
    let n = lo.len() / 8 * 8;
    let (pl, ph) = (lo.as_mut_ptr(), hi.as_mut_ptr());
    for k in (0..n).step_by(8) {
        let (a, b) = (load8(pl.add(k)), load8(ph.add(k)));
        let w = tab.load::<INV>(j0 + k);
        let (x, y) = if DIF {
            (add(&a, &b), mul(&sub::<INV>(&a, &b), &w))
        } else {
            add_sub::<INV>(&a, &mul(&b, &w))
        };
        store8(pl.add(k), &x);
        store8(ph.add(k), &y);
    }
    n
}

/// The scalar `quad_dit` with `s2 = 1` over `x0.len() / 8 * 8` quads; returns how many.
/// `tab` is the size-`4 half` table.
///
/// # Safety
/// IFMA available; four strips of one length, `j0 + len <= half`.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn quad_dit<const INV: bool>(
    strips: [&mut [Fr]; 4],
    tab: &Table,
    half: usize,
    j0: usize,
) -> usize {
    let n = strips[0].len() / 8 * 8;
    let p = strips.map(|s| s.as_mut_ptr());
    for k in (0..n).step_by(8) {
        let j = j0 + k;
        let w1 = tab.load_even::<INV>(2 * j);
        let y: [V; 4] = std::array::from_fn(|i| load8(p[i].add(k)));
        let (a0, a1) = add_sub::<INV>(&y[0], &mul(&y[1], &w1));
        let (a2, a3) = add_sub::<INV>(&y[2], &mul(&y[3], &w1));
        let (s, d) = add_sub::<INV>(&a0, &mul(&a2, &tab.load::<INV>(j)));
        store8(p[0].add(k), &s);
        store8(p[2].add(k), &d);
        let (s, d) = add_sub::<INV>(&a1, &mul(&a3, &tab.load::<INV>(j + half)));
        store8(p[1].add(k), &s);
        store8(p[3].add(k), &d);
    }
    n
}

/// The scalar `quad_dif` with `s1 = 1` over `y0.len() / 8 * 8` quads; returns how many.
/// `tab` is the size-`4 hh` table.
///
/// # Safety
/// IFMA available; four strips of one length, `j0 + len <= hh`.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn quad_dif<const INV: bool>(
    strips: [&mut [Fr]; 4],
    tab: &Table,
    hh: usize,
    j0: usize,
) -> usize {
    let n = strips[0].len() / 8 * 8;
    let p = strips.map(|s| s.as_mut_ptr());
    for k in (0..n).step_by(8) {
        let j = j0 + k;
        let z: [V; 4] = std::array::from_fn(|i| load8(p[i].add(k)));
        let a0 = add(&z[0], &z[2]);
        let a2 = mul(&sub::<INV>(&z[0], &z[2]), &tab.load::<INV>(j));
        let a1 = add(&z[1], &z[3]);
        let a3 = mul(&sub::<INV>(&z[1], &z[3]), &tab.load::<INV>(j + hh));
        let w2 = tab.load_even::<INV>(2 * j);
        store8(p[0].add(k), &add(&a0, &a1));
        store8(p[1].add(k), &mul(&sub::<INV>(&a0, &a1), &w2));
        store8(p[2].add(k), &add(&a2, &a3));
        store8(p[3].add(k), &mul(&sub::<INV>(&a2, &a3), &w2));
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::AdditiveGroup;
    use snarkrs_field::Field;

    fn sample(n: usize, seed: u64) -> Vec<Fr> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (Fr::from(s) * Fr::from(s.rotate_left(29)) + Fr::from(s >> 3)).square()
            })
            .collect()
    }

    fn have() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512ifma")
    }

    #[test]
    fn round_trip_and_mul_match_ark() {
        if !have() {
            return;
        }
        let xs = sample(64, 7);
        let ws = sample(64, 11);
        let w16: Vec<Fr> = ws.iter().map(|w| *w * Fr::from(16u64)).collect();
        let mut out = vec![Fr::ZERO; 64];
        unsafe {
            store8(out.as_mut_ptr(), &load8(xs.as_ptr()));
            assert_eq!(&out[..8], &xs[..8]);
            let v = load8_stride8(xs.as_ptr());
            store8(out.as_mut_ptr(), &v);
            for t in 0..8 {
                assert_eq!(out[t], xs[8 * t]);
            }
            let mut o2 = vec![Fr::ZERO; 64];
            store8_stride8(o2.as_mut_ptr().add(3), &load8(xs.as_ptr()));
            for t in 0..8 {
                assert_eq!(o2[3 + 8 * t], xs[t]);
            }
            let p = mul(&load8(xs.as_ptr()), &load8(w16.as_ptr()));
            store8(out.as_mut_ptr(), &p);
            for t in 0..8 {
                assert_eq!(out[t], xs[t] * ws[t]);
            }
            let mut rows: Vec<V> = (0..8).map(|i| load8(xs.as_ptr().add(8 * i))).collect();
            transpose8(&mut rows);
            for (i, r) in rows.iter().enumerate() {
                store8(out.as_mut_ptr(), r);
                for t in 0..8 {
                    assert_eq!(out[t], xs[8 * t + i]);
                }
            }
        }
    }

    #[test]
    fn edge_values_survive_add_and_sub() {
        if !have() {
            return;
        }
        let m1 = -Fr::ONE;
        let xs = [
            Fr::ZERO,
            Fr::ONE,
            m1,
            m1,
            Fr::ZERO,
            Fr::ONE,
            m1,
            Fr::from(2u64),
        ];
        let ys = [m1, m1, m1, Fr::ZERO, Fr::ZERO, Fr::ONE, Fr::ONE, m1];
        let mut out = [Fr::ZERO; 8];
        unsafe {
            let (a, b) = (load8(xs.as_ptr()), load8(ys.as_ptr()));
            store8(out.as_mut_ptr(), &add(&a, &b));
            for t in 0..8 {
                assert_eq!(out[t], xs[t] + ys[t]);
            }
            store8(out.as_mut_ptr(), &sub_raw(&a, &b));
            for t in 0..8 {
                assert_eq!(out[t], xs[t] - ys[t]);
            }
        }
    }
}
