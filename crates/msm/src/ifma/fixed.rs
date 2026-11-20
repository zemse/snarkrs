//! `P_i <- [k] P_i` on G1 with one `k` for every point, eight points a vector.
//!
//! This is `zkey contribute`'s and `zkey beacon`'s whole cost: sections 8 and 9 are scaled
//! by a single `invDelta`. With one scalar there is one digit sequence, so every point
//! takes the same doublings and additions in the same order, and the batch-affine ladder of
//! [`crate::glv::mul_batch_xyzz_in`] runs lane-parallel with no per-point scheduling at all:
//! GLV-split `k`, one width-[`W`](crate::glv) NAF per half, a table of odd multiples per
//! point (the `phi` half reads the same table through `beta x`), and one shared inversion
//! per ladder step, through Montgomery's trick over two interleaved chains of vectors.
//!
//! Arithmetic is the parent module's radix-2^52 Montgomery field. The output is affine,
//! which is canonical, so the bytes written are those of any other way of computing it.
//!
//! An exceptional addition (`acc = +-q`) depends only on the scalar, never on the point,
//! because every G1 point but the identity has the same prime order. When one comes up the
//! kernel says so and changes nothing, and the caller takes the generic path, which has its
//! own handling. The identity is carried as the generator and written back as the identity.

use super::*;
use crate::glv::{decompose, wnaf, BETA, DIGITS};
use ark_ec::AffineRepr;
use snarkrs_field::{Fr, G1Affine};

type G1Cfg = snarkrs_field::g1::Config;

/// `[k] P` for every point, in place. `false` when this CPU has no IFMA or the scalar hits
/// an exceptional addition; the points are then untouched.
pub(crate) fn mul_fixed_g1(points: &mut [G1Affine], k: &Fr) -> bool {
    if !available() {
        return false;
    }
    // SAFETY: `available` checked the target features.
    unsafe { mul_fixed_g1_avx(points, k) }
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn mul_fixed_g1_avx(points: &mut [G1Affine], k: &Fr) -> bool {
    let n = points.len();
    if n == 0 {
        return true;
    }
    let (k1, k2) = decompose(k);
    let mut d1 = [0i8; DIGITS];
    let mut d2 = [0i8; DIGITS];
    let n1 = wnaf(k1.unsigned_abs(), &mut d1);
    let n2 = wnaf(k2.unsigned_abs(), &mut d2);
    let steps = n1.max(n2);
    if steps == 0 {
        // k = 0 mod r.
        points.fill(G1Affine::identity());
        return true;
    }
    let top = d1[..n1]
        .iter()
        .chain(&d2[..n2])
        .map(|d| d.unsigned_abs() as usize)
        .max()
        .unwrap_or(1);
    let tlen = top / 2 + 1;

    // An even number of groups of eight, so the two inversion chains are the same length.
    let g = n.div_ceil(16) * 2;
    let (gx, gy) = G1Cfg::raw_xy(&G1Affine::generator());
    let mut tx: Vec<V> = Vec::with_capacity(tlen * g);
    let mut ty: Vec<V> = Vec::with_capacity(tlen * g);
    for grp in 0..g {
        let (mut xs, mut ys) = ([gx; 8], [gy; 8]);
        for l in 0..8 {
            if let Some(p) = points.get(grp * 8 + l) {
                if !p.infinity {
                    (xs[l], ys[l]) = G1Cfg::raw_xy(p);
                }
            }
        }
        // SAFETY: the target features are enabled here.
        unsafe {
            tx.push(V::enter(&xs));
            ty.push(V::enter(&ys));
        }
    }
    let mut pre: Vec<V> = vec![splat(&ONE); g];

    // Odd multiples: 2P, then (2e + 1) P = (2e - 1) P + 2P.
    if tlen > 1 {
        let (mut dx, mut dy) = (tx.clone(), ty.clone());
        batch_dbl(&mut dx, &mut dy, &mut pre);
        for e in 1..tlen {
            let mut ax = tx[(e - 1) * g..e * g].to_vec();
            let mut ay = ty[(e - 1) * g..e * g].to_vec();
            if !batch_add(&mut ax, &mut ay, &dx, &dy, &mut pre) {
                return false;
            }
            tx.extend_from_slice(&ax);
            ty.extend_from_slice(&ay);
        }
    }

    // SAFETY: the target features are enabled here.
    let beta = unsafe { V::enter(&[RawFq(BETA.0 .0); 8]) };
    let zero = [_mm512_setzero_si512(); 5];
    let (mut qx, mut qy) = (vec![zero; g], vec![zero; g]);
    let (mut ax, mut ay): (Vec<V>, Vec<V>) = (Vec::new(), Vec::new());
    for s in (0..steps).rev() {
        if !ax.is_empty() {
            batch_dbl(&mut ax, &mut ay, &mut pre);
        }
        for (d, phi, neg) in [(d1[s], false, k1 < 0), (d2[s], true, k2 < 0)] {
            if d == 0 {
                continue;
            }
            let e = (d.unsigned_abs() >> 1) as usize;
            let flip = (d < 0) != neg;
            let (ex, ey) = (&tx[e * g..(e + 1) * g], &ty[e * g..(e + 1) * g]);
            for j in 0..g {
                qx[j] = if phi { mul(&ex[j], &beta) } else { ex[j] };
                qy[j] = if flip { sub(&zero, &ey[j]) } else { ey[j] };
            }
            if ax.is_empty() {
                ax.clone_from(&qx);
                ay.clone_from(&qy);
            } else if !batch_add(&mut ax, &mut ay, &qx, &qy, &mut pre) {
                return false;
            }
        }
    }

    for grp in 0..g {
        // SAFETY: the target features are enabled here.
        let (xs, ys) = unsafe { (ax[grp].leave(), ay[grp].leave()) };
        for l in 0..8 {
            if let Some(p) = points.get_mut(grp * 8 + l) {
                if !p.infinity {
                    *p = G1Affine::new_unchecked(xs[l].to_fq(), ys[l].to_fq());
                }
            }
        }
    }
    true
}

/// `a + b` for `a, b < 4p`, left unreduced: below `8p`, which [`mul`] still takes. Its
/// output is below `p + ab / 2^260`, so under `2p` for inputs up to `8p`, because
/// `64 p < 2^260`. Only ever a multiplicand, never stored.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn add_nr(a: &V, b: &V) -> V {
    let mask = _mm512_set1_epi64(MASK as i64);
    let mut s = [_mm512_setzero_si512(); 5];
    let mut carry = _mm512_setzero_si512();
    for j in 0..5 {
        let t = _mm512_add_epi64(_mm512_add_epi64(a[j], b[j]), carry);
        carry = _mm512_srli_epi64::<52>(t);
        s[j] = if j < 4 { _mm512_and_si512(t, mask) } else { t };
    }
    s
}

/// `a + 2p - b` for `a, b < 2p`, left unreduced: in `(0, 4p)`. A multiplicand only, as
/// [`add_nr`].
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub_nr(a: &V, b: &V) -> V {
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
        carry = _mm512_srai_epi64::<52>(t);
        s[j] = if j < 4 { _mm512_and_si512(t, mask) } else { t };
    }
    s
}

/// Montgomery's trick over `den(j)` for every group `j`, as two chains of half the groups
/// each, walked side by side. `f(j, 1 / den(j))` is called backwards, two groups at a time,
/// after `den(j)` has been read for the last time. `false`, before any call, when a lane's
/// denominator is zero.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn invert_groups(
    g: usize,
    pre: &mut [V],
    den: impl Fn(usize) -> V,
    mut f: impl FnMut(usize, &V),
) -> bool {
    let h = g / 2;
    let (mut a, mut b) = (splat(&ONE), splat(&ONE));
    for j in 0..h {
        pre[j] = a;
        pre[h + j] = b;
        a = mul(&a, &den(j));
        b = mul(&b, &den(h + j));
    }
    let ab = mul(&a, &b);
    if is_zero(&ab) != 0 {
        return false;
    }
    // SAFETY: the target features are enabled here, and no lane is zero.
    let inv = unsafe { ab.invert() };
    let (mut ia, mut ib) = (mul(&inv, &b), mul(&inv, &a));
    for j in (0..h).rev() {
        let (i0, i1) = (mul(&ia, &pre[j]), mul(&ib, &pre[h + j]));
        ia = mul(&ia, &den(j));
        ib = mul(&ib, &den(h + j));
        f(j, &i0);
        f(h + j, &i1);
    }
    true
}

/// `(x, y) <- 2 (x, y)` for every lane: `l = 3 x^2 / 2 y`, `x' = l^2 - 2 x`,
/// `y' = l (x - x') - y`. No point of G1 has `y = 0`.
#[target_feature(enable = "avx512f,avx512ifma")]
fn batch_dbl(x: &mut [V], y: &mut [V], pre: &mut [V]) {
    let g = x.len();
    let (xr, yr) = (x.as_mut_ptr(), y.as_mut_ptr());
    // SAFETY: `den` reads entry `j` only before `f` writes it (see `invert_groups`), and
    // both stay inside the slices.
    let ok = invert_groups(
        g,
        pre,
        |j| unsafe {
            let yj = &*yr.add(j);
            add_nr(yj, yj)
        },
        |j, inv| unsafe {
            let (px, py) = (*xr.add(j), *yr.add(j));
            let xx = mul(&px, &px);
            let l = mul(&add_nr(&add_nr(&xx, &xx), &xx), inv);
            let x3 = sub(&sub(&mul(&l, &l), &px), &px);
            let y3 = sub(&mul(&l, &sub_nr(&px, &x3)), &py);
            *xr.add(j) = x3;
            *yr.add(j) = y3;
        },
    );
    debug_assert!(ok, "a G1 point of order two");
}

/// `(x, y) <- (x, y) + (qx, qy)` for every lane: `l = (qy - y) / (qx - x)`,
/// `x' = l^2 - x - qx`, `y' = l (x - x') - y`. `false`, with nothing changed, when a lane
/// has `x = qx`.
#[target_feature(enable = "avx512f,avx512ifma")]
fn batch_add(x: &mut [V], y: &mut [V], qx: &[V], qy: &[V], pre: &mut [V]) -> bool {
    let g = x.len();
    let (xr, yr) = (x.as_mut_ptr(), y.as_mut_ptr());
    // SAFETY: as in `batch_dbl`.
    invert_groups(
        g,
        pre,
        |j| unsafe { sub_nr(&qx[j], &*xr.add(j)) },
        |j, inv| unsafe {
            let (px, py) = (*xr.add(j), *yr.add(j));
            let l = mul(&sub_nr(&qy[j], &py), inv);
            let x3 = sub(&sub(&mul(&l, &l), &px), &qx[j]);
            let y3 = sub(&mul(&l, &sub_nr(&px, &x3)), &py);
            *xr.add(j) = x3;
            *yr.add(j) = y3;
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::CurveGroup;
    use ark_ff::{BigInt, One, PrimeField};
    use ark_std::{test_rng, UniformRand};
    use snarkrs_field::G1Projective;

    #[test]
    fn matches_ark_for_every_length_and_edge_scalars() {
        if !available() {
            eprintln!("no AVX-512 IFMA on this CPU, skipped");
            return;
        }
        let mut rng = test_rng();
        let mut scalars = vec![
            Fr::from(0u64),
            Fr::one(),
            Fr::from(2u64),
            Fr::from(3u64),
            -Fr::one(),
            -Fr::from(2u64),
            crate::glv::LAMBDA,
            -crate::glv::LAMBDA,
            crate::glv::LAMBDA + Fr::one(),
            Fr::from(u64::MAX),
            Fr::from_bigint(BigInt::new([0, 0, 1, 0])).unwrap(),
        ];
        for _ in 0..6 {
            scalars.push(Fr::rand(&mut rng));
        }
        for (t, k) in scalars.iter().enumerate() {
            for n in [1usize, 7, 8, 9, 16, 17, 40] {
                let mut pts: Vec<G1Affine> = (0..n)
                    .map(|_| G1Projective::rand(&mut rng).into_affine())
                    .collect();
                if n > 3 {
                    pts[2] = G1Affine::identity();
                }
                let want: Vec<G1Affine> = pts.iter().map(|p| (*p * k).into_affine()).collect();
                let mut got = pts.clone();
                if mul_fixed_g1(&mut got, k) {
                    assert_eq!(got, want, "scalar {t}, n = {n}");
                } else {
                    // A random scalar meets an exceptional addition with probability ~2^-120.
                    assert!(t < 11, "random scalar {t} refused");
                    assert_eq!(got, pts, "a refusal changes nothing");
                }
            }
        }
    }
}
