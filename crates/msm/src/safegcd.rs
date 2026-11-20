//! Inversion in BN254's `Fq` by Bernstein-Yang divsteps ("Fast constant-time gcd
//! computation and modular inversion", 2019), in the variable-time form libsecp256k1 runs
//! (`modinv64_var`): 62 divsteps at a time on the low limbs, each batch a 2x2 matrix then
//! applied to the full-width `f, g` and to the Bezout coefficients `d, e`.
//!
//! This replaces arkworks' binary extended Euclid where an inversion is on the hot path: the
//! lane ladders invert once per batched round, a few hundred times per batch of points,
//! and arkworks' takes several times as long. Both are variable-time, as is every ladder
//! that calls them.
//!
//! Numbers are five signed 62-bit limbs, value `sum v_i 2^(62 i)`; every limb but the top
//! is in `[0, 2^62)` between operations.

use ark_ff::{BigInt, Field, PrimeField};
use snarkrs_field::{Fq, Fq2};

const M62: u64 = u64::MAX >> 2;

type S62 = [i64; 5];

/// The modulus, limb by limb.
const P: [u64; 4] = <Fq as PrimeField>::MODULUS.0;

const fn to_s62(l: [u64; 4]) -> S62 {
    [
        (l[0] & M62) as i64,
        ((l[0] >> 62 | l[1] << 2) & M62) as i64,
        ((l[1] >> 60 | l[2] << 4) & M62) as i64,
        ((l[2] >> 58 | l[3] << 6) & M62) as i64,
        (l[3] >> 56) as i64,
    ]
}

const MOD: S62 = to_s62(P);

/// `p^-1 mod 2^62`, by Newton's iteration from `p` (which is its own inverse mod 8).
const MOD_INV62: u64 = {
    let mut x = P[0];
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(P[0].wrapping_mul(x)));
        i += 1;
    }
    x & M62
};

/// The transition matrix of 62 divsteps, scaled by `2^62`.
struct Trans {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

/// 62 divsteps on the low bits of `f` (odd) and `g`, skipping runs of zeros in `g` and
/// eliminating up to `min(eta + 1, steps left)` bits of `g` per step.
#[inline]
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64) -> (i64, Trans) {
    let (mut u, mut v, mut q, mut r) = (1u64, 0u64, 0u64, 1u64);
    let (mut f, mut g) = (f0, g0);
    let mut i: u32 = 62;
    loop {
        let zeros = (g | (u64::MAX << i)).trailing_zeros();
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros as i64;
        i -= zeros;
        if i == 0 {
            break;
        }
        let w;
        if eta < 0 {
            eta = -eta;
            let t = f;
            f = g;
            g = t.wrapping_neg();
            let t = u;
            u = q;
            q = t.wrapping_neg();
            let t = v;
            v = r;
            r = t.wrapping_neg();
            let limit = ((eta + 1) as u32).min(i);
            let m = (u64::MAX >> (64 - limit)) & 63;
            w = f
                .wrapping_mul(g)
                .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                & m;
        } else {
            let limit = ((eta + 1) as u32).min(i);
            let m = (u64::MAX >> (64 - limit)) & 15;
            let x = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = x.wrapping_neg().wrapping_mul(g) & m;
        }
        g = g.wrapping_add(f.wrapping_mul(w));
        q = q.wrapping_add(u.wrapping_mul(w));
        r = r.wrapping_add(v.wrapping_mul(w));
    }
    (
        eta,
        Trans {
            u: u as i64,
            v: v as i64,
            q: q as i64,
            r: r as i64,
        },
    )
}

/// `[d, e] <- (t [d, e] + p [md, me]) / 2^62`, with `md, me` chosen to make the division
/// exact and to keep both in `(-2p, p)`.
#[inline]
fn update_de(d: &mut S62, e: &mut S62, t: &Trans) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    let sd = d[4] >> 63;
    let se = e[4] >> 63;
    let mut md = (u & sd).wrapping_add(v & se);
    let mut me = (q & sd).wrapping_add(r & se);
    let mut cd = u as i128 * d[0] as i128 + v as i128 * e[0] as i128;
    let mut ce = q as i128 * d[0] as i128 + r as i128 * e[0] as i128;
    md = md.wrapping_sub((MOD_INV62.wrapping_mul(cd as u64).wrapping_add(md as u64) & M62) as i64);
    me = me.wrapping_sub((MOD_INV62.wrapping_mul(ce as u64).wrapping_add(me as u64) & M62) as i64);
    cd += MOD[0] as i128 * md as i128;
    ce += MOD[0] as i128 * me as i128;
    debug_assert!(cd as u64 & M62 == 0 && ce as u64 & M62 == 0);
    cd >>= 62;
    ce >>= 62;
    for i in 1..5 {
        cd += u as i128 * d[i] as i128 + v as i128 * e[i] as i128 + MOD[i] as i128 * md as i128;
        ce += q as i128 * d[i] as i128 + r as i128 * e[i] as i128 + MOD[i] as i128 * me as i128;
        d[i - 1] = (cd as u64 & M62) as i64;
        e[i - 1] = (ce as u64 & M62) as i64;
        cd >>= 62;
        ce >>= 62;
    }
    d[4] = cd as i64;
    e[4] = ce as i64;
}

/// `[f, g] <- t [f, g] / 2^62` over the low `len` limbs.
#[inline]
fn update_fg(len: usize, f: &mut S62, g: &mut S62, t: &Trans) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    let mut cf = u as i128 * f[0] as i128 + v as i128 * g[0] as i128;
    let mut cg = q as i128 * f[0] as i128 + r as i128 * g[0] as i128;
    debug_assert!(cf as u64 & M62 == 0 && cg as u64 & M62 == 0);
    cf >>= 62;
    cg >>= 62;
    for i in 1..len {
        cf += u as i128 * f[i] as i128 + v as i128 * g[i] as i128;
        cg += q as i128 * f[i] as i128 + r as i128 * g[i] as i128;
        f[i - 1] = (cf as u64 & M62) as i64;
        g[i - 1] = (cg as u64 & M62) as i64;
        cf >>= 62;
        cg >>= 62;
    }
    f[len - 1] = cf as i64;
    g[len - 1] = cg as i64;
}

/// `r` in `(-2p, p)` times `sign` (the sign of the final `f`, `+-1`) into `[0, p)`.
#[inline]
fn normalize(r: &mut S62, sign: i64) {
    let m62 = M62 as i64;
    let add = |r: &mut S62| {
        let c = r[4] >> 63;
        for i in 0..5 {
            r[i] += MOD[i] & c;
        }
    };
    let carry = |r: &mut S62| {
        for i in 0..4 {
            r[i + 1] += r[i] >> 62;
            r[i] &= m62;
        }
    };
    add(r);
    let neg = sign >> 63;
    for x in r.iter_mut() {
        *x = (*x ^ neg) - neg;
    }
    carry(r);
    add(r);
    carry(r);
}

/// `x^-1 mod p` for a plain (not Montgomery) `x` in `[1, p)`; `None` for zero.
pub fn inv_mod_p(x: [u64; 4]) -> Option<[u64; 4]> {
    if x == [0; 4] {
        return None;
    }
    let mut d: S62 = [0; 5];
    let mut e: S62 = [1, 0, 0, 0, 0];
    let mut f = MOD;
    let mut g = to_s62(x);
    let mut len = 5;
    let mut eta: i64 = -1;
    loop {
        let (eta2, t) = divsteps_62_var(eta, f[0] as u64, g[0] as u64);
        eta = eta2;
        update_de(&mut d, &mut e, &t);
        update_fg(len, &mut f, &mut g, &t);
        if g[0] == 0 && g[1..len].iter().all(|&x| x == 0) {
            break;
        }
        let fln = f[len - 1];
        let gln = g[len - 1];
        // Shorten while both tops are sign words: fold them into the limb below.
        if len >= 2 && (fln ^ (fln >> 63)) == 0 && (gln ^ (gln >> 63)) == 0 {
            f[len - 2] |= ((fln as u64) << 62) as i64;
            g[len - 2] |= ((gln as u64) << 62) as i64;
            len -= 1;
        }
    }
    // gcd(p, x) = 1, so f is +-1 here.
    normalize(&mut d, f[len - 1]);
    let r = [
        d[0] as u64,
        d[1] as u64,
        d[2] as u64,
        d[3] as u64,
        d[4] as u64,
    ];
    Some([
        r[0] | r[1] << 62,
        r[1] >> 2 | r[2] << 60,
        r[2] >> 4 | r[3] << 58,
        r[3] >> 6 | r[4] << 56,
    ])
}

/// The inverse of `x` in `Fq`, `None` for zero.
pub fn fq_inv(x: &Fq) -> Option<Fq> {
    let plain = inv_mod_p(x.into_bigint().0)?;
    Some(Fq::from_bigint(BigInt::new(plain)).expect("a canonical residue"))
}

/// The inverse in `Fq2 = Fq[u]/(u^2 + 1)`: `(c0 - c1 u) / (c0^2 + c1^2)`.
pub fn fq2_inv(x: &Fq2) -> Option<Fq2> {
    let n = fq_inv(&(x.c0.square() + x.c1.square()))?;
    Some(Fq2::new(x.c0 * n, -(x.c1 * n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{UniformRand, Zero};
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn matches_arkworks() {
        let mut rng = StdRng::seed_from_u64(0x5afe);
        let mut xs: Vec<Fq> = (0..20000).map(|_| Fq::rand(&mut rng)).collect();
        let edge = [1u64, 2, 3, 7].map(Fq::from);
        xs.extend(edge);
        xs.extend(edge.map(|x| -x));
        for k in 0..254 {
            xs.push(Fq::from(BigInt::<4>::from(1u64) << k as u32));
            let mut b = BigInt::<4>::from(1u64) << k as u32;
            ark_ff::BigInteger::sub_with_borrow(&mut b, &BigInt::from(1u64));
            if let Some(f) = Fq::from_bigint(b) {
                xs.push(f);
            }
        }
        for x in &xs {
            if x.is_zero() {
                continue;
            }
            assert_eq!(fq_inv(x), x.inverse(), "{x}");
        }
        assert_eq!(fq_inv(&Fq::from(0u64)), None);
        for _ in 0..2000 {
            let x = Fq2::rand(&mut rng);
            assert_eq!(fq2_inv(&x), x.inverse());
        }
    }
}
