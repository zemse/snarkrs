//! Variable-time modular inversion by Bernstein-Yang divsteps, 62 at a time, after
//! libsecp256k1's `modinv64_var` (MIT): the signed-62 representation, the transition
//! matrix found on the bottom 64 bits, and the matrix applied to the full-width `d, e`
//! (with the multiple of the modulus that keeps them integral) and `f, g`.
//!
//! ark-ff's binary extended Euclid takes about 6.0 us for a BN254 Fq element on one core of
//! a 2.1 GHz Sapphire Rapids VM; this takes 2.0. The batch-affine fills spend one inversion
//! a round, so this is a real share of a round at the narrow windows. Variable time, as the
//! Euclid it replaces: the running time depends on the value inverted.

/// Five limbs, `sum v[i] 2^(62 i)`, the bottom four in `[0, 2^62)` when normalised.
#[derive(Clone, Copy)]
struct Signed62([i64; 5]);

const M62: u64 = u64::MAX >> 2;

/// What one modulus needs: itself in signed-62 form and its inverse mod 2^62.
pub(crate) struct ModInfo {
    modulus: Signed62,
    modulus_inv62: u64,
}

impl ModInfo {
    /// `m` odd, below 2^256; `neg_inv` is `-m^-1 mod 2^64`, the Montgomery constant.
    pub(crate) const fn new(m: &[u64; 4], neg_inv: u64) -> Self {
        ModInfo {
            modulus: to_62(m),
            modulus_inv62: neg_inv.wrapping_neg() & M62,
        }
    }
}

const fn to_62(a: &[u64; 4]) -> Signed62 {
    Signed62([
        (a[0] & M62) as i64,
        ((a[0] >> 62 | a[1] << 2) & M62) as i64,
        ((a[1] >> 60 | a[2] << 4) & M62) as i64,
        ((a[2] >> 58 | a[3] << 6) & M62) as i64,
        (a[3] >> 56) as i64,
    ])
}

fn from_62(v: &Signed62) -> [u64; 4] {
    let v = v.0.map(|x| x as u64);
    [
        v[0] | v[1] << 62,
        v[1] >> 2 | v[2] << 60,
        v[2] >> 4 | v[3] << 58,
        v[3] >> 6 | v[4] << 56,
    ]
}

/// The 2x2 matrix of 62 divsteps, scaled by 2^62.
struct Trans {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

/// 62 divsteps on the bottom bits of `f` and `g`, several at a time where `g` is even or a
/// small multiple of `f` cancels its low bits. `eta` is minus the paper's delta.
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64) -> (i64, Trans) {
    let (mut u, mut v, mut q, mut r) = (1u64, 0u64, 0u64, 1u64);
    let (mut f, mut g) = (f0, g0);
    let mut i: u32 = 62;
    loop {
        // A sentinel bit stops the count at `i`.
        let zeros = (g | (u64::MAX << i)).trailing_zeros();
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros as i64;
        i -= zeros;
        if i == 0 {
            break;
        }
        let w = if eta < 0 {
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
            // Cancel up to six bits of g, no more than i and no more than eta + 1.
            let limit = (eta as u32 + 1).min(i);
            let m = (u64::MAX >> (64 - limit)) & 63;
            f.wrapping_mul(g)
                .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                & m
        } else {
            // Up to four bits here, eta tending to be small on this side.
            let limit = (eta as u32 + 1).min(i);
            let m = (u64::MAX >> (64 - limit)) & 15;
            let w0 = f.wrapping_add(((f.wrapping_add(1)) & 4) << 1);
            w0.wrapping_neg().wrapping_mul(g) & m
        };
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

/// `[d, e] = (t [d, e] + m [md, me]) / 2^62`, with `md, me` chosen to make the division
/// exact and to keep `d, e` in `(-2m, m)`.
fn update_de_62(d: &mut Signed62, e: &mut Signed62, t: &Trans, mi: &ModInfo) {
    let [d0, d1, d2, d3, d4] = d.0;
    let [e0, e1, e2, e3, e4] = e.0;
    let (u, v, q, r) = (t.u as i128, t.v as i128, t.q as i128, t.r as i128);
    let m = mi.modulus.0;
    let sd = d4 >> 63;
    let se = e4 >> 63;
    let mut md = (t.u & sd).wrapping_add(t.v & se);
    let mut me = (t.q & sd).wrapping_add(t.r & se);
    let mut cd = u * d0 as i128 + v * e0 as i128;
    let mut ce = q * d0 as i128 + r * e0 as i128;
    md = md.wrapping_sub(
        (mi.modulus_inv62
            .wrapping_mul(cd as u64)
            .wrapping_add(md as u64)
            & M62) as i64,
    );
    me = me.wrapping_sub(
        (mi.modulus_inv62
            .wrapping_mul(ce as u64)
            .wrapping_add(me as u64)
            & M62) as i64,
    );
    cd += m[0] as i128 * md as i128;
    ce += m[0] as i128 * me as i128;
    debug_assert!(cd as u64 & M62 == 0 && ce as u64 & M62 == 0);
    cd >>= 62;
    ce >>= 62;
    let limbs = [(d1, e1), (d2, e2), (d3, e3), (d4, e4)];
    for (k, &(dk, ek)) in limbs.iter().enumerate() {
        cd += u * dk as i128 + v * ek as i128 + m[k + 1] as i128 * md as i128;
        ce += q * dk as i128 + r * ek as i128 + m[k + 1] as i128 * me as i128;
        if k < 3 {
            d.0[k] = (cd as i64) & M62 as i64;
            e.0[k] = (ce as i64) & M62 as i64;
            cd >>= 62;
            ce >>= 62;
        }
    }
    d.0[3] = (cd as i64) & M62 as i64;
    e.0[3] = (ce as i64) & M62 as i64;
    cd >>= 62;
    ce >>= 62;
    d.0[4] = cd as i64;
    e.0[4] = ce as i64;
}

/// `[f, g] = t [f, g] / 2^62` over the bottom `len` limbs.
fn update_fg_62_var(len: usize, f: &mut [i64; 5], g: &mut [i64; 5], t: &Trans) {
    let (u, v, q, r) = (t.u as i128, t.v as i128, t.q as i128, t.r as i128);
    let mut cf = u * f[0] as i128 + v * g[0] as i128;
    let mut cg = q * f[0] as i128 + r * g[0] as i128;
    debug_assert!(cf as u64 & M62 == 0 && cg as u64 & M62 == 0);
    cf >>= 62;
    cg >>= 62;
    for i in 1..len {
        let (fi, gi) = (f[i] as i128, g[i] as i128);
        cf += u * fi + v * gi;
        cg += q * fi + r * gi;
        f[i - 1] = (cf as i64) & M62 as i64;
        g[i - 1] = (cg as i64) & M62 as i64;
        cf >>= 62;
        cg >>= 62;
    }
    f[len - 1] = cf as i64;
    g[len - 1] = cg as i64;
}

/// `r` in `(-2m, m)`, negated when `sign` is negative, into `[0, m)`.
fn normalize_62(r: &mut Signed62, sign: i64, mi: &ModInfo) {
    let m = mi.modulus.0;
    let mut x = r.0;
    let add = x[4] >> 63;
    for i in 0..5 {
        x[i] += m[i] & add;
    }
    let neg = sign >> 63;
    for v in x.iter_mut() {
        *v = (*v ^ neg) - neg;
    }
    let carry = |x: &mut [i64; 5]| {
        for i in 0..4 {
            x[i + 1] += x[i] >> 62;
            x[i] &= M62 as i64;
        }
    };
    carry(&mut x);
    let add = x[4] >> 63;
    for i in 0..5 {
        x[i] += m[i] & add;
    }
    carry(&mut x);
    r.0 = x;
}

/// `x^-1 mod m`, canonical, for `x` in `[1, m)`; 0 for 0.
pub(crate) fn inv_mod(x: &[u64; 4], mi: &ModInfo) -> [u64; 4] {
    let mut d = Signed62([0; 5]);
    let mut e = Signed62([1, 0, 0, 0, 0]);
    let mut f = mi.modulus.0;
    let mut g = to_62(x).0;
    let mut len = 5;
    let mut eta = -1i64;
    loop {
        let (eta2, t) = divsteps_62_var(eta, f[0] as u64, g[0] as u64);
        eta = eta2;
        update_de_62(&mut d, &mut e, &t, mi);
        update_fg_62_var(len, &mut f, &mut g, &t);
        if g[0] == 0 && g[1..len].iter().all(|&l| l == 0) {
            break;
        }
        let (fnl, gnl) = (f[len - 1], g[len - 1]);
        let cond = ((len as i64 - 2) >> 63) | (fnl ^ (fnl >> 63)) | (gnl ^ (gnl >> 63));
        if cond == 0 {
            f[len - 2] |= ((fnl as u64) << 62) as i64;
            g[len - 2] |= ((gnl as u64) << 62) as i64;
            len -= 1;
        }
    }
    normalize_62(&mut d, f[len - 1], mi);
    from_62(&d)
}
