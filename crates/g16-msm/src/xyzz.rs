//! Extended-Jacobian (XYZZ) curve arithmetic over the branch-free raw field layer.
//!
//! This is the CPU twin of the point arithmetic in `g16-metal/src/shaders/msm.metal`
//! (stanzas madd-2008-s, mdbl-2008-s, add-2008-s, dbl-2008-s-1), formula for formula:
//! x = X/ZZ, y = Y/ZZZ with the invariant ZZ^3 = ZZZ^2, identity encoded as ZZ == 0.
//! Mixed addition is 8M + 2S against ark's Jacobian madd at 7M + 4S, and, because the
//! field layer is `g16_field::raw`, none of those operations ends in a compare-and-branch
//! reduction. Profiling put a G1 mixed add through `ark_ec` at 247 ns against a 148 ns
//! multiply floor; this module exists to close that gap in the MSM bucket loop, which is
//! 15.5 million of exactly these additions per 140k-constraint proof.
//!
//! Formulas from the standard XYZZ addition stanzas with a = 0 (both BN254 groups); the
//! Metal shader is the reference implementation here, already audited against the CPU
//! prover end to end.
//!
//! Everything is differential-tested against `ark_ec` in `tests/xyzz_matches_ark.rs`,
//! including the doubling and cancellation corners a random sweep essentially never hits.

use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{Field, Zero};
use g16_field::raw::{RawField, RawFq, RawFq2};

/// The accumulator. `ZZ == 0` is the identity, matching the GPU layout.
#[derive(Clone, Copy, Debug)]
pub struct Xyzz<F> {
    pub x: F,
    pub y: F,
    pub zz: F,
    pub zzz: F,
}

impl<F: RawField> Xyzz<F> {
    pub const ZERO: Xyzz<F> = Xyzz {
        x: F::ZERO,
        y: F::ZERO,
        zz: F::ZERO,
        zzz: F::ZERO,
    };

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.zz.is_zero()
    }

    #[inline(always)]
    fn from_affine(x: F, y: F) -> Self {
        Xyzz {
            x,
            y,
            zz: F::ONE,
            zzz: F::ONE,
        }
    }

    /// mdbl-2008-s with a = 0: double an affine point straight into XYZZ.
    #[inline(always)]
    fn dbl_affine(px: F, py: F) -> Self {
        if py.is_zero() {
            // A 2-torsion point, which BN254's odd-order groups do not contain; the
            // identity is the mathematically correct answer anyway.
            return Self::ZERO;
        }
        let u = py.double();
        let v = u.sqr();
        let w = u.mul(v);
        let s = px.mul(v);
        let xx = px.sqr();
        let m = xx.double().add(xx);
        let x = m.sqr().sub(s.double());
        let y = m.mul(s.sub(x)).sub(w.mul(py));
        Xyzz {
            x,
            y,
            zz: v,
            zzz: w,
        }
    }

    /// dbl-2008-s-1 with a = 0.
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        if self.is_zero() || self.y.is_zero() {
            return Self::ZERO;
        }
        let u = self.y.double();
        let v = u.sqr();
        let w = u.mul(v);
        let s = self.x.mul(v);
        let xx = self.x.sqr();
        let m = xx.double().add(xx);
        let x = m.sqr().sub(s.double());
        let y = m.mul(s.sub(x)).sub(w.mul(self.y));
        Xyzz {
            x,
            y,
            zz: v.mul(self.zz),
            zzz: w.mul(self.zzz),
        }
    }

    /// madd-2008-s: `self += (px, py)`, 8M + 2S. The bucket-loop workhorse.
    ///
    /// The caller guarantees `(px, py)` is NOT the point at infinity; the prescan filters
    /// those out before anything reaches a bucket (34% of the B query on real keys).
    #[inline(always)]
    pub fn madd(&mut self, px: F, py: F) {
        if self.is_zero() {
            *self = Self::from_affine(px, py);
            return;
        }
        let u2 = px.mul(self.zz);
        let s2 = py.mul(self.zzz);
        let p = u2.sub(self.x);
        let r = s2.sub(self.y);
        if p.is_zero() {
            // Same x: either the same point (double it) or its negative (cancel). Both
            // can happen in a bucket: a zkey does not guarantee distinct bases.
            *self = if r.is_zero() {
                Self::dbl_affine(px, py)
            } else {
                Self::ZERO
            };
            return;
        }
        let pp = p.sqr();
        let ppp = p.mul(pp);
        let q = self.x.mul(pp);
        let x = r.sqr().sub(ppp).sub(q.double());
        let y = r.mul(q.sub(x)).sub(self.y.mul(ppp));
        self.x = x;
        self.y = y;
        self.zz = self.zz.mul(pp);
        self.zzz = self.zzz.mul(ppp);
    }

    /// add-2008-s: `self += other`, 12M + 2S. Used by the running-sum reduction, which
    /// is `2 * 2^(c-1)` of these per window against `n` mixed additions.
    #[inline(always)]
    pub fn add_assign(&mut self, o: &Self) {
        if o.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = *o;
            return;
        }
        let u1 = self.x.mul(o.zz);
        let u2 = o.x.mul(self.zz);
        let s1 = self.y.mul(o.zzz);
        let s2 = o.y.mul(self.zzz);
        let p = u2.sub(u1);
        let r = s2.sub(s1);
        if p.is_zero() {
            *self = if r.is_zero() { self.dbl() } else { Self::ZERO };
            return;
        }
        let pp = p.sqr();
        let ppp = p.mul(pp);
        let q = u1.mul(pp);
        let x = r.sqr().sub(ppp).sub(q.double());
        let y = r.mul(q.sub(x)).sub(s1.mul(ppp));
        self.x = x;
        self.y = y;
        self.zz = self.zz.mul(o.zz).mul(pp);
        self.zzz = self.zzz.mul(o.zzz).mul(ppp);
    }
}

/// The per-group glue: raw coordinate extraction and the XYZZ -> Jacobian exit.
/// Implemented for exactly the two BN254 groups the prover uses.
pub trait RawCurve: SWCurveConfig + Sized {
    type RF: RawField;
    /// `(x, y)` as raw limbs. Caller must have excluded the point at infinity.
    fn raw_xy(p: &Affine<Self>) -> (Self::RF, Self::RF);
    fn field_back(f: Self::RF) -> Self::BaseField;
    /// Inverse of a nonzero element, through ark's extended Euclid. Off the hot path:
    /// the batch-affine fill calls it once per shared-inversion round, never per point.
    fn raw_inv(f: Self::RF) -> Self::RF;
}

impl RawCurve for g16_field::g1::Config {
    type RF = RawFq;
    #[inline(always)]
    fn raw_xy(p: &Affine<Self>) -> (RawFq, RawFq) {
        (RawFq::from_fq(&p.x), RawFq::from_fq(&p.y))
    }
    #[inline(always)]
    fn field_back(f: RawFq) -> g16_field::Fq {
        f.to_fq()
    }
    fn raw_inv(f: RawFq) -> RawFq {
        RawFq::from_fq(&f.to_fq().inverse().expect("inverse of zero"))
    }
}

impl RawCurve for g16_field::g2::Config {
    type RF = RawFq2;
    #[inline(always)]
    fn raw_xy(p: &Affine<Self>) -> (RawFq2, RawFq2) {
        (RawFq2::from_fq2(&p.x), RawFq2::from_fq2(&p.y))
    }
    #[inline(always)]
    fn field_back(f: RawFq2) -> g16_field::Fq2 {
        f.to_fq2()
    }
    fn raw_inv(f: RawFq2) -> RawFq2 {
        RawFq2::from_fq2(&f.to_fq2().inverse().expect("inverse of zero"))
    }
}

/// XYZZ -> ark Jacobian, multiplication only (no inversion): with `Z = ZZ * ZZZ`,
/// `Z^2 = ZZ^5` and `Z^3 = ZZZ^5` (both by `ZZZ^2 = ZZ^3`), so
/// `X_j = x * Z^2 = X * ZZ^4` and `Y_j = y * Z^3 = Y * ZZZ^4`. Seven multiplies, run
/// once per window chunk, not per point.
pub fn to_projective<P: RawCurve>(p: &Xyzz<P::RF>) -> Projective<P> {
    if p.is_zero() {
        return Projective::zero();
    }
    let zz2 = p.zz.sqr();
    let zz4 = zz2.sqr();
    let zzz2 = p.zzz.sqr();
    let zzz4 = zzz2.sqr();
    let xj = p.x.mul(zz4);
    let yj = p.y.mul(zzz4);
    let zj = p.zz.mul(p.zzz);
    Projective::new_unchecked(P::field_back(xj), P::field_back(yj), P::field_back(zj))
}
