//! Host-side checks of what a submission wrote, for the ceremony seams.
//!
//! A completion token says the last dispatch of a command buffer ran; it says nothing
//! about the stores before it. Metal has reported buffers `Completed`, token in place,
//! with values it did not store (BUG-28), and the MSM already refuses a window sum off
//! the curve for that reason (`msm.rs`, `combine`). The ceremony kernels' outputs go
//! straight into a file, so every readback of theirs is checked here the same way, inside
//! the `cb::with_retry` closure that made it: a failure is a [`ProveError::Device`] with
//! no `kIOGPU` name in it, which `with_retry` retries blind from the unchanged inputs, so
//! a transient fault is re-run and a persistent one fails the operation with nothing
//! written.
//!
//! Two kinds of check. Every point read back is held against the curve equation, which
//! catches a wrong word anywhere in it (the addition formulas are rational maps, so a
//! point off the curve stays off through everything downstream). One point per readback
//! is also recomputed on the host and compared exactly, which is what catches a readback
//! that is on the curve and wrong: the previous pass's points left in a destination, or
//! a buffer that was never written. The position comes from the chunk, not from the OS,
//! so a failure replays.

use ark_ec::short_weierstrass::SWCurveConfig;
use snarkrs_field::raw::{RawField, RawFq, RawFq2};
use snarkrs_field::{Fq, PrimeField};
use snarkrs_groth16::ProveError;
use snarkrs_msm::glv::GlvField;
use snarkrs_msm::xyzz::Xyzz;

use crate::layout::{PackedFq, PackedFq2, PackedG1Affine, PackedG2Affine};
use crate::msm::{PackedXyzzG1, PackedXyzzG2};

/// The coordinate field of one of the two groups, with its curve constant.
pub(crate) trait CurveField: GlvField {
    /// `b` in `y^2 = x^3 + b`, Montgomery form. Built from arkworks on each call, so a
    /// scan takes it once and passes it down.
    fn b() -> Self;
}

impl CurveField for RawFq {
    fn b() -> Self {
        RawFq::from_fq(&snarkrs_field::g1::Config::COEFF_B)
    }
}

impl CurveField for RawFq2 {
    fn b() -> Self {
        RawFq2::from_fq2(&snarkrs_field::g2::Config::COEFF_B)
    }
}

/// A wire struct whose every field element is in `[0, p)`.
///
/// `PackedFq::to_fq` is `new_unchecked` over the device's words, and the raw layer's
/// multiply assumes reduced inputs, so this comes before any arithmetic on a readback: a
/// wrong word that is not a field element is refused here rather than multiplied.
pub(crate) trait Reduced {
    fn reduced(&self) -> bool;
}

impl Reduced for PackedFq {
    fn reduced(&self) -> bool {
        self.to_fq().0 < Fq::MODULUS
    }
}

impl Reduced for PackedFq2 {
    fn reduced(&self) -> bool {
        self.c0.reduced() && self.c1.reduced()
    }
}

impl Reduced for PackedXyzzG1 {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced() && self.zz.reduced() && self.zzz.reduced()
    }
}

impl Reduced for PackedXyzzG2 {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced() && self.zz.reduced() && self.zzz.reduced()
    }
}

impl Reduced for PackedG1Affine {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced()
    }
}

impl Reduced for PackedG2Affine {
    fn reduced(&self) -> bool {
        self.x.reduced() && self.y.reduced()
    }
}

/// Whether `p` is a point of the curve.
///
/// `ZZ == 0` is the identity whatever the other coordinates hold: the GLV ladder leaves
/// the FFT through Jacobian, and a Jacobian identity's `X` and `Y` come along. Otherwise
/// `ZZZ^2 == ZZ^3`, the representation's invariant, and `Y^2 == X^3 + b ZZ^3`, the curve
/// equation at `x = X/ZZ`, `y = Y/ZZZ` with the denominators cleared. Six multiplications.
pub(crate) fn xyzz_on_curve<F: RawField>(p: &Xyzz<F>, b: F) -> bool {
    if p.is_zero() {
        return true;
    }
    let zz3 = p.zz.sqr().mul(p.zz);
    if p.zzz.sqr() != zz3 {
        return false;
    }
    p.y.sqr() == p.x.sqr().mul(p.x).add(b.mul(zz3))
}

/// Whether two XYZZ representatives are the same point: `X1 ZZ2 == X2 ZZ1` and
/// `Y1 ZZZ2 == Y2 ZZZ1`, or both the identity. No inversion.
pub(crate) fn same_point<F: RawField>(a: &Xyzz<F>, b: &Xyzz<F>) -> bool {
    if a.is_zero() || b.is_zero() {
        return a.is_zero() && b.is_zero();
    }
    a.x.mul(b.zz) == b.x.mul(a.zz) && a.y.mul(b.zzz) == b.y.mul(a.zzz)
}

/// The position a spot check recomputes, in `0..n`, from `salt`.
///
/// splitmix64's finaliser over the salt: a chunk index or a dispatch's parameters, so
/// consecutive readbacks probe different positions and the same readback probes the same
/// one on every run and every attempt.
pub(crate) fn spot(n: usize, salt: u64) -> usize {
    debug_assert!(n > 0, "a spot check over nothing");
    let mut z = salt.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z % n as u64) as usize
}

/// The error a readback that fails a check is refused with. A device fault rather than a
/// backend one, so [`crate::cb::with_retry`] re-runs the submission; the reason names the
/// submission and what was wrong, in the spelling the MSM's check uses.
pub(crate) fn wrong(context: &str, what: impl std::fmt::Display) -> ProveError {
    ProveError::Device {
        backend: "metal",
        reason: format!("{context}: {what}: a value read back is not one the GPU wrote"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::{AffineRepr, CurveGroup, Fr, G1Affine, G2Affine};

    fn scaled_g1(k: u64, u: u64) -> Xyzz<RawFq> {
        let p = (G1Affine::generator() * Fr::from(k)).into_affine();
        let u = RawFq::from_fq(&Fq::from(u | 1));
        let zz = u.sqr();
        let zzz = zz.mul(u);
        Xyzz {
            x: RawFq::from_fq(&p.x).mul(zz),
            y: RawFq::from_fq(&p.y).mul(zzz),
            zz,
            zzz,
        }
    }

    fn scaled_g2(k: u64, u: u64) -> Xyzz<RawFq2> {
        let p = (G2Affine::generator() * Fr::from(k)).into_affine();
        let u = RawFq2 {
            c0: RawFq::from_fq(&Fq::from(u | 1)),
            c1: RawFq::ONE,
        };
        let zz = u.sqr();
        let zzz = zz.mul(u);
        Xyzz {
            x: RawFq2::from_fq2(&p.x).mul(zz),
            y: RawFq2::from_fq2(&p.y).mul(zzz),
            zz,
            zzz,
        }
    }

    /// Any representative passes, the identity passes with garbage beside its zero `ZZ`,
    /// and one wrong word in any coordinate fails: the invariant catches `ZZ` and `ZZZ`,
    /// the curve equation `X` and `Y`.
    #[test]
    fn the_curve_check_takes_every_representative_and_refuses_one_wrong_word() {
        let b1 = RawFq::b();
        let b2 = RawFq2::b();
        assert!(xyzz_on_curve(&Xyzz::<RawFq>::ZERO, b1));
        assert!(xyzz_on_curve(
            &Xyzz {
                x: RawFq::ONE,
                y: RawFq::ONE,
                zz: RawFq::ZERO,
                zzz: RawFq::ZERO
            },
            b1
        ));
        for (k, u) in [(1, 1), (7, 3), (11, 0x5eed), (1 << 40, 99)] {
            let p = scaled_g1(k, u);
            assert!(xyzz_on_curve(&p, b1), "G1 [{k}] at u={u}");
            let q = scaled_g2(k, u);
            assert!(xyzz_on_curve(&q, b2), "G2 [{k}] at u={u}");
            let flip = |f: RawFq| RawFq([f.0[0] ^ 1, f.0[1], f.0[2], f.0[3]]);
            assert!(!xyzz_on_curve(&Xyzz { x: flip(p.x), ..p }, b1));
            assert!(!xyzz_on_curve(&Xyzz { y: flip(p.y), ..p }, b1));
            assert!(!xyzz_on_curve(
                &Xyzz {
                    zz: flip(p.zz),
                    ..p
                },
                b1
            ));
            assert!(!xyzz_on_curve(
                &Xyzz {
                    zzz: flip(p.zzz),
                    ..p
                },
                b1
            ));
            let flip2 = |f: RawFq2| RawFq2 {
                c0: flip(f.c0),
                ..f
            };
            assert!(!xyzz_on_curve(&Xyzz { x: flip2(q.x), ..q }, b2));
            assert!(!xyzz_on_curve(
                &Xyzz {
                    zzz: flip2(q.zzz),
                    ..q
                },
                b2
            ));
        }
    }

    #[test]
    fn same_point_sees_through_the_representative() {
        let a = scaled_g1(7, 3);
        let b = scaled_g1(7, 1000);
        let c = scaled_g1(8, 3);
        assert!(same_point(&a, &b));
        assert!(!same_point(&a, &c));
        assert!(!same_point(&a, &Xyzz::ZERO));
        assert!(same_point(&Xyzz::<RawFq>::ZERO, &Xyzz::ZERO));
    }

    #[test]
    fn a_spot_is_in_range_and_moves_with_the_salt() {
        assert_eq!(spot(1, 0), 0);
        assert_eq!(spot(1, 12345), 0);
        let picks: std::collections::BTreeSet<usize> = (0..64).map(|s| spot(1 << 18, s)).collect();
        assert!(picks.len() > 60, "{} distinct of 64", picks.len());
        for s in 0..1000u64 {
            assert!(spot(65, s) < 65);
        }
        assert_eq!(spot(65, 7), spot(65, 7));
    }

    #[test]
    fn a_word_at_or_past_the_modulus_is_not_reduced() {
        let one = PackedFq::from_fq(&Fq::from(1u64));
        assert!(one.reduced());
        let mut v = one;
        v.v = [0xffff_ffff; 8];
        assert!(!v.reduced());
        let mut p = PackedFq::default();
        let m = Fq::MODULUS;
        for (i, limb) in m.0.iter().enumerate() {
            p.v[2 * i] = *limb as u32;
            p.v[2 * i + 1] = (*limb >> 32) as u32;
        }
        assert!(!p.reduced(), "the modulus itself");
        assert!(PackedFq::default().reduced());
    }
}
