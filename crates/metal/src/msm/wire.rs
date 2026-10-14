//! The structs the MSM kernels read and write, mirrored field for field from
//! `msm.metal`, and the conversion of the XYZZ points they hand back.

use ark_ff::Zero;
use snarkrs_field::{G1Projective, G2Projective};

use crate::layout::{Packed, PackedFq, PackedFq2};

/// Mirrors `struct MsmParams` in `msm.metal`. All counts, no pointers, passed by
/// `setBytes` so it never needs a buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MsmParams {
    pub(super) n: u32,
    pub(super) c: u32,
    pub(super) n_windows: u32,
    pub(super) n_buckets: u32,
    pub(super) cap: u32,
    pub(super) scalar_off: u32,
    pub(super) base_off: u32,
    pub(super) ones_groups: u32,
    pub(super) slice_len: u32,
    pub(super) slices: u32,
    pub(super) reduce_groups: u32,
    pub(super) merge_wide_base: u32,
    pub(super) merge_wide_rows: u32,
    pub(super) separate_ones: u32,
    pub(super) dummy_rows: u32,
    pub(super) seg_first: u32,
    pub(super) skip: u32,
}

/// A G1 point in extended Jacobian coordinates, as the kernels write it: 128 bytes,
/// `x`, `y`, `zz`, `zzz`, with `x/zz` and `y/zzz` the affine coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PackedXyzzG1 {
    pub x: PackedFq,
    pub y: PackedFq,
    pub zz: PackedFq,
    pub zzz: PackedFq,
}

/// The G2 twin, 256 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PackedXyzzG2 {
    pub x: PackedFq2,
    pub y: PackedFq2,
    pub zz: PackedFq2,
    pub zzz: PackedFq2,
}

const _: () = {
    assert!(core::mem::size_of::<MsmParams>() == 68);
    assert!(core::mem::size_of::<PackedXyzzG1>() == 128);
    assert!(core::mem::size_of::<PackedXyzzG2>() == 256);
};

unsafe impl Packed for PackedXyzzG1 {}
unsafe impl Packed for PackedXyzzG2 {}

/// XYZZ to arkworks' Jacobian, with no field inversion.
///
/// XYZZ carries the invariant `ZZ^3 = ZZZ^2`, so setting the Jacobian `Z = ZZZ` gives
/// `Z^2 = ZZ^3` and the point `(X * ZZ^2, Y * ZZ^3, ZZZ)` has
/// `x = X*ZZ^2 / ZZ^3 = X/ZZ` and `y = Y*ZZ^3 / ZZZ^3 = Y/ZZZ`, which is exactly the
/// XYZZ point. Three multiplications, versus two inversions if this went through affine,
/// and there are hundreds of these per proof.
impl PackedXyzzG1 {
    pub fn to_projective(&self) -> G1Projective {
        let zz = self.zz.to_fq();
        if zz.is_zero() {
            return G1Projective::zero();
        }
        let x = self.x.to_fq();
        let y = self.y.to_fq();
        let zzz = self.zzz.to_fq();
        let zz2 = zz * zz;
        G1Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz)
    }
}

impl PackedXyzzG2 {
    pub fn to_projective(&self) -> G2Projective {
        let zz = self.zz.to_fq2();
        if zz.is_zero() {
            return G2Projective::zero();
        }
        let x = self.x.to_fq2();
        let y = self.y.to_fq2();
        let zzz = self.zzz.to_fq2();
        let zz2 = zz * zz;
        G2Projective::new_unchecked(x * zz2, y * zz2 * zz, zzz)
    }
}
