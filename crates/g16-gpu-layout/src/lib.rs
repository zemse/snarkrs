//! The packed layouts that cross the Rust/GPU boundary, defined once for every backend.
//!
//! # One definition, three languages
//!
//! Every `#[repr(C)]` type below has a byte-for-byte twin in each GPU backend's kernel
//! source: `g16-metal/src/shaders/bn254_fr.metal` in MSL and
//! `g16-gpu-kernels/src/kernels/bn254_fr.cuh` in CUDA C. Each kernel side carries a
//! `static_assert` on `sizeof`, this side carries a `const` assertion on `size_of`, and
//! each backend keeps a test that greps its own kernel source for the exact constant
//! lines so the copies cannot drift silently. If you change a struct here you are
//! changing a wire format shared by two GPUs; change all three in the same commit.
//!
//! This crate exists so that Metal and CUDA cannot disagree about what a point is. The
//! two backends run on different hardware with different compilers, and a benchmark that
//! compares them is only meaningful if they are fed bit-identical inputs.
//!
//! # Why repacking is not optional
//!
//! Measured on this workspace's arkworks version: `size_of::<ark_ec::G1Affine>()` is 72,
//! not 64, because the affine type carries an `infinity: bool` plus padding, and
//! `size_of::<ark_ec::G2Affine>()` is 136, not 128. Neither `ark_ff::Fp` nor the affine
//! types are `repr(C)`, so their field order is not even guaranteed. Byte-casting a
//! `Vec<G1Affine>` into a device buffer hands the GPU a 72-byte stride while the kernel
//! reads 64, and every point after the first is garbage. There is no way around an
//! explicit repack, so this module makes the repack the only path.
//!
//! # Montgomery form, and the asymmetry that silently breaks proofs
//!
//! arkworks stores an `Fp` internally in Montgomery form with `R = 2^256`, which is
//! exactly the radix both GPU CIOS routines use. So:
//!
//! * [`PackedFr`] and [`PackedFq`] hold **Montgomery** limbs, copied straight out of
//!   `Fp::0.0` with no conversion on either side. This is what field kernels want: the
//!   NTT, the coset shift and `H = A*B - C` are all multiplications and additions, and
//!   Montgomery form is closed under both.
//! * [`PackedScalar`] holds **standard** limbs, via `into_bigint()`. This is what an MSM
//!   wants, because Pippenger slices a scalar into window digits and a window digit of a
//!   Montgomery representative is a digit of `a*R mod n`, which is a different number.
//!
//! Getting this backwards produces a proof that is wrong by a factor of `R` and fails
//! verification with nothing else to go on, so the two types are deliberately distinct
//! and neither converts into the other.
use core::mem::{align_of, size_of};

use ark_ff::{BigInt, PrimeField};
use g16_field::{Fq, Fq2, Fr, G1Affine, G2Affine};

/// Limbs per field element. 32-bit limbs, not 64.
///
/// Justification, in order of weight:
///
/// 1. Measured on this M2 Max during the scouting phase: an 8 x u32 CIOS Montgomery
///    multiply that widens both factors to `ulong` only at the multiply-accumulate site
///    ran at 4.51 G mul/s, against 3.49 G mul/s for the same algorithm written with
///    `mulhi` and 32-bit carry chains. The `ulong` form is 29% faster because the Metal
///    compiler lowers `(ulong)x * (ulong)y` with both factors 32-bit-widened onto the
///    native 32x32->64 path, whereas a genuine 64x64 product is emulated.
/// 2. Both Apple-targeted references agree on 8 x u32: zkmopro/gpu-acceleration
///    (`shader/misc/types.metal`, `NUM_LIMBS 8`, `LOG_LIMB_SIZE 32`) and
///    zkonduit/metal-msm (`UnsignedInteger<8>`). The one project that uses 4 x u64
///    (lambdaworks) hand-emulates a `u128` class underneath, paying the emulated 64-bit
///    multiply on every limb product.
/// 3. `R = 2^(32 * 8) = 2^256` then coincides exactly with arkworks' own Montgomery
///    radix, which is what makes the zero-conversion packing above possible.
pub const LIMBS: usize = 8;

/// BN254 scalar field modulus `r`, little-endian 32-bit limbs.
///
/// `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`
pub const FR_MODULUS: [u32; LIMBS] = [
    0xf000_0001,
    0x43e1_f593,
    0x79b9_7091,
    0x2833_e848,
    0x8181_585d,
    0xb850_45b6,
    0xe131_a029,
    0x3064_4e72,
];

/// `-r^{-1} mod 2^32`, the CIOS per-limb reduction multiplier for `Fr`.
///
/// Checked against `ark-ff` in [`tests::montgomery_constants_match_ark`] rather than
/// trusted: a wrong `N0` produces a multiply that is wrong for almost every input, which
/// this crate's GPU-vs-host test would catch, but a wrong `N0` that is *right* for the
/// handful of small vectors a human tries by hand is exactly how this bug ships.
pub const FR_N0: u32 = 0xefff_ffff;

/// BN254 base field modulus `q`, little-endian 32-bit limbs.
pub const FQ_MODULUS: [u32; LIMBS] = [
    0xd87c_fd47,
    0x3c20_8c16,
    0x6871_ca8d,
    0x9781_6a91,
    0x8181_585d,
    0xb850_45b6,
    0xe131_a029,
    0x3064_4e72,
];

/// `-q^{-1} mod 2^32`. Equals 3834012553, the same value zkmopro and zkonduit ship for
/// BN254 `Fq`, which is a cheap independent cross-check on the derivation.
pub const FQ_N0: u32 = 0xe486_6389;

/// BN254 scalar field element, 32 bytes, **Montgomery form**, little-endian 32-bit limbs.
///
/// MSL twin: `struct Fr { uint v[8]; }` in `shaders/bn254_fr.metal`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedFr {
    pub v: [u32; LIMBS],
}

/// BN254 scalar as an integer in `[0, r)`, 32 bytes, **standard form**, little-endian
/// 32-bit limbs. For MSM window decomposition only. See the module docs.
///
/// MSL twin: `struct Scalar { uint v[8]; }`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedScalar {
    pub v: [u32; LIMBS],
}

/// BN254 base field element, 32 bytes, **Montgomery form**, little-endian 32-bit limbs.
///
/// MSL twin: `struct Fq { uint v[8]; }`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedFq {
    pub v: [u32; LIMBS],
}

/// `Fq2 = Fq[u]/(u^2 + 1)`, 64 bytes, `c0` then `c1`, each Montgomery.
///
/// MSL twin: `struct Fq2 { Fq c0; Fq c1; }`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedFq2 {
    pub c0: PackedFq,
    pub c1: PackedFq,
}

/// G1 affine point, exactly 64 bytes: `x` then `y`, no flag word.
///
/// # How infinity is signalled
///
/// The all-zero encoding, `x == 0 && y == 0`, means the point at infinity. This is
/// unambiguous rather than a convention: BN254 G1 is `y^2 = x^3 + 3`, so `(0, 0)` fails
/// the curve equation (`0 != 3`) and can never be a real point. Three things follow, and
/// all three are why the sentinel beats a 65th byte:
///
/// * The stride stays a power of two, 64 bytes, so a base vector is naturally aligned and
///   a thread's load of one point is two 32-byte segments rather than a straddle.
/// * A freshly allocated Metal buffer is already zero-filled, so an accumulator array
///   starts at infinity with no memset kernel and no host staging buffer.
/// * The test is `x | y == 0` over 16 words, which is branch-free.
///
/// This matters in practice and is not a theoretical case: snarkjs zkeys really do
/// contain points at infinity in the A, B and C query vectors, and the reference Metal
/// MSM implementations that ignore the arkworks `infinity` flag lift `(0, 0)` into a live
/// non-identity projective point and poison the bucket it lands in.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedG1Affine {
    pub x: PackedFq,
    pub y: PackedFq,
}

/// G2 affine point, exactly 128 bytes: `x` then `y`, each an [`PackedFq2`]. Infinity is
/// the all-zero encoding, for the same reason as [`PackedG1Affine`] (BN254 G2 is
/// `y^2 = x^3 + 3/(9 + u)`, whose constant term is nonzero, so `(0, 0)` is off-curve).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedG2Affine {
    pub x: PackedFq2,
    pub y: PackedFq2,
}

// The wire format. A future arkworks bump, a stray `#[repr(align)]`, or someone adding a
// field cannot get past these.
const _: () = {
    assert!(size_of::<PackedFr>() == 32);
    assert!(size_of::<PackedScalar>() == 32);
    assert!(size_of::<PackedFq>() == 32);
    assert!(size_of::<PackedFq2>() == 64);
    assert!(size_of::<PackedG1Affine>() == 64);
    assert!(size_of::<PackedG2Affine>() == 128);
    // MSL's `uint` is 4-byte aligned and Metal requires buffer offsets to be a multiple
    // of 4. Anything larger here would mean Rust inserted padding the shader does not
    // know about.
    assert!(align_of::<PackedFr>() == 4);
    assert!(align_of::<PackedG1Affine>() == 4);
    assert!(align_of::<PackedG2Affine>() == 4);
    // arkworks' backing integer is 4 x u64 = 32 bytes for both BN254 fields. If this ever
    // stops holding, `from_*`/`to_*` below are reinterpreting the wrong number of words.
    assert!(size_of::<BigInt<4>>() == 32);
};

#[inline]
fn split(limbs: [u64; 4]) -> [u32; LIMBS] {
    let mut out = [0u32; LIMBS];
    for (i, w) in limbs.iter().enumerate() {
        out[2 * i] = *w as u32;
        out[2 * i + 1] = (*w >> 32) as u32;
    }
    out
}

#[inline]
fn join(v: [u32; LIMBS]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, w) in out.iter_mut().enumerate() {
        *w = u64::from(v[2 * i]) | (u64::from(v[2 * i + 1]) << 32);
    }
    out
}

impl PackedFr {
    pub const ZERO: Self = Self { v: [0; LIMBS] };

    /// Montgomery limbs, taken verbatim from arkworks' internal representation. No
    /// conversion happens on either side of the boundary.
    #[inline]
    pub fn from_fr(x: &Fr) -> Self {
        Self { v: split(x.0 .0) }
    }

    /// Inverse of [`Self::from_fr`]. Uses `new_unchecked`, which stores the limbs as the
    /// Montgomery representative rather than converting into it. Going through
    /// `Fr::from_bigint` here instead would multiply by `R` a second time and every value
    /// would come back wrong by a factor of `R`.
    #[inline]
    pub fn to_fr(&self) -> Fr {
        Fr::new_unchecked(BigInt::new(join(self.v)))
    }

    pub fn pack_slice(xs: &[Fr]) -> Vec<Self> {
        xs.iter().map(Self::from_fr).collect()
    }

    /// Packs directly into caller memory, which on this platform is meant to be the
    /// pointer from `MTLBuffer::contents()`. Unified memory means there is no second copy
    /// after this, so this is the whole upload.
    ///
    /// Panics if the destination is not exactly as long as the source, because a short
    /// destination would leave the tail of a device buffer holding whatever was there
    /// before, and a proof built on it would fail verification with no other symptom.
    pub fn pack_into(xs: &[Fr], out: &mut [Self]) {
        assert_eq!(xs.len(), out.len(), "packed destination length mismatch");
        for (dst, src) in out.iter_mut().zip(xs) {
            *dst = Self::from_fr(src);
        }
    }

    pub fn unpack_slice(xs: &[Self]) -> Vec<Fr> {
        xs.iter().map(Self::to_fr).collect()
    }
}

impl PackedScalar {
    pub const ZERO: Self = Self { v: [0; LIMBS] };

    /// Standard form, `x` as an integer in `[0, r)`.
    #[inline]
    pub fn from_fr(x: &Fr) -> Self {
        Self {
            v: split(x.into_bigint().0),
        }
    }

    /// `None` if the limbs are not a canonical residue, which for a scalar is a real
    /// possibility (unlike a Montgomery representative, which is canonical by
    /// construction) and must not be papered over with a silent reduction.
    #[inline]
    pub fn to_fr(&self) -> Option<Fr> {
        Fr::from_bigint(BigInt::new(join(self.v)))
    }

    pub fn pack_slice(xs: &[Fr]) -> Vec<Self> {
        xs.iter().map(Self::from_fr).collect()
    }

    pub fn pack_into(xs: &[Fr], out: &mut [Self]) {
        assert_eq!(xs.len(), out.len(), "packed destination length mismatch");
        for (dst, src) in out.iter_mut().zip(xs) {
            *dst = Self::from_fr(src);
        }
    }
}

impl PackedFq {
    pub const ZERO: Self = Self { v: [0; LIMBS] };

    #[inline]
    pub fn from_fq(x: &Fq) -> Self {
        Self { v: split(x.0 .0) }
    }

    #[inline]
    pub fn to_fq(&self) -> Fq {
        Fq::new_unchecked(BigInt::new(join(self.v)))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.v.iter().fold(0u32, |a, b| a | b) == 0
    }
}

impl PackedFq2 {
    pub const ZERO: Self = Self {
        c0: PackedFq::ZERO,
        c1: PackedFq::ZERO,
    };

    #[inline]
    pub fn from_fq2(x: &Fq2) -> Self {
        Self {
            c0: PackedFq::from_fq(&x.c0),
            c1: PackedFq::from_fq(&x.c1),
        }
    }

    #[inline]
    pub fn to_fq2(&self) -> Fq2 {
        Fq2::new(self.c0.to_fq(), self.c1.to_fq())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

impl PackedG1Affine {
    /// The point at infinity, and also what a zeroed Metal buffer already contains.
    pub const INFINITY: Self = Self {
        x: PackedFq::ZERO,
        y: PackedFq::ZERO,
    };

    #[inline]
    pub fn from_affine(p: &G1Affine) -> Self {
        // Read the flag, do not read x and y and hope. An arkworks identity has
        // x = 0, y = 0, infinity = true, so for the identity specifically the two paths
        // agree; for anything else the flag is the only correct source of truth and
        // reading it costs one branch at prepare time, once per base, forever.
        if p.infinity {
            Self::INFINITY
        } else {
            Self {
                x: PackedFq::from_fq(&p.x),
                y: PackedFq::from_fq(&p.y),
            }
        }
    }

    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// Does not check the curve equation: these come back from a kernel that was handed
    /// on-curve inputs, and re-checking every point would cost more than the kernel.
    #[inline]
    pub fn to_affine(&self) -> G1Affine {
        if self.is_infinity() {
            G1Affine::identity()
        } else {
            G1Affine::new_unchecked(self.x.to_fq(), self.y.to_fq())
        }
    }

    pub fn pack_slice(ps: &[G1Affine]) -> Vec<Self> {
        ps.iter().map(Self::from_affine).collect()
    }

    pub fn pack_into(ps: &[G1Affine], out: &mut [Self]) {
        assert_eq!(ps.len(), out.len(), "packed destination length mismatch");
        for (dst, src) in out.iter_mut().zip(ps) {
            *dst = Self::from_affine(src);
        }
    }
}

impl PackedG2Affine {
    pub const INFINITY: Self = Self {
        x: PackedFq2::ZERO,
        y: PackedFq2::ZERO,
    };

    #[inline]
    pub fn from_affine(p: &G2Affine) -> Self {
        if p.infinity {
            Self::INFINITY
        } else {
            Self {
                x: PackedFq2::from_fq2(&p.x),
                y: PackedFq2::from_fq2(&p.y),
            }
        }
    }

    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    #[inline]
    pub fn to_affine(&self) -> G2Affine {
        if self.is_infinity() {
            G2Affine::identity()
        } else {
            G2Affine::new_unchecked(self.x.to_fq2(), self.y.to_fq2())
        }
    }

    pub fn pack_slice(ps: &[G2Affine]) -> Vec<Self> {
        ps.iter().map(Self::from_affine).collect()
    }

    pub fn pack_into(ps: &[G2Affine], out: &mut [Self]) {
        assert_eq!(ps.len(), out.len(), "packed destination length mismatch");
        for (dst, src) in out.iter_mut().zip(ps) {
            *dst = Self::from_affine(src);
        }
    }
}

/// Marker for a type whose in-memory bytes are exactly what the GPU should see.
///
/// # Safety
///
/// Implementors must be `#[repr(C)]`, contain nothing but `u32` (directly or through
/// other `Packed` types), have no padding, and treat every bit pattern as valid. All of
/// that is enforced for the types below by the `const` block above plus inspection: each
/// one is a `[u32; 8]` or a tuple of them, so there is nowhere for padding to hide.
pub unsafe trait Packed: Copy {}

unsafe impl Packed for PackedFr {}
unsafe impl Packed for PackedScalar {}
unsafe impl Packed for PackedFq {}
unsafe impl Packed for PackedFq2 {}
unsafe impl Packed for PackedG1Affine {}
unsafe impl Packed for PackedG2Affine {}

/// Byte view of a packed slice, for `MTLDevice::newBuffer*`.
pub fn as_bytes<T: Packed>(items: &[T]) -> &[u8] {
    // SAFETY: `Packed` promises no padding and no invalid bit patterns, so every byte of
    // the slice is initialised and readable. The result borrows `items`, so the lifetime
    // is the source's.
    unsafe {
        core::slice::from_raw_parts(items.as_ptr().cast::<u8>(), size_of::<T>() * items.len())
    }
}

/// Mutable byte view, for reading a kernel's output back out of a shared buffer.
///
/// # Safety
///
/// The caller must not write a value that violates `T`'s invariants. For every type in
/// this module there are none, so this is safe in practice for all of them; it stays
/// `unsafe` so that a future `Packed` type with an invariant cannot quietly inherit it.
pub unsafe fn as_bytes_mut<T: Packed>(items: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            items.as_mut_ptr().cast::<u8>(),
            size_of::<T>() * items.len(),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ark_ff::{One, Zero};
    use g16_field::{CurveGroup, G1Projective, G2Projective, PrimeGroup};

    use crate::testrng::SplitMix64;

    /// `R mod r`, computed independently of arkworks. This is what the Montgomery
    /// representative of 1 must be, and asserting it pins down that `LIMBS = 8` with
    /// `R = 2^256` really is arkworks' radix and not a coincidence that happens to
    /// round-trip.
    const FR_R_MOD_N: [u32; LIMBS] = [
        0x4fff_fffb,
        0xac96_341c,
        0x9f60_cd29,
        0x36fc_7695,
        0x7879_462e,
        0x666e_a36f,
        0x9a07_df2f,
        0x0e0a_77c1,
    ];

    /// Every magic number in this file, re-derived from or checked against `ark-ff`
    /// rather than trusted.
    #[test]
    fn montgomery_constants_match_ark() {
        for (name, modulus, n0, ark_mod) in [
            ("Fr", FR_MODULUS, FR_N0, Fr::MODULUS.0),
            ("Fq", FQ_MODULUS, FQ_N0, Fq::MODULUS.0),
        ] {
            assert_eq!(
                join(modulus),
                ark_mod,
                "{name}: modulus limbs disagree with ark-ff"
            );
            // n0 == -m^{-1} mod 2^32. Everything above limb 0 is irrelevant mod 2^32, so
            // this is the whole condition, not a weakening of it.
            let low = u64::from(modulus[0]).wrapping_mul(u64::from(n0)) as u32;
            assert_eq!(low, u32::MAX, "{name}: N0 is not -m^-1 mod 2^32");
        }
        // The value both Apple-targeted reference implementations ship for BN254 Fq,
        // reached here by an independent derivation.
        assert_eq!(FQ_N0, 3_834_012_553);
    }

    #[test]
    fn fr_round_trips_through_montgomery_limbs() {
        let mut rng = SplitMix64(0xC0FF_EE00);
        for x in [Fr::zero(), Fr::one(), -Fr::one(), Fr::from(2u64)]
            .into_iter()
            .chain((0..4096).map(|_| rng.next_fr()))
        {
            assert_eq!(PackedFr::from_fr(&x).to_fr(), x);
            assert_eq!(PackedScalar::from_fr(&x).to_fr(), Some(x));
        }
    }

    /// The two Fr encodings must genuinely differ, otherwise the module docs are a lie
    /// and someone will use them interchangeably.
    #[test]
    fn montgomery_and_standard_encodings_are_not_the_same() {
        let one = Fr::one();
        assert_eq!(PackedScalar::from_fr(&one).v, [1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(PackedFr::from_fr(&one).v, FR_R_MOD_N);
        assert_ne!(PackedFr::from_fr(&one).v, PackedScalar::from_fr(&one).v);
    }

    /// A scalar buffer that is not a canonical residue must be rejected, not silently
    /// reduced. `r` itself is the obvious case a fuzzer or a corrupt buffer produces.
    #[test]
    fn a_non_canonical_scalar_is_rejected() {
        assert_eq!(PackedScalar { v: FR_MODULUS }.to_fr(), None);
        assert_eq!(
            PackedScalar {
                v: [u32::MAX; LIMBS]
            }
            .to_fr(),
            None
        );
    }

    fn rand_g1(rng: &mut SplitMix64) -> G1Affine {
        (G1Projective::generator() * rng.next_fr()).into_affine()
    }

    fn rand_g2(rng: &mut SplitMix64) -> G2Affine {
        (G2Projective::generator() * rng.next_fr()).into_affine()
    }

    #[test]
    fn g1_packs_to_64_bytes_and_round_trips_including_infinity() {
        let mut rng = SplitMix64(7);
        assert_eq!(as_bytes(&[PackedG1Affine::INFINITY]).len(), 64);

        let inf = G1Affine::identity();
        assert!(PackedG1Affine::from_affine(&inf).is_infinity());
        assert_eq!(PackedG1Affine::from_affine(&inf).to_affine(), inf);

        for _ in 0..256 {
            let p = rand_g1(&mut rng);
            let packed = PackedG1Affine::from_affine(&p);
            assert!(!packed.is_infinity(), "a random point is not infinity");
            assert_eq!(packed.to_affine(), p);
        }
        // A real point can never collide with the infinity sentinel, because (0, 0) is
        // off-curve for y^2 = x^3 + 3.
        assert!(!G1Affine::new_unchecked(Fq::zero(), Fq::zero()).is_on_curve());
    }

    #[test]
    fn g2_packs_to_128_bytes_and_round_trips_including_infinity() {
        let mut rng = SplitMix64(9);
        assert_eq!(as_bytes(&[PackedG2Affine::INFINITY]).len(), 128);

        let inf = G2Affine::identity();
        assert!(PackedG2Affine::from_affine(&inf).is_infinity());
        assert_eq!(PackedG2Affine::from_affine(&inf).to_affine(), inf);

        for _ in 0..64 {
            let p = rand_g2(&mut rng);
            let packed = PackedG2Affine::from_affine(&p);
            assert!(!packed.is_infinity());
            assert_eq!(packed.to_affine(), p);
        }
        assert!(!G2Affine::new_unchecked(Fq2::zero(), Fq2::zero()).is_on_curve());
    }

    /// The arkworks strides this module exists to avoid. If these ever become 64 and 128
    /// the repack is still correct, just no longer load-bearing; if they change to some
    /// third value the comment at the top of this file is stale.
    #[test]
    fn ark_affine_strides_are_still_the_ones_documented() {
        assert_eq!(size_of::<G1Affine>(), 72);
        assert_eq!(size_of::<G2Affine>(), 136);
        assert_ne!(size_of::<G1Affine>(), size_of::<PackedG1Affine>());
    }

    #[test]
    fn packed_slices_have_the_declared_stride() {
        assert_eq!(as_bytes(&vec![PackedG1Affine::INFINITY; 5]).len(), 5 * 64);
        assert_eq!(as_bytes(&vec![PackedG2Affine::INFINITY; 5]).len(), 5 * 128);
        assert_eq!(as_bytes(&vec![PackedFr::ZERO; 5]).len(), 5 * 32);
        // and the bytes really are the limbs, low word first
        let x = PackedFr::from_fr(&Fr::one());
        assert_eq!(&as_bytes(&[x])[..4], &FR_R_MOD_N[0].to_le_bytes());
    }

    /// Packing must be a pure function of the input, and the bulk paths must agree with
    /// the single-element one. A `pack_into` that wrote a stale tail would be invisible
    /// until a proof failed to verify.
    #[test]
    fn packing_is_a_pure_function() {
        let mut rng = SplitMix64(11);
        let xs: Vec<Fr> = (0..64).map(|_| rng.next_fr()).collect();
        assert_eq!(PackedFr::pack_slice(&xs), PackedFr::pack_slice(&xs));
        let mut out = vec![PackedFr::ZERO; xs.len()];
        PackedFr::pack_into(&xs, &mut out);
        assert_eq!(out, PackedFr::pack_slice(&xs));
        assert_eq!(PackedFr::unpack_slice(&out), xs);
    }
}

/// A deterministic PRNG for tests in the GPU backend crates.
///
/// SplitMix64, Steele et al. Dependency-free on purpose: the backends that consume this
/// crate exist to have very few dependencies, and a test RNG is not worth one.
pub mod testrng {
    use g16_field::Fr;

    pub struct SplitMix64(pub u64);

    impl SplitMix64 {
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform on `[0, r)` up to the usual negligible bias of reducing 256 random
        /// bits mod a 254-bit modulus.
        pub fn next_fr(&mut self) -> Fr {
            use ark_ff::PrimeField;
            let mut b = [0u8; 32];
            for c in b.chunks_mut(8) {
                c.copy_from_slice(&self.next_u64().to_le_bytes());
            }
            Fr::from_le_bytes_mod_order(&b)
        }
    }
}
