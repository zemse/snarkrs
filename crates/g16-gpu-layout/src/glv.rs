//! The GLV twiddle table behind the group inverse FFT, decomposed once for every backend.
//!
//! `ptau prepare`'s FFT kernels run their ladder as `[k1]P + [k2]phi(P)`, and the lattice
//! that splits `k` is the host's problem: the twiddles are powers of a fixed root, so the
//! whole table is decomposed here and a thread never runs the lattice at all. The kernel
//! twins that read the layout are `pt_mul_glv` in `g16-metal/src/shaders/fft.metal` and in
//! `g16-gpu-kernels/src/kernels/fft.cu`, and each backend keeps a test that greps its own
//! kernel source for `GLV_WORDS`, `GLV_RECODE_BITS` and `FQ_BETA`, so the three copies of
//! the constants cannot drift silently.
//!
//! One `beta`, two lattices. The endomorphism `phi(x, y) = (beta x, y)` acts as
//! multiplication by an eigenvalue of `x^2 + x + 1` mod r, and G1 and G2 are the two
//! eigenspaces: `phi` is `[lambda]` on G1 and `[lambda^2]` on G2. So the kernel applies
//! one endomorphism and the host selects the lattice per group, through the `GLVConfig`
//! type parameter below. [`tests::beta_is_one_cube_root_and_the_eigenvalues_are_two`]
//! asserts that split rather than assuming it, because swapping the two lattices is a
//! change nothing else in either backend would notice.

use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ff::{BigInteger, PrimeField};
use g16_field::{FftField, Field, Fr};
use rayon::prelude::*;

use crate::{Packed, LIMBS};

/// A scalar already through the GLV lattice: `k == +-k1 + lambda * (+-k2)`, 36 bytes,
/// both magnitudes **standard form** and under `2^127`.
///
/// `k[0..4]` is `|k1|` and `k[4..8]` is `|k2|`, four 32-bit limbs each rather than eight,
/// which is the whole point: a half-width magnitude is half the ladder. `sign` carries
/// bit 0 for `k1 < 0` and bit 1 for `k2 < 0`, and a signed ladder spends nothing on
/// either, since negating a point is negating one coordinate.
///
/// Kernel twin: nine consecutive words read out of a flat `uint` array, `GLV_WORDS` in
/// `shaders/fft.metal` and `kernels/fft.cu`. There is no struct on either kernel side
/// because the twiddle table is bound as a flat word array, the same way
/// [`crate::PackedScalar`] tables are.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PackedGlv {
    pub k: [u32; LIMBS],
    pub sign: u32,
}

const _: () = {
    assert!(core::mem::size_of::<PackedGlv>() == 36);
    assert!(core::mem::align_of::<PackedGlv>() == 4);
};

// SAFETY: `repr(C)`, nine `u32`, no padding, every bit pattern valid.
unsafe impl Packed for PackedGlv {}

/// `first * W^i` for `i < 2^(bits-1)`, with `W` the primitive `2^bits`-th root, in
/// STANDARD form. `first` is `Fr::ONE` for the table every mix pass shares and `1/n` for
/// the fused final pass's copy, which is how the scaling costs no scalar of its own.
///
/// One plain table serves every mix pass of the block. snarkjs uses a different root per pass,
/// `roots[exp]` the primitive `2^exp`-th root (`build_fft.js:44-63`, mirrored by
/// `prepare::root_table`), and `roots[exp] == W^(2^(bits-exp))`, so pass `exp`'s twiddle
/// `roots[exp]^j` is this table at `j << (bits - exp)`. The largest index a pass reads is
/// `(2^(exp-1) - 1) << (bits - exp)`, which is under `2^(bits-1)`, so the table is exactly
/// half the block and no pass runs off it.
///
/// The de-Montgomery step is here and not in a kernel. `g1m_timesFr` is
/// `frm_fromMontgomery` then a variable-base multiplication (`build_bn128.js:60-76`), and a
/// window digit of a Montgomery representative is a digit of `a*R mod r`, which is a
/// different number: getting it backwards produces points that are wrong by a factor of R
/// and a file that verifies against nothing.
///
/// Each entry then goes through the GLV lattice ([`decompose`]) so a thread never does.
/// That is the whole reason GLV is affordable in this kernel and not in the ceremony
/// ladders next door, whose scalars are only known on the device.
///
/// The walk is a serial recurrence, so it is chunked the way `ifft` (prepare.rs:273)
/// chunks its own: each chunk raises `root` to its own start index and walks from there.
/// The chunking is not for the `Fr` multiply but for the decomposition, which is 1.1 us on
/// this M2 Max against the multiply's tens of nanoseconds. A power-15 command decomposes
/// 237,568 twiddles, so serially that is a quarter of a second with the device idle for
/// all of it; measured on the pool it is 52 ms.
pub fn twiddle_table<C: GLVConfig<ScalarField = Fr>>(bits: u32, first: Fr) -> Vec<PackedGlv> {
    /// Entries a rayon task walks, and small on purpose. A section builds these tables
    /// while a worker thread is already running the sub-crossover blocks on the same pool,
    /// so the chunk has to be small enough that a table still spreads across what is left:
    /// at 4,096 a power-15 prepare ran 2.384 s and at 1,024 it ran 2.280 s, medians of four
    /// and six warm runs. From there it flattens. Every table in a power-15 command costs
    /// 63 ms at 1,024 and 52 ms at 256, which is 0.5% of the run and inside its wall-clock
    /// noise, so this is a floor rather than a tuned number: the `root^start` a chunk
    /// begins with is about 30 `Fr` operations against 256 decompositions, and going lower
    /// starts paying for that.
    const CHUNK: usize = 256;

    let half = 1usize << (bits - 1);
    let mut root = Fr::TWO_ADIC_ROOT_OF_UNITY;
    for _ in bits..Fr::TWO_ADICITY {
        root.square_in_place();
    }
    let mut out = vec![PackedGlv::default(); half];
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let mut w = first * root.pow([(ci * CHUNK) as u64]);
            for slot in chunk.iter_mut() {
                *slot = decompose::<C>(&w);
                w *= root;
            }
        });
    out
}

/// One twiddle through the GLV lattice: `k == +-k1 + lambda * (+-k2)`, both magnitudes
/// under `2^127`, in the layout `pt_mul_glv` reads.
///
/// arkworks' own decomposition, not one written here. It is `ark_bn254`'s LLL-reduced
/// basis and `num_bigint`, which is why it costs 1.1 us; what it buys is that the lattice
/// constants and the rounding are not this project's to get subtly wrong, and the
/// eigenvalue that goes with them is the one `ark_bn254` publishes per group.
///
/// The bound is asserted, not assumed. Only four limbs of each magnitude are written, so
/// a decomposition that ever came back wider would be silently truncated into a different
/// scalar and a wrong point. The widest seen over 200,000 scalars of each group is 127
/// bits, which is `|n11| + |n21|` and not the `(|n11| + |n21|)/2` a
/// round-to-nearest would give: arkworks divides a negative product and `num_bigint`
/// truncates toward zero, so the rounding error is up to a whole basis vector.
pub fn decompose<C: GLVConfig<ScalarField = Fr>>(k: &Fr) -> PackedGlv {
    let ((pos1, k1), (pos2, k2)) = C::scalar_decomposition(*k);
    let mut out = PackedGlv {
        k: [0; 8],
        sign: u32::from(!pos1) | (u32::from(!pos2) << 1),
    };
    for (half, mag) in [(0usize, k1), (4, k2)] {
        let b = mag.into_bigint().0;
        assert!(
            b[2] == 0 && b[3] == 0 && b[1] >> 63 == 0,
            "GLV magnitude is {} bits, past the 127 the table holds",
            mag.into_bigint().num_bits()
        );
        for (i, w) in b[..2].iter().enumerate() {
            out.k[half + 2 * i] = *w as u32;
            out.k[half + 2 * i + 1] = (*w >> 32) as u32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testrng::SplitMix64;
    use g16_field::{AffineRepr, CurveGroup, Fq, G1Affine, G2Affine};

    /// The endomorphism the kernels apply is one `Fq` multiply on X, and both groups get
    /// the same `beta`. What differs is the eigenvalue: `x^2 + x + 1` has two roots mod r
    /// and G1 and G2 are its two eigenspaces, so G2's `lambda` is G1's squared and the two
    /// decompose against different lattices. Asserted here rather than assumed, because
    /// swapping the two lattices is a change nothing else in either backend would notice.
    #[test]
    fn beta_is_one_cube_root_and_the_eigenvalues_are_two() {
        use ark_ff::AdditiveGroup;

        let beta = <g16_field::g1::Config as GLVConfig>::ENDO_COEFFS[0];
        assert_ne!(beta, Fq::ONE);
        assert_eq!(
            beta * beta * beta,
            Fq::ONE,
            "beta is not a cube root of one"
        );
        let beta2 = <g16_field::g2::Config as GLVConfig>::ENDO_COEFFS[0];
        assert_eq!(beta2.c0, beta, "G2 takes a different beta");
        assert_eq!(beta2.c1, Fq::ZERO, "G2's beta is not in the prime field");

        let l1 = <g16_field::g1::Config as GLVConfig>::LAMBDA;
        let l2 = <g16_field::g2::Config as GLVConfig>::LAMBDA;
        assert_eq!(l1 * l1, l2, "G2's eigenvalue is not G1's squared");
        assert_eq!(l1 * l1 * l1, Fr::ONE);

        let p1 = G1Affine::generator();
        let mut e1 = p1;
        e1.x *= beta;
        assert_eq!(e1, (p1 * l1).into_affine(), "phi is not [lambda] on G1");
        let p2 = G2Affine::generator();
        let mut e2 = p2;
        e2.x *= beta2;
        assert_eq!(e2, (p2 * l2).into_affine(), "phi is not [lambda^2] on G2");
    }

    /// [`decompose`]'s output means what the kernels read it as, over the scalars most
    /// likely to break it: the ends of the field, the twiddles themselves, and the `1/n`
    /// the fused final pass folds in. `k1` and `k2` are checked as SIGNED values against
    /// the group's own eigenvalue, and the width is checked against the four limbs the
    /// table holds, which is the assertion that stops a truncated magnitude.
    #[test]
    fn a_decomposed_twiddle_still_means_the_same_scalar() {
        fn check<C: GLVConfig<ScalarField = Fr>>(name: &str, ks: &[Fr]) {
            let mut widest = 0u32;
            for k in ks {
                let g = decompose::<C>(k);
                let mag = |half: usize| {
                    let mut b = [0u64; 4];
                    for (i, w) in b[..2].iter_mut().enumerate() {
                        *w =
                            u64::from(g.k[half + 2 * i]) | (u64::from(g.k[half + 2 * i + 1]) << 32);
                    }
                    Fr::from(ark_ff::BigInt(b))
                };
                let k1 = mag(0);
                let k2 = mag(4);
                widest = widest
                    .max(k1.into_bigint().num_bits())
                    .max(k2.into_bigint().num_bits());
                let s1 = if g.sign & 1 == 0 { k1 } else { -k1 };
                let s2 = if g.sign & 2 == 0 { k2 } else { -k2 };
                assert_eq!(
                    s1 + C::LAMBDA * s2,
                    *k,
                    "{name}: decomposition of {k} is not it"
                );
                assert!(
                    g.sign < 4,
                    "{name}: sign word has bits the kernels do not read"
                );
            }
            assert!(
                widest <= 127,
                "{name}: {widest} bits is past the table's 127"
            );
        }

        let mut ks = vec![
            Fr::from(0u64),
            Fr::ONE,
            -Fr::ONE,
            -Fr::from(2u64),
            <g16_field::g1::Config as GLVConfig>::LAMBDA,
            <g16_field::g2::Config as GLVConfig>::LAMBDA,
            Fr::TWO_ADIC_ROOT_OF_UNITY,
        ];
        // The 1/n of every block a ptau file up to power 28 contains, and a walk of the
        // deepest root, which is what the table is actually made of.
        for bits in 0..=28u32 {
            ks.push(
                Fr::from(1u64 << bits)
                    .inverse()
                    .expect("a power of two is a unit mod r"),
            );
        }
        let mut w = Fr::ONE;
        for _ in 0..4096 {
            ks.push(w);
            w *= Fr::TWO_ADIC_ROOT_OF_UNITY;
        }
        let mut rng = SplitMix64(0x91d_c0de);
        for _ in 0..4096 {
            ks.push(rng.next_fr());
        }
        check::<g16_field::g1::Config>("G1", &ks);
        check::<g16_field::g2::Config>("G2", &ks);
    }
}
