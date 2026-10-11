//! `powersoftau prepare phase2`: sections 12 to 15 from sections 2 to 5.
//!
//! This is the expensive command in the whole ceremony, and the reason is one line of
//! ffjavascript. `G.lagrangeEvaluations` (`powersoftau_preparephase2.js:87`) routes to
//! `_fft` with `fnFFTMix = "g1m_fftMix"`, and `build_bn128.js:76` wires that FFT with
//! `opGtimesF = "g1m_timesFr"`. So this is an **inverse FFT over the group**, not over the
//! scalar field: every butterfly twiddle is a full 254-bit point scalar multiplication.
//! [`snarkrs_ntt`] operates on `&mut [Fr]` and cannot express it, which is why the transform
//! lives here rather than there.
//!
//! The output layout is a concatenation of per-power blocks in ascending `p`, each holding
//! `2^p` points, so block `2^k` starts at element `2^k - 1`
//! (`powersoftau_preparephase2.js:56-62`). Section 12 gets one extra block at `power+1`
//! and sections 13 to 15 stop at `power`, which is exactly why `writeHs` can ask section
//! 12 for `2*domainSize` points when `cirPower == power` and the other three are only ever
//! read at the `domainSize` block.
//!
//! Three details that are easy to lose:
//!
//! * **The `power+1` block of section 12 is padded with the point at infinity.** Section 2
//!   holds `2n - 1` points but the block wants `2n`, so snarkjs reads `(nPoints-1)` points
//!   and writes `G1.zeroAffine` into the last slot
//!   (`powersoftau_preparephase2.js:78-81`). That is the *input* to the transform; no
//!   point of the stored block is infinity.
//! * **At `bits == Fr::TWO_ADICITY + 1` the transform splits into two cosets**
//!   (`engine_fft.js:483-497`) and the block comes out as two contiguous halves rather
//!   than interleaved. `writeHs` knows this and reads the second half directly instead of
//!   striding (`zkey_new.js:192-194`), so the two must agree.
//! * **The new header is written with no `ceremonyPower`**, and `writePTauHeader` then
//!   defaults it to `power` (`powersoftau_utils.js:30`). Preparing a `powersoftau
//!   truncate` output therefore *erases* the mark that says it was truncated. Copying the
//!   old `ceremonyPower` through would be the more sensible file, and it would also not be
//!   the file snarkjs writes.
//!
//! # What the transform actually is, once the worker plumbing is removed
//!
//! `_fft` splits the buffer into chunks, runs `fftMix` inside each and `fftJoin` between
//! them, then reassembles. Both are the same decimation-in-time butterfly at different
//! strides (`build_fft.js:657-748` and `:1114-1237`), and the chunking changes nothing
//! about the result, so the whole of it is one in-place radix-2 DIT pass set over a
//! bit-reversed input. The tail is where it gets non-obvious: `fftFinal`
//! (`build_fft.js:1239-1348`) reverses each chunk and scales it by `1/n`, and then the
//! inverse branch of the reassembly rotates the chunks (`engine_fft.js:229-237`). The two
//! together are the textbook identity
//!
//! ```text
//!   ifft(a)[0] = X[0]/n,   ifft(a)[i] = X[n-i]/n
//! ```
//!
//! with `X` the forward transform. Reading either half of that on its own gives an output
//! that is a rotation away from correct, which is a shape a spot check on element 0 does
//! not catch.

use std::path::Path;

use ark_ec::short_weierstrass::{Affine, Projective};
use rayon::prelude::*;
use snarkrs_field::raw::{RawFq, RawFq2};
use snarkrs_field::{
    raw::RawField, FftField, Field, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, One, Zero,
};
use snarkrs_msm::glv::{self, GlvField};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};
use snarkrs_msm::{AccelError, GroupFft};

use crate::ptau::{
    Ptau, LAGRANGE_SECTIONS, PTAU_MAGIC, S_ALPHA_TAU_G1, S_BETA_TAU_G1, S_CONTRIBUTIONS, S_HEADER,
    S_LAGRANGE_ALPHA_TAU_G1, S_LAGRANGE_BETA_TAU_G1, S_LAGRANGE_TAU_G1, S_LAGRANGE_TAU_G2,
    S_TAU_G1, S_TAU_G2,
};
use crate::write::BinFileWriter;
use crate::{ptau, CeremonyError};

/// `createBinFile(newPTauFilename, "ptau", 1, 11)` (`powersoftau_preparephase2.js:29`).
/// Not `PTAU_MAX_VERSION`: that is the highest version this crate will *read*.
const PTAU_VERSION: u32 = 1;

/// Most butterflies per rayon task, which is also the batch [`glv::mul_batch_xyzz`]
/// shares its inversions across: an addition step of that ladder involves the sixth of the
/// batch whose digit is nonzero, so the batch wants to be large. Single thread on an M2
/// Max a G1 multiplication is 35.5 us alone in XYZZ against 44.6, 31.6 and 28.5 us a point
/// in batches of 64, 256 and 1024 (G2: 136 against 70, 57 and 54).
const BUTTERFLIES_PER_TASK: usize = 1024;

/// Below this many butterflies a task multiplies one point at a time instead, which is
/// what a pass too small to give every thread a full batch gets.
const MIN_BATCH: usize = 256;

/// Butterflies per task on the one-point path. Each one is a point scalar multiplication
/// in the tens of microseconds, so this keeps the task count high enough to balance
/// rather than amortising the fork: at 32 a task is on the order of a millisecond.
const SINGLE_PER_TASK: usize = 32;

/// Points per [`batch_to_affine`] inversion. Montgomery's trick trades one inversion for
/// three multiplications a point, so the batch only has to be long enough that the
/// inversion disappears; past a few hundred it buys nothing and costs residency.
const AFFINE_BATCH: usize = 1024;

/// Largest block [`process_section`] hands to the backend together with its smaller
/// neighbours in one [`PrepareCurve::ifft_many`] call. Blocks above it go one at a time,
/// for two reasons that stop at the same size: one pass of such a block is already a
/// dispatch big enough to fill a device on its own, and the co-submitted blocks are
/// resident together, so the cap is what keeps the resident set near the largest block
/// rather than the whole section.
const COSCHEDULE_MAX: usize = 1 << 18;

/// The coset generator ffjavascript uses for the two-coset split, `nqr^2` with `nqr` the
/// smallest quadratic non-residue (`f1field.js:55`, `build_fft.js:79`). On BN254's `Fr`
/// that is `5^2`.
fn shift() -> Fr {
    Fr::from(25u64)
}

/// The two directions [`RawCurve`] does not carry, because the MSM's bucket loop needs
/// neither: a base field element back into raw limbs, and a single inversion.
///
/// Both are per batch, not per point. `invert` runs once per [`AFFINE_BATCH`] points and
/// goes through arkworks rather than [`snarkrs_field::raw`], which has no inversion at all.
pub trait PrepareCurve: RawCurve<RF: GlvField> {
    fn raw(f: &Self::BaseField) -> Self::RF;
    fn invert(f: Self::RF) -> Option<Self::RF>;
    /// The one call the transform cannot express generically: [`GroupFft`] has a separate
    /// entry point per group, the same way [`snarkrs_msm::MsmBackend`] does.
    fn ifft(fft: &dyn GroupFft, a: &mut [Xyzz<Self::RF>]) -> Result<(), AccelError>;
    /// [`PrepareCurve::ifft`] over several blocks in one call, so a device backend can
    /// co-schedule them. See [`GroupFft::ifft_g1_many`].
    fn ifft_many(
        fft: &dyn GroupFft,
        blocks: &mut [&mut [Xyzz<Self::RF>]],
    ) -> Result<(), AccelError>;
    /// The section read and write, which [`Ptau`] and [`BinFileWriter`] also split per
    /// group. Carried here so [`process_section`] is one function rather than a pair.
    fn read_points(src: &Ptau, id: u32, n: usize) -> Result<Vec<Affine<Self>>, CeremonyError>;
    fn write_block(out: &mut BinFileWriter, block: &[Affine<Self>]) -> Result<(), CeremonyError>;
}

impl PrepareCurve for snarkrs_field::g1::Config {
    #[inline(always)]
    fn raw(f: &Fq) -> RawFq {
        RawFq::from_fq(f)
    }
    fn invert(f: RawFq) -> Option<RawFq> {
        f.to_fq().inverse().map(|i| RawFq::from_fq(&i))
    }
    fn ifft(fft: &dyn GroupFft, a: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
        fft.ifft_g1(a)
    }
    fn ifft_many(fft: &dyn GroupFft, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<(), AccelError> {
        fft.ifft_g1_many(blocks)
    }
    fn read_points(src: &Ptau, id: u32, n: usize) -> Result<Vec<G1Affine>, CeremonyError> {
        src.g1_points(id, 0, n)
    }
    fn write_block(out: &mut BinFileWriter, block: &[G1Affine]) -> Result<(), CeremonyError> {
        out.write_g1_slice(block)
    }
}

impl PrepareCurve for snarkrs_field::g2::Config {
    #[inline(always)]
    fn raw(f: &Fq2) -> RawFq2 {
        RawFq2::from_fq2(f)
    }
    fn invert(f: RawFq2) -> Option<RawFq2> {
        f.to_fq2().inverse().map(|i| RawFq2::from_fq2(&i))
    }
    fn ifft(fft: &dyn GroupFft, a: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
        fft.ifft_g2(a)
    }
    fn ifft_many(fft: &dyn GroupFft, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<(), AccelError> {
        fft.ifft_g2_many(blocks)
    }
    fn read_points(src: &Ptau, id: u32, n: usize) -> Result<Vec<G2Affine>, CeremonyError> {
        src.g2_points(id, 0, n)
    }
    fn write_block(out: &mut BinFileWriter, block: &[G2Affine]) -> Result<(), CeremonyError> {
        out.write_g2_slice(block)
    }
}

// ---------------------------------------------------------------------------
// The primitive a GPU kernel replaces
// ---------------------------------------------------------------------------

/// `[k] p`, the butterfly twiddle, and the one operation this whole file exists to run
/// `n log n` times.
///
/// This is `g1m_timesFr` / `g2m_timesFr`, which is `frm_fromMontgomery` followed by
/// `g1m_timesScalar` (`build_bn128.js:60-76`): the scalar leaves Montgomery form and the
/// multiplication is an ordinary variable-base one. Nothing in the transform constrains
/// *how* it is done, only that the answer is `[k] p`, so a Metal or CUDA kernel replaces
/// exactly this function and nothing around it.
///
/// Kept generic over the coordinate field rather than over the curve: G1 and G2 differ
/// only in which field the [`Xyzz`] coordinates live in, and the GPU port wants the same.
/// The CPU path is [`glv::mul`], two interleaved 127-bit wNAF ladders over `p` and its
/// endomorphism image, half the doublings of a plain 254-bit one.
#[inline]
pub fn point_times_fr<F: GlvField>(p: &Xyzz<F>, k: &Fr) -> Xyzz<F> {
    glv::mul(p, k)
}

/// `-p`. Negating `y` alone is correct in XYZZ for the same reason it is in Jacobian, and
/// it leaves the identity (`zz == 0`) alone because its `y` is already zero.
#[inline(always)]
fn negated<F: RawField>(p: &Xyzz<F>) -> Xyzz<F> {
    Xyzz { y: p.y.neg(), ..*p }
}

// ---------------------------------------------------------------------------
// Batch projective to affine
// ---------------------------------------------------------------------------

/// Montgomery's trick over a whole vector: one field inversion per [`AFFINE_BATCH`]
/// points instead of one per point.
///
/// XYZZ needs both `ZZ^-1` and `ZZZ^-1`, which looks like two inversions a point until
/// you invert their product: `t = ZZ*ZZZ`, and then `ZZ^-1 = ZZZ*t^-1` and
/// `ZZZ^-1 = ZZ*t^-1`. So one accumulator entry a point, the same as Jacobian.
pub fn batch_to_affine<P: PrepareCurve>(points: &[Xyzz<P::RF>]) -> Vec<Affine<P>> {
    let mut out = vec![Affine::<P>::identity(); points.len()];
    out.par_chunks_mut(AFFINE_BATCH)
        .zip(points.par_chunks(AFFINE_BATCH))
        .for_each(|(out, batch)| batch_to_affine_serial::<P>(batch, out));
    out
}

pub(crate) fn batch_to_affine_serial<P: PrepareCurve>(
    points: &[Xyzz<P::RF>],
    out: &mut [Affine<P>],
) {
    // `prefix[i]` is the product of every `t` before `i`, skipping the points at infinity
    // so a zero never enters the accumulator.
    let mut prefix = Vec::with_capacity(points.len());
    let mut acc = P::RF::ONE;
    for p in points {
        prefix.push(acc);
        if !p.is_zero() {
            acc = acc.mul(p.zz.mul(p.zzz));
        }
    }
    // A product of nonzero field elements, or `ONE` when every point was infinity.
    let mut inv = P::invert(acc).expect("product of nonzero field elements is invertible");

    for i in (0..points.len()).rev() {
        let p = &points[i];
        if p.is_zero() {
            out[i] = Affine::<P>::identity();
            continue;
        }
        let t = p.zz.mul(p.zzz);
        let t_inv = inv.mul(prefix[i]);
        inv = inv.mul(t);
        let x = p.x.mul(p.zzz).mul(t_inv);
        let y = p.y.mul(p.zz).mul(t_inv);
        out[i] = Affine::new_unchecked(P::field_back(x), P::field_back(y));
    }
}

// ---------------------------------------------------------------------------
// The transform
// ---------------------------------------------------------------------------

/// `ROOTs[j]`, the primitive `2^j`-th root of unity, for `j = 0..=bits`
/// (`build_fft.js:44-63`). ffjavascript builds the table downward from `nqr^rem`; that
/// value is arkworks' `TWO_ADIC_ROOT_OF_UNITY`, and the downward recurrence is the same
/// squaring, so the tables agree entry for entry (pinned in `tests/prepare.rs`).
fn root_table(bits: u32) -> Vec<Fr> {
    let mut roots = vec![Fr::one(); bits as usize + 1];
    let mut root = Fr::TWO_ADIC_ROOT_OF_UNITY;
    for _ in bits..Fr::TWO_ADICITY {
        root.square_in_place();
    }
    for j in (1..=bits as usize).rev() {
        roots[j] = root;
        root.square_in_place();
    }
    roots
}

fn bit_reverse<T>(a: &mut [T], bits: u32) {
    if bits == 0 {
        return;
    }
    for i in 0..a.len() {
        let r = (i as u32).reverse_bits() >> (u32::BITS - bits);
        if i < r as usize {
            a.swap(i, r as usize);
        }
    }
}

/// One `fftMix` pass at exponent `exp`, with `w` the primitive `2^exp`-th root.
///
/// snarkjs walks `W` forward one multiply at a time, which is serial by construction. The
/// same twiddle sequence is recovered here per task from `w^first`, so a task can start
/// anywhere in the group without replaying the ones before it.
///
/// `scale`, when given, multiplies both outputs of every butterfly: the last pass carries
/// the transform's `1/n` this way (see [`ifft`]).
fn mix_pass<F: GlvField>(a: &mut [Xyzz<F>], exp: u32, w: Fr, scale: Option<Fr>) {
    let per_group = 1usize << exp;
    let half = per_group >> 1;
    // Batches only where every thread still gets four of at least MIN_BATCH; a smaller
    // pass keeps the one-point path in tasks of SINGLE_PER_TASK, fine enough that the
    // efficiency cores do not set its pace.
    let fill = a.len() / 2 / (4 * rayon::current_num_threads());
    let batched = fill >= MIN_BATCH;
    let per_task = if batched {
        prev_power_of_two(fill.min(BUTTERFLIES_PER_TASK))
    } else {
        SINGLE_PER_TASK
    };
    let run = |st: &mut glv::BatchState<F>, runs: Vec<(&mut [Xyzz<F>], &mut [Xyzz<F>], usize)>| {
        if batched {
            butterflies(st, runs, w, scale)
        } else {
            for (lo, hi, first) in runs {
                butterflies_each(lo, hi, w, first, scale);
            }
        }
    };
    // Tasks are sized in butterflies, not in groups: the first pass has one butterfly per
    // group and the last has one group, so splitting on either alone leaves a pass serial.
    if half <= per_task {
        let groups_per_task = per_task / half;
        a.par_chunks_mut(per_group * groups_per_task).for_each_init(
            glv::BatchState::default,
            |st, slab| {
                run(
                    st,
                    slab.chunks_mut(per_group)
                        .map(|group| {
                            let (lo, hi) = group.split_at_mut(half);
                            (lo, hi, 0)
                        })
                        .collect(),
                )
            },
        );
    } else {
        // The groups in parallel as well as the tasks inside each: a pass with a few large
        // groups, 64 groups of 4096 say, is otherwise a few tasks at a time.
        a.par_chunks_mut(per_group).for_each(|group| {
            let (lo, hi) = group.split_at_mut(half);
            lo.par_chunks_mut(per_task)
                .zip(hi.par_chunks_mut(per_task))
                .enumerate()
                .for_each_init(glv::BatchState::default, |st, (c, (lo, hi))| {
                    run(st, vec![(lo, hi, c * per_task)])
                });
        });
    }
}

fn prev_power_of_two(x: usize) -> usize {
    1 << (usize::BITS - 1 - x.leading_zeros())
}

/// [`butterflies`] one point at a time in XYZZ, for passes too small to batch.
fn butterflies_each<F: GlvField>(
    lo: &mut [Xyzz<F>],
    hi: &mut [Xyzz<F>],
    w: Fr,
    first: usize,
    scale: Option<Fr>,
) {
    let mut tw = w.pow([first as u64]);
    if let Some(s) = scale {
        tw *= s;
    }
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = if tw.is_one() {
            *v
        } else {
            point_times_fr(v, &tw)
        };
        if let Some(s) = scale {
            *u = point_times_fr(u, &s);
        }
        let mut sum = *u;
        sum.add_assign(&t);
        let mut diff = *u;
        diff.add_assign(&negated(&t));
        *u = sum;
        *v = diff;
        tw *= w;
    }
}

/// `lo[i], hi[i] <- lo[i] + w^(first+i) hi[i], lo[i] - w^(first+i) hi[i]` over every run,
/// both outputs times `s` when `scale` is `Some(s)`.
///
/// The `w^0` case is skipped rather than multiplied. snarkjs does the multiplication, but
/// `[1]P == P`, and at exponent 1 it is every butterfly in the pass. A scaled butterfly
/// is `[s] lo[i] +- [s w^j] hi[i]`: two multiplications for two scaled outputs, where
/// scaling after the pass would cost those two on top of the twiddle.
///
/// Every multiplication of the call goes through one [`glv::mul_batch_xyzz`], whose
/// outputs are affine, so the sum and difference are mixed additions.
fn butterflies<F: GlvField>(
    st: &mut glv::BatchState<F>,
    mut runs: Vec<(&mut [Xyzz<F>], &mut [Xyzz<F>], usize)>,
    w: Fr,
    scale: Option<Fr>,
) {
    let mut vs = Vec::new();
    let mut ks = Vec::new();
    let mut twiddled = Vec::new();
    for (_, hi, first) in &runs {
        let mut tw = w.pow([*first as u64]);
        if let Some(s) = scale {
            tw *= s;
        }
        for v in hi.iter() {
            let one = tw.is_one();
            twiddled.push(!one);
            if !one {
                vs.push(*v);
                ks.push(tw);
            }
            tw *= w;
        }
    }
    let mut ts = vec![Xyzz::ZERO; vs.len()];
    glv::mul_batch_xyzz_in(st, &vs, &ks, &mut ts);
    let mut us = Vec::new();
    if let Some(s) = scale {
        let lo: Vec<Xyzz<F>> = runs
            .iter()
            .flat_map(|(lo, _, _)| lo.iter().copied())
            .collect();
        us = vec![Xyzz::ZERO; lo.len()];
        glv::mul_batch_xyzz_in(st, &lo, &vec![s; lo.len()], &mut us);
    }

    let (mut k, mut j) = (0, 0);
    for (lo, hi, _) in runs.iter_mut() {
        for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
            if scale.is_some() {
                *u = us[j];
            }
            let mut sum = *u;
            let mut diff = *u;
            if twiddled[j] {
                // Affine: a mixed addition each.
                let t = &ts[k];
                k += 1;
                if !t.is_zero() {
                    sum.madd(t.x, t.y);
                    diff.madd(t.x, t.y.neg());
                }
            } else {
                sum.add_assign(v);
                diff.add_assign(&negated(v));
            }
            *u = sum;
            *v = diff;
            j += 1;
        }
    }
}

/// The passes at `exp` and `exp + 1` as one radix-4 pass, with `w` the primitive
/// `2^(exp+1)`-th root; `None` if they are too small to batch, for the caller to run one
/// at a time instead.
///
/// With `h = 2^(exp-1)`, `v = w^j` and `i = w^h` (a fourth root of unity), the two passes
/// take `x0..x3 = a[g+j], a[g+j+h], a[g+j+2h], a[g+j+3h]` to
///
/// ```text
///   A = [v^2] x1,   B = [v] x2 + [v^3] x3,   C = [v i] x2 - [v^3 i] x3
///   x0 + A + B,   x0 - A + C,   x0 + A - B,   x0 - A - C
/// ```
///
/// which is three ladders for four points where the two passes take four: `B` and `C`
/// are sums of two products, and two products under one ladder share its doublings
/// ([`glv::mul_sum_batch_xyzz_in`]). On G1, where the doublings are most of a
/// multiplication, that is about a twelfth less work; on G2, whose additions weigh more,
/// a sum of two costs nearly two products and this gains nothing, so G2 keeps radix 2.
///
/// `scale` multiplies all four outputs, `[s] x0` taking a ladder of its own.
fn mix_pass4<F: GlvField>(a: &mut [Xyzz<F>], exp: u32, w: Fr, scale: Option<Fr>) -> Option<()> {
    let h = 1usize << (exp - 1);
    let fill = a.len() / 4 / (4 * rayon::current_num_threads());
    if F::DIMS != 2 || fill < MIN_BATCH {
        return None;
    }
    let per_task = prev_power_of_two(fill.min(BUTTERFLIES_PER_TASK));
    let i4 = w.pow([h as u64]);
    let run = |st: &mut glv::BatchState<F>, runs: Vec<([&mut [Xyzz<F>]; 4], usize)>| {
        butterflies4(st, runs, w, i4, scale)
    };
    if h <= per_task {
        let groups_per_task = per_task / h;
        a.par_chunks_mut(4 * h * groups_per_task).for_each_init(
            glv::BatchState::default,
            |st, slab| {
                run(
                    st,
                    slab.chunks_mut(4 * h)
                        .map(|g| (quarters(g, h), 0))
                        .collect(),
                )
            },
        );
    } else {
        a.par_chunks_mut(4 * h).for_each(|group| {
            let [q0, q1, q2, q3] = quarters(group, h);
            q0.par_chunks_mut(per_task)
                .zip(q1.par_chunks_mut(per_task))
                .zip(q2.par_chunks_mut(per_task))
                .zip(q3.par_chunks_mut(per_task))
                .enumerate()
                .for_each_init(glv::BatchState::default, |st, (c, (((q0, q1), q2), q3))| {
                    run(st, vec![([q0, q1, q2, q3], c * per_task)])
                });
        });
    }
    Some(())
}

fn quarters<T>(g: &mut [T], h: usize) -> [&mut [T]; 4] {
    let (q0, rest) = g.split_at_mut(h);
    let (q1, rest) = rest.split_at_mut(h);
    let (q2, q3) = rest.split_at_mut(h);
    [q0, q1, q2, q3]
}

/// [`mix_pass4`]'s butterflies over every run, `([x0, x1, x2, x3], first j)`. Every
/// ladder of the call goes through two batches, the single products and the sums.
fn butterflies4<F: GlvField>(
    st: &mut glv::BatchState<F>,
    mut runs: Vec<([&mut [Xyzz<F>]; 4], usize)>,
    w: Fr,
    i4: Fr,
    scale: Option<Fr>,
) {
    let s = scale.unwrap_or(Fr::one());
    let (mut one_p, mut one_k) = (Vec::new(), Vec::new());
    let (mut two_p, mut two_k) = (Vec::new(), Vec::new());
    for ([x0, x1, x2, x3], first) in &runs {
        let mut v = w.pow([*first as u64]);
        for k in 0..x0.len() {
            if v.is_one() && scale.is_none() {
                // j = 0: A = x1 and B = x2 + x3 need no product, C = [i] (x2 - x3) one.
                let mut d = x2[k];
                d.add_assign(&negated(&x3[k]));
                one_p.push(d);
                one_k.push(i4);
            } else {
                let (v2, v3) = (v * v, v * v * v);
                if scale.is_some() {
                    one_p.push(x0[k]);
                    one_k.push(s);
                }
                one_p.push(x1[k]);
                one_k.push(s * v2);
                two_p.extend([x2[k], x3[k], x2[k], x3[k]]);
                two_k.extend([s * v, s * v3, s * v * i4, -(s * v3 * i4)]);
            }
            v *= w;
        }
    }
    let mut one = vec![Xyzz::ZERO; one_p.len()];
    glv::mul_batch_xyzz_in(st, &one_p, &one_k, &mut one);
    let mut two = vec![Xyzz::ZERO; two_p.len() / 2];
    glv::mul_sum_batch_xyzz_in(st, &two_p, &two_k, 2, &mut two);

    // `p + q` and `p - q`, `q` affine (a mixed addition, the identity skipped) or not.
    let pm = |p: &Xyzz<F>, q: &Xyzz<F>, affine: bool| {
        let (mut plus, mut minus) = (*p, *p);
        if affine {
            if !q.is_zero() {
                plus.madd(q.x, q.y);
                minus.madd(q.x, q.y.neg());
            }
        } else {
            plus.add_assign(q);
            minus.add_assign(&negated(q));
        }
        (plus, minus)
    };
    let (mut o, mut t) = (0, 0);
    for ([x0, x1, x2, x3], first) in runs.iter_mut() {
        let mut v = w.pow([*first as u64]);
        for k in 0..x0.len() {
            let (p, m, b, c, affine_b);
            if v.is_one() && scale.is_none() {
                (p, m) = pm(&x0[k], &x1[k], false);
                let mut sum = x2[k];
                sum.add_assign(&x3[k]);
                (b, c, affine_b) = (sum, one[o], false);
                o += 1;
            } else {
                let base = if scale.is_some() {
                    o += 1;
                    one[o - 1]
                } else {
                    x0[k]
                };
                (p, m) = pm(&base, &one[o], true);
                o += 1;
                (b, c, affine_b) = (two[t], two[t + 1], true);
                t += 2;
            }
            let (z0, z2) = pm(&p, &b, affine_b);
            let (z1, z3) = pm(&m, &c, true);
            x0[k] = z0;
            x1[k] = z1;
            x2[k] = z2;
            x3[k] = z3;
            v *= w;
        }
    }
}

/// In-place inverse FFT over the group. `a.len()` must be a power of two no larger than
/// `2^Fr::TWO_ADICITY`; the caller has already checked both.
///
/// This is the whole of what a [`GroupFft`] backend replaces, which is why it is one
/// function taking one slice: every pass is here and nothing else is.
pub(crate) fn ifft<F: GlvField>(a: &mut [Xyzz<F>]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    bit_reverse(a, bits);
    let roots = root_table(bits);
    // `fftFinal` plus the inverse reassembly: scale by `1/n` and reverse everything but
    // element 0. See the module doc for why splitting these apart produces an answer that
    // is off by a rotation and still looks plausible. The scale rides on the last pass,
    // which already multiplies half its inputs, so it costs `n/2` multiplications on top
    // of the transform rather than `n`.
    let size_inv = Fr::from(n as u64)
        .inverse()
        .expect("a power of two is a unit mod r");
    // Passes in pairs ending at the last, through `mix_pass4` where it takes them; an odd
    // count leaves the first pass, which multiplies nothing, on its own.
    let mut exp = 1;
    if bits % 2 == 1 {
        mix_pass(a, 1, roots[1], (bits == 1).then_some(size_inv));
        exp = 2;
    }
    while exp < bits {
        let scale = (exp + 1 == bits).then_some(size_inv);
        if mix_pass4(a, exp, roots[exp as usize + 1], scale).is_none() {
            mix_pass(a, exp, roots[exp as usize], None);
            mix_pass(a, exp + 1, roots[exp as usize + 1], scale);
        }
        exp += 2;
    }
    a[1..].reverse();
}

/// One block, to the backend or to the CPU.
///
/// The crossover is the backend's own number, not this file's: a block below
/// [`GroupFft::min_block`] is a transform a device loses on, and those blocks are the same
/// short handful whatever the power, so routing them home costs nothing to measure and
/// nothing to get slightly wrong.
fn ifft_block<P: PrepareCurve>(
    fft: &dyn GroupFft,
    a: &mut [Xyzz<P::RF>],
) -> Result<(), CeremonyError> {
    if a.len() >= fft.min_block() {
        P::ifft(fft, a)?;
    } else {
        ifft(a);
    }
    Ok(())
}

/// `prepareLagrangeEvaluation` (`build_fft.js:991-1113`), the elementwise preamble of the
/// two-coset split.
///
/// With `m = a.len()/2`, `S = shift^m` and `c = 1/(1 - S)`, for every `i < m`:
///
/// ```text
///   t0[i], t1[i]  <-  c (t1[i] - S t0[i]),   c s^-i (t0[i] - t1[i])
/// ```
///
/// Feeding it `t0[i] = tau^i`, `t1[i] = tau^(m+i)` and running an inverse FFT over each
/// half leaves the Lagrange basis of the double domain `H` union `shift*H` evaluated at
/// `tau`: the first half over `H`, the second over `shift*H`. That is the identity the
/// unit test below checks, since no `power` this crate will ever see reaches the
/// `bits == TWO_ADICITY + 1` gate that puts this on the real path.
fn lagrange_split<F: GlvField>(a: &mut [Xyzz<F>]) {
    let m = a.len() / 2;
    let shift_to_m = shift().pow([m as u64]);
    let s_const = (Fr::one() - shift_to_m)
        .inverse()
        .expect("shift^m is not 1 at m = 2^k for k <= TWO_ADICITY");
    let shift_inv = shift().inverse().expect("shift is nonzero");

    let (t0, t1) = a.split_at_mut(m);
    t0.par_chunks_mut(BUTTERFLIES_PER_TASK)
        .zip(t1.par_chunks_mut(BUTTERFLIES_PER_TASK))
        .enumerate()
        .for_each(|(c, (t0, t1))| {
            let mut w = s_const * shift_inv.pow([(c * BUTTERFLIES_PER_TASK) as u64]);
            for (x, y) in t0.iter_mut().zip(t1.iter_mut()) {
                let mut u = *y;
                u.add_assign(&negated(&point_times_fr(x, &shift_to_m)));
                let mut d = *x;
                d.add_assign(&negated(y));
                *x = point_times_fr(&u, &s_const);
                *y = point_times_fr(&d, &w);
                w *= shift_inv;
            }
        });
}

/// `G.lagrangeEvaluations` (`engine_fft.js:466-518`) over one block.
fn lagrange_evaluations<P: PrepareCurve>(
    points: &[Affine<P>],
    fft: &dyn GroupFft,
) -> Result<Vec<Affine<P>>, CeremonyError> {
    let n = points.len();
    if !n.is_power_of_two() {
        return Err(CeremonyError::BadParams(format!(
            "lagrange evaluations need a power-of-two block, got {n} points"
        )));
    }
    let bits = n.trailing_zeros();
    if bits > Fr::TWO_ADICITY + 1 {
        return Err(CeremonyError::CircuitTooBig(bits));
    }

    let mut a: Vec<Xyzz<P::RF>> = points.par_iter().map(xyzz_from_affine::<P>).collect();
    if bits <= Fr::TWO_ADICITY {
        ifft_block::<P>(fft, &mut a)?;
    } else {
        lagrange_split(&mut a);
        let (t0, t1) = a.split_at_mut(n / 2);
        // Serial rather than `rayon::join`, because the two halves would otherwise submit
        // to the same device at once. This branch needs `bits == TWO_ADICITY + 1` and no
        // ptau this crate will ever see reaches it.
        ifft_block::<P>(fft, t0)?;
        ifft_block::<P>(fft, t1)?;
    }
    Ok(batch_to_affine::<P>(&a))
}

#[inline]
fn xyzz_from_affine<P: PrepareCurve>(p: &Affine<P>) -> Xyzz<P::RF> {
    if p.infinity {
        return Xyzz::ZERO;
    }
    let (x, y) = P::raw_xy(p);
    Xyzz {
        x,
        y,
        zz: P::RF::ONE,
        zzz: P::RF::ONE,
    }
}

/// Jacobian into XYZZ: `ZZ = Z^2`, `ZZZ = Z^3` reproduces both `x = X/Z^2 = X/ZZ` and
/// `y = Y/Z^3 = Y/ZZZ` and satisfies the `ZZ^3 == ZZZ^2` invariant, so no coordinate
/// moves. Two multiplications, no inversion.
#[inline]
fn xyzz_from_projective<P: PrepareCurve>(p: &Projective<P>) -> Xyzz<P::RF> {
    if p.z.is_zero() {
        return Xyzz::ZERO;
    }
    let z = P::raw(&p.z);
    let zz = z.sqr();
    Xyzz {
        x: P::raw(&p.x),
        y: P::raw(&p.y),
        zz,
        zzz: zz.mul(z),
    }
}

// ---------------------------------------------------------------------------
// Public transforms
// ---------------------------------------------------------------------------

/// The Lagrange evaluations of one block of G1 points: an inverse FFT over the group,
/// including the two-coset split at `bits == Fr::TWO_ADICITY + 1`.
///
/// `points.len()` must be a power of two. Above `2^(TWO_ADICITY + 1)` there is no root of
/// unity to build from, and that is [`CeremonyError::CircuitTooBig`].
pub fn lagrange_evaluations_g1(
    points: &[G1Affine],
    fft: &dyn GroupFft,
) -> Result<Vec<G1Affine>, CeremonyError> {
    lagrange_evaluations::<snarkrs_field::g1::Config>(points, fft)
}

/// [`lagrange_evaluations_g1`] over G2, for section 13.
pub fn lagrange_evaluations_g2(
    points: &[G2Affine],
    fft: &dyn GroupFft,
) -> Result<Vec<G2Affine>, CeremonyError> {
    lagrange_evaluations::<snarkrs_field::g2::Config>(points, fft)
}

/// In-place radix-2 inverse FFT over G1, twiddles applied as scalar multiplications.
///
/// Projective in and out because the butterflies are additions and the caller batches back
/// to affine once, not `2^p` times.
pub fn group_ifft_g1(a: &mut [G1Projective], fft: &dyn GroupFft) -> Result<(), CeremonyError> {
    group_ifft::<snarkrs_field::g1::Config>(a, fft)
}

fn group_ifft<P: PrepareCurve>(
    a: &mut [Projective<P>],
    fft: &dyn GroupFft,
) -> Result<(), CeremonyError> {
    let n = a.len();
    if !n.is_power_of_two() {
        return Err(CeremonyError::BadParams(format!(
            "group ifft needs a power-of-two length, got {n}"
        )));
    }
    if n.trailing_zeros() > Fr::TWO_ADICITY {
        return Err(CeremonyError::CircuitTooBig(n.trailing_zeros()));
    }
    let mut work: Vec<Xyzz<P::RF>> = a.par_iter().map(xyzz_from_projective::<P>).collect();
    ifft_block::<P>(fft, &mut work)?;
    a.par_iter_mut()
        .zip(work.par_iter())
        .for_each(|(out, p)| *out = to_projective::<P>(p));
    Ok(())
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

/// Element count of a prepared section, given `power`: `2^(power+2) - 1` for section 12,
/// which carries the extra `power+1` block, and `2^(power+1) - 1` for 13, 14 and 15.
pub fn prepared_element_count(section: u32, power: u32) -> Option<usize> {
    if !LAGRANGE_SECTIONS.contains(&section) {
        return None;
    }
    ptau::expected_section_elements(section, power).map(|n| n as usize)
}

/// Run the whole command: read sections 2 to 5, write 12 to 15, and copy everything else
/// through. The output declares 11 sections.
pub fn prepare_phase2(
    ptau_in: &Path,
    ptau_out: &Path,
    fft: &dyn GroupFft,
) -> Result<(), CeremonyError> {
    crate::write::staged(ptau_out, |stage| prepare_phase2_inner(ptau_in, stage, fft))
}

fn prepare_phase2_inner(
    ptau_in: &Path,
    ptau_out: &Path,
    fft: &dyn GroupFft,
) -> Result<(), CeremonyError> {
    let src = Ptau::open(ptau_in)?;
    let power = src.header().power;
    // Section 12's last block is `2^(power+1)` points, and `lagrangeEvaluations` refuses
    // anything past `2^(TWO_ADICITY+1)`, so `power == 28` is the ceiling.
    if power > Fr::TWO_ADICITY {
        return Err(CeremonyError::CircuitTooBig(power));
    }

    let mut out = BinFileWriter::create(
        ptau_out,
        PTAU_MAGIC,
        PTAU_VERSION,
        crate::phase1::PTAU_PREPARED_SECTIONS,
    )?;

    out.start_section(S_HEADER)?;
    out.write_prime(&crate::q_le())?;
    out.write_u32(power)?;
    // `writePTauHeader` is called with no `ceremonyPower` and defaults it to `power`
    // (`powersoftau_utils.js:30`), so preparing a truncated file un-marks it. Copying
    // `src.header().ceremony_power` here would produce a better file and a different one.
    out.write_u32(power)?;
    out.end_section()?;

    for id in ptau::POINT_SECTIONS.iter().chain(&[S_CONTRIBUTIONS]) {
        out.write_section_verbatim(*id, src.section(*id)?)?;
    }

    process_section::<snarkrs_field::g1::Config>(&src, &mut out, S_TAU_G1, S_LAGRANGE_TAU_G1, fft)?;
    process_section::<snarkrs_field::g2::Config>(&src, &mut out, S_TAU_G2, S_LAGRANGE_TAU_G2, fft)?;
    process_section::<snarkrs_field::g1::Config>(
        &src,
        &mut out,
        S_ALPHA_TAU_G1,
        S_LAGRANGE_ALPHA_TAU_G1,
        fft,
    )?;
    process_section::<snarkrs_field::g1::Config>(
        &src,
        &mut out,
        S_BETA_TAU_G1,
        S_LAGRANGE_BETA_TAU_G1,
        fft,
    )?;

    out.finish()
}

/// The input of block `p`: the first `2^p` points of the section. The one exception is
/// section 12's `power+1` block, whose section holds `2n - 1` points against the `2n` the
/// block wants, so the last input slot is the point at infinity
/// (`powersoftau_preparephase2.js:78-81`).
fn block_input<P: PrepareCurve>(
    src: &Ptau,
    from: u32,
    p: u32,
    power: u32,
) -> Result<Vec<Affine<P>>, CeremonyError> {
    let n = 1usize << p;
    if p <= power {
        P::read_points(src, from, n)
    } else {
        let mut input = P::read_points(src, from, n - 1)?;
        input.push(Affine::<P>::identity());
        Ok(input)
    }
}

/// `processSection` for one pair (`powersoftau_preparephase2.js:52-70`).
///
/// One block per power, ascending, and one extra block for section 2. Blocks above
/// [`COSCHEDULE_MAX`] are transformed and written one at a time so the resident set is
/// the largest block rather than the whole section: at power 28 that is 2^29 points of
/// section 12's `4n - 1`. Blocks at or below it are co-submitted, which costs nothing
/// there (they sum to at most twice the cap) and is what lets a device fill itself from
/// passes too small to fill it alone.
///
/// Blocks below the backend's crossover are a different schedule. Each of them would run
/// on the CPU through `ifft_block` anyway, and run in the gaps between device blocks they
/// serialize with a device that sits idle for them: measured at power 15 on an M2 Max,
/// 1.12 s of a 5.87 s wall. They are independent transforms, so a worker thread runs them
/// while the device blocks proceed, and only the writes keep the ascending block order.
fn process_section<P: PrepareCurve>(
    src: &Ptau,
    out: &mut BinFileWriter,
    from: u32,
    to: u32,
    fft: &dyn GroupFft,
) -> Result<(), CeremonyError> {
    let power = src.header().power;
    let last = if from == S_TAU_G1 { power + 1 } else { power };
    let small = (0..=last)
        .take_while(|&p| (1usize << p) < fft.min_block())
        .count() as u32;
    // Backend blocks up to [`COSCHEDULE_MAX`] go to the backend in one call, so a device
    // can co-dispatch their passes instead of running each under-filled block alone.
    let batched = (small..=last)
        .take_while(|&p| (1usize << p) <= COSCHEDULE_MAX)
        .count() as u32;
    out.start_section(to)?;
    std::thread::scope(|s| -> Result<(), CeremonyError> {
        let mut cpu = Some(
            s.spawn(move || -> Result<Vec<Vec<Affine<P>>>, CeremonyError> {
                (0..small)
                    .map(|p| {
                        lagrange_evaluations::<P>(&block_input::<P>(src, from, p, power)?, fft)
                    })
                    .collect()
            }),
        );
        let mut held: Vec<Vec<Affine<P>>> = Vec::new();

        if batched > 0 {
            let mut works: Vec<Vec<Xyzz<P::RF>>> = (small..small + batched)
                .map(|p| -> Result<Vec<Xyzz<P::RF>>, CeremonyError> {
                    Ok(block_input::<P>(src, from, p, power)?
                        .par_iter()
                        .map(xyzz_from_affine::<P>)
                        .collect())
                })
                .collect::<Result<_, _>>()?;
            let mut refs: Vec<&mut [Xyzz<P::RF>]> =
                works.iter_mut().map(|w| w.as_mut_slice()).collect();
            P::ifft_many(fft, &mut refs)?;
            for w in &works {
                emit_block::<P>(&mut cpu, &mut held, out, batch_to_affine::<P>(w))?;
            }
        }
        for p in small + batched..=last {
            let block = lagrange_evaluations::<P>(&block_input::<P>(src, from, p, power)?, fft)?;
            emit_block::<P>(&mut cpu, &mut held, out, block)?;
        }
        if let Some(h) = cpu.take() {
            for b in h.join().expect("cpu ifft worker panicked")? {
                P::write_block(out, &b)?;
            }
            for b in held.drain(..) {
                P::write_block(out, &b)?;
            }
        }
        Ok(())
    })?;
    out.end_section()
}

/// Write one backend block, or hold it while the sub-crossover worker still owes the
/// blocks that precede it in the file.
///
/// The worker's blocks together are smaller than one backend block, so in practice it
/// finishes during the first backend block or two and `held` never grows to much of the
/// section.
fn emit_block<'scope, P: PrepareCurve>(
    cpu: &mut Option<
        std::thread::ScopedJoinHandle<'scope, Result<Vec<Vec<Affine<P>>>, CeremonyError>>,
    >,
    held: &mut Vec<Vec<Affine<P>>>,
    out: &mut BinFileWriter,
    block: Vec<Affine<P>>,
) -> Result<(), CeremonyError> {
    if let Some(h) = cpu.take_if(|h| h.is_finished()) {
        for b in h.join().expect("cpu ifft worker panicked")? {
            P::write_block(out, &b)?;
        }
    }
    if cpu.is_none() {
        for b in held.drain(..) {
            P::write_block(out, &b)?;
        }
        P::write_block(out, &block)
    } else {
        held.push(block);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_field::{CurveGroup, PrimeGroup};

    /// The Lagrange basis of `H` evaluated at `tau`, the naive way: `L_j(tau)` is the
    /// product over `k != j` of `(tau - h_k) / (h_j - h_k)`. Quadratic, and the point of
    /// it is that it shares no code with the transform.
    fn naive_lagrange(domain: &[Fr], tau: Fr) -> Vec<Fr> {
        (0..domain.len())
            .map(|j| {
                let mut num = Fr::one();
                let mut den = Fr::one();
                for (k, h) in domain.iter().enumerate() {
                    if k != j {
                        num *= tau - h;
                        den *= domain[j] - h;
                    }
                }
                num * den.inverse().unwrap()
            })
            .collect()
    }

    fn powers(tau: Fr, n: usize) -> Vec<G1Affine> {
        let mut acc = Fr::one();
        (0..n)
            .map(|_| {
                let p = (G1Projective::generator() * acc).into_affine();
                acc *= tau;
                p
            })
            .collect()
    }

    /// The plain path: `ifft([tau^i]G)[j]` must be `[L_j(tau)]G` over the `n`-th roots.
    #[test]
    fn ifft_of_the_powers_is_the_lagrange_basis() {
        let tau = Fr::from(123_456_789u64);
        for bits in 0..7u32 {
            let n = 1usize << bits;
            let got = lagrange_evaluations_g1(&powers(tau, n), &crate::CpuGroupFft).unwrap();

            let root = root_table(bits)[bits as usize];
            let mut domain = Vec::with_capacity(n);
            let mut h = Fr::one();
            for _ in 0..n {
                domain.push(h);
                h *= root;
            }
            for (j, l) in naive_lagrange(&domain, tau).into_iter().enumerate() {
                assert_eq!(
                    got[j],
                    (G1Projective::generator() * l).into_affine(),
                    "j={j}"
                );
            }
        }
    }

    /// The two-coset path, which `power <= 28` never reaches on the real command and which
    /// therefore has no snarkjs output to diff against. Run it at `m = 8` instead, where
    /// the double domain `H` union `shift*H` is small enough to interpolate directly.
    #[test]
    fn the_coset_split_is_the_lagrange_basis_of_the_double_domain() {
        let tau = Fr::from(987_654_321u64);
        let bits = 3u32;
        let m = 1usize << bits;

        let mut a: Vec<Xyzz<RawFq>> = powers(tau, 2 * m)
            .iter()
            .map(xyzz_from_affine::<snarkrs_field::g1::Config>)
            .collect();
        lagrange_split(&mut a);
        let (t0, t1) = a.split_at_mut(m);
        ifft(t0);
        ifft(t1);
        let got = batch_to_affine::<snarkrs_field::g1::Config>(&a);

        let root = root_table(bits)[bits as usize];
        let mut domain = Vec::with_capacity(2 * m);
        let mut h = Fr::one();
        for _ in 0..m {
            domain.push(h);
            h *= root;
        }
        for i in 0..m {
            domain.push(domain[i] * shift());
        }
        for (j, l) in naive_lagrange(&domain, tau).into_iter().enumerate() {
            assert_eq!(
                got[j],
                (G1Projective::generator() * l).into_affine(),
                "j={j}"
            );
        }
    }

    /// The radix-4 passes against radix 2 alone, at sizes where they batch: on one thread,
    /// so a block of 2^13 or 2^14 is large enough, with an odd and an even pass count, both
    /// task shapes (groups smaller and larger than a task) and the identity among the
    /// inputs.
    #[test]
    fn radix_4_passes_match_radix_2() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            for bits in [13u32, 14] {
                let n = 1usize << bits;
                let g = xyzz_from_projective::<snarkrs_field::g1::Config>(
                    &(G1Projective::generator() * Fr::from(3u64)),
                );
                let mut acc = g;
                let mut a: Vec<Xyzz<RawFq>> = (0..n)
                    .map(|_| {
                        acc.add_assign(&g);
                        acc
                    })
                    .collect();
                a[5] = Xyzz::ZERO;
                let mut want = a.clone();
                let bits_n = want.len().trailing_zeros();
                bit_reverse(&mut want, bits_n);
                let roots = root_table(bits_n);
                let size_inv = Fr::from(n as u64).inverse().unwrap();
                for exp in 1..=bits_n {
                    let scale = (exp == bits_n).then_some(size_inv);
                    mix_pass(&mut want, exp, roots[exp as usize], scale);
                }
                want[1..].reverse();
                assert!(mix_pass4(&mut a.clone(), 2, roots[3], None).is_some());
                ifft(&mut a);
                let g1 = batch_to_affine::<snarkrs_field::g1::Config>;
                assert_eq!(g1(&a), g1(&want), "bits={bits}");
            }
        });
    }

    /// `point_times_fr` against arkworks, over the corners a random sweep misses: the
    /// identity, the scalar 1, and a scalar whose top wNAF digit carries.
    #[test]
    fn point_times_fr_matches_ark() {
        let base = G1Projective::generator() * Fr::from(7u64);
        let p = xyzz_from_projective::<snarkrs_field::g1::Config>(&base);
        for k in [
            Fr::one(),
            Fr::from(2u64),
            Fr::from(u64::MAX),
            -Fr::one(),
            Fr::TWO_ADIC_ROOT_OF_UNITY,
        ] {
            let got = to_projective::<snarkrs_field::g1::Config>(&point_times_fr(&p, &k));
            assert_eq!(got.into_affine(), (base * k).into_affine());
            assert!(point_times_fr(&Xyzz::<RawFq>::ZERO, &k).is_zero());
        }
    }
}
