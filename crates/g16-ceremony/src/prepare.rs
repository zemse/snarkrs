//! `powersoftau prepare phase2`: sections 12 to 15 from sections 2 to 5.
//!
//! This is the expensive command in the whole ceremony, and the reason is one line of
//! ffjavascript. `G.lagrangeEvaluations` (`powersoftau_preparephase2.js:87`) routes to
//! `_fft` with `fnFFTMix = "g1m_fftMix"`, and `build_bn128.js:76` wires that FFT with
//! `opGtimesF = "g1m_timesFr"`. So this is an **inverse FFT over the group**, not over the
//! scalar field: every butterfly twiddle is a full 254-bit point scalar multiplication.
//! [`g16_ntt`] operates on `&mut [Fr]` and cannot express it, which is why the transform
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
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{
    raw::RawField, BigInteger, FftField, Field, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine,
    G2Projective, One, PrimeField, Zero,
};
use g16_msm::xyzz::{to_projective, RawCurve, Xyzz};
use rayon::prelude::*;

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

/// wNAF window for the twiddle multiplication. The table is `2^(w-2)` odd multiples, so
/// the cost is `2^(w-2) - 1` additions to build it against a density of one nonzero digit
/// every `w + 1` bits. Over BN254's 254-bit scalars the minimum sits flat across 4, 5 and
/// 6; 5 costs 8 table entries and about 42 additions on top of the 254 doublings.
const WNAF_WIDTH: usize = 5;

/// Odd multiples `1P, 3P, ..., (2^(w-1)-1)P`.
const WNAF_TABLE_LEN: usize = 1 << (WNAF_WIDTH - 2);

/// Butterflies per rayon task. Each one is a point scalar multiplication in the tens of
/// microseconds, so this is chosen to keep the task count high enough to balance rather
/// than to amortise the fork: at 32 a task is on the order of a millisecond.
const BUTTERFLIES_PER_TASK: usize = 32;

/// Points per [`batch_to_affine`] inversion. Montgomery's trick trades one inversion for
/// three multiplications a point, so the batch only has to be long enough that the
/// inversion disappears; past a few hundred it buys nothing and costs residency.
const AFFINE_BATCH: usize = 1024;

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
/// goes through arkworks rather than [`g16_field::raw`], which has no inversion at all.
pub trait PrepareCurve: RawCurve {
    fn raw(f: &Self::BaseField) -> Self::RF;
    fn invert(f: Self::RF) -> Option<Self::RF>;
}

impl PrepareCurve for g16_field::g1::Config {
    #[inline(always)]
    fn raw(f: &Fq) -> RawFq {
        RawFq::from_fq(f)
    }
    fn invert(f: RawFq) -> Option<RawFq> {
        f.to_fq().inverse().map(|i| RawFq::from_fq(&i))
    }
}

impl PrepareCurve for g16_field::g2::Config {
    #[inline(always)]
    fn raw(f: &Fq2) -> RawFq2 {
        RawFq2::from_fq2(f)
    }
    fn invert(f: RawFq2) -> Option<RawFq2> {
        f.to_fq2().inverse().map(|i| RawFq2::from_fq2(&i))
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
/// Kept generic over [`RawField`] rather than over the curve: G1 and G2 differ only in
/// which field the [`Xyzz`] coordinates live in, and the GPU port wants the same.
#[inline]
pub fn point_times_fr<F: RawField>(p: &Xyzz<F>, k: &Fr) -> Xyzz<F> {
    if p.is_zero() {
        return Xyzz::ZERO;
    }
    // wNAF digits are odd and in `-(2^(w-1)-1) ..= 2^(w-1)-1`, so `|d| >> 1` indexes the
    // odd-multiple table directly.
    let digits = k
        .into_bigint()
        .find_wnaf(WNAF_WIDTH)
        .expect("WNAF_WIDTH is in 2..64");

    let two_p = p.dbl();
    let mut table = [Xyzz::<F>::ZERO; WNAF_TABLE_LEN];
    table[0] = *p;
    for i in 1..WNAF_TABLE_LEN {
        let mut next = table[i - 1];
        next.add_assign(&two_p);
        table[i] = next;
    }

    let mut acc = Xyzz::<F>::ZERO;
    for &d in digits.iter().rev() {
        acc = acc.dbl();
        match d {
            0 => {}
            d if d > 0 => acc.add_assign(&table[(d as usize) >> 1]),
            d => acc.add_assign(&negated(&table[((-d) as usize) >> 1])),
        }
    }
    acc
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

fn batch_to_affine_serial<P: PrepareCurve>(points: &[Xyzz<P::RF>], out: &mut [Affine<P>]) {
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
fn mix_pass<F: RawField>(a: &mut [Xyzz<F>], exp: u32, w: Fr) {
    let per_group = 1usize << exp;
    let half = per_group >> 1;
    // Tasks are sized in butterflies, not in groups: the first pass has one butterfly per
    // group and the last has one group, so splitting on either alone leaves a pass serial.
    let groups_per_task = BUTTERFLIES_PER_TASK.div_ceil(half).max(1);

    a.par_chunks_mut(per_group * groups_per_task)
        .for_each(|slab| {
            for group in slab.chunks_mut(per_group) {
                let (lo, hi) = group.split_at_mut(half);
                if half <= BUTTERFLIES_PER_TASK {
                    butterflies(lo, hi, w, 0);
                } else {
                    lo.par_chunks_mut(BUTTERFLIES_PER_TASK)
                        .zip(hi.par_chunks_mut(BUTTERFLIES_PER_TASK))
                        .enumerate()
                        .for_each(|(c, (lo, hi))| butterflies(lo, hi, w, c * BUTTERFLIES_PER_TASK));
                }
            }
        });
}

/// `lo[i], hi[i] <- lo[i] + w^(first+i) hi[i], lo[i] - w^(first+i) hi[i]`.
///
/// The `w^0` case is skipped rather than multiplied. snarkjs does the multiplication, but
/// `[1]P == P`, and at exponent 1 it is every butterfly in the pass.
fn butterflies<F: RawField>(lo: &mut [Xyzz<F>], hi: &mut [Xyzz<F>], w: Fr, first: usize) {
    let mut tw = w.pow([first as u64]);
    for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = if tw.is_one() {
            *v
        } else {
            point_times_fr(v, &tw)
        };
        let mut sum = *u;
        sum.add_assign(&t);
        let mut diff = *u;
        diff.add_assign(&negated(&t));
        *u = sum;
        *v = diff;
        tw *= w;
    }
}

/// In-place inverse FFT over the group. `a.len()` must be a power of two no larger than
/// `2^Fr::TWO_ADICITY`; the caller has already checked both.
fn ifft<F: RawField>(a: &mut [Xyzz<F>]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    bit_reverse(a, bits);
    let roots = root_table(bits);
    for exp in 1..=bits {
        mix_pass(a, exp, roots[exp as usize]);
    }

    // `fftFinal` plus the inverse reassembly, as one step: scale by `1/n` and reverse
    // everything but element 0. See the module doc for why splitting these apart produces
    // an answer that is off by a rotation and still looks plausible.
    let size_inv = Fr::from(n as u64)
        .inverse()
        .expect("a power of two is a unit mod r");
    a.par_iter_mut()
        .for_each(|p| *p = point_times_fr(p, &size_inv));
    a[1..].reverse();
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
fn lagrange_split<F: RawField>(a: &mut [Xyzz<F>]) {
    let m = a.len() / 2;
    let shift_to_m = shift().pow([m as u64]);
    let s_const = (Fr::one() - shift_to_m)
        .inverse()
        .expect("shift has order q-1, so shift^m is never 1 for m below it");
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
        ifft(&mut a);
    } else {
        lagrange_split(&mut a);
        let (t0, t1) = a.split_at_mut(n / 2);
        rayon::join(|| ifft(t0), || ifft(t1));
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
pub fn lagrange_evaluations_g1(points: &[G1Affine]) -> Result<Vec<G1Affine>, CeremonyError> {
    lagrange_evaluations::<g16_field::g1::Config>(points)
}

/// [`lagrange_evaluations_g1`] over G2, for section 13.
pub fn lagrange_evaluations_g2(points: &[G2Affine]) -> Result<Vec<G2Affine>, CeremonyError> {
    lagrange_evaluations::<g16_field::g2::Config>(points)
}

/// In-place radix-2 inverse FFT over G1, twiddles applied as scalar multiplications.
///
/// Projective in and out because the butterflies are additions and the caller batches back
/// to affine once, not `2^p` times.
pub fn group_ifft_g1(a: &mut [G1Projective]) -> Result<(), CeremonyError> {
    group_ifft::<g16_field::g1::Config>(a)
}

/// [`group_ifft_g1`] over G2.
pub fn group_ifft_g2(a: &mut [G2Projective]) -> Result<(), CeremonyError> {
    group_ifft::<g16_field::g2::Config>(a)
}

fn group_ifft<P: PrepareCurve>(a: &mut [Projective<P>]) -> Result<(), CeremonyError> {
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
    ifft(&mut work);
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
pub fn prepare_phase2(ptau_in: &Path, ptau_out: &Path) -> Result<(), CeremonyError> {
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

    process_section_g1(&src, &mut out, S_TAU_G1, S_LAGRANGE_TAU_G1)?;
    process_section_g2(&src, &mut out, S_TAU_G2, S_LAGRANGE_TAU_G2)?;
    process_section_g1(&src, &mut out, S_ALPHA_TAU_G1, S_LAGRANGE_ALPHA_TAU_G1)?;
    process_section_g1(&src, &mut out, S_BETA_TAU_G1, S_LAGRANGE_BETA_TAU_G1)?;

    out.finish()
}

/// `processSection` for a G1 pair (`powersoftau_preparephase2.js:52-70`).
///
/// One block per power, ascending, and one extra block for section 2. Blocks are
/// transformed and written one at a time so the resident set is the largest block rather
/// than the whole section: at power 28 the difference is 32 GB against 1 TB.
fn process_section_g1(
    src: &Ptau,
    out: &mut BinFileWriter,
    from: u32,
    to: u32,
) -> Result<(), CeremonyError> {
    let power = src.header().power;
    out.start_section(to)?;
    for p in 0..=power {
        let block = lagrange_evaluations_g1(&src.g1_points(from, 0, 1usize << p)?)?;
        out.write_g1_slice(&block)?;
    }
    if from == S_TAU_G1 {
        // Section 2 holds `2n - 1` points and this block wants `2n`, so the last input
        // slot is the point at infinity (`powersoftau_preparephase2.js:78-81`).
        let n = 1usize << (power + 1);
        let mut input = src.g1_points(from, 0, n - 1)?;
        input.push(G1Affine::identity());
        let block = lagrange_evaluations_g1(&input)?;
        out.write_g1_slice(&block)?;
    }
    out.end_section()
}

/// [`process_section_g1`] for section 3, the only G2 pair, which has no `power+1` block.
fn process_section_g2(
    src: &Ptau,
    out: &mut BinFileWriter,
    from: u32,
    to: u32,
) -> Result<(), CeremonyError> {
    let power = src.header().power;
    out.start_section(to)?;
    for p in 0..=power {
        let block = lagrange_evaluations_g2(&src.g2_points(from, 0, 1usize << p)?)?;
        out.write_g2_slice(&block)?;
    }
    out.end_section()
}

#[cfg(test)]
mod tests {
    use super::*;
    use g16_field::{CurveGroup, PrimeGroup};

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
            let got = lagrange_evaluations_g1(&powers(tau, n)).unwrap();

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
            .map(xyzz_from_affine::<g16_field::g1::Config>)
            .collect();
        lagrange_split(&mut a);
        let (t0, t1) = a.split_at_mut(m);
        ifft(t0);
        ifft(t1);
        let got = batch_to_affine::<g16_field::g1::Config>(&a);

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

    /// `point_times_fr` against arkworks, over the corners a random sweep misses: the
    /// identity, the scalar 1, and a scalar whose top wNAF digit carries.
    #[test]
    fn point_times_fr_matches_ark() {
        let base = G1Projective::generator() * Fr::from(7u64);
        let p = xyzz_from_projective::<g16_field::g1::Config>(&base);
        for k in [
            Fr::one(),
            Fr::from(2u64),
            Fr::from(u64::MAX),
            -Fr::one(),
            Fr::TWO_ADIC_ROOT_OF_UNITY,
        ] {
            let got = to_projective::<g16_field::g1::Config>(&point_times_fr(&p, &k));
            assert_eq!(got.into_affine(), (base * k).into_affine());
            assert!(point_times_fr(&Xyzz::<RawFq>::ZERO, &k).is_zero());
        }
    }
}
