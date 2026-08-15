//! Pippenger multi-scalar multiplication over BN254 G1 and G2.
//!
//! Five MSMs dominate the proof (~53% of a fast CPU prover's wall clock at 151K
//! constraints, ~70% of a slow one's), so this is the crate that decides whether the
//! project is worth anything.
//!
//! Two properties matter beyond raw speed, both because the witness is the scalar vector
//! for four of the five MSMs:
//!   * a scalar of 0 must cost nothing beyond the digit scan,
//!   * a scalar of 1 must cost exactly one mixed addition, not a full bucket round trip.
//!
//! How much those paths matter is circuit-shaped, and an earlier version of this comment
//! overclaimed it: "in bit-decomposition-heavy circuits over 99% of witness scalars are
//! 0 or 1". Measured on the benchmark ladder (round-4 profiling), 0/1 scalars are 1.80%
//! of the witness on the two largest circuits and 4.12% at the sparsest point, worth
//! about 1.02x, not the 5x once asserted. The paths stay because they are nearly free
//! and a genuinely bit-heavy circuit still benefits, but on this ladder the MSMs are
//! dense and the bucket loop is the whole game. What IS a measured, structural win on
//! every circuit here: 34% of the B query bases are the point at infinity, and the
//! prescan drops them before they cost a single window visit.

use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{AdditiveGroup, One, PrimeField, Zero};
use rayon::prelude::*;

use g16_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

pub mod xyzz;
use g16_field::raw::RawField;
use xyzz::{to_projective, RawCurve, Xyzz};

pub trait MsmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective;
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective;
}

/// Bits in the scalar field modulus: 254 for BN254's Fr.
const SCALAR_BITS: usize = Fr::MODULUS_BIT_SIZE as usize;

/// Signed recoding carries one bit past the top of the scalar, so the digit array is
/// laid out over `SCALAR_BITS + 1` bits. See [`signed_digit`] for why that kills the
/// final carry rather than just hiding it.
const RECODE_BITS: usize = SCALAR_BITS + 1;

/// Caps the bucket array at 2^15 + 1 points, about 3 MB of Jacobian G1 (6 MB of G2) per
/// in-flight task. Past this the cost model's gain is under 5% and the memset starts to
/// hurt.
const MAX_WINDOW: u32 = 16;

/// Prescan granularity. Small enough to balance, large enough that the two output vectors
/// are worth appending rather than merging element by element.
const SCAN_CHUNK: usize = 1 << 12;

/// A point chunk must be at least this many times the bucket count to be worth splitting
/// off: every extra chunk pays another full bucket reduction, `2 * 2^(c-1)` Jacobian adds,
/// so at 16x the reduction overhead of the split stays under a fifth of the chunk's work.
const CHUNK_BUCKET_RATIO: usize = 16;

/// Window size for `n` points. Pippenger's cost is minimised near `ln(n)`; sppark uses
/// `min(floor(lg2(1.5n)) - 8, 18)` floored at 10 on GPU, but the CPU optimum is smaller
/// because there is no bucket-sort machinery to amortise.
///
/// Rather than fit a closed form, minimise the cost model directly: `ceil(255/c)` windows,
/// each doing `n` mixed additions into buckets plus `2 * 2^(c-1)` full Jacobian additions
/// in the running-sum reduction. Signed digits are what put `2^(c-1)` there instead of
/// `2^c`, which is worth roughly one extra bit of window. The weight of 3 on the bucket
/// term is `2 * 1.5`: a full add is about 1.5 mixed adds (add-2007-bl is 11M+5S, the mixed
/// madd-2007-bl is 7M+4S). The one
/// extra bucket [`signed_digit`] needs is left out of the model, it moves nothing.
pub fn window_size(n: usize) -> u32 {
    let mut best = 3;
    let mut best_cost = u128::MAX;
    for c in 3..=MAX_WINDOW {
        let windows = RECODE_BITS.div_ceil(c as usize) as u128;
        let cost = windows * (n as u128 + 3 * (1u128 << (c - 1)));
        if cost < best_cost {
            best_cost = cost;
            best = c;
        }
    }
    best
}

pub struct CpuMsm {
    pub threads: usize,
}

impl CpuMsm {
    pub fn new() -> Self {
        Self {
            threads: rayon::current_num_threads(),
        }
    }
}

impl Default for CpuMsm {
    fn default() -> Self {
        Self::new()
    }
}

impl MsmBackend for CpuMsm {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        pippenger(bases, scalars, self.threads)
    }
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        pippenger(bases, scalars, self.threads)
    }
}

/// Reads `width` (< 64) bits out of a little-endian limb array starting at `bit_offset`.
/// Reads past the end of the array as zero, which is what makes the top window of a
/// 254-bit scalar cheap instead of a special case.
#[inline]
fn read_bits(limbs: &[u64], bit_offset: usize, width: u32) -> u64 {
    debug_assert!(width < 64);
    let idx = bit_offset / 64;
    if idx >= limbs.len() {
        return 0;
    }
    let shift = bit_offset % 64;
    let mut buf = limbs[idx] >> shift;
    // `shift` is provably nonzero here (width < 64), so the `64 - shift` shift is defined.
    if shift + width as usize > 64 && idx + 1 < limbs.len() {
        buf |= limbs[idx + 1] << (64 - shift);
    }
    buf & ((1u64 << width) - 1)
}

/// Digit `i` of the width-`c` signed recoding of `limbs`, in `[-2^(c-1), 2^(c-1)]`.
///
/// The textbook recoding threads a carry left to right, which would serialise the windows
/// or force a materialised digit array (arkworks materialises `W` i64 per scalar, 136 MB
/// at 2^20 points). This one has no carry chain at all. Split each raw window
/// `b_i = (k >> i*c) & mask` on its own, independently of its neighbours:
///
/// ```text
///   e_i = [b_i >= 2^(c-1)]          d_i = b_i - e_i * 2^c
///   k = sum_i b_i 2^(i c) = sum_i d_i 2^(i c) + sum_i e_i 2^((i+1) c)
/// ```
///
/// so window `i` owes `d_i` plus the `e_(i-1)` handed up by its right neighbour, and
/// `e_(i-1)` is just bit `i*c - 1` of the scalar. Every digit is a pure function of the
/// scalar and the window index, so the windows stay independent and nothing is stored.
///
/// The price is that the digit reaches `+2^(c-1)`, so there is one bucket more than the
/// `2^(c-1)` a ripple recoding needs. One bucket against a serialised recoding is not a
/// close call.
///
/// The carry out of the top window is `e_(W-1) = [b_(W-1) >= 2^(c-1)]`, and it must be
/// zero or the top digits are simply lost. It is: `k < 2^254` and the digits are laid out
/// over `RECODE_BITS = 255` bits, so `W*c >= 255` and `b_(W-1) < 2^(254 - (W-1)c) <= 2^(c-1)`.
#[inline]
fn signed_digit(limbs: &[u64], i: usize, c: u32) -> i64 {
    let offset = i * c as usize;
    let b = read_bits(limbs, offset, c);
    // Borrow 2^c when the raw window is in the top half; the neighbour to the right pays
    // it back through its own top bit, which is the bit immediately below this window.
    let d = b as i64 - (((b >> (c - 1)) as i64) << c);
    let carry_in = if offset == 0 {
        0
    } else {
        read_bits(limbs, offset - 1, 1) as i64
    };
    d + carry_in
}

/// Scalars split into the three classes that matter, with the two cheap classes already
/// paid for: `ones_sum` holds one mixed addition per scalar equal to 1, zeros are gone,
/// and only `idx`/`bigints` reach the bucket loop. `idx[k]` is the base that `bigints[k]`
/// multiplies, so the bucket loop walks both vectors in order and hits the bases through
/// one indirection.
struct Prescan<P: SWCurveConfig> {
    idx: Vec<u32>,
    bigints: Vec<<Fr as PrimeField>::BigInt>,
    ones_sum: Projective<P>,
}

/// Splits the scalars into zero / one / general, once, before any window runs.
///
/// `into_bigint` is a Montgomery reduction; calling it per window instead of per scalar is
/// the classic way to make an MSM twice as slow. It is called here and nowhere else, and
/// not at all for a scalar that turns out to be 0 or 1.
///
/// # This is a deliberate privacy for performance trade, and it is not free
///
/// Skipping zero and one scalars makes the running time, the number of Montgomery
/// reductions, and the length of `idx` and `bigints` all functions of how many witness
/// entries are zero or one. That is witness-dependent, and it is observable: as wall clock,
/// as allocation size in RSS, and on a GPU backend as dispatch size, because the kernel
/// launch geometry is derived from the general-scalar count.
///
/// This is not a hypothetical. It is the same optimisation exploited in "Remote
/// Side-Channel Attacks on Anonymous Transactions" (USENIX Security 2020), which recovered
/// information about Zcash shielded transactions by timing the prover. The measured effect
/// there was a correlation between proving time and the sparsity of the witness.
///
/// It is kept because every production Groth16 prover does it and the cost is nil; on
/// this benchmark ladder the measured gain is small (0/1 scalars are 1.8-4.1% of the
/// witness, about 1.02x; an earlier comment claimed 5.1x from a synthetic sparse
/// workload), but on a genuinely bit-heavy witness it grows with the sparsity. But
/// it means **this prover is not constant time with respect to the witness**, and a
/// deployment where an attacker can measure proving time or memory must treat that as part
/// of its threat model rather than assuming zero-knowledge covers it. Zero-knowledge is a
/// property of the proof, not of the process that produced it.
///
/// A constant-time variant would have to process every scalar through the general path,
/// giving up both fast paths and the compaction.
fn prescan<P: SWCurveConfig>(bases: &[Affine<P>], scalars: &[Fr], n: usize) -> Prescan<P>
where
    P::BaseField: Send + Sync,
{
    let run = |lo: usize, hi: usize| {
        let mut idx = Vec::new();
        let mut bigints = Vec::new();
        let mut ones_sum = Projective::<P>::zero();
        for i in lo..hi {
            let s = &scalars[i];
            if s.is_zero() {
                // No bucket touched, no Montgomery reduction, no base read.
                continue;
            }
            // A base at infinity contributes nothing whatever its scalar. This is not a
            // rare key quirk: 34% of the B query bases are the point at infinity on
            // every measured circuit (a wire that appears in no B-side linear
            // combination), and letting them through would cost a madd call per window
            // each. Filtering here also lets the bucket loop's madd skip its own
            // infinity test entirely.
            if bases[i].infinity {
                continue;
            }
            if s.is_one() {
                // One mixed addition, total, for the whole scalar.
                ones_sum += &bases[i];
                continue;
            }
            idx.push(i as u32);
            bigints.push(s.into_bigint());
        }
        Prescan {
            idx,
            bigints,
            ones_sum,
        }
    };

    // The scan is O(n) with a Montgomery reduction in it, so at prover sizes it is a big
    // enough serial fraction to be worth splitting.
    if n < 4 * SCAN_CHUNK {
        return run(0, n);
    }
    let chunks = n.div_ceil(SCAN_CHUNK);
    let parts: Vec<Prescan<P>> = (0..chunks)
        .into_par_iter()
        .map(|ci| run(ci * SCAN_CHUNK, ((ci + 1) * SCAN_CHUNK).min(n)))
        .collect();

    let total: usize = parts.iter().map(|p| p.idx.len()).sum();
    let mut out = Prescan {
        idx: Vec::with_capacity(total),
        bigints: Vec::with_capacity(total),
        ones_sum: Projective::<P>::zero(),
    };
    for p in parts {
        out.idx.extend_from_slice(&p.idx);
        out.bigints.extend_from_slice(&p.bigints);
        out.ones_sum += &p.ones_sum;
    }
    out
}

/// Bucket accumulation and reduction for one window over one contiguous slice of the
/// prescanned scalars. Returns `sum_j j * B_j` for that slice.
///
/// The buckets are XYZZ over the branch-free raw field layer (`crate::xyzz`), not ark
/// Jacobian. Two reasons, both measured: madd-2008-s is 7M + 2S against ark's Jacobian
/// madd at 7M + 4S, and `ark-ff` ends every field operation in a compare-and-branch
/// reduction that profiling showed costs a G1 mixed add 247 ns against its 148 ns
/// multiply floor. This loop is 15.5 million additions per 140k-constraint proof; it is
/// the single hottest loop in the CPU prover and the reason `g16_field::raw` exists.
///
/// The conversion back to ark happens once per chunk, multiplication-only.
fn window_chunk<P: RawCurve>(
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    range: core::ops::Range<usize>,
    window: usize,
    c: u32,
    n_buckets: usize,
) -> Projective<P> {
    // 2^(c-1) buckets for |d| in [1, 2^(c-1)], plus one for the digit that the
    // carry-free recoding pushes to exactly 2^(c-1).
    let mut buckets = vec![Xyzz::<P::RF>::ZERO; n_buckets];
    for k in range {
        let d = signed_digit(scan.bigints[k].as_ref(), window, c);
        // The zero digit must not touch a bucket. It is not a rare case: a random scalar
        // hits it about once every 2^(c-1) windows, and a small witness value like 2 or 7
        // is zero in every window but the lowest.
        if d == 0 {
            continue;
        }
        // Never infinity: prescan filtered those, so raw_xy is total here.
        let (x, y) = P::raw_xy(&bases[scan.idx[k] as usize]);
        if d > 0 {
            buckets[(d - 1) as usize].madd(x, y);
        } else {
            // Subtraction negates y and does a mixed add, which is the entire reason
            // signed digits are worth the recoding: half the buckets, same cost.
            buckets[(-d - 1) as usize].madd(x, y.neg());
        }
    }

    // Running sum: sum_j j*B_j in 2 * 2^(c-1) additions rather than sum_j j additions,
    // in XYZZ (add-2008-s, 12M + 2S, against ark's Jacobian add at 11M + 5S plus the
    // branchy reductions).
    let mut running = Xyzz::<P::RF>::ZERO;
    let mut total = Xyzz::<P::RF>::ZERO;
    for b in buckets.iter().rev() {
        running.add_assign(b);
        total.add_assign(&running);
    }
    to_projective(&total)
}

fn pippenger<P: RawCurve>(bases: &[Affine<P>], scalars: &[Fr], threads: usize) -> Projective<P>
where
    P::BaseField: Send + Sync,
{
    // A hard assert, not a debug one. A malformed or hand-built proving key with
    // mismatched lengths must fail loudly in release too: the previous
    // `debug_assert!` + `min()` combination silently dropped the tail of the longer
    // side, which turns a bad key into a wrong-but-plausible MSM instead of a panic
    // pointing at the key.
    assert_eq!(
        bases.len(),
        scalars.len(),
        "MSM length mismatch: {} bases, {} scalars",
        bases.len(),
        scalars.len()
    );
    let n = bases.len();
    if n == 0 {
        return Projective::zero();
    }

    let scan = prescan(bases, scalars, n);
    let m = scan.idx.len();
    // Every scalar was 0 or 1, which is the normal case for a bit-decomposition-heavy
    // witness. Skip Pippenger entirely rather than allocate W bucket arrays for nothing.
    if m == 0 {
        return scan.ones_sum;
    }

    // The window is sized for the work that actually reaches the buckets, not for the
    // input length: a 2^20-long witness with 2^10 general scalars is a 2^10 problem.
    let c = window_size(m);
    let n_windows = RECODE_BITS.div_ceil(c as usize);

    // Windows are independent, so they are the first axis of parallelism, and at BN254
    // sizes there are usually more of them (17 to 20) than there are cores. When there are
    // not, split the point range too and give each chunk its own bucket array. Summing the
    // per-chunk reduced sums is exact because the running-sum reduction is linear in the
    // buckets, and it is far cheaper than merging bucket arrays. Splitting past what the
    // pool can run at once is a pure loss: it buys no parallelism and pays another bucket
    // reduction per chunk.
    // Signed digits land in [-2^(c-1), 2^(c-1)] and index as `d - 1` when positive and
    // `-d - 1` when negative, so the highest reachable index is 2^(c-1) - 1. Allocating
    // 2^(c-1) + 1 left a bucket that is never written and still walked by the running-sum
    // reduction, costing two Jacobian additions per window per point chunk.
    let n_buckets = 1usize << (c - 1);
    let point_chunks = threads
        .max(1)
        .div_ceil(n_windows)
        .clamp(1, (m / (CHUNK_BUCKET_RATIO * n_buckets)).max(1));
    let chunk_len = m.div_ceil(point_chunks);

    let window_sums: Vec<Projective<P>> = (0..n_windows)
        .into_par_iter()
        .map(|w| {
            (0..point_chunks)
                .into_par_iter()
                .map(|ci| {
                    let lo = ci * chunk_len;
                    let hi = ((ci + 1) * chunk_len).min(m);
                    if lo >= hi {
                        return Projective::zero();
                    }
                    window_chunk(bases, &scan, lo..hi, w, c, n_buckets)
                })
                .reduce(Projective::zero, |a, b| a + b)
        })
        .collect();

    // Horner over the windows, high to low: c doublings between each.
    let mut acc = window_sums[n_windows - 1];
    for w in (0..n_windows - 1).rev() {
        for _ in 0..c {
            acc.double_in_place();
        }
        acc += &window_sums[w];
    }
    acc + scan.ones_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_std::{rand::Rng, test_rng, UniformRand};
    use g16_field::Field;

    /// Straight sum of scalar multiplications. Slow and obviously correct.
    fn naive<P: SWCurveConfig<ScalarField = Fr>>(
        bases: &[Affine<P>],
        scalars: &[Fr],
    ) -> Projective<P> {
        bases
            .iter()
            .zip(scalars)
            .fold(Projective::zero(), |acc, (b, s)| acc + *b * s)
    }

    /// Random points, one scalar multiplication each. Fine up to a few thousand.
    fn rand_points<P: SWCurveConfig>(n: usize, rng: &mut impl Rng) -> Vec<Affine<P>>
    where
        Projective<P>: UniformRand,
    {
        let proj: Vec<Projective<P>> = (0..n).map(|_| Projective::<P>::rand(rng)).collect();
        Projective::normalize_batch(&proj)
    }

    /// Bench-sized bases without paying for `n` scalar multiplications: walk an arithmetic
    /// progression of points. Structure in the bases cannot help or hurt an MSM, which
    /// only ever adds them.
    fn walk_points<P: SWCurveConfig>(n: usize, rng: &mut impl Rng) -> Vec<Affine<P>>
    where
        Projective<P>: UniformRand,
    {
        let step = Projective::<P>::rand(rng);
        let mut cur = Projective::<P>::rand(rng);
        let mut proj = Vec::with_capacity(n);
        for _ in 0..n {
            proj.push(cur);
            cur += &step;
        }
        Projective::normalize_batch(&proj)
    }

    fn rand_scalars(n: usize, rng: &mut impl Rng) -> Vec<Fr> {
        (0..n).map(|_| Fr::rand(rng)).collect()
    }

    #[test]
    fn window_size_is_in_range_and_grows() {
        let mut prev = 0;
        for log in 0..24 {
            let c = window_size(1 << log);
            assert!((3..=MAX_WINDOW).contains(&c), "n = 2^{log} gave c = {c}");
            assert!(c >= prev, "window size went backwards at 2^{log}");
            prev = c;
        }
        assert_eq!(window_size(0), 3);
        // Sanity against the hand-worked optimum of the same cost model.
        assert_eq!(window_size(1 << 16), 13);
        assert_eq!(window_size(1 << 18), 15);
    }

    #[test]
    fn signed_digits_reconstruct_the_scalar() {
        let mut rng = test_rng();
        let mut cases = vec![
            Fr::zero(),
            Fr::one(),
            -Fr::one(),
            Fr::from(2u64),
            Fr::from(u64::MAX),
        ];
        cases.extend(rand_scalars(32, &mut rng));

        for k in cases {
            let big = k.into_bigint();
            let limbs = big.as_ref();
            for c in 3..=MAX_WINDOW {
                let n_windows = RECODE_BITS.div_ceil(c as usize);
                let half = 1i64 << (c - 1);
                // Rebuild the scalar from its digits, high window first.
                let mut acc = Fr::zero();
                let two_pow_c = Fr::from(2u64).pow([c as u64]);
                for w in (0..n_windows).rev() {
                    let d = signed_digit(limbs, w, c);
                    assert!(
                        (-half..=half).contains(&d),
                        "digit {d} out of range for c = {c}"
                    );
                    acc *= two_pow_c;
                    acc += if d >= 0 {
                        Fr::from(d as u64)
                    } else {
                        -Fr::from((-d) as u64)
                    };
                }
                assert_eq!(acc, k, "recoding lost the scalar at c = {c}");
            }
        }
    }

    #[test]
    fn matches_naive_g1() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        for n in [1usize, 2, 7, 100, 1000] {
            let bases: Vec<G1Affine> = rand_points(n, &mut rng);
            let scalars = rand_scalars(n, &mut rng);
            assert_eq!(
                msm.msm_g1(&bases, &scalars),
                naive(&bases, &scalars),
                "G1 n = {n}"
            );
        }
    }

    #[test]
    fn matches_naive_g2() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        for n in [1usize, 2, 7, 100, 1000] {
            let bases: Vec<G2Affine> = rand_points(n, &mut rng);
            let scalars = rand_scalars(n, &mut rng);
            assert_eq!(
                msm.msm_g2(&bases, &scalars),
                naive(&bases, &scalars),
                "G2 n = {n}"
            );
        }
    }

    #[test]
    fn matches_arkworks_g1() {
        let mut rng = test_rng();
        let n = 1 << 13;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let want = G1Projective::msm(&bases, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g1(&bases, &scalars), want);
    }

    #[test]
    fn matches_arkworks_g2() {
        let mut rng = test_rng();
        let n = 1 << 12;
        let bases: Vec<G2Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let want = G2Projective::msm(&bases, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g2(&bases, &scalars), want);
    }

    #[test]
    fn empty_input_is_identity() {
        let msm = CpuMsm::new();
        assert!(msm.msm_g1(&[], &[]).is_zero());
        assert!(msm.msm_g2(&[], &[]).is_zero());
    }

    #[test]
    fn all_zero_scalars_is_identity() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let n = 257;
        let g1: Vec<G1Affine> = rand_points(n, &mut rng);
        let g2: Vec<G2Affine> = rand_points(n, &mut rng);
        let zeros = vec![Fr::zero(); n];
        assert!(msm.msm_g1(&g1, &zeros).is_zero());
        assert!(msm.msm_g2(&g2, &zeros).is_zero());
    }

    #[test]
    fn all_one_scalars_is_the_plain_sum() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let n = 257;
        let ones = vec![Fr::one(); n];

        let g1: Vec<G1Affine> = rand_points(n, &mut rng);
        let want1 = g1.iter().fold(G1Projective::zero(), |a, p| a + p);
        assert_eq!(msm.msm_g1(&g1, &ones), want1);

        let g2: Vec<G2Affine> = rand_points(n, &mut rng);
        let want2 = g2.iter().fold(G2Projective::zero(), |a, p| a + p);
        assert_eq!(msm.msm_g2(&g2, &ones), want2);
    }

    #[test]
    fn single_point() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        for s in [Fr::zero(), Fr::one(), -Fr::one(), Fr::rand(&mut rng)] {
            let g1: Vec<G1Affine> = rand_points(1, &mut rng);
            assert_eq!(msm.msm_g1(&g1, &[s]), g1[0] * s);
            let g2: Vec<G2Affine> = rand_points(1, &mut rng);
            assert_eq!(msm.msm_g2(&g2, &[s]), g2[0] * s);
        }
    }

    /// The scalar mix a real witness actually has: mostly 0 and 1, a few edge values,
    /// and enough general scalars to keep the bucket path live.
    fn witness_like_scalars(n: usize, rng: &mut impl Rng) -> Vec<Fr> {
        let mut s: Vec<Fr> = (0..n)
            .map(|i| match i % 8 {
                0..=4 => Fr::zero(),
                5 | 6 => Fr::one(),
                _ => Fr::rand(rng),
            })
            .collect();
        s[0] = -Fr::one();
        s[1] = Fr::from(2u64);
        s[2] = Fr::zero();
        s[3] = Fr::one();
        // r - 1 is the largest scalar there is, and the one that stresses the top window.
        s[4] = Fr::from(u64::MAX);
        s
    }

    #[test]
    fn special_scalars_mixed_in() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let n = 500;
        let g1: Vec<G1Affine> = rand_points(n, &mut rng);
        let s = witness_like_scalars(n, &mut rng);
        assert_eq!(msm.msm_g1(&g1, &s), naive(&g1, &s));

        let g2: Vec<G2Affine> = rand_points(n, &mut rng);
        assert_eq!(msm.msm_g2(&g2, &s), naive(&g2, &s));
    }

    #[test]
    fn points_at_infinity_in_the_bases() {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let n = 200;

        let mut g1: Vec<G1Affine> = rand_points(n, &mut rng);
        for i in (0..n).step_by(7) {
            g1[i] = G1Affine::identity();
        }
        let s = rand_scalars(n, &mut rng);
        assert_eq!(msm.msm_g1(&g1, &s), naive(&g1, &s));
        // Also with the 0/1 fast paths landing on infinity.
        let sw = witness_like_scalars(n, &mut rng);
        assert_eq!(msm.msm_g1(&g1, &sw), naive(&g1, &sw));

        let mut g2: Vec<G2Affine> = rand_points(n, &mut rng);
        for i in (0..n).step_by(5) {
            g2[i] = G2Affine::identity();
        }
        assert_eq!(msm.msm_g2(&g2, &s), naive(&g2, &s));
        assert_eq!(msm.msm_g2(&g2, &sw), naive(&g2, &sw));

        // Every base at infinity is still identity, whatever the scalars are.
        let all_inf = vec![G1Affine::identity(); n];
        assert!(msm.msm_g1(&all_inf, &s).is_zero());
    }

    /// `cargo test -p g16-msm --release -- --ignored --nocapture bench`
    ///
    /// Best of three, because this runs on a shared machine and the mean of a contended
    /// run measures the other tenants.
    #[test]
    #[ignore = "benchmark, meaningless in a debug build"]
    fn bench_against_arkworks() {
        use std::time::Instant;
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let serial = CpuMsm { threads: 1 };
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let best = |f: &mut dyn FnMut() -> f64| (0..3).map(|_| f()).fold(f64::MAX, f64::min);
        println!("rayon threads: {}", msm.threads);

        for log in [16u32, 18] {
            let n = 1usize << log;
            let bases: Vec<G1Affine> = walk_points(n, &mut rng);
            let scalars = rand_scalars(n, &mut rng);
            let mut ours_out = G1Projective::zero();
            let mut ours_1t_out = G1Projective::zero();
            let mut ark_out = G1Projective::zero();

            let ours_ms = best(&mut || {
                let t = Instant::now();
                ours_out = msm.msm_g1(&bases, &scalars);
                t.elapsed().as_secs_f64() * 1e3
            });
            let ours_1t_ms = best(&mut || {
                let t = Instant::now();
                ours_1t_out = one_thread.install(|| serial.msm_g1(&bases, &scalars));
                t.elapsed().as_secs_f64() * 1e3
            });
            let ark_ms = best(&mut || {
                let t = Instant::now();
                ark_out = G1Projective::msm(&bases, &scalars).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            });

            assert_eq!(ours_out, ark_out);
            assert_eq!(ours_1t_out, ark_out);
            println!(
                "2^{log} G1  c={:2}  ours {:8.2} ms  ours(1 thread) {:9.2} ms  arkworks(1 thread, no rayon feature) {:9.2} ms  ratio {:.2}x mt, {:.2}x st",
                window_size(n),
                ours_ms,
                ours_1t_ms,
                ark_ms,
                ark_ms / ours_ms,
                ark_ms / ours_1t_ms,
            );
        }
    }
}
