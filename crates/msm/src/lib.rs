//! Pippenger multi-scalar multiplication over BN254 G1 and G2.
//!
//! Five MSMs dominate the proof (~53% of a fast CPU prover's wall clock at 151K
//! constraints, ~70% of a slow one's), so this is the crate that decides whether the
//! project is worth anything.
//!
//! Two properties matter beyond raw speed, both because the witness is the scalar vector
//! for four of the five MSMs:
//!   * a scalar of 0 must cost nothing beyond the digit scan,
//!   * a scalar of 1 must cost exactly one addition, not a full bucket round trip.
//!
//! How much those paths matter is circuit-shaped, and an earlier version of this comment
//! overclaimed it: "in bit-decomposition-heavy circuits over 99% of witness scalars are
//! 0 or 1". Measured on the benchmark ladder (round-4 profiling), 0/1 scalars are 1.80%
//! of the witness on the two largest circuits and 4.12% at the sparsest point, worth
//! about 1.02x, not the 5x once asserted. The paths stay because they are nearly free
//! and a genuinely bit-heavy circuit still benefits, but on this ladder the MSMs are
//! dense and the bucket loop is the whole game. What IS a measured, structural win on
//! every key measured so far: a large fraction of the B query bases are the point at
//! infinity, and the prescan drops them before they cost a single window visit. The
//! figures are on `prescan`.

use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{AdditiveGroup, One, PrimeField, Zero};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use snarkrs_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

pub mod accel;
/// The curve and spot checks the GPU backends hold a ceremony readback to.
pub mod check;
pub mod glv;
pub mod safegcd;
pub mod xyzz;
use snarkrs_field::raw::RawField;
use xyzz::{to_projective, RawCurve, Xyzz};

pub use accel::{AccelError, CheckedGroupFft, GroupFft, KeyScale};

/// A backend for the five MSMs the proof is made of.
///
/// `bases` and `scalars` must be the same length, and a backend has to treat a mismatch as
/// a malformed key rather than as a short input: reducing to the shorter prefix returns a
/// valid-looking group element for a proof that is wrong. The CPU path asserts on it (see
/// `pippenger`) and every other backend is expected to be just as loud.
///
/// An empty input is the identity. A base at infinity contributes nothing whatever its
/// scalar, which is a live path and not a defensive one: most of a B query can be
/// infinity. See `prescan` for the measurements.
pub trait MsmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective;
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective;

    /// Fallible G1 MSM for ceremony callers. Rejects unequal input lengths.
    ///
    /// The default delegates to the legacy method and cannot recover its panics. Device
    /// adapters must override this method to return upload and execution errors instead.
    fn try_msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> Result<G1Projective, AccelError> {
        if bases.len() != scalars.len() {
            return Err(AccelError::Shape {
                op: "msm_g1",
                reason: format!("{} bases but {} scalars", bases.len(), scalars.len()),
            });
        }
        Ok(self.msm_g1(bases, scalars))
    }

    /// Fallible G2 MSM, with the same default and device override contract as
    /// [`Self::try_msm_g1`].
    fn try_msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> Result<G2Projective, AccelError> {
        if bases.len() != scalars.len() {
            return Err(AccelError::Shape {
                op: "msm_g2",
                reason: format!("{} bases but {} scalars", bases.len(), scalars.len()),
            });
        }
        Ok(self.msm_g2(bases, scalars))
    }
}

/// Bits in the scalar field modulus: 254 for BN254's Fr.
pub const SCALAR_BITS: usize = Fr::MODULUS_BIT_SIZE as usize;

/// Signed recoding carries one bit past the top of the scalar, so the digit array is
/// laid out over `SCALAR_BITS + 1` bits. See `signed_digit` for why that kills the
/// final carry rather than just hiding it.
///
/// Public because every GPU backend retypes it in its own shader language and asserts
/// against this one; a copy one below it is wrong only for scalars above `2^((nw-1)*c)`,
/// which a random test with small scalars never reaches.
pub const RECODE_BITS: usize = SCALAR_BITS + 1;

/// Caps the bucket array at 2^15 XYZZ points, 4 MiB of G1 (8 MiB of G2) per in-flight
/// task. A window this wide takes the batch-affine fill instead, which holds 2 MiB of
/// affine G1 plus a 4 MiB XYZZ side array it only allocates once a bucket contends. Past
/// this the cost model's gain is under 5% and the memset starts to hurt.
const MAX_WINDOW: u32 = 16;

/// Prescan granularity. Small enough to balance, large enough that the two output vectors
/// are worth appending rather than merging element by element.
const SCAN_CHUNK: usize = 1 << 12;

/// A point chunk must be at least this many times the bucket count to be worth splitting
/// off: every extra chunk pays another full bucket reduction, `2 * 2^(c-1)` additions,
/// so at 16x the reduction overhead of the split stays under a fifth of the chunk's work.
const CHUNK_BUCKET_RATIO: usize = 16;

/// Most entries per shared inversion in the batch-affine fill. Conflicts (a second point
/// for a bucket that already has a pending addition) grow as `batch^2 / n_buckets` while
/// the inversion amortises as `1 / batch`; at 1024 over the 4096 buckets H uses, ~12% of
/// points conflict (nearly all rescued by the retry pass, see [`BatchFill`]) and the
/// shared inversion costs well under one Fq product per point.
const BATCH: usize = 1024;

/// The batch-affine fill only runs when [`batch_size`] gives at least this many entries
/// per inversion, which is 256 buckets (c = 9). At 128 the shared inversion still costs
/// under one Fq product per point; narrower windows mean a small `m`, where the fill is
/// not the cost that matters, and they keep the XYZZ fill.
const MIN_BATCH: usize = 128;

/// Buckets per point of a piece, see [`split_grain`].
const GRAIN_BUCKET_RATIO: usize = 4;

/// Fewest points in a piece, about 0.1 ms of G1 on a performance core.
const MIN_GRAIN: usize = 1024;

/// Batch size for a window of `n_buckets`: half the buckets, capped at [`BATCH`]. Below
/// the cap this holds the conflict share `batch / n_buckets` where the 2048-bucket window
/// has it, instead of letting it climb as the window narrows. It also keeps the batch
/// below the bucket count, which the busy map needs: a round holds at most one pending
/// addition per bucket, so a batch at or past the bucket count starves the flush trigger.
fn batch_size(n_buckets: usize) -> usize {
    BATCH.min(n_buckets / 2)
}

/// Window size for `n` points. Pippenger's cost is minimised near `ln(n)`; sppark uses
/// `min(floor(lg2(1.5n)) - 8, 18)` floored at 10 on GPU, but the CPU optimum is smaller
/// because there is no bucket-sort machinery to amortise.
///
/// Rather than fit a closed form, minimise the cost model directly: `ceil(255/c)` windows,
/// each doing one bucket visit per point plus a running-sum reduction of two additions per
/// bucket over `2^(c-1)` buckets. Signed digits are what put `2^(c-1)` there instead of
/// `2^c`, which is worth roughly one extra bit of window. The weight of 3 on the bucket
/// term is the XYZZ fill's ratio: a visit is madd-2008-s (8M + 2S), a bucket two add-2008-s
/// (12M + 2S each). Windows from c = 9 up take the batch-affine fill instead, a 5M + 1S
/// visit and two affine additions per bucket behind a shared inversion (12M + 2S, see
/// [`segmented_running_sum_at`]), so their ratio is nearer 2.3. Weighting 2.3 picks the
/// same window as 3 at every power of two from 2^15 up, so the model is kept on 3. The
/// one extra bucket `signed_digit` needs is left out of the model, it moves nothing.
///
/// A forced-window sweep (`sweep_forced_window`, plus the same over js_16x16_d32's real
/// key and witness; M2 Max, 12 threads, medians of 7 to 9 interleaved reps, two runs) has
/// the pick best or tied: dense 2^18 c = 15 at 69.6-71.1 ms against 70.9-72.2 at 13 and
/// 14, 75-77 at 12, 78-79 at 16; dense 2^21 c = 16 at 434-439 ms against 437-444 at 15,
/// 525-526 at 14; a_query (137105 general scalars) c = 13 at 39.6-39.9 ms against
/// 39.4-40.7 for 12, 14 and 15, and 47 at 16.
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
    /// Send every scalar through the bucket loop, so the cost follows the key and not the
    /// witness. Off by default because of what it costs; see [`Work`].
    pub constant_work: bool,
}

impl CpuMsm {
    pub fn new() -> Self {
        Self {
            threads: rayon::current_num_threads(),
            constant_work: false,
        }
    }

    /// [`CpuMsm::new`] with `constant_work` on.
    pub fn constant_work() -> Self {
        Self {
            constant_work: true,
            ..Self::new()
        }
    }

    fn work(&self) -> Work {
        if self.constant_work {
            Work::Constant
        } else {
            Work::Variable
        }
    }
}

/// Whether the MSM's cost may depend on the scalar values.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Work {
    /// Every scalar whose base is not the point at infinity goes through the bucket loop,
    /// and a zero digit costs a mixed addition into a bucket nobody reads. Which bases are
    /// at infinity is fixed by the key, so the scalar count, the window width, every
    /// allocation and the number of additions depend on the key and not on the witness.
    ///
    /// That closes the channel of "Remote Side-Channel Attacks on Anonymous Transactions"
    /// (USENIX Security 2020), which recovered Zcash witness sparsity from proving time. It
    /// is not constant time in the strict sense: the batched fill's conflict rounds and the
    /// order of bucket hits still depend on the digit values, as in every Pippenger.
    ///
    /// What it costs is what the witness's zeros and ones were saving, so it is priced per
    /// circuit. Warm CPU proofs on the M2 Max, 15 reps, three alternating rounds against the
    /// variable-time path: js_16x16_d32 399-404 ms against 390-392 (+2.5%), keccak256
    /// 520-534 against 161-162 (3.3x), rsa2048 668-700 against 169-170 (about 4x). The
    /// earlier "under 2%" held for the dense circuits only.
    Constant,
    /// Zero scalars are dropped, one scalars cost one addition in total, and the window is
    /// sized for the rest. Running time, allocation size and the number of Montgomery
    /// reductions then follow how many witness entries are zero or one.
    Variable,
}

impl Default for CpuMsm {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuMsm {
    /// [`MsmBackend::msm_g1`] for a caller that is done with `scalars`: the scan converts
    /// them in place instead of beside them, so the MSM holds no second scalar vector. On
    /// return `scalars` no longer holds the field elements it was given, and still holds
    /// everything they reveal; scrubbing it is the caller's job.
    pub fn msm_g1_in_place(&self, bases: &[G1Affine], scalars: &mut [Fr]) -> G1Projective {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM length mismatch: {} bases, {} scalars",
            bases.len(),
            scalars.len()
        );
        if bases.is_empty() {
            return G1Projective::zero();
        }
        let work = self.work();
        pippenger_scanned(
            bases,
            prescan_in_place(bases, scalars, work),
            self.threads,
            work,
        )
    }
}

impl MsmBackend for CpuMsm {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        pippenger(bases, scalars, self.threads, self.work())
    }
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        pippenger(bases, scalars, self.threads, self.work())
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
    if shift + width as usize > 64 && idx + 1 < limbs.len() {
        // The branch condition forces `shift > 64 - width >= 1`, so the `64 - shift`
        // shift is in range.
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
/// paid for: `ones_sum` holds one addition per scalar equal to 1, zeros are gone,
/// and only `idx`/`bigints` reach the bucket loop. `idx[k]` is the base that `bigints[k]`
/// multiplies, so the bucket loop walks both vectors in order and hits the bases through
/// one indirection.
struct Prescan<'a, P: SWCurveConfig> {
    /// `u32` to halve the footprint of a witness-length vector; a key with more than
    /// `2^32` bases is not representable and is asserted against in `prescan`.
    idx: Vec<u32>,
    /// Borrowed only by [`prescan_in_place`], from the caller's own scalar buffer.
    bigints: Cow<'a, [BigInt]>,
    ones_sum: Projective<P>,
}

/// An owned `bigints` is the standard form of witness scalars, so it is zeroed when it goes.
/// A borrowed one is its owner's to scrub.
impl<P: SWCurveConfig> Drop for Prescan<'_, P> {
    fn drop(&mut self) {
        if let Cow::Owned(v) = &mut self.bigints {
            scrub(v);
        }
    }
}

/// Zero `v` through volatile writes spread over the pool, then one fence. The same writes
/// as `snarkrs_groth16::scrub`, over `BigInt` rather than `Fr`.
fn scrub(v: &mut [BigInt]) {
    v.par_chunks_mut(SCAN_CHUNK).for_each(|chunk| {
        for x in chunk {
            for limb in x.0.iter_mut() {
                // SAFETY: `limb` is a valid, aligned, exclusive `&mut u64`.
                unsafe { core::ptr::write_volatile(limb, 0) };
            }
        }
    });
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

type BigInt = <Fr as PrimeField>::BigInt;

/// Points a [`OnesFold`] holds before it adds them in pairs.
const ONES_BATCH: usize = 1024;

/// Fewest points left at the end of a [`OnesFold`] that are worth an inversion.
const ONES_MIN: usize = 64;

/// The plain sum of the bases whose scalar is one. The points wait in a buffer, and a
/// full buffer is halved: its upper half is added to its lower half point by point
/// behind one inversion, as in [`BatchFill::flush`], 5M + 1S an addition against the
/// 8M + 2S of a mixed one. What is left at the end is halved down to [`ONES_MIN`] points,
/// and those go into `rest` one by one. On one thread of the M2 Max (`bench_ones_fold`)
/// a point costs 112 ns against 135 for a chain of mixed additions on G1 and 234 against
/// 331 on G2, and at the 1000 points of a bit-heavy scan chunk 119 and 245.
struct OnesFold<P: RawCurve> {
    xy: Vec<[P::RF; 2]>,
    /// The product of the denominators before each pair, see [`batch_invert`].
    prefix: Vec<P::RF>,
    rest: Xyzz<P::RF>,
}

impl<P: RawCurve> OnesFold<P> {
    fn new() -> Self {
        OnesFold {
            xy: Vec::new(),
            prefix: Vec::new(),
            rest: Xyzz::ZERO,
        }
    }

    /// The caller guarantees `(x, y)` is not the point at infinity.
    #[inline(always)]
    fn add(&mut self, x: P::RF, y: P::RF) {
        if self.xy.len() == ONES_BATCH {
            self.halve();
        }
        self.xy.push([x, y]);
    }

    /// `xy[k] += xy[k + half]` for every `k` below `half`, an odd point out kept as it is.
    fn halve(&mut self) {
        let n = self.xy.len();
        let half = n / 2;
        self.prefix.clear();
        let mut acc = P::RF::ONE;
        for k in 0..half {
            let upper = self.xy[k + half];
            let dx = upper[0].sub(self.xy[k][0]);
            self.prefix.push(acc);
            if dx.is_zero() {
                // The same point twice or a point and its negative: a zkey does not
                // guarantee distinct bases. `madd` knows both cases, so the upper point
                // goes there and the lower one stays.
                self.rest.madd(upper[0], upper[1]);
            } else {
                acc = acc.mul(dx);
            }
        }
        let mut inv = P::raw_inv(acc);
        for k in (0..half).rev() {
            let (lower, upper) = (self.xy[k], self.xy[k + half]);
            let dx = upper[0].sub(lower[0]);
            if !dx.is_zero() {
                self.xy[k] = affine_add(lower, upper, inv.mul(self.prefix[k]));
                inv = inv.mul(dx);
            }
        }
        if n % 2 == 1 {
            self.xy[half] = self.xy[n - 1];
        }
        self.xy.truncate(n - half);
    }

    fn finish(mut self) -> Projective<P> {
        while self.xy.len() >= ONES_MIN {
            self.halve();
        }
        for p in &self.xy {
            self.rest.madd(p[0], p[1]);
        }
        to_projective(&self.rest)
    }
}

/// What [`prescan`] sums its one scalars with: [`FastFill::Ones`] where the CPU has it,
/// else [`OnesFold`].
enum AnyOnes<P: FastFill> {
    Fast(P::Ones),
    Slow(OnesFold<P>),
}

impl<P: FastFill> AnyOnes<P> {
    fn new() -> Self {
        match P::ones() {
            Some(fast) => AnyOnes::Fast(fast),
            None => AnyOnes::Slow(OnesFold::new()),
        }
    }

    /// The caller guarantees `(x, y)` is not the point at infinity.
    #[inline(always)]
    fn add(&mut self, x: P::RF, y: P::RF) {
        match self {
            AnyOnes::Fast(f) => f.add(x, y),
            AnyOnes::Slow(f) => f.add(x, y),
        }
    }

    fn finish(self) -> Projective<P> {
        match self {
            AnyOnes::Fast(f) => f.finish(),
            AnyOnes::Slow(f) => f.finish(),
        }
    }
}

/// A plain sum of affine points, none at infinity; see [`OnesFold`].
trait OnesSum<P: RawCurve> {
    fn add(&mut self, x: P::RF, y: P::RF);
    fn finish(self) -> Projective<P>;
}

impl<P: RawCurve> OnesSum<P> for OnesFold<P> {
    #[inline(always)]
    fn add(&mut self, x: P::RF, y: P::RF) {
        OnesFold::add(self, x, y)
    }
    fn finish(self) -> Projective<P> {
        OnesFold::finish(self)
    }
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
/// It is the default because every production Groth16 prover does it and because on a
/// bit-heavy circuit it is most of the MSM: the constant-work path costs 2.5% on the dense
/// js_16x16_d32 (0/1 scalars are 1.8% of its witness) but 3.3x on keccak256 and about 4x on
/// rsa2048. So by default **this prover is not constant time with respect to the witness**,
/// and a deployment where an attacker can measure proving time or memory must either turn
/// on [`CpuMsm::constant_work`] (`g16 prove --constant-work`) or treat the channel as part
/// of its threat model. Zero-knowledge is a property of the proof, not of the process that
/// produced it.
///
/// [`Work::Constant`] processes every scalar through the general path and gives up both
/// fast paths and the compaction.
fn prescan<P: FastFill>(
    bases: &[Affine<P>],
    scalars: &[Fr],
    n: usize,
    work: Work,
) -> Prescan<'static, P>
where
    P::BaseField: Send + Sync,
{
    debug_assert!(n <= u32::MAX as usize);
    let run = |lo: usize, hi: usize| {
        let mut idx = Vec::new();
        let mut bigints = Vec::new();
        let mut ones = AnyOnes::<P>::new();
        for i in lo..hi {
            let s = &scalars[i];
            if work == Work::Variable && s.is_zero() {
                // No bucket touched, no Montgomery reduction, no base read.
                continue;
            }
            // A base at infinity contributes nothing whatever its scalar. This is not a
            // rare key quirk: 34% of the B query bases are the point at infinity on
            // every circuit in the benchmark ladder, and 61% on the csp keys (see
            // `snarkrs-metal::msm`), because a wire that appears in no B-side linear
            // combination still owns a slot. Letting them through would cost a madd
            // call per window each. Filtering here also lets the bucket loop's madd
            // skip its own infinity test entirely.
            if bases[i].infinity {
                continue;
            }
            if work == Work::Variable && s.is_one() {
                // One addition, total, for the whole scalar. On a bit-heavy witness
                // this loop runs tens of thousands of times per MSM, see [`OnesFold`].
                let (x, y) = P::raw_xy(&bases[i]);
                ones.add(x, y);
                continue;
            }
            if bigints.len() == bigints.capacity() {
                // Grown by hand so the outgrown buffer is scrubbed rather than freed as is.
                let mut grown = Vec::with_capacity((2 * bigints.len()).max(4));
                grown.extend_from_slice(&bigints);
                scrub(&mut bigints);
                bigints = grown;
            }
            idx.push(i as u32);
            bigints.push(s.into_bigint());
        }
        Prescan {
            idx,
            bigints: Cow::Owned(bigints),
            ones_sum: ones.finish(),
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
    let mut idx = Vec::with_capacity(total);
    let mut bigints = Vec::with_capacity(total);
    let mut ones_sum = Projective::<P>::zero();
    for p in &parts {
        idx.extend_from_slice(&p.idx);
        bigints.extend_from_slice(&p.bigints);
        ones_sum += &p.ones_sum;
    }
    // Each part scrubs its chunk as it drops; in parallel, not inside the serial merge.
    parts.into_par_iter().for_each(drop);
    Prescan {
        idx,
        bigints: Cow::Owned(bigints),
        ones_sum,
    }
}

/// [`prescan`] without the second copy of the scalars: each general scalar's standard
/// form is written over the front of `scalars` itself, in order, and the scan borrows it
/// from there. Same classes, same order, same `idx`, so the MSM it feeds is the same sum.
///
/// Each chunk compacts into its own front in parallel, then the chunks are moved down one
/// after another. A chunk lands at or below where it starts, which is what makes the
/// serial order safe; on a dense vector like H no chunk moves at all.
fn prescan_in_place<'a, P: FastFill>(
    bases: &[Affine<P>],
    scalars: &'a mut [Fr],
    work: Work,
) -> Prescan<'a, P>
where
    P::BaseField: Send + Sync,
{
    // The borrowed `[BigInt]` below is read through the `Fr` storage it was written into.
    const {
        assert!(size_of::<Fr>() == size_of::<BigInt>());
        assert!(align_of::<Fr>() == align_of::<BigInt>());
        assert!(core::mem::offset_of!(Fr, 0) == 0);
    }
    let n = scalars.len();
    debug_assert!(n <= u32::MAX as usize);
    let run = |lo: usize, chunk: &mut [Fr]| {
        let mut idx = Vec::new();
        let mut ones = AnyOnes::<P>::new();
        for j in 0..chunk.len() {
            let s = chunk[j];
            let i = lo + j;
            // The same three filters as `prescan`, in the same order.
            if work == Work::Variable && s.is_zero() {
                continue;
            }
            if bases[i].infinity {
                continue;
            }
            if work == Work::Variable && s.is_one() {
                let (x, y) = P::raw_xy(&bases[i]);
                ones.add(x, y);
                continue;
            }
            chunk[idx.len()].0 = s.into_bigint();
            idx.push(i as u32);
        }
        (idx, ones.finish())
    };
    let parts: Vec<(Vec<u32>, Projective<P>)> = scalars
        .par_chunks_mut(SCAN_CHUNK)
        .enumerate()
        .map(|(ci, chunk)| run(ci * SCAN_CHUNK, chunk))
        .collect();

    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut idx = Vec::with_capacity(total);
    let mut ones_sum = Projective::<P>::zero();
    for (ci, (part, sum)) in parts.into_iter().enumerate() {
        let (from, to) = (ci * SCAN_CHUNK, idx.len());
        if from != to {
            scalars.copy_within(from..from + part.len(), to);
        }
        idx.extend_from_slice(&part);
        ones_sum += &sum;
    }
    let front = &scalars[..total];
    // SAFETY: the const block above pins `Fr` to a `BigInt` at offset 0 with the same size
    // and alignment, so the slice has the same stride and every element's first 32 bytes
    // are the `BigInt` written into it above.
    let bigints = unsafe { core::slice::from_raw_parts(front.as_ptr().cast::<BigInt>(), total) };
    Prescan {
        idx,
        bigints: Cow::Borrowed(bigints),
        ones_sum,
    }
}

/// Bucket accumulation and reduction for one window over one contiguous slice of the
/// prescanned scalars. Returns `sum_j j * B_j` for that slice.
///
/// The buckets are XYZZ over the branch-free raw field layer (`crate::xyzz`), not ark
/// Jacobian. Two reasons, both measured: madd-2008-s is 8M + 2S against ark's Jacobian
/// madd at 7M + 4S, and `ark-ff` ends every field operation in a compare-and-branch
/// reduction that profiling showed costs a G1 mixed add 247 ns against its 148 ns
/// multiply floor. This loop is 15.5 million additions per 140k-constraint proof; it is
/// the single hottest loop in the CPU prover and the reason `snarkrs_field::raw` exists.
///
/// The conversion back to ark happens once per chunk, multiplication-only.
fn window_chunk<P: RawCurve>(
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    range: core::ops::Range<usize>,
    window: usize,
    c: u32,
    work: Work,
) -> Projective<P> {
    // `d` is in [-2^(c-1), 2^(c-1)], so `2^(c-1)` buckets is tight.
    let n_buckets = 1usize << (c - 1);
    let mut buckets = vec![Xyzz::<P::RF>::ZERO; n_buckets];
    // Where a zero digit's addition goes under `Work::Constant`. Never read into the
    // result; `black_box` below keeps the additions from being optimised out.
    let mut dummy = Xyzz::<P::RF>::ZERO;
    for k in range {
        let d = signed_digit(scan.bigints[k].as_ref(), window, c);
        // Never infinity: prescan filtered those, so raw_xy is total here.
        let (x, y) = P::raw_xy(&bases[scan.idx[k] as usize]);
        // The zero digit must not touch a real bucket. It is not a rare case: a random
        // scalar hits it about once every 2^(c-1) windows, and a small witness value like
        // 2 or 7 is zero in every window but the lowest.
        if d == 0 {
            if work == Work::Constant {
                dummy.madd(x, y);
            }
            continue;
        }
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
    core::hint::black_box(&dummy);
    to_projective(&total)
}

/// Affine buckets filled by batched affine additions sharing one field inversion
/// (Montgomery's trick). An affine addition with its inverse in hand is 2M + 1S, and the
/// inversion machinery costs 3M per entry, so a bucket visit is 5M + 1S against the XYZZ
/// madd's 8M + 2S; the buckets are also half the size, 64 against 128 bytes of G1.
///
/// The scheduling constraint is that one round may hold at most one pending addition per
/// bucket, because every entry snapshots its bucket at flush time. `busy` enforces it: a
/// second point for a busy bucket waits in `retry` and is rescheduled as soon as a flush
/// clears the round, and a third (two retries sharing one bucket) drops to the lazily
/// allocated XYZZ `side` buckets. That last hop is what bounds the degenerate case of a
/// whole chunk landing in one bucket (equal scalars over equal bases) at one madd per
/// point instead of a one-entry flush per point.
struct BatchFill<P: RawCurve> {
    /// Bucket `(x, y)`, together so a visit is one cache line rather than two.
    xy: Vec<[P::RF; 2]>,
    /// Bucket holds a point. Set by the first point, cleared again by a cancellation.
    occupied: Vec<bool>,
    /// Bucket has a pending addition in the current round.
    busy: Vec<bool>,
    /// The current round: bucket index and the point to add, sign already applied.
    pending: Vec<(u32, P::RF, P::RF)>,
    /// Conflicts waiting for the next flush.
    retry: Vec<(u32, P::RF, P::RF)>,
    /// XYZZ escape hatch for buckets too contended for the round machinery.
    side: Option<SideBuckets<P::RF>>,
    dens: Vec<P::RF>,
    prefix: Vec<P::RF>,
    ops: Vec<Op>,
    /// Pending (or retry) count that triggers a flush; see [`batch_size`].
    batch: usize,
}

/// The XYZZ buckets [`BatchFill`] falls back to, stored only for the buckets that needed
/// one. A dense array of every bucket was allocated per window chunk the first time any
/// bucket contended, which on the benchmark ladder is nearly every chunk: 2 MiB of G1 or
/// 4 MiB of G2 each, 84 MiB of churn per `js_16x16_d32` proof and 21 MiB at its peak, for
/// a handful of buckets. The slot map is 4 bytes a bucket.
struct SideBuckets<F> {
    /// Index into `points` per bucket; 0 is the shared zero, so an untouched bucket
    /// reads as the identity without a branch.
    slot: Vec<u32>,
    points: Vec<Xyzz<F>>,
}

impl<F: RawField> SideBuckets<F> {
    fn new(n_buckets: usize) -> Self {
        SideBuckets {
            slot: vec![0; n_buckets],
            points: vec![Xyzz::ZERO],
        }
    }

    fn entry(&mut self, b: usize) -> &mut Xyzz<F> {
        if self.slot[b] == 0 {
            self.slot[b] = self.points.len() as u32;
            self.points.push(Xyzz::ZERO);
        }
        &mut self.points[self.slot[b] as usize]
    }
}

impl<F> core::ops::Index<usize> for SideBuckets<F> {
    type Output = Xyzz<F>;

    fn index(&self, b: usize) -> &Xyzz<F> {
        &self.points[self.slot[b] as usize]
    }
}

/// What one pending entry turned out to be, decided before the shared inversion and
/// replayed after it.
#[derive(Clone, Copy)]
enum Op {
    Add,
    Dbl,
    Cancel,
}

impl<P: RawCurve> BatchFill<P> {
    fn new(n_buckets: usize, batch: usize) -> Self {
        // Retries rescheduled after a flush ride on top of the trigger threshold, so
        // the round vectors get headroom past the batch.
        let cap = batch + batch / 2;
        BatchFill {
            xy: vec![[P::RF::ZERO; 2]; n_buckets],
            occupied: vec![false; n_buckets],
            busy: vec![false; n_buckets],
            pending: Vec::with_capacity(cap),
            retry: Vec::new(),
            side: None,
            dens: Vec::with_capacity(cap),
            prefix: Vec::with_capacity(cap),
            ops: Vec::with_capacity(cap),
            batch,
        }
    }

    /// `bucket b += (x, y)`, eventually. Free when the bucket is empty, one pending
    /// entry when it is quiet, a retry when it is contended.
    fn insert(&mut self, b: usize, x: P::RF, y: P::RF) {
        if !self.occupied[b] {
            self.xy[b] = [x, y];
            self.occupied[b] = true;
            return;
        }
        if self.busy[b] {
            self.retry.push((b as u32, x, y));
            // Also a flush trigger: with degenerate digits it is the conflicts, not
            // the pending entries, that accumulate, and this bounds them.
            if self.retry.len() >= self.batch {
                self.flush();
                self.reschedule();
            }
            return;
        }
        self.busy[b] = true;
        self.pending.push((b as u32, x, y));
        if self.pending.len() >= self.batch {
            self.flush();
            self.reschedule();
        }
    }

    /// Second chance for conflicts after a flush cleared the busy map. A retry that
    /// conflicts again shares its bucket with another retry and drops to the side
    /// buckets rather than looping.
    fn reschedule(&mut self) {
        while let Some((b, x, y)) = self.retry.pop() {
            let b = b as usize;
            if !self.occupied[b] {
                self.xy[b] = [x, y];
                self.occupied[b] = true;
            } else if self.busy[b] {
                self.side
                    .get_or_insert_with(|| SideBuckets::new(self.xy.len()))
                    .entry(b)
                    .madd(x, y);
            } else {
                self.busy[b] = true;
                self.pending.push((b as u32, x, y));
            }
        }
    }

    /// One shared-inversion round: decide each entry's case, batch-invert the
    /// denominators with Montgomery's trick, apply. The case split runs before the
    /// product so a zero never reaches the inversion: a cancellation contributes ONE,
    /// and the doubling denominator `2y` is nonzero because the odd-order BN254 groups
    /// have no 2-torsion.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let n = self.pending.len();
        self.dens.clear();
        self.ops.clear();
        for &(b, px, py) in &self.pending {
            let [bx, by] = self.xy[b as usize];
            let dx = px.sub(bx);
            let (op, den) = if !dx.is_zero() {
                (Op::Add, dx)
            } else if py == by {
                (Op::Dbl, by.double())
            } else {
                // Same x, different y: the negative. The bucket empties.
                (Op::Cancel, P::RF::ONE)
            };
            self.ops.push(op);
            self.dens.push(den);
        }
        // The running products are two chains, one per half, walked side by side, whose
        // totals share the inversion. The point is the backward walk: it then holds two
        // independent entries at each step, and two additions with their formulas
        // interleaved ([`Self::add2`]) keep the multiplier busy where one addition's
        // dependent multiplies, each a long asm block, outrun what the core looks ahead.
        let h = n / 2;
        self.prefix.clear();
        self.prefix.resize(n, P::RF::ONE);
        let (mut a, mut c) = (P::RF::ONE, P::RF::ONE);
        for k in 0..h {
            self.prefix[k] = a;
            self.prefix[h + k] = c;
            a = a.mul(self.dens[k]);
            c = c.mul(self.dens[h + k]);
        }
        if n % 2 == 1 {
            self.prefix[n - 1] = c;
            c = c.mul(self.dens[n - 1]);
        }
        let ac = P::raw_inv(a.mul(c));
        let (mut ia, mut ic) = (ac.mul(c), ac.mul(a));
        if n % 2 == 1 {
            self.apply(n - 1, ic.mul(self.prefix[n - 1]));
            ic = ic.mul(self.dens[n - 1]);
        }
        for k in (0..h).rev() {
            let (da, dc) = (ia.mul(self.prefix[k]), ic.mul(self.prefix[h + k]));
            if matches!(self.ops[k], Op::Add) && matches!(self.ops[h + k], Op::Add) {
                self.add2(k, da, h + k, dc);
            } else {
                self.apply(k, da);
                self.apply(h + k, dc);
            }
            ia = ia.mul(self.dens[k]);
            ic = ic.mul(self.dens[h + k]);
        }
        self.pending.clear();
    }

    /// Two additions into two different buckets, the formulas interleaved step by step so
    /// each multiply has an independent one beside it.
    #[inline(always)]
    fn add2(&mut self, i: usize, di: P::RF, j: usize, dj: P::RF) {
        let (bi, pxi, pyi) = self.pending[i];
        let (bj, pxj, pyj) = self.pending[j];
        let (bi, bj) = (bi as usize, bj as usize);
        let [xi, yi] = self.xy[bi];
        let [xj, yj] = self.xy[bj];
        let li = pyi.sub(yi).mul(di);
        let lj = pyj.sub(yj).mul(dj);
        let si = li.sqr();
        let sj = lj.sqr();
        let x3i = si.sub(xi).sub(pxi);
        let x3j = sj.sub(xj).sub(pxj);
        let y3i = li.mul(xi.sub(x3i)).sub(yi);
        let y3j = lj.mul(xj.sub(x3j)).sub(yj);
        self.xy[bi] = [x3i, y3i];
        self.xy[bj] = [x3j, y3j];
        self.busy[bi] = false;
        self.busy[bj] = false;
    }

    /// Pending entry `i`, its denominator's inverse in hand.
    #[inline(always)]
    fn apply(&mut self, i: usize, d_inv: P::RF) {
        let (b, px, py) = self.pending[i];
        let b = b as usize;
        let [bx, by] = self.xy[b];
        match self.ops[i] {
            Op::Add => {
                let lambda = py.sub(by).mul(d_inv);
                let x3 = lambda.sqr().sub(bx).sub(px);
                self.xy[b] = [x3, lambda.mul(bx.sub(x3)).sub(by)];
            }
            Op::Dbl => {
                let xx = bx.sqr();
                let lambda = xx.double().add(xx).mul(d_inv);
                let x3 = lambda.sqr().sub(bx.double());
                self.xy[b] = [x3, lambda.mul(bx.sub(x3)).sub(by)];
            }
            Op::Cancel => self.occupied[b] = false,
        }
        self.busy[b] = false;
    }

    /// Flush the stragglers. Two flushes settle everything: after the first, a retry
    /// either schedules cleanly or drops to the side buckets in [`Self::reschedule`].
    fn settle(&mut self) {
        self.flush();
        self.reschedule();
        self.flush();
    }

    /// Settle, then reduce, on this thread.
    #[cfg(test)]
    fn finish(mut self) -> Projective<P> {
        self.settle();
        to_projective(&affine_running_sum::<P>(
            &mut self.xy,
            &mut self.occupied,
            self.side.as_ref(),
        ))
    }
}

/// The running sum `sum_j (j + 1) * B_j` over the settled buckets, the side buckets folded
/// in first when the chunk was degenerate enough to have any.
#[cfg(test)]
fn affine_running_sum<P: RawCurve>(
    xy: &mut [[P::RF; 2]],
    occupied: &mut [bool],
    side: Option<&SideBuckets<P::RF>>,
) -> Xyzz<P::RF> {
    if let Some(side) = side {
        fold_side::<P>(xy, occupied, side);
    }
    segmented_running_sum::<P>(xy, occupied)
}

/// A bucket range this short keeps the XYZZ running sum: the segmented one below would
/// pay an inversion a step for a handful of segments. At 128 buckets of G1 the XYZZ sum
/// measures 315 ns a bucket against 328 segmented, at 256 it is 292 to 317 against 285.
const REDUCE_XYZZ: usize = 128;

/// Segment length of [`segmented_running_sum`] over `n` buckets, a power of two near
/// `sqrt(n) / 2`. A step costs one inversion whatever the segment count and every segment
/// costs a fix-up at the end, so the two balance near `2 * sqrt(n)` segments.
fn reduce_segment(n: usize) -> usize {
    1 << (n.next_power_of_two().trailing_zeros() / 2).saturating_sub(1)
}

/// The same running sum as [`window_chunk`], in XYZZ with the occupied buckets entering as
/// mixed additions: one dependent chain, a madd and an add (20M + 4S) a bucket.
fn xyzz_running_sum<F: RawField>(xy: &[[F; 2]], occupied: &[bool]) -> Xyzz<F> {
    let mut plain = Xyzz::ZERO;
    xyzz_running_sums(xy, occupied, &mut plain)
}

/// [`xyzz_running_sum`], the plain sum of the buckets left in `plain`.
fn xyzz_running_sums<F: RawField>(
    xy: &[[F; 2]],
    occupied: &[bool],
    plain: &mut Xyzz<F>,
) -> Xyzz<F> {
    let mut running = Xyzz::<F>::ZERO;
    let mut total = Xyzz::<F>::ZERO;
    for j in (0..xy.len()).rev() {
        if occupied[j] {
            running.madd(xy[j][0], xy[j][1]);
        }
        total.add_assign(&running);
    }
    *plain = running;
    total
}

/// [`xyzz_running_sum`] past [`REDUCE_XYZZ`] buckets, at [`reduce_segment`].
fn segmented_running_sum<P: RawCurve>(xy: &[[P::RF; 2]], occupied: &[bool]) -> Xyzz<P::RF> {
    if xy.len() <= REDUCE_XYZZ {
        return xyzz_running_sum(xy, occupied);
    }
    segmented_running_sum_at::<P>(xy, occupied, reduce_segment(xy.len()))
}

/// Buckets per task of [`chunked_running_sum`]. A chunk is its own
/// [`segmented_running_sum`], so a shorter one shares each inversion between fewer
/// segments: 221 ns a bucket of G1 at 2048 against 189 over 2^15 at once.
const REDUCE_CHUNK: usize = 2048;

/// [`segmented_running_sum`] with the bucket range cut into chunks of [`REDUCE_CHUNK`],
/// one task each, so the reduction of a wide window is not one thread's to finish while
/// the rest of the pool has run out of points.
fn chunked_running_sum<P: RawCurve>(xy: &[[P::RF; 2]], occupied: &[bool]) -> Xyzz<P::RF> {
    chunked_running_sum_at::<P>(xy, occupied, REDUCE_CHUNK)
}

/// [`chunked_running_sum`] at a given chunk, a power of two. A chunk's sum counts its
/// buckets from 1, so chunk `r` still owes `r * chunk` times the plain sum of its
/// buckets, which is settled as the segments are in [`segmented_running_sum_at`].
fn chunked_running_sum_at<P: RawCurve>(
    xy: &[[P::RF; 2]],
    occupied: &[bool],
    chunk: usize,
) -> Xyzz<P::RF> {
    debug_assert!(chunk.is_power_of_two());
    if xy.len() <= chunk {
        return segmented_running_sum::<P>(xy, occupied);
    }
    let parts: Vec<_> = xy
        .par_chunks(chunk)
        .zip(occupied.par_chunks(chunk))
        .map(|(xy, occupied)| {
            let mut plain = Xyzz::ZERO;
            let sum = if xy.len() <= REDUCE_XYZZ {
                xyzz_running_sums(xy, occupied, &mut plain)
            } else {
                segmented_running_sums_at::<P>(
                    xy,
                    occupied,
                    reduce_segment(xy.len()),
                    Some(&mut plain),
                )
            };
            (sum, plain)
        })
        .collect();
    let mut sum = Xyzz::<P::RF>::ZERO;
    let mut running = Xyzz::<P::RF>::ZERO;
    let mut owed = Xyzz::<P::RF>::ZERO;
    for (r, (part, plain)) in parts.iter().enumerate().rev() {
        sum.add_assign(part);
        if r > 0 {
            running.add_assign(plain);
            owed.add_assign(&running);
        }
    }
    for _ in 0..chunk.trailing_zeros() {
        owed = owed.dbl();
    }
    sum.add_assign(&owed);
    sum
}

/// What one side of a segment's step needs from the shared inversion. A first point and a
/// cancellation need nothing and are settled before it.
#[derive(Clone, Copy, PartialEq)]
enum Step {
    Skip,
    Add,
    Dbl,
}

/// The running sum in affine coordinates. The bucket range is cut into segments of
/// `seg` (a power of two), each with its own running sum and total, and all segments
/// step together: per bucket two affine additions, `running += B_j` and
/// `total += running`, with one inversion shared by the whole step as in
/// [`BatchFill::flush`]. That is 12M + 2S a bucket with every multiply independent of
/// its neighbours, against 20M + 4S in one dependent chain. The total lags the running sum
/// by a step, so both additions of a step read the same settled values and need no
/// inversion of their own; one last step catches it up.
///
/// A segment's total counts its buckets from 1, so segment `s` still owes
/// `s * seg * R_s`, with `R_s` its final running sum. `sum_s s * R_s` is this same sum
/// over the segments, `seg` is a power of two, and the fix-up is one recursion and
/// `log2(seg)` doublings.
fn segmented_running_sum_at<P: RawCurve>(
    xy: &[[P::RF; 2]],
    occupied: &[bool],
    seg: usize,
) -> Xyzz<P::RF> {
    segmented_running_sums_at::<P>(xy, occupied, seg, None)
}

/// [`segmented_running_sum_at`], the plain sum of the buckets left in `plain` for a
/// caller that asks for it.
fn segmented_running_sums_at<P: RawCurve>(
    xy: &[[P::RF; 2]],
    occupied: &[bool],
    seg: usize,
    plain: Option<&mut Xyzz<P::RF>>,
) -> Xyzz<P::RF> {
    debug_assert!(seg.is_power_of_two());
    let n = xy.len();
    if n == 0 {
        if let Some(plain) = plain {
            *plain = Xyzz::ZERO;
        }
        return Xyzz::ZERO;
    }
    let k = n.div_ceil(seg);
    let mut run = vec![[P::RF::ZERO; 2]; k];
    let mut tot = vec![[P::RF::ZERO; 2]; k];
    let mut run_occ = vec![false; k];
    let mut tot_occ = vec![false; k];
    // The segments with an addition in the current step: segment, total side, running
    // side. The two sides' denominators go to one product chain each.
    let mut steps: Vec<(u32, Step, Step)> = Vec::with_capacity(k);
    let mut dens: Vec<[P::RF; 2]> = Vec::with_capacity(k);
    let mut prefix: Vec<[P::RF; 2]> = Vec::with_capacity(k);

    for t in 0..=seg {
        steps.clear();
        dens.clear();
        for s in 0..k {
            let [rx, ry] = run[s];
            let (mut step_t, mut den_t) = (Step::Skip, P::RF::ONE);
            if run_occ[s] {
                let [tx, ty] = tot[s];
                let dx = rx.sub(tx);
                if !tot_occ[s] {
                    tot[s] = run[s];
                    tot_occ[s] = true;
                } else if !dx.is_zero() {
                    (step_t, den_t) = (Step::Add, dx);
                } else if ry == ty {
                    (step_t, den_t) = (Step::Dbl, ty.double());
                } else {
                    tot_occ[s] = false;
                }
            }
            let (mut step_r, mut den_r) = (Step::Skip, P::RF::ONE);
            // High bucket first; the last step has no bucket, it only settles the totals.
            let j = s * seg + (seg - 1).wrapping_sub(t);
            if t < seg && j < n && occupied[j] {
                let [bx, by] = xy[j];
                let dx = bx.sub(rx);
                if !run_occ[s] {
                    run[s] = xy[j];
                    run_occ[s] = true;
                } else if !dx.is_zero() {
                    (step_r, den_r) = (Step::Add, dx);
                } else if by == ry {
                    (step_r, den_r) = (Step::Dbl, ry.double());
                } else {
                    // The xy stays: a pending total still reads it below.
                    run_occ[s] = false;
                }
            }
            if step_t != Step::Skip || step_r != Step::Skip {
                steps.push((s as u32, step_t, step_r));
                dens.push([den_t, den_r]);
            }
        }
        if steps.is_empty() {
            continue;
        }

        // Two product chains sharing the inversion, as in `BatchFill::flush`, so the
        // backward walk holds a segment's two additions at once.
        prefix.clear();
        let (mut a, mut c) = (P::RF::ONE, P::RF::ONE);
        for d in &dens {
            prefix.push([a, c]);
            a = a.mul(d[0]);
            c = c.mul(d[1]);
        }
        let ac = P::raw_inv(a.mul(c));
        let (mut ia, mut ic) = (ac.mul(c), ac.mul(a));
        for i in (0..steps.len()).rev() {
            let (s, step_t, step_r) = steps[i];
            let s = s as usize;
            let (dt, dr) = (ia.mul(prefix[i][0]), ic.mul(prefix[i][1]));
            ia = ia.mul(dens[i][0]);
            ic = ic.mul(dens[i][1]);
            let (r, tt) = (run[s], tot[s]);
            if step_t == Step::Add && step_r == Step::Add {
                // Both formulas interleaved, see `BatchFill::add2`.
                let [bx, by] = xy[s * seg + seg - 1 - t];
                let lt = r[1].sub(tt[1]).mul(dt);
                let lr = by.sub(r[1]).mul(dr);
                let st = lt.sqr();
                let sr = lr.sqr();
                let x3t = st.sub(tt[0]).sub(r[0]);
                let x3r = sr.sub(r[0]).sub(bx);
                let y3t = lt.mul(tt[0].sub(x3t)).sub(tt[1]);
                let y3r = lr.mul(r[0].sub(x3r)).sub(r[1]);
                tot[s] = [x3t, y3t];
                run[s] = [x3r, y3r];
                continue;
            }
            match step_t {
                Step::Add => tot[s] = affine_add(tt, r, dt),
                Step::Dbl => tot[s] = affine_dbl(tt, dt),
                Step::Skip => {}
            }
            match step_r {
                Step::Add => run[s] = affine_add(r, xy[s * seg + seg - 1 - t], dr),
                Step::Dbl => run[s] = affine_dbl(r, dr),
                Step::Skip => {}
            }
        }
    }

    let mut sum = Xyzz::<P::RF>::ZERO;
    for s in 0..k {
        if tot_occ[s] {
            sum.madd(tot[s][0], tot[s][1]);
        }
    }
    if let Some(plain) = plain {
        *plain = Xyzz::ZERO;
        for s in 0..k {
            if run_occ[s] {
                plain.madd(run[s][0], run[s][1]);
            }
        }
    }
    // Segment s weighs s, so segment 0 drops out and segment 1 is bucket 0 of the
    // recursion.
    let mut owed = segmented_running_sum::<P>(&run[1..], &run_occ[1..]);
    for _ in 0..seg.trailing_zeros() {
        owed = owed.dbl();
    }
    sum.add_assign(&owed);
    sum
}

/// `a + p` for two affine points of different x, `1 / (p.x - a.x)` in hand.
#[inline(always)]
fn affine_add<F: RawField>(a: [F; 2], p: [F; 2], d_inv: F) -> [F; 2] {
    let lambda = p[1].sub(a[1]).mul(d_inv);
    let x3 = lambda.sqr().sub(a[0]).sub(p[0]);
    [x3, lambda.mul(a[0].sub(x3)).sub(a[1])]
}

/// `2a` for an affine point, `1 / (2 a.y)` in hand.
#[inline(always)]
fn affine_dbl<F: RawField>(a: [F; 2], d_inv: F) -> [F; 2] {
    let xx = a[0].sqr();
    let lambda = xx.double().add(xx).mul(d_inv);
    let x3 = lambda.sqr().sub(a[0].double());
    [x3, lambda.mul(a[0].sub(x3)).sub(a[1])]
}

/// Montgomery's trick over nonzero elements, in place.
fn batch_invert<P: RawCurve>(v: &mut [P::RF]) {
    if v.is_empty() {
        return;
    }
    let mut prefix = Vec::with_capacity(v.len());
    let mut acc = P::RF::ONE;
    for d in v.iter() {
        prefix.push(acc);
        acc = acc.mul(*d);
    }
    let mut inv = P::raw_inv(acc);
    for (d, p) in v.iter_mut().zip(prefix).rev() {
        let next = inv.mul(*d);
        *d = inv.mul(p);
        inv = next;
    }
}

/// Add every live side bucket into its affine bucket, so the running sum only sees affine
/// points. Two inversions for the lot: one takes the side points to affine, with
/// `1 / ZZ = (ZZ / ZZZ)^2` by the XYZZ invariant, the other is the additions'.
fn fold_side<P: RawCurve>(xy: &mut [[P::RF; 2]], occupied: &mut [bool], side: &SideBuckets<P::RF>) {
    let live: Vec<usize> = (0..xy.len())
        .filter(|&j| side.slot[j] != 0 && !side[j].is_zero())
        .collect();
    let mut inv: Vec<P::RF> = live.iter().map(|&j| side[j].zzz).collect();
    batch_invert::<P>(&mut inv);
    let points = live.iter().zip(&inv).map(|(&j, &izzz)| {
        let p = &side[j];
        (j, [p.x.mul(p.zz.mul(izzz).sqr()), p.y.mul(izzz)])
    });
    add_points::<P>(xy, occupied, points);
}

/// `bucket j += p` for each `(j, p)`, no bucket twice, behind one inversion.
fn add_points<P: RawCurve>(
    xy: &mut [[P::RF; 2]],
    occupied: &mut [bool],
    points: impl Iterator<Item = (usize, [P::RF; 2])>,
) {
    let mut adds: Vec<(usize, [P::RF; 2], Step)> = Vec::with_capacity(points.size_hint().0);
    let mut dens: Vec<P::RF> = Vec::with_capacity(points.size_hint().0);
    for (j, p) in points {
        let dx = p[0].sub(xy[j][0]);
        if !occupied[j] {
            xy[j] = p;
            occupied[j] = true;
        } else if !dx.is_zero() {
            adds.push((j, p, Step::Add));
            dens.push(dx);
        } else if p[1] == xy[j][1] {
            adds.push((j, p, Step::Dbl));
            dens.push(p[1].double());
        } else {
            occupied[j] = false;
        }
    }
    batch_invert::<P>(&mut dens);
    for (&(j, p, step), &d_inv) in adds.iter().zip(&dens) {
        xy[j] = match step {
            Step::Add => affine_add(xy[j], p, d_inv),
            _ => affine_dbl(xy[j], d_inv),
        };
    }
}

/// Settle every fill of one window and add the rest into the first, bucket by bucket,
/// which then holds the window in affine buckets alone. A bucket that is empty on either
/// side costs nothing, so the fills together make as many additions as one fill over
/// all their points would have. The bucket range is cut as in [`chunked_running_sum`].
fn merge_fills<P: RawCurve>(fills: &mut [BatchFill<P>]) {
    fills.par_iter_mut().for_each(|fill| {
        fill.settle();
        if let Some(side) = fill.side.take() {
            fold_side::<P>(&mut fill.xy, &mut fill.occupied, &side);
        }
    });
    let Some((first, rest)) = fills.split_first_mut() else {
        return;
    };
    if rest.is_empty() {
        return;
    }
    let rest = &*rest;
    first
        .xy
        .par_chunks_mut(REDUCE_CHUNK)
        .zip(first.occupied.par_chunks_mut(REDUCE_CHUNK))
        .enumerate()
        .for_each(|(r, (xy, occupied))| {
            let lo = r * REDUCE_CHUNK;
            for other in rest {
                let theirs = other.xy[lo..][..xy.len()].iter().copied().enumerate();
                let live = theirs.filter(|&(j, _)| other.occupied[lo + j]);
                add_points::<P>(xy, occupied, live);
            }
        });
}

/// [`window_chunk`]'s bucket visits with the fill batched: affine buckets, additions
/// applied in rounds that share one field inversion. See [`BatchFill`] for the machinery
/// and the conflict story. Only worth it with buckets to spare; [`pippenger`] switches on
/// [`MIN_BATCH`].
fn window_fill<P: RawCurve>(
    fill: &mut BatchFill<P>,
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    range: core::ops::Range<usize>,
    window: usize,
    c: u32,
    work: Work,
) {
    // See `window_chunk`. XYZZ rather than a batched slot: a zero scalar sends every one
    // of its digits here, and one batched bucket that contended would push them all through
    // the retry and side-bucket machinery instead.
    let mut dummy = Xyzz::<P::RF>::ZERO;
    for k in range {
        let d = signed_digit(scan.bigints[k].as_ref(), window, c);
        // Never infinity: prescan filtered those, so raw_xy is total here.
        let (x, y) = P::raw_xy(&bases[scan.idx[k] as usize]);
        if d == 0 {
            if work == Work::Constant {
                dummy.madd(x, y);
            }
            continue;
        }
        if d > 0 {
            fill.insert((d - 1) as usize, x, y);
        } else {
            fill.insert((-d - 1) as usize, x, y.neg());
        }
    }
    core::hint::black_box(&dummy);
}

/// One range of one window through [`window_fill`] and reduced on its own, which is what
/// the tests hold the other fills to.
#[cfg(test)]
fn window_chunk_batch<P: RawCurve>(
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    range: core::ops::Range<usize>,
    window: usize,
    c: u32,
    work: Work,
) -> Projective<P> {
    // `d` is in [-2^(c-1), 2^(c-1)], so `2^(c-1)` buckets is tight.
    let n_buckets = 1usize << (c - 1);
    let mut fill = BatchFill::<P>::new(n_buckets, batch_size(n_buckets));
    window_fill(&mut fill, bases, scan, range, window, c, work);
    fill.finish()
}

/// Points per piece of [`ganged_windows`] for a window of `n_buckets`. A piece is what a
/// thread commits to at a time, so the last one out of a slow core is what the pool waits
/// for at the end of the MSMs; a thread that joins a window for one piece pays for
/// clearing and later scanning a bucket array of its own, which bounds it from below.
fn split_grain(n_buckets: usize) -> usize {
    (n_buckets / GRAIN_BUCKET_RATIO).max(MIN_GRAIN)
}

/// One window of [`ganged_windows`]: its point range as numbered pieces that threads
/// claim, and the fills of the threads that claimed any.
struct Gang<P: RawCurve> {
    /// The next unclaimed piece. Runs past the piece count once the window is claimed out.
    next: AtomicUsize,
    /// Threads between [`Self::enter`] and [`Self::leave`], for the next thread's choice.
    inside: AtomicUsize,
    /// The fills of the threads that have left, and whether the window has been closed.
    left: Mutex<(Vec<BatchFill<P>>, bool)>,
    sum: OnceLock<Projective<P>>,
}

impl<P: RawCurve> Gang<P> {
    fn new() -> Self {
        Gang {
            next: AtomicUsize::new(0),
            inside: AtomicUsize::new(0),
            left: Mutex::new((Vec::new(), false)),
            sum: OnceLock::new(),
        }
    }

    /// Whether the window still takes threads. Counted in before the first claim, so the
    /// window cannot close under a thread that holds a piece.
    fn enter(&self) -> bool {
        let left = self.left.lock().unwrap();
        if !left.1 {
            self.inside.fetch_add(1, Ordering::Relaxed);
        }
        !left.1
    }

    /// Hand in the fill, if any piece was claimed. The last thread out of a window with
    /// no piece left gets every fill back and closes it.
    fn leave(&self, fill: Option<BatchFill<P>>, pieces: usize) -> Option<Vec<BatchFill<P>>> {
        let mut left = self.left.lock().unwrap();
        left.0.extend(fill);
        let inside = self.inside.fetch_sub(1, Ordering::Relaxed) - 1;
        if inside > 0 || self.next.load(Ordering::Relaxed) < pieces {
            return None;
        }
        left.1 = true;
        Some(core::mem::take(&mut left.0))
    }
}

/// Every window of an MSM through the batch-affine fill, on `threads` threads of the pool.
///
/// The tasks are not the windows. A window is a list of pieces of `grain` points, and a
/// thread takes a whole window that nobody has started while there is one, then joins
/// the started window with the most unclaimed pieces a thread, with a fill of its own.
/// So a window one thread works alone is one fill, the windows left at the end are
/// shared out piece by piece, and no thread waits on another's window: with equal tasks
/// a window each, 17 of them on eight fast and four slow cores had the fast ones idle
/// while the slow ones finished their first. The last thread out of a window adds its
/// fills together ([`merge_fills`]) and reduces once ([`chunked_running_sum`]), both cut
/// into tasks for whoever is free.
fn ganged_windows<P: RawCurve>(
    bases: &[Affine<P>],
    scan: &Prescan<P>,
    c: u32,
    work: Work,
    threads: usize,
    grain: usize,
) -> Vec<Projective<P>>
where
    P::BaseField: Send + Sync,
{
    let m = scan.idx.len();
    // `d` is in [-2^(c-1), 2^(c-1)], so `2^(c-1)` buckets is tight.
    let n_buckets = 1usize << (c - 1);
    let pieces = m.div_ceil(grain);
    let gangs: Vec<Gang<P>> = (0..RECODE_BITS.div_ceil(c as usize))
        .map(|_| Gang::new())
        .collect();
    let fresh = AtomicUsize::new(0);
    let pick = || {
        let w = fresh.fetch_add(1, Ordering::Relaxed);
        if w < gangs.len() {
            return Some(w);
        }
        let share = |g: &Gang<P>| {
            let unclaimed = pieces.saturating_sub(g.next.load(Ordering::Relaxed));
            (unclaimed << 8) / (g.inside.load(Ordering::Relaxed) + 1)
        };
        let (w, most) = gangs.iter().map(share).enumerate().max_by_key(|s| s.1)?;
        (most > 0).then_some(w)
    };
    let run = || {
        while let Some(w) = pick() {
            let gang = &gangs[w];
            if !gang.enter() {
                continue;
            }
            let mut fill = None;
            loop {
                let piece = gang.next.fetch_add(1, Ordering::Relaxed);
                if piece >= pieces {
                    break;
                }
                let fill =
                    fill.get_or_insert_with(|| BatchFill::new(n_buckets, batch_size(n_buckets)));
                let range = piece * grain..((piece + 1) * grain).min(m);
                window_fill(fill, bases, scan, range, w, c, work);
            }
            if let Some(mut fills) = gang.leave(fill, pieces) {
                merge_fills(&mut fills);
                // No fill at all is an MSM of no points.
                let sum = fills.first().map_or(Xyzz::ZERO, |fill| {
                    chunked_running_sum::<P>(&fill.xy, &fill.occupied)
                });
                // Closed once, by this thread.
                let _ = gang.sum.set(to_projective(&sum));
            }
        }
    };
    rayon::scope(|s| {
        for _ in 1..threads {
            s.spawn(|_| run());
        }
        run();
    });
    gangs
        .into_iter()
        .map(|g| g.sum.into_inner().expect("every window was closed"))
        .collect()
}

/// A per-group replacement for [`window_fill`], prepared once per MSM. Only an
/// x86_64 with AVX-512 IFMA has one (see `ifma`); everything else keeps the portable fill.
trait FastFill: RawCurve {
    /// Whatever the replacement needs per MSM, built from the prescan.
    type Prepared: Sync;
    /// `None` when the replacement cannot run here.
    fn prepare(bases: &[Affine<Self>], scan: &Prescan<Self>) -> Option<Self::Prepared>;
    /// The replacement for [`OnesFold`].
    type Ones: OnesSum<Self>;
    /// `None` when the replacement cannot run here.
    fn ones() -> Option<Self::Ones>;
    fn window_chunk(
        prep: &Self::Prepared,
        bases: &[Affine<Self>],
        scan: &Prescan<Self>,
        range: core::ops::Range<usize>,
        window: usize,
        c: u32,
        work: Work,
    ) -> Projective<Self>;
}

/// No replacement: `prepare` always declines, so `window_chunk` is never reached.
#[cfg(not(target_arch = "x86_64"))]
macro_rules! no_fast_fill {
    ($cfg:ty) => {
        impl FastFill for $cfg {
            type Prepared = core::convert::Infallible;
            fn prepare(_: &[Affine<Self>], _: &Prescan<Self>) -> Option<Self::Prepared> {
                None
            }
            type Ones = OnesFold<Self>;
            fn ones() -> Option<OnesFold<Self>> {
                None
            }
            fn window_chunk(
                prep: &Self::Prepared,
                _: &[Affine<Self>],
                _: &Prescan<Self>,
                _: core::ops::Range<usize>,
                _: usize,
                _: u32,
                _: Work,
            ) -> Projective<Self> {
                match *prep {}
            }
        }
    };
}

#[cfg(not(target_arch = "x86_64"))]
no_fast_fill!(snarkrs_field::g1::Config);
#[cfg(not(target_arch = "x86_64"))]
no_fast_fill!(snarkrs_field::g2::Config);

#[cfg(target_arch = "x86_64")]
mod ifma;

fn pippenger<P: FastFill>(
    bases: &[Affine<P>],
    scalars: &[Fr],
    threads: usize,
    work: Work,
) -> Projective<P>
where
    P::BaseField: Send + Sync,
{
    // A hard assert, not a debug one: a length mismatch is a malformed key and must
    // fail in release too.
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
    pippenger_scanned(bases, prescan(bases, scalars, n, work), threads, work)
}

/// [`pippenger`] from a finished prescan.
fn pippenger_scanned<P: FastFill>(
    bases: &[Affine<P>],
    scan: Prescan<'_, P>,
    threads: usize,
    work: Work,
) -> Projective<P>
where
    P::BaseField: Send + Sync,
{
    let m = scan.idx.len();
    // Nothing reached the general path: every scalar was 0 or 1, or its base was
    // infinity. Skip Pippenger rather than allocate W bucket arrays for nothing.
    if m == 0 {
        return scan.ones_sum;
    }

    // The window is sized for the work that actually reaches the buckets, not for the
    // input length: a 2^20-long witness with 2^10 general scalars is a 2^10 problem.
    pippenger_windowed(bases, scan, window_size(m), threads, work)
}

/// [`pippenger_scanned`] at a given window, which the forced-window sweep sets directly.
fn pippenger_windowed<P: FastFill>(
    bases: &[Affine<P>],
    scan: Prescan<'_, P>,
    c: u32,
    threads: usize,
    work: Work,
) -> Projective<P>
where
    P::BaseField: Send + Sync,
{
    let m = scan.idx.len();
    let n_windows = RECODE_BITS.div_ceil(c as usize);

    // Windows are independent, so they are the first axis of parallelism, and at BN254
    // sizes there are usually more of them (17 to 20) than there are cores. The wide
    // windows, the batch-affine ones, go to [`ganged_windows`], which shares the point
    // range of the last windows out between the threads. The XYZZ fill and the x86 one
    // keep the older rule: when there are fewer windows than twice the pool, split the
    // point range too and give each chunk its own bucket array, reduced on its own.
    // Summing the per-chunk reduced sums is exact because the running-sum reduction is
    // linear in the buckets. Twice the pool rather than once: 16 windows on 12 threads are
    // two waves of equal tasks, the second a third full, and with four of the twelve being
    // efficiency cores at under half a performance core's pace the last task out sets the
    // wall. On anon-aadhaar's H (2^21 points, c = 16, M2 Max) one chunk a window measured
    // 588 ms, two 535, three 531, four 537: past that each chunk is another bucket
    // reduction. Under this rule the forced-window sweep at [`window_size`] still has its
    // pick best or tied.
    // Signed digits land in [-2^(c-1), 2^(c-1)] and index as `d - 1` when positive and
    // `-d - 1` when negative, so the highest reachable index is 2^(c-1) - 1 and 2^(c-1)
    // buckets cover every digit.
    let n_buckets = 1usize << (c - 1);
    let point_chunks = (2 * threads.max(1))
        .div_ceil(n_windows)
        .clamp(1, (m / (CHUNK_BUCKET_RATIO * n_buckets)).max(1));
    let chunk_len = m.div_ceil(point_chunks);
    let batched = batch_size(n_buckets) >= MIN_BATCH;
    let fast = if batched {
        P::prepare(bases, &scan)
    } else {
        None
    };

    let window_sums: Vec<Projective<P>> = if batched && fast.is_none() {
        ganged_windows(
            bases,
            &scan,
            c,
            work,
            threads.max(1),
            split_grain(n_buckets),
        )
    } else {
        (0..n_windows)
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
                        // The narrow windows stay on the XYZZ fill, which has no round
                        // overhead to amortise.
                        if let Some(prep) = &fast {
                            P::window_chunk(prep, bases, &scan, lo..hi, w, c, work)
                        } else {
                            window_chunk(bases, &scan, lo..hi, w, c, work)
                        }
                    })
                    .reduce(Projective::zero, |a, b| a + b)
            })
            .collect()
    };

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
mod batch_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_std::{rand::Rng, test_rng, UniformRand};
    use snarkrs_field::Field;

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
    fn fallible_defaults_reject_both_length_mismatches() {
        let cpu = CpuMsm::new();
        let backend: &dyn MsmBackend = &cpu;
        let g1 = [G1Affine::identity(); 2];
        let g2 = [G2Affine::identity(); 2];
        let scalars = [Fr::one(); 2];
        for (bases, scalars_len) in [(0, 1), (1, 0), (1, 2), (2, 1)] {
            assert!(matches!(
                backend.try_msm_g1(&g1[..bases], &scalars[..scalars_len]),
                Err(AccelError::Shape { op: "msm_g1", .. })
            ));
            assert!(matches!(
                backend.try_msm_g2(&g2[..bases], &scalars[..scalars_len]),
                Err(AccelError::Shape { op: "msm_g2", .. })
            ));
        }
    }

    #[test]
    fn checked_group_fft_rejects_non_power_of_two_blocks() {
        use snarkrs_field::raw::{RawFq, RawFq2};
        struct Noop;
        impl GroupFft for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn ifft_g1(&self, _: &mut [Xyzz<RawFq>]) -> Result<(), AccelError> {
                Ok(())
            }
            fn ifft_g2(&self, _: &mut [Xyzz<RawFq2>]) -> Result<(), AccelError> {
                Ok(())
            }
        }
        let fft = CheckedGroupFft::<dyn GroupFft>(Box::new(Noop));
        let mut g1 = [Xyzz::<RawFq>::ZERO; 8];
        let mut g2 = [Xyzz::<RawFq2>::ZERO; 8];
        for n in [0, 3, 5, 6, 7] {
            assert!(matches!(
                fft.ifft_g1(&mut g1[..n]),
                Err(AccelError::Shape { .. })
            ));
            assert!(matches!(
                fft.ifft_g2(&mut g2[..n]),
                Err(AccelError::Shape { .. })
            ));
        }
        for n in [1, 2, 4, 8] {
            fft.ifft_g1(&mut g1[..n]).unwrap();
            fft.ifft_g2(&mut g2[..n]).unwrap();
        }
        let (a, b) = g1.split_at_mut(4);
        let (c, _) = b.split_at_mut(3);
        assert!(matches!(
            fft.ifft_g1_many(&mut [a, c]),
            Err(AccelError::Shape { .. })
        ));
        assert_eq!(fft.name(), "noop");
    }

    #[test]
    fn fallible_defaults_match_legacy_and_reference() {
        let cpu = CpuMsm::new();
        let backend: &dyn MsmBackend = &cpu;
        assert!(backend.try_msm_g1(&[], &[]).unwrap().is_zero());
        assert!(backend.try_msm_g2(&[], &[]).unwrap().is_zero());
        let mut rng = test_rng();
        let g1 = rand_points::<snarkrs_field::g1::Config>(33, &mut rng);
        let g2 = rand_points::<snarkrs_field::g2::Config>(33, &mut rng);
        let scalars = rand_scalars(33, &mut rng);
        assert_eq!(
            backend.try_msm_g1(&g1, &scalars).unwrap(),
            naive(&g1, &scalars)
        );
        assert_eq!(
            backend.try_msm_g2(&g2, &scalars).unwrap(),
            naive(&g2, &scalars)
        );
        assert_eq!(
            backend.try_msm_g1(&g1, &scalars).unwrap(),
            backend.msm_g1(&g1, &scalars)
        );
        assert_eq!(
            backend.try_msm_g2(&g2, &scalars).unwrap(),
            backend.msm_g2(&g2, &scalars)
        );
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

    /// [`OnesFold`] and, where the CPU has one, the [`FastFill::Ones`] replacement.
    fn ones_fold_case<P: FastFill>(points: &[Affine<P>]) {
        let mut fold = OnesFold::<P>::new();
        let mut any = AnyOnes::<P>::new();
        for p in points {
            let (x, y) = P::raw_xy(p);
            fold.add(x, y);
            any.add(x, y);
        }
        let want = points.iter().fold(Projective::<P>::zero(), |a, p| a + p);
        assert_eq!(fold.finish(), want, "n = {}", points.len());
        assert_eq!(any.finish(), want, "n = {}, fast path", points.len());
    }

    /// Every size at which the fold changes path: nothing to add, too few points for an
    /// inversion, an odd one out, a full buffer and several of them.
    #[test]
    fn ones_fold_matches_the_plain_sum() {
        let mut rng = test_rng();
        let small = [0, 1, 2, 3, ONES_MIN - 1, ONES_MIN, ONES_MIN + 1];
        let full = [ONES_BATCH, ONES_BATCH + 1, 3 * ONES_BATCH + 5];
        for n in small.into_iter().chain(full) {
            ones_fold_case(&walk_points::<snarkrs_field::g1::Config>(n, &mut rng));
            ones_fold_case(&walk_points::<snarkrs_field::g2::Config>(n, &mut rng));
        }
    }

    /// Pairs the shared inversion cannot take: the same point twice, a point and its
    /// negative, and a buffer of nothing else.
    #[test]
    fn ones_fold_survives_equal_and_opposite_points() {
        fn case<P: FastFill>(rng: &mut impl Rng)
        where
            Projective<P>: UniformRand,
        {
            // A full buffer pairs `k` with `k + ONES_BATCH / 2`.
            let half = ONES_BATCH / 2;
            let mut points: Vec<Affine<P>> = walk_points(3 * ONES_BATCH, rng);
            for k in (0..half - 3).step_by(6) {
                points[k + half] = points[k];
                points[k + 3 + half] = -points[k + 3];
            }
            ones_fold_case(&points);
            ones_fold_case(&[points[0]; 2 * ONES_BATCH + 3]);
            let opposite: Vec<Affine<P>> = (0..ONES_BATCH + 2)
                .map(|k| if k < half { points[0] } else { -points[0] })
                .collect();
            ones_fold_case(&opposite);
        }
        let mut rng = test_rng();
        case::<snarkrs_field::g1::Config>(&mut rng);
        case::<snarkrs_field::g2::Config>(&mut rng);
    }

    /// No scalar equal to one, one, two, half of them and all of them, across the
    /// parallel scan's chunks.
    #[test]
    fn one_scalars_at_every_count() {
        let mut rng = test_rng();
        let n = 4 * SCAN_CHUNK + 77;
        let g1: Vec<G1Affine> = walk_points(n, &mut rng);
        let g2: Vec<G2Affine> = walk_points(n, &mut rng);
        let msm = CpuMsm::new();
        for ones in [0, 1, 2, n / 2, n] {
            let mut s = rand_scalars(n, &mut rng);
            for k in 0..ones {
                s[(k * 7919) % n] = Fr::one();
            }
            let want1 = G1Projective::msm(&g1, &s).unwrap();
            assert_eq!(msm.msm_g1(&g1, &s), want1, "{ones} ones");
            assert_eq!(msm.msm_g2(&g2, &s), G2Projective::msm(&g2, &s).unwrap());
            assert_eq!(msm.msm_g1_in_place(&g1, &mut s), want1, "{ones} ones");
        }
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
        // r - 1 is the largest scalar there is, and the one that stresses the top window.
        s[0] = -Fr::one();
        s[1] = Fr::from(2u64);
        s[2] = Fr::zero();
        s[3] = Fr::one();
        // Fits in one limb, so every window above the lowest handful is zero.
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

    /// The in-place scan against the copying one, across chunk boundaries: a dense vector
    /// where no chunk moves, a witness-like one where every chunk after the first does,
    /// and bases at infinity dropping entries mid-chunk.
    #[test]
    fn in_place_scan_matches_the_copying_scan() {
        let mut rng = test_rng();
        for n in [7, 300, 3 * SCAN_CHUNK + 123] {
            let mut g1: Vec<G1Affine> = walk_points(n, &mut rng);
            for i in (5..n).step_by(4099) {
                g1[i] = G1Affine::identity();
            }
            let all_zero = vec![Fr::zero(); n];
            for s in [
                rand_scalars(n, &mut rng),
                witness_like_scalars(n, &mut rng),
                all_zero,
            ] {
                for work in [Work::Variable, Work::Constant] {
                    let want = prescan(&g1, &s, n, work);
                    let mut buf = s.clone();
                    let got = prescan_in_place(&g1, &mut buf, work);
                    assert_eq!(got.idx, want.idx, "n = {n}");
                    assert_eq!(got.bigints[..], want.bigints[..], "n = {n}");
                    assert_eq!(got.ones_sum, want.ones_sum, "n = {n}");
                }
                for msm in [CpuMsm::new(), CpuMsm::constant_work()] {
                    let mut buf = s.clone();
                    assert_eq!(msm.msm_g1_in_place(&g1, &mut buf), msm.msm_g1(&g1, &s));
                }
            }
        }
        assert!(CpuMsm::new().msm_g1_in_place(&[], &mut []).is_zero());
    }

    /// `scrub` zeroes every limb across chunk boundaries, and a borrowed scan leaves its
    /// owner's buffer alone when it drops.
    #[test]
    fn scrub_zeroes_and_a_borrowed_scan_does_not() {
        let mut rng = test_rng();
        let n = 2 * SCAN_CHUNK + 5;
        let mut v: Vec<BigInt> = rand_scalars(n, &mut rng)
            .iter()
            .map(|s| s.into_bigint())
            .collect();
        scrub(&mut v);
        assert!(v.iter().all(|x| x.0 == [0; 4]));

        let g1: Vec<G1Affine> = walk_points(n, &mut rng);
        let mut buf = rand_scalars(n, &mut rng);
        let want: Vec<BigInt> = buf.iter().map(|s| s.into_bigint()).collect();
        drop(prescan_in_place(&g1, &mut buf, Work::Constant));
        assert!(buf.iter().zip(&want).all(|(s, w)| s.0 == *w));
    }

    /// Drives the batch-affine fill against the XYZZ fill window by window, at windows
    /// narrow enough that every scheduling path fires: direct sets, pending rounds,
    /// retries, and the side buckets.
    fn batch_fill_case<P: RawCurve>(bases: &[Affine<P>], scalars: &[Fr]) {
        let scan = Prescan::<P> {
            idx: (0..bases.len() as u32).collect(),
            bigints: scalars.iter().map(|s| s.into_bigint()).collect(),
            ones_sum: Projective::zero(),
        };
        for c in [5u32, 8] {
            for w in 0..RECODE_BITS.div_ceil(c as usize) {
                assert_eq!(
                    window_chunk_batch(bases, &scan, 0..bases.len(), w, c, Work::Constant),
                    window_chunk(bases, &scan, 0..bases.len(), w, c, Work::Constant),
                    "window {w} at c = {c}"
                );
            }
        }
    }

    /// A base set with duplicates (the flush's doubling case) and cancelling pairs (its
    /// cancellation case), under shared scalars so they meet in the same buckets.
    #[test]
    fn batch_fill_matches_the_xyzz_fill() {
        let mut rng = test_rng();
        let n = 600;
        let mut g1: Vec<G1Affine> = rand_points(n, &mut rng);
        let mut s = rand_scalars(n, &mut rng);
        for i in 0..n / 4 {
            // One point under one scalar: every window funnels into a single bucket,
            // which is the contention the retry queue and side buckets exist for.
            g1[i] = g1[0];
            s[i] = s[0];
        }
        for i in n / 4..n / 2 {
            // Cancelling pairs: the same base negated, under one scalar, meets its
            // partner in every bucket it touches.
            g1[i] = if i % 2 == 0 { g1[n / 4] } else { -g1[n / 4] };
            s[i] = s[n / 4];
        }
        s[n - 1] = Fr::zero();
        batch_fill_case::<snarkrs_field::g1::Config>(&g1, &s);

        let mut g2: Vec<G2Affine> = rand_points(n / 2, &mut rng);
        for i in 0..n / 8 {
            g2[i] = if i % 2 == 0 { g2[0] } else { -g2[0] };
            s[i] = s[1];
        }
        batch_fill_case::<snarkrs_field::g2::Config>(&g2, &s[..n / 2]);
    }

    /// 2^15 points is where [`window_size`] first picks a window wide enough (c = 12,
    /// 2048 buckets) for the batch-affine fill to run at the full [`BATCH`].
    #[test]
    fn matches_arkworks_g1_through_the_batch_fill() {
        let mut rng = test_rng();
        let n = 1usize << 15;
        assert_eq!(
            batch_size(1usize << (window_size(n) - 1)),
            BATCH,
            "2^15 no longer reaches the full batch; move this test"
        );
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let want = G1Projective::msm(&bases, &scalars).unwrap();
        assert_eq!(CpuMsm::new().msm_g1(&bases, &scalars), want);
    }

    /// One base, one scalar value, 2^15 copies: every point of every window lands in
    /// the same bucket, so the whole MSM runs down the retry and side-bucket path.
    #[test]
    fn batch_fill_survives_a_single_hot_bucket() {
        let mut rng = test_rng();
        let n = 1usize << 15;
        let p: Vec<G1Affine> = rand_points(1, &mut rng);
        let s = Fr::rand(&mut rng);
        let want = p[0] * (s * Fr::from(n as u64));
        assert_eq!(CpuMsm::new().msm_g1(&vec![p[0]; n], &vec![s; n]), want);
    }

    /// The first lengths choosing c = 9, 10 and 11: the batch fill below the full
    /// [`BATCH`], at half the bucket count (128 to 512).
    fn narrow_batch_fill<P: FastFill<ScalarField = Fr>>()
    where
        Projective<P>: UniformRand,
        P::BaseField: Send + Sync,
    {
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        for (n, c) in [(3_329, 9), (5_889, 10), (16_897, 11)] {
            assert_eq!(window_size(n), c);
            let batch = batch_size(1usize << (c - 1));
            assert!((MIN_BATCH..BATCH).contains(&batch), "c = {c}");
            let bases: Vec<Affine<P>> = walk_points(n, &mut rng);
            let scalars = rand_scalars(n, &mut rng);
            let want = Projective::<P>::msm(&bases, &scalars).unwrap();
            assert_eq!(
                pippenger(&bases, &scalars, msm.threads, Work::Variable),
                want
            );
            assert_eq!(pippenger(&bases, &scalars, 1, Work::Constant), want);
        }

        // Degenerate digits at the smallest batch. Equal scalars put every point of a
        // window in one bucket; equal bases on top make each round all doublings.
        let n = 3_329;
        let bases: Vec<Affine<P>> = walk_points(n, &mut rng);
        let s = Fr::rand(&mut rng);
        let want = bases.iter().fold(Projective::<P>::zero(), |acc, b| acc + b) * s;
        assert_eq!(
            pippenger(&bases, &vec![s; n], msm.threads, Work::Variable),
            want
        );
        let hot = vec![bases[0]; n];
        assert_eq!(
            pippenger(&hot, &vec![s; n], msm.threads, Work::Variable),
            bases[0] * (s * Fr::from(n as u64))
        );
    }

    #[test]
    fn matches_arkworks_through_the_narrow_batch_fill_g1() {
        narrow_batch_fill::<snarkrs_field::g1::Config>();
    }

    #[test]
    fn matches_arkworks_through_the_narrow_batch_fill_g2() {
        narrow_batch_fill::<snarkrs_field::g2::Config>();
    }

    /// `cargo test -p snarkrs-msm --release -- --ignored --nocapture bench`
    ///
    /// Best of three, because this runs on a shared machine and the mean of a contended
    /// run measures the other tenants.
    #[test]
    #[ignore = "benchmark, meaningless in a debug build"]
    fn bench_against_arkworks() {
        use std::time::Instant;
        let mut rng = test_rng();
        let msm = CpuMsm::new();
        let serial = CpuMsm {
            threads: 1,
            ..CpuMsm::new()
        };
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

    /// `cargo test -p snarkrs-msm --release -- --ignored --nocapture bench_ones_fold`
    ///
    /// One thread, [`OnesFold`] against the chain of mixed additions it replaced, at the
    /// point counts a scan chunk of a bit-heavy witness holds and well past them. Best of
    /// seven, the two taking turns.
    #[test]
    #[ignore = "benchmark, meaningless in a debug build"]
    fn bench_ones_fold() {
        use std::time::Instant;
        fn case<P: FastFill>(name: &str, rng: &mut impl Rng)
        where
            Projective<P>: UniformRand,
        {
            for n in [200usize, 1000, 4096, 65536] {
                let points: Vec<Affine<P>> = walk_points(n, rng);
                let reps = (1 << 20) / n;
                let ns = |t: Instant| t.elapsed().as_secs_f64() * 1e9 / (reps * n) as f64;
                let mut best = [f64::MAX; 3];
                let mut out = [Projective::<P>::zero(); 3];
                for _ in 0..7 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        let mut chain = Xyzz::<P::RF>::ZERO;
                        for p in &points {
                            let (x, y) = P::raw_xy(p);
                            chain.madd(x, y);
                        }
                        out[0] = to_projective(&chain);
                    }
                    best[0] = best[0].min(ns(t));
                    let t = Instant::now();
                    for _ in 0..reps {
                        let mut fold = OnesFold::<P>::new();
                        for p in &points {
                            let (x, y) = P::raw_xy(p);
                            fold.add(x, y);
                        }
                        out[1] = fold.finish();
                    }
                    best[1] = best[1].min(ns(t));
                    let t = Instant::now();
                    for _ in 0..reps {
                        let mut fold = AnyOnes::<P>::new();
                        for p in &points {
                            let (x, y) = P::raw_xy(p);
                            fold.add(x, y);
                        }
                        out[2] = fold.finish();
                    }
                    best[2] = best[2].min(ns(t));
                }
                assert_eq!(out[0], out[1]);
                assert_eq!(out[0], out[2]);
                println!(
                    "{name} n={n:6}  madd chain {:6.1} ns  fold {:6.1}  fast fold {:6.1} ns a point",
                    best[0], best[1], best[2]
                );
            }
        }
        let mut rng = test_rng();
        case::<snarkrs_field::g1::Config>("G1", &mut rng);
        case::<snarkrs_field::g2::Config>("G2", &mut rng);
    }

    /// `cargo test -p snarkrs-msm --release -- --ignored --nocapture sweep_forced_window`
    ///
    /// Times the bucket phase at each window from 12 to 16 against what `window_size`
    /// picks, on dense scalars (the H MSM's shape) at 2^18 and 2^21 points. The windows
    /// take turns within each rep so drift on a shared machine lands on all of them; the
    /// prescan is outside the clock because it does not depend on the window.
    #[test]
    #[ignore = "benchmark, meaningless in a debug build"]
    fn sweep_forced_window() {
        use std::time::Instant;
        let mut rng = test_rng();
        let threads = rayon::current_num_threads();
        const REPS: usize = 7;
        println!("rayon threads: {threads}");
        for log in [18u32, 21] {
            let n = 1usize << log;
            let bases: Vec<G1Affine> = walk_points(n, &mut rng);
            let scalars = rand_scalars(n, &mut rng);
            let windows: Vec<u32> = (12..=MAX_WINDOW).collect();
            let mut ms = vec![Vec::with_capacity(REPS); windows.len()];
            let mut want = None;
            for _ in 0..REPS {
                for (i, &c) in windows.iter().enumerate() {
                    let scan = prescan(&bases, &scalars, n, Work::Variable);
                    let t = Instant::now();
                    let got = pippenger_windowed(&bases, scan, c, threads, Work::Variable);
                    ms[i].push(t.elapsed().as_secs_f64() * 1e3);
                    assert_eq!(*want.get_or_insert(got), got, "c = {c}");
                }
            }
            for (c, mut t) in windows.into_iter().zip(ms) {
                t.sort_by(f64::total_cmp);
                println!(
                    "2^{log} G1  c={c:2}{}  median {:8.2} ms  min {:8.2}  max {:8.2}",
                    if c == window_size(n) { "*" } else { " " },
                    t[REPS / 2],
                    t[0],
                    t[REPS - 1],
                );
            }
        }
    }
}
