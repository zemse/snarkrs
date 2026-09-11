//! Number-theoretic transforms over `Fr`, and the coset machinery for Jordi's trick.
//!
//! The prover needs `6 * NTT_n` per proof: three iNTTs to take the A/B/C evaluations on
//! the domain into coefficient form, three forward NTTs to re-evaluate them on a disjoint
//! coset. The coset shift between them is a pure elementwise map and is exposed
//! separately so a backend can fuse it into an NTT epilogue.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use g16_field::{Domain, Field, Fr};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Forward,
    Inverse,
}

/// The transform primitives a backend must provide. Deliberately slice-shaped: the CPU
/// backend operates in place on host memory, and the Metal backend implements this trait
/// only for testing kernels in isolation. The *production* GPU path does not go through
/// this trait, because round-tripping every transform across the boundary would defeat
/// the point; it implements [`g16_core::PreparedCircuit::compute_h`] directly and keeps
/// the three vectors device-resident across all six transforms.
pub trait NttBackend: Send + Sync {
    fn name(&self) -> &'static str;
    /// In-place radix-2 NTT. `a.len()` must equal `domain.size`.
    fn ntt(&self, domain: &Domain, a: &mut [Fr], dir: Direction);
    /// In-place `a[i] *= shift^i`.
    fn distribute_powers(&self, a: &mut [Fr], shift: Fr);
}

/// Below this length the serial path wins. A transform is `log n` sequential passes and
/// every one pays a full rayon fork/join, so the overhead scales with `log n` while the
/// work only starts to dominate at `n log n`.
///
/// This was first set to 2^13 from a sweep taken while a `snarkjs powersoftau prepare
/// phase2` was using 900% CPU, which inflated the parallel path's apparent cost by up to
/// 6x. Re-measured on a quiet 12-core M2 Max (forward transform, release): the parallel
/// path is 1.44x faster than serial at 2^12, 1.20x at 2^11, and break-even at 2^10. The
/// old constant left a visible discontinuity in the sweep, where 2^13 completed twice the
/// work of 2^12 in less wall time because only one of them took the parallel path.
///
/// Benchmark numbers taken under load are worse than no numbers, because they look like
/// measurements.
const PARALLEL_THRESHOLD: usize = 1 << 10;

/// Rayon tasks per pass, as a multiple of the thread count. Slight oversubscription lets
/// work stealing even out threads that lose time to cache misses; going much higher just
/// buys fork/join overhead.
const TASKS_PER_THREAD: usize = 4;

/// Never hand a task fewer butterflies than this. This is what keeps the early passes
/// (where a "block" is two elements) from degenerating into one rayon task per butterfly.
const MIN_BUTTERFLIES_PER_TASK: usize = 64;

/// Elements per fused-pass block: all passes whose butterflies stay inside a block of
/// this size run back to back on it while it sits in cache, so `log2(CACHE_BLOCK)` of a
/// transform's passes cost one trip through memory rather than one each. 2^11 is 64 KB
/// of `Fr`, half a performance core's 128 KB L1d. Swept against 2^10 and 2^12 on the
/// csp artifacts (warm median of 25, ntt_ms at 2^16 / 2^17): 11.58/23.09, 11.87/22.92,
/// 11.78/23.40. All three tie within run-to-run noise, so the middle one stays: it fits
/// L1d with room for the twiddle lines and leaves 32 blocks at 2^16 for a 12-thread pool.
const CACHE_BLOCK: usize = 1 << 11;

/// Twiddle tables are keyed by `(size, root)` rather than by size alone because `Domain`
/// exposes its fields publicly, so a hand-built `Domain` could carry a root that is not
/// the one `Domain::new` derives for that size.
type TwiddleCache = RwLock<HashMap<(usize, Fr), Arc<Vec<Fr>>>>;

/// Multi-threaded CPU NTT.
pub struct CpuNtt {
    pub threads: usize,
    /// `Domain::twiddles()` rebuilds the whole table on every call, and the prover runs
    /// six transforms per proof over the same two tables. The cache has to live here:
    /// `NttBackend::ntt` receives a `&Domain`, not a prepared table.
    twiddles: TwiddleCache,
    /// Tables for [`Self::coset_scale_bitrev`], keyed by size, shift and `size_inv`.
    /// The scale is public on `Domain`, so two same-sized domains can differ in it.
    coset_tables: RwLock<HashMap<(usize, Fr, Fr), Arc<Vec<Fr>>>>,
}

impl CpuNtt {
    pub fn new() -> Self {
        Self {
            // The global rayon pool is what `par_chunks_mut` actually runs on, so sizing
            // tasks against any other number would be a lie.
            threads: rayon::current_num_threads().max(1),
            twiddles: RwLock::new(HashMap::new()),
            coset_tables: RwLock::new(HashMap::new()),
        }
    }

    fn tasks(&self) -> usize {
        self.threads.max(1) * TASKS_PER_THREAD
    }

    /// Cached twiddle table for this direction. Returns an `Arc` so the table is neither
    /// copied nor held under the lock while the transform runs.
    fn twiddles(&self, domain: &Domain, dir: Direction) -> Arc<Vec<Fr>> {
        let root = match dir {
            Direction::Forward => domain.group_gen,
            Direction::Inverse => domain.group_gen_inv,
        };
        let key = (domain.size, root);

        // A poisoned lock here would mean a panic inside a HashMap lookup, which cannot
        // leave the map torn, so recovering beats poisoning every later proof.
        if let Some(hit) = self
            .twiddles
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Arc::clone(hit);
        }

        let built = Arc::new(match dir {
            Direction::Forward => domain.twiddles(),
            Direction::Inverse => domain.twiddles_inv(),
        });
        // Built outside the write lock, so two threads can race on a cold cache. Whoever
        // inserts first wins and the loser drops its copy; both tables hold equal values.
        let mut guard = self.twiddles.write().unwrap_or_else(|e| e.into_inner());
        Arc::clone(guard.entry(key).or_insert(built))
    }

    /// The transform, with the serial/parallel choice left to the caller so tests can
    /// drive both paths over one input.
    fn transform(&self, domain: &Domain, a: &mut [Fr], dir: Direction, parallel: bool) {
        check_domain(domain, a.len());
        let n = a.len();
        // Size 1 has an empty twiddle table and `size_inv == 1`, so both directions are
        // the identity. Bailing here also keeps the `usize::BITS - log_size` shift below
        // from being a shift by the full word width.
        if n <= 1 {
            return;
        }

        let twiddles = self.twiddles(domain, dir);
        bit_reverse_permute(a, domain.log_size);
        self.dit_passes(a, &twiddles, parallel);

        if dir == Direction::Inverse {
            // The Domain hands out `1/n` but never applies it; the iNTT owns that.
            let scale = domain.size_inv;
            if parallel {
                let chunk = chunk_len(n, self.tasks());
                a.par_chunks_mut(chunk)
                    .for_each(|part| part.iter_mut().for_each(|x| *x *= scale));
            } else {
                a.iter_mut().for_each(|x| *x *= scale);
            }
        }
    }

    /// Decimation in time: bit-reversed input, natural output, half-size doubling per
    /// pass.
    fn dit_passes(&self, a: &mut [Fr], twiddles: &[Fr], parallel: bool) {
        let n = a.len();
        let mut half = 1usize;
        if parallel && n > CACHE_BLOCK {
            // Every pass with `2 * half <= CACHE_BLOCK` moves data only within one
            // aligned CACHE_BLOCK-sized block, so all of them run back to back per
            // block: one fork/join and one trip through memory instead of one of each
            // per pass. The twiddle indexing is unchanged because a butterfly's table
            // index depends on its offset within its 2*half block, not on where the
            // block sits in the array.
            a.par_chunks_mut(CACHE_BLOCK).for_each(|block| {
                let mut h = 1usize;
                while h < CACHE_BLOCK {
                    serial_pass::<false>(block, h, twiddles, n / (2 * h));
                    h <<= 1;
                }
            });
            self.dit_outer_passes(a, twiddles, CACHE_BLOCK);
            return;
        }
        while half < n {
            // Butterfly `j` of a block needs `root^(j * n / 2half)`, and the table holds
            // `root^i` in natural order, so a pass is a strided read into it.
            let stride = n / (2 * half);
            if parallel {
                parallel_pass::<false>(a, half, twiddles, stride, self.tasks());
            } else {
                serial_pass::<false>(a, half, twiddles, stride);
            }
            half <<= 1;
        }
    }

    /// The DIT passes above the fused block, radix-4: consecutive passes at `half` and
    /// `2 * half` run as one trip through the vector, so what the block fusion did for
    /// the small-half passes this does for the large-stride ones, halving their number.
    /// One radix-2 pass mops up when the remaining count is odd.
    fn dit_outer_passes(&self, a: &mut [Fr], twiddles: &[Fr], mut half: usize) {
        let n = a.len();
        while half * 4 <= n {
            parallel_quad_pass_dit(a, half, twiddles, self.tasks());
            half <<= 2;
        }
        if half < n {
            parallel_pass::<false>(a, half, twiddles, n / (2 * half), self.tasks());
        }
    }

    /// The mirror for DIF: passes at `half` and `half / 2` pair up, walking down until
    /// only the fused-block passes below `floor` remain.
    fn dif_outer_passes(&self, a: &mut [Fr], twiddles: &[Fr], floor: usize) {
        let n = a.len();
        let mut half = n / 2;
        while half > 2 * floor {
            parallel_quad_pass_dif(a, half, twiddles, self.tasks());
            half >>= 2;
        }
        if half > floor {
            parallel_pass::<true>(a, half, twiddles, n / (2 * half), self.tasks());
        }
    }

    /// Decimation in frequency: natural input, bit-reversed output, half-size halving
    /// per pass. Same twiddle tables, same per-pass indexing, opposite pass order, and
    /// the mirror image of the fusion above: here it is the *final* passes that stay
    /// inside one block.
    fn dif_passes(&self, a: &mut [Fr], twiddles: &[Fr], parallel: bool) {
        let n = a.len();
        if parallel && n > CACHE_BLOCK {
            self.dif_outer_passes(a, twiddles, CACHE_BLOCK / 2);
            a.par_chunks_mut(CACHE_BLOCK).for_each(|block| {
                let mut h = CACHE_BLOCK / 2;
                while h >= 1 {
                    serial_pass::<true>(block, h, twiddles, n / (2 * h));
                    h >>= 1;
                }
            });
            return;
        }
        let mut half = n / 2;
        while half >= 1 {
            let stride = n / (2 * half);
            if parallel {
                parallel_pass::<true>(a, half, twiddles, stride, self.tasks());
            } else {
                serial_pass::<true>(a, half, twiddles, stride);
            }
            half >>= 1;
        }
    }

    /// Inverse transform, natural input to *bit-reversed* output, and deliberately
    /// without the `1/n` scale: [`Self::coset_scale_bitrev`] folds it into its table, so
    /// applying it here as well would scale twice.
    ///
    /// The point of the bit-reversed order is [`Self::ntt_from_bitrev`]: a decimation-
    /// in-frequency inverse feeding a decimation-in-time forward cancels both
    /// permutations, which for the prover's iNTT -> coset shift -> NTT chain removes six
    /// serial passes per proof. Measured before the change, the permutation was 12.5% of
    /// a transform's wall clock at 2^16, and it was the only serial loop inside an
    /// otherwise parallel stage.
    pub fn intt_to_bitrev(&self, domain: &Domain, a: &mut [Fr]) {
        check_domain(domain, a.len());
        if a.len() <= 1 {
            return;
        }
        let twiddles = self.twiddles(domain, Direction::Inverse);
        self.dif_passes(a, &twiddles, a.len() >= PARALLEL_THRESHOLD);
    }

    /// Forward transform, *bit-reversed* input to natural output. The other half of the
    /// [`Self::intt_to_bitrev`] pairing.
    pub fn ntt_from_bitrev(&self, domain: &Domain, a: &mut [Fr]) {
        check_domain(domain, a.len());
        if a.len() <= 1 {
            return;
        }
        let twiddles = self.twiddles(domain, Direction::Forward);
        self.dit_passes(a, &twiddles, a.len() >= PARALLEL_THRESHOLD);
    }

    /// `a[p] *= shift^bitrev(p) / n` for a vector in bit-reversed coefficient order: the
    /// coset shift *and* the iNTT's deferred scale in one multiply per element. The
    /// power ladder cannot run in bit-reversed order, so the table is built in natural
    /// order, permuted once, and cached; `shift` is fixed per circuit, so per proof this
    /// stage is three sequential table reads.
    pub fn coset_scale_bitrev(&self, domain: &Domain, a: &mut [Fr], shift: Fr) {
        check_domain(domain, a.len());
        let table = self.coset_table(domain, shift);
        if a.len() < PARALLEL_THRESHOLD {
            for (x, s) in a.iter_mut().zip(table.iter()) {
                *x *= s;
            }
            return;
        }
        let chunk = chunk_len(a.len(), self.tasks());
        a.par_chunks_mut(chunk)
            .zip(table.par_chunks(chunk))
            .for_each(|(part, tab)| {
                for (x, s) in part.iter_mut().zip(tab.iter()) {
                    *x *= s;
                }
            });
    }

    /// Stages 1-3 for one vector: iNTT, coset shift with the deferred `1/n`, forward
    /// NTT, natural order in and out. Equal to the bit to [`Self::intt_to_bitrev`] then
    /// [`Self::coset_scale_bitrev`] then [`Self::ntt_from_bitrev`], and the reason it
    /// exists is what those three would each pay for separately: the final DIF passes,
    /// the table multiply and the initial DIT passes all move data only within one
    /// aligned CACHE_BLOCK, so here the entire middle runs per block in a single
    /// dispatch and the hand-off costs one trip through memory instead of three.
    pub fn intt_coset_ntt(&self, domain: &Domain, a: &mut [Fr], shift: Fr) {
        check_domain(domain, a.len());
        let n = a.len();
        if n <= CACHE_BLOCK || n < PARALLEL_THRESHOLD {
            // Below the block size the three stages are one cache-resident sweep each
            // anyway, so the composition is the fused path.
            self.intt_to_bitrev(domain, a);
            self.coset_scale_bitrev(domain, a, shift);
            self.ntt_from_bitrev(domain, a);
            return;
        }

        let inv = self.twiddles(domain, Direction::Inverse);
        let fwd = self.twiddles(domain, Direction::Forward);
        let table = self.coset_table(domain, shift);

        self.dif_outer_passes(a, &inv, CACHE_BLOCK / 2);
        a.par_chunks_mut(CACHE_BLOCK)
            .zip(table.par_chunks(CACHE_BLOCK))
            .for_each(|(block, tab)| {
                let mut h = CACHE_BLOCK / 2;
                while h >= 1 {
                    serial_pass::<true>(block, h, &inv, n / (2 * h));
                    h >>= 1;
                }
                for (x, s) in block.iter_mut().zip(tab.iter()) {
                    *x *= s;
                }
                let mut h = 1usize;
                while h < CACHE_BLOCK {
                    serial_pass::<false>(block, h, &fwd, n / (2 * h));
                    h <<= 1;
                }
            });
        self.dit_outer_passes(a, &fwd, CACHE_BLOCK);
    }

    /// Cached `[shift^bitrev(p) / n]` table for [`Self::coset_scale_bitrev`], built and
    /// raced exactly like [`Self::twiddles`].
    fn coset_table(&self, domain: &Domain, shift: Fr) -> Arc<Vec<Fr>> {
        let key = (domain.size, shift, domain.size_inv);
        if let Some(hit) = self
            .coset_tables
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Arc::clone(hit);
        }

        let n = domain.size;
        let mut table = vec![Fr::ONE; n];
        let chunk = chunk_len(n, self.tasks());
        table
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, part)| {
                // The same per-chunk ladder re-entry as `distribute`, seeded with `1/n` so
                // the scale costs nothing extra.
                let mut acc = domain.size_inv * shift.pow([(c * chunk) as u64]);
                for x in part.iter_mut() {
                    *x = acc;
                    acc *= shift;
                }
            });
        bit_reverse_permute(&mut table, domain.log_size);

        let built = Arc::new(table);
        let mut guard = self.coset_tables.write().unwrap_or_else(|e| e.into_inner());
        Arc::clone(guard.entry(key).or_insert(built))
    }

    /// `a[i] *= shift^i`, with the path choice exposed for the same reason as
    /// [`Self::transform`].
    fn distribute(&self, a: &mut [Fr], shift: Fr, parallel: bool) {
        if !parallel {
            let mut acc = Fr::ONE;
            for x in a.iter_mut() {
                *x *= acc;
                acc *= shift;
            }
            return;
        }

        let chunk = chunk_len(a.len(), self.tasks());
        a.par_chunks_mut(chunk).enumerate().for_each(|(c, part)| {
            // The running product is a serial dependency chain, so each chunk re-enters
            // it at its own `shift^(chunk_start)`: one square-and-multiply ladder per
            // chunk instead of one per element, and no chain crossing a chunk boundary.
            let mut acc = shift.pow([(c * chunk) as u64]);
            for x in part.iter_mut() {
                *x *= acc;
                acc *= shift;
            }
        });
    }
}

impl Default for CpuNtt {
    fn default() -> Self {
        Self::new()
    }
}

impl NttBackend for CpuNtt {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn ntt(&self, domain: &Domain, a: &mut [Fr], dir: Direction) {
        self.transform(domain, a, dir, a.len() >= PARALLEL_THRESHOLD);
    }
    fn distribute_powers(&self, a: &mut [Fr], shift: Fr) {
        self.distribute(a, shift, a.len() >= PARALLEL_THRESHOLD);
    }
}

/// Public domain fields must still describe the radix-2 layout before any mutation.
fn check_domain(domain: &Domain, len: usize) {
    assert_eq!(
        len, domain.size,
        "ntt input length must equal the domain size"
    );
    assert!(
        domain.size.is_power_of_two(),
        "ntt domain size must be a power of two"
    );
    assert_eq!(
        domain.log_size,
        domain.size.trailing_zeros(),
        "ntt domain log_size must match its size"
    );
}

/// Elements per rayon task, floored so a task always carries real work.
fn chunk_len(n: usize, tasks: usize) -> usize {
    n.div_ceil(tasks.max(1))
        .max(MIN_BUTTERFLIES_PER_TASK)
        .min(n.max(1))
}

/// Moves element `i` to `bit_reverse(i)`, which is what turns the recursive even/odd split
/// of decimation in time into a flat in-place loop.
fn bit_reverse_permute(a: &mut [Fr], log_n: u32) {
    let n = a.len();
    if n <= 2 {
        return;
    }
    let shift = usize::BITS - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        // Swap each pair once, otherwise every element lands back where it started.
        if i < j {
            a.swap(i, j);
        }
    }
}

/// One run of butterflies. `j0` is the index of `lo[0]` within its block, which is nonzero
/// only when a block has been split across tasks. `DIF` selects the decimation: `false`
/// multiplies by the twiddle on the way in (time), `true` on the way out (frequency).
/// A const generic so the branch is compiled away rather than sitting in the hot loop.
#[inline]
fn butterflies<const DIF: bool>(
    lo: &mut [Fr],
    hi: &mut [Fr],
    twiddles: &[Fr],
    stride: usize,
    j0: usize,
) {
    for (k, (x, y)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
        if DIF {
            let t = (*x - *y) * twiddles[(j0 + k) * stride];
            *x += *y;
            *y = t;
        } else {
            let t = *y * twiddles[(j0 + k) * stride];
            *y = *x - t;
            *x += t;
        }
    }
}

/// Two consecutive DIT passes, `half` and `2 * half`, over one run of quads. The quad at
/// `j` is `(x0, x1, x2, x3) = (a[j], a[j+half], a[j+2half], a[j+3half])` within a
/// `4 * half` block; the first pass pairs (x0,x1) and (x2,x3), the second (x0,x2) and
/// (x1,x3). Same four multiplies the two passes would spend, half their loads and
/// stores. `s2` is the second pass's stride `n / (4 * half)`; the first pass's is twice
/// that.
#[inline]
fn quad_dit(strips: [&mut [Fr]; 4], twiddles: &[Fr], s2: usize, half: usize, j0: usize) {
    let [x0, x1, x2, x3] = strips;
    for k in 0..x0.len() {
        let j = j0 + k;
        let w1 = twiddles[2 * j * s2];
        let t0 = x1[k] * w1;
        let a0 = x0[k] + t0;
        let a1 = x0[k] - t0;
        let t1 = x3[k] * w1;
        let a2 = x2[k] + t1;
        let a3 = x2[k] - t1;
        let u0 = a2 * twiddles[j * s2];
        x0[k] = a0 + u0;
        x2[k] = a0 - u0;
        let u1 = a3 * twiddles[(j + half) * s2];
        x1[k] = a1 + u1;
        x3[k] = a1 - u1;
    }
}

/// The DIF mirror: passes `half` then `half / 2` over quads
/// `(y0, y1, y2, y3) = (a[j], a[j+half/2], a[j+half], a[j+3half/2])` within a
/// `2 * half` block. `s1` is the first pass's stride `n / (2 * half)`; the second's is
/// twice that, and `hh` is `half / 2`.
#[inline]
fn quad_dif(strips: [&mut [Fr]; 4], twiddles: &[Fr], s1: usize, hh: usize, j0: usize) {
    let [y0, y1, y2, y3] = strips;
    for k in 0..y0.len() {
        let j = j0 + k;
        let a0 = y0[k] + y2[k];
        let a2 = (y0[k] - y2[k]) * twiddles[j * s1];
        let a1 = y1[k] + y3[k];
        let a3 = (y1[k] - y3[k]) * twiddles[(j + hh) * s1];
        let w2 = twiddles[2 * j * s1];
        y0[k] = a0 + a1;
        y1[k] = (a0 - a1) * w2;
        y2[k] = a2 + a3;
        y3[k] = (a2 - a3) * w2;
    }
}

fn serial_pass<const DIF: bool>(a: &mut [Fr], half: usize, twiddles: &[Fr], stride: usize) {
    for block in a.chunks_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        butterflies::<DIF>(lo, hi, twiddles, stride, 0);
    }
}

/// The passes themselves are sequential (pass `k+1` reads what pass `k` wrote), but every
/// butterfly inside a pass is independent, so the parallelism goes here.
fn parallel_pass<const DIF: bool>(
    a: &mut [Fr],
    half: usize,
    twiddles: &[Fr],
    stride: usize,
    tasks: usize,
) {
    let block_len = 2 * half;
    let blocks = a.len() / block_len;

    if blocks >= tasks {
        // Early passes: many small blocks. Group whole blocks per task so rayon never
        // splits down to a single butterfly.
        let per_task = blocks
            .div_ceil(tasks)
            .max(MIN_BUTTERFLIES_PER_TASK.div_ceil(half));
        a.par_chunks_mut(per_task * block_len).for_each(|group| {
            for block in group.chunks_mut(block_len) {
                let (lo, hi) = block.split_at_mut(half);
                butterflies::<DIF>(lo, hi, twiddles, stride, 0);
            }
        });
    } else {
        // Late passes: few blocks, and the last one covers the whole array. The only
        // parallelism left is inside a block, so split the butterfly range itself.
        let chunk = half
            .div_ceil(tasks.div_ceil(blocks))
            .max(MIN_BUTTERFLIES_PER_TASK)
            .min(half);
        a.par_chunks_mut(block_len).for_each(|block| {
            let (lo, hi) = block.split_at_mut(half);
            lo.par_chunks_mut(chunk)
                .zip(hi.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(c, (l, h))| butterflies::<DIF>(l, h, twiddles, stride, c * chunk));
        });
    }
}

/// Splits one `4 * half`-sized quad block into its four strips.
fn quad_strips(block: &mut [Fr], half: usize) -> [&mut [Fr]; 4] {
    let (lo, hi) = block.split_at_mut(2 * half);
    let (x0, x1) = lo.split_at_mut(half);
    let (x2, x3) = hi.split_at_mut(half);
    [x0, x1, x2, x3]
}

/// [`parallel_pass`] for a fused DIT pass pair. Only the outer passes come here, so a
/// block is at least `4 * CACHE_BLOCK` elements and needs no grouping; the split branch
/// mirrors the radix-2 one for the late passes where blocks are fewer than tasks.
fn parallel_quad_pass_dit(a: &mut [Fr], half: usize, twiddles: &[Fr], tasks: usize) {
    let n = a.len();
    let s2 = n / (4 * half);
    let block_len = 4 * half;
    let blocks = n / block_len;
    if blocks >= tasks {
        a.par_chunks_mut(block_len).for_each(|block| {
            quad_dit(quad_strips(block, half), twiddles, s2, half, 0);
        });
    } else {
        let chunk = half
            .div_ceil(tasks.div_ceil(blocks))
            .max(MIN_BUTTERFLIES_PER_TASK)
            .min(half);
        a.par_chunks_mut(block_len).for_each(|block| {
            let [x0, x1, x2, x3] = quad_strips(block, half);
            x0.par_chunks_mut(chunk)
                .zip(x1.par_chunks_mut(chunk))
                .zip(x2.par_chunks_mut(chunk))
                .zip(x3.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(c, (((p0, p1), p2), p3))| {
                    quad_dit([p0, p1, p2, p3], twiddles, s2, half, c * chunk)
                });
        });
    }
}

/// The DIF twin: blocks are `2 * half` and the strips are `half / 2` long.
fn parallel_quad_pass_dif(a: &mut [Fr], half: usize, twiddles: &[Fr], tasks: usize) {
    let n = a.len();
    let s1 = n / (2 * half);
    let hh = half / 2;
    let block_len = 2 * half;
    let blocks = n / block_len;
    if blocks >= tasks {
        a.par_chunks_mut(block_len).for_each(|block| {
            quad_dif(quad_strips(block, hh), twiddles, s1, hh, 0);
        });
    } else {
        let chunk = hh
            .div_ceil(tasks.div_ceil(blocks))
            .max(MIN_BUTTERFLIES_PER_TASK)
            .min(hh);
        a.par_chunks_mut(block_len).for_each(|block| {
            let [y0, y1, y2, y3] = quad_strips(block, hh);
            y0.par_chunks_mut(chunk)
                .zip(y1.par_chunks_mut(chunk))
                .zip(y2.par_chunks_mut(chunk))
                .zip(y3.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(c, (((p0, p1), p2), p3))| {
                    quad_dif([p0, p1, p2, p3], twiddles, s1, hh, c * chunk)
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Neither is re-exported through the names the library half of this file uses:
    // `ZERO` lives on AdditiveGroup and `GENERATOR` on FftField.
    use ark_ff::AdditiveGroup;
    use g16_field::FftField;

    /// Deterministic pseudo-random field elements. A xorshift keeps the tests
    /// reproducible without pulling in an RNG crate just for this.
    fn sample(n: usize, seed: u64) -> Vec<Fr> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                Fr::from(s)
            })
            .collect()
    }

    fn domain(n: usize) -> Domain {
        Domain::new(n).unwrap()
    }

    /// O(n^2) reference: `X[k] = sum_j x[j] * root^(j*k)`.
    fn naive_dft(x: &[Fr], root: Fr) -> Vec<Fr> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Fr::ZERO;
                for (j, &xj) in x.iter().enumerate() {
                    acc += xj * root.pow([((j * k) % n) as u64]);
                }
                acc
            })
            .collect()
    }

    fn naive_cyclic_convolution(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        let n = a.len();
        let mut out = vec![Fr::ZERO; n];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[(i + j) % n] += ai * bj;
            }
        }
        out
    }

    #[test]
    fn round_trips_for_every_size_up_to_2_12() {
        let ntt = CpuNtt::new();
        for log in 1..=12u32 {
            let n = 1usize << log;
            let d = domain(n);
            let original = sample(n, 0xC0FFEE + log as u64);
            let mut a = original.clone();
            ntt.ntt(&d, &mut a, Direction::Forward);
            assert_ne!(a, original, "forward transform was a no-op at n = {n}");
            ntt.ntt(&d, &mut a, Direction::Inverse);
            assert_eq!(a, original, "round trip failed at n = {n}");
        }
    }

    #[test]
    fn round_trips_through_the_parallel_path() {
        // The loop above stays under PARALLEL_THRESHOLD, so these are the sizes that
        // exercise `ntt` as the prover will actually hit it.
        let ntt = CpuNtt::new();
        for log in [13u32, 14] {
            let n = 1usize << log;
            let d = domain(n);
            let original = sample(n, 0xBEEF + log as u64);
            let mut a = original.clone();
            ntt.ntt(&d, &mut a, Direction::Forward);
            ntt.ntt(&d, &mut a, Direction::Inverse);
            assert_eq!(a, original, "round trip failed at n = {n}");
        }
    }

    #[test]
    fn size_one_domain_is_the_identity() {
        let ntt = CpuNtt::new();
        let d = domain(1);
        let original = sample(1, 7);
        for dir in [Direction::Forward, Direction::Inverse] {
            let mut a = original.clone();
            ntt.ntt(&d, &mut a, dir);
            assert_eq!(a, original, "{dir:?} on a single point should do nothing");
        }
    }

    #[test]
    fn matches_the_naive_dft() {
        let ntt = CpuNtt::new();
        for log in [0u32, 1, 2, 3, 4] {
            let n = 1usize << log;
            let d = domain(n);
            let x = sample(n, 0xABCD + log as u64);

            let mut fwd = x.clone();
            ntt.ntt(&d, &mut fwd, Direction::Forward);
            assert_eq!(fwd, naive_dft(&x, d.group_gen), "forward, n = {n}");

            // The iNTT is the same sum over the inverse root, scaled by 1/n.
            let mut inv = x.clone();
            ntt.ntt(&d, &mut inv, Direction::Inverse);
            let want: Vec<Fr> = naive_dft(&x, d.group_gen_inv)
                .into_iter()
                .map(|v| v * d.size_inv)
                .collect();
            assert_eq!(inv, want, "inverse, n = {n}");
        }
    }

    #[test]
    fn pointwise_product_is_cyclic_convolution() {
        let ntt = CpuNtt::new();
        for log in [3u32, 6, 10] {
            let n = 1usize << log;
            let d = domain(n);
            let a = sample(n, 11 + log as u64);
            let b = sample(n, 99 + log as u64);

            let mut fa = a.clone();
            let mut fb = b.clone();
            ntt.ntt(&d, &mut fa, Direction::Forward);
            ntt.ntt(&d, &mut fb, Direction::Forward);
            let mut prod: Vec<Fr> = fa.iter().zip(fb.iter()).map(|(x, y)| *x * y).collect();
            ntt.ntt(&d, &mut prod, Direction::Inverse);

            assert_eq!(prod, naive_cyclic_convolution(&a, &b), "n = {n}");
        }
    }

    #[test]
    fn distribute_powers_matches_a_sequential_loop() {
        let ntt = CpuNtt::new();
        let shift = Fr::from(5u64);
        // Sizes below and above PARALLEL_THRESHOLD, so both paths of the public entry
        // point are covered.
        for n in [1usize, 2, 64, 4096, 16384] {
            let original = sample(n, 0x5EED + n as u64);

            let mut want = original.clone();
            let mut acc = Fr::ONE;
            for x in want.iter_mut() {
                *x *= acc;
                acc *= shift;
            }

            let mut got = original.clone();
            ntt.distribute_powers(&mut got, shift);
            assert_eq!(got, want, "n = {n}");
        }
    }

    #[test]
    fn distribute_powers_agrees_across_paths() {
        let ntt = CpuNtt::new();
        // The real caller shifts by the coset generator, so use that one here.
        let shift = Fr::GENERATOR;
        for n in [64usize, 1024, 4096] {
            let original = sample(n, 0xD15EA5E + n as u64);
            let mut serial = original.clone();
            let mut parallel = original.clone();
            ntt.distribute(&mut serial, shift, false);
            ntt.distribute(&mut parallel, shift, true);
            assert_eq!(serial, parallel, "n = {n}");
        }
    }

    #[test]
    fn serial_and_parallel_transforms_agree() {
        let ntt = CpuNtt::new();
        // Spans both branches of `parallel_pass`: late passes with fewer blocks than
        // tasks, and early passes with far more.
        for log in [4u32, 8, 11, 13] {
            let n = 1usize << log;
            let d = domain(n);
            let x = sample(n, 0xFACE + log as u64);
            for dir in [Direction::Forward, Direction::Inverse] {
                let mut serial = x.clone();
                let mut parallel = x.clone();
                ntt.transform(&d, &mut serial, dir, false);
                ntt.transform(&d, &mut parallel, dir, true);
                assert_eq!(serial, parallel, "{dir:?} disagreed at n = {n}");
            }
        }
    }

    #[test]
    fn bitrev_pipeline_matches_the_natural_order_stages() {
        // The prover's stages 1-3 for one vector, both ways. Field arithmetic is exact,
        // so the two must agree to the bit, not approximately. Sizes straddle
        // PARALLEL_THRESHOLD so both path choices inside each method are covered.
        let ntt = CpuNtt::new();
        for log in 0..=20u32 {
            let n = 1usize << log;
            let d = domain(n);
            // The shift the real caller uses: the 2n-th root whose square is the
            // domain's own generator.
            let shift = domain(2 * n).group_gen;
            let x = sample(n, 0xC05E7 + log as u64);

            let mut want = x.clone();
            // The serial radix-2 path shares neither the fused blocks nor the outer
            // radix-4 kernels. Comparing two fused paths could hide a paired error.
            ntt.transform(&d, &mut want, Direction::Inverse, false);
            ntt.distribute(&mut want, shift, false);
            ntt.transform(&d, &mut want, Direction::Forward, false);

            let mut got = x.clone();
            ntt.intt_to_bitrev(&d, &mut got);
            ntt.coset_scale_bitrev(&d, &mut got, shift);
            ntt.ntt_from_bitrev(&d, &mut got);

            assert_eq!(got, want, "n = {n}");

            let mut fused = x;
            ntt.intt_coset_ntt(&d, &mut fused, shift);
            assert_eq!(fused, want, "fused, n = {n}");
        }
    }

    #[test]
    fn the_fused_pipeline_matches_its_three_stages() {
        // Sizes on both sides of CACHE_BLOCK, so the fall-through composition and the
        // fused middle are both exercised, plus the fused path at more than one depth
        // of outer passes.
        let ntt = CpuNtt::new();
        for log in [4u32, 10, 11, 12, 14, 15] {
            let n = 1usize << log;
            let d = domain(n);
            let shift = domain(2 * n).group_gen;
            let x = sample(n, 0xF05E + log as u64);

            let mut want = x.clone();
            ntt.intt_to_bitrev(&d, &mut want);
            ntt.coset_scale_bitrev(&d, &mut want, shift);
            ntt.ntt_from_bitrev(&d, &mut want);

            let mut got = x.clone();
            ntt.intt_coset_ntt(&d, &mut got, shift);
            assert_eq!(got, want, "n = {n}");
        }
    }

    #[test]
    fn intt_to_bitrev_is_the_unscaled_intt_permuted() {
        // Pins each half of the pairing on its own, so a failure in the pipeline test
        // above points at one method rather than at their composition.
        let ntt = CpuNtt::new();
        for log in [0u32, 1, 3, 8, 11, 12, 13, 16, 17] {
            let n = 1usize << log;
            let d = domain(n);
            let x = sample(n, 0xB17 + log as u64);

            let mut want = x.clone();
            ntt.ntt(&d, &mut want, Direction::Inverse);
            let n_as_fr = Fr::from(n as u64);
            want.iter_mut().for_each(|v| *v *= n_as_fr);
            bit_reverse_permute(&mut want, d.log_size);

            let mut got = x.clone();
            ntt.intt_to_bitrev(&d, &mut got);
            assert_eq!(got, want, "n = {n}");

            // And feeding it forward restores the scaled input, since the pair is a
            // round trip up to the deferred 1/n.
            ntt.ntt_from_bitrev(&d, &mut got);
            let want_scaled: Vec<Fr> = x.iter().map(|v| *v * n_as_fr).collect();
            assert_eq!(got, want_scaled, "round trip, n = {n}");
        }
    }

    #[test]
    fn twiddle_cache_returns_the_same_table() {
        let ntt = CpuNtt::new();
        let d = domain(256);
        let first = ntt.twiddles(&d, Direction::Forward);
        let second = ntt.twiddles(&d, Direction::Forward);
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache missed on a warm domain"
        );
        assert_eq!(*first, d.twiddles());
        assert_eq!(*ntt.twiddles(&d, Direction::Inverse), d.twiddles_inv());
    }

    #[test]
    fn concurrent_transforms_share_one_cache() {
        // `PreparedCircuit` is Send + Sync and `prove` may run concurrently, so the cache
        // has to survive several threads hitting a cold domain at once.
        let ntt = CpuNtt::new();
        let d = domain(512);
        let x = sample(d.size, 0x1234);
        let mut want = x.clone();
        ntt.transform(&d, &mut want, Direction::Forward, false);

        let results: Vec<Vec<Fr>> = (0..8)
            .into_par_iter()
            .map(|_| {
                let mut a = x.clone();
                ntt.ntt(&d, &mut a, Direction::Forward);
                a
            })
            .collect();
        for (i, got) in results.iter().enumerate() {
            assert_eq!(*got, want, "thread {i}");
        }
    }

    #[test]
    fn coset_cache_tracks_the_domain_and_shift() {
        let ntt = CpuNtt::new();
        for n in [1, 2, 4096, 8192, 4096] {
            let mut d = domain(n);
            for shift in [Fr::ZERO, Fr::ONE, Fr::from(7u64), domain(2 * n).group_gen] {
                for scale in [d.size_inv, Fr::from(3u64), d.size_inv] {
                    d.size_inv = scale;
                    let mut got = vec![Fr::ONE; n];
                    ntt.coset_scale_bitrev(&d, &mut got, shift);
                    let mut want: Vec<_> = (0..n).map(|i| scale * shift.pow([i as u64])).collect();
                    bit_reverse_permute(&mut want, d.log_size);
                    assert_eq!(got, want, "n = {n}, shift = {shift}, scale = {scale}");
                }
            }
        }
    }

    #[test]
    fn changed_roots_and_cold_concurrent_cosets_match_serial() {
        let ntt = CpuNtt::new();
        let mut d = domain(4096);
        for _ in 0..3 {
            std::mem::swap(&mut d.group_gen, &mut d.group_gen_inv);
            let shift = d.group_gen + Fr::ONE;
            let x = sample(d.size, 42);
            let mut want = x.clone();
            ntt.transform(&d, &mut want, Direction::Inverse, false);
            ntt.distribute(&mut want, shift, false);
            ntt.transform(&d, &mut want, Direction::Forward, false);
            (0..8).into_par_iter().for_each(|_| {
                let mut got = x.clone();
                ntt.intt_coset_ntt(&d, &mut got, shift);
                assert_eq!(got, want);
            });
        }
    }

    #[test]
    fn malformed_domains_are_rejected_before_mutation() {
        type Entry = fn(&CpuNtt, &Domain, &mut [Fr]);
        let entries: [Entry; 6] = [
            |ntt, d, a| ntt.ntt(d, a, Direction::Forward),
            |ntt, d, a| ntt.ntt(d, a, Direction::Inverse),
            CpuNtt::intt_to_bitrev,
            CpuNtt::ntt_from_bitrev,
            |ntt, d, a| ntt.coset_scale_bitrev(d, a, Fr::ONE),
            |ntt, d, a| ntt.intt_coset_ntt(d, a, Fr::ONE),
        ];
        let ntt = CpuNtt::new();
        // 3072 would leave a partial fused block; 1024 and 2048 are handled without
        // block fusion. CACHE_BLOCK divides every valid larger radix-2 domain exactly.
        assert!(CACHE_BLOCK.is_power_of_two());
        for (size, log_size, len) in [(0, 0, 0), (3, 2, 3), (3072, 12, 3072), (8, 2, 8), (8, 3, 7)]
        {
            let mut d = domain(8);
            d.size = size;
            d.log_size = log_size;
            for entry in entries {
                let original = sample(len, 17);
                let mut a = original.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    entry(&ntt, &d, &mut a);
                }));
                assert!(
                    result.is_err(),
                    "accepted size {size}, log {log_size}, len {len}"
                );
                assert_eq!(a, original, "mutated invalid input");
            }
        }
    }

    #[test]
    #[should_panic(expected = "domain size")]
    fn rejects_a_length_mismatch() {
        let ntt = CpuNtt::new();
        let d = domain(8);
        let mut a = sample(4, 1);
        ntt.ntt(&d, &mut a, Direction::Forward);
    }
}
