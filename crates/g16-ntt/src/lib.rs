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

/// Below this length the serial path wins. A transform is `log n` sequential passes, and
/// every one of them pays a full rayon fork/join, so the overhead scales with `log n`
/// while the work only starts to dominate at `n log n`. Measured on a 12-core M-series
/// (forward transform, release): 2^12 ran at 0.76x of serial, 2^14 at 1.98x, 2^16 at
/// 3.4x, so the crossover sits near 2^13 rather than the 2^10 that the butterfly count
/// alone suggests. The measurement was taken on a busy machine, so if anything the true
/// crossover is lower; erring high only costs a fraction of a millisecond on domains this
/// small, and real circuits are 2^16 and up.
const PARALLEL_THRESHOLD: usize = 1 << 13;

/// Rayon tasks per pass, as a multiple of the thread count. Slight oversubscription lets
/// work stealing even out threads that lose time to cache misses; going much higher just
/// buys fork/join overhead.
const TASKS_PER_THREAD: usize = 4;

/// Never hand a task fewer butterflies than this. This is what keeps the early passes
/// (where a "block" is two elements) from degenerating into one rayon task per butterfly.
const MIN_BUTTERFLIES_PER_TASK: usize = 64;

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
}

impl CpuNtt {
    pub fn new() -> Self {
        Self {
            // The global rayon pool is what `par_chunks_mut` actually runs on, so sizing
            // tasks against any other number would be a lie.
            threads: rayon::current_num_threads().max(1),
            twiddles: RwLock::new(HashMap::new()),
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
        assert_eq!(
            a.len(),
            domain.size,
            "ntt input length must equal the domain size"
        );
        let n = a.len();
        // Size 1 has an empty twiddle table and `size_inv == 1`, so both directions are
        // the identity. Bailing here also keeps the `usize::BITS - log_size` shift below
        // from being a shift by the full word width.
        if n <= 1 {
            return;
        }

        let twiddles = self.twiddles(domain, dir);
        bit_reverse_permute(a, domain.log_size);

        // Decimation in time: permuted input, natural output, half-size doubling per pass.
        let mut half = 1usize;
        while half < n {
            // Butterfly `j` of a block needs `root^(j * n / 2half)`, and the table holds
            // `root^i` in natural order, so a pass is a strided read into it.
            let stride = n / (2 * half);
            if parallel {
                parallel_pass(a, half, &twiddles, stride, self.tasks());
            } else {
                serial_pass(a, half, &twiddles, stride);
            }
            half <<= 1;
        }

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
/// only when a block has been split across tasks.
#[inline]
fn butterflies(lo: &mut [Fr], hi: &mut [Fr], twiddles: &[Fr], stride: usize, j0: usize) {
    for (k, (x, y)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
        let t = *y * twiddles[(j0 + k) * stride];
        *y = *x - t;
        *x += t;
    }
}

fn serial_pass(a: &mut [Fr], half: usize, twiddles: &[Fr], stride: usize) {
    for block in a.chunks_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        butterflies(lo, hi, twiddles, stride, 0);
    }
}

/// The passes themselves are sequential (pass `k+1` reads what pass `k` wrote), but every
/// butterfly inside a pass is independent, so the parallelism goes here.
fn parallel_pass(a: &mut [Fr], half: usize, twiddles: &[Fr], stride: usize, tasks: usize) {
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
                butterflies(lo, hi, twiddles, stride, 0);
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
                .for_each(|(c, (l, h))| butterflies(l, h, twiddles, stride, c * chunk));
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
        for log in [3u32, 4] {
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
    #[should_panic(expected = "domain size")]
    fn rejects_a_length_mismatch() {
        let ntt = CpuNtt::new();
        let d = domain(8);
        let mut a = sample(4, 1);
        ntt.ntt(&d, &mut a, Direction::Forward);
    }
}
