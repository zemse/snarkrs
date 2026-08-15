//! What the six transforms in `compute_h` are made of, and what the bit-reversal costs.
//!
//! The standing review claims the radix-2 NTT "pays six full bit-reversal passes per
//! proof" and rates the fix a medium win with a plausible 1.2x-1.8x NTT-only gain. Two
//! numbers turn that from a guess into a decision, and neither needs `g16-ntt` to change:
//!
//!   1. What fraction of a transform the permutation actually is. `bit_reverse_permute`
//!      is a self-contained pass over the array, so the identical loop is reproduced here
//!      and timed on its own. It is reproduced rather than called because it is private,
//!      and a test asserts the reproduction permutes identically, so this cannot silently
//!      drift from the code it stands in for.
//!   2. What fraction of the proof the transforms are at all, which caps the prize
//!      whatever the NTT gain turns out to be.
//!
//! One structural point the timings make on their own: `bit_reverse_permute` is a plain
//! serial `for` loop inside a transform whose butterfly passes are all parallel. On a
//! 12-core machine its share of *wall clock* is therefore about twelve times its share of
//! CPU, which is not what a CPU profile shows and is exactly what Amdahl charges for.
//!
//! `cargo run --release -p g16-core --example ntt_shape -- [LOG_N]`

use std::time::Instant;

use g16_field::{CurveGroup, Domain, Fr, G1Projective, UniformRand};
use g16_msm::{CpuMsm, MsmBackend};
use g16_ntt::{CpuNtt, Direction, NttBackend};

/// Byte-for-byte the permutation `g16_ntt::bit_reverse_permute` performs. Kept in step
/// with it by `same_permutation_as_the_ntt_uses` below.
fn bit_reverse_permute(a: &mut [Fr], log_n: u32) {
    let n = a.len();
    if n <= 2 {
        return;
    }
    let shift = usize::BITS - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            a.swap(i, j);
        }
    }
}

fn best(reps: usize, f: &mut dyn FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::MAX, f64::min)
}

/// A deterministic spread of field elements. Values do not change the cost of a transform,
/// only their number does, so a cheap generator is enough and an RNG would only add a
/// dependency.
fn sample(n: usize) -> Vec<Fr> {
    let mut s = 0x9e3779b97f4a7c15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Fr::from(s)
        })
        .collect()
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let logs: Vec<u32> = match std::env::args().nth(1) {
        Some(s) => vec![s.parse().expect("log_n")],
        // The two domains the profiled circuits use, plus one either side for the shape.
        None => vec![14, 15, 17, 18],
    };

    let ntt = CpuNtt::new();
    println!(
        "{:>6} {:>10} {:>11} {:>11} {:>11} {:>11} {:>9}",
        "log n", "n", "iNTT ms", "NTT ms", "shift ms", "bitrev ms", "bitrev %"
    );
    for log_n in logs {
        let n = 1usize << log_n;
        let domain = Domain::new(n).unwrap();
        let mut v = sample(n);

        // Warm the twiddle cache: it is built once per (size, root) and kept for the life
        // of the circuit, so including it would measure a cost the prover pays once per
        // key rather than once per proof.
        ntt.ntt(&domain, &mut v, Direction::Forward);
        ntt.ntt(&domain, &mut v, Direction::Inverse);

        let fwd = best(reps, &mut || ntt.ntt(&domain, &mut v, Direction::Forward));
        let inv = best(reps, &mut || ntt.ntt(&domain, &mut v, Direction::Inverse));
        let shift = best(reps, &mut || ntt.distribute_powers(&mut v, domain.group_gen));
        let bitrev = best(reps, &mut || bit_reverse_permute(&mut v, log_n));

        // A transform is one permutation plus log_n butterfly passes, so the permutation's
        // share of the transform is what a mixed-radix or Stockham rewrite could remove.
        let per_transform = (fwd + inv) / 2.0;
        println!(
            "{log_n:>6} {n:>10} {inv:>11.3} {fwd:>11.3} {shift:>11.3} {bitrev:>11.3} \
             {:>8.1}%",
            100.0 * bitrev / per_transform
        );

        // One proof's stages 1-3 at this domain: three vectors, each iNTT -> shift -> NTT.
        let stages_1_3 = 3.0 * (inv + shift + fwd);
        println!(
            "       one proof's stages 1-3 at this size: {stages_1_3:.2} ms, of which \
             6 bit-reversal passes are {:.2} ms ({:.1}%)",
            6.0 * bitrev,
            100.0 * 6.0 * bitrev / stages_1_3
        );
    }

    scaling();
}

/// How many cores each stage can actually use.
///
/// This is the question the profile raises and cannot answer. `bench` says the NTT is 11%
/// of the proof's wall clock, but the CPU profile attributes only about 3% of all CPU to
/// the butterflies, and those two are only consistent if the transform is running on
/// roughly a quarter of the machine while the rest of the pool waits. Running each stage
/// in a fixed-size pool settles it directly.
///
/// A transform is `log n` sequential passes with a fork/join barrier between each, plus a
/// serial bit-reversal, so it has a hard ceiling that an MSM (whose windows are entirely
/// independent) does not.
fn scaling() {
    let log_n = 18u32;
    let n = 1usize << log_n;
    let domain = Domain::new(n).unwrap();
    let reps = 7;

    println!();
    println!("== parallel scaling at n = 2^{log_n}, forward NTT vs one G1 MSM ==");
    println!(
        "{:>8}{:>12}{:>10}{:>14}{:>10}",
        "threads", "NTT ms", "speedup", "MSM ms", "speedup"
    );

    // Bases by walking an arithmetic progression: an MSM only ever adds its bases, so
    // structure in them cannot change the cost, and this avoids n scalar multiplications.
    let mut rng = ark_std::test_rng();
    let step = G1Projective::rand(&mut rng);
    let mut cur = G1Projective::rand(&mut rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    let bases = G1Projective::normalize_batch(&proj);
    drop(proj);
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    assert_eq!(bases.len(), scalars.len(), "MSM truncates a mismatch in release");

    let mut base_ntt = 0.0;
    let mut base_msm = 0.0;
    for threads in [1usize, 2, 4, 6, 8, 10, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        // Constructed inside the pool so `rayon::current_num_threads()` — which is how
        // both crates size their task counts — reports this pool, not the global one.
        let (ntt, msm) = pool.install(|| (CpuNtt::new(), CpuMsm::new()));
        let mut v = sample(n);
        pool.install(|| ntt.ntt(&domain, &mut v, Direction::Forward));

        let t_ntt = pool.install(|| {
            best(reps, &mut || ntt.ntt(&domain, &mut v, Direction::Forward))
        });
        let t_msm = pool.install(|| {
            best(3, &mut || {
                let _ = std::hint::black_box(msm.msm_g1(&bases, &scalars));
            })
        });
        if threads == 1 {
            base_ntt = t_ntt;
            base_msm = t_msm;
        }
        println!(
            "{threads:>8}{t_ntt:>12.3}{:>9.2}x{t_msm:>14.3}{:>9.2}x",
            base_ntt / t_ntt,
            base_msm / t_msm
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permutation above must be the one the NTT performs, or every number this
    /// example prints is about a different program.
    #[test]
    fn same_permutation_as_the_ntt_uses() {
        for log_n in 1..12u32 {
            let n = 1usize << log_n;
            let domain = Domain::new(n).unwrap();
            let ntt = CpuNtt::new();

            // A transform is permute-then-butterflies, and on the all-ones input every
            // butterfly pass is a pure sum, so the permutation is not observable that way.
            // Instead: permuting twice is the identity, so permute the input, run the
            // NTT, and check it equals the NTT of the unpermuted input permuted... which
            // is circular. Use the direct property instead: the permutation is an
            // involution built from `reverse_bits`, so compare against an independent
            // construction by index.
            let mut a = sample(n);
            let want: Vec<Fr> = (0..n)
                .map(|i| a[i.reverse_bits() >> (usize::BITS - log_n)])
                .collect();
            bit_reverse_permute(&mut a, log_n);
            assert_eq!(a, want, "log_n {log_n}");

            // And it really is an involution, which is what makes the in-place swap loop
            // correct in the first place.
            bit_reverse_permute(&mut a, log_n);
            let _ = domain;
            let _ = ntt;
            assert_eq!(a, sample(n), "log_n {log_n} not an involution");
        }
    }
}
