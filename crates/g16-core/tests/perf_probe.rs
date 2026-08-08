//! Performance probe. Not part of the test contract: every test here is `#[ignore]`d and
//! exists to produce numbers, not to assert behaviour.
//!
//! Run with:
//!   cargo test -p g16-core --release --test perf_probe -- --ignored --nocapture
//!   RAYON_NUM_THREADS=1 cargo test -p g16-core --release --test perf_probe -- --ignored --nocapture
//!
//! Everything reports best-of-N rather than a mean or median. On a contended machine the
//! mean measures the other tenants; the minimum is the closest thing to the uncontended
//! cost that a shared box will give up.

use std::time::Instant;

use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_std::{rand::Rng, test_rng, UniformRand};
use g16_field::*;
use g16_msm::{window_size, CpuMsm, MsmBackend};
use g16_ntt::{CpuNtt, Direction, NttBackend};

/// Best of `reps` runs, in milliseconds.
fn best_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// `n` bases without paying `n` scalar multiplications: an arithmetic progression of
/// points. An MSM only ever adds its bases, so structure in them cannot help or hurt.
fn walk_points<P: ark_ec::short_weierstrass::SWCurveConfig>(
    n: usize,
    rng: &mut impl Rng,
) -> Vec<ark_ec::short_weierstrass::Affine<P>>
where
    ark_ec::short_weierstrass::Projective<P>: UniformRand,
{
    let step = ark_ec::short_weierstrass::Projective::<P>::rand(rng);
    let mut cur = ark_ec::short_weierstrass::Projective::<P>::rand(rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
        proj.push(cur);
        cur += &step;
    }
    ark_ec::short_weierstrass::Projective::normalize_batch(&proj)
}

fn rand_scalars(n: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..n).map(|_| Fr::rand(rng)).collect()
}

/// The scalar mix the four witness MSMs actually see: mostly 0 and 1.
fn witness_scalars(n: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..n)
        .map(|i| match i % 8 {
            0..=4 => Fr::zero(),
            5 | 6 => Fr::one(),
            _ => Fr::rand(rng),
        })
        .collect()
}

fn threads() -> usize {
    rayon::current_num_threads()
}

/// Our MSM against arkworks at the two sizes the task names, plus the neighbours that
/// show where the window-size choice changes.
#[test]
#[ignore = "probe"]
fn msm_vs_arkworks() {
    let mut rng = test_rng();
    let msm = CpuMsm::new();
    println!(
        "\n== msm_g1, dense random scalars, rayon threads = {} ==",
        threads()
    );
    println!(
        "{:>5} {:>4} {:>10} {:>12} {:>10} {:>10}",
        "log n", "c", "ours ms", "ns/point", "ark ms", "ark/ours"
    );
    for log in [14u32, 15, 16, 17, 18] {
        let n = 1usize << log;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let reps = if log >= 18 { 3 } else { 5 };

        let mut ours_out = G1Projective::zero();
        let ours = best_ms(reps, || ours_out = msm.msm_g1(&bases, &scalars));
        let mut ark_out = G1Projective::zero();
        let ark = best_ms(reps, || {
            ark_out = G1Projective::msm(&bases, &scalars).unwrap()
        });
        assert_eq!(
            ours_out, ark_out,
            "our MSM disagrees with arkworks at 2^{log}"
        );

        println!(
            "{:>5} {:>4} {:>10.2} {:>12.1} {:>10.2} {:>10.2}",
            log,
            window_size(n),
            ours,
            ours * 1e6 / n as f64,
            ark,
            ark / ours
        );
    }
}

/// The scalar mix matters more than the length: four of the five Groth16 MSMs are fed the
/// witness, which is overwhelmingly 0 and 1.
#[test]
#[ignore = "probe"]
fn msm_scalar_mix() {
    let mut rng = test_rng();
    let msm = CpuMsm::new();
    println!(
        "\n== msm_g1 by scalar mix, rayon threads = {} ==",
        threads()
    );
    for log in [16u32, 18] {
        let n = 1usize << log;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let dense = rand_scalars(n, &mut rng);
        let sparse = witness_scalars(n, &mut rng);
        let reps = if log >= 18 { 3 } else { 5 };
        let d = best_ms(reps, || {
            let _ = std::hint::black_box(msm.msm_g1(&bases, &dense));
        });
        let s = best_ms(reps, || {
            let _ = std::hint::black_box(msm.msm_g1(&bases, &sparse));
        });
        println!("2^{log}  dense {d:8.2} ms   witness-like (62% zero, 25% one) {s:8.2} ms");
    }
}

/// Six transforms per proof, so this is the number that matters, not one.
#[test]
#[ignore = "probe"]
fn ntt_sweep() {
    let mut rng = test_rng();
    let ntt = CpuNtt::new();
    println!("\n== forward NTT, rayon threads = {} ==", threads());
    println!(
        "{:>5} {:>10} {:>12} {:>14}",
        "log n", "ms", "ns/point", "6x (per proof)"
    );
    for log in [12u32, 13, 14, 16, 18, 20] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let reps = if log >= 20 { 3 } else { 5 };
        let ms = best_ms(reps, || ntt.ntt(&domain, &mut a, Direction::Forward));
        println!(
            "{:>5} {:>10.3} {:>12.2} {:>14.2}",
            log,
            ms,
            ms * 1e6 / n as f64,
            ms * 6.0
        );
    }
}

/// `distribute_powers` is stage 2, run three times per proof.
#[test]
#[ignore = "probe"]
fn distribute_powers_sweep() {
    let mut rng = test_rng();
    let ntt = CpuNtt::new();
    let shift = Fr::from(7u64);
    println!("\n== distribute_powers, rayon threads = {} ==", threads());
    for log in [12u32, 14, 16, 18, 20] {
        let n = 1usize << log;
        let mut a = rand_scalars(n, &mut rng);
        let ms = best_ms(5, || ntt.distribute_powers(&mut a, shift));
        println!(
            "2^{log:<2}  {ms:8.3} ms  ({:.2} ns/point)",
            ms * 1e6 / n as f64
        );
    }
}

/// What a 2^20 proof would cost, composed from stages measured at 2^20 rather than
/// extrapolated. The H MSM is dense; the four witness MSMs are witness-like.
///
/// Every stage is timed once per round and the rounds are interleaved, so a load spike
/// lands on all five stages instead of inflating whichever one happened to be running.
/// Each stage reports its own minimum across rounds.
#[test]
#[ignore = "probe"]
fn composed_stage_split_at_2_20() {
    let mut rng = test_rng();
    let msm = CpuMsm::new();
    let ntt = CpuNtt::new();
    let log = 20u32;
    let n = 1usize << log;

    let domain = Domain::new(n).unwrap();
    let mut v = rand_scalars(n, &mut rng);
    let shift = Fr::from(7u64);
    let bases: Vec<G1Affine> = walk_points(n, &mut rng);
    let dense = rand_scalars(n, &mut rng);
    let sparse = witness_scalars(n, &mut rng);
    let g2: Vec<G2Affine> = walk_points(n, &mut rng);

    // Warm up: first-touch page faults on ~250 MB and the 16 MB twiddle table must not
    // land inside a timed region.
    ntt.ntt(&domain, &mut v, Direction::Forward);
    ntt.distribute_powers(&mut v, shift);
    let _ = std::hint::black_box(msm.msm_g1(&bases, &dense));
    let _ = std::hint::black_box(msm.msm_g1(&bases, &sparse));
    let _ = std::hint::black_box(msm.msm_g2(&g2, &sparse));

    let (mut ntt_one, mut dist_one) = (f64::MAX, f64::MAX);
    let (mut h_msm, mut w_msm_g1, mut w_msm_g2) = (f64::MAX, f64::MAX, f64::MAX);
    let lap = |best: &mut f64, t: Instant| {
        *best = best.min(t.elapsed().as_secs_f64() * 1e3);
    };
    for _ in 0..5 {
        let t = Instant::now();
        ntt.ntt(&domain, &mut v, Direction::Forward);
        lap(&mut ntt_one, t);
        let t = Instant::now();
        ntt.distribute_powers(&mut v, shift);
        lap(&mut dist_one, t);
        let t = Instant::now();
        let _ = std::hint::black_box(msm.msm_g1(&bases, &dense));
        lap(&mut h_msm, t);
        let t = Instant::now();
        let _ = std::hint::black_box(msm.msm_g1(&bases, &sparse));
        lap(&mut w_msm_g1, t);
        let t = Instant::now();
        let _ = std::hint::black_box(msm.msm_g2(&g2, &sparse));
        lap(&mut w_msm_g2, t);
    }

    // Six transforms, three coset shifts.
    let ntt_total = ntt_one * 6.0 + dist_one * 3.0;
    // Stages 5-9: A(G1), B(G1), L(G1) witness-like, B(G2) witness-like, H(G1) dense.
    let msm_total = w_msm_g1 * 3.0 + w_msm_g2 + h_msm;
    let total = ntt_total + msm_total;

    println!("\n== composed 2^20 proof, rayon threads = {} ==", threads());
    println!("  one forward NTT      {ntt_one:9.2} ms");
    println!("  one distribute       {dist_one:9.2} ms");
    println!("  H MSM (dense G1)     {h_msm:9.2} ms");
    println!("  witness MSM G1       {w_msm_g1:9.2} ms");
    println!("  witness MSM G2       {w_msm_g2:9.2} ms");
    println!("  ---");
    println!(
        "  NTT+shift total      {ntt_total:9.2} ms   {:5.1}%",
        100.0 * ntt_total / total
    );
    println!(
        "  MSM total            {msm_total:9.2} ms   {:5.1}%",
        100.0 * msm_total / total
    );
    println!("  (gather and pointwise excluded: they need a real key)");
}

/// Exact replica of `g16_ntt`'s private `bit_reverse_permute`, so its cost can be
/// separated from the butterfly passes. It is serial in the library, and at 2^20 it is a
/// random-access pass over 32 MB of `Fr`.
fn bit_reverse_permute_replica(a: &mut [Fr], log_n: u32) {
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

/// How much of one NTT is the serial bit-reversal? This is the Amdahl ceiling on the
/// whole transform.
#[test]
#[ignore = "probe"]
fn bit_reversal_share_of_the_ntt() {
    let mut rng = test_rng();
    let ntt = CpuNtt::new();
    println!(
        "\n== bit-reversal vs whole NTT, rayon threads = {} ==",
        threads()
    );
    println!(
        "{:>5} {:>12} {:>12} {:>10}",
        "log n", "bitrev ms", "full ntt ms", "bitrev %"
    );
    for log in [16u32, 18, 20] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let reps = if log >= 20 { 3 } else { 5 };
        let br = best_ms(reps, || bit_reverse_permute_replica(&mut a, log));
        let full = best_ms(reps, || ntt.ntt(&domain, &mut a, Direction::Forward));
        println!(
            "{:>5} {:>12.2} {:>12.2} {:>9.1}%",
            log,
            br,
            full,
            100.0 * br / full
        );
    }
}

/// Thread scaling, measured by interleaving the 1-thread and N-thread runs inside one
/// process rather than comparing two separate runs. On a contended box the interleaved
/// ratio is the only trustworthy number: both arms see the same neighbours.
///
/// The 1-thread arm builds its `CpuNtt`/`CpuMsm` *inside* `pool.install`, because both
/// cache `rayon::current_num_threads()` at construction and size their task counts from
/// it.
#[test]
#[ignore = "probe"]
fn thread_scaling_interleaved() {
    let mut rng = test_rng();
    let nthreads = rayon::current_num_threads();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    println!("\n== interleaved thread scaling, global pool = {nthreads} threads ==");
    println!(
        "{:<28} {:>10} {:>12} {:>10} {:>12}",
        "stage", "1 thr ms", "N thr ms", "speedup", "efficiency"
    );

    let row = |name: &str, one_ms: f64, many_ms: f64| {
        let sp = one_ms / many_ms;
        println!(
            "{:<28} {:>10.2} {:>12.2} {:>9.2}x {:>11.0}%",
            name,
            one_ms,
            many_ms,
            sp,
            100.0 * sp / nthreads as f64
        );
    };

    for log in [16u32, 18, 20] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let reps = if log >= 20 { 3 } else { 5 };

        // Interleave: 1 thread, N threads, 1 thread, ... so drift in machine load hits
        // both arms equally.
        let mut one_best = f64::MAX;
        let mut many_best = f64::MAX;
        let ntt_many = CpuNtt::new();
        // Built inside the pool so it caches threads = 1, and built ONCE so its twiddle
        // cache is warm on every rep, exactly like the N-thread arm's.
        let ntt1 = one.install(CpuNtt::new);
        // Warm both twiddle caches before timing anything.
        one.install(|| ntt1.ntt(&domain, &mut a, Direction::Forward));
        ntt_many.ntt(&domain, &mut a, Direction::Forward);
        for _ in 0..reps {
            one.install(|| {
                let t = Instant::now();
                ntt1.ntt(&domain, &mut a, Direction::Forward);
                one_best = one_best.min(t.elapsed().as_secs_f64() * 1e3);
            });
            let t = Instant::now();
            ntt_many.ntt(&domain, &mut a, Direction::Forward);
            many_best = many_best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        row(&format!("ntt forward 2^{log}"), one_best, many_best);
    }

    for log in [16u32, 18] {
        let n = 1usize << log;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let reps = 3;
        let mut one_best = f64::MAX;
        let mut many_best = f64::MAX;
        let msm_many = CpuMsm::new();
        let msm1 = one.install(CpuMsm::new);
        for _ in 0..reps {
            one.install(|| {
                let t = Instant::now();
                let _ = std::hint::black_box(msm1.msm_g1(&bases, &scalars));
                one_best = one_best.min(t.elapsed().as_secs_f64() * 1e3);
            });
            let t = Instant::now();
            let _ = std::hint::black_box(msm_many.msm_g1(&bases, &scalars));
            many_best = many_best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        row(&format!("msm_g1 dense 2^{log}"), one_best, many_best);
    }

    // distribute_powers: pure elementwise, the easiest thing in the prover to scale.
    for log in [18u32, 20] {
        let n = 1usize << log;
        let mut a = rand_scalars(n, &mut rng);
        let shift = Fr::from(7u64);
        let mut one_best = f64::MAX;
        let mut many_best = f64::MAX;
        let d_many = CpuNtt::new();
        let d1 = one.install(CpuNtt::new);
        for _ in 0..5 {
            one.install(|| {
                let t = Instant::now();
                d1.distribute_powers(&mut a, shift);
                one_best = one_best.min(t.elapsed().as_secs_f64() * 1e3);
            });
            let t = Instant::now();
            d_many.distribute_powers(&mut a, shift);
            many_best = many_best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        row(&format!("distribute_powers 2^{log}"), one_best, many_best);
    }
}

/// Wall clock alone cannot tell "rayon is not parallelising" apart from "rayon is
/// parallelising but the cores are already busy". CPU time can: if the N-thread arm burns
/// N times the CPU seconds for the same wall clock, the threads really did run and the
/// work simply does not scale.
fn cpu_time_s() -> f64 {
    // getrusage(RUSAGE_SELF) via the libc the std already links.
    #[repr(C)]
    #[derive(Default)]
    struct Timeval {
        sec: i64,
        usec: i32,
        _pad: i32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Rusage {
        utime: Timeval,
        stime: Timeval,
        rest: [i64; 14],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut r = Rusage::default();
    unsafe { getrusage(0, &mut r) };
    r.utime.sec as f64 + r.utime.usec as f64 / 1e6 + r.stime.sec as f64 + r.stime.usec as f64 / 1e6
}

#[test]
#[ignore = "probe"]
fn cpu_seconds_per_stage() {
    let mut rng = test_rng();
    let nthreads = rayon::current_num_threads();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    println!("\n== CPU seconds vs wall clock, global pool = {nthreads} threads ==");
    println!(
        "{:<24} {:>8} {:>10} {:>10} {:>12}",
        "stage/arm", "wall ms", "cpu ms", "cpu/wall", "cores used"
    );

    let probe = |name: &str, reps: usize, mut f: Box<dyn FnMut() + '_>| {
        let c0 = cpu_time_s();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let wall = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let cpu = (cpu_time_s() - c0) * 1e3 / reps as f64;
        println!(
            "{:<24} {:>8.2} {:>10.2} {:>10.2} {:>12.2}",
            name,
            wall,
            cpu,
            cpu / wall,
            cpu / wall
        );
    };

    for log in [18u32, 20] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let ntt_many = CpuNtt::new();
        ntt_many.ntt(&domain, &mut a, Direction::Forward);
        probe(
            &format!("ntt 2^{log} N-thread"),
            3,
            Box::new(|| ntt_many.ntt(&domain, &mut a, Direction::Forward)),
        );
    }

    for log in [18u32] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let ntt1 = one.install(CpuNtt::new);
        one.install(|| ntt1.ntt(&domain, &mut a, Direction::Forward));
        probe(
            &format!("ntt 2^{log} 1-thread"),
            3,
            Box::new(|| one.install(|| ntt1.ntt(&domain, &mut a, Direction::Forward))),
        );
    }

    for log in [18u32] {
        let n = 1usize << log;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let msm = CpuMsm::new();
        probe(
            &format!("msm 2^{log} N-thread"),
            3,
            Box::new(|| {
                let _ = std::hint::black_box(msm.msm_g1(&bases, &scalars));
            }),
        );
    }
}

/// Scaling curve over explicit pool sizes. The point of stopping at 4 is that this box
/// has roughly four idle cores, so 1 -> 2 -> 4 measures the algorithm's scalability
/// without measuring the scheduler fighting the other tenants for cores 5..12.
#[test]
#[ignore = "probe"]
fn scaling_curve_small_pools() {
    let mut rng = test_rng();
    let sizes = [1usize, 2, 4, 8, 12];
    let pools: Vec<rayon::ThreadPool> = sizes
        .iter()
        .map(|&t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        })
        .collect();

    println!("\n== scaling curve (ms, best of reps; speedup vs 1 thread) ==");
    print!("{:<22}", "stage");
    for t in sizes {
        print!("{:>16}", format!("{t} thr"));
    }
    println!();

    // NTT at 2^18 and 2^20.
    for log in [18u32, 20] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        let reps = if log >= 20 { 3 } else { 5 };
        let mut base = 0.0f64;
        print!("{:<22}", format!("ntt 2^{log}"));
        for (i, pool) in pools.iter().enumerate() {
            let ntt = pool.install(CpuNtt::new);
            pool.install(|| ntt.ntt(&domain, &mut a, Direction::Forward)); // warm twiddles
            let mut best = f64::MAX;
            for _ in 0..reps {
                pool.install(|| {
                    let t = Instant::now();
                    ntt.ntt(&domain, &mut a, Direction::Forward);
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                });
            }
            if i == 0 {
                base = best;
            }
            print!("{:>16}", format!("{best:.1} ({:.2}x)", base / best));
        }
        println!();
    }

    // MSM at 2^18, dense.
    {
        let n = 1usize << 18;
        let bases: Vec<G1Affine> = walk_points(n, &mut rng);
        let scalars = rand_scalars(n, &mut rng);
        let mut base = 0.0f64;
        print!("{:<22}", "msm_g1 2^18");
        for (i, pool) in pools.iter().enumerate() {
            let msm = pool.install(CpuMsm::new);
            let mut best = f64::MAX;
            for _ in 0..3 {
                pool.install(|| {
                    let t = Instant::now();
                    let _ = std::hint::black_box(msm.msm_g1(&bases, &scalars));
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                });
            }
            if i == 0 {
                base = best;
            }
            print!("{:>16}", format!("{best:.1} ({:.2}x)", base / best));
        }
        println!();
    }

    // distribute_powers at 2^20: pure elementwise, the scalability upper bound.
    {
        let n = 1usize << 20;
        let mut a = rand_scalars(n, &mut rng);
        let shift = Fr::from(7u64);
        let mut base = 0.0f64;
        print!("{:<22}", "distribute 2^20");
        for (i, pool) in pools.iter().enumerate() {
            let d = pool.install(CpuNtt::new);
            let mut best = f64::MAX;
            for _ in 0..5 {
                pool.install(|| {
                    let t = Instant::now();
                    d.distribute_powers(&mut a, shift);
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                });
            }
            if i == 0 {
                base = best;
            }
            print!("{:>16}", format!("{best:.1} ({:.2}x)", base / best));
        }
        println!();
    }
}

/// The NTT switches from the serial to the parallel path at `PARALLEL_THRESHOLD = 2^13`.
/// A fine sweep across that boundary shows whether the constant is in the right place: if
/// ns/point drops sharply *at* 2^13, everything below it was being left on one core for
/// no reason. `js_1x1_d8` runs at domain 2^12, i.e. entirely on the serial side.
#[test]
#[ignore = "probe"]
fn ntt_around_the_parallel_threshold() {
    let mut rng = test_rng();
    let ntt = CpuNtt::new();
    println!(
        "\n== NTT across PARALLEL_THRESHOLD (2^13), rayon threads = {} ==",
        threads()
    );
    println!(
        "{:>5} {:>10} {:>12} {:>10}",
        "log n", "ms", "ns/point", "path"
    );
    for log in [10u32, 11, 12, 13, 14, 15] {
        let n = 1usize << log;
        let domain = Domain::new(n).unwrap();
        let mut a = rand_scalars(n, &mut rng);
        ntt.ntt(&domain, &mut a, Direction::Forward); // warm twiddles
        let ms = best_ms(50, || ntt.ntt(&domain, &mut a, Direction::Forward));
        println!(
            "{:>5} {:>10.4} {:>12.2} {:>10}",
            log,
            ms,
            ms * 1e6 / n as f64,
            if n >= (1 << 13) { "parallel" } else { "serial" }
        );
    }
}
