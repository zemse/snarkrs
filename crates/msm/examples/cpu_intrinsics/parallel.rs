//! Optional aggregate samples. Dispatch and completion remain in the wall time.

use super::*;
use std::mem::size_of;

type WorkerResult<T, const L: usize> = (usize, Option<usize>, [T; L]);

pub(super) fn validate_work(
    rounds: usize,
    lanes: usize,
    threads: usize,
    element_bytes: usize,
) -> ProbeResult<usize> {
    if rounds == 0 || lanes == 0 || threads == 0 || element_bytes == 0 {
        return Err("parallel work dimensions must be positive".into());
    }
    let operations = rounds
        .checked_mul(lanes)
        .and_then(|n| n.checked_mul(threads))
        .ok_or("parallel operation count overflow")?;
    let worker_bytes = lanes
        .checked_mul(element_bytes)
        .and_then(|n| n.checked_add(3 * size_of::<usize>()))
        .ok_or("parallel worker storage overflow")?;
    let total_bytes = worker_bytes
        .checked_mul(threads)
        .ok_or("parallel storage sum overflow")?;
    if total_bytes > isize::MAX as usize {
        return Err("parallel storage exceeds Vec allocation capacity".into());
    }
    Ok(operations)
}

fn worker_seeds<T: Copy>(
    threads: usize,
    mut make: impl FnMut(&mut StdRng) -> T,
) -> ProbeResult<Vec<[T; 8]>> {
    validate_work(1, 8, threads, size_of::<T>())?;
    let mut seeds = Vec::new();
    seeds.try_reserve_exact(threads)?;
    for worker in 0..threads {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(worker as u64));
        seeds.push(std::array::from_fn(|_| make(&mut rng)));
    }
    Ok(seeds)
}

fn kernel<T: Copy + Send + Sync, const L: usize>(
    pool: &rayon::ThreadPool,
    seeds: &[[T; 8]],
    rounds: usize,
    op: impl Fn(usize, T) -> T + Sync,
) -> Vec<WorkerResult<T, L>> {
    pool.broadcast(|ctx| {
        let worker = ctx.index();
        let initial = black_box(std::array::from_fn(|i| seeds[worker][i]));
        let result = chains::<_, L>(initial, rounds, |x| op(worker, x));
        (worker, rayon::current_thread_index(), result)
    })
}

fn verify_workers<T, const L: usize>(results: &[WorkerResult<T, L>], threads: usize) {
    assert_eq!(results.len(), threads);
    for (i, (worker, actual, _)) in results.iter().enumerate() {
        assert_eq!(*worker, i);
        assert_eq!(*actual, Some(i));
    }
}

fn sample<T: Copy + Send + Sync, const L: usize>(
    name: &str,
    seeds: &[[T; 8]],
    op: impl Fn(usize, T) -> T + Sync,
    hash: impl Fn(T) -> u64,
    cfg: &Config,
    pool: &rayon::ThreadPool,
) -> ProbeResult {
    let threads = pool.current_num_threads();
    let operations = validate_work(cfg.rounds, L, threads, size_of::<T>())?;
    assert_eq!(seeds.len(), threads);
    black_box(kernel::<_, L>(pool, seeds, cfg.rounds, &op));
    for rep in 0..cfg.reps {
        let input = black_box(seeds);
        let start = Instant::now();
        let results = kernel::<_, L>(pool, input, cfg.rounds, &op);
        let elapsed = elapsed_ns(start)?;
        verify_workers(&results, threads);
        let sum = checksum(
            results
                .iter()
                .flat_map(|(_, _, x)| x.iter().copied())
                .map(&hash),
        );
        let seed_bytes = std::mem::size_of_val(seeds);
        println!("{{\"kind\":\"parallel_primitive\",\"name\":{},\"mode\":\"aggregate_interleaved_throughput\",\"sample\":{rep},\"chains_per_worker\":{L},\"actual_kernel_threads\":{threads},\"operations\":{operations},\"elapsed_ns\":{elapsed},\"aggregate_ns_per_operation\":{},\"aggregate_operations_per_s\":{},\"preallocated_seed_bytes\":{seed_bytes},\"dispatches\":1,\"dispatch_and_completion_included\":true,\"checksum\":\"{sum:016x}\"}}", json(name), elapsed as f64 / operations as f64, operations as f64 * 1e9 / elapsed as f64);
    }
    Ok(())
}

fn measure<T: Copy + Send + Sync>(
    name: &str,
    seeds: &[[T; 8]],
    op: impl Fn(usize, T) -> T + Sync,
    hash: impl Fn(T) -> u64 + Copy,
    cfg: &Config,
    pool: &rayon::ThreadPool,
) -> ProbeResult {
    sample::<_, 1>(name, seeds, &op, hash, cfg, pool)?;
    sample::<_, 4>(name, seeds, &op, hash, cfg, pool)?;
    sample::<_, 8>(name, seeds, &op, hash, cfg, pool)
}

fn dispatch(cfg: &Config, pool: &rayon::ThreadPool) -> ProbeResult {
    let threads = pool.current_num_threads();
    for rep in 0..=cfg.reps {
        let start = Instant::now();
        let results = pool.broadcast(|ctx| {
            (
                ctx.index(),
                rayon::current_thread_index(),
                [black_box(ctx.index() as u64)],
            )
        });
        let elapsed = elapsed_ns(start)?;
        verify_workers(&results, threads);
        let sum = checksum(results.iter().map(|(_, _, x)| x[0]));
        if rep > 0 {
            println!("{{\"kind\":\"parallel_dispatch\",\"name\":\"broadcast_trivial_closure\",\"sample\":{},\"actual_kernel_threads\":{threads},\"dispatches\":1,\"elapsed_ns\":{elapsed},\"ns_per_dispatch\":{elapsed},\"subtracted_from_kernel_time\":false,\"checksum\":\"{sum:016x}\"}}", rep - 1);
        }
    }
    Ok(())
}

type MixedInputs<F> = (Vec<[Xyzz<F>; 8]>, Vec<(F, F)>);

fn mixed_inputs<P: RawCurve>(threads: usize) -> ProbeResult<MixedInputs<P::RF>> {
    use ark_ec::short_weierstrass::Projective;
    use snarkrs_field::raw::RawField;
    let seeds = worker_seeds(threads, |rng| loop {
        let point = Projective::<P>::rand(rng).into_affine();
        if !point.is_zero() {
            let (x, y) = P::raw_xy(&point);
            break Xyzz {
                x,
                y,
                zz: P::RF::ONE,
                zzz: P::RF::ONE,
            };
        }
    })?;
    validate_work(1, 1, threads, size_of::<(P::RF, P::RF)>())?;
    let mut operands = Vec::new();
    operands.try_reserve_exact(threads)?;
    for (worker, lanes) in seeds.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(worker as u64) ^ 0xadd0_2026);
        loop {
            let point = Projective::<P>::rand(&mut rng).into_affine();
            if point.is_zero() {
                continue;
            }
            let (x, y) = P::raw_xy(&point);
            if lanes.iter().all(|p| p.x != x) {
                operands.push((x, y));
                break;
            }
        }
    }
    Ok((seeds, operands))
}

pub(super) fn differential_gate(pool: &rayon::ThreadPool) -> ProbeResult {
    differential_width::<1>(pool)?;
    differential_width::<4>(pool)?;
    differential_width::<8>(pool)
}

fn differential_width<const L: usize>(pool: &rayon::ThreadPool) -> ProbeResult {
    let threads = pool.current_num_threads();
    {
        let seeds = worker_seeds(threads, Fr::rand)?;
        let results = kernel::<_, L>(pool, &seeds, 17, |i, x| x * seeds[i][7]);
        verify_workers(&results, threads);
        for (i, _, result) in results {
            for (lane, x) in result.into_iter().enumerate() {
                assert_eq!(x, seeds[i][lane] * seeds[i][7].pow([17]));
            }
        }
    }
    {
        let seeds = worker_seeds(threads, |rng| RawFq::from_fq(&Fq::rand(rng)))?;
        let results = kernel::<_, L>(pool, &seeds, 17, |i, x| x.mul(seeds[i][7]));
        verify_workers(&results, threads);
        for (i, _, result) in results {
            for (lane, x) in result.into_iter().enumerate() {
                assert_eq!(
                    x.to_fq(),
                    seeds[i][lane].to_fq() * seeds[i][7].to_fq().pow([17])
                );
            }
        }
    }
    {
        let seeds = worker_seeds(threads, |rng| RawFq2::from_fq2(&Fq2::rand(rng)))?;
        let results = kernel::<_, L>(pool, &seeds, 17, |i, x| x.mul(seeds[i][7]));
        verify_workers(&results, threads);
        for (i, _, result) in results {
            for (lane, x) in result.into_iter().enumerate() {
                assert_eq!(
                    x.to_fq2(),
                    seeds[i][lane].to_fq2() * seeds[i][7].to_fq2().pow([17])
                );
            }
        }
    }
    mixed_differential_gate::<g1::Config, L>(pool)?;
    mixed_differential_gate::<g2::Config, L>(pool)
}

fn mixed_differential_gate<P: RawCurve, const L: usize>(pool: &rayon::ThreadPool) -> ProbeResult {
    use ark_ec::short_weierstrass::Affine;
    let threads = pool.current_num_threads();
    let (seeds, operands) = mixed_inputs::<P>(threads)?;
    let results = kernel::<_, L>(pool, &seeds, 17, |i, mut p| {
        p.madd(operands[i].0, operands[i].1);
        p
    });
    verify_workers(&results, threads);
    for (i, _, result) in results {
        let rhs =
            Affine::<P>::new_unchecked(P::field_back(operands[i].0), P::field_back(operands[i].1));
        for (lane, p) in result.into_iter().enumerate() {
            let initial = to_projective::<P>(&seeds[i][lane]);
            let expected = initial + rhs.into_group() * P::ScalarField::from(17u64);
            assert_eq!(to_projective::<P>(&p).into_affine(), expected.into_affine());
        }
    }
    Ok(())
}

pub(super) fn run(cfg: &Config, pool: &rayon::ThreadPool) -> ProbeResult {
    let threads = pool.current_num_threads();
    validate_work(cfg.rounds, 8, threads, size_of::<Xyzz<RawFq2>>())?;
    println!("{{\"kind\":\"parallel_provenance\",\"worker_seed_rule\":\"StdRng(SEED wrapping_add worker_index)\",\"actual_pool_threads\":{threads},\"chain_counts\":[1,4,8],\"oracle_rounds\":17,\"oracle_primitive_count\":5,\"measured_results_oracle_checked\":false,\"saturation_certified\":false}}");
    dispatch(cfg, pool)?;
    {
        let seeds = worker_seeds(threads, Fr::rand)?;
        measure("Fr.mul", &seeds, |i, x| x * seeds[i][7], fr_hash, cfg, pool)?;
    }
    {
        let seeds = worker_seeds(threads, |rng| RawFq::from_fq(&Fq::rand(rng)))?;
        measure(
            "RawFq.mul",
            &seeds,
            |i, x| x.mul(seeds[i][7]),
            fq_hash,
            cfg,
            pool,
        )?;
    }
    {
        let seeds = worker_seeds(threads, |rng| RawFq2::from_fq2(&Fq2::rand(rng)))?;
        measure(
            "RawFq2.mul",
            &seeds,
            |i, x| x.mul(seeds[i][7]),
            fq2_hash,
            cfg,
            pool,
        )?;
    }
    {
        let (seeds, operands) = mixed_inputs::<g1::Config>(threads)?;
        measure(
            "G1.XYZZ.mixed_add",
            &seeds,
            |i, mut p| {
                p.madd(operands[i].0, operands[i].1);
                p
            },
            |p| checksum([fq_hash(p.x), fq_hash(p.y), fq_hash(p.zz), fq_hash(p.zzz)]),
            cfg,
            pool,
        )?;
    }
    {
        let (seeds, operands) = mixed_inputs::<g2::Config>(threads)?;
        measure(
            "G2.XYZZ.mixed_add",
            &seeds,
            |i, mut p| {
                p.madd(operands[i].0, operands[i].1);
                p
            },
            |p| {
                checksum([
                    fq2_hash(p.x),
                    fq2_hash(p.y),
                    fq2_hash(p.zz),
                    fq2_hash(p.zzz),
                ])
            },
            cfg,
            pool,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn actual_parallel_kernels_match_field_and_group_oracles() {
        for threads in [1, 2, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            differential_gate(&pool).unwrap();
        }
    }

    fn check_operand_selection<P: RawCurve>() {
        let (seeds, operands) = mixed_inputs::<P>(3).unwrap();
        assert_eq!(seeds.len(), operands.len());
        for (lanes, (x, _)) in seeds.iter().zip(operands) {
            assert!(lanes.iter().all(|p| p.x != x));
        }
    }

    #[test]
    fn mixed_operands_are_distinct_from_all_initial_lanes() {
        check_operand_selection::<g1::Config>();
        check_operand_selection::<g2::Config>();
    }

    #[test]
    fn checks_work_and_storage_before_allocation() {
        assert_eq!(validate_work(17, 8, 3, 32).unwrap(), 408);
        assert!(validate_work(usize::MAX, 8, 2, 32).is_err());
        assert!(validate_work(1, 8, usize::MAX, 32).is_err());
        assert!(validate_work(1, 8, 2, usize::MAX).is_err());
        assert!(validate_work(1, 8, (isize::MAX as usize) / 64, 32).is_err());
        assert!(worker_seeds::<u64>(usize::MAX, |_| 1).is_err());
    }

    #[test]
    fn workers_are_distinct_and_operation_counts_are_exact() {
        for threads in [1, 2, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let seeds = worker_seeds(threads, Fr::rand).unwrap();
            assert!(threads == 1 || seeds[0] != seeds[1]);
            let visits = AtomicUsize::new(0);
            let results = kernel::<_, 8>(&pool, &seeds, 17, |i, x| {
                visits.fetch_add(1, Ordering::Relaxed);
                x * seeds[i][7]
            });
            verify_workers(&results, threads);
            assert_eq!(visits.load(Ordering::Relaxed), 17 * 8 * threads);
            for (i, _, result) in results {
                assert_eq!(result, seeds[i].map(|x| x * seeds[i][7].pow([17])));
            }
        }
    }
}
