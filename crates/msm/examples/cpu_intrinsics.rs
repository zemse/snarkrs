//! Actual CPU primitives, not a proving benchmark. See cpu_intrinsics.md.

use std::{env, error::Error, hint::black_box, time::Instant};

use ark_std::rand::{rngs::StdRng, SeedableRng};
use snarkrs_field::{
    g1, g2,
    raw::{RawFq, RawFq2},
    AffineRepr, CurveGroup, Field, Fq, Fq2, Fr, G1Projective, G2Projective, PrimeField,
    UniformRand,
};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};

const SEED: u64 = 0x5eed_2026;
type ProbeResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

fn elapsed_ns(start: Instant) -> ProbeResult<u128> {
    let ns = start.elapsed().as_nanos();
    positive_duration(ns)
}

fn positive_duration(ns: u128) -> ProbeResult<u128> {
    if ns == 0 {
        return Err("sample below timer resolution; increase PROBE_ROUNDS or PROBE_PASSES".into());
    }
    Ok(ns)
}

struct Config {
    rounds: usize,
    reps: usize,
    passes: usize,
    sets: Vec<usize>,
    threads: usize,
}

fn setting(name: &str, default: usize) -> ProbeResult<usize> {
    let n = match env::var(name) {
        Ok(s) => s.parse()?,
        Err(env::VarError::NotPresent) => default,
        Err(e) => return Err(e.into()),
    };
    if n == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(n)
}

impl Config {
    fn load() -> ProbeResult<Self> {
        let sets = env::var("PROBE_BYTES").unwrap_or_else(|_| "32768,1048576,16777216".into());
        let sets: Vec<usize> = sets.split(',').map(str::parse).collect::<Result<_, _>>()?;
        if sets.is_empty() || sets.iter().any(|n| *n == 0 || n % 8 != 0) {
            return Err("PROBE_BYTES must contain positive multiples of eight".into());
        }
        let cfg = Self {
            rounds: setting("PROBE_ROUNDS", 16384)?,
            reps: setting("PROBE_REPS", 5)?,
            passes: setting("PROBE_PASSES", 16)?,
            sets,
            threads: setting("PROBE_THREADS", std::thread::available_parallelism()?.get())?,
        };
        for &bytes in &cfg.sets {
            memory_layout(bytes, cfg.passes)?;
        }
        Ok(cfg)
    }
}

fn json(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn checksum(words: impl IntoIterator<Item = u64>) -> u64 {
    words.into_iter().fold(SEED, |a, b| a.rotate_left(7) ^ b)
}

fn fr_hash(x: Fr) -> u64 {
    checksum(x.into_bigint().0)
}
fn fq_hash(x: RawFq) -> u64 {
    checksum(x.0)
}
fn fq2_hash(x: RawFq2) -> u64 {
    checksum(x.c0.0.into_iter().chain(x.c1.0))
}

// Interleave independent chains in the inner loop, not whole chains in sequence.
#[inline(never)]
fn chains<T: Copy, const L: usize>(mut x: [T; L], rounds: usize, op: impl Fn(T) -> T) -> [T; L] {
    for _ in 0..rounds {
        for v in &mut x {
            *v = op(*v);
        }
    }
    black_box(x)
}

fn sample<T: Copy, const L: usize>(
    name: &str,
    seeds: [T; 8],
    op: impl Fn(T) -> T + Copy,
    hash: impl Fn(T) -> u64,
    cfg: &Config,
) -> ProbeResult {
    let initial = std::array::from_fn(|i| seeds[i]);
    black_box(chains::<T, L>(black_box(initial), cfg.rounds, op));
    for rep in 0..cfg.reps {
        let initial = black_box(initial);
        let t = Instant::now();
        let result = chains::<T, L>(initial, cfg.rounds, op);
        let elapsed = elapsed_ns(t)?;
        let sum = checksum(result.into_iter().map(&hash));
        let operations = cfg.rounds as u128 * L as u128;
        println!(
            "{{\"kind\":\"primitive\",\"name\":{},\"mode\":\"{}\",\"chains\":{L},\"threads\":1,\"sample\":{rep},\"operations\":{operations},\"elapsed_ns\":{elapsed},\"ns_per_operation\":{},\"checksum\":\"{sum:016x}\"}}",
            json(name),
            if L == 1 { "dependent_latency" } else { "interleaved_throughput" },
            elapsed as f64 / operations as f64,
        );
    }
    Ok(())
}

fn primitive<T: Copy>(
    name: &str,
    seeds: [T; 8],
    op: impl Fn(T) -> T + Copy,
    hash: impl Fn(T) -> u64 + Copy,
    cfg: &Config,
) -> ProbeResult {
    sample::<_, 1>(name, seeds, op, hash, cfg)?;
    sample::<_, 4>(name, seeds, op, hash, cfg)?;
    sample::<_, 8>(name, seeds, op, hash, cfg)?;
    Ok(())
}

fn field_probes(cfg: &Config) -> ProbeResult {
    let mut rng = StdRng::seed_from_u64(SEED);
    let fr: [Fr; 8] = std::array::from_fn(|_| Fr::rand(&mut rng));
    let fq: [RawFq; 8] = std::array::from_fn(|_| RawFq::from_fq(&Fq::rand(&mut rng)));
    let fq2: [RawFq2; 8] = std::array::from_fn(|_| RawFq2::from_fq2(&Fq2::rand(&mut rng)));
    let b = black_box(fr[7]);
    primitive("Fr.mul", fr, |x| x * b, fr_hash, cfg)?;
    primitive("Fr.square", fr, |x| x.square(), fr_hash, cfg)?;
    primitive("Fr.add", fr, |x| x + b, fr_hash, cfg)?;
    primitive(
        "Fr.inverse_reciprocal_chain",
        fr,
        |x| x.inverse().expect("nonzero seed"),
        fr_hash,
        cfg,
    )?;
    let b = black_box(fq[7]);
    primitive("RawFq.mul", fq, |x| x.mul(b), fq_hash, cfg)?;
    primitive("RawFq.square", fq, |x| x.sqr(), fq_hash, cfg)?;
    primitive("RawFq.add", fq, |x| x.add(b), fq_hash, cfg)?;
    primitive(
        "RawFq.raw_inv_bridge_reciprocal_chain",
        fq,
        <g1::Config as RawCurve>::raw_inv,
        fq_hash,
        cfg,
    )?;
    let b = black_box(fq2[7]);
    primitive("RawFq2.mul", fq2, |x| x.mul(b), fq2_hash, cfg)?;
    primitive("RawFq2.square", fq2, |x| x.sqr(), fq2_hash, cfg)?;
    primitive("RawFq2.add", fq2, |x| x.add(b), fq2_hash, cfg)?;
    primitive(
        "RawFq2.raw_inv_bridge_reciprocal_chain",
        fq2,
        <g2::Config as RawCurve>::raw_inv,
        fq2_hash,
        cfg,
    )?;
    Ok(())
}

fn mixed_probes(cfg: &Config) -> ProbeResult {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points: [G1Projective; 8] = std::array::from_fn(|_| G1Projective::rand(&mut rng));
    let seeds = points.map(|p| {
        let (x, y) = <g1::Config as RawCurve>::raw_xy(&p.into_affine());
        Xyzz {
            x,
            y,
            zz: RawFq::ONE,
            zzz: RawFq::ONE,
        }
    });
    let (px, py) = black_box(<g1::Config as RawCurve>::raw_xy(
        &G1Projective::rand(&mut rng).into_affine(),
    ));
    primitive(
        "G1.XYZZ.mixed_add",
        seeds,
        |mut p| {
            p.madd(px, py);
            p
        },
        |p| checksum([fq_hash(p.x), fq_hash(p.y), fq_hash(p.zz), fq_hash(p.zzz)]),
        cfg,
    )?;
    let points: [G2Projective; 8] = std::array::from_fn(|_| G2Projective::rand(&mut rng));
    let seeds = points.map(|p| {
        let (x, y) = <g2::Config as RawCurve>::raw_xy(&p.into_affine());
        Xyzz {
            x,
            y,
            zz: RawFq2::ONE,
            zzz: RawFq2::ONE,
        }
    });
    let (px, py) = black_box(<g2::Config as RawCurve>::raw_xy(
        &G2Projective::rand(&mut rng).into_affine(),
    ));
    primitive(
        "G2.XYZZ.mixed_add",
        seeds,
        |mut p| {
            p.madd(px, py);
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
    )?;
    Ok(())
}

fn memory_layout(bytes: usize, passes: usize) -> ProbeResult<(usize, usize, usize)> {
    if bytes == 0 || bytes % 8 != 0 || passes == 0 {
        return Err(
            "memory buffers require positive multiples of eight and positive passes".into(),
        );
    }
    if bytes > isize::MAX as usize {
        return Err("PROBE_BYTES exceeds Vec allocation capacity".into());
    }
    let traffic = bytes.checked_mul(passes).ok_or("byte count overflow")?;
    let working_set = bytes.checked_mul(2).ok_or("working set overflow")?;
    let copy_traffic = traffic.checked_mul(2).ok_or("copy byte count overflow")?;
    Ok((traffic, working_set, copy_traffic))
}

fn memory_probes(cfg: &Config) -> ProbeResult {
    for &bytes in &cfg.sets {
        let (traffic, working_set, copy_traffic) = memory_layout(bytes, cfg.passes)?;
        let n = bytes / 8;
        let mut src = Vec::<u64>::new();
        let mut dst = Vec::<u64>::new();
        src.try_reserve_exact(n)?;
        dst.try_reserve_exact(n)?;
        src.extend((0..n).map(|i| (i as u64).wrapping_mul(SEED)));
        dst.resize(n, 0);
        for rep in 0..=cfg.reps {
            let t = Instant::now();
            for _ in 0..cfg.passes {
                black_box(dst.as_mut_slice()).copy_from_slice(black_box(src.as_slice()));
            }
            let elapsed = elapsed_ns(t)?;
            assert_eq!(dst, src);
            let sum = checksum(dst.iter().copied());
            if rep > 0 {
                println!("{{\"kind\":\"memory\",\"name\":\"copy_from_slice\",\"threads\":1,\"sample\":{},\"passes\":{},\"buffer_bytes\":{bytes},\"working_set_bytes\":{working_set},\"logical_bytes\":{copy_traffic},\"elapsed_ns\":{elapsed},\"logical_gb_per_s\":{},\"checksum\":\"{sum:016x}\"}}", rep - 1, cfg.passes, copy_traffic as f64 / elapsed as f64);
            }
            let t = Instant::now();
            let mut sum = 0u64;
            for _ in 0..cfg.passes {
                let pass = black_box(src.as_slice())
                    .iter()
                    .fold(0u64, |a, b| a.wrapping_add(*b));
                sum = sum.wrapping_add(black_box(pass));
            }
            let elapsed = elapsed_ns(t)?;
            let expected = src
                .iter()
                .fold(0u64, |a, b| a.wrapping_add(*b))
                .wrapping_mul(cfg.passes as u64);
            assert_eq!(sum, expected);
            if rep > 0 {
                println!("{{\"kind\":\"memory\",\"name\":\"stream_read_u64_sum\",\"threads\":1,\"sample\":{},\"passes\":{},\"buffer_bytes\":{bytes},\"working_set_bytes\":{bytes},\"logical_bytes\":{traffic},\"elapsed_ns\":{elapsed},\"logical_gb_per_s\":{},\"checksum\":\"{sum:016x}\"}}", rep - 1, cfg.passes, traffic as f64 / elapsed as f64);
            }
        }
    }
    Ok(())
}

fn fork_join(cfg: &Config, pool: &rayon::ThreadPool) -> ProbeResult {
    pool.install(|| {
        for rep in 0..=cfg.reps {
            let t = Instant::now();
            let mut sum = 0u64;
            for i in 0..cfg.rounds {
                let (a, b) = rayon::join(|| black_box(i as u64), || black_box(i as u64 + 1));
                sum = sum.wrapping_add(a ^ b);
            }
            let elapsed = elapsed_ns(t)?;
            let expected = (0..cfg.rounds).fold(0u64, |a, i| a.wrapping_add(i as u64 ^ (i as u64 + 1)));
            assert_eq!(sum, expected);
            if rep > 0 {
                println!("{{\"kind\":\"scheduling\",\"name\":\"rayon_join_two_trivial_closures\",\"threads\":{},\"sample\":{},\"joins\":{},\"elapsed_ns\":{elapsed},\"ns_per_join\":{},\"checksum\":\"{sum:016x}\"}}", pool.current_num_threads(), rep - 1, cfg.rounds, elapsed as f64 / cfg.rounds as f64);
            }
        }
        Ok(())
    })
}

// Differential gates run before any timing, using the production raw/XYZZ paths.
fn correctness() {
    let mut rng = StdRng::seed_from_u64(SEED);
    for _ in 0..32 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);
        let x = RawFq::from_fq(&a);
        let y = RawFq::from_fq(&b);
        assert_eq!(x.mul(y).to_fq(), a * b);
        assert_eq!(x.sqr().to_fq(), a.square());
        assert_eq!(x.add(y).to_fq(), a + b);
        assert_eq!(
            <g1::Config as RawCurve>::raw_inv(x).to_fq(),
            a.inverse().unwrap()
        );
        let a = Fq2::rand(&mut rng);
        let b = Fq2::rand(&mut rng);
        let x = RawFq2::from_fq2(&a);
        let y = RawFq2::from_fq2(&b);
        assert_eq!(x.mul(y).to_fq2(), a * b);
        assert_eq!(x.sqr().to_fq2(), a.square());
        assert_eq!(x.add(y).to_fq2(), a + b);
        assert_eq!(
            <g2::Config as RawCurve>::raw_inv(x).to_fq2(),
            a.inverse().unwrap()
        );
    }
    mixed_gate::<g1::Config>();
    mixed_gate::<g2::Config>();
    let seeds = std::array::from_fn(|i| Fr::from(i as u64 + 2));
    let got = chains::<_, 8>(seeds, 17, |x| x * Fr::from(11u64));
    assert_eq!(got, seeds.map(|x| x * Fr::from(11u64).pow([17])));
}

fn mixed_gate<P: RawCurve>() {
    let base = ark_ec::short_weierstrass::Affine::<P>::generator();
    let (x, y) = P::raw_xy(&base);
    let mut p = Xyzz::<P::RF>::ZERO;
    let mut oracle = base.into_group();
    for i in 0..17 {
        p.madd(x, y);
        if i != 0 {
            oracle += base;
        }
        assert_eq!(to_projective::<P>(&p).into_affine(), oracle.into_affine());
    }
    let (nx, ny) = P::raw_xy(&-base);
    p.madd(nx, ny);
    oracle -= base;
    assert_eq!(to_projective::<P>(&p).into_affine(), oracle.into_affine());
    let mut cancel = Xyzz::<P::RF>::ZERO;
    cancel.madd(x, y);
    cancel.madd(nx, ny);
    assert!(cancel.is_zero());
    cancel.madd(x, y);
    assert_eq!(to_projective::<P>(&cancel).into_affine(), base);
}

fn main() -> ProbeResult {
    let cfg = Config::load()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()?;
    correctness();
    println!("{{\"kind\":\"provenance\",\"schema\":1,\"arch\":{},\"os\":{},\"debug_assertions\":{},\"compiled_adx\":{},\"compiled_bmi2\":{},\"configured_threads\":{},\"actual_pool_threads\":{},\"rounds\":{},\"reps\":{},\"seed\":{SEED},\"revision_declared\":{},\"build_declared\":{},\"contention_declared\":{},\"correctness\":\"passed\"}}", json(env::consts::ARCH), json(env::consts::OS), cfg!(debug_assertions), cfg!(target_feature = "adx"), cfg!(target_feature = "bmi2"), cfg.threads, pool.current_num_threads(), cfg.rounds, cfg.reps, json(&env::var("PROBE_REV").unwrap_or_else(|_| "unknown".into())), json(&env::var("PROBE_BUILD").unwrap_or_else(|_| "unknown".into())), json(&env::var("PROBE_CONTENTION").unwrap_or_else(|_| "unknown".into())));
    field_probes(&cfg)?;
    mixed_probes(&cfg)?;
    memory_probes(&cfg)?;
    fork_join(&cfg, &pool)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_memory_is_rejected_before_reservation() {
        let cfg = Config {
            rounds: 32,
            reps: 1,
            passes: 1,
            sets: vec![usize::MAX & !7],
            threads: 1,
        };
        assert!(memory_layout(cfg.sets[0], cfg.passes).is_err());
        assert!(memory_probes(&cfg).is_err());
    }

    #[test]
    fn memory_layout_checks_capacity_and_all_byte_counts() {
        assert_eq!(memory_layout(4096, 16).unwrap(), (65536, 8192, 131072));
        assert!(memory_layout(8, usize::MAX).is_err());
        assert!(memory_layout(8, usize::MAX / 8).is_err());
        assert!(memory_layout(0, 1).is_err());
        assert!(memory_layout(7, 1).is_err());
        assert!(memory_layout(8, 0).is_err());
        assert!(memory_layout((isize::MAX as usize) & !7, 1).is_ok());
    }

    #[test]
    fn rejects_unresolved_timer_samples() {
        assert!(positive_duration(0).is_err());
        assert_eq!(positive_duration(1).unwrap(), 1);
    }

    #[test]
    fn differential_gates() {
        correctness();
    }

    #[test]
    fn interleaving_counts_every_chain() {
        let seeds = [3u64; 8];
        assert_eq!(chains::<_, 1>([3], 17, |x| x + 2), [37]);
        assert_eq!(chains::<_, 4>([3; 4], 17, |x| x + 2), [37; 4]);
        assert_eq!(chains::<_, 8>(seeds, 17, |x| x + 2), [37; 8]);
    }

    #[test]
    fn provenance_json_escaping() {
        assert_eq!(json("a\n\"\\"), "\"a\\u000a\\\"\\\\\"");
    }
}
