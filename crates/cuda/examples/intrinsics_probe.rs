//! Opt-in calibration, not a production benchmark or a throughput/floor claim.
//!
//! Run with `cargo run --release --locked -p snarkrs-cuda --features cuda
//! --example intrinsics_probe`. Configuration is entirely through environment variables:
//! G16_PROBE_FAMILY=all|g2|fq|stores|overhead (default overhead),
//! G16_PROBE_SHAPE=all|latency|production|saturation, G16_PROBE_OP=all|add|double|offset,
//! G16_PROBE_SPAN_MIB=all|20|40, G16_PROBE_STORE=all|contiguous|strided,
//! G16_PROBE_BUFFERS=all|reused|rotated. Numeric settings: REPS=20, WARMUPS=2,
//! STEPS=32, OFFSET=504, ROTATION=4, QUEUE=32, SEED=1, DEVICE=0, SAT_BLOCKS=0
//! (automatic: 8 blocks per actual SM), SAT_THREADS=256, all with G16_PROBE_ prefix.
//! G16_CUDA_FF_PTX and G16_CUDA_SYNC retain their production meanings and defaults.
//!
//! Each warm/measured repetition is checked before its sample is accepted. Transfers,
//! poison, allocation and the CPU oracle are outside all reported intervals. Event
//! times retain the driver's float precision; event pairs have finite resolution and
//! queued timings may include host submission gaps. Launch builders are prepared before
//! timing. Host submission includes intermediate waits in the per-launch-sync case;
//! launch_sync_wall includes the end-event record and stream wait, not the start-event
//! record. No-op work is explicitly separate from arithmetic/stores. Reused/rotated
//! stores describe buffer addresses: every sample follows a full-span host poison
//! upload, not a fully cache-warm floor measurement.
//! Timing method: NVIDIA CUDA Best Practices Guide, section 9.1.2:
//! https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers
//! Event precision: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html

use std::{env, error::Error, time::Instant};

use ark_ec::AdditiveGroup;
use cudarc::driver::{sys, CudaEvent, CudaFunction, LaunchConfig, PushKernelArg};
use snarkrs_cuda::{as_words, from_words, Cuda};
use snarkrs_field::{
    AffineRepr, CurveGroup, Field, Fq, Fq2, Fr, G2Affine, G2Projective, PrimeGroup, Zero,
};
use snarkrs_gpu_layout::{PackedFq, PackedFq2, Reduced};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const POISON: u32 = 0xa5c3_7e19;
const SOURCE: &str = include_str!("../src/kernels/intrinsics_probe.cu");

#[derive(Debug)]
struct Config {
    family: String,
    shape: String,
    op: String,
    span: String,
    store: String,
    buffers: String,
    reps: usize,
    warmups: usize,
    steps: u32,
    offset: u32,
    rotation: usize,
    queue: usize,
    seed: u64,
    device: usize,
    sat_blocks: u32,
    sat_threads: u32,
}

fn setting(name: &str) -> Result<Option<String>> {
    match env::var(format!("G16_PROBE_{name}")) {
        Ok(s) => Ok(Some(s)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn choice(name: &str, value: Option<String>, default: &str, allowed: &[&str]) -> Result<String> {
    let s = value.unwrap_or_else(|| default.to_owned());
    if !allowed.contains(&s.as_str()) {
        return Err(format!("G16_PROBE_{name} must be one of {allowed:?}, got {s:?}").into());
    }
    Ok(s)
}

fn number<T>(name: &str, value: Option<String>, default: T, min: T, max: T) -> Result<T>
where
    T: std::str::FromStr + PartialOrd + std::fmt::Display + Copy,
{
    let n = match value {
        Some(s) => s
            .parse()
            .map_err(|_| format!("invalid G16_PROBE_{name}: {s:?}"))?,
        None => default,
    };
    if n < min || n > max {
        return Err(format!("G16_PROBE_{name} must be in {min}..={max}").into());
    }
    Ok(n)
}

impl Config {
    fn load() -> Result<Self> {
        let c = Self {
            family: choice(
                "FAMILY",
                setting("FAMILY")?,
                "overhead",
                &["all", "g2", "fq", "stores", "overhead"],
            )?,
            shape: choice(
                "SHAPE",
                setting("SHAPE")?,
                "all",
                &["all", "latency", "production", "saturation"],
            )?,
            op: choice(
                "OP",
                setting("OP")?,
                "all",
                &["all", "add", "double", "offset"],
            )?,
            span: choice(
                "SPAN_MIB",
                setting("SPAN_MIB")?,
                "all",
                &["all", "20", "40"],
            )?,
            store: choice(
                "STORE",
                setting("STORE")?,
                "all",
                &["all", "contiguous", "strided"],
            )?,
            buffers: choice(
                "BUFFERS",
                setting("BUFFERS")?,
                "all",
                &["all", "reused", "rotated"],
            )?,
            reps: number("REPS", setting("REPS")?, 20, 1, 10000)?,
            warmups: number("WARMUPS", setting("WARMUPS")?, 2, 1, 10000)?,
            steps: number("STEPS", setting("STEPS")?, 32, 1, 1000000)?,
            offset: number("OFFSET", setting("OFFSET")?, 504, 1, 65535)?,
            rotation: number("ROTATION", setting("ROTATION")?, 4, 2, 16)?,
            queue: number("QUEUE", setting("QUEUE")?, 32, 1, 100000)?,
            seed: number("SEED", setting("SEED")?, 1, 1, u64::MAX)?,
            device: number("DEVICE", setting("DEVICE")?, 0, 0, 1024)?,
            sat_blocks: number("SAT_BLOCKS", setting("SAT_BLOCKS")?, 0, 0, 65535)?,
            sat_threads: number("SAT_THREADS", setting("SAT_THREADS")?, 256, 1, 1024)?,
        };
        validate_shape(c.sat_blocks, c.sat_threads, true)?;
        Ok(c)
    }

    fn includes(&self, family: &str) -> bool {
        selected(&self.family, family)
    }
}

fn selected(selection: &str, value: &str) -> bool {
    selection == "all" || selection == value
}

fn geometry(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn validate_shape(blocks: u32, threads: u32, automatic: bool) -> Result<()> {
    let n = u64::from(blocks) * u64::from(threads);
    if threads == 0 || threads > 1024 || n > 4_194_304 || (n == 0 && !automatic) {
        return Err(
            "arithmetic shape must contain 1..=4194304 threads, at most 1024 per block".into(),
        );
    }
    Ok(())
}

fn shapes(c: &Config, cuda: &Cuda) -> Result<Vec<(&'static str, LaunchConfig)>> {
    let blocks = if c.sat_blocks == 0 {
        (cuda.sm_count() as u32)
            .checked_mul(8)
            .ok_or("SM count overflow")?
    } else {
        c.sat_blocks
    };
    let shapes = [
        ("latency", geometry(1, 1)),
        ("production", geometry(160, 64)),
        ("saturation", geometry(blocks, c.sat_threads)),
    ];
    let mut out = Vec::new();
    for (name, shape) in shapes {
        if selected(&c.shape, name) {
            validate_shape(shape.grid_dim.0, shape.block_dim.0, false)?;
            out.push((name, shape));
        }
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct Sample {
    gpu_us: f64,
    submit_us: f64,
    wall_us: f64,
}

fn summary(label: &str, metric: &str, values: impl Iterator<Item = f64>) {
    let mut v: Vec<_> = values.collect();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    let median = if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    };
    println!("summary label={label} metric={metric} n={} median_us={median:.6} min_us={:.6} max_us={:.6}", v.len(), v[0], v[v.len()-1]);
}

struct Timer<'a> {
    cuda: &'a Cuda,
    start: CudaEvent,
    end: CudaEvent,
}

impl<'a> Timer<'a> {
    fn new(cuda: &'a Cuda) -> Result<Self> {
        Ok(Self {
            cuda,
            start: cuda
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            end: cuda
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
        })
    }

    // Launch builders and all buffers/events are allocated before this interval.
    fn run(&self, mut launch: impl FnMut() -> Result<()>) -> Result<Sample> {
        let stream = self.cuda.stream();
        stream.synchronize()?;
        self.start.record(stream)?;
        let t = Instant::now();
        launch()?;
        let submit_us = t.elapsed().as_secs_f64() * 1e6;
        self.end.record(stream)?;
        stream.synchronize()?;
        let wall_us = t.elapsed().as_secs_f64() * 1e6;
        let gpu_us = f64::from(self.start.elapsed_ms(&self.end)?) * 1000.0;
        Ok(Sample {
            gpu_us,
            submit_us,
            wall_us,
        })
    }
}

fn measure(
    cuda: &Cuda,
    c: &Config,
    label: &str,
    mut repetition: impl FnMut(usize, &Timer<'_>) -> Result<Sample>,
) -> Result<()> {
    let timer = Timer::new(cuda)?;
    let mut samples = Vec::with_capacity(c.reps);
    for rep in 0..c.warmups + c.reps {
        // The callback checks every output after timing, before returning the sample.
        let sample = repetition(rep, &timer)?;
        let Sample {
            gpu_us,
            submit_us,
            wall_us,
        } = sample;
        let warm = rep < c.warmups;
        println!("sample label={label} phase={} rep={} gpu_us={gpu_us:.6} host_submit_us={submit_us:.6} launch_sync_wall_us={wall_us:.6} verified=true", if warm { "warmup" } else { "measured" }, if warm { rep } else { rep-c.warmups });
        if !warm {
            samples.push(sample);
        }
    }
    summary(label, "gpu", samples.iter().map(|s| s.gpu_us));
    summary(label, "host_submit", samples.iter().map(|s| s.submit_us));
    summary(label, "launch_sync_wall", samples.iter().map(|s| s.wall_us));
    Ok(())
}

fn compile(cuda: &Cuda, define: &str, unit: &'static str, name: &str) -> Result<CudaFunction> {
    let src = format!(
        "{define}\n{}\n{SOURCE}",
        snarkrs_cuda::kernels::unit_field_probe()
    );
    Ok(cuda.functions(unit, &src, &[name])?.remove(0))
}

fn expected_g2(p: G2Affine, q: G2Affine, op: &str, steps: u32, offset: u32) -> G2Affine {
    let mut x = p.into_group();
    for _ in 0..steps {
        match op {
            "add" => x += q,
            "double" => {
                x.double_in_place();
            }
            "offset" => x *= Fr::from(offset),
            _ => unreachable!(),
        }
    }
    x.into_affine()
}

fn verify_g2(words: &[u32], expected: G2Affine, n: usize) -> Result<()> {
    let coords: Vec<PackedFq2> = from_words(words).ok_or("G2 output layout")?;
    if coords.len() != n * 4 {
        return Err("G2 output length".into());
    }
    for (i, p) in coords.chunks_exact(4).enumerate() {
        if p.iter().any(|v| !v.reduced()) {
            return Err(format!("noncanonical G2 coordinate at {i}").into());
        }
        let [x, y, zz, zzz] = [p[0].to_fq2(), p[1].to_fq2(), p[2].to_fq2(), p[3].to_fq2()];
        let valid = if expected.is_zero() {
            x.is_zero() && y.is_zero() && zz.is_zero() && zzz.is_zero()
        } else {
            !zz.is_zero()
                && !zzz.is_zero()
                && zz.square() * zz == zzz.square()
                && x == expected.x * zz
                && y == expected.y * zzz
        };
        if !valid {
            return Err(format!("G2 oracle mismatch at thread {i}").into());
        }
    }
    Ok(())
}

fn g2_xyzz_inputs(p: G2Affine, q: G2Affine, seed: u64) -> [[PackedFq2; 4]; 2] {
    let zp = Fq2::new(Fq::from(seed) + Fq::from(3u32), Fq::from(5u32));
    let zq = Fq2::new(Fq::from(seed) + Fq::from(11u32), Fq::from(7u32));
    [(p, zp), (q, zq)].map(|(p, z)| {
        let zz = z.square();
        let zzz = zz * z;
        [p.x * zz, p.y * zzz, zz, zzz].map(|v| PackedFq2::from_fq2(&v))
    })
}

fn g2(cuda: &Cuda, c: &Config) -> Result<()> {
    // Different nonidentity points avoid the first-add equality shortcut. Full runtime
    // XYZZ inputs have distinct non-unit Fq2 scales; every thread has the same chain.
    let p = (G2Projective::generator() * Fr::from(c.seed)).into_affine();
    let q = (G2Projective::generator() * (Fr::from(c.seed) + Fr::from(17u32))).into_affine();
    let [packed_p, packed_q] = g2_xyzz_inputs(p, q, c.seed);
    verify_g2(as_words(&packed_p), p, 1)?;
    verify_g2(as_words(&packed_q), q, 1)?;
    for (op, define, unit) in [
        ("add", "#define G16_PROBE_G2 0", "intrinsic_g2_add"),
        ("double", "#define G16_PROBE_G2 1", "intrinsic_g2_double"),
        ("offset", "#define G16_PROBE_G2 2", "intrinsic_g2_offset"),
    ] {
        if !selected(&c.op, op) {
            continue;
        }
        let function = compile(cuda, define, unit, "intrinsic_g2")?;
        let expected = expected_g2(p, q, op, c.steps, c.offset);
        for (shape_name, shape) in shapes(c, cuda)? {
            let n = shape.grid_dim.0 as usize * shape.block_dim.0 as usize;
            let points: Vec<PackedFq2> = std::iter::repeat_n(packed_p, n)
                .chain(std::iter::repeat_n(packed_q, n))
                .flatten()
                .collect();
            let input = cuda.stream().clone_htod(as_words(&points))?;
            let poison = vec![POISON; n * 64];
            let mut output = cuda.stream().clone_htod(&poison)?;
            let label = format!("g2_{op}_{shape_name}");
            let calls = n as u64 * u64::from(c.steps);
            let bits = 32 - c.offset.leading_zeros();
            println!("config label={label} blocks={} threads={} outputs={n} steps={} top_level_calls={calls} offset={} ladder_dbl_calls={} ladder_add_calls={} input_case=nonidentity_runtime_xyzz_distinct_nonunit_fq2_scales_uniform exceptional_paths=possible_not_assumed_away", shape.grid_dim.0, shape.block_dim.0, c.steps, c.offset, if op == "offset" { calls * u64::from(bits) } else { 0 }, if op == "offset" { calls * u64::from(c.offset.count_ones()) } else { 0 });
            measure(cuda, c, &label, |_, timer| {
                cuda.stream().memcpy_htod(&poison, &mut output)?;
                let sample = {
                    let mut launch = cuda.stream().launch_builder(&function);
                    launch
                        .arg(&input)
                        .arg(&mut output)
                        .arg(&c.steps)
                        .arg(&c.offset);
                    timer.run(|| {
                        // SAFETY: input has two n-point XYZZ arrays, four packed Fq2 per 256B
                        // record in x/y/zz/zzz order; output has n complete XYZZ points.
                        unsafe { launch.launch(shape) }?;
                        Ok(())
                    })?
                };
                verify_g2(&cuda.stream().clone_dtoh(&output)?, expected, n)?;
                Ok(sample)
            })?;
        }
    }
    Ok(())
}

fn fq(cuda: &Cuda, c: &Config) -> Result<()> {
    let function = compile(cuda, "#define G16_PROBE_FQ", "intrinsic_fq", "intrinsic_fq")?;
    let x = Fq::from(c.seed);
    let y = x + Fq::from(17u32);
    let expected = PackedFq::from_fq(&(x * y.pow([u64::from(c.steps)])));
    for (shape_name, shape) in shapes(c, cuda)? {
        let n = shape.grid_dim.0 as usize * shape.block_dim.0 as usize;
        let mut inputs = vec![PackedFq::from_fq(&x); n];
        inputs.extend(vec![PackedFq::from_fq(&y); n]);
        let input = cuda.stream().clone_htod(as_words(&inputs))?;
        let poison = vec![POISON; n * 8];
        let mut output = cuda.stream().clone_htod(&poison)?;
        let label = format!("fq_mul_{shape_name}");
        println!(
            "config label={label} blocks={} threads={} steps={} fq_mul_calls={} outputs={n}",
            shape.grid_dim.0,
            shape.block_dim.0,
            c.steps,
            n as u64 * u64::from(c.steps)
        );
        measure(cuda, c, &label, |_, timer| {
            cuda.stream().memcpy_htod(&poison, &mut output)?;
            let sample = {
                let mut launch = cuda.stream().launch_builder(&function);
                launch.arg(&input).arg(&mut output).arg(&c.steps);
                timer.run(|| {
                    // SAFETY: two n-element Fq input arrays and one n-element output array.
                    unsafe { launch.launch(shape) }?;
                    Ok(())
                })?
            };
            let got = cuda.stream().clone_dtoh(&output)?;
            if got.len() != n * 8 || got.chunks_exact(8).any(|w| w != as_words(&[expected])) {
                return Err("Fq oracle mismatch".into());
            }
            Ok(sample)
        })?;
    }
    Ok(())
}

fn store_word(index: usize, strided: bool) -> u32 {
    if !strided || (32..48).contains(&(index % 64)) {
        0
    } else {
        POISON
    }
}

fn verify_store(words: &[u32], strided: bool) -> Result<()> {
    for (i, &word) in words.iter().enumerate() {
        if word != store_word(i, strided) {
            return Err(format!("store oracle mismatch at word {i}").into());
        }
    }
    Ok(())
}

fn stores(cuda: &Cuda, c: &Config) -> Result<()> {
    let clear = compile(
        cuda,
        "#define G16_PROBE_STORES",
        "intrinsic_stores",
        "intrinsic_clear",
    )?;
    let contiguous = compile(
        cuda,
        "#define G16_PROBE_STORES",
        "intrinsic_stores",
        "intrinsic_contiguous",
    )?;
    for mib in [20, 40] {
        if !selected(&c.span, &mib.to_string()) {
            continue;
        }
        let words = mib * 1024 * 1024 / 4;
        let poison = vec![POISON; words];
        for (pattern, strided) in [("contiguous", false), ("strided", true)] {
            if !selected(&c.store, pattern) {
                continue;
            }
            for (mode, count) in [("reused", 1), ("rotated", c.rotation)] {
                if !selected(&c.buffers, mode) {
                    continue;
                }
                let mut buffers = (0..count)
                    .map(|_| cuda.stream().clone_htod(&poison))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let work = if strided { words / 64 } else { words } as u32;
                let shape = geometry(work.div_ceil(256), 256);
                let function = if strided { &clear } else { &contiguous };
                let label = format!("sm_store_{mib}MiB_{pattern}_{mode}");
                println!("config label={label} span_bytes={} useful_bytes={} rotation_buffers={count} rotation_span_bytes={} blocks={} threads=256 poison=full_span_host_upload_before_each_rep zz_offset_bytes=128 row_bytes=256", words*4, if strided { words } else { words*4 }, words*4*count, shape.grid_dim.0);
                measure(cuda, c, &label, |rep, timer| {
                    let output = &mut buffers[rep % count];
                    cuda.stream().memcpy_htod(&poison, output)?;
                    let sample = {
                        let mut launch = cuda.stream().launch_builder(function);
                        launch.arg(&mut *output).arg(&work);
                        timer.run(|| {
                            // SAFETY: work is either the full word count or the number of 256B rows.
                            unsafe { launch.launch(shape) }?;
                            Ok(())
                        })?
                    };
                    verify_store(&cuda.stream().clone_dtoh(output)?, strided)?;
                    Ok(sample)
                })?;
            }
        }
    }
    Ok(())
}

fn overhead(cuda: &Cuda, c: &Config) -> Result<()> {
    let function = compile(
        cuda,
        "#define G16_PROBE_OVERHEAD",
        "intrinsic_overhead",
        "intrinsic_empty",
    )?;
    for (label, launches, waits) in [
        ("noop_event_pair", 0, false),
        ("noop_empty_kernel", 1, false),
        ("noop_queued_sequence", c.queue, false),
        ("noop_per_launch_sync_sequence", c.queue, true),
    ] {
        println!("config label={label} launches={launches} intermediate_stream_sync={waits} blocks=1 threads=1 oracle=no_output_driver_success_only");
        measure(cuda, c, label, |_, timer| {
            let mut launch = cuda.stream().launch_builder(&function);
            timer.run(|| {
                for _ in 0..launches {
                    // SAFETY: the empty kernel has no arguments or memory accesses.
                    unsafe { launch.launch(geometry(1, 1)) }?;
                    if waits {
                        cuda.stream().synchronize()?;
                    }
                }
                Ok(())
            })
        })?;
    }
    Ok(())
}

fn main() -> Result<()> {
    // Parse every setting before even trying to load the CUDA driver.
    let c = Config::load()?;
    println!("intrinsics_probe config={c:?}");
    for name in ["G16_CUDA_FF_PTX", "G16_CUDA_SYNC", "CUDA_VISIBLE_DEVICES"] {
        println!("environment {name}={:?}", env::var_os(name));
    }
    let cuda = Cuda::new(c.device)?;
    println!("device name={:?} capability={:?} sms={} stream=production_default timing=driver_f32_ms_to_f64_us no_floor_or_speedup_claim=true", cuda.device_name(), cuda.compute_capability(), cuda.sm_count());
    use sys::CUdevice_attribute::*;
    for (name, attr) in [
        (
            "max_threads_per_block",
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ),
        (
            "max_threads_per_sm",
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        ),
        ("warp_size", CU_DEVICE_ATTRIBUTE_WARP_SIZE),
        ("clock_rate_khz", CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
        (
            "memory_clock_rate_khz",
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        ),
        (
            "memory_bus_width_bits",
            CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        ),
        ("l2_bytes", CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
        (
            "registers_per_sm",
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        ),
        (
            "shared_bytes_per_sm",
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        ),
    ] {
        println!(
            "device_attribute {name}={}",
            cuda.context().attribute(attr)?
        );
    }
    if c.sat_threads
        > cuda
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as u32
    {
        return Err("SAT_THREADS exceeds device limit".into());
    }
    if c.includes("g2") {
        g2(&cuda, &c)?;
    }
    if c.includes("fq") {
        fq(&cuda, &c)?;
    }
    if c.includes("stores") {
        stores(&cuda, &c)?;
    }
    if c.includes("overhead") {
        overhead(&cuda, &c)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkrs_gpu_layout::PackedG2Affine;

    #[test]
    fn scaled_xyzz_inputs_round_trip() {
        assert_eq!(std::mem::size_of::<[PackedFq2; 4]>(), 256);
        for seed in [1, 73, u64::MAX] {
            let p = (G2Projective::generator() * Fr::from(seed)).into_affine();
            let q = (G2Projective::generator() * (Fr::from(seed) + Fr::from(17u32))).into_affine();
            let inputs = g2_xyzz_inputs(p, q, seed);
            let mut scales = Vec::new();
            for (packed, expected) in inputs.iter().zip([p, q]) {
                verify_g2(as_words(packed), expected, 1).unwrap();
                let decoded: Vec<PackedFq2> = from_words(as_words(packed)).unwrap();
                let zz = decoded[2].to_fq2();
                let zzz = decoded[3].to_fq2();
                let z = zzz / zz;
                assert!(!z.is_zero());
                assert_ne!(z, Fq2::from(1u32));
                assert!(!z.c0.is_zero() && !z.c1.is_zero());
                assert_eq!(zz, z.square());
                assert_eq!(zzz, z.square() * z);
                assert_eq!(decoded[0].to_fq2() / zz, expected.x);
                assert_eq!(decoded[1].to_fq2() / zzz, expected.y);
                scales.push(z);
            }
            assert_ne!(scales[0], scales[1]);
        }
    }

    #[test]
    fn invalid_config_values_fail() {
        for s in ["0", "-1", "bad", "10001", ""] {
            assert!(number("REPS", Some(s.into()), 20usize, 1, 10000).is_err());
        }
        assert_eq!(number("REPS", None, 20usize, 1, 10000).unwrap(), 20);
        assert!(choice("SPAN_MIB", Some("21".into()), "all", &["all", "20", "40"]).is_err());
        assert!(number("OFFSET", Some("65536".into()), 504u32, 1, 65535).is_err());
        assert!(number("ROTATION", Some("1".into()), 4usize, 2, 16).is_err());
        assert!(validate_shape(65535, 1024, true).is_err());
        assert!(validate_shape(1, 0, false).is_err());
        assert!(validate_shape(0, 256, false).is_err());
        validate_shape(0, 256, true).unwrap();
        validate_shape(160, 64, false).unwrap();
    }

    #[test]
    fn clear_layout_and_full_span_oracle() {
        assert_eq!(std::mem::size_of::<PackedFq2>(), 64);
        assert_eq!(std::mem::size_of::<PackedG2Affine>(), 128);
        let mut words: Vec<_> = (0..128).map(|i| store_word(i, true)).collect();
        assert_eq!(&words[32..48], &[0; 16]);
        assert_eq!(&words[0..32], &[POISON; 32]);
        assert_eq!(&words[48..96], &[POISON; 48]);
        verify_store(&words, true).unwrap();
        words[127] = 0;
        assert!(verify_store(&words, true).is_err());
        words[127] = POISON;
        words[33] = POISON;
        assert!(verify_store(&words, true).is_err());
        verify_store(&[0; 128], false).unwrap();
    }

    #[test]
    fn cpu_g2_expected_and_coordinate_oracle() {
        let p = G2Affine::generator();
        let q = (p * Fr::from(7u32)).into_affine();
        assert_eq!(
            expected_g2(p, q, "add", 3, 5),
            (p * Fr::from(22u32)).into_affine()
        );
        assert_eq!(
            expected_g2(p, q, "double", 3, 5),
            (p * Fr::from(8u32)).into_affine()
        );
        assert_eq!(
            expected_g2(p, q, "offset", 3, 5),
            (p * Fr::from(125u32)).into_affine()
        );
        let z = Fq2::from(9u32);
        let packed = [
            PackedFq2::from_fq2(&(p.x * z.square())),
            PackedFq2::from_fq2(&(p.y * z.square() * z)),
            PackedFq2::from_fq2(&z.square()),
            PackedFq2::from_fq2(&(z.square() * z)),
        ];
        verify_g2(as_words(&packed), p, 1).unwrap();
        for i in 0..4 {
            let mut bad = packed;
            bad[i] = PackedFq2::ZERO;
            assert!(verify_g2(as_words(&bad), p, 1).is_err());
        }
    }
}
