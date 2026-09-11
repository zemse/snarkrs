//! `g16 ptau fft-bench`: the CUDA FFT kernel-variant sweep, in one process.
//!
//! The economics this command exists for: any edit to `kernels/fft.cu` costs a full
//! NVRTC + ptxas rebuild on the measuring machine, minutes of billed GPU time, because
//! the PTX cache keys on the source. So the candidate kernels are all instantiated in
//! one translation unit (`G16_CUDA_FFT_VARIANTS=1`, see `g16-cuda/src/fft.rs`
//! `VARIANTS`) and this command pays one compile, one context, and one input
//! generation for the whole table. Every row after the first costs only its own
//! kernel time.
//!
//! What a row measures: the wall clock of one `ifft_many` call over `--copies`
//! independent blocks of `2^power` synthetic points, which includes the host pack,
//! PCIe both ways and the unpack, exactly what `ptau prepare` pays per block. Per-pass
//! device-only times come from `--time` (CUDA events around every launch, printed to
//! stderr), and the fixed startup costs from `G16_CUDA_TIME=1`, so the three layers of
//! the cost can be told apart in a single run.
//!
//! Correctness is checked against the CPU transform per variant as curve points, not
//! bytes: different ladders produce different XYZZ representatives of the same point,
//! and only `ptau prepare`'s exit through `batch_to_affine` makes bytes comparable.
//! The byte-identity oracle therefore stays the full `ptau prepare` sha256 comparison;
//! this check catches a wrong point, not a wrong representative.

use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use g16_ceremony::CpuGroupFft;
use g16_cuda::fft::FftKernels;
use g16_cuda::Cuda;
use g16_field::raw::{RawFq, RawFq2};
use g16_field::{CurveGroup, G1Projective, G2Projective, PrimeGroup};
use g16_msm::xyzz::{to_projective, Xyzz};
use g16_msm::GroupFft;

#[derive(clap::Args, Debug)]
pub struct Args {
    /// log2 of the block length; the transform is one block of 2^POWER points.
    #[arg(long)]
    pub power: u32,
    /// Variants to run, comma separated, in order. Default: every variant the compiled
    /// unit carries (all of them, since this command compiles the experiment set).
    #[arg(long, value_delimiter = ',')]
    pub variants: Vec<String>,
    /// Threads-per-block values to sweep. Launches clamp to a variant's own
    /// `__launch_bounds__` maximum.
    #[arg(long, value_delimiter = ',', default_value = "128")]
    pub blocks: Vec<u32>,
    /// g1, g2 or both.
    #[arg(long, default_value = "both")]
    pub group: String,
    /// Timed repetitions per row. The first row of a fresh process is also the warmup;
    /// discard iter 0 of the very first row when reading the table.
    #[arg(long, default_value_t = 3)]
    pub iters: usize,
    /// Independent copies of the block handed to one `ifft_many` call, the co-dispatch
    /// shape `prepare` produces at small powers.
    #[arg(long, default_value_t = 1)]
    pub copies: usize,
    /// CUDA streams the copies spread over. 1 is the shipped serial shape; raising it
    /// with `--copies` above 1 measures what concurrent blocks are worth.
    #[arg(long, default_value_t = 1)]
    pub streams: usize,
    /// Per-pass device times via CUDA events, printed to stderr.
    #[arg(long, default_value_t = false)]
    pub time: bool,
    /// Compare each variant's first result against the CPU transform.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub check: bool,
}

/// The group-shaped plumbing, so the sweep is written once.
trait BenchGroup {
    type Raw: Copy + Send + Sync + 'static;
    const NAME: &'static str;

    fn gen(n: usize) -> Vec<Xyzz<Self::Raw>>;
    fn ifft_cpu(a: &mut [Xyzz<Self::Raw>]) -> Result<()>;
    fn ifft_dev(k: &FftKernels, blocks: &mut [&mut [Xyzz<Self::Raw>]]) -> Result<()>;
    /// Equality as curve points; see the module docs for why bytes cannot serve.
    fn pt_eq(a: &Xyzz<Self::Raw>, b: &Xyzz<Self::Raw>) -> bool;
}

struct G1;
struct G2;

impl BenchGroup for G1 {
    type Raw = RawFq;
    const NAME: &'static str = "G1";

    /// Successive multiples of the generator: deterministic, cheap (one addition per
    /// point plus one batch normalization), and structure-free for this workload, whose
    /// per-thread cost is driven by the twiddle scalars rather than by the points.
    fn gen(n: usize) -> Vec<Xyzz<RawFq>> {
        let g = G1Projective::generator();
        let mut acc = g;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(acc);
            acc += g;
        }
        G1Projective::normalize_batch(&pts)
            .into_iter()
            .map(|p| Xyzz {
                x: RawFq::from_fq(&p.x),
                y: RawFq::from_fq(&p.y),
                zz: RawFq::ONE,
                zzz: RawFq::ONE,
            })
            .collect()
    }

    fn ifft_cpu(a: &mut [Xyzz<RawFq>]) -> Result<()> {
        CpuGroupFft.ifft_g1(a).map_err(|e| anyhow!("{e}"))
    }

    fn ifft_dev(k: &FftKernels, blocks: &mut [&mut [Xyzz<RawFq>]]) -> Result<()> {
        k.ifft_g1_many(blocks).map_err(|e| anyhow!("{e}"))
    }

    fn pt_eq(a: &Xyzz<RawFq>, b: &Xyzz<RawFq>) -> bool {
        to_projective::<g16_field::g1::Config>(a) == to_projective::<g16_field::g1::Config>(b)
    }
}

impl BenchGroup for G2 {
    type Raw = RawFq2;
    const NAME: &'static str = "G2";

    fn gen(n: usize) -> Vec<Xyzz<RawFq2>> {
        let g = G2Projective::generator();
        let mut acc = g;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(acc);
            acc += g;
        }
        G2Projective::normalize_batch(&pts)
            .into_iter()
            .map(|p| Xyzz {
                x: RawFq2::from_fq2(&p.x),
                y: RawFq2::from_fq2(&p.y),
                zz: RawFq2::ONE,
                zzz: RawFq2::ONE,
            })
            .collect()
    }

    fn ifft_cpu(a: &mut [Xyzz<RawFq2>]) -> Result<()> {
        CpuGroupFft.ifft_g2(a).map_err(|e| anyhow!("{e}"))
    }

    fn ifft_dev(k: &FftKernels, blocks: &mut [&mut [Xyzz<RawFq2>]]) -> Result<()> {
        k.ifft_g2_many(blocks).map_err(|e| anyhow!("{e}"))
    }

    fn pt_eq(a: &Xyzz<RawFq2>, b: &Xyzz<RawFq2>) -> bool {
        to_projective::<g16_field::g2::Config>(a) == to_projective::<g16_field::g2::Config>(b)
    }
}

/// Ladders one block of `2^p` points costs with the fused final pass:
/// mix passes 1..p-1 contribute `n/2 - n/2^e` each and the fused pass `n`, which sums
/// to `(p-1) * 2^(p-1) + 2`. The Ml/s column divides by this, so it is comparable
/// across powers and against the Metal ladder rates.
fn ladders(p: u32) -> u64 {
    if p == 0 {
        return 0;
    }
    (u64::from(p) - 1) * (1u64 << (p - 1)) + 2
}

pub fn run(args: Args) -> Result<()> {
    if args.power == 0 || args.power > 24 {
        bail!(
            "--power must be in 1..=24 (2^{} points is not a sweep)",
            args.power
        );
    }
    // The gated variants exist in the unit only when the define is in the source
    // (kernels.rs reads this env at compile time). Anything beyond the shipped `c5`
    // needs it, so default it on; an explicit value, e.g. `0` to sweep only the cheap
    // shipped unit, is respected.
    let wants_gated = args.variants.is_empty() || args.variants.iter().any(|v| v != "c5");
    if wants_gated && std::env::var_os("G16_CUDA_FFT_VARIANTS").is_none() {
        std::env::set_var("G16_CUDA_FFT_VARIANTS", "1");
    }

    let ordinal = match std::env::var("G16_CUDA_DEVICE") {
        Ok(v) => v.trim().parse::<usize>()?,
        Err(_) => 0,
    };
    let t = Instant::now();
    let cuda = Cuda::new(ordinal).map_err(|e| anyhow!("no usable CUDA device {ordinal}: {e}"))?;
    let (maj, min) = cuda.compute_capability();
    println!(
        "fft-bench: context {:.1} ms, {} (cc {maj}.{min}, {} SMs)",
        t.elapsed().as_secs_f64() * 1e3,
        cuda.device_name(),
        cuda.sm_count()
    );
    println!(
        "fft-bench: G16_CUDA_FF_PTX={}, G16_CUDA_FFT_VARIANTS={}",
        std::env::var("G16_CUDA_FF_PTX").unwrap_or_else(|_| "unset".into()),
        std::env::var("G16_CUDA_FFT_VARIANTS").unwrap_or_else(|_| "unset".into()),
    );

    // The one compile of the run. Cold on a fresh source hash this is the long pole,
    // by design paid here once; set G16_CUDA_TIME=1 for the nvrtc/load split.
    let t = Instant::now();
    let module = FftKernels::compile(&cuda)?;
    println!(
        "fft-bench: unit compile+load {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    let probe = FftKernels::from_module(&cuda, module.clone())?;
    let available = probe.variant_names();
    let selected: Vec<String> = if args.variants.is_empty() {
        available.iter().map(|s| s.to_string()).collect()
    } else {
        for v in &args.variants {
            if !available.iter().any(|a| a == v) {
                bail!("variant {v:?} is not in the compiled unit; available: {available:?}");
            }
        }
        args.variants.clone()
    };
    drop(probe);

    println!(
        "fft-bench: power {}, {} ladders/block, copies {}, streams {}, iters {}",
        args.power,
        ladders(args.power),
        args.copies,
        args.streams,
        args.iters
    );
    println!("group variant block   iter    wall_s      Ml/s  check");

    if !["g1", "g2", "both"].contains(&args.group.as_str()) {
        bail!("--group must be g1, g2 or both");
    }
    // The closure keeps the module type (a cudarc handle) inside g16-cuda's API; every
    // instance it makes shares the one compiled module, so a row costs a function bind
    // and nothing else.
    let mk = |variant: &str, block: u32| -> Result<FftKernels> {
        Ok(FftKernels::from_module(&cuda, module.clone())?
            .with_variant(variant)
            .map_err(|e| anyhow!("{e}"))?
            .with_min_block(1)
            .with_block(block)
            .with_streams(args.streams)
            .with_timing(args.time))
    };
    let mut mismatches = 0usize;
    if args.group == "g1" || args.group == "both" {
        mismatches += run_group::<G1, _>(&args, &mk, &selected)?;
    }
    if args.group == "g2" || args.group == "both" {
        mismatches += run_group::<G2, _>(&args, &mk, &selected)?;
    }
    // A wrong variant fails the command, but only after every row ran: the compile this
    // process paid for should never be wasted on an early exit.
    if mismatches > 0 {
        bail!("{mismatches} row(s) disagreed with the CPU transform; see MISMATCH above");
    }
    Ok(())
}

/// Runs one group's rows and returns how many failed the CPU comparison.
fn run_group<G: BenchGroup, F: Fn(&str, u32) -> Result<FftKernels>>(
    args: &Args,
    mk: &F,
    selected: &[String],
) -> Result<usize> {
    let n = 1usize << args.power;
    let t = Instant::now();
    let base = G::gen(n);
    println!(
        "fft-bench: {} input 2^{} generated in {:.2} s",
        G::NAME,
        args.power,
        t.elapsed().as_secs_f64()
    );

    let reference = if args.check {
        let mut r = base.clone();
        let t = Instant::now();
        G::ifft_cpu(&mut r)?;
        println!(
            "fft-bench: {} cpu reference ifft {:.2} s (this host's fallback rate)",
            G::NAME,
            t.elapsed().as_secs_f64()
        );
        Some(r)
    } else {
        None
    };

    let total_ladders = ladders(args.power) * args.copies as u64;
    let mut mismatches = 0usize;
    for variant in selected {
        for &block in &args.blocks {
            let k = mk(variant, block)?;
            for iter in 0..args.iters {
                let mut copies: Vec<Vec<Xyzz<G::Raw>>> = vec![base.clone(); args.copies];
                let t = Instant::now();
                {
                    let mut refs: Vec<&mut [Xyzz<G::Raw>]> =
                        copies.iter_mut().map(|c| c.as_mut_slice()).collect();
                    G::ifft_dev(&k, &mut refs)?;
                }
                let wall = t.elapsed().as_secs_f64();

                let check = match (&reference, iter) {
                    (Some(r), 0) => {
                        let ok = copies
                            .iter()
                            .all(|c| c.iter().zip(r.iter()).all(|(a, b)| G::pt_eq(a, b)));
                        if !ok {
                            mismatches += 1;
                        }
                        if ok {
                            "ok"
                        } else {
                            "MISMATCH"
                        }
                    }
                    (Some(_), _) => "-",
                    (None, _) => "off",
                };
                println!(
                    "{:<5} {:<7} {:<7} {:<4} {:>9.3} {:>9.3}  {check}",
                    G::NAME,
                    variant,
                    block,
                    iter,
                    wall,
                    total_ladders as f64 / wall / 1e6
                );
            }
        }
    }
    Ok(mismatches)
}
