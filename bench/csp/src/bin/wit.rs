//! `g16-csp-wit`: time witness generation on its own, nothing else.
//!
//! The bench binary reports witness generation as one number inside a proof. This one
//! runs only that call, so a sampling profiler attributing time inside the linked
//! witnesscalc library is not also reading the prover's stack.
//!
//! Every iteration compares its `.wtns` bytes against the first iteration's, outside the
//! timed region. A witness that stops matching is not a faster witness.

use anyhow::{ensure, Result};
use clap::Parser;
use g16_csp::{circuits, inputs, Target, Variant};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "time circom witness generation, one variant per row")]
struct Cli {
    /// Restrict to one target. Default: all of them.
    #[arg(long, value_enum)]
    target: Option<Target>,
    /// Restrict to one input size. Default: every size the target defines.
    #[arg(long)]
    input_size: Option<usize>,
    /// Timed iterations per variant, after one warm-up.
    #[arg(long, default_value_t = 20)]
    reps: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let variants: Vec<Variant> = Variant::all()
        .into_iter()
        .filter(|v| cli.target.is_none_or(|t| t == v.target))
        .filter(|v| cli.input_size.is_none_or(|n| n == v.input_size))
        .collect();
    ensure!(!variants.is_empty(), "no variant matches that selection");

    println!(
        "{:<14} {:>10} {:>12} {:>9} {:>9} {:>9} {:>9}",
        "variant", "entries", "wtns bytes", "min ms", "median", "mean", "max ms"
    );
    for v in variants {
        let input = inputs::json(v.target, v.input_size);
        let reference = circuits::witness(v, input.clone())?;
        let entries = (reference.len() - 64) / 32;

        let mut times = Vec::with_capacity(cli.reps);
        for _ in 0..cli.reps {
            let t = Instant::now();
            let wtns = circuits::witness(v, input.clone())?;
            times.push(t.elapsed());
            ensure!(
                wtns == reference,
                "{}: witness changed between reps",
                v.name()
            );
        }
        times.sort_unstable();
        let mean = times.iter().sum::<Duration>() / times.len() as u32;
        println!(
            "{:<14} {:>10} {:>12} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            v.name(),
            entries,
            reference.len(),
            ms(times[0]),
            ms(times[times.len() / 2]),
            ms(mean),
            ms(times[times.len() - 1]),
        );
    }
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
