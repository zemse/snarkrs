//! `g16-csp`: run the ethproofs client-side-proving benchmark against this prover.

use anyhow::Result;
use clap::{Parser, Subcommand};
use g16_csp::{run, Backend, Prover, Target, Variant};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "ethproofs client-side-proving benchmarks, run against g16")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Time proving and verification, and write one metrics row per variant.
    Bench(BenchArgs),
    /// Write `input.json`, `circuit.wtns`, `vkey.json` and `public.json` next to each
    /// zkey, so `g16 bench` and the audit tests can use the same circuits.
    Artifacts(ArtifactArgs),
    /// Print the variants and their constraint counts, proving nothing.
    List(ArtifactArgs),
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long, default_value = "bench/artifacts/csp")]
    artifacts: PathBuf,
    #[arg(long, default_value = "bench/vendor/csp-benchmarks/circom/circuits")]
    witness_src: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    /// Measure the `rapidsnark` CLI at this path instead of our prover, over the same
    /// zkeys and the same witness. `--backend` is then ignored.
    #[arg(long, value_name = "BIN")]
    rapidsnark: Option<PathBuf>,
    /// Timed iterations per variant, after one warm-up. Upstream publishes a mean of ten.
    #[arg(long, default_value_t = 10)]
    reps: usize,
    #[arg(long, default_value = "bench/results/csp/metrics")]
    out_dir: PathBuf,
    /// Peak-RSS samples per variant, each a fresh `g16-csp-mem` process. 0 skips it.
    #[arg(long, default_value_t = 10)]
    mem_reps: usize,
    /// Override the single-shot prover. Defaults to `g16-csp-mem` beside this binary.
    #[arg(long)]
    mem_bin: Option<PathBuf>,
    /// Restrict to one target. Default: all of them.
    #[arg(long, value_enum)]
    target: Option<Target>,
    /// Restrict to one input size. Default: every size the target defines.
    #[arg(long)]
    input_size: Option<usize>,
}

#[derive(clap::Args)]
struct ArtifactArgs {
    #[arg(long, default_value = "bench/artifacts/csp")]
    artifacts: PathBuf,
    #[arg(long, value_enum)]
    target: Option<Target>,
    #[arg(long)]
    input_size: Option<usize>,
}

fn selected(target: Option<Target>, input_size: Option<usize>) -> Vec<Variant> {
    Variant::all()
        .into_iter()
        .filter(|v| target.is_none_or(|t| t == v.target))
        .filter(|v| input_size.is_none_or(|n| n == v.input_size))
        .collect()
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Bench(a) => bench(a),
        Cmd::Artifacts(a) => artifacts(a),
        Cmd::List(a) => list(a),
    }
}

fn bench(a: BenchArgs) -> Result<()> {
    let variants = selected(a.target, a.input_size);
    anyhow::ensure!(!variants.is_empty(), "no variant matches that selection");
    let mem_bin = match a.mem_bin {
        Some(p) => p,
        None => std::env::current_exe()?
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no directory for the running binary"))?
            .join("g16-csp-mem"),
    };
    anyhow::ensure!(
        a.mem_reps == 0 || a.rapidsnark.is_some() || mem_bin.is_file(),
        "{} missing; build it or pass --mem-reps 0",
        mem_bin.display()
    );
    let prover = match a.rapidsnark {
        Some(bin) => {
            anyhow::ensure!(bin.is_file(), "{} is not a file", bin.display());
            Prover::Rapidsnark(bin)
        }
        None => Prover::G16(a.backend),
    };
    let cfg = run::Config {
        artifacts: &a.artifacts,
        witness_src: &a.witness_src,
        prover,
        reps: a.reps,
        out_dir: &a.out_dir,
        mem_reps: a.mem_reps,
        mem_bin: &mem_bin,
    };

    println!(
        "{:<16} {:>11} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9} {:>10}",
        "variant",
        "constraints",
        "witness",
        "zkey",
        "prepare",
        "prove",
        "total",
        "verify",
        "peak rss"
    );
    for v in variants {
        let (row, b) = run::variant(v, &cfg)?;
        println!(
            "{:<16} {:>11} {:>8.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>8.1}ms {:>7.2}ms {:>8.0}MB",
            b.variant,
            row.num_constraints,
            b.witness_ms,
            b.zkey_load_ms,
            b.prepare_ms,
            b.prove_ms,
            b.total_ms,
            b.verify_vkey_ms,
            row.peak_memory as f64 / 1e6
        );
    }
    println!("\nrows written to {}", a.out_dir.display());
    Ok(())
}

fn artifacts(a: ArtifactArgs) -> Result<()> {
    for v in selected(a.target, a.input_size) {
        let dir = a.artifacts.join(v.name());
        let zkey = v.zkey(&a.artifacts);
        if !zkey.is_file() {
            println!("skip  {:<16} no circuit.zkey", v.name());
            continue;
        }
        let input_json = g16_csp::inputs::json(v.target, v.input_size);
        let wtns = g16_csp::circuits::witness(v, input_json.clone())?;
        let pk = g16_zkey::ProvingKey::load(&zkey)?;
        let witness = g16_zkey::wtns::Witness::from_bytes(wtns.clone())?.0;
        let public = &witness[1..=pk.n_public];

        std::fs::write(dir.join("input.json"), &input_json)?;
        std::fs::write(dir.join("circuit.wtns"), &wtns)?;
        g16_ceremony::vkey::export_verification_key(&zkey, &dir.join("vkey.json"))?;
        std::fs::write(
            dir.join("public.json"),
            g16_core::json::public_to_string(public),
        )?;
        println!(
            "ok    {:<16} {} constraints, witness {} bytes",
            v.name(),
            run::num_constraints(&pk),
            wtns.len()
        );
    }
    Ok(())
}

fn list(a: ArtifactArgs) -> Result<()> {
    println!(
        "{:<16} {:>12} {:>14}",
        "variant", "constraints", "zkey bytes"
    );
    for v in selected(a.target, a.input_size) {
        let zkey = v.zkey(&a.artifacts);
        let Ok(meta) = std::fs::metadata(&zkey) else {
            println!("{:<16} {:>12} {:>14}", v.name(), "-", "missing");
            continue;
        };
        let pk = g16_zkey::ProvingKey::load(&zkey)?;
        println!(
            "{:<16} {:>12} {:>14}",
            v.name(),
            run::num_constraints(&pk),
            meta.len()
        );
    }
    Ok(())
}
