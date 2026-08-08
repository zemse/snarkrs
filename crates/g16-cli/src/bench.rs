//! `g16 bench`: cold and warm proving timings, with every proof verified first.
//!
//! Cold and warm each mean exactly one thing here, and the difference is the whole point
//! of the subcommand:
//!
//! * **cold** - every rep pays the full cost. The zkey parse, the witness parse, backend
//!   construction and `PreparedCircuit::prepare` all sit inside the timed region and
//!   everything is dropped afterwards, so no rep can inherit a warm cache from the one
//!   before it. This is what a CLI prover actually does, and on a GPU it is where the
//!   key upload shows up.
//! * **warm** - setup is paid once outside the timed region and only `prove()` is timed.
//!   This is what a resident service does, and it is the only mode in which a GPU can
//!   look good, which is exactly why vendor charts prefer it. The setup cost is still
//!   reported, in its own `prepare_ms` column, instead of being quietly dropped.
//!
//! One asymmetry worth stating rather than hiding: `bench/wrappers/rapidsnark-warm` mmaps
//! the `.wtns` once but re-parses it inside each timed call, while our warm loop parses
//! the witness once outside. For the artifacts here that parse is a few hundred
//! microseconds against tens of milliseconds of proving, so it does not change any
//! conclusion, but it does bias in our favour and it is not measured away.
//!
//! Timings are reported as median with min/max. Means over 15 reps on a laptop measure
//! whatever else the OS decided to do during the run.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use g16_core::{prove::prove, verify::verify, StageTimings};
use g16_field::Fr;
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

use crate::artifacts::{self, Variant};
use crate::{BackendKind, HostInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Cold,
    Warm,
    Both,
}

#[derive(clap::Args, Debug)]
pub struct Args {
    /// Directory of artifacts, one subdirectory per circuit variant.
    #[arg(long)]
    pub artifacts: PathBuf,
    /// Variant to benchmark; repeat for several. Default: every complete variant.
    #[arg(long, value_name = "NAME")]
    pub variant: Vec<String>,
    #[arg(long, default_value_t = 15)]
    pub reps: usize,
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    pub backend: BackendKind,
    #[arg(long, value_enum, default_value_t = Mode::Both)]
    pub mode: Mode,
    /// Append-free CSV output. Without it the numbers are only printed.
    #[arg(long, value_name = "FILE")]
    pub csv: Option<PathBuf>,
}

pub const CSV_HEADER: &str = "host,os,arch,cores,variant,constraints,prover,backend,mode,rep,ms,prepare_ms,gather_us,ntt_us,pointwise_us,msm_us,assemble_us,verified";

/// One proof: what it cost and whether it was accepted. `prepare_ms` is setup only, and
/// for a cold rep it is the part of `ms` spent before `prove()` was entered.
pub struct Row {
    pub variant: String,
    pub constraints: i64,
    pub backend: &'static str,
    pub mode: &'static str,
    pub rep: usize,
    pub ms: f64,
    pub prepare_ms: f64,
    pub timings: StageTimings,
    pub verified: bool,
}

impl Row {
    fn to_csv(&self, host: &HostInfo) -> String {
        let t = self.timings;
        format!(
            "{},{},{},{},{},{},ours,{},{},{},{:.3},{:.3},{},{},{},{},{},{}",
            host.host,
            host.os,
            host.arch,
            host.cores,
            self.variant,
            self.constraints,
            self.backend,
            self.mode,
            self.rep,
            self.ms,
            self.prepare_ms,
            t.gather_us,
            t.ntt_us,
            t.pointwise_us,
            t.msm_us,
            t.assemble_us,
            if self.verified { "yes" } else { "no" },
        )
    }
}

pub fn run(args: Args) -> Result<()> {
    if args.reps == 0 {
        bail!("--reps must be at least 1");
    }
    let host = HostInfo::detect();
    let variants = artifacts::selected(&args.artifacts, &args.variant)?;
    let modes: &[Mode] = match args.mode {
        Mode::Cold => &[Mode::Cold],
        Mode::Warm => &[Mode::Warm],
        Mode::Both => &[Mode::Cold, Mode::Warm],
    };

    let mut rows = Vec::new();
    for v in &variants {
        let constraints = v.constraints();
        println!("\n=== {} ({constraints} constraints) ===", v.name);
        for mode in modes {
            let got = match mode {
                Mode::Cold => cold(v, constraints, &args)?,
                Mode::Warm => warm(v, constraints, &args)?,
                Mode::Both => unreachable!("expanded above"),
            };
            report(&got);
            rows.extend(got);
        }
    }

    if let Some(path) = &args.csv {
        let mut out = String::from(CSV_HEADER);
        out.push('\n');
        for r in &rows {
            out.push_str(&r.to_csv(&host));
            out.push('\n');
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {} rows to {}", rows.len(), path.display());
    }
    Ok(())
}

/// The public signals snarkjs would publish for this witness: `w[1..=n_public]`.
fn public_of(witness: &[Fr], n_public: usize) -> Result<Vec<Fr>> {
    if witness.len() <= n_public {
        bail!(
            "witness has {} entries, too short for {n_public} public signals",
            witness.len()
        );
    }
    Ok(witness[1..=n_public].to_vec())
}

fn cold(v: &Variant, constraints: i64, args: &Args) -> Result<Vec<Row>> {
    // The verifying key is not part of proving, so it is loaded once and stays out of
    // every timed region in both modes.
    let vk = VerifyingKey::from_json(&v.vkey())
        .with_context(|| format!("{}: loading vkey.json", v.name))?;
    let mut rng = ark_std::rand::thread_rng();
    let mut rows = Vec::with_capacity(args.reps);

    for rep in 1..=args.reps {
        let mut t = StageTimings::default();
        let start = Instant::now();

        let pk = ProvingKey::load(&v.zkey())
            .with_context(|| format!("{}: loading circuit.zkey", v.name))?;
        let n_public = pk.n_public;
        let witness = Witness::load(&v.wtns())
            .with_context(|| format!("{}: loading circuit.wtns", v.name))?
            .0;
        let circuit = crate::make_backend(args.backend)?.prepare(pk)?;
        let prepare_ms = ms(start);

        let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t)?;
        let total_ms = ms(start);

        // Verification is outside the timed region but before the row is recorded: a
        // benchmark that times a broken prover is worse than no benchmark.
        verify(&vk, &public_of(&witness, n_public)?, &proof).with_context(|| {
            format!(
                "{} cold rep {rep}: our own verifier rejected the proof",
                v.name
            )
        })?;

        rows.push(Row {
            variant: v.name.clone(),
            constraints,
            backend: args.backend.as_str(),
            mode: "cold",
            rep,
            ms: total_ms,
            prepare_ms,
            timings: t,
            verified: true,
        });
        // Explicit, because the whole meaning of "cold" is that nothing survives a rep.
        drop(circuit);
    }
    Ok(rows)
}

fn warm(v: &Variant, constraints: i64, args: &Args) -> Result<Vec<Row>> {
    let vk = VerifyingKey::from_json(&v.vkey())
        .with_context(|| format!("{}: loading vkey.json", v.name))?;
    let mut rng = ark_std::rand::thread_rng();

    let start = Instant::now();
    let pk =
        ProvingKey::load(&v.zkey()).with_context(|| format!("{}: loading circuit.zkey", v.name))?;
    let n_public = pk.n_public;
    let witness = Witness::load(&v.wtns())
        .with_context(|| format!("{}: loading circuit.wtns", v.name))?
        .0;
    let circuit = crate::make_backend(args.backend)?.prepare(pk)?;
    let prepare_ms = ms(start);
    let public = public_of(&witness, n_public)?;

    let mut rows = Vec::with_capacity(args.reps);
    for rep in 1..=args.reps {
        let mut t = StageTimings::default();
        let start = Instant::now();
        let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t)?;
        let prove_ms = ms(start);

        verify(&vk, &public, &proof).with_context(|| {
            format!(
                "{} warm rep {rep}: our own verifier rejected the proof",
                v.name
            )
        })?;

        rows.push(Row {
            variant: v.name.clone(),
            constraints,
            backend: args.backend.as_str(),
            mode: "warm",
            rep,
            ms: prove_ms,
            prepare_ms,
            timings: t,
            verified: true,
        });
    }
    Ok(rows)
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// Median the way `statistics.median` does it, so our summary lines and
/// `run-comparison.py`'s agree on the same data.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("timings are never NaN"));
    let mid = s.len() / 2;
    if s.len() % 2 == 1 {
        s[mid]
    } else {
        (s[mid - 1] + s[mid]) / 2.0
    }
}

fn report(rows: &[Row]) {
    let Some(first) = rows.first() else { return };
    let ms: Vec<f64> = rows.iter().map(|r| r.ms).collect();
    let lo = ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  {:6} {:5}  median {:8.1} ms (min {:.1} max {:.1})  prepare {:.1} ms  n={}",
        first.backend,
        first.mode,
        median(&ms),
        lo,
        hi,
        median(&rows.iter().map(|r| r.prepare_ms).collect::<Vec<_>>()),
        rows.len(),
    );
    let stage = |f: fn(&StageTimings) -> u64| {
        median(
            &rows
                .iter()
                .map(|r| f(&r.timings) as f64)
                .collect::<Vec<_>>(),
        ) / 1000.0
    };
    println!(
        "         stages (median ms): gather {:.2}  ntt {:.2}  pointwise {:.2}  msm {:.2}  assemble {:.2}",
        stage(|t| t.gather_us),
        stage(|t| t.ntt_us),
        stage(|t| t.pointwise_us),
        stage(|t| t.msm_us),
        stage(|t| t.assemble_us),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_matches_pythons_definition() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[7.0]), 7.0);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn csv_row_has_one_field_per_header_column() {
        let host = HostInfo::detect();
        let row = Row {
            variant: "v".into(),
            constraints: 42,
            backend: "cpu",
            mode: "warm",
            rep: 3,
            ms: 1.5,
            prepare_ms: 2.25,
            timings: StageTimings {
                gather_us: 1,
                ntt_us: 2,
                pointwise_us: 3,
                msm_us: 4,
                assemble_us: 5,
            },
            verified: true,
        };
        let line = row.to_csv(&host);
        assert_eq!(
            line.split(',').count(),
            CSV_HEADER.split(',').count(),
            "{line}"
        );
        // The stage split has to survive into the CSV, otherwise `bench` reports a total
        // with no way to see which stage owns it.
        assert!(line.ends_with(",1,2,3,4,5,yes"), "{line}");
        assert!(line.contains(",ours,cpu,warm,3,1.500,2.250,"), "{line}");
    }
}
