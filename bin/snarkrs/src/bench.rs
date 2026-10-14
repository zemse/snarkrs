//! `snarkrs bench`: cold and warm proving timings, with every proof verified first.
//!
//! Each mode means exactly one thing here, and its name is what the CSV `mode` column says,
//! so a row never has to be read against the command line that produced it:
//!
//! * **cold-inprocess** (`--mode cold` is the old name for it) - every rep pays the full
//!   cost inside this process. The zkey parse, the witness parse, backend construction and
//!   `PreparedCircuit::prepare` all sit inside the timed region and everything is dropped
//!   afterwards, so no rep can inherit a warm cache from the one before it. This is what a
//!   worker that re-prepares per job does, and on a GPU it is where the key upload shows up.
//! * **cold-process** - one fresh `snarkrs groth16 prove` process per rep, timed around the
//!   whole invocation: process start, dynamic linking, key parse, prove, JSON write, exit.
//!   This is what a CLI prover costs, and what round 1 and `run-comparison.py` call cold.
//!   `bench/results/README.md` measured it about 200 ms above cold-inprocess at 140,261
//!   constraints on both cpu and cuda, which is process startup.
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

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use snarkrs_field::{Fr, ADX_BMI2};
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::{prove::prove_unchecked, verify::verify, MsmSplit, StageTimings};

use crate::artifacts::{self, Variant};
use crate::{BackendKind, HostInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Parse, prepare and prove inside this process, per rep.
    #[value(name = "cold-inprocess", alias = "cold")]
    ColdInprocess,
    /// One fresh `snarkrs groth16 prove` process per rep.
    #[value(name = "cold-process")]
    ColdProcess,
    Warm,
    /// cold-inprocess and warm.
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
    /// Time `prove --constant-work` instead: fixed MSM sizing without zero/one fast
    /// paths. cpu, metal, wgpu and cuda backends; not a constant-time guarantee.
    #[arg(long)]
    pub constant_work: bool,
    /// Append-free CSV output. Without it the numbers are only printed.
    #[arg(long, value_name = "FILE")]
    pub csv: Option<PathBuf>,
    /// Also have this snarkjs binary verify the first proof of every variant in every mode,
    /// outside the timed region. Our verifier checks every rep, but it shares the field,
    /// curve and pairing code with the prover; snarkjs shares none of it.
    #[arg(long, value_name = "PATH")]
    pub snarkjs: Option<PathBuf>,
}

pub const CSV_HEADER: &str = "host,os,arch,cores,variant,constraints,prover,backend,mode,rep,ms,prepare_ms,gather_us,ntt_us,pointwise_us,msm_us,assemble_us,verified,adx_bmi2,msm_a_us,msm_b_g2_us,msm_b_g1_us,msm_l_us,msm_h_us";

/// One proof and what it cost. `prepare_ms` is setup only, and for a cold-inprocess rep it
/// is the part of `ms` spent before `prove()` was entered. A cold-process rep sees only
/// the child's wall clock, so it has neither, and its CSV cells are empty rather than zero.
pub struct Row {
    pub variant: String,
    pub constraints: i64,
    pub backend: &'static str,
    pub mode: &'static str,
    pub rep: usize,
    pub ms: f64,
    pub prepare_ms: Option<f64>,
    pub timings: Option<StageTimings>,
}

impl Row {
    /// The `verified` column is always `yes`: a rep whose proof the verifier rejects
    /// leaves through `?` and is never recorded, and `bench/scripts/bench_external.py`
    /// refuses the same way. The column is schema parity with that writer.
    fn to_csv(&self, host: &HostInfo) -> String {
        let prepare = self.prepare_ms.map_or(String::new(), |p| format!("{p:.3}"));
        let stages = self.timings.map_or(",,,,".to_string(), |t| {
            format!(
                "{},{},{},{},{}",
                t.gather_us, t.ntt_us, t.pointwise_us, t.msm_us, t.assemble_us
            )
        });
        // Empty when the backend could not time the five MSMs apart, see `MsmSplit`.
        let split = self
            .timings
            .and_then(|t| t.msm_split)
            .map_or(",,,,".to_string(), |m| {
                format!(
                    "{},{},{},{},{}",
                    m.a_us, m.b_g2_us, m.b_g1_us, m.l_us, m.h_us
                )
            });
        format!(
            "{},{},{},{},{},{},ours,{},{},{},{:.3},{},{},yes,{},{}",
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
            prepare,
            stages,
            if ADX_BMI2 { "yes" } else { "no" },
            split,
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
        Mode::ColdInprocess => &[Mode::ColdInprocess],
        Mode::ColdProcess => &[Mode::ColdProcess],
        Mode::Warm => &[Mode::Warm],
        Mode::Both => &[Mode::ColdInprocess, Mode::Warm],
    };

    // Which x86 build this is, so a baseline build is never read as an adx+bmi2 one.
    println!("adx+bmi2 build: {}", if ADX_BMI2 { "yes" } else { "no" });
    let mut rows = Vec::new();
    for v in &variants {
        let constraints = v.constraints();
        let work = if args.constant_work {
            ", constant work"
        } else {
            ""
        };
        println!("\n=== {} ({constraints} constraints{work}) ===", v.name);
        for mode in modes {
            let got = match mode {
                Mode::ColdInprocess => cold_inprocess(v, constraints, &args)?,
                Mode::ColdProcess => cold_process(v, constraints, &args)?,
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

fn backend(args: &Args) -> Result<Box<dyn snarkrs_groth16::Backend>> {
    if args.constant_work {
        crate::make_constant_work_backend(args.backend)
    } else {
        crate::make_backend(args.backend)
    }
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

/// `snarkjs groth16 verify` on one proof, when `--snarkjs` names a binary. BUG-22: the only
/// check that catches a defect our prover and our verifier share.
fn foreign_verify(
    args: &Args,
    v: &Variant,
    proof: &snarkrs_groth16::Proof,
    public: &[Fr],
    mode: &str,
) -> Result<()> {
    let Some(bin) = &args.snarkjs else {
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("g16-bench-snarkjs-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (p, q) = (dir.join("proof.json"), dir.join("public.json"));
    crate::json::write_proof(&p, proof)?;
    crate::json::write_public(&q, public)?;
    let out = std::process::Command::new(bin)
        .args(["groth16", "verify"])
        .arg(v.vkey())
        .arg(&q)
        .arg(&p)
        .output()
        .with_context(|| format!("running {}", bin.display()))?;
    std::fs::remove_dir_all(&dir).ok();
    check_foreign_verify_output(&out).with_context(|| {
        format!(
            "{} {mode}: our verifier accepted the {} proof and snarkjs rejected it",
            v.name,
            args.backend.as_str()
        )
    })?;
    println!("  snarkjs accepted the {mode} proof");
    Ok(())
}

fn check_foreign_verify_output(out: &std::process::Output) -> Result<()> {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() || !stdout.contains("OK!") {
        bail!(
            "verifier status: {}; stdout: {stdout}; stderr: {stderr}",
            out.status
        );
    }
    Ok(())
}

fn cold_inprocess(v: &Variant, constraints: i64, args: &Args) -> Result<Vec<Row>> {
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
        // Before the prepare, so a witness too short for the key fails here rather than
        // after a full proof. The copy is `n_public` field elements.
        let public = public_of(&witness, n_public)?;
        let circuit = backend(args)?.prepare(pk)?;
        let prepare_ms = ms(start);

        let proof = prove_unchecked(circuit.as_ref(), &witness, &mut rng, &mut t)?;
        let total_ms = ms(start);

        // Verification is outside the timed region but before the row is recorded: a
        // benchmark that times a broken prover is worse than no benchmark. That is also why
        // this is `prove_unchecked`: the same check inside the timed region would bill it
        // to the prover, and rapidsnark and snarkjs, the columns beside ours, run none.
        verify(&vk, &public, &proof).with_context(|| {
            format!(
                "{} cold-inprocess rep {rep}: our own verifier rejected the proof",
                v.name
            )
        })?;
        if rep == 1 {
            foreign_verify(args, v, &proof, &public, "cold-inprocess")?;
        }

        rows.push(Row {
            variant: v.name.clone(),
            constraints,
            backend: args.backend.as_str(),
            mode: "cold-inprocess",
            rep,
            ms: total_ms,
            prepare_ms: Some(prepare_ms),
            timings: Some(t),
        });
        // Explicit, because the whole meaning of cold is that nothing survives a rep.
        drop(circuit);
    }
    Ok(rows)
}

fn cold_process(v: &Variant, constraints: i64, args: &Args) -> Result<Vec<Row>> {
    let vk = VerifyingKey::from_json(&v.vkey())
        .with_context(|| format!("{}: loading vkey.json", v.name))?;
    // The expected public signals come from the witness, read once outside the timed
    // region: the child has to prove this witness, not merely produce a valid proof.
    let witness = Witness::load(&v.wtns())
        .with_context(|| format!("{}: loading circuit.wtns", v.name))?
        .0;
    let public = public_of(&witness, vk.ic.len().saturating_sub(1))?;
    let exe = std::env::current_exe().context("locating the snarkrs binary")?;
    let dir = std::env::temp_dir().join(format!("g16-bench-process-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (proof_path, public_path) = (dir.join("proof.json"), dir.join("public.json"));

    let mut rows = Vec::with_capacity(args.reps);
    for rep in 1..=args.reps {
        for f in [&proof_path, &public_path] {
            std::fs::remove_file(f).ok();
        }
        let start = Instant::now();
        let out = prove_command(&exe, v, args, &proof_path, &public_path)
            .output()
            .with_context(|| format!("running {}", exe.display()))?;
        let total_ms = ms(start);
        if !out.status.success() {
            bail!(
                "{} cold-process rep {rep}: {} exited {}; stderr: {}",
                v.name,
                exe.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }

        let proof = crate::json::read_proof(&proof_path)
            .with_context(|| format!("{} cold-process rep {rep}: reading proof.json", v.name))?;
        let written = crate::json::read_public(&public_path)
            .with_context(|| format!("{} cold-process rep {rep}: reading public.json", v.name))?;
        if written != public {
            bail!(
                "{} cold-process rep {rep}: public.json differs from the witness",
                v.name
            );
        }
        verify(&vk, &public, &proof).with_context(|| {
            format!(
                "{} cold-process rep {rep}: our own verifier rejected the proof",
                v.name
            )
        })?;
        if rep == 1 {
            foreign_verify(args, v, &proof, &public, "cold-process")?;
        }

        rows.push(Row {
            variant: v.name.clone(),
            constraints,
            backend: args.backend.as_str(),
            mode: "cold-process",
            rep,
            ms: total_ms,
            prepare_ms: None,
            timings: None,
        });
    }
    std::fs::remove_dir_all(&dir).ok();
    Ok(rows)
}

/// The child a cold-process rep times. Self-verify is off for the same reason the
/// in-process modes call `prove_unchecked`: the check runs here, outside the timed region.
fn prove_command(
    exe: &Path,
    v: &Variant,
    args: &Args,
    proof: &Path,
    public: &Path,
) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["groth16", "prove"])
        .arg(v.zkey())
        .arg(v.wtns())
        .arg(proof)
        .arg(public)
        .args(["--backend", args.backend.as_str(), "--self-verify", "false"]);
    if args.constant_work {
        cmd.arg("--constant-work");
    }
    cmd
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
    let circuit = backend(args)?.prepare(pk)?;
    let prepare_ms = ms(start);
    let public = public_of(&witness, n_public)?;

    let mut rows = Vec::with_capacity(args.reps);
    for rep in 1..=args.reps {
        let mut t = StageTimings::default();
        let start = Instant::now();
        let proof = prove_unchecked(circuit.as_ref(), &witness, &mut rng, &mut t)?;
        let prove_ms = ms(start);

        verify(&vk, &public, &proof).with_context(|| {
            format!(
                "{} warm rep {rep}: our own verifier rejected the proof",
                v.name
            )
        })?;
        if rep == 1 {
            foreign_verify(args, v, &proof, &public, "warm")?;
        }

        rows.push(Row {
            variant: v.name.clone(),
            constraints,
            backend: args.backend.as_str(),
            mode: "warm",
            rep,
            ms: prove_ms,
            prepare_ms: Some(prepare_ms),
            timings: Some(t),
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
    // One call reports one mode, so either every row has the split or none does.
    let prepare: Vec<f64> = rows.iter().filter_map(|r| r.prepare_ms).collect();
    let prepare = if prepare.is_empty() {
        String::new()
    } else {
        format!("  prepare {:.1} ms", median(&prepare))
    };
    println!(
        "  {:6} {:14}  median {:8.1} ms (min {:.1} max {:.1}){prepare}  n={}",
        first.backend,
        first.mode,
        median(&ms),
        lo,
        hi,
        rows.len(),
    );
    let timings: Vec<StageTimings> = rows.iter().filter_map(|r| r.timings).collect();
    if timings.is_empty() {
        return;
    }
    let stage = |f: fn(&StageTimings) -> u64| {
        median(&timings.iter().map(|t| f(t) as f64).collect::<Vec<_>>()) / 1000.0
    };
    println!(
        "         stages (median ms): gather {:.2}  ntt {:.2}  pointwise {:.2}  msm {:.2}  assemble {:.2}",
        stage(|t| t.gather_us),
        stage(|t| t.ntt_us),
        stage(|t| t.pointwise_us),
        stage(|t| t.msm_us),
        stage(|t| t.assemble_us),
    );
    let splits: Vec<_> = timings.iter().filter_map(|t| t.msm_split).collect();
    if splits.is_empty() {
        return;
    }
    let msm = |f: fn(&MsmSplit) -> u64| {
        median(&splits.iter().map(|m| f(m) as f64).collect::<Vec<_>>()) / 1000.0
    };
    println!(
        "         msms (median ms, overlapping): a {:.2}  b_g2 {:.2}  b_g1 {:.2}  l {:.2}  h {:.2}",
        msm(|m| m.a_us),
        msm(|m| m.b_g2_us),
        msm(|m| m.b_g1_us),
        msm(|m| m.l_us),
        msm(|m| m.h_us),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(unix, windows))]
    fn verifier_output(code: u32, stdout: &str, stderr: &str) -> std::process::Output {
        #[cfg(unix)]
        let status = {
            use std::os::unix::process::ExitStatusExt;
            std::process::ExitStatus::from_raw((code as i32) << 8)
        };
        #[cfg(windows)]
        let status = {
            use std::os::windows::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code)
        };
        std::process::Output {
            status,
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn foreign_verify_rejects_nonzero_exit_even_with_ok() {
        let out = verifier_output(7, "OK! accepted", "verifier failed afterwards");
        let error = check_foreign_verify_output(&out).unwrap_err().to_string();
        assert!(error.contains(&out.status.to_string()), "{error}");
        assert!(error.contains("stdout: OK! accepted"), "{error}");
        assert!(
            error.contains("stderr: verifier failed afterwards"),
            "{error}"
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn foreign_verify_rejects_zero_exit_without_ok() {
        let out = verifier_output(0, "no valid-proof marker", "OK! on stderr only");
        let error = check_foreign_verify_output(&out).unwrap_err().to_string();
        assert!(error.contains(&out.status.to_string()), "{error}");
        assert!(error.contains("stdout: no valid-proof marker"), "{error}");
        assert!(error.contains("stderr: OK! on stderr only"), "{error}");
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn foreign_verify_accepts_zero_exit_with_ok() {
        let out = verifier_output(0, "snarkjs: OK!", "verifier diagnostics");
        check_foreign_verify_output(&out).unwrap();
    }

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
            prepare_ms: Some(2.25),
            timings: Some(StageTimings {
                gather_us: 1,
                ntt_us: 2,
                pointwise_us: 3,
                msm_us: 4,
                assemble_us: 5,
                verify_us: 0,
                msm_split: Some(MsmSplit {
                    a_us: 6,
                    b_g2_us: 7,
                    b_g1_us: 8,
                    l_us: 9,
                    h_us: 10,
                }),
            }),
        };
        let line = row.to_csv(&host);
        assert_eq!(
            line.split(',').count(),
            CSV_HEADER.split(',').count(),
            "{line}"
        );
        // The stage split has to survive into the CSV, otherwise `bench` reports a total
        // with no way to see which stage owns it.
        assert!(line.contains(",1,2,3,4,5,yes,"), "{line}");
        let adx = if ADX_BMI2 { "yes" } else { "no" };
        assert!(line.ends_with(&format!(",{adx},6,7,8,9,10")), "{line}");
        assert!(line.contains(",ours,cpu,warm,3,1.500,2.250,"), "{line}");
    }

    #[test]
    fn cold_process_row_leaves_what_it_cannot_see_empty() {
        let host = HostInfo::detect();
        let row = Row {
            variant: "v".into(),
            constraints: 42,
            backend: "cpu",
            mode: "cold-process",
            rep: 1,
            ms: 1.5,
            prepare_ms: None,
            timings: None,
        };
        let line = row.to_csv(&host);
        assert_eq!(
            line.split(',').count(),
            CSV_HEADER.split(',').count(),
            "{line}"
        );
        assert!(
            line.contains(",ours,cpu,cold-process,1,1.500,,,,,,,yes,"),
            "{line}"
        );
        assert!(line.ends_with(",,,,,"), "{line}");
    }

    #[test]
    fn cold_is_still_the_in_process_mode() {
        use clap::ValueEnum;
        assert_eq!(Mode::from_str("cold", false), Ok(Mode::ColdInprocess));
        assert_eq!(
            Mode::from_str("cold-inprocess", false),
            Ok(Mode::ColdInprocess)
        );
        assert_eq!(Mode::from_str("cold-process", false), Ok(Mode::ColdProcess));
    }
}
