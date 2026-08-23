//! `g16` - prove, verify and benchmark.
//!
//!   g16 prove  --zkey c.zkey --witness c.wtns --proof p.json --public pub.json
//!              [--backend cpu|wgpu|metal|cuda] [--stage-timings]
//!   g16 verify --vkey vkey.json --proof p.json --public pub.json
//!   g16 bench  --artifacts DIR [--variant NAME]... [--reps 15] [--backend cpu|wgpu|...]
//!              [--mode cold|warm|both] [--csv out.csv]
//!
//! `prove` writes snarkjs' `proof.json` and `public.json` verbatim, so the output is
//! checkable by `snarkjs groth16 verify` and by rapidsnark's verifier, not only by ours.
//! Cold and warm are `bench`'s whole reason to exist: see `bench.rs` for what each one
//! puts inside the timed region and why reporting only one of them misleads.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use g16_cli::{bench, json, make_backend, BackendKind};
use g16_core::{prove::prove, verify::verify, StageTimings};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

#[derive(Parser)]
#[command(
    name = "g16",
    version,
    about = "Groth16 prover for BN254, snarkjs-compatible"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Prove: zkey + witness -> proof.json and public.json.
    Prove {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        witness: PathBuf,
        #[arg(long, value_name = "FILE")]
        proof: PathBuf,
        #[arg(long, value_name = "FILE")]
        public: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
        /// Print the per-stage split of the proving time.
        #[arg(long)]
        stage_timings: bool,
        /// Verify the proof before writing it, and fail rather than emit one that does not
        /// verify. On by default; the cost is under 3% of a proof.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        self_verify: bool,
    },
    /// Verify a proof against snarkjs' verification_key.json.
    Verify {
        #[arg(long, value_name = "FILE")]
        vkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        proof: PathBuf,
        #[arg(long, value_name = "FILE")]
        public: PathBuf,
    },
    /// Benchmark proving, cold and warm, verifying every proof.
    Bench(bench::Args),
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Prove {
            zkey,
            witness,
            proof,
            public,
            backend,
            stage_timings,
            self_verify,
        } => run_prove(
            &zkey,
            &witness,
            &proof,
            &public,
            backend,
            stage_timings,
            self_verify,
        ),
        Cmd::Verify {
            vkey,
            proof,
            public,
        } => run_verify(&vkey, &proof, &public),
        Cmd::Bench(args) => bench::run(args),
    }
}

fn run_prove(
    zkey: &std::path::Path,
    witness: &std::path::Path,
    proof_out: &std::path::Path,
    public_out: &std::path::Path,
    backend: BackendKind,
    stage_timings: bool,
    self_verify: bool,
) -> Result<()> {
    let pk = ProvingKey::load(zkey).with_context(|| format!("loading {}", zkey.display()))?;
    let n_public = pk.n_public;
    let w = Witness::load(witness)
        .with_context(|| format!("loading {}", witness.display()))?
        .0;
    let circuit = make_backend(backend)?.prepare(pk)?;

    // Stage 10's blinders come from the OS CSPRNG. Nothing in this binary offers a seed
    // override: a reused (r, s) across two proofs of different witnesses leaks the
    // witness, so the deterministic path stays a test-only entry point in g16-core.
    let mut rng = ark_std::rand::thread_rng();
    let mut t = StageTimings::default();
    let started = std::time::Instant::now();
    let proof = prove(circuit.as_ref(), &w, &mut rng, &mut t)?;
    let elapsed = started.elapsed();

    // The public signals are the witness prefix, which is what snarkjs publishes. Taken
    // from the witness rather than copied from an existing public.json so that `prove`
    // needs nothing but the zkey and the witness.
    anyhow::ensure!(
        w.len() > n_public,
        "witness has {} entries, too short for {n_public} public signals",
        w.len()
    );
    // Verify before writing, not after, so a bad proof never reaches the filesystem where
    // something downstream might pick it up. `bench` has verified every timed rep since it
    // was written, on the grounds that timing a broken prover is worse than not timing one;
    // `prove` is the command people actually ship with and it was the one path that skipped
    // the check. A dropped GPU kernel error or a stale pooled buffer fails exactly here.
    let public = &w[1..=n_public];
    if self_verify {
        let vk = &circuit.key().vk;
        match verify(vk, public, &proof) {
            Ok(()) => {}
            Err(e) => anyhow::bail!(
                "the proof this run produced does not verify against the key it was proved \
                 with ({e}). Nothing has been written. This is a bug in the prover or a sign \
                 of a faulty accelerator, not a bad witness: an inconsistent witness yields a \
                 proof that fails verification elsewhere, not one that fails against its own \
                 key. Re-run with --self-verify=false to write it anyway."
            ),
        }
    }

    json::write_proof(proof_out, &proof)?;
    json::write_public(public_out, public)?;

    if stage_timings {
        let total = elapsed.as_micros() as u64;
        let known = t.gather_us + t.ntt_us + t.pointwise_us + t.msm_us + t.assemble_us;
        println!("backend        {}", circuit.backend_name());
        println!("domain size    {}", circuit.domain_size());
        println!("gather      us {:>10}", t.gather_us);
        println!("ntt         us {:>10}", t.ntt_us);
        println!("pointwise   us {:>10}", t.pointwise_us);
        println!("msm         us {:>10}", t.msm_us);
        println!("assemble    us {:>10}", t.assemble_us);
        // Attributed and total are printed separately rather than one being derived from
        // the other, so any stage that forgets to report shows up as a gap.
        println!("attributed  us {:>10}", known);
        println!("total       us {:>10}", total);
        println!("proved in {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    }
    Ok(())
}

fn run_verify(
    vkey: &std::path::Path,
    proof: &std::path::Path,
    public: &std::path::Path,
) -> Result<()> {
    let vk =
        VerifyingKey::from_json(vkey).with_context(|| format!("loading {}", vkey.display()))?;
    let public = json::read_public(public)?;
    let proof = json::read_proof(proof)?;
    verify(&vk, &public, &proof)?;
    println!("OK");
    Ok(())
}
