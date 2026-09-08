//! `g16` - prove, verify, benchmark, and run the ceremony that produces a key.
//!
//!   g16 prove  --zkey c.zkey --witness c.wtns --proof p.json --public pub.json
//!              [--backend cpu|wgpu|metal|cuda] [--stage-timings]
//!   g16 verify --vkey vkey.json --proof p.json --public pub.json
//!   g16 trace  --zkey c.zkey --witness c.wtns [--backend cpu|wgpu|...] [--out t.txt]
//!   g16 bench  --artifacts DIR [--variant NAME]... [--reps 15] [--backend cpu|wgpu|...]
//!              [--mode cold|warm|both] [--csv out.csv]
//!
//!   g16 setup  --r1cs c.r1cs --ptau prepared.ptau --out c_0000.zkey
//!
//!   g16 ptau info       --ptau p.ptau
//!   g16 ptau new        --power 12 --out p_0000.ptau
//!   g16 ptau contribute --ptau p_0000.ptau --out p_0001.ptau --entropy STRING [--name S]
//!   g16 ptau beacon     --ptau p_0001.ptau --out p_final.ptau
//!                       --beacon-hash HEX --num-iterations-exp N [--name S]
//!   g16 ptau prepare    --ptau p_final.ptau --out prepared.ptau
//!   g16 ptau verify     --ptau p.ptau
//!
//!   g16 zkey contribute               --zkey c_0000.zkey --out c_0001.zkey
//!                                     --entropy STRING [--name S]
//!   g16 zkey beacon                   --zkey c_0001.zkey --out c_final.zkey
//!                                     --beacon-hash HEX --num-iterations-exp N [--name S]
//!   g16 zkey verify                   --zkey c_final.zkey --ptau prepared.ptau
//!                                     (--r1cs c.r1cs | --init c_0000.zkey)
//!   g16 zkey export-verificationkey   --zkey c_final.zkey --out vkey.json
//!
//! `prove` writes snarkjs' `proof.json` and `public.json` verbatim, so the output is
//! checkable by `snarkjs groth16 verify` and by rapidsnark's verifier, not only by ours.
//! Cold and warm are `bench`'s whole reason to exist: see `bench.rs` for what each one
//! puts inside the timed region and why reporting only one of them misleads.
//!
//! The ceremony commands are the same trade in the other direction: they write files
//! snarkjs itself has to accept, so their formats are not ours to choose. `g16-ceremony`
//! carries the byte-level reasoning and the snarkjs line numbers; this file only parses
//! arguments and prints what came back.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use g16_ceremony::{contribute as zkey_mpc, phase1, prepare, ptau as ptau_file, setup, vkey};
use g16_cli::{bench, json, make_backend, BackendKind};
use g16_core::{
    prove::{prove, prove_trace},
    verify::verify,
    StageTimings,
};
use g16_msm::{CpuMsm, MsmBackend};
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
    /// Print a deterministic execution trace, for diffing against another machine.
    ///
    /// Debugging only. It pins stage 10's blinders, so it emits no proof.json: see
    /// `g16_core::trace`.
    Trace {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        witness: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
        /// Write the trace here instead of to stdout, so two of them can be diffed.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
    /// Benchmark proving, cold and warm, verifying every proof.
    Bench(bench::Args),
    /// Groth16 phase-2 setup: circuit + prepared powers of tau -> an initial zkey.
    ///
    /// The key this writes has delta = 1 and is not safe to prove with. It is the input to
    /// `zkey contribute`, which is what gives it a delta nobody knows.
    Setup {
        #[arg(long, value_name = "FILE")]
        r1cs: PathBuf,
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Powers of tau: the circuit-independent phase-1 ceremony.
    Ptau {
        #[command(subcommand)]
        cmd: PtauCmd,
    },
    /// Proving keys: the circuit-specific phase-2 ceremony.
    Zkey {
        #[command(subcommand)]
        cmd: ZkeyCmd,
    },
}

#[derive(Subcommand)]
enum PtauCmd {
    /// Report the header, every section's declared and present size, and the contribution
    /// count.
    ///
    /// Reads leniently, so it describes a truncated download instead of refusing to open
    /// it. That is the only command here that does.
    Info {
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
    },
    /// Write a fresh accumulator at 2^POWER. Fully determined by `--power`: no randomness
    /// and no timestamps, so two runs produce identical bytes.
    New {
        #[arg(long)]
        power: u32,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Add a contribution from your own entropy.
    ///
    /// Non-deterministic by design: the entropy string is mixed with 64 bytes from the OS
    /// CSPRNG, so the same string twice gives two different keys.
    Contribute {
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        #[arg(long, value_name = "STRING")]
        entropy: String,
        /// Recorded in the contribution and printed by every later `verify`.
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// Add the final contribution from a public beacon, the one anyone can reproduce.
    Beacon {
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        /// Hex, even length, at most 255 bytes.
        #[arg(long, value_name = "HEX")]
        beacon_hash: String,
        /// SHA-256 is iterated 2^N times over the beacon hash. 10 to 63.
        #[arg(long, value_name = "N")]
        num_iterations_exp: String,
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// Precompute the Lagrange sections phase 2 needs. This is the expensive one: every
    /// butterfly of its inverse FFT is a full point scalar multiplication.
    Prepare {
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Check the contribution chain and recompute the challenge hashes.
    Verify {
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
    },
}

#[derive(Subcommand)]
enum ZkeyCmd {
    /// Add a contribution from your own entropy. Sections 3 to 7 are copied verbatim, so
    /// this costs two MSM-free rescalings and nothing else.
    Contribute {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        #[arg(long, value_name = "STRING")]
        entropy: String,
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// Add the final contribution from a public beacon.
    Beacon {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        #[arg(long, value_name = "HEX")]
        beacon_hash: String,
        #[arg(long, value_name = "N")]
        num_iterations_exp: String,
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// Check the contribution chain and that the final key is a consistent rescaling of
    /// the initial one.
    ///
    /// With `--init` that is all it checks: nothing ties the initial key to any circuit.
    /// `--r1cs` re-runs the whole setup to produce the initial key itself, which is the
    /// only form that checks the key against the circuit, and it costs a full setup.
    Verify {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        ptau: PathBuf,
        #[arg(long, value_name = "FILE", conflicts_with = "init")]
        r1cs: Option<PathBuf>,
        #[arg(long, value_name = "FILE", required_unless_present = "r1cs")]
        init: Option<PathBuf>,
    },
    /// Write snarkjs' verification_key.json: the four verifier points, the precomputed
    /// pairing, and IC.
    ExportVerificationkey {
        #[arg(long, value_name = "FILE")]
        zkey: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
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
        Cmd::Trace {
            zkey,
            witness,
            backend,
            out,
        } => run_trace(&zkey, &witness, backend, out.as_deref()),
        Cmd::Bench(args) => bench::run(args),
        Cmd::Setup { r1cs, ptau, out } => run_setup(&r1cs, &ptau, &out),
        Cmd::Ptau { cmd } => run_ptau(cmd),
        Cmd::Zkey { cmd } => run_zkey(cmd),
    }
}

/// The MSM behind `setup` and the two same-ratio checks.
///
/// Only the CPU backend exists as an `MsmBackend` today: `g16-metal` and `g16-cuda`
/// implement `g16_core::Backend`, the whole prover, not this trait, so there is nothing yet
/// to select between and no `--backend` flag that would not be a lie. The parameter is
/// threaded through the library anyway, so putting setup on a GPU is a constructor swap
/// here rather than a change to every ceremony signature.
fn msm_backend() -> impl MsmBackend {
    CpuMsm::new()
}

fn run_setup(r1cs: &std::path::Path, ptau: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let msm = msm_backend();
    let report = setup::setup(r1cs, ptau, out, &msm)
        .with_context(|| format!("setting up {}", r1cs.display()))?;
    println!("constraints    {}", report.n_constraints);
    println!("vars           {}", report.n_vars);
    println!("public         {}", report.n_public);
    println!(
        "domain size    {} (2^{})",
        report.domain_size, report.cir_power
    );
    println!("coefficients   {}", report.n_coefs);
    println!("circuit hash   {}", hex::encode(report.cs_hash));
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn run_ptau(cmd: PtauCmd) -> Result<()> {
    match cmd {
        PtauCmd::Info { ptau } => {
            let file = ptau_file::Ptau::open_lenient(&ptau)
                .with_context(|| format!("opening {}", ptau.display()))?;
            let info = file.info();
            println!("power          {}", info.power);
            println!("ceremony power {}", info.ceremony_power);
            println!("prepared       {}", info.prepared);
            println!("contributions  {}", info.n_contributions);
            println!(
                "sections       {} of {} declared",
                info.sections.len(),
                info.declared_sections
            );
            println!(
                "bytes          {} (chain ends at {})",
                info.file_bytes, info.chain_end
            );
            // Declared, present and expected are printed side by side rather than reduced
            // to a verdict: a length that is short is truncation, a length that disagrees
            // with `power` is corruption, and the two need different responses.
            println!("  id     declared      present     expected");
            for s in &info.sections {
                let expected = match s.expected {
                    Some(e) => e.to_string(),
                    None => "-".to_string(),
                };
                println!(
                    "  {:>3} {:>12} {:>12} {:>12}",
                    s.id, s.declared, s.present, expected
                );
            }
            if let Some(t) = info.truncation {
                println!(
                    "TRUNCATED: section {:?} declares {} bytes, {} present",
                    t.section, t.declared, t.present
                );
            }
            Ok(())
        }
        PtauCmd::New { power, out } => {
            let challenge = phase1::ptau_new(power, &out)
                .with_context(|| format!("writing {}", out.display()))?;
            println!("first challenge {}", hex::encode(challenge));
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        PtauCmd::Contribute {
            ptau,
            out,
            entropy,
            name,
        } => {
            let report = phase1::contribute(&ptau, &out, name.as_deref(), &entropy)
                .with_context(|| format!("contributing to {}", ptau.display()))?;
            print_phase1(&report);
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        PtauCmd::Beacon {
            ptau,
            out,
            beacon_hash,
            num_iterations_exp,
            name,
        } => {
            let (hash, exp) = phase1::parse_beacon_args(&beacon_hash, &num_iterations_exp)?;
            let report = phase1::beacon(&ptau, &out, name.as_deref(), &hash, exp)
                .with_context(|| format!("beaconing {}", ptau.display()))?;
            print_phase1(&report);
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        PtauCmd::Prepare { ptau, out } => {
            prepare::prepare_phase2(&ptau, &out)
                .with_context(|| format!("preparing {}", ptau.display()))?;
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        PtauCmd::Verify { ptau } => {
            let report =
                phase1::verify(&ptau).with_context(|| format!("verifying {}", ptau.display()))?;
            println!("power          {}", report.power);
            println!("ceremony power {}", report.ceremony_power);
            println!("prepared       {}", report.prepared);
            for (i, h) in report.contribution_hashes.iter().enumerate() {
                println!("contribution {i:>3} {}", hex::encode(h));
            }
            // Saying which check was skipped is the point: a truncated file cannot have
            // its final next-challenge compared, and reporting "OK" without that caveat
            // overstates what was verified.
            if !report.next_challenge_checked {
                println!("next challenge NOT checked: this file is truncated");
            }
            println!("OK");
            Ok(())
        }
    }
}

fn print_phase1(report: &phase1::Phase1Report) {
    println!("contribution   {}", report.index);
    println!("response hash  {}", hex::encode(report.response_hash));
    println!("next challenge {}", hex::encode(report.next_challenge));
}

fn run_zkey(cmd: ZkeyCmd) -> Result<()> {
    match cmd {
        ZkeyCmd::Contribute {
            zkey,
            out,
            entropy,
            name,
        } => {
            let msm = msm_backend();
            let report = zkey_mpc::contribute(&zkey, &out, name.as_deref(), &entropy, &msm)
                .with_context(|| format!("contributing to {}", zkey.display()))?;
            println!("contribution   {}", report.index);
            println!("hash           {}", hex::encode(report.hash));
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        ZkeyCmd::Beacon {
            zkey,
            out,
            beacon_hash,
            num_iterations_exp,
            name,
        } => {
            let (hash, exp) = phase1::parse_beacon_args(&beacon_hash, &num_iterations_exp)?;
            let msm = msm_backend();
            let report = zkey_mpc::beacon(&zkey, &out, name.as_deref(), &hash, exp, &msm)
                .with_context(|| format!("beaconing {}", zkey.display()))?;
            println!("contribution   {}", report.index);
            println!("hash           {}", hex::encode(report.hash));
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        ZkeyCmd::Verify {
            zkey,
            ptau,
            r1cs,
            init,
        } => {
            let msm = msm_backend();
            let report = match (&r1cs, &init) {
                (Some(r1cs), _) => zkey_mpc::verify_from_r1cs(r1cs, &ptau, &zkey, &msm),
                (None, Some(init)) => zkey_mpc::verify_from_init(init, &ptau, &zkey, &msm),
                // clap's `required_unless_present` already rules this out; the arm exists
                // so the match is total rather than a panic waiting for a flag change.
                (None, None) => anyhow::bail!("one of --r1cs or --init is required"),
            }
            .with_context(|| format!("verifying {}", zkey.display()))?;
            println!("vars           {}", report.n_vars);
            println!("public         {}", report.n_public);
            println!("domain size    {}", report.domain_size);
            for (i, h) in report.contribution_hashes.iter().enumerate() {
                println!("contribution {i:>3} {}", hex::encode(h));
            }
            if r1cs.is_none() {
                println!("note: --init checks the chain, not that the key matches a circuit");
            }
            println!("OK");
            Ok(())
        }
        ZkeyCmd::ExportVerificationkey { zkey, out } => {
            vkey::export_verification_key(&zkey, &out)
                .with_context(|| format!("exporting from {}", zkey.display()))?;
            eprintln!("wrote {}", out.display());
            Ok(())
        }
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

    // Stage 10's blinders come from the OS CSPRNG. Nothing on this command offers a seed
    // override: a reused (r, s) across two proofs of different witnesses leaks the witness.
    // `g16 trace` does fix them, and writes no proof.json for exactly that reason.
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

/// Prints what two machines must agree on, given the same zkey and witness.
///
/// No `--self-verify` and no `--proof`: this is not a way to obtain a proof. The blinders are
/// fixed constants, so the proof it computes is one nobody may publish, and the only reason
/// stage 11 runs at all is that a difference which first shows up in the assembled points and
/// not in any MSM would say the fault is in `assemble` rather than on the device.
fn run_trace(
    zkey: &std::path::Path,
    witness: &std::path::Path,
    backend: BackendKind,
    out: Option<&std::path::Path>,
) -> Result<()> {
    let pk = ProvingKey::load(zkey).with_context(|| format!("loading {}", zkey.display()))?;
    let w = Witness::load(witness)
        .with_context(|| format!("loading {}", witness.display()))?
        .0;
    let circuit = make_backend(backend)?.prepare(pk)?;
    let mut t = StageTimings::default();
    let trace = prove_trace(circuit.as_ref(), &w, &mut t)?;

    match out {
        Some(path) => {
            std::fs::write(path, trace.to_text())
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => print!("{}", trace.to_text()),
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
