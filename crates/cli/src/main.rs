//! `snarkrs` - a drop-in for the snarkjs 0.7.6 command line, Groth16 on BN254.
//!
//!   snarkrs powersoftau new bn128 <power> [powersoftau_0000.ptau]         (ptn)
//!   snarkrs powersoftau contribute <in.ptau> <out.ptau> [-e=TEXT] [-n=NAME] (ptc)
//!   snarkrs powersoftau beacon <in.ptau> <out.ptau> <hashHex> <iterExp>    (ptb)
//!   snarkrs powersoftau prepare phase2 <in.ptau> <out.ptau>                (pt2)
//!   snarkrs powersoftau verify <in.ptau>                                   (ptv)
//!   snarkrs powersoftau export challenge <in.ptau> [challenge]            (ptec)
//!   snarkrs powersoftau challenge contribute bn128 <challenge> [response] [-e=TEXT]
//!                                                                          (ptcc)
//!   snarkrs powersoftau import response <old.ptau> <response> <new.ptau> [-n=NAME]
//!                                                  [-nopoints]             (ptir)
//!   snarkrs groth16 setup [circuit.r1cs] [powersoftau.ptau] [circuit_0000.zkey]   (g16s)
//!   snarkrs groth16 prove [circuit_final.zkey] [witness.wtns] [proof.json] [public.json]
//!                                                                          (g16p)
//!   snarkrs groth16 fullprove [input.json] [circuit.wasm] [circuit_final.zkey] [proof.json]
//!                                                  [public.json]           (g16f)
//!   snarkrs groth16 verify [verification_key.json] [public.json] [proof.json]     (g16v)
//!   snarkrs wtns calculate [circuit.wasm] [input.json] [witness.wtns]      (wc)
//!   snarkrs wtns debug [circuit.wasm] [input.json] [witness.wtns] [circuit.sym] [-g] [-s] [-t]
//!                                                                          (wd)
//!   snarkrs zkey contribute <in.zkey> <out.zkey> [-e=TEXT] [-n=NAME]       (zkc)
//!   snarkrs zkey beacon <in.zkey> <out.zkey> <hashHex> <iterExp>           (zkb)
//!   snarkrs zkey verify r1cs [circuit.r1cs] [powersoftau.ptau] [circuit_final.zkey] (zkv)
//!   snarkrs zkey verify init [circuit_0000.zkey] [powersoftau.ptau] [circuit_final.zkey]
//!                                                                          (zkvi)
//!   snarkrs zkey export verificationkey [circuit_final.zkey] [circuit_vk.json]    (zkev)
//!   snarkrs zkey export solidityverifier [circuit_final.zkey] [verifier.sol]     (zkesv)
//!   snarkrs zkey export bellman <in.zkey> [circuit.mpcparams]              (zkeb)
//!   snarkrs zkey bellman contribute bn128 <in.mpcparams> <out.mpcparams> [-e=TEXT] (zkbc)
//!   snarkrs zkey import bellman <old.zkey> <in.mpcparams> <new.zkey> [-n=NAME]    (zkib)
//!   snarkrs file info <file>                                               (fi)
//!
//!   snarkrs bench  --artifacts DIR [--variant NAME]... [--reps 15] [--backend cpu|wgpu|...]
//!                  [--mode cold|warm|both] [--csv out.csv]
//!   snarkrs trace  --zkey c.zkey --witness c.wtns [--backend cpu|wgpu|...] [--out t.txt]
//!   snarkrs fft-bench --power N [--variants NAME,...] [--block-size N,...]  (cuda only)
//!
//! The snarkjs commands take snarkjs' words, aliases, positionals, defaults and options;
//! `cli.rs` turns that line into one clap parses, and has the table. Our own options sit on
//! top as `--` flags: `groth16 prove` and `fullprove` take `--backend cpu|wgpu|metal|cuda`,
//! `--stage-timings`, `--self-verify true|false`, `--vkey vkey.json`, `--fallback` and
//! `--constant-work`, and the ceremony commands take `--backend cpu|metal`.
//!
//! Exit codes are snarkjs': 0 for success, 1 for a failure or a proof or file that does not
//! verify, 99 for a line that names no command or gives it the wrong parameters. The log
//! lines are too (`log.rs`), on stdout, so a script that greps `snarkJS: OK!` keeps working.
//!
//! `groth16 prove` writes snarkjs' `proof.json` and `public.json` verbatim, so the output is
//! checkable by `snarkjs groth16 verify` and by rapidsnark's verifier, not only by ours.
//! Cold and warm are `bench`'s whole reason to exist: see `bench.rs` for what each one
//! puts inside the timed region and why reporting only one of them misleads.
//!
//! The ceremony commands are the same trade in the other direction: they write files
//! snarkjs itself has to accept, so their formats are not ours to choose. `snarkrs-ceremony`
//! carries the byte-level reasoning and the snarkjs line numbers; this file only parses
//! arguments and prints what came back.
//!
//! Their `--backend` is not `prove`'s. `prove` selects a whole `snarkrs_groth16::Backend`; a
//! ceremony command selects one primitive, and the eight that have the flag are the eight
//! with a primitive worth moving. The bar every one of them is held to is that
//! `--backend cpu` and `--backend metal` write **byte-identical** files, which is what
//! carries the snarkjs equivalence the CPU path already has.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use snarkrs_ceremony::{
    bellman, challenge, contribute as zkey_mpc, phase1, prepare, ptau as ptau_file, setup,
    solidity, vkey, CeremonyError, CpuGroupFft, CpuKeyScale,
};
use snarkrs_cli::{
    bench,
    cli::{self, Parsed, Support},
    json, log, make_backend, BackendKind,
};
use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::{
    prove::{prove, prove_trace, prove_unchecked},
    verify::{verify, VerifyError},
    StageTimings,
};
use snarkrs_msm::{CpuMsm, GroupFft, KeyScale, MsmBackend};
#[cfg(feature = "witness-wasm")]
use snarkrs_witness::Input;
use snarkrs_witness::{native, WitnessError};

/// snarkjs' exit code for a line it could not use.
const BAD_USAGE: u8 = 99;

#[derive(Parser)]
#[command(
    name = "snarkrs",
    version,
    about = "A drop-in for the snarkjs 0.7.6 command line: Groth16 on BN254",
    disable_help_subcommand = true
)]
struct Cli {
    /// Debug-level log lines, snarkjs' `-v`.
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Powers of tau: the circuit-independent phase-1 ceremony.
    Powersoftau {
        #[command(subcommand)]
        cmd: PtauCmd,
    },
    /// Groth16 setup, prove and verify.
    Groth16 {
        #[command(subcommand)]
        cmd: Groth16Cmd,
    },
    /// Proving keys: the circuit-specific phase-2 ceremony.
    Zkey {
        #[command(subcommand)]
        cmd: ZkeyCmd,
    },
    /// Witnesses: circuit + input -> witness.wtns.
    Wtns {
        #[command(subcommand)]
        cmd: WtnsCmd,
    },
    /// iden3 binary files.
    File {
        #[command(subcommand)]
        cmd: FileCmd,
    },
    /// Benchmark proving, cold and warm, verifying every proof.
    Bench(bench::Args),
    /// Print a deterministic execution trace, for diffing against another machine.
    ///
    /// Debugging only. It pins stage 10's blinders, so it emits no proof.json: see
    /// `snarkrs_groth16::trace`.
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
    /// Sweep the CUDA FFT kernel variants: one NVRTC compile carries every candidate,
    /// and the fixed context and compile cost is paid once for the whole table.
    #[cfg(feature = "cuda")]
    FftBench(snarkrs_cli::fftbench::Args),
}

/// `--name` and `--entropy` as snarkjs reads them: `-n=VALUE`, and a bare `-n` is the
/// string "true", which is what `options.name` holds in that case.
#[derive(clap::Args)]
struct Contributor {
    /// Recorded in the contribution and printed by every later `verify`. `-n=NAME`.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    name: Option<String>,
}

#[derive(clap::Args)]
struct Entropy {
    /// Mixed with 64 bytes from the OS CSPRNG. Prompted for on stdin when absent, as
    /// snarkjs does. `-e=TEXT`.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    entropy: Option<String>,
}

/// cpu or metal. Both must write byte-identical output; that equivalence is what carries
/// the snarkjs one.
#[derive(clap::Args)]
struct CeremonyBackend {
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
}

#[derive(Subcommand)]
enum PtauCmd {
    /// Write a fresh accumulator at 2^POWER. Fully determined by the power: no randomness
    /// and no timestamps, so two runs produce identical bytes.
    New {
        /// bn128 (also bn254, alt_bn128). bls12381 is not supported.
        curve: String,
        /// 1 to 28.
        power: String,
        /// Defaults to powersOfTau<POWER>_0000.ptau.
        out: Option<PathBuf>,
    },
    /// Add a contribution from your own entropy.
    ///
    /// Non-deterministic by design: the entropy string is mixed with 64 bytes from the OS
    /// CSPRNG, so the same string twice gives two different keys.
    Contribute {
        ptau: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        contributor: Contributor,
        #[command(flatten)]
        entropy: Entropy,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
    /// Add the final contribution from a public beacon, the one anyone can reproduce.
    Beacon {
        ptau: PathBuf,
        out: PathBuf,
        /// Hex, even length, at most 255 bytes.
        beacon_hash: String,
        /// SHA-256 is iterated 2^N times over the beacon hash. 10 to 63.
        num_iterations_exp: String,
        #[command(flatten)]
        contributor: Contributor,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
    /// Precompute the Lagrange sections phase 2 needs.
    Prepare {
        #[command(subcommand)]
        cmd: PrepareCmd,
    },
    /// Check the contribution chain and recompute the challenge hashes.
    Verify { ptau: PathBuf },
    Export {
        #[command(subcommand)]
        cmd: PtauExportCmd,
    },
    Challenge {
        #[command(subcommand)]
        cmd: PtauChallengeCmd,
    },
    Import {
        #[command(subcommand)]
        cmd: PtauImportCmd,
    },
}

#[derive(Subcommand)]
enum PtauExportCmd {
    /// Write the challenge for the next contribution: the last response hash and the
    /// points, uncompressed. Deterministic, byte for byte snarkjs'.
    Challenge {
        ptau: PathBuf,
        #[arg(default_value = "challenge")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum PtauChallengeCmd {
    /// Contribute to a challenge file without the ptau, writing a response to import.
    Contribute {
        /// bn128 (also bn254, alt_bn128). bls12381 is not supported.
        curve: String,
        challenge: String,
        /// Defaults to the challenge name with its extension changed to `.response`.
        response: Option<PathBuf>,
        #[command(flatten)]
        entropy: Entropy,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
}

#[derive(Subcommand)]
enum PtauImportCmd {
    /// Import a response onto the ptau its challenge came from, as a new contribution.
    Response {
        ptau: PathBuf,
        response: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        contributor: Contributor,
        /// Write only the header and the contribution chain. A later import onto the file
        /// works; nothing else that needs the points does. `-nopoints`.
        #[arg(long)]
        nopoints: bool,
        /// Accepted and ignored, as in snarkjs 0.7.6, whose check is a TODO. `-nocheck`.
        #[arg(long)]
        nocheck: bool,
    },
}

#[derive(Subcommand)]
enum PrepareCmd {
    /// The expensive one: every butterfly of its inverse FFT is a full point scalar
    /// multiplication.
    Phase2 {
        ptau: PathBuf,
        out: PathBuf,
        /// cpu, metal or cuda. All must write byte-identical output; that equivalence is
        /// what carries the snarkjs one.
        #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
    },
}

/// `groth16 prove`'s options of ours, which `fullprove` takes too.
#[derive(clap::Args)]
struct ProveOpts {
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    /// Print the per-stage split of the proving time.
    #[arg(long)]
    stage_timings: bool,
    /// Verify the proof before writing it, and fail rather than emit one that does not
    /// verify. On by default; the cost is under 3% of a proof.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    self_verify: bool,
    /// The circuit's verification_key.json, from a source independent of the zkey.
    /// Checked against the zkey's public-input count before proving, and the proof is
    /// verified against it before anything is written.
    #[arg(long, value_name = "FILE")]
    vkey: Option<PathBuf>,
    /// When a GPU proof fails its self-verify or the device faults, prove once more on
    /// the same backend and then on the CPU, reporting each failure. A logic bug fails
    /// the same way every time; a transient accelerator fault does not. No effect with
    /// --self-verify false or --backend cpu.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fallback: bool,
    /// MSMs whose cost follows the key and not the witness, so proving time does not
    /// reveal how many witness entries are zero or one (the timing channel USENIX
    /// Security 2020 used on Zcash). From 2.5% (cpu), 26% (metal) or 55% (wgpu) slower
    /// on a dense circuit to 4x (cpu), 6x (metal) or 10x (wgpu) on a bit-heavy one. cpu,
    /// metal and wgpu backends.
    #[arg(long)]
    constant_work: bool,
}

#[derive(Subcommand)]
enum Groth16Cmd {
    /// Phase-2 setup: circuit + prepared powers of tau -> an initial zkey.
    ///
    /// The key this writes has delta = 1 and is not safe to prove with. It is the input to
    /// `zkey contribute`, which is what gives it a delta nobody knows.
    Setup {
        #[arg(default_value = "circuit.r1cs")]
        r1cs: PathBuf,
        #[arg(default_value = "powersoftau.ptau")]
        ptau: PathBuf,
        #[arg(default_value = "circuit_0000.zkey")]
        out: PathBuf,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
    /// Prove: zkey + witness -> proof.json and public.json.
    Prove {
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
        #[arg(default_value = "witness.wtns")]
        witness: PathBuf,
        #[arg(default_value = "proof.json")]
        proof: PathBuf,
        #[arg(default_value = "public.json")]
        public: PathBuf,
        #[command(flatten)]
        opts: ProveOpts,
        /// snarkjs' `-protocol`, which its groth16 prove accepts and never reads.
        #[arg(long, hide = true, require_equals = true, num_args = 0..=1)]
        protocol: Option<String>,
    },
    /// Compute the witness and prove it: input.json + witness generator + zkey ->
    /// proof.json and public.json.
    ///
    /// The generator is circom's circuit.wasm or the native binary `circom --c` builds,
    /// told apart by content. With the wasm the witness never touches the disk.
    Fullprove {
        #[arg(default_value = "input.json")]
        input: PathBuf,
        /// snarkjs' usage line says circuit_final.wasm; the code it runs defaults to
        /// circuit.wasm, and so does this.
        #[arg(default_value = "circuit.wasm")]
        wasm: PathBuf,
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
        #[arg(default_value = "proof.json")]
        proof: PathBuf,
        #[arg(default_value = "public.json")]
        public: PathBuf,
        #[command(flatten)]
        opts: ProveOpts,
    },
    /// Verify a proof against snarkjs' verification_key.json.
    Verify {
        #[arg(default_value = "verification_key.json")]
        vkey: PathBuf,
        #[arg(default_value = "public.json")]
        public: PathBuf,
        #[arg(default_value = "proof.json")]
        proof: PathBuf,
    },
}

#[derive(Subcommand)]
enum ZkeyCmd {
    /// Add a contribution from your own entropy. Sections 3 to 7 are copied verbatim, so
    /// this costs two MSM-free rescalings and nothing else.
    Contribute {
        zkey: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        contributor: Contributor,
        #[command(flatten)]
        entropy: Entropy,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
    /// Add the final contribution from a public beacon.
    Beacon {
        zkey: PathBuf,
        out: PathBuf,
        beacon_hash: String,
        num_iterations_exp: String,
        #[command(flatten)]
        contributor: Contributor,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
    /// Check the contribution chain and that the final key is a consistent rescaling of
    /// the initial one.
    Verify {
        #[command(subcommand)]
        cmd: ZkeyVerifyCmd,
    },
    Export {
        #[command(subcommand)]
        cmd: ZkeyExportCmd,
    },
    Bellman {
        #[command(subcommand)]
        cmd: ZkeyBellmanCmd,
    },
    Import {
        #[command(subcommand)]
        cmd: ZkeyImportCmd,
    },
}

#[derive(Subcommand)]
enum ZkeyBellmanCmd {
    /// Contribute to a bellman MPCParameters file, as Zcash's phase2 tool does. The
    /// result goes back into the zkey with `zkey import bellman`.
    Contribute {
        /// bn128 (also bn254, alt_bn128). bls12381 is not supported.
        curve: String,
        input: PathBuf,
        output: PathBuf,
        #[command(flatten)]
        entropy: Entropy,
        #[command(flatten)]
        backend: CeremonyBackend,
    },
}

#[derive(Subcommand)]
enum ZkeyImportCmd {
    /// Take the contributions a bellman MPCParameters file adds onto the zkey it was
    /// exported from. Deterministic, byte for byte snarkjs'.
    Bellman {
        zkey: PathBuf,
        mpcparams: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        contributor: Contributor,
    },
}

#[derive(Subcommand)]
enum ZkeyVerifyCmd {
    /// Re-run the whole setup to produce the initial key, then check the chain from it.
    /// The only form that checks the key against the circuit, and it costs a full setup.
    R1cs {
        #[arg(default_value = "circuit.r1cs")]
        r1cs: PathBuf,
        #[arg(default_value = "powersoftau.ptau")]
        ptau: PathBuf,
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
    },
    /// Check the chain from a given initial key. Nothing ties that key to any circuit.
    Init {
        #[arg(default_value = "circuit_0000.zkey")]
        init: PathBuf,
        #[arg(default_value = "powersoftau.ptau")]
        ptau: PathBuf,
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
    },
}

#[derive(Subcommand)]
enum ZkeyExportCmd {
    /// Write snarkjs' verification_key.json: the four verifier points, the precomputed
    /// pairing, and IC.
    Verificationkey {
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
        /// snarkjs' usage line says verification_key.json; the code it runs writes
        /// circuit_vk.json, and so does this.
        #[arg(default_value = "circuit_vk.json")]
        out: PathBuf,
    },
    /// Write a Solidity verifier with snarkjs' interface, so its soliditycalldata calls it
    /// unchanged. The contract is our own MIT one, not snarkjs' GPL-3.0 template.
    Solidityverifier {
        #[arg(default_value = "circuit_final.zkey")]
        zkey: PathBuf,
        #[arg(default_value = "verifier.sol")]
        out: PathBuf,
    },
    /// Write the zkey as a bellman MPCParameters file for Zcash's phase2 tool.
    /// Deterministic, byte for byte snarkjs'.
    Bellman {
        zkey: PathBuf,
        #[arg(default_value = "circuit.mpcparams")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum WtnsCmd {
    /// Compute the witness for an input, byte for byte the file snarkjs writes.
    ///
    /// The generator is circom's circuit.wasm, or the native binary `circom --c` builds
    /// (with its .dat beside it), told apart by content and not by name.
    Calculate {
        #[arg(default_value = "circuit.wasm")]
        wasm: PathBuf,
        #[arg(default_value = "input.json")]
        input: PathBuf,
        #[arg(default_value = "witness.wtns")]
        witness: PathBuf,
    },
    /// Compute the witness with the module's own sanity checks on. Wasm only.
    Debug {
        #[arg(default_value = "circuit.wasm")]
        wasm: PathBuf,
        #[arg(default_value = "input.json")]
        input: PathBuf,
        #[arg(default_value = "witness.wtns")]
        witness: PathBuf,
        /// Defaults to the wasm's name with its extension changed to `.sym`.
        sym: Option<PathBuf>,
        /// Log every signal read. circom 2 modules have no hook for it, so, as in snarkjs,
        /// this prints nothing for them. `-g`.
        #[arg(long)]
        get: bool,
        /// Log every signal write. As `--get`. `-s`.
        #[arg(long)]
        set: bool,
        /// Log every component start and finish. As `--get`. `-t`.
        #[arg(long)]
        trigger: bool,
    },
}

#[derive(Subcommand)]
enum FileCmd {
    /// Report the header, every section's declared and present size, and the contribution
    /// count.
    ///
    /// Reads leniently, so it describes a truncated download instead of refusing to open
    /// it. That is the only command here that does.
    ///
    /// The file is optional in snarkjs' usage line and required by the code behind it,
    /// which throws without one. So it fails here too, with exit 1 and not clap's 99.
    Info { file: Option<PathBuf> },
}

fn main() -> ExitCode {
    let line = match cli::normalise(std::env::args_os()) {
        Parsed::Clap(line) => line,
        Parsed::Help => {
            print!("{}", cli::help_all());
            return ExitCode::SUCCESS;
        }
        Parsed::NoCommand => {
            print!("{}", cli::help_all());
            return ExitCode::from(BAD_USAGE);
        }
        Parsed::Unknown(_) => {
            println!("Invalid command");
            print!("{}", cli::help_all());
            return ExitCode::from(BAD_USAGE);
        }
        Parsed::Unsupported(c) => {
            let words = c.words().join(" ");
            log::error(match c.support {
                Support::Never => {
                    format!("`{words}` is not supported: snarkrs is Groth16 on BN254 only")
                }
                _ => format!("`{words}` is a snarkjs command snarkrs does not implement yet"),
            });
            return ExitCode::FAILURE;
        }
    };
    let cli = match Cli::try_parse_from(line) {
        Ok(cli) => cli,
        Err(e) => {
            // Help and version are errors to clap and successes to everyone else.
            let _ = e.print();
            return match e.use_stderr() {
                true => ExitCode::from(BAD_USAGE),
                false => ExitCode::SUCCESS,
            };
        }
    };
    log::set_verbose(cli.verbose);
    // One proof and exit, so there is nothing for the cache to be reused by. Not `bench`,
    // whose warm loop is a long-lived prover and would fault its buffers in on every rep.
    // fullprove's native witness binary inherits the setting, as `snarkrs-csp-mem`'s does.
    if matches!(
        cli.cmd,
        Cmd::Groth16 {
            cmd: Groth16Cmd::Prove { .. } | Groth16Cmd::Fullprove { .. }
        }
    ) {
        snarkrs_groth16::malloc::reexec_without_large_cache();
    }
    match run(cli.cmd) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            // `logger.error(err)` in cli.js, which prints the thrown Error as `Error: msg`.
            log::error(format!("Error: {e:#}"));
            ExitCode::FAILURE
        }
    }
}

/// Runs one command. `Ok(1)` is a verdict: a proof or a file that does not verify.
fn run(cmd: Cmd) -> Result<u8> {
    match cmd {
        Cmd::Powersoftau { cmd } => run_ptau(cmd),
        Cmd::Groth16 { cmd } => run_groth16(cmd),
        Cmd::Zkey { cmd } => run_zkey(cmd),
        Cmd::Wtns { cmd } => run_wtns(cmd),
        Cmd::File {
            cmd: FileCmd::Info { file },
        } => {
            let file = file.ok_or_else(|| anyhow::anyhow!("file info needs a file"))?;
            run_file_info(&file).map(|()| 0)
        }
        Cmd::Bench(args) => bench::run(args).map(|()| 0),
        Cmd::Trace {
            zkey,
            witness,
            backend,
            out,
        } => run_trace(&zkey, &witness, backend, out.as_deref()).map(|()| 0),
        #[cfg(feature = "cuda")]
        Cmd::FftBench(args) => snarkrs_cli::fftbench::run(args).map(|()| 0),
    }
}

#[derive(Clone, Copy, Debug)]
enum CeremonySeam {
    Msm,
    GroupFft,
    KeyScale,
}

/// Selection without opening a device. A compiled backend can still fail to initialise.
fn ceremony_backend_kind(kind: BackendKind, seam: CeremonySeam) -> Result<BackendKind> {
    match kind {
        BackendKind::Cpu => Ok(kind),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Cuda if matches!(seam, CeremonySeam::GroupFft) && cfg!(feature = "cuda") => {
            Ok(kind)
        }
        BackendKind::Cuda => Err(no_cuda()),
        BackendKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Ok(kind)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(no_metal())
            }
        }
    }
}

/// The MSM behind `setup` and the two same-ratio checks.
///
/// The three constructors below are `make_backend` for the ceremony: same three failure
/// modes, same three messages, same refusal to fall back to the CPU when the asked-for
/// backend is not there. A benchmark that silently measures the other backend is worse than
/// no number, and a `.zkey` that silently came from the other backend is worse than no key.
fn msm_backend(kind: BackendKind) -> Result<Box<dyn MsmBackend>> {
    match ceremony_backend_kind(kind, CeremonySeam::Msm)? {
        BackendKind::Cpu => Ok(Box::new(CpuMsm::new())),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Metal => metal_msm(),
        BackendKind::Cuda => Err(no_cuda()),
    }
}

/// The group inverse FFT behind `ptau prepare`.
fn fft_backend(kind: BackendKind) -> Result<Box<dyn GroupFft>> {
    match ceremony_backend_kind(kind, CeremonySeam::GroupFft)? {
        BackendKind::Cpu => Ok(Box::new(CpuGroupFft)),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Metal => metal_fft(),
        BackendKind::Cuda => cuda_fft(),
    }
}

/// The batch apply-key behind every contribute and beacon command.
fn key_backend(kind: BackendKind) -> Result<Box<dyn KeyScale>> {
    match ceremony_backend_kind(kind, CeremonySeam::KeyScale)? {
        BackendKind::Cpu => Ok(Box::new(CpuKeyScale)),
        BackendKind::Wgpu => Err(no_wgpu()),
        BackendKind::Metal => metal_key(),
        BackendKind::Cuda => Err(no_cuda()),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_msm() -> Result<Box<dyn MsmBackend>> {
    Ok(Box::new(snarkrs_metal::MetalMsmBackend::new().map_err(
        |e| anyhow::anyhow!("backend `metal` is unavailable: {e}"),
    )?))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_fft() -> Result<Box<dyn GroupFft>> {
    Ok(Box::new(snarkrs_metal::MetalGroupFft::new().map_err(
        |e| anyhow::anyhow!("backend `metal` is unavailable: {e}"),
    )?))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_key() -> Result<Box<dyn KeyScale>> {
    Ok(Box::new(snarkrs_metal::MetalKeyScale::new().map_err(
        |e| anyhow::anyhow!("backend `metal` is unavailable: {e}"),
    )?))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_msm() -> Result<Box<dyn MsmBackend>> {
    Err(no_metal())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_fft() -> Result<Box<dyn GroupFft>> {
    Err(no_metal())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_key() -> Result<Box<dyn KeyScale>> {
    Err(no_metal())
}

#[cfg(feature = "cuda")]
fn cuda_fft() -> Result<Box<dyn GroupFft>> {
    Ok(Box::new(snarkrs_cuda::CudaGroupFft::new().map_err(
        |e| anyhow::anyhow!("backend `cuda` is unavailable: {e}"),
    )?))
}

#[cfg(not(feature = "cuda"))]
fn cuda_fft() -> Result<Box<dyn GroupFft>> {
    Err(no_cuda())
}

/// The feature is off, or it is on and the target is not macOS. Two different fixes, so
/// two different messages, as in `make_backend`.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn no_metal() -> anyhow::Error {
    if cfg!(feature = "metal") {
        anyhow::anyhow!(
            "backend `metal` is unavailable: the `metal` feature is enabled but this target \
             is not macOS, so snarkrs-metal compiled to nothing"
        )
    } else {
        anyhow::anyhow!(
            "backend `metal` is unavailable: this binary was built WITHOUT the `metal` \
             feature. Rebuild with `cargo build --release --features metal`."
        )
    }
}

/// CUDA has kernels for the prover and for `ptau prepare`'s group FFT, and none for the
/// two ceremony seams this message now covers. Saying "not built in" when the feature
/// *is* on would send someone to rebuild a binary that already has everything it is
/// going to get.
fn no_cuda() -> anyhow::Error {
    if cfg!(feature = "cuda") {
        anyhow::anyhow!(
            "backend `cuda` is unavailable for this ceremony command: snarkrs-cuda implements \
             the prover and the `ptau prepare` group FFT; the ceremony MSM and apply-key \
             have no CUDA kernels yet"
        )
    } else {
        anyhow::anyhow!(
            "backend `cuda` is unavailable: this binary was built WITHOUT the `cuda` \
             feature. Rebuild with `cargo build --release --features cuda`."
        )
    }
}

/// Unlike the other two this is not a "yet". A ceremony kernel is a point scalar
/// multiplication, which inlines the point operations three or more times over a 256-byte
/// `Xyzz<Fq2>`, and that is exactly the shape WebKit 323560 miscompiles on iOS.
fn no_wgpu() -> anyhow::Error {
    anyhow::anyhow!(
        "backend `wgpu` is unavailable for the ceremony: these kernels are Metal and CUDA \
         only, because a scalar-multiplication ladder is the shape WebKit 323560 \
         miscompiles. Use `--backend cpu` or `--backend metal`."
    )
}

/// `getRandomRng` in snarkjs' `misc.js`: no `-e`, or an empty one, and the contributor is
/// asked on stdin until they type something. A closed stdin is an error rather than
/// snarkjs' silent exit, so a script that forgot `-e` fails instead of hanging.
fn entropy_or_prompt(given: Option<String>) -> Result<String> {
    if let Some(e) = given.filter(|e| !e.is_empty()) {
        return Ok(e);
    }
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    loop {
        print!("Enter a random text. (Entropy): ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            anyhow::bail!("no entropy: stdin closed before a line was read. Pass -e=TEXT");
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if !line.is_empty() {
            return Ok(line.to_string());
        }
    }
}

fn run_ptau(cmd: PtauCmd) -> Result<u8> {
    match cmd {
        PtauCmd::New { curve, power, out } => {
            cli::check_curve(&curve)?;
            let power: u32 = power
                .parse()
                .ok()
                .filter(|p| (1..=28).contains(p))
                .ok_or_else(|| anyhow::anyhow!("Power must be between 1 and 28"))?;
            let out = out.unwrap_or_else(|| format!("powersOfTau{power}_0000.ptau").into());
            let challenge = phase1::ptau_new(power, &out)
                .with_context(|| format!("writing {}", out.display()))?;
            log::info(log::format_hash(&challenge, "First Contribution Hash:"));
            log::debug(format!("wrote {}", out.display()));
        }
        PtauCmd::Contribute {
            ptau,
            out,
            contributor,
            entropy,
            backend,
        } => {
            let key = key_backend(backend.backend)?;
            let entropy = entropy_or_prompt(entropy.entropy)?;
            let report = phase1::contribute(
                &ptau,
                &out,
                contributor.name.as_deref(),
                &entropy,
                key.as_ref(),
            )
            .with_context(|| format!("contributing to {}", ptau.display()))?;
            print_phase1(&report, &out);
        }
        PtauCmd::Beacon {
            ptau,
            out,
            beacon_hash,
            num_iterations_exp,
            contributor,
            backend,
        } => {
            let (hash, exp) = phase1::parse_beacon_args(&beacon_hash, &num_iterations_exp)?;
            let key = key_backend(backend.backend)?;
            let report = phase1::beacon(
                &ptau,
                &out,
                contributor.name.as_deref(),
                &hash,
                exp,
                key.as_ref(),
            )
            .with_context(|| format!("beaconing {}", ptau.display()))?;
            print_phase1(&report, &out);
        }
        PtauCmd::Prepare {
            cmd: PrepareCmd::Phase2 { ptau, out, backend },
        } => {
            let fft = fft_backend(backend)?;
            prepare::prepare_phase2(&ptau, &out, fft.as_ref())
                .with_context(|| format!("preparing {}", ptau.display()))?;
            log::debug(format!("wrote {}", out.display()));
        }
        PtauCmd::Verify { ptau } => {
            let report = match phase1::verify(&ptau) {
                Ok(r) => r,
                // A file that does not verify is a verdict, as in snarkjs: the reason on
                // an ERROR line and exit 1.
                Err(e) => {
                    log::error(format!("{:#}", anyhow::Error::from(e)));
                    return Ok(1);
                }
            };
            log::info("Powers Of tau file OK!");
            log::debug(format!(
                "power {}, ceremony power {}, prepared {}",
                report.power, report.ceremony_power, report.prepared
            ));
            // Newest first, as `powersoftau_verify.js` prints them.
            for (i, h) in report.contribution_hashes.iter().enumerate().rev() {
                log::info("-----------------------------------------------------");
                log::info(format!("Contribution #{}:", i + 1));
                log::info(log::format_hash(h, "Response Hash:"));
            }
            log::info("-----------------------------------------------------");
            // Saying which check was skipped is the point: a truncated file cannot have
            // its final next-challenge compared, and reporting "OK" without that caveat
            // overstates what was verified.
            if !report.next_challenge_checked {
                log::warn("next challenge NOT checked: this file is truncated");
            }
            if !report.prepared {
                log::warn(
                    "this file does not contain phase2 precalculated values. Please run: \n   \
                     snarkrs \"powersoftau prepare phase2\" to prepare this file to be used in \
                     the phase2 ceremony.",
                );
            }
            log::info("Powers of Tau Ok!");
        }
        PtauCmd::Export {
            cmd: PtauExportCmd::Challenge { ptau, out },
        } => {
            let report = challenge::export_challenge(&ptau, &out)
                .with_context(|| format!("exporting from {}", ptau.display()))?;
            log::info(log::format_hash(
                &report.last_response_hash,
                "Last Response Hash: ",
            ));
            log::info(log::format_hash(
                &report.challenge_hash,
                "New Challenge Hash: ",
            ));
            log::debug(format!("wrote {}", out.display()));
        }
        PtauCmd::Challenge {
            cmd:
                PtauChallengeCmd::Contribute {
                    curve,
                    challenge,
                    response,
                    entropy,
                    backend,
                },
        } => {
            cli::check_curve(&curve)?;
            let response =
                response.unwrap_or_else(|| cli::change_ext(&challenge, "response").into());
            let key = key_backend(backend.backend)?;
            let entropy = entropy_or_prompt(entropy.entropy)?;
            let report = challenge::challenge_contribute(
                Path::new(&challenge),
                &response,
                &entropy,
                key.as_ref(),
            )
            .with_context(|| format!("contributing to {challenge}"))?;
            log::debug(format!("Power to tau size: {}", report.power));
            log::info(log::format_hash(
                &report.claimed_previous_response,
                "Claimed Previous Response Hash: ",
            ));
            log::info(log::format_hash(
                &report.challenge_hash,
                "Current Challenge Hash: ",
            ));
            log::info(log::format_hash(
                &report.response_hash,
                "Contribution Response Hash: ",
            ));
            log::debug(format!("wrote {}", response.display()));
        }
        PtauCmd::Import {
            cmd:
                PtauImportCmd::Response {
                    ptau,
                    response,
                    out,
                    contributor,
                    nopoints,
                    nocheck: _,
                },
        } => {
            let report = challenge::import_response(
                &ptau,
                &response,
                &out,
                contributor.name.as_deref(),
                !nopoints,
            )
            .with_context(|| format!("importing {}", response.display()))?;
            // Without the points there is no next challenge to hash, only the `0xff`
            // placeholder, and snarkjs prints no line for it.
            if nopoints {
                log::info(log::format_hash(
                    &report.response_hash,
                    "Contribution Response Hash imported: ",
                ));
            } else {
                print_phase1(&report, &out);
            }
        }
    }
    Ok(0)
}

fn print_phase1(report: &phase1::Phase1Report, out: &Path) {
    log::info(log::format_hash(
        &report.response_hash,
        "Contribution Response Hash imported: ",
    ));
    log::info(log::format_hash(
        &report.next_challenge,
        "Next Challenge Hash: ",
    ));
    log::debug(format!(
        "contribution #{} written to {}",
        report.index,
        out.display()
    ));
}

fn run_groth16(cmd: Groth16Cmd) -> Result<u8> {
    match cmd {
        Groth16Cmd::Setup {
            r1cs,
            ptau,
            out,
            backend,
        } => {
            let msm = msm_backend(backend.backend)?;
            let report = setup::setup(&r1cs, &ptau, &out, msm.as_ref())
                .with_context(|| format!("setting up {}", r1cs.display()))?;
            log::debug(format!(
                "constraints {}, vars {}, public {}, domain size {} (2^{}), coefficients {}",
                report.n_constraints,
                report.n_vars,
                report.n_public,
                report.domain_size,
                report.cir_power,
                report.n_coefs
            ));
            log::info(log::format_hash(&report.cs_hash, "Circuit hash: "));
            log::debug(format!("wrote {}", out.display()));
            Ok(0)
        }
        Groth16Cmd::Prove {
            zkey,
            witness,
            proof,
            public,
            opts,
            protocol: _,
        } => run_prove(&zkey, WitnessFrom::File(&witness), &proof, &public, &opts).map(|()| 0),
        Groth16Cmd::Fullprove {
            input,
            wasm,
            zkey,
            proof,
            public,
            opts,
        } => run_fullprove(&input, &wasm, &zkey, &proof, &public, &opts),
        Groth16Cmd::Verify {
            vkey,
            public,
            proof,
        } => run_verify(&vkey, &public, &proof),
    }
}

fn run_zkey(cmd: ZkeyCmd) -> Result<u8> {
    match cmd {
        ZkeyCmd::Contribute {
            zkey,
            out,
            contributor,
            entropy,
            backend,
        } => {
            let key = key_backend(backend.backend)?;
            let entropy = entropy_or_prompt(entropy.entropy)?;
            let report = zkey_mpc::contribute(
                &zkey,
                &out,
                contributor.name.as_deref(),
                &entropy,
                key.as_ref(),
            )
            .with_context(|| format!("contributing to {}", zkey.display()))?;
            log::info(log::format_hash(&report.hash, "Contribution Hash: "));
            log::debug(format!(
                "contribution #{} written to {}",
                report.index,
                out.display()
            ));
        }
        ZkeyCmd::Beacon {
            zkey,
            out,
            beacon_hash,
            num_iterations_exp,
            contributor,
            backend,
        } => {
            let (hash, exp) = phase1::parse_beacon_args(&beacon_hash, &num_iterations_exp)?;
            let key = key_backend(backend.backend)?;
            let report = zkey_mpc::beacon(
                &zkey,
                &out,
                contributor.name.as_deref(),
                &hash,
                exp,
                key.as_ref(),
            )
            .with_context(|| format!("beaconing {}", zkey.display()))?;
            log::info(log::format_hash(&report.hash, "Contribution Hash: "));
            log::debug(format!(
                "contribution #{} written to {}",
                report.index,
                out.display()
            ));
        }
        ZkeyCmd::Verify { cmd } => {
            // No `--backend`: with the r1cs form this re-runs the whole setup, and a
            // verifier that shares the accelerator with the thing it is checking checks less.
            let msm = msm_backend(BackendKind::Cpu)?;
            let (report, zkey) = match &cmd {
                ZkeyVerifyCmd::R1cs { r1cs, ptau, zkey } => (
                    zkey_mpc::verify_from_r1cs(r1cs, ptau, zkey, msm.as_ref()),
                    zkey,
                ),
                ZkeyVerifyCmd::Init { init, ptau, zkey } => (
                    zkey_mpc::verify_from_init(init, ptau, zkey, msm.as_ref()),
                    zkey,
                ),
            };
            let report = match report {
                Ok(r) => r,
                Err(e) => {
                    let e = anyhow::Error::from(e).context(format!("verifying {}", zkey.display()));
                    log::error(format!("{e:#}"));
                    return Ok(1);
                }
            };
            log::debug(format!(
                "vars {}, public {}, domain size {}",
                report.n_vars, report.n_public, report.domain_size
            ));
            // Newest first, as `zkey_verify_frominit.js` prints them.
            for (i, h) in report.contribution_hashes.iter().enumerate().rev() {
                log::info("-------------------------");
                log::info(log::format_hash(h, &format!("contribution #{}:", i + 1)));
            }
            log::info("-------------------------");
            if matches!(cmd, ZkeyVerifyCmd::Init { .. }) {
                log::debug("the init form checks the chain, not that the key matches a circuit");
            }
            log::info("ZKey Ok!");
        }
        ZkeyCmd::Export {
            cmd: ZkeyExportCmd::Verificationkey { zkey, out },
        } => {
            log::info("EXPORT VERIFICATION KEY STARTED");
            vkey::export_verification_key(&zkey, &out)
                .with_context(|| format!("exporting from {}", zkey.display()))?;
            log::info("> Detected protocol: groth16");
            log::info("EXPORT VERIFICATION KEY FINISHED");
            log::debug(format!("wrote {}", out.display()));
        }
        ZkeyCmd::Export {
            cmd: ZkeyExportCmd::Solidityverifier { zkey, out },
        } => {
            // snarkjs builds the contract from the verification key, and prints that
            // export's three lines on the way.
            log::info("EXPORT VERIFICATION KEY STARTED");
            let source = solidity::solidity_verifier(&zkey)
                .with_context(|| format!("exporting from {}", zkey.display()))?;
            log::info("> Detected protocol: groth16");
            log::info("EXPORT VERIFICATION KEY FINISHED");
            std::fs::write(&out, source).with_context(|| format!("writing {}", out.display()))?;
            log::debug(format!("wrote {}", out.display()));
        }
        ZkeyCmd::Export {
            cmd: ZkeyExportCmd::Bellman { zkey, out },
        } => {
            bellman::export_bellman(&zkey, &out)
                .with_context(|| format!("exporting from {}", zkey.display()))?;
            log::debug(format!("wrote {}", out.display()));
        }
        ZkeyCmd::Bellman {
            cmd:
                ZkeyBellmanCmd::Contribute {
                    curve,
                    input,
                    output,
                    entropy,
                    backend,
                },
        } => {
            cli::check_curve(&curve)?;
            let key = key_backend(backend.backend)?;
            let entropy = entropy_or_prompt(entropy.entropy)?;
            let hash = bellman::bellman_contribute(&input, &output, &entropy, key.as_ref())
                .with_context(|| format!("contributing to {}", input.display()))?;
            log::info(log::format_hash(&hash, "Contribution Hash: "));
            log::debug(format!("wrote {}", output.display()));
        }
        ZkeyCmd::Import {
            cmd:
                ZkeyImportCmd::Bellman {
                    zkey,
                    mpcparams,
                    out,
                    contributor,
                },
        } => {
            match bellman::import_bellman(&zkey, &mpcparams, &out, contributor.name.as_deref()) {
                Ok(report) => log::debug(format!(
                    "{} contributions imported onto {}, written to {}",
                    report.contribution_hashes.len() - report.n_prior,
                    zkey.display(),
                    out.display()
                )),
                // A file that does not continue this zkey's chain is a verdict, as in
                // `zkey_import_bellman.js`: its reason on an ERROR line and exit 1.
                Err(CeremonyError::Verification(why)) => {
                    log::error(capitalise(&why));
                    return Ok(1);
                }
                Err(e) => {
                    return Err(anyhow::Error::from(e)
                        .context(format!("importing {}", mpcparams.display())))
                }
            }
        }
    }
    Ok(0)
}

/// snarkjs starts its verdicts with a capital; snarkrs-ceremony's errors do not.
fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

/// `file info`. For a ptau this is our lenient report rather than snarkjs' section table:
/// it keeps working on a truncated download, which is when anyone runs it.
fn run_file_info(file: &Path) -> Result<()> {
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    match ext {
        "ptau" => {}
        // TODO: a generic iden3 binfile reader for the other three.
        "zkey" | "r1cs" | "wtns" => {
            anyhow::bail!("file info for .{ext} files is not implemented yet, only .ptau")
        }
        _ => anyhow::bail!("Extension {ext} is not allowed."),
    }
    let file = ptau_file::Ptau::open_lenient(file)
        .with_context(|| format!("opening {}", file.display()))?;
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
    // Declared, present and expected are printed side by side rather than reduced to a
    // verdict: a length that is short is truncation, a length that disagrees with `power`
    // is corruption, and the two need different responses.
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

/// A witness that could not be computed is reported the way snarkjs reports it: its error
/// class and message on an ERROR line, and exit 1.
fn witness_failed(e: &WitnessError) -> u8 {
    log::error(e.snarkjs_line());
    1
}

/// The witness for `input`, in memory, from either kind of generator.
fn generate_witness(generator: &Path, input: &Path) -> Result<Vec<Fr>, WitnessError> {
    // snarkjs reads the input before it opens the wasm, so a missing input is reported
    // first.
    let text = snarkrs_witness::read_file(input)?;
    match native::detect(generator)? {
        native::Kind::Wasm => wasm_witness(generator, &text, false),
        native::Kind::Native => native::calculate(generator, input, None),
    }
}

#[cfg(feature = "witness-wasm")]
fn wasm_witness(wasm: &Path, input: &[u8], sanity_check: bool) -> Result<Vec<Fr>, WitnessError> {
    let input = Input::from_json_str(&String::from_utf8_lossy(input))?;
    let calc = snarkrs_witness::WitnessCalculator::from_file(wasm)?;
    match sanity_check {
        true => calc.calculate_sanity_checked(&input),
        false => calc.calculate(&input),
    }
}

#[cfg(not(feature = "witness-wasm"))]
fn wasm_witness(wasm: &Path, _: &[u8], _: bool) -> Result<Vec<Fr>, WitnessError> {
    Err(WitnessError::Unsupported(format!(
        "{} is a wasm witness calculator, and this snarkrs was built without the \
         `witness-wasm` feature. Pass the native binary circom --c builds, or rebuild with \
         `cargo build --release --features witness-wasm`.",
        wasm.display()
    )))
}

fn run_wtns(cmd: WtnsCmd) -> Result<u8> {
    match cmd {
        WtnsCmd::Calculate {
            wasm,
            input,
            witness,
        } => {
            let done =
                snarkrs_witness::read_file(&input).and_then(|text| match native::detect(&wasm)? {
                    native::Kind::Wasm => {
                        let mut w = wasm_witness(&wasm, &text, false)?;
                        let r = snarkrs_witness::write_wtns(&witness, &w);
                        snarkrs_groth16::scrub(&mut w);
                        r
                    }
                    native::Kind::Native => native::calculate_to_file(&wasm, &input, &witness),
                });
            if let Err(e) = done {
                return Ok(witness_failed(&e));
            }
            log::debug(format!("wrote {}", witness.display()));
            log::tip_witness();
        }
        WtnsCmd::Debug {
            wasm,
            input,
            witness,
            sym,
            get: _,
            set: _,
            trigger: _,
        } => {
            let sym = sym.unwrap_or_else(|| cli::change_ext(&wasm.to_string_lossy(), "sym").into());
            let done = snarkrs_witness::read_file(&input).and_then(|text| {
                if native::detect(&wasm)? == native::Kind::Native {
                    return Err(WitnessError::Unsupported(format!(
                        "wtns debug runs the wasm witness calculator only, and {} is a native \
                         binary. Use wtns calculate with it, or pass circuit.wasm",
                        wasm.display()
                    )));
                }
                // snarkjs loads the symbols whatever the flags, so a missing .sym fails here
                // too. A circom 2 module calls none of the hooks that would print them.
                snarkrs_witness::read_file(&sym)?;
                let mut w = wasm_witness(&wasm, &text, true)?;
                let r = snarkrs_witness::write_wtns(&witness, &w);
                snarkrs_groth16::scrub(&mut w);
                r
            });
            if let Err(e) = done {
                return Ok(witness_failed(&e));
            }
            log::debug(format!("wrote {}", witness.display()));
        }
    }
    Ok(0)
}

/// `groth16 fullprove`: `prove`, with the witness computed here instead of read from a file.
/// From the wasm it stays in memory; a native binary writes it to a private temp file that
/// is zeroed and removed before the key is even loaded.
fn run_fullprove(
    input: &Path,
    generator: &Path,
    zkey: &Path,
    proof: &Path,
    public: &Path,
    opts: &ProveOpts,
) -> Result<u8> {
    let generate = || generate_witness(generator, input).map_err(anyhow::Error::from);
    match run_prove(
        zkey,
        WitnessFrom::Generate(Box::new(generate)),
        proof,
        public,
        opts,
    ) {
        Ok(()) => {
            log::tip_witness();
            Ok(0)
        }
        Err(e) => match e.downcast_ref::<WitnessError>() {
            Some(we) => Ok(witness_failed(we)),
            None => Err(e),
        },
    }
}

/// Where `run_prove` gets its witness.
enum WitnessFrom<'a> {
    /// A `.wtns`, as `groth16 prove` takes it.
    File(&'a Path),
    /// Computed in this process, as `groth16 fullprove` does. Called once the process can
    /// no longer dump core, and before the key is loaded, so a bad input fails fast.
    Generate(Box<dyn FnOnce() -> Result<Vec<Fr>> + 'a>),
}

/// A witness that is zeroed when it drops, on the error paths as well.
struct Scrubbed(Vec<Fr>);

impl Drop for Scrubbed {
    fn drop(&mut self) {
        snarkrs_groth16::scrub(&mut self.0);
    }
}

fn run_prove(
    zkey: &Path,
    witness: WitnessFrom,
    proof_out: &Path,
    public_out: &Path,
    opts: &ProveOpts,
) -> Result<()> {
    let ProveOpts {
        backend,
        stage_timings,
        self_verify,
        ref vkey,
        fallback,
        constant_work,
    } = *opts;
    // Before the key is read: a backend this binary cannot run, or one that cannot keep
    // --constant-work's promise, is refused without the load.
    let backend_of = |kind| {
        if constant_work {
            snarkrs_cli::make_constant_work_backend(kind)
        } else {
            make_backend(kind)
        }
    };
    let prover = backend_of(backend)?;
    // This process is about to hold the witness in the clear. `panic = "abort"` means a crash
    // runs no destructor, so nothing below gets to scrub it, and a core file would write it
    // to disk.
    no_core_dumps();
    let (generated, witness) = match witness {
        WitnessFrom::File(path) => (None, Some(path)),
        WitnessFrom::Generate(generate) => (Some(Scrubbed(generate()?)), None),
    };
    let pk = ProvingKey::load(zkey).with_context(|| format!("loading {}", zkey.display()))?;
    let n_public = pk.n_public;
    // `n_public` comes from the zkey and decides how much of the witness is written to
    // public.json. A key that overstates it publishes private wires, and its own vk agrees
    // with it, so self-verify cannot tell. An independent vkey can: its IC has exactly one
    // point per public input plus the constant wire.
    let vk = vkey
        .as_deref()
        .map(|path| {
            VerifyingKey::from_json(path).with_context(|| format!("loading {}", path.display()))
        })
        .transpose()?;
    match &vk {
        Some(vk) => anyhow::ensure!(
            vk.ic.len() == n_public + 1,
            "the zkey says {n_public} public inputs and the vkey says {}; they are not the \
             same circuit, and proving would write {} to public.json",
            vk.ic.len() - 1,
            if n_public + 1 > vk.ic.len() {
                "private wires"
            } else {
                "too few signals"
            }
        ),
        // With no vkey, the one lie that can be refused without one: a key that calls every
        // wire public would write the whole witness out.
        None => anyhow::ensure!(
            n_public + 1 < pk.n_vars,
            "the zkey declares all {} witness wires public, so public.json would hold the \
             entire witness; pass --vkey to confirm the circuit really has no private inputs",
            pk.n_vars - 1
        ),
    }
    let owned = match (generated, witness) {
        (Some(w), _) => {
            // A generator built from another circuit, which a .wtns cannot be checked for
            // until the prover trips over it.
            anyhow::ensure!(
                w.0.len() == pk.n_vars,
                "the witness generator produced {} wires and the zkey has {}: they are not \
                 the same circuit",
                w.0.len(),
                pk.n_vars
            );
            w
        }
        (None, Some(path)) => Scrubbed(
            Witness::load(path)
                .with_context(|| format!("loading {}", path.display()))?
                .0,
        ),
        (None, None) => unreachable!("a witness is either a file or generated"),
    };
    let w: &[Fr] = &owned.0;
    // Checked here, not at the slice below, so a witness that does not match the key costs
    // the load and nothing else. On a large circuit the proof is tens of seconds.
    anyhow::ensure!(
        w.len() > n_public,
        "witness has {} entries, too short for {n_public} public signals",
        w.len()
    );
    let circuit = prover.prepare(pk)?;

    // Stage 10's blinders come from the OS CSPRNG. Nothing on this command offers a seed
    // override: a reused (r, s) across two proofs of different witnesses leaks the witness.
    // `snarkrs trace` does fix them, and writes no proof.json for exactly that reason.
    let mut rng = ark_std::rand::thread_rng();
    let mut t = StageTimings::default();
    let started = std::time::Instant::now();
    // Verify before writing, not after, so a bad proof never reaches the filesystem where
    // something downstream might pick it up. `prove` does it, and validates the proof's
    // points first; see its docs for why that check is also what keeps a hostile key from
    // reading the witness out of `C`.
    // Set when the proof came from the CPU fallback, so --stage-timings names the backend
    // that actually produced the numbers it prints.
    let fell_back = std::cell::Cell::new(false);
    let proof = if self_verify {
        let attempt = if fallback {
            snarkrs_cli::fallback::prove_with_fallback(
                circuit.as_ref(),
                w,
                &mut rng,
                &mut t,
                || {
                    // The key moved into the accelerator at prepare, so the CPU loads its
                    // own. Under --constant-work the fallback keeps the promise too.
                    let pk = ProvingKey::load(zkey)?;
                    Ok(backend_of(BackendKind::Cpu)?.prepare(pk)?)
                },
                &mut |step| {
                    use snarkrs_cli::fallback::Fallback::*;
                    if matches!(step, Cpu { .. }) {
                        fell_back.set(true);
                    }
                    match step {
                        Retry { backend, cause } => eprintln!(
                            "warning: the {backend} proof {cause}; proving again on {backend}"
                        ),
                        Cpu { backend, cause } => eprintln!(
                            "warning: the second {backend} proof {cause}; proving on the cpu"
                        ),
                        DeviceSuspect { backend } => eprintln!(
                            "warning: the cpu proof verified, so {backend} failed the same \
                             proof twice; treat that device as suspect"
                        ),
                    }
                },
            )
        } else {
            prove(circuit.as_ref(), w, &mut rng, &mut t).map_err(Into::into)
        };
        attempt.map_err(|e| {
            e.context(
                "Nothing has been written. Re-run with --self-verify=false to write it anyway.",
            )
        })?
    } else {
        prove_unchecked(circuit.as_ref(), w, &mut rng, &mut t)?
    };
    let elapsed = started.elapsed();

    // The public signals are the witness prefix, which is what snarkjs publishes. Taken
    // from the witness rather than copied from an existing public.json so that `prove`
    // needs nothing but the zkey and the witness.
    let public = &w[1..=n_public];
    if let (true, Some(vk)) = (self_verify, &vk) {
        verify(vk, public, &proof).map_err(|e| {
            anyhow::anyhow!(
                "the proof verifies against the zkey's own key but not against --vkey ({e}). \
                 Nothing has been written. The two keys disagree, or the zkey was misread."
            )
        })?;
    }

    json::write_proof(proof_out, &proof)?;
    json::write_public(public_out, public)?;
    // Written out, so the in-memory copy has nothing left to do. `Scrubbed` zeroes it as
    // it drops, and the backend's own scratch (H, stage 0's B and C) is zeroed by
    // `snarkrs-groth16` as it drops.

    if stage_timings {
        let total = elapsed.as_micros() as u64;
        let known =
            t.gather_us + t.ntt_us + t.pointwise_us + t.msm_us + t.assemble_us + t.verify_us;
        match fell_back.get() {
            true => println!(
                "backend        cpu (fallback from {})",
                circuit.backend_name()
            ),
            false => println!("backend        {}", circuit.backend_name()),
        }
        println!("domain size    {}", circuit.domain_size());
        println!("gather      us {:>10}", t.gather_us);
        println!("ntt         us {:>10}", t.ntt_us);
        println!("pointwise   us {:>10}", t.pointwise_us);
        println!("msm         us {:>10}", t.msm_us);
        println!("assemble    us {:>10}", t.assemble_us);
        println!("verify      us {:>10}", t.verify_us);
        // Attributed and total are printed separately rather than one being derived from
        // the other, so any stage that forgets to report shows up as a gap.
        println!("attributed  us {:>10}", known);
        println!("total       us {:>10}", total);
        println!("proved in {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    }
    Ok(())
}

/// No core file for this process, so a crash while the witness is in memory cannot write it
/// to disk. On Linux the process is also marked non-dumpable, which additionally stops
/// another process of the same user from attaching to it or reading `/proc/<pid>/mem`.
/// Best effort: failures are ignored, because refusing to prove would not make anything
/// safer.
fn no_core_dumps() {
    #[cfg(unix)]
    {
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `zero` is a valid, initialised `rlimit` that outlives the call.
        unsafe {
            libc::setrlimit(libc::RLIMIT_CORE, &zero);
        }
        #[cfg(target_os = "linux")]
        // SAFETY: PR_SET_DUMPABLE takes an integer argument and no pointers.
        unsafe {
            libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0);
        }
    }
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

/// `groth16 verify`: the verdict on an INFO or ERROR line and in the exit code, worded as
/// `groth16_verify.js` words it. A file that cannot be read is an error, not a verdict.
fn run_verify(vkey: &Path, public: &Path, proof: &Path) -> Result<u8> {
    let vk =
        VerifyingKey::from_json(vkey).with_context(|| format!("loading {}", vkey.display()))?;
    let public = json::read_public(public)?;
    let proof = json::read_proof(proof)?;
    match verify(&vk, &public, &proof) {
        Ok(()) => {
            log::info("OK!");
            Ok(0)
        }
        Err(VerifyError::PairingFailed) => {
            log::error("Invalid proof");
            Ok(1)
        }
        Err(VerifyError::InvalidProof(what)) => {
            log::error("Proof commitments are not valid.");
            log::debug(format!("proof element {what}"));
            Ok(1)
        }
        Err(e) => {
            log::error(format!("Invalid proof: {e}"));
            Ok(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// clap checks a derive tree only when asked, and a clash between a global `--verbose`
    /// and a subcommand's flag would otherwise surface as a panic on some user's line.
    #[test]
    fn the_clap_tree_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    /// Every command the table marks as supported has to reach a clap subcommand, or the
    /// normaliser hands clap words it rejects with exit 99.
    #[test]
    fn every_supported_snarkjs_command_parses() {
        for c in cli::COMMANDS.iter().filter(|c| c.support == Support::Yes) {
            let mut line = vec!["snarkrs".to_string()];
            line.extend(c.words().iter().map(|w| w.to_string()));
            // The required positionals, so only the words are under test.
            let required = c
                .params()
                .split_whitespace()
                .filter(|p| p.starts_with('<'))
                .count();
            line.extend((0..required).map(|i| format!("p{i}")));
            if let Err(e) = Cli::try_parse_from(&line) {
                panic!("{line:?}: {e}");
            }
        }
    }

    /// Every seam needs a CPU implementation under the same spelling, because
    /// `--backend cpu` is the file every accelerated run is `cmp`'d against and the only
    /// one already held against snarkjs byte for byte.
    #[test]
    fn every_ceremony_seam_has_a_cpu_backend() {
        assert_eq!(msm_backend(BackendKind::Cpu).unwrap().name(), "cpu");
        assert_eq!(fft_backend(BackendKind::Cpu).unwrap().name(), "cpu");
        assert_eq!(key_backend(BackendKind::Cpu).unwrap().name(), "cpu");
    }

    fn backend_error<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("selected an unavailable ceremony backend"),
            Err(e) => e.to_string(),
        }
    }

    /// All three real constructors must reject WebGPU, even when its feature is on.
    #[test]
    fn wgpu_rejects_every_ceremony_seam() {
        for e in [
            backend_error(msm_backend(BackendKind::Wgpu)),
            backend_error(fft_backend(BackendKind::Wgpu)),
            backend_error(key_backend(BackendKind::Wgpu)),
        ] {
            assert!(
                e.contains("backend `wgpu` is unavailable for the ceremony"),
                "{e}"
            );
            assert!(e.contains("--backend cpu"), "{e}");
        }
    }

    #[test]
    fn cuda_rejects_ceremony_msm_and_key_scaling() {
        for e in [
            backend_error(msm_backend(BackendKind::Cuda)),
            backend_error(key_backend(BackendKind::Cuda)),
        ] {
            assert!(e.contains("backend `cuda` is unavailable"), "{e}");
            if cfg!(feature = "cuda") {
                assert!(e.contains("MSM and apply-key"), "{e}");
                assert!(e.contains("no CUDA kernels"), "{e}");
                assert!(!e.contains("WITHOUT"), "{e}");
            } else {
                assert!(e.contains("WITHOUT the `cuda` feature"), "{e}");
                assert!(e.contains("--features cuda"), "{e}");
            }
        }
    }

    #[test]
    fn cuda_group_fft_selection_respects_the_feature() {
        if cfg!(feature = "cuda") {
            // Selection only: this test must not load a CUDA driver.
            assert!(matches!(
                ceremony_backend_kind(BackendKind::Cuda, CeremonySeam::GroupFft),
                Ok(BackendKind::Cuda)
            ));
        } else {
            let e = backend_error(fft_backend(BackendKind::Cuda));
            assert!(e.contains("WITHOUT the `cuda` feature"), "{e}");
            assert!(e.contains("--features cuda"), "{e}");
        }
    }

    #[test]
    fn supported_ceremony_selection_keeps_the_requested_backend() {
        for seam in [
            CeremonySeam::Msm,
            CeremonySeam::GroupFft,
            CeremonySeam::KeyScale,
        ] {
            assert!(matches!(
                ceremony_backend_kind(BackendKind::Cpu, seam),
                Ok(BackendKind::Cpu)
            ));
            if cfg!(all(feature = "metal", target_os = "macos")) {
                assert!(matches!(
                    ceremony_backend_kind(BackendKind::Metal, seam),
                    Ok(BackendKind::Metal)
                ));
            } else {
                assert!(ceremony_backend_kind(BackendKind::Metal, seam).is_err());
            }
        }
    }

    /// Exercise clap and the command handlers, not just the capability selector.
    #[test]
    fn unsupported_ceremony_commands_fail_before_io_or_entropy_prompt() {
        let dir =
            std::env::temp_dir().join(format!("snarkrs-cli-capabilities-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("output");
        for kind in [BackendKind::Wgpu, BackendKind::Cuda, BackendKind::Metal] {
            for (seam, words) in [
                (CeremonySeam::Msm, &["groth16", "setup"][..]),
                (
                    CeremonySeam::GroupFft,
                    &["powersoftau", "prepare", "phase2"][..],
                ),
                (CeremonySeam::KeyScale, &["zkey", "contribute"][..]),
            ] {
                let unavailable = match kind {
                    BackendKind::Wgpu => true,
                    BackendKind::Cuda => {
                        !cfg!(feature = "cuda") || !matches!(seam, CeremonySeam::GroupFft)
                    }
                    BackendKind::Metal => !cfg!(all(feature = "metal", target_os = "macos")),
                    BackendKind::Cpu => false,
                };
                if !unavailable {
                    continue;
                }
                let mut args: Vec<std::ffi::OsString> = std::iter::once("snarkrs")
                    .chain(words.iter().copied())
                    .map(Into::into)
                    .collect();
                args.push(dir.join("missing-input").into_os_string());
                if matches!(seam, CeremonySeam::Msm) {
                    args.push(dir.join("missing-ptau").into_os_string());
                }
                args.push(out.as_os_str().to_owned());
                // If selection regresses, the test must still never prompt on stdin.
                if matches!(seam, CeremonySeam::KeyScale) {
                    args.push("--entropy=test".into());
                }
                args.extend(["--backend".into(), kind.as_str().into()]);
                let cli = Cli::try_parse_from(args).unwrap();
                let e = backend_error(run(cli.cmd));
                assert!(
                    e.contains(&format!("backend `{}` is unavailable", kind.as_str())),
                    "{seam:?}: {e}"
                );
                assert!(!out.exists(), "{seam:?}: unsupported backend wrote output");
            }
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Feature and target failures need different fixes, and both must name Metal.
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn unavailable_metal_reports_the_feature_or_target() {
        for e in [
            backend_error(msm_backend(BackendKind::Metal)),
            backend_error(fft_backend(BackendKind::Metal)),
            backend_error(key_backend(BackendKind::Metal)),
        ] {
            assert!(e.contains("backend `metal` is unavailable"), "{e}");
            if cfg!(feature = "metal") {
                assert!(e.contains("not macOS"), "{e}");
            } else {
                assert!(e.contains("WITHOUT the `metal` feature"), "{e}");
                assert!(e.contains("--features metal"), "{e}");
            }
        }
    }
}
