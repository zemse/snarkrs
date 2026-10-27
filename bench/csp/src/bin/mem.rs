//! One proof, one process, then exit: the shape `/usr/bin/time -l` can measure.
//!
//! Peak RSS is a property of the whole process, so anything else this binary did would be
//! folded into the number. It therefore parses its arguments, proves once and exits. The
//! prove is the checked one, so the proof is self-verified inside the measured process,
//! the same as in the timing driver's `prove_ms`.

use anyhow::Result;
use clap::Parser;
use snarkrs_csp::{circuits, inputs, Backend, Target, Variant};
use snarkrs_formats::wtns::Witness;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    target: Target,
    #[arg(long)]
    input_size: usize,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    #[arg(long, default_value = "bench/artifacts/csp")]
    artifacts: PathBuf,
    /// Accept a key with `gamma_g2 == delta_g2`, as `snarkrs-csp bench` does.
    #[arg(long)]
    allow_unblinded_zkey: bool,
}

fn main() -> Result<()> {
    // The published peak is this process's, so it runs the way `snarkrs groth16 prove`
    // does.
    snarkrs_groth16::malloc::reexec_without_large_cache();
    let args = Args::parse();
    let v = Variant {
        target: args.target,
        input_size: args.input_size,
    };

    let wtns = circuits::witness(v, inputs::json(v.target, v.input_size))?;
    let (pk, key) = snarkrs_csp::load_key(&v.zkey(&args.artifacts), args.allow_unblinded_zkey)?;
    let witness = Witness::from_bytes(wtns)?.0;
    let circuit = snarkrs_csp::make_backend(args.backend)?.prepare(pk)?;
    let proof = key.prove(
        circuit.as_ref(),
        &witness,
        &mut snarkrs_groth16::StageTimings::default(),
    )?;
    std::hint::black_box(&proof);
    Ok(())
}
