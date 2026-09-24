//! One proof, one process, then exit: the shape `/usr/bin/time -l` can measure.
//!
//! Peak RSS is a property of the whole process, so anything else this binary did would be
//! folded into the number. It therefore parses three arguments, proves once, and drops
//! the proof on the floor without verifying it - the bench driver already established
//! that the same proof verifies.

use anyhow::Result;
use clap::Parser;
use g16_csp::{circuits, inputs, Backend, Target, Variant};
use g16_zkey::{wtns::Witness, ProvingKey};
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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let v = Variant {
        target: args.target,
        input_size: args.input_size,
    };

    let wtns = circuits::witness(v, inputs::json(v.target, v.input_size))?;
    let pk = ProvingKey::load(&v.zkey(&args.artifacts))?;
    let witness = Witness::from_bytes(wtns)?.0;
    let circuit = g16_csp::make_backend(args.backend)?.prepare(pk)?;
    let proof = g16_core::prove::prove(
        circuit.as_ref(),
        &witness,
        &mut ark_std::rand::thread_rng(),
        &mut g16_core::StageTimings::default(),
    )?;
    std::hint::black_box(&proof);
    Ok(())
}
