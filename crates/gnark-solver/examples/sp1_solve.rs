//! Solve SP1's Groth16 circuit for one witness, timing the load and the solve.
//!
//! `cargo run --release -p snarkrs-gnark-solver --example sp1_solve -- CIRCUIT WITNESS [OUT]`
//!
//! CIRCUIT is `groth16_circuit.bin`, WITNESS the witness JSON SP1 writes, and OUT, when
//! given, receives the solution as gnark-crypto's `unsafe` dump of W, A and B.

use snarkrs_gnark_solver::{sp1, GnarkR1cs, Solver};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [circuit, witness, rest @ ..] = args.as_slice() else {
        return Err("usage: sp1_solve CIRCUIT WITNESS [OUT]".into());
    };

    let t = Instant::now();
    let r1cs = GnarkR1cs::read(circuit)?;
    eprintln!(
        "load {:.3} s: {} constraints, {} wires, {} instructions, {} levels",
        t.elapsed().as_secs_f64(),
        r1cs.nb_constraints(),
        r1cs.nb_wires(),
        r1cs.nb_instructions(),
        r1cs.nb_levels()
    );

    let t = Instant::now();
    let (public, secret) = sp1::Witness::from_json(&std::fs::read(witness)?)?.assignment()?;
    eprintln!("witness {:.3} s", t.elapsed().as_secs_f64());

    let solver = Solver::sp1();
    let t = Instant::now();
    let solution = solver.solve(&r1cs, &public, &secret)?;
    eprintln!("solve {:.3} s", t.elapsed().as_secs_f64());

    if let Some(out) = rest.first() {
        let t = Instant::now();
        solution.write_dump(std::io::BufWriter::new(std::fs::File::create(out)?))?;
        eprintln!("dump {:.3} s", t.elapsed().as_secs_f64());
    }
    Ok(())
}
