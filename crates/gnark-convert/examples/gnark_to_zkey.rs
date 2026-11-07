//! A gnark key and its R1CS to a zkey, and optionally a gnark witness dump to the `.wtns`
//! that proves with it.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark-convert --example gnark_to_zkey -- \
//!     <pk.dump> <vk.bin> <circuit.r1cs> <out.zkey> [<witness.bin> <out.wtns>]
//! ```
//!
//! `snarkjs groth16 prove out.zkey out.wtns proof.json public.json` then proves, and the
//! `proof` example turns its output into gnark's files for gnark's verifier. Peak RSS is the
//! caller's to measure, `/usr/bin/time -l` on macOS. An `<out.zkey>` of `/dev/null` times the
//! conversion without the disk space for its output.

use snarkrs_ceremony::CpuGroupFft;
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_gnark_convert::to_zkey::{write_zkey, write_zkey_unstaged};
use snarkrs_gnark_solver::GnarkR1cs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let (pk_path, vk_path, r1cs_path, out, wtns) = match &args[..] {
        [pk, vk, r1cs, out] => (pk, vk, r1cs, out, None),
        [pk, vk, r1cs, out, w, wtns] => (pk, vk, r1cs, out, Some((w, wtns))),
        _ => {
            return Err(
                "usage: gnark_to_zkey <pk.dump> <vk.bin> <circuit.r1cs> <out.zkey> \
                        [<witness.bin> <out.wtns>]"
                    .into(),
            )
        }
    };

    let t = Instant::now();
    let vk = GnarkVerifyingKey::read(vk_path)?;
    let pk = GnarkProvingKey::read_dump(pk_path)?;
    println!("key loaded in {:.2} s", secs(t));
    let t = Instant::now();
    let r1cs = GnarkR1cs::read(r1cs_path)?;
    println!(
        "R1CS loaded in {:.2} s: {} constraints, {} wires",
        secs(t),
        r1cs.nb_constraints(),
        r1cs.nb_wires()
    );

    let t = Instant::now();
    let report = if out.as_os_str() == "/dev/null" {
        write_zkey_unstaged(&pk, &vk, &r1cs, out, &CpuGroupFft)?
    } else {
        write_zkey(&pk, &vk, &r1cs, out, &CpuGroupFft)?
    };
    println!("zkey written in {:.2} s: {report:?}", secs(t));

    if let Some((w, wtns)) = wtns {
        let t = Instant::now();
        let witness = GnarkWitness::read_dump(w)?;
        snarkrs_witness::write_wtns(wtns, witness.w())?;
        println!("wtns written in {:.2} s", secs(t));
    }
    Ok(())
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}
