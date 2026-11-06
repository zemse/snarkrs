//! Prove a gnark witness dump against a gnark key on the CPU, check the proof with snarkrs'
//! verifier, and write it for gnark's.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark --example prove -- \
//!     <pk.dump> <vk.bin> <witness.bin> <out dir>
//! ```
//!
//! Writes `<out dir>/proof.bin` (gnark `Proof.WriteRawTo`) and `<out dir>/public.bin`, which
//! `go run . verify <vk.bin> <out dir>/public.bin <out dir>/proof.bin` in `test-support/gnark`
//! checks with gnark's `groth16.Verify`.

use snarkrs_gnark::proof::{write_proof, write_public};
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::{Backend, StageTimings};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [pk_path, vk_path, w_path, out] = &args[..] else {
        return Err("usage: prove <pk.dump> <vk.bin> <witness.bin> <out dir>".into());
    };

    let t = Instant::now();
    let vk = GnarkVerifyingKey::read(vk_path)?;
    let pk = GnarkProvingKey::read_dump(pk_path)?.into_proving_key(&vk)?;
    println!("key loaded in {:.0} ms", ms(t));

    let t = Instant::now();
    let circuit = CpuBackend::new().prepare(pk)?;
    println!("prepared in {:.0} ms", ms(t));

    let t = Instant::now();
    let witness = GnarkWitness::read_dump(w_path)?;
    println!("witness loaded in {:.0} ms", ms(t));

    let mut timings = StageTimings::default();
    let t = Instant::now();
    let proof = snarkrs_gnark::prove(
        &*circuit,
        &witness,
        &mut ark_std::rand::thread_rng(),
        &mut timings,
    )?;
    println!("proved and self-verified in {:.0} ms: {timings:?}", ms(t));

    let public = witness
        .public(vk.n_public())
        .ok_or("the witness is shorter than the public inputs")?;
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("proof.bin"), write_proof(&proof))?;
    std::fs::write(out.join("public.bin"), write_public(public))?;
    println!("wrote {}", out.display());
    Ok(())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
