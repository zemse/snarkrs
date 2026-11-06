//! Size up a gnark key and witness: the dump header, then a timed full load of each file
//! given, checked against each other.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark --example inspect -- <pk.dump> [vk.bin] [witness.bin]
//! ```
//!
//! Peak RSS is the caller's to measure, `/usr/bin/time -l` on macOS.

use snarkrs_gnark::pk::DumpHeader;
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let Some(pk_path) = args.first() else {
        return Err("usage: inspect <pk.dump> [vk.bin] [witness.bin]".into());
    };

    let t = Instant::now();
    let header = DumpHeader::read(pk_path)?;
    println!("header ({:.1} ms): {header:#?}", ms(t));
    println!(
        "public wires, constant included: {}",
        header.n_public_wires()
    );

    let t = Instant::now();
    let pk = GnarkProvingKey::read_dump(pk_path)?;
    println!("proving key decoded and checked in {:.0} ms", ms(t));

    if let Some(vk_path) = args.get(1) {
        let vk = GnarkVerifyingKey::read(vk_path)?;
        println!("verifying key: {} K points", vk.k.len());
        let same = vk.alpha_g1 == pk.alpha_g1
            && vk.beta_g1 == pk.beta_g1
            && vk.delta_g1 == pk.delta_g1
            && vk.beta_g2 == pk.beta_g2
            && vk.delta_g2 == pk.delta_g2
            && vk.k.len() == header.n_public_wires();
        println!("verifying key matches the proving key: {same}");
    }

    if let Some(w_path) = args.get(2) {
        let t = Instant::now();
        let w = GnarkWitness::read_dump(w_path)?;
        println!(
            "witness decoded in {:.0} ms: {} wires, {} constraint rows",
            ms(t),
            w.w().len(),
            w.n_constraints()
        );
        println!(
            "witness fits the key: wires {}, rows within the domain {}",
            w.w().len() == header.n_wires,
            w.n_constraints() <= header.domain_size
        );
    }
    Ok(())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
