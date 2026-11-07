//! Time the H basis change both ways on a domain of `2^k`, and check the round trip.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark-convert --example basis -- <k>
//! ```
//!
//! The points are multiples of the generator, which costs the transform what any key's
//! points cost it.

use rayon::prelude::*;
use snarkrs_ceremony::CpuGroupFft;
use snarkrs_field::*;
use snarkrs_gnark_convert::basis::{gnark_z_from_zkey_h, zkey_h_from_gnark_z};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k: u32 = std::env::args().nth(1).ok_or("usage: basis <k>")?.parse()?;
    let n = 1usize << k;
    let chunk = 1 << 12;
    let g = G1Affine::generator();
    let h: Vec<G1Affine> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .flat_map_iter(|c| {
            let mut p = g * Fr::from((c * chunk + 1) as u64);
            let mut run = Vec::with_capacity(chunk);
            for _ in 0..chunk.min(n) {
                run.push(p);
                p += g;
            }
            G1Projective::normalize_batch(&run)
        })
        .collect();

    let t = Instant::now();
    let z = gnark_z_from_zkey_h(&h, &CpuGroupFft)?;
    println!(
        "2^{k}: zkey H to gnark Z in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let back = zkey_h_from_gnark_z(&z, &CpuGroupFft)?;
    println!(
        "2^{k}: gnark Z to zkey H in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    if gnark_z_from_zkey_h(&back, &CpuGroupFft)? != z {
        return Err("the round trip does not give gnark's Z back".into());
    }
    println!(
        "2^{k}: round trip checked in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
