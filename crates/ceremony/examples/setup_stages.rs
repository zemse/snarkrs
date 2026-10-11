//! Wall clock of `groth16 setup`'s stages, in process, through the same public functions
//! `setup` calls. No file is written; the composed points are checked to be on the curve.
//!
//! `cargo run --release -p snarkrs-ceremony --example setup_stages -- R1CS PTAU`
//!
//! One `stage ms` line per stage, in setup's order.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use snarkrs_ceremony::ptau::Ptau;
use snarkrs_ceremony::r1cs::R1cs;
use snarkrs_ceremony::setup::{
    compose_points_g1, compose_points_g2, hash_h_points, process_constraints, write_hs,
    LagrangeBlocks,
};
use snarkrs_ceremony::transcript::Transcript;
use snarkrs_msm::CpuMsm;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let r1cs_path = args.next().ok_or("usage: setup_stages R1CS PTAU")?;
    let ptau_path = args.next().ok_or("usage: setup_stages R1CS PTAU")?;
    let msm = CpuMsm::new();
    let mut lap = Instant::now();
    let mut stage = |name: &str| {
        println!("{name} {:.1}", lap.elapsed().as_secs_f64() * 1e3);
        lap = Instant::now();
    };

    let ptau = Ptau::open(Path::new(&ptau_path))?;
    let r1cs = R1cs::open(Path::new(&r1cs_path))?;
    let domain = r1cs.header().domain_size();
    stage("open");
    let acc = process_constraints(&r1cs, domain)?;
    stage("process_constraints");
    let bases = LagrangeBlocks::read(&ptau, domain)?;
    stage("lagrange_blocks");
    let mut on_curve = true;
    let ic = compose_points_g1(&acc.ic, &bases, &r1cs, &msm)?;
    on_curve &= ic.iter().all(|p| p.is_on_curve());
    stage("compose_ic");
    let hs = write_hs(&ptau, domain)?;
    on_curve &= hs.iter().all(|p| p.is_on_curve());
    stage("write_hs");
    let mut cs = Transcript::new();
    hash_h_points(&ptau, domain, &mut cs)?;
    stage("hash_h_points");
    for (name, slots) in [
        ("compose_c", &acc.c),
        ("compose_a", &acc.a),
        ("compose_b1", &acc.b1),
    ] {
        let pts = compose_points_g1(slots, &bases, &r1cs, &msm)?;
        on_curve &= pts.iter().all(|p| p.is_on_curve());
        cs.update_g1_all(&pts);
        stage(name);
    }
    let b2 = compose_points_g2(&acc.b2, &bases, &r1cs, &msm)?;
    on_curve &= b2.iter().all(|p| p.is_on_curve());
    cs.update_g2_all(&b2);
    stage("compose_b2");
    if !on_curve {
        return Err("a composed point is off the curve".into());
    }
    Ok(())
}
