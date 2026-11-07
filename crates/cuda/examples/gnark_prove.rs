//! Prove a gnark witness dump against a gnark key with [`GnarkProver`], and report what it
//! cost: construction by phase, the device budget against what the driver says was taken,
//! a cold proof and warm ones by stage, and the peak host RSS of each phase. Writes the last
//! proof for gnark's own verifier.
//!
//! ```text
//! cargo run --release -p snarkrs-cuda --features cuda --example gnark_prove -- \
//!     <pk.dump> <vk.bin> <witness.bin> <out dir> [warm proofs, default 3]
//! ```
//!
//! `G16_CUDA_DEVICE` picks the device (default 0), `G16_CUDA_MSM_CHUNK` the MSM run as a
//! power of two (default 22), and `G16_GNARK_WARM_UP=0` skips construction's warm-up pass,
//! so the first proof is the first launch of every kernel. Then, in `test-support/gnark`:
//!
//! ```text
//! go run . verify <vk.bin> <out dir>/public.bin <out dir>/proof.bin
//! ```
//!
//! The witness is read whole before the first proof, as the caller's solver buffers would
//! be, and handed to `prove` borrowed. Host RSS figures are Linux's `VmHWM` with the peak
//! reset between phases (`/proc/self/clear_refs`), so each is that phase's own peak; on any
//! other system they print as unavailable.

use snarkrs_cuda::{required_device_bytes, Cuda, GnarkProver, GnarkProverConfig};
use snarkrs_gnark::pk::DumpHeader;
use snarkrs_gnark::proof::{write_proof, write_public};
use snarkrs_gnark::{GnarkVerifyingKey, GnarkWitness};
use snarkrs_groth16::StageTimings;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (paths, warm) = match args.len() {
        4 => (&args[..4], 3),
        5 => (&args[..4], args[4].parse::<usize>()?),
        _ => {
            return Err(
                "usage: gnark_prove <pk.dump> <vk.bin> <witness.bin> <out dir> [warm]".into(),
            )
        }
    };
    let [pk_path, vk_path, w_path, out] = [0, 1, 2, 3].map(|i| PathBuf::from(&paths[i]));
    let device = env_usize("G16_CUDA_DEVICE", 0)?;
    let msm_chunk = 1usize << env_usize("G16_CUDA_MSM_CHUNK", 22)?;

    let header = DumpHeader::read(&pk_path)?;
    let budget = required_device_bytes(&header, msm_chunk)?;
    println!(
        "key: domain 2^{}, {} wires, {} private; MSM runs of 2^{}",
        header.domain_size.trailing_zeros(),
        header.n_wires,
        header.g1_k,
        msm_chunk.trailing_zeros()
    );
    println!("budget: {} ({budget:?})", gib(budget.total));

    let free_before = Cuda::new_private(device)?.context().mem_get_info()?.0;
    reset_peak();
    let t = Instant::now();
    let vk = GnarkVerifyingKey::read(&vk_path)?;
    let config = GnarkProverConfig {
        device,
        msm_chunk,
        warm_up: env_usize("G16_GNARK_WARM_UP", 1)? != 0,
    };
    let prover = GnarkProver::open_with(&pk_path, &vk, config)?;
    let open_ms = ms(t);
    let (free_after, total) = prover.device_memory()?;
    println!(
        "constructed on {} in {open_ms:.0} ms: {:?}",
        prover.device_name(),
        prover.cost()
    );
    println!(
        "device: {} taken by construction, {} budgeted, {} of {} free now",
        gib(free_before.saturating_sub(free_after) as u64),
        gib(budget.total),
        gib(free_after as u64),
        gib(total as u64)
    );
    println!("host peak RSS, construction: {}", peak_rss());

    reset_peak();
    let t = Instant::now();
    let witness = GnarkWitness::read_dump(&w_path)?;
    println!(
        "witness: {} wires, {} rows, read in {:.0} ms; host RSS now {}",
        witness.w().len(),
        witness.n_constraints(),
        ms(t),
        rss()
    );
    let borrowed = GnarkWitness::from_parts(witness.w(), witness.aw(), witness.bw())?;

    let mut rng = ark_std::rand::thread_rng();
    let mut proof = None;
    for rep in 0..=warm {
        reset_peak();
        let mut timings = StageTimings::default();
        let t = Instant::now();
        proof = Some(prover.prove(&borrowed, &mut rng, &mut timings)?);
        let label = if rep == 0 { "cold" } else { "warm" };
        println!(
            "{label} proof {rep}: {:.0} ms, upload {:.1} ms, MSM device {:.1} ms; {timings:?}",
            ms(t),
            prover.circuit().last_witness_upload_us() as f64 / 1e3,
            prover.circuit().last_msm_device_us() as f64 / 1e3,
        );
        println!(
            "  host peak RSS {}; device {} free",
            peak_rss(),
            gib(prover.device_memory()?.0 as u64)
        );
    }

    let proof = proof.expect("at least the cold proof ran");
    let public = witness
        .public(vk.n_public())
        .ok_or("the witness is shorter than the public inputs")?;
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("proof.bin"), write_proof(&proof))?;
    std::fs::write(out.join("public.bin"), write_public(public))?;
    println!("wrote {}", out.display());
    Ok(())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(v) => Ok(v.trim().parse()?),
        Err(_) => Ok(default),
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// A field of `/proc/self/status`, in kB, where there is one.
fn status_kb(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(field))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn peak_rss() -> String {
    status_kb("VmHWM:").map_or("unavailable".into(), |kb| gib(kb * 1024))
}

fn rss() -> String {
    status_kb("VmRSS:").map_or("unavailable".into(), |kb| gib(kb * 1024))
}

/// Linux 4.0 and later reset `VmHWM` to the current RSS on a write of 5.
fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}
