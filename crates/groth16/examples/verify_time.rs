//! Time `verify` in process, for proofs too fast to time through the CLI.
//!
//! `snarkrs groth16 verify` is one pairing check behind a process start and three JSON
//! parses, which together outweigh the check. This loads the files once, checks that the
//! proof verifies and that a tampered public signal does not, then times `verify` alone.
//!
//! `cargo run --release -p snarkrs-groth16 --example verify_time -- REPS DIR...`
//!
//! Each DIR holds vkey.json, public.json and proof.json. One line per DIR:
//! `name reps median_us min_us max_us`.

use std::{error::Error, path::Path, time::Instant};

use snarkrs_field::One;
use snarkrs_formats::VerifyingKey;
use snarkrs_groth16::{
    json::{proof_from_str, public_from_str},
    verify::verify,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let reps: usize = args
        .next()
        .ok_or("usage: verify_time REPS DIR...")?
        .parse()?;
    for dir in args {
        let dir = Path::new(&dir);
        let vk = VerifyingKey::from_json(&dir.join("vkey.json"))?;
        let public = public_from_str(&std::fs::read_to_string(dir.join("public.json"))?)?;
        let proof = proof_from_str(&std::fs::read_to_string(dir.join("proof.json"))?)?;
        verify(&vk, &public, &proof).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut bad = public.clone();
        if let Some(x) = bad.first_mut() {
            *x += snarkrs_field::Fr::one();
            if verify(&vk, &bad, &proof).is_ok() {
                return Err(format!("{}: a tampered public signal verified", dir.display()).into());
            }
        }
        for _ in 0..reps.div_ceil(10) {
            verify(&vk, &public, &proof)?;
        }
        let mut us: Vec<f64> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                let ok = verify(&vk, &public, &proof).is_ok();
                let e = t.elapsed().as_secs_f64() * 1e6;
                assert!(ok, "{}: verify failed mid-run", dir.display());
                e
            })
            .collect();
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = us.len() / 2;
        let median = if us.len() % 2 == 1 {
            us[mid]
        } else {
            (us[mid - 1] + us[mid]) / 2.0
        };
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        println!(
            "{name} {reps} {median:.2} {:.2} {:.2}",
            us[0],
            us[us.len() - 1]
        );
    }
    Ok(())
}
