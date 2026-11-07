//! Re-encode a proof, its public inputs and its verifying key between snarkjs and gnark.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark-convert --example proof -- to-gnark \
//!     <verification_key.json> <proof.json> <public.json> <out dir> [<circuit.zkey>]
//! cargo run --release -p snarkrs-gnark-convert --example proof -- to-snarkjs \
//!     <vk.bin> <proof.bin> <public> <out dir>
//! ```
//!
//! `to-gnark` writes `vk.bin` (`WriteTo`, compressed), `vk.raw.bin` (`WriteRawTo`),
//! `proof.bin` (`WriteRawTo`), `proof.compressed.bin` (`WriteTo`), `public.bin` (32-byte
//! big-endian scalars, the `test-support/gnark` layout) and `public.wtns` (gnark's
//! `witness.MarshalBinary` public witness). With the zkey, `vk_beta_1` and `vk_delta_1`,
//! which `verification_key.json` leaves out and gnark's verifier never reads, come from its
//! header; without it they are written as infinity. `to-snarkjs` takes either public layout
//! and writes `verification_key.json`, `proof.json` and `public.json`. Both check the proof
//! against the key first and refuse to convert one that does not verify.

use snarkrs_formats::{ProvingKey, VerifyingKey};
use snarkrs_gnark::proof::{read_proof, read_public, write_public};
use snarkrs_gnark::GnarkVerifyingKey;
use snarkrs_gnark_convert::proof::*;
use snarkrs_groth16::json::{proof_from_str, proof_to_string, public_from_str, public_to_string};
use snarkrs_groth16::verify::verify;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<PathBuf> = args.iter().skip(1).map(PathBuf::from).collect();
    match (args.first().map(String::as_str), &paths[..]) {
        (Some("to-gnark"), [vk, proof, public, out, zkey @ ..]) if zkey.len() <= 1 => {
            let vk = VerifyingKey::from_json(vk)?;
            let proof = proof_from_str(&std::fs::read_to_string(proof)?)?;
            let public = public_from_str(&std::fs::read_to_string(public)?)?;
            verify(&vk, &public, &proof)?;
            let (beta, delta) = match zkey.first() {
                Some(z) => {
                    let z = ProvingKey::load(z)?;
                    if z.vk.ic != vk.ic || z.vk.delta_g2 != vk.delta_g2 {
                        return Err("the zkey is not the verifying key's".into());
                    }
                    (Some(z.beta_g1), Some(z.delta_g1))
                }
                None => (None, None),
            };
            let gvk = gnark_vk_from_snarkjs(&vk, beta, delta);
            std::fs::create_dir_all(out)?;
            std::fs::write(out.join("vk.bin"), gnark_vk(&gvk, true))?;
            std::fs::write(out.join("vk.raw.bin"), gnark_vk(&gvk, false))?;
            std::fs::write(out.join("proof.bin"), gnark_proof(&proof, false))?;
            std::fs::write(out.join("proof.compressed.bin"), gnark_proof(&proof, true))?;
            std::fs::write(out.join("public.bin"), write_public(&public))?;
            std::fs::write(out.join("public.wtns"), gnark_public_witness(&public))?;
        }
        (Some("to-snarkjs"), [vk, proof, public, out]) => {
            let vk = GnarkVerifyingKey::read(vk)?;
            let proof = read_proof(&std::fs::read(proof)?)?;
            let raw = std::fs::read(public)?;
            // A public witness has a 12-byte header, so it is never whole scalars.
            let public = if raw.len() % 32 == 12 {
                read_gnark_public_witness(&raw)?
            } else {
                read_public(&raw)?
            };
            verify(&vk.to_verifying_key(), &public, &proof)?;
            std::fs::create_dir_all(out)?;
            std::fs::write(out.join("verification_key.json"), snarkjs_vk(&vk))?;
            std::fs::write(out.join("proof.json"), proof_to_string(&proof))?;
            std::fs::write(out.join("public.json"), public_to_string(&public))?;
        }
        _ => {
            return Err(
                "usage: proof to-gnark <verification_key.json> <proof.json> \
                        <public.json> <out dir> [<circuit.zkey>]\n       proof to-snarkjs \
                        <vk.bin> <proof.bin> <public> <out dir>"
                    .into(),
            )
        }
    }
    println!("ok");
    Ok(())
}
