//! GNK-08: proofs, public inputs and verifying keys across the two libraries. A snarkjs
//! proof converted to gnark's forms passes gnark's `groth16.Verify`, and a gnark proof
//! converted to snarkjs JSON passes snarkrs' verifier and `snarkjs groth16 verify`.

mod common;

use common::*;
use snarkrs_field::*;
use snarkrs_formats::{ProvingKey, VerifyingKey};
use snarkrs_gnark::proof::{read_proof, read_public, write_public};
use snarkrs_gnark::GnarkVerifyingKey;
use snarkrs_gnark_convert::proof::*;
use snarkrs_groth16::json::{proof_from_str, proof_to_string, public_from_str, public_to_string};
use snarkrs_groth16::verify::verify;

#[test]
fn gnark_files_reencode_to_gnarks_own_bytes() {
    let vk_bytes = std::fs::read(toy("vk.bin")).unwrap();
    let vk = GnarkVerifyingKey::from_bytes(&vk_bytes).unwrap();
    assert_eq!(gnark_vk(&vk, true), vk_bytes, "WriteTo");
    assert_eq!(
        GnarkVerifyingKey::from_bytes(&gnark_vk(&vk, false)).unwrap(),
        vk,
        "WriteRawTo"
    );
    let proof_bytes = std::fs::read(toy("proof.bin")).unwrap();
    let proof = read_proof(&proof_bytes).unwrap();
    assert_eq!(gnark_proof(&proof, false), proof_bytes, "WriteRawTo");
    let compressed = gnark_proof(&proof, true);
    assert_eq!(compressed.len(), COMPRESSED_PROOF_BYTES);
    assert_eq!(read_proof(&compressed).unwrap(), proof, "WriteTo");
}

/// Every artifact's `vkey.json` comes back byte for byte through gnark's key.
#[test]
fn a_snarkjs_vk_survives_gnarks_form() {
    for name in ["tiny_mul", "js_1x1_d8"] {
        let Some(dir) = artifact(name, &["vkey.json"]) else {
            continue;
        };
        let text = std::fs::read_to_string(dir.join("vkey.json")).unwrap();
        let vk = VerifyingKey::from_json_str(&text).unwrap();
        let gnark = gnark_vk_from_snarkjs(&vk, None, None);
        assert_eq!(snarkjs_vk(&gnark), text, "{name}");
    }
}

#[test]
fn a_snarkjs_proof_passes_gnark_verify() {
    for name in ["tiny_mul", "js_1x1_d8"] {
        let files = ["circuit.zkey", "vkey.json", "proof.json", "public.json"];
        let Some(dir) = artifact(name, &files) else {
            continue;
        };
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
        let vk = VerifyingKey::from_json_str(&read("vkey.json")).unwrap();
        let proof = proof_from_str(&read("proof.json")).unwrap();
        let public = public_from_str(&read("public.json")).unwrap();
        verify(&vk, &public, &proof).unwrap();
        let zkey = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();

        let out = scratch(&format!("to-gnark-{name}"));
        let write = |f: &str, b: &[u8]| {
            let p = out.join(f);
            std::fs::write(&p, b).unwrap();
            p
        };
        // The zkey's beta_g1 and delta_g1, compressed; the placeholders gnark never reads,
        // raw.
        let full = gnark_vk_from_snarkjs(&vk, Some(zkey.beta_g1), Some(zkey.delta_g1));
        let vks = [
            write("vk.bin", &gnark_vk(&full, true)),
            write(
                "vk.raw.bin",
                &gnark_vk(&gnark_vk_from_snarkjs(&vk, None, None), false),
            ),
        ];
        let proofs = [
            write("proof.bin", &gnark_proof(&proof, true)),
            write("proof.raw.bin", &gnark_proof(&proof, false)),
        ];
        let publics = [
            write("public.bin", &write_public(&public)),
            write("public.wtns", &gnark_public_witness(&public)),
        ];
        for vk in &vks {
            for p in &proofs {
                for x in &publics {
                    if let Some(ok) = gnark_verify(vk, x, p) {
                        assert!(ok, "{name}: gnark refused {vk:?} {x:?} {p:?}");
                    }
                }
            }
        }
        let mut wrong = public.clone();
        wrong[0] += Fr::ONE;
        let wrong = write("wrong.bin", &write_public(&wrong));
        if let Some(ok) = gnark_verify(&vks[0], &wrong, &proofs[0]) {
            assert!(!ok, "{name}: gnark accepted a wrong public input");
        }
        std::fs::remove_dir_all(out).unwrap();
    }
}

#[test]
fn a_gnark_proof_passes_snarkjs_verify() {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let proof = read_proof(&std::fs::read(toy("proof.bin")).unwrap()).unwrap();
    let public = read_public(&std::fs::read(toy("public.bin")).unwrap()).unwrap();

    let vk_json = snarkjs_vk(&vk);
    let (proof_json, public_json) = (proof_to_string(&proof), public_to_string(&public));
    // snarkrs' snarkjs path, from the JSON alone.
    verify(
        &VerifyingKey::from_json_str(&vk_json).unwrap(),
        &public_from_str(&public_json).unwrap(),
        &proof_from_str(&proof_json).unwrap(),
    )
    .unwrap();

    let out = scratch("to-snarkjs");
    let write = |f: &str, b: &str| {
        let p = out.join(f);
        std::fs::write(&p, b).unwrap();
        p
    };
    let (v, p, x) = (
        write("verification_key.json", &vk_json),
        write("proof.json", &proof_json),
        write("public.json", &public_json),
    );
    if let Some(ok) = snarkjs_verify(&v, &x, &p) {
        assert!(ok, "snarkjs refused gnark's proof");
    }
    let mut wrong = public.clone();
    wrong[1] += Fr::ONE;
    let w = write("wrong.json", &public_to_string(&wrong));
    assert!(verify(&vk.to_verifying_key(), &wrong, &proof).is_err());
    if let Some(ok) = snarkjs_verify(&v, &w, &p) {
        assert!(!ok, "snarkjs accepted a wrong public input");
    }
    // And gnark's own public witness of the same inputs reads back to them.
    assert_eq!(
        read_gnark_public_witness(&gnark_public_witness(&public)).unwrap(),
        public
    );
    std::fs::remove_dir_all(out).unwrap();
}
