//! GNK-06: circom zkeys to gnark's dump and verifying key. snarkrs' gnark path proves the
//! dump from A.w and B.w rows of the zkey's section 4, and the proof passes snarkrs' and
//! snarkjs' verifiers against the original `vkey.json`, and gnark's `groth16.Verify` against
//! the exported `vk.bin`. And a dump read and written again is gnark's bytes, through a zkey
//! and back included.

mod common;

use ark_std::rand::{rngs::StdRng, SeedableRng};
use common::*;
use snarkrs_ceremony::CpuGroupFft;
use snarkrs_formats::wtns::Witness;
use snarkrs_formats::{ProvingKey, VerifyingKey};
use snarkrs_gnark::proof::write_public;
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_gnark_convert::proof::{gnark_proof, gnark_vk};
use snarkrs_gnark_convert::to_gnark::{gnark_keys, write_dump};
use snarkrs_gnark_convert::zkey_rows;
use snarkrs_gnark_solver::GnarkR1cs;
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::json::{proof_to_string, public_from_str};
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, StageTimings};

#[test]
fn a_dump_rewrites_to_gnarks_bytes() {
    let bytes = std::fs::read(toy("pk.dump")).unwrap();
    let pk = GnarkProvingKey::from_bytes(&bytes).unwrap();
    let mut out = Vec::new();
    write_dump(&pk, &mut out).unwrap();
    assert!(out == bytes, "the rewritten dump differs from gnark's");
}

/// gnark to zkey to gnark gives gnark's dump back byte for byte: every query carries over
/// as it is, and the forward transform recovers `G1.Z` exactly from the converted H.
#[test]
fn gnark_to_zkey_and_back_is_the_identity() {
    let bytes = std::fs::read(toy("pk.dump")).unwrap();
    let pk = GnarkProvingKey::from_bytes(&bytes).unwrap();
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let r1cs = GnarkR1cs::read(&toy("circuit.r1cs")).unwrap();
    let out = scratch("round-trip");
    let zkey = out.join("toy.zkey");
    snarkrs_gnark_convert::to_zkey::write_zkey(&pk, &vk, &r1cs, &zkey, &CpuGroupFft).unwrap();
    let (back, back_vk) = gnark_keys(&ProvingKey::load(&zkey).unwrap(), &CpuGroupFft).unwrap();
    let mut dump = Vec::new();
    write_dump(&back, &mut dump).unwrap();
    assert!(
        dump == bytes,
        "the dump through a zkey differs from gnark's"
    );
    assert_eq!(
        gnark_vk(&back_vk, true),
        std::fs::read(toy("vk.bin")).unwrap()
    );
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn circom_zkeys_prove_on_the_gnark_path() {
    for name in ["tiny_mul", "js_1x1_d8"] {
        let files = ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"];
        let Some(dir) = artifact(name, &files) else {
            continue;
        };
        let zkey = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let [aw, bw] = zkey_rows(&zkey, &w);
        let (pk, vk) = gnark_keys(&zkey, &CpuGroupFft).unwrap();
        drop(zkey);

        let out = scratch(&format!("from-zkey-{name}"));
        let (dump, vk_bin) = (out.join("pk.dump"), out.join("vk.bin"));
        write_dump(
            &pk,
            std::io::BufWriter::new(std::fs::File::create(&dump).unwrap()),
        )
        .unwrap();
        std::fs::write(&vk_bin, gnark_vk(&vk, true)).unwrap();

        // Back through snarkrs-gnark's checked readers, as any gnark key is proved.
        let vk = GnarkVerifyingKey::read(&vk_bin).unwrap();
        let key = GnarkProvingKey::read_dump(&dump)
            .unwrap()
            .into_proving_key(&vk)
            .unwrap();
        let circuit = CpuBackend::new().prepare(key).unwrap();
        let witness = GnarkWitness::from_parts(&w, &aw, &bw).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let proof =
            snarkrs_gnark::prove(&*circuit, &witness, &mut rng, &mut StageTimings::default())
                .unwrap();

        // The original key's verifiers.
        let vkey = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let public_json = std::fs::read_to_string(dir.join("public.json")).unwrap();
        let public = public_from_str(&public_json).unwrap();
        verify(&vkey, &public, &proof).unwrap();
        let proof_json = out.join("proof.json");
        std::fs::write(&proof_json, proof_to_string(&proof)).unwrap();
        if let Some(ok) = snarkjs_verify(
            &dir.join("vkey.json"),
            &dir.join("public.json"),
            &proof_json,
        ) {
            assert!(ok, "{name}: snarkjs refused the gnark-path proof");
        }
        // And gnark's, against the exported key.
        let (public_bin, proof_bin) = (out.join("public.bin"), out.join("proof.bin"));
        std::fs::write(&public_bin, write_public(&public)).unwrap();
        std::fs::write(&proof_bin, gnark_proof(&proof, false)).unwrap();
        if let Some(ok) = gnark_verify(&vk_bin, &public_bin, &proof_bin) {
            assert!(ok, "{name}: gnark refused the gnark-path proof");
        }
        std::fs::remove_dir_all(out).unwrap();
    }
}
