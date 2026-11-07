//! GNK-07: the gnark toy key to a zkey. snarkrs' zkey path proves it from a `.wtns` of the
//! gnark witness, and the proof passes gnark's `groth16.Verify` against the original gnark
//! verifying key. snarkjs' own `groth16 prove` proves it too, and so does gnark's check.

mod common;

use ark_std::rand::{rngs::StdRng, SeedableRng};
use common::*;
use snarkrs_ceremony::CpuGroupFft;
use snarkrs_field::*;
use snarkrs_formats::ProvingKey;
use snarkrs_gnark::proof::write_public;
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_gnark_convert::proof::gnark_proof;
use snarkrs_gnark_convert::to_zkey::write_zkey;
use snarkrs_gnark_convert::{zkey_rows, ConvertError};
use snarkrs_gnark_solver::GnarkR1cs;
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::json::proof_from_str;
use snarkrs_groth16::{Backend, StageTimings};
use std::path::Path;

fn toy_keys() -> (GnarkProvingKey, GnarkVerifyingKey, GnarkR1cs) {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
    let r1cs = GnarkR1cs::read(&toy("circuit.r1cs")).unwrap();
    (pk, vk, r1cs)
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn the_toy_zkey_proves_against_gnarks_own_vk() {
    let (pk, vk, r1cs) = toy_keys();
    let out = scratch("toy-zkey");
    let zkey = out.join("toy.zkey");
    let report = write_zkey(&pk, &vk, &r1cs, &zkey, &CpuGroupFft).unwrap();
    assert_eq!(report.n_constraints, 2071);
    assert_eq!((report.n_public, report.domain_size), (4, 4096));

    // Every check `load` makes, curve checks and the gamma != delta refusal included.
    let key = ProvingKey::load(&zkey).unwrap();
    assert!(key.vk.ic == vk.k && key.vk.gamma_g2 == vk.gamma_g2);

    // Section 4 is gnark's L and R: its rows on the gnark witness are gnark's A.w and B.w.
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let w = witness.w().to_vec();
    let [aw, bw] = zkey_rows(&key, &w);
    let nc = witness.n_constraints();
    assert!(aw[..nc] == *witness.aw() && bw[..nc] == *witness.bw());
    assert!(aw[nc..].iter().chain(&bw[nc..]).all(Zero::is_zero));

    // snarkrs' zkey path. `prove` checks the result against the zkey's own vk.
    let circuit = CpuBackend::new().prepare(key).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let proof =
        snarkrs_groth16::prove::prove(&*circuit, &w, &mut rng, &mut StageTimings::default())
            .unwrap();
    let public = witness.public(vk.n_public()).unwrap();
    let public_bin = write(&out, "public.bin", &write_public(public));
    let proof_bin = write(&out, "proof.bin", &gnark_proof(&proof, false));
    if let Some(ok) = gnark_verify(&toy("vk.bin"), &public_bin, &proof_bin) {
        assert!(ok, "gnark refused a proof from the converted zkey");
    }

    // snarkjs proves the same zkey from a .wtns of the gnark witness.
    let wtns = out.join("toy.wtns");
    snarkrs_witness::write_wtns(&wtns, &w).unwrap();
    let (proof_json, public_json) = (out.join("proof.json"), out.join("public.json"));
    let proved = snarkjs_run(&[
        "groth16".as_ref(),
        "prove".as_ref(),
        zkey.as_os_str(),
        wtns.as_os_str(),
        proof_json.as_os_str(),
        public_json.as_os_str(),
    ]);
    if proved == Some(true) {
        let proof = proof_from_str(&std::fs::read_to_string(&proof_json).unwrap()).unwrap();
        let proof_bin = write(&out, "snarkjs.bin", &gnark_proof(&proof, true));
        if let Some(ok) = gnark_verify(&toy("vk.bin"), &public_bin, &proof_bin) {
            assert!(ok, "gnark refused snarkjs' proof from the converted zkey");
        }
    } else if proved == Some(false) {
        panic!("snarkjs could not prove the converted zkey");
    }

    // The verifying key a zkey export gives is gnark's, as snarkjs JSON.
    let exported = out.join("vkey.json");
    snarkrs_ceremony::vkey::export_verification_key(&zkey, &exported).unwrap();
    assert_eq!(
        std::fs::read_to_string(&exported).unwrap(),
        snarkrs_gnark_convert::proof::snarkjs_vk(&vk)
    );
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn an_r1cs_from_another_circuit_is_refused() {
    let (pk, vk, _) = toy_keys();
    let other = GnarkR1cs::read(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../gnark-solver/testdata/frontend.r1cs"),
    )
    .unwrap();
    let out = scratch("wrong-r1cs");
    let e = write_zkey(&pk, &vk, &other, &out.join("x.zkey"), &CpuGroupFft).unwrap_err();
    assert!(matches!(e, ConvertError::Mismatch { .. }), "{e}");
    assert!(!out.join("x.zkey").exists());
    std::fs::remove_dir_all(out).unwrap();
}
