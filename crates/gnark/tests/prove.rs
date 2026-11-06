//! The gate for proving a gnark key: snarkrs proves the toy fixture on the CPU, and both
//! gnark's own `groth16.Verify` (through `test-support/gnark`) and snarkrs' verifier accept
//! the proof against gnark's verifying key. A tampered witness does not prove.
//!
//! The gnark half needs `go` on `PATH`; without it that half skips with a message, or fails
//! under `G16_REQUIRE_TESTS=1`.

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use ark_std::rand::{rngs::StdRng, SeedableRng};
use snarkrs_field::*;
use snarkrs_gnark::proof::{read_proof, write_proof, write_public};
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, PreparedCircuit, ProveError, StageTimings};
use std::path::{Path, PathBuf};
use std::process::Command;

fn harness() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-support/gnark")
}

fn toy(name: &str) -> PathBuf {
    harness().join("toy").join(name)
}

fn circuit() -> (Box<dyn PreparedCircuit>, GnarkVerifyingKey) {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump"))
        .unwrap()
        .into_proving_key(&vk)
        .unwrap();
    (CpuBackend::new().prepare(pk).unwrap(), vk)
}

/// `go run . verify` in the harness: `Some(true)` when gnark accepts, `None` without Go.
fn gnark_verify(dir: &Path, public: &[u8], proof: &[u8]) -> Option<bool> {
    if Command::new("go").arg("version").output().is_err() {
        resources::skip("gnark verify: no `go` on PATH, so gnark's own verifier was not run");
        return None;
    }
    std::fs::write(dir.join("public.bin"), public).unwrap();
    std::fs::write(dir.join("proof.bin"), proof).unwrap();
    let out = Command::new("go")
        .current_dir(harness())
        .args(["run", ".", "verify"])
        .arg(toy("vk.bin"))
        .arg(dir.join("public.bin"))
        .arg(dir.join("proof.bin"))
        .output()
        .expect("go run");
    Some(out.status.success())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("snarkrs-gnark-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_snarkrs_proof_passes_gnark_verify() {
    let (circuit, vk) = circuit();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    // `prove` checks the proof against the circuit's key, which is gnark's vk, so a proof
    // that comes back has already passed snarkrs' verifier once.
    let proof =
        snarkrs_gnark::prove(&*circuit, &witness, &mut rng, &mut StageTimings::default()).unwrap();

    let public = witness.public(vk.n_public()).unwrap();
    verify(&vk.to_verifying_key(), public, &proof).unwrap();
    assert_eq!(read_proof(&write_proof(&proof)).unwrap(), proof);

    let dir = scratch("accept");
    let public_bytes = write_public(public);
    assert_eq!(public_bytes, std::fs::read(toy("public.bin")).unwrap());
    if let Some(accepted) = gnark_verify(&dir, &public_bytes, &write_proof(&proof)) {
        assert!(accepted, "gnark's groth16.Verify rejected snarkrs' proof");
        // And it is a real check: the same proof under another public input is refused.
        let mut other = public.to_vec();
        other[0] += Fr::ONE;
        let refused = gnark_verify(&dir, &write_public(&other), &write_proof(&proof));
        assert_eq!(refused, Some(false), "gnark accepted a wrong public input");
    }
    std::fs::remove_dir_all(dir).ok();
}

/// The reading half of the interchange, the other way round: gnark's own proof of the
/// same witness passes snarkrs' verifier against the same key.
#[test]
fn a_gnark_proof_passes_snarkrs_verify() {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let proof = read_proof(&std::fs::read(toy("proof.bin")).unwrap()).unwrap();
    verify(
        &vk.to_verifying_key(),
        witness.public(vk.n_public()).unwrap(),
        &proof,
    )
    .unwrap();
}

#[test]
fn a_tampered_witness_does_not_prove() {
    let (circuit, _) = circuit();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let mut rng = StdRng::seed_from_u64(8);

    // A private wire, a public wire, and one row of each of A.w and B.w: every one leaves a
    // proof that does not verify, which the check of the result refuses.
    let n = witness.w().len();
    for (what, wire, row_a, row_b) in [
        ("private wire", Some(n - 1), None, None),
        ("public wire", Some(1), None, None),
        ("A.w row", None, Some(0), None),
        ("B.w row", None, None, Some(witness.n_constraints() - 1)),
    ] {
        let (mut w, mut aw, mut bw) = (
            witness.w().to_vec(),
            witness.aw().to_vec(),
            witness.bw().to_vec(),
        );
        if let Some(i) = wire {
            w[i] += Fr::ONE;
        }
        if let Some(i) = row_a {
            aw[i] += Fr::ONE;
        }
        if let Some(i) = row_b {
            bw[i] += Fr::ONE;
        }
        let tampered = GnarkWitness::from_parts(&w, &aw, &bw).unwrap();
        let got =
            snarkrs_gnark::prove(&*circuit, &tampered, &mut rng, &mut StageTimings::default());
        assert!(
            matches!(got, Err(ProveError::SelfVerify(_))),
            "{what}: {:?}",
            got.map(|_| ())
        );
    }
}

/// Rows past the domain, or A and B of different lengths, are refused before any work.
#[test]
fn rows_that_do_not_fit_the_domain_are_refused() {
    let (circuit, _) = circuit();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let too_many = vec![Fr::ONE; circuit.domain_size() + 1];
    let err = circuit
        .h_and_msms_from_rows(
            witness.w(),
            &too_many,
            &too_many,
            &mut StageTimings::default(),
        )
        .err()
        .unwrap();
    assert!(err.to_string().contains("at most the domain size"), "{err}");
    let err = circuit
        .h_and_msms_from_rows(
            witness.w(),
            witness.aw(),
            &witness.bw()[1..],
            &mut StageTimings::default(),
        )
        .err()
        .unwrap();
    assert!(err.to_string().contains("one per constraint"), "{err}");
}

/// A gnark key has no matrices, so the zkey path refuses it rather than gathering nothing.
#[test]
fn the_zkey_path_refuses_a_gnark_key() {
    let (circuit, _) = circuit();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let err = circuit
        .compute_h(witness.w(), &mut StageTimings::default())
        .err()
        .unwrap();
    assert!(err.to_string().contains("no constraint matrices"), "{err}");
}

#[test]
fn a_verifying_key_from_another_setup_is_refused() {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let mut other = vk.clone();
    other.alpha_g1 = (other.alpha_g1 + other.alpha_g1).into_affine();
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
    let err = pk.into_proving_key(&other).err().unwrap().to_string();
    assert!(err.contains("G1.Alpha"), "{err}");
}
