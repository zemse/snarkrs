//! A gnark key on a real NVIDIA device: the CUDA rows path against the CPU backend, element
//! for element, and [`GnarkProver`]'s proof through gnark's own `groth16.Verify`.
//!
//! The CPU backend is the oracle, as in `pipeline_gpu.rs`: its gnark path passes gnark's
//! verifier (`crates/gnark/tests/prove.rs`), and stages 0 to 9 are deterministic, so the
//! five MSM outputs of the two backends must be the same points. A proof is not, its
//! blinders are random, so the proof is checked by verifying it.
//!
//! Without a device every test skips with a line on stderr, or fails under
//! `G16_REQUIRE_TESTS=1`. The gnark half of the verify test also needs `go` on `PATH`.

#![cfg(feature = "cuda")]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use std::path::{Path, PathBuf};
use std::process::Command;

use ark_std::rand::{rngs::StdRng, SeedableRng};
use snarkrs_cuda::{CudaBackend, GnarkProver};
use snarkrs_field::*;
use snarkrs_gnark::proof::{write_proof, write_public};
use snarkrs_gnark::{GnarkProvingKey, GnarkVerifyingKey, GnarkWitness};
use snarkrs_groth16::cpu::CpuBackend;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, MsmOutputs, PreparedCircuit, ProveError, StageTimings};

fn harness() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-support/gnark")
}

fn toy(name: &str) -> PathBuf {
    harness().join("toy").join(name)
}

fn vk() -> GnarkVerifyingKey {
    GnarkVerifyingKey::read(&toy("vk.bin")).unwrap()
}

fn cpu() -> Box<dyn PreparedCircuit> {
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump"))
        .unwrap()
        .into_proving_key(&vk())
        .unwrap();
    CpuBackend::new().prepare(pk).unwrap()
}

/// The toy key on [`GnarkProver`], or `None` with the reason when there is no device.
fn prover() -> Option<GnarkProver> {
    match GnarkProver::open(&toy("pk.dump"), &vk()) {
        Ok(p) => Some(p),
        Err(e) => {
            resources::skip(format_args!("no usable CUDA device for GnarkProver: {e}"));
            None
        }
    }
}

fn rows(circuit: &dyn PreparedCircuit, w: &GnarkWitness<'_>) -> MsmOutputs {
    circuit
        .h_and_msms_from_rows(w.w(), w.aw(), w.bw(), &mut StageTimings::default())
        .unwrap()
}

fn assert_same(got: &MsmOutputs, want: &MsmOutputs, what: &str) {
    assert_eq!(got.a_g1.into_affine(), want.a_g1.into_affine(), "{what}: A");
    assert_eq!(
        got.b_g2.into_affine(),
        want.b_g2.into_affine(),
        "{what}: B in G2"
    );
    assert_eq!(
        got.b_g1.into_affine(),
        want.b_g1.into_affine(),
        "{what}: B in G1"
    );
    assert_eq!(got.l_g1.into_affine(), want.l_g1.into_affine(), "{what}: L");
    assert_eq!(got.h_g1.into_affine(), want.h_g1.into_affine(), "{what}: H");
}

/// Both CUDA entry points, the generic `Backend::prepare` of a decoded key and the
/// streamed `GnarkProver`, give the CPU backend's five MSM outputs for the toy witness.
#[test]
fn the_cuda_rows_path_is_the_cpus() {
    let Some(prover) = prover() else { return };
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let want = rows(&*cpu(), &witness);
    assert_same(&rows(prover.circuit(), &witness), &want, "GnarkProver");
    // Twice, so the second proof runs on the pooled scratch and the arena's buffers.
    assert_same(
        &rows(prover.circuit(), &witness),
        &want,
        "GnarkProver again",
    );

    let pk = GnarkProvingKey::read_dump(&toy("pk.dump"))
        .unwrap()
        .into_proving_key(&vk())
        .unwrap();
    let generic = CudaBackend::new().unwrap().prepare(pk).unwrap();
    assert_same(&rows(&*generic, &witness), &want, "CudaBackend::prepare");
    let err = generic
        .compute_h(witness.w(), &mut StageTimings::default())
        .err()
        .unwrap();
    assert!(err.to_string().contains("no constraint matrices"), "{err}");
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

/// A `GnarkProver` proof passes snarkrs' verifier and gnark's `groth16.Verify`, and a
/// tampered row does not prove.
#[test]
fn a_gnark_prover_proof_passes_gnark_verify() {
    let Some(prover) = prover() else { return };
    let vk = vk();
    let witness = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let mut t = StageTimings::default();
    let proof = prover.prove(&witness, &mut rng, &mut t).unwrap();
    eprintln!("GnarkProver on {}: {t:?}", prover.device_name());
    let public = witness.public(vk.n_public()).unwrap();
    verify(&vk.to_verifying_key(), public, &proof).unwrap();

    let dir = std::env::temp_dir().join(format!("snarkrs-cuda-gnark-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(accepted) = gnark_verify(&dir, &write_public(public), &write_proof(&proof)) {
        assert!(accepted, "gnark's groth16.Verify rejected the CUDA proof");
    }
    std::fs::remove_dir_all(dir).ok();

    let mut aw = witness.aw().to_vec();
    aw[0] += Fr::ONE;
    let tampered = GnarkWitness::from_parts(witness.w(), &aw, witness.bw()).unwrap();
    let got = prover.prove(&tampered, &mut rng, &mut StageTimings::default());
    assert!(
        matches!(got, Err(ProveError::SelfVerify(_))),
        "{:?}",
        got.map(|_| ())
    );
}

/// What construction took from the device is no more than it budgeted. Device memory is
/// per process, so this only means something with `--test-threads=1`.
#[test]
fn construction_takes_no_more_than_its_budget() {
    let Some(before) = snarkrs_cuda::Cuda::new(0)
        .ok()
        .and_then(|c| c.context().mem_get_info().ok())
    else {
        resources::skip("no usable CUDA device");
        return;
    };
    let Some(prover) = prover() else { return };
    let (after, _) = prover.device_memory().unwrap();
    let taken = before.0.saturating_sub(after) as u64;
    eprintln!(
        "toy key: {taken} bytes taken, {} budgeted ({:?})",
        prover.budget().total,
        prover.budget()
    );
    assert!(
        taken <= prover.budget().total,
        "{taken} > {:?}",
        prover.budget()
    );
}
