//! Required production-scale proof success, not a capacity-refusal or readiness test.
//! Running this gate requires separate approval for the domain 2^22 allocation.

#[path = "../../../test-support/conformance.rs"]
mod guards;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use snarkrs_cli::json;
use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::prove::prove_unchecked;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, Proof, ProveError, StageTimings};

const REQUIRED: [&str; 1] = ["large/js_384x384_d32"];
const DOMAIN: usize = 1 << 22;

fn run(cmd: &mut Command) -> Output {
    cmd.output()
        .unwrap_or_else(|e| panic!("UNAVAILABLE: failed to run {cmd:?}: {e}"))
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn proof_on_backend(
    backend: &dyn Backend,
    key: ProvingKey,
    witness: &[Fr],
    timings: &mut StageTimings,
) -> Result<(Proof, Vec<Fr>), ProveError> {
    let circuit = backend.prepare(key)?;
    assert_eq!(circuit.backend_name(), backend.name());
    let proof = prove_unchecked(
        circuit.as_ref(),
        witness,
        &mut ark_std::rand::rngs::OsRng,
        timings,
    )?;
    Ok((proof, witness[1..=circuit.n_public()].to_vec()))
}

#[test]
fn proof_uses_the_supplied_backend_instance() {
    use snarkrs_field::{G1Affine, G2Affine};
    use snarkrs_formats::Coefficients;
    use snarkrs_groth16::PreparedCircuit;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Inspected(AtomicUsize);
    impl Backend for Inspected {
        fn name(&self) -> &'static str {
            "mock-inspected"
        }
        fn prepare(&self, _: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(ProveError::Backend {
                backend: "mock-inspected",
                reason: "inspected instance".into(),
            })
        }
    }
    let inspected = Inspected(AtomicUsize::new(0));
    let other = Inspected(AtomicUsize::new(0));
    let g1 = G1Affine::identity();
    let g2 = G2Affine::identity();
    let key = ProvingKey {
        n_vars: 1,
        n_public: 0,
        domain_size: 1,
        alpha_g1: g1,
        beta_g1: g1,
        beta_g2: g2,
        delta_g1: g1,
        delta_g2: g2,
        a_query: vec![g1],
        b_g1_query: vec![g1],
        b_g2_query: vec![g2],
        l_query: vec![],
        h_query: vec![g1],
        coeffs: Coefficients {
            row_ptr: [vec![0, 0], vec![0, 0]],
            signal: [vec![], vec![]],
            value: [vec![], vec![]],
        },
        vk: VerifyingKey {
            alpha_g1: g1,
            beta_g2: g2,
            gamma_g2: g2,
            delta_g2: g2,
            ic: vec![g1],
        },
    };
    let result = proof_on_backend(
        &inspected,
        key,
        &[Fr::from(1u64)],
        &mut StageTimings::default(),
    );
    assert!(guards::validation_rejection(&result, "inspected instance"));
    assert_eq!(inspected.0.load(Ordering::SeqCst), 1);
    assert_eq!(other.0.load(Ordering::SeqCst), 0);
}

#[test]
#[ignore = "requires separate large-run approval: domain 2^22 proof, no expected refusal"]
fn every_backend_proves_the_large_artifacts() {
    let backend = std::env::var("G16_CONFORMANCE_BACKEND").ok();
    let work = std::env::var("G16_CONFORMANCE_WORK").ok();
    let profile = std::env::var("G16_CONFORMANCE_PROFILE").ok();
    let s = guards::selection(backend.as_deref(), work.as_deref(), profile.as_deref()).unwrap();
    let fixtures = std::env::var("G16_LARGE_FIXTURES").ok();
    guards::fixtures(fixtures.as_deref(), &REQUIRED).unwrap();
    if s.backend == "wgpu" {
        assert_eq!(
            std::env::var("G16_WGPU_LIMITS").as_deref(),
            Ok(s.profile),
            "UNAVAILABLE: WGPU request must match selected profile"
        );
    }
    assert!(
        match s.backend {
            "cpu" => true,
            "metal" => cfg!(all(feature = "metal", target_os = "macos")),
            "wgpu" => cfg!(feature = "wgpu"),
            "cuda" => cfg!(feature = "cuda"),
            _ => false,
        },
        "UNAVAILABLE: backend not built for this target"
    );
    let root = std::env::var_os("G16_CONFORMANCE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts"));
    for name in REQUIRED {
        guards::files(
            &root.join(name),
            &["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"],
        )
        .unwrap_or_else(|e| panic!("{e}: {name}"));
    }
    let verifier = std::env::var_os("SNARKJS").unwrap_or_else(|| "snarkjs".into());
    let tiny = root.join("tiny_mul");
    guards::files(&tiny, &["vkey.json", "public.json", "proof.json"]).unwrap();
    let preflight = run(Command::new(&verifier)
        .args(["groth16", "verify"])
        .arg(tiny.join("vkey.json"))
        .arg(tiny.join("public.json"))
        .arg(tiny.join("proof.json")));
    assert!(
        preflight.status.success() && text(&preflight).contains("OK!"),
        "UNAVAILABLE: independent verifier preflight: {}",
        text(&preflight)
    );
    let revision = run(Command::new("git").args(["rev-parse", "HEAD"]));
    assert!(revision.status.success());
    eprintln!(
        "revision={} backend={} work={} profile={}",
        String::from_utf8_lossy(&revision.stdout).trim(),
        s.backend,
        s.work,
        s.profile
    );
    let backend = guards::backend(s);
    let out = std::env::temp_dir().join(format!("snarkrs-large-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let mut completed = 0;
    for name in REQUIRED {
        let dir = root.join(name);
        let key = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        assert_eq!(
            key.domain_size, DOMAIN,
            "UNAVAILABLE: unexpected required domain"
        );
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let reference = json::read_public(&dir.join("public.json")).unwrap();
        let mut timings = StageTimings::default();
        let (proof, public) = proof_on_backend(backend.as_ref(), key, &witness, &mut timings)
            .expect(
                "proof failed (OOM/timeout/panic/device errors are NOT expected capacity refusals)",
            );
        assert_eq!(public, reference);
        let proof_path = out.join("proof.json");
        let public_path = out.join("public.json");
        json::write_proof(&proof_path, &proof).unwrap();
        json::write_public(&public_path, &public).unwrap();
        let serialized_public = json::read_public(&public_path).unwrap();
        assert_eq!(serialized_public, reference);
        verify(
            &vk,
            &serialized_public,
            &json::read_proof(&proof_path).unwrap(),
        )
        .expect("own verifier rejected serialized proof");
        let o = run(Command::new(&verifier)
            .args(["groth16", "verify"])
            .arg(dir.join("vkey.json"))
            .arg(&public_path)
            .arg(&proof_path));
        assert!(
            o.status.success() && text(&o).contains("OK!"),
            "independent verifier rejected: {}",
            text(&o)
        );
        completed += 1;
        eprintln!("PASS fixture={name} domain={DOMAIN} same-inspected-instance+proof+public+own+independent=accepted timings={timings:?}");
    }
    std::fs::remove_dir_all(&out).unwrap();
    assert_eq!(completed, REQUIRED.len());
    eprintln!("PASS completed={completed}/{}", REQUIRED.len());
}
