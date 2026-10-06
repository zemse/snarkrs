//! Required, bounded backend parity. One backend/work/profile per process, no fallback.

#[path = "../../../test-support/conformance.rs"]
mod guards;

use std::path::{Path, PathBuf};
use std::process::Command;

use ark_ec::CurveGroup;
use snarkrs_cli::{json, make_backend, make_constant_work_backend, BackendKind};
use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::prove::prove_with_blinders;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{Backend, HPoly, PreparedCircuit, ProveError, StageTimings};

const REQUIRED: [&str; 2] = ["tiny_mul", "sha256"];

fn root() -> PathBuf {
    std::env::var_os("G16_CONFORMANCE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts"))
}

fn backend(s: guards::Selection<'_>) -> Box<dyn Backend> {
    #[cfg(feature = "wgpu")]
    if s.backend == "wgpu" {
        let p = if s.work == "constant" {
            snarkrs_wgpu::WgpuProver::constant_work()
        } else {
            snarkrs_wgpu::WgpuProver::new()
        }
        .expect("UNAVAILABLE: WGPU device");
        let d = p.device();
        eprintln!(
            "device={:?}\n{}\nauto_fallback={:?}",
            d.adapter_info(),
            d.limits_table(),
            d.auto_fallback()
        );
        assert!(
            matches!(
                format!("{:?}", d.adapter_info().device_type).as_str(),
                "DiscreteGpu" | "IntegratedGpu"
            ),
            "UNAVAILABLE: software/unknown adapter is not physical GPU evidence"
        );
        if s.profile == "auto" {
            assert!(
                d.auto_fallback().is_none(),
                "UNAVAILABLE: Auto fell back to Floor"
            );
            let caps = d.granted_limits();
            assert!(
                caps.max_buffer_size > 256 * 1024 * 1024
                    && caps.max_storage_buffer_binding_size > 128 * 1024 * 1024,
                "UNAVAILABLE: Auto capacity does not exceed Floor"
            );
        }
        return Box::new(p);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if s.backend == "metal" {
        let p = if s.work == "constant" {
            snarkrs_metal::MetalBackend::constant_work()
        } else {
            snarkrs_metal::MetalBackend::new()
        }
        .expect("UNAVAILABLE: Metal device");
        eprintln!("device={}", p.device().name());
        return Box::new(p);
    }
    let kind = match s.backend {
        "cpu" => BackendKind::Cpu,
        "metal" => BackendKind::Metal,
        "wgpu" => BackendKind::Wgpu,
        "cuda" => BackendKind::Cuda,
        _ => unreachable!(),
    };
    let p = if s.work == "constant" {
        make_constant_work_backend(kind)
    } else {
        make_backend(kind)
    }
    .expect("UNAVAILABLE: backend");
    eprintln!(
        "device={} (no device attestation exposed by factory)",
        s.backend
    );
    p
}

fn independent(verifier: &Path, dir: &Path, proof: &Path, public: &Path) {
    let o = Command::new(verifier)
        .args(["groth16", "verify"])
        .arg(dir.join("vkey.json"))
        .arg(public)
        .arg(proof)
        .output()
        .expect("UNAVAILABLE: independent verifier executable");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        o.status.success() && text.contains("OK!"),
        "independent verifier rejected proof: {text}"
    );
}

fn proof(c: &dyn PreparedCircuit, w: &[Fr], r: Fr, s: Fr) -> snarkrs_groth16::Proof {
    prove_with_blinders(c, w, r, s, &mut StageTimings::default()).expect("pinned proof")
}

fn tiny_guards(b: &dyn Backend, c: &dyn PreparedCircuit, dir: &Path, w: &[Fr]) {
    let load = || ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    type Mutation = fn(&mut ProvingKey);
    let queries: [(&str, Mutation); 2] = [
        ("a_query", |k| {
            k.a_query.pop();
        }),
        ("h_query", |k| {
            k.h_query.pop();
        }),
    ];
    for (name, mutate) in queries {
        let mut k = load();
        mutate(&mut k);
        if b.name() == "cpu" {
            // CPU query shapes are checked at msms, not at the compute_h oracle boundary.
            let bad = b.prepare(k).expect("CPU stage-only prepare");
            let mut t = StageTimings::default();
            let h = bad.compute_h(w, &mut t).unwrap();
            assert!(
                matches!(bad.msms(w, &h, &mut t), Err(ProveError::Backend { .. })),
                "{name}: malformed query accepted by msms"
            );
        } else {
            assert!(
                b.prepare(k).is_err(),
                "{name}: malformed query accepted by prepare"
            );
        }
    }
    let mutations: [(&str, Mutation); 4] = [
        ("n_public", |k| {
            k.n_public = k.n_vars + 1;
        }),
        ("CSR first row", |k| {
            k.coeffs.row_ptr[0][0] = 1;
        }),
        ("CSR signal", |k| {
            k.coeffs.signal[0][0] = k.n_vars as u32;
        }),
        ("CSR values", |k| {
            k.coeffs.value[0].pop();
        }),
    ];
    for (name, mutate) in mutations {
        let mut k = load();
        mutate(&mut k);
        assert!(b.prepare(k).is_err(), "{name}: malformed key/CSR accepted");
    }
    let mut t = StageTimings::default();
    for n in [0, 1, w.len() - 1, w.len() + 1] {
        let mut bad = w.to_vec();
        bad.resize(n, Fr::from(1u64));
        assert!(matches!(
            c.compute_h(&bad, &mut t),
            Err(ProveError::WitnessLength { .. })
        ));
    }
    let h = c.compute_h(w, &mut t).unwrap();
    assert!(matches!(
        c.msms(&w[..w.len() - 1], &h, &mut t),
        Err(ProveError::WitnessLength { .. })
    ));
    for n in [0, c.domain_size() - 1, c.domain_size() + 1] {
        assert!(c
            .msms(w, &HPoly::Host(vec![Fr::from(0u64); n]), &mut t)
            .is_err());
    }
    let mut bad = w.to_vec();
    bad[0] = Fr::from(0u64);
    assert!(matches!(
        prove_with_blinders(c, &bad, Fr::from(0u64), Fr::from(0u64), &mut t),
        Err(ProveError::ConstantWire)
    ));
    std::thread::scope(|scope| {
        let handles: Vec<_> = [(2u64, 3u64, 5u64), (7, 11, 13)]
            .into_iter()
            .map(|(a, b, d)| {
                scope.spawn(move || {
                    let w = [1, a * b * d, a, b, d, a * b].map(Fr::from);
                    let want = proof(c, &w, Fr::from(3u64), Fr::from(5u64));
                    let got = proof(c, &w, Fr::from(3u64), Fr::from(5u64));
                    assert_eq!((got.a, got.b, got.c), (want.a, want.b, want.c));
                    verify(&c.key().vk, &w[1..=c.n_public()], &got).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });
}

#[test]
#[ignore = "required fixtures, explicit selection and independent verifier; serial GPU slot"]
fn required_backend_conformance() {
    let backend_name = std::env::var("G16_CONFORMANCE_BACKEND").ok();
    let work = std::env::var("G16_CONFORMANCE_WORK").ok();
    let profile = std::env::var("G16_CONFORMANCE_PROFILE").ok();
    let s =
        guards::selection(backend_name.as_deref(), work.as_deref(), profile.as_deref()).unwrap();
    let fixtures = std::env::var("G16_CONFORMANCE_FIXTURES").ok();
    guards::fixtures(fixtures.as_deref(), &REQUIRED).unwrap();
    if s.backend == "wgpu" {
        assert_eq!(
            std::env::var("G16_WGPU_LIMITS").as_deref(),
            Ok(s.profile),
            "UNAVAILABLE: explicit WGPU request must match selected profile"
        );
    }
    let root = root();
    for name in REQUIRED {
        guards::files(
            &root.join(name),
            &["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"],
        )
        .unwrap_or_else(|e| panic!("{e}: {}", root.join(name).display()));
    }
    let tiny = root.join("tiny_mul");
    guards::files(
        &tiny,
        &["h_expected.json", "msm_expected.json", "proof.json"],
    )
    .unwrap();
    let verifier = PathBuf::from(std::env::var_os("SNARKJS").unwrap_or_else(|| "snarkjs".into()));
    independent(
        &verifier,
        &tiny,
        &tiny.join("proof.json"),
        &tiny.join("public.json"),
    );
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(revision.status.success());
    eprintln!(
        "revision={} backend={} work={} profile={}",
        String::from_utf8_lossy(&revision.stdout).trim(),
        s.backend,
        s.work,
        s.profile
    );
    let b = backend(s);
    let cpu_backend = make_backend(BackendKind::Cpu).unwrap();
    let out = std::env::temp_dir().join(format!("snarkrs-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let mut completed = 0;
    for name in REQUIRED {
        let dir = root.join(name);
        let load = || ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let cpu = cpu_backend.prepare(load()).unwrap();
        let c = b.prepare(load()).unwrap();
        assert_eq!(c.backend_name(), s.backend);
        let mut t = StageTimings::default();
        let ch = cpu.compute_h(&w, &mut t).unwrap();
        let h = c.compute_h(&w, &mut t).unwrap();
        let host = c.h_to_host(&h).expect("H readback");
        assert_eq!(host, cpu.h_to_host(&ch).unwrap(), "{name}: CPU H");
        let want = cpu.msms(&w, &ch, &mut t).unwrap();
        assert_eq!(
            c.msms(&w, &h, &mut t).unwrap(),
            want,
            "{name}: all five resident MSMs"
        );
        assert_eq!(
            c.msms(&w, &HPoly::Host(host.clone()), &mut t).unwrap(),
            want,
            "{name}: host H MSMs"
        );
        if name == "tiny_mul" {
            let expected: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("h_expected.json")).unwrap(),
            )
            .unwrap();
            let expected: Vec<Fr> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| json::parse_field(v, "H").unwrap())
                .collect();
            assert_eq!(host, expected, "independent tiny H oracle");
            let v: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("msm_expected.json")).unwrap(),
            )
            .unwrap();
            for (key, p) in [
                ("a_g1", want.a_g1),
                ("b_g1", want.b_g1),
                ("l_g1", want.l_g1),
                ("h_g1", want.h_g1),
            ] {
                assert_eq!(p.into_affine(), json::read_g1(&v[key], key).unwrap());
            }
            assert_eq!(
                want.b_g2.into_affine(),
                json::read_g2(&v["b_g2"], "b_g2").unwrap()
            );
            tiny_guards(b.as_ref(), c.as_ref(), &dir, &w);
        }
        let public = &w[1..=c.n_public()];
        assert_eq!(public, json::read_public(&dir.join("public.json")).unwrap());
        for (i, (r, s)) in [(0u64, 0u64), (12345, 67890)].into_iter().enumerate() {
            let got = proof(c.as_ref(), &w, Fr::from(r), Fr::from(s));
            let want = proof(cpu.as_ref(), &w, Fr::from(r), Fr::from(s));
            assert_eq!(
                (got.a, got.b, got.c),
                (want.a, want.b, want.c),
                "{name}: pinned {i}"
            );
            let again = proof(c.as_ref(), &w, Fr::from(r), Fr::from(s));
            assert_eq!((got.a, got.b, got.c), (again.a, again.b, again.c));
            verify(&vk, public, &got).unwrap();
            let p = out.join(format!("{name}-{i}-proof.json"));
            let signals = out.join(format!("{name}-{i}-public.json"));
            json::write_proof(&p, &got).unwrap();
            json::write_public(&signals, public).unwrap();
            independent(&verifier, &dir, &p, &signals);
        }
        completed += 1;
        eprintln!("PASS fixture={name} domain={} H=CPU all5MSMs=CPU pinned=2 own+independent=accepted oracle={}", c.domain_size(), if name == "tiny_mul" { "independent-H+MSM" } else { "CPU-stage-only" });
    }
    std::fs::remove_dir_all(out).unwrap();
    assert_eq!(completed, REQUIRED.len());
    eprintln!("PASS completed={completed}/{}", REQUIRED.len());
}
