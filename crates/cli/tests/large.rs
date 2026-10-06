//! Required production-scale proof success, not a capacity-refusal or readiness test.
//! Running this gate requires separate approval for the domain 2^22 allocation.

#[path = "../../../test-support/conformance.rs"]
mod guards;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
    #[cfg(feature = "wgpu")]
    if s.backend == "wgpu" {
        let p = snarkrs_wgpu::WgpuProver::new().expect("UNAVAILABLE: WGPU device preflight");
        let d = p.device();
        eprintln!(
            "preflight_device={:?}\n{}\nauto_fallback={:?}",
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
            let caps = d.granted_limits();
            assert!(
                d.auto_fallback().is_none()
                    && caps.max_buffer_size > 256 * 1024 * 1024
                    && caps.max_storage_buffer_binding_size > 128 * 1024 * 1024,
                "UNAVAILABLE: Auto fell back or did not raise capacity"
            );
        }
    }
    let revision = run(Command::new("git").args(["rev-parse", "HEAD"]));
    assert!(revision.status.success());
    eprintln!("revision={} backend={} work={} profile={} physical_device=unattested (retain CLI device diagnostics)", String::from_utf8_lossy(&revision.stdout).trim(), s.backend, s.work, s.profile);
    let out = std::env::temp_dir().join(format!("snarkrs-large-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let mut completed = 0;
    for name in REQUIRED {
        let dir = root.join(name);
        let key = snarkrs_formats::ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        assert_eq!(
            key.domain_size, DOMAIN,
            "UNAVAILABLE: unexpected required domain"
        );
        drop(key);
        let reference = snarkrs_cli::json::read_public(&dir.join("public.json")).unwrap();
        let proof = out.join("proof.json");
        let public = out.join("public.json");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_snarkrs"));
        cmd.args(["groth16", "prove"])
            .arg(dir.join("circuit.zkey"))
            .arg(dir.join("circuit.wtns"))
            .arg(&proof)
            .arg(&public)
            .args([
                "--backend",
                s.backend,
                "--fallback",
                "false",
                "--self-verify",
                "false",
                "--stage-timings",
            ]);
        if s.work == "constant" {
            cmd.arg("--constant-work");
        }
        let o = run(&mut cmd);
        eprintln!("fixture={name} domain={DOMAIN}:\n{}", text(&o));
        assert!(
            o.status.success(),
            "proof failed (OOM/timeout/panic/device errors are NOT expected capacity refusals): {}",
            text(&o)
        );
        assert_eq!(snarkrs_cli::json::read_public(&public).unwrap(), reference);
        let o = run(Command::new(env!("CARGO_BIN_EXE_snarkrs"))
            .args(["groth16", "verify"])
            .arg(dir.join("vkey.json"))
            .arg(&public)
            .arg(&proof));
        assert!(
            o.status.success() && text(&o).contains("OK"),
            "own verifier rejected: {}",
            text(&o)
        );
        let o = run(Command::new(&verifier)
            .args(["groth16", "verify"])
            .arg(dir.join("vkey.json"))
            .arg(&public)
            .arg(&proof));
        assert!(
            o.status.success() && text(&o).contains("OK!"),
            "independent verifier rejected: {}",
            text(&o)
        );
        completed += 1;
        eprintln!("PASS fixture={name} domain={DOMAIN} proof+public+own+independent=accepted");
    }
    std::fs::remove_dir_all(&out).unwrap();
    assert_eq!(completed, REQUIRED.len());
    eprintln!("PASS completed={completed}/{}", REQUIRED.len());
}
