use std::path::Path;

use snarkrs_cli::{make_backend, make_constant_work_backend, BackendKind};
use snarkrs_groth16::{Backend, HPoly, ProveError};

pub fn backend(s: Selection<'_>) -> Box<dyn Backend> {
    #[cfg(feature = "wgpu")]
    if s.backend == "wgpu" {
        assert_eq!(
            std::env::var("G16_WGPU_LIMITS").as_deref(),
            Ok(s.profile),
            "UNAVAILABLE: WGPU request must match selection"
        );
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

pub fn h_location(h: &HPoly, backend: &str) -> bool {
    match (backend, h) {
        ("cpu", HPoly::Host(_)) => true,
        ("metal" | "wgpu" | "cuda", HPoly::Device { tag, .. }) => *tag == backend,
        _ => false,
    }
}

pub fn validation_rejection<T>(result: &Result<T, ProveError>, expected: &str) -> bool {
    matches!(result, Err(ProveError::Backend { reason, .. }) if reason == expected)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection<'a> {
    pub backend: &'a str,
    pub work: &'a str,
    pub profile: &'a str,
}

pub fn selection<'a>(
    backend: Option<&'a str>,
    work: Option<&'a str>,
    profile: Option<&'a str>,
) -> Result<Selection<'a>, &'static str> {
    let backend = backend.ok_or("UNAVAILABLE: explicit backend required")?;
    let work = work.ok_or("UNAVAILABLE: explicit work mode required")?;
    let profile = profile.ok_or("UNAVAILABLE: explicit profile required")?;
    if !matches!(backend, "cpu" | "metal" | "wgpu" | "cuda") {
        return Err("UNAVAILABLE: select exactly one known backend");
    }
    if !matches!(work, "variable" | "constant") {
        return Err("UNAVAILABLE: select exactly one known work mode");
    }
    if backend == "cuda" && work == "constant" {
        return Err("UNAVAILABLE: CUDA constant work is not landed");
    }
    if (backend == "wgpu" && !matches!(profile, "floor" | "auto"))
        || (backend != "wgpu" && profile != "none")
    {
        return Err("UNAVAILABLE: WGPU requires floor/auto, other backends require none");
    }
    Ok(Selection {
        backend,
        work,
        profile,
    })
}

pub fn fixtures(selected: Option<&str>, required: &[&str]) -> Result<(), &'static str> {
    let selected = selected.ok_or("UNAVAILABLE: explicit fixture list required")?;
    if selected.split(',').count() != required.len() {
        return Err("UNAVAILABLE: fixture count differs from required cases");
    }
    for name in required {
        if selected.split(',').filter(|s| s == name).count() != 1 {
            return Err("UNAVAILABLE: missing, duplicate or unknown fixture");
        }
    }
    Ok(())
}

pub fn files(dir: &Path, required: &[&str]) -> Result<(), &'static str> {
    if required.iter().all(|file| dir.join(file).is_file()) {
        Ok(())
    } else {
        Err("UNAVAILABLE: required fixture file missing")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_storage_and_tag_guards() {
        let host = HPoly::Host(vec![]);
        assert!(h_location(&host, "cpu"));
        for backend in ["metal", "wgpu", "cuda"] {
            assert!(!h_location(&host, backend));
            let device = HPoly::Device {
                tag: backend,
                len: 0,
                data: std::sync::Arc::new(()),
            };
            assert!(h_location(&device, backend));
            assert!(!h_location(&device, "cpu"));
            for other in ["metal", "wgpu", "cuda"] {
                assert_eq!(h_location(&device, other), backend == other);
            }
        }
    }

    #[test]
    fn device_fault_is_not_input_validation() {
        let expected = "a_query has 5 bases, n_vars is 6";
        let invalid: Result<(), ProveError> = Err(ProveError::Backend {
            backend: "wgpu",
            reason: expected.into(),
        });
        let fault: Result<(), ProveError> = Err(ProveError::Device {
            backend: "wgpu",
            reason: expected.into(),
        });
        let unrelated: Result<(), ProveError> = Err(ProveError::Backend {
            backend: "wgpu",
            reason: "device unavailable".into(),
        });
        assert!(validation_rejection(&invalid, expected));
        assert!(fault.is_err());
        assert!(!validation_rejection(&fault, expected));
        assert!(!validation_rejection(&unrelated, expected));
        assert!(!validation_rejection(&Ok(()), expected));
    }

    #[test]
    fn selection_guards() {
        assert!(selection(None, Some("variable"), Some("none")).is_err());
        assert!(selection(Some("cpu"), None, Some("none")).is_err());
        assert!(selection(Some("wgpu"), Some("variable"), None).is_err());
        for backend in ["", "cpu,cpu", "cpu,metal", "unknown", " cpu"] {
            assert!(selection(Some(backend), Some("variable"), Some("none")).is_err());
        }
        assert!(selection(Some("cuda"), Some("constant"), Some("none")).is_err());
        assert!(selection(Some("cpu"), Some("variable"), Some("floor")).is_err());
        assert!(selection(Some("wgpu"), Some("variable"), Some("raised")).is_err());
        assert!(selection(Some("cpu"), Some("other"), Some("none")).is_err());
        for backend in ["cpu", "metal", "wgpu", "cuda"] {
            assert!(selection(
                Some(backend),
                Some("variable"),
                Some(if backend == "wgpu" { "floor" } else { "none" })
            )
            .is_ok());
        }
        assert!(selection(Some("wgpu"), Some("constant"), Some("auto")).is_ok());
    }

    #[test]
    fn fixture_guards() {
        let required = ["tiny_mul", "sha256"];
        for selected in [
            None,
            Some(""),
            Some("tiny_mul"),
            Some("tiny_mul,tiny_mul"),
            Some("tiny_mul,unknown"),
            Some("tiny_mul,sha256,sha256"),
        ] {
            assert!(fixtures(selected, &required).is_err());
        }
        assert!(fixtures(Some("tiny_mul,sha256"), &required).is_ok());
        assert!(fixtures(Some("sha256,tiny_mul"), &required).is_ok());
        assert!(files(
            Path::new(env!("CARGO_MANIFEST_DIR")),
            &["absent-conformance-fixture"]
        )
        .is_err());
        assert!(files(Path::new(env!("CARGO_MANIFEST_DIR")), &["Cargo.toml"]).is_ok());
    }
}
