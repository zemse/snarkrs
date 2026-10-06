use std::path::Path;

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
