//! Check the field-layer patches against the representation they replace.
//!
//! `build.rs` changes how `mul_s1s2` tags a short product, and a witness that still
//! hashes correctly is not enough evidence for that: it only exercises the paths these
//! sixteen circuits happen to take. `fr_paths.cpp` compares the short and long forms of
//! the same value across every consumer in the field API, which is the property the
//! patch actually needs. Running it the other way, with negatives allowed, fails, so the
//! restriction in the patch is pinned by a test rather than by a comment.
//!
//! The C++ links against the field library this crate's build produced, so the test
//! covers the patched code that ships rather than a copy of it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `<witnesscalc>` as `build.rs` left it, or `None` when the build was not run with the
/// patches (`G16_CSP_STOCK_FR`), in which case there is nothing here to check.
fn witnesscalc() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("G16_CSP_WITNESSCALC"));
    dir.join("package/lib/libfr.a").is_file().then_some(dir)
}

fn compile(w: &Path, out: &Path, allow_negative: bool) -> bool {
    let mut c = Command::new("c++");
    c.args(["-O2", "-std=gnu++11", "-w", "-DNDEBUG"])
        .args(["-DUSE_ASM", "-DARCH_ARM64", "-D_LONG_LONG_LIMB"])
        .arg(format!("-I{}", w.join("src").display()))
        .arg(format!("-I{}", w.join("build").display()))
        .arg(format!(
            "-I{}",
            w.join("depends/gmp/package/include").display()
        ));
    if allow_negative {
        c.arg("-DG16_ALLOW_NEGATIVE");
    }
    c.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fr_paths.cpp"))
        .arg(w.join("package/lib/libfr.a"))
        .arg(w.join("depends/gmp/package/lib/libgmp.a"))
        .arg("-o")
        .arg(out);
    c.status().is_ok_and(|s| s.success())
}

fn run(bin: &Path) -> (bool, String) {
    let out = Command::new(bin).output().expect("running the probe");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

#[test]
fn short_and_long_forms_agree_on_every_consumer() {
    let Some(w) = witnesscalc() else { return };
    let dir = std::env::temp_dir().join("g16-csp-fr-paths");
    std::fs::create_dir_all(&dir).unwrap();

    let bin = dir.join("positive");
    assert!(compile(&w, &bin, false), "compiling tests/fr_paths.cpp");
    let (ok, out) = run(&bin);
    assert!(ok, "{out}");
    assert!(out.starts_with("PASS"), "{out}");
}

/// The negative half of the same comparison, which is why the patch excludes it. If this
/// ever starts passing then `sub_s1l2n` reduces after all and the restriction in
/// `patch_fr_generic` can be lifted.
#[test]
fn negative_shorts_do_not_agree_and_that_is_why_they_are_excluded() {
    let Some(w) = witnesscalc() else { return };
    let dir = std::env::temp_dir().join("g16-csp-fr-paths");
    std::fs::create_dir_all(&dir).unwrap();

    let bin = dir.join("negative");
    assert!(compile(&w, &bin, true), "compiling tests/fr_paths.cpp");
    let (ok, out) = run(&bin);
    assert!(
        !ok && out.starts_with("FAIL"),
        "negative shorts agreed with the long form, which the patch assumes they do not:\n{out}"
    );
}
