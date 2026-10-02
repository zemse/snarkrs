use std::fmt::Display;
use std::path::Path;

pub fn required() -> bool {
    std::env::var_os("G16_REQUIRE_TESTS").as_deref() == Some(std::ffi::OsStr::new("1"))
}

#[track_caller]
pub fn skip(reason: impl Display) {
    assert!(!required(), "G16_REQUIRE_TESTS=1: {reason}");
    eprintln!("SKIPPED: {reason} (set G16_REQUIRE_TESTS=1 to require coverage)");
}

#[allow(dead_code)]
#[track_caller]
pub fn skip_vector(reason: impl Display) {
    assert!(
        std::env::var_os("G16_REQUIRE_VECTORS").is_none(),
        "G16_REQUIRE_VECTORS is set: {reason}"
    );
    skip(reason);
}

#[allow(dead_code)]
pub fn complete_files(dir: &Path, files: &[&str], test: &str) -> bool {
    for file in files {
        let path = dir.join(file);
        if !path.is_file() {
            skip(format_args!("{test}: missing {}", path.display()));
            return false;
        }
    }
    true
}
