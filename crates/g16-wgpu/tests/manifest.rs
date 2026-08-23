//! One check, on this crate's own `Cargo.toml`, that no compiler error can state clearly.
//!
//! Host only. It parses text and opens no device, so it runs everywhere `cargo test` runs.

use std::path::Path;

/// `crate-type` must not contain `cdylib` while the workspace release profile is
/// `panic = "abort"`, because the combination breaks `cargo test --workspace --release` and
/// the error it produces names the wrong crate.
///
/// # The mechanism, because the failure does not describe itself
///
/// Cargo drops `-C extra-filename` for a lib target whose `crate-type` list contains
/// `cdylib`, so the shared object gets a predictable name. Both crate types come out of one
/// rustc invocation, so the `rlib` loses its hash too and lands at
/// `target/release/deps/libg16_wgpu.rlib` with no unit identity in the name. The workspace
/// release profile sets `panic = "abort"`; a test harness needs `unwind`; so every library in
/// a `--release` test build is compiled twice, and with `cdylib` present both units write
/// that one path. Cargo warns ("output filename collision", cargo issue 6313, "this may
/// become a hard error") and then a downstream unit fails.
///
/// Measured at U11, with `g16-wgpu` in `g16-cli`'s default feature set:
/// `cargo build --workspace --release --tests` failed **5 times out of 5** after touching
/// `crates/g16-wgpu/src/lib.rs`, with `error[E0463]: can't find crate for g16_cli` pointing at
/// `crates/g16-cli/src/main.rs`. Nothing in that message mentions `g16-wgpu`, a crate type, or
/// a panic strategy. With `crate-type = ["rlib"]` the same command passed 3 times out of 3
/// and emitted no collision warning.
///
/// Earlier notes twice specified `["cdylib", "rlib"]`, and
/// they were written against a scratch crate that nothing else in a workspace depended on.
/// U13's browser entry point still needs `cdylib`; it needs wasm-bindgen, js-sys and web-sys
/// too, none of which belong in the native tree, so it gets its own wrapper crate.
#[test]
fn the_lib_is_rlib_only_while_the_release_profile_aborts() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).expect("this crate's Cargo.toml");
    let crate_type = text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("crate-type"))
        .expect("[lib] crate-type is set explicitly, so that this test has something to read");

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
    let aborts = std::fs::read_to_string(&workspace)
        .expect("the workspace Cargo.toml")
        .split("[profile.release]")
        .nth(1)
        .and_then(|rest| rest.split("\n[").next())
        .is_some_and(|section| section.contains("panic = \"abort\""));

    assert!(
        !(aborts && crate_type.contains("cdylib")),
        "[lib] {crate_type} and the workspace release profile is panic = \"abort\". Those two \
         together make cargo write one unhashed libg16_wgpu.rlib from two units, and \
         `cargo test --workspace --release` then fails with `can't find crate for g16_cli`. \
         Put the cdylib in a wasm wrapper crate instead."
    );
}
