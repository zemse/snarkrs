//! Compile and link the circom witness generators.
//!
//! The `.cpp`/`.dat` pairs are the ones the upstream benchmark ships, so the witness this
//! crate computes is the same witness the `circom` row computes. They are not vendored
//! into this repo: `bench/scripts/csp-fetch.sh` clones them, and this build reads them
//! from that checkout.

use std::path::{Path, PathBuf};

/// Circuits whose generator is linked in. Keep in step with `witness!` in `circuits.rs`.
const CIRCUITS: &[&str] = &[
    "sha256_128",
    "sha256_256",
    "sha256_512",
    "sha256_1024",
    "sha256_2048",
    "keccak_128",
    "keccak_256",
    "keccak_512",
    "keccak_1024",
    "keccak_2048",
    "poseidon_2",
    "poseidon_4",
    "poseidon_8",
    "poseidon_12",
    "poseidon_16",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("../vendor/csp-benchmarks/circom/circuits");
    let staged = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("circuits");

    // witnesscalc-adapter compiles every `.cpp` in one flat directory, and upstream keeps
    // each circuit in a subdirectory of its own. Hard link rather than copy: poseidon_16
    // alone is 20 MB of generated C++.
    std::fs::create_dir_all(&staged).unwrap();
    for name in CIRCUITS {
        let family = name.rsplit_once('_').expect("name is family_size").0;
        for ext in ["cpp", "dat"] {
            let from = src.join(family).join(name).join(format!("{name}.{ext}"));
            if !from.is_file() {
                panic!(
                    "missing {}\nrun bench/scripts/csp-fetch.sh first",
                    from.display()
                );
            }
            link(&from, &staged.join(format!("{name}.{ext}")));
        }
    }
    println!("cargo:rerun-if-changed={}", src.display());

    witnesscalc_adapter::build_and_link(staged.to_str().unwrap());

    // `build_and_link` emits `-l dylib=witnesscalc_<circuit>` but no rpath, so without
    // this the binary links and then dies at start-up with "no LC_RPATH's found".
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for sub in [
        "witnesscalc/package/lib",
        "witnesscalc/build_witnesscalc/src",
    ] {
        let dir = out.join(sub);
        if dir.is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}

fn link(from: &Path, to: &Path) {
    if to.exists() {
        std::fs::remove_file(to).ok();
    }
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to).expect("staging witness generator source");
    }
}
