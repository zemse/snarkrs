//! Compile and link the circom witness generators.
//!
//! The `.cpp`/`.dat` pairs are the ones the upstream benchmark ships, so the witness this
//! crate computes is the same witness the `circom` row computes. They are not vendored
//! into this repo: `bench/scripts/csp-fetch.sh` clones them, and this build reads them
//! from that checkout.
//!
//! `ecdsa_32` is the exception upstream makes too: 57 MB of generated C++ is not in their
//! tree either, so `csp-fetch.sh` runs circom over `ecdsa_32.circom` to produce it in
//! place, with the same `--O2 --c` upstream's own build script uses.
//!
//! The witnesscalc checkout is patched before it is built; see [`patch_fr`].

use std::path::{Path, PathBuf};
use std::process::Command;

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
    "ecdsa_32",
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

    // Clone before `build_and_link` would, so there is a checkout to patch. It only
    // clones when the directory is absent, so it finds this one and goes straight to the
    // build.
    let witnesscalc = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("witnesscalc");
    if !witnesscalc.exists() {
        clone_witnesscalc(&witnesscalc);
    }
    println!("cargo:rerun-if-env-changed=G16_CSP_STOCK_FR");
    if std::env::var_os("G16_CSP_STOCK_FR").is_none() {
        patch_fr(&witnesscalc);
    }

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

fn clone_witnesscalc(dir: &Path) {
    let run = |args: &[&str], cwd: Option<&Path>| {
        let mut c = Command::new("git");
        c.args(args);
        if let Some(d) = cwd {
            c.current_dir(d);
        }
        let ok = c.status().expect("git is not on PATH").success();
        assert!(ok, "git {} failed", args.join(" "));
    };
    run(
        &[
            "clone",
            "https://github.com/zkmopro/witnesscalc.git",
            dir.to_str().unwrap(),
        ],
        None,
    );
    run(&["submodule", "update", "--init", "--recursive"], Some(dir));
}

/// Give `Fr_mod` and `Fr_idiv` a fast path for small non-negative operands.
///
/// circom compiles witness-time arithmetic on `var`s to these two calls, so a keccak
/// round's `(x + 1) % 5` and a rotation's `r % 64` arrive here as two small non-negative
/// integers. The shipped implementation answers them with three `mpz_init`/`mpz_clear`
/// pairs and a multi-precision division, which measures 53.7 ns for a remainder that fits
/// in a register and accounts for 38% of keccak_2048's witness generation. `Fr_toMpz`
/// normalises every element into [0, q) before dividing, so when neither operand is long
/// and both are non-negative the machine remainder is that same value, and `Fr_fromMpz`
/// would have tagged the result `Fr_SHORT` because it fits.
///
/// A short element with a negative `shortVal` stands for `shortVal + q`, a 254 bit
/// number, so that case falls through to the general path along with every long operand.
/// `ecdsa_32`'s bigint helpers divide by 2^64 and stay on the general path throughout.
///
/// Set `G16_CSP_STOCK_FR` to build the checkout unpatched, which is how the before number
/// in `bench/results/csp` was measured.
fn patch_fr(witnesscalc: &Path) {
    const MARKER: &str = "g16: short-operand fast path";
    let fr = witnesscalc.join("build/fr.cpp");
    let src = std::fs::read_to_string(&fr).expect("witnesscalc checkout has no build/fr.cpp");
    if src.contains(MARKER) {
        return;
    }
    let mut out = src;
    for (name, op) in [("Fr_mod", "%"), ("Fr_idiv", "/")] {
        let sig = format!("void {name}(PFrElement r, PFrElement a, PFrElement b) {{");
        let at = out
            .find(&sig)
            .unwrap_or_else(|| panic!("{name} is not declared the way this patch expects"));
        let fast = format!(
            "\n    // {MARKER}\n\
             \x20   if (!((a->type | b->type) & Fr_LONG) && a->shortVal >= 0 && b->shortVal > 0) {{\n\
             \x20       int32_t v = a->shortVal {op} b->shortVal;\n\
             \x20       r->type = Fr_SHORT;\n\
             \x20       r->shortVal = v;\n\
             \x20       return;\n\
             \x20   }}\n"
        );
        out.insert_str(at + sig.len(), &fast);
    }
    std::fs::write(&fr, out).expect("writing the patched build/fr.cpp");
}
