//! Emits `cfg(unoptimized)` so [`prove`](../src/prove.rs) can refuse to run at opt-level 0.
//!
//! There is no stable `cfg` for the optimisation level, and `debug_assertions` is not a
//! stand-in for one: a test profile that sets `opt-level = 3` alongside `overflow-checks`
//! wants the checks and the speed at once, and keying off `debug_assertions` would refuse
//! that build. `OPT_LEVEL` is the real thing, and cargo hands it to build scripts as the
//! level the package is being built at.
//!
//! Level 1 counts as unoptimised on purpose. It is far closer to 0 than to 2 for the CIOS
//! multiply, which is the loop that decides proving time.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(unoptimized)");
    let level = std::env::var("OPT_LEVEL").unwrap_or_default();
    // For the `ProveError::Unoptimized` message, which names the level it refused.
    println!("cargo::rustc-env=SNARKRS_OPT_LEVEL={level}");
    if matches!(level.as_str(), "0" | "1") {
        println!("cargo::rustc-cfg=unoptimized");
    }
}
