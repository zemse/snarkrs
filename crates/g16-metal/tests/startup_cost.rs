//! What `MetalBackend::new()` actually costs, cold and warm.
//!
//! Written for the Metal review. `field_gpu::msl_runtime_compile_cost` salts the source
//! so it always measures an uncached compile, but it reports `newLibraryWithSource` and
//! the pipeline-state loop as two numbers without saying which of them a real first run
//! pays. They are not the same kind of cost:
//!
//! * `newLibraryWithSource` compiles MSL to AIR, the portable IR.
//! * `newComputePipelineStateWithFunction` compiles AIR to this GPU's machine code.
//!
//! Both are cached by macOS keyed on the source, and **the second is by far the larger
//! of the two on a cold cache**. Quoting a fast pipeline-state number taken with a warm
//! cache next to a slow compile number taken with a cold one mixes the two regimes and
//! understates first-launch startup several-fold, which is exactly the number that
//! matters on a client device the first time the app is opened.
//!
//! Run with `cargo test -p g16-metal --release --test startup_cost -- --nocapture`.

#![cfg(target_os = "macos")]

use g16_metal::MetalBackend;
use metal::{CompileOptions, Device};
use std::time::Instant;

/// A small but genuinely compiled kernel. Big enough that AIR-to-ISA is real work,
/// small enough that the test stays quick.
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k0(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    uint a = i; for (uint j = 0; j < 8; ++j) { a = a * 1664525u + 1013904223u; } o[i] = a;
}
kernel void k1(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    uint a = i ^ 0x9E3779B9u; for (uint j = 0; j < 8; ++j) { a = a * 22695477u + 1u; } o[i] = a;
}
"#;

fn salted() -> String {
    let salt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("// salt {salt}\n{SRC}")
}

/// The two halves of a cold compile, then the same source again warm.
#[test]
fn cold_and_warm_compile_are_different_regimes() {
    let device = Device::system_default().expect("no Metal device");
    let src = salted();

    let t = Instant::now();
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile failed");
    let cold_lib = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    for n in ["k0", "k1"] {
        let f = lib.get_function(n, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap();
    }
    let cold_pso = t.elapsed().as_secs_f64() * 1e3;

    // Byte-identical source, same process: both layers should now hit the cache.
    let t = Instant::now();
    let lib2 = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("recompile failed");
    let warm_lib = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    for n in ["k0", "k1"] {
        let f = lib2.get_function(n, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap();
    }
    let warm_pso = t.elapsed().as_secs_f64() * 1e3;

    println!("STARTUP cold  library {cold_lib:7.1} ms   2 pipeline states {cold_pso:7.1} ms   total {:7.1} ms", cold_lib + cold_pso);
    println!("STARTUP warm  library {warm_lib:7.1} ms   2 pipeline states {warm_pso:7.1} ms   total {:7.1} ms", warm_lib + warm_pso);
    println!(
        "STARTUP pipeline-state creation is {:.0}x more expensive cold than warm",
        cold_pso / warm_pso.max(1e-3)
    );
}

/// What the prover actually pays to stand the backend up, twice in one process.
///
/// The second call is the number a resident service sees; the first is what a freshly
/// launched CLI sees with an already-warm OS shader cache. Neither is the true
/// first-ever-launch cost, which is the cold number in the test above.
#[test]
fn metal_backend_new_cost() {
    let t = Instant::now();
    let _b1 = MetalBackend::new().expect("no Metal device");
    let first = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let _b2 = MetalBackend::new().expect("no Metal device");
    let second = t.elapsed().as_secs_f64() * 1e3;

    println!("BACKEND MetalBackend::new() first {first:.1} ms, second in-process {second:.1} ms");
}
