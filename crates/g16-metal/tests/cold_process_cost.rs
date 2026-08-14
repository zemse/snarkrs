// The `metal` crate is a macOS-only dependency, so on any other target `g16_metal`
// exports nothing and this file cannot compile. Gate the whole test crate rather
// than each test: `cargo test --workspace` is run on Linux by the EC2 sweep, and a
// macOS-only test crate breaking it there hides real failures behind a build error.
#![cfg(target_os = "macos")]

//! Where the ~20 ms Metal prepare floor actually goes, measured in a FRESH process.
//!
//! `startup_cost.rs` reports `MetalBackend::new()` at ~2.7 ms, but it runs after another
//! test in the same process has already created an `MTLDevice`. That hides the one-time
//! Metal runtime + driver initialisation, which a real `g16 prove` invocation pays.
//! Run alone: `cargo test -p g16-metal --release --test cold_process_cost -- --nocapture`.

use std::time::Instant;

#[test]
fn first_device_in_process_dominates_the_prepare_floor() {
    let t0 = Instant::now();
    let device = metal::Device::system_default().expect("no Metal device");
    let device_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let backend = g16_metal::MetalBackend::with_device(device.clone()).expect("backend");
    let new_ms = t1.elapsed().as_secs_f64() * 1e3;

    // A second device handle, to show the first call is the one that pays.
    let t2 = Instant::now();
    let _d2 = metal::Device::system_default().expect("device");
    let device2_ms = t2.elapsed().as_secs_f64() * 1e3;

    // A second backend, to separate cached-shader compile from runtime init.
    let t3 = Instant::now();
    let _b2 = g16_metal::MetalBackend::with_device(device).expect("backend");
    let new2_ms = t3.elapsed().as_secs_f64() * 1e3;

    println!("COLD-PROCESS Device::system_default() first {device_ms:8.2} ms, second {device2_ms:8.2} ms");
    println!(
        "COLD-PROCESS MetalBackend::new()        first {new_ms:8.2} ms, second {new2_ms:8.2} ms"
    );
    println!(
        "COLD-PROCESS one-time Metal floor            {:8.2} ms",
        device_ms + new_ms
    );
    let _ = backend;
}
