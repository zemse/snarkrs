//! The known-answer battery in `src/selftest.rs`, run on this machine's GPU.
//!
//! That file is what stands between a browser that miscompiles this crate's WGSL and a proof
//! that verifies nowhere, and until this test existed nothing ran it outside a browser. Two
//! of its seven checks run on every page load; the other five run only when a visitor is
//! handed `?selftest=1` by hand, which is to say almost never.
//!
//! The expectations are not trivial reads of the kernel either. They are host-side
//! derivations of the same computation: 200,000 dependent xorshift rounds mirrored in Rust,
//! a per-workgroup barrier sum, a 64-bit product split against the emulated `mul64`, three
//! dynamic uniform offsets that have to land on three different parameter blocks. A `want`
//! that drifted to agree with a wrong kernel would pass forever, and the guard would have
//! stopped guarding with nothing to say so.
//!
//! This machine can run wgpu and the whole battery costs about a second, so the whole thing
//! runs here on every `cargo test`. What that catches is a check that has stopped agreeing
//! with a GPU known to be right; what it cannot catch is a check that has stopped being able
//! to fail at all, which is why the mismatch reporting itself is pinned separately in the
//! unit test beside `compare`.
//!
//! Native only, for the reason in `tests/device.rs`: there is no wasm test runner here.

use std::sync::OnceLock;

use g16_wgpu::selftest::{as_error, first_failure, guard, run, run_json, Battery};
use g16_wgpu::{Check, LimitsProfile, WgpuBackend};

// The battery compiles seven throwaway modules and submits seven times, and one of its checks
// asserts that `onSubmittedWorkDone` took at least 200 us. That bound is a floor, so another
// process saturating the GPU can only push it further from failing; the lock is taken because
// a second of dispatches would otherwise land in the middle of someone else's measurement.
// See tests/gpulock.
#[path = "gpulock/mod.rs"]
mod gpulock;

/// The profile a browser gets. `create_prover` opens the device before it runs the guard set,
/// and `Floor` is what the WebGPU spec default grants, so a check that needed anything above
/// it would be a check the browser cannot run.
fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

fn report(checks: &[Check]) -> String {
    checks
        .iter()
        .map(|c| {
            format!(
                "  {:<24} {:>8} us  {}",
                c.name,
                c.us,
                if c.ok { "ok" } else { c.detail.as_str() }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_check_in_the_full_battery_agrees_with_this_gpu() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let checks = pollster::block_on(run(floor(), Battery::Full));
    println!("{}", report(&checks));

    // The order is asserted and not just the set, because it is the diagnostic: the checks
    // are arranged so that the first failure names the layer, and a reordering that put the
    // field prelude before the bare storage write would report "the field is wrong" on a
    // device where no dispatch writes anything at all.
    let names: Vec<&str> = checks.iter().map(|c| c.name).collect();
    assert_eq!(
        names,
        [
            "storage_write",
            "fence_is_a_fence",
            "mul64",
            "field_ops",
            "atomics",
            "workgroup_barrier",
            "dynamic_uniform_offsets",
        ]
    );
    assert_eq!(
        first_failure(&checks),
        None,
        "the battery found this machine's GPU wrong:\n{}",
        report(&checks)
    );
}

#[test]
fn the_guard_set_is_the_two_checks_create_prover_refuses_a_device_on() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let checks = pollster::block_on(run(floor(), Battery::Guard));
    println!("{}", report(&checks));

    // Exactly these two, in this order. `Guard` runs unconditionally on every page load and
    // is the only thing standing in front of the Safari failure it was written for, so
    // quietly dropping the field prelude out of it would cost nothing visible until a
    // browser produced a 3 ms proof again.
    let names: Vec<&str> = checks.iter().map(|c| c.name).collect();
    assert_eq!(names, ["storage_write", "field_ops"]);

    assert!(
        pollster::block_on(guard(floor())).is_ok(),
        "the guard set failed on this machine:\n{}",
        report(&checks)
    );
}

#[test]
fn the_json_the_page_reads_carries_every_check_and_parses() {
    let _gpu_timing = gpulock::exclusive_gpu();
    let text = pollster::block_on(run_json(floor(), Battery::Full));
    println!("{text}");

    // Parsed rather than pattern-matched, because this string is hand-assembled and the
    // browser is the only other thing that reads it. A detail carrying a Metal compiler's
    // diagnostics is where an unescaped character would come from, and there the page would
    // report a syntax error instead of the failure it was built to show.
    let rows: serde_json::Value = serde_json::from_str(&text).expect("selftest JSON");
    let rows = rows.as_array().expect("a JSON array");
    assert_eq!(rows.len(), 7);
    for row in rows {
        assert!(row["name"].as_str().is_some());
        assert!(row["us"].as_u64().is_some());
        assert_eq!(row["ok"], serde_json::Value::Bool(true), "{row}");
    }
}

#[test]
fn a_failed_check_is_named_and_not_swallowed() {
    // No GPU: this is the reporting chain `create_prover` refuses on, and a battery that
    // found a broken device and then returned `Ok` would be worse than no battery at all.
    let checks = vec![
        Check {
            name: "storage_write",
            ok: true,
            detail: String::new(),
            us: 1,
        },
        Check {
            name: "field_ops",
            ok: false,
            detail: "word 0 is 0x00000000, expected 0x00000001".to_string(),
            us: 2,
        },
    ];
    let msg = first_failure(&checks).expect("a failed check must be reported");
    assert!(msg.contains("field_ops"), "{msg}");
    assert!(msg.contains("0x00000001"), "{msg}");
    // Same text, wrapped in the `Backend` variant `create_prover` hands to the page.
    let e = as_error(&checks).unwrap().to_string();
    assert!(e.contains(&msg), "{e}");

    assert_eq!(first_failure(&checks[..1]), None);
}
