//! Print a curve's generated point module, so the shader an iPhone refuses to run can be
//! read, diffed, and pasted straight into a page that runs it without wasm in the way.
//!
//! `cargo run -p snarkrs-wgpu --example dump_points_wgsl -- g2 > g2.wgsl`, and a second argument
//! sets `gen::points::POINT_BODY`, which is what a device that cannot compile the
//! straight-line form needs read back.
fn main() {
    let curve = match std::env::args().nth(1).as_deref() {
        Some("g1") => snarkrs_wgpu::gen::points::G1,
        _ => snarkrs_wgpu::gen::points::G2,
    };
    if let Some(n) = std::env::args().nth(2).and_then(|s| s.parse().ok()) {
        // Run the environment seed first, so the argument is what wins rather than what
        // `G16_WGPU_POINT_BODY` happens to be in this shell.
        snarkrs_wgpu::gen::points::point_body();
        snarkrs_wgpu::gen::points::POINT_BODY.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    print!(
        "{}",
        snarkrs_wgpu::gen::points::points_module(snarkrs_wgpu::gen::Variant::Cios32Unrolled, curve)
    );
}
