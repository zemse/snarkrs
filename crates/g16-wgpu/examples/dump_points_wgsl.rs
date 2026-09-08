//! Print a curve's generated point module, so the shader an iPhone refuses to run can be
//! read, diffed, and pasted straight into a page that runs it without wasm in the way.
//!
//! `cargo run -p g16-wgpu --example dump_points_wgsl -- g2 > g2.wgsl`, and a second argument
//! sets `gen::points::POINT_BODY`, which is what a device that cannot compile the
//! straight-line form needs read back.
fn main() {
    let curve = match std::env::args().nth(1).as_deref() {
        Some("g1") => g16_wgpu::gen::points::G1,
        _ => g16_wgpu::gen::points::G2,
    };
    if let Some(n) = std::env::args().nth(2).and_then(|s| s.parse().ok()) {
        g16_wgpu::gen::points::POINT_BODY.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    print!(
        "{}",
        g16_wgpu::gen::points::points_module(g16_wgpu::gen::Variant::Cios32Unrolled, curve)
    );
}
