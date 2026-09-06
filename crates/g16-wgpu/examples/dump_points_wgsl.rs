//! Print a curve's generated point module, so the shader an iPhone refuses to run can be
//! read, diffed, and pasted straight into a page that runs it without wasm in the way.
//!
//! `cargo run -p g16-wgpu --example dump_points_wgsl -- g2 > g2.wgsl`
fn main() {
    let curve = match std::env::args().nth(1).as_deref() {
        Some("g1") => g16_wgpu::gen::points::G1,
        _ => g16_wgpu::gen::points::G2,
    };
    print!(
        "{}",
        g16_wgpu::gen::points::points_module(g16_wgpu::gen::Variant::Cios32Unrolled, curve)
    );
}
