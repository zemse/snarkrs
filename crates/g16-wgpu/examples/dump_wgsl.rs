//! Print the generated field prelude, so a shader can be read, diffed or pasted into a
//! WGSL playground without a GPU in the loop. `cargo run -p g16-wgpu --example dump_wgsl`.
fn main() {
    let variant = match std::env::args().nth(1).as_deref() {
        Some("nocarry") => g16_wgpu::gen::Variant::NoCarry13x20,
        _ => g16_wgpu::gen::Variant::Cios32Unrolled,
    };
    print!("{}", g16_wgpu::gen::field_module(variant));
}
