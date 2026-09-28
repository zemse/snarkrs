//! Print the generated field prelude, so a shader can be read, diffed or pasted into a
//! WGSL playground without a GPU in the loop. `cargo run -p snarkrs-wgpu --example dump_wgsl`.
fn main() {
    let variant = match std::env::args().nth(1).as_deref() {
        Some("nocarry") => snarkrs_wgpu::gen::Variant::NoCarry13x20,
        _ => snarkrs_wgpu::gen::Variant::Cios32Unrolled,
    };
    print!("{}", snarkrs_wgpu::gen::field_module(variant));
}
