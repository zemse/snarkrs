//! Print the generated stages 1 to 3 module for a list of tile sizes, so a shader can be
//! read, diffed or pasted into a WGSL playground without a GPU in the loop.
//!
//! `cargo run -p g16-wgpu --example dump_ntt_wgsl -- 6 9`, defaulting to the tiles a 2^18
//! domain produces at the shipped pass cap.
fn main() {
    let ks: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ks = if ks.is_empty() { vec![6] } else { ks };
    print!(
        "{}",
        g16_wgpu::gen::ntt::ntt_module(g16_wgpu::gen::Variant::Cios32Unrolled, &ks)
    );
}
