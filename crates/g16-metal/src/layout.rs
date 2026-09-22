//! The Metal end of the packed layouts that cross the Rust/GPU boundary.
//!
//! The types themselves are [`g16_gpu_layout`]'s and are re-exported here rather than
//! declared again, because the CUDA backend reads the identical bytes: two Rust
//! definitions that merely happen to agree would let a fix to one of them silently
//! invalidate the Metal-vs-CUDA comparison this project exists to make. What stays here
//! is the part that is Metal's: the import path this crate's drivers use, and the drift
//! guard below. See that crate's module docs for why an explicit repack is unavoidable
//! (`ark_ec::G1Affine` is 72 bytes, not 64) and for the Montgomery-versus-standard split
//! between [`PackedFr`] and [`PackedScalar`].
//!
//! # This module and `src/shaders/*.metal` must be changed together
//!
//! Every `#[repr(C)]` type re-exported below has a byte-for-byte twin declared in
//! [`crate::kernels::FR_MSL`] (`src/shaders/bn254_fr.metal`). The MSL side carries a
//! `static_assert` on each `sizeof`, `g16-gpu-layout` carries a `const` assertion on each
//! `size_of`, and [`tests::msl_declares_the_same_constants`] greps the shader source for
//! the exact constant lines so the two cannot drift silently. If you change a struct in
//! `g16-gpu-layout` you are changing a wire format two GPUs read; go change this shader
//! and the CUDA header in the same commit.

pub use g16_gpu_layout::{
    as_bytes, as_bytes_mut, Packed, PackedFq, PackedFq2, PackedFr, PackedG1Affine, PackedG2Affine,
    PackedGlv, PackedScalar, FQ_MODULUS, FQ_N0, FR_MODULUS, FR_N0, LIMBS,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The drift guard. The shader carries the same constants as literal text; if either
    /// side is edited alone this fails. The packing itself is tested in `g16-gpu-layout`,
    /// where it is written.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_declares_the_same_constants() {
        let src = crate::kernels::FR_MSL;
        let want_n = format!(
            "constant uint FR_N[8] = {{ {} }};",
            FR_MODULUS
                .iter()
                .map(|l| format!("0x{l:08x}u"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(
            src.contains(&want_n),
            "shaders/bn254_fr.metal does not contain the line:\n{want_n}"
        );
        let want_n0 = format!("constant uint FR_N0 = 0x{FR_N0:08x}u;");
        assert!(
            src.contains(&want_n0),
            "shaders/bn254_fr.metal does not contain the line:\n{want_n0}"
        );
        assert!(
            src.contains("struct Fr {\n    uint v[8];\n};"),
            "the MSL Fr struct no longer mirrors PackedFr"
        );
    }
}
