//! Stage 4 in WGSL, standalone: `H = A*B - C` elementwise, written in both encodings.
//!
//! A port of `g16-metal/src/shaders/pointwise.metal`'s `g16_h_join`, where it is a
//! reference path only. **Here it is what the prover dispatches**, which is not what design
//! §4 and §8 specify and is the result of measuring them: the fused alternative
//! ([`crate::gen::ntt`]'s `ntt_*_join_k{K}`, stage 4 folded into the store epilogue of the
//! last batch of C's forward transform) moves 29% less data and runs 5% to 9% slower on
//! every artifact. The table and the argument are on [`crate::stages::Stage4`]; both paths
//! are kept, both are tested against `g16_core::cpu`, and the default is asserted to be the
//! faster of the two rather than pinned to today's answer.
//!
//! It is also the only way to get the three coset vectors materialised on the device, since
//! the fused path deliberately never writes C's final transform.
//!
//! # No division by Z(coset), and that is not an omission
//!
//! snarkjs' `joinABC` computes exactly `a*b - c` and feeds it straight to the H multiexp.
//! The division by the vanishing polynomial is folded into the section 9 bases at setup:
//! they are the odd Lagrange polynomials of the 2n domain, and `P = A*B - C` vanishes on the
//! even points, so `sum_i P(inc^(2i+1)) * hExps[i]` already equals `[P(tau)]_1`, which is
//! `[H(tau) * Z(tau)]_1`. Dividing here as well double counts Z and produces a proof that
//! fails verification with nothing else to go on.
//! `g16_core::cpu::CpuCircuit::compute_h` makes the same choice and the same argument.
//!
//! # Two output buffers, and why both
//!
//! `h_mont` is Montgomery limbs, which is `g16_gpu_layout::PackedFr` and arkworks' own
//! internal form, so a host comparison against the CPU backend is a byte comparison.
//! `h_std` is the standard form integer in `[0, r)`, which is `PackedScalar` and is what
//! stage 9's Pippenger has to slice into window digits: a window digit of a Montgomery
//! representative is a digit of `a*R mod r`, a different number. Getting the two backwards
//! yields a proof wrong by a factor of R that fails verification with nothing else to go on,
//! which is why `tests/stages.rs` checks each buffer against its own host encoder rather
//! than checking that the two differ.
//!
//! The alternative is one buffer plus a conversion dispatch later, costing a whole extra
//! read and write of the domain (`2n * 32` bytes) plus a dispatch, against `n * 32` bytes of
//! extra writes here. On a stage that is already memory bound the second write is the
//! cheaper of the two.
//!
//! # The element range is in the uniform block for the same reason stage 0's is
//!
//! `maxComputeWorkgroupsPerDimension` is 65535 at every tier of every browser, so at
//! [`WORKGROUP`] threads a single dispatch covers 16,776,960 elements and a 2^25 domain does
//! not fit. The kernel takes `[lo, hi)` and the host emits as many dispatches as it needs;
//! below 2^24 that is one dispatch and the parameter costs one add.

use std::fmt::Write as _;

use g16_gpu_layout::LIMBS;

use crate::gen::field::Variant;

/// Threads per workgroup, emitted as a literal so the shader and the host dispatch count
/// cannot disagree.
///
/// **256, which is what design §4 says, and it is the first inherited constant in this crate
/// that has survived being measured.** Microseconds for one standalone stage 4 over the whole
/// domain, submit overhead differenced out,
/// `tests/stages.rs::the_standalone_join_workgroup_size_is_measured`, M2 Max through naga to
/// MSL. Per-cell medians of four **release** runs:
///
/// ```text
/// domain      32     64    128    256
/// 2^12      30.0   17.0   19.5   15.0
/// 2^14      22.0   26.0   24.5   24.0
/// 2^15      44.5   37.0   35.0   32.5
/// 2^17     118.0  116.5  112.5  111.0
/// 2^18     212.5  209.0  210.0  206.5
/// total    427.0  405.5  401.5  389.0
/// ```
///
/// 256 has the lowest total and is best or tied at four of the five domains. 32 is 10% off
/// at 2^12 and 37% off at 2^15 and is the only column that is ever clearly wrong.
///
/// # The debug build says the opposite, and that is a measurement bug, not a result
///
/// Run the same sweep under `cargo test` without `--release` and 128 wins with 256 up to 37%
/// behind at 2^12. The harness times `1 + 40` encodes of the dispatch against `1` and divides
/// the difference by 40, which cancels the *fixed* cost of a submit and a fence but not the
/// **per-encode host cost**, and that scales with the pass count exactly as the kernel does.
/// In a debug build `ComputePass::set_bind_group` plus `dispatch_workgroups` is expensive
/// enough to swamp a 15-microsecond kernel: the same 2^12 cell reads 58 to 80 us in debug and
/// 15 in release, and even 2^18 reads 250 against 207.
///
/// So this sweep is only meaningful in release, and the test says so. Worth knowing more
/// widely: `tests/gather.rs` and `tests/ntt.rs` use the same differencing harness, and their
/// published tables were taken in debug. The NTT's numbers are milliseconds and the effect is
/// small there, but stage 0's smallest cells are around 1 ms and could move. Filed in
/// `TASKS.md`.
///
/// **The honest headline is still that none of this matters.** The whole kernel is 0.21 ms at
/// 2^18 and the entire spread across four workgroup sizes is 38 microseconds, against about
/// 21 ms for stages 0 to 4 at that domain.
///
/// naga to MSL on Apple silicon. Chrome compiles through Tint, so the ranking may differ
/// there, which is why the size stays a parameter of [`h_join_module_at`].
pub const WORKGROUP: u32 = 256;

/// The one entry point this module defines.
pub const ENTRY: &str = "h_join";

/// Storage buffers the entry point declares. The floor allows 8.
///
/// Asserted against the generated source and against the bind group layout by
/// `tests/stages.rs`, so this constant cannot drift away from either.
pub const STORAGE_BUFFERS: u32 = 5;

// Binding numbers, in group 0. Shared with [`crate::pointwise`] so the layout, the bind
// group and the shader are built from one list.
pub const BIND_PARAMS: u32 = 0;
pub const BIND_A: u32 = 1;
pub const BIND_B: u32 = 2;
pub const BIND_C: u32 = 3;
pub const BIND_H_MONT: u32 = 4;
pub const BIND_H_STD: u32 = 5;

/// The whole standalone stage 4 module at [`WORKGROUP`].
pub fn h_join_module(v: Variant) -> String {
    h_join_module_at(v, WORKGROUP)
}

/// Same, at a chosen workgroup size.
///
/// # Panics
///
/// On [`Variant::NoCarry13x20`], for the same reason `gen::gather` and `gen::ntt` do: its
/// `R = 2^260` wire format is not `g16_gpu_layout::PackedFr` and no host packer produces it,
/// so the kernel would read 20-limb elements out of a buffer written as 8-limb ones and
/// produce plausible garbage.
///
/// On a workgroup size outside `1..=256`, which is `maxComputeInvocationsPerWorkgroup` at
/// the floor. This adapter allows 1024 and no browser does, so the check is against the
/// floor and not against the device.
pub fn h_join_module_at(v: Variant, workgroup: u32) -> String {
    assert_eq!(
        v.limbs(),
        LIMBS,
        "stage 4 reads and writes g16_gpu_layout::PackedFr, which is {LIMBS} limbs; \
         variant {v:?} is {} and has no host packer",
        v.limbs()
    );
    assert!(
        workgroup > 0 && workgroup <= 256,
        "workgroup size {workgroup} is outside 1..=256, the floor's \
         maxComputeInvocationsPerWorkgroup"
    );

    // Only the `Fr` half of the prelude, exactly as `gen::ntt` does. `Fq` and `Fq2` are
    // about 55 KiB of straight-line multiply that this kernel never calls, and
    // `crate::pipelines` records 155 to 174 ms of naga time for a 29 KiB single-field
    // module, so shipping the curve field here would multiply this module's compile for
    // nothing.
    let mut s = format!(
        "// Generated by g16-wgpu::gen::pointwise, variant {v:?}, workgroup {workgroup}.\n\
         // Do not edit by hand.\n"
    );
    if v.needs_mul64() {
        s.push_str(crate::gen::field::MUL64);
    }
    s.push_str(&crate::gen::field::FR.ops(v));
    s.push_str(&entry_point_at(workgroup));
    s
}

/// Just the entry point, assuming the `Fr` prelude is already in scope.
///
/// Separate from [`h_join_module_at`] so a later unit can concatenate stage 4 into a bigger
/// module over one copy of the prelude if the compile cost ever argues for it.
pub fn entry_point_at(workgroup: u32) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "
// ---------------------------------------------------------------------------
// Stage 4, standalone. Generated by g16-wgpu::gen::pointwise; do not edit by hand.
// The prover fuses this into the NTT store epilogue instead; see the module docs.
// ---------------------------------------------------------------------------

// Mirrors g16_wgpu::pointwise::HJoinParams. 16 bytes, padded to the 16-byte alignment WGSL
// gives every uniform struct, so the two are obviously the same size rather than
// accidentally the same size.
struct HJoinParams {{
    lo: u32,
    hi: u32,
    pad0: u32,
    pad1: u32,
}};

@group(0) @binding({BIND_PARAMS}) var<uniform> P: HJoinParams;
@group(0) @binding({BIND_A}) var<storage, read> A: array<Fr>;
@group(0) @binding({BIND_B}) var<storage, read> B: array<Fr>;
@group(0) @binding({BIND_C}) var<storage, read> C: array<Fr>;
@group(0) @binding({BIND_H_MONT}) var<storage, read_write> H_MONT: array<Fr>;
@group(0) @binding({BIND_H_STD}) var<storage, read_write> H_STD: array<Fr>;

@compute @workgroup_size({workgroup})
fn {ENTRY}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    // The last workgroup of a dispatch runs past hi whenever the element count is not a
    // multiple of {workgroup}. A domain is a power of two so that only bites on a chunk
    // boundary, but a chunk boundary is exactly where an unguarded write lands in the next
    // chunk's data rather than off the end of the buffer, where it would be dropped.
    // tests/stages.rs binds an output one element longer than the range and checks the
    // slack, because with exactly-sized buffers an over-run here is invisible.
    if (i >= P.hi) {{ return; }}

    // A*B - C, and no division by Z(coset). See the module docs; this is snarkjs' joinABC
    // and the Z is already folded into the zkey's section 9 bases.
    let h = fr_sub(fr_mul(A[i], B[i]), C[i]);
    H_MONT[i] = h;
    // fr_from_mont is a Montgomery multiply by the integer 1, so this is one multiply and
    // no branch. Standard form is what stage 9 slices into window digits.
    H_STD[i] = fr_from_mont(h);
}}
"
    )
    .unwrap();
    s
}
