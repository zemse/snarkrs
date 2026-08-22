//! Stage 0 in WGSL: the CSR gather of A and B, and the `C = A*B` that no zkey stores.
//!
//! # This is not a translation of the Metal kernel, and it cannot be
//!
//! `g16-metal/src/shaders/gather.metal:60-71` binds **ten** storage buffers: `row_ptr`,
//! `signal` and `value` twice, then the witness and three outputs. The WebGPU spec floor
//! allows 8 per pipeline layout, this adapter reports 9 under
//! `STRICT_WEBGPU_COMPLIANCE` (see `crate::device`), and Chrome's own Metal backend caps at
//! 10 with nothing to spare. Splitting the bindings across bind groups does not help,
//! because `maxStorageBuffersPerShaderStage` counts the whole pipeline layout and
//! `maxBindGroups` is 4 in every browser at every tier.
//!
//! So the A and B CSR are concatenated into one `(row_ptr, signal, value)` triple and the
//! kernel is told where each matrix starts through the uniform block. That is
//! 3 CSR + witness + 3 outputs = **7 storage buffers**, one under the Floor, and the scalar
//! `n` that Metal passes as `buffer(10)` moves into the same uniform block rather than
//! costing a binding of its own.
//!
//! The concatenation is not a repack. `row_ptr`, `signal` and `value` for B are appended to
//! A's **unchanged**, with `row_base_b = domain_size + 1` and `nz_base_b = nnz_a` carried in
//! the uniform, so the device buffers are the host arrays copied end to end and
//! `tests/gather.rs::the_concatenated_csr_is_the_two_matrices_end_to_end` can compare them
//! byte for byte against `pk.coeffs`. Biasing B's `row_ptr` entries by `nnz_a` on the host
//! would save one `u32` add per row and make that check impossible, which is a bad trade at
//! roughly 2 nonzeros per row and 8 to 16 Montgomery multiplies of real work behind it.
//!
//! # One thread per row, which is the Metal kernel's argument and it survives unchanged
//!
//! Measured on the real proving keys, mean row length / empty rows / longest row:
//!
//! ```text
//! js_1x1_d8    domain 2^12   A 2.31 / 732 / 65      B 3.70 / 737 / 64
//! js_2x2_d16   domain 2^14   A 1.51 / 6224 / 67     B 2.34 / 6231 / 66
//! js_8x8_d32   domain 2^17   A 1.13 / 60696 / 79    B 1.70 / 60715 / 78
//! js_16x16_d32 domain 2^18   A 1.13 / 121848 / 95   B 1.69 / 121883 / 94
//! ```
//!
//! The mean row is one or two terms and nearly half the rows are empty, so any mapping that
//! spends a workgroup or a subgroup on a row idles 31 lanes of 32 and then needs a reduction
//! to combine a sum that is almost always a single term. One thread per row instead puts 128
//! consecutive rows in one workgroup, and rows adjacent in a circom circuit have similar
//! lengths, so divergence is bounded by the longest row in the workgroup rather than in the
//! matrix. An empty row costs one comparison.
//!
//! Nothing here writes a location another thread reads: no barrier, no atomic, no ordering
//! requirement. That is what the CSR sort in `g16-zkey` bought.
//!
//! # Why `C` is computed here rather than gathered
//!
//! There is no C matrix in a snarkjs zkey. snarkjs' `buildABC1` fills C by multiplying the A
//! and B evaluations pointwise, which is exact rather than an approximation, because the
//! R1CS constraint *is* `a*b = c` and the rows of the CSR are the constraint rows. Doing it
//! in this kernel reuses two values already in registers, so C costs one Montgomery multiply
//! and zero extra loads. A separate pass would cost `2n * 32` bytes of reads on a stage that
//! is already memory bound.
//!
//! # The row range is in the uniform block, and that is not decoration
//!
//! `maxComputeWorkgroupsPerDimension` is 65535 in every browser at every tier, so at 128
//! threads per workgroup a single dispatch covers 8,388,480 rows. A 2^23 domain is 8,388,608
//! rows, **128 over**, and design §7 already commits to generating artifacts past 2^21. So
//! the kernel takes `[row_lo, row_hi)` and the host emits as many dispatches as it needs.
//! Below 2^23 that is exactly one dispatch and the parameter costs one add. At the 64 the
//! design specified the cliff was at 2^22, one artifact generation away.

use std::fmt::Write as _;

use g16_gpu_layout::LIMBS;

use crate::gen::field::{field_module, Variant};

/// Threads per workgroup, and the divisor the host's dispatch count is computed from.
/// Emitted into the WGSL as a literal, so the shader and the dispatch math cannot disagree.
///
/// **128, not the 64 design §4 specifies, because 64 was measured to lose on every
/// artifact.** `g16-metal` uses a 64-wide threadgroup here and the design inherited that
/// number; MSL and WGSL are not the same compiler and the inheritance does not hold.
/// `tests/gather.rs::the_design_picked_the_workgroup_size_that_wins` sweeps 32, 64, 128 and
/// 256 on every artifact, on this M2 Max through naga to MSL, kernel time in microseconds
/// with submit overhead removed, three runs agreeing to under 1.5%:
///
/// ```text
/// artifact          32     64    128    256
/// js_1x1_d8   2^12  1074   1057   961    920
/// js_2x2_d16  2^14  1048   1105  1017    919
/// js_2x2_d32  2^15  1042   1105   929    978
/// js_8x8_d32  2^17  1749   2142  1949   2051
/// js_16x16    2^18  3119   3321  3075   3082
/// total             8032   8730  7931   7950
/// worst regret      +16.7% +23.9% +11.4% +17.3%
/// ```
///
/// 64 is the slowest column outright and is 24% off the best at `js_8x8_d32`. 128 has both
/// the lowest total and the smallest worst case, so it is what ships. There is no smooth
/// occupancy story behind the shape of those rows and none is offered: the numbers reproduce,
/// the explanation does not exist yet.
///
/// This is naga to MSL on Apple silicon. Chrome compiles through Tint and other platforms
/// emit HLSL or SPIR-V, so the ranking may not hold there, which is why the size stays a
/// parameter of [`entry_point_at`] and the sweep stays a test. U14 should re-run it in the
/// browser.
pub const WORKGROUP: u32 = 128;

/// The one entry point this module defines.
pub const ENTRY: &str = "gather_abc";

/// Storage buffers the entry point declares. The Floor allows 8.
///
/// Asserted against the generated source and against the bind group layout by
/// `tests/gather.rs`, so this constant cannot drift away from either.
pub const STORAGE_BUFFERS: u32 = 7;

/// Binding numbers, in group 0. Shared with [`crate::gather`] so the layout, the bind group
/// and the shader are built from one list.
pub const BIND_PARAMS: u32 = 0;
pub const BIND_ROW_PTR: u32 = 1;
pub const BIND_SIGNAL: u32 = 2;
pub const BIND_VALUE: u32 = 3;
pub const BIND_WITNESS: u32 = 4;
pub const BIND_OUT_A: u32 = 5;
pub const BIND_OUT_B: u32 = 6;
pub const BIND_OUT_C: u32 = 7;

/// The whole stage 0 module at [`WORKGROUP`]: the field prelude followed by [`ENTRY`].
pub fn gather_module(v: Variant) -> String {
    gather_module_at(v, WORKGROUP)
}

/// Same, at a chosen workgroup size.
///
/// # Panics
///
/// On [`Variant::NoCarry13x20`]. That variant's `R` is `2^260` rather than `2^256`, so its
/// wire format is not `g16_gpu_layout::PackedFr` and the host has no packer for it. The
/// panic is deliberate: the alternative is a kernel that reads 20-limb elements out of a
/// buffer written as 8-limb ones and produces plausible garbage.
///
/// Also on a workgroup size of zero or over 256, which is
/// `maxComputeInvocationsPerWorkgroup` at the Floor.
pub fn gather_module_at(v: Variant, workgroup: u32) -> String {
    assert_eq!(
        v.limbs(),
        LIMBS,
        "the gather reads and writes g16_gpu_layout::PackedFr, which is {LIMBS} limbs; \
         variant {v:?} is {} and has no host packer",
        v.limbs()
    );
    let mut s = field_module(v);
    s.push_str(&entry_point_at(workgroup));
    s
}

/// Just the entry point at [`WORKGROUP`], assuming the field prelude is already in scope.
///
/// Separate from [`gather_module`] so U7 can concatenate stage 0 and the NTT into one
/// module over a single copy of the prelude if the compile cost ever argues for it. It does
/// not today: `tests/gather.rs` reports the whole stage 0 module at 5.4 ms of module plus
/// pipeline creation, against the 5 s bar `tests/device.rs` tracks.
pub fn entry_point() -> String {
    entry_point_at(WORKGROUP)
}

/// Just the entry point, at a chosen workgroup size.
pub fn entry_point_at(workgroup: u32) -> String {
    // 256 is `maxComputeInvocationsPerWorkgroup` at the Floor. This adapter allows 1024 and
    // the browser does not, so the check is against the Floor and not against the device.
    assert!(
        workgroup > 0 && workgroup <= 256,
        "workgroup size {workgroup} is outside 1..=256"
    );
    let mut s = String::new();
    writeln!(
        s,
        "
// ---------------------------------------------------------------------------
// Stage 0: the CSR gather. Generated by g16-wgpu::gen::gather; do not edit by hand.
// ---------------------------------------------------------------------------

// A and B live in one triple of buffers. `row_base_*` is where a matrix's row_ptr starts
// inside ROW_PTR, and `nz_base_*` is where its nonzeros start inside SIGNAL and VALUE.
// ROW_PTR entries are stored unbiased, exactly as g16-zkey produced them, so nz_base has
// to be added here rather than folded in on the host. See the module docs.
struct GatherParams {{
    row_lo: u32,
    row_hi: u32,
    row_base_a: u32,
    row_base_b: u32,
    nz_base_a: u32,
    nz_base_b: u32,
    pad0: u32,
    pad1: u32,
}};

@group(0) @binding({BIND_PARAMS}) var<uniform> P: GatherParams;
@group(0) @binding({BIND_ROW_PTR}) var<storage, read> ROW_PTR: array<u32>;
@group(0) @binding({BIND_SIGNAL}) var<storage, read> SIGNAL: array<u32>;
@group(0) @binding({BIND_VALUE}) var<storage, read> VALUE: array<Fr>;
@group(0) @binding({BIND_WITNESS}) var<storage, read> WITNESS: array<Fr>;
@group(0) @binding({BIND_OUT_A}) var<storage, read_write> OUT_A: array<Fr>;
@group(0) @binding({BIND_OUT_B}) var<storage, read_write> OUT_B: array<Fr>;
@group(0) @binding({BIND_OUT_C}) var<storage, read_write> OUT_C: array<Fr>;

// One CSR row of one matrix. `acc` is a function-scope Fr, but every index into it is a
// literal inside fr_add and fr_mul, so it is not the dynamically indexed local array that
// the sweep measured a 3.8x penalty for. `k` indexes storage,
// which is memory either way.
fn gather_row(row_base: u32, nz_base: u32, row: u32) -> Fr {{
    var acc = fr_zero();
    let lo = nz_base + ROW_PTR[row_base + row];
    let hi = nz_base + ROW_PTR[row_base + row + 1u];
    for (var k = lo; k < hi; k = k + 1u) {{
        acc = fr_add(acc, fr_mul(VALUE[k], WITNESS[SIGNAL[k]]));
    }}
    return acc;
}}

@compute @workgroup_size({workgroup})
fn {ENTRY}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = P.row_lo + gid.x;
    // The last workgroup of a dispatch runs past row_hi whenever the row count is not a
    // multiple of {workgroup}, which is the common case: a domain is a power of two but a
    // chunk boundary need not be.
    if (row >= P.row_hi) {{ return; }}

    let a = gather_row(P.row_base_a, P.nz_base_a, row);
    let b = gather_row(P.row_base_b, P.nz_base_b, row);
    OUT_A[row] = a;
    OUT_B[row] = b;
    OUT_C[row] = fr_mul(a, b);
}}
"
    )
    .unwrap();
    s
}
