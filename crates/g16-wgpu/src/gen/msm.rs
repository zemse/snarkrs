//! The MSM digit pipeline in WGSL: the counting sort that makes one thread own one bucket.
//!
//! A port of the first five kernels of `g16-metal/src/shaders/msm.metal`: `zero_u32`,
//! `fr_mont_to_std`, `msm_count`, `msm_scan`, `msm_scatter`. Everything here is integer
//! work on scalars. No curve arithmetic, no `Fq`, no `Fq2`; U9 and U10 add those.
//!
//! # Why a counting sort at all, in one paragraph
//!
//! Pippenger wants `buckets[d] += P`, and there is no atomic on a 128-byte point on any GPU
//! and no CAS on a struct in WGSL, which has `atomic<u32>` and `atomic<i32>` and nothing
//! else. So the write conflict is removed by construction instead: count how many points
//! land in each `(window, bucket)`, prefix-sum the counts into run offsets, scatter each
//! point index into its run. After the scatter **every bucket is written by exactly one
//! thread**, so the accumulation needs no synchronisation of any kind. Order inside a bucket
//! is not preserved and does not matter, because bucket accumulation is commutative. The
//! keys are dense small integers, so this is O(n) and cheaper than sorting the pairs.
//!
//! Both atomics used here are `atomicAdd` on a plain `u32`, which is all WGSL gives and all
//! this needs.
//!
//! # The scalars 0 and 1 never reach a bucket, and that is worth 5.1x
//!
//! [`ENTRY_COUNT`] and [`ENTRY_SCATTER`] test for both before touching a window. A zero
//! costs one 8-word read; a one is routed to `msm_ones_*` at U9 and costs one mixed addition
//! for the whole scalar. This is not a micro-optimisation: the signed recoding sends every
//! scalar equal to 1 to digit +1 of window 0, so bucket `(0, 0)` would collect *every*
//! one-scalar, and since one thread owns one bucket that single thread would serially
//! accumulate 100k points while the rest of the machine idled. Four of the five MSMs take the
//! witness as scalars and a bit-heavy circuit is over 99% zeros and ones.
//!
//! It also means the prover is not constant time with respect to the witness. `security/`
//! and `g16-msm`'s `prescan` docs carry that analysis; it is a deliberate trade, not an
//! oversight.
//!
//! # The recoding, and why the top carry is provably zero
//!
//! Digit by digit the same carry-free signed recoding `g16-msm` uses on the CPU, so a
//! mismatch between the two MSMs can only come from the point arithmetic:
//!
//! ```text
//!   b_w = bits [w*c, w*c + c) of the scalar
//!   d_w = b_w - 2^c * [b_w >= 2^(c-1)]  +  bit (w*c - 1) of the scalar
//! ```
//!
//! Every digit is a pure function of the scalar and the window index, so the windows stay
//! independent and no digit array is ever materialised (arkworks materialises `W` i64 per
//! scalar, which is 136 MB at 2^20 points). `d_w` lands in `[-2^(c-1), 2^(c-1)]`, so bucket
//! index `|d_w| - 1` is in `[0, 2^(c-1) - 1]` and there are `2^(c-1)` buckets.
//!
//! The borrow taken by window `w` is repaid by window `w+1` reading bit `(w+1)*c - 1`. The
//! top window has no `w+1` to repay it, so **the top window must never borrow** or the
//! recoding silently loses `2^(W*c)` and produces a wrong point with nothing in any log.
//! It never does, for BN254: the digits are laid out over `RECODE_BITS = 255` bits, so
//! `W*c >= 255` and the top window's sign bit sits at index `W*c - 1 >= 254`. Every scalar is
//! below `r`, whose top 16 bits are `0x3064 = 12388 < 2^15`, so `r < 2^254` and bits 254 and
//! 255 are zero in every scalar there is. `tests/msm_digits.rs` checks it two ways: directly,
//! that the top window's sign bit is clear, over 10^5 random scalars plus `r - 1` at every
//! `c` in `2..=16`; and by reconstructing the scalar from its digits over a smaller sample,
//! which is what checks the telescoping argument behind the direct test rather than trusting
//! the algebra. The largest top window it has ever seen is 12388, which is `0x3064`, `r`'s own
//! top 16 bits, so the bound is tight and reached.
//!
//! # What is emitted where, and why it is one module and not two
//!
//! Three builders, because the choice between them is a measurement. [`digits_module_at`]
//! holds the four kernels that touch no field arithmetic at all and is 10.5 KiB;
//! [`mont_module_at`] holds `fr_mont_to_std`, which needs the whole 30 KiB `Fr` prelude for
//! one call to `fr_from_mont`; [`fused_module_at`] holds all five behind one copy of the
//! prelude and is what ships.
//!
//! **That is the opposite of what design §4 and `crate::pipelines` assume.** Both commit to
//! many small modules on the strength of a 129 s pipeline creation recorded elsewhere for one
//! monolithic shader. Cold on this M2 Max the fused form is 17% cheaper, about 34 ms, and
//! warm the two are identical. The table, the mechanism and the reason the split is kept
//! selectable are on [`crate::msm::ModuleShape`].
//!
//! # No dynamically indexed local array anywhere, and that is measured here too
//!
//! The limb-width sweep measured a 3.8x penalty for indexing a
//! function-scope `array<u32, N>` by a loop variable, because naga lowers it to an MSL stack
//! array and a dynamically indexed stack array does not stay in registers on Apple silicon.
//! The digit extractor's natural form is `s[bit_off >> 5]`, which is exactly that shape and
//! is on the hot path 20 times per scalar. [`LimbPick`] carries both forms, the table, and
//! the answer, which is that on this hardware they are the same speed and the rule does not
//! reach as far as it was assumed to.

use std::fmt::Write as _;

use g16_gpu_layout::LIMBS;

use crate::gen::field::Variant;

// ---------------------------------------------------------------------------
// Entry point names and bindings
// ---------------------------------------------------------------------------

/// Clears a `u32` range. The counting-sort counters need it because the scratch pool hands
/// back used buffers, unlike the bucket array whose kernel writes rather than accumulates.
pub const ENTRY_ZERO: &str = "zero_u32";
/// Montgomery limbs (`g16_gpu_layout::PackedFr`) to standard limbs (`PackedScalar`).
pub const ENTRY_MONT: &str = "fr_mont_to_std";
/// Stage 1 of the counting sort: how many points land in each `(window, bucket)`.
pub const ENTRY_COUNT: &str = "msm_count";
/// Stage 2: exclusive prefix sum of the counts inside each window, biased by the window's
/// base offset in the entry array. One workgroup per window.
pub const ENTRY_SCAN: &str = "msm_scan";
/// Stage 3: scatter each point index into its bucket's run.
pub const ENTRY_SCATTER: &str = "msm_scatter";

/// Binding numbers, group 0.
///
/// Every resource variable in [`digits_module`] gets its own number even though no entry
/// point uses more than three of them, so the module is legal under the loosest and the
/// strictest reading of WGSL's "two resource variables in one entry point's resource
/// interface must not share a group and binding". Each entry point gets its own pipeline
/// layout declaring only what it reads, which is what keeps the storage-buffer counts at
/// 1, 2, 2 and 3 against the browser floor's 8. A pipeline layout is allowed to declare
/// bindings the shader ignores, so this costs nothing.
pub const BIND_PARAMS: u32 = 0;
/// `zero_u32`: the range to clear.
pub const BIND_ZERO_BUF: u32 = 1;
/// `msm_count` and `msm_scatter`: standard-form scalars.
pub const BIND_SCALARS: u32 = 2;
/// `msm_count`: the counters, as atomics.
pub const BIND_COUNTS_ATOMIC: u32 = 3;
/// `msm_scan`: the same counters, read back as plain `u32`.
pub const BIND_COUNTS_READ: u32 = 4;
/// `msm_scan`: the run offsets it writes.
pub const BIND_CURSOR_WRITE: u32 = 5;
/// `msm_scatter`: the same offsets, bumped atomically.
pub const BIND_CURSOR_ATOMIC: u32 = 6;
/// `msm_scatter`: `(row, point << 1 | sign)` per emitted digit.
pub const BIND_ENTRIES: u32 = 7;
/// `fr_mont_to_std`: the Montgomery source. Numbered past the digit kernels' bindings rather
/// than restarting at 1, so [`fused_module_at`] is a plain concatenation of the two modules
/// with one copy of the parameter struct and no renumbering.
pub const BIND_MONT_SRC: u32 = 8;
/// `fr_mont_to_std`: the standard-form destination.
pub const BIND_MONT_DST: u32 = 9;

/// Storage buffers each entry point's pipeline layout declares. The floor allows 8.
///
/// Asserted against the layouts `crate::msm` actually builds by `tests/msm_digits.rs`, so
/// these cannot drift away from the thing the device enforces.
pub const STORAGE_ZERO: u32 = 1;
pub const STORAGE_MONT: u32 = 2;
pub const STORAGE_COUNT: u32 = 2;
pub const STORAGE_SCAN: u32 = 2;
pub const STORAGE_SCATTER: u32 = 3;

// ---------------------------------------------------------------------------
// Workgroup sizes
// ---------------------------------------------------------------------------

/// Threads per workgroup for the five entry points, emitted as literals so the shader and the
/// host dispatch count cannot disagree.
///
/// **Four of these five are not what design §4 says.** §4 carried `g16-metal`'s threadgroup
/// sizes over on the assumption that MSL's occupancy reasoning transfers; U5 found it does not
/// for `gather_abc` (24% at 2^17), U6 found it does not for the NTT (7.8x), U7 found it does
/// for `h_join`. Measured by `tests/msm_digits.rs::the_digit_workgroup_sizes_are_measured`,
/// 65,536 general scalars at `c = 13`, one kernel's size varied at a time, each cell the
/// median of three, each column the median of four **release** runs, microseconds per
/// dispatch group:
///
/// ```text
/// kernel              16     32     64    128    256    §4   ships
/// zero_u32           8.4    6.7    5.5    5.3    6.4   256     128
/// fr_mont_to_std    35.7   21.3   20.1   19.8   20.5   256     128
/// msm_count         58.9   55.4   57.2   59.7   62.4    64      32
/// msm_scan         263.9  157.8  100.8   59.9   36.8   256     256
/// msm_scatter      103.9  115.4  110.7  106.1  113.4    64     128
/// ```
///
/// Run to run these are stable to about 1%, except `zero_u32`, whose whole row spans 3
/// microseconds and is mostly measurement noise: it clears 320 KB in 5.3, which is 60 GB/s,
/// and nothing in that row is worth an argument. 128 has the lowest median and that is the
/// only reason it is there.
///
/// `msm_scan` is the one §4 got right, and it got it right by the largest margin in the
/// table: 256 is 7x faster than 16 and 1.6x faster than 128. The Hillis-Steele scan does
/// `log2(threads)` passes over a fixed number of buckets, so a wider workgroup is strictly
/// less work as well as more parallelism, which is the opposite of the NTT tile, where a wide
/// workgroup idles lanes.
///
/// `msm_count` and `msm_scatter` are the two that dominate, and **they disagree with each
/// other even though they extract the same digits**: count wants 32 and scatter wants 128,
/// and each is 8% worse at the other's size. The only difference between them is that the
/// scatter also does an `atomicAdd` on a cursor and an 8-byte write per entry, so the right
/// workgroup size here is a property of the memory traffic rather than of the arithmetic.
/// That is worth knowing before U9 sweeps the point stages: it means one answer will not do
/// for all of them.
///
/// Metal's 64 is 3% off on count and 4% off on scatter, which makes this the first table in
/// this crate where the inherited constant is nearly right.
///
/// # The one place a measurement is not being followed, and what that costs
///
/// **`msm_scatter` is fastest at 16 threads, by 2.1% over the 128 that ships, in ten runs out
/// of ten.** 16 half-fills this hardware's 32-wide SIMD group, and the plausible mechanism is
/// that halving the live lanes per group halves the contention when several of them bump the
/// same cursor. Note that 32, which exactly fills a group, is the *worst* cell in the row,
/// which fits that story.
///
/// It is not shipped, and the reasons are stated rather than buried. The gain is 2.2
/// microseconds on a kernel that runs three times per proof, so about 7 microseconds against
/// a proof this backend is trying to bring under 200 milliseconds: 0.004%. And a workgroup
/// that deliberately half-fills a SIMD group is a bet on one SIMD width, on a backend whose
/// whole reason to exist is the three platforms nobody here has measured; 128 is within 4% of
/// the best cell at every width in this table, which 16 is not. If U14 measures it again
/// under Tint and 16 still wins, take it.
///
/// Release only. U7 measured that this differencing harness inverts its own answer in a debug
/// build, because it cancels the fixed cost of a submit but not the per-encode host cost,
/// which scales with the dispatch count exactly as the kernel does.
///
/// naga to MSL. Chrome compiles through Tint, so U14 re-runs this rather than trusting it,
/// which is why every one of these is a parameter of [`digits_module_at`], [`mont_module_at`]
/// and [`fused_module_at`] and not a literal in the shader text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workgroups {
    pub zero: u32,
    pub mont: u32,
    pub count: u32,
    pub scan: u32,
    pub scatter: u32,
}

impl Default for Workgroups {
    fn default() -> Self {
        Self {
            zero: 128,
            mont: 128,
            count: 32,
            scan: 256,
            scatter: 128,
        }
    }
}

impl Workgroups {
    /// Every size the same, for the sweep in `tests/msm_digits.rs`.
    pub fn uniform(n: u32) -> Self {
        Self {
            zero: n,
            mont: n,
            count: n,
            scan: n,
            scatter: n,
        }
    }
}

// ---------------------------------------------------------------------------
// How a scalar limb is selected
// ---------------------------------------------------------------------------

/// How `sc_pick` reaches limb `i` of a scalar when `i` is a runtime value.
///
/// The digit extractor reads bits `[w*c, w*c+c)`, so the limb it wants is
/// `(w*c) >> 5`, which depends on the window loop counter. There is no way to make that a
/// literal without generating one entry point per `c`, and `c` is chosen at run time from
/// the general-scalar count.
///
/// # Measured, and the design's rule does not reach this far
///
/// `tests/msm_digits.rs::the_limb_pick_strategy_is_measured`, 65,536 general scalars at
/// `c = 13`, each cell the median of three, each column the median of four release runs,
/// microseconds:
///
/// ```text
/// kernel          Index   Select   Select/Index
/// msm_count        55.8     55.8          1.000
/// msm_scatter     106.4    106.6          1.002
/// ```
///
/// **They are the same, and that is a result rather than a tie to shrug at.** The 3.8x cliff
/// the sweep measured is about a **mutable** function-scope
/// `var` array that a loop writes and reads back across iterations; `s` here is a `let`-bound
/// value, read through a dynamic index and never written, and naga hands that to MSL as a
/// value its own compiler can keep in registers and select over. So "unroll every limb loop"
/// is a rule about the CIOS accumulator, and it does not generalise to every dynamic index.
/// Nobody in this crate had checked that before, the check is cheap, and it matters because
/// the select chain is 7 extra instructions per limb read and there are four limb reads per
/// window per scalar.
///
/// The select chain ships anyway. It costs nothing here, it is the form that is guaranteed
/// rather than the form that happened to be optimised, and Tint is a different compiler.
/// [`Index`](Self::Index) stays so U14 can re-run the comparison in Chrome in one line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LimbPick {
    /// A straight-line 8-way `select` chain over literal indices. Out of range reads as 0,
    /// which is what makes the highest window cheap instead of a special case.
    #[default]
    Select,
    /// `s[i]` with a bounds test. One instruction in the source and an indexed load in the
    /// output.
    Index,
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// The four scalar-only kernels at the default shape.
pub fn digits_module() -> String {
    digits_module_at(Workgroups::default(), LimbPick::default())
}

/// Same, at chosen workgroup sizes and limb-pick strategy.
///
/// # Panics
///
/// On a workgroup size outside `1..=256`, which is `maxComputeInvocationsPerWorkgroup` at the
/// floor. This adapter allows 1024 and no browser does, so the check is against the floor and
/// not against the device.
pub fn digits_module_at(wg: Workgroups, pick: LimbPick) -> String {
    check_sizes(&[
        ("zero", wg.zero),
        ("count", wg.count),
        ("scan", wg.scan),
        ("scatter", wg.scatter),
    ]);
    let mut s = header(pick, wg);
    s.push_str(&params_struct());
    s.push_str(&digit_bindings());
    s.push_str(&scalar_helpers(pick));
    s.push_str(&entry_zero(wg.zero));
    s.push_str(&entry_count(wg.count, pick));
    s.push_str(&entry_scan(wg.scan));
    s.push_str(&entry_scatter(wg.scatter, pick));
    s
}

/// `fr_mont_to_std` plus the `Fr` prelude it needs, at the default workgroup size.
pub fn mont_module(v: Variant) -> String {
    mont_module_at(v, Workgroups::default().mont)
}

/// Same, at a chosen workgroup size.
///
/// # Panics
///
/// On [`Variant::NoCarry13x20`], for the same reason `gen::gather`, `gen::ntt` and
/// `gen::pointwise` do: its `R = 2^260` wire format is not `g16_gpu_layout::PackedFr` and no
/// host packer produces it, so the kernel would read 20-limb elements out of a buffer written
/// as 8-limb ones and produce plausible garbage.
pub fn mont_module_at(v: Variant, workgroup: u32) -> String {
    check_variant(v);
    check_sizes(&[("mont", workgroup)]);
    let mut s = format!(
        "// Generated by g16-wgpu::gen::msm, variant {v:?}, workgroup {workgroup}.\n\
         // Do not edit by hand.\n"
    );
    s.push_str(&fr_prelude(v));
    s.push_str(&params_struct());
    s.push_str(&mont_bindings());
    s.push_str(&entry_mont(workgroup));
    s
}

/// All five entry points behind one copy of the `Fr` prelude.
///
/// The alternative to shipping two modules, kept so the choice is a measurement rather than
/// an assumption. `tests/msm_digits.rs::the_digit_modules_are_split_for_a_measured_reason`
/// builds this and the two-module form and prints both costs; the answer and the numbers are
/// on [`crate::msm::MsmDigits`].
pub fn fused_module_at(v: Variant, wg: Workgroups, pick: LimbPick) -> String {
    check_variant(v);
    check_sizes(&[
        ("zero", wg.zero),
        ("mont", wg.mont),
        ("count", wg.count),
        ("scan", wg.scan),
        ("scatter", wg.scatter),
    ]);
    let mut s = header(pick, wg);
    s.push_str(&fr_prelude(v));
    s.push_str(&params_struct());
    s.push_str(&digit_bindings());
    s.push_str(&mont_bindings());
    s.push_str(&scalar_helpers(pick));
    s.push_str(&entry_zero(wg.zero));
    s.push_str(&entry_mont(wg.mont));
    s.push_str(&entry_count(wg.count, pick));
    s.push_str(&entry_scan(wg.scan));
    s.push_str(&entry_scatter(wg.scatter, pick));
    s
}

fn header(pick: LimbPick, wg: Workgroups) -> String {
    format!(
        "// Generated by g16-wgpu::gen::msm, {pick:?} limb pick, workgroups zero {} mont {} \
         count {} scan {} scatter {}.\n// Do not edit by hand.\n",
        wg.zero, wg.mont, wg.count, wg.scan, wg.scatter
    )
}

fn check_sizes(sizes: &[(&str, u32)]) {
    for &(name, n) in sizes {
        assert!(
            n > 0 && n <= 256,
            "{name} workgroup size {n} is outside 1..=256, the floor's \
             maxComputeInvocationsPerWorkgroup"
        );
    }
}

fn check_variant(v: Variant) {
    assert_eq!(
        v.limbs(),
        LIMBS,
        "fr_mont_to_std reads g16_gpu_layout::PackedFr and writes PackedScalar, both {LIMBS} \
         limbs; variant {v:?} is {} and has no host packer",
        v.limbs()
    );
}

/// `mul64` plus the `Fr` half of the field prelude. `Fq` and `Fq2` are about 55 KiB of
/// straight-line multiply that nothing in this file calls.
fn fr_prelude(v: Variant) -> String {
    let mut s = String::new();
    if v.needs_mul64() {
        s.push_str(crate::gen::field::MUL64);
    }
    s.push_str(&crate::gen::field::FR.ops(v));
    s
}

/// The `MsmParams` fields, shared verbatim by both modules so the two declarations of the
/// struct cannot drift.
///
/// Eight `u32`, 32 bytes, already a multiple of the 16 WGSL rounds every uniform struct up
/// to, so there is no invisible tail padding for `crate::msm::MsmParams` to disagree with.
const PARAM_FIELDS: &str = "\
    // Elements this kernel's domain holds: scalars for count and scatter, words for\n\
    // zero_u32, field elements for fr_mont_to_std.\n\
    n: u32,\n\
    // Window width in bits, 2..=16.\n\
    c: u32,\n\
    // ceil(255 / c).\n\
    n_windows: u32,\n\
    // 2^(c-1).\n\
    n_buckets: u32,\n\
    // Entries reserved per window in the entry array. Must be at least the number of\n\
    // scalars in range that are neither 0 nor 1, or the scatter runs off the end of its\n\
    // window and WGSL drops the write in silence.\n\
    cap: u32,\n\
    // Element offset into the scalar buffer, so the L MSM's private suffix shares the\n\
    // witness buffer with A and B instead of uploading a second copy.\n\
    scalar_off: u32,\n\
    // First element of this dispatch. maxComputeWorkgroupsPerDimension is 65535 at every\n\
    // tier of every browser, so a domain past workgroup * 65535 needs more than one\n\
    // dispatch and each has to know where it starts.\n\
    lo: u32,\n\
    pad0: u32,\n";

/// The parameter struct and its uniform binding. One copy per module, whichever module.
fn params_struct() -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
struct MsmParams {{
{PARAM_FIELDS}}};

@group(0) @binding({BIND_PARAMS}) var<uniform> P: MsmParams;
"
    );
    s
}

/// The `Scalar` alias and the seven storage declarations the four digit kernels use between
/// them. No entry point touches more than three.
fn digit_bindings() -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Standard form, NOT Montgomery: this is g16_gpu_layout::PackedScalar, the integer in
// [0, r), which is what a window digit is a digit of. Deliberately not named `Fr`, because
// `Fr` in every other module of this crate is the Montgomery representative and reading one
// as the other is a proof wrong by a factor of R.
alias Scalar = array<u32, {LIMBS}>;

@group(0) @binding({BIND_ZERO_BUF}) var<storage, read_write> ZBUF: array<u32>;
@group(0) @binding({BIND_SCALARS}) var<storage, read> SCALARS: array<Scalar>;
@group(0) @binding({BIND_COUNTS_ATOMIC}) var<storage, read_write> COUNTS_A: array<atomic<u32>>;
@group(0) @binding({BIND_COUNTS_READ}) var<storage, read> COUNTS_R: array<u32>;
@group(0) @binding({BIND_CURSOR_WRITE}) var<storage, read_write> CURSOR_W: array<u32>;
@group(0) @binding({BIND_CURSOR_ATOMIC}) var<storage, read_write> CURSOR_A: array<atomic<u32>>;
@group(0) @binding({BIND_ENTRIES}) var<storage, read_write> ENTRIES: array<vec2<u32>>;
"
    );
    s
}

/// `fr_mont_to_std`'s two, which are `Fr` and so need the prelude in scope.
fn mont_bindings() -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
@group(0) @binding({BIND_MONT_SRC}) var<storage, read> MSRC: array<Fr>;
@group(0) @binding({BIND_MONT_DST}) var<storage, read_write> MDST: array<Fr>;
"
    );
    s
}

fn entry_mont(workgroup: u32) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
// Montgomery limbs to standard limbs. The NTT and stage 4 leave H in Montgomery form on the
// device; Pippenger slices the *integer* into window digits, and a window digit of a
// Montgomery representative is a digit of `a*R mod r`, a different number. Getting this
// backwards yields a proof wrong by a factor of R that fails verification with nothing else
// to go on.
@compute @workgroup_size({workgroup})
fn {ENTRY_MONT}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    if (i >= P.n) {{ return; }}
    // fr_from_mont is a Montgomery multiply by the integer 1, so this is one multiply and no
    // branch.
    MDST[i] = fr_from_mont(MSRC[i]);
}}
"
    );
    s
}

/// How the digit helpers take a scalar, as `(declaration, call site)`.
fn pick_args(pick: LimbPick) -> (String, String) {
    match pick {
        LimbPick::Select => {
            let decl = (0..LIMBS)
                .map(|i| format!("s{i}: u32"))
                .collect::<Vec<_>>()
                .join(", ");
            let call = (0..LIMBS)
                .map(|i| format!("s{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            (decl, call)
        }
        LimbPick::Index => ("s: Scalar".to_string(), "s".to_string()),
    }
}

/// `sc_pick`, `sc_bits`, `sc_digit`: the whole recoding, with no state and no allocation.
fn scalar_helpers(pick: LimbPick) -> String {
    let (decl, call) = pick_args(pick);
    let mut s = String::new();

    let _ = write!(
        s,
        "
// ---------------------------------------------------------------------------
// The signed recoding. Identical digit for digit to g16_msm::signed_digit, so a mismatch
// between the CPU and GPU MSMs can only come from the point arithmetic.
// ---------------------------------------------------------------------------

// Limb i, reading past the top of the scalar as zero. That out-of-range zero is what makes
// the highest window cheap instead of a special case, and it is also what lets the carry
// read at window 0 (bit offset 0 - 1, which wraps to 0xffffffff) fall out as zero with no
// branch.
"
    );
    match pick {
        LimbPick::Select => {
            let _ = writeln!(s, "fn sc_pick({decl}, i: u32) -> u32 {{");
            let _ = writeln!(s, "    var r = select(0u, s0, i == 0u);");
            for j in 1..LIMBS {
                let _ = writeln!(s, "    r = select(r, s{j}, i == {j}u);");
            }
            let _ = writeln!(s, "    return r;\n}}");
        }
        LimbPick::Index => {
            let _ = writeln!(s, "fn sc_pick({decl}, i: u32) -> u32 {{");
            let _ = writeln!(s, "    if (i >= {LIMBS}u) {{ return 0u; }}");
            let _ = writeln!(s, "    return s[i];\n}}");
        }
    }

    let _ = write!(
        s,
        "
// `width` bits starting at bit `off`, little endian across limbs. `width` is at most 16
// here, which is MAX_WINDOW.
fn sc_bits({decl}, off: u32, width: u32) -> u32 {{
    let idx = off >> 5u;
    let sh = off & 31u;
    let w0 = sc_pick({call}, idx);
    let w1 = sc_pick({call}, idx + 1u);
    // `w1 << (32u - sh)` is the obvious spelling and is wrong at sh == 0: WGSL leaves a
    // shift by the full word width indeterminate, MSL and SPIR-V both call it undefined,
    // and sh == 0 is every window when c divides 32. Shifting by `31u - sh` and then by one
    // more is defined for every sh in 0..=31 and gives 0 at sh == 0, which is exactly the
    // wanted \"no contribution from the next limb\".
    let buf = (w0 >> sh) | ((w1 << (31u - sh)) << 1u);
    return buf & ((1u << width) - 1u);
}}

// Signed digit `w`, as (magnitude, sign). Magnitude 0 means the digit is zero and no bucket
// is touched, so the caller never handles a negative index.
fn sc_digit({decl}, w: u32, c: u32) -> vec2<u32> {{
    let off = w * c;
    let b = sc_bits({call}, off, c);
    // The borrow window w-1 took, which is just bit (w*c - 1) of the scalar. At w == 0 the
    // offset wraps and sc_pick reads out of range, which is zero, which is the right answer.
    let carry = sc_bits({call}, off - 1u, 1u);
    // Borrow 2^c when the raw window is in the top half; window w+1 pays it back through the
    // carry above. The top window never takes this branch for BN254; see the module docs.
    let top = (b >> (c - 1u)) & 1u;
    let m_neg = (1u << c) - b - carry;
    let m_pos = b + carry;
    let mag = select(m_pos, m_neg, top != 0u);
    // A borrow can land on magnitude zero, when b = 2^c - 1 and the carry is 1, and the MSL
    // original spends a test turning that digit's sign back off. That test is dead in both:
    // every caller skips a zero-magnitude digit before it reads the sign, so \"minus zero\"
    // is never observable. Removing it here rather than porting it, because a branch no test
    // can reach is a branch a mutation test cannot kill, and this one was found by exactly
    // that.
    let neg = select(0u, 1u, top != 0u);
    return vec2<u32>(mag, neg);
}}
"
    );
    s
}

/// The eight-limb load plus the zero and one tests, shared by count and scatter.
fn classify(pick: LimbPick) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "    let s = SCALARS[P.scalar_off + i];");
    // In Select mode the eight limbs are pulled out into named locals once and every later
    // reference, the classification included, goes through those rather than through `s`, so
    // the only indexed read of the array is this one and it is at literal indices.
    let name: Vec<String> = match pick {
        LimbPick::Select => {
            for j in 0..LIMBS {
                let _ = writeln!(s, "    let s{j} = s[{j}];");
            }
            (0..LIMBS).map(|j| format!("s{j}")).collect()
        }
        LimbPick::Index => (0..LIMBS).map(|j| format!("s[{j}]")).collect(),
    };
    let hi = name[1..].join(" | ");
    let zero = &name[0];
    let _ = write!(
        s,
        "    // The two cheap classes leave here, and the two tests are not the same kind of
    // thing, which is worth knowing before anyone deletes one.
    //
    // The ZERO test is a pure deoptimisation to remove: every digit of the scalar 0 is 0, so
    // a zero scalar emits no entry whether or not this line is here. Deleting it was a
    // mutation the whole acceptance suite passed, and correctly so. It stays because it
    // turns 20 window extractions into one 8-word read, and a bit-heavy witness is mostly
    // zeros.
    //
    // The ONE test is correctness, once U9's msm_ones_* exists. The signed recoding sends
    // every scalar equal to 1 to digit +1 of window 0, so without this line bucket (0, 0)
    // collects one entry per one-scalar and every one of those points is then added a
    // second time by msm_ones_*. It is also what stops that single bucket, owned by a single
    // thread, from serially accumulating 100k points while the rest of the machine idles.
    let hi = {hi};
    if ((hi | {zero}) == 0u) {{ return; }}
    if (hi == 0u && {zero} == 1u) {{ return; }}
"
    );
    s
}

fn entry_zero(workgroup: u32) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
@compute @workgroup_size({workgroup})
fn {ENTRY_ZERO}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    // Guarded rather than left to WebGPU dropping the out-of-range write, because a pooled
    // counts buffer is larger than the rows this plan uses and the slack belongs to whoever
    // gets it next. tests/msm_digits.rs binds a word of slack and checks it survives.
    if (i >= P.n) {{ return; }}
    ZBUF[i] = 0u;
}}
"
    );
    s
}

fn entry_count(workgroup: u32, pick: LimbPick) -> String {
    let (_, call) = pick_args(pick);
    let mut s = String::new();
    let _ = write!(
        s,
        "
@compute @workgroup_size({workgroup})
fn {ENTRY_COUNT}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    if (i >= P.n) {{ return; }}
{}\
\x20   for (var w = 0u; w < P.n_windows; w = w + 1u) {{
        let d = sc_digit({call}, w, P.c);
        if (d.x == 0u) {{ continue; }}
        atomicAdd(&COUNTS_A[w * P.n_buckets + (d.x - 1u)], 1u);
    }}
}}
",
        classify(pick)
    );
    s
}

fn entry_scan(workgroup: u32) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "
// One workgroup per window. The scan is over 2^(c-1) counters, which is 4096 at c = 13, so
// it walks the window in chunks of {workgroup} and carries a running total across them.
var<workgroup> SCAN: array<u32, {workgroup}>;

@compute @workgroup_size({workgroup})
fn {ENTRY_SCAN}(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) tid: u32) {{
    let w = wid.x;
    // Window w's entries live in [w*cap, (w+1)*cap), so the exclusive prefix sum is biased
    // by w*cap and the scatter's atomic bump needs no second offset.
    var running = w * P.cap;
    // Uniform, because P is a uniform buffer, which is what keeps the barriers below inside
    // uniform control flow. WGSL makes that a hard error where MSL merely makes it undefined.
    let chunks = (P.n_buckets + {workgroup}u - 1u) / {workgroup}u;
    for (var ch = 0u; ch < chunks; ch = ch + 1u) {{
        let idx = ch * {workgroup}u + tid;
        var v = 0u;
        if (idx < P.n_buckets) {{ v = COUNTS_R[w * P.n_buckets + idx]; }}
        SCAN[tid] = v;
        workgroupBarrier();
        // Hillis-Steele inclusive scan. The read is separated from the write by a barrier on
        // both sides, which is what makes the in-place update safe. Literal bound, so the
        // barriers sit in uniform control flow.
        for (var d = 1u; d < {workgroup}u; d = d << 1u) {{
            var x = 0u;
            if (tid >= d) {{ x = SCAN[tid - d]; }}
            workgroupBarrier();
            SCAN[tid] = SCAN[tid] + x;
            workgroupBarrier();
        }}
        if (idx < P.n_buckets) {{
            // Inclusive minus own value is exclusive.
            CURSOR_W[w * P.n_buckets + idx] = running + SCAN[tid] - v;
        }}
        // The last lane's inclusive total is the whole chunk, including the lanes that were
        // out of range and contributed zero. Read before the barrier that lets the next
        // chunk overwrite SCAN.
        let total = SCAN[{last}u];
        workgroupBarrier();
        running = running + total;
    }}
}}
",
        last = workgroup - 1
    );
    s
}

fn entry_scatter(workgroup: u32, pick: LimbPick) -> String {
    let (_, call) = pick_args(pick);
    let mut s = String::new();
    let _ = write!(
        s,
        "
// After this kernel CURSOR holds each run's END, and the run's start is `end - count`. The
// bump is a relaxed atomicAdd, so points land inside their run in an arbitrary order, which
// is fine because bucket accumulation is commutative.
//
// An entry is (row, point << 1 | sign) with row = w * n_buckets + bucket. The row is stored
// rather than recomputed because U9's segmented accumulation walks a fixed-length slice of
// this array and has to discover where one bucket's run ends, which it cannot do from the
// point index alone.
//
// `slot` is not bounds checked, and it is a proven bound rather than an omission: window w's
// entries are written into [w*cap, w*cap + cap) and at most one entry per general scalar per
// window is emitted, so `cap >= general scalars in range` is enough. crate::msm::DigitPlan
// derives cap from the classification the caller already did and never from a guess.
@compute @workgroup_size({workgroup})
fn {ENTRY_SCATTER}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = P.lo + gid.x;
    if (i >= P.n) {{ return; }}
{}\
\x20   for (var w = 0u; w < P.n_windows; w = w + 1u) {{
        let d = sc_digit({call}, w, P.c);
        if (d.x == 0u) {{ continue; }}
        let row = w * P.n_buckets + (d.x - 1u);
        let slot = atomicAdd(&CURSOR_A[row], 1u);
        // i, not scalar_off + i: the point stages index the base vector as
        // bases[base_off + (e >> 1)], so the entry carries the position within this MSM.
        ENTRIES[slot] = vec2<u32>(row, (i << 1u) | d.y);
    }}
}}
",
        classify(pick)
    );
    s
}
