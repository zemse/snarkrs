//! Stages 1 to 3 in WGSL: the six NTTs, the bit-reverse, the `1/n` normalisation and the
//! coset shift, plus the stage 4 epilogue.
//!
//! The `*_join_*` entry points are generated, tested and **not what the prover dispatches**.
//! U7 measured the fusion against the standalone `h_join` kernel and it lost by 5% to 9% on
//! every artifact; the table and the mechanism are on [`crate::stages::Stage4`]. They stay
//! because the ranking is one machine's naga-to-MSL measurement and U14 re-runs it in Chrome.
//!
//! The algorithm is `g16-metal/src/shaders/ntt.metal`'s, which is in turn
//! `g16_ntt::CpuNtt`'s: decimation in time, bit-reverse then `log n` passes with
//! `half = 1, 2, 4, ...`, butterfly `j` of a block reading `twiddles[j * n / (2*half)]` out
//! of a natural-order table. Same order, same table, so the three backends agree bit for bit
//! and not merely up to a permutation. What follows is only about the four places WGSL
//! forces a different *shape* from MSL.
//!
//! # 1. One entry point per mode, because the Metal head needs eight storage buffers
//!
//! `ntt.metal` has one head and one tail kernel taking `store_mode` as an argument. Metal's
//! validation layer objects to a null binding for a declared buffer whether or not the
//! shader reads it on this path, so `g16-metal/src/stages.rs:672-703` binds all eight of the
//! head's buffers on every dispatch, including the four the plain path never touches. Eight
//! is exactly `maxStorageBuffersPerShaderStage` at the browser floor, with nothing left, and
//! `crate::device` measured that even the `Raised` profile on this M2 Max only reports 9
//! under strict WebGPU compliance.
//!
//! Generating four entry points instead removes the dead bindings and the dispatch-uniform
//! branch together, and costs nothing because the source is emitted anyway. Measured storage
//! buffers per pipeline layout, counted by [`storage_buffers`] from the same table
//! [`crate::ntt`] builds the layouts from:
//!
//! | entry point | storage buffers | design §4 claimed |
//! |---|---|---|
//! | `ntt_head_plain_k{K}` | 4 | 4 |
//! | `ntt_head_join_k{K}` | **7** | 6 |
//! | `ntt_tail_plain_k{K}` | 2 | 2 |
//! | `ntt_tail_join_k{K}` | 6 | 6 |
//!
//! **The design is wrong about `ntt_head_join`, by one.** It counted `src`, `join_a`,
//! `join_b`, `h_mont`, `h_std` and the twiddles and forgot the coset-power table. The head
//! is the *only* batch of a transform whenever `log_n <= K`, so on any domain of 2^9 or
//! smaller the joined batch is also the batch carrying the stage 2 coset shift and it has to
//! read `ptab`. Seven is still one under the floor, so nothing has to change, but a design
//! that says six leaves no margin for the reader who adds an eighth.
//!
//! # 2. The tile size is a compile-time constant, so there is one pipeline per distinct `k`
//!
//! MSL takes the threadgroup array as an entry-point parameter sized by the host
//! (`threadgroup Fr* sh [[threadgroup(0)]]` plus `setThreadgroupMemoryLength`). WGSL has no
//! equivalent and no runtime-sized `var<workgroup>`, so the slice length is baked into the
//! source and the generator emits one set of entry points per distinct `k` that
//! [`crate::ntt::split_passes`] produces. That is at most two, because `split_passes` splits
//! evenly and so only ever yields `base` and `base + 1`.
//!
//! Under the floor, `maxComputeWorkgroupStorageSize` is 16384 and an `Fr` is 32 bytes, so
//! `K <= 9` and a 2^18 domain splits 9 + 9: the same two dispatches per transform Metal
//! uses on this hardware. Under a 20x13 field an `Fr` would be 80 bytes, only 204 elements
//! would fit, and 2^18 would need three dispatches per transform. One more point for 8x32.
//!
//! # 3. Uniformity is a hard error in WGSL, not undefined behaviour
//!
//! MSL merely produces undefined behaviour when a `threadgroup_barrier` is reached by some
//! lanes and not others. WGSL rejects the shader outright. Two consequences here:
//!
//! * Every loop that contains a `workgroupBarrier()` has a **literal** bound. The pass loop
//!   runs `t = 0 .. K` with `K` written into the source, not read from a uniform.
//! * The strided worker loops Metal writes as `for (m = tid; m < blk; m += tgsz)` become a
//!   literal-bound repeat loop with the range check as an `if` *inside* it, so the barrier
//!   that follows sits at loop-body level and never inside a divergent branch. When the
//!   workgroup size divides the slice the check is not emitted at all.
//!
//! # 4. No user function takes a `ptr<workgroup, ...>`
//!
//! `unrestricted_pointer_parameters` ships in Chrome and Safari and is unimplemented in
//! naga, so `g16_ntt_batch` cannot be a shared function the way it is in MSL. The batch body
//! is emitted inline into each of the four entry points instead. It is about twenty lines;
//! the expensive code is behind `fr_mul` and `fr_add`, which are ordinary functions and are
//! emitted once.
//!
//! # What is deliberately not done
//!
//! No twiddle caching in workgroup memory. At K = 9 the slice is already 16 KiB of the
//! floor's 16 KiB budget, and the twiddles one batch touches are another 2^(K-1) values.
//! Same conclusion `ntt.metal` reached with twice the budget.

use std::fmt::Write as _;

use g16_gpu_layout::LIMBS;

use crate::gen::field::Variant;

/// Threads per workgroup for an entry point at tile size `k`: `2^(k-1)` clamped to
/// `32..=128`.
///
/// **Design §4 says a flat 256 for these kernels, and 256 is the worst of the four sizes
/// swept, at every domain.** 256 is `g16-metal`'s `PREFERRED_THREADS`, inherited on the
/// assumption that MSL's occupancy argument transfers. U5 already found that inheritance cost
/// stage 0 up to 24%; here it costs up to 7.8x. Microseconds for one
/// `iNTT -> coset -> NTT` chain at the shipped tile, submit overhead differenced out,
/// `tests/ntt.rs::the_ntt_shape_is_measured_and_not_inherited`:
///
/// ```text
/// domain   k    wg=32   wg=64  wg=128  wg=256    this rule
/// 2^12     6      485     447     454     753          481
/// 2^14     7     1104     561     881    1661          560
/// 2^16     8     2432    2328    1837    3567         1834
/// 2^18     6     5533    8390   15957   43167         5550
/// ```
///
/// Two effects, and both are visible in the table rather than argued from a manual.
///
/// **A batch of `k` passes has only `2^(k-1)` butterflies per slice, so a workgroup wider than
/// that idles the surplus lanes for the whole kernel.** At `k = 6` and 256 threads, 224 of
/// 256 lanes have nothing to do in every pass, and the 2^18 row is 7.8x. The `k = 7` and
/// `k = 8` rows both put their minimum exactly at `2^(k-1)`, which is the rule.
///
/// **Below 32 there is nothing left to win,** because 32 is the SIMD group width on this
/// hardware and a narrower workgroup wastes lanes inside the group instead of outside it. So
/// the clamp floors at 32 rather than following `2^(k-1)` down to 8, and `k <= 6` all get 32.
///
/// Worst regret against the best column is 7.6%, at 2^12, which is 0.03 ms. The upper clamp
/// at 128 never binds while [`PREFERRED_FUSED`] is 8; it is there so a future `Raised` tile
/// cannot silently ask for a 512-thread workgroup that no browser grants.
///
/// Measured through naga to MSL on an M2 Max. Chrome compiles through Tint, so U14 re-runs
/// this, which is why the size stays a parameter of [`ntt_module_at`].
pub fn workgroup_for(k: u32) -> u32 {
    (1u32 << k.saturating_sub(1)).clamp(32, 128)
}

/// Largest number of passes to fuse into one dispatch, **8 and not the 9 the floor allows**.
///
/// Design §4 says "under the Floor, 16384 / 32 = 512 elements, so K <= 9 and a 2^18 domain
/// splits 9 + 9, the same two dispatches per transform as Metal". Both halves of that are
/// true and the conclusion costs 5x. Microseconds for one `iNTT -> coset -> NTT` chain, each
/// cell at [`workgroup_for`]'s size for its own tile, submit overhead differenced out,
/// `tests/ntt.rs::the_ntt_shape_is_measured_and_not_inherited` on an M2 Max:
///
/// ```text
/// domain      mf 4      mf 5      mf 6      mf 7      mf 8      mf 9
/// 2^12    370 (k4)  382 (k4)  490 (k6)  489 (k6)  482 (k6)  488 (k6)
/// 2^14   1418 (k4)  922 (k5)  923 (k5)  564 (k7)  558 (k7)  565 (k7)
/// 2^16   3995 (k4) 3995 (k4) 1996 (k6) 1997 (k6) 1836 (k8) 1830 (k8)
/// 2^18  18623 (k4)10864 (k5) 5558 (k6) 5537 (k6) 5548 (k6)28178 (k9)
/// ```
///
/// At 2^18 the design's 9 + 9 costs **28.2 ms** and 6 + 6 + 6 costs **5.5 ms**, for one extra
/// dispatch per transform. Three extra dispatches over the six transforms of a proof, at the
/// 2 to 3 microseconds an extra dispatch in an open encoder costs, buy back 68 ms.
///
/// The mechanism is not proven here and will not be asserted as if it were. A `k = 9` tile is
/// `512 * 32 = 16384` bytes, the floor's *entire* workgroup storage budget, so at most one
/// such workgroup is resident per core wherever the hardware budget is under 32 KiB, and a
/// memory-bound kernel with one resident group has nothing to hide latency behind. That is
/// consistent with every row above and with `k = 8` (8192 bytes) being the best cell at 2^16.
/// It is also one data point, because `split_passes` cannot produce a 9 at any smaller domain
/// and this repo has no artifact past 2^18. The number reproduces across runs to under 8%;
/// the explanation is a hypothesis.
///
/// So the cap is 8, and it is optimal or within 0.3% at 2^14, 2^16 and 2^18. It is 30% off at
/// 2^12, where the best is `mf = 4` and the gap is 0.11 ms per chain, 0.34 ms over the three
/// vectors of a proof. A per-domain table would fit that back, and four artifacts across six
/// configurations is not enough evidence to write one: the best `k` goes 4, 7, 8, 6 as the
/// domain grows, which no rule this data can distinguish from noise reproduces.
pub const PREFERRED_FUSED: u32 = 8;

/// Largest `k` any device can support for BN254 `Fr`, independent of the memory budget and
/// of what is fast.
///
/// 2^10 elements at 32 bytes is 32768 bytes, which is the largest workgroup allocation this
/// M2 Max permits and twice the floor's, so this is a statement about the field rather than
/// about the hardware. The memory ceiling is recomputed from the granted limits in
/// [`crate::ntt::Ntt::max_fused`] and is 9 at the floor; what actually ships is
/// [`PREFERRED_FUSED`], which is 8 and is a measurement rather than a limit.
pub const MAX_FUSED_PASSES: u32 = 10;

/// `scale_mode`: the head multiplies nothing into the loaded element.
pub const SCALE_NONE: u32 = 0;
/// `scale_mode`: multiply by `kscale`, which is the iNTT's `1/n`.
pub const SCALE_CONST: u32 = 1;
/// `scale_mode`: multiply by `PTAB[s]`, which is the stage 2 coset shift `shift^s`.
pub const SCALE_TABLE: u32 = 2;

// Binding numbers, in group 0, shared by every entry point in the module. Each entry point's
// pipeline layout declares only the subset it uses, which is the whole point of splitting the
// kernels; see the module docs. Held here so `crate::ntt` builds the layouts, the bind groups
// and the shader from one list.
pub const BIND_PARAMS: u32 = 0;
pub const BIND_SRC: u32 = 1;
pub const BIND_DST: u32 = 2;
pub const BIND_TW: u32 = 3;
pub const BIND_PTAB: u32 = 4;
pub const BIND_JOIN_A: u32 = 5;
pub const BIND_JOIN_B: u32 = 6;
pub const BIND_H_MONT: u32 = 7;
pub const BIND_H_STD: u32 = 8;

/// Which of the four entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mode {
    /// First batch of a transform: out of place, bit-reverse and load scale fused in.
    pub head: bool,
    /// Last batch of C's forward transform: writes `H = a*b - x` instead of `x`.
    pub join: bool,
}

impl Mode {
    pub const HEAD_PLAIN: Self = Self {
        head: true,
        join: false,
    };
    pub const HEAD_JOIN: Self = Self {
        head: true,
        join: true,
    };
    pub const TAIL_PLAIN: Self = Self {
        head: false,
        join: false,
    };
    pub const TAIL_JOIN: Self = Self {
        head: false,
        join: true,
    };
    /// All four, in a fixed order, so a test can iterate them.
    pub const ALL: [Self; 4] = [
        Self::HEAD_PLAIN,
        Self::HEAD_JOIN,
        Self::TAIL_PLAIN,
        Self::TAIL_JOIN,
    ];

    /// The entry point name at tile size `k`.
    pub fn entry(self, k: u32) -> String {
        format!(
            "ntt_{}_{}_k{k}",
            if self.head { "head" } else { "tail" },
            if self.join { "join" } else { "plain" }
        )
    }

    /// Binding numbers this mode's pipeline layout must declare, params first.
    ///
    /// This is the list [`crate::ntt::Ntt`] builds the layout from and the list the shader
    /// is generated against, so a test counting storage buffers here counts what the device
    /// will enforce.
    pub fn bindings(self) -> Vec<u32> {
        let mut v = vec![BIND_PARAMS];
        if self.head {
            v.push(BIND_SRC);
            if !self.join {
                v.push(BIND_DST);
            }
        } else {
            v.push(BIND_DST);
        }
        v.push(BIND_TW);
        if self.head {
            v.push(BIND_PTAB);
        }
        if self.join {
            v.extend([BIND_JOIN_A, BIND_JOIN_B, BIND_H_MONT, BIND_H_STD]);
        }
        v
    }

    /// Storage buffers this mode's pipeline layout declares. Everything but `BIND_PARAMS`,
    /// which is a uniform and does not compete for the floor's 8.
    pub fn storage_buffers(self) -> u32 {
        self.bindings().len() as u32 - 1
    }
}

/// Storage buffers the busiest of the four entry points declares. 7, see the module docs.
pub fn storage_buffers() -> u32 {
    Mode::ALL.iter().map(|m| m.storage_buffers()).max().unwrap()
}

/// Bytes of workgroup storage one entry point at tile size `k` allocates.
///
/// `2^k` elements at 32 bytes each. Asserted against the floor's 16384 by
/// [`ntt_module_at`] and, independently, by `tests/ntt.rs` against the granted limit.
pub fn workgroup_bytes(k: u32, v: Variant) -> u64 {
    (v.bytes_per_elem() as u64) << k
}

/// The whole stages 1 to 3 module: the `Fr` prelude, then four entry points for each `k` in
/// `ks`, each at the workgroup size [`workgroup_for`] measured for its own tile.
pub fn ntt_module(v: Variant, ks: &[u32]) -> String {
    ntt_module_at(v, ks, None)
}

/// Same, with every entry point forced to one workgroup size.
///
/// `None` is the per-tile rule, which is what ships. `Some(w)` is for the sweep that produced
/// that rule, and for U14 to run it again under Tint.
///
/// # Panics
///
/// On [`Variant::NoCarry13x20`], for the same reason `gen::gather` does: its `R = 2^260`
/// wire format is not `g16_gpu_layout::PackedFr` and no host packer produces it.
///
/// On a workgroup size outside `1..=256` (the floor's `maxComputeInvocationsPerWorkgroup`),
/// on `k > MAX_FUSED_PASSES`, and on a `k` whose slice would exceed the floor's 16384-byte
/// workgroup allocation. All three are assertions rather than errors because they are
/// statements about the generator's own arguments, and a device that could satisfy them is
/// not a device a browser will ever be.
pub fn ntt_module_at(v: Variant, ks: &[u32], workgroup: Option<u32>) -> String {
    assert_eq!(
        v.limbs(),
        LIMBS,
        "the NTT reads and writes g16_gpu_layout::PackedFr, which is {LIMBS} limbs; \
         variant {v:?} is {} and has no host packer",
        v.limbs()
    );
    if let Some(w) = workgroup {
        assert!(
            w > 0 && w <= 256,
            "workgroup size {w} is outside 1..=256, the floor's \
             maxComputeInvocationsPerWorkgroup"
        );
    }
    assert!(!ks.is_empty(), "no tile sizes to generate");

    // Only the Fr half of the prelude. Fq and Fq2 are 55 KiB of straight-line multiply that
    // no NTT kernel calls, and the module doc on `crate::pipelines` records 155 to 174 ms of
    // naga time for a 29 KiB single-field module, so shipping the curve field here would
    // roughly triple this module's compile for nothing.
    let mut s = format!(
        "// Generated by g16-wgpu::gen::ntt, variant {v:?}, tiles {ks:?}, workgroup {}.\n\
         // Do not edit by hand.\n",
        match workgroup {
            Some(w) => format!("{w} forced"),
            None => "per tile".to_string(),
        }
    );
    if v.needs_mul64() {
        s.push_str(crate::gen::field::MUL64);
    }
    s.push_str(&crate::gen::field::FR.ops(v));
    s.push_str(&prelude());

    let mut seen: Vec<u32> = Vec::new();
    for &k in ks {
        assert!(
            k <= MAX_FUSED_PASSES,
            "tile size k = {k} is over MAX_FUSED_PASSES = {MAX_FUSED_PASSES}"
        );
        let bytes = workgroup_bytes(k, v);
        assert!(
            bytes <= 16384,
            "tile size k = {k} needs {bytes} bytes of workgroup storage, over the floor's \
             16384; maxComputeWorkgroupStorageSize never rises in a browser's default tier"
        );
        if seen.contains(&k) {
            continue;
        }
        seen.push(k);
        writeln!(s, "\nvar<workgroup> SH{k}: array<Fr, {}>;", 1u32 << k).unwrap();
        let wg = workgroup.unwrap_or_else(|| workgroup_for(k));
        for m in Mode::ALL {
            s.push_str(&entry_point(m, k, wg));
        }
    }
    s
}

/// Every module-scope declaration the four entry points draw from.
///
/// All nine bindings are declared once even though no entry point uses more than eight of
/// them. WebGPU validates a pipeline layout against the entry point's *static* resource
/// interface, so a variable a given entry point never reads costs it nothing and does not
/// have to appear in its layout. That is what makes the split worth doing at all.
fn prelude() -> String {
    format!(
        "
// ---------------------------------------------------------------------------
// Stages 1 to 3: the NTT. Generated by g16-wgpu::gen::ntt; do not edit by hand.
// ---------------------------------------------------------------------------

// Mirrors g16_wgpu::ntt::NttParams. 48 bytes.
//
// kscale is eight flat u32 and not an `Fr`, which is `array<u32, 8>`. In the uniform address
// space WGSL rounds every array element's stride up to 16 bytes, so an `Fr` member here
// would occupy 128 bytes and read back as one limb per 16, i.e. plausible garbage with no
// validation error anywhere. Storage buffers have no such rule, which is why `array<Fr>`
// below is fine and this is not.
struct NttParams {{
    log_n: u32,
    s0: u32,
    scale_mode: u32,
    pad0: u32,
    ks0: u32, ks1: u32, ks2: u32, ks3: u32,
    ks4: u32, ks5: u32, ks6: u32, ks7: u32,
}};

@group(0) @binding({BIND_PARAMS}) var<uniform> P: NttParams;
// Head only: the transform's input, read out of order through the bit-reverse.
@group(0) @binding({BIND_SRC}) var<storage, read> SRC: array<Fr>;
// The head's output and every later batch's in-place buffer. One binding, because the tail
// operates on exactly the buffer the head wrote.
@group(0) @binding({BIND_DST}) var<storage, read_write> DST: array<Fr>;
// Powers of the domain's own root, natural order, n/2 entries. Forward or inverse table
// depending on which the host bound.
@group(0) @binding({BIND_TW}) var<storage, read> TW: array<Fr>;
// shift^j for j in [0, n), a primitive 2n-th root's powers. NOT the twiddles; see
// g16_core::cpu::CpuCircuit::new for why that specific element.
@group(0) @binding({BIND_PTAB}) var<storage, read> PTAB: array<Fr>;
// Join only: the two coset vectors stage 4 multiplies, and H in both encodings.
@group(0) @binding({BIND_JOIN_A}) var<storage, read> JOIN_A: array<Fr>;
@group(0) @binding({BIND_JOIN_B}) var<storage, read> JOIN_B: array<Fr>;
@group(0) @binding({BIND_H_MONT}) var<storage, read_write> H_MONT: array<Fr>;
@group(0) @binding({BIND_H_STD}) var<storage, read_write> H_STD: array<Fr>;

fn ntt_kscale() -> Fr {{
    return Fr(P.ks0, P.ks1, P.ks2, P.ks3, P.ks4, P.ks5, P.ks6, P.ks7);
}}
"
    )
}

/// A literal-bound repeat loop that covers `count` items with `workgroup` threads.
///
/// Returns the loop header and footer. Metal writes these as `for (m = tid; m < blk;
/// m += tgsz)`, whose trip count is not a compile-time constant; a `workgroupBarrier()` after
/// such a loop is what WGSL's uniformity analysis rejects. Here the trip count is a literal
/// and the range test is an ordinary `if` inside the body, which reconverges before the
/// barrier. When the workgroup size divides `count` the test is not emitted at all, which is
/// every batch of every artifact at the shipped shape: [`workgroup_for`] sets the width to
/// the butterfly count, so the pass loop is one rep with no test and the load and store loops
/// are two reps with no test.
fn worker_loop(name: &str, count: u32, workgroup: u32, indent: &str) -> (String, String) {
    let reps = count.div_ceil(workgroup);
    let mut head = String::new();
    let mut foot = String::new();
    if reps == 1 {
        writeln!(head, "{indent}{{ let {name} = tid;").unwrap();
    } else {
        writeln!(
            head,
            "{indent}for (var rep = 0u; rep < {reps}u; rep = rep + 1u) {{ \
             let {name} = tid + rep * {workgroup}u;"
        )
        .unwrap();
    }
    if reps * workgroup > count {
        writeln!(head, "{indent}    if ({name} < {count}u) {{").unwrap();
        writeln!(foot, "{indent}    }}").unwrap();
    }
    writeln!(foot, "{indent}}}").unwrap();
    (head, foot)
}

/// One entry point: load, `k` passes with a barrier between them, store.
fn entry_point(m: Mode, k: u32, workgroup: u32) -> String {
    let blk = 1u32 << k;
    let name = m.entry(k);
    let mut s = String::new();

    writeln!(s, "\n@compute @workgroup_size({workgroup})").unwrap();
    writeln!(
        s,
        "fn {name}(@builtin(workgroup_id) wid: vec3<u32>,\n         \
         @builtin(local_invocation_index) tid: u32) {{"
    )
    .unwrap();

    // Where this workgroup's slice starts, and the low bits every index in it shares.
    if m.head {
        writeln!(s, "    let base = wid.x << {k}u;").unwrap();
        writeln!(s, "    let low = 0u;").unwrap();
        writeln!(
            s,
            "    // 32 - log_n, masked so a one-point domain (log_n = 0) does not shift by\n\
             \x20   // the full word width, which WGSL leaves indeterminate. The reversed\n\
             \x20   // index is thrown away in that case anyway."
        )
        .unwrap();
        writeln!(s, "    let revshift = (32u - P.log_n) & 31u;").unwrap();
    } else {
        writeln!(
            s,
            "    // Group g owns low = g mod 2^s0 and high = g >> s0, so the two halves of g\n\
             \x20   // cover exactly the log_n - {k} bits outside this batch: the groups\n\
             \x20   // partition the domain with no overlap and no gap."
        )
        .unwrap();
        writeln!(s, "    let low = wid.x & ((1u << P.s0) - 1u);").unwrap();
        writeln!(s, "    let high = wid.x >> P.s0;").unwrap();
        writeln!(s, "    let base = low + (high << (P.s0 + {k}u));").unwrap();
    }

    // ---- load ----
    let (lh, lf) = worker_loop("m", blk, workgroup, "    ");
    s.push_str(&lh);
    if m.head {
        writeln!(s, "        let i = base + m;").unwrap();
        writeln!(
            s,
            "        // After the bit-reverse position i holds src[reverse(i)], so the\n\
             \x20       // thread owning output i simply reads the reversed index. That is a\n\
             \x20       // whole read and write of the domain saved per transform, six times\n\
             \x20       // per proof, against the CPU backend's separate permutation pass."
        )
        .unwrap();
        writeln!(
            s,
            "        let src_i = select(0u, reverseBits(i) >> revshift, P.log_n != 0u);"
        )
        .unwrap();
        writeln!(s, "        var x = SRC[src_i];").unwrap();
        writeln!(
            s,
            "        // Dispatch-uniform, so this costs a scalar compare and never diverges."
        )
        .unwrap();
        writeln!(
            s,
            "        if (P.scale_mode == {SCALE_CONST}u) {{ x = fr_mul(x, ntt_kscale()); }}"
        )
        .unwrap();
        writeln!(
            s,
            "        else if (P.scale_mode == {SCALE_TABLE}u) {{\n\
             \x20           // Indexed by the SOURCE index, which is the natural-order\n\
             \x20           // position this shift power belongs to. Indexing by i would\n\
             \x20           // apply shift^bitrev(j), a different and wrong vector.\n\
             \x20           x = fr_mul(x, PTAB[src_i]);\n\
             \x20       }}"
        )
        .unwrap();
        writeln!(s, "        SH{k}[m] = x;").unwrap();
    } else {
        writeln!(s, "        SH{k}[m] = DST[base + (m << P.s0)];").unwrap();
    }
    s.push_str(&lf);
    writeln!(s, "    workgroupBarrier();").unwrap();

    // ---- the k passes ----
    if k >= 1 {
        let halves = 1u32 << (k - 1);
        writeln!(
            s,
            "\n    // Pass t flips bit s0 + t of the element index, so the k passes of this\n\
             \x20   // batch touch only bits [s0, s0+k) and every group can run all of them\n\
             \x20   // out of workgroup memory with a barrier in between. Literal bound: the\n\
             \x20   // barrier below must sit in uniform control flow or the shader is\n\
             \x20   // rejected, which is a hard error in WGSL and merely UB in MSL."
        )
        .unwrap();
        writeln!(s, "    for (var t = 0u; t < {k}u; t = t + 1u) {{").unwrap();
        writeln!(s, "        let hl = 1u << t;").unwrap();
        // Identical text in the head and the tail: the head is only ever dispatched with
        // s0 = 0 (`crate::ntt` asserts it), so writing the general form here means the two
        // bodies cannot drift and a head dispatched at the wrong s0 is wrong consistently
        // rather than half-applied.
        writeln!(s, "        let twshift = P.log_n - P.s0 - t - 1u;").unwrap();
        let (bh, bf) = worker_loop("bfy", halves, workgroup, "        ");
        s.push_str(&bh);
        writeln!(s, "            let jm = bfy & (hl - 1u);").unwrap();
        writeln!(s, "            let lo = ((bfy >> t) << (t + 1u)) | jm;").unwrap();
        writeln!(s, "            let hi = lo + hl;").unwrap();
        writeln!(s, "            let u = SH{k}[lo];").unwrap();
        writeln!(
            s,
            "            // Butterfly j of a block wants root^(j * n / 2^(t+1)); the table\n\
             \x20           // is natural order, so that is a strided read. For a strided\n\
             \x20           // slice (s0 > 0) the index within the block is low + jm*2^s0."
        )
        .unwrap();
        writeln!(
            s,
            "            let v = fr_mul(SH{k}[hi], TW[(low + (jm << P.s0)) << twshift]);"
        )
        .unwrap();
        writeln!(s, "            SH{k}[lo] = fr_add(u, v);").unwrap();
        writeln!(s, "            SH{k}[hi] = fr_sub(u, v);").unwrap();
        s.push_str(&bf);
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(s, "    }}").unwrap();
    }

    // ---- store ----
    writeln!(s).unwrap();
    let (sh, sf) = worker_loop("m", blk, workgroup, "    ");
    s.push_str(&sh);
    let index = if m.head {
        "base + m"
    } else {
        "base + (m << P.s0)"
    };
    if m.join {
        writeln!(s, "        let i = {index};").unwrap();
        writeln!(
            s,
            "        // Stage 4, fused. No division by Z(coset): snarkjs' joinABC computes\n\
             \x20       // exactly a*b - c and the Z is already folded into the zkey's\n\
             \x20       // section 9 bases. See g16-metal/src/shaders/pointwise.metal."
        )
        .unwrap();
        writeln!(
            s,
            "        let h = fr_sub(fr_mul(JOIN_A[i], JOIN_B[i]), SH{k}[m]);"
        )
        .unwrap();
        writeln!(s, "        H_MONT[i] = h;").unwrap();
        writeln!(
            s,
            "        // Standard form for stage 9: a window digit of a Montgomery\n\
             \x20       // representative is a digit of a*R mod r, a different number.\n\
             \x20       // fr_from_mont is a Montgomery multiply by the integer 1."
        )
        .unwrap();
        writeln!(s, "        H_STD[i] = fr_from_mont(h);").unwrap();
    } else {
        writeln!(s, "        DST[{index}] = SH{k}[m];").unwrap();
    }
    s.push_str(&sf);
    writeln!(s, "}}").unwrap();
    s
}
