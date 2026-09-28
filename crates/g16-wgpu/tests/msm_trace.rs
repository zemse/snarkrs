//! Where a large MSM stage's time goes, kernel by kernel, and what this machine's measured
//! intrinsics say each kernel could cost. BUG-30's trace.
//!
//! Both tests are ignored: they take the GPU lock, print tables and assert nothing. The
//! artifact one takes its key from `G16_TRACE_ARTIFACT` (a directory holding `circuit.zkey`
//! and `circuit.wtns`) and its profile from `G16_WGPU_LIMITS`:
//!
//! ```text
//! G16_TRACE_ARTIFACT=bench/artifacts/large/js_384x384_d32 G16_WGPU_LIMITS=auto \
//!     cargo test --release -p g16-wgpu --test msm_trace -- --ignored --nocapture
//! ```
//!
//! The stage is timed whole through the prover first, then every sub-MSM the batch would cut
//! is stood up on its own and each of its kernels is submitted alone and timed, medians of
//! three, so the difference between the two is the submission and readback overhead. The
//! kernels are timed in the batch's order and over the batch's buffers, so a sort feeds the
//! point stage it is timed next to; a kernel timed three times runs its second and third
//! over what its first left, which changes no kernel's cost.

mod gpulock;
mod msmcommon;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ark_ff::UniformRand;
use ark_std::test_rng;
use g16_core::{Backend, StageTimings};
use g16_field::{CurveGroup, Fq, Fq2, G1Projective, G2Projective, One, Zero};
use g16_gpu_layout::{as_bytes, PackedFq, PackedFq2, PackedG1Affine, PackedG2Affine};
use g16_wgpu::batch::base_chunk_elems;
use g16_wgpu::gen::field::Variant;
use g16_wgpu::gen::points as wgsl;
use g16_wgpu::stages::{WgpuHandle, TAG};
use g16_wgpu::{
    DigitBuffers, DigitPlan, G1Bases, G2Bases, Kernels, LimitsProfile, MsmPoints, ParamRing,
    PointBuffers, PointCurve, WgpuBackend, WgpuProver, Work,
};
use g16_zkey::{wtns::Witness, ProvingKey};
use msmcommon::{median, submit_sealed};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One sealed submission of `encode`, in milliseconds of wall time from encode to verified
/// readback, which at the sizes here is the GPU time plus about 0.4 ms.
fn timed(b: &WgpuBackend, what: &str, encode: impl Fn(&mut wgpu::ComputePass<'_>)) -> f64 {
    let t = std::time::Instant::now();
    submit_sealed(b, what, encode);
    t.elapsed().as_secs_f64() * 1e3
}

/// Median of three timed submissions.
fn timed3(b: &WgpuBackend, what: &str, encode: impl Fn(&mut wgpu::ComputePass<'_>)) -> f64 {
    median((0..3).map(|_| timed(b, what, &encode)).collect())
}

fn storage(b: &WgpuBackend, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buf = b.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        b.queue().write_buffer(&buf, 0, bytes);
    }
    buf
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// The intrinsics: what one multiply, one mixed addition and one byte cost here
// ---------------------------------------------------------------------------

/// A probe kernel over its own bind group 1, appended to a generated module so it runs the
/// exact `fq_mul` and `pt_madd_*` the prover runs.
struct Probe {
    kernels: Kernels,
    bgl: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
}

impl Probe {
    fn build(
        b: &WgpuBackend,
        label: &str,
        src: &str,
        entry: &str,
        group: u32,
        read_only: &[bool],
    ) -> Self {
        let bgl = b
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &read_only
                    .iter()
                    .enumerate()
                    .map(|(i, ro)| storage_entry(i as u32, *ro))
                    .collect::<Vec<_>>(),
            });
        // Group 0 stays empty so the generated module's own declarations, which the probe
        // never touches, need no resources.
        let empty = b
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("probe empty"),
                entries: &[],
            });
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = (0..=group)
            .map(|g| if g == group { Some(&bgl) } else { Some(&empty) })
            .collect();
        let layout = b
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &groups,
                immediate_size: 0,
            });
        let kernels = Kernels::build(b, label, src, &layout, &[entry]).expect("probe module");
        Self {
            kernels,
            bgl,
            _layout: layout,
        }
    }

    fn bind(&self, b: &WgpuBackend, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        b.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe"),
            layout: &self.bgl,
            entries: &entries,
        })
    }
}

/// `n` distinct affine points, a random start stepped by a random point, so a million of
/// them cost a million additions rather than a million scalar multiplications.
fn walk<G: CurveGroup + UniformRand>(
    n: usize,
    rng: &mut impl ark_std::rand::Rng,
) -> Vec<G::Affine> {
    let step = G::rand(rng);
    let mut p = G::rand(rng);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(p);
        p += step;
    }
    G::normalize_batch(&out)
}

/// Threads in the arithmetic chains. Enough to fill 38 cores many times over, so the number
/// is a throughput and not a latency.
const CHAIN_THREADS: u32 = 1 << 18;

#[test]
#[ignore = "BUG-30's intrinsics: prints a table, asserts nothing"]
fn what_the_intrinsics_measure() {
    let _lock = gpulock::exclusive_gpu();
    let profile = LimitsProfile::from_env().expect("G16_WGPU_LIMITS");
    let b = pollster::block_on(WgpuBackend::with_profile(profile)).expect("device");
    let mut rng = test_rng();
    println!(
        "intrinsics, {} profile, medians of 3 sealed submissions",
        profile.as_str()
    );

    // ---- one Fq multiply, and one Fq2 multiply in several spellings, as a chain per thread ----
    let prelude = g16_wgpu::gen::field_module(Variant::default());
    let iters = 128u32;
    // (name, element type of IO, functions appended to the prelude, how the loop starts, the
    // loop body, how the loop ends). Every Fq2 spelling computes the same Karatsuba product.
    let spellings: [(&str, &str, &str, &str, &str, &str); 11] = [
        ("fq_mul", "Fq", "", "var a = IO[gid.x]; let m = IO[gid.x ^ 1u];", "a = fq_mul(a, m);", "IO[gid.x] = a;"),
        // Two and three multiplies per iteration through two and three call sites, each
        // over the same few live values, so what they add is copies and not pressure.
        ("fq 2 sites", "Fq", "", "var a = IO[gid.x]; let m = IO[gid.x ^ 1u]; let m2 = IO[gid.x ^ 2u];", "a = fq_mul(a, m); a = fq_mul(a, m2);", "IO[gid.x] = a;"),
        ("fq 3 sites", "Fq", "", "var a = IO[gid.x]; let m = IO[gid.x ^ 1u]; let m2 = IO[gid.x ^ 2u];", "a = fq_mul(a, m); a = fq_mul(a, m2); a = fq_mul(a, m);", "IO[gid.x] = a;"),
        // Karatsuba driven through one fq_mul call site, the operands picked by an if chain
        // and by indexing a three-slot array.
        (
            "fq2 one site arr",
            "Fq2",
            "fn fq2_mul_a(a: Fq2, b: Fq2) -> Fq2 {
    var l: array<Fq, 3>;
    var r: array<Fq, 3>;
    var p: array<Fq, 3>;
    l[0] = a.c0; l[1] = a.c1; l[2] = fq_add(a.c0, a.c1);
    r[0] = b.c0; r[1] = b.c1; r[2] = fq_add(b.c0, b.c1);
    for (var s = 0u; s < 3u; s = s + 1u) { p[s] = fq_mul(l[s], r[s]); }
    return Fq2(fq_sub(p[0], p[1]), fq_sub(fq_sub(p[2], p[0]), p[1]));
}",
            "var a = IO[gid.x]; let m = IO[gid.x ^ 1u];",
            "a = fq2_mul_a(a, m);",
            "IO[gid.x] = a;",
        ),
        (
            "fq2 one site",
            "Fq2",
            "fn fq2_mul_1(a: Fq2, b: Fq2) -> Fq2 {
    var v0: Fq;
    var v1: Fq;
    var x: Fq;
    for (var s = 0u; s < 3u; s = s + 1u) {
        var l: Fq;
        var r: Fq;
        if (s == 0u) { l = a.c0; r = b.c0; }
        else if (s == 1u) { l = a.c1; r = b.c1; }
        else { l = fq_add(a.c0, a.c1); r = fq_add(b.c0, b.c1); }
        let p = fq_mul(l, r);
        if (s == 0u) { v0 = p; } else if (s == 1u) { v1 = p; } else { x = p; }
    }
    return Fq2(fq_sub(v0, v1), fq_sub(fq_sub(x, v0), v1));
}",
            "var a = IO[gid.x]; let m = IO[gid.x ^ 1u];",
            "a = fq2_mul_1(a, m);",
            "IO[gid.x] = a;",
        ),
        ("fq2_mul", "Fq2", "", "var a = IO[gid.x]; let m = IO[gid.x ^ 1u];", "a = fq2_mul(a, m);", "IO[gid.x] = a;"),
        (
            "fq2 inline",
            "Fq2",
            "",
            "var a0 = IO[gid.x].c0; var a1 = IO[gid.x].c1; let m0 = IO[gid.x ^ 1u].c0; let m1 = IO[gid.x ^ 1u].c1;",
            "let v0 = fq_mul(a0, m0); let v1 = fq_mul(a1, m1); let x = fq_mul(fq_add(a0, a1), fq_add(m0, m1)); a1 = fq_sub(fq_sub(x, v0), v1); a0 = fq_sub(v0, v1);",
            "IO[gid.x] = Fq2(a0, a1);",
        ),
        (
            "fq2 struct expr",
            "Fq2",
            "",
            "var a = IO[gid.x]; let m = IO[gid.x ^ 1u];",
            "let v0 = fq_mul(a.c0, m.c0); let v1 = fq_mul(a.c1, m.c1); let x = fq_mul(fq_add(a.c0, a.c1), fq_add(m.c0, m.c1)); a = Fq2(fq_sub(v0, v1), fq_sub(fq_sub(x, v0), v1));",
            "IO[gid.x] = a;",
        ),
        (
            "fq2 split params",
            "Fq2",
            "fn fq2_mul_p(a0: Fq, a1: Fq, b0: Fq, b1: Fq) -> Fq2 {
    let v0 = fq_mul(a0, b0);
    let v1 = fq_mul(a1, b1);
    let x = fq_mul(fq_add(a0, a1), fq_add(b0, b1));
    return Fq2(fq_sub(v0, v1), fq_sub(fq_sub(x, v0), v1));
}",
            "var a0 = IO[gid.x].c0; var a1 = IO[gid.x].c1; let m0 = IO[gid.x ^ 1u].c0; let m1 = IO[gid.x ^ 1u].c1;",
            "let r = fq2_mul_p(a0, a1, m0, m1); a0 = r.c0; a1 = r.c1;",
            "IO[gid.x] = Fq2(a0, a1);",
        ),
        (
            "fq2 ptr out",
            "Fq2",
            "fn fq2_mul_ptr(a0: Fq, a1: Fq, b0: Fq, b1: Fq, o0: ptr<function, Fq>, o1: ptr<function, Fq>) {
    let v0 = fq_mul(a0, b0);
    let v1 = fq_mul(a1, b1);
    let x = fq_mul(fq_add(a0, a1), fq_add(b0, b1));
    *o0 = fq_sub(v0, v1);
    *o1 = fq_sub(fq_sub(x, v0), v1);
}",
            "var a0 = IO[gid.x].c0; var a1 = IO[gid.x].c1; let m0 = IO[gid.x ^ 1u].c0; let m1 = IO[gid.x ^ 1u].c1;",
            "fq2_mul_ptr(a0, a1, m0, m1, &a0, &a1);",
            "IO[gid.x] = Fq2(a0, a1);",
        ),
        (
            "fq in struct",
            "Fq",
            "struct W { f: Fq }
fn w_mul(a: W, b: W) -> W { return W(fq_mul(a.f, b.f)); }",
            "var a = W(IO[gid.x]); let m = W(IO[gid.x ^ 1u]);",
            "a = w_mul(a, m);",
            "IO[gid.x] = a.f;",
        ),
    ];
    let fq_bytes: Vec<u8> = {
        let xs: Vec<PackedFq> = (0..CHAIN_THREADS)
            .map(|_| PackedFq::from_fq(&Fq::rand(&mut rng)))
            .collect();
        as_bytes(&xs).to_vec()
    };
    let fq2_bytes: Vec<u8> = {
        let xs: Vec<PackedFq2> = (0..CHAIN_THREADS)
            .map(|_| PackedFq2::from_fq2(&Fq2::rand(&mut rng)))
            .collect();
        as_bytes(&xs).to_vec()
    };
    // A fresh entry point name per run: Metal caches compiled functions on disk by name and
    // text, and the pipeline time is only a compile time when it misses.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    for (name, ty, fns, head, body, tail) in spellings {
        let entry = format!("chain_{nonce}");
        let src = format!(
            "{prelude}
{fns}
@group(1) @binding(0) var<storage, read_write> IO: array<{ty}>;
@compute @workgroup_size(256)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    {head}
    for (var i = 0u; i < {iters}u; i = i + 1u) {{ {body} }}
    {tail}
}}"
        );
        let probe = Probe::build(&b, name, &src, &entry, 1, &[false]);
        let io = storage(&b, name, if ty == "Fq" { &fq_bytes } else { &fq2_bytes });
        let bind = probe.bind(&b, &[&io]);
        let ms = timed3(&b, name, |pass| {
            pass.set_pipeline(probe.kernels.get(&entry).unwrap());
            pass.set_bind_group(1, &bind, &[]);
            pass.dispatch_workgroups(CHAIN_THREADS / 256, 1, 1);
        });
        let per_iter = match name {
            "fq 2 sites" => 2.0,
            "fq 3 sites" => 3.0,
            _ => 1.0,
        };
        let muls = f64::from(CHAIN_THREADS) * f64::from(iters) * per_iter;
        println!(
            "  {name:<18} {ms:8.2} ms for {:.1}M, {:.3} G/s, {:.2} ns each, pipeline {:.0} ms",
            muls / 1e6,
            muls / ms / 1e6,
            ms * 1e6 / muls,
            probe.kernels.cost().pipeline_us as f64 / 1e3
        );
    }

    // ---- one mixed addition per curve, over bases read sequentially and at random ----
    let n_bases = 1u32 << 20;
    let g1 = walk::<G1Projective>(n_bases as usize, &mut rng);
    let g2 = walk::<G2Projective>(n_bases as usize, &mut rng);
    let b1 = storage(&b, "g1 bases", as_bytes(&PackedG1Affine::pack_slice(&g1)));
    let b2 = storage(&b, "g2 bases", as_bytes(&PackedG2Affine::pack_slice(&g2)));
    let iters = 64u32;
    let threads = 1u32 << 17;
    for curve in [wgsl::G1, wgsl::G2] {
        let sfx = curve.suffix;
        let src = format!(
            "{}
@group(1) @binding(0) var<storage, read_write> ACC: array<{pt}>;
@group(1) @binding(1) var<storage, read> BB: array<{aff}>;
@group(1) @binding(2) var<storage, read> MODE: array<u32>;
@compute @workgroup_size(128)
fn chain(@builtin(global_invocation_id) gid: vec3<u32>) {{
    var acc = pt_zero_{sfx}();
    var j = gid.x;
    for (var i = 0u; i < {iters}u; i = i + 1u) {{
        if (MODE[0] != 0u) {{
            j = (j * 2654435761u) ^ (j >> 13u);
        }} else {{
            j = j + {threads}u;
        }}
        acc = pt_madd_{sfx}(acc, BB[j & {mask}u]);
    }}
    ACC[gid.x] = acc;
}}",
            g16_wgpu::gen::points_module(Variant::default(), curve),
            pt = curve.pt,
            aff = curve.aff,
            mask = n_bases - 1,
        );
        let probe = Probe::build(
            &b,
            &format!("madd {sfx}"),
            &src,
            "chain",
            1,
            &[false, true, true],
        );
        let acc = storage(
            &b,
            "acc",
            &vec![0u8; (threads as u64 * curve.point_bytes) as usize],
        );
        let bases = if curve.suffix == wgsl::G1.suffix {
            &b1
        } else {
            &b2
        };
        for (mode, label) in [(0u32, "sequential"), (1, "random")] {
            let m = storage(&b, "mode", &mode.to_le_bytes());
            let bind = probe.bind(&b, &[&acc, bases, &m]);
            let ms = timed3(&b, "madd", |pass| {
                pass.set_pipeline(probe.kernels.get("chain").unwrap());
                pass.set_bind_group(1, &bind, &[]);
                pass.dispatch_workgroups(threads / 128, 1, 1);
            });
            let adds = f64::from(threads) * f64::from(iters);
            println!(
                "  madd {sfx} {label:<10} {ms:8.2} ms for {:.1}M, {:.2} ns each ({} threads x {iters})",
                adds / 1e6,
                ms * 1e6 / adds,
                threads
            );
        }
    }

    // ---- bytes: a sequential copy, and 128-byte fetches at random addresses ----
    let mib = 128u64;
    let words = (mib << 20) / 16;
    let src_copy = "
@group(1) @binding(0) var<storage, read> SRC: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read_write> DST: array<vec4<u32>>;
@compute @workgroup_size(256)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    DST[gid.x] = SRC[gid.x];
}";
    let probe = Probe::build(&b, "copy", src_copy, "copy", 1, &[true, false]);
    let a = storage(&b, "copy src", &vec![1u8; (mib << 20) as usize]);
    let c = storage(&b, "copy dst", &vec![0u8; (mib << 20) as usize]);
    let bind = probe.bind(&b, &[&a, &c]);
    let ms = timed3(&b, "copy", |pass| {
        pass.set_pipeline(probe.kernels.get("copy").unwrap());
        pass.set_bind_group(1, &bind, &[]);
        pass.dispatch_workgroups((words / 256) as u32, 1, 1);
    });
    println!(
        "  copy {mib} MiB      {ms:8.2} ms, {:.1} GB/s read + write",
        2.0 * (mib << 20) as f64 / ms / 1e6
    );
    // Eight vec4 loads per thread from one random 128-byte line of the G2 base vector.
    let src_gather = format!(
        "
@group(1) @binding(0) var<storage, read> SRC: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read_write> DST: array<vec4<u32>>;
@compute @workgroup_size(128)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {{
    var j = gid.x;
    var acc = vec4<u32>(0u);
    for (var i = 0u; i < 32u; i = i + 1u) {{
        j = (j * 2654435761u) ^ (j >> 13u);
        let at = (j & {}u) * 8u;
        for (var k = 0u; k < 8u; k = k + 1u) {{ acc = acc ^ SRC[at + k]; }}
    }}
    DST[gid.x] = acc;
}}",
        n_bases - 1
    );
    let probe = Probe::build(&b, "gather", &src_gather, "gather", 1, &[true, false]);
    let bind = probe.bind(&b, &[&b2, &c]);
    let threads = 1u32 << 18;
    let ms = timed3(&b, "gather", |pass| {
        pass.set_pipeline(probe.kernels.get("gather").unwrap());
        pass.set_bind_group(1, &bind, &[]);
        pass.dispatch_workgroups(threads / 128, 1, 1);
    });
    let fetches = f64::from(threads) * 32.0;
    println!(
        "  gather 128 B     {ms:8.2} ms for {:.1}M fetches from {} MiB, {:.2} ns each, {:.1} GB/s",
        fetches / 1e6,
        (u64::from(n_bases) * 128) >> 20,
        ms * 1e6 / fetches,
        fetches * 128.0 / ms / 1e6
    );

    // ---- one empty sealed submission ----
    let ms = median((0..9).map(|_| timed(&b, "empty", |_| {})).collect());
    println!("  empty submit     {ms:8.3} ms encode to verified readback");
}

// ---------------------------------------------------------------------------
// The trace: one artifact's MSM stage, kernel by kernel
// ---------------------------------------------------------------------------

/// One sub-MSM the batch would run: its scalar range and plan, and which base buffer each
/// job reads.
struct Piece {
    lo: u32,
    dplan: DigitPlan,
}

/// `MsmBatch::pieces` for a group whose jobs all start at base offset 0: the cap halves
/// until every piece's plan fits `limit`, and a piece never crosses a base buffer.
#[allow(clippy::too_many_arguments)]
fn pieces(
    n: u32,
    scalar_off: u32,
    general: Option<u32>,
    curves: &[wgsl::Curve],
    limit: u64,
    chunk: u32,
    g1: &MsmPoints<g16_wgpu::G1Curve>,
    g2: &MsmPoints<g16_wgpu::G2Curve>,
) -> Vec<Piece> {
    let fits = |dplan: &DigitPlan| -> bool {
        if dplan.largest_binding() > limit {
            return false;
        }
        curves.iter().all(|c| {
            let largest = if c.suffix == wgsl::G1.suffix {
                g1.plan_points(dplan, 0).unwrap().largest_binding(dplan, *c)
            } else {
                g2.plan_points(dplan, 0).unwrap().largest_binding(dplan, *c)
            };
            largest <= limit
        })
    };
    let mut cap = n;
    loop {
        let mut out = Vec::new();
        let mut lo = 0u32;
        let mut ok = true;
        while lo < n {
            let boundary = (u64::from(lo) / u64::from(chunk) + 1) * u64::from(chunk);
            let hi = u64::from(lo.saturating_add(cap).min(n)).min(boundary) as u32;
            let len = hi - lo;
            let dplan = DigitPlan::with_work(
                len,
                scalar_off + lo,
                general.map(|x| x.min(len)),
                Work::Variable,
            )
            .unwrap();
            if !fits(&dplan) {
                ok = false;
                break;
            }
            out.push(Piece { lo, dplan });
            lo = hi;
        }
        if ok {
            return out;
        }
        cap = if cap.is_power_of_two() {
            cap / 2
        } else {
            1 << (31 - cap.leading_zeros())
        };
    }
}

/// Milliseconds per kernel of one sub-MSM's one job.
#[derive(Default, Clone, Copy)]
struct Phases {
    zero: f64,
    count: f64,
    scan: f64,
    scatter: f64,
    clear: f64,
    segmented: f64,
    merge: f64,
    reduce: f64,
    ones: f64,
    /// One window's clear, accumulation and merge on its own, times the window count: what
    /// the stage costs cut into one-window slabs.
    slab1_x_w: f64,
}

impl Phases {
    fn sort(&self) -> f64 {
        self.zero + self.count + self.scan + self.scatter
    }
    fn points(&self) -> f64 {
        self.clear + self.segmented + self.merge + self.reduce + self.ones
    }
    fn add(&mut self, o: &Phases) {
        self.zero += o.zero;
        self.count += o.count;
        self.scan += o.scan;
        self.scatter += o.scatter;
        self.clear += o.clear;
        self.segmented += o.segmented;
        self.merge += o.merge;
        self.reduce += o.reduce;
        self.ones += o.ones;
        self.slab1_x_w += o.slab1_x_w;
    }
}

/// Times every kernel of one job over one piece, the sort included when `with_sort`.
#[allow(clippy::too_many_arguments)]
fn probe_job<C: PointCurve>(
    b: &WgpuBackend,
    d: &g16_wgpu::MsmDigits,
    p: &MsmPoints<C>,
    dplan: &DigitPlan,
    local: u32,
    scalars: &wgpu::Buffer,
    bases: &wgpu::Buffer,
    sort: &DigitBuffers,
    with_sort: bool,
) -> Phases {
    let pplan = p.plan_points(dplan, local).unwrap();
    let pts = PointBuffers::new(b, dplan, &pplan, 0, p.curve()).unwrap();
    let w = dplan.n_windows();
    let mut slabs: Vec<std::ops::Range<u32>> = Vec::new();
    slabs.push(0..1);
    if w > 1 {
        slabs.push(1..w);
    }
    let slots = d.sort_slots(dplan) + p.slots(dplan, &pplan) + p.slots_slabs(dplan, &pplan, &slabs);
    let mut ring = ParamRing::new(b, "trace ring", slots).unwrap();
    let soff = d.plan_sort(dplan, &mut ring).unwrap();
    let poff = p.plan(dplan, &pplan, &mut ring).unwrap();
    let poff1 = p.plan_slabs(dplan, &pplan, &mut ring, &slabs).unwrap();
    ring.flush(b);
    let sbind = d.bind_sort(b, &ring, dplan, scalars, sort).unwrap();
    let pbind = p
        .bind_all(b, &ring, dplan, &pplan, scalars, bases, sort, &pts)
        .unwrap();
    let mut ph = Phases::default();
    if with_sort {
        ph.zero = timed3(b, "zero", |pass| {
            d.encode_zero_rows(pass, dplan, &sbind, &soff).unwrap()
        });
        ph.count = timed3(b, "count", |pass| {
            d.encode_count(pass, dplan, &sbind, &soff).unwrap()
        });
        ph.scan = timed3(b, "scan", |pass| {
            d.encode_scan(pass, dplan, &sbind, &soff).unwrap()
        });
        ph.scatter = timed3(b, "scatter", |pass| {
            d.encode_scatter(pass, dplan, &sbind, &soff).unwrap()
        });
        // The counters the scan turned into cursors are what the scatter advanced; run the
        // whole sort once more so the point stage reads a sort run exactly once.
        submit_sealed(b, "sort", |pass| {
            d.encode_sort(pass, dplan, &sbind, &soff).unwrap()
        });
    }
    ph.clear = timed3(b, "clear", |pass| {
        p.encode_clear(pass, dplan, &pbind, &poff).unwrap()
    });
    ph.segmented = timed3(b, "segmented", |pass| {
        p.encode_segmented(pass, &pplan, &pbind, &poff).unwrap()
    });
    ph.merge = timed3(b, "merge", |pass| {
        p.encode_merge(pass, dplan, &pbind, &poff).unwrap()
    });
    ph.reduce = timed3(b, "reduce", |pass| {
        p.encode_reduce(pass, dplan, &pbind, &poff).unwrap()
    });
    ph.ones = timed3(b, "ones", |pass| {
        p.encode_ones(pass, &pplan, &pbind, &poff).unwrap()
    });
    ph.slab1_x_w = f64::from(w)
        * timed3(b, "slab", |pass| {
            p.encode_slab(pass, dplan, &pplan, &pbind, &poff1, 0)
                .unwrap()
        });
    ph
}

#[test]
#[ignore = "BUG-30's trace: minutes on a 2^22 key, prints a table, asserts nothing"]
fn where_a_large_msm_stage_goes() {
    let _lock = gpulock::exclusive_gpu();
    let dir = std::env::var("G16_TRACE_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("bench/artifacts/js_16x16_d32"));
    let dir = if dir.is_absolute() {
        dir
    } else {
        root().join(dir)
    };
    if !dir.join("circuit.zkey").is_file() {
        eprintln!(
            "SKIPPED where_a_large_msm_stage_goes: no circuit.zkey under {}",
            dir.display()
        );
        return;
    }
    let profile = LimitsProfile::from_env().expect("G16_WGPU_LIMITS");
    let b = Arc::new(pollster::block_on(WgpuBackend::with_profile(profile)).expect("device"));
    let prover = WgpuProver::with_device(Arc::clone(&b)).expect("prover");
    let limit = b.granted_limits().max_storage_buffer_binding_size;
    let chunk = base_chunk_elems(&b);

    let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("circuit.zkey");
    let witness = Witness::load(&dir.join("circuit.wtns"))
        .expect("circuit.wtns")
        .0;
    let n_vars = pk.n_vars as u32;
    let private_from = (pk.n_public + 1) as u32;
    let domain = pk.domain_size as u32;
    let l_len = pk.l_query.len() as u32;
    let (mut g_all, mut g_priv) = (0u32, 0u32);
    for (i, x) in witness.iter().enumerate() {
        if !(x.is_zero() || x.is_one()) {
            g_all += 1;
            if i >= private_from as usize {
                g_priv += 1;
            }
        }
    }
    println!(
        "{} at the {} profile: n_vars {n_vars}, general {g_all} ({g_priv} private), domain \
         {domain}, binding limit {} MiB, base chunk {chunk}",
        dir.file_name().unwrap().to_string_lossy(),
        profile.as_str(),
        limit >> 20
    );

    let a_bases = G1Bases::upload(&b, &pk.a_query).unwrap();
    let b_g1_bases = G1Bases::upload(&b, &pk.b_g1_query).unwrap();
    let b_g2_bases = G2Bases::upload(&b, &pk.b_g2_query).unwrap();
    let l_bases = G1Bases::upload(&b, &pk.l_query).unwrap();
    let h_bases = G1Bases::upload(&b, &pk.h_query).unwrap();
    let circuit = prover.prepare(pk).expect("prepare");

    // ---- the stage whole, through the prover ----
    let mut t = StageTimings::default();
    let h = circuit.compute_h(&witness, &mut t).expect("compute_h");
    println!(
        "stages 0-4: gather {:.1} ms, ntt {:.1} ms",
        t.gather_us as f64 / 1e3,
        t.ntt_us as f64 / 1e3
    );
    let mut whole = Vec::new();
    for _ in 0..3 {
        let mut t = StageTimings::default();
        circuit.msms(&witness, &h, &mut t).expect("msms");
        whole.push(t.msm_us as f64 / 1e3);
    }
    let batch = prover.msm();
    println!(
        "stages 5-9 whole: {:.1} ms (runs {:?}), {} submits, {:?} sub-MSMs (G1, G2), {} B readback",
        median(whole.clone()),
        whole,
        batch.last_submits(),
        batch.last_sub_msms(),
        batch.last_readback_bytes()
    );

    // ---- the same work, one kernel at a time ----
    let handle = h.device_handle::<WgpuHandle>(TAG).expect("a wgpu handle");
    let ws = handle.witness_std();
    let d = batch.digits();
    let (g1, g2) = (batch.g1(), batch.g2());
    struct Grp<'a> {
        name: &'static str,
        scalars: &'a wgpu::Buffer,
        scalar_off: u32,
        n: u32,
        general: Option<u32>,
        jobs: Vec<(&'static str, wgsl::Curve, &'a wgpu::Buffer)>,
        chunks: Vec<Vec<&'a wgpu::Buffer>>,
    }
    fn chunks_g1(x: &G1Bases) -> Vec<&wgpu::Buffer> {
        (0..x.chunks()).map(|i| x.chunk(i)).collect()
    }
    fn chunks_g2(x: &G2Bases) -> Vec<&wgpu::Buffer> {
        (0..x.chunks()).map(|i| x.chunk(i)).collect()
    }
    let groups = [
        Grp {
            name: "witness",
            scalars: ws,
            scalar_off: 0,
            n: n_vars,
            general: Some(g_all),
            jobs: vec![
                ("A g1", wgsl::G1, a_bases.chunk(0)),
                ("B g2", wgsl::G2, b_g2_bases.chunk(0)),
                ("B g1", wgsl::G1, b_g1_bases.chunk(0)),
            ],
            chunks: vec![
                chunks_g1(&a_bases),
                chunks_g2(&b_g2_bases),
                chunks_g1(&b_g1_bases),
            ],
        },
        Grp {
            name: "L",
            scalars: ws,
            scalar_off: private_from,
            n: l_len,
            general: Some(g_priv),
            jobs: vec![("L g1", wgsl::G1, l_bases.chunk(0))],
            chunks: vec![chunks_g1(&l_bases)],
        },
        Grp {
            name: "H",
            scalars: handle.h_std(),
            scalar_off: 0,
            n: domain,
            general: None,
            jobs: vec![("H g1", wgsl::G1, h_bases.chunk(0))],
            chunks: vec![chunks_g1(&h_bases)],
        },
    ];

    println!(
        "{:<8} {:>5} {:>8} {:>8} {:>3} {:>3} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>9} {:>7} {:>8} {:>7} | {:>8} {:>8} {:>7}",
        "job", "piece", "n", "cap", "c", "w", "zero", "count", "scan", "scatter", "clear",
        "segment", "merge", "reduce", "ones", "ns/entry", "slab1xw", "total"
    );
    let mut sum = Phases::default();
    let mut per_job: Vec<(&str, Phases, u32)> = Vec::new();
    for g in &groups {
        let curves: Vec<wgsl::Curve> = g.jobs.iter().map(|j| j.1).collect();
        let ps = pieces(g.n, g.scalar_off, g.general, &curves, limit, chunk, g1, g2);
        for (pi, piece) in ps.iter().enumerate() {
            let dplan = &piece.dplan;
            let sort = DigitBuffers::new(&b, dplan, 0).unwrap();
            for (ji, (name, curve, _)) in g.jobs.iter().enumerate() {
                let ci = (piece.lo / chunk) as usize;
                let local = piece.lo % chunk;
                let bases = g.chunks[ji][ci];
                let ph = if curve.suffix == wgsl::G1.suffix {
                    probe_job(&b, d, g1, dplan, local, g.scalars, bases, &sort, ji == 0)
                } else {
                    probe_job(&b, d, g2, dplan, local, g.scalars, bases, &sort, ji == 0)
                };
                let entries = f64::from(dplan.n_windows()) * f64::from(dplan.cap());
                println!(
                    "{:<8} {:>5} {:>8} {:>8} {:>3} {:>3} | {:>7.2} {:>7.2} {:>7.2} {:>7.2} | {:>7.2} {:>9.2} {:>7.2} {:>8.2} {:>7.2} | {:>8.1} {:>8.1} {:>7.1}",
                    name,
                    format!("{}{pi}/{}", &g.name[..1], ps.len()),
                    dplan.n(),
                    dplan.cap(),
                    dplan.c(),
                    dplan.n_windows(),
                    ph.zero,
                    ph.count,
                    ph.scan,
                    ph.scatter,
                    ph.clear,
                    ph.segmented,
                    ph.merge,
                    ph.reduce,
                    ph.ones,
                    ph.segmented * 1e6 / entries,
                    ph.slab1_x_w,
                    ph.sort() + ph.points()
                );
                sum.add(&ph);
                match per_job.iter_mut().find(|(n, _, _)| n == name) {
                    Some(row) => {
                        row.1.add(&ph);
                        row.2 += 1;
                    }
                    None => per_job.push((name, ph, 1)),
                }
            }
        }
    }
    println!("per job, summed over its pieces (ms):");
    for (name, ph, n) in &per_job {
        println!(
            "  {name:<6} {n:>2} pieces: sort {:>8.1}  clear {:>7.1}  segmented {:>8.1}  merge {:>7.1}  reduce {:>8.1}  ones {:>6.1}  total {:>8.1}",
            ph.sort(), ph.clear, ph.segmented, ph.merge, ph.reduce, ph.ones, ph.sort() + ph.points()
        );
    }
    println!(
        "kernels summed: sort {:.1} ms (zero {:.1}, count {:.1}, scan {:.1}, scatter {:.1}), clear {:.1}, segmented {:.1}, merge {:.1}, reduce {:.1}, ones {:.1}; total {:.1} ms against {:.1} ms whole",
        sum.sort(),
        sum.zero,
        sum.count,
        sum.scan,
        sum.scatter,
        sum.clear,
        sum.segmented,
        sum.merge,
        sum.reduce,
        sum.ones,
        sum.sort() + sum.points(),
        median(whole)
    );
}

/// One curve's accumulation at every workgroup size, at 2^19 general scalars and c = 13:
/// the floor's sub-MSM at a 2^22 key, 20 windows of 4,096 slices, so 81,920 threads. The
/// sweeps in `tests/msm_g1.rs` and `tests/msm_g2.rs` run 32,768 scalars at c = 12, 5,632
/// threads, where 44 groups of 128 leave most of 38 cores idle whatever the kernel costs.
fn sweep_segmented<C: PointCurve>(
    b: &WgpuBackend,
    bbuf: &wgpu::Buffer,
    rng: &mut impl ark_std::rand::Rng,
) {
    let n = 1u32 << 19;
    let scalars: Vec<g16_field::Fr> = (0..n).map(|_| g16_field::Fr::rand(rng)).collect();
    let (words, general) = g16_wgpu::msm::pack_scalars(&scalars);
    let sbuf = storage(b, "scalars", bytemuck::cast_slice(&words));
    let dplan = DigitPlan::new(n, 0, Some(general)).unwrap();
    let d = g16_wgpu::MsmDigits::new(b).unwrap();
    let sort = DigitBuffers::new(b, &dplan, 0).unwrap();
    let sizes = [32u32, 64, 128, 256];
    let modules: Vec<MsmPoints<C>> = sizes
        .iter()
        .map(|&sz| {
            let mut wg = C::WGSL.wg;
            wg.segmented = sz;
            MsmPoints::with_shape(b, wg, 65535).unwrap()
        })
        .collect();
    let mut samples = vec![Vec::new(); sizes.len()];
    for round in 0..5 {
        for (k, p) in modules.iter().enumerate() {
            let pplan = p.plan_points(&dplan, 0).unwrap();
            let pts = PointBuffers::new(b, &dplan, &pplan, 0, p.curve()).unwrap();
            let slots = d.sort_slots(&dplan) + p.slots(&dplan, &pplan);
            let mut ring = ParamRing::new(b, "sweep ring", slots).unwrap();
            let soff = d.plan_sort(&dplan, &mut ring).unwrap();
            let poff = p.plan(&dplan, &pplan, &mut ring).unwrap();
            ring.flush(b);
            let sbind = d.bind_sort(b, &ring, &dplan, &sbuf, &sort).unwrap();
            let pbind = p
                .bind_all(b, &ring, &dplan, &pplan, &sbuf, bbuf, &sort, &pts)
                .unwrap();
            if round == 0 && k == 0 {
                submit_sealed(b, "sort", |pass| {
                    d.encode_sort(pass, &dplan, &sbind, &soff).unwrap()
                });
            }
            submit_sealed(b, "clear", |pass| {
                p.encode_clear(pass, &dplan, &pbind, &poff).unwrap()
            });
            samples[k].push(timed(b, "segmented", |pass| {
                p.encode_segmented(pass, &pplan, &pbind, &poff).unwrap()
            }));
        }
    }
    println!(
        "{} at {n} scalars ({general} general), c = {}, {} windows, medians of 5",
        C::WGSL.entry_segmented(),
        dplan.c(),
        dplan.n_windows()
    );
    for (k, sz) in sizes.iter().enumerate() {
        let ms = median(samples[k].clone());
        println!(
            "  workgroup {sz:>3}: {ms:8.2} ms, {:.1} ns per entry",
            ms * 1e6 / (f64::from(dplan.n_windows()) * f64::from(dplan.cap()))
        );
    }
}

#[test]
#[ignore = "BUG-30's sweep: prints a table, asserts nothing"]
fn the_segmented_workgroup_at_scale() {
    let _lock = gpulock::exclusive_gpu();
    let profile = LimitsProfile::from_env().expect("G16_WGPU_LIMITS");
    let b = pollster::block_on(WgpuBackend::with_profile(profile)).expect("device");
    let mut rng = test_rng();
    let n = 1usize << 19;
    let g1 = walk::<G1Projective>(n, &mut rng);
    let b1 = storage(&b, "g1 bases", as_bytes(&PackedG1Affine::pack_slice(&g1)));
    sweep_segmented::<g16_wgpu::G1Curve>(&b, &b1, &mut rng);
    let g2 = walk::<G2Projective>(n, &mut rng);
    let b2 = storage(&b, "g2 bases", as_bytes(&PackedG2Affine::pack_slice(&g2)));
    sweep_segmented::<g16_wgpu::G2Curve>(&b, &b2, &mut rng);
}
