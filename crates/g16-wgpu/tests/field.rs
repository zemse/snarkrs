//! The generated field prelude, run on a real GPU and checked against `ark-ff` bit for bit.
//!
//! # Why this runs on the device and not on a WGSL interpreter
//!
//! Because the failure this catches is a codegen failure. The arithmetic in
//! `src/gen/field.rs` was lifted from a generator the limb-width sweep already
//! validated against `num-bigint`, so the interesting question is not "is CIOS right", it is
//! "does naga lower 300 lines of straight-line 32-bit carry chains to something that computes
//! what the source says". That question has no answer short of running it.
//!
//! # Two things this harness does deliberately
//!
//! **The device is requested with `wgpu::Limits::default()`, not `adapter.limits()`.** Those
//! are the WebGPU spec floor: 128 MiB storage binding, 16 KiB workgroup storage, 8 storage
//! buffers per pipeline layout. A device created without `required_limits` silently gets the
//! floor in a browser even on an adapter offering 4 GiB, and research file 01 records heliax
//! shipping a kernel that only fit native limits twice.
//!
//! **`InstanceFlags::STRICT_WEBGPU_COMPLIANCE` is set in code, not left to the environment.**
//! It is the same switch as `WGPU_STRICT_WEBGPU_COMPLIANCE=1`, and setting it here means a
//! run that forgot the variable still fails on a Metal-only construct rather than passing and
//! failing in Chrome at U13. `SHADER_INT64` is true on native Metal through wgpu and does not
//! exist in WebGPU, so this is a live trap and not a hypothetical one.

use std::collections::HashMap;
use std::sync::OnceLock;

use ark_ff::{Field as _, PrimeField, Zero as _};
use num_bigint::BigUint;
use num_traits::One as _;

use g16_field::{Fq, Fq2, Fr};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::{PackedFq, PackedFq2, PackedFr, PackedScalar, FQ_MODULUS, FR_MODULUS, LIMBS};
use g16_wgpu::gen::{field_module, Variant};

// ---------------------------------------------------------------------------
// The kernels under test
// ---------------------------------------------------------------------------

/// One entry point: how wide an operand is, how wide a result is, and the call.
///
/// `WORDS` is in `u32`s: 8 for an `Fr` or `Fq`, 16 for an `Fq2`, 1 for a predicate. Binary
/// and unary share one shader and one bind group layout, and a unary entry point simply
/// never reads `IN_B`; an explicit pipeline layout is allowed to declare a binding the shader
/// does not use, which is what makes one bind group serve all 26 pipelines.
struct Op {
    entry: &'static str,
    ty: &'static str,
    in_words: usize,
    out_words: usize,
    expr: &'static str,
}

const OPS: &[Op] = &[
    Op {
        entry: "k_fr_mul",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_mul(a, b)",
    },
    Op {
        entry: "k_fr_add",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_add(a, b)",
    },
    Op {
        entry: "k_fr_sub",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_sub(a, b)",
    },
    Op {
        entry: "k_fr_neg",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_neg(a)",
    },
    Op {
        entry: "k_fr_sqr",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_sqr(a)",
    },
    Op {
        entry: "k_fr_from_mont",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_from_mont(a)",
    },
    Op {
        entry: "k_fr_to_mont",
        ty: "Fr",
        in_words: 8,
        out_words: 8,
        expr: "fr_to_mont(a)",
    },
    Op {
        entry: "k_fr_is_zero",
        ty: "Fr",
        in_words: 8,
        out_words: 1,
        expr: "fr_is_zero(a)",
    },
    Op {
        entry: "k_fr_eq",
        ty: "Fr",
        in_words: 8,
        out_words: 1,
        expr: "fr_eq(a, b)",
    },
    Op {
        entry: "k_fq_mul",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_mul(a, b)",
    },
    Op {
        entry: "k_fq_add",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_add(a, b)",
    },
    Op {
        entry: "k_fq_sub",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_sub(a, b)",
    },
    Op {
        entry: "k_fq_neg",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_neg(a)",
    },
    Op {
        entry: "k_fq_sqr",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_sqr(a)",
    },
    Op {
        entry: "k_fq_from_mont",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_from_mont(a)",
    },
    Op {
        entry: "k_fq_to_mont",
        ty: "Fq",
        in_words: 8,
        out_words: 8,
        expr: "fq_to_mont(a)",
    },
    Op {
        entry: "k_fq_is_zero",
        ty: "Fq",
        in_words: 8,
        out_words: 1,
        expr: "fq_is_zero(a)",
    },
    Op {
        entry: "k_fq_eq",
        ty: "Fq",
        in_words: 8,
        out_words: 1,
        expr: "fq_eq(a, b)",
    },
    Op {
        entry: "k_fq2_mul",
        ty: "Fq2",
        in_words: 16,
        out_words: 16,
        expr: "fq2_mul(a, b)",
    },
    Op {
        entry: "k_fq2_add",
        ty: "Fq2",
        in_words: 16,
        out_words: 16,
        expr: "fq2_add(a, b)",
    },
    Op {
        entry: "k_fq2_sub",
        ty: "Fq2",
        in_words: 16,
        out_words: 16,
        expr: "fq2_sub(a, b)",
    },
    Op {
        entry: "k_fq2_neg",
        ty: "Fq2",
        in_words: 16,
        out_words: 16,
        expr: "fq2_neg(a)",
    },
    Op {
        entry: "k_fq2_sqr",
        ty: "Fq2",
        in_words: 16,
        out_words: 16,
        expr: "fq2_sqr(a)",
    },
    Op {
        entry: "k_fq2_is_zero",
        ty: "Fq2",
        in_words: 16,
        out_words: 1,
        expr: "fq2_is_zero(a)",
    },
    Op {
        entry: "k_fq2_eq",
        ty: "Fq2",
        in_words: 16,
        out_words: 1,
        expr: "fq2_eq(a, b)",
    },
];

/// Load and store are unrolled too, for the same reason the arithmetic is: a `var` of array
/// type indexed by a loop variable is the thing measured at 3.8x slower, and
/// there is no reason for a test kernel to be shaped differently from a real one.
fn load(name: &str, buf: &str, ty: &str, words: usize) -> String {
    let mut s = format!("    var {name}: {ty};\n");
    if ty == "Fq2" {
        for k in 0..8 {
            s += &format!("    {name}.c0[{k}] = {buf}[base + {k}u];\n");
            s += &format!("    {name}.c1[{k}] = {buf}[base + {}u];\n", k + 8);
        }
    } else {
        for k in 0..words {
            s += &format!("    {name}[{k}] = {buf}[base + {k}u];\n");
        }
    }
    s
}

fn kernel(op: &Op) -> String {
    let Op {
        entry,
        ty,
        in_words,
        out_words,
        expr,
    } = op;
    let mut s = format!(
        "\n@compute @workgroup_size(64)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \x20   let i = gid.x;\n    if (i >= P.x) {{ return; }}\n    let base = i * {in_words}u;\n"
    );
    s += &load("a", "IN_A", ty, *in_words);
    s += &load("b", "IN_B", ty, *in_words);
    if *out_words == 1 {
        s += &format!("    OUT[i] = select(0u, 1u, {expr});\n");
    } else {
        s += &format!("    var r: {ty} = {expr};\n    let ob = i * {out_words}u;\n");
        if *ty == "Fq2" {
            for k in 0..8 {
                s += &format!("    OUT[ob + {k}u] = r.c0[{k}];\n");
                s += &format!("    OUT[ob + {}u] = r.c1[{k}];\n", k + 8);
            }
        } else {
            for k in 0..*out_words {
                s += &format!("    OUT[ob + {k}u] = r[{k}];\n");
            }
        }
    }
    s + "}\n"
}

const BINDINGS: &str = r#"
@group(0) @binding(0) var<storage, read> IN_A: array<u32>;
@group(0) @binding(1) var<storage, read> IN_B: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
"#;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| pollster::block_on(init()))
}

async fn init() -> Gpu {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.flags |= wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no wgpu adapter");
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("g16-wgpu field test"),
            required_features: wgpu::Features::empty(),
            // The browser floor, on purpose. See the module docs.
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("no wgpu device at Limits::default()");

    let mut src = field_module(Variant::Cios32Unrolled);
    src.push_str(BINDINGS);
    for op in OPS {
        src.push_str(&kernel(op));
    }

    let t0 = std::time::Instant::now();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field"),
        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
    });

    let entry = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, true),
            entry(1, true),
            entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let mut pipelines = HashMap::new();
    for op in OPS {
        pipelines.insert(
            op.entry,
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(op.entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(op.entry),
                compilation_options: Default::default(),
                cache: None,
            }),
        );
    }
    // Tracked from the first commit rather than discovered later: research file 03 records
    // 129 s of pipeline creation for one monolithic shader with an unrolled multiply inlined
    // at a dozen sites, and the MSM kernels will have exactly that shape.
    println!(
        "adapter: {} ({:?}, {:?})\nshader: {:.1} KiB, {} entry points, module + {} pipelines in {:.0} ms",
        info.name,
        info.backend,
        info.device_type,
        src.len() as f64 / 1024.0,
        OPS.len(),
        OPS.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    Gpu {
        device,
        queue,
        layout,
        pipelines,
    }
}

impl Gpu {
    /// Run one entry point over `count` elements and read the result back.
    fn run(&self, entry: &str, a: &[u32], b: &[u32], out_words: usize, count: usize) -> Vec<u32> {
        use wgpu::util::DeviceExt as _;
        let dev = &self.device;
        let mk = |data: &[u32]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let buf_a = mk(a);
        let buf_b = mk(b);
        let out_bytes = (out_words * count * 4) as u64;
        let buf_o = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_p = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[count as u32, 0u32, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_o.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[entry]);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&buf_o, 0, &staging, 0, out_bytes);
        self.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

// ---------------------------------------------------------------------------
// Test vectors
// ---------------------------------------------------------------------------

fn big(v: &[u32; LIMBS]) -> BigUint {
    BigUint::from_slice(v)
}

fn limbs(x: &BigUint) -> [u32; LIMBS] {
    let mut out = [0u32; LIMBS];
    for (i, d) in x.iter_u32_digits().enumerate() {
        assert!(i < LIMBS, "value does not fit in 256 bits");
        out[i] = d;
    }
    out
}

/// Montgomery representatives that are worth trying by hand, all canonical (`< m`).
///
/// The brief asks for 0, 1, `R-1` and `m-1`. Two of those need a word about what they mean.
/// `1` here is the *representative* 1, which is the field element `R^-1`, and the field
/// element 1 is a separate entry whose representative is `R mod m`. `R-1` is `2^256 - 1`,
/// which is not a canonical representative at all (it exceeds `m`), so what is tested is
/// `(R - 1) mod m`, the largest value congruent to it; the all-ones limb pattern itself is
/// outside the contract of every routine here and feeding it in would test nothing.
fn edge_reps(m: &BigUint) -> Vec<[u32; LIMBS]> {
    let one = BigUint::one();
    let r = BigUint::one() << 256;
    let r_mod = &r % m;
    vec![
        limbs(&BigUint::from(0u32)),
        limbs(&one),
        limbs(&(&one + &one)),
        limbs(&(m - &one)), // m-1, the largest legal representative
        limbs(&(m - &one - &one)),
        limbs(&r_mod), // the field element 1
        limbs(&(&r_mod - &one)),
        limbs(&((&r - &one) % m)), // "R-1", reduced so it is representable
        limbs(&(m / 2u32)),
        limbs(&(m >> 128)),          // only the low half populated
        limbs(&(m - (&one << 200))), // a long run of set bits, still below m
    ]
}

fn fr_of(v: [u32; LIMBS]) -> Fr {
    PackedFr { v }.to_fr()
}
fn fq_of(v: [u32; LIMBS]) -> Fq {
    PackedFq { v }.to_fq()
}

fn fr_pairs(n: usize) -> (Vec<Fr>, Vec<Fr>) {
    let m = big(&FR_MODULUS);
    let edges: Vec<Fr> = edge_reps(&m).into_iter().map(fr_of).collect();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for x in &edges {
        for y in &edges {
            a.push(*x);
            b.push(*y);
        }
    }
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    while a.len() < n + edges.len() * edges.len() {
        a.push(rng.next_fr());
        b.push(rng.next_fr());
    }
    (a, b)
}

/// `seed` is a parameter and not a constant because this is called twice to build the `Fq2`
/// inputs, and with a fixed seed both calls return the same vectors: `a == b` for every pair,
/// `fq2_sub` is only ever asked for zero, and `fq2_mul` collapses into `fq2_sqr` because
/// Karatsuba's cross term `(a0+a1)(b0+b1) - a0 b0 - a1 b1` is `2 a0 a1` when `b == a`. An
/// operand swap inside `fq2_mul` was measured passing this whole file with the seed shared.
fn fq_pairs(n: usize, seed: u64) -> (Vec<Fq>, Vec<Fq>) {
    let m = big(&FQ_MODULUS);
    let edges: Vec<Fq> = edge_reps(&m).into_iter().map(fq_of).collect();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for x in &edges {
        for y in &edges {
            a.push(*x);
            b.push(*y);
        }
    }
    let mut rng = SplitMix64(seed);
    while a.len() < n + edges.len() * edges.len() {
        let mut bytes = [0u8; 32];
        for c in bytes.chunks_mut(8) {
            c.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        a.push(Fq::from_le_bytes_mod_order(&bytes));
        for c in bytes.chunks_mut(8) {
            c.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        b.push(Fq::from_le_bytes_mod_order(&bytes));
    }
    (a, b)
}

fn flat_fr(xs: &[Fr]) -> Vec<u32> {
    xs.iter().flat_map(|x| PackedFr::from_fr(x).v).collect()
}
fn flat_fq(xs: &[Fq]) -> Vec<u32> {
    xs.iter().flat_map(|x| PackedFq::from_fq(x).v).collect()
}
fn flat_fq2(xs: &[Fq2]) -> Vec<u32> {
    xs.iter()
        .flat_map(|x| {
            let p = PackedFq2::from_fq2(x);
            p.c0.v.into_iter().chain(p.c1.v)
        })
        .collect()
}

/// Compare a device result against a host reference, and say exactly which element and which
/// limb disagreed. A bare `assert_eq!` on two 100k-word vectors is unreadable.
fn expect(entry: &str, got: &[u32], want: &[u32], words: usize) {
    for (i, (g, w)) in got.chunks(words).zip(want.chunks(words)).enumerate() {
        assert_eq!(
            g, w,
            "{entry}: element {i} disagrees with ark-ff\n  gpu:  {g:08x?}\n  host: {w:08x?}"
        );
    }
    assert_eq!(got.len(), want.len(), "{entry}: length");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 4096 random pairs plus every ordered pair of the edge representatives, through every `Fr`
/// entry point, against `ark-ff`.
#[test]
fn fr_matches_ark_on_random_and_edge_inputs() {
    let g = gpu();
    let (a, b) = fr_pairs(4096);
    let n = a.len();
    let fa = flat_fr(&a);
    let fb = flat_fr(&b);
    println!("fr: {n} pairs ({} of them edge cases)", n - 4096);

    let cases: Vec<(&str, Vec<u32>)> = vec![
        (
            "k_fr_mul",
            flat_fr(&a.iter().zip(&b).map(|(x, y)| *x * y).collect::<Vec<_>>()),
        ),
        (
            "k_fr_add",
            flat_fr(&a.iter().zip(&b).map(|(x, y)| *x + y).collect::<Vec<_>>()),
        ),
        (
            "k_fr_sub",
            flat_fr(&a.iter().zip(&b).map(|(x, y)| *x - y).collect::<Vec<_>>()),
        ),
        (
            "k_fr_neg",
            flat_fr(&a.iter().map(|x| -*x).collect::<Vec<_>>()),
        ),
        (
            "k_fr_sqr",
            flat_fr(&a.iter().map(|x| x.square()).collect::<Vec<_>>()),
        ),
        // from_mont takes the Montgomery limbs to the standard-form limbs, which is exactly
        // what PackedScalar is, and to_mont is its inverse. Between them they are the
        // conversion stage 8 will run on the device instead of on a single browser thread.
        (
            "k_fr_from_mont",
            a.iter().flat_map(|x| PackedScalar::from_fr(x).v).collect(),
        ),
    ];
    for (entry, want) in cases {
        let got = g.run(entry, &fa, &fb, 8, n);
        expect(entry, &got, &want, 8);
    }

    // to_mont reads standard-form limbs and writes Montgomery ones, so it is fed the output
    // of from_mont and must return the input.
    let std_form: Vec<u32> = a.iter().flat_map(|x| PackedScalar::from_fr(x).v).collect();
    let got = g.run("k_fr_to_mont", &std_form, &fb, 8, n);
    expect("k_fr_to_mont", &got, &fa, 8);

    let got = g.run("k_fr_is_zero", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().map(|x| u32::from(x.is_zero())).collect();
    expect("k_fr_is_zero", &got, &want, 1);

    let got = g.run("k_fr_eq", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| u32::from(x == y)).collect();
    expect("k_fr_eq", &got, &want, 1);
}

/// The same over `Fq`. Not a formality: `Fq` is a different modulus and a different `n0`, and
/// the whole claim of the generator is that one emitter serves both.
#[test]
fn fq_matches_ark_on_random_and_edge_inputs() {
    let g = gpu();
    let (a, b) = fq_pairs(4096, 0xBF58_476D_1CE4_E5B9);
    let n = a.len();
    let fa = flat_fq(&a);
    let fb = flat_fq(&b);
    println!("fq: {n} pairs ({} of them edge cases)", n - 4096);

    let cases: Vec<(&str, Vec<u32>)> = vec![
        (
            "k_fq_mul",
            flat_fq(&a.iter().zip(&b).map(|(x, y)| *x * y).collect::<Vec<_>>()),
        ),
        (
            "k_fq_add",
            flat_fq(&a.iter().zip(&b).map(|(x, y)| *x + y).collect::<Vec<_>>()),
        ),
        (
            "k_fq_sub",
            flat_fq(&a.iter().zip(&b).map(|(x, y)| *x - y).collect::<Vec<_>>()),
        ),
        (
            "k_fq_neg",
            flat_fq(&a.iter().map(|x| -*x).collect::<Vec<_>>()),
        ),
        (
            "k_fq_sqr",
            flat_fq(&a.iter().map(|x| x.square()).collect::<Vec<_>>()),
        ),
        (
            "k_fq_from_mont",
            a.iter()
                .flat_map(|x| {
                    let mut v = [0u32; LIMBS];
                    for (i, w) in x.into_bigint().0.iter().enumerate() {
                        v[2 * i] = *w as u32;
                        v[2 * i + 1] = (*w >> 32) as u32;
                    }
                    v
                })
                .collect(),
        ),
    ];
    for (entry, want) in cases {
        let got = g.run(entry, &fa, &fb, 8, n);
        expect(entry, &got, &want, 8);
    }

    let got = g.run("k_fq_is_zero", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().map(|x| u32::from(x.is_zero())).collect();
    expect("k_fq_is_zero", &got, &want, 1);

    let got = g.run("k_fq_eq", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| u32::from(x == y)).collect();
    expect("k_fq_eq", &got, &want, 1);
}

/// `Fq2 = Fq[u]/(u^2 + 1)`, where G2 lives. The multiply is Karatsuba and the squaring is
/// `(a0+a1)(a0-a1) + 2 a0 a1 u`, so neither is a transcription of the definition and both can
/// be wrong in ways `fq_mul` being right does not catch.
#[test]
fn fq2_matches_ark_on_random_and_edge_inputs() {
    let g = gpu();
    // Two seeds, not one. The edge block at the head of each vector is still the same 121
    // ordered pairs either way, so the a == b cases are not lost, only stopped from being the
    // only cases.
    let (c0a, c1a) = fq_pairs(2048, 0xBF58_476D_1CE4_E5B9);
    let (c0b, c1b) = fq_pairs(2048, 0x94D0_49BB_1331_11EB);
    let n = c0a.len();
    let a: Vec<Fq2> = c0a
        .iter()
        .zip(&c1a)
        .map(|(x, y)| Fq2::new(*x, *y))
        .collect();
    let b: Vec<Fq2> = c0b
        .iter()
        .zip(&c1b)
        .map(|(x, y)| Fq2::new(*x, *y))
        .collect();
    let fa = flat_fq2(&a);
    let fb = flat_fq2(&b);
    println!("fq2: {n} pairs");

    let cases: Vec<(&str, Vec<u32>)> = vec![
        (
            "k_fq2_mul",
            flat_fq2(&a.iter().zip(&b).map(|(x, y)| *x * y).collect::<Vec<_>>()),
        ),
        (
            "k_fq2_add",
            flat_fq2(&a.iter().zip(&b).map(|(x, y)| *x + y).collect::<Vec<_>>()),
        ),
        (
            "k_fq2_sub",
            flat_fq2(&a.iter().zip(&b).map(|(x, y)| *x - y).collect::<Vec<_>>()),
        ),
        (
            "k_fq2_neg",
            flat_fq2(&a.iter().map(|x| -*x).collect::<Vec<_>>()),
        ),
        (
            "k_fq2_sqr",
            flat_fq2(&a.iter().map(|x| x.square()).collect::<Vec<_>>()),
        ),
    ];
    for (entry, want) in cases {
        let got = g.run(entry, &fa, &fb, 16, n);
        expect(entry, &got, &want, 16);
    }

    let got = g.run("k_fq2_is_zero", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().map(|x| u32::from(x.is_zero())).collect();
    expect("k_fq2_is_zero", &got, &want, 1);

    let got = g.run("k_fq2_eq", &fa, &fb, 1, n);
    let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| u32::from(x == y)).collect();
    expect("k_fq2_eq", &got, &want, 1);
}

/// The `>` versus `>=` trap in a conditional reduction, tested where it is actually reachable.
///
/// Research file 02 §5 flags ICME's `conditional_reduce`, which uses a `bigint_gt` that
/// returns 0 on equality (`bigint.template.wgsl:38-48`), and predicts that "a Montgomery
/// product whose raw result is exactly `p` is stored as `p` rather than `0`". That prediction
/// is wrong, and the brief asks for the exactly-`m` product case, so here is what actually
/// holds.
///
/// **A Montgomery product can never come out exactly `m`.** CIOS produces the unique
/// `T` in `[0, 2m)` with `T = a*b*R^-1 (mod m)`. `T = m` forces `a*b = 0 (mod m)`; `m` is
/// prime, so `a = 0` or `b = 0` as reduced representatives, so `a*b = 0` exactly and the whole
/// accumulator is zero, giving `T = 0`, not `m`. The case is unreachable and is not covered
/// here because it does not exist. This test asserts that too, over the random sample.
///
/// The trap is real one step away, in two places that *are* reachable:
///
/// * the tightest product above the modulus, `T = m + 1`, constructed below by choosing
///   `b = R * a^-1 mod m` (which forces the residue to 1) and walking `a` until the raw `T`
///   lands in the upper half. A `>` test returns `m + 1`, which is not even a canonical
///   representative.
/// * an addition whose sum is exactly `m`, which is trivially constructible as `1 + (m-1)` and
///   is where ICME's shared `conditional_reduce` would really break, since `field_add` calls
///   it too.
#[test]
fn the_conditional_subtraction_triggers_at_the_boundary() {
    let g = gpu();
    let m = big(&FR_MODULUS);
    let r = BigUint::one() << 256;
    // -m^{-1} mod 2^256, the full-width REDC multiplier. CIOS computes the same T word by
    // word; full-width REDC is just the cheapest way to predict it exactly on the host.
    let m_inv = mod_inverse_pow2(&m, 256);
    let n_prime = (&r - m_inv) % &r;
    let redc = |a: &BigUint, b: &BigUint| -> BigUint {
        let x = a * b;
        let q = ((&x % &r) * &n_prime) % &r;
        (&x + &q * &m) / &r
    };

    // Walk `a` until the raw product lands at exactly m + 1.
    let mut found = None;
    for k in 2u32..10_000 {
        let a = BigUint::from(k);
        let b = (&r * a.modpow(&(&m - 2u32), &m)) % &m;
        let t = redc(&a, &b);
        if t == &m + BigUint::one() {
            found = Some((a, b));
            break;
        }
    }
    let (a, b) = found.expect("no pair with raw Montgomery product m+1 in the first 10k tries");
    println!("raw product exactly m+1 at a = {a}, b = {b}");

    let fa = limbs(&a);
    let fb = limbs(&b);
    let got = g.run("k_fr_mul", &fa, &fb, 8, 1);
    // The reduced answer is 1: the residue was chosen to be 1 and the conditional subtraction
    // must fire. A `>` test would return m + 1 here.
    let mut want = [0u32; LIMBS];
    want[0] = 1;
    assert_eq!(got, want, "raw product m+1 was not reduced");

    // The same boundary in the addition, where the raw value really is exactly m.
    let one = {
        let mut v = [0u32; LIMBS];
        v[0] = 1;
        v
    };
    let m_minus_1 = limbs(&(&m - BigUint::one()));
    let got = g.run("k_fr_add", &one, &m_minus_1, 8, 1);
    assert_eq!(got, [0u32; LIMBS], "1 + (m-1) did not reduce to 0");

    // And the claim that exactly-m is unreachable, checked rather than asserted from theory.
    let mut rng = SplitMix64(0xDEAD_BEEF_0000_0001);
    for _ in 0..20_000 {
        let x = big(&PackedFr::from_fr(&rng.next_fr()).v);
        let y = big(&PackedFr::from_fr(&rng.next_fr()).v);
        assert_ne!(
            redc(&x, &y),
            m,
            "a raw Montgomery product came out exactly m"
        );
    }
}

/// `x^-1 mod 2^bits` for odd `x`, by Newton iteration. Only used to predict the raw CIOS
/// output on the host in the boundary test above.
fn mod_inverse_pow2(x: &BigUint, bits: u32) -> BigUint {
    let modulus = BigUint::one() << bits;
    let x = x % &modulus;
    let mut inv = BigUint::one();
    for _ in 0..bits {
        let t = (&x * &inv) % &modulus;
        inv = (&inv * ((&modulus + BigUint::from(2u32) - t) % &modulus)) % &modulus;
    }
    inv
}

/// Every operand the field layer will ever see is a canonical representative, and every result
/// it produces has to be one too, or a later `fr_eq` silently says two equal elements differ.
/// Checked directly rather than inferred from the ark comparison, because ark would agree with
/// a non-canonical result that happened to be congruent.
#[test]
fn every_result_is_a_canonical_representative() {
    let g = gpu();
    let (a, b) = fr_pairs(4096);
    let n = a.len();
    let fa = flat_fr(&a);
    let fb = flat_fr(&b);
    let m = big(&FR_MODULUS);
    for entry in ["k_fr_mul", "k_fr_add", "k_fr_sub", "k_fr_neg", "k_fr_sqr"] {
        let got = g.run(entry, &fa, &fb, 8, n);
        for (i, w) in got.chunks(8).enumerate() {
            let mut v = [0u32; LIMBS];
            v.copy_from_slice(w);
            assert!(big(&v) < m, "{entry}: element {i} is not reduced: {v:08x?}");
        }
    }
}
