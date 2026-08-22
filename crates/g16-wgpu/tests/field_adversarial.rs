//! Adversarial checks on the generated field prelude, written against `tests/field.rs` rather
//! than alongside it.
//!
//! `tests/field.rs` checks 4096 random pairs plus 121 ordered edge pairs against `ark-ff`. That
//! is a good test and it passes. This file exists because three things it cannot reach are
//! exactly the three places a Montgomery field layer goes wrong quietly:
//!
//! 1. **`mul64` at its true maximum.** The CIOS emitter used to justify its carry word with
//!    "`p.y <= 2^32 - 2^17` leaves room for the two carry bits". Neither half of that is true:
//!    `mul64(0xffffffff, 0xffffffff).y` is `0xfffffffe`, `2^17 - 2` larger than the stated
//!    bound, and the carry out reaches 2, not 1. The conclusion survives on the joint bound
//!    instead (see [`the_cios_carry_word_never_wraps_and_has_exactly_zero_headroom`] and the
//!    rewritten comment in `src/gen/field.rs`), but a wrong number in a load-bearing bound is
//!    how the next person rearranges `mul64` and breaks it.
//!
//! 2. **Limbs that are actually all ones.** A canonical BN254 element may have limbs 0 to 6 all
//!    `0xffffffff`, because the modulus' top limb is `0x30644e72` and anything below it in the
//!    top word is legal regardless of the rest. Neither modulus' own limbs look like that, and
//!    the chance that 4096 uniform draws produce one is `2^-224`. So the worst input `mul64` and
//!    the CIOS carry chain will ever see is not in the existing test set at all.
//!
//! 3. **The constants that carry the `R` factor.** `fr_one()` returns `FR_R`, not
//!    `FR_STD_ONE`. `g16-gpu-layout`'s module docs are blunt that mixing the two produces a
//!    proof wrong by a factor of `R` that fails verification with nothing else to go on.
//!    `fr_one`, `fq_one`, `fq2_one` and `fq_to_mont` are emitted, are never called by
//!    `tests/field.rs`, and would pass every test there while being wrong.
//!
//! 4. **`Fq2` with two different operands.** `tests/field.rs` builds its `Fq2` inputs from two
//!    calls to `fq_pairs(2048)`, and `fq_pairs` is a pure function of `n` seeded with a fixed
//!    constant, so the two calls return the same vectors and `a == b` for all 2169 elements.
//!    `fq2_sub` is therefore only ever asked for zero, `fq2_eq` only ever for true, and
//!    `fq2_mul` degenerates into `fq2_sqr`: Karatsuba's cross term
//!    `(a0+a1)(b0+b1) - a0 b0 - a1 b1` collapses to `2 a0 a1` when `b == a`, so swapping a
//!    `b.c1` for an `a.c1` anywhere in it is invisible. [`fq2_is_asymmetric_and_still_right`]
//!    feeds two independent streams.
//!
//! Device at `wgpu::Limits::default()` with `STRICT_WEBGPU_COMPLIANCE`, same as `tests/field.rs`
//! and for the same reason.

use std::collections::HashMap;
use std::sync::OnceLock;

use ark_ff::{Field as _, PrimeField};
use num_bigint::BigUint;
use num_traits::One as _;

use g16_field::{Fq, Fq2, Fr};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::{PackedFq, PackedFq2, PackedFr, FQ_MODULUS, FR_MODULUS, LIMBS};
use g16_wgpu::gen::{field_module, Variant};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

const BINDINGS: &str = r#"
@group(0) @binding(0) var<storage, read> IN_A: array<u32>;
@group(0) @binding(1) var<storage, read> IN_B: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
"#;

/// `mul64` on its own, one 32x32 product per thread, so the pair `(0xffffffff, 0xffffffff)`
/// can be fed directly instead of hoping a field element contains it.
const K_MUL64: &str = r#"
@compute @workgroup_size(64)
fn k_mul64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.x) { return; }
    let p = mul64(IN_A[i], IN_B[i]);
    OUT[2u * i] = p.x;
    OUT[2u * i + 1u] = p.y;
}
"#;

/// Every emitted nullary constant, written once by thread 0. 48 words:
/// `fr_one`, `fq_one`, `fq2_one` (c0 then c1), `fr_zero`, `fq_zero`.
const K_CONSTS: &str = r#"
@compute @workgroup_size(64)
fn k_consts(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let a = fr_one();
    let b = fq_one();
    let c = fq2_one();
    let d = fr_zero();
    let e = fq_zero();
    for (var k = 0u; k < 8u; k = k + 1u) {
        OUT[k] = a[k];
        OUT[8u + k] = b[k];
        OUT[16u + k] = c.c0[k];
        OUT[24u + k] = c.c1[k];
        OUT[32u + k] = d[k];
        OUT[40u + k] = e[k];
    }
}
"#;

/// One binary or unary entry point over 8-word or 16-word operands.
struct Op {
    entry: &'static str,
    ty: &'static str,
    words: usize,
    expr: &'static str,
}

const OPS: &[Op] = &[
    Op {
        entry: "k_fr_mul",
        ty: "Fr",
        words: 8,
        expr: "fr_mul(a, b)",
    },
    Op {
        entry: "k_fr_add",
        ty: "Fr",
        words: 8,
        expr: "fr_add(a, b)",
    },
    Op {
        entry: "k_fr_sub",
        ty: "Fr",
        words: 8,
        expr: "fr_sub(a, b)",
    },
    Op {
        entry: "k_fr_sqr",
        ty: "Fr",
        words: 8,
        expr: "fr_sqr(a)",
    },
    Op {
        entry: "k_fr_mul_one",
        ty: "Fr",
        words: 8,
        expr: "fr_mul(a, fr_one())",
    },
    Op {
        entry: "k_fq_mul",
        ty: "Fq",
        words: 8,
        expr: "fq_mul(a, b)",
    },
    Op {
        entry: "k_fq_add",
        ty: "Fq",
        words: 8,
        expr: "fq_add(a, b)",
    },
    Op {
        entry: "k_fq_sub",
        ty: "Fq",
        words: 8,
        expr: "fq_sub(a, b)",
    },
    Op {
        entry: "k_fq_sqr",
        ty: "Fq",
        words: 8,
        expr: "fq_sqr(a)",
    },
    Op {
        entry: "k_fq_mul_one",
        ty: "Fq",
        words: 8,
        expr: "fq_mul(a, fq_one())",
    },
    // The one entry point tests/field.rs emits and then never dispatches.
    Op {
        entry: "k_fq_to_mont",
        ty: "Fq",
        words: 8,
        expr: "fq_to_mont(a)",
    },
    Op {
        entry: "k_fq_from_mont",
        ty: "Fq",
        words: 8,
        expr: "fq_from_mont(a)",
    },
    Op {
        entry: "k_fq2_mul",
        ty: "Fq2",
        words: 16,
        expr: "fq2_mul(a, b)",
    },
    Op {
        entry: "k_fq2_sqr",
        ty: "Fq2",
        words: 16,
        expr: "fq2_sqr(a)",
    },
    Op {
        entry: "k_fq2_add",
        ty: "Fq2",
        words: 16,
        expr: "fq2_add(a, b)",
    },
    Op {
        entry: "k_fq2_sub",
        ty: "Fq2",
        words: 16,
        expr: "fq2_sub(a, b)",
    },
];

fn load(name: &str, buf: &str, ty: &str) -> String {
    let mut s = format!("    var {name}: {ty};\n");
    if ty == "Fq2" {
        for k in 0..8 {
            s += &format!("    {name}.c0[{k}] = {buf}[base + {k}u];\n");
            s += &format!("    {name}.c1[{k}] = {buf}[base + {}u];\n", k + 8);
        }
    } else {
        for k in 0..8 {
            s += &format!("    {name}[{k}] = {buf}[base + {k}u];\n");
        }
    }
    s
}

fn kernel(op: &Op) -> String {
    let Op {
        entry,
        ty,
        words,
        expr,
    } = op;
    let mut s = format!(
        "\n@compute @workgroup_size(64)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \x20   let i = gid.x;\n    if (i >= P.x) {{ return; }}\n    let base = i * {words}u;\n"
    );
    s += &load("a", "IN_A", ty);
    s += &load("b", "IN_B", ty);
    s += &format!("    var r: {ty} = {expr};\n");
    if *ty == "Fq2" {
        for k in 0..8 {
            s += &format!("    OUT[base + {k}u] = r.c0[{k}];\n");
            s += &format!("    OUT[base + {}u] = r.c1[{k}];\n", k + 8);
        }
    } else {
        for k in 0..8 {
            s += &format!("    OUT[base + {k}u] = r[{k}];\n");
        }
    }
    s + "}\n"
}

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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("g16-wgpu field adversarial"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("no wgpu device at Limits::default()");

    let mut src = field_module(Variant::Cios32Unrolled);
    src.push_str(BINDINGS);
    src.push_str(K_MUL64);
    src.push_str(K_CONSTS);
    for op in OPS {
        src.push_str(&kernel(op));
    }

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field adversarial"),
        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
    });
    let sto = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
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
            sto(0, true),
            sto(1, true),
            sto(2, false),
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
    for entry in ["k_mul64", "k_consts"] {
        pipelines.insert(
            entry,
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            }),
        );
    }
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
    Gpu {
        device,
        queue,
        layout,
        pipelines,
    }
}

impl Gpu {
    fn run(
        &self,
        entry: &str,
        a: &[u32],
        b: &[u32],
        out_words_total: usize,
        count: usize,
    ) -> Vec<u32> {
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
        let out_bytes = (out_words_total * 4) as u64;
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
            pass.dispatch_workgroups(count.div_ceil(64).max(1) as u32, 1, 1);
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
// Helpers
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

/// The largest canonical representative whose low seven limbs are all `0xffffffff`.
///
/// Both BN254 moduli have top limb `0x30644e72`, so `0x30644e71` in the top word plus all ones
/// below it is strictly less than either modulus and is therefore a legal operand everywhere.
/// This is the input that drives `mul64` to its documented maximum and the CIOS carry chain to
/// its worst case, and nothing in `tests/field.rs` produces it.
fn all_ones_below_top(m: &[u32; LIMBS]) -> [u32; LIMBS] {
    let mut v = [u32::MAX; LIMBS];
    v[LIMBS - 1] = m[LIMBS - 1] - 1;
    assert!(big(&v) < big(m), "constructed operand is not canonical");
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `mul64` against the host's own 64-bit product, on the pairs that stress it rather than on
/// whatever a uniform draw happens to produce.
///
/// The set is every combination of the six 32-bit values that sit on a 16-bit boundary
/// (`0`, `1`, `0xffff`, `0x10000`, `0xffff0000`, `0xffffffff`), which is where a `>> 16` or a
/// `& 0xffff` gets it wrong, plus 4096 random pairs. It also pins the true maximum of the high
/// word, because the emitted comment states a bound that is `2^17 - 2` too small.
#[test]
fn mul64_is_exact_including_at_its_true_maximum() {
    let g = gpu();
    let corners: [u32; 8] = [
        0,
        1,
        2,
        0xffff,
        0x1_0000,
        0x1_0001,
        0xffff_0000,
        0xffff_ffff,
    ];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for x in corners {
        for y in corners {
            a.push(x);
            b.push(y);
        }
    }
    let mut rng = SplitMix64(0x5851_F42D_4C95_7F2D);
    for _ in 0..4096 {
        let w = rng.next_u64();
        a.push(w as u32);
        b.push((w >> 32) as u32);
    }
    let n = a.len();
    let got = g.run("k_mul64", &a, &b, 2 * n, n);

    let mut max_hi = 0u32;
    for i in 0..n {
        let want = a[i] as u64 * b[i] as u64;
        let have = got[2 * i] as u64 | ((got[2 * i + 1] as u64) << 32);
        assert_eq!(
            have, want,
            "mul64({:#x}, {:#x}) = {have:#x}, want {want:#x}",
            a[i], b[i]
        );
        max_hi = max_hi.max(got[2 * i + 1]);
    }

    // The claim in MUL64's comment is `p11 <= 2^32 - 2^17 + 1`, which is right, and then the
    // CIOS emitter's comment says `p.y <= 2^32 - 2^17`, which is not what follows from it.
    assert_eq!(max_hi, 0xffff_fffe, "mul64 high word maximum");
    assert!(
        max_hi > 0xffff_ffffu32 - (1 << 17),
        "the high word really does exceed 2^32 - 2^17"
    );
    println!("mul64: {n} pairs exact, high word maximum {max_hi:#010x} (= 2^32 - 2)");
}

/// The nullary constants, which carry the whole Montgomery-versus-standard distinction and
/// which `tests/field.rs` emits but never dispatches.
///
/// `fr_one()` must be `R mod m`, not `1`. If it returned `FR_STD_ONE` the multiply would still
/// be right, every existing test would still pass, and every proof would come out wrong by a
/// factor of `R` with no other symptom. `g16-gpu-layout`'s module docs say exactly that.
#[test]
fn the_field_one_is_the_montgomery_one_and_not_the_integer_one() {
    let g = gpu();
    let out = g.run("k_consts", &[0u32; 4], &[0u32; 4], 48, 1);

    let want_fr_one = PackedFr::from_fr(&Fr::ONE).v;
    let want_fq_one = PackedFq::from_fq(&Fq::ONE).v;
    assert_eq!(&out[0..8], &want_fr_one, "fr_one() is not R mod r");
    assert_eq!(&out[8..16], &want_fq_one, "fq_one() is not R mod q");
    let want_fq2_one = PackedFq2::from_fq2(&Fq2::ONE);
    assert_eq!(&out[16..24], &want_fq2_one.c0.v, "fq2_one().c0");
    assert_eq!(&out[24..32], &want_fq2_one.c1.v, "fq2_one().c1");
    assert_eq!(&out[32..40], &[0u32; 8], "fr_zero()");
    assert_eq!(&out[40..48], &[0u32; 8], "fq_zero()");

    // And the reason it matters: it has to be the multiplicative identity on the device.
    // A standard-form 1 here would return a*R^-1 instead of a.
    let mut rng = SplitMix64(0x2545_F491_4F6C_DD1D);
    let xs: Vec<Fr> = (0..512).map(|_| rng.next_fr()).collect();
    let flat: Vec<u32> = xs.iter().flat_map(|x| PackedFr::from_fr(x).v).collect();
    let got = g.run("k_fr_mul_one", &flat, &flat, flat.len(), xs.len());
    assert_eq!(got, flat, "fr_mul(a, fr_one()) is not a");

    let qs: Vec<Fq> = (0..512)
        .map(|_| {
            let mut bytes = [0u8; 32];
            for c in bytes.chunks_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            Fq::from_le_bytes_mod_order(&bytes)
        })
        .collect();
    let flat: Vec<u32> = qs.iter().flat_map(|x| PackedFq::from_fq(x).v).collect();
    let got = g.run("k_fq_mul_one", &flat, &flat, flat.len(), qs.len());
    assert_eq!(got, flat, "fq_mul(a, fq_one()) is not a");
}

/// `fq_to_mont`, the one emitted entry point `tests/field.rs` builds a pipeline for and then
/// never dispatches. Checked as the inverse of `fq_from_mont` and against `ark-ff` directly.
#[test]
fn fq_to_mont_is_the_inverse_of_fq_from_mont() {
    let g = gpu();
    let mut rng = SplitMix64(0x1428_57AB_1428_57AB);
    let xs: Vec<Fq> = (0..1024)
        .map(|_| {
            let mut bytes = [0u8; 32];
            for c in bytes.chunks_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            Fq::from_le_bytes_mod_order(&bytes)
        })
        .collect();
    let mont: Vec<u32> = xs.iter().flat_map(|x| PackedFq::from_fq(x).v).collect();
    let std_form: Vec<u32> = xs
        .iter()
        .flat_map(|x| {
            let mut v = [0u32; LIMBS];
            for (i, w) in x.into_bigint().0.iter().enumerate() {
                v[2 * i] = *w as u32;
                v[2 * i + 1] = (*w >> 32) as u32;
            }
            v
        })
        .collect();

    let got = g.run("k_fq_from_mont", &mont, &mont, mont.len(), xs.len());
    assert_eq!(got, std_form, "fq_from_mont");
    let got = g.run(
        "k_fq_to_mont",
        &std_form,
        &std_form,
        std_form.len(),
        xs.len(),
    );
    assert_eq!(got, mont, "fq_to_mont");
}

/// The operand `tests/field.rs` cannot reach: low seven limbs all `0xffffffff`.
///
/// This is a legal canonical element of both fields and it is the worst case for every carry
/// chain in the crate, because it makes `mul64(a[j], b[i])` return `0xfffffffe` in the high word
/// for 49 of the 64 limb products in a single `fr_mul`. The chance of a uniform draw producing
/// one is `2^-224`, so 4096 of them will not.
#[test]
fn the_all_ones_operand_is_legal_and_handled() {
    let g = gpu();

    for (name, modulus) in [("fr", FR_MODULUS), ("fq", FQ_MODULUS)] {
        let m = big(&modulus);
        let hi = all_ones_below_top(&modulus);
        // A family around it, all canonical: the all-ones value, its neighbours, m-1, and the
        // values that make one limb of the product straddle a 2^32 boundary.
        let mut reps: Vec<[u32; LIMBS]> = vec![
            hi,
            limbs(&(big(&hi) - BigUint::one())),
            limbs(&(&m - BigUint::one())),
            limbs(&(&m >> 1)),
            {
                let mut v = [0u32; LIMBS];
                v[0] = u32::MAX;
                v
            },
            {
                let mut v = [u32::MAX; LIMBS];
                v[LIMBS - 1] = 0;
                v
            },
        ];
        reps.retain(|v| big(v) < m);
        assert!(
            reps.len() >= 6,
            "{name}: lost a representative to the canonical filter"
        );

        let mut a = Vec::new();
        let mut b = Vec::new();
        for x in &reps {
            for y in &reps {
                a.push(*x);
                b.push(*y);
            }
        }
        let n = a.len();
        let fa: Vec<u32> = a.iter().flat_map(|v| *v).collect();
        let fb: Vec<u32> = b.iter().flat_map(|v| *v).collect();

        // Reference computed with num-bigint straight from the definitions, not with ark, so
        // this does not inherit any assumption ark and the generator might share.
        let r_inv = BigUint::from(2u32)
            .modpow(&(&m - BigUint::from(2u32)), &m)
            .modpow(&BigUint::from(256u32), &m);
        let want_mul: Vec<u32> = a
            .iter()
            .zip(&b)
            .flat_map(|(x, y)| limbs(&((big(x) * big(y) % &m) * &r_inv % &m)))
            .collect();
        let want_add: Vec<u32> = a
            .iter()
            .zip(&b)
            .flat_map(|(x, y)| limbs(&((big(x) + big(y)) % &m)))
            .collect();
        let want_sub: Vec<u32> = a
            .iter()
            .zip(&b)
            .flat_map(|(x, y)| limbs(&((big(x) + &m - big(y)) % &m)))
            .collect();
        let want_sqr: Vec<u32> = a
            .iter()
            .flat_map(|x| limbs(&((big(x) * big(x) % &m) * &r_inv % &m)))
            .collect();

        for (suffix, want) in [
            ("mul", &want_mul),
            ("add", &want_add),
            ("sub", &want_sub),
            ("sqr", &want_sqr),
        ] {
            let entry: &'static str = match (name, suffix) {
                ("fr", "mul") => "k_fr_mul",
                ("fr", "add") => "k_fr_add",
                ("fr", "sub") => "k_fr_sub",
                ("fr", "sqr") => "k_fr_sqr",
                ("fq", "mul") => "k_fq_mul",
                ("fq", "add") => "k_fq_add",
                ("fq", "sub") => "k_fq_sub",
                _ => "k_fq_sqr",
            };
            let got = g.run(entry, &fa, &fb, 8 * n, n);
            for (i, (gc, wc)) in got.chunks(8).zip(want.chunks(8)).enumerate() {
                assert_eq!(
                    gc, wc,
                    "{entry}: pair {i}\n  gpu:  {gc:08x?}\n  host: {wc:08x?}"
                );
                let mut v = [0u32; LIMBS];
                v.copy_from_slice(gc);
                assert!(big(&v) < m, "{entry}: pair {i} is not reduced");
            }
        }
        println!("{name}: {n} extreme pairs through mul/add/sub/sqr, all reduced");
    }
}

/// The `>` versus `>=` boundary in `fq_mul`, which `tests/field.rs` only tests for `Fr`.
///
/// `Fq` is a different modulus and a different `n0`, and the conditional subtraction is emitted
/// separately for it. The construction is the same: pick `b = R * a^-1 mod q` so the residue is
/// forced to 1, then walk `a` until the raw CIOS value lands at exactly `q + 1`. A `>` test
/// returns `q + 1`, which is not a canonical representative at all.
#[test]
fn the_conditional_subtraction_triggers_at_the_boundary_for_fq_too() {
    let g = gpu();
    let m = big(&FQ_MODULUS);
    let r = BigUint::one() << 256;
    let m_inv = {
        let modulus = &r;
        let x = &m % modulus;
        let mut inv = BigUint::one();
        for _ in 0..256 {
            let t = (&x * &inv) % modulus;
            inv = (&inv * ((modulus + BigUint::from(2u32) - t) % modulus)) % modulus;
        }
        inv
    };
    let n_prime = (&r - m_inv) % &r;
    let redc = |a: &BigUint, b: &BigUint| -> BigUint {
        let x = a * b;
        let q = ((&x % &r) * &n_prime) % &r;
        (&x + &q * &m) / &r
    };

    let mut found = None;
    for k in 2u32..10_000 {
        let a = BigUint::from(k);
        let b = (&r * a.modpow(&(&m - 2u32), &m)) % &m;
        if redc(&a, &b) == &m + BigUint::one() {
            found = Some((a, b));
            break;
        }
    }
    let (a, b) = found.expect("no Fq pair with raw Montgomery product q+1 in the first 10k tries");
    println!("fq: raw product exactly q+1 at a = {a}");
    let got = g.run("k_fq_mul", &limbs(&a), &limbs(&b), 8, 1);
    let mut want = [0u32; LIMBS];
    want[0] = 1;
    assert_eq!(got, want, "fq: raw product q+1 was not reduced");

    // The same boundary in fq_add, where the raw sum really is exactly q.
    let mut one = [0u32; LIMBS];
    one[0] = 1;
    let got = g.run("k_fq_add", &one, &limbs(&(&m - BigUint::one())), 8, 1);
    assert_eq!(got, [0u32; LIMBS], "fq: 1 + (q-1) did not reduce to 0");

    // fq_sub at its own boundary: 0 - 1 must add the modulus back, giving q-1.
    let got = g.run("k_fq_sub", &[0u32; LIMBS], &one, 8, 1);
    assert_eq!(
        got,
        limbs(&(&m - BigUint::one())),
        "fq: 0 - 1 did not wrap to q-1"
    );
}

/// The carry bound, checked rather than asserted.
///
/// This mirrors the emitted CIOS inner loop in `u64` on the host and watches the two quantities
/// the shader cannot afford to have wrong:
///
/// * `c = p.y + cy + cy2`, which is a `u32` in the shader. The emitted comment justifies it with
///   "`p.y <= 2^32 - 2^17` leaves room for the two carry bits", and that argument does not hold:
///   `p.y` reaches `2^32 - 2` and the carry out reaches 2. What actually holds is the joint
///   bound `a[j]*b[i] + t[j] + c <= (2^32-1)^2 + 2*(2^32-1) = 2^64 - 1`, so the pair
///   `(c_new, t_new)` is a 64-bit value and `c_new <= 2^32 - 1` by construction.
/// * the ninth accumulator word `t8` after the round shift, which the emitter drops on the floor
///   because `T < 2m < 2^256`. If it were ever nonzero the single conditional subtraction would
///   be the wrong reduction, not merely a slow one.
///
/// Run over the extreme operands above plus random pairs. It reports the observed maxima so the
/// "zero headroom" claim is a measurement and not a hope.
#[test]
fn the_cios_carry_word_never_wraps_and_has_exactly_zero_headroom() {
    let m = big(&FR_MODULUS);
    let two_m = &m * 2u32;

    let mut inputs: Vec<([u32; LIMBS], [u32; LIMBS])> = Vec::new();
    let hi = all_ones_below_top(&FR_MODULUS);
    let m1 = limbs(&(&m - BigUint::one()));
    for x in [
        hi,
        m1,
        [
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            0,
        ],
    ] {
        for y in [hi, m1, [u32::MAX, 0, 0, 0, 0, 0, 0, 0]] {
            inputs.push((x, y));
        }
    }
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    for _ in 0..20_000 {
        inputs.push((
            PackedFr::from_fr(&rng.next_fr()).v,
            PackedFr::from_fr(&rng.next_fr()).v,
        ));
    }

    let np = g16_gpu_layout::FR_N0 as u64;
    let mut max_c = 0u64;
    let mut max_carry_out = 0u64;
    let mut max_t8 = 0u64;
    let mut max_t = BigUint::from(0u32);

    for (a, b) in &inputs {
        // Ten accumulator words, exactly as `mul_cios32` emits them.
        let mut t = [0u64; 10];
        for &bi in b.iter() {
            let mut c = 0u64;
            for (j, &aj) in a.iter().enumerate() {
                let p = aj as u64 * bi as u64;
                // The shader computes t[j] + p.x with a carry, then + c with a second carry,
                // and forms c = p.y + cy + cy2. Done in u64 here so a wrap is visible instead
                // of silent.
                let s = t[j] + (p & 0xffff_ffff) + c;
                let carry_out = s >> 32;
                t[j] = s & 0xffff_ffff;
                let c_new = (p >> 32) + carry_out;
                assert!(
                    c_new <= u32::MAX as u64,
                    "CIOS carry word wrapped: p.y = {:#x}, carry out = {carry_out}, c = {c_new:#x}",
                    p >> 32
                );
                max_carry_out = max_carry_out.max(carry_out);
                max_c = max_c.max(c_new);
                c = c_new;
            }
            let s = t[8] + c;
            t[9] = s >> 32;
            t[8] = s & 0xffff_ffff;

            let mm = (t[0] * np) & 0xffff_ffff;
            let mut c = 0u64;
            for (j, &mj) in FR_MODULUS.iter().enumerate() {
                let p = mm * mj as u64;
                let s = t[j] + (p & 0xffff_ffff) + c;
                let carry_out = s >> 32;
                if j == 0 {
                    assert_eq!(s & 0xffff_ffff, 0, "the CIOS low word did not cancel");
                } else {
                    t[j - 1] = s & 0xffff_ffff;
                }
                let c_new = (p >> 32) + carry_out;
                assert!(c_new <= u32::MAX as u64, "CIOS reduce carry word wrapped");
                max_carry_out = max_carry_out.max(carry_out);
                max_c = max_c.max(c_new);
                c = c_new;
            }
            let s = t[8] + c;
            t[7] = s & 0xffff_ffff;
            t[8] = t[9] + (s >> 32);
            assert!(
                t[8] <= u32::MAX as u64,
                "the ninth accumulator word wrapped"
            );
            max_t8 = max_t8.max(t[8]);
        }

        // T < 2m is the whole reason one conditional subtraction is enough, and t8 == 0 is the
        // reason the emitter is allowed to ignore it.
        let mut v = [0u32; LIMBS];
        for (i, w) in t[..8].iter().enumerate() {
            v[i] = *w as u32;
        }
        let raw = big(&v) + (BigUint::from(t[8]) << 256);
        assert!(raw < two_m, "CIOS intermediate is not below 2m");
        assert_eq!(t[8], 0, "the ninth accumulator word is not zero");
        if raw > max_t {
            max_t.clone_from(&raw);
        }
    }

    // Zero headroom is a measurement, not a figure of speech: the carry word really does reach
    // 0xffffffff on inputs a field element can hold.
    println!(
        "cios over {} pairs: max carry word {max_c:#x}, max carry out {max_carry_out}, max t8 {max_t8}",
        inputs.len()
    );
    assert_eq!(
        max_c,
        u32::MAX as u64,
        "the carry word never reached its bound"
    );
    assert_eq!(max_carry_out, 2, "the inner carry out never reached 2");
    assert_eq!(max_t8, 0);
    assert!(max_t < two_m);
    // And the largest raw T seen is genuinely above m, so the conditional subtraction is being
    // exercised rather than being dead code the tests never trip.
    assert!(max_t > m, "no raw CIOS output above m was produced");
}

/// `Fq2` where the two operands are genuinely different, which `tests/field.rs` never does.
///
/// Its `a` and `b` come from two calls to `fq_pairs(2048)`, a pure function of `n` with a fixed
/// seed, so they are the same vector: every one of its 2169 `fq2_mul` cases is really an
/// `fq2_sqr`, every `fq2_sub` case is `a - a`, and every `fq2_eq` case is true. Karatsuba hides
/// an operand mix-up under `b == a`, because `(a0+a1)(b0+b1) - a0 b0 - a1 b1` is `2 a0 a1`
/// either way. Two independent streams here, and the reference is `num-bigint` on the
/// definition of `Fq2 = Fq[u]/(u^2 + 1)` rather than `ark-ff`, so a shared assumption cannot
/// cancel out.
#[test]
fn fq2_is_asymmetric_and_still_right() {
    let g = gpu();
    let m = big(&FQ_MODULUS);
    let r_inv = BigUint::from(2u32)
        .modpow(&(&m - BigUint::from(2u32)), &m)
        .modpow(&BigUint::from(256u32), &m);
    let mm = |x: &BigUint, y: &BigUint| (x * y % &m) * &r_inv % &m;

    // Two streams, and deliberately different seeds. 1024 pairs is plenty: this is looking for
    // an operand swap, which shows up on the first non-symmetric input.
    let mut ra = SplitMix64(0x0BAD_C0DE_0BAD_C0DE);
    let mut rb = SplitMix64(0xFEED_FACE_CAFE_BEEF);
    let draw = |rng: &mut SplitMix64| -> BigUint {
        let mut bytes = [0u8; 32];
        for c in bytes.chunks_mut(8) {
            c.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes) % &m
    };
    let n = 1024;
    let mut a: Vec<(BigUint, BigUint)> = Vec::with_capacity(n);
    let mut b: Vec<(BigUint, BigUint)> = Vec::with_capacity(n);
    for _ in 0..n {
        a.push((draw(&mut ra), draw(&mut ra)));
        b.push((draw(&mut rb), draw(&mut rb)));
    }
    // A handful of asymmetric corners on top, where c0 and c1 differ in the way that catches a
    // swapped component rather than a swapped operand.
    let zero = BigUint::from(0u32);
    let one = BigUint::one();
    a.push((one.clone(), zero.clone()));
    b.push((zero.clone(), one.clone()));
    a.push((zero.clone(), one.clone()));
    b.push((one.clone(), zero.clone()));
    a.push((&m - &one, one.clone()));
    b.push((one.clone(), &m - &one));
    let n = a.len();
    assert!(
        a.iter().zip(&b).all(|(x, y)| x != y),
        "the two streams collided, which is the bug this test exists to avoid"
    );

    let flat = |v: &[(BigUint, BigUint)]| -> Vec<u32> {
        v.iter()
            .flat_map(|(c0, c1)| limbs(c0).into_iter().chain(limbs(c1)))
            .collect()
    };
    let fa = flat(&a);
    let fb = flat(&b);

    // (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u, straight from the
    // definition, no Karatsuba, so the device's shortcut is checked against the long form.
    let want_mul: Vec<u32> = a
        .iter()
        .zip(&b)
        .flat_map(|((a0, a1), (b0, b1))| {
            let c0 = (mm(a0, b0) + &m - mm(a1, b1)) % &m;
            let c1 = (mm(a0, b1) + mm(a1, b0)) % &m;
            limbs(&c0).into_iter().chain(limbs(&c1))
        })
        .collect();
    let want_add: Vec<u32> = a
        .iter()
        .zip(&b)
        .flat_map(|((a0, a1), (b0, b1))| {
            limbs(&((a0 + b0) % &m))
                .into_iter()
                .chain(limbs(&((a1 + b1) % &m)))
        })
        .collect();
    let want_sub: Vec<u32> = a
        .iter()
        .zip(&b)
        .flat_map(|((a0, a1), (b0, b1))| {
            limbs(&((a0 + &m - b0) % &m))
                .into_iter()
                .chain(limbs(&((a1 + &m - b1) % &m)))
        })
        .collect();
    let want_sqr: Vec<u32> = a
        .iter()
        .flat_map(|(a0, a1)| {
            let c0 = (mm(a0, a0) + &m - mm(a1, a1)) % &m;
            let c1 = (mm(a0, a1) + mm(a1, a0)) % &m;
            limbs(&c0).into_iter().chain(limbs(&c1))
        })
        .collect();

    for (entry, want) in [
        ("k_fq2_mul", &want_mul),
        ("k_fq2_add", &want_add),
        ("k_fq2_sub", &want_sub),
        ("k_fq2_sqr", &want_sqr),
    ] {
        let got = g.run(entry, &fa, &fb, 16 * n, n);
        for (i, (gc, wc)) in got.chunks(16).zip(want.chunks(16)).enumerate() {
            assert_eq!(
                gc, wc,
                "{entry}: pair {i}\n  gpu:  {gc:08x?}\n  host: {wc:08x?}"
            );
        }
    }
    // fq2_sub on distinct operands must not be zero, or this test degenerates the same way the
    // one it replaces did.
    assert!(
        want_sub.chunks(16).any(|c| c.iter().any(|w| *w != 0)),
        "every fq2_sub result was zero"
    );
    println!("fq2: {n} asymmetric pairs through mul/add/sub/sqr");
}
