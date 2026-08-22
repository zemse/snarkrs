//! U4's acceptance test: the device layer, on a real GPU.
//!
//! Four things are checked, and the first two are the ones design §8 names.
//!
//! 1. Every limit requested under `Floor` is no better than the WebGPU spec default. The
//!    predicate is `Limits::check_limits_with_fail_fn`, not a hand-written `<=` over each
//!    field, because `<=` is the wrong comparison for the two `min_*_offset_alignment`
//!    limits: there a *smaller* request is the stronger demand, and a hand-rolled test would
//!    wave through a device that had been asked for 32-byte uniform alignment. wgpu carries
//!    the per-limit ordering, so use it.
//! 2. Module plus pipeline creation is under 5 seconds. Kill line for the browser at U9 is
//!    10 s (design §9 risk 2); this is the same number tracked from the first commit, on the
//!    generated field prelude rather than on a toy shader, so it measures a compiler and not
//!    a fixed overhead.
//! 3. The parameter ring actually addresses distinct blocks. Three dispatches of one kernel
//!    from one 256-byte-strided uniform buffer, one submit, and each dispatch has to have
//!    seen its own parameters. A ring that silently handed every dispatch slot zero would
//!    pass a "does it run" test and fail this one.
//! 4. The readback helper returns the bytes the GPU wrote. This is the function that has to
//!    be correct on native and in a browser from one source, so it is exercised natively here
//!    and again by U13's browser test. Deleting the `device.poll` inside it makes test 3 hang
//!    past 60 s rather than finish in 40 ms, which is how the design's claim that "device.poll
//!    cannot be the mechanism" was checked and found to be web-only.
//!
//! Native only. There is no wasm test runner in this workspace, and `pollster::block_on` is
//! confined to the native dev-dependency table, because on wasm32 its executor is
//! `thread::park` in a loop and the browser event loop never runs.

use std::sync::OnceLock;

use g16_wgpu::gen::{field_module, Variant};
use g16_wgpu::{Kernels, LimitsProfile, ParamRing, Readback, WgpuBackend};

// ---------------------------------------------------------------------------
// The kernel the ring and the readback are tested through
// ---------------------------------------------------------------------------

/// Deliberately trivial arithmetic and a non-trivial addressing pattern.
///
/// `base` is what makes the three dispatches write to disjoint regions and what proves the
/// dynamic offset landed on the right slot: if two dispatches read the same parameter block,
/// one region stays zero and the other is written twice.
const RING_WGSL: &str = r#"
struct Params {
    n: u32,
    base: u32,
    scale: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> OUT: array<u32>;

@compute @workgroup_size(64)
fn k_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    OUT[P.base + i] = P.scale * (i + 1u);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    base: u32,
    scale: u32,
    _pad: u32,
}

/// A realistic compile workload for the 5 second assertion.
///
/// Six entry points over the generated field prelude rather than a toy shader, because the
/// number being watched is a shader-compiler number and a 20-line kernel would report the
/// fixed overhead instead. The prelude alone is 58,800 bytes of straight-line 32-bit carry
/// chains for `Fr`, `Fq` and `Fq2`; `tests/field.rs` builds the 83.4 KiB, 25 entry point
/// version of the same thing.
const FIELD_ENTRIES: [&str; 6] = [
    "k_fr_mul", "k_fr_add", "k_fr_sub", "k_fq_mul", "k_fq_add", "k_fq_sub",
];

fn field_test_module() -> String {
    let mut src = field_module(Variant::Cios32Unrolled);
    src.push_str(
        "\n@group(0) @binding(0) var<storage, read> IN_A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> OUT: array<u32>;\n",
    );
    for entry in FIELD_ENTRIES {
        let ty = if entry.starts_with("k_fr") {
            "Fr"
        } else {
            "Fq"
        };
        let f = &entry[2..];
        src.push_str(&format!(
            "\n@compute @workgroup_size(64)\n\
             fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             \x20   let base = gid.x * 8u;\n\
             \x20   var a: {ty};\n\
             \x20   var b: {ty};\n\
             \x20   for (var k = 0u; k < 8u; k = k + 1u) {{ a[k] = IN_A[base + k]; b[k] = IN_A[base + 8u + k]; }}\n\
             \x20   let r = {f}(a, b);\n\
             \x20   for (var k = 0u; k < 8u; k = k + 1u) {{ OUT[base + k] = r[k]; }}\n\
             }}\n"
        ));
    }
    src
}

// ---------------------------------------------------------------------------
// Device, built once for the whole test binary
// ---------------------------------------------------------------------------

fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn the_floor_profile_asks_for_nothing_better_than_the_webgpu_spec_default() {
    let b = floor();
    let info = b.adapter_info();
    println!(
        "adapter: {} ({:?}, {:?}, driver {:?} {:?})",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    print!("{}", b.limits_table());

    let spec = wgpu::Limits::default();
    let mut over: Vec<String> = Vec::new();
    b.requested_limits()
        .check_limits_with_fail_fn(&spec, false, |name, asked, allowed| {
            over.push(format!("{name}: asked {asked}, spec default {allowed}"));
        });
    assert!(
        over.is_empty(),
        "the Floor profile asked for better than the WebGPU spec default:\n  {}",
        over.join("\n  ")
    );

    // The granted set is the requested set on native wgpu by construction
    // ("exactly the specified limits, and no better or worse"), so this asserts the
    // construction rather than the hardware. It is here because the same assertion in a
    // browser is not a tautology, and U13 runs this file's logic there.
    assert!(
        b.granted_limits().check_limits(&spec),
        "the device reports better than the spec default under Floor"
    );
}

#[test]
fn the_raised_profile_is_the_adapters_own_limits_and_still_builds_a_pipeline() {
    let b = pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Raised))
        .expect("no wgpu device at the Raised profile");
    print!("{}", b.limits_table());

    let spec = wgpu::Limits::default();
    let r = b.requested_limits();
    assert_eq!(
        r,
        &b.adapter_limits(),
        "Raised must request exactly what the adapter offers"
    );

    // Raised is *almost* a superset of Floor, and the exception is worth pinning rather than
    // asserting away. Measured on this M2 Max, wgpu 30.0.1, Metal, strict compliance on: the
    // only field where the adapter is worse than the spec default is
    // `max_buffers_and_acceleration_structures_per_shader_stage`, spec 28 against adapter 0,
    // because this adapter has no acceleration structures at all. It is inert for
    // buffer-only pipelines, which the build below is here to demonstrate rather than assume.
    // Any *other* regression means a kernel that compiles at Floor might not at Raised, and
    // that has to fail this test rather than surface as a validation error at U11.
    const KNOWN_REGRESSION: &str = "max_buffers_and_acceleration_structures_per_shader_stage";
    let mut worse: Vec<String> = Vec::new();
    spec.check_limits_with_fail_fn(r, false, |name, spec_v, adapter_v| {
        if name != KNOWN_REGRESSION {
            worse.push(format!("{name}: spec {spec_v}, adapter {adapter_v}"));
        }
    });
    assert!(
        worse.is_empty(),
        "Raised is worse than Floor on limits this backend uses:\n  {}",
        worse.join("\n  ")
    );

    // The 0 above does not stop a compute pipeline with a uniform and a storage buffer from
    // building, and this is the proof.
    let dev = b.device();
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ring io"),
        entries: &[ParamRing::layout_entry(0), storage_entry(1, false)],
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ring"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    Kernels::build(&b, "ring", RING_WGSL, &pl, &["k_fill"])
        .expect("the Raised profile could not build a two-binding compute pipeline");

    println!(
        "raised over floor on this adapter: storage binding {:.0}x, workgroup storage {:.0}x, \
         invocations {:.0}x, storage buffers per stage {} against {}",
        r.max_storage_buffer_binding_size as f64 / spec.max_storage_buffer_binding_size as f64,
        r.max_compute_workgroup_storage_size as f64
            / spec.max_compute_workgroup_storage_size as f64,
        r.max_compute_invocations_per_workgroup as f64
            / spec.max_compute_invocations_per_workgroup as f64,
        r.max_storage_buffers_per_shader_stage,
        spec.max_storage_buffers_per_shader_stage,
    );
}

#[test]
fn an_unrecognised_limits_profile_is_an_error_rather_than_a_silent_floor() {
    assert_eq!(LimitsProfile::parse("floor").unwrap(), LimitsProfile::Floor);
    assert_eq!(
        LimitsProfile::parse(" RAISED ").unwrap(),
        LimitsProfile::Raised
    );
    assert_eq!(LimitsProfile::parse("").unwrap(), LimitsProfile::Floor);
    let e = LimitsProfile::parse("rasied").unwrap_err().to_string();
    assert!(e.contains("rasied"), "{e}");
    // The default has to be Floor, not "whatever the machine offers". This is the whole
    // premise of the unit.
    assert_eq!(LimitsProfile::default(), LimitsProfile::Floor);
}

#[test]
fn module_and_pipeline_creation_is_under_five_seconds() {
    let b = floor();
    let dev = b.device();

    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field io"),
        entries: &[storage_entry(0, true), storage_entry(1, false)],
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("field"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let src = field_test_module();
    let k = Kernels::build(b, "field", &src, &pl, &FIELD_ENTRIES).expect("field module failed");
    println!("{}", k.summary());

    let cost = k.cost();
    assert_eq!(cost.compile_us, cost.module_us + cost.pipeline_us);
    assert!(
        cost.compile_us < 5_000_000,
        "module plus pipeline creation took {:.2} s, over the 5 s bar",
        cost.compile_us as f64 / 1e6
    );
    // Zero would mean the clock, not the compiler, and this is what would catch a wasm build
    // silently reporting nothing.
    assert!(
        cost.compile_us > 0,
        "compile_us is zero, so nothing timed it"
    );

    // The backend-wide accumulator is what U11 reports, and it is a sum over every module.
    // Only printed and bounded below, never asserted equal: the other tests in this binary
    // share this device and run in whatever order the harness picks.
    let total = b.prepare_cost();
    println!(
        "prepare cost across this device so far: {} modules, {} pipelines, \
         module {:.1} ms + pipelines {:.1} ms = compile_us {}",
        total.modules,
        total.pipelines,
        total.module_us as f64 / 1e3,
        total.pipeline_us as f64 / 1e3,
        total.compile_us,
    );
    assert!(
        total.compile_us >= cost.compile_us,
        "the accumulator lost a module"
    );
    k.get("k_fr_mul").expect("k_fr_mul pipeline missing");
    assert!(
        k.get("k_nope").is_err(),
        "a missing entry point must be an error"
    );
}

#[test]
fn one_uniform_ring_feeds_three_dispatches_from_three_dynamic_offsets() {
    let b = floor();
    let dev = b.device();

    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ring io"),
        entries: &[ParamRing::layout_entry(0), storage_entry(1, false)],
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ring"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let k = Kernels::build(b, "ring", RING_WGSL, &pl, &["k_fill"]).expect("ring module failed");

    const N: u32 = 64;
    const DISPATCHES: u32 = 3;
    let words = (N * DISPATCHES) as u64;
    let out = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ring out"),
        size: words * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut ring = ParamRing::new(b, "ring params", 8).expect("ring");
    assert_eq!(
        ring.stride(),
        b.granted_limits().min_uniform_buffer_offset_alignment,
        "the stride must come from the device, not from a constant"
    );
    let scales = [7u32, 11, 13];
    let offsets: Vec<u32> = (0..DISPATCHES)
        .map(|d| {
            ring.push(&Params {
                n: N,
                base: d * N,
                scale: scales[d as usize],
                _pad: 0,
            })
            .expect("push")
        })
        .collect();
    assert_eq!(
        offsets,
        vec![0, ring.stride(), 2 * ring.stride()],
        "slots must be consecutive multiples of the alignment"
    );
    ring.flush(b);

    let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(ring.binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.as_entire_binding(),
            },
        ],
    });

    let rb = Readback::new(b, "ring readback", words * 4).expect("readback");
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(k.get("k_fill").unwrap());
        for &off in &offsets {
            pass.set_bind_group(0, &bind, &[off]);
            pass.dispatch_workgroups(N.div_ceil(64), 1, 1);
        }
    }
    rb.copy_from(&mut enc, &out, 0, words * 4).unwrap();
    let bytes = pollster::block_on(rb.submit_and_read(b, enc, words * 4)).expect("readback failed");

    let got: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
    for d in 0..DISPATCHES {
        for i in 0..N {
            let want = scales[d as usize] * (i + 1);
            let idx = (d * N + i) as usize;
            assert_eq!(
                got[idx], want,
                "dispatch {d} element {i}: the ring handed out the wrong slot"
            );
        }
    }
    println!(
        "param ring: {} slots x {} B stride, {} dispatches from one buffer and one submit, \
         {} B read back",
        ring.slots(),
        ring.stride(),
        ring.used(),
        bytes.len()
    );

    // A block that does not fit a slot and a ring that runs out both have to be errors, not
    // a silent overwrite of the neighbouring slot.
    let mut tiny = ParamRing::new(b, "tiny", 1).expect("tiny ring");
    tiny.push(&Params {
        n: 1,
        base: 0,
        scale: 1,
        _pad: 0,
    })
    .unwrap();
    assert!(
        tiny.push(&Params {
            n: 1,
            base: 0,
            scale: 1,
            _pad: 0
        })
        .is_err(),
        "a full ring must refuse the next push"
    );
    tiny.reset();
    assert!(tiny
        .push(&Params {
            n: 1,
            base: 0,
            scale: 1,
            _pad: 0
        })
        .is_ok());
}

#[test]
fn a_misshapen_readback_is_refused_before_it_reaches_the_device() {
    let b = floor();
    assert!(
        Readback::new(b, "odd", 6).is_err(),
        "6 bytes is not a multiple of 4"
    );
    assert!(
        Readback::new(b, "empty", 0).is_err(),
        "a zero length map is not legal"
    );
    let rb = Readback::new(b, "small", 16).unwrap();
    let src = b.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = b.device().create_command_encoder(&Default::default());
    assert!(
        rb.copy_from(&mut enc, &src, 0, 64).is_err(),
        "copying 64 bytes into a 16 byte readback must be an error, not a truncation"
    );
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
