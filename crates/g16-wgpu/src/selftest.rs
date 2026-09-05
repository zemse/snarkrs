//! A known-answer battery that says which half of the GPU is broken, from inside the
//! browser that broke it.
//!
//! # Why this exists
//!
//! Safari 26.6 produced a Groth16 proof that snarkjs rejected, in 3 ms, for work that takes
//! Chrome 290 ms. Every error hook the crate has (`on_uncaptured_error`,
//! `set_device_lost_callback`, the `take_error()` calls after each module build, after
//! stages 0 to 4 and after the readback) stayed silent. A whole-proof failure with no error
//! is unsearchable: it could be a dropped submit, a miscompiled kernel, a broken fence, or
//! any one of thirty dispatches producing the wrong bytes.
//!
//! So this narrows it before anything is guessed at. Each check is a complete
//! encode/dispatch/readback round trip with a host-computed answer, wrapped in its own
//! validation error scope, and they are ordered so that the first failure names the layer:
//! the dispatch mechanism, then the emulated 64-bit multiply, then Montgomery arithmetic,
//! then atomics, then workgroup memory and barriers, then dynamic uniform offsets. A run
//! where every check passes and the proof is still wrong is itself a result, and a different
//! one from a run where `mul64` fails.
//!
//! # Why error scopes are safe here and not in the prover
//!
//! [`crate::device::WgpuBackend::exclusive`] argues that `pushErrorScope`/`popErrorScope`
//! cannot be used to attribute a proof's errors, because the scope stack is device-wide and
//! two concurrent proofs would nest each other's scopes. That argument is about concurrency,
//! not about error scopes. This function is `&mut`-shaped in practice: it takes the same
//! exclusive guard the prover does, so nothing else on the device is pushing scopes while it
//! runs, and each scope is pushed and popped inside one check.

use g16_field::Fr;
use g16_gpu_layout::{PackedFr, LIMBS};
use wgpu::util::DeviceExt;

use crate::device::{bad, WgpuBackend};
use crate::gen::field::{field_module, Variant, MUL64};

/// One check and what it found.
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    /// Empty when `ok`. Otherwise the first mismatch, or the validation error, in a form
    /// that can be pasted into a bug report.
    pub detail: String,
    /// Wall microseconds for the whole round trip, including compile.
    pub us: u64,
}

impl Check {
    fn json(&self) -> String {
        format!(
            r#"{{"name":"{}","ok":{},"us":{},"detail":{}}}"#,
            self.name,
            self.ok,
            self.us,
            json_str(&self.detail)
        )
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Every check, as a JSON array. Never returns an error: a check that could not even be set
/// up is reported as a failed check, because the caller is a browser page whose whole job is
/// to show what happened.
pub async fn run_json(backend: &WgpuBackend, battery: Battery) -> String {
    let checks = run(backend, battery).await;
    let body = checks
        .iter()
        .map(Check::json)
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!("[\n  {body}\n]")
}

/// How much of the battery to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Battery {
    /// The two checks that stand between this prover and a silently wrong proof, and nothing
    /// else. Measured at about 350 ms on Safari 26.6 and 40 ms on Chrome 152, both dominated
    /// by the first shader compile of the process. Run unconditionally when the device is
    /// opened; see [`crate::wasm::create_prover`].
    Guard,
    /// Everything. Adds the fence timing, `mul64` on its own, atomics, workgroup memory and
    /// dynamic uniform offsets: the checks that turn "the GPU is wrong" into "*this* is
    /// wrong". Costs about a second, and is only run when a page asks for it.
    Full,
}

/// The battery, in the order that makes a failure diagnostic.
pub async fn run(backend: &WgpuBackend, battery: Battery) -> Vec<Check> {
    let mut out = Vec::new();
    // First, because if a dispatch cannot write a buffer at all then nothing below means
    // anything and no kernel is at fault.
    out.push(storage_write(backend).await);
    if battery == Battery::Full {
        out.push(fence_is_a_fence(backend).await);
        out.push(mul64(backend).await);
    }
    // Second, and in the guard set, because this is the one that caught the Safari bug: the
    // field prelude is in front of every kernel the prover compiles, so a browser that cannot
    // translate it cannot run any of them.
    out.push(field_ops(backend).await);
    if battery == Battery::Full {
        out.push(atomics(backend).await);
        out.push(workgroup_barrier(backend).await);
        out.push(dynamic_uniform_offsets(backend).await);
    }
    out
}

/// Runs [`Battery::Guard`] and turns the first failure into a `ProveError`.
///
/// This is what stands between a browser that miscompiles our WGSL and a proof that verifies
/// nowhere. It is deliberately not optional: the failure it catches raises nothing through
/// `on_uncaptured_error` on Safari, produces no `getCompilationInfo()` message, and costs
/// three milliseconds, so every other signal a caller could look at says the proof is fine.
pub async fn guard(backend: &WgpuBackend) -> Result<(), g16_core::ProveError> {
    let checks = run(backend, Battery::Guard).await;
    match as_error(&checks) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------------------

/// One storage buffer at `@group(0) @binding(0)`, optionally seeded, dispatched once and read
/// back, with a validation scope around the whole thing.
struct Kernel<'a> {
    label: &'a str,
    source: String,
    /// Words in the storage buffer. Seeded with `seed`, zero-filled past its end.
    words: usize,
    seed: Vec<u32>,
    groups: u32,
}

impl Kernel<'_> {
    async fn go(self, backend: &WgpuBackend) -> Result<Vec<u32>, String> {
        let device = backend.device();
        // One scope for compile, bind and submit together. Popping it after the readback
        // would also catch anything the map raised, but a map error is already a `Result`
        // and folding the two would make the message ambiguous about which failed.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let bytes = (self.words * 4) as u64;
        let mut init = self.seed.clone();
        init.resize(self.words, 0);
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(self.label),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(self.label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(self.label),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(self.label),
            source: wgpu::ShaderSource::Wgsl(self.source.as_str().into()),
        });
        // The one call in this crate that asks the browser's own compiler what it thought,
        // rather than waiting for a dispatch to write nothing. Chrome and Safari both report
        // WGSL errors here and neither is required to fail `createShaderModule` itself.
        let info = module.get_compilation_info().await;
        let complaints: Vec<String> = info
            .messages
            .iter()
            .filter(|m| m.message_type == wgpu::CompilationMessageType::Error)
            .map(|m| match &m.location {
                Some(l) => format!("line {}: {}", l.line_number, m.message),
                None => m.message.clone(),
            })
            .collect();
        if !complaints.is_empty() {
            let _ = scope.pop().await;
            return Err(condense(&format!(
                "WGSL rejected: {}",
                complaints.join("; ")
            )));
        }

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(self.label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.label),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(self.label),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(self.groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);

        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&staging, wgpu::MapMode::Read, 0..bytes, move |r| {
            let _ = tx.send(r);
        });
        backend.submit([enc.finish()]);
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        rx.recv_async()
            .await
            .map_err(|_| "the map callback was dropped".to_string())?
            .map_err(|e| format!("map: {e}"))?;

        let view = staging
            .slice(0..bytes)
            .get_mapped_range()
            .map_err(|e| format!("mapped range: {e}"))?;
        let got: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();

        if let Some(e) = scope.pop().await {
            return Err(format!("validation: {}", condense(&e.to_string())));
        }
        if let Some(e) = backend.take_error() {
            return Err(format!("uncaptured: {}", condense(&e.to_string())));
        }
        Ok(got)
    }
}

/// Trims a device error down to something a page can show.
///
/// Safari answers a failed pipeline build with the *entire* generated Metal source, which for
/// the field prelude is over 100 KB, followed by the compiler's diagnostics. The diagnostics
/// are the whole content and they are at the end, so this keeps every line that carries one
/// plus a short head, and says how much it dropped. The full text is still in the browser
/// console, where wgpu logs it.
fn condense(msg: &str) -> String {
    const HEAD: usize = 200;
    let diagnostics: Vec<&str> = msg
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("error:") || l.contains("warning:"))
        .collect();
    if diagnostics.is_empty() || msg.len() <= HEAD {
        return msg.chars().take(600).collect();
    }
    let head: String = msg.chars().take(HEAD).collect();
    // Characters and not bytes, because `head` is taken in characters and a count in the
    // other unit would not add up to the whole.
    format!(
        "{head} [{} more characters of generated source elided] {}",
        msg.chars().count().saturating_sub(HEAD),
        diagnostics.join(" | ")
    )
}

/// Compares and formats the first mismatch, which is the only one worth printing.
fn compare(got: &[u32], want: &[u32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("read {} words, expected {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            let zeros = got.iter().filter(|x| **x == 0).count();
            return Err(format!(
                "word {i} is {g:#010x}, expected {w:#010x} ({zeros} of {} words are zero, \
                 which is what an unwritten buffer looks like)",
                got.len()
            ));
        }
    }
    Ok(())
}

fn finish(name: &'static str, t0: web_time::Instant, r: Result<(), String>) -> Check {
    let us = t0.elapsed().as_micros() as u64;
    match r {
        Ok(()) => Check {
            name,
            ok: true,
            detail: String::new(),
            us,
        },
        Err(detail) => Check {
            name,
            ok: false,
            detail,
            us,
        },
    }
}

// ---------------------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------------------

/// Does a compute dispatch write a storage buffer at all?
///
/// If this fails, nothing else in the file means anything: the whole encode, submit and
/// readback mechanism is broken on this browser and no kernel is at fault.
async fn storage_write(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let n = 256usize;
    let k = Kernel {
        label: "selftest storage_write",
        source: r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = gid.x * 2u + 1u;
}
"#
        .to_string(),
        words: n,
        seed: Vec::new(),
        groups: (n / 64) as u32,
    };
    let want: Vec<u32> = (0..n as u32).map(|i| i * 2 + 1).collect();
    let r = k.go(backend).await.and_then(|got| compare(&got, &want));
    finish("storage_write", t0, r)
}

/// Does `wait_for_submitted_work` actually wait?
///
/// A kernel with a loop the compiler cannot fold away, submitted and fenced with nothing read
/// back, then read back separately. The fence is timed on its own. On a working browser the
/// fence takes most of the round trip; a fence that returns in microseconds while the answer
/// is nonetheless correct means every stage timing in the prover is a lie and, worse, that
/// nothing in the prover is ordered against GPU completion.
async fn fence_is_a_fence(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let device = backend.device();
    let n = 1024usize;
    let bytes = (n * 4) as u64;
    let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 200k rounds of a dependent xorshift: not vectorisable, not foldable, and the result
    // is stored, so no compiler is entitled to delete it.
    var x: u32 = gid.x * 2654435761u + 1u;
    for (var i: u32 = 0u; i < 200000u; i = i + 1u) {
        x = x ^ (x << 13u);
        x = x ^ (x >> 17u);
        x = x ^ (x << 5u);
    }
    out[gid.x] = x;
}
"#;
    let run = async {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selftest fence"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selftest fence"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selftest fence"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selftest fence"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("selftest fence"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("selftest fence"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("selftest fence"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("selftest fence"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((n / 64) as u32, 1, 1);
        }
        let t = web_time::Instant::now();
        backend.submit([enc.finish()]);
        backend
            .wait_for_submitted_work()
            .await
            .map_err(|e| format!("{e}"))?;
        let fence_us = t.elapsed().as_micros() as u64;

        // Now read it back through a second submit, and time that too. The map cannot resolve
        // before the copy has run, so this number is a lower bound on the real GPU time
        // whatever the fence says.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selftest fence readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("selftest fence readback"),
        });
        enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&staging, wgpu::MapMode::Read, 0..bytes, move |r| {
            let _ = tx.send(r);
        });
        backend.submit([enc.finish()]);
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        rx.recv_async()
            .await
            .map_err(|_| "the map callback was dropped".to_string())?
            .map_err(|e| format!("map: {e}"))?;
        let view = staging
            .slice(0..bytes)
            .get_mapped_range()
            .map_err(|e| format!("mapped range: {e}"))?;
        let got: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        if let Some(e) = scope.pop().await {
            return Err(format!("validation: {}", condense(&e.to_string())));
        }

        let want: Vec<u32> = (0..n as u32).map(xorshift_200k).collect();
        compare(&got, &want)?;
        // 200k dependent rounds over 1024 lanes is milliseconds of real work on any GPU.
        // The threshold is deliberately far below what it costs, so this only fires on a
        // fence that is not one at all.
        if fence_us < 200 {
            return Err(format!(
                "the answer is right but the fence returned in {fence_us} us; \
                 onSubmittedWorkDone is not waiting for the GPU on this browser"
            ));
        }
        Ok(())
    };
    let r = run.await;
    finish("fence_is_a_fence", t0, r)
}

fn xorshift_200k(lane: u32) -> u32 {
    let mut x = lane.wrapping_mul(2654435761).wrapping_add(1);
    for _ in 0..200000u32 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
    }
    x
}

/// The emulated 32x32 -> 64 multiply, on inputs chosen to carry into every partial product.
///
/// WGSL has no 64-bit integer, so `gen::field::MUL64` builds one out of four 16-bit products
/// and a recombination with, as its own comment says, zero headroom. Every field operation in
/// this backend is built on it, so a browser whose compiler reassociates or narrows any of
/// those adds produces a prover that is wrong everywhere and fails no validation anywhere.
async fn mul64(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let lanes = 128usize;
    let source = format!(
        "{MUL64}
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let a = 0x9e3779b9u * (i + 1u) + 0xffff0000u;
    let b = 0x85ebca6bu ^ (i * 2654435761u);
    let p = mul64(a, b);
    out[2u * i] = p.x;
    out[2u * i + 1u] = p.y;
}}
"
    );
    let k = Kernel {
        label: "selftest mul64",
        source,
        words: lanes * 2,
        seed: Vec::new(),
        groups: (lanes / 64) as u32,
    };
    let mut want = Vec::with_capacity(lanes * 2);
    for i in 0..lanes as u32 {
        let a = 0x9e3779b9u32
            .wrapping_mul(i.wrapping_add(1))
            .wrapping_add(0xffff0000);
        let b = 0x85ebca6bu32 ^ i.wrapping_mul(2654435761);
        let p = a as u64 * b as u64;
        want.push(p as u32);
        want.push((p >> 32) as u32);
    }
    let r = k.go(backend).await.and_then(|got| compare(&got, &want));
    finish("mul64", t0, r)
}

/// The whole field prelude, with a call graph that reaches every part of it.
///
/// `fr_mul`, `fr_add` and `fr_sub` are checked against the host; everything else (the `Fq`
/// operations, `Fq2`, the Montgomery lift and reduction, the constants) is called and its
/// result folded into one word that nothing predicts. The point of calling it is not the
/// answer, it is that a browser cannot dead-strip a function whose result is stored, so a
/// construct that only this browser rejects has to be compiled and has to be reported.
///
/// Named `field_ops` and not `fr_mul` because a failure here is a failure of the prelude as a
/// whole, which is what every kernel in the prover puts in front of its own entry points.
async fn field_ops(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let v = Variant::Cios32Unrolled;
    let source = format!(
        "{}
@group(0) @binding(0) var<storage, read_write> io: array<u32>;
@compute @workgroup_size(1)
fn main() {{
    var a: Fr;
    var b: Fr;
    var qa: Fq;
    var qb: Fq;
    for (var i: u32 = 0u; i < {LIMBS}u; i = i + 1u) {{
        a[i] = io[i];
        b[i] = io[{LIMBS}u + i];
        qa[i] = io[i];
        qb[i] = io[{LIMBS}u + i];
    }}
    let c = fr_mul(a, b);
    let d = fr_add(a, b);
    let e = fr_sub(a, b);
    for (var i: u32 = 0u; i < {LIMBS}u; i = i + 1u) {{
        io[2u * {LIMBS}u + i] = c[i];
        io[3u * {LIMBS}u + i] = d[i];
        io[4u * {LIMBS}u + i] = e[i];
    }}
    // Everything the prover also compiles, reached so that it cannot be stripped. The value
    // is not predicted here; a wrong answer in this half shows up as a wrong proof, and a
    // construct the browser cannot translate shows up as this whole check failing.
    let q = fq_add(fq_mul(qa, qb), fq_sub(fq_neg(qa), fq_sqr(qb)));
    let r = fq_from_mont(fq_to_mont(q));
    let z = fq2_mul(Fq2(qa, qb), fq2_sqr(Fq2(qb, qa)));
    let w = fr_from_mont(fr_to_mont(fr_neg(fr_one())));
    var touch: u32 = 0u;
    for (var i: u32 = 0u; i < {LIMBS}u; i = i + 1u) {{
        touch = touch ^ r[i] ^ z.c0[i] ^ z.c1[i] ^ w[i];
    }}
    if (fr_is_zero(fr_zero()) && fq2_is_zero(fq2_zero()) && fq2_eq(fq2_one(), fq2_one())
        && fr_eq(a, a) && fq_is_zero(fq_zero())) {{
        touch = touch + 1u;
    }}
    io[5u * {LIMBS}u] = touch;
}}
",
        field_module(v)
    );

    // Two values with no special structure, so a limb that is dropped or a carry that is lost
    // changes the answer. Derived from the field itself rather than typed out, so this cannot
    // drift from whatever `g16-field` says BN254's scalar field is.
    let a = Fr::from(0x0123_4567_89ab_cdefu64) * Fr::from(7u64) - Fr::from(3u64);
    let b = Fr::from(0xfedc_ba98_7654_3210u64) * Fr::from(11u64) + Fr::from(5u64);
    let mut seed = Vec::with_capacity(2 * LIMBS);
    seed.extend_from_slice(&PackedFr::from_fr(&a).v);
    seed.extend_from_slice(&PackedFr::from_fr(&b).v);

    let mut want = seed.clone();
    want.extend_from_slice(&PackedFr::from_fr(&(a * b)).v);
    want.extend_from_slice(&PackedFr::from_fr(&(a + b)).v);
    want.extend_from_slice(&PackedFr::from_fr(&(a - b)).v);

    let k = Kernel {
        label: "selftest field_ops",
        source,
        words: 5 * LIMBS + 1,
        seed,
        groups: 1,
    };
    let r = k
        .go(backend)
        .await
        .and_then(|got| compare(&got[..want.len()], &want));
    finish("field_ops", t0, r)
}

/// `atomicAdd` into a storage buffer, which is how the MSM's counting sort builds its
/// histogram. A browser that drops contended atomic adds produces bucket counts that do not
/// match the entries written into them, and the MSM reads past or short of its own data.
async fn atomics(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let buckets = 8usize;
    let lanes = 1024u32;
    let k = Kernel {
        label: "selftest atomics",
        source: format!(
            r#"
@group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    atomicAdd(&counts[gid.x % {buckets}u], 1u);
}}
"#
        ),
        words: buckets,
        seed: Vec::new(),
        groups: lanes / 64,
    };
    let want = vec![lanes / buckets as u32; buckets];
    let r = k.go(backend).await.and_then(|got| compare(&got, &want));
    finish("atomics", t0, r)
}

/// Workgroup storage plus `workgroupBarrier`, which every NTT butterfly pass depends on.
///
/// The reduction is deliberately written so that a barrier the compiler drops, or one that
/// does not synchronise, gives a wrong sum rather than a slow one.
async fn workgroup_barrier(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let n = 256usize;
    let k = Kernel {
        label: "selftest workgroup_barrier",
        source: r#"
var<workgroup> tile: array<u32, 64>;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // Written by one lane, read by its neighbour, so the barrier is the only thing making
    // the read defined.
    tile[lid.x] = gid.x * 3u + 1u;
    workgroupBarrier();
    var s: u32 = 0u;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        s = s + tile[(lid.x + i) % 64u];
    }
    out[gid.x] = s;
}
"#
        .to_string(),
        words: n,
        seed: Vec::new(),
        groups: (n / 64) as u32,
    };
    let want: Vec<u32> = (0..n as u32)
        .map(|gid| {
            let base = (gid / 64) * 64;
            (0..64u32).map(|j| (base + j) * 3 + 1).sum()
        })
        .collect();
    let r = k.go(backend).await.and_then(|got| compare(&got, &want));
    finish("workgroup_barrier", t0, r)
}

/// One uniform buffer, three dispatches, three dynamic offsets.
///
/// This is the shape of [`crate::params::ParamRing`], which every kernel in the prover reads
/// its per-dispatch parameters from. If a browser ignores the dynamic offset, every dispatch
/// after the first runs with the first one's parameters: the right number of dispatches, no
/// validation error, and an answer that is wrong in a way nothing else here would catch.
async fn dynamic_uniform_offsets(backend: &WgpuBackend) -> Check {
    let t0 = web_time::Instant::now();
    let device = backend.device();
    let run = async {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        // 256 is `minUniformBufferOffsetAlignment` at every tier of every browser, per the
        // module docs on `crate::device`.
        let stride = 256usize;
        let blocks = 3usize;
        let mut ring = vec![0u8; stride * blocks];
        for (b, chunk) in ring.chunks_mut(stride).enumerate() {
            // Each block carries {base, value}: where to write and what to write.
            chunk[0..4].copy_from_slice(&((b * 4) as u32).to_le_bytes());
            chunk[4..8].copy_from_slice(&(0xa5a50000u32 + b as u32).to_le_bytes());
        }
        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("selftest ring"),
            contents: &ring,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let words = blocks * 4;
        let bytes = (words * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selftest ring out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selftest ring"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(8),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selftest ring"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selftest ring"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct P { base: u32, value: u32 }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    out[p.base + lid.x] = p.value + lid.x;
}
"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("selftest ring"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("selftest ring"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubo,
                        offset: 0,
                        size: wgpu::BufferSize::new(8),
                    }),
                },
            ],
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selftest ring staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("selftest ring"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("selftest ring"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            for b in 0..blocks {
                pass.set_bind_group(0, &bind, &[(b * stride) as u32]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes);
        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&staging, wgpu::MapMode::Read, 0..bytes, move |r| {
            let _ = tx.send(r);
        });
        backend.submit([enc.finish()]);
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        rx.recv_async()
            .await
            .map_err(|_| "the map callback was dropped".to_string())?
            .map_err(|e| format!("map: {e}"))?;
        let view = staging
            .slice(0..bytes)
            .get_mapped_range()
            .map_err(|e| format!("mapped range: {e}"))?;
        let got: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        if let Some(e) = scope.pop().await {
            return Err(format!("validation: {}", condense(&e.to_string())));
        }

        let mut want = vec![0u32; words];
        for b in 0..blocks {
            for l in 0..4usize {
                want[b * 4 + l] = 0xa5a50000 + b as u32 + l as u32;
            }
        }
        compare(&got, &want)
    };
    let r = run.await;
    finish("dynamic_uniform_offsets", t0, r)
}

/// Every error message the browser's own WGSL compiler produced for the modules the prover
/// already built, as JSON. Empty means it had nothing to say.
///
/// Separate from the battery because it does not run anything: it asks the modules that are
/// already compiled what they were told. On the web this is the only channel that reports a
/// WGSL problem at all, since `createShaderModule` is not required to fail.
pub async fn module_diagnostics(kernels: &[&crate::pipelines::Kernels]) -> String {
    let mut rows = Vec::new();
    for k in kernels {
        let info = k.module().get_compilation_info().await;
        let msgs: Vec<String> = info
            .messages
            .iter()
            .filter(|m| m.message_type != wgpu::CompilationMessageType::Info)
            .map(|m| match &m.location {
                Some(l) => format!("{:?} line {}: {}", m.message_type, l.line_number, m.message),
                None => format!("{:?}: {}", m.message_type, m.message),
            })
            .collect();
        rows.push(format!(
            r#"{{"module":{},"messages":[{}]}}"#,
            json_str(k.label()),
            msgs.iter()
                .map(|m| json_str(m))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    format!("[{}]", rows.join(","))
}

/// Fails the whole battery loudly rather than returning a list nobody reads.
///
/// The prover must never hand back a proof on a device that cannot compute. See
/// `crate::wasm::create_prover`, which calls this and refuses to open the device on a
/// failure.
pub fn first_failure(checks: &[Check]) -> Option<String> {
    checks
        .iter()
        .find(|c| !c.ok)
        .map(|c| format!("GPU self-test {:?} failed: {}", c.name, c.detail))
}

/// A `wgpu`-only convenience so the caller does not have to name `bad`.
pub fn as_error(checks: &[Check]) -> Option<g16_core::ProveError> {
    first_failure(checks).map(bad)
}
