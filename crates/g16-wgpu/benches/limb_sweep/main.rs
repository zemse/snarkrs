//! Limb-width sweep for BN254 `Fr` Montgomery multiplication in WGSL.
//!
//! This is the sweep that decided the field representation for the whole backend. Run it with
//! `cargo bench -p g16-wgpu --bench limb_sweep`.
//!
//! WGSL has no 64-bit integer type and no `mulhi`, so neither our Metal CIOS (which widens
//! both factors to `ulong` at the multiply-accumulate site) nor our CUDA one can be ported.
//! This measures the real options on this machine instead of inheriting the literature's
//! answer. Two results came out of it and the second is the bigger one: 32-bit limbs beat
//! 13-bit ones by 1.32x while being 2.5x smaller, and unrolling the limb loops is worth 2.6x
//! to 3.8x, more than any choice of limb width.
//!
//! # What is measured, and what would make a result a lie
//!
//! Each thread chains 256 Montgomery multiplies into itself so the compiler cannot hoist or
//! drop them, 65,536 threads, best of five submissions after a warm-up. **Every variant is
//! validated against `num-bigint` across all 256 chained multiplies before its time is
//! printed**, so a wrong-but-fast variant cannot win the table. The device is requested at
//! `wgpu::Limits::default()` (the browser floor) with `STRICT_WEBGPU_COMPLIANCE`, because a
//! variant that only compiles on native Metal is not a variant we can ship.
//!
//! The two variants the backend ships come from `g16_wgpu::gen` itself, not from a copy, so
//! this measures the code the prover runs. Everything else lives in `variants.rs` and exists
//! only to be beaten.

mod variants;

use std::time::Instant;

use g16_gpu_layout::FR_MODULUS;
use g16_wgpu::gen::{Variant, FR, MUL64};
use num_bigint::BigUint;
use num_traits::One;

const ELEMS: u32 = 1 << 16;
const ITERS: u32 = 256;

fn modulus() -> BigUint {
    BigUint::from_slice(&FR_MODULUS)
}

/// `-r^{-1} mod 2^b`, by Newton iteration on the odd modulus.
fn neg_inv(r: &BigUint, b: u32) -> u32 {
    let m = BigUint::one() << b;
    let r_low = r % &m;
    let mut inv = BigUint::one();
    for _ in 0..b {
        let t = (&r_low * &inv) % &m;
        let two_minus = (&m + BigUint::from(2u32) - t) % &m;
        inv = (&inv * two_minus) % &m;
    }
    let neg = (&m - inv) % &m;
    neg.iter_u32_digits().next().unwrap_or(0)
}

fn to_limbs(x: &BigUint, b: u32, n: usize) -> Vec<u32> {
    let mask = (BigUint::one() << b) - BigUint::one();
    (0..n)
        .map(|i| {
            let d = (x >> (b as usize * i)) & &mask;
            d.iter_u32_digits().next().unwrap_or(0)
        })
        .collect()
}

fn from_limbs(v: &[u32], b: u32) -> BigUint {
    v.iter()
        .enumerate()
        .fold(BigUint::from(0u32), |acc, (i, l)| {
            acc + (BigUint::from(*l) << (b as usize * i))
        })
}

/// SplitMix64, so runs are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_fr(&mut self, r: &BigUint) -> BigUint {
        let mut bytes = [0u8; 32];
        for c in bytes.chunks_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes) % r
    }
}

struct Sweep {
    name: String,
    b: u32,
    n: usize,
    src: String,
}

fn sweep_variants(r: &BigUint) -> Vec<Sweep> {
    let mut out = Vec::new();

    // The looped family, three algorithms. Present only as the control for unrolling.
    for b in [11u32, 12, 13, 14, 15] {
        let n = limbs_for(b);
        out.push(Sweep {
            name: format!("{b}-bit x {n} limbs (lazy carry)"),
            b,
            n,
            src: variants::lazy_cios(b, n, &to_limbs(r, b, n), neg_inv(r, b))
                + &variants::driver(n, &format!("array<u32, {n}>"), "montmul"),
        });
    }
    {
        let (b, n) = (16u32, 16usize);
        out.push(Sweep {
            name: "16-bit x 16 limbs (split product)".into(),
            b,
            n,
            src: variants::cios16(n, &to_limbs(r, b, n), neg_inv(r, b))
                + &variants::driver(n, &format!("array<u32, {n}>"), "montmul"),
        });
    }
    {
        let (b, n) = (32u32, 8usize);
        out.push(Sweep {
            name: "32-bit x 8 limbs (emulated mul64), looped".into(),
            b,
            n,
            src: variants::cios32_looped(&to_limbs(r, b, n), neg_inv(r, b))
                + &variants::driver(n, &format!("array<u32, {n}>"), "montmul"),
        });
    }

    // The unrolled lazy-carry family, so "13-bit wins" can be tested rather than repeated.
    for b in [11u32, 12, 13, 14, 15] {
        let n = limbs_for(b);
        out.push(Sweep {
            name: format!("{b}-bit x {n} limbs, unrolled (lazy carry)"),
            b,
            n,
            src: variants::lazy_cios_unrolled(b, n, &to_limbs(r, b, n), neg_inv(r, b))
                + &variants::driver(n, &format!("array<u32, {n}>"), "montmul"),
        });
    }

    // The two the backend carries, taken from the backend's own generator.
    for v in [Variant::NoCarry13x20, Variant::Cios32Unrolled] {
        let (b, n) = (v.limb_bits(), v.limbs());
        let mut src = String::new();
        if v.needs_mul64() {
            src.push_str(MUL64);
        }
        src.push_str(&FR.ops(v));
        src.push_str(&variants::driver(n, "Fr", "fr_mul"));
        out.push(Sweep {
            name: format!("{b}-bit x {n} limbs, unrolled, g16_wgpu::gen {v:?}"),
            b,
            n,
            src,
        });
    }
    out
}

/// Limbs needed to hold a 255-bit value at `b` bits each. 255 and not 254 because the signed
/// recoding downstream needs the extra bit, and because a layout that cannot hold `r-1` with
/// room for one carry is not a layout.
fn limbs_for(b: u32) -> usize {
    let n = 254usize.div_ceil(b as usize);
    if n * b as usize >= 255 {
        n
    } else {
        n + 1
    }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let r = modulus();
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.flags |= wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no adapter");
    println!("adapter: {}", adapter.get_info().name);

    // Deliberately the browser's default limits, not this adapter's. A kernel that only
    // fits in native Metal's headroom is not a kernel we can ship.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("no device");

    let mut rng = Rng(0xC0FF_EE11);
    let a_std: Vec<BigUint> = (0..ELEMS).map(|_| rng.next_fr(&r)).collect();
    let b_std: Vec<BigUint> = (0..ELEMS).map(|_| rng.next_fr(&r)).collect();

    println!(
        "\n{ELEMS} elements, {ITERS} chained multiplies each = {:.1}M multiplies per run\n",
        (ELEMS as f64 * ITERS as f64) / 1e6
    );
    println!(
        "{:<58} {:>6} {:>8} {:>8} {:>9} {:>8} {:>6}",
        "variant", "bytes", "ms", "Gmul/s", "src KiB", "build ms", "ok"
    );
    println!("{}", "-".repeat(112));

    let mut rows = Vec::new();
    for v in sweep_variants(&r) {
        let big_r = BigUint::one() << (v.b as usize * v.n);
        let mont = |x: &BigUint| (x * &big_r) % &r;

        let mut ha = Vec::with_capacity(ELEMS as usize * v.n);
        let mut hb = Vec::with_capacity(ELEMS as usize * v.n);
        for i in 0..ELEMS as usize {
            ha.extend(to_limbs(&mont(&a_std[i]), v.b, v.n));
            hb.extend(to_limbs(&mont(&b_std[i]), v.b, v.n));
        }

        let (ms, out, compile_ms, src_len) = match bench(&device, &queue, &v, &ha, &hb).await {
            Ok(x) => x,
            Err(e) => {
                println!("{:<58} {}", v.name, e);
                continue;
            }
        };

        // With ITERS chained multiplies by b the closed form is a*b^ITERS in Montgomery form.
        // Check the first few elements against bigint arithmetic.
        let mut correct = true;
        for i in 0..8usize {
            let mut want = mont(&a_std[i]);
            let bm = mont(&b_std[i]);
            let rinv = big_r.modpow(&(&r - BigUint::from(2u32)), &r);
            for _ in 0..ITERS {
                want = (&want * &bm % &r) * &rinv % &r;
            }
            let got = from_limbs(&out[i * v.n..(i + 1) * v.n], v.b);
            if got != want {
                correct = false;
                break;
            }
        }

        let gmul = (ELEMS as f64 * ITERS as f64) / (ms / 1e3) / 1e9;
        let bytes = v.n * 4;
        println!(
            "{:<58} {:>6} {:>8.2} {:>8.3} {:>9.1} {:>8.1} {:>6}",
            v.name,
            bytes,
            ms,
            gmul,
            src_len as f64 / 1024.0,
            compile_ms,
            if correct { "yes" } else { "NO" }
        );
        rows.push((v.name.clone(), bytes, gmul, correct));
    }

    println!();
    if let Some(best) = rows
        .iter()
        .filter(|r| r.3)
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    {
        println!(
            "fastest correct: {} at {:.3} Gmul/s, {} bytes per element",
            best.0, best.2, best.1
        );
    }
}

async fn bench(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    v: &Sweep,
    ha: &[u32],
    hb: &[u32],
) -> Result<(f64, Vec<u32>, f64, usize), String> {
    use wgpu::util::DeviceExt;

    let t_compile = Instant::now();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&v.name),
        source: wgpu::ShaderSource::Wgsl(v.src.as_str().into()),
    });
    if let Some(e) = scope.pop().await {
        return Err(format!("shader: {e}"));
    }

    let bytes = (ha.len() * 4) as u64;
    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(ha),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(hb),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_o = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [ITERS, ELEMS, 0u32, 0u32];
    let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
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

    let groups = ELEMS.div_ceil(64);
    let dispatch = |label: &str| {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.finish()
    };

    // Warm up: first submission pays pipeline creation and any lazy Metal compile.
    queue.submit([dispatch("warmup")]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let reps = 5;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        queue.submit([dispatch("bench")]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&buf_o, 0, &staging, 0, bytes);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Ok((best, out, compile_ms, v.src.len()))
}
