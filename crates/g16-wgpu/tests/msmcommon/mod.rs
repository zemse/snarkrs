//! What the G1 and G2 MSM suites genuinely share.
//!
//! `tests/msm_g1.rs` and `tests/msm_g2.rs` ask the same questions of two curves, and about a
//! third of each file was the same code: the sentinel fill, the two readbacks, the scalar
//! generators and the two statistics the sweeps print. Those depend on nothing about the
//! curve, so they live here and the two suites cannot drift on them. What is *not* here is
//! everything that touches a point: the base generators, `run_msm`, and the timing harness
//! all name a curve in their types, and forcing them through a trait would buy nothing but a
//! layer to read past.
//!
//! Each test binary is its own process, so the statics below are per binary. That is
//! deliberate for [`exclusive`], which serialises one binary's own tests against each other;
//! `tests/gpulock` is the one that spans processes.

// The two suites do not use an identical subset: the G1 file reaches for the artifact-backed
// helpers and the G2 file does not, so anything unused in one binary would warn there.
#![allow(dead_code)]

use ark_std::rand::Rng;
use ark_std::UniformRand;
use g16_field::{Fr, One, Zero};
use g16_wgpu::{Readback, WgpuBackend};

/// The sentinel every output buffer is pre-filled with. Not zero, so a kernel that writes
/// nothing is caught rather than silently agreeing with an identity oracle, and not a
/// plausible bucket row either.
pub const SENTINEL: u32 = 0xDEAD_BEEF;

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

pub fn storage_words(b: &WgpuBackend, label: &str, data: &[u32]) -> wgpu::Buffer {
    let bytes = (data.len().max(1) * 4) as u64;
    let buf = b.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        b.queue().write_buffer(&buf, 0, bytemuck::cast_slice(data));
    }
    buf
}

pub fn fill(b: &WgpuBackend, buf: &wgpu::Buffer, word: u32) {
    let words = (buf.size() / 4) as usize;
    b.queue()
        .write_buffer(buf, 0, bytemuck::cast_slice(&vec![word; words]));
}

pub fn read_bytes(b: &WgpuBackend, buf: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let rb = Readback::new(b, "msm readback", bytes).expect("readback");
    let mut enc = b.device().create_command_encoder(&Default::default());
    rb.copy_from(&mut enc, buf, 0, bytes).expect("copy");
    pollster::block_on(rb.submit_and_read(b, enc, bytes)).expect("read")
}

pub fn read_words(b: &WgpuBackend, buf: &wgpu::Buffer) -> Vec<u32> {
    read_bytes(b, buf, buf.size())
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/// `n` scalars, none of them 0 or 1, so every one reaches a bucket.
pub fn general_scalars(n: usize, rng: &mut impl Rng) -> Vec<Fr> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let x = Fr::rand(rng);
        if !(x.is_zero() || x.is_one()) {
            out.push(x);
        }
    }
    out
}

/// A witness-shaped vector: `general_ppm` parts per million are general, the rest split
/// between 0 and 1. That is the shape design §5 sizes the window for, and it is the only
/// shape that exercises `msm_ones_*` and the bucket path at the same time.
pub fn witness_shaped(n: usize, general_ppm: u32, rng: &mut impl Rng) -> Vec<Fr> {
    (0..n)
        .map(|_| {
            let r: u32 = rng.gen_range(0..1_000_000);
            if r < general_ppm {
                let mut x = Fr::rand(rng);
                while x.is_zero() || x.is_one() {
                    x = Fr::rand(rng);
                }
                x
            } else if r & 1 == 0 {
                Fr::zero()
            } else {
                Fr::one()
            }
        })
        .collect()
}

/// How many of `xs` are neither 0 nor 1, which is what a [`g16_wgpu::DigitPlan`] sizes from.
pub fn general_count(xs: &[Fr]) -> u32 {
    xs.iter().filter(|x| !(x.is_zero() || x.is_one())).count() as u32
}

// ---------------------------------------------------------------------------
// Statistics, and the in-process GPU lock
// ---------------------------------------------------------------------------

pub fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

pub fn argmin(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// Serialises the timing tests in one binary against each other and against the heavy
/// correctness ones.
///
/// Tests in one binary run in parallel by default and every one of them dispatches on the
/// same device, so an unguarded sweep measures whatever else happened to be resident. This
/// does not defend against the other test binaries, which run in their own processes;
/// `tests/gpulock::exclusive_gpu` is the one that does.
pub fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
