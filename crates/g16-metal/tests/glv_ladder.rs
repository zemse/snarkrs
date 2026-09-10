//! The GLV ladder on its own, against arkworks, over scalars a `ptau prepare` cannot
//! produce.
//!
//! `fft::tests` compares whole transforms against the CPU backend, which is the check the
//! byte-identity claim rests on, but it only ever feeds the ladder twiddles: powers of a
//! two-adic root and the `1/n` scalings. This file reaches the kernel directly and puts
//! arbitrary `(k1, k2)` pairs through it, including the widths and sign combinations the
//! lattice never emits, because the one thing a real transform cannot exercise is the top
//! window of a magnitude that actually reaches 127 bits.
//!
//! The dispatch is `fft_mix_scale`, with `n = 2` and the `lo` input the point at infinity.
//! That kernel is `out[0] = [s] in[0] + [k] in[1]`, so a zero `in[0]` makes `out[0]` the
//! ladder's answer and nothing else.

#![cfg(target_os = "macos")]

use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ff::{BigInt, Field};
use g16_field::{AffineRepr, CurveGroup, Fq, Fq2, Fr, G1Affine, G2Affine};
use g16_metal::kernels::{CEREMONY_MSL, FFT_MSL, FR_MSL, MSM_MSL};
use g16_metal::layout::{as_bytes, PackedFq, PackedFq2, PackedGlv};
use g16_metal::msm::{PackedXyzzG1, PackedXyzzG2};
use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Mirrors `struct FftParams`; only `n` is read by `fft_mix_scale`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FftParams {
    n: u32,
    span: u32,
    log_span: u32,
    tw_shift: u32,
    gid_off: u32,
}

/// `k1` and `k2` as the kernel reads them, without going through the lattice. `neg1` and
/// `neg2` are applied to the *magnitudes*, so the scalar this stands for is
/// `+-k1 + lambda * (+-k2)`.
fn entry(k1: u128, neg1: bool, k2: u128, neg2: bool) -> PackedGlv {
    let mut g = PackedGlv {
        k: [0; 8],
        sign: u32::from(neg1) | (u32::from(neg2) << 1),
    };
    for (half, v) in [(0usize, k1), (4, k2)] {
        for i in 0..4 {
            g.k[half + i] = (v >> (32 * i)) as u32;
        }
    }
    g
}

fn scalar_of<C: GLVConfig<ScalarField = Fr>>(g: &PackedGlv) -> Fr {
    let mag = |half: usize| {
        let mut b = [0u64; 4];
        for (i, w) in b[..2].iter_mut().enumerate() {
            *w = u64::from(g.k[half + 2 * i]) | (u64::from(g.k[half + 2 * i + 1]) << 32);
        }
        Fr::from(BigInt(b))
    };
    let k1 = mag(0);
    let k2 = mag(4);
    let s1 = if g.sign & 1 == 0 { k1 } else { -k1 };
    let s2 = if g.sign & 2 == 0 { k2 } else { -k2 };
    s1 + C::LAMBDA * s2
}

/// One device, one library, one queue and one pipeline per kernel, for the whole file.
///
/// Not tidiness. A pipeline and a command queue per dispatch runs out of something around
/// two thousand of them, and what that looks like is a command buffer that never runs and
/// a destination buffer still full of the zeros it was allocated with, which reads back as
/// the point at infinity and is indistinguishable from a ladder bug.
struct Harness {
    device: Device,
    lib: Library,
    queue: CommandQueue,
    psos: RefCell<HashMap<String, ComputePipelineState>>,
}

impl Harness {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let src = format!("{FR_MSL}\n{MSM_MSL}\n{CEREMONY_MSL}\n{FFT_MSL}\n");
        let lib = device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("MSL compiles");
        let queue = device.new_command_queue();
        Some(Self {
            device,
            lib,
            queue,
            psos: RefCell::new(HashMap::new()),
        })
    }

    fn pso(&self, name: &str) -> ComputePipelineState {
        if let Some(p) = self.psos.borrow().get(name) {
            return p.clone();
        }
        let f = self.lib.get_function(name, None).expect("kernel");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");
        self.psos.borrow_mut().insert(name.to_owned(), p.clone());
        p
    }

    /// One `fft_mix_scale` dispatch of a single thread over `[infinity, p]`, returning
    /// `out[0] == [k] p`.
    fn ladder<T: Copy + g16_metal::layout::Packed>(
        &self,
        kernel: &str,
        p: T,
        tbl: &[PackedGlv; 2],
    ) -> T {
        let pso = self.pso(kernel);
        let zero = [0u8; 256];
        let mut input = zero[..core::mem::size_of::<T>()].to_vec();
        input.extend_from_slice(as_bytes(core::slice::from_ref(&p)));
        let buf = |b: &[u8]| {
            self.device.new_buffer_with_data(
                b.as_ptr().cast(),
                b.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let src = buf(&input);
        let dst = self
            .device
            .new_buffer(input.len() as u64, MTLResourceOptions::StorageModeShared);
        let stw = buf(as_bytes(tbl));
        let params = FftParams {
            n: 2,
            ..Default::default()
        };

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&src), 0);
        enc.set_buffer(1, Some(&dst), 0);
        enc.set_buffer(2, Some(&stw), 0);
        enc.set_bytes(
            3,
            core::mem::size_of::<FftParams>() as u64,
            core::ptr::addr_of!(params).cast(),
        );
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        // A dispatch that did not run leaves `dst` zeroed, which reads back as the point
        // at infinity. Checked so that it is reported as what it is.
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "{kernel}: command buffer did not complete"
        );
        // SAFETY: the command buffer completed and the kernel wrote two points of this type.
        unsafe { *dst.contents().cast::<T>() }
    }
}

/// The magnitudes worth trying: the ends of the 127-bit range the table holds, the
/// borrow boundary of every compiled window width, and a few shapes that make the
/// recoding's carry run the whole length of the value.
fn magnitudes() -> Vec<u128> {
    let mut v = vec![
        0,
        1,
        2,
        (1u128 << 127) - 1,
        (1u128 << 127) - 2,
        1u128 << 126,
        (1u128 << 126) - 1,
        0x5555_5555_5555_5555_5555_5555_5555_5555,
        0x2aaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa,
        0x7fff_ffff_0000_0000_ffff_ffff_0000_0001,
    ];
    // One below, at, and one above each window boundary, which is where the borrow of
    // `2^c` and the carry the neighbour pays it back with have to agree.
    for c in [2u32, 3, 4, 5] {
        for w in [1u32, 2, 25, 31, 42, 63] {
            let off = c * w;
            if off >= 127 {
                continue;
            }
            for d in [0i64, -1, 1] {
                let base = 1u128 << off;
                v.push((base as i128 + i128::from(d)) as u128 & ((1u128 << 127) - 1));
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn pairs() -> Vec<PackedGlv> {
    let mags = magnitudes();
    let mut out = Vec::new();
    for (i, k1) in mags.iter().enumerate() {
        // Pair each magnitude with a different one so the two halves never share a shape,
        // and rotate through all four sign combinations.
        let k2 = mags[(i * 7 + 3) % mags.len()];
        for s in 0..4u32 {
            out.push(entry(*k1, s & 1 != 0, k2, s & 2 != 0));
        }
    }
    out
}

fn pack_g1(p: &G1Affine) -> PackedXyzzG1 {
    PackedXyzzG1 {
        x: PackedFq::from_fq(&p.x),
        y: PackedFq::from_fq(&p.y),
        zz: PackedFq::from_fq(&Fq::from(1u64)),
        zzz: PackedFq::from_fq(&Fq::from(1u64)),
    }
}

fn pack_g2(p: &G2Affine) -> PackedXyzzG2 {
    PackedXyzzG2 {
        x: PackedFq2::from_fq2(&p.x),
        y: PackedFq2::from_fq2(&p.y),
        zz: PackedFq2::from_fq2(&Fq2::new(Fq::from(1u64), Fq::from(0u64))),
        zzz: PackedFq2::from_fq2(&Fq2::new(Fq::from(1u64), Fq::from(0u64))),
    }
}

#[test]
fn the_glv_ladder_agrees_with_arkworks_on_adversarial_digit_streams() {
    let Some(h) = Harness::new() else { return };
    let cases = pairs();

    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    for c in [2u32, 3, 4, 5] {
        for g in &cases {
            let tbl = [*g, *g];

            let k = scalar_of::<g16_field::g1::Config>(g);
            let got = h.ladder(&format!("fft_mix_scale_g1_c{c}"), pack_g1(&g1), &tbl);
            let want = (g1 * k).into_affine();
            let x = got.x.to_fq();
            let zz = got.zz.to_fq();
            let (gx, gy) = if zz == Fq::from(0u64) {
                (Fq::from(0u64), Fq::from(0u64))
            } else {
                (
                    x * zz.inverse().unwrap(),
                    got.y.to_fq() * got.zzz.to_fq().inverse().unwrap(),
                )
            };
            assert_eq!(
                (gx, gy),
                (want.x, want.y),
                "G1 c={c}: k1 {:x?} k2 {:x?} sign {}",
                &g.k[0..4],
                &g.k[4..8],
                g.sign
            );

            let k = scalar_of::<g16_field::g2::Config>(g);
            let got = h.ladder(&format!("fft_mix_scale_g2_c{c}"), pack_g2(&g2), &tbl);
            let want = (g2 * k).into_affine();
            let zz = got.zz.to_fq2();
            let (gx, gy) = if zz == Fq2::new(Fq::from(0u64), Fq::from(0u64)) {
                (
                    Fq2::new(Fq::from(0u64), Fq::from(0u64)),
                    Fq2::new(Fq::from(0u64), Fq::from(0u64)),
                )
            } else {
                (
                    got.x.to_fq2() * zz.inverse().unwrap(),
                    got.y.to_fq2() * got.zzz.to_fq2().inverse().unwrap(),
                )
            };
            assert_eq!(
                (gx, gy),
                (want.x, want.y),
                "G2 c={c}: k1 {:x?} k2 {:x?} sign {}",
                &g.k[0..4],
                &g.k[4..8],
                g.sign
            );
        }
    }
}

/// The point at infinity through the ladder, which is a live input: ptau section 12's
/// `power+1` block is padded with it.
#[test]
fn infinity_survives_the_glv_ladder() {
    let Some(h) = Harness::new() else { return };
    let g = entry(u128::from(u64::MAX), true, 12345, false);
    let out = h.ladder("fft_mix_scale_g1_c5", PackedXyzzG1::default(), &[g, g]);
    assert_eq!(out.zz.v, [0; 8], "infinity came back with a live ZZ");
    assert_eq!(out.zzz.v, [0; 8]);
}
