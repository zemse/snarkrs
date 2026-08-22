//! Adversarial device tests for stages 0 to 4 and the digit pipeline, written by the
//! verifier of U5 to U8 against oracles those units did not use.
//!
//! # Why these exist when U5 to U8 already compare against `g16-core`
//!
//! Every correctness test in `tests/gather.rs`, `tests/ntt.rs` and `tests/stages.rs` checks
//! the device against another implementation *in this repository*: `CpuCircuit::gather`,
//! `CpuNtt`, `CpuCircuit::compute_h`. Those are good oracles and they are not independent
//! ones. `CpuNtt` and the WGSL kernels are the same decimation-in-time algorithm over the
//! same `Domain::twiddles()` table, ported by reading one from the other, so a shared
//! misunderstanding of the convention (which root is forward, where the bit-reverse goes,
//! what the coset shift multiplies) agrees with itself and no test in the crate says a word.
//!
//! So the oracles here are definitional and quadratic:
//!
//! * the transform against `X[j] = sum_i x[i] * g^(i*j)` evaluated term by term;
//! * the coset chain against "interpolate, then evaluate at `shift * g^j`", with the
//!   interpolation also done by a quadratic sum;
//! * `h_std` against `h_mont * R^-1 mod r` in `num-bigint`, which knows nothing about
//!   arkworks' Montgomery form and cannot agree with it about a factor of `R`.
//!
//! # Rules the inputs follow
//!
//! Fixed seeds. Never symmetric: transform inputs are pseudo-random rather than constant,
//! because a constant vector transforms to `(n*c, 0, 0, ...)` and a wrong twiddle index is
//! invisible on the zeros; and the three coset vectors stage 4 reads are asserted pairwise
//! distinct before `H = A*B - C` is checked against them, because `a*a - c` and `a*b - a`
//! are indistinguishable from the right answer when two of them coincide. The previous
//! round's Fq2 bug survived a whole acceptance suite by every input having `a == b`.
//!
//! Domains include 2^0, 2^1 and 2^2, which are smaller than the 32-thread workgroup the
//! generator's clamp floors at, so the worker loop's range check is actually asked a
//! question, and 2^10, which is the first size `split_passes` cuts in two at the shipped
//! cap and therefore the first where the head runs 32 workgroups and the tail reads a
//! strided slice.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use g16_core::cpu::CpuCircuit;
use g16_core::{HPoly, PreparedCircuit as _, StageTimings};
use g16_field::{Domain, Field as _, Fr, One as _, Zero as _};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::{PackedFr, PackedScalar, LIMBS};
use g16_wgpu::gather::{fr_buffer, fr_words};
use g16_wgpu::stages::TAG;
use g16_wgpu::{
    Direction, Epilogue, HStages, LimitsProfile, Ntt, NttTables, ParamRing, Readback, Scale,
    Stage4, Transform, WgpuBackend, WgpuHandle,
};
use g16_zkey::{wtns::Witness, ProvingKey};
use num_bigint::BigUint;

// ---------------------------------------------------------------------------
// Device and artifacts
// ---------------------------------------------------------------------------

fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

fn artifacts() -> Vec<(String, PathBuf)> {
    let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
    else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

const SENTINEL: u32 = 0xffff_ffff;

fn sample(n: usize, seed: u64) -> Vec<Fr> {
    let mut r = SplitMix64(seed);
    (0..n).map(|_| r.next_fr()).collect()
}

/// `n + 1` elements, prefilled with a sentinel so a kernel that wrote nothing cannot agree
/// with an oracle that is zero, then optionally overwritten with `xs`.
fn scratch(label: &str, n: u32, xs: Option<&[Fr]>) -> wgpu::Buffer {
    let b = floor();
    let buf = fr_buffer(b, label, n + 1).expect("scratch buffer");
    let fill = vec![SENTINEL; (n as usize + 1) * LIMBS];
    b.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&fill));
    if let Some(xs) = xs {
        assert_eq!(xs.len(), n as usize);
        b.queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(&fr_words(xs)));
    }
    buf
}

fn read_words(buf: &wgpu::Buffer, n: u32) -> (Vec<u32>, Vec<u32>) {
    let b = floor();
    let bytes = (n as u64 + 1) * (LIMBS * 4) as u64;
    let rb = Readback::new(b, "verify readback", bytes).expect("readback");
    let mut enc = b.device().create_command_encoder(&Default::default());
    rb.copy_from(&mut enc, buf, 0, bytes).expect("copy");
    let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).expect("readback failed");
    let words = bytemuck::cast_slice::<u8, u32>(&raw).to_vec();
    let cut = n as usize * LIMBS;
    (words[..cut].to_vec(), words[cut..].to_vec())
}

fn compare(what: &str, got: &[u32], want: &[Fr]) {
    let want = fr_words(want);
    assert_eq!(got.len(), want.len(), "{what}: wrong length");
    for (i, (g, w)) in got
        .chunks_exact(LIMBS)
        .zip(want.chunks_exact(LIMBS))
        .enumerate()
    {
        assert_eq!(
            g, w,
            "{what}: element {i} is {g:08x?} on the device, {w:08x?} in the oracle"
        );
    }
}

// ---------------------------------------------------------------------------
// The quadratic oracles
// ---------------------------------------------------------------------------

/// `X[j] = sum_i x[i] * root^(i*j)`, term by term.
///
/// Deliberately not a fast transform. A radix-2 reference would share the butterfly, the
/// twiddle indexing and the bit-reverse with the thing it is checking, which is how a
/// convention error hides.
fn naive_dft(x: &[Fr], root: Fr) -> Vec<Fr> {
    let n = x.len();
    (0..n)
        .map(|j| {
            let mut acc = Fr::zero();
            for (i, xi) in x.iter().enumerate() {
                acc += *xi * root.pow([((i * j) % n) as u64]);
            }
            acc
        })
        .collect()
}

/// Runs a chain of transforms on the device and returns the result words plus the slack.
///
/// The buffers ping-pong because the head is out of place by construction: it reads
/// `SRC[reverse(i)]` and writes `DST[i]`, and the reversed index of one workgroup's slice
/// lands in another's.
fn device_chain(log_n: u32, x: &[Fr], steps: &[(Direction, Scale)]) -> (Vec<u32>, Vec<u32>) {
    let b = floor();
    let n = 1u32 << log_n;
    assert_eq!(x.len(), n as usize);
    let ntt = Ntt::new(b, log_n).expect("ntt pipelines");
    let tables = NttTables::new(b, n as usize).expect("ntt tables");

    let v = scratch("verify v", n, Some(x));
    let t = scratch("verify t", n, None);
    let mut ring = ParamRing::new(b, "verify params", (steps.len() * ntt.dispatches()) as u32)
        .expect("parameter ring");
    let mut planned = Vec::with_capacity(steps.len());
    for (i, &(dir, scale)) in steps.iter().enumerate() {
        let (src, dst) = if i % 2 == 0 { (&v, &t) } else { (&t, &v) };
        planned.push(
            ntt.plan(
                b,
                &tables,
                &mut ring,
                &Transform {
                    dir,
                    scale,
                    src,
                    dst,
                    epilogue: Epilogue::Plain,
                },
            )
            .expect("plan"),
        );
    }
    ring.flush(b);
    let mut enc = b.device().create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for p in &planned {
            ntt.encode(&mut pass, p).expect("encode");
        }
    }
    b.queue().submit([enc.finish()]);
    b.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    if let Some(e) = b.take_error() {
        panic!("device error at 2^{log_n}: {e}");
    }
    let out = if steps.len() % 2 == 1 { &t } else { &v };
    read_words(out, n)
}

// ---------------------------------------------------------------------------
// 1. The transform, against the definition
// ---------------------------------------------------------------------------

#[test]
fn the_transform_matches_a_term_by_term_dft_at_every_small_domain() {
    // 2^0 through 2^7, then 2^10. The first three are smaller than the 32-thread floor the
    // generator clamps the workgroup size to, so the worker loop's range check runs; 2^7 is
    // the first that fills a 64-thread workgroup exactly; and 2^10 is the first that
    // `split_passes` cuts into two batches at the shipped cap of 8, so it is the only size
    // here where the head dispatches 32 workgroups and the tail reads a strided slice. A
    // term-by-term DFT is quadratic, so 2^10 is a million multiplies and 2^12 would be
    // sixteen; the strided path is already covered at 2^10 and the artifacts cover 2^18.
    for log_n in (0..=7u32).chain([10]) {
        let n = 1usize << log_n;
        let domain = Domain::new(n).expect("domain");
        let x = sample(n, 0x5eed_0000 + log_n as u64);
        // Never a constant vector: a constant transforms to (n*c, 0, ..), and a wrong
        // twiddle index is invisible on the zeros.
        assert!(
            n == 1 || x.windows(2).any(|w| w[0] != w[1]),
            "2^{log_n}: the input is constant"
        );

        let (fwd, slack) = device_chain(log_n, &x, &[(Direction::Forward, Scale::None)]);
        compare(
            &format!("2^{log_n} forward"),
            &fwd,
            &naive_dft(&x, domain.group_gen),
        );
        assert!(
            slack.iter().all(|&w| w == SENTINEL),
            "2^{log_n} forward: a store index ran one element past the domain"
        );

        let (inv, slack) = device_chain(log_n, &x, &[(Direction::Inverse, Scale::None)]);
        compare(
            &format!("2^{log_n} inverse"),
            &inv,
            &naive_dft(&x, domain.group_gen_inv),
        );
        assert!(
            slack.iter().all(|&w| w == SENTINEL),
            "2^{log_n} inverse: a store index ran one element past the domain"
        );

        // And the normalisation the prover actually fuses in, which is the only thing that
        // makes the inverse an inverse.
        let (round, _) = device_chain(
            log_n,
            &x,
            &[
                (Direction::Inverse, Scale::SizeInv),
                (Direction::Forward, Scale::None),
            ],
        );
        compare(&format!("2^{log_n} round trip"), &round, &x);
    }
    println!("domains 2^0..2^7 and 2^10, forward and inverse, against a term-by-term DFT");
}

// ---------------------------------------------------------------------------
// 2. The coset chain, against interpolate-then-evaluate
// ---------------------------------------------------------------------------

#[test]
fn the_coset_chain_evaluates_the_interpolant_on_the_shifted_domain() {
    for log_n in [1u32, 2, 3, 5, 6, 10] {
        let n = 1usize << log_n;
        let domain = Domain::new(n).expect("domain");
        let tables = NttTables::new(floor(), n).expect("tables");
        let shift = tables.coset_shift();
        // The shift is a primitive 2n-th root, not the domain's own generator. If it were
        // the latter the coset would be the domain and this test would be vacuous.
        assert_eq!(
            shift * shift,
            domain.group_gen,
            "2^{log_n}: wrong coset shift"
        );

        let y = sample(n, 0xc0_5e70_0000 + log_n as u64);

        // Coefficients of the interpolant, by the definition: c[i] = (1/n) sum_j y[j] g^-ij.
        let mut coeffs = naive_dft(&y, domain.group_gen_inv);
        for c in &mut coeffs {
            *c *= domain.size_inv;
        }
        // Evaluated on the shifted domain, also by the definition.
        let want: Vec<Fr> = (0..n)
            .map(|j| {
                let pt = shift * domain.group_gen.pow([j as u64]);
                let mut acc = Fr::zero();
                let mut p = Fr::one();
                for c in &coeffs {
                    acc += *c * p;
                    p *= pt;
                }
                acc
            })
            .collect();

        let (got, slack) = device_chain(
            log_n,
            &y,
            &[
                (Direction::Inverse, Scale::SizeInv),
                (Direction::Forward, Scale::CosetPowers),
            ],
        );
        compare(&format!("2^{log_n} coset"), &got, &want);
        assert!(
            slack.iter().all(|&w| w == SENTINEL),
            "2^{log_n} coset: a store index ran one element past the domain"
        );
    }
    println!("the iNTT -> coset -> NTT chain evaluates the interpolant at shift * g^j");
}

// ---------------------------------------------------------------------------
// 3. Montgomery against standard, in an arithmetic that knows nothing about either
// ---------------------------------------------------------------------------

fn to_big(words: &[u32]) -> BigUint {
    BigUint::from_slice(words)
}

/// BN254's `r` and `R = 2^256 mod r`, built from `g16-field` rather than retyped.
fn modulus_and_r() -> (BigUint, BigUint) {
    let r_modulus = to_big(&PackedScalar::from_fr(&-Fr::one()).v) + 1u32;
    // PackedFr is the Montgomery representative, so the Montgomery form of 1 *is* R mod r.
    let r = to_big(&PackedFr::from_fr(&Fr::one()).v);
    (r_modulus, r)
}

#[test]
fn h_std_is_h_mont_divided_by_r_in_an_independent_arithmetic() {
    let b = floor();
    let found = artifacts();
    assert!(
        !found.is_empty(),
        "no artifacts under bench/artifacts; the symlink into the main worktree is missing \
         and this test would otherwise pass by doing nothing"
    );
    let (modulus, r) = modulus_and_r();
    // R^-1 mod r, by Fermat: r is prime, so R^(r-2) is the inverse. Computed in num-bigint,
    // so nothing about arkworks' Montgomery form is being trusted here.
    let r_inv = r.modpow(&(&modulus - 2u32), &modulus);
    assert_eq!((&r * &r_inv) % &modulus, BigUint::from(1u32));

    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let stages = HStages::new(b, &pk).expect("stages");
        for mode in [Stage4::Standalone, Stage4::Fused] {
            let h = pollster::block_on(stages.compute_h_with(
                b,
                &w,
                mode,
                &mut StageTimings::default(),
            ))
            .expect("compute_h");
            let g = h.device_handle::<WgpuHandle>(TAG).expect("wgpu handle");
            let mont = pollster::block_on(g.h_mont_words(b)).expect("h_mont");
            let std = pollster::block_on(g.h_std_words(b)).expect("h_std");
            assert_eq!(mont.len(), std.len());

            let mut differed = 0usize;
            for (i, (m, s)) in mont
                .chunks_exact(LIMBS)
                .zip(std.chunks_exact(LIMBS))
                .enumerate()
            {
                let mb = to_big(m);
                let sb = to_big(s);
                assert!(
                    mb < modulus && sb < modulus,
                    "{name} {mode:?}: element {i} is not a canonical residue"
                );
                assert_eq!(
                    sb,
                    (&mb * &r_inv) % &modulus,
                    "{name} {mode:?}: element {i}: h_std is not h_mont / R. A swap of the two \
                     buffers is a proof wrong by a factor of R that verifies as false with \
                     nothing else to go on."
                );
                if m != s {
                    differed += 1;
                }
            }
            // Without this the assertion above would also pass on a kernel that wrote the
            // same thing into both buffers, if H happened to be zero everywhere.
            assert!(
                differed * 2 > mont.len() / LIMBS,
                "{name} {mode:?}: only {differed} of {} elements differ between the two \
                 encodings, so this check is nearly vacuous",
                mont.len() / LIMBS
            );
            drop(h);
        }
        println!("{name}: h_std == h_mont * R^-1 mod r on both stage 4 paths");
    }
}

// ---------------------------------------------------------------------------
// 4. Concurrency, across both stage 4 paths at once
// ---------------------------------------------------------------------------

#[test]
fn one_circuit_proves_concurrently_across_both_stage_four_paths() {
    let b = floor();
    let Some((name, dir)) = artifacts().into_iter().find(|(n, _)| n == "js_1x1_d8") else {
        panic!("js_1x1_d8 is missing from bench/artifacts");
    };
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
    let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;

    // Six distinct witnesses, so a proof that picked up another thread's scratch or another
    // thread's parameter block produces a different answer rather than the same one.
    let mut rng = SplitMix64(0x9e37_79b9_7f4a_7c15);
    let ws: Vec<Vec<Fr>> = (0..6)
        .map(|k| {
            let mut v = w.clone();
            // Perturb everything past the public inputs, which is what a real witness varies.
            for (j, x) in v.iter_mut().enumerate().skip(1) {
                if j % 3 == k % 3 {
                    *x += rng.next_fr();
                }
            }
            v
        })
        .collect();
    assert!(
        ws.windows(2).all(|p| p[0] != p[1]),
        "the six witnesses are not distinct"
    );

    let cpu = CpuCircuit::new(ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey"))
        .expect("cpu circuit");
    let want: Vec<Vec<u32>> = ws
        .iter()
        .map(
            |w| match cpu.compute_h(w, &mut StageTimings::default()).unwrap() {
                HPoly::Host(v) => fr_words(&v),
                HPoly::Device { .. } => unreachable!(),
            },
        )
        .collect();
    assert!(
        want.windows(2).all(|p| p[0] != p[1]),
        "the six H polynomials are not distinct, so a mixed-up scratch would be invisible"
    );

    let stages = &HStages::new(b, &pk).expect("stages");
    std::thread::scope(|s| {
        for (k, (w, want)) in ws.iter().zip(&want).enumerate() {
            s.spawn(move || {
                // Alternating modes, so the two dispatch shapes and the two ring-slot
                // counts are in flight at the same time.
                let mode = if k % 2 == 0 {
                    Stage4::Standalone
                } else {
                    Stage4::Fused
                };
                let h = pollster::block_on(stages.compute_h_with(
                    b,
                    w,
                    mode,
                    &mut StageTimings::default(),
                ))
                .expect("compute_h");
                let g = h.device_handle::<WgpuHandle>(TAG).unwrap();
                let got = pollster::block_on(g.h_mont_words(b)).unwrap();
                assert_eq!(
                    &got, want,
                    "thread {k} ({mode:?}) disagreed with the CPU backend"
                );
            });
        }
    });
    println!(
        "{name}: 6 concurrent proofs, alternating Fused and Standalone, {} scratch sets \
         pooled after",
        stages.pooled()
    );
}

// ---------------------------------------------------------------------------
// 5. H against the three buffers stage 4 was actually bound to, on a real key
// ---------------------------------------------------------------------------

/// Every other check on stage 4 compares `H` to a whole `CpuCircuit::compute_h`, which
/// recomputes A, B and C for itself. So a stage 4 that read the wrong pair of device buffers
/// would have to be caught by the transform disagreeing, not by stage 4 disagreeing. This
/// closes that: it reads back the three coset vectors the kernel was bound to and checks
/// `H = A*B - C` against those, on real keys.
///
/// The three are asserted pairwise distinct first, because `a*a - c`, `a*b - a` and
/// `a*b - b` are all indistinguishable from the right answer when two of the vectors
/// coincide, and the previous round's Fq2 bug survived a whole suite for exactly that
/// reason. A's coset evaluations are also asserted mostly nonzero, so the comparison is not
/// passing on a buffer nobody wrote.
#[test]
fn h_is_a_times_b_minus_c_against_the_three_buffers_the_kernel_was_bound_to() {
    let b = floor();
    let found = artifacts();
    assert!(!found.is_empty(), "no artifacts under bench/artifacts");
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let n = pk.domain_size;
        let stages = HStages::new(b, &pk).expect("stages");
        // Standalone and not Fused, because only the standalone path leaves C holding its
        // finished forward transform. The fused path writes H instead of C, which is the
        // write the fusion exists to save, so there would be no C to compare against.
        let h = pollster::block_on(stages.compute_h_with(
            b,
            &w,
            Stage4::Standalone,
            &mut StageTimings::default(),
        ))
        .expect("compute_h");
        let g = h.device_handle::<WgpuHandle>(TAG).expect("handle");
        let (ba, bb, bc) = g.coset();
        let bytes = n as u64 * (LIMBS * 4) as u64;
        let read = |buf: &wgpu::Buffer| {
            let rb = Readback::new(b, "coset", bytes).unwrap();
            let mut enc = b.device().create_command_encoder(&Default::default());
            rb.copy_from(&mut enc, buf, 0, bytes).unwrap();
            let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).unwrap();
            bytemuck::cast_slice::<u8, u32>(&raw).to_vec()
        };
        let (a, bv, c) = (read(ba), read(bb), read(bc));
        assert_ne!(a, bv, "{name}: A and B coincide on the coset");
        assert_ne!(a, c, "{name}: A and C coincide on the coset");
        assert_ne!(bv, c, "{name}: B and C coincide on the coset");
        let nonzero = a
            .chunks_exact(LIMBS)
            .filter(|e| e.iter().any(|&x| x != 0))
            .count();
        assert!(
            nonzero * 2 > n,
            "{name}: only {nonzero} of {n} coset evaluations of A are nonzero, so the \
             comparison below would pass on a kernel that wrote nothing"
        );
        // h = a*b - c elementwise, checked against the two buffers actually bound, so a
        // stage 4 that read the wrong pair of inputs shows up here even if the CPU oracle
        // happened to agree.
        let mont = pollster::block_on(g.h_mont_words(b)).expect("h_mont");
        for i in 0..n {
            let get = |v: &[u32]| {
                let mut t = [0u32; LIMBS];
                t.copy_from_slice(&v[i * LIMBS..(i + 1) * LIMBS]);
                PackedFr { v: t }.to_fr()
            };
            let want = get(&a) * get(&bv) - get(&c);
            assert_eq!(
                &mont[i * LIMBS..(i + 1) * LIMBS],
                &PackedFr::from_fr(&want).v[..],
                "{name}: h[{i}] is not a[{i}]*b[{i}] - c[{i}]"
            );
        }
        drop(h);
        println!("{name}: h == a*b - c against the three coset buffers the kernel was bound to");
    }
}

// ---------------------------------------------------------------------------
// 6. Every dynamic offset is 256-byte aligned, which native does not require
// ---------------------------------------------------------------------------

/// `minStorageBufferOffsetAlignment` is 32 on this adapter and 256 in every browser, and
/// `minUniformBufferOffsetAlignment` is likewise stricter on the web. Every storage binding
/// in this crate is `as_entire_binding()`, so its offset is 0 and the question does not
/// arise; the uniform ring is the one place an offset is computed, and it reads its stride
/// from the *granted* limits rather than from a literal, which is right on `Raised` and is
/// the thing that has to be pinned on `Floor`.
#[test]
fn every_uniform_dynamic_offset_is_a_multiple_of_256_at_the_floor() {
    let b = floor();
    let limits = b.granted_limits();
    assert_eq!(
        limits.min_uniform_buffer_offset_alignment, 256,
        "the Floor profile granted a {}-byte uniform alignment; a ring built on that passes \
         natively and fails in Chrome, which never goes below 256",
        limits.min_uniform_buffer_offset_alignment
    );
    assert_eq!(
        limits.min_storage_buffer_offset_alignment, 256,
        "the Floor profile granted a {}-byte storage alignment",
        limits.min_storage_buffer_offset_alignment
    );

    let mut ring = ParamRing::new(b, "alignment probe", 40).expect("ring");
    assert_eq!(ring.stride(), 256);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Block {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }
    for i in 0..40u32 {
        let off = ring
            .push(&Block {
                a: i,
                b: i,
                c: i,
                d: i,
            })
            .expect("push");
        assert_eq!(
            off % 256,
            0,
            "slot {i} landed at offset {off}, which is not 256-byte aligned"
        );
        assert_eq!(off, i * 256, "slot {i} is not where the stride puts it");
    }
    // One past the end has to be an error rather than an overwrite of slot 0.
    assert!(
        ring.push(&Block {
            a: 0,
            b: 0,
            c: 0,
            d: 0
        })
        .is_err(),
        "a full ring accepted a 41st block"
    );
    println!("40 ring slots, every dynamic offset a multiple of 256");
}

// ---------------------------------------------------------------------------
// 7. The shipped tile is one the generator will actually emit, at both profiles
// ---------------------------------------------------------------------------

/// The generator's ceiling and the host validator's ceiling are the same number, at both
/// profiles.
///
/// # What this used to allow
///
/// `gen::ntt::ntt_module_at` asserts the tile fits **16384** bytes, the browser floor's
/// `maxComputeWorkgroupStorageSize`, so `k = 10` (32768 bytes) always panics.
/// `Ntt::max_fused` used to divide the **granted** limit instead, and this adapter grants
/// 32768, so under `Raised` it returned 10 and `Ntt::with_shape(b, log_n, 10, None)` passed
/// its own `max_fused <= ceiling` check and then panicked inside the generator where a
/// `ProveError` was the documented behaviour. Nothing shipping reached it, because
/// `preferred_fused` is `min(max_fused, 8)`, which is why it was a wart and not a bug.
///
/// U11 closed it: both sides now read `WgpuBackend::ceiling_workgroup_bytes`, which is the
/// smaller of the granted limit and the floor. So this asserts three things rather than
/// printing one of them:
///
/// 1. the tile the prover asks for is one the generator will emit, at every profile;
/// 2. the tile `max_fused` reports as the ceiling is *also* one the generator will emit,
///    which is the half that used to be false; and
/// 3. asking for one pass more than the ceiling is a `ProveError` and not a panic.
#[test]
fn the_shipped_ntt_tile_is_one_the_generator_will_emit_at_both_profiles() {
    /// Bytes the generator refuses above. Deliberately a literal here and read from
    /// `gen::ntt`'s own assertion message there, so the two are compared rather than shared.
    const GENERATOR_CEILING_BYTES: u64 = 16384;
    for profile in [LimitsProfile::Floor, LimitsProfile::Raised] {
        let b = pollster::block_on(WgpuBackend::with_profile(profile))
            .unwrap_or_else(|e| panic!("no device at {profile:?}: {e}"));
        let shipped = Ntt::preferred_fused(&b);
        let ceiling = Ntt::max_fused(&b);
        let bytes = g16_wgpu::gen::ntt::workgroup_bytes(shipped, g16_wgpu::gen::Variant::default());
        assert!(
            bytes <= GENERATOR_CEILING_BYTES,
            "{profile:?}: the prover asks for a k = {shipped} tile, which is {bytes} bytes, \
             and the generator refuses anything over {GENERATOR_CEILING_BYTES}"
        );
        // And the shape it produces really does build, which is the end-to-end version of
        // the same statement.
        Ntt::with_shape(&b, 12, shipped, None).expect("the shipped tile did not build");

        // The half that used to be printed as a NOTE. `max_fused` is documented as the
        // largest `k` a caller may pass, so a caller who believes it must not hit a panic.
        let ceiling_bytes =
            g16_wgpu::gen::ntt::workgroup_bytes(ceiling, g16_wgpu::gen::Variant::default());
        assert!(
            ceiling_bytes <= GENERATOR_CEILING_BYTES,
            "{profile:?}: Ntt::max_fused reports {ceiling}, which is {ceiling_bytes} bytes and \
             over the generator's {GENERATOR_CEILING_BYTES}. with_shape would accept it and \
             the generator would panic instead of returning ProveError."
        );
        Ntt::with_shape(&b, 12, ceiling, None).unwrap_or_else(|e| {
            panic!(
                "{profile:?}: the reported ceiling k = {ceiling} did \
                 not build: {e}"
            )
        });

        // One past the ceiling is an error, not a panic. This is the statement that a
        // `ProveError` is still the documented failure mode after the two were unified.
        let over = Ntt::with_shape(&b, 12, ceiling + 1, None);
        assert!(
            over.is_err(),
            "{profile:?}: with_shape accepted k = {} , one past its own ceiling",
            ceiling + 1
        );

        // Same statement for the invocation ceiling, which had the identical shape in five
        // places: 1024 granted here, 256 asserted by every generator.
        assert_eq!(
            b.ceiling_invocations(),
            256,
            "{profile:?}: the invocation ceiling has to be the floor's 256, whatever the \
             adapter grants"
        );
        assert!(
            Ntt::with_shape(&b, 12, shipped, Some(512)).is_err(),
            "{profile:?}: a 512-thread NTT workgroup is over the floor and must be refused \
             here rather than panic in the generator"
        );

        println!(
            "{profile:?}: granted workgroup storage {} B, ceiling {} B, max_fused {ceiling}, \
             shipped tile k = {shipped} ({bytes} B)",
            b.granted_limits().max_compute_workgroup_storage_size,
            b.ceiling_workgroup_bytes(),
        );
    }
}
