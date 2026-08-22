//! U6's acceptance tests: stages 1 to 3 on a real GPU, against `g16-ntt` and against the
//! browser floor.
//!
//! # What is being pinned
//!
//! 1. **Every domain from 2^3 to 2^18**, forward and inverse, elementwise against
//!    `g16_ntt::CpuNtt`, which is the same transform the CPU backend proves with. Not a
//!    round trip: `iNTT(NTT(x)) == x` holds for any pair of mutually inverse maps, including
//!    a pair that both apply the wrong permutation, so it is checked as well but it is never
//!    the only check.
//! 2. **The coset shift**, which is the one part of stages 1 to 3 that is not a transform.
//!    Checked as the whole `iNTT -> shift -> NTT` chain the prover runs, against the CPU
//!    backend doing the same three steps.
//! 3. **The fused stage 4 epilogue.** U7 owns wiring it, but this unit generates it, and a
//!    kernel that ships untested until the unit after next is a kernel nobody checks.
//! 4. **The floor, programmatically.** Workgroup storage, workgroup size and storage buffers
//!    per pipeline layout, computed from the generator rather than observed to work here.
//!
//! # The input rules, which exist because the last round's verifier found bugs that hid
//!
//! **Never symmetric.** The join test's `a` and `b` are different vectors, so an operand
//! swap or an `a*a` has to change a number. The transform inputs are pseudo-random from a
//! fixed seed, never constant and never a delta, because a constant vector transforms to
//! `(n*c, 0, 0, ...)` and a wrong twiddle index is invisible on the zeros.
//!
//! **Outputs start as a sentinel, never zero,** with one slack element past the domain. A
//! kernel that wrote nothing would otherwise agree with the oracle wherever the oracle is
//! zero, and an off-by-one in the store index would land somewhere already zero.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::sync::OnceLock;

use g16_field::{Domain, Field as _, Fr};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::LIMBS;
use g16_ntt::{CpuNtt, Direction as CpuDirection, NttBackend as _};
use g16_wgpu::gather::{fr_buffer, fr_words};
use g16_wgpu::gen::ntt as wgsl;
use g16_wgpu::gen::ntt::Mode;
use g16_wgpu::{Direction, Epilogue, LimitsProfile, Ntt, NttTables, ParamRing, Readback};
use g16_wgpu::{Scale, Transform, WgpuBackend};

// ---------------------------------------------------------------------------
// Device, built once for the whole binary
// ---------------------------------------------------------------------------

fn floor() -> &'static WgpuBackend {
    static B: OnceLock<WgpuBackend> = OnceLock::new();
    B.get_or_init(|| {
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile")
    })
}

/// The sentinel every device buffer starts as. All-ones is above the modulus, so it is not a
/// value any correct transform can produce, and it is not the zero a bug would coincide with.
const SENTINEL: u32 = 0xffff_ffff;

/// Pseudo-random field elements from a fixed seed, so a failure reproduces.
fn sample(n: usize, seed: u64) -> Vec<Fr> {
    let mut r = SplitMix64(seed);
    (0..n).map(|_| r.next_fr()).collect()
}

/// A device buffer of `n + 1` elements, prefilled with [`SENTINEL`] and then optionally
/// overwritten with `xs`.
///
/// The extra element is slack: nothing should ever write it, and [`check_slack`] says so.
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

/// Reads `n + 1` elements back and splits off the slack element.
fn read(buf: &wgpu::Buffer, n: u32) -> (Vec<u32>, Vec<u32>) {
    let b = floor();
    let bytes = (n as u64 + 1) * (LIMBS * 4) as u64;
    let rb = Readback::new(b, "ntt readback", bytes).expect("readback");
    let mut enc = b.device().create_command_encoder(&Default::default());
    rb.copy_from(&mut enc, buf, 0, bytes).expect("copy");
    let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).expect("readback failed");
    let words = bytemuck::cast_slice::<u8, u32>(&raw).to_vec();
    let cut = n as usize * LIMBS;
    (words[..cut].to_vec(), words[cut..].to_vec())
}

fn check_slack(what: &str, slack: &[u32]) {
    assert!(
        slack.iter().all(|&x| x == SENTINEL),
        "{what}: the kernel wrote one element past the domain, so a store index is off by \
         one (slack is {slack:08x?})"
    );
}

/// Elementwise, in device words, naming the element that first disagreed.
fn compare(what: &str, got: &[u32], want: &[Fr]) {
    let want = fr_words(want);
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: {} words back, oracle has {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got
        .chunks_exact(LIMBS)
        .zip(want.chunks_exact(LIMBS))
        .enumerate()
    {
        assert_eq!(
            g, w,
            "{what}: element {i} is {g:08x?} on the device, {w:08x?} on the CPU"
        );
    }
}

// ---------------------------------------------------------------------------
// Running a chain of transforms
// ---------------------------------------------------------------------------

/// One transform of a chain: which twiddle table and what the head scales by on load.
#[derive(Clone, Copy)]
struct Step(Direction, Scale);

/// Uploads `x`, runs `steps` back to back in one submit, reads the result back.
///
/// The buffers ping-pong, because the head is out of place by construction: it reads
/// `SRC[reverse(i)]` and the reversed index of one workgroup's slice lands in another's, so
/// an in-place head would read values a different workgroup had already overwritten.
///
/// Returns the result words, the slack element of the buffer it landed in, and the dispatch
/// count, so a test can assert the batch split it thought it was exercising actually ran.
fn run_chain(
    log_n: u32,
    max_fused: Option<u32>,
    workgroup: Option<u32>,
    x: &[Fr],
    steps: &[Step],
) -> (Vec<u32>, Vec<u32>, usize) {
    let b = floor();
    let n = 1u32 << log_n;
    assert_eq!(x.len(), n as usize);

    let ntt = Ntt::with_shape(
        b,
        log_n,
        max_fused.unwrap_or_else(|| Ntt::preferred_fused(b)),
        workgroup,
    )
    .expect("ntt pipelines");
    let tables = NttTables::new(b, n as usize).expect("ntt tables");

    let mut v = scratch("ntt v", n, Some(x));
    let mut t = scratch("ntt t", n, None);

    let mut ring = ParamRing::new(b, "ntt params", (steps.len() * ntt.dispatches()) as u32)
        .expect("parameter ring");
    let mut planned = Vec::with_capacity(steps.len());
    for (i, s) in steps.iter().enumerate() {
        let (src, dst) = if i % 2 == 0 { (&v, &t) } else { (&t, &v) };
        planned.push(
            ntt.plan(
                b,
                &tables,
                &mut ring,
                &Transform {
                    dir: s.0,
                    scale: s.1,
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
        panic!("device error running {} transforms: {e}", steps.len());
    }

    // The last step wrote whichever buffer it had as `dst`.
    let out = if steps.len() % 2 == 1 { &mut t } else { &mut v };
    let (words, slack) = read(out, n);
    (words, slack, steps.len() * ntt.dispatches())
}

/// `g16-ntt`'s answer to the same chain, in the device's own terms.
///
/// Two places where this is not simply a call to `CpuNtt::ntt`, and both are the fusions
/// design §8 U6 asks for rather than sleight of hand:
///
/// * The device fuses the load scale in front of the transform; the CPU applies it as its
///   own pass. The transform is `Fr`-linear, so the two are the same map, and that identity
///   is exactly what makes the fusion legal.
/// * `CpuNtt`'s inverse applies `1/n` itself and the device's never does: on the device the
///   normalisation *is* [`Scale::SizeInv`], which is why the prover passes it. So the CPU's
///   own copy is undone here unconditionally, and a device inverse driven with
///   `Scale::SizeInv` then lands back on plain `CpuNtt::ntt(Inverse)`. The test below
///   asserts that agreement directly rather than trusting this comment.
fn cpu_chain(domain: &Domain, coset_shift: Fr, x: &[Fr], steps: &[Step]) -> Vec<Fr> {
    let ntt = CpuNtt::new();
    let n = Fr::from(domain.size as u64);
    let mut v = x.to_vec();
    for s in steps {
        match s.1 {
            Scale::None => {}
            Scale::SizeInv => v.iter_mut().for_each(|e| *e *= domain.size_inv),
            Scale::CosetPowers => ntt.distribute_powers(&mut v, coset_shift),
        }
        let dir = match s.0 {
            Direction::Forward => CpuDirection::Forward,
            Direction::Inverse => CpuDirection::Inverse,
        };
        ntt.ntt(domain, &mut v, dir);
        if dir == CpuDirection::Inverse {
            v.iter_mut().for_each(|e| *e *= n);
        }
    }
    v
}

// ---------------------------------------------------------------------------
// 1. Every domain, both directions, against g16-ntt
// ---------------------------------------------------------------------------

#[test]
fn forward_and_inverse_transforms_match_g16_ntt_at_every_domain() {
    let mut compile_us = 0u64;
    let before = floor().prepare_cost();
    // Down to 2^0, three sizes below what §8 U6 asks for, because the degenerate domain is
    // where the head's `32 - log_n` shift would be a shift by the full word width and WGSL
    // leaves that indeterminate. The generator masks it; nothing else would notice.
    for log_n in 0..=18u32 {
        let n = 1usize << log_n;
        let domain = Domain::new(n).unwrap();
        let x = sample(n, 0xC0FFEE ^ log_n as u64);
        // A transform of a constant or a delta hides a wrong twiddle index behind zeros, so
        // the input is asserted to be neither before anything is compared. A one- or
        // two-point domain has no room for that and is exempt.
        assert!(
            n < 4 || x.windows(2).any(|w| w[0] != w[1]),
            "2^{log_n}: the input is constant, which would make this test vacuous"
        );

        // The inverse carries its 1/n as the load scale, because that is what the iNTT is:
        // `CpuNtt` applies it after the passes and the device before them, and the two agree
        // only if the device is driven the way the prover drives it.
        for (name, dir, scale) in [
            ("forward", Direction::Forward, Scale::None),
            ("inverse", Direction::Inverse, Scale::SizeInv),
        ] {
            let steps = [Step(dir, scale)];
            let (got, slack, dispatches) = run_chain(log_n, None, None, &x, &steps);
            let want = cpu_chain(&domain, Fr::ONE, &x, &steps);

            // The oracle checked against `g16-ntt` called plainly, so a mistake in
            // `cpu_chain`'s scale bookkeeping cannot excuse a wrong kernel.
            let mut direct = x.clone();
            CpuNtt::new().ntt(
                &domain,
                &mut direct,
                match dir {
                    Direction::Forward => CpuDirection::Forward,
                    Direction::Inverse => CpuDirection::Inverse,
                },
            );
            assert_eq!(want, direct, "2^{log_n} {name}: cpu_chain is not g16-ntt");
            // A one-point transform really is the identity, so only the sizes above it can
            // make this claim.
            assert!(
                n == 1 || want != x,
                "2^{log_n} {name}: the oracle returned its input, so this proves nothing"
            );
            compare(&format!("2^{log_n} {name}"), &got, &want);
            check_slack(&format!("2^{log_n} {name}"), &slack);
            if log_n == 18 {
                // 6 + 6 + 6 and not the design's 9 + 9: see gen::ntt::PREFERRED_FUSED.
                assert_eq!(
                    dispatches, 3,
                    "2^18 is three batches of six at the shipped shape"
                );
            }
        }

        // Round trip, kept because it is cheap, never relied on alone: a pair of mutually
        // inverse wrong maps passes it.
        let (got, _, _) = run_chain(
            log_n,
            None,
            None,
            &x,
            &[
                Step(Direction::Forward, Scale::None),
                Step(Direction::Inverse, Scale::SizeInv),
            ],
        );
        compare(&format!("2^{log_n} round trip"), &got, &x);
    }
    let after = floor().prepare_cost();
    compile_us += after.compile_us - before.compile_us;
    println!(
        "domains 2^3..2^18: {} modules + {} pipelines in {:.0} ms",
        after.modules - before.modules,
        after.pipelines - before.pipelines,
        compile_us as f64 / 1e3
    );
}

// ---------------------------------------------------------------------------
// 2. The coset shift, as the prover actually runs it
// ---------------------------------------------------------------------------

#[test]
fn the_coset_shift_path_matches_the_cpu_coset_shift() {
    for log_n in [3u32, 4, 7, 10, 13, 16, 18] {
        let n = 1usize << log_n;
        let domain = Domain::new(n).unwrap();
        let coset_shift = Domain::new(2 * n).unwrap().group_gen;
        let x = sample(n, 0xC05E7 ^ log_n as u64);

        // Stages 1 to 3 for one vector: iNTT with 1/n on the load, then the coset shift on
        // the load of the forward transform.
        let steps = [
            Step(Direction::Inverse, Scale::SizeInv),
            Step(Direction::Forward, Scale::CosetPowers),
        ];
        let (got, slack, dispatches) = run_chain(log_n, None, None, &x, &steps);
        let want = cpu_chain(&domain, coset_shift, &x, &steps);
        compare(&format!("2^{log_n} coset chain"), &got, &want);
        check_slack(&format!("2^{log_n} coset chain"), &slack);

        // The shift is the load scale of the second transform and nothing else, so the same
        // chain without it must disagree. Otherwise the table could be all ones and every
        // assertion above would still pass.
        let unshifted = cpu_chain(
            &domain,
            coset_shift,
            &x,
            &[
                Step(Direction::Inverse, Scale::SizeInv),
                Step(Direction::Forward, Scale::None),
            ],
        );
        assert_ne!(
            want, unshifted,
            "2^{log_n}: the coset shift changed nothing, so this test cannot see it"
        );

        // The tables really are a primitive 2n-th root's powers and not the domain's own.
        assert_eq!(coset_shift * coset_shift, domain.group_gen, "2^{log_n}");

        println!(
            "2^{log_n}: coset chain over {n} points, {dispatches} dispatches, shift^2 == group_gen"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. The pass split does not change the answer
// ---------------------------------------------------------------------------

#[test]
fn every_pass_split_computes_the_same_transform() {
    let log_n = 10u32;
    let n = 1usize << log_n;
    let domain = Domain::new(n).unwrap();
    let coset_shift = Domain::new(2 * n).unwrap().group_gen;
    let x = sample(n, 0x59117);
    let steps = [
        Step(Direction::Inverse, Scale::SizeInv),
        Step(Direction::Forward, Scale::CosetPowers),
    ];
    let want = cpu_chain(&domain, coset_shift, &x, &steps);

    let ceiling = Ntt::max_fused(floor());
    let mut seen = Vec::new();
    // Every split up to the memory ceiling, including the k = 9 the shipped configuration
    // does not use: it has to stay correct even though it is 4.7x slower at 2^18.
    for max_fused in 1..=ceiling {
        let (got, slack, dispatches) = run_chain(log_n, Some(max_fused), None, &x, &steps);
        compare(&format!("max_fused {max_fused}"), &got, &want);
        check_slack(&format!("max_fused {max_fused}"), &slack);
        seen.push((max_fused, dispatches));
    }
    // 1 pass per batch is ten batches per transform, 9 is two. If every split produced the
    // same dispatch count the loop would only ever have exercised one code path.
    let counts: Vec<usize> = seen.iter().map(|(_, d)| *d).collect();
    assert!(
        counts.iter().min() != counts.iter().max(),
        "every max_fused produced {counts:?} dispatches, so the strided tail never ran"
    );
    println!("2^{log_n} chain, (max_fused, dispatches): {seen:?}");
}

// ---------------------------------------------------------------------------
// 4. The fused stage 4 epilogue
// ---------------------------------------------------------------------------

#[test]
fn the_fused_join_epilogue_writes_h_in_both_encodings() {
    let b = floor();
    // 2^7 is a single batch at the floor, so the head itself joins; 2^13 is two, so the
    // joined batch is a strided tail. Both paths, and they are different entry points.
    for log_n in [7u32, 13] {
        let n = 1usize << log_n;
        let nn = n as u32;
        let domain = Domain::new(n).unwrap();
        let coset_shift = Domain::new(2 * n).unwrap().group_gen;

        // Deliberately three different vectors. The last round's Fq2 bug survived because
        // every test input had a == b.
        let x = sample(n, 0x101 ^ log_n as u64);
        let ja = sample(n, 0x202 ^ log_n as u64);
        let jb = sample(n, 0x303 ^ log_n as u64);
        assert!(
            ja.iter().zip(&jb).all(|(p, q)| p != q),
            "ja == jb somewhere"
        );

        let ntt = Ntt::new(b, log_n).expect("ntt pipelines");
        let tables = NttTables::new(b, n).expect("ntt tables");
        assert_eq!(
            ntt.dispatches() == 1,
            log_n <= Ntt::preferred_fused(b),
            "2^{log_n}: the batch count is not what this test assumed"
        );

        let src = scratch("join src", nn, Some(&x));
        let dst = scratch("join dst", nn, None);
        let a = scratch("join a", nn, Some(&ja));
        let bb = scratch("join b", nn, Some(&jb));
        let h_mont = scratch("h_mont", nn, None);
        let h_std = scratch("h_std", nn, None);

        let mut ring = ParamRing::new(b, "join params", ntt.dispatches() as u32).expect("ring");
        let plan = ntt
            .plan(
                b,
                &tables,
                &mut ring,
                &Transform {
                    dir: Direction::Forward,
                    scale: Scale::CosetPowers,
                    src: &src,
                    dst: &dst,
                    epilogue: Epilogue::Join {
                        a: &a,
                        b: &bb,
                        h_mont: &h_mont,
                        h_std: &h_std,
                    },
                },
            )
            .expect("plan");
        ring.flush(b);

        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            ntt.encode(&mut pass, &plan).expect("encode");
        }
        b.queue().submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        if let Some(e) = b.take_error() {
            panic!("device error on the join path: {e}");
        }

        // The transform the join replaced, computed by the CPU, then stage 4 on top.
        let transformed = cpu_chain(
            &domain,
            coset_shift,
            &x,
            &[Step(Direction::Forward, Scale::CosetPowers)],
        );
        let want_mont: Vec<Fr> = transformed
            .iter()
            .zip(&ja)
            .zip(&jb)
            .map(|((c, p), q)| *p * q - c)
            .collect();

        let (mont, mont_slack) = read(&h_mont, nn);
        let (std, std_slack) = read(&h_std, nn);
        compare(&format!("2^{log_n} h_mont"), &mont, &want_mont);
        check_slack(&format!("2^{log_n} h_mont"), &mont_slack);
        check_slack(&format!("2^{log_n} h_std"), &std_slack);

        // h_std is the same value out of Montgomery form, which is what a window digit of a
        // Pippenger scalar has to be taken from. Compared against arkworks' own
        // `into_bigint`, not against a second device kernel.
        for (i, (w, got)) in want_mont.iter().zip(std.chunks_exact(LIMBS)).enumerate() {
            let want: Vec<u32> = g16_gpu_layout::PackedScalar::from_fr(w).v.to_vec();
            assert_eq!(
                got,
                &want[..],
                "2^{log_n} h_std: element {i} is {got:08x?}, standard form is {want:08x?}"
            );
        }

        // The saving the fusion claims is one whole write of C. What that means depends on
        // the batch count, so both cases are checked rather than the convenient one:
        //
        //   single batch  the head itself joins, so DST is never written at all;
        //   two batches   the head writes the half-transformed vector into DST and the
        //                 joined tail reads it and writes only H, so DST must hold neither
        //                 the sentinel nor the finished transform.
        let (dst_words, _) = read(&dst, nn);
        let all_sentinel = dst_words.iter().all(|&w| w == SENTINEL);
        if ntt.dispatches() == 1 {
            assert!(
                all_sentinel,
                "2^{log_n}: the single joined batch also wrote DST, so it is not saving the \
                 write it claims to"
            );
        } else {
            assert!(!all_sentinel, "2^{log_n}: the head wrote nothing into DST");
            let finished = g16_wgpu::gather::fr_words(&transformed);
            assert_ne!(
                dst_words, finished,
                "2^{log_n}: DST holds the finished transform, so the joined tail wrote it \
                 as well and the fusion saved nothing"
            );
        }

        // An operand swap or an a*a would have to change these, so state that they are not
        // accidentally equal.
        assert_ne!(want_mont, transformed, "2^{log_n}: H equals C");
        println!(
            "2^{log_n}: join over {n} points, {} dispatches",
            ntt.dispatches()
        );
    }
}

// ---------------------------------------------------------------------------
// 5. The floor, computed from the generator
// ---------------------------------------------------------------------------

#[test]
fn every_generated_kernel_fits_the_browser_floor() {
    // The WebGPU specification defaults, not this adapter's. A kernel that only fits native
    // Metal's headroom is a kernel that fails in a stock browser.
    const FLOOR_WORKGROUP_STORAGE: u64 = 16384;
    const FLOOR_INVOCATIONS: u32 = 256;
    const FLOOR_STORAGE_BUFFERS: u32 = 8;

    let v = g16_wgpu::gen::Variant::Cios32Unrolled;
    let ceiling = wgsl::MAX_FUSED_PASSES;

    // Every k the floor permits, not merely the ones this machine's artifacts produce.
    let mut largest_ok = 0;
    for k in 0..=ceiling {
        let bytes = wgsl::workgroup_bytes(k, v);
        if bytes <= FLOOR_WORKGROUP_STORAGE {
            largest_ok = k;
        }
    }
    assert_eq!(
        largest_ok, 9,
        "16384 bytes / 32 bytes per Fr is 512 elements, so the largest tile is 2^9"
    );
    assert_eq!(
        Ntt::max_fused(floor()),
        9,
        "the Floor profile should derive max_fused 9 from its granted workgroup storage"
    );
    // What ships is 8, not 9, and that is a measurement rather than a limit. The sweep is in
    // `the_ntt_shape_is_measured_and_not_inherited`; this only pins the constant.
    assert_eq!(Ntt::preferred_fused(floor()), 8);
    for (k, want) in [(1u32, 32u32), (4, 32), (6, 32), (7, 64), (8, 128), (9, 128)] {
        assert_eq!(
            wgsl::workgroup_for(k),
            want,
            "workgroup_for({k}) should be 2^(k-1) clamped to 32..=128"
        );
    }

    for m in Mode::ALL {
        let entries = Ntt::bind_group_layout_entries(m);
        let storage = entries
            .iter()
            .filter(|e| {
                matches!(
                    e.ty,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    }
                )
            })
            .count() as u32;
        assert_eq!(
            storage,
            m.storage_buffers(),
            "{}: the layout and Mode::storage_buffers disagree",
            m.entry(9)
        );
        assert!(
            storage <= FLOOR_STORAGE_BUFFERS,
            "{} declares {storage} storage buffers, the floor allows {FLOOR_STORAGE_BUFFERS}",
            m.entry(9)
        );
        println!(
            "{:22} {storage} storage + 1 uniform in 1 bind group",
            m.entry(9)
        );
    }
    // The correction to design §4, asserted rather than left as prose: the head's join
    // variant needs seven, not the six the design tabulates, because on a domain of 2^9 or
    // smaller the joined batch is also the batch carrying the coset shift and so reads PTAB.
    assert_eq!(Mode::HEAD_JOIN.storage_buffers(), 7);
    assert_eq!(Mode::HEAD_PLAIN.storage_buffers(), 4);
    assert_eq!(Mode::TAIL_PLAIN.storage_buffers(), 2);
    assert_eq!(Mode::TAIL_JOIN.storage_buffers(), 6);

    // Cross-check against the emitted text, so neither the table above nor the shader can
    // grow one without the other.
    let src = wgsl::ntt_module(v, &[0, 1, 8, 9]);
    assert_eq!(
        src.matches("var<storage").count(),
        8,
        "the module declares a different number of storage bindings than Mode::bindings maps"
    );
    for k in [0u32, 1, 8, 9] {
        let decl = format!("var<workgroup> SH{k}: array<Fr, {}>;", 1u32 << k);
        assert!(src.contains(&decl), "missing {decl}");
        for m in Mode::ALL {
            assert!(
                src.contains(&format!("fn {}(", m.entry(k))),
                "missing entry point {}",
                m.entry(k)
            );
        }
    }
    let sizes: Vec<&str> = src
        .match_indices("@workgroup_size(")
        .map(|(i, _)| &src[i..])
        .collect();
    assert_eq!(sizes.len(), 16, "4 modes x 4 tile sizes");
    for s in sizes {
        let n: u32 = s["@workgroup_size(".len()..]
            .split(')')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            n > 0 && n <= FLOOR_INVOCATIONS,
            "@workgroup_size({n}) is over the floor's {FLOOR_INVOCATIONS}"
        );
    }
    println!(
        "largest tile the floor allows k = {largest_ok} ({} bytes of workgroup storage); \
         shipped cap k = {} at {} threads; module for k in [0,1,8,9] is {:.1} KiB",
        wgsl::workgroup_bytes(largest_ok, v),
        wgsl::PREFERRED_FUSED,
        wgsl::workgroup_for(wgsl::PREFERRED_FUSED),
        src.len() as f64 / 1024.0
    );
}

// ---------------------------------------------------------------------------
// 6. A tile that does not fit is rejected by the generator, not by the driver
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "over the floor's 16384")]
fn a_tile_over_the_floor_is_a_generator_panic() {
    wgsl::ntt_module_at(g16_wgpu::gen::Variant::Cios32Unrolled, &[10], None);
}

#[test]
fn a_max_fused_over_the_granted_budget_is_refused() {
    let Err(e) = Ntt::with_shape(floor(), 12, 10, None) else {
        panic!("max_fused 10 needs 32 KiB of workgroup storage and the Floor grants 16")
    };
    assert!(
        format!("{e}").contains("max_fused 10 is outside 1..=9"),
        "unhelpful error: {e}"
    );
}

// ---------------------------------------------------------------------------
// 7. The shape, measured rather than inherited
// ---------------------------------------------------------------------------

/// Microseconds for one `iNTT -> coset -> NTT` chain at a given shape, submit overhead
/// removed by differencing 1 chain against `1 + REPEATS` in a single submit.
///
/// `tests/gather.rs` measured that fixed cost at 230 us on this machine, which would swamp
/// the 0.4 ms a 2^12 chain takes. `None` means the many-chain submit came back no slower than
/// the one-chain submit, which happens when another test is contending for the same GPU;
/// reporting that as a number is the failure `bench/scripts` was fixed for.
fn chain_us(log_n: u32, max_fused: u32, workgroup: Option<u32>, x: &[Fr]) -> Option<f64> {
    const REPEATS: u32 = 10;
    let b = floor();
    let n = 1usize << log_n;
    let nn = n as u32;
    let ntt = Ntt::with_shape(b, log_n, max_fused, workgroup).expect("pipelines");
    let tables = NttTables::new(b, n).expect("tables");
    let v = scratch("shape v", nn, Some(x));
    let t = scratch("shape t", nn, None);
    let mut ring = ParamRing::new(b, "shape params", 2 * ntt.dispatches() as u32).expect("ring");
    let p0 = ntt
        .plan(
            b,
            &tables,
            &mut ring,
            &Transform {
                dir: Direction::Inverse,
                scale: Scale::SizeInv,
                src: &v,
                dst: &t,
                epilogue: Epilogue::Plain,
            },
        )
        .expect("plan");
    let p1 = ntt
        .plan(
            b,
            &tables,
            &mut ring,
            &Transform {
                dir: Direction::Forward,
                scale: Scale::CosetPowers,
                src: &t,
                dst: &v,
                epilogue: Epilogue::Plain,
            },
        )
        .expect("plan");
    ring.flush(b);

    let once = |chains: u32| -> u128 {
        let t0 = std::time::Instant::now();
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..chains {
                ntt.encode(&mut pass, &p0).expect("encode");
                ntt.encode(&mut pass, &p1).expect("encode");
            }
        }
        b.queue().submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        t0.elapsed().as_micros()
    };
    once(1);
    once(1 + REPEATS);
    let mut one: Vec<u128> = (0..3).map(|_| once(1)).collect();
    let mut many: Vec<u128> = (0..3).map(|_| once(1 + REPEATS)).collect();
    one.sort_unstable();
    many.sort_unstable();
    (many[1] > one[1]).then(|| (many[1] - one[1]) as f64 / REPEATS as f64)
}

/// The four artifact domains, so the sweep is over sizes this prover actually meets.
const SWEEP_DOMAINS: [u32; 4] = [12, 14, 16, 18];

/// Design §4 fixes both halves of the NTT's shape and both are wrong here, one of them by
/// 4.7x. This is the measurement that says so, and it re-runs on every `cargo test`.
///
/// Two sweeps, because the two knobs interact: passes fused per dispatch decides the tile,
/// and the tile decides how many butterflies a workgroup has to share out.
#[test]
fn the_ntt_shape_is_measured_and_not_inherited() {
    let b = floor();
    let inputs: Vec<(u32, Vec<Fr>)> = SWEEP_DOMAINS
        .iter()
        .map(|&log_n| (log_n, sample(1usize << log_n, 0x5EED ^ log_n as u64)))
        .collect();

    // ---- the tile: how many passes to fuse ----
    println!(
        "one iNTT + coset + NTT chain in microseconds, submit overhead differenced out.\n\
         Passes fused per dispatch, each column at its own measured workgroup size:"
    );
    let ceiling = Ntt::max_fused(b);
    print!("{:>8}", "domain");
    for mf in 4..=ceiling {
        print!("{:>16}", format!("mf {mf}"));
    }
    println!();
    let mut shipped_regret: f64 = 0.0;
    for (log_n, x) in &inputs {
        print!("{:>8}", format!("2^{log_n}"));
        let mut best = f64::MAX;
        let mut shipped = None;
        for mf in 4..=ceiling {
            let us = chain_us(*log_n, mf, None, x);
            let k = g16_wgpu::split_passes(*log_n, mf)[0].k;
            match us {
                Some(us) => {
                    print!("{:>16}", format!("{us:.0} (k{k})"));
                    best = best.min(us);
                    if mf == Ntt::preferred_fused(b) {
                        shipped = Some(us);
                    }
                }
                None => print!("{:>16}", format!("- (k{k})")),
            }
        }
        println!();
        if let (Some(sh), true) = (shipped, best < f64::MAX) {
            shipped_regret = shipped_regret.max(sh / best - 1.0);
        }
    }
    println!(
        "shipping mf {} (gen::ntt::PREFERRED_FUSED), worst regret {:.1}% over these domains",
        Ntt::preferred_fused(b),
        shipped_regret * 100.0
    );

    // ---- the workgroup, at the shipped tile ----
    const SIZES: [u32; 4] = [32, 64, 128, 256];
    println!("\nThreads per workgroup at the shipped tile:");
    print!("{:>8}{:>6}", "domain", "k");
    for wg in SIZES {
        print!("{wg:>9}");
    }
    print!("{:>9}", "rule");
    println!();
    let mut wg_regret: f64 = 0.0;
    for (log_n, x) in &inputs {
        let mf = Ntt::preferred_fused(b);
        let k = g16_wgpu::split_passes(*log_n, mf)[0].k;
        print!("{:>8}{:>6}", format!("2^{log_n}"), k);
        let mut best = f64::MAX;
        for wg in SIZES {
            // Correctness at every size, always. Only the timing is allowed to be absent.
            match chain_us(*log_n, mf, Some(wg), x) {
                Some(us) => {
                    print!("{us:>9.0}");
                    best = best.min(us);
                }
                None => print!("{:>9}", "-"),
            }
        }
        match chain_us(*log_n, mf, None, x) {
            Some(us) => {
                print!("{us:>9.0}");
                if best < f64::MAX {
                    wg_regret = wg_regret.max(us / best - 1.0);
                }
            }
            None => print!("{:>9}", "-"),
        }
        println!();
    }
    println!(
        "the shipped rule is 2^(k-1) clamped to 32..=128 (gen::ntt::workgroup_for), worst \
         regret {:.1}%",
        wg_regret * 100.0
    );

    // Deliberately loose, and it is the table above that a human reads. Under the default
    // parallel `cargo test` this contends with the other tests on the same GPU, and
    // `tests/gather.rs` records that a tighter bar was flaky about one run in three. A flaky
    // test gets deleted rather than fixed, so this one only catches a shape that is wrong by
    // a factor.
    assert!(
        shipped_regret < 0.5,
        "the shipped pass cap is more than 1.5x off the best column; re-read the table"
    );
    assert!(
        wg_regret < 0.5,
        "the shipped workgroup rule is more than 1.5x off the best column; re-read the table"
    );
}
