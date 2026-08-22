//! U5's acceptance tests: stage 0 on a real GPU, against `g16_core::cpu` and against the
//! browser's 8-storage-buffer floor.
//!
//! # The three things being pinned
//!
//! 1. **Correctness on every artifact.** `g16_core::cpu::CpuCircuit::gather` is the oracle,
//!    called directly rather than through `compute_h`. `compute_h` folds stages 0 to 4
//!    together, so comparing only its output lets a wrong gather cancel against a wrong
//!    transform, and those are separate units developed in parallel here.
//! 2. **The buffer count.** `gather.metal` binds 10 storage buffers, the Floor allows 8, and
//!    the whole restructuring in `gen::gather` exists to get to 7. That number is asserted
//!    from the bind group layout the pipeline is actually built from, and cross-checked
//!    against the generated WGSL, so neither can grow silently.
//! 3. **The chunked dispatch.** Every artifact here is 2^18 rows and takes one dispatch, so
//!    the multi-dispatch path that a 2^23 domain needs would otherwise ship untested. The
//!    synthetic test forces `rows_per_dispatch` down and requires more than one.
//!
//! # Two rules the inputs follow, because the last round's verifier found bugs that hid
//!
//! **Never symmetric.** A and B in the synthetic CSR have different row lengths, different
//! signals and different values, and no witness entry equals another. An operand swap between
//! the two matrices, or `out_c = a*a`, has to change a number.
//!
//! **Outputs are pre-filled with a sentinel, never left zero.** Nearly half the rows of a real
//! key are empty and their correct gather result is exactly zero, so a kernel that wrote
//! nothing at all would agree with the oracle on those rows against a zero-initialised buffer.
//! The buffers here start as all-ones words, which is not a canonical field element, and one
//! extra row past the end is checked to be still all-ones so an off-by-one in the `row >=
//! row_hi` guard cannot pass.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ark_ff::Zero as _;
use g16_core::cpu::CpuCircuit;
use g16_core::PreparedCircuit as _;
use g16_field::Fr;
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::LIMBS;
use g16_wgpu::gather::{fr_buffer, fr_words, upload_fr};
use g16_wgpu::gen::gather as wgsl;
use g16_wgpu::{CsrHost, CsrTables, GatherAbc, LimitsProfile, ParamRing, Readback, WgpuBackend};
use g16_zkey::{wtns::Witness, Coefficients, ProvingKey};

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

// ---------------------------------------------------------------------------
// Running the kernel
// ---------------------------------------------------------------------------

/// The sentinel every output row starts as. All-ones is above both moduli, so it is not a
/// value any correct gather can produce, and it is not the zero an empty row produces either.
const SENTINEL: u32 = 0xffff_ffff;

struct Run {
    a: Vec<u32>,
    b: Vec<u32>,
    c: Vec<u32>,
    /// The one row past `n_rows` in each buffer, which must still be [`SENTINEL`].
    tail: [Vec<u32>; 3],
    dispatches: u32,
    /// Microseconds one gather costs on the device, submit overhead removed.
    ///
    /// Taken as `(time for 1 + REPEATS passes) - (time for 1 pass)` divided by `REPEATS`,
    /// both in a single submit, so the fixed cost of encode plus submit plus the fence wait
    /// cancels. That fixed cost is not small: an 8-row gather measures 230 us of it here,
    /// against design §3's "0.1 to 0.3 ms" for an empty submit, so a wall-clock number at
    /// these sizes would be mostly overhead and would not scale with the row count. This is
    /// still not a kernel timing; GPU timestamp queries are U11's.
    ///
    /// `None` when the many-pass submit came back no slower than the one-pass submit, which
    /// means another test on the same device was contending and the difference is noise
    /// rather than work. Reporting a junk number as a number is the exact failure
    /// `bench/scripts` was fixed for; this reports it as absent.
    kernel_us: Option<f64>,
}

/// Uploads, dispatches and reads back all three outputs.
///
/// `rows_per_dispatch` of `None` is the device's own cap, which is one dispatch for anything
/// under 2^23 rows.
fn run_gather(
    coeffs: &Coefficients,
    witness: &[Fr],
    domain_size: usize,
    rows_per_dispatch: Option<u32>,
    workgroup: Option<u32>,
) -> Run {
    let b = floor();
    let host = CsrHost::build(coeffs, domain_size, witness.len()).expect("CSR rejected");
    let csr = CsrTables::upload(b, &host).expect("CSR upload");

    let gather = match (rows_per_dispatch, workgroup) {
        (None, None) => GatherAbc::new(b),
        (Some(r), None) => GatherAbc::with_rows_per_dispatch(b, r),
        (r, Some(wg)) => GatherAbc::with_shape(
            b,
            r.unwrap_or_else(|| b.granted_limits().max_compute_workgroups_per_dimension * wg),
            wg,
        ),
    }
    .expect("gather pipeline");

    let n = csr.n_rows();
    let w = upload_fr(b, "witness", witness).expect("witness upload");

    // One row of slack, pre-filled along with everything else, so an over-run has somewhere
    // to land where it will be noticed.
    let rows = n + 1;
    let fill = vec![SENTINEL; rows as usize * LIMBS];
    let outs: Vec<wgpu::Buffer> = ["out_a", "out_b", "out_c"]
        .iter()
        .map(|label| {
            let buf = fr_buffer(b, label, rows).expect("output buffer");
            b.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&fill));
            buf
        })
        .collect();

    let mut ring =
        ParamRing::new(b, "gather params", gather.dispatches(n).max(1)).expect("parameter ring");
    let offsets = gather.plan(&csr, &mut ring).expect("plan");
    ring.flush(b);
    let bind = gather
        .bind(b, &csr, &ring, &w, &outs[0], &outs[1], &outs[2])
        .expect("bind");

    // The dispatches and the first output's copy share one encoder, which is the shape U7
    // needs: stages 0 to 4 are one submit. The other two copies follow in their own, because
    // `Readback` always lands at staging offset 0 and one buffer cannot hold three.
    let bytes = rows as u64 * (LIMBS * 4) as u64;
    let mut read: Vec<Vec<u32>> = Vec::with_capacity(3);
    for (i, out) in outs.iter().enumerate() {
        let rb = Readback::new(b, "gather readback", bytes).expect("readback");
        let mut enc = b.device().create_command_encoder(&Default::default());
        if i == 0 {
            let mut pass = enc.begin_compute_pass(&Default::default());
            gather
                .encode(&mut pass, &bind, &csr, &offsets)
                .expect("encode");
        }
        rb.copy_from(&mut enc, out, 0, bytes).expect("copy");
        let raw = pollster::block_on(rb.submit_and_read(b, enc, bytes)).expect("readback failed");
        read.push(bytemuck::cast_slice::<u8, u32>(&raw).to_vec());
    }

    // Timing, after the correctness reads so a hang shows up as a wrong answer first. The
    // gather is idempotent (it writes the same values from the same inputs), so running it
    // many times in one submit is a legitimate way to amortise the submit cost away.
    const REPEATS: u32 = 40;
    let once = |passes: u32| -> u128 {
        let t0 = std::time::Instant::now();
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..passes {
                gather
                    .encode(&mut pass, &bind, &csr, &offsets)
                    .expect("encode");
            }
        }
        b.queue().submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        t0.elapsed().as_micros()
    };
    for _ in 0..3 {
        once(1);
        once(1 + REPEATS);
    }
    let mut one_us: Vec<u128> = (0..5).map(|_| once(1)).collect();
    let mut many_us: Vec<u128> = (0..5).map(|_| once(1 + REPEATS)).collect();
    one_us.sort_unstable();
    many_us.sort_unstable();
    let kernel_us =
        (many_us[2] > one_us[2]).then(|| (many_us[2] - one_us[2]) as f64 / REPEATS as f64);

    let split = |v: &Vec<u32>| {
        let cut = n as usize * LIMBS;
        (v[..cut].to_vec(), v[cut..].to_vec())
    };
    let (a, ta) = split(&read[0]);
    let (bb, tb) = split(&read[1]);
    let (c, tc) = split(&read[2]);
    Run {
        a,
        b: bb,
        c,
        tail: [ta, tb, tc],
        dispatches: offsets.len() as u32,
        kernel_us,
    }
}

/// Elementwise, in device words, with the field element index in the message.
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

fn check_tail(run: &Run) {
    for (name, t) in ["out_a", "out_b", "out_c"].iter().zip(&run.tail) {
        assert!(
            t.iter().all(|&x| x == SENTINEL),
            "{name}: the kernel wrote one row past row_hi, so the bounds guard is off by one \
             (tail is {t:08x?})"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Every artifact, against the CPU backend
// ---------------------------------------------------------------------------

#[test]
fn the_gather_matches_the_cpu_backend_on_every_artifact() {
    let found = artifacts();
    assert!(
        !found.is_empty(),
        "no artifacts under bench/artifacts; the symlink into the main worktree is missing \
         and this test would otherwise pass by doing nothing"
    );

    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let domain_size = pk.domain_size;
        let n_vars = pk.n_vars;

        // The premise the one-thread-per-row mapping rests on, restated from the key rather
        // than quoted from a comment.
        let stats = |m: usize| -> (f64, usize, usize) {
            let rp = &pk.coeffs.row_ptr[m];
            let rows = rp.len() - 1;
            let nnz = *rp.last().unwrap() as usize;
            let empty = (0..rows).filter(|&c| rp[c] == rp[c + 1]).count();
            let longest = (0..rows)
                .map(|c| (rp[c + 1] - rp[c]) as usize)
                .max()
                .unwrap_or(0);
            (nnz as f64 / rows as f64, empty, longest)
        };
        let (ma, ea, la) = stats(0);
        let (mb, eb, lb) = stats(1);

        let run = run_gather(&pk.coeffs, &w, domain_size, None, None);

        let cpu = CpuCircuit::new(pk).expect("cpu circuit");
        let want_a = cpu.gather(0, &w);
        let want_b = cpu.gather(1, &w);
        let want_c: Vec<Fr> = want_a.iter().zip(&want_b).map(|(x, y)| *x * y).collect();

        compare(&format!("{name} out_a"), &run.a, &want_a);
        compare(&format!("{name} out_b"), &run.b, &want_b);
        compare(&format!("{name} out_c"), &run.c, &want_c);
        check_tail(&run);

        let nonzero = want_a.iter().filter(|x| !x.is_zero()).count();
        println!(
            "{name}: n_vars {n_vars} domain 2^{} rows {domain_size} dispatches {}  \
             A mean {ma:.2} empty {ea} longest {la}  B mean {mb:.2} empty {eb} longest {lb}  \
             nonzero A rows {nonzero}  kernel {}",
            domain_size.trailing_zeros(),
            run.dispatches,
            match run.kernel_us {
                Some(us) => format!(
                    "{us:.0} us ({:.1} ns/row)",
                    us * 1000.0 / domain_size as f64
                ),
                None => "unmeasurable under contention".to_string(),
            },
        );
        // An all-zero A would make every assertion above vacuous, which is exactly the shape
        // "the kernel ran but wrote nothing" takes on the empty half of the domain.
        assert!(
            nonzero > 0,
            "{name}: the oracle's A is entirely zero, so this artifact proves nothing"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The buffer count, which is the whole reason the kernel was restructured
// ---------------------------------------------------------------------------

#[test]
fn the_gather_pipeline_layout_declares_at_most_eight_storage_buffers() {
    let b = floor();
    let entries = GatherAbc::bind_group_layout_entries();

    let storage = GatherAbc::storage_buffer_count();
    let uniform = entries
        .iter()
        .filter(|e| {
            matches!(
                e.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                }
            )
        })
        .count() as u32;

    // The Floor, not this adapter's 9. A kernel sized for the ninth passes here and fails in
    // a stock browser, which research file 01 records heliax shipping twice.
    let spec = wgpu::Limits::default();
    assert_eq!(spec.max_storage_buffers_per_shader_stage, 8);
    assert!(
        storage <= spec.max_storage_buffers_per_shader_stage,
        "the gather declares {storage} storage buffers, the WebGPU floor allows {}",
        spec.max_storage_buffers_per_shader_stage
    );
    assert_eq!(
        storage,
        wgsl::STORAGE_BUFFERS,
        "gen::gather::STORAGE_BUFFERS says {} but the bind group layout has {storage}",
        wgsl::STORAGE_BUFFERS
    );
    assert_eq!(
        uniform, 1,
        "the scalar parameters must be one uniform block"
    );

    // Independently counted from the generated source, so a binding added to the shader
    // without a matching layout entry (or the reverse) fails here rather than at dispatch.
    let src = wgsl::gather_module(Default::default());
    let declared = src.matches("var<storage").count() as u32;
    assert_eq!(
        declared, storage,
        "the WGSL declares {declared} storage buffers, the layout declares {storage}"
    );
    assert_eq!(src.matches("var<uniform").count() as u32, uniform);

    // One bind group. `maxBindGroups` is 4 in every browser at every tier, so this has room,
    // but the count is asserted because splitting bindings across groups is the fix that
    // does *not* work for the storage limit and someone will try it.
    assert!(
        src.matches("@group(0)").count() as u32 == storage + uniform,
        "every binding must be in group 0"
    );

    // And the real enforcement: the pipeline builds on a device whose granted limit is 8.
    let g = GatherAbc::new(b).expect("the gather pipeline does not build at the Floor");
    assert_eq!(b.granted_limits().max_storage_buffers_per_shader_stage, 8);
    println!(
        "gather: {storage} storage + {uniform} uniform in 1 bind group, \
         {} bytes of WGSL, {}",
        g.source_len(),
        g.kernels().summary(),
    );
    println!(
        "metal counterpart binds 10 storage buffers (gather.metal:60-71); the floor is {}",
        spec.max_storage_buffers_per_shader_stage
    );
}

// ---------------------------------------------------------------------------
// 3. A synthetic ragged CSR, asymmetric on purpose, and the chunked dispatch
// ---------------------------------------------------------------------------

/// A CSR shaped like a real key's but deliberately asymmetric between A and B.
///
/// Row lengths follow different patterns per matrix, including empty rows, singletons and one
/// long row per matrix at a different index, so a swap of the two matrices' bases cannot
/// produce the same answer. Values and signals are drawn from one stream, so no value in A
/// appears in B.
fn synthetic(rng: &mut SplitMix64, n_rows: usize, n_vars: usize) -> Coefficients {
    let mut row_ptr = [
        Vec::with_capacity(n_rows + 1),
        Vec::with_capacity(n_rows + 1),
    ];
    let mut signal = [Vec::new(), Vec::new()];
    let mut value = [Vec::new(), Vec::new()];

    for m in 0..2 {
        row_ptr[m].push(0u32);
        for c in 0..n_rows {
            // Two different raggedness patterns. A is sparser, like the measured 1.13 per
            // row; B is denser, like the measured 1.69. Both have a long row, at different
            // places, and long stretches of empties.
            let len = if m == 0 {
                match c % 7 {
                    0 | 1 | 2 | 5 => 0,
                    3 => 1,
                    4 => 2,
                    _ => 3,
                }
            } else {
                match c % 5 {
                    0 | 3 => 0,
                    1 => 2,
                    2 => 1,
                    _ => 4,
                }
            };
            let len = if c == 97 * (m + 1) { 95 } else { len };
            for _ in 0..len {
                signal[m].push((rng.next_u64() % n_vars as u64) as u32);
                value[m].push(rng.next_fr());
            }
            row_ptr[m].push(signal[m].len() as u32);
        }
    }
    Coefficients {
        row_ptr,
        signal,
        value,
    }
}

/// The gather, written the obvious way with arkworks, as an oracle independent of
/// `g16_core::cpu` for the synthetic case.
fn reference(coeffs: &Coefficients, w: &[Fr], m: usize, n_rows: usize) -> Vec<Fr> {
    (0..n_rows)
        .map(|c| {
            let lo = coeffs.row_ptr[m][c] as usize;
            let hi = coeffs.row_ptr[m][c + 1] as usize;
            let mut acc = Fr::zero();
            for k in lo..hi {
                acc += coeffs.value[m][k] * w[coeffs.signal[m][k] as usize];
            }
            acc
        })
        .collect()
}

#[test]
fn a_ragged_asymmetric_csr_gathers_exactly_and_chunks_identically() {
    // Hundreds of rows, not 8: the field gate found an 8-element check passing 24% of the
    // time with the conditional subtraction deleted, and stage 0 has the same exposure
    // through fr_mul.
    //
    // 1000 and not 1024, deliberately. A row count that is a multiple of the workgroup size
    // means the last thread of the last workgroup is the last row, so the guard is never
    // asked a question and neither an off-by-one in it nor a dispatch that drops its partial
    // trailing workgroup changes any output. Both mutations were caught only by `tiny_mul`
    // (8 rows) until this constant stopped being a power of two.
    const ROWS: usize = 1000;
    const VARS: usize = 701;
    let mut rng = SplitMix64(0x5eed_0000_0000_0005);
    let coeffs = synthetic(&mut rng, ROWS, VARS);
    let w: Vec<Fr> = (0..VARS).map(|_| rng.next_fr()).collect();

    let want_a = reference(&coeffs, &w, 0, ROWS);
    let want_b = reference(&coeffs, &w, 1, ROWS);
    let want_c: Vec<Fr> = want_a.iter().zip(&want_b).map(|(x, y)| *x * y).collect();

    // A and B must actually differ, or an operand swap is invisible. This is the check the
    // Fq2 test did not have.
    let differing = want_a.iter().zip(&want_b).filter(|(x, y)| x != y).count();
    assert!(
        differing > ROWS / 2,
        "only {differing} of {ROWS} rows differ between A and B; the inputs are too symmetric \
         to catch an operand swap"
    );
    let nz_a = want_a.iter().filter(|x| !x.is_zero()).count();
    let nz_b = want_b.iter().filter(|x| !x.is_zero()).count();
    assert!(nz_a > 100 && nz_b > 100, "{nz_a} / {nz_b} nonzero rows");

    let one = run_gather(&coeffs, &w, ROWS, None, None);
    assert_eq!(
        one.dispatches, 1,
        "{ROWS} rows is one dispatch at the Floor"
    );
    compare("synthetic out_a", &one.a, &want_a);
    compare("synthetic out_b", &one.b, &want_b);
    compare("synthetic out_c", &one.c, &want_c);
    check_tail(&one);

    // The multi-dispatch path. 1024 rows in chunks of three workgroups is a short last
    // chunk plus a partial last workgroup inside it, so `row_lo`, the per-chunk `row_hi` and
    // the tail guard are all exercised at once. Every artifact on this machine is one
    // dispatch, so without this the path a 2^23 domain needs would never run.
    let chunk = 3 * wgsl::WORKGROUP;
    let many = run_gather(&coeffs, &w, ROWS, Some(chunk), None);
    assert_eq!(
        many.dispatches,
        (ROWS as u32).div_ceil(chunk),
        "{ROWS} rows in chunks of {chunk}"
    );
    compare("chunked out_a", &many.a, &want_a);
    compare("chunked out_b", &many.b, &want_b);
    compare("chunked out_c", &many.c, &want_c);
    check_tail(&many);

    println!(
        "synthetic: {ROWS} rows, {VARS} vars, A nnz {} B nnz {}, {differing} rows differ, \
         {nz_a}/{nz_b} nonzero; 1 dispatch and {} dispatches agree elementwise",
        coeffs.signal[0].len(),
        coeffs.signal[1].len(),
        many.dispatches,
    );
}

// ---------------------------------------------------------------------------
// 4. The concatenation itself, with no GPU involved
// ---------------------------------------------------------------------------

#[test]
fn the_concatenated_csr_is_the_two_matrices_end_to_end() {
    const ROWS: usize = 300;
    const VARS: usize = 211;
    let mut rng = SplitMix64(0x5eed_0000_0000_0105);
    let coeffs = synthetic(&mut rng, ROWS, VARS);
    let host = CsrHost::build(&coeffs, ROWS, VARS).expect("CSR rejected");

    // Unbiased and byte for byte. If B's row_ptr were biased by nnz_a on the host this
    // comparison would be impossible, which is why it is not.
    assert_eq!(host.row_ptr.len(), 2 * (ROWS + 1));
    assert_eq!(&host.row_ptr[..ROWS + 1], &coeffs.row_ptr[0][..]);
    assert_eq!(&host.row_ptr[ROWS + 1..], &coeffs.row_ptr[1][..]);
    assert_eq!(host.row_base, [0, (ROWS + 1) as u32]);

    let nnz_a = coeffs.signal[0].len();
    let nnz_b = coeffs.signal[1].len();
    assert!(nnz_a > 0 && nnz_b > 0 && nnz_a != nnz_b);
    assert_eq!(host.nnz, [nnz_a as u32, nnz_b as u32]);
    assert_eq!(host.nz_base, [0, nnz_a as u32]);
    assert_eq!(&host.signal[..nnz_a], &coeffs.signal[0][..]);
    assert_eq!(&host.signal[nnz_a..], &coeffs.signal[1][..]);
    assert_eq!(host.value.len(), (nnz_a + nnz_b) * LIMBS);
    assert_eq!(
        &host.value[..nnz_a * LIMBS],
        &fr_words(&coeffs.value[0])[..]
    );
    assert_eq!(
        &host.value[nnz_a * LIMBS..],
        &fr_words(&coeffs.value[1])[..]
    );

    // The base offsets the kernel is handed, spelled out. `nz_base_b == nnz_a` is the one
    // number that turns a working A into a silently wrong B.
    let p = host.params();
    assert_eq!(p.row_base_a, 0);
    assert_eq!(p.row_base_b, (ROWS + 1) as u32);
    assert_eq!(p.nz_base_a, 0);
    assert_eq!(p.nz_base_b, nnz_a as u32);
    assert_eq!(p.row_lo, 0);
    assert_eq!(p.row_hi, ROWS as u32);
    assert_eq!(std::mem::size_of_val(&p), 32, "the WGSL struct is 32 bytes");
}

// ---------------------------------------------------------------------------
// 5. Malformed input is refused on the host, because the device will not refuse it
// ---------------------------------------------------------------------------

#[test]
fn a_csr_the_device_would_read_out_of_bounds_is_refused_on_the_host() {
    const ROWS: usize = 64;
    const VARS: usize = 40;
    let mut rng = SplitMix64(0x5eed_0000_0000_0205);
    let good = synthetic(&mut rng, ROWS, VARS);
    assert!(CsrHost::build(&good, ROWS, VARS).is_ok());

    let msg = |c: &Coefficients, rows: usize, vars: usize| -> String {
        CsrHost::build(c, rows, vars)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "accepted".into())
    };

    // A signal past the end of the witness. On the device this reads zero and the proof is
    // quietly wrong; here it has to be an error naming the matrix.
    let mut bad_sig = synthetic(&mut SplitMix64(1), ROWS, VARS);
    bad_sig.signal[1][0] = VARS as u32;
    let m = msg(&bad_sig, ROWS, VARS);
    assert!(m.contains("signal") && m.contains('B'), "{m}");

    // A decreasing row_ptr makes `lo < hi` an unbounded loop on the device, which is a hang.
    let mut bad_ptr = synthetic(&mut SplitMix64(2), ROWS, VARS);
    let last = bad_ptr.row_ptr[0][ROWS];
    bad_ptr.row_ptr[0][ROWS] = 0;
    assert!(last > 0);
    let m = msg(&bad_ptr, ROWS, VARS);
    assert!(m.contains("monotone"), "{m}");

    // row_ptr of the wrong length: the B rows would start at the wrong place in the
    // concatenated buffer and every B row would be shifted.
    let m = msg(&good, ROWS + 1, VARS);
    assert!(m.contains("row_ptr entries"), "{m}");

    // The arrays disagreeing with the nonzero count.
    let mut short = synthetic(&mut SplitMix64(3), ROWS, VARS);
    short.signal[0].pop();
    let m = msg(&short, ROWS, VARS);
    assert!(m.contains("signals"), "{m}");

    assert!(msg(&good, 0, VARS).contains("domain size is zero"));
    assert!(msg(&good, ROWS, 0).contains("n_vars is zero"));
}

#[test]
fn an_output_buffer_too_short_for_the_domain_is_refused_before_the_dispatch() {
    const ROWS: usize = 128;
    const VARS: usize = 64;
    let b = floor();
    let mut rng = SplitMix64(0x5eed_0000_0000_0305);
    let coeffs = synthetic(&mut rng, ROWS, VARS);
    let w: Vec<Fr> = (0..VARS).map(|_| rng.next_fr()).collect();

    let host = CsrHost::build(&coeffs, ROWS, VARS).unwrap();
    let csr = CsrTables::upload(b, &host).unwrap();
    let gather = GatherAbc::new(b).unwrap();
    let ring = ParamRing::new(b, "p", 4).unwrap();

    let wbuf = upload_fr(b, "w", &w).unwrap();
    let full = |label| fr_buffer(b, label, ROWS as u32).unwrap();
    let (a, bb, c) = (full("a"), full("b"), full("c"));
    let short = fr_buffer(b, "short", ROWS as u32 - 1).unwrap();

    assert!(gather.bind(b, &csr, &ring, &wbuf, &a, &bb, &c).is_ok());
    let e = gather
        .bind(b, &csr, &ring, &wbuf, &a, &bb, &short)
        .unwrap_err()
        .to_string();
    assert!(e.contains("out_c"), "{e}");

    // A witness shorter than the signals the CSR references reads zeros on the device.
    let short_w = fr_buffer(b, "short w", VARS as u32 - 1).unwrap();
    let e = gather
        .bind(b, &csr, &ring, &short_w, &a, &bb, &c)
        .unwrap_err()
        .to_string();
    assert!(e.contains("witness"), "{e}");

    // And a dispatch plan that does not match the row count.
    let mut enc = b.device().create_command_encoder(&Default::default());
    let mut pass = enc.begin_compute_pass(&Default::default());
    let bind = gather.bind(b, &csr, &ring, &wbuf, &a, &bb, &c).unwrap();
    let e = gather
        .encode(&mut pass, &bind, &csr, &[])
        .unwrap_err()
        .to_string();
    assert!(e.contains("dispatches"), "{e}");
}

// ---------------------------------------------------------------------------
// 6. The workgroup size, measured rather than inherited
// ---------------------------------------------------------------------------

/// Design §4 puts the gather at 64 threads per workgroup, which is `g16-metal`'s threadgroup
/// size, which was chosen on Metal's occupancy and not on wgpu's. Three of the four units
/// before this one found the design wrong about something they could measure, so this
/// measures it. The generator takes the size as a parameter for exactly this reason.
///
/// The kernel is checked against the oracle at every size, so a size that is fast because it
/// is wrong fails rather than wins.
#[test]
fn the_design_picked_the_workgroup_size_that_wins() {
    let found = artifacts();
    assert!(!found.is_empty(), "no artifacts under bench/artifacts");
    const SIZES: [u32; 4] = [32, 64, 128, 256];

    let mut worst_ratio = 0.0f64;
    let mut worst_at = String::new();
    let mut judged = 0usize;
    for (name, dir) in &found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let w = Witness::load(&dir.join("circuit.wtns")).expect("wtns").0;
        let domain_size = pk.domain_size;
        let cpu = CpuCircuit::new(pk).expect("cpu circuit");
        let want_a = cpu.gather(0, &w);
        let want_b = cpu.gather(1, &w);

        let mut rows = Vec::new();
        let mut usable = true;
        for wg in SIZES {
            let run = run_gather(&cpu.key().coeffs, &w, domain_size, None, Some(wg));
            // The correctness half never depends on the timing half. Every size gathers the
            // same vectors or this test fails, contention or not.
            compare(&format!("{name} wg{wg} out_a"), &run.a, &want_a);
            compare(&format!("{name} wg{wg} out_b"), &run.b, &want_b);
            check_tail(&run);
            match run.kernel_us {
                Some(us) => rows.push((wg, us)),
                None => usable = false,
            }
        }
        if !usable {
            println!(
                "{name} (2^{}): timing discarded, another test held the GPU. Correctness at \
                 all four workgroup sizes still checked.",
                domain_size.trailing_zeros()
            );
            continue;
        }
        let best = rows
            .iter()
            .cloned()
            .fold((0u32, f64::MAX), |acc, r| if r.1 < acc.1 { r } else { acc });
        let shipped = rows
            .iter()
            .find(|(wg, _)| *wg == wgsl::WORKGROUP)
            .expect("the shipped workgroup size is not in the sweep")
            .1;
        let ratio = shipped / best.1;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_at = name.clone();
        }
        judged += 1;
        let table: Vec<String> = rows
            .iter()
            .map(|(wg, us)| format!("{wg}: {us:.0}"))
            .collect();
        println!(
            "{name} (2^{}) us by workgroup size, submit overhead removed: {}  ->  best {} \
             at {:.0}, shipped {} at {shipped:.0} ({:+.1}%)",
            domain_size.trailing_zeros(),
            table.join("  "),
            best.0,
            best.1,
            wgsl::WORKGROUP,
            (ratio - 1.0) * 100.0,
        );
    }

    if judged == 0 {
        println!("every artifact's timing was discarded; run with --test-threads=1 to get a sweep");
        return;
    }

    // Two bars, because this is a timing test inside a binary of other GPU tests.
    //
    // The hard one is 1.5x, which catches a workgroup size that is catastrophically wrong.
    // It is deliberately loose: `cargo test` runs this file's tests in parallel by default,
    // so the sweep contends with the artifact test for the same GPU and the numbers inflate
    // unevenly. A 15% bar here fails perhaps one run in three for no reason, and a flaky
    // test gets deleted rather than fixed.
    //
    // The soft one is 15%, printed rather than asserted. Run alone
    // (`--test-threads=1 the_design_picked`) the measurement is stable to about 1.5% and 128
    // is within 12% of the best size on every artifact. That is the number the table in
    // `gen::gather::WORKGROUP` records, and the printed line is how a regression gets noticed.
    if worst_ratio > 1.15 {
        println!(
            "NOTE: workgroup {} is {:.0}% off the best size on {worst_at}. Re-run with \
             --test-threads=1 before believing it: under contention these numbers inflate \
             unevenly.",
            wgsl::WORKGROUP,
            (worst_ratio - 1.0) * 100.0
        );
    }
    assert!(
        worst_ratio <= 1.5,
        "workgroup {} is {:.0}% worse than the best size on {worst_at}; \
         change gen::gather::WORKGROUP and say so",
        wgsl::WORKGROUP,
        (worst_ratio - 1.0) * 100.0
    );
    println!(
        "worst case for the shipped size {} over {judged} artifacts: {:+.1}% on {worst_at}",
        wgsl::WORKGROUP,
        (worst_ratio - 1.0) * 100.0
    );
}
