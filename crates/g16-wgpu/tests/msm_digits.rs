//! U8's acceptance tests: the MSM digit pipeline on a real GPU, against a host counting sort
//! written from the algorithm rather than from the kernel.
//!
//! # What is being pinned
//!
//! 1. **`(counts, cursor, entries)` reproduce a host reference exactly, up to within-bucket
//!    order**, at seven window widths from `c = 3` to `c = 16`, with a nonzero `scalar_off`,
//!    and with the entry array pre-filled with a sentinel so a scatter that writes one slot
//!    too far is visible. Every output buffer here is exactly the size the kernel should
//!    write, and WebGPU drops an out-of-range storage write in silence, so without a
//!    sentinel an over-run is *unobservable*. U7 shipped two such bugs into a review.
//! 2. **The top window's carry is zero.** Over 10^5 random scalars and `r - 1` at every `c`
//!    in `2..=16`, and additionally by full reconstruction of the scalar from its digits over
//!    a smaller sample. A nonzero carry there is silent and produces a wrong point.
//! 3. **The device recoding equals the host recoding scalar for scalar**, not just in
//!    aggregate: an entry carries `(point index, sign)`, so comparing the set of entries in
//!    each bucket is an elementwise check on 20 digits of every scalar. Together with 2 that
//!    is what makes the device's recoding exact rather than merely self-consistent.
//! 4. **The degenerate inputs a real witness actually contains**: all zero, all one, 99%
//!    zeros and ones, and 99% ones. A bit-heavy circuit is over 99% zeros and ones, which is
//!    exactly why the window is sized from the general-scalar count and not from `n`.
//! 5. **`fr_mont_to_std` is the conversion and not a copy.** Checked against
//!    `PackedScalar::from_fr` elementwise, and separately asserted to differ from the input
//!    at every index, because a kernel that copied would agree with a Montgomery oracle.
//! 6. **Every pipeline layout declares at most 8 storage buffers**, programmatically, from
//!    the list the layout is actually built from.
//!
//! # Rules the inputs follow
//!
//! **Never symmetric, never degenerate by accident.** Every scalar vector has distinct
//! entries, spans both signs of the recoding at every window, and is asserted to produce a
//! nonzero number of entries before anything is compared. The degenerate vectors are
//! deliberate and are their own test.
//!
//! **Hundreds to thousands, never eight.** The field gate found that a kernel with the
//! conditional subtraction deleted passed an 8-element check 24% of the time.
//!
//! **Fixed seeds**, so a failure reproduces.
//!
//! Native only, for the reason in `tests/device.rs`.

use std::sync::OnceLock;

use g16_field::{Field as _, Fr};
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::{PackedFr, PackedScalar, LIMBS};
use g16_wgpu::gen::msm as wgsl;
use g16_wgpu::msm::{DigitBuffers, DigitPlan, MsmDigits};
use g16_wgpu::{LimitsProfile, ParamRing, Readback, WgpuBackend};
use num_bigint::{BigInt, BigUint};
use num_traits::{One as _, Zero as _};

#[path = "gpulock/mod.rs"]
mod gpulock;

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

/// The pipelines, built once. Compiling the `Fr` prelude is 78 ms of naga and there are a
/// dozen tests here.
fn digits() -> &'static MsmDigits {
    static D: OnceLock<MsmDigits> = OnceLock::new();
    D.get_or_init(|| MsmDigits::new(floor()).expect("digit pipelines"))
}

// ---------------------------------------------------------------------------
// Scalar sets
// ---------------------------------------------------------------------------

/// `n` distinct scalars, none of them 0 or 1, so every one of them reaches a bucket.
fn general(rng: &mut SplitMix64, n: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let x = rng.next_fr();
        if !(x.is_zero() || x.is_one()) {
            out.push(x);
        }
    }
    out
}

/// A witness-shaped vector: `general_ppm` parts per million are general, the rest split
/// between 0 and 1. That is the shape design §5 sizes the window for, and the reason
/// `window_size` takes `m` rather than `n`.
fn witness_shaped(rng: &mut SplitMix64, n: usize, general_ppm: u32) -> Vec<Fr> {
    (0..n)
        .map(|_| {
            let r = (rng.next_u64() % 1_000_000) as u32;
            if r < general_ppm {
                let mut x = rng.next_fr();
                while x.is_zero() || x.is_one() {
                    x = rng.next_fr();
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

/// Every scalar the recoding could plausibly get wrong.
///
/// Deliberately not deduplicated: `2^0` and `Fr::one()` are the same value at two indices,
/// and so are `2^1` and `2^1 - 1 + 1`, which is what makes the per-bucket comparison a check
/// on the *set of point indices* rather than only on the multiset of digits.
///
/// `r - 1` is the one the acceptance names, because it is the largest scalar there is and so
/// the only one whose top window could carry. The rest cover the borrow boundary at every
/// window (`2^(c-1)` exactly, and one either side), the all-ones low word, powers of two on
/// and off a limb boundary, and the values that make `sc_bits`'s shift-by-32 trap fire if it
/// were written the obvious way.
fn adversarial() -> Vec<Fr> {
    let mut v: Vec<Fr> = vec![
        Fr::zero(),
        Fr::one(),
        Fr::from(2u64),
        Fr::from(3u64),
        -Fr::one(),                    // r - 1, the top-carry witness
        -Fr::from(2u64),               // r - 2
        -Fr::from(3u64),               // r - 3
        Fr::from(u32::MAX as u64),     // all ones in limb 0
        Fr::from(u32::MAX as u64 + 1), // the limb boundary
        Fr::from(u64::MAX),            // all ones in limbs 0 and 1
    ];
    // Every power of two the scalar field can hold, so every window in every `c` sees both a
    // lone set bit at its bottom and a lone set bit at its sign position.
    for bit in 0..254u32 {
        v.push(Fr::from(2u64).pow([u64::from(bit)]));
    }
    // One either side of every power of two, which is where the borrow flips.
    for bit in 1..254u32 {
        let p = Fr::from(2u64).pow([u64::from(bit)]);
        v.push(p - Fr::one());
        v.push(p + Fr::one());
    }
    v
}

// ---------------------------------------------------------------------------
// The host reference: a counting sort written from the algorithm, not the kernel
// ---------------------------------------------------------------------------

/// Standard-form limbs, which is exactly what the device buffer holds.
fn std_words(x: &Fr) -> [u32; LIMBS] {
    PackedScalar::from_fr(x).v
}

fn to_big(words: &[u32; LIMBS]) -> BigUint {
    BigUint::from_slice(words)
}

/// `width` bits of `k` starting at `off`, one bit at a time.
///
/// Deliberately not the kernel's word-shifting form: a reference that reproduced the shader's
/// arithmetic would agree with it about a shift bug. `BigUint::bit` reads past the top as
/// false, which is the same convention `sc_pick` gets from returning zero out of range.
fn ref_bits(k: &BigUint, off: usize, width: u32) -> u64 {
    let mut v = 0u64;
    for j in 0..width as usize {
        if k.bit((off + j) as u64) {
            v |= 1u64 << j;
        }
    }
    v
}

/// Digit `w` of the width-`c` signed recoding, as the signed integer it is.
///
/// `b_w - 2^c * [b_w >= 2^(c-1)] + bit(w*c - 1)`, straight from the definition.
fn ref_digit(k: &BigUint, w: usize, c: u32) -> i64 {
    let off = w * c as usize;
    let b = ref_bits(k, off, c) as i64;
    let borrow = (b >> (c - 1)) & 1;
    let carry = if off == 0 {
        0
    } else {
        i64::from(k.bit((off - 1) as u64))
    };
    b - (borrow << c) + carry
}

fn n_windows(c: u32) -> usize {
    255usize.div_ceil(c as usize)
}

/// What the device should produce, for one scalar range at one window width.
struct Reference {
    /// `n_windows * n_buckets` counters.
    counts: Vec<u32>,
    /// Exclusive prefix of `counts` inside each window, biased by `w * cap`. This is what
    /// `msm_scan` writes and what `msm_scatter` starts from.
    starts: Vec<u32>,
    /// Per row, the `(point index, sign)` pairs that belong in it, sorted so the comparison
    /// is up to within-bucket order.
    rows: Vec<Vec<(u32, u32)>>,
}

fn reference(scalars: &[Fr], scalar_off: usize, n: usize, c: u32, cap: u32) -> Reference {
    let w_count = n_windows(c);
    let n_buckets = 1usize << (c - 1);
    let rows_len = w_count * n_buckets;
    let mut counts = vec![0u32; rows_len];
    let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); rows_len];

    for i in 0..n {
        let x = &scalars[scalar_off + i];
        // The two classes that never reach a bucket.
        if x.is_zero() || x.is_one() {
            continue;
        }
        let k = to_big(&std_words(x));
        for w in 0..w_count {
            let d = ref_digit(&k, w, c);
            if d == 0 {
                continue;
            }
            let mag = d.unsigned_abs() as usize;
            assert!(
                mag <= n_buckets,
                "digit {d} at c={c} w={w} is outside [-2^(c-1), 2^(c-1)]"
            );
            let row = w * n_buckets + (mag - 1);
            counts[row] += 1;
            rows[row].push((i as u32, u32::from(d < 0)));
        }
    }

    let mut starts = vec![0u32; rows_len];
    for w in 0..w_count {
        let mut running = w as u32 * cap;
        for bkt in 0..n_buckets {
            starts[w * n_buckets + bkt] = running;
            running += counts[w * n_buckets + bkt];
        }
    }
    for r in &mut rows {
        r.sort_unstable();
    }
    Reference {
        counts,
        starts,
        rows,
    }
}

// ---------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------

fn storage_words(backend: &WgpuBackend, label: &str, data: &[u32]) -> wgpu::Buffer {
    let bytes = (data.len().max(1) * 4) as u64;
    let buf = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        backend
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
    }
    buf
}

fn fill(backend: &WgpuBackend, buf: &wgpu::Buffer, word: u32) {
    let words = (buf.size() / 4) as usize;
    backend
        .queue()
        .write_buffer(buf, 0, bytemuck::cast_slice(&vec![word; words]));
}

fn read_words(backend: &WgpuBackend, buf: &wgpu::Buffer) -> Vec<u32> {
    let bytes = buf.size();
    let rb = Readback::new(backend, "u8 readback", bytes).expect("readback");
    let mut enc = backend.device().create_command_encoder(&Default::default());
    rb.copy_from(&mut enc, buf, 0, bytes).expect("copy");
    let raw = pollster::block_on(rb.submit_and_read(backend, enc, bytes)).expect("read");
    bytemuck::cast_slice(&raw).to_vec()
}

/// The sentinel every output buffer is pre-filled with. Not zero, so a kernel that writes
/// nothing is caught; not a plausible count or offset either.
const SENTINEL: u32 = 0xDEAD_BEEF;

struct SortRun {
    counts: Vec<u32>,
    cursor: Vec<u32>,
    entries: Vec<u32>,
}

/// Uploads the scalars, runs the whole counting sort in one submit, reads the three buffers
/// back.
///
/// `slack` extra elements are allocated on every output and pre-filled with [`SENTINEL`], so
/// a kernel that writes one element past its range is visible instead of being dropped by
/// WebGPU's out-of-range write rule.
fn run_sort(d: &MsmDigits, plan: &DigitPlan, words: &[u32], slack: u32) -> SortRun {
    run_sort_reporting(d, plan, words, slack).0
}

fn run_sort_reporting(
    d: &MsmDigits,
    plan: &DigitPlan,
    words: &[u32],
    slack: u32,
) -> (SortRun, u64) {
    let b = floor();
    let scalars = storage_words(b, "u8 scalars", words);
    let bufs = DigitBuffers::new(b, plan, slack).expect("digit buffers");
    fill(b, &bufs.counts, SENTINEL);
    fill(b, &bufs.cursor, SENTINEL);
    fill(b, &bufs.entries, SENTINEL);

    let mut ring = ParamRing::new(b, "u8 ring", d.sort_slots(plan) + 4).expect("ring");
    let offsets = d.plan_sort(plan, &mut ring).expect("plan");
    ring.flush(b);
    let binds = d.bind_sort(b, &ring, plan, &scalars, &bufs).expect("bind");

    let mut enc = b.device().create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        d.encode_sort(&mut pass, plan, &binds, &offsets)
            .expect("encode");
    }
    b.submit([enc.finish()]);
    b.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    assert!(b.take_error().is_none(), "device error during the sort");

    (
        SortRun {
            counts: read_words(b, &bufs.counts),
            cursor: read_words(b, &bufs.cursor),
            entries: read_words(b, &bufs.entries),
        },
        bufs.bytes(),
    )
}

/// Compares one device run against the reference, and returns how many entries were checked
/// so a caller can assert the test was not vacuous.
fn check_against_reference(
    run: &SortRun,
    r: &Reference,
    plan: &DigitPlan,
    slack: u32,
    what: &str,
) -> u64 {
    let rows = plan.rows() as usize;
    assert_eq!(
        &run.counts[..rows],
        &r.counts[..],
        "{what}: counts disagree with the host reference"
    );
    // After the scatter the cursor holds each run's END, which is start + count. Checking the
    // end rather than the start is strictly stronger: it catches a scan that produced the
    // right offsets and a scatter that bumped the wrong row.
    let ends: Vec<u32> = r.starts.iter().zip(&r.counts).map(|(s, c)| s + c).collect();
    assert_eq!(
        &run.cursor[..rows],
        &ends[..],
        "{what}: cursor after the scatter is not start + count"
    );

    let mut checked = 0u64;
    let mut seen = vec![false; (plan.entries() + slack) as usize];
    for (row, &end) in ends.iter().enumerate().take(rows) {
        let end = end as usize;
        let start = r.starts[row] as usize;
        let mut got: Vec<(u32, u32)> = (start..end)
            .map(|slot| {
                seen[slot] = true;
                let e = run.entries[slot * 2 + 1];
                let stored_row = run.entries[slot * 2];
                assert_eq!(
                    stored_row, row as u32,
                    "{what}: entry at slot {slot} carries row {stored_row}, expected {row}"
                );
                (e >> 1, e & 1)
            })
            .collect();
        got.sort_unstable();
        assert_eq!(
            got, r.rows[row],
            "{what}: bucket row {row} holds the wrong (point, sign) set"
        );
        checked += got.len() as u64;
    }

    // Every slot outside a run must still hold the sentinel. This is the check that makes an
    // over-run visible: the entry array is exactly `n_windows * cap` long, so without it a
    // scatter that walked one slot too far would write out of range and WebGPU would drop it.
    for (slot, touched) in seen.iter().enumerate() {
        if !touched {
            assert_eq!(
                run.entries[slot * 2],
                SENTINEL,
                "{what}: slot {slot} is outside every run and was written"
            );
            assert_eq!(run.entries[slot * 2 + 1], SENTINEL, "{what}: slot {slot}.y");
        }
    }
    // And the slack past the counter rows, which zero_u32 and msm_scan must both leave alone.
    for i in rows..(plan.rows() + slack) as usize {
        assert_eq!(run.counts[i], SENTINEL, "{what}: counts slack at {i}");
        assert_eq!(run.cursor[i], SENTINEL, "{what}: cursor slack at {i}");
    }
    checked
}

// ---------------------------------------------------------------------------
// 1. The acceptance test: the sort reproduces a host counting sort at several c
// ---------------------------------------------------------------------------

#[test]
fn the_counting_sort_matches_a_host_reference_at_every_window_width() {
    let d = digits();
    // c = 3 is the smallest the model will ever pick, 16 the largest; 5 divides 255 exactly
    // so W*c is 255 on the nose and the top window's sign bit sits at 254; 8 and 16 make
    // every window start on a byte boundary and so exercise `sh == 0` in `sc_bits`, which is
    // the shift-by-32 trap; 11 and 13 are the widths Metal actually chooses.
    const WIDTHS: [u32; 7] = [3, 5, 8, 11, 13, 16, 4];
    let mut total = 0u64;
    let mut rows = String::new();

    for &c in &WIDTHS {
        // A different vector per width, and a nonzero scalar_off on half of them, which is
        // the L MSM's shape: its scalars are the private suffix of the witness and it shares
        // the buffer with A and B rather than uploading a second copy.
        let mut rng = SplitMix64(0x0C8_0001 ^ u64::from(c));
        let all = general(&mut rng, 2400);
        let off = if c % 2 == 0 { 371 } else { 0 };
        let n = all.len() - off;
        let words: Vec<u32> = all.iter().flat_map(std_words).collect();

        let plan = DigitPlan::with_c(n as u32, off as u32, Some(n as u32), c).expect("plan");
        let slack = 3;
        let (run, scratch) = run_sort_reporting(d, &plan, &words, slack);
        let r = reference(&all, off, n, c, plan.cap());
        let checked = check_against_reference(&run, &r, &plan, slack, &format!("c={c}"));

        // Not vacuous: every one of the 2400 scalars is general, and at c = 16 the sparsest
        // window still fills most of them, so the entry count is in the tens of thousands.
        assert!(
            checked > n as u64,
            "c={c}: only {checked} entries, the sort cannot have run"
        );
        total += checked;
        rows.push_str(&format!(
            "c={c:<3} W={:<3} buckets={:<6} off={off:<4} n={n:<5} entries={checked:<7} \
             scratch {:.2} MiB\n",
            plan.n_windows(),
            plan.n_buckets(),
            scratch as f64 / (1024.0 * 1024.0)
        ));
    }
    println!("{rows}{total} entries checked against the host reference in total");
}

// ---------------------------------------------------------------------------
// 2. The scan on its own, because the combined check would hide a scan that is wrong
//    in a way the scatter happens to undo
// ---------------------------------------------------------------------------

#[test]
fn the_scan_alone_produces_the_exclusive_prefix_the_scatter_needs() {
    let b = floor();
    let d = digits();
    let mut rows = String::new();
    // c = 2 gives 2 buckets, far under one workgroup; c = 9 gives 256, exactly one; c = 16
    // gives 32768, 128 chunks, which is the path that carries a running total across chunks
    // and the only one where the chunk loop can be wrong.
    for c in [2u32, 3, 9, 10, 13, 16] {
        let mut rng = SplitMix64(0x5CA1 ^ u64::from(c));
        let all = general(&mut rng, 1500);
        let n = all.len() as u32;
        let words: Vec<u32> = all.iter().flat_map(std_words).collect();
        let plan = DigitPlan::with_c(n, 0, Some(n), c).expect("plan");

        let scalars = storage_words(b, "scan scalars", &words);
        let bufs = DigitBuffers::new(b, &plan, 2).expect("bufs");
        fill(b, &bufs.counts, SENTINEL);
        fill(b, &bufs.cursor, SENTINEL);

        let mut ring = ParamRing::new(b, "scan ring", d.sort_slots(&plan) + 4).unwrap();
        let offsets = d.plan_sort(&plan, &mut ring).unwrap();
        ring.flush(b);
        let binds = d.bind_sort(b, &ring, &plan, &scalars, &bufs).unwrap();

        // zero, count and scan only. The scatter is what turns the cursor from starts into
        // ends, so leaving it out is the only way to see what the scan wrote.
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            d.encode_zero_rows(&mut pass, &plan, &binds, &offsets)
                .unwrap();
            d.encode_count(&mut pass, &plan, &binds, &offsets).unwrap();
            d.encode_scan(&mut pass, &plan, &binds, &offsets).unwrap();
        }
        b.submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert!(b.take_error().is_none());

        let cursor = read_words(b, &bufs.cursor);
        let counts = read_words(b, &bufs.counts);
        let r = reference(&all, 0, n as usize, c, plan.cap());
        let rows_len = plan.rows() as usize;
        assert_eq!(&counts[..rows_len], &r.counts[..], "c={c}: counts");
        assert_eq!(
            &cursor[..rows_len],
            &r.starts[..],
            "c={c}: the scan did not write the exclusive prefix biased by w*cap"
        );
        // The scan must write every row, including the ones whose count is zero, or the
        // scatter starts from a sentinel.
        assert!(
            !cursor[..rows_len].contains(&SENTINEL),
            "c={c}: the scan left a row unwritten"
        );
        let nonzero = r.counts.iter().filter(|&&x| x > 0).count();
        rows.push_str(&format!(
            "c={c:<3} rows={rows_len:<6} chunks={:<4} nonempty buckets={nonzero}\n",
            plan.n_buckets().div_ceil(d.workgroups().scan)
        ));
    }
    println!("{rows}");
}

// ---------------------------------------------------------------------------
// 3. The top window's carry, which is silent when it is wrong
// ---------------------------------------------------------------------------

#[test]
fn the_top_windows_carry_is_zero_over_a_hundred_thousand_scalars() {
    // 10^5 random scalars plus every adversarial one, at every width the model can pick.
    // The property being checked is `e_(W-1) = [b_(W-1) >= 2^(c-1)] == 0`, which is exactly
    // "the top window does not borrow": the recoding telescopes to
    // `k - e_(W-1) * 2^(W*c)`, so a nonzero carry there loses 2^(W*c) with nothing to
    // observe. The full reconstruction below is what checks that telescoping claim itself
    // rather than trusting the algebra.
    let mut rng = SplitMix64(0x0CA_7711);
    let mut ks: Vec<BigUint> = (0..100_000)
        .map(|_| to_big(&std_words(&rng.next_fr())))
        .collect();
    ks.extend(adversarial().iter().map(|x| to_big(&std_words(x))));
    let r_minus_1 = to_big(&std_words(&-Fr::one()));
    assert!(ks.contains(&r_minus_1), "r - 1 must be in the sample");

    let mut worst_top = 0u64;
    let mut worst_mag = 0i64;
    for c in 2..=16u32 {
        let w_count = n_windows(c);
        let top_off = (w_count - 1) * c as usize;
        for k in &ks {
            let b_top = ref_bits(k, top_off, c);
            assert_eq!(
                (b_top >> (c - 1)) & 1,
                0,
                "c={c}: the top window's raw bits are {b_top:#x}, which borrows, so the \
                 recoding loses 2^{}",
                w_count * c as usize
            );
            worst_top = worst_top.max(b_top);
            // Every digit's magnitude has to index a real bucket, or the atomicAdd lands
            // outside the counter array and WebGPU drops it in silence.
            for w in 0..w_count {
                let d = ref_digit(k, w, c);
                assert!(
                    d.abs() <= 1i64 << (c - 1),
                    "c={c} w={w}: digit {d} is outside [-2^(c-1), 2^(c-1)]"
                );
                worst_mag = worst_mag.max(d.abs());
            }
        }
    }
    println!(
        "{} scalars x c in 2..=16: no top-window borrow; largest top window {worst_top}, \
         largest digit magnitude {worst_mag}",
        ks.len()
    );

    // The exact statement, on a smaller sample because it is BigInt arithmetic: the digits
    // sum back to the scalar. If the top window ever carried, this is the equality that
    // would break, and it breaks by exactly 2^(W*c).
    let sample: Vec<&BigUint> = ks
        .iter()
        .take(1500)
        .chain(ks.iter().rev().take(800))
        .collect();
    for c in 2..=16u32 {
        let w_count = n_windows(c);
        for k in &sample {
            let mut acc = BigInt::zero();
            let mut shift = BigInt::one();
            let step = BigInt::from(1u64) << c;
            for w in 0..w_count {
                acc += BigInt::from(ref_digit(k, w, c)) * &shift;
                shift *= &step;
            }
            assert_eq!(
                acc,
                BigInt::from((*k).clone()),
                "c={c}: the digits do not sum back to the scalar"
            );
        }
    }
    println!(
        "{} scalars x c in 2..=16: the digits reconstruct the scalar exactly",
        sample.len()
    );
}

// ---------------------------------------------------------------------------
// 4. The device half of 3: the kernel agrees with the host on the extremes
// ---------------------------------------------------------------------------

#[test]
fn the_device_recoding_agrees_with_the_host_on_r_minus_one_and_the_extremes() {
    let d = digits();
    let all = adversarial();
    let n = all.len() as u32;
    let words: Vec<u32> = all.iter().flat_map(std_words).collect();
    let general_count = all.iter().filter(|x| !(x.is_zero() || x.is_one())).count() as u32;
    assert!(general_count > 700, "the adversarial set got smaller");

    let mut line = String::new();
    for c in [2u32, 3, 5, 7, 8, 11, 13, 16] {
        let plan = DigitPlan::with_c(n, 0, Some(general_count), c).expect("plan");
        let run = run_sort(d, &plan, &words, 2);
        let r = reference(&all, 0, n as usize, c, plan.cap());
        let checked = check_against_reference(&run, &r, &plan, 2, &format!("adversarial c={c}"));
        // The entry set is compared per bucket including the sign bit, so this is an
        // elementwise check of every digit of every one of these scalars, r - 1 included.
        line.push_str(&format!("c={c} {checked} entries  "));
    }
    println!("{n} adversarial scalars ({general_count} general, r-1 among them): {line}");
}

// ---------------------------------------------------------------------------
// 5. The degenerate witnesses a real circuit contains
// ---------------------------------------------------------------------------

#[test]
fn the_degenerate_witnesses_a_real_circuit_contains() {
    let d = digits();
    let n = 4096usize;
    let cases: [(&str, Vec<Fr>); 6] = [
        ("all zero", vec![Fr::zero(); n]),
        ("all one", vec![Fr::one(); n]),
        (
            "99% zeros and ones",
            witness_shaped(&mut SplitMix64(0x05A_9E01), n, 10_000),
        ),
        (
            "99% ones",
            (0..n)
                .map(|i| {
                    if i % 128 == 7 {
                        let mut rng = SplitMix64(0xA111 ^ i as u64);
                        let mut x = rng.next_fr();
                        while x.is_zero() || x.is_one() {
                            x = rng.next_fr();
                        }
                        x
                    } else {
                        Fr::one()
                    }
                })
                .collect(),
        ),
        (
            "one general scalar in four thousand",
            (0..n)
                .map(|i| {
                    if i == 2831 {
                        Fr::from(12345u64)
                    } else {
                        Fr::zero()
                    }
                })
                .collect(),
        ),
        // H's shape, and the only case in this file that reaches `window_size` at a width
        // where `255 / c` and `255.div_ceil(c)` differ. Everything else fixes `c` through
        // `with_c`, and a mutation replacing that ceiling with a floor slipped the whole
        // suite because of exactly that gap.
        (
            "every scalar general, as H is",
            general(&mut SplitMix64(0x0DE5_0001), n),
        ),
    ];

    for (name, all) in cases {
        // `pack_scalars` is the host packer: one pass producing the standard-form limbs and
        // the general-scalar count together, because that count is what the window is sized
        // from and walking the vector a second time for it would be free work.
        let (words, general_count) = g16_wgpu::msm::pack_scalars(&all);
        assert_eq!(
            words,
            all.iter().flat_map(std_words).collect::<Vec<u32>>(),
            "{name}: pack_scalars does not agree with PackedScalar::from_fr"
        );
        // The design's "c is chosen from m, not from the input length", checked rather than
        // assumed: a 4096-long witness with 43 general scalars is a 43 scalar problem.
        let c = g16_wgpu::window_size(general_count as usize);
        let plan = DigitPlan::new(n as u32, 0, Some(general_count)).expect("plan");
        assert_eq!(plan.c(), c);
        assert_eq!(
            plan.n_windows(),
            255u32.div_ceil(c),
            "{name}: the window count is not ceil(255/c)"
        );
        let run = run_sort(d, &plan, &words, 2);
        let r = reference(&all, 0, n, plan.c(), plan.cap());
        let checked = check_against_reference(&run, &r, &plan, 2, name);

        // The claim being tested is not just "it matches the reference". It is that no zero
        // and no one scalar reaches a bucket. The signed recoding sends every scalar equal to
        // 1 to digit +1 of window 0, so without the classification bucket (0, 0) would hold
        // one entry per one-scalar, and since one thread owns one bucket that thread would
        // serially accumulate the whole witness. The "all one" case below is exactly that
        // trap: 4096 ones, and bucket (0, 0) must be empty.
        let rows = plan.rows() as usize;
        let bucket_00 = run.counts[0];
        let ones = all.iter().filter(|x| x.is_one()).count();
        assert_eq!(
            bucket_00, r.counts[0],
            "{name}: bucket (0,0) holds {bucket_00}, the reference says {}",
            r.counts[0]
        );
        assert!(
            u64::from(bucket_00) < ones as u64 || ones == 0,
            "{name}: bucket (0,0) holds {bucket_00} of the {ones} one-scalars, so the \
             classification is not keeping them out"
        );
        // Every entry that exists came from a general scalar, checked against the input
        // rather than against the reference that produced it.
        for (row, pairs) in r.rows.iter().enumerate() {
            for &(i, _) in pairs {
                assert!(
                    !(all[i as usize].is_zero() || all[i as usize].is_one()),
                    "{name}: a 0 or 1 scalar reached bucket row {row}"
                );
            }
        }
        // And nothing was counted at all when nothing is general, which is the all-zero and
        // all-one cases and is where a kernel that forgot the classification would show a
        // count of 4096 rather than a mismatch of one.
        let total: u64 = run.counts[..rows].iter().map(|&x| u64::from(x)).sum();
        assert_eq!(
            total, checked,
            "{name}: counts and entries disagree on the total"
        );
        if general_count == 0 {
            assert_eq!(
                total, 0,
                "{name}: no scalar is general and yet {total} were counted"
            );
        } else {
            assert!(
                total > 0,
                "{name}: {general_count} general scalars and no entries"
            );
        }
        println!(
            "{name:<34} n={n} general={general_count:<5} c={:<3} entries={checked:<6} \
             bucket(0,0)={bucket_00} of {ones} ones",
            plan.c()
        );
    }
}

// ---------------------------------------------------------------------------
// 6. fr_mont_to_std
// ---------------------------------------------------------------------------

#[test]
fn fr_mont_to_std_matches_the_host_encoder() {
    let b = floor();
    let d = digits();
    let mut rng = SplitMix64(0x0F0F_1234);
    let mut xs: Vec<Fr> = adversarial();
    xs.extend(general(&mut rng, 3000));
    let n = xs.len() as u32;

    let mont: Vec<u32> = xs.iter().flat_map(|x| PackedFr::from_fr(x).v).collect();
    let want: Vec<u32> = xs.iter().flat_map(std_words).collect();
    // A kernel that copied its input would pass against a Montgomery oracle and fail here,
    // which is the swap `tests/stages.rs` documents as producing a proof wrong by a factor
    // of R and nothing else to go on.
    let differ = mont
        .chunks_exact(LIMBS)
        .zip(want.chunks_exact(LIMBS))
        .filter(|(a, c)| a != c)
        .count();
    assert!(
        differ >= xs.len() - 2,
        "only {differ} of {} scalars change under from_mont; the two encodings agree at 0 \
         and nowhere else, so this oracle is broken",
        xs.len()
    );

    let src = storage_words(b, "mont src", &mont);
    let dst = storage_words(b, "mont dst", &vec![SENTINEL; want.len() + LIMBS]);
    let mut ring = ParamRing::new(b, "mont ring", 8).unwrap();
    let offsets = d.plan_mont(n, &mut ring).unwrap();
    ring.flush(b);
    let bind = d.bind_mont(b, &ring, n, &src, &dst).unwrap();
    let mut enc = b.device().create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        d.encode_mont(&mut pass, &bind, n, &offsets).unwrap();
    }
    b.submit([enc.finish()]);
    b.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    assert!(b.take_error().is_none());

    let got = read_words(b, &dst);
    assert_eq!(&got[..want.len()], &want[..], "fr_mont_to_std");
    // One element of slack, untouched, because an off-by-one past the range is otherwise
    // dropped in silence.
    assert!(
        got[want.len()..].iter().all(|&x| x == SENTINEL),
        "fr_mont_to_std wrote past its range"
    );
    println!("fr_mont_to_std: {n} scalars, {differ} of them change under from_mont");
}

// ---------------------------------------------------------------------------
// 7. zero_u32
// ---------------------------------------------------------------------------

#[test]
fn zero_u32_clears_exactly_the_range_it_was_given() {
    let b = floor();
    // A range that is not a multiple of any workgroup size, so the last workgroup is partial
    // and an unguarded write lands in the slack.
    const N: u32 = 1000;
    for wgpd in [65535u32, 1, 2] {
        let d = MsmDigits::with_shape(
            b,
            wgsl::Workgroups::default(),
            wgsl::LimbPick::default(),
            wgpd,
        )
        .expect("pipelines");
        let buf = storage_words(b, "zero target", &vec![SENTINEL; (N + 5) as usize]);
        let mut ring = ParamRing::new(b, "zero ring", 64).unwrap();
        let offsets = d.plan_zero(N, &mut ring).unwrap();
        ring.flush(b);
        let bind = d.bind_zero(b, &ring, &buf);
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            d.encode_zero(&mut pass, &bind, N, &offsets).unwrap();
        }
        b.submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let got = read_words(b, &buf);
        assert!(
            got[..N as usize].iter().all(|&x| x == 0),
            "{} dispatches: the range was not cleared",
            offsets.len()
        );
        assert!(
            got[N as usize..].iter().all(|&x| x == SENTINEL),
            "{} dispatches: zero_u32 wrote past its range",
            offsets.len()
        );
        println!(
            "zero_u32 over {N} words in {} dispatches: slack untouched",
            offsets.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 8. The floor's storage-buffer budget
// ---------------------------------------------------------------------------

#[test]
fn every_digit_pipeline_layout_declares_at_most_eight_storage_buffers() {
    let d = digits();
    let counts = MsmDigits::storage_buffer_counts();
    let want = [
        (wgsl::ENTRY_ZERO, wgsl::STORAGE_ZERO),
        (wgsl::ENTRY_MONT, wgsl::STORAGE_MONT),
        (wgsl::ENTRY_COUNT, wgsl::STORAGE_COUNT),
        (wgsl::ENTRY_SCAN, wgsl::STORAGE_SCAN),
        (wgsl::ENTRY_SCATTER, wgsl::STORAGE_SCATTER),
    ];
    assert_eq!(
        counts, want,
        "the generator's constants and the layouts disagree"
    );
    for (name, n) in counts {
        // 8 is the floor. Under STRICT_WEBGPU_COMPLIANCE this adapter reports 9 and the
        // Raised profile would buy exactly one more, so there is no headroom to spend.
        assert!(
            n <= 8,
            "{name} declares {n} storage buffers, over the floor's 8"
        );
    }
    let cost = d.cost();
    println!(
        "{}\n{:?} module shape, {} modules, {} pipelines, {} bytes of WGSL, {:.1} KiB, \
         built in {:.1} ms",
        counts
            .iter()
            .map(|(n, c)| format!("{n}: {c} storage + 1 uniform in 1 bind group"))
            .collect::<Vec<_>>()
            .join("\n"),
        d.module_shape(),
        cost.modules,
        cost.pipelines,
        d.source_len(),
        d.source_len() as f64 / 1024.0,
        cost.compile_us as f64 / 1e3
    );
}

// ---------------------------------------------------------------------------
// 8b. The one bug in here that no behavioural test can catch
// ---------------------------------------------------------------------------

/// `msm_scan`'s Hillis-Steele step needs a barrier between the read and the write, and a
/// missing one is a race that this suite cannot see.
///
/// Found by mutation testing: deleting `workgroupBarrier()` between `x = SCAN[tid - d]` and
/// `SCAN[tid] = SCAN[tid] + x` leaves every test in this file **passing**, because on this
/// hardware the 32 lanes of a SIMD group run in lockstep and enough of the workgroup stays
/// in step for the answer to come out right anyway. It is still a race: WGSL's memory model
/// gives no ordering between two invocations' accesses to workgroup memory without one, and
/// a wider SIMD group or a compiler that reorders the load would break it. Racing code that
/// happens to work is not correct code, and it is exactly the kind of thing that works on
/// the machine it was written on and fails in a browser on somebody else's GPU.
///
/// So the shape is pinned instead of the behaviour. Four barriers in `msm_scan`: one after
/// the chunk load, one either side of the accumulate, one before the next chunk overwrites
/// `SCAN`. Assert on the generated text and say plainly that this is a weaker test than a
/// behavioural one, rather than pretend the suite covers it.
#[test]
fn the_scans_barriers_are_pinned_because_a_missing_one_races_silently() {
    let src = wgsl::digits_module();
    let scan = &src[src.find("fn msm_scan").expect("msm_scan")..];
    let scan = &scan[..scan.find("\n// After this kernel").unwrap_or(scan.len())];
    let barriers = scan.matches("workgroupBarrier()").count();
    assert_eq!(
        barriers, 4,
        "msm_scan has {barriers} barriers, expected 4. Deleting the one between the scan's \
         read and its write is a race every test in this file passes through; if this count \
         changed on purpose, change it here and say why."
    );
    // And the two that bracket the accumulate really do bracket it, rather than both sitting
    // on the same side of it.
    let acc = scan
        .find("SCAN[tid] = SCAN[tid] + x;")
        .expect("the accumulate");
    let before = scan[..acc].matches("workgroupBarrier()").count();
    assert_eq!(
        before, 2,
        "the accumulate has {before} barriers before it, expected 2 (the chunk load's and \
         the read's)"
    );
    // Every loop that contains a barrier must have a bound WGSL can see as uniform, or the
    // shader is rejected outright rather than merely being wrong. The inner one is literal;
    // the outer one is a uniform-buffer read, which naga's uniformity analysis accepts.
    assert!(
        scan.contains("for (var d = 1u; d < 256u; d = d << 1u)"),
        "the inner scan loop no longer has a literal bound"
    );
    println!("msm_scan: {barriers} barriers, {before} of them before the accumulate");
}

// ---------------------------------------------------------------------------
// 9. Chunked dispatch, which no artifact reaches and which therefore ships untested
//    unless it is forced
// ---------------------------------------------------------------------------

#[test]
fn the_counting_sort_survives_being_split_across_dispatches() {
    let b = floor();
    let mut rng = SplitMix64(0x0C40_0F1E);
    let all = general(&mut rng, 1700);
    let n = all.len() as u32;
    let words: Vec<u32> = all.iter().flat_map(std_words).collect();
    let c = 11;

    let mut shapes = Vec::new();
    let mut counts: Vec<(usize, usize, usize)> = Vec::new();
    for wgpd in [65535u32, 3, 1] {
        let d = MsmDigits::with_shape(
            b,
            wgsl::Workgroups::default(),
            wgsl::LimbPick::default(),
            wgpd,
        )
        .expect("pipelines");
        let plan = DigitPlan::with_c(n, 0, Some(n), c).expect("plan");
        let mut ring = ParamRing::new(b, "chunk ring", d.sort_slots(&plan) + 4).unwrap();
        let offsets = d.plan_sort(&plan, &mut ring).unwrap();
        let run = run_sort(&d, &plan, &words, 2);
        let r = reference(&all, 0, n as usize, c, plan.cap());
        check_against_reference(&run, &r, &plan, 2, &format!("wgpd={wgpd}"));
        counts.push((
            offsets.zero.len(),
            offsets.count.len(),
            offsets.scatter.len(),
        ));
        shapes.push(format!(
            "{} workgroups/dispatch: {} dispatches ({} zero, {} count, 1 scan, {} scatter)",
            wgpd,
            offsets.total(),
            offsets.zero.len(),
            offsets.count.len(),
            offsets.scatter.len()
        ));
    }
    // The point of the test: the smallest shape really did split the work, so the multi
    // dispatch path was exercised rather than the single dispatch one three times. Counted
    // rather than assumed, and against the shipped workgroup size rather than a literal,
    // because a literal here would go stale the next time the sweep moves a constant.
    let (first, last) = (&counts[0], counts.last().unwrap());
    assert_eq!(
        *first,
        (1, 1, 1),
        "the unchunked shape used more than one dispatch each"
    );
    assert!(
        last.0 > 1 && last.1 > 1 && last.2 > 1,
        "the forced shape did not actually chunk: {last:?}"
    );
    println!("{}", shapes.join("\n"));
}

// ---------------------------------------------------------------------------
// 10. The measured constants
// ---------------------------------------------------------------------------

/// Microseconds for one encode of `f`, with the fixed cost of a submit and a fence
/// differenced out. Same harness as `tests/stages.rs`, and the same caveat: it cancels the
/// fixed submit cost but not the per-encode host cost, so it is only meaningful in release.
fn timed(reps: u32, mut f: impl FnMut(&mut wgpu::ComputePass<'_>)) -> Option<f64> {
    let b = floor();
    let once = |f: &mut dyn FnMut(&mut wgpu::ComputePass<'_>), passes: u32| -> u128 {
        let t0 = std::time::Instant::now();
        let mut enc = b.device().create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..passes {
                f(&mut pass);
            }
        }
        b.queue().submit([enc.finish()]);
        b.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        t0.elapsed().as_micros()
    };
    for _ in 0..2 {
        once(&mut f, 1);
        once(&mut f, 1 + reps);
    }
    let mut one: Vec<u128> = (0..5).map(|_| once(&mut f, 1)).collect();
    let mut many: Vec<u128> = (0..5).map(|_| once(&mut f, 1 + reps)).collect();
    one.sort_unstable();
    many.sort_unstable();
    (many[2] > one[2]).then(|| (many[2] - one[2]) as f64 / reps as f64)
}

/// Which kernel a sweep cell is timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kern {
    Zero,
    Mont,
    Count,
    Scan,
    Scatter,
}

const KERNS: [(Kern, &str); 5] = [
    (Kern::Zero, "zero_u32"),
    (Kern::Mont, "fr_mont_to_std"),
    (Kern::Count, "msm_count"),
    (Kern::Scan, "msm_scan"),
    (Kern::Scatter, "msm_scatter"),
];

/// Microseconds per dispatch group for one kernel, at one shape.
///
/// **`msm_scatter` is the only one that is not idempotent**: it bumps a cursor, so every
/// repeat pushes its writes past the end of the run and times a different thing. Its cell is
/// therefore `scan + scatter` per repeat, with the scan resetting the cursor, minus a `scan`
/// measured in the same setup. Every other kernel's own size is the only thing varied, so
/// that subtracted baseline is constant across a row.
fn measure(kern: Kern, wg: wgsl::Workgroups, pick: wgsl::LimbPick, n: u32) -> Option<f64> {
    let b = floor();
    let d = MsmDigits::with_shape(b, wg, pick, 65535).expect("pipelines");
    let mut rng = SplitMix64(0x5033_00AB);
    // All general, so the kernels do the work the hot path does rather than exiting early on
    // the 0 and 1 classes. A witness-shaped vector would measure the early exit.
    let all = general(&mut rng, n as usize);
    let words: Vec<u32> = all.iter().flat_map(std_words).collect();
    let mont: Vec<u32> = all.iter().flat_map(|x| PackedFr::from_fr(x).v).collect();
    let plan = DigitPlan::with_c(n, 0, Some(n), 13).expect("plan");

    let scalars = storage_words(b, "sweep scalars", &words);
    let src = storage_words(b, "sweep mont", &mont);
    let dst = storage_words(b, "sweep std", &vec![0u32; words.len()]);
    let bufs = DigitBuffers::new(b, &plan, 0).expect("bufs");

    let mut ring = ParamRing::new(b, "sweep ring", d.sort_slots(&plan) + 16).unwrap();
    let so = d.plan_sort(&plan, &mut ring).unwrap();
    let mo = d.plan_mont(n, &mut ring).unwrap();
    ring.flush(b);
    let binds = d.bind_sort(b, &ring, &plan, &scalars, &bufs).unwrap();
    let mbind = d.bind_mont(b, &ring, n, &src, &dst).unwrap();

    // Run the sort once so the scan and the scatter see a real histogram rather than an
    // empty one, which would make their branches unrepresentative.
    let mut enc = b.device().create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        d.encode_sort(&mut pass, &plan, &binds, &so).unwrap();
    }
    b.submit([enc.finish()]);
    b.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    match kern {
        Kern::Zero => timed(40, |p| d.encode_zero_rows(p, &plan, &binds, &so).unwrap()),
        Kern::Mont => timed(40, |p| d.encode_mont(p, &mbind, n, &mo).unwrap()),
        Kern::Count => timed(40, |p| d.encode_count(p, &plan, &binds, &so).unwrap()),
        Kern::Scan => timed(40, |p| d.encode_scan(p, &plan, &binds, &so).unwrap()),
        Kern::Scatter => {
            let scan = timed(40, |p| d.encode_scan(p, &plan, &binds, &so).unwrap())?;
            let pair = timed(40, |p| {
                d.encode_scan(p, &plan, &binds, &so).unwrap();
                d.encode_scatter(p, &plan, &binds, &so).unwrap();
            })?;
            (pair > scan).then_some(pair - scan)
        }
    }
}

/// `wg` with one field replaced, so a row of the sweep varies exactly one kernel's size and
/// leaves the baseline the scatter's cell subtracts alone.
fn with_size(mut wg: wgsl::Workgroups, kern: Kern, size: u32) -> wgsl::Workgroups {
    match kern {
        Kern::Zero => wg.zero = size,
        Kern::Mont => wg.mont = size,
        Kern::Count => wg.count = size,
        Kern::Scan => wg.scan = size,
        Kern::Scatter => wg.scatter = size,
    }
    wg
}

#[test]
fn the_digit_workgroup_sizes_are_measured() {
    // 16 is in the sweep because the first run of it showed msm_count improving all the way
    // down to 32, and a sweep whose winner is at the edge of the range has not found the
    // minimum. 32 is this hardware's SIMD group width, so 16 half-fills a group.
    const SIZES: [u32; 5] = [16, 32, 64, 128, 256];
    const REPS: usize = 3;
    let n = 1u32 << 16;
    let base = wgsl::Workgroups::default();
    // One throwaway cell. The first measurement in a fresh process pays first-touch page
    // faulting on every buffer it just allocated, and it reads 2 to 3x high; U7 measured the
    // same effect on the stage 4 sweep.
    let _ = measure(Kern::Count, base, wgsl::LimbPick::default(), n);
    let mut out = format!(
        "{:<16}{}   ships\n",
        "kernel",
        SIZES
            .iter()
            .map(|w| format!("{w:>8}"))
            .collect::<Vec<_>>()
            .join("")
    );
    for (kern, name) in KERNS {
        let cells: Vec<Option<f64>> = SIZES
            .iter()
            .map(|&s| {
                let mut v: Vec<f64> = (0..REPS)
                    .filter_map(|_| {
                        measure(kern, with_size(base, kern, s), wgsl::LimbPick::default(), n)
                    })
                    .collect();
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied()
            })
            .collect();
        let ships = with_size(base, kern, 0);
        let shipped = match kern {
            Kern::Zero => base.zero,
            Kern::Mont => base.mont,
            Kern::Count => base.count,
            Kern::Scan => base.scan,
            Kern::Scatter => base.scatter,
        };
        let _ = ships;
        assert!(
            SIZES.contains(&shipped),
            "{name} ships at {shipped}, which is not one of the sizes swept, so the table \
             below is about a neighbour of it"
        );
        out.push_str(&format!(
            "{name:<16}{}{shipped:>8}\n",
            cells
                .iter()
                .map(|v| match v {
                    Some(x) => format!("{x:8.1}"),
                    None => "       ?".to_string(),
                })
                .collect::<Vec<_>>()
                .join("")
        ));
    }
    println!(
        "{n} general scalars at c = 13, microseconds per dispatch group, one kernel's size \
         varied at a time.\nmsm_scatter's cell is (scan + scatter) minus scan measured in the \
         same setup; see `measure`.\n{out}"
    );
    #[cfg(debug_assertions)]
    println!(
        "NOTE: debug build. These numbers are dominated by wgpu's host-side encode cost and \
         the ranking is not the kernel's; re-run with --release."
    );
}

#[test]
fn the_limb_pick_strategy_is_measured() {
    // Timing. Takes the cross-process lock, because cargo runs the test binaries in
    // parallel and a sibling binary saturating the GPU makes every number here fiction.
    let _gpu_timing = gpulock::exclusive_gpu();
    let n = 1u32 << 16;
    let wg = wgsl::Workgroups::default();
    let mut out = String::new();
    let _ = measure(Kern::Count, wg, wgsl::LimbPick::default(), n);
    for (kern, name) in [(Kern::Count, "msm_count"), (Kern::Scatter, "msm_scatter")] {
        let med = |pick| {
            let mut v: Vec<f64> = (0..3).filter_map(|_| measure(kern, wg, pick, n)).collect();
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied()
        };
        let idx = med(wgsl::LimbPick::Index);
        let sel = med(wgsl::LimbPick::Select);
        let ratio = match (idx, sel) {
            (Some(i), Some(s)) => format!("{:.3}", s / i),
            _ => "?".to_string(),
        };
        out.push_str(&format!(
            "{name:<14}{:>9.1}{:>9.1}   Select/Index {ratio}\n",
            idx.unwrap_or(f64::NAN),
            sel.unwrap_or(f64::NAN)
        ));
    }
    println!(
        "{n} general scalars at c = 13, microseconds.\n{:<14}{:>9}{:>9}\n{out}",
        "kernel", "Index", "Select"
    );
    #[cfg(debug_assertions)]
    println!("NOTE: debug build; re-run with --release.");
}

#[test]
fn the_digit_modules_are_split_for_a_measured_reason() {
    const ENTRIES: [&str; 5] = [
        wgsl::ENTRY_ZERO,
        wgsl::ENTRY_MONT,
        wgsl::ENTRY_COUNT,
        wgsl::ENTRY_SCAN,
        wgsl::ENTRY_SCATTER,
    ];
    let b = floor();
    let pick = wgsl::LimbPick::default();
    let v = g16_wgpu::gen::Variant::default();

    // Layouts, one per entry point, exactly as `MsmDigits` builds them, so the only thing
    // that differs between the two shapes is how the source is split.
    let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        let bgl = b
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
        let pl = b
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        (bgl, pl)
    };
    let (_z, lz) = layout("z", &MsmDigits::zero_entries());
    let (_c, lc) = layout("c", &MsmDigits::count_entries());
    let (_s, ls) = layout("s", &MsmDigits::scan_entries());
    let (_x, lx) = layout("x", &MsmDigits::scatter_entries());
    let (_m, lm) = layout("m", &MsmDigits::mont_entries());

    let build = |label: &str, src: String, entries: &[(&str, &wgpu::PipelineLayout)]| {
        let t0 = std::time::Instant::now();
        let m = b
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
            });
        let naga_ms = t0.elapsed().as_micros() as f64 / 1e3;
        let t1 = std::time::Instant::now();
        for (e, l) in entries {
            let _ = b
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(l),
                    module: &m,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                });
        }
        (naga_ms, t1.elapsed().as_micros() as f64 / 1e3, src.len())
    };

    // **Cold against cold, which is the only comparison worth having, and getting there is
    // harder than it looks.** Metal keeps an on-disk function cache keyed on the MSL it is
    // handed, and it survives the process, so a constant chosen to be "a shape nothing has
    // compiled" is cold exactly once and warm on every run after that. naga strips WGSL
    // comments, so a nonce comment does not miss the cache either: it reports a hash lookup
    // as a compile. What does miss is a different function *name*, so every entry point below
    // is renamed with a per-run nonce.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let rename = |src: String| -> String {
        let mut out = src;
        for e in ENTRIES {
            out = out.replace(&format!("fn {e}("), &format!("fn {e}_{nonce}("));
        }
        out
    };
    let cold_name = |e: &str| format!("{e}_{nonce}");
    let wg = wgsl::Workgroups::default();

    let (dn, dp, dl) = build(
        "digits",
        rename(wgsl::digits_module_at(wg, pick)),
        &[
            (&cold_name(wgsl::ENTRY_ZERO), &lz),
            (&cold_name(wgsl::ENTRY_COUNT), &lc),
            (&cold_name(wgsl::ENTRY_SCAN), &ls),
            (&cold_name(wgsl::ENTRY_SCATTER), &lx),
        ],
    );
    let (mn, mp, ml) = build(
        "mont",
        rename(wgsl::mont_module_at(v, wg.mont)),
        &[(&cold_name(wgsl::ENTRY_MONT), &lm)],
    );
    let (fn_, fp, fl) = build(
        "fused",
        rename(wgsl::fused_module_at(v, wg, pick))
            .replace(&format!("_{nonce}"), &format!("_{nonce}f")),
        &[
            (&format!("{}_{nonce}f", wgsl::ENTRY_ZERO), &lz),
            (&format!("{}_{nonce}f", wgsl::ENTRY_MONT), &lm),
            (&format!("{}_{nonce}f", wgsl::ENTRY_COUNT), &lc),
            (&format!("{}_{nonce}f", wgsl::ENTRY_SCAN), &ls),
            (&format!("{}_{nonce}f", wgsl::ENTRY_SCATTER), &lx),
        ],
    );
    // Warm, for the ratio: the shipped names, which this process and every previous one has
    // already compiled. Everything after the first proof on a machine pays this column.
    let (wn, wp, _) = build(
        "digits warm",
        wgsl::digits_module_at(wg, pick),
        &[
            (wgsl::ENTRY_ZERO, &lz),
            (wgsl::ENTRY_COUNT, &lc),
            (wgsl::ENTRY_SCAN, &ls),
            (wgsl::ENTRY_SCATTER, &lx),
        ],
    );
    let (wn2, wp2, _) = build(
        "mont warm",
        wgsl::mont_module_at(v, wg.mont),
        &[(wgsl::ENTRY_MONT, &lm)],
    );
    assert!(b.take_error().is_none(), "a module failed to build");

    let row = |name: &str, kib: f64, naga: f64, pipe: f64| {
        format!(
            "{name:<32}{kib:>7.1}{naga:>10.1}{pipe:>13.1}{:>10.1}\n",
            naga + pipe
        )
    };
    println!(
        "{:<32}{:>7}{:>10}{:>13}{:>10}\n{}{}{}{}{}",
        "shape",
        "KiB",
        "naga ms",
        "pipelines ms",
        "total ms",
        row("cold: digits, no prelude", dl as f64 / 1024.0, dn, dp),
        row("cold: fr_mont_to_std, prelude", ml as f64 / 1024.0, mn, mp),
        row(
            "cold: the two as shipped",
            (dl + ml) as f64 / 1024.0,
            dn + mn,
            dp + mp
        ),
        row("cold: all five in one module", fl as f64 / 1024.0, fn_, fp),
        row(
            "warm: the two as shipped",
            (dl + ml) as f64 / 1024.0,
            wn + wn2,
            wp + wp2
        ),
    );
    // The default is asserted to be the faster of the two rather than pinned to today's
    // answer, exactly as U7 does for the stage 4 fusion, so a reversal under Tint at U14 fails
    // loudly instead of shipping quietly.
    let split_total = dn + mn + dp + mp;
    let fused_total = fn_ + fp;
    let faster = if fused_total <= split_total {
        g16_wgpu::msm::ModuleShape::Fused
    } else {
        g16_wgpu::msm::ModuleShape::Split
    };
    assert_eq!(
        g16_wgpu::msm::ModuleShape::default(),
        faster,
        "the shipped module shape is not the faster one: Split {split_total:.1} ms cold, \
         Fused {fused_total:.1} ms cold"
    );
    println!(
        "cold ratio Split/Fused {:.3}; the default is {:?}",
        split_total / fused_total,
        g16_wgpu::msm::ModuleShape::default()
    );
    // No bar on the absolute numbers; they are the adapter's compiler, not ours. What is
    // pinned is the order of magnitude, because design §9 risk 2 puts the kill line at 10 s of
    // MSM pipeline creation in Chrome and this is the first unit that adds MSM pipelines at
    // all.
    assert!(
        split_total < 5_000.0 && fused_total < 5_000.0,
        "cold pipeline creation is over 5 s, halfway to design §9 risk 2's kill line"
    );
}
