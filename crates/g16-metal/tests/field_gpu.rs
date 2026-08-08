//! GPU-vs-`ark-ff` validation for the BN254 `Fr` MSL prelude.
//!
//! This is the test the rest of the Metal backend rests on. A wrong Montgomery constant
//! or an off-by-one in the CIOS carry chain does not crash and does not produce obviously
//! wrong-looking limbs; it produces a proof that fails to verify, at which point every
//! stage of the pipeline is a suspect. So the field is checked against the same arkworks
//! implementation the CPU backend uses, element by element, before anything else exists.
//!
//! Everything here compiles MSL at runtime through `newLibraryWithSource`. There is no
//! offline Metal toolchain on the development machine and none is required.

#![cfg(target_os = "macos")]

use std::ffi::c_void;
use std::time::Instant;

use g16_field::{Field, Fr, One, PrimeField, Zero};
use g16_metal::kernels::FR_MSL;
use g16_metal::layout::{as_bytes, Packed, PackedFr, FR_MODULUS, LIMBS};
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

/// Test-only kernels. Deliberately *not* in `src/shaders/`: nothing in the prover
/// dispatches these, and shipping them in the production library would mean paying their
/// compile time on every `prepare`.
const TEST_MSL: &str = r#"
kernel void t_add(device const Fr* a, device const Fr* b, device Fr* out,
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_add(a[gid], b[gid]);
}
kernel void t_sub(device const Fr* a, device const Fr* b, device Fr* out,
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_sub(a[gid], b[gid]);
}
kernel void t_mul(device const Fr* a, device const Fr* b, device Fr* out,
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_mul(a[gid], b[gid]);
}
kernel void t_sqr(device const Fr* a, device const Fr* b, device Fr* out,
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_sqr(a[gid]);
}
kernel void t_neg(device const Fr* a, device const Fr* b, device Fr* out,
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_neg(a[gid]);
}
// Round trip through standard form. Montgomery in, Montgomery out, so the host expects
// the identity map; anything else means R^2 or the reduction is wrong.
kernel void t_mont_roundtrip(device const Fr* a, device const Fr* b, device Fr* out,
                             uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_to_mont(fr_from_mont(a[gid]));
}
// Predicates, one byte of answer per lane, so is_zero/eq are exercised rather than
// assumed. Packed as: bit 0 = fr_is_zero(a), bit 1 = fr_eq(a, b), bit 2 = fr_eq(a, a).
kernel void t_predicates(device const Fr* a, device const Fr* b, device uint* out,
                         uint gid [[thread_position_in_grid]]) {
    uint r = 0u;
    r |= fr_is_zero(a[gid]) ? 1u : 0u;
    r |= fr_eq(a[gid], b[gid]) ? 2u : 0u;
    r |= fr_eq(a[gid], a[gid]) ? 4u : 0u;
    out[gid] = r;
}
// ALU-bound throughput: one dependent multiply chain per lane, so the measurement is
// latency x width and not memory bandwidth. `iters` is a runtime value specifically so
// the compiler cannot fold the chain away.
kernel void bench_chain(device const Fr* seeds, device Fr* out, constant uint& iters,
                        uint gid [[thread_position_in_grid]]) {
    Fr x = seeds[gid & 255u];
    Fr m = seeds[(gid + 1u) & 255u];
    for (uint i = 0u; i < iters; i++) {
        x = fr_mul(x, m);
    }
    out[gid] = x;
}
// Streaming throughput: one multiply per 96 bytes of traffic. This is the regime an NTT
// butterfly actually runs in, and it is the number that decides whether an NTT kernel is
// worth writing.
kernel void bench_stream(device const Fr* a, device const Fr* b, device Fr* out,
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = fr_mul(a[gid], b[gid]);
}
"#;

// ---------------------------------------------------------------------------
// Host-side scaffolding
// ---------------------------------------------------------------------------

/// SplitMix64. Deterministic so a failure is reproducible from the seed printed with it,
/// and dependency-free so validating a field does not pull in an RNG crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_fr(&mut self) -> Fr {
        let mut b = [0u8; 32];
        for c in b.chunks_mut(8) {
            c.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        Fr::from_le_bytes_mod_order(&b)
    }
}

struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    library: metal::Library,
}

impl Gpu {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let src = format!("{FR_MSL}\n{TEST_MSL}");
        let library = device
            .new_library_with_source(&src, &CompileOptions::new())
            // The compiler's diagnostics are the only thing that tells you which line of
            // MSL is wrong, so they are propagated verbatim rather than summarised.
            .unwrap_or_else(|e| panic!("MSL compile failed:\n{e}"));
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            library,
        }
    }

    fn pipeline(&self, name: &str) -> ComputePipelineState {
        let f = self
            .library
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("no kernel {name}: {e}"));
        self.device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("pipeline {name}: {e}"))
    }

    fn shared<T: Packed>(&self, data: &[T]) -> Buffer {
        let bytes = as_bytes(data);
        self.device.new_buffer_with_data(
            bytes.as_ptr() as *const c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn empty(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    /// One binary kernel, one command buffer, one wait. Returns the outputs.
    fn run_binop(
        &self,
        pso: &ComputePipelineState,
        a: &[PackedFr],
        b: &[PackedFr],
    ) -> Vec<PackedFr> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let ba = self.shared(a);
        let bb = self.shared(b);
        let out = self.empty(n * std::mem::size_of::<PackedFr>());

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&bb), 0);
        enc.set_buffer(2, Some(&out), 0);
        // Never hardcode 1024. A kernel carrying a Montgomery multiply reports a lower
        // maximum than the device advertises, because of register pressure, and Metal
        // rejects a threadgroup larger than the pipeline's own limit.
        let tg = pso.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        read_back(&out, n)
    }
}

fn read_back<T: Copy>(b: &Buffer, n: usize) -> Vec<T> {
    // SAFETY: the buffer was allocated with exactly `n * size_of::<T>()` bytes in shared
    // storage, the GPU work that wrote it has completed, and `T` here is always a packed
    // POD with no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(b.contents() as *const T, n) }.to_vec()
}

fn pack(xs: &[Fr]) -> Vec<PackedFr> {
    PackedFr::pack_slice(xs)
}

/// Compares GPU output against a host closure, element by element. Returns the number
/// that matched and panics on the first that does not, printing both sides in standard
/// form so the failure is readable rather than eight hex words.
fn expect_all(
    label: &str,
    a: &[Fr],
    b: &[Fr],
    got: &[PackedFr],
    want: impl Fn(Fr, Fr) -> Fr,
) -> usize {
    assert_eq!(got.len(), a.len());
    for i in 0..a.len() {
        let expected = want(a[i], b[i]);
        let actual = got[i].to_fr();
        assert_eq!(
            actual,
            expected,
            "{label}: mismatch at index {i}\n  a = {}\n  b = {}\n  gpu = {}\n  ark = {}",
            a[i].into_bigint(),
            b[i].into_bigint(),
            actual.into_bigint(),
            expected.into_bigint(),
        );
    }
    a.len()
}

/// The edge cases that a random sweep will essentially never generate: the boundaries of
/// the conditional subtraction, both directions of the sub borrow, and the largest
/// possible product.
fn edge_pairs() -> Vec<(Fr, Fr)> {
    let zero = Fr::zero();
    let one = Fr::one();
    let two = Fr::from(2u64);
    let m1 = -one; // r - 1, the largest element
    let m2 = m1 - one;
    let half = Fr::from(2u64).inverse().unwrap();
    vec![
        // add: exactly hits the conditional subtraction from below, at, and above.
        (m1, one), // r - 1 + 1 = 0, the wrap
        (m2, one), // r - 2 + 1 = r - 1, no wrap, one below the boundary
        (m1, two), // r - 1 + 2 = 1, one above
        (m1, m1),  // r - 2, the largest sum
        (zero, zero),
        (zero, one),
        (one, zero),
        // sub: borrow and no borrow.
        (zero, one), // 0 - 1 = r - 1
        (one, m1),   // 1 - (r-1) = 2
        (m1, m1),
        // mul: the identities and the extreme.
        (zero, zero),
        (zero, m1),
        (m1, zero),
        (one, m1),
        (m1, one),
        (m1, m1), // (r-1)^2 = 1
        (two, half),
        (half, two),
    ]
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// The one that matters. Every operation, against `ark-ff`, on the same inputs.
#[test]
fn gpu_field_matches_ark_ff() {
    let gpu = Gpu::new();
    let mut rng = SplitMix64(0x5EED_0001);

    // 10,000 random pairs is the floor the brief asks for; 16,384 is a round dispatch and
    // costs nothing extra, so the multiply gets that plus the edge cases.
    const N_RANDOM: usize = 16_384;
    let mut a: Vec<Fr> = (0..N_RANDOM).map(|_| rng.next_fr()).collect();
    let mut b: Vec<Fr> = (0..N_RANDOM).map(|_| rng.next_fr()).collect();

    // Sprinkle in structurally interesting values so the random block also exercises
    // zero and one operands rather than only generic 254-bit residues.
    for (i, (x, y)) in edge_pairs().into_iter().enumerate() {
        a[i] = x;
        b[i] = y;
    }
    // Some deliberately near-modulus operands: r - k for small k, which is where an
    // off-by-one in the conditional subtraction hides.
    for k in 1..=64u64 {
        let i = 1024 + k as usize;
        a[i] = -Fr::from(k);
        b[i] = -Fr::from(65 - k);
    }

    let pa = pack(&a);
    let pb = pack(&b);
    let n = a.len();

    let mut matched = Vec::new();

    let got = gpu.run_binop(&gpu.pipeline("t_add"), &pa, &pb);
    matched.push(("add", expect_all("add", &a, &b, &got, |x, y| x + y)));

    let got = gpu.run_binop(&gpu.pipeline("t_sub"), &pa, &pb);
    matched.push(("sub", expect_all("sub", &a, &b, &got, |x, y| x - y)));

    let got = gpu.run_binop(&gpu.pipeline("t_neg"), &pa, &pb);
    matched.push(("neg", expect_all("neg", &a, &b, &got, |x, _| -x)));

    let got = gpu.run_binop(&gpu.pipeline("t_mul"), &pa, &pb);
    matched.push(("mul", expect_all("mul", &a, &b, &got, |x, y| x * y)));

    let got = gpu.run_binop(&gpu.pipeline("t_sqr"), &pa, &pb);
    matched.push(("sqr", expect_all("sqr", &a, &b, &got, |x, _| x * x)));

    let got = gpu.run_binop(&gpu.pipeline("t_mont_roundtrip"), &pa, &pb);
    matched.push((
        "mont_roundtrip",
        expect_all("mont_roundtrip", &a, &b, &got, |x, _| x),
    ));

    // Predicates. `b` is compared against `a` here, and the first slots deliberately
    // contain equal and zero operands, so all three bits are exercised in both states.
    let pso = gpu.pipeline("t_predicates");
    let ba = gpu.shared(&pa);
    let bb = gpu.shared(&pb);
    let out = gpu.empty(n * 4);
    let cb = gpu.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&ba), 0);
    enc.set_buffer(1, Some(&bb), 0);
    enc.set_buffer(2, Some(&out), 0);
    let tg = pso.max_total_threads_per_threadgroup().min(256);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let flags: Vec<u32> = read_back(&out, n);
    let mut pred_matched = 0usize;
    let mut saw_zero = 0usize;
    let mut saw_eq = 0usize;
    for i in 0..n {
        let want = u32::from(a[i].is_zero()) | (u32::from(a[i] == b[i]) << 1) | 4; // fr_eq(a, a) must always hold
        assert_eq!(flags[i], want, "predicates: mismatch at index {i}");
        saw_zero += usize::from(a[i].is_zero());
        saw_eq += usize::from(a[i] == b[i]);
        pred_matched += 1;
    }
    // A predicate test that never sees a true case proves nothing.
    assert!(saw_zero > 0, "no zero operand reached the predicate kernel");
    assert!(saw_eq > 0, "no equal pair reached the predicate kernel");
    matched.push(("predicates", pred_matched));

    for (op, count) in &matched {
        println!("VECTORS MATCHED  {op:<16} {count}");
    }
    println!(
        "VECTORS MATCHED  {:<16} {}",
        "total",
        matched.iter().map(|(_, c)| c).sum::<usize>()
    );
}

/// Named vectors that can be checked by hand, so a reviewer does not have to trust
/// `ark-ff` either. `3R * 5R == 15R` is the same vector the scouting phase used to
/// validate its standalone CIOS microbenchmarks.
#[test]
fn gpu_known_vectors() {
    let gpu = Gpu::new();
    let mul = gpu.pipeline("t_mul");
    let add = gpu.pipeline("t_add");
    let sub = gpu.pipeline("t_sub");

    let f = |n: u64| Fr::from(n);
    let m1 = -Fr::one();

    // (input a, input b, expected) for the multiply.
    let mul_cases: Vec<(Fr, Fr, Fr)> = vec![
        (f(3), f(5), f(15)), // 3R * 5R == 15R in Montgomery form
        (f(0), f(0), f(0)),
        (f(0), m1, f(0)),    // multiply by zero
        (Fr::one(), m1, m1), // multiply by one
        (m1, Fr::one(), m1),
        (m1, m1, Fr::one()), // (r-1)^2 == 1
        (f(2), Fr::from(2u64).inverse().unwrap(), Fr::one()),
        // 2^128 squared, which straddles the limb boundaries the CIOS carry chain walks.
        (
            f(1) * Fr::from(2u64).pow([128u64]),
            Fr::from(2u64).pow([128u64]),
            Fr::from(2u64).pow([128u64]) * Fr::from(2u64).pow([128u64]),
        ),
    ];
    let (ma, mb): (Vec<Fr>, Vec<Fr>) = mul_cases.iter().map(|(x, y, _)| (*x, *y)).unzip();
    let got = gpu.run_binop(&mul, &pack(&ma), &pack(&mb));
    for (i, (_, _, want)) in mul_cases.iter().enumerate() {
        assert_eq!(got[i].to_fr(), *want, "known mul vector {i}");
    }

    // Wraparound at the modulus, both directions.
    let add_cases: Vec<(Fr, Fr, Fr)> = vec![
        (m1, Fr::one(), Fr::zero()), // r-1 + 1 wraps to 0
        (m1, f(2), Fr::one()),
        (m1, m1, m1 - Fr::one()), // r-2, the largest reachable sum
        (Fr::zero(), Fr::zero(), Fr::zero()),
    ];
    let (aa, ab): (Vec<Fr>, Vec<Fr>) = add_cases.iter().map(|(x, y, _)| (*x, *y)).unzip();
    let got = gpu.run_binop(&add, &pack(&aa), &pack(&ab));
    for (i, (_, _, want)) in add_cases.iter().enumerate() {
        assert_eq!(got[i].to_fr(), *want, "known add vector {i}");
    }

    let sub_cases: Vec<(Fr, Fr, Fr)> = vec![
        (Fr::zero(), Fr::one(), m1), // 0 - 1 borrows and yields r-1
        (Fr::zero(), Fr::zero(), Fr::zero()),
        (Fr::one(), m1, f(2)),
        (m1, m1, Fr::zero()),
    ];
    let (sa, sb): (Vec<Fr>, Vec<Fr>) = sub_cases.iter().map(|(x, y, _)| (*x, *y)).unzip();
    let got = gpu.run_binop(&sub, &pack(&sa), &pack(&sb));
    for (i, (_, _, want)) in sub_cases.iter().enumerate() {
        assert_eq!(got[i].to_fr(), *want, "known sub vector {i}");
    }

    println!(
        "KNOWN VECTORS MATCHED  mul {} add {} sub {}",
        mul_cases.len(),
        add_cases.len(),
        sub_cases.len()
    );
}

/// A representative that is not fully reduced must not be produced by any operation.
/// The CIOS single conditional subtraction is only sound because the intermediate stays
/// below 2r; if that reasoning is wrong the symptom is an output in [r, 2^256), which
/// still round-trips through `new_unchecked` and would pass an equality test against a
/// host value computed the same wrong way. Comparing the raw limbs against r is the only
/// way to catch it.
#[test]
fn gpu_outputs_are_always_fully_reduced() {
    let gpu = Gpu::new();
    let mut rng = SplitMix64(0x5EED_0002);

    const N: usize = 8192;
    let mut a: Vec<Fr> = (0..N).map(|_| rng.next_fr()).collect();
    let mut b: Vec<Fr> = (0..N).map(|_| rng.next_fr()).collect();
    // Bias hard toward the top of the range, where a missing subtraction shows up.
    for i in 0..1024 {
        a[i] = -Fr::from(i as u64 + 1);
        b[i] = -Fr::from(i as u64 + 1);
    }
    let (pa, pb) = (pack(&a), pack(&b));

    let mut checked = 0usize;
    for name in ["t_add", "t_sub", "t_mul", "t_sqr", "t_neg"] {
        let got = gpu.run_binop(&gpu.pipeline(name), &pa, &pb);
        for (i, g) in got.iter().enumerate() {
            assert!(
                less_than_modulus(&g.v),
                "{name}: output at index {i} is not reduced below r: {:08x?}",
                g.v
            );
            checked += 1;
        }
    }
    println!("REDUCTION CHECKED  {checked} outputs below r");
}

fn less_than_modulus(v: &[u32; LIMBS]) -> bool {
    for i in (0..LIMBS).rev() {
        if v[i] != FR_MODULUS[i] {
            return v[i] < FR_MODULUS[i];
        }
    }
    false // equal to r is not reduced
}

/// Throughput, because a kernel nobody timed is not a result. Prints two numbers:
/// the ALU-bound rate (dependent chain, no memory traffic) and the streaming rate
/// (one multiply per 96 bytes moved), which is the regime an NTT butterfly lives in.
#[test]
fn gpu_field_multiply_throughput() {
    let gpu = Gpu::new();
    let mut rng = SplitMix64(0x5EED_0003);

    println!(
        "DEVICE  {}  unified={}  maxThreadsPerThreadgroup={}  threadgroupMem={}",
        gpu.device.name(),
        gpu.device.has_unified_memory(),
        gpu.device.max_threads_per_threadgroup().width,
        gpu.device.max_threadgroup_memory_length(),
    );

    // ---- ALU-bound: dependent multiply chain, one per lane ----
    let chain = gpu.pipeline("bench_chain");
    println!(
        "PIPELINE bench_chain maxTotalThreadsPerThreadgroup={}  (device advertises {})",
        chain.max_total_threads_per_threadgroup(),
        gpu.device.max_threads_per_threadgroup().width
    );

    const THREADS: usize = 1 << 20;
    const ITERS: u32 = 512;
    let seeds: Vec<Fr> = (0..256).map(|_| rng.next_fr()).collect();
    let bseeds = gpu.shared(&pack(&seeds));
    let out = gpu.empty(THREADS * std::mem::size_of::<PackedFr>());

    // Swept rather than picked. The scouting phase measured 4.51 G mul/s for this exact
    // CIOS on this machine without saying at what threadgroup size, and the reference
    // implementations disagree with each other (lambdaworks prefers 256, zkonduit clamps
    // to 64), so the only honest thing is to try them and print the table.
    let cap = chain.max_total_threads_per_threadgroup();
    let mut best = (0f64, 0u64);
    for tg in [32u64, 64, 128, 256, 512, 1024] {
        if tg > cap {
            continue;
        }
        let mut times = Vec::new();
        for rep in 0..7 {
            let t0 = Instant::now();
            let cb = gpu.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&chain);
            enc.set_buffer(0, Some(&bseeds), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 4, &ITERS as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(THREADS as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if rep > 1 {
                times.push(dt);
            }
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median = times[times.len() / 2];
        let muls = THREADS as f64 * ITERS as f64;
        let rate = muls / median / 1e9;
        println!(
            "THROUGHPUT alu-bound  threadgroup {tg:<4} {:.3} ms  {rate:.2} G mul/s",
            median * 1e3
        );
        if rate > best.0 {
            best = (rate, tg);
        }
    }
    println!(
        "THROUGHPUT alu-bound  BEST {:.2} G mul/s at threadgroup {}",
        best.0, best.1
    );

    // Correctness of the chain result, so this is not timing a kernel that computes
    // nothing. One lane, replayed on the host.
    let res: Vec<PackedFr> = read_back(&out, 4);
    let mut want = seeds[0];
    let m = seeds[1];
    for _ in 0..ITERS {
        want *= m;
    }
    assert_eq!(res[0].to_fr(), want, "bench chain computed the wrong value");

    // ---- Streaming: one multiply per element, 96 bytes of traffic each ----
    let stream = gpu.pipeline("bench_stream");
    for log_n in [12usize, 16, 18, 20, 22] {
        let n = 1usize << log_n;
        let xs: Vec<Fr> = (0..n).map(|_| rng.next_fr()).collect();
        let ys: Vec<Fr> = (0..n).map(|_| rng.next_fr()).collect();
        let ba = gpu.shared(&pack(&xs));
        let bb = gpu.shared(&pack(&ys));
        let bo = gpu.empty(n * std::mem::size_of::<PackedFr>());
        let tg = stream.max_total_threads_per_threadgroup().min(256);

        // 15 reps with the first 5 dropped. A freshly allocated shared buffer is faulted
        // in on first touch, which the scouting phase measured at 3.6x the warm cost, so
        // an under-warmed run reports page faults rather than bandwidth.
        let mut times = Vec::new();
        for rep in 0..15 {
            let t0 = Instant::now();
            let cb = gpu.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&stream);
            enc.set_buffer(0, Some(&ba), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&bo), 0);
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if rep >= 5 {
                times.push(dt);
            }
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median = times[times.len() / 2];
        // Spot-check the first element so a broken kernel cannot post a fast time.
        let got: Vec<PackedFr> = read_back(&bo, 1);
        assert_eq!(got[0].to_fr(), xs[0] * ys[0]);
        println!(
            "THROUGHPUT streaming  2^{log_n:<2}  {:.4} ms  {:.3} G mul/s  {:.1} GB/s",
            median * 1e3,
            n as f64 / median / 1e9,
            n as f64 * 96.0 / median / 1e9
        );
    }

    // ---- The floor everything above sits on ----
    let mut times = Vec::new();
    for _ in 0..64 {
        let t0 = Instant::now();
        let cb = gpu.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&stream);
        enc.set_buffer(0, Some(&bseeds), 0);
        enc.set_buffer(1, Some(&bseeds), 0);
        enc.set_buffer(2, Some(&out), 0);
        enc.dispatch_threads(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "DISPATCH FLOOR  one command buffer, commit + wait, 32 threads: {:.4} ms median",
        times[times.len() / 2] * 1e3
    );
}

/// Compiling MSL at runtime is not free and must never sit inside a timed proof.
///
/// The source is salted with a unique comment so this measures a real compile. macOS
/// keeps a persistent shader cache keyed on the source text, and a second compile of
/// byte-identical MSL returns in well under a millisecond, which would make this test
/// report a reassuring number that no first run will ever see.
#[test]
fn msl_runtime_compile_cost() {
    let device = Device::system_default().expect("no Metal device");
    let salt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let src = format!("// cache salt {salt}\n{FR_MSL}\n{TEST_MSL}");
    let t0 = Instant::now();
    let library = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("MSL compile failed");
    let compile = t0.elapsed();

    let names = [
        "t_add",
        "t_sub",
        "t_mul",
        "t_sqr",
        "t_neg",
        "t_mont_roundtrip",
        "t_predicates",
        "bench_chain",
        "bench_stream",
    ];
    let t0 = Instant::now();
    for n in names {
        let f = library.get_function(n, None).unwrap();
        let _ = device.new_compute_pipeline_state_with_function(&f).unwrap();
    }
    let pipelines = t0.elapsed();

    println!(
        "STARTUP  newLibraryWithSource {:.1} ms  +  {} pipeline states {:.1} ms",
        compile.as_secs_f64() * 1e3,
        names.len(),
        pipelines.as_secs_f64() * 1e3
    );
}
