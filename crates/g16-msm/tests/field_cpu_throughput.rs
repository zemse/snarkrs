//! CPU BN254 `Fr` multiply throughput, in the same two regimes `g16-metal`'s
//! `gpu_field_multiply_throughput` measures on the GPU, so the two numbers can be divided.
//!
//! Written for the Metal review: the backend's headline claim is a GPU-over-CPU field
//! multiply ratio, and the repository had a measured GPU number with no measured CPU
//! number to divide it by. Both halves of a ratio have to be measured on the same machine
//! in the same run or the ratio is decoration.
//!
//! Two regimes, because they answer different questions:
//!
//! * **ALU bound**: independent dependent-multiply chains, everything in registers. This
//!   is the arithmetic ceiling, the number that says whether the silicon is worth using.
//! * **Streaming**: `c[i] = a[i] * b[i]` over an array far larger than cache, so every
//!   multiply drags 96 bytes through the memory system. This is what an NTT butterfly or
//!   an MSM bucket update actually looks like, and it is the regime a real prover lives in.
//!
//! Run with `cargo test -p g16-msm --release --test field_cpu_throughput -- --nocapture`.
//! Release matters: a debug build of `ark-ff` reports roughly a fiftieth of this.

use ark_bn254::Fr;
use ark_ff::UniformRand;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Same generator as the Metal test so the two runs multiply comparable values. A field
/// multiply is data independent in Montgomery CIOS, but using the same distribution
/// removes the question.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl ark_std::rand::RngCore for SplitMix64 {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        SplitMix64::next_u64(self)
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), ark_std::rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// `ITERS` dependent multiplies. Returned, never discarded, so nothing here is dead code
/// the optimiser may delete; the chain is also verified against a replay below.
#[inline(never)]
fn chain(mut x: Fr, m: Fr, iters: u32) -> Fr {
    for _ in 0..iters {
        x *= m;
    }
    x
}

const ITERS: u32 = 512;

fn report(label: &str, muls: f64, secs: f64) {
    println!(
        "CPU THROUGHPUT {label:28} {:8.3} ms  {:6.3} G mul/s",
        secs * 1e3,
        muls / secs / 1e9
    );
}

/// Median of the middle reps, dropping the first two as warmup, matching the GPU test.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
fn cpu_field_multiply_throughput() {
    let mut rng = SplitMix64(0x5EED_0003);
    let threads = rayon::current_num_threads();
    println!("CPU  logical cores {}  rayon threads {threads}", num_cpus());

    // ---- ALU bound, single thread ----
    // 4096 independent chains so the out-of-order engine always has work, but each chain
    // is internally serial, exactly like the GPU kernel's per-lane chain.
    const LANES: usize = 4096;
    let seeds: Vec<Fr> = (0..LANES).map(|_| Fr::rand(&mut rng)).collect();
    let m = Fr::rand(&mut rng);
    let muls = LANES as f64 * ITERS as f64;

    let mut times = Vec::new();
    let mut last = Fr::from(0u64);
    for rep in 0..7 {
        let t0 = Instant::now();
        let mut acc = Fr::from(0u64);
        for &s in &seeds {
            acc += chain(black_box(s), black_box(m), ITERS);
        }
        let dt = t0.elapsed().as_secs_f64();
        last = black_box(acc);
        if rep > 1 {
            times.push(dt);
        }
    }
    let st_secs = median(times);
    report("alu-bound 1 thread", muls, st_secs);
    let single = muls / st_secs / 1e9;

    // The chain must actually compute the chain. One lane, replayed.
    let mut want = seeds[0];
    for _ in 0..ITERS {
        want *= m;
    }
    assert_eq!(chain(seeds[0], m, ITERS), want, "chain is wrong");
    let _ = last;

    // ---- ALU bound, all rayon threads ----
    // More lanes so every worker gets a full slice; per-lane work is unchanged.
    const MT_LANES: usize = 1 << 16;
    let mseeds: Vec<Fr> = (0..MT_LANES).map(|_| Fr::rand(&mut rng)).collect();
    let mt_muls = MT_LANES as f64 * ITERS as f64;
    let mut times = Vec::new();
    for rep in 0..7 {
        let t0 = Instant::now();
        let acc: Fr = mseeds
            .par_iter()
            .map(|&s| chain(black_box(s), black_box(m), ITERS))
            .reduce(|| Fr::from(0u64), |a, b| a + b);
        let dt = t0.elapsed().as_secs_f64();
        black_box(acc);
        if rep > 1 {
            times.push(dt);
        }
    }
    let mt_secs = median(times);
    report(&format!("alu-bound {threads} threads"), mt_muls, mt_secs);
    let multi = mt_muls / mt_secs / 1e9;
    println!(
        "CPU SCALING    alu-bound {single:.3} -> {multi:.3} G mul/s, {:.2}x on {threads} threads",
        multi / single
    );

    // ---- Streaming, all rayon threads, the regime a prover is actually in ----
    // Same sizes the GPU test sweeps, so the two tables line up row for row.
    for log_n in [12usize, 16, 18, 20, 22] {
        let n = 1usize << log_n;
        let xs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let ys: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let mut out = vec![Fr::from(0u64); n];

        let mut times = Vec::new();
        for rep in 0..15 {
            let t0 = Instant::now();
            out.par_iter_mut()
                .zip(xs.par_iter().zip(ys.par_iter()))
                .for_each(|(o, (a, b))| *o = *a * *b);
            let dt = t0.elapsed().as_secs_f64();
            if rep >= 5 {
                times.push(dt);
            }
        }
        let secs = median(times);
        // A broken kernel must not be able to post a fast time.
        assert_eq!(out[0], xs[0] * ys[0]);
        assert_eq!(out[n - 1], xs[n - 1] * ys[n - 1]);
        println!(
            "CPU THROUGHPUT streaming 2^{log_n:<2} {threads} threads {:8.4} ms  {:6.3} G mul/s  {:6.1} GB/s",
            secs * 1e3,
            n as f64 / secs / 1e9,
            n as f64 * 96.0 / secs / 1e9
        );
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}
