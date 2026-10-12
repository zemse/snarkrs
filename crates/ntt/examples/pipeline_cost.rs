//! The prover's NTT stage on its own: three `intt_coset_ntt` pipelines at once, as
//! `compute_h` runs them, then one alone, then one on a single thread, in cache (2^11) and
//! at the full size. Every timed output is checked at two coset points against the
//! barycentric evaluation of its input, which shares no code with the transform.
//!
//! `cargo run --release -p snarkrs-ntt --example pipeline_cost -- [LOG_N]` (default 21)
//!
//! Prints `what ms` lines, the median of `REPS` (default 7) runs.

use std::time::Instant;

use ark_ff::{batch_inversion, Field};
use snarkrs_field::{Domain, Fr};
use snarkrs_ntt::CpuNtt;

fn sample(n: usize, seed: u64) -> Vec<Fr> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Fr::from(s)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// `w^j / (z - w^j) * (z^n - 1) / n` for every `j`: the barycentric weights at `z`, so
/// `P(z)` for the polynomial through `(w^j, v_j)` is their dot product with `v`.
fn weights(d: &Domain, z: Fr) -> Vec<Fr> {
    let n = d.size;
    let mut pw = Vec::with_capacity(n);
    let mut acc = Fr::ONE;
    for _ in 0..n {
        pw.push(acc);
        acc *= d.group_gen;
    }
    let mut den: Vec<Fr> = pw.iter().map(|w| z - w).collect();
    batch_inversion(&mut den);
    let c = (z.pow([n as u64]) - Fr::ONE) * d.size_inv;
    pw.iter().zip(&den).map(|(w, i)| *w * i * c).collect()
}

/// Output `k` of the pipeline is `P(shift w^k)`.
struct Check {
    k: [usize; 2],
    w: [Vec<Fr>; 2],
}

impl Check {
    fn new(d: &Domain, shift: Fr, k: [usize; 2]) -> Self {
        let z = |k: usize| shift * d.group_gen.pow([k as u64]);
        Check {
            k,
            w: [weights(d, z(k[0])), weights(d, z(k[1]))],
        }
    }

    fn assert(&self, input: &[Fr], out: &[Fr], what: &str) {
        for (k, w) in self.k.iter().zip(&self.w) {
            let want: Fr = input.iter().zip(w).map(|(v, w)| *v * w).sum();
            assert_eq!(out[*k], want, "{what}: output {k}");
        }
    }
}

fn run(log_n: u32, reps: usize) {
    let n = 1usize << log_n;
    let d = Domain::new(n).unwrap();
    let shift = Domain::new(2 * n).unwrap().group_gen;
    let ins: Vec<Vec<Fr>> = (0..3).map(|i| sample(n, 0x5eed + i)).collect();
    let check = Check::new(&d, shift, [1, n - 3]);
    let ntt = CpuNtt::new();
    ntt.prepare_intt_coset_ntt(&d, shift);

    let mut bufs = ins.clone();
    let mut t = Vec::new();
    for _ in 0..reps {
        for (b, i) in bufs.iter_mut().zip(&ins) {
            b.copy_from_slice(i);
        }
        let [a, b, c] = &mut bufs[..] else {
            unreachable!()
        };
        let t0 = Instant::now();
        rayon::join(
            || ntt.intt_coset_ntt(&d, a, shift),
            || {
                rayon::join(
                    || ntt.intt_coset_ntt(&d, b, shift),
                    || ntt.intt_coset_ntt(&d, c, shift),
                )
            },
        );
        t.push(t0.elapsed().as_secs_f64() * 1e3);
        for (i, o) in ins.iter().zip(&bufs) {
            check.assert(i, o, "three");
        }
    }
    println!("2^{log_n} three {:.2}", median(t));

    let mut t = Vec::new();
    for _ in 0..reps {
        bufs[0].copy_from_slice(&ins[0]);
        let t0 = Instant::now();
        ntt.intt_coset_ntt(&d, &mut bufs[0], shift);
        t.push(t0.elapsed().as_secs_f64() * 1e3);
        check.assert(&ins[0], &bufs[0], "one");
    }
    println!("2^{log_n} one {:.2}", median(t));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let single = pool.install(CpuNtt::new);
    pool.install(|| single.prepare_intt_coset_ntt(&d, shift));
    let mut t = Vec::new();
    for _ in 0..reps.min(3) {
        bufs[0].copy_from_slice(&ins[0]);
        let t0 = Instant::now();
        pool.install(|| single.intt_coset_ntt(&d, &mut bufs[0], shift));
        t.push(t0.elapsed().as_secs_f64() * 1e3);
        check.assert(&ins[0], &bufs[0], "one thread");
    }
    let ms = median(t);
    // 2 log n passes of n / 2 butterflies, plus the n coset multiplies.
    let bf = (log_n as f64) * n as f64;
    println!(
        "2^{log_n} one_thread {ms:.2} ({:.2} ns a butterfly)",
        ms * 1e6 / bf
    );
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let log_n: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("log_n"))
        .unwrap_or(21);
    run(11, reps.max(25));
    run(log_n, reps);
}
