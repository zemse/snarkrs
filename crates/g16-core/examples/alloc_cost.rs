//! What the MSM's scratch allocation actually costs, in milliseconds.
//!
//! The dhat run says one warm proof of `js_8x8_d32` allocates 103 MiB in ~2,460 blocks,
//! most of it bucket arrays. "103 MiB" is not a decision: reusing the scratch is only
//! worth building if those megabytes cost enough time to notice against a 350 ms proof.
//!
//! So this reproduces the allocation pattern exactly and times it alone:
//!
//!   * `vec![Projective::zero(); 2^(c-1)]`, once per window per MSM. Note this cannot be
//!     a `calloc`: arkworks' projective zero is `(0, 1, 0)` and `1` in Montgomery form is
//!     `R mod p`, so the buffer has a nonzero byte pattern and every element is written.
//!     A reuse scheme still has to re-zero the buckets between windows, so the honest
//!     saving is the allocator and the page faults, not the fill. Both are timed.
//!   * the prescan's `idx`/`bigints` push-and-merge, which allocates the compacted scalar
//!     set twice: once per chunk by repeated growth, once more into the merged output.
//!
//! `cargo run --release -p g16-core --example alloc_cost`

use std::time::Instant;

use g16_field::{Fr, G1Projective, G2Projective, PrimeField, UniformRand, Zero};
use g16_msm::window_size;

const RECODE_BITS: usize = 255;

fn best(reps: usize, f: &mut dyn FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // (label, general scalars, domain) for the two circuits the profile covers.
    for (name, generals, domain) in [("js_8x8_d32", 69372usize, 131072usize)] {
        let c = window_size(generals);
        let ch = window_size(domain);
        let n_buckets = 1usize << (c - 1);
        let h_buckets = 1usize << (ch - 1);
        let windows = RECODE_BITS.div_ceil(c as usize);
        let h_windows = RECODE_BITS.div_ceil(ch as usize);

        println!("=== {name} ===");
        println!(
            "witness MSMs: c={c} buckets={n_buckets} windows={windows}   \
             H MSM: c={ch} buckets={h_buckets} windows={h_windows}"
        );

        // One proof's worth: three G1 witness MSMs (A, B1, L) + one G2 (B2) + the H MSM.
        // Allocated and immediately dropped, one at a time, exactly as `window_chunk` does.
        let g1_bytes = 3 * windows * n_buckets * size_of::<G1Projective>()
            + h_windows * h_buckets * size_of::<G1Projective>();
        let g2_bytes = windows * n_buckets * size_of::<G2Projective>();

        let ms = best(reps, &mut || {
            for _ in 0..3 {
                for _ in 0..windows {
                    let b = vec![G1Projective::zero(); n_buckets];
                    std::hint::black_box(&b);
                }
            }
            for _ in 0..h_windows {
                let b = vec![G1Projective::zero(); h_buckets];
                std::hint::black_box(&b);
            }
            for _ in 0..windows {
                let b = vec![G2Projective::zero(); n_buckets];
                std::hint::black_box(&b);
            }
        });
        println!(
            "bucket arrays, allocate + fill + free:  {ms:8.3} ms  \
             ({:.1} MiB G1 + {:.1} MiB G2)",
            g1_bytes as f64 / (1024.0 * 1024.0),
            g2_bytes as f64 / (1024.0 * 1024.0),
        );

        // The same buffers reused: one allocation, then re-zeroed per window. This is what
        // the "reuse MSM scratch" proposal buys, so the difference between the two lines
        // is the entire prize. Two re-zero implementations, because `fill` specialises and
        // an `iter_mut` loop does not, and quoting only the slower one would understate
        // the proposal.
        let mut b1 = vec![G1Projective::zero(); n_buckets];
        let mut bh = vec![G1Projective::zero(); h_buckets];
        let mut b2 = vec![G2Projective::zero(); n_buckets];
        let ms_reuse = best(reps, &mut || {
            for _ in 0..3 {
                for _ in 0..windows {
                    b1.iter_mut().for_each(|x| *x = G1Projective::zero());
                    std::hint::black_box(&b1);
                }
            }
            for _ in 0..h_windows {
                bh.iter_mut().for_each(|x| *x = G1Projective::zero());
                std::hint::black_box(&bh);
            }
            for _ in 0..windows {
                b2.iter_mut().for_each(|x| *x = G2Projective::zero());
                std::hint::black_box(&b2);
            }
        });
        let ms_fill = best(reps, &mut || {
            for _ in 0..3 {
                for _ in 0..windows {
                    b1.fill(G1Projective::zero());
                    std::hint::black_box(&b1);
                }
            }
            for _ in 0..h_windows {
                bh.fill(G1Projective::zero());
                std::hint::black_box(&bh);
            }
            for _ in 0..windows {
                b2.fill(G2Projective::zero());
                std::hint::black_box(&b2);
            }
        });
        let ms_reuse = ms_reuse.min(ms_fill);
        println!("same buffers reused, re-zeroed only:    {ms_reuse:8.3} ms  (iter_mut {:.3}, fill {:.3})", ms_reuse.max(ms_fill), ms_fill);
        println!(
            "so the whole prize for scratch reuse is {:.3} ms per proof, single threaded",
            ms - ms_reuse
        );

        // Prescan: build idx/bigints by pushing, then merge into one output, as prescan
        // does for each of the five MSMs. Timed with real Montgomery reductions in it so
        // the allocation share can be compared against the work it is mixed with.
        let mut rng = ark_std::test_rng();
        let scalars: Vec<Fr> = (0..generals).map(|_| Fr::rand(&mut rng)).collect();
        let ms_scan = best(reps, &mut || {
            let mut idx: Vec<u32> = Vec::new();
            let mut bigints: Vec<<Fr as PrimeField>::BigInt> = Vec::new();
            for (i, s) in scalars.iter().enumerate() {
                idx.push(i as u32);
                bigints.push(s.into_bigint());
            }
            let mut out_i: Vec<u32> = Vec::with_capacity(idx.len());
            let mut out_b: Vec<<Fr as PrimeField>::BigInt> = Vec::with_capacity(bigints.len());
            out_i.extend_from_slice(&idx);
            out_b.extend_from_slice(&bigints);
            std::hint::black_box((&out_i, &out_b));
        });
        // Same reductions, no growth and no merge copy: the floor a reuse scheme reaches.
        let mut idx = vec![0u32; generals];
        let mut bigints = vec![<Fr as PrimeField>::BigInt::from(0u64); generals];
        let ms_scan_reuse = best(reps, &mut || {
            for (i, s) in scalars.iter().enumerate() {
                idx[i] = i as u32;
                bigints[i] = s.into_bigint();
            }
            std::hint::black_box((&idx, &bigints));
        });
        println!(
            "prescan push + merge:  {ms_scan:8.3} ms   into preallocated: {ms_scan_reuse:8.3} ms   \
             difference {:.3} ms per MSM",
            ms_scan - ms_scan_reuse
        );
    }
}
