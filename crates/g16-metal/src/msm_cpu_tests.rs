//! Algebraic checks that never create a Metal device. Integer coefficients make each
//! bucket's weight observable independently of the curve formulas in the shader.

use super::*;
use g16_field::{CurveGroup, Fq, PrimeGroup};

fn suffix(values: &mut [i64], width: usize) {
    let mut d = 1;
    while d < width {
        let before = values.to_vec();
        for lane in 0..values.len() {
            if lane + d < values.len() {
                values[lane] += before[lane + d];
            }
        }
        d *= 2;
    }
}

/// Simulate the shader's two scans, including inactive segments and partial SIMD
/// groups. The oracle below is a direct weighted sum with no scans or segmentation.
fn scan_pairs(buckets: &[i64], groups: usize, threads: usize, width: usize) -> Vec<(i64, i64)> {
    let chunk = buckets.len().div_ceil(groups);
    let seg_len = chunk.div_ceil(threads);
    let mut pairs = Vec::new();
    for g in 0..groups {
        let mut runs = vec![0; threads];
        let mut totals = vec![0; threads];
        for t in 0..threads {
            let lo = g * chunk + t * seg_len;
            let hi = (lo + seg_len).min((g + 1) * chunk).min(buckets.len());
            for j in (lo..hi).rev() {
                runs[t] += buckets[j];
                totals[t] += runs[t];
            }
        }
        for (q, p) in runs.chunks(width).zip(totals.chunks(width)) {
            let mut suff = q.to_vec();
            suffix(&mut suff, width);
            let mut v = p.to_vec();
            for lane in 1..v.len() {
                v[lane] += seg_len as i64 * suff[lane];
            }
            suffix(&mut v, width);
            pairs.push((v[0], suff[0]));
        }
    }
    pairs
}

fn packed_coefficient(k: i64) -> PackedXyzzG1 {
    if k == 0 {
        return PackedXyzzG1::default();
    }
    let p = (G1Projective::generator() * coefficient(k)).into_affine();
    PackedXyzzG1 {
        x: PackedFq::from_fq(&p.x),
        y: PackedFq::from_fq(&p.y),
        zz: PackedFq::from_fq(&Fq::one()),
        zzz: PackedFq::from_fq(&Fq::one()),
    }
}

fn coefficient(k: i64) -> Fr {
    let v = Fr::from(k.unsigned_abs());
    if k < 0 {
        -v
    } else {
        v
    }
}

#[test]
fn scan_and_host_fold_match_each_bucket_weight() {
    // One impulse per bucket proves the linear weighting, rather than checking only
    // one dense vector where compensating weight errors could cancel.
    for n in [2usize, 3, 8, 31, 32, 33, 65] {
        for groups in [1usize, 3, 8].into_iter().filter(|g| *g <= n) {
            for threads in [1, 7, 16, 31, 32, 48, 64] {
                for width in [8, 16, 32, 64] {
                    let layout = ReduceLayout::new(n, groups, threads, width);
                    for j in 0..n {
                        let mut buckets = vec![0; n];
                        buckets[j] = 1;
                        let pairs = scan_pairs(&buckets, groups, threads, width);
                        assert_eq!(pairs.len(), layout.slots);
                        let got: i64 = pairs
                            .iter()
                            .enumerate()
                            .map(|(s, (p, q))| {
                                let base = (s / layout.simdgroups) * layout.chunk
                                    + (s % layout.simdgroups) * layout.stride;
                                p + base as i64 * q
                            })
                            .sum();
                        assert_eq!(
                            got,
                            (j + 1) as i64,
                            "n={n} groups={groups} threads={threads} width={width} j={j}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn production_host_fold_handles_cancellation_and_clipped_groups() {
    for (n, groups, threads, width) in [
        (2, 1, 64, 32),
        (2, 2, 7, 16),
        (65, 3, 48, 32),
        (256, 2, 64, 16),
        (4096, 8, 64, 32),
        (16384, 8, 64, 64),
        (32768, 3, 31, 16),
    ] {
        let layout = ReduceLayout::new(n, groups, threads, width);
        for zero in [false, true] {
            let buckets: Vec<i64> = (0..n)
                .map(|j| {
                    if zero {
                        0
                    } else {
                        ((j * 17 + 3) % 23) as i64 - 11
                    }
                })
                .collect();
            let pairs = scan_pairs(&buckets, groups, threads, width);
            let packed: Vec<_> = pairs
                .iter()
                .flat_map(|&(p, q)| [packed_coefficient(p), packed_coefficient(q)])
                .collect();
            let want: i64 = buckets
                .iter()
                .enumerate()
                .map(|(j, b)| (j + 1) as i64 * b)
                .sum();
            assert_eq!(
                layout.fold(&packed).into_affine(),
                (G1Projective::generator() * coefficient(want)).into_affine()
            );
        }
    }
}

#[test]
fn shader_uses_the_pipeline_simd_width() {
    let shader = MSM_MSL
        .split("inline void msm_reduce_scan_impl")
        .nth(1)
        .unwrap()
        .split("kernel void msm_reduce_g2")
        .next()
        .unwrap();
    assert!(shader.contains("[[threads_per_simdgroup]]"));
    assert!(shader.contains("min(width, tcount - sg * width)"));
    assert!(!shader.contains("32u"));
    // The old allocation reserved four points here, but four SIMD groups write eight.
    assert_eq!(2 * ReduceLayout::new(4096, 8, 64, 16).simdgroups, 8);
}

#[test]
fn wide_merge_tree_preserves_every_spill_at_threshold_boundaries() {
    for threads in [1, 7, 16, 31, 32, 48, 64] {
        for span in [0usize, 1, 3, 4, 5, 63, 64, 65, 512] {
            for start in [0, 1, 63] {
                let values: Vec<i64> = (0..span).map(|i| i as i64 - 2).collect();
                let mut shared = vec![0; threads];
                for (tid, acc) in shared.iter_mut().enumerate() {
                    for k in (start + tid..start + span).step_by(threads) {
                        *acc += values[k - start];
                    }
                }
                let mut stride = 1;
                while stride < threads {
                    for tid in (0..threads).step_by(2 * stride) {
                        if tid + stride < threads {
                            shared[tid] += shared[tid + stride];
                        }
                    }
                    stride *= 2;
                }
                assert_eq!(shared[0], values.iter().sum::<i64>());
            }
        }
    }
}

#[test]
fn ordinary_recoding_preserves_unclassified_ones_and_top_carries() {
    for c in 2..=16usize {
        // Two full windows followed by one live carry bit, plus the one-window case.
        for bits in [1, c, 2 * c + 1] {
            let windows = bits.div_ceil(c);
            let buckets = 1 << (c - 1);
            for scalar in [0u64, 1, (1u64 << (bits - 1)) - 1] {
                let mut reconstructed = 0i128;
                let mut nonzero = 0;
                for w in 0..windows {
                    let off = w * c;
                    let raw = (scalar >> off) & ((1 << c) - 1);
                    let carry = if off == 0 {
                        0
                    } else {
                        (scalar >> (off - 1)) & 1
                    };
                    let digit = raw as i64 + carry as i64 - (((raw >> (c - 1)) << c) as i64);
                    assert!(digit.unsigned_abs() as usize <= buckets);
                    if digit != 0 {
                        nonzero += 1;
                    }
                    reconstructed += (digit as i128) << off;
                }
                assert_eq!(reconstructed, scalar as i128);
                if scalar == 1 {
                    assert_eq!(nonzero, 1);
                }
            }
        }
    }
    let routing = "sc_is_zero(s) || (p.separate_ones != 0u && sc_is_one(s))";
    assert_eq!(
        MSM_MSL.matches(routing).count(),
        2,
        "count and scatter must agree"
    );
}
