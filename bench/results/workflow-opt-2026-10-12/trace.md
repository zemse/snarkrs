# Measured costs and floors (Mac CPU)

The operation counts and floors are [the previous round's](../workflow-opt-2026-10-11/trace.md);
this file has what was measured again and the MSM floor re-derived.

## Intrinsics, re-measured

| primitive | value | source |
|---|---|---|
| raw.rs Fq mul | 11.07 ns throughput, 15.95 ns latency (13.9 / 20.4 in the previous round, which ran on a loaded machine) | `raw_ops` |
| raw.rs Fq add / sub | 1.47 / 1.49 ns throughput, 2.15 / 2.21 latency | same |
| own aarch64 squaring (not committed) | 10.15 ns throughput, 16.64 latency | same |
| arkworks Fq / Fq2 inverse | 3.1 / 3.2 us | probe, `GlvField::inv` |

## Where the kernel's time goes

`mul_batch` single thread, us a G1 multiplication, by phase (temporary instrumentation,
1024 a batch, before the interleaving): digits 0.72, tables 1.0, ladder 20.1 (doublings
14.3, additions 5.0, scans 0.8). G2: digits 0.75, tables 4.3, ladder 33 (doublings 19.4,
additions 12.7). One batched G1 doubling measured 107 to 113 ns and one addition 92 to
96 against 5M + 2S plus adds (about 85 ns) and 5M + 1S (74 ns) at the measured multiply.

`ptau prepare` 2^18, `sample` over 60 s: 96% of the workers' time in the ladder
(`batch_dbl`/`batch_add` 74%, the ladder's scans and setup 8%, arkworks inversion 3.3%,
the one-point path of small G2 passes 4%), 4% waiting. User time 658 s over 68 s of wall:
compute bound, not idle.

H MSM of anon-aadhaar (2^21 dense scalars, c = 16), `sample`: 65% of the workers in the
batch-affine flush, 18% waiting (16 windows on 12 threads), 7% digit extraction and
insertion, 9% the running sum. Single thread, one window over 2^20 points: 119 ns a
point-window, of which about 85% the flush.

## Costs after, single thread (`glv_cost`, us a multiplication)

| | before | after | floor (9.2 ns mul) |
|---|---:|---:|---:|
| G1 batch 1024 | 21.0 to 22.6 | 20.0 to 20.7 | 12.3 |
| G1 sum of two, 1024 | (new) | 27.0 to 27.7 | |
| G2 batch 1024 | 35.1 to 37.9 | 36.2 to 37.0 | 26.8 |
| MSM fill, ns a point-window | 119 | 108 | 61 |

The spread is the machine across the day; every comparison in [lanes.md](lanes.md) is
interleaved in one session.

## MSM floor, re-derived

anon-aadhaar's five MSMs: H is 2^21 general scalars, 16 windows of c = 16, so 33.5 M
batch-affine bucket additions at 5M + 1S + 6A: 61 ns at the 9.2 ns intrinsic multiply,
76 ns at the 11.1 ns measured one. Running sums, 2 x 2^15 XYZZ additions (12M + 2S, about
130 ns) per window and point chunk, 32 of them: 0.27 core-s. The four witness MSMs are 12%
of the serial work (`msm_shape`). On 9.6 P-core equivalents: (33.5 M x 61 ns + 0.27 s) /
9.6 x 1.12 = 270 ms at the intrinsic multiply, about 330 ms at the measured one. The
previous round's 254 ms holds to within the running sums it left out. Measured after:
532 ms in the prover.
