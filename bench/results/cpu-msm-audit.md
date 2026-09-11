# CPU MSM second-pass audit

2026-09-11, Apple M2 Max, 12 CPU cores. Started from `bc2c35d`, after rebasing
onto `main`. All production changes tried in this pass were discarded. The retained
change is six adversarial tests, each group getting its own denominator, scheduling,
and public-threshold test. Neither `compute_h` nor a GPU backend was edited.

No arithmetic defect was found for valid BN254 subgroup points. In `BatchFill::flush`,
a normal denominator is a nonzero x difference, equal points use nonzero 2y, and
cancellation substitutes ONE before entering the prefix product. The prefix stored
at index i excludes denominator i; the reverse pass recovers its inverse before
removing it from the accumulated inverse. The tests move doubling and cancellation
through every position of a 17-entry batch, then exercise entirely cancelling rounds
of lengths 0, 1, 1023, 1024, and 1025. Every settled bucket is compared with ark point
arithmetic, followed by an independently weighted sum.

The scheduler also held up. Tests assert that both the pending and retry flush
triggers fire, that a second conflict during rescheduling actually allocates the
side buckets, and that an affine cancellation does not erase a live side bucket.
They cover side cancellation, affine refill, repeated hot-bucket rounds, both
occupied and empty destinations during rescheduling, and finish with queued work.
Pending entries stay at most BATCH and retries below BATCH after each insertion in
the repeated-conflict case. The public threshold is exactly 2048 buckets: the
planner switches from c=11 at 30,720 general scalars to c=12 at 30,721. Splitting one
scalar gives equivalent MSMs on either side, in both groups. Infinity with each
scalar class is filtered before raw coordinates enter the batch path; ones also
exercise doubling and cancellation across the parallel prescan boundary.

The original tests already covered some doubling, cancellation, and a large hot
bucket. What they did not pin was exception position in the inversion vector,
explicit scheduler reachability and bounds, a live side bucket under an empty
affine bucket, or the exact public boundary for G2. The new tests address those gaps.

The fresh baseline was built with
`cargo build --release -p g16-cli --features metal`, then measured before source edits
with `g16 bench --artifacts bench/artifacts/csp --backend cpu --mode warm --reps 25`.
The artifacts were accessed through a worktree-local symlink to the existing data;
the main checkout was not modified. Every reported proof verified. Values below
are minimum / median / maximum, in milliseconds.

| Initial baseline | Whole proof | MSM stage |
|---|---:|---:|
| sha256_128 | 42.767 / 46.483 / 76.770 | 31.231 / 34.823 / 46.939 |
| sha256_256 | 80.161 / 86.236 / 100.306 | 57.419 / 61.750 / 69.835 |

The measured experiment sized each window by its remaining scalar bits, then applied
the existing batch threshold to that smaller bucket count:

```rust
let remaining = SCALAR_BITS.saturating_sub(w * c as usize);
let n_buckets = 1usize << remaining.min(c as usize - 1);
```

This was placed inside the window map in `pippenger`, leaving recoding, the chosen
c, and point chunking unchanged. At c=13 the top window needs at most 128 buckets
instead of 4096; at c=12 it needs four instead of 2048. Narrow top windows therefore
used XYZZ directly. This is an allocation bound derived from the scalar bit width,
not a scan of the witness's largest scalar.

Baseline and candidate binaries were staged at separate paths under
`/tmp/g16-cpu-msm-audit/`. The orders were A/B, B/A, A/B, with 25 warm repetitions per
CSP circuit in each invocation. No build or other CPU test from this lane overlapped
these timings. Another GPU lane shared the CPU and thermal budget throughout the
session. All 300 measured CSP proofs in the comparison verified.

| Circuit | Round | Baseline MSM | Candidate MSM |
|---|---:|---:|---:|
| sha256_128 | 1 | 29.787 / 32.136 / 37.442 | 30.442 / 32.915 / 35.622 |
| sha256_128 | 2 | 30.889 / 32.433 / 35.693 | 29.041 / 31.921 / 34.585 |
| sha256_128 | 3 | 30.228 / 32.439 / 36.963 | 31.238 / 32.667 / 36.120 |
| sha256_256 | 1 | 55.082 / 58.885 / 68.996 | 54.939 / 58.465 / 64.319 |
| sha256_256 | 2 | 55.114 / 60.106 / 65.901 | 56.259 / 59.737 / 66.313 |
| sha256_256 | 3 | 56.055 / 58.449 / 64.386 | 55.989 / 59.661 / 73.610 |

Pooled over 75 repetitions per circuit and binary:

| Circuit | Baseline whole proof | Candidate whole proof | Baseline MSM | Candidate MSM |
|---|---:|---:|---:|---:|
| sha256_128 | 40.888 / 44.034 / 52.064 | 40.346 / 43.960 / 49.811 | 29.787 / 32.385 / 37.442 | 29.041 / 32.551 / 36.120 |
| sha256_256 | 76.701 / 81.645 / 92.283 | 77.373 / 81.977 / 95.774 | 55.082 / 58.932 / 68.996 | 54.939 / 59.304 / 73.610 |

The same A/B, B/A, A/B procedure covered all 13 available ladder artifacts, with
three repetitions per invocation, nine per artifact and binary. All 234 measured
proofs verified, including tiny_mul and anon-aadhaar. These short, contended runs
are regression screens, not settled speed estimates.

| Ladder artifact | Baseline MSM | Candidate MSM |
|---|---:|---:|
| anon-aadhaar | 855.382 / 869.718 / 922.139 | 812.381 / 870.601 / 895.611 |
| js_16x16_d32 | 327.928 / 337.538 / 351.852 | 320.757 / 333.872 / 357.204 |
| js_1x1_d8 | 14.228 / 14.837 / 15.245 | 14.210 / 14.731 / 15.088 |
| js_2x2_d16 | 38.563 / 40.256 / 42.521 | 38.992 / 41.198 / 48.203 |
| js_2x2_d32 | 62.559 / 65.979 / 72.686 | 65.385 / 66.823 / 74.774 |
| js_8x8_d32 | 177.272 / 179.855 / 195.350 | 178.347 / 185.591 / 203.436 |
| keccak256 | 108.902 / 114.434 / 116.793 | 111.292 / 113.292 / 355.744 |
| railgun-01x01 | 47.228 / 48.976 / 52.588 | 47.489 / 52.301 / 59.725 |
| railgun-13x01 | 243.614 / 252.420 / 260.248 | 238.049 / 256.272 / 263.724 |
| rsa2048 | 117.555 / 122.609 / 132.808 | 116.694 / 123.464 / 135.951 |
| sha256 | 30.266 / 32.321 / 33.526 | 29.744 / 31.097 / 36.212 |
| tiny_mul | 0.173 / 0.213 / 0.240 | 0.172 / 0.215 / 0.319 |
| tornado | 91.080 / 93.206 / 97.550 | 87.342 / 92.364 / 98.980 |

Rejected: the CSP MSM medians moved 32.385 to 32.551 ms and 58.932 to 59.304 ms,
with inconsistent minima. The ladder was mixed; railgun-01x01 moved 48.976 to
52.301 ms, while anon-aadhaar was effectively unchanged. A keccak256 candidate
outlier reached 355.744 ms in MSM alone. None of this supports a robust speedup.
The experiment was reverted. In particular, an empty high bucket already costs
only zero checks during reduction, and bypassing batching also gives up the affine
additions that the original top window manages to schedule. Fewer allocated buckets
does not necessarily mean less expensive curve arithmetic.

The cache premise deserves a correction before building a sorted fill. `sysctl`
reported 128 KiB L1D on each performance core, 64 KiB on each efficiency core,
16 MiB L2 per four performance cores, and 4 MiB L2 for the efficiency cluster.
G1 affine bucket coordinates at c=13 occupy 4096 * 64 = 256 KiB, not 400 KiB.
Occupied/busy maps add 8 KiB. With the reserved 1536-entry capacities, pending
entries add 108 KiB, denominators and prefixes add 96 KiB, and operation tags add
1.5 KiB, before retries or lazy side storage. The ordinary scratch allocation is
therefore about 470 KiB per window. G2's coordinate storage is twice as large.

The H input allocations at 2^16 are about 6.75 MiB: 4.5 MiB of 72-byte ark affine
bases, 2 MiB of scalar bigints, and 0.25 MiB of indices. Four workers share those
inputs and add about 1.84 MiB of ordinary scratch. This fits a performance cluster's
16 MiB L2, although it exceeds L1. At 2^17 the corresponding total is about
15.34 MiB, leaving little headroom. The efficiency cluster's 4 MiB L2 cannot hold
even the smaller full input set. This arithmetic describes allocation footprints,
not measured miss rates; sequential access and cache replacement still matter.

A counting sort needs counts, offsets and cursors, plus at least 0.5 MiB per
2^16-point window to write and read compact four-byte sorted entries. Recomputing
digits during scatter adds another 2 MiB bigint traversal per window; materialising
digits trades that traversal for another buffer. Copying coordinates instead of
indices would also write and read a much larger buffer. The histogram itself is
small enough for L1. Sorted indices make bucket state local while making base
access scattered over the 4.5 MiB input.

Finishing one sorted bucket at a time in XYZZ gives up the affine fill's 5M+1S
visit in favour of 8M+2S. For 19 full c=13 windows at 2^16, 19 * (65536 - 4096) * 4
is about 4.67 million additional products counting a square as one multiply, before the more
expensive XYZZ reduction and sorting overhead. Using the supplied 0.60 G products/s
as a proxy, that alone costs about 7.8 ms. I did not build that version.
Preserving affine batching would instead require rounds over distinct bucket
cursors, with an explicit plan for ragged tails. That is still plausible, but the
cache benefit must pay for the sort, scattered base loads, and scheduling. It is
not established by counting field products. I did not repeat the earlier c sweep,
compact-entry trial, GLV proposal, or five-MSM nesting experiments.

The claimed product floor also needs qualified units. `flush` spends one product
building a prefix, two unwinding it, and two applying an ordinary affine addition,
plus one square. The amortised inversion work is already inside 5M+1S; adding
another 3M double-counts it. Nineteen full windows, including mixed bucket reduction
and full running-sum additions, are roughly
19 * ((65536 - 4096) * 6 + 4096 * 24) = 8.87 million products, before the top window,
actual inversions and other overhead. The supplied throughput was measured for Fr,
whereas this loop uses raw Fq and has prefix dependencies, field additions, memory
traffic and uneven core speeds. Treat 18 ms as a rough modelling estimate, not a
measured lower bound proving that the residual time is all cache or digit scanning.

The correctness results and operation/byte counts are solid. The timing deltas are
contended and do not establish a speedup or a precise regression. Next I would
measure raw Fq throughput and per-window completion times with the actual pool,
then prototype an affine scheduler over blocks of buckets on a standalone H benchmark only
if cache misses and the efficiency-core tail justify it. It should be tested on
both CSP H and the dense G2 witness queries from the ladder. The 1024-entry batch
size has not been shown optimal for those different cache footprints.

Raw measurements are retained locally in the ignored
`bench/results/cpu-msm-audit.csv` (584 verified rows). The individual run logs,
staged binaries, runner and discarded patch are under `/tmp/g16-cpu-msm-audit/`.

Validation completed on the retained implementation:

- `cargo test --release -p g16-msm`: 24 passed, one timing test ignored.
- `cargo test --release -p g16-core`: 22 reported passes and 10 ignored probes.
  The 19 unit tests exercised the available artifact ladder. The original MSM and
  H fixtures only covered tiny_mul. The FFT oracle initially had no local vector
  directory; after linking the existing vectors into this worktree, its compiled
  integration test passed against ffjavascript at 2^13, 2^14, and 2^16.
- The existing `bench/scripts/oracle/gen-msm-expected.mjs` was run using the installed
  snarkjs dependencies for sha256_128, sha256_256, and js_16x16_d32. Keys and witnesses
  were linked into temporary fixture directories, and the existing
  `msms_match_snarkjs` integration test was run with this worktree's artifact link
  temporarily pointing there. All 15 affine MSM outputs matched. The original
  artifact link was restored immediately. This checks the dense G2 query on the
  synthetic ladder as well as H on both CSP circuits. The generated fixtures remain
  in `/tmp/g16-cpu-msm-audit/oracle-artifacts/`.
- `cargo test --release -p g16-metal` built but failed: 31 reported passes,
  11 failures, five ignored. Every failure reported no Metal device. Of the reported
  passes, 18 are host checks and 13 ceremony/FFT tests return early without a
  device. Cargo stopped after the failing unit target, so the device integration
  suites did not execute. No GPU arithmetic or cross-backend equality was validated
  on a device in this lane.
- `cargo test --release -p g16-metal msl_declares -- --nocapture` passed all five
  constant guards, including Fq, Fr, threadgroup, recoding, and GLV declarations.
  Packing/stride and non-canonical-scalar host checks also passed in the full run.
- The final CLI was rebuilt with `cargo build --release -p g16-cli --features metal`.
  Proofs were generated with `--backend cpu` for both CSP circuits, then verified
  with `snarkjs groth16 verify`. Both printed `OK!`. Metal proof generation and
  verification were unavailable without a device.

The test commit was rebased onto the intervening GPU commits on main before the
Metal checks and final CLI build. Those commits changed only the Metal backend.
The branch is rebased again after the report commit for a fast-forward merge.
