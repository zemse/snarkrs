# T4 intrinsic calibration and c13 phase observations, 2026-11-11 UTC

Validated measurements, not an optimisation win or a refreshed full-prover table.
Rust source `4956315`, g4dn.xlarge, 4 vCPU, 16 GiB host RAM, Tesla T4 reporting
15360 MiB and 40 SMs. Driver 595.112.07, CUDA 13.2. All binaries were locally
cross-built with locked dependencies, GNU 2.31 and ADX/BMI2.

The README's published table is g4dn.2xlarge with 8 vCPUs. This smaller calibration
host is labeled separately; its CPU timings are not direct comparisons with that
published row.

## Correctness and provenance

- All 26 payload hashes rechecked locally against the retrieved build manifest;
  the guest checks each payload hash before running it. Retrieved worker and
  launcher bytes match those hashes.
- Standard gate passed 121 executions: warm 1, ordinary library 61, selected
  ignored library 23, legacy 1, adversarial 15, pipeline 11, field 5 and gnark 4.
  The extra ignored test is the CPU-only count probe. Fresh gnark receipt present.
- Probe smoke passed all 24 configurations. Measured groups contain 9 G2 cases
  at each of 32 and 128 steps, 3 Fq cases, 8 store cases and 4 overhead cases.
- Every arithmetic/store warm and measured repetition checks all outputs before
  accepting a sample. G2 checks include canonical coordinates, the XYZZ relation
  and equality with the independent CPU answer. Store checks include untouched
  poisoned words. No-op probes require successful driver execution, not a
  numerical oracle for nonexistent outputs.
- Raw sample counts, uniqueness, values and recomputed summary statistics pass
  parent validation. Both phase smoke and measured datasets contain all 88 rows;
  measured rows have exactly 15 samples, with CPU checks outside clocks.
- Five standard CLI proofs, including the expected CPU fallback, and the final
  CUDA proof passed independent native macOS rapidsnark against original local
  vkeys. Final public signals match the standard SHA-256 case. Linux independent
  verification and our verifier also passed.

No dependency, lockfile, remote tool installation or production arithmetic
change was needed. The extended every-variant sweep was not run.

## Measured primitives

These are medians of 20 repetitions after 3 warmups. CUDA events retain float
precision. Arithmetic buffers, launch builders, poison uploads, readback and
CPU verification are outside the measured interval.

| primitive | dispatch work | device median |
|---|---|---:|
| G2 XYZZ addition | 160 x 64 threads, 128 dependent calls/thread | 10.888096 ms |
| G2 XYZZ doubling | 160 x 64 threads, 128 dependent calls/thread | 6.806912 ms |
| G2 offset ladder, k=504 | 160 x 64 threads, 128 ladders/thread | 119.181343 ms |
| Fq multiplication | 320 x 256 threads, 4096 dependent calls/thread | 47.475040 ms |
| contiguous SM stores | 20 MiB, reused / rotated | 95.056 / 94.736 us |
| strided SM stores | 5 MiB useful over 20 MiB, reused / rotated | 717.504 / 687.888 us |
| contiguous SM stores | 40 MiB, reused / rotated | 190.128 / 188.848 us |
| strided SM stores | 10 MiB useful over 40 MiB, reused / rotated | 1746.000 / 1724.960 us |
| event pair | no kernel | 1.504 us |
| empty kernel | one launch | 3.904 us |
| queued empty kernels | 32 launches | 73.584 us |
| empty kernels with per-launch waits | 32 launches | 170.960 us |

The measured Fq saturation rate is about 7.068 billion multiplies/s. The G2
addition/doubling production-grid rates are about 120.38/192.56 million calls/s.
These are aggregate rates across the dispatch, not single-thread latency.
Single-thread 128-step chains instead measure about 78.96/49.94 us per G2 call.

G2 inputs are full runtime-loaded XYZZ records with distinct, valid, non-unit
Fq2 scales. This avoids compiler-known affine unit-Z operands. Loop counts are
runtime inputs and final outputs are observable. All threads share inputs and
offset, unlike heterogeneous production reduction lanes.

Store reuse/rotation refers to buffer addresses. Every repetition follows a
full-span host poison upload outside clocks, so these are not fully cache-warm
floors. The strided pattern writes only `zz`, 64 bytes every 256 bytes. Useful
bandwidth is not physical transaction bandwidth. Contiguous full-span stores
are a possible alternative mapping, not an accepted sparse-store speedup.

Clocks were not locked. Samples during the phase run span 1170 to 1590 MHz SM;
the Fq stage spans 1275 to 1590 MHz and stores 1575/1590 MHz. These polls cover
whole stages, not each individual kernel repetition. The measurements are not
clock-normalized guarantees or datasheet roofs.

## c13 G2 phases

Resident synthetic inputs, 65536 and 262144 points, mixed and uniform scalar
sets, default 8 reduce groups. Each cell is the median of 15 checked repetitions
after 2 warmups, in milliseconds. The isolated phases insert waits and reuse
scratch; the normal batch allocates its scratch without per-phase waits.

| set / n | accumulation | clear | reduce | ones | host combine | normal batch device | normal batch wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed / 65536 | 10.033 | 0.886 | 5.722 | 2.127 | 0.640 | 22.623 | 23.422 |
| mixed / 262144 | 31.716 | 0.872 | 6.030 | 2.388 | 0.870 | 47.860 | 49.843 |
| uniform / 65536 | 12.868 | 0.924 | 5.994 | 0.031 | 0.571 | 23.462 | 25.053 |
| uniform / 262144 | 56.975 | 0.900 | 6.076 | 0.059 | 0.601 | 70.666 | 73.283 |

The eight isolated GPU phase medians sum to 22.448, 47.860, 23.525 and 70.845 ms
in table order, close to normal batch device times. Sums of medians are not
medians of sums, and allocation/fencing differs, but there is little overlap in
this path. Host combine is wall time and is excluded from those GPU sums.

Reduce and clear are nearly flat despite fourfold input growth. Accumulation
scales and has larger exposure, but raw Fq pricing does not establish its
removable cost. The current fixed-cost scope remains reduce and clear first.

## What is not yet established

Source counts and reproduction commands are in
[cuda-fixed-cost-counts.md](cuda-fixed-cost-counts.md). A mixed addition is
8 Fq2 multiplies plus 2 squares, not the old comment's 7 multiplies. Generic XYZZ
addition/doubling correspond to 40/24 Fq multiplies. Initial infinity operations
inside offset ladders are not nonexceptional arithmetic.

Exact reduce exceptional-path counts, divergent warp work and tree costs are
still needed before calling a composite-math estimate a numerical floor. The
matching strided-store references leave only about 0.15 to 0.24 ms of gap to the
observed clear; alternative coalesced mappings might do better but are untested.
No change to defaults, no accepted 0.8 ms clear saving and no reduce speedup is
claimed here. Same-session measurements must include host folding and normal
batch wall time before accepting any candidate.

## Resources and evidence

All retained samples satisfy the unchanged 90% device and 4 GiB available-host
RAM guards. Sampled peak was 6507 MiB during the standard gate; minimum available
host RAM was 13546904 KiB. These are sampled observations, not exact transient
peaks. Live clock, power and temperature samples are retained per stage.

All 19 batch commands completed. Results were retrieved before shutdown;
controller exit was zero, native lifecycle status and independent EC2 readback
confirmed stopped. Only this batch's SSH rule was revoked. All AWS stop deadlines
remain armed. The stopped root EBS volume remains retained and billed.

Raw archive: `target/cuda-validation-20261111/intrinsic-results.tar.gz`, SHA256
`fa53e345467b70b38cc045e74bf3b4ea69bd45a57ae740ba35056b6fbe1dc4a6`.
Parent checks and full raw per-repetition data are retained in private session
state. No S3 objects or original fixtures were deleted.

Neither `node` nor `nodejs` was on PATH. A bounded regular-file and symlink search
under `/usr/bin`, `/home/ubuntu`, `/opt` and `/usr/local` through depth 12 found
neither name. This is not proof of absence everywhere. Node installation and
private S3 deletion still require explicit approval; the final SnarkJS cells
must not be silently omitted.
