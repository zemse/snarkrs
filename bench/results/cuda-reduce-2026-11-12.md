# T4 reduction-geometry calibration, 2026-11-12 UTC

Keep RG8. This complete retry provides no reason to change the default reduction
geometry: RG4, RG16 and RG32 increased G2 reduction, allocating-batch device and
allocating-batch wall medians against both separate RG8 baselines in every
measured scalar-set/size tuple. No kernel, default or end-to-end speedup lands.
The [earlier incomplete batch](cuda-reduce-incomplete-2026-11-12.md) remains a
failed run, not a source of accepted optimisation measurements.

## Protocol and independent acceptance

Rust source `b88ce578cc01762b62742ad927d59044a1516b20`, g4dn.xlarge,
4 vCPU, 16 GiB host RAM, Tesla T4 reporting 15360 MiB. Locked GNU 2.31
ADX/BMI2 cross-builds were prepared locally. This is not the README's published
8-vCPU g4dn.2xlarge benchmark protocol.

- Controller exit zero; all 46 guarded commands completed. All 27 frozen payload
  hashes and retrieved worker sources match the manifest. All nine ELF files
  (seven Rust executables and two foreign verifiers) are unchanged from the
  earlier attempt.
- Standard gate: 124 executions, comprising ordinary library 63, selected ignored
  library 24, legacy 1, warm 1, adversarial 15, pipeline 11, field 5 and gnark 4.
  The fresh per-job independent gnark success marker was checked.
- Serial order: RG8-before, RG4, RG16, RG32, RG8-after. Each case has 88 distinct
  phase rows with 15 measured repetitions, following two source-fixed warmups.
  CPU answers are checked outside clocks on every repetition. No concurrent
  device experiment was run.
- All cases use c13, n=65536 and 262144, mixed/uniform scalars and G1/G2.
  Eight matching geometry headers and one matching Shapes record were checked
  per case. Other settings remain wide_span=32, scatter_rows=16384, w=20,
  wide=2048 and split=16384.
- All 15 CLI proofs passed native macOS rapidsnark against original local vkeys.
  The original six zkey/witness/vkey hashes match; public signals were derived
  from the pinned original witness prefixes and compared with original public
  files and retrieved statements. Own and Linux independent positive receipts
  were also checked. This includes five standard proofs and default/constant-work
  proofs for each of the five geometry cases.

## G2 medians

Milliseconds. Baselines stay separate, without averaging or interpolation.
`combine` is host wall time including readback and host combination, not a pure
CPU arithmetic measurement. `device` and `wall` are the normal allocating batch,
not sums of isolated preallocated phase medians.

| Scalars | n | Case | Reduce | Combine | Batch device | Batch wall |
|---|---:|---|---:|---:|---:|---:|
| mixed | 65536 | RG8-before | 5.637 | 0.646 | 22.355 | 23.216 |
| mixed | 65536 | RG4 | 7.423 | 0.468 | 24.171 | 24.845 |
| mixed | 65536 | RG16 | 8.138 | 0.977 | 24.992 | 26.121 |
| mixed | 65536 | RG32 | 14.006 | 1.658 | 30.885 | 32.824 |
| mixed | 65536 | RG8-after | 5.635 | 0.633 | 22.336 | 23.256 |
| mixed | 262144 | RG8-before | 5.924 | 0.871 | 46.811 | 48.791 |
| mixed | 262144 | RG4 | 7.585 | 0.691 | 48.530 | 50.252 |
| mixed | 262144 | RG16 | 8.557 | 1.197 | 49.291 | 51.694 |
| mixed | 262144 | RG32 | 14.675 | 1.901 | 55.523 | 58.492 |
| mixed | 262144 | RG8-after | 5.919 | 0.882 | 47.034 | 49.026 |
| uniform | 65536 | RG8-before | 5.826 | 0.583 | 22.863 | 24.493 |
| uniform | 65536 | RG4 | 7.489 | 0.392 | 24.346 | 25.804 |
| uniform | 65536 | RG16 | 8.413 | 0.911 | 25.580 | 27.536 |
| uniform | 65536 | RG32 | 14.028 | 1.603 | 31.153 | 33.801 |
| uniform | 65536 | RG8-after | 5.771 | 0.567 | 23.024 | 24.596 |
| uniform | 262144 | RG8-before | 5.968 | 0.593 | 68.453 | 71.068 |
| uniform | 262144 | RG4 | 7.825 | 0.423 | 70.333 | 72.724 |
| uniform | 262144 | RG16 | 8.636 | 0.946 | 71.275 | 74.151 |
| uniform | 262144 | RG32 | 14.961 | 1.645 | 77.559 | 81.233 |
| uniform | 262144 | RG8-after | 5.939 | 0.590 | 69.185 | 71.758 |

RG4 reduces the combine median in all four G2 tuples, but that does not compensate
for its device reduction cost. RG16 and RG32 increase both reduce and combine.
Every G2 candidate's reduce raw minimum exceeds both baselines' raw maxima.
This is descriptive separation of observed ranges, not a confidence interval.

## G1 controls and variability

G1 reduce medians, milliseconds:

| Scalars | n | RG8-before | RG4 | RG16 | RG32 | RG8-after |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 65536 | 1.270 | 1.407 | 2.195 | 4.088 | 1.228 |
| mixed | 262144 | 1.346 | 1.585 | 2.464 | 4.441 | 1.388 |
| uniform | 65536 | 1.306 | 1.496 | 2.290 | 4.305 | 1.311 |
| uniform | 262144 | 1.522 | 1.829 | 2.744 | 4.920 | 1.577 |

All G1 candidate reduction medians also exceed both baselines. RG16/RG32 reduce
raw ranges exceed both baseline ranges in all four tuples; RG4 has that separation
in two tuples and overlapping ranges in two. Do not turn all median differences
into claims of separated distributions.

Full-batch G1 results do not justify an across-the-board claim either. For mixed
n=262144, RG4 wall 13.714 ms lies between RG8-before 13.682 and RG8-after
13.719 ms. For uniform n=262144, RG4 device/wall 18.401/19.718 ms and RG16
18.530/19.986 ms must be read beside both 18.056/19.453 and 18.547/19.970 ms
baselines. The later baseline itself drifted upward. None establishes a robust
G1 improvement or rescues the G2 regression.

All eight tuples and eleven phases were inspected with raw ranges and inclusive
quartiles for each candidate against each baseline separately. Repetitions are
within one process per case, not independent experimental runs. Clock polling
covers whole stages, not individual kernels or tuples. Range/IQR summaries do not
provide a physical SIMT operation count, hardware floor or significance test.
The [source-count model](cuda-reduce-counts.md) remains a conditional arithmetic
reference, not a prediction that a lower source count must run faster.

## Preparation diagnosis, memory and retention

The retry kept all numerical checks and guards. Whole-queue admission increased
from 25 to 45 minutes, and the correctness-only CLI prove cap increased from 120
to 600 seconds. CLI preparation logging was enabled; measured phase settings did
not change. The first constant-work CLI log reports 176181.4 ms of NVRTC work
for `msm-constant` and 164.5 ms of module loading. That is a measured preparation
cost in this retry, not proof of the earlier empty-log timeout's exact cause or
why an effective cache entry was unavailable.

Forced-c13 CLI proofs are correctness/preparation diagnostics, not default
end-to-end performance measurements. No cache priming, clearing or configuration
change was made for this retry.

All stage samples satisfy the unchanged 90% device ceiling, 13824/15360 MiB,
and 4 GiB available-host floor. Observed device peak was 6507 MiB and minimum
available host RAM was 13692860 KiB. These are sampled bounds, not exact transient
maxima. The controller retrieved results before stopping; native lifecycle and
separate EC2 readbacks confirmed stopped. Only this batch's SSH ingress rule was
revoked. Independent AWS stop schedules were retained.

Private raw archive retained locally:
`target/cuda-validation-20261111/reduce-retry-results.tar.gz`, SHA256:

`4444b6696a4c02cc988a978967599f73e9097c4a03f5c1cb170aedc243aba6bb`

Next candidate: an opt-in variable-work clear dispatch using the existing
contiguous zero kernel, with a fresh same-box baseline, unchanged CPU/proof
oracles and full-batch device/wall checks. Constant-work projective identity
clearing must remain unchanged. No clear-kernel improvement is implied here.
