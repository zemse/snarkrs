# T4 full-clear characterisation, 2026-11-12 UTC

The corrected experimental path passed the complete device protocol. In this one
controlled microbenchmark, its batch-device and batch-wall medians are lower than
each of four separate references for all eight tuples. This is descriptive median
ordering, not a significance finding or an end-to-end proof speedup. The production
default is unchanged; coordinated public-gate integration and a fresh final-source
gate remain required before landing the candidate.

## Source and protocol

Candidate `cf8381e09ef780587a5f1eef5e2e037e5dd447c3`, baseline
`176a0a5b2681a0b9228e53beeb89c5e5e619c505`. Hardware: g4dn.xlarge, four vCPU,
16 GiB host RAM, Tesla T4 reporting 15360 MiB. This is calibration, not the
published eight-vCPU g4dn.2xlarge benchmark matrix.

The candidate uses the existing same-stream zero_u32 kernel for active
Variable-work bucket words, with checked word counts. Constant work retains its
typed identity clear. Same-stream ordering is a safety assumption; immutable
kernel arguments are not a cross-stream write guarantee.

The fixture correction changes only the tail of the two local Constant65 gate
vectors, ensuring observable nonidentity dummy arithmetic. Variable65, global
scalar generation, pure timing datasets, CUDA source and every existing assertion
remain unchanged from the original full-clear candidate. The native suite passed
67 tests with 29 ignored; independent locked GNU 2.31 ADX/BMI2 cross-builds matched
all seven preserved Linux Rust executables byte-for-byte.

The full frozen run completed **55/55 commands and 259 test executions**:

| Group | Passing executions |
|---|---:|
| Explicit preparation | 1 |
| Standard gate, full-clear OFF | 128 |
| Standard gate, full-clear ON | 128 |
| Explicit poisoned-scope and numerical MSM gates | 2 |

Each standard configuration includes library67, selected ignored24, legacy1,
warm1, adversarial15, pipeline11, field5 and fresh independent gnark4. These are
executions, not 259 distinct test functions.

All twenty retrieved proofs passed fresh native macOS rapidsnark verification
against original local vkeys, pinned original zkey/witness/vkey hashes and
witness-derived public signals. An independent audit repeated all twenty native
verifications with separate receipts: 20/20 valid again. The inventory comprises
18 GPU proofs and two deliberate CPU VRAM-refusal proofs. Own and Linux independent
receipts also pass. Controlled c13 CLI proofs establish correctness, not a proof
runtime comparison.

## Separate-reference measurements

Five cases ran in order: original-before, candidate-OFF-before, candidate-ON,
candidate-OFF-after, original-after. Each retained 88 rows: eight mixed/uniform,
G1/G2, 65536/262144 tuples times eleven phases, with 15 raw samples per row.
Total: 440 rows and 6600 samples. Settings were c13, 20 windows, wide2048,
reduce-groups8 and scatter split16384. Source-fixed warm-ups were retained.

The table shows ON device/wall medians in milliseconds. Each reference column
shows its own device/wall median delta percentage, 100*(ON/reference-1), rounded
from the retained full-precision analysis. Negative means ON is lower. References
and their samples are never pooled.

| Tuple | ON device / wall ms | Original before delta % | Original after delta % | OFF before delta % | OFF after delta % |
|---|---:|---:|---:|---:|---:|
| mixed G1 65536 | 5.384 / 5.713 | -4.76 / -4.57 | -5.14 / -5.35 | -4.84 / -4.94 | -5.18 / -5.13 |
| mixed G1 262144 | 12.251 / 13.582 | -2.66 / -2.12 | -1.10 / -1.05 | -1.50 / -0.79 | -0.60 / -0.23 |
| mixed G2 65536 | 21.926 / 22.774 | -3.00 / -2.54 | -2.21 / -1.93 | -2.98 / -2.63 | -3.40 / -3.02 |
| mixed G2 262144 | 45.815 / 47.697 | -1.41 / -1.28 | -2.32 / -2.20 | -1.60 / -1.64 | -1.63 / -1.54 |
| uniform G1 65536 | 5.949 / 6.274 | -2.20 / -1.95 | -2.87 / -2.74 | -4.19 / -3.88 | -3.92 / -3.63 |
| uniform G1 262144 | 17.413 / 18.805 | -1.85 / -1.62 | -1.95 / -1.62 | -1.19 / -1.25 | -3.73 / -3.46 |
| uniform G2 65536 | 21.864 / 23.425 | -4.47 / -4.08 | -4.57 / -4.15 | -3.92 / -3.65 | -3.84 / -3.51 |
| uniform G2 262144 | 67.521 / 70.153 | -0.74 / -0.59 | -1.57 / -1.43 | -1.10 / -0.97 | -1.36 / -1.11 |

Device median reductions range from 0.60% to 5.18%; wall reductions from 0.23% to
5.35%, over the 32 separate comparisons for each clock. Full ranges overlap in
28/32 device and 30/32 wall comparisons. Inclusive IQRs overlap in 8/32 comparisons
for each clock, covering both large-G1 datasets against all four references.
The smallest wall-median reduction, mixed G1 262144 versus OFF-after, is 0.229%
with overlapping IQRs and ranges. Do not present this as separation of all samples.

Clear medians are lower by 84.69% to 90.16% across the separate references:
0.047-0.049 ms for G1 and 0.092-0.094 ms for G2 with ON. Adjacent phases are not
uniformly better. Signed accumulation median changes range from -2.15% to +4.95%,
reduce from -2.91% to +5.59%, and combine from -2.69% to +5.03%. Combine includes
readback plus host work, not a pure device kernel. Inspect these with the measured
batch totals; do not sum separately summarised phase medians or attribute every
batch difference solely to clear.

All 352 unpooled comparisons retain their raw arrays, medians, inclusive quartiles,
ranges and signed deltas. This sequential, unpaired single run supplies no
confidence, theoretical-floor, automatic-winner or default-adoption claim.

## Guards, audit and retention

All 55 resource logs passed unchanged guards, with 4494 retained samples. Minimum
available host RAM was 13652608 KiB (13.020 GiB); maximum device use was
6507/15360 MiB (42.36%). These are sampled global observations, not exact transient
peaks or per-process attribution. The 4 GiB host floor, 90% device ceiling and
300-second retrieval reserve were unchanged. The 4500-second admission was a
heuristic, not a worst-case completion guarantee within the 90-minute window.

The fresh parent comparator invoked a new verifier subprocess, required zero exit
and exclusively consumed its current committed report. Independent audit checked
203 archived-file/extracted pairs and complete work/out membership, all frozen
payload pins, ordered progress, test/proof inventories and every phase statistic.
This is narrow archive correspondence, not safe-extraction certification.

Results were retrieved before shutdown. Fresh native and separate EC2 readbacks
confirmed stopped. Only this run's newly created SSH rule was revoked. All AWS stop
deadlines were retained; only the later local reminder was canceled after shutdown.

The private raw archive is retained with the token results. A convenience copy is
`target/cuda-validation-20261111/clear-gatefix-v2-results.tar.gz`, SHA256:

`db4c2f8c6e55bbb655a718b676d594e4f03503e42a201b7f8ddfb5e55db55eef`

The [first incomplete attempt](cuda-clear-incomplete-2026-11-12.md),
[second incomplete attempt](cuda-clear-retry-incomplete-2026-11-12.md), and rejected
intermediate provenance packet remain negative history. This successful run does
not retroactively promote them. Next: coordinate the public gate's explicit
opt-in tests and require fresh final-source validation, without changing the
default or claiming final benchmark readiness from this microbenchmark alone.
