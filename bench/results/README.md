# Results

`comparison.csv` is the raw per-rep data. Every row is a proof that was verified before its
timing was recorded, through three independent oracles: our own pairing check, snarkjs
`groth16 verify`, and rapidsnark's C++ verifier.

Apple M2 Max (8P + 4E, 96 GB), 10 reps, medians, load average ~3.5.

## Warm: setup paid once, steady state

| variant | constraints | rapidsnark | our CPU | our Metal | Metal vs rapidsnark | Metal vs our CPU |
|---|---:|---:|---:|---:|---:|---:|
| tiny_mul | 2 | 1.0 | 1.0 | 3.9 | 0.26x | 0.25x |
| js_1x1_d8 | 3,359 | 25.0 | 23.7 | **18.9** | 1.32x | 1.25x |
| js_2x2_d16 | 10,153 | 56.0 | 60.1 | **22.1** | 2.54x | 2.72x |
| js_2x2_d32 | 17,929 | 89.0 | 99.6 | **26.8** | 3.32x | 3.72x |
| js_8x8_d32 | 70,357 | 280.5 | 326.3 | **74.1** | 3.78x | 4.40x |
| js_16x16_d32 | 140,261 | 554.5 | 614.4 | **128.8** | **4.31x** | 4.77x |

## Cold: one fresh process per proof, so zkey parse, shader compile and upload are all inside

| variant | constraints | rapidsnark | our CPU | our Metal | Metal vs rapidsnark |
|---|---:|---:|---:|---:|---:|
| tiny_mul | 2 | 4.6 | 6.0 | 37.1 | 0.12x |
| js_1x1_d8 | 3,359 | 28.6 | 29.6 | 62.2 | 0.46x |
| js_2x2_d16 | 10,153 | 60.3 | 70.5 | 70.7 | 1.00x |
| js_2x2_d32 | 17,929 | 96.2 | 133.6 | **74.9** | 1.28x |
| js_8x8_d32 | 70,357 | 297.5 | 348.6 | **151.9** | 1.96x |
| js_16x16_d32 | 140,261 | 558.2 | 650.0 | **228.5** | 2.44x |

All times in milliseconds. Bold is the fastest prover in that row.

## How to read this

**Metal crosses over at roughly 3,400 constraints warm, and roughly 15,000 cold.** Below
that it loses, and it loses badly at the bottom: `tiny_mul` cold is 37 ms of which about 54
ms of shader compilation is amortised across the process, against a 4.6 ms rapidsnark
proof. There is deliberately no size gate that would quietly run the CPU instead, because
then `--backend metal` would report a number belonging to the other backend.

**The cold column is where the fixed costs live and it is the honest one to quote for a
CLI.** `newLibraryWithSource` costs 53.9 ms once per process for the whole kernel library,
and it is paid before any proving starts.

**The warm column is what a resident proving service actually sees**, and there Metal is
4.3x rapidsnark at 140K constraints.

## Stage split, js_2x2_d32, warm

| stage | CPU | Metal |
|---|---:|---:|
| gather | 816 us | 43 us |
| NTT | 12,632 us | 7,393 us |
| pointwise | 752 us | 0 us (fused) |
| MSM | 87,140 us | 48,380 us |
| assemble | 525 us | 644 us |
| **total** | **101.9 ms** | **56.5 ms** |

Assemble is the ten O(1) group operations of stage 11 and runs on the host in both cases.

## Why this beats the prior art, which it should not be assumed to

zkmopro's Metal-MSM v2, the closest reference implementation on this hardware, publishes
its own MSM as **slower than arkworks CPU at every size measured**: 22.3x slower at 2^12
down to 1.6x slower at 2^24. lambdaworks' Metal MSM carries a race condition documented in
its own source comments. So a Metal Groth16 backend beating CPU is not the expected
outcome and the result deserves suspicion rather than celebration.

What we measured that explains the headroom:

| | this M2 Max |
|---|---|
| CPU field multiply, 12 threads | 0.60 G mul/s |
| GPU field multiply, peak ALU-bound | 4.14 G mul/s |
| GPU field multiply, streaming 2^20 to 2^22 | 2.4 to 2.9 G mul/s |
| GPU field multiply, streaming 2^18 | 0.32 G mul/s, memory-bound at 30 GB/s |
| Dispatch floor, commit and wait | 0.149 ms |
| Shader compile, whole library | 53.9 ms once |

The 2^18 streaming row is the ceiling on the current NTT: at our largest domain that stage
is memory-bound at a fourteenth of peak, which is why the NTT only improves 1.7x while the
gather improves 19x.

## What these numbers are not

- **Not comparable to the earlier VM results** in the research corpus. That harness measured
  `js_1x1_d8` at 7,211 constraints; the same nominal parameters compile to 3,359 here under
  a different circom and circomlib. Within a single run every prover proves the same
  artifact, which is the property that matters.
- **Not from a silent machine.** Load average was about 3.5 on a 12-core box. Medians over
  10 reps absorb most of that; the min and max columns in the CSV are the honest guide to
  how much they did not. One CPU cold row (`js_2x2_d32`, min 107.2 max 263.9) is visibly
  contended and should not be quoted.
- **Not a CUDA result.** No NVIDIA backend exists and none can be tested on this machine.
- **Witness generation is excluded throughout**, for every prover equally. It is 
  unoffloadable CPU work and would compress every ratio here.
