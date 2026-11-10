# CPU proofs beside CUDA proofs on a T4 host, 2026-11-10

CARRY-03 and the OPT-20 decision: how much of the host a CUDA proof leaves idle, what the
CPU backend proves on the same host, and what CPU proof streams add when they run beside GPU
proof streams. Main `35b4645` (the c = 13 MSM of `f8a858d` and `14bdaf9`) plus the
`G16_BENCH_CPU` hook in `snarkrs bench`, which prints each warm proof's wall-clock span and
the process's user+sys CPU time across it.

Runner `bench/aws/coschedule-guest.sh`, tables `bench/aws/coschedule-report.py`. Every stream
is one `snarkrs bench --mode warm` process (`--threads N` sets its rayon pool, every proof
verified); the streams of an experiment start together and run 15 to 30 s. Two boxes, one
session each, driver 595.112.07, binary zigbuilt with `+adx,+bmi2`: a g4dn.xlarge (4 vCPU =
2 cores x 2 threads, Xeon 8259CL) and a g4dn.2xlarge (8 vCPU = 4 x 2, same CPU). One run per
cell, so differences under about 2% are noise.

Throughput is `prove-only`: 1000 / median prove ms of each stream, summed. The bench verifies
each proof on the CPU between reps, so a GPU stream sits in prove() 79 to 100% of the wall
clock (90 to 100% above sha256); the loop rate (proofs that finished in the window all
streams overlap, over the window) is in the raw tables and moves the same way.

## Summary

- **A CUDA proof keeps exactly one host core busy, and leaves the rest idle.** Host CPU time
  per proof equals its wall time (1.00 to 1.08 cores) for every circuit above tiny_mul, in
  both work modes. That is the waiting thread spinning in the driver's synchronize, as far
  as can be told without a profiler: `crates/cuda` sets no `CU_CTX_SCHED_*` flag, and the
  default spins when there are fewer contexts than cores. 75% of the xlarge and 87% of
  the 2xlarge is idle for the whole proof.
- **CPU proof streams beside one GPU stream add 6 to 34% throughput on the 2xlarge**
  (`g1c7`), at a 0 to 5% cost to the GPU stream's own latency. On the xlarge (`g1c3`) the
  gain is 0 to 3% on variable work and 6 to 15% on constant work: 3 vCPUs, one of them the
  spinning thread's hyperthread sibling, are worth little against a T4.
- **Two GPU processes on one T4 add 14 to 18% on the small variable-work circuits**
  (sha256, keccak256) and 0 to 4% on anon-aadhaar; on constant work they lose 5 to 8%.
  With a CPU stream on the remaining six vCPUs (`g2c6`) the 2xlarge reaches +13 to +22% on
  variable work, the best variable-work configuration measured.
- **OPT-20: reject.** Its ceiling, with the CPU's MSM rate as measured beside a GPU stream,
  is 2.8 to 4.5% of a variable-work proof on the xlarge and 5.6 to 9.3% on the 2xlarge, and
  process-level co-scheduling already gets that much or more in throughput with no prover
  change. Constant work has a larger ceiling (12 to 25% on the 2xlarge), because the
  constant-work CUDA MSM is only 2.5 to 6.2x the 8-vCPU host's; that is the constant-work
  kernel's gap to close, not a reason for a device split.
- **tiny_mul proves faster on the CPU than on CUDA**: 1.2 ms against 2.2 ms (variable) and
  30.7 ms (constant work). The CUDA constant-work floor is a 30 ms MSM regardless of size.

## Where a CUDA proof's time goes, and the host's

Single stream, warm, medians. Stage columns are ms (`StageTimings`); host CPU is user+sys of
the whole process across one proof; cores busy = host CPU / prove.

g4dn.xlarge (4 vCPU), CUDA uses `--threads 4`, CPU `--threads 4`:

| circuit | work | backend | prove ms | gather | ntt | pointwise | msm | assemble | host CPU ms / proof | cores busy | host idle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tiny_mul` | var | cuda | 2.2 | 0.0 | 0.0 | 0.0 | 1.8 | 0.2 | 3.5 | 1.54 | 62% |
| `sha256` | var | cuda | 15.6 | 2.2 | 1.4 | 0.0 | 11.5 | 0.2 | 16.8 | 1.08 | 73% |
| `keccak256` | var | cuda | 41.2 | 6.8 | 5.9 | 0.0 | 27.2 | 0.2 | 42.4 | 1.03 | 74% |
| `rsa2048` | var | cuda | 52.1 | 5.5 | 5.9 | 0.0 | 39.4 | 0.2 | 53.4 | 1.02 | 74% |
| `anon-aadhaar` | var | cuda | 310.0 | 31.5 | 57.7 | 0.0 | 200.6 | 0.2 | 311.4 | 1.00 | 75% |
| `tiny_mul` | cw | cuda | 30.7 | 0.0 | 0.0 | 0.0 | 30.2 | 0.2 | 31.9 | 1.04 | 74% |
| `sha256` | cw | cuda | 124.7 | 1.8 | 1.4 | 0.0 | 119.3 | 0.2 | 126.1 | 1.01 | 75% |
| `keccak256` | cw | cuda | 301.4 | 4.7 | 6.5 | 0.0 | 282.6 | 0.2 | 302.7 | 1.00 | 75% |
| `rsa2048` | cw | cuda | 262.4 | 3.8 | 6.3 | 0.0 | 245.7 | 0.2 | 263.7 | 1.00 | 75% |
| `anon-aadhaar` | cw | cuda | 1296.9 | 22.1 | 59.6 | 0.0 | 1173.1 | 0.2 | 1298.5 | 1.00 | 75% |
| `tiny_mul` | var | cpu | 1.2 | 0.0 | 0.0 | 0.0 | 0.4 | 0.2 | 2.8 | 2.44 | 39% |
| `sha256` | var | cpu | 176.7 | 3.1 | 36.5 | 0.9 | 135.5 | 0.2 | 680.9 | 3.85 | 4% |
| `keccak256` | var | cpu | 649.6 | 7.6 | 165.9 | 3.5 | 471.4 | 0.2 | 2500.3 | 3.85 | 4% |
| `rsa2048` | var | cpu | 697.5 | 10.2 | 165.7 | 3.5 | 515.8 | 0.2 | 2708.6 | 3.88 | 3% |
| `anon-aadhaar` | var | cpu | 5078.0 | 101.2 | 1543.0 | 27.9 | 3393.4 | 0.2 | 19991.3 | 3.94 | 2% |
| `tiny_mul` | cw | cpu | 1.2 | 0.0 | 0.0 | 0.0 | 0.5 | 0.2 | 3.2 | 2.66 | 34% |
| `sha256` | cw | cpu | 619.4 | 3.3 | 36.5 | 0.9 | 578.0 | 0.2 | 2439.4 | 3.94 | 2% |
| `keccak256` | cw | cpu | 2474.5 | 7.7 | 165.5 | 3.5 | 2295.9 | 0.2 | 9723.6 | 3.93 | 2% |
| `rsa2048` | cw | cpu | 3173.4 | 10.7 | 164.9 | 3.5 | 2987.1 | 0.2 | 12478.4 | 3.93 | 2% |
| `anon-aadhaar` | cw | cpu | 11892.6 | 103.1 | 1536.6 | 27.8 | 10209.5 | 0.2 | 46957.8 | 3.95 | 1% |

g4dn.2xlarge (8 vCPU), `--threads 8` (tiny_mul not run):

| circuit | work | backend | prove ms | gather | ntt | pointwise | msm | assemble | host CPU ms / proof | cores busy | host idle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sha256` | var | cuda | 15.5 | 2.1 | 1.4 | 0.0 | 11.6 | 0.2 | 16.6 | 1.07 | 87% |
| `keccak256` | var | cuda | 41.6 | 6.9 | 6.0 | 0.0 | 27.3 | 0.2 | 42.8 | 1.03 | 87% |
| `rsa2048` | var | cuda | 52.6 | 5.7 | 5.9 | 0.0 | 39.6 | 0.2 | 53.8 | 1.02 | 87% |
| `anon-aadhaar` | var | cuda | 311.6 | 31.8 | 57.9 | 0.0 | 201.5 | 0.2 | 313.0 | 1.00 | 87% |
| `sha256` | cw | cuda | 125.0 | 1.8 | 1.4 | 0.0 | 119.4 | 0.2 | 126.2 | 1.01 | 87% |
| `keccak256` | cw | cuda | 303.7 | 4.7 | 6.4 | 0.0 | 284.9 | 0.3 | 305.1 | 1.00 | 87% |
| `rsa2048` | cw | cuda | 263.9 | 3.9 | 6.3 | 0.0 | 246.9 | 0.2 | 265.2 | 1.00 | 87% |
| `anon-aadhaar` | cw | cuda | 1304.1 | 22.4 | 59.8 | 0.0 | 1180.3 | 0.2 | 1305.7 | 1.00 | 87% |
| `sha256` | var | cpu | 97.0 | 1.7 | 18.7 | 0.5 | 75.5 | 0.2 | 666.2 | 6.86 | 14% |
| `keccak256` | var | cpu | 360.9 | 3.9 | 84.3 | 1.8 | 269.7 | 0.2 | 2501.2 | 6.93 | 13% |
| `rsa2048` | var | cpu | 379.4 | 5.1 | 84.1 | 1.8 | 287.7 | 0.2 | 2708.8 | 7.14 | 11% |
| `anon-aadhaar` | var | cpu | 2620.6 | 51.4 | 775.3 | 14.0 | 1746.6 | 0.2 | 20182.8 | 7.70 | 4% |
| `sha256` | cw | cpu | 318.7 | 1.8 | 18.8 | 0.5 | 296.8 | 0.2 | 2435.2 | 7.64 | 4% |
| `keccak256` | cw | cpu | 1282.6 | 4.1 | 84.4 | 1.8 | 1191.9 | 0.2 | 9671.4 | 7.54 | 6% |
| `rsa2048` | cw | cpu | 1630.7 | 5.7 | 83.9 | 1.8 | 1538.1 | 0.2 | 12312.0 | 7.55 | 6% |
| `anon-aadhaar` | cw | cpu | 6142.0 | 51.9 | 774.0 | 14.0 | 5275.8 | 0.2 | 47259.8 | 7.69 | 4% |

- The CUDA proof is the same on both boxes (within 1%): host size does not bind a single GPU
  proof. The CPU proof is 1.8 to 2.0x faster on the 2xlarge.
- The MSM is 65 to 76% of a variable-work CUDA proof and 90 to 98% of a constant-work one.
  The CUDA proof is 11 to 16x faster than the xlarge CPU proof on variable work but only 5.0
  to 12x on constant work, and 2.5 to 6.2x the 2xlarge CPU on constant work: the
  constant-work CUDA MSM costs 5.8 to 10.4x the variable one (sha256 119 against 11.5 ms),
  the CPU's 3.0 to 5.8x.

## Throughput, streams side by side

Total prove-only proofs a second, and the change against one GPU stream alone (`g1`). `gN`
is N CUDA processes, `cK` a CPU process with K threads; in `g1c3`, `g1c7`, `g2c6` each CUDA
process runs `--threads 1` and the CPU stream gets the rest, in `g1c4` both take all 4.

g4dn.xlarge:

| circuit | work | g1 | g2 | g1c3 | g1c4 |
|---|---|---:|---:|---:|---:|
| `tiny_mul` | var | 448.2 | 748.8 (+67%) | 1158.9 (+159%) | 1111.9 (+148%) |
| `sha256` | var | 64.98 | 75.63 (+16.4%) | 66.80 (+2.8%) | 66.12 (+1.8%) |
| `keccak256` | var | 24.54 | 27.92 (+13.8%) | 24.49 (-0.2%) | 24.44 (-0.4%) |
| `rsa2048` | var | 19.25 | 21.14 (+9.8%) | 19.68 (+2.2%) | 19.57 (+1.7%) |
| `anon-aadhaar` | var | 3.21 | 3.21 (0.0%) | 3.29 (+2.5%) | 3.28 (+2.2%) |
| `tiny_mul` | cw | 32.63 | 31.40 (-3.8%) | 791.7 | 747.2 |
| `sha256` | cw | 8.04 | 7.51 (-6.6%) | 9.24 (+14.9%) | 9.18 (+14.2%) |
| `keccak256` | cw | 3.31 | 3.13 (-5.4%) | 3.63 (+9.7%) | 3.61 (+9.1%) |
| `rsa2048` | cw | 3.80 | 3.58 (-5.8%) | 4.03 (+6.1%) | 3.98 (+4.7%) |
| `anon-aadhaar` | cw | 0.77 | 0.72 (-6.5%) | 0.83 (+7.8%) | 0.82 (+6.5%) |

g4dn.2xlarge:

| circuit | work | g1 | g2 | g1c7 | g2c6 |
|---|---|---:|---:|---:|---:|
| `sha256` | var | 63.76 | 75.18 (+17.9%) | 70.67 (+10.8%) | 76.54 (+20.0%) |
| `keccak256` | var | 24.03 | 27.39 (+14.0%) | 25.48 (+6.0%) | 29.36 (+22.2%) |
| `rsa2048` | var | 18.88 | 18.83 (-0.3%) | 20.82 (+10.3%) | 22.33 (+18.3%) |
| `anon-aadhaar` | var | 3.20 | 3.32 (+3.7%) | 3.52 (+10.0%) | 3.63 (+13.4%) |
| `sha256` | cw | 7.99 | 7.42 (-7.1%) | 10.69 (+33.8%) | 9.97 (+24.8%) |
| `keccak256` | cw | 3.30 | 3.12 (-5.5%) | 3.99 (+20.9%) | 3.76 (+13.9%) |
| `rsa2048` | cw | 3.78 | 3.51 (-7.1%) | 4.31 (+14.0%) | 4.01 (+6.1%) |
| `anon-aadhaar` | cw | 0.77 | 0.71 (-7.8%) | 0.91 (+18.2%) | 0.84 (+9.1%) |

- **The GPU stream barely notices a CPU stream.** Its prove time beside 7 CPU threads: sha256
  15.7 to 16.3 ms, keccak256 41.6 to 43.6, rsa2048 53.0 to 54.2, anon-aadhaar 312.6 to
  313.7. `--threads 1` on the CUDA process is enough; giving both all cores (`g1c4`) is
  0.2 to 1.4 points worse than splitting them (`g1c3`).
- **The CPU stream keeps 87 to 92% of its alone rate** beside a GPU stream on the 2xlarge
  (7 threads: sha256 105 against 97 ms on 8), 75 to 79% on the xlarge (3 threads).
- **Two CUDA processes overlap what one leaves idle** (launch gaps, host gather, the
  synchronize round trips) only where those are a large share: sha256 and keccak256 gain,
  anon-aadhaar does not, and constant work, which is MSM-bound, loses 5 to 8% to the GPU
  time-slicing. rsa2048 gained 9.8% on the xlarge and nothing on the 2xlarge; one run each,
  so treat that cell as unresolved.
- **4 vCPUs are the binding limit on the xlarge.** The same experiment on the 2xlarge gains
  2 to 4 times as much, because the CPU stream there has 7 vCPUs (3 free physical cores)
  instead of 3 (1 free core and the spinning thread's sibling).

## OPT-20: the ceiling of a device-split MSM

The best static split gives the CPU the share r_c / (r_c + r_g) of every MSM, so the MSM stage
drops from T_g to T_g T_c / (T_g + T_c), a saving of T_g^2 / (T_g + T_c). T_g is the CUDA MSM
stage above; T_c is the CPU's MSM stage alone on every core, and as measured beside a GPU
stream on the remaining cores (`g1c3` / `g1c7`), which already pays the contention a split
would cause. The ceiling ignores what a split adds: a second bucket reduce and window
combine per MSM on the CPU, the host copy of the bases the CUDA backend does not keep today,
and a mis-calibrated fraction.

| circuit | work | xlarge: GPU MSM ms | CPU MSM beside GPU | ceiling ms | % of proof | 2xlarge: GPU MSM ms | CPU MSM beside GPU | ceiling ms | % of proof |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sha256` | var | 11.5 | 175.3 | 0.7 | 4.5% | 11.6 | 80.7 | 1.4 | 9.3% |
| `keccak256` | var | 27.2 | 623.2 | 1.1 | 2.8% | 27.3 | 290.4 | 2.3 | 5.6% |
| `rsa2048` | var | 39.4 | 691.2 | 2.1 | 4.1% | 39.6 | 314.7 | 4.4 | 8.4% |
| `anon-aadhaar` | var | 200.6 | 4362.3 | 8.8 | 2.8% | 201.5 | 2042.1 | 18.1 | 5.8% |
| `sha256` | cw | 119.3 | 727.8 | 16.8 | 13.5% | 119.4 | 337.2 | 31.2 | 25.0% |
| `keccak256` | cw | 282.6 | 2927.1 | 24.9 | 8.3% | 284.9 | 1328.1 | 50.3 | 16.6% |
| `rsa2048` | cw | 245.7 | 3786.7 | 15.0 | 5.7% | 246.9 | 1726.0 | 30.9 | 11.7% |
| `anon-aadhaar` | cw | 1173.1 | 13079.6 | 96.6 | 7.4% | 1180.3 | 5891.7 | 197.0 | 15.1% |

With T_c alone on every core the xlarge ceilings are 3.6 to 5.8% (variable) and 7.1 to
16.4% (constant work), the 2xlarge 6.0 to 9.9% and 12.9 to 27.4%. tiny_mul is left out: its
CPU MSM (0.4 ms) beats the GPU's (1.8 ms), so the answer there is to prove on the CPU.

**Recommendation: reject OPT-20.**

- On variable work, the default, the ceiling is at most 4.5% of a proof on the xlarge and
  9.3% on the 2xlarge, before the split's own overheads. The item's estimate (about 9% on a
  T4 host) was the best case of the bigger box, not the typical one.
- For throughput, process-level co-scheduling already beats it with no prover change:
  `g1c7` adds 6 to 11% and `g2c6` 13 to 22% on variable work on the 2xlarge, against an
  OPT-20 ceiling of 5.6 to 9.3%. The two do not stack well either: a split takes the cores
  the CPU stream would use.
- What OPT-20 alone offers is single-proof latency, and only on constant work is that
  sizeable (12 to 25% on the 2xlarge). There the gap is the constant-work CUDA MSM itself,
  10x its variable path for sha256 against 4x on the CPU; closing that helps every host,
  while a split would need per-host calibration, a resident host copy of the bases, and
  high-risk plumbing across the groth16 and backend crates.

## What this means for COST.md

Every GPU row there is a single GPU stream with the host idle. On the g4dn.2xlarge, one GPU
stream plus a 7-thread CPU stream (or two GPU streams plus a 6-thread CPU stream on small
variable-work circuits) is 13 to 22% more proofs for the same instance-hour on variable work
and 14 to 34% on constant work (the better of `g1c7` and `g2c6` per circuit). On the
g4dn.xlarge a CPU stream adds 0 to 3% on variable work and 6 to 15% on constant work; a
second GPU process adds 0 to 16% on variable work there and loses on constant work. The rows stay as measured; the co-scheduled figure is a deployment choice, not a
prover change.

## New leads

- The CUDA context's default scheduling spins a whole core per waiting proof. A blocking
  sync (`CU_CTX_SCHED_BLOCKING_SYNC` on the primary context) would hand that core to a
  co-scheduled CPU stream; worth one same-box A/B of the GPU stream's latency against the
  CPU stream's gain (on the xlarge it is a quarter of the host).
- Small circuits should prove on the CPU: tiny_mul 1.2 ms against 2.2 ms (CUDA variable) and
  30.7 ms (CUDA constant work). A size threshold in the CLI/library backend choice.
- The constant-work CUDA MSM has a fixed floor of about 30 ms (tiny_mul) and costs 5.8 to
  10.4x the variable path, against 3.0 to 5.8x on the CPU.
- Two CUDA processes gain 14 to 18% on sha256 and keccak256: the GPU idles between one
  proof's stages. Overlapping two proofs inside one process (two streams) could take that
  without a second key upload.
