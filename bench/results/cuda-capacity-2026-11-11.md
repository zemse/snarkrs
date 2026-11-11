# Real p24 CUDA proof within the T4 headroom guard, 2026-11-11 UTC

The real onion p24 proof completed with an explicit 1536 MiB VRAM reserve and
passed independent verification. Its sampled device maximum was 12139 MiB,
below the unchanged 13824 MiB ceiling. This completes the T4 p21 through p24
correctness coverage together with the earlier p21 and p22 observations. It does
not establish a complete p24 proof with the reserve unset.

These are correctness and capacity observations, not performance rows. Memory
tracing adds synchronization; no speedup or benchmark timing is accepted here.

## Configuration

One g4dn.2xlarge, 32 GiB host RAM, Tesla T4 reporting 15360 MiB, driver 595.112.07,
CUDA 13.2. Rust source `86dd965`, locally cross-built on the M2 Max with locked
dependencies, target `x86_64-unknown-linux-gnu.2.31` and
`-C target-feature=+adx,+bmi2`. No Rust or Go toolchain was installed remotely,
and no Node installation was performed.

Proofs used `--backend cuda --fallback false`. No MSM window override was set.
The reserve scenarios used `G16_CUDA_VRAM_RESERVE=1610612736`; the default scenario
used zero. All three enabled `G16_CUDA_MEMORY_TRACE=1`, `G16_CUDA_PREPARE=1` and
`G16_CUDA_TIME=1` for diagnosis.

The observer still requires at least 4 GiB available host RAM and at most 90%
device usage. It samples every 0.2 seconds plus query time. Device maxima are
sampled, not exact transient allocation peaks.

## Results

| fixture | extra reserve, MiB | result | sampled device maximum, MiB | minimum available host RAM, GiB |
|---|---:|---|---:|---:|
| onion_p23 | 0 | independently and internally verified | 10091 | 26.621 |
| onion_p23 | 1536 | independently and internally verified | 10091 | 26.614 |
| onion_p24 | 1536 | independently and internally verified | 12139 | 23.640 |

Every retained memory sample satisfied both unchanged guards. The p24 sampled
maximum was 1685 MiB below the device ceiling. Sampled host VmHWM was 3.507 GiB
for both p23 runs and 6.764 GiB for p24.

All twelve runtime-file digests matched the certified manifests on the guest.
After retrieval, the coordinator rehashed the original local files and compared
them with those receipts. Every onion proof matched the original reference
public signals, passed Linux rapidsnark and our verifier, and was independently
reverified locally with native macOS rapidsnark and the original local vkey.

The trace showed p24 selecting runs of 8388608 points from a cap of 16777216,
under a queried scratch budget of 3334275072 bytes. All ten p24 job-output buffer
shapes matched the planner's requested word counts. Each p23 run used one
five-job batch, also with exact requested/allocated logical shapes. These
logical lengths do not include driver allocation rounding.

## What changed

- `4ab71a4`: account for every reduction partial in the window-sum buffer. The
  previous estimate omitted the reduction-group multiplier. A regression failed
  before the fix and passed afterward; this small correction alone is not claimed
  to explain the earlier headroom aborts.
- `464b63b`: an opt-in extra free-memory reserve, default zero. Memory plans use
  current free bytes less the existing 64 MiB allowance and the extra reserve,
  then apply the existing operation limit. Gnark construction and arena
  allocation also check the reserve.
- `412fe8c`: default-off phase snapshots, selected runs and actual logical buffer
  shapes. These diagnostics deliberately perturb synchronization.
- `86dd965`: repair 21 existing public CUDA rustdoc link errors.

The normal zkey path does not use `MSM_CHUNK`: it tries all five jobs, then fits
scheduled runs if necessary. The entire stage scratch stays resident during
MSMs. For p24, bases, transform tables and one stage scratch alone request
10209830048 bytes, before CSR data, live indices and runtime overhead. An
operation-level `G16_CUDA_VRAM_LIMIT` is not a total-residency ceiling.

The reserve is a planning policy, not a hard process allocation cap. Set it
before construction and keep it unchanged during the prover's lifetime.
Separate contexts, other processes, lazy driver allocations and allocator
rounding are not isolated by it. Keep the external guard.

## Gates and evidence

The standard device gate passed 120 test executions: warm-up 1, ordinary library
61, selected ignored library 22, separate legacy accumulator 1, adversarial 15,
pipeline 11, field 5 and gnark 4. Four default/constant-work CUDA CLI cases and
the expected CPU VRAM-refusal fallback passed. The extended every-variant sweep
was not run.

Additional cases passed: malformed reserve refusal, maximal reserve refusal
without output files, two positive-reserve CUDA CLI proofs with both verifiers,
and four positive-reserve gnark test executions with a fresh independent oracle
receipt. The coordinator locally reverified ten retrieved proofs in total:
three onions, four standard CUDA CLI cases, the expected CPU fallback and two
positive-reserve CLI cases.

Host checks: 61 CUDA library tests passed, 25 device tests ignored; formatting,
strict CUDA rustdoc and both locked Linux cross-builds passed. An independent
read-only source review found no new actionable defect or deadlock.

Raw archive: `target/cuda-validation-20261111/onion-capacity-results.tar.gz`,
SHA256 `4d1ad2c92b54496fbbe622a1a9e92c6ff66c6eb0aae0de8e720590ffa3592d31`.
It contains the sanitized input/build manifests, runtime checksum receipts,
worker source, gate logs, oracle receipts, proofs, verifier logs and memory
traces. Parent verification details are retained in the private session state.

## Cleanup and remaining work

The batch retrieved its results before shutdown. Controller exit was zero;
native lifecycle status and an independent EC2 readback confirmed stopped.
Only this batch's SSH ingress rule was revoked. All AWS stop deadlines remain
armed, and the stopped instance and its EBS volume are retained. Local signed
GET capability files were removed; private S3 staging still awaits explicit
deletion approval. No original fixture was removed.

The expanded executable-file search found no `node` under `/home/ubuntu`,
`/opt` or `/usr/local` through depth 12. This is not proof of absence everywhere;
`nodejs` and other preinstalled alternatives still need checking. Portable Node
installation has not been approved.

Next: serial intrinsic-led CUDA work, then the final T4 table with the required
repetitions and accurate machine stamps. The current 8-vCPU host must not be
silently reported as the published 4-vCPU machine.
