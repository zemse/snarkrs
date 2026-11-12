# Incomplete T4 full-clear calibration, 2026-11-12 UTC

No candidate, default change or performance win is accepted from this batch.
The first standard gate exceeded its 1,200-second outer stage cap. The controller
retrieved the partial evidence and stopped the instance. Controller and terminal
status are both 1. None of the 54 planned commands completed as a whole.

## Protocol and partial evidence

Candidate Rust source `68e314d869ee726b08a15a95af4cb12b2a7d86b2`, unmerged,
g4dn.xlarge, 4 vCPU, 16 GiB host RAM, Tesla T4 reporting 15360 MiB. Locked GNU
2.31 cross-builds with ADX/BMI2 were prepared locally. This is not the README's
published 8-vCPU g4dn.2xlarge benchmark protocol.

The proposed clear is opt-in. The failed first gate had the override off; the
on gate, two explicit new device gates, five phase cases and twenty-proof
inventory were not reached. No full-clear device correctness or comparative
performance acceptance follows from this run.

The retained first-gate log records these successful test executions:

| Group | Passed |
|---|---:|
| Kernel-unit warm preparation | 1 |
| Ordinary library | 66 |
| Selected ignored library | 24 |
| Legacy accumulator | 1 |
| Total partial executions | 92 |

The adversarial group started but has no successful completion in the retained
log. The required standard gate has 127 executions. Neither its complete
inventory nor a fresh independent gnark receipt is established here. The full
parent validator rejects the controller failure; the comparator's actual fresh
validator subprocess exits nonzero and no accepted analysis is published.

## Failure and limits

Warm preparation passed in 1,034.72 seconds. The later selected ignored-library
process passed in 104.51 seconds. The outer first-gate resource series ends at
1,200.031 seconds. This was the stage cap, not exhaustion of the independent AWS
deadline.

The warm log's unit summaries include approximately 1,034 seconds for MSM,
constant-work MSM and FFT preparation. These are preparation elapsed times,
not an isolated NVRTC measurement or proof of a cache failure. Detailed per-unit
internal logs were not retained by the success-only gate-output copy because
the standard gate did not complete.

The existing guest script budgets 30 minutes for its warm step and documents
substantial cold preparation. Wrapping that entire gate in 20 minutes was not
a valid cold-path budget. A retry must budget preparation explicitly without
removing existing gate checks or assuming every later process is warm.

All 4,899 retained resource samples satisfy the unchanged limits: used device
memory at most 13824/15360 MiB and available host RAM at least 4 GiB. Observed
peak device usage was 6507 MiB; minimum available host RAM was 6715104 KiB.
These are sampled bounds, not exact transient maxima or CPU-contention evidence.

## Retention and next gate

Results were retrieved before shutdown. Native lifecycle status and a separate
EC2 readback both confirmed stopped. Only this batch's newly created SSH ingress
rule was revoked. All independent AWS stop schedules were retained.

The private raw archive is retained locally as
`target/cuda-validation-20261111/clear-incomplete-results.tar.gz`, SHA256:

`997d70812b3f77b169d12c25fe90b7d23cedb8350dfea2e64c2d4dca684176f3`

Preserve this failed attempt. A follow-up needs a frozen, reviewed preparation
budget and the complete off/on gates, explicit new device checks, original
fixtures and independent proof verification. Keep before/after original and
candidate-off references separate when comparing candidate-on. Clear time,
downstream accumulation/reduction, readback plus host combination, allocating
batch device time and wall time must be considered together. Missing evidence
cannot be replaced by the partial ordinary-library and warm successes above.
