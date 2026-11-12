# Incomplete T4 reduction-geometry calibration, 2026-11-12 UTC

No candidate, default change or performance win is accepted from this batch.
The standard gate and all five phase cases completed, but the first forced-c13
constant-work CLI proof exceeded its 120-second stage cap. The supervisor killed
its session-contained descendants. The controller retrieved the partial evidence
and stopped the instance. Controller and terminal status are both 1, not success.

## Protocol and completed evidence

Rust source `b88ce57`, g4dn.xlarge, 4 vCPU, 16 GiB host RAM, Tesla T4 reporting
15360 MiB. Locked GNU 2.31 cross-builds with ADX/BMI2 were prepared locally.
This is not the README's published 8-vCPU g4dn.2xlarge benchmark protocol.

- All 27 frozen payload hashes and retrieved worker sources match. Original
  tiny-mul and SHA-256 zkey, witness and vkey hashes match the payload manifest.
- The standard gate passed 124 executions: ordinary library 63, selected ignored
  library 24, legacy 1, warm 1, adversarial 15, pipeline 11, field 5 and gnark 4.
  Its fresh per-job independent gnark marker is present.
- The serial phase order was RG8-before, RG4, RG16, RG32, RG8-after. Every case
  completed all 88 distinct raw phase rows with 15 measured samples per row.
  The pinned test performs two warmups and checks CPU answers outside clocks.
- All cases used c13, n=65536 and 262144, mixed and uniform scalars, G1 and G2.
  Each has eight matching geometry headers and one matching Shapes record.
  Other shape fields remain wide_span=32 and scatter_rows=16384; effective
  headers report w=20, wide=2048 and split=16384.
- The five standard CLI proofs and the first additional default-work c13/RG8
  proof passed native macOS rapidsnark against original local vkeys. Public
  signals match the public prefix derived from the pinned original witnesses,
  not just another retrieved proof. Linux independent and own positive receipts
  were also checked for all six existing proofs.

These are completeness and integrity checks on partial evidence. They do not
replace the nine missing sweep proofs or turn the failed batch into a passed
optimisation gate. The strict full-protocol parent validator rejects this run.

## Failure and limits

The planned protocol has 46 guarded commands. Only the first 19 completed:
standard gate, its five independent checks, five phase/parser pairs, then the
first default-work proof and its two verifier checks. The next command was the
constant-work proof for the same c13/RG8 case. It crossed its stage cap at the
last resource sample of 120.126 seconds, with no proof produced.

This was the per-command cap, not exhaustion of the independent AWS deadline.
The CLI log is empty, so it does not establish which preparation stage consumed
the time. At the final sample, device usage was 107 MiB and board power was
31.77 W. Expensive module compilation or loading is a hypothesis, not a measured
attribution. The source documents substantial cold NVRTC and driver-JIT costs;
a bounded follow-up needs explicit preparation timings rather than assuming
that the standard gate warmed every later process's effective cache.

The constant-work proof checks were correctness-only. They had c13 and RG forced
and CPU fallback disabled. Neither the completed proof nor the timed-out attempt
is a default end-to-end proof timing measurement.

All completed and failed-stage resource samples satisfy the unchanged limits:
used device memory at most 13824/15360 MiB and available host RAM at least 4 GiB.
Observed peak device usage was 6507 MiB; minimum available host RAM was
13443576 KiB. These are sampled bounds, not exact transient maxima.

## Retention and next gate

Results were retrieved before shutdown. Native lifecycle status and a separate
EC2 readback both confirmed stopped. Only this batch's SSH ingress rule was
revoked. Independent AWS stop schedules were retained.

The private raw archive is retained locally as
`target/cuda-validation-20261111/reduce-incomplete-results.tar.gz`, SHA256:

`b58ad9cc41d9eba7e4b630f0f219dc9b414fed9e22c34b639ff5c8767c3a1790`

A follow-up must preserve this failed result, budget cold preparation explicitly,
keep the same numerical checks and memory limits, and complete both default and
constant-work independent proof checks for every geometry. Only a complete gate
can support comparison of isolated reduce, host combine and allocating batch
device/wall metrics against both separate RG8 baselines. No baseline pooling,
automatic default selection or end-to-end speed claim follows from these data.
