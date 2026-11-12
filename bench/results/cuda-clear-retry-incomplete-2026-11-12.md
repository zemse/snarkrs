# Incomplete T4 full-clear retry, 2026-11-12 UTC

No optimisation, default change or performance win is accepted. This retry
completed 14 of 55 planned commands, then the new full-clear MSM test failed a
CPU-side fixture precondition. Controller and terminal status are both 1. The
controlled five-case phase sweep and complete twenty-proof inventory were not
reached. The [first failed attempt](cuda-clear-incomplete-2026-11-12.md) remains
preserved separately.

## Protocol and partial acceptance

Candidate Rust source `68e314d869ee726b08a15a95af4cb12b2a7d86b2`, still unmerged.
Hardware: g4dn.xlarge, 4 vCPU, 16 GiB host RAM, Tesla T4 reporting 15360 MiB.
Locked GNU 2.31 ADX/BMI2 cross-builds were prepared locally. This is not the
published 8-vCPU g4dn.2xlarge benchmark protocol.

The retry adds one explicit cold-preparation test capped at 1800 seconds before
all original 54 commands and 20 proof specifications, unchanged. Minimum
admission is 4500 seconds with a 300-second retrieval reserve. That admission
is a heuristic, not a worst-case completion guarantee within the stop window.
The original standard gates retain their warm steps and 1200-second caps.

| Completed group | Passing test executions |
|---|---:|
| Explicit preparation | 1 |
| Standard gate, full-clear off | 127 |
| Standard gate, full-clear on | 127 |
| New poisoned-scope gate | 1 |
| Total partial passes | 256 |

Each standard gate has ordinary library 66, selected ignored library 24, legacy
1, warm 1, adversarial 15, pipeline 11, field 5 and gnark 4. Both fresh independent
gnark receipts were checked. The next new MSM test has one failed result, not a
passing completion. Inner case progress is not extra separately passed tests.

All ten available standard CLI proofs passed fresh native macOS rapidsnark
verification against original local vkeys. Original zkey, witness and vkey
hashes match the manifest; retrieved public signals match the pinned witness
prefixes and original public files. Own and Linux independent receipts were
checked too. These ten proofs comprise eight GPU proofs and two deliberate
CPU VRAM-refusal proofs. They do not replace the ten missing later proofs.

## Fixture failure

The failing case is Constant work, n=65, c=13, slice=64, the local pure-uniform
fixture. An independent integer-only reconstruction of its exact seeded scalars,
signed-digit recoding, row geometry and base weights found all 1300 signed digits
nonzero. Therefore all 1280 CPU-expected dummy sums legitimately equal zero.
The helper nevertheless requires nonzero expected dummy arithmetic for every
nonempty Constant fixture.

The panic occurs at that CPU precondition, after the digit/count/cursor/entry
checks but before this case's point launch and numerical MSM comparison. It is
not a GPU instrumentation-counter failure or evidence of a CUDA clear defect.
The same fixture at c=16 has the same precondition problem in CPU reconstruction;
that later Constant case was not reached on the device.

A correction must make the local gate fixture exercise a nonidentity dummy
bucket while retaining every numerical, digit, dummy-bucket and accounting
assertion. The global scalar generator and pure timing datasets must not change.
This diagnosis does not accept the candidate or mark unexecuted checks passed.

## Preparation, guards and retention

Explicit preparation passed in 321.76 seconds; both later standard gates passed.
Its retained raw log separates NVRTC and module-load timings. This does not
establish why the earlier aggregate warm preparation took 1034.72 seconds, nor
guarantee useful cache persistence across other processes or later runs.

All 3703 retained samples across the fifteen entered stages satisfy the unchanged
4 GiB available-host floor and 90% device ceiling. Sampled peak device usage was
6533/15360 MiB; minimum host available was 13631080 KiB. These are sampled bounds,
not exact transient peaks or complete-protocol memory acceptance.

The full parent validator rejects this failed controller. The comparator's fresh
validator subprocess exits nonzero; no accepted timing analysis is published.
A separate negative-attempt audit verified all 103 archive-member pairs and
complete extracted work/out membership, not extraction-safety certification.

Results were retrieved before shutdown. Native status and a separate EC2 readback
confirmed stopped. Only this retry's newly created SSH rule was revoked. All AWS
stop deadlines were retained.

The private raw archive is retained at
`target/cuda-validation-20261111/clear-retry-incomplete-results.tar.gz`, SHA256:

`7080bc72a42b7a58d4522759546de2444a4ddf7f1cb940179191ec8b15ab1b00`

Next: correct the invalid local fixture setup with a CPU invariant regression,
freeze its new source and artifacts, then require a fresh complete device run.
Neither partial proof success nor a fixture-only correction substitutes for the
remaining gate, five controlled timing cases and full independent proof inventory.
