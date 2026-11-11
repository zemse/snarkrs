# Real onion CUDA correctness on a T4, 2026-11-11 UTC

Earlier partial validation, not a complete ladder or a performance result. Real
p21, p22 and p23 proofs passed; p24 was still open because both attempts crossed
the existing headroom guard. No default tuning or safety threshold was changed.
The later [capacity validation](cuda-capacity-2026-11-11.md) completed p24 with an
explicit extra VRAM reserve and the same guard.

One g4dn.2xlarge, 32 GiB host RAM, Tesla T4 reporting 15360 MiB, driver 595.112.07,
CUDA 13.2. The locally zigbuilt binary is from `792eb10`, target
`x86_64-unknown-linux-gnu.2.31`, with `-C target-feature=+adx,+bmi2`. Proofs used
`--backend cuda --fallback false`. The standard gate passed first; see
`cuda-gate-2026-11-11.md`.

## Observations

| fixture | window selection | result | sampled device maximum, MiB |
|---|---|---|---:|
| onion_p21 | default | independently and internally verified | 2699 |
| onion_p22 | default | independently and internally verified | 5163 |
| onion_p23 | default | independently and internally verified | 10091 |
| onion_p24 | default | headroom abort | 14059 |
| onion_p24 | G16_CUDA_MSM_C=12 | headroom abort | 14443 |

The existing observer requires at least 4 GiB available host RAM and at most 90%
of total device memory, here 13824 MiB. It samples VmHWM and nvidia-smi every
0.2 seconds plus query time. These maxima are sampled, not exact allocation peaks;
the aborted runs did not reach completion. Neither is an accepted p24 proof or
evidence of a driver OOM. The smaller-window retry did not resolve this gate.

All 24 runtime files matched the certified SHA256 manifests before the default
run. The p24 retry checked its six files again, and its retrieved run-config
records the override as 12. Each accepted proof matched the reference public
signals, passed the independent Linux rapidsnark verifier and our verifier, and
was independently reverified after retrieval with the native macOS rapidsnark
verifier and the original local verification key.

The first observation included cold setup/JIT work. No timing from these
observations is accepted as a benchmark row or a speedup measurement.

## Evidence and cleanup

Locally retained raw archives:

- `target/cuda-validation-20261111/onion-default-results.tar.gz`, SHA256
  `b451ea6cc8cde5b60cbb2f149d22107930d703b9604402298d2b6b01a083978e`.
- `target/cuda-validation-20261111/onion-p24-c12-results.tar.gz`, SHA256
  `60792cebef328666b1cdef2b04e8b65580de7de4c69b7df9ca303f61d552967e`.

Both jobs retrieved their results before stopping. Native lifecycle status
confirmed the instance stopped after the retry. Independent stop deadlines were
left armed, and only this session's SSH ingress rule was revoked. The stopped
instance and its EBS volume are retained. Temporary private S3 staging remains
pending explicit deletion approval; no original local fixtures were removed.

Follow-up: the [capacity validation](cuda-capacity-2026-11-11.md) completed and
independently verified p24 without relaxing the headroom gate. The full T4 table
refresh and remaining optimisation work are separate, unfinished tasks.
