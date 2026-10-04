# L40S onion ladder results

- 2026-10-04, us-east-1, g6e.xlarge, NVIDIA L40S, 4 vCPU, 32 GiB host specification.
- Instance i-0686d78246debd4bf. Existing certified Poseidon(2) fixtures, benchmark-only keys.
- Three cold and five warm repetitions per CPU/CUDA configuration. One proof in flight.
- All 64 retrieved rows passed schema, sequence, count, finite-timing and verification-scope validation.

## timings

Seconds, median (min to max). Speedup = CPU median / CUDA median.

| Domain | Mode | CPU seconds | CUDA seconds | Speedup |
| --- | --- | ---: | ---: | ---: |
| p21 | cold | 12.548 (12.519 to 12.558) | 2.026 (2.018 to 2.030) | 6.19x |
| p21 | warm | 11.982 (11.757 to 12.074) | 0.363 (0.314 to 0.364) | 33.01x |
| p22 | cold | 24.724 (24.626 to 24.949) | 3.664 (3.660 to 3.686) | 6.75x |
| p22 | warm | 23.412 (23.224 to 23.436) | 0.718 (0.630 to 0.720) | 32.60x |
| p23 | cold | 48.914 (48.666 to 49.216) | 6.971 (6.961 to 6.977) | 7.02x |
| p23 | warm | 46.279 (45.953 to 46.594) | 1.438 (1.258 to 1.441) | 32.18x |
| p24 | cold | 97.078 (96.952 to 98.282) | 13.539 (13.530 to 13.548) | 7.17x |
| p24 | warm | 93.070 (92.127 to 93.424) | 2.920 (2.520 to 2.924) | 31.88x |

Cold = fresh process, parsing/upload included, after CUDA JIT warmup. Warm = resident proving, preparation excluded. First CUDA probes and independent verification are outside repetition timing. Small repetition counts provide descriptive ranges, not confidence intervals.

Every proof passed our verifier. Every cold proof passed rapidsnark. Independent warm verification covers the first proof of each backend/domain through the rapidsnark adapter. Remote snarkjs compatibility and rapidsnark prover performance were not measured. Initial load was 1.24, with an explicit ceiling of 4; rows are marked loaded=no.

## first-proof memory gate

All four CUDA probes matched reference public signals and passed both verifiers before repetitions. Sampled host VmHWM, minimum system MemAvailable and total device memory use, every 0.2 seconds plus query time:

| Domain | Host HWM GiB | Minimum available host GiB | Device peak MiB |
| --- | ---: | ---: | ---: |
| p21 | 1.498 | 28.066 | 4,287 |
| p22 | 2.894 | 27.054 | 8,095 |
| p23 | 5.624 | 25.127 | 15,743 |
| p24 | 11.135 | 21.386 | 30,975 |

Driver-reported device capacity: 46,068 MiB. No sample crossed the 90% device-use or 4 GiB available-host thresholds. These are first-proof observations, not exact whole-run peaks. No OOM or configuration failure occurred. The unchanged repetition plan passed the conservative budget check.

## provenance and recovery

Local archive: `bench/results/aws-i-0686d78246debd4bf/`. Raw results and large fixtures are not shipped in this repository.

- `remote/onion-ladder/comparison.csv`, 64 rows.
- CSV SHA256: `0b5a6837495314eab84fc664cae7995e8626aea1e18554e0a11bd57fc5eedc76`.
- `remote/onion-ladder/validated-summary.json`, `first-proofs.json`, per-domain memory samples and both verifier logs.
- `local-validation.json`, `RESULTS.md`, `job.log`, cloud deadline records and watchdog logs.
- Detached runner log: `target/aws-bundles/l40s-ladder.log`.

Launch at 20:05:10 UTC. Cloud forced-stop deadline at 23:05:08 UTC, guest 60-minute lease / 180-minute hard cap and separate laptop watchdog. Results were retrieved before graceful stop requested at 21:37:02 UTC. Forced-stop fallback requested after four minutes in stopping. EC2 and the runner confirmed stopped by 21:46:40 UTC. Final-state evidence is retained locally; forced shutdown can require filesystem repair. No termination or EBS deletion.

Fresh official Linux on-demand rate: $1.861/hour, three-hour compute cap $5.583, excluding storage and transfer. Retained volumes across the L40S, old T4 and two timer-test micros total 256 GiB gp3. At an assumed $0.08/GiB-month, baseline storage is $20.48/month; the storage rate was not freshly verified.
