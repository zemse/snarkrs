# L40S onion ladder results

Chained Poseidon(2), benchmark-only keys. NVIDIA L40S, 4 vCPU, 32 GiB host RAM. Three cold and five warm repetitions per CPU/CUDA configuration, one proof in flight.

## timings

Seconds, median (min to max). Speedup = CPU median / CUDA median.

| Domain | Constraints | Mode | CPU seconds | CUDA seconds | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| 2^21 | 1,572,714 | cold | 12.548 (12.519 to 12.558) | 2.026 (2.018 to 2.030) | 6.19x |
| 2^21 | 1,572,714 | warm | 11.982 (11.757 to 12.074) | 0.363 (0.314 to 0.364) | 33.01x |
| 2^22 | 3,145,428 | cold | 24.724 (24.626 to 24.949) | 3.664 (3.660 to 3.686) | 6.75x |
| 2^22 | 3,145,428 | warm | 23.412 (23.224 to 23.436) | 0.718 (0.630 to 0.720) | 32.60x |
| 2^23 | 6,291,373 | cold | 48.914 (48.666 to 49.216) | 6.971 (6.961 to 6.977) | 7.02x |
| 2^23 | 6,291,373 | warm | 46.279 (45.953 to 46.594) | 1.438 (1.258 to 1.441) | 32.18x |
| 2^24 | 12,582,746 | cold | 97.078 (96.952 to 98.282) | 13.539 (13.530 to 13.548) | 7.17x |
| 2^24 | 12,582,746 | warm | 93.070 (92.127 to 93.424) | 2.920 (2.520 to 2.924) | 31.88x |

Cold includes fresh process startup, parsing and upload, after CUDA JIT warmup. Warm excludes preparation. Independent verification is outside timing. Small repetition counts provide descriptive ranges, not confidence intervals.

All 64 records passed validation. Every proof passed our verifier; independent rapidsnark verification covered every cold proof and the first warm proof per backend/domain.

Per-repetition workload sizes and timings: `../results/l40s-onion-timings.csv`.
