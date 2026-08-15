# Performance

Proving latency per machine, per backend, per circuit. Generated from
`bench/results/history.csv` by `bench/scripts/perf_history.py render`.
Do not edit by hand; add a measurement with `perf_history.py ingest`.

`baseline` is the `round3-before` measurement where the machine has
one, and the generic `baseline` label otherwise. `current` is the newest
measurement for that configuration. A negative delta is a speedup.

Every timing is the median over the run's reps, and every proof behind a
timing was verified before it was recorded.

**A delta is only meaningful between two runs taken under the same
conditions.** The original `baseline` rows are a different session on a
different day, and on the M2 Max they disagreed with a controlled
re-measurement of the *unchanged* tree by 6%, which is larger than most
wins worth reporting. Rows compared against that label are therefore an
indication and not a result. The `round3-before` / `round3-after` pair was
taken interleaved in one session, alternating order each round, and is the
only comparison here that isolates a code change from the machine.

## Circuits

| variant | constraints |
|---|---:|
| tiny_mul | 2 |
| js_1x1_d8 | 3,359 |
| js_2x2_d16 | 10,153 |
| js_2x2_d32 | 17,929 |
| js_8x8_d32 | 70,357 |
| js_16x16_d32 | 140,261 |

## Apple-M2-Max

`arm64`, 12 logical cores, accelerator: Apple M2 Max (38-core GPU)

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.1 | 1.1 | +4.1% | 206 | 212 | +2.9% |
| cpu | js_1x1_d8 | 25.7 | 22.3 | -13.0% | 20634 | 17084 | -17.2% |
| cpu | js_2x2_d16 | 66.0 | 55.7 | -15.5% | 56920 | 46776 | -17.8% |
| cpu | js_2x2_d32 | 109.8 | 94.3 | -14.1% | 96610 | 81010 | -16.1% |
| cpu | js_8x8_d32 | 370.2 | 312.6 | -15.6% | 327730 | 270159 | -17.6% |
| cpu | js_16x16_d32 | 705.0 | 587.8 | -16.6% | 619032 | 503726 | -18.6% |
| metal | tiny_mul | 4.6 | 3.9 | -14.5% | 3154 | 2436 | -22.8% |
| metal | js_1x1_d8 | 20.2 | 19.6 | -3.2% | 18310 | 17620 | -3.8% |
| metal | js_2x2_d16 | 23.3 | 22.9 | -1.7% | 20816 | 20592 | -1.1% |
| metal | js_2x2_d32 | 27.4 | 27.2 | -1.0% | 24820 | 24560 | -1.0% |
| metal | js_8x8_d32 | 77.7 | 77.7 | +0.1% | 71904 | 72036 | +0.2% |
| metal | js_16x16_d32 | 131.4 | 131.4 | +0.0% | 122737 | 122812 | +0.1% |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.9 | 1.9 | +1.1% | 216 | 222 | +2.5% |
| cpu | js_1x1_d8 | 27.7 | 24.2 | -12.7% | 20850 | 17014 | -18.4% |
| cpu | js_2x2_d16 | 68.1 | 59.0 | -13.4% | 56016 | 46458 | -17.1% |
| cpu | js_2x2_d32 | 114.8 | 98.2 | -14.5% | 96294 | 80316 | -16.6% |
| cpu | js_8x8_d32 | 384.8 | 326.9 | -15.1% | 327967 | 270142 | -17.6% |
| cpu | js_16x16_d32 | 732.9 | 615.0 | -16.1% | 620419 | 504598 | -18.7% |
| metal | tiny_mul | 14.1 | 13.9 | -1.6% | 9135 | 8830 | -3.3% |
| metal | js_1x1_d8 | 34.5 | 34.1 | -1.0% | 23968 | 23790 | -0.7% |
| metal | js_2x2_d16 | 41.9 | 41.8 | -0.3% | 28775 | 28619 | -0.5% |
| metal | js_2x2_d32 | 50.2 | 50.4 | +0.4% | 33624 | 33338 | -0.9% |
| metal | js_8x8_d32 | 119.7 | 118.9 | -0.6% | 81770 | 81202 | -0.7% |
| metal | js_16x16_d32 | 198.4 | 196.0 | -1.2% | 134002 | 133046 | -0.7% |

## aws-g4dn.2xlarge-Tesla-T4

`x86_64`, 8 logical cores, accelerator: Tesla T4

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.4 | 1.4 | - | 322 | 322 | - |
| cpu | js_1x1_d8 | 71.1 | 71.1 | - | 65733 | 65733 | - |
| cpu | js_2x2_d16 | 195.1 | 195.1 | - | 180589 | 180589 | - |
| cpu | js_2x2_d32 | 330.4 | 330.4 | - | 305037 | 305037 | - |
| cpu | js_8x8_d32 | 1150.4 | 1150.4 | - | 1050006 | 1050006 | - |
| cpu | js_16x16_d32 | 2192.0 | 2192.0 | - | 1987136 | 1987136 | - |
| cuda | tiny_mul | 3.0 | 3.0 | - | 1874 | 1874 | - |
| cuda | js_1x1_d8 | 16.7 | 16.7 | - | 15278 | 15278 | - |
| cuda | js_2x2_d16 | 23.7 | 23.7 | - | 21782 | 21782 | - |
| cuda | js_2x2_d32 | 29.5 | 29.5 | - | 27138 | 27138 | - |
| cuda | js_8x8_d32 | 89.7 | 89.7 | - | 81742 | 81742 | - |
| cuda | js_16x16_d32 | 152.7 | 152.7 | - | 137266 | 137266 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 5.0 | 5.0 | - | - | - | - |
| cpu | js_1x1_d8 | 77.5 | 77.5 | - | - | - | - |
| cpu | js_2x2_d16 | 204.9 | 204.9 | - | - | - | - |
| cpu | js_2x2_d32 | 347.4 | 347.4 | - | - | - | - |
| cpu | js_8x8_d32 | 1221.4 | 1221.4 | - | - | - | - |
| cpu | js_16x16_d32 | 2301.2 | 2301.2 | - | - | - | - |
| cuda | tiny_mul | 403.8 | 403.8 | - | - | - | - |
| cuda | js_1x1_d8 | 417.8 | 417.8 | - | - | - | - |
| cuda | js_2x2_d16 | 431.5 | 431.5 | - | - | - | - |
| cuda | js_2x2_d32 | 446.4 | 446.4 | - | - | - | - |
| cuda | js_8x8_d32 | 562.8 | 562.8 | - | - | - | - |
| cuda | js_16x16_d32 | 695.6 | 695.6 | - | - | - | - |

## c7a.2xlarge

`x86_64`, 8 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 0.9 | 0.9 | - | 162 | 162 | - |
| cpu | js_1x1_d8 | 29.0 | 29.0 | - | 26312 | 26312 | - |
| cpu | js_2x2_d16 | 77.0 | 77.0 | - | 70999 | 70999 | - |
| cpu | js_2x2_d32 | 132.4 | 132.4 | - | 122202 | 122202 | - |
| cpu | js_8x8_d32 | 451.4 | 451.4 | - | 409242 | 409242 | - |
| cpu | js_16x16_d32 | 862.0 | 862.0 | - | 781816 | 781816 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.5 | 1.5 | - | 160 | 160 | - |
| cpu | js_1x1_d8 | 30.5 | 30.5 | - | 26112 | 26112 | - |
| cpu | js_2x2_d16 | 80.7 | 80.7 | - | 71537 | 71537 | - |
| cpu | js_2x2_d32 | 137.8 | 137.8 | - | 122396 | 122396 | - |
| cpu | js_8x8_d32 | 465.7 | 465.7 | - | 412272 | 412272 | - |
| cpu | js_16x16_d32 | 903.5 | 903.5 | - | 793130 | 793130 | - |

## c7a.4xlarge

`x86_64`, 16 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.0 | 1.0 | - | 168 | 168 | - |
| cpu | js_1x1_d8 | 17.7 | 17.7 | - | 14070 | 14070 | - |
| cpu | js_2x2_d16 | 45.6 | 45.6 | - | 38728 | 38728 | - |
| cpu | js_2x2_d32 | 76.7 | 76.7 | - | 65824 | 65824 | - |
| cpu | js_8x8_d32 | 253.3 | 253.3 | - | 219916 | 219916 | - |
| cpu | js_16x16_d32 | 477.6 | 477.6 | - | 417870 | 417870 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.8 | 1.8 | - | 170 | 170 | - |
| cpu | js_1x1_d8 | 20.0 | 20.0 | - | 13948 | 13948 | - |
| cpu | js_2x2_d16 | 48.9 | 48.9 | - | 37630 | 37630 | - |
| cpu | js_2x2_d32 | 81.4 | 81.4 | - | 65134 | 65134 | - |
| cpu | js_8x8_d32 | 268.1 | 268.1 | - | 217678 | 217678 | - |
| cpu | js_16x16_d32 | 505.2 | 505.2 | - | 416780 | 416780 | - |

## c7a.xlarge

`x86_64`, 4 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 0.9 | 0.9 | - | 200 | 200 | - |
| cpu | js_1x1_d8 | 55.4 | 55.4 | - | 51860 | 51860 | - |
| cpu | js_2x2_d16 | 150.9 | 150.9 | - | 139751 | 139751 | - |
| cpu | js_2x2_d32 | 252.1 | 252.1 | - | 234040 | 234040 | - |
| cpu | js_8x8_d32 | 863.4 | 863.4 | - | 793720 | 793720 | - |
| cpu | js_16x16_d32 | 1659.0 | 1659.0 | - | 1514136 | 1514136 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.5 | 1.5 | - | 212 | 212 | - |
| cpu | js_1x1_d8 | 57.2 | 57.2 | - | 51764 | 51764 | - |
| cpu | js_2x2_d16 | 152.4 | 152.4 | - | 139502 | 139502 | - |
| cpu | js_2x2_d32 | 258.8 | 258.8 | - | 236349 | 236349 | - |
| cpu | js_8x8_d32 | 885.7 | 885.7 | - | 800226 | 800226 | - |
| cpu | js_16x16_d32 | 1724.5 | 1724.5 | - | 1550288 | 1550288 | - |

## c7g.large

`arm64`, 2 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.3 | 1.3 | - | 347 | 347 | - |
| cpu | js_1x1_d8 | 145.2 | 145.2 | - | 139568 | 139568 | - |
| cpu | js_2x2_d16 | 389.8 | 389.8 | - | 369922 | 369922 | - |
| cpu | js_2x2_d32 | 670.3 | 670.3 | - | 630734 | 630734 | - |
| cpu | js_8x8_d32 | 2327.1 | 2327.1 | - | 2156953 | 2156953 | - |
| cpu | js_16x16_d32 | 4539.1 | 4539.1 | - | 4181886 | 4181886 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 2.0 | 2.0 | - | 362 | 362 | - |
| cpu | js_1x1_d8 | 147.4 | 147.4 | - | 139604 | 139604 | - |
| cpu | js_2x2_d16 | 394.2 | 394.2 | - | 369652 | 369652 | - |
| cpu | js_2x2_d32 | 677.9 | 677.9 | - | 630808 | 630808 | - |
| cpu | js_8x8_d32 | 2360.0 | 2360.0 | - | 2155946 | 2155946 | - |
| cpu | js_16x16_d32 | 4597.3 | 4597.3 | - | 4169354 | 4169354 | - |

## c7g.medium

`arm64`, 1 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.5 | 1.5 | - | 558 | 558 | - |
| cpu | js_1x1_d8 | 275.2 | 275.2 | - | 265489 | 265489 | - |
| cpu | js_2x2_d16 | 763.0 | 763.0 | - | 725824 | 725824 | - |
| cpu | js_2x2_d32 | 1293.1 | 1293.1 | - | 1216926 | 1216926 | - |
| cpu | js_8x8_d32 | 4442.1 | 4442.1 | - | 4108986 | 4108986 | - |
| cpu | js_16x16_d32 | 8541.1 | 8541.1 | - | 7839429 | 7839429 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 2.2 | 2.2 | - | 568 | 568 | - |
| cpu | js_1x1_d8 | 277.5 | 277.5 | - | 265472 | 265472 | - |
| cpu | js_2x2_d16 | 768.2 | 768.2 | - | 725798 | 725798 | - |
| cpu | js_2x2_d32 | 1301.1 | 1301.1 | - | 1216993 | 1216993 | - |
| cpu | js_8x8_d32 | 4488.1 | 4488.1 | - | 4109080 | 4109080 | - |
| cpu | js_16x16_d32 | 8643.4 | 8643.4 | - | 7854394 | 7854394 | - |

## c7g.xlarge

`arm64`, 4 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.2 | 1.2 | - | 243 | 243 | - |
| cpu | js_1x1_d8 | 71.4 | 71.4 | - | 67541 | 67541 | - |
| cpu | js_2x2_d16 | 196.4 | 196.4 | - | 185120 | 185120 | - |
| cpu | js_2x2_d32 | 334.7 | 334.7 | - | 310472 | 310472 | - |
| cpu | js_8x8_d32 | 1139.3 | 1139.3 | - | 1051279 | 1051279 | - |
| cpu | js_16x16_d32 | 2192.2 | 2192.2 | - | 2009834 | 2009834 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.9 | 1.9 | - | 246 | 246 | - |
| cpu | js_1x1_d8 | 73.5 | 73.5 | - | 67361 | 67361 | - |
| cpu | js_2x2_d16 | 199.1 | 199.1 | - | 183706 | 183706 | - |
| cpu | js_2x2_d32 | 341.8 | 341.8 | - | 313484 | 313484 | - |
| cpu | js_8x8_d32 | 1159.0 | 1159.0 | - | 1048426 | 1048426 | - |
| cpu | js_16x16_d32 | 2210.9 | 2210.9 | - | 1987066 | 1987066 | - |

## c7i.xlarge

`x86_64`, 4 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.2 | 1.2 | - | 305 | 305 | - |
| cpu | js_1x1_d8 | 85.8 | 85.8 | - | 81088 | 81088 | - |
| cpu | js_2x2_d16 | 236.2 | 236.2 | - | 221832 | 221832 | - |
| cpu | js_2x2_d32 | 399.4 | 399.4 | - | 373022 | 373022 | - |
| cpu | js_8x8_d32 | 1385.3 | 1385.3 | - | 1272988 | 1272988 | - |
| cpu | js_16x16_d32 | 2649.4 | 2649.4 | - | 2413412 | 2413412 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.7 | 1.7 | - | 270 | 270 | - |
| cpu | js_1x1_d8 | 87.8 | 87.8 | - | 81703 | 81703 | - |
| cpu | js_2x2_d16 | 239.5 | 239.5 | - | 222278 | 222278 | - |
| cpu | js_2x2_d32 | 403.5 | 403.5 | - | 372153 | 372153 | - |
| cpu | js_8x8_d32 | 1394.7 | 1394.7 | - | 1268568 | 1268568 | - |
| cpu | js_16x16_d32 | 2690.0 | 2690.0 | - | 2416108 | 2416108 | - |

## c8g.4xlarge

`arm64`, 16 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.0 | 1.0 | - | 162 | 162 | - |
| cpu | js_1x1_d8 | 17.9 | 17.9 | - | 15510 | 15510 | - |
| cpu | js_2x2_d16 | 46.6 | 46.6 | - | 41669 | 41669 | - |
| cpu | js_2x2_d32 | 80.5 | 80.5 | - | 73116 | 73116 | - |
| cpu | js_8x8_d32 | 262.0 | 262.0 | - | 237334 | 237334 | - |
| cpu | js_16x16_d32 | 504.2 | 504.2 | - | 456016 | 456016 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.8 | 1.8 | - | 160 | 160 | - |
| cpu | js_1x1_d8 | 19.9 | 19.9 | - | 15386 | 15386 | - |
| cpu | js_2x2_d16 | 50.7 | 50.7 | - | 41809 | 41809 | - |
| cpu | js_2x2_d32 | 85.2 | 85.2 | - | 72175 | 72175 | - |
| cpu | js_8x8_d32 | 277.6 | 277.6 | - | 236842 | 236842 | - |
| cpu | js_16x16_d32 | 538.7 | 538.7 | - | 459144 | 459144 | - |

## c8g.large

`arm64`, 2 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.1 | 1.1 | - | 293 | 293 | - |
| cpu | js_1x1_d8 | 121.2 | 121.2 | - | 116350 | 116350 | - |
| cpu | js_2x2_d16 | 331.8 | 331.8 | - | 314494 | 314494 | - |
| cpu | js_2x2_d32 | 572.5 | 572.5 | - | 538714 | 538714 | - |
| cpu | js_8x8_d32 | 1979.1 | 1979.1 | - | 1833608 | 1833608 | - |
| cpu | js_16x16_d32 | 3724.4 | 3724.4 | - | 3417766 | 3417766 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.6 | 1.6 | - | 300 | 300 | - |
| cpu | js_1x1_d8 | 122.9 | 122.9 | - | 116353 | 116353 | - |
| cpu | js_2x2_d16 | 335.2 | 335.2 | - | 314428 | 314428 | - |
| cpu | js_2x2_d32 | 578.6 | 578.6 | - | 538745 | 538745 | - |
| cpu | js_8x8_d32 | 1999.0 | 1999.0 | - | 1834192 | 1834192 | - |
| cpu | js_16x16_d32 | 3765.4 | 3765.4 | - | 3419592 | 3419592 | - |

## c8g.xlarge

`arm64`, 4 logical cores, accelerator: none

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.0 | 1.0 | - | 202 | 202 | - |
| cpu | js_1x1_d8 | 61.8 | 61.8 | - | 58351 | 58351 | - |
| cpu | js_2x2_d16 | 166.3 | 166.3 | - | 156138 | 156138 | - |
| cpu | js_2x2_d32 | 279.2 | 279.2 | - | 260649 | 260649 | - |
| cpu | js_8x8_d32 | 959.7 | 959.7 | - | 881924 | 881924 | - |
| cpu | js_16x16_d32 | 1826.8 | 1826.8 | - | 1667834 | 1667834 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.6 | 1.6 | - | 232 | 232 | - |
| cpu | js_1x1_d8 | 62.3 | 62.3 | - | 57214 | 57214 | - |
| cpu | js_2x2_d16 | 169.8 | 169.8 | - | 155742 | 155742 | - |
| cpu | js_2x2_d32 | 286.7 | 286.7 | - | 262298 | 262298 | - |
| cpu | js_8x8_d32 | 979.6 | 979.6 | - | 884241 | 884241 | - |
| cpu | js_16x16_d32 | 1856.1 | 1856.1 | - | 1657764 | 1657764 | - |

## g4dn.2xlarge

`x86_64`, 8 logical cores, accelerator: Tesla T4

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.4 | 1.4 | - | 302 | 302 | - |
| cpu | js_1x1_d8 | 68.2 | 68.2 | - | 63280 | 63280 | - |
| cpu | js_2x2_d16 | 185.2 | 185.2 | - | 171954 | 171954 | - |
| cpu | js_2x2_d32 | 313.5 | 313.5 | - | 289354 | 289354 | - |
| cpu | js_8x8_d32 | 1094.1 | 1094.1 | - | 998354 | 998354 | - |
| cpu | js_16x16_d32 | 2073.5 | 2073.5 | - | 1875438 | 1875438 | - |
| cuda | tiny_mul | 2.9 | 2.9 | - | 1840 | 1840 | - |
| cuda | js_1x1_d8 | 16.6 | 16.6 | - | 15173 | 15173 | - |
| cuda | js_2x2_d16 | 23.4 | 23.4 | - | 21634 | 21634 | - |
| cuda | js_2x2_d32 | 29.3 | 29.3 | - | 26998 | 26998 | - |
| cuda | js_8x8_d32 | 89.1 | 89.1 | - | 81290 | 81290 | - |
| cuda | js_16x16_d32 | 151.8 | 151.8 | - | 136328 | 136328 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 2.3 | 2.3 | - | 308 | 308 | - |
| cpu | js_1x1_d8 | 70.3 | 70.3 | - | 62964 | 62964 | - |
| cpu | js_2x2_d16 | 190.8 | 190.8 | - | 173108 | 173108 | - |
| cpu | js_2x2_d32 | 322.6 | 322.6 | - | 291989 | 291989 | - |
| cpu | js_8x8_d32 | 1109.5 | 1109.5 | - | 990872 | 990872 | - |
| cpu | js_16x16_d32 | 2109.8 | 2109.8 | - | 1870674 | 1870674 | - |
| cuda | tiny_mul | 195.2 | 195.2 | - | 1974 | 1974 | - |
| cuda | js_1x1_d8 | 213.0 | 213.0 | - | 15246 | 15246 | - |
| cuda | js_2x2_d16 | 224.1 | 224.1 | - | 21198 | 21198 | - |
| cuda | js_2x2_d32 | 234.4 | 234.4 | - | 26725 | 26725 | - |
| cuda | js_8x8_d32 | 357.9 | 357.9 | - | 80756 | 80756 | - |
| cuda | js_16x16_d32 | 473.2 | 473.2 | - | 136347 | 136347 | - |

## g4dn.xlarge

`x86_64`, 4 logical cores, accelerator: Tesla T4

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.5 | 1.5 | - | 402 | 402 | - |
| cpu | js_1x1_d8 | 137.0 | 137.0 | - | 130177 | 130177 | - |
| cpu | js_2x2_d16 | 376.0 | 376.0 | - | 354672 | 354672 | - |
| cpu | js_2x2_d32 | 640.5 | 640.5 | - | 598350 | 598350 | - |
| cpu | js_8x8_d32 | 2228.4 | 2228.4 | - | 2052662 | 2052662 | - |
| cpu | js_16x16_d32 | 4343.1 | 4343.1 | - | 3963317 | 3963317 | - |
| cuda | tiny_mul | 3.0 | 3.0 | - | 1858 | 1858 | - |
| cuda | js_1x1_d8 | 16.7 | 16.7 | - | 15250 | 15250 | - |
| cuda | js_2x2_d16 | 23.7 | 23.7 | - | 21764 | 21764 | - |
| cuda | js_2x2_d32 | 29.5 | 29.5 | - | 27012 | 27012 | - |
| cuda | js_8x8_d32 | 89.6 | 89.6 | - | 81579 | 81579 | - |
| cuda | js_16x16_d32 | 152.4 | 152.4 | - | 136624 | 136624 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 2.3 | 2.3 | - | 408 | 408 | - |
| cpu | js_1x1_d8 | 140.1 | 140.1 | - | 130795 | 130795 | - |
| cpu | js_2x2_d16 | 379.1 | 379.1 | - | 352662 | 352662 | - |
| cpu | js_2x2_d32 | 649.0 | 649.0 | - | 599555 | 599555 | - |
| cpu | js_8x8_d32 | 2272.9 | 2272.9 | - | 2073219 | 2073219 | - |
| cpu | js_16x16_d32 | 4373.3 | 4373.3 | - | 3946659 | 3946659 | - |
| cuda | tiny_mul | 203.0 | 203.0 | - | 1988 | 1988 | - |
| cuda | js_1x1_d8 | 220.4 | 220.4 | - | 15306 | 15306 | - |
| cuda | js_2x2_d16 | 233.8 | 233.8 | - | 21287 | 21287 | - |
| cuda | js_2x2_d32 | 247.3 | 247.3 | - | 26858 | 26858 | - |
| cuda | js_8x8_d32 | 369.9 | 369.9 | - | 81058 | 81058 | - |
| cuda | js_16x16_d32 | 489.7 | 489.7 | - | 136538 | 136538 | - |

## g5.xlarge

`x86_64`, 4 logical cores, accelerator: NVIDIA A10G

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.3 | 1.3 | - | 353 | 353 | - |
| cpu | js_1x1_d8 | 103.9 | 103.9 | - | 98118 | 98118 | - |
| cpu | js_2x2_d16 | 282.5 | 282.5 | - | 267018 | 267018 | - |
| cpu | js_2x2_d32 | 480.1 | 480.1 | - | 450893 | 450893 | - |
| cpu | js_8x8_d32 | 1627.2 | 1627.2 | - | 1505288 | 1505288 | - |
| cpu | js_16x16_d32 | 3091.6 | 3091.6 | - | 2834576 | 2834576 | - |
| cuda | tiny_mul | 2.7 | 2.7 | - | 1660 | 1660 | - |
| cuda | js_1x1_d8 | 14.5 | 14.5 | - | 13179 | 13179 | - |
| cuda | js_2x2_d16 | 20.3 | 20.3 | - | 18846 | 18846 | - |
| cuda | js_2x2_d32 | 23.0 | 23.0 | - | 21366 | 21366 | - |
| cuda | js_8x8_d32 | 55.7 | 55.7 | - | 50184 | 50184 | - |
| cuda | js_16x16_d32 | 86.1 | 86.1 | - | 75244 | 75244 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 2.2 | 2.2 | - | 367 | 367 | - |
| cpu | js_1x1_d8 | 106.1 | 106.1 | - | 98424 | 98424 | - |
| cpu | js_2x2_d16 | 286.9 | 286.9 | - | 266826 | 266826 | - |
| cpu | js_2x2_d32 | 487.5 | 487.5 | - | 451653 | 451653 | - |
| cpu | js_8x8_d32 | 1648.3 | 1648.3 | - | 1506752 | 1506752 | - |
| cpu | js_16x16_d32 | 3117.9 | 3117.9 | - | 2821854 | 2821854 | - |
| cuda | tiny_mul | 169.4 | 169.4 | - | 1937 | 1937 | - |
| cuda | js_1x1_d8 | 184.6 | 184.6 | - | 13212 | 13212 | - |
| cuda | js_2x2_d16 | 191.6 | 191.6 | - | 18202 | 18202 | - |
| cuda | js_2x2_d32 | 197.4 | 197.4 | - | 20824 | 20824 | - |
| cuda | js_8x8_d32 | 286.6 | 286.6 | - | 50182 | 50182 | - |
| cuda | js_16x16_d32 | 360.9 | 360.9 | - | 75176 | 75176 | - |

## g5g.xlarge

`arm64`, 4 logical cores, accelerator: NVIDIA T4G

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 3.0 | 3.0 | - | 624 | 624 | - |
| cpu | js_1x1_d8 | 190.5 | 190.5 | - | 180766 | 180766 | - |
| cpu | js_2x2_d16 | 524.9 | 524.9 | - | 494427 | 494427 | - |
| cpu | js_2x2_d32 | 909.4 | 909.4 | - | 846954 | 846954 | - |
| cpu | js_8x8_d32 | 3093.5 | 3093.5 | - | 2838116 | 2838116 | - |
| cpu | js_16x16_d32 | 5845.4 | 5845.4 | - | 5314712 | 5314712 | - |
| cuda | tiny_mul | 4.6 | 4.6 | - | 2100 | 2100 | - |
| cuda | js_1x1_d8 | 19.2 | 19.2 | - | 16359 | 16359 | - |
| cuda | js_2x2_d16 | 25.9 | 25.9 | - | 22689 | 22689 | - |
| cuda | js_2x2_d32 | 32.1 | 32.1 | - | 28482 | 28482 | - |
| cuda | js_8x8_d32 | 96.7 | 96.7 | - | 87492 | 87492 | - |
| cuda | js_16x16_d32 | 167.1 | 167.1 | - | 150338 | 150338 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 4.7 | 4.7 | - | 631 | 631 | - |
| cpu | js_1x1_d8 | 195.9 | 195.9 | - | 180567 | 180567 | - |
| cpu | js_2x2_d16 | 531.5 | 531.5 | - | 490834 | 490834 | - |
| cpu | js_2x2_d32 | 914.7 | 914.7 | - | 836386 | 836386 | - |
| cpu | js_8x8_d32 | 3123.2 | 3123.2 | - | 2822500 | 2822500 | - |
| cpu | js_16x16_d32 | 6151.9 | 6151.9 | - | 5520459 | 5520459 | - |
| cuda | tiny_mul | 248.1 | 248.1 | - | 2228 | 2228 | - |
| cuda | js_1x1_d8 | 264.0 | 264.0 | - | 16382 | 16382 | - |
| cuda | js_2x2_d16 | 279.0 | 279.0 | - | 22524 | 22524 | - |
| cuda | js_2x2_d32 | 292.0 | 292.0 | - | 28372 | 28372 | - |
| cuda | js_8x8_d32 | 430.7 | 430.7 | - | 87392 | 87392 | - |
| cuda | js_16x16_d32 | 572.7 | 572.7 | - | 151198 | 151198 | - |

## g6.xlarge

`x86_64`, 4 logical cores, accelerator: NVIDIA L4

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.1 | 1.1 | - | 316 | 316 | - |
| cpu | js_1x1_d8 | 90.9 | 90.9 | - | 86171 | 86171 | - |
| cpu | js_2x2_d16 | 247.4 | 247.4 | - | 233063 | 233063 | - |
| cpu | js_2x2_d32 | 426.5 | 426.5 | - | 399496 | 399496 | - |
| cpu | js_8x8_d32 | 1434.1 | 1434.1 | - | 1323929 | 1323929 | - |
| cpu | js_16x16_d32 | 2725.4 | 2725.4 | - | 2494696 | 2494696 | - |
| cuda | tiny_mul | 2.0 | 2.0 | - | 1181 | 1181 | - |
| cuda | js_1x1_d8 | 11.5 | 11.5 | - | 10439 | 10439 | - |
| cuda | js_2x2_d16 | 16.9 | 16.9 | - | 15646 | 15646 | - |
| cuda | js_2x2_d32 | 20.0 | 20.0 | - | 18446 | 18446 | - |
| cuda | js_8x8_d32 | 50.3 | 50.3 | - | 45242 | 45242 | - |
| cuda | js_16x16_d32 | 77.3 | 77.3 | - | 67712 | 67712 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.8 | 1.8 | - | 311 | 311 | - |
| cpu | js_1x1_d8 | 92.5 | 92.5 | - | 85804 | 85804 | - |
| cpu | js_2x2_d16 | 250.9 | 250.9 | - | 233418 | 233418 | - |
| cpu | js_2x2_d32 | 428.7 | 428.7 | - | 396562 | 396562 | - |
| cpu | js_8x8_d32 | 1455.2 | 1455.2 | - | 1328590 | 1328590 | - |
| cpu | js_16x16_d32 | 2762.7 | 2762.7 | - | 2495486 | 2495486 | - |
| cuda | tiny_mul | 151.0 | 151.0 | - | 1219 | 1219 | - |
| cuda | js_1x1_d8 | 163.6 | 163.6 | - | 10460 | 10460 | - |
| cuda | js_2x2_d16 | 173.2 | 173.2 | - | 15221 | 15221 | - |
| cuda | js_2x2_d32 | 176.9 | 176.9 | - | 18006 | 18006 | - |
| cuda | js_8x8_d32 | 256.8 | 256.8 | - | 45276 | 45276 | - |
| cuda | js_16x16_d32 | 320.8 | 320.8 | - | 67775 | 67775 | - |

## g6e.xlarge

`x86_64`, 4 logical cores, accelerator: NVIDIA L40S

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.1 | 1.1 | - | 310 | 310 | - |
| cpu | js_1x1_d8 | 90.6 | 90.6 | - | 85698 | 85698 | - |
| cpu | js_2x2_d16 | 247.9 | 247.9 | - | 233803 | 233803 | - |
| cpu | js_2x2_d32 | 421.5 | 421.5 | - | 394416 | 394416 | - |
| cpu | js_8x8_d32 | 1431.0 | 1431.0 | - | 1320374 | 1320374 | - |
| cpu | js_16x16_d32 | 2699.3 | 2699.3 | - | 2470075 | 2470075 | - |
| cuda | tiny_mul | 2.1 | 2.1 | - | 1210 | 1210 | - |
| cuda | js_1x1_d8 | 11.9 | 11.9 | - | 10858 | 10858 | - |
| cuda | js_2x2_d16 | 16.9 | 16.9 | - | 15650 | 15650 | - |
| cuda | js_2x2_d32 | 18.8 | 18.8 | - | 17502 | 17502 | - |
| cuda | js_8x8_d32 | 44.5 | 44.5 | - | 40069 | 40069 | - |
| cuda | js_16x16_d32 | 66.0 | 66.0 | - | 57557 | 57557 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.8 | 1.8 | - | 305 | 305 | - |
| cpu | js_1x1_d8 | 92.5 | 92.5 | - | 85712 | 85712 | - |
| cpu | js_2x2_d16 | 250.9 | 250.9 | - | 233164 | 233164 | - |
| cpu | js_2x2_d32 | 427.5 | 427.5 | - | 395154 | 395154 | - |
| cpu | js_8x8_d32 | 1452.6 | 1452.6 | - | 1325088 | 1325088 | - |
| cpu | js_16x16_d32 | 2759.8 | 2759.8 | - | 2488590 | 2488590 | - |
| cuda | tiny_mul | 186.2 | 186.2 | - | 1234 | 1234 | - |
| cuda | js_1x1_d8 | 203.2 | 203.2 | - | 10874 | 10874 | - |
| cuda | js_2x2_d16 | 204.7 | 204.7 | - | 15101 | 15101 | - |
| cuda | js_2x2_d32 | 217.4 | 217.4 | - | 17069 | 17069 | - |
| cuda | js_8x8_d32 | 268.1 | 268.1 | - | 40100 | 40100 | - |
| cuda | js_16x16_d32 | 322.3 | 322.3 | - | 57532 | 57532 | - |

## local-m2-max

`arm64`, 12 logical cores, accelerator: Apple M2 Max 38-core GPU

### warm

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.0 | 1.0 | - | 208 | 208 | - |
| cpu | js_1x1_d8 | 23.0 | 23.0 | - | 18269 | 18269 | - |
| cpu | js_2x2_d16 | 56.7 | 56.7 | - | 48986 | 48986 | - |
| cpu | js_2x2_d32 | 97.0 | 97.0 | - | 84859 | 84859 | - |
| cpu | js_8x8_d32 | 324.9 | 324.9 | - | 287138 | 287138 | - |
| cpu | js_16x16_d32 | 609.2 | 609.2 | - | 531270 | 531270 | - |
| metal | tiny_mul | 3.4 | 3.4 | - | 2098 | 2098 | - |
| metal | js_1x1_d8 | 18.9 | 18.9 | - | 17156 | 17156 | - |
| metal | js_2x2_d16 | 22.3 | 22.3 | - | 20362 | 20362 | - |
| metal | js_2x2_d32 | 27.6 | 27.6 | - | 24852 | 24852 | - |
| metal | js_8x8_d32 | 73.7 | 73.7 | - | 68810 | 68810 | - |
| metal | js_16x16_d32 | 126.6 | 126.6 | - | 118758 | 118758 | - |

### cold

| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | tiny_mul | 1.6 | 1.6 | - | 205 | 205 | - |
| cpu | js_1x1_d8 | 24.6 | 24.6 | - | 18300 | 18300 | - |
| cpu | js_2x2_d16 | 59.7 | 59.7 | - | 49204 | 49204 | - |
| cpu | js_2x2_d32 | 98.3 | 98.3 | - | 82041 | 82041 | - |
| cpu | js_8x8_d32 | 333.3 | 333.3 | - | 283338 | 283338 | - |
| cpu | js_16x16_d32 | 631.0 | 631.0 | - | 532254 | 532254 | - |
| metal | tiny_mul | 11.7 | 11.7 | - | 7894 | 7894 | - |
| metal | js_1x1_d8 | 32.4 | 32.4 | - | 23284 | 23284 | - |
| metal | js_2x2_d16 | 37.7 | 37.7 | - | 27361 | 27361 | - |
| metal | js_2x2_d32 | 45.4 | 45.4 | - | 32003 | 32003 | - |
| metal | js_8x8_d32 | 111.5 | 111.5 | - | 79206 | 79206 | - |
| metal | js_16x16_d32 | 186.7 | 186.7 | - | 131002 | 131002 | - |

## Measurement log

| label | machines | rows |
|---|---|---:|
| baseline | Apple-M2-Max, aws-g4dn.2xlarge-Tesla-T4, c7a.2xlarge, c7a.4xlarge +15 more | 336 |
| round3-before | Apple-M2-Max | 24 |
| round3-after | Apple-M2-Max | 24 |

