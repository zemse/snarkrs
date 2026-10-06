# CPU compact twiddles

2026-10-06, Apple M2 Max, 12 Rayon threads. Compared with `bfb9797`, CPU-only
release builds, Cargo `--locked`, default compiler settings. No transpose path,
new threshold, dependency or lockfile change.

The fused 2048-element blocks read a strided subset of the full twiddle table.
Packing that subset gives a 32 KiB table per direction, cached by its sampled
root. The arithmetic and outer radix-4 passes are unchanged.

## Method

Nine fixed existing circuits, serial baseline/candidate/baseline controls per
workload. Each configuration used 15 warm proofs after two verified warmups and
three cold proofs. Warm times cover the complete resident proving call; cold
includes key/witness parsing, preparation and proving, not OS page-cache eviction.
Verification is outside the timed region and precedes recording each result.

Every proof verified locally. The first and last of every configuration also
verified with snarkjs, with public outputs compared numerically against the
fixture reference. Both binaries had identical temporary measurement changes;
those changes are not part of the adoption diff. No build or test from this study
ran during measurements.

Another session compiled during the original p23 controls. That triple was
excluded from the decision and repeated in full. Tables below use the fresh p23
triple. Normal desktop background work remained; this was not an isolated host.

The first pass showed small-case regressions. Tiny multiplication, SHA-256 and
joinsplit p17 were each repeated in two further baseline/candidate/baseline rounds,
with 25 warm and 15 cold proofs per configuration. All 96 CSV configurations were
validated for exact counts, repetition sequences, finite timings and verification.
In total, 1260 measured proofs verified locally and 192 first/last proofs passed
the independent verifier and reference-public check. Raw operational logs remain
outside version control.

## Full-proof medians

Milliseconds. A and C are the bracketing baseline medians; B is compact.

| Circuit | Domain | Warm A | Warm B | Warm C | Cold A | Cold B | Cold C |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_mul | 2^3 | 0.622 | 0.610 | 0.613 | 1.344 | 1.511 | 1.378 |
| sha256 | 2^16 | 42.005 | 42.817 | 42.219 | 50.716 | 51.764 | 53.341 |
| tornado | 2^15 | 98.913 | 96.913 | 96.673 | 100.561 | 99.349 | 102.385 |
| keccak256 | 2^18 | 158.758 | 155.643 | 157.327 | 181.062 | 175.475 | 182.951 |
| js_8x8_d32 | 2^17 | 205.751 | 218.327 | 211.516 | 221.649 | 225.303 | 218.972 |
| onion_p21 | 2^21 | 2541.953 | 2448.571 | 2560.624 | 2731.883 | 2568.635 | 2753.556 |
| onion_p22 | 2^22 | 5243.198 | 4955.939 | 5445.625 | 5387.925 | 5191.931 | 5462.263 |
| onion_p23, fresh | 2^23 | 10747.605 | 9838.268 | 10398.079 | 11179.433 | 10467.757 | 10648.418 |
| onion_p24 | 2^24 | 21347.662 | 19875.422 | 20798.848 | 21617.617 | 21115.061 | 22084.858 |

| Circuit | Warm gain vs A and C | Absolute warm baseline drift | Cold gain vs A and C | Absolute cold baseline drift |
|---|---:|---:|---:|---:|
| onion_p21 | 3.67% to 4.38% | 0.73% | 5.98% to 6.72% | 0.79% |
| onion_p22 | 5.48% to 8.99% | 3.86% | 3.64% to 4.95% | 1.38% |
| onion_p23, fresh | 5.38% to 8.46% | 3.25% | 1.70% to 6.37% | 4.75% |
| onion_p24 | 4.44% to 6.90% | 2.57% | 2.32% to 4.39% | 2.16% |

Observed warm whole-proof ranges, minimum to maximum, in milliseconds. The ranges
overlap; median differences are not statistical confidence intervals.

| Circuit | Baseline A range | Compact range | Baseline C range |
|---|---:|---:|---:|
| onion_p21 | 2478.671 to 3180.833 | 2385.892 to 2653.451 | 2514.479 to 2689.411 |
| onion_p22 | 5051.159 to 5765.992 | 4831.059 to 5291.701 | 5091.155 to 6604.333 |
| onion_p23, fresh | 10149.515 to 13658.811 | 9604.357 to 10831.374 | 10080.497 to 10690.632 |
| onion_p24 | 20293.108 to 27515.526 | 19584.293 to 21588.968 | 20183.463 to 22093.374 |

## Concurrent NTT window

Warm median milliseconds from the real prover's three concurrent iNTT/coset/NTT
pipelines, including their table preparation and synchronization. These are not
single-vector microbenchmarks or sums of overlapping vector timings.

| Circuit | Baseline A | Compact | Baseline C |
|---|---:|---:|---:|
| onion_p21 | 472.402 | 383.605 | 498.175 |
| onion_p22 | 1100.982 | 869.171 | 1173.531 |
| onion_p23, fresh | 2288.505 | 1858.344 | 2353.994 |
| onion_p24 | 4893.152 | 4114.708 | 4816.875 |

## Small-case rechecks

Full-proof median gain relative to both bracketing controls, ordered from lower to
higher. Negative means slower. Baseline drift is absolute, warm / cold.

| Circuit | Round | Warm gain | Cold gain | Baseline drift |
|---|---:|---:|---:|---:|
| tiny_mul | 1 | 0.98% to 1.94% | 0.87% to 2.25% | 0.98% / 1.40% |
| tiny_mul | 2 | -0.17% to 3.04% | -0.71% to 1.24% | 3.21% / 1.93% |
| sha256 | 1 | -0.19% to 1.43% | 1.75% to 4.15% | 1.64% / 2.51% |
| sha256 | 2 | -1.10% to 1.88% | -0.12% to 0.73% | 2.95% / 0.86% |
| js_8x8_d32 | 1 | -1.88% to -0.31% | 1.81% to 4.78% | 1.56% / 3.13% |
| js_8x8_d32 | 2 | 0.79% to 4.99% | 1.39% to 8.54% | 4.42% / 7.83% |

The initial 9.65% to 12.43% tiny-cold regression did not recur. Joinsplit's initial
3.22% to 6.11% warm regression narrowed to 0.31% to 1.88% in the first recheck and
reversed in the second. SHA-256 remained near break-even. These repeats do not
support a reproducible material regression, or a small-domain speedup claim.

## Correctness and decision

Passed 19 NTT tests, including sampled-root cache sharing, both directions and
odd-powered roots; 21 strict Groth16 unit tests across six fixtures; saved snarkjs
proof acceptance on three available fixtures; H and five-MSM references on the
available tiny fixture; exact ffjavascript vectors at 2^13, 2^14 and 2^16; CPU CLI
roundtrips across six fixtures. Missing saved reference files were explicitly
reported, not counted as additional oracle coverage. Domain validation, changed
roots, concurrent cold caches and edge-size tests remain intact.

Adopt the compact-only candidate for independent verification before merging.
Warm end-to-end gains at p21 through p24 exceed both baseline controls and each
workload's baseline drift. The repeated small-case checks did not reproduce the
initial material regressions. No threshold is introduced. Cold p23/p24 gains are
less clearly separated from drift and should not be presented as certain wins.
These conclusions are specific to this CPU, thread count and workload set; no new
theoretical-floor or cross-machine performance claim is made.
