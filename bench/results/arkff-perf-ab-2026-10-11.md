# Stock vs performance-patched arkworks 0.5 field code: A/B, 2026-10-11

Two builds of snarkrs from one source tree, one toolchain and one set of flags, differing only in
which checkout of arkworks-rs/algebra the `[patch.crates-io]` block points at
(`bench/scripts/ark-patch.sh`, built by `bench/scripts/arkab-build.sh`):

- **stock**: tag `v0.5.0` (7ad88c4). Every crate's `src` was diffed against the crates.io 0.5.0
  package (ark-ff, -macros, -asm, -ec, -poly, -serialize, -serialize-derive, ark-bn254) and is
  identical, so this is what the prover ships with today.
- **perf**: branch `perf/bn254-field-0.5` (651414b), `v0.5.0` plus the three commits of
  `perf/bn254-field` cherry-picked (30fb421 row-shaped mont mul/square carry chains, e2139df
  branch-free add/sub/double/neg plus `adc`/`sbb` fallbacks, 651414b `sum_of_products`). Only the
  CHANGELOG hunks conflicted; they were dropped. No code conflicts; the 0.6 to 0.5 context
  differences in those files are cosmetic.

ark-ff, ark-ff-macros, ark-ff-asm, ark-ec, ark-poly, ark-serialize and ark-serialize-derive come
from the checkout; `cargo tree -d -e normal` shows no duplicate ark crate on either target.
ark-bn254 stays on crates.io in both builds: the tag's `curves/bn254` is versioned
`0.5.0-alpha.0`, which cargo will not substitute for `0.5`, and its `src` is byte-identical to
crates.io 0.5.0. Its `Fq`/`Fr` arithmetic is expanded by the patched ark-ff-macros, so the patch
still reaches it (the field timings below confirm it). ark-std is not in the algebra repo.

Tests on both checkouts, release mode: ark-ff 22 + 42, ark-test-curves 249 (perf only),
ark-bn254 103, all pass, the same counts on stock. Running them at the tag needs two temporary,
uncommitted manifest edits that apply to stock and perf alike: ark-ff's `[dev-dependencies]` are
commented out at `v0.5.0`, and `curves/` patches ark-relations/r1cs-std/std to git HEAD, which no
longer builds against 0.5. Also `--cap-lints warn`, because rustc 1.99's check-cfg denies the
derive's `cfg(feature = "asm")` under ark-bn254's `#![deny(warnings)]`.

Runner `bench/scripts/arkab.py`, report `bench/scripts/arkab_report.py`. Every step runs under
both builds back to back, the build order reversing each round (AB, BA, AB), 3 rounds. Prove is
`snarkrs bench --mode warm`, 10 reps per round, every rep self-verified. Verify is
`verify_time` (`crates/groth16/examples/verify_time.rs`), which checks the proof verifies and that
a tampered public signal is rejected, then times 500 in-process `verify` calls per round. Each
ceremony step is the wall clock of one CLI process and counts only on exit 0 and, for verifiers,
their OK line. Prove cells are the median of all 30 reps; the other cells are the median of the 3
round medians. Change is perf against stock.

## Summary

Median change across circuits (or sizes); range in brackets. x86 and CUDA were not run (see
"Not run").

| workflow | Mac CPU | Mac Metal | x86 CPU (c7i) | CUDA |
|---|---|---|---|---|
| prove, warm, 9 circuits | **-3.4%** (-1.3 to -8.2) | +0.2% (-0.5 to +1.3) | not run | not run |
| verify, 9 circuits | **-29.6%** (-24.7 to -30.9) | CPU only, = Mac CPU | not run | not run |
| ptau contribute, 2^13 and 2^19 | **-22.1%** (-21.4, -22.8) | -0.5% (-0.2, -0.8) | not run | not run |
| ptau verify, 2^13 and 2^19 | -5.5% (-10.5, -0.4) | CPU only, = Mac CPU | not run | not run |
| ptau prepare phase2, 2^13 and 2^18 | +0.1% (+0.3, -0.2) | -0.8% (+0.4, -2.1) | not run | not run |
| groth16 setup (phase 2), 9 circuits | **-17.5%** (-5.9 to -18.9) | **-13.4%** (+1.4 to -17.9) | not run | not run |
| zkey contribute, 9 circuits | **-20.3%** (-10.4 to -22.6) | +1.1% (-2.0 to +5.3) | not run | not run |
| zkey verify init, 9 circuits | -1.7% (-3.0 to +0.7) | CPU only, = Mac CPU | not run | not run |

Per-circuit highlights, Mac CPU: anon-aadhaar prove 1331.5 to 1222.8 ms (-8.2%), keccak256 and
sha256 -7.3%, rsa2048 -6.6%; the 2^15 circuits -1.3% to -3.4%. Verify 1.41 to 0.99 ms on
anon-aadhaar. `ptau contribute` at 2^19 38.0 to 29.3 s. `zkey contribute` on anon-aadhaar 29.1 to
23.2 s, `groth16 setup` on anon-aadhaar 25.3 to 21.6 s.

## Findings

- **The expectation holds, with one correction: the gain is wherever arkworks types do the
  arithmetic, and that includes the CPU NTT, but not `ptau prepare`.**
  - Prove on CPU: the NTT is 24 to 27% faster on every circuit (anon-aadhaar 408.6 to 306.0 ms,
    the 2^18 circuits about 43 to 32 ms) and pointwise 10 to 31% faster; gather is mixed and
    under 2 ms except on anon-aadhaar (10.0 to 8.2 ms).
    The MSM, our own field code (`crates/field/src/raw.rs`), moves by under 1.5% either way.
    So the end-to-end gain scales with the NTT's share of the proof: 8% on anon-aadhaar and the
    2^18 circuits with small MSMs (rsa2048, keccak256), 1 to 3% where the MSM dominates.
  - Prove on Metal: no change (-0.5% to +1.3%). Field math is in the kernels; the host part
    that arkworks touches is under 1% of a Metal proof.
  - Verify -25 to -31%: Miller loop and final exponentiation are all Fq/Fq2/Fq12 arithmetic.
    The circuits with more public inputs (js_16x16_d32, 34) gain least because the
    `aggregate_public` double-and-add is a larger share.
  - Ceremony on CPU: `ptau contribute` -22%, `zkey contribute` -20%, `groth16 setup` -18%. These
    are arkworks point scalar multiplications and field batches.
  - `ptau prepare` does not move on CPU or Metal (within ±0.3% on CPU). Its point FFT runs on our
    own code. `ptau verify` at 2^19 and `zkey verify init` move under 2%: their time is in our MSM and
    file I/O, and only the small pairing part (2^13 ptau verify -10.5%) is arkworks.
  - On Metal, `ptau contribute` and `zkey contribute` run on the GPU and do not move. `groth16
    setup` still gains 13%: part of setup stays on the host in arkworks code.
- **Field primitives, single threaded (`field_ops`, M2 Max, two passes each, ns):** Fq mul
  latency 14.0 to 13.9, throughput 11.9 to 11.1; Fr mul throughput 12.1 to 11.5; Fq2 mul 51.1 to
  39.9 (-22%); G1 mixed add 218.5 to 176.2 (-19%). **Fr add latency got worse, 1.20 to 2.19 ns**
  (throughput 1.04 to 1.87). That bench is a dependent chain whose compare branch the predictor
  learns perfectly, the case the branch-free select gives up. The NTT and pointwise gains say it
  does not cost the prover anything here, but it is a real regression in that microbenchmark and
  worth a look upstream.
- The perf build is never slower on a workflow beyond noise. The only positive cells are Metal
  rows and CPU `ptau prepare`, all within ±1.5% except Metal `zkey contribute` on js_2x2_d32
  (+5.3%, a 93 ms step with one 110 ms round) and Metal setup on rsa2048 (+1.4%, one 4.4 s
  round).

## Correctness

- Every timed proof self-verified: 540 warm proofs per backend (9 circuits x 10 reps x 3 rounds x
  2 builds), 1080 in all. The runner recorded 0 failures in either run.
- `verify_time` rejected a tampered public signal for every circuit and build before timing, and
  asserts each of its 500 x 3 x 9 x 2 calls per run returned OK.
- Every ceremony output checked by our own verifier was accepted: `ptau verify` on every
  contribute output (2 sizes x 3 rounds x 2 builds), `zkey verify init` on every contributed key
  (9 circuits x 3 rounds x 2 builds).
- snarkjs 0.7.6: **CPU run 26 of 26 accepted** (18 proofs, both builds' contributed 2^13 ptau and
  prepared 2^13 ptau, both builds' contributed js_2x2_d32 and js_16x16_d32 zkeys via `zkey verify`
  against r1cs and ptau). **Metal run 24 of 24** (the same, with js_2x2_d32's zkey only).
- **Byte-identical outputs: 23 of 23 deterministic outputs agree across builds and rounds, per
  backend (90 files hashed per run):** `groth16 setup` for all 9 circuits (every round), `ptau
  prepare` at 2^13 and 2^18 (every round), `ptau beacon` at 2^13 and 2^19, `zkey beacon` for all 9
  circuits, and the deterministic prove trace (`snarkrs trace`, js_2x2_d32).

## Caveats

- **The Mac was shared.** Each timed command waited until other processes used fewer than 3 cores
  (CPU run) or 4.5 cores (Metal run, raised after macOS daemons held the machine at 3 to 4 cores
  for 25 minutes), measured from `ps`, since the load average still carries the runner's own
  last step. Another session's job still started mid-step at least once: the CPU run's round-1
  stock `ptau prepare` at 2^18 took 395.7 s against 228 s in rounds 2 and 3. Rows recorded with
  other processes at or above the gate are marked `*` in the tables. Medians absorb single
  outliers, and the rounds agree in direction on every row the summary calls a gain.
- The Metal run was stopped once and resumed (`arkab.py --resume`). Only the unit that was cut
  short, round 1 perf `zkey contribute` on js_16x16_d32, was redone.
- One machine, one run per backend. No x86 or CUDA numbers (below), so the x86 asm/mulx
  interaction with these changes is unmeasured. x86 keeps the `adc`/`sbb` intrinsics, and BN254's
  mul/square still take the derive's Rust path under `+adx,+bmi2`, as the x86 A/B of
  2026-10-10 found.
- The Mac binaries were built from 3322c15 plus the uncommitted `verify_time` example and runner
  scripts, which 4d90f7b then committed (the prover source is unchanged between the two). sha256:
  stock `snarkrs` dc858a08...fa83c, perf `snarkrs` ccd7594c...e1c3f1 (full hashes below).

## Not run

- **x86 CPU (c7i.4xlarge) and CUDA (g4dn.2xlarge): not run.** Both bundles are built and
  prepared (`target/arkab/bundle-x86`, `target/arkab/bundle-cuda`, `bench/aws/arkab-ec2.py`). The
  c7i instance i-06f9561dc66691a0b launched, but SSH never connected: this Mac's public IP had
  changed to one not in the `agent-ssh` security group, and adding it was refused in this
  session. The instance was stopped from the runner side. The run needs that ingress rule (or a
  run from an allowed IP) to proceed.
- x86 binaries (`cargo zigbuild --release --target x86_64-unknown-linux-gnu -p snarkrs
  --no-default-features --features cpu`, `-C target-feature=+adx,+bmi2`, from 4d90f7b): stock
  `f1593bb98736040336dfc5e9e5a0b8e6f71b27881766226fd496fb0a97bf27f1`, perf
  `7a92de2507135cf5533a3ba3579dd1bb2c43c40a9da87856788d1e7d347221b1`. CUDA (`--features cpu,cuda`,
  same flags): stock `9454a1c9b17becf495b786eaeb0225be899df4df88f10d537179206d0cea8477`, perf
  `d2dada33ab78db231f026494feb56e14247d7fa16d05ed2328a13fc54c863824`.

## Setup

| | |
|---|---|
| machine | Apple M2 Max, 12 cores (8P + 4E), 96 GiB, macOS |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28), release profile (fat LTO, 1 CGU) |
| build | `cargo build --release -p snarkrs --features metal`, plus `-p snarkrs-groth16 --example verify_time --example field_ops`, one target dir per build |
| stock `snarkrs` | `dc858a0821eb7ee24b84230a7dc989a95bf95df5b5b4fd410707f271cb0fa83c` |
| perf `snarkrs` | `ccd7594c570bd0ab5206ffb728d1172d943a5b18eefc64b113cb48683ee1c3f1` |
| stock / perf `verify_time` | `ba9601a4...2f953a` / `70be8237...e9daae` |
| phase 1 inputs | contribute and verify: `local_13.ptau`, `local_19.ptau`; prepare: `local_13.ptau`, `ppot_0080_18.ptau` (sizes from ceremony-cpu.md and ceremony-metal.md) |
| phase 2 inputs | `groth16 setup` from each circuit's r1cs and the smallest `ppot_0080_NN.ptau` that covers its domain (2^15 to 2^21), then `zkey contribute` and `zkey verify init` on that key |
| snarkjs | 0.7.6 |

Reproduce:

```
bench/scripts/arkab-build.sh mac target/arkab/bin-mac stock=../algebra-v05-stock perf=../algebra-v05-perf
bench/scripts/arkab.py --bin-dir target/arkab/bin-mac --artifacts bench/artifacts --ptau-dir bench/ptau \
    --out target/arkab/mac-cpu --backend cpu --snarkjs-zkeys js_2x2_d32 js_16x16_d32 --max-load 3
bench/scripts/arkab.py ... --out target/arkab/mac-metal --backend metal --max-load 4.5 \
    --steps proof trace prove contribute beacon prepare setup zcontribute zbeacon
bench/scripts/arkab_report.py target/arkab/mac-cpu --snarkjs $(which snarkjs) --artifacts bench/artifacts --ptau-dir bench/ptau
```

## Mac M2 Max, CPU backend

### prove (warm), ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 66.6 | 65.7 | -1.3% | 62.5 / 62.9 ; 67.8 / 66.8 ; 67.9 / 66.1 |
| railgun-01x01 | 52.8 | 51.0 | -3.4% | 51.0 / 49.7 ; 54.0 / 51.1 ; 52.8 / 52.0 |
| tornado | 87.2 | 85.4 | -2.0% | 83.2 / 82.3 ; 89.1 / 88.2 ; 87.3 / 86.0 |
| sha256 | 45.4 | 42.1 | -7.3% | 43.0 / 39.3 ; 45.5 / 43.2 ; 49.5 / 43.5* |
| railgun-13x01 | 306.6 | 296.5 | -3.3% | 293.5 / 288.5 ; 308.9 / 302.4 ; 310.2 / 298.1 |
| js_16x16_d32 | 398.3 | 386.6 | -2.9% | 381.1 / 368.2 ; 402.5 / 389.1 ; 405.0 / 394.0 |
| rsa2048 | 171.6 | 160.3 | -6.6% | 167.0 / 155.9 ; 172.2 / 161.7 ; 174.2 / 161.7 |
| keccak256 | 163.5 | 151.6 | -7.3% | 157.3 / 147.4 ; 165.6 / 154.3 ; 165.7 / 154.9 |
| anon-aadhaar | 1331.5 | 1222.8 | -8.2% | 1407.0 / 1175.7* ; 1333.0 / 1228.4 ; 1325.8 / 1229.2* |

### verify (in process), ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 1.540 | 1.091 | -29.2% | 1.521 / 1.076 ; 1.540 / 1.091 ; 1.543 / 1.094 |
| railgun-01x01 | 1.467 | 1.032 | -29.6% | 1.445 / 1.018 ; 1.467 / 1.032* ; 1.468 / 1.032 |
| tornado | 1.375 | 0.960 | -30.2% | 1.358 / 0.947 ; 1.375 / 0.960* ; 1.379 / 0.962 |
| sha256 | 1.245 | 0.863 | -30.7% | 1.230 / 0.851 ; 1.250 / 0.863* ; 1.245 / 0.863 |
| railgun-13x01 | 2.388 | 1.744 | -26.9% | 2.358 / 1.718 ; 2.388 / 1.744* ; 2.388 / 1.745 |
| js_16x16_d32 | 3.674 | 2.765 | -24.7% | 3.641 / 2.723 ; 3.678 / 2.765* ; 3.674 / 2.766* |
| rsa2048 | 1.786 | 1.284 | -28.1% | 1.760 / 1.266 ; 1.786 / 1.284* ; 1.788 / 1.286* |
| keccak256 | 1.248 | 0.862 | -30.9% | 1.234 / 0.853 ; 1.248 / 0.862* ; 1.249 / 0.864* |
| anon-aadhaar | 1.414 | 0.988 | -30.1% | 1.397 / 0.974 ; 1.414 / 0.988* ; 1.414 / 0.992* |

### ptau contribute, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| 2^13 | 611.3 | 480.8 | -21.4% | 570.0 / 444.4 ; 620.0 / 480.9 ; 611.3 / 480.8* |
| 2^19 | 37991 | 29315 | -22.8% | 37053 / 28249 ; 37991 / 29321 ; 38131 / 29315 |

### ptau verify, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| 2^13 | 64.0 | 57.3 | -10.5% | 62.2 / 57.3 ; 65.6 / 57.4 ; 64.0 / 56.8 |
| 2^19 | 2266.1 | 2256.6 | -0.4% | 2235.7 / 2233.9 ; 2266.1 / 2256.6 ; 2266.9 / 2260.5 |

### ptau prepare phase2, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| 2^13 | 5472.7 | 5490.3 | +0.3% | 5335.4 / 5490.3 ; 5493.6 / 5470.6 ; 5472.7 / 5491.3 |
| 2^18 | 228056 | 227671 | -0.2% | 395716 / 237321* ; 228056 / 227532 ; 227280 / 227671 |

### groth16 setup, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 2098.8 | 1724.6 | -17.8% | 1972.7 / 1597.7 ; 2098.8 / 1724.6 ; 2102.4 / 1756.4 |
| railgun-01x01 | 496.4 | 410.2 | -17.4% | 488.3 / 410.2 ; 496.4 / 423.7 ; 500.3 / 410.1 |
| tornado | 1221.2 | 1006.0 | -17.6% | 1163.6 / 982.5 ; 1237.9 / 1006.0 ; 1221.2 / 1034.5 |
| sha256 | 2857.7 | 2320.7 | -18.8% | 2855.7 / 2148.2 ; 2857.7 / 2320.7 ; 3233.4 / 2531.8 |
| railgun-13x01 | 3249.0 | 2680.3 | -17.5% | 3119.3 / 2514.5 ; 3249.0 / 2680.3 ; 3271.1 / 2707.0 |
| js_16x16_d32 | 16616 | 13473 | -18.9% | 15431 / 12696 ; 16616 / 13645 ; 17027 / 13473* |
| rsa2048 | 1982.7 | 1866.1 | -5.9% | 1909.5 / 1796.2 ; 1987.7 / 1866.1 ; 1982.7 / 1886.9 |
| keccak256 | 3455.4 | 2861.5 | -17.2% | 3296.7 / 2721.1 ; 3455.4 / 2871.1 ; 3460.8 / 2861.5 |
| anon-aadhaar | 25335 | 21584 | -14.8% | 24491 / 21557* ; 25335 / 21584* ; 25343 / 21681* |

### zkey contribute, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 467.1 | 418.6 | -10.4% | 464.6 / 645.1* ; 467.1 / 418.6 ; 489.3 / 371.3 |
| railgun-01x01 | 495.8 | 393.8 | -20.6% | 499.7 / 387.4 ; 495.8 / 399.2 ; 489.4 / 393.8 |
| tornado | 565.1 | 461.9 | -18.3% | 537.6 / 461.9 ; 604.6 / 455.6 ; 565.1 / 474.8 |
| sha256 | 1081.9 | 902.2 | -16.6% | 1047.5 / 856.2 ; 1148.8 / 923.7 ; 1081.9 / 902.2 |
| railgun-13x01 | 3583.5 | 2852.9 | -20.4% | 3555.4 / 2790.2 ; 3654.7 / 2928.5 ; 3583.5 / 2852.9 |
| js_16x16_d32 | 3668.6 | 2839.9 | -22.6% | 3410.5 / 2651.3 ; 3680.6 / 3016.0 ; 3668.6 / 2839.9 |
| rsa2048 | 3870.2 | 3106.3 | -19.7% | 3661.3 / 3106.3 ; 4193.2 / 3082.2 ; 3870.2 / 3254.4 |
| keccak256 | 4567.2 | 3552.6 | -22.2% | 4690.0 / 3413.5 ; 4471.9 / 3552.6 ; 4567.2 / 3554.8 |
| anon-aadhaar | 29089 | 23179 | -20.3% | 29926 / 23717 ; 29089 / 22296 ; 27067 / 23179 |

### zkey verify init, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 68.8 | 67.8 | -1.4% | 68.1 / 74.9 ; 68.8 / 67.8 ; 69.9 / 67.7 |
| railgun-01x01 | 70.3 | 68.2 | -2.9% | 70.3 / 65.8 ; 71.9 / 69.1 ; 70.2 / 68.2 |
| tornado | 73.3 | 71.1 | -3.0% | 72.0 / 71.1 ; 73.3 / 72.4 ; 74.4 / 70.2 |
| sha256 | 127.3 | 125.2 | -1.7% | 122.5 / 118.2 ; 127.3 / 125.2 ; 130.4 / 126.0 |
| railgun-13x01 | 383.8 | 379.0 | -1.2% | 377.3 / 368.4 ; 388.6 / 379.0 ; 383.8 / 385.1 |
| js_16x16_d32 | 380.9 | 376.1 | -1.3% | 380.9 / 372.8 ; 389.0 / 376.1 ; 380.0 / 390.2 |
| rsa2048 | 408.4 | 399.8 | -2.1% | 404.4 / 399.8 ; 408.4 / 401.0 ; 415.7 / 399.5 |
| keccak256 | 423.1 | 426.2 | +0.7% | 418.6 / 413.6 ; 433.9 / 426.2 ; 423.1 / 434.4 |
| anon-aadhaar | 2784.4 | 2736.3 | -1.7% | 2779.0 / 2792.8 ; 2852.2 / 2736.3 ; 2784.4 / 2711.4 |

### prove stage medians, ms

| circuit | variant | gather | ntt | pointwise | msm | assemble |
|---|---|---:|---:|---:|---:|---:|
| js_2x2_d32 | stock | 0.790 | 5.069 | 0.197 | 60.2 | 0.108 |
| js_2x2_d32 | perf | 0.685 | 3.844 | 0.165 | 60.6 | 0.094 |
| railgun-01x01 | stock | 0.536 | 5.103 | 0.196 | 46.4 | 0.105 |
| railgun-01x01 | perf | 0.491 | 3.744 | 0.162 | 46.2 | 0.101 |
| tornado | stock | 0.623 | 5.166 | 0.190 | 80.5 | 0.112 |
| tornado | perf | 0.613 | 3.830 | 0.170 | 80.1 | 0.098 |
| sha256 | stock | 0.841 | 10.3 | 0.296 | 33.4 | 0.108 |
| sha256 | perf | 0.778 | 7.676 | 0.231 | 32.7 | 0.094 |
| railgun-13x01 | stock | 1.345 | 42.7 | 0.794 | 260.5 | 0.109 |
| railgun-13x01 | perf | 1.473 | 32.2 | 0.619 | 261.5 | 0.098 |
| js_16x16_d32 | stock | 3.534 | 43.1 | 0.802 | 349.2 | 0.113 |
| js_16x16_d32 | perf | 3.472 | 32.0 | 0.619 | 350.2 | 0.095 |
| rsa2048 | stock | 1.680 | 42.4 | 0.801 | 126.0 | 0.110 |
| rsa2048 | perf | 1.620 | 31.9 | 0.599 | 125.2 | 0.095 |
| keccak256 | stock | 1.408 | 43.0 | 0.783 | 116.9 | 0.114 |
| keccak256 | perf | 1.387 | 32.2 | 0.606 | 117.6 | 0.096 |
| anon-aadhaar | stock | 9.962 | 408.6 | 5.462 | 902.7 | 0.117 |
| anon-aadhaar | perf | 8.198 | 306.0 | 3.787 | 899.5 | 0.101 |

Deterministic outputs: 23 of 23 byte-identical across variants and rounds (90 files hashed).

`*`: some row of that round ran while other processes used 3 or more cores.

Self-verified warm proofs: 540. Failures recorded by the runner: 0.

snarkjs 0.7.6: 26 of 26 accepted.

## Mac M2 Max, Metal backend

### prove (warm), ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 20.8 | 21.1 | +1.3% | 20.9 / 21.0 ; 20.9 / 23.9 ; 20.7 / 25.1 |
| railgun-01x01 | 20.7 | 20.7 | -0.3% | 20.7 / 20.7 ; 20.7 / 20.7 ; 20.8 / 20.6 |
| tornado | 25.4 | 25.5 | +0.6% | 25.4 / 25.5 ; 25.4 / 29.6 ; 25.4 / 25.2 |
| sha256 | 11.4 | 11.4 | -0.5% | 13.0 / 12.5 ; 11.1 / 11.4 ; 11.4 / 11.3 |
| railgun-13x01 | 90.3 | 90.1 | -0.2% | 90.2 / 89.9 ; 92.9 / 90.7 ; 90.4 / 90.6 |
| js_16x16_d32 | 91.2 | 90.9 | -0.3% | 91.0 / 90.3 ; 93.3 / 91.5 ; 90.6 / 90.7 |
| rsa2048 | 35.4 | 35.5 | +0.2% | 35.5 / 35.2 ; 35.6 / 35.6 ; 35.2 / 35.9 |
| keccak256 | 27.3 | 27.3 | +0.3% | 27.3 / 27.3 ; 27.3 / 27.3 ; 27.1 / 27.3 |
| anon-aadhaar | 185.7 | 186.0 | +0.2% | 184.9 / 185.2 ; 185.7 / 201.0 ; 186.1 / 186.1 |

### ptau contribute, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| 2^13 | 127.5 | 126.5 | -0.8% | 1062.1 / 126.5 ; 126.0 / 137.9 ; 127.5 / 125.7 |
| 2^19 | 5302.0 | 5290.9 | -0.2% | 5215.1 / 5218.8 ; 5468.7 / 5435.8 ; 5302.0 / 5290.9 |

### ptau prepare phase2, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| 2^13 | 1194.6 | 1199.8 | +0.4% | 2768.4 / 1263.0 ; 1192.2 / 1191.9 ; 1194.6 / 1199.8 |
| 2^18 | 19481 | 19078 | -2.1% | 18723 / 18628 ; 19995 / 20331 ; 19481 / 19078 |

### groth16 setup, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 2854.1 | 2603.3 | -8.8% | 2854.1 / 2705.1 ; 2900.8 / 2557.2 ; 2836.1 / 2603.3 |
| railgun-01x01 | 513.8 | 445.0 | -13.4% | 551.0 / 454.9 ; 509.1 / 445.0 ; 513.8 / 444.5 |
| tornado | 1115.4 | 962.2 | -13.7% | 1199.8 / 976.7 ; 1115.4 / 912.9 ; 1112.8 / 962.2 |
| sha256 | 2827.3 | 2437.8 | -13.8% | 3071.2 / 2373.1 ; 2827.3 / 2437.8 ; 2641.1 / 2509.1 |
| railgun-13x01 | 3296.3 | 2704.7 | -17.9% | 3296.3 / 2704.7 ; 5616.4 / 2705.8 ; 3095.9 / 2551.5 |
| js_16x16_d32 | 21107 | 20052 | -5.0% | 21397 / 20052 ; 21107 / 19922 ; 20619 / 20089 |
| rsa2048 | 3840.5 | 3895.1 | +1.4% | 3911.5 / 3895.1 ; 3764.0 / 4365.0 ; 3840.5 / 3743.1 |
| keccak256 | 3286.5 | 2734.0 | -16.8% | 3283.0 / 2734.0 ; 3286.5 / 2750.4 ; 3287.9 / 2726.3 |
| anon-aadhaar | 25187 | 21806 | -13.4% | 25410 / 22206 ; 25187 / 21806 ; 25041 / 21407 |

### zkey contribute, ms

| item | stock | perf | change | per round (stock / perf) |
|---|---:|---:|---:|---|
| js_2x2_d32 | 92.7 | 97.6 | +5.3% | 92.5 / 96.1 ; 95.8 / 109.7 ; 92.7 / 97.6 |
| railgun-01x01 | 95.4 | 95.9 | +0.5% | 95.4 / 95.9 ; 94.6 / 96.6 ; 96.1 / 94.0 |
| tornado | 106.9 | 108.1 | +1.1% | 106.9 / 108.1 ; 106.5 / 108.4 ; 107.8 / 102.1 |
| sha256 | 160.9 | 160.3 | -0.3% | 158.3 / 161.4 ; 166.2 / 160.3 ; 160.9 / 155.6 |
| railgun-13x01 | 415.2 | 421.7 | +1.6% | 414.5 / 412.4 ; 418.8 / 421.7* ; 415.2 / 428.8 |
| js_16x16_d32 | 420.9 | 418.4 | -0.6% | 422.0 / 425.7 ; 420.9 / 417.7 ; 415.1 / 418.4 |
| rsa2048 | 464.5 | 477.6 | +2.8% | 464.5 / 511.0 ; 461.8 / 460.9 ; 468.3 / 477.6 |
| keccak256 | 515.6 | 505.2 | -2.0% | 515.6 / 555.8 ; 569.5 / 505.0 ; 514.5 / 505.2 |
| anon-aadhaar | 3036.4 | 3086.4 | +1.6% | 3133.6 / 3086.4 ; 3036.4 / 3100.0 ; 3027.3 / 3032.7 |

### prove stage medians, ms

| circuit | variant | gather | ntt | pointwise | msm | assemble |
|---|---|---:|---:|---:|---:|---:|
| js_2x2_d32 | stock | 0.017 | 1.977 | 0.000 | 18.6 | 0.104 |
| js_2x2_d32 | perf | 0.018 | 1.997 | 0.000 | 18.9 | 0.090 |
| railgun-01x01 | stock | 0.019 | 1.545 | 0.000 | 18.9 | 0.100 |
| railgun-01x01 | perf | 0.020 | 1.546 | 0.000 | 18.8 | 0.099 |
| tornado | stock | 0.026 | 1.821 | 0.000 | 23.2 | 0.109 |
| tornado | perf | 0.025 | 1.839 | 0.000 | 23.4 | 0.093 |
| sha256 | stock | 0.044 | 2.453 | 0.000 | 8.647 | 0.099 |
| sha256 | perf | 0.050 | 2.446 | 0.000 | 8.604 | 0.094 |
| railgun-13x01 | stock | 0.141 | 40.1 | 0.000 | 49.7 | 0.111 |
| railgun-13x01 | perf | 0.140 | 38.6 | 0.000 | 50.5 | 0.103 |
| js_16x16_d32 | stock | 0.142 | 47.3 | 0.000 | 42.7 | 0.109 |
| js_16x16_d32 | perf | 0.149 | 46.2 | 0.000 | 44.2 | 0.109 |
| rsa2048 | stock | 0.164 | 13.8 | 0.000 | 21.2 | 0.111 |
| rsa2048 | perf | 0.167 | 13.7 | 0.000 | 21.4 | 0.099 |
| keccak256 | stock | 0.211 | 9.091 | 0.000 | 17.9 | 0.107 |
| keccak256 | perf | 0.197 | 9.012 | 0.000 | 17.9 | 0.095 |
| anon-aadhaar | stock | 1.113 | 79.1 | 0.000 | 105.1 | 0.129 |
| anon-aadhaar | perf | 1.110 | 79.6 | 0.000 | 105.2 | 0.102 |

Deterministic outputs: 23 of 23 byte-identical across variants and rounds (90 files hashed).

`*`: some row of that round ran while other processes used 4.5 or more cores.

Self-verified warm proofs: 540. Failures recorded by the runner: 0.

snarkjs 0.7.6: 24 of 24 accepted.

