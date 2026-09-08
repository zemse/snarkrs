# The ceremony on the CPU: snarkjs 0.7.6 against g16

Measured, not quoted. This file exists to support one claim, that the phase-1 and phase-2
ceremony no longer needs snarkjs, so every correctness row below is either a whole-file byte
comparison or snarkjs' own verifier saying yes. None of them is "the output looks plausible".

The circuits are real production ones, not the small powers the unit tests use: two circuits
from an external circuits repo, each compiled at `--O1` and `--O2`, 45k to 616k constraints
over domains 2^16 to 2^20.

## Method

| | |
|---|---|
| machine | Apple M2 Max, 12 cores, 96 GiB, macOS 26.6 |
| g16 | commit `cee8458`, `cargo build --release` |
| snarkjs | 0.7.6, node v25.9.0 |
| circuits | an external circuits repo, circom 2.2.3 |
| phase-1 reps | 5 per power, at powers 8, 10, 12 and 14 |
| phase-2 reps | 3 per build |

The snarkjs used is the copy in the circuits repo's own `node_modules`, which is the install
that produced those artifacts. Its output is therefore the oracle the shipped files were
actually made with, not a same-version rebuild that happens to agree.

Another lane rebuilt `target/release/g16` partway through the correctness runs, so every
correctness row was re-run afterwards against a single binary pinned at `cee8458`, and all of
them still hold. The timings were all taken after that rebuild.

Within a rep the two implementations run back to back on the same input file, so a machine
whose load is drifting penalises both equally. **The machine was not quiet:** other lanes ran
cargo tests and snarkjs throughout, and the one-minute load average ranged 13.8 to 51.8 on 12
cores. Median and minimum agree to within a few percent on every phase-2 row, so the
absolutes are usable, but the ratio is the number that survives the contention.

**There are no Metal or CUDA columns, and the reason is structural rather than "CPU
milestone".** `g16 setup` does route its MSM through `g16_msm::MsmBackend`, so the seam is
real. But `g16-metal` and `g16-cuda` implement `g16_core::Backend`, the whole prover, not that
trait, so there is nothing yet to select between and `g16 setup` has no `--backend` flag to
measure. Putting setup on a GPU is a constructor swap in `msm_backend()`
(`crates/g16-cli/src/main.rs:296`), not a change to any ceremony signature.

## What round-tripped

All four builds, at every step. `setup` is compared whole-file against a freshly generated
snarkjs `_0000.zkey`; `vkey` against the `_verification_key.json` shipped beside the build.

| build | opt | constraints | domain | section-4 coefficients | ptau | setup byte-equal | `snarkjs zkey verify` | vkey byte-equal |
|---|---|---:|---:|---:|---|:---:|:---:|:---:|
| `prod_small_O2` | O2 | 45,118 | 2^16 | 776,667 | `ppot_0080_16` | yes | ZKey Ok! | yes |
| `prod_small_O1` | O1 | 115,739 | 2^17 | 104,862 | `ppot_0080_17` | yes | ZKey Ok! | yes |
| `prod_large_O2` | O2 | 246,030 | 2^18 | 3,926,499 | `ppot_0080_18` | yes | ZKey Ok! | yes |
| `prod_large_O1` | O1 | 616,191 | 2^20 | 698,280 | `ppot_0080_21` | yes | ZKey Ok! | yes |

The vkey column is the stronger of the two vkey checks available: it exports from the
**shipped** zkey, which is a contributed and beaconed key snarkjs wrote on another run, and
the result matches the shipped JSON byte for byte, one-space indent and absent trailing
newline included.

### The fully native chain

`g16 setup` then `g16 zkey contribute` then `g16 zkey beacon`, with no snarkjs anywhere in
the chain, on all four builds. Each final key was then handed to three verifiers:

| build | `g16 zkey verify --init` | `g16 zkey verify --r1cs` | `snarkjs zkey verify` |
|---|:---:|:---:|:---:|
| `prod_small_O2` | OK | OK | ZKey Ok! |
| `prod_small_O1` | OK | OK | ZKey Ok! |
| `prod_large_O2` | OK | OK | ZKey Ok! |
| `prod_large_O1` | OK | OK | ZKey Ok! |

`--r1cs` is the only form that ties the key to a circuit, and it costs a full setup to run.

### `ptau info`

Thirteen files: the nine in `bench/ptau/`, and the four of the power-20 chain shipped with
the circuits. Every one reported its power, ceremony power and prepared flag correctly, in
0.00 s of wall clock each by `time`, since the command only walks the section table and never
reads a point. The seven `ppot_0080_*` files all report ceremony power 28 against their own
smaller power, which is the truncation marker, and 62 contributions.

One file is not what its name says, and that is covered below.

## Phase 1

Median of 5, seconds.

| power | command | snarkjs | g16 | ratio |
|---:|---|---:|---:|---:|
| 8 | `ptau new` | 0.29 | **0.01** | 24x |
| 8 | `ptau contribute` | 0.43 | **0.15** | 2.8x |
| 8 | `ptau beacon` | 0.42 | **0.15** | 2.8x |
| 8 | `ptau prepare` | 1.72 | **0.35** | 5.0x |
| 10 | `ptau new` | 0.26 | **0.01** | 33x |
| 10 | `ptau contribute` | 0.81 | **0.47** | 1.7x |
| 10 | `ptau beacon` | 0.75 | **0.48** | 1.5x |
| 10 | `ptau prepare` | 6.47 | **1.26** | 5.1x |
| 12 | `ptau new` | 0.36 | **0.01** | 36x |
| 12 | `ptau contribute` | 2.31 | **0.61** | 3.8x |
| 12 | `ptau beacon` | 2.22 | **0.52** | 4.2x |
| 12 | `ptau prepare` | 25.22 | **3.00** | 8.4x |
| 14 | `ptau new` | 0.51 | **0.01** | 34x |
| 14 | `ptau contribute` | 6.41 | **1.35** | 4.7x |
| 14 | `ptau beacon` | 6.42 | **1.27** | 5.1x |
| 14 | `ptau prepare` | 135.78 | **18.31** | 7.4x |

Two rows here mean less than they look.

**`ptau new` is a startup benchmark, not an arithmetic one.** It writes powers of the plain
generators with no group operations at all, so snarkjs' 260 to 510 ms is node reaching its
own main function. The 34x is real wall clock a user waits, but nobody should read it as a
statement about field code.

**`contribute` and `beacon` below power 12 are dominated by fixed cost on both sides**, which
is why the ratio at power 10 (1.5x) is worse than at power 8 (2.8x) and both are worse than
power 14 (5x). Only the power-14 row is measuring the rescaling itself.

`prepare` is the honest one, and it is also the only phase-1 command that costs real time at
production power. Its ratio climbs from 5x at powers 8 and 10 to 7-8x at 12 and 14, which is
what a fixed per-process cost amortising away looks like.

## Phase 2

Median of 3, seconds, on the real builds.

| build | command | snarkjs | g16 | ratio |
|---|---|---:|---:|---:|
| `prod_small_O2` | `setup` | 73.12 | **9.13** | 8.0x |
| `prod_small_O2` | `zkey contribute` | 4.99 | **0.94** | 5.3x |
| `prod_small_O2` | `zkey beacon` | 4.87 | **0.96** | 5.1x |
| `prod_small_O1` | `setup` | 16.73 | **2.71** | 6.2x |
| `prod_small_O1` | `zkey contribute` | 10.28 | **2.05** | 5.0x |
| `prod_small_O1` | `zkey beacon` | 10.34 | **2.04** | 5.1x |
| `prod_large_O2` | `setup` | 282.94 | **44.62** | 6.3x |
| `prod_large_O2` | `zkey contribute` | 20.12 | **4.16** | 4.8x |
| `prod_large_O2` | `zkey beacon` | 20.91 | **4.11** | 5.1x |
| `prod_large_O1` | `setup` | 74.69 | **15.29** | 4.9x |
| `prod_large_O1` | `zkey contribute` | 67.13 | **13.72** | 4.9x |
| `prod_large_O1` | `zkey beacon` | 66.06 | **13.84** | 4.8x |

`export-verificationkey` is not in the table because it does not belong in one: it reads five
short sections and writes 6 to 39 KB of JSON, in 8 to 10 ms on every build regardless of size.
snarkjs has no comparable single-file command that is worth timing against it.

## Two cost models, and neither is the constraint count

The four builds happen to separate the variables that usually move together, so the phase-2
numbers pin down what each command actually charges for.

**`setup` tracks section-4 coefficients, not constraints and not the domain.**
`prod_small_O2` has 2.6x *fewer* constraints than the O1 build of the same circuit
and half the domain, and its setup takes **3.4x longer** (9.13 s against 2.71 s). The one
thing that goes the other way is the coefficient count, 776,667 against 104,862, which is 7.4x.
The same inversion appears on the other circuit: the O2 build has a quarter the domain of the
O1 build and takes 2.9x longer, and again it is the one with 5.6x the coefficients. This is
the density effect the circuits' own HANDOFF warns about, and for setup it is not a small
correction, it dominates.

**`contribute` and `beacon` track the rescaled point count, which is `domainSize` plus
`nVars - nPublic - 1`.** Those are exactly zkey sections 9 and 8, the only two
`zkey_contribute.js:91-92` multiplies by `invDelta`. Section lengths read back off the four
zkeys confirm the formula exactly, and the implied throughput is flat across a 15x size range:

| build | section 8 (L) | section 9 (H) | total points | g16 | snarkjs |
|---|---:|---:|---:|---:|---:|
| `prod_small_O2` | 45,126 | 65,536 | 110,662 | 118 k pt/s | 22 k pt/s |
| `prod_small_O1` | 115,746 | 131,072 | 246,818 | 121 k pt/s | 24 k pt/s |
| `prod_large_O2` | 245,665 | 262,144 | 507,809 | 122 k pt/s | 25 k pt/s |
| `prod_large_O1` | 615,825 | 1,048,576 | 1,664,401 | 121 k pt/s | 25 k pt/s |

Flat to within 4% over 15x, which is worth knowing because it makes `contribute` the one
ceremony command whose cost can be predicted from the circuit shape alone, before building
anything.

## What only appeared at production size

Nothing broke. The one thing that did surface is a snarkjs limit, not a g16 one.

**A power-20 `prepare phase2` under snarkjs was abandoned after 43 minutes, and the file it
left behind is still sitting there named as if it had succeeded.** The circuits repo's
`ptau/pot20_final.ptau` is not a prepared file at all. Its own build log ends with

    phase1.sh: line 28: 58753 Terminated: 15   snarkjs powersoftau prepare phase2 ...

still inside the tauG2 transform. That is SIGTERM, so this is not evidence that snarkjs
cannot do the job, only that it had not done it in 43 minutes and something gave up on it.
For scale, snarkjs' own power-14 prepare above is 136 s, and power 20 is 64x the elements at
1.4x the depth, which puts a straight extrapolation near two and a half hours.

The file it left behind is 679 MB
against the 1,208 MB a complete one needs, and `g16 ptau info` says so rather than failing:
section 13 declares zero bytes, and the walk then reads section 12's data as a header and
reports a section id of 45,883,430 declaring 5.29 exabytes. This is what the lenient index
mode is for. Every other command in the repo would reject the file, correctly, but only `info`
can tell you *why*.

g16 completed the same job on the same input:

| | snarkjs | g16 |
|---|---|---|
| power-20 `prepare phase2` | abandoned at ~43 min, incomplete | **962 s (16.0 min)**, complete |
| output | 679,480,389 bytes, truncated mid-section-13 | 1,207,962,589 bytes, all 11 sections at expected size |
| peak RSS | not reached | 1.22 GB |

The output was then checked five ways rather than trusted:

- `g16 ptau info`: 11 of 11 sections, every one at its expected length, `prepared true`.
- `g16 ptau verify`: OK, 17 s.
- **`snarkjs powersoftau verify`: Powers of Tau Ok!, 118 s.** This is the check that matters,
  because it is the only one that pairs the Lagrange sections 12 to 15 back against section 2
  instead of trusting anything g16 computed. A file that is merely self-consistent fails here.
- `g16 setup` and `snarkjs groth16 setup` of the 2^20 build against **this file**: byte
  identical. That is an exact fit, `cirPower == power`, so it also exercises the
  `zkey_new.js:511` overrun that reads past ptau section 2 into section 3's header.
- `snarkjs zkey verify` on the resulting key: ZKey Ok!

So the file g16 produced is not merely well formed. snarkjs' own phase-1 verifier accepts it,
snarkjs' own setup reads it to the same bytes g16 does, and snarkjs' own phase-2 verifier
accepts the key that comes out.

## Memory

Peak resident set for `g16 setup`, which is the largest of the phase-2 commands.

| build | domain | peak RSS |
|---|---:|---:|
| `prod_small_O2` | 2^16 | 226 MB |
| `prod_small_O1` | 2^17 | 275 MB |
| `prod_large_O2` | 2^18 | 1,012 MB |
| `prod_large_O1` | 2^20 | 1,778 MB |

snarkjs was given `--max-old-space-size=49152` throughout, which is what the circuits repo's
own `phase2.sh` passes. Whether it would survive on the default heap was not tested, so this
is a note about how it is run in practice rather than a claim about its floor.
