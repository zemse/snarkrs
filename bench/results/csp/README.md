# The ethproofs client-side-proving benchmark: local runs

[ethproofs.org/csp-benchmarks](https://ethproofs.org/csp-benchmarks) publishes six numbers
per circuit for seventeen proving systems. One of the seventeen is `circom`, which is
circom + witnesscalc + rapidsnark. This run keeps the first two and swaps the third for
this prover, across eleven variants. The protocol, and the places it deliberately departs
from upstream, are in [`bench/csp/README.md`](../../csp/README.md) and are not restated
here.

**The number worth taking away is 1.76x to 2.36x, not 8x.** Both are below. The first is
against rapidsnark run on this machine, over these keys, in this loop. The second is
against the row the page currently serves, which was measured on other hardware against
circuits 1.74x larger with the witness generator upstream ships. The second is the one that
will get quoted and it is the one that is wrong.

This file was first written from a run on 2026-09-24 at the stock witness generator. Every
table except the poseidon one and the witness generation section below is from the
2026-10-27 re-run; the poseidon one is from 2026-10-28.

## Method

| | |
|---|---|
| machine | Apple M2 Max, 12 cores, 96 GiB, macOS 27.0 |
| date | 2026-10-27, at `72b3946` |
| statistic | one warm-up, then the mean of 10 timed iterations; median, min and max kept alongside |
| zkeys | the published `zkeys-v2` release, checksum verified, not regenerated locally |
| witness | witnesscalc, in process, inside the timed region, identical for all three provers |
| witness generator | `bench/csp/build.rs` as of this commit: the `Fr_mod`/`Fr_idiv` paths of `b414260`, the short products of `e83ee96` and the small `Fr_pow` of `3465d0b`, over generators compiled with `--sanity_check 0` (`f73229f`) |
| provers | `g16` cpu, `g16` metal, rapidsnark; one loop, only the prover swapped |
| variants | sha256 and keccak at 128 to 2048 bytes, ecdsa at 32; poseidon only under the unblinded exception, see below |
| memory | 10 samples per variant, `MallocLargeCache=0` in the sampling process |
| raw output | `bench/results/csp/metrics/*.json`, untracked, which is why the numbers are in this file |

Every duration below is milliseconds unless the column header says otherwise. Every one is
a full `proof_duration` in the page's sense: witness generation, the zkey read, and the
proof, because that is what the protocol times.

The harness has no contention guard of its own, so each (prover, target) pass started only
once the one-minute load average was under 3 and the other processes held under two cores.
A second full pass straight after agreed with this one within 3.2% on every cell.

## rapidsnark on the same machine

This is the only comparison in this file with one variable in it. Same box, same zkeys, same
in-process witness buffer, same loop, prover swapped.

| variant | constraints | rapidsnark | g16 cpu | g16 metal | rs / cpu | rs / metal |
| ------- | ----------: | ---------: | ------: | --------: | -------: | ---------: |
| `sha256_128` | 53,048 | 92.39 | **47.36** | 49.73 | 1.95x | 1.86x |
| `sha256_256` | 89,224 | 153.35 | 87.02 | **65.63** | 1.76x | 2.34x |
| `sha256_512` | 161,576 | 314.57 | 155.81 | **109.88** | 2.02x | 2.86x |
| `sha256_1024` | 306,280 | 626.93 | 286.37 | **195.02** | 2.19x | 3.21x |
| `sha256_2048` | 595,688 | 1,170.26 | 556.11 | **369.85** | 2.10x | 3.16x |
| `keccak_128` | 93,184 | 169.62 | 86.04 | **69.88** | 1.97x | 2.43x |
| `keccak_256` | 187,328 | 353.41 | 163.51 | **123.22** | 2.16x | 2.87x |
| `keccak_512` | 375,744 | 690.88 | 309.19 | **222.29** | 2.23x | 3.11x |
| `keccak_1024` | 752,576 | 1,364.03 | 611.49 | **410.07** | 2.23x | 3.33x |
| `keccak_2048` | 1,506,240 | 2,789.26 | 1,181.99 | **803.39** | 2.36x | 3.47x |

On the ten hash circuits the CPU backend is 1.76x to 2.36x of rapidsnark, up from 1.35x to
1.59x on 2026-09-24. Metal is 1.86x to 3.47x and trends with size: monotone on keccak, and
on sha256 up to 3.21x at 1,024 bytes before 3.16x at 2,048. At `sha256_128` the CPU backend
now edges Metal, because the fixed cost Metal pays per proof is the same at every size.

rapidsnark's own prove time is within 5% of the September run on nine of the eleven
variants. The other two, `sha256_1024` and `keccak_2048`, had September means 7.6% and 8.6%
above their medians from a few slow iterations. What moved its totals is the witness
generator, which it shares. The rest of the change is ours.

## Against the published circom row

At face value we are 7.31x to 8.69x the published `circom` row on the hash circuits. That
is not a prover result, for three reasons, two of them quantifiable from the data.

**The circuits shrank.** The published numbers were taken against `zkeys-v1`. Their
`sha256_128` is 94,976 constraints; the one in this run is 53,048. Their `keccak_128` is
163,638 against 93,184. The ratio is 1.74x to 1.79x at every hash size.

**The machine differed.** The page measures on an AWS `mac2.metal`, an Apple M1 with 8
cores and 16 GB. This is an M2 Max with 12 cores.

**The witness generator differed.** The page runs witnesscalc as upstream ships it. This run
runs the patched one, which is 6.3x faster on `keccak_2048` than the stock generator was
inside this same harness in September.

Normalising by constraint count removes the first term:

| variant | our constraints | published constraints | published ms | g16 cpu ms | face value | per constraint |
| ------- | --------------: | --------------------: | -----------: | ---------: | ---------: | -------------: |
| `sha256_128` | 53,048 | 94,976 | 355.71 | 47.36 | 7.51x | 4.19x |
| `sha256_256` | 89,224 | 158,656 | 635.82 | 87.02 | 7.31x | 4.11x |
| `sha256_512` | 161,576 | 286,016 | 1,209.97 | 155.81 | 7.77x | 4.39x |
| `sha256_1024` | 306,280 | 540,736 | 2,389.19 | 286.37 | 8.34x | 4.73x |
| `sha256_2048` | 595,688 | 1,050,176 | 4,810.00 | 556.11 | 8.65x | 4.91x |
| `keccak_128` | 93,184 | 163,638 | 655.26 | 86.04 | 7.62x | 4.34x |
| `keccak_256` | 187,328 | 327,244 | 1,272.34 | 163.51 | 7.78x | 4.45x |
| `keccak_512` | 375,744 | 654,456 | 2,513.49 | 309.19 | 8.13x | 4.67x |
| `keccak_1024` | 752,576 | 1,308,880 | 5,088.06 | 611.49 | 8.32x | 4.78x |
| `keccak_2048` | 1,506,240 | 2,617,728 | 10,269.19 | 1,181.99 | 8.69x | 5.00x |

The second and third terms are measurable together, because rapidsnark is on both sides of
them. Per constraint, the published rapidsnark is 2.06x to 2.33x the rapidsnark measured
here. Multiply that by the 1.76x to 2.36x prover gap from the previous section and the 4.11x
to 5.00x per-constraint column comes back. So of a headline near 8x, roughly 1.75x is the
circuit, roughly 2.2x is the machine and the witness generator together, and roughly 2.1x
is the prover. Only a `mac2.metal` run with the stock generator would split the middle term.

Two smaller columns, for completeness. `verify_duration` reproduces the published shape
(10.7 ms to 273.6 ms) only because the protocol re-reads the zkey and derives the verifier
from it; the honest verifier cost is `verify_vkey_ms` in the breakdown files, which is 1.20
ms to 1.32 ms across all ten variants and does not move with circuit size. `proof_size`
is 980 to 989 bytes here against 918 to 1,020 published; the difference is in the JSON, not
the proof, which is two G1 points and one G2 point either way.

## Where the time goes

Witness generation plus the zkey read is 26.2% to 30.6% of every CPU measurement, and the
fraction does not trend away with size:

| variant | witness | zkey read | prepare | prove | total | witness + zkey |
| ------- | ------: | --------: | ------: | ----: | ----: | -------------: |
| `sha256_128` | 4.45 | 9.40 | 1.40 | 32.11 | 47.36 | 29.2% |
| `sha256_256` | 7.62 | 15.80 | 2.25 | 61.35 | 87.02 | 26.9% |
| `sha256_512` | 13.87 | 27.01 | 3.87 | 111.06 | 155.81 | 26.2% |
| `sha256_1024` | 27.19 | 50.06 | 7.13 | 201.99 | 286.37 | 27.0% |
| `sha256_2048` | 53.48 | 96.21 | 14.18 | 392.24 | 556.11 | 26.9% |
| `keccak_128` | 5.43 | 18.41 | 2.37 | 59.83 | 86.04 | 27.7% |
| `keccak_256` | 11.12 | 35.54 | 4.48 | 112.37 | 163.51 | 28.5% |
| `keccak_512` | 22.32 | 69.41 | 8.86 | 208.60 | 309.19 | 29.7% |
| `keccak_1024` | 44.88 | 137.58 | 17.86 | 411.17 | 611.49 | 29.8% |
| `keccak_2048` | 89.81 | 271.53 | 36.50 | 784.15 | 1,181.99 | 30.6% |

The witness is now the smaller half of that. On `keccak_2048` it was 569.79 ms in
September and is 89.81 ms here, and the zkey read, 271.53 ms, is three times it. That read
is the protocol re-reading 1.15 GB from disk on every iteration.

This is the reason a proving-only number cannot be put next to this page. Halving the CPU
prove time on `keccak_2048` would move the published total by 33%, not by 50%, and driving
prove to zero would still leave 397.8 ms on the clock.

It is also the ceiling on what a GPU can deliver end to end:

| variant | witness | zkey read | prepare | prove | total | witness + zkey |
| ------- | ------: | --------: | ------: | ----: | ----: | -------------: |
| `sha256_128` | 4.96 | 10.05 | 9.64 | 25.08 | 49.73 | 30.2% |
| `sha256_256` | 8.05 | 16.11 | 14.10 | 27.37 | 65.63 | 36.8% |
| `sha256_512` | 14.97 | 28.27 | 22.70 | 43.94 | 109.88 | 39.4% |
| `sha256_1024` | 29.05 | 53.61 | 40.61 | 71.76 | 195.02 | 42.4% |
| `sha256_2048` | 58.85 | 104.01 | 82.29 | 124.70 | 369.85 | 44.0% |
| `keccak_128` | 5.43 | 19.73 | 15.00 | 29.73 | 69.88 | 36.0% |
| `keccak_256` | 11.14 | 38.55 | 25.23 | 48.31 | 123.22 | 40.3% |
| `keccak_512` | 22.18 | 75.59 | 46.70 | 77.82 | 222.29 | 44.0% |
| `keccak_1024` | 45.03 | 150.07 | 91.00 | 123.98 | 410.07 | 47.6% |
| `keccak_2048` | 89.40 | 296.49 | 182.40 | 235.10 | 803.39 | 48.0% |

Take `sha256_1024`. Metal's prove is 71.76 ms against the CPU's 201.99 ms, 2.81x. The totals
are 286.37 ms against 195.02 ms, 1.47x. The gap is witness (29.05 ms), the zkey read
(53.61 ms), and the host-to-device upload the Metal backend charges to prepare (40.61 ms),
none of which the GPU shortens. `keccak_2048` is starker: 3.34x on prove, 1.47x on total,
with witness and zkey read at 48.0% of the measurement. If prepare and prove both went to
zero on that variant, the total would improve by 2.08x and no further.

## Memory

| variant | g16 cpu | g16 metal | published circom | cpu B/constraint | metal B/constraint | published B/constraint |
| ------- | ------: | --------: | ---------------: | ---------------: | -----------------: | ---------------------: |
| `sha256_128` | 67.3 | 103.6 | 162.3 | 1,269 | 1,953 | 1,708 |
| `sha256_256` | 102.9 | 155.2 | 249.9 | 1,153 | 1,740 | 1,575 |
| `sha256_512` | 182.7 | 255.4 | 452.1 | 1,130 | 1,581 | 1,581 |
| `sha256_1024` | 334.5 | 457.0 | 646.5 | 1,092 | 1,492 | 1,196 |
| `sha256_2048` | 654.3 | 809.1 | 1,352.4 | 1,098 | 1,358 | 1,288 |
| `keccak_128` | 108.5 | 161.0 | 246.6 | 1,164 | 1,728 | 1,507 |
| `keccak_256` | 205.9 | 275.1 | 473.2 | 1,099 | 1,469 | 1,446 |
| `keccak_512` | 390.1 | 514.9 | 722.0 | 1,038 | 1,370 | 1,103 |
| `keccak_1024` | 778.7 | 940.6 | 1,377.9 | 1,035 | 1,250 | 1,053 |
| `keccak_2048` | 1,527.6 | 1,841.6 | 2,679.3 | 1,014 | 1,223 | 1,024 |

This was the one column where we lost in September, at 1,862 to 2,046 bytes per constraint
on the CPU. It no longer is: the CPU backend now peaks at 1,014 to 1,269 bytes per
constraint against the published 1,024 to 1,708, below the published row on all ten hash
circuits once normalised, by 1% on `keccak_2048` and by up to 29% on `sha256_512`. In
absolute terms `keccak_2048` peaks at 1.53 GB here against the published 2.68 GB.

Metal costs another 1.21x to 1.54x over the CPU backend, because the device buffers sit on
top of the host copies the zkey parse already made. Per constraint that puts it level with
the published row on `sha256_512` and 2% to 25% above it on the other nine.

It still fits the box the page uses. The worst case in the table is `keccak_2048` on Metal
at 1,842 MB on a 16 GB `mac2.metal`.

Two caveats on this table. `peak_memory` for our rows comes from a separate sampling process
rather than the timed loop, and that process runs with macOS malloc's large cache turned
off, which a library user embedding the prover does not get by default. rapidsnark has no
column at all: the sampler links our prover, so those rows carry the collector's 0 marker
instead of a number taken from the wrong process.

## Cross-checks

**Constraint counts.** All ten sha256 and keccak counts match the upstream circuit table in
[`bench/vendor/csp-benchmarks/circom/README.md`](../../vendor/csp-benchmarks/circom/README.md)
exactly: 53,048 / 89,224 / 161,576 / 306,280 / 595,688 for sha256 and 93,184 / 187,328 /
375,744 / 752,576 / 1,506,240 for keccak.

**Key bytes.** The hash variants differ from the published `preprocessing_size`, which is
the v1 to v2 shrink rather than a mismatch: 1,152,894,061 for `keccak_2048` against the
published 1,339,459,874.

**Verification.** Each variant's warm-up proof is verified before any timing is recorded,
with the verifying key derived from the published zkey, and the run aborts on a rejection.
All thirty-three rows (eleven variants, three provers) cleared that gate, so our verifier
accepts rapidsnark's proofs over the published keys as readily as its own.

**An independent verifier was not run this time.** The check above uses our verifier on both
sides, so it cannot catch an error we make in both directions at once. `SNARKJS_VERIFY=1`
now wires `snarkjs groth16 verify` into the harness, but this run did not set it. The
evidence that the encoding is snarkjs' rather than merely self-consistent is still the
September bring-up check on `poseidon_2`, `sha256_128` and `keccak_128`, and the oracle's
own runs on `sha256_128` and `keccak_128`.

## ecdsa

The fourth circom target sits in its own section because it has no published row to compare
against: the page's `circom` entry covers sha256, keccak and poseidon only. secp256k1
signature verification, 512,955 constraints, the same three provers.

| prover | witness | zkey read | prepare | prove | total | peak rss |
| ------ | ------: | --------: | ------: | ----: | ----: | -------: |
| rapidsnark | 435.4 | n/a | 3.0 | 856.1 | 1,294.5 | not measured |
| `g16` cpu | 431.9 | 77.0 | 11.4 | 243.4 | 763.7 | 474 MB |
| `g16` metal | 429.3 | 83.3 | 53.9 | **87.6** | **654.0** | 629 MB |

This is the clearest case in the file of the point the previous section makes. On the
proving call alone Metal is 9.77x rapidsnark, the widest margin anywhere here. End to end it
is 1.98x, because witnesscalc takes about 430 ms whoever proves afterwards: 34% of
rapidsnark's measurement, 57% of the CPU's, 66% of Metal's. The witness generator is still
the target, and on this circuit the patches below moved it least, 672 ms to 430 ms.

The zkey read behaves differently here too. At 334 MB for 512,955 constraints the key is
larger per constraint than any hash circuit, because the width-12 comb table is inlined,
and `preprocessing_size` is 412,467,641 bytes once the 59.5 MB `.cpp` and 18.6 MB `.dat`
the page counts are added.

## poseidon, unblinded

The five poseidon keys on the page are unblinded: `gamma_g2` equals `delta_g2`, so anyone
can forge a proof against them. `ProvingKey::load` refuses such a key, so these rows were
measured on 2026-10-28 at `a02df99` under `--allow-unblinded-zkey`, which skips that one
comparison and runs every other check. Every row here is labelled `unblinded` in its
`feat` (`cpu-unblinded`, `metal-unblinded`, `auto-cpu-unblinded`, rapidsnark's
`unblinded`); none of them is fit for submission. Same protocol as the hash rows, one
warm-up then the mean of 10, 10 memory samples, each pass started under the contention
guard. The cell is the median of three passes; the old cell is the September run at the
stock witness generator.

| variant | constraints | rapidsnark | g16 cpu | g16 auto | g16 metal | cpu, September |
| ------- | ----------: | ---------: | ------: | -------: | --------: | -------------: |
| `poseidon_2` | 517 | 10.82 | **5.82** | 6.17 | 17.92 | 5.15 |
| `poseidon_4` | 736 | 11.75 | **6.41** | 6.08 | 17.69 | 5.56 |
| `poseidon_8` | 1,171 | 14.69 | **7.95** | 7.51 | 18.54 | 7.63 |
| `poseidon_12` | 1,613 | 15.65 | **8.34** | 8.11 | 18.44 | 8.37 |
| `poseidon_16` | 2,092 | 19.74 | **9.88** | 9.91 | 20.96 | 15.77 |

`auto` runs the CPU on every poseidon size, all of them under 65,536 constraints, so it and
the cpu column are the same prover measured twice: the gap between them, up to 6%, is the
noise at these sizes. Peak RSS is 14 to 21 MiB on the CPU and 37 to 45 MiB on Metal.

The CPU went from 15.77 to 9.88 ms at `poseidon_16` and is 1.8x to 2.0x rapidsnark at
every size. At `poseidon_2` and `poseidon_4` it is 13% and 15% slower than the September
cell, beyond the noise here and not explained: rapidsnark, which shares the witness, moved
by under 6%. Of the 5.82 ms at `poseidon_2`, 4.27 is `prove_ms` (2.12 of it MSM), 1.07 the
zkey read, 0.25 the witness and 0.23 the prepare.

Metal loses to both at every size, because its prepare and prove come to 16 to 19 ms
whatever the circuit: dispatch and pipeline setup, not arithmetic. Its column is the median
of six back-to-back passes. The first two Metal passes of the session came in at 30 to 55
ms per proof, and the third at 19 to 22, before settling; they are left out. Why the first
runs of a session are slower was not isolated.

Constraint counts and key bytes match the published row exactly, which is why poseidon
needs no per-constraint correction.

## Witness generation: 1.37x, and where the rest of it is

This section is the September measurement of the first patch, `b414260`, and is kept as
it was. Three more followed the next day and the run above includes all of them:
`f73229f` builds the sha256, keccak and ecdsa generators with `--sanity_check 0`,
`e83ee96` keeps a short product short, and `3465d0b` answers a small `Fr_pow` without
GMP. Their commits measured the suite at 1,506.62 ms to 779.95 ms on top of this patch,
and `keccak_2048` at 88 ms against 569 ms stock.

`bench/csp/build.rs` patches the witnesscalc checkout before it is built. The numbers
below are that patch, measured by `g16-csp-wit`, which runs witness generation with no
prover attached so a sampling profile of it is not also a profile of the prover.

`G16_CSP_STOCK_FR=1 CARGO_TARGET_DIR=target-stock cargo build --release --bin g16-csp-wit`
builds the same crate unpatched. Both binaries were then run alternately, five rounds of
ten repetitions each, and the minimum of the fifty is reported: a single threaded
measurement on a laptop competes with whatever else is running, and the minimum is the
sample that got a performance core to itself. Every repetition compares its `.wtns` bytes
against the first, and all sixteen witnesses are byte identical to the ones the stock
build produces.

| variant | stock ms | patched ms | x |
| ------- | -------: | ---------: | --: |
| `sha256_128` | 13.10 | 10.24 | 1.28 |
| `sha256_256` | 22.53 | 16.96 | 1.33 |
| `sha256_512` | 40.78 | 30.90 | 1.32 |
| `sha256_1024` | 77.55 | 58.60 | 1.32 |
| `sha256_2048` | 150.79 | 114.76 | 1.31 |
| `keccak_128` | 29.51 | 16.03 | 1.84 |
| `keccak_256` | 65.09 | 39.23 | 1.66 |
| `keccak_512` | 136.90 | 86.10 | 1.59 |
| `keccak_1024` | 282.38 | 177.64 | 1.59 |
| `keccak_2048` | 569.25 | 364.17 | 1.56 |
| `poseidon_2` | 0.13 | 0.13 | 1.00 |
| `poseidon_4` | 0.18 | 0.19 | 0.95 |
| `poseidon_8` | 0.31 | 0.31 | 1.00 |
| `poseidon_12` | 0.43 | 0.48 | 0.90 |
| `poseidon_16` | 0.63 | 0.62 | 1.02 |
| `ecdsa_32` | 680.90 | 590.26 | 1.15 |
| suite | 2,070.46 | 1,506.62 | **1.37** |

### What was wrong

circom compiles witness time arithmetic on `var`s into `Fr_mod` and `Fr_idiv`. A keccak
round's `(x + 1) % 5` and a rotation's `r % 64` therefore arrive at the field library as
two small integers, and the shipped bodies in `build/fr.cpp` answer each one with three
`mpz_init`/`mpz_clear` pairs and a multi precision division. Counted by a build with
instrumented call sites, `keccak_2048` makes **4,300,800** such calls per witness and
`ecdsa_32` makes **2,027,436**. The divisors are `circuitConstants[11] = 5` and
`circuitConstants[3] = 64`.

Three shapes cover almost all of them. Two small non-negative shorts is the keccak and
sha256 case; a power of two divisor is what `ecdsa_32`'s bigint gadgets divide by
throughout, and the answer is a shift and a mask; a single limb divisor with a single limb
dividend is one `udiv`. Anything wider still goes to GMP. `Fr_toMpz` normalises an element
into [0, q) before dividing and `Fr_fromMpz` tags the result `Fr_SHORT` when it fits in a
signed int, leaving `longVal` alone, so these paths reproduce the shipped representation
and not merely the shipped value.

Measured on this machine, single threaded, best of nine over 262,144 operations:

| operation | ns/op |
| --------- | ----: |
| `Fr_mod(short, 5)` as shipped | 53.69 |
| `Fr_mod(short, 5)` short path | 1.18 |
| `Fr_idiv(short, 5)` as shipped | 55.12 |
| `Fr_idiv(short, 5)` short path | 1.18 |
| `Fr_mod(254 bit, 2^64)` as shipped | 64.61 |
| `Fr_rawMMul`, the montgomery core | 11.78 |
| `Fr_mul`, the wrapper | 11.92 |
| `Fr_add`, two shorts | 1.72 |
| `Fr_copy` | 1.28 |
| `Fr_toLongNormal` | 7.10 |

### The profile before and after

`sample` at 1 ms, leaf attribution, converted to ms of the measured call. The point of
this table is that the patch removed the class it aimed at and left everything else alone.

| class | keccak stock | keccak patched | ecdsa stock | ecdsa patched |
| ----- | -----------: | -------------: | ----------: | ------------: |
| Fr raw asm core | 240.6 | 239.2 | 123.3 | 114.7 |
| malloc/free | 119.8 | **0.1** | 73.5 | 15.2 |
| gmp bignum | 94.4 | **0.0** | 101.2 | 41.5 |
| Fr wrappers | 97.9 | 96.9 | 193.6 | 189.4 |
| circuit code | 6.1 | 11.0 | 97.4 | 104.1 |
| memory move | 2.5 | 3.5 | 80.6 | 85.2 |

On `keccak_2048` the 214.2 ms of malloc and GMP is gone rather than reduced. On `ecdsa_32`
118 ms of 174.7 goes and 56.7 ms stays, which is correct: its dividends are genuine multi
limb values and a 254 bit division is what they need.

### What is left, and what it is worth

The arithmetic floor, from the measured intrinsics and the constraint counts, assuming one
montgomery multiply per non-linear constraint, three add or sub to form each linear
combination, and one 40 byte signal store per signal:

| circuit | floor | patched | over floor |
| ------- | ----: | ------: | ---------: |
| `keccak_2048` | 34.8 ms | 364.17 ms | 10.5x |
| `sha256_2048` | 13.5 ms | 114.76 ms | 8.5x |
| `ecdsa_32` | 53.7 ms | 590.26 ms | 11.0x |

So roughly 10% of what remains is arithmetic the constraints require. The rest is the
interpreter: `keccak_2048` makes about forty `Fr_*` calls per constraint against a minimum
of about four, and every one of them branches on the operand's type tag before doing any
work. That is the `Fr wrappers` row above, 96.9 ms on keccak and 189.4 ms on ecdsa, and it
cannot be fixed from outside the code generator. Emitting typed calls, where a signal
produced by a multiply is known to be `Fr_LONGMONTGOMERY` and a loop variable is known to
be `Fr_SHORT`, is the change that would collect it.

`Fr_rawMMul` is at its floor and is not worth attacking. A four limb CIOS montgomery
multiply needs 2n^2 + n = 36 products, so 68 multiplier pipe instructions, which is 34
cycles on two multiply pipes. It measures 11.78 ns, 43 cycles at 3.68 GHz, which is the
dependent chain latency. The ceiling is 22% and out of order execution already overlaps
independent multiplies.

`ecdsa_32`'s remaining 190 ms of wrappers and 85 ms of `memmove` are a circuit property,
not a witnesscalc one. Its 0xPARC bigint gadgets carry each 64 bit limb in a 40 byte field
element and pass arrays by value across circom function boundaries, which is where
`long_div`, `short_div` and `long_scalar_mult` spend their time.

### Whether a GPU can take this over

No, on these numbers. Level scheduling the witness DAG is the only shape that parallelises,
and the arithmetic it would parallelise is about 35 ms of `keccak_2048`, not 364. The
remainder is type dispatch and per-call overhead that static compilation removes on the CPU
before a GPU sees the problem.

The sync cost decides it. keccak-f is roughly 200 dependent levels per permutation, so the
circuit is about 2,000 levels deep at a width near 3,600. Compute per level is around
0.2 us and traffic around 1.2 us at the 283 GB/s this machine's GPU streams, both
negligible; 2,000 dependent dispatches at 6 to 8 us each is 15 to 20 ms. That is a ceiling
of about 2x over a static CPU evaluator that would have to be written anyway to produce the
DAG.

The published prior art agrees rather than conflicting. ZKPoG (eprint 2025/765) is the only
work that puts witness generation on a GPU, for Plonky2 rather than circom; it reports
14.8x on witness generation against 25.7x for the rest of its pipeline, which leaves
witness generation at 58% of proving, and its fully sequential circuit gets 1.0x. On the
circom side there is nothing: iden3's `evaluate_parallel` is commented out, and mopro
measures `circom-witnesscalc`, the flat DAG evaluator a GPU port would start from, as
slower than witnesscalc on exactly these shapes on Apple silicon (Keccak 69.6 ms against
63.9 ms, SHA256 32 ms against 22 ms).

Two cheap measurements would settle it if it is ever worth revisiting: time 2,000 trivial
dependent dispatches in one Metal command buffer, and level sort the `keccak_2048` witness
DAG for its true depth and width. Nobody has published the second number for a circom
circuit.

### This helps the published row too

The fix is in `build/fr.cpp`, which is checked into zkmopro/witnesscalc and generated by
iden3's ffiasm. It is not specific to this prover: the `circom` row on the page uses the
same witness generator, so adopting it upstream would move their numbers by the same
factor and narrow the gap measured in the sections above. That is the honest reading and
it is an argument for sending it, not for sitting on it.

## What is missing

`blake3` and `poseidon2` are the other two upstream targets and have no circom circuit, so
the page has no circom row for them either. Nothing else on the page is a circom target.

A `mac2.metal` run is missing. Every ratio against the published row here carries a
hardware term that only running the same harness on the same instance type can remove.
