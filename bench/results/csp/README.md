# The ethproofs client-side-proving benchmark: first local run

[ethproofs.org/csp-benchmarks](https://ethproofs.org/csp-benchmarks) publishes six numbers
per circuit for seventeen proving systems. One of the seventeen is `circom`, which is
circom + witnesscalc + rapidsnark. This run keeps the first two and swaps the third for
this prover, across fifteen variants. The protocol, and the two places it deliberately
departs from upstream, are in [`bench/csp/README.md`](../../csp/README.md) and are not
restated here.

**The number worth taking away is 1.35x to 1.59x, not 5x.** Both are below. The first is
against rapidsnark run on this machine, over these keys, in this loop. The second is
against the row the page currently serves, which was measured on other hardware against
circuits 1.74x larger. The second is the one that will get quoted and it is the one that is
wrong.

## Method

| | |
|---|---|
| machine | Apple M2 Max, 12 cores, 96 GiB, macOS 27.0 |
| date | 2026-09-24 |
| statistic | one warm-up, then the mean of 10 timed iterations; median, min and max kept alongside |
| zkeys | the published `zkeys-v2` release, checksum verified, not regenerated locally |
| witness | witnesscalc, in process, inside the timed region, identical for all three provers |
| provers | `g16` cpu, `g16` metal, rapidsnark; one loop, only the prover swapped |
| variants | sha256 and keccak at 128 to 2048 bytes, poseidon at 2 to 16 field elements |
| raw output | `bench/results/csp/metrics/*.json`, untracked, which is why the numbers are in this file |

Every duration below is milliseconds unless the column header says otherwise. Every one is
a full `proof_duration` in the page's sense: witness generation, the zkey read, and the
proof, because that is what the protocol times.

## rapidsnark on the same machine

This is the only comparison in this file with one variable in it. Same box, same zkeys, same
in-process witness buffer, same loop, prover swapped.

| variant | constraints | rapidsnark | g16 cpu | g16 metal | rs / cpu | rs / metal |
| ------- | ----------: | ---------: | ------: | --------: | -------: | ---------: |
| `sha256_128` | 53,048 | 104.41 | 71.09 | **63.10** | 1.47x | 1.65x |
| `sha256_256` | 89,224 | 174.78 | 127.13 | **92.97** | 1.37x | 1.88x |
| `sha256_512` | 161,576 | 346.84 | 239.16 | **179.98** | 1.45x | 1.93x |
| `sha256_1024` | 306,280 | 740.26 | 464.75 | **280.12** | 1.59x | 2.64x |
| `sha256_2048` | 595,688 | 1,293.66 | 956.80 | **480.49** | 1.35x | 2.69x |
| `keccak_128` | 93,184 | 202.26 | 143.80 | **116.11** | 1.41x | 1.74x |
| `keccak_256` | 187,328 | 420.56 | 275.30 | **204.39** | 1.53x | 2.06x |
| `keccak_512` | 375,744 | 816.63 | 556.45 | **349.57** | 1.47x | 2.34x |
| `keccak_1024` | 752,576 | 1,635.86 | 1,164.47 | **666.94** | 1.40x | 2.45x |
| `keccak_2048` | 1,506,240 | 3,669.14 | 2,311.01 | **1,277.14** | 1.59x | 2.87x |
| `poseidon_2` | 517 | 11.03 | **5.15** | 20.69 | 2.14x | 0.53x |
| `poseidon_4` | 736 | 11.61 | **5.56** | 19.80 | 2.09x | 0.59x |
| `poseidon_8` | 1,171 | 14.45 | **7.63** | 20.93 | 1.89x | 0.69x |
| `poseidon_12` | 1,613 | 15.69 | **8.37** | 21.40 | 1.87x | 0.73x |
| `poseidon_16` | 2,092 | 18.74 | **15.77** | 34.94 | 1.19x | 0.54x |

On the ten hash circuits the CPU backend is 1.35x to 1.59x of rapidsnark and shows no trend
with size. Metal is 1.65x to 2.87x and does trend: 1.65x at `sha256_128`, 2.87x at
`keccak_2048`, monotone in between on both targets. That is the expected shape, because the
fixed cost Metal pays per proof is the same at every size.

Means and medians agree everywhere except one cell. `poseidon_16` on the CPU has a mean of
15.77 ms against a median of 11.64 ms and a minimum of 10.47 ms, because one of the ten
iterations took 33.33 ms. On the median that row is 1.59x rather than 1.19x. The page
publishes a mean, so the table carries the mean; this is the single cell where it misleads.

## Against the published circom row

At face value we are 4.37x to 5.14x the published `circom` row on the hash circuits and
2.02x to 2.85x on poseidon. Neither is a prover result, for two reasons that are both
quantifiable from the data.

**The circuits shrank.** The published numbers were taken against `zkeys-v1`. Their
`sha256_128` is 94,976 constraints; the one in this run is 53,048. Their `keccak_128` is
163,638 against 93,184. The ratio is 1.74x to 1.79x at every hash size. Poseidon did not
change at all, which is visible in the same file: all five poseidon constraint counts are
identical on both sides, and so are all five zkey sizes to the byte. That is why the
poseidon face-value ratio needs no correction and the hash one does.

**The machine differed.** The page measures on an AWS `mac2.metal`, an Apple M1 with 8
cores and 16 GB. This is an M2 Max with 12 cores.

Normalising by constraint count removes the first term:

| variant | our constraints | published constraints | published ms | g16 cpu ms | face value | per constraint |
| ------- | --------------: | --------------------: | -----------: | ---------: | ---------: | -------------: |
| `sha256_128` | 53,048 | 94,976 | 355.71 | 71.09 | 5.00x | 2.79x |
| `sha256_256` | 89,224 | 158,656 | 635.82 | 127.13 | 5.00x | 2.81x |
| `sha256_512` | 161,576 | 286,016 | 1,209.97 | 239.16 | 5.06x | 2.86x |
| `sha256_1024` | 306,280 | 540,736 | 2,389.19 | 464.75 | 5.14x | 2.91x |
| `sha256_2048` | 595,688 | 1,050,176 | 4,810.00 | 956.80 | 5.03x | 2.85x |
| `keccak_128` | 93,184 | 163,638 | 655.26 | 143.80 | 4.56x | 2.59x |
| `keccak_256` | 187,328 | 327,244 | 1,272.34 | 275.30 | 4.62x | 2.65x |
| `keccak_512` | 375,744 | 654,456 | 2,513.49 | 556.45 | 4.52x | 2.59x |
| `keccak_1024` | 752,576 | 1,308,880 | 5,088.06 | 1,164.47 | 4.37x | 2.51x |
| `keccak_2048` | 1,506,240 | 2,617,728 | 10,269.19 | 2,311.01 | 4.44x | 2.56x |
| `poseidon_2` | 517 | 517 | 14.66 | 5.15 | 2.85x | 2.85x |
| `poseidon_4` | 736 | 736 | 15.34 | 5.56 | 2.76x | 2.76x |
| `poseidon_8` | 1,171 | 1,171 | 20.81 | 7.63 | 2.73x | 2.73x |
| `poseidon_12` | 1,613 | 1,613 | 23.10 | 8.37 | 2.76x | 2.76x |
| `poseidon_16` | 2,092 | 2,092 | 31.81 | 15.77 | 2.02x | 2.02x |

The second term is measurable too, because rapidsnark is on both sides of it. Per
constraint, the published rapidsnark is 1.61x to 2.11x the rapidsnark measured here on the
hash circuits. Multiply that by the 1.35x to 1.59x prover gap from the previous section and
the 2.51x to 2.91x per-constraint column comes back. So of a headline near 4.5x, roughly
1.75x is the circuit, roughly 1.8x is the machine, and roughly 1.45x is the prover.

Two smaller columns, for completeness. `verify_duration` reproduces the published shape
(12.0 ms to 268.0 ms) only because the protocol re-reads the zkey and derives the verifier
from it; the honest verifier cost is `verify_vkey_ms` in the breakdown files, which is 1.12
ms to 1.19 ms across all fifteen variants and does not move with circuit size. `proof_size`
is 818 to 990 bytes here against 918 to 1,020 published; the difference is in the JSON, not
the proof, which is two G1 points and one G2 point either way.

## Where the time goes

Witness generation plus the zkey read is 20.0% to 37.3% of every CPU measurement, and the
fraction does not trend away with size:

| variant | witness | zkey read | prepare | prove | total | witness + zkey |
| ------- | ------: | --------: | ------: | ----: | ----: | -------------: |
| `sha256_128` | 13.42 | 11.01 | 0.84 | 45.82 | 71.09 | 34.4% |
| `sha256_256` | 22.73 | 17.59 | 0.97 | 85.85 | 127.13 | 31.7% |
| `sha256_512` | 40.43 | 30.53 | 1.53 | 166.68 | 239.16 | 29.7% |
| `sha256_1024` | 77.35 | 54.62 | 2.60 | 330.19 | 464.75 | 28.4% |
| `sha256_2048` | 149.34 | 105.15 | 4.78 | 697.53 | 956.80 | 26.6% |
| `keccak_128` | 29.57 | 20.45 | 1.11 | 92.66 | 143.80 | 34.8% |
| `keccak_256` | 64.50 | 37.37 | 1.67 | 171.76 | 275.30 | 37.0% |
| `keccak_512` | 136.23 | 71.25 | 3.09 | 345.88 | 556.45 | 37.3% |
| `keccak_1024` | 282.39 | 137.00 | 5.74 | 739.34 | 1,164.47 | 36.0% |
| `keccak_2048` | 569.79 | 266.78 | 10.25 | 1,464.19 | 2,311.01 | 36.2% |
| `poseidon_2` | 0.19 | 1.00 | 0.17 | 3.79 | 5.15 | 23.0% |
| `poseidon_4` | 0.27 | 1.02 | 0.18 | 4.10 | 5.56 | 23.1% |
| `poseidon_8` | 0.44 | 1.08 | 0.20 | 5.90 | 7.63 | 20.0% |
| `poseidon_12` | 0.57 | 1.17 | 0.20 | 6.43 | 8.37 | 20.8% |
| `poseidon_16` | 0.93 | 2.42 | 0.22 | 12.20 | 15.77 | 21.3% |

This is the reason a proving-only number cannot be put next to this page. Halving the CPU
prove time on `keccak_2048` would move the published total by 32%, not by 50%, and driving
prove to zero would still leave 846.8 ms on the clock.

It is also the ceiling on what a GPU can deliver end to end:

| variant | witness | zkey read | prepare | prove | total | witness + zkey |
| ------- | ------: | --------: | ------: | ----: | ----: | -------------: |
| `sha256_128` | 13.39 | 10.98 | 10.61 | 28.12 | 63.10 | 38.6% |
| `sha256_256` | 22.60 | 17.47 | 16.65 | 36.25 | 92.97 | 43.1% |
| `sha256_512` | 41.01 | 30.45 | 28.64 | 79.88 | 179.98 | 39.7% |
| `sha256_1024` | 77.43 | 55.45 | 52.70 | 94.53 | 280.12 | 47.4% |
| `sha256_2048` | 151.28 | 105.33 | 99.95 | 123.93 | 480.49 | 53.4% |
| `keccak_128` | 29.18 | 19.93 | 17.64 | 49.35 | 116.11 | 42.3% |
| `keccak_256` | 65.61 | 37.27 | 32.17 | 69.34 | 204.39 | 50.3% |
| `keccak_512` | 136.44 | 70.24 | 58.20 | 84.69 | 349.57 | 59.1% |
| `keccak_1024` | 281.75 | 136.01 | 111.08 | 138.10 | 666.94 | 62.6% |
| `keccak_2048` | 571.27 | 266.15 | 211.61 | 228.10 | 1,277.14 | 65.6% |
| `poseidon_2` | 0.23 | 0.92 | 2.86 | 16.68 | 20.69 | 5.5% |
| `poseidon_4` | 0.28 | 1.12 | 3.12 | 15.27 | 19.80 | 7.1% |
| `poseidon_8` | 0.56 | 1.19 | 3.06 | 16.12 | 20.93 | 8.4% |
| `poseidon_12` | 0.68 | 1.26 | 2.96 | 16.49 | 21.40 | 9.1% |
| `poseidon_16` | 1.17 | 1.50 | 3.22 | 29.06 | 34.94 | 7.6% |

Take `sha256_1024`. Metal's prove is 94.53 ms against the CPU's 330.19 ms, 3.49x. The totals
are 280.12 ms against 464.75 ms, 1.66x. The gap is witness (77.43 ms), the zkey read
(55.45 ms), and the host-to-device upload the Metal backend charges to prepare (52.70 ms),
none of which the GPU shortens. `keccak_2048` is starker: 6.42x on prove, 1.81x on total,
with witness and zkey read at 65.6% of the measurement. If prepare and prove both went to
zero on that variant, the total would improve by 1.53x and no further.

**Metal loses to the CPU backend on every poseidon size, and to rapidsnark as well.** It is
2.22x to 4.02x slower than the CPU backend and 1.36x to 1.88x slower than rapidsnark. The
cause is in the prepare and prove columns: they sum to 19.54 ms at `poseidon_2` and 19.46 ms
at `poseidon_12`, a circuit with three times the constraints. A cost that flat is dispatch
and upload, not arithmetic. At 517 to 2,092 constraints there is nothing to amortise it
over, and the CPU backend finishes `poseidon_12`'s prove in 6.43 ms. This is the correct
outcome for a GPU at this size, not a defect.

## Memory

The one column where we lose.

| variant | g16 cpu | g16 metal | published circom | cpu B/constraint | metal B/constraint | published B/constraint |
| ------- | ------: | --------: | ---------------: | ---------------: | -----------------: | ---------------------: |
| `sha256_128` | 105.0 | 148.3 | 162.3 | 1,979 | 2,795 | 1,708 |
| `sha256_256` | 176.6 | 238.9 | 249.9 | 1,979 | 2,678 | 1,575 |
| `sha256_512` | 311.3 | 426.1 | 452.1 | 1,927 | 2,637 | 1,581 |
| `sha256_1024` | 588.4 | 794.4 | 646.5 | 1,921 | 2,594 | 1,196 |
| `sha256_2048` | 1,120.6 | 1,507.5 | 1,352.4 | 1,881 | 2,531 | 1,288 |
| `keccak_128` | 190.7 | 269.0 | 246.6 | 2,046 | 2,887 | 1,507 |
| `keccak_256` | 370.1 | 512.9 | 473.2 | 1,976 | 2,738 | 1,446 |
| `keccak_512` | 726.9 | 992.5 | 722.0 | 1,935 | 2,641 | 1,103 |
| `keccak_1024` | 1,422.7 | 1,935.9 | 1,377.9 | 1,890 | 2,572 | 1,053 |
| `keccak_2048` | 2,804.7 | 3,815.4 | 2,679.3 | 1,862 | 2,533 | 1,024 |
| `poseidon_2` | 14.8 | 31.4 | 17.8 | 28,655 | 60,713 | 34,422 |
| `poseidon_4` | 15.6 | 32.4 | 18.9 | 21,237 | 43,959 | 25,660 |
| `poseidon_8` | 17.7 | 34.2 | 20.2 | 15,112 | 29,165 | 17,229 |
| `poseidon_12` | 19.0 | 36.1 | 21.9 | 11,762 | 22,382 | 13,556 |
| `poseidon_16` | 21.6 | 38.5 | 24.7 | 10,333 | 18,391 | 11,793 |

Across the ten hash circuits the CPU backend peaks at 1,862 to 2,046 bytes per constraint
against the published 1,024 to 1,708, so 1.16x to 1.82x more once normalised. In absolute
terms it is worse than the normalised figure suggests: `keccak_2048` peaks at 2.61 GiB here
against the published 2.50 GiB, on a circuit with 1.74x fewer constraints. Poseidon runs the
other way, below the published figure at all five sizes, but at 517 constraints that column
is measuring process overhead rather than the prover.

Metal costs another 1.35x to 1.41x on the hash circuits and 1.78x to 2.12x on poseidon,
because the device buffers sit on top of the host copies the zkey parse already made and
nothing is released between the parse and the upload.

It still fits the box the page uses. The worst case in the table is `keccak_2048` on Metal
at 3,815 MB, 3.55 GiB, on a 16 GB `mac2.metal`. The published circom row peaked at 2.50 GiB
there on a circuit 1.74x larger.

Two caveats on this table. `peak_memory` for our rows comes from a separate sampling process
rather than the timed loop. rapidsnark has no column at all: the sampler links our prover, so
those rows carry the collector's 0 marker instead of a number taken from the wrong process.

## Cross-checks

**Constraint counts.** All ten sha256 and keccak counts match the upstream circuit table in
[`bench/vendor/csp-benchmarks/circom/README.md`](../../vendor/csp-benchmarks/circom/README.md)
exactly: 53,048 / 89,224 / 161,576 / 306,280 / 595,688 for sha256 and 93,184 / 187,328 /
375,744 / 752,576 / 1,506,240 for keccak. The five poseidon counts are not in that table;
they match the published circom rows exactly at 517 / 736 / 1,171 / 1,613 / 2,092, which is
also the evidence that poseidon did not change between `zkeys-v1` and `zkeys-v2`.

**Key bytes.** `preprocessing_size` reproduces the published value to the byte on all five
poseidon variants, `poseidon_16` at 22,286,842. The hash variants differ, which is the v1 to
v2 shrink again rather than a mismatch: 1,152,894,061 for `keccak_2048` against the published
1,339,459,874.

**Verification.** Each variant's warm-up proof is verified before any timing is recorded,
with the verifying key derived from the published zkey, and the run aborts on a rejection.
All forty-five rows (fifteen variants, three provers) cleared that gate, so our verifier
accepts rapidsnark's proofs over the published keys as readily as its own.

**An independent verifier, but only by hand.** The check above uses our verifier on both
sides, so it cannot catch an error we make in both directions at once. During bring-up
`snarkjs groth16 verify` accepted our Metal proofs against the published verification keys
for `poseidon_2`, `sha256_128` and `keccak_128`, which is the evidence that the encoding is
snarkjs' and not merely self-consistent. That check is three variants and a manual command,
not something the harness runs, and until it is wired in the other twelve rest on our
verifier alone.

## What is missing

`ecdsa` is the fourth circom target on the page and there is no row for it here. The fifteen
variants above are every one that had a measurement when these numbers were taken.
`blake3` and `poseidon2`, the other two upstream targets, have no circom circuit, so the
page has no circom row for them either.

A `mac2.metal` run is also missing. Every ratio against the published row here carries a
hardware term that only running the same harness on the same instance type can remove.
