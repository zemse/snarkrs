# Measured costs and floors (Mac CPU)

The operation counts are [the first round's](../workflow-opt-2026-10-11/trace.md); this
file has what was measured again this round and the two floors re-derived, G2's
multiplication and the NTT.

## Intrinsics, re-measured

Single thread, `raw_ops` (eight interleaved chains unless noted), on a machine with one
other process at about 1.1 cores.

| primitive | value | source |
|---|---|---|
| raw.rs Fq mul | 11.33 to 11.42 ns throughput, 16.6 latency | `raw_ops RawFq` |
| raw.rs Fq add / sub | 1.51 / 1.51 ns throughput, 2.26 latency | same |
| Fq2 mul, three fused multiplies (before) | 38.0 to 39.7 ns | `raw_ops`, old code kept beside the new for the A/B |
| Fq2 mul, three wide products and two reductions, own asm (after) | 31.5 ns | `raw_ops RawFq2` |
| the same in portable u128 Rust | 49.6 ns | same, not kept |
| Fq2 square, two fused multiplies | 26.5 ns | probe |
| Fq2 square, lazy (two wide products, two reductions) | 28.2 ns | probe, not kept |
| Fq mul as wide product plus own reduction | 11.83 ns, against 11.42 fused | probe, not kept |
| DIT butterfly over a 64 KB block, tight loop | 15.6 ns (multiply alone 11.8) | probe |
| NTT pipeline, one thread, per radix-2-equivalent butterfly | 17.8 before, 15.8 after | `pipeline_cost` |

The multiply is 41 to 42 cycles at about 3.65 GHz against the 34 its 68 multiplier
operations need; the butterfly's add and sub cost 3.8 ns over it, more than their
throughput alone (3.0), which puts the remaining NTT gap in the field layer, not in the
transform's structure.

## G2 multiplication floor, re-derived

The Fq2 multiply is now 3 wide products (32 multiplier operations each) and 2 reductions
(32 each with the bottom product skipped): 160 operations, 80 cycles, 22 ns at the
intrinsic rate, against 3 x 9.2 = 27.6 before. The batch inversion now runs over norms in
Fq: per element about 6 Fq multiplies (norm 1, chains 3, rebuild 2) against 9 for an Fq2
inversion chain.

| | before | after |
|---|---:|---:|
| Fq2 M (plus its adds) | 32.6 ns | 26 ns |
| doubling: 4 M2 + 2 S2 + inversion share | 228 ns | 202 ns |
| addition: 3 M2 + 1 S2 + inversion share | 196 ns | 154 ns |
| 65 doublings + 43 additions + tables | 26.8 us | about 23 us |
| measured, `glv_cost` batch 1024 | 36.3 to 37.0 us | 28.3 to 28.5 us |

ptau prepare 2^18: 17.4 M G1 x 12.3 us + 4.2 M G2 x 23 us = 311 core-s, 32 s on 9.6 P-core
equivalents (34 s before).

## NTT floor, re-derived

anon-aadhaar's stage is three `intt_coset_ntt` pipelines at n = 2^21 run at once. A
transform is 21 passes of n/2 butterflies (10.5 n butterflies, one add and one sub each);
the multiplies are 10.5 n before and 9.6 n after the radix-8 head, which skips seven of
the twelve multiplies of the first three passes. A pipeline is two transforms and n
coset multiplies.

| | per pipeline | three pipelines, 9.6 cores |
|---|---:|---:|
| intrinsic: 9.2 ns mul, 1.0 ns add/sub | 2 x (9.6 n x 9.2 + 21 n x 1.0) + 9.2 n = 228 n ns = 479 ms | 150 ms |
| measured rates: 11.4 ns mul, 1.5 ns add/sub | 2 x (9.6 n x 11.4 + 21 n x 1.5) + 11.4 n = 293 n ns = 616 ms | 193 ms |
| memory: 11 trips of 128 MB a pipeline | 1.4 GB | about 40 ms at 100 GB/s, overlapped |

Measured (`pipeline_cost`, outputs checked): one thread 696 ms a pipeline (1.13x the
measured-rate floor); three at once 248.7 ms, 8.4 effective cores of 9.6. The previous
round's 154 ms floor is the intrinsic one with radix-2 multiply counts.

Per stage at 2^21, one pipeline, before the packed tables (ms, one thread / twelve):
outer DIF passes 186 / 25.7, the fused middle 402 / 46.5 (351 / 39 after the radix-8
head), outer DIT passes 192 / 27. Every outer radix-4 pair cost 33 to 35 ms on one thread
except the two next to the fused block (twiddle stride 8 KB): 50 and 54 ms, 7.3 against
4.6 ms on twelve. Those two are what the packed tables fixed.

## Prepare and the rest of the ceremony

Unchanged shape: prepare is about 70% G1 multiplications by core time (17.4 M at 20 us)
and 30% G2 (4.2 M at 28 us, 37 before), so the G2 lane moves it by about 5 to 7%. G1's
`mul_batch` is 20 us against 12.3 (15.2 at the measured multiply) and did not move.
