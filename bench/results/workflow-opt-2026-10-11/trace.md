# Computation trace, floors and headroom per workflow (Mac CPU)

Phase 1 and 2 of the [optimise-by-intrinsic](README.md) pass: what each workflow asks of the
machine, counted without timing, then the floor those counts imply on this M2 Max. Counts come
from `crates/ceremony/examples/ceremony_trace.rs` (ceremony) and
`crates/groth16/examples/cpu_trace.rs` (prove); both print counts only.

## Intrinsics used

| primitive | value | source |
|---|---|---|
| clock under load, P core | 3.62 to 3.69 GHz | algebra `perf-notes/INTRINSICS.md` |
| mul / umulh | 2 a cycle, 3 cycle latency | same |
| 4-limb Montgomery mul, mul-pipe bound | 68 mul ops / 2 = 34 cyc = 9.2 ns | same, FLOOR.md |
| best known mul sequence (gnark arm64 asm) | 10.1 ns throughput | same |
| raw.rs Fq mul before / after (this harness) | 19.4 / 13.9 ns throughput, 25.7 / 20.4 ns latency | `crates/field/examples/raw_ops.rs` |
| raw.rs Fq add before / after | 3.30 / 1.84 ns throughput, 6.43 / 2.72 ns latency | same |
| 4-limb modular add, flag-op bound | about 1.0 ns | algebra FLOOR.md |
| cores | 8 P + 4 E; counted as 9.6 P-core equivalents (E at 0.4) | assumption, see below |

`raw_ops` reads its operands from memory every iteration, so its mul runs the gnark sequence
at 13.9 ns where the algebra harness, with operands in registers, measured 10.1 ns. Floors below
use the intrinsic bounds (9.2 ns mul, 1.0 ns add), not either harness. The 0.4 weight for an
E core is an assumption, not a measurement; `cpu_intrinsics` with `PROBE_ALL_CORE=1` is the
probe that would replace it.

Every operand set below is L1/L2 resident except the NTT and MSM vectors, and those segments
are compute bound at the measured rates (the NTT's outer passes move 3.8 GB on anon-aadhaar,
about 30 ms at 130 GB/s, against 150+ ms of arithmetic), so every floor here is a compute floor.

## Per-operation floors

| operation | count | floor |
|---|---|---|
| G1 variable-base multiplication, 254-bit scalar, batch-affine GLV | 127 doublings (7M + 5A), about 49 additions (6M + 6A), table 1D + 7A | 12.3 us |
| G2 variable-base multiplication, batch-affine four-way GLV-GLS | 65 doublings, about 43 additions in Fq2 (M = 3 Fq M + 5 A), tables | 26.8 us |
| NTT butterfly (Fr) | 1 mul + 1 add + 1 sub | 11.2 ns |
| MSM bucket addition, batch affine (G1) | 6M + 6A | 61 ns |

## Workflows

### ptau prepare phase2 (power 18)

Four sections, blocks `2^0..2^19` (section 12) or `2^0..2^18`, each block an inverse FFT over
the group with one full scalar multiplication per nontrivial twiddle, then (before the fold
lane) `n` more for `1/n`:

| | G1 | G2 |
|---|---:|---:|
| scalar multiplications | 18,350,138 | 4,456,467 |
| doublings, 254-bit wNAF (before) | 4.63 G | 1.12 G |
| doublings, two-way GLV | 2.23 G | 0.54 G |
| butterfly additions | 36.7 M | 8.9 M |

The fold of `1/n` into the last pass removes about 5% of the multiplications (8.5n instead of
9n per 2^18 block). Floor: 17.4 M G1 x 12.3 us + 4.2 M G2 x 26.8 us = 327 core-s, 34 s on 9.6
cores.

### ptau contribute (power 19)

One multiplication per point, each by its own scalar `first * inc^i`: 2,097,151 G1 (sections 2,
4, 5) and 524,289 G2 (sections 3, 6). Double-and-add (arkworks' affine path, before): 528 M G1
and 132 M G2 doublings; two-way GLV: 263 M and 66 M. Floor: 2.10 M x 12.3 us + 0.52 M x 26.8
us = 39.8 core-s, 4.1 s, plus about 0.5 s of serialisation, BLAKE2b over 128 MB and file I/O:
4.6 s.

### groth16 setup

Per signal, the slot's terms summed; slots of 32 terms or more go to the MSM, the rest are
one multiplication per term. Most coefficients are 1, -1 or small, which arkworks' affine
multiply walks as 254-bit scalars when negative:

| circuit | small-slot multiplications | doublings, double-and-add | doublings, GLV |
|---|---:|---:|---:|
| js_2x2_d32 | 188 k | 37.5 M | 13.3 M |
| js_16x16_d32 | 1.47 M | 294 M | 104 M |
| anon-aadhaar | 8.56 M | 736 M | 26.8 M |

anon-aadhaar's small-slot coefficients: 6.20 M within one bit of 0 or r, 1.77 M of 2 to 16 bits,
0.51 M of 17 to 64, 71 k of 65 to 128, 13 k longer.

### zkey contribute

Sections 8 and 9 scaled by one constant scalar: `nVars - nPublic - 1` plus `domainSize` G1
points, 3.2 M on anon-aadhaar. Floor 3.2 M x 12.3 us = 39 core-s, 4.1 s, plus I/O.

### prove (anon-aadhaar, domain 2^21)

| segment | count | floor |
|---|---|---:|
| NTT, three iNTT-coset-NTT pipelines | 132 M radix-2-equivalent butterflies | 154 ms |
| MSM | H: 2^21 dense G1 scalars, 16 windows of c = 16 (33.5 M bucket additions); A, L, B G1, B G2: 53 k, 87 k, 15 k, 15 k general scalars plus 273 k, 404 k, 224 k, 224 k ones | about 40 M G1-addition equivalents, 254 ms |
| gather, pointwise, assemble | 2.2 M Fr products, 2.1 M subtractions | under 10 ms |

### verify

One four-pair multi-Miller loop and one final exponentiation, all arkworks Fq/Fq2/Fq12 code,
plus `n_public` G1 multiplications (34 on js_16x16_d32). The pairing is the floor-relevant
part; the arkworks perf lane measured 119 us for one Miller loop and 192 us for the final
exponentiation with a faster field, so about 0.5 ms is a fair floor for the whole check.
