# Lanes, one at a time, each against its own baseline

Phase 4 of the [optimise-by-intrinsic](README.md) pass. Every lane is an interleaved A/B with
`bench/scripts/arkab.py`, three rounds, the variant order reversed each round, against the
binary of the commit before it, so each delta is that lane's alone. Prove is the median of 30
self-verified warm proofs; every other cell is the median of three round values. `*` marks a
round in which another process used three cores or more (the Mac was shared with other
sessions all day, see the README); where a round was disturbed the clean rounds are quoted.

## L1 GLV on raw limbs (70ed41f, 2db989d, c90a54c, 0f05068)

`glv::mul`: `k = k1 + k2 lambda`, both halves below 2^127, two width-5 wNAF ladders sharing
their doublings, on the XYZZ raw-limb arithmetic the MSM already uses. Used for prepare's
twiddles (a 254-bit wNAF before), the key scaling of contribute and the beacons, setup's
below-MSM slots and verify's public-input sum (all arkworks' affine double-and-add before,
which walks every negative coefficient as a 254-bit scalar).

| workflow | base | glv | change |
|---|---:|---:|---:|
| ptau contribute 2^13 / 2^19 (ms) | 681.7 / 43,474 | 394.0 / 24,201 | -42% / -44% |
| ptau prepare 2^13 / 2^18 (ms) | 5,900 / 243,393 | 3,718 / 154,894 | -37% / -36% |
| groth16 setup js_2x2_d32 / js_16x16_d32 / anon-aadhaar, round 1 (ms) | 2,279 / 17,629 / 28,086 | 1,323 / 10,065 / 4,284 | -42% / -43% / -85% |
| zkey contribute, same, round 1 (ms) | 555 / 4,041 / 31,024 | 288 / 1,938 / 15,477 | -48% / -52% / -50% |
| verify js_16x16_d32 / railgun-13x01 / anon-aadhaar (ms) | 4.536 / 2.952 / 1.745 | 2.929 / 2.166 / 1.606 | -35% / -27% / -8% |

Rounds 2 and 3 ran while the Mac was being taken by other work (round 3 everything 2.5x
slower for both builds), so the phase-2 cells quote round 1. 10 of 10 deterministic outputs
byte-identical; ptau verify and zkey verify accepted every output.

## L2 prepare: fold 1/n into the last pass (a801d78)

The last pass computes `[s] lo +- [s w^j] hi`, so the `1/n` costs `n/2` multiplications instead
of `n`. Prepare 2^13 3,295 to 3,078 ms (-6.6%), 2^18 135,826 to 127,159 ms (-6.4%), all three
rounds in the same direction; outputs byte-identical.

## L3 NTT butterflies on RawFr (a417ee1)

With raw.rs's multiply as it was (19.4 ns against arkworks' 17.2), only the branch-free add and
sub paid: NTT stage -2% to -5% (anon-aadhaar 382.9 to 373.7 ms), whole proof within noise
(-2.3% to +0.8%). Kept because the field lane below makes RawFr the faster path; its NTT
gain is counted there.

## L4 phase-1 key scaling in sixteen-chunk groups (416759a)

Phase 1 called KeyScale per response chunk (16384 G1 or 8192 G2 points) and CpuKeyScale split
that into 1024-point tasks: 16 or 8 tasks on a 12-thread pool, then a serial write and hash.
Now sixteen chunks per call and 256-point tasks; the hash still takes one response chunk per
update. contribute 2^19 20,397 to 16,361 ms (-19.8%), 2^13 -11.3%; zkey contribute -5.5% to
-10.2%. Beacons at 2^13 and 2^19 byte-identical, which covers the group boundary.

## L5 batch-affine GLV, four-way on G2 (d8a906d, f16cf0e, 09c65ef, 7cea0d2, 352d0be)

`glv::mul_batch` runs a task's multiplications in lockstep in affine coordinates with one
shared inversion a step (affine doubling 3M + 2S and addition 2M + S plus three multiplications
of inversion share, against XYZZ's 6M + 3S and 12M + 2S). G2 adds `psi` (`[6x^2]` on G2) to `phi`
and splits scalars four ways over an LLL-reduced basis of 64-bit vectors: 65 steps instead of
127. Single thread (`glv_cost`, us per point): G1 35.5 (`glv::mul`) to 28.5 at 1024 a batch,
G2 135.7 to 53.9.

Round 1 against L4 (rounds 2 and 3 had other processes on 4 to 12 cores): contribute 2^19
17,482 to 11,136 ms (-36%), zkey contribute anon-aadhaar 13,970 to 11,703 (-16%),
js_16x16_d32 1,700 to 1,487 (-13%), prepare 2^18 137,866 to 118,726 (-14%), but prepare 2^13
3,287 to 4,703 ms (+43%, all rounds). The 2^13 regression and most of the missing 2^18 gain
were one bug, found by sampling the process: the rewritten pass walked a pass's groups one
after another whenever a group was larger than a task, so a pass of 64 groups of 4096 ran two
tasks at a time. Fixed in 46d4823 (L7).

The first version of the batch kept each point's whole state in one 2.5 to 4.6 KB struct and
scanned every struct on every ladder step; it was replaced by step-major digit rows (7cea0d2)
before it was measured end to end.

## L6 aarch64 assembly for raw.rs (4ec7f10)

LLVM will not emit a subs/sbcs chain against a constant modulus; it adds the complement and
carries borrows through cinc/sbc or cset/csinc pairs, and in the multiply it rematerialises the
modulus inside the loop. On aarch64 add, sub and the multiply's reduction are now one carry
chain, one borrow chain and four csel, and the multiply is gnark-crypto's arm64 `mul`.
Portable code everywhere else (x86_64 codegen unchanged, IFMA path untouched).

`raw_ops`, single thread, ns (dependent chain / 8 chains / random stream):

| op | before | after |
|---|---|---|
| Fq add | 6.43 / 3.30 / 3.40 | 2.72 / 1.84 / 1.81 |
| Fq sub | 5.39 / 2.49 / 2.64 | 2.71 / 1.84 / 1.83 |
| Fq mul | 25.7 / 19.4 / 19.6 | 20.4 / 13.9 / 13.9 |
| Fr add / sub / mul | same within 0.1 ns as Fq | |
| arkworks Fq add / mul, for scale | 3.16 / 2.51 / 6.00 ; 19.2 / 17.2 / 17.4 | |

Against L5, three clean rounds:

| workflow | L5 | L6 | change |
|---|---:|---:|---:|
| prove, nine circuits (ms) | js_2x2_d32 69.1 ... anon-aadhaar 1,338 | 51.2 ... 996 | -24.0% to -26.3% |
| anon-aadhaar NTT / MSM (ms) | 401.5 / 918.8 | 293.9 / 686.6 | -27% / -25% |
| prepare 2^13 / 2^18 (ms) | 4,690 / 118,443 | 3,143 / 88,936 | -33% / -25% |
| contribute 2^13 / 2^19 (ms) | 331 / 11,106 | 255 / 8,513 | -23% / -23% |
| setup js_2x2_d32 / js_16x16_d32 / anon-aadhaar (ms) | 1,208 / 9,275 / 3,737 | 862 / 6,562 / 3,305 | -29% / -29% / -12% |
| zkey contribute, same (ms) | 223 / 1,486 / 11,487 | 178 / 1,158 / 9,150 | -20% / -22% / -20% |
| ptau verify 2^19, zkey verify anon-aadhaar (ms) | 2,265, 2,894 | 1,829, 2,323 | -19%, -20% |
| verify (ms) | js_16x16_d32 2.380 | 2.055 | -0.2% to -13.6% |

540 self-verified proofs, 11 of 11 deterministic outputs byte-identical, snarkjs 0.7.6 26 of
26.

Rejected in the same lane, measured with `raw_ops`:

- Row-shaped multiply in pure Rust (the arkworks lane's shape): 16.1 ns throughput against
  19.4, but the ceremony crate's `cargo test --release` build ran for over 20 minutes without
  finishing (killed), against 45 s with the assembly and 3 minutes with the old code. Not
  shippable.
- One-bit-carry (`overflowing_add` pair) add and sub in pure Rust: add throughput 2.90,
  stream 3.81 (worse than 3.40), sub stream 3.01 (worse than 2.64); LLVM vectorised the stream
  loop and kept cset/csinc pairs.
- `a + b` beside `a - (m - b)` (two parallel chains, the latency idea from the arkworks lane),
  in Rust: no better than above; in assembly: latency 2.52 against 2.72, throughput 2.36
  against 1.84. The serial shape is better in two of three modes, so it is the one shipped and
  A/B'd end to end.
- Sub written as add of the complement in Rust: LLVM still emits cmp/ccmp/cset/csinc.

## L7 working-memory reuse, small-pass tasks, parallel groups (327138a, 46d4823)

`BatchState` keeps a task's tables between calls (4.6 MB a G2 batch), small passes keep
32-butterfly tasks, and the pass runs its groups in parallel as well as the tasks inside
them. The first two alone were measured against L6 and moved nothing beyond noise (prepare
2^18 89,009 to 88,633 ms, zkey contribute anon-aadhaar -2.8%). With the parallel groups,
interleaved single runs: prepare 2^13 2,990 to 2,050 ms (three rounds), 2^18 85,440 to
64,330 ms; the final A/B in the README has the three-round numbers.
