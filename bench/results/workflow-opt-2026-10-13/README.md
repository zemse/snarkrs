# G2's field arithmetic, the NTT, and main on perf arkworks, Mac CPU, 2026-10-13

The third optimise-by-intrinsic pass over snarkrs on the M2 Max, CPU backend, picking up
[the previous round](../workflow-opt-2026-10-12/README.md)'s two largest leftovers: `ptau
prepare`'s G2 side (and the contribute steps on the same kernel), and the proving NTT. Then
one combined A/B answering whether main built on perf arkworks gets both its own and perf
arkworks' gains. Branch `worktree-agent-a62135994a0d39a2f`, 4 code commits on 586f42d. No
dependency or lockfile change; the perf-arkworks variant is a build-time patch only.

- [trace.md](trace.md): intrinsics re-measured, the G2 and NTT floors re-derived.
- [lanes.md](lanes.md): each change with its own A/B, and what was tried and rejected.
- [ab-reports.md](ab-reports.md): the final CPU A/B (three variants) and the Metal one.

## Start vs end

`base` is 586f42d on stock arkworks (7ad88c4), `final` is 825abfc on stock arkworks,
`finalperf` is 825abfc on perf arkworks (651414b, `perf/bn254-field-0.5`). One interleaved
A/B with `bench/scripts/arkab.py`, three rounds, variant order rotating each round; prove is
the median of 30 self-verified warm proofs, every other cell the median of three wall clocks
counted only on exit 0 (and the verifiers' OK line).

| circuit | prove, ms (base / final / finalperf) | verify, ms | setup, ms | zkey verify, ms |
|---|---|---|---|---|
| js_2x2_d32 | 44.8 / 42.2 / 42.4 | 1.302 / 1.298 / 0.921 | 489 / 469 / 465 | 56.7 / 56.7 / 53.1 |
| railgun-01x01 | 36.5 / 34.9 / 34.9 | 1.272 / 1.270 / 0.896 | 72.6 / 70.2 / 76.0 | 56.9 / 57.0 / 54.2 |
| tornado | 57.6 / 55.7 / 55.4 | 1.242 / 1.243 / 0.870 | 202 / 186 / 192 | 59.9 / 59.9 / 58.2 |
| sha256 | 30.4 / 29.2 / 28.7 | 1.200 / 1.197 / 0.822 | 514 / 454 / 463 | 98.4 / 98.2 / 96.4 |
| railgun-13x01 | 209.9 / 198.6 / 198.3 | 1.584 / 1.583 / 1.200 | 382 / 361 / 364 | 276.8 / 274.9 / 267.7 |
| js_16x16_d32 | 265.9 / 255.5 / 253.4 | 2.028 / 2.027 / 1.631 | 3,066 / 2,901 / 2,943 | 275.0 / 278.3 / 271.4 |
| rsa2048 | 114.5 / 111.0 / 110.5 | 1.512 / 1.516 / 1.133 | 675 / 643 / 643 | 287.8 / 285.5 / 282.9 |
| keccak256 | 107.4 / 104.5 / 105.3 | 1.197 / 1.198 / 0.823 | 341 / 325 / 308 | 324.6 / 323.1 / 319.8 |
| anon-aadhaar | 785.3 / 748.4 / 737.7 | 1.256 / 1.247 / 0.876 | 2,593 / 2,340 / 2,317 | 1,832 / 1,845 / 1,789 |

| item | ptau contribute, ms | ptau verify, ms | ptau prepare phase2, ms |
|---|---|---|---|
| 2^13 | 222.1 / 211.8 / 207.5 | 57.6 / 57.2 / 49.9 | 1,884 / 1,755 / 1,759 |
| 2^19 (prepare: 2^18) | 7,365 / 6,827 / 6,889 | 1,806 / 1,707 / 1,712 | 57,992 / 54,719 / 51,952 |

anon-aadhaar's prove stages, ms (base / final / finalperf): NTT 266.0 / 229.7 / 226.5
(-13.6% / -14.8%), MSM 501.9 / 500.7 / 497.4, gather 8.0 / 9.7 / 7.8, pointwise 4.8 / 5.1 /
3.5. zkey contribute is G1 only and moved within noise (-5.6% to +3.1%, both variants).

Prepare 2^18 is the noisiest row: rounds 1 and 2 ran while other processes used three or
more cores (base 89.5 s in round 1). The clean round 3 reads 58.0 / 55.0 / 58.9 s, and the
lane's own interleaved runs read 57.6/54.7/54.0/59.6 s (base/final/final/base, -6.6%). On
prepare final and finalperf are the same code over the raw field layer; their difference
here is noise.

## Does main on perf arkworks get both gains

Yes. `finalperf` keeps every one of `final`'s gains over `base` (prove -2.0% to -6.1%, the
NTT stage, setup, ptau contribute) and adds perf arkworks' own, which `final` does not have:

- Groth16 verify -19.6% to -31.5% on CPU (1.25 to 0.88 ms on anon-aadhaar), and the same on
  the Metal build (-18.6% to -31.5%).
- ptau verify 2^13 -13.4%, zkey verify init -1.3% to -6.4%.
- The prove pointwise stage -28% (anon-aadhaar 5.1 to 3.5 ms) and gather; prove overall
  -1.7% to +0.8% against `final` (anon-aadhaar 748.4 to 737.7 ms).

Nothing is clearly worse on perf arkworks: the cells where `finalperf` reads above `final`
(railgun-01x01 setup +8% at 70 ms, zkey contribute -1% to +3%) are inside each row's own
round-to-round spread. Metal prove does not move with perf arkworks (-0.8% to +0.4%); Metal
verify does.

## What moved it

1. **Fq2 multiply with two Montgomery reductions** (lane 1). Karatsuba's three products
   taken 512 bits wide, `c1 = (a0 + a1)(b0 + b1) - v0 - v1` and `c0 = v0 - v1` formed
   unreduced, only those two reduced. Own aarch64 asm for the wide product and a reduction
   that skips each round's bottom product. Fq2 mul 38 to 40 down to 31.5 ns; G2 batch
   multiplication 36.3 to 37.0 down to 30.8 us.
2. **G2's batch inversion over norms** (lane 1). The running products of every batched
   step in Fq rather than Fq2, each inverse rebuilt as conj(d)/norm(d): about six Fq
   multiplies an element instead of nine. G2 batch 30.6 to 28.4 us.
3. **Packed twiddle tables for the NTT's outer passes** (lane 2). The two radix-4 pairs
   next to the fused block read the full table at an 8 KB stride and took 1.5x every other
   pair; each pass now reads a packed table, which is the smaller transform's cached one.
   Three pipelines at 2^21 274 to 249 ms.
4. **Radix-8 head and tail in the NTT's fused block** (lane 2). Seven of the twelve
   multiplies of the first three passes are by one; one kernel per run of eight does the
   rest. Three pipelines 291 to 271 ms.

## Floors and headroom, before and after

Floors from [trace.md](trace.md). Ranked by absolute headroom after.

| segment | floor | before | after | after / floor | bound |
|---|---:|---:|---:|---:|---|
| ptau prepare 2^18 | 32 s | 58.0 s | 54.7 s | 1.7x | compute, G1 multiplications 70% of it |
| zkey contribute anon-aadhaar | 4.5 s | 7.38 s | 7.31 s | 1.6x | compute, G1 only |
| ptau contribute 2^19 | about 4.4 s | 7.37 s | 6.83 s | 1.6x | compute |
| setup anon-aadhaar | about 0.6 s | 2.59 s | 2.34 s | 3.9x | compute plus serial parse and I/O |
| prove anon-aadhaar | 0.42 s | 0.785 s | 0.748 s | 1.9x | MSM 501 against 260, NTT 230 against 150 |
| verify | about 0.5 ms | 1.20 to 2.03 ms | 0.82 to 1.63 on perf arkworks | 1.6 to 3.3x | pairing arithmetic |

The NTT floor is 150 ms at the intrinsic multiply and 193 ms at the measured one; one
thread now runs 1.13x the measured-rate floor, so what is left is the field multiply
itself and 8.4 effective cores out of 9.6.

## Rejected or no gain, with the measurement

- Lazy Fq2 multiply in portable u128 Rust: 49.6 ns against 39.7 (asm, kept on aarch64 only;
  x86_64 keeps three fused multiplies).
- Lazy Fq2 square: 28.2 ns against 26.5. Fq multiply as wide product plus separate
  reduction: 11.83 against 11.42 ns.
- Radix 4 on G2 again (a sum of two now 1.48 products, so 0.99 of two passes) and radix 8 on
  G1 prepare (9.9 us a point-pass against radix 4's 9.25, from measured sum costs). Not built.
- NTT in-block radix-4 pairs: 351 to 346 ms one thread, three pipelines within noise.
- NTT passes between block and a 1 MB L2 chunk run per chunk: 247.6 against 243 to 246 ms.
  Not bandwidth bound.
- NTT `TASKS_PER_THREAD` 2 or 8 against 4: within noise. Coset multiply on RawFr: 692 to
  687 ms one thread, within noise.

## Correctness

- Final CPU A/B: 810 self-verified warm proofs, 0 runner failures; `verify_time` rejected a
  tampered public signal for every circuit under all three builds; **23 of 23
  deterministic outputs byte-identical** across variants and rounds (135 files: setup for
  all nine circuits, prepare at 2^13 and 2^18, both beacons, zkey beacon for all nine, the
  prove trace); **snarkjs 0.7.2 accepted 39 of 39**. Metal A/B: 540 proofs, 0 failures.
  Lane runs: prepare 2^18 and 2^13 output hashes identical to base in every run.
- Tests, `cargo test --release -p <crate> -- --skip campaign`, all pass: snarkrs-field 12 +
  1 + 5 (new: `mul_wide_and_sub_wide_match_u128`, `fq_reduce_matches_ark` up to q 2^256 - 1,
  `raw_fq2_mul_every_edge_coefficient`, norm against ark), snarkrs-msm 35 + 8 + 2
  (`invert_each_matches_one_at_a_time` now over both fields, G2 through norms),
  snarkrs-ntt 20 (new: `stride_twiddles_are_the_smaller_transforms_tables`),
  snarkrs-groth16 26, snarkrs-ceremony every suite including the snarkjs byte-for-byte ones
  and `setup_metal` (with a metal build at `target/release/snarkrs`). Mutations (a wrong
  reduction carry, a swapped eighth root) fail them.
- `cargo zigbuild --release --target x86_64-unknown-linux-gnu -p snarkrs --no-default-features
  --features cpu` with `+adx,+bmi2` builds (the one warning, an unused macro, predates this);
  `cargo check -p snarkrs --features metal` and `-p snarkrs-metal --tests` pass; fmt clean;
  clippy adds no warning.

## x86

Not timed. What x86_64 runs differently after this round:

- **Unchanged on x86_64**: `RawFq2::mul` (the lazy multiply and `raw/wide.rs` are aarch64
  only; x86_64 keeps three `mont_mul`s).
- **Changed on x86_64**, both target-independent:
  - `glv::GlvField::norms` / `inv_from_norm` and `invert_pairs` in
    `crates/msm/src/glv/batch.rs`: G2's batch inversion goes through norms in Fq (two
    `RawFq::sqr` and an add per element, then two `RawFq::mul` back). Fewer multiplies on
    any target; it is on the raw.rs multiply, so X86-05 applies. Every G2 `mul_batch`
    caller: prepare, ptau contribute, setup.
  - `crates/ntt/src/lib.rs`: `dit_head` / `dif_tail` (RawFr arithmetic, the same type the
    radix-2 in-block passes they replace used) and `stride_twiddles` for the outer passes
    (indexing only). Every `CpuNtt` transform above 2^11: the prover's `intt_coset_ntt`,
    `ntt`, `intt_to_bitrev`, `ntt_from_bitrev`, and the ceremony's uses of `ntt`.

## What is left, ranked by absolute headroom

1. **ptau prepare, about 23 s over its floor at 2^18**, now mostly G1: `mul_batch` 20 us
   against 12.3 (15.2 at the measured multiply). Leads: a lazy `[0, 2p)` form through the
   ladder to drop the multiply's conditional subtraction; a faster Fq inversion than
   arkworks' (3.1 us, about 2.7% of a G1 multiplication at 1024 a batch).
2. **zkey and ptau contribute, about 2.5 to 3 s over**, the same G1 kernel.
3. **prove anon-aadhaar, 0.33 s over**: H MSM fill as last round (108 ns a point-window
   against 76); the NTT 230 against 150 to 193 ms, now limited by the field multiply (11.4
   against 9.2 ns) and its add/sub (3.8 ns a butterfly against 3.0).
4. **setup anon-aadhaar, 1.7 s over**: parse, IC's MSM, `write_hs`, I/O, as last round.
5. **verify**: perf arkworks takes it to 0.82 to 1.63 ms; merging perf arkworks is the
   lever, not more snarkrs code.

## Setup

| | |
|---|---|
| machine | Apple M2 Max, 12 cores (8P + 4E), 96 GiB, macOS, on AC |
| toolchain | rustc 1.99.0, release profile (fat LTO, 1 CGU) |
| build | `bench/scripts/arkab-build.sh mac` per variant, base from a detached checkout of 586f42d; Cargo.toml and Cargo.lock restored after each |
| base `snarkrs` | `0d489eb997bf9f5adab8fefc581b53cc0dde8aebc94af66ae675372056c56b9a` |
| final `snarkrs` | `2b75e047010a6e4635038329451c4cb916c28f317efec6e10c9dcaf5128f331b` |
| finalperf `snarkrs` | `bedde67bbb0ac148d12c8a7828a8c33dbd5185031318d6ee442c68d6cda22ea5` |
| inputs | as the previous rounds |
| snarkjs | 0.7.2 |

Another agent shared the Mac (x86 work, local builds); every timing session and every heavy
build here held a shared lock, and arkab ran with `--max-load 3`. Background load from
other users' processes was 1 to 2 cores most of the day.

Reproduce (bin dir holds `snarkrs-<v>`, `verify_time-<v>`):

```
bench/scripts/arkab.py --bin-dir BIN --variants base final finalperf --backend cpu \
    --artifacts bench/artifacts --ptau-dir bench/ptau --out OUT --max-load 3 --rounds 3 \
    --snarkjs-zkeys js_2x2_d32 js_16x16_d32
bench/scripts/arkab_report.py OUT --snarkjs SNARKJS --artifacts bench/artifacts --ptau-dir bench/ptau
cargo run --release -p snarkrs-field --example raw_ops -- RawFq RawFq2
cargo run --release -p snarkrs-msm --example glv_cost -- 1024 2048
cargo run --release -p snarkrs-ntt --example pipeline_cost -- 21
```
