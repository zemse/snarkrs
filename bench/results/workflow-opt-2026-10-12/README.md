# Ceremony kernels and the proving MSM against measured floors, Mac CPU, 2026-10-12

The second optimise-by-intrinsic pass over snarkrs on the M2 Max, CPU backend, picking up
[the previous round](../workflow-opt-2026-10-11/README.md)'s ranked leftovers: `ptau prepare`,
the two contribute steps, `groth16 setup`, and the proving MSM. Its intrinsics, floors and
reproduce commands are reused; what was re-measured is in [trace.md](trace.md). Stock
arkworks 0.5, no dependency or lockfile change. Branch `worktree-agent-a083538a0146ac12c`,
11 commits on 8416f70 (the work started on 1f85bd4; the rebase picked up bench tooling only).

- [trace.md](trace.md): the per-operation costs measured this round and the floors.
- [lanes.md](lanes.md): each change with its own A/B, and what was tried and rejected.
- [ab-reports.md](ab-reports.md): the final and the lane 1 A/B reports as printed.

## Start vs end

`base` is 1f85bd4, `final` is 196d6bb. One interleaved A/B with `bench/scripts/arkab.py`,
three rounds, order reversed each round, every circuit, both phase-1 sizes; prove is the
median of 30 self-verified warm proofs, every other cell the median of three wall clocks
counted only on exit 0 (and the verifiers' OK line).

| circuit | prove, ms | setup, ms | zkey contribute, ms | zkey verify, ms |
|---|---|---|---|---|
| js_2x2_d32 | 51.7 to 47.1 (-8.8%) | 837 to 489 (-41.5%) | 169 to 154 (-8.9%) | 60.0 to 57.1 (-4.9%) |
| railgun-01x01 | 40.5 to 39.1 (-3.4%) | 88.5 to 71.8 (-18.8%) | 178 to 159 (-10.6%) | 59.8 to 57.2 (-4.4%) |
| tornado | 64.4 to 61.5 (-4.5%) | 367 to 199 (-45.8%) | 213 to 184 (-13.6%) | 63.0 to 60.2 (-4.5%) |
| sha256 | 33.7 to 31.4 (-7.0%) | 748 to 529 (-29.3%) | 355 to 308 (-13.2%) | 106.8 to 103.7 (-2.8%) |
| railgun-13x01 | 231.9 to 218.9 (-5.6%) | 558 to 372 (-33.3%) | 1,121 to 1,002 (-10.6%) | 316.5 to 278.9 (-11.9%) |
| js_16x16_d32 | 294.8 to 283.1 (-4.0%) | 6,560 to 3,254 (-50.4%) | 1,106 to 1,011 (-8.5%) | 335.4 to 275.9 (-17.7%) |
| rsa2048 | 127.9 to 119.4 (-6.7%) | 834 to 704 (-15.6%) | 1,267 to 1,129 (-10.9%) | 340.4 to 299.1 (-12.1%) |
| keccak256 | 121.0 to 113.2 (-6.4%) | 408 to 322 (-21.0%) | 1,366 to 1,238 (-9.4%) | 347.7 to 326.0 (-6.2%) |
| anon-aadhaar | 975.5 to 840.1 (-13.9%) | 3,302 to 2,467 (-25.3%) | 8,713 to 7,859 (-9.8%) | 2,300 to 1,891 (-17.8%) |

| item | ptau contribute, ms | ptau verify, ms | ptau prepare phase2, ms |
|---|---|---|---|
| 2^13 | 254 to 234 (-8.2%) | 60.6 to 60.2 | 2,168 to 2,000 (-7.8%) |
| 2^19 (prepare: 2^18) | 8,446 to 7,876 (-6.7%) | 1,866 to 1,858 | 66,778 to 57,975 (-13.2%) |

anon-aadhaar's prove stages, ms: MSM 669.4 to 531.7 (-20.6%), NTT 286.3 to 289.3 (not
touched), gather 9.4 to 8.8, pointwise 5.2. `zkey verify init` re-runs setup and checks with
random-linear-combination multiexps, so it gains from both lanes. Groth16 verify and ptau verify are unchanged (-0.6% to +0.4%), as
expected; every other cell moved in the same direction in all three rounds.

## What moved it

1. **Two points at a time in the batched formulas** (lanes 1 and 3). A batch-affine
   doubling or addition is a chain of dependent multiplies, each an asm block of about 130
   instructions, so one point at a time leaves the multiplier waiting: the core's lookahead
   does not reach the next point. Two running-product chains per shared inversion hand out
   inverses in pairs, and the formulas of the two run interleaved step by step. MSM fill
   121.6 to 108 ns a point-window single thread, `mul_batch` G1 -8% to -9%.
2. **setup's small slots through batched ladders** (lane 2). One `glv::mul` per term
   became one ragged batch per task (`mul_sums_xyzz_in`): up to 16 terms of a slot share a
   ladder's doublings, the task shares each step's inversion, G2 splits four ways, `1` and
   `-1` are additions. The multiexp threshold moved from 32 to 256 terms. js_16x16_d32
   setup 6.56 to 3.25 s.
3. **Radix 4 for prepare's G1 passes** (lane 1): `B = [v] x2 + [v^3] x3` and its partner
   are sums of two products under one ladder, three ladders for four points.
4. **Pippenger tasks for twice the pool** (lane 3): 16 windows on 12 threads were two
   waves; H 588 to 535 ms.
5. Smaller: two-chain inversion in `mul_batch` (G1 -5%), G2 prepare passes batched from 64
   butterflies, `phi psi` tables from the `psi` table (G2 -2%), setup's point hashing
   encoded in parallel and overlapped with the next chunk (anon-aadhaar `hash_h_points`
   559 to 190 ms), bucket `(x, y)` in one cache line (-2.5% fill).

## Floors and headroom, before and after

Floors from [trace.md](trace.md) (the previous round's, the MSM one re-derived). Ranked
by absolute headroom after.

| segment | floor | before | after | after / floor | bound |
|---|---:|---:|---:|---:|---|
| ptau prepare 2^18 | 34 s | 66.8 s | 58.0 s | 1.7x | compute (point multiplications) |
| zkey contribute anon-aadhaar | 4.5 s | 8.71 s | 7.86 s | 1.7x | compute |
| ptau contribute 2^19 | 4.6 s | 8.45 s | 7.88 s | 1.7x | compute |
| setup anon-aadhaar | about 0.6 s | 3.30 s | 2.47 s | 4.1x | compute plus serial parse and I/O |
| setup js_16x16_d32 | 2.5 s | 6.56 s | 3.25 s | 1.3x | compute |
| prove anon-aadhaar | 0.42 s | 0.976 s | 0.840 s | 2.0x | MSM 532 against 260, NTT 289 against 154 |
| verify | about 0.5 ms | 1.21 to 2.05 ms | same | 2.4 to 4x | pairing arithmetic |

## Rejected or no gain, with the measurement

- Dedicated aarch64 Montgomery squaring (own SOS code, 56 multiplier ops against 68):
  `raw_ops` throughput 11.07 to 10.15 ns but latency 15.95 to 16.64; `glv_cost` G1 batch
  unchanged (22.65/22.56 to 22.68/22.69 us), and again unchanged after the interleaving
  (20.45/20.00 to 20.06/20.19). Not committed.
- Four running-product chains in `mul_batch`: G1 1024 20.69/20.49 to 20.99/21.12 us, none.
- Two chains in the MSM flush applied one at a time: 121.4/118.1 to 122.5/122.6 ns, none;
  only the interleaving pays. Four entries interleaved: 107.3/109.5 to 116.7/113.8 ns,
  worse (register spills).
- Radix 4 on G2: a sum of two G2 products costs 1.51 single ones (56.7 against 37.6 us), so
  a pair of passes would cost 1.005 of two radix-2 passes. G2 keeps radix 2.
- Larger prepare batches (4096 butterflies a task): no clear change at 2^18 inside noise;
  the radix-4 batches are already 1024 and 2048 outputs.
- Wider or narrower H window (c = 13, 15) and four or eight point chunks a window: none
  better than c = 16 at two chunks (657, 600 against 562 ms; 537, 595).

## Correctness

- Final A/B: 540 self-verified warm proofs, 0 runner failures; `verify_time` rejected a
  tampered public signal for every circuit under both builds; **23 of 23 deterministic
  outputs byte-identical** across builds and rounds (setup for all nine circuits, prepare
  at 2^13 and 2^18, both beacons, zkey beacon for all nine, the prove trace; 90 files);
  **snarkjs 0.7.6 accepted 26 of 26**. The lane 1 A/B: 4 of 4 identical.
- Tests, `cargo test --release -p <crate> -- --skip campaign`, all pass: snarkrs-msm 35 + 8
  + 2 (new: `invert_each` at lengths 1 to 9, sums of one to three terms, ragged sums with
  empty ones, a sum of 31, cancelling and repeated terms, the identity and every edge
  scalar, both groups); snarkrs-ceremony with the snarkjs byte-for-byte setup, phase 1,
  phase 2 and prepare reproductions (new: `radix_4_passes_match_radix_2` on one thread at
  2^13 and 2^14, `small_slots_stay_on_cpu` with the error ordering; `recovery.rs`' fixture
  raised to 257 constraints to stay above the new threshold); snarkrs-groth16 26;
  snarkrs-ntt 19; snarkrs-field 10 + 1 + 4.
- `cargo zigbuild --release --target x86_64-unknown-linux-gnu -p snarkrs --no-default-features
  --features cpu` with `+adx,+bmi2` builds (the one warning, an unused macro on x86, was
  there before). `cargo check -p snarkrs --features metal` and `-p snarkrs-metal --tests`
  pass.

## x86

Not timed (no x86 machine in this task). Nothing here moved an x86 hot path from arkworks
onto the raw field layer: prepare, contribute and setup were already on `RawFq`/`RawFq2`
through `glv` before this round (setup's per-term `glv::mul_affine` is now the batched
`mul_sums_xyzz_in`, same field layer), and the MSM changes are in the portable batch-affine
fill, which an IFMA machine does not run (`ifma.rs` only reshapes its exported buckets for
the shared running sum). The interleaving and the radix 4 are layout and scheduling over
the same arithmetic, so they should not be worse there, but X86-05 (raw.rs multiply slower
than ark's on x86) still applies to all of it and is unmeasured for these kernels.

## What is left, ranked by absolute headroom

1. **ptau prepare, 24 s over its floor at 2^18.** G1 `mul_batch` is 20.0 us against 12.3
   (doubling about 100 ns against 75 at the measured multiply), G2 37 against 26.8. The
   G2 ladder did not respond to interleaving; its Fq2 multiply as three Montgomery
   reductions is the next lever (a lazy-reduction Fq2 product, two reductions instead of
   three, own code). Metal does this command in 19 s.
2. **zkey contribute and ptau contribute, 3.3 s each over**, the same kernel.
3. **prove anon-aadhaar, 0.42 s over**: H MSM fill 108 ns a point-window against 76 at the
   measured multiply; the bucket running sum is about 8% of the MSM; the NTT (289 against
   154 ms) was not in this round's lanes.
4. **setup anon-aadhaar, 1.9 s over**: what is left is the r1cs pass (0.19 s), IC's single
   666k-term MSM (0.25 s), `write_hs` (0.12 s), compose_a's large MSM slots, and file I/O.
5. **verify**: the pairing, as before.

## Setup

| | |
|---|---|
| machine | Apple M2 Max, 12 cores (8P + 4E), 96 GiB, macOS, on AC |
| toolchain | rustc 1.99.0, release profile (fat LTO, 1 CGU) |
| build | `cargo build --release -p snarkrs --features metal` plus `verify_time`, from `git archive` of each commit into one shared target dir |
| base `snarkrs` | `97e7c9803dfef6b47a28d734a5c2558928bed8a95fc4dae5115a78eb8b8f1daf` |
| final `snarkrs` | `a635037fc42d78d8d6c7f8c2c08146f699bc2f2dc8fc1033d497df5a59b2d53b` |
| inputs | as the previous round |
| snarkjs | 0.7.6 |

The Mac was shared with another agent's builds for the first half of the round; every
timed command waited for it (arkab's `--max-load 3` plus a check that the coordinator's
go file existed), and single runs taken while it was busy were discarded.

Reproduce (bin dir holds `snarkrs-<v>`, `verify_time-<v>`):

```
bench/scripts/arkab.py --bin-dir BIN --variants base final --backend cpu --artifacts bench/artifacts \
    --ptau-dir bench/ptau --out OUT --max-load 3 --rounds 3 --snarkjs-zkeys js_2x2_d32 js_16x16_d32
bench/scripts/arkab_report.py OUT --snarkjs SNARKJS --artifacts bench/artifacts --ptau-dir bench/ptau
cargo run --release -p snarkrs-msm --example glv_cost -- 1024 2048
cargo run --release -p snarkrs-ceremony --example setup_stages -- R1CS PTAU
cargo run --release -p snarkrs-groth16 --example msm_shape -- DIR_WITH_ANON_AADHAAR_ONLY
```
