# Five workflows against measured floors, Mac CPU, 2026-10-11

An optimise-by-intrinsic pass over snarkrs on the M2 Max, CPU backend: witness generation,
proving, phase 1 (`powersoftau contribute`, `verify`, `prepare phase2`), phase 2 (`groth16
setup`, `zkey contribute`, `zkey verify`) and Groth16 verification. Stock arkworks 0.5
throughout; no dependency, lockfile or arkworks change. Branch `worktree-agent-ad00d0510ef79aaa4`,
18 commits on 96340d7.

- [trace.md](trace.md): the counted work per workflow, the intrinsics, the floors.
- [lanes.md](lanes.md): the seven lanes, each A/B'd against the commit before it, and what was
  tried and rejected.
- [ab-reports.md](ab-reports.md): every A/B's full report, per round, as `arkab_report.py`
  printed it.

## Start vs end

`base` is 96340d7, `final` is 46d4823. One interleaved A/B with `bench/scripts/arkab.py`, three
rounds, order reversed each round, every circuit, both phase-1 sizes of the arkab report. Prove:
median of 30 self-verified warm proofs; verify: median of three 500-call medians
(`verify_time`, which first checks a tampered public signal is rejected); ceremony: median of
three wall clocks, each counted only on exit 0 and, for the verifiers, their OK line.

Per circuit (change is end against start):

| circuit | prove, ms | verify, ms | setup, ms | zkey contribute, ms | zkey verify, ms |
|---|---|---|---|---|---|
| js_2x2_d32 | 68.0 to 50.9 (-25.1%) | 1.551 to 1.316 (-15.1%) | 1,994 to 802 (-59.8%) | 447 to 164 (-63.4%) | 68.2 to 59.0 (-13.4%) |
| railgun-01x01 | 54.1 to 40.1 (-25.9%) | 1.478 to 1.290 (-12.7%) | 479 to 87 (-81.9%) | 484 to 166 (-65.7%) | 69.0 to 59.0 (-14.4%) |
| tornado | 87.2 to 65.4 (-25.0%) | 1.385 to 1.264 (-8.8%) | 1,195 to 368 (-69.2%) | 559 to 214 (-61.6%) | 72.8 to 61.6 (-15.4%) |
| sha256 | 45.3 to 34.8 (-23.2%) | 1.256 to 1.212 (-3.6%) | 2,883 to 739 (-74.4%) | 1,149 to 344 (-70.0%) | 129.6 to 101.7 (-21.5%) |
| railgun-13x01 | 316.8 to 238.2 (-24.8%) | 2.397 to 1.603 (-33.1%) | 3,086 to 572 (-81.5%) | 3,613 to 1,081 (-70.1%) | 382.1 to 305.2 (-20.1%) |
| js_16x16_d32 | 402.2 to 297.8 (-25.9%) | 3.687 to 2.057 (-44.2%) | 16,232 to 6,224 (-61.7%) | 3,581 to 1,089 (-69.6%) | 396.8 to 310.9 (-21.6%) |
| rsa2048 | 171.6 to 129.2 (-24.8%) | 1.796 to 1.531 (-14.7%) | 1,988 to 807 (-59.4%) | 3,862 to 1,268 (-67.2%) | 418.6 to 337.8 (-19.3%) |
| keccak256 | 164.3 to 123.1 (-25.1%) | 1.252 to 1.213 (-3.0%) | 3,497 to 397 (-88.6%) | 4,592 to 1,434 (-68.8%) | 428.8 to 359.8 (-16.1%) |
| anon-aadhaar | 1,341.9 to 974.8 (-27.4%) | 1.417 to 1.271 (-10.3%) | 26,074 to 3,306 (-87.3%) | 28,499 to 8,996 (-68.4%) | 2,817 to 2,355 (-16.4%) |

Phase 1:

| item | ptau contribute, ms | ptau verify, ms | ptau prepare phase2, ms |
|---|---|---|---|
| 2^13 | 617.9 to 253.2 (-59.0%) | 65.1 to 59.2 (-9.0%) | 5,478 to 2,182 (-60.2%) |
| 2^19 (prepare: 2^18) | 38,446 to 8,539 (-77.8%) | 2,275 to 1,826 (-19.7%) | 229,863 to 67,467 (-70.6%) |

anon-aadhaar's prove stages, ms: NTT 408.9 to 289.0, MSM 916.6 to 660.2, gather 8.9 to 9.5,
pointwise 5.4 to 5.3. Every cell moved in all three rounds; the `*` rounds (other processes
on three or more cores) touch only setup on anon-aadhaar and two single cells, and all three
anon-aadhaar setup rounds agree (3,294 / 3,306 / 3,307 ms).

**Witness generation** (`wtns calculate`'s in-process path, `crates/witness/examples/wtns_time.rs`,
five runs each, every witness compared with the artifact's `circuit.wtns`) was not changed by
any lane and is listed for its headroom. Compile is wasmtime/Cranelift on the circuit wasm, run
is one witness; node is `node generate_witness.js` (snarkjs' path, V8) end to end:

| circuit | compile, ms | run, ms | node total, ms |
|---|---:|---:|---:|
| js_2x2_d32 | 185 | 14.4 | |
| railgun-01x01 | 217 | 36.4 | |
| sha256 | 19 | 107 | |
| railgun-13x01 | 287 | 100 | |
| js_16x16_d32 | 193 | 111 | |
| rsa2048 | 13 | 1,722 | 1,390 |
| keccak256 | 22 | 190 | |
| anon-aadhaar | 296 | 3,205 | 3,030 |

tornado is a circom 1 circuit, which the wasm calculator refuses by design.

## What moved it

1. **GLV on raw limbs** (L1, `glv::mul`). Every place a point is multiplied by its own scalar
   was arkworks' affine double-and-add over all 254 bits (or, in prepare, a 254-bit wNAF):
   key scaling, prepare's twiddles, setup's small slots, verify's public inputs. Two 127-bit
   halves halve the doublings; setup gains most because its coefficients are mostly -1 and
   small negatives, 254-bit scalars to double-and-add and one- or two-digit ones to GLV
   (anon-aadhaar: 736 M doublings to 27 M).
2. **Batch-affine GLV, four-way on G2** (L5, L7, `glv::mul_batch`). Lockstep ladders in affine
   coordinates, one inversion a step; G2 also splits on `psi`, 65 steps instead of 127. G2
   multiplication 136 to 54 us single thread. Prepare only paid once its passes ran their
   groups in parallel (L7): the rewrite had serialised large-group passes, which a `sample`
   profile showed as half the pool waiting.
3. **aarch64 assembly in raw.rs** (L6). Add, sub and the multiply's reduction as single
   carry/borrow chains plus csel, the multiply as gnark-crypto's arm64 `mul`: add 3.30 to 1.84 ns,
   mul 19.4 to 13.9 ns throughput, no mode slower. raw.rs carries the MSM, the NTT (L3) and now
   all of the ceremony's point arithmetic, so this is the one change that moves every workflow:
   prove -24% to -26% on all nine circuits.
4. Smaller: phase 1 key scaling in sixteen-chunk calls with 256-point tasks (L4, contribute
   -20%), and prepare's `1/n` folded into the last pass (L2, -6%).

## Floors and headroom, before and after

Floors from [trace.md](trace.md): intrinsic multiply 9.2 ns and add 1.0 ns, 9.6 P-core
equivalents, the cheapest known formulas. "Before" is base, "after" is final, both from the
table above; ranked by absolute headroom before.

| segment | floor | before | after | after / floor | bound |
|---|---:|---:|---:|---:|---|
| ptau prepare 2^18 | 34 s | 229.9 s | 67.5 s | 2.0x | compute (point multiplications) |
| ptau contribute 2^19 | 4.6 s | 38.4 s | 8.54 s | 1.9x | compute |
| setup anon-aadhaar | about 0.6 s | 26.1 s | 3.31 s | 5.5x | compute plus a serial hash |
| zkey contribute anon-aadhaar | 4.5 s | 28.5 s | 9.00 s | 2.0x | compute |
| setup js_16x16_d32 | 2.5 s | 16.2 s | 6.22 s | 2.5x | compute |
| prove anon-aadhaar | 0.42 s | 1.342 s | 0.975 s | 2.3x | compute (MSM 254 of it, NTT 154) |
| prove js_16x16_d32 | 0.07 s | 0.402 s | 0.298 s | 4.3x | compute |
| verify | about 0.5 ms | 1.25 to 3.69 ms | 1.21 to 2.06 ms | 2.4 to 4x | pairing arithmetic |

Not derived: zkey verify (2.36 s on anon-aadhaar) and ptau verify (1.83 s at 2^19), where part
of the time is reading the files back, which was not measured on its own; witness generation,
which runs the circuit's own wasm field arithmetic (the practical reference is V8: node's whole
run is 13% below ours on anon-aadhaar and 20% below on rsa2048).

Single-thread costs behind the ceremony rows (`glv_cost`, us per multiplication, final code):
G1 arkworks 74.0, `glv::mul` 26.6, `mul_batch` 22.9 at 1024 a batch (floor 12.3); G2 arkworks
211.0, `glv::mul` 75.6, `mul_batch` 37.3 (floor 26.8).

## Rejected or no gain, with the measurement

- Row-shaped multiply in pure Rust: 16.1 ns against 19.4, but a ceremony test build that
  inlines it did not finish in 20 minutes. The assembly is faster (13.9) and compiles in 45 s.
- One-bit-carry add/sub in pure Rust: add stream 3.40 to 3.81 ns, sub stream 2.64 to 3.01;
  sub as add-of-complement: LLVM still emits cmp/ccmp/cset/csinc.
- Parallel-chains add (`a + b` beside `a - (m - b)`), the latency idea from the arkworks lane: in
  assembly latency 2.72 to 2.52 ns but throughput 1.84 to 2.36; the serial shape stays.
- NTT on RawFr with the old multiply (L3): NTT -2% to -5%, prove within noise. Kept as the
  vehicle for L6.
- Working-memory reuse and 32-butterfly small-pass tasks without the parallel-group fix (L7
  first half): prepare 2^18 -0.4%, zkey contribute -2.8%. Kept; harmless and needed by the fix.
- First batch-affine layout, one struct per point scanned every step: replaced before it was
  measured end to end (step-major digit rows, 7cea0d2).

## Correctness

- Final A/B: 540 self-verified warm proofs, 0 runner failures; `verify_time` rejected a tampered
  public signal for every circuit under both builds before timing; `ptau verify` accepted every
  contribute output and `zkey verify init` every contributed key; **23 of 23 deterministic
  outputs byte-identical** across builds and rounds (setup for all nine circuits every round,
  prepare at 2^13 and 2^18 every round, ptau beacon at 2^13 and 2^19, zkey beacon for all nine,
  the prove trace; 90 files hashed); **snarkjs 0.7.6 accepted 26 of 26** (18 proofs, both
  builds' contributed and prepared 2^13 ptau, both builds' contributed js_2x2_d32 and
  js_16x16_d32 zkeys against r1cs and ptau). Every lane A/B hashed its own deterministic
  outputs too, all identical.
- Tests, `cargo test --release -p <crate> -- --skip campaign`, all pass: snarkrs-field 10 + 1 +
  4 (raw_matches_ark now runs every pair of fourteen carry-chain edge limb patterns per field,
  0, 1, m-1, m-2, (m+-1)/2, all-ones low limbs, beside the value edges and 200,000 random
  pairs, on Fq, Fq2 and Fr); snarkrs-msm 34 + 4 (glv_matches_ark: `glv::mul` and `mul_batch`
  on both groups against arkworks over the edge scalars, the identity, repeated points and
  opposite scalars, affine and non-normalised inputs) + 2; snarkrs-ntt 19; snarkrs-ceremony 148
  (lib 32, phase1 20 and phase2 11 with the snarkjs byte-for-byte reproductions, prepare 7
  including local_13 against snarkjs, ptau 11, setup 4, setup_metal 2, and the rest);
  snarkrs-groth16 26.
- `cargo zigbuild --release --target x86_64-unknown-linux-gnu -p snarkrs --no-default-features
  --features cpu` with `+adx,+bmi2` builds; raw.rs's portable path is the code that was there,
  so x86_64 codegen of the field layer and the IFMA path are unchanged. `cargo check -p snarkrs
  --features metal` and `cargo check -p snarkrs-metal --tests` pass. CUDA not built (no nvcc).

## What is left, ranked by absolute headroom

1. **ptau prepare, 33 s over its floor at 2^18.** The batched ladder is still 1.9x its own
   arithmetic floor on G1 (22.9 against 12.3 us): the inversion share and per-step scans at
   1024 a batch, and multiplies at 13.9 rather than 9.2 ns. Larger batches for the late passes,
   and keeping prepare's points affine between passes (each pass now normalises its inputs),
   are the next steps. The Metal backend already does this command in 19 s.
2. **zkey contribute and ptau contribute, about 4 s each over floor**, the same kernel.
3. **setup, 3.7 s over on js_16x16_d32**: `compose_c` 2.70 s and `compose_b2` 2.57 s of 6.2
   (`crates/ceremony/examples/setup_stages.rs`) are full-size coefficients still multiplied one
   at a time in XYZZ (G2 75.6 us each); batching every small slot's terms through `mul_batch`
   and summing per slot afterwards should about halve both. On anon-aadhaar the serial
   `hash_h_points` (0.59 s of 3.3) is next.
4. **Witness generation, 0.2 to 0.3 s a call of Cranelift compile** on the large circuits, paid
   by every `wtns calculate` and `fullprove`. Caching the compiled module keyed by the wasm's
   hash would remove it; that is a feature with a trust question (wasmtime's deserialisation is
   unsafe on a tampered file), so it is a decision, not a lane. node's whole run is 13% to 20%
   below ours on the two large circuits.
5. **prove, 0.55 s over on anon-aadhaar**: the MSM is 2.6x its batch-affine floor (660 against
   254 ms, dominated by the dense 2^21 H MSM), the NTT 1.9x (289 against 154).
6. **verify, about 0.7 to 1.5 ms over**: the pairing is arkworks Fq/Fq2/Fq12; porting the Miller
   loop and final exponentiation onto raw.rs is the only lever left there.

## Caveats

- **The Mac was shared.** Other sessions ran multi-core jobs (a `poseidon2-gen` at up to 11
  cores, `node` generators, Spotlight indexing) through most of the day, and the machine slept
  once mid-run for 3.75 hours. arkab waits until other processes use fewer than three cores
  before each command, but a job that starts mid-command is not caught; such rounds are marked
  `*`, and lanes.md quotes clean rounds where a round was hit. Every lane delta above is
  same-session, interleaved and in the same direction in its clean rounds.
- One machine. x86_64 builds (`cargo zigbuild ... --features cpu`, `+adx,+bmi2`) and nothing in
  them changed but GLV and batch-affine, which are portable; not timed, no AWS access in this
  task. Metal and the `snarkrs --features metal` binary build and its tests type-check; CUDA
  not built (no nvcc).
- The E-core weight in the floors (0.4 of a P core) is assumed, not measured.

## Setup

| | |
|---|---|
| machine | Apple M2 Max, 12 cores (8P + 4E), 96 GiB, macOS, on AC |
| toolchain | rustc 1.99.0, release profile (fat LTO, 1 CGU) |
| build | `cargo build --release -p snarkrs --features metal` plus `verify_time`, `field_ops`, `wtns_time` examples, from `git archive` of each commit into one shared target dir |
| base `snarkrs` | `613c8f74aee3eeb38d798e37fa6ee5960f35e08af916d32517151e222a914370` |
| final `snarkrs` | `d2ba8dc54c89a28b5e52c93bd4a815e2d0ca2777e38965d9ef971ea0e58cc754` |
| inputs | as `arkff-perf-ab-2026-10-11.md`: contribute and verify `local_13`, `local_19`; prepare `local_13`, `ppot_0080_18`; setup from each circuit's r1cs and the smallest covering `ppot_0080_NN` |
| snarkjs | 0.7.6 |

Reproduce a lane or the final run (bin dir holds `snarkrs-<v>`, `verify_time-<v>`):

```
bench/scripts/arkab.py --bin-dir BIN --variants base final --backend cpu --artifacts bench/artifacts \
    --ptau-dir bench/ptau --out OUT --max-load 3 --snarkjs-zkeys js_2x2_d32 js_16x16_d32
bench/scripts/arkab_report.py OUT --snarkjs SNARKJS --artifacts bench/artifacts --ptau-dir bench/ptau
cargo run --release -p snarkrs-field --example raw_ops
cargo run --release -p snarkrs-msm --example glv_cost
cargo run --release -p snarkrs-ceremony --example ceremony_trace -- --prepare 18 --contribute 19 R1CS...
```
