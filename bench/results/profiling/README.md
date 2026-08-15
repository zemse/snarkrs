# Where a Groth16 proof spends its time

Profiling round, CPU backend, Apple M2 Max (8 performance + 4 efficiency cores, 12
logical), macOS 15, rustc release profile (`opt-level=3`, `lto="fat"`,
`codegen-units=1`). Every number below names the command that produced it. Nothing here
was optimised; this lane only measured.

Raw output for each instrument is in this directory. Read `msm-shape.txt`,
`field-ops.txt` and `ntt-shape.txt` first if you only read one thing.

---

## 1. The shape of the curve

```
bench/scripts/profile-wallclock.sh 10 2
```

hyperfine over the whole `g16 prove` process: exec, dyld, zkey parse, witness parse,
prepare, prove, JSON write, exit. Median of 10 runs after 2 warmups.

| variant | constraints | median ms | min | max | sd | cores busy | us/constraint |
|---|---:|---:|---:|---:|---:|---:|---:|
| process floor (`g16 --version`) | 0 | 3.1 | 2.9 | 4.0 | 0.3 | 0.83x | - |
| tiny_mul | 2 | 4.2 | 3.6 | 4.6 | 0.4 | 2.43x | 2088 |
| js_1x1_d8 | 3,359 | 43.1 | 32.5 | 51.0 | 6.3 | 6.02x | 12.84 |
| js_2x2_d16 | 10,153 | 71.5 | 66.6 | 97.8 | 11.0 | 8.87x | 7.04 |
| js_2x2_d32 | 17,929 | 121.1 | 108.6 | 128.8 | 6.5 | 8.73x | 6.76 |
| js_8x8_d32 | 70,357 | 356.7 | 345.3 | 374.0 | 9.6 | 9.63x | 5.07 |
| js_16x16_d32 | 140,261 | 684.9 | 655.9 | 717.5 | 17.3 | 9.41x | 4.88 |

`cores busy` is `(user + system) / wall`. A perfectly parallel proof on this machine
would read 12.00x. It does not, and section 5 is about why.

Cost per constraint falls with size and flattens near 4.9 us: doubling the circuit from
70,357 to 140,261 constraints (1.99x) costs 1.92x the time. The prover is linear at the
top of the ladder, and everything below ~10K constraints is dominated by fixed costs.

### The in-process split

```
target/release/g16 bench --artifacts bench/artifacts --reps 10 --mode both --backend cpu
```

Warm (key resident, only `prove()` timed), medians over 10 reps, in ms:

| variant | total | gather | ntt | pointwise | msm | assemble | msm share |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_mul | 0.9 | 0.13 | 0.00 | 0.01 | 0.21 | 0.57 | 23% |
| js_1x1_d8 | 22.9 | 0.40 | 3.57 | 0.26 | 18.09 | 0.57 | 79% |
| js_2x2_d16 | 56.7 | 0.47 | 6.08 | 0.45 | 48.90 | 0.57 | 86% |
| js_2x2_d32 | 96.3 | 0.60 | 10.00 | 0.71 | 83.89 | 0.57 | 87% |
| js_8x8_d32 | 352.3 | 1.99 | 34.79 | 2.29 | 309.82 | 0.58 | 88% |
| js_16x16_d32 | 652.6 | 3.45 | 69.95 | 4.22 | 574.25 | 0.59 | 88% |

Cold adds the parse and prepare: 18.6-24.1 ms at 140K constraints, about 3% of a cold
proof. That part is not interesting and is not discussed further.

**`assemble` is a constant 0.57-0.59 ms at every circuit size**, from 2 constraints to
140,261. It is 0.09% of the largest proof and 63% of the smallest.

---

## 2. What the five MSMs are actually made of

```
cargo build --release -p g16-core --example msm_shape
REPS=5 ./target/release/examples/msm_shape bench/artifacts
```

Each MSM timed alone (best of 5), so these are disjoint costs rather than the overlapped
windows the prover sees. `js_16x16_d32`:

| msm | len | zeros | ones | general | 0/1 % | \|s\|<=8 | inf bases | inf % | c | ms | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A -> G1 | 140,824 | 2,316 | 216 | 138,292 | 1.80% | 7 | 1,187 | 0.84% | 13 | 88.1 | 15.4% |
| B -> G2 | 140,824 | 2,316 | 216 | 138,292 | 1.80% | 7 | 47,803 | 33.95% | 13 | 166.3 | 29.1% |
| B -> G1 | 140,824 | 2,316 | 216 | 138,292 | 1.80% | 7 | 47,803 | 33.95% | 13 | 60.5 | 10.6% |
| L -> G1 | 140,789 | 2,316 | 214 | 138,259 | 1.80% | 7 | 0 | 0.00% | 13 | 88.7 | 15.6% |
| H -> G1 | 262,144 | 0 | 0 | 262,144 | 0.00% | 0 | 0 | 0.00% | 15 | 166.9 | 29.3% |

Per general scalar: A 0.637 us, L 0.642 us, H 0.637 us. The three agree to within 1%, so
**MSM cost is exactly linear in the general-scalar count** and nothing else about the
input matters. B->G1 comes in at 0.437 us only because a third of its bases are the point
at infinity and those adds return early. B->G2 is 1.202 us, 2.75x its G1 twin over
identical scalars, which is what an `Fq2` multiply costs against an `Fq` one.

### The 0/1 claim in `g16-msm`'s module docs is wrong by a factor of fifty

> "In bit-decomposition-heavy circuits over 99% of witness scalars are 0 or 1, so these
> two paths are not micro-optimisations, they are most of the work."
> -- `crates/g16-msm/src/lib.rs:13`

Measured on every artifact in the ladder:

| variant | 0/1 share of witness scalars |
|---|---:|
| js_1x1_d8 | 4.12% |
| js_2x2_d16 | 2.83% |
| js_2x2_d32 | 1.78% |
| js_8x8_d32 | 1.80% |
| js_16x16_d32 | 1.80% |

Not 99%. **1.8%.** The four witness MSMs are dense, not sparse. Two consequences:

* The prescan fast path caps out at a 1.018x speedup on this workload, not the "about
  5.1x" the same doc block claims. It should be kept (it is nearly free and does help
  other circuit shapes) but it is not where the work is.
* The comment in `crates/g16-core/src/cpu.rs:255` about "the four cheap witness MSMs
  (mostly 0/1 scalars)" describes a circuit this repo does not benchmark. A, B1 and L
  each cost about the same as half the H MSM.

Both of those comments should be corrected. They are in the other lane's files, so this
lane did not touch them.

### The `-1` and small-constant fast path is worth 0.005%

The standing review proposes extending the fast path to `|s| <= 8` and rates it a medium
win. Measured: **7 such scalars out of 140,824**, and zero on three of the six circuits.
Refuted.

### One third of the B query bases are the point at infinity

47,803 of 140,824, on both B->G2 and B->G1, and the same ~34% on every circuit in the
ladder. `prescan` classifies scalars and never looks at the base, so each of those still
pays an `into_bigint` and 20 windows of digit extraction to add a point that contributes
nothing.

The prize is small, because the add itself already returns early. From the per-scalar
rates above: B->G1's 90,489 non-infinity generals at A's 0.639 us-per-useful-scalar rate
would be 57.8 ms; B->G1 actually costs 60.5 ms. The residual, ~2.7 ms, is what the 47,803
wasted `into_bigint` calls and window scans cost. B->G2 pays the same scalar-side work
(digit extraction does not know which curve it is feeding), so the total is **~5 ms of a
653 ms proof, 0.8%.**

### The five-way `rayon::join` buys 1.02x at scale

| variant | serial sum of 5 MSMs | prover, 5 overlapped | speedup |
|---|---:|---:|---:|
| tiny_mul | 0.303 ms | 0.171 ms | 1.77x |
| js_1x1_d8 | 22.74 ms | 20.88 ms | 1.09x |
| js_2x2_d16 | 57.77 ms | 52.41 ms | 1.10x |
| js_2x2_d32 | 93.48 ms | 82.21 ms | 1.14x |
| js_8x8_d32 | 294.2 ms | 282.0 ms | 1.04x |
| js_16x16_d32 | 570.5 ms | 564.6 ms | 1.01x |

The nest is worth keeping (it raises core occupancy, section 5) but its stated
justification -- draining four cheap MSMs while the dense one runs -- does not survive
contact with the measurement, because there are no cheap MSMs.

---

## 3. Where the CPU actually is

```
samply record --save-only --no-open --unstable-presymbolicate --rate 4000 \
  --iteration-count 3 -o bench/results/profiling/samply-js_16x16_d32.json.gz -- \
  ./target/profiling/g16 prove --zkey bench/artifacts/js_16x16_d32/circuit.zkey ...
python3 bench/scripts/samply-report.py bench/results/profiling/samply-js_16x16_d32.json.gz
```

Self time, all 13 threads folded together, each sample weighted by its `threadCPUDelta`
rather than counted (the macOS sampler drops samples under load; this profile has threads
with equal CPU time and a 3x spread in sample count).

**Note on inclusive time: do not trust it here.** rayon steals work, so a stolen closure's
stack is rooted at the worker thread rather than at the function that spawned it.
`msm_g1` reads 25% inclusive in a proof that is 88% MSM. Self time is sound.

| group | js_16x16_d32 | js_8x8_d32 |
|---|---:|---:|
| arkworks `Fp` (ark-ff) | 62.80% | 62.64% |
| rayon scheduler + inlined `par_iter` bodies | 11.33% | 10.20% |
| arkworks curve (ark-ec) | 9.86% | 11.05% |
| idle / spin (`swtch_pri`, cond wait) | 7.85% | 8.21% |
| arkworks `Fq2` (G2 tower) | 3.89% | 3.76% |
| g16-ntt | 3.02% | 3.10% |
| g16-zkey | 0.52% | 0.36% |
| g16-msm (`prescan` only; everything else inlined) | 0.12% | 0.09% |

Top functions by self time on `js_16x16_d32`:

| % | ms | function |
|---:|---:|---|
| 27.64 | 5565 | `MontBackend::mul_assign` |
| 20.50 | 4129 | `Fp::sum_of_products` |
| 11.81 | 2378 | `MontBackend::square_in_place` |
| 10.90 | 2194 | `rayon::iter::plumbing::bridge_producer_consumer::helper` |
| 8.58 | 1728 | `Projective::add_assign<Affine>` (mixed add) |
| 6.41 | 1292 | `swtch_pri` (rayon spin) |
| 2.91 | 587 | `g16_ntt::butterflies` |
| 1.94 | 391 | `QuadExtField::square_in_place` |
| 1.80 | 362 | `QuadExtField::double_in_place` |
| 1.69 | 340 | `MontBackend::sub_assign` |
| 1.28 | 258 | `Projective::add_assign<&Projective>` (full add) |

Two symbols need disambiguating, and `--callers` does it:

* **`Fp::sum_of_products` (20.5%) is two different things.** arkworks uses it both for the
  last step of every short-Weierstrass addition (`ark-ec-0.5.0/src/models/short_weierstrass/group.rs:404,529`)
  and for every `Fq2` multiplication (`ark-ff-0.5.0/src/fields/models/quadratic_extension.rs:656`).
  The caller breakdown shows both, so this line is G1 curve arithmetic *and* G2 tower
  arithmetic together.
* **`bridge_producer_consumer::helper` (10.9%) is not scheduler overhead.** `window_chunk`
  and `pippenger` do not exist as symbols in the binary (`nm target/profiling/g16 | grep
  window_chunk` is empty); they are inlined into the rayon closure. That 10.9% is the MSM
  bucket loop's own code: signed-digit recoding, bucket indexing, the zero-digit branch.
  The upper bound on the part of it that is `compute_h`'s `par_iter` bodies is 1.4%,
  because gather plus pointwise is 7.67 ms of a 652 ms proof even at perfect scaling.

---

## 4. The instruction stream, and what it means for the field lane

```
cargo asm -p g16-ntt --lib 27 --simplify   # index 27 is g16_ntt::butterflies
```

Full dump in `asm-butterflies.txt`. One butterfly is `t = y * w; y = x - t; x += t`, so
one `Fr` multiply, one modular subtract, one modular add. The compiler emits **412
instructions** for it:

| | count |
|---|---:|
| `mul` | 37 |
| `umulh` | 32 |
| `adds` / `adcs` / `cinc` / `adc` | 105 |
| `cmp` / `cmn` / `cset` | 48 |
| **conditional branches** (`b.hs`, `b.ne`, `b.lo`, `b.ls`, `b.hi`, `b.eq`) | **27** |
| `csel` | 2 |

**Yes, the compiler emits the conditional subtraction the source implies, and it emits it
as a branch, not as a select.** After the Montgomery reduction the code walks a four-limb
lexicographic comparison against the modulus (`LBB35_7` -> `LBB35_10` -> `LBB35_13` ->
`LBB35_16`), branching at each limb, and only takes the subtract path on the branch that
falls through to `LBB35_16`. The modular subtract that follows does the same thing again
for its borrow fixup. A Montgomery reduction output is uniform in `[0, 2p)`, so the first
of those branches is close to a coin flip on real data. There are 27 of them per
butterfly against 2 `csel`.

That is the concrete, actionable version of the review's "hot butterflies use fully
generic field operations": the reduction is not just present, it is unpredictably
branchy. A branch-free `csel`/`sbcs` reduction, or lazy reduction that keeps values in
`[0, 2p)` across the butterfly, attacks exactly this.

### Cost of every primitive the prover is made of

```
cargo build --release -p g16-core --example field_ops
REPS=9 ./target/release/examples/field_ops
```

Single threaded, best of 9, 2^20 operations per pass. Latency is a dependency chain;
throughput uses four independent accumulators. Cycles at 3.68 GHz.

| operation | lat ns | lat cyc | thr ns | thr cyc |
|---|---:|---:|---:|---:|
| `Fr` mul (Montgomery, 4 limbs) | 16.18 | 60 | 13.68 | 50 |
| `Fr` add (+ conditional subtract) | 1.35 | 5 | 1.19 | 4 |
| `Fr` sub (+ conditional add) | 2.58 | 10 | - | - |
| `Fq` mul | 15.85 | 58 | 13.45 | 49 |
| `Fq2` mul (G2 tower) | 56.77 | 209 | - | - |
| `Fr::into_bigint` + add | 5.60 | 21 | - | - |
| G1 mixed add (proj += affine) | 250.6 | 922 | 247.2 | 910 |
| G1 full add (proj += proj) | 321.3 | 1182 | - | - |
| G2 mixed add | 809.3 | 2978 | 893.6 | 3289 |
| G2 full add | 996.7 | 3668 | - | - |

Three things fall out of this table.

**A curve addition does not overlap with itself.** G1 mixed add throughput (247 ns) is
indistinguishable from its latency (250 ns): four independent additions run no faster
than four dependent ones. The bucket loop is therefore not limited by the dependency
chain between buckets, and reordering or software-pipelining the bucket updates will buy
nothing. Whatever the mixed add is saturating, it is saturating it already.

**About 40% of a mixed add is not multiplication.** `madd-2007-bl` is 7M+4S, so 11
multiply-class `Fq` operations, which at the 13.45 ns throughput above is a 148 ns floor.
Measured 247 ns. The other 99 ns is the modular adds and subtracts with their branchy
fixups, and the moves between them.

**The `asm` feature is doing nothing on this machine.** The workspace root declares
`ark-ff = { ..., features = ["asm"] }`, and every one of those assembly paths is gated on
`target_arch = "x86_64"` (`ark-ff-0.5.0/src/fields/models/fp/montgomery_backend.rs:157,165,230,237`).
On Apple silicon the "asm" build is the generic Rust build. That is a root `Cargo.toml`
observation and this lane did not change it.

### The whole MSM is one number times another

Per proof at 140,261 constraints:

* A, B->G1, L: 138,292 general scalars, c=13, 20 windows -> 8.30M G1 mixed adds
* H: 262,144 general scalars, c=15, 17 windows -> 4.46M G1 mixed adds
* B->G2: 138,292 general scalars, 20 windows -> 2.77M G2 mixed adds

12.76M x 247 ns + 2.77M x 894 ns = **5.62 s of CPU**. Observed: 574 ms of wall at ~9.4
cores busy = 5.4 s. The model closes to within 4%.

So bucket reduction, signed-digit recoding, prescan, allocation and window sizing are all
rounding error, and **the MSM is exactly 15.5 million curve additions**. Every candidate
optimisation should be judged on whether it reduces that count or the 247 ns.

---

## 5. Parallel efficiency

```
REPS=25 ./target/release/examples/ntt_shape     # bottom section
```

One forward NTT and one G1 MSM at n = 2^18, each run inside a fixed-size rayon pool
(constructed inside the pool, so `rayon::current_num_threads()` reports it):

| threads | NTT ms | speedup | MSM ms | speedup |
|---:|---:|---:|---:|---:|
| 1 | 63.0 | 1.00x | 1273.1 | 1.00x |
| 2 | 33.1 | 1.90x | 663.0 | 1.92x |
| 4 | 17.3 | 3.65x | 371.3 | 3.43x |
| 8 | 10.2 | 6.17x | 220.9 | 5.76x |
| 12 | 10.6 | 5.95x | 175.2 | 7.26x |

Both saturate well short of 12. The honest ceiling on this machine is not 12x either: an
M2 Max has 8 performance and 4 efficiency cores, and E-cores run integer work at roughly a
third of P-core throughput, which puts the ceiling near **9.3x**. That third is a
platform figure, not something this lane measured -- macOS gives no core affinity control,
so it could not be. Treat 9.3x as approximate. Against it the MSM reaches 78% and the NTT
64%, and the direction of the gap does not depend on the exact ratio.

The whole-proof number agrees: 9.41 cores busy out of 12 (section 1), of which the CPU
profile attributes 7.85% to spinning and waiting. Net, roughly **28% of the machine is not
doing arithmetic during a proof.**

For reference, the repo's own comparison against arkworks
(`cargo test -p g16-msm --release --lib -- --ignored --nocapture`):

```
2^16 G1  c=13  ours 46.14 ms  ours(1 thread) 385.21 ms  arkworks(1 thread) 377.27 ms  ratio 8.18x mt, 0.98x st
2^18 G1  c=15  ours 172.45 ms ours(1 thread) 1259.59 ms arkworks(1 thread) 1351.05 ms  ratio 7.83x mt, 1.07x st
```

Single threaded we are arkworks, to within 7%. The multithreaded advantage is entirely
rayon. There is no algorithmic edge to lose and none to coast on.

---

## 6. The NTT

```
REPS=25 ./target/release/examples/ntt_shape
```

| log n | n | iNTT ms | NTT ms | coset shift ms | bit-reversal ms | bitrev % of a transform |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 16,384 | 0.960 | 0.881 | 0.075 | 0.055 | 6.0% |
| 15 | 32,768 | 1.578 | 1.494 | 0.138 | 0.136 | 8.9% |
| 17 | 131,072 | 5.271 | 4.980 | 0.463 | 0.643 | 12.6% |
| 18 | 262,144 | 10.968 | 10.349 | 0.878 | 1.267 | 11.9% |

The bit-reversal is reproduced identically in the example (it is private in `g16-ntt`) and
pinned by a test that checks it is the same permutation, so it cannot drift from the code
it stands in for.

**The six bit-reversal passes are 7.6 ms of a 653 ms proof: 1.16%.** That is the ceiling
on the review's "radix-2 NTT pays six full bit-reversal passes per proof". A Stockham or
mixed-radix rewrite that removed the permutation entirely and changed nothing else could
not do better than 1.16%.

The review's separate claim of a plausible 1.2x-1.8x NTT-only gain is worth capping too:
the whole NTT stage is 69.95 ms of 652.6 ms, so **even a perfect 1.8x on all six
transforms is 4.8% of the proof.**

`distribute_powers` is 0.878 ms per call at 2^18, three calls per proof, 2.6 ms = 0.40% of
the proof. Precomputing the `shift^i` table removes one multiply of the two per element,
so it is worth at most 0.2% and probably less once the table's memory traffic is paid.

Worth noting for whoever does touch the NTT: `bit_reverse_permute` is a plain serial `for`
loop inside a transform whose butterfly passes are all parallel, so its share of wall
clock is several times its share of CPU. That is part of why the NTT only reaches 6x.

---

## 7. Allocation

```
cargo run --release -p g16-cli --features dhat-heap --bin g16-dhat -- bench/artifacts/js_16x16_d32
python3 bench/scripts/dhat-report.py bench/results/profiling/dhat-js_16x16_d32.json
```

The binary is behind an off-by-default feature with `required-features`, so a release
build does not contain it: `nm target/release/g16 | grep -c dhat` returns 0.

`js_16x16_d32`, phase by phase:

| phase | allocations | MiB allocated |
|---|---:|---:|
| zkey load | 247 | 94.77 |
| witness load | 7,273 | 20.15 |
| prepare (backend) | 1 | 0.00 |
| prove #1 (cold caches) | 4,673 | 187.54 |
| prove #2 (warm) | 4,667 | 179.54 |

Peak live heap 158.0 MiB in 432 blocks. `js_8x8_d32` is 2,462 allocations and 103.3 MiB
per warm proof, peak 81.6 MiB.

By owner, `js_8x8_d32`, two proofs (so halve for per-proof):

| % total | MiB | blocks | owner |
|---:|---:|---:|---|
| 36.49 | 97.85 | 7,119 | rayon `helper` -- i.e. the inlined `window_chunk` bucket arrays |
| 21.07 | 56.50 | 4,528 | `g16_msm::prescan::{{closure}}` |
| 8.78 | 23.55 | 5 | `g16_zkey::read_g1_section` |
| 8.70 | 23.33 | 32 | `msm_g1` (window sums, merged prescan output) |
| 5.97 | 16.00 | 4 | `compute_h` |
| 5.97 | 16.00 | 4 | `CpuCircuit::gather` |
| 5.49 | 14.73 | 8 | `g16_zkey::read_coefficients` |

The standing review is right about the pattern and this confirms it: five MSMs per proof,
each allocating prescan vectors, a merged output, one bucket array per window, and window
sums. The question it could not answer is what that costs.

```
cargo build --release -p g16-core --example alloc_cost
REPS=20 ./target/release/examples/alloc_cost
```

Reproducing the exact pattern for one `js_8x8_d32` proof. That circuit's domain is 2^17,
so every MSM lands on c=13: four G1 MSMs x 20 windows x 4096 buckets and one G2 MSM the
same, 30.0 MiB of G1 buckets and 15.0 MiB of G2:

```
bucket arrays, allocate + fill + free:     0.895 ms
same buffers reused, re-zeroed only:       0.852 ms
prescan push + merge:                      0.726 ms   into preallocated: 0.547 ms
```

**Reusing the bucket scratch is worth 0.043 ms per proof.** Note the buffers cannot be
`calloc`'d away: arkworks' projective zero is `(0, 1, 0)` and `1` in Montgomery form is
`R mod p`, so a reuse scheme still has to write every element back, and the fill is the
part that costs. The saving is the allocator call and the page faults, and there are only
~2,460 of them across a 350 ms proof.

The prescan's push-and-merge is worth more, 0.18 ms per MSM, 0.89 ms per proof.

Total prize for "reuse MSM scratch buffers": **0.93 ms of a 352 ms proof, 0.27%.**
Refuted as an optimisation worth building. Caveat: these are single-threaded measurements
and the real prover allocates from 12 threads at once, but at ~205 large allocations per
thread per proof, allocator contention cannot turn 0.27% into something that matters.

---

## 8. Per-annotation timing

```
cargo run --release -p g16-core --features hotpath --example hotpath_prove -- bench/artifacts/js_8x8_d32 12
```

hotpath annotations in `g16-core` only, behind an off-by-default feature. Instrumented, so
these are shapes rather than timings; use section 1 for numbers. P50 over 12 reps:

| region | calls | P50 |
|---|---:|---:|
| stages 5-9 msms | 12 | 353.4 ms |
| s9 MSM H -> G1 | 12 | 328.2 ms |
| s6 MSM B -> G2 | 12 | 327.9 ms |
| s8 MSM L -> G1 | 12 | 319.8 ms |
| s5 MSM A -> G1 | 12 | 307.5 ms |
| s7 MSM B -> G1 | 12 | 298.8 ms |
| stages 0-4 compute_h | 12 | 75.2 ms |
| s1 iNTT (x3) | 36 | 12.21 ms |
| s3 NTT (x3) | 36 | 10.62 ms |
| s2 coset shift (x3) | 36 | 0.92 ms |
| s0 gather A and B | 12 | 1.61 ms |
| **s11 blind (6 scalar mults)** | 12 | **611.8 us** |
| s11 normalise (2 batch inversions) | 12 | 9.3 us |

The five MSM rows overlap: each of the five occupies almost the whole 353 ms window,
which is what "1.02x from the join nest" looks like from the inside.

The last two rows explain the constant 0.57 ms `assemble` from section 1. It is
**six variable-base scalar multiplications**, not the batch inversion, which is 1.5% of the
stage. That cost is independent of circuit size, so it is 0.09% of a 140K proof and 63%
of a 2-constraint one.

This measurement caught a bug in its own harness, which is worth recording: with small
fixed blinders (`Fr::from(0x5eed)`) the stage measured 74 us, because `mul_bigint` skips a
scalar's leading zero bits. Full-width blinders reproduce the real 611 us. An assertion
now guards it.

---

## 9. Ranked, by expected value

Prize is quoted against a warm `js_16x16_d32` proof, 652.6 ms.

### 1. Make a curve addition cost less. Prize: up to 1.55x. Confidence: high on the measurement, medium on the achievable fraction.

88% of the proof is MSM, the MSM is 15.5M curve additions, and a G1 mixed add costs 247 ns
against a 148 ns floor set by its own 11 `Fq` multiplications. The 99 ns difference is
modular add/sub with branchy conditional fixups (27 conditional branches per butterfly in
the disassembly; the same reduction sits inside every curve-add step). Closing it entirely
would take the MSM from 574 ms to 344 ms and the proof to 422 ms, **1.55x**.

*For the field lane specifically:* the target is not the Montgomery multiply's 68 multiply
instructions, which are near the machine's limit at 50 cycles throughput. The target is
everything wrapped around them -- the four-limb branchy compare against the modulus after
every single operation. Branch-free `csel`/`sbcs` reduction, or lazy reduction keeping
values in `[0, 2p)` between operations, is where the 99 ns lives. `Fr` sub at 10 cycles
versus `Fr` add at 5 is the same effect visible in a two-instruction operation.

### 2. Batch affine bucket accumulation. Prize: plausibly 1.3-1.5x. Confidence: medium. Not measured here.

Same 15.5M additions, attacked from the other side: replace the 11-multiply Jacobian mixed
add with affine addition under a batched inversion (~6 multiplies amortised). The
measurement that supports this is that curve-add throughput equals curve-add latency
(247 vs 250 ns), so the bucket loop has no spare overlap to exploit and the only way down
is fewer field operations per update. This lane did not implement it, so the ratio is from
the formula, not from this machine.

### 3. Parallel efficiency. Prize: up to 1.24x. Confidence: high on the gap, medium on closing it.

One MSM at 2^18 reaches 7.26x on 12 threads against a realistic 9.3x ceiling for 8P+4E.
The proof as a whole occupies 9.41 of 12 cores and spends 7.85% of that spinning. Getting
the MSM from 7.26x to 9.3x takes 574 ms to 448 ms and the proof to 527 ms. The likely
lever is chunk sizing that accounts for P-core/E-core asymmetry, plus a tail that does not
leave one window running alone. Part of the gap is structural (the NTT's `log n`
fork/join barriers and its serial bit-reversal) and will not come back.

### 4. The whole NTT. Prize: at most 4.8%. Confidence: high on the cap.

69.95 ms of 652.6 ms. A perfect 1.8x on all six transforms is 31 ms. Within that, the six
bit-reversal passes are 7.6 ms (1.16%), so a Stockham rewrite that only removes the
permutation is capped at 1.16%. Worth doing only after 1-3.

### 5. Skip infinity bases in the prescan. Prize: 0.8%. Confidence: medium.

34% of the B query is the identity point on every circuit measured, and `prescan` never
looks at a base. Cheap to implement (one `is_zero()` in the prescan loop) but small,
because the add already short-circuits; what is saved is 47,803 Montgomery reductions and
their window scans, twice. The 0.8% is a residual between two measured rates rather than a
directly measured saving, hence medium confidence.

### 6. Stage 11's six scalar multiplications. Prize: 0.09% at 140K, 60% at 2 constraints. Confidence: high.

Only worth touching if small circuits are a target. wNAF or a fixed-base table for
`delta_g1`/`delta_g2` would cut most of the 611 us.

### Checked and NOT worth it

* **Reuse MSM scratch buffers** (review: small win, high confidence). Measured **0.27%**,
  and the naive reuse implementation was *slower* than allocating fresh until the re-zero
  was written with `fill`. Section 7.
* **Fast-path `-1` and small constants** (review: medium win). **7 scalars out of 140,824**,
  zero on half the ladder. 0.005%. Section 2.
* **Precompute coset powers in `distribute_powers`** (review: small win, high confidence).
  The whole call is 0.40% of the proof and the fix removes at most half of it. Section 6.
* **GLV endomorphism for G1** (review: large win, confidence medium). Not measured, but the
  repo's own cost model in `window_size` refutes it: with `cost = ceil(bits/c) * (n + 3 *
  2^(c-1))`, the A/B/L MSM at n=138,292 and 255 bits costs 3.012M at c=13; GLV gives
  n=276,584 at 129 bits, whose best is 2.889M at c=13. **4.3%**, before paying for 138,292
  endomorphism applications and scalar decompositions. The H MSM comes out at 2.5%. GLV
  halves the window count and doubles the point count, and in Pippenger with `n >> 2^c`
  those cancel; only the bucket-reduction term improves. This is model arithmetic, not a
  measurement, but it is the repo's own model and it should be checked before anyone
  spends a week on it.
* **Window sizing.** The cost model uses a weight of 3 on the bucket-reduction term from a
  claimed 1.5x full-add-to-mixed-add ratio. Measured ratio is 1.30 (321.3 vs 247.2 ns), so
  the correct weight is 2.6. Re-minimising with 2.6 still gives c=13 at n=138,292 and c=15
  at n=262,144. The window sizes are right; the constant in the comment is slightly off and
  changes nothing.

---

## 10. Things that need a file outside this lane

This lane edited only `crates/g16-core/**`, `crates/g16-cli/**`, `bench/scripts/**` and
`bench/results/profiling/**`. Four things want changing elsewhere:

1. **`crates/g16-msm/src/lib.rs:13`** -- "over 99% of witness scalars are 0 or 1" and "worth
   about 5.1x on this workload". Measured 1.8%, cap 1.018x. The doc block is load-bearing:
   it is the stated justification for the prescan design and it is cited by the security
   analysis.
2. **`crates/g16-core/src/cpu.rs:255`** -- "the four cheap witness MSMs (mostly 0/1
   scalars)". This file *is* in this lane, but the sentence is about `g16-msm`'s behaviour
   and correcting it alongside (1) keeps the two consistent, so it was left for whoever
   fixes (1).
3. **Root `Cargo.toml`** -- `ark-ff` is declared with `features = ["asm"]`, which is
   x86_64-only and a no-op on every Apple machine this project benchmarks on. It is not
   wrong, but it reads as if the field arithmetic is assembly-accelerated here, and it is
   not.
4. **`crates/g16-msm/src/lib.rs:305`** -- `pippenger` only `debug_assert`s that bases and
   scalars are the same length and otherwise `min()`s them, so a mismatch silently
   truncates in release. Every probe this lane wrote asserts lengths explicitly for that
   reason. Flagged by the standing review too; repeating it because a profiling harness is
   exactly the kind of caller that builds MSM inputs by hand.

---

## Reproducing all of it

```sh
cargo build --release
cargo build --profile profiling                       # for samply, symbols resolve

bench/scripts/profile-wallclock.sh 10 2               # section 1

cargo build --release -p g16-core --examples
REPS=5  ./target/release/examples/msm_shape bench/artifacts   # section 2
REPS=9  ./target/release/examples/field_ops                   # section 4
REPS=25 ./target/release/examples/ntt_shape                   # sections 5, 6
REPS=20 ./target/release/examples/alloc_cost                  # section 7

samply record --save-only --no-open --unstable-presymbolicate --rate 4000 \
  --iteration-count 3 -o bench/results/profiling/samply-js_16x16_d32.json.gz -- \
  ./target/profiling/g16 prove \
    --zkey bench/artifacts/js_16x16_d32/circuit.zkey \
    --witness bench/artifacts/js_16x16_d32/circuit.wtns \
    --proof /tmp/p.json --public /tmp/pub.json
python3 bench/scripts/samply-report.py \
  bench/results/profiling/samply-js_16x16_d32.json.gz --top=30   # section 3
python3 bench/scripts/samply-report.py \
  bench/results/profiling/samply-js_16x16_d32.json.gz --callers=sum_of_products

cargo asm -p g16-ntt --lib 27 --simplify              # section 4, butterflies

cargo run --release -p g16-cli --features dhat-heap --bin g16-dhat -- \
  bench/artifacts/js_16x16_d32 bench/results/profiling/dhat-js_16x16_d32.json
python3 bench/scripts/dhat-report.py \
  bench/results/profiling/dhat-js_16x16_d32.json --top=12         # section 7

cargo run --release -p g16-core --features hotpath --example hotpath_prove -- \
  bench/artifacts/js_8x8_d32 12                                   # section 8
```

`cargo flamegraph` was not run: on macOS it needs `dtrace` under `sudo`, and this lane was
told to report rather than run anything needing sudo. `samply` needs neither and resolves
Rust inline frames better. For a flame graph from the same recording:

```sh
python3 bench/scripts/samply-report.py   bench/results/profiling/samply-js_16x16_d32.json.gz   --folded=/tmp/folded.txt --top=1
inferno-flamegraph /tmp/folded.txt > /tmp/g16.svg   # or flamegraph.pl
```

The collapsed-stack files are not committed: they are 2.5 MB each and one command away
from the `.json.gz` recordings that are.
