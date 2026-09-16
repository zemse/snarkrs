Second opinion on `0a2e6a1`, `25536d4`, and `75ed942`, based on `75ed942`.
All timings below come from the campaign, not measurements in this worktree. The
review used source inspection, integer arithmetic, and CPU tests. No Metal device
was requested. The production coefficients are unchanged.

**Findings in the three commits**

1. **A real SIMD-width bug in `25536d4`, outside the M2's 32-lane path.**
   The shader's scan bounds, output indexing, allocation, and host bases all assumed
   32. At 64 threads and SIMD width 16 there are four SIMD groups, which write eight
   points, but the allocation reserves four per threadgroup. The shader also strides
   successive threadgroups by four points, so their writes overlap. At SIMD width
   64 the scan omits the upper half of the group and the host expects an unwritten
   second pair. This is a correctness and memory-safety issue, not an M2 speed claim.
   The patch queries the reduce pipeline's `thread_execution_width`, uses
   `threads_per_simdgroup` in MSL, and derives allocation and host bases from that
   width. Apple's [SIMD group documentation](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups)
   explicitly makes this a pipeline property.

2. **No arithmetic bug found in the scan at width 32.** For a segment starting at
   `lo`, its loop emits `P = sum (j-lo+1) B_j` and `Q = sum B_j`. Within a SIMD
   group, `lo = base + t*l`, where `l` is the segment length. With suffixes
   `S_t = sum_{u>=t} Q_u`, each `Q_u` occurs exactly `u` times in
   `sum_{t>=1} S_t`. Consequently the emitted pair is
   `C = sum P_t + l * sum_{t>=1} S_t`, `Q = S_0`; adding `base*Q` is precisely
   the required weighted sum. The host applies the same identity a second time
   when bases are uniformly spaced, and explicit bases otherwise. G2's existing
   `sum(P + lo*Q)` is the same identity without the scan. Empty segments contribute
   the identity, and clipped final chunks still have the prescribed segment bases.
   The host's uniform-path comment is too broad: unforced tiny shapes such as four
   buckets are not uniform. Its explicit-base fallback handles them correctly.

3. **No arithmetic bug found in the wide merge.** Plain and wide dispatches own
   disjoint rows. A row's slice interval is `floor(start/L)..floor((end-1)/L)`;
   striding it by thread count partitions every slice exactly once. The two tagged
   spill slots are tested independently, then the tree and original bucket are
   added once. The tree's `tid+s<tcount` check handles non-power-of-two thread
   counts. The empty-row return is uniform within a threadgroup, so no barrier
   divergence follows. A one-bit top window owns one row. A one-window host-tail
   plan bypasses both GPU merge dispatches and folds the spill slots on the host.

4. **The cost model prices a different merge path from the planner on small inputs.**
   `window_size_for` always uses the default slice 64; `Plan::new` can select 8.
   At `m=512, recode_bits=255`, the selected `c=5` has 51 windows, 16 top buckets,
   and modeled span `512/16/64 = 0.5`. Its modeled fat term is 13.5 us. The
   dispatched slice is 8 and span is exactly 4, so the wide kernel engages, whose
   own model would charge 101.6875 us. Total candidate cost changes from
   116.05488 to 204.24238 us. Applying actual slice lengths to all candidates
   makes `c=8` win at 170.19616 us. This proves internal inconsistency; it does
   not prove `c=8` faster on hardware. The slice mismatch predates these commits,
   but the newly introduced branch makes it a wrong-kernel assumption too.

5. **The claimed WIDE_US bracket is only stable at the measured anchors.** For
   `m=131072, recode_bits=40`, both candidates really use slice 64:

   | candidate | windows | buckets/window | span | cost with WIDE_US=100 |
   |---|---:|---:|---:|---:|
   | c=10 | 4 | 512 | 4 | 1845.43662 us |
   | c=14 | 3 | 8192 | 1 | 1841.69184 us |

   `WIDE_US=60` selects c=10; 100 or 160 selects c=14. The crossover is
   96.25522 us, inside the claimed bracket, and the chosen coefficient leaves
   only 3.74478 us margin. At `m=16384, bits=14`, the same bracket flips c=7 to
   c=14. The arithmetic reproducer lists more examples. These are demonstrated
   sensitivities, not measured regressions.

6. **Span four and the single ROW_US are unvalidated extrapolations.** The wide
   measurements were at estimated spans 16 and 32, not 4. Immediately below four
   the modeled fat cost tends to 108 us; at four it drops to 101.6875 us. Real
   slice spans are also alignment dependent: a run of exactly `4L` entries can
   intersect five slices. The model ignores `G16_METAL_MSM_WIDE`, including zero,
   even though the planner honors it. It charges a GPU merge/reduce price to
   one-window host-tail plans that dispatch neither. ROW_US=0.022 changes
   `(m=512,bits=65)` from c=5 to c=6 compared with 0.025, despite that shape using
   small slices and a different reduction regime from the calibration. G2 keeps
   the old reduce but uses the repriced selector. None of these establishes a
   faster replacement coefficient, so this patch does not refit one.

7. **The local tiny_mul slowdown cannot be a changed window choice.** Its witness
   is `[1,105,3,5,7,15]`, hence the chunk bit bound is 7 and recode bound is 8.
   A/B have five general scalars; L has two. Both pre-refit and current models
   choose c=4, two windows. H has eight entries and both choose c=3, 85 windows.
   No candidate winning any of these plans reaches the modeled wide threshold.
   Forcing wide off therefore cannot identify the refit as the cause. Plausible
   alternatives are the scan commit's larger readback and parallel window fold,
   code generation, or measurement conditions. An isolated revision comparison
   is needed to attribute 0.1 ms. The host-only audit checks the actual fixture;
   its old L report incorrectly used the subrange's bit bound instead of the
   classification chunk's bound. That reporting error is fixed here too.

CPU tests cover every individual bucket weight in small shapes, partial SIMD
groups, widths 8/16/32/64, thread counts including 7/31/48, nonuniform chunks,
two buckets, all-zero buckets, real G1 cancellation, both host fold branches,
wide spans 3/4/5, and signed recoding with one window and a one-bit top carry.
They check the algebra and host implementation, not Metal execution or its compiler.

**Batch-affine verdict**

Do not rewrite the current dependent-chain accumulator around per-threadgroup
inversions. A global affine reduction tree is a separate, plausible research
prototype, but the available measurements do not justify a production rewrite.

Let A be the number of bucket visits: 2,621,440 / 5,242,880 / 8,912,896. XYZZ
prices a nontrivial visit at 10 products. Affine addition with a supplied inverse
costs 2M+1S; Montgomery batching costs `3(K-1)M + I` for K denominators. Thus
the ideal price is `6 + (I-3)/K`. Doublings cost one additional square;
cancellations must bypass division, with denominator one participating in the
batch product. Same-bucket updates cannot snapshot the same accumulator twice.

| products/visit | keccak_128 | keccak_256 | keccak_512 |
|---|---:|---:|---:|
| 10, XYZZ arithmetic budget | 5.878 ms | 11.755 ms | 19.984 ms |
| 7, ideal affine budget | 4.114 ms | 8.229 ms | 13.989 ms |
| 6, inversion-free limit | 3.527 ms | 7.053 ms | 11.990 ms |
| savings at 7 products | 1.763 ms | 3.527 ms | 5.995 ms |

These are work estimates, not strict floors for the existing kernel: each run's
first point copies into the identity without a mixed add, and zero digits and
infinity bases cost less. In particular, the supplied 10-product arithmetic and
accum times imply 82.7/82.4/85.2% utilization before subtracting submission floors,
not 84 to 93%. This does not reopen the settled tuning sweeps.

For q, binary exponentiation by q-2 takes 253 squarings and 109 multiplies, 362
products. Even ignoring SIMD waste, K=256 costs 7.402 products/add and K=512
costs 6.701. A product tree retaining K leaves and K-1 internal nodes needs
`32*(2K-1)` bytes: 16,352 bytes at K=256 and 32,736 at K=512. The latter leaves
32 bytes of threadgroup memory for everything else.

Worse, one root inversion occupies only one lane of a 32-lane SIMD group. It
still issues that group's arithmetic instructions. Against a full-lane ALU roof,
its lane-slot cost is approximately `32*362/K`, not `362/K`. Including the
product tree and affine update gives about 51.24 or 28.62 product-equivalents
at K=256 or 512. To get below 10 requires K about 2,896, which cannot fit this
tree in 32 KB. Another threadgroup can hide latency but cannot turn the 31 masked
lanes of an issued instruction into useful inversions. A faster inversion or a
cooperative inversion algorithm would need its own measurement; the 362 count
is for the explicit binary-exponentiation design, not a universal lower bound.

A separate inversion dispatch can put independent roots in adjacent lanes. That
requires global scratch and synchronization. A favorable two-pass accounting per
affine addition is: prefix pass reads two 64-byte points and writes denominator
and prefix, 192 bytes and about one product; apply pass reads those points and
64-byte scratch and writes the 64-byte result, 256 bytes and about five products.
Root processing, metadata, exceptional cases, and compaction are omitted. With
products and memory overlapped within each pass, its optimistic roofline is

`A * [max(1/F,192/B) + max(5/F,256/B)]`, with F=4.46e9 products/s.

| effective bandwidth in both passes | keccak_128 | keccak_256 | keccak_512 |
|---|---:|---:|---:|
| 101.4 GB/s | 11.582 ms | 23.164 ms | 39.378 ms |
| 330.7 GB/s | 4.461 ms | 8.922 ms | 15.167 ms |

Root inversions must then be added back. At K=256, even perfectly utilized
362-product inversions cost another 0.831 / 1.662 / 2.826 ms; at K=512 they
cost 0.416 / 0.831 / 1.413 ms. Thus the high-bandwidth advantage over the XYZZ
arithmetic budget narrows to about 0.586 / 1.171 / 1.991 ms for K=256, or
1.001 / 2.002 / 3.404 ms for K=512, before synchronization and other overhead.

The traffic is 1.174 / 2.349 / 3.993 GB over the whole accumulation, versus
0.168 / 0.336 / 0.570 GB of baseline point reads, before scratch/spill differences.
Even with free root inversions this design needs roughly 200 GB/s to beat the
10-product arithmetic budget; including ideal root arithmetic raises the crossover
to about 239 GB/s at K=256 or 215 GB/s at K=512. The device reaches that regime
only in the largest supplied streaming
measurement; it is not valid to apply 330.7 GB/s to every shrinking round.

Keeping one affine state per original slice preserves collision freedom, even
when two slices belong to the same bucket, because they own separate partials.
But it needs up to 63 dependent update rounds after initialization. At least
three dispatches per round are about 0.38 to 0.57 ms at the supplied 2 to 3 us
dispatch cost, with synchronization and poorly utilized inversion batches on
top. Those rounds have only 40,960 / 81,920 / 139,264 slice states, precisely
where the supplied bandwidth measurements do not support a 330.7 GB/s assumption.
These are extra dispatches within a command buffer, not 0.129 ms commits each.

A segmented affine pair-reduction tree is the better alternative to test if this
campaign continues: disjoint pairs from the same bucket are legal independent
additions, and a 64-entry slice takes six levels instead of 63 dependent rounds.
It must compact or carry odd runs, preserve bucket boundaries, and retain the
two-spill ownership contract or replace it explicitly. It still pays the global
traffic above, while later levels shrink into slower bandwidth regimes. At a
uniform 330.7 GB/s the modeled advantage over XYZZ's arithmetic budget is only
1.417 / 2.834 / 4.817 ms before inversions and these costs. That is enough for a bounded
prototype, not evidence that the full rewrite wins.

**Other opportunities, ordered by potential critical-path milliseconds**

| opportunity | keccak_128 | keccak_256 | keccak_512 | evidence |
|---|---:|---:|---:|---|
| affine tree at peak bandwidth, K=256..512 | 0.59..1.00 | 1.17..2.00 | 1.99..3.40 | conditional advantage over XYZZ arithmetic budget, no measured win |
| understand and improve reduce | unknown | unknown | unknown | 0.51 / 0.51 / 1.75 ms of nominal bulk arithmetic remains even with every weighting tax removed |
| digits to supplied bandwidth floor | at most about 0.27 | 0.42 | 1.08 | subtract the approximately 0.15 ms phase floor first |
| remove H ones scan | about 0.08 | 0.07 | 0.12 | phase work after floor subtraction; overlap means wall savings can be smaller |

The reduce's 0.95 ms is not unexplained zero work: two full additions over
81,920 buckets already price at 2,293,760 field products, or 0.514 ms, plus a
10.486 MB bucket stream, control flow, and the remaining tree. The corresponding
large shape has 278,528 buckets and 1.749 ms nominal bulk arithmetic. Changing
group count partitions nearly the same work, so an invariant time is consistent
with throughput. It does not identify the limiting instruction or prove a
bandwidth bottleneck. The old deleted-tax measurement and the current scan also
have different reduction tails. The new opt-in body probe independently measures
the same strided loads, Q alone, and P+Q, at fixed RG=8. It reports pipeline
limits and driver times without emitting an invalid proof. It omits both final
trees/scans and writes more partials than production; use its deltas to diagnose,
not its absolute time as a production prediction.

For digits, split the phase before attributing everything to atomics. Each of
count and scatter issues up to A atomics, but `msm_scan` performs 18 barriers
per 256-counter chunk: 288 per window at c=13, and 1,152 at c=15, with only
20/17 threadgroups. This is a proved barrier count and a suspected latency cost.
A block scan that scans per-thread totals once, or a SIMD-local scan with one
inter-SIMD combine, can remove most of these barriers without changing digit
representation. That proposal is about the integer prefix scan, not any of the
settled point-reduction experiments.

A dense per-tile histogram is unattractive: with 1,024 scalars/tile, partial
counts alone occupy 41.94 / 83.89 / 570.43 MB. Writing and reading them once
costs at least 0.254 / 0.507 / 3.450 ms even at 330.7 GB/s. A c=15 histogram
is also 64 KB and does not fit the threadgroup budget. At c=13, 1,024 uniform
digits occupy about 906 of 4,096 buckets, so local combining removes only about
12% of global count updates. Warp combining similarly saves about 0.4% in a
full c=13 window. Neither eliminates scatter reservations. Hot top-window rows
are a different distribution, but constitute only one window.

A compact single-pass count-and-scatter cannot know arbitrary row offsets before
the counts exist. Fixed reservations must handle the all-equal input, not just
average occupancy. Alternatives are block lists with overflow and compaction, or
emit-once plus radix sorting. Two radix passes already read and write about 32A
bytes of entries, 83.89 / 167.77 / 285.21 MB, before initial emission and count
tables. These might compete at the largest size, but are not a free removal of
one recode pass. Materializing digits alone removes no atomics and adds traffic.

The ones patch sets a common `separate_ones` parameter for count and scatter.
Classified witness inputs keep their gather. Unclassified inputs use their
existing capacity n and emit scalar 1 as one low-window entry. Their ones output
has length zero and no kernel reads or writes it. This works for arbitrary
device scalars, including all ones; no probability assumption about H is needed.
`G16_METAL_MSM_ROUTE_ONES=0` restores the old scan for comparison. A GPU correctness
test exercises all-zero, all-one, and mixed inputs for G1 and G2 with scratch reuse.

**Validation and requested measurements**

Locally passed: the full release `g16-msm` and `g16-core` suites, five Metal
CPU-only algebra/routing tests, and compilation of every `g16-metal` test target.
Runtime MSL compilation and GPU correctness remain untested here. In particular,
compilation of `audit_metal_vs_cpu` is not a pass of its concurrent-proof gate.

Run all commands from this worktree. Use a shell without unrelated `G16_METAL_*`
overrides. No new window or slice sweeps on the three keccak H shapes are requested.

1. Correctness first, especially the eight-concurrent-proof test. Preserve normal
   test concurrency. The fourth command repeats the critical audit suite explicitly.

   ```sh
   cargo test --release -p g16-metal
   cargo test --release -p g16-msm
   cargo test --release -p g16-core
   cargo test --release -p g16-metal --test audit_metal_vs_cpu
   ```

2. Build the CLI once, then interleave ones routing on/off in production mode.
   Return warm medians and minima from the CSVs. This also checks that the SIMD
   width fix has not disturbed the production keccak choices.

   ```sh
   cargo build --release -p g16-cli --no-default-features --features metal --bin g16
   for round in 1 2 3; do
     msm_trial=0
     for route in 0 1 1 0; do
       msm_trial=$((msm_trial + 1))
       G16_METAL_MSM_ROUTE_ONES=$route target/release/g16 bench \
         --backend metal --mode warm --reps 20 --artifacts bench/artifacts/csp \
         --variant keccak_128 --variant keccak_256 --variant keccak_512 \
         --variant sha256_128 --variant sha256_256 \
         --csv /tmp/codex-msm-route-$round-$msm_trial.csv
     done
   done
   ```

3. Split digits, keeping overlap off and the old ones path for direct comparison
   with the supplied phase table. Return each H count/scan/scatter driver time,
   adjacent wall time, and the printed c/window/cap. Each extra isolated phase
   adds its own submission floor; do not sum these into a production estimate.

   ```sh
   G16_METAL_OVERLAP=0 G16_METAL_CB_TIMES=1 G16_METAL_MSM_PHASES=3 \
     G16_METAL_MSM_ROUTE_ONES=0 target/release/g16 bench \
     --backend metal --mode warm --reps 10 --artifacts bench/artifacts/csp \
     --variant keccak_128 --variant keccak_256 --variant keccak_512 \
     > /tmp/codex-msm-digits.log 2>&1
   ```

4. Diagnose the reduce floor at fixed RG=8. The opt-in test validates its diagnostic
   outputs on the CPU and reports per-dispatch driver medians with 16 dispatches
   per command buffer. Mode 0 is a full-point checksum load, 1 computes Q, and 2
   computes both running sums and emits P.
   These are warm-cache diagnostics. Return all six medians and pipeline limits.

   ```sh
   cargo test --release -p g16-metal --lib msm::gpu_probes::review_reduce_body \
     -- --ignored --exact --nocapture
   ```

5. Test the two demonstrated model risks, not the settled keccak H widths.
   All synthetic results are checked against a CPU MSM. Return shape lines and
   medians. Repeat these loops interleaved if a difference is close to noise.

   ```sh
   for c in 5 8; do
     G16_REVIEW_N=512 G16_REVIEW_BITS=255 G16_METAL_MSM_C=$c \
       cargo test --release -p g16-metal --lib msm::gpu_probes::review_plan_shape \
       -- --ignored --exact --nocapture
   done
   for c in 10 14; do
     G16_REVIEW_N=131072 G16_REVIEW_BITS=40 G16_METAL_MSM_C=$c \
       cargo test --release -p g16-metal --lib msm::gpu_probes::review_plan_shape \
       -- --ignored --exact --nocapture
   done
   ```

6. Test the previously unmeasured wide threshold with fixed c and L. These sizes
   bracket the planner's estimated span four. Compare explicit enable/disable,
   with identical inputs and allocation shape, and return merge driver times.
   This is not a sweep of the already settled slice length.

   ```sh
   for n in 32767 32768 32769; do
     for wide in 0 1 4; do
       G16_REVIEW_N=$n G16_REVIEW_BITS=255 G16_METAL_MSM_C=13 \
         G16_METAL_MSM_L=64 G16_METAL_MSM_WIDE=$wide \
         G16_METAL_MSM_PHASES=2 G16_METAL_CB_TIMES=1 \
         cargo test --release -p g16-metal --lib msm::gpu_probes::review_plan_shape \
         -- --ignored --exact --nocapture
     done
   done
   ```

7. Attribute tiny_mul's 0.1 ms using the three revisions separately. This builds
   archive copies under a new temporary directory and reads this worktree's
   artifacts; it changes no checkout. First isolate the scan commit, then the
   model commit, using the same toolchain, flags, and interleaved order. Return
   warm medians/minima. No flags are needed to force a width because the natural
   choices are already identical.

   ```sh
   msm_review_root=$(mktemp -d /tmp/g16-msm-revisions.XXXXXX)
   msm_review_artifacts="$PWD/bench/artifacts"
   for rev in 0a2e6a1 25536d4 75ed942; do
     mkdir "$msm_review_root/$rev"
     git archive "$rev" | tar -xf - -C "$msm_review_root/$rev"
     CARGO_TARGET_DIR="$msm_review_root/target" cargo build --release \
       --manifest-path "$msm_review_root/$rev/Cargo.toml" \
       -p g16-cli --no-default-features --features metal --bin g16
     cp "$msm_review_root/target/release/g16" "$msm_review_root/g16-$rev"
   done
   for round in 1 2 3; do
     for rev in 0a2e6a1 25536d4 75ed942 75ed942 25536d4 0a2e6a1; do
       "$msm_review_root/g16-$rev" bench --backend metal --mode warm --reps 40 \
         --artifacts "$msm_review_artifacts" --variant tiny_mul
     done
   done
   ```

Host-only reproduction, which needs no measurement-machine time:

```sh
python3 bench/scripts/audit_metal_msm_model.py
cargo test --release -p g16-metal --lib msm::cpu_tests
cargo run --release -p g16-metal --example msm_audit -- bench/artifacts/tiny_mul
```

No end-to-end speedup is claimed for this branch before those GPU results arrive.
