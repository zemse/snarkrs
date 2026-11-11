# CUDA c13 structural counts

CPU-only trace of the synthetic `variable_work_phase_timings` inputs. The count
probe and GPU benchmark share their scalar generator. This is not a timing,
throughput measurement or GPU correctness result.

Run without a CUDA device:

```sh
cargo test --locked -p snarkrs-cuda --features cuda --lib -- \
  --ignored --exact msm::tests::variable_work_operation_counts --nocapture \
  > "$TMPDIR/msm-counts.log" 2>&1 &&
sed -n 's/^counts //p' "$TMPDIR/msm-counts.log"
```

The output is CSV containing only counts and shape metadata. The test asserts
its reduction-call, clear-byte and launch-count invariants before accepting
rows. Source: `crates/cuda/src/msm.rs`, `variable_work_operation_counts`.

## Inputs and geometry

Variable work, forced c13, 20 windows, 4096 buckets per window, 64 scalars per
accumulation slice. The scatter count assumes 4 MiB L2, explicitly reported as
`l2_assumed_bytes`; query the device before using this shape in a timing model.

| set | n | zero | one | general | slices | spill slots | digit entry capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 65536 | 0 | 0 | 65536 | 1024 | 40960 | 1310720 |
| mixed | 65536 | 13108 | 13107 | 39321 | 615 | 24600 | 786420 |
| uniform | 262144 | 0 | 0 | 262144 | 4096 | 163840 | 5242880 |
| mixed | 262144 | 52429 | 52429 | 157286 | 2458 | 98320 | 3145720 |

The mixed generator repeats zero, one, minus one, random, random. Counts use
actual generated values, not an assumption that random scalars avoid zero/one.
Digit capacity is an upper bound, not the actual emitted nonzero digit count.

## Fixed work per G2 job

- 14 kernel launches: count clear and histogram 2, scan 1, scatter 5, bucket clear
  1, accumulation 1, merge 2, reduce 1 and ones 1.
- Reduce: 160 blocks, 64 threads per block, 184160 addition call sites excluding
  offset-ladder internals, 10240 offset ladders and 7 barriers per block.
- Clear: 5242880 useful bytes stored over a 20971520-byte bucket span. Only the
  64-byte `zz` field of each 256-byte G2 record is written, not the full span.
- Readback: 49152 bytes at n=65536 and 73728 bytes at n=262144, including ones
  partials. Host folding and stream synchronization are separate work.

Addition call sites are not counts of generic field arithmetic. Identity,
equality and cancellation branches, and the offset ladders, must be accounted
for before deriving an arithmetic floor. No measured G2 or strided-store floor
is accepted yet.

## Collecting phase samples on the GPU host

The timing harness emits every measured sample after checking all warm and
measured answers against the CPU. The parser requires all 88 phase rows, exact
repetition counts and successful single-test completion. It rejects duplicates,
unexpected shapes, missing phases and invalid numeric samples. It does not
replace the independent full-proof oracle or attest the build provenance.

```sh
G16_CUDA_MEMORY_TRACE=0 G16_CUDA_MSM_C=13 \
G16_REVIEW_N=65536,262144 G16_REVIEW_SET=mixed,uniform G16_REVIEW_REPS=15 \
cargo test --locked -p snarkrs-cuda --features cuda --release --lib -- \
  --ignored --exact msm::tests::variable_work_phase_timings --nocapture \
  > "$TMPDIR/msm-phases.log" 2>&1 &&
python3 bench/scripts/cuda_phase_samples.py "$TMPDIR/msm-phases.log" --reps 15
```

For a cross-built batch, invoke the matching prebuilt library test executable
with the same test filter and environment instead of Cargo. `combine` and
`batch_wall` use host wall time; other phases use CUDA events. Isolated phases
insert waits and reuse scratch. The separate normal batch allocates scratch,
so its device time need not equal the sum of isolated phase medians. These
commands define the protocol; no GPU sample has yet been accepted for it.
