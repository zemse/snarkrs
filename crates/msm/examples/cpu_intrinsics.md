# CPU intrinsic probes

Two standalone examples, current dependencies only. Production paths unchanged.
No timings here establish an x86 result or an uncontended machine baseline.

## reconcile first

Signed Pippenger, batch-affine wide windows, XYZZ narrow windows, compact blocked
NTT and DIF/DIT cancellation already exist. Competitor versions of these are not
new implementation tasks. `RawFq::mul` and `RawFq2::mul` call the custom
`crates/field/src/raw.rs` arithmetic. Enabling ark-ff assembly does not replace it.

Existing support: groth16 `field_ops`, `trace_shape`, `msm_shape`, `ntt_shape`,
`hotpath_prove`, `alloc_cost`; msm `tests/field_cpu_throughput.rs`; field
`tests/raw_matches_ark.rs`; `bench/results/profiling/README.md`.
The historical field example probes ark Fq/Fq2, assumes a clock frequency, and
claims saturation at four chains. The old CPU throughput test finishes one whole
dependent chain before starting the next. Neither establishes interleaved raw-field
throughput. Historical profiles are leads, not this campaign's baseline.

## run

From an absolute checkout path, use its Cargo.toml as `--manifest-path`.
Build with `--locked --offline`; no dependencies or lockfile changes needed.

- `cargo run --release -p snarkrs-msm --example cpu_intrinsics`
- `PROBE_FIXTURE=/absolute/path/js_2x2_d32 cargo run --release -p snarkrs-groth16 --example cpu_trace`
- Repeat the trace for `js_16x16_d32`. Initial fixtures have 17,929 and 140,261 constraints.
  Read actual domain/query sizes from the trace, not from constraint counts.
- Tests: `cargo test -p snarkrs-msm --example cpu_intrinsics` and
  `cargo test -p snarkrs-groth16 --example cpu_trace`.

`cpu_intrinsics` emits JSONL, one provenance record then individual samples.
Configure with positive `PROBE_ROUNDS` (default 16384), `PROBE_REPS` (5),
`PROBE_PASSES` (16), `PROBE_THREADS` (available parallelism), and comma-separated
`PROBE_BYTES` (32768,1048576,16777216; positive multiples of eight).
Bytes are per buffer: copy retains two buffers, stream read one.
Record `PROBE_REV`, `PROBE_BUILD` (rustc version, target, profile, RUSTFLAGS), and
`PROBE_CONTENTION`. These strings are caller declarations, not attestation.
Output also records compiled ADX/BMI2 flags, architecture, OS, debug assertions,
configured threads and actual dedicated Rayon pool size. Primitive/memory probes
run on one caller thread by default; the join probe runs inside that pool.
`PROBE_ALL_CORE=1` enables additional aggregate samples; omitted or `0` preserves
the existing single-caller probes. Other values are configuration errors.
Keep CPU model, affinity/core class, governor, thermal state, input SHA-256s,
command and dirty-tree state alongside the output. No cycles inferred from GHz.

## measurements

- Actual Fr, RawFq and RawFq2 multiply, square and add; actual raw-inverse bridges
  through `RawCurve::raw_inv`; G1/G2 `Xyzz::madd` on the public production types.
  No visibility changes or source copies. Raw inverse measurements include the
  bridge to ark fields, exactly as batch-affine rounds use it.
- One dependent chain measures latency including loop cost. Four/eight chains are
  interleaved in the inner loop, with operation count `rounds * chains`.
  These are single-thread multi-chain throughput samples, not certified saturated
  throughput and not all-core throughput. Inspect emitted code and extend the
  chain sweep before calling any value a ceiling.
- Inverse reciprocal chains alternate between a nonzero seed and its reciprocal.
  They do not represent arbitrary-input inversion distributions. No extra field
  operation is hidden in the reported inversion count.
- Copy uses `copy_from_slice`; stream read sums u64 words. Logical bytes count copy
  reads plus writes, or stream reads. Decimal GB/s equals bytes/ns. Cache residency,
  write allocation and prefetch can make physical traffic different. These are
  single-thread warm, repeated working-set samples, not DRAM bandwidth peaks.
- Rayon probe times repeated joins with two trivial black-boxed closures inside a
  prebuilt dedicated pool. Pool construction and `install` entry are excluded;
  loop/checksum/closure costs remain. It is not an OS thread creation measurement.

Each kernel gets one discarded warmup; all measured samples remain in output.
Zero-duration samples fail with an instruction to increase rounds or passes, rather
than substituting a duration or claiming a rate below timer resolution.
Seeds, operands and results cross black-box boundaries. Checksums are computed
outside field/copy timing and are noncryptographic. Stream and join accumulation
are part of those timed kernels. Differential gates compare raw operations with
arkworks and XYZZ additions with ark group arithmetic before any timing. Memory
copies, stream sums and join results are checked every pass group.

## optional aggregate throughput

`PROBE_ALL_CORE=1 PROBE_THREADS=8` uses the existing dedicated pool, not a new
pool or an extrapolation from one thread. It broadcasts one kernel to every pool
worker for Fr/RawFq/RawFq2 multiply and G1/G2 XYZZ mixed-add. Each worker interleaves
1/4/8 independently seeded chains. Seed construction/reservation is outside timing;
the worker seed is `SEED wrapping_add worker_index`. Differential gates are retained.

`parallel_primitive` records use `aggregate_interleaved_throughput` even with one
chain per worker: that is an aggregate rate, not dependent-chain latency. Actual
kernel worker indices are checked against the executing Rayon thread indices after
timing. Operations are exactly `rounds * chains_per_worker * actual_kernel_threads`.
Counts and seed/result storage bounds are validated before pool/seed allocation.
Aggregate rates divide this count by a common caller wall interval, including pool
entry, broadcast dispatch, skew, loop/operand lookup, result collection and completion.
No simultaneous-start barrier or subtraction is hidden. Checksums/worker checks
are outside timing, and pool construction is excluded.

`parallel_dispatch` separately reports a broadcast of a trivial black-boxed closure
to every worker, including return-value collection and worker-index bookkeeping.
It is a distinct overhead sample, not subtracted from any kernel measurement.
These measurements are not certified saturation, hardware instruction ceilings or
CPU-scaling predictions. A pool of eight workers need not equal eight physical cores.
Capture core class/affinity and competing work before interpreting rates.

Parallel memory sweeps, u64 wide-multiply/carry probes and real batch-affine round
counts remain deferred. Existing memory samples are still single-threaded. No floor
may substitute these single-thread memory numbers for measured aggregate bandwidth.

## computation trace and remaining work

`cpu_trace` loads one key/witness and emits counts without proving or timing.
It checks dimensions, counts gather zero/one shortcuts, reports actual host sizes,
logical C/H/scrub bytes, variable-work MSM prescan categories and window sizes.
The ordered trace records concurrent gather, three joined NTT pipelines, H, five
joined MSMs, then assembly/checked verification. It does not dump witness values,
but witness-dependent categories/checksums are sensitive diagnostic output.

Radix-2-equivalent butterfly counts are algorithmic counts, not actual Fr operations
of compact/fused kernels. H is deliberately not computed: its scalar distribution
is unknown, and the dense window candidate is not a measured dispatch. Query
storage is a footprint, not byte traffic. Unknown operation/traffic counts are null.
The old trace's full dense-H mixed-add estimate is not an exact bucket-add count.

Before deriving floors: count compact-kernel operations, actual signed-digit
occupancy, batch-affine inversion rounds/conflicts, point-chunk splitting and key
preparation. Add representative batch-affine rounds and aggregate stream probes
if those segments have measured headroom. Run the optional aggregate field probes
on the measurement machine rather than multiplying single-thread rates by core count.
Combine counts with same-machine, uncontended samples and verified stage wall times. Account for overlap rather
than adding concurrent stage times. Broaden circuits after the initial pair.
