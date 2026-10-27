# The ethproofs client-side-proving benchmark

[ethproofs.org/csp-benchmarks](https://ethproofs.org/csp-benchmarks) publishes six numbers
per circuit for seventeen proving systems, measured quarterly on an AWS `mac2.metal`
(Apple M1, 8 cores, 16 GB). One of those seventeen is `circom`, which is circom +
witnesscalc + rapidsnark. This crate keeps circom and witnesscalc and swaps rapidsnark
for this prover. The measurement and witnesscalc changes below also matter when comparing
rows; this is not an otherwise identical integration.

Upstream: <https://github.com/ethereum/csp-benchmarks>.

## Running it

```sh
bench/scripts/csp-fetch.sh      # once: 3.6 GB of published zkeys, checksum verified
bench/scripts/csp-bench.sh      # every backend this build has, then the comparison
bench/scripts/csp-bench.sh --backend cpu --reps 5 --mem-reps 3     # a quicker pass
SNARKJS_VERIFY=1 bench/scripts/csp-bench.sh --target sha256 --input-size 128
G16_CSP_ALLOW_UNBLINDED=1 bench/scripts/csp-bench.sh --target poseidon
```

`--target` and `--input-size` restrict the run to matching variants. `SNARKJS_VERIFY=1`
adds the independent check described under submission readiness.
`G16_CSP_ALLOW_UNBLINDED=1` is the poseidon key exception described there; without it
the poseidon keys are refused.

`csp-fetch.sh` also clones the upstream circuit sources into `bench/vendor`, which is where
`build.rs` reads the witness generators from. Nothing under `bench/vendor` or
`bench/artifacts` is tracked.

To write `input.json`, `circuit.wtns`, `vkey.json` and `public.json` next to each zkey, so
that `snarkrs bench` and the backend audit tests can use the same circuits:

```sh
bench/csp/target/release/snarkrs-csp artifacts
```

## What the protocol actually measures

Three things are easy to get wrong and each moves the number by more than the prover does.

**Witness generation is inside the timed region.** Upstream's `prove` spawns the
witnesscalc thread itself and the criterion closure joins it. Timing only the prover
reports a number 10-20% below what the page compares against.

**So is the zkey read.** `groth16_prover_zkey_file_wrapper` takes a path, so every timed
iteration re-reads the whole key: 1.1 GB per iteration for `keccak_2048`. This is the cold
mode of `snarkrs bench`, not the warm one, and a warm number is not comparable to the page.

**Verification re-reads the zkey too**, and derives the verifying key from it rather than
loading a `verification_key.json`. That is why the published verify times run to hundreds
of milliseconds for a pairing check that costs about one. Our row reports the same thing so
the column means the same thing; the honest verifier cost is in `verify_vkey_ms` in the
breakdown file.

## Measurement differences to disclose

**The sampling protocol.** Upstream sets criterion's `sample_size(10)` and publishes its
mean point estimate. That is ten samples, not necessarily ten proofs or exactly 55:
[Criterion's default sampling mode](https://docs.rs/criterion/latest/criterion/enum.SamplingMode.html)
chooses linear or flat sampling based on the warm-up iteration time. We instead use one
warm-up then the arithmetic mean of `--reps` timed iterations, without Criterion's
warm-up or iteration selection. Median, min and max go in the breakdown. Record `--reps`
and `--mem-reps` with any comparison; the estimator and sampling protocol are not identical.

**The witness hand-off.** Upstream converts witnesscalc's `.wtns` buffer to `Vec<BigUint>`
and back again, because that is the shape `circom-prover`'s API takes. We hand the buffer
straight to `Witness::from_bytes`. The conversion is overhead rapidsnark's binding forced
on them and a real integration would not pay it, so it is not reproduced; it is worth a few
milliseconds at the sizes here.

**The witness arithmetic.** `build.rs` patches witnesscalc's field arithmetic by default,
including `Fr_mod`, `Fr_idiv` and `Fr_pow`. These are local changes, not an upstream
witnesscalc baseline. `G16_CSP_STOCK_FR` skips patch application; use a clean witnesscalc
build checkout for a stock baseline, since it does not undo an earlier patch there.
Record the actual arithmetic mode of the binary that produced the row.

**Validation is included.** The native prover uses checked key loading and checked
`prove`, so point/key validation is inside `zkey_load_ms` and proof self-verification is
inside `prove_ms`. Under the poseidon key exception the same work runs less one O(1)
comparison. The memory binary also uses checked `prove`. The driver separately
verifies the warm-up proof before reporting timings; that is not an independent oracle.
`SNARKJS_VERIFY=1` adds one, outside the timed region.

**The macOS allocator.** `snarkrs-csp-mem` attempts to re-exec with `MallocLargeCache=0`
before generating the witness. An existing value of `MallocLargeCache` is respected, and
an exec failure leaves the original process running. The timing driver does not set it.
Record the effective setting for each run, not just the default intent: a library user
embedding the prover does not get this memory-process policy automatically.
`peak_memory` is the mean of whole-process maximum RSS samples from `/usr/bin/time`, not
Metal buffer bytes or phase-local footprint. `--mem-reps 0` leaves a missing-measurement
marker of zero, not a measured zero-byte peak.

## Submission readiness

The harness is for local comparisons, not a submission-ready row. In particular, the
all-target commands above stop at `poseidon_2` unless `G16_CSP_ALLOW_UNBLINDED=1` is set,
because the published poseidon keys are unblinded. Historical end-to-end results do not
establish current readiness.

* **The poseidon keys need an explicit exception.** All five published poseidon keys have
  `gamma_g2 == delta_g2`: no phase-2 contribution, so anyone can forge a proof against
  them, and `ProvingKey::load` and the checked `prove` refuse them. Default runs keep that
  refusal. `--allow-unblinded-zkey` on `bench`, `artifacts`, `list` and
  `snarkrs-csp-mem` (`G16_CSP_ALLOW_UNBLINDED=1` in `csp-bench.sh`) loads with
  `ProvingKey::load_allowing_unblinded` and proves and verifies with the matching
  `_allowing_unblinded` functions, which drop that one comparison and run every other
  check: the container, header, infinity, curve and IC checks at load, and the proof
  element checks and pairing in the self-verify. It applies only to a key that has that
  shape; any other key in the same run loads the default way. A row measured under it
  has `unblinded` in `feat` (`cpu-unblinded`, so in its file name too),
  `unblinded_zkey: true` in the breakdown, and the run prints a warning per variant. The
  exception exists to time these benchmark keys; such a key is not fit for anything else.
* **Backend selection is explicit.** `--backend cpu` and `--backend metal` are separate
  runs, with `feat: cpu` or `feat: metal` and `backend: g16/cpu` or `backend: g16/metal`
  in the breakdown, and neither ever falls back to the other: `make_backend` refuses an
  unavailable backend. `--backend auto` is the mixed row policy. Metal loses to the CPU
  on every poseidon size, because its fixed prepare and prove cost is larger than the
  whole CPU proof there, so `auto` runs the CPU below `AUTO_METAL_FROM` constraints
  (65,536) and Metal at or above. The threshold comes from the M2 Max runs in
  `bench/results/csp`: the CPU won at 53,048 constraints and below, Metal at 89,224 and
  above, and nothing in between was measured, so 2^16 is a conservative pick inside
  that gap rather than a measured crossover; `lib.rs` has the numbers. Each cell names
  the backend it ran: `feat: auto-cpu` or `feat: auto-metal`, the same in the
  breakdown's `backend`, plus `auto_metal_from`. An `auto` cell that needs Metal in a
  build without it fails rather than running the CPU. `csp_report.py` prints an `auto`
  run as one table with a backend column. Any submission of a mixed row still needs the
  upstream schema question below settled.
* **GPU metadata needs agreement.** The checked-out upstream schema's `acceleration`
  enum has `precompile` and `inline`, not GPU backends. Our rows omit that field and use
  `feat` for the backend. Do not put `metal` into `acceleration` or reuse a zkVM value to
  imply GPU support. Agree GPU reporting with upstream before submitting; no schema
  acceptance is established here.
* **Independent verification is opt-in.** With `SNARKJS_VERIFY=1`, `csp-bench.sh` fails
  up front if `snarkjs` is not on PATH, runs `snarkrs-csp artifacts` for the selected
  variants, then passes `--snarkjs` to `bench`. Each variant's warm-up proof is written
  as snarkjs JSON and checked with `snarkjs groth16 verify` against that variant's
  `public.json` and a vkey that `snarkjs zkey export verificationkey` reads from the zkey
  itself, after our own verifier and before any timed iteration. A missing snarkjs, a
  nonzero exit from either command, an export that leaves no groth16 vkey, or a verify
  without `OK!` fails the run, names the variant and leaves the proof and exported vkey
  on disk. The check is independent of our verifier, proof encoder and key decoding; the
  `vkey.json` that `artifacts` writes with our ceremony code is not used. `public.json`
  still comes from our witness decoding. It is off by default; a row without it has only
  the internal checks.
* **Hardware and protocol need disclosure.** Keep the machine, revision, build features,
  key release/checksums, witness arithmetic mode, sampling counts, validation policy and
  effective allocator settings beside any submitted rows. Laptop results are not
  `mac2.metal` results. AWS allocation, upstream schema changes and row submission need
  separate approval; this harness does not establish those decisions.

## Circuits

The zkeys are the published ones, downloaded from the upstream `zkeys-v2` release and
checked against the manifest it ships. They are not regenerated locally: a local setup
would give a different delta and a different key size, and `preprocessing_size` is one of
the six numbers reported.

| target   | sizes                     | constraints        |
| -------- | ------------------------- | ------------------ |
| sha256   | 128 - 2048 bytes          | 53,048 - 595,688   |
| keccak   | 128 - 2048 bytes          | 93,184 - 1,506,240 |
| poseidon | 2 - 16 field elements     | 517 - 2,092        |
| ecdsa    | one secp256k1 signature   | 512,955            |

`ecdsa`'s witness generator is the one upstream does not ship: 57 MB of generated C++,
because the width-12 comb table is inlined in the circuit. `csp-fetch.sh` compiles it with
`--O2 --c`, the flags upstream's own build script uses, so that first fetch needs circom
2.2.3 on PATH. `blake3` and `poseidon2`, the other two upstream targets, have no circom
circuit at all.

## Comparing against the published row

`bench/csp/published-circom.json` is the `circom` row as the page serves it, and
`bench/scripts/csp_report.py` puts ours beside it. Two caveats travel with every such
comparison and the report prints both:

* Those numbers were measured on `mac2.metal`. A ratio against a laptop is a ratio against
  a laptop. `bench/aws` exists to close that gap.
* They were measured against **zkeys-v1**, and the circuits have since shrunk: upstream's
  `keccak_128` was 163,638 constraints, the one in this directory is 93,184. Comparing by
  input size overstates any speedup by about 1.75x on keccak. The report carries both
  constraint counts and a per-million-constraint rate for that reason.

`ecdsa` has no counterpart in that row at all: the page's `circom` entry does not cover
it. The report prints dashes on that line, and the per-million-constraint rate beside the
other targets is the only thing on it that compares to anything.
