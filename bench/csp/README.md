# The ethproofs client-side-proving benchmark

[ethproofs.org/csp-benchmarks](https://ethproofs.org/csp-benchmarks) publishes six numbers
per circuit for seventeen proving systems, measured quarterly on an AWS `mac2.metal`
(Apple M1, 8 cores, 16 GB). One of those seventeen is `circom`, which is circom +
witnesscalc + rapidsnark. This crate keeps the first two and swaps the third for this
prover, so the only difference between our row and theirs is the prover.

Upstream: <https://github.com/ethereum/csp-benchmarks>.

## Running it

```sh
bench/scripts/csp-fetch.sh      # once: 3.6 GB of published zkeys, checksum verified
bench/scripts/csp-bench.sh      # every backend this build has, then the comparison
bench/scripts/csp-bench.sh --backend cpu --reps 5 --mem-reps 3     # a quicker pass
```

`csp-fetch.sh` also clones the upstream circuit sources into `bench/vendor`, which is where
`build.rs` reads the witness generators from. Nothing under `bench/vendor` or
`bench/artifacts` is tracked.

To write `input.json`, `circuit.wtns`, `vkey.json` and `public.json` next to each zkey, so
that `g16 bench` and the backend audit tests can use the same circuits:

```sh
bench/csp/target/release/g16-csp artifacts
```

## What the protocol actually measures

Three things are easy to get wrong and each moves the number by more than the prover does.

**Witness generation is inside the timed region.** Upstream's `prove` spawns the
witnesscalc thread itself and the criterion closure joins it. Timing only the prover
reports a number 10-20% below what the page compares against.

**So is the zkey read.** `groth16_prover_zkey_file_wrapper` takes a path, so every timed
iteration re-reads the whole key: 1.1 GB per iteration for `keccak_2048`. This is the cold
mode of `g16 bench`, not the warm one, and a warm number is not comparable to the page.

**Verification re-reads the zkey too**, and derives the verifying key from it rather than
loading a `verification_key.json`. That is why the published verify times run to hundreds
of milliseconds for a pairing check that costs about one. Our row reports the same thing so
the column means the same thing; the honest verifier cost is in `verify_vkey_ms` in the
breakdown file.

## Two deliberate deviations

**The sampling ramp.** Upstream drives criterion at `sample_size(10)` and publishes the
mean point estimate, reaching it through a ramp of 1, 2, ... 10 iterations per sample plus a
three second warm-up: 55 proofs for one number. At `keccak_2048`'s several seconds a proof
that is most of an hour for one cell of a fifteen cell table. We keep the statistic, one
warm-up then the mean of `--reps` timed iterations, and drop the ramp. Median, min and max
go in the breakdown, because a mean of ten on a laptop is one background process away from
being wrong and the spread is the only way to see it happen.

**The witness hand-off.** Upstream converts witnesscalc's `.wtns` buffer to `Vec<BigUint>`
and back again, because that is the shape `circom-prover`'s API takes. We hand the buffer
straight to `Witness::from_bytes`. The conversion is overhead rapidsnark's binding forced
on them and a real integration would not pay it, so it is not reproduced; it is worth a few
milliseconds at the sizes here.

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

`ecdsa` is the fourth circom target and is not here yet: its witness generator is 57 MB of
generated C++ that upstream compiles from source on demand rather than shipping, so it needs
a circom run before it can be linked. `blake3` and `poseidon2`, the other two upstream
targets, have no circom circuit at all.

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
