# gpu-snark

A Groth16 prover for BN254, built so the same pipeline runs on CPU, Apple Metal, or CUDA
behind feature flags, and so the three can be measured against each other and against
rapidsnark on identical artifacts.

Circuits and keys are circom/snarkjs `.zkey` and `.wtns`, and the proofs it emits are
byte-compatible with `snarkjs groth16 verify`.

## Why the crates are split the way they are

Stage numbering follows [Bloemen's Groth16 write-up](https://xn--2-umb.com/22/groth16/),
expanded into implementable steps:

```
 -1  witness generation       CPU only, forever. Data-dependent control flow.
  0  coefficient gather       A,B,C evals = w . (A,B,C)     |
  1  3 x iNTT                 into coefficient form         | backend::compute_h
  2  3 x coset shift          x[i] *= g^i                   |   stages 0-4
  3  3 x NTT on the coset     evaluations on y              |
  4  H = A*B - C              elementwise                   |
  5  MSM A -> G1  (zkey s5)   scalars = full witness        |
  6  MSM B -> G2  (zkey s7)   scalars = full witness        | backend::msms
  7  MSM B -> G1  (zkey s6)   scalars = full witness        |   stages 5-9
  8  MSM L -> G1  (zkey s8)   scalars = private witness     |
  9  MSM H -> G1  (zkey s9)   scalars = H evaluations       |
 10  sample r, s              CPU only. CSPRNG, trust boundary.
 11  blind and assemble       CPU. ~10 group ops, O(1).
```

The backend boundary is drawn at **stage groups**, not individual primitives. A trait with
`ntt()` and `msm()` methods would read more cleanly, but it forces a host round trip
between every step, so a GPU backend built against it measures the bus instead of the
arithmetic. As written, a backend keeps all three domain vectors resident across all six
transforms and runs the five MSMs on its own queues.

Everything above the field is ours: zkey parsing, the CSR gather, NTT, Pippenger, the
prover, the verifier, and the kernels. Field and curve arithmetic is arkworks, because
beating `ark-ff`'s Montgomery multiplication without hand-written assembly is not a
realistic use of effort and is not where the interesting work is.

| crate | what it owns |
|---|---|
| `g16-field` | BN254 type aliases, the radix-2 domain, coset generator |
| `g16-zkey` | `.zkey` and `.wtns` parsing, section 4 sorted into CSR |
| `g16-ntt` | NTT/iNTT and the coset shift, CPU |
| `g16-msm` | Pippenger MSM over G1 and G2, CPU |
| `g16-core` | the `Backend`/`PreparedCircuit` contract, CPU backend, prove, verify |
| `g16-metal` | MSL kernels and host code, `--features metal` |
| `g16-cuda` | stub, `--features cuda` |
| `g16-cli` | the `g16` binary: prove, verify, bench |

### One thing the CSR gather buys

snarkjs stores section 4 as an unordered `(matrix, constraint, signal, coef)` list, and
rapidsnark consumes it as a scatter under 1,024 striped mutexes. A GPU cannot copy that:
there is no 32-byte atomic, and a CAS loop over four 64-bit limbs on a contended row is a
livelock hazard. Sorting the list by `(matrix, constraint)` once at key load turns the
per-proof stage into a race-free gather that both backends share. It is a one-time cost
that makes the GPU backend possible at all.

## Building

```sh
cargo build --release                      # CPU only
cargo build --release --features metal     # + Apple Metal
```

Metal needs no offline toolchain: MSL is compiled at runtime through
`MTLDevice::newLibraryWithSource`, which uses the compiler shipped with the OS.

## Benchmarks

```sh
bench/scripts/gen-artifacts.sh      # circuits, keys, witnesses (needs circom + snarkjs)
bench/scripts/build-rapidsnark.sh   # the CPU baseline, plus a warm-mode wrapper
bench/scripts/run-comparison.py     # cold and warm, every prover, same artifacts
```

**Cold and warm each mean exactly one thing here.** Cold is one fresh process per proof,
so the zkey parse and, on GPU, the upload sit inside the timed region; that is what every
CLI prover actually does. Warm pays setup once and then proves in a loop; that is what a
resident service does, and it is the only mode in which a GPU backend can look good, which
is exactly why vendor charts prefer it. Reporting one without the other is how those
charts mislead.

rapidsnark's stock CLI reparses the zkey on every invocation and so can only ever report
cold. `bench/wrappers/rapidsnark-warm` drives the same library through the object API its
own `proverServer` uses, where the parse happens once in `groth16_prover_create`. The
proving code is untouched; only the boundary of the timed region moves.

**Every recorded timing belongs to a proof that was verified first**, through three
oracles that answer different questions: our own verifier (is the pairing check right),
`rapidsnark-verify` (does an independent C++ implementation agree), and `snarkjs groth16
verify` (is the JSON encoding ecosystem-compatible).

## Honest limits

- The proving keys under `bench/artifacts/` come from a **locally generated** powers of
  tau, not a real ceremony. They are benchmark keys and must never be used for anything
  else. This does not affect timing or correctness measurements, only soundness.
- The joinsplit circuit here compiles under circom 2.1.4 and reports 3,359 constraints for
  `js_1x1_d8`, against 7,211 for the same nominal parameters in earlier runs on a
  different circom and circomlib. Numbers here are therefore **not** directly comparable
  to that earlier data; what matters is that every prover in a given run proves the same
  artifact.
- No CUDA backend exists and none can be tested on this machine.
