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
cargo build --release --features cuda      # + NVIDIA CUDA
```

Neither GPU backend needs an offline toolchain. Metal compiles MSL at runtime through
`MTLDevice::newLibraryWithSource`; CUDA compiles its kernels at runtime through NVRTC. The
CUDA crate uses `cudarc` with `dynamic-loading`, so `--features cuda` **builds on a machine
with no NVIDIA driver and no CUDA toolkit at all**, including this MacBook. Only running it
needs a card. That is what made it possible to develop the backend here and test it on a
rented T4.

One caveat specific to CUDA, in full: the first run on a fresh machine spends 270 to 300
seconds compiling kernels before it proves anything, in two stages that are both cached in
`~/.nv/ComputeCache` afterwards. See `bench/results/device-microbench.md`.

## Benchmarks

```sh
bench/scripts/gen-artifacts.sh      # circuits, keys, witnesses (needs circom + snarkjs)
bench/scripts/build-rapidsnark.sh   # the CPU baseline, plus a warm-mode wrapper
bench/scripts/run-comparison.py     # cold and warm, every prover, same artifacts
bench/aws/provision.sh              # a Tesla T4 box, with a dead-man switch
bench/aws/run-gpu-bench.sh          # sync, build, warm the kernel cache, benchmark, pull
bench/aws/terminate.sh              # and prove it is gone
```

Results and the full write-up are in [`bench/results/`](bench/results/). Two machines, one
file each, never merged: an M2 Max MacBook Pro and an EC2 g4dn.2xlarge with a Tesla T4.

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
- **A large speedup multiplier over a CPU baseline mostly measures the baseline.** Warm at
  140K constraints the CUDA backend is 10.34x rapidsnark on the EC2 box and Metal is 4.31x
  on the Mac, but in absolute milliseconds the Mac's GPU is the faster of the two (128.8
  against 152.7). rapidsnark is 2.8x slower on the Xeon than on the M2 Max for identical
  work. Read `bench/results/README.md` before quoting any ratio from here.
- **The CUDA MSM is not tuned.** It is 92% of the proof at every size, and a forced window
  width beats the automatic choice by 9% at 2^18. The window cost model still carries
  constants measured on Apple silicon.
- **This prover is not constant time with respect to the witness.** The MSM skips zero and
  one scalars, which makes running time and allocation size a function of witness sparsity.
  That is the optimisation exploited against Zcash in USENIX Security 2020. It is kept
  because it is worth about 5.1x and every production Groth16 prover does it, but it means
  zero knowledge holds for the proof and not for the process that produced it.
- **A `.zkey` is trusted input.** Only the O(1) points are validated on load; the query
  sections are millions of points and a subgroup check each would dominate key load. If the
  key and the witness have different owners, which is exactly proving-as-a-service, run
  `snarkjs zkey verify <r1cs> <ptau> <zkey>` first. That is the only non-circular check:
  deriving a vkey from the zkey and verifying against it proves nothing about the zkey.

## Security

The audit ran four threat-research notes, four code audits, and
the adversarial suites they produced. Roughly 3,200 proofs over 200 distinct statements found
no defect in the prover itself, and CPU and Metal are bit-identical at pinned blinders.

The findings were in key handling, and the two serious ones are fixed: a malicious `.zkey`
could silently switch zero knowledge off while proofs still verified against the genuine
verification key, and a 4 KB file could trigger a 34 GB allocation. Both are pinned by
regression tests.
