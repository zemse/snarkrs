# gpu-snark

A Groth16 prover for BN254 that runs the same pipeline on CPU, Apple Metal, or NVIDIA CUDA,
and is measured against rapidsnark and snarkjs on identical artifacts.

- **Drop-in with circom and snarkjs.** Reads `.zkey` and `.wtns` as they are written today.
  The proofs it emits are byte-compatible with `snarkjs groth16 verify`.
- **Three backends, one pipeline, behind feature flags.** CPU is always there; `--features
  metal` and `--features cuda` add the others. The backend boundary is drawn at stage
  groups, not at individual primitives, so a GPU keeps its data resident across all six
  transforms instead of round-tripping to the host between them.
- **No GPU toolchain to install.** Metal compiles MSL at runtime and CUDA compiles through
  NVRTC, and `g16-cuda` uses `cudarc` with dynamic loading, so `--features cuda` builds on a
  machine with no NVIDIA driver and no CUDA toolkit at all. Only running it needs a card.
- **Self-verifying.** `prove` verifies before it writes, and fails rather than emit a proof
  that does not verify. A dropped kernel or a stale pooled buffer dies there instead of
  becoming somebody else's problem. It costs under 3% of a proof.
- **Cold and warm are both reported, always.** Cold is one fresh process per proof, so the
  key parse and the GPU upload are inside the timed region. Warm pays setup once. Quoting
  only one of them is how most prover charts mislead.
- **Every number is traceable.** One results file per machine, named after the machine and
  the commit that produced it, and every timing belongs to a proof that was verified first.

## Benchmarks

Median milliseconds, lower is better. **Warm**: setup paid once, then proving in a loop,
which is what a resident service sees.

| program | constraints | machine | gpu-snark cpu | gpu-snark metal | gpu-snark cuda | rapidsnark | snarkjs |
|---|---:|---|---:|---:|---:|---:|---:|
| `railgun-01x01` | 20,135 | apple-m2-max | 65.4 | 29.1 |  | 72.0 |  |
| `tornado` | 28,275 | apple-m2-max | 102.9 | 40.2 |  | 117.5 |  |
| `sha256` | 59,281 | apple-m2-max | 57.3 | 21.0 |  | 72.0 |  |
| `railgun-13x01` | 141,276 | apple-m2-max | 390.3 | 125.2 |  | 421.0 |  |
| `rsa2048` | 190,945 | apple-m2-max | 225.4 | 56.8 |  | 344.5 |  |
| `keccak256` | 239,176 | apple-m2-max | 216.1 | 38.8 |  | 289.0 |  |
| `anon-aadhaar` | 1,115,080 | apple-m2-max | 1764.5 | 263.2 |  | 2120.0 |  |

**Cold**, one fresh process per proof, key parse and GPU upload inside the timed region,
which is what a CLI does:

| program | constraints | machine | gpu-snark cpu | gpu-snark metal | gpu-snark cuda | rapidsnark | snarkjs |
|---|---:|---|---:|---:|---:|---:|---:|
| `railgun-01x01` | 20,135 | apple-m2-max | 65.1 | 46.3 |  | 77.0 | 879.7 |
| `tornado` | 28,275 | apple-m2-max | 107.3 | 59.0 |  | 123.1 | 1260.3 |
| `sha256` | 59,281 | apple-m2-max | 64.9 | 46.5 |  | 80.4 | 1141.2 |
| `railgun-13x01` | 141,276 | apple-m2-max | 401.7 | 168.9 |  | 436.8 | 4340.2 |
| `rsa2048` | 190,945 | apple-m2-max | 244.9 | 110.8 |  | 360.4 | 3928.5 |
| `keccak256` | 239,176 | apple-m2-max | 231.2 | 87.3 |  | 308.3 | 3380.8 |
| `anon-aadhaar` | 1,115,080 | apple-m2-max | 1894.5 | 523.1 |  | 2195.3 | 22392.5 |

A blank cell is a comparison that machine could not make: the backend does not exist there,
or the prover has no such mode. It is not a zero. snarkjs has no warm mode to measure
because its CLI is the only interface it offers.

The circuits are compiled from their upstream sources rather than reimplemented, so the
constraint counts are the ones those projects actually ship:

| program | what it proves | source |
|---|---|---|
| `railgun-01x01` | Railgun joinsplit, 1 input 1 output, depth 16 | [Railgun-Privacy/circuits-v2](https://github.com/Railgun-Privacy/circuits-v2) |
| `tornado` | Tornado Cash withdraw, 20-level Merkle path | [tornadocash/tornado-core](https://github.com/tornadocash/tornado-core) |
| `sha256` | SHA-256 of a 512-bit message | [circomlib](https://github.com/iden3/circomlib) |
| `railgun-13x01` | Railgun joinsplit, 13 inputs 1 output | [Railgun-Privacy/circuits-v2](https://github.com/Railgun-Privacy/circuits-v2) |
| `rsa2048` | RSA-2048 PKCS#1 v1.5 signature, exponent 65537 | [zkemail/zk-email-verify](https://github.com/zkemail/zk-email-verify) |
| `keccak256` | Ethereum keccak256 over one 135-byte block | [vocdoni/keccak256-circom](https://github.com/vocdoni/keccak256-circom) |
| `anon-aadhaar` | Aadhaar QR signature, RSA-2048 plus SHA-256, with selective disclosure | [anon-aadhaar/anon-aadhaar](https://github.com/anon-aadhaar/anon-aadhaar) |

Two things worth reading before quoting any of this.

**Constraint count is a poor predictor of proving time.** `keccak256` has 25% more
constraints than `rsa2048` and proves faster on every backend here. The MSM dominates, and
its cost tracks how many witness scalars are neither 0 nor 1, not how many constraints there
are. A bit-decomposition-heavy circuit carries a witness full of scalars that never reach a
bucket.

**The GPU margin is a function of circuit size.** At 20k constraints cold, Metal is about
1.4x the CPU backend: the key upload sits inside the timed region and there is not enough
arithmetic to pay for it. At 1.1M warm it is 6.7x the CPU backend and 8.1x rapidsnark. A
single headline multiplier taken from either end would misrepresent the other, which is why
there is no headline multiplier here.

Reproduce any row:

```sh
bench/scripts/run-benchmark.sh --reps 10
```

It works out which backends the box can run, builds only those, downloads whatever proving
keys it is missing, and benchmarks against rapidsnark and snarkjs on the same artifacts. It
refuses to record anything while the machine is loaded, because a timing taken on a busy box
measures the other tenant.

## Using it from the command line

```sh
cargo build --release                      # CPU only
cargo build --release --features metal     # + Apple Metal
cargo build --release --features cuda      # + NVIDIA CUDA
```

```sh
g16 prove  --zkey circuit.zkey --witness circuit.wtns \
           --proof proof.json --public public.json \
           [--backend cpu|metal|cuda] [--stage-timings] [--self-verify true|false]

g16 verify --vkey verification_key.json --proof proof.json --public public.json

g16 bench  --artifacts <DIR> [--variant NAME]... [--reps N] \
           [--backend cpu|metal|cuda] [--mode cold|warm|both] [--csv FILE]
```

The output is what snarkjs expects, so the two are interchangeable in either direction:

```sh
g16 prove --zkey circuit.zkey --witness circuit.wtns \
          --proof proof.json --public public.json --backend metal
snarkjs groth16 verify verification_key.json public.json proof.json    # OK!
```

`--stage-timings` prints where the time went, which is the fastest way to find out whether
a circuit is MSM-bound or transform-bound:

```
gather      us       5244
ntt         us       9802
pointwise   us        475
msm         us      54626
assemble    us        602
```

## Using it as a library

```rust
use g16_core::{prove::prove, verify::verify, Backend, StageTimings};
use g16_zkey::{wtns::Witness, ProvingKey};

// Parse once. On a GPU backend `prepare` is also where the key is uploaded, so hold the
// prepared circuit and prove against it repeatedly: that is exactly the difference between
// the warm and the cold columns above.
let pk = ProvingKey::load(std::path::Path::new("circuit.zkey"))?;
let n_public = pk.n_public;
let circuit = g16_metal::MetalBackend::new()?.prepare(pk)?;

let w = Witness::load(std::path::Path::new("circuit.wtns"))?.0;
let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut ark_std::rand::thread_rng(), &mut t)?;

// The public signals are the witness prefix, which is what snarkjs publishes.
let public = &w[1..=n_public];
verify(&circuit.key().vk, public, &proof)?;
```

Swap `MetalBackend` for `g16_core::cpu::CpuBackend` or `g16_cuda::CudaBackend` to change
where it runs; nothing else in the snippet changes, which is the point of the trait.

The blinders come from the OS CSPRNG. There is no seed override on this path on purpose: a
reused `(r, s)` across two proofs of different witnesses leaks the witness, so the
deterministic entry point stays test-only.

## How it is put together

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

The backend boundary is at **stage groups**, not individual primitives. A trait with
`ntt()` and `msm()` methods would read more cleanly, but it forces a host round trip between
every step, so a GPU backend built against it measures the bus instead of the arithmetic.

Everything above the field is ours: zkey parsing, the CSR gather, NTT, Pippenger, the
prover, the verifier, and the kernels. Field and curve arithmetic is arkworks, because
beating `ark-ff`'s Montgomery multiplication without hand-written assembly is not where the
interesting work is.

| crate | what it owns |
|---|---|
| `g16-field` | BN254 type aliases and the radix-2 domain |
| `g16-zkey` | `.zkey` and `.wtns` parsing, section 4 sorted into CSR |
| `g16-ntt` | NTT/iNTT and the coset shift, CPU |
| `g16-msm` | Pippenger MSM over G1 and G2, CPU |
| `g16-core` | the `Backend`/`PreparedCircuit` contract, CPU backend, prove, verify |
| `g16-gpu-layout` | the packed field and scalar layouts both GPU backends share |
| `g16-metal` | MSL kernels and host code, `--features metal` |
| `g16-cuda` | CUDA kernels and host code, `--features cuda` |
| `g16-cli` | the `g16` binary: prove, verify, bench |

### One thing the CSR gather buys

snarkjs stores section 4 as an unordered `(matrix, constraint, signal, coef)` list, and
rapidsnark consumes it as a scatter under 1,024 striped mutexes. A GPU cannot copy that:
there is no 32-byte atomic, and a CAS loop over four 64-bit limbs on a contended row is a
livelock hazard. Sorting the list by `(matrix, constraint)` once at key load turns the
per-proof stage into a race-free gather that both backends share. It is a one-time cost at
key load that makes the GPU backend possible at all.

## Honest limits

- **The benchmark keys are not production keys.** Phase 1 is real: the prepared phase-2
  files from contribution 0080 of the [Perpetual Powers of
  Tau](https://github.com/privacy-ethereum/perpetualpowersoftau) ceremony. Phase 2 is a
  single local contribution made by this repo's own scripts, so the keys are sound only if
  you trust that one contributor, which is to say they are sound for measuring proving time
  and for nothing else. This does not affect timing or correctness, only soundness.
- **A large speedup multiplier over a CPU baseline mostly measures the baseline.** rapidsnark
  is significantly slower on some hosts than others for identical work, so a ratio quoted
  without naming the machine is close to meaningless. That is why the table above carries
  the machine in every row rather than collapsing to a single "Nx faster" headline.
- **This prover is not constant time with respect to the witness.** The MSM skips zero and
  one scalars, which makes running time and allocation size a function of witness sparsity.
  That is the optimisation exploited against Zcash in USENIX Security 2020. It is kept
  because every production Groth16 prover does it, but note the payoff is modest and easy to
  overstate: measured on this benchmark ladder, 0/1 scalars are 1.8% to 4.1% of the MSM
  workload. It grows with sparsity on genuinely bit-heavy witnesses. Zero knowledge holds
  for the proof and not for the process that produced it.
- **A `.zkey` is trusted input.** Only the O(1) points are validated on load; the query
  sections are millions of points and a subgroup check each would dominate key load. If the
  key and the witness have different owners, which is exactly proving-as-a-service, run
  `snarkjs zkey verify <r1cs> <ptau> <zkey>` first. That is the only non-circular check:
  deriving a vkey from the zkey and verifying against it proves nothing about the zkey.
- **No browser backend yet.** A WebGPU path is in progress and is not part of any number here.

## Security

An audit pass ran four threat-research notes, four code audits, and the adversarial suites
they produced. Roughly 3,200 proofs over 200 distinct statements found no defect in the
prover itself, and CPU and Metal are bit-identical at pinned blinders.

The findings were in key handling, and the two serious ones are fixed: a malicious `.zkey`
could silently switch zero knowledge off while proofs still verified against the genuine
verification key, and a 4 KB file could trigger a 34 GB allocation. Both are pinned by
regression tests.

## References

Things actually leaned on while building this, not a reading list.

- Remco Bloemen, [**The Groth16 prover, step by
  step**](https://xn--2-umb.com/22/groth16/). The stage numbering used throughout this repo
  comes from here.
- [iden3/snarkjs](https://github.com/iden3/snarkjs) and
  [iden3/rapidsnark](https://github.com/iden3/rapidsnark), the reference implementations
  this one is checked against, and
  [iden3/binfileutils](https://github.com/iden3/binfileutils) for the `.zkey` container
  layout.
- [arkworks-rs/circom-compat](https://github.com/arkworks-rs/circom-compat), for how the
  circom witness and key formats map onto arkworks types.
- Explicit-Formulas Database, [shortened weierstrass
  XYZZ](https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html) and
  [jacobian](https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html). The MSM's point
  representation is from here.
- Gabizon, Williamson, Ciobotaru, [**PLONK**](https://eprint.iacr.org/2019/953) and the
  Frozen Heart disclosures from Trail of Bits,
  [part 1](https://blog.trailofbits.com/2022/04/13/part-1-coordinated-disclosure-of-vulnerabilities-affecting-girault-bulletproofs-and-plonk/),
  for how a soundness bug hides in an implementation rather than a paper.
- [0xPARC/zk-bug-tracker](https://github.com/0xPARC/zk-bug-tracker), and the
  [Groth16 proof malleability](https://ethresear.ch/t/transaction-malleability-attack-of-groth16-proof/15881)
  thread, both of which fed the adversarial test suite.
- ["Zero-knowledge proofs of non-knowledge"](https://github.com/cryptosubtlety/00/blob/main/00.pdf),
  on why a prover that emits a valid proof is not the same as a prover that is correct.
- [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau), the
  source of the phase 1 parameters.
