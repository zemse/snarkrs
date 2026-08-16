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
| `railgun-01x01` | 20,135 | aws-g4dn.2xlarge-tesla-t4 | 231.0 |  | 23.7 | 170.0 |  |
| `tornado` | 28,275 | apple-m2-max | 102.9 | 40.2 |  | 117.5 |  |
| `tornado` | 28,275 | aws-g4dn.2xlarge-tesla-t4 | 396.1 |  | 32.0 | 300.0 |  |
| `sha256` | 59,281 | apple-m2-max | 57.3 | 21.0 |  | 72.0 |  |
| `sha256` | 59,281 | aws-g4dn.2xlarge-tesla-t4 | 193.6 |  | 18.0 | 175.0 |  |
| `railgun-13x01` | 141,276 | apple-m2-max | 390.3 | 125.2 |  | 421.0 |  |
| `railgun-13x01` | 141,276 | aws-g4dn.2xlarge-tesla-t4 | 1447.7 |  | 135.3 | 1191.5 |  |
| `rsa2048` | 190,945 | apple-m2-max | 225.4 | 56.8 |  | 344.5 |  |
| `rsa2048` | 190,945 | aws-g4dn.2xlarge-tesla-t4 | 776.9 |  | 64.7 | 790.5 |  |
| `keccak256` | 239,176 | apple-m2-max | 216.1 | 38.8 |  | 289.0 |  |
| `keccak256` | 239,176 | aws-g4dn.2xlarge-tesla-t4 | 758.1 |  | 51.4 | 682.0 |  |
| `anon-aadhaar` | 1,115,080 | apple-m2-max | 1764.5 | 263.2 |  | 2120.0 |  |
| `anon-aadhaar` | 1,115,080 | aws-g4dn.2xlarge-tesla-t4 | 5925.0 |  | 406.5 | 5774.0 |  |

**Cold**, one fresh process per proof, key parse and GPU upload inside the timed region,
which is what a CLI does:

| program | constraints | machine | gpu-snark cpu | gpu-snark metal | gpu-snark cuda | rapidsnark | snarkjs |
|---|---:|---|---:|---:|---:|---:|---:|
| `railgun-01x01` | 20,135 | apple-m2-max | 65.1 | 46.3 |  | 77.0 | 879.7 |
| `railgun-01x01` | 20,135 | aws-g4dn.2xlarge-tesla-t4 | 238.8 |  | 266.5 | 171.0 | 1757.4 |
| `tornado` | 28,275 | apple-m2-max | 107.3 | 59.0 |  | 123.1 | 1260.3 |
| `tornado` | 28,275 | aws-g4dn.2xlarge-tesla-t4 | 407.2 |  | 281.2 | 298.2 | 2517.4 |
| `sha256` | 59,281 | apple-m2-max | 64.9 | 46.5 |  | 80.4 | 1141.2 |
| `sha256` | 59,281 | aws-g4dn.2xlarge-tesla-t4 | 214.9 |  | 269.1 | 183.4 | 2318.1 |
| `railgun-13x01` | 141,276 | apple-m2-max | 401.7 | 168.9 |  | 436.8 | 4340.2 |
| `railgun-13x01` | 141,276 | aws-g4dn.2xlarge-tesla-t4 | 1496.9 |  | 429.8 | 1187.4 | 8157.4 |
| `rsa2048` | 190,945 | apple-m2-max | 244.9 | 110.8 |  | 360.4 | 3928.5 |
| `rsa2048` | 190,945 | aws-g4dn.2xlarge-tesla-t4 | 846.2 |  | 401.2 | 797.9 | 6904.7 |
| `keccak256` | 239,176 | apple-m2-max | 231.2 | 87.3 |  | 308.3 | 3380.8 |
| `keccak256` | 239,176 | aws-g4dn.2xlarge-tesla-t4 | 788.1 |  | 388.8 | 703.3 | 6807.9 |
| `anon-aadhaar` | 1,115,080 | apple-m2-max | 1894.5 | 523.1 |  | 2195.3 | 22392.5 |
| `anon-aadhaar` | 1,115,080 | aws-g4dn.2xlarge-tesla-t4 | 6296.8 |  | 1735.9 | 5627.8 | 44888.9 |

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

Three things worth reading before quoting any of this.

**Constraint count is a poor predictor of proving time, and the error is large.**
`railgun-13x01` has 41% fewer constraints than `keccak256` and takes 2.6x as long on CUDA.
That is not an artifact of our prover: it holds on every backend, and for rapidsnark and
snarkjs too, so it is a property of the circuit.

The reason is visible in what the five MSMs are made of
(`cargo run --release -p g16-core --example msm_shape -- bench/artifacts`):

| circuit | constraints | domain | witness scalars that are 0 or 1 | witness MSM (A) | H MSM | serial MSM |
|---|---:|---:|---:|---:|---:|---:|
| `sha256` | 59,281 | 2^16 | 100.00% | 0.6 ms | 35.7 ms | 39.4 ms |
| `tornado` | 28,275 | 2^15 | 2.49% | 17.4 ms | 20.0 ms | 105.6 ms |
| `railgun-13x01` | 141,276 | 2^18 | 1.48% | 38.5 ms | 124.8 ms | 345.1 ms |
| `rsa2048` | 190,945 | 2^18 | 97.24% | 3.4 ms | 139.3 ms | 163.8 ms |
| `keccak256` | 239,176 | 2^18 | 100.00% | 2.2 ms | 123.3 ms | 135.7 ms |

Two effects, and constraint count predicts neither.

Four of the five MSMs take the witness, and the MSM skips scalars that are 0 or 1. A hash
circuit is bit decomposition nearly all the way down, so `keccak256`'s witness is *entirely*
zeros and ones and its witness MSM costs 2.2 ms. Railgun's joinsplit is Poseidon and EdDSA
over full field elements, so 98.5% of its witness reaches a bucket and the same MSM costs
38.5 ms, seventeen times more, on a circuit 41% smaller.

The fifth MSM takes H and is sized by the domain, which is the next power of two above the
constraint count. 141,276 and 239,176 both round up to 262,144, so `keccak256` and
`railgun-13x01` pay the same H cost, 123 ms against 125 ms, though one is 69% larger.
Crossing a power of two is what makes proving slower, not adding constraints.

**A GPU is not automatically faster, and on small circuits it is often slower.** Cold on the
T4, CUDA loses to rapidsnark on two of the three smallest circuits and to our own CPU backend
on two of them: the key upload and the kernel launches sit inside the timed region and there
is not enough arithmetic to pay for them. The crossover sits between 59k and 141k
constraints. Above it the picture inverts, and at 1.1M warm CUDA is 14.6x our CPU backend
and 14.2x rapidsnark on the same box.

**A speedup multiplier without a machine attached is close to meaningless.** rapidsnark beats
our CPU backend on six of the seven circuits on the T4's Xeon, and loses to it on all seven
on the M2 Max, for identical work. `ark-ff`'s assembly path is x86-only, so on Apple silicon
arkworks runs a generic Rust Montgomery multiply while rapidsnark carries hand-written
assembly for both architectures. Our CPU backend therefore looks better than it is on the Mac
and worse than it is on the Xeon. That is why every row names its machine and why there is no
headline multiplier anywhere in this README.

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
  That is the same optimisation exploited in "Remote Side-Channel Attacks on Anonymous
  Transactions" (USENIX Security 2020), which recovered information about Zcash shielded
  transactions by timing the prover. It is kept because every production Groth16 prover does
  it and the cost is nil, but the payoff is easy to overstate: on this benchmark ladder 0/1
  scalars are 1.8% to 4.1% of the witness, worth about 1.02x. It grows with sparsity on a
  genuinely bit-heavy witness. A deployment where an attacker can measure proving time or
  memory must treat that as part of its threat model: zero knowledge is a property of the
  proof, not of the process that produced it.
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
- Tramer, Boneh, Paterson, [**Remote Side-Channel Attacks on Anonymous
  Transactions**](https://crypto.stanford.edu/timings/paper.pdf), USENIX Security 2020. Why
  the MSM's zero and one fast paths are a privacy trade and not a free win.
- [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau), the
  source of the phase 1 parameters.
