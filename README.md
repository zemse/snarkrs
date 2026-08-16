# gpu snark: g16

- Snarkjs compatible .zkey and .wtns format
- CPU (pure rust)
- Apple Metal
- NVIDIA CUDA
- BN254 support
- Backends behind a rust feature flags
- Library support

## benchmarks

### apple-m2-max

> note: our CPU benchmarks appear faster than rapidsnark (CPU only) on apple silicon

**warm**:

| program         | constraints | g16 cpu | g16 metal | g16 cuda | rapidsnark | snarkjs |
| --------------- | ----------: | ------: | --------: | -------: | ---------: | ------: |
| `railgun-01x01` |      20,135 |    65.4 |      29.1 |          |       72.0 |         |
| `tornado`       |      28,275 |   102.9 |      40.2 |          |      117.5 |         |
| `sha256`        |      59,281 |    57.3 |      21.0 |          |       72.0 |         |
| `railgun-13x01` |     141,276 |   390.3 |     125.2 |          |      421.0 |         |
| `rsa2048`       |     190,945 |   225.4 |      56.8 |          |      344.5 |         |
| `keccak256`     |     239,176 |   216.1 |      38.8 |          |      289.0 |         |
| `anon-aadhaar`  |   1,115,080 |  1764.5 |     263.2 |          |     2120.0 |         |

**cold**:

| program         | constraints | g16 cpu | g16 metal | g16 cuda | rapidsnark | snarkjs |
| --------------- | ----------: | ------: | --------: | -------: | ---------: | ------: |
| `railgun-01x01` |      20,135 |    65.1 |      46.3 |          |       77.0 |   879.7 |
| `tornado`       |      28,275 |   107.3 |      59.0 |          |      123.1 |  1260.3 |
| `sha256`        |      59,281 |    64.9 |      46.5 |          |       80.4 |  1141.2 |
| `railgun-13x01` |     141,276 |   401.7 |     168.9 |          |      436.8 |  4340.2 |
| `rsa2048`       |     190,945 |   244.9 |     110.8 |          |      360.4 |  3928.5 |
| `keccak256`     |     239,176 |   231.2 |      87.3 |          |      308.3 |  3380.8 |
| `anon-aadhaar`  |   1,115,080 |  1894.5 |     523.1 |          |     2195.3 | 22392.5 |

### g4dn.2xlarge

**warm**:

| program         | constraints | g16 cpu | g16 metal | g16 cuda | rapidsnark | snarkjs |
| --------------- | ----------: | ------: | --------: | -------: | ---------: | ------: |
| `railgun-01x01` |      20,135 |   231.0 |           |     23.7 |      170.0 |         |
| `tornado`       |      28,275 |   396.1 |           |     32.0 |      300.0 |         |
| `sha256`        |      59,281 |   193.6 |           |     18.0 |      175.0 |         |
| `railgun-13x01` |     141,276 |  1447.7 |           |    135.3 |     1191.5 |         |
| `rsa2048`       |     190,945 |   776.9 |           |     64.7 |      790.5 |         |
| `keccak256`     |     239,176 |   758.1 |           |     51.4 |      682.0 |         |
| `anon-aadhaar`  |   1,115,080 |  5925.0 |           |    406.5 |     5774.0 |         |

**cold**:

| program         | constraints | g16 cpu | g16 metal | g16 cuda | rapidsnark | snarkjs |
| --------------- | ----------: | ------: | --------: | -------: | ---------: | ------: |
| `railgun-01x01` |      20,135 |   238.8 |           |    266.5 |      171.0 |  1757.4 |
| `tornado`       |      28,275 |   407.2 |           |    281.2 |      298.2 |  2517.4 |
| `sha256`        |      59,281 |   214.9 |           |    269.1 |      183.4 |  2318.1 |
| `railgun-13x01` |     141,276 |  1496.9 |           |    429.8 |     1187.4 |  8157.4 |
| `rsa2048`       |     190,945 |   846.2 |           |    401.2 |      797.9 |  6904.7 |
| `keccak256`     |     239,176 |   788.1 |           |    388.8 |      703.3 |  6807.9 |
| `anon-aadhaar`  |   1,115,080 |  6296.8 |           |   1735.9 |     5627.8 | 44888.9 |

> note: constraint count is a poor predictor of proving time. e.g. `railgun-13x01` has fewer constraints than `keccak256` but takes more to prove.

to reproduce the benchmarks you can use the script on machine of interest:

```sh
bench/scripts/run-benchmark.sh --reps 10
```

## using it from the command line

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

## using it as a library

```rust
use g16_core::{prove::prove, verify::verify, Backend, StageTimings};
use g16_zkey::{wtns::Witness, ProvingKey};

// warm pk once for repeated proving the same circuit
let pk = ProvingKey::load(std::path::Path::new("circuit.zkey"))?;
let n_public = pk.n_public;
let circuit = g16_metal::MetalBackend::new()?.prepare(pk)?;

let w = Witness::load(std::path::Path::new("circuit.wtns"))?.0;
let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut ark_std::rand::thread_rng(), &mut t)?;

let public = &w[1..=n_public];
verify(&circuit.key().vk, public, &proof)?;
```

Swap `MetalBackend` for `g16_core::cpu::CpuBackend` or `g16_cuda::CudaBackend` to change
where it runs; nothing else in the snippet changes, which is the point of the trait.

The blinders come from the OS CSPRNG. There is no seed override on this path on purpose: a
reused `(r, s)` across two proofs of different witnesses leaks the witness, so the
deterministic entry point stays test-only.

## References

- [Remco Bloemen: The Groth16 prover, step by step](https://xn--2-umb.com/22/groth16/).
- [iden3/snarkjs](https://github.com/iden3/snarkjs)
- [iden3/rapidsnark](https://github.com/iden3/rapidsnark)
- [arkworks-rs/circom-compat](https://github.com/arkworks-rs/circom-compat)
- [0xPARC/zk-bug-tracker](https://github.com/0xPARC/zk-bug-tracker)
- [Groth16 proof malleability](https://ethresear.ch/t/transaction-malleability-attack-of-groth16-proof/15881).
- [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau).
