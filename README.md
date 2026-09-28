# gpu snark: g16

- Snarkjs compatible .zkey and .wtns format
- CPU (pure rust)
- Apple Metal
- NVIDIA CUDA
- WebGPU ([demo](https://gpu-snark.vercel.app/))
- BN254 support
- Backends behind a rust feature flags
- Library support

## benchmarks

### apple-m2-max

> note: our CPU benchmarks appear faster than rapidsnark (CPU only) on apple silicon

**proving (warm)**:

<table>
<thead>
<tr><th rowspan="2">program</th><th rowspan="2" align="right">constraints</th><th>GPU</th><th colspan="3">CPU</th></tr>
<tr><th align="right">g16 metal (ms)</th><th align="right">g16 cpu (ms)</th><th align="right">rapidsnark (ms)</th><th align="right">snarkjs (ms)</th></tr>
</thead>
<tbody>
<tr><td><code>railgun-01x01</code></td><td align="right">20,135</td><td align="right">29.1</td><td align="right">65.4</td><td align="right">72.0</td><td align="right"></td></tr>
<tr><td><code>tornado</code></td><td align="right">28,275</td><td align="right">40.2</td><td align="right">102.9</td><td align="right">117.5</td><td align="right"></td></tr>
<tr><td><code>sha256</code></td><td align="right">59,281</td><td align="right">21.0</td><td align="right">57.3</td><td align="right">72.0</td><td align="right"></td></tr>
<tr><td><code>railgun-13x01</code></td><td align="right">141,276</td><td align="right">125.2</td><td align="right">390.3</td><td align="right">421.0</td><td align="right"></td></tr>
<tr><td><code>rsa2048</code></td><td align="right">190,945</td><td align="right">56.8</td><td align="right">225.4</td><td align="right">344.5</td><td align="right"></td></tr>
<tr><td><code>keccak256</code></td><td align="right">239,176</td><td align="right">38.8</td><td align="right">216.1</td><td align="right">289.0</td><td align="right"></td></tr>
<tr><td><code>anon-aadhaar</code></td><td align="right">1,115,080</td><td align="right">263.2</td><td align="right">1764.5</td><td align="right">2120.0</td><td align="right"></td></tr>
</tbody>
</table>

> blank = not measurable on this box: snarkjs has no warm mode, its CLI starts a fresh
> process per proof. Its cold numbers are in `bench/results/machines/`.

**trusted setup** (faster of cpu/metal in bold):

phase 1, powers of tau, run once for every circuit that follows:

| command            | input    | g16 cpu (s) | g16 metal (s) | snarkjs (s) |
| ------------------ | -------- | ----------: | ------------: | ----------: |
| `ptau new`         | power 14 |**0.01** |         — |     0.5 |
| `ptau contribute`  | power 19 |    36.3 |  **5.2**  |   162.3 |
| `ptau beacon`      | power 19 |    36.2 |  **5.2**  |   162.0 |
| `ptau prepare`     | power 20 |   929.8 | **83.8**  |  ~9,300 |
| `ptau verify`      | power 20 |**16.9** |         — |   116.4 |

phase 2, the proving key, run once per circuit:

| command                | domain | g16 cpu (s) | g16 metal (s) | snarkjs (s) |
| ---------------------- | ------ | ----------: | ------------: | ----------: |
| `setup` circom         | 2^20   |     17.4 | **17.1**  |    86.4 |
| `zkey contribute`      | 2^20   |     16.4 |  **1.8**  |    72.6 |
| `zkey beacon`          | 2^20   |     16.3 |  **1.8**  |    73.6 |

### g4dn.2xlarge

**warm**:

<table>
<thead>
<tr><th rowspan="2">program</th><th rowspan="2" align="right">constraints</th><th>GPU</th><th colspan="3">CPU</th></tr>
<tr><th align="right">g16 cuda (ms)</th><th align="right">g16 cpu (ms)</th><th align="right">rapidsnark (ms)</th><th align="right">snarkjs (ms)</th></tr>
</thead>
<tbody>
<tr><td><code>railgun-01x01</code></td><td align="right">20,135</td><td align="right">23.7</td><td align="right">231.0</td><td align="right">170.0</td><td align="right"></td></tr>
<tr><td><code>tornado</code></td><td align="right">28,275</td><td align="right">32.0</td><td align="right">396.1</td><td align="right">300.0</td><td align="right"></td></tr>
<tr><td><code>sha256</code></td><td align="right">59,281</td><td align="right">18.0</td><td align="right">193.6</td><td align="right">175.0</td><td align="right"></td></tr>
<tr><td><code>railgun-13x01</code></td><td align="right">141,276</td><td align="right">135.3</td><td align="right">1447.7</td><td align="right">1191.5</td><td align="right"></td></tr>
<tr><td><code>rsa2048</code></td><td align="right">190,945</td><td align="right">64.7</td><td align="right">776.9</td><td align="right">790.5</td><td align="right"></td></tr>
<tr><td><code>keccak256</code></td><td align="right">239,176</td><td align="right">51.4</td><td align="right">758.1</td><td align="right">682.0</td><td align="right"></td></tr>
<tr><td><code>anon-aadhaar</code></td><td align="right">1,115,080</td><td align="right">406.5</td><td align="right">5925.0</td><td align="right">5774.0</td><td align="right"></td></tr>
</tbody>
</table>

> blank = not measurable on this box: snarkjs has no warm mode, its CLI starts a fresh
> process per proof. Its cold numbers are in `bench/results/machines/`.

> note: constraint count is a poor predictor of proving time. e.g. `railgun-13x01` has fewer constraints than `keccak256` but takes more to prove.

to reproduce the benchmarks you can use the script on machine of interest:

```sh
bench/scripts/run-benchmark.sh --reps 10
```

## using it from the command line

```sh
cargo install --path crates/cli                     # CPU and WebGPU
cargo install --path crates/cli --features metal    # + Apple Metal
cargo install --path crates/cli --features cuda     # + NVIDIA CUDA
```

The package is `snarkrs-cli` and the binary is `snarkrs`, a drop-in for the snarkjs 0.7.6 command line on Groth16 and
BN254: the same commands, aliases, positional file names and defaults, `-e=`/`-n=`/`-v`
options, exit codes (0 ok, 1 failed or invalid, 99 bad usage) and `[INFO]  snarkJS: OK!`
log lines. `snarkrs --help` lists what it runs; a snarkjs command it does not run yet
fails with exit 1 and says so.

```sh
snarkrs powersoftau new bn128 12                              # powersOfTau12_0000.ptau
snarkrs powersoftau contribute powersOfTau12_0000.ptau pot_0001.ptau -e=... -n=me
snarkrs powersoftau prepare phase2 pot_0001.ptau powersoftau.ptau
snarkrs groth16 setup circuit.r1cs powersoftau.ptau circuit_0000.zkey
snarkrs zkey contribute circuit_0000.zkey circuit_final.zkey -e=...
snarkrs zkey export verificationkey circuit_final.zkey verification_key.json
snarkrs groth16 prove circuit_final.zkey witness.wtns proof.json public.json \
        [--backend cpu|wgpu|metal|cuda] [--stage-timings] [--self-verify true|false] \
        [--vkey verification_key.json] [--constant-work]
snarkrs groth16 verify verification_key.json public.json proof.json

snarkrs bench --artifacts <DIR> [--variant NAME]... [--reps N] \
        [--backend cpu|wgpu|metal|cuda] [--mode cold|warm|both] [--csv FILE]
```

The short aliases work too (`ptn`, `ptc`, `pt2`, `g16s`, `zkc`, `zkev`, `g16p`, `g16v`, ...),
and the ceremony commands take `--backend cpu|metal`. The output is what snarkjs expects,
so the two are interchangeable in either direction:

```sh
snarkrs g16p circuit.zkey circuit.wtns proof.json public.json --backend metal
snarkjs groth16 verify verification_key.json public.json proof.json    # OK!
```

### witnesses

`wtns calculate` and `groth16 fullprove` take circom's `circuit_js/circuit.wasm` or the
native binary `circom --c` builds. The file's first bytes decide which, not its name:

```sh
snarkrs wtns calculate circuit_js/circuit.wasm input.json witness.wtns
snarkrs wtns calculate circuit_cpp/circuit input.json witness.wtns   # circuit.dat beside it
snarkrs groth16 fullprove input.json circuit_js/circuit.wasm circuit_final.zkey \
        proof.json public.json [--backend cpu|wgpu|metal|cuda] [...groth16 prove's options]
```

The wasm runs in wasmtime, reads `input.json` the way snarkjs does, prints the circuit's
`log()` lines and its errors the way snarkjs does, and writes the same `.wtns` byte for
byte. `fullprove` keeps that witness in memory. A native binary runs as a subprocess and
`fullprove` hands it a private temp file that is zeroed and deleted once read. `wtns debug`
is wasm only. `--no-default-features --features cpu,wgpu` builds without the
`witness-wasm` feature: no wasmtime, native binaries only.

On Apple Silicon the C++ from `circom --c` does not build as it comes: `fr.cpp` passes
`uint64_t*` where GMP takes `mp_limb_t*`, which is `unsigned long` there, and `main.cpp`
includes `nlohmann/json.hpp`, which the generated Makefile never points at. `--no_asm`
leaves out `fr.asm` and the nasm it needs.

`--stage-timings` prints where the time went, which is the fastest way to find out whether
a circuit is MSM-bound or transform-bound:

## using it as a library

```toml
snarkrs = { git = "https://github.com/zemse/gpu-snark", features = ["metal"] }
```

The prover, verifier, key formats and CPU backend are always in. The rest is opt in, so a
build compiles only the backend it runs on:

| feature | adds |
| --- | --- |
| `metal` | `snarkrs::metal`, Apple GPUs |
| `cuda` | `snarkrs::cuda`, NVIDIA GPUs |
| `wgpu` | `snarkrs::wgpu`, WebGPU |
| `ceremony` | `snarkrs::ceremony`, powers of tau and phase 2 |
| `witness` | `snarkrs::witness`, circom's native witness binary |
| `witness-wasm` | `witness` plus circom's `circuit.wasm` on wasmtime |

```rust
use snarkrs::{prove, verify, Backend, ProvingKey, StageTimings, Witness};

// warm pk once for repeated proving the same circuit
let pk = ProvingKey::load("circuit.zkey".as_ref())?;
let n_public = pk.n_public;
let circuit = snarkrs::metal::MetalBackend::new()?.prepare(pk)?;

let w = Witness::load("circuit.wtns".as_ref())?.0;
let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut snarkrs::rand::thread_rng(), &mut t)?;

let public = &w[1..=n_public];
verify(&circuit.key().vk, public, &proof)?;
snarkrs::write_proof("proof.json".as_ref(), &proof)?;
snarkrs::write_public("public.json".as_ref(), public)?;
```

Swap `MetalBackend` for `snarkrs::CpuBackend` or `snarkrs::cuda::CudaBackend` to change where
it runs; nothing else in the snippet changes, which is the point of the trait.

The blinders come from the OS CSPRNG. There is no seed override on this path on purpose: a
reused `(r, s)` across two proofs of different witnesses leaks the witness, so the
deterministic entry point stays test-only.

### witness from memory

`prove` takes the witness as `&[Fr]`, so it never has to be a file:

```rust
use snarkrs::witness::{Input, WitnessCalculator};
use snarkrs::{prove, Backend, ProvingKey, StageTimings};

let pk = ProvingKey::load("circuit.zkey".as_ref())?;
let circuit = snarkrs::metal::MetalBackend::new()?.prepare(pk)?;

// compile the wasm once, then one witness per input (feature `witness-wasm`)
let calc = WitnessCalculator::from_file("circuit_js/circuit.wasm".as_ref())?;
let w = calc.calculate(&Input::from_json_str(r#"{"a": "3", "b": "11"}"#)?)?;

let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut snarkrs::rand::thread_rng(), &mut t)?;
```

`w` is just `1`, the public signals, then the other wires in circom's order (the second
column of the `.sym` file). Anything that computes it can feed `prove`, so the fastest
witness generator is one written in optimised Rust for your circuit, with no wasm and no
`.wtns` written and read back in between. `snarkrs` prints a tip saying so on stderr after
`wtns calculate` and `fullprove`; `SNARKRS_NO_TIPS=1` turns it off.

## what it checks

Every entry point validates its input by default, and each has an `_unchecked` twin that
skips the check for input you already trust:

| checked | what it adds | unchecked |
| --- | --- | --- |
| `ProvingKey::load`, `from_bytes` | every point on the curve; refuses a key with no phase-2 contribution | `load_unchecked`, `from_bytes_unchecked` |
| `VerifyingKey::from_json` | refuses a key anyone can forge against | `from_json_unchecked` |
| `verify` | proof points on the curve, in the subgroup, not infinity | `verify_unchecked` |
| `prove` | verifies its own proof before returning it | `prove_unchecked` |

The check in `prove` is also what stops a hostile zkey from reading the witness out of the
proof, so a key from someone else should only ever meet `prove`, and should still be checked
with `snarkrs zkey verify` against the circuit and the ptau. Proving time depends on how many
witness entries are zero or one unless you pass `--constant-work` (cpu, metal and wgpu
backends, 2.5% to about 10x slower depending on the circuit and backend). The audit and its
current status are in
[`security/README.md`](security/README.md).

## References

- [Remco Bloemen: The Groth16 prover, step by step](https://xn--2-umb.com/22/groth16/).
- [iden3/snarkjs](https://github.com/iden3/snarkjs)
- [iden3/rapidsnark](https://github.com/iden3/rapidsnark)
- [arkworks-rs/circom-compat](https://github.com/arkworks-rs/circom-compat)
- [0xPARC/zk-bug-tracker](https://github.com/0xPARC/zk-bug-tracker)
- [Groth16 proof malleability](https://ethresear.ch/t/transaction-malleability-attack-of-groth16-proof/15881).
- [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau).
