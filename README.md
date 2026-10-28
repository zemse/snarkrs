# gpu snark: g16

- Snarkjs compatible .zkey and .wtns format
- CPU (pure rust)
- Apple Metal
- NVIDIA CUDA
- WebGPU ([demo](https://gpu-snark.vercel.app/))
- BN254 support
- Backends behind a rust feature flags
- Library support

[Backend support and validation](BACKENDS.md) covers work modes, ceremony capabilities,
failure policies and the hardware actually tested.

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
<tr><td><code>railgun-01x01</code></td><td align="right">20,135</td><td align="right">12.2</td><td align="right">34.7</td><td align="right">72.0</td><td align="right"></td></tr>
<tr><td><code>tornado</code></td><td align="right">28,275</td><td align="right">20.1</td><td align="right">54.2</td><td align="right">118.7</td><td align="right"></td></tr>
<tr><td><code>sha256</code></td><td align="right">59,281</td><td align="right">11.1</td><td align="right">29.2</td><td align="right">71.7</td><td align="right"></td></tr>
<tr><td><code>railgun-13x01</code></td><td align="right">141,276</td><td align="right">52.5</td><td align="right">194.9</td><td align="right">419.9</td><td align="right"></td></tr>
<tr><td><code>rsa2048</code></td><td align="right">190,945</td><td align="right">30.3</td><td align="right">110.5</td><td align="right">340.8</td><td align="right"></td></tr>
<tr><td><code>keccak256</code></td><td align="right">239,176</td><td align="right">27.6</td><td align="right">105.1</td><td align="right">284.1</td><td align="right"></td></tr>
<tr><td><code>anon-aadhaar</code></td><td align="right">1,115,080</td><td align="right">172.9</td><td align="right">741.1</td><td align="right">2101.5</td><td align="right"></td></tr>
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
<tr><td><code>railgun-01x01</code></td><td align="right">20,135</td><td align="right">22.8</td><td align="right">117.7</td><td align="right">172.2</td><td align="right"></td></tr>
<tr><td><code>tornado</code></td><td align="right">28,275</td><td align="right">31.3</td><td align="right">197.1</td><td align="right">299.7</td><td align="right"></td></tr>
<tr><td><code>sha256</code></td><td align="right">59,281</td><td align="right">15.6</td><td align="right">97.5</td><td align="right">178.8</td><td align="right"></td></tr>
<tr><td><code>railgun-13x01</code></td><td align="right">141,276</td><td align="right">132.0</td><td align="right">693.1</td><td align="right">1166.0</td><td align="right"></td></tr>
<tr><td><code>rsa2048</code></td><td align="right">190,945</td><td align="right">61.0</td><td align="right">370.8</td><td align="right">754.8</td><td align="right"></td></tr>
<tr><td><code>keccak256</code></td><td align="right">239,176</td><td align="right">50.1</td><td align="right">355.6</td><td align="right">661.1</td><td align="right"></td></tr>
<tr><td><code>anon-aadhaar</code></td><td align="right">1,115,080</td><td align="right">394.4</td><td align="right">2521.6</td><td align="right">5219.0</td><td align="right"></td></tr>
</tbody>
</table>

> blank = not measurable on this box: snarkjs has no warm mode, its CLI starts a fresh
> process per proof. Its cold numbers are in `bench/results/machines/`.

> note: g16 cpu here is built with `RUSTFLAGS="-C target-feature=+adx,+bmi2"`, as the x86_64 release binaries are. A build without the flags now runs the same field assembly, choosing the mulx multiply at run time when CPUID reports ADX and BMI2, and measured 1.5% to 3.0% slower on this box. Before that dispatch it was 1.5x to 1.6x slower and lost to rapidsnark on the two smallest circuits.

> note: constraint count is a poor predictor of proving time. e.g. `railgun-13x01` has fewer constraints than `keccak256` but takes more to prove.

to reproduce the benchmarks you can use the script on machine of interest:

```sh
bench/scripts/run-benchmark.sh --reps 10
```

`npm --prefix bench test` checks these warm proving tables against
`bench/results/machines/` and runs the table-check regressions, without running proofs.
The separate trusted setup tables are not covered. Suspect runs cannot support a table;
multiple records for one machine require an explicit source selection rather than silently
picking a newer run.

## using it from the command line

```sh
cargo install --path bin/snarkrs                     # CPU and WebGPU
cargo install --path bin/snarkrs --features metal    # + Apple Metal
cargo install --path bin/snarkrs --features cuda     # + NVIDIA CUDA
```

On x86_64 a plain install runs the field assembly, with the mulx multiply whenever the CPU
reports ADX and BMI2, so no `RUSTFLAGS` are needed. `RUSTFLAGS="-C target-feature=+adx,+bmi2"` buys the last few
percent, and the binary then refuses to start on a CPU without them.

The package and the binary are both `snarkrs`, a drop-in for the snarkjs 0.7.6 command line on Groth16 and
BN254: the same commands, aliases, positional file names and defaults, `-e=`/`-n=`/`-v`
options and exit codes (0 ok, 1 failed or invalid, 99 bad usage). Results such as `OK!`
print bare on stdout, and warnings and errors go to stderr as `warning:` and `error:`
lines. `snarkrs --help` lists what it runs; a snarkjs command it does not run yet
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
        [--backend cpu|wgpu|metal|cuda] [--mode cold-inprocess|cold-process|warm|both] \
        [--csv FILE]
```

The short aliases work too (`ptn`, `ptc`, `pt2`, `g16s`, `zkc`, `zkev`, `g16p`, `g16v`, ...),
and the ceremony commands generally take `--backend cpu|metal`. CUDA supports only the
`ptau prepare` group FFT, not ceremony MSM or key scaling; WGPU supports none of those
three ceremony seams. The output is what snarkjs expects, so the two are interchangeable
in either direction:

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
snarkrs-lib = { git = "https://github.com/zemse/snarkrs", features = ["metal"] }
```

Proving needs an optimised build. In a debug build `prove` returns `ProveError::Unoptimized`
rather than run tens of times slower, so optimise dependencies there too, which keeps your
own code quick to compile and debug:

```toml
[profile.dev.package."*"]
opt-level = 3
```

The prover, verifier, key formats and CPU backend are always in. The rest is opt in, so a
build compiles only the backend it runs on:

| feature | adds |
| --- | --- |
| `metal` | `snarkrs_lib::metal`, Apple GPUs |
| `cuda` | `snarkrs_lib::cuda`, NVIDIA GPUs |
| `wgpu` | `snarkrs_lib::wgpu`, WebGPU |
| `ceremony` | `snarkrs_lib::ceremony`, powers of tau and phase 2 |
| `witness` | `snarkrs_lib::witness`, circom's native witness binary |
| `witness-wasm` | `witness` plus circom's `circuit.wasm` on wasmtime |

```rust
use snarkrs_lib::{prove, verify, Backend, ProvingKey, StageTimings, Witness};

// warm pk once for repeated proving the same circuit
let pk = ProvingKey::load("circuit.zkey".as_ref())?;
let n_public = pk.n_public;
let circuit = snarkrs_lib::metal::MetalBackend::new()?.prepare(pk)?;

let w = Witness::load("circuit.wtns".as_ref())?.0;
let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut snarkrs_lib::rand::thread_rng(), &mut t)?;

let public = &w[1..=n_public];
verify(&circuit.key().vk, public, &proof)?;
snarkrs_lib::write_proof("proof.json".as_ref(), &proof)?;
snarkrs_lib::write_public("public.json".as_ref(), public)?;
```

Swap `MetalBackend` for `snarkrs_lib::CpuBackend` or `snarkrs_lib::cuda::CudaBackend` to change where
it runs; nothing else in the snippet changes, which is the point of the trait.

`circuit.key()` preserves the dimensions, full verification key (including IC), and the
five assembly headers (`alpha_g1`, `beta_g1`, `beta_g2`, `delta_g1`, `delta_g2`). Bulk
coefficients and query vectors are backend-dependent and may be released after upload.
It is not guaranteed to be a complete key for preparing another backend; retain or reload
the original key for that.

The blinders come from the OS CSPRNG. There is no seed override on this path on purpose: a
reused `(r, s)` across two proofs of different witnesses leaks the witness, so the
deterministic entry point stays test-only.

### witness from memory

`prove` takes the witness as `&[Fr]`, so it never has to be a file:

```rust
use snarkrs_lib::witness::{Input, WitnessCalculator};
use snarkrs_lib::{prove, Backend, ProvingKey, StageTimings};

let pk = ProvingKey::load("circuit.zkey".as_ref())?;
let circuit = snarkrs_lib::metal::MetalBackend::new()?.prepare(pk)?;

// compile the wasm once, then one witness per input (feature `witness-wasm`)
let calc = WitnessCalculator::from_file("circuit_js/circuit.wasm".as_ref())?;
let w = calc.calculate(&Input::from_json_str(r#"{"a": "3", "b": "11"}"#)?)?;

let mut t = StageTimings::default();
let proof = prove(circuit.as_ref(), &w, &mut snarkrs_lib::rand::thread_rng(), &mut t)?;
```

`w` is just `1`, the public signals, then the other wires in circom's order (the second
column of the `.sym` file). Anything that computes it can feed `prove`, so the fastest
witness generator is one written in optimised Rust for your circuit, with no wasm and no
`.wtns` written and read back in between. `snarkrs` prints a tip saying so on stderr after
`wtns calculate` and `fullprove`; `SNARKRS_NO_TIPS=1` turns it off.

## what it checks

These APIs have checked defaults and explicit `_unchecked` alternatives for input you
already trust. This is not a claim that every entry point has an unchecked twin:

| checked | what it adds | unchecked |
| --- | --- | --- |
| `ProvingKey::load`, `from_bytes` | every point on the curve; refuses a key with no phase-2 contribution | `load_unchecked`, `from_bytes_unchecked` |
| `VerifyingKey::from_json` | refuses a key anyone can forge against | `from_json_unchecked` |
| `verify` | proof points on the curve, in the subgroup, not infinity | `verify_unchecked` |
| `prove` | verifies its own proof before returning it | `prove_unchecked` |

The check in `prove` is also what stops a hostile zkey from reading the witness out of the
proof, so a key from someone else should only ever meet `prove`, and should still be checked
with `snarkrs zkey verify` against the circuit and the ptau. Proving time depends on how many
witness entries are zero or one unless you pass `--constant-work` (cpu, metal, wgpu and cuda
backends, with circuit-dependent overhead). This fixes the scalar-dependent MSM sizing
and disables zero/one fast paths, not all witness-dependent work: GPU bucket occupancy,
atomics contention and accumulation still depend on the witness. Constant-work is not
constant-time. CUDA's bounded Tesla T4 validation and measured overhead are in the
[backend matrix](BACKENDS.md#evidence). The audit and its current status are in
[`security/README.md`](security/README.md).

## References

- [Remco Bloemen: The Groth16 prover, step by step](https://xn--2-umb.com/22/groth16/).
- [iden3/snarkjs](https://github.com/iden3/snarkjs)
- [iden3/rapidsnark](https://github.com/iden3/rapidsnark)
- [arkworks-rs/circom-compat](https://github.com/arkworks-rs/circom-compat)
- [0xPARC/zk-bug-tracker](https://github.com/0xPARC/zk-bug-tracker)
- [Groth16 proof malleability](https://ethresear.ch/t/transaction-malleability-attack-of-groth16-proof/15881).
- [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau).
