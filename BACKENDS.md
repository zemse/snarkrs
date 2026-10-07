# backends

Scope: main `d506763981fb9234d32b191ca47f9b45930da4d0`. Source support is not hardware
validation. Historical benchmark tables are not current conformance evidence.

## implementation

| backend | proving NTT | G1/G2 MSM | H storage | host bulk key retained | work modes | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | CPU | CPU | host | yes | variable, constant | native Rust |
| Metal | GPU | GPU | device | no | variable, constant | Apple Metal, feature-gated |
| CUDA | GPU | GPU | device | yes | variable only | NVIDIA CUDA/NVRTC, feature-gated |
| WGPU | GPU | GPU | device | yes | variable, constant | native WGPU or browser WebGPU, feature-gated |

The GPU proving paths keep H resident between its computation and the H MSM. Debug
readback and host-H comparisons are separate from that normal path. A device H handle
belongs to the circuit that computed it, not another circuit or backend.

`PreparedCircuit::key()` guarantees `n_vars`, `n_public`, `domain_size`, the full `vk`
(including `ic`), and `alpha_g1`, `beta_g1`, `beta_g2`, `delta_g1`, `delta_g2`. Dimensions
agree with the circuit accessors. Bulk coefficients and the five query vectors are
backend-dependent: Metal releases them after upload. Retain or reload the original key
to prepare another backend; `key()` is not a re-preparation contract.

Variable work is the default. Constant work fixes scalar-dependent MSM sizing and
removes zero/one fast paths, but does not make GPU occupancy, contention or all other
witness-dependent work constant-time. CUDA constant work is unavailable on this main;
held D2 changes are not landed support.

| ceremony seam | CPU | Metal | CUDA | WGPU |
| --- | --- | --- | --- | --- |
| MSM (`setup`) | supported | supported | unsupported | unsupported |
| group FFT (`ptau prepare`) | supported | supported | supported | unsupported |
| key scaling (contribute/beacon) | supported | supported | unsupported | unsupported |

These are capability boundaries, not a promise that every ceremony command uses a GPU.
Unsupported CLI selections fail before input/output work or entropy prompts rather than
silently substituting CPU. Supported accelerators still require their build feature and
runtime. See the [selection gates](crates/cli/src/main.rs) and
[prepared-circuit contract](crates/groth16/src/lib.rs).

## failure policies

Checked `prove` verifies its proof before returning it. Backend-internal submission
retries are distinct from CLI proof retries: Metal and native WGPU retry eligible
interrupted submissions, without switching to CPU. Library proving and browser WebGPU
proving do not automatically fall back to CPU.

The CLI defaults to self-verification and fallback. After a GPU self-verification failure
or device fault it retries the proof once on that backend, then reloads the key for a
checked CPU proof. The CPU factory must identify as CPU and match dimensions, full VK/IC
and the five assembly headers. Deterministic input errors do not trigger this policy;
CPU preparation or proving errors propagate, not success. Timings name the successful
backend. `--fallback=false` disables CLI fallback; `--self-verify=false` bypasses checked
proving and disables that fallback policy. See the [CLI policy](crates/cli/src/fallback.rs).

WGPU `Auto` is limit negotiation, not CPU backend selection. It raises buffer and
storage-binding capacity while keeping Floor kernel geometry; a refused Auto device
request can retry with Floor limits on WebGPU and records the reason. Chunked base
bindings are implemented, but other buffers, bindings and total memory still bound
capacity. A capacity refusal is not a proof-success result.

Accelerated ceremony outputs use a reserved sibling stage and publish by rename only
after the operation completes. Errors leave the destination unchanged; cleanup is best
effort. The [writer contract](crates/ceremony/src/write.rs) assumes a trusted output
directory, not adversarial path replacement. Validated Unix/APFS behavior is atomic
visibility, not crash durability (no fsync), universal filesystem alias handling, or
arithmetic verification. Low-level `BinFileWriter::create` still truncates directly.
Metal's sealed ceremony/group-FFT submissions detect incomplete execution. A matching
completion token does not detect completed but wrong arithmetic.

## evidence

CURRENT means implemented in this source. EXECUTED means the named run passed within
its stated scope. COMPILE means build coverage only. BLOCKED/NOT RUN and UNSUPPORTED
are not passes. The runs below are recorded evidence, not new runs performed for this
documentation change. Commands and detailed F evidence live in
[test-support/README.md](test-support/README.md#required-cli-conformance-stage4-par-f).

| status | scope | evidence and limits |
| --- | --- | --- |
| EXECUTED, coordinator F reruns on `2348fb0` | CPU variable/constant; Metal variable/constant on Apple M2 Max; WGPU Floor and Auto variable/constant on Apple M2 Max IntegratedGpu, Metal transport | 9 guards passed. The 2 proof gates are ignored by default, but explicit selection passed all 8 jobs, 16/16 fixture cases. |
| EXECUTED, F agent before review | capacity and negative selection checks | 3 CPU capacity tests and exact Floor `2^23` pre-allocation refusal passed; 5 negative process probes failed nonzero with `UNAVAILABLE`. These are not coordinator reruns or large proof passes. |
| COMPILE, F | CPU/CUDA integration targets; default plus Metal/WGPU targets | release/locked/offline builds passed; CUDA had no current physical run. |
| EXECUTED, Stage3 | 5 fallback tests, 56 web tests; desktop Floor Chrome 154 smoke, both work modes | 14 selftests and 4 snarkjs-accepted GPU proofs passed. Browser scope is bounded tiny-circuit validation, not phones or production assets. |
| EXECUTED, C3 `621a9e0` | Metal sealed ceremony/group FFT | 68 tests passed, 1 measurement ignored. Synthetic fault recovery and CPU comparisons, not a production arithmetic guard or BUG-28 real-load closure. |
| EXECUTED, main `d506763` | domain `2^22` same-instance large proof gate, variable work: CPU, Metal, WGPU Floor/Auto | 4/4 jobs passed, each `1/1` on `large/js_384x384_d32`, totaling 4 large cases; both verifiers and reference public signals accepted, no fallback. Large constant work and domains above `2^22` remain unexecuted. |
| CURRENT, historical only | NVIDIA variable-work prover and CUDA ceremony group FFT | source support and historical benchmarks, not current hardware validation. |
| BLOCKED / UNSUPPORTED on main | held CUDA D2 constant-work changes | require physical NVRTC/PTX arithmetic and default/constant proof gates before merge; no approved NVIDIA host or rental. |
| BLOCKED | phones, Safari and other browser/device combinations | not currently validated by the desktop smoke or shader compilation. |

Each corrected F case covered `tiny_mul` (domain 8) or SHA-256 (domain 65536): CPU H and
all five MSM comparisons, exact GPU Device-H backend tag (CPU Host-H), resident versus
host-H, zero/nonzero pinned proofs accepted by both our verifier and snarkjs, and repeat
proving. Tiny also had independent H/MSM vectors, malformed-key and concurrent proving
checks against sequential CPU references. SHA had a CPU stage oracle and independent
proof verification, not an independent stage oracle. These are direct-API/JSON
interoperability checks, not CLI argument-dispatch proof coverage.

F Floor requested/granted buffer and storage-binding limits were 268435456/134217728
bytes. Auto requested/granted 4294967295/4294967292 bytes without fallback. Both kept
16 KiB workgroup storage and 256 invocations. This is bounded proof and negotiation
evidence, not universal capacity.

The later large run used unchanged tracked source at `d506763`, domain 4194304, on the
same inspected backend instance without fallback. Metal reported Apple M2 Max; WGPU
reported Apple M2 Max IntegratedGpu with Metal transport. Floor requested/granted
268435456/134217728 buffer/storage-binding bytes; Auto requested/granted
4294967295/4294967292 without fallback. Both kept 16384 workgroup bytes and 256
invocations. Our verifier accepted each serialized proof against the fixture vkey;
snarkjs accepted it and public signals matched the reference. Four proof/public pairs
were retained and their sizes, JSON and SHA-256 hashes checked.

Large evidence is retained at repository-root `target/large-validation-main/summary.json`
and `target/large-validation-main/handoff.json`, with per-job logs and proof/public JSON
under that directory (local, ignored, not published). The original F worktree evidence
was removed by automatic cleanup; it is not a retained evidence path. These four large
cases are separate from the bounded eight jobs and sixteen cases above. Normal memory
pressure was observed. Elapsed times include builds; sampled aggregate process RSS is
not physical footprint. This validation is not a benchmark or speed comparison, or CLI
argument-dispatch coverage. See the [large-run evidence](test-support/README.md#large-main-execution-evidence-d506763).

Stage3's local smoke evidence is `target/parity-browser/stage3-coordinator-smoke/evidence.json`
(ignored, not published). Rust wasm source revision `d16e568` and supplied-byte SHA-256
`aa141e9c237d276c3dc5984d09f02d38286aaa179fd2e6dd474ae3921c3c04a3` identify caller-declared,
pre-wasm-opt release assets. The run substituted a tiny catalogue/configuration. Adapter
identity came from a separately requested window adapter, not worker-device attestation;
see the [smoke harness](web/scripts/browser-parity.mjs).

This evidence does not establish full backend parity. NVIDIA, phones, large constant-work
runs, domains above `2^22` and a production wrong-arithmetic guard remain open. Default
full CPU duplication has not been approved for its cost; sealed completion and synthetic
recovery do not replace it.
