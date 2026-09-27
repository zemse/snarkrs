# Security and hardening report

Four audits and three adversarial test suites, run against the tree at commit `e04b9af`
on an Apple M2 Max, macOS 26.6. Raw notes are in `security/notes/`: four `audit-*.md`
files (our code) and four `research-*.md` files (prior art, CVEs, and the theory the
audits are measured against). The test suites are
`crates/g16-cli/tests/{campaign,verifier_negative,edge_witness}.rs`.

Nothing in this report is a fix. The audits were report-only and the test lanes added
test files only. The findings below are as audited, and their file and line references are
to that tree; the table that follows says what has landed since.

## Status as of 2026-09-27

| # | Finding | Status |
|---|---|---|
| 1 | Hostile zkey zeroes the blinding | Fixed, `3b1333e` |
| 2, 17 | `domainSize` allocation bomb, no two-adicity bound | Fixed, `3b1333e` |
| 3 | Query sections never validated | Fixed, `9445d85`: every point on the curve at load; the G2 subgroup is checked on the proof's `B` (`d170f78`, `3a610a5`) because per point it costs 1.2-6.6 s. `load_unchecked` skips it |
| 4 | Metal command buffer status never checked | Fixed, `69d2802` (`cb::wait_ok`) |
| 5 | Threadgroup memory never cleared | Fixed, `dd21174` |
| 6 | Witness-dependent MSM cost | CPU: opt-in constant work, `ae15a9b` (`prove --constant-work`; +2.5% dense, 3.3-4x bit-heavy). Metal: opt-in constant work, `f48db69` (`--backend metal`; +15% dense, 7-11x bit-heavy); bucket occupancy and stage 0's 0/1 multiply skip remain. wgpu, cuda: **open** |
| 7 | GPU concurrency failure | Fixed on Metal, `2e2d06f`, and on wgpu, `4ce2c96`, where wgpu hid the kill and proofs came back wrong: macOS `ImpactingInteractivity` kills of a proving command buffer, now retried whole. One unchecked proof came back wrong after a GPU hang and recovery under deliberate three-process overload: **open**, caught by `prove`'s self-verify |
| 8 | Lenient proof JSON encoding | Fixed, `d170f78` |
| 9 | `verify()` validates nothing | Fixed, `d170f78`; `verify_unchecked` is the bare check |
| 10 | Section 4 read twice, second pass unchecked | Fixed, `9445d85` |
| 11 | Coordinates never range-checked | Fixed, `77230e1` |
| 12 | SIGBUS on concurrent truncation | Accepted risk |
| 13 | `n_public` trusted from the key | Fixed, `ded32c4` (`prove --vkey`; an all-public key is refused without it) |
| 14 | Metal pools keep the previous witness | Fixed, `dd21174` |
| 15 | Bucket-clear couplings | Fixed, `dd21174` |
| 16 | Exceptional additions untested | Fixed, `dd21174` |
| 18, 19 | Section-count reservation, 32-bit arithmetic | Accepted risk |
| 20 | Section 10 never read | Accepted risk; `g16 zkey verify` is the non-circular check |
| 21, 22 | Infinity in a proof, degenerate key | Fixed, `d170f78` |
| 23 | Empty IC panics | Fixed, `b46c3e8` |
| 24 | `multi_pairing` unwraps | Fixed, `b48d1b7` |
| 25 | No memory hygiene | Fixed where it can be, `159c672`: scrubbed witness images, no core file from `prove` |
| 26 | `prove_with_blinders` public | Fixed, `6841752`: behind a dev-only feature; `assemble` draws its own blinders |
| 27 | Metal fragility items | Fixed, `dd21174` |
| 28 | In-process witness not checked | Fixed, `3a610a5` |
| 29 | Shifted query sections leak through `C` | Found since; see below. Closed on every checked path, `3a610a5` |
| 30 | Browser prove returned unchecked proofs | Found since. Fixed, `3a610a5`, `6841752` |
| 31 | Key with no phase-2 contribution accepted | Found since. Fixed, `9445d85` |
| 32 | wgpu floor profile cannot hold anon-aadhaar's G2 bases | Found since. **Open**, U15 (chunked base bindings); the `auto` profile works |

## Found since the audit

**29. A hostile key with valid points leaks the witness through `C`.** Finding 1 zeroed
points; this one needs none. Shift the L base of one private wire by `E` and every load
check passes, yet `C` moves by exactly `w * E`, which whoever holds that key's trapdoor can
read back by computing the honest `C` from `A` and `B`. Run, not reasoned:
`g16-core::prove::tests::a_shifted_l_query_leaks_through_c_and_only_the_checked_prove_stops_it`.
The defence is the self-verify that `prove` now runs by default: with `delta` nonzero in both
groups `A` and `B` are uniform, so a proof that verifies under a fixed key has `C` determined
by `A`, `B` and the statement. What is left is one bit per call, whether that check passed,
and the key chooses the linear condition on the witness that decides it. Only `g16 zkey
verify` against the circuit and the ptau closes that, and `prove_unchecked` is exposed to the
whole leak.

**30. The browser proved without checking.** The wasm prove path assembled its own proof and
returned it unverified, on the one backend whose key arrives from someone else and whose
shaders WebKit has miscompiled. Its warm and cold paths now draw, assemble and verify through
one function.

**31. A key with no phase-2 contribution was accepted.** `gamma_g2 == delta_g2` is what a key
looks like before phase 2, and anyone can forge against it with `A = alpha`, `B = beta`,
`C = -L`; the Foom and Veil verifiers were drained that way (zkSecurity, February 2026).
`ProvingKey::load` and `VerifyingKey::from_json` refuse it; the `_unchecked` loaders accept it
for development keys.

**32. The wgpu floor profile has a capacity limit.** Under the browser's 128 MiB binding floor,
anon-aadhaar's G2 bases (140.9 MB) do not fit and the backend refuses with a clean error. The
`auto` profile, which is what the browser requests, raises the limit to what the adapter
grants. Chunking the base bindings is U15.

## What we checked, and how

Four dimensions, each read line by line and then attacked.

**Untrusted inputs** (`g16-zkey`). Every line of `binfile.rs`, `lib.rs` and `wtns.rs`
read. Then **60 crafted mutants** of `bench/artifacts/tiny_mul/circuit.zkey` and
`circuit.wtns` plus two live timing races against the 94 MB `js_16x16_d32` key, all run
through `target/release/g16`. Executed, not reasoned: the 34 GB allocation, the SIGBUS,
the SIGABRT, and the zero-knowledge break all have run output behind them.

**Verifier soundness** (`g16-core/src/verify.rs`, `g16-cli/src/json.rs`, the vkey reader
in `g16-zkey`). A probe binary outside the repo linked against the real crates and run
against the real `js_2x2_d16` artifact. Re-randomisation, non-canonical encodings, the
infinity cases and the degenerate-key case were all executed. A genuine order-10069 twist
point was constructed to test the G2 subgroup check rather than assuming it works.
arkworks 0.5.0 sources were read to establish what arkworks does and does not check.

**Leakage** (blinders, randomness, side channels, residual data). The randomness path was
verified by reading the resolved dependencies under `~/.cargo/registry`, not the docs:
`Cargo.lock` pin, `ark-std`'s `pub use rand`, `ThreadRng`'s ChaCha12 core, its `CryptoRng`
impl, and the `pthread_atfork` reseed handler. The witness sparsity of every benchmark
artifact was measured with a throwaway parser. Three CLI runs were compared for proof
distinctness. The timing correlation itself was inferred from code structure plus the
published Zcash measurement, not re-measured here.

**GPU backend** (Metal, as the model for the CUDA port). All 1158 lines of `msm.metal`
read plus the host code that drives it. `cargo test --workspace --release --features
metal` run: green. The race-freedom argument, the addition-formula guards, the buffer
initialisation table and the window-sizing bound were established by reading, not by
instrumenting the GPU.

**Adversarial tests**, all executed. Roughly 2,000 proofs across six artifacts with fresh
blinders, 200 genuinely different statements on `tiny_mul` (not just different blinders),
144 proofs from 8 threads sharing one `PreparedCircuit`, 42 degenerate-blinder corners,
CPU versus Metal at pinned blinders, 40 single-bit coordinate flips per fixture, element
substitution and cross-proof splices, infinity in every position, negation of every
element, wrong public inputs of every shape, wrong-length witnesses through all three
entry points, and a constant-one wire set to 0, 2 and `r-1`.

**No test suite found a prover defect.** Every proof produced by an honest key and an
honest witness verified, under our verifier, and (for the shipped artifacts) under snarkjs
and rapidsnark. Every case that should reject, rejected, with the right error variant.
The findings below all come from the audits.

## Findings

Severity is real exploitability, not theoretical class. `[I]` inputs, `[S]` soundness,
`[L]` leakage, `[G]` GPU.

### Critical

**1. `[I]` A malicious zkey can silently destroy zero knowledge while proofs still verify.**
`binfile.rs:214-237` maps affine `(0,0)` to the identity, which is correct for
ffjavascript's encoding. But arkworks' `is_on_curve` and
`is_in_correct_subgroup_assuming_on_curve` both return true for the identity, so the six
header checks at `lib.rs:158-163` pass on an all-zero point. Zero `beta_g1`, `delta_g1`
and the whole of `b_g1_query` (section 6) and the blinding collapses: `pib1` becomes the
identity and the proof is an honest one at `r = 0`.

CONFIRMED by running: three proofs from the mutated `tiny_mul` key, each `verify: OK`
against the **untouched genuine `vkey.json`**, with `pi_a` bit-identical every time. None
of the three edited quantities appears in the verification key, so no verifier anywhere
can detect this: not ours, not snarkjs, not rapidsnark. `pi_a = alpha + sum w_j A_j` is
then a deterministic function of the full witness, which is a witness-confirmation oracle
and, for any low-entropy witness field, outright recovery.

Status: **open**. Fix is two O(1) gates: reject `is_zero()` on all six header points (they
are `[x]_1` and `[x]_2` for nonzero toxic waste and can never legitimately be infinity),
and reject an all-identity query section. Neither can false-positive on a real snarkjs key.

### High

**2. `[I]` `domainSize` allocation bomb: 34 GB from a 4 KB file.** `lib.rs:130-135` bounds
`domain_size` only to nonzero-power-of-two, so `2^31` survives. `lib.rs:255` and `:283`
then size four `u32` vectors of `domain_size + 1` from it. The check that refutes the lie,
`expect_records` on section 9, runs five statements later at `lib.rs:151`. CONFIRMED:
`2^24` costs 275 MB and 0.09 s, `2^29` costs 8.6 GB and 2.92 s, `2^31` costs
34,366,930,944 bytes and 11.49 s, all before erroring on section 9. Same class as
CVE-2024-50354 in gnark. Status: **open**. The fix is pure reordering plus one bound
(`domain_size <= 1 << Fr::TWO_ADICITY`), and it also closes finding 17.

**3. `[I]` Query sections 5 to 9 are never validated.** Deliberate, for load performance,
and documented at `lib.rs:153-157`. CONFIRMED: bit flips in `a_query[0]`, `b_g2_query[0]`
and `h_query[0]`, and `a_query[0] = (1,0)`, all **proved successfully** and emitted
off-curve `pi_a`, `pi_b`, `pi_c`. Section 3 (`ic`) is the control and fails correctly at
load. G2 has a large composite cofactor, so an on-curve off-subgroup base leaks witness
bits with no off-curve trickery, and it composes with finding 1. Status: **accepted risk
for a trusted key, open for proving-as-a-service**. The fix is a cached batched
random-linear-combination check, one MSM per section, behind
`ProvingKey::load_untrusted`, not a per-point check on the hot path.

**4. `[G]` No `MTLCommandBuffer` status or error is ever checked.** Six commit sites:
`msm.rs:636-637`, `msm.rs:746-747`, `stages.rs:527-528`, `:558-559`, `:567-568`,
`:585-586`. Exhaustive grep for `.status()`, `.error()` and `add_completed_handler`
returns zero hits. `waitUntilCompleted` returns normally on
`MTLCommandBufferStatusError`, and the next statement is `read_back` on a **pooled**
buffer, so a GPU fault yields the previous proof's window sums and a silently wrong proof
with no error anywhere. Status: **open**. Fix is one `submit_and_wait` helper returning
`ProveError`.

**5. `[L]` Metal kernels never clear threadgroup memory.** `ntt.metal:100,137,192` holds a
2^10-element slice of A, B or C, which is a direct linear image of the witness;
`msm.metal:697,1084,1094,1143,1154` likewise. This is the surface of CVE-2023-4969
LeftoverLocals (Trail of Bits, 16 January 2024): local memory readable across processes,
Apple affected, M3 and A17 fixed, M2 not fixed at disclosure. This machine is an M2 Max.
INFERRED, not tested: whether macOS 26.6 patches the M2 family. Status: **open**. Fix is
the vendor's own recommendation, volatile zero stores plus a barrier at kernel exit in
five kernels, with cost in the noise.

**6. `[L]` Witness-dependent MSM cost, and we serialise the measurement.**
`g16-msm/src/lib.rs:187,191` skip zero and one scalars; the survivor count `m` sizes the
window, the digit arrays and the Pippenger decision. Metal reproduces it: `msm.rs:594-608`
builds a witness-derived bitmap and `:788-792` sizes the GPU allocation and dispatch grid
from it. This is the Zcash channel from USENIX Security 2020 (correlation 0.57 over 4,000
Sapling proofs), except we also print `msm_us` at `main.rs:126` and write it to CSV at
`bench.rs:62,96`, so no microarchitectural attack is needed.

Calibrated by measurement on our own `.wtns` files: general scalars are 83.3% for
`tiny_mul`, then 95.9%, 97.2%, 98.2%, 98.2%, 98.2% up to `js_16x16_d32`. So the channel is
narrow on these circuits and enormous on the bit-decomposition circuits the optimisation
targets. Note that `g16-msm/src/lib.rs:12-13` claims "over 99% of witness scalars are 0 or
1", which is true of no artifact we ship, and the module docs justify the fast path at
about 5.1x while dropping it costs under 2% on these same artifacts. Those two numbers
should be reconciled.

Status: **documented, not mitigated** (commit `e04b9af` added the threat-model note to
`g16-msm`). The timing channel is an accepted risk. Serialising `msm_us` into artifacts an
operator might publish is **open** and is the cheap half of the fix.

### Medium

**7. `[G]` One unreproduced concurrency failure.**
`backend::tests::one_metal_circuit_proves_concurrently` failed once, then passed 150+
subsequent runs (30 sequential, 45 at 3x concurrency, 60 at 6x, plus cross-process
contention). The panic message was lost. The pool disciplines and the shared-`Plan` path
were eliminated by reading, all being read-only or mutex-guarded with `give` after
`wait_until_completed`. Status (2026-09-27): **reproduced and fixed** in `2e2d06f`. macOS killed one of the proof's
command buffers (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`), which `cb::wait_ok`
reported and the proving path did not retry; 10 of 20 runs failed while the machine was shared.
`compute_h` and the variable MSM batch now retry the whole submission (`cb::with_retry`), and
`a_retried_submission_gives_the_same_proof` pins that a retry is exact. Under a deliberate
three-process overload the GPU hung and recovered, and one unchecked proof came back wrong:
**open**, and `prove`'s self-verify is the guard for that case. Follow-up `7f34c54`: when the retries run
out, or wgpu loses the device, the error is `ProveError::Device`, and `g16 prove`'s fallback
retries it once on the device and then proves on the CPU instead of failing.

The same kill reaches the wgpu backend, and wgpu 30 does not report it: wgpu-hal's Metal fence
treats an errored command buffer as completed, so the readback held the previous proof's window
sums (29 of 40 CLI proofs wrong under load). Since `4ce2c96` every wgpu submission ends with a
token written by its last dispatch, and a readback without it is refused, retried up to four
times, then reported as `ProveError::Device`.

**8. `[S]` `proof.json` has no canonical encoding, and our two JSON readers disagree.**
`json.rs:130-146` and `:156-169` accept any nonzero Jacobian `z` and normalise, so there
are about 2^254 accepted encodings of every proof before any group-level malleability.
VERIFIED: five random `z` gave five distinct accepted encodings of one point. Worse, when
`z == 0` the reader returns at `json.rs:134-135` and `:159-160` **before** parsing `x` and
`y`, so `["<p>","<p>","0"]` and even `[{"lol":1},[1,2,3],"0"]` are accepted. Our own vkey
reader forbids exactly this (`g16-zkey/src/lib.rs:418,447` require `z` to be 1 or `[1,0]`).
Status: **open**. Fix: parse `x` and `y` before the zero test, require `z == 1` and
`z == [1,0]`, delete the now-dead Jacobian branch.

**9. `[S]` `verify()` performs no validation; every check lives in the CLI.**
`verify.rs:19-36` with `Proof`'s fields `pub` at `lib.rs:49-53`. VERIFIED by reading
`ark-ec-0.5.0/src/pairing.rs:104-109`: `multi_pairing` validates nothing. Any non-CLI
caller (library, FFI, fuzz harness, `ark-serialize` with `Validate::No`) gets an
unvalidated verifier. Off-curve and off-subgroup inputs happen to return `false` today,
but that is a BN254 accident, not an API guarantee, and the AGM soundness argument stops
applying once an off-subgroup element reaches the Miller loop. Status: **documented**
(commit `e04b9af`), **code fix open**. Validating at the top of `verify` costs about 100
microseconds against a 1 to 2 millisecond pairing.

**10. `[I]` `read_coefficients` reads the mmap twice, and pass 2 omits the range check.**
Pass 1 checks `constraint < domain_size` at `lib.rs:257-263`; pass 2 indexes
`cursor[m][constraint]` at `lib.rs:292` without rechecking. CONFIRMED with a live race
against the 94 MB key: 4 of 6 trials gave `rc=134` (SIGABRT) with
`index out of bounds: the len is 262145 but the index is 4294967295`, and `262145` is
`domain_size + 1`, which pins the panic to line 292. The quiet variant, an in-range flip
between passes, corrupts the CSR with no bounds check firing and no error. Status:
**open**. Fix: decode section 4 once into owned storage.

**11. `[I]`, `[S]` Field coordinates from the binary path are never range-checked.**
`binfile.rs:185-187` and `:202-204` are `new_unchecked`; only `fr_normal` for `.wtns`
checks. CONFIRMED: `alpha_g1` stored as `x + k*q`, `y + k*q` loads, passes `check_g1` and
**verifies OK** for k = 1, 2, 3 (k = 4 fails), so every coordinate has at least three
alternate byte encodings and hash-based key pinning is defeated. Also path-dependent: the
same `+q` on `a_query[0]` verifies while on `h_query[0]` it yields an off-curve `pi_c`
(mechanism inferred, not confirmed). Status: **open**. Fix is a comparison against
`MODULUS`, not `from_bigint`, since the stored value is already Montgomery.

**12. `[I]` SIGBUS on concurrent truncation of a mapped key, and it is not catchable.**
CONFIRMED against our binary: truncating at 0.02 s and 0.05 s gives `rc=138`, silently;
at 0.10 s and 0.20 s the load has finished and `rc=0`. Status: **accepted risk**, which is
the standard and unavoidable mmap caveat. Two properties keep the window to the duration of
`load()` and must be protected as invariants: `BinFile` is a local in `lib.rs:101` with
every section copied to owned `Vec`s, and no decoder ever forms a pointer into the mapping.
The mitigation is a README line about who may write the key file.

**13. `[L]` `n_public` is trusted from the `.zkey` and decides what we publish.**
`main.rs:92` reads it, `main.rs:116` writes `&w[1..=n_public]` to `public.json`. The only
validation is `n_vars >= n_public + 1`, so `n_public = n_vars - 1` publishes the entire
private witness to a file the user believes is public. The resulting proof would fail
against the honest vkey, but the values are already on disk by then. Status: **open**.
Fix: cross-check `ic.len() == n_public + 1` against an optional `--vkey`, and hard-error
when `n_public + 1 == n_vars`.

**14. `[L]` Metal scratch pools retain the previous proof's witness.**
`stages.rs:323-331,339,460-474,820-826` and `msm.rs:435-473`; the comment at
`msm.rs:980-981` already admits it. Everything is `StorageModeShared`, so these are host
pages rather than discrete VRAM, which makes this an un-zeroed-heap problem: it matters for
a resident proving service, not for a CLI that exits. Status: **open**. Fix: memset in
`Drop` and `Pool::give`, which preserves the first-touch page-fault rationale the pool
exists for and costs about 1 ms at 2^18.

**15. `[G]` Two bucket-clearing couplings that are correct only by accident.**
`msm_clear_impl` clears only `zz` (`msm.metal:864`), correct today solely because every
consumer tests `zz` first; three of four coordinates hold the previous proof's live points.
And the `G16_METAL_MSM_LEGACY_ACC=1` path dispatches no clear at all (`msm.rs:970-979`),
correct only because `msm_accumulate_impl` writes every row unconditionally
(`msm.metal:797`). Either invariant is one plausible optimisation away from a wrong point.
Status: **open**. Fix: dispatch the clear on both paths and pin the `zz`-first contract with
a test.

**16. `[G]` No test targets the exceptional addition cases.** The unit tests build bases as
successive generator multiples (`msm.rs:1213-1216`), so `P == Q`, `P == -Q` and a base at
infinity never reach `pt_madd` on purpose. They are hit incidentally by the artifact tests,
but bucket order comes from a nondeterministic scatter cursor (`msm.metal:755`), so
incidental coverage is not stable run to run, and the only all-artifact sweep
(`five_msms_match_cpu_on_every_artifact`, `msm.rs:1708`) is `#[ignore]`d. These guards are
the single most important correctness property in the backend. Status: **open**. This is
also the CUDA port's acceptance test.

### Low

17. `[I]` No `TWO_ADICITY` bound at parse time, so `2^29` costs 8.6 GB before `Domain::new`
    refuses it. Closed by finding 2's fix. **Open.**
18. `[I]` `Vec::with_capacity(n_sections)` reserves up to 103 GB of address space, measured
    at 6.6 MB RSS and 0.00 s. **Accepted risk** on 64-bit, one line to cap.
19. `[I]` `expect_records` uses unchecked `n * stride` and `binfile.rs:83` truncates `u64 as
    usize`. 32-bit only, not exploitable on any target we build. **Accepted risk**, written
    down so the assumption is explicit.
20. `[I]` Section 10 (phase-2 contributions) is never read, so zkey provenance is unchecked
    and `vkey.json` derived from the same zkey is circular. **Accepted risk**: the only
    non-circular check is `snarkjs zkey verify <r1cs> <ptau> <zkey>`, which is an operator
    obligation and belongs in the README.
21. `[S]` Infinity in a proof degrades the 4-pair check into a 3-pair one rather than
    failing (`ark-ec` drops zero pairs at `bn/mod.rs:55-65`). VERIFIED not exploitable:
    `A`, `B`, `C` and all-infinity each return `false`, and forging the degraded equation
    needs `alpha*beta/delta` in G1, which the CRS does not publish. **Open**, cheap to make
    structural instead of arithmetic.
22. `[S]` A degenerate `VerifyingKey` (all four pairs containing a zero element) makes
    `verify` accept everything, because the empty product is `Fq12::one()`. VERIFIED with a
    hand-built key. Unreachable through both shipped loaders, reachable because the fields
    are `pub`. **Open.**
23. `[S]` `aggregate_public` panics on an empty `ic` (`verify.rs:42` then `:50`). VERIFIED.
    Unreachable through both loaders, reachable for a library caller. **Open.**
24. `[S]` `multi_pairing` unwraps a fallible final exponentiation. INFERRED, never reached
    in a probe. Closed for free by finding 9. **Open.**
25. `[L]` No memory hygiene anywhere: grep for `zeroize|mlock|madvise|memset_s|
    explicit_bzero|setrlimit|RLIMIT_CORE|prctl` across `crates/` returns nothing, and
    `Cargo.toml:53` sets `panic = "abort"`, so a panic core-dumps the witness with no `Drop`
    running. The `.wtns` is mmap'd **and** copied to a `Vec<Fr>`, so it exists twice.
    **Open**, with the honest caveat that the on-disk `.wtns` is the larger leak and no
    in-memory fix touches it.
26. `[L]` `prove_with_blinders` is unrestricted public API (`prove.rs:26`). Nothing in the
    binary reaches it. Gating it is only half a fix: `CryptoRng` constrains the algorithm
    and never the seed, so `StdRng::from_seed([0;32])` reaches deterministic blinders
    through the safe entry point, which is exactly what our own tests do. **Open.**
27. `[G]` Four latent-fragility items: the spill tag is published before the spill point
    (safe only while the phases are separate dispatches), `f_neg(sentinel) == sentinel` is
    load-bearing and undocumented, pooled buffers are never scrubbed (this is finding 14
    seen from the GPU side), and the `stages.rs` scratch pool is uncapped. **Open.**
28. Design gap, not a defect: `prove`, `compute_h` and `msms` accept an in-memory witness
    with `w[0] != 1` and do the full NTT and MSM work before the caller discovers the proof
    is useless. The guard exists only in `Witness::load`, so an FFI or in-process integrator
    gets no protection. The proof does not verify either way, which is the property that
    matters. **Open**, one line in `CpuCircuit::check_witness`.

## What was cleared

These were suspected, checked, and are fine. Listed so nobody re-audits them.

**Blinder sampling is correct and better than the references.** VERIFIED by reading
`ark-ff-0.5.0/src/fields/models/fp/mod.rs:521-547`: masked rejection sampling, **exactly**
uniform on `[0, p)`, which is what keeps Groth16's zero knowledge perfect rather than
statistical. rapidsnark confines its blinders to `[0, 2^248)`; ours does not. `r = 0` and
`r = s` each happen with probability `2^-254` and neither should be rejected, because
rejecting would break the uniformity that the ZK proof depends on. Two independent draws
off the same stream, so no reuse.

**`thread_rng` is a CSPRNG here**, VERIFIED by reading the resolved dependencies rather
than the docs: `Cargo.lock` pins rand 0.8.8, `ark_std::rand` is `pub use rand`, the core is
`ChaCha12Core` reseeded from `OsRng` every 64 KiB, `impl CryptoRng for ThreadRng` is
present, and fork protection is real via `libc::pthread_atfork`. The feature wiring fails
to compile rather than silently downgrading. The one hand-rolled RNG in the tree
(`g16-msm/tests/field_cpu_throughput.rs:41`) does not implement `CryptoRng`.

**No RNG reuse in the bench loops.** `bench.rs:163,211` take one `thread_rng()` outside the
rep loop and pass `&mut rng`, which is the correct pattern because `ThreadRng` is an `Rc`
handle onto one thread-local stream. Empirically confirmed: three CLI runs on the same
witness gave three distinct proofs with identical `public.json`.

**No witness data in any error, log, or `Debug` impl.** VERIFIED exhaustively over every
`format!` and `#[error]` in all six crates: every message interpolates lengths, indices,
section ids and backend names only. No `Debug` on `HPoly`, `MsmOutputs`, `Witness`,
`ScalarBuf` or `Scratch`. The only non-test prints are names and `u64` counters. This is
cleaner than most provers. The one exception is finding 13.

**The G2 subgroup check is real and works.** A genuine order-10069 twist point was
constructed (confirmed `[10069]T == 0` and `2p - r = 0 mod 10069`), added to a real `pi_b`,
and `read_g2` rejected it at `json.rs:124`. The G1 subgroup call compiles to `true` and
that is correct, since BN254 G1 has cofactor 1; keep the call in case the curve changes.

**Public input handling is exact and stricter than both arkworks and snarkjs.**
`verify.rs:42-49` VERIFIED in both directions. Coordinates at or above the modulus are
rejected rather than reduced (`json.rs:40-42`), which is what defeats the `s + r` public
signal substitution. There is no `q - y` negation bug: `-proof.a` uses arkworks' flag-based
`Neg`. The vkey JSON reader (`g16-zkey/src/lib.rs:326-377`) is the strict one, and
`json.rs` should be raised to it rather than the reverse.

**The zkey container layer is genuinely solid.** Every length reaching a slice is
bounds-checked, section lengths use `checked_add`, there is exactly one `unsafe` in the
crate (`Mmap::map`), no `from_raw_parts`, no `transmute`, no `align_to`, and every integer
decode goes through `<[u8;N]>::try_into` so a crafted file cannot produce a misaligned
load. 39 malformed-container mutants all produced clean `Result` errors. **No mutant
produced memory unsafety.** This is a real advantage over rapidsnark, which type-puns
`*(u_int32_t*)(addr + pos)` at file-controlled offsets.

**The Metal bucket accumulation is race-free and the addition-formula guards are
complete.** Exclusive row ownership via a counting sort, not atomics: each row is
contiguous after the scatter, interior runs are privately owned, and boundary runs go to
per-thread spill slots. Atomics touch `uint` counters only. The double-count case was
checked term by term and cannot occur. All three exceptional cases are guarded (operand
identity, accumulator identity, `P == Q` to a doubling and `P == -Q` to zero), infinity is
read from the arkworks flag at the host pack rather than re-derived from coordinates, and
the conditional subtraction is branchless via `select()`. This backend does not have
lambdaworks' bucket race or zkmopro's dropped infinity flag.

**Groth16 malleability is not a bug and the negation case is not either.** A test lane's
first draft asserted that `(-A, -B, C)` must be rejected. It must not:
`e(-A, -B) = e(A, B)`, so this is the `z = -1` instance of rescaling and it is the cheapest
malleability there is, two field negations and no inversion. If a future change makes the
verifier reject it, that change is wrong.

**Structurally inapplicable, so nobody should chase them.** Frozen Heart needs a
Fiat-Shamir transform, and there is none anywhere in `prove.rs` or `verify.rs`; Groth16 is
non-interactive by trusted setup. Biased blinders are not a lattice attack here either:
`r` and `s` appear only in the exponent, so there is no hidden-number problem to build over.

## Zero knowledge: can this prover produce a proof that reveals things?

Yes, under four conditions. One of them is reachable today with an honest CLI invocation.

**On an honest key and an honest witness, the proof itself is fine.** The blinding is
correct, the blinders are uniform on the full scalar field from a CSPRNG, and stage 11 was
checked against the corners where a transcription error would cancel: 42 degenerate
`(r, s)` pairs across six variants all verify, including `r = s` (where the
`- delta_g1 * (r*s)` correction is most likely to be wrong) and `r = s = -1` (where that
cross term flips sign). Two proofs of the same witness differ in all three elements and
both verify. That is the property Groth16 asks for and we have it.

The four ways it breaks:

**1. A hostile `.zkey`. Reachable, confirmed, and invisible.** Finding 1. Zero
`beta_g1`, `delta_g1` and section 6 and the prover produces an unblinded proof that
verifies against the genuine verification key with a bit-identical `pi_a` every run. The
victim sees nothing. This is the one condition that is live today, needs no side channel,
no co-tenant and no library misuse, and it is why finding 1 is the top of the list.

**2. Observation of the proving process. Reachable given an observer.** Finding 6. Running
time, allocation size and GPU dispatch geometry are all functions of witness sparsity, and
we print the measurement. Zero knowledge is a property of the proof, not of the process
that produced it, and this prover is not constant time with respect to the witness. On a
shared or profiled machine that is a real channel, and it is worst on exactly the
bit-decomposition circuits the optimisation exists for.

**3. Residual data. Reachable for a resident service, not for the CLI.** Findings 5 and 14.
Pooled Metal buffers keep the previous proof's witness, its three domain vectors and both
copies of H in host-visible memory until a same-sized proof overwrites them, and threadgroup
memory is never cleared, which is the LeftoverLocals surface on an unpatched M2. The CLI
exits, so this is a service problem. There is also no `mlock`, no zeroing, and
`panic = "abort"` core-dumps the lot.

**4. Library misuse. Reachable for an integrator, not through our binary.** Findings 26 and
13. `prove_with_blinders` is public, and gating it does not close the hole because a seeded
`StdRng` satisfies `CryptoRng` and reaches the same place through the safe entry point.
Separately, `n_public` is trusted from the key, so a mismatched key writes private witness
values into `public.json`.

No correctness test can catch a broken stage 10. `prove.rs:355`
(`zero_blinders_still_verify`) asserts the opposite by design, and it is right to: a proof
with `r = s = 0` verifies under our verifier, under snarkjs and under rapidsnark. A ZK
break is silent by construction, which is why it gets audited rather than tested.

## Groth16 malleability

A Groth16 proof is not an identity, and this crate cannot make it one.

Anyone holding a valid `(A, B, C)` can produce a different, equally valid proof of the same
statement without knowing the witness. Pick any nonzero `z` in `Fr` and take
`(A z^-1, B z, C)`: by bilinearity `e(A z^-1, B z) = e(A, B)`, and the right side of the
pairing check never mentions `A` or `B`. VERIFIED on the real `js_2x2_d16` artifact,
accepted 10 out of 10, along with the full re-randomisation
`A' = z1^-1 A`, `B' = z1 B + z1 z2 [delta]_2`, `C' = C + z2 A`, which moves all three
elements and was also accepted 10 out of 10. Groth16 is weakly simulation-extractable
(randomisable, statement-non-malleable) and never strongly SE; see Baghery, Kohlweiss, Siim
and Volkhov, eprint 2020/811.

The encoding layer adds more non-uniqueness on top: about 2^254 accepted `proof.json` files
per proof today (finding 8). Fixing that reduces the encodings but does not touch the group
level, which is roughly `r` proofs per proof and is irreducible.

**What a downstream consumer must therefore not do.** Never derive an identity from proof
bytes. Concretely, a proof or a hash of one must never be used as a nullifier, a replay
key, a deduplication key, a transaction identity, a commit-reveal binding, an idempotency
token, or a cache key assumed collision-free per statement. Every one of those is defeated
by a single field inversion performed by anyone who has already seen a valid proof.

The canonical identity of a Groth16 proof is `(verifying key, public inputs)`. Anti-replay
belongs inside the public inputs, as a nullifier the circuit computes, or in application
state.

**Do not try to fix this with a uniqueness or canonicalisation check.** The attacker picks
`z`, so such a check only starts rejecting honest proofs. The property is pinned by
`malleated_proof_still_verifies` in `crates/g16-cli/tests/verifier_negative.rs` and
documented on `Proof` and `verify`, precisely so that it reads as intentional rather than as
a bug someone should go and repair.

## What we deliberately do not defend against

**The trusted setup.** Groth16 needs a per-circuit ceremony, and anyone holding the toxic
waste can forge proofs for any statement. This prover consumes the ceremony's output and
has no way to check it. Not our boundary.

**A malicious `.zkey`, by default.** We read the key as trusted input. Only the O(1) points
are validated (six header points plus `ic`); the query sections, which are millions of
points, are not, because a subgroup check each would dominate key load. That is a
performance decision and it is written down at `lib.rs:153-157`. It is also the boundary
findings 1, 3 and 11 sit on, and the boundary moves the moment the key and the witness have
different owners, which is exactly proving-as-a-service. The operator's obligation, which
belongs in the top-level README, is `snarkjs zkey verify <r1cs> <ptau> <zkey>`. That is the
only non-circular check: `snarkjs zkey export verificationkey` derives the vkey from the
zkey, so "it verifies against my vkey" proves nothing about an untrusted key. Findings 1
and 2 are still worth fixing because both are O(1) and both close a silent path.

**Concurrent mutation of a mapped key file.** SIGBUS is unavoidable while mmap is used, and
mmap is the right call for a 100 MB to multi-GB key. The window is the duration of
`load()`, which is milliseconds, and it stays that way only while `ProvingKey` owns every
byte. The mitigation is operational: the key must not be writable by anyone else while it
is being loaded.

**A compromised host.** If an attacker can read our process memory, attach a debugger, read
swap, or collect a core dump, they have the witness, and nothing in a prover changes that.
The findings under residual data are about narrowing the window for a *resident service*,
not about defending a lost host.

**The `.wtns` file on disk.** It is the witness, in the clear, written by whoever generated
it. No amount of in-memory hygiene touches it. We should say so rather than imply the
prover protects the witness.

**Constant-time proving.** We keep the zero and one MSM fast path, as every production
Groth16 prover does. Finding 6 is therefore an accepted risk with a documented threat model
rather than an open bug, and a deployment where an attacker can time the prover must treat
that as part of its own threat model.

**Witness generation.** Stage -1 is circom's, runs outside this codebase, and has
data-dependent control flow by construction.
