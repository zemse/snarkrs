# Audit: leakage (blinders, randomness, side channels, residual data)

Scope: `crates/g16-core/src/prove.rs`, `crates/g16-cli/src/main.rs`,
`crates/g16-cli/src/bench.rs`, every site that produces or consumes randomness, every
error/log/serialisation path that could carry witness data, and the Metal backend's
buffer lifetime. Companion to `security/notes/research-leakage.md`, which supplies the
theory; this file only reports what is in our tree.

Method. Claims tagged **VERIFIED** come from reading the exact file and line cited, from
reading the resolved dependency sources under
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`, or from running the release
binary and a throwaway parser in the scratchpad. Claims tagged **INFERRED** were not
executed. No repo file was modified. Machine: Apple M2 Max, macOS 26.6 (25G72).

---

## Direct answers to the five questions

### 1. Can `r` or `s` ever be zero, and does it matter?

Yes, with probability `1/p` each, and no, it does not matter.

`prove.rs:18-19` draws `r` and `s` with two independent `Fr::rand` calls. **VERIFIED** by
reading `ark-ff-0.5.0/src/fields/models/fp/mod.rs:521-547`: the `Standard` distribution
for `Fp` is masked rejection sampling. It fills the limbs from the RNG, masks off
`num_bits_to_shave()` high bits, and loops until `!tmp.is_geq_modulus()`. The result is
**exactly** uniform on `[0, p)`, not statistically close to it, which is what keeps
Groth16's zero knowledge *perfect* rather than statistical. Zero is one of the `p` values
it can return, at probability `2^-254`. There is no rejection of zero and there should
not be: rejecting it would make the distribution non-uniform, which is a worse bug than
the `2^-254` event it avoids. Rapidsnark, by contrast, zeroes `r` then fills
`sizeof(r) - 1` bytes, confining the blinder to `[0, 2^248)`; ours is strictly better.

`r == s` is likewise a `2^-254` accident, not a code path. The two draws are separate
statements against the same `&mut R`, so they consume disjoint segments of the stream.

The thing that *would* matter is a future edit that samples once and uses the value
twice, or that hard-codes a blinder. Research finding 3 shows `r == s` is a full break
(two pairings per candidate witness), and finding 5 shows `r == 0` is CVE-2024-45040 in
gnark. Note that **no correctness test in this repo can catch either**: `prove.rs:355-367`
(`zero_blinders_still_verify`) exists precisely to assert that `r = s = 0` still
verifies, under our verifier, and the same proof would pass snarkjs and rapidsnark.
`distinct_blinders_give_distinct_proofs_that_both_verify` (prove.rs:333-351) is the
closest guard, and it only proves the two proofs differ, which `r = s` also satisfies.

**Verdict: fine as written. No change needed. See L6 for the guard that is missing.**

### 2. Is there a path in the shipped binary to `prove_with_blinders`, or an RNG reuse that repeats `(r, s)`?

No to both. **VERIFIED**.

`grep -rn prove_with_blinders crates/` returns call sites only in `prove.rs:20` (from
`prove` itself), `prove.rs:321,322,364` (`#[cfg(test)]`), `g16-metal/src/backend.rs:462,
485,516` (`#[cfg(test)]`), and `g16-metal/tests/*.rs`. Nothing in `g16-cli` references it.
The binary's only entry is `main.rs:104`, which calls `prove` with
`ark_std::rand::thread_rng()` from `main.rs:101`. There is no `--seed` flag and no env
override anywhere in `Cli`.

**The bench loops are clean.** `bench.rs:163` (cold) and `bench.rs:211` (warm) each call
`thread_rng()` once, outside the rep loop, and pass `&mut rng` into every `prove`. That is
the *correct* pattern, not a reuse bug: `ThreadRng` is a handle (`Rc<UnsafeCell<...>>`,
`rand-0.8.8/src/rngs/thread.rs:63-66`) onto one thread-local `ReseedingRng`, so all three
handles in the process are the same stream and every `Fr::rand` advances it. Two reps can
only collide on `(r, s)` with probability `2^-508`.

Empirically confirmed: three `g16 prove` runs on `bench/artifacts/js_1x1_d8` produced
three distinct `proof.json` files (md5 `1055fa18…`, `c2985aa1…`, `68076fda…`) with
byte-identical `public.json` (`f16917b0…`). Same witness, same key, three different
proofs, which is exactly what a working stage 10 looks like.

One thing the bench does that is worth naming rather than flagging: warm mode proves the
**same witness** `--reps` times (default 15) and the CSV records all 15. That is
harmless. Groth16 ZK is closed under repetition with fresh `(r, s)`; it is only reuse
that breaks it.

### 3. Is `thread_rng` a CSPRNG in the version this repo pins?

Yes. **VERIFIED by reading the resolved dependency, not the docs.**

* `Cargo.lock:779-782` pins `rand 0.8.8`; `Cargo.lock:790-793` `rand_chacha 0.3.1`;
  `Cargo.lock:800-803` `rand_core 0.6.4`; `Cargo.lock:194-196` `ark-std 0.5.0`.
* `ark-std-0.5.0/src/rand_helper.rs:9` is `pub use rand;`, so `ark_std::rand` *is* the
  `rand` crate. No wrapper, no re-implementation.
* `rand-0.8.8/src/rngs/thread.rs:63-66`: `ThreadRng` wraps
  `ReseedingRng<Core, OsRng>`, and `rand-0.8.8/src/rngs/std.rs:13` is
  `pub(crate) use rand_chacha::ChaCha12Core as Core`. So it is **ChaCha12 seeded from
  `OsRng`**, reseeded from `OsRng` every 64 KiB (`thread.rs:38`).
* `rand-0.8.8/src/rngs/thread.rs:130`: `impl CryptoRng for ThreadRng {}`.
* Fork protection is real, not just documented:
  `rand-0.8.8/src/rngs/adapter/reseeding.rs:309-320` calls `libc::pthread_atfork` with
  the handler on all three slots, bumping `RESEEDING_RNG_FORK_COUNTER`, and
  `reseeding.rs:195` registers it. This is the protection snarkjs' memoised module-scope
  ChaCha lacks (research finding 8).
* The feature wiring is correct and deliberate: `crates/g16-cli/Cargo.toml` requests
  `ark-std = { workspace = true, features = ["std", "getrandom"] }` where the workspace
  default is `default-features = false`. `ark-std-0.5.0/Cargo.toml` defines
  `getrandom = ["rand/std"]`, and `rand/std` is what gates `thread_rng` at all. Without
  that line the CLI would not compile rather than silently fall back, which is the right
  failure mode. The comment at `main.rs:98-100` and `g16-cli/Cargo.toml` already says so.

One residual, unfixable by us and worth understanding: the `CryptoRng` bound at
`prove.rs:9` is a **marker trait about the algorithm, not about the seed**.
`StdRng::from_seed([0u8; 32])` satisfies it, which is exactly what the tests at
`prove.rs:220, 245, 277, 303, 340-341` do. A downstream caller can therefore get
deterministic, repeatable `(r, s)` through the *safe* entry point without ever touching
`prove_with_blinders`. This is a property of `rand`'s trait design, not a bug here, but it
does mean gating `prove_with_blinders` is at most half a fix. See L6.

Cleanliness check on the other side: the only hand-rolled RNG in the tree,
`SplitMix64` in `crates/g16-msm/tests/field_cpu_throughput.rs:41-58`, implements
`RngCore` only and **not** `CryptoRng`. **VERIFIED** by grep: `CryptoRng` appears in
exactly two places in `crates/`, both on `prove.rs:9` and its comment. Nobody has
back-doored the bound.

### 4. Does any error path, log, debug print, or serialised output expose witness values or intermediate polynomials?

No. **VERIFIED** by reading every `format!` and `#[error]` in `g16-core/src`,
`g16-zkey/src`, `g16-metal/src`, `g16-msm/src`, `g16-ntt/src`, `g16-field/src`.

* Every error message interpolates **lengths, indices, section ids, domain sizes and
  backend names only**. Representative: `g16-core/src/lib.rs:37` (`{got}`/`{want}` are
  counts), `cpu.rs:237-241` and `cpu.rs:252-257` (lengths), `g16-zkey/src/lib.rs:261`
  (`constraint {constraint} is outside domain size`), `binfile.rs:92` (byte lengths). Not
  one of them formats an `Fr` or a `Vec<Fr>`.
* No `Debug` derive lands on a witness-bearing type. `HPoly` (`lib.rs:68-79`),
  `MsmOutputs` (`lib.rs:115-121`), `Witness` (`g16-zkey/src/wtns.rs:13`),
  `ScalarBuf` (`g16-metal/src/msm.rs:330-334`) and `Scratch`
  (`g16-metal/src/stages.rs:323-331`) have **no** `Debug`. The three `Debug`s that do
  exist are on `ProveError` (`lib.rs:35`), `Proof` (`lib.rs:48`, public data by
  definition) and `StageTimings` (`lib.rs:125`, counters).
* No `println!`/`eprintln!`/`dbg!` in a non-test code path prints a field element. The
  complete non-test set is `main.rs:121-132` (backend name, domain size, five `u64`
  microsecond counters), `main.rs:147` (`"OK"`), `bench.rs:119,142,278,296` (variant
  name, counts, timings). `eprintln!` appears only inside `#[cfg(test)]` harnesses.
* `json::write_public` (`main.rs:116`) writes `&w[1..=n_public]`. That is by design, but
  the bound comes from an untrusted file. See L3.

**Verdict: clean, apart from L3. This is better than most provers.**

### 5. Does the Metal backend leave witness data in GPU buffers that outlive the proof, and does it matter?

Yes it does, and it matters in one specific way that has a CVE attached, plus one
ordinary way that does not.

**The ordinary way (L4, medium).** Per-proof scratch is **pooled, not freed, and never
zeroed**. `stages.rs:323-331` defines `Scratch { witness, a, b, c, t, h_mont, h_std }`,
`stages.rs:339` types the pool as `Arc<Mutex<Vec<Scratch>>>`, `stages.rs:460-474` pops
from it or allocates, and `stages.rs:820-826` (`impl Drop for HHandle`) pushes the whole
set back when the caller drops the `HPoly`. So after a proof returns, a live
`MetalCircuit` still holds: the packed witness, the three domain vectors A, B and C
(which are `w · A`, `w · B`, `w · C`, a linear function of the witness), and both copies
of H. The MSM side does the same at `msm.rs:435-473`: `Pool::take` reuses the smallest
buffer that fits and `Pool::give` retains up to 48 of them. The code is aware of this:
`msm.rs:980-981` says outright "A pooled bucket array holds the previous proof's points".
The witness `ScalarBuf` at `backend.rs:283` is *not* pooled and is dropped per proof, but
dropping a `MTLBuffer` does not scrub it either.

How much this matters: every buffer here is `MTLResourceOptions::StorageModeShared`
(**VERIFIED**: that is the only storage mode used anywhere in `g16-metal/src`). On Apple
silicon that is unified memory, so these are ordinary host pages. There is no discrete
VRAM holding a copy, and the pages are zeroed by the kernel before another process gets
them. So this is the same threat model as an un-zeroed `Vec<Fr>`: it matters for a
long-lived server process (heap inspection, a core dump, swap), not for the CLI, which
exits immediately.

**The way with a CVE (L2, high).** Threadgroup (local) memory. `ntt.metal:100, 137, 192`
declare `threadgroup Fr* sh`, and `stages.rs:36-47` and `stages.rs:185-193` size that
slice at the full 32768-byte threadgroup budget, 2^10 field elements. Each NTT
threadgroup pulls a slice of A, B or C into local memory, runs several passes there, and
exits **without clearing it**. Same in the MSM: `msm.metal:697`
(`threadgroup uint tmp[SCAN_TG]`, the digit prefix scan) and `msm.metal:1084, 1094, 1143,
1154` (`threadgroup PtG1/PtG2 shared[REDUCE_TG]`).

That is exactly the surface of **LeftoverLocals, CVE-2023-4969** (Trail of Bits,
16 Jan 2024). Per their write-up: the vulnerability "allows attackers to recover data
from GPU local memory created by another process", it is specifically **local/shared
memory, not global**, Apple, AMD, Qualcomm and Imagination were affected while NVIDIA,
Arm and Intel were not, A17 and M3 contain fixes, and the MacBook Air **M2 remained
vulnerable** at publication. Their developer-side mitigation, for exactly the case where
you cannot rely on a driver patch, is to "modify the source code of all GPU kernels that
use local memory" to store zeros into it before the kernel completes, marking the stores
`volatile` so the compiler cannot elide them.

This machine is an M2 Max. **INFERRED, not verified: I did not test whether macOS 26.6
patches the M2 family.** Apple shipped fixes over 2024 and it may well be closed here.
But the kernels ship to whatever machine runs them, our threadgroup arrays hold a direct
linear function of the witness, and the mitigation is a handful of zero stores at the end
of three kernels. The cost/benefit is not close.

---

## Findings, ranked by real exploitability

### L1. Witness-dependent MSM cost, and we serialise the measurement (high as a library, medium as this CLI)

**What.** The CPU prescan skips scalars that are zero or one:
`crates/g16-msm/src/lib.rs:187` (`if s.is_zero() { continue }`) and `:191`
(`if s.is_one() { ones_sum += &bases[i]; continue }`). The surviving count `m` then sizes
the window (`:298 let c = window_size(m)`), the digit arrays, and the whole Pippenger
decision (`:292-294` skips Pippenger entirely when `m == 0`). `idx` and `bigints`
(`:182-183`) are witness-length-dependent allocations. The Metal path reproduces it
exactly: `msm.rs:594-608` builds `general_prefix: Vec<u32>`, a witness-derived bitmap;
`msm.rs:347-350` (`general_in`) reads it; `msm.rs:788-792` sets
`cap = general.max(1)`, so the **GPU entry-array allocation and dispatch grid are sized
by the witness**.

This is the Zcash channel verbatim. USENIX Security 2020 (Tramèr, Boneh, Paterson)
measured R = 0.57 between proving time and non-zero witness entries over 4,000 Sapling
proofs and concluded the implementation is "not zero-knowledge in practice".

**We make it worse than Zcash did in one respect: we publish the measurement.**
`main.rs:126` prints `msm_us` under `--stage-timings`, and `bench.rs:62` puts
`msm_us` in `CSV_HEADER` with `bench.rs:96` writing it per rep. An operator who ships a
benchmark CSV alongside proofs hands over a witness statistic in a machine-readable
column. No microarchitectural attack required.

**Why it matters, calibrated.** I measured the actual skip fraction on our own artifacts
(scratchpad parser over `.wtns` section 2):

| variant | n | zero | one | general | general % |
| --- | --- | --- | --- | --- | --- |
| tiny_mul | 6 | 0 | 1 | 5 | 83.3 |
| js_1x1_d8 | 3,373 | 125 | 14 | 3,234 | 95.9 |
| js_2x2_d16 | 10,194 | 258 | 30 | 9,906 | 97.2 |
| js_2x2_d32 | 18,002 | 290 | 30 | 17,682 | 98.2 |
| js_8x8_d32 | 70,640 | 1,163 | 105 | 69,372 | 98.2 |
| js_16x16_d32 | 140,824 | 2,316 | 216 | 138,292 | 98.2 |

So on these circuits only about 1.8% of scalars take the fast path, and the leaked
quantity has a narrow dynamic range. Worth stating plainly, because
`g16-msm/src/lib.rs:12-13` claims "In bit-decomposition-heavy circuits over 99% of
witness scalars are 0 or 1" and **none of our benchmark circuits are anywhere near
that**. The claim is presumably true of the circuits it was written for; it is not true
of what we benchmark. The channel is therefore small here and enormous for the
bit-decomposition circuits the optimisation targets, which is the uncomfortable
direction: the fast path leaks most exactly where it pays most.

**INFERRED, not measured here:** I did not run a controlled witness-varying timing
experiment. The correlation is asserted from the code structure plus the cited USENIX
measurement, not reproduced on this prover.

**Fix.** Three tiers, pick by threat model.

1. *Cheapest, do it regardless.* Stop serialising the signal by default. Gate `msm_us`
   in the `--stage-timings` output and the CSV behind an explicit
   `--i-am-benchmarking-not-proving` style flag, or emit it only when the witness came
   from a benchmark artifact. Costs nothing, removes the no-effort exploit path.
2. *Cheap and effective.* Keep the prescan but make the **shape** witness-independent:
   size `idx`, `bigints`, the window `c`, the Metal `cap` and the dispatch grid from `n`
   (or from a padded bucket of `n` rounded up to a power of two), not from `m`. Keep the
   `is_zero`/`is_one` early-out inside the loop for speed if you like; what leaks is the
   allocation and the dispatch dimension, which is what a co-tenant or a profiler reads.
   You give up the "2^20-long witness with 2^10 general scalars is a 2^10 problem" win.
3. *Full fix.* Drop the zero/one special cases in the proving path entirely and do the
   constant-time thing: every scalar touches a bucket. Given the 98% general fraction
   measured above, on our own artifacts this costs under 2%.

Document whichever you choose in `g16-msm/src/lib.rs`'s module docs next to the existing
performance rationale, because a future reader will otherwise re-add the optimisation.

### L2. Metal kernels never clear threadgroup memory (high)

**What.** `crates/g16-metal/src/shaders/ntt.metal:100, 137, 192` (`threadgroup Fr* sh`,
holding a 2^10-element slice of the A/B/C domain vectors),
`crates/g16-metal/src/shaders/msm.metal:697` (`threadgroup uint tmp[SCAN_TG]`),
`msm.metal:1084, 1094, 1143, 1154` (`threadgroup PtG1/PtG2 shared[REDUCE_TG]`). None of
them zero their local arrays before returning.

**Why it matters.** CVE-2023-4969 / LeftoverLocals: a co-resident process can dispatch a
kernel that reads uninitialised local memory and recover the previous kernel's contents.
Apple was among the affected vendors; M3 and A17 are fixed, M2 was not at disclosure.
The A/B/C slices in `sh` are `w · A`, `w · B`, `w · C` evaluated on the domain, a direct
linear image of the witness. Recovering enough of them, with the QAP matrices from the
public `.zkey`, is a linear system in the witness.

**Fix.** Trail of Bits' own recommendation: at the end of each kernel that uses
threadgroup memory, have every thread write zeros over its slice, mark the writes
`volatile` so the MSL compiler cannot eliminate them, and `threadgroup_barrier` before
returning. Five kernels, a few lines each, and it runs once per threadgroup rather than
once per element pass, so the cost is in the noise against the NTT butterflies. Add a
comment citing the CVE so nobody optimises it back out.

Reference: <https://blog.trailofbits.com/2024/01/16/leftoverlocals-listening-to-llm-responses-through-leaked-gpu-local-memory/>

### L3. `n_public` is trusted from the `.zkey`, and it decides what we publish (medium)

**What.** `main.rs:92` takes `let n_public = pk.n_public;` straight from the proving key,
and `main.rs:116` writes `&w[1..=n_public]` to the user's `public.json`. The only
validation is `g16-zkey/src/lib.rs:139-142`, which rejects a key where
`n_vars < n_public + 1`, and `main.rs:110-114`, which repeats the same length check. So
`n_public = n_vars - 1` is accepted, and the CLI will happily write **the entire private
witness** into a file the user believes contains public signals.

**Why it matters.** It needs a hostile or simply mismatched `.zkey`, and the resulting
proof would be rejected by a verifier holding the honest `vkey`. But the leak has already
happened by then: the private values are on disk, in the file the user is about to
publish. "The proof failed to verify" is not a warning that you just published your
witness. Every existing test passes, because they all use self-consistent artifacts.

**Fix.** Two options, ideally both.

* Cross-check `n_public` against a second source before writing. The natural one is the
  vkey: `VerifyingKey` carries `ic`, and `ic.len() == n_public + 1` is a hard identity.
  Add an optional `--vkey` to `prove` and require the match when it is supplied; refuse
  to write `public.json` on mismatch.
* Refuse outright when `n_public` is implausible relative to the key, for example
  `n_public + 1 == n_vars` (a circuit with zero private wires proves nothing and is
  almost certainly an attack or a corrupt file). Emit a hard error, not a warning.

This overlaps `security/notes/audit-inputs.md`; it is listed here because the *impact* is
a confidentiality loss, not a soundness one.

### L4. Metal scratch pools retain the previous proof's witness in host-visible memory (medium)

**What.** `stages.rs:323-331`, `:339`, `:460-474`, `:820-826`; `msm.rs:435-473`,
`:980-981`. Detailed in answer 5 above. `StorageModeShared` throughout, so these are host
pages, not discrete VRAM.

**Why it matters.** For a resident proving service (which is the whole point of
`prepare`, per `lib.rs:137-141`), the witness of every proof ever computed is one
`memcpy` away in the process heap until the pooled buffer happens to be reused by a proof
of at least the same size. A smaller subsequent circuit does not overwrite the tail. A
core dump, a heap dump, or a swap file captures all of it. For the CLI, which exits, this
is close to irrelevant.

**Fix.** Zero on return, not on take. In `impl Drop for HHandle` (`stages.rs:820-826`) and
in `Pool::give` (`msm.rs:463-472`), memset the buffer contents through the shared
`contents()` pointer before pushing back. The pool's own justification is first-touch page
faulting (`msm.rs:430-434`: 15.2 GB/s cold against 54.8 GB/s warm), and a memset keeps the
pages resident, so this preserves the entire reason the pool exists. It costs one
sequential write over about 48 MB at 2^18, roughly 1 ms, on a path that already pays a
0.16 ms submission floor per command buffer. Make it a runtime option if that 1 ms is
load-bearing, but default it on.

### L5. No memory hygiene anywhere, and `panic = "abort"` (medium-low)

**What.** `grep -rn 'zeroize|mlock|madvise|MADV_|memset_s|explicit_bzero|setrlimit|RLIMIT_CORE|prctl' crates/`
returns **nothing** (exit 1). **VERIFIED.** Nothing is scrubbed, nothing is `mlock`ed, core
dumps are not suppressed. `Cargo.toml:53` sets `panic = "abort"` for the release profile,
so a panic in the prover produces a core dump (subject to system settings) with the full
witness, the three domain vectors and H in it, and no `Drop` runs first.

The `.wtns` is additionally **mmap'd**, not just read: `g16-zkey/src/binfile.rs:53`
(`unsafe { Mmap::map(&file)? }`), and `wtns.rs:29-34` then materialises a second copy as a
`Vec<Fr>`. So the witness exists twice in the address space, once as file-backed pages
that the kernel may write back.

**Why it matters.** This is the lowest-glamour and highest-certainty item on the list: it
is not an attack, it is the absence of defence in depth, and the 2024 SNARK vulnerability
survey (arXiv:2402.15293) notes ZK breaks are essentially undetectable after the fact.

**Fix, honestly scoped.** Do not promise more than is achievable.

* The `.wtns` on disk is the largest leak and no in-memory fix touches it. Say so in the
  README rather than implying the prover protects the witness.
* `Vec<Fr>` cannot be reliably zeroed if it reallocates. `wtns.rs:31-34` collects through
  rayon, which pre-sizes, but that is an implementation detail of `IndexedParallelIterator`
  and not a contract. If you want a scrubbed witness, reserve capacity explicitly first
  and wrap it in a type whose `Drop` memsets, and accept that it is best-effort.
* `mlock` on the witness buffer is cheap and stops it reaching swap. `RLIMIT_CORE = 0`
  (or a `--no-core-dump` flag) is a two-line change and removes the `panic = "abort"`
  exposure entirely.
* Leave the `.zkey` mmap alone. It is public by construction and the toxic waste was
  destroyed at the ceremony; there is nothing to protect.

### L6. `prove_with_blinders` is unrestricted public API (low, and only half a fix)

**What.** `prove.rs:26` is `pub fn` inside `pub mod prove` (`lib.rs:29`), with no
`#[cfg(test)]`, no `#[doc(hidden)]` and no feature gate. The doc comment at
`prove.rs:22-25` says "Used only by tests" and "Never use in production", which is
documentation, not an API boundary.

**Why it matters, and why it is only low.** Nothing in our binary reaches it (answer 2),
so this is a hazard for a future downstream consumer of `g16-core`, not a live exposure.
And the gate would not close the hole, because `prove()` itself accepts
`StdRng::from_seed(...)`: `CryptoRng` constrains the algorithm, never the seed. Our own
tests at `prove.rs:220, 245, 277, 303, 340-341` do exactly that.

**Fix.** Cheap, so do it, but do not oversell it.

* Put `prove_with_blinders` behind a `#[cfg(any(test, feature = "test-blinders"))]` or at
  minimum `#[doc(hidden)]` plus a name that reads as dangerous
  (`prove_with_blinders_insecure`). The Metal integration tests under
  `crates/g16-metal/tests/` need it, so a non-default feature is the right shape, not
  `cfg(test)`.
* Separately, if the library will ever be embedded, add a `prove_from_os_rng` that takes
  no RNG at all and sources `thread_rng()` internally. That is the entry point a caller
  cannot get wrong, and it is the one a service should be told to use.
* Do **not** try to detect a bad RNG at runtime. There is nothing to detect.

### L7. Things I checked and found fine (informational)

Stated explicitly so nobody re-audits them.

* **Stage 10 sampling.** Two independent draws, exact uniformity via masked rejection
  sampling, ChaCha12 reseeded from `OsRng` with `pthread_atfork` protection. Better than
  rapidsnark (248-bit blinders, `std::random_device`) and better than snarkjs (memoised
  module-scope ChaCha, no fork protection). **VERIFIED** end to end.
* **Bench RNG handling.** Correct. One thread-local stream, three handles, advanced by
  every draw. **VERIFIED** by reading and by three distinct proofs from three runs.
* **Error and log surfaces.** No witness value, no polynomial coefficient, no field
  element in any error string or non-test print. **VERIFIED** by exhaustive grep over
  `format!` and `#[error]`.
* **`Debug` derives.** None on a witness-bearing type. **VERIFIED.**
* **`CryptoRng` bound integrity.** The one hand-rolled RNG in the tree does not implement
  it. **VERIFIED.**
* **Frozen Heart.** Structurally inapplicable. There is no Fiat-Shamir transform anywhere
  in `prove.rs` or `verify.rs`; Groth16 is non-interactive by trusted setup, not by
  transcript hashing. Nothing to check.
* **Biased blinders as a lattice attack.** Not applicable. `r` and `s` appear only in the
  exponent (`prove.rs:44-47`), never in a scalar equation the attacker can read, so there
  is no hidden-number problem to build a lattice over. Recorded so nobody chases it.
* **The `.zkey` mmap.** Fine. Public data.

---

## What CUDA must do differently

Written for the lane currently building `crates/g16-cuda`. I did not read that crate (lane
discipline), so this is a checklist, not a review.

1. **Never generate blinders on the device.** `r` and `s` stay on the host, from the
   `RngCore + CryptoRng` that `g16_core::prove` already owns. Do not call cuRAND for them.
   cuRAND's default generators (XORWOW, MRG32k3a, Philox) are **not** CSPRNGs, they are
   simulation PRNGs, and a device-side blinder is also unauditable from the host. The
   backend trait does not ask for randomness and must not acquire any: `PreparedCircuit`
   (`g16-core/src/lib.rs:146-165`) exposes only `compute_h` and `msms`. Keep it that way.
2. **LeftoverLocals does not apply, but the discipline still should.** Trail of Bits found
   NVIDIA **not** affected by CVE-2023-4969. So `__shared__` memory is not the same
   liability there that Metal's `threadgroup` memory is on M2. Zeroing shared memory at
   kernel exit is still cheap insurance and costs nothing measurable; do it if the CUDA
   kernels stage witness-derived data through shared memory (a shared-memory NTT
   certainly will).
3. **Device global memory is a real difference from Metal.** On Apple silicon our buffers
   are host pages the kernel scrubs before another process sees them. On a discrete NVIDIA
   card the witness lives in VRAM, and a pooled allocator that reuses a `cudaMalloc`
   region inside one process hands the previous proof's contents to the next caller.
   `cudaFree` does not zero. If the CUDA backend copies the CPU/Metal pooling design
   (`stages.rs:460-474`, `msm.rs:435-473`) it inherits L4 with a longer-lived and less
   observable backing store. `cudaMemsetAsync` on release, not on acquire.
4. **Pinned host memory cuts both ways.** `cudaHostAlloc` pages are page-locked, so they
   cannot reach swap, which is a hygiene win over our current mmap plus `Vec`. They are
   also never zeroed on free and are not returned to the OS promptly. Memset before
   `cudaFreeHost`.
5. **The L1 channel is worse on CUDA, not better.** If the CUDA MSM replicates the CPU
   prescan (`g16-msm/src/lib.rs:177-204`) or the Metal `general_prefix`
   (`msm.rs:594-608`), the witness-dependent quantity becomes a **grid dimension**, which
   is visible in Nsight, in `nvidia-smi` utilisation sampling, and to any co-tenant on a
   shared or MPS-partitioned GPU. That is a far more accessible observation point than a
   wall-clock measurement on a laptop CLI. Size the grid from `n`, do the zero-scalar work
   anyway, and take the sub-2% hit measured in L1.
6. **No `printf` in kernels on the proving path.** Device-side `printf` writes to a host
   buffer that the driver flushes on sync, and a debug print of a scalar or a bucket index
   is a witness leak into stdout. Gate any such print behind a non-default feature.
7. **Do not stash the H buffer on the circuit struct.** `HPoly::Device`
   (`g16-core/src/lib.rs:68-79`) carries the handle through the *value* specifically so a
   `&self` circuit stays safe for concurrent proofs (`lib.rs:63-67`). A device global
   holding "the last H" would both race and retain witness data across proofs. The Metal
   backend gets this right via `HHandle` ownership (`stages.rs:757-826`); copy that shape.
8. **Do not add a seed flag to the CLI for CUDA determinism.** If reproducible CUDA runs
   are needed for debugging, use the same `prove_with_blinders` test-only entry the Metal
   tests use (`crates/g16-metal/tests/audit_metal_vs_cpu.rs:113`), behind the feature gate
   proposed in L6. A `--seed` on the shipped binary is a one-line path to research finding
   2's byte-identical-proof tier.
