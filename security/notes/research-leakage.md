# Leakage: how a Groth16 prover can reveal the witness

Research note, dimension "leakage". Answers the question *"could it generate a proof that
reveals things?"*

Every claim below is tagged:

* **[VERIFIED-CODE]** — I read the relevant source in this repo (or in snarkjs / rapidsnark /
  rand / ark-ff, at the commit/version noted) and am reporting what it actually does.
* **[SOURCE]** — read from a paper, advisory, or blog post; cited inline.
* **[DERIVED]** — algebra I worked out here. Every step is written out so it can be checked.

This note is **report-only**. No source file was modified.

---

## 0. Executive answer

Yes, a Groth16 prover can leak the witness, and the failure is not exotic. There are three
distinct families:

1. **Blinder failure** (`r`, `s` zero / equal / reused / predictable / low-entropy). This is
   the catastrophic one. Groth16's zero-knowledge is *perfect and unconditional* — it rests on
   no hardness assumption at all — but *only* because `r` and `s` are uniform and independent.
   Break that and you do not degrade to "computationally ZK"; for the enumerable-witness case
   you degrade to **no hiding whatsoever**, because `A` and `B` become *binding, non-hiding*
   Pedersen-style commitments to the witness under bases that ship in the public `.zkey`.
2. **Side channels** (timing, memory-access pattern, allocation size). Already demonstrated
   against a production Groth16 deployment (Zcash, USENIX Security 2020) with the *exact*
   optimisation this repo implements: skipping MSM terms whose scalar is zero.
3. **Hygiene** (witness on disk, witness in un-zeroed heap, core dumps, an untrusted `.zkey`
   choosing what counts as "public").

The most valuable single external data point: of 141 SNARK vulnerabilities catalogued in the
2024 survey *"What don't we know? Understanding Security Vulnerabilities in SNARKs"*
(arXiv:2402.15293), only **3** broke zero-knowledge — and **all 3 were in the backend layer**,
i.e. the prover, not the circuit **[SOURCE: arXiv:2402.15293v1, Table 3]**. ZK-breaking bugs are
rare, and they live precisely where this codebase is.

---

## 1. Notation and the exact algebra of the blinders

Working in the exponent (`[x]_1` = `x·G1`, `[x]_2` = `x·G2`), Groth16's prover is
**[SOURCE: Groth, "On the Size of Pairing-based Non-interactive Arguments", eprint 2016/260,
§3.2 `Prove`]**:

```
A = α + Σ_{i=0..m} a_i·u_i(x) + r·δ            in G1
B = β + Σ_{i=0..m} a_i·v_i(x) + s·δ            in G2
C = ( Σ_{i=ℓ+1..m} a_i(β·u_i(x) + α·v_i(x) + w_i(x)) + h(x)t(x) ) / δ
    + A·s + B·r − r·s·δ                        in G1
```

Define the **unblinded parts**:

```
A₀ := α + Σ a_i·u_i(x)          (a G1 point the attacker can recompute from a witness guess)
B₀ := β + Σ a_i·v_i(x)          (same, in G2)
L  := ( Σ_{i>ℓ} a_i(β u_i + α v_i + w_i) + h·t ) / δ
```

so `A = A₀ + rδ`, `B = B₀ + sδ`, and expanding `C` **[DERIVED]**:

```
C = L + s(A₀ + rδ) + r(B₀ + sδ) − rsδ = L + s·A₀ + r·B₀ + r·s·δ
```

This matches `crates/g16-core/src/prove.rs` line-for-line **[VERIFIED-CODE]**:

```rust
let pi_a = m.a_g1 + pk.alpha_g1 + pk.delta_g1 * r;
let pi_b = m.b_g2 + pk.beta_g2  + pk.delta_g2 * s;
let pib1 = m.b_g1 + pk.beta_g1  + pk.delta_g1 * s;
let pi_c = m.l_g1 + m.h_g1 + pi_a * s + pib1 * r - pk.delta_g1 * (r * s);
```

### What the attacker knows

Everything except `(a_i)` and `(r, s)`. Specifically the `.zkey` is public, and it contains:

* `{ u_i(x)·G1 }` — section 5, the A-MSM bases **[VERIFIED-CODE: `g16-zkey/src/lib.rs`,
  `a_query`, length `n_vars`]**
* `{ v_i(x)·G2 }` — section 7, `b_g2_query` **[VERIFIED-CODE]**
* `α·G1`, `β·G2`, `δ·G1`, `δ·G2` — zkey header / verifying key **[VERIFIED-CODE:
  `pk.alpha_g1`, `pk.beta_g2`, `pk.delta_g1`, `pk.delta_g2` used in `prove.rs`]**

So an attacker with a candidate witness `a*` can compute `A₀*` and `B₀*` **with public data
only**. This is the crux: `A₀` is a *binding but non-hiding* commitment to the witness, and
`rδ` is the entire hiding term. It is a one-time pad in the exponent.

### Why the honest scheme is perfectly ZK

Groth's own proof, quoted verbatim **[SOURCE: eprint 2016/260, proof of Theorem 1]**:

> "Perfect zero-knowledge follows from both real proofs and simulated proofs having uniformly
> random field elements A, B. These elements uniquely determine C through the verification
> equation, so real proofs and simulated proofs have identical probability distributions."

The simulator picks `A, B ← Z_p` uniformly and solves for `C` using the trapdoor
**[SOURCE: eprint 2016/260, §3.2 `Sim`]**. So:

* **Zero-knowledge is PERFECT and UNCONDITIONAL.** It does not need discrete log, q-SDH,
  AGM, GGM, or a random oracle. Nothing. It needs exactly one thing: `r` and `s` uniform in
  `F_r` and independent of each other and of the witness.
* **Soundness is the assumption-heavy half**: "statistical knowledge soundness against
  adversaries that only use a polynomial number of generic bilinear group operations"
  **[SOURCE: eprint 2016/260, Theorem 2]**, plus a trusted setup whose toxic waste
  (`α, β, γ, δ, x`) was destroyed.

That asymmetry is the whole point of this note. Soundness is where the scary assumptions are;
ZK is *supposed* to be free. It is free only if stage 10 is correct.

---

## 2. Failure modes of `r` and `s`, worked out

Throughout, "candidate-test oracle" means: given a guess `a*` of the witness, the attacker can
decide `a* == a` using only public data and a handful of group/pairing operations. That turns
ZK into a brute-force search whose cost is the entropy of the private witness. For circuits
whose private inputs are a vote, a bit, an age, a small balance, a member of a known set, or a
password from a wordlist, this is a total break.

### 2.1 `r = 0`

`A = A₀`. Attacker computes `A₀* = α·G1 + Σ a*_i·(u_i(x)·G1)` and tests `A₀* == A`.
**Candidate-test oracle in one MSM.** [DERIVED]

Additional consequence: two proofs of the same witness now have identical `A`. **Total
linkability** even without guessing anything.

This is not hypothetical — see §4.1, where gnark shipped exactly this shape for its commitment
extension and it became CVE-2024-45040 (High).

### 2.2 `s = 0`

Symmetric: `B = B₀`, test `B₀* == B` in G2. Same oracle, same linkability. [DERIVED]

### 2.3 `r = s = 0`

`A = A₀`, `B = B₀`, `C = L`. The proof is a **deterministic function of the witness**. Two
proofs of the same witness are byte-identical; the candidate-test oracle works on any of the
three elements. [DERIVED]

Note that `prove_with_blinders(circuit, witness, Fr::zero(), Fr::zero(), ...)` still produces a
proof that **verifies**. The verifier has no way to detect an unblinded proof — the verification
equation is satisfied for any `(r, s)` including `(0, 0)`. **[DERIVED; consistent with the
verification equation in `crates/g16-core/src/verify.rs`]** So there is no test-suite signal, no
runtime error, and no external oracle (snarkjs, rapidsnark) that will catch this. It is a
silent, total privacy failure that passes every correctness test in this repo.

### 2.4 `r = s` (same value used twice) — a non-obvious break

Even with `r` uniformly random, setting `s = r` gives a candidate-test oracle. [DERIVED]

Attacker computes, for a guess `a*`:

```
P := A − A₀*   ∈ G1
Q := B − B₀*   ∈ G2
```

and checks the pairing equation `e(P, δ·G2) == e(δ·G1, Q)`.

* If `a* = a`: `P = r·δ·G1` and `Q = r·δ·G2`, so both sides equal `e(G1,G2)^{rδ²}`. Passes.
* If `a* ≠ a`: write `Δ_A = Σ(a_i − a*_i)u_i(x)` and `Δ_B = Σ(a_i − a*_i)v_i(x)`. Then
  `P = Δ_A + rδ`, `Q = Δ_B + rδ`, and the check reduces to `δ·(Δ_A + rδ) == δ·(Δ_B + rδ)`,
  i.e. `Δ_A == Δ_B`, i.e. `Σ Δa_i·u_i(x) == Σ Δa_i·v_i(x)`. Generically false.

So `r = s` costs 2 pairings per candidate and needs no secret. Anyone "optimising" stage 10 to a
single field sample — a plausible mistake, and one a correctness test would never catch — has
destroyed zero-knowledge.

**Verified in our code**: `prove()` draws them separately, `let r = Fr::rand(rng); let s =
Fr::rand(rng);` **[VERIFIED-CODE: `crates/g16-core/src/prove.rs`]**. Correct.

### 2.5 `(r, s)` reused across two proofs — full derivation

This is the ECDSA-nonce-reuse analogue, and the question explicitly asks for the algebra rather
than the assertion. [DERIVED]

Two proofs `π = (A, B, C)` of witness `a` and `π' = (A', B', C')` of witness `a'`, produced with
the *same* `(r, s)`:

```
A − A' = (A₀ + rδ) − (A₀' + rδ) = A₀ − A₀' = Σ_i (a_i − a'_i)·u_i(x)
B − B' = (B₀ + sδ) − (B₀' + sδ) = B₀ − B₀' = Σ_i (a_i − a'_i)·v_i(x)
```

**The blinding term cancels exactly.** The attacker now holds `Δ_A`, a publicly-computable
function of the witness *difference*, with no randomness in it at all. Three consequences,
in increasing severity:

1. **`a = a'` (same witness, e.g. re-proving after a crash or a retry):** `A = A'`, `B = B'`,
   `C = C'`. The two proofs are **byte-identical**. Any observer learns the two proofs are of
   the same witness. In a mixer / nullifier / anonymous-credential setting this is the whole
   privacy property, gone, with zero computation.

2. **`a ≠ a'`, difference guessable:** the attacker enumerates candidate differences `Δa*` and
   tests `Σ Δa*_i·U_i == A − A'` where `U_i = u_i(x)·G1` are the public section-5 bases. If the
   two witnesses differ in a handful of low-entropy coordinates (two votes, two amounts, two
   ages), this is a small search. Meet-in-the-middle halves the exponent when the difference
   splits into two independent halves.

3. **`a'` fully known to the attacker** (they made the second proof themselves, or it is a proof
   of a public/test vector): then `A₀ = (A − A') + A₀'` recovers the *unblinded* `A₀` for the
   secret witness. From there, §2.1's candidate-test oracle applies to `a` with **no residual
   randomness**. This is complete recovery for any enumerable witness.

The correct mental model: `rδ` is a one-time pad over `A₀`. Reusing `r` is reusing a one-time
pad key across two messages, and `A − A'` is the classic "XOR of two ciphertexts is the XOR of
the plaintexts".

### 2.6 Predictable `r` (seeded / counter / non-cryptographic RNG)

If the attacker can predict or recompute `r`, they compute `A − r·(δ·G1) = A₀` directly and
apply §2.1. `δ·G1` is public (it is `pk.delta_g1`, read straight out of the zkey header
**[VERIFIED-CODE: `crates/g16-zkey/src/lib.rs`]**). Same for `s` and `B`. [DERIVED]

Concrete ways this happens in practice:
* seeding a PRNG with a timestamp, PID, or block height;
* `StdRng::from_seed([7u8; 32])` — which is literally what the tests in `prove.rs` do
  **[VERIFIED-CODE: `end_to_end_our_proof_verifies` uses `StdRng::from_seed([7u8; 32])`]**. Fine
  for a test; catastrophic if that pattern is copy-pasted into a service;
* `fork()`ing after the CSPRNG has been seeded, so parent and child produce identical streams
  (see §4.5 for how snarkjs and this repo differ here);
* VM snapshot / restore, which replays the same RNG state.

### 2.7 Low-entropy or biased `r`

If `r` has only `k` bits of entropy, the attacker brute-forces the `2^k` candidates for `r`,
unblinds, and applies §2.1. Total work `2^k · (cost of one MSM)`. [DERIVED]

**What does *not* happen, and it matters:** unlike ECDSA, a *biased* `r` does **not** give a
lattice / hidden-number-problem attack. In ECDSA the nonce appears in a scalar equation
`s = k⁻¹(h + r·d)` that the attacker sees in the clear, which is what makes HNP work. Here `r`
only ever appears *in the exponent* (`A = A₀ + rδ`), and recovering it from `A` is a discrete-log
problem. So a slightly-biased `r` is not exploitable by lattice reduction; it is only exploitable
by exhausting the reduced range. That distinction is worth stating precisely because "biased
nonce = lattice attack" is the reflex, and here the reflex is wrong. [DERIVED]

A concrete instance of a range-reduced blinder in a shipping prover: rapidsnark, §4.4.

### 2.8 What the leak actually is, stated precisely

In every case above the attacker gets a **candidate-test oracle**, not a printout of the
witness. Recovering `a` from an unblinded `A₀` in the general, high-entropy case means solving a
discrete-log-relation / subset-sum-in-the-exponent problem, for which no polynomial algorithm is
known. So the honest statement is:

> Broken blinding drops Groth16 from **perfect, unconditional** zero-knowledge to **at best
> computational hiding under a DL-type assumption**, and to **no hiding at all** against any
> adversary who can enumerate or partially guess the private witness.

For most real circuits the private witness is *exactly* the thing an adversary can guess (is
this voter's ballot a yes? is this balance above 1 ETH? is this the address on my watchlist?),
which is why the gnark advisory in §4.1 was rated High rather than Informational.

Note also that even in the "computational" case there is **no information-theoretic privacy
left**: `A₀` is a *binding* commitment, so it determines the witness uniquely as a mathematical
object. An unbounded adversary reads it off. Perfect ZK is genuinely lost, not merely weakened.

---

## 3. What this repo actually does — verified

All **[VERIFIED-CODE]**.

### 3.1 Stage 10 is correct

`crates/g16-core/src/prove.rs`:

* `prove<R: RngCore + CryptoRng>` — the `CryptoRng` bound is present and is the right bound. It
  is a marker trait, so it is a *lint*, not a proof (a caller can implement `CryptoRng` on
  anything), but it stops the accidental `SmallRng`/`StdRng::seed_from_u64` mistake at compile
  time.
* `r` and `s` are drawn as two independent `Fr::rand(rng)` calls. Not `r = s`, not derived from
  each other, not derived from the witness. Correct per §2.4.
* `ark_ff` `Fp`'s `Distribution<Fp>` impl uses **masked rejection sampling**: it samples random
  limbs, masks off the excess high bits (`num_bits_to_shave`), and loops while
  `tmp.is_geq_modulus()` **[VERIFIED-CODE: `ark-ff-0.5.0/src/fields/models/fp/mod.rs`]**. So `r`
  is *exactly* uniform on `[0, p)` — no modular bias, unlike a naive `random_bytes mod p`. This
  is what preserves *perfect* (not statistical) ZK.
* The CLI uses `ark_std::rand::thread_rng()`
  **[VERIFIED-CODE: `crates/g16-cli/src/main.rs:101`, `bench.rs:163,211`]**. In the resolved
  lockfile this is `rand 0.8.8` **[VERIFIED-CODE: `Cargo.lock`]**, whose `ThreadRng` is
  `ReseedingRng<ChaCha12Core, OsRng>` with a 64 KiB reseed threshold **and Unix fork protection
  via `pthread_atfork`** **[VERIFIED-CODE: `rand-0.8.8/src/rngs/thread.rs`,
  `src/rngs/adapter/reseeding.rs` — `fork::register_fork_handler()`, `is_forked()`]**. So a
  `fork()` in a proving server will *not* replay `(r, s)`. This is a genuinely good default and
  is better than snarkjs (§4.5).
* Nothing in the CLI offers a seed override.

**Verdict: stage 10 is the strongest part of the leakage surface.** The residual risks are the
three below.

### 3.2 `prove_with_blinders` is public API — the one stage-10 finding

`pub fn prove_with_blinders(...)` lives in `pub mod prove`, is **not** `#[cfg(test)]`-gated, and
is **not** feature-gated **[VERIFIED-CODE: `crates/g16-core/src/prove.rs`,
`crates/g16-core/src/lib.rs:29 pub mod prove`]**. Any downstream crate can call
`g16_core::prove::prove_with_blinders(c, w, Fr::zero(), Fr::zero(), &mut t)` and get a proof that
verifies under our verifier, snarkjs, and rapidsnark, while leaking per §2.3.

The doc comment says "Never use in production", which is the right intent, but a doc comment is
not an API boundary. The standard remedies (in increasing strength) are `#[doc(hidden)]`, a
`#[cfg(any(test, feature = "unsafe-deterministic-blinders"))]` gate, or moving it behind a
`__private` module. Note the deliberate design choice already made — that this is a separate
function rather than an `Option<(Fr, Fr)>` parameter on `prove` — is correct and should be kept;
the gap is only that the door is unlocked.

This is the single highest-value, lowest-cost hardening in the whole leakage dimension.

### 3.3 The MSM prescan is a witness-dependent timing/memory channel

`crates/g16-msm/src/lib.rs`, `fn prescan` **[VERIFIED-CODE]**:

```rust
if s.is_zero() { continue; }              // no bucket touched, no base read
if s.is_one()  { ones_sum += &bases[i]; continue; }
idx.push(i as u32);
bigints.push(s.into_bigint());
```

The Metal backend does the same classification on the host, and additionally materialises a
`general_prefix: Vec<u32>` — an exclusive prefix-sum over "is this scalar neither 0 nor 1"
**[VERIFIED-CODE: `crates/g16-metal/src/msm.rs`, `upload_scalars`, and the
`general_count(range)` accessor at line ~348 that sizes the GPU work from it]**.

Consequences:

* **Total prover time is a function of the number of zero and one entries in the witness.** The
  bucket loop runs over `idx.len()` = the count of "general" scalars, and `into_bigint()` (a
  Montgomery reduction) is called exactly that many times.
* On the Metal path, the *dispatch size* is derived from `general_prefix`, so the same
  dependence shows up in GPU kernel duration and in the size of the compacted buffers.
* `idx` and `bigints` are `Vec`s whose **length** equals the general-scalar count. That is a
  witness-dependent allocation size, observable in RSS, in allocator behaviour, and (on the
  Metal path) in the size of the `MTLBuffer` allocations.

This is the *identical* optimisation that was exploited against Zcash — see §4.3. It is not a
theoretical concern; it has a published attack with a measured correlation coefficient.

To be fair to the code: this optimisation is a large, real speedup (an MSM over a sparse witness
is dominated by the skipped terms), and every production Groth16 prover does it. The point is
that it is a **deliberate privacy/performance trade** that should be *documented as such*, not an
accident. See §7 for what a constant-time option would cost.

### 3.4 `n_public` is taken from the (possibly untrusted) `.zkey`

`crates/g16-cli/src/main.rs` writes the public signals as `&w[1..=n_public]` where `n_public`
comes straight from `.zkey` section 2 **[VERIFIED-CODE: `main.rs`,
`crates/g16-zkey/src/lib.rs:113`]**. The only validation is `n_vars >= n_public + 1`
**[VERIFIED-CODE: `g16-zkey/src/lib.rs:138`]** plus the CLI's `w.len() > n_public`.

So a `.zkey` that declares an inflated `nPublic` — with sections 3 and 8 sized consistently, so
every internal check passes — causes the prover to **write private witness entries into
`public.json`**. This is a genuine "the prover reveals things" path, and it is invisible: the
proof still verifies, and `public_json_matches_the_witness_prefix` would still pass because it
uses the same `n_public`.

The honest framing is that the `.zkey` *defines* which wires are public, so a prover cannot
detect the lie from the zkey alone. The defence is out-of-band: verify the zkey against the
circuit and the ceremony transcript (`snarkjs zkey verify`) before proving with it, and/or pin
the expected `n_public` at the call site. Worth a note in the CLI docs and a test that a zkey
with a mismatched `nPublic` is refused when an expected value is supplied.

### 3.5 Timings are exposed

`--stage-timings` prints `gather_us / ntt_us / pointwise_us / msm_us / assemble_us`
**[VERIFIED-CODE: `crates/g16-cli/src/main.rs`]**. `msm_us` is precisely the quantity §3.3
makes witness-dependent. Harmless in a local CLI; a direct leak if a hosted proving service ever
surfaces these numbers, or times its own responses.

### 3.6 No memory hygiene at all

`grep -rn "zeroize|Zeroize|mlock|madvise"` across `crates/` returns **nothing**
**[VERIFIED-CODE]**. See §6.

---

## 4. Real incidents and audit findings

### 4.1 gnark — CVE-2024-45040 / GHSA-9xcg-3q8v-7fq6 (the closest real-world analogue)

**"gnark commitments to private witnesses in Groth16 as implemented break zero-knowledge
property"**, High severity, affects `github.com/consensys/gnark < 0.11.0`, published 2024-09-06
**[SOURCE: https://github.com/advisories/GHSA-9xcg-3q8v-7fq6]**.

Quoting the advisory directly:

> "The commitment to private witnesses `w_i` is computed as `c = sum_i w_i * b_i` where `b_i`
> would be `ProvingKey.CommitmentKeys[0].Basis[i]` in the code. While this is a binding
> commitment, **it is not hiding**. In practice, an adversary will know the points `b_i`, as
> they are part of the proving key, and can verify correctness of a guess for the values of
> `w_i` by computing `c'` as the right hand side of the above formula, and checking whether
> `c'` is equal to `c`. I attach a proof of concept that demonstrates this. **This breaks the
> perfect zero-knowledge property of Groth16**, so the Groth16 scheme using commitments to
> private witnesses as implemented by gnark fails to be a zk-SNARK."

This is §2.1 of this document, verbatim, in a production library used by Linea and others. Note
the shape of the mistake: not a maths error, not a broken RNG — someone shipped a
witness-dependent group element with **no blinding term attached**, and the docs actively implied
the committed values stayed private. The recommended fix was to add a hiding term as in
LegoSNARK (eprint 2019/142). Fixed in gnark 0.11.0.

Sibling advisory **CVE-2024-45039 / GHSA-q3hw-3gm4-w5cr**, same date, same extension, is a
*soundness* break with multiple commitments **[SOURCE:
https://github.com/Consensys/gnark/security/advisories/GHSA-q3hw-3gm4-w5cr]**. Two bugs in the
same feature, one per security property.

Also relevant, from gnark's own documentation: *"gnark makes no security guarantees such as
constant time implementation or side-channel attack resistance"*
**[SOURCE: https://docs.gnark.consensys.io/overview]**. An explicit, upstream statement that
mainstream Groth16 provers do not attempt §3.3.

### 4.2 The "Frozen Heart" family — and why Groth16 is structurally immune

Trail of Bits disclosed **Frozen Heart** (FOrging of ZEro kNowledge proofs) in April 2022:
critical vulnerabilities in Girault's proof of knowledge, Bulletproofs, and PlonK
implementations, all caused by insecure Fiat–Shamir transforms where the challenge hash omitted
part of the transcript **[SOURCE:
https://blog.trailofbits.com/2022/04/13/part-1-coordinated-disclosure-of-vulnerabilities-affecting-girault-bulletproofs-and-plonk/
and the Bulletproofs/PlonK follow-ups, 2022-04-15 / 2022-04-18]**.

**Frozen Heart is a soundness class, not a leakage class** — it lets an attacker *forge* proofs
of false statements, not learn witnesses. And it does not apply to Groth16 at all: Groth16 is
natively non-interactive with a structured reference string and has **no Fiat–Shamir challenge
anywhere in the prover or verifier**. There is no transcript to under-hash
**[VERIFIED-CODE: `crates/g16-core/src/{prove,verify}.rs` contain no hashing of any kind]**.

Worth stating explicitly so it is not chased: the correct Groth16 analogue of "Frozen Heart" for
*this* dimension is not a hash bug, it is the blinder bug of §2, and its exemplar is
CVE-2024-45040.

### 4.3 Zcash — Remote Side-Channel Attacks on Anonymous Transactions (USENIX Security 2020)

Tramèr, Boneh, Paterson. **[SOURCE: https://crypto.stanford.edu/timings/paper.pdf, §6.1;
published USENIX Security 2020]**. This is the single most directly relevant prior work.

From the paper:

> "We show that for Zcash's zkSNARK system, proving times heavily depend on the value of the
> prover's witness."

> "Zcash uses the Groth16 proof system [23]. ... the prover encodes the witness as a vector
> `(a1, …, am)` of field elements, and ... the prover's main computation is a
> 'multi-exponentiation' of the form `Σ a_i G_i`. **Importantly, Zcash's implementation optimizes
> away terms `a_i G_i` where `a_i = 0`. The proof time thus correlates with the number of
> non-zero field elements in the prover's witness.** Since the transaction amount is encoded in
> binary in the witness, its Hamming weight influences the proving time."

Measured result: 200 amounts of the form `2^t`, 20 transactions each (4,000 proofs total), giving
**R = 0.57** correlation between proving time and transaction amount. The paper notes this
"could suffice to confidently identify rare transactions of large value", and that timing
fingerprints Zcash's zero-value "dummy Notes" particularly well.

The paper's own conclusion:

> "Our experiments show that the current implementation is therefore **not zero-knowledge in
> practice**: the information gleaned from timing leakage invalidates the zero-knowledge
> property."

> "It should also motivate the development of **constant-time implementations of cryptographic
> primitives such as zkSNARK provers**."

Useful contrast the same paper draws: Monero's Bulletproofs range proof is *inherently* immune
because it commits to both `a_L` (the bit decomposition) and `a_R = a_L − 1^n`, so the number of
EC operations is constant regardless of the value — "this property is inherent to the proof
protocol described by Bünz et al. and was **not** included as an explicit countermeasure". They
still observed ~0.5 ms of non-constant-time variation in Monero's multi-exponentiation from
"data-dependent operations and memory-access patterns", judged too small for a single remote
measurement but noting "performing local attacks would be a much simpler matter".

**Direct read-across to this repo:** our `prescan` (§3.3) skips zeros *and* ones, so it leaks
strictly more than the Zcash implementation the paper attacked. An adversary co-located on the
machine (a shared prover host, a cloud tenant, a browser/WASM context, another process reading
`/proc` or `mach` timing) is in the "local attack" regime the paper says is "a much simpler
matter".

### 4.4 rapidsnark — how a shipping prover samples its blinders

**[VERIFIED-CODE: `iden3/rapidsnark`, `src/groth16.cpp` and `src/random_generator.hpp`, `main`
branch as fetched]**

```cpp
E.fr.copy(r, E.fr.zero());
E.fr.copy(s, E.fr.zero());
randombytes_buf((void *)&(r.v[0]), sizeof(r)-1);
randombytes_buf((void *)&(s.v[0]), sizeof(s)-1);
```

and, unless the build defines `USE_SODIUM`:

```cpp
inline void randombytes_buf(void * const buf, const size_t size) {
    std::random_device engine;
    std::uniform_int_distribution<uint8_t> distr;
    uint8_t *buffer = static_cast<uint8_t*>(buf);
    for(size_t i = 0; i < size; i++) buffer[i] = distr(engine);
}
```

Two observations, both worth carrying into our own review:

1. **`sizeof(r) - 1`.** `r` is zeroed first, then 31 of its 32 bytes are filled, leaving the top
   byte at zero. So rapidsnark's blinders are uniform on `[0, 2^248)`, not `[0, p)` with
   `p ≈ 2^254`. That is presumably deliberate (it guarantees `r < p` without rejection
   sampling), and the ~6 bits of lost entropy are irrelevant to any brute force. But it means
   `r` covers about 1/64th of the field, so the statistical distance from uniform is ≈ 0.98 and
   **rapidsnark's Groth16 is not *perfectly* zero-knowledge as Groth proves it** — it is
   computationally ZK, since distinguishing requires deciding whether `A − A₀*` lies in
   `{ r·δ : r < 2^248 }`, i.e. a discrete-log problem. [DERIVED] A theoretical deviation, not a
   practical break — but a clean illustration that "close enough to uniform" silently downgrades
   the one property Groth16 gives you unconditionally. Our `ark_ff` rejection sampling (§3.1)
   does not have this issue.
2. **`std::random_device`.** The C++ standard does *not* require `std::random_device` to be
   non-deterministic; implementations are permitted to return a fixed PRNG stream, and MinGW-w64's
   libstdc++ notoriously did exactly that for years. On glibc and libc++ it reads
   `/dev/urandom` and is fine. rapidsnark's `USE_SODIUM` path (libsodium's `randombytes_buf`) is
   the safe one. Anyone porting rapidsnark to a new toolchain inherits this footgun. Contrast
   our `CryptoRng` bound plus `rand`'s `ReseedingRng<ChaCha12, OsRng>`, which is a genuine
   CSPRNG on every supported target.

### 4.5 snarkjs — how the reference implementation samples its blinders

**[VERIFIED-CODE: `iden3/snarkjs`, `src/groth16_prove.js:103-104`; `iden3/ffjavascript`,
`src/random.js`, `src/wasm_field1.js`, `src/f1field.js`, `master` as fetched]**

```js
const r = curve.Fr.random();
const s = curve.Fr.random();
```

For bn128, `curve.Fr` is a `WasmField1`, whose `random()` is `this.fromRng(getThreadRng())`.
`getThreadRng()` **memoises a single ChaCha instance** seeded once per process from
`crypto.randomFillSync` (Node) or `crypto.getRandomValues` (browser):

```js
let threadRng = null;
export function getThreadRng() {
    if (threadRng) return threadRng;
    threadRng = new ChaCha(getRandomSeed());
    return threadRng;
}
```

Assessment: the seed is 256 bits from the OS CSPRNG and ChaCha is a fine stream cipher, so the
`(r, s)` stream is cryptographically sound in the normal case. Two caveats:

* **No fork protection.** The ChaCha state is cached in module scope with no `pthread_atfork`
  equivalent. Any post-seed process clone (a forked worker, a VM snapshot/restore, a serverless
  container image snapshotted after warm-up) replays the *same* `(r, s)` and lands squarely in
  §2.5. Node's `child_process.fork` spawns a fresh process so it is safe; the risk is
  snapshot-based platforms and any native `fork()`. Our `rand 0.8` `ThreadRng` explicitly handles
  this (§3.1).
* The pure-JS `ZqField.random()` fallback in `f1field.js` uses `res % this.p` over
  `2 * bitLength / 8` bytes — modular reduction of a value ~2× the modulus width, so the bias is
  ~2^-254 and irrelevant. Not the path bn128 takes anyway.

### 4.6 Proof malleability / re-randomisation — a nuance, not a leak

Groth16 proofs are **re-randomisable**: given a valid `(A, B, C)` anyone can produce a fresh,
differently-encoded valid proof of the same statement without the witness
**[SOURCE: Baghery, Kohlweiss, Siim, Volkhov, "Another Look at Extraction and Randomization of
Groth's zk-SNARK", eprint 2020/811; discussed at
https://ethresear.ch/t/transaction-malleability-attack-of-groth16-proof/15881]**.

Two implications for this dimension:

* This is why Groth16 is only *weakly* simulation-extractable and why nullifier-based systems
  must bind the proof to a nullifier rather than treating the proof bytes as unique.
* It does **not** rescue the §2.5 reuse leak. Re-randomising fixes the "byte-identical proofs"
  symptom while leaving the algebraic relation `A − A' = Δ_A` recoverable up to the known
  re-randomisation factors. Do not treat re-randomisation as a substitute for fresh `(r, s)`.

### 4.7 Survey baseline

**[SOURCE: "What don't we know? Understanding Security Vulnerabilities in SNARKs",
arXiv:2402.15293v1, Tables 2 and 3]** — 141 vulnerability reports from audits, security
disclosures, and bug trackers, classified by layer and impact:

| Layer | Soundness | Completeness | **Zero-Knowledge** |
|---|---|---|---|
| Integration | 11 | 2 | 0 |
| Circuit | 94 | 5 | 0 |
| Frontend | 2 | 4 | 0 |
| **Backend (the prover)** | 17 | 3 | **3** |

Their taxonomy names our exact category: *"V11. Prover Error: Issues within the prover of the
proof system ... These vulnerabilities can lead to the prover mistakenly rejecting valid
witnesses, accepting invalid ones, **or breaking the zero-knowledge property**"*, and among root
causes: *"the use of **poor-quality or predictable randomness sources** can compromise crucial
ZKP properties"*.

They also note, importantly for calibrating how much to trust the empirical record:

> "instances of vulnerabilities being actively exploited are rare. There are no incidents of
> blackhat attacks publicly disclosed related to SNARKs. Furthermore, in applications that
> leverage the zero-knowledge property of SNARKs, **it can be especially challenging to determine
> if an attack has occurred due to the privacy-preserving nature of these systems**."

That last sentence is the reason this dimension deserves defence-in-depth rather than
wait-and-see: a ZK break is, by construction, silent.

### 4.8 Mitigation literature

**[SOURCE: "A New Multiscalar Multiplication Method Resistant to Timing Side-Channel Attacks",
eprint 2026/966]** — abstract, verbatim on the problem statement:

> "existing implementations remain vulnerable to timing attacks due to **irregular scalar
> representations and conditional operations on zero digits**. ... Our main contribution is a
> new scalar recoding algorithm that transforms conventional q-ary representations containing
> zero digits into equivalent non-zero representations. This ensures that all scalar digits are
> processed uniformly, eliminating timing-based side-channel leaks. ... To the best of our
> knowledge, **this is the first MSM algorithm explicitly designed to mitigate timing attacks
> within the Pippenger bucket method framework**."

The authors claim their variant is ~25% *faster* than baseline Pippenger, which if it holds up
would make constant-time free. Worth reading properly before writing off §3.3 as an unavoidable
trade — but note this is a 2026 preprint with no independent implementation yet, so treat the
performance claim as unverified.

Note also our own recoding is already carry-free and signed-digit
**[VERIFIED-CODE: `crates/g16-msm/src/lib.rs`, `signed_digit`, documented as having "no carry
chain at all", digits in `[-2^(c-1), 2^(c-1)]`]**, which is a good starting point — the leak is
in `prescan`'s zero/one skip and in the per-window bucket sparsity, not in the recoding.

---

## 5. Side channels, mapped onto the stage list

Stage numbering per `crates/g16-core/src/lib.rs`. Attacker model in brackets.

| Stage | Data-dependent? | What leaks |
|---|---|---|
| **-1** witness generation | Yes, heavily — circom's witness calculator has data-dependent control flow (branches, `<--`, comparisons). Out of scope for this crate but the largest channel in the pipeline. | Everything. |
| **0** coefficient gather | Structurally oblivious: a fixed walk over section-4 coefficient triples. **[VERIFIED-CODE: `read_coefficients` / gather in `cpu.rs`]** Values feed a Montgomery multiply-accumulate. | Only micro-level: `ark_ff`'s `subtract_modulus` is a conditional subtraction, so a single field mul is not strictly constant-time. Requires local, high-precision measurement. Low practical severity. |
| **1-3** iNTT / coset / NTT | **Oblivious.** Butterfly schedule, twiddle order, and memory access are fixed by `domain_size` alone; no branch depends on a coefficient. Good. | Nothing beyond `domain_size`, which is public. |
| **4** `H = A*B − C` | Oblivious, elementwise. | Nothing. |
| **5-9** the five MSMs | **NO — this is the leak.** `prescan` skips `s == 0` and `s == 1` (§3.3), and Pippenger's bucket occupancy is itself scalar-dependent even after the prescan. | Number of zero / one / general witness entries; per-window digit distribution. Exactly the Zcash channel (§4.3), and strictly more of it. |
| **10** sample `r`, `s` | `ark_ff` rejection sampling loops a variable number of times, but on RNG output, not on the witness. | Nothing witness-related. |
| **11** assemble | `G1Projective::normalize_batch` performs a batch inversion; `ark_ff` inversion is variable-time (binary extended Euclid). Its input is the `Z` coordinate of `pi_a = A₀ + rδ`, which depends on the *secret* `r`. **[VERIFIED-CODE: `prove.rs` `normalize_batch`]** | In principle, information about `r`; and leaking `r` unblinds `A₀` (§2.6). Extremely hard to exploit (one measurement, one inversion, buried under ~10⁹ prior cycles), but it is not *nothing*, and it is the one place where a *timing* channel touches the blinder rather than the witness. |

### The attacker who can observe the process

Ordered by capability:

1. **Remote, times the whole `prove` call.** Learns total duration → the number of non-{0,1}
   witness entries (§3.3) → the Hamming-weight-style leak of §4.3. This is the demonstrated
   attack.
2. **Co-located (same host / same cloud tenant / same browser process).** Adds cache and
   memory-access observations: which bases were touched (`bases[i]` is read only for non-zero
   `s`), which buckets were hit per window, allocation sizes of `idx`/`bigints`. The Zcash paper
   explicitly says local attacks are "a much simpler matter". On the Metal path, GPU counters and
   buffer sizes are additional observables.
3. **Same process (a malicious dependency, a WASM host, a debugger).** Reads the witness out of
   the heap directly (§6). Side channels are irrelevant at this point.
4. **Can invoke the prover repeatedly on a chosen or partially-chosen witness.** Averages away
   noise; the 0.5 ms-scale variations the Zcash paper dismissed for single remote measurements
   become reliable.

The practical guidance from the Zcash paper still holds: for a **local, offline** prover (a user
proving on their own laptop, this repo's CLI as used today) the timing channel has no observer
and is close to moot. It becomes real the moment proving is a *service* — a hosted prover, a
shared sequencer, a WASM prover in a page with hostile JS, or a "proving-as-a-service" backend.
Since this repo's stated shape is "a server holds one resident key and proves many witnesses"
**[VERIFIED-CODE: doc comment on `one_prepared_circuit_proves_concurrently` in `prove.rs`]**, that
is exactly the deployment being aimed at.

One more, easy to miss: **concurrency is itself a channel.** `PreparedCircuit` is `Send + Sync`
and multiple proofs run on one instance. Two concurrent proofs contend for the same rayon pool,
so proof A's duration depends on proof B's witness sparsity — a cross-tenant channel in a
multi-user prover.

---

## 6. Memory hygiene

All **[VERIFIED-CODE]** unless noted.

**Nothing is zeroed anywhere.** `grep -rn "zeroize|Zeroize|mlock|madvise|MADV"` over `crates/`
returns no matches. Concretely, the following live in ordinary heap memory for the duration of a
proof and are freed without scrubbing:

* the witness `Vec<Fr>` from `Witness::load` (`crates/g16-zkey/src/wtns.rs` — it copies section 2
  into an owned `Vec<Fr>`, so there are at least two copies: the file's page cache and the heap
  vector);
* the three domain vectors `A(X), B(X), C(X)` and the H coefficients, each `domain_size` field
  elements and each a deterministic function of the witness (`HPoly::Host(Vec<Fr>)`);
* `prescan`'s `idx: Vec<u32>` and `bigints: Vec<BigInt>` — note `idx` is *itself* a compact
  encoding of "which witness entries are neither 0 nor 1", i.e. a witness-derived bitmap sitting
  in the heap;
* on the Metal path, `general_prefix: Vec<u32>` (the same bitmap, as a prefix sum) plus every
  `MTLResourceOptions::StorageModeShared` buffer, which on unified-memory Apple silicon is
  host-visible process memory that the driver does not scrub on release.

Threat consequences:

* **Core dumps.** A panic with `panic = "abort"`, an `unwrap` in a Metal path, an OOM kill, or a
  segfault in FFI produces a dump containing the full witness if `ulimit -c` is non-zero or macOS
  is configured to write to `/cores`. macOS also writes `.ips` crash reports, which include
  register state and can include stack contents.
* **Swap.** Nothing is `mlock`ed, so witness pages can be paged out. Mitigating on macOS: swap
  has been encrypted by default since 10.7, and the key is per-boot **[SOURCE: Apple platform
  security documentation]**. Not mitigating on a Linux prover host with a plain swap partition.
* **Heap reuse / process introspection.** Freed-but-unzeroed witness data is readable by
  anything with `PTRACE`/`task_for_pid`, by a later allocation in a long-lived server, or by a
  malicious crate in the dependency tree.
* **The `.wtns` file itself is the biggest leak in the whole system.** It is the plaintext
  witness on disk, in a well-known format, and the CLI's normal workflow leaves it there
  indefinitely. No amount of in-memory scrubbing helps if the file survives. This is worth
  calling out loudly in the CLI docs.

**The proving key is not secret.** The `.zkey` is mmap'd read-only via `memmap2`
(`crates/g16-zkey/src/binfile.rs:55`) and contains only public setup data — the toxic waste
`α, β, γ, δ, x` was destroyed at ceremony time and is not in the file. So proving-key memory
hygiene does *not* matter for confidentiality. (It matters for *integrity*: §3.4, and the mmap
means a concurrent writer to the file can change bases under a running proof, which is a
correctness/DoS concern for a different dimension.)

**What "good" looks like** (for the recommendations stage, not applied here): `Zeroize` +
`ZeroizeOnDrop` on the witness vector and every derived domain vector; `mlock`/`VM_INHERIT_NONE`
on those pages; `setrlimit(RLIMIT_CORE, 0)` and `PR_SET_DUMPABLE`/`PT_DENY_ATTACH` in a
production prover; and deleting or `shred`ing the `.wtns` after use. Note that `Vec<Fr>` cannot
be reliably zeroed if it ever reallocates — the old buffer is gone before you can scrub it — so
capacity must be reserved up front. This is a real design constraint, not a one-line fix.

---

## 7. Ranked findings for the next stage

Ordered by (severity × likelihood), with what a test or fix would look like.

1. **`prove_with_blinders` is unrestricted public API (§3.2).** An `r = s = 0` proof verifies
   under all three oracles and leaks per §2.3. Gate it. *Test:* assert that a
   `prove_with_blinders(_, _, 0, 0, _)` proof reproduces `α·G1 + Σ a_i U_i` exactly, i.e. write
   the attack down as a test so the property is documented and any future regression in `prove`
   is caught. *Fix:* `#[cfg(any(test, feature = "..."))]` or `#[doc(hidden)]`.

2. **MSM prescan is a witness-dependent timing/memory channel (§3.3), the exact channel exploited
   against Zcash (§4.3).** *Test:* an empirical one — prove a witness that is 90% zeros and one
   that is 10% zeros, same circuit, and measure `msm_us`. If the gap is measurable (it will be),
   the leak is real and quantified. *Fix options:* (a) document it as an explicit trade in
   `g16-msm`'s module docs, which is the honest minimum; (b) offer an opt-in constant-time mode
   that processes all `n` scalars; (c) evaluate eprint 2026/966's non-zero recoding (§4.8).

3. **No blinder-quality regression test.** *Test:* draw N proofs of a fixed witness with a fresh
   `thread_rng` and assert all `pi_a` differ; assert `r != s` is structurally guaranteed by two
   separate `Fr::rand` calls; assert `prove` cannot be called with a non-`CryptoRng`. Cheap, and
   it pins the §2.4 / §2.5 properties.

4. **`n_public` trusted from the zkey (§3.4).** A crafted-but-internally-consistent zkey makes
   the CLI dump private witness entries into `public.json`. *Fix:* an optional
   `--expect-public <n>` and a doc note that `snarkjs zkey verify` against the circuit is a
   prerequisite for proving with a zkey you did not build.

5. **No memory hygiene (§6).** Witness, `.wtns` file, domain vectors, `prescan` bitmaps, Metal
   shared buffers — none zeroed, none locked, no core-dump suppression.

6. **Timings exposed (§3.5).** `--stage-timings` prints `msm_us`, which is finding #2's
   observable. Fine locally; a hazard if the crate is wrapped in a service.

7. **Variable-time batch inversion at stage 11 (§5).** The only timing channel that touches `r`
   rather than the witness. Very hard to exploit; worth a comment, not a rewrite.

8. **Concurrency as a cross-tenant channel (§5).** Two proofs sharing one `PreparedCircuit` and
   one rayon pool leak sparsity into each other's wall-clock time. Relevant only for the
   multi-tenant server shape, which is the stated target.

### Things explicitly ruled out

* **Frozen Heart does not apply.** Groth16 has no Fiat–Shamir transform; there is no transcript
  hash to under-bind (§4.2). Do not spend time here.
* **Biased `r` is not a lattice attack.** No hidden-number problem exists because `r` never
  appears in a scalar equation the attacker can read (§2.7). The only exploit path is exhausting
  the reduced range.
* **The proving key needs no confidentiality.** The `.zkey` is public by construction; mmap'ing
  it is fine (§6).
* **Stages 1–4 are oblivious.** The NTT/coset/pointwise pipeline has a fixed schedule. There is
  no leak to hunt there (§5).

---

## Sources

* Jens Groth, *On the Size of Pairing-based Non-interactive Arguments*, IACR ePrint 2016/260 —
  https://eprint.iacr.org/2016/260.pdf (Prove/Sim in §3.2; perfect ZK in Theorems 1 and 2)
* GHSA-9xcg-3q8v-7fq6 / CVE-2024-45040, *gnark commitments to private witnesses in Groth16 as
  implemented break zero-knowledge property* — https://github.com/advisories/GHSA-9xcg-3q8v-7fq6
* GHSA-q3hw-3gm4-w5cr / CVE-2024-45039, *Groth16 commitment extension unsound for more than one
  commitment* — https://github.com/Consensys/gnark/security/advisories/GHSA-q3hw-3gm4-w5cr
* Tramèr, Boneh, Paterson, *Remote Side-Channel Attacks on Anonymous Transactions*, USENIX
  Security 2020 — https://crypto.stanford.edu/timings/paper.pdf (§6.1 on Groth16 prover timing)
* Trail of Bits, *Coordinated disclosure of vulnerabilities affecting Girault, Bulletproofs, and
  PlonK* (2022-04-13) and the Bulletproofs / PlonK Frozen Heart posts (2022-04-15 / 04-18) —
  https://blog.trailofbits.com/2022/04/13/part-1-coordinated-disclosure-of-vulnerabilities-affecting-girault-bulletproofs-and-plonk/
* *What don't we know? Understanding Security Vulnerabilities in SNARKs*, arXiv:2402.15293 —
  https://arxiv.org/html/2402.15293v1 (Tables 2/3; taxonomy V11)
* Baghery, Kohlweiss, Siim, Volkhov, *Another Look at Extraction and Randomization of Groth's
  zk-SNARK*, ePrint 2020/811 — https://eprint.iacr.org/2020/811
* *A New Multiscalar Multiplication Method Resistant to Timing Side-Channel Attacks*, ePrint
  2026/966 — https://eprint.iacr.org/2026/966
* gnark documentation, security caveats — https://docs.gnark.consensys.io/overview
* Source read directly: `iden3/snarkjs` `src/groth16_prove.js`; `iden3/ffjavascript`
  `src/random.js`, `src/wasm_field1.js`, `src/f1field.js`; `iden3/rapidsnark` `src/groth16.cpp`,
  `src/random_generator.hpp`; `iden3/ffiasm` generated `fr.hpp`; `rand` 0.8.8
  `src/rngs/thread.rs`, `src/rngs/adapter/reseeding.rs`; `ark-ff` 0.5.0
  `src/fields/models/fp/mod.rs`; and this repository's `g16-core`, `g16-msm`, `g16-metal`,
  `g16-zkey`, `g16-cli`.
