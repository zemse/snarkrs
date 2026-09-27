# Verifier soundness pitfalls, Groth16 on BN254 (alt_bn128)

Research phase, dimension "soundness". Report only, no source files were edited.

Every claim below carries one of these markers:

* **[RUN]** — I proved it by compiling and running code against `ark-bn254 0.5.0` and/or against
  the real snarkjs artifacts in `bench/artifacts/`. The probe programs live in the session
  scratchpad at
  `/private/tmp/claude-501/-Users-sohamzemse-workspace-research/70a8845a-3b6f-45eb-be7b-0ecbbc2ff8b2/scratchpad/subprobe/`
  (`src/main.rs`, `src/bin/mal.rs`, `src/bin/enc.rs`, `src/bin/noncanon.rs`). They are throwaway,
  outside the repo, and depend on nothing in it.
* **[READ]** — I read the cited source or spec text myself and am quoting or paraphrasing it.
* **[DERIVED]** — arithmetic or reasoning I did on top of a [READ] fact, not executed end to end.
* **[LIT]** — from a paper or advisory I fetched but did not independently reproduce.

---

## 0. The numbers, fixed once

BN254 as used by circom/snarkjs/Ethereum, i.e. `alt_bn128`, seed `x = 4965661367192848881`.

| quantity | value |
| --- | --- |
| `p` (base field `Fq`) | `21888242871839275222246405745257275088696311157297823662689037894645226208583` |
| `r` (scalar field `Fr`) | `21888242871839275222246405745257275088548364400416034343698204186575808495617` |
| `#E(Fq)` | `= r` exactly, so **G1 cofactor `h1 = 1`** |
| `#E'(Fq2)` | `= r * h2` |
| `h2` (G2 cofactor) | `21888242871839275222246405745257275088844257914179612981679871602714643921549` |
| `h2` factored | `10069 * 2173824895405628684302950218021379986974303100027769687325441613140792921` |

* `h1 = 1` and `h2` as above are the `COFACTOR` constants in `ark-bn254 0.5.0`
  (`curves/bn254/src/curves/g1.rs`, `g2.rs`). **[READ]**
* `h2 == 2p - r` **[RUN]** — the standard BN sextic-twist identity, confirmed numerically.
* `p > r`, and `p - r = 147946756881789318990833708069417712966`. So an `Fq` element always fits
  in the same 32 bytes as an `Fr` element, but not conversely; a value in `[r, p)` is a legal
  coordinate and an illegal scalar. **[RUN]**
* **`h2` has the small prime factor 10069.** `E'(Fq2)` therefore contains a subgroup of order
  10069, and points of that order are trivially constructible by anyone: sample a random point
  on the twist and multiply by `#E'(Fq2)/10069`. **[RUN]** BN254 is *not* subgroup-secure in the
  sense of Barreto–Costello–Misoczki–Naehrig–Pereira–Zanon, "Subgroup security in pairing-based
  cryptography", eprint 2015/247, which proposes exactly the property BN254 lacks. **[LIT]**
  `2^254`-ish `h2` with a 14-bit factor means the small-subgroup search space is 10069 elements,
  not `2^254`.

---

## 1. Subgroup membership: what is actually required, and the folklore that is wrong

### 1.1 The rule

> **G1: on-curve is sufficient. G2: on-curve is not sufficient, a real subgroup check is required.**

`ark-bn254`'s own G1 implementation is the cleanest statement of the first half:

```rust
// curves/bn254/src/curves/g1.rs
fn is_in_correct_subgroup_assuming_on_curve(_p: &G1Affine) -> bool {
    // G1 = E(Fq) so if the point is on the curve, it is also in the subgroup.
    true
}
```

**[READ]** It is a compile-time `true`. Any call to it on G1 is dead code. **[RUN]** I sampled
random on-curve G1 points (solve `y^2 = x^3 + 3`, take a square root) and every one reported
`is_in_correct_subgroup_assuming_on_curve() == true` — as it must, since the function ignores its
argument.

Consequence for our code: `checked_g1` in `crates/g16-cli/src/json.rs` calls both `is_on_curve`
and `is_in_correct_subgroup_assuming_on_curve`. The second call is a no-op on BN254 G1. It is
harmless and correct to keep (it documents intent and survives a curve swap), but it must not be
mistaken for the thing that provides the security — for G1 the security comes entirely from
`is_on_curve`. **[READ]** of `crates/g16-cli/src/json.rs:107-115`.

G2 is the opposite. `ark-bn254` implements a real check:

```rust
// curves/bn254/src/curves/g2.rs
fn is_in_correct_subgroup_assuming_on_curve(point: &G2Affine) -> bool {
    // Subgroup check from section 4.3 of https://eprint.iacr.org/2022/352.pdf.
    // Checks that [p]P = [6X^2]P
    let x_times_point = point.mul_bigint(SIX_X_SQUARED);
    let p_times_point = p_power_endomorphism(point);
    x_times_point.eq(&p_times_point)
}
```

**[READ]** (El Housni, Guillevic, Piellard, "Co-factor clearing and subgroup membership testing on
pairing-friendly curves", eprint 2022/352, §4.3.) Note it is not a naive `[h2]P == O` cofactor
multiplication; it is the endomorphism-based test, one `mul_bigint` by a ~127-bit scalar plus a
Frobenius, which is much cheaper. That matters for us: the check is not free but it is not a full
254-bit scalar mul either.

### 1.2 The folklore, and why it is wrong

The common claim is: *"the final exponentiation maps everything into `mu_r`, and a point of order
coprime to `r` therefore pairs to 1, so the G2 subgroup check does not actually change the pairing
result."* The first half is true, the conclusion does not follow, and it is **false in practice**.

I constructed `T` in `E'(Fq2)` of exact order 10069 and measured, with `ark-bn254`'s optimal-ate
pairing: **[RUN]**

```
T on curve                : true
T in prime subgroup       : false
[10069] T == O            : true
e(A, T) != 1              : true      <-- the folklore predicts this is 1
e(A, T)^r == 1            : true      <-- so it IS in mu_r
e(A, T)^10069 == 1        : false     <-- and it has large order in mu_r, not order 10069
e(A, B) == e(A, B + T)    : false
e(A, B + T) == e(A,B) * e(A,T)  : false   <-- not even additive in the 2nd argument
e([k]A, T) == e(A, T)^k         : false   <-- not even bilinear in the 1st argument
```

The reason the folklore fails: the *reduced Tate* pairing is bilinear on the whole of
`E'(Fq2)[r']`, and for it the "small order dies in `mu_r`" argument is sound. The *optimal ate*
pairing that every BN254 library actually implements is only equal to a fixed power of the Tate
pairing when `Q` lies in the eigenspace `G2 = E'[r] ∩ ker(pi_p - [p])`. The Miller loop is run with
the ate loop parameter `6x + 2`, and off that eigenspace the resulting function is simply not the
pairing at all. `e(A, T)` lands in `mu_r` (the image of `x -> x^((p^12-1)/r)` is the order-`r`
subgroup, and `r` is prime, so a non-trivial output has order exactly `r`) but its value is
algebraically unstructured.

### 1.3 So what does an attacker actually get?

Being precise here, because this is where most write-ups overclaim in the other direction.

**What is *not* true:** there is no known efficient recipe that turns "the verifier skips the G2
subgroup check" into a Groth16 forgery on BN254. Concretely, taking a valid proof and adding a
small-order `T` to `B` makes it **fail**, not pass — that is exactly what
`e(A,B) != e(A,B+T)` says. **[RUN]** Any forgery attempt has to solve for a target value in `mu_r`,
which is a discrete log. **[DERIVED]**

**What *is* true, and is enough reason to always do the check:**

1. **The security proof evaporates.** Groth16's soundness argument (AGM / generic bilinear group)
   assumes the verifier is evaluating a bilinear map on `G1 x G2`. The measurements above show
   that with an off-subgroup `B` the verifier is evaluating something that is bilinear in neither
   argument. Nothing about Groth16's soundness has been proved for that object. "No known attack"
   is not "proved secure". **[DERIVED]**
2. **Cross-implementation divergence, which is a live consensus bug.** EIP-197 *mandates* the G2
   subgroup check (see §7). A proof that our Rust verifier accepts and the on-chain precompile
   rejects, or vice versa, is a real failure for any pipeline that pre-verifies off-chain and
   settles on-chain. Divergence is the practical attack surface here, not a direct forgery.
3. **The check is the only thing standing between "this is a G2 element" and "this is 762 bits of
   attacker-chosen data fed into a Miller loop."** Off-subgroup inputs are also where
   implementation-level bugs live (degenerate line functions, denominators, the `z = 0` branches
   in projective formulas).

**Recommendation:** always do the G2 subgroup check on every attacker-supplied `B`, and skip it on
G1 only if you have written down, in a comment, that `h1 = 1` is the reason.

---

## 2. On-curve checks

An off-curve point is not in *any* subgroup of `E`, so the on-curve check is logically prior to
everything else — `is_in_correct_subgroup_assuming_on_curve` is named for a precondition, and
arkworks does not enforce it for you.

* **[RUN]** With `ark-serialize 0.5`, `deserialize_with_mode(.., Validate::No)` accepts an
  off-curve G1 point and an off-subgroup G2 point without complaint; `Validate::Yes` (which is
  what plain `deserialize_compressed` / `deserialize_uncompressed` use) rejects both.
  So the arkworks footgun is precisely the `_unchecked` / `Validate::No` variants, and they exist
  and are used in performance-sensitive code all over the ecosystem.
* **[READ]** `crates/g16-cli/src/json.rs` builds points with `G1Affine::new_unchecked` /
  `G2Affine::new_unchecked` and then explicitly calls `is_on_curve()`. That is the correct
  pattern. `G1Affine::new` (the checked constructor) *panics* on a bad point rather than
  returning an error, which is why `new_unchecked` + explicit check is the right choice for a
  parser fed untrusted JSON. The module doc comment already says this.
* **What an attacker gains from an off-curve point:** the point lives on a different curve
  `y^2 = x^3 + b'` (same `a = 0`, different `b'`), whose group order factors completely
  differently and generally has many small factors — that is the classical invalid-curve attack.
  For an operation that multiplies the attacker's point by a *secret* scalar and leaks the
  result, this recovers the secret modulo the small factors. **[LIT]** (Biehl–Meyer–Müller
  invalid-curve attacks; the same shape as the 2015 "Practical Invalid Curve Attacks on TLS-ECDH".)
  **Groth16 verification does not multiply the proof by a secret**, so the classical invalid-curve
  key-recovery does not apply directly. The concrete risks that *do* apply here are:
  - the incomplete addition/doubling formulas and the Miller loop can hit undefined states
    (division by zero, wrong branch) on inputs the code assumes are on-curve, giving a panic
    (DoS) or, worse, a garbage-but-successful pairing result;
  - the same divergence problem as §1.3(2): EIP-196/197 reject off-curve inputs, so an off-curve
    point our verifier accepts is a proof no chain will ever accept.
* **`(0, 0)` is not on the curve.** `y^2 = x^3 + 3` gives `0 = 3`. **[RUN]** ark's
  `G1Affine::new_unchecked(0, 0).is_on_curve() == false`. This is exactly why ark can use `(0,0)`
  as its internal zero flag, and why the `.zkey` reader in `crates/g16-zkey/src/binfile.rs`
  translates snarkjs' `(0,0)` to `G1Affine::identity()` before any curve check. **[READ]**
  If it did not, every `(0,0)`-encoded infinity in a `.zkey` would be rejected as off-curve.

---

## 3. Point at infinity

Three separate places to get this wrong.

### 3.1 In the encoding

The identity is the one point with no affine `(x, y)`, so every format needs a special case, and
formats disagree on what it is:

| format | identity encoding |
| --- | --- |
| EIP-196 / EIP-197 (precompiles) | `(0, 0)` **[READ]**: *"the point at infinity is encoded as `(0, 0)`"* |
| snarkjs `.zkey` binary sections | `(0, 0)` **[READ]** of `crates/g16-zkey/src/binfile.rs:212-236` |
| snarkjs / ffjavascript JSON | Jacobian `["0","1","0"]` for G1, `[["0","0"],["1","0"],["0","0"]]` for G2 **[READ]** of `crates/g16-cli/src/json.rs:57-73` |
| `ark-serialize 0.5` uncompressed G1 | 64 bytes, all zero except the **last** byte `0x40` **[RUN]** |
| `ark-serialize 0.5` compressed G1 | 32 bytes, all zero except the last byte `0x40` **[RUN]** |
| Zcash/BLS12-381 style (`gnark`, `blst`) | flag bits in the **first** byte, `0b110…` **[LIT]** |

The failure mode is a decoder and an encoder that disagree about which of these they speak. A
decoder that reads `(0,0)` from a `.zkey` and hands the literal affine pair `(0,0)` to an
on-curve check rejects a legitimate key; a decoder that reads `["0","1","0"]` and forgets the
`z = 0` branch computes `1/0`.

### 3.2 In the proof elements

**[RUN]** on the real `js_2x2_d16` artifacts: substituting the identity for `A`, for `B`, or for
all three makes the pairing check **fail**. So "infinity in the proof" is not, on its own, an
accept-anything bug for the standard Groth16 equation:

```
A = inf      -> verifies: false
B = inf      -> verifies: false
A=B=C = inf  -> verifies: false
```

Reasoning about why, since "it fails on this one instance" is weak evidence: with `A = inf` the
check collapses to `e(alpha,beta) * e(L,gamma) * e(C,delta) == 1`. Making that hold for a chosen
`L` requires finding `C` with `e(C, delta)` equal to a specific target — fixed-argument pairing
inversion, believed hard. **[DERIVED]** The honest statement is: *infinity in `A` or `B` is not a
known break of Groth16 verification, but it is a proof carrying zero information about `A`, and
several protocols layered on top do break.*

Where infinity **does** break things:

* **Any verifier with a special case for zero.** The canonical real-world instance is the
  **Aztec Plonk "0 bug"** (Nguyen Thoi Minh Quan; catalogued as entry 7 in the 0xPARC
  zk-bug-tracker): *"By manually setting two elements to 0, the verifier accepts any proof
  regardless of other elements, allowing proof forgery."* The root cause is confusion between
  "0 means the point at infinity" and "0 means the field element 0". **[LIT]**
* **Negation of `y = 0`.** The idiom `-P = (x, q - y)` produces `q` when `y = 0`, and `q` is not a
  valid field element. snarkjs' generated Solidity verifier gets this right by writing
  `mod(sub(q, y), q)` rather than `sub(q, y)`. **[READ]** of
  `snarkjs/templates/verifier_groth16.sol.ejs`. A hand-rolled verifier that writes `q - y` has a
  latent bug on exactly the infinity-adjacent input.
* **Public-input aggregation.** If every public input is zero and `IC[0]` is somehow the identity,
  `L` is the identity. Not attacker-controlled in a correctly generated key, but it is the kind of
  degenerate state a fuzzer will find.

### 3.3 In the public input encoding

`L_bar = IC[0] + sum_i w_i * IC[i]`. The identity can show up here through `w_i = 0` (legitimate)
or through an `IC[i]` that is the identity (a malformed key). Neither is a soundness break by
itself, but note that a verifier which special-cases "if `L` is infinity, skip that pairing"
silently changes the verified statement. Do not add such a special case; `multi_pairing` handles
the identity correctly.

---

## 4. Non-canonical field encodings and compressed-point parsing

### 4.1 The canonical-range rule

A coordinate is a residue mod `p`, so exactly one 256-bit integer in `[0, p)` represents it. Any
decoder that *reduces* rather than *rejects* creates multiple byte encodings of the same point,
which is a malleability bug even when it is not a soundness bug.

* **EIP-196 and EIP-197 both reject.** EIP-197: *"An encoding value of `p` or larger is invalid"*
  and *"if the length of the input is incorrect or any of the inputs are not elements of the
  respective group or are not encoded correctly, the call fails."* **[READ]**
* **`ark-serialize 0.5` rejects.** **[RUN]** Overwriting the `x` limbs of a serialized G1 point
  with exactly `p` makes `deserialize_uncompressed` and `deserialize_compressed` return an error,
  in both compressed and uncompressed modes.
* **snarkjs' JS verifier rejects out-of-range *public inputs*** via
  `checkValueBelongToField(curve, v) = 0 <= v < r`, but does **not** range-check the proof
  coordinates in JS — it relies on `G1.isValid` / `G2.isValid` in ffjavascript. **[READ]** of
  `snarkjs/src/groth16_verify.js`.
* **Our own reader rejects.** `parse_field` in `crates/g16-cli/src/json.rs` compares the parsed
  `BigUint` against `F::MODULUS` and bails before converting, rather than calling
  `from_le_bytes_mod_order` on it. That is the right order of operations — `from_le_bytes_mod_order`
  would silently turn a malformed proof into a well-formed different one. **[READ]**

### 4.2 The `.zkey` binary path is *not* range-checked

`crates/g16-zkey/src/binfile.rs:186` is `Fq::new_unchecked(bigint(b))` — it installs the 4 limbs
verbatim as an arkworks Montgomery representative with no bound check. I measured what that
buys an attacker who controls a `.zkey`: **[RUN]**

```
x2 = Fq::new_unchecked(mont_limbs(x) + p)     // same value mod p, different representative
x2 == x  (arkworks PartialEq, limbwise)  : false
x2.into_bigint() == x.into_bigint()      : true
G1Affine::new_unchecked(x2, y).is_on_curve() : true      <-- passes the curve check
G1Affine::new_unchecked(x2, y) == G       : false        <-- but compares unequal to the same point
x2 * 1 == x * 1                          : true          <-- arithmetic is correct after one mul
```

So a `.zkey` can carry a non-canonical representative that survives `check_g1`, computes
correctly, and yet compares unequal to the canonical encoding of the same point. Impact is low —
the `.zkey` is trusted-setup input, not attacker input in the threat model where the setup is
trusted — but it is a hazard for anything that hashes or `==`-compares vk points (vk digest
pinning, `PreparedVerifyingKey` caching, cross-checking two decoders). `Fr` is handled correctly:
`fr_normal` uses `Fr::from_bigint(..)` which returns `None` above the modulus. **[READ]** The
asymmetry between `fq` and `fr_normal` is worth flagging on its own.

### 4.3 Compressed points: flag bits

`ark-serialize 0.5` on BN254 G1 (measured, not read from docs) **[RUN]**:

* compressed size 32 bytes, uncompressed 64 bytes;
* the coordinate is **little-endian**, so the flags live in the top two bits of the **last** byte;
* `0x80` = "y is the larger/positive root", `0x40` = infinity. For a point `P` I measured last
  byte `0x84`, for `-P` `0x04` — identical `x` bytes, the sign bit is the only difference;
* the identity's compressed encoding is 31 zero bytes then `0x40`.

Pitfalls, and how ark 0.5 handles each:

| attack on the encoding | ark 0.5 result |
| --- | --- |
| infinity flag set **plus** a non-zero `x` (compressed) | **rejected**, `UnexpectedFlags` **[RUN]** |
| infinity flag set plus non-zero `x, y` (uncompressed) | **rejected**, `UnexpectedFlags` **[RUN]** |
| both flag bits set at once | **rejected**, `UnexpectedFlags` **[RUN]** |
| `x` = `p` exactly (non-canonical) | **rejected** **[RUN]** |
| `x` with no square root (not an x-coordinate) | **rejected** **[RUN]** |
| off-curve uncompressed, `Validate::No` | **accepted** **[RUN]** |
| off-subgroup G2, `Validate::No` | **accepted** **[RUN]** |

Verdict: `ark-serialize 0.5`'s checked path is genuinely strict, and the *only* way to get a bad
point through it is to ask for it (`Validate::No` / `*_unchecked`). Two ecosystem-wide traps
remain:

1. **Flag-bit placement differs between families.** arkworks: top bits of the **last** byte
   (little-endian). Zcash / BLS12-381 / `gnark` / `blst`: top three bits of the **first** byte
   (big-endian), with a separate "compressed" bit. A decoder written against one convention and
   fed the other reads garbage and may still find a valid point. **[LIT]** / **[RUN]** for the
   ark half.
2. **The infinity flag + junk coordinate is the classic multi-encoding bug.** If a decoder returns
   the identity while ignoring the rest of the bytes, there are `2^254` byte strings that all
   decode to the identity. ark rejects this; many hand-rolled decoders do not. This is the
   BN254/BLS analogue of what ZIP-216 fixed for Ed25519-style encodings in Zcash. **[LIT]**

Our repo does not currently have a compressed-point path at all — the `.zkey` and `.wtns` formats
are uncompressed and the JSON is decimal — so this section is forward-looking. **[READ]** of
`crates/g16-zkey/src/binfile.rs` (`G1_BYTES`/`G2_BYTES` record strides) and `json.rs`.

### 4.4 Our JSON reader accepts many encodings of the same point

`read_g1` / `read_g2` in `crates/g16-cli/src/json.rs` accept a full Jacobian triple and
de-projectivise with `x/z^2, y/z^3` for any non-zero `z`. **[READ]** That means
`(x*z^2, y*z^3, z)` for every `z != 0` is a distinct JSON document that decodes to the same
proof and verifies. snarkjs itself always writes `z = 1`, and the `.zkey`'s own JSON reader
(`crates/g16-zkey/src/lib.rs:423`) is stricter — it *requires* `z == 1` or `z == 0`. The two
readers in this repo disagree about what a legal encoding is. **[READ]** This is an
encoding-malleability finding, not a soundness one, but see §5.4: it matters the moment anyone
hashes `proof.json`.

---

## 5. Groth16 proof malleability: `(A, B, C) -> (A * z^-1, B * z, C)`

### 5.1 Why it works

The verification equation is

```
e(A, B) = e(alpha, beta) * e(L, gamma) * e(C, delta)
```

`A` appears only in the first pairing's left slot, `B` only in its right slot, and the pairing is
bilinear:

```
e(A * z^-1, B * z) = e(A, B)^(z^-1 * z) = e(A, B)
```

for any `z` in `Fr^*`. Nothing else in the equation moves. `z` is a free parameter the prover
never committed to, so anyone holding a valid proof — not just the prover — can produce
unboundedly many other valid proofs of the same statement.

**[RUN]**, on the real snarkjs `js_2x2_d16` proof:

```
baseline verifies          : true
rescaled (A/z, Bz, C)      : true   (bytes differ: true)
10 independent random z    : all verify
```

There is a richer two-parameter re-randomisation that moves **all three** elements, using the fact
that `delta` is in the verifier's key:

```
A' = z1^-1 * A
B' = z1 * B + (z1 * z2) * [delta]_2
C' = C + z2 * A
```

Check: `e(A', B') = e(A,B) * e(z2 * A, delta)`, and the extra factor is absorbed exactly by
`e(C', delta) = e(C, delta) * e(z2 * A, delta)`. **[RUN]** on the same artifact:
`full rerandomised verifies: true (all three differ: true)`.

This is the standard result: Groth16 is **randomizable**. Baghery, Kohlweiss, Siim and Volkhov,
"Another Look at Extraction and Randomization of Groth's zk-SNARK" (eprint 2020/811), formalise
it — Groth16 achieves *weak* simulation extractability, "a relaxed weaker notion, that allows
proof randomization, while guaranteeing statement non-malleability", and explicitly **not** strong
SE, which would imply proof non-malleability. **[LIT]**

### 5.2 What it does NOT break

It is **not** a soundness break. The statement is unchanged: the public inputs are untouched, `L`
is untouched, and no new statement becomes provable. Anyone who can re-randomise a proof already
had a valid proof of that exact statement. Groth16's knowledge-soundness is unaffected; what fails
is only proof *uniqueness*. Say this precisely in any write-up, because "Groth16 is malleable" is
routinely misread as "Groth16 is unsound".

### 5.3 What it DOES break

1. **Proof bytes as a nullifier / replay key / dedup key.** Any system that says "we have seen
   this proof before, reject it" by hashing `(pi_a, pi_b, pi_c)` is defeated in one scalar
   inversion. The correct nullifier is a value *inside the circuit*, constrained and exposed as a
   public input — that is why Zcash, Tornado-style mixers, and Semaphore all derive the nullifier
   from the note/identity secret, never from the proof.
2. **Proof bytes as a transaction identity.** If a tx hash covers the proof bytes, an observer can
   front-run with a re-randomised proof carrying the same public inputs but a different tx hash,
   and claim any reward tied to inclusion. This is the standard "witness malleability" shape.
3. **Signature-of-knowledge / UC composition.** Anything that needs strong SE (a UC-secure
   protocol, a proof used as a signature) needs Groth16 wrapped — the two transformations in
   eprint 2020/811 (`Int-Groth16`, `Ext-Groth16`), or a strongly-SE scheme such as Groth–Maller
   SE-SNARK. **[LIT]**
4. **Any "the prover must have been the one who submitted this" inference.** Re-randomisation is
   available to any relay that saw the proof.

### 5.4 The encoding layer adds *more* non-uniqueness on top

Even holding `(A, B, C)` fixed as curve points, the byte encodings are not unique:

* our JSON reader accepts every Jacobian `z != 0` (§4.4) — infinitely many `proof.json` files per
  proof; **[READ]**
* the **snarkjs-generated Solidity verifier accepts 5 distinct calldata values for `_pA[1]`**.
  The template computes `-A.y` as `mod(sub(q, calldataload(add(pA, 32))), q)`. Yul's `sub` wraps
  mod `2^256`, so for `y >= q` the result is `(2^256 - y) mod q`, not `(-y) mod q`, while `A.x`
  and `B`, `C` are passed to the precompile raw and so must be canonical. **[READ]** of the
  template. I solved for the alternatives against the real `js_2x2_d16` proof and found exactly
  4 extra `uint256` values of `_pA[1]`, all `>= q`, that produce the identical negated point
  (`2^256 / q = 5`, hence 5 encodings total). **[DERIVED]** — arithmetic on the template's
  semantics, not executed on an EVM. Anyone keying replay protection on the calldata rather than
  on a public input gets 5 free replays before even touching the `z`-rescaling.

**Rule to write down:** the only unique, canonical identity of a Groth16 proof is
`(vk_hash, public_inputs)`. Never the proof bytes.

---

## 6. Public input handling

### 6.1 Wrong count — the most under-appreciated one

`IC` has `n_public + 1` entries; `IC[0]` is the coefficient of the constant-one wire and is never
transmitted. So the caller supplies exactly `len(IC) - 1` scalars. Two of the three major
implementations **do not enforce this**:

* **arkworks does not check.** **[READ]** `ark-groth16` `src/verifier.rs`:

  ```rust
  let mut g_ic = pvk.vk.gamma_abc_g1[0].into_group();
  for (i, b) in public_inputs.iter().zip(pvk.vk.gamma_abc_g1.iter().skip(1)) {
      g_ic.add_assign(&b.mul_bigint(i.into_bigint()));
  }
  ```

  `zip` stops at the shorter iterator. Pass fewer public inputs than the vk declares and the
  trailing `IC` entries are silently dropped (equivalent to setting those inputs to zero); pass
  more and the extras are silently ignored. `verify_proof` adds no length check of its own.
* **snarkjs' JS verifier does not check either.** **[READ]** `src/groth16_verify.js` loops
  `for (let i = 0; i < publicSignals.length; i++)` and indexes `vk_verifier.IC[i+1]`, with no
  comparison against `vk_verifier.nPublic`. Too few signals silently truncates; too many throws
  on `undefined` rather than returning a clean `false`.
* **gnark *does* check.** **[READ]** `backend/groth16/bn254/verify.go`:

  ```go
  if len(publicWitness) != nbPublicVars-1 {
      return fmt.Errorf("invalid witness size, got %d, expected %d (public - ONE_WIRE)", ...)
  }
  ```
* **Our verifier checks.** **[READ]** `crates/g16-core/src/verify.rs::aggregate_public` compares
  `public.len()` against `vk.ic.len() - 1` and returns `VerifyError::PublicInputCount`. Correct,
  and stricter than both arkworks and snarkjs. Keep it and add a regression test.

Why the truncation matters: the accepted proof is a genuine proof for the *padded-with-zeros*
instance. If the application layer reads meaning positionally — "signal 3 is the recipient",
"signal 5 is the amount" — then a short `public.json` shifts nothing but *drops* trailing signals
from the verified statement while the application still has its own idea of what they were. It is
a statement-substitution bug hiding inside a length mismatch.

**[RUN]** For completeness I checked that truncating the real 6-signal `public.json` to every
length `0..5` and re-running the pairing **fails** every time. Truncation on its own does not
produce an accepting proof for the original instance — the risk is the *other* direction: an
attacker proves the zero-padded instance honestly and then presents it against a verifier that
does not notice the count is short.

### 6.2 Out-of-range values

A public signal must be in `[0, r)`. If the verifier reduces instead of rejecting, `s` and `s + r`
are two different documents that verify against the same proof.

**[RUN]** on the real artifact: `Fr::from(s + r) == Fr::from(s)` is true, and substituting `s + r`
for the first public signal in `public.json` still verifies. If the application reads the raw
decimal string out of `public.json` — as an address, an amount, a Merkle root, a nullifier — it
sees a *different* value than the one the circuit was proved against. That is a genuine
application-level soundness break sitting one layer above the pairing.

Who rejects:

* snarkjs JS: yes, `checkValueBelongToField` requires `0 <= v < r`. **[READ]**
* snarkjs Solidity: yes, `checkField(v)` reverts unless `v < r`, applied to every `_pubSignals[i]`.
  **[READ]**
* our `read_public`: yes, `parse_field::<Fr>` bails at `>= MODULUS`. **[READ]**
  `crates/g16-cli/src/json.rs`.
* arkworks: N/A — its API takes `&[E::ScalarField]`, already-reduced by construction. The risk
  moves to whoever converts bytes/strings into `Fr`; `Fr::from_le_bytes_mod_order` and
  `BigUint -> Fr` both reduce silently. Anyone writing a JSON or calldata front-end on top of
  `ark-groth16` owns this check themselves.

### 6.3 The constant-one wire

`IC[0]` is the coefficient of witness index 0, which circom fixes to 1. Three ways to get it wrong:

1. **Forgetting it**, i.e. computing `L = sum_i w_i * IC[i]` over `i >= 1` only. Every proof
   fails; caught immediately, not a security bug.
2. **Letting it be supplied.** If the verifier accepts `n_public + 1` scalars and multiplies
   `IC[0]` by the first, the prover controls the constant-one wire. That is a full soundness break:
   the constant-one wire is what turns the R1CS into an affine system, and a prover who can set it
   to anything can rescale the entire instance. gnark's error message names this explicitly:
   `"expected %d (public - ONE_WIRE)"`. **[READ]**
3. **Off-by-one against a length check.** The safe invariant is a single one:
   `len(public) + 1 == len(IC)`, asserted in one place. Our `aggregate_public` does exactly this
   via `vk.ic.len().saturating_sub(1)` — note `saturating_sub` also makes an empty (malformed)
   `ic` produce `want = 0` rather than panicking, but it would then index `vk.ic[0]` and panic on
   an empty vector. **[READ]** A `vk.ic.is_empty()` guard at parse time is the clean fix; the
   `.zkey` reader may already guarantee non-empty, which is worth confirming separately.

---

## 7. What the reference implementations and standards actually mandate

| | on-curve G1 | subgroup G1 | on-curve G2 | subgroup G2 | coord `< p` | pubinput `< r` | pubinput count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **EIP-197** precompile **[READ]** | yes | n/a (`h1=1`) | yes | **yes, mandated** | yes | n/a | n/a |
| **gnark** `bn254` **[READ]** | via `IsInSubGroup` | yes | yes | yes | yes (`SetBytes`) | yes (`fr.Vector`) | **yes** |
| **snarkjs** JS **[READ]** | `G1.isValid` | — | `G2.isValid` | `G2.isValid` | via `fromObject` | **yes** | **no** |
| **snarkjs** Solidity **[READ]** | delegated to precompile | n/a | delegated | delegated | delegated | **yes** (`checkField`) | fixed-size calldata array |
| **arkworks** `ark-groth16` **[READ]** | only in `Validate::Yes` deserialization | trivially true | ditto | ditto | ditto | caller's problem | **no** (`zip`) |
| **this repo** (`json.rs` + `verify.rs`) **[READ]** | yes | yes (no-op) | yes | yes | yes | yes | **yes** |

Exact quotes worth keeping:

* **EIP-197**: *"Elliptic curve points are encoded as a Jacobian pair `(X, Y)` where the point at
  infinity is encoded as `(0, 0)`."* / *"An encoding value of `p` or larger is invalid."* /
  *"If the length of the input is incorrect or any of the inputs are not elements of the
  respective group or are not encoded correctly, the call fails."* / for G1, validation is
  *"verifying the encoding of the coordinates and checking that they satisfy the curve equation
  (or is the encoding of infinity)"*, while for G2 *"the order of the element has to be checked to
  be equal to the group order `q`."* **[READ]** — this is the single clearest statement in any
  standard of the G1/G2 asymmetry described in §1.
* **gnark**, `backend/groth16/bn254/prove.go:41-44` **[READ]**:

  ```go
  // isValid ensures proof elements are in the correct subgroup
  func (proof *Proof) isValid() bool {
      return proof.Ar.IsInSubGroup() && proof.Krs.IsInSubGroup() && proof.Bs.IsInSubGroup()
  }
  ```

  called from `Verify` before any Miller loop, returning
  `errCorrectSubgroupCheckFailed = errors.New("points in the proof are not in the correct subgroup")`.
  Note it checks all three, G1 elements included — cheap on BN254 (`IsInSubGroup` on G1 is a
  curve-equation check) and curve-agnostic.
* **snarkjs**, `src/groth16_verify.js` **[READ]**: `isWellConstructed` calls `G1.isValid(pi_a)`,
  `G2.isValid(pi_b)`, `G1.isValid(pi_c)`; `publicInputsAreValid` calls `checkValueBelongToField`
  on every signal. Both were added after the fact — earlier snarkjs releases verified without
  them, which is why pinning a snarkjs version matters when it is used as an oracle.

---

## 8. Observations on this repo (report only, nothing edited)

Positive, verified by reading the code:

1. `crates/g16-cli/src/json.rs::parse_field` rejects `>= MODULUS` rather than reducing — stricter
   than a naive `from_le_bytes_mod_order` reader, and stricter than what an `ark-groth16` caller
   gets by default.
2. `checked_g1` / `checked_g2` do on-curve **and** subgroup before the point escapes the parser.
3. `crates/g16-core/src/verify.rs::aggregate_public` enforces `public.len() == ic.len() - 1`,
   which neither `ark-groth16` nor `snarkjs`'s JS verifier does.
4. The 4-pair `multi_pairing` with `-A` is the standard single-final-exponentiation form and
   matches arkworks' 3-pair-plus-precomputed-`alpha*beta` variant in substance.

Gaps worth carrying into the audit phase:

1. **Validation lives in the CLI, not at the API boundary.** `g16_core::verify` takes
   `&Proof { pub a, pub b, pub c }` with public fields and performs **no** on-curve or subgroup
   check. Any caller that is not `g16-cli` — a library consumer, an FFI shim, a fuzz harness —
   gets an unvalidated verifier. A `Proof::from_parts(...) -> Result<Proof, _>` that does the
   checks, or a `verify` that re-checks, would close this. **[READ]** of
   `crates/g16-core/src/lib.rs:45-53` and `crates/g16-core/src/verify.rs`.
2. **Two JSON point readers with different strictness.** `crates/g16-zkey/src/lib.rs:423,452`
   require `z == 1` or `z == 0`; `crates/g16-cli/src/json.rs:130,160` accept any non-zero `z`.
   Pick one. The stricter one is the right one for anything attacker-supplied. **[READ]**
3. **`binfile::fq` is `new_unchecked` with no range check**, while `binfile::fr_normal` correctly
   uses `from_bigint` and rejects out-of-range. §4.2 shows what slips through. **[READ]** + **[RUN]**
4. **`aggregate_public` indexes `vk.ic[0]` after a `saturating_sub`** — an empty `ic` would panic
   rather than error. Confirm the `.zkey` reader guarantees `ic` is non-empty. **[READ]**
5. **No negative tests for off-subgroup G2 in the proof path.** §1.2's construction (sample on the
   twist, multiply by `#E'(Fq2)/10069`) is 20 lines and gives a deterministic, checked-in
   adversarial `pi_b` that must be rejected. That is the highest-value new test on this dimension.

---

## 9. Sources

Read directly (raw source or spec text, fetched during this session):

* EIP-197, "Precompiled contracts for optimal ate pairing check on the elliptic curve alt_bn128" —
  https://eips.ethereum.org/EIPS/eip-197
* EIP-196 (G1 add/mul precompiles) — https://eips.ethereum.org/EIPS/eip-196
* `arkworks-rs/groth16`, `src/verifier.rs` —
  https://raw.githubusercontent.com/arkworks-rs/groth16/master/src/verifier.rs
* `arkworks-rs/algebra`, `curves/bn254/src/curves/g1.rs` and `g2.rs` —
  https://raw.githubusercontent.com/arkworks-rs/algebra/master/curves/bn254/src/curves/g1.rs
  (and `/g2.rs`)
* `iden3/snarkjs`, `src/groth16_verify.js` —
  https://raw.githubusercontent.com/iden3/snarkjs/master/src/groth16_verify.js
* `iden3/snarkjs`, `templates/verifier_groth16.sol.ejs` —
  https://raw.githubusercontent.com/iden3/snarkjs/master/templates/verifier_groth16.sol.ejs
* `Consensys/gnark`, `backend/groth16/bn254/verify.go` and `prove.go` —
  https://raw.githubusercontent.com/Consensys/gnark/master/backend/groth16/bn254/verify.go
* `iden3/ffjavascript`, `src/wasm_curve.js`
* This repo: `crates/g16-core/src/verify.rs`, `crates/g16-core/src/lib.rs`,
  `crates/g16-cli/src/json.rs`, `crates/g16-zkey/src/binfile.rs`, `crates/g16-zkey/src/lib.rs`

Papers and advisories (fetched, not independently reproduced):

* Barreto, Costello, Misoczki, Naehrig, Pereira, Zanon, "Subgroup security in pairing-based
  cryptography", eprint 2015/247 — https://eprint.iacr.org/2015/247
* El Housni, Guillevic, Piellard, "Co-factor clearing and subgroup membership testing on
  pairing-friendly curves", eprint 2022/352 — https://eprint.iacr.org/2022/352 (§4.3 is the G2
  subgroup test arkworks implements)
* Baghery, Kohlweiss, Siim, Volkhov, "Another Look at Extraction and Randomization of Groth's
  zk-SNARK", eprint 2020/811 — https://eprint.iacr.org/2020/811
* 0xPARC zk-bug-tracker, entry 7, "Aztec Plonk Verifier: 0 Bug" (Nguyen Thoi Minh Quan) —
  https://github.com/0xPARC/zk-bug-tracker

Note on method: DuckDuckGo (`ddg`) returned a CAPTCHA on every query this session, so all external
material above was fetched directly from known primary URLs (`curl` on raw source, `WebFetch` on
the spec and eprint pages) rather than found through search. Nothing here rests on a search-result
snippet.
