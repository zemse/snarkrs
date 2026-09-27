# Audit: soundness of the verifier and the proof/vkey deserializers

Scope: `crates/g16-core/src/verify.rs`, `crates/g16-cli/src/json.rs`, and the vkey JSON
reader that the `verify` subcommand actually calls, which is *not* in `json.rs` but in
`crates/g16-zkey/src/lib.rs` (`VerifyingKey::from_json`, lines 326-377, with `json_g1` at
407 and `json_g2` at 429).

Method: every claim tagged VERIFIED was produced by a probe binary linked against the real
crates (path dependencies, no repo files touched) running against
`bench/artifacts/js_2x2_d16/{proof,public,vkey}.json`, or by reading the arkworks 0.5.0
sources in `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`. Claims tagged
INFERRED were not executed. Probe sources are outside the repo, at
`/private/tmp/claude-501/-Users-sohamzemse-workspace-research/70a8845a-3b6f-45eb-be7b-0ecbbc2ff8b2/scratchpad/sndprobe/`.
No source file in the repo was modified.

## Answer to the three direct questions

### 1. Which checks do we perform, and which does arkworks perform for us

| Check | Who does it | Where | Real or vacuous |
| --- | --- | --- | --- |
| Coordinate `< p` (canonical field element) | us | `json.rs:40-42` (`parse_field`), `g16-zkey/src/lib.rs:396` (`json_fq` via `Fq::from_bigint`) | real, VERIFIED |
| On-curve, G1 | us | `json.rs:108-110` | real, VERIFIED |
| On-curve, G2 | us | `json.rs:119-121` | real, VERIFIED |
| Subgroup, G1 | us, but the callee is a constant | `json.rs:111` calls `ark_bn254::g1::Config::is_in_correct_subgroup_assuming_on_curve`, which is a literal `true` at `ark-bn254-0.5.0/src/curves/g1.rs:52-56` | vacuous by construction, and correctly so: G1 cofactor is 1 (`g1.rs:20-21`), so on-curve *is* subgroup. VERIFIED by reading |
| Subgroup, G2 | us | `json.rs:124` -> `ark-bn254-0.5.0/src/curves/g2.rs:54-62`, the `[p]P == [6X^2]P` endomorphism test from eprint 2022/352 sec 4.3 | real and load bearing, VERIFIED against a genuine order-10069 twist point |
| Infinity accepted as a legal proof element | us, permissively | `json.rs:134-135`, `json.rs:159-160` | accepted, see finding S5 |
| Public input count | us | `verify.rs:42-49` | real and exact, VERIFIED |
| Anything at all inside `verify()` | nobody | `verify.rs:19-36` | see finding S3 |

arkworks performs **no** validation on the path we use. `Bn254::multi_pairing`
(`ark-ec-0.5.0/src/pairing.rs:104-109`) goes straight to `multi_miller_loop` with no curve
or subgroup test. We never touch `ark-serialize`'s checked deserializer for proofs, because
the proof arrives as snarkjs JSON and is built with `G1Affine::new_unchecked` /
`G2Affine::new_unchecked` in `json.rs:107` and `json.rs:118`. So `ark-serialize`'s strictness
(the `Validate::Yes` path) is simply not in our proof pipeline at all. The only thing
arkworks does implicitly is **drop** pairs where either element is zero, at
`ark-ec-0.5.0/src/models/bn/mod.rs:55-65`, which is the mechanism behind findings S5 and S6.

### 2. Case trace

**Coordinate `>= p`.** Caught. `json.rs:40-42` compares the parsed `BigUint` against
`F::MODULUS` and bails before `from_le_bytes_mod_order` can silently reduce. VERIFIED: the
existing unit test `rejects_a_coordinate_at_or_above_the_modulus` (json.rs:257) covers it, and
my probe confirmed the same rejection for a public signal shifted by `+r`
(`public signal 0: ... is not less than the field modulus`). One hole: if the Jacobian `z` is
zero the reader returns before ever parsing `x` and `y`, so `["<p>","<p>","0"]` and even
`[{"lol":1},[1,2,3],"0"]` are accepted. VERIFIED. See S2.

**Off-curve point.** Caught. `json.rs:108-110` for G1, `json.rs:119-121` for G2. VERIFIED:
`read_g1(["1","3","1"])` errors (existing test at json.rs:271), and my probe's off-curve G2
`([1,0],[1,0])` never reaches the pairing through the reader.

**Point at infinity.** Accepted by the reader as a legal encoding (`json.rs:134-135` for
`z == 0`, `json.rs:159-160` for `z == [0,0]`), then handed to the pairing. It is **not** an
accept-anything bug: VERIFIED on the real `js_2x2_d16` artifact that `A=inf`, `B=inf`,
`C=inf` and all-infinity each make `verify` return `false`. But the reason they fail is
arithmetic, not a structural rejection, and the failure mode is the interesting part. See S5.

**Wrong number of public inputs.** Caught, exactly, at `verify.rs:42-49`. VERIFIED both
directions: five signals against a six-signal key gives `expected 6 public inputs, got 5`,
seven gives `expected 6 public inputs, got 7`. This is stricter than `ark-groth16`
(`prepare_inputs` zips and truncates) and stricter than snarkjs' JS verifier, per the
research note. Nothing to fix. There is one panic reachable only from a library caller, S7.

### 3. Rescaled proofs

**Yes, and it is correct Groth16 behaviour.** VERIFIED on the real artifact:

- `(A * z^-1, B * z, C)` verified for 10 out of 10 random `z` in `Fr`.
- The full two-parameter re-randomisation `A' = z1^-1 A`, `B' = z1 B + z1 z2 [delta]_2`,
  `C' = C + z2 A`, which moves all three group elements, verified 10 out of 10.

Both are cheap: one field inversion and a handful of group operations, no witness, no key.
Groth16 is weakly simulation-extractable (randomisable, statement-non-malleable), never
strongly SE. See Baghery, Kohlweiss, Siim, Volkhov, eprint 2020/811.

**What a downstream consumer must therefore not do.** Never derive an identity from proof
bytes. Concretely, do not use the proof (or a hash of it) as: a nullifier, a replay-protection
key, a deduplication key, a transaction identity, a commit-reveal binding, an idempotency
token, or a cache key that is assumed collision-free per statement. Any of those is defeated
by one scalar inversion by anyone who has already seen a valid proof. The canonical identity of
a Groth16 proof is `(vk_hash, public_inputs)`, and replay protection must live in the public
inputs (a nullifier the circuit computes) or in application state, never in the proof
encoding. This should be a doc comment on `Proof` and on `verify`.

## Findings, ranked by real exploitability

### S1. MEDIUM. Proof malleability is unmitigated and undocumented

**What.** `verify` accepts re-randomised proofs, as it must. Nothing in the crate says so.
`crates/g16-core/src/lib.rs:46-53` documents `Proof` only as "serialises to snarkjs'
proof.json shape". `verify.rs:19` has no note.

**Where.** `crates/g16-core/src/verify.rs:19-36`, `crates/g16-core/src/lib.rs:46-53`.

**Why it matters.** The exploitable thing is not the verifier, it is the next person to
build on it. A prover crate that hands back a `Proof` with no warning is exactly how a
nullifier-from-proof-bytes bug gets written. VERIFIED 20 out of 20 accepted re-randomisations
above.

**Fix.** Documentation only, no code change. Add to the `Proof` doc comment and to `verify`:
"a valid proof can be re-randomised by anyone who holds it into a different, equally valid
proof of the same statement. Proof bytes are not an identity. Use `(vk_hash, public_inputs)`."
Optionally ship the re-randomisation as a test so the property is pinned rather than folklore.

### S2. MEDIUM. `proof.json` has no canonical encoding, and our two JSON readers disagree

**What.** `read_g1` and `read_g2` accept *any* nonzero Jacobian `z` and normalise
`(X/z^2, Y/z^3)`. Every `z` in `Fq*` gives a different `proof.json` that decodes to the same
proof, so roughly `p` (about 2^254) distinct files per proof, before any group-level
malleability. Separately, when `z == 0` the reader returns the identity **without ever
parsing `x` and `y`**, so those two slots may contain out-of-range integers, or objects and
arrays that are not even strings.

**Where.** `crates/g16-cli/src/json.rs:130-146` (`read_g1`) and `156-169` (`read_g2`); the
early returns are `json.rs:134-135` and `json.rs:159-160`.

**Why it matters.** Three separate bites.
1. Our own vkey reader forbids exactly this: `g16-zkey/src/lib.rs:418-422` rejects `z != 1`
   for G1 and `447-451` rejects `z != [1,0]` for G2. Two readers in one repo disagree on what
   a legal snarkjs point encoding is. Whichever is right, they should not differ.
2. Verdict divergence with the oracles. A file with `z = 7` is accepted by us. snarkjs and
   rapidsnark are not obliged to accept it, and the whole benchmark rests on the three
   verifiers agreeing. This turns "our verifier said yes" into a claim that does not transfer.
3. It hands the attacker in S1 a strictly cheaper attack: no group operations at all, just
   rewrite the file, and it works even against a consumer that canonicalises the *point* but
   hashes the *file*.

The `z == 0` sub-case is the same class as the snarkjs Solidity `_pA[1]` finding in the
research note (five calldata encodings of one negated point), except ours admits arbitrary
JSON in the ignored slots, not just five integers.

**VERIFIED.** Five random `z` values produced five distinct `pi_a` encodings, all accepted by
`proof_from_value`, all decoding to the identical `G1Affine`, all verifying. `["<p>","<p>","0"]`
accepted. `[{"lol":1},[1,2,3],"0"]` accepted. `[[{"a":1},"x"],["y","z"],["0","0"]]` accepted for
`pi_b`.

**Fix.** Make `json.rs` match `g16-zkey/src/lib.rs`: require `z == 1` (G1) and `z == [1,0]`
(G2) after the zero test, and parse `x` and `y` *before* the zero test so the identity encoding
is pinned to exactly `["0","1","0"]` and `[["0","0"],["1","0"],["0","0"]]`, which is what our
own writer at `json.rs:60` and `json.rs:72` emits. The Jacobian branch at `json.rs:139-143`
then becomes dead and can go; its own comment already admits "snarkjs always writes z = 1 ...
this branch only matters for hand-written input". Deleting it removes the divergence and one
field inversion per point.

### S3. MEDIUM. `verify()` performs zero validation; all of it lives in the CLI

**What.** `g16_core::verify::verify` takes `&Proof`, whose three fields are `pub`
(`lib.rs:49-53`), and does no on-curve, subgroup, or canonicality check. Every check in the
table above is in `g16-cli/src/json.rs`, a *binary support crate*.

**Where.** `crates/g16-core/src/verify.rs:19-36`; `crates/g16-core/src/lib.rs:49-53`.

**Why it matters.** The security boundary is drawn at the wrong layer. Any consumer that is
not our CLI gets an unvalidated verifier: a library user constructing `Proof` by hand, an FFI
shim, a fuzz harness, a future gRPC service, or anyone who deserialises with `ark-serialize`
and `Validate::No` (which, per the research note, accepts off-curve G1 and off-subgroup G2).
The G2 subgroup check is the one that matters, and the crate that owns the pairing does not
own it.

**VERIFIED.** Calling `verify` directly with an off-subgroup `B` (real `pi_b` plus an
order-10069 twist point) returns `false` rather than erroring, and with an off-curve `A` or
`B` also returns `false`. So today the failure is benign on this curve. That is a property of
BN254 and of arkworks' arithmetic, not a property our API guarantees, and the AGM soundness
proof stops applying the moment an off-subgroup element reaches the Miller loop.

**Fix.** Either (a) validate at the top of `verify`: `is_on_curve` on all three plus
`is_in_correct_subgroup_assuming_on_curve` on `proof.b`, roughly 100 microseconds against a
1 to 2 millisecond pairing, with a new `VerifyError::InvalidPoint` variant; or (b) make
`Proof`'s fields private behind a checked constructor `Proof::new(a, b, c) -> Result<..>` so
an unvalidated `Proof` cannot be spelled. (a) is cheaper to land and defends the existing API.
Keep the reader's checks too, so a bad file is rejected with a good error before the pairing.

### S4. LOW. Query sections of the `.zkey` are never curve or subgroup validated

**What.** `crates/g16-zkey/src/lib.rs:153-170` validates only the six header points and `IC`.
Sections 5 to 9 (`a_query`, `b_g1_query`, `b_g2_query`, `l_query`, `h_query`) are read with
`g1`/`g2` (`binfile.rs:214`, `binfile.rs:226`), both of which use `new_unchecked`, and are
never checked. The comment at `lib.rs:153-158` states this is deliberate.

**Where.** `crates/g16-zkey/src/lib.rs:145-151` (reads), `153-170` (partial checks).

**Why it matters.** The performance argument is sound and the `.zkey` is a trusted setup
artifact, so this is not a break. But it means a hostile or corrupted `.zkey` yields a `pi_b`
that is off-subgroup, which our own reader would then reject (good) while an on-chain
verifier that skips the check would not (bad). It is a self-inconsistency, not an attack.

**Fix.** Keep the fast path. Add an opt-in `ProvingKey::validate_queries()` doing the full
subgroup sweep in parallel, wire it to a `--validate-key` CLI flag, and run it in CI over the
bench artifacts. Do not put it on the hot path.

### S5. LOW / INFORMATIONAL. Infinity in a proof degrades the equation instead of failing it

**What.** `ark-ec-0.5.0/src/models/bn/mod.rs:55-65` filters out any pair where either element
is zero *before* the Miller loop. Our verifier passes four pairs and never checks for
infinity, so a proof with `A = inf` silently turns the four-pair check into a three-pair check
`e(alpha,beta) e(L,gamma) e(C,delta) == 1`.

**Where.** `crates/g16-core/src/verify.rs:25-29`.

**Why it matters.** VERIFIED that this does not accept anything on the real artifact:
`A=inf`, `B=inf`, `C=inf` and all-infinity all return `false`. Forging the degraded equation
would require producing `C` with `e(C, delta) = e(-alpha, beta) e(-L, gamma)`, which needs
`alpha*beta/delta` in G1, a value the CRS does not publish. So there is no exploit here.
It is listed because it is precisely the shape of the Aztec Plonk "0 bug" (zk-bug-tracker
entry 7: set two elements to zero and the verifier accepts), and because a future
special-cased or hand-written pairing would not necessarily inherit the benign outcome.

**Fix.** Reject infinity for `proof.a`, `proof.b`, `proof.c` explicitly (an honest prover never
produces one, since `A = [alpha + sum + r*delta]` with `r` uniform), and add the four
infinity cases as regression tests. Cheap, and it makes the guarantee structural rather than
arithmetic.

### S6. LOW / INFORMATIONAL. A degenerate verifying key makes `verify` accept everything

**What.** If every one of the four pairs has at least one zero element, the filter at
`bn/mod.rs:55-65` empties the pair list, the Miller loop's `product` over an empty iterator is
`Fq12::one()`, the final exponentiation leaves it at one, and `result.is_zero()` at
`verify.rs:30` is true (`PairingOutput::is_zero` is `self.0.is_one()`,
`ark-ec-0.5.0/src/pairing.rs:187-189`). `verify` returns `Ok(())` for any proof.

**Where.** `crates/g16-core/src/verify.rs:25-34`.

**Why it matters.** VERIFIED: a hand-built `VerifyingKey` with `alpha_g1`, `gamma_g2`,
`delta_g2` and all of `ic` set to the identity accepts an arbitrary proof. It is not reachable
through either shipped loader, because `ProvingKey::load` runs `check_g1`/`check_g2` on the
header points and `IC` (`lib.rs:160-169`) and `VerifyingKey::from_json` runs the same via
`json_g1`/`json_g2`, and neither rejects the identity explicitly but a real setup never emits
one. It matters only for a caller that builds a `VerifyingKey` literal, which the `pub` fields
invite.

**Fix.** Reject an identity `alpha_g1`, `beta_g2`, `gamma_g2` or `delta_g2` at key load, in
both `ProvingKey::load` and `VerifyingKey::from_json`. One line each in `check_g1`/`check_g2`
behind a flag, or a separate `reject_identity` helper. `IC` entries may legitimately be the
identity, so do not blanket-reject there.

### S7. LOW. `aggregate_public` panics on an empty `ic`

**What.** `verify.rs:42` computes `want = vk.ic.len().saturating_sub(1)`, so an empty `ic`
gives `want == 0`, an empty `public` slice passes the length check at `43`, and `verify.rs:50`
then indexes `vk.ic[0]` and panics.

**Where.** `crates/g16-core/src/verify.rs:42` and `crates/g16-core/src/verify.rs:50`.

**Why it matters.** VERIFIED: `index out of bounds: the len is 0 but the index is 0` at
`verify.rs:50`. Unreachable through both shipped constructors (`from_json` requires
`IC.len() == nPublic + 1 >= 1` at `g16-zkey/src/lib.rs:357`; `load` reads `n_public + 1`
points at `lib.rs:145`). Reachable for any library caller, because `VerifyingKey`'s fields are
`pub`. This is a panic, not an accept, so it is availability not soundness.

**Fix.** Replace the `saturating_sub` with an explicit empty test returning a new
`VerifyError::MalformedKey`, so the impossible case is an error rather than a subtraction that
lies.

### S8. LOW. `multi_pairing` unwraps a fallible final exponentiation

**What.** `ark-ec-0.5.0/src/pairing.rs:108` is
`Self::final_exponentiation(Self::multi_miller_loop(a, b)).unwrap()`, and the BN
implementation at `ark-ec-0.5.0/src/models/bn/mod.rs:116` is `f.inverse().map(...)`, returning
`None` when the Miller output is zero in `Fq12`.

**Where.** `crates/g16-core/src/verify.rs:25`.

**Why it matters.** With validated inputs the Miller loop output is a product of nonzero line
evaluations and cannot be zero, so this is unreachable in the CLI. Combined with S3, an
unvalidated library caller feeding a crafted off-curve point has a candidate panic-DoS.
INFERRED: I did not construct an input that reaches it. My off-curve probes returned
`Ok(false)`, so I have no evidence the case is reachable at all.

**Fix.** Nothing, if S3 is fixed. Validating points at the top of `verify` closes this by
construction. Do not paper over it with `catch_unwind`.

### S9. LOW. `binfile::fq` accepts non-canonical field elements; `fr_normal` and `json_fq` do not

**What.** `crates/g16-zkey/src/binfile.rs:185-187` is `Fq::new_unchecked(bigint(b))` with no
range test, while `fr_normal` at `binfile.rs:192-197` correctly uses `Fr::from_bigint` and
rejects, and `json_fq` at `g16-zkey/src/lib.rs:396` also rejects.

**Where.** `crates/g16-zkey/src/binfile.rs:185-187`.

**Why it matters.** A stored `mont(x) + p` is arithmetically the same element under
Montgomery multiplication and passes `is_on_curve()`, but compares unequal to the canonical
point under `PartialEq`. That breaks vk hashing, point equality, and any two-decoder
cross-check (ours against snarkjs) that compares points rather than pairings. It also makes
the `(0,0)` infinity sentinel at `binfile.rs:217` and `binfile.rs:232` *miss* a non-canonical
zero, which fails safe (the point then fails `is_on_curve`) but fails confusingly. This is
trusted-setup input, so low severity. Confirms the research note's item 8.

**Fix.** Make `fq` return `Result<Fq, ZkeyError>` via `Fq::from_bigint`, matching `fr_normal`.
This touches `g1`, `g2`, `read_g1_section`, `read_g2_section` and the rayon closures, so it is
a real refactor, not a one-liner. If the per-point cost on the query sections is unacceptable,
gate it to the O(1) header points and `IC` only, which is where it actually matters.

## What is fine, plainly

- **Coordinates at or above the modulus are rejected, not reduced.** `json.rs:40-42`. This is
  the check that stops the research note's item 4 (substituting `s + r` for a public signal),
  and we pass where `ark-groth16` pushes the problem onto the caller. VERIFIED against the
  real artifact.
- **The G2 subgroup check is real and it works.** VERIFIED end to end: I sampled a point on
  the twist, multiplied by `#E'(Fq2)/10069` to land on genuine 10069-torsion (confirmed
  `[10069]T == 0` and `is_in_correct_subgroup_assuming_on_curve() == false`), added it to a
  real `pi_b`, and `read_g2` rejected it with `pi_b: point is not in the prime-order subgroup`.
  The comment at `json.rs:122-123` is accurate. Also confirmed `2p - r` is divisible by 10069.
- **The G1 subgroup call is a no-op and that is correct.** BN254 G1 has cofactor 1
  (`ark-bn254-0.5.0/src/curves/g1.rs:20-21`), so on-curve implies in-subgroup and the callee
  is a compile-time `true`. Keep the call: it is free after inlining and it is the right thing
  to write if the curve ever changes.
- **The public input count check is exact and is stricter than both arkworks and snarkjs.**
  `verify.rs:42-49`. Do not relax it.
- **The infinity sentinel translation is right.** `.zkey`'s `(0,0)` is converted to a real
  arkworks identity at `binfile.rs:217-221` and `binfile.rs:232-236` *before* any curve check,
  which is necessary because `(0,0)` does not satisfy `y^2 = x^3 + 3`. The writer emits
  ffjavascript's `["0","1","0"]` rather than `(0,0)` (`json.rs:58-60`), which is also right.
- **There is no `q - y` negation bug.** We negate through `-proof.a` at `verify.rs:26`, which
  is arkworks' `Neg` on `Affine` and handles the identity via the flag, not via arithmetic on
  `y`. The class of bug that bites the snarkjs Solidity template does not exist here.
- **The vkey JSON reader is strict.** `g16-zkey/src/lib.rs:326-377` checks `protocol`,
  `curve`, `IC.len() == nPublic + 1`, canonical `Fq`, `z` in `{0, 1}`, on-curve and subgroup
  for every point including all of `IC`. This is the good reader. `json.rs` should be brought
  up to it, not the other way round.

## What CUDA must do differently

Nothing in `verify.rs` should ever run on a device, and none of the above is a CUDA bug today
(the crate is a stub). Four things the CUDA backend must not get wrong, all of which the Metal
backend already gets right and can be copied from.

1. **Translate the `(0,0)` infinity sentinel on the host before upload.** `g16-metal/src/layout.rs:340-344`
   reads the arkworks `infinity` flag rather than testing the coordinates, and
   `layout.rs:151-157` documents why: real A, B and C query sections do contain points at
   infinity, and a Jacobian addition formula lifts `(0,0)` into a live non-identity point that
   poisons whichever bucket it lands in. A CUDA backend that `memcpy`s `.zkey` section bytes
   straight into device memory inherits `(0,0)` with no flag and will silently produce wrong
   MSM results, which surface as an unverifiable proof with nothing else to go on. Use the
   same all-zero-means-infinity packed convention (`layout.rs:141-151`) and the same
   flag-driven pack.
2. **Fully reduce Montgomery limbs before any host-visible comparison.** Lazily reduced
   (non-canonical) limbs are arithmetically fine on device but break every equality test:
   the device-side `is_infinity()` idiom `x == 0 && y == 0` is wrong on non-canonical limbs,
   and cross-backend equality assertions against the CPU reference will fail spuriously.
   This is finding S9 restated as a device concern.
3. **Normalise MSM output to affine on the host and write `z = 1`.** Do not emit raw Jacobian
   into `proof.json` on the theory that our own reader accepts any `z` (S2). It does today;
   snarkjs and rapidsnark are not obliged to, and the three-oracle property is the point of
   the JSON format.
4. **Do not add a "GPU is fast, skip validation" key-loading path.** The existing tradeoff at
   `g16-zkey/src/lib.rs:153-158` (validate the O(1) points, skip the query sections) is
   already the aggressive one. If CUDA ever grows its own `.zkey` reader or a device-side
   pairing, the G2 subgroup check must stay on the host, before any kernel launch: it is a
   Frobenius endomorphism plus a scalar multiplication, exactly the kind of thing that is easy
   to fuse into a kernel and get subtly wrong, and it is the single check on the whole
   verification path that is doing real work.

## Suggested order of work

1. S2, then S3. They are the two that change what our verifier accepts, and both are small.
2. S1 documentation and the re-randomisation regression test, which is 20 lines and pins a
   property everyone gets wrong.
3. S5 infinity rejection plus the four regression cases, and the checked-in adversarial
   `pi_b` the research note recommends (I built one; it is about 30 lines with `num-bigint`,
   and the twist point search terminated in 2 to 3 tries).
4. S6, S7 as a single "make the key type impossible to misuse" change.
5. S4, S8, S9 when convenient. S8 disappears for free once S3 lands.
