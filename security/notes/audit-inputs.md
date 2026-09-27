# Audit: untrusted inputs (`.zkey`, `.wtns`, `verification_key.json`)

Dimension: **inputs**. Companion to `research-inputs.md`, which established the threat
model and the prior art. This file is the audit of our own code against it.

Scope audited by reading, line by line:

* `crates/g16-zkey/src/binfile.rs` (253 lines, all of it)
* `crates/g16-zkey/src/lib.rs` (458 lines, all of it)
* `crates/g16-zkey/src/wtns.rs` (48 lines, all of it)
* supporting: `crates/g16-core/src/cpu.rs:56-146`, `crates/g16-core/src/prove.rs:8-60`,
  `crates/g16-field/src/lib.rs:34-75`, `crates/g16-cli/src/json.rs:31-44`, `Cargo.toml:49-57`

Scope audited by running: **60 crafted mutants** of `bench/artifacts/tiny_mul/circuit.zkey`
and `circuit.wtns`, plus two timing races against `bench/artifacts/js_16x16_d32/circuit.zkey`.
Binary under test: `target/release/g16`, built with `cargo build --release -p g16-cli`.
Every result below marked CONFIRMED was produced by running that binary. Everything marked
INFERRED was reasoned from source and is labelled as such.

No source file was modified.

## Verdict in one paragraph

The **container** layer (`binfile.rs`) is solid. Every length and count that reaches a slice
is bounds-checked against the actual mapping before use, the section-length add is
`checked_add`, there is exactly one `unsafe` in the crate and it is `Mmap::map`, and every
integer decode goes through `<[u8;N]>::try_into` so there is no alignment assumption a
crafted file can violate. 27 malformed-container mutants all produced clean `Result` errors
with no crash and no allocation blowup. The **semantic** layer above it is where the holes
are: one count (`domainSize`) is used to size an allocation before the length that would
refute it is checked, five sections are not validated at all, the affine `(0, 0)` infinity
encoding is accepted on points whose being-the-identity silently destroys zero knowledge,
and the section-4 decoder reads the same mmap bytes twice with the range check applied on
only the first read.

The headline result is finding 1: a zkey that loads without a warning, produces proofs that
**verify OK against the genuine untouched `vkey.json`**, and whose `pi_a` is bit-identical
across runs. Full ZK break, invisible to every verifier.

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | **CRITICAL** | Infinity-encoded `beta_g1` + `delta_g1` + section 6 kills blinding while proofs still verify | `binfile.rs:217,232`; `lib.rs:158-163` |
| 2 | **HIGH** | `domainSize` allocation bomb: 34 GB from a 4 KB file, because the refuting check runs later | `lib.rs:130-135,146,151,255,283` |
| 3 | **HIGH** | Sections 5, 6, 7, 8, 9 are never validated: invalid-curve / off-subgroup base attack | `lib.rs:147-151,153-157` |
| 4 | **MEDIUM** | `read_coefficients` reads the mmap twice; the second pass omits the `constraint` range check, giving a hard abort | `lib.rs:284-296` (esp. `292`) |
| 5 | **MEDIUM** | SIGBUS on concurrent truncation of a mapped key file, uncatchable | `binfile.rs:51-55` |
| 6 | **MEDIUM** | Base-field and section-4 scalar coordinates are never range-checked; non-canonical encodings load and behave path-dependently | `binfile.rs:185-187,202-204` |
| 7 | LOW | No `domain_size <= 2^TWO_ADICITY` bound at parse time | `lib.rs:130-135` |
| 8 | LOW | `Vec::with_capacity(n_sections)` reserves 103 GB of address space | `binfile.rs:73` |
| 9 | LOW | `expect_records` uses unchecked `n * stride`; `u64 -> usize` section length truncates on 32-bit | `binfile.rs:242,83` |
| 10 | LOW | Section 10 (phase-2 contributions) is never read, so zkey provenance is unchecked | `lib.rs:100-191` |
| 11 | INFO | `BadMagic` is reused for "file shorter than 12 bytes" and its text is hardcoded to "zkey" even for `.wtns` | `binfile.rs:57-59`, `lib.rs:78-79` |
| 12 | INFO | `panic = "abort"` means no parser panic is recoverable by a library consumer | `Cargo.toml:53` |

---

## 1. CRITICAL: a zkey that verifies perfectly and has no zero knowledge

**What.** `binfile::g1` (`binfile.rs:214-222`) and `binfile::g2` (`binfile.rs:226-237`) map
the affine pair `(0, 0)` to `G1Affine::identity()` / `G2Affine::identity()`. That mapping is
correct and necessary, because ffjavascript's `toRprLEM` really does write infinity that way.
The problem is what happens next: arkworks' `is_on_curve()` returns `true` for the identity
(it short-circuits on the infinity flag) and so does
`is_in_correct_subgroup_assuming_on_curve()`. So the six header checks at `lib.rs:158-163`
pass on an all-zero point, and `lib.rs:147-151` does not check the query sections at all.

Three of the quantities the prover blinds with are therefore attacker-settable to the
identity, and **none of the three appears in the verification key**:

```
prove.rs:45   let pi_a  = m.a_g1 + pk.alpha_g1 + pk.delta_g1 * r;
prove.rs:47   let pib1  = m.b_g1 + pk.beta_g1  + pk.delta_g1 * s;
prove.rs:48   let pi_c  = m.l_g1 + m.h_g1 + pi_a * s + pib1 * r - pk.delta_g1 * (r * s);
```

Set `delta_g1 = O` and `pi_a` loses its `r`. Set `beta_g1 = O` and the whole of section 6
(`b_g1_query`) to `O` as well, and `pib1` collapses to `O`, so `pi_c` reduces to exactly the
honest `L + H + s*pi_a`. The result is an honest proof at `r = 0` with `s` still random,
which satisfies the pairing equation.

**CONFIRMED by running.** Mutant `15_stealth2.zkey`: `beta_g1` (offset 188), `delta_g1`
(offset 508) and all 384 bytes of section 6 (offset 1744) zeroed in `tiny_mul/circuit.zkey`.
Three independent `g16 prove` runs, each verified against the **untouched original**
`bench/artifacts/tiny_mul/vkey.json`:

```
verify: OK | pi_a[0]= 1498819189192755330655081916  pi_b00= 1440103742751005  pi_c[0]= 1500803385703062
verify: OK | pi_a[0]= 1498819189192755330655081916  pi_b00= 2109543650718424  pi_c[0]= 1731242806728901
verify: OK | pi_a[0]= 1498819189192755330655081916  pi_b00= 1426316562412904  pi_c[0]= 1910975869352092
all pi_a identical: True
```

For contrast, the clean key gives a different `pi_a` on every run, and the weaker mutant
`07_delta_zero.zkey` (only `delta1`/`delta2` zeroed, per `research-inputs.md` finding 2)
freezes both `pi_a` and `pi_b` but fails the pairing check, so a victim would notice. The
`15_stealth2` variant is the one that matters: it is silent end to end.

**Why it matters.** `pi_a = alpha_g1 + sum_j w_j * A_j` becomes a deterministic function of
the full secret witness. The key holder, who knows the circuit and the `A_j`, gets a
witness-confirmation oracle: guess a witness, recompute `pi_a`, compare. For any witness with
low entropy in the fields that matter (a bit vector, a balance, an age, a password preimage,
a membership index) that is outright recovery, not just a distinguisher. And because
`beta_g1`, `delta_g1`, and `b_g1_query` are absent from the vkey, **no verifier can detect
this**: not ours, not snarkjs, not rapidsnark. The only thing that catches it is
`snarkjs zkey verify <r1cs> <ptau> <zkey>`, which byte-compares sections 3 to 7 against a
recomputed setup. This is the Aztec Plonk "0 bug" class (0xPARC ZK Bug Tracker entry 7).

**Fix.** Two independent gates, both cheap and both O(1):

1. In `ProvingKey::load`, after the six header points are decoded, reject the identity where
   it is meaningless. `alpha_g1`, `beta_g1`, `beta_g2`, `gamma_g2`, `delta_g1`, `delta_g2` are
   all `[x]_1` or `[x]_2` for a secret nonzero toxic-waste scalar, so **none of them may be the
   point at infinity in an honest setup**. Add `if p.is_zero() { return Err(...) }` alongside
   the existing `check_g1`/`check_g2` calls at `lib.rs:158-163`. This alone blocks the
   demonstrated attack, costs six comparisons, and cannot false-positive on a real snarkjs key.
2. Reject an all-identity query section. A section 5/6/7/8/9 in which every point is infinity
   is never a real setup output. A cheap version is "at least one non-identity point", checked
   during the existing `par_chunks_exact` pass with a `reduce`.

Neither is a substitute for the operator obligation to run `snarkjs zkey verify` on an
untrusted zkey, which should be stated in the README, but both close the silent path.

---

## 2. HIGH: `domainSize` allocation bomb, 34 GB from a 4 KB file

**What.** `lib.rs:114` reads `domain_size` as a raw `u32`. `lib.rs:130-135` checks only that
it is nonzero and a power of two, so up to `2^31` survives. `lib.rs:146` then calls
`read_coefficients`, which at `lib.rs:255` does

```rust
let mut counts = [vec![0u32; domain_size + 1], vec![0u32; domain_size + 1]];
```

and at `lib.rs:283` clones both of them:

```rust
let mut cursor = [row_ptr[0].clone(), row_ptr[1].clone()];
```

That is four `u32` vectors of `domain_size + 1` elements, `16 * domain_size` bytes, sized
entirely from a header field. The check that refutes the lie, `expect_records` on section 9,
does not run until `lib.rs:151`, five statements later.

**CONFIRMED by running.** Patched the 4-byte `domainSize` at file offset 120 of the 4 047-byte
`tiny_mul/circuit.zkey`. `/usr/bin/time -l ./target/release/g16 prove ...`:

| `domainSize` | max RSS | wall | outcome |
|---|---|---|---|
| `2^24` | 275 628 032 B (275 MB) | 0.09 s | `malformed section 9: expected 16777216 records of 64 bytes (1073741824), got 512` |
| `2^29` | 8 597 110 784 B (8.6 GB) | 2.92 s | same shape, section 9 |
| `2^31` | 34 366 930 944 B (34.4 GB) | 11.49 s | same shape, section 9 |

`34 366 930 944 = 4 * 2^31 * 4` exactly, which pins the allocation to `counts` plus the
`cursor` clone and confirms there is no other contributor. The error text naming **section 9**
proves the ordering: the cheap check that would have rejected the file in microseconds ran
after the 34 GB was already committed.

On this machine (large RAM plus macOS overcommit) it degrades to a slow error. On a smaller
host the allocator refuses, and a Rust allocation failure is an `abort`, not a recoverable
`Err`. With `panic = "abort"` set in `Cargo.toml:53` there is nothing a library consumer can
do about it either.

Note that `2^29` is already past `Fr::TWO_ADICITY = 28`, so that file can never produce a proof
under any circumstances, and we still spent 8.6 GB and 2.9 s finding out.

**Why it matters.** This is one 4-byte edit in a 4 KB file. Any service that accepts a
user-supplied zkey (multi-tenant proving, a CI job, a CLI pointed at a downloaded key) is a
memory-exhaustion DoS with a trivially cheap request. Same defect class as
**CVE-2024-50354 / GHSA-cph5-3pgr-c82g** (gnark Groth16 key deserialization OOM, CVSS 5.5).

**Fix.** Two lines, both before `read_coefficients` at `lib.rs:146`:

```rust
// 1. a domain larger than the 2-adicity of Fr cannot exist at all.
if domain_size > (1usize << Fr::TWO_ADICITY) {
    return Err(ZkeyError::Malformed { section: 2, reason: format!(
        "domain size {domain_size} exceeds the 2-adicity of Fr") });
}
// 2. size the allocation from bytes present, not from a claimed count.
expect_records(file.unique_section(9)?, domain_size, G1_BYTES, 9)?;
```

The second is the gnark fix pattern verbatim: never size an allocation from a file-supplied
count, size it from the bytes that are actually there and then check the count against them.
The section-9 length is the independent witness for `domain_size`, and it is already
computed a few lines later, so this is pure reordering plus one bound.

---

## 3. HIGH: sections 5 to 9 are never validated

**What.** `lib.rs:153-157` documents the omission deliberately, for performance:

```rust
// Only the O(1) points are validated on load. The query sections are millions of
// points on a real circuit and a subgroup check each would dominate key load, so
// they are checked in tests instead.
```

`read_g1_section` (`lib.rs:206-212`) and `read_g2_section` (`lib.rs:214-218`) run
`expect_records` and then `par_chunks_exact(...).map(g1)`, with no `is_on_curve` and no
subgroup check. Only section 3 (`ic`) is checked, at `lib.rs:164-166`.

**CONFIRMED by running.** Four mutants, all of which **proved successfully with no error**:

| mutant | edit | `prove` | `verify` against real vkey |
|---|---|---|---|
| `09_aquery_bitflip` | flip bit 0 of `a_query[0].x` (offset 1348) | rc=0 | `pi_a: point is not on the curve` |
| `09_aquery_one_zero` | `a_query[0] = (1, 0)` | rc=0 | `pi_a: point is not on the curve` |
| `09_bg2_bitflip` | flip bit 0 of `b_g2_query[0]` (offset 2140) | rc=0 | `pi_b: point is not on the curve` |
| `09_hquery_bitflip` | flip bit 0 of `h_query[0]` (offset 3060) | rc=0 | `pi_c: point is not on the curve` |
| `09_ic_bitflip` | flip bit 0 of `ic[0]` (offset 712) | **rc=1** | `malformed section 3: ic[0] is not a valid G1 point` |

The `ic` row is the control: section 3 is checked, so it fails at load. Everything else sails
through and emits an off-curve proof element. Incidentally `b_g2_query[0]` in `tiny_mul` is
64 zero bytes, i.e. the infinity encoding, so that bit flip turned an intended identity into
non-identity garbage, which is finding 1's mechanism seen from the other side.

**Why it matters.** The prover happily runs its MSM over points that are not on the curve.
That is the setup for an invalid-curve attack (Biehl-Meyer-Muller): the key holder sets
`a_query[j] = P` and everything else to the identity, picks `P` on a twist of smooth order,
and reads `w_j` mod a product of small primes out of `pi_a`. BN254 G1 has cofactor 1, so an
on-curve G1 base is safe; **G2 (section 7) has a large composite cofactor, so a plain
on-curve but off-subgroup base already leaks** with no off-curve trickery at all. This is
only reachable when the zkey and the witness have different owners, which is precisely
proving-as-a-service, and it composes with finding 1: zero the blinders first, then the
extraction is exact rather than statistical.

**Fix.** Do not add a per-point check to the hot path; the comment is right that it would
dominate load. Add a **batched, cached** validation instead:

* per section, draw random scalars `c_i` and check `is_on_curve` plus
  `(sum c_i * P_i)` landing in the correct subgroup. That is one extra MSM per section, once,
  not one check per point per proof. Off-curve points are caught by the cheap coordinate test,
  which is a couple of field multiplications each and parallelises inside the existing
  `par_chunks_exact`.
* gate it behind an explicit `ProvingKey::load_untrusted(path)` (or a `validate: bool`) so the
  trusted-key benchmark path keeps its current cost, and make the plain `load` the untrusted
  one so the safe choice is the default.
* cache the verdict next to the zkey so a long-running service pays it once.

Whatever the shape, keep the ordering in `check_g1`/`check_g2` (`lib.rs:220-238`):
`is_on_curve()` **first**, then `is_in_correct_subgroup_assuming_on_curve()`. The "assuming"
is load-bearing; **CVE-2025-30147** (Besu / gnark-crypto) was a consensus fork caused exactly
by treating the subgroup check as implying on-curve. Our current order is correct. Do not
invert it in the new code.

---

## 4. MEDIUM: `read_coefficients` reads the mmap twice and range-checks only the first read

**What.** `read_coefficients` (`lib.rs:246-303`) is a counting sort in two passes over the
same mapped bytes.

Pass 1, `lib.rs:256-265`, validates:

```rust
let (m, constraint, _, _) = coef_indices(data, i)?;
if constraint >= domain_size { return Err(...); }
counts[m][constraint + 1] += 1;
```

Pass 2, `lib.rs:284-296`, re-reads the same records and validates only `sig`:

```rust
let (m, constraint, sig, off) = coef_indices(data, i)?;
if sig >= n_vars { return Err(...); }
let slot = cursor[m][constraint] as usize;   // <-- line 292, constraint NOT re-checked
```

`data` is a slice of a `MAP_SHARED` mapping. The single-read assumption that makes line 292
safe is exactly the assumption an attacker who owns the file can break.

**CONFIRMED by running.** Started `g16 prove` on a copy of the 94 MB
`js_16x16_d32/circuit.zkey` while a second process `pwrite`-flipped record 0's `constraint`
field (file offset 2972) between `0xFFFFFFFF` and its real value in a tight loop for 0.9 s.
Six trials:

```
trial1 rc=134 :: index out of bounds: the len is 262145 but the index is 4294967295
trial2 rc=0
trial3 rc=134 :: index out of bounds: the len is 262145 but the index is 4294967295
trial4 rc=0
trial5 rc=134 :: index out of bounds: the len is 262145 but the index is 4294967295
trial6 rc=134 :: index out of bounds: the len is 262145 but the index is 4294967295
```

`262145 = 2^18 + 1 = domain_size + 1`, which is the length of `cursor[m]`, and the reported
index is `0xFFFFFFFF` with no `+ 1`, so the panic is at `lib.rs:292` and not at the pass-1
line `counts[m][constraint + 1]`. If pass 1 had seen the bad value it would have returned the
clean `constraint ... is outside domain size` error, which is what the static mutant
`11_coef_constraint_oob.zkey` produces. `rc = 134` is `SIGABRT`: with `panic = "abort"` at
`Cargo.toml:53` this is a hard process kill, not a `Result`, not even an unwind.

**Why it matters.** Two consequences, in order of how much they should bother us:

* **Silent CSR corruption.** The loud case is the out-of-range one above. The quiet case is a
  `constraint` that changes to a *different in-range* value between the passes. Then
  `cursor[m][constraint]` is a valid index, `slot` comes from the wrong row, and the write at
  `signal[m][slot]` lands inside another row's region. No bounds check fires, no error is
  returned, and the key loads with a quietly wrong QAP. That produces a proof that fails to
  verify (best case) or, if an attacker can steer it, a QAP that is not the circuit anyone
  audited.
* **Uncatchable DoS.** The loud case aborts a proving service outright.

Bounds checking does mean this is not memory unsafety. It is a correctness and availability
bug, not a UB bug.

**Fix.** Cheapest and best: **stop reading the mapping twice**. Decode section 4 once into an
owned `Vec<(u8, u32, u32, Fr)>` (or three parallel vectors), validate during that single pass,
and run both counting-sort passes over the owned copy. The extra memory is `n_coefs * 44`
bytes, which is the size of the section we already mapped, and the parse is already O(n) so
there is no asymptotic cost. If the copy is unacceptable, the minimum patch is to repeat the
`constraint >= domain_size` check inside pass 2 at `lib.rs:291`, which converts the abort into
a clean `Err` but does **not** fix the silent-corruption case.

This is the same property finding 5 depends on, stated positively: **the mapping is read
exactly once, into owned storage, and never consulted again**. Sections 3 and 5 to 9 already
satisfy it. Section 4 is the only one that does not.

---

## 5. MEDIUM: SIGBUS on concurrent truncation, and it is not catchable

**What.** `binfile.rs:51-55`:

```rust
let file = std::fs::File::open(path)?;
// Safety: we only ever hand out shared slices of the mapping, and the mapping
// outlives them. A concurrent truncation of the file would be UB, which is the
// standard and unavoidable caveat of mmap on any key file.
let map = unsafe { Mmap::map(&file)? };
```

The safety comment is honest and correct. Touching a page of a mapping that is past the
file's new end raises `SIGBUS` on every POSIX system; `mmap(2)` says so and memmap2's own docs
say file-backed constructors are `unsafe` for precisely this reason.

**CONFIRMED by running, against our own binary.** Copied the 94 MB
`js_16x16_d32/circuit.zkey`, started `g16 prove` on the copy, and truncated the copy to 4096
bytes after a delay:

```
rc=138 delay=0.02
rc=138 delay=0.05
rc=0   delay=0.10
rc=0   delay=0.20
```

`rc = 138 = 128 + 10 = SIGBUS`. No message on stderr, no error, no unwinding, no `Drop`.
The delay sweep is itself useful evidence: past ~0.1 s the load has completed and the crash
stops being reachable, which matches the structural claim below.

**Why it matters, and what already limits it.** Two mitigating properties, both **verified by
reading**, and both worth protecting as invariants:

1. `ProvingKey` has no lifetime parameter (`lib.rs:25-49`) and `BinFile` is a **local** inside
   `load()` (`lib.rs:101`). Every section is copied into an owned `Vec` before `load` returns.
   So the exposure window is the duration of the `load()` call, not the lifetime of the key.
   The delay sweep above is the empirical confirmation.
2. No decoder ever forms a `*const u32` or `*const u64` into the mapping. `bigint`
   (`binfile.rs:176-182`) goes through `u64_at`, which is
   `u64::from_le_bytes(b[off..off+8].try_into()...)`, a byte-slice copy with no alignment
   requirement. This is a genuine advantage over rapidsnark's `binfile_utils.cpp`, which does
   type-punned unaligned `*(u_int32_t*)(addr + pos)` loads at attacker-chosen offsets.

**Fix.** SIGBUS on a file another party can truncate is not fully fixable while mmap is used,
and mmap is the right call for a 100 MB to multi-GB key. What is fixable is the blast radius:

* document the operator obligation: a zkey being loaded must not be writable by anyone other
  than the operator, and should be on a path the operator controls for the duration of the
  load. That is one README line and it is the actual mitigation.
* keep property 1 above as a hard invariant. Never let `ProvingKey` (or any future
  zero-copy variant) borrow the `Mmap`; the moment it does, the window goes from milliseconds
  to the process lifetime and the SIGBUS becomes reachable during proving.
* fix finding 4, which removes the only double-read and therefore the only TOCTOU window that
  is worse than SIGBUS.
* if a hardened mode is ever wanted, `read()` the file into a `Vec` instead of mapping it,
  behind a flag. It costs one full copy and removes the class entirely.

---

## 6. MEDIUM: field coordinates are never range-checked

**What.** `binfile::fq` (`binfile.rs:185-187`) is

```rust
pub fn fq(b: &[u8]) -> Fq { Fq::new_unchecked(bigint(b)) }
```

`bigint` accepts any 256-bit value and `new_unchecked` installs the limbs as the Montgomery
representation without reduction. BN254's `q` is about 254 bits, so roughly four distinct
256-bit values map to each field element and all four are reachable from the file.
`fr_double_montgomery` (`binfile.rs:202-204`) has the identical gap for section-4 scalars.
Only `fr_normal` (`binfile.rs:192-197`), used for `.wtns`, goes through `Fr::from_bigint` and
rejects a value at or above the modulus. The asymmetry is deliberate but undocumented.

Arkworks' Montgomery multiplication for BN254 uses the no-carry optimisation, which is only
correct for inputs below the modulus, so `is_on_curve()` on a non-canonical value is testing
arithmetic that is outside its own contract.

**CONFIRMED by running.** Two experiments.

*Encoding malleability.* Replaced `alpha_g1`'s stored Montgomery `x` and `y` with `x + k*q`,
`y + k*q` (same field element, different bytes):

| k | `prove` | `verify` against real vkey |
|---|---|---|
| 1 | rc=0 | **OK** |
| 2 | rc=0 | **OK** |
| 3 | rc=0 | **OK** |
| 4 | rc=1 | `malformed section 2: alpha_g1 is not a valid G1 point` |

So for `k <= 3` the parser accepts the alternate encoding, `check_g1` passes, and the proof is
still correct. Every base-field coordinate in the file therefore has at least three additional
valid byte encodings. Across the six header points alone that is `>= 3^12` distinct files that
load to the identical key and produce identical proofs.

*Path-dependent corruption.* Applied the same `+q` bump to a query base:

| mutant | `prove` | `verify` |
|---|---|---|
| `13_aquery_noncanon` (`a_query[0].x += q`, `.y += q`) | rc=0 | **OK** |
| `13_hquery_noncanon` (`h_query[0].x += q`, `.y += q`) | rc=0 | `pi_c: point is not on the curve` |
| `13_coef_plus_r` (section-4 coefficient `+= r`) | rc=0 | `pairing check failed` |
| `11_coef_value_max` (section-4 coefficient `= 2^256 - 1`) | rc=0 | `pairing check failed` |

The same transformation is harmless in one section and produces an off-curve output in
another. I have **not** verified the mechanism; the plausible explanation is that
`a_query[0]`'s scalar is `w[0] = 1`, which `g16-msm` routes through a single mixed addition
(the fast path documented at `crates/g16-msm/src/lib.rs:9-13`), while `h_query[0]` goes through
bucket accumulation and doublings. That is INFERRED, not confirmed. What is confirmed is the
observable: non-canonical input is not uniformly tolerated.

**Why it matters.** No confirmed break, which is why this is MEDIUM and not higher. But three
concrete consequences:

* **Hash-based key identity is defeated.** "Is this the zkey I audited?" answered by hashing
  the file is wrong: at least `3^12` files answer differently and prove identically. Anyone
  building key pinning, dedup, or a content-addressed key store on top of `g16-zkey` needs to
  know this.
* **`is_zero()` and `PartialEq` compare limbs, not values.** The infinity detection at
  `binfile.rs:217` and `binfile.rs:232` is `x.is_zero() && y.is_zero()`, which is a limb
  comparison. A coordinate encoded as `q` (value 0, non-canonical) is not detected as
  infinity. That direction happens to fail closed here (`new_unchecked(q, q)` fails
  `is_on_curve`, verified by the `k = 4` row above failing), but any future equality or
  zero test on parser output inherits the trap.
* **The GPU backends inherit it.** See "What CUDA must do differently".

**Fix.** Make `fq` and `fr_double_montgomery` fallible and reject at or above the modulus,
mirroring what `fr_normal` already does:

```rust
pub fn fq(b: &[u8], section: u32) -> Result<Fq, ZkeyError> {
    let bi = bigint(b);
    if bi >= Fq::MODULUS { return Err(ZkeyError::Malformed { section,
        reason: "coordinate is not below the base field modulus".into() }); }
    Ok(Fq::new_unchecked(bi))
}
```

Note this is a **comparison against `MODULUS`, not a `from_bigint` call**: `from_bigint`
would also convert out of Montgomery form, which is wrong here because the stored value
already is the Montgomery representation. One 4-limb comparison per coordinate, which is
noise next to the point decode it sits inside. Do it for the header and `ic` unconditionally;
for the query sections it folds naturally into the batched validation of finding 3.

---

## 7. LOW: no `TWO_ADICITY` bound at parse time

`lib.rs:130-135` checks `domain_size != 0 && domain_size.is_power_of_two()` and nothing more.
The real bound, `log2(domain_size) <= Fr::TWO_ADICITY = 28`, is enforced only in
`Domain::new` (`g16-field/src/lib.rs:48-51`), called from `CpuBackend::prepare`
(`cpu.rs:60-66`), which is after the whole key has been parsed and allocated. **CONFIRMED**:
`01_ds_2p29.zkey` spent 8.6 GB and 2.92 s before erroring, on a domain size that can never
have a primitive root. Fix is the first half of finding 2's patch.

## 8. LOW: `Vec::with_capacity(n_sections)`

`binfile.rs:73` reserves `n_sections * 24` bytes from an unvalidated `u32`, up to 103 GB.
**CONFIRMED harmless in practice**: `03_nsections_max.zkey` (`nSections = 0xFFFFFFFF`)
measured **6 635 520 B max RSS, 0.00 s**, erroring cleanly with
`malformed section 0: section header 10 runs past end of file`. It is address space only on
64-bit. Still worth capping, because it costs one line and because a host with overcommit
disabled turns it into an abort: `Vec::with_capacity(n_sections.min(64))`, or just
`Vec::new()`, since a real binfile has ten sections and the push loop reallocates a handful of
times at most.

## 9. LOW: two integer-width issues, both 32-bit-only

* `expect_records` (`binfile.rs:242`) computes `let want = n * stride;` unchecked. On 64-bit
  this cannot wrap: the largest reachable product is `n_vars_max * G2_BYTES = 2^32 * 128`,
  about `5.5e11`, comfortably inside `usize`. **CONFIRMED**: `02_nvars_max.zkey` printed
  `expected 4294967295 records of 64 bytes (274877906880)` with no wrap and 7 MB RSS. On a
  32-bit target it wraps and a lying count could be made to match a short section.
  `checked_mul` costs nothing.
* `binfile.rs:83`, `let len = u64_at(&map, pos + 4) as usize;` truncates a `u64` to 32 bits on a
  32-bit target, so a declared length of `0x1_0000_0000` reads as `0`. On 64-bit it is exact,
  and the subsequent `checked_add` plus `end > map.len()` catch everything. **CONFIRMED** on
  64-bit: `04_seclen_u64max` gives `section length overflows`, `04_seclen_4gib` gives
  `length 4294967296 runs past end of file`. Use `usize::try_from(u64_at(..))` and error on
  failure.

Neither is exploitable on any target we build for today. Listed so the 32-bit assumption is
written down rather than implicit.

## 10. LOW: section 10 is never read

`ProvingKey::load` reads sections 1 to 9 and ignores section 10, the phase-2 contribution
chain. Nothing in our prover, and nothing in Groth16 verification, constrains sections 4 to 9
against the r1cs or the powers-of-tau. `snarkjs zkey export verificationkey` derives the vkey
*from the zkey*, so "it verifies against my vkey" is circular and proves nothing about an
untrusted zkey. The only non-circular check is `snarkjs zkey verify <r1cs> <ptau> <zkey>`,
which re-runs the setup in memory and byte-compares sections 3 to 7 plus `sameRatio` pairings
for 8 and 9 plus the full hash chain. That is far too expensive to do inline. The fix is
documentation, not code: state in the README that validating a third-party zkey is the
operator's obligation and name the exact command.

## 11. INFO: two error-reporting papercuts

* `binfile.rs:57-59` returns `ZkeyError::BadMagic([0; 4])` for any file shorter than 12 bytes,
  so an 11-byte file reports `not a zkey file (bad magic [0, 0, 0, 0])`. **CONFIRMED**
  (`05_trunc_11`). Misleading but harmless; a distinct `TooShort` variant would be clearer.
* `ZkeyError::BadMagic`'s message at `lib.rs:78-79` is hardcoded to "not a zkey file", but the
  same type is used by the `.wtns` reader. **CONFIRMED**: a zero-byte `.wtns` reports
  `not a zkey file (bad magic [0, 0, 0, 0])`. Carry the expected magic in the variant.

## 12. INFO: `panic = "abort"` removes the safety net

`Cargo.toml:53` sets `panic = "abort"` for the release profile. Every finding above that ends
in a panic (finding 4, and any residual indexing bug) is therefore an immediate `SIGABRT` in a
release build, with no unwind and no `catch_unwind` available to a service embedding
`g16-zkey` as a library. This is the right choice for a prover binary and the wrong one for a
library, and it raises the severity of every panic in the parser by one notch. Worth a
sentence in the crate docs so an embedder knows what they are getting.

---

## Answers to the four specific questions

### (a) Every length and count read from the file, and its check

Container, `binfile.rs`:

| Field | Read at | Checked against the mapping? |
|---|---|---|
| file length >= 12 | `57` | yes, before any header read |
| magic (4 B) | `60-63` | yes, byte-compared |
| `version` (u32) | `64-70` | yes, `<= max_version` |
| `nSections` (u32) | `71` | **not bounded**, but only drives the loop trip count and `Vec::with_capacity` (finding 8); every iteration is bounds-checked at `76` |
| section header `pos + 12` | `76` | yes, `> map.len()` rejects |
| `sectionLength` (u64) | `83` | yes: `checked_add` at `85`, then `end > map.len()` at `89`. Truncates on 32-bit (finding 9) |
| slice `map[start..start+len]` | `114` | yes, `start + len == end <= map.len()` from the `open` loop invariant |
| `Cursor::take(n)` | `137-145` | yes, `checked_add` then `end > data.len()` |

zkey semantics, `lib.rs`:

| Field | Read at | Checked against the mapping? |
|---|---|---|
| `protocol` (s1) | `104` | yes, `== 1` |
| `n8q`, `q` bytes | `110` | yes, length and byte-compare (both `q` **and** `r`, unlike rapidsnark which only checks `r`) |
| `n8r`, `r` bytes | `111` | yes |
| `nVars` (u32) | `112` | **not bounded directly**; refuted by `expect_records` on sections 5, 6, 7 at `147-149`. No allocation is sized from it before then, so this is safe. CONFIRMED (`02_nvars_max`, 7 MB RSS) |
| `nPublic` (u32) | `113` | yes, `n_vars >= n_public + 1` at `138-143`, then section 3 length at `145` and section 8 length at `150` |
| `domainSize` (u32) | `114` | **INADEQUATE.** Only nonzero-and-power-of-two at `130-135`. Sizes four allocations at `255` and `283` before the refuting section-9 length check at `151`. **Finding 2** |
| six header points | `117-122` | yes, `Cursor::take` bounds each, and `s2.remaining() != 0` at `123-128` rejects trailing bytes |
| section 3 length | `145` | yes, `expect_records(_, n_public + 1, 64)` |
| `nCoefs` (u32, s4) | `252` | yes, `expect_records(&data[4..], n_coefs, 44)` at `253`. `&data[4..]` cannot panic because `c.u32()` at `252` already proved `data.len() >= 4` |
| s4 `matrix` per record | `311-319` | yes, `m > 1` rejected |
| s4 `constraint` per record | `257-263` | yes in **pass 1**; **missing in pass 2** at `292`. **Finding 4** |
| s4 `signal` per record | `286-291` | yes in pass 2; re-checked again in `cpu.rs:99-102` |
| s4 record slicing `data[base..base+44]` | `306-321` | yes, implied by `expect_records` at `253`: `base + 44 <= 4 + n_coefs*44 == data.len()` |
| sections 5, 6, 7, 8, 9 lengths | `147-151` | yes, `expect_records` for each |
| section 5-9 point **contents** | `211`, `217` | **no validation at all. Finding 3** |
| section 10 | never read | n/a. **Finding 10** |

wtns, `wtns.rs`:

| Field | Read at | Checked? |
|---|---|---|
| magic `wtns`, version <= 2 | `17` | yes, via `BinFile::open` |
| `n8` (u32) | `20-23` | yes, `== 32` |
| modulus bytes | `24-26` | yes, byte-compared against `Fr::MODULUS` |
| `nWitness` (u32) | `27` | yes, `expect_records(data, n_witness, 32, 2)` at `30`. No allocation precedes it |
| each 32-byte value | `31-34` | yes, `fr_normal` rejects `>= r` rather than reducing |
| `witness[0] == 1` | `39-44` | yes |
| `nWitness` vs zkey `nVars` | not here | checked one layer up at `cpu.rs:139-146`. CONFIRMED (`w2_extra_var`: `witness has 7 entries, proving key expects 6`) |

### (b) Unchecked arithmetic on offsets

Four sites, none exploitable on a 64-bit target:

1. `binfile.rs:76`, `pos + 12 > map.len()`. Unchecked add. The loop invariant from `89-96`
   keeps `pos <= map.len()`, so overflow needs `map.len() > usize::MAX - 12`. Not reachable.
2. `binfile.rs:83`, `u64 as usize`. Lossless on 64-bit; truncates on 32-bit. Finding 9.
3. `binfile.rs:242`, `n * stride`. Max reachable product about `5.5e11`. Wraps on 32-bit only.
   Finding 9.
4. `lib.rs:307`, `4 + i * COEF_RECORD`. Bounded by `expect_records` at `253`, which already
   proved `n_coefs * 44 + 4 == data.len()`, so `i * 44` is at most a file length. Safe.

Also checked and clean: `lib.rs:145` `n_public + 1` (`n_public` is a `u32` widened to `usize`,
so `+1` cannot wrap on 64-bit, and `138-143` runs first anyway); `lib.rs:150`
`n_vars - n_public - 1` (guarded by `138-143`, **CONFIRMED** by `02_npub_gt_nvars` and
`02_npub_max` both erroring cleanly); `lib.rs:255` `domain_size + 1` (`domain_size` is a
power-of-two `u32`, at most `2^31`); `lib.rs:264` and `269-273`, the `u32` histogram and prefix
sum (bounded by `n_coefs <= u32::MAX`, and reaching a wrap would need a 189 GB file).

### (c) `unsafe`, `from_raw_parts`, `transmute` over mmap bytes

**Exactly one `unsafe` in the whole of `g16-zkey`**, verified by grep over `crates/g16-zkey/src/`:

```
crates/g16-zkey/src/binfile.rs:55:        let map = unsafe { Mmap::map(&file)? };
```

No `from_raw_parts`, no `transmute`, no `align_to`, no `as *const T` anywhere in the crate.

The Montgomery reads, which is where the question points, are clean on the alignment axis.
`bigint` (`binfile.rs:176-182`) builds its four limbs with
`u64::from_le_bytes(b[off..off+8].try_into()...)`, a byte copy. `u32_at` and `u64_at`
(`binfile.rs:167-173`) do the same. There is no pointer cast into the mapping at any offset,
so a crafted file cannot produce a misaligned load, and the decoders work identically on a
strict-alignment target. This is a real advantage over rapidsnark's `binfile_utils.cpp`, which
type-puns `*(u_int32_t*)(addr + pos)` at file-controlled offsets.

Size assumptions are all discharged by the callers. `bigint` needs 32 bytes and is reached
only from `fq` (fed exact 32-byte subslices by `g1`/`g2`, themselves fed exact 64/128-byte
chunks by `par_chunks_exact`), from `fr_normal` (fed by `par_chunks_exact(32)`), and from
`fr_double_montgomery` (fed `&data[off..off + FR_BYTES]` where `off` is bounded by
`expect_records`). `g1` and `g2` slice fixed offsets inside their chunk. All verified by
reading; no crafted file changes any of these lengths, because `par_chunks_exact` and
`expect_records` fix them before the decoder sees a byte.

What the single `unsafe` does *not* cover is finding 5 (the file changing under us), which is
a property of mmap and not of any cast.

### (d) Behaviour on the four named malformed inputs

All CONFIRMED by running the release binary.

**Truncated file.** Clean `Err` at every truncation point, no crash, no allocation:

```
0 bytes    not a zkey file (bad magic [0, 0, 0, 0])          <- also finding 11
4 bytes    not a zkey file (bad magic [0, 0, 0, 0])
11 bytes   not a zkey file (bad magic [0, 0, 0, 0])
12 bytes   malformed section 0: section header 0 runs past end of file
23 bytes   malformed section 0: section header 0 runs past end of file
40 bytes   malformed section 2: length 660 runs past end of file
700 bytes  malformed section 0: section header 2 runs past end of file
3000 bytes malformed section 8: length 128 runs past end of file
```

Same for `.wtns` (`w3_trunc_*`): clean errors at 0, 12, 24, 60, 100, 267 bytes.
The one thing truncation *does* break is truncation of a file already mapped, which is
finding 5.

**Zero-section file.** `nSections = 0` gives `missing section 1`. `nSections = 3` gives
`missing section 4`. Both clean, both immediate. Dropping sections 9 and 10 physically gives
`malformed section 0: section header 8 runs past end of file`.

**Section length `0xFFFFFFFFFFFFFFFF`.** `malformed section 1: section length overflows`,
from the `checked_add` at `binfile.rs:85`. Immediate, 6.5 MB RSS. Neighbouring values:
`0xFFFFFFFFFFFFFFF0` gives the same overflow error; `0x1_0000_0000` gives
`length 4294967296 runs past end of file`; `0` gives
`length 2834678415362 runs past end of file` (the zeroed length makes the parser read the
*next* header at the wrong offset, which is exactly the desired failure and is caught by the
same bound). No crash in any case.

**`domainSize` beyond `Fr::TWO_ADICITY`.** Rejected, but **only after the allocation**. See
finding 2's table: `2^29` costs 8.6 GB and 2.92 s and `2^31` costs 34.4 GB and 11.49 s before
the section-9 length check fires. `domain_size = 0` and `domain_size = 3` are both rejected
immediately at 6.5 MB by the power-of-two test. This is the single most valuable fix in the
report after finding 1.

---

## Full mutation matrix

60 mutants run against `target/release/g16`. Every one either produced a clean `Result` error
or is listed as an accepted-but-wrong key below. **No mutant produced memory unsafety.** The
only hard crashes were the two deliberate races (findings 4 and 5), which require write access
to the file while it is mapped.

**Rejected cleanly (39):** `01_ds_zero`, `01_ds_three`, `01_ds_2p24`, `01_ds_2p29`,
`01_ds_2p31` (the last three only after the allocation, finding 2), `02_nvars_max`,
`02_npub_max`, `02_npub_gt_nvars`, `03_nsections_zero`, `03_nsections_three`,
`03_nsections_max`, `04_seclen_u64max`, `04_seclen_nearmax`, `04_seclen_4gib`,
`04_seclen_zero`, `05_trunc_{0,4,11,12,23,40,700,3000}`, `06_badmagic`, `06_badversion`,
`09_ic_bitflip`, `10_dup_section2`, `11_ncoefs_max`, `11_ncoefs_zero`,
`11_coef_constraint_oob`, `11_coef_signal_oob`, `11_coef_matrix2`, `12_missing_sec9`,
`13_alpha_plus4q`, `w1_bad_modulus`, `w1_n8_max`, `w1_nwitness_max`, `w1_nwitness_zero`,
`w2_w0_not_one`, `w2_value_eq_r`, `w2_value_max`, `w2_extra_var`, `w3_trunc_*`, `w4_dup_sec2`.

**Accepted, produced a proof that fails verification (7):** `07_alpha_zero`, `07_delta_zero`,
`09_aquery_bitflip`, `09_aquery_one_zero`, `09_bg2_bitflip`, `09_hquery_bitflip`,
`13_hquery_noncanon`, `13_coef_plus_r`, `11_coef_value_max`, `14_stealth_zk_break`.
Loud failure, but the key loaded silently, which is finding 3 and finding 6.

**Accepted, produced a proof that verifies (5):** `08_alpha_noncanon_x`,
`08_alpha_noncanon_y`, `13_alpha_plus2q`, `13_alpha_plus3q`, `13_aquery_noncanon`
(finding 6, encoding malleability), and **`15_stealth2`** (finding 1, the ZK break).

---

## What CUDA must do differently

The CUDA lane inherits the parser wholesale, so findings 1 to 3 apply unchanged. Four things
are specifically worse or specifically different on GPU.

1. **Every finding-6 non-canonical field element gets shipped to device memory.**
   `g16-gpu-layout` packs `Fq`/`Fr` limbs and hands them to the device via
   `as_bytes::<Packed>` (`crates/g16-gpu-layout/src/lib.rs:438-466`), and `to_fq`/`to_fq2` on
   the way back are `new_unchecked` (`lib.rs:301, 370, 414`). A device-side Montgomery CIOS
   multiply makes the same "input is below the modulus" assumption arkworks does, and it has
   no `debug_assert` and no bounds check to fall back on. The confirmed CPU observation that
   `+q` is harmless in one MSM path and produces an off-curve result in another
   (`13_aquery_noncanon` vs `13_hquery_noncanon`) is a warning that this is path-dependent;
   on GPU the divergent path may be a different kernel entirely. **Range-check `Fq` and `Fr`
   at parse time (finding 6's fix) before any of this is worth reasoning about**, because it
   is the only place a single check covers every backend.

2. **A GPU allocation bomb is worse than a host one.** Finding 2's `34 GB` was host RAM that
   the OS could overcommit and later reclaim. `cudaMalloc` sized off the same unvalidated
   `domainSize` fails hard, and the failure path in a CUDA backend is typically an error code
   that is easy to convert into an `unwrap`. Do the `domain_size <= 1 << Fr::TWO_ADICITY`
   bound and the section-9 cross-check **before** any device allocation, and never size a
   `cudaMalloc` from a header field that has not been cross-checked against a section length.

3. **Do not let device buffers borrow the mmap.** The property that keeps finding 5 to a
   millisecond window is that `ProvingKey` owns every byte and `BinFile` dies at the end of
   `load()` (`lib.rs:101`). It will be tempting to `cudaHostRegister` the mapping and DMA
   straight from it to skip a copy on a 1 GB zkey. That converts a load-duration SIGBUS window
   into a proving-duration one, and a SIGBUS taken inside a pinned-memory DMA is considerably
   less pleasant than one taken on a CPU load. If zero-copy upload is wanted, copy the section
   into pinned host memory first and upload from there.

4. **`(0, 0)` must mean infinity on the device too, and it must be the *same* decision.**
   `binfile::g1`/`g2` resolve the infinity encoding on the host at `binfile.rs:217, 232`, so
   whatever reaches the packing layer is already an arkworks `Affine` with its infinity flag
   set. `PackedG1Affine` must preserve that flag rather than re-deriving it from the
   coordinates on the device, and the CUDA MSM must skip infinity bases rather than feeding
   `(0, 0)` into a mixed-addition formula, where it is a valid-looking off-curve point.
   `g16-gpu-layout` already asserts `!G1Affine::new_unchecked(Fq::zero(), Fq::zero()).is_on_curve()`
   (`lib.rs:580, 598`), which is the right instinct; keep that assertion and add the
   round-trip test that an infinity base survives host to device to host unchanged. Note that
   `b_g2_query[0]` in `bench/artifacts/tiny_mul` is genuinely all zeros, so this path is
   exercised by the smallest artifact we have. Use it.

---

## What I did not verify

* I did not verify **why** `13_aquery_noncanon` verifies while `13_hquery_noncanon` does not.
  The MSM fast-path explanation is inferred from `crates/g16-msm/src/lib.rs:9-13`, not
  confirmed by instrumenting the MSM.
* I did not attempt an actual invalid-curve witness extraction (finding 3). I confirmed that
  off-curve and identity bases are accepted and that the resulting proof elements are off
  curve; the recovery step is from the literature (Biehl-Meyer-Muller), not run here.
* I did not test the 32-bit truncation and wrap paths in finding 9. There is no 32-bit target
  in this workspace; both are read-only conclusions from the source.
* I did not test `verification_key.json` parsing beyond reading it. `lib.rs:386-455` and
  `g16-cli/src/json.rs:31-44` both range-check against the modulus before constructing, and
  `json_g1`/`json_g2` call `check_g1`/`check_g2`, so the JSON path looks materially stricter
  than the binary path. That is a reading, not a fuzzing result.
* I did not reproduce the silent-CSR-corruption half of finding 4 (an in-range `constraint`
  flip between the two passes). I reproduced the out-of-range half, which proves the check is
  missing; the in-range consequence follows from the code but was not observed.
* Reproduction files and the crafting scripts are in the session scratchpad at
  `/private/tmp/claude-501/-Users-sohamzemse-workspace-research/70a8845a-3b6f-45eb-be7b-0ecbbc2ff8b2/scratchpad/mut/`.
  They are throwaway; the two Python crafting scripts are short enough to reconstruct from the
  offsets given above (section 2 starts at byte 40; `nVars` at 112, `nPublic` at 116,
  `domainSize` at 120, `alpha1` at 124, `beta1` at 188, `beta2` at 252, `gamma2` at 380,
  `delta1` at 508, `delta2` at 572; sections 3/4/5/6/7/8/9 start at 712/980/1348/1744/2140/2920/3060).
