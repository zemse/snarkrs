# Malformed and adversarial input handling: `.zkey` and `.wtns`

Research note for the `inputs` dimension. Scope: everything the prover parses from a file an
attacker may control. **Report only** — no source file was modified.

Every claim below is tagged:

- **[VERIFIED-CODE]** — read out of our source at the cited path/line.
- **[VERIFIED-RUN]** — I built a malformed file and ran `target/release/g16` (or `python3`) against
  it in this session; the observed output is quoted.
- **[SOURCE]** — read out of an upstream file / spec / advisory at the cited URL.

Reproduction files from this session live in the session scratchpad
`/private/tmp/claude-501/-Users-sohamzemse-workspace-research/70a8845a-3b6f-45eb-be7b-0ecbbc2ff8b2/scratchpad/`
(`domain_bomb.zkey`, `zero_delta.zkey`, `offcurve_a.zkey`, `badpoint_a.zkey`, `nvars_bomb.zkey`,
`seclen_overflow.zkey`, `nsections_bomb.zkey`, `wtns_*.wtns`, `sigbus_demo.bin`). They are not in
the repo; the crafting script is reproduced inline in §7 so the tests can be regenerated.

---

## 1. The formats, from the authoritative source

There is no format RFC. The definition is the iden3 source, in three files.

### 1.1 The "binfile" container (shared by `.zkey`, `.wtns`, `.r1cs`, `.ptau`)

**[SOURCE]** `readBinFile` in
<https://github.com/iden3/binfileutils/blob/master/src/binfileutils.js>:

```js
const b = await fd.read(4);                       // 4 magic bytes, ASCII
let v = await fd.readULE32();                     // version
if (v>maxVersion) throw new Error("Version not supported");
const nSections = await fd.readULE32();           // u32 section count
for (let i=0; i<nSections; i++) {
    let ht = await fd.readULE32();                // u32 section id
    let hl = await fd.readULE64();                // u64 section length
    sections[ht].push({p: fd.pos, size: hl});
    fd.pos += hl;                                 // NOTE: no bound check at all
}
```

Facts that matter for a parser:

- Layout is `magic[4] | u32 version | u32 nSections | (u32 id, u64 len, payload)*`. All
  little-endian. Header is 12 bytes.
- **Section ids are neither ordered nor unique.** `sections[ht]` is an array. Duplicate ids are
  legal at the container level; `startReadUniqueSection` rejects duplicates only for the sections
  that reader happens to open (`if (sections[idSection].length>1) throw ... "Section Duplicated"`).
  A section id nobody reads can be duplicated freely, and a *trailing* duplicate of a section id
  can shadow nothing in snarkjs but could shadow something in a naive "first match wins" parser.
- **The reference reader performs no bounds check on `hl` whatsoever.** `fd.pos += hl` just walks
  off the end; the failure surfaces later as a short read. A parser must do this check itself.
- `endReadSection` cross-checks that the reader consumed exactly `size` bytes ("Invalid section
  size reading"), so a section that is *longer* than its declared records is a format error in
  snarkjs too, not merely slack.

### 1.2 `.zkey` (Groth16, protocol id 1)

**[SOURCE]** `zkey_utils.js` header comment and `writeHeader`/`writeZKey`/`readHeaderGroth16`,
<https://github.com/iden3/snarkjs/blob/master/src/zkey_utils.js>. Magic `"zkey"`; written with
version **1** and 10 sections (`createBinFile(zkeyName, "zkey", 1, 10, ...)` in `zkey_new.js`),
read with `maxVersion = 2` (`readBinFile(zkeyFileName, "zkey", 2, ...)` in `groth16_prove.js`).

| id | contents | length invariant |
|----|----------|------------------|
| 1  | `u32 protocolId` (1 = groth16, 2 = plonk, 10 = fflonk, per `zkey_constants.js`) | 4 bytes |
| 2  | `u32 n8q, q[n8q], u32 n8r, r[n8r], u32 nVars, u32 nPublic, u32 domainSize`, then `alpha1(G1) beta1(G1) beta2(G2) gamma2(G2) delta1(G1) delta2(G2)` | exactly `4+n8q+4+n8r+12+64+64+128+128+64+128` |
| 3  | IC | `(nPublic+1) * 64` |
| 4  | `u32 nCoefs`, then `nCoefs * (u32 matrix, u32 constraint, u32 signal, Fr[n8r])` | `4 + nCoefs*(12+n8r)` |
| 5  | A bases, G1 | `nVars * 64` |
| 6  | B1 bases, G1 | `nVars * 64` |
| 7  | B2 bases, G2 | `nVars * 128` |
| 8  | L bases, G1 (private wires only) | `(nVars - nPublic - 1) * 64` |
| 9  | H bases, G1 | `domainSize * 64` |
| 10 | phase-2 contribution chain | variable |

Note the header comment in `zkey_utils.js` lists the point order as
`alpha1, beta1, delta1, beta2, gamma2, delta2` — **the comment is wrong**; the code writes
`alpha1, beta1, beta2, gamma2, delta1, delta2`. Our reader follows the code
(`crates/g16-zkey/src/lib.rs:116-121`) **[VERIFIED-CODE]**, which is correct.

Encodings, all three of them, confirmed against the writer **[SOURCE]**:

- Curve coordinates: `curve.G1.toRprLEM` — raw Montgomery limbs, LE. Stored integer is `x·R mod q`.
- Section 4 values: `writeFr2(n)` does `n = Scalar.mod(Scalar.mul(n, R2r), zkey.r)` where
  `R2r = R²  mod r` — so the stored integer is `v·R² mod r`, **double** Montgomery.
- Point at infinity is written by ffjavascript as the affine pair `(0,0)`, which is not on the
  curve.

### 1.3 `.wtns`

**[SOURCE]** `wtns_utils.js`, <https://github.com/iden3/snarkjs/blob/master/src/wtns_utils.js>.
Magic `"wtns"`, read with `maxVersion = 2`.

- Section 1: `u32 n8, prime[n8], u32 nWitness`.
- Section 2: `nWitness * n8` bytes, **plain little-endian integers** (`writeBigInt` →
  `Scalar.toRprLE`), *not* Montgomery. This is the encoding asymmetry that silently corrupts
  proofs if you get it backwards.

### 1.4 The cross-field checks snarkjs itself performs before proving

**[SOURCE]** `groth16_prove.js` lines 28-47:

```js
if (zkey.protocol != "groth16") throw new Error("zkey file is not groth16");
if (!Scalar.eq(zkey.r,  wtns.q))  throw new Error("Curve of the witness does not match the curve of the proving key");
if (wtns.nWitness != zkey.nVars)  throw new Error(`Invalid witness length. Circuit: ${zkey.nVars}, witness: ${wtns.nWitness}`);
```

That is the *entire* validation snarkjs does. It never checks section lengths against the header,
never range-checks `domainSize`, never checks a single point.

**This is the minimum bar, and our parser already clears it and more** — see §2.

---

## 2. What our parser already gets right

All **[VERIFIED-CODE]**, paths relative to the repo root, plus **[VERIFIED-RUN]** where noted.

| Check | Where | Evidence |
|---|---|---|
| Magic + version ceiling | `crates/g16-zkey/src/binfile.rs:57-72` | |
| File shorter than 12-byte header rejected | `binfile.rs:57-59` | |
| Section header runs past EOF rejected | `binfile.rs:75-81` | |
| `pos + len` overflow rejected with `checked_add` | `binfile.rs:85-88` | **[VERIFIED-RUN]** `seclen_overflow.zkey` (len = `0xFFFF_FFFF_FFFF_FFF0`) → `malformed section 1: section length overflows` |
| Section end past EOF rejected | `binfile.rs:89-95` | |
| Duplicate section id rejected for *every* section we read | `binfile.rs:104-118` (stricter than snarkjs, which only checks it per-`startReadUniqueSection`) | |
| Cursor `take` is bounds- and overflow-checked | `binfile.rs:137-146` | |
| `q` and `r` compared byte-for-byte against BN254 | `lib.rs:110-111`, `lib.rs:195-203` (`check_modulus`); stricter than rapidsnark, which only checks `r` | |
| No trailing bytes in section 2 | `lib.rs:123-129` | |
| `domainSize` is a nonzero power of two | `lib.rs:130-136` | |
| `nVars >= nPublic + 1` (would underflow the section-8 length) | `lib.rs:138-144` | |
| **Every bulk section length is cross-checked against the header** (`expect_records`) | `binfile.rs:241-256`, called from `lib.rs:145-151` and `lib.rs:206-217` | **[VERIFIED-RUN]** `nvars_bomb.zkey` (nVars = `0xFFFFFFFF`) → `malformed section 5: expected 4294967295 records of 64 bytes (274877906880), got 384`, with no large allocation. This is the single most valuable check we have and neither snarkjs, rapidsnark nor ark-circom does it. |
| Section-4 `matrix` id must be 0 or 1 | `lib.rs:306-320` (`coef_indices`) | |
| Section-4 `constraint < domainSize` | `lib.rs:258-263` | |
| Section-4 `signal < nVars` | `lib.rs:286-291` (and again in `g16-core/src/cpu.rs:99`) | |
| Section-4 record count cross-checked against section length | `lib.rs:253` | stricter than rapidsnark, which *derives* `nCoefs` from the section length (`zkey_utils.cpp`: `h->nCoefs = f->getSectionSize(4) / (12 + h->n8r);`) and so cannot detect a disagreement |
| The six O(1) points and all of IC are on-curve and in the prime-order subgroup | `lib.rs:153-166` | on-curve is tested **before** subgroup, which is the correct order — see §5.3 |
| `.wtns` `n8 == 32` and prime == BN254 `r` | `wtns.rs:20-26` | |
| `.wtns` values rejected if `>= r` (no silent reduction) | `binfile.rs:192-198` (`fr_normal`) | **[VERIFIED-RUN]** `wtns_over_modulus.wtns` → `malformed section 2: field element is not below the scalar modulus` |
| `.wtns` `nWitness` cross-checked against section-2 length | `wtns.rs:30` | **[VERIFIED-RUN]** `wtns_badcount.wtns` (claims 100, holds 6) → `expected 100 records of 32 bytes (3200), got 192` |
| `witness[0] == 1` | `wtns.rs:39-45` | **[VERIFIED-RUN]** `wtns_bad_one.wtns` → `witness[0] is not 1` |
| `witness.len() == nVars` | `g16-core/src/cpu.rs:139-147`, `g16-metal/src/backend.rs:230` | |
| **Public signals / proof coordinates rejected if `>= modulus`** | `g16-cli/src/json.rs:34-44` (`parse_field`) | This is exactly CVE-2023-33252 (§4.1) and we are not vulnerable to it. |
| Proof points read from JSON are on-curve and in-subgroup | `g16-cli/src/json.rs:106-128` | **[VERIFIED-RUN]** verifying `offcurve_a`'s proof → `pi_a: point is not on the curve` |

The mmap window is also narrower than it looks: `BinFile` is a local in `ProvingKey::load` /
`Witness::load` and every section is copied into an owned `Vec` before it returns
(`lib.rs:145-151`, `wtns.rs:32-37`) **[VERIFIED-CODE]**. Nothing in `ProvingKey` borrows the
mapping. So the SIGBUS exposure of §3 is bounded to the duration of the load call, not to the
lifetime of the key. That is a real mitigating fact and worth keeping true.

---

## 3. mmap-specific hazards

### 3.1 SIGBUS on truncation-while-mapped is real, and it is not a Rust error

**[SOURCE]** POSIX `mmap` (<https://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html>):

> References within the address range starting at pa and continuing for len bytes to whole pages
> following the end of an object shall result in delivery of a **SIGBUS** signal. ... Memory access
> within the mapping but beyond the current end of the underlying objects may result in SIGBUS
> signals being sent to the process. **The reason for this is that the size of the object can be
> manipulated by other processes and can change at any moment.**

**[SOURCE]** Linux `mmap(2)` (<https://man7.org/linux/man-pages/man2/mmap.2.html>):

> SIGBUS: Attempted access to a page of the buffer that lies beyond the end of the mapped file.

**[VERIFIED-RUN]** Reproduced on this machine with the same syscall Rust uses:

```python
open(p,"wb").write(b"A"*(64*1024)); f=open(p,"r+b")
m=mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
os.truncate(p, 4096)      # "another process" shrinks the file
print(m[40960])           # -> process killed
```

Exit status **138 = 128 + 10 = SIGBUS** (macOS signal 10). No exception, no unwinding, no
`catch_unwind`, no `Drop`. The process dies mid-proof and any partially written `proof.json`
survives.

Consequences we must accept and state, not "fix":

- `unsafe { Mmap::map(&file) }` is unsafe for exactly this reason. **[SOURCE]** memmap2 docs
  (<https://docs.rs/memmap2/latest/memmap2/struct.Mmap.html>): "All file-backed memory map
  constructors are marked unsafe because of the potential for Undefined Behavior (UB) using the map
  if the underlying file is subsequently modified, in or out of process. ... Solutions such as file
  permissions, locks or process-private (e.g. unlinked) files exist but are platform specific and
  limited."
- Truncation is the *benign* case (SIGBUS, a crash). **In-place modification** of the mapped bytes
  while we parse is worse: it is UB in Rust terms, and practically it means a TOCTOU between our
  `expect_records` length check and the `par_chunks_exact` that consumes the bytes. An attacker who
  can write the file concurrently can pass the check and then supply different data.
- Our comment at `binfile.rs:51-54` calls this "the standard and unavoidable caveat of mmap on any
  key file" **[VERIFIED-CODE]**. That is accurate but under-sells the mitigations that exist:
  `O_EXLOCK`/`flock(LOCK_EX)` on macOS/Linux, or copying the file to a process-private temp and
  unlinking it, or simply `read`ing instead of mapping for files below some size threshold.

### 3.2 Alignment

Our decoders never form a `*const u32`/`*const u64` into the mapping; they use
`u32::from_le_bytes(b[off..off+4].try_into()...)` on a byte slice (`binfile.rs:161-171`)
**[VERIFIED-CODE]**. That is alignment- and endianness-correct on every target.

Contrast **[SOURCE]** rapidsnark `src/binfile_utils.cpp`:

```cpp
u_int32_t res = *((u_int32_t *)((u_int64_t)addr + pos));   // readU32LE
u_int64_t res = *((u_int64_t *)((u_int64_t)addr + pos));   // readU64LE
```

Type-punned unaligned loads at an attacker-chosen offset. Section payloads start at
`12 + Σ(12 + len_i)`, so `pos` is fully attacker-controlled and need not be aligned. UB in C++,
a fault on strict-alignment targets, and it also assumes a little-endian host.

### 3.3 Integer overflow in offset arithmetic

Ours is clean: `checked_add` on the section walk and in `Cursor::take` (§2).

**[SOURCE]** rapidsnark `binfile_utils.cpp` `readFileData` is *not*:

```cpp
sections[sType].push_back(Section((void*)((u_int64_t)addr + pos), sSize));
pos += sSize;
if (pos > size) { throw std::range_error(...); }
```

`pos += sSize` on `u_int64_t` wraps. A section length near `2^64` makes `pos` wrap to a small
value, the range check passes, and a `Section` with a colossal `size` has already been pushed —
`getSectionSize()` then hands that size to callers. Also note the `Section` is recorded *before*
the bounds check.

One residual in ours: `expect_records` computes `n * stride` with a plain `*`
(`binfile.rs:242`) **[VERIFIED-CODE]**. On 64-bit this cannot overflow (`n <= 2^32-1`,
`stride <= 128`), but on a 32-bit target it wraps and the length check becomes a no-op. We ship
64-bit only today; a `checked_mul` costs nothing and removes the footgun.

---

## 4. Known CVEs and audit findings in zk tooling

### 4.1 CVE-2023-33252 / GHSA-xp5g-jhg3-3rg2 — snarkjs, "double spend", HIGH

**[SOURCE]** <https://github.com/advisories/GHSA-xp5g-jhg3-3rg2>: "iden3 snarkjs through 0.6.11
allows double spending because there is no validation that the publicSignals length is less than
the field modulus." The verifier accepted a public signal `s` and `s + r`, which aggregate to the
same `L_bar` because scalar multiplication is mod `r` — so one proof "verified" against two
different public-signal vectors.

We are not vulnerable: `parse_field` in `g16-cli/src/json.rs:34-44` bails on `n >= modulus` for
both `Fq` and `Fr`, with a dedicated test `rejects_a_coordinate_at_or_above_the_modulus`
**[VERIFIED-CODE]**. Note the failure mode is *silent reduction*: any decoder that reaches for
`from_le_bytes_mod_order` on untrusted input reintroduces this bug. There are two such call sites
in test code (`g16-core/src/prove.rs` test helpers, `g16-cli/src/json.rs` after the range check) —
both are fine, but the pattern is one to grep for.

### 4.2 CVE-2024-50354 / GHSA-cph5-3pgr-c82g — gnark, allocation bomb in key deserialization

**[SOURCE]** <https://github.com/Consensys/gnark/security/advisories/GHSA-cph5-3pgr-c82g>:
"deserialization of Groth16 verification keys allocate excessive memory, consuming a lot of
resources and triggering a crash with the error `fatal error: runtime: out of memory`". CVSS 5.5,
"Prover and verifier denial of service in case of maliciously crafted inputs (public key,
verification key)."

The fix pattern is worth copying **[SOURCE]** (PR #1307, commit `47ae846`):

```go
-  vk.CommitmentKeys = make([]pedersen.VerifyingKey, nbCommitments)   // count from the file
-  for i := range vk.CommitmentKeys { ... }
+  for i := 0; i < int(nbCommitments); i++ {
+      commitmentKey := pedersen.VerifyingKey{}
+      ... read ...
+      vk.CommitmentKeys = append(vk.CommitmentKeys, commitmentKey)   // bounded by bytes present
+  }
+  if len(vk.CommitmentKeys) != int(nbCommitments) { return ... error }
```

i.e. **never size an allocation from a count in the file; size it from the bytes actually
available, then check the count matches.** The advisory's stated workaround is also relevant to a
proving service: "run key verification as a separate service which halts the verification pipeline
in case of OOM when verification keys come from untrusted sources."

We have exactly this bug. See §6.1.

### 4.3 CVE-2025-30147 / GHSA-jcp8-gh74-97hq — Besu/gnark-crypto, subgroup check that was not an
on-curve check

**[SOURCE]** NVD: "there are EC points which may be crafted which are in the correct subgroup but
are not on the curve and the besu-native gnark implementation was relying on subgroup checks to
perform point-on-curve checks as well. The version of gnark-crypto used at the time did not do this
check when performing subgroup checks." Result: consensus divergence on `ALTBN128_ADD/MUL/PAIRING`.

Direct lesson for us: arkworks names its method
`is_in_correct_subgroup_assuming_on_curve` — the "assuming" is load-bearing. Calling it without a
preceding `is_on_curve()` is the Besu bug. Our `check_g1`/`check_g2`
(`crates/g16-zkey/src/lib.rs:220-238`) and `checked_g1`/`checked_g2`
(`crates/g16-cli/src/json.rs:106-128`) both short-circuit on-curve first **[VERIFIED-CODE]**.
Correct. Any new validation code must preserve that order.

### 4.4 CVE-2023-44273 — gnark-crypto signature malleability from missing interval check on
deserialization. **[SOURCE]** NVD. Same root cause as 4.1: deserializing a scalar without asserting
it is canonical.

### 4.5 The Aztec Plonk "0 bug" — point-at-infinity representation confusion

**[SOURCE]** 0xPARC ZK Bug Tracker, entry 7
(<https://github.com/0xPARC/zk-bug-tracker#aztec-2>), finder Nguyen Thoi Minh Quan
(<https://github.com/cryptosubtlety/00/blob/main/00.pdf>):

> "When [W_z]₁ and [W_zw]₁ are checked in the verifier code, a value of `0` is recognized as not on
> the elliptic curve, but the code does not fail immediately. The verifier continues on and later
> recognizes the `0` value as the point at infinity. This causes the pairing equation to be
> satisfied, and therefore the proof is successfully verified."
>
> "A simple to understand fix would be to agree on a consistent representation of the point at
> infinity."

This is the direct precedent for our `(0,0) → identity` mapping in `binfile::g1`/`g2`
(`binfile.rs:214-238`) **[VERIFIED-CODE]**. The mapping itself is *required* — ffjavascript writes
infinity that way — but it means an attacker-supplied zkey can force any point in the key to the
identity, and the identity passes both `is_on_curve` and the subgroup check. See §6.3 for what that
buys an attacker.

### 4.6 No CVE exists for the `.zkey`/`.wtns` parsers themselves

**[VERIFIED-RUN]** NVD keyword searches for `snarkjs`, `circom`, `gnark`, `zero-knowledge proof`
return only the entries above; the GitHub Advisory database has exactly one advisory affecting the
npm package `snarkjs` (4.1) and none for `ffjavascript`, `circomlibjs`, `r1csfile`, or
`@iden3/binfileutils`, and none for the Rust crates `ark-circom`/`ark-groth16`/`ark-serialize`/
`memmap2`. **Absence of a CVE here is not evidence of safety** — it reflects that nobody has fuzzed
these parsers, which is exactly why this dimension is worth work. Reading the two most-used
alternative readers turned up unreported defects in minutes:

**rapidsnark** (`src/binfile_utils.cpp`, `src/zkey_utils.cpp`, `src/fileloader.cpp`) **[SOURCE]**:
u64 overflow in the section walk (§3.3); unaligned type-punned loads (§3.2); `nCoefs` derived by
division from the section size rather than read and cross-checked; `n8q`/`n8r` from the file used
as read lengths with no sanity bound; `qPrime` never validated (only `rPrime`, via `PrimeIsValid`
in `prover.cpp:110`); no length check of section 5/6/7/8/9 against `nVars`/`domainSize`.

**ark-circom / circom-compat** (`src/zkey.rs`) **[SOURCE]**
<https://github.com/arkworks-rs/circom-compat/blob/master/src/zkey.rs>, a Rust reader for the same
format, is markedly worse than ours:

- `std::str::from_utf8(&magic[..]).unwrap()` — **panics** on non-UTF-8 magic bytes; and the magic
  is stored in `ftype` but **never compared to `"zkey"`**, and `version` is never checked.
- `reader.seek(SeekFrom::Current(section_length as i64))` — a `u64` length `>= 2^63` becomes a
  negative seek, walking backwards through the file.
- `get_section(id)` is `self.sections.get(&id).unwrap()[0]` — **panics** on a missing section.
- `matrices()`: `vec![vec![vec![]; header.domain_size as usize]; 2]` — the same allocation bomb as
  ours, then `matrices[matrix as usize][constraint as usize]` indexes unchecked → **panic** on
  `matrix >= 2` or `constraint >= domain_size`; and `max_constraint_index as usize - header.n_public`
  underflows.
- `g1_section(num, id)` seeks to the section start and reads `num` points **without comparing
  `num * 64` to the section length**, so a short section silently reads into the next one.
- `deserialize_g1` calls `G1Affine::new(x, y)`, arkworks' *asserting* constructor — a point off the
  curve **panics** rather than erroring.

If anyone ever proposes replacing `g16-zkey` with `ark-circom` "because it is maintained", this list
is the answer.

---

## 5. Can a malicious `.zkey` mislead the operator?

### 5.1 Can a proof verify against a *different* vkey? — Yes, and it is not really an attack on the
parser

Groth16 verification consumes only `(alpha1, beta2, gamma2, delta2, IC)`. Sections 4-9 of the zkey
are *completely unconstrained* by the vkey. So:

- A zkey whose section-2 vk material is byte-identical to the operator's published `vkey.json`, but
  whose sections 4-9 encode a different QAP, is accepted by every reader including ours. Whether the
  resulting proof still verifies depends on whether the attacker can keep the pairing identity true.
- If the attacker holds the phase-2 toxic waste (`delta`), they can: forge proofs for any statement
  themselves, and construct L/H bases that make *the operator's* prover emit a verifying proof for a
  statement the operator did not intend. This is the standard subverted-setup property, and no
  amount of parser hardening touches it.
- **There is no way for a prover to detect this locally.** `snarkjs zkey export verificationkey`
  derives `vkey.json` *from the zkey*, so "the proof verifies against my vkey" is circular reasoning
  when the zkey is untrusted.

The only non-circular check is `snarkjs zkey verify <r1cs> <ptau> <zkey>` **[SOURCE]**
(`zkey_verify_fromr1cs.js`, which runs `newZKey(r1cs, ptau)` into memory and then
`phase2verifyFromInit`). `zkey_verify_frominit.js` **[SOURCE]** then checks, in order: same curve;
same `nVars/nPublic/domainSize`; `sameRatio` pairing checks that alpha1/beta1/beta2/gamma2/delta1/
delta2 match the powers-of-tau; sections 3 (IC), 4 (coeffs), 5 (A), 6 (B1), 7 (B2) are
**byte-identical** to the freshly derived initial zkey; sections 8 (L) and 9 (H) match the initial
ones up to the accumulated `delta` via `sameRatio`; and the whole phase-2 contribution chain in
section 10 hashes correctly.

**Operational conclusion:** a zkey is trusted input in the same sense a binary is. If the deployment
is proving-as-a-service with customer-supplied zkeys, `zkey verify` against the customer's r1cs +
a known ptau is the *only* thing that binds the key to a circuit, it costs a full setup re-run, and
it is not something our prover does or should do inline.

### 5.2 What a malicious zkey *can* do to a prover: witness extraction

This is the finding that matters for a GPU proving service, and it is not covered by anything in §2.

The proof is linear in the witness under attacker-chosen bases:

```
pi_a = Σ_j a_query[j]·w_j + alpha1 + r·delta1
```

An attacker who supplies `a_query` supplies the coefficients of that linear form. Concretely:

1. **Isolate one wire.** Set `a_query[j] = P` for the target `j` and `a_query[k] = identity` for all
   `k != j`. Then `pi_a = w_j·P + alpha1 + r·delta1`.
2. **Kill the blinding.** Set `delta1 = delta2 = (0,0)` and `alpha1 = (0,0)`. Both decode to the
   identity (§4.5) and both pass our `check_g1`/`check_g2`. Now `pi_a = w_j·P` exactly, a
   deterministic function of one secret wire.
3. **Make the DLP easy.** Either pick `P` such that `w_j·P` is searchable for the expected range of
   `w_j` (bit wires, small counters, timestamps — a brute-force over `2^32` candidates is trivial),
   or use an **invalid-curve / small-subgroup** point:
   - BN254 `G1` has cofactor 1, so on-curve points give no small subgroup. But **we never check
     on-curve for sections 5, 6, 8, 9**, and short-Weierstrass addition formulas do not involve the
     curve constant `b` — so a point with `y² != x³ + 3` lands the whole MSM on a *different* curve
     `y² = x³ + b'`, whose order is generically smooth. This is the classic invalid-curve attack
     (Biehl–Meyer–Müller, CRYPTO 2000) and it recovers `w_j` modulo each small factor.
   - BN254 `G2` has a large composite cofactor, so for section 7 a **plain on-curve, off-subgroup**
     point already yields a small-order component — no invalid-curve trick needed.

Step 2 is **[VERIFIED-RUN]**. I zeroed `delta1` and `delta2` in section 2 of `tiny_mul/circuit.zkey`
(`zero_delta.zkey`) and proved the same witness twice:

```
$ g16 prove --zkey zero_delta.zkey --witness circuit.wtns ...   # run 1
$ g16 prove --zkey zero_delta.zkey --witness circuit.wtns ...   # run 2
$ diff zd_1.json zd_2.json
4c4
<  "pi_c": [...]
---
>  "pi_c": [...]
```

Only `pi_c` differs. **`pi_a` and `pi_b` are bit-identical across two runs with fresh OS entropy** —
the zero-knowledge blinding on A and B is gone, and the key loaded without a single warning. The
control (honest zkey, two runs) differs in all three points. Note also that `delta2 = identity`
makes `e(C, delta2) = 1`, so the verification equation for that key degenerates and proofs under it
are trivially forgeable — an operator who publishes such a vkey has no soundness either.

Step 3's precondition is **[VERIFIED-RUN]**: I flipped one bit in the first `a_query` point
(`offcurve_a.zkey`) and separately set it to `(x=1, y=0)`, which is off the curve
(`badpoint_a.zkey`). Both **proved successfully with no error**, wrote a complete `proof.json`, and
the resulting `pi_a` is itself off the curve — our own verifier rejects it at read time
(`pi_a: point is not on the curve`), but *the prover already did the invalid-curve arithmetic*,
which is where the leak happens. The load-time comment at `crates/g16-zkey/src/lib.rs:153-157`
**[VERIFIED-CODE]** explicitly documents that sections 5-9 are unchecked, with the (sound)
performance rationale that a subgroup check per point would dominate key load.

**This is a real threat only when the zkey and the witness have different owners.** For the
single-owner CLI it is not a vulnerability. For a proving service — the deployment a fast GPU
Groth16 prover exists for — it is the primary one, and the mitigation is a *one-time, cached* key
validation (batched subgroup check via a random linear combination: one MSM per section instead of
`n` subgroup checks), not a per-proof cost.

### 5.3 Field elements are never range-checked

`binfile::fq` is `Fq::new_unchecked(bigint(b))` (`binfile.rs:185-187`) **[VERIFIED-CODE]** — any
256-bit value is accepted as a base-field coordinate. BN254's `q` is ~254 bits, so roughly 4× the
canonical values are non-canonical and reachable. arkworks' Montgomery reduction is only correct for
limbs below the modulus; feeding it larger limbs is not memory-unsafe but produces silently wrong
arithmetic, and `is_on_curve()` on such a value is testing a garbage point. `fr_double_montgomery`
(`binfile.rs:202-204`) has the same property for section-4 coefficients. (`fr_normal`, used for
`.wtns`, *does* range-check — the asymmetry is deliberate but undocumented.)

---

## 6. Denial of service

### 6.1 CONFIRMED: `domainSize` allocation bomb — 34 GB and 12 s from a 4 KB file

**[VERIFIED-CODE]** `read_coefficients` in `crates/g16-zkey/src/lib.rs:251-284`:

```rust
let mut counts = [vec![0u32; domain_size + 1], vec![0u32; domain_size + 1]];
...
let mut cursor = [row_ptr[0].clone(), row_ptr[1].clone()];
```

`domain_size` comes from section 2 and is only validated to be a nonzero power of two
(`lib.rs:130-136`), so it may be up to `2^31`. Those four vectors are `4 × (2^31+1) × 4 B = 34.4 GB`.
Nothing bounds `domain_size` against the section-9 length before this point, and the
`Domain::new` check against `Fr::TWO_ADICITY` (= 28, so `domainSize <= 2^28`) lives in
`g16-field/src/lib.rs:49-51` and only runs later, inside `CpuBackend::prepare`.

**[VERIFIED-RUN]** I patched the 4-byte `domainSize` field of `bench/artifacts/tiny_mul/circuit.zkey`
(a **4 047-byte** file) from `8` to `2^31` and ran `g16 prove`:

```
Error: loading .../domain_bomb.zkey
Caused by:
    malformed section 9: expected 2147483648 records of 64 bytes (137438953472), got 512
       11.99 real         9.19 user         2.62 sys
         34361081856  maximum resident set size
```

**34 361 081 856 bytes = 32 GiB resident, 12 seconds of wall clock, from a one-field edit to a 4 KB
file.** The error message that eventually comes out is the *section-9* check — i.e. the check that
would have caught this instantly ran too late. One byte of input, 34 GB of output. On a machine
without that much RAM+swap the allocation fails and Rust **aborts** (allocation failure is not a
recoverable `Err`), so this is a hard process kill, not a caught error.

This is the same defect class as CVE-2024-50354 (§4.2), in the same position (key deserialization),
with a worse amplification ratio.

The fix is cheap and belongs in the header parse, before any allocation: reject
`domain_size > 1 << Fr::TWO_ADICITY` (28), and additionally require
`section9.len() == domain_size * 64` *before* `read_coefficients` runs. Either check alone kills it.
Reordering `read_g1_section(&file, 9, domain_size)` above `read_coefficients` would also fix it, but
an explicit bound is the honest fix because it does not depend on statement order surviving a
refactor.

### 6.2 `nSections` reservation — real but low severity on 64-bit

**[VERIFIED-CODE]** `binfile.rs:73`: `Vec::with_capacity(n_sections)` where `n_sections` is a `u32`
straight from the file. At `0xFFFFFFFF` that is a `103 GB` reservation.

**[VERIFIED-RUN]** `nsections_bomb.zkey` (12 bytes: `"zkey" | 1 | 0xFFFFFFFF`) → clean error
`malformed section 0: section header 0 runs past end of file`, **6.6 MB peak RSS, 0.00 s**. On
64-bit macOS/Linux the reservation is address space only, never touched, so it is free. It is still
worth capping (a real zkey has 10 sections; anything over, say, 64 is nonsense) because: allocation
failure aborts rather than errors, 32-bit targets would fail outright, and it costs one comparison.
Severity: low. Do not spend the fix budget here before §6.1.

### 6.3 Non-bombs, for the record

- `nVars = 0xFFFFFFFF` → caught by `expect_records` on section 5 with no allocation
  **[VERIFIED-RUN]**.
- `nPublic` is bounded by `nVars` (`lib.rs:138`) and cross-checked twice, by the section-3 length
  (`nPublic+1` points) and the section-8 length (`nVars-nPublic-1` points) **[VERIFIED-CODE]**. An
  attacker cannot shrink `nPublic` to make us publish fewer public signals than the circuit has
  without also resizing two sections — at which point it is simply a different circuit.
- `nCoefs` is bounded by the section-4 length **[VERIFIED-CODE]**, and the CSR output vectors are
  sized from the histogram, hence by `nCoefs`. No amplification.
- Section-4 records with `constraint >= domain_size` or `signal >= nVars` are rejected
  **[VERIFIED-CODE]** (`lib.rs:258-263`, `lib.rs:286-291`), and again defensively in
  `cpu.rs:99`.
- The pathological *cost* case is legitimate-but-huge: `domainSize = 2^28` with a matching 17 GB
  section 9. That is a real proving key, not an attack; the defence is an operator-configured
  ceiling, not a parser check.

---

## 7. Reproduction script

Regenerates every malformed file used above from `bench/artifacts/tiny_mul/`. Run from any
scratch directory; it writes nothing into the repo.

```python
import struct
SRC = "bench/artifacts/tiny_mul/circuit.zkey"
d0  = open(SRC, "rb").read()

def sections(d):
    pos, s = 12, {}
    while pos + 12 <= len(d):
        sid,  = struct.unpack_from("<I", d, pos)
        slen, = struct.unpack_from("<Q", d, pos + 4)
        s.setdefault(sid, []).append((pos + 12, slen))
        pos += 12 + slen
    return s

S  = sections(d0)
p2 = S[2][0][0]                       # section 2 payload
n8q, = struct.unpack_from("<I", d0, p2)
n8r, = struct.unpack_from("<I", d0, p2 + 4 + n8q)
hdr  = p2 + 4 + n8q + 4 + n8r          # -> nVars, nPublic, domainSize

# 6.1  domainSize = 2^31   -> 34 GB, 12 s
d = bytearray(d0); struct.pack_into("<I", d, hdr + 8, 1 << 31)
open("domain_bomb.zkey", "wb").write(bytes(d))

# 6.3  nVars = 2^32-1      -> clean error
d = bytearray(d0); struct.pack_into("<I", d, hdr, 0xFFFFFFFF)
open("nvars_bomb.zkey", "wb").write(bytes(d))

# 3.3  section length overflow -> clean error
d = bytearray(d0); struct.pack_into("<Q", d, 16, 0xFFFFFFFFFFFFFFF0)
open("seclen_overflow.zkey", "wb").write(bytes(d))

# 6.2  nSections = 2^32-1  -> clean error, 103 GB virtual reservation
open("nsections_bomb.zkey", "wb").write(b"zkey" + struct.pack("<II", 1, 0xFFFFFFFF))

# 5.2  delta1 = delta2 = (0,0) -> loads fine, pi_a/pi_b become deterministic
pts = hdr + 12                          # alpha1 64, beta1 64, beta2 128, gamma2 128, delta1 64, delta2 128
d1  = pts + 64 + 64 + 128 + 128
d = bytearray(d0); d[d1:d1+64] = b"\0"*64; d[d1+64:d1+192] = b"\0"*128
open("zero_delta.zkey", "wb").write(bytes(d))

# 5.2  off-curve base in section 5 -> proves successfully, emits an off-curve pi_a
p5 = S[5][0][0]
d = bytearray(d0); d[p5] ^= 1
open("offcurve_a.zkey", "wb").write(bytes(d))
d = bytearray(d0); d[p5:p5+64] = b"\0"*64; d[p5] = 1
open("badpoint_a.zkey", "wb").write(bytes(d))
```

---

## 8. Sources

Primary (read in full or in the cited region during this session):

- snarkjs `src/zkey_utils.js`, `src/wtns_utils.js`, `src/groth16_prove.js`, `src/zkey_new.js`,
  `src/zkey_constants.js`, `src/zkey_verify_fromr1cs.js`, `src/zkey_verify_frominit.js`,
  `src/wtns_check.js` — <https://github.com/iden3/snarkjs>
- `@iden3/binfileutils` `src/binfileutils.js` — <https://github.com/iden3/binfileutils>
- rapidsnark `src/binfile_utils.cpp`, `src/zkey_utils.cpp`, `src/wtns_utils.cpp`,
  `src/fileloader.cpp`, `src/prover.cpp` — <https://github.com/iden3/rapidsnark>
- circom-compat `src/zkey.rs` — <https://github.com/arkworks-rs/circom-compat>
- POSIX `mmap` — <https://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html>
- Linux `mmap(2)` — <https://man7.org/linux/man-pages/man2/mmap.2.html>
- memmap2 `Mmap` docs — <https://docs.rs/memmap2/latest/memmap2/struct.Mmap.html>
- RUSTSEC-2020-0077 (`memmap` unmaintained; `memmap2` is the maintained fork — we already use it)

Advisories:

- CVE-2023-33252 / GHSA-xp5g-jhg3-3rg2 (snarkjs public-signal range check)
- CVE-2024-50354 / GHSA-cph5-3pgr-c82g + PR Consensys/gnark#1307 (Groth16 key deserialization DoS)
- CVE-2025-30147 / GHSA-jcp8-gh74-97hq (subgroup check used as an on-curve check)
- CVE-2023-44273 (gnark-crypto signature malleability, missing canonicality check)

Secondary:

- 0xPARC ZK Bug Tracker — <https://github.com/0xPARC/zk-bug-tracker>, entry 7 "Aztec Plonk Verifier:
  0 Bug"; original writeup Nguyen Thoi Minh Quan,
  <https://github.com/cryptosubtlety/00/blob/main/00.pdf>
- Biehl, Meyer, Müller, "Differential Fault Attacks on Elliptic Curve Cryptosystems", CRYPTO 2000 —
  the invalid-curve attack cited in §5.2 (cited from memory of the standard result, not re-read this
  session).

Search tooling note: the `ddg` CLI hit a DuckDuckGo CAPTCHA on every query this session
(HTTP 202, interactive input unavailable), so all web research above was done by fetching known
primary URLs directly plus the NVD and GitHub Advisory REST APIs. No claim rests on a search-result
snippet.
