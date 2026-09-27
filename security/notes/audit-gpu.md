# Audit: GPU backend (Metal), as the model for the CUDA port

Scope: `crates/g16-metal/src/{msm.rs, stages.rs, backend.rs, layout.rs}` and
`crates/g16-metal/src/shaders/{msm.metal, ntt.metal, gather.metal, bn254_fr.metal}`.

Method: read every line of `msm.metal` (1158 lines) and the host code that drives it, then ran
the suite. Everything below marked VERIFIED was established by reading the specific lines cited
or by running the command shown. Everything marked INFERRED is reasoning from the code that I did
not separately execute.

Headline: **the bucket accumulation is genuinely race-free and the addition-formula guards are
complete.** This backend does not have the two bugs that the research dimension found in
lambdaworks and zkmopro. The findings below are about error handling, latent fragility, and test
coverage, not about a live wrong answer. One intermittent test failure is unresolved.

---

## Test suite state

VERIFIED, run at audit time:

```
cd path/to/snarkrs
cargo test --workspace --release --features metal
```

All test binaries green. Aggregate across the workspace: 0 failed. The `g16-metal` lib binary
alone reports `25 passed; 0 failed; 4 ignored`.

The four ignored `g16-metal` tests are:

| test | ignore reason |
|---|---|
| `msm::tests::five_msms_match_cpu_on_every_artifact` | "minutes on the CPU oracle side" |
| `msm::tests::measure_against_cpu` | measurement, not a check |
| `msm::tests::measure_per_msm` | measurement, not a check |
| `backend::tests::measure_prepare` | measurement, not a check |

Note that the only test which sweeps **every** benchmark artifact through the five MSMs against
the CPU oracle is one of the ignored ones. The default run covers `tiny_mul` and `js_1x1_d8` only
(`five_msms_match_cpu_on_tiny_mul`, `five_msms_match_cpu_on_js_1x1_d8`). See finding G5.

---

## Answers to the six questions

### 1. How do we accumulate buckets, and is it race-free?

**Mechanism: exclusive row ownership via a counting sort, plus two per-thread spill slots. No
atomics touch point data. VERIFIED race-free.**

The pipeline is five dispatches (`msm.rs:850-882` for the digit stages, `msm.rs:980-1006` for the
point stages):

1. `msm_count` (`msm.metal:655`) counts `(window, bucket)` occupancy with
   `atomic_fetch_add_explicit(..., memory_order_relaxed)` at `msm.metal:679`. Atomic is on a
   `uint` counter, not on a point.
2. `msm_scan` (`msm.metal:691`) exclusive-prefix-sums the counts, one threadgroup per window.
3. `msm_scatter` (`msm.metal:731`) bumps a per-row cursor with a relaxed 32-bit
   `atomic_fetch_add` (`msm.metal:755`) and writes `entries[slot] = uint2(row, point<<1|sign)`
   (`msm.metal:756`).
4. `msm_segmented_*` (`msm.metal:881`) does the point work.
5. `msm_merge_*` (`msm.metal:973`) folds spills in.

The race-freedom argument, which I checked term by term:

- After the scatter, all entries for a given `row` occupy one contiguous range
  `[cursor[row] - counts[row], cursor[row])`, because the cursor starts at that row's exclusive
  prefix and only that row's threads bump it. So **a row lives in exactly one contiguous run**.
- `msm_segmented_impl` gives thread `k` the slice `[lo, hi) = [k*slice_len, min(lo+slice_len, used))`.
  Walking the slice it detects a row change on `e.x != cur_row`.
  - A run that is neither the slice's first nor its last is wholly inside this slice. Because the
    run is contiguous and unique, **no other thread can ever see that row**, so it is written
    unsynchronised with `buckets[cur_row] = acc` at **`msm.metal:924`**. This is the exclusive
    ownership.
  - The first and last runs may continue into a neighbour, so they go to this thread's own two
    spill slots, `head_slot = 2*gid` and `tail_slot = 2*gid+1` (`msm.metal:899-900`,
    written at `msm.metal:917-919` and `msm.metal:938-944`). Two slots per thread, indexed by
    `gid`, so **no two threads ever write the same slot**.
- `msm_merge_impl` runs one thread per row (`msm.metal:973`), recomputes the run's slice span
  `k_lo = start/slice_len`, `k_hi = (end-1)/slice_len` (`msm.metal:991-992`) and adds only the
  spill slots whose tag equals its own row (`msm.metal:995-1002`). One thread per row again, so
  the read-modify-write of `buckets[row]` is exclusive.

I also checked the double-count question, which is the non-obvious part: can a row be both
direct-written and spilled? No. A direct write happens only for a run that is neither first nor
last in its slice, which forces `k_lo == k_hi == k`, and in slice `k` the head spill carries the
first run's row and the tail spill carries the last run's row, neither of which is ours.
VERIFIED by reading `msm.metal:912-944` against `msm.metal:988-1003`.

Cross-dispatch ordering is Metal's, not ours: the encoder is created with
`cb.new_compute_command_encoder()` (`msm.rs:738`), which is `MTLDispatchTypeSerial`, so dispatches
run in encode order with automatic hazard tracking on these (non-heap, tracked) buffers. See
"What CUDA must do differently", item 1.

**This is the cheap form of cuZK/sppark exclusive row ownership, and it is the right design.** It
does not have lambdaworks' last-writer-wins bucket race.

### 2. Are all bucket and accumulator buffers explicitly zeroed, or do we rely on Metal zero-fill?

**Neither, exactly: we rely on explicit kernel writes, and we do not rely on Metal's zero-fill
anywhere that matters. VERIFIED.**

The allocation site is `Pool::take` at **`msm.rs:441-461`**, ending in
`self.device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)` at **`msm.rs:460`**.
Critically, `take` first tries to hand back a **recycled** buffer from `self.free`
(`msm.rs:443-457`), so the Metal zero-fill guarantee is defeated from the second proof onward by
construction. The code knows this; the comment at `msm.rs:850-852` says so.

Buffer by buffer, what guarantees it is initialised before it is read:

| buffer | alloc | initialised by | verified |
|---|---|---|---|
| `counts` | `msm.rs:830` | `zero_u32` dispatch, `msm.rs:853-857` | yes |
| `cursor` | `msm.rs:831` | `msm_scan` writes every row, `msm.metal:715` | yes |
| `entries` | `msm.rs:834` | scatter writes exactly the used prefix; segmented reads only `[lo, hi)` with `hi <= used` (`msm.metal:898`, `msm.metal:907`) | yes |
| `buckets` | `msm.rs:912` | `msm_clear_g1/g2` dispatch, `msm.rs:982-985` — **partial, see G3** | partial |
| `spill_rows` | `msm.rs:917` | kernel writes `MSM_NO_ROW` to both slots at `msm.metal:901-902`, **before** the `lo >= used` early return at `msm.metal:905` | yes |
| `spill_pts` | `msm.rs:916` | only read when the paired `spill_rows` tag matches, so read implies written | yes |
| `window_sums` | `msm.rs:918` | `msm_reduce_*`, one write per window | yes |
| `ones` | `msm.rs:919` | `msm_ones_*`, one write per group | yes |

Two details worth recording because a port will get them wrong:

- `spill_rows`' sentinel is `MSM_NO_ROW = 0xffffffff` (`msm.metal:855`), **not zero**. Row 0 is a
  valid row. A port that "optimised away" the `MSM_NO_ROW` stores and leaned on a zeroed
  allocation would silently fold slot 0's garbage point into bucket 0 of window 0. The current
  code writes the sentinel before the bounds return, which is correct and deliberate.
- `dispatch_1d` uses `enc.dispatch_threads` (`msm.rs:1094`), the non-uniform-threadgroup form, so
  exactly `n` threads launch. The `if (w >= p.n_windows) return;` guard at `msm.metal:892` is
  therefore defensive rather than load-bearing, and it correctly sits **before** the `spill_rows`
  stores so it cannot write out of bounds.

### 3. Which addition formula, and are the exceptional cases handled?

**XYZZ (`Xyzz<F>`, `msm.metal:363-371`), with `ZZ == 0` as the identity. All three exceptional
cases are explicitly guarded. VERIFIED.**

Formulas: `madd-2008-s` for bucket accumulation (`pt_madd`, `msm.metal:461`), `add-2008-s` for
bucket reduction (`pt_add`, `msm.metal:496`), `mdbl-2008-s` / `dbl-2008-s-1` with `a = 0`
(`pt_dbl_affine` `msm.metal:418`, `pt_dbl` `msm.metal:440`).

Guards in `pt_madd`:

```
461: inline Xyzz<F> pt_madd(Xyzz<F> acc, Aff<F> p) {
462:     if (aff_is_inf(p)) { return acc; }              // operand is identity
465:     if (pt_is_zero(acc)) { return pt_from_affine(p); } // accumulator is identity
...
472:     if (f_is_zero(pp_)) {
473:         // same x
477:         if (f_is_zero(rr)) { return pt_dbl_affine(p); }  // P == Q
479:         return pt_zero<F>();                            // P == -Q
480:     }
```

`pt_add` carries the same set at `msm.metal:497-500` (both identities) and `msm.metal:509-514`
(`P == Q` to `pt_dbl(a)`, `P == -Q` to `pt_zero`). `pt_dbl` and `pt_dbl_affine` additionally guard
`f_is_zero(p.y)` (`msm.metal:419`, `msm.metal:441`), which is the 2-torsion case that BN254's
odd-order groups cannot contain; returning the identity there is the correct answer anyway.

I checked the correctness of each branch rather than just its presence:
`pp_ = p.x*acc.zz - acc.x = acc.zz*(p.x - x_acc)`, and `acc.zz != 0` on that path because the
`pt_is_zero(acc)` guard already fired, so `pp_ == 0` iff the affine x-coordinates match.
`rr == 0` then splits same-point from opposite-point. `pt_dbl_affine(p)` returns a valid
representation of `2P`, and since `acc == P` projectively, `2P == acc + p`. Correct.

The research dimension's point 3 applies and this code answers it: `P == -Q` is self-healing under
the raw formula but `P == Q` and "operand is identity" are not, so a partial guard set would look
correct in testing. **We have the full set.**

The soundness of these guards depends on `f_is_zero` / `f_eq` being exact, which in turn depends
on every `Fq` operation returning a canonical residue in `[0, q)`. VERIFIED: `fq_add`
(`msm.metal:141`) ends in `fq_cond_sub_n`; `fq_mul` (`msm.metal:193`) ends in `fq_cond_sub_n` at
`msm.metal:228`; `fq_sub` (`msm.metal:152`) masks the modulus back in and cannot exceed `q`. There
is no lazy-reduction path. This is load-bearing and is called out again for CUDA below.

### 4. Do we handle a base at infinity, and a scalar of 0 or 1?

**Yes to all three, and the infinity handling is done properly at the host pack, not ignored.
VERIFIED.**

*Base at infinity.* `PackedG1Affine::from_affine` (`layout.rs:338-351`) reads
`p.infinity` and maps it to `Self::INFINITY = (0, 0)` (`layout.rs:332-335`). The comment at
`layout.rs:340-343` is explicit that the flag is the source of truth, not the coordinates. On the
device, `aff_is_inf` tests `x == 0 && y == 0` (`msm.metal:394-397`). `(0, 0)` cannot collide with a
real point: `y^2 = x^3 + 3` at `x = 0` gives `y^2 = 3`, so `y != 0`. Round-tripping is tested
(`layout.rs:583`, `layout.rs:603`).

This is exactly the bug the research found in zkmopro's `pack_affine_and_scalars`, and we do not
have it. Given the measurement in the research note that `js_16x16_d32`'s B query is 33.9%
infinity, this path is not hypothetical.

One subtlety that is load-bearing and undocumented: the sign flip in the accumulation
(`msm.metal:793` legacy, **`msm.metal:931`** segmented) does `b.y = f_neg(b.y)` **before**
`pt_madd` inspects `aff_is_inf`. This is safe only because `f_neg(0) == 0`, so negation maps the
infinity sentinel to itself. VERIFIED by reading `fq_sub` (`msm.metal:152-169`): with `a = b = 0`
the borrow is 0, the mask is 0, the result is 0. See finding G7.

*Scalar 0 and 1.* Both are diverted out of Pippenger entirely. `msm_count` and `msm_scatter`
return early on `sc_is_zero(s) || sc_is_one(s)` (`msm.metal:665`, `msm.metal:745`), so a zero or
one scalar emits no entries and touches no bucket. Scalar 1 is then summed by `msm_ones_*`
(`msm.metal:1102`) in one mixed addition per base, tree-reduced in threadgroup memory, and added
to the Horner result on the host at `msm.rs:1050-1052` / `msm.rs:1066-1068`. Host and device agree
on the classification because the host classifies with `Fr::is_zero()/is_one()`
(`msm.rs:597-600`) and the device classifies standard-form limbs written by
`PackedScalar::from_fr`.

The `ones` stride loop (`msm.metal:1112`) covers `[0, n)` exactly once: the set
`{g*tcount + tid}` over `g < ones_groups`, `tid < tcount` is exactly `[0, stride)`, and the loop
steps by `stride`. VERIFIED by inspection.

Sizing is safe when the host cannot classify. For H the scalars are device-resident and
`general_prefix` is `None`, so `general_in` reports the whole range as general
(`msm.rs:347-353`), which over-sizes `cap` and the entry array rather than under-sizing it. That
is the safe direction, and the comment at `msm.rs:345-347` says so. **This is the one place a port
could introduce a heap overflow and it is correct here.**

### 5. Is the conditional subtraction branchless?

**Yes, in both fields. VERIFIED.**

`fq_cond_sub_n` (`msm.metal:125-140`) computes the reduced value unconditionally into `red[]`,
derives `take` from the borrow bit and the high word, and selects with
`out.v[i] = select(a.v[i], red[i], take)` at `msm.metal:136`. No `if`. `fr_cond_sub_n`
(`bn254_fr.metal:114-130`) is identical in shape, with the same `select` at
`bn254_fr.metal:127`. `fq_sub` / `fr_sub` avoid a branch differently, by masking the modulus in
(`msm.metal:159-168`, `bn254_fr.metal:150-157`).

This matches sppark's predicated approach and beats lambdaworks' divergent `if`. It is the correct
answer to the ZKProphet measurement in the research note (conditional-reduction branches at 70.5%
of `FF_add`/`FF_sub` latency, 2.4x divergence cost).

Note that the field arithmetic is branchless but the **point** arithmetic is not: `pt_madd` has
four early-return branches. That is the right trade (the exceptional branches are rare and the
guards are mandatory for correctness), and it is not a timing-side-channel concern here because
the bases are public zkey data. Flagging only so a port does not "fix" it.

### 6. Suite result

Covered at the top. Green, with one intermittent failure discussed as G2.

---

## Findings

### G1. HIGH — No `MTLCommandBuffer` status or error is ever checked, at any of six commit sites

**What.** After every `commit()` / `wait_until_completed()` pair the code proceeds directly to
reading the shared buffers. `commandBuffer.status` and `commandBuffer.error` are never consulted.

**Where.** VERIFIED by exhaustive grep over `crates/g16-metal/src/`: zero hits for `.status()`,
`.error()`, `MTLCommandBufferStatus`, or `add_completed_handler`. The six sites are
`msm.rs:636-637`, `msm.rs:746-747`, `stages.rs:527-528`, `stages.rs:558-559`,
`stages.rs:567-568`, `stages.rs:585-586`.

**Why it matters.** `waitUntilCompleted` returns when the command buffer reaches a terminal state,
and `MTLCommandBufferStatusError` is terminal. On a GPU fault, an out-of-memory condition, a
device removal, or a hang timeout, the wait returns normally and the kernels' writes may be
partially or entirely absent. The very next thing the code does is
`read_back(&self.window_sums, ...)` (`msm.rs:1021`, `msm.rs:1044`) on a **pooled** buffer, so what
it reads is not zero and not garbage, it is the previous proof's window sums. The prover then
assembles and returns a proof that is silently wrong, with no error anywhere. In a proving service
that is a wrong answer served as if it were right.

This is also why G2 below is currently undiagnosable.

**Fix.** After each `wait_until_completed()`, check
`cb.status() == MTLCommandBufferStatus::Completed` and otherwise return
`ProveError` carrying `cb.error()`'s localized description. Six call sites, and it belongs in a
small helper (`fn submit_and_wait(cb) -> Result<(), ProveError>`) so a seventh site cannot forget.

**CUDA.** Must do this too, and CUDA makes it easier to get wrong: `cudaMemcpy` and
`cudaStreamSynchronize` return sticky errors from *earlier* async launches, so an unchecked
`<<<>>>` launch surfaces as a confusing failure at the next sync. Check `cudaGetLastError()` after
every launch and the return code of every sync.

---

### G2. HIGH (unresolved) — `one_metal_circuit_proves_concurrently` failed once and I could not reproduce it

**What.** Running `cargo test -p g16-metal --release`, the lib binary reported
`FAILED. 24 passed; 1 failed; 4 ignored`, the failure being
`backend::tests::one_metal_circuit_proves_concurrently`. I did not capture the panic message
before the next run overwrote the terminal state, which was my error.

**Where.** Test at `backend.rs:503-533`. It runs four `prove_with_blinders` calls concurrently
against one `PreparedCircuit`, over all six artifacts, and verifies each proof.

**Reproduction attempts, all VERIFIED negative.** After the failure:

- 8 further `cargo test -p g16-metal --release --lib` runs: all green.
- 30 sequential runs of the compiled lib binary directly: all green.
- 15 rounds of 3 concurrent copies of the lib binary (45 runs): all green.
- 10 rounds of 6 concurrent copies (60 runs): all green.
- 6 rounds running the lib binary concurrently with the `compute_h_gpu`, `independent_audit`,
  `msm_phases` and `field_gpu` integration binaries: all green.

So roughly 1 failure in 150+ observed runs, seen once, under whatever contention `cargo test`'s
own parallel binary scheduling produced.

**Why it matters.** This is precisely the shape the research dimension warns about: an ordering
that is decided by a nondeterministic scatter cursor, so a defect shows up on run one thousand and
one. It may be benign (a transient Metal error, which G1 would have converted into a clean
diagnosis) or it may be a real concurrency defect. I could not distinguish, and I am not going to
claim it is fine because I failed to reproduce it 150 times.

**What I checked and found sound**, so these are eliminated as causes (INFERRED from reading, not
from instrumentation):

- `Pool::take`/`give` (`msm.rs:441-471`) are mutex-guarded and `give` is called only after
  `wait_until_completed` returns (`msm.rs:747` then `msm.rs:760`), so a buffer cannot be handed to
  a second thread while the GPU still reads it.
- The `Scratch` pool in `stages.rs` (`take_scratch` at `stages.rs:460`, `HHandle::drop` at
  `stages.rs:820-826`) has the same discipline; the scratch is returned only when the `HHandle`
  drops, and the handle outlives the `msms` call that reads `h_mont`.
- Plans are shared between jobs by `(ScalarBuf pointer, offset, n)` (`msm.rs:711`), and all three
  consumers only *read* the shared `counts`/`cursor`/`entries`.
- Two concurrently-live `ScalarBuf`s cannot alias in that key, since both are borrowed for the
  duration of the call.

**Fix.** Three things, in order: (a) land G1 so a transient command-buffer error stops being
invisible; (b) make the test print the actual mismatch (which artifact, which of the four threads,
and whether it was `prove` erroring or `verify` rejecting) rather than a bare `unwrap`; (c) add a
looped stress variant behind `#[ignore]` that runs the concurrent prove a few hundred times, so
the next occurrence is caught with evidence attached.

---

### G3. MEDIUM — `msm_clear_impl` clears only `zz`, leaving three of four coordinates holding the previous proof's points

**What.** The bucket clear writes `buckets[gid].zz = 0` and nothing else.

**Where.** `msm.metal:858-866`, the store at **`msm.metal:864`**. Dispatched at `msm.rs:982-985`.

**Why it matters.** It is correct **today**, and only because every consumer tests `zz` before
touching the other coordinates: `pt_is_zero` (`msm.metal:389`) tests `zz`; `pt_add`'s two identity
guards (`msm.metal:497-500`) test `zz`; the host's `PackedXyzzG1::to_projective` and its G2 twin
return the identity on `zz.is_zero()` before reading `x`, `y`, `zzz` (`msm.rs:262-263`,
`msm.rs:276-277`). Any future reader that computes on `x` or `y` before the `zz` test, or any
formula rewritten to be exception-free and therefore guard-free, reads the previous proof's live
group elements out of a recycled buffer. The failure would not be a crash, it would be a wrong
point.

The saving is real (a full clear is 4x the store traffic on a 128-byte or 256-byte struct), so I
am not arguing for a full clear. I am arguing the invariant should be enforced rather than
remembered.

**Fix.** Either (a) add a `static_assert`-style comment contract naming the three functions that
must test `zz` first and a test that fails if a bucket's stale `x` can influence the result (seed
the bucket buffer with a known non-identity point, run an MSM whose row set does not cover it,
assert the answer is unchanged), or (b) clear all four coordinates behind an env flag so the
"is the partial clear hiding a bug" question can be answered in one run rather than argued.

**CUDA.** Same choice, but the stakes are higher because `cudaMalloc` does not zero at all, so
there is no fallback if the clear is skipped. See "What CUDA must do differently", item 2.

---

### G4. MEDIUM — the legacy accumulation path skips the clear, and is correct only by an undocumented coupling

**What.** With `G16_METAL_MSM_LEGACY_ACC=1` the host dispatches `msm_accumulate_*` and **does not
dispatch the clear kernel at all**.

**Where.** `msm.rs:970-993`: the `if legacy_accumulate()` arm (`msm.rs:970-979`) has no
`clear_pso` dispatch; only the `else` arm (`msm.rs:981-985`) clears.

**Why it matters.** It is correct, but for a reason stated nowhere near the branch:
`msm_accumulate_impl` writes `buckets[gid] = acc` **unconditionally** for every
`gid < n_windows * n_buckets` (`msm.metal:797`), including rows with zero count, where `acc` is
`pt_zero()`. So every bucket is fully overwritten and the previous proof's contents cannot survive.
Change that kernel to skip empty rows as an optimisation, which is an obvious and tempting change,
and the legacy path starts reading recycled buckets while the default path stays correct. The
comparison the flag exists to enable would then be comparing a correct kernel against a broken one.

**Fix.** Dispatch the clear on both paths. It costs one dispatch on a path that is already
documented as 3.6x slower (`msm.metal:824-828`), so the cost is irrelevant and the coupling goes
away. Failing that, put the invariant in a comment at `msm.rs:970` and in `msm.metal:774`.

---

### G5. MEDIUM — no test targets the exceptional addition cases directly, and the only all-artifact test is `#[ignore]`d

**What.** The unit-level MSM tests build their bases as successive multiples of the generator:

```
backend/msm.rs:1213-1216
let mut cur = G1Projective::generator();
for i in 0..n { cur += G1Projective::generator(); bases.push(cur.into_affine()); ... }
```

so every base is distinct and no base is the negative of another and none is the identity. The
tests are deliberately dense in **scalar** 0 and 1 (`msm.rs:1218-1222`, and the comment at
`msm.rs:1217` says so), which is good, but they never present `P == Q`, `P == -Q`, or a base at
infinity to `pt_madd`.

**Where.** `msm.rs:1199-1241` (`small_g1_msm_matches_a_naive_sum`), `msm.rs:1242-1271`
(G2 twin). The all-artifact sweep `five_msms_match_cpu_on_every_artifact` is at `msm.rs:1708` and
is `#[ignore]`d ("minutes on the CPU oracle side"), VERIFIED from the `--list` output.

**Why it matters.** The guards at `msm.metal:462`, `msm.metal:465`, `msm.metal:472-480` are the
single most important correctness property in this backend, and nothing in the default test run
exercises them on purpose. They are exercised **incidentally**, by
`five_msms_match_cpu_on_tiny_mul` and `five_msms_match_cpu_on_js_1x1_d8` and by
`backend::tests::a_metal_proof_verifies_on_every_artifact`, since real zkeys contain infinity
points. But incidental coverage of a branch means that deleting the branch might still pass, and
which exceptional pair gets presented to the formula depends on the nondeterministic scatter order
(`msm.metal:755`), so incidental coverage is not even stable run to run.

**Fix.** One new test file with three deterministic cases, each checked against a naive host sum:
(a) a base vector containing `G1Affine::identity()` at several positions with non-trivial scalars;
(b) a base vector containing exact duplicate bases assigned scalars that land them in the same
bucket; (c) a base vector containing `P` and `-P` pairs. For (b) and (c), pin the window with
`G16_METAL_MSM_C` and pin the slice length with `G16_METAL_MSM_L=1` so the bucket assignment is
forced rather than hoped for. This is the highest-value test to add and it also becomes the CUDA
port's acceptance test.

---

### G6. LOW — spill tag is stored before the spill point, which is safe now and unsafe if the phases are ever fused

**What.** In `msm_segmented_impl` the row tag is written first and the point second:

```
msm.metal:917-919   spill_rows[head_slot] = cur_row;
                    spill_pts[head_slot]  = acc;
```

and again at `msm.metal:938-944`.

**Why it matters.** Nothing today: `msm_merge_*` is a separate dispatch on a serial encoder, so
all of the segmented kernel's stores are visible before any merge thread runs. But the merge reads
`spill_pts[slot]` **guarded by** `spill_rows[slot] == row` (`msm.metal:995-1002`), which is a
publish/consume pair, and it is published in the wrong order. Fuse the two phases into one kernel
with a grid-wide barrier (the obvious "save a dispatch" optimisation, and exactly what sppark's
`this_grid().sync()` invites) and this becomes a real reordering hazard.

**Fix.** Swap the two stores so the point is published before its tag, and note the reason. Free.

**CUDA.** Same swap, plus a `__threadfence()` between them if the phases are ever fused. Better:
keep them as separate launches, which is what cuZK does and what item 4 below recommends anyway.

---

### G7. LOW — the infinity sentinel's survival through negation is load-bearing and undocumented

**What.** `msm.metal:931` (and `msm.metal:793`) negate `b.y` for a negative signed digit *before*
`pt_madd` checks `aff_is_inf(p)` at `msm.metal:462`. Correctness therefore requires
`f_neg((0,0)) == (0,0)`.

**Why it matters.** It holds: `fq_neg(0) = fq_sub(0, 0)`, and `fq_sub` (`msm.metal:152`) produces
borrow 0, mask 0, result 0. VERIFIED by reading. But it is an implicit contract between the
sentinel encoding, the negation, and the guard's position, spelled out nowhere. A port that
represents infinity with a separate flag (as arkworks does, and as `layout.rs:344` deliberately
translates away) and negates by `y = q - y` would map the sentinel to `(0, q)` or to a set flag
plus a junk `y`, and the guard would stop firing on a third of the B query.

**Fix.** Either a comment at `msm.metal:929-932`, or move the `aff_is_inf` test to before the
negation at the two call sites, which costs nothing and removes the dependency entirely. I prefer
the latter.

---

### G8. LOW — recycled buffers are never scrubbed, so a proof's intermediate points persist in host-visible memory

**What.** `Pool::give` returns buffers to the free list unchanged (`msm.rs:463-471`) and `take`
hands them out for a different purpose (`msm.rs:441-461`). They are `StorageModeShared`, so they
are CPU-addressable for the process's lifetime. `Pool::give` also truncates the free list to 48
buffers (`msm.rs:469-471`), which releases the excess to Metal, again without scrubbing.

**Why it matters.** Not a correctness issue, and the bucket points are functions of public zkey
bases and the witness. It matters only insofar as the witness is the secret: bucket partial sums
are `sum(base_i)` over an index set determined by witness digits, and those buffers sit in
readable process memory long after the proof is returned. This is a remanence note for the leakage
dimension, not a break. Flagging it here because the pooling design is mine to audit and the
leakage audit will not be reading `msm.rs:463`.

**Fix.** Out of scope for this dimension; defer to the leakage audit. If it is wanted, scrub on
`give` rather than on `take`, so the cost is off the critical path.

---

### G9. LOW — the `stages.rs` scratch pool is unbounded

**What.** `HHandle::drop` pushes into the pool with no cap (`stages.rs:820-826`), unlike
`msm.rs`'s pool which truncates to 48 (`msm.rs:469-471`).

**Why it matters.** Each `Scratch` is seven domain-sized buffers, roughly 48 MB at `2^18`
(the figure in the comment at `stages.rs:320-321`). The high-water mark is bounded by peak
concurrency, so this is not a leak, but a burst of concurrent proves permanently pins peak times
48 MB. Minor.

**Fix.** Mirror `msm.rs`'s truncation, with a cap expressed in buffers rather than bytes.

---

### G10. INFORMATIONAL — two sizing invariants I verified because a port will break them

Both hold. Recording the proofs so the CUDA port has them.

**(a) The top window never produces an unpaid borrow.** The signed recoding
(`msm.metal:577-596`) borrows `2^c` when the raw window is in the top half and repays it through
the next window's carry bit. The top window has no neighbour above it, so an unpaid borrow would
be a wrong scalar. It cannot happen because `n_windows = RECODE_BITS.div_ceil(c)` with
`RECODE_BITS = 255` (`msm.rs:89`, `msm.rs:790`), so `n_windows * c >= 255`, so the borrow
threshold is `2^(n_windows*c - 1) >= 2^254`. And `2^254` is
`28948022309329048855892746252171976963317496166410141009864396001978282409984`, while the BN254
scalar field order is
`21888242871839275222246405745257275088548364400416034343698204186575808495617`, so every scalar
is below the threshold. A port that sizes windows as `ceil(254/c)` breaks this silently.

**(b) The entry array cannot overflow.** `cap = general.max(1)` (`msm.rs:792`) and the buffer is
`n_windows * cap * 8` bytes (`msm.rs:834`). Each general scalar emits at most one entry per window
(`msm.metal:748-757`), zero and one scalars emit none, so per-window entries are bounded by
`general` exactly. When the host cannot classify (device-resident H scalars) `general_in` returns
the full range length (`msm.rs:350`), which over-sizes. VERIFIED.

**(c) `G16_METAL_MSM_C` allows `c = 2`, which the cost model never selects.** `window_size` clamps
the env override to `[2, MAX_WINDOW]` (`msm.rs:169`) while the search loop starts at `c = 3`
(`msm.rs:182`). I checked `c = 2` by hand: digits land in `[-2, 2]`, so `mag - 1` is in
`[0, 1]` and `n_buckets = 2`, in range. No bug, but the asymmetry is worth a comment since the
override is the only way to reach it and it is therefore the least-tested configuration.

---

## What CUDA must do differently

The Metal backend is a sound model, but five of its correctness properties are supplied by Metal
rather than by our code. A line-by-line port inherits the code and loses the guarantees.

**1. Inter-dispatch ordering is free here and is not free in CUDA.** All five MSM phases are
encoded into one `MTLComputeCommandEncoder` created with `cb.new_compute_command_encoder()`
(`msm.rs:738`), which is `MTLDispatchTypeSerial`: dispatches execute in encode order and Metal
inserts the hazard barriers automatically for tracked resources. `count -> scan -> scatter`,
and `clear -> segmented -> merge -> reduce`, are ordered by the platform.
In CUDA that ordering exists only within a single stream. Launch all phases on one stream, or
insert explicit events. Do **not** reach for `cooperative_groups::this_grid().sync()` to fuse them
(sppark's approach): it requires `cudaLaunchCooperativeKernel` and a co-resident block count, and
cuZK's `cudaStreamSynchronize`-between-phases is both simpler and a closer match to what this code
already does.

**2. `cudaMalloc` does not zero. `MTLDevice.makeBuffer` does.** Our `spill_rows` initialisation
(`msm.metal:901-902`) and our `counts` zeroing (`msm.rs:853-857`) are explicit and port cleanly.
Our `buckets` clear is explicit too (`msm.rs:982-985`) but is **partial** (G3), and the legacy path
has none at all (G4). Under CUDA the legacy path would produce garbage on run one. Port with a
`cudaMemsetAsync` on the consuming stream for every bucket and spill array, on every path, and add
an explicit identity write on the empty path rather than trusting a memset to encode it.

**3. Every `Fq` output must stay canonical.** The exceptional-case guards at `msm.metal:462`,
`msm.metal:472-480`, `msm.metal:497-514` are equality tests on limb arrays (`fq_is_zero`
`msm.metal:107`, `fq_eq` `msm.metal:115`). They are sound **only** because `fq_add`, `fq_sub` and
`fq_mul` all reduce into `[0, q)`. Adopting lazy reduction (outputs in `[0, 2q)`), which is a
standard CUDA speedup and is what sppark's `uadd` variant plays with, breaks every one of those
guards at once: `q` and `0` stop comparing equal, `P == -Q` stops being detected, and the MSM
returns a wrong point on inputs that our benchmark zkeys demonstrably contain. If lazy reduction
is wanted, the guards must switch to a canonicalise-then-compare, and that must be a deliberate,
tested change.

**4. The PTX carry chain has no Metal analogue.** Our field arithmetic is written in portable
64-bit accumulator style (`msm.metal:141-151`, `msm.metal:193-231`) and the Metal compiler is free
to schedule it. A CUDA port that switches to `add.cc` / `addc.cc` inline PTX for speed inherits
PTX ISA 9.7.2's rule that `CC.CF` is not preserved across calls and is intended for straight-line
sequences. Any control flow, including compiler-inserted scheduling, between the two silently
yields a wrong field element. This is why sppark does `#define asm asm volatile`. If the port keeps
the portable form, this hazard does not exist; if it reaches for PTX, it does, and it produces a
wrong answer rather than a crash.

**5. Branchlessness must be preserved deliberately.** `fq_cond_sub_n`'s `select`
(`msm.metal:136`) and `fr_cond_sub_n`'s (`bn254_fr.metal:127`) map to CUDA predication, but only
if written as a select or `?:` and not as an `if`. On the measurements in the research note this
is worth 2.4x on divergent warps, and lambdaworks' divergent `if` is the anti-pattern to avoid.

**6. Port the exclusive-ownership design, not just the kernels.** The property that makes
`buckets[cur_row] = acc` (`msm.metal:924`) safe without any atomic is that the counting sort makes
each row contiguous and each interior run privately owned. If a port replaces the counting sort
with a direct scatter into buckets, it becomes lambdaworks' documented last-writer-wins race, and
there is no 256-bit atomic to fix it with. Keep the sort.

**7. Carry over the host-side infinity handling.** `PackedG1Affine::from_affine`'s flag test
(`layout.rs:344`) is not optional. zkmopro's packer drops the arkworks `infinity` flag and lifts
`(0, 0)` to a projective point that is not on the curve at all, and with `js_16x16_d32`'s B query
being 33.9% infinity that poisons a third of the MSM. The CUDA host packer must read the flag,
and the device must keep `f_neg(sentinel) == sentinel` or test for infinity before negating (G7).

---

## What I did not check

- I did not instrument the GPU to confirm the segmented kernel's exclusive-ownership claim
  empirically (for example by having each thread record which rows it direct-wrote and asserting
  the sets are disjoint). The argument above is from reading the code and the entry-array
  invariant, not from a run.
- I did not re-derive the research dimension's zkey measurements (33.9% infinity in
  `js_16x16_d32`'s B query, 64 exact `±` pairs in `js_2x2_d32`'s A query). I take them from
  `research-gpu.md` and rely on them only for arguing that the guards matter, not for any claim
  about our code.
- I did not audit `pointwise.metal` or the `backend.rs` stage-10/11 assembly, which fall under
  other dimensions.
- G2 is unresolved. I could not reproduce it in 150+ attempts and I am not asserting it is benign.
