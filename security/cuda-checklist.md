# CUDA backend: correctness checklist

Ordered by how bad it is to get wrong silently. Everything here is a way a GPU backend
produces a wrong or unsafe result **without an error**, drawn from the four audits in
`security/notes/`. Items the Metal backend already solves cite the file to copy from.

The Metal backend is a sound model, but five of its correctness properties come from Metal
rather than from our code. A line-by-line port inherits the code and loses the guarantees.
Those are items 3, 5, 6 and 7.

---

## 1. Blinders never touch the device

**Hazard.** `r` and `s` generated on device destroy zero knowledge silently and
unauditably. cuRAND's default generators (XORWOW, MRG32k3a, Philox) are simulation PRNGs,
not CSPRNGs. A proof with predictable blinders verifies under our verifier, under snarkjs
and under rapidsnark, so no oracle catches it.

**Requirement.** `r` and `s` stay on the host, drawn from the `RngCore + CryptoRng` that
`snarkrs_groth16::prove` already owns. The backend trait never asks for randomness:
`PreparedCircuit` (`crates/groth16/src/lib.rs:146-165`) exposes `compute_h` and `msms` and
nothing else. Do not link cuRAND into the proving path. Do not add a `--seed` flag to the
CLI; use `prove_with_blinders` for reproducible debugging, as the Metal integration tests
do (`crates/metal/tests/audit_metal_vs_cpu.rs:113`).

**Test.** `grep -rn 'curand\|rand' crates/cuda/src/` returns nothing in the proving
path. Prove the same witness twice through `prove()` and assert `A`, `B` and `C` all
differ, then assert both verify.

---

## 2. The infinity flag must survive the host pack, and the device must skip it

**Hazard.** The `.zkey` encodes infinity as affine `(0,0)`, which is not on the curve.
A Jacobian or XYZZ mixed-addition formula lifts `(0,0)` into a live, off-curve point that
poisons whichever bucket it lands in. The result is a wrong MSM and an unverifiable proof
with nothing else to go on. This is not hypothetical: `js_16x16_d32`'s B query is 33.9%
infinity, and `b_g2_query[0]` in `bench/artifacts/tiny_mul` is genuinely 64 zero bytes, so
the smallest artifact we ship exercises the path. zkmopro's packer has exactly this bug.

**Requirement.** Read the arkworks `infinity` flag at the host pack, never re-derive it
from coordinates on the device. Copy `PackedG1Affine::from_affine`
(`crates/metal/src/layout.rs:338-351`, rationale at `layout.rs:141-157`). Keep the all-zero
sentinel convention and keep the assertion that `(0,0)` is not on the curve
(`layout.rs:580,598`). The device must test for infinity **before** any negation for a
signed digit, or preserve `f_neg(sentinel) == sentinel` deliberately: Metal's sign flip at
`msm.metal:931` happens before the `aff_is_inf` check at `msm.metal:462` and is safe only
because `fq_neg(0) == 0`. A port that negates by `y = q - y` maps the sentinel to `(0, q)`
and the guard stops firing on a third of the B query. Prefer moving the infinity test
before the negation, which removes the dependency for free.

**Test.** Round-trip an infinity base host to device to host and assert it is unchanged.
Run an MSM whose base vector contains `G1Affine::identity()` at several positions with
non-trivial scalars, against a naive host sum. Then run the five MSMs on `tiny_mul` and
`js_16x16_d32` against the CPU oracle.

---

## 3. Every `Fq` output stays canonical. No lazy reduction.

**Hazard.** The exceptional-case guards are limb-equality tests (`fq_is_zero`
`msm.metal:107`, `fq_eq` `msm.metal:115`). They are sound only because `fq_add`, `fq_sub`
and `fq_mul` all reduce into `[0, q)`. Lazy reduction (outputs in `[0, 2q)`) is a standard
CUDA speedup and it breaks **every guard at once**: `q` and `0` stop comparing equal,
`P == -Q` stops being detected, and the MSM returns a wrong point on inputs our benchmark
zkeys demonstrably contain. The device infinity test `x == 0 && y == 0` is also wrong on
non-canonical limbs, which folds this into item 2.

**Requirement.** Every field op ends with a conditional subtraction into `[0, q)`, as
`fq_cond_sub_n` does (`msm.metal:125-140`). If lazy reduction is wanted later it must be a
deliberate change that first converts the guards to canonicalise-then-compare. Keep the
conditional subtraction **branchless**, written as a select or `?:` and not as an `if`;
this is worth about 2.4x on divergent warps and lambdaworks' divergent `if` is the
anti-pattern.

**Test.** Fuzz each field op against `ark-ff` on random and boundary inputs (`0`, `1`,
`q-1`, `q-2`, values near `2q`) and assert bit-equal canonical output. Then the item 4
exceptional-case suite, which is what actually detects a guard that stopped firing.

---

## 4. The three exceptional addition cases must be guarded, and tested on purpose

**Hazard.** `P == Q`, `P == -Q` and an operand at infinity all break an incomplete
addition formula. `P == -Q` is self-healing under the raw formula, the other two are not,
so a partial guard set looks correct in testing. Metal has the full set
(`msm.metal:462` operand identity, `465` accumulator identity, `472-480` the same-x split),
mirrored in `pt_add` at `497-514`, but **nothing in either backend's default test run
exercises them on purpose**: the unit tests build bases as successive generator multiples,
so those pairs never reach the formula. Incidental coverage from real zkeys depends on a
nondeterministic scatter order, so it is not stable run to run.

**Requirement.** Port the full guard set, including the `f_is_zero(p.y)` 2-torsion case in
the doubling routines. Then write the deterministic test the Metal backend is missing.

**Test.** Three cases, each against a naive host sum, with the window pinned (`c`) and the
slice length pinned to 1 so bucket assignment is forced rather than hoped for:
(a) identity bases at several positions with non-trivial scalars; (b) exact duplicate bases
with scalars that land them in the same bucket, forcing `P == Q`; (c) `P` and `-P` pairs,
forcing `P == -Q`. This is the port's acceptance test, and it is worth writing before the
kernels are finished.

---

## 5. Check every launch and every sync. Never read a pooled buffer on an unchecked path.

**Hazard.** Metal has this bug and it is the highest-severity GPU finding: no command
buffer status is checked at any of six commit sites, so a GPU fault returns normally and
the next `read_back` reads the **previous proof's** window sums out of a recycled buffer.
The output is a silently wrong proof with no error anywhere. CUDA makes it easier to get
wrong: `cudaMemcpy` and `cudaStreamSynchronize` return sticky errors from *earlier* async
launches, so an unchecked `<<<>>>` surfaces as a confusing failure at the next sync, or not
at all.

**Requirement.** `cudaGetLastError()` after every launch, checked return code on every
sync and every copy, all funnelled through one helper that returns `ProveError`, so a
later call site cannot forget. No read-back happens on a path where an error was skipped.

**Test.** Force a fault (an illegal grid dimension or a deliberate out-of-bounds write in a
debug-only kernel) and assert `prove` returns `Err`, not a proof. Run the suite under
`compute-sanitizer`.

---

## 6. `cudaMalloc` does not zero, and Metal does. Initialise everything explicitly.

**Hazard.** The Metal backend already recycles buffers through a pool, which defeats
Metal's zero-fill from the second proof onward, so the initialisation is explicit and ports
cleanly. Two exceptions do not: the bucket clear writes **only** `zz`
(`msm.metal:864`), correct today only because every consumer tests `zz` first, and the
`G16_METAL_MSM_LEGACY_ACC=1` path dispatches **no clear at all**, correct only because that
kernel writes every row unconditionally. Under CUDA those produce garbage on run one.

**Requirement.** `cudaMemsetAsync` on the consuming stream for every bucket, spill and
counter array, on every path including any legacy or comparison path. Write the identity
explicitly rather than trusting a memset to encode it. Note the spill sentinel is
`MSM_NO_ROW = 0xffffffff`, **not** zero, because row 0 is a valid row
(`msm.metal:855,901-902`); a port that leans on a zeroed allocation folds slot 0's garbage
into bucket 0 of window 0. The sentinel stores must stay **before** the bounds early
return, as they are in Metal.

**Test.** Seed every pooled buffer with a known non-identity point before an MSM whose row
set does not cover it, and assert the answer is unchanged. Run the same MSM twice from one
pooled context and assert bit-identical results.

---

## 7. Phase ordering is a Metal guarantee. In CUDA it must be one stream.

**Hazard.** All five MSM phases are encoded into one `MTLComputeCommandEncoder` created
with `MTLDispatchTypeSerial` (`msm.rs:738`): dispatches run in encode order and Metal
inserts hazard barriers automatically. `count -> scan -> scatter` and
`clear -> segmented -> merge -> reduce` are ordered by the platform, not by our code. In
CUDA that ordering exists only within a single stream.

**Requirement.** Launch all phases on one stream, or insert explicit events. Do **not**
reach for `cooperative_groups::this_grid().sync()` to fuse phases: it needs
`cudaLaunchCooperativeKernel` and a co-resident block count, and cuZK's
`cudaStreamSynchronize` between phases is both simpler and a closer match to the code being
ported. If phases are ever fused anyway, note that Metal publishes the spill **tag** before
the spill **point** (`msm.metal:917-919`), which is a publish/consume pair in the wrong
order: swap the two stores, and add a `__threadfence()` between them.

**Test.** `compute-sanitizer --tool racecheck` on the full MSM. Run one MSM 100 times and
assert bit-identical output, then run concurrent proves against one `PreparedCircuit` and
assert every proof verifies.

---

## 8. Port the exclusive-ownership design, not just the kernels

**Hazard.** The property that makes `buckets[cur_row] = acc` safe with no atomic
(`msm.metal:924`) is that the counting sort makes each row contiguous, so an interior run
is privately owned by one thread, and boundary runs go to per-thread spill slots `2*gid`
and `2*gid+1`. Replace the counting sort with a direct scatter into buckets and it becomes
lambdaworks' documented last-writer-wins race, and there is no 256-bit atomic to fix it
with. The failure is a wrong point on some runs and not others.

**Requirement.** Keep the sort. Keep the two-slot-per-thread spill and the one-thread-per-
row merge. Atomics stay on `uint` counters only.

**Test.** As item 7, plus a run where each thread records which rows it direct-wrote,
asserting the sets are disjoint. Neither backend has this test today.

---

## 9. Validate the zkey header before any `cudaMalloc`

**Hazard.** `domainSize` is bounded only to nonzero-power-of-two, and four host vectors are
sized from it before the section-9 length that would refute it is checked. Measured: 34 GB
and 11.49 s from a 4 KB file. On the host that degrades to a slow error under overcommit.
`cudaMalloc` sized from the same unvalidated field fails hard, and the failure path in a
CUDA backend is typically an error code that is easy to `unwrap`.

**Requirement.** Enforce `domain_size <= 1 << Fr::TWO_ADICITY` and cross-check the section
9 length **before** any device allocation. Never size a `cudaMalloc` from a header field
that has not been cross-checked against a section length. This is a `snarkrs-formats` fix that the
CUDA lane depends on rather than owns; until it lands, do not allocate device memory from
`domain_size` on an untrusted key.

**Test.** A mutated zkey with `domainSize = 2^31` must error before allocating anything,
host or device. Measure peak RSS and peak device memory to confirm.

---

## 10. Do not DMA from the mmap, and scrub what you free

**Hazard.** `cudaHostRegister`ing the `.zkey` mapping to skip a copy on a 1 GB key converts
a load-duration SIGBUS window into a proving-duration one, and a SIGBUS taken inside a
pinned-memory DMA is considerably less pleasant than one taken on a CPU load. Separately,
`cudaFree` does not zero and `cudaFreeHost` does not either, so a pooled allocator inside
one process hands the previous proof's witness-derived points to the next caller, and
device VRAM is a longer-lived and less observable store than Apple's unified pages.

**Requirement.** Keep the invariant that `ProvingKey` owns every byte and the mapping dies
at the end of `load()` (`crates/formats/src/lib.rs:101`). If zero-copy upload is wanted, copy the
section into pinned host memory first and upload from there. `cudaMemsetAsync` on release,
not on acquire, so the cost is off the critical path; memset pinned host buffers before
`cudaFreeHost`. Do not stash the H buffer on the circuit struct: `HPoly::Device` carries the
handle through the *value* so a `&self` circuit stays safe for concurrent proofs. Copy
`HHandle`'s ownership shape (`crates/metal/src/stages.rs:757-826`).

**Test.** Truncate a key file mid-load in a loop and confirm the crash window is bounded by
`load()` and does not extend into proving. Allocate, free, reallocate and read back, and
assert zeros.

---

## 11. Window count comes from 255 bits, not 254

**Hazard.** The signed recoding borrows `2^c` when the raw window is in the top half and
repays it through the next window's carry bit. The top window has no neighbour above it, so
an unpaid borrow is a wrong scalar. It cannot happen only because
`n_windows = RECODE_BITS.div_ceil(c)` with `RECODE_BITS = 255` (`msm.rs:89,790`), which
puts the borrow threshold at `2^254`, above the BN254 scalar field order. A port that sizes
windows as `ceil(254/c)` breaks this **silently**.

**Requirement.** Keep 255. Also note that an env override allows `c = 2` while the cost
model never selects it, so `c = 2` is the least-tested configuration if such an override is
ported.

**Test.** For every `c` the backend can select, and for `c = 2` if reachable, run an MSM
with scalars near the field order (`r-1`, `r-2`, `2^253`, `2^254 - 1`) against the CPU
oracle.

---

## 12. Normalise output on the host, and keep the three-oracle property

**Hazard.** Our own `proof.json` reader accepts any nonzero Jacobian `z`, so emitting raw
Jacobian would appear to work. snarkjs and rapidsnark are not obliged to accept it, and
agreement across the three verifiers is the entire point of the JSON format.

**Requirement.** Normalise MSM output to affine on the host and write `z = 1`. Fully reduce
Montgomery limbs before any host-visible comparison, including any cross-backend equality
assertion against the CPU reference, which will otherwise fail spuriously.

**Test.** The existing CPU-versus-backend parity test shape:
`prove_with_blinders(cpu, w, r, s)` and `prove_with_blinders(cuda, w, r, s)` must serialise
to the **same compressed bytes** on all six artifacts, and independent blinders must produce
**different** bytes, so the test cannot pass on a backend that ignores its inputs. The Metal
version is `metal_and_cpu_agree_on_every_variant` in `bin/snarkrs/tests/campaign.rs`.
Then `snarkjs groth16 verify` and rapidsnark on the output.

---

## 13. Do not widen the side channel, and do not print from kernels

**Hazard.** The zero and one scalar fast path already makes running time and allocation a
function of witness sparsity. On CUDA the witness-dependent quantity becomes a **grid
dimension**, visible in Nsight, in `nvidia-smi` utilisation sampling, and to any co-tenant
on a shared or MPS-partitioned GPU. That is a far more accessible observation point than a
wall-clock reading on a laptop CLI. Separately, device-side `printf` writes to a host buffer
the driver flushes on sync, so a debug print of a scalar or a bucket index is a witness leak
into stdout.

**Requirement.** Size the grid and the entry array from `n`, not from the general-scalar
count, and take the sub-2% hit measured on our artifacts. Keep the fast path inside the loop
if it is worth it; what leaks is the launch geometry. Gate any kernel `printf` behind a
non-default feature and keep it out of release builds.

**Test.** Two witnesses of the same length with very different sparsity must produce
identical grid dimensions and identical allocation sizes. Assert it directly rather than
eyeballing a profiler.

---

## 14. If you reach for PTX, keep the carry chain straight-line

**Hazard.** Our field arithmetic is portable 64-bit accumulator style and the compiler is
free to schedule it. Switching to `add.cc` / `addc.cc` inline PTX for speed inherits PTX ISA
9.7.2's rule that `CC.CF` is not preserved across calls and is intended for straight-line
sequences. Any control flow between the two, including compiler-inserted scheduling,
silently yields a wrong field element. This is why sppark does `#define asm asm volatile`.

**Requirement.** Prefer the portable form, in which case this hazard does not exist. If PTX
is used, mark it `volatile`, keep each carry chain in one basic block, and never let a
function call or a branch sit inside one.

**Test.** The item 3 field fuzz against `ark-ff`, run at every optimisation level the build
uses, plus a full-scalar MSM against the CPU oracle. A broken carry chain produces a wrong
answer, not a crash, so only differential testing finds it.
