# GPU correctness hazards for MSM and NTT (BN254 Groth16)

Research note for the CUDA backend now being written (`crates/g16-cuda`) and for auditing the
existing Metal backend (`crates/g16-metal`). Report only; no source file was modified.

Every claim below is tagged:

- **[VERIFIED]** — I read our source, read the upstream source, or ran a measurement myself in this session.
- **[SOURCE]** — quoted from a primary document (vendor doc, paper, upstream repo) that I fetched.
- **[REASONED]** — my analysis, derived from the above, not directly quoted.

One correction up front: the task brief asked me to cover "shared memory bank conflicts causing
wrong results rather than just slowness". That premise is wrong. NVIDIA documents bank conflicts as
a pure throughput effect, and the hardware resolves them by serializing, not by dropping accesses.
The real shared-memory correctness hazards are different, and are covered in §7.3. I would rather
say so than write a section defending a claim that is not true.

---

## 1. Bucket accumulation races in parallel Pippenger

### 1.1 Why the naive kernel is a race

Pippenger's accumulation phase is, per window `w`:

```
for i in 0..n:  buckets[w][digit(s_i, w)] += P_i
```

Mapped to the GPU as one thread per `(i, w)` pair, that inner statement is a
read-modify-write of a 96–128 byte object (Jacobian or XYZZ, 3–4 field elements of 32 bytes).
Two threads whose scalars produce the same digit in the same window both read the old bucket,
both compute `old + P`, and both store. One store overwrites the other. The point that lost is
simply gone from the sum. **[REASONED]**

There is no hardware fix available: the widest atomic on any GPU is 128 *bits*
(`atomicCAS` on `unsigned long long`, or a 16-byte `red`/`atom` in newer PTX), and a BN254 G1
XYZZ accumulator is 128 *bytes*. G2 is 256 bytes. So `buckets[d] += P` can never be a single
atomic instruction. **[REASONED, consistent with our own `crates/g16-metal/src/shaders/msm.metal`
header comment: "There is no 32-byte atomic on any GPU"]** **[VERIFIED — read our source]**

Crucially, this is a *silent* race. The result is a valid group element, just the wrong one.
There is no NaN, no crash, no assertion. In Groth16 it surfaces only as a pairing check that
fails, and if the loser point happens to be the identity (very common, see §5) the proof can even
pass. Bucket races cannot be found by "does it run" testing; they need a CPU cross-check.
**[REASONED]**

### 1.2 lambdaworks: a self-admitted race, still in `main`

`crates/gpu/src/metal/shaders/msm/msm.metal` in lambdaclass/lambdaworks, fetched from
`raw.githubusercontent.com/lambdaclass/lambdaworks/main/...` in this session. Verbatim: **[VERIFIED — fetched and read]**

```
// Bucket accumulation kernel
// WARNING: This kernel has a RACE CONDITION when multiple threads write to the same bucket.
// For production use, implement one of:
// 1. Sorting-based approach (cuZK paper) - sort (bucket_idx, point) pairs, then scan
// 2. Atomic operations for point addition (complex for 256-bit)
// 3. Per-thread local buckets with tree reduction
//
// Current behavior: The last thread to write to a bucket wins, producing incorrect results
// when multiple points map to the same bucket.
```

and at the write site inside `kernel void bucket_accumulation(...)`:

```
    // Load current bucket value, add point, store back
    // WARNING: This read-modify-write is NOT atomic and causes race conditions
    // when multiple threads access the same bucket.
    JacobianPoint bucket = load_bucket(buckets, global_bucket_idx);
    bucket = jacobian_add(bucket, p, BLS12_381_P, BLS12_381_INV);
    store_bucket(buckets, global_bucket_idx, bucket);
```

Note what the comment enumerates: the *same three* fixes listed below. This is the canonical
worked example of "a GPU MSM that compiles, runs, is fast, and is wrong".

Two secondary bugs in the same file, worth naming because they are the same class of thing:

- `bucket_accumulation` never initializes `buckets`; it only ever does read-modify-write on
  whatever was in the buffer. It relies on the host having zeroed it. **[VERIFIED — read the kernel]**
- `bucket_reduction` calls `load_point(buckets, ...)` rather than `load_bucket(...)`. Both
  functions are byte-identical in that file, so it happens to work, but the pairing of
  `store_bucket`/`load_point` is a landmine for any future layout change. **[VERIFIED]**

### 1.3 The three standard fixes, and which one to pick

**(a) Per-thread bucket copies + tree merge.** Each thread gets a private `2^(c-1)`-entry
accumulator array; merge at the end. Removes the race by construction. Dead on arrival on
arithmetic grounds: at `c = 11` and 128-byte XYZZ that is 128 KB *per thread*, and with 10^5
threads that is 12 GB before you have added a single point. Worse, the merge itself costs
`threads * 2^(c-1)` point additions, which for any realistic thread count exceeds the `n`
additions it was parallelizing. Viable only in a coarse form: a small fixed number of *replicas*
(e.g. 8–32 copies of the bucket array, thread `t` writes replica `t % R`), which cuts the collision
probability by `R` but does not eliminate it, so it still needs one of the other two. **[REASONED]**

**(b) Exclusive row ownership (cuZK / sppark).** Sort or count the `(bucket, point)` pairs so
each bucket's contributing points are a contiguous run, then give exactly one thread ownership of
one bucket. After the sort, no two threads ever touch the same bucket, so the accumulation needs
no synchronization at all.

cuZK phrases this as sparse-matrix algebra: build an ELL matrix with `t` rows (one per thread) and
`2^s - 1` columns (one per bucket), convert to CSR, transpose in parallel, then SpMV against the
all-ones vector. Quoting the paper (ePrint 2022/1321, fetched and read this session): **[SOURCE]**

> "we first divide EC points into t parts. For each part, we add and store the points with the
> same scalar value into the same entry of a sparse matrix. This matrix is in ELL format with t
> rows and 2s −1 columns... we convert this sparse matrix from ELL format to CSR format and then
> transpose it. Next, we perform the sparse matrix-vector multiplication (SPMV) on the transposed
> matrix with a scalar vector whose elements are all equal to 1."

and on synchronization:

> "To guarantee correct calculations in Algorithm 3 and Algorithm 4, we use stream barriers,
> implemented by the cudaStreamSynchronize function, to synchronize all launched threads. These
> barriers in our algorithm provide the same functionality as global barriers..."

That is the key structural point for the CUDA lane: cuZK's correctness argument rests on
**global** barriers between phases, not on intra-block `__syncthreads()`. Count, prefix-sum,
scatter, and accumulate are four *separate kernel launches* (or stream-synchronized phases),
because a prefix sum over the whole bucket array cannot be completed inside one block.

sppark implements the same shape without the sparse-matrix vocabulary. In
`msm/pippenger.cuh` (fetched this session), `accumulate()` reads a precomputed `histogram` to
find its own bucket's run, then: **[VERIFIED — read the source]**

```
        if ((len -= idx) && !(x == 0 && y == 0)) {
            ...
            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);
            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                ... bucket.add(p, digit >> 31);
            }
            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }
```

Three things to copy from this exactly:

1. The bucket is accumulated in a **register** (`bucket_t bucket`) and stored **once** at the end.
   There is no read-modify-write on device memory at all.
2. The only atomics in the whole kernel are `atomicAdd(&current, ...)` on a single `uint32_t`
   work-stealing counter — a scalar, never a curve point.
3. The `else` branch explicitly writes `.inf()`. sppark does **not** rely on the buffer being
   pre-zeroed. See §4.

**(c) Atomics.** Only usable indirectly: a 32-bit `atomicAdd` on a *counter*, plus lock-free
scatter of *indices*. A spin-lock per bucket (`atomicCAS` on a lock word, then a plain
read-modify-write of the point) is technically correct but pathological on a GPU: lanes in the
same warp contending for the same lock deadlock under pre-Volta lockstep execution, and even on
Volta+ with independent thread scheduling the serialization destroys the point of using a GPU.
Do not build this. **[REASONED]**

### 1.4 What our Metal backend does, and what CUDA should copy

Our `msm.metal` implements (b) as a **counting sort**, which is strictly cheaper than cuZK's
transpose because the keys are already dense small integers: **[VERIFIED — read the source and the
kernel list]**

- `msm_count` — 32-bit `atomic_fetch_add_explicit(&counts[...], 1u, memory_order_relaxed)` on
  per-bucket counters.
- `msm_scan` — threadgroup prefix sum over each window's counts, producing row offsets.
- `msm_scatter` — 32-bit `atomic_fetch_add_explicit(&cursor[row], 1u, ...)` to claim a slot,
  writes a `uint2` of `(row, point_index_with_sign_bit)`.
- `msm_segmented_*` — one thread per *slice* of the entry array, not per bucket, so load imbalance
  (measured 691x on `js_2x2_d32` at `c=11`, per the source comment) does not serialize the dispatch.
  Runs that straddle a slice boundary spill to `spill_pts` / `spill_rows`; strictly interior runs
  are written directly with no synchronization.
- `msm_merge_*` — one thread per bucket folds its own spills in.

The atomics are on `uint`s only, never on field elements. This is the correct design and the CUDA
lane should mirror it. **[VERIFIED]**

Two CUDA-specific translation notes:

- Our Metal `msm_scan` is one threadgroup per window, so the prefix sum fits in one block. In
  CUDA, if `n_buckets` exceeds one block's capacity you need a two-level scan (CUB's
  `DeviceScan::ExclusiveSum`) — a single-block scan silently produces wrong offsets for the tail
  buckets, and wrong offsets produce a *plausible but wrong* MSM. **[REASONED]**
- The Metal version relies on Metal's automatic hazard tracking to order `scatter → segmented →
  merge` within one command encoder. In CUDA, kernels launched into the **same stream** are
  ordered, but kernels in different streams are not. If the CUDA backend overlaps the five MSMs
  across streams (which it should, for throughput), each MSM's phases must stay within one stream,
  or be joined with events. **[REASONED]**

---

## 2. Non-deterministic reduction order

### 2.1 The result *should* be order independent

Elliptic curve point addition on a prime-order group is associative and commutative, so
`Σ P_i` is well-defined independent of order. In the counting-sort design, order *within* a bucket
is decided by which thread wins the `atomic_fetch_add` on the cursor — genuinely nondeterministic
run to run. Mathematically that is fine. **[REASONED]** Our own source states this explicitly:
"Order within a bucket is not preserved, which does not matter because bucket accumulation is
commutative." **[VERIFIED — read `msm.metal`]**

### 2.2 Where it stops being order independent

Five places, all of which have bitten real implementations:

**(a) Incomplete addition formulas.** The group law is commutative; the *formulas* are only
defined on a subset of input pairs. `add-2008-s` and `madd-2008-s` divide by `U2 - U1`, so they
produce garbage when `x1 == x2`. Whether that pair is ever *presented* to the formula depends on
the order in which the bucket is summed. Concretely: a bucket containing `{P, Q, -P}` summed in
that order computes `(P + Q)` then `((P+Q) + (-P))` and never hits an exception, while the same
bucket summed as `{P, -P, Q}` hits `P + (-P)` on step one. Same set, same correct answer in the
group, different exposure to the bug. **This is why an unguarded GPU MSM can pass a thousand tests
and fail on the thousand-and-first — the exception is order-triggered, and the order is decided by
a race.** **[REASONED]** This is the single most important finding in this note.

**(b) The projective representative differs, even when the point is right.** `(X, Y, ZZ, ZZZ)` and
`(Xλ², Yλ³, ZZλ², ZZZλ³)` are the same affine point. Two different summation orders give the same
affine point with *different* limbs. Any test that memcmp's GPU output against CPU output will
fail even when both are correct. Cross-checks must normalize (or compare via `ark_ec`'s `PartialEq`
on projective, which does the cross-multiply) before asserting. **[REASONED]**

**(c) Non-canonical field representatives.** If `f_add`/`f_mul` can return a value in `[q, 2q)`
rather than `[0, q)`, then `f_is_zero` and `f_eq` become order dependent: the same mathematical
zero compares equal on one path and not on another, and the exceptional-case guards
(`if (f_is_zero(pp_))`) then fire or don't fire depending on how the value was produced. I checked
ours: `fq_add` ends with `fq_cond_sub_n`, `fq_mul` (CIOS) ends with `fq_cond_sub_n(out, t[FQ_LIMBS])`,
and `fq_sub` masks in `FQ_N` on borrow — all outputs are canonical. **[VERIFIED — read
`msm.metal` lines 118–228]** **A CUDA port that adopts a lazy-reduction field (outputs in `[0, 2q)`,
reduce only at the end, which is a common speed trick) breaks every equality guard in the point
formulas and must switch the guards to `f_sub(a,b) == 0` after a full reduction.** **[REASONED]**

**(d) Floating point.** Not applicable — ZKP field arithmetic is integer-only. ZKProphet confirms:
"finite-field operations in ZKPs exclusively use the integer execution units." **[SOURCE]**

**(e) Atomic float accumulation.** Not applicable here for the same reason, but named because it
is the usual answer to "non-deterministic reduction order" in GPU literature and is a red herring
in this domain. **[REASONED]**

---

## 3. Incomplete addition formulas: exactly which pairs break

### 3.1 The exceptional sets

For **Jacobian mixed addition** (`madd-2007-bl`, `Z2 = 1`) with
`U2 = X2·Z1²`, `S2 = Y2·Z1³`, `H = U2 - X1`, `r = 2(S2 - S1)`:

| condition | what happens | correct answer |
|---|---|---|
| `H == 0` and `r == 0` (i.e. `P == Q`) | `Z3 = ... · H = 0`, output is the identity | should be `2P` |
| `H == 0` and `r != 0` (i.e. `P == -Q`) | `Z3 = 0`, output is the identity | is the identity — **accidentally correct** |
| `Z1 == 0` (`P` is the identity) | all of `U2, S2` become 0; `Z3 = 0` | should be `Q` |
| `Q` is the identity | Jacobian mixed addition has *no representation* for an affine identity — see §5 | should be `P` |

For **XYZZ addition** (`add-2008-s`) with `U1 = X1·ZZ2`, `U2 = X2·ZZ1`, `S1 = Y1·ZZZ2`,
`S2 = Y2·ZZZ1`, `P = U2-U1`, `R = S2-S1`:

| condition | what happens | correct answer |
|---|---|---|
| `P == 0, R == 0` (`P == Q`) | `PP = PPP = 0`, so `ZZ3 = ZZ3 = 0` → identity | should be `2P` |
| `P == 0, R != 0` (`P == -Q`) | `ZZ3 = 0` → identity | is the identity — accidentally correct |
| `ZZ1 == 0` (first operand is identity) | `U2 = S2 = 0`, `P = -U1`, generally nonzero → **produces a garbage point that is not on the curve** | should be `Q` |
| `ZZ2 == 0` (second operand is identity) | symmetric garbage | should be `P` |

For **XYZZ mixed addition** (`madd-2008-s`, `ZZ2 = ZZZ2 = 1`), same table with `U1 = X1`, `S1 = Y1`.

Note the asymmetry that trips people up: the `P == -Q` case is *self-healing* (the formula outputs
`ZZ3 = 0`, which is the identity encoding, which is the right answer), whereas `P == Q` and
"operand is the identity" are **not**. An implementation with only a `P == -Q` guard and no
doubling guard will look correct in ad-hoc testing and be wrong in production. **[REASONED]**

### 3.2 How likely is this with real witness data? Measured.

I parsed the six benchmark `.zkey` files in `bench/artifacts/` directly (Python, section walk over
the snarkjs zkey container: 4-byte magic `zkey`, `u32` version, `u32` nSections, then per section a
`u32` id and `u64` length). **[VERIFIED — ran it in this session]**

Points at infinity, stored in the zkey as literal all-zero 64-byte (G1) / 128-byte (G2) blobs:

| artifact | A (§5, G1) | B1 (§6, G1) | B2 (§7, G2) | C (§8, G1) | H (§9, G1) |
|---|---|---|---|---|---|
| `tiny_mul` | 1 / 6 | 4 / 6 | 4 / 6 | 0 / 2 | 0 / 8 |
| `js_1x1_d8` | 29 / 3,373 | 1,117 / 3,373 | 1,117 / 3,373 | 0 / 3,368 | 0 / 4,096 |
| `js_2x2_d16` | 87 / 10,194 | 3,423 / 10,194 | 3,423 / 10,194 | 0 / 10,187 | 0 / 16,384 |
| `js_2x2_d32` | 151 / 18,002 | 6,111 / 18,002 | 6,111 / 18,002 | 0 / 17,995 | 0 / 32,768 |
| `js_8x8_d32` | 595 / 70,640 | 23,979 / 70,640 | 23,979 / 70,640 | 0 / 70,621 | 0 / 131,072 |
| `js_16x16_d32` | 1,187 / 140,824 | **47,803 / 140,824** | **47,803 / 140,824** | 0 / 140,789 | 0 / 262,144 |

**33.9% of the B query bases on `js_16x16_d32` are the point at infinity.** This is not a corner
case, it is a third of the input. The A query is ~0.8%, and C and H have none.

Duplicate and negated bases in `js_2x2_d32`: **[VERIFIED — ran it]**

- `A_G1`: 18,002 entries, 17,852 distinct. Zero duplicated *non-infinity* points, but
  **64 distinct x-coordinates each appear exactly twice with y-values summing to `q`** — i.e. 64
  exact `±` pairs, 128 points. Sample index pairs: `(13, 147)`, `(14, 1221)`, `(15, 1222)`,
  `(16, 1223)`, `(17, 1224)`.
- `B1_G1`, `C_G1`, `H_G1`: zero duplicated points, zero shared-x groups (all their non-distinctness
  is the infinity blob).

So both exceptional inputs are present in a real, unremarkable circom zkey:

- **`P == -Q`** fires when base `i` and base `j = -i` land in the same bucket with the *same* sign
  digit.
- **`P == Q`** fires when they land in the same bucket with *opposite* sign digits, because the
  signed recoding negates one of them: `P` and `-(-P) = P`.

Rough rate for `js_2x2_d32`'s A MSM: 64 pairs × ~23 windows at `c = 11` × P(both scalars produce
the same digit magnitude in that window) ≈ 64 × 23 / 1024 ≈ **1.4 expected exceptional pairs per
proof**, before counting the infinity bases at all. This is not a "will never happen in practice"
hazard; it happens on essentially every proof. **[REASONED, from VERIFIED counts]**

### 3.3 The standard guards

The reference implementations converge on the same shape. sppark, `ec/xyzz_t.hpp`
(fetched this session): **[VERIFIED]**

```
        if (p2.is_inf()) { ... }
        else if (is_inf()) { ... }
        ...
        if (!P.is_zero()) {         /* X1!=X2 */          ... general add ...
        } else if (R.is_zero()) {   /* X1==X2 && Y1==Y2 */ ... double |p1| ...
        } else {                    /* X1==X2 && Y1==-Y2 */ p31.inf();
        }
```

zkmopro `shader/curve/xyzz.metal` (fetched this session) is the same, with a doc comment that
states the contract: "Handles a == identity, a == b (doubling) and a == -b." **[VERIFIED]**

Our `msm.metal` `pt_madd` has the full guard set and a comment that names the exact reason it
cannot be skipped: **[VERIFIED — read the source]**

```
    if (f_is_zero(pp_)) {
        // Same x. Either the same point (double it) or its negative (they cancel).
        // A bucket can hit both: the signed digit recoding puts a point and its negation
        // in different buckets, but two *different* bases in one bucket can still be
        // equal or opposite, and a zkey is not guaranteed to have distinct bases.
        if (f_is_zero(rr)) { return pt_dbl_affine(p); }
        return pt_zero<F>();
    }
```

The §3.2 measurement is the empirical confirmation of that comment: 64 such pairs exist in
`js_2x2_d32` alone. **[VERIFIED]**

**Actionable for the CUDA lane:** implement `add`/`madd` with the four-way guard from day one.
Do not defer it as an optimization to remove later. The cost is two `is_zero` tests on values you
have already computed (`P` and `R`), which is negligible against 7M+2S. The alternative — a
"complete" formula (Renes–Costello–Batina, projective, `a = 0` case: 12M + 3 add-by-`b3`) — is
correct for *all* inputs including the identity but costs ~1.7x a mixed addition, and is the wrong
trade for the accumulation inner loop. Use guards there; complete formulas are only worth it if
you need constant-time behaviour, which a prover does not (all inputs are public setup data plus
the prover's own witness). **[REASONED]**

---

## 4. Uninitialized device memory

### 4.1 Why an unzeroed bucket array is a correctness bug

Pippenger's accumulation phase leaves most buckets *untouched* — at `c = 11` there are 1024 buckets
per window and, on a sparse witness, the great majority receive no points. In the exclusive-ownership
design, a thread that owns an empty bucket either writes the identity or writes nothing. If it
writes nothing and the buffer was not zeroed, the reduction phase reads whatever bytes were there —
the previous proof's points, another allocation's data, or driver garbage — and folds them into the
window sum. The proof is then wrong in a way that varies run to run and disappears under a debugger
that happens to zero memory. **[REASONED]**

### 4.2 XYZZ makes the identity free *if* memory is zero, and this is the trap

In XYZZ, `ZZ == 0` encodes the identity. An all-zero 128-byte block therefore *is* a valid identity
element, and a zero-filled bucket array is already correctly initialized with no memset kernel.
zkmopro's shader says so in its first line of comment: "The identity is encoded as ZZ = 0, which
matches zero-initialized buffers." **[VERIFIED — read `shader/curve/xyzz.metal`]** Our own
`msm.metal` header says the same: "Identity is ZZ == 0, which is what a freshly allocated Metal
buffer already contains, so a bucket array needs no memset kernel." **[VERIFIED]**

This is true on Metal, and Apple documents it. `MTLDevice.makeBuffer(length:options:)` — fetched
from Apple's docs in this session — has the abstract: **[SOURCE]**

> "Creates a buffer the method clears with zero values."

**It is not true on CUDA.** The CUDA Runtime API doc for `cudaMalloc`, fetched this session, says: **[SOURCE]**

> "Allocates size bytes of linear memory on the device and returns in *devPtr a pointer to the
> allocated memory. The allocated memory is suitably aligned for any kind of variable. **The memory
> is not cleared.**"

and `cudaMallocManaged` carries the identical sentence. **[SOURCE]**

**This is the single highest-risk portability trap in porting our Metal design to CUDA.** The Metal
kernels are correct *because of a platform guarantee that CUDA explicitly does not make*. A
line-by-line CUDA port of `msm_segmented_impl`, which writes only the buckets its slices touch,
produces garbage on the first run. Fix: `cudaMemsetAsync(buckets, 0, bytes, stream)` before the
accumulate kernel, on the same stream, every time. It is a `memset` of `n_windows * n_buckets * 128`
bytes, which at `c = 11` and 23 windows is ~3 MB, i.e. ~10 µs of bandwidth. Do not try to save it.

Even on Metal, the guarantee is narrower than it looks. Three carve-outs I verified: **[VERIFIED]**

- `MTLHeap.makeBuffer(length:options:)` is documented only as "Creates a buffer on the heap" — **no
  zeroing clause**. Heap suballocation reuses memory. Ours does not use heaps today; if it ever
  does, the assumption silently breaks.
- `makeBuffer(bytesNoCopy:...)` obviously wraps existing memory and zeroes nothing.
- **Buffer pooling defeats it entirely.** Our `MetalMsm` has a `Pool` (`msm.rs`, `fn take`/`fn give`)
  that recycles buffers across proofs, and the source is explicit that the second proof onward gets
  dirty memory: "A pooled bucket array holds the previous proof's points, and a bucket that no
  slice writes directly has to read as the identity." That is why `msm_clear_g1`/`msm_clear_g2`
  exist and are dispatched before `msm_segmented_*`. **[VERIFIED — read `msm.rs` `Outputs::encode`]**

Two audit observations on our Metal clear path, neither currently a bug: **[VERIFIED — read the source]**

1. `msm_clear_impl` clears **only `zz`**, deliberately ("clearing the other three coordinates would
   be pure memory traffic"). That is sound today because every reader — `pt_is_zero`, `pt_add`,
   `pt_madd`, and the host conversion in `msm.rs` (`if zz.is_zero()`) — tests `zz` first. It is a
   latent trap: any future kernel that reads `x`, `y`, or `zzz` without first checking `zz` reads
   the previous proof's data. Worth an assertion or a comment at each new reader.
2. In the `G16_METAL_MSM_LEGACY_ACC=1` path the clear kernel is *not* dispatched. I checked
   `msm_accumulate_impl` and it writes `buckets[gid] = acc` unconditionally for every
   `gid < n_windows * n_buckets`, so every bucket is fully overwritten and the missing clear is
   harmless. Correct, but only by construction — if that kernel ever gains an early `return` for
   empty buckets it becomes a stale-read bug.

sppark takes the opposite, safer stance: it never relies on zeroed memory and writes `.inf()`
explicitly for empty buckets (`buckets[y][x].inf();` in the `else` branch of `accumulate`, and
`out.inf()` / `res.inf()` throughout `integrate`). **[VERIFIED]** For the CUDA backend, prefer
sppark's discipline: an explicit `memset` *and* an explicit `.inf()` on the empty path. The
redundancy costs nothing measurable and removes an entire bug class.

---

## 5. Zero scalars, one scalars, and points at infinity in the bases

### 5.1 Infinity in the bases: the `(0,0)` problem

arkworks represents the affine identity as `x = 0, y = 0` plus a separate `infinity: bool` field.
From `ark-ec` `models/short_weierstrass/affine.rs` (fetched this session): **[SOURCE]**

```rust
    pub const fn identity() -> Self {
        // Setting these to zero is *load-bearing* and important.
        // These are the values that represent the identity element
        // when `P::ZeroFlag` is `()`.
```

A GPU packer that copies `pt.x` and `pt.y` into a device buffer and **drops the flag** hands the
kernel the coordinate pair `(0, 0)`.

zkmopro's host packer does exactly this. `mopro-msm/src/msm/metal_msm/utils/limbs_conversion.rs`,
`pack_affine_and_scalars` (fetched this session) — it iterates `bases`, writes `pt.x` and `pt.y`
limb by limb, and never tests `pt.infinity` or `pt.is_zero()`: **[VERIFIED]**

```rust
        let mut coords = vec![0u32; num_elements * coords_per_point];
        ...
        // Coordinates are copied in arkworks' internal Montgomery representation
        // ... so no Montgomery reduction happens on the host
```

The GPU-side `xyzz_from_affine` then does `r.zz = r.zzz = one`, i.e. it lifts `(0,0)` to
`(0, 0, 1, 1)`. **[VERIFIED — read `shader/curve/xyzz.metal`]**

**The consequence.** BN254 G1 is `y² = x³ + 3`. At `(0,0)`: `0 ≠ 3`. `(0,0)` is not on the curve
and not in the group. Everything downstream is arithmetic on a meaningless coordinate pair:

- The bucket that receives it accumulates a value with no group-theoretic meaning; it is not "off
  by the identity", it is off by an element of no group at all, and there is no `k` such that the
  error is `k·G`.
- The error propagates through the window reduction and the Horner combination into the final `A`,
  `B`, or `C` element of the proof.
- The pairing check fails. The proof is rejected. It is a liveness failure, not a soundness one —
  but it is a liveness failure on **33.9% of the B-query bases** for `js_16x16_d32` (§3.2), so it is
  not intermittent, it is total.
- Worse, on a *sparse* witness the poisoned bases may all carry zero scalars and be skipped, so the
  bug hides. It surfaces the first time a nonzero witness value happens to select an infinity base.

Our Metal backend handles this correctly, with a design note that anticipated exactly this failure.
`crates/g16-metal/src/layout.rs`: **[VERIFIED — read the source]**

```
/// The all-zero encoding, `x == 0 && y == 0`, means the point at infinity. This is
/// unambiguous rather than a convention: BN254 G1 is `y^2 = x^3 + 3`, so `(0, 0)` fails
/// the curve equation (`0 != 3`) and can never be a real point.
/// ...
/// This matters in practice and is not a theoretical case: snarkjs zkeys really do
/// contain points at infinity in the A, B and C query vectors, and the reference Metal
/// MSM implementations that ignore the arkworks `infinity` flag lift `(0, 0)` into a live
/// non-identity projective point and poison the bucket it lands in.
```

with the kernel-side guard `aff_is_inf(p)` → `f_is_zero(p.x) && f_is_zero(p.y)` checked as the
*first* statement of `pt_madd`. §3.2's measurement is the numeric proof of that comment.

sppark's approach is different but equivalent: a distinct `affine_inf_t` type carrying the flag,
and `ZZZ = ZZ = field_t::one(a.is_inf())` on conversion, where `one(true)` yields zero. **[VERIFIED
— read `ec/xyzz_t.hpp`]**

**Actionable for CUDA:** pick one and enforce it with a `static_assert`-grade test. Either carry the
flag in a 65th/129th byte (costs alignment, gives a non-power-of-two stride and a straddling load)
or use the `(0,0)` sentinel (keeps a 64/128-byte power-of-two stride, needs a branch-free
`x|y == 0` test over 16 words). Our Metal chose the sentinel for the stride reason; the CUDA lane
should match so the two backends share the packing code and the same tests. Whichever is chosen,
**write a test that feeds a base vector containing infinity at index 0, in the middle, and at the
last index, with a nonzero scalar on each, and asserts equality against `ark_ec::VariableBaseMSM`.**

### 5.2 Zero scalars

A zero scalar contributes nothing and must be *skipped*, not routed to bucket 0. Two failure modes:

- **Correctness:** if `digit == 0` maps to bucket index `0` rather than being skipped, and bucket 0
  is later weighted by `0` in the reduction, the answer is accidentally right; if the reduction
  weights buckets `1..2^c` and bucket 0 is included, it is wrong. cuZK is explicit: "the points
  corresponding to zero scalars have no effect on the final result and can be skipped." **[SOURCE]**
  sppark encodes the same by excluding `(x == 0 && y == 0)` from accumulation (`!(x == 0 && y == 0)`
  in the `if` above — that is window 0, bucket 0). **[VERIFIED]**
- **Performance collapse:** in a bit-heavy circom circuit, most witness values are `0` or `1`. If
  they are not special-cased they all pile into a handful of buckets and the single thread owning
  that bucket serializes the entire dispatch.

### 5.3 One scalars — the trap that turns a GPU into a single thread

This one is specific to Groth16 with circom witnesses and is worth stating loudly because it is not
in any of the papers. Our `msm.metal` header names it: **[VERIFIED — read the source]**

```
//    Leaving the ones in Pippenger would have been correct but pathological, and this is
//    the specific trap worth naming: the signed recoding sends every scalar equal to 1 to
//    digit +1 of window 0, so bucket (0, 0) would collect *every* one-scalar, and since
//    one thread owns one bucket that single thread would serially accumulate 100k points
//    while the other quarter-million threads sat idle. Special-casing 1 is not a
//    micro-optimisation here, it is what stops the dispatch degenerating to one thread.
```

Four of the five Groth16 MSMs take the witness as scalars, and in a bit-heavy circuit "over 99% of
those are 0 or 1" (our source's claim). Our host code filters at `if !(s.is_zero() || s.is_one())`
before the counting sort and routes ones to a dedicated `msm_ones_*` kernel. **[VERIFIED — read
`msm.rs` line 598]**

The correctness angle: the `ones` path must produce *exactly* `Σ P_i` over the one-scalar indices
and must not double-count with the general path. A CUDA port that special-cases ones must make the
two paths a strict partition of the index set, and must test the boundary (a witness of all ones, a
witness of all zeros, a witness with exactly one non-{0,1} value). **[REASONED]**

---

## 6. Montgomery reduction on the GPU: the conditional-subtraction branch

### 6.1 The measurement

ZKProphet (arXiv 2509.22684, "ZKProphet: Understanding Performance of Zero-Knowledge Proofs on
GPUs"), fetched and text-extracted in this session. The relevant passage: **[SOURCE]**

> "This computation can be divided into two portions: a field operation (compute), control flow
> operations to determine if a reduction is necessary (branch), and a field operation to reduce the
> value (compute). Our investigation shows that the branches determining conditional reduction make
> up **70.5% of the overall execution latency**. In the absence of any branches, compute operations,
> FF_add and FF_sub, require 72 cycles each."

and the divergence data: **[SOURCE]**

> "FF_add and FF_sub exhibit branch efficiencies of **52.5% and 56.2%** respectively, stemming from
> the sequential comparison between the corresponding limbs of the result and the field
> modulus/zero. These divergences causes a **2.4× increase in execution cycles (72 to 244)**."

> "The branch divergence is only responsible for 3.8% of the total cycles in FF_mul and FF_sqr
> compared to 70.5% for FF_add and FF_sub. **Branch efficiency is critical for optimizing
> performance**, as this metric is less than 50% in MSM implementations."

Why `FF_add`/`FF_sub` diverge and `FF_mul` does not: after a multiply, *almost every* product needs
the final reduction, so all 32 lanes take the same branch (96.9% efficiency for `FF_sqr`). After an
add of two uniformly random field elements, roughly half need it, and the *limb at which the
comparison resolves* differs per lane, so the lanes fan out across many branch targets. ZKProphet
attributes it to "the sequential comparison between the corresponding limbs". **[SOURCE + REASONED]**

### 6.2 What branchless looks like, in production

sppark's `ff/mont_t.cuh` `operator-=` (fetched this session) is the canonical branchless
conditional add-back. It computes the add-back **unconditionally** into a scratch array and then
applies it with a *predicated move*, never a jump: **[VERIFIED]**

```
        asm("sub.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(b[0]));
        for (i = 1; i < n; i++)
            asm("subc.cc.u32 %0, %0, %1;" : "+r"(even[i]) : "r"(b[i]));
        asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

        asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
        for (i = 1; i < n-1; i++)
            asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
        asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));

        asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
        for (i = 0; i < n; i++)
            asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));
        asm("}");
```

Three properties to copy:
1. The borrow is captured as a **value** (`subc.u32 borrow, 0, 0` — the standard idiom for
   materializing CC.CF), not consumed by a branch.
2. The corrected result is computed on both paths; only the *select* is conditional.
3. The select is `@%pred mov.b32`, which is predicated execution — no divergence, no reconvergence
   point, uniform 1-cycle-per-limb cost.

sppark also carries `add` and **`uadd`** variants of the point addition. Reading `ec/xyzz_t.hpp`, I
confirmed `uadd` is not "unsafe add" — it is the **uniform / divergence-free** add: the same
complete case analysis, but expressed as a `switch`-scheduled straight-line sequence with
`field_t::csel(...)` and `czero(...)` instead of `if`/`else`. It is selected at the call site by
`if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE) bucket.add(p, ...) else bucket.uadd(p, ...)`,
i.e. the branchy version is used only when the point is small enough that its code fits the L1
instruction cache. **[VERIFIED — read both files]** That is an unusually direct confirmation of
ZKProphet's conclusion in shipping code, and it is a design decision the CUDA lane should replicate
for G2 (256-byte points, where `uadd` is what sppark picks).

### 6.3 Our Metal backend already does this

`fq_cond_sub_n` (and its `Fr` twin) computes the subtraction unconditionally and selects with MSL's
`select()`, which lowers to a predicated/`csel` form rather than a branch: **[VERIFIED — read the source]**

```
inline Fq fq_cond_sub_n(Fq a, uint hi) {
    uint red[8];
    ulong borrow = 0;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)FQ_N[i] - borrow;
        red[i] = (uint)d;
        borrow = (d >> 63) & (ulong)1;
    }
    bool take = (hi != 0u) || (borrow == 0);
    Fq out;
    for (uint i = 0; i < FQ_LIMBS; i++) {
        out.v[i] = select(a.v[i], red[i], take);
    }
    return out;
}
```

Same shape as sppark's. Note the header comment already says "why this is a `select` rather than an
`if`". The `fq_is_zero`/`fq_eq` helpers are likewise branch-free (`acc |= ...` over all limbs, then
one compare) rather than early-returning, which matters because they are called inside the
exceptional-case guards on every mixed addition. **[VERIFIED]**

Contrast lambdaworks' Metal, which uses `if (borrow == 0) return reduced; return result;` in
`mont_reduce` and `if (carry || borrow == 0) return reduced;` in `field_add` — the divergent form
ZKProphet measured at 52.5% branch efficiency. **[VERIFIED — read `lw_msm.metal`]**

**Actionable for CUDA:** write `fq_add`, `fq_sub`, and the CIOS final reduction with
`setp` + `@%pred mov` (or `__builtin_assume`-free `csel`-style ternaries the compiler reliably
predicates), and verify with `cuobjdump -sass` that no `BRA`/`SSY` appears in the reduction. Do not
trust `-O3` to predicate a multi-limb comparison loop for you.

---

## 7. CUDA-specific hazards

### 7.1 The PTX carry chain is not modeled by the compiler — this is a correctness bug, not a perf one

PTX ISA 8.3 §9.7.2 "Extended-Precision Integer Arithmetic Instructions", fetched this session: **[SOURCE]**

> "Instructions add.cc, addc, sub.cc, subc, mad.cc and madc reference an implicitly specified
> condition code register (CC) having a single carry flag bit (CC.CF) holding carry-in/carry-out or
> borrow-in/borrow-out... **No other instructions access the condition code, and there is no support
> for setting, clearing, or testing the condition code. The condition code register is not preserved
> across calls and is mainly intended for use in straight-line code sequences** for computing
> extended-precision integer addition, subtraction, and multiplication."

Consequences for a hand-written bignum layer:

- Any control flow between `add.cc.u32` and the following `addc.cc.u32` can destroy the carry.
  This includes control flow the *compiler* inserted — nvcc does not model CC.CF and is free to
  schedule other instructions into the gap.
- Each limb operation must be its own `asm volatile` (sppark uses `#define asm asm volatile`
  precisely for this), and the whole chain must be straight-line. **[VERIFIED — read
  `ff/mont_t.cuh` lines 14–16]**
- This is why sppark computes `even + MOD` unconditionally *before* the predicate: putting an `if`
  in the middle of the carry chain would break it.
- A carry-chain break produces a *wrong field element*, not a crash. It shows up as a failing
  pairing check, weeks later, on one circuit size.

**Actionable:** do not write multi-limb arithmetic as ordinary C with `unsigned long long`
intermediate carries and expect the compiler to fuse it. Either use inline PTX in the sppark shape,
or use a 64-bit-accumulator-per-32-bit-limb schoolbook form (which is what our MSL does and is
carry-flag-free by construction), and *test it against `ark_ff` on random inputs including the
values `q-1`, `q-2`, `0`, `1`, `R`, `R-1`*.

### 7.2 `__syncthreads()` in divergent code

CUDA C++ Programming Guide §10.6 "Synchronization Functions", fetched this session: **[SOURCE]**

> "`__syncthreads()` **is allowed in conditional code but only if the conditional evaluates
> identically across the entire thread block**, otherwise the code execution is likely to hang or
> produce unintended side effects."

The Pippenger reduction and the NTT butterfly kernel are exactly the places this bites, because both
naturally want an early `return` for out-of-range threads:

```cuda
// WRONG
if (tid >= n) return;
shared[tid] = load(...);
__syncthreads();          // some threads already exited -> UB
```

```cuda
// RIGHT
shared[tid] = (tid < n) ? load(...) : identity();
__syncthreads();          // every thread in the block reaches this
if (tid < n) { store(...); }
```

Our Metal `msm_reduce_impl` and `g16_ntt_batch` use the second shape — the barriers sit at the
uniform loop level, outside the `if (gid < ...)` guards, and the strided inner loops
(`for (bfy = tid; bfy < halves; bfy += tgsz)`) mean every thread reaches every barrier. **[VERIFIED
— read `msm.metal` and `ntt.metal`]** The CUDA port must preserve that structure, not "simplify" it
into an early return.

Note that Metal's `threadgroup_barrier` has the same requirement, so this is not a new class of bug
in CUDA — but CUDA's `__syncthreads()` is far more commonly written after an early return because
the `if (i < n) return;` idiom is so idiomatic in CUDA and rarer in MSL.

### 7.3 Warp divergence, independent thread scheduling, and warp-synchronous code

This is the CUDA hazard that has no Metal analogue and is the one most likely to catch a port.

Programming Guide, Volta / Compute Capability 7.x: **[SOURCE]**

> "[threads] can now diverge and reconverge at sub-warp granularity. **Independent Thread Scheduling
> can lead to a rather different set of threads participating in the executed code than intended if
> the developer made assumptions about warp-synchronicity of previous hardware architectures. In
> particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should
> be revisited** to ensure compatibility with NVIDIA Volta and beyond."

and on `__syncwarp`: **[SOURCE]**

> "Each calling thread must have its own bit set in the mask and **all non-exited threads named in
> mask must execute a corresponding `__syncwarp()` with the same mask, or the result is undefined**."

Concretely for an MSM/NTT port:

- A tree reduction inside a warp written as `if (tid < 16) s[tid] += s[tid+16];` with no
  `__syncwarp()` and a `volatile` array — the classic pre-Volta idiom copied from old reduction
  tutorials — is **undefined behaviour on every GPU we would target**. It gives wrong sums
  non-deterministically. Use `__syncwarp()` or, better, `cooperative_groups::thread_block_tile<32>`
  and `tile.shfl_down(...)`.
- `__shfl_sync(0xffffffff, x, 0)` with a hardcoded full mask is only valid if all 32 lanes are
  active at that instruction. In sppark's `accumulate`, the shuffle is placed at the *bottom* of the
  `while (x < ...)` loop, before the condition is re-tested, precisely so no lane has exited yet.
  **[VERIFIED — read the source]** Copying that loop but moving the shuffle inside the body would
  break it.
- Do not use `__activemask()` to build the mask. It returns whatever happens to be converged, which
  under ITS is a racy, scheduler-dependent value.

The *performance* form of divergence — variable-length bucket runs giving lanes different trip
counts — is real but is a throughput issue only, and our segmented-slice design already addresses it
(fixed-length slices, imbalance measured at 691x for the naive form).

**Shared-memory bank conflicts specifically:** the Programming Guide says: **[SOURCE]**

> "if two addresses of a memory request fall in the same memory bank, there is a bank conflict and
> the access has to be serialized. The hardware splits a memory request with bank conflicts into as
> many separate conflict-free requests as necessary, **decreasing throughput** by a factor equal to
> the number of separate memory requests."

So bank conflicts are **a throughput bug, never a correctness bug** — the hardware serializes and
all accesses complete. The genuine shared-memory correctness hazards are (i) missing/divergent
`__syncthreads()` (§7.2), (ii) warp-synchronous assumptions (above), and (iii) **dynamic shared
memory aliasing**: the Guide notes that with `extern __shared__ float array[]`, "**All variables
declared in this fashion, start at the same address in memory**, so that the layout of the variables
in the array must be explicitly managed through offsets." **[SOURCE]** Declaring both
`extern __shared__ uint4 scratch_[];` and a second `extern __shared__` array of a different type in
the same kernel silently aliases them. sppark's `integrate` reinterprets one declaration
(`auto* scratch = reinterpret_cast<bucket_h*>(scratch_);`) rather than declaring two. **[VERIFIED]**
A related silent-corruption bug: launching with a smaller `sharedMemBytes` third launch argument
than the kernel indexes — CUDA does not bounds-check shared memory, and the overflow lands in
another block's window.

### 7.4 Grid-wide synchronization requires a cooperative launch

sppark's `accumulate` ends with `cooperative_groups::this_grid().sync();`. **[VERIFIED]** The
Programming Guide: **[SOURCE]**

> "launching the kernel it is necessary to use, instead of the `<<<...>>>` execution configuration
> syntax, the `cudaLaunchCooperativeKernel` CUDA runtime launch API... To guarantee co-residency of
> the thread blocks on the GPU, the number of blocks launched needs to be carefully considered."

A grid sync launched with `<<<...>>>`, or with more blocks than fit co-resident on the device, hangs
or produces undefined results. cuZK avoids the issue entirely by using `cudaStreamSynchronize`
between phases (§1.3) — separate kernel launches instead of a grid barrier. **For the CUDA lane,
prefer separate kernel launches.** It costs a few microseconds of launch latency per phase, it
mirrors what our Metal backend already does (one dispatch per phase inside one command buffer), it
keeps the code portable, and it removes an entire class of occupancy-dependent hangs.

### 7.5 Stream ordering and async memset

- `cudaMemset` (non-async) is stream-ordered with respect to the null stream but is *not* a host
  sync; `cudaMemsetAsync(ptr, 0, n, stream)` must go on the *same* stream as the consuming kernel.
  A `cudaMemsetAsync` on stream A followed by an accumulate kernel on stream B is a race.
- Pinned-host / managed memory read on the host without `cudaStreamSynchronize` (or an event) after
  the last kernel returns stale data. This is a much easier mistake in CUDA than in Metal, where
  `waitUntilCompleted` on the command buffer is the only way to get results back.
- **[REASONED]**

---

## 8. NTT-specific GPU correctness hazards

Less written about than MSM, but ZKProphet's headline finding is that NTT is now the bottleneck —
"NTT... account for up to 90% of the proof generation latency on GPUs when paired with optimized MSM
implementations" **[SOURCE]** — so the CUDA lane will spend real effort here.

**8.1 In-place bit-reversal is a race.** The natural kernel is:

```cuda
uint j = brev(i, log_n);
if (i < j) { swap(a[i], a[j]); }     // guard is load-bearing
```

Without the `i < j` guard, both thread `i` and thread `j` perform the swap and the pair ends up
*unswapped* — and since threads are not synchronized, some pairs get swapped once and some twice,
non-deterministically. Even *with* the guard, an in-place permutation over the whole array in one
kernel needs no barrier (each pair is touched by exactly one thread), but any attempt to fuse it
with the first butterfly pass in the same kernel does. Our Metal sidesteps the whole thing by
folding the permutation into the *load* of an out-of-place head kernel: "after the permutation
position i holds src[reverse(i)], so the thread that owns output position i simply reads src at the
reversed index." **[VERIFIED — read `ntt.metal`]** Copy that; it is both faster (saves a full
read+write of the domain per transform, six times per proof) and race-free by construction.
**[REASONED]**

**8.2 A butterfly pass needs a global barrier, not a block barrier.** Pass `t` of a radix-2
decimation-in-time NTT flips bit `t` of the index. Passes whose bits all lie within one shared-memory
tile can be batched behind `__syncthreads()`; passes that cross tiles need every block to have
finished the previous pass, which `__syncthreads()` does not give. Getting this wrong reads
half-updated values and produces a wrong-but-plausible polynomial. Our Metal computes exactly which
passes may share a tile: "the passes in [s0, s0+k) touch only bits [s0, s0+k), and a group is any
set of indices agreeing on every other bit". **[VERIFIED]** The CUDA port must keep that algebra and
put a kernel-launch boundary (or a cooperative grid sync, with §7.4's caveats) at each batch edge.

**8.3 Twiddle table indexing.** The twiddle index in a batched kernel is not the naive
`(m mod 2^t) << (log_n - t - 1)` — it depends on the *low* bits of the tile's global position.
Our `g16_ntt_batch` uses `tw[(low + (jm << s0)) << twshift]` and the comment spells out the
derivation. **[VERIFIED]** An off-by-one in `twshift` produces a transform that is wrong but has the
right magnitude, which no smoke test catches. The only reliable check is a bit-exact comparison
against the CPU NTT on random input at every `log_n` from 1 to `log_n_max`.

**8.4 The coset shift and the `1/n` normalization must not be applied twice.** Our Metal folds both
into the load epilogue (`G16_SCALE_CONST`, `G16_SCALE_TABLE`) rather than running separate passes,
justified by `Fr`-linearity. **[VERIFIED]** If the CUDA port keeps them as separate kernels *and*
also sets the scale mode, the result is scaled by `1/n²`. Test: `intt(ntt(x)) == x` bit-exactly.

**8.5 `Fr` is not `Fq`.** The NTT is over the scalar field `Fr` and the MSM point arithmetic is over
the base field `Fq`. Our MSL keeps two deliberately separate copies of the field code with a
comment explaining why a template was rejected, and `msm.rs` greps the shader source for the
constant lines so the host and device moduli cannot drift. **[VERIFIED]** A CUDA port that templates
over the modulus must have an equivalent host/device constant cross-check, because a swapped modulus
gives *valid-looking* field elements and a proof that simply fails to verify.

---

## 9. Concrete checklist

### For the CUDA backend being written now

1. `cudaMemsetAsync` the bucket array to zero on the consuming stream before every accumulate.
   Do not port the Metal "fresh buffers are zeroed" assumption. **(§4)**
2. Additionally write `.inf()` explicitly on the empty-bucket path, sppark style. **(§4)**
3. Implement `madd`/`add` with all four guards (`acc == identity`, `base == identity`, `P == Q`,
   `P == -Q`) from the first commit. **(§3)**
4. Never read-modify-write a bucket from more than one thread. Atomics on `u32` counters only.
   Accumulate in registers, store once. **(§1)**
5. Handle infinity bases: decide the encoding (flag byte vs `(0,0)` sentinel), match the Metal
   backend, and test with infinity at index 0, mid, and last. 33.9% of `js_16x16_d32`'s B bases
   are infinity. **(§5.1)**
6. Special-case scalar `0` (skip) and scalar `1` (dedicated kernel). Verify the two paths partition
   the index set. **(§5.2, §5.3)**
7. Branchless conditional subtraction via `setp` + `@pred mov`. Verify with `cuobjdump -sass` that
   the reduction contains no branch. **(§6)**
8. Straight-line carry chains, one `asm volatile` per limb op, no control flow inside. **(§7.1)**
9. No `__syncthreads()` after an early `return`. No warp-synchronous reduction without
   `__syncwarp()` or `cooperative_groups`. No `__activemask()`. **(§7.2, §7.3)**
10. Separate kernel launches between MSM phases and between NTT pass batches, rather than a
    cooperative grid sync. **(§7.4, §8.2)**

### For the Metal audit

Everything above is already handled correctly. The residual items are latent rather than live:

- `msm_clear_impl` clears only `zz`; any future reader of `x`/`y`/`zzz` that does not check `zz`
  first reads the previous proof's data. Consider clearing all four, or asserting the invariant.
  **(§4.2)**
- The `G16_METAL_MSM_LEGACY_ACC=1` path skips the clear kernel and is correct only because
  `msm_accumulate_impl` writes every bucket unconditionally. Fragile if that kernel ever gains an
  early return. **(§4.2)**
- Metal's zero-fill guarantee holds for `MTLDevice.makeBuffer(length:options:)` but is *not*
  documented for `MTLHeap.makeBuffer`. If the pool ever moves to a heap, the invariant breaks
  silently. **(§4.2)**

### Tests worth writing (as new files, per lane discipline)

- Random MSM vs `ark_ec::VariableBaseMSM` at several sizes, comparing **normalized** points, run
  many times to catch order-dependent exceptions. **(§2.2)**
- Adversarial base vectors: all-infinity; alternating `P, -P`; a bucket forced to contain
  `{P, P}`; a bucket forced to contain `{P, -P}`; a single base repeated `n` times.
- Adversarial scalars: all zero; all one; all `q-1`; exactly one nonzero at index `n-1`.
- Back-to-back proofs from a pooled backend, asserting the second matches the first (catches
  stale-buffer bugs that a single-shot test cannot see).
- `intt(ntt(x)) == x` bit-exactly at every `log_n`.

---

## 10. Sources

All fetched in this session unless noted.

**Papers**
- Lu, Wang, Yang, Jiang, Ma. *cuZK: Accelerating Zero-Knowledge Proof with A Faster Parallel
  Multi-Scalar Multiplication Algorithm on GPUs.* IACR ePrint 2022/1321.
  <https://eprint.iacr.org/2022/1321.pdf> — §3.2, §4.2 (ELL→CSR→transpose→SpMV), §4.2 stream barriers.
- *ZKProphet: Understanding Performance of Zero-Knowledge Proofs on GPUs.* arXiv:2509.22684v1.
  <https://arxiv.org/pdf/2509.22684v1> — 70.5% branch latency, 52.5%/56.2% branch efficiency,
  72→244 cycle divergence penalty, NTT at up to 90% of latency, integer-pipeline-only claim.
- Explicit-Formulas Database, short Weierstrass XYZZ and Jacobian:
  <https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html>,
  <https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html> (referenced, not re-fetched; the
  formula names `add-2008-s`, `madd-2008-s`, `dbl-2008-s-1`, `add-2007-bl`, `madd-2007-bl`,
  `dbl-2009-l` are cited from the upstream sources below which quote them).

**Vendor documentation**
- NVIDIA CUDA Runtime API, Memory Management (`cudaMalloc`, `cudaMallocManaged`): "The memory is not
  cleared." <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>
- NVIDIA CUDA C++ Programming Guide 12.3 (PDF): §10.6 `__syncthreads()` in conditional code;
  §10.19/§10.x `__syncwarp` mask rules; Volta Independent Thread Scheduling and warp-synchronous
  code; shared memory bank conflicts ("decreasing throughput"); `extern __shared__` aliasing;
  `cudaLaunchCooperativeKernel`.
  <https://docs.nvidia.com/cuda/archive/12.3.0/pdf/CUDA_C_Programming_Guide.pdf>
- NVIDIA PTX ISA 8.3 (PDF) §9.7.2 Extended-Precision Integer Arithmetic: CC.CF "is not preserved
  across calls and is mainly intended for use in straight-line code sequences."
  <https://docs.nvidia.com/cuda/archive/12.3.0/pdf/ptx_isa_8.3.pdf>
- Apple, `MTLDevice.makeBuffer(length:options:)`: "Creates a buffer the method clears with zero
  values." <https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)>
- Apple, `MTLHeap.makeBuffer(length:options:)`: "Creates a buffer on the heap." (no zeroing clause)
  <https://developer.apple.com/documentation/metal/mtlheap/makebuffer(length:options:)>

**Upstream implementations (source read directly)**
- lambdaclass/lambdaworks, `crates/gpu/src/metal/shaders/msm/msm.metal` — the self-admitted bucket
  race, quoted verbatim in §1.2; branchy `mont_reduce`/`field_add`.
- supranational/sppark, `msm/pippenger.cuh` — exclusive bucket ownership, `atomicAdd` on a `u32`
  work counter, explicit `.inf()` on empty buckets, `this_grid().sync()`.
- supranational/sppark, `ec/xyzz_t.hpp` — complete `add` case analysis; `uadd` as the
  divergence-free variant selected by point size; `field_t::one(is_inf)` infinity encoding.
- supranational/sppark, `ff/mont_t.cuh` — branchless `operator-=` via `setp` + `@%top mov.b32`;
  `#define asm asm volatile`.
- zkmopro/gpu-acceleration, `mopro-msm/src/msm/metal_msm/shader/curve/xyzz.metal` — guarded XYZZ
  formulas, "identity is ZZ = 0, which matches zero-initialized buffers".
- zkmopro/gpu-acceleration, `mopro-msm/src/msm/metal_msm/utils/limbs_conversion.rs` —
  `pack_affine_and_scalars` copies `pt.x`/`pt.y` with no `infinity` check.
- zkmopro/mopro, `docs/blog/2025-07-28-metal-msm-v2.md` — cuZK Algorithm 4 / SMVP / pBPR lineage.
- arkworks-rs/algebra, `ec/src/models/short_weierstrass/affine.rs` — `identity()` sets `x = y = 0`,
  "Setting these to zero is *load-bearing*".

**Our own source (read in this session, not modified)**
- `crates/g16-metal/src/shaders/msm.metal` — counting-sort design rationale, XYZZ guards,
  `fq_cond_sub_n`, `msm_clear_impl`, `msm_segmented_impl`, `msm_merge_impl`.
- `crates/g16-metal/src/shaders/ntt.metal` — fused bit-reversal, batched passes, twiddle index algebra.
- `crates/g16-metal/src/layout.rs` — `(0,0)` infinity sentinel and its justification.
- `crates/g16-metal/src/msm.rs` — buffer `Pool`, clear dispatch, `0`/`1` scalar filtering.

**Measurements I ran**
- Section walk over `bench/artifacts/*/circuit.zkey` counting all-zero point blobs per section
  (§3.2 table).
- Duplicate-point and shared-x analysis of `js_2x2_d32` section 5, confirming 64 exact `±` pairs
  with `y_i + y_j ≡ 0 (mod q)`.
