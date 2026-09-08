// Stages 1-3: the six NTTs, the bit-reverse permutation, the iNTT normalisation and the
// coset shift, all folded into two kernels.
//
// Twin of crates/g16-metal/src/shaders/ntt.metal. Depends on bn254_fr.cuh being
// concatenated ahead of this file. No #include: NVRTC has no filesystem.
//
// ============================================================================
// THE ALGORITHM IS THE CPU BACKEND'S, EXACTLY
// ============================================================================
//
// g16_ntt::CpuNtt is decimation in time: bit-reverse the input, then log(n) passes with
// half = 1, 2, 4, ..., n/2, where butterfly j of a block reads twiddles[j * n/(2*half)]
// from a natural-order table of the root's powers. Everything below computes that same
// sequence of butterflies in that same order with that same twiddle table, so the CPU,
// the Metal and the CUDA backends agree bit for bit and not merely up to a permutation.
//
// Two of the CPU backend's separate steps are folded into the load epilogue, which is
// exact rather than approximate:
//
//   * The iNTT's 1/n normalisation. The CPU multiplies every output by size_inv after the
//     transform; we multiply every input by it before. The transform is Fr-linear, so
//     these are the same map, and doing it on load costs nothing because the element is
//     already in a register.
//   * The stage 2 coset shift, x[j] *= shift^j. Applied to the input of the forward
//     transform, again on load. The powers are NOT the twiddles: the twiddles are powers
//     of the domain's own root of unity, the shift is a primitive 2n-th root (see
//     CpuCircuit::new for why that specific element and not Fr::GENERATOR). g16_field's
//     Domain does not build this table and does not cache the twiddles either, so the
//     host builds all three once at prepare time and leaves them resident.
//
// ============================================================================
// BATCHING: WHY TWO LAUNCHES PER TRANSFORM AND NOT log(n)
// ============================================================================
//
// One launch per pass means log(n) round trips through device memory. At 2^18 that is
// 18 passes x 16 MB of read+write traffic. On Apple silicon the measured streaming rate
// for this field at that size was 30 GB/s against a bus that does 300, i.e. memory bound
// with the ALU idle; on a discrete card the same shape holds and the DRAM round trip is
// worse, not better. So passes are batched: a block pulls a slice into __shared__, runs
// several passes there with only a __syncthreads() between them, and writes back once.
//
// Which passes can share a slice is fixed by the index algebra. Pass t (0-based) flips
// bit t of an element index. So the passes in [s0, s0+k) touch only bits [s0, s0+k), and
// a group is any set of indices agreeing on every other bit: fix `low` = bits [0, s0) and
// `high` = bits [s0+k, log n), vary the k middle bits. That is a contiguous run of 2^k
// elements when s0 = 0, and a run strided by 2^s0 otherwise. This is the standard six-step
// decomposition, written as one kernel that takes (s0, k) rather than two kernels.
//
// ----------------------------------------------------------------------------
// THE SHARED MEMORY BUDGET, WORKED OUT FOR CUDA RATHER THAN COPIED FROM METAL
// ----------------------------------------------------------------------------
//
// The slice size is capped by __shared__ memory per block, which is NOT the same number
// as Metal's maxThreadgroupMemoryLength and must not be assumed to be:
//
//   * Every architecture from Volta up gives a block 48 KB without an opt in. Going above
//     that needs cudaFuncSetAttribute(MaxDynamicSharedMemorySize), which raises the
//     ceiling to 64 KB on Turing (sm_75), 99 KB on the consumer Ampere and Ada parts
//     (sm_86, sm_89) and 163 KB on A100 (sm_80). That opt in is deliberately not taken:
//     it buys one extra pass at most (see the arithmetic below) and costs a per-function
//     driver call plus an architecture-dependent limit the host would have to query and
//     get right on every card.
//   * An Fr is 32 bytes, so 48 KB holds 1536 of them. A slice must be a power of two
//     because it is 2^k elements, and the largest power of two at or below 1536 is
//     1024 = 2^10. So k <= 10.
//   * Raising the ceiling to 64 KB would give 2048 elements and k <= 11, one extra pass.
//     At 2^18 that turns an 18 = 9 + 9 split into 18 = 9 + 9 again (two batches either
//     way), so it changes nothing at the sizes this prover actually runs.
//
// k <= 10 is the same cap the M2 Max reaches from its 32768-byte threadgroup limit
// (32768 / 32 = 1024 exactly), by arithmetic coincidence rather than by copying. That is
// convenient: both backends run the identical pass split at every artifact size, so a
// Metal-versus-CUDA timing compares two implementations of one schedule.
//
// The host splits log(n) into ceil(log n / 10) batches of near-equal size rather than
// greedily filling the first: an 18 = 10 + 8 split wastes half the second kernel's
// parallelism and, at other sizes, produces the degenerate two-element tail kernel that
// profiling of bellperson caught at 2^26. 18 = 9 + 9 instead, which is 512 elements and
// 16 KB per block. That also matters for occupancy here in a way it did not on Metal: a
// full 1024-element slice is 32 KB, and a Turing SM has 64 KB of shared memory in total,
// so a maximal batch would cap residency at two blocks per SM. 16 KB allows four, as far
// as shared memory is concerned; registers may still bind first, which is a reason to
// measure rather than to assume.
//
// At the 2^18 domain of the largest artifact that is TWO launches per transform, twelve
// for the six transforms of a proof, and the whole of stages 0 to 4 is fourteen launches
// on one stream with one synchronize at the end. A CUDA kernel launch on an already-warm
// stream is a few microseconds of host time and is asynchronous, so the fourteen queue up
// without the host blocking; the Metal backend batches into one command buffer for the
// same reason and reaches the same place.
//
// ============================================================================
// WHAT IS *NOT* DONE HERE
// ============================================================================
//
// No twiddle caching in __shared__. The slice is already 16 KB of the 48 KB budget and
// the twiddles a batch touches are 2^(k-1) distinct values, another 8 KB at k = 9, which
// would push a block to 24 KB and halve how many fit on an SM. The twiddle reads hit L2
// hard (every block in a launch reads the same table) so the cache already does most of
// this. Left on the table deliberately.

#ifndef G16_NTT_CU
#define G16_NTT_CU

// Defined in pointwise.cu, which g16_gpu_kernels::unit_stages concatenates AFTER this file.
//
// Note the ordering differs from the Metal backend on purpose. There, pointwise.metal is
// placed ahead of ntt.metal so the definition is in scope. Here the unit order is fixed
// as gather / ntt / pointwise by g16_gpu_kernels::unit_stages, so the fused store
// epilogue below reaches its helper through this forward declaration instead. Same
// translation unit, so __forceinline__ still applies; only the textual order changed.
__device__ __forceinline__ void g16_store_h(Fr* h_mont, Fr* h_std, u32 i, Fr v);

// Load-time scaling and the store-time epilogue. Both are uniform across the whole
// launch, so each costs a scalar compare and neither ever diverges a warp.
//
// Enumerators rather than the `__constant__` the general MSL translation rule would give
// for `constant uint`, for two reasons. That rule is about tables the kernel reads at run
// time, like FR_N in bn254_fr.cuh; these are compile-time tags and want to fold into
// immediate operands, not into a load from constant memory. And two of the five (NONE and
// PLAIN) are never compared against, since the branches test only for the non-default
// modes, so as variables they draw nvcc warning #177 "declared but never referenced".
// Deleting them would leave the mode space half documented, and the host-side Rust mirror
// does use both values. An enumerator is never an unused variable.
enum G16ScaleMode : u32 {
    G16_SCALE_NONE  = 0u, // forward transform of an already-shifted vector
    G16_SCALE_CONST = 1u, // iNTT: multiply by size_inv
    G16_SCALE_TABLE = 2u, // stage 2: multiply by shift^j, j the natural index
};

enum G16StoreMode : u32 {
    G16_STORE_PLAIN = 0u,
    G16_STORE_JOIN  = 1u, // stage 4 fused in: write H = A*B - this, not this
};

// Mirrors g16_cuda::stages::NttParams. 52 bytes, all 4-byte aligned, no padding, which is
// what makes it safe to hand across as a by-value kernel parameter: CUDA copies the
// struct into the parameter space verbatim and the Rust side is #[repr(C)] with the same
// five u32 followed by a PackedFr.
//
// Passed BY VALUE, not by pointer, everywhere. Mixing the two is the kind of mismatch
// that reads a modulus out of the wrong offset and produces a wrong proof rather than a
// fault, so it is stated here as a contract and not left to each call site.
struct NttParams {
    u32 log_n;      // log2 of the domain size
    u32 s0;         // index of the first pass in this batch
    u32 k;          // number of passes in this batch; slice is 2^k elements
    u32 scale_mode;
    u32 store_mode;
    Fr  kscale;     // the G16_SCALE_CONST multiplier, Montgomery form
};

static_assert(sizeof(NttParams) == 52, "NttParams must match stages::NttParams");

// Reverse all 32 bits of x. MSL has `reverse_bits` as a language builtin; CUDA has the
// __brev intrinsic, which lowers to the single PTX instruction brev.b32.
//
// Written as a named wrapper rather than inlined at the call site because it is the one
// place where a builtin is assumed to exist under NVRTC (which compiles without any CUDA
// header). If a future NVRTC drops it, the fix is the six-line shift-and-mask reversal
// here and nothing else changes.
__device__ __forceinline__ u32 g16_brev(u32 x) {
    return __brev(x);
}

// The k passes of one batch, over a slice already resident in __shared__.
//
// `low` is bits [0, s0) of every index in the slice, which is what makes the twiddle
// index depend on more than the local position: at global pass s0 + t the butterfly's
// index within its block is low + (m mod 2^t) * 2^s0, where m is the local index, and the
// twiddle is that times 2^(log_n - s0 - t - 1). For the s0 = 0 head this collapses to the
// familiar (m mod 2^t) << (log_n - t - 1).
//
// `sh` is a pointer into __shared__ and `tw` a pointer into global memory. CUDA infers
// both address spaces from the pointer's provenance, which is why the MSL `threadgroup` /
// `device` qualifiers simply disappear in the port rather than turning into anything.
__device__ __forceinline__ void g16_ntt_batch(Fr* sh,
                                              const Fr* tw,
                                              u32 log_n, u32 s0, u32 k, u32 low,
                                              u32 tid, u32 tgsz)
{
    u32 halves = 1u << (k - 1u); // butterflies per pass; k >= 1 whenever this is called
    for (u32 t = 0u; t < k; t++) {
        u32 hl = 1u << t;
        u32 twshift = log_n - s0 - t - 1u;
        for (u32 bfy = tid; bfy < halves; bfy += tgsz) {
            u32 jm = bfy & (hl - 1u);
            u32 lo = ((bfy >> t) << (t + 1u)) | jm;
            u32 hi = lo + hl;
            Fr u = sh[lo];
            Fr v = fr_mul(sh[hi], tw[(low + (jm << s0)) << twshift]);
            sh[lo] = fr_add(u, v);
            sh[hi] = fr_sub(u, v);
        }
        // Outside the strided loop, so every thread in the block reaches it the same
        // number of times even when halves < blockDim.x and some threads do no work at
        // all. A __syncthreads() that only some threads reach is undefined behaviour on
        // CUDA, not merely slow.
        __syncthreads();
    }
}

// Batch 0. Out of place, src -> dst, and it absorbs the bit-reverse permutation into the
// load: after the permutation position i holds src[reverse(i)], so the thread that owns
// output position i simply reads src at the reversed index. The CPU backend does the same
// permutation as an in-place pass of swaps; skipping it as a launch saves a full read
// and write of the domain per transform, six times per proof.
//
// LAUNCH: grid_dim.x = max(1, n >> k) blocks, block_dim.x free (256), and
// shared_mem_bytes = (1 << k) * 32, the slice and nothing else. One BLOCK owns the
// contiguous run of 2^k elements starting at blockIdx.x << k and runs all k passes over
// it; one THREAD strides over that run by blockDim.x, so it handles 2^k / blockDim.x
// elements on load and store and the same count of butterflies per pass. blockDim.x need
// not divide 2^k and need not be at most 2^k; both loops are `for (m = tid; m < blk;
// m += tgsz)` precisely so any block width is legal.
extern "C" __global__ void g16_ntt_head(
    const Fr*       src,
    Fr*             dst,
    const Fr*       tw,
    const Fr*       ptab,   // shift^j table, G16_SCALE_TABLE only
    const Fr*       join_a, // G16_STORE_JOIN only
    const Fr*       join_b,
    Fr*             h_mont,
    Fr*             h_std,
    const NttParams p)
{
    // Dynamic __shared__, sized by the launch's shared_mem_bytes. It has to be dynamic
    // rather than a fixed `__shared__ Fr sh[1024]`: k varies with the domain and with the
    // pass split, and a static 32 KB array would charge every launch the maximum and cut
    // occupancy to one block per SM even for a 2^12 domain.
    extern __shared__ Fr sh[];

    u32 tid = threadIdx.x;
    u32 tgsz = blockDim.x;
    u32 blk = 1u << p.k;
    u32 base = blockIdx.x << p.k;
    // A domain of size 1 has log_n = 0, and `>> 32` is undefined, so the degenerate case
    // is a uniform branch rather than a shift the compiler is free to fold to anything.
    u32 revshift = 32u - p.log_n;

    for (u32 m = tid; m < blk; m += tgsz) {
        u32 i = base + m;
        u32 s = (p.log_n == 0u) ? 0u : (g16_brev(i) >> revshift);
        Fr x = src[s];
        if (p.scale_mode == G16_SCALE_CONST) {
            x = fr_mul(x, p.kscale);
        } else if (p.scale_mode == G16_SCALE_TABLE) {
            // Indexed by the *source* index, which is the natural-order position the
            // shift power belongs to. Indexing by `i` here would apply shift^bitrev(j),
            // which is a different and wrong vector.
            x = fr_mul(x, ptab[s]);
        }
        sh[m] = x;
    }
    __syncthreads();

    if (p.k > 0u) {
        g16_ntt_batch(sh, tw, p.log_n, 0u, p.k, 0u, tid, tgsz);
    }

    for (u32 m = tid; m < blk; m += tgsz) {
        u32 i = base + m;
        if (p.store_mode == G16_STORE_JOIN) {
            g16_store_h(h_mont, h_std, i, fr_sub(fr_mul(join_a[i], join_b[i]), sh[m]));
        } else {
            dst[i] = sh[m];
        }
    }
}

// Every batch after the first. In place on `a`, slice strided by 2^s0.
//
// Block g owns the slice with low = g mod 2^s0 and high = g >> s0, i.e. base index
// low + high * 2^(s0+k) and elements base + m * 2^s0 for m in [0, 2^k). There are
// 2^(log_n - k) blocks and the two halves of g cover exactly the log_n - k bits outside
// the batch, so the blocks partition the domain with no overlap and no gap.
//
// LAUNCH: identical geometry to the head, grid_dim.x = max(1, n >> k) blocks,
// shared_mem_bytes = (1 << k) * 32. This kernel is never launched with k = 0, because a
// zero-width batch only ever appears as the single batch of a size-1 domain and that one
// is served by the head.
extern "C" __global__ void g16_ntt_tail(
    Fr*             a,
    const Fr*       tw,
    const Fr*       join_a,
    const Fr*       join_b,
    Fr*             h_mont,
    Fr*             h_std,
    const NttParams p)
{
    extern __shared__ Fr sh[];

    u32 tid = threadIdx.x;
    u32 tgsz = blockDim.x;
    u32 blk = 1u << p.k;
    u32 low = blockIdx.x & ((1u << p.s0) - 1u);
    u32 high = blockIdx.x >> p.s0;
    u32 base = low + (high << (p.s0 + p.k));

    for (u32 m = tid; m < blk; m += tgsz) {
        sh[m] = a[base + (m << p.s0)];
    }
    __syncthreads();

    g16_ntt_batch(sh, tw, p.log_n, p.s0, p.k, low, tid, tgsz);

    for (u32 m = tid; m < blk; m += tgsz) {
        u32 i = base + (m << p.s0);
        if (p.store_mode == G16_STORE_JOIN) {
            g16_store_h(h_mont, h_std, i, fr_sub(fr_mul(join_a[i], join_b[i]), sh[m]));
        } else {
            a[i] = sh[m];
        }
    }
}

#endif // G16_NTT_CU
