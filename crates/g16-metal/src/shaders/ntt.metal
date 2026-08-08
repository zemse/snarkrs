// Stages 1-3: the six NTTs, the bit-reverse permutation, the iNTT normalisation and the
// coset shift, all folded into two kernels.
//
// Depends on bn254_fr.metal and pointwise.metal being concatenated ahead of this file.
//
// ============================================================================
// THE ALGORITHM IS THE CPU BACKEND'S, EXACTLY
// ============================================================================
//
// g16_ntt::CpuNtt is decimation in time: bit-reverse the input, then log(n) passes with
// half = 1, 2, 4, ..., n/2, where butterfly j of a block reads twiddles[j * n/(2*half)]
// from a natural-order table of the root's powers. Everything below computes that same
// sequence of butterflies in that same order with that same twiddle table, so the two
// backends agree bit for bit and not merely up to a permutation.
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
//     Domain does not build this table, so the host builds it once at prepare time.
//
// ============================================================================
// BATCHING: WHY TWO DISPATCHES PER TRANSFORM AND NOT log(n)
// ============================================================================
//
// One dispatch per pass means log(n) round trips through device memory. At 2^18 that is
// 18 passes x 16 MB of read+write traffic, and the measured streaming rate for this
// field at that size is 30 GB/s against a bus that does 300, i.e. the kernel is memory
// bound with the ALU idle. So passes are batched: a threadgroup pulls a slice into
// threadgroup memory, runs several passes there with only a barrier between them, and
// writes back once.
//
// Which passes can share a slice is fixed by the index algebra. Pass t (0-based) flips
// bit t of an element index. So the passes in [s0, s0+k) touch only bits [s0, s0+k), and
// a group is any set of indices agreeing on every other bit: fix `low` = bits [0, s0) and
// `high` = bits [s0+k, log n), vary the k middle bits. That is a contiguous run of 2^k
// elements when s0 = 0, and a run strided by 2^s0 otherwise. This is the standard six-step
// decomposition, written as one kernel that takes (s0, k) rather than two kernels.
//
// The slice size is capped by threadgroup memory: 32768 bytes on this device and 32 bytes
// per Fr, so 1024 elements, so k <= 10. The host splits log(n) into ceil(log n / 10)
// batches of near-equal size rather than greedily filling the first: an 18 = 10 + 8 split
// wastes half the second kernel's parallelism and, at other sizes, produces the degenerate
// two-element tail kernel that profiling of bellperson caught at 2^26. 18 = 9 + 9 instead,
// which is 512 elements and 16 KB per group, leaving room for two resident groups per core.
//
// At the 2^18 domain of the largest artifact that is TWO dispatches per transform, twelve
// for the six transforms of a proof, and the whole of stages 0 to 4 is fourteen dispatches
// inside one command buffer with one wait. The measured cost of an extra dispatch in an
// already open command buffer is a couple of microseconds; the cost of a separate command
// buffer is 0.15 ms, which is why this is not encoded as fourteen command buffers.
//
// ============================================================================
// WHAT IS *NOT* DONE HERE
// ============================================================================
//
// No twiddle caching in threadgroup memory. The slice is already 16 KB of the 32 KB
// budget and the twiddles a batch touches are 2^(k-1) distinct values, another 16 KB,
// which would leave nothing and halve occupancy. Left on the table deliberately.

#ifndef G16_NTT_METAL
#define G16_NTT_METAL

// Load-time scaling. The branch is on a value that is uniform across the whole dispatch,
// so it costs a scalar compare and never diverges a SIMD group.
constant uint G16_SCALE_NONE  = 0u; // forward transform of an already-shifted vector
constant uint G16_SCALE_CONST = 1u; // iNTT: multiply by size_inv
constant uint G16_SCALE_TABLE = 2u; // stage 2: multiply by shift^j, j the natural index

// Store-time epilogue, also dispatch-uniform.
constant uint G16_STORE_PLAIN = 0u;
constant uint G16_STORE_JOIN  = 1u; // stage 4 fused in: write H = A*B - this, not this

// Mirrors g16_metal::stages::NttParams. 52 bytes, all 4-byte aligned, no padding.
struct NttParams {
    uint log_n;      // log2 of the domain size
    uint s0;         // index of the first pass in this batch
    uint k;          // number of passes in this batch; slice is 2^k elements
    uint scale_mode;
    uint store_mode;
    Fr   kscale;     // the G16_SCALE_CONST multiplier, Montgomery form
};

static_assert(sizeof(NttParams) == 52, "NttParams must match stages::NttParams");

// The k passes of one batch, over a slice already resident in threadgroup memory.
//
// `low` is bits [0, s0) of every index in the slice, which is what makes the twiddle
// index depend on more than the local position: at global pass s0 + t the butterfly's
// index within its block is low + (m mod 2^t) * 2^s0, where m is the local index, and the
// twiddle is that times 2^(log_n - s0 - t - 1). For the s0 = 0 head this collapses to the
// familiar (m mod 2^t) << (log_n - t - 1).
inline void g16_ntt_batch(threadgroup Fr* sh,
                          device const Fr* tw,
                          uint log_n, uint s0, uint k, uint low,
                          uint tid, uint tgsz)
{
    uint halves = 1u << (k - 1u); // butterflies per pass; k >= 1 whenever this is called
    for (uint t = 0u; t < k; t++) {
        uint hl = 1u << t;
        uint twshift = log_n - s0 - t - 1u;
        for (uint bfy = tid; bfy < halves; bfy += tgsz) {
            uint jm = bfy & (hl - 1u);
            uint lo = ((bfy >> t) << (t + 1u)) | jm;
            uint hi = lo + hl;
            Fr u = sh[lo];
            Fr v = fr_mul(sh[hi], tw[(low + (jm << s0)) << twshift]);
            sh[lo] = fr_add(u, v);
            sh[hi] = fr_sub(u, v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Batch 0. Out of place, src -> dst, and it absorbs the bit-reverse permutation into the
// load: after the permutation position i holds src[reverse(i)], so the thread that owns
// output position i simply reads src at the reversed index. The CPU backend does the same
// permutation as an in-place pass of swaps; skipping it as a dispatch saves a full read
// and write of the domain per transform, six times per proof.
kernel void g16_ntt_head(
    device const Fr*     src    [[buffer(0)]],
    device Fr*           dst    [[buffer(1)]],
    device const Fr*     tw     [[buffer(2)]],
    device const Fr*     ptab   [[buffer(3)]], // shift^j table, G16_SCALE_TABLE only
    device const Fr*     join_a [[buffer(4)]], // G16_STORE_JOIN only
    device const Fr*     join_b [[buffer(5)]],
    device Fr*           h_mont [[buffer(6)]],
    device Fr*           h_std  [[buffer(7)]],
    constant NttParams&  p      [[buffer(8)]],
    threadgroup Fr*      sh     [[threadgroup(0)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    uint blk = 1u << p.k;
    uint base = tg << p.k;
    // A domain of size 1 has log_n = 0, and `>> 32` is undefined, so the degenerate case
    // is a uniform branch rather than a shift the compiler is free to fold to anything.
    uint revshift = 32u - p.log_n;

    for (uint m = tid; m < blk; m += tgsz) {
        uint i = base + m;
        uint s = (p.log_n == 0u) ? 0u : (reverse_bits(i) >> revshift);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.k > 0u) {
        g16_ntt_batch(sh, tw, p.log_n, 0u, p.k, 0u, tid, tgsz);
    }

    for (uint m = tid; m < blk; m += tgsz) {
        uint i = base + m;
        if (p.store_mode == G16_STORE_JOIN) {
            g16_store_h(h_mont, h_std, i, fr_sub(fr_mul(join_a[i], join_b[i]), sh[m]));
        } else {
            dst[i] = sh[m];
        }
    }
}

// Every batch after the first. In place on `a`, slice strided by 2^s0.
//
// Threadgroup g owns the slice with low = g mod 2^s0 and high = g >> s0, i.e. base index
// low + high * 2^(s0+k) and elements base + m * 2^s0 for m in [0, 2^k). There are
// 2^(log_n - k) groups and the two halves of g cover exactly the log_n - k bits outside
// the batch, so the groups partition the domain with no overlap and no gap.
kernel void g16_ntt_tail(
    device Fr*           a      [[buffer(0)]],
    device const Fr*     tw     [[buffer(2)]],
    device const Fr*     join_a [[buffer(4)]],
    device const Fr*     join_b [[buffer(5)]],
    device Fr*           h_mont [[buffer(6)]],
    device Fr*           h_std  [[buffer(7)]],
    constant NttParams&  p      [[buffer(8)]],
    threadgroup Fr*      sh     [[threadgroup(0)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    uint blk = 1u << p.k;
    uint low = tg & ((1u << p.s0) - 1u);
    uint high = tg >> p.s0;
    uint base = low + (high << (p.s0 + p.k));

    for (uint m = tid; m < blk; m += tgsz) {
        sh[m] = a[base + (m << p.s0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    g16_ntt_batch(sh, tw, p.log_n, p.s0, p.k, low, tid, tgsz);

    for (uint m = tid; m < blk; m += tgsz) {
        uint i = base + (m << p.s0);
        if (p.store_mode == G16_STORE_JOIN) {
            g16_store_h(h_mont, h_std, i, fr_sub(fr_mul(join_a[i], join_b[i]), sh[m]));
        } else {
            a[i] = sh[m];
        }
    }
}

#endif // G16_NTT_METAL
