// Stage 0: the CSR coefficient gather, plus the C matrix that snarkjs never stores.
//
// Twin of crates/g16-metal/src/shaders/gather.metal. Depends on bn254_fr.cuh being
// concatenated ahead of this file by kernels::unit_stages; it carries no #include of its
// own because NVRTC has no filesystem.
//
// ============================================================================
// WHY ONE THREAD PER ROW
// ============================================================================
//
// The rows are short and ragged, and they get more so as the circuit grows. Measured on
// the real proving keys, mean row length / empty rows / longest row:
//
//   js_1x1_d8    domain 2^12   A 2.31 / 732 / 65      B 3.70 / 737 / 64
//   js_2x2_d16   domain 2^14   A 1.51 / 6224 / 67     B 2.34 / 6231 / 66
//   js_8x8_d32   domain 2^17   A 1.13 / 60696 / 79    B 1.70 / 60715 / 78
//   js_16x16_d32 domain 2^18   A 1.13 / 121848 / 95   B 1.69 / 121883 / 94
//
// So the mean row is one or two terms, nearly half the rows are empty (the domain rounds
// up well past the constraint count), and the longest row in the whole key is under a
// hundred. That rules out every cooperative mapping:
//
//   * One block per row would put 2.31 multiplies on a 32-lane warp and run at
//     single-digit occupancy, and it would need a __shared__ reduction to combine partial
//     sums that are almost always a single term.
//   * One warp per row is the same argument with a smaller constant: 31 of 32 lanes idle
//     on the average row.
//   * A flat "one thread per nonzero" mapping is a scatter, which needs either atomics
//     (there is no 32-byte atomic) or a segmented reduction over the whole nonzero
//     stream. The CSR sort in g16-zkey exists precisely so we do not have to do that.
//
// One thread per row instead puts 32 consecutive rows in one warp, so a warp retires
// roughly 74 (A) to 118 (B) Montgomery multiplies with full lane utilisation. Divergence
// is bounded by the longest row in the warp rather than the longest row in the matrix,
// and neighbouring constraint rows in a circom circuit have similar lengths, so the
// ragged tail is short. An empty row costs one comparison.
//
// The argument carries over from Metal unchanged because a Metal SIMD group and a CUDA
// warp are both 32 lanes with the same "the group waits for its slowest lane" cost model.
//
// Nothing here writes a location another thread reads, so there is no barrier, no atomic
// and no ordering requirement. That is the whole payoff of the CSR sort.
//
// ============================================================================
// WHY C IS COMPUTED HERE AND NOT GATHERED
// ============================================================================
//
// There is no C matrix in a snarkjs zkey. snarkjs' buildABC1 fills C by multiplying the
// A and B evaluations pointwise, which is exact rather than an approximation: the R1CS
// constraint *is* a*b = c, so on the constraint domain the C evaluation of row i is the
// product of the A and B evaluations of row i. Doing it in this kernel means the product
// reuses the two values already in registers, so C costs one Montgomery multiply per row
// and zero extra loads. A separate pass would cost a full read of A and B (2n * 32 bytes)
// on a stage that is already memory bound, and on a discrete card that traffic is real
// GDDR bandwidth rather than a read out of Apple's unified pool.

#ifndef G16_GATHER_CU
#define G16_GATHER_CU

// out_a[c] = sum over CSR row c of value_a[k] * witness[signal_a[k]]
// out_b[c] = the same for matrix B
// out_c[c] = out_a[c] * out_b[c]
//
// `witness` is Montgomery-form Fr (layout::PackedFr), as are the CSR values, so the
// products need no conversion in either direction.
//
// LAUNCH: grid_dim.x = ceil(n / block_dim.x), block_dim.x free (256, the Metal twin's
// preference, is the right starting point). One thread owns exactly one output row and
// touches no other thread's data, so a block is just 256 independent rows.
//
// No __restrict__ on the read-only pointers. It would unlock the non-coherent cache path,
// but nothing in this signature tells the compiler that `witness` and the CSR arrays come
// from disjoint allocations, and a __restrict__ that is ever a lie is a wrong proof with
// no diagnostic. The gather is bound by the dependent load witness[signal[k]] in any
// case, which no cache hint fixes.
extern "C" __global__ void g16_gather_abc(
    const u32* row_ptr_a,
    const u32* signal_a,
    const Fr*  value_a,
    const u32* row_ptr_b,
    const u32* signal_b,
    const Fr*  value_b,
    const Fr*  witness,
    Fr*        out_a,
    Fr*        out_b,
    Fr*        out_c,
    const u32  n)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    // In Metal this guard was belt and braces, because dispatch_threads sizes the final
    // threadgroup exactly. Here it is load bearing. A CUDA grid is always a whole number
    // of blocks, so for any n that is not a multiple of block_dim.x the last block runs
    // threads past the end, and without this guard they would read row_ptr[n + 1] and
    // write past out_a.
    if (gid >= n) {
        return;
    }

    Fr a = fr_zero();
    u32 lo = row_ptr_a[gid];
    u32 hi = row_ptr_a[gid + 1u];
    for (u32 k = lo; k < hi; k++) {
        a = fr_add(a, fr_mul(value_a[k], witness[signal_a[k]]));
    }

    Fr b = fr_zero();
    lo = row_ptr_b[gid];
    hi = row_ptr_b[gid + 1u];
    for (u32 k = lo; k < hi; k++) {
        b = fr_add(b, fr_mul(value_b[k], witness[signal_b[k]]));
    }

    out_a[gid] = a;
    out_b[gid] = b;
    out_c[gid] = fr_mul(a, b);
}

#endif // G16_GATHER_CU
