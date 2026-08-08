// Stage 0: the CSR coefficient gather, plus the C matrix that snarkjs never stores.
//
// Depends on bn254_fr.metal being concatenated ahead of this file.
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
//   * One threadgroup per row would put 2.31 multiplies on a 32-wide SIMD group and run
//     at single-digit occupancy, and it would need a threadgroup reduction to combine
//     partial sums that are almost always a single term.
//   * One SIMD group per row is the same argument with a smaller constant: 31 of 32
//     lanes idle on the average row.
//   * A flat "one thread per nonzero" mapping is a scatter, which needs either atomics
//     (there is no 32-byte atomic) or a segmented reduction over the whole nonzero
//     stream. The CSR sort in g16-zkey exists precisely so we do not have to do that.
//
// One thread per row instead puts 32 consecutive rows in one SIMD group, so a group
// retires roughly 74 (A) to 118 (B) Montgomery multiplies with full lane utilisation.
// Divergence is bounded by the longest row in the group rather than the longest row in
// the matrix, and neighbouring constraint rows in a circom circuit have similar lengths,
// so the ragged tail is short. An empty row costs one comparison.
//
// Nothing here writes a location another thread reads, so there is no barrier, no
// atomic and no ordering requirement. That is the whole payoff of the CSR sort.
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
// on a stage that is already memory-bound.

#ifndef G16_GATHER_METAL
#define G16_GATHER_METAL

// out_a[c] = sum over CSR row c of value_a[k] * witness[signal_a[k]]
// out_b[c] = the same for matrix B
// out_c[c] = out_a[c] * out_b[c]
//
// `witness` is Montgomery-form Fr (layout::PackedFr), as are the CSR values, so the
// products need no conversion in either direction.
kernel void g16_gather_abc(
    device const uint* row_ptr_a [[buffer(0)]],
    device const uint* signal_a  [[buffer(1)]],
    device const Fr*   value_a   [[buffer(2)]],
    device const uint* row_ptr_b [[buffer(3)]],
    device const uint* signal_b  [[buffer(4)]],
    device const Fr*   value_b   [[buffer(5)]],
    device const Fr*   witness   [[buffer(6)]],
    device Fr*         out_a     [[buffer(7)]],
    device Fr*         out_b     [[buffer(8)]],
    device Fr*         out_c     [[buffer(9)]],
    constant uint&     n         [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    // dispatch_threads gives a non-uniform final threadgroup, so this is belt and braces
    // rather than load bearing. It costs one uniform comparison.
    if (gid >= n) {
        return;
    }

    Fr a = fr_zero();
    uint lo = row_ptr_a[gid];
    uint hi = row_ptr_a[gid + 1u];
    for (uint k = lo; k < hi; k++) {
        a = fr_add(a, fr_mul(value_a[k], witness[signal_a[k]]));
    }

    Fr b = fr_zero();
    lo = row_ptr_b[gid];
    hi = row_ptr_b[gid + 1u];
    for (uint k = lo; k < hi; k++) {
        b = fr_add(b, fr_mul(value_b[k], witness[signal_b[k]]));
    }

    out_a[gid] = a;
    out_b[gid] = b;
    out_c[gid] = fr_mul(a, b);
}

#endif // G16_GATHER_METAL
