// Stage 4: H = A*B - C, elementwise, and the two encodings it is written in.
//
// Depends on bn254_fr.metal being concatenated ahead of this file, and must itself be
// concatenated ahead of ntt.metal, which calls `g16_store_h` from its fused epilogue.
//
// ============================================================================
// NO DIVISION BY Z(coset). THIS IS NOT AN OMISSION.
// ============================================================================
//
// snarkjs' joinABC computes exactly a*b - c and feeds it straight to the H multiexp. The
// division by the vanishing polynomial is folded into the section 9 bases at setup: they
// are the odd Lagrange polynomials of the 2n domain, and P = A*B - C vanishes on the even
// points (those are the constraint rows), so sum_i P(inc^(2i+1)) * hExps[i] already
// equals [P(tau)]_1 = [H(tau) * Z(tau)]_1. Dividing here as well double-counts Z and
// produces a proof that fails verification with nothing else to go on. The CPU backend
// (g16_core::cpu::CpuCircuit::compute_h) makes the same choice and the same argument.
//
// ============================================================================
// TWO OUTPUT BUFFERS, AND WHY BOTH
// ============================================================================
//
// H leaves this stage in two encodings written in the same pass:
//
//   h_mont  Montgomery limbs (layout::PackedFr). This is what any further field
//           arithmetic wants, and it is what a host readback compares against the CPU
//           backend, since PackedFr::to_fr is the identity on arkworks' internal form.
//   h_std   standard form, the integer in [0, r) (layout::PackedScalar). This is what
//           stage 9's Pippenger MSM wants, because a window digit of a Montgomery
//           representative is a digit of a*R mod r, which is a different number. Getting
//           that backwards yields a proof wrong by a factor of R.
//
// The alternative is one buffer plus a conversion dispatch later, which costs an extra
// full read and write of the domain (2n * 32 bytes) plus a command-buffer slot, against
// n * 32 bytes of extra writes here. On a stage that is already memory bound the second
// write is the cheaper of the two, and it removes a coordination point between this
// stage and the MSM stage entirely.

#ifndef G16_POINTWISE_METAL
#define G16_POINTWISE_METAL

// Writes one H coefficient in both encodings. `v` is Montgomery form.
//
// fr_from_mont is a Montgomery multiply by the integer 1, i.e. a bare Montgomery
// reduction, so h_std costs one multiply per element and no branch.
inline void g16_store_h(device Fr* h_mont, device Fr* h_std, uint i, Fr v) {
    h_mont[i] = v;
    h_std[i] = fr_from_mont(v);
}

// Standalone stage 4. The production path does not dispatch this: the same arithmetic is
// fused into the store epilogue of the final NTT batch (see G16_STORE_JOIN in ntt.metal),
// which saves a whole read of A, B and C and a whole write of C. This kernel is kept
// because it is the unfused reference the fused path is checked against, and because a
// caller that wants the three coset vectors materialised (a debugging dump, say) needs a
// way to finish without them being consumed in flight.
kernel void g16_h_join(
    device const Fr* a      [[buffer(0)]],
    device const Fr* b      [[buffer(1)]],
    device const Fr* c      [[buffer(2)]],
    device Fr*       h_mont [[buffer(3)]],
    device Fr*       h_std  [[buffer(4)]],
    constant uint&   n      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    g16_store_h(h_mont, h_std, gid, fr_sub(fr_mul(a[gid], b[gid]), c[gid]));
}

#endif // G16_POINTWISE_METAL
