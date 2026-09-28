// Stage 4: H = A*B - C, elementwise, in the encoding stage 9 reads.
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
// ONE OUTPUT BUFFER, IN STANDARD FORM
// ============================================================================
//
// H leaves this stage as h_std, the integer in [0, r) (layout::PackedScalar). That is
// what stage 9's Pippenger MSM wants, because a window digit of a Montgomery
// representative is a digit of a*R mod r, which is a different number. Getting that
// backwards yields a proof wrong by a factor of R.
//
// A Montgomery copy was written beside it for host readbacks, and nothing on the proving
// path read it: 134 MB at 2^22 held until the H MSM finished. A readback converts h_std
// on the host instead (stages::HHandle::to_host).

#ifndef G16_POINTWISE_METAL
#define G16_POINTWISE_METAL

// Writes one H coefficient in standard form. `v` is Montgomery form.
//
// fr_from_mont is a Montgomery multiply by the integer 1, i.e. a bare Montgomery
// reduction, so h_std costs one multiply per element and no branch.
inline void g16_store_h(device Fr* h_std, uint i, Fr v) {
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
    device Fr*       h_std  [[buffer(3)]],
    constant uint&   n      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    g16_store_h(h_std, gid, fr_sub(fr_mul(a[gid], b[gid]), c[gid]));
}

#endif // G16_POINTWISE_METAL
