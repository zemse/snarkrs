// Stage 4: H = A*B - C, elementwise, and the two encodings it is written in.
//
// Twin of crates/g16-metal/src/shaders/pointwise.metal. Depends on bn254_fr.cuh being
// concatenated ahead of it. No #include: NVRTC has no filesystem.
//
// ORDERING NOTE. In the Metal backend this file is concatenated BEFORE ntt.metal, because
// the fused NTT epilogue calls g16_store_h and MSL needs the definition in scope. Here
// g16_gpu_kernels::unit_stages fixes the order as gather / ntt / pointwise, so ntt.cu
// carries a forward declaration of g16_store_h and the definition stays here, next to the
// stage it belongs to. Same translation unit, so the function still inlines into the NTT
// epilogue. If you move this file's contents, move the declaration in ntt.cu with it.
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
// (g16_core::cpu::CpuCircuit::compute_h) and the Metal backend make the same choice and
// the same argument; this has been checked against real proofs, it is not a guess.
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
// The alternative is one buffer plus a conversion launch later, which costs an extra
// full read and write of the domain (2n * 32 bytes) plus a launch, against n * 32 bytes
// of extra writes here. On a stage that is already memory bound the second write is the
// cheaper of the two, and it removes a coordination point between this stage and the MSM
// stage entirely. On a discrete card there is a second reason: h_std is the only buffer
// the MSM stage needs, so writing it here is what lets H stay device resident
// (g16_core::HPoly::Device) and never cross PCIe at all.

#ifndef G16_POINTWISE_CU
#define G16_POINTWISE_CU

// Writes one H coefficient in both encodings. `v` is Montgomery form.
//
// fr_from_mont is a Montgomery multiply by the integer 1, i.e. a bare Montgomery
// reduction, so h_std costs one multiply per element and no branch.
//
// Declared in ntt.cu, which is concatenated ahead of this file; the signature there must
// stay character-for-character the same as this one.
__device__ __forceinline__ void g16_store_h(Fr* h_mont, Fr* h_std, u32 i, Fr v) {
    h_mont[i] = v;
    h_std[i] = fr_from_mont(v);
}

// Standalone stage 4. The production path does not launch this: the same arithmetic is
// fused into the store epilogue of the final NTT batch (see G16_STORE_JOIN in ntt.cu),
// which saves a whole read of A, B and C and a whole write of C. This kernel is kept
// because it is the unfused reference the fused path is checked against, and because a
// caller that wants the three coset vectors materialised (a debugging dump, say) needs a
// way to finish without them being consumed in flight.
//
// LAUNCH: grid_dim.x = ceil(n / block_dim.x), block_dim.x free (256). One thread does one
// element; a block is 256 independent elements with no sharing, no barrier and no shared
// memory.
extern "C" __global__ void g16_h_join(
    const Fr* a,
    const Fr* b,
    const Fr* c,
    Fr*       h_mont,
    Fr*       h_std,
    const u32 n)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load bearing on CUDA, unlike on Metal: the grid is a whole number of blocks, so the
    // tail block runs threads past n and without this they would write past h_mont.
    if (gid >= n) {
        return;
    }
    g16_store_h(h_mont, h_std, gid, fr_sub(fr_mul(a[gid], b[gid]), c[gid]));
}

// Montgomery limbs to standard limbs, element-wise. Stage 0.5 of the witness path: the
// witness is uploaded once, in Montgomery form for the gather, and this converts it in
// place on the device into the standard-form copy the MSM digit decomposition needs, so
// the MSM stage never packs or uploads the witness a second time (that used to be a
// second full PCIe transfer plus a host-side Montgomery reduction per element, per
// proof). One thread per element, no sharing, tail-guarded like everything else.
extern "C" __global__ void g16_mont_to_std(
    const Fr* in,
    Fr*       out,
    const u32 n)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = fr_from_mont(in[gid]);
}

#endif // G16_POINTWISE_CU
