// Opt-in calibration only. Concatenated after the shared field/curve preamble.
// Runtime inputs, loop bounds and final writes keep the dependent work observable.

#ifdef G16_PROBE_G2
extern "C" __global__ void intrinsic_g2(
    const AffG2* input, PtG2* output, u32 steps, u32 offset)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    PtG2 p = pt_from_affine(input[gid]);
#if G16_PROBE_G2 == 0
    PtG2 q = pt_from_affine(input[gid + gridDim.x * blockDim.x]);
#endif
#pragma unroll 1
    for (u32 i = 0; i < steps; i++) {
#if G16_PROBE_G2 == 0
        p = pt_add(p, q);
#elif G16_PROBE_G2 == 1
        p = pt_dbl(p);
#else
        p = pt_mul_small(p, offset);
#endif
    }
    output[gid] = p;
}
#endif

#ifdef G16_PROBE_FQ
extern "C" __global__ void intrinsic_fq(const Fq* input, Fq* output, u32 steps)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    Fq x = input[gid];
    Fq y = input[gid + gridDim.x * blockDim.x];
#pragma unroll 1
    for (u32 i = 0; i < steps; i++) {
        x = fq_mul(x, y);
    }
    output[gid] = x;
}
#endif

#ifdef G16_PROBE_STORES
// Identical G2 identity-clear semantics: write zz at byte offset 128 of each 256B row.
extern "C" __global__ void intrinsic_clear(PtG2* output, u32 rows)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < rows) {
        Fq2 z;
        f_set_zero(z);
        output[gid].zz = z;
    }
}

extern "C" __global__ void intrinsic_contiguous(u32* output, u32 words)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < words) output[gid] = 0u;
}
#endif

#ifdef G16_PROBE_OVERHEAD
extern "C" __global__ void intrinsic_empty() {}
#endif
