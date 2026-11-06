// Stage 0 for a key that carries no constraint matrices, and the tail that takes H to the
// basis a gnark key's Z is laid out for.
//
// No Metal or WGSL twin: only this backend proves from A.w and B.w rows on a device.
// Depends on bn254_fr.cuh, on g16_brev in ntt.cu and on g16_store_h in pointwise.cu, so
// kernels::unit_stages concatenates it after both. No #include: NVRTC has no filesystem.

#ifndef G16_ROWS_CU
#define G16_ROWS_CU

// Stage 0 from the caller's rows. The host has copied A.w and B.w into a[0..rows) and
// b[0..rows); this writes C = A*B on those rows, for the reason gather.cu gives, and zero
// in all three past them. The tail has to be written every proof: the transforms run in
// place on a pooled buffer, so what the last proof left there is its coset evaluations.
//
// LAUNCH: grid_dim.x = ceil(n / block_dim.x), block_dim.x free (256), one thread an element.
extern "C" __global__ void g16_rows_abc(
    Fr*       a,
    Fr*       b,
    Fr*       c,
    const u32 rows,
    const u32 n)
{
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    if (gid < rows) {
        c[gid] = fr_mul(a[gid], b[gid]);
    } else {
        Fr z = fr_zero();
        a[gid] = z;
        b[gid] = z;
        c[gid] = z;
    }
}

// g holds the coefficients of P(x) = V(shift x), natural order: the inverse transform of
// V = A*B - C on the odd coset. P's coefficient j is V's times shift^j, so V's is
// g[j] * shift^-j, and shift is a primitive 2n-th root, so shift^-j = shift^(2n - j) =
// -shift^(n - j) for 0 < j < n, which the coset power table already holds. Position p of
// the output is V's coefficient bitrev(p): the order gnark's computeH leaves H in, and the
// one pk.G1.Z is laid out for.
//
// V is t * H with t = x^n - 1, which is -2 everywhere on the odd coset, so these are H's
// coefficients times -2; the host multiplies the MSM result by -1/2 instead of n elements
// by it here. Coefficient n - 1 sits at p = n - 1 (it reverses to itself) and must be zero.
//
// LAUNCH: grid_dim.x = ceil(n / block_dim.x), block_dim.x free (256), one thread an output.
extern "C" __global__ void g16_gnark_coeffs(
    const Fr* g,
    const Fr* pows,
    Fr*       h_mont,
    Fr*       h_std,
    const u32 log_n)
{
    u32 n = 1u << log_n;
    u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    // `>> 32` is undefined, so a one-point domain takes the uniform branch.
    u32 j = (log_n == 0u) ? 0u : (g16_brev(p) >> (32u - log_n));
    Fr x = g[j];
    if (j != 0u) {
        x = fr_neg(fr_mul(x, pows[n - j]));
    }
    g16_store_h(h_mont, h_std, p, x);
}

#endif // G16_ROWS_CU
