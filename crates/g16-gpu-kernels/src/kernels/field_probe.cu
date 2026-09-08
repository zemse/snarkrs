// Correctness probe for the Fr layer. Not part of the proving pipeline: it exists so the
// field arithmetic can be checked against arkworks element by element before any of the
// pipeline is trusted, which is the step the reference GPU MSM implementations skip and
// then spend their time debugging a wrong MSM that is really a wrong multiply.

extern "C" __global__ void fr_probe(
    const Fr* __restrict__ a,
    const Fr* __restrict__ b,
    Fr* __restrict__ out_add,
    Fr* __restrict__ out_sub,
    Fr* __restrict__ out_mul,
    Fr* __restrict__ out_neg,
    Fr* __restrict__ out_sqr,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fr x = a[i];
    Fr y = b[i];
    out_add[i] = fr_add(x, y);
    out_sub[i] = fr_sub(x, y);
    out_mul[i] = fr_mul(x, y);
    out_neg[i] = fr_neg(x);
    out_sqr[i] = fr_sqr(x);
}

// Writes the Montgomery representative of 1 and a round trip through standard form.
// A wrong FR_R or FR_R2 shows up here and nowhere else until proofs start failing.
extern "C" __global__ void fr_constants(Fr* __restrict__ out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    out[0] = fr_one();
    out[1] = fr_zero();
    out[2] = fr_from_mont(fr_one());          // must be the integer 1
    out[3] = fr_to_mont(fr_from_mont(fr_one())); // must be fr_one() again
}
