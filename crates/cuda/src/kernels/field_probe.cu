// Correctness probes for the field layers. Not part of the proving pipeline: they exist so
// the field arithmetic can be checked against arkworks element by element before any of
// the pipeline is trusted, which is the step the reference GPU MSM implementations skip
// and then spend their time debugging a wrong MSM that is really a wrong multiply.
//
// Three fields, three probes. `Fr` is what the FFT stages compute in; the MSM computes in
// `Fq` and `Fq2`, which are a deliberate second copy of the same CIOS layout in
// bn254_curve.cuh, so a correct `fr_mul` says nothing about `fq_mul`. This unit is
// assembled as FR_CUH + CURVE_CUH + this file (kernels.rs), so the Fq and Fq2 functions
// probed here are the very definitions the MSM and FFT units compile, not a third copy.
//
// Every probe has the same shape: a and b in, one output array per operation, a flags
// word for the predicates the point formulas branch on. Adding an operation is one more
// output pointer and one more line in each of the two places (here and tests/field_gpu.rs).

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

// Bit layout of the flags word, shared by fq_probe and fq2_probe.
#define PROBE_FLAG_IS_ZERO 1u
#define PROBE_FLAG_EQ 2u

// `out_dbl` is fq_add(x, x), which is how the point formulas double a coordinate (there
// is no fq_dbl), and the same-operand case fq_sqr shares with it.
extern "C" __global__ void fq_probe(
    const Fq* __restrict__ a,
    const Fq* __restrict__ b,
    Fq* __restrict__ out_add,
    Fq* __restrict__ out_sub,
    Fq* __restrict__ out_mul,
    Fq* __restrict__ out_neg,
    Fq* __restrict__ out_sqr,
    Fq* __restrict__ out_dbl,
    unsigned int* __restrict__ out_flags,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fq x = a[i];
    Fq y = b[i];
    out_add[i] = fq_add(x, y);
    out_sub[i] = fq_sub(x, y);
    out_mul[i] = fq_mul(x, y);
    out_neg[i] = fq_neg(x);
    out_sqr[i] = fq_sqr(x);
    out_dbl[i] = fq_add(x, x);
    out_flags[i] = (fq_is_zero(x) ? PROBE_FLAG_IS_ZERO : 0u)
                 | (fq_eq(x, y) ? PROBE_FLAG_EQ : 0u);
}

// Fq2 has no mul-by-nonresidue or conjugate of its own: u^2 = -1 is folded into fq2_mul
// and fq2_sqr, so those two plus add/sub/neg are the whole surface the G2 formulas use.
extern "C" __global__ void fq2_probe(
    const Fq2* __restrict__ a,
    const Fq2* __restrict__ b,
    Fq2* __restrict__ out_add,
    Fq2* __restrict__ out_sub,
    Fq2* __restrict__ out_mul,
    Fq2* __restrict__ out_neg,
    Fq2* __restrict__ out_sqr,
    Fq2* __restrict__ out_dbl,
    unsigned int* __restrict__ out_flags,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fq2 x = a[i];
    Fq2 y = b[i];
    out_add[i] = fq2_add(x, y);
    out_sub[i] = fq2_sub(x, y);
    out_mul[i] = fq2_mul(x, y);
    out_neg[i] = fq2_neg(x);
    out_sqr[i] = fq2_sqr(x);
    out_dbl[i] = fq2_add(x, x);
    out_flags[i] = (fq2_is_zero(x) ? PROBE_FLAG_IS_ZERO : 0u)
                 | (fq2_eq(x, y) ? PROBE_FLAG_EQ : 0u);
}

// fq_one() is FQ_R copied out of constant memory. None of the element-wise probes above
// reads FQ_R, so a wrong value is invisible to them and shows up only here, decoded against
// arkworks' one. Fq has no to/from-Montgomery conversion because nothing on the device
// needs one: points arrive already in Montgomery form.
extern "C" __global__ void fq_constants(Fq* __restrict__ out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    out[0] = fq_one();
    out[1] = fq_zero();
    Fq2 one2 = fq2_one();
    out[2] = one2.c0;   // must be fq_one()
    out[3] = one2.c1;   // must be fq_zero()
}
