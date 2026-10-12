# Lanes, one at a time, each against the commit before it

Micro numbers are interleaved A B B A in one session (`raw_ops`, `glv_cost`,
`pipeline_cost`); single prepare runs were timed by wall clock with the output hashed, the
load average logged around each. Every timing session held the shared Mac lock, so no
build or other agent's timing overlapped it. End-to-end numbers are the final three-variant
A/B in [ab-reports.md](ab-reports.md).

## Lane 1: ptau prepare, G2 side (and the shared kernel)

| commit | change | measurement |
|---|---|---|
| 3ec5555 | Fq2 multiply as three wide products and two Montgomery reductions, own aarch64 asm (`raw/wide.rs`) | `raw_ops` Fq2 mul 39.7/38.0 to 31.5 ns; `glv_cost` G2 batch 1024 37.04/30.80/30.78/36.31 us, 2048 36.16/30.02/29.92/36.31, sum of two 53.96/45.64/45.04/54.67; G1 unchanged; prepare 2^18 59.8/56.9/56.8/60.0 s, output identical |
| 0495521 | G2's batch inversion over norms in Fq, inverses rebuilt as conj(d)/norm(d) | `glv_cost` G2 batch 1024 30.59/28.47/28.27/30.33 us, 2048 30.35/28.58/27.98/30.07, sum of two 46.33/42.64/41.55/45.10; G1 20.24/19.95/20.34/20.25; prepare 2^18 (both commits against base) 57.6/54.7/54.0/59.6 s, output identical |

The lazy bound: inputs below q < 2^254, the Karatsuba sums below 2q, so c1 = a0 b1 + a1 b0
is below 2q^2 and c0 = a0 b0 - a1 b1 (plus q 2^256 when negative) below q 2^256; the
reduction takes anything below q 2^256 (the low half reduced on its own is at most q, the
high half is below q, one conditional subtraction). Tested against a u128 schoolbook and
arkworks on carry-edge limbs, products of the 2q - 2 maximum, q 2^256 - 1, and every edge
value in all four Fq2 coefficients; a broken reduction carry (mutation) fails the tests.

Contribute is the same kernel, so it gains with no change of its own; G1 is untouched.

Rejected:

- The lazy multiply in portable u128 Rust: 49.6 ns against 39.7 for three fused asm
  multiplies. LLVM carries the chains through cset/cinc. Hence the asm, and x86_64 keeps
  the old multiply.
- Lazy Fq2 square, `(a0 + a1)(a0 + q - a1)` and `a0 (2 a1)` as wide products: 28.2 ns
  against 26.5. The square is already two products; the saving is two conditional
  subtractions and the extra subtraction costs more.
- Fq multiply as wide product plus the separate reduction (64 multiplier operations
  against 68): 11.83 ns against 11.42. The fused row-by-row shape wins on one element.
- Radix 4 on G2 again: a sum of two G2 products now costs 1.48 single ones (41.6 to 42.6 against
  28.5 us), so four points by three ladders cost 0.99 of two radix-2 passes. Not worth it.
- Radix 8 for G1 prepare (three passes, sums of up to four products under one ladder):
  with the measured costs (a k-term G1 sum about 13 + 7k us) it is 9.9 us a point-pass
  against radix 4's 9.25. Analysis only.

## Lane 2: NTT on the Mac

| commit | change | measurement |
|---|---|---|
| 8dddc4a | fused block's first three DIT passes (and last three DIF) as one radix-8 kernel: five multiplies a run of eight instead of twelve, no one/two/four-butterfly inner loops | `pipeline_cost` 2^21, ms, three pipelines 290.7/274.2/268.8/292.7, one 98.7/91.5/92.3/97.4, one thread 786.7/734.4/730.9/784.1 |
| 825abfc | outer passes read a twiddle table packed for their stride (`stride_twiddles`, the smaller transform's own cached table) | three 274.2/248.8/248.7/268.8, one 91.5/85.7/84.1/92.3, one thread 734.4/696.0/696.1/730.9 |

Base to final: three pipelines 291.7 to 248.8 ms (-15%), one thread 785 to 696 ms (17.8 to
15.8 ns a butterfly). Every timed output checked at two coset points against a barycentric
evaluation of the input; `snarkrs-ntt` tests compare the fused paths against the serial
radix-2 transform up to 2^20, and a swapped eighth root (mutation) fails five of them.

Rejected:

- The remaining in-block passes as radix-4 pairs: middle stage one thread 351 to 346 ms,
  three pipelines 269 to 271 ms. Loads and stores are not where the butterfly's time goes.
- Passes between the 64 KB block and a 1 MB chunk run per chunk while it sits in L2 (DRAM
  trips 5 to 3 a direction): three pipelines 243 to 246 against 247.6. The outer passes
  are not bandwidth bound.
- `TASKS_PER_THREAD` 2 and 8 against 4: 250.0/248.8 and 246.4/244.8 against 247.6/245.2 ms.
- The coset multiply through RawFr on aarch64: one thread 692 to 687 ms, inside noise.

## Lane 3: combined build

No code. One arkab run, three variants: `base` (586f42d on stock arkworks 7ad88c4),
`final` (825abfc on stock), `finalperf` (825abfc on perf arkworks 651414b). Results and the
answer to "does main on perf arkworks get both gains" are in the [README](README.md).
