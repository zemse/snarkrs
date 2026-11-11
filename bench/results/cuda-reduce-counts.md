# CUDA G2 reduce source counts

CPU-only replay against source at `7e65985` (calibration source `4956315`,
subsequent changes were documentation/comments). This is count evidence, not an
optimisation, measurement, performance claim or numerical timing floor. See
[cuda-fixed-cost-counts.md](cuda-fixed-cost-counts.md) and
[cuda-intrinsics-2026-11-11.md](cuda-intrinsics-2026-11-11.md).

## Model

`review_scalars` supplies the actual uniform/mixed inputs. `gate_digits` uses
independent-window carry from the preceding bit, not propagated signed-digit
carry. Bases are exactly 2G through (n+1)G. General scalars contribute
sign(digit) * (original index + 2) to bucket abs(digit)-1. Zero and one scalars
are excluded from these buckets; ones are a separate contribution. Each input's
coefficient matrix is built once and reused for all four RG shapes. `emitted`
is actual nonzero digits E, not entry capacity. `nonzero_buckets` counts
nonidentity G2 bucket values, not merely buckets with entries.

Coefficients are signed integers in the prime-order generator. With
S = sum(2..n+1) = n(n+3)/2, every intermediate in a window, including offset
ladders and tree partials, has absolute value at most S * 2^(c-1). This follows
by bounding each partial's nonnegative bucket weight by 2^(c-1), and each
scalar's contribution by its base weight. For n <= 262144 and c <= 13 this is
140739098968064, below 2^48 and far below the subgroup order (254 bits).
Pairwise differences are also below the subgroup order. Thus integer zero,
equality and opposite coefficients exactly classify the source group branches.
All coefficient additions and multiplications use checked arithmetic. The
bound does not apply across windows: full MSM reconstruction uses Fr Horner
with radix 2^c, then adds the separate ones contribution.

The replay follows `msm_reduce_impl` descending run/tot sums, MSB-first
`pt_mul_small` including its initial infinity double/add and early return for
zero point or multiplier, final mine addition, and shared binary tree.
Geometry and `reduce_threads` come from the host helpers, with c13 and slice64.
Host group folding and cross-window Horner are checked but not counted as
CUDA reduce work. Accumulation, merge, ones scan and host combine are excluded.

Addition categories are ordered exactly as the source guards: left identity
(including both zero), right identity, finite equal, finite cancellation,
generic. Identities perform no field arithmetic. Equal and cancel each perform
4 Fq2 multiplies before the branch; equal additionally invokes a generic XYZZ
doubling (6M+3S). Generic addition is 12M+2S. `ladder_double_identity` and
`ladder_double_generic` exclude doubles induced by equal additions, whose count
is exactly `equal`. `ladders` counts calls including early returns;
`ladder_adds` counts only actually executed internal addition calls.

For every row the replay asserts:

- sum of all five addition categories = 2 * windows * buckets + ladders +
  blocks * (threads-1) + ladder_adds.
- Each replayed window equals an independent integer weighted bucket sum.

The field totals are computed as:

- Fq2 M = 4*(equal+cancel) + 12*generic + 6*(ladder_double_generic+equal).
- Fq2 S = 2*generic + 3*(ladder_double_generic+equal).

These two identities are also audited independently with Python integers;
Fq equivalents are 3*M+2*S because Fq2 multiply/square uses 3/2 Fq multiplies.
These are mathematical source-branch counts, **not observed PTX/SASS instruction
counts**, register spills, physical per-warp work, SIMT reconvergence or a
guaranteed timing floor. Per-warp physical simulation is out of scope. In
particular these totals do not establish the conditional composite budget as a
floor; heterogeneous lane behavior still requires separate evidence.

## Reproduction and checks

```sh
cargo fmt --all -- --check
cargo test --locked -p snarkrs-cuda --features cuda --lib reduce_
cargo test --locked -p snarkrs-cuda --features cuda --lib geometry
cargo test --locked -p snarkrs-cuda --features cuda --lib \
  gate_recoding_oracle_reconstructs_carry_boundaries
cargo test --locked -p snarkrs-cuda --features cuda --lib -- \
  --ignored --exact msm::tests::variable_work_reduce_counts --nocapture
ulimit -n 4096
RUSTFLAGS='-C target-feature=+adx,+bmi2' cargo zigbuild --locked \
  --target x86_64-unknown-linux-gnu.2.31 -p snarkrs-cuda \
  --features cuda --tests --lib
```

Observed on the native macOS host: `reduce_` passed 5 tests with the count probe
ignored; `geometry` passed 3; carry reconstruction passed 1; the explicitly
ignored count probe passed 1 and emitted the 16 rows below. No device was opened.
The two new ordinary tests cover branch categories, ladder zero/one/13 cases,
checked maximum coefficient bound, c2/c3/c13 carry boundaries, mixed scalars,
zeros and ones with original indices retained, and RG1/4/8/16/32. Toy full MSMs
agree with an independent Fr weighted scalar sum and `CpuMsm` G2. The existing
recoding reconstruction test now also includes c13. Python integer auditing
passed all 16 addition reconciliations, field totals and the maximum bound.

Locked Linux GNU 2.31 ADX/BMI2 library-test cross-build passed after raising the
shell file-descriptor limit. The first link attempt failed with
`ProcessFdQuotaExceeded`; no source change or installation was needed. This is
cross-build coverage only, not Linux execution or GPU coverage. Existing CSV
schemas are unchanged.

## Exact probe output

```csv
reduce_counts set,n,c,slice,windows,buckets,rg,threads,blocks,emitted,nonzero_buckets,ladders,ladder_adds,left_identity,right_identity,equal,cancel,generic,ladder_double_identity,ladder_double_generic,fq2_m,fq2_s,fq_m_equivalent
reduce_counts uniform,65536,13,64,20,4096,4,64,80,1310203,77921,5120,19465,23082,24,0,0,170359,4851,48634,2336112,486620,7981576
reduce_counts uniform,65536,13,64,20,4096,8,64,160,1310203,77921,10240,43798,38188,24,0,0,189746,9721,97322,2860884,671458,9925568
reduce_counts uniform,65536,13,64,20,4096,16,64,320,1310203,77921,20480,97334,68400,24,0,0,233390,19461,194678,3968748,1050814,14007872
reduce_counts uniform,65536,13,64,20,4096,32,64,640,1310203,77921,40960,214146,128824,24,0,0,330418,38941,389370,6301236,1828946,22561600
reduce_counts mixed,65536,13,64,20,4096,4,64,80,760018,77793,5120,19465,23102,142,9,0,170212,4851,48634,2334438,486353,7976020
reduce_counts mixed,65536,13,64,20,4096,8,64,160,760018,77793,10240,43798,38218,137,16,0,189587,9721,97322,2859136,671188,9919784
reduce_counts mixed,65536,13,64,20,4096,16,64,320,760018,77793,20480,97334,68467,120,37,1,233189,19461,194678,3966710,1050523,14001176
reduce_counts mixed,65536,13,64,20,4096,32,64,640,760018,77793,40960,214144,128947,92,66,1,330158,38940,389365,6298750,1828609,22553468
reduce_counts uniform,262144,13,64,20,4096,4,64,80,5240726,77921,5120,19465,23082,24,0,0,170359,4851,48634,2336112,486620,7981576
reduce_counts uniform,262144,13,64,20,4096,8,64,160,5240726,77921,10240,43798,38188,24,0,0,189746,9721,97322,2860884,671458,9925568
reduce_counts uniform,262144,13,64,20,4096,16,64,320,5240726,77921,20480,97334,68400,24,0,0,233390,19461,194678,3968748,1050814,14007872
reduce_counts uniform,262144,13,64,20,4096,32,64,640,5240726,77921,40960,214146,128824,24,0,0,330418,38941,389370,6301236,1828946,22561600
reduce_counts mixed,262144,13,64,20,4096,4,64,80,3040027,77921,5120,19465,23082,24,0,0,170359,4851,48634,2336112,486620,7981576
reduce_counts mixed,262144,13,64,20,4096,8,64,160,3040027,77921,10240,43798,38188,24,0,0,189746,9721,97322,2860884,671458,9925568
reduce_counts mixed,262144,13,64,20,4096,16,64,320,3040027,77921,20480,97334,68400,24,0,0,233390,19461,194678,3968748,1050814,14007872
reduce_counts mixed,262144,13,64,20,4096,32,64,640,3040027,77921,40960,214146,128824,24,0,0,330418,38941,389370,6301236,1828946,22561600
```
