# Graviton CPU matrix: snarkrs vs rapidsnark, 2026-10-10

Source 3968e54 (crates identical to e8449f9 in the x86 matrix), native aarch64 build on Ubuntu 24.04,
no target flags. rapidsnark 81eddf1 / ffiasm aa90166, static aarch64 build with its ARM64 field
assembly (`TARGET_PLATFORM=aarch64`, `USE_ASM=ON`), same warm wrapper as x86. One disposable EC2
instance per type (us-east-1, Ubuntu 24.04 arm64), runner `bench/aws/matrix.py`. Warm proving, all
logical CPUs for both provers (rapidsnark's ffiasm pool matched the vCPU count on every box), 10
reps, median. Our every rep self-verified; our proof plus rapidsnark's first and last proof per
circuit passed snarkjs locally with public signals equal to the reference: 135/135 checks.

The account's on-demand vCPU quota is still 64, so every type above it was refused at launch with
VcpuLimitExceeded: c8g.metal-48xl, c9g.metal-48xl, and the x86 metals c8i.metal-96xl,
c8a.metal-48xl, c7a.metal-48xl, c7i.metal-48xl, c6a.metal, c5.metal. No two-socket host, ARM or
x86, has been measured yet.

## Findings

- We beat rapidsnark on every Graviton box and circuit: 1.19x to 5.48x, even though our aarch64 path
  has no hand-written code and rapidsnark uses its ARM64 assembly.
- Core count decides the gap, as on x86. 8 to 16 vCPU: 1.19x to 2.10x. 64 vCPU: 2.37x to 5.48x.
- Graviton 4 at 8 vCPU (c8g.2xlarge) is our weakest row, 1.19x to 1.87x; the small circuits are
  close to a tie.
- Graviton 4 at 64 vCPU (c8g.16xlarge): anon-aadhaar 290 ms vs rapidsnark 1593 ms. Only c8a.16xlarge
  (199 ms) and c8i.16xlarge (258 ms) were faster in the x86 matrix.
- Metal vs VM, same chip and core count (c7g.metal vs c7g.16xlarge): within 3% for both provers on
  every circuit except tornado (10%). Nitro overhead does not matter here.
- Unlike the IFMA x86 boxes, the MSM still dominates: anon-aadhaar MSM is 2.4x to 3.7x the NTT on
  every Graviton. An aarch64 MSM lane is where the remaining ARM headroom is.
- Memory: same pattern as x86, about 1.25 to 1.3 GiB ours vs about 1 GiB rapidsnark on anon-aadhaar.

## Caveats

- One run per instance type, not interleaved repeats; both provers ran back to back on the same host.
- Warm only. rapidsnark's warm call re-parses the witness inside the timed region; ours does not.
- Ubuntu 24.04 here vs 22.04 in the x86 matrix (our binary needs glibc 2.39). Both provers are
  affected alike; the rapidsnark build is static.

## Machines

| instance | CPU | vCPU | threads/core | sockets | NUMA | L3 | ISA | build | accepted |
|---|---|---:|---:|---:|---:|---|---|---|---|
| c8g.2xlarge | Neoverse-V2 | 8 | 1 | 1 | 1 | 36 MiB (1 instance) | SVE2 | arm64 | yes |
| c6g.4xlarge | Neoverse-N1 | 16 | 1 | 1 | 1 | 32 MiB (1 instance) | NEON only | arm64 | yes |
| c7g.16xlarge | Neoverse-V1 | 64 | 1 | 1 | 1 | 32 MiB (1 instance) | SVE | arm64 | yes |
| c7g.metal | Neoverse-V1 | 64 | 1 | 1 | 1 | 32 MiB (1 instance) | SVE | arm64 | yes |
| c8g.16xlarge | Neoverse-V2 | 64 | 1 | 1 | 1 | 36 MiB (1 instance) | SVE2 | arm64 | yes |

## Speedup over rapidsnark (rapidsnark median / ours median)

| instance | js_2x2_d32 | railgun-01x01 | tornado | sha256 | railgun-13x01 | js_16x16_d32 | rsa2048 | keccak256 | anon-aadhaar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c8g.2xlarge | 1.19x | 1.22x | 1.29x | 1.63x | 1.45x | 1.45x | 1.87x | 1.61x | 1.53x |
| c6g.4xlarge | 1.72x | 1.63x | 1.83x | 1.93x | 1.69x | 1.89x | 2.10x | 1.69x | 2.02x |
| c7g.16xlarge | 2.60x | 2.65x | 2.62x | 2.87x | 2.38x | 2.73x | 3.63x | 2.92x | 4.60x |
| c7g.metal | 2.42x | 2.50x | 2.76x | 2.69x | 2.37x | 2.74x | 3.66x | 2.90x | 4.54x |
| c8g.16xlarge | 2.43x | 2.64x | 2.59x | 2.87x | 2.65x | 2.91x | 4.08x | 3.27x | 5.48x |

## Ours, warm median ms

| instance | js_2x2_d32 | railgun-01x01 | tornado | sha256 | railgun-13x01 | js_16x16_d32 | rsa2048 | keccak256 | anon-aadhaar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c8g.2xlarge | 102.0 | 78.9 | 132.0 | 66.9 | 474.9 | 623.4 | 262.9 | 261.3 | 1963.6 |
| c6g.4xlarge | 169.1 | 140.2 | 222.2 | 143.4 | 822.3 | 1014.6 | 553.6 | 535.8 | 2919.0 |
| c7g.16xlarge | 25.1 | 21.1 | 32.5 | 28.8 | 129.5 | 142.9 | 104.1 | 101.6 | 385.5 |
| c7g.metal | 24.8 | 20.4 | 29.1 | 28.7 | 127.1 | 140.4 | 101.3 | 100.0 | 384.5 |
| c8g.16xlarge | 23.1 | 18.4 | 28.6 | 24.9 | 103.5 | 119.2 | 82.8 | 81.8 | 290.4 |

## rapidsnark, warm median ms

| instance | js_2x2_d32 | railgun-01x01 | tornado | sha256 | railgun-13x01 | js_16x16_d32 | rsa2048 | keccak256 | anon-aadhaar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c8g.2xlarge | 121.8 | 96.5 | 170.5 | 109.0 | 686.3 | 903.5 | 492.0 | 422.0 | 3005.7 |
| c6g.4xlarge | 290.0 | 228.3 | 407.6 | 276.4 | 1390.5 | 1914.6 | 1163.7 | 907.0 | 5910.7 |
| c7g.16xlarge | 65.1 | 56.1 | 85.1 | 82.7 | 308.9 | 390.6 | 377.8 | 297.0 | 1771.7 |
| c7g.metal | 60.0 | 51.1 | 80.5 | 77.1 | 301.7 | 384.3 | 370.5 | 289.9 | 1744.6 |
| c8g.16xlarge | 56.0 | 48.5 | 74.2 | 71.3 | 274.0 | 346.7 | 338.0 | 267.7 | 1592.5 |

## Our stage split, anon-aadhaar (median ms)

| instance | gather | ntt | msm | prepare | peak RSS ours / rapidsnark MiB |
|---|---:|---:|---:|---:|---|
| c8g.2xlarge | 18.457 | 565.731 | 1360.611 | 243.543 | 1247.6 / 976.9 |
| c6g.4xlarge | 24.45 | 971.835 | 1889.612 | 466.723 | 1247.6 / 1008.4 |
| c7g.16xlarge | 8.19 | 88.245 | 271.775 | 213.071 | 1295.0 / 987.9 |
| c7g.metal | 7.874 | 86.482 | 268.648 | 211.462 | 1274.9 / 989.6 |
| c8g.16xlarge | 7.152 | 73.102 | 196.918 | 171.708 | 1296.6 / 989.9 |
