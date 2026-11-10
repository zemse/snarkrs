# CUDA power-of-two size sweep on a T4, 2026-11-10

BUG-04 and OPT-08: where a Tesla T4 (15 GB usable VRAM) stops proving, and what the MSM's
VRAM headroom has to cover. One process per domain and phase,
`bench/scripts/cuda-size-sweep.sh` around `size_sweep_point` in
`crates/cuda/tests/size_sweep_gpu.rs`. The key is synthetic: `n_vars` = 0.75 of the domain
(the onion ladder's wires over its domain), two random CSR matrices of about 1.5 entries a
row, distinct generator multiples as bases, a uniform random witness (the MSMs' worst case:
no zeros or ones). A random key verifies against nothing, so the `cpu` phase proves the same
key on the CPU backend at the same blinders and the proof must be bit-identical.

g4dn.xlarge (4 vCPU, 16 GB host) for 2^16 to 2^24, g4dn.2xlarge (8 vCPU, 32 GB) for 2^25,
driver 595.112.07. 2^16 to 2^22 run as one batch and were measured on main `99ecd41`; 2^23
with `28dc889` (the budget synchronizes before it reads free memory); 2^24 and 2^25 with
`be5f18c` too (64 MiB headroom).

## Result

| domain | runs | prove ms | MSM ms | VRAM peak MiB | host VmHWM MiB | CPU match | CPU prove s |
|---|---:|---:|---:|---:|---:|---|---:|
| 2^16 | 1 | 47 | 43 | 251 | 3285 * | yes | 0.8 |
| 2^17 | 1 | 85 | 75 | 379 | 382 | yes | 1.5 |
| 2^18 | 1 | 164 | 142 | 635 | 442 | yes | 2.8 |
| 2^19 | 1 | 314 | 270 | 1147 | 564 | yes | 5.1 |
| 2^20 | 1 | 648 | 563 | 2139 | 892 | yes | 9.8 |
| 2^21 | 1 | 1263 | 1085 | 4123 | 1556 | yes | 19.0 |
| 2^22 | 1 | 2479 | 2110 | 8091 | 2890 | yes | 37.6 |
| 2^23 | 3 | 5002 | 4212 | 10619 | 5725 | yes | 74.8 |
| 2^24 | 10 | 9990 | 8387 | 14843 | 11429 | yes | 142.0 |
| 2^25 | - | - | - | 13563 | 22849 | - | - |

Prove ms is a warm `prove_with_blinders` after one MSM call on the same context; VRAM peak
is nvidia-smi every 100 ms over the whole process; VmHWM is the `cuda` process (from 2^22 up
the `cpu` process peaks within 5% of it). * 2^16 ran first on a fresh box and its VmHWM is the NVRTC
compile of both units (391 s cold, 0.2 s warm after).

- **2^16 to 2^24 prove on a g4dn.xlarge**, every proof bit-identical to the CPU's. From
  2^23 the five MSMs no longer fit at once and run cut into runs.
- **2^25 does not.** Preparing the key fails in the base/CSR upload with the driver's
  `CUDA_ERROR_OUT_OF_MEMORY`, surfaced as `ProveError::Device` after 95 s of key building:
  an error, not a crash or hang, but not a planned refusal either. Only the MSM scratch is
  budgeted; the resident key (7.8 GB at 2^24) and the stage 0-4 scratch (3.8 GB at 2^24)
  are allocated unchecked.
- **Host RAM is the xlarge's limit at the same step.** A 2^25 key holds 22.8 GB of host
  memory, past the xlarge's 16 GB; 2^24 needs 11.4 GB and fits. On the xlarge both host
  and VRAM stop at 2^25; the 2xlarge shows the device stops there on its own.
- **Before `28dc889` 2^24 was refused mid-proof.** A dropped buffer is a `cuMemFreeAsync`
  whose pages the pool returns only at a synchronize, so each run's budget query saw the
  last run's scratch as taken: "an MSM batch needs 1638055936 bytes of device memory and
  1448935424 are available to it", from a schedule planned to fit.

## VRAM headroom

Each MSM call's device peak is the free memory before it less the least free a 200 us poller
saw during it; its planned bytes are the least `with_vram_limit` that still runs it as one
batch, bisected to 64 KiB.

| domain | planned MiB | peak, first call | peak, warm | over plan |
|---|---:|---:|---:|---:|
| 2^16 | 82.2 | 64 | 64 | 0 |
| 2^17 | 161.2 | 160 | 160 | 0 |
| 2^18 | 319.2 | 320 | 320 | 0.8 |
| 2^19 | 635.3 | 640 | 640 | 4.7 |
| 2^20 | 1267.3 | 1280 | 1280 | 12.7 |
| 2^21 | 2531.3 | 2528 | 2528 | 0 |
| 2^22 | 5059.3 | 5056 | 5056 | 0 |

Free memory moves in 32 MiB steps, the first call on a context peaks no higher than a warm
one (no local-memory reservation lands inside a batch), and nothing reaches a second step.
`VRAM_HEADROOM` went from an unmeasured 256 MiB to 64 MiB. A/B on one box, interleaved,
5 reps each, prove ms at 256 / 64 MiB: 2^16 45-48 / 46-48, 2^20 646-650 / 644-651, one run
each. At 2^24 the runs go 20 to 10 and the proof 10.39 s to 9.99 s; the least free memory
during that proof was 70 MiB.
