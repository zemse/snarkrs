# Device microbenchmarks

Measured, not quoted. These are the numbers that explain the end-to-end results, and a few
of them are surprising enough to be worth reading before believing any GPU prover benchmark,
including this one.

Method: `bench/aws/` provisions the NVIDIA box; the probes are small standalone programs
that compile the same `bn254_fr.cuh` / `bn254_fr.metal` the prover uses. The multiply figure
is a dependent chain (`a = fr_mul(a, b)` for a fixed `b`), so it cannot be unrolled away, with
2^20 threads and 512 iterations. Streaming is one load, one add and one store per element.

| | Apple M2 Max (38-core GPU) | NVIDIA Tesla T4 (40 SM, sm_75) |
|---|---:|---:|
| sustained Fr multiply | 4.14 G mul/s | **8.11 G mul/s** |
| best block / threadgroup size | 32 | 64 to 128 (flat above) |
| streaming bandwidth, 2^22 elements | **283 GB/s** | 172 GB/s |
| streaming bandwidth, 2^18 elements | 30 GB/s | **157 GB/s** |
| kernel launch + sync round trip | 0.149 ms | **0.006 ms** |
| CPU-side field multiply, 1 thread | 0.06 G mul/s | n/a |
| CPU-side field multiply, all threads | 0.60 G mul/s (12) | n/a |

Four consequences, all visible in the end-to-end numbers:

1. **The T4 does about twice the field multiplies.** MSM is multiply bound, so the MSM stage
   should be faster on the T4 than on the M2 Max despite the T4 being the cheaper part.
2. **The T4 has 60% of the bandwidth.** The NTT is bandwidth bound, so it should not improve
   by the same factor. Note the M2 Max NTT streams at about 30 GB/s against its own 283 GB/s
   roof, so it is nowhere near its limit either; the bottleneck there is the access pattern,
   not the bus.
3. **CUDA's launch overhead is 25x lower.** Metal's 0.149 ms is per command-buffer
   commit-and-wait, and it is what makes `tiny_mul` cost 37 ms cold on Metal against 6 ms on
   the CPU. CUDA has no equivalent floor, so the small-circuit crossover sits much lower.
4. **Apple's GPU needs a much larger working set to reach its bandwidth.** At 2^18 elements
   the T4 is already at 157 GB/s while the M2 Max is at 30 GB/s, an order of magnitude apart,
   even though the M2 Max wins by 1.6x at 2^22. This is a second, independent reason the
   Metal backend does badly on small circuits.

## The NVRTC compile time, and the cache nobody tells you about

The first NVRTC compile of the MSM translation unit for `sm_75` takes **113.6 seconds**. The
second takes **0.18 seconds**. The difference is not our cache and not warm page cache: the
NVIDIA driver transparently caches NVRTC output in `~/.nv/ComputeCache`. Deleting that
directory reproduces the 113.6 s exactly, and leaves a 30 MB entry behind afterwards.

This matters for anyone reading a CUDA prover benchmark. A cold-start number taken on a
machine that has run the prover before is measuring a cache hit, and the same number taken on
a fresh CI runner is two minutes worse. Ours is reported both ways.

| | value |
|---|---:|
| MSM unit, first compile (`~/.nv` cleared) | 113.6 s |
| MSM unit, subsequent compiles | 0.18 s |
| stages unit, first compile | 1.3 s |
| PTX emitted for the MSM unit | 787,873 lines / 30.8 MB |
| driver JIT of that PTX (PTX to SASS) | 120 ms |
| Metal equivalent (`newLibraryWithSource`, whole library) | 53.9 ms |

The size comes from `__forceinline__` on 60 functions and 18 templates instantiated over both
`Fq` and `Fq2`. Two ways to shrink it were measured and neither is worth taking:

- **Removing every `#pragma unroll` changes nothing:** 109.7 s against 108.0 s.
- **Demoting `__forceinline__` to `__inline__` is a trap.** It cuts NVRTC to 25.7 s and the
  PTX to 235,000 lines, but leaves 4 functions out of line with 350 call sites, and the
  driver's PTX-to-SASS JIT then takes **29.6 seconds** to load it, against 120 ms for the
  fully inlined version. Fully inlined PTX is straight-line code the JIT merely assembles;
  PTX with calls makes the JIT redo the interprocedural work itself. Total cost is worse and
  the generated code is worse, so the inlining stays and the result is cached instead.

`g16-cuda` therefore keeps its own PTX cache keyed by source and architecture. It is not
redundant with the driver's: the driver's is a fixed-size LRU shared by every CUDA process on
the machine (`CUDA_CACHE_MAXSIZE`, one gigabyte by default), so a 30 MB entry in it can be
evicted by unrelated work and reintroduce a two minute stall months later on a machine that
has been fine all along.
