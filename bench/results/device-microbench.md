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

## Getting the kernels onto the card the first time costs about five minutes

The first run of the CUDA backend on a fresh machine spends roughly **270 to 300 seconds**
before it proves anything. Where that goes is worth being precise about, because the obvious
one-line summary is wrong.

There are two expensive stages, not one, and both are cached by the same place:

| stage | cold | warm |
|---|---:|---:|
| NVRTC, source to PTX | 113.6 s | 0.18 s |
| driver JIT, PTX to SASS, inside `cuModuleLoad` | about 175 s | 0.12 s |
| stages unit, source to PTX | 1.3 s | — |

The NVRTC figure is corroborated offline: `nvcc -arch=compute_75 -ptx` on identical source
takes 108 s and emits byte-identical PTX.

**`~/.nv/ComputeCache` caches both stages, not just the JIT.** Delete it and the next NVRTC
compile takes 113.6 s again; the one after takes 0.18 s. This is the fact that matters when
reading anyone's CUDA prover benchmark, ours included: a cold-start number measured on a
machine that has run the prover before is measuring a cache hit, and the same number on a
fresh CI runner with no persistent `$HOME` is five minutes worse. Ours is reported both ways.

For comparison, Metal's whole kernel library compiles in 53.9 ms through
`newLibraryWithSource`, with no equivalent cliff. This is the one place CUDA is structurally
worse than Metal in this project, and it is entirely a consequence of how much code the
inlining generates.

| | value |
|---|---:|
| PTX emitted for the MSM unit | 787,873 lines / 30.8 MB |
| Metal equivalent, whole library | 53.9 ms, no disk cache needed |

The size comes from `__forceinline__` on 60 functions and 18 templates instantiated over both
`Fq` and `Fq2`. Two ways to shrink it were measured and neither is worth taking:

- **Removing every `#pragma unroll` changes nothing:** 109.7 s against 108.0 s.
- **Demoting `__forceinline__` to `__inline__` is a trap.** It cuts NVRTC to 25.7 s and the
  PTX to 235,000 lines, but leaves 4 functions out of line with 350 call sites, and the
  driver JIT then takes **29.6 seconds** on a warm cache where the fully inlined version
  takes **120 ms**. Fully inlined PTX is straight-line code the JIT merely assembles; PTX
  with calls makes it redo the interprocedural work itself. Total cost is worse and the
  generated code is worse.

`g16-cuda` keeps its own PTX cache, and it is worth being honest about what that buys: it
covers the NVRTC stage only. With `~/.nv` warm it saves nothing measurable, 0.30 s either
way. With `~/.nv` cold it removes 113 s of the 283, and the remaining 175 s of driver JIT is
not cacheable by a process through that API. It is kept because the driver's cache is a
fixed-size LRU shared by every CUDA process on the machine (`CUDA_CACHE_MAXSIZE`, one
gigabyte by default), so a 30 MB entry in it is evictable by unrelated work.
