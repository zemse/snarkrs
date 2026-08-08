# Results

`comparison.csv` is the raw per-rep data. Every row is a proof that was verified before
its timing was recorded.

## CPU, Apple M2 Max (8P + 4E, 96 GB), 10 reps, medians

| variant | constraints | rapidsnark cold | rapidsnark warm | ours cold | ours warm | ours/rs cold | ours/rs warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_mul | 2 | 4.7 | 1.0 | 4.8 | 1.0 | 1.03x | 0.95x |
| js_1x1_d8 | 3,359 | 29.2 | 24.0 | 28.3 | 22.1 | 0.97x | 0.92x |
| js_2x2_d16 | 10,153 | 57.8 | 54.0 | 67.5 | 57.5 | 1.17x | 1.06x |
| js_2x2_d32 | 17,929 | 94.4 | 88.0 | 103.8 | 94.2 | 1.10x | 1.07x |
| js_8x8_d32 | 70,357 | 289.3 | 289.0 | 343.7 | 321.2 | 1.19x | 1.11x |
| js_16x16_d32 | 140,261 | 554.0 | 538.5 | 639.5 | 596.3 | 1.15x | 1.11x |

All times in milliseconds. Below 1.00x we are faster.

Read it as: at parity below about 5K constraints, and 11 to 19 percent behind rapidsnark
across the range that matters. rapidsnark is hand-tuned C++ over ffiasm's generated field
assembly, so being within 20 percent of it in safe Rust on the first pass is a reasonable
place to start from, not a place to stop.

## Where the gap is

MSM is 76 to 82 percent of our proving time, against the 50 to 70 percent the stage should
occupy. The cause is measured and specific: bucket accumulation is Jacobian. rapidsnark
and sppark accumulate buckets in **affine** coordinates using batch addition, trading a
field inversion amortised across the whole bucket array for the 5 to 8 extra field
multiplications every Jacobian addition costs. Our MSM is already at parity with
arkworks (1.00 to 1.07x single-threaded, 1.00 to 1.04x with rayon), and it scales 6.9x on
12 threads, so this is algorithmic headroom rather than a tuning or threading defect.

## What these numbers are not

- **Not comparable to the earlier VM results** in the research corpus. That harness
  measured `js_1x1_d8` at 7,211 constraints; the same nominal parameters compile to 3,359
  here under a different circom and circomlib. Within a single run every prover proves the
  same artifact, which is the property that matters for a comparison.
- **Not from a silent machine.** Load average during the run was around 10 on a 12-core
  box, mostly from the benchmark itself. Medians over 10 reps absorb most of that; the
  spread columns in the CSV are the honest guide to how much they did not.
- **Cold here is genuinely cold**: one fresh process per proof, so the zkey parse is inside
  the timed region. Our cold-warm gap is small because our parse is fast, not because the
  measurement is lenient.
