# Lanes, one at a time, each against the commit before it

Micro numbers are interleaved A B B A in one session (`glv_cost`, `setup_stages`,
`msm_shape`, or a single-thread fill probe); end-to-end lane numbers are `arkab.py`, three
rounds, order reversed each round. Single prepare runs below were timed with
`/usr/bin/time` and discarded when another process went over two cores during the run.

## Lane 1: ptau prepare (and the shared kernel)

| commit | change | measurement |
|---|---|---|
| a15e904 | two running-product chains in `mul_batch`'s inversion | G1 batch 1024 21.57/20.32/20.30/21.25 us, 4096 20.81/20.09/19.86/20.31; G2 unchanged |
| ff2e3d2 | `mul_sum_batch_xyzz_in`, sums under one ladder | sum of two: G1 1.34x a product, G2 1.51x; one-term path unchanged |
| b572aa5 | radix-4 G1 passes in prepare | prepare 2^18 65.97/62.83/62.80/65.80 s (-4.7%), output identical |
| 97e93f6 | G2 passes batched from 64 butterflies | prepare 2^13 2.18/2.03/2.01/2.16 s (-6.9%), 2^16 15.62/15.06/15.07/15.58 (-3.5%) |
| adfb5ec | `phi psi` table from the `psi` one | G2 batch 1024 37.85/37.03/36.99/37.68 us (-2.0%) |

Lane A/B, three rounds (ab-reports.md): prepare 2^18 66,624 to 61,007 ms (-8.4%), 2^13
-7.4%, contribute 2^19 -2.8%; 4 of 4 outputs identical. The interleaving (lane 3's
finding, applied back to `mul_batch` at the end) took prepare 2^18 the rest of the way to
57,975 ms in the final A/B.

Rejected: dedicated squaring (no gain twice, see the README), four inversion chains (none),
larger batches (none visible at 2^18), radix 4 on G2 (a sum of two G2 products costs 1.51).

## Lane 2: contribute, setup

Contribute is `mul_batch` on every point (`sample`: kernel bound, a few percent waiting),
so its gains are lane 1's: ptau contribute 2^19 -6.7%, zkey contribute -8.5% to -13.6% in
the final A/B. Its scalar is constant across a section, which buys nothing in the ladder
(every point still doubles 127 times).

| commit | change | measurement |
|---|---|---|
| 7bff2b1 | `mul_sums_xyzz_in`: a term count per sum, rounds by nonzero digit, tables only as long as used | `glv_cost` unchanged for full-size scalars |
| 2b58f3f | setup's small slots through batched ladders, threshold 32 to 256, `1`/`-1` as additions | js_16x16_d32 compose 5.97 to 3.08 s (setup_stages: c 2603/1533/1554/2449, a 434/261/263/470, b1 769/394/414/811, b2 2188/879/883/2228 ms); anon-aadhaar compose_c 816/660/660/978 |
| 7303d19 | setup's point hashing encoded in parallel, hash overlapped with the next chunk | anon-aadhaar hash_h_points 559 to 331 to 190 ms; compose stages -7% to -15% from the section hashing |

Sweeps behind the constants (js_16x16_d32 compose_c, ms): threshold 32 2136, 128 1807,
256 1782, 512 1774; terms a sum 4 2074, 8 1834/1779, 16 1679/1558, 32 1807/1608; terms a
task 2048 1900/1759, 4096 1679/1558, 8192 1700/1518. A first version without the `1`/`-1`
shortcut regressed anon-aadhaar's compose_b2 by 11% (391 to 437 ms); the shortcut fixed it.

## Lane 3: proving MSM

| commit | change | measurement |
|---|---|---|
| dab65e5 | Pippenger tasks sized for twice the pool | H 588 to 535 ms (two chunks a window; three 531, four 537) |
| 9b24cfb | bucket `(x, y)` in one line, flush additions two at a time | probe 119.0/119.3 to 115.0/117.2 ns (paired), then 121.4/121.9 to 106.8/109.8 (interleaved); H 536/548 to 462/469 ms |
| 196d6bb | the same interleaving in `mul_batch` | G1 batch 1024 22.46/20.70/20.47/22.27 us, sum of two -9%, G2 -1% |

Final A/B: anon-aadhaar MSM stage 669 to 532 ms, prove 976 to 840 ms; every circuit's
prove -3.4% to -13.9%.

Rejected: two chains without interleaving (121.4/118.1 to 122.5/122.6 ns), four entries
interleaved (107.3/109.5 to 116.7/113.8, spills), c = 13 and 15 (657 and 600 against 562
ms), eight chunks a window (595).
