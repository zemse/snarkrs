# The ceremony on Metal: snarkjs 0.7.6, g16 on twelve cores, g16 on the GPU

Measured, not quoted. This is the companion to [`ceremony-cpu.md`](ceremony-cpu.md), which
established that g16's CPU ceremony writes snarkjs' bytes. Six commands now take a
`--backend metal`, and the bar every one of them is held to is not "the proof verifies" and
not "snarkjs accepts it". It is that `--backend cpu` and `--backend metal` write
**byte-identical files**. Because the CPU path is already byte-identical to snarkjs 0.7.6, a
`cmp` between the two backends carries the snarkjs claim across. snarkjs' own verifiers were
run anyway, but the `cmp` is the check that catches a wrong point.

## The result

Median of 3 whole-process runs, seconds, on the largest production input each command has.

| command | input | snarkjs 0.7.6 | g16 cpu | g16 metal | metal vs cpu | metal vs snarkjs |
|---|---|---:|---:|---:|---:|---:|
| `ptau prepare` | power 20 | ~9,300 (extrapolated) | 963.85 | **142.62** | **6.76x** | ~65x |
| `zkey contribute` | 2^20, `--O1` | 72.57 | 16.35 | **1.80** | **9.08x** | 40.3x |
| `zkey beacon` | 2^20, `--O1` | 73.55 | 16.29 | **1.80** | **9.05x** | 40.9x |
| `ptau beacon` | power 19 | 161.97 | 36.98 | **5.57** | **6.64x** | 29.1x |
| `ptau contribute` | power 19 | 162.28 | 36.90 | **5.59** | **6.60x** | 29.0x |
| `setup`, circom `--O1` | 2^20 | 86.44 | **17.36** | 17.14 | 1.01x | 5.0x |
| `setup`, circom `--O2` | 2^18 | 329.39 | **52.40** | 137.14 | **0.38x** | 2.4x |

Five of the six win. `setup` is the sixth and it is bold in the CPU column on purpose: on
circom's default output the GPU is a wash, and on the dense `--O2` output it is a 2.6x loss.
Its `--backend cpu` stays the default and the reason is in its own section below.

Every one of those Metal outputs is byte-identical to the CPU one it is timed against.

## Method

| | |
|---|---|
| machine | Apple M2 Max, 12 CPU cores, 38 GPU cores, 96 GiB, macOS 26.6 (25G72) |
| g16 | `cargo build --release --features metal` at `693ac58`, one binary, sha256 `bf9b30f4…0beac7` |
| snarkjs | 0.7.6, node v25.9.0, `--max-old-space-size=49152` |
| circuits | the four production builds, circom 2.2.3 |
| phase 1 | `bench/ptau/local_13.ptau` and `local_19.ptau`, `ppot_0080_{17,18}.ptau`, `pot20_beacon.ptau` from the circuits repo, plus powers 8 to 16 built here by `g16 ptau new` |
| reps | 3 per cell (5 on one row, 20 on another, both said so below), medians reported |
| timing | `/usr/bin/time -p` around the whole process, so process start, the mmap, the file read and the write are all inside the window |

The binary was copied out of `target/release` and every run went through the copy, because
another lane rebuilt `target/release/g16` in the middle of the sweep. The only commit between
the two builds is `86024e3`, which floors `G16_METAL_FFT_BUDGET` at 1; that variable was never
set here, so the two binaries are the same program for every number below. The checksum was
sampled every two minutes for the rest of the run and did not move.

**Implementations alternate inside a rep, and their starting position rotates across reps.**
Alternating removes drift only if the order does not matter, and on this machine it does: a
`snarkjs groth16 setup` holds nine cores for five minutes, and whatever runs next runs on a
hot chip with a cold page cache. So rep 1 is snarkjs, cpu, metal; rep 2 is cpu, metal,
snarkjs; rep 3 is metal, snarkjs, cpu.

**Machine load, and it cost a whole pass.** This box is shared with three other agent
sessions. The first phase-2 pass ran back to back with no idle gaps, its one-minute load
average ran 4.2 to 47.5 on 12 cores (median 24.9), and a sibling lane's own power-20
`ptau prepare` held ten of the twelve cores through twenty minutes of it. That pass's
`--backend cpu` numbers landed 40% to 70% above the baseline `ceremony-cpu.md` published for
the same commands, so it was discarded and re-run.

Every timed run reported below does two things first: it waits until no other session's `g16`
or `snarkjs` is above 50% of a core, and it idles 20 to 30 seconds. Under that protocol the
load average ran 2.1 to 14.7 (median 5.8), most of it the benchmark itself, and the three
reps agree to within 2% on almost every row. The `--backend cpu` figures then land within a
few percent of `ceremony-cpu.md`: `setup` on `prod_small_O2` at 10.07 s against its
9.13, `zkey beacon` at 2^20 at 16.29 s against its 13.84.

**The snarkjs install changed hands mid-sweep.** The copy used for reps 1 and 2 is the one in
the circuits repo's own `node_modules`, which is the install that produced the shipped
artifacts. Another session deleted that directory during rep 2. Rep 3 runs a separate snarkjs
0.7.6 from the npx cache, and the byte comparison was repeated against it: its `groth16 setup`
and `zkey beacon` output is identical to g16's on all four builds, which is what makes the two
installs interchangeable in the table. One cell is a median of 2 rather than 3, `zkey beacon`
on the 2^20 build, because the deletion landed inside that run.

## Byte equality

The bar. Whole files, `cmp`, no tolerance and no section-by-section allowance.

| command | inputs | cpu = metal | cpu = snarkjs |
|---|---|:---:|:---:|
| `setup` | 4 production builds | 4 of 4 | 4 of 4 |
| `zkey beacon` | 4 production builds | 4 of 4 | 4 of 4 |
| `ptau prepare` | powers 13, 15, 16 | 3 of 3 | 3 of 3 |
| `ptau prepare` | powers 17, 18, 20 | 3 of 3 | not run |
| `ptau beacon` | powers 8, 10, 12, 13, 14, 16, 19 | 7 of 7 | 3 of 3 (13, 16, 19) |
| `zkey contribute`, `ptau contribute` | — | see below | see below |

The `cpu = snarkjs` column stops at power 16 for `prepare` for one reason only: snarkjs takes
436 s at power 16 and the cost roughly doubles per power, so the powers above it were left to
the extrapolation in that command's own section.

**Neither `contribute` is byte-comparable at the CLI, by design.** Both mix 64 OS-random
bytes into the seed, in g16 as in snarkjs. The equality is checked one level down instead, in
`crates/g16-ceremony/tests/contribute_metal.rs`, where `phase1::contribute_with` and
`contribute::contribute_with` take the RNG and both backends draw the same delta. It is the
same kernel `beacon` uses on the same two sections, so `beacon`'s CLI `cmp` covers the
arithmetic and the test covers the rest.

snarkjs' own verifiers, on files only the GPU produced:

- **`snarkjs powersoftau verify` on the Metal power-20 prepared file: Powers of Tau Ok!**, in
  116.41 s. This is the one that pairs the Lagrange sections 12 to 15 back against section 2
  rather than trusting anything g16 computed.
- `g16 ptau verify` on the same file: OK, 16.90 s. `g16 ptau info`: 11 of 11 sections, every
  one at its expected length, 1,207,962,589 bytes, `prepared true`.

One trap worth recording, because it makes a check look stronger than it is: run
`snarkjs powersoftau verify` on a file whose section 13 is short and it prints
`Powers of Tau Ok!` anyway, with a warning that the file has no phase-2 precalculated values.
It verified the phase-1 chain and skipped the sections `prepare` exists to write. A truncated
prepare output passes it. `cmp` does not, and `g16 ptau info` names the short section.

## `ptau prepare`, the command this backend exists for

962 s on the CPU at power 20 is the slowest thing in the project, and it is the reason the
group FFT got a kernel.

| power | input | snarkjs | g16 cpu | g16 metal | metal, best of 3 | cpu / metal |
|---:|---|---:|---:|---:|---:|---:|
| 13 | `local_13.ptau` | 44.04 | 5.11 | 2.52 | 2.51 | 2.04x |
| 15 | built here | 202.90 | 22.62 | 5.79 | 5.78 | 3.91x |
| 16 | built here | 436.05 | 48.08 | 12.67 | 9.84 | 4.89x |
| 17 | `ppot_0080_17.ptau` | — | 100.07 | 17.85 | 17.75 | 5.64x |
| 18 | `ppot_0080_18.ptau` | — | 212.74 | 34.17 | 34.01 | 6.26x |
| 20 | `pot20_beacon.ptau` | ~9,300 | **963.85** | **142.62** | 142.62 | **6.76x** |

Powers 15 and 16 are built here by `g16 ptau new` plus a beacon rather than taken from
`bench/ptau`, because every `ppot_0080_*` file is a truncate output and a truncated ptau turns
on a snarkjs check that cannot pass. That is snarkjs disagreeing with snarkjs, not with us,
and it would have cost the `cpu = snarkjs` column at those two powers for no reason.

Power 20 uses the circuits repo's own `pot20_beacon.ptau`, which is an untruncated two
contribution chain at exactly the required power. It is the same input `ceremony-cpu.md`
measured its 962 s on, and the three CPU runs here came out at 963.85, 972.08 and 963.61 s.

**The snarkjs power-20 figure is an extrapolation and is labelled as one.** Measured snarkjs
prepares are 44.04 s at power 13, 202.90 at 15 and 436.05 at 16, which is 2.15x per power over
that range, giving 9,300 s at power 20. The same extrapolation applied to g16's own CPU
measurements predicts 971 s where 963.85 was measured, 0.8% out, which is what makes the
snarkjs figure worth printing. `ceremony-cpu.md` records an actual snarkjs attempt at this
power that was abandoned after 43 minutes with the file half written, so 2.6 hours is
consistent with the only real datum there is.

### Why the speedup climbs with the power, and where it stops

The GPU's fixed costs are per command and per block, its arithmetic is per butterfly, so the
ratio rises until the largest blocks dominate. The shipped crossover sends every block under
2^12 points to the CPU (`MIN_BLOCK`, `crates/g16-metal/src/fft.rs:75`); at power 13 that is
eleven of the fifteen block sizes and at power 20 it is 0.18% of the work. Above power 18 the
curve flattens, and 6.76x at power 20 is where it lands.

Wall clock is no longer the host's fault: a power-20 `--backend metal` run spends 13 s of user
CPU against 143 s of wall. Packing points into the shared buffer, building a twiddle table per
block, and reading and writing 1.2 GB are together under a tenth of the run.

### The Metal group FFT loses command buffers, and it is not rare

Four `ptau prepare --backend metal` runs in this sweep died outright:

    backend `metal` failed during group ifft over G2: command buffer did not complete
    (status Error): Impacting Interactivity
    (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)

One at power 15, one at power 17 and **both scheduled power-20 runs**, out of 17 Metal prepare
runs in the timed sweep. All four are the G2 pass, never G1, which is what you would expect:
`Xyzz<Fq2>` is 256 bytes and G2's per-command-buffer budget is 2^17 ladders against G1's 2^19,
so a G2 submission is the longest thing the command sends. `FftKernels::dispatch_with_retry`
re-submits a killed pass four times with a 200, 400, 800 ms backoff, and all four attempts
failed each time.

The kills clustered in one 80-minute window. A Safari page with a live WebKit GPU process was
on the machine through it, which is exactly what the error name says the trigger is: macOS
kills a compute submission that is holding the GPU away from something interactive, and 1.4 s
of backoff does not outlast a browser tab. The window closed on its own. A dedicated sweep
run afterwards, **20 consecutive power-16 prepares, had no failures at all** and every output
was identical to the CPU reference, at a median of 10.39 s and a minimum of 10.23 s. The
power-20 headline above is from a re-run taken after the window closed, and it succeeded on
the first attempt.

Two things follow, and only one of them is a benchmark result. The measured one: a Metal
prepare that has to retry costs seconds, not a failure, and it is why the power-16 three-rep
median above (12.67 s) sits above the twenty-rep median (10.39 s). The other is a defect
report rather than a number: **`ptau prepare --backend metal` is not safe to leave unattended
on a machine anyone is using.** A 16-minute CPU run always finishes; a 2.4-minute GPU run
fails outright if a browser wants the GPU at the wrong moment, after having done most of the
work. The retry budget is the thing to change, not the kernel.

One datum on the knob that exists for this. `G16_METAL_FFT_BUDGET=8192`, sixteen times shorter
than G2's default submission, completed power 20 in **184.88 s** with byte-identical output.
That is 30% slower than the default 142.62 s. It is one run, so it says what the shorter
submission costs and nothing at all about whether it survives contention better.

## The four apply-key commands

`ptau contribute`, `ptau beacon`, `zkey contribute` and `zkey beacon` are the same primitive
twice: rescale a vector of points by a geometric series of the contributor's key. Phase 1 does
it to five ptau sections over both groups, phase 2 to zkey sections 8 and 9 over G1 only.

### Phase 1

| power | input | command | snarkjs | g16 cpu | g16 metal | snarkjs / cpu | cpu / metal |
|---:|---|---|---:|---:|---:|---:|---:|
| 13 | `local_13.ptau` | `contribute` | 2.74 | 0.57 | 0.13 | 4.8x | 4.4x |
| 13 | `local_13.ptau` | `beacon` | 2.73 | 0.58 | 0.13 | 4.7x | 4.5x |
| 16 | built here | `contribute` | 21.32 | 4.62 | 0.75 | 4.6x | 6.2x |
| 16 | built here | `beacon` | 20.47 | 4.60 | 0.74 | 4.5x | 6.2x |
| 19 | `local_19.ptau` | `contribute` | 162.28 | 36.90 | **5.59** | 4.4x | **6.6x** |
| 19 | `local_19.ptau` | `beacon` | 161.97 | 36.98 | **5.57** | 4.4x | **6.6x** |

### Phase 2

| build | domain | command | snarkjs | g16 cpu | g16 metal | snarkjs / cpu | cpu / metal |
|---|---:|---|---:|---:|---:|---:|---:|
| `prod_small_O2` | 2^16 | `contribute` | 5.08 | 1.06 | 0.16 | 4.8x | 6.6x |
| `prod_small_O2` | 2^16 | `beacon` | 5.22 | 1.08 | 0.16 | 4.8x | 6.8x |
| `prod_small_O1` | 2^17 | `contribute` | 10.78 | 2.47 | 0.30 | 4.4x | 8.2x |
| `prod_small_O1` | 2^17 | `beacon` | 11.41 | 2.49 | 0.29 | 4.6x | 8.6x |
| `prod_large_O2` | 2^18 | `contribute` | 22.81 | 5.10 | 0.60 | 4.5x | 8.5x |
| `prod_large_O2` | 2^18 | `beacon` | 23.48 | 5.12 | 0.60 | 4.6x | 8.5x |
| `prod_large_O1` | 2^20 | `contribute` | 72.57 | 16.35 | **1.80** | 4.4x | **9.1x** |
| `prod_large_O1` | 2^20 | `beacon` | 73.55 | 16.29 | **1.80** | 4.5x | **9.1x** |

PLACEHOLDER_DEEP

**Phase 2 wins harder than phase 1, and the split is structural.** `zkey contribute` is 95%
to 98% one `p * k` loop, so moving that loop moves the command. `ptau contribute` rescales
*more* points than the 2^20 zkey does and still reaches only 6.6x, because after the multiply
it compresses every point and folds it into a BLAKE2b response hash, and both of those stay on
the CPU and were never the target. The other half of the difference is the group mix: phase 2
is G1 only, phase 1 is G1 and G2, and G2's ladder is the slower of the two.

### The crossover

`MetalKeyScale` sends any call shorter than 256 points to the host
(`KEY_MIN_POINTS`, `crates/g16-metal/src/ceremony.rs:211`). It is a live path rather than a
guard: a power-8 `.ptau` has 511 points in its largest section and a `tiny_mul` zkey has 19.

PLACEHOLDER_SMALL

## `setup`, the one that loses

| build | domain | terms in slots >= 32 | snarkjs | g16 cpu | g16 metal | metal / cpu |
|---|---:|---:|---:|---:|---:|---:|
| `prod_small_O1` | 2^17 | 3.8% | 19.45 | 3.08 | 3.20 | 0.96x |
| `prod_large_O1` | 2^20 | 4.8% | 86.44 | 17.36 | 17.14 | 1.01x |
| `prod_small_O2` | 2^16 | 75.8% | 83.90 | 10.07 | 28.22 | **0.36x** |
| `prod_large_O2` | 2^18 | 74.2% | 329.39 | 52.40 | 137.14 | **0.38x** |

Both `--O2` builds are byte-identical on the two backends, including the G2 multiexp that puts
500 K and 2.5 M `Fq2` terms through the device, so this is a performance result and not a
correctness one.

`setup` reaches the device through `g16_msm::MsmBackend`, one `msm_g1` or `msm_g2` per
accumulator slot, and only for slots of 32 terms or more (`MULTIEXP_MIN_TERMS`,
`crates/g16-ceremony/src/setup.rs:93`). That threshold was chosen for a CPU. The mean slot
that reaches it holds 85 terms, and `prod_large_O2` makes 97,648 such calls.
At 0.149 ms for a bare Metal commit-and-wait, before any buffer allocation or base upload,
those calls alone are 14.5 s of round trip on a 52 s command; measured against the CPU's own
multiexp time the setup profile puts a real call at about 1.15 ms, which is 112 s. That is the
whole difference and the kernel is not at fault: the same run burns roughly half the user CPU
of the CPU run, so the arithmetic did move off the cores, onto a queue that cost more than it
saved.

The ceiling matters more than the loss. Delete the multiexp entirely, give it zero cost, and
circom's default `--O1` output gets **1.00x**: 16.60 s against 16.67 at 2^20. `MsmBackend`
covers 0.25% of the arithmetic on an `--O1` build and 45% on an `--O2` one, and circom 2.2.x
defaults to `--O1`. Batching the calls into one dispatch would address only the density nobody
compiles by default. The seam worth building is per-term rather than per-slot, N terms in and
N products out, which the setup lane costed at 3.4x to 4.9x on every density including the
default one.

Until that exists, `--backend cpu` is the default for `setup` and `--backend metal` stays
wired for the `cmp` it makes possible.

## The commands with no Metal path

Four of the ten ceremony commands have no `--backend` flag, and none of them is an oversight.

| command | why |
|---|---|
| `ptau new` | writes powers of the plain generators, so it performs no group operation at all. 0.01 s at every power in `ceremony-cpu.md`. |
| `ptau info` | walks the section table and never reads a point. 0.00 s by `time`. |
| `ptau verify`, `zkey verify` | pairings and hash chains, not scalar multiplication. `zkey verify --r1cs` does re-run a whole setup and so does touch the MSM, and the CLI pins that to `cpu` on purpose (`crates/g16-cli/src/main.rs:620`): a verifier that trusts the backend under test is not a verifier. |
| `zkey export-verificationkey` | reads five short sections and writes 6 to 39 KB of JSON, 8 to 10 ms on every build regardless of size. |

## Memory

PLACEHOLDER_MEM

## Against the arithmetic ceiling

`device-microbench.md` measures 4.14 G `Fq` multiplies per second on this GPU against 0.60 G
on twelve CPU threads, so the honest first guess for any of these commands is **6.9x**, and
nothing here should be believed if it lands far above that without a reason.

| command | measured | against 6.9x |
|---|---:|---|
| `ptau prepare`, power 20 | 6.76x | at it |
| `ptau contribute` / `beacon`, power 19 | 6.6x | at it |
| `zkey contribute` / `beacon`, 2^20 | 9.1x | above it, checked below |
| `setup`, `--O1` | 1.01x | far below it |
| `setup`, `--O2` | 0.38x | far below it |

**The two phase-1 commands and `prepare` land on the ceiling**, which is the result the
profile predicted and the one that needs no defending. What is left in `prepare` is host work
that was never a target (13 s of user CPU against 143 s of wall) and in `ptau contribute` it
is the point compression and the BLAKE2b response hash, which is where the next win there is.

**The phase-2 pair comes in above 6.9x, so it was checked three ways rather than reported.**
First, the divisor: the `--backend cpu` figure is 16.3 s from five reps on a quiet box, which
is *worse* than the 13.84 s `ceremony-cpu.md` published for the same command and input, so the
ratio is not inflated by a soft CPU baseline; using 13.84 gives 7.7x and the conclusion is the
same. Second, the work: the two outputs are byte-identical, so the GPU did the same
arithmetic on the same points and did not skip anything. Third, the ceiling itself is wrong
for a point ladder. The microbenchmark chains each multiply on the previous one and a point
formula's multiplies are independent, so the ladder gets instruction-level parallelism the
probe cannot; the ceremony kernels lane measured 7.0 G mul/s on G1 and 5.6 on G2. Against
G1's 7.0 the ceiling for a G1-only command is **11.7x**, and 9.1x sits under it with the file
read, the file write and the host pack in between.

That also explains the ordering in the table without any appeal to noise. `zkey contribute` is
G1 only and gets the 11.7x ceiling; `ptau contribute` and `prepare` move both groups and G2's
ladder is 20% slower, so their ceiling is lower and they sit right on the microbenchmark's
figure.

**`setup` is bounded by launch latency, not arithmetic**, and that is the whole of its story.
It is not I/O bound (decode is 0.2% of the command), not memory bound, and not short of
parallel work. It issues up to 1.85 M scalar multiplications of one to thirty terms each, and
`MsmBackend` cannot see any of them: they are below the multiexp threshold. The ones it can
see average 85 terms, which is 85 point operations behind a millisecond of queue.

## What a reader should take from this

Two of the six commands are worth running on the GPU without qualification. `zkey contribute`
and `zkey beacon` go from 16 s to 1.8 s on a 2^20 production key, which is 40x snarkjs and
9x our own CPU path, and they have no failure mode of their own because below 256 points the
work never reaches a kernel.

`ptau prepare` is the one that changes what is possible rather than what is convenient. Power
20 goes from a snarkjs job nobody finished, to 16 minutes on twelve cores, to 2.4 minutes on
the GPU. It is also the one command here that can fail: it lost four command buffers today,
both of its scheduled power-20 runs among them, and the four-attempt retry did not save any
of them. Run it on a machine nobody is looking at, check the exit status, and keep the CPU
path for the run that has to finish.

`ptau contribute` and `ptau beacon` are a solid 6.6x that is already at the arithmetic
ceiling; the next thing to attack there is the compression and the response hash, not the
kernel.

`setup` should stay on the CPU. Its GPU seam is the wrong seam, it addresses 4.8% of the work
on circuits compiled the way circom compiles them by default, and it makes the dense case 2.6x
worse. The flag stays because the byte comparison it enables is worth more than the timing it
loses.
