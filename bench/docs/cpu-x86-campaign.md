# x86 CPU optimisation campaign

Target: c7i.4xlarge, x86 Ubuntu, 16 vCPU, 32 GiB RAM. Compare snarkrs with rapidsnark and investigate applicable gnark techniques. No new performance result has been accepted yet.

## comparison gates

- Pin prover commits, compiler versions, build flags and artifact hashes.
- Use BN254, fixed fixtures at two or more sizes, and the same key and witness wherever formats permit.
- For gnark, document any circuit or setup differences. Equivalent constraint counts alone do not establish equivalent work.
- Measure key preparation, cold process time and warm proving separately. Verification stays outside proving time but gates every accepted result.
- Check foreign verifier exit status as well as its success marker. Use an independent verifier at the start and end of each experiment.
- Record actual worker counts, CPU topology, affinity, peak memory and machine load. Run one benchmark process at a time.
- Compare generic x86 and explicit ADX/BMI2 builds before attributing differences to MSM or FFT algorithms. The workspace manifest documents the earlier assembly-path gap.
- Use 15 to 25 warm repetitions where the runtime budget permits. Record median and spread, not only the fastest sample.

## investigation order

1. Map preparation, witness handling, polynomial evaluation, FFT, MSM and proof assembly to source functions. Record operation counts, bytes moved and parallel barriers.
2. Inspect pinned rapidsnark and gnark sources. Separate confirmed implementations from untested explanations for their performance.
3. Measure field arithmetic, curve operations, memory bandwidth at relevant working-set sizes, and thread-pool overhead on the benchmark host.
4. Measure each pipeline segment at two input sizes. Derive computation floors from measured primitives and state the assumptions.
5. Rank segments by absolute wall-time headroom. Test one hypothesis at a time in a separate worktree.
6. Re-measure each worker's baseline, verify correctness, and independently reproduce the improvement before merging. Sweep thread counts and fixture sizes to reject overfitting.

Candidate investigations, not accepted optimisations: assembly activation, field reduction and squaring, G1/G2 MSM windows and scalar recoding, endomorphisms, batch inversion, FFT layout and twiddle reuse, allocation reuse, key decoding and scheduling.

## initial source audit

The CPU backend already uses CSR gather, zero/one witness fast paths, cached and blocked NTTs, permutation cancellation, signed Pippenger, batch-affine wide-window fill, XYZZ narrow-window fill, and concurrent MSMs. Do not count rediscovering these as new optimisations.

The custom MSM field kernel in `crates/field/src/raw.rs::mont_mul` is separate from ark-ff's assembly backend. ADX/BMI2 flags alone do not establish that this kernel becomes assembly-optimised. Inspect and measure Fr, RawFq and RawFq2 separately.

Initial fixed comparison fixtures: `js_2x2_d32` (17,929 constraints) and `js_16x16_d32` (140,261 constraints). Extend to larger and structurally different fixtures before selecting defaults. Existing timings on c7a and Apple silicon are prior evidence, not c7i measurements.

Direct `snarkrs bench` self-verifies every repetition, but its independent verifier covers the first proof per fixture/mode. The comparison harness checks reference public outputs and verifier status; the existing rapidsnark warm loop exposes only its final surviving proof. Strengthen that path before claiming every warm iteration is independently verified.

## competitor sources

Source-only inspections, pinned before the campaign:

- [rapidsnark](https://github.com/iden3/rapidsnark/tree/81eddf1a536d26497b237c0b8a04fe90baf7e439), `81eddf1a536d26497b237c0b8a04fe90baf7e439`.
- [ffiasm](https://github.com/iden3/ffiasm/tree/aa90166dc4c5a075b835a398e15cc1e06ac90e95), rapidsnark submodule `aa90166dc4c5a075b835a398e15cc1e06ac90e95`.
- [gnark](https://github.com/Consensys/gnark/tree/3ac4de3e14374de08e9966c329251f9365506372), `3ac4de3e14374de08e9966c329251f9365506372`.
- [gnark-crypto](https://github.com/Consensys/gnark-crypto/tree/8911ee45d22bdf9d1c6ab4affe2d4f7ffaedd557), `8911ee45d22bdf9d1c6ab4affe2d4f7ffaedd557`. A gnark executable must instead use its `go.mod` pin, `1409fe59205234d1fc62769923623ecba82e1ef7`, unless the dependency change is explicitly part of an experiment.

rapidsnark's generated Fq/Fr assembly and gnark-crypto's four-word field assembly use MULX/ADCX/ADOX. Their MSM paths use signed windows; gnark-crypto also selects batch-affine buckets by occupancy. These are source findings, not measured explanations for a c7i performance gap.

Bulk gnark-crypto MultiExp does not use the GLV decomposition found in individual scalar multiplication. Do not claim GLV is required for parity without a measured experiment. Compare H transforms together with the key's H basis representation, not by transform count alone.

Gnark's native Prove includes witness solving, while rapidsnark accepts solved witnesses. Its native keys are not Circom zkeys. Separate solver time and disclose query/domain differences. rapidsnark and ffiasm are LGPLv3; gnark and gnark-crypto are Apache-2.0. Review license obligations before copying code.

## rental and isolation

Prepare builds and runtime-only bundles before starting the rental. Do not compile on the benchmark host. Inspect competitor clones as source under `~/workspace/opensource/external`; execute their build scripts only in disposable isolation.

An approved independent AWS stop deadline must be armed before starting an existing stopped benchmark instance. Retain the deadline after early completion, retrieve results and stop immediately. Scheduler delivery is a backstop, not a guaranteed real-time stop. Do not terminate instances or delete retained storage as part of this campaign.

## results to retain

For every experiment: source revision, input hashes, exact commands, build flags, thread count, verified raw samples, stage timings and peak memory. Record successful changes and rejected hypotheses with their measurements. Keep the remaining optimisation gaps separate from merged changes.
