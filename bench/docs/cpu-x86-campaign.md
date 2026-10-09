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

## rental and isolation

Prepare builds and runtime-only bundles before starting the rental. Do not compile on the benchmark host. Inspect competitor clones as source under `~/workspace/opensource/external`; execute their build scripts only in disposable isolation.

An approved independent AWS stop deadline must be armed before starting an existing stopped benchmark instance. Retain the deadline after early completion, retrieve results and stop immediately. Scheduler delivery is a backstop, not a guaranteed real-time stop. Do not terminate instances or delete retained storage as part of this campaign.

## results to retain

For every experiment: source revision, input hashes, exact commands, build flags, thread count, verified raw samples, stage timings and peak memory. Record successful changes and rejected hypotheses with their measurements. Keep the remaining optimisation gaps separate from merged changes.
