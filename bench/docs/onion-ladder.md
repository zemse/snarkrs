# poseidon onion ladder

- BN254 Groth16, circomlib `Poseidon(2)` (two inputs, not the Poseidon2 permutation).
- Private seed and one private salt per hash. Every chain link constrained, final digest public.
- Deterministic benchmark inputs and public contribution entropy. Never production keys.
- Preparation local only. No ptau downloads, phase2 preparation, full archive verification, or server provisioning.

## sizing

Measured with circom 2.2.3, snarkjs 0.7.2 and `--O1`:

| hashes | constraints | wires | public inputs | outputs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 517 | 520 | 0 | 1 |
| 8 | 4,136 | 4,146 | 0 | 1 |

`--O1` avoids the global O2 nonlinear substitution pass on a large chained circuit. It retains linear constraints. No O2 large-circuit experiment needed.

Domain = ceil_pow2(constraints + public inputs + outputs + 1). Target about 75% occupancy, safely above the previous power. These ladder counts are predictions from the measured pilots; each compiled header must match before continuing.

| fixture | hashes | expected constraints | expected wires | requested domain |
| --- | ---: | ---: | ---: | ---: |
| onion_p21 | 3,042 | 1,572,714 | 1,575,758 | 2^21 |
| onion_p22 | 6,084 | 3,145,428 | 3,151,514 | 2^22 |
| onion_p23 | 12,169 | 6,291,373 | 6,303,544 | 2^23 |
| onion_p24 | 24,338 | 12,582,746 | 12,607,086 | 2^24 |

## ptau access

Archive: `$G16_PPOT_ARCHIVE`, default `~/workspace/ppot-archive/ptau`.

- p21/p22 use `ppot_0080_21.ptau` and `ppot_0080_22.ptau`.
- p23/p24 use prepared `ppot_0080_final.ptau`, power 28, directly.
- Installed snarkjs `src/zkey_new.js` lines 45, 60, 145 to 152 use indexed section access and domain-sized ranges from sections 12 to 15. `writeHs` also reads a domain-sized range from section 12, not the whole archive.
- Setup reads roughly 448 bytes per domain element from prepared ptau sections, about 3.5 GiB at p23 and 7 GiB at p24, plus headers. Its constraint objects and point composition need additional RAM.
- Installed `src/powersoftau_truncate.js` generates every lower power in a loop. Do not call the truncate CLI on the 288 GiB archive: direct random access avoids needless copies.

Source reference: installed snarkjs 0.7.2 under `node_modules/snarkjs/src/`.

## run and resume

From the repo root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 bench/scripts/gen-onion-artifacts.py --smoke-only
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s bench/scripts -p test_onion_artifacts.py -v
nohup env PYTHONDONTWRITEBYTECODE=1 python3 bench/scripts/gen-onion-artifacts.py > target/onion-prep/runner.log 2>&1 < /dev/null &
echo $! > target/onion-prep/runner.pid
```

- Serial p21, p22, p23, p24. One runner protected by a local file lock.
- Node heap 40 GiB, four JS workers, four Rayon threads. Aggregate process-group RSS cap 64 GiB.
- Check RSS and host swap every three seconds. Stop if swap exceeds 2 GiB or grows by 1 GiB within a stage. Existing swap above 2 GiB blocks a new stage.
- Before larger stages, project measured lower-power peak RSS with 25% overhead. Pause if the projection exceeds the cap, including p24. Do not raise limits merely to force p24 through.
- `target/onion-prep/active.json` contains the latest child PID, log, RSS and elapsed time. `runner.log` records each command.
- `kill -TERM <runner-pid>` stops the active process group and records a failed/paused stage. Do not use SIGKILL for normal stopping.
- Resume with the same command. Completed stages require the exact output list, matching sizes/mtime and SHA256 content digests. Existing files alone, including `.part` files, never imply success. Invalidation removes downstream completion state and the shipping marker.
- Changed source, tool version, generator or ptau identity blocks reuse for inspection, rather than silently mixing artifacts.

Each fixture is in `bench/artifacts/large/onion_pNN`. `metadata.json` records versions, hash count, ptau, actual R1CS counts/domain, stage logs, timings, sampled memory peaks and completion state.

Stages: compile, R1CS info, witness generation, mandatory snarkjs witness check, snarkjs setup, explicit benchmark contribution, vkey export, native CPU reference proof, independent snarkjs verify. The exported vkey must have gamma != delta. `ready: true` is written only after all stages succeed. The small p13 smoke fixture completed this entire pipeline locally; that does not imply any large fixture is ready.

## existing fixtures

Older metadata has sizes/mtime only and fails the content-aware readiness check. Run `bench/scripts/revalidate_onion.py` with the fixture directories, serially. It regenerates the witness and requires identical bytes, checks R1CS constraints, re-exports the vkey from the zkey, generates a fresh CPU proof and independently verifies both fresh and existing proofs. Inputs must remain unchanged across validation. It preserves the original metadata in `revalidation/metadata-before.json`, records content digests and generates shipping manifests only after success. Failed/interrupted revalidation leaves `ready: false`; the same command retries it. Original generator/version provenance is retained, not silently rewritten for generator reuse.

## shipping

Only ship a fixture after `bench/scripts/check_onion_ready.py` succeeds, not merely because `metadata.json` says `ready: true`:

- `circuit.zkey`
- `circuit.wtns`
- `vkey.json`
- `public.json`
- `r1cs-info.txt`
- `ship-metadata.json`

`--manifest` writes the recorded certified digests, not fresh hashes of unverified bytes. The safe runner checks these manifests on the guest before proving. Keep R1CS, generated WASM/JS, input, genesis key, reference proof and detailed logs local. On the NVIDIA server, prove from the prebuilt key/witness and independently verify against this vkey/public digest before accepting benchmark timings. Preparation scripts never launch infrastructure. The coordinator handles approved rental and transfer.
