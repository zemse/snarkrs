# CPU runtime campaign

`cpu-campaign.py` prepares and transfers a runtime whitelist, then runs one serial
CPU campaign on an already-running Linux x86_64 host. It never invokes AWS, builds,
installs dependencies, installs watchdogs or changes an instance lifecycle.

## local preparation

Supply a locally certified runtime directory containing `bin/snarkrs-generic`,
`bin/snarkrs-adx`, `bin/rapidsnark`, `bin/rapidsnark-warm`, `bin/rapidsnark-verify`
and `certification.json`. Optional `lib/` contains only the required shared runtime
libraries, with no compiler or vendor trees. Validate the Linux runtime closure in
an isolated environment before transferring it. Packing checks ELF architecture,
identity and supplied certification metadata, not whether those tests really ran.

The certification object has a `binaries` entry for each filename. Each entry needs
`sha256`, full 40-character `source` revision, `compiler`, exact `flags` and
`verification` (commands, evidence paths and results of local correctness tests).
Generic flags must be the empty string. ADX flags must be exactly
`-C target-feature=+adx,+bmi2`. Both snarkrs builds must have the same source revision.
Record target triple and linker details in `compiler`; record all competitor flags
in its `flags`. The verifier and warm wrapper also need pinned source identities.
For optional shared libraries, add `libraries` mapping each filename to its SHA256.
Do not claim native CPU tuning or gnark performance from these labels.

The pack command copies only these runtimes, the driver, existing comparison and
oracle helpers, and six files per fixed fixture: zkey, witness, vkey, reference
public signals, input JSON and constraint metadata. Every copied file is hashed.
A corrupt transfer or changed binary fails before proving.

```bash
python3 bench/aws/cpu-campaign.py pack \
  --runtime target/cpu-x86/runtime \
  --artifacts bench/artifacts \
  --bundle target/cpu-x86/bundle
```

Packing refuses an existing bundle. Inspect bundle size and available root disk
space before deploying, especially on the target's 8 GiB root volume.

## coordinator gates and transfer

The coordinator alone arms the independent AWS deadline, starts the stopped target,
validates use through the native capability and installs the existing tested
`idle-watchdog.sh` timer. This driver requires an active watchdog state directory
with writable `lease` and directory, a `config` file and no stale `done` marker.
It renews that existing lease every 30 seconds and marks `done` after success or
failure. State-file presence does not attest timer installation or firing; the
coordinator verifies that separately. It does not install another shutdown mechanism
or cancel any deadline.

Deploy only after those gates. Replace `HOST` and `KEY` with coordinator-approved
values. The SSH host key must already be trusted. The remote parent must exist;
the remote directory itself must not exist. Deploy only transfers files, it does
not start the campaign. Retrieve uses a new local directory and retains partial
results even when the guest command failed.

```bash
python3 bench/aws/cpu-campaign.py deploy \
  --host HOST --key KEY --remote /home/ubuntu/cpu-x86 \
  --local target/cpu-x86/bundle
```

## guest invocation

The required warm interface is `rapidsnark-warm ZKEY WTNS FINAL_PROOF FINAL_PUBLIC
REPS VKEY --proof-dir DIR`. It must self-verify and export `proof-{rep}.json` and
`public-{rep}.json` for every repetition outside proving time, emit fractional
millisecond samples, and report `ffiasm_pool_threads=16` on stderr from the actual
pool. Missing diagnostics or per-rep outputs fail closed. Commit 88fd115 alone lacks
these exports and diagnostics; certify the worker's completed follow-up in disposable
isolation before use. No competitor execution on the Mac is part of this driver.

Coordinator-approved guest command, after confirming wrapper support:

```bash
python3 /home/ubuntu/cpu-x86/bench/aws/cpu-campaign.py guest \
  --root /home/ubuntu/cpu-x86 --cold-reps 3 --warm-reps 15 \
  --budget-seconds 3600 --max-load 1.0
```

Default plan: js_2x2_d32 and js_16x16_d32, snarkrs requested Rayon budgets 1/4/8/16,
generic then ADX. The driver asserts 16 logical host CPUs. Unmodified ffiasm uses
`hardware_concurrency`, not `OMP_NUM_THREADS`; rapidsnark runs only native-16, once
per fixture/mode, never labelled as a matched smaller-thread competitor. Warm actual
pool size is gated on its diagnostic. Cold pool size is source-inferred, explicitly
not an actual cold measurement. Requested/environment counts are separate fields;
snarkrs actual worker count and CPU affinity remain unmeasured.

The one-minute load ceiling is recorded before each configuration. The driver waits
for its own previous load to decay, within the overall budget; it never uses
`--allow-loaded`. This is an initial-load gate, not continuous external-contention
detection. Idle waits consume budget. Budget expiry kills the command process group,
retains logs and rejects an incomplete campaign, without lowering repetitions.
Choose the budget explicitly inside the coordinator's remaining deadline, leaving
retrieval and shutdown margin. No automatic rental cost or deadline inference occurs.

## acceptance and limits

Each process gets raw stdout, stderr, GNU time peak RSS and exit status. GNU time
reports the command's peak child RSS, not aggregate physical footprint. CPU topology,
flags, OS release, source/build certification and all runtime/artifact/input hashes
are retained under `results/`. Temporary harness runtime layouts stay outside that
results tree, so retrieval does not follow their binary/artifact symlinks. No remote
compilation is present.

Snarkrs uses the existing comparison harness with `--warm-oracle`. Cold output pairs
are independently checked and compared to reference public signals every repetition.
Warm rows carry actual stage timings and preparation time, self-verification of every
repetition and independent/reference checking of the first exported proof. They are
explicitly labelled first-proof, never independently-each-proof. The harness does not
export later warm proofs. Cold process rows have no stage split; blank is not zero.

Cold snarkrs CLI timing includes its default self-verification; rapidsnark CLI timing
does not. Warm proving excludes verification on both sides; rapidsnark additionally
reparses its mapped witness inside the timed call. Report these scope differences,
not an unqualified cold proving speed ratio. Rapidsnark cold wall time also includes
GNU time and driver log/metadata overhead, explicitly labelled in its records.
This driver does not change CLI policy.

Every rapidsnark cold and warm proof/public pair must pass our `snarkrs-generic
groth16 verify VKEY PUBLIC PROOF` outside timing, plus reference-public checks.
Rapidsnark's own verifier is self-verification, not its independent oracle. Warm
acceptance requires zero wrapper exit, exact export/count/sequence and positive
finite fractional samples for the whole run; failed or partial warm runs are rejected.
All saved snarkrs rows need matching configuration, positive finite timings,
`verified=yes`, `loaded=no`, known verification scope and finite warm stage fields.
A nonzero process exit or validation failure prevents its acceptance record.

Snarkjs encoding compatibility is `unknown`; the guest has no snarkjs cross-check.
Gnark comparison is `not measured`. An accepted configuration is not a completed
campaign. Require `complete.json` and inspect all 36 configurations (32 snarkrs,
4 native-16 rapidsnark) before reporting the campaign. Failures retain prior accepted records,
raw files and `failure.json`; nothing deletes or overwrites the run directory.

```bash
python3 bench/aws/cpu-campaign.py retrieve \
  --host HOST --key KEY --remote /home/ubuntu/cpu-x86 \
  --local target/cpu-x86/retrieved
```

After retrieval, the coordinator stops the instance through its approved lifecycle
path. Keep the independent stop deadline armed.
