# The cross-machine sweep

Provisions a matrix of EC2 machines, benchmarks the prover on each, and terminates every
one of them. The point is not raw speed: it is **cost per proof**, which is a different
question with a different answer.

## Running it

```sh
./stage.sh                      # source + prover inputs to S3, once
./sweep.sh 10 4                 # 10 reps, 4 boxes at a time, both waves, then teardown
./bench-machine.sh g6.xlarge 10 # or one machine at a time
./bench-local.sh 10             # this Mac, into the same schema
./terminate.sh                  # the sweep already does this; run it again to prove it
python3 ../scripts/cost.py      # -> ../results/COST.md
```

## Prebuilt NVIDIA runs with automatic stop

```sh
export G16_AWS_STOP_ROLE_ARN=arn:aws:iam::123456789012:role/g16-benchmark-stop
bash bench/aws/run-gpu-bench-safe.sh 10
G16_IDLE_MINUTES=120 G16_MAX_MINUTES=240 bash bench/aws/run-gpu-bench-safe.sh 10
```

Run these from the repository root. The safe runner builds locally with the existing
cargo-zigbuild toolchain before launching a g4dn.xlarge. It uses the regional agent SSH
key and agent-ssh security group, which must already exist and admit your current IP.
The GPU AMI supplies the NVIDIA driver; no Rust compiler is installed on the instance.
Override the type with G16_AWS_TYPE (an x86_64 GPU entry in machines.csv).
An executable x86_64 Linux rapidsnark verifier is required before launch; set G16_VERIFY_BIN
when bench/bin/rapidsnark-verify is a macOS binary. The runner ships this independent oracle
and verifies the CUDA warmup before benchmarking. Add variant names after the repetition
count to limit artifact transfers, for example `1 tiny_mul js_1x1_d8 sha256` for a smoke run.
Set G16_ARTIFACT_ROOT to select a different local artifact directory, such as
bench/artifacts/large. Selected variants are checked for missing runtime files before launch.
Only runtime scripts, proving keys, witnesses, verification keys, public signals and metadata
are uploaded; compiler worktrees, R1CS files and witness-generator builds stay local.
Onion fixtures require complete stage-output lists and recorded SHA256 content digests.
Older size/mtime-only fixtures must pass fresh local revalidation before manifest generation.
Guest checksums are compared against those certified digests before any large proof runs.
Without variant names it transfers the top-level collection, excluding large/ and csp/.

Cold proofs are independently verified with rapidsnark. The first warm proof of each
variant/backend is also independently verified through rapidsnark-oracle, using the CLI's
foreign-verifier interface. Every warm proof still passes our own verifier. CSV rows record
that independent warm verification covers the first proof, not every repetition. This adapter
is not snarkjs, so its acceptance does not establish snarkjs encoding compatibility.
Incomplete configurations make the comparison runner exit unsuccessfully while retaining
partial results.

The ordered onion_p21 through onion_p24 ladder uses run-onion-ladder.py. The positional
repetition count selects cold repetitions; warm repetitions are five. Each domain first gets
one CUDA proof checked by both verifiers before any repetitions. Host VmHWM, available RAM
and sampled device memory are recorded. Less than 4 GiB available RAM or more than 90%
device memory aborts the first-proof gate. Repeated configurations fail fast and retain
completed CSV rows. The budget check assumes the default 180-minute hard cap and reserves
ten minutes for retrieval; use that cap for this ladder. Its eightfold CPU allowance is an
estimate, not a measured CPU runtime. A budget-risk rejection does not silently reduce counts.

The opt-in test_aws_shutdown_live.py --execute checks a real Scheduler stop and installed
guest timers on a small instance. --execute-guest runs only the guest checks on a new small
instance. Test timers use runtime-only activation so subsequent test boots cannot inherit an
expired short lease. Both modes stop and preserve the instance and EBS; neither terminates.
These live tests are not executed by npm test.

The runner downloads results into bench/results/aws-INSTANCE and stops the instance on
success, failure or a handled interrupt. It confirms the stopped state rather than merely
submitting a stop request. If the graceful-stop waiter times out, it requests a forced stop
and confirms the state again. Forced shutdown can leave the retained guest filesystem in
need of repair. Logs include the instance ID, region and source commit.

The instance also checks a job lease every minute. The laptop renews it during transfers
and execution; one hour without renewal stops the machine. This is orchestration idle,
not CPU/GPU utilization, so a quiet stage is not mistaken for inactivity. A finished job
has a five-minute results-recovery grace period even if the laptop disconnects. A separate
three-hour hard cap stops a hung job despite continuing heartbeats. Both limits are
configurable before launch; checks can run up to one minute after their threshold.

Stopped instances retain EBS data and storage charges. Termination is a separate explicit
cleanup decision after results are recovered. The guest shutdown policy is stop, not
terminate. Guest timers cannot guarantee shutdown if the OS freezes or bootstrap never
executes; local cleanup cannot survive SIGKILL or a lost laptop. The runner now requires an
independent EventBridge Scheduler stop deadline. Caller identity and schedule-group access
are checked before building. Immediately after launch it creates and reads back a one-time
UTC forced-stop target before waiting for the instance or transferring artifacts. The target
remains armed after local exits and auto-deletes after execution. Do not restart a stopped
instance while that deadline remains armed. Schedule timing has one-minute precision.

Set G16_AWS_STOP_ROLE_ARN to the administrator-provisioned execution role. Optional
G16_AWS_SCHEDULER_GROUP defaults to default. See [deadline-iam.md](deadline-iam.md) for
narrow caller permissions, role trust and required real firing tests. Missing configuration
or preflight failure prevents launch; arming/read-back failure requests immediate cleanup.
A background laptop watchdog also retries transient AWS failures and escalates stalled
stops after 180 seconds. Its logs and schedule records are saved beside the instance results.
There is still a launch-to-schedule creation window, and AWS control-plane outages remain
outside these guarantees. If a stop cannot be confirmed, the runner fails and prints the
instance ID for manual recovery.
Do not use the old run-gpu-bench-prebuilt.sh for unattended launches.

AWS behavior: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html.

## Nothing survives the sweep

A GPU box left running over a weekend costs more than this entire exercise, so there are
three independent guarantees and no single point of failure:

1. **Per-lane `trap ... EXIT INT TERM`** in `bench-machine.sh`. The launch and the teardown
   are in one process, so any failure between them still terminates. A provision script and
   a separate teardown script would leak an instance every time the middle step died.
2. **A dead-man switch inside each instance.** user-data runs `shutdown -h +N` and the
   instance is launched with `--instance-initiated-shutdown-behavior terminate`, so the
   poweroff terminates rather than stopping. If the driving session is killed, the box kills
   itself.
3. **`sweep.sh`'s unconditional teardown**, which runs even when every lane failed, and
   `terminate.sh`, which finds instances by tag when the instance id has been lost.

The `Owner` tag is load bearing. The IAM policy grants `RunInstances` only when the request
carries `Owner=${aws:username}`, and `TerminateInstances` only on instances already tagged
that way. Launching without it strands a billing instance you are not allowed to kill.

## Where the numbers in `machines.csv` come from

Specs are `ec2:DescribeInstanceTypes`. Prices are AWS's own published on-demand list for
us-east-1 Linux, the same feed the EC2 pricing page reads, fetched from
`b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/`,
published at the time of the sweep. The Pricing API (`pricing:GetProducts`) is denied to this IAM user, so
that static feed is the authoritative source here rather than anything remembered.

`spot-use1.csv` is `ec2:DescribeSpotPriceHistory` sampled at the time of the sweep. **Spot prices move**;
re-sample before quoting them. They are carried because nobody runs a batch prover
on-demand and because the discount is not uniform: 60-66% on these CPU families, 41% on
g4dn, and 22% on the Ada GPUs, which is large enough to reorder the ranking.

## Three things this harness learned the hard way

**Do not use the EC2 regional apt mirror.** Measured from a c7a.xlarge in us-east-1:
`us-east-1.ec2.archive.ubuntu.com` served `jammy/universe/Packages.gz` at 486 kB/s with
repeated timeouts, while `archive.ubuntu.com` served the same object from the same instance
in 69 ms. That was ten minutes of billed boot before anything was built. user-data rewrites
the mirror before the first `apt-get` call.

**Ubuntu upgrades itself mid-run.** `apt-daily-upgrade.timer` starts unattended-upgrade at a
random time of day, and on 2026-10-27 it tripped `run-benchmark.sh`'s contention guard in
the middle of a sweep. Every launcher's user-data embeds `quiet-apt.sh`, which stops and
masks both apt timers, stops a running upgrade and waits for the dpkg lock before anything
else runs. user-data runs on the first boot only, so a box that was set up some other way
gets it once by hand: `ssh HOST 'sudo bash -s' < bench/aws/quiet-apt.sh`.

**The first CUDA run on a fresh box is not a proof, it is a compile.** `~/.nv/ComputeCache`
is empty on a new instance, so the first `--backend cuda` invocation pays NVRTC
source-to-PTX and then the driver's PTX-to-SASS JIT: 113 s + 175 s on a T4. That is a real
per-machine deployment cost and it is recorded in each box's `meta.json` as
`first_compile_s`, deliberately outside every timed region. A benchmark that leaves it in
rep 1 reports a two-minute outlier; one that runs on a machine with a warm `~/.nv` and calls
the result "cold" is measuring a cache hit.
