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

## Two things this harness learned the hard way

**Do not use the EC2 regional apt mirror.** Measured from a c7a.xlarge in us-east-1:
`us-east-1.ec2.archive.ubuntu.com` served `jammy/universe/Packages.gz` at 486 kB/s with
repeated timeouts, while `archive.ubuntu.com` served the same object from the same instance
in 69 ms. That was ten minutes of billed boot before anything was built. user-data rewrites
the mirror before the first `apt-get` call.

**The first CUDA run on a fresh box is not a proof, it is a compile.** `~/.nv/ComputeCache`
is empty on a new instance, so the first `--backend cuda` invocation pays NVRTC
source-to-PTX and then the driver's PTX-to-SASS JIT: 113 s + 175 s on a T4. That is a real
per-machine deployment cost and it is recorded in each box's `meta.json` as
`first_compile_s`, deliberately outside every timed region. A benchmark that leaves it in
rep 1 reports a two-minute outlier; one that runs on a machine with a warm `~/.nv` and calls
the result "cold" is measuring a cache hit.
