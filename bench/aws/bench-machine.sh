#!/usr/bin/env bash
# Benchmark one instance type end to end and terminate it, whatever happens.
#
#   bench-machine.sh <instance-type> [reps]
#
# The whole lifecycle is in one process with one EXIT trap, because the alternative -- a
# provision step and a separate teardown step -- leaks a GPU instance every time the middle
# step dies. Three independent things have to fail before this bills overnight:
#
#   the trap below, the dead-man switch inside the instance, and terminate.sh's sweep.
#
# What runs on the box, in order, and why:
#
#   build          release binary, plus test binaries in the same cargo invocation so the
#                  compile work is shared with the gate below.
#   correctness    `cargo test --release`. This is not ceremony. The sweep spans two
#                  instruction sets and four NVIDIA architectures, and the failure mode
#                  that matters is a machine that proves *fast* and *wrong*. A timing from
#                  a box that did not pass its own known-answer vectors is not a result.
#   first compile  GPU only, and only when asked: NVRTC plus the driver's PTX-to-SASS JIT
#                  with both caches cleared. It is minutes, it is a real deployment cost,
#                  and it is not part of any per-proof number, so it is measured once and
#                  reported separately.
#   warm           one throwaway proof so the kernel cache is not inside rep 1.
#   bench          cold and warm, every proof verified by g16 itself before it is recorded.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TYPE="${1:?usage: bench-machine.sh <instance-type> [reps]}"
REPS="${2:-10}"
HOURS="${G16_AWS_HOURS:-3}"
TESTS="${G16_TESTS:-full}"          # full | none
FIRST_COMPILE="${G16_FIRST_COMPILE:-0}"
VARIANTS="${G16_VARIANTS:-}"        # empty = every variant present

ARCH="$(machine_field "$TYPE" arch)"
GPU="$(machine_field "$TYPE" gpu)"
PRICE="$(machine_field "$TYPE" usd_per_hour)"
VCPU="$(machine_field "$TYPE" vcpu)"
OUT="$REPO_ROOT/bench/results/sweep/$TYPE"
mkdir -p "$OUT"
LOG="$OUT/run.log"
: > "$LOG"

say() { printf '[%s] %s\n' "$TYPE" "$*" | tee -a "$LOG"; }

INSTANCE=""
T_LAUNCH=0
cleanup() {
  local code=$?
  if [ -n "$INSTANCE" ]; then
    say "terminating $INSTANCE"
    aws_ ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null 2>&1 \
      && aws_ ec2 wait instance-terminated --instance-ids "$INSTANCE" >/dev/null 2>&1 \
      && say "confirmed terminated $INSTANCE" \
      || say "WARNING: could not confirm termination of $INSTANCE -- run bench/aws/terminate.sh"
    local mins=$(( ( $(date +%s) - T_LAUNCH + 59 ) / 60 ))
    say "billed about ${mins} min at \$${PRICE}/hr = \$$(python3 -c "print(f'{$mins/60*$PRICE:.2f}')")"
  fi
  say "exit $code"
  exit $code
}
trap cleanup EXIT INT TERM

say "arch=$ARCH vcpu=$VCPU gpu=${GPU:-none} price=\$$PRICE/hr reps=$REPS"

AMI="$(resolve_ami "$TYPE")"
SG="$(ensure_key_and_sg)"
USER_NAME="$(iam_user)"
say "ami=$AMI sg=$SG"

# Bootstrap runs while we wait for ssh, so its apt work overlaps the boot. The dead-man
# switch is the first line: if everything below this point dies, the box still stops billing.
UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$((HOURS * 60))
exec > /var/log/g16-bootstrap.log 2>&1
set -x
# Do not use the EC2 regional apt mirror. Measured from a c7a.xlarge in us-east-1 during
# this sweep: us-east-1.ec2.archive.ubuntu.com served jammy/universe Packages.gz at
# 486 kB/s and timed out repeatedly, while archive.ubuntu.com served the same object from
# the same instance in 69 ms. That cost ten minutes of billed boot before anything was
# built, on a box that only needs one package.
sed -i 's|http://[a-z0-9-]*\\.ec2\\.archive\\.ubuntu\\.com|http://archive.ubuntu.com|g' \
  /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true
# The Deep Learning AMIs already carry a toolchain and CUDA; only the plain Ubuntu images
# need apt at all, and then only for a C compiler to link Rust against.
if ! command -v cc >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=20 update
  apt-get install -y build-essential
fi
sudo -u ubuntu bash -lc 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal'
touch /home/ubuntu/BOOTSTRAP_DONE
EOF

T_LAUNCH=$(date +%s)
launch_once() {
  aws_ ec2 run-instances \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${G16_AWS_VOL_GB:-100},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Owner,Value=$USER_NAME},{Key=Name,Value=g16-$TYPE},{Key=purpose,Value=$PURPOSE},{Key=autoterminate,Value=${HOURS}h}]" \
  --query 'Instances[0].InstanceId' --output text
}
# VcpuLimitExceeded and InsufficientInstanceCapacity are both transient in a sweep: the
# first clears when a sibling lane finishes, the second when the AZ frees a card. Losing a
# machine from the matrix because of either would leave a hole in the comparison, so retry.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if INSTANCE="$(launch_once 2>"$OUT/launch.err")"; then break; fi
  err="$(tr -d '\n' < "$OUT/launch.err" | cut -c1-160)"
  case "$err" in
    *VcpuLimitExceeded*|*InsufficientInstanceCapacity*|*RequestLimitExceeded*|*Unavailable*)
      say "launch attempt $attempt bounced ($err); retrying in 90s"; INSTANCE=""; sleep 90;;
    *) say "launch failed, not retryable: $err"; rm -f "$UD"; exit 1;;
  esac
done
rm -f "$UD"
[ -n "$INSTANCE" ] || { say "could not launch after 10 attempts"; exit 1; }
say "launched $INSTANCE"

aws_ ec2 wait instance-running --instance-ids "$INSTANCE"
IP="$(aws_ ec2 describe-instances --instance-ids "$INSTANCE" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
say "running at $IP"

say "waiting for ssh"
for i in $(seq 1 60); do rsh "$IP" true 2>/dev/null && break; sleep 10; done
rsh "$IP" true || { say "ssh never came up"; exit 1; }

say "waiting for bootstrap (apt + rustup)"
for i in $(seq 1 90); do rsh "$IP" 'test -f ~/BOOTSTRAP_DONE' 2>/dev/null && break; sleep 10; done
rsh "$IP" 'test -f ~/BOOTSTRAP_DONE' || { say "bootstrap never finished"; rsh "$IP" 'sudo tail -40 /var/log/g16-bootstrap.log' | tee -a "$LOG"; exit 1; }
say "bootstrap done after $(( ($(date +%s) - T_LAUNCH) / 60 )) min"

SRC_URL="$(presign src.tgz)"
ART_URL="$(presign artifacts.tgz)"
say "fetching source and artifacts from s3"
rsh "$IP" "set -e; mkdir -p ~/g16/bench/artifacts
  curl -sS -o /tmp/src.tgz '$SRC_URL'; tar xzf /tmp/src.tgz -C ~/g16
  curl -sS -o /tmp/art.tgz '$ART_URL'; tar xzf /tmp/art.tgz -C ~/g16/bench/artifacts
  du -sh ~/g16" 2>&1 | tee -a "$LOG"

FEATURES=""; BACKENDS="cpu"
if [ -n "$GPU" ]; then FEATURES="--features cuda"; BACKENDS="cpu cuda"; fi

say "host facts"
rsh "$IP" 'source ~/.cargo/env
  echo "--- cpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|MHz|BogoMIPS" | sed "s/^/    /"
  echo "--- mem"; free -g | head -2 | sed "s/^/    /"
  echo "--- rustc"; rustc -Vv | head -3 | sed "s/^/    /"
  command -v nvidia-smi >/dev/null && { echo "--- gpu"; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | sed "s/^/    /"; }
  echo "--- nvrtc"; (ldconfig -p | grep -c nvrtc) 2>/dev/null || echo 0' 2>&1 | tee -a "$LOG"

say "building (release${FEATURES:+, cuda})"
BUILD_T0=$(date +%s)
rsh "$IP" "source ~/.cargo/env; cd ~/g16 && cargo build --release -p g16-cli $FEATURES 2>&1 | tail -3" 2>&1 | tee -a "$LOG"
say "build took $(( $(date +%s) - BUILD_T0 ))s"

TESTS_RESULT=skipped
if [ "$TESTS" = full ]; then
  say "correctness gate: cargo test --release $FEATURES"
  TEST_T0=$(date +%s)
  if rsh "$IP" "source ~/.cargo/env; cd ~/g16 && cargo test --release --workspace $FEATURES 2>&1 | grep -E '^test result|^error|panicked' | tail -40" 2>&1 | tee -a "$LOG" | grep -q 'FAILED\|error\[' ; then
    TESTS_RESULT=FAILED; say "CORRECTNESS GATE FAILED -- timings from this box are not trustworthy"
  else
    TESTS_RESULT=passed
  fi
  say "tests took $(( $(date +%s) - TEST_T0 ))s -> $TESTS_RESULT"
fi

FIRST_COMPILE_S=""
if [ -n "$GPU" ] && [ "$FIRST_COMPILE" = 1 ]; then
  say "measuring a genuine first kernel compile (both caches cleared)"
  FIRST_COMPILE_S="$(rsh "$IP" 'source ~/.cargo/env; cd ~/g16
    rm -rf ~/.cache/g16-cuda ~/.nv
    s=$(date +%s.%N)
    ./target/release/g16 prove --zkey bench/artifacts/tiny_mul/circuit.zkey \
      --witness bench/artifacts/tiny_mul/circuit.wtns --proof /tmp/fc.json \
      --public /tmp/fcp.json --backend cuda >/dev/null 2>&1
    e=$(date +%s.%N); echo "$e - $s" | bc' 2>/dev/null | tr -d '\r')"
  say "first compile + first proof: ${FIRST_COMPILE_S}s"
fi

if [ -n "$GPU" ]; then
  # This box has never run the prover, so ~/.nv and ~/.cache/g16-cuda are both empty and
  # this first proof pays the whole kernel pipeline: NVRTC source-to-PTX, then the driver's
  # PTX-to-SASS JIT inside cuModuleLoad. On a T4 that was measured at 113 s + 175 s. It is
  # a genuine deployment cost, it is per machine and not per proof, and it must not land
  # inside rep 1 of a benchmark. So it is paid here, timed here, and reported on its own.
  say "first CUDA run on a fresh box: paying NVRTC + driver JIT, outside the timed region"
  W0=$(date +%s)
  rsh "$IP" 'source ~/.cargo/env; cd ~/g16 && ./target/release/g16 prove \
    --zkey bench/artifacts/tiny_mul/circuit.zkey --witness bench/artifacts/tiny_mul/circuit.wtns \
    --proof /tmp/w.json --public /tmp/wp.json --backend cuda >/dev/null 2>&1 && echo ok' 2>&1 | tee -a "$LOG"
  [ -z "$FIRST_COMPILE_S" ] && FIRST_COMPILE_S=$(( $(date +%s) - W0 ))
  say "first kernel build took ${FIRST_COMPILE_S}s (paid once per machine, not per proof)"
  rsh "$IP" 'source ~/.cargo/env; cd ~/g16 && ./target/release/g16 prove \
    --zkey bench/artifacts/tiny_mul/circuit.zkey --witness bench/artifacts/tiny_mul/circuit.wtns \
    --proof /tmp/w.json --public /tmp/wp.json --backend cuda >/dev/null 2>&1 && echo warmed' 2>&1 | tee -a "$LOG"
fi

VARG=""
for v in $VARIANTS; do VARG="$VARG --variant $v"; done

for b in $BACKENDS; do
  say "benchmarking backend=$b mode=both reps=$REPS"
  rsh "$IP" "source ~/.cargo/env; cd ~/g16 && uptime && ./target/release/g16 bench \
    --artifacts bench/artifacts $VARG --reps $REPS --backend $b --mode both \
    --csv ~/g16/out-$b.csv" 2>&1 | tee -a "$LOG"
  scp -q -i "$KEY_PATH" "${SSH_OPTS[@]}" "ubuntu@$IP:~/g16/out-$b.csv" "$OUT/$b.csv" \
    && say "pulled $OUT/$b.csv ($(wc -l < "$OUT/$b.csv" | tr -d ' ') lines)"
done

CPUMODEL="$(rsh "$IP" 'lscpu | sed -n "s/^Model name: *//p" | head -1' 2>/dev/null | tr -d '\r')"
GPUNAME="$(rsh "$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1' 2>/dev/null | tr -d '\r')"
python3 - "$OUT/meta.json" <<PY
import json,sys
json.dump(dict(instance_type="$TYPE", arch="$ARCH", vcpu=$VCPU,
               cores=$(machine_field "$TYPE" cores), mem_gib=$(machine_field "$TYPE" mem_gib),
               gpu="$GPUNAME", gpu_sku="$GPU", cpu_model="""$CPUMODEL""",
               usd_per_hour=$PRICE, region="$REGION", reps=$REPS,
               tests="$TESTS_RESULT",
               first_compile_s=${FIRST_COMPILE_S:-None},
               git="$(cd "$REPO_ROOT" && git rev-parse --short HEAD)",
               dirty=$( [ -n "$(cd "$REPO_ROOT" && git status --porcelain)" ] && echo True || echo False )),
          open(sys.argv[1],"w"), indent=2)
PY
say "wrote $OUT/meta.json"
