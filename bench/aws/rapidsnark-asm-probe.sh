#!/usr/bin/env bash
# Is rapidsnark faster than our prover because of its algorithm, or because of its
# hand-written x86_64 assembly?
#
#   rapidsnark-asm-probe.sh [instance-type] [reps]
#
# The last round left this unresolved and it mattered: our CPU prover measured 0.92x to
# 1.19x of rapidsnark on an M2 Max but 1.35x to 1.49x SLOWER on a Xeon. Same Rust, same
# arkworks. The explanation offered was rapidsnark's `src/asm` field arithmetic, which
# exists only for x86_64, so on Apple silicon both sides fall back to compiler-generated
# code. That was an inference from two machines with four other differences between them.
#
# This settles it on ONE machine. rapidsnark ships two build targets that differ in exactly
# one flag:
#
#   make host         -> cmake -DUSE_ASM=YES (default)  -> package/
#   make host_noasm   -> cmake -DUSE_ASM=NO             -> package_noasm/
#
# Same source, same compiler, same CPU, same run. If rapidsnark-noasm lands next to our
# prover while rapidsnark-asm is well ahead of both, the gap is the assembly and nothing
# else. If rapidsnark-noasm is still well ahead of us, the explanation was wrong and the
# difference is algorithmic, which is a much more useful thing to learn.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TYPE="${1:-c7a.2xlarge}"
REPS="${2:-10}"
HOURS="${G16_AWS_HOURS:-2}"
VARIANTS="${G16_VARIANTS:-js_2x2_d32 js_8x8_d32 js_16x16_d32}"
ARCH="$(machine_field "$TYPE" arch)"
PRICE="$(machine_field "$TYPE" usd_per_hour)"
OUT="$REPO_ROOT/bench/results/rapidsnark-asm"
mkdir -p "$OUT"
LOG="$OUT/$TYPE.log"; : > "$LOG"
say() { printf '[asm-probe %s] %s\n' "$TYPE" "$*" | tee -a "$LOG"; }

INSTANCE=""; T0=$(date +%s)
cleanup() {
  local code=$?
  if [ -n "$INSTANCE" ]; then
    say "terminating $INSTANCE"
    aws_ ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null 2>&1 \
      && aws_ ec2 wait instance-terminated --instance-ids "$INSTANCE" >/dev/null 2>&1 \
      && say "confirmed terminated" || say "WARNING: run bench/aws/terminate.sh"
    say "billed about $(( ($(date +%s) - T0 + 59) / 60 )) min at \$$PRICE/hr"
  fi
  exit $code
}
trap cleanup EXIT INT TERM

AMI="$(resolve_ami "$TYPE")"; SG="$(ensure_key_and_sg)"; USER_NAME="$(iam_user)"
say "arch=$ARCH ami=$AMI price=\$$PRICE/hr"

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$((HOURS * 60))
exec > /var/log/g16-bootstrap.log 2>&1
set -x
# Delimiter is # not |: the alternation contains a | and sed would read it as the end
# of the pattern, failing with unbalanced parentheses behind the `|| true`.
sed -i -E 's#http://[a-z0-9-]+\\.ec2\\.(archive|ports)\\.ubuntu\\.com#http://\\1.ubuntu.com#g' \
  /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true
export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=20 update
# rapidsnark needs rather more than our prover: cmake and m4 for the vendored GMP, nasm for
# the x86_64 assembler, libsodium for its randomness, libomp for its thread pool.
apt-get install -y build-essential cmake m4 nasm libsodium-dev libomp-dev git curl patch
sudo -u ubuntu bash -lc 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal'
touch /home/ubuntu/BOOTSTRAP_DONE
EOF

INSTANCE="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":80,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Owner,Value=$USER_NAME},{Key=Name,Value=g16-asmprobe},{Key=purpose,Value=$PURPOSE},{Key=autoterminate,Value=${HOURS}h}]" \
  --query 'Instances[0].InstanceId' --output text)"
rm -f "$UD"; say "launched $INSTANCE"
aws_ ec2 wait instance-running --instance-ids "$INSTANCE"
IP="$(aws_ ec2 describe-instances --instance-ids "$INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
say "at $IP; waiting for ssh"
for i in $(seq 1 60); do rsh "$IP" true 2>/dev/null && break; sleep 10; done
for i in $(seq 1 90); do rsh "$IP" 'test -f ~/BOOTSTRAP_DONE' 2>/dev/null && break; sleep 10; done
rsh "$IP" 'test -f ~/BOOTSTRAP_DONE' || { say "bootstrap failed"; rsh "$IP" 'sudo tail -30 /var/log/g16-bootstrap.log'; exit 1; }
say "bootstrap done"

rsh "$IP" "set -e; mkdir -p ~/g16/bench/artifacts
  curl -sS -o /tmp/s.tgz '$(presign src.tgz)'; tar xzf /tmp/s.tgz -C ~/g16
  curl -sS -o /tmp/a.tgz '$(presign artifacts.tgz)'; tar xzf /tmp/a.tgz -C ~/g16/bench/artifacts" 2>&1 | tail -2 | tee -a "$LOG"

say "building our prover"
rsh "$IP" 'source ~/.cargo/env; cd ~/g16 && cargo build --release -p g16-cli 2>&1 | tail -2' 2>&1 | tee -a "$LOG"

say "building rapidsnark twice: USE_ASM=YES and USE_ASM=NO"
rsh "$IP" 'set -e; source ~/.cargo/env; cd ~/g16
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  mkdir -p bench/vendor && cd bench/vendor
  [ -d rapidsnark ] || git clone --recursive --depth 1 -q https://github.com/iden3/rapidsnark.git
  cd rapidsnark
  # Upstream build_gmp.sh breaks out of its mirror loop on transfer success but verifies the
  # checksum only afterwards, then deletes the archive -- so one corrupt mirror kills the
  # build with good mirrors untried. Patched here for the same reason as on the other boxes.
  [ -f build_gmp.sh.orig ] || { cp build_gmp.sh build_gmp.sh.orig
    patch -p0 < ../../scripts/rapidsnark-gmp-mirror-fallback.patch || cp build_gmp.sh.orig build_gmp.sh; }
  [ -d depends/gmp/package ] || ./build_gmp.sh host >/dev/null 2>&1
  make host        > /tmp/build_asm.log   2>&1 || { echo "ASM BUILD FAILED"; tail -20 /tmp/build_asm.log; }
  make host_noasm  > /tmp/build_noasm.log 2>&1 || { echo "NOASM BUILD FAILED"; tail -20 /tmp/build_noasm.log; }
  ls -d package package_noasm 2>&1' 2>&1 | tail -6 | tee -a "$LOG"

say "wiring both binaries and both warm wrappers"
rsh "$IP" 'set -e; cd ~/g16; mkdir -p bench/bin
  R=bench/vendor/rapidsnark
  for v in asm noasm; do
    P=$R/package; [ $v = noasm ] && P=$R/package_noasm
    cp $P/bin/prover   bench/bin/rapidsnark-$v
    cp $P/bin/verifier bench/bin/rapidsnark-verify-$v 2>/dev/null || true
    g++ -std=c++17 -O3 -fopenmp -o bench/bin/rapidsnark-warm-$v bench/wrappers/rapidsnark-warm/main.cpp \
      -I$P/include -I$R/src -I$R/depends/json/include -L$P/lib \
      -lrapidsnark -lrapidsnark-fr-fq -lfr -lfq -lgmp -lpthread -Wl,-rpath,$(cd $P/lib && pwd) \
      2>/dev/null && echo "warm wrapper $v ok" || echo "warm wrapper $v FAILED"
  done
  ls -la bench/bin/' 2>&1 | tail -10 | tee -a "$LOG"

say "measuring: ours vs rapidsnark-asm vs rapidsnark-noasm, cold and warm"
rsh "$IP" "set -e; source ~/.cargo/env; cd ~/g16
  echo 'variant,prover,mode,rep,ms' > /tmp/asm.csv
  for v in $VARIANTS; do
    Z=bench/artifacts/\$v/circuit.zkey; W=bench/artifacts/\$v/circuit.wtns; K=bench/artifacts/\$v/vkey.json
    for p in asm noasm; do
      for i in \$(seq 1 $REPS); do
        s=\$(date +%s%N); ./bench/bin/rapidsnark-\$p \$Z \$W /tmp/p.json /tmp/pub.json >/dev/null 2>&1; e=\$(date +%s%N)
        ./target/release/g16 verify --vkey \$K --proof /tmp/p.json --public /tmp/pub.json >/dev/null 2>&1 \\
          && echo \"\$v,rapidsnark-\$p,cold,\$i,\$(( (e-s)/1000000 ))\" >> /tmp/asm.csv
      done
      if [ -x bench/bin/rapidsnark-warm-\$p ]; then
        ./bench/bin/rapidsnark-warm-\$p \$Z \$W /tmp/p.json /tmp/pub.json $REPS 2>/dev/null \\
          | awk -v v=\$v -v p=\$p '/^rep /{print v\",rapidsnark-\"p\",warm,\"\$2\",\"\$NF}' >> /tmp/asm.csv
      fi
    done
  done
  ./target/release/g16 bench --artifacts bench/artifacts $(for v in $VARIANTS; do printf -- '--variant %s ' "\$v"; done) \\
    --reps $REPS --backend cpu --mode both --csv /tmp/ours.csv >/dev/null
  wc -l /tmp/asm.csv /tmp/ours.csv" 2>&1 | tail -5 | tee -a "$LOG"

scp -q -i "$KEY_PATH" "${SSH_OPTS[@]}" "ubuntu@$IP:/tmp/asm.csv"  "$OUT/$TYPE-rapidsnark.csv"
scp -q -i "$KEY_PATH" "${SSH_OPTS[@]}" "ubuntu@$IP:/tmp/ours.csv" "$OUT/$TYPE-ours.csv"
say "pulled results into $OUT"
