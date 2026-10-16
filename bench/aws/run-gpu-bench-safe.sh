#!/usr/bin/env bash
# Build before renting, run a bounded job, preserve results and stop the instance.
# Usage: bash bench/aws/run-gpu-bench-safe.sh [reps] [variant ...]
set -euo pipefail
source "$(dirname "$0")/lib.sh"

REPS="${1:-10}"
if (( $# > 0 )); then shift; fi
VARIANTS=("$@")
if (( $# > 0 )); then
  for variant in "${VARIANTS[@]}"; do
    [[ "$variant" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$ ]] || { echo 'invalid variant' >&2; exit 1; }
  done
fi
ARTIFACT_ROOT="${G16_ARTIFACT_ROOT:-$REPO_ROOT/bench/artifacts}"
if [[ "$ARTIFACT_ROOT" != "$REPO_ROOT/bench/artifacts" ]] && (( ${#VARIANTS[@]} == 0 )); then
  echo 'specify variants when using a custom artifact root' >&2; exit 1
fi
if (( ${#VARIANTS[@]} > 0 )); then
  for variant in "${VARIANTS[@]}"; do
    for artifact in circuit.zkey circuit.wtns vkey.json public.json r1cs-info.txt; do
      [[ -s "$ARTIFACT_ROOT/$variant/$artifact" ]] || {
        echo "missing artifact: $ARTIFACT_ROOT/$variant/$artifact" >&2; exit 1;
      }
    done
    if [[ "$variant" == onion_p* ]]; then
      python3 "$REPO_ROOT/bench/scripts/check_onion_ready.py" "$ARTIFACT_ROOT/$variant"
    fi
  done
fi
VERIFY_BIN="${G16_VERIFY_BIN:-$REPO_ROOT/bench/bin/rapidsnark-verify}"
[[ -x "$VERIFY_BIN" ]] && file "$VERIFY_BIN" | grep -q 'ELF 64-bit.*x86-64' || {
  echo 'provide an x86_64 Linux rapidsnark verifier via G16_VERIFY_BIN before renting' >&2; exit 1;
}
TYPE="${G16_AWS_TYPE:-g4dn.xlarge}"
IDLE_MIN="${G16_IDLE_MINUTES:-60}"
MAX_MIN="${G16_MAX_MINUTES:-180}"
VOL_GB="${G16_AWS_VOL_GB:-120}"
KEY_NAME="agent-$REGION"
KEY_PATH="$HOME/.claude/skills/aws/keys/$KEY_NAME.pem"
ID= IP= UD= RESULT_DIR= HEARTBEAT=
for value in "$REPS" "$IDLE_MIN" "$MAX_MIN" "$VOL_GB"; do
  [[ "$value" =~ ^[1-9][0-9]{0,5}$ ]] || { echo 'expected positive integers' >&2; exit 1; }
done
[[ -f "$KEY_PATH" ]] || { echo "missing reusable SSH key: $KEY_PATH" >&2; exit 1; }
STOP_ROLE="${G16_AWS_STOP_ROLE_ARN:-}"
STOP_GROUP="${G16_AWS_SCHEDULER_GROUP:-default}"
DEADLINE_HELPER="$REPO_ROOT/bench/scripts/aws_stop_deadline.py"
python3 "$DEADLINE_HELPER" preflight --region "$REGION" --role "$STOP_ROLE" \
  --group "$STOP_GROUP" --minutes "$MAX_MIN"
[[ "$(machine_field "$TYPE" arch)" == x86_64 ]] && has_gpu "$TYPE" || {
  echo 'this runner requires an x86_64 GPU type in machines.csv' >&2; exit 1;
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$HEARTBEAT" ]]; then
    kill "$HEARTBEAT" 2>/dev/null || true
    wait "$HEARTBEAT" 2>/dev/null || true
  fi
  if [[ -n "$ID" ]]; then
    echo "==> stopping $ID (results remain on EBS)"
    if aws_ ec2 stop-instances --instance-ids "$ID" >/dev/null; then
      if aws_ ec2 wait instance-stopped --instance-ids "$ID"; then
        echo "stopped: $ID"
      else
        echo "graceful stop not confirmed; forcing stop of $ID (guest filesystem may need repair)" >&2
        if aws_ ec2 stop-instances --instance-ids "$ID" --force --skip-os-shutdown >/dev/null &&
           aws_ ec2 wait instance-stopped --instance-ids "$ID"; then
          echo "stopped after forced shutdown: $ID"
        else
          echo "STOP NOT CONFIRMED: $ID in $REGION; guest watchdog remains armed" >&2
          rc=1
        fi
      fi
    else
      echo "STOP NOT CONFIRMED: $ID in $REGION; guest watchdog remains armed" >&2
      rc=1
    fi
  fi
  [[ -z "$UD" ]] || rm -f "$UD"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if (( ${#VARIANTS[@]} > 0 )); then
  for variant in "${VARIANTS[@]}"; do
    if [[ "$variant" == onion_p* ]]; then
      python3 "$REPO_ROOT/bench/scripts/check_onion_ready.py" "$ARTIFACT_ROOT/$variant" \
        --manifest "$REPO_ROOT/target/aws-bundles/$variant.sha256"
    fi
  done
fi

BIN="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release/snarkrs"
(cd "$REPO_ROOT" && cargo zigbuild --locked --release -p snarkrs --features cuda --target x86_64-unknown-linux-gnu.2.31)
file "$BIN" | grep -q 'ELF 64-bit.*x86-64'
COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
AMI=$(resolve_ami "$TYPE")
[[ "$AMI" == ami-* ]] || { echo 'no GPU AMI found' >&2; exit 1; }
SG=$(aws_ ec2 describe-security-groups --filters Name=group-name,Values=agent-ssh --query 'SecurityGroups[0].GroupId' --output text)
[[ "$SG" == sg-* ]] || { echo 'missing agent-ssh security group' >&2; exit 1; }
aws_ ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null

UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
set -euo pipefail
exec > /var/log/g16-bootstrap.log 2>&1
# Arm the hard deadline before any other setup. Re-armed at every boot by systemd below.
shutdown -h +$MAX_MIN
mkdir -p /var/lib/g16-watchdog
printf '%s %s %s\n' '$((IDLE_MIN * 60))' '$((MAX_MIN * 60))' 300 > /var/lib/g16-watchdog/config
touch /var/lib/g16-watchdog/lease
cat > /usr/local/sbin/g16-watchdog <<'WATCHDOG'
$(cat "$REPO_ROOT/bench/aws/idle-watchdog.sh")
WATCHDOG
chmod 755 /usr/local/sbin/g16-watchdog
cat > /etc/systemd/system/g16-watchdog.service <<'SERVICE'
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/g16-watchdog
SERVICE
cat > /etc/systemd/system/g16-watchdog.timer <<'TIMER'
[Timer]
OnBootSec=60
OnUnitActiveSec=60
AccuracySec=1
[Install]
WantedBy=timers.target
TIMER
systemctl daemon-reload
systemctl enable --now g16-watchdog.timer
nvidia-smi > /home/ubuntu/nvidia-smi.txt
touch /home/ubuntu/BOOTSTRAP_DONE
EOF

# Use the request start, not guest boot time, so setup cannot extend the cap.
LAUNCHED_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
ID=$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOL_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Owner,Value=$OWNER},{Key=Name,Value=g16-cuda-EPHEMERAL},{Key=purpose,Value=$PURPOSE},{Key=idle-minutes,Value=$IDLE_MIN},{Key=max-minutes,Value=$MAX_MIN}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "launched: $ID in $REGION"
RESULT_DIR="$REPO_ROOT/bench/results/aws-$ID"
mkdir -p "$RESULT_DIR"
printf '%s\n' "$REGION $ID $TYPE $COMMIT" > "$RESULT_DIR/instance.txt"
printf 'launched: %s in %s\n' "$ID" "$REGION" > "$RESULT_DIR/launch.log"
python3 "$DEADLINE_HELPER" arm --region "$REGION" --role "$STOP_ROLE" \
  --group "$STOP_GROUP" --minutes "$MAX_MIN" --instance "$ID" \
  --launched-at "$LAUNCHED_AT" --result-dir "$RESULT_DIR" > "$RESULT_DIR/stop-deadline.log" 2>&1
# Leave the AWS deadline armed on every exit, even after a confirmed local stop.
if [[ -f "$REPO_ROOT/bench/scripts/watch_aws_bench.py" ]]; then
  nohup python3 "$REPO_ROOT/bench/scripts/watch_aws_bench.py" \
    "$RESULT_DIR/launch.log" "$$" --minutes "$MAX_MIN" \
    > "$RESULT_DIR/local-watchdog.log" 2>&1 < /dev/null &
fi
aws_ ec2 wait instance-running --instance-ids "$ID"
[[ "$(aws_ ec2 describe-instances --instance-ids "$ID" --query "Reservations[0].Instances[0].Tags[?Key=='Owner'].Value | [0]" --output text)" == "$OWNER" ]]
IP=$(aws_ ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
for attempt in {1..60}; do
  if rsh "$IP" 'test -f ~/BOOTSTRAP_DONE && sudo systemctl is-active --quiet g16-watchdog.timer'; then
    break
  fi
  sleep 10
done
rsh "$IP" 'test -f ~/BOOTSTRAP_DONE && sudo systemctl is-active --quiet g16-watchdog.timer'

# The lease covers transfers too; without the laptop it expires after the idle limit.
(
  while true; do
    rsh "$IP" 'sudo touch /var/lib/g16-watchdog/lease' || exit
    sleep 30
  done
) > "$RESULT_DIR/heartbeat.log" 2>&1 &
HEARTBEAT=$!
rsh "$IP" 'mkdir -p ~/g16/target/release ~/g16/bench/bin ~/g16/bench/artifacts'
scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "$BIN" "ubuntu@$IP:g16/target/release/snarkrs"
scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "$VERIFY_BIN" "ubuntu@$IP:g16/bench/bin/rapidsnark-verify"
RSYNC_SSH="ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o ConnectTimeout=10"
scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "$REPO_ROOT/bench/aws/rapidsnark-oracle.sh" "ubuntu@$IP:g16/bench/bin/rapidsnark-oracle"
scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "$REPO_ROOT/bench/scripts/check_reference_public.py" "ubuntu@$IP:g16/bench/bin/check_reference_public.py"
rsh "$IP" 'chmod 755 ~/g16/bench/bin/rapidsnark-oracle; mkdir -p ~/g16/bench/scripts'
# Ship runtime scripts only, never compiler worktrees or circuit-build directories.
rsync -aL -e "$RSYNC_SSH" "$REPO_ROOT/bench/scripts/" "ubuntu@$IP:g16/bench/scripts/"
ARTIFACT_FILTER=(--include '*/' --include circuit.zkey --include circuit.wtns
  --include vkey.json --include public.json --include r1cs-info.txt --include ship-metadata.json --exclude '*')
rsync -aL -e "$RSYNC_SSH" "${ARTIFACT_FILTER[@]}" \
  "$REPO_ROOT/bench/artifacts/tiny_mul" "ubuntu@$IP:g16/bench/artifacts/"
if (( ${#VARIANTS[@]} > 0 )); then
  for variant in "${VARIANTS[@]}"; do
    rsync -aL -e "$RSYNC_SSH" "${ARTIFACT_FILTER[@]}" \
      "$ARTIFACT_ROOT/$variant" "ubuntu@$IP:g16/bench/artifacts/"
    if [[ "$variant" == onion_p* ]]; then
      scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "$REPO_ROOT/target/aws-bundles/$variant.sha256" \
        "ubuntu@$IP:g16/bench/artifacts/$variant/bundle-sha256.txt"
    fi
  done
else
  rsync -aL -e "$RSYNC_SSH" --exclude /large/ --exclude /csp/ "${ARTIFACT_FILTER[@]}" \
    "$ARTIFACT_ROOT/" "ubuntu@$IP:g16/bench/artifacts/"
fi
JOB_RC=0
rsh "$IP" "bash -s -- $REPS ${VARIANTS[*]:-}" <<'JOB' > "$RESULT_DIR/job.log" 2>&1 || JOB_RC=$?
set -euo pipefail
trap 'sudo touch /var/lib/g16-watchdog/done' EXIT
cd ~/g16
if (( $# > 1 )); then
  for variant in "${@:2}"; do
    if [[ "$variant" == onion_p* ]]; then
      (cd "bench/artifacts/$variant" && sha256sum -c bundle-sha256.txt)
    fi
  done
fi
./target/release/snarkrs groth16 prove bench/artifacts/tiny_mul/circuit.zkey bench/artifacts/tiny_mul/circuit.wtns /tmp/warm.json /tmp/warmp.json --backend cuda
./bench/bin/rapidsnark-verify bench/artifacts/tiny_mul/vkey.json /tmp/warmp.json /tmp/warm.json
reps="$1"
shift
if [[ "${1:-}" == onion_p21 ]]; then
  python3 bench/scripts/run-onion-ladder.py --cold-reps "$reps" --warm-reps 5 --variants "$@"
elif (( $# > 0 )); then
  python3 bench/scripts/run-comparison.py --reps "$reps" --backends cpu cuda --skip-rapidsnark --warm-oracle "$PWD/bench/bin/rapidsnark-oracle" --variants "$@"
else
  python3 bench/scripts/run-comparison.py --reps "$reps" --backends cpu cuda --skip-rapidsnark --warm-oracle "$PWD/bench/bin/rapidsnark-oracle"
fi
JOB
# A failed job may still have useful partial results. Copy before the EXIT stop.
scp -i "$KEY_PATH" "${SSH_OPTS[@]}" "ubuntu@$IP:nvidia-smi.txt" "$RESULT_DIR/"
rsync -a -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
  "ubuntu@$IP:g16/bench/results/" "$RESULT_DIR/remote/"
echo "results: $RESULT_DIR"
exit "$JOB_RC"
