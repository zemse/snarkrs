#!/usr/bin/env bash
# Run the ethproofs client-side-proving benchmark on the hardware ethproofs runs it on:
# an AWS `mac2.metal`, Apple M1, 8 cores, 16 GB. Every published `circom` number on
# https://ethproofs.org/csp-benchmarks came off one of these, so this is the only
# configuration in which our row and theirs can be compared without an asterisk.
#
#   ./mac2-metal.sh up        # allocate the host, launch, provision, leave it running
#   ./mac2-metal.sh run       # build and benchmark on the running box, pull results back
#   ./mac2-metal.sh down      # terminate the instance and try to release the host
#   ./mac2-metal.sh status
#
# Read this before running `up`:
#
#   * **mac2.metal is a Dedicated Host, not an instance type you can just launch.** The
#     host is allocated first and the instance is placed onto it.
#   * **Mac Dedicated Hosts bill for a 24 hour minimum and cannot be released before it.**
#     `down` terminates the instance immediately, which stops nothing: the host is what
#     costs money, and `release-hosts` is refused until the 24 hours are up. Budget for a
#     full day whatever the benchmark takes. `down` prints the earliest release time and
#     the command to run then.
#   * **This needs `ec2:AllocateHosts` and `ec2:ReleaseHosts`,** which the `bench-runner`
#     IAM user does not have. Check before you plan around it:
#       aws iam simulate-principal-policy \
#         --policy-source-arn arn:aws:iam::123456789012:user/bench-runner \
#         --action-names ec2:AllocateHosts ec2:ReleaseHosts --output text
#     `implicitDeny` means the policy has to be widened first.
#   * **16 GB of RAM is the constraint that decides whether this works at all.** Check the
#     local `peak_memory` for keccak_2048 before allocating: the published circom row peaks
#     at 2.7 GB on the larger v1 circuit, so there is room, but a prover that mmaps a
#     1.1 GB key and then materialises tables from it can find the ceiling quickly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

REGION="${G16_AWS_REGION:-us-east-1}"
AZ="${G16_AWS_AZ:-us-east-1b}"
PURPOSE=groth16-csp-benchmark
NAME=g16-csp-mac2
KEY_NAME="${G16_AWS_KEY:-agent-$REGION}"
KEY_PATH="${G16_AWS_KEY_PATH:-$HOME/.claude/skills/aws/keys/$KEY_NAME.pem}"
SG_NAME="${G16_AWS_SG:-agent-ssh}"
STATE="$HERE/.mac2-state"

aws_() { aws --region "$REGION" "$@"; }
owner() { aws sts get-caller-identity --query Arn --output text | sed 's#.*/##'; }
log() { printf '\n=== %s ===\n' "$*"; }

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=40)
rsh() { ssh -i "$KEY_PATH" "${SSH_OPTS[@]}" "ec2-user@$(public_ip)" "$@"; }

save() { printf '%s=%s\n' "$1" "$2" >> "$STATE"; }
get()  { [ -f "$STATE" ] && awk -F= -v k="$1" '$1==k{v=$2} END{print v}' "$STATE"; }

public_ip() {
  aws_ ec2 describe-instances --instance-ids "$(get instance)" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

cmd_up() {
  [ -f "$KEY_PATH" ] || { echo "no key at $KEY_PATH (see the aws skill's registry)" >&2; exit 1; }
  local host; host="$(get host)"
  if [ -z "$host" ]; then
    log "allocating a mac2.metal dedicated host in $AZ (24 hour minimum billing)"
    host="$(aws_ ec2 allocate-hosts --instance-type mac2.metal --availability-zone "$AZ" \
        --quantity 1 --auto-placement on \
        --tag-specifications "ResourceType=dedicated-host,Tags=[{Key=Owner,Value=$(owner)},{Key=purpose,Value=$PURPOSE}]" \
        --query 'HostIds[0]' --output text)"
    save host "$host"
    save allocated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host $host"
  fi

  local instance; instance="$(get instance)"
  if [ -z "$instance" ]; then
    local ami sg
    # Pinned to the macOS 15 line rather than newest: 26 and 27 exist, and a benchmark host
    # that silently moves to a new OS between quarters is a variable nobody wanted.
    ami="$(aws_ ec2 describe-images --owners amazon \
      --filters 'Name=name,Values=amzn-ec2-macos-15.*' 'Name=architecture,Values=arm64_mac' \
                'Name=state,Values=available' \
      --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)"
    sg="$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
      --query 'SecurityGroups[0].GroupId' --output text)"
    log "launching mac2.metal on $host from $ami"
    instance="$(aws_ ec2 run-instances --instance-type mac2.metal --image-id "$ami" \
        --placement "HostId=$host,Tenancy=host" \
        --key-name "$KEY_NAME" --security-group-ids "$sg" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Owner,Value=$(owner)},{Key=Name,Value=$NAME},{Key=purpose,Value=$PURPOSE}]" \
        --query 'Instances[0].InstanceId' --output text)"
    save instance "$instance"
  fi

  # A Mac instance takes several minutes longer than a Linux one: the host runs a scrubbing
  # workflow on first placement and macOS itself boots slowly.
  log "waiting for $instance to run"
  aws_ ec2 wait instance-running --instance-ids "$instance"
  log "waiting for sshd on $(public_ip)"
  for _ in $(seq 1 60); do
    if rsh true 2>/dev/null; then break; fi
    sleep 20
  done
  rsh true || { echo "no ssh after 20 minutes" >&2; exit 1; }
  log "up: ssh -i $KEY_PATH ec2-user@$(public_ip)"
}

cmd_provision() {
  log "installing the toolchain"
  # The macOS AMI ships Xcode command line tools and homebrew; rust it does not.
  rsh 'command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable'
  rsh 'brew list cmake >/dev/null 2>&1 || brew install cmake'

  log "copying the repo (source only)"
  rsync -az --delete -e "ssh -i $KEY_PATH ${SSH_OPTS[*]}" \
    --exclude target --exclude bench/artifacts --exclude bench/ptau \
    --exclude bench/node_modules --exclude bench/vendor --exclude bench/bin \
    --exclude .git --exclude 'bench/csp/target' \
    "$REPO_ROOT/" "ec2-user@$(public_ip):g16/"

  log "fetching circuits and zkeys on the box (3.6 GB, from GitHub not from here)"
  rsh 'cd g16 && bash bench/scripts/csp-fetch.sh'
}

cmd_run() {
  log "building and benchmarking"
  rsh 'cd g16 && source $HOME/.cargo/env && bash bench/scripts/csp-bench.sh --reps 10 --mem-reps 10' \
    | tee "$REPO_ROOT/bench/results/csp/mac2-metal.log"
  log "pulling metrics back"
  mkdir -p "$REPO_ROOT/bench/results/csp/mac2-metal"
  rsync -az -e "ssh -i $KEY_PATH ${SSH_OPTS[*]}" \
    "ec2-user@$(public_ip):g16/bench/results/csp/metrics/" \
    "$REPO_ROOT/bench/results/csp/mac2-metal/"
  python3 "$REPO_ROOT/bench/scripts/csp_report.py" --metrics "$REPO_ROOT/bench/results/csp/mac2-metal"
}

cmd_down() {
  local instance host allocated
  instance="$(get instance)"; host="$(get host)"; allocated="$(get allocated_at)"
  if [ -n "$instance" ]; then
    log "terminating $instance"
    aws_ ec2 terminate-instances --instance-ids "$instance" >/dev/null
    aws_ ec2 wait instance-terminated --instance-ids "$instance"
  fi
  if [ -n "$host" ]; then
    log "releasing $host"
    if ! aws_ ec2 release-hosts --host-ids "$host"; then
      cat >&2 <<MSG

The host is still billing. Mac dedicated hosts have a 24 hour minimum and this one was
allocated at $allocated. Release it after that with:

  aws ec2 release-hosts --region $REGION --host-ids $host

MSG
      exit 1
    fi
  fi
  rm -f "$STATE"
}

cmd_status() {
  printf 'host      %s\n' "$(get host)"
  printf 'allocated %s\n' "$(get allocated_at)"
  printf 'instance  %s\n' "$(get instance)"
  [ -n "$(get host)" ] && aws_ ec2 describe-hosts --host-ids "$(get host)" \
    --query 'Hosts[0].[State,AvailabilityZone,AllocationTime]' --output text
  [ -n "$(get instance)" ] && aws_ ec2 describe-instances --instance-ids "$(get instance)" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text
  true
}

case "${1:-}" in
  up)        cmd_up; cmd_provision ;;
  provision) cmd_provision ;;
  run)       cmd_run ;;
  down)      cmd_down ;;
  status)    cmd_status ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
