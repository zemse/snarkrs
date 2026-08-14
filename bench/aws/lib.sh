#!/usr/bin/env bash
# Shared plumbing for the cross-machine sweep. Sourced by stage.sh and bench-machine.sh.
#
# Everything here exists because a sweep that launches thirteen boxes has thirteen chances
# to leak a running GPU instance. The two rules that keep that from happening:
#
#   1. Every instance carries purpose=groth16-cuda-benchmark, so terminate.sh can find and
#      kill it even if the driving process died without knowing the instance id.
#   2. Every instance carries its own dead-man switch (shutdown + terminate-on-shutdown),
#      so it dies on its own if BOTH the driver and the operator forget it.
#
# The Owner tag is load bearing: the IAM policy grants RunInstances only when the request
# carries Owner=${aws:username}, and TerminateInstances only on instances already tagged
# that way. Launching without it strands a billing instance that you are not allowed to
# terminate, which is the worst possible failure here.

REGION="${G16_AWS_REGION:-us-east-1}"
BUCKET="${G16_S3_BUCKET:-gpu-snark-bench}"
KEY_NAME="${G16_AWS_KEY:-g16-cuda-bench}"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SG_NAME="${G16_AWS_SG:-g16-cuda-bench-sg}"
PURPOSE=groth16-cuda-benchmark

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

aws_() { aws --region "$REGION" "$@"; }

iam_user() { aws sts get-caller-identity --query 'Arn' --output text | sed 's#.*/##'; }

# Machine facts come from the committed table, never from memory. machines.csv is generated
# from ec2:DescribeInstanceTypes and AWS's published on-demand price list; see its header.
machine_field() {  # machine_field <instance-type> <column>
  python3 - "$REPO_ROOT/bench/aws/machines.csv" "$1" "$2" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    if r['instance_type']==sys.argv[2]:
        print(r[sys.argv[3]]); break
else:
    sys.exit(f"unknown instance type {sys.argv[2]} (add it to machines.csv)")
PY
}

has_gpu() { [ -n "$(machine_field "$1" gpu)" ]; }

# AMI choice is a function of (architecture, needs a GPU driver). The Deep Learning Base
# AMIs ship the driver and CUDA toolkit already; building those from a bare Ubuntu image
# costs about fifteen minutes of billed time per box and is the most common way this fails.
resolve_ami() {  # resolve_ami <instance-type>
  local arch; arch="$(machine_field "$1" arch)"
  local name
  if has_gpu "$1"; then
    if [ "$arch" = arm64 ]; then
      # Pinned to 22.04, not "newest Ubuntu". The 26.04 arm64 DLAMI ships an NVRTC newer
      # than its own driver (both boxes carried driver 595.91.07, but the arm64 image had
      # 4 nvrtc libraries against 28 on the x86 one), so every module load failed with
      # CUDA_ERROR_UNSUPPORTED_PTX_VERSION: "the provided PTX was compiled with an
      # unsupported toolchain". Matching the x86 side's 22.04 also keeps the two
      # architectures on the same OS, which is one less difference between them.
      name='Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*'
    else
      name='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*'
    fi
    aws_ ec2 describe-images --owners amazon \
      --filters "Name=name,Values=$name" "Name=state,Values=available" \
                "Name=architecture,Values=$arch" \
      --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text
  else
    local a=amd64; [ "$arch" = arm64 ] && a=arm64
    aws_ ec2 describe-images --owners 099720109477 \
      --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-${a}-server-*" \
                "Name=state,Values=available" \
      --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text
  fi
}

ensure_key_and_sg() {
  if ! aws_ ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
    aws_ ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text > "$KEY_PATH"
    chmod 400 "$KEY_PATH"
  fi
  local vpc sg myip
  vpc="$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  sg="$(aws_ ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$vpc" \
         --description 'ssh for groth16 benchmarking' --query GroupId --output text 2>/dev/null \
       || aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
         --query 'SecurityGroups[0].GroupId' --output text)"
  myip="$(curl -s https://checkip.amazonaws.com)/32"
  aws_ ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
    --cidr "$myip" >/dev/null 2>&1 || true
  echo "$sg"
}

presign() { aws_ s3 presign "s3://$BUCKET/$1" --expires-in "${2:-28800}"; }

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=20)
rsh() { ssh -i "$KEY_PATH" "${SSH_OPTS[@]}" "ubuntu@$1" "${@:2}"; }
