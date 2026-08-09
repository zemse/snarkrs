#!/usr/bin/env bash
# Terminate the benchmark box and prove it is gone.
#
# "I ran terminate-instances" is not the same as "I am not being billed". This waits for
# the state to actually reach `terminated` and then re-lists every instance tagged for this
# exercise, so the exit is evidence rather than an assumption. A GPU instance left running
# by a command that returned successfully but did not take effect is the expensive failure
# mode here.
#
# With no argument it finds every instance tagged purpose=groth16-cuda-benchmark that is
# not already terminated, which is what you want if you have lost the instance id.
set -euo pipefail
REGION="${G16_AWS_REGION:-us-east-1}"

if [ $# -ge 1 ]; then
  IDS="$*"
else
  IDS="$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:purpose,Values=groth16-cuda-benchmark" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)"
fi

if [ -z "${IDS// /}" ]; then
  echo "nothing to terminate: no live instance tagged purpose=groth16-cuda-benchmark"
else
  echo "terminating: $IDS"
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS \
    --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids $IDS
  echo "confirmed terminated"
fi

echo
echo "=== any instance still alive on this account ==="
aws ec2 describe-instances --region "$REGION" \
  --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`]|[0].Value]' \
  --output text || true
echo "(empty above means nothing is running)"

echo
echo "=== volumes not attached to anything (these still bill) ==="
aws ec2 describe-volumes --region "$REGION" --filters "Name=status,Values=available" \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output text || true
echo "(empty above means no orphaned volumes)"

echo
echo "The key pair and security group are left in place: neither costs anything, and"
echo "deleting them would make the next provision.sh run slower for no saving."
