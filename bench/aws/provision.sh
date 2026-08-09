#!/usr/bin/env bash
# Bring up the GPU box the CUDA backend is developed and benchmarked on, then tear it down
# with terminate.sh. Everything here is what was actually run, including the two things
# that are not obvious.
#
# 1. THE OWNER TAG IS LOAD BEARING. The IAM policy on this account grants ec2:RunInstances
#    only when the request carries Owner=${aws:username}, and grants TerminateInstances
#    only on instances already tagged that way. A launch without it fails with a bare
#    UnauthorizedOperation that names no condition. The same tag is what makes the instance
#    terminable afterwards, so getting it wrong would strand a billing GPU.
#
# 2. THE DEAD-MAN SWITCH IS NOT OPTIONAL. user-data schedules a poweroff, and the instance
#    is launched with --instance-initiated-shutdown-behavior terminate, so the poweroff
#    terminates rather than stopping. If the session driving this box dies, is interrupted,
#    or is simply forgotten, the instance kills itself and billing stops. An EC2 GPU left
#    running over a weekend costs more than this entire exercise.
#
# A dry-run probe is NOT enough to confirm you may launch: RunInstances validates the AMI
# id before it checks authorization, so a probe with a placeholder AMI returns
# InvalidAMIID.Malformed and tells you nothing about permissions.
set -euo pipefail

REGION="${G16_AWS_REGION:-us-east-1}"
TYPE="${G16_AWS_TYPE:-g4dn.2xlarge}"   # 8 vCPU + 1 Tesla T4. The cheap CUDA box.
                                        # g4dn.xlarge is cheaper but 4 vCPU makes the CPU
                                        # baseline weak and the builds slow enough that the
                                        # saving is mostly given back in billed wall clock.
VOL_GB="${G16_AWS_VOL_GB:-120}"
HOURS="${G16_AWS_HOURS:-6}"
KEY_NAME="${G16_AWS_KEY:-g16-cuda-bench}"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SG_NAME="${G16_AWS_SG:-g16-cuda-bench-sg}"

USER_NAME="$(aws sts get-caller-identity --query 'Arn' --output text | sed 's#.*/##')"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
echo "iam user: $USER_NAME   your ip: $MYIP   region: $REGION"

# The Deep Learning Base OSS Nvidia Driver AMI ships the driver and several CUDA toolkits
# preinstalled. Building that from a bare Ubuntu image takes 15 minutes of billed time and
# is the most common way this setup fails. Resolved by name so it tracks the latest build;
# the SSM public-parameter path would be tidier but needs ssm:GetParametersByPath, which
# this IAM user does not have.
AMI="$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)"
echo "ami: $AMI"

if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
    --query KeyMaterial --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  echo "created $KEY_PATH"
fi

VPC="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text)"
SG="$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
        --description "ssh for groth16 cuda benchmarking" --vpc-id "$VPC" \
        --query GroupId --output text 2>/dev/null \
      || aws ec2 describe-security-groups --region "$REGION" --group-names "$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text)"
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP" >/dev/null 2>&1 || true
echo "sg: $SG (ssh open to $MYIP only)"

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/g16-bootstrap.log 2>&1
set -x
shutdown -h +$((HOURS * 60))
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential cmake m4 nasm libsodium-dev git curl pkg-config python3 unzip
sudo -u ubuntu bash -lc 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal'
sudo -u ubuntu bash -lc 'echo ". \\\$HOME/.cargo/env" >> ~/.bashrc'
nvidia-smi > /home/ubuntu/nvidia-smi.txt 2>&1
touch /home/ubuntu/BOOTSTRAP_DONE
EOF

ID="$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOL_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Owner,Value=$USER_NAME},{Key=Name,Value=g16-cuda-bench},{Key=purpose,Value=groth16-cuda-benchmark},{Key=autoterminate,Value=${HOURS}h}]" \
  --query 'Instances[0].InstanceId' --output text)"
rm -f "$UD"
echo "launched: $ID"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat <<EOF

instance:  $ID
address:   ubuntu@$IP
key:       $KEY_PATH
self-terminates in $HOURS hours unless you terminate it sooner

  ssh -i $KEY_PATH ubuntu@$IP
  ./bench/aws/terminate.sh $ID

Bootstrap (apt, rustup) takes a few minutes. It is done when ~/BOOTSTRAP_DONE exists.
EOF
