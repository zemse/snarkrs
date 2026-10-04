#!/usr/bin/env python3
"""Arm and verify an AWS-side one-shot emergency stop deadline."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ACCOUNT = "123456789012"
EXPECTED_IDENTITY = f"arn:aws:iam::{ACCOUNT}:user/bench-runner"
TARGET = "arn:aws:scheduler:::aws-sdk:ec2:stopInstances"
COMMAND_TIMEOUT = 30


def validate_config(region, role, group, minutes, instance=None):
    if not re.fullmatch(r"(?:us-(?:east|west)|eu-(?:west|central|north|south)|ap-(?:east|south|southeast|northeast)|ca-(?:central|west)|sa-east|me-(?:south|central)|af-south|il-central)-[1-9]", region):
        raise ValueError("invalid commercial AWS region")
    if not re.fullmatch(rf"arn:aws:iam::{ACCOUNT}:role/(?:[A-Za-z0-9_+=,.@-]+/)*[A-Za-z0-9_+=,.@-]{{1,64}}", role) or len(role) > 2048:
        raise ValueError("G16_AWS_STOP_ROLE_ARN must be an exact same-account IAM role ARN")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", group):
        raise ValueError("invalid Scheduler group")
    if type(minutes) is not int or not 1 <= minutes <= 1440:
        raise ValueError("max minutes must be between 1 and 1440")
    if instance is not None and not re.fullmatch(r"i-(?:[a-f0-9]{8}|[a-f0-9]{17})", instance):
        raise ValueError("invalid EC2 instance ID")


def aws_json(region, args, *, command=subprocess.run):
    env = dict(os.environ, AWS_MAX_ATTEMPTS="2", AWS_RETRY_MODE="standard", AWS_PAGER="")
    result = command(
        ["aws", "--region", region, "--cli-connect-timeout", "5",
         "--cli-read-timeout", "10", "--no-cli-pager", *args, "--output", "json"],
        capture_output=True, text=True, timeout=COMMAND_TIMEOUT, env=env,
    )
    if result.returncode:
        raise RuntimeError(f"AWS {args[0]} {args[1]} failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def check_identity(region, *, command=subprocess.run):
    identity = aws_json(region, ["sts", "get-caller-identity"], command=command)
    if identity.get("Arn") != EXPECTED_IDENTITY or identity.get("Account") != ACCOUNT:
        raise ValueError("refusing an unexpected AWS identity")


def preflight(region, role, group, minutes, *, command=subprocess.run):
    validate_config(region, role, group, minutes)
    check_identity(region, command=command)
    result = aws_json(region, ["scheduler", "get-schedule-group", "--name", group], command=command)
    expected_arn = f"arn:aws:scheduler:{region}:{ACCOUNT}:schedule-group/{group}"
    # AWS omits the ARN for the default group.
    group_arn = result.get("Arn", expected_arn if group == "default" else None)
    if (result.get("Name") != group or result.get("State") != "ACTIVE" or
            group_arn != expected_arn):
        raise ValueError("Scheduler group is not active or does not match")


def deadline_time(launched_at, minutes):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", launched_at):
        raise ValueError("launch timestamp must be UTC YYYY-MM-DDTHH:MM:SSZ")
    launched = datetime.strptime(launched_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return launched + timedelta(minutes=minutes)


def arm(region, role, group, minutes, instance, launched_at, result_dir, *,
        command=subprocess.run, now=lambda: datetime.now(timezone.utc)):
    validate_config(region, role, group, minutes, instance)
    deadline = deadline_time(launched_at, minutes)
    current = now()
    launched = deadline - timedelta(minutes=minutes)
    if launched > current or deadline <= current:
        raise ValueError("launch timestamp is in the future or deadline has already passed")
    check_identity(region, command=command)
    name = f"g16-stop-{instance}-{launched.strftime('%Y%m%dT%H%M%S')}"
    request = {
        "Name": name, "GroupName": group,
        "ScheduleExpression": f"at({deadline.strftime('%Y-%m-%dT%H:%M:%S')})",
        "ScheduleExpressionTimezone": "UTC", "State": "ENABLED",
        "FlexibleTimeWindow": {"Mode": "OFF"}, "ActionAfterCompletion": "DELETE",
        "Target": {"Arn": TARGET, "RoleArn": role,
                   "Input": json.dumps({"InstanceIds": [instance], "Force": True,
                                        "SkipOsShutdown": True}),
                   "RetryPolicy": {"MaximumEventAgeInSeconds": 86400, "MaximumRetryAttempts": 185}},
    }
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "stop-deadline-request.json").write_text(json.dumps(request, indent=2) + "\n")
    created = aws_json(region, ["scheduler", "create-schedule", "--cli-input-json", json.dumps(request)], command=command)
    (result_dir / "stop-deadline-create.json").write_text(json.dumps(created, indent=2) + "\n")
    actual = aws_json(region, ["scheduler", "get-schedule", "--name", name, "--group-name", group], command=command)
    (result_dir / "stop-deadline-observed.json").write_text(json.dumps(actual, indent=2) + "\n")
    expected_arn = f"arn:aws:scheduler:{region}:{ACCOUNT}:schedule/{group}/{name}"
    if created.get("ScheduleArn") != expected_arn or actual.get("Arn") != expected_arn:
        raise ValueError("deadline schedule ARN verification failed")
    for key in ("Name", "GroupName", "ScheduleExpression", "ScheduleExpressionTimezone",
                "State", "FlexibleTimeWindow", "ActionAfterCompletion"):
        if actual.get(key) != request[key]:
            raise ValueError(f"deadline verification failed: {key}")
    target = actual.get("Target", {})
    for key in ("Arn", "RoleArn", "RetryPolicy"):
        if target.get(key) != request["Target"][key]:
            raise ValueError(f"deadline target verification failed: {key}")
    if json.loads(target.get("Input", "null")) != json.loads(request["Target"]["Input"]):
        raise ValueError("deadline target input verification failed")
    if now() >= deadline:
        raise ValueError("deadline passed while arming schedule")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "arm"))
    parser.add_argument("--region", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--group", default="default")
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--instance")
    parser.add_argument("--launched-at")
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            preflight(args.region, args.role, args.group, args.minutes)
        else:
            if not all((args.instance, args.launched_at, args.result_dir)):
                raise ValueError("arm requires instance, launched-at and result-dir")
            actual = arm(args.region, args.role, args.group, args.minutes, args.instance,
                         args.launched_at, args.result_dir)
            print(f"AWS stop deadline verified: {actual['Arn']} {actual['ScheduleExpression']}")
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"AWS stop deadline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
