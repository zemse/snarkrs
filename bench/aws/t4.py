#!/usr/bin/env python3
"""Local leases and serialized stop confirmation for one persistent T4."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aws_stop_deadline as deadline


class T4:
    def __init__(self, env=os.environ, command=subprocess.run):
        self.region = env.get("G16_AWS_REGION", "us-east-1")
        self.instance = env.get("G16_T4_INSTANCE", "")
        self.owner = env.get("G16_AWS_OWNER", "bench-runner")
        self.account = env.get("G16_AWS_ACCOUNT", "123456789012")
        self.role = env.get("G16_AWS_STOP_ROLE_ARN", "")
        self.group = env.get("G16_AWS_SCHEDULER_GROUP", "default")
        self.schedule = env.get("G16_T4_STOP_SCHEDULE", "")
        self.root = Path(env.get("G16_T4_STATE_DIR", str(Path.home() / ".cache/snarkrs/t4")))
        deadline.validate_config(self.region, self.role, self.group, 1, self.instance)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.schedule):
            raise ValueError("G16_T4_STOP_SCHEDULE must name the independently armed schedule")
        self.command = command

    def aws(self, *args):
        return deadline.aws_json(self.region, list(args), command=self.command)

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Never replace the lock inode, including when resetting state.
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def load(self):
        path = self.root / "state.json"
        if not path.exists():
            return {"scope": [self.account, self.region, self.instance], "leases": {}}
        state = json.loads(path.read_text())
        if state.get("scope") != [self.account, self.region, self.instance]:
            raise ValueError("state directory belongs to another instance scope")
        if not isinstance(state.get("leases"), dict):
            raise ValueError("invalid lease state")
        return state

    def save(self, state):
        temp = self.root / "state.json.new"
        with temp.open("w") as output:
            json.dump(state, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temp.replace(self.root / "state.json")

    def describe(self):
        identity = self.aws("sts", "get-caller-identity")
        if (identity.get("Account") != self.account or
                identity.get("Arn") != f"arn:aws:iam::{self.account}:user/{self.owner}"):
            raise ValueError("refusing an unexpected AWS identity")
        result = self.aws("ec2", "describe-instances", "--instance-ids", self.instance)
        instances = [i for r in result.get("Reservations", []) for i in r.get("Instances", [])]
        if len(instances) != 1:
            raise ValueError("expected exactly one instance")
        instance = instances[0]
        tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
        if (instance.get("InstanceId") != self.instance or tags.get("Owner") != self.owner or
                instance.get("InstanceType") not in ("g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge")):
            raise ValueError("instance ID, Owner or T4 type mismatch")
        return instance

    def verify_deadline(self):
        actual = self.aws("scheduler", "get-schedule", "--name", self.schedule,
                          "--group-name", self.group)
        arn = f"arn:aws:scheduler:{self.region}:{self.account}:schedule/{self.group}/{self.schedule}"
        if (actual.get("Arn") != arn or actual.get("Name") != self.schedule or
                actual.get("GroupName") != self.group or actual.get("State") != "ENABLED" or
                actual.get("ScheduleExpressionTimezone") != "UTC" or
                actual.get("FlexibleTimeWindow") != {"Mode": "OFF"}):
            raise ValueError("independent deadline configuration mismatch")
        expression = actual.get("ScheduleExpression", "")
        if not re.fullmatch(r"at\(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\)", expression):
            raise ValueError("deadline must be a one-shot UTC schedule")
        stop_at = datetime.strptime(expression, "at(%Y-%m-%dT%H:%M:%S)").replace(tzinfo=timezone.utc)
        remaining = (stop_at - datetime.now(timezone.utc)).total_seconds()
        if not 0 < remaining <= 86400:
            raise ValueError("deadline expired or more than 24 hours away")
        target = actual.get("Target", {})
        payload = json.loads(target.get("Input", "null"))
        if (target.get("Arn") != deadline.TARGET or target.get("RoleArn") != self.role or
                not isinstance(payload, dict) or payload.get("InstanceIds") != [self.instance] or
                payload.get("Force") is not True or payload.get("SkipOsShutdown") is not True):
            raise ValueError("independent deadline target mismatch")
        return stop_at.isoformat()

    def acquire(self, lane):
        with self.locked():
            state = self.load()
            instance = self.describe()
            if instance["State"]["Name"] != "running":
                raise ValueError("instance is not running; coordinator must arm/start through native lifecycle")
            stop_at = self.verify_deadline()
            boot = instance.get("LaunchTime")
            if not boot:
                raise ValueError("instance has no boot generation")
            if state.get("boot") != boot:
                state = {"scope": state["scope"], "boot": boot, "leases": {}}
            if lane in state["leases"]:
                raise ValueError("lane already leased; release its exact token first")
            token = secrets.token_hex(16)
            state["leases"][lane] = token
            self.save(state)
            return {"instance": self.instance, "lane": lane, "token": token,
                    "host": instance.get("PublicIpAddress"), "deadline": stop_at}

    def wait_stopped(self):
        # Caller holds the same lock as acquire, including throughout forced fallback.
        end = time.monotonic() + 300
        while True:
            state = self.describe()["State"]["Name"]
            if state == "stopped":
                return
            if state != "stopping":
                raise ValueError(f"stop interrupted by state {state}; refusing forced stop")
            if time.monotonic() >= end:
                raise TimeoutError("instance still stopping")
            time.sleep(5)

    def stop_locked(self):
        # A failed or timed-out stop never clears the last lease.
        self.aws("ec2", "stop-instances", "--instance-ids", self.instance)
        try:
            self.wait_stopped()
        except TimeoutError:
            if self.describe()["State"]["Name"] != "stopping":
                raise ValueError("instance changed before forced stop")
            self.aws("ec2", "stop-instances", "--instance-ids", self.instance,
                     "--force", "--skip-os-shutdown")
            self.wait_stopped()

    def release(self, lane, token):
        with self.locked():
            state = self.load()
            if state["leases"].get(lane) != token:
                return {"released": False, "reason": "stale or unknown lease"}
            instance = self.describe()
            if state.get("boot") != instance.get("LaunchTime"):
                return {"released": False, "reason": "stale boot generation"}
            remaining = dict(state["leases"])
            del remaining[lane]
            if not remaining:
                current = instance["State"]["Name"]
                if current in ("running", "stopping"):
                    self.stop_locked()
                elif current != "stopped":
                    raise ValueError(f"refusing release in state {current}")
            state["leases"] = remaining
            self.save(state)
            return {"released": True, "leases": sorted(remaining)}

    def stop_idle(self):
        with self.locked():
            state = self.load()
            if state["leases"]:
                raise ValueError("refusing stop while leases exist; release exact tokens")
            current = self.describe()["State"]["Name"]
            if current in ("running", "stopping"):
                self.stop_locked()
            elif current != "stopped":
                raise ValueError(f"refusing stop in state {current}")
            return {"state": "stopped"}

    def status(self):
        with self.locked():
            state = self.load()
            instance = self.describe()
            return {"instance": self.instance, "instance_type": instance["InstanceType"],
                    "state": instance["State"]["Name"], "leases": sorted(state["leases"]),
                    "boot": state.get("boot")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("stop")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("lane")
    release = sub.add_parser("release")
    release.add_argument("lane")
    release.add_argument("token")
    args = parser.parse_args()
    try:
        if hasattr(args, "lane") and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,40}", args.lane):
            raise ValueError("invalid lane name")
        if args.action == "release" and not re.fullmatch(r"[a-f0-9]{32}", args.token):
            raise ValueError("invalid lease token")
        controller = T4()
        if args.action == "acquire":
            result = controller.acquire(args.lane)
        elif args.action == "release":
            result = controller.release(args.lane, args.token)
        elif args.action == "stop":
            result = controller.stop_idle()
        else:
            result = controller.status()
        print(json.dumps(result))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"t4: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
