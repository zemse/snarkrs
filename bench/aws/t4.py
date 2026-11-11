#!/usr/bin/env python3
"""Local leases and serialized stop confirmation for one persistent T4."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aws_stop_deadline as deadline


class Remote:
    def __init__(self, host, env=os.environ, command=subprocess.run):
        ipaddress.ip_address(host)
        user = env.get("G16_T4_SSH_USER", "ubuntu")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
            raise ValueError("invalid SSH user")
        self.host = f"{user}@{host}"
        key = env.get("G16_T4_SSH_KEY", str(Path.home() / ".ssh" /
                      (env.get("G16_AWS_KEY", "g16-cuda-bench") + ".pem")))
        if not Path(key).is_file():
            raise ValueError("set G16_T4_SSH_KEY to an existing SSH key")
        self.options = ["-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                        "-o", "StrictHostKeyChecking=accept-new"]
        self.command = command

    def call(self, args, **kwargs):
        timeout = kwargs.pop("timeout", 30)
        result = self.command(args, stderr=subprocess.PIPE, timeout=timeout, **kwargs)
        if result.returncode:
            error = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else result.stderr
            raise RuntimeError(f"SSH/transfer failed: {error}")
        return result

    def prepare(self, job, ship):
        root = f".cache/snarkrs/t4-jobs/{job['token']}"
        code = f"from pathlib import Path; (Path.home() / {root!r} / 'work').mkdir(parents=True, exist_ok=True)"
        self.call(["ssh", *self.options, self.host, shlex.join(["python3", "-c", code])],
                  stdout=subprocess.PIPE)
        self.call(["scp", *self.options, str(Path(__file__).with_name("t4-job.py")),
                   f"{self.host}:{root}/helper.py"], stdout=subprocess.PIPE, timeout=600)
        self.call(["scp", *self.options, "-r", str(ship) + "/.", f"{self.host}:{root}/work/"],
                  stdout=subprocess.PIPE, timeout=600)

    def argv(self, job, action):
        helper = f".cache/snarkrs/t4-jobs/{job['token']}/helper.py"
        return ["ssh", *self.options, self.host,
                shlex.join(["python3", helper, action, job["token"], job["lane"]])]

    def request(self, job, action, spec=None):
        result = self.call(self.argv(job, action), stdout=subprocess.PIPE,
                           input=json.dumps(spec).encode() if spec else None)
        return json.loads(result.stdout)

    def collect(self, job, destination):
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / "results.tar.gz"
        with archive.open("wb") as output:
            self.call(self.argv(job, "collect"), stdout=output)
        names = set()
        with tarfile.open(archive) as bundle:
            for item in bundle:
                names.add(item.name)
                path = destination / item.name
                if (not item.isfile() or Path(item.name).is_absolute() or
                        ".." in Path(item.name).parts or path.is_symlink()):
                    raise ValueError("unsafe result archive")
                path.parent.mkdir(parents=True, exist_ok=True)
                if any(parent.is_symlink() for parent in path.parents):
                    raise ValueError("symlink in result destination")
                with bundle.extractfile(item) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if not {"terminal.json", "job.log"} <= names:
            raise ValueError("incomplete result archive")
        return json.loads((destination / "terminal.json").read_text())


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
            job_path = self.root / "jobs" / f"{token}.json"
            if job_path.exists() and not json.loads(job_path.read_text()).get("synced"):
                raise ValueError(f"managed job is unconfirmed; resume {lane} {token}")
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

    def save_job(self, job):
        directory = self.root / "jobs"
        directory.mkdir(exist_ok=True)
        path = directory / f"{job['token']}.json"
        temp = path.with_suffix(".new")
        with temp.open("w") as output:
            json.dump(job, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temp.replace(path)

    def load_job(self, lane, token):
        job = json.loads((self.root / "jobs" / f"{token}.json").read_text())
        if job.get("lane") != lane or job.get("token") != token:
            raise ValueError("job lane/token mismatch")
        with self.locked():
            state = self.load()
            if not job.get("synced") and state["leases"].get(lane) != token:
                raise ValueError("job no longer owns its lease")
        return job

    def run_job(self, lane, ship, argv, remote_factory=Remote):
        ship = Path(ship).resolve()
        if not ship.is_dir() or not argv:
            raise ValueError("run requires a ship directory and argv after --")
        for path in ship.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("ship directory must contain regular files and directories only")
        job = self.acquire(lane)
        job.update(argv=argv, ship=str(ship), synced=False, phase="preparing")
        self.save_job(job)
        print(f"t4: checkpoint token={job['token']}; resume {lane} {job['token']}", file=sys.stderr)
        try:
            return self.resume_job(lane, job["token"], remote_factory)
        except (Exception, KeyboardInterrupt) as error:
            raise RuntimeError(f"job unresolved, lease retained: token={job['token']}; "
                               f"resume {lane} {job['token']}: {error}") from error

    def resume_job(self, lane, token, remote_factory=Remote):
        # One collector per token; cancellation only sets a guest-side request marker.
        with (self.root / "jobs" / f"{token}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            job = self.load_job(lane, token)
            if not job.get("synced"):
                remote = remote_factory(job["host"])
                if job["phase"] == "preparing":
                    remote.prepare(job, Path(job["ship"]))
                    job["phase"] = "submitted"
                    self.save_job(job)
                spec = {"token": token, "lane": lane, "argv": job["argv"],
                        "deadline": datetime.fromisoformat(job["deadline"]).timestamp() - 30}
                # Submission retries use the same immutable request and guest supervisor lock.
                if time.time() < spec["deadline"]:
                    with self.locked():
                        state = self.load()
                        instance = self.describe()
                        if (state["leases"].get(lane) != token or
                                instance["State"]["Name"] != "running" or
                                state.get("boot") != instance.get("LaunchTime")):
                            raise ValueError("job lost its running lease before submission")
                        if self.verify_deadline() != job["deadline"]:
                            raise ValueError("independent deadline changed; refusing resubmission")
                        try:
                            remote.request(job, "start", spec)
                        except (RuntimeError, OSError, subprocess.TimeoutExpired):
                            print(f"t4: submission unconfirmed; polling token={token}", file=sys.stderr)
                failures = 0
                end = datetime.fromisoformat(job["deadline"]).timestamp() + 30
                while time.time() <= end:
                    try:
                        status = remote.request(job, "status")
                        if status.get("token") != token or status.get("lane") != lane:
                            raise ValueError("remote job identity mismatch")
                        if status.get("terminal") is True:
                            destination = self.root / "results" / token
                            terminal = remote.collect(job, destination)
                            if (terminal != status or type(terminal.get("returncode")) is not int):
                                raise ValueError("retrieved job status mismatch")
                            job.update(synced=True, terminal=terminal, results=str(destination))
                            self.save_job(job)
                            break
                        failures = 0
                    except (RuntimeError, OSError, ValueError, tarfile.TarError, subprocess.TimeoutExpired) as error:
                        failures += 1
                        if failures >= 3:
                            raise RuntimeError(f"job unresolved; lease/deadline retained: resume {lane} {token}: {error}") from error
                    time.sleep(2)
                if not job.get("synced"):
                    raise RuntimeError(f"terminal status unavailable; lease retained: resume {lane} {token}")
            self.release(lane, token)
            print(f"t4: token={token}, results={job['results']}, exit={job['terminal']['returncode']}",
                  file=sys.stderr)
            return job["terminal"]["returncode"]

    def job_request(self, lane, token, action, remote_factory=Remote):
        job = self.load_job(lane, token)
        if job.get("synced"):
            return job["terminal"]
        result = remote_factory(job["host"]).request(job, action)
        if result.get("token") != token or result.get("lane") != lane:
            raise ValueError("remote job identity mismatch")
        return result


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
    run = sub.add_parser("run")
    run.add_argument("lane")
    run.add_argument("ship_dir")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    for action in ("resume", "job-status", "cancel"):
        job = sub.add_parser(action)
        job.add_argument("lane")
        job.add_argument("token")
    args = parser.parse_args()
    try:
        if hasattr(args, "lane") and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,40}", args.lane):
            raise ValueError("invalid lane name")
        if hasattr(args, "token") and not re.fullmatch(r"[a-f0-9]{32}", args.token):
            raise ValueError("invalid lease token")
        controller = T4()
        if args.action == "run":
            if not args.argv:
                raise ValueError("usage: run <lane> <ship-dir> -- <argv...>")
            rc = controller.run_job(args.lane, args.ship_dir, args.argv)
            return rc if rc >= 0 else 128 - rc
        elif args.action == "resume":
            rc = controller.resume_job(args.lane, args.token)
            return rc if rc >= 0 else 128 - rc
        elif args.action in ("cancel", "job-status"):
            result = controller.job_request(args.lane, args.token,
                                            "cancel" if args.action == "cancel" else "status")
        elif args.action == "acquire":
            result = controller.acquire(args.lane)
        elif args.action == "release":
            result = controller.release(args.lane, args.token)
        elif args.action == "stop":
            result = controller.stop_idle()
        else:
            result = controller.status()
        print(json.dumps(result))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, tarfile.TarError, subprocess.TimeoutExpired) as error:
        print(f"t4: {error}", file=sys.stderr)
        if args.action in ("resume", "cancel", "job-status"):
            print(f"t4: lease not automatically released; job-status {args.lane} {args.token}; "
                  f"resume {args.lane} {args.token}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("t4: interrupted; leases remain intact. Use the checkpoint token to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
