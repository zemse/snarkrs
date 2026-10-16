#!/usr/bin/env python3
"""Opt-in real EC2 shutdown tests. Preserve the instance and its EBS volume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import aws_stop_deadline as deadline

ROOT = Path(__file__).resolve().parents[2]
REGION = "us-east-1"
ROLE = os.environ.get("G16_AWS_STOP_ROLE_ARN", f"arn:aws:iam::{deadline.ACCOUNT}:role/g16-benchmark-stop")
OWNER = os.environ.get("G16_AWS_OWNER", "bench-runner")
KEY_NAME = os.environ.get("G16_AWS_KEY", "agent-us-east-1")
SG = os.environ.get("G16_AWS_SG_ID", "sg-0123456789abcdef0")
KEY = Path.home() / f".claude/skills/aws/keys/{KEY_NAME}.pem"
OUT = ROOT / "target/aws-bundles/shutdown-live"


def main():
    global OUT
    resume_hard = len(sys.argv) == 3 and sys.argv[1] == "--execute-hard"
    guest_only = sys.argv[1:] == ["--execute-guest"] or resume_hard
    if not resume_hard and sys.argv[1:] not in (["--execute"], ["--execute-guest"]):
        raise SystemExit("requires --execute, --execute-guest or --execute-hard INSTANCE")
    if resume_hard:
        OUT = OUT.with_name("shutdown-live-hard")
    elif guest_only:
        OUT = OUT.with_name("shutdown-live-guest")
    OUT.mkdir(parents=True, exist_ok=True)
    deadline.preflight(REGION, ROLE, "default", 4)
    instance = None
    schedule = None
    evidence = []

    def aws(*args):
        return deadline.aws_json(REGION, list(args))

    def state():
        return aws("ec2", "describe-instances", "--instance-ids", instance)["Reservations"][0]["Instances"][0]

    def wait(wanted, seconds=360):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            actual = state()["State"]["Name"]
            print(datetime.now(timezone.utc).isoformat(), instance, actual, flush=True)
            if actual == wanted:
                return
            time.sleep(10)
        raise RuntimeError(f"instance did not become {wanted}")

    def ssh(command, check=True):
        ip = state()["PublicIpAddress"]
        return subprocess.run(["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no",
                               "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=5",
                               f"ubuntu@{ip}", command], capture_output=True, text=True,
                              timeout=30, check=check)

    def ready():
        for _ in range(36):
            if ssh("test -f /var/lib/cloud/instance/boot-finished", check=False).returncode == 0:
                return
            time.sleep(5)
        raise RuntimeError("SSH readiness failed")

    def arm(minutes, name):
        nonlocal schedule
        schedule = deadline.arm(REGION, ROLE, "default", minutes, instance,
                                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), OUT / name)["Name"]

    def disarm():
        nonlocal schedule
        if schedule:
            listed = aws("scheduler", "list-schedules", "--name-prefix", schedule)["Schedules"]
            if listed:
                aws("scheduler", "delete-schedule", "--name", schedule, "--group-name", "default")
            schedule = None

    try:
        if resume_hard:
            candidate = sys.argv[2]
            deadline.validate_config(REGION, ROLE, "default", 8, candidate)
            if aws("scheduler", "list-schedules", "--name-prefix", f"g16-stop-{candidate}-")["Schedules"]:
                raise RuntimeError("refusing to restart with an armed deadline")
            observed = aws("ec2", "describe-instances", "--instance-ids", candidate)["Reservations"][0]["Instances"][0]
            tags = {t["Key"]: t["Value"] for t in observed.get("Tags", [])}
            if (observed["State"]["Name"] != "stopped" or observed["InstanceType"] != "t3.micro" or
                    tags.get("Owner") != OWNER or tags.get("Name") != "g16-shutdown-test-EPHEMERAL"):
                raise RuntimeError("resume requires a stopped owned t3.micro shutdown-test instance")
            instance = candidate
            aws("ec2", "start-instances", "--instance-ids", instance)
        else:
            launched = aws("ec2", "run-instances", "--image-id", "ami-062944a84867f2386",
                           "--instance-type", "t3.micro", "--key-name", KEY_NAME,
                           "--security-group-ids", SG,
                           "--instance-initiated-shutdown-behavior", "stop",
                           "--block-device-mappings", '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":8,"VolumeType":"gp3","DeleteOnTermination":false}}]',
                           "--tag-specifications", f"ResourceType=instance,Tags=[{{Key=Owner,Value={OWNER}}},{{Key=Name,Value=g16-shutdown-test-EPHEMERAL}}]")
            instance = launched["Instances"][0]["InstanceId"]
            (OUT / "instance.json").write_text(json.dumps(launched, indent=2))
        arm(8 if guest_only else 4, "initial" if guest_only else "cloud")
        wait("running")
        ready()
        assert {t["Key"]: t["Value"] for t in state()["Tags"]}["Owner"] == OWNER
        if not guest_only:
            ssh("systemctl is-active g16-watchdog.timer || true; test ! -e /run/systemd/shutdown/scheduled")
            wait("stopped", 420)
            assert not aws("scheduler", "list-schedules", "--name-prefix", schedule)["Schedules"]
            evidence.append({"test": "cloud-forced-stop", "state": state(), "guest_shutdown_armed": False})
            disarm()
        modes = ("hard",) if resume_hard else ("idle", "hard", "completion", "failure")
        for index, mode in enumerate(modes):
            if not (guest_only and index == 0):
                assert not aws("scheduler", "list-schedules", "--name-prefix", f"g16-stop-{instance}-")["Schedules"]
                aws("ec2", "start-instances", "--instance-ids", instance)
                arm(8, mode)
                wait("running")
                ready()
            (OUT / f"{mode}-previous-guest.log").write_text(ssh("sudo grep g16-watchdog /var/log/syslog || true").stdout)
            script = (ROOT / "bench/aws/idle-watchdog.sh").read_text()
            setup = "sudo systemctl stop g16-watchdog.timer 2>/dev/null || true; sudo mkdir -p /var/lib/g16-watchdog; sudo rm -f /var/lib/g16-watchdog/done; "
            setup += "sudo tee /usr/local/sbin/g16-watchdog >/dev/null <<'SCRIPT'\n" + script + "\nSCRIPT\n"
            setup += "sudo chmod 755 /usr/local/sbin/g16-watchdog\nsudo tee /etc/systemd/system/g16-watchdog.service >/dev/null <<'SERVICE'\n[Service]\nType=oneshot\nExecStart=/usr/local/sbin/g16-watchdog\nSERVICE\n"
            setup += "sudo tee /etc/systemd/system/g16-watchdog.timer >/dev/null <<'TIMER'\n[Timer]\nOnBootSec=5\nOnUnitActiveSec=5\nAccuracySec=1\n[Install]\nWantedBy=timers.target\nTIMER\n"
            config = "30 3600 20" if mode == "idle" else "3600 3600 20"
            if mode == "hard":
                uptime = int(float(ssh("cat /proc/uptime").stdout.split()[0]))
                config = f"3600 {uptime + 45} 20"
            setup += f"echo '{config}' | sudo tee /var/lib/g16-watchdog/config >/dev/null\nsudo touch /var/lib/g16-watchdog/lease\nsudo systemctl daemon-reload\nsudo systemctl enable --runtime --now g16-watchdog.timer\nsudo systemctl is-active g16-watchdog.timer\n"
            ssh(setup)
            if mode == "hard":
                ssh("nohup bash -c 'while true; do sudo touch /var/lib/g16-watchdog/lease; date -u +%FT%TZ; sleep 2; done' >/home/ubuntu/g16-heartbeat.log 2>&1 </dev/null &")
            if mode in ("completion", "failure"):
                result = ssh("bash -c 'trap \"sudo touch /var/lib/g16-watchdog/done\" EXIT; echo retrieved-before-stop > /tmp/g16-result.txt; exit " + ("7" if mode == "failure" else "0") + "'", check=False)
                assert result.returncode == (7 if mode == "failure" else 0)
                (OUT / f"{mode}-retrieved.txt").write_text(ssh("cat /tmp/g16-result.txt").stdout)
            wait("stopped", 180)
            observed = state()
            if observed.get("StateReason", {}).get("Code") != "Client.InstanceInitiatedShutdown":
                raise RuntimeError(f"{mode}: stop was not guest initiated")
            evidence.append({"test": mode, "state": observed, "heartbeat": mode == "hard"})
            disarm()
            (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))
        assert not aws("scheduler", "list-schedules", "--name-prefix", f"g16-stop-{instance}-")["Schedules"]
        aws("ec2", "start-instances", "--instance-ids", instance)
        arm(8, "retrieve")
        wait("running")
        ready()
        guest_log = ssh("sudo grep g16-watchdog /var/log/syslog || true").stdout
        (OUT / "guest-reasons.log").write_text(guest_log)
        (OUT / "hard-heartbeat.log").write_text(ssh("cat /home/ubuntu/g16-heartbeat.log").stdout)
        reasons = ("hard-limit",) if resume_hard else ("idle", "hard-limit", "job-finished")
        for reason in reasons:
            if f"poweroff: {reason}" not in guest_log:
                raise RuntimeError(f"missing guest shutdown reason: {reason}")
        if not resume_hard and guest_log.count("poweroff: job-finished") < 2:
            raise RuntimeError("missing completion/failure shutdown logs")
        print(f"PASS: installed guest shutdown tests {modes}, evidence retrieved", flush=True)
    finally:
        if instance:
            aws("ec2", "stop-instances", "--instance-ids", instance)
            wait("stopped")
            disarm()
            (OUT / "final-state.json").write_text(json.dumps(state(), indent=2))
        (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
