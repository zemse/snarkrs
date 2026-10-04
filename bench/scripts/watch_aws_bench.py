#!/usr/bin/env python3
"""Separate laptop-side stop watchdog for a finite safe-runner job."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import time


EXPECTED_IDENTITY = "arn:aws:iam::123456789012:user/bench-runner"


def runner_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def validate_config(pid, minutes, force_after, poll_seconds, command_timeout,
                    retry_attempts, retry_backoff):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be positive")
    for name, value in (("minutes", minutes), ("force_after", force_after),
                        ("poll_seconds", poll_seconds), ("command_timeout", command_timeout),
                        ("retry_backoff", retry_backoff)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(retry_attempts, int) or not 1 <= retry_attempts <= 10:
        raise ValueError("retry_attempts must be between 1 and 10")


def aws_command(argv, *, command, clock, sleep, command_timeout,
                retry_attempts, retry_backoff, retry_until=None):
    for attempt in range(retry_attempts):
        timeout = command_timeout
        if retry_until is not None and clock() < retry_until:
            timeout = min(timeout, max(0.001, retry_until - clock()))
        try:
            result = command(argv, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout.strip()
            print(f"AWS command failed ({result.returncode}): {result.stderr.strip()}", flush=True)
        except (subprocess.TimeoutExpired, OSError) as error:
            print(f"AWS command failed: {error}", flush=True)
        if attempt + 1 == retry_attempts:
            break
        delay = min(retry_backoff * 2 ** attempt, command_timeout)
        if retry_until is not None:
            remaining = retry_until - clock()
            if remaining <= 0:
                break
            delay = min(delay, remaining)
        sleep(delay)
        if retry_until is not None and clock() >= retry_until:
            break
    return None


def monitor(log, pid, *, minutes=180, force_after=180, poll_seconds=30,
            command_timeout=30, retry_attempts=3, retry_backoff=2,
            clock=time.monotonic, sleep=time.sleep, aliveness=runner_alive,
            command=subprocess.run):
    validate_config(pid, minutes, force_after, poll_seconds, command_timeout,
                    retry_attempts, retry_backoff)
    if log.is_dir():
        raise ValueError("log must be a file path")
    deadline = clock() + minutes * 60

    def aws(argv, retry_until=None):
        return aws_command(
            argv, command=command, clock=clock, sleep=sleep,
            command_timeout=command_timeout, retry_attempts=retry_attempts,
            retry_backoff=retry_backoff, retry_until=retry_until,
        )

    identity = aws(["aws", "sts", "get-caller-identity", "--query", "Arn", "--output", "text"],
                   retry_until=deadline)
    if identity != EXPECTED_IDENTITY:
        raise SystemExit("unexpected or unavailable AWS identity")
    instance = None
    first_stop = None
    while True:
        if instance is None:
            try:
                text = log.read_text(errors="replace")
            except OSError as error:
                text = ""
                print(f"cannot read launch log: {error}", flush=True)
            match = re.search(r"^launched: (i-[a-f0-9]+) in ([a-z0-9-]+)$", text, re.M)
            if match:
                instance, region = match.groups()
                print(f"watching {instance} in {region}", flush=True)
        alive = aliveness(pid)
        if instance is not None:
            prefix = ["aws", "--region", region, "ec2"]
            boundary = deadline if first_stop is None else first_stop + force_after
            state = None
            if first_stop is not None or clock() < deadline:
                state = aws(
                    prefix + ["describe-instances", "--instance-ids", instance,
                              "--query", "Reservations[0].Instances[0].State.Name", "--output", "text"],
                    retry_until=min(boundary, clock()) if not alive else boundary,
                )
            if state in ("stopped", "terminated"):
                print(f"confirmed {instance}: {state}", flush=True)
                return
            if first_stop is not None or not aliveness(pid) or clock() >= deadline:
                if first_stop is None:
                    first_stop = clock()
                forced = clock() >= first_stop + force_after
                print(f"requesting {'forced ' if forced else ''}stop for {instance}", flush=True)
                argv = prefix + ["stop-instances", "--instance-ids", instance]
                if forced:
                    argv += ["--force", "--skip-os-shutdown"]
                aws(argv, retry_until=first_stop + force_after if not forced else None)
        elif not alive:
            print("runner exited before an instance ID was recorded; check launch log", flush=True)
            return
        elif clock() >= deadline:
            raise SystemExit("deadline reached without an instance ID; check launch log")
        delay = poll_seconds
        boundary = deadline if first_stop is None else first_stop + force_after
        if clock() < boundary:
            delay = min(delay, boundary - clock())
        sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("pid", type=int)
    parser.add_argument("--minutes", type=int, default=180)
    parser.add_argument("--force-after", type=float, default=180, help="seconds before forced stop")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--command-timeout", type=float, default=30)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2)
    args = parser.parse_args()
    try:
        monitor(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
