#!/usr/bin/env python3
"""Detached, token-scoped guest jobs. No AWS or guest provisioning."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time


def atomic(path, value):
    temp = path.with_suffix(".new")
    with temp.open("w") as output:
        json.dump(value, output)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


@contextmanager
def locked(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def session_pids(session):
    # timeout(1) can create additional process groups within the job's session.
    if Path("/proc/self/stat").exists():
        pids = [p.name for p in Path("/proc").iterdir() if p.name.isdecimal()]
    else:
        pids = subprocess.check_output(["ps", "-axo", "pid="], text=True).split()
    found = []
    for value in pids:
        pid = int(value)
        try:
            if os.getsid(pid) == session:
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists() and stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z":
                    continue
                found.append(pid)
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            pass
    return found


def kill_session(session, sig):
    for pid in session_pids(session):
        try:
            if os.getsid(pid) == session:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass


class Job:
    def __init__(self, root, token):
        if not re.fullmatch(r"[a-f0-9]{32}", token):
            raise ValueError("invalid job token")
        self.root = Path(root).resolve()
        self.token = token
        self.path = self.root / token

    def spec(self, lane):
        spec = json.loads((self.path / "spec.json").read_text())
        if spec.get("token") != self.token or spec.get("lane") != lane:
            raise ValueError("job token/lane mismatch")
        return spec

    def start(self, spec):
        if (spec.get("token") != self.token or
                not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,40}", spec.get("lane", "")) or
                not isinstance(spec.get("argv"), list) or not spec["argv"] or
                any(not isinstance(a, str) or "\0" in a for a in spec["argv"]) or
                type(spec.get("deadline")) not in (int, float)):
            raise ValueError("invalid job request")
        if not time.time() < spec["deadline"] <= time.time() + 86400:
            raise ValueError("job deadline expired or more than 24 hours away")
        if not (self.path / "work").is_dir():
            raise ValueError("job payload is not staged")
        with locked(self.path / "control.lock"):
            target = self.path / "spec.json"
            if target.exists():
                if json.loads(target.read_text()) != spec:
                    raise ValueError("immutable job request mismatch")
            else:
                atomic(target, spec)
            if not (self.path / "terminal.json").exists():
                with (self.path / "supervisor.log").open("ab") as log:
                    subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--root", str(self.root),
                         "supervise", self.token, spec["lane"]],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True,
                    )
        return {"token": self.token, "accepted": True}

    def status(self, lane):
        self.spec(lane)
        terminal = self.path / "terminal.json"
        if terminal.exists():
            return json.loads(terminal.read_text())
        return {"token": self.token, "lane": lane, "terminal": False,
                "state": "cancel-requested" if (self.path / "cancel.json").exists() else "active"}

    def cancel(self, lane):
        with locked(self.path / "control.lock"):
            self.spec(lane)
            if not (self.path / "terminal.json").exists():
                atomic(self.path / "cancel.json", {"token": self.token, "lane": lane})
        return self.status(lane)

    def collect(self, lane):
        if not self.status(lane).get("terminal"):
            raise ValueError("cannot collect an unconfirmed job")
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
            for name in ("terminal.json", "job.log", "supervisor.log", "work/out"):
                path = self.path / name
                if path.is_symlink():
                    raise ValueError("refusing symlink in results")
                if not path.exists():
                    continue
                paths = [path] if path.is_file() else sorted(path.rglob("*"))
                for item in paths:
                    if item.is_symlink():
                        raise ValueError("refusing symlink in results")
                    if item.is_dir():
                        continue
                    if not item.is_file():
                        raise ValueError("results must contain regular files only")
                    archive.add(item, arcname=str(item.relative_to(self.path)), recursive=False)

    def heartbeat(self):
        lease = Path("/var/lib/g16-watchdog/lease")
        if lease.exists():
            os.utime(lease, None)

    def supervise(self, lane):
        # All submit retries compete for one immutable job, never rerun a terminal job.
        with (self.path / "supervisor.lock").open("a") as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if (self.path / "terminal.json").exists() or (self.path / "started.json").exists():
                return
            spec = self.spec(lane)
            child = None
            rc = 125
            reason = "supervisor-error"
            with (self.root / "gpu.lock").open("a") as gpu, (self.path / "job.log").open("ab", buffering=0) as log:
                try:
                    heartbeat_at = 0
                    while True:
                        if time.monotonic() >= heartbeat_at:
                            self.heartbeat()
                            heartbeat_at = time.monotonic() + 30
                        if (self.path / "cancel.json").exists():
                            rc, reason = 130, "cancelled"
                            break
                        if time.time() >= spec["deadline"]:
                            rc, reason = 124, "deadline"
                            break
                        try:
                            fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            time.sleep(0.2)
                            continue
                        atomic(self.path / "started.json", {"token": self.token, "lane": lane})
                        child = subprocess.Popen(spec["argv"], cwd=self.path / "work",
                                                 stdin=subprocess.DEVNULL, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(gpu.fileno(),))
                        break
                    if child is not None:
                        reason = "exit"
                        while child.poll() is None:
                            if time.monotonic() >= heartbeat_at:
                                self.heartbeat()
                                heartbeat_at = time.monotonic() + 30
                            if (self.path / "cancel.json").exists():
                                reason = "cancelled"
                                break
                            if time.time() >= spec["deadline"]:
                                reason = "deadline"
                                break
                            time.sleep(0.2)
                        if reason != "exit":
                            kill_session(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                pass
                        kill_session(child.pid, signal.SIGKILL)
                        child.wait()
                        # Hold the GPU mutex until even timeout-created groups have exited.
                        while session_pids(child.pid):
                            kill_session(child.pid, signal.SIGKILL)
                            time.sleep(0.2)
                        rc = {"cancelled": 130, "deadline": 124}.get(reason, child.returncode)
                except Exception as error:
                    rc, reason = 125, "supervisor-error"
                    log.write(f"supervisor: {error}\n".encode())
                    if child is not None:
                        kill_session(child.pid, signal.SIGKILL)
                        child.wait()
                        while session_pids(child.pid):
                            kill_session(child.pid, signal.SIGKILL)
                            time.sleep(0.2)
                # terminal.json is the only authority for cleanup on the host.
                atomic(self.path / "terminal.json", {"token": self.token, "lane": lane,
                       "terminal": True, "returncode": rc, "reason": reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".cache/snarkrs/t4-jobs")
    parser.add_argument("action", choices=("start", "status", "cancel", "collect", "supervise"))
    parser.add_argument("token")
    parser.add_argument("lane")
    args = parser.parse_args()
    try:
        job = Job(args.root, args.token)
        if args.action == "start":
            spec = json.load(sys.stdin)
            if spec.get("lane") != args.lane:
                raise ValueError("job lane mismatch")
            result = job.start(spec)
        elif args.action == "collect":
            job.collect(args.lane)
            return 0
        elif args.action == "supervise":
            job.supervise(args.lane)
            return 0
        else:
            result = getattr(job, args.action)(args.lane)
        print(json.dumps(result))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"t4-job: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
