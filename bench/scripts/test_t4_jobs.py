#!/usr/bin/env python3
"""Host-only jobs: fake SSH/EC2, real local detached supervisors and workloads."""
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_t4_lifecycle import AWS, config, t4

HELPER = Path(t4.__file__).with_name("t4-job.py")
SPEC = importlib.util.spec_from_file_location("t4_job", HELPER)
guest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guest)
SLEEP = time.sleep


class SSH:
    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir()
        self.ack_disconnect = False
        self.status_failures = 0
        self.collect_failure = False
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if args[0] == "scp":
            source, destination = args[-2:]
            destination = self.home / destination.split(":", 1)[1]
            if "-r" in args:
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] != "ssh":
            raise AssertionError(f"unexpected transport: {args}")
        remote = shlex.split(args[-1])
        action = remote[-3] if len(remote) >= 5 else None
        if action == "status" and self.status_failures:
            self.status_failures -= 1
            return subprocess.CompletedProcess(args, 255, b"", b"mock disconnected")
        if action == "collect" and self.collect_failure:
            return subprocess.CompletedProcess(args, 255, b"", b"mock retrieval disconnected")
        self.assert_python(remote)
        result = subprocess.run([sys.executable, *remote[1:]], cwd=self.home,
                                env=dict(os.environ, HOME=str(self.home)), **kwargs)
        if action == "start" and self.ack_disconnect:
            self.ack_disconnect = False
            return subprocess.CompletedProcess(args, 255, b"", b"mock lost submit acknowledgement")
        return result

    @staticmethod
    def assert_python(remote):
        if remote[0] != "python3":
            raise AssertionError(f"unexpected remote command: {remote}")


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "guest-jobs"
        self.root.mkdir()
        self.jobs = []
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.stop_jobs)

    def stop_jobs(self):
        for job, spec in self.jobs:
            try:
                job.cancel(spec["lane"])
                self.wait(job, spec["lane"], timeout=8)
            except (ValueError, FileNotFoundError, AssertionError):
                pass

    def launch(self, code, lane="a", seconds=10, argv=None):
        token = secrets.token_hex(16)
        job = guest.Job(self.root, token)
        (job.path / "work").mkdir(parents=True)
        spec = {"token": token, "lane": lane,
                "argv": argv or [sys.executable, "-c", code], "deadline": time.time() + seconds}
        self.submit(job, spec)
        self.jobs.append((job, spec))
        return job, spec

    def submit(self, job, spec):
        result = subprocess.run([sys.executable, str(HELPER), "--root", str(self.root),
                                 "start", job.token, spec["lane"]], input=json.dumps(spec),
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def wait(self, job, lane="a", timeout=8):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            status = job.status(lane)
            if status.get("terminal"):
                return status
            SLEEP(0.05)
        self.fail(f"job never reached terminal: {job.path}")

    def test_detached_submit_is_idempotent_and_preserves_literal_argv(self):
        literal = "space 'quote' ; $(touch SHOULD_NOT_EXIST)"
        code = "from pathlib import Path; import sys,time; print(sys.argv[1], flush=True); time.sleep(.4); Path('once').open('a').write('x')"
        job, spec = self.launch(code, argv=[sys.executable, "-c", code, literal])
        self.submit(job, spec)
        self.assertEqual(self.wait(job)["returncode"], 0)
        self.submit(job, spec)
        self.assertEqual((job.path / "work/once").read_text(), "x")
        self.assertIn(literal, (job.path / "job.log").read_text())
        self.assertFalse((job.path / "work/SHOULD_NOT_EXIST").exists())
        changed = dict(spec, argv=["false"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            job.start(changed)

    def test_guest_gpu_flock_serializes_real_workloads(self):
        busy = self.base / "busy"
        code = ("import os,time; from pathlib import Path; "
                f"p=Path({str(busy)!r}); fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY); "
                "time.sleep(.5); os.close(fd); p.unlink()")
        first, _ = self.launch(code)
        second, _ = self.launch(code, lane="b")
        self.assertEqual(self.wait(first)["returncode"], 0)
        self.assertEqual(self.wait(second, "b")["returncode"], 0)

    def test_cancel_rejects_wrong_lane_and_never_touches_other_token(self):
        first, _ = self.launch("import time; time.sleep(3)")
        second, _ = self.launch("print('second survived')", lane="b")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            first.cancel("b")
        with self.assertRaises(FileNotFoundError):
            guest.Job(self.root, "f" * 32).cancel("a")
        first.cancel("a")
        self.assertEqual(self.wait(first)["returncode"], 130)
        self.assertEqual(self.wait(second, "b")["returncode"], 0)

    def test_deadline_bounds_running_and_waiting_jobs(self):
        first, _ = self.launch("import time; time.sleep(5)", seconds=2)
        marker = self.base / "must-not-run"
        second, _ = self.launch(f"from pathlib import Path; Path({str(marker)!r}).touch()",
                                lane="b", seconds=.5)
        self.assertEqual(self.wait(second, "b")["returncode"], 124)
        self.assertFalse(marker.exists())
        self.assertEqual(self.wait(first)["returncode"], 124)

    def test_cancel_cleans_additional_process_groups_in_job_session(self):
        code = ("import os,time; from pathlib import Path; pid=os.fork(); "
                "os.setpgrp() if pid==0 else None; "
                "Path('descendant').write_text(str(os.getpid())) if pid==0 else None; time.sleep(5)")
        job, _ = self.launch(code)
        marker = job.path / "work/descendant"
        end = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < end:
            SLEEP(.05)
        self.assertTrue(marker.exists())
        pid = int(marker.read_text())
        job.cancel("a")
        self.assertEqual(self.wait(job)["returncode"], 130)
        # Linux may retain a reparented zombie briefly; it is not executing GPU work.
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().rsplit(") ", 1)[1].split()[0], "Z")
        else:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_started_marker_prevents_reexecution_after_supervisor_loss(self):
        token = secrets.token_hex(16)
        job = guest.Job(self.root, token)
        (job.path / "work").mkdir(parents=True)
        (job.path / "started.json").write_text(json.dumps({"token": token, "lane": "a"}))
        spec = {"token": token, "lane": "a", "argv": [sys.executable, "-c", "print('must not run')"],
                "deadline": time.time() + 5}
        self.submit(job, spec)
        SLEEP(.3)
        self.assertFalse((job.path / "job.log").exists())
        self.assertFalse(job.status("a")["terminal"])


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.remote = object.__new__(t4.Remote)
        self.remote.options = []
        self.remote.host = "ubuntu@192.0.2.1"
        self.job = {"token": "0" * 32, "deadline": "1970-01-01T01:16:40+00:00"}

    def test_large_payload_uses_remaining_deadline_not_ten_minute_cap(self):
        with patch.object(t4.time, "time", return_value=1000), \
                patch.object(self.remote, "call") as call:
            self.remote.prepare(self.job, Path("payload"))
        self.assertEqual([c.kwargs["timeout"] for c in call.call_args_list], [30, 600, 3300])

    def test_expiring_deadline_makes_no_transfer_calls(self):
        with patch.object(t4.time, "time", return_value=4301), \
                patch.object(self.remote, "call") as call:
            with self.assertRaisesRegex(ValueError, "insufficient deadline"):
                self.remote.prepare(self.job, Path("payload"))
        call.assert_not_called()

    def test_budget_is_rechecked_before_payload(self):
        with patch.object(t4.time, "time", side_effect=[1000, 1000, 4301]), \
                patch.object(self.remote, "call") as call:
            with self.assertRaisesRegex(ValueError, "insufficient deadline"):
                self.remote.prepare(self.job, Path("payload"))
        self.assertEqual(call.call_count, 2)
        self.assertNotIn("-r", call.call_args.args[0])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = config(self.base / "state")
        self.aws = AWS()
        self.controller = t4.T4(self.env, command=self.aws)
        self.ssh = SSH(self.base / "home")
        key = self.base / "mock-key"
        key.touch()
        self.ssh_env = {"G16_T4_SSH_KEY": str(key)}
        self.ship = self.base / "ship with spaces"
        self.ship.mkdir()
        self.addCleanup(self.cleanup_jobs)
        self.sleep_patch = patch.object(t4.time, "sleep", lambda seconds: SLEEP(min(seconds, .05)))
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def remote(self, host):
        return t4.Remote(host, env=self.ssh_env, command=self.ssh)

    def cleanup_jobs(self):
        root = self.ssh.home / ".cache/snarkrs/t4-jobs"
        if root.exists():
            for path in root.iterdir():
                if path.is_dir() and (path / "spec.json").exists():
                    spec = json.loads((path / "spec.json").read_text())
                    guest.Job(root, path.name).cancel(spec["lane"])
            end = time.monotonic() + 8
            while time.monotonic() < end:
                if all((p / "terminal.json").exists() for p in root.iterdir()
                       if p.is_dir() and (p / "spec.json").exists()):
                    break
                SLEEP(.05)

    def saved_job(self):
        return json.loads(next((self.controller.root / "jobs").glob("*.json")).read_text())

    def test_disconnect_after_submit_and_during_poll_reconnects_once(self):
        self.ssh.ack_disconnect = True
        self.ssh.status_failures = 1
        code = "from pathlib import Path; import time; Path('once').open('a').write('x'); print('saved log', flush=True); time.sleep(.5)"
        rc = self.controller.run_job("a", self.ship, [sys.executable, "-c", code], self.remote)
        self.assertEqual(rc, 0)
        job = self.saved_job()
        work = self.ssh.home / f".cache/snarkrs/t4-jobs/{job['token']}/work"
        self.assertEqual((work / "once").read_text(), "x")
        self.assertIn("saved log", (Path(job["results"]) / "job.log").read_text())
        self.assertTrue(job["synced"])
        self.assertEqual(self.aws.state, "stopped")
        self.assertEqual(self.controller.load()["leases"], {})

    def test_unreachable_terminal_retains_lease_then_resume_recovers_failure(self):
        self.ssh.status_failures = 10
        code = "from pathlib import Path; import time; print('failure log',flush=True); Path('once').open('a').write('x'); time.sleep(.4); raise SystemExit(7)"
        with self.assertRaisesRegex(RuntimeError, "lease retained"):
            self.controller.run_job("a", self.ship, [sys.executable, "-c", code], self.remote)
        job = self.saved_job()
        self.assertEqual(self.aws.state, "running")
        with self.assertRaisesRegex(ValueError, "managed job is unconfirmed"):
            self.controller.release("a", job["token"])
        self.assertEqual(self.controller.load()["leases"], {"a": job["token"]})
        self.ssh.status_failures = 0
        self.assertEqual(self.controller.resume_job("a", job["token"], self.remote), 7)
        job = self.saved_job()
        self.assertIn("failure log", (Path(job["results"]) / "job.log").read_text())
        work = self.ssh.home / f".cache/snarkrs/t4-jobs/{job['token']}/work"
        self.assertEqual((work / "once").read_text(), "x")
        self.assertEqual(self.aws.state, "stopped")

    def test_failed_result_transfer_retains_terminal_job_and_lease(self):
        self.ssh.collect_failure = True
        code = "from pathlib import Path; Path('out/nested').mkdir(parents=True); Path('out/nested/report.txt').write_text('report'); print('ok')"
        with self.assertRaisesRegex(RuntimeError, "lease retained"):
            self.controller.run_job("a", self.ship, [sys.executable, "-c", code], self.remote)
        job = self.saved_job()
        self.assertFalse(job["synced"])
        self.assertEqual(self.aws.state, "running")
        self.ssh.collect_failure = False
        self.assertEqual(self.controller.resume_job("a", job["token"], self.remote), 0)
        job = self.saved_job()
        self.assertEqual((Path(job["results"]) / "work/out/nested/report.txt").read_text(), "report")

    def test_local_cross_token_cancel_is_rejected_before_ssh(self):
        self.ssh.status_failures = 10
        with self.assertRaises(RuntimeError):
            self.controller.run_job("a", self.ship,
                                    [sys.executable, "-c", "import time; time.sleep(2)"], self.remote)
        job = self.saved_job()
        calls = len(self.ssh.calls)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.controller.job_request("b", job["token"], "cancel", self.remote)
        with self.assertRaises(FileNotFoundError):
            self.controller.job_request("a", "e" * 32, "cancel", self.remote)
        self.assertEqual(len(self.ssh.calls), calls)
        self.controller.job_request("a", job["token"], "cancel", self.remote)
        self.ssh.status_failures = 0
        self.assertEqual(self.controller.resume_job("a", job["token"], self.remote), 130)

    def test_ship_symlinks_are_refused_before_acquisition(self):
        (self.ship / "link").symlink_to(HELPER)
        with self.assertRaisesRegex(ValueError, "regular files"):
            self.controller.run_job("a", self.ship, ["true"], self.remote)
        self.assertFalse(self.aws.calls)

    def test_run_cli_preserves_argv_and_returns_job_failure(self):
        argv = ["t4.py", "run", "a", str(self.ship), "--", "printf", "literal ; $HOME"]
        with patch.object(sys, "argv", argv), patch.object(t4, "T4", return_value=self.controller):
            with patch.object(self.controller, "run_job", return_value=7) as run:
                self.assertEqual(t4.main(), 7)
                run.assert_called_once_with("a", str(self.ship), ["printf", "literal ; $HOME"])

    def test_cuda_gate_uses_run_managed_cleanup_and_environment(self):
        gate = Path(t4.__file__).with_name("cuda-gate.sh").read_text()
        self.assertNotIn('release "$LANE"', gate)
        self.assertIn('"$T4" run "$LANE" "$SHIP" --', gate)
        self.assertIn('source "$HOME/g16-env.sh"', gate)


if __name__ == "__main__":
    unittest.main()
