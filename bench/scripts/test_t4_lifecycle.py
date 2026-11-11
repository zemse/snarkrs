#!/usr/bin/env python3
"""Host-only T4 lifecycle tests. Every AWS request uses an in-process fake."""
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

SPEC = importlib.util.spec_from_file_location("t4", Path(__file__).resolve().parents[1] / "aws/t4.py")
t4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t4)

ACCOUNT = "123456789012"
INSTANCE = "i-0123456789abcdef0"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/benchmark-stop"


def config(root):
    return {"G16_T4_STATE_DIR": str(root), "G16_T4_INSTANCE": INSTANCE,
            "G16_AWS_REGION": "us-east-1", "G16_AWS_ACCOUNT": ACCOUNT,
            "G16_AWS_OWNER": "bench-runner", "G16_AWS_STOP_ROLE_ARN": ROLE,
            "G16_T4_STOP_SCHEDULE": "mock-stop"}


class AWS:
    def __init__(self):
        self.calls = []
        self.state = "running"
        self.instance_type = "g4dn.xlarge"
        self.boot = "2026-01-01T00:00:00Z"
        self.owner = "bench-runner"
        self.identity = f"arn:aws:iam::{ACCOUNT}:user/bench-runner"
        self.fail_stop = False
        self.stop_responses = []
        self.stop_at = datetime.now(timezone.utc) + timedelta(minutes=30)
        self.schedule_patch = {}
        self.payload = {"InstanceIds": [INSTANCE], "Force": True, "SkipOsShutdown": True}

    def __call__(self, args, **kwargs):
        start = args.index("--no-cli-pager") + 1
        operation = args[start:-2]
        self.calls.append(operation)
        action = tuple(operation[:2])
        if action == ("sts", "get-caller-identity"):
            result = {"Account": ACCOUNT, "Arn": self.identity}
        elif action == ("ec2", "describe-instances"):
            result = {"Reservations": [{"Instances": [{"InstanceId": INSTANCE,
                      "InstanceType": self.instance_type, "State": {"Name": self.state},
                      "Tags": [{"Key": "Owner", "Value": self.owner}],
                      "PublicIpAddress": "192.0.2.1", "LaunchTime": self.boot}]}]}
        elif action == ("scheduler", "get-schedule"):
            result = {"Arn": f"arn:aws:scheduler:us-east-1:{ACCOUNT}:schedule/default/mock-stop",
                      "Name": "mock-stop", "GroupName": "default", "State": "ENABLED",
                      "ScheduleExpressionTimezone": "UTC", "FlexibleTimeWindow": {"Mode": "OFF"},
                      "ScheduleExpression": self.stop_at.strftime("at(%Y-%m-%dT%H:%M:%S)"),
                      "Target": {"Arn": t4.deadline.TARGET, "RoleArn": ROLE,
                                 "Input": json.dumps(self.payload)}}
            result.update(self.schedule_patch)
        elif action == ("ec2", "stop-instances"):
            if self.stop_responses:
                code, self.state, diagnostic = self.stop_responses.pop(0)
                return subprocess.CompletedProcess(args, code, "{}" if code == 0 else "", diagnostic)
            if self.fail_stop:
                return subprocess.CompletedProcess(args, 1, "", "mock stop failure")
            self.state = "stopped"
            result = {}
        else:
            raise AssertionError(f"unexpected AWS operation: {operation}")
        return subprocess.CompletedProcess(args, 0, json.dumps(result), "")


def release_worker(env, token, entered, proceed, results):
    aws = AWS()
    controller = t4.T4(env, command=aws)
    attempts = 0

    def wait_stopped():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            aws.state = "stopping"
            entered.set()
            if not proceed.wait(10):
                raise RuntimeError("test barrier timeout")
            raise TimeoutError("mock graceful timeout")
        aws.state = "stopped"

    controller.wait_stopped = wait_stopped
    try:
        result = controller.release("a", token)
        results.put(("release", result, aws.calls))
    except Exception as error:
        results.put(("release-error", str(error), aws.calls))


def acquire_worker(env, started, results):
    aws = AWS()
    aws.state = "stopped"
    started.set()
    try:
        t4.T4(env, command=aws).acquire("b")
        results.put(("unexpected-acquire", {}, aws.calls))
    except ValueError as error:
        results.put(("acquire", str(error), aws.calls))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = config(self.temp.name)
        self.aws = AWS()
        self.controller = t4.T4(self.env, command=self.aws)

    def stops(self):
        return [c for c in self.aws.calls if c[:2] == ["ec2", "stop-instances"]]

    def test_last_release_stops_and_duplicate_is_noop(self):
        a = self.controller.acquire("a")
        b = self.controller.acquire("b")
        self.assertNotEqual(a["token"], b["token"])
        self.controller.release("a", a["token"])
        self.assertFalse(self.stops())
        self.controller.release("b", b["token"])
        self.assertEqual(self.aws.state, "stopped")
        count = len(self.aws.calls)
        self.assertFalse(self.controller.release("b", b["token"])["released"])
        self.assertEqual(len(self.aws.calls), count)

    def test_stale_same_lane_release_cannot_stop_new_holder(self):
        old = self.controller.acquire("a")
        self.controller.release("a", old["token"])
        self.aws.state = "running"
        new = self.controller.acquire("a")
        count = len(self.stops())
        self.assertFalse(self.controller.release("a", old["token"])["released"])
        self.assertEqual(len(self.stops()), count)
        self.assertEqual(self.controller.load()["leases"], {"a": new["token"]})

    def test_stale_boot_release_never_stops(self):
        old = self.controller.acquire("a")
        self.aws.boot = "2026-01-02T00:00:00Z"
        self.assertFalse(self.controller.release("a", old["token"])["released"])
        new = self.controller.acquire("b")
        self.assertEqual(self.controller.load()["leases"], {"b": new["token"]})
        self.assertFalse(self.stops())

    def test_acquire_refuses_nonrunning_and_duplicate_lane(self):
        for state in ("stopping", "stopped", "pending", "terminated"):
            self.aws.state = state
            with self.assertRaisesRegex(ValueError, "not running"):
                self.controller.acquire("a")
        self.aws.state = "running"
        self.controller.acquire("a")
        with self.assertRaisesRegex(ValueError, "already leased"):
            self.controller.acquire("a")
        self.assertFalse(self.stops())

    def test_supported_single_t4_types_report_actual_type(self):
        for instance_type in ("g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge"):
            with self.subTest(instance_type=instance_type):
                self.aws.instance_type = instance_type
                lease = self.controller.acquire("a")
                self.assertEqual(self.controller.status()["instance_type"], instance_type)
                self.controller.release("a", lease["token"])
                self.aws.state = "running"

    def test_unsupported_types_refuse_acquire_and_stop(self):
        for instance_type in ("g5.xlarge", "g4dn.12xlarge", "c6i.2xlarge"):
            with self.subTest(instance_type=instance_type):
                self.aws.instance_type = instance_type
                with self.assertRaisesRegex(ValueError, "T4 type mismatch"):
                    self.controller.acquire("a")
                with self.assertRaisesRegex(ValueError, "T4 type mismatch"):
                    self.controller.stop_idle()
                self.assertFalse(self.controller.load()["leases"])
                self.assertFalse(self.stops())

    def test_deadline_requires_boolean_force_and_skip_os_shutdown(self):
        for flag in ("Force", "SkipOsShutdown"):
            for value in (None, False, 0, 1, "true"):
                with self.subTest(flag=flag, value=value):
                    self.aws.payload = {"InstanceIds": [INSTANCE], "Force": True,
                                        "SkipOsShutdown": True}
                    if value is None:
                        del self.aws.payload[flag]
                    else:
                        self.aws.payload[flag] = value
                    with self.assertRaisesRegex(ValueError, "target mismatch"):
                        self.controller.acquire("a")
                    self.assertFalse(self.controller.load()["leases"])
        self.assertFalse(self.stops())

    def test_deadline_failures_never_grant_lease(self):
        for change in ({"State": "DISABLED"}, {"ScheduleExpressionTimezone": "Asia/Kolkata"},
                       {"FlexibleTimeWindow": {"Mode": "FLEXIBLE"}}, {"Target": {}},
                       {"ScheduleExpression": "rate(1 hour)"}, {"Arn": "wrong"}):
            self.aws.schedule_patch = change
            with self.assertRaises((ValueError, TypeError)):
                self.controller.acquire("a")
            self.assertFalse(self.controller.load()["leases"])
        self.aws.schedule_patch = {}
        for minutes in (-1, 1441):
            self.aws.stop_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            with self.assertRaisesRegex(ValueError, "deadline"):
                self.controller.acquire("a")
        self.assertFalse(self.stops())

    def test_identity_owner_and_wrong_target_fail_closed(self):
        self.aws.identity = f"arn:aws:iam::{ACCOUNT}:root"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.controller.acquire("a")
        self.aws.identity = f"arn:aws:iam::{ACCOUNT}:user/bench-runner"
        self.aws.owner = "other"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.controller.acquire("a")
        self.aws.owner = "bench-runner"
        self.aws.payload = {"InstanceIds": ["i-fffffffffffffffff"]}
        with self.assertRaisesRegex(ValueError, "target"):
            self.controller.acquire("a")
        self.assertFalse(self.controller.load()["leases"])

    def test_failed_stop_keeps_lease_and_blocks_idle_stop(self):
        lease = self.controller.acquire("a")
        self.aws.fail_stop = True
        with self.assertRaisesRegex(RuntimeError, "mock stop failure"):
            self.controller.release("a", lease["token"])
        self.assertEqual(self.controller.load()["leases"], {"a": lease["token"]})
        with self.assertRaisesRegex(ValueError, "leases exist"):
            self.controller.stop_idle()

    def test_incorrect_instance_state_accepts_only_confirmed_stopped(self):
        diagnostic = ("An error occurred (IncorrectInstanceState) when calling the "
                      "StopInstances operation: instance is not in a stoppable state")
        lease = self.controller.acquire("a")
        for state in ("running", "stopping", "pending"):
            with self.subTest(state=state):
                self.aws.state = "running"
                self.aws.stop_responses = [(254, state, diagnostic)]
                with self.assertRaisesRegex(RuntimeError, "IncorrectInstanceState"):
                    self.controller.release("a", lease["token"])
                self.assertEqual(self.controller.load()["leases"], {"a": lease["token"]})
                self.assertTrue(all("--force" not in call for call in self.stops()))
        self.aws.state = "running"
        self.aws.stop_responses = [(254, "stopped", diagnostic)]
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertTrue(self.controller.release("a", lease["token"])["released"])
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn(diagnostic, stderr.getvalue())
        self.assertEqual(self.controller.load()["leases"], {})
        calls = len(self.aws.calls)
        self.assertFalse(self.controller.release("a", lease["token"])["released"])
        self.assertEqual(len(self.aws.calls), calls)

    def test_other_stop_errors_remain_errors_even_when_stopped(self):
        lease = self.controller.acquire("a")
        for code, diagnostic in (
                (254, "An error occurred (UnauthorizedOperation) when calling the StopInstances operation: denied"),
                (255, "Could not connect to the endpoint URL"),
                (255, "connection failed: IncorrectInstanceState")):
            with self.subTest(code=code, diagnostic=diagnostic):
                self.aws.state = "running"
                self.aws.stop_responses = [(code, "stopped", diagnostic)]
                with self.assertRaises(RuntimeError) as error:
                    self.controller.release("a", lease["token"])
                self.assertIn(diagnostic, str(error.exception))
                self.assertEqual(self.aws.calls[-1][:2], ["ec2", "stop-instances"])
                self.assertEqual(self.controller.load()["leases"], {"a": lease["token"]})
                self.assertTrue(all("--force" not in call for call in self.stops()))

    def test_forced_stop_state_race_warns_after_confirming_stopped(self):
        diagnostic = ("An error occurred (IncorrectInstanceState) when calling the "
                      "StopInstances operation: independent shutdown completed")
        lease = self.controller.acquire("a")
        self.aws.stop_responses = [(0, "stopping", ""), (254, "stopped", diagnostic)]
        with patch.object(self.controller, "wait_stopped", side_effect=TimeoutError):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertTrue(self.controller.release("a", lease["token"])["released"])
        self.assertIn(diagnostic, stderr.getvalue())
        self.assertEqual(len(self.stops()), 2)
        self.assertIn("--force", self.stops()[1])
        self.assertEqual(self.controller.load()["leases"], {})

    def test_timeout_rechecks_state_before_force(self):
        lease = self.controller.acquire("a")
        with patch.object(self.controller, "wait_stopped", side_effect=TimeoutError):
            with self.assertRaisesRegex(ValueError, "changed before forced"):
                self.controller.release("a", lease["token"])
        self.assertEqual(len(self.stops()), 1)
        self.assertNotIn("--force", self.stops()[0])
        self.assertIn("a", self.controller.load()["leases"])

    def test_wait_stopped_never_forces_restarted_box(self):
        with self.assertRaisesRegex(ValueError, "refusing forced"):
            self.controller.wait_stopped()
        self.assertFalse(self.stops())

    def test_state_scope_and_corruption_fail_closed(self):
        self.controller.acquire("a")
        env = dict(self.env, G16_T4_INSTANCE="i-fffffffffffffffff")
        with self.assertRaisesRegex(ValueError, "scope"):
            t4.T4(env, command=self.aws).status()
        (Path(self.temp.name) / "state.json").write_text("invalid json")
        with self.assertRaises(ValueError):
            self.controller.stop_idle()
        self.assertFalse(self.stops())

    def test_stop_without_leases_and_no_deadline_mutation(self):
        self.controller.stop_idle()
        self.assertEqual(self.aws.state, "stopped")
        self.assertTrue(all(c[1] not in ("start-instances", "run-instances", "terminate-instances",
                                       "delete-schedule", "update-schedule", "create-schedule")
                            for c in self.aws.calls))

    def test_cross_process_acquire_waits_through_forced_stop(self):
        lease = self.controller.acquire("a")
        ctx = multiprocessing.get_context("fork")
        entered, proceed, started = ctx.Event(), ctx.Event(), ctx.Event()
        results = ctx.Queue()
        release = ctx.Process(target=release_worker,
                              args=(self.env, lease["token"], entered, proceed, results))
        acquire = ctx.Process(target=acquire_worker, args=(self.env, started, results))
        release.start()
        try:
            self.assertTrue(entered.wait(5))
            acquire.start()
            self.assertTrue(started.wait(5))
            with self.assertRaises(queue.Empty):
                results.get(timeout=0.3)
            self.assertTrue(acquire.is_alive())
            proceed.set()
            rows = [results.get(timeout=10), results.get(timeout=10)]
            release.join(5)
            acquire.join(5)
            self.assertEqual(release.exitcode, 0)
            self.assertEqual(acquire.exitcode, 0)
            released = next(r for r in rows if r[0] == "release")
            self.assertTrue(released[1]["released"])
            stops = [c for c in released[2] if c[:2] == ["ec2", "stop-instances"]]
            self.assertEqual(len(stops), 2)
            self.assertIn("--force", stops[1])
            self.assertEqual(next(r for r in rows if r[0] == "acquire")[1].split(";")[0],
                             "instance is not running")
            self.assertEqual(self.controller.load()["leases"], {})
        finally:
            proceed.set()
            for process in (release, acquire):
                if process.pid:
                    process.join(5)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_wrapper_help_and_invalid_input_do_not_call_aws(self):
        wrapper = Path(t4.__file__).with_suffix(".sh")
        env = dict(os.environ, **self.env, PATH="/usr/bin:/bin")
        result = subprocess.run(["/bin/bash", str(wrapper), "--help"], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for args in (["acquire", "../bad"], ["release", "a", "bad"], ["run"]):
            result = subprocess.run(["/bin/bash", str(wrapper), *args], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("No such file", result.stderr)


if __name__ == "__main__":
    unittest.main()
