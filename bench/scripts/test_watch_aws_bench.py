import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("watch_aws_bench.py")
spec = importlib.util.spec_from_file_location("watch_aws_bench", SCRIPT)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class FakeRuntime:
    def __init__(self, states, stops=(), alive=False, identity=watchdog.EXPECTED_IDENTITY):
        self.now = 0
        self.states = list(states)
        self.stops = list(stops)
        self.alive = alive
        self.identity = identity
        self.calls = []
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.now > 2000:
            raise AssertionError("monitor failed to finish")

    def aliveness(self, pid):
        return self.alive(self.now) if callable(self.alive) else self.alive

    def command(self, argv, **kwargs):
        self.calls.append((self.now, argv, kwargs))
        if "get-caller-identity" in argv:
            response = self.identity
        elif "describe-instances" in argv:
            if not self.states:
                raise AssertionError("unexpected describe")
            response = self.states.pop(0)
        elif "stop-instances" in argv:
            response = self.stops.pop(0) if self.stops else ""
        else:
            raise AssertionError(f"unexpected command: {argv}")
        if isinstance(response, subprocess.TimeoutExpired):
            self.now += kwargs["timeout"]
            raise response
        if isinstance(response, Exception):
            raise response
        if isinstance(response, int):
            return subprocess.CompletedProcess(argv, response, "", "test failure")
        return subprocess.CompletedProcess(argv, 0, response, "")

    def stop_calls(self):
        return [(now, argv) for now, argv, _ in self.calls if "stop-instances" in argv]


class WatchAwsBenchTests(unittest.TestCase):
    def run_monitor(self, runtime, text="launched: i-123abc in us-east-1\n", **options):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "launch.log"
            log.write_text(text)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                watchdog.monitor(
                    log, 123, clock=runtime.clock, sleep=runtime.sleep,
                    aliveness=runtime.aliveness, command=runtime.command, **options,
                )
            return output.getvalue()

    def test_describe_timeout_recovers_without_stopping_live_runner(self):
        runtime = FakeRuntime([subprocess.TimeoutExpired("aws", 30), "running", "stopped"],
                              alive=True)
        output = self.run_monitor(runtime)
        self.assertIn("confirmed i-123abc: stopped", output)
        self.assertEqual(runtime.stop_calls(), [])
        self.assertEqual(runtime.sleeps, [2, 30])

    def test_stop_timeout_recovers_and_requires_confirmation(self):
        runtime = FakeRuntime(["running", "stopping", "stopped"],
                              stops=[subprocess.TimeoutExpired("aws", 30), ""])
        output = self.run_monitor(runtime)
        self.assertEqual(len(runtime.stop_calls()), 3)
        self.assertIn("confirmed i-123abc: stopped", output)
        self.assertNotIn("--force", str(runtime.stop_calls()))

    def test_failed_describe_still_reaches_deadline_stop(self):
        runtime = FakeRuntime([7, OSError("unavailable"), 9,
                               subprocess.TimeoutExpired("aws", 30), "stopped"], alive=True)
        output = self.run_monitor(runtime, minutes=1)
        self.assertEqual(len(runtime.stop_calls()), 1)
        self.assertEqual(runtime.stop_calls()[0][0], 60)
        self.assertIn("confirmed i-123abc: stopped", output)
        self.assertTrue(all(call[2]["timeout"] <= 30 for call in runtime.calls))

    def test_persistent_describe_timeouts_do_not_delay_deadline_stop(self):
        runtime = FakeRuntime([subprocess.TimeoutExpired("aws", 30),
                               subprocess.TimeoutExpired("aws", 30), "stopped"], alive=True)
        self.run_monitor(runtime, minutes=1)
        self.assertEqual(runtime.stop_calls()[0][0], 60)
        self.assertEqual(runtime.calls[2][2]["timeout"], 28)

    def test_failed_stop_attempts_do_not_reset_escalation_clock(self):
        runtime = FakeRuntime(["running", "stopping", "stopping", "stopped"],
                              stops=[subprocess.TimeoutExpired("aws", 30), OSError("offline"), 8])
        output = self.run_monitor(runtime, force_after=60)
        stops = runtime.stop_calls()
        self.assertEqual(stops[0][0], 0)
        forced = [(now, argv) for now, argv in stops if "--force" in argv]
        self.assertEqual(forced[0][0], 60)
        self.assertIn("--skip-os-shutdown", forced[0][1])
        self.assertIn("confirmed i-123abc: stopped", output)

    def test_stalled_successful_stop_escalates_at_default_interval(self):
        runtime = FakeRuntime(["running"] + ["stopping"] * 6 + ["stopped"])
        self.run_monitor(runtime)
        stops = runtime.stop_calls()
        self.assertEqual([now for now, _ in stops], [0, 30, 60, 90, 120, 150, 180])
        self.assertTrue(all("--force" not in argv for _, argv in stops[:-1]))
        self.assertEqual(stops[-1][1][-2:], ["--force", "--skip-os-shutdown"])

    def test_force_boundary_caps_poll_delay(self):
        runtime = FakeRuntime(["running", "stopping", "stopped"])
        self.run_monitor(runtime, force_after=5)
        self.assertEqual(runtime.stop_calls()[1][0], 5)
        self.assertIn("--force", runtime.stop_calls()[1][1])

    def test_forced_stop_failure_retries_until_terminated_confirmation(self):
        runtime = FakeRuntime(["running", "stopping", "stopping", "terminated"],
                              stops=["", 8, OSError("offline"), ""])
        output = self.run_monitor(runtime, force_after=5)
        self.assertEqual(len(runtime.stop_calls()), 5)
        self.assertTrue(all("--force" in argv for _, argv in runtime.stop_calls()[1:]))
        self.assertIn("confirmed i-123abc: terminated", output)

    def test_already_stopped_or_terminated_does_not_request_stop(self):
        for state in ("stopped", "terminated"):
            with self.subTest(state=state):
                runtime = FakeRuntime([state])
                self.run_monitor(runtime)
                self.assertEqual(runtime.stop_calls(), [])

    def test_live_runner_is_not_stopped_before_deadline(self):
        runtime = FakeRuntime(["running", "running", "terminated"], alive=True)
        self.run_monitor(runtime)
        self.assertEqual(runtime.stop_calls(), [])

    def test_stop_remains_latched_if_aliveness_changes(self):
        runtime = FakeRuntime(["running", "stopping", "stopped"], alive=lambda now: now > 0)
        self.run_monitor(runtime)
        self.assertEqual(len(runtime.stop_calls()), 2)

    def test_wrong_or_unavailable_identity_blocks_ec2_calls(self):
        for identity in ("arn:aws:iam::other:user/bench-runner", 9,
                         OSError("missing aws"), subprocess.TimeoutExpired("aws", 30)):
            with self.subTest(identity=identity):
                runtime = FakeRuntime([], identity=identity)
                with self.assertRaisesRegex(SystemExit, "AWS identity"):
                    self.run_monitor(runtime)
                self.assertLessEqual(len(runtime.calls), 3)
                self.assertTrue(all("get-caller-identity" in argv for _, argv, _ in runtime.calls))

    def test_invalid_configs_do_not_call_aws(self):
        for options in ({"minutes": 0}, {"force_after": -1}, {"force_after": float("nan")},
                        {"poll_seconds": 0}, {"command_timeout": float("inf")},
                        {"retry_attempts": 0}, {"retry_attempts": 11}, {"retry_backoff": 0}):
            with self.subTest(options=options):
                runtime = FakeRuntime([])
                with self.assertRaises(ValueError):
                    self.run_monitor(runtime, **options)
                self.assertEqual(runtime.calls, [])
        for pid in (0, -1):
            with self.assertRaises(ValueError):
                watchdog.validate_config(pid, 180, 180, 30, 30, 3, 2)

    def test_missing_instance_at_deadline_is_reported(self):
        runtime = FakeRuntime([], alive=True)
        with self.assertRaisesRegex(SystemExit, "without an instance ID"):
            self.run_monitor(runtime, text="", minutes=1)
        self.assertEqual(runtime.now, 60)
        self.assertEqual(runtime.stop_calls(), [])

    def test_runner_exit_without_instance_never_stops(self):
        runtime = FakeRuntime([])
        output = self.run_monitor(runtime, text="")
        self.assertIn("runner exited before an instance ID", output)
        self.assertEqual(runtime.stop_calls(), [])


if __name__ == "__main__":
    unittest.main()
