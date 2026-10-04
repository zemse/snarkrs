import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import aws_stop_deadline as deadline


REGION = "us-east-1"
ROLE = "arn:aws:iam::123456789012:role/bench/g16-stop"
INSTANCE = "i-0123456789abcdef0"
LAUNCHED = "2026-07-01T23:30:00Z"
NOW = datetime(2026, 7, 1, 23, 31, tzinfo=timezone.utc)


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.request = None
        self.mutate = lambda value: value
        self.identity = {"Arn": deadline.EXPECTED_IDENTITY, "Account": deadline.ACCOUNT}
        self.group = {"Name": "default", "State": "ACTIVE",
                      "Arn": f"arn:aws:scheduler:{REGION}:{deadline.ACCOUNT}:schedule-group/default"}

    def command(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if "get-caller-identity" in argv:
            output = self.identity
        elif "get-schedule-group" in argv:
            output = self.group
        elif "create-schedule" in argv:
            self.request = json.loads(argv[argv.index("--cli-input-json") + 1])
            self.arn = f"arn:aws:scheduler:{REGION}:{deadline.ACCOUNT}:schedule/default/{self.request['Name']}"
            output = {"ScheduleArn": self.arn}
        elif "get-schedule" in argv:
            output = copy.deepcopy(self.request)
            output["Arn"] = self.arn
            output = self.mutate(output)
        else:
            self.fail(f"unexpected AWS call: {argv}")
        return subprocess.CompletedProcess(argv, 0, json.dumps(output), "")

    def arm(self, directory, **kwargs):
        return deadline.arm(REGION, ROLE, "default", 180, INSTANCE, LAUNCHED,
                            directory, command=self.command, now=lambda: NOW, **kwargs)

    def test_request_and_verified_records(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = self.arm(directory)
            self.assertEqual(self.request["ScheduleExpression"], "at(2026-07-02T02:30:00)")
            self.assertEqual(actual["ScheduleExpressionTimezone"], "UTC")
            self.assertEqual(actual["FlexibleTimeWindow"], {"Mode": "OFF"})
            self.assertEqual(actual["ActionAfterCompletion"], "DELETE")
            self.assertEqual(actual["State"], "ENABLED")
            target = actual["Target"]
            self.assertEqual(target["Arn"], deadline.TARGET)
            self.assertEqual(target["RoleArn"], ROLE)
            self.assertEqual(target["RetryPolicy"], {
                "MaximumEventAgeInSeconds": 86400, "MaximumRetryAttempts": 185})
            self.assertEqual(json.loads(target["Input"]), {
                "InstanceIds": [INSTANCE], "Force": True, "SkipOsShutdown": True})
            self.assertEqual(json.loads((Path(directory) / "stop-deadline-request.json").read_text()), self.request)
            self.assertTrue((Path(directory) / "stop-deadline-create.json").exists())
            self.assertTrue((Path(directory) / "stop-deadline-observed.json").exists())
            for argv, kwargs in self.calls:
                self.assertEqual(kwargs["timeout"], 30)
                self.assertEqual(kwargs["env"]["AWS_MAX_ATTEMPTS"], "2")
                self.assertEqual(kwargs["env"]["AWS_RETRY_MODE"], "standard")
                self.assertIn("--cli-connect-timeout", argv)
                self.assertIn("--cli-read-timeout", argv)
            self.assertFalse(any("delete-schedule" in argv for argv, _ in self.calls))

    def test_empty_successful_aws_response(self):
        def command(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.assertEqual(deadline.aws_json(REGION, ["scheduler", "delete-schedule"], command=command), {})

    def test_read_only_preflight(self):
        deadline.preflight(REGION, ROLE, "default", 180, command=self.command)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("get-schedule-group", self.calls[1][0])

    def test_default_group_without_arn(self):
        self.group.pop("Arn")
        deadline.preflight(REGION, ROLE, "default", 180, command=self.command)
        self.group["Name"] = "custom"
        with self.assertRaisesRegex(ValueError, "group"):
            deadline.preflight(REGION, ROLE, "custom", 180, command=self.command)

    def test_identity_guard(self):
        for identity in ({"Arn": deadline.EXPECTED_IDENTITY, "Account": "000000000000"},
                         {"Arn": "arn:aws:iam::123456789012:user/other", "Account": deadline.ACCOUNT}):
            with self.subTest(identity=identity):
                self.identity = identity
                self.calls.clear()
                with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "identity"):
                    self.arm(directory)
                self.assertEqual(len(self.calls), 1)
                self.assertIsNone(self.request)

    def test_group_guard(self):
        for key, value in (("Name", "other"), ("State", "DELETING"), ("Arn", "wrong")):
            with self.subTest(key=key):
                saved = self.group.copy()
                self.group[key] = value
                with self.assertRaisesRegex(ValueError, "group"):
                    deadline.preflight(REGION, ROLE, "default", 180, command=self.command)
                self.group = saved

    def test_verification_fails_closed(self):
        mutations = {
            "ScheduleExpression": "at(2026-07-03T02:30:00)",
            "ScheduleExpressionTimezone": "America/New_York",
            "FlexibleTimeWindow": {"Mode": "FLEXIBLE", "MaximumWindowInMinutes": 15},
            "ActionAfterCompletion": "NONE", "State": "DISABLED",
            "Name": "other", "GroupName": "other", "Arn": "wrong",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                self.mutate = lambda result: dict(result, **{key: value})
                with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    self.arm(directory)
        for key, value in (("Arn", "wrong"), ("RoleArn", "wrong"),
                           ("RetryPolicy", {}), ("Input", json.dumps({"InstanceIds": [INSTANCE]}))):
            with self.subTest(target_key=key):
                def mutate(result):
                    result["Target"][key] = value
                    return result
                self.mutate = mutate
                with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    self.arm(directory)

    def test_timeout_and_command_failure_propagate(self):
        for failure in ("timeout", "error"):
            for operation in ("get-caller-identity", "create-schedule", "get-schedule"):
                with self.subTest(failure=failure, operation=operation):
                    def command(argv, **kwargs):
                        if operation in argv:
                            if failure == "timeout":
                                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                            return subprocess.CompletedProcess(argv, 9, "", "denied")
                        return self.command(argv, **kwargs)
                    with tempfile.TemporaryDirectory() as directory, self.assertRaises(
                            subprocess.TimeoutExpired if failure == "timeout" else RuntimeError):
                        deadline.arm(REGION, ROLE, "default", 180, INSTANCE, LAUNCHED,
                                     directory, command=command, now=lambda: NOW)

    def test_timestamp_validation(self):
        self.assertEqual(deadline.deadline_time("2026-12-31T23:59:00Z", 2),
                         datetime(2027, 1, 1, 0, 1, tzinfo=timezone.utc))
        for timestamp in ("2026-07-01T23:30:00", "2026-07-01T23:30:00+00:00",
                          "2026-02-30T00:00:00Z", "2026-07-02T00:00:00Z",
                          "2026-07-01T00:00:00Z"):
            with self.subTest(timestamp=timestamp):
                self.calls.clear()
                with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    deadline.arm(REGION, ROLE, "default", 180, INSTANCE, timestamp,
                                 directory, command=self.command, now=lambda: NOW)
                self.assertEqual(self.calls, [])

    def test_invalid_configs_make_no_aws_calls(self):
        configs = (("region", "us-gov-west-1"), ("region", "us-east-1;echo x"),
                   ("role", ""), ("role", ROLE.replace("123456789012", "000000000000")),
                   ("role", "arn:aws:iam::123456789012:user/foo"),
                   ("role", ROLE + "/*"), ("group", "bad/group"), ("group", "x" * 65),
                   ("minutes", 0), ("minutes", 1441), ("minutes", True),
                   ("instance", "i-test"), ("instance", "--help"))
        for key, value in configs:
            with self.subTest(key=key, value=value):
                config = dict(region=REGION, role=ROLE, group="default", minutes=180, instance=INSTANCE)
                config[key] = value
                with self.assertRaises(ValueError):
                    deadline.validate_config(**config)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
