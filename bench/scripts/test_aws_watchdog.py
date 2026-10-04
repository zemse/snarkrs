import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


AWS = Path(__file__).resolve().parents[1] / "aws"


class WatchdogTests(unittest.TestCase):
    def reason(self, uptime, now, last, done=0):
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; shutdown_reason "${@:2}"', "test",
             str(AWS / "idle-watchdog.sh"), str(uptime), str(now), str(last),
             str(done), "3600", "10800", "300"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    def test_active_lease(self):
        self.assertEqual(self.reason(7200, 10000, 9999), "")

    def test_idle_boundary(self):
        self.assertEqual(self.reason(4000, 10000, 6401), "")
        self.assertEqual(self.reason(4000, 10000, 6400), "idle")

    def test_hard_cap_even_with_heartbeat(self):
        self.assertEqual(self.reason(10800, 10000, 10000), "hard-limit")

    def test_completion_overrides_heartbeat_after_grace(self):
        self.assertEqual(self.reason(4000, 10000, 10000, 9701), "")
        self.assertEqual(self.reason(4000, 10000, 10000, 9700), "job-finished")


class RunnerTests(unittest.TestCase):
    def run_runner(self, failure="", variants=(), custom_root=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "bench/aws"
            scripts.mkdir(parents=True)
            for name in ("lib.sh", "machines.csv", "idle-watchdog.sh", "run-gpu-bench-safe.sh", "rapidsnark-oracle.sh"):
                shutil.copy(AWS / name, scripts / name)
            helpers = root / "bench/scripts"
            helpers.mkdir()
            for name in ("aws_stop_deadline.py", "watch_aws_bench.py"):
                shutil.copy(AWS.parent / "scripts" / name, helpers)
            for variant in variants:
                directory = root / "bench/artifacts/large" / variant
                directory.mkdir(parents=True)
                for name in ("circuit.zkey", "circuit.wtns", "vkey.json", "public.json", "r1cs-info.txt"):
                    if failure != "artifact" or name != "circuit.wtns":
                        (directory / name).write_text("fixture")
            tools = root / "tools"
            tools.mkdir()
            key = root / ".claude/skills/aws/keys/agent-us-east-1.pem"
            key.parent.mkdir(parents=True)
            key.touch()
            verifier = root / "bench/bin/rapidsnark-verify"
            verifier.parent.mkdir()
            verifier.touch()
            verifier.chmod(0o755)
            mock = '''#!/bin/bash
set -eu
name=${0##*/}
echo "$name $*" >> "$MOCK_LOG"
case "$name" in
  cargo) [[ "$FAILURE" != build ]] || exit 7 ;;
  python3)
    if [[ "$1" == - || "$1" == */aws_stop_deadline.py ]]; then exec "$REAL_PYTHON" "$@"; fi
    [[ "$FAILURE" != unready ]] || exit 5 ;;
  file)
    if [[ "$FAILURE" == verifier ]]; then echo 'Mach-O 64-bit arm64';
    else echo 'ELF 64-bit x86-64'; fi ;;
  git) echo abc123 ;;
  aws)
    case "$*" in
      *get-caller-identity*) echo '{"Arn":"arn:aws:iam::123456789012:user/bench-runner","Account":"123456789012"}' ;;
      *get-schedule-group*)
        [[ "$FAILURE" != group ]] || exit 9
        echo '{"Name":"default","State":"ACTIVE","Arn":"arn:aws:scheduler:us-east-1:123456789012:schedule-group/default"}' ;;
      *create-schedule*)
        [[ "$FAILURE" != scheduler ]] || exit 9
        "$REAL_PYTHON" -c 'import json,os,sys; a=sys.argv; r=json.loads(a[a.index("--cli-input-json")+1]); r["Arn"]="arn:aws:scheduler:us-east-1:123456789012:schedule/"+r["GroupName"]+"/"+r["Name"]; open(os.environ["MOCK_SCHEDULE"],"w").write(json.dumps(r)); print(json.dumps({"ScheduleArn":r["Arn"]}))' "$@" ;;
      *get-schedule*)
        [[ "$FAILURE" != verification ]] || exit 9
        cat "$MOCK_SCHEDULE" ;;
      *describe-images*) echo ami-test ;;
      *describe-security-groups*) echo sg-test ;;
      *run-instances*)
        for arg in "$@"; do
          case "$arg" in
            file://*)
              userdata=${arg#file://}
              bash -n "$userdata"
              grep -q 'shutdown -h +180' "$userdata"
              grep -q 'systemctl enable --now g16-watchdog.timer' "$userdata"
              grep -q "'3600' '10800' 300" "$userdata"
              ;;
          esac
        done
        echo i-0123456789abcdef0 ;;
      *stop-instances*) [[ "$FAILURE" != stop ]] || exit 9 ;;
      *'wait instance-stopped'*)
        [[ "$FAILURE" != stall ]] || grep -q 'stop-instances.*--force' "$MOCK_LOG" ;;
      *describe-instances*Owner*) echo bench-runner ;;
      *describe-instances*) echo 127.0.0.1 ;;
    esac ;;
  ssh)
    case "$*" in
      *'bash -s --'*) cat >/dev/null; [[ "$FAILURE" != job ]] || exit 8 ;;
    esac ;;
  scp) [[ "$FAILURE" != transfer ]] || exit 6 ;;
  sleep) /bin/sleep 0.05 ;;
esac
'''
            for name in ("cargo", "file", "git", "aws", "ssh", "scp", "rsync", "sleep", "python3", "nohup"):
                path = tools / name
                path.write_text(mock)
                path.chmod(0o755)
            log = root / "calls.log"
            log.touch()
            artifact_root = root / "bench/artifacts"
            if variants or custom_root:
                artifact_root /= "large"
            env = dict(os.environ, HOME=str(root), PATH=f"{tools}:/usr/bin:/bin",
                       MOCK_LOG=str(log), MOCK_SCHEDULE=str(root / "schedule.json"),
                       FAILURE=failure, REAL_PYTHON=sys.executable,
                       G16_AWS_STOP_ROLE_ARN=("" if failure == "role" else
                           "arn:aws:iam::123456789012:role/g16-emergency-stop"),
                       G16_AWS_SCHEDULER_GROUP="default",
                       G16_AWS_REGION="us-east-1", G16_AWS_TYPE="g4dn.xlarge",
                       G16_ARTIFACT_ROOT=str(artifact_root), G16_VERIFY_BIN=str(verifier))
            result = subprocess.run(
                ["bash", str(scripts / "run-gpu-bench-safe.sh"), "1", *variants],
                env=env, capture_output=True, text=True, timeout=15,
            )
            return result, log.read_text()

    def test_custom_root_without_variants_never_launches(self):
        result, calls = self.run_runner(custom_root=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("specify variants", result.stderr)
        self.assertNotIn("run-instances", calls)

    def test_incompatible_verifier_never_launches(self):
        result, calls = self.run_runner("verifier")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("run-instances", calls)
        self.assertIn("G16_VERIFY_BIN", result.stderr)

    def test_build_failure_never_launches(self):
        result, calls = self.run_runner("build")
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("run-instances", calls)

    def test_missing_role_fails_before_build_or_launch(self):
        result, calls = self.run_runner("role")
        self.assertEqual(result.returncode, 1)
        self.assertIn("G16_AWS_STOP_ROLE_ARN", result.stderr)
        self.assertNotIn("cargo ", calls)
        self.assertNotIn("run-instances", calls)
        self.assertNotIn("get-caller-identity", calls)

    def test_group_permission_failure_never_launches(self):
        result, calls = self.run_runner("group")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("cargo ", calls)
        self.assertNotIn("run-instances", calls)

    def test_scheduler_failure_stops_before_wait_running(self):
        for failure in ("scheduler", "verification"):
            with self.subTest(failure=failure):
                result, calls = self.run_runner(failure)
                self.assertEqual(result.returncode, 1)
                self.assertIn("stop-instances", calls)
                self.assertIn("wait instance-stopped", calls)
                self.assertNotIn("wait instance-running", calls)
                self.assertNotIn("delete-schedule", calls)

    def test_schedule_call_order(self):
        result, calls = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        operations = ("get-schedule-group", "cargo ", "run-instances",
                      "create-schedule", "scheduler get-schedule --", "wait instance-running")
        positions = [calls.index(operation) for operation in operations]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("delete-schedule", calls)

    def test_local_watchdog_uses_separate_launch_log(self):
        result, calls = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        invocation = next(line for line in calls.splitlines() if line.startswith("nohup "))
        self.assertIn("watch_aws_bench.py", invocation)
        self.assertIn("/aws-i-0123456789abcdef0/launch.log", invocation)
        self.assertIn("--minutes 180", invocation)
        self.assertLess(calls.index("scheduler get-schedule --"), calls.index("nohup "))

    def test_success_stops_and_confirms(self):
        result, calls = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(calls.index("cargo "), calls.index("run-instances"))
        self.assertIn("stop-instances", calls)
        self.assertIn("wait instance-stopped", calls)
        self.assertIn("--instance-initiated-shutdown-behavior stop", calls)
        self.assertIn("Key=Owner,Value=bench-runner", calls)

    def test_selected_large_artifacts_use_runtime_only_transfer(self):
        result, calls = self.run_runner(variants=("onion_p21",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("artifacts/large/onion_p21", calls)
        self.assertIn("--include circuit.zkey", calls)
        self.assertIn("--exclude *", calls)
        self.assertIn("rapidsnark-oracle.sh", calls)
        self.assertNotIn("csp/target-stock", calls)

    def test_missing_artifacts_never_launch(self):
        result, calls = self.run_runner("artifact", variants=("onion_p21",))
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing artifact", result.stderr)
        self.assertNotIn("run-instances", calls)

    def test_unready_onion_never_launches(self):
        result, calls = self.run_runner("unready", variants=("onion_p21",))
        self.assertEqual(result.returncode, 5)
        self.assertIn("check_onion_ready.py", calls)
        self.assertNotIn("run-instances", calls)

    def test_stalled_graceful_stop_uses_forced_fallback(self):
        result, calls = self.run_runner("stall")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--force --skip-os-shutdown", calls)
        self.assertIn("stopped after forced shutdown", result.stdout)

    def test_job_failure_preserves_exit_and_stops(self):
        result, calls = self.run_runner("job")
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertIn("stop-instances", calls)
        self.assertIn("bench/results/", calls)

    def test_transfer_failure_stops(self):
        result, calls = self.run_runner("transfer")
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertIn("stop-instances", calls)

    def test_stop_failure_is_not_success(self):
        result, calls = self.run_runner("stop")
        self.assertEqual(result.returncode, 1)
        self.assertIn("STOP NOT CONFIRMED", result.stderr)


if __name__ == "__main__":
    unittest.main()
