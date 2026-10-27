import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
import types
import unittest
from unittest import mock

AWS = Path(__file__).resolve().parents[1] / "aws"

# systemctl, pkill, fuser, sleep, dpkg and touch, logged. fuser reports the lock held for
# the first $BUSY calls.
MOCK = r'''#!/bin/bash
name=${0##*/}
echo "$name $*" >> "$MOCK_LOG"
case "$name" in
  fuser)
    n=$(grep -c '^fuser ' "$MOCK_LOG")
    [ "$n" -le "$BUSY" ]; exit ;;
esac
exit 0
'''


class QuietAptTests(unittest.TestCase):
    def run_quiet(self, busy):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in ("systemctl", "pkill", "fuser", "sleep", "dpkg", "touch"):
                (tools / name).write_text(MOCK)
                (tools / name).chmod(0o755)
            log = root / "calls.log"
            log.touch()
            env = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin", MOCK_LOG=str(log), BUSY=str(busy))
            result = subprocess.run(["bash", str(AWS / "quiet-apt.sh")], env=env,
                                    capture_output=True, text=True, timeout=30)
            return result, log.read_text().splitlines()

    def test_timers_are_stopped_and_masked_before_the_lock_wait(self):
        result, calls = self.run_quiet(busy=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        units = "apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service"
        self.assertEqual(calls[0], f"systemctl stop {units}")
        self.assertEqual(calls[1], f"systemctl mask {units}")
        self.assertEqual(calls[2], "pkill -TERM -f /usr/bin/unattended-upgrade")
        self.assertEqual(sum(c.startswith("sleep ") for c in calls), 2)
        self.assertEqual(calls[-2], "dpkg --configure -a")
        self.assertEqual(calls[-1], "touch /var/lib/g16-apt-quiet")

    def test_a_lock_that_never_frees_leaves_no_marker(self):
        result, calls = self.run_quiet(busy=10**6)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("still held after 900s", result.stderr)
        self.assertEqual(sum(c.startswith("sleep ") for c in calls), 180)
        self.assertFalse(any(c.startswith(("touch", "dpkg")) for c in calls))

    def test_every_bash_launcher_embeds_it_before_its_own_apt_work(self):
        for name in ("bench-machine.sh", "provision.sh", "rapidsnark-asm-probe.sh", "run-gpu-bench-safe.sh"):
            text = (AWS / name).read_text()
            ud = re.search(r'cat > "\$UD" <<EOF\n(.*?)\nEOF\n', text, re.S)
            self.assertIsNotNone(ud, name)
            body = ud.group(1)
            self.assertIn("\n$QUIET_APT\n", body, name)
            self.assertLess(body.index("shutdown -h +"), body.index("$QUIET_APT"), name)
            if "apt-get" in body:
                self.assertLess(body.index("$QUIET_APT"), body.index("apt-get"), name)
            assign = text.index('QUIET_APT="$(')
            self.assertLess(assign, text.index('cat > "$UD" <<EOF'), name)

    def test_matrix_user_data_carries_it(self):
        spec = importlib.util.spec_from_file_location("matrix", AWS / "matrix.py")
        matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix)
        calls = []

        def aws(*args, check=True):
            calls.append(args)
            if args[1] == "describe-instance-type-offerings":
                return {"InstanceTypeOfferings": [{"Location": "us-east-1a"}]}
            if args[1] == "describe-subnets":
                return {"Subnets": [{"AvailabilityZone": "us-east-1a", "SubnetId": "subnet-test"}]}
            if args[1] == "run-instances":
                return {"Instances": [{"InstanceId": "i-test"}]}
            raise AssertionError(args)

        box = matrix.Box("c7i.xlarge", Path(tempfile.gettempdir()))
        box.arch = "x86_64"
        args = types.SimpleNamespace(ami="ami-test", max_minutes=90, disk_gb=30)
        with mock.patch.object(matrix, "aws", aws), mock.patch.object(matrix, "say", lambda _: None):
            matrix.launch(box, args)
        run = next(c for c in calls if c[1] == "run-instances")
        ud = run[run.index("--user-data") + 1]
        self.assertTrue(ud.startswith("#!/bin/bash\nshutdown -h +90\n"))
        self.assertIn("systemctl mask apt-daily.timer apt-daily-upgrade.timer", ud)
        self.assertEqual(box.id, "i-test")


if __name__ == "__main__":
    unittest.main()
