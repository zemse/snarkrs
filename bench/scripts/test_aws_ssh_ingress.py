import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

AWS = Path(__file__).resolve().parents[1] / "aws"

MOCK_AWS = r'''#!/bin/bash
echo "aws $*" >> "$MOCK_LOG"
case "$*" in
  *get-caller-identity*) echo 'arn:aws:iam::123456789012:user/bench-runner' ;;
  *describe-key-pairs*) ;;
  *describe-vpcs*) echo vpc-test ;;
  *create-security-group*) echo sg-test ;;
  *authorize-security-group-ingress*)
    [[ "$AUTHORIZE" != dup ]] || { echo 'InvalidPermission.Duplicate' >&2; exit 254; }
    echo sgr-0123456789abcdef0 ;;
  *revoke-security-group-ingress*) [[ "$REVOKE" != fail ]] || exit 254 ;;
  *describe-images*) echo ami-test ;;
  *run-instances*) [[ "$RUNFAIL" != 1 ]] || { echo 'UnauthorizedOperation' >&2; exit 254; }; echo i-0123456789abcdef0 ;;
esac
exit 0
'''

MOCK_CURL = '#!/bin/bash\nprintf "%s\\n" "$CHECKIP"\n'

# Same shape as the cleanup traps in bench-machine.sh and rapidsnark-asm-probe.sh.
DRIVER = r'''
set -euo pipefail
source "$ROOT/bench/aws/lib.sh"
cleanup() {
  local code=$?
  aws_ ec2 terminate-instances --instance-ids i-test >/dev/null 2>&1 || true
  revoke_ssh_ingress
  exit $code
}
trap cleanup EXIT INT TERM
SG="$(ensure_key_and_sg)"
open_ssh_ingress "$SG"
exit "${DRIVER_EXIT:-0}"
'''


class SshIngressTests(unittest.TestCase):
    def run_script(self, script, authorize="ok", revoke="ok", checkip="203.0.113.7",
                   driver_exit="0", runfail="0", held=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bench/aws").mkdir(parents=True)
            for name in ("lib.sh", "machines.csv", "bench-machine.sh"):
                shutil.copy(AWS / name, root / "bench/aws" / name)
            tools = root / "tools"
            tools.mkdir()
            (tools / "aws").write_text(MOCK_AWS)
            (tools / "curl").write_text(MOCK_CURL)
            (tools / "sleep").write_text("#!/bin/bash\n")
            for tool in tools.iterdir():
                tool.chmod(0o755)
            driver = root / "driver.sh"
            driver.write_text(DRIVER)
            log = root / "calls.log"
            log.touch()
            env = dict(os.environ, HOME=str(root), ROOT=str(root), MOCK_LOG=str(log),
                       PATH=f"{tools}:{Path(sys.executable).parent}:/usr/bin:/bin",
                       AUTHORIZE=authorize, REVOKE=revoke, CHECKIP=checkip,
                       DRIVER_EXIT=driver_exit, RUNFAIL=runfail, G16_SSH_INGRESS_HELD=held)
            target = driver if script == "driver" else root / "bench/aws" / script[0]
            args = [] if script == "driver" else script[1:]
            result = subprocess.run(["bash", str(target), *args], env=env,
                                    capture_output=True, text=True, timeout=30)
            calls = log.read_text().splitlines()
            return result, calls

    @staticmethod
    def count(calls, word):
        return sum(1 for c in calls if word in c)

    def test_authorize_asks_for_rule_id_and_only_the_host_32(self):
        _, calls = self.run_script("driver")
        auth = [c for c in calls if "authorize-security-group-ingress" in c]
        self.assertEqual(len(auth), 1)
        self.assertIn("--cidr 203.0.113.7/32", auth[0])
        self.assertIn("SecurityGroupRuleId", auth[0])

    def test_revoked_after_normal_run(self):
        result, calls = self.run_script("driver")
        self.assertEqual(result.returncode, 0)
        revoke = [c for c in calls if "revoke-security-group-ingress" in c]
        self.assertEqual(len(revoke), 1)
        self.assertIn("--group-id sg-test", revoke[0])
        self.assertIn("--security-group-rule-ids sgr-0123456789abcdef0", revoke[0])

    def test_revoked_after_failing_run_after_terminate_and_exit_code_kept(self):
        result, calls = self.run_script("driver", driver_exit="7")
        self.assertEqual(result.returncode, 7)
        term = next(i for i, c in enumerate(calls) if "terminate-instances" in c)
        rev = next(i for i, c in enumerate(calls) if "revoke-security-group-ingress" in c)
        self.assertLess(term, rev)

    def test_revoke_failure_does_not_mask_exit_status(self):
        ok, _ = self.run_script("driver", revoke="fail")
        self.assertEqual(ok.returncode, 0)
        bad, calls = self.run_script("driver", revoke="fail", driver_exit="7")
        self.assertEqual(bad.returncode, 7)
        self.assertEqual(self.count(calls, "terminate-instances"), 1)

    def test_preexisting_rule_is_not_revoked(self):
        result, calls = self.run_script("driver", authorize="dup")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.count(calls, "authorize-security-group-ingress"), 1)
        self.assertEqual(self.count(calls, "revoke-security-group-ingress"), 0)

    def test_non_ipv4_checkip_never_opens_a_rule(self):
        for bad in ("", "<html>blocked</html>", "2001:db8::1", "999.1.1.1", "0.0.0.0/0", "1.2.3"):
            result, calls = self.run_script("driver", checkip=bad)
            self.assertNotEqual(result.returncode, 0, bad)
            self.assertEqual(self.count(calls, "authorize-security-group-ingress"), 0, bad)
            self.assertEqual(self.count(calls, "revoke-security-group-ingress"), 0, bad)

    def test_held_rule_is_neither_opened_nor_revoked(self):
        result, calls = self.run_script("driver", held="1")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.count(calls, "authorize-security-group-ingress"), 0)
        self.assertEqual(self.count(calls, "revoke-security-group-ingress"), 0)

    def test_bench_machine_revokes_when_launch_fails(self):
        result, calls = self.run_script(("bench-machine.sh", "c7i.xlarge", "1"), runfail="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.count(calls, "authorize-security-group-ingress"), 1)
        self.assertEqual(self.count(calls, "revoke-security-group-ingress"), 1)

    def test_bench_machine_keeps_a_rule_it_does_not_own(self):
        result, calls = self.run_script(("bench-machine.sh", "c7i.xlarge", "1"),
                                        runfail="1", authorize="dup")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.count(calls, "revoke-security-group-ingress"), 0)


if __name__ == "__main__":
    unittest.main()
