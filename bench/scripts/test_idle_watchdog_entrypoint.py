import os
from pathlib import Path
import subprocess
import tempfile
import unittest


WATCHDOG = Path(__file__).resolve().parents[1] / "aws/idle-watchdog.sh"


class WatchdogEntrypointTests(unittest.TestCase):
    def run_watchdog(self, uptime, lease, done=None, config="3600 10800 300\n"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            (root / "config").write_text(config)
            (root / "lease").touch()
            (root / "uptime").write_text(f"{uptime}.00 0.00\n")
            if done is not None:
                (root / "done").touch()
            commands = {
                "date": 'echo 10000',
                "stat": 'case "$*" in */done) echo "$DONE" ;; *) echo "$LEASE" ;; esac',
                "logger": 'echo "$*" >> "$LOG"',
                "shutdown": 'echo "$*" >> "$SHUTDOWN_LOG"',
            }
            for name, body in commands.items():
                path = tools / name
                path.write_text(f"#!/bin/bash\nset -eu\n{body}\n")
                path.chmod(0o755)
            log = root / "events.log"
            shutdown = root / "shutdown.log"
            env = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin", LEASE=str(lease),
                       DONE=str(done or 0), LOG=str(log), SHUTDOWN_LOG=str(shutdown),
                       G16_WATCHDOG_STATE=str(root), G16_WATCHDOG_UPTIME=str(root / "uptime"),
                       G16_WATCHDOG_SHUTDOWN=str(tools / "shutdown"))
            result = subprocess.run(["bash", str(WATCHDOG)], env=env, capture_output=True,
                                    text=True, timeout=5)
            return (result, log.read_text() if log.exists() else "",
                    shutdown.read_text() if shutdown.exists() else "")

    def test_live_lease_does_not_shutdown(self):
        result, events, shutdown = self.run_watchdog(4000, 9999)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events, "")
        self.assertEqual(shutdown, "")

    def test_idle_fires_shutdown_command(self):
        result, events, shutdown = self.run_watchdog(4000, 6400)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("poweroff: idle", events)
        self.assertEqual(shutdown, "-h now\n")

    def test_hard_limit_fires_despite_live_lease(self):
        result, events, shutdown = self.run_watchdog(10800, 10000)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("poweroff: hard-limit", events)
        self.assertEqual(shutdown, "-h now\n")

    def test_finished_job_fires_despite_live_lease(self):
        result, events, shutdown = self.run_watchdog(4000, 10000, done=9700)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("poweroff: job-finished", events)
        self.assertEqual(shutdown, "-h now\n")

    def test_incomplete_configuration_rejected_without_shutdown(self):
        result, events, shutdown = self.run_watchdog(4000, 6400, config="3600 10800\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid watchdog configuration", result.stderr)
        self.assertEqual(events, "")
        self.assertEqual(shutdown, "")


if __name__ == "__main__":
    unittest.main()
