import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
RUN = SCRIPTS / "run-benchmark.sh"

# Answers `ps -o pgid= -p PID` with our own group and each `ps -Ao pcpu=,pgid=` with the
# next foreign total from SAMPLES, plus a busy process in our own group that must not count.
PS = """#!/bin/bash
if [ "$1" = "-o" ]; then echo " 4242"; exit 0; fi
n=$(cat "$COUNT" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$COUNT"
echo "$(sed -n "${n}p" "$SAMPLES") 1"
echo "900.0 4242"
"""


class ContentionGuardTests(unittest.TestCase):
    def guard(self, samples, *args):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / "ps").write_text(PS)
            (d / "ps").chmod(0o755)
            (d / "samples").write_text("".join(f"{s}\n" for s in samples))
            env = dict(os.environ, PATH=f"{d}:{os.environ['PATH']}", COUNT=str(d / "count"),
                       SAMPLES=str(d / "samples"), G16_BASELINE_INTERVAL="0")
            env.pop("MAX_FOREIGN", None)
            result = subprocess.run(["bash", str(RUN), "--guard-only", *args], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            count = d / "count"
            calls = int(count.read_text()) if count.exists() else 0
            return result.stdout, calls

    def test_desktop_baseline_keeps_the_old_ceiling(self):
        out, calls = self.guard([158, 70, 73, 59, 105])
        self.assertEqual(calls, 5)
        self.assertIn("baseline 73% of a core", out)
        self.assertIn("ceiling 200%", out)

    def test_quiet_box_gets_half_a_core_over_its_baseline(self):
        out, _ = self.guard([5, 4, 6, 5, 30])
        self.assertIn("baseline 5% of a core", out)
        self.assertIn("ceiling 55%", out)

    def test_moderate_baseline_scales_by_three(self):
        out, _ = self.guard([40, 40, 40, 40, 40])
        self.assertIn("ceiling 120%", out)

    def test_loaded_start_cannot_lift_the_ceiling(self):
        out, _ = self.guard([1100, 1000, 1200, 1000, 1000])
        self.assertIn("baseline 1000% of a core", out)
        self.assertIn("ceiling 200%", out)

    def test_explicit_ceiling_wins_without_sampling(self):
        out, calls = self.guard([5, 5, 5, 5, 5], "--max-foreign", "300")
        self.assertEqual(calls, 0)
        self.assertIn("ceiling 300% of a core, set explicitly", out)


if __name__ == "__main__":
    unittest.main()
