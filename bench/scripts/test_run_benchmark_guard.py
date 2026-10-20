import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import foreign_cpu


SCRIPTS = Path(__file__).resolve().parent
RUN = SCRIPTS / "run-benchmark.sh"

# Answers `ps -o pgid= -p PID` with our own group (or PGID and PGID_EXIT when set) and each
# `ps -Ao pcpu=,pgid=` with the next foreign total from SAMPLES, plus a busy process in our
# own group that must not count.
PS = """#!/bin/bash
if [ "$1" = "-o" ]; then printf '%s\\n' "${PGID- 4242}"; exit "${PGID_EXIT:-0}"; fi
n=$(cat "$COUNT" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$COUNT"
echo "$(sed -n "${n}p" "$SAMPLES") 1"
echo "900.0 4242"
"""

# Stands in for python3 on the /proc path: prints the next line of SAMPLES as is, which may
# be empty or not a number, and exits with PY_EXIT.
PY = """#!/bin/bash
n=$(cat "$COUNT" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$COUNT"
sed -n "${n}p" "$SAMPLES"
exit "${PY_EXIT:-0}"
"""


QUIET = [5] * 13


class ContentionGuardTests(unittest.TestCase):
    def guard(self, samples, *args, status=0, proc=None, py_exit=None, pgid=None,
              pgid_exit=None):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / "ps").write_text(PS)
            (d / "ps").chmod(0o755)
            if py_exit is not None:
                (d / "python3").write_text(PY)
                (d / "python3").chmod(0o755)
                proc = d / "proc"
                proc.mkdir()
                (proc / "stat").write_text("cpu  1 2 3 4\n")
            (d / "samples").write_text("".join(f"{s}\n" for s in samples))
            env = dict(os.environ, PATH=f"{d}:{os.environ['PATH']}", COUNT=str(d / "count"),
                       SAMPLES=str(d / "samples"), G16_BASELINE_INTERVAL="0",
                       G16_PROC=str(proc or d / "no-proc"))
            env.pop("MAX_FOREIGN", None)
            if py_exit is not None:
                env["PY_EXIT"] = str(py_exit)
            if pgid is not None:
                env["PGID"] = pgid
            if pgid_exit is not None:
                env["PGID_EXIT"] = str(pgid_exit)
            result = subprocess.run(["bash", str(RUN), "--guard-only", *args], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, status, result.stderr)
            count = d / "count"
            calls = int(count.read_text()) if count.exists() else 0
            return result.stdout, calls, result.stderr

    def test_desktop_baseline_keeps_the_old_ceiling(self):
        out, calls, _ = self.guard([158, 70, 73, 59, 105] + [70] * 10 + [70])
        self.assertEqual(calls, 16)
        self.assertIn("baseline 70% of a core", out)
        self.assertIn("ceiling 200% = max(3x median, highest + 50) capped at 200", out)
        self.assertIn("stops on two checks in a row over it, 0s apart, measured as ps %cpu", out)

    def test_quiet_box_gets_half_a_core_over_its_highest_sample(self):
        out, _, _ = self.guard([5, 4, 6, 5, 3] + QUIET[:10] + [5])
        self.assertIn("baseline 5% of a core", out)
        self.assertIn("ceiling 56%", out)

    def test_low_median_desktop_clears_its_idle_spikes(self):
        out, _, _ = self.guard([159, 51, 26, 41, 27, 30, 35, 40, 41, 44, 38, 29, 33, 45, 50, 40])
        self.assertIn("baseline 40% of a core", out)
        self.assertIn("ceiling 200%", out)

    def test_one_wakeup_on_a_quiet_box_lifts_the_ceiling_over_it(self):
        out, _, _ = self.guard(QUIET + [6, 30, 5])
        self.assertIn("ceiling 80%", out)

    def test_moderate_baseline_scales_by_three(self):
        out, _, _ = self.guard([40] * 16)
        self.assertIn("ceiling 120%", out)

    def test_loaded_start_cannot_lift_the_ceiling(self):
        out, _, _ = self.guard([1100, 1000, 1200] + [1000] * 12 + [1000, 1000], status=3)
        self.assertIn("baseline 1000% of a core", out)
        self.assertIn("ceiling 200%", out)

    def test_explicit_ceiling_wins_without_sampling(self):
        out, calls, _ = self.guard([5, 5], "--max-foreign", "300")
        self.assertEqual(calls, 1)
        self.assertIn("ceiling 300% of a core, set explicitly; stops on two checks", out)

    def test_one_spike_over_the_ceiling_does_not_stop_the_run(self):
        _, calls, err = self.guard([150, 50], "--max-foreign", "100")
        self.assertEqual(calls, 2)
        self.assertIn("150% then 50% (ceiling 100%), a passing spike", err)

    def test_two_checks_in_a_row_over_the_ceiling_stop_the_run(self):
        _, calls, err = self.guard([150, 120], "--max-foreign", "100", status=3)
        self.assertEqual(calls, 2)
        self.assertIn("using 150% then 120% of a core (ceiling 100%)", err)

    def test_allow_loaded_records_two_breaches_and_carries_on(self):
        _, _, err = self.guard([150, 120], "--max-foreign", "100", "--allow-loaded")
        self.assertIn("WARNING: 150% then 120% of a core", err)

    def test_quiet_box_still_stops_on_a_held_core(self):
        _, _, err = self.guard(QUIET + [6, 5, 110, 105], status=3)
        self.assertIn("using 110% then 105% of a core (ceiling 56%)", err)

    def test_proc_is_used_when_it_exists(self):
        # A static /proc has no CPU deltas, so the busy foreign process ps would report is
        # never consulted and the baseline is zero.
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "stat").write_text("cpu  1 2 3 4\n")
            (proc / "77").mkdir()
            (proc / "77" / "stat").write_text(STAT.format(pgrp=1, utime=500, stime=20))
            out, calls, _ = self.guard([900] * 17, proc=proc)
        self.assertEqual(calls, 0)
        self.assertIn("baseline 0% of a core", out)
        self.assertIn("ceiling 50%", out)
        self.assertIn(f"measured as utime+stime deltas from {proc}", out)

    def test_an_empty_startup_reading_refuses_to_run(self):
        out, calls, err = self.guard(QUIET[:3] + [""] + QUIET, status=4, py_exit=0)
        self.assertEqual(calls, 4)
        self.assertNotIn("contention guard:", out)
        self.assertIn("could not measure foreign CPU at startup, got '' (exit 0)", err)

    def test_a_garbage_startup_reading_refuses_to_run(self):
        _, _, err = self.guard(["nan"] + QUIET, status=4, py_exit=0)
        self.assertIn("got 'nan' (exit 0) from utime+stime deltas", err)

    def test_a_failing_measurement_refuses_to_run_even_if_it_printed_a_number(self):
        _, calls, err = self.guard(["0"] + QUIET, status=4, py_exit=127)
        self.assertEqual(calls, 1)
        self.assertIn("got '0' (exit 127)", err)

    def test_an_empty_check_is_recorded_as_unmeasured_not_passed(self):
        _, calls, err = self.guard([""], "--max-foreign", "100", status=4, py_exit=0)
        self.assertEqual(calls, 1)
        self.assertIn("could not measure foreign CPU during: --guard-only, got ''", err)

    def test_a_garbage_confirming_reading_is_neither_a_breach_nor_a_pass(self):
        _, calls, err = self.guard(["150", "1.5e2"], "--max-foreign", "100", status=4,
                                   py_exit=0)
        self.assertEqual(calls, 2)
        self.assertNotIn("passing spike", err)
        self.assertIn("during: --guard-only, got '1.5e2'", err)

    def test_a_failing_process_group_lookup_refuses_to_run(self):
        out, calls, err = self.guard(QUIET, "--max-foreign", "100", status=4, pgid=" 4242",
                                     pgid_exit=1)
        self.assertEqual(calls, 0)
        self.assertEqual(out, "")
        self.assertIn("could not read this script's process group, got '4242' (exit 1)", err)

    def test_an_empty_process_group_refuses_to_run(self):
        _, calls, err = self.guard(QUIET, status=4, pgid="")
        self.assertEqual(calls, 0)
        self.assertIn("could not read this script's process group, got '' (exit 0)", err)


STAT = ("77 (tmux: server (1)) S 1 {pgrp} 77 0 -1 4194560 3020 0 0 0 {utime} {stime} 0 0 20 "
        "0 1 0 1234 9400320 1100 18446744073709551615 1 1 0 0 0 0 0 4096 81923 0 0 0 17 3 "
        "0 0 0 0 0\n")


class ProcStatTests(unittest.TestCase):
    def test_parse_skips_a_command_name_with_spaces_and_parentheses(self):
        self.assertEqual(foreign_cpu.parse_stat(STAT.format(pgrp=4242, utime=500, stime=20)),
                         (4242, 520))

    def test_foreign_percent_is_the_tick_delta_over_the_interval(self):
        before = {"10": (1, 1000), "11": (4242, 0), "12": (7, 50)}
        after = {"10": (1, 1150), "11": (4242, 5000), "12": (7, 50), "13": (9, 30)}
        # 150 ticks from the long-lived daemon, 30 from a process born inside the interval,
        # none from our own group, at 100 ticks a second over two seconds.
        self.assertEqual(foreign_cpu.foreign_percent(before, after, 4242, 100, 2.0), 90.0)

    def test_reused_pid_never_counts_negative(self):
        self.assertEqual(foreign_cpu.foreign_percent({"10": (1, 900)}, {"10": (1, 5)}, 4242,
                                                     100, 1.0), 0.0)

    def test_snapshot_reads_every_pid_and_ignores_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, pgrp in (("77", 1), ("78", 4242)):
                (proc / pid).mkdir()
                (proc / pid / "stat").write_text(STAT.format(pgrp=pgrp, utime=3, stime=4))
            (proc / "self").mkdir()
            (proc / "stat").write_text("cpu  1 2 3 4\n")
            (proc / "79").mkdir()
            self.assertEqual(foreign_cpu.snapshot(proc), {"77": (1, 7), "78": (4242, 7)})


if __name__ == "__main__":
    unittest.main()
