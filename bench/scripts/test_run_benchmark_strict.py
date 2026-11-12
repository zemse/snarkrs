import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_validate_t4_benchmark as fixtures


SCRIPTS = Path(__file__).resolve().parent
RUN = SCRIPTS / "run-benchmark.sh"
CIRCUITS = "railgun-01x01 tornado sha256 railgun-13x01 rsa2048 keccak256 anon-aadhaar"


class StrictHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strict harness ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = RUN.read_text()
        self.env = dict(os.environ, STRICT_T4="1", EXPECTED_HOST="published-host",
                        LIVE_HOST="published-host", LIVE_OS="Linux", LIVE_ARCH="x86_64",
                        CORES="8", MACHINE="g4dn.2xlarge", BACKENDS="cpu cuda", EXTERNAL="1",
                        RAPIDSNARK="/fake tools/rapidsnark", RAPIDSNARK_WARM="/fake tools/warm",
                        RAPIDSNARK_VERIFY="/fake tools/verify", SNARKJS="/fake tools/snarkjs",
                        REPS="10", SNARKJS_REPS="3", CIRCUITS=CIRCUITS, ALLOW_LOADED="0",
                        LOADED_ANYWAY="0", UNMEASURED="", BENCH=str(SCRIPTS.parent),
                        OUT=str(self.root / "report with spaces.md"), RAW_DIR=str(self.root / "raw evidence"),
                        ARTIFACTS=str(self.root / "artifacts with spaces"), CALLS=str(self.root / "calls"),
                        PYTHON=sys.executable, PROVERS="rapidsnark snarkjs", FAIL_MODE="",
                        COMMIT="fixture", NOTE="", GUARD_DESC="fixture", MAX_FOREIGN="50")

    def shell(self, script, *args):
        return subprocess.run(["bash", "-c", "set -euo pipefail\n" + script, "strict-test", *args],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def preflight(self, parse_args=None):
        section = self.source.split("# ---------------------------------------------------------------- strict T4 preflight\n", 1)[1]
        section = section.split("# snarkjs never gets more reps", 1)[0]
        setup = """
log() { :; }
hostname() { echo "$LIVE_HOST"; }
uname() { if [ "$1" = -s ]; then echo "$LIVE_OS"; else echo "$LIVE_ARCH"; fi; }
"""
        if parse_args is not None:
            parser = self.source.split('REPS="${G16_REPS:-15}"', 1)[1].split('log() {', 1)[0]
            setup = 'REPS="${G16_REPS:-15}"' + parser + setup
        return self.shell(setup + section + '\necho PREFLIGHT_PASSED\n', *(parse_args or []))

    def test_preflight_requires_each_exact_condition(self):
        self.assertEqual(self.preflight().returncode, 0)
        cases = {"EXPECTED_HOST": ["", "wrong", "unknown"], "LIVE_HOST": ["different"],
                 "LIVE_OS": ["Darwin", "linux"], "LIVE_ARCH": ["arm64"], "CORES": ["4", "16"],
                 "MACHINE": ["g4dn.xlarge", "tesla-t4"], "BACKENDS": ["cpu", "cpu cuda metal"],
                 "EXTERNAL": ["0"], "RAPIDSNARK": [""], "RAPIDSNARK_WARM": [""],
                 "RAPIDSNARK_VERIFY": [""], "SNARKJS": [""], "REPS": ["9", "15"],
                 "SNARKJS_REPS": ["2", "10"], "ALLOW_LOADED": ["1"],
                 "CIRCUITS": ["sha256", CIRCUITS + " sha256", CIRCUITS.replace("tornado", "unknown"),
                              CIRCUITS.replace("tornado", "sha256")]}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    old = self.env[field]
                    self.env[field] = value
                    result = self.preflight()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("PREFLIGHT_PASSED", result.stdout)
                    self.env[field] = old

    def test_real_option_parser_and_preflight_order(self):
        self.env.update(G16_REPS="10", G16_CIRCUITS=CIRCUITS)
        good = ["--strict-t4", "--expected-host", "published-host"]
        self.assertEqual(self.preflight(good).returncode, 0)
        for args in (["--strict-t4"], good + ["--skip-external"], good + ["--allow-loaded"],
                     good + ["--reps", "2"], good + ["--snarkjs-reps", "12"]):
            with self.subTest(args=args):
                self.assertNotEqual(self.preflight(args).returncode, 0)
        self.assertLess(self.source.index("strict_t4_preflight\n"), self.source.index("# ---------------------------------------------------------------- build\n"))
        self.assertLess(self.source.index("strict_t4_preflight\n"), self.source.index("if [ \"$SNARKJS_REPS\" -gt"))
        # The ordinary parser still permits the old partial-friendly defaults.
        self.env.update(G16_REPS="15", G16_CIRCUITS="sha256", LIVE_HOST="other", CORES="4")
        self.assertEqual(self.preflight([]).returncode, 0)

    def test_missing_ready_artifacts_refuses_before_dispatch(self):
        section = self.source.split('log "artifacts"\n', 1)[1]
        section = section.split("# ---------------------------------------------------------------- benchmark\n", 1)[0]
        setup = """
log() { :; }
fetch() { [ "$1" != tornado ]; }
"""
        result = self.shell(setup + section + '\necho WOULD_TIME\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("WOULD_TIME", result.stdout)
        self.assertIn("missing READY artifacts for tornado", result.stderr)
        self.env["STRICT_T4"] = "0"
        self.assertEqual(self.shell(setup + section + '\necho WOULD_TIME\n').returncode, 0)

    def fixture(self, missing=False):
        fixture = fixtures.T4BenchmarkTests("test_complete_581_rows_63_cells_and_exact_medians")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        if missing:
            del fixture.rows[0][:10]
        fixture.save()
        self.env.update(FIXTURE_OURS=str(fixture.paths[0]), FIXTURE_EXTERNAL=str(fixture.paths[1]),
                        READY=CIRCUITS)

    def dispatch(self):
        section = self.source.split("# ---------------------------------------------------------------- benchmark\n", 1)[1]
        section = section.split("# ---------------------------------------------------------------- report\n", 1)[0]
        renderer = 'python3 "$BENCH/scripts/render_machine.py"' + self.source.split(
            'python3 "$BENCH/scripts/render_machine.py"', 1)[1]
        setup = """
log() { :; }
check_load() { :; }
g16() {
  printf '%s\\0' G16 "$@" >> "$CALLS"
  local mode="" csv=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --mode) mode="$2"; shift 2 ;;
      --csv) csv="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [ ! -e "$TMP/ours-written" ]; then
    cp "$FIXTURE_OURS" "$csv"; touch "$TMP/ours-written"
  fi
  [ "$FAIL_MODE" != "$mode" ] || return 17
}
G16=g16
python3() {
  printf '%s\\0' PYTHON "$@" >> "$CALLS"
  case "$1" in
    */bench_external.py)
      if [ ! -e "$TMP/external-written" ]; then
        cp "$FIXTURE_EXTERNAL" "$TMP/external-fixture.csv"; touch "$TMP/external-written"
      fi
      [ "$FAIL_MODE" != external ] || return 19 ;;
    */validate_t4_benchmark.py) command "$PYTHON" "$@" ;;
    */render_machine.py)
      shift
      while [ "$#" -gt 0 ]; do
        if [ "$1" = --out ]; then printf 'synthetic report\\n' > "$2"; return; fi
        shift
      done
      return 9 ;;
    *) return 99 ;;
  esac
}
"""
        return self.shell(setup + section + '\nGPU_NAME="fixture"\nCPU_NAME="fixture"\n' + renderer)

    def calls(self):
        path = Path(self.env["CALLS"])
        return path.read_bytes().decode() if path.exists() else ""

    def test_success_validates_separate_csvs_and_retains_summary_before_renderer(self):
        self.fixture()
        result = self.dispatch()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((Path(self.env["RAW_DIR"]) / "validation.json").read_text())
        self.assertEqual((summary["cell_count"], summary["row_count"]), (63, 581))
        self.assertTrue(Path(self.env["OUT"]).exists())
        calls = self.calls()
        self.assertLess(calls.index("validate_t4_benchmark.py"), calls.index("render_machine.py"))
        for value in (self.env["EXPECTED_HOST"], self.env["SNARKJS"], self.env["ARTIFACTS"],
                      str(Path(self.env["RAW_DIR"]) / "cpu-railgun-01x01.csv")):
            self.assertIn("\0" + value + "\0", calls)
        self.assertEqual(calls.count("\0bench\0"), 28)
        self.assertIn("\0".join(["--expected-os", "Linux", "--expected-arch", "x86_64",
                                  "--expected-cores", "8", ""]), calls)

    def test_incomplete_csv_never_invokes_renderer_and_retains_raw(self):
        self.fixture(missing=True)
        result = self.dispatch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete cell", result.stderr)
        self.assertNotIn("render_machine.py", self.calls())
        self.assertFalse(Path(self.env["OUT"]).exists())
        self.assertTrue(list(Path(self.env["RAW_DIR"]).glob("*.csv")))

    def test_each_prover_failure_is_immediate_and_raw_survives(self):
        self.fixture()
        for mode, code, g16_calls in (("both", 17, 1), ("cold-process", 17, 2), ("external", 19, 28)):
            with self.subTest(mode=mode):
                self.env.update(FAIL_MODE=mode, RAW_DIR=str(self.root / ("raw " + mode)),
                                CALLS=str(self.root / ("calls " + mode)))
                result = self.dispatch()
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(self.calls().count("\0bench\0"), g16_calls)
                self.assertNotIn("validate_t4_benchmark.py", self.calls())
                self.assertNotIn("render_machine.py", self.calls())
                self.assertFalse(Path(self.env["OUT"]).exists())
                self.assertTrue(list(Path(self.env["RAW_DIR"]).glob("*.csv")))

    def test_suspect_unchecked_or_allow_loaded_blocks_publication(self):
        self.fixture()
        for field, value in (("LOADED_ANYWAY", "1"), ("UNMEASURED", "fixture"), ("ALLOW_LOADED", "1")):
            with self.subTest(field=field):
                old = self.env[field]
                self.env.update({field: value}, RAW_DIR=str(self.root / field), CALLS=str(self.root / (field + " calls")))
                result = self.dispatch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SUSPECT/UNCHECKED", result.stderr)
                self.assertNotIn("render_machine.py", self.calls())
                self.assertFalse(Path(self.env["OUT"]).exists())
                self.env[field] = old

    def test_ordinary_mode_still_renders_partial_failed_runs_without_validation(self):
        self.fixture(missing=True)
        self.env.update(STRICT_T4="0", READY="fixture", FAIL_MODE="both", LOADED_ANYWAY="1", UNMEASURED="fixture")
        result = self.dispatch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("render_machine.py", self.calls())
        self.assertNotIn("validate_t4_benchmark.py", self.calls())
        self.assertTrue(Path(self.env["OUT"]).exists())


if __name__ == "__main__":
    unittest.main()
