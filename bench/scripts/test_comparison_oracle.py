import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comparison", SCRIPTS / "run-comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonOracleTests(unittest.TestCase):
    def run_comparison(self, failure="", cold=1, warm=1, fail_fast=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "bench/artifacts"
            variant = artifacts / "oracle_test"
            variant.mkdir(parents=True)
            (variant / "r1cs-info.txt").write_text("# of Constraints: 10\n")
            binary = root / "target/release/snarkrs"
            binary.parent.mkdir(parents=True)
            binary.touch()
            oracle = root / "rapidsnark-oracle"
            csv_path = root / "result.csv"
            commands = []
            warm_path = Path("/tmp/g16bench_warm_oracle_test_cpu.csv")
            warm_path.write_text("stale\n")

            def sh(cmd):
                commands.append(cmd)
                if "bench" in cmd:
                    self.assertFalse(warm_path.exists())
                    if failure == "oracle":
                        return subprocess.CompletedProcess(cmd, 1, "", "oracle rejected proof")
                    if failure != "missing":
                        with warm_path.open("w") as f:
                            f.write("variant,prover,backend,mode,rep,ms,verified\n")
                            if failure != "empty":
                                verified = "no" if failure == "unverified" else "yes"
                                for rep in range(1, warm + 1):
                                    f.write(f"oracle_test,ours,cpu,warm,{rep},1.25,{verified}\n")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            args = ["run-comparison.py", "--variants", "oracle_test", "--reps", "1",
                    "--backends", "cpu", "--csv", str(csv_path), "--skip-rapidsnark",
                    "--warm-oracle", str(oracle), "--cold-reps", str(cold),
                    "--warm-reps", str(warm)]
            if fail_fast:
                args.append("--fail-fast")
            status = 0
            try:
                with mock.patch.object(sys, "argv", args), \
                        mock.patch.object(comparison, "HERE", root / "bench"), \
                        mock.patch.object(comparison, "ART", artifacts), \
                        mock.patch.object(comparison, "BIN", root / "bin"), \
                        mock.patch.object(comparison, "detect_machine", return_value="fixture"), \
                        mock.patch.object(comparison, "detect_gpu", return_value="fixture"), \
                        mock.patch.object(comparison, "load_average", return_value=0), \
                        mock.patch.object(comparison, "HAVE_SNARKJS", False), \
                        mock.patch.object(comparison, "time_cold", return_value=([2.5] * cold, None)) as cold_call, \
                        mock.patch.object(comparison, "verify_snarkjs", return_value=None), \
                        mock.patch.object(comparison, "sh", side_effect=sh), \
                        contextlib.redirect_stdout(io.StringIO()):
                    comparison.main()
                    self.assertEqual(cold_call.call_args.args[1], cold)
            except SystemExit as error:
                status = error.code
            finally:
                warm_path.unlink(missing_ok=True)
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            return status, rows, commands

    def test_fail_fast_retains_completed_rows(self):
        status, rows, commands = self.run_comparison("oracle", fail_fast=True)
        self.assertIn("oracle rejected proof", status)
        self.assertEqual([r["mode"] for r in rows], ["cold"])
        self.assertEqual(len(commands), 1)

    def test_distinct_cold_and_warm_repetitions(self):
        status, rows, commands = self.run_comparison(cold=3, warm=5)
        self.assertEqual(status, 0)
        self.assertEqual([r["rep"] for r in rows if r["mode"] == "cold"], ["1", "2", "3"])
        self.assertEqual([r["rep"] for r in rows if r["mode"] == "warm"], ["1", "2", "3", "4", "5"])
        self.assertEqual(commands[0][commands[0].index("--reps") + 1], "5")

    def test_warm_foreign_oracle_is_called_and_scope_recorded(self):
        status, rows, commands = self.run_comparison()
        self.assertEqual(status, 0)
        self.assertIn("--snarkjs", commands[0])
        warm = next(row for row in rows if row["mode"] == "warm")
        self.assertEqual(warm["independent_verifier"], "rapidsnark-oracle")
        self.assertEqual(warm["independent_verification_scope"], "first-proof")
        self.assertNotEqual(warm.get("snarkjs_compatible"), "yes")

    def test_oracle_rejection_fails_run_and_retains_partial_results(self):
        status, rows, _ = self.run_comparison("oracle")
        self.assertEqual(status, 1)
        self.assertEqual([row["mode"] for row in rows], ["cold"])

    def test_warm_rows_reject_invalid_timings_and_configuration(self):
        row = dict(variant="tiny", backend="cuda", prover="ours", mode="warm",
                   verified="yes", rep="1", ms="1.25")
        self.assertTrue(comparison.valid_warm_rows([row], "tiny", "cuda", 1))
        for field, value in (("ms", ""), ("ms", "nan"), ("ms", "inf"), ("ms", "-1"),
                             ("variant", "other"), ("backend", "cpu"), ("mode", "cold"),
                             ("prover", "other"), ("rep", "2")):
            with self.subTest(field=field, value=value):
                invalid = dict(row, **{field: value})
                self.assertFalse(comparison.valid_warm_rows([invalid], "tiny", "cuda", 1))
        missing = dict(row)
        del missing["ms"]
        self.assertFalse(comparison.valid_warm_rows([missing], "tiny", "cuda", 1))

    def test_cold_proof_for_different_public_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public.json").write_text('["123"]')
            generated = root / "generated.json"
            def sh(cmd):
                generated.write_text('["124"]')
                return subprocess.CompletedProcess(cmd, 0, "", "")
            with mock.patch.object(comparison, "sh", side_effect=sh), \
                    mock.patch.object(comparison, "verify_fast") as verify:
                times, error = comparison.time_cold(["prover"], 1, root / "vkey.json",
                                                    root / "proof.json", generated)
            self.assertIsNone(times)
            self.assertIn("differ", error)
            verify.assert_not_called()

    def test_verifier_failure_status_overrides_success_marker(self):
        result = subprocess.CompletedProcess([], 1, "Result: Valid proof", "")
        with mock.patch.object(comparison, "sh", return_value=result):
            self.assertFalse(comparison.verify_fast("vk", "public", "proof"))
        result = subprocess.CompletedProcess([], 1, "OK!", "")
        with mock.patch.object(comparison, "sh", return_value=result), \
                mock.patch.object(comparison, "HAVE_SNARKJS", True):
            self.assertFalse(comparison.verify_snarkjs("vk", "public", "proof"))

    def test_missing_empty_and_unverified_csv_are_failures(self):
        for failure in ("missing", "empty", "unverified"):
            with self.subTest(failure=failure):
                status, rows, _ = self.run_comparison(failure)
                self.assertEqual(status, 1)
                self.assertEqual([row["mode"] for row in rows], ["cold"])


class RapidsnarkAdapterTests(unittest.TestCase):
    def run_adapter(self, output, code=0, stderr=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "rapidsnark-oracle"
            shutil.copy(SCRIPTS.parent / "aws/rapidsnark-oracle.sh", adapter)
            shutil.copy(SCRIPTS / "check_reference_public.py", root / "check_reference_public.py")
            (root / "public.json").write_text('["123"]')
            (root / "generated.json").write_text('[123]')
            verifier = root / "rapidsnark-verify"
            redirect = " >&2" if stderr else ""
            verifier.write_text(f"#!/bin/bash\nprintf '%s\\n' '{output}'{redirect}\nexit {code}\n")
            verifier.chmod(0o755)
            return subprocess.run(["bash", str(adapter), "groth16", "verify", str(root / "vk"), str(root / "generated.json"), "proof"],
                                  capture_output=True, text=True)

    def test_valid_proof_emits_required_marker(self):
        result = self.run_adapter("Result: Valid proof")
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK!", result.stdout)

    def test_valid_proof_on_stderr_emits_required_marker(self):
        result = self.run_adapter("Result: Valid proof", stderr=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK!", result.stdout)

    def test_invalid_proof_or_failed_process_is_rejected(self):
        for output, code in (("Result: Invalid proof", 0), ("Result: Valid proof", 1)):
            result = self.run_adapter(output, code)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("OK!", result.stdout)


if __name__ == "__main__":
    unittest.main()
