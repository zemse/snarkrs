import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import bench_external as bench
import render_machine


class ExternalVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.variant = self.root / "tiny"
        self.variant.mkdir()
        for name in ("circuit.zkey", "circuit.wtns", "vkey.json"):
            (self.variant / name).write_text("{}")
        self.cold = self.root / "cold"
        self.warm = self.root / "warm"
        self.oracle = self.root / "oracle"
        for path in (self.cold, self.warm, self.oracle):
            path.touch()
        self.csv = self.root / "external.csv"
        self.calls = []
        self.checks = []
        self.fail = ""

    def sh(self, cmd):
        self.calls.append(cmd)
        if cmd[0] == str(self.oracle):
            self.checks.append(Path(cmd[-1]).read_text())
            valid = self.fail != "verify"
            return subprocess.CompletedProcess(cmd, 0, "", "Valid proof" if valid else "invalid")
        warm = cmd[0] == str(self.warm)
        if self.fail == "exit":
            return subprocess.CompletedProcess(cmd, 1, "", "failed")
        proof, public = cmd[3:5]
        self.assertFalse(Path(proof).exists())
        self.assertFalse(Path(public).exists())
        if self.fail != "missing":
            Path(proof).write_text("final" if warm else "cold")
            if self.fail != "missing-public":
                Path(public).write_text("[]")
        stdout = "rep 1 1.25\nrep 2 2.5\n" if warm else ""
        if self.fail == "no-reps":
            stdout = "prepared\n"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    def run_main(self, cold=True, warm=True):
        args = ["bench_external.py", "--artifacts", str(self.root), "--variant", "tiny",
                "--csv", str(self.csv), "--reps", "2", "--skip-snarkjs",
                "--rapidsnark", str(self.cold) if cold else "",
                "--rapidsnark-warm", str(self.warm) if warm else "",
                "--rapidsnark-verify", str(self.oracle)]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(bench, "resolve_snarkjs", return_value=""), \
                mock.patch.object(bench, "sh", side_effect=self.sh), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return bench.main()

    def rows(self):
        with self.csv.open(newline="") as f:
            return list(csv.DictReader(f))

    def test_cold_checks_each_proof_and_records_oracle(self):
        self.assertEqual(self.run_main(warm=False), 0)
        rows = self.rows()
        self.assertEqual(self.checks, ["cold", "cold"])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["verified"], "yes")
            self.assertEqual(row["verification_scope"], "each-proof")
            self.assertEqual(row["verification_oracle"], "rapidsnark-verify")
        self.assertEqual([r["rep"] for r in rows], ["1", "2"])

    def test_warm_checks_only_final_proof_and_labels_all_timings(self):
        self.assertEqual(self.run_main(cold=False), 0)
        rows = self.rows()
        self.assertEqual(self.checks, ["final"])
        self.assertEqual([r["ms"] for r in rows], ["1.25", "2.5"])
        for row in rows:
            self.assertEqual(row["verified"], "yes")
            self.assertEqual(row["verification_scope"], "final-proof-only")
            self.assertEqual(row["verification_oracle"], "rapidsnark-verify")

    def test_failed_or_missing_proofs_record_nothing(self):
        for mode in ("cold", "warm"):
            for failure in ("exit", "verify", "missing", "missing-public"):
                with self.subTest(mode=mode, failure=failure):
                    self.fail = failure
                    self.assertEqual(self.run_main(cold=mode == "cold", warm=mode == "warm"), 0)
                    self.assertFalse(self.csv.exists())
        self.fail = "no-reps"
        self.assertEqual(self.run_main(cold=False), 0)
        self.assertFalse(self.csv.exists())

    def test_nonzero_oracle_exit_rejects_success_text(self):
        original = self.sh

        def sh(cmd):
            result = original(cmd)
            if cmd[0] == str(self.oracle):
                result.returncode = 1
            return result

        with mock.patch.object(self, "sh", side_effect=sh):
            self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.checks, ["cold", "final"])
        self.assertFalse(self.csv.exists())

    def test_later_cold_failure_discards_batch_and_clears_stale_outputs(self):
        calls = 0
        original = self.sh

        def sh(cmd):
            nonlocal calls
            if cmd[0] == str(self.cold):
                calls += 1
                if calls == 2:
                    self.fail = "missing"
            return original(cmd)

        with mock.patch.object(self, "sh", side_effect=sh):
            self.assertEqual(self.run_main(warm=False), 0)
        self.assertEqual(self.checks, ["cold"])
        self.assertFalse(self.csv.exists())

    def test_cold_rows_survive_failed_warm_without_false_warm_records(self):
        original = self.sh

        def sh(cmd):
            if cmd[0] == str(self.warm):
                self.fail = "missing"
            return original(cmd)

        with mock.patch.object(self, "sh", side_effect=sh):
            self.assertEqual(self.run_main(), 0)
        self.assertEqual([r["mode"] for r in self.rows()], ["cold", "cold"])

    def test_current_header_appends_and_readers_keep_warm_timings(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.run_main(), 0)
        rows = self.rows()
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(None not in r for r in rows))
        loaded = render_machine.load(str(self.root))
        self.assertEqual(len(loaded), 8)
        self.assertEqual(render_machine.med(loaded, prover="rapidsnark", mode="warm"), 1.875)

    def test_renderer_documents_scope_without_per_repetition_claim(self):
        self.assertEqual(self.run_main(), 0)
        out = self.root / "machine.md"
        args = ["render_machine.py", "--csv-dir", str(self.root), "--machine", "tiny",
                "--reps", "2", "--backends", "cpu", "--provers", "rapidsnark",
                "--out", str(out)]
        with mock.patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            render_machine.main()
        text = out.read_text()
        self.assertIn("only the final proof to pass, not every repetition", text)
        self.assertIn("missing metadata is unknown provenance", text)
        self.assertNotIn("Every proof behind every number", text)

    def test_selected_fallback_oracle_is_recorded_on_every_row(self):
        for name in ("snarkrs verify", "snarkjs verify"):
            with self.subTest(oracle=name):
                self.csv.unlink(missing_ok=True)
                verify = mock.Mock(return_value=True)
                with mock.patch.object(bench, "make_verifier", return_value=(verify, name)):
                    self.assertEqual(self.run_main(), 0)
                self.assertEqual(verify.call_count, 3)
                self.assertEqual({r["verification_oracle"] for r in self.rows()}, {name})

    def test_empty_file_gets_current_header(self):
        self.csv.touch()
        self.assertEqual(self.run_main(cold=False), 0)
        with self.csv.open(newline="") as f:
            self.assertEqual(next(csv.reader(f)), bench.FIELDS)

    def test_legacy_and_mismatched_headers_refused_unchanged(self):
        for fields in (bench.FIELDS[:-2], list(reversed(bench.FIELDS))):
            with self.subTest(fields=fields):
                with self.csv.open("w", newline="") as f:
                    csv.writer(f).writerow(fields)
                before = self.csv.read_bytes()
                self.assertEqual(self.run_main(), 1)
                self.assertEqual(self.csv.read_bytes(), before)
                self.assertEqual(self.calls, [])

    def test_optional_provers_and_missing_oracle_record_nothing(self):
        self.assertEqual(self.run_main(cold=False, warm=False), 0)
        self.assertFalse(self.csv.exists())
        self.oracle.unlink()
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.csv.exists())

    def test_oracle_fallback_names_and_order(self):
        with mock.patch.object(bench, "sh", return_value=subprocess.CompletedProcess([], 0, "OK", "")):
            verify, name = bench.make_verifier("", str(self.cold), "snarkjs")
            self.assertEqual(name, "snarkrs verify")
            self.assertTrue(verify("vkey", "public", "proof"))
            verify, name = bench.make_verifier("", "", "snarkjs")
            self.assertEqual(name, "snarkjs verify")
            self.assertTrue(verify("vkey", "public", "proof"))
        self.assertEqual(bench.make_verifier("", "", ""), (None, ""))


if __name__ == "__main__":
    unittest.main()
