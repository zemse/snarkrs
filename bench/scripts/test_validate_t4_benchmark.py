import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import bench_external
import validate_t4_benchmark as validator


OURS = "host,os,arch,cores,variant,constraints,prover,backend,mode,rep,ms,prepare_ms,gather_us,ntt_us,pointwise_us,msm_us,assemble_us,verified,adx_bmi2,msm_a_us,msm_b_g2_us,msm_b_g1_us,msm_l_us,msm_h_us,msm_serial".split(",")
EXTERNAL = OURS[:18] + ["verification_scope", "verification_oracle"]
CIRCUITS = [("railgun-01x01", 20135), ("tornado", 28275), ("sha256", 59281),
            ("railgun-13x01", 141276), ("rsa2048", 190945),
            ("keccak256", 239176), ("anon-aadhaar", 1115080)]


class T4BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root / "ours.csv", self.root / "external.csv"]
        self.stamps = dict(expected_host="published-host", expected_os="Linux",
                           expected_arch="x86_64", expected_cores=8)
        self.rows = [[], []]
        for variant, constraints in CIRCUITS:
            cells = [("ours", backend, mode, 10)
                     for backend in ("cpu", "cuda")
                     for mode in ("warm", "cold-inprocess", "cold-process")]
            cells += [("rapidsnark", "cpu", "warm", 10),
                      ("rapidsnark", "cpu", "cold", 10), ("snarkjs", "cpu", "cold", 3)]
            for prover, backend, mode, reps in cells:
                for rep in range(1, reps + 1):
                    ours = prover == "ours"
                    header = OURS if ours else EXTERNAL
                    row = dict.fromkeys(header, "")
                    row.update(host="published-host", os="Linux", arch="x86_64", cores="8",
                               variant=variant, constraints=str(constraints), prover=prover,
                               backend=backend, mode=mode, rep=str(rep), ms=str(rep), verified="yes")
                    if ours:
                        row["adx_bmi2"] = "yes"
                        if mode != "cold-process":
                            row.update(prepare_ms="0", gather_us="0", ntt_us="1", pointwise_us="2",
                                       msm_us="3", assemble_us="4", msm_serial="no")
                    else:
                        row.update(verification_scope="each-proof" if mode == "cold" else "final-proof-only",
                                   verification_oracle="rapidsnark-verify")
                    self.rows[0 if ours else 1].append(row)

    def save(self):
        for path, rows, header in zip(self.paths, self.rows, (OURS, EXTERNAL)):
            with path.open("w", newline="") as dest:
                writer = csv.DictWriter(dest, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

    def validate(self):
        self.save()
        return validator.validate(self.paths, **self.stamps)

    def test_complete_581_rows_63_cells_and_exact_medians(self):
        summary = self.validate()
        self.assertEqual([len(rows) for rows in self.rows], [420, 161])
        self.assertEqual(summary["row_count"], 581)
        self.assertEqual(summary["cell_count"], 63)
        self.assertEqual(summary["validation"], "csv-only")
        self.assertIn("not full benchmark readiness", summary["limitations"])
        for cell in summary["cells"]:
            snarkjs = cell["prover"] == "snarkjs"
            self.assertEqual(cell["count"], 3 if snarkjs else 10)
            self.assertEqual(cell["median_ms"], 2 if snarkjs else 5.5)
        for rows in self.rows:
            rows.reverse()
        self.save()
        self.assertEqual(summary, validator.validate(list(reversed(self.paths)), **self.stamps))

    def test_finite_large_timings_keep_summary_finite(self):
        for rows in self.rows:
            for row in rows:
                row["ms"] = "1e308"
        summary = self.validate()
        self.assertTrue(all(cell["median_ms"] == 1e308 for cell in summary["cells"]))
        json.dumps(summary, allow_nan=False)

    def test_headers_match_actual_producers(self):
        self.assertEqual(EXTERNAL, bench_external.FIELDS)
        rust = (Path(__file__).resolve().parents[2] / "bin/snarkrs/src/bench.rs").read_text()
        self.assertIn('pub const CSV_HEADER: &str = "' + ",".join(OURS) + '";', rust)

    def test_missing_cell_and_wrong_row_counts(self):
        for kind in ("cell", "one", "extra", "duplicate-instead-of-last"):
            with self.subTest(kind=kind):
                original = self.rows[0][:]
                if kind == "cell":
                    del self.rows[0][:10]
                elif kind == "one":
                    self.rows[0].pop()
                elif kind == "extra":
                    self.rows[0].append(self.rows[0][0].copy())
                else:
                    self.rows[0][-1] = self.rows[0][-2].copy()
                with self.assertRaises(ValueError):
                    self.validate()
                self.rows[0] = original

    def test_invalid_existing_fields(self):
        cases = {
            "host": ["different-host", ""], "os": ["linux", "Darwin"],
            "arch": ["arm64"], "cores": ["4", "08", ""],
            "variant": ["extra-circuit"], "constraints": ["20136", "-1", "20135.0"],
            "backend": ["metal", "auto", "constant-work"], "mode": ["cold", "both"],
            "prover": ["snarkjs", "unknown"], "verified": ["no", "na", ""],
            "rep": ["0", "11", "1.0", "01", ""],
            "ms": ["NaN", "inf", "-inf", "-1", "0", "", "garbage"],
            "adx_bmi2": ["no", ""], "msm_serial": ["yes", ""],
            "prepare_ms": ["NaN", "inf", "-1"],
            "gather_us": ["NaN", "inf", "-1"],
            "msm_a_us": ["0", "1", "NaN"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    old = self.rows[0][0][field]
                    self.rows[0][0][field] = value
                    with self.assertRaises(ValueError):
                        self.validate()
                    self.rows[0][0][field] = old

    def test_every_optional_stage_rejects_nonfinite_or_negative(self):
        for rows, index in ((self.rows[0], 0), (self.rows[1], 0)):
            for field in OURS[11:17]:
                for value in ("NaN", "inf", "-inf", "-0.1"):
                    with self.subTest(prover=rows[index]["prover"], field=field, value=value):
                        old = rows[index][field]
                        rows[index][field] = value
                        with self.assertRaises(ValueError):
                            self.validate()
                        rows[index][field] = old

    def test_optional_stages_may_be_empty_or_nonnegative(self):
        self.rows[0][0]["gather_us"] = ""
        self.rows[1][0]["prepare_ms"] = "0.125"
        self.assertEqual(self.validate()["row_count"], 581)

    def test_cold_process_preparation_stages_and_serial_must_be_empty(self):
        row = next(r for r in self.rows[0] if r["mode"] == "cold-process")
        for field in OURS[11:17] + OURS[19:]:
            with self.subTest(field=field):
                row[field] = "no" if field == "msm_serial" else "0"
                with self.assertRaises(ValueError):
                    self.validate()
                row[field] = ""

    def test_external_scope_oracle_backend_mode_rep_and_producer(self):
        for mode in ("warm", "cold"):
            row = next(r for r in self.rows[1] if r["mode"] == mode)
            for field, value in (("verification_scope", "each-proof" if mode == "warm" else "final-proof-only"),
                                 ("verification_scope", ""), ("verification_oracle", "unknown"),
                                 ("verification_oracle", ""), ("backend", "cuda"),
                                 ("mode", "cold-inprocess"), ("prover", "ours")):
                with self.subTest(mode=mode, field=field, value=value):
                    old = row[field]
                    row[field] = value
                    with self.assertRaises(ValueError):
                        self.validate()
                    row[field] = old
        row = next(r for r in self.rows[1] if r["prover"] == "snarkjs")
        row["rep"] = "4"
        with self.assertRaises(ValueError):
            self.validate()

    def test_known_oracles_are_accepted_without_claiming_independence(self):
        for oracle in ("rapidsnark-verify", "snarkrs verify", "snarkjs verify"):
            for row in self.rows[1]:
                row["verification_oracle"] = oracle
            self.assertEqual(self.validate()["row_count"], 581)

    def test_independent_expected_stamps_required_and_t4_specific(self):
        for field, value in (("expected_host", ""), ("expected_host", "unknown"),
                             ("expected_host", "other"), ("expected_os", "linux"),
                             ("expected_arch", "arm64"), ("expected_cores", 4)):
            with self.subTest(field=field):
                old = self.stamps[field]
                self.stamps[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.stamps[field] = old
        for field in self.stamps:
            stamps = self.stamps.copy()
            del stamps[field]
            with self.assertRaises(TypeError):
                validator.validate(self.paths, **stamps)

    def test_exact_schema_field_counts_legacy_and_heterogeneous_concat(self):
        self.save()
        original = self.paths[0].read_text()
        lines = original.splitlines(keepends=True)
        mutations = ["", lines[0], original.replace(",msm_serial", "", 1),
                     original.replace("host,os", "os,host", 1),
                     original.replace("host,os", "timestamp,host,os", 1),
                     lines[0] + lines[1].rstrip("\n") + ",extra\n" + "".join(lines[2:]),
                     lines[0] + ",".join(lines[1].split(",")[:-1]) + "\n" + "".join(lines[2:]),
                     original + self.paths[1].read_text(), original + "\n"]
        for content in mutations:
            with self.subTest(content=content[:60]):
                self.paths[0].write_text(content)
                with self.assertRaises(ValueError):
                    validator.validate(self.paths, **self.stamps)
        self.paths[0].write_text(original)
        self.paths[1].write_text(",".join(EXTERNAL[:-2]) + "\n")
        with self.assertRaises(ValueError):
            validator.validate(self.paths, **self.stamps)

    def test_cli_requires_stamps_and_emits_no_summary_on_failure(self):
        self.save()
        args = [str(p) for p in self.paths]
        for option in ([], ["--expected-host", "published-host"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validator.main(args + option)
        options = ["--expected-host", "published-host", "--expected-os", "Linux",
                   "--expected-arch", "x86_64", "--expected-cores", "8"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(validator.main(args + options), 0)
        self.assertEqual(json.loads(output.getvalue())["row_count"], 581)
        self.rows[0].pop()
        self.save()
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(validator.main(args + options), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("incomplete cell", error.getvalue())


if __name__ == "__main__":
    unittest.main()
