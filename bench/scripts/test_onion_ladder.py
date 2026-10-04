import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("ladder", Path(__file__).with_name("run-onion-ladder.py"))
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


class OnionLadderTests(unittest.TestCase):
    def rows(self):
        return [dict(variant=f"onion_p{power}", backend=backend, mode=mode, prover="ours",
                     verified="yes", rep=str(rep), ms="1.25", snarkjs_compatible="unknown",
                     independent_verification_scope="each-proof" if mode == "cold" else "first-proof")
                for power in range(21, 25) for backend in ("cpu", "cuda")
                for mode, count in (("cold", 3), ("warm", 5)) for rep in range(1, count + 1)]

    def validate(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows()[0]))
                writer.writeheader()
                writer.writerows(rows)
            return ladder.validate(path, [f"onion_p{p}" for p in range(21, 25)], 3, 5)

    def test_complete_ladder(self):
        summary = self.validate(self.rows())
        self.assertEqual(len(summary), 16)
        self.assertEqual(sum(s["count"] for s in summary), 64)

    def test_missing_duplicate_and_reordered_rows(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + [rows[0]], [rows[1], rows[0], *rows[2:]]):
            with self.assertRaises(RuntimeError):
                self.validate(invalid)

    def test_invalid_proof_timing_scope_and_compatibility(self):
        for field, value in (("verified", "no"), ("ms", "nan"), ("ms", "0"),
                             ("ms", "inf"), ("backend", "metal"), ("prover", "rapidsnark"),
                             ("independent_verification_scope", "first-proof"),
                             ("snarkjs_compatible", "yes")):
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                with self.assertRaises(RuntimeError):
                    self.validate(rows)


if __name__ == "__main__":
    unittest.main()
