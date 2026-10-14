import contextlib
import csv
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

import render_machine

HEADER = ["variant", "constraints", "prover", "backend", "mode", "rep", "ms", "verified"]


class ColdModes(unittest.TestCase):
    def render(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "ours.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(HEADER)
                for i, (mode, ms) in enumerate(rows, 1):
                    w.writerow(["v", "8", "ours", "cpu", mode, i, ms, "yes"])
            out = root / "machine.md"
            args = ["render_machine.py", "--csv-dir", str(root), "--machine", "m",
                    "--reps", "2", "--backends", "cpu", "--provers", "", "--out", str(out)]
            with mock.patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                render_machine.main()
            return out.read_text()

    def cell(self, text, section):
        body = re.search(rf"^## {section}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S).group(1)
        row = next(l for l in body.splitlines() if l.startswith("| `v`"))
        return row.split("|")[3].strip()

    def test_cold_column_is_the_fresh_process_and_inprocess_gets_its_own_section(self):
        text = self.render([("cold-process", 10), ("cold-process", 12),
                            ("cold-inprocess", 5), ("warm", 1)])
        self.assertEqual(self.cell(text, "cold"), "11.0")
        self.assertEqual(self.cell(text, "cold-inprocess"), "5.0")
        self.assertEqual(self.cell(text, "warm"), "1.0")

    def test_csv_from_before_the_split_renders_as_it_did(self):
        text = self.render([("cold", 7), ("warm", 1)])
        self.assertEqual(self.cell(text, "cold"), "7.0")
        self.assertNotIn("## cold-inprocess", text)


if __name__ == "__main__":
    unittest.main()
