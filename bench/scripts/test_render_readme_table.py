import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import render_readme_table as renderer


class ReadmeTables(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        sources = self.root / "machines"
        sources.mkdir()
        self.source = sources / "machine.md"
        self.source.write_text(
            "# test-machine\n\nCommit `abc123`.\n\n## warm\n\n"
            "| circuit | constraints | gpu-snark cpu | gpu-snark metal | "
            "gpu-snark cuda | rapidsnark | snarkjs |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
            "| `small` | 1,234 | 10.0 | 2.0 |  | - |  |\n"
        )
        self.table = renderer.readme_table(
            renderer.parse_machine_file(self.source)["modes"]["warm"])
        self.readme = self.root / "README.md"
        self.write_readme(self.table)

    def write_readme(self, table):
        self.readme.write_text(
            "# project\n\n## benchmarks\n\n### test-machine\n\n"
            "**warm**:\n\n" + table + "\n\n"
            "**trusted setup**:\n\n| cpu | metal |\n| 10 | 20 |\n\n"
            "## using it\n\nUnrelated prose.\n")

    def check(self, files=None):
        with contextlib.redirect_stdout(io.StringIO()):
            renderer.check_readme(self.readme, files or [self.source])

    def test_matching_table(self):
        self.check()
        self.assertIn("g16 metal (ms)", self.table)
        self.assertIn('<td align="right">1,234</td><td align="right">2.0</td>'
                      '<td align="right">10.0</td><td align="right">-</td>'
                      '<td align="right"></td>', self.table)

    def test_stale_timing(self):
        self.write_readme(self.table.replace(">2.0<", ">1.0<"))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.check()

    def test_stale_constraint_count(self):
        self.write_readme(self.table.replace("1,234", "1,233"))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.check()

    def test_missing_row(self):
        self.write_readme("\n".join(line for line in self.table.splitlines()
                                    if "<code>small</code>" not in line))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.check()

    def test_wrong_backend_label(self):
        self.write_readme(self.table.replace("g16 metal", "g16 cuda"))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.check()

    def test_blank_is_not_failure(self):
        self.write_readme(self.table.replace('<td align="right"></td>',
                                             '<td align="right">-</td>'))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.check()

    def test_missing_machine(self):
        self.readme.write_text("## benchmarks\n\n## using it\n")
        with self.assertRaisesRegex(ValueError, "machines differ"):
            self.check()

    def test_duplicate_sources(self):
        with self.assertRaisesRegex(ValueError, "multiple source records"):
            self.check([self.source, self.source])

    def test_suspect_source_cannot_support_table(self):
        self.source.write_text(self.source.read_text() + "\nSUSPECT\n")
        with self.assertRaisesRegex(ValueError, "no non-suspect"):
            self.check()

    def test_source_without_warm_rows(self):
        self.source.write_text(self.source.read_text().replace("## warm", "## cold"))
        with self.assertRaisesRegex(ValueError, "no warm"):
            self.check()

    def test_cuda_table(self):
        self.source.write_text(self.source.read_text().replace("| 2.0 |  |", "|  | 2.0 |"))
        self.write_readme(self.table.replace("g16 metal", "g16 cuda"))
        self.check()

    def test_ambiguous_gpu_columns(self):
        self.source.write_text(self.source.read_text().replace("| 2.0 |  |", "| 2.0 | 3.0 |"))
        with self.assertRaisesRegex(ValueError, "exactly one measured GPU"):
            self.check()

    def test_missing_table(self):
        self.write_readme("")
        with self.assertRaisesRegex(ValueError, "expected one warm"):
            self.check()

    def test_extra_table(self):
        self.write_readme(self.table + "\n" + self.table)
        with self.assertRaisesRegex(ValueError, "expected one warm"):
            self.check()

    def test_missing_benchmarks_section(self):
        self.readme.write_text("# project\n")
        with self.assertRaisesRegex(ValueError, "no benchmarks section"):
            self.check()

    def test_cli_success_without_writing_readme(self):
        before = self.readme.read_bytes()
        result = subprocess.run(
            [sys.executable, renderer.__file__, "--dir", str(self.source.parent),
             "--check-readme", str(self.readme)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 README warm tables match", result.stdout)
        self.assertEqual(self.readme.read_bytes(), before)

    def test_cli_exits_nonzero_without_writing_stale_readme(self):
        self.write_readme(self.table.replace(">2.0<", ">1.0<"))
        before = self.readme.read_bytes()
        result = subprocess.run(
            [sys.executable, renderer.__file__, "--dir", str(self.source.parent),
             "--check-readme", str(self.readme)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale", result.stderr)
        self.assertEqual(self.readme.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
