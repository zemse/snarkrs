import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUN = Path(__file__).resolve().parent / "run-benchmark.sh"


class RawEvidenceTests(unittest.TestCase):
    def prepare(self, out, raw="", status=0, cwd=None):
        source = RUN.read_text()
        setup = source.split("# ---------------------------------------------------------------- benchmark\n", 1)[1]
        setup = setup.split("for b in $BACKENDS; do\n", 1)[0]
        script = "set -eu\nlog() { :; }\n" + setup + """
printf 'sample\n' > "$TMP/cpu.csv"
printf '%s\n' "$TMP"
exit "$TEST_EXIT"
"""
        return subprocess.run(
            ["bash", "-c", script],
            env=dict(os.environ, OUT=str(out), RAW_DIR=str(raw), TEST_EXIT=str(status)),
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )

    def test_default_raw_directory_survives_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "results with spaces" / "machine.md"
            paths = []
            for status in (0, 17):
                with self.subTest(status=status):
                    result = self.prepare(out, status=status)
                    self.assertEqual(result.returncode, status, result.stderr)
                    path = Path(result.stdout.strip())
                    self.assertEqual(path.parent, out.parent)
                    self.assertTrue(path.name.startswith("machine.raw."))
                    self.assertEqual((path / "cpu.csv").read_text(), "sample\n")
                    paths.append(path)
            self.assertNotEqual(*paths)

    def test_explicit_raw_directory_survives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw evidence"
            result = self.prepare(root / "machine.md", raw, status=17)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), raw)
            self.assertEqual((raw / "cpu.csv").read_text(), "sample\n")

    def test_existing_raw_directory_is_refused_without_touching_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "existing"
            raw.mkdir()
            evidence = raw / "cpu.csv"
            evidence.write_text("previous run\n")
            result = self.prepare(root / "machine.md", raw)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertEqual(evidence.read_text(), "previous run\n")

    def test_dash_leading_raw_directory_is_a_path_not_an_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result = self.prepare(root / "machine.md", "--help", cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), root / "--help")
            self.assertEqual((root / "--help" / "cpu.csv").read_text(), "sample\n")

    def test_raw_directory_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "previous"
            target.mkdir()
            raw = root / "symlink"
            raw.symlink_to(target, target_is_directory=True)
            result = self.prepare(root / "machine.md", raw)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
