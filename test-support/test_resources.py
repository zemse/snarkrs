"""Artifact-free coverage guards, with environment changes confined to children."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="snarkrs-resources-")
        cls.root = Path(cls.temp.name)
        source = cls.root / "driver.rs"
        source.write_text(
            "mod resources { include!("
            + json.dumps(str(ROOT / "test-support/resources.rs"))
            + "); }\n"
            + """
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "artifact" => resources::skip("synthetic artifact test: missing circuit.zkey"),
        "vector" => resources::skip_vector("synthetic vector test: missing fft_13.json"),
        "device" => resources::skip("synthetic device test: no usable GPU/backend; provide a supported device"),
        "files" => println!("complete={}", resources::complete_files(
            std::path::Path::new(&args[2]), &["circuit.zkey", "circuit.wtns"], "synthetic fixture test")),
        _ => panic!("unknown probe"),
    }
}
"""
        )
        cls.driver = cls.root / "driver"
        subprocess.run(
            [shutil.which("rustc"), "--edition=2021", str(source), "-o", str(cls.driver)],
            check=True,
            timeout=120,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def child_env(self, strict=None, legacy=None):
        env = os.environ.copy()
        env.pop("G16_REQUIRE_TESTS", None)
        env.pop("G16_REQUIRE_VECTORS", None)
        if strict is not None:
            env["G16_REQUIRE_TESTS"] = strict
        if legacy is not None:
            env["G16_REQUIRE_VECTORS"] = legacy
        env["CARGO_NET_OFFLINE"] = "true"
        return env

    def probe(self, kind, strict=None, legacy=None, path=None):
        args = [str(self.driver), kind]
        if path is not None:
            args.append(str(path))
        return subprocess.run(
            args, env=self.child_env(strict, legacy), capture_output=True, text=True, timeout=120
        )

    def test_missing_artifact_modes(self):
        for strict in (None, "1", "0"):
            with self.subTest(strict=strict):
                result = self.probe("artifact", strict)
                self.assertEqual(result.returncode == 0, strict != "1")
                self.assertIn("synthetic artifact test: missing circuit.zkey", result.stderr)
                self.assertIn("G16_REQUIRE_TESTS=1", result.stderr)
                self.assertEqual("SKIPPED:" in result.stderr, strict != "1")

    def test_missing_vector_modes(self):
        for strict in (None, "1", "0"):
            with self.subTest(strict=strict):
                result = self.probe("vector", strict)
                self.assertEqual(result.returncode == 0, strict != "1")
                self.assertIn("missing fft_13.json", result.stderr)

    def test_legacy_vectors_remain_required(self):
        for strict in (None, "1", "0"):
            for legacy in ("", "1", "0"):
                with self.subTest(strict=strict, legacy=legacy):
                    result = self.probe("vector", strict, legacy)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("G16_REQUIRE_VECTORS is set", result.stderr)
                    self.assertIn("missing fft_13.json", result.stderr)

    def test_legacy_vectors_do_not_require_artifacts(self):
        result = self.probe("artifact", "0", "0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("SKIPPED:", result.stderr)

    def test_missing_device_modes(self):
        for strict in (None, "1", "0"):
            with self.subTest(strict=strict):
                result = self.probe("device", strict)
                self.assertEqual(result.returncode == 0, strict != "1")
                self.assertIn("provide a supported device", result.stderr)

    def test_missing_and_incomplete_files(self):
        missing = self.root / "absent-fixture"
        incomplete = self.root / "incomplete-fixture"
        incomplete.mkdir()
        (incomplete / "circuit.zkey").write_bytes(b"synthetic")
        for path, absent in ((missing, "circuit.zkey"), (incomplete, "circuit.wtns")):
            for strict in (None, "1", "0"):
                with self.subTest(path=path, strict=strict):
                    result = self.probe("files", strict, path=path)
                    self.assertEqual(result.returncode == 0, strict != "1")
                    self.assertIn("synthetic fixture test", result.stderr)
                    self.assertIn(str(path / absent), result.stderr)
                    if strict != "1":
                        self.assertIn("complete=false", result.stdout)

    def test_complete_files(self):
        complete = self.root / "complete-fixture"
        complete.mkdir()
        for name in ("circuit.zkey", "circuit.wtns"):
            (complete / name).write_bytes(b"synthetic")
        for strict in (None, "1", "0"):
            with self.subTest(strict=strict):
                result = self.probe("files", strict, legacy="0", path=complete)
                self.assertEqual(result.returncode, 0)
                self.assertIn("complete=true", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_actual_formats_export_entrypoint(self):
        cargo = shutil.which("cargo")
        for strict in (None, "1", "0"):
            with self.subTest(strict=strict):
                env = self.child_env(strict, legacy="0")
                env["SNARKJS"] = str(self.root / "absent-snarkjs")
                result = subprocess.run(
                    [cargo, "test", "--locked", "-p", "snarkrs-formats", "--test",
                     "snarkjs_exports", "zkey_export_json_matches_snarkjs", "--", "--exact", "--nocapture"],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
                )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode == 0, strict != "1", output)
                self.assertIn("running 1 test", output)
                self.assertIn("zkey_export_json_matches_snarkjs: snarkjs not found", output)
                self.assertIn("set SNARKJS to snarkjs 0.7.6", output)
                self.assertEqual("SKIPPED:" in output, strict != "1")


if __name__ == "__main__":
    unittest.main()
