import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
CHECK = SCRIPTS / "check-witness.sh"
TORNADO = SCRIPTS.parent / "artifacts/tornado"


def real_snarkjs():
    return shutil.which("snarkjs") or os.path.expanduser("~/Library/pnpm/snarkjs")


class CheckWitnessTests(unittest.TestCase):
    def run_check(self, body):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "snarkjs"
            stub.write_text(f"#!/bin/bash\n{body}\n")
            stub.chmod(0o755)
            return subprocess.run(["bash", str(CHECK), "tornado", "c.r1cs", "c.wtns"],
                                  env=dict(os.environ, SNARKJS=str(stub)),
                                  capture_output=True, text=True, timeout=10)

    def test_correct_witness_passes(self):
        result = self.run_check('[ "$*" = "wtns check c.r1cs c.wtns" ] || exit 9\n'
                                'echo "WITNESS IS CORRECT"')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_check_aborts_naming_the_circuit(self):
        result = self.run_check('echo "WITNESS IS NOT CORRECT"; exit 1')
        self.assertEqual(result.returncode, 1)
        self.assertIn("tornado: c.wtns does not satisfy c.r1cs", result.stderr)
        self.assertIn("WITNESS IS NOT CORRECT", result.stderr)

    def test_success_exit_without_the_verdict_aborts(self):
        result = self.run_check("echo 'WITNESS IS NOT CORRECT'; exit 0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("tornado:", result.stderr)

    def test_missing_snarkjs_aborts(self):
        result = subprocess.run(["bash", str(CHECK), "tornado", "c.r1cs", "c.wtns"],
                                env=dict(os.environ, SNARKJS="/nonexistent/snarkjs"),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("tornado:", result.stderr)

    @unittest.skipUnless((TORNADO / "circuit.wtns").is_file() and os.access(real_snarkjs(), os.X_OK),
                         "needs bench/artifacts/tornado and snarkjs")
    def test_tornado_artifact_passes_and_a_flipped_bit_fails(self):
        env = dict(os.environ, SNARKJS=real_snarkjs())
        good = subprocess.run(["bash", str(CHECK), "tornado", str(TORNADO / "circuit.r1cs"),
                               str(TORNADO / "circuit.wtns")], env=env, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(good.returncode, 0, good.stderr)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "circuit.wtns"
            data = bytearray((TORNADO / "circuit.wtns").read_bytes())
            data[len(data) - 32 * 100] ^= 1
            bad.write_bytes(bytes(data))
            result = subprocess.run(["bash", str(CHECK), "tornado", str(TORNADO / "circuit.r1cs"),
                                     str(bad)], env=env, capture_output=True, text=True,
                                    timeout=120)
        self.assertEqual(result.returncode, 1)
        self.assertIn("tornado: ", result.stderr)
        self.assertIn("does not satisfy", result.stderr)


class GenTornadoTests(unittest.TestCase):
    def run_gen(self, env):
        return subprocess.run(["bash", str(SCRIPTS / "gen-tornado.sh")], env=env,
                              capture_output=True, text=True, timeout=10)

    def test_requires_circom1(self):
        env = {k: v for k, v in os.environ.items() if k != "CIRCOM1"}
        result = self.run_gen(env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CIRCOM1", result.stderr)

    def test_refuses_circom_2(self):
        with tempfile.TemporaryDirectory() as directory:
            circom = Path(directory) / "circom"
            circom.write_text("#!/bin/bash\necho 'circom compiler 2.2.3'\n")
            circom.chmod(0o755)
            result = self.run_gen(dict(os.environ, CIRCOM1=str(circom)))
        self.assertEqual(result.returncode, 1)
        self.assertIn("need circom 0.5.x", result.stderr)


if __name__ == "__main__":
    unittest.main()
