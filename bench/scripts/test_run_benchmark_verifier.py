import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUN = Path(__file__).resolve().parent / "run-benchmark.sh"


class BenchmarkVerifierTests(unittest.TestCase):
    def test_independent_verifier_is_passed_to_both_and_cold_process(self):
        source = RUN.read_text()
        # Run the actual dispatch loop with a recording CLI, without builds or proofs.
        loop = source.split("for b in $BACKENDS; do\n", 1)[1]
        loop = "for b in $BACKENDS; do\n" + loop.split("# The external provers", 1)[0]
        setup = """
set -eu
log() { :; }
check_load() { :; }
g16() { printf '%s\\0' CALL "$@" >> "$CALLS"; }
G16=g16
BACKENDS='cpu cuda'
READY='fixture'
REPS=2
"""
        for verifier in ["/independent verifier/snarkjs", ""]:
            with self.subTest(verifier=verifier), tempfile.TemporaryDirectory() as directory:
                calls_path = Path(directory) / "calls"
                env = dict(os.environ, CALLS=str(calls_path), SNARKJS=verifier,
                           TMP=directory, ARTIFACTS=f"{directory}/artifacts")
                result = subprocess.run(["bash", "-c", setup + loop], env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = calls_path.read_bytes().decode().split("CALL\0")[1:]
                self.assertEqual(len(calls), 4)
                for call, (backend, mode) in zip(calls, [
                    ("cpu", "both"), ("cpu", "cold-process"),
                    ("cuda", "both"), ("cuda", "cold-process"),
                ]):
                    args = call.rstrip("\0").split("\0")
                    suffix = "-process" if mode == "cold-process" else ""
                    expected = [
                        "bench", "--artifacts", env["ARTIFACTS"], "--variant", "fixture",
                        "--backend", backend, "--mode", mode, "--reps", "2",
                        "--csv", f"{directory}/{backend}-fixture{suffix}.csv",
                    ]
                    if verifier:
                        expected += ["--snarkjs", verifier]
                    self.assertEqual(args, expected)


if __name__ == "__main__":
    unittest.main()
