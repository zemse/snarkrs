import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
AWS = ROOT / "bench/aws"


def executable(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
    path.chmod(0o755)


class CudaGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.log = self.root / "calls"
        self.env = dict(os.environ, PATH=f"{self.tools}:/usr/bin:/bin",
                        CALLS=str(self.log))

    def run_script(self, path, *args):
        return subprocess.run(["/bin/bash", str(path), *args], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def guest(self, mode="verify", verifier_rc=0):
        guest = self.root / "guest"
        guest.mkdir()
        script = guest / "cuda-gate-guest.sh"
        shutil.copy2(AWS / script.name, script)
        bins = guest / "bin"
        bins.mkdir()
        for name in ("lib", "adversarial_gpu", "pipeline_gpu", "field_gpu"):
            executable(bins / name, "exit 0")
        executable(self.tools / "timeout", 'shift; exec "$@"')
        executable(self.tools / "nvidia-smi", "echo stub-device")
        # A system Go must never be consulted, even if installed on the guest.
        executable(self.tools / "go", 'echo unexpected-system-go >> "$CALLS"; exit 99')
        executable(bins / "snarkrs", '''
if [[ $2 == prove && ${G16_CUDA_VRAM_LIMIT:-} == 1 ]]; then
  echo 'warning: VRAM refusal'
fi
''')
        executable(bins / "gnark_gpu", '''
echo "required=${G16_REQUIRE_TESTS:-}" >> "$CALLS"
case "$MODE" in
  skip) echo 'SKIPPED: no `go` on PATH'; exit 0 ;;
  omit) exit 0 ;;
  unsupported) go run . gen nowhere; exit $? ;;
  probe) go version; exit 0 ;;
esac
go version
go run . verify "$G16_GNARK_HARNESS/toy/vk.bin" public.bin proof.bin
if [[ $MODE == skip-after ]]; then echo 'SKIPPED: another fixture'; fi
''')
        executable(bins / "gnark-verifier", '''
printf 'oracle:%s\\n' "$*" >> "$CALLS"
exit "$VERIFIER_RC"
''')
        self.env.update(MODE=mode, VERIFIER_RC=str(verifier_rc))
        return script

    def test_guest_without_go_uses_shipped_oracle(self):
        script = self.guest()
        result = self.run_script(script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.log.read_text()
        self.assertIn("required=1\n", calls)
        self.assertIn("oracle:verify ", calls)
        self.assertIn("/gnark/toy/vk.bin public.bin proof.bin", calls)
        self.assertNotIn("unexpected-system-go", calls)
        self.assertIn("GATE PASSED", result.stdout)

    def test_guest_missing_verifier_fails_before_device_work(self):
        script = self.guest()
        (script.parent / "bin/gnark-verifier").unlink()
        result = self.run_script(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing bin/gnark-verifier", result.stderr)
        self.assertNotIn("stub-device", result.stdout)

    def test_guest_cannot_pass_without_successful_oracle(self):
        for mode, rc in (("skip", 0), ("omit", 0), ("probe", 0),
                         ("unsupported", 0), ("verify", 1), ("verify", 126),
                         ("skip-after", 0)):
            with self.subTest(mode=mode, rc=rc):
                # Each run has a fresh output directory, including its receipt.
                if (self.root / "guest").exists():
                    shutil.rmtree(self.root / "guest")
                result = self.run_script(self.guest(mode, rc))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("GATE FAILED", result.stdout)

    def test_guest_clears_previous_success_receipt(self):
        script = self.guest()
        result = self.run_script(script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.env["MODE"] = "omit"
        result = self.run_script(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gnark-oracle-not-run", result.stdout)

    def host(self):
        repo = self.root / "repo"
        (repo / "bench/aws").mkdir(parents=True)
        script = repo / "bench/aws/cuda-gate.sh"
        for name in ("cuda-gate.sh", "cuda-gate-guest.sh"):
            shutil.copy2(AWS / name, script.parent / name)
        target = repo / "target"
        target.mkdir()
        for circuit in ("tiny_mul", "sha256"):
            fixture = repo / "bench/artifacts" / circuit
            fixture.mkdir(parents=True)
            for name in ("circuit.zkey", "circuit.wtns", "vkey.json"):
                (fixture / name).touch()
        toy = repo / "test-support/gnark/toy"
        toy.mkdir(parents=True)
        (toy / "vk.bin").touch()
        for name in ("go.mod", "go.sum", "main.go"):
            (toy.parent / name).touch()
        messages = []
        for name in ("lib", "adversarial_gpu", "pipeline_gpu", "field_gpu", "gnark_gpu", "snarkrs"):
            binary = target / name
            executable(binary, "exit 0")
            messages.append(json.dumps({
                "reason": "compiler-artifact", "executable": str(binary),
                "target": {"name": name, "kind": ["lib" if name == "lib" else "bin"]},
                "profile": {"test": name != "snarkrs"},
            }))
        (target / "build.json").write_text("\n".join(messages) + "\n")
        executable(self.tools / "file", '''
if [[ ${BAD_ELF:-0} == 1 ]]; then echo 'Mach-O arm64'; else
  echo 'ELF 64-bit LSB executable, x86-64, statically linked'
fi
''')
        executable(self.tools / "git", "exit 0")
        executable(self.tools / "cargo", '''
echo cargo >> "$CALLS"
cat "$CARGO_TARGET_DIR/build.json.saved"
''')
        shutil.copy2(target / "build.json", target / "build.json.saved")
        executable(self.tools / "t4", 'echo "t4:$*" >> "$CALLS"')
        self.env.update(G16_T4=str(self.tools / "t4"), CARGO_TARGET_DIR=str(target),
                        G16_ARTIFACTS=str(repo / "bench/artifacts"))
        self.env.pop("G16_GNARK_VERIFIER", None)
        executable(self.tools / "go", '''
echo "go:$*:$GOOS:$GOARCH:$CGO_ENABLED:$GOTOOLCHAIN:$GOPROXY:$GOSUMDB" >> "$CALLS"
[[ $1 == build && $2 == -mod=readonly && $3 == -o && $5 == . ]]
printf '#!/bin/bash\\nexit 0\\n' > "$4"
chmod +x "$4"
''')
        return script, target

    def test_host_cross_builds_locally_offline_and_ships_binary_not_sources(self):
        script, target = self.host()
        result = self.run_script(script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.log.read_text()
        self.assertIn("go:build -mod=readonly -o ", calls)
        self.assertIn(":linux:amd64:0:local:off:off", calls)
        self.assertTrue((target / "ship/bin/gnark-verifier").is_file())
        self.assertTrue((target / "ship/gnark/toy/vk.bin").is_file())
        self.assertFalse((target / "ship/gnark/main.go").exists())
        self.assertIn("t4:run ", calls)

    def test_host_no_build_requires_existing_verifier_before_t4(self):
        script, target = self.host()
        result = self.run_script(script, "--no-build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing executable", result.stderr)
        self.assertFalse(self.log.exists())
        executable(target / "gnark-verifier", "exit 0")
        result = self.run_script(script, "--no-build")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.log.read_text()
        self.assertNotIn("go:", calls)
        self.assertNotIn("cargo", calls)

    def test_host_explicit_verifier_is_not_rebuilt(self):
        script, target = self.host()
        verifier = self.root / "prebuilt verifier"
        executable(verifier, "exit 0")
        self.env["G16_GNARK_VERIFIER"] = str(verifier)
        result = self.run_script(script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("go:", self.log.read_text())
        self.assertEqual((target / "ship/bin/gnark-verifier").read_bytes(), verifier.read_bytes())

    def test_host_rejects_wrong_binary_format_before_t4(self):
        script, target = self.host()
        executable(target / "gnark-verifier", "exit 0")
        self.env["BAD_ELF"] = "1"
        result = self.run_script(script, "--no-build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("static Linux x86-64 ELF", result.stderr)
        self.assertFalse(self.log.exists())

    def test_host_failed_go_build_does_not_run_t4_or_reuse_stale_binary(self):
        script, target = self.host()
        executable(target / "gnark-verifier", "exit 0")
        executable(self.tools / "go", "exit 1")
        result = self.run_script(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
