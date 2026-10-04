import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import revalidate_onion as revalidation


generator = revalidation.generator


class FakeRunner:
    def __init__(self, fail="", mutate=False):
        self.fail = fail
        self.mutate = mutate

    def command(self, command, log, directory, required=None):
        name = log.stem
        if name == self.fail:
            raise RuntimeError("verification failed")
        if name == "witness":
            shutil.copy(directory / "circuit.wtns", command[-1])
        elif name == "vkey":
            shutil.copy(directory / "vkey.json", command[-1])
        elif name == "prove":
            (directory / "revalidation/proof.json").write_text("fresh proof")
            shutil.copy(directory / "public.json", directory / "revalidation/public.json")
        elif name == "verify-existing" and self.mutate:
            (directory / "circuit.zkey").write_text("changed key")
        log.write_text(required or "successful command")
        return {"seconds": 1, "peak_rss_bytes": 1024}


class RevalidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / "onion_p10"
        self.directory.mkdir()
        for names in generator.STAGE_OUTPUTS.values():
            for name in names:
                path = self.directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
        header = struct.pack("<I", 32) + bytes(32) + struct.pack("<IIIIQI", 520, 1, 0, 2, 520, 517)
        (self.directory / "circuit.r1cs").write_bytes(
            struct.pack("<4sII", b"r1cs", 1, 1) + struct.pack("<IQ", 1, len(header)) + header)
        (self.directory / "vkey.json").write_text(json.dumps(
            {"nPublic": 1, "vk_gamma_2": [1], "vk_delta_2": [2]}))
        (self.directory / "public.json").write_text('["123"]')
        self.state = {"ready": True, "domain": 1 << 10, "config": {"requested_power": 10},
                      **generator.r1cs_header(self.directory / "circuit.r1cs"),
                      "stages": {stage: {"outputs": {
                          name: generator.fingerprint(self.directory / name) for name in names}}
                          for stage, names in generator.STAGE_OUTPUTS.items()}}
        generator.atomic_json(self.directory / "metadata.json", self.state)
        generator.atomic_json(self.directory / "ship-metadata.json", {"ready": True, "domain": 1 << 10})

    def run_revalidation(self, runner):
        with patch.object(generator, "ROOT", self.root):
            revalidation.revalidate(self.directory, runner)

    def test_legacy_fixture_requires_fresh_validation_before_certification(self):
        with self.assertRaises(ValueError):
            revalidation.check_ready(self.directory)
        self.run_revalidation(FakeRunner())
        certified = revalidation.check_ready(self.directory)
        self.assertTrue(certified["ready"])
        self.assertIn("content_revalidation", certified)
        self.assertTrue((self.root / "target/aws-bundles/onion_p10.sha256").exists())
        self.assertEqual(json.loads((self.directory / "revalidation/metadata-before.json").read_text()), self.state)
        for record in certified["stages"].values():
            self.assertTrue(all("sha256" in output for output in record["outputs"].values()))

    def test_failed_verification_never_marks_ready_or_writes_manifest(self):
        with self.assertRaises(RuntimeError):
            self.run_revalidation(FakeRunner(fail="verify"))
        failed = json.loads((self.directory / "metadata.json").read_text())
        self.assertFalse(failed["ready"])
        self.assertEqual(failed["status"], "revalidation_failed")
        self.assertFalse((self.root / "target/aws-bundles/onion_p10.sha256").exists())

    def test_failed_revalidation_can_retry_without_blessing_outputs(self):
        with self.assertRaises(RuntimeError):
            self.run_revalidation(FakeRunner(fail="verify"))
        self.run_revalidation(FakeRunner())
        self.assertTrue(revalidation.check_ready(self.directory)["ready"])

    def test_unready_fixture_without_revalidation_backup_is_rejected(self):
        self.state["ready"] = False
        generator.atomic_json(self.directory / "metadata.json", self.state)
        with self.assertRaisesRegex(ValueError, "not marked ready"):
            self.run_revalidation(FakeRunner())

    def test_change_during_revalidation_never_certifies(self):
        with self.assertRaisesRegex(ValueError, "changed during revalidation"):
            self.run_revalidation(FakeRunner(mutate=True))
        self.assertFalse(json.loads((self.directory / "metadata.json").read_text())["ready"])

    def test_missing_output_record_rejected_before_commands(self):
        self.state["stages"]["verify"]["outputs"] = {}
        generator.atomic_json(self.directory / "metadata.json", self.state)
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.run_revalidation(FakeRunner())


if __name__ == "__main__":
    unittest.main()
