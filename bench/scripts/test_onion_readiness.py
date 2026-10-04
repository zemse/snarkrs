import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import check_onion_ready as readiness
from check_reference_public import matches_reference, public_values


class OnionReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = {"ready": True, "config": {"requested_power": 21},
                      "domain_power": 21, "domain": 1 << 21, "stages": {}}
        for stage, names in readiness.generator.STAGE_OUTPUTS.items():
            record = {}
            for name in names:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
                record[name] = readiness.generator.output_fingerprint(path)
            self.state["stages"][stage] = {"outputs": record}
        (self.root / "ship-metadata.json").write_text(json.dumps({"ready": True, "domain": 1 << 21}))
        self.state["shipping_outputs"] = {
            name: readiness.generator.output_fingerprint(self.root / name)
            for name in readiness.generator.SHIP_FILES
        }
        self.save()

    def save(self):
        (self.root / "metadata.json").write_text(json.dumps(self.state))

    def test_verified_complete_fixture_is_ready(self):
        self.assertTrue(readiness.check_ready(self.root)["ready"])

    def test_false_ready_missing_stage_changed_output_and_wrong_domain_rejected(self):
        cases = ("flag", "stage", "changed", "domain")
        for case in cases:
            with self.subTest(case=case):
                state = json.loads(json.dumps(self.state))
                if case == "flag":
                    state["ready"] = False
                elif case == "stage":
                    del state["stages"]["verify"]
                elif case == "changed":
                    state["stages"]["prove"]["outputs"]["proof.json"]["size"] += 1
                else:
                    state["domain_power"] = 22
                (self.root / "metadata.json").write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    readiness.check_ready(self.root)
        self.save()

    def test_incomplete_or_empty_stage_output_lists_rejected(self):
        for stage in readiness.generator.STAGES:
            for incomplete in (False, True):
                with self.subTest(stage=stage, incomplete=incomplete):
                    state = json.loads(json.dumps(self.state))
                    outputs = state["stages"][stage]["outputs"]
                    if incomplete:
                        outputs.pop(next(iter(outputs)))
                    else:
                        outputs.clear()
                    (self.root / "metadata.json").write_text(json.dumps(state))
                    with self.assertRaises(ValueError):
                        readiness.check_ready(self.root)
        self.save()

    def test_same_size_corruption_with_preserved_mtime_rejected(self):
        path = self.root / "circuit.zkey"
        stat = path.stat()
        path.write_text("corrupt")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(readiness.generator.fingerprint(path),
                         {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        destination = self.root / "bundle/sha256.txt"
        with self.assertRaises(ValueError):
            readiness.write_bundle_manifest(self.root, destination)
        self.assertFalse(destination.exists())

    def test_missing_shipping_digests_rejected(self):
        del self.state["shipping_outputs"]
        self.save()
        with self.assertRaises(ValueError):
            readiness.check_ready(self.root)

    def test_changed_shipping_metadata_rejected(self):
        (self.root / "ship-metadata.json").write_text(json.dumps({"ready": True, "domain": 1 << 21, "extra": True}))
        with self.assertRaises(ValueError):
            readiness.check_ready(self.root)

    def test_changed_runtime_file_prevents_manifest_creation(self):
        (self.root / "circuit.wtns").write_text("changed witness")
        destination = self.root / "bundle/sha256.txt"
        with self.assertRaises(ValueError):
            readiness.write_bundle_manifest(self.root, destination)
        self.assertFalse(destination.exists())

    def test_bundle_manifest_contains_correct_runtime_hashes(self):
        destination = self.root / "bundle/sha256.txt"
        readiness.write_bundle_manifest(self.root, destination)
        lines = destination.read_text().splitlines()
        self.assertEqual(len(lines), 6)
        for line in lines:
            digest, name = line.split("  ")
            self.assertEqual(digest, hashlib.sha256((self.root / name).read_bytes()).hexdigest())
            self.assertNotIn(".r1cs", name)


class ReferencePublicTests(unittest.TestCase):
    def test_numeric_equivalence_and_different_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public.json").write_text('["123"]')
            generated = root / "generated.json"
            generated.write_text('[123]')
            self.assertTrue(matches_reference(root / "vkey.json", generated))
            generated.write_text('["124"]')
            self.assertFalse(matches_reference(root / "vkey.json", generated))

    def test_invalid_signal_types_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public.json"
            for text in ('[true]', '[1.5]', '{}', '["bad"]'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    public_values(path)


if __name__ == "__main__":
    unittest.main()
