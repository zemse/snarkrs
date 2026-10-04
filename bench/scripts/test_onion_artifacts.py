import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("onion", Path(__file__).with_name("gen-onion-artifacts.py"))
onion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onion)


class Sizing(unittest.TestCase):
    def test_public_output_crosses_boundary(self):
        self.assertEqual(onion.domain_power(1022, 0, 1), 10)
        self.assertEqual(onion.domain_power(1023, 0, 1), 11)
        self.assertEqual(onion.domain_power(1021, 1, 1), 10)
        self.assertEqual(onion.domain_power(1022, 1, 1), 11)

    def test_ladder_domains(self):
        for slope, intercept in ((517, 0), (243, 2), (500, 7)):
            for power in (21, 22, 23, 24):
                count = onion.choose_hash_count(power, slope, intercept)
                total = slope * count + intercept + 2
                self.assertGreater(total, 1 << (power - 1))
                self.assertLess(total, 1 << power)
                self.assertEqual(onion.domain_power(total - 2), power)
                self.assertLess(abs(total / (1 << power) - 0.75), 0.001)

    def test_impossible_sizing_rejected(self):
        with self.assertRaises(ValueError):
            onion.choose_hash_count(2, 517, 0)

    def test_r1cs_binary_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "circuit.r1cs"
            header = struct.pack("<I", 32) + bytes(32) + struct.pack("<IIIIQI", 900, 1, 0, 9, 1234, 800)
            path.write_bytes(struct.pack("<4sII", b"r1cs", 1, 1) + struct.pack("<IQ", 1, len(header)) + header)
            info = onion.r1cs_header(path)
            self.assertEqual(info["constraints"], 800)
            self.assertEqual(info["wires"], 900)
            self.assertEqual(info["domain_power"], 10)


class Resumability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root / "circuit.zkey"
        self.key.write_bytes(b"complete key")
        self.state = {"ready": True, "stages": {"contribute": {"outputs": {
            "circuit.zkey": onion.output_fingerprint(self.key)}}}}

    def test_marked_output_reused(self):
        self.assertTrue(onion.complete(self.state, "contribute", self.root))

    def test_partial_output_not_reused(self):
        (self.root / "circuit.zkey.part").write_bytes(b"partial")
        self.assertFalse(onion.complete({}, "contribute", self.root))

    def test_unmarked_final_not_reused(self):
        self.assertFalse(onion.complete({}, "contribute", self.root))

    def test_changed_or_missing_output_not_reused(self):
        self.key.write_bytes(b"truncated")
        self.assertFalse(onion.complete(self.state, "contribute", self.root))
        self.key.unlink()
        self.assertFalse(onion.complete(self.state, "contribute", self.root))

    def test_empty_incomplete_extra_and_legacy_outputs_not_reused(self):
        for outputs in ({}, {"other": onion.output_fingerprint(self.key)},
                        {"circuit.zkey": onion.fingerprint(self.key)},
                        {"circuit.zkey": onion.output_fingerprint(self.key), "extra": {}}):
            with self.subTest(outputs=outputs):
                state = {"stages": {"contribute": {"outputs": outputs}}}
                self.assertFalse(onion.complete(state, "contribute", self.root))

    def test_invalidation_removes_downstream_and_ready(self):
        self.state["stages"].update({name: {} for name in onion.STAGES})
        onion.invalidate_from(self.state, "setup")
        self.assertFalse(self.state["ready"])
        self.assertEqual(set(self.state["stages"]), {"compile", "info", "witness", "check"})

    def test_atomic_json_replaces_state(self):
        path = self.root / "metadata.json"
        onion.atomic_json(path, {"ready": False})
        onion.atomic_json(path, {"ready": True})
        self.assertEqual(json.loads(path.read_text()), {"ready": True})
        self.assertFalse((self.root / "metadata.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
