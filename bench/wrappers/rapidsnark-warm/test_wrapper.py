#!/usr/bin/env python3
"""Exercise the trusted wrapper against stub prover and verifier APIs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class WarmWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "fileloader.hpp").write_text('''
#pragma once
namespace BinFileUtils {
class FileLoader {
public:
    FileLoader(const std::string &) {}
    const void *dataBuffer() { return "data"; }
    unsigned long long dataSize() { return 4; }
};
}
''')
        (cls.root / "prover.h").write_text('''
#pragma once
#define PROVER_OK 0
int groth16_public_size_for_zkey_buf(const void *, unsigned long long,
    unsigned long long *, char *, unsigned long long);
void groth16_proof_size(unsigned long long *);
int groth16_prover_create(void **, const void *, unsigned long long,
    char *, unsigned long long);
int groth16_prover_prove(void *, const void *, unsigned long long, char *,
    unsigned long long *, char *, unsigned long long *, char *, unsigned long long);
void groth16_prover_destroy(void *);
''')
        (cls.root / "verifier.h").write_text('''
#pragma once
#define VERIFIER_VALID_PROOF 0
int groth16_verify(const char *, const char *, const char *, char *, unsigned long);
''')
        (cls.root / "stub.cpp").write_text('''
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include "prover.h"
#include "verifier.h"
static int proofs = 0, verified = 0;
int groth16_public_size_for_zkey_buf(const void *, unsigned long long,
    unsigned long long *n, char *, unsigned long long) { *n = 64; return 0; }
void groth16_proof_size(unsigned long long *n) { *n = 64; }
int groth16_prover_create(void **p, const void *, unsigned long long,
    char *, unsigned long long) { *p = &proofs; return 0; }
int groth16_prover_prove(void *, const void *, unsigned long long, char *p,
    unsigned long long *ps, char *u, unsigned long long *us, char *, unsigned long long) {
    if (proofs != verified) return 1;
    ++proofs;
    *ps = std::snprintf(p, 64, "{\\"rep\\":%d}", proofs);
    *us = std::snprintf(u, 64, "[%d]", proofs);
    return 0;
}
int groth16_verify(const char *p, const char *u, const char *v, char *e, unsigned long) {
    if (std::getenv("SLOW_VERIFY"))
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    char expected[64];
    std::snprintf(expected, sizeof(expected), "{\\"rep\\":%d}", proofs);
    if (std::strcmp(expected, p) || std::strcmp(v, "{}")) return 2;
    std::snprintf(expected, sizeof(expected), "[%d]", proofs);
    if (std::strcmp(expected, u)) return 2;
    const char *fail = std::getenv("FAIL_REP");
    if (fail && proofs == std::atoi(fail)) {
        std::strcpy(e, "rejected");
        return std::getenv("VERIFY_ERROR") ? 2 : 1;
    }
    ++verified;
    return 0;
}
void groth16_prover_destroy(void *) {
    const char *path = std::getenv("DESTROY_MARKER");
    if (path) std::ofstream(path) << verified;
}
''')
        cls.binary = cls.root / "warm"
        subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-pthread", "-I", str(cls.root), str(HERE / "main.cpp"),
                        str(cls.root / "stub.cpp"), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_wrapper(self, reps="3", vkey="{}", extra_env=None, bad_output=False):
        self.case = tempfile.TemporaryDirectory()
        self.addCleanup(self.case.cleanup)
        self.path = Path(self.case.name)
        (self.path / "vkey.json").write_text(vkey)
        self.marker = self.path / "destroyed"
        env = dict(os.environ, DESTROY_MARKER=str(self.marker))
        env.update(extra_env or {})
        self.proof = self.path / ("missing/proof.json" if bad_output else "proof.json")
        return subprocess.run([str(self.binary), "key", "witness", str(self.proof),
                               str(self.path / "public.json"), reps,
                               str(self.path / "vkey.json")], env=env,
                              capture_output=True, text=True, timeout=10)

    def test_verifies_every_repetition(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 3)
        self.assertEqual(self.marker.read_text(), "3")
        self.assertEqual(self.proof.read_text(), '{"rep":3}')

    def test_rejects_middle_proof_and_cleans_up(self):
        result = self.run_wrapper(extra_env={"FAIL_REP": "2"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn("verification failed", result.stderr)
        self.assertEqual(self.marker.read_text(), "1")
        self.assertFalse(self.proof.exists())

    def test_verifier_error_is_not_success(self):
        result = self.run_wrapper(extra_env={"FAIL_REP": "1", "VERIFY_ERROR": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.marker.read_text(), "0")

    def test_verification_is_outside_timer(self):
        result = self.run_wrapper(extra_env={"SLOW_VERIFY": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(int(line.split()[2]) < 100
                            for line in result.stdout.splitlines()))

    def test_rejects_invalid_counts(self):
        for reps in ("0", "-1", "2junk", "9999999999999999999999"):
            with self.subTest(reps=reps):
                result = self.run_wrapper(reps=reps)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_rejects_empty_key(self):
        result = self.run_wrapper(vkey="")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_output_error_fails(self):
        result = self.run_wrapper(reps="1", bad_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.marker.read_text(), "1")

    def test_key_argument_is_required(self):
        result = subprocess.run([str(self.binary), "key", "witness", "p", "u", "1"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("<vkey.json>", result.stderr)


if __name__ == "__main__":
    unittest.main()
