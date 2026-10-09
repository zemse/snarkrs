#!/usr/bin/env python3
"""Correctness-only QEMU gate for the fixed x86 campaign fixtures."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        sys.exit("run inside the disposable arm64 Linux guest")
    bundle, fixtures, manifest = map(Path, sys.argv[1:])
    metadata = bundle / "metadata"
    (metadata / "certification.json").unlink(missing_ok=True)
    records = json.loads(manifest.read_text())
    for record in records["fixtures"]:
        path = fixtures / record["path"]
        if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            sys.exit(f"fixture digest mismatch: {path}")
    variants = ("js_2x2_d32", "js_16x16_d32")
    results = []
    commands = []
    env = dict(os.environ, OMP_NUM_THREADS="2")

    def run(label, command, marker=None):
        print(f"correctness gate: {label}", flush=True)
        commands.append(dict(label=label, argv=list(map(str, command))))
        result = subprocess.run(list(map(str, command)), env=env, text=True,
                                capture_output=True, timeout=900)
        (metadata / f"qemu-{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode != 0 or (marker and marker not in result.stdout + result.stderr):
            sys.exit(f"failed {label}: status {result.returncode}, see retained log")
        return result

    def check_public(path, reference):
        actual = [int(value) for value in json.loads(path.read_text())]
        expected = [int(value) for value in json.loads(reference.read_text())]
        if actual != expected:
            sys.exit(f"public signal mismatch: {path}")

    for variant in variants:
        fixture = fixtures / variant
        proof = metadata / f"{variant}-proof.json"
        public = metadata / f"{variant}-public.json"
        qemu = ["qemu-x86_64", "-cpu", "max"]
        run(f"{variant}-cold", qemu + [bundle / "bin/rapidsnark", fixture / "circuit.zkey",
                                     fixture / "circuit.wtns", proof, public])
        check_public(public, fixture / "public.json")
        run(f"{variant}-cold-self", qemu + [bundle / "bin/rapidsnark-verify",
                                          fixture / "vkey.json", public, proof], "Result: Valid proof")
        run(f"{variant}-cold-independent", ["snarkjs", "groth16", "verify",
                                            fixture / "vkey.json", public, proof], "OK!")
        exports = metadata / f"{variant}-exports"
        exports.mkdir(exist_ok=True)
        for rep in (1, 2):
            for name in ("proof", "public"):
                (exports / f"{name}-{rep}.json").unlink(missing_ok=True)
        result = run(f"{variant}-warm", qemu + [bundle / "bin/rapidsnark-warm",
                     fixture / "circuit.zkey", fixture / "circuit.wtns", proof, public,
                     "2", fixture / "vkey.json", "--proof-dir", exports])
        repetitions = [line.split() for line in result.stdout.splitlines() if line.startswith("rep ")]
        if len(repetitions) != 2 or [row[1] for row in repetitions] != ["1", "2"]:
            sys.exit(f"incomplete warm correctness run: {variant}")
        pools = [int(line.split("=", 1)[1]) for line in result.stderr.splitlines()
                 if line.startswith("ffiasm_pool_threads=")]
        if len(pools) != 1 or pools[0] <= 0:
            sys.exit(f"missing actual ffiasm pool instrumentation: {variant}")
        for rep in (1, 2):
            exported_proof = exports / f"proof-{rep}.json"
            exported_public = exports / f"public-{rep}.json"
            check_public(exported_public, fixture / "public.json")
            run(f"{variant}-warm-export-{rep}-independent",
                ["snarkjs", "groth16", "verify", fixture / "vkey.json",
                 exported_public, exported_proof], "OK!")
        if proof.read_bytes() != (exports / "proof-2.json").read_bytes() or public.read_bytes() != (exports / "public-2.json").read_bytes():
            sys.exit(f"final outputs differ from last exported proof: {variant}")
        check_public(public, fixture / "public.json")
        run(f"{variant}-warm-self", qemu + [bundle / "bin/rapidsnark-verify",
                                          fixture / "vkey.json", public, proof], "Result: Valid proof")
        run(f"{variant}-warm-independent", ["snarkjs", "groth16", "verify",
                                            fixture / "vkey.json", public, proof], "OK!")
        results.append(dict(variant=variant, cold_proofs=1, warm_proofs=2,
                            self_verification_scope="each-proof",
                            independent_verifier="snarkjs",
                            independent_verification_scope="cold-and-each-exported-warm-and-final-proof",
                            ffiasm_pool_threads_actual=pools[0],
                            omp_num_threads_setting=env["OMP_NUM_THREADS"]))
    run("snarkjs-version", ["npm", "ls", "-g", "snarkjs", "--depth=0"])
    (metadata / "fixture-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (metadata / "smoke-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (metadata / "certification.json").write_text(json.dumps(dict(
        status="passed", execution="qemu-x86_64 -cpu max in disposable arm64 Linux guest",
        performance_results_accepted=False, native_x86_validation="pending",
        binary_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((bundle / "bin").iterdir())},
        fixtures=results), indent=2) + "\n")
    print("CORRECTNESS_GATE_PASSED (no benchmark timings accepted)", flush=True)


if __name__ == "__main__":
    main()
