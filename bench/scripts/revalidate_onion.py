#!/usr/bin/env python3
"""Certify existing onion fixtures with content digests and fresh local proofs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time

from check_onion_ready import check_ready, generator, write_bundle_manifest
from check_reference_public import public_values


def snapshot(directory):
    return {stage: {name: generator.output_fingerprint(directory / name) for name in names}
            for stage, names in generator.STAGE_OUTPUTS.items()}


def require_legacy_outputs(state, directory):
    if state.get("ready") is not True:
        raise ValueError("fixture is not marked ready; inspect before revalidation")
    for stage, names in generator.STAGE_OUTPUTS.items():
        outputs = state.get("stages", {}).get(stage, {}).get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != set(names):
            raise ValueError(f"missing required {stage} outputs")
        for name, info in outputs.items():
            actual = generator.fingerprint(directory / name)
            if actual != {key: info.get(key) for key in ("size", "mtime_ns")}:
                raise ValueError(f"changed legacy output: {name}")
            if "sha256" in info and generator.output_fingerprint(directory / name) != info:
                raise ValueError(f"changed certified output: {name}")
    header = generator.r1cs_header(directory / "circuit.r1cs")
    power = state["config"]["requested_power"]
    if state.get("domain") != 1 << power or header["domain_power"] != power:
        raise ValueError("compiled domain does not match requested power")
    if any(state.get(key) != value for key, value in header.items()):
        raise ValueError("compiled header differs from recorded circuit")
    if header["public_inputs"] != 0 or header["outputs"] != 1:
        raise ValueError("unexpected onion public signal count")


def revalidate(directory, runner):
    directory = Path(directory).resolve()
    path = directory / "metadata.json"
    state = json.loads(path.read_text())
    work = directory / "revalidation"
    backup = work / "metadata-before.json"
    candidate = state
    if state.get("ready") is not True and state.get("status") in ("revalidating", "revalidation_failed"):
        original = json.loads(backup.read_text())
        if original.get("ready") is not True or original.get("config") != state.get("config"):
            raise ValueError("invalid revalidation backup")
        candidate = dict(state, ready=True)
    require_legacy_outputs(candidate, directory)
    before = snapshot(directory)
    work.mkdir(exist_ok=True)
    if not backup.exists():
        generator.atomic_json(backup, state)
    state.update(ready=False, status="revalidating")
    generator.atomic_json(path, state)
    timings = {}

    def run(name, command, required=None):
        timings[name] = runner.command(command, work / f"{name}.log", directory, required)

    try:
        run("witness", ["node", directory / "circuit_js/generate_witness.js",
                        directory / "circuit_js/circuit.wasm", directory / "input.json",
                        work / "circuit.wtns"])
        if generator.output_fingerprint(work / "circuit.wtns")["sha256"] != before["witness"]["circuit.wtns"]["sha256"]:
            raise ValueError("regenerated witness differs from existing witness")
        run("check", [generator.SNARKJS, "wtns", "check", directory / "circuit.r1cs",
                      directory / "circuit.wtns"], "WITNESS IS CORRECT")
        run("vkey", [generator.SNARKJS, "zkey", "export", "verificationkey",
                     directory / "circuit.zkey", work / "vkey.json"])
        key = json.loads((work / "vkey.json").read_text())
        if key != json.loads((directory / "vkey.json").read_text()):
            raise ValueError("verification key does not match proving key")
        if key["nPublic"] != 1 or key["vk_gamma_2"] == key["vk_delta_2"]:
            raise ValueError("unexpected or unblinded verification key")
        run("prove", [generator.ROOT / "target/release/snarkrs", "groth16", "prove",
                      directory / "circuit.zkey", directory / "circuit.wtns",
                      work / "proof.json", work / "public.json", "--backend", "cpu"])
        if public_values(work / "public.json") != public_values(directory / "public.json"):
            raise ValueError("fresh proof public digest differs from fixture reference")
        run("verify", [generator.SNARKJS, "groth16", "verify", directory / "vkey.json",
                       work / "public.json", work / "proof.json"], "OK!")
        run("verify-existing", [generator.SNARKJS, "groth16", "verify", directory / "vkey.json",
                                directory / "public.json", directory / "proof.json"], "OK!")
        after = snapshot(directory)
        if after != before:
            raise ValueError("fixture changed during revalidation")
        for stage, outputs in after.items():
            state["stages"][stage]["outputs"] = outputs
        state["content_revalidation"] = dict(
            completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), timings=timings,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            proof=generator.output_fingerprint(work / "proof.json"),
            public=generator.output_fingerprint(work / "public.json"),
            note="Legacy stage digests baselined after fresh witness, key and proof validation.",
        )
        state["ship_files"] = list(generator.SHIP_FILES)
        state["shipping_outputs"] = {name: generator.output_fingerprint(directory / name)
                                     for name in generator.SHIP_FILES}
        state.update(ready=True, status="ready")
        state.pop("error", None)
        generator.atomic_json(path, state)
        check_ready(directory)
        write_bundle_manifest(directory, generator.ROOT / "target/aws-bundles" / f"{directory.name}.sha256")
    except BaseException as error:
        state.update(ready=False, status="revalidation_failed", error=str(error))
        generator.atomic_json(path, state)
        raise
    print(f"CERTIFIED {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--heap-gib", type=int, default=40)
    parser.add_argument("--max-rss-gib", type=int, default=64)
    parser.add_argument("--max-swap-gib", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.heap_gib <= 48 or not 1 <= args.max_rss_gib <= 72 or args.max_swap_gib < 0:
        parser.error("invalid memory guard limits")
    signal.signal(signal.SIGTERM, generator.stop)
    generator.WORK.mkdir(parents=True, exist_ok=True)
    with (generator.WORK / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = generator.Runner(args)
        for directory in args.directories:
            revalidate(directory, runner)


if __name__ == "__main__":
    main()
