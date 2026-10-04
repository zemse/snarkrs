#!/usr/bin/env python3
"""Gate repeated onion benchmarks on independently verified CUDA memory observations."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from check_reference_public import matches_reference

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/results/onion-ladder"


def device_memory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, timeout=10, check=True)
    return [int(v.strip()) for v in result.stdout.strip().split(",")]


def observe(variant):
    artifact = ROOT / "bench/artifacts" / variant
    proof, public = OUT / f"{variant}-proof.json", OUT / f"{variant}-public.json"
    command = [str(ROOT / "target/release/snarkrs"), "groth16", "prove",
               str(artifact / "circuit.zkey"), str(artifact / "circuit.wtns"),
               str(proof), str(public), "--backend", "cuda"]
    samples = []
    started = time.monotonic()
    with (OUT / f"{variant}-first.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                status = Path(f"/proc/{process.pid}/status")
                try:
                    data = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
                    rss = int(data.get("VmHWM", "0 kB").split()[0])
                except FileNotFoundError:
                    break
                mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available = int(mem["MemAvailable"].split()[0])
                used, total = device_memory()
                samples.append({"seconds": time.monotonic() - started, "host_hwm_kib": rss,
                                "host_available_kib": available, "device_used_mib": used,
                                "device_total_mib": total})
                if available < 4 * 1024 * 1024 or used > total * 0.90:
                    raise RuntimeError(f"{variant}: insufficient host/device headroom")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            (OUT / f"{variant}-memory-samples.json").write_text(json.dumps(samples, indent=2))
    if process.returncode != 0 or not samples:
        raise RuntimeError(f"{variant}: CUDA first proof failed: {process.returncode}")
    if not matches_reference(artifact / "vkey.json", public):
        raise RuntimeError(f"{variant}: public signals mismatch")
    verified = subprocess.run([str(ROOT / "bench/bin/rapidsnark-verify"),
                               str(artifact / "vkey.json"), str(public), str(proof)],
                              capture_output=True, text=True, timeout=120)
    (OUT / f"{variant}-verify.log").write_text(verified.stdout + verified.stderr)
    if verified.returncode or "Result: Valid proof" not in verified.stdout + verified.stderr:
        raise RuntimeError(f"{variant}: independent verification failed")
    own = subprocess.run([str(ROOT / "target/release/snarkrs"), "groth16", "verify",
                          str(artifact / "vkey.json"), str(public), str(proof)],
                         capture_output=True, text=True, timeout=120)
    (OUT / f"{variant}-own-verify.log").write_text(own.stdout + own.stderr)
    if own.returncode:
        raise RuntimeError(f"{variant}: own verification failed")
    return {"variant": variant, "seconds": time.monotonic() - started,
            "host_hwm_kib": max(s["host_hwm_kib"] for s in samples),
            "host_min_available_kib": min(s["host_available_kib"] for s in samples),
            "device_sampled_peak_mib": max(s["device_used_mib"] for s in samples),
            "device_total_mib": samples[0]["device_total_mib"],
            "independently_verified": True, "own_verified": True,
            "sampling": "VmHWM and nvidia-smi every 0.2s plus query time; device peak is sampled"}


def validate(path, variants, cold, warm):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(v, b, m): cold if m == "cold" else warm
                for v in variants for b in ("cpu", "cuda") for m in ("cold", "warm")}
    actual = {}
    for row in rows:
        key = row["variant"], row["backend"], row["mode"]
        if key not in expected or row["prover"] != "ours" or row["verified"] != "yes":
            raise RuntimeError("unexpected or unverified row")
        if not math.isfinite(float(row["ms"])) or float(row["ms"]) <= 0:
            raise RuntimeError("invalid timing")
        scope = "each-proof" if row["mode"] == "cold" else "first-proof"
        if row["independent_verification_scope"] != scope:
            raise RuntimeError("invalid independent verification scope")
        if row["snarkjs_compatible"] != "unknown":
            raise RuntimeError("unexpected remote snarkjs compatibility claim")
        actual.setdefault(key, []).append(row)
    if set(actual) != set(expected):
        raise RuntimeError("incomplete configuration set")
    summary = []
    for key, count in expected.items():
        group = actual[key]
        if [r["rep"] for r in group] != [str(i) for i in range(1, count + 1)]:
            raise RuntimeError("invalid repetition count/sequence")
        times = [float(r["ms"]) for r in group]
        summary.append({"variant": key[0], "backend": key[1], "mode": key[2],
                        "count": count, "median_ms": statistics.median(times),
                        "min_ms": min(times), "max_ms": max(times)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--cold-reps", type=int, default=3)
    parser.add_argument("--warm-reps", type=int, default=5)
    args = parser.parse_args()
    if args.variants != [f"onion_p{p}" for p in range(21, 25)]:
        parser.error("requires the complete ordered p21 through p24 ladder")
    OUT.mkdir(parents=True, exist_ok=True)
    observations = []
    for variant in args.variants:
        observation = observe(variant)
        observations.append(observation)
        (OUT / "first-proofs.json").write_text(json.dumps(observations, indent=2))
        print(json.dumps(observation), flush=True)
    # Reserve enough of the guest hard cap for retrieval and stop.
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    remaining = 180 * 60 - uptime - 600
    estimate = sum(o["seconds"] for o in observations) * (args.cold_reps + args.warm_reps + 2) * 8
    if estimate > remaining:
        raise RuntimeError(f"repetition budget risk: conservative {estimate:.0f}s, available {remaining:.0f}s; do not silently reduce counts")
    (OUT / "repetition-plan.json").write_text(json.dumps({"cold": args.cold_reps, "warm": args.warm_reps,
                                                        "conservative_seconds": estimate,
                                                        "available_seconds": remaining}, indent=2))
    csv_path = OUT / "comparison.csv"
    subprocess.run(["python3", str(ROOT / "bench/scripts/run-comparison.py"),
                    "--cold-reps", str(args.cold_reps), "--warm-reps", str(args.warm_reps),
                    "--backends", "cpu", "cuda", "--skip-rapidsnark", "--fail-fast", "--max-load", "4",
                    "--warm-oracle", str(ROOT / "bench/bin/rapidsnark-oracle"),
                    "--csv", str(csv_path), "--variants", *args.variants], check=True)
    summary = validate(csv_path, args.variants, args.cold_reps, args.warm_reps)
    (OUT / "validated-summary.json").write_text(json.dumps(summary, indent=2))
    print("PASS: all repetition counts, sequences, timings and verification scopes", flush=True)


if __name__ == "__main__":
    main()
