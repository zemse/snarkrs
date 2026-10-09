#!/usr/bin/env python3
"""Runtime-only CPU campaign. Lifecycle and watchdog installation belong to the caller."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
VARIANTS = ("js_2x2_d32", "js_16x16_d32")
THREADS = (1, 4, 8, 16)
FLAGS = {"generic": "", "adx": "-C target-feature=+adx,+bmi2"}
BINARIES = ("snarkrs-generic", "snarkrs-adx", "rapidsnark", "rapidsnark-warm", "rapidsnark-verify")
ARTIFACTS = (
    "circuit.zkey",
    "circuit.wtns",
    "vkey.json",
    "public.json",
    "input.json",
    "r1cs-info.txt",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def certify(runtime):
    cert = read_json(runtime / "certification.json")
    for name in BINARIES:
        entry = cert["binaries"][name]
        path = runtime / "bin" / name
        with path.open("rb") as stream:
            header = stream.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
            raise ValueError(f"{name}: expected Linux x86_64 ELF")
        if digest(path) != entry["sha256"] or not entry["compiler"] or not entry["verification"]:
            raise ValueError(f"{name}: missing or mismatched local certification")
        if not re.fullmatch(r"[0-9a-f]{40}", entry["source"]):
            raise ValueError(f"{name}: source must be a full revision")
    for label, flags in FLAGS.items():
        if cert["binaries"][f"snarkrs-{label}"]["flags"] != flags:
            raise ValueError(f"{label}: build flags must be exactly {flags!r}")
    if cert["binaries"]["snarkrs-generic"]["source"] != cert["binaries"]["snarkrs-adx"]["source"]:
        raise ValueError("generic and ADX sources differ")
    return cert


def pack(runtime, artifacts, destination):
    cert = certify(runtime)
    destination.mkdir()  # Refuse to mix a new bundle with old files.
    copies = [(runtime / "certification.json", "certification.json")]
    copies += [(runtime / "bin" / name, f"bench/bin/{name}") for name in BINARIES]
    copies += [
        (artifacts / v / name, f"bench/artifacts/{v}/{name}")
        for v in VARIANTS
        for name in ARTIFACTS
    ]
    copies += [
        (HERE / "cpu-campaign.py", "bench/aws/cpu-campaign.py"),
        (HERE / "rapidsnark-oracle.sh", "bench/bin/rapidsnark-oracle"),
        (
            HERE.parent / "scripts" / "check_reference_public.py",
            "bench/bin/check_reference_public.py",
        ),
    ]
    for name in ("run-comparison.py", "check_reference_public.py"):
        copies.append((HERE.parent / "scripts" / name, f"bench/scripts/{name}"))
    # Optional pre-certified dynamic runtime closure, not compiler/vendor trees.
    lib = runtime / "lib"
    if lib.exists():
        for path in sorted(lib.iterdir()):
            if not path.is_file() or ".so" not in path.name:
                raise ValueError(f"unexpected runtime library: {path}")
            if cert.get("libraries", {}).get(path.name) != digest(path):
                raise ValueError(f"uncertified runtime library: {path}")
            copies.append((path, f"lib/{path.name}"))
    hashes = {}
    for source, relative in copies:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if relative.startswith("bench/bin/") and target.name != "check_reference_public.py":
            target.chmod(0o755)
        hashes[relative] = digest(target)
    save_json(destination / "manifest.json", hashes)
    verify_bundle(destination)


def verify_bundle(root):
    hashes = read_json(root / "manifest.json")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "results" not in p.relative_to(root).parts
    }
    if actual != set(hashes):
        raise ValueError("runtime bundle file set differs from manifest")
    for relative, expected in hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
            raise ValueError("unsafe manifest path")
        if digest(root / path) != expected:
            raise ValueError(f"runtime digest mismatch: {relative}")
    # The same binary identities are rechecked before any guest execution.
    cert = read_json(root / "certification.json")
    for name in BINARIES:
        if digest(root / "bench/bin" / name) != cert["binaries"][name]["sha256"]:
            raise ValueError(f"certification digest mismatch: {name}")
    return hashes


def transfer(host, key, remote, local, retrieve=False):
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-"):
        raise ValueError("invalid SSH host")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote) or ".." in Path(remote).parts:
        raise ValueError("remote must be a simple absolute path")
    options = ["-i", str(key), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    if retrieve:
        local.mkdir()  # Never overwrite a prior retrieval.
        subprocess.run(["scp", *options, "-r", f"{host}:{remote}/results", str(local)], check=True)
    else:
        verify_bundle(local)
        subprocess.run(["ssh", *options, host, "mkdir " + shlex.quote(remote)], check=True)
        subprocess.run(["scp", *options, "-r", str(local) + "/.", f"{host}:{remote}/"], check=True)


def validate_rows(rows, variant, prover, mode, reps):
    if len(rows) != reps or [r.get("rep") for r in rows] != [str(i) for i in range(1, reps + 1)]:
        raise ValueError("incomplete repetition count or sequence")
    for row in rows:
        if any(
            row.get(k) != value
            for k, value in (
                ("variant", variant),
                ("prover", prover),
                ("mode", mode),
                ("verified", "yes"),
                ("backend", "cpu"),
            )
        ):
            raise ValueError("unverified or wrong configuration")
        if not math.isfinite(float(row["ms"])) or float(row["ms"]) <= 0:
            raise ValueError("invalid timing")
        if row.get("loaded", "no") != "no":
            raise ValueError("loaded timing rejected")


def warm_samples(text, reps):
    samples = []
    for line in text.splitlines():
        if line.startswith("rep "):
            fields = line.split()
            if len(fields) != 3 or fields[1] != str(len(samples) + 1):
                raise ValueError("invalid warm repetition sequence")
            if "." not in fields[2] and "e" not in fields[2].lower():
                raise ValueError("high-resolution fractional warm samples required")
            value = float(fields[2])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid warm timing")
            samples.append(value)
    if len(samples) != reps:
        raise ValueError("incomplete warm timings")
    return samples


def run_logged(command, directory, env, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("campaign budget exhausted; repetitions not reduced")
    save_json(
        directory / "command.json",
        {
            "argv": command,
            "environment": {
                key: env[key]
                for key in (
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "OMP_DYNAMIC",
                    "LD_LIBRARY_PATH",
                )
                if key in env
            },
        },
    )
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr-rss.log").open(
        "w"
    ) as stderr:
        try:
            process = subprocess.Popen(
                ["/usr/bin/time", "-v", *command],
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        except OSError as error:
            save_json(directory / "exit.json", {"exit": None, "spawn_error": str(error)})
            raise
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            save_json(directory / "exit.json", {"exit": process.returncode, "timeout": True})
            raise ValueError("campaign budget exhausted during configuration")
    save_json(directory / "exit.json", {"exit": code, "timeout": False})
    if code != 0:
        raise ValueError(f"configuration exited {code}: {directory}")
    return (directory / "stdout.log").read_text()


def idle(deadline, ceiling):
    while True:
        load = os.getloadavg()[0]
        if load <= ceiling:
            return load
        if time.monotonic() + 5 >= deadline:
            raise ValueError(f"loaded host rejected: load1={load}, ceiling={ceiling}")
        time.sleep(5)


def verify_proof(root, directory, variant, public, proof, env, deadline):
    if not public.is_file() or not proof.is_file():
        raise ValueError("missing per-iteration public/proof output")
    vkey = str(root / f"bench/artifacts/{variant}/vkey.json")
    reference = directory / "reference"
    reference.mkdir()
    run_logged(
        ["python3", str(root / "bench/bin/check_reference_public.py"), vkey, str(public)],
        reference,
        env,
        deadline,
    )
    output = run_logged(
        [
            str(root / "bench/bin/snarkrs-generic"),
            "groth16",
            "verify",
            vkey,
            str(public),
            str(proof),
        ],
        directory,
        env,
        deadline,
    )
    if "OK!" not in output:
        raise ValueError("independent snarkrs verifier omitted its success marker")


def native_pool(stderr):
    pools = re.findall(r"ffiasm_pool_threads\s*[:=]\s*(\d+)", stderr)
    if not pools or any(int(value) != 16 for value in pools):
        raise ValueError("missing or non-16 actual ffiasm pool diagnostic")
    return 16


def guest(root, cold, warm, budget, max_load, watchdog):
    import fcntl

    verify_bundle(root)
    if min(cold, warm, budget) <= 0 or not math.isfinite(max_load) or max_load < 0:
        raise ValueError("invalid repetitions, budget or load ceiling")
    if (
        not (watchdog / "config").is_file()
        or not (watchdog / "lease").is_file()
        or (watchdog / "done").exists()
    ):
        raise ValueError("coordinator-installed active watchdog required")
    if not os.access(watchdog / "lease", os.W_OK) or not os.access(watchdog, os.W_OK):
        raise ValueError("watchdog lease/done must be writable by the campaign user")
    if platform_machine() != "x86_64":
        raise ValueError("x86_64 guest required")
    if os.cpu_count() != 16:
        raise ValueError("exactly 16 logical host CPUs required for native rapidsnark")
    cpuinfo = Path("/proc/cpuinfo").read_text()
    if not all(flag in cpuinfo.split() for flag in ("adx", "bmi2")):
        raise ValueError("ADX/BMI2 unavailable")
    results = root / "results"
    results.mkdir()
    lock = Path("/tmp/snarkrs-cpu-campaign.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("another CPU campaign is active")
    deadline = time.monotonic() + budget
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            (watchdog / "lease").touch()
            stop.wait(30)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        (results / "cpuinfo.txt").write_text(cpuinfo)
        shutil.copyfile("/etc/os-release", results / "os-release.txt")
        save_json(results / "manifest.json", read_json(root / "manifest.json"))
        shutil.copyfile(root / "certification.json", results / "certification.json")
        env = dict(os.environ, LD_LIBRARY_PATH=str(root / "lib"), RAYON_NUM_THREADS="1")
        topology = results / "topology"
        topology.mkdir()
        run_logged(["lscpu", "--extended"], topology, env, deadline)
        save_json(
            results / "plan.json",
            {
                "snarkrs_requested_threads": THREADS,
                "rapidsnark_native_threads": 16,
                "host_logical_cpus": os.cpu_count(),
                "variants": VARIANTS,
                "cold": cold,
                "warm": warm,
                "budget_seconds": budget,
                "max_load": max_load,
                "snarkjs_compatible": "unknown",
                "gnark_comparison": "not measured",
            },
        )
        for threads in THREADS:
            env.update(
                RAYON_NUM_THREADS=str(threads),
                OMP_NUM_THREADS=str(threads),
                OMP_THREAD_LIMIT=str(threads),
                OMP_DYNAMIC="FALSE",
            )
            for label in FLAGS:
                # Separate runtime trees retain the comparison harness's fixed layout.
                tree = root / f"runtime-{label}"
                if not tree.exists():
                    (tree / "target/release").mkdir(parents=True)
                    (tree / "target/release/snarkrs").symlink_to(
                        root / f"bench/bin/snarkrs-{label}"
                    )
                    shutil.copytree(root / "bench/scripts", tree / "bench/scripts")
                    (tree / "bench/artifacts").symlink_to(root / "bench/artifacts")
                    (tree / "bench/bin").symlink_to(root / "bench/bin")
                for variant in VARIANTS:
                    directory = results / f"{label}-{threads}-{variant}"
                    directory.mkdir()
                    load = idle(deadline, max_load)
                    save_json(directory / "load.json", {"load1": load, "max_load": max_load})
                    csv_path = directory / "raw.csv"
                    command = [
                        "python3",
                        str(tree / "bench/scripts/run-comparison.py"),
                        "--skip-rapidsnark",
                        "--backends",
                        "cpu",
                        "--variants",
                        variant,
                        "--cold-reps",
                        str(cold),
                        "--warm-reps",
                        str(warm),
                        "--csv",
                        str(csv_path),
                        "--warm-oracle",
                        str(root / "bench/bin/rapidsnark-oracle"),
                        "--max-load",
                        str(max_load),
                        "--fail-fast",
                    ]
                    run_logged(command, directory, env, deadline)
                    with csv_path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    for mode, reps in (("cold", cold), ("warm", warm)):
                        selected = [r for r in rows if r["mode"] == mode]
                        validate_rows(selected, variant, "ours", mode, reps)
                        scope = "each-proof" if mode == "cold" else "first-proof"
                        if any(r.get("independent_verification_scope") != scope for r in selected):
                            raise ValueError("unexpected independent verification scope")
                        if mode == "warm":
                            for row in selected:
                                for field in (
                                    "prepare_ms",
                                    "gather_us",
                                    "ntt_us",
                                    "pointwise_us",
                                    "msm_us",
                                    "assemble_us",
                                ):
                                    value = float(row[field])
                                    if not math.isfinite(value) or value < 0:
                                        raise ValueError(f"invalid stage timing: {field}")
                    if len(rows) != cold + warm or any(
                        r.get("snarkjs_compatible") != "unknown" for r in rows
                    ):
                        raise ValueError("unexpected row count or encoding claim")
                    save_json(
                        directory / "accepted.json",
                        {
                            "build": label,
                            "flags": FLAGS[label],
                            "requested_rayon_threads": threads,
                            "actual_pool_threads": None,
                            "rows": len(rows),
                            "cold_scope": "each-proof",
                            "warm_independent_scope": "first-proof",
                            "warm_self_scope": "each-proof",
                        },
                    )
        # ffiasm uses hardware_concurrency, not the OpenMP environment budget.
        env.update(RAYON_NUM_THREADS="16", OMP_NUM_THREADS="16", OMP_THREAD_LIMIT="16")
        for variant in VARIANTS:
            competitor(root, results, variant, cold, warm, env, deadline, max_load)
        save_json(results / "complete.json", {"configurations": 36, "status": "verified"})
    except Exception as error:
        save_json(results / "failure.json", {"error": str(error)})
        raise
    finally:
        stop.set()
        worker.join()
        (watchdog / "done").touch()
        lock.close()


def platform_machine():
    import platform

    return platform.machine()


def competitor(root, results, variant, cold, warm, env, deadline, max_load):
    if os.cpu_count() != 16 or env.get("OMP_NUM_THREADS") != "16":
        raise ValueError("rapidsnark is native-16 only; OpenMP cannot resize ffiasm")
    artifact = root / "bench/artifacts" / variant
    # Warm diagnostics gate the native pool claim before any competitor acceptance.
    for mode, reps in (("warm", warm), ("cold", cold)):
        directory = results / f"rapidsnark-native16-{variant}-{mode}"
        directory.mkdir()
        save_json(
            directory / "load.json", {"load1": idle(deadline, max_load), "max_load": max_load}
        )
        rows = []
        if mode == "cold":
            for rep in range(1, reps + 1):
                sample = directory / str(rep)
                sample.mkdir()
                proof, public = sample / "proof.json", sample / "public.json"
                command = [
                    str(root / "bench/bin/rapidsnark"),
                    str(artifact / "circuit.zkey"),
                    str(artifact / "circuit.wtns"),
                    str(proof),
                    str(public),
                ]
                start = time.monotonic()
                run_logged(command, sample, env, deadline)
                elapsed = (time.monotonic() - start) * 1000
                oracle = sample / "oracle"
                oracle.mkdir()
                verify_proof(root, oracle, variant, public, proof, env, deadline)
                rows.append(elapsed)
        else:
            proof_dir = directory / "proofs"
            proof_dir.mkdir()
            command = [
                str(root / "bench/bin/rapidsnark-warm"),
                str(artifact / "circuit.zkey"),
                str(artifact / "circuit.wtns"),
                str(directory / "last-proof.json"),
                str(directory / "last-public.json"),
                str(reps),
                str(artifact / "vkey.json"),
                "--proof-dir",
                str(proof_dir),
            ]
            rows = warm_samples(run_logged(command, directory, env, deadline), reps)
            native_pool((directory / "stderr-rss.log").read_text())
            expected = {
                f"{kind}-{rep}.json" for kind in ("proof", "public") for rep in range(1, reps + 1)
            }
            if {p.name for p in proof_dir.iterdir()} != expected:
                raise ValueError("incomplete or unexpected per-iteration exports")
            for rep in range(1, reps + 1):
                oracle = directory / f"oracle-{rep}"
                oracle.mkdir()
                public = proof_dir / f"public-{rep}.json"
                proof = proof_dir / f"proof-{rep}.json"
                verify_proof(root, oracle, variant, public, proof, env, deadline)
        save_json(
            directory / "accepted.json",
            {
                "prover": "rapidsnark",
                "thread_mode": "native16",
                "actual_pool_threads": 16 if mode == "warm" else None,
                "source_expected_pool_threads": 16,
                "pool_evidence": "warm wrapper diagnostic"
                if mode == "warm"
                else "same native hardware_concurrency source on 16-CPU host",
                "environment_omp_threads": env["OMP_NUM_THREADS"],
                "independent_oracle": "snarkrs-generic groth16 verify",
                "variant": variant,
                "mode": mode,
                "ms": rows,
                "timing_scope": "driver wall including GNU time/log overhead, excluding verification"
                if mode == "cold"
                else "wrapper proving call only",
                "verified": "yes",
                "independent_scope": "each-proof",
                "snarkjs_compatible": "unknown",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    for action in ("deploy", "retrieve"):
        p = sub.add_parser(action)
        p.add_argument("--host", required=True)
        p.add_argument("--key", type=Path, required=True)
        p.add_argument("--remote", required=True)
        p.add_argument("--local", type=Path, required=True)
    p = sub.add_parser("guest")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cold-reps", type=int, default=3)
    p.add_argument("--warm-reps", type=int, default=15)
    p.add_argument("--budget-seconds", type=int, default=3600)
    p.add_argument("--max-load", type=float, default=1.0)
    p.add_argument("--watchdog", type=Path, default=Path("/var/lib/g16-watchdog"))
    args = parser.parse_args()
    try:
        if args.action == "pack":
            pack(args.runtime.resolve(), args.artifacts.resolve(), args.bundle.resolve())
        elif args.action in ("deploy", "retrieve"):
            transfer(
                args.host,
                args.key.resolve(),
                args.remote,
                args.local.resolve(),
                args.action == "retrieve",
            )
        else:
            guest(
                args.root.resolve(),
                args.cold_reps,
                args.warm_reps,
                args.budget_seconds,
                args.max_load,
                args.watchdog,
            )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"CPU campaign failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
