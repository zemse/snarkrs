#!/usr/bin/env python3
# Local benchmark keys only. Never use the public entropy in production.
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "target/onion-prep"
OUT = ROOT / "bench/artifacts/large"
CIRCOM = Path.home() / ".cargo/bin/circom"
SNARKJS = Path.home() / "Library/pnpm/snarkjs"
SNARKJS_PACKAGE = Path.home() / "Library/pnpm/global/5/node_modules/snarkjs/package.json"
PTAU = Path(os.environ.get("G16_PPOT_ARCHIVE", Path.home() / "workspace/ppot-archive/ptau"))
SOURCE = ROOT / "bench/circuits/poseidon-onion.circom"
GIB = 1024 ** 3
STAGE_OUTPUTS = {
    "compile": ["circuit.r1cs", "circuit_js/circuit.wasm", "circuit_js/generate_witness.js", "circuit_js/witness_calculator.js"],
    "info": ["r1cs-info.txt"], "witness": ["input.json", "circuit.wtns"],
    "check": ["check.log"], "setup": ["circuit_0.zkey"],
    "contribute": ["circuit.zkey"], "vkey": ["vkey.json"],
    "prove": ["proof.json", "public.json"], "verify": ["verify.log"],
}
STAGES = tuple(STAGE_OUTPUTS)
SHIP_FILES = ("circuit.zkey", "circuit.wtns", "vkey.json", "public.json", "r1cs-info.txt", "ship-metadata.json")


def domain_power(constraints, public_inputs=0, outputs=1):
    return (constraints + public_inputs + outputs).bit_length()


def choose_hash_count(power, slope, intercept, public_inputs=0, outputs=1):
    count = math.floor(((1 << power) * 0.75 - intercept - public_inputs - outputs - 1) / slope)
    if count < 1 or domain_power(slope * count + intercept, public_inputs, outputs) != power:
        raise ValueError("hash count does not fit requested domain")
    return count


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def fingerprint(path):
    stat = path.stat()
    if not stat.st_size:
        raise ValueError(f"empty output: {path}")
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def output_fingerprint(path):
    before = fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if fingerprint(path) != before:
        raise ValueError(f"output changed while hashing: {path}")
    return dict(before, sha256=digest.hexdigest())


def complete(state, stage, directory):
    record = state.get("stages", {}).get(stage)
    if not isinstance(record, dict):
        return False
    outputs = record.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(STAGE_OUTPUTS[stage]):
        return False
    try:
        return all(output_fingerprint(directory / name) == info for name, info in outputs.items())
    except (OSError, ValueError):
        return False


def invalidate_from(state, stage):
    for name in STAGES[STAGES.index(stage):]:
        state.setdefault("stages", {}).pop(name, None)
    state["ready"] = False


def r1cs_header(path):
    with path.open("rb") as f:
        magic, version, sections = struct.unpack("<4sII", f.read(12))
        if magic != b"r1cs" or version != 1:
            raise ValueError("invalid R1CS")
        for _ in range(sections):
            section, size = struct.unpack("<IQ", f.read(12))
            end = f.tell() + size
            if section == 1:
                field_size = struct.unpack("<I", f.read(4))[0]
                f.seek(field_size, 1)
                wires, outputs, public_inputs, private_inputs, labels, constraints = struct.unpack("<IIIIQI", f.read(28))
                return dict(wires=wires, outputs=outputs, public_inputs=public_inputs,
                            private_inputs=private_inputs, constraints=constraints,
                            domain_power=domain_power(constraints, public_inputs, outputs))
            f.seek(end)
    raise ValueError("missing R1CS header")


def swap_bytes():
    text = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used = ([\d.]+)([MG])", text)
    if not match:
        raise RuntimeError("cannot monitor swap usage")
    return int(float(match[1]) * (1024 ** (2 if match[2] == "M" else 3)))


def group_rss(pid):
    text = subprocess.check_output(["/bin/ps", "-axo", "pgid=,rss="], text=True)
    return sum(int(rss) * 1024 for group, rss in (line.split() for line in text.splitlines()) if int(group) == pid)


class Runner:
    def __init__(self, args):
        self.args = args
        self.env = dict(os.environ, NODE_OPTIONS=f"--max-old-space-size={args.heap_gib * 1024}",
                        RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", UV_THREADPOOL_SIZE="4")
        # ffjavascript checks navigator first on recent Node versions.
        cap = WORK / "cap-workers.cjs"
        cap.write_text("const os = require('os'); const cpus = os.cpus; os.cpus = () => cpus().slice(0, 4);\n"
                       "if (globalThis.navigator) Object.defineProperty(globalThis.navigator, 'hardwareConcurrency', {value: 4, configurable: true});\n")
        self.env["NODE_OPTIONS"] += f" --require={cap}"

    def command(self, command, log, cwd, required=None):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {log}: {' '.join(map(str, command))}", flush=True)
        before_swap = swap_bytes()
        if before_swap > self.args.max_swap_gib * GIB:
            raise RuntimeError("paused: host swap already exceeds limit")
        peak = 0
        started = time.time()
        with log.open("w") as output:
            proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=self.env,
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    rss = group_rss(proc.pid)
                    peak = max(peak, rss)
                    atomic_json(WORK / "active.json", dict(pid=proc.pid, log=str(log),
                                rss_gib=round(rss / GIB, 3), peak_rss_gib=round(peak / GIB, 3),
                                elapsed_seconds=round(time.time() - started)))
                    if rss > self.args.max_rss_gib * GIB:
                        raise RuntimeError("paused: process group RSS exceeds limit")
                    if swap_bytes() > min(self.args.max_swap_gib * GIB, before_swap + GIB):
                        raise RuntimeError("paused: host swap exceeded limit or grew by 1 GiB")
                    time.sleep(3)
                if proc.returncode:
                    raise RuntimeError(f"command exited {proc.returncode}: {log}")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        if required and required not in log.read_text():
            raise RuntimeError(f"missing success marker {required!r}: {log}")
        return dict(seconds=round(time.time() - started, 2), peak_rss_bytes=peak, log=str(log))

    def compile(self, directory, count):
        wrapper = directory / "circuit.circom"
        wrapper.write_text(f'pragma circom 2.1.4;\ninclude "poseidon-onion.circom";\ncomponent main = PoseidonOnion({count});\n')
        return self.command([CIRCOM, wrapper, "--r1cs", "--wasm", "--O1", "-o", directory,
                            "-l", ROOT / "bench/node_modules/circomlib/circuits",
                            "-l", SOURCE.parent], directory / "compile.log", directory)

    def pilot(self, versions):
        path = WORK / "pilot.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["versions"] == versions:
                return old
        measurements = []
        for count in (1, 8):
            directory = WORK / f"pilot_{count}"
            directory.mkdir(exist_ok=True)
            timing = self.compile(directory, count)
            self.command([SNARKJS, "r1cs", "info", directory / "circuit.r1cs"],
                         directory / "info.log", directory, "# of Constraints")
            measurements.append(dict(hash_count=count, **r1cs_header(directory / "circuit.r1cs"), **timing))
        slope, rem = divmod(measurements[1]["constraints"] - measurements[0]["constraints"], 7)
        if rem or slope <= 0:
            raise RuntimeError("pilot constraint slope is not linear")
        result = dict(versions=versions, measurements=measurements, constraints_per_hash=slope,
                      constraint_intercept=measurements[0]["constraints"] - slope)
        atomic_json(path, result)
        return result

    def fixture(self, power, pilot, versions, smoke=False):
        count = choose_hash_count(power, pilot["constraints_per_hash"], pilot["constraint_intercept"])
        directory = WORK / "smoke_p13" if smoke else OUT / f"onion_p{power}"
        directory.mkdir(parents=True, exist_ok=True)
        ptau = PTAU / ("ppot_0080_21.ptau" if smoke else
                       f"ppot_0080_{power}.ptau" if power <= 22 else "ppot_0080_final.ptau")
        config = dict(hash_count=count, requested_power=power, compiler_flags=["--O1"],
                      versions=versions, ptau_path=str(ptau), ptau_fingerprint=fingerprint(ptau),
                      benchmark_entropy="poseidon-onion-public-benchmark-entropy-not-secure")
        path = directory / "metadata.json"
        state = json.loads(path.read_text()) if path.exists() else dict(config=config, stages={}, ready=False)
        if state["config"] != config:
            raise RuntimeError(f"configuration changed; inspect existing artifacts before resetting {path}")
        for stage in STAGES:
            if complete(state, stage, directory):
                continue
            invalidate_from(state, stage)
            (directory / "ship-metadata.json").unlink(missing_ok=True)
            state.update(status="running", current_stage=stage)
            atomic_json(path, state)
            try:
                self.preflight(power, stage)
                log = directory / f"{stage}.log"
                if stage == "compile":
                    timing = self.compile(directory, count)
                    header = r1cs_header(directory / "circuit.r1cs")
                    if header["domain_power"] != power or header["constraints"] != count * pilot["constraints_per_hash"] + pilot["constraint_intercept"]:
                        raise RuntimeError("compiled circuit does not match requested domain or pilot sizing")
                    state.update(header)
                    state["domain"] = 1 << power
                elif stage == "info":
                    timing = self.command([SNARKJS, "r1cs", "info", directory / "circuit.r1cs"], directory / "r1cs-info.txt", directory, "# of Constraints")
                elif stage == "witness":
                    atomic_json(directory / "input.json", dict(seed="123456789", salt=[str((i + 1) * 104729 + 17) for i in range(count)]))
                    timing = self.command(["node", directory / "circuit_js/generate_witness.js", directory / "circuit_js/circuit.wasm",
                                           directory / "input.json", directory / "circuit.wtns.part"], log, directory)
                    (directory / "circuit.wtns.part").replace(directory / "circuit.wtns")
                elif stage == "check":
                    timing = self.command([SNARKJS, "wtns", "check", directory / "circuit.r1cs", directory / "circuit.wtns"], log, directory, "WITNESS IS CORRECT")
                elif stage in ("setup", "contribute", "vkey"):
                    name = {"setup": "circuit_0.zkey", "contribute": "circuit.zkey", "vkey": "vkey.json"}[stage]
                    tmp = directory / (name + ".part")
                    if stage == "setup":
                        cmd = [SNARKJS, "groth16", "setup", directory / "circuit.r1cs", ptau, tmp]
                    elif stage == "contribute":
                        cmd = [SNARKJS, "zkey", "contribute", directory / "circuit_0.zkey", tmp,
                               "--name=onion-benchmark", "-e=" + config["benchmark_entropy"]]
                    else:
                        cmd = [SNARKJS, "zkey", "export", "verificationkey", directory / "circuit.zkey", tmp]
                    timing = self.command(cmd, log, directory, "Circuit hash" if stage == "setup" else None)
                    fingerprint(tmp)
                    if stage == "vkey":
                        key = json.loads(tmp.read_text())
                        if key["vk_gamma_2"] == key["vk_delta_2"] or key["nPublic"] != 1:
                            raise RuntimeError("unblinded key or unexpected public signal count")
                    tmp.replace(directory / name)
                elif stage == "prove":
                    prover = ROOT / "target/release/snarkrs"
                    if not prover.is_file():
                        raise RuntimeError("native CPU reference prover missing; no automatic build")
                    timing = self.command([prover, "groth16", "prove", directory / "circuit.zkey", directory / "circuit.wtns",
                                           directory / "proof.json.part", directory / "public.json.part", "--backend", "cpu"], log, directory)
                    public = json.loads((directory / "public.json.part").read_text())
                    if len(public) != 1 or int(public[0]) == 0:
                        raise RuntimeError("unexpected or trivial public digest")
                    for name in ("proof.json", "public.json"):
                        (directory / (name + ".part")).replace(directory / name)
                    state["reference_prover"] = str(prover)
                else:
                    timing = self.command([SNARKJS, "groth16", "verify", directory / "vkey.json", directory / "public.json",
                                           directory / "proof.json"], log, directory, "OK!")
                state["stages"][stage] = dict(**timing, completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                              outputs={name: output_fingerprint(directory / name) for name in STAGE_OUTPUTS[stage]})
                state.update(status="in_progress", current_stage=stage)
                atomic_json(path, state)
            except BaseException as error:
                state.update(status="paused_or_failed", error=str(error), current_stage=stage)
                atomic_json(path, state)
                raise
        if not all(complete(state, stage, directory) for stage in STAGES):
            raise RuntimeError("stage outputs changed before shipping certification")
        state.update(ready=True, status="ready", current_stage="verify")
        state.pop("error", None)
        state["ship_files"] = list(SHIP_FILES)
        atomic_json(directory / "ship-metadata.json", dict(hash_count=count, domain=1 << power,
                    constraints=state["constraints"], wires=state["wires"], versions=versions,
                    benchmark_only=True, ready=True, verified_by="snarkjs groth16 verify",
                    ptau_path=str(ptau), completed_stages=list(STAGES)))
        state["shipping_outputs"] = {name: output_fingerprint(directory / name) for name in SHIP_FILES}
        atomic_json(path, state)
        print(f"READY {directory}", flush=True)

    def preflight(self, power, stage):
        # Account for allocator and object overhead above linear scaling.
        for previous in range(21, power):
            path = OUT / f"onion_p{previous}/metadata.json"
            if path.exists():
                record = json.loads(path.read_text()).get("stages", {}).get(stage)
                if record:
                    projected = record["peak_rss_bytes"] * (2 ** (power - previous)) * 1.25
                    if projected > self.args.max_rss_gib * GIB:
                        raise RuntimeError(f"paused: p{power} {stage} projects {projected / GIB:.1f} GiB from p{previous}, above RSS cap")


def stop(signum, frame):
    raise KeyboardInterrupt(f"stopped by signal {signum}")


def main():
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description="Prepare local Poseidon onion benchmark fixtures serially.")
    parser.add_argument("--powers", nargs="+", type=int, default=[21, 22, 23, 24], choices=[21, 22, 23, 24])
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--heap-gib", type=int, default=40)
    parser.add_argument("--max-rss-gib", type=int, default=64)
    parser.add_argument("--max-swap-gib", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.heap_gib <= 48 or not 1 <= args.max_rss_gib <= 72:
        parser.error("heap must be 1..48 GiB and RSS cap 1..72 GiB")
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        versions = dict(circom=subprocess.check_output([CIRCOM, "--version"], text=True).strip(),
                        snarkjs=json.loads(SNARKJS_PACKAGE.read_text())["version"],
                        circuit_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        runner = Runner(args)
        pilot = runner.pilot(versions)
        print(json.dumps(pilot, indent=2), flush=True)
        if args.smoke_only:
            runner.fixture(13, pilot, versions, smoke=True)
        elif not args.pilot_only:
            for power in sorted(set(args.powers)):
                runner.fixture(power, pilot, versions)


if __name__ == "__main__":
    main()
