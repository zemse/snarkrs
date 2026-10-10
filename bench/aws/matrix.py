#!/usr/bin/env python3
"""CPU prover vs rapidsnark across EC2 instance types, one disposable instance per run.

    matrix.py prep --out target/matrix/bundle
    matrix.py prep --arch arm64 --out target/matrix/bundle-arm64
    matrix.py run --instance-type c6i.2xlarge --bundle target/matrix/bundle \
        --arm64-bundle target/matrix/bundle-arm64 --results target/matrix/results --summary target/matrix/summary/c6i.2xlarge.json

`run` launches a tagged instance that terminates itself on shutdown, arms a guest
shutdown at the hard cap plus an AWS scheduler stop as a backstop, uploads the bundle,
runs `matrix-guest.sh`, pulls the results back, terminates the instance in a `finally`,
verifies every retrieved proof locally with snarkjs and writes the summary JSON.
Nothing is compiled on the instance.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE = "bench"
REGION = "us-east-1"
OWNER = "bench-runner"
KEY_NAME = "agent-us-east-1"
KEY = Path.home() / ".claude/skills/aws/keys/agent-us-east-1.pem"
SG = "sg-0123456789abcdef0"
STOP_ROLE = "arn:aws:iam::123456789012:role/g16-benchmark-stop"
# Canonical's Ubuntu images: 22.04 on x86, 24.04 on Graviton, whose bundle is built on 24.04.
UBUNTU_OWNER = "099720109477"
UBUNTU_NAME = {"x86_64": "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
               "arm64": "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"}
SNARKJS = Path.home() / "Library/pnpm/snarkjs"

CIRCUITS = [
    "js_2x2_d32",
    "railgun-01x01",
    "tornado",
    "sha256",
    "railgun-13x01",
    "js_16x16_d32",
    "rsa2048",
    "keccak256",
    "anon-aadhaar",
]
CIRCUIT_FILES = ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json", "r1cs-info.txt"]
BUILD = REPO / "target/matrix/build"
BINARIES = {
    "x86_64": {
        "snarkrs-adx": BUILD / "adx/x86_64-unknown-linux-gnu/release/snarkrs",
        "snarkrs-generic": BUILD / "generic/x86_64-unknown-linux-gnu/release/snarkrs",
        "rapidsnark-warm": REPO / "target/cpu-x86-campaign/rapidsnark-bundle/bin/rapidsnark-warm",
    },
    "arm64": {
        "snarkrs-arm64": BUILD / "arm64/release/snarkrs",
        "rapidsnark-warm": BUILD / "rapidsnark-arm64/bin/rapidsnark-warm",
    },
}
FLAGS = {"x86_64": {"snarkrs-adx": "-C target-feature=+adx,+bmi2", "snarkrs-generic": ""},
         "arm64": {"snarkrs-arm64": "native aarch64-linux build on Ubuntu 24.04, no target flags"}}
ISA_FLAGS = ["adx", "bmi2", "avx2", "avx512f", "avx512ifma", "asimd", "sve", "sve2", "i8mm"]


def say(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ} {msg}", flush=True)


def sh(cmd, check=True, capture=True, **kw):
    r = subprocess.run(cmd, check=False, text=True, capture_output=capture, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed ({r.returncode}): {r.stderr.strip()}")
    return r


def aws(*args, check=True):
    r = sh(["aws", "--profile", PROFILE, "--region", REGION, *args, "--output", "json"], check=check)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# prep


def prep(args):
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists; prep writes a fresh bundle")
    (out / "bin").mkdir(parents=True)
    manifest = {"source": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip(),
                "dirty": bool(sh(["git", "-C", str(REPO), "status", "--porcelain", "--", "crates", "bin", "Cargo.toml", "Cargo.lock"]).stdout.strip()),
                "arch": args.arch, "flags": FLAGS[args.arch], "files": {}}
    for name, src in BINARIES[args.arch].items():
        shutil.copy2(src, out / "bin" / name)
        os.chmod(out / "bin" / name, 0o755)
    shutil.copy2(REPO / "bench/aws/matrix-guest.sh", out / "matrix-guest.sh")
    for c in CIRCUITS:
        (out / "artifacts" / c).mkdir(parents=True)
        for f in CIRCUIT_FILES:
            shutil.copy2(REPO / "bench/artifacts" / c / f, out / "artifacts" / c / f)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(out))] = sha256(p)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    say(f"bundle {out}: {len(manifest['files'])} files, source {manifest['source']}")


# ---------------------------------------------------------------------------
# run


class Box:
    def __init__(self, itype, results):
        self.itype = itype
        self.results = results
        self.arch = None
        self.id = None
        self.ip = None
        self.schedule = None
        self.known_hosts = results / "known_hosts"

    def ssh(self, command, check=True, timeout=None):
        return sh(["ssh", *self.opts(), f"ubuntu@{self.ip}", command], check=check, timeout=timeout)

    def opts(self):
        return ["-i", str(KEY), "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
                "-o", "BatchMode=yes"]


def arch_of(itype):
    archs = aws("ec2", "describe-instance-types", "--instance-types", itype)["InstanceTypes"][0][
        "ProcessorInfo"]["SupportedArchitectures"]
    return "arm64" if "arm64" in archs else "x86_64"


def ubuntu_ami(arch):
    imgs = aws("ec2", "describe-images", "--owners", UBUNTU_OWNER,
               "--filters", f"Name=name,Values={UBUNTU_NAME[arch]}", "Name=state,Values=available")
    return max(imgs["Images"], key=lambda i: i["CreationDate"])["ImageId"]


def launch(box, args):
    zones = aws("ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                "--filters", f"Name=instance-type,Values={box.itype}")["InstanceTypeOfferings"]
    zones = sorted(z["Location"] for z in zones)
    if not zones:
        raise RuntimeError(f"{box.itype} is not offered in {REGION}")
    subnets = {s["AvailabilityZone"]: s["SubnetId"] for s in
               aws("ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true")["Subnets"]}
    ami = ubuntu_ami(box.arch)
    # The guest hard cap: shutdown with terminate behaviour ends the instance.
    user_data = f"#!/bin/bash\nshutdown -h +{args.max_minutes}\n"
    tags = (f"ResourceType=instance,Tags=[{{Key=Owner,Value={OWNER}}},"
            f"{{Key=Name,Value=g16-matrix-{box.itype}-EPHEMERAL}}]")
    last = None
    for z in zones:
        if z not in subnets:
            continue
        r = aws("ec2", "run-instances", "--image-id", ami, "--instance-type", box.itype,
                "--key-name", KEY_NAME, "--security-group-ids", SG, "--subnet-id", subnets[z],
                "--instance-initiated-shutdown-behavior", "terminate",
                "--block-device-mappings",
                f"DeviceName=/dev/sda1,Ebs={{VolumeSize={args.disk_gb},VolumeType=gp3,DeleteOnTermination=true}}",
                "--user-data", user_data, "--tag-specifications", tags, check=False)
        if isinstance(r, dict):
            box.id = r["Instances"][0]["InstanceId"]
            say(f"launched {box.id} ({box.itype}) in {z}, ami {ami}")
            return
        last = r.stderr.strip()
        say(f"launch in {z} refused: {last.splitlines()[-1] if last else '?'}")
        if "InsufficientInstanceCapacity" not in last and "Unsupported" not in last:
            break
    raise RuntimeError(f"could not launch {box.itype}: {last}")


def arm_backstop(box, args):
    at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=args.max_minutes + 20)
    box.schedule = f"g16-matrix-stop-{box.id}"
    target = {"Arn": "arn:aws:scheduler:::aws-sdk:ec2:stopInstances", "RoleArn": STOP_ROLE,
              "Input": json.dumps({"InstanceIds": [box.id]}),
              "RetryPolicy": {"MaximumEventAgeInSeconds": 3600, "MaximumRetryAttempts": 10}}
    aws("scheduler", "create-schedule", "--name", box.schedule,
        "--schedule-expression", f"at({at:%Y-%m-%dT%H:%M:%S})", "--schedule-expression-timezone", "UTC",
        "--flexible-time-window", "Mode=OFF", "--action-after-completion", "DELETE",
        "--target", json.dumps(target))
    say(f"backstop stop schedule {box.schedule} at {at:%H:%M}Z")


def wait_ready(box):
    aws("ec2", "wait", "instance-running", "--instance-ids", box.id, check=True)
    box.ip = aws("ec2", "describe-instances", "--instance-ids", box.id)["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    say(f"running at {box.ip}; waiting for the console host key")
    # The host key comes from the console log, not from whoever answers on port 22.
    want = None
    # Bare-metal hosts take far longer to post than VMs.
    deadline = time.time() + (2400 if ".metal" in box.itype else 900)
    while time.time() < deadline and want is None:
        out = aws("ec2", "get-console-output", "--instance-id", box.id, "--latest", check=False)
        text = out.get("Output", "") if isinstance(out, dict) else ""
        m = re.search(r"BEGIN SSH HOST KEY FINGERPRINTS.*?(SHA256:\S+) \S+ \(ED25519\)", text, re.S)
        if m:
            want = m.group(1)
        else:
            time.sleep(15)
    if want is None:
        raise RuntimeError("no ED25519 host key fingerprint in the console log in time")
    while True:
        k = sh(["ssh-keyscan", "-T", "10", "-t", "ed25519", box.ip], check=False).stdout
        if k.strip():
            break
        time.sleep(5)
    tmp = box.results / "keyscan"
    tmp.write_text(k)
    got = sh(["ssh-keygen", "-lf", str(tmp)]).stdout.split()[1]
    if got != want:
        raise RuntimeError(f"host key mismatch: console {want}, scan {got}")
    box.known_hosts.write_text(k)
    tmp.unlink()
    for _ in range(40):
        if box.ssh("true", check=False).returncode == 0:
            say(f"ssh ok, host key {want}")
            return
        time.sleep(5)
    raise RuntimeError("ssh never came up")


def upload(box, bundle):
    say("uploading bundle")
    t = time.time()
    tar = subprocess.Popen(["tar", "-C", str(bundle), "-cf", "-", "."], stdout=subprocess.PIPE)
    r = subprocess.run(["ssh", *box.opts(), f"ubuntu@{box.ip}", "mkdir -p ~/m && tar -C ~/m -xf -"],
                       stdin=tar.stdout, capture_output=True, text=True)
    tar.wait()
    if r.returncode or tar.returncode:
        raise RuntimeError(f"upload failed: {r.stderr}")
    bad = box.ssh("cd ~/m && python3 -c \"import hashlib,json;m=json.load(open('manifest.json'));"
                  "print(sum(hashlib.sha256(open(p,'rb').read()).hexdigest()!=h for p,h in m['files'].items()))\"").stdout.strip()
    if bad != "0":
        raise RuntimeError(f"{bad} files changed in transfer")
    say(f"uploaded and hash-checked in {time.time() - t:.0f}s")


def run_guest(box, args):
    circuits = " ".join(args.circuits)
    box.ssh(f"cd ~/m && nohup bash matrix-guest.sh /home/ubuntu/m {args.reps} {circuits} "
            "> guest.log 2>&1 < /dev/null &")
    say("guest started")
    seen = 0
    deadline = time.time() + args.max_minutes * 60
    while time.time() < deadline:
        time.sleep(30)
        r = box.ssh("cat ~/m/results/progress.log 2>/dev/null; echo ===; cat ~/m/results/DONE 2>/dev/null; "
                    "echo ===; uptime", check=False)
        if r.returncode:
            say(f"poll failed: {r.stderr.strip()[:200]}")
            continue
        progress, done, up = r.stdout.split("===\n")
        lines = progress.splitlines()
        for l in lines[seen:]:
            say(f"  guest: {l}")
        seen = len(lines)
        if done.strip():
            say(f"guest finished, failed={done.strip()}; load{up.strip().split('load')[-1]}")
            return int(done.strip())
    raise RuntimeError("guest exceeded the time cap")


def retrieve(box):
    dest = box.results / "guest"
    dest.mkdir()
    ssh = subprocess.Popen(["ssh", *box.opts(), f"ubuntu@{box.ip}", "tar -C ~/m -cf - results guest.log manifest.json"],
                           stdout=subprocess.PIPE)
    r = subprocess.run(["tar", "-C", str(dest), "-xf", "-"], stdin=ssh.stdout, capture_output=True, text=True)
    ssh.wait()
    if r.returncode or ssh.returncode:
        raise RuntimeError(f"retrieve failed: {r.stderr}")
    say(f"retrieved results into {dest}")


def terminate(box):
    if box.id:
        say(f"terminating {box.id}")
        aws("ec2", "terminate-instances", "--instance-ids", box.id, check=False)
        r = aws("ec2", "wait", "instance-terminated", "--instance-ids", box.id, check=False)
        state = aws("ec2", "describe-instances", "--instance-ids", box.id)["Reservations"][0]["Instances"][0]["State"]["Name"]
        say(f"{box.id} is {state}")
    if box.schedule:
        aws("scheduler", "delete-schedule", "--name", box.schedule, check=False)


# ---------------------------------------------------------------------------
# local verification and summary


def ints(path):
    return [int(x) for x in json.loads(Path(path).read_text())]


def snarkjs_ok(vkey, public, proof):
    r = sh([str(SNARKJS), "groth16", "verify", str(vkey), str(public), str(proof)], check=False)
    return r.returncode == 0 and "OK!" in r.stdout


def median(xs):
    return round(statistics.median(xs), 3) if xs else None


def summarise(box, args, guest_failed):
    g = box.results / "guest/results"
    flags = next((l.split(":", 1)[1] for l in (g / "cpuinfo.txt").read_text().splitlines()
                  if l.startswith(("flags", "Features"))), "").split()
    model = next((l.split(":", 1)[1].strip() for l in (g / "lscpu.txt").read_text().splitlines()
                  if l.startswith("Model name")), "?")
    lscpu = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
             for l in (g / "lscpu.txt").read_text().splitlines() if ":" in l}
    summary = {
        "instance_type": box.itype, "instance_id": box.id, "cpu": model,
        "logical_cpus": int((g / "nproc.txt").read_text()),
        "threads_per_core": lscpu.get("Thread(s) per core"), "sockets": lscpu.get("Socket(s)"),
        "numa_nodes": lscpu.get("NUMA node(s)"), "l3": lscpu.get("L3 cache"),
        "arch": box.arch, "isa": {f: f in flags for f in ISA_FLAGS},
        "build": (g / "build.txt").read_text().strip(), "reps": args.reps,
        "bundle": json.loads((box.results / "guest/manifest.json").read_text())["source"],
        "guest_failed": guest_failed, "circuits": {},
    }
    ok_all = not guest_failed
    for c in args.circuits:
        d = g / c
        art = REPO / "bench/artifacts" / c
        ref = ints(art / "public.json")
        row = {}
        if (d / "ours.csv").exists():
            rows = [r for r in csv.DictReader(open(d / "ours.csv")) if r["mode"] == "warm"]
            ms = [float(r["ms"]) for r in rows]
            row["ours_ms"] = {"median": median(ms), "min": min(ms), "max": max(ms), "n": len(ms)}
            row["ours_stages_ms"] = {k: median([float(r[f"{k}_us"]) / 1000 for r in rows])
                                     for k in ["gather", "ntt", "pointwise", "msm", "assemble"]}
            row["ours_prepare_ms"] = median([float(r["prepare_ms"]) for r in rows])
            row["ours_self_verified_reps"] = sum(r["verified"] == "yes" for r in rows)
        if (d / "rs.out").exists():
            ms = [float(l.split()[2]) for l in (d / "rs.out").read_text().splitlines() if l.startswith("rep ")]
            if ms:
                row["rapidsnark_ms"] = {"median": median(ms), "min": min(ms), "max": max(ms), "n": len(ms)}
            pool = re.search(r"ffiasm_pool_threads=(\d+)", (d / "rs.err").read_text())
            row["rapidsnark_pool_threads"] = int(pool.group(1)) if pool else None
        for k in ["ours", "rs"]:
            if (d / f"{k}.rss").exists():
                row[f"{k}_peak_rss_mib"] = round(int((d / f"{k}.rss").read_text()) / 1024, 1)
        checks = {}
        pairs = {"ours": (d / "ours-public.json", d / "ours-proof.json"),
                 "rapidsnark_last": (d / "rs-last-public.json", d / "rs-last-proof.json")}
        rs_proofs = sorted((d / "rs-proofs").glob("proof-*.json"), key=lambda p: int(re.sub(r"\D", "", p.name)))
        if rs_proofs:
            first = rs_proofs[0]
            pairs["rapidsnark_first"] = (first.with_name(first.name.replace("proof", "public")), first)
        for name, (pub, prf) in pairs.items():
            good = pub.exists() and prf.exists() and snarkjs_ok(art / "vkey.json", pub, prf) and ints(pub) == ref
            checks[name] = good
            ok_all &= good
        row["snarkjs_and_reference_public"] = checks
        if row.get("ours_ms") and row.get("rapidsnark_ms"):
            row["speedup_vs_rapidsnark"] = round(row["rapidsnark_ms"]["median"] / row["ours_ms"]["median"], 3)
        ok_all &= bool(row.get("ours_ms")) and bool(row.get("rapidsnark_ms"))
        summary["circuits"][c] = row
    summary["accepted"] = ok_all
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    say(f"summary {out}: accepted={ok_all}")
    for c, r in summary["circuits"].items():
        say(f"  {c:15} ours {r.get('ours_ms', {}).get('median')} ms  rapidsnark "
            f"{r.get('rapidsnark_ms', {}).get('median')} ms  x{r.get('speedup_vs_rapidsnark')}  "
            f"checks {r['snarkjs_and_reference_public']}")


def run(args):
    arch = arch_of(args.instance_type)
    bundle = Path(args.arm64_bundle if arch == "arm64" else args.bundle or "").resolve()
    if not (bundle / "manifest.json").exists():
        sys.exit(f"{bundle} is not a prepared bundle")
    if json.loads((bundle / "manifest.json").read_text()).get("arch", "x86_64") != arch:
        sys.exit(f"{bundle} is not an {arch} bundle")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = Path(args.results).resolve() / f"{args.instance_type}-{stamp}"
    results.mkdir(parents=True)
    box = Box(args.instance_type, results)
    box.arch = arch
    guest_failed = None
    try:
        launch(box, args)
        arm_backstop(box, args)
        wait_ready(box)
        upload(box, bundle)
        guest_failed = run_guest(box, args)
        retrieve(box)
    except BaseException as e:
        say(f"ERROR: {e!r}")
        if box.ip and box.id and not (results / "guest").exists():
            try:
                retrieve(box)
            except Exception as e2:
                say(f"partial retrieve failed: {e2}")
        raise
    finally:
        terminate(box)
    summarise(box, args, guest_failed)


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    # Any kill still runs `run`'s finally, which terminates the instance.
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--out", required=True)
    a.add_argument("--arch", choices=list(BINARIES), default="x86_64")
    b = sub.add_parser("run")
    b.add_argument("--instance-type", required=True)
    b.add_argument("--bundle", help="x86_64 bundle")
    b.add_argument("--arm64-bundle", help="arm64 bundle, used for Graviton types")
    b.add_argument("--results", required=True, help="parent directory; each run gets a fresh subdirectory")
    b.add_argument("--summary", required=True, help="JSON file to write")
    b.add_argument("--reps", type=int, default=10)
    b.add_argument("--max-minutes", type=int, default=120, help="guest hard cap; the instance terminates itself")
    b.add_argument("--disk-gb", type=int, default=40)
    b.add_argument("--circuits", nargs="+", default=CIRCUITS)
    args = p.parse_args()
    prep(args) if args.cmd == "prep" else run(args)


if __name__ == "__main__":
    main()
