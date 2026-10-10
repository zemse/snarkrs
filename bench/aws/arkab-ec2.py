#!/usr/bin/env python3
"""Run bench/scripts/arkab.py (stock vs perf arkworks A/B) on one disposable EC2 instance.

    arkab-ec2.py prep --bin-dir target/arkab/bin-x86 --out target/arkab/bundle-x86 \
        --phase2 js_2x2_d32 ... --ptau local_13.ptau ...
    arkab-ec2.py run --instance-type c7i.4xlarge --bundle target/arkab/bundle-x86 \
        --results target/arkab/ec2 -- --backend cpu --phase2 js_2x2_d32 ...

Launch, backstop, host key, upload, retrieve and terminate are matrix.py's: the instance is
tagged Owner=bench-runner, shuts itself down (terminate behaviour) at the hard cap, has an
AWS scheduler stop as a backstop, its host key is taken from the console log and checked
strictly, and it is terminated in a `finally`. A GPU type gets the Deep Learning Base AMI,
which carries the NVIDIA driver and NVRTC; nothing is compiled on the instance. Arguments
after `--` go to arkab.py on the guest.
"""

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix  # noqa: E402
from matrix import REPO, aws, say, sha256, sh  # noqa: E402

CIRCUIT_FILES = ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"]
DLAMI = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"


def prep(args):
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists; prep writes a fresh bundle")
    bins = Path(args.bin_dir).resolve()
    (out / "bin").mkdir(parents=True)
    build = json.loads((bins / "build.json").read_text())
    for v in build["variants"]:
        for f in ["snarkrs", "verify_time", "field_ops"]:
            shutil.copy2(bins / f"{f}-{v}", out / "bin" / f"{f}-{v}")
    shutil.copy2(REPO / "bench/scripts/arkab.py", out / "arkab.py")
    art = REPO / "bench/artifacts"
    for c in args.circuits:
        (out / "artifacts" / c).mkdir(parents=True)
        files = CIRCUIT_FILES + ([p.name for p in (art / c).glob("*.r1cs")] if c in args.phase2 else [])
        for f in files:
            shutil.copy2(art / c / f, out / "artifacts" / c / f)
    if args.warm:
        (out / "artifacts" / args.warm).mkdir()
        for f in CIRCUIT_FILES:
            shutil.copy2(art / args.warm / f, out / "artifacts" / args.warm / f)
    (out / "ptau").mkdir()
    for p in args.ptau:
        shutil.copy2(Path(args.ptau_dir) / p, out / "ptau" / p)
    manifest = {"source": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip(),
                "arch": "x86_64", "build": build, "files": {}}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(out))] = sha256(p)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    say(f"bundle {out}: {len(manifest['files'])} files")


def gpu_of(itype):
    t = aws("ec2", "describe-instance-types", "--instance-types", itype)["InstanceTypes"][0]
    return bool(t.get("GpuInfo"))


def dlami():
    imgs = aws("ec2", "describe-images", "--owners", "amazon", "--filters", f"Name=name,Values={DLAMI}",
               "Name=state,Values=available", "Name=architecture,Values=x86_64")["Images"]
    img = max(imgs, key=lambda i: i["CreationDate"])
    root = next(b for b in img["BlockDeviceMappings"] if b["DeviceName"] == img["RootDeviceName"])
    if img["RootDeviceName"] != "/dev/sda1":
        raise RuntimeError(f"{img['ImageId']} roots at {img['RootDeviceName']}, matrix.py maps /dev/sda1")
    return img["ImageId"], root["Ebs"]["VolumeSize"], img["Name"]


def run_guest(box, args, variants):
    guest = " ".join(args.guest)
    vs = " ".join(variants)
    # field_ops twice per build first: the per-primitive check that the builds differ here.
    pre = (f"for r in 1 2; do for v in {vs}; do echo == $v round $r; bin/field_ops-$v; done; done "
           f"> results/field_ops.txt 2>&1; ")
    if args.warm:
        # The first NVRTC compile of a unit costs about two minutes; pay it before any timing.
        pre += (f"for v in {vs}; do bin/snarkrs-$v groth16 prove artifacts/{args.warm}/circuit.zkey "
                f"artifacts/{args.warm}/circuit.wtns /tmp/w.json /tmp/wp.json --backend cuda "
                f"> results/warm-$v.log 2>&1; done; ")
    cmd = ("cd ~/m && mkdir -p results && lscpu > results/lscpu.txt && nproc > results/nproc.txt && "
           "(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > results/gpu.txt 2>&1 || true) && "
           f"nohup bash -c '{pre}python3 arkab.py --bin-dir bin --artifacts artifacts --ptau-dir ptau "
           f"--variants {vs} --out results {guest}' > guest.log 2>&1 < /dev/null &")
    box.ssh(cmd)
    say(f"guest started: {guest}")
    seen = 0
    deadline = time.time() + args.max_minutes * 60
    while time.time() < deadline:
        time.sleep(30)
        r = box.ssh("cat ~/m/guest.log 2>/dev/null; echo ===; cat ~/m/results/DONE 2>/dev/null; echo ===; uptime",
                    check=False)
        if r.returncode:
            say(f"poll failed: {r.stderr.strip()[:200]}")
            continue
        log, done, up = r.stdout.split("===\n")
        lines = log.splitlines()
        for line in lines[seen:]:
            say(f"  guest: {line}")
        seen = len(lines)
        if done.strip():
            say(f"guest finished, failures={done.strip()}; load{up.strip().split('load')[-1]}")
            return int(done.strip())
    raise RuntimeError("guest exceeded the time cap")


def run(args):
    bundle = Path(args.bundle).resolve()
    if not (bundle / "manifest.json").exists():
        sys.exit(f"{bundle} is not a prepared bundle")
    if matrix.arch_of(args.instance_type) != "x86_64":
        sys.exit("x86_64 types only")
    if gpu_of(args.instance_type):
        args.ami, root_gb, name = dlami()
        args.disk_gb = max(args.disk_gb, root_gb + 40)
        say(f"GPU type: {name} ({args.ami}), root {args.disk_gb} GB")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = Path(args.results).resolve() / f"{args.instance_type}-{stamp}"
    results.mkdir(parents=True)
    box = matrix.Box(args.instance_type, results)
    box.arch = "x86_64"
    try:
        matrix.launch(box, args)
        matrix.arm_backstop(box, args)
        matrix.wait_ready(box)
        matrix.upload(box, bundle)
        variants = list(json.loads((bundle / "manifest.json").read_text())["build"]["variants"])
        failures = run_guest(box, args, variants)
        matrix.retrieve(box)
    except BaseException as e:
        say(f"ERROR: {e!r}")
        if box.ip and box.id and not (results / "guest").exists():
            try:
                matrix.retrieve(box)
            except Exception as e2:
                say(f"partial retrieve failed: {e2}")
        raise
    finally:
        matrix.terminate(box)
    (results / "instance.json").write_text(json.dumps({"instance_type": box.itype, "instance_id": box.id,
                                                        "ami": getattr(args, "ami", None),
                                                        "guest_failures": failures}, indent=1) + "\n")
    say(f"results in {results}")


def main():
    matrix.signal.signal(matrix.signal.SIGTERM, matrix.interrupted)
    matrix.signal.signal(matrix.signal.SIGHUP, matrix.interrupted)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--bin-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--circuits", nargs="+", default=matrix.CIRCUITS)
    a.add_argument("--phase2", nargs="*", default=[], help="circuits whose r1cs is shipped")
    a.add_argument("--ptau-dir", default=str(REPO / "bench/ptau"))
    a.add_argument("--ptau", nargs="*", default=[])
    a.add_argument("--warm", help="an extra small circuit, for the CUDA kernel-cache warm-up")
    b = sub.add_parser("run")
    b.add_argument("--instance-type", required=True)
    b.add_argument("--bundle", required=True)
    b.add_argument("--results", required=True)
    b.add_argument("--max-minutes", type=int, default=180)
    b.add_argument("--disk-gb", type=int, default=60)
    b.add_argument("--warm", help="circuit to prove once per build on CUDA before anything is timed")
    b.add_argument("guest", nargs=argparse.REMAINDER, help="-- then arkab.py arguments")
    args = p.parse_args()
    if args.cmd == "prep":
        prep(args)
    else:
        args.name = "g16-arkab"
        if args.guest[:1] == ["--"]:
            args.guest = args.guest[1:]
        run(args)


if __name__ == "__main__":
    main()
