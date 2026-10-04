#!/usr/bin/env python3
"""Benchmark our prover against rapidsnark on the same artifacts, cold and warm.

Every recorded timing belongs to a proof that was independently verified first. A
benchmark that times a broken prover is worse than no benchmark, and "it produced a
file" is not the same as "it produced a proof".

Three verification oracles are available and they answer different questions:
  our own verifier   does our pairing check accept it
  rapidsnark-verify  does an independent C++ implementation accept it
  snarkjs verify     is the JSON encoding ecosystem-compatible
The loop uses rapidsnark-verify because it is fast; snarkjs runs once per
configuration as the encoding cross-check, because it costs about a second a call.

Cold and warm mean exactly one thing each here:
  cold  one fresh process per proof, so the zkey parse and (on GPU) the upload are
        inside the timed region. This is what every CLI prover actually does.
  warm  setup paid once, then prove in a loop. This is what a resident service does,
        and it is the only mode in which a GPU backend can look good, which is
        precisely why vendor charts prefer it.
"""
import argparse, csv, json, math, os, platform, shutil, statistics, subprocess, sys, time
from pathlib import Path

from check_reference_public import matches_reference

HERE = Path(__file__).resolve().parent.parent
BIN = HERE / "bin"
ART = HERE / "artifacts"


def sh(cmd, **kw):
    # A probe for a tool that may not exist is a normal outcome, not a crash. `nvidia-smi`
    # is absent on every Mac, and letting the FileNotFoundError escape made detect_gpu()
    # kill the whole run before its own Darwin fallback could be reached.
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=f"{cmd[0]}: not found")


def load_average():
    """One-minute load average, or None where the platform will not say."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def verify_fast(vkey, public, proof):
    # rapidsnark's verifier prints its result on stderr, not stdout, so both streams
    # have to be checked. Reading only stdout makes every proof look invalid.
    r = sh([str(BIN / "rapidsnark-verify"), str(vkey), str(public), str(proof)])
    return r.returncode == 0 and "Result: Valid proof" in (r.stdout + r.stderr)


HAVE_SNARKJS = shutil.which("snarkjs") is not None


def verify_snarkjs(vkey, public, proof):
    """True, False, or None when snarkjs is not installed.

    The tri-state matters. An earlier version returned False when snarkjs was simply
    absent, which wrote "snarkjs_compatible: no" into the CSV and read as a real
    encoding incompatibility. Not measured and measured-bad have to look different.
    """
    if not HAVE_SNARKJS:
        return None
    r = sh(["snarkjs", "groth16", "verify", str(vkey), str(public), str(proof)])
    return r.returncode == 0 and "OK!" in (r.stdout + r.stderr)


def compat_cell(c):
    return "unknown" if c is None else ("yes" if c else "no")


def detect_gpu():
    """Name of the accelerator, for the record. Empty when there is none to name."""
    r = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().splitlines()[0].strip()
    if platform.system() == "Darwin":
        r = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return ""


def detect_machine():
    """A short label for the box, so two result sets can never be mistaken for one.

    On EC2 the instance type is the only label anyone can act on, and it is available
    from the instance metadata service without credentials. IMDSv2 needs the token
    dance; a one second timeout keeps this from stalling on a non-EC2 host.
    """
    if platform.system() == "Linux":
        tok = sh(["curl", "-s", "-m", "1", "-X", "PUT",
                  "http://169.254.169.254/latest/api/token",
                  "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"])
        if tok.returncode == 0 and tok.stdout.strip():
            r = sh(["curl", "-s", "-m", "1",
                    "http://169.254.169.254/latest/meta-data/instance-type",
                    "-H", f"X-aws-ec2-metadata-token: {tok.stdout.strip()}"])
            if r.returncode == 0 and r.stdout.strip():
                # The bare EC2 instance type. It is the canonical name for the box and
                # the string you type to rent the same one again. An earlier version
                # appended the GPU, giving aws-g4dn.2xlarge-Tesla-T4, which buries a proper
                # slug inside a compound a reader cannot parse: there is no way to tell
                # where the instance type ends and the accelerator begins. The GPU is
                # already recorded separately in the run stamp's `gpu` field.
                return r.stdout.strip()
    if platform.system() == "Darwin":
        return detect_gpu().replace(" ", "-") or platform.node()
    return platform.node()


def time_cold(cmd, reps, vkey, proof_out, public_out):
    """One fresh process per rep. Wall clock around the whole invocation."""
    out = []
    for i in range(reps):
        for f in (proof_out, public_out):
            Path(f).unlink(missing_ok=True)
        t0 = time.perf_counter()
        r = sh(cmd)
        ms = (time.perf_counter() - t0) * 1000.0
        if r.returncode != 0:
            return None, f"exit {r.returncode}: {(r.stderr or r.stdout)[:200]}"
        try:
            if not matches_reference(vkey, public_out):
                return None, "public signals differ from the fixture reference"
        except (OSError, ValueError) as error:
            return None, f"public signal check failed: {error}"
        if not verify_fast(vkey, public_out, proof_out):
            return None, "proof did NOT verify"
        out.append(ms)
    return out, None


def valid_warm_rows(rows, variant, backend, reps):
    try:
        return (len(rows) == reps
                and [row["rep"] for row in rows] == [str(i) for i in range(1, reps + 1)]
                and all(row["variant"] == variant and row["backend"] == backend
                        and row["mode"] == "warm" and row["prover"] == "ours"
                        and row["verified"] == "yes"
                        and math.isfinite(float(row["ms"])) and float(row["ms"]) >= 0
                        for row in rows))
    except (KeyError, ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--cold-reps", type=int, default=None)
    ap.add_argument("--warm-reps", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=["cpu"],
                    help="our backends to test, e.g. cpu metal")
    ap.add_argument("--csv", default=None,
                    help="default: results/comparison-<machine>.csv, so two machines "
                         "never overwrite each other")
    ap.add_argument("--machine", default=None,
                    help="label for this box, recorded in every row (default: autodetected)")
    ap.add_argument("--skip-rapidsnark", action="store_true")
    ap.add_argument("--fail-fast", action="store_true",
                    help="retain completed rows and stop on the first failed configuration")
    ap.add_argument("--warm-oracle", default=None,
                    help="independent verifier with the snarkjs command interface; "
                         "does not imply snarkjs encoding compatibility")
    ap.add_argument("--max-load", type=float, default=None,
                    help="refuse to run when the 1-minute load average is above this "
                         "(default: cores/4). A timing taken on a busy box is not a "
                         "measurement of this program.")
    ap.add_argument("--allow-loaded", action="store_true",
                    help="run anyway, and mark every row loaded=yes")
    args = ap.parse_args()
    cold_reps = args.reps if args.cold_reps is None else args.cold_reps
    warm_reps = args.reps if args.warm_reps is None else args.warm_reps
    if min(args.reps, cold_reps, warm_reps) <= 0:
        ap.error("reps must be positive")

    machine = args.machine or detect_machine()
    gpu = detect_gpu()
    if args.csv is None:
        safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in machine)
        args.csv = str(HERE / "results" / f"comparison-{safe}.csv")

    g16 = HERE.parent / "target" / "release" / "snarkrs"
    variants = args.variants or sorted(
        d.name for d in ART.iterdir() if d.is_dir() and (d / "circuit.zkey").exists())
    if not variants:
        sys.exit(f"no artifacts with a circuit.zkey under {ART}")

    host = platform.node()
    os_name, arch = platform.system(), platform.machine()
    cores = os.cpu_count()
    rows, notes = [], []

    def note(message):
        notes.extend([message])
        if args.fail_fast:
            outp = Path(args.csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            if rows:
                with outp.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list({k: None for r in rows for k in r}))
                    writer.writeheader()
                    writer.writerows(rows)
            sys.exit(message)

    print(f"machine: {machine}   gpu: {gpu or '(none)'}   cores: {cores}   "
          f"snarkjs: {'yes' if HAVE_SNARKJS else 'NOT INSTALLED, encoding cross-check skipped'}")
    if args.skip_rapidsnark:
        print("scope: ours-only CPU/GPU timings; rapidsnark prover timings explicitly skipped")
    else:
        for name in ("rapidsnark", "rapidsnark-warm"):
            if not (BIN / name).exists():
                note(f"requested competitor missing: {BIN / name}; pass --skip-rapidsnark for ours-only timings")
    # A benchmark run on a loaded machine measures the other tenant, not the prover. This
    # round found a local run at load average 20 where the 70k-constraint CPU proof came out
    # SLOWER than the 140k one, which is arithmetically impossible and was the only reason
    # the contamination was noticed at all. Refuse by default rather than rely on noticing.
    load = load_average()
    ceiling = args.max_load if args.max_load is not None else (cores or 4) / 4
    loaded = load is not None and load > ceiling
    if loaded and not args.allow_loaded:
        sys.exit(
            f"load average is {load:.2f}, above the ceiling of {ceiling:.2f} for a "
            f"{cores}-core box.\nA timing taken now measures whatever else is running. "
            f"Wait for the box to go idle, or pass --allow-loaded to record anyway "
            f"(every row will be marked loaded=yes so the numbers can never be mistaken "
            f"for clean ones)."
        )
    if load is not None:
        print(f"load average: {load:.2f} (ceiling {ceiling:.2f})"
              + ("  MARKED LOADED" if loaded else ""))
    stamp = dict(machine=machine, gpu=gpu, host=host, os=os_name, arch=arch, cores=cores,
                 load1=("" if load is None else round(load, 2)),
                 loaded=("yes" if loaded else "no"))

    for v in variants:
        d = ART / v
        zkey, wtns, vkey = d / "circuit.zkey", d / "circuit.wtns", d / "vkey.json"
        try:
            nc = int(next(l.split()[-1] for l in (d / "r1cs-info.txt").read_text().splitlines()
                          if "# of Constraints" in l))
        except Exception:
            nc = -1
        print(f"\n=== {v} ({nc} constraints) ===", flush=True)

        configs = []
        if not args.skip_rapidsnark and (BIN / "rapidsnark").exists():
            configs.append(("rapidsnark", "cpu", "cold",
                            [str(BIN / "rapidsnark"), str(zkey), str(wtns),
                             "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json"]))
        for b in args.backends:
            if not g16.exists():
                note(f"{g16} not built, skipped our {b} backend")
                break
            configs.append((f"ours", b, "cold",
                            [str(g16), "groth16", "prove", str(zkey), str(wtns),
                             "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json",
                             "--backend", b]))

        for prover, backend, mode, cmd in configs:
            ms, err = time_cold(cmd, cold_reps, vkey,
                                "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json")
            if err:
                print(f"  {prover:12s} {backend:6s} {mode:5s}  FAILED: {err}", flush=True)
                note(f"{v}/{prover}/{backend}/{mode}: {err}")
                continue
            compat = verify_snarkjs(vkey, "/tmp/g16bench_public.json", "/tmp/g16bench_proof.json")
            print(f"  {prover:12s} {backend:6s} {mode:5s}  median {statistics.median(ms):8.1f} ms "
                  f"(min {min(ms):.1f} max {max(ms):.1f})  snarkjs-compatible: {compat_cell(compat)}", flush=True)
            for i, m in enumerate(ms, 1):
                rows.append(dict(**stamp, variant=v,
                                 constraints=nc, prover=prover, backend=backend, mode=mode,
                                 rep=i, ms=round(m, 3), verified="yes",
                                 snarkjs_compatible=compat_cell(compat),
                                 independent_verifier="rapidsnark-verify",
                                 independent_verification_scope="each-proof"))

        # warm: rapidsnark through its object API, ours through `snarkrs bench --mode warm`
        if not args.skip_rapidsnark and (BIN / "rapidsnark-warm").exists():
            r = sh([str(BIN / "rapidsnark-warm"), str(zkey), str(wtns),
                    "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json", str(warm_reps)])
            times = [float(l.split()[-1]) for l in r.stdout.splitlines() if l.startswith("rep ")]
            if times and verify_fast(vkey, "/tmp/g16bench_public.json", "/tmp/g16bench_proof.json"):
                print(f"  {'rapidsnark':12s} {'cpu':6s} {'warm':5s}  median {statistics.median(times):8.1f} ms "
                      f"(min {min(times):.1f} max {max(times):.1f})", flush=True)
                for i, m in enumerate(times, 1):
                    rows.append(dict(**stamp, variant=v,
                                     constraints=nc, prover="rapidsnark", backend="cpu", mode="warm",
                                     rep=i, ms=m, verified="yes",
                                     snarkjs_compatible=compat_cell(True if HAVE_SNARKJS else None)))
            else:
                note(f"{v}/rapidsnark/warm: no timings or proof failed to verify")

        for b in args.backends:
            if not g16.exists():
                break
            warm_cmd = [str(g16), "bench", "--artifacts", str(ART), "--variant", v,
                        "--reps", str(warm_reps), "--backend", b, "--mode", "warm",
                        "--csv", f"/tmp/g16bench_warm_{v}_{b}.csv"]
            if args.warm_oracle:
                warm_cmd.extend(["--snarkjs", args.warm_oracle])
            p = Path(f"/tmp/g16bench_warm_{v}_{b}.csv")
            p.unlink(missing_ok=True)
            r = sh(warm_cmd)
            if r.returncode != 0:
                note(f"{v}/ours/{b}/warm: exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
                print(f"  {'ours':12s} {b:6s} {'warm':5s}  FAILED", flush=True)
                continue
            if p.exists():
                with p.open() as f:
                    got = list(csv.DictReader(f))
                if not valid_warm_rows(got, v, b, warm_reps):
                    note(f"{v}/ours/{b}/warm: incomplete or unverified timings")
                    continue
                # snarkrs bench writes its own columns; stamp the machine on them too or the
                # warm rows would be the only ones in the file with no machine label.
                for row in got:
                    row.update(stamp)
                    row["snarkjs_compatible"] = "unknown"
                    if args.warm_oracle:
                        row["independent_verifier"] = Path(args.warm_oracle).name
                        row["independent_verification_scope"] = "first-proof"
                rows.extend(got)
                t = [float(x["ms"]) for x in got if x.get("ms")]
                if t:
                    print(f"  {'ours':12s} {b:6s} {'warm':5s}  median {statistics.median(t):8.1f} ms", flush=True)
            else:
                note(f"{v}/ours/{b}/warm: no timing CSV")

    outp = Path(args.csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list({k: None for r in rows for k in r})
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {outp}")
    if notes:
        print("\nNOT MEASURED:")
        for n in notes:
            print(f"  - {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
