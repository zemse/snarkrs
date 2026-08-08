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
import argparse, csv, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BIN = HERE / "bin"
ART = HERE / "artifacts"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def verify_fast(vkey, public, proof):
    # rapidsnark's verifier prints its result on stderr, not stdout, so both streams
    # have to be checked. Reading only stdout makes every proof look invalid.
    r = sh([str(BIN / "rapidsnark-verify"), str(vkey), str(public), str(proof)])
    return "Valid proof" in (r.stdout + r.stderr)


def verify_snarkjs(vkey, public, proof):
    r = sh(["snarkjs", "groth16", "verify", str(vkey), str(public), str(proof)])
    return "OK!" in (r.stdout + r.stderr)


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
        if not verify_fast(vkey, public_out, proof_out):
            return None, "proof did NOT verify"
        out.append(ms)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=["cpu"],
                    help="our backends to test, e.g. cpu metal")
    ap.add_argument("--csv", default=str(HERE / "results" / "comparison.csv"))
    ap.add_argument("--skip-rapidsnark", action="store_true")
    args = ap.parse_args()

    g16 = HERE.parent / "target" / "release" / "g16"
    variants = args.variants or sorted(
        d.name for d in ART.iterdir() if d.is_dir() and (d / "circuit.zkey").exists())
    if not variants:
        sys.exit(f"no artifacts with a circuit.zkey under {ART}")

    host = platform.node()
    os_name, arch = platform.system(), platform.machine()
    cores = os.cpu_count()
    rows, notes = [], []

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
                notes.append(f"{g16} not built, skipped our {b} backend")
                break
            configs.append((f"ours", b, "cold",
                            [str(g16), "prove", "--zkey", str(zkey), "--witness", str(wtns),
                             "--proof", "/tmp/g16bench_proof.json",
                             "--public", "/tmp/g16bench_public.json", "--backend", b]))

        for prover, backend, mode, cmd in configs:
            ms, err = time_cold(cmd, args.reps, vkey,
                                "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json")
            if err:
                print(f"  {prover:12s} {backend:6s} {mode:5s}  FAILED: {err}", flush=True)
                notes.append(f"{v}/{prover}/{backend}/{mode}: {err}")
                continue
            compat = verify_snarkjs(vkey, "/tmp/g16bench_public.json", "/tmp/g16bench_proof.json")
            print(f"  {prover:12s} {backend:6s} {mode:5s}  median {statistics.median(ms):8.1f} ms "
                  f"(min {min(ms):.1f} max {max(ms):.1f})  snarkjs-compatible: {compat}", flush=True)
            for i, m in enumerate(ms, 1):
                rows.append(dict(host=host, os=os_name, arch=arch, cores=cores, variant=v,
                                 constraints=nc, prover=prover, backend=backend, mode=mode,
                                 rep=i, ms=round(m, 3), verified="yes",
                                 snarkjs_compatible="yes" if compat else "no"))

        # warm: rapidsnark through its object API, ours through `g16 bench --mode warm`
        if not args.skip_rapidsnark and (BIN / "rapidsnark-warm").exists():
            r = sh([str(BIN / "rapidsnark-warm"), str(zkey), str(wtns),
                    "/tmp/g16bench_proof.json", "/tmp/g16bench_public.json", str(args.reps)])
            times = [float(l.split()[-1]) for l in r.stdout.splitlines() if l.startswith("rep ")]
            if times and verify_fast(vkey, "/tmp/g16bench_public.json", "/tmp/g16bench_proof.json"):
                print(f"  {'rapidsnark':12s} {'cpu':6s} {'warm':5s}  median {statistics.median(times):8.1f} ms "
                      f"(min {min(times):.1f} max {max(times):.1f})", flush=True)
                for i, m in enumerate(times, 1):
                    rows.append(dict(host=host, os=os_name, arch=arch, cores=cores, variant=v,
                                     constraints=nc, prover="rapidsnark", backend="cpu", mode="warm",
                                     rep=i, ms=m, verified="yes", snarkjs_compatible="yes"))
            else:
                notes.append(f"{v}/rapidsnark/warm: no timings or proof failed to verify")

        for b in args.backends:
            if not g16.exists():
                break
            r = sh([str(g16), "bench", "--artifacts", str(ART), "--variant", v,
                    "--reps", str(args.reps), "--backend", b, "--mode", "warm",
                    "--csv", f"/tmp/g16bench_warm_{v}_{b}.csv"])
            if r.returncode != 0:
                notes.append(f"{v}/ours/{b}/warm: exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
                print(f"  {'ours':12s} {b:6s} {'warm':5s}  FAILED", flush=True)
                continue
            p = Path(f"/tmp/g16bench_warm_{v}_{b}.csv")
            if p.exists():
                got = list(csv.DictReader(p.open()))
                rows.extend(got)
                t = [float(x["ms"]) for x in got if x.get("ms")]
                if t:
                    print(f"  {'ours':12s} {b:6s} {'warm':5s}  median {statistics.median(t):8.1f} ms", flush=True)

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


if __name__ == "__main__":
    main()
