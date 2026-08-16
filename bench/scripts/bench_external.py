#!/usr/bin/env python3
"""Time the provers that are not ours: rapidsnark and snarkjs, on the same artifacts.

Rows are written in the same schema `g16 bench --csv` emits, so one renderer reads every
prover's numbers out of one directory without knowing which tool produced them. The only
column that tells them apart is `prover`: `g16 bench` stamps `ours`, this stamps
`rapidsnark` or `snarkjs`.

Cold and warm mean here exactly what they mean for our backend. Cold is one fresh process
per proof, so the zkey parse is inside the timed region. Warm pays the parse once. That
distinction is the whole reason `rapidsnark-warm` exists: rapidsnark's stock CLI reparses
the zkey on every invocation and so can only ever report cold, and comparing our warm
number against its cold one would be a lie in our favour.

snarkjs is cold-only and deliberately gets fewer reps: it is wasm, it is minutes at 10^5
constraints, and a median over 3 slow reps is a better use of wall clock than a median
over 15. snarkjs has no warm mode to measure at all, because its CLI is the only interface
it offers.

Nothing here is required to exist. A missing rapidsnark binary or an uninstalled snarkjs
is a normal outcome on a fresh box, so this script skips that prover and still exits 0,
letting the run that called it produce a table with that prover's column left blank.

Every timing kept belongs to a proof that was verified first. If no verifier is available
at all, the timings are discarded rather than recorded unverified, because a benchmark of
a prover that is silently emitting garbage is worse than no benchmark.
"""
import argparse, csv, os, platform, shutil, statistics, subprocess, sys, tempfile, time

FIELDS = ["host", "os", "arch", "cores", "variant", "constraints", "prover", "backend",
          "mode", "rep", "ms", "prepare_ms", "gather_us", "ntt_us", "pointwise_us",
          "msm_us", "assemble_us", "verified"]

# Where a pnpm or npm install of snarkjs lands when its bin directory is not on the PATH
# of a non-interactive shell, which is the usual case when this runs from a script.
SNARKJS_FALLBACKS = [
    "~/Library/pnpm/snarkjs",        # pnpm on macOS
    "~/.local/share/pnpm/snarkjs",   # pnpm on Linux
    "~/.npm-global/bin/snarkjs",
    "/usr/local/bin/snarkjs",
    "/opt/homebrew/bin/snarkjs",
]


def sh(cmd):
    """Run a command that may not exist. A probe for an absent tool is a normal outcome,
    not a crash: letting FileNotFoundError escape would abort the whole benchmark run
    because one optional prover was not installed."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, PermissionError, OSError) as e:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(e))


def base_row():
    return {"host": platform.node(), "os": platform.system(),
            "arch": platform.machine(), "cores": os.cpu_count() or 0}


def constraints_of(d):
    p = os.path.join(d, "r1cs-info.txt")
    if not os.path.exists(p):
        return -1
    with open(p) as fh:
        for line in fh:
            if "# of Constraints" in line:
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return -1
    return -1


def resolve_snarkjs(explicit):
    """The launcher path, or "" when snarkjs is not on this box. Empty is a normal answer."""
    if explicit:
        p = os.path.expanduser(explicit)
        return p if os.path.exists(p) and os.access(p, os.X_OK) else ""
    found = shutil.which("snarkjs")
    if found:
        return found
    for cand in SNARKJS_FALLBACKS:
        cand = os.path.expanduser(cand)
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return ""


def make_verifier(rapidsnark_verify, g16, snarkjs):
    """Pick a verification oracle, cheapest first, and return (fn, name) or (None, "").

    Three oracles answer three different questions, and any one of them catches a prover
    that is emitting nonsense:
      rapidsnark-verify  an independent C++ implementation accepts it
      g16 verify         our own pairing check accepts it
      snarkjs verify     the JSON encoding is what the ecosystem expects
    The cold loop calls this once per rep, so cost decides the order: the two native
    verifiers are milliseconds, snarkjs is about a second a call.
    """
    if rapidsnark_verify and os.path.exists(rapidsnark_verify):
        def f(vkey, public, proof):
            # rapidsnark's verifier prints its result on stderr, not stdout, so both
            # streams have to be checked. Reading only stdout makes every proof look bad.
            r = sh([rapidsnark_verify, vkey, public, proof])
            return "Valid proof" in (r.stdout + r.stderr)
        return f, "rapidsnark-verify"
    if g16 and os.path.exists(g16):
        def f(vkey, public, proof):
            r = sh([g16, "verify", "--vkey", vkey, "--proof", proof, "--public", public])
            return r.returncode == 0
        return f, "g16 verify"
    if snarkjs:
        def f(vkey, public, proof):
            r = sh([snarkjs, "groth16", "verify", vkey, public, proof])
            return r.returncode == 0 and "OK" in (r.stdout + r.stderr)
        return f, "snarkjs verify"
    return None, ""


def clear(*paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def time_cold(cmd, reps, verify, vkey, proof, public):
    """One fresh process per rep, wall clock around the whole invocation.

    The output files are removed before every rep. Without that, a prover that exits 0
    without writing anything gets verified against the previous rep's proof, and the run
    reports timings for proofs that were never produced.
    """
    out = []
    for _ in range(reps):
        clear(proof, public)
        t = time.perf_counter()
        r = sh(cmd)
        dt = (time.perf_counter() - t) * 1e3
        if r.returncode != 0:
            return None, f"exit {r.returncode}: {(r.stderr or r.stdout)[:200].strip()}"
        if not (os.path.exists(proof) and os.path.exists(public)):
            return None, "prover exited 0 but wrote no proof"
        if not verify(vkey, public, proof):
            return None, "proof did NOT verify"
        out.append(dt)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--snarkjs-reps", type=int, default=3)
    ap.add_argument("--rapidsnark", default="")
    ap.add_argument("--rapidsnark-warm", default="")
    ap.add_argument("--rapidsnark-verify", default="",
                    help="fast verification oracle; falls back to g16, then snarkjs")
    ap.add_argument("--g16", default="", help="our CLI, used as a fallback verifier")
    ap.add_argument("--snarkjs", default="",
                    help="path to the snarkjs launcher (default: PATH, then the usual "
                         "pnpm and npm locations)")
    ap.add_argument("--skip-rapidsnark", action="store_true")
    ap.add_argument("--skip-snarkjs", action="store_true")
    a = ap.parse_args()

    d = os.path.join(a.artifacts, a.variant)
    zkey, wtns = os.path.join(d, "circuit.zkey"), os.path.join(d, "circuit.wtns")
    vkey = os.path.join(d, "vkey.json")
    for p in (zkey, wtns, vkey):
        if not os.path.exists(p):
            print(f"  external: {a.variant} has no {os.path.basename(p)}, skipping")
            return 0
    nc = constraints_of(d)

    snarkjs = resolve_snarkjs(a.snarkjs)
    verify, oracle = make_verifier(a.rapidsnark_verify, a.g16, snarkjs)
    if verify is None:
        # Refuse rather than record. An unverified timing looks exactly like a verified one
        # in the finished table, and there is no way to tell them apart after the fact.
        print("  external: no verification oracle available "
              "(rapidsnark-verify, g16 and snarkjs are all missing), recording nothing")
        return 0

    # A private scratch directory per invocation. Fixed /tmp names collide when two
    # benchmark runs overlap on one box, and a collision there means one run verifies the
    # other run's proof.
    tmpdir = tempfile.mkdtemp(prefix="g16-external-")
    tmp_proof = os.path.join(tmpdir, "proof.json")
    tmp_public = os.path.join(tmpdir, "public.json")

    rows = []

    def emit(prover, mode, ms_list):
        for i, ms in enumerate(ms_list, 1):
            r = base_row()
            r.update(variant=a.variant, constraints=nc, prover=prover, backend="cpu",
                     mode=mode, rep=i, ms=round(ms, 3), verified="yes")
            rows.append(r)

    try:
        have_rs = bool(a.rapidsnark) and os.path.exists(a.rapidsnark)
        if a.skip_rapidsnark or not have_rs:
            why = "--skip-rapidsnark" if a.skip_rapidsnark else "binary not found"
            print(f"  rapidsnark   cold  skipped ({why})")
        else:
            ms, err = time_cold([a.rapidsnark, zkey, wtns, tmp_proof, tmp_public],
                                a.reps, verify, vkey, tmp_proof, tmp_public)
            if ms:
                emit("rapidsnark", "cold", ms)
                print(f"  rapidsnark   cold  median {statistics.median(ms):8.1f} ms  "
                      f"({len(ms)} reps, verified by {oracle})")
            else:
                print(f"  rapidsnark   cold  FAILED: {err}")

        have_warm = bool(a.rapidsnark_warm) and os.path.exists(a.rapidsnark_warm)
        if a.skip_rapidsnark or not have_warm:
            why = "--skip-rapidsnark" if a.skip_rapidsnark else "binary not found"
            print(f"  rapidsnark   warm  skipped ({why})")
        else:
            clear(tmp_proof, tmp_public)
            r = sh([a.rapidsnark_warm, zkey, wtns, tmp_proof, tmp_public, str(a.reps)])
            if r.returncode != 0:
                print(f"  rapidsnark   warm  FAILED: exit {r.returncode}: "
                      f"{(r.stderr or r.stdout)[:160].strip()}")
            else:
                # `rep <i> <ms>` on stdout, one line per rep. The loop lives inside the
                # binary, so only the last proof survives to be checked: unlike the cold
                # path this verifies once, not per rep.
                ms = []
                for line in r.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == "rep":
                        try:
                            ms.append(float(parts[2]))
                        except ValueError:
                            pass
                if not ms:
                    print("  rapidsnark   warm  FAILED: no `rep` lines on stdout")
                elif not (os.path.exists(tmp_proof) and os.path.exists(tmp_public)):
                    print("  rapidsnark   warm  FAILED: no proof written, "
                          f"{len(ms)} timings discarded")
                elif not verify(vkey, tmp_public, tmp_proof):
                    print("  rapidsnark   warm  FAILED: proof did NOT verify, "
                          f"{len(ms)} timings discarded")
                else:
                    emit("rapidsnark", "warm", ms)
                    print(f"  rapidsnark   warm  median {statistics.median(ms):8.1f} ms  "
                          f"({len(ms)} reps, verified by {oracle})")

        if a.skip_snarkjs or not snarkjs:
            why = "--skip-snarkjs" if a.skip_snarkjs else "not installed"
            print(f"  snarkjs      cold  skipped ({why})")
        else:
            reps = max(1, min(a.snarkjs_reps, a.reps))
            ms, err = time_cold([snarkjs, "groth16", "prove", zkey, wtns,
                                 tmp_proof, tmp_public],
                                reps, verify, vkey, tmp_proof, tmp_public)
            if ms:
                emit("snarkjs", "cold", ms)
                print(f"  snarkjs      cold  median {statistics.median(ms):8.1f} ms  "
                      f"({len(ms)} reps, verified by {oracle})")
            else:
                print(f"  snarkjs      cold  FAILED: {err}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if rows:
        parent = os.path.dirname(os.path.abspath(a.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        new = not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
