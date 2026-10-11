#!/usr/bin/env python3
"""Tables and correctness checks for one bench/scripts/arkab.py output directory.

    arkab_report.py OUT_DIR [--title T] [--snarkjs PATH --artifacts DIR --ptau-dir DIR] [--json F]

Prove: median over every warm rep of every round, plus per-round medians and the stage split.
Verify: median of the per-round medians (each over --verify-reps calls). Ceremony steps: median
of the per-round wall clocks. Change is perf against the first variant.

Checks: every row came from a run that exited 0 (arkab.py records nothing else), the failure
list is empty, every deterministic output hashed the same under both variants in every round,
and with --snarkjs, snarkjs 0.7.6 accepts the kept proofs, the round-1 phase-1 files and the
kept phase-2 keys.
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

STEPS = [("prove", "prove (warm)"), ("verify", "verify (in process)"), ("contribute", "ptau contribute"),
         ("pverify", "ptau verify"), ("prepare", "ptau prepare phase2"), ("setup", "groth16 setup"),
         ("zcontribute", "zkey contribute"), ("zverify", "zkey verify init")]
STAGES = ["gather", "ntt", "pointwise", "msm", "assemble"]


def med(xs):
    return statistics.median(xs) if xs else None


def pct(a, b):
    return None if not a or not b else 100 * (b / a - 1)


def fmt(ms):
    if ms is None:
        return "-"
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}" if ms < 10000 else f"{ms:.0f}"


def fpct(p):
    return "-" if p is None else f"{p:+.1f}%"


def snarkjs(bin_, *args):
    r = subprocess.run([bin_, *map(str, args)], capture_output=True, text=True)
    text = r.stdout + r.stderr
    # "OK!" from groth16 verify, "ZKey Ok!" and "Powers of Tau Ok!" from the other two.
    return r.returncode == 0 and ("OK!" in text or "Ok!" in text), text[-300:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--title", default=None)
    p.add_argument("--snarkjs")
    p.add_argument("--artifacts")
    p.add_argument("--ptau-dir")
    p.add_argument("--json")
    p.add_argument("--more", nargs="*", default=[],
                   help="further arkab.py output dirs of the same builds, e.g. re-runs; their rounds follow")
    p.add_argument("--busy", type=float, default=3.0,
                   help="a row whose others_cores reached this ran beside another job (marked *)")
    p.add_argument("--drop-busy", action="store_true",
                   help="drop both variants' rows of any (step, item, round) with a busy row")
    a = p.parse_args()
    out = Path(a.out)
    rows, hashes, failures = [], [], []
    for i, d in enumerate([out, *map(Path, a.more)]):
        for r in json.loads((d / "rows.json").read_text()):
            rows.append({**r, "round": r["round"] + 100 * i})
        hashes += [{**h, "round": h["round"] + 100 * i} for h in json.loads((d / "hashes.json").read_text())]
        failures += json.loads((d / "failures.json").read_text())
    busy = {(r["step"], r["item"], r["round"]) for r in rows if r.get("others_cores", 0) >= a.busy}
    dropped = 0
    if a.drop_busy:
        dropped = len(busy)
        rows = [r for r in rows if (r["step"], r["item"], r["round"]) not in busy]
        busy = set()
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    base, new = variants[0], variants[1]
    rounds = sorted({r["round"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["step"], r["item"], r["variant"], r["round"])].append(r)
    summary = {"variants": variants, "rounds": rounds, "steps": {}, "failures": len(failures),
               "busy_cells_dropped": dropped}
    lines = [f"## {a.title or out.name}\n"]

    for step, label in STEPS:
        items = list(dict.fromkeys(r["item"] for r in rows if r["step"] == step))
        if not items:
            continue
        lines.append(f"### {label}, ms\n")
        others = variants[1:]
        lines.append(f"| item | {' | '.join(variants)} | " + " | ".join(f"{v} vs {base}" for v in others)
                     + f" | per round ({' / '.join(variants)}) |")
        lines.append("|---|" + "---:|" * (len(variants) + len(others)) + "---|")
        summary["steps"][step] = {}
        for it in items:
            cell, per_round = {}, []
            for v in variants:
                rms = [med([x["ms"] for x in by[(step, it, v, r)]]) for r in rounds]
                if step == "prove":
                    cell[v] = med([x["ms"] for r in rounds for x in by[(step, it, v, r)]])
                else:
                    cell[v] = med([x for x in rms if x is not None])
                per_round.append(rms)
            change = pct(cell[base], cell[new])
            changes = {v: pct(cell[base], cell[v]) for v in others}
            rr = " ; ".join(" / ".join(fmt(x) for x in xs) + ("*" if (step, it, r) in busy else "")
                            for *xs, r in zip(*per_round, rounds) if any(x is not None for x in xs))
            lines.append(f"| {it} | " + " | ".join(fmt(cell[v]) for v in variants) + " | "
                         + " | ".join(fpct(changes[v]) for v in others) + f" | {rr} |")
            s = {v: cell[v] for v in variants}
            s["change_pct"] = change
            s["change_pct_vs_" + base] = changes
            s["n"] = {v: sum(len(by[(step, it, v, r)]) for r in rounds) for v in variants}
            if step == "prove":
                s["stages"] = {v: {k: med([x[k] for r in rounds for x in by[(step, it, v, r)]]) for k in STAGES}
                               for v in variants}
            summary["steps"][step][it] = s
        lines.append("")

    if "prove" in summary["steps"]:
        lines.append("### prove stage medians, ms\n")
        lines.append("| circuit | variant | " + " | ".join(STAGES) + " |")
        lines.append("|---|---|" + "---:|" * len(STAGES))
        for it, s in summary["steps"]["prove"].items():
            for v in variants:
                st = s["stages"][v]
                lines.append(f"| {it} | {v} | " + " | ".join(fmt(st[k]) for k in STAGES) + " |")
        lines.append("")

    # Deterministic outputs: one hash per (what, round, variant); all of a `what` must agree.
    groups = defaultdict(set)
    count = defaultdict(int)
    for h in hashes:
        groups[h["what"]].add(h["sha256"])
        count[h["what"]] += 1
    det = {w: {"identical": len(s) == 1, "files_hashed": count[w]} for w, s in groups.items()}
    summary["deterministic"] = det
    lines.append(f"Deterministic outputs: {sum(d['identical'] for d in det.values())} of {len(det)} "
                 f"byte-identical across variants and rounds ({sum(count.values())} files hashed)"
                 + ("" if all(d["identical"] for d in det.values()) else
                    ": DIFFER " + ", ".join(w for w, d in det.items() if not d["identical"])) + ".\n")
    if busy:
        lines.append(f"`*`: some row of that round ran while other processes used {a.busy:g} or more cores.\n")
    if dropped:
        lines.append(f"{dropped} (step, item, round) cells that ran beside another job were dropped, "
                     "both variants.\n")
    prove_reps = sum(1 for r in rows if r["step"] == "prove")
    summary["self_verified_prove_reps"] = prove_reps
    lines.append(f"Self-verified warm proofs: {prove_reps}. Failures recorded by the runner: {len(failures)}.\n")

    if a.snarkjs:
        checks = {}
        keep = out / "keep"
        for v in variants:
            for d in sorted((keep / v / "proofs").glob("*")):
                checks[f"{v} proof {d.name}"] = snarkjs(a.snarkjs, "groth16", "verify", d / "vkey.json",
                                                        d / "public.json", d / "proof.json")
            for f in sorted((keep / v).glob("*.ptau")):
                checks[f"{v} ptau {f.name}"] = snarkjs(a.snarkjs, "powersoftau", "verify", f)
            for z in sorted((keep / v / "zkeys").glob("*.zkey")):
                art = Path(a.artifacts) / z.stem
                r1cs = next(iter(sorted(art.glob("*.r1cs"))))
                ptau = json.loads((out / "ptau-for.json").read_text())[z.stem]
                checks[f"{v} zkey {z.stem}"] = snarkjs(a.snarkjs, "zkey", "verify", r1cs,
                                                       Path(a.ptau_dir) / ptau, z)
        ok = sum(c[0] for c in checks.values())
        summary["snarkjs"] = {k: c[0] for k, c in checks.items()}
        ver = subprocess.run([a.snarkjs], capture_output=True, text=True).stdout.split()[0]
        lines.append(f"{ver}: {ok} of {len(checks)} accepted"
                     + ("" if ok == len(checks) else ": REJECTED " + ", ".join(k for k, c in checks.items() if not c[0]))
                     + ".\n")
    print("\n".join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
