#!/usr/bin/env python3
"""Tables from sync-ab-guest.sh's out/ directory: warm CUDA prove time and host CPU time per
G16_CUDA_SYNC mode, against `auto` (the variable unset).

A run's prove time is the median of its proofs; a cell is the median of its runs over the
rounds. The change against auto is per round (same round, same circuit), then the median of
those, with the range over rounds. cores busy = median over all proofs of cpu_ms / prove_ms.

usage: sync-ab-report.py <out-dir>
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"cpu-usage: variant=\S+ rep=(\d+) start_ms=[\d.]+ end_ms=[\d.]+ "
                  r"prove_ms=([\d.]+) cpu_ms=([\d.]+)")
NAME = re.compile(r"(.+)-([a-z]+)-r(\d+)\.err$")
ORDER = ["railgun-01x01", "tornado", "sha256", "railgun-13x01", "rsa2048", "keccak256",
         "anon-aadhaar", "tiny_mul"]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def load(d):
    # runs[circuit][mode][round] = list of (prove, cpu)
    runs = defaultdict(lambda: defaultdict(dict))
    for f in d.glob("*.err"):
        m = NAME.match(f.name)
        if not m:
            continue
        proofs = [(float(p), float(c)) for _r, p, c in LINE.findall(f.read_text())]
        if proofs:
            runs[m[1]][m[2]][int(m[3])] = proofs
    return runs


def table(work, runs):
    modes = sorted({m for c in runs.values() for m in c}, key=lambda m: (m != "auto", m))
    others = [m for m in modes if m != "auto"]
    head = ["circuit"] + [f"{m} ms" for m in modes] + [f"{m} vs auto" for m in others] \
        + [f"{m} cores" for m in modes]
    print(f"### {work}\n")
    print("| " + " | ".join(head) + " |")
    print("|---" + "|---:" * (len(head) - 1) + "|")
    for circuit in sorted(runs, key=lambda c: ORDER.index(c) if c in ORDER else 99):
        by = runs[circuit]
        cells = [f"`{circuit}`"]
        rmed = {m: {r: med([p for p, _ in v]) for r, v in by.get(m, {}).items()} for m in modes}
        for m in modes:
            cells.append(f"{med(list(rmed[m].values())):.1f}")
        for m in others:
            deltas = [100 * (rmed[m][r] / rmed["auto"][r] - 1)
                      for r in rmed[m] if r in rmed.get("auto", {})]
            cells.append(f"{med(deltas):+.1f}% ({min(deltas):+.1f} to {max(deltas):+.1f})"
                         if deltas else "")
        for m in modes:
            ratios = [c / p for v in by.get(m, {}).values() for p, c in v if p > 0]
            cells.append(f"{med(ratios):.2f}")
        print("| " + " | ".join(cells) + " |")
    print()


def main():
    out = Path(sys.argv[1])
    for d in sorted(out.glob("sync-*")):
        table(d.name[len("sync-"):], load(d))


if __name__ == "__main__":
    main()
