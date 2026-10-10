#!/usr/bin/env python3
"""Markdown tables from `matrix.py run --guest ab` summaries: matrix-ab-report.py SUMMARY_DIR > report.md"""

import json
import sys
from pathlib import Path

summaries = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*.json"))]
summaries.sort(key=lambda s: (s["logical_cpus"], s["instance_type"]))


def isa(s):
    i = s["isa"]
    if i["avx512ifma"]:
        return "AVX-512 IFMA"
    if i["avx512f"]:
        return "AVX-512, no IFMA"
    return "AVX2 only"


def pct(v):
    return "-" if v is None else f"{v:+.1f}%"


print("## Machines\n")
print("| instance | CPU | vCPU | ISA | builds | rounds x reps | accepted |")
print("|---|---|---:|---|---|---|---|")
for s in summaries:
    print(f"| {s['instance_type']} | {s['cpu']} | {s['logical_cpus']} | {isa(s)} | {s['build']} | "
          f"{s['rounds']} x {s['reps']} | {'yes' if s['accepted'] else 'NO'} |")

for s in summaries:
    builds = s["build"].split()
    others = [b for b in builds if b != "generic"]
    print(f"\n## {s['instance_type']}, warm median ms over all rounds\n")
    print("| circuit | " + " | ".join(builds) + " | " + " | ".join(f"{b} vs generic" for b in others)
          + " | rapidsnark |")
    print("|---|" + "---:|" * (len(builds) + len(others) + 1))
    for c, r in s["circuits"].items():
        cells = [f"{r[b]['ms']['median']:.1f}" for b in builds]
        cells += [pct(r[b].get("change_vs_generic_pct")) for b in others]
        cells.append(f"{(r.get('rapidsnark_ms') or {}).get('median', float('nan')):.1f}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    print(f"\nPer-round medians, ms ({' / '.join(builds)}; round order alternates):\n")
    print("| circuit | " + " | ".join(f"round {i + 1}" for i in range(s["rounds"])) + " |")
    print("|---|" + "---|" * s["rounds"])
    for c, r in s["circuits"].items():
        rounds = zip(*(r[b]["round_medians_ms"] for b in builds))
        print(f"| {c} | " + " | ".join(" / ".join(f"{x:.1f}" for x in rnd) for rnd in rounds) + " |")
    a = s["circuits"].get("anon-aadhaar")
    if a:
        print("\nanon-aadhaar stage medians, ms:\n")
        print("| build | gather | ntt | pointwise | msm | assemble | prepare |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for b in builds:
            st = a[b]["stages_ms"]
            print(f"| {b} | {st['gather']:.1f} | {st['ntt']:.1f} | {st['pointwise']:.1f} | {st['msm']:.1f} | "
                  f"{st['assemble']:.1f} | {a[b]['prepare_ms']:.1f} |")
