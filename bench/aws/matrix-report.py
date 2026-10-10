#!/usr/bin/env python3
"""Markdown tables from matrix.py summaries: matrix-report.py SUMMARY_DIR > report.md"""

import json
import sys
from pathlib import Path

summaries = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*.json"))]
summaries.sort(key=lambda s: (s["logical_cpus"], s["instance_type"]))
circuits = list(summaries[0]["circuits"]) if summaries else []


def isa(s):
    i = s["isa"]
    if s.get("arch") == "arm64":
        return "SVE2" if i.get("sve2") else "SVE" if i.get("sve") else "NEON only"
    if i["avx512ifma"]:
        return "AVX-512 IFMA"
    if i["avx512f"]:
        return "AVX-512, no IFMA"
    return "AVX2 only"


print("## Machines\n")
print("| instance | CPU | vCPU | threads/core | sockets | NUMA | L3 | ISA | build | accepted |")
print("|---|---|---:|---:|---:|---:|---|---|---|---|")
for s in summaries:
    print(f"| {s['instance_type']} | {s['cpu']} | {s['logical_cpus']} | {s['threads_per_core']} | "
          f"{s['sockets']} | {s['numa_nodes']} | {s['l3']} | {isa(s)} | {s['build']} | "
          f"{'yes' if s['accepted'] else 'NO'} |")

for title, key in [("Speedup over rapidsnark (rapidsnark median / ours median)", "speedup"),
                   ("Ours, warm median ms", "ours"), ("rapidsnark, warm median ms", "rs")]:
    print(f"\n## {title}\n")
    print("| instance | " + " | ".join(circuits) + " |")
    print("|---|" + "---:|" * len(circuits))
    for s in summaries:
        cells = []
        for c in circuits:
            r = s["circuits"][c]
            v = {"speedup": r.get("speedup_vs_rapidsnark"),
                 "ours": (r.get("ours_ms") or {}).get("median"),
                 "rs": (r.get("rapidsnark_ms") or {}).get("median")}[key]
            cells.append("-" if v is None else (f"{v:.2f}x" if key == "speedup" else f"{v:.1f}"))
        print(f"| {s['instance_type']} | " + " | ".join(cells) + " |")

print("\n## Our stage split, anon-aadhaar (median ms)\n")
print("| instance | gather | ntt | msm | prepare | peak RSS ours / rapidsnark MiB |")
print("|---|---:|---:|---:|---:|---|")
for s in summaries:
    r = s["circuits"].get("anon-aadhaar", {})
    st = r.get("ours_stages_ms", {})
    print(f"| {s['instance_type']} | {st.get('gather')} | {st.get('ntt')} | {st.get('msm')} | "
          f"{r.get('ours_prepare_ms')} | {r.get('ours_peak_rss_mib')} / {r.get('rs_peak_rss_mib')} |")
