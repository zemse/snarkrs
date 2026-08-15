#!/usr/bin/env python3
"""Rank the allocating call paths in a dhat JSON heap profile.

The dh_view web viewer is a tree, which is the right shape for browsing and the wrong
shape for a report: to answer "does the MSM's scratch churn matter" you want the paths
sorted by bytes, with the frames that identify them and nothing else.

Two orderings, because they answer different questions:
  * total bytes  - allocation churn. What the allocator and the zeroing cost.
  * bytes at peak - what the process actually has to hold at once (dhat's "t-gmax").

Usage: bench/scripts/dhat-report.py PROFILE.json [--top N]
"""
import json
import re
import sys

# Frames that identify no call site: the allocator shim, Vec's own growth machinery,
# rayon's plumbing. Skipped when picking a path's label so the label lands on the code
# that asked for the memory.
NOISE = re.compile(
    r"dhat::Alloc|GlobalAlloc|raw_vec|alloc::alloc|RawVec|"
    r"__rust_alloc|realloc|core::alloc|alloc::vec::Vec<T,A>::"
)


def label(frames, ftbl, depth=3):
    out = []
    for fi in frames:
        name = ftbl[fi]
        # dhat frames are "0xADDR: symbol (file:line)"; the address is noise in a report.
        name = re.sub(r"^0x[0-9a-f]+: ", "", name)
        name = re.sub(r" \(.*\)$", "", name)
        if name == "[root]" or NOISE.search(name):
            continue
        out.append(name)
        if len(out) >= depth:
            break
    return out or ["<allocator internals only>"]


def mib(b):
    return b / (1024.0 * 1024.0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 15
    for f in sys.argv[1:]:
        if f.startswith("--top"):
            top = int(f.split("=")[1]) if "=" in f else 15
    if not args:
        print(__doc__)
        sys.exit(2)

    d = json.load(open(args[0]))
    ftbl = d["ftbl"]
    pps = d["pps"]
    total_b = sum(p["tb"] for p in pps)
    total_k = sum(p["tbk"] for p in pps)
    peak_b = sum(p.get("gb", 0) for p in pps)

    print(f"command: {d['cmd']}")
    print(
        f"total: {mib(total_b):.2f} MiB in {total_k} blocks   "
        f"at global peak: {mib(peak_b):.2f} MiB"
    )

    for key, title, denom in [
        ("tb", "total bytes allocated (churn)", total_b),
        ("gb", "bytes live at the global peak", peak_b),
    ]:
        print()
        print(f"== top {top} call paths by {title} ==")
        ranked = sorted(pps, key=lambda p: -p.get(key, 0))[:top]
        for p in ranked:
            v = p.get(key, 0)
            if v == 0:
                continue
            blocks = p["tbk"] if key == "tb" else p.get("gbk", 0)
            avg = v / blocks if blocks else 0
            print(
                f"{100 * v / denom:6.2f}%  {mib(v):9.2f} MiB  {blocks:>7} blocks  "
                f"avg {avg / 1024:.1f} KiB"
            )
            for i, frame in enumerate(label(p["fs"], ftbl)):
                print(f"          {'  ' * i}{frame}")


if __name__ == "__main__":
    main()
