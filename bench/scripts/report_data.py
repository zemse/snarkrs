#!/usr/bin/env python3
"""Collapse the sweep into one JSON blob for the HTML report.

Kept separate from cost.py so the numbers in the report and the numbers in COST.md come
from one implementation of the cost model rather than two that can drift apart.
"""
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "sweep"


def load():
    od, spot = {}, {}
    for r in csv.DictReader((ROOT / "aws" / "machines.csv").open()):
        od[r["instance_type"]] = float(r["usd_per_hour"])
    p = ROOT / "aws" / "spot-use1.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            t, v = r["instance_type"], float(r["spot_usd_per_hour"])
            if t not in spot or v < spot[t]:
                spot[t] = v

    machines, cells = {}, []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        mf = d / "meta.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text())
        m = meta["instance_type"]
        price = od.get(m) or meta.get("usd_per_hour")
        meta["usd_per_hour"] = price
        meta["spot_usd_per_hour"] = spot.get(m)
        machines[m] = meta
        for f in d.glob("*.csv"):
            by = defaultdict(list)
            for r in csv.DictReader(f.open()):
                if not r.get("ms") or r.get("verified") not in ("yes", "true", "True", "1"):
                    continue
                by[(r["backend"], r["mode"], r["variant"], int(r["constraints"]))].append(
                    (float(r["ms"]), float(r.get("prepare_ms") or 0)))
            for (b, mode, v, nc), vals in by.items():
                ms = statistics.median(x[0] for x in vals)
                cells.append(dict(
                    machine=m, backend=b, mode=mode, variant=v, constraints=nc,
                    ms=round(ms, 2), lo=round(min(x[0] for x in vals), 2),
                    hi=round(max(x[0] for x in vals), 2), n=len(vals),
                    prepare_ms=round(statistics.median(x[1] for x in vals), 2),
                    usd_per_k=round(price * ms / 1000 / 3600 * 1000, 6),
                    usd_per_k_spot=round((spot.get(m) or price) * ms / 1000 / 3600 * 1000, 6),
                ))
    return dict(machines=machines, cells=cells)


if __name__ == "__main__":
    d = load()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "report-data.json"
    out.write_text(json.dumps(d, indent=1))
    print(f"{len(d['machines'])} machines, {len(d['cells'])} cells -> {out}")
