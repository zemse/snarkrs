#!/usr/bin/env python3
"""Append-only performance history, and the PERFORMANCE.md rendered from it.

One row per (machine, backend, mode, circuit) per measurement run. Optimisation
commits quote the delta between their row and the row before it.

  perf_history.py ingest <csv> --machine M --label L [--commit SHA] [--date D]
                              [--gpu G] [--note "..."]
  perf_history.py seed        # rebuild the baseline from bench/results/sweep + comparison CSVs
  perf_history.py render      # regenerate PERFORMANCE.md
"""
import argparse, csv, json, os, statistics, subprocess, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "bench", "results")
HISTORY = os.path.join(RESULTS, "history.csv")
PERF_MD = os.path.join(ROOT, "PERFORMANCE.md")

FIELDS = ["date", "commit", "label", "machine", "gpu", "arch", "cores", "backend",
          "mode", "variant", "constraints", "reps", "ms_median", "ms_min",
          "prepare_ms", "gather_us", "ntt_us", "pointwise_us", "msm_us",
          "assemble_us", "note"]

STAGES = ["gather_us", "ntt_us", "pointwise_us", "msm_us", "assemble_us"]


def med(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return round(statistics.median(vals), 3) if vals else ""


def aggregate(rows):
    """rows: list of dicts from a bench CSV. -> keyed medians, ours-only."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("prover", "ours") != "ours":
            continue
        if r.get("verified", "yes") not in ("yes", "na", ""):
            continue
        key = (r["backend"], r["mode"], r["variant"], r["constraints"])
        groups[key].append(r)
    out = []
    for (backend, mode, variant, constraints), g in sorted(groups.items()):
        ms = [float(r["ms"]) for r in g]
        rec = {
            "backend": backend, "mode": mode, "variant": variant,
            "constraints": int(constraints), "reps": len(g),
            "ms_median": round(statistics.median(ms), 3),
            "ms_min": round(min(ms), 3),
            "arch": g[0].get("arch", ""), "cores": g[0].get("cores", ""),
            "csv_gpu": g[0].get("gpu", ""),
            "prepare_ms": med([r.get("prepare_ms", "") for r in g]),
        }
        for s in STAGES:
            rec[s] = med([r.get(s, "") for r in g])
        out.append(rec)
    return out


def read_history():
    if not os.path.exists(HISTORY):
        return []
    with open(HISTORY) as f:
        return list(csv.DictReader(f))


def write_history(rows):
    with open(HISTORY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return ""


def today():
    return subprocess.check_output(["date", "+%Y-%m-%d"], text=True).strip()


def ingest(path, machine, label, commit, date, gpu, note, existing=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    recs = aggregate(rows)
    hist = existing if existing is not None else read_history()
    for rec in recs:
        rec.update({"date": date, "commit": commit, "label": label,
                    "machine": machine, "gpu": rec.pop("csv_gpu", "") or gpu,
                    "note": note})
        hist.append(rec)
    return hist, len(recs)


def cmd_seed(args):
    date = args.date or today()
    hist = []
    n = 0
    sweep = os.path.join(RESULTS, "sweep")
    if os.path.isdir(sweep):
        for name in sorted(os.listdir(sweep)):
            d = os.path.join(sweep, name)
            if not os.path.isdir(d):
                continue
            metap = os.path.join(d, "meta.json")
            meta = json.load(open(metap)) if os.path.exists(metap) else {}
            gpu = meta.get("gpu") or "none"
            commit = meta.get("git", "")
            for backend_csv in sorted(os.listdir(d)):
                if not backend_csv.endswith(".csv"):
                    continue
                hist, k = ingest(os.path.join(d, backend_csv), name, "baseline",
                                 commit, date, gpu, "round-2 sweep", existing=hist)
                n += k
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("comparison-") and f.endswith(".csv"):
            machine = f[len("comparison-"):-len(".csv")]
            hist, k = ingest(os.path.join(RESULTS, f), machine, "baseline",
                             "", date, "", "round-1 comparison", existing=hist)
            n += k
    write_history(hist)
    print(f"seeded {n} rows into {HISTORY}")


def cmd_ingest(args):
    hist, k = ingest(args.csv, args.machine, args.label, args.commit or git_sha(),
                     args.date or today(), args.gpu or "", args.note or "")
    write_history(hist)
    print(f"ingested {k} rows for {args.machine} / {args.label}")


def fmt(v, nd=1):
    if v in ("", None):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return str(v)


def delta(new, old):
    try:
        new, old = float(new), float(old)
    except (TypeError, ValueError):
        return "-"
    if old == 0:
        return "-"
    pct = (new - old) / old * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def cmd_render(args):
    hist = read_history()
    if not hist:
        print("no history; run `perf_history.py seed` first", file=sys.stderr)
        return 1
    # key -> ordered list of measurements
    series = defaultdict(list)
    for r in hist:
        series[(r["machine"], r["backend"], r["mode"], r["variant"])].append(r)

    machines = sorted({r["machine"] for r in hist})
    labels = []
    for r in hist:
        if r["label"] not in labels:
            labels.append(r["label"])

    L = []
    L.append("# Performance")
    L.append("")
    L.append("Proving latency per machine, per backend, per circuit. Generated from")
    L.append("`bench/results/history.csv` by `bench/scripts/perf_history.py render`.")
    L.append("Do not edit by hand; add a measurement with `perf_history.py ingest`.")
    L.append("")
    L.append("`baseline` is the state of the tree before the optimisation round that")
    L.append("began on 2026-08-13. `current` is the newest measurement for that")
    L.append("configuration. A negative delta is a speedup.")
    L.append("")
    L.append("Every timing is the median over the run's reps, and every proof behind a")
    L.append("timing was verified before it was recorded.")
    L.append("")
    L.append("## Circuits")
    L.append("")
    L.append("| variant | constraints |")
    L.append("|---|---:|")
    seen = {}
    for r in hist:
        seen[r["variant"]] = int(r["constraints"])
    for v, c in sorted(seen.items(), key=lambda kv: kv[1]):
        L.append(f"| {v} | {c:,} |")
    L.append("")

    for m in machines:
        rows = [r for r in hist if r["machine"] == m]
        gpu = next((r["gpu"] for r in rows if r["gpu"] and r["gpu"] != "none"), "none")
        arch = next((r["arch"] for r in rows if r["arch"]), "")
        cores = next((r["cores"] for r in rows if r["cores"]), "")
        L.append(f"## {m}")
        L.append("")
        L.append(f"`{arch}`, {cores} logical cores, accelerator: {gpu}")
        L.append("")
        for mode in ("warm", "cold"):
            mrows = [r for r in rows if r["mode"] == mode]
            if not mrows:
                continue
            L.append(f"### {mode}")
            L.append("")
            L.append("| backend | circuit | baseline ms | current ms | delta | baseline msm us | current msm us | msm delta |")
            L.append("|---|---|---:|---:|---:|---:|---:|---:|")
            keys = sorted({(r["backend"], int(r["constraints"]), r["variant"]) for r in mrows},
                          key=lambda k: (k[0], k[1]))
            for backend, _c, variant in keys:
                s = series[(m, backend, mode, variant)]
                base = next((r for r in s if r["label"] == "baseline"), s[0])
                cur = s[-1]
                L.append(
                    f"| {backend} | {variant} | {fmt(base['ms_median'])} | {fmt(cur['ms_median'])} | "
                    f"{delta(cur['ms_median'], base['ms_median']) if cur is not base else '-'} | "
                    f"{fmt(base['msm_us'], 0)} | {fmt(cur['msm_us'], 0)} | "
                    f"{delta(cur['msm_us'], base['msm_us']) if cur is not base else '-'} |")
            L.append("")

    L.append("## Measurement log")
    L.append("")
    L.append("| label | machines | rows |")
    L.append("|---|---|---:|")
    for lab in labels:
        rs = [r for r in hist if r["label"] == lab]
        ms = sorted({r["machine"] for r in rs})
        shown = ", ".join(ms[:4]) + (f" +{len(ms)-4} more" if len(ms) > 4 else "")
        L.append(f"| {lab} | {shown} | {len(rs)} |")
    L.append("")

    with open(PERF_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {PERF_MD} ({len(hist)} history rows, {len(machines)} machines)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--date"); s.set_defaults(fn=cmd_seed)
    i = sub.add_parser("ingest")
    i.add_argument("csv")
    i.add_argument("--machine", required=True)
    i.add_argument("--label", required=True)
    i.add_argument("--commit"); i.add_argument("--date")
    i.add_argument("--gpu"); i.add_argument("--note")
    i.set_defaults(fn=cmd_ingest)
    r = sub.add_parser("render"); r.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
