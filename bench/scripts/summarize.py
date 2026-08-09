#!/usr/bin/env python3
"""Turn the per-machine comparison CSVs into the markdown tables in bench/results/README.md.

One section per machine, always. Two machines with different CPUs, different GPUs and
different core counts do not belong in one table, and a reader who sees them merged will
draw a conclusion that is not in the data.

Medians, with min and max, never means. A single slow rep from a scheduler hiccup moves a
mean and does not move a median, and these runs are short enough that one hiccup is likely.

Reps are not discarded here. In warm mode the first rep of our own backends carries work
that is genuinely lazy (the NTT twiddle tables are built on first use), so rep 1 is
systematically slower than the rest. With ten reps the median is unaffected, which is the
reason the median is what gets reported. The min column is where that first rep shows up
as a gap.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Display order, smallest circuit first.
def variant_key(row):
    try:
        return int(row["constraints"])
    except (KeyError, ValueError):
        return 0


def load():
    by_machine = defaultdict(list)
    for f in sorted(RESULTS.glob("comparison-*.csv")):
        for r in csv.DictReader(f.open()):
            if r.get("ms"):
                by_machine[r.get("machine") or f.stem].append(r)
    return by_machine


def stat(rows):
    t = [float(r["ms"]) for r in rows]
    return statistics.median(t), min(t), max(t), len(t)


def label(prover, backend):
    return "rapidsnark" if prover == "rapidsnark" else f"ours/{backend}"


def section(machine, rows, out):
    meta = rows[0]
    out.append(f"\n## {machine}\n")
    out.append(
        f"`{meta['os']} {meta['arch']}`, {meta['cores']} logical cores, "
        f"accelerator: {meta.get('gpu') or 'none'}. "
        f"Every timing below belongs to a proof that was verified before it was recorded.\n"
    )

    variants = sorted({r["variant"] for r in rows},
                      key=lambda v: variant_key(next(r for r in rows if r["variant"] == v)))
    cols = sorted({label(r["prover"], r["backend"]) for r in rows},
                  key=lambda c: (c != "rapidsnark", c))

    for mode in ("cold", "warm"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        out.append(f"\n### {mode}\n")
        out.append("| circuit | constraints | " + " | ".join(f"{c} (ms)" for c in cols) +
                   " | best vs rapidsnark |")
        out.append("|---|---:|" + "---:|" * (len(cols) + 1))
        for v in variants:
            vr = [r for r in sel if r["variant"] == v]
            if not vr:
                continue
            nc = vr[0]["constraints"]
            cells, meds = [], {}
            for c in cols:
                cr = [r for r in vr if label(r["prover"], r["backend"]) == c]
                if not cr:
                    cells.append("n/a")
                    continue
                med, lo, hi, n = stat(cr)
                meds[c] = med
                cells.append(f"{med:.1f}")
            rs = meds.get("rapidsnark")
            ours = {k: m for k, m in meds.items() if k != "rapidsnark"}
            if rs and ours:
                best = min(ours, key=lambda k: ours[k])
                ratio = f"{rs / ours[best]:.2f}x ({best.split('/')[-1]})"
            else:
                ratio = "n/a"
            out.append(f"| {v} | {nc} | " + " | ".join(cells) + f" | {ratio} |")

    # Stage split, where a backend reported one.
    staged = [r for r in rows if r.get("msm_us")]
    if staged:
        backends = sorted({label(r["prover"], r["backend"]) for r in staged})
        out.append("\n### where the time goes (warm, medians, microseconds)\n")
        out.append("| circuit | backend | gather | ntt | pointwise | msm | assemble | prepare (ms) |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for v in variants:
            for b in backends:
                sr = [r for r in staged
                      if r["variant"] == v and label(r["prover"], r["backend"]) == b]
                if not sr:
                    continue

                def med(field):
                    vals = [float(r[field]) for r in sr if r.get(field)]
                    return statistics.median(vals) if vals else 0.0

                out.append(
                    f"| {v} | {b} | {med('gather_us'):.0f} | {med('ntt_us'):.0f} | "
                    f"{med('pointwise_us'):.0f} | {med('msm_us'):.0f} | "
                    f"{med('assemble_us'):.0f} | {med('prepare_ms'):.1f} |"
                )


def main():
    by_machine = load()
    if not by_machine:
        sys.exit(f"no comparison-*.csv with timings under {RESULTS}")
    out = []
    for machine in sorted(by_machine):
        section(machine, by_machine[machine], out)
    text = "\n".join(out) + "\n"
    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        p = RESULTS / "TABLES.md"
        p.write_text("# Benchmark tables\n\nGenerated by `bench/scripts/summarize.py`. "
                     "Do not edit by hand.\n" + text)
        print(f"wrote {p}")
    else:
        print(text)


if __name__ == "__main__":
    main()
