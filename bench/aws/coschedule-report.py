#!/usr/bin/env python3
"""Tables from coschedule-guest.sh's out/ directory: per-stage splits, host CPU use, and
throughput of concurrent proof streams.

Every stream is a `snarkrs bench --mode warm` under G16_BENCH_CPU=1, which prints one
`cpu-usage:` line a proof: wall-clock start and end, prove time, and the CPU time the whole
process spent across it. Two rates are reported for a stream:

* prove-only, 1000 / median prove ms: what a service that does nothing between proofs
  would get at that contention level. The headline.
* loop, proofs that ran inside the overlap window of all the line's streams, divided by
  the window. The bench verifies every proof on the CPU between reps, so this is lower,
  and `duty` (time inside prove() over the window) says by how much.

usage: coschedule-report.py <out-dir> [--nproc N]
"""
import csv
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r"cpu-usage: variant=(\S+) rep=(\d+) start_ms=([\d.]+) end_ms=([\d.]+) "
                  r"prove_ms=([\d.]+) cpu_ms=([\d.]+)")
STAGES = ["gather_us", "ntt_us", "pointwise_us", "msm_us", "assemble_us"]


def stream(dir_, i, spec):
    backend, threads, _reps, *pin = spec.split(":")
    proofs = []
    for m in LINE.finditer((dir_ / f"{i}-{backend}.err").read_text()):
        proofs.append(dict(start=float(m[3]), end=float(m[4]), prove=float(m[5]),
                           cpu=float(m[6])))
    rows = list(csv.DictReader((dir_ / f"{i}-{backend}.csv").open())) \
        if (dir_ / f"{i}-{backend}.csv").exists() else []
    return dict(backend=backend, threads=int(threads), pin=pin[0] if pin else "",
                proofs=proofs, rows=rows)


def load(out):
    tags = {}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        specs = [l.split(maxsplit=1) for l in (d / "streams.txt").read_text().splitlines()]
        tags[d.name] = [stream(d, int(i), s.strip()) for i, s in specs]
    return tags


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def split_table(tags, nproc):
    print("| circuit | work | backend | prove ms | gather | ntt | pointwise | msm | assemble "
          "| host CPU ms / proof | cores busy | host idle |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, streams in tags.items():
        if not tag.startswith("split-"):
            continue
        _, backend, rest = tag.split("-", 2)
        circuit, work = rest.rsplit("-", 1)
        s = streams[0]
        stage = [med([int(r[k]) for r in s["rows"]]) / 1000 for k in STAGES]
        prove = med([p["prove"] for p in s["proofs"]])
        cpu = med([p["cpu"] for p in s["proofs"]])
        busy = cpu / prove
        print(f"| `{circuit}` | {work} | {backend} | {prove:.1f} | "
              + " | ".join(f"{x:.1f}" for x in stage)
              + f" | {cpu:.1f} | {busy:.2f} | {100 * (1 - busy / nproc):.0f}% |")


def cosched_table(tags):
    print("| experiment | circuit | work | stream | prove ms | prove-only /s | loop /s | duty "
          "| total prove-only /s | total loop /s |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, streams in tags.items():
        if tag.startswith(("split-", "warmup")):
            continue
        name, circuit, work = tag.split("~")
        lo = max(s["proofs"][0]["start"] for s in streams if s["proofs"])
        hi = min(s["proofs"][-1]["end"] for s in streams if s["proofs"])
        window = max(hi - lo, 1e-9)
        tot_p = tot_l = 0.0
        cells = []
        for s in streams:
            prove = med([p["prove"] for p in s["proofs"]])
            inside = [p for p in s["proofs"] if p["start"] >= lo and p["end"] <= hi]
            # Proofs straddling an edge count by the share of them inside the window.
            frac = sum(max(0.0, min(p["end"], hi) - max(p["start"], lo)) / (p["end"] - p["start"])
                       for p in s["proofs"] if p["end"] > p["start"])
            duty = sum(max(0.0, min(p["end"], hi) - max(p["start"], lo))
                       for p in s["proofs"]) / window
            rate_p = 1000 / prove
            rate_l = 1000 * frac / window
            tot_p += rate_p
            tot_l += rate_l
            pin = f" on {s['pin']}" if s["pin"] else ""
            cells.append((f"{s['backend']} x{s['threads']}{pin}", prove, rate_p, rate_l, duty,
                           len(inside)))
        for j, (label, prove, rp, rl, duty, _n) in enumerate(cells):
            tail = f" {tot_p:.2f} | {tot_l:.2f} |" if j == 0 else " | |"
            head = f"| {name} | `{circuit}` | {work} |" if j == 0 else "| | | |"
            print(f"{head} {label} | {prove:.1f} | {rp:.2f} | {rl:.2f} | {duty:.2f} |{tail}")


def stage_med(s, key):
    return med([int(r[key]) for r in s["rows"]]) / 1000


def opt20_table(tags, nproc):
    """OPT-20's best case: the CPU takes a share of each MSM in proportion to its rate, so
    the MSM stage drops from T_g to T_g T_c / (T_g + T_c), a saving of T_g^2 / (T_g + T_c).
    T_c is the CPU's MSM stage alone on every core, and again as measured beside a GPU stream
    on the remaining cores (the g1c<n-1> line), which already pays the contention."""
    print("| circuit | work | GPU prove ms | GPU MSM ms | CPU MSM ms, alone | ceiling ms | "
          "ceiling % of proof | CPU MSM ms, beside GPU | ceiling ms | ceiling % of proof |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, streams in tags.items():
        if not tag.startswith("split-cuda-"):
            continue
        rest = tag[len("split-cuda-"):]
        circuit, work = rest.rsplit("-", 1)
        g = streams[0]
        prove = med([p["prove"] for p in g["proofs"]])
        tg = stage_med(g, "msm_us")
        tc = stage_med(tags[f"split-cpu-{circuit}-{work}"][0], "msm_us")
        cells = [f"{prove:.1f}", f"{tg:.1f}", f"{tc:.1f}"]
        save = tg * tg / (tg + tc)
        cells += [f"{save:.1f}", f"{100 * save / prove:.1f}%"]
        beside = tags.get(f"g1c{nproc - 1}~{circuit}~{work}")
        cpu = [s for s in beside or [] if s["backend"] == "cpu" and s["rows"]]
        if cpu:
            tc3 = stage_med(cpu[0], "msm_us")
            save3 = tg * tg / (tg + tc3)
            cells += [f"{tc3:.1f}", f"{save3:.1f}", f"{100 * save3 / prove:.1f}%"]
        else:
            cells += ["", "", ""]
        print(f"| `{circuit}` | {work} | " + " | ".join(cells) + " |")


def main():
    out = Path(sys.argv[1])
    nproc = int(sys.argv[sys.argv.index("--nproc") + 1]) if "--nproc" in sys.argv else 4
    tags = load(out)
    print(f"## Single streams (nproc {nproc})\n")
    split_table(tags, nproc)
    print("\n## Concurrent streams\n")
    cosched_table(tags)
    print("\n## OPT-20 ceiling\n")
    opt20_table(tags, nproc)


if __name__ == "__main__":
    main()
