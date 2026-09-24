#!/usr/bin/env python3
"""Put our metrics rows next to the published circom row.

The published numbers were taken on an AWS mac2.metal (M1, 8 cores, 16 GB) against the
zkeys-v1 circuits. Ours are whatever machine ran bench/scripts/csp-bench.sh, against
zkeys-v2. Both differences move the numbers, and neither is visible in a bare ratio, so
the table carries the constraint count from each side and a per-million-constraint rate
that is at least defensible across the circuit change. A ratio on a different machine is
still a ratio on a different machine; the only honest fix for that is to run this on a
mac2.metal, which bench/aws exists to do.
"""

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[2]
PUBLISHED = HERE / "bench/csp/published-circom.json"
ORDER = {"poseidon": 0, "sha256": 1, "keccak": 2}


def load_ours(metrics_dir):
    rows = {}
    for path in sorted(pathlib.Path(metrics_dir).glob("*_metrics.json")):
        r = json.loads(path.read_text())
        rows.setdefault(r.get("feat") or "cpu", {})[(r["target"], r["input_size"])] = r
    return rows


def ms(nanos):
    return nanos / 1e6


def fmt(x, unit="", width=9):
    return f"{x:>{width}.1f}{unit}" if x else f"{'-':>{width + len(unit)}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="bench/results/csp/metrics")
    args = ap.parse_args()

    published = json.loads(PUBLISHED.read_text())
    theirs = {(r["target"], r["input_size"]): r for r in published["rows"]}
    ours = load_ours(args.metrics)
    if not ours:
        raise SystemExit(f"no *_metrics.json under {args.metrics}")

    print(f"published circom: {published['_host']}")
    print(f"  {published['_caveat']}\n")

    for backend, rows in sorted(ours.items()):
        print(f"=== g16 / {backend} ===")
        head = (
            f"{'variant':<15}{'constraints':>12}{'ours ms':>10}{'circom ms':>11}"
            f"{'speedup':>9}{'their nc':>11}{'ours ms/Mc':>12}{'circom ms/Mc':>14}"
            f"{'ours MB':>9}{'circom MB':>11}"
        )
        print(head)
        for key in sorted(rows, key=lambda k: (ORDER.get(k[0], 9), k[1])):
            o = rows[key]
            t = theirs.get(key)
            name = f"{key[0]}_{key[1]}"
            o_ms, o_nc = ms(o["proof_duration"]), o["num_constraints"]
            o_rate = o_ms / (o_nc / 1e6)
            o_mb = o["peak_memory"] / 1e6
            if not t:
                print(f"{name:<15}{o_nc:>12}{o_ms:>10.1f}{'-':>11}{'-':>9}{'-':>11}{o_rate:>12.1f}{'-':>14}{o_mb:>9.0f}{'-':>11}")
                continue
            t_ms, t_nc = ms(t["proof_duration"]), t["num_constraints"]
            t_rate = t_ms / (t_nc / 1e6)
            print(
                f"{name:<15}{o_nc:>12}{o_ms:>10.1f}{t_ms:>11.1f}{t_ms / o_ms:>8.2f}x"
                f"{t_nc:>11}{o_rate:>12.1f}{t_rate:>14.1f}"
                f"{o_mb:>9.0f}{t['peak_memory'] / 1e6:>11.0f}"
            )
        print()

    if len(ours) > 1:
        print("=== our backends against each other (proving time, ms) ===")
        names = sorted(ours)
        keys = sorted(
            set().union(*(set(v) for v in ours.values())),
            key=lambda k: (ORDER.get(k[0], 9), k[1]),
        )
        print(f"{'variant':<15}" + "".join(f"{n:>10}" for n in names))
        for key in keys:
            cells = "".join(
                f"{ms(ours[n][key]['proof_duration']):>10.1f}" if key in ours[n] else f"{'-':>10}"
                for n in names
            )
            print(f"{key[0] + '_' + str(key[1]):<15}{cells}")


if __name__ == "__main__":
    main()
