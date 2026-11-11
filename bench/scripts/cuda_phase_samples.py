#!/usr/bin/env python3
"""Validate raw c13 MSM phase samples; these are not end-to-end proof timings."""

import argparse
import itertools
import json
from pathlib import Path
import re
import statistics
import sys

PHASES = (
    "count", "scan", "scatter", "clear", "accum", "merge", "reduce", "ones",
    "combine", "batch_device", "batch_wall",
)
SETS = ("mixed", "uniform")
GROUPS = ("g1", "g2")
SIZES = (65536, 262144)
SAMPLE = re.compile(
    r"phase_samples ([a-z0-9_]+) ([a-z0-9_]+) n=(\d+) c=(\d+) "
    r"phase=([a-z0-9_]+) ms=(\[.*\])"
)


def parse_samples(text, repetitions):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    completions = re.findall(r"^test result: (.*)$", text, re.M)
    if len(completions) != 1 or not completions[0].startswith("ok. 1 passed; 0 failed;"):
        raise ValueError("missing successful single-test completion")
    expected = set(itertools.product(SETS, GROUPS, SIZES, PHASES))
    found = {}
    for line in text.splitlines():
        if not line.startswith("phase_samples "):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError("malformed phase sample row")
        scalar_set, group, size, window, phase, raw = match.groups()
        key = (scalar_set, group, int(size), phase)
        if int(window) != 13 or key not in expected:
            raise ValueError(f"unexpected phase configuration: {key}, c={window}")
        if key in found:
            raise ValueError(f"duplicate phase samples: {key}")
        values = json.loads(raw)
        if len(values) != repetitions or any(
            type(value) not in (int, float) or not 0 <= value <= sys.float_info.max
            for value in values
        ):
            raise ValueError(f"invalid sample count or value: {key}")
        ordered = sorted(values)
        middle = statistics.median(ordered)
        if not 0 <= middle <= sys.float_info.max:
            raise ValueError(f"nonfinite median: {key}")
        found[key] = {
            "set": scalar_set, "group": group, "n": int(size), "c": 13,
            "phase": phase,
            "clock": "wall" if phase in ("combine", "batch_wall") else "device",
            "samples_ms": values, "median_ms": middle,
            "min_ms": ordered[0], "max_ms": ordered[-1],
        }
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} phase rows")
    return {
        "schema": 1, "repetitions": repetitions, "median_method": "standard",
        "scope": "resident-input MSM, not a full proof",
        "rows": [found[key] for key in sorted(found)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()
    try:
        result = parse_samples(args.log.read_text(), args.reps)
    except (ValueError, OSError) as error:
        parser.exit(1, f"phase samples: {error}\n")
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
