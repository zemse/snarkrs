#!/usr/bin/env python3
"""Merge every recorded machine file into the one table the README publishes.

The README quotes numbers. Numbers that are retyped by hand drift from the numbers that
were measured, and the drift is always in the flattering direction, so nothing here is
retyped: this reads the committed `bench/results/machines/*.md` files, which are the
written-up record of actual runs, and emits the merged table from them.

That makes the tracked markdown the source of truth. A number can only reach the README by
first having been produced by `run-benchmark.sh` on a real machine and committed, which
also means a claim in the README can always be traced back to the machine and the commit
that produced it.

Sorted by program then machine, so a reader comparing one circuit across machines reads
down a contiguous block rather than hunting.

  render_readme_table.py [--dir bench/results/machines] [--mode warm|cold] [--out FILE]
"""
import argparse
import glob
import os
import re
import sys

# Ascending, so the table reads small circuits to large and the reader can see the shape of
# how each prover scales rather than a list in whatever order the filesystem produced.
COLUMNS = ["gpu-snark cpu", "gpu-snark metal", "gpu-snark cuda", "rapidsnark", "snarkjs"]


def parse_machine_file(path):
    """-> (machine, commit, {mode: {circuit: {"constraints": int, col: cell}}})"""
    text = open(path).read()

    m = re.search(r"^# (.+)$", text, re.M)
    machine = m.group(1).strip() if m else os.path.basename(path)
    m = re.search(r"^Commit `([^`]+)`", text, re.M)
    commit = m.group(1) if m else "unknown"

    # A file recorded through the load guard's override says so in its own note. Those
    # numbers include whatever else was running, so they are not allowed into the README.
    suspect = "SUSPECT" in text

    modes = {}
    for mode in ("warm", "cold"):
        body = re.search(rf"^## {mode}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
        if not body:
            continue
        rows = {}
        for line in body.group(1).splitlines():
            line = line.strip()
            if not line.startswith("| `"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 2 + len(COLUMNS):
                continue
            circuit = cells[0].strip("`")
            try:
                constraints = int(cells[1].replace(",", ""))
            except ValueError:
                continue
            rows[circuit] = {"constraints": constraints}
            rows[circuit].update(dict(zip(COLUMNS, cells[2:])))
        if rows:
            modes[mode] = rows
    return machine, commit, modes, suspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/results/machines")
    ap.add_argument("--mode", default="warm", choices=["warm", "cold"])
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.md")))
    if not files:
        sys.exit(f"no machine files under {a.dir}; run bench/scripts/run-benchmark.sh first")

    # circuit -> machine -> cells
    table = {}
    constraints = {}
    machines_seen = []
    skipped = []
    for f in files:
        machine, commit, modes, suspect = parse_machine_file(f)
        if suspect:
            skipped.append((machine, commit))
            continue
        rows = modes.get(a.mode)
        if not rows:
            continue
        label = machine
        if label not in machines_seen:
            machines_seen.append(label)
        for circuit, cells in rows.items():
            constraints[circuit] = cells["constraints"]
            table.setdefault(circuit, {})[label] = cells

    out = []
    out.append(f"| program | constraints | machine | {' | '.join(COLUMNS)} |")
    out.append("|---|---:|---|" + "---:|" * len(COLUMNS))
    for circuit in sorted(table, key=lambda c: constraints[c]):
        for machine in sorted(table[circuit]):
            cells = table[circuit][machine]
            vals = " | ".join(cells.get(c, "") for c in COLUMNS)
            out.append(f"| `{circuit}` | {constraints[circuit]:,} | {machine} | {vals} |")

    text = "\n".join(out) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        open(a.out, "w").write(text)
    for machine, commit in skipped:
        print(f"skipped {machine} at {commit}: recorded on a loaded box", file=sys.stderr)
    print(f"{len(table)} circuits across {len(machines_seen)} machines, mode {a.mode}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
