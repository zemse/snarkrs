#!/usr/bin/env python3
"""Build the README's benchmark section from the committed machine result files.

The README quotes numbers. Numbers that are retyped by hand drift from the numbers that
were measured, and the drift is always in the flattering direction, so nothing here is
retyped: this reads `bench/results/machines/*.md`, which are the written-up record of
actual runs, and emits the section from them. A figure can therefore always be traced back
to the machine and the commit that produced it.

One section per machine rather than one table with a machine column. Proving time varies
more between machines than between circuits, so interleaving them puts two numbers that
differ by 5x on adjacent lines and invites reading down a column that is not a comparison
of anything. Grouped by machine, each table compares provers on one box, which is the only
comparison the numbers actually support.

  render_readme_table.py [--dir bench/results/machines] [--out FILE]
"""
import argparse
import glob
import os
import re
import sys

COLUMNS = ["gpu-snark cpu", "gpu-snark metal", "gpu-snark cuda", "rapidsnark", "snarkjs"]

# Presentation only, and only a fallback. Machine files now record `cpu:` and `accelerator:`
# as their own fields, and when those are present the heading is built from them. These
# entries exist for files recorded before those fields did, where the hardware survives only
# inside the slug and cannot be recovered from it without knowing where the name ends.
# Nothing here invents a fact: each value restates what its slug already encodes.
LEGACY_LABELS = {
    "apple-m2-max": "Apple M2 Max",
    "aws-g4dn.2xlarge-tesla-t4": "AWS g4dn.2xlarge, NVIDIA Tesla T4",
}


def parse_machine_file(path):
    text = open(path).read()

    def field(pat, default=""):
        m = re.search(pat, text, re.M)
        return m.group(1).strip() if m else default

    machine = field(r"^# (.+)$", os.path.basename(path))
    commit = field(r"^Commit `([^`]+)`", "unknown")
    cpu = field(r"^cpu: (.+)$")
    gpu = field(r"^accelerator: (.+)$")

    m = re.search(r"^`([^`]+)`, (\d+) logical cores, os `([^`]+)`", text, re.M)
    arch, cores, osname = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")

    # A file recorded through the contention guard's override says so in its own note. Those
    # timings include whatever else was running, so they never reach the README.
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
            try:
                constraints = int(cells[1].replace(",", ""))
            except ValueError:
                continue
            rows[cells[0].strip("`")] = {
                "constraints": constraints,
                **dict(zip(COLUMNS, cells[2:])),
            }
        if rows:
            modes[mode] = rows

    return {
        "machine": machine, "commit": commit, "cpu": cpu, "gpu": gpu,
        "arch": arch, "cores": cores, "os": osname,
        "modes": modes, "suspect": suspect,
    }


def heading(info):
    if info["cpu"] or info["gpu"]:
        parts = [p for p in (info["cpu"], info["gpu"]) if p]
        return ", ".join(parts)
    return LEGACY_LABELS.get(info["machine"], info["machine"])


def table(rows):
    out = [f"| program | constraints | {' | '.join(COLUMNS)} |",
           "|---|---:|" + "---:|" * len(COLUMNS)]
    for circuit in sorted(rows, key=lambda c: rows[c]["constraints"]):
        cells = rows[circuit]
        vals = " | ".join(cells.get(c, "") for c in COLUMNS)
        out.append(f"| `{circuit}` | {cells['constraints']:,} | {vals} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/results/machines")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.md")))
    if not files:
        sys.exit(f"no machine files under {a.dir}; run bench/scripts/run-benchmark.sh first")

    L, skipped = [], []
    for f in files:
        info = parse_machine_file(f)
        if info["suspect"]:
            skipped.append(info)
            continue
        if not info["modes"]:
            continue
        L.append(f"### {heading(info)}")
        L.append("")
        L.append(f"`{info['arch']}`, {info['cores']} logical cores, {info['os']}. "
                 f"Measured at commit `{info['commit']}`.")
        L.append("")
        for mode, blurb in (
            ("warm", "**Warm**, setup paid once then proving in a loop, which is what a "
                     "resident service sees:"),
            ("cold", "**Cold**, one fresh process per proof so the key parse and any GPU "
                     "upload sit inside the timed region, which is what a CLI does:"),
        ):
            rows = info["modes"].get(mode)
            if not rows:
                continue
            L.append(blurb)
            L.append("")
            L.append(table(rows))
            L.append("")

    text = "\n".join(L).rstrip() + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        open(a.out, "w").write(text)
    for info in skipped:
        print(f"skipped {info['machine']} at {info['commit']}: recorded on a loaded box",
              file=sys.stderr)
    print(f"{len(files) - len(skipped)} machines rendered", file=sys.stderr)


if __name__ == "__main__":
    main()
