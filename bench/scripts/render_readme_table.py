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
  render_readme_table.py --check-readme README.md

The check covers the README's warm proving tables, not the separate ceremony timings.
"""
import argparse
import glob
import os
import re
import sys

COLUMNS = ["gpu-snark cpu", "gpu-snark metal", "gpu-snark cuda", "rapidsnark", "snarkjs"]


def parse_machine_file(path):
    with open(path) as f:
        text = f.read()

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


def describe(info):
    """The line under the heading: what this machine is, in the machine's own words.

    The heading itself is the slug, which for EC2 is the bare instance type. That is the
    canonical name for the box and the string you type to rent the same one again, so it
    should not be decorated. Everything else the run detected goes here instead of being
    concatenated into the slug.
    """
    bits = [f"`{info['arch']}`", f"{info['cores']} logical cores", info["os"]]
    if info["cpu"]:
        bits.append(info["cpu"])
    if info["gpu"]:
        bits.append(info["gpu"])
    return ", ".join(bits) + f". Measured at commit `{info['commit']}`."


def table(rows):
    out = [f"| program | constraints | {' | '.join(COLUMNS)} |",
           "|---|---:|" + "---:|" * len(COLUMNS)]
    for circuit in sorted(rows, key=lambda c: rows[c]["constraints"]):
        cells = rows[circuit]
        vals = " | ".join(cells.get(c, "") for c in COLUMNS)
        out.append(f"| `{circuit}` | {cells['constraints']:,} | {vals} |")
    return "\n".join(out)


def readme_table(rows):
    backends = [b for b in ("metal", "cuda")
                if any(r[f"gpu-snark {b}"] for r in rows.values())]
    if len(backends) != 1:
        raise ValueError("README tables require exactly one measured GPU backend")
    backend = backends[0]
    out = ["<table>", "<thead>",
           '<tr><th rowspan="2">program</th><th rowspan="2" align="right">constraints</th>'
           '<th>GPU</th><th colspan="3">CPU</th></tr>',
           f'<tr><th align="right">g16 {backend} (ms)</th>'
           '<th align="right">g16 cpu (ms)</th><th align="right">rapidsnark (ms)</th>'
           '<th align="right">snarkjs (ms)</th></tr>',
           "</thead>", "<tbody>"]
    columns = [f"gpu-snark {backend}", "gpu-snark cpu", "rapidsnark", "snarkjs"]
    for circuit in sorted(rows, key=lambda c: rows[c]["constraints"]):
        cells = rows[circuit]
        values = [f"{cells['constraints']:,}"] + [cells[c] for c in columns]
        out.append(f"<tr><td><code>{circuit}</code></td>" +
                   "".join(f'<td align="right">{v}</td>' for v in values) + "</tr>")
    return "\n".join(out + ["</tbody>", "</table>"])


def check_readme(path, files):
    with open(path) as f:
        text = f.read()
    benchmark = re.search(r"^## benchmarks\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not benchmark:
        raise ValueError("README has no benchmarks section")
    sections = re.findall(r"^### ([^\n]+)\n(.*?)(?=^### |\Z)",
                          benchmark.group(1), re.M | re.S)
    actual = {}
    for machine, body in sections:
        tables = re.findall(r"<table>.*?</table>", body, re.S)
        if machine in actual or len(tables) != 1:
            raise ValueError(f"{machine}: expected one warm proving table")
        actual[machine] = tables[0]

    expected = {}
    for file in files:
        info = parse_machine_file(file)
        machine = info["machine"]
        if info["suspect"]:
            continue
        if machine in expected:
            raise ValueError(f"{machine}: multiple source records; select one explicitly")
        if "warm" not in info["modes"]:
            raise ValueError(f"{file}: no warm measurements")
        expected[machine] = readme_table(info["modes"]["warm"])
    if not expected:
        raise ValueError("no non-suspect warm measurements")
    if actual.keys() != expected.keys():
        raise ValueError("README machines differ from non-suspect source records")
    for machine, table_text in expected.items():
        if actual[machine] != table_text:
            raise ValueError(f"{machine}: README warm table is stale; update it from the recorded results")
    print(f"{len(expected)} README warm tables match the recorded results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/results/machines")
    ap.add_argument("--out", default="-")
    ap.add_argument("--check-readme", metavar="FILE",
                    help="fail if README warm proving tables differ from the recorded results")
    a = ap.parse_args()
    if a.check_readme and a.out != "-":
        ap.error("--check-readme cannot be combined with --out")

    files = sorted(glob.glob(os.path.join(a.dir, "*.md")))
    if not files:
        sys.exit(f"no machine files under {a.dir}; run bench/scripts/run-benchmark.sh first")

    if a.check_readme:
        try:
            check_readme(a.check_readme, files)
        except ValueError as e:
            sys.exit(str(e))
        return

    L, skipped = [], []
    for f in files:
        info = parse_machine_file(f)
        if info["suspect"]:
            skipped.append(info)
            continue
        if not info["modes"]:
            continue
        L.append(f"### {info['machine']}")
        L.append("")
        L.append(describe(info))
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
        with open(a.out, "w") as f:
            f.write(text)
    for info in skipped:
        print(f"skipped {info['machine']} at {info['commit']}: recorded on a loaded box",
              file=sys.stderr)
    print(f"{len(files) - len(skipped)} machines rendered", file=sys.stderr)


if __name__ == "__main__":
    main()
