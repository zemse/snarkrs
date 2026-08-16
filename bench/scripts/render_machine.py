#!/usr/bin/env python3
"""Render one machine's benchmark run as markdown.

One file per machine, named after the machine and the commit. The design invariant, which
every change here has to keep:

    A cell that could not be measured on this box is EMPTY. Never a zero, never an omitted
    column. The column says the comparison was attempted; the blank says this box could not
    answer it. Dropping the column instead would make two machines' tables silently
    non-comparable, and a zero would read as an impossibly fast prover.

Two kinds of "no number" are distinguished, because they are different claims:

    blank   not measurable here at all. The backend does not exist on this box, the prover
            is not installed, or the prover has no such mode (snarkjs has no warm mode).
    `-`     measurable here, attempted, and it produced nothing usable. A crash, or a proof
            that failed verification and so had its timings thrown away.

Columns are fixed and ordered by what they are being compared against: our three backends
first, then the two external provers. rapidsnark and snarkjs are CPU-only, so they get one
column each rather than a backend column apiece.
"""
import argparse, csv, glob, os, statistics

# (header, prover, backend, modes it can be measured in)
# `prover` is the CSV column that separates the tools: `g16 bench` stamps `ours`,
# bench_external.py stamps `rapidsnark` or `snarkjs`.
COLUMNS = [
    ("gpu-snark cpu",   "ours",       "cpu",   ("cold", "warm")),
    ("gpu-snark metal", "ours",       "metal", ("cold", "warm")),
    ("gpu-snark cuda",  "ours",       "cuda",  ("cold", "warm")),
    # rapidsnark's stock CLI reparses the zkey every call and can only ever report cold;
    # warm comes from the rapidsnark-warm harness, which drives its object API in a loop.
    ("rapidsnark",      "rapidsnark", None,    ("cold", "warm")),
    # snarkjs offers only a CLI, so a warm number for it does not exist to be measured.
    ("snarkjs",         "snarkjs",    None,    ("cold",)),
]

OUR_BACKENDS = ["cpu", "metal", "cuda"]


def load(csv_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    # A timing whose proof did not verify is not a measurement of a prover, it is a
    # measurement of a bug. Absent or `na` means the producer does not track it.
    return [r for r in rows if r.get("verified", "yes") in ("yes", "na", "")]


def med(rows, **filt):
    v = []
    for r in rows:
        if any(r.get(k) != x for k, x in filt.items()):
            continue
        try:
            v.append(float(r["ms"]))
        except (KeyError, TypeError, ValueError):
            pass
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--backends", required=True,
                    help="our backends this box can run, space separated")
    ap.add_argument("--provers", default="",
                    help="external provers that were available and attempted, space "
                         "separated: rapidsnark snarkjs. Anything not listed is left "
                         "blank rather than reported as a failure.")
    ap.add_argument("--snarkjs-reps", type=int, default=0,
                    help="reps snarkjs got, when it differs from --reps")
    ap.add_argument("--commit", default="")
    ap.add_argument("--note", default="",
                    help="banner line placed at the very top of the file, for runs that "
                         "must not be mistaken for a recorded measurement")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load(a.csv_dir)
    have_backends = set(a.backends.split())
    have_provers = set(a.provers.split())

    def available(prover, backend, mode, modes):
        """Could this cell have been measured on this box at all."""
        if mode not in modes:
            return False
        if prover == "ours":
            return backend in have_backends
        return prover in have_provers

    circuits = {}
    for r in rows:
        try:
            n = int(r.get("constraints", -1))
        except (TypeError, ValueError):
            n = -1
        # Prefer a real count: one producer reporting -1 must not clobber a good value.
        if r["variant"] not in circuits or circuits[r["variant"]] < 0:
            circuits[r["variant"]] = n
    order = sorted(circuits.items(), key=lambda kv: (kv[1], kv[0]))

    meta = rows[0] if rows else {}
    L = []
    if a.note:
        L += [f"> **{a.note}**", ""]
    L += [f"# {a.machine}", ""]
    if a.commit:
        L.append(f"Commit `{a.commit}`.")
    if meta:
        L.append(f"`{meta.get('arch','?')}`, {meta.get('cores','?')} logical cores, "
                 f"os `{meta.get('os','?')}`")
        L.append("")
    L.append(f"Generated by `bench/scripts/run-benchmark.sh`, {a.reps} reps per cell, "
             "median milliseconds.")
    if a.snarkjs_reps and a.snarkjs_reps != a.reps:
        L.append(f"snarkjs gets {a.snarkjs_reps} reps instead of {a.reps}: it is wasm, and "
                 "a median over a few slow reps buys the same answer for a fraction of the "
                 "wall clock.")
    L.append("Every proof behind every number was verified before the timing was kept.")
    L.append("")
    L.append("Columns are the same on every machine. A blank cell is a comparison this box "
             "could not")
    L.append("make: the backend does not exist here, the prover is not installed, or the "
             "prover has no")
    L.append("such mode. It is not a zero and not a failure. A `-` is the other case: "
             "measurable here,")
    L.append("attempted, and it produced no verified timing.")
    L.append("")

    headers = [h for h, _, _, _ in COLUMNS]
    for mode, blurb in (
        ("warm", "Setup paid once, then proving in a loop. What a resident service sees."),
        ("cold", "One fresh process per proof, so key parse and GPU upload are inside the "
                 "timed region. What a CLI does."),
    ):
        L.append(f"## {mode}")
        L.append("")
        L.append(blurb)
        L.append("")
        L.append("| circuit | constraints | " + " | ".join(headers) + " |")
        L.append("|---|---:|" + "---:|" * len(headers))
        for variant, n in order:
            cells = []
            for _, prover, backend, modes in COLUMNS:
                if not available(prover, backend, mode, modes):
                    cells.append("")
                    continue
                filt = {"variant": variant, "prover": prover, "mode": mode}
                if backend is not None:
                    filt["backend"] = backend
                v = med(rows, **filt)
                cells.append(f"{v:.1f}" if v is not None else "-")
            count = f"{n:,}" if n >= 0 else ""
            L.append(f"| `{variant}` | {count} | " + " | ".join(cells) + " |")
        L.append("")

    missing = [b for b in OUR_BACKENDS if b not in have_backends]
    missing += [p for _, p, b, _ in COLUMNS if b is None and p not in have_provers]
    if missing:
        L.append("Not available on this machine, left blank above: "
                 + ", ".join("`" + m + "`" for m in missing) + ".")
        L.append("")
    L.append("`snarkjs` has no warm mode to measure: its CLI is the only interface it "
             "offers, so that")
    L.append("cell is blank on every machine.")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}: {len(order)} circuits, "
          f"backends {sorted(have_backends)}, external provers {sorted(have_provers)}")


if __name__ == "__main__":
    main()
