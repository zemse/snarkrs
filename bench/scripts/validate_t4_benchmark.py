#!/usr/bin/env python3
"""Opt-in CSV-only check of the published T4 matrix, not benchmark readiness.

Cannot attest real hardware, source/binary hashes, actual proof verification,
absence of CPU fallback, or contention/memory guards from CSV alone.
"""
import argparse
import csv
import json
import math
import sys


CIRCUITS = {
    "railgun-01x01": 20135,
    "tornado": 28275,
    "sha256": 59281,
    "railgun-13x01": 141276,
    "rsa2048": 190945,
    "keccak256": 239176,
    "anon-aadhaar": 1115080,
}
PREFIX = "host,os,arch,cores,variant,constraints,prover,backend,mode,rep,ms,prepare_ms,gather_us,ntt_us,pointwise_us,msm_us,assemble_us,verified".split(",")
SPLIT = "msm_a_us,msm_b_g2_us,msm_b_g1_us,msm_l_us,msm_h_us".split(",")
OURS_HEADER = PREFIX + ["adx_bmi2"] + SPLIT + ["msm_serial"]
EXTERNAL_HEADER = PREFIX + ["verification_scope", "verification_oracle"]
STAGES = PREFIX[11:17]
ORACLES = {"rapidsnark-verify", "snarkrs verify", "snarkjs verify"}
LIMITATIONS = (
    "CSV-only validation; cannot attest real hardware, source/binary hashes, "
    "actual proof verification, absence of CPU fallback, or contention/memory guards. "
    "This is not full benchmark readiness."
)


def expected_cells():
    cells = {}
    for variant in CIRCUITS:
        for backend in ("cpu", "cuda"):
            for mode in ("warm", "cold-inprocess", "cold-process"):
                cells[(variant, "ours", backend, mode)] = 10
        for mode in ("warm", "cold"):
            cells[(variant, "rapidsnark", "cpu", mode)] = 10
        cells[(variant, "snarkjs", "cpu", "cold")] = 3
    return cells


def number(value, field, positive=False):
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"{field}: invalid number {value!r}") from None
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{field}: must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def validate(paths, *, expected_host, expected_os, expected_arch, expected_cores):
    """Validate separate producer CSVs completely before computing any medians."""
    if not expected_host or expected_host.strip() != expected_host or expected_host == "unknown":
        raise ValueError("expected host must be an independently supplied nonempty stamp")
    # Both current producers use platform-style spellings, not Rust's lowercase OS.
    if (expected_os, expected_arch, expected_cores) != ("Linux", "x86_64", 8):
        raise ValueError("published T4 protocol requires Linux, x86_64, 8 cores")
    stamps = dict(host=expected_host, os=expected_os, arch=expected_arch,
                  cores=str(expected_cores))
    expected = expected_cells()
    samples = {}
    for path in paths:
        try:
            with open(path, newline="", encoding="utf-8") as source:
                reader = csv.reader(source, strict=True)
                header = next(reader, None)
                if header == OURS_HEADER:
                    ours = True
                elif header == EXTERNAL_HEADER:
                    ours = False
                else:
                    raise ValueError("header differs from both current producer schemas")
                count = 0
                for fields in reader:
                    count += 1
                    if len(fields) != len(header):
                        raise ValueError(f"line {reader.line_num}: wrong field count")
                    row = dict(zip(header, fields))
                    where = f"line {reader.line_num}"
                    for field, value in stamps.items():
                        if row[field] != value:
                            raise ValueError(f"{where}: mismatched {field} stamp")
                    key = tuple(row[k] for k in ("variant", "prover", "backend", "mode"))
                    if key not in expected or (row["prover"] == "ours") != ours:
                        raise ValueError(f"{where}: unexpected cell or producer {key}")
                    if row["constraints"] != str(CIRCUITS[row["variant"]]):
                        raise ValueError(f"{where}: wrong constraints for {row['variant']}")
                    if row["verified"] != "yes":
                        raise ValueError(f"{where}: verified must be yes")
                    if row["rep"] not in {str(i) for i in range(1, expected[key] + 1)}:
                        raise ValueError(f"{where}: repetition ID outside exact protocol range")
                    rep = int(row["rep"])
                    cell = samples.setdefault(key, {})
                    if rep in cell:
                        raise ValueError(f"{where}: duplicate repetition {rep} for {key}")
                    ms = number(row["ms"], f"{where} ms", positive=True)
                    optional = STAGES + (SPLIT if ours else [])
                    for field in optional:
                        if row[field] != "":
                            number(row[field], f"{where} {field}")
                    if ours:
                        cold = row["mode"] == "cold-process"
                        if row["adx_bmi2"] != "yes":
                            raise ValueError(f"{where}: adx_bmi2 must be yes")
                        if row["msm_serial"] != ("" if cold else "no"):
                            raise ValueError(f"{where}: normal nonserial metadata required")
                        if any(row[field] != "" for field in SPLIT):
                            raise ValueError(f"{where}: serial MSM split values are not normal timings")
                        if cold and any(row[field] != "" for field in STAGES):
                            raise ValueError(f"{where}: cold-process preparation/stages must be empty")
                    else:
                        scope = "each-proof" if row["mode"] == "cold" else "final-proof-only"
                        if row["verification_scope"] != scope:
                            raise ValueError(f"{where}: wrong verification scope")
                        if row["verification_oracle"] not in ORACLES:
                            raise ValueError(f"{where}: unknown verification oracle")
                    cell[rep] = ms
                if not count:
                    raise ValueError("CSV contains no measured rows")
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"{path}: {exc}") from None
    for key, reps in expected.items():
        if set(samples.get(key, {})) != set(range(1, reps + 1)):
            raise ValueError(f"incomplete cell {key}: expected repetition IDs 1..{reps}")
    cells = []
    for key in sorted(expected):
        values = sorted(samples[key].values())
        middle = len(values) // 2
        median = values[middle]
        if len(values) % 2 == 0:
            lower = values[middle - 1]
            # Avoid overflow when averaging two finite positive values.
            median = lower + (median - lower) / 2
        cells.append(dict(zip(("variant", "prover", "backend", "mode"), key),
                          count=len(values), median_ms=median))
    return {"validation": "csv-only", "limitations": LIMITATIONS, "expected_stamps": stamps,
            "cell_count": len(cells), "row_count": sum(c["count"] for c in cells), "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="separate producer CSV files, never heterogeneous concatenation")
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--expected-os", required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--expected-cores", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = validate(args.csv, expected_host=args.expected_host, expected_os=args.expected_os,
                          expected_arch=args.expected_arch, expected_cores=args.expected_cores)
    except (ValueError, OSError) as exc:
        print(f"T4 CSV validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
