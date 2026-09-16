#!/usr/bin/env python3
"""Reproduce the 75ed942 selector and its assumptions without a GPU or timings.

The constants are snapshots of 25536d4 and 75ed942, not a replacement selector.
Pass m and recode_bits to print every candidate, or omit them for the review cases.
"""

import argparse
from dataclasses import dataclass


def ceil_div(n, d):
    return (n + d - 1) // d


@dataclass
class Candidate:
    c: int
    windows: int
    buckets: int
    top: int
    span: float
    actual_slice: int
    cost: float


def candidates(m, bits, row=0.022, wide=100.0, use_actual_slice=False):
    result = []
    for c in range(3, 17):
        windows = ceil_div(bits, c)
        buckets = 1 << (c - 1)
        top = 1 << (bits - (windows - 1) * c - 1)
        actual_slice = 64
        while actual_slice > 8 and windows * ceil_div(max(m, 1), actual_slice) < 4096:
            actual_slice //= 2
        span = m / top / (actual_slice if use_actual_slice else 64)
        fat = wide + 27 * span / 64 if wide is not None and span >= 4 else 27 * span
        cost = 0.00324 * windows * m + row * windows * buckets + fat
        result.append(Candidate(c, windows, buckets, top, span, actual_slice, cost))
    return result


def winner(m, bits, **kwargs):
    return min(candidates(m, bits, **kwargs), key=lambda p: p.cost)


def report(m, bits):
    print(f"\nm={m} recode_bits={bits}")
    for label, kwargs in [
        ("25536d4", dict(row=0.025, wide=None)),
        ("75ed942", {}),
        ("actual slice only, not a refit", dict(use_actual_slice=True)),
        ("WIDE_US=60", dict(wide=60)),
        ("WIDE_US=160", dict(wide=160)),
    ]:
        best = winner(m, bits, **kwargs)
        print(f"  {label}: c={best.c}, cost={best.cost:.6f} us")
    print("  c  w  buckets  model_span  actual_L  actual_span  wide  host_tail  cost_us")
    for p in candidates(m, bits):
        actual_span = m / p.top / p.actual_slice
        print(f" {p.c:2} {p.windows:2} {p.buckets:8} {p.span:11.4f}"
              f" {p.actual_slice:9} {actual_span:12.4f} {actual_span >= 4!s:>5}"
              f" {p.windows == 1 and p.buckets <= 256!s:>10} {p.cost:10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m", type=int, nargs="?")
    parser.add_argument("bits", type=int, nargs="?", default=255)
    args = parser.parse_args()
    if args.m is not None:
        if args.m < 0 or not 1 <= args.bits <= 255:
            parser.error("m must be nonnegative and bits must be in 1..255")
        report(args.m, args.bits)
        return
    for m, bits in [(5, 8), (2, 8), (8, 255), (512, 255),
                    (32768, 255), (65536, 255), (131072, 255),
                    (262144, 255), (524288, 255), (2097152, 255)]:
        report(m, bits)
    print("\nFirst 12 power-of-two shapes sensitive to the claimed WIDE_US bracket:")
    found = 0
    for bits in range(1, 256):
        for exp in range(2, 22):
            m = 1 << exp
            lo = winner(m, bits, wide=60)
            hi = winner(m, bits, wide=160)
            if lo.c != hi.c:
                print(f"  m={m}, bits={bits}: c={lo.c} at 60, c={hi.c} at 160")
                found += 1
                if found == 12:
                    return


if __name__ == "__main__":
    main()
