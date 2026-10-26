#!/usr/bin/env python3
"""Percent of one core used by processes outside a process group, measured from /proc.

procps `ps -o pcpu` is CPU time over the whole lifetime of the process, so a long-lived
daemon that has just become busy barely moves it. This reads utime+stime from every
/proc/<pid>/stat twice, an interval apart, and divides the difference by the interval.

A process that starts and exits inside the interval is in neither reading. Its time lands in
its parent's cutime+cstime when the parent reaps it, so those count too, less what a child
that was already running had used before the interval.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import time


def parse_stat(line):
    """(ppid, pgrp, utime+stime+cutime+cstime in clock ticks) from one /proc/<pid>/stat line."""
    # comm is in parentheses and may itself hold spaces or parentheses, so split after the last one.
    fields = line[line.rindex(")") + 2:].split()
    return fields[1], int(fields[2]), sum(int(f) for f in fields[11:15])


def snapshot(proc):
    taken = {}
    for stat in Path(proc).glob("[0-9]*/stat"):
        try:
            taken[stat.parent.name] = parse_stat(stat.read_text())
        except (OSError, ValueError, IndexError):
            # The process exited between the listing and the read.
            continue
    return taken


def foreign_percent(before, after, me, hz, seconds):
    # A child that exits inside the interval hands its whole lifetime to its parent's
    # cutime+cstime, including the part already counted in the first reading.
    gone = defaultdict(int)
    for pid, (ppid, _, used) in before.items():
        if pid not in after:
            gone[ppid] += used
    ticks = 0
    for pid, (_, pgrp, used) in after.items():
        if pgrp == me:
            continue
        # A process that started inside the interval spent all of its time inside it.
        ticks += max(used - before.get(pid, (None, pgrp, 0))[2] - gone[pid], 0)
    return 100.0 * ticks / hz / max(seconds, 1e-3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgid", type=int, required=True, help="our own process group, left out")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--proc", default="/proc")
    a = ap.parse_args()
    start, before = time.monotonic(), snapshot(a.proc)
    time.sleep(a.interval)
    after = snapshot(a.proc)
    seconds = time.monotonic() - start
    print(f"{foreign_percent(before, after, a.pgid, os.sysconf('SC_CLK_TCK'), seconds):.0f}")


if __name__ == "__main__":
    main()
