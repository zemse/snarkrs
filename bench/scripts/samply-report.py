#!/usr/bin/env python3
"""Aggregate a samply profile into self time and inclusive time per function.

Why not just open the samply UI: the UI is per-thread and this prover runs 12 rayon
workers, so every interesting number is spread across 13 call trees. This folds them into
one, weights each sample by the CPU time it represents rather than counting samples, and
prints something that can be pasted into a report and re-derived later.

Weighting matters here. samply's macOS sampler drops samples under load, and this profile
shows threads with equal CPU time and a 3x spread in sample count. Counting samples would
therefore over-weight whichever workers the sampler happened to keep up with. Each sample
carries `threadCPUDelta` (microseconds of CPU since the previous sample on that thread),
which is the honest weight.

Usage:
    bench/scripts/samply-report.py PROFILE.json.gz [--top N] [--group]
"""
import bisect
import gzip
import json
import os
import re
import sys
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        return json.load(fh)


def symbolicator(profile_path):
    """Map "0x1234" placeholder names back to real symbols.

    samply writes address placeholders into the profile and resolves them in the browser
    at load time, so a saved profile read from a script has no names in it. With
    `--unstable-presymbolicate` it also writes a `.syms.json` sidecar holding, per library,
    a sorted `(rva, size, symbol)` table. This bisects that table.

    Without the sidecar every name stays an address and the report is useless, so its
    absence is an error rather than a silent degradation.
    """
    sidecar = profile_path + ".syms.json"
    if profile_path.endswith(".gz"):
        sidecar = profile_path[: -len(".gz")] + ".syms.json"
    if not os.path.exists(sidecar):
        sys.exit(
            f"missing {sidecar}\n"
            "re-record with: samply record --save-only --unstable-presymbolicate ..."
        )
    with open(sidecar) as fh:
        syms = json.load(fh)
    strings = syms["string_table"]
    tables = {}
    for lib in syms["data"]:
        entries = sorted(lib["symbol_table"], key=lambda e: e["rva"])
        tables[lib["debug_name"]] = (
            [e["rva"] for e in entries],
            entries,
        )

    def resolve(lib_name, name):
        if not name.startswith("0x"):
            return name
        table = tables.get(lib_name)
        if table is None:
            return f"{lib_name}!{name}"
        rvas, entries = table
        addr = int(name, 16)
        i = bisect.bisect_right(rvas, addr) - 1
        if i < 0:
            return f"{lib_name}!{name}"
        e = entries[i]
        # `size` is authoritative when present: an address past the end of a symbol
        # belongs to no symbol, and guessing would attribute padding to the neighbour.
        if e.get("size") and addr >= e["rva"] + e["size"]:
            return f"{lib_name}!{name}"
        return strings[e["symbol"]]

    return resolve


def demangle_group(name):
    """Bucket a symbol by the thing it belongs to, for the coarse view."""
    # Matched against the whole symbol, not its prefix: a trait impl symbol starts with
    # "<", so `ark_ff` is never the first token even when ark-ff owns every instruction
    # in the function. Order matters, first match wins.
    for needle, label in [
        ("quadratic_extension", "arkworks Fq2 (G2 tower)"),
        ("ark_ff", "arkworks Fp (ark-ff)"),
        ("ark_ec", "arkworks curve (ark-ec)"),
        ("ark_bn254", "arkworks bn254"),
        ("ark_serialize", "arkworks serialize"),
        ("g16_msm", "g16-msm"),
        ("g16_ntt", "g16-ntt"),
        ("g16_core", "g16-core"),
        ("g16_zkey", "g16-zkey"),
        ("g16_field", "g16-field"),
        ("rayon", "rayon (scheduler + inlined par_iter bodies)"),
        ("crossbeam", "rayon (scheduler + inlined par_iter bodies)"),
        ("swtch_pri", "idle / spin (thread_switch, cond wait)"),
        ("cthread_yield", "idle / spin (thread_switch, cond wait)"),
        ("psynch_cv", "idle / spin (thread_switch, cond wait)"),
        ("psynch_mutex", "idle / spin (thread_switch, cond wait)"),
        ("std::", "std"),
        ("core::", "core"),
        ("alloc::", "alloc"),
    ]:
        if needle in name:
            return label
    if re.match(r"^(_?malloc|free|szone|nanov2|tiny_|small_|large_)", name):
        return "malloc/free"
    if name.startswith("__") or name.startswith("_platform") or "libsystem" in name:
        return "libsystem"
    return "other"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    top = 30
    folded_out = None
    callers_of = None
    depth = 6
    for f in flags:
        if f.startswith("--top"):
            top = int(f.split("=")[1]) if "=" in f else 30
        if f.startswith("--callers="):
            callers_of = f.split("=", 1)[1]
        if f.startswith("--depth="):
            depth = int(f.split("=", 1)[1])
        if f.startswith("--folded="):
            folded_out = f.split("=", 1)[1]
    if not args:
        print(__doc__)
        sys.exit(2)

    prof = load(args[0])
    resolve = symbolicator(args[0])
    lib_names = [lib.get("debugName") or lib.get("name") for lib in prof["libs"]]
    self_us = defaultdict(float)
    incl_us = defaultdict(float)
    group_us = defaultdict(float)
    caller_us = defaultdict(float)
    folded = defaultdict(float)
    total = 0.0

    for thread in prof["threads"]:
        strings = thread["stringArray"]
        funcs = thread["funcTable"]["name"]
        frame_func = thread["frameTable"]["func"]
        stack_prefix = thread["stackTable"]["prefix"]
        stack_frame = thread["stackTable"]["frame"]
        samples = thread["samples"]
        stacks = samples["stack"]
        cpu = samples.get("threadCPUDelta") or [None] * len(stacks)

        # Resolve each stack node's function name once; a stack table is a tree with heavy
        # sharing, so this is much cheaper than walking names per sample.
        # func -> library, so an address placeholder can be looked up in the right table.
        res_lib = thread["resourceTable"]["lib"]
        func_res = thread["funcTable"]["resource"]

        def func_name(fi):
            raw = strings[funcs[fi]]
            r = func_res[fi]
            lib = lib_names[res_lib[r]] if r is not None and r >= 0 else "?"
            return resolve(lib, raw)

        cache = {}
        node_name = []
        for f in stack_frame:
            fi = frame_func[f]
            if fi not in cache:
                cache[fi] = func_name(fi)
            node_name.append(cache[fi])

        for s, w in zip(stacks, cpu):
            if s is None:
                continue
            weight = float(w) if w else 0.0
            if weight <= 0:
                continue
            total += weight
            self_us[node_name[s]] += weight
            group_us[demangle_group(node_name[s])] += weight
            # Inclusive: credit each distinct function on the stack once, so recursion
            # does not multiply-count.
            seen = set()
            node = s
            while node is not None:
                seen.add(node_name[node])
                node = stack_prefix[node]
            for n in seen:
                incl_us[n] += weight
            # Rayon steals work, so a stolen closure's stack is rooted at the worker
            # thread rather than at the function that spawned it. Inclusive time under a
            # caller therefore undercounts badly, and the only honest way to ask "which
            # loop is this hot leaf in" is to look at the frames immediately above it.
            if folded_out is not None:
                # Flamegraph "collapsed stack" format: root-first, semicolon separated,
                # weight last. inferno-collapse / flamegraph.pl read this directly, which
                # is how a flame graph gets made here without dtrace and without sudo.
                chain = []
                node = s
                while node is not None:
                    chain.append(node_name[node])
                    node = stack_prefix[node]
                folded[";".join(reversed(chain))] += weight
            if callers_of is not None and callers_of in node_name[s]:
                chain = []
                node = stack_prefix[s]
                while node is not None and len(chain) < depth:
                    chain.append(node_name[node])
                    node = stack_prefix[node]
                caller_us[" <- ".join(chain)] += weight

    if total == 0:
        print("no weighted samples; profile has no threadCPUDelta")
        sys.exit(1)

    print(f"total CPU time sampled: {total / 1e3:.1f} ms across {len(prof['threads'])} threads")
    print()
    print("== self time by group ==")
    for name, us in sorted(group_us.items(), key=lambda kv: -kv[1]):
        print(f"{100 * us / total:6.2f}%  {us / 1e3:9.1f} ms  {name}")

    print()
    print(f"== self time, top {top} functions ==")
    for name, us in sorted(self_us.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{100 * us / total:6.2f}%  {us / 1e3:9.1f} ms  {name}")

    if callers_of is not None:
        matched = sum(v for k, v in self_us.items() if callers_of in k)
        print()
        print(f"== callers of {callers_of!r} ({100 * matched / total:.2f}% self time), top {top} ==")
        for chain, us in sorted(caller_us.items(), key=lambda kv: -kv[1])[:top]:
            print(f"{100 * us / total:6.2f}%  {us / 1e3:9.1f} ms")
            for i, frame in enumerate(chain.split(" <- ")):
                print(f"        {'  ' * i}{frame}")

    if folded_out is not None:
        with open(folded_out, "w") as fh:
            for chain, us in sorted(folded.items(), key=lambda kv: -kv[1]):
                fh.write(f"{chain} {int(round(us))}\n")
        print(f"\nwrote {len(folded)} folded stacks (weight = CPU microseconds) to {folded_out}")

    print()
    print(f"== inclusive time, top {top} functions ==")
    for name, us in sorted(incl_us.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{100 * us / total:6.2f}%  {us / 1e3:9.1f} ms  {name}")


if __name__ == "__main__":
    main()
