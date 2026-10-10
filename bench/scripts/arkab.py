#!/usr/bin/env python3
"""Interleaved A/B of two snarkrs builds that differ only in their arkworks checkout.

    arkab.py --bin-dir DIR --variants stock perf --backend cpu --artifacts bench/artifacts \
        --ptau-dir bench/ptau --out target/arkab/mac-cpu [--rounds 3] [--steps ...]

DIR holds `snarkrs-<variant>` and `verify_time-<variant>` for each variant. Every round runs
every step under both variants back to back, the variant order reversing each round (AB, BA,
AB), so drift in the machine lands on both. Steps:

  prove     `snarkrs bench --mode warm`, every rep self-verified by the bench itself
  proof     one `groth16 prove` per variant and circuit (round 1), the verify step's input
  verify    `verify_time`: verify in process, after checking a tampered public signal fails
  trace     `snarkrs trace` on the first circuit (round 1): the deterministic prove trace
  contribute, pverify, prepare
            phase 1 at each `--contribute` / `--prepare` size: `powersoftau contribute`, then
            `powersoftau verify` on its output, and `powersoftau prepare phase2`
  beacon    `powersoftau beacon` at each contribute size (round 1): deterministic output
  setup, zcontribute, zverify, zbeacon
            phase 2 per `--phase2` circuit: `groth16 setup`, `zkey contribute`, `zkey verify
            init`, and (round 1) `zkey beacon`

Every ceremony timing is the wall clock of one CLI process, counted only when the process
exits 0 and, for the verifiers, prints its OK line. Deterministic outputs (prepare, setup,
beacons, trace) are hashed per variant and round so they can be compared byte for byte.
Round-1 outputs worth a second verifier are kept under OUT/keep for snarkjs afterwards.

With `--max-load`, each timed command waits until the one-minute load average is below it.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

CIRCUITS = ["js_2x2_d32", "railgun-01x01", "tornado", "sha256", "railgun-13x01",
            "js_16x16_d32", "rsa2048", "keccak256", "anon-aadhaar"]
BEACON = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"


def say(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def domain_power(zkey):
    """log2 of the groth16 domain size, from the zkey's section 2."""
    with open(zkey, "rb") as f:
        if f.read(4) != b"zkey":
            raise ValueError(f"{zkey} is not a zkey")
        _, nsec = struct.unpack("<II", f.read(8))
        for _ in range(nsec):
            typ, size = struct.unpack("<IQ", f.read(12))
            if typ != 2:
                f.seek(size, 1)
                continue
            n8q = struct.unpack("<I", f.read(4))[0]
            f.seek(n8q, 1)
            n8r = struct.unpack("<I", f.read(4))[0]
            f.seek(n8r, 1)
            _, _, domain = struct.unpack("<III", f.read(12))
            return domain.bit_length() - 1
    raise ValueError(f"{zkey} has no groth16 header")


class Runner:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out).resolve()
        self.tmp = self.out / "tmp"
        self.keep = self.out / "keep"
        self.rows = []
        self.hashes = []
        self.failures = []
        for d in (self.out, self.tmp, self.keep):
            d.mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "commands.log", "a")

    def bin(self, v, name="snarkrs"):
        return str(Path(self.a.bin_dir) / f"{name}-{v}")

    def wait_load(self):
        if self.a.max_load is None:
            return
        while True:
            load = os.getloadavg()[0]
            if load < self.a.max_load:
                return
            say(f"  load {load:.2f} >= {self.a.max_load}, waiting")
            time.sleep(30)

    def run(self, cmd, ok_text=None, timed=True):
        """Run one command; return its wall clock in ms, or None if it failed."""
        if timed:
            self.wait_load()
        load = os.getloadavg()[0]
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        ms = (time.perf_counter() - t) * 1000
        text = r.stdout + r.stderr
        good = r.returncode == 0 and (ok_text is None or ok_text in text)
        self.log.write(f"$ {' '.join(map(str, cmd))}\n# exit {r.returncode} {ms:.1f} ms load {load:.2f}\n"
                       f"{text[-4000:]}\n")
        self.log.flush()
        if not good:
            self.failures.append({"cmd": [str(c) for c in cmd], "exit": r.returncode, "tail": text[-2000:]})
            say(f"  FAILED ({r.returncode}): {' '.join(map(str, cmd))}\n{text[-1500:]}")
            return None
        return ms

    def record(self, rnd, v, step, item, ms, **extra):
        row = {"round": rnd, "variant": v, "step": step, "item": item, "ms": ms,
               "load": round(os.getloadavg()[0], 2), **extra}
        self.rows.append(row)
        self.save()

    def digest(self, rnd, v, what, path):
        self.hashes.append({"round": rnd, "variant": v, "what": what, "sha256": sha256(path)})
        self.save()

    def save(self):
        (self.out / "rows.json").write_text(json.dumps(self.rows, indent=0))
        (self.out / "hashes.json").write_text(json.dumps(self.hashes, indent=0))
        (self.out / "failures.json").write_text(json.dumps(self.failures, indent=1))

    # -- steps -------------------------------------------------------------------------

    def prove(self, rnd, v, c):
        f = self.out / "prove" / f"{v}-{c}-r{rnd}.csv"
        f.parent.mkdir(exist_ok=True)
        ms = self.run([self.bin(v), "bench", "--artifacts", self.a.artifacts, "--variant", c,
                       "--mode", "warm", "--reps", str(self.a.reps), "--backend", self.a.backend,
                       "--csv", str(f)])
        if ms is None:
            return
        rows = [r for r in csv.DictReader(open(f)) if r["mode"] == "warm"]
        if len(rows) != self.a.reps or any(r["verified"] != "yes" for r in rows):
            self.failures.append({"step": "prove", "variant": v, "circuit": c, "round": rnd,
                                  "why": f"{len(rows)} rows"})
            return
        for r in rows:
            self.record(rnd, v, "prove", c, float(r["ms"]), rep=int(r["rep"]),
                        **{k: int(r[f"{k}_us"]) / 1000 for k in
                           ["gather", "ntt", "pointwise", "msm", "assemble"]},
                        prepare_ms=float(r["prepare_ms"]))

    def proof(self, v, c):
        art = Path(self.a.artifacts) / c
        d = self.keep / v / "proofs" / c
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(art / "vkey.json", d / "vkey.json")
        ms = self.run([self.bin(v), "groth16", "prove", art / "circuit.zkey", art / "circuit.wtns",
                       d / "proof.json", d / "public.json", "--backend", self.a.backend,
                       "--vkey", art / "vkey.json"], timed=False)
        if ms is not None and json.loads((d / "public.json").read_text()) != json.loads(
                (art / "public.json").read_text()):
            self.failures.append({"step": "proof", "variant": v, "circuit": c, "why": "public differs"})

    def verify(self, rnd, v, circuits):
        dirs = [self.keep / v / "proofs" / c for c in circuits]
        dirs = [d for d in dirs if (d / "proof.json").exists()]
        self.wait_load()
        r = subprocess.run([self.bin(v, "verify_time"), str(self.a.verify_reps), *map(str, dirs)],
                           capture_output=True, text=True)
        self.log.write(f"$ verify_time {v}\n# exit {r.returncode}\n{r.stdout}{r.stderr}\n")
        if r.returncode:
            self.failures.append({"step": "verify", "variant": v, "round": rnd, "tail": r.stderr[-2000:]})
            say(f"  verify_time FAILED: {r.stderr[-800:]}")
            return
        for line in r.stdout.splitlines():
            name, reps, med, lo, hi = line.split()
            self.record(rnd, v, "verify", name, float(med) / 1000, reps=int(reps),
                        min_ms=float(lo) / 1000, max_ms=float(hi) / 1000)

    def trace(self, v, c):
        art = Path(self.a.artifacts) / c
        out = self.tmp / f"trace-{v}.txt"
        if self.run([self.bin(v), "trace", "--zkey", art / "circuit.zkey", "--witness",
                     art / "circuit.wtns", "--backend", self.a.backend, "--out", out],
                    timed=False) is not None:
            self.digest(1, v, f"trace {c}", out)
            out.unlink()

    def phase1(self, rnd, v, steps):
        b = ["--backend", self.a.ceremony_backend]
        for power, src in self.a.contribute:
            out = self.tmp / f"contrib-{v}-{power}.ptau"
            if "contribute" in steps:
                ms = self.run([self.bin(v), "powersoftau", "contribute", src, out, "--name=arkab",
                               "--entropy=arkab bench entropy", *b])
                if ms is not None:
                    self.record(rnd, v, "contribute", f"2^{power}", ms)
                    if "pverify" in steps:
                        ms = self.run([self.bin(v), "powersoftau", "verify", out], ok_text="Powers of Tau Ok!")
                        if ms is not None:
                            self.record(rnd, v, "pverify", f"2^{power}", ms)
                    if rnd == 1 and power == min(p for p, _ in self.a.contribute):
                        shutil.copy2(out, self.keep / v / f"contrib-{power}.ptau")
                out.unlink(missing_ok=True)
            if "beacon" in steps and rnd == 1:
                ms = self.run([self.bin(v), "powersoftau", "beacon", src, out, BEACON, "10",
                               "--name=arkab beacon", *b])
                if ms is not None:
                    self.record(rnd, v, "beacon", f"2^{power}", ms)
                    self.digest(rnd, v, f"ptau beacon 2^{power}", out)
                out.unlink(missing_ok=True)
        if "prepare" in steps:
            pb = ["--backend", self.a.prepare_backend]
            for power, src in self.a.prepare:
                out = self.tmp / f"prep-{v}-{power}.ptau"
                ms = self.run([self.bin(v), "powersoftau", "prepare", "phase2", src, out, *pb])
                if ms is not None:
                    self.record(rnd, v, "prepare", f"2^{power}", ms)
                    self.digest(rnd, v, f"prepare 2^{power}", out)
                    if rnd == 1 and power == min(p for p, _ in self.a.prepare):
                        shutil.copy2(out, self.keep / v / f"prep-{power}.ptau")
                out.unlink(missing_ok=True)

    def phase2(self, rnd, v, c, steps):
        art = Path(self.a.artifacts) / c
        r1cs = next(iter(sorted(art.glob("*.r1cs"))), None)
        power = domain_power(art / "circuit.zkey")
        ptau = self.ptau_for(power)
        f = self.out / "ptau-for.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        m[c] = ptau.name
        f.write_text(json.dumps(m, indent=1))
        b = ["--backend", self.a.ceremony_backend]
        init = self.tmp / f"{c}-{v}-0000.zkey"
        final = self.tmp / f"{c}-{v}-0001.zkey"
        ms = self.run([self.bin(v), "groth16", "setup", r1cs, ptau, init, *b], timed="setup" in steps)
        if ms is None:
            return
        if "setup" in steps:
            self.record(rnd, v, "setup", c, ms, ptau=ptau.name)
        self.digest(rnd, v, f"setup {c}", init)
        if "zcontribute" in steps:
            ms = self.run([self.bin(v), "zkey", "contribute", init, final, "--name=arkab",
                           "--entropy=arkab bench entropy", *b])
            if ms is not None:
                self.record(rnd, v, "zcontribute", c, ms)
                if "zverify" in steps:
                    ms = self.run([self.bin(v), "zkey", "verify", "init", init, ptau, final], ok_text="ZKey Ok!")
                    if ms is not None:
                        self.record(rnd, v, "zverify", c, ms)
                if rnd == 1 and c in self.a.snarkjs_zkeys:
                    d = self.keep / v / "zkeys"
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(final, d / f"{c}.zkey")
        if "zbeacon" in steps and rnd == 1:
            if self.run([self.bin(v), "zkey", "beacon", init, final, BEACON, "10", "--name=arkab beacon", *b],
                        timed=False) is not None:
                self.digest(rnd, v, f"zkey beacon {c}", final)
        init.unlink(missing_ok=True)
        final.unlink(missing_ok=True)

    def ptau_for(self, power):
        for p in range(power, 29):
            f = Path(self.a.ptau_dir) / f"ppot_0080_{p:02d}.ptau"
            if f.exists():
                return f
        raise FileNotFoundError(f"no ppot_0080_NN.ptau at or above power {power} in {self.a.ptau_dir}")

    # -- driver ------------------------------------------------------------------------

    def go(self):
        a = self.a
        steps = set(a.steps)
        say(f"variants {a.variants}, backend {a.backend}, rounds {a.rounds}, steps {sorted(steps)}")
        if "proof" in steps or "verify" in steps:
            for v in a.variants:
                for c in a.circuits:
                    self.proof(v, c)
        if "trace" in steps:
            for v in a.variants:
                self.trace(v, a.circuits[0])
        for rnd in range(1, a.rounds + 1):
            order = list(a.variants) if rnd % 2 else list(reversed(a.variants))
            say(f"round {rnd}: {' then '.join(order)}")
            if "prove" in steps:
                for c in a.circuits:
                    for v in order:
                        say(f"  r{rnd} prove {c} {v}")
                        self.prove(rnd, v, c)
            if "verify" in steps:
                for v in order:
                    self.verify(rnd, v, a.circuits)
            if steps & {"contribute", "pverify", "prepare", "beacon"}:
                for v in order:
                    say(f"  r{rnd} phase 1 {v}")
                    self.phase1(rnd, v, steps)
            if steps & {"setup", "zcontribute", "zverify", "zbeacon"}:
                for c in a.phase2:
                    for v in order:
                        say(f"  r{rnd} phase 2 {c} {v}")
                        self.phase2(rnd, v, c, steps)
        say(f"done, {len(self.rows)} rows, {len(self.failures)} failures")
        (self.out / "DONE").write_text(f"{len(self.failures)}\n")


def size_list(xs):
    out = []
    for x in xs:
        p, f = x.split(":", 1)
        out.append((int(p), f))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--variants", nargs=2, default=["stock", "perf"])
    p.add_argument("--backend", default="cpu", help="prove backend")
    p.add_argument("--ceremony-backend", default=None, help="contribute/beacon/setup backend (default: --backend)")
    p.add_argument("--prepare-backend", default=None, help="ptau prepare backend (default: --backend)")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--ptau-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--verify-reps", type=int, default=500)
    p.add_argument("--circuits", nargs="+", default=CIRCUITS)
    p.add_argument("--phase2", nargs="*", default=CIRCUITS)
    p.add_argument("--contribute", nargs="*", default=["13:local_13.ptau", "19:local_19.ptau"],
                   help="POWER:FILE, FILE relative to --ptau-dir")
    p.add_argument("--prepare", nargs="*", default=["13:local_13.ptau", "18:ppot_0080_18.ptau"])
    p.add_argument("--snarkjs-zkeys", nargs="*", default=["js_2x2_d32"],
                   help="phase-2 circuits whose round-1 contributed key is kept for snarkjs")
    p.add_argument("--steps", nargs="+",
                   default=["proof", "trace", "prove", "verify", "contribute", "pverify", "beacon", "prepare",
                            "setup", "zcontribute", "zverify", "zbeacon"])
    p.add_argument("--max-load", type=float, default=None)
    a = p.parse_args()
    a.ceremony_backend = a.ceremony_backend or a.backend
    a.prepare_backend = a.prepare_backend or a.backend
    a.contribute = [(n, Path(a.ptau_dir) / f) for n, f in size_list(a.contribute)]
    a.prepare = [(n, Path(a.ptau_dir) / f) for n, f in size_list(a.prepare)]
    Runner(a).go()


if __name__ == "__main__":
    sys.exit(main())
