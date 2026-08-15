#!/usr/bin/env bash
# Wall-clock decomposition across the circuit ladder.
#
# Two different questions, two different tools, and conflating them is the usual way a
# prover benchmark ends up misleading:
#
#   1. hyperfine times the whole `g16 prove` PROCESS. That is what a CLI user pays:
#      exec, dyld, zkey parse, witness parse, prepare, prove, JSON write, exit. It is the
#      only number here that includes the parts of the cost nobody instruments.
#   2. `g16 bench` times the inside: cold (parse + prepare + prove, per rep) and warm
#      (prove only, key resident), with the five-stage split from StageTimings.
#
# Reporting only (2) flatters us by the whole parse; reporting only (1) hides which stage
# owns the time. Both get written, and the report is expected to quote the gap.
#
# Usage: bench/scripts/profile-wallclock.sh [reps] [warmup]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
reps="${1:-10}"
warmup="${2:-2}"
art="$root/bench/artifacts"
out="$root/bench/results/profiling"
bin="$root/target/release/g16"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

[ -x "$bin" ] || { echo "build first: cargo build --release" >&2; exit 1; }
mkdir -p "$out"

# Ladder in constraint order. tiny_mul has no manifest row (2 constraints, from the
# r1cs info file) but it is the left-hand anchor of the curve, so it stays in.
variants=(tiny_mul js_1x1_d8 js_2x2_d16 js_2x2_d32 js_8x8_d32 js_16x16_d32)

echo "== hyperfine: whole-process cold prove, $reps runs, $warmup warmup =="
for v in "${variants[@]}"; do
  d="$art/$v"
  [ -f "$d/circuit.zkey" ] || { echo "skip $v: no circuit.zkey"; continue; }
  hyperfine \
    --warmup "$warmup" --runs "$reps" \
    --command-name "$v" \
    --export-json "$out/hyperfine-$v.json" \
    --export-markdown "$out/hyperfine-$v.md" \
    "$bin prove --zkey $d/circuit.zkey --witness $d/circuit.wtns --proof $tmp/p.json --public $tmp/pub.json"
done

# The process floor: what `g16` costs before it has done any proving at all. Subtracting
# this from the tiny_mul number is what separates "our prover is slow" from "exec and
# dyld are slow", and at 2 constraints those are the same order of magnitude.
echo "== hyperfine: process floor (--version, no proving at all) =="
hyperfine --warmup "$warmup" --runs 50 --command-name "process-floor" \
  --export-json "$out/hyperfine-process-floor.json" \
  --export-markdown "$out/hyperfine-process-floor.md" \
  "$bin --version"

echo "== g16 bench: cold and warm, in-process, with the stage split =="
"$bin" bench --artifacts "$art" --reps "$reps" --mode both --backend cpu \
  --csv "$out/stage-split.csv" | tee "$out/stage-split.txt"

echo
echo "wrote $out/{hyperfine-*.json,stage-split.csv,stage-split.txt}"
