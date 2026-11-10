#!/usr/bin/env bash
# Runs ON the GPU box, from the directory it was shipped in: warm CUDA latency and host CPU
# time per G16_CUDA_SYNC mode, interleaved, for the blocking-sync A/B.
#
# Every round runs every circuit once per mode, the modes in a rotated order each round, so
# drift on the box spreads over all of them. `auto` leaves G16_CUDA_SYNC unset (the driver's
# default). Each run is one `snarkrs bench --mode warm` under G16_BENCH_CPU=1, which prints
# one `cpu-usage:` line a proof. bench/aws/sync-ab-report.py turns out/ into the tables.
#
# usage: sync-ab-guest.sh <var|cw> <rounds> <reps> <modes,comma,separated> <circuit[:reps]>...
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
BIN=$HERE/bin/snarkrs
work=$1 rounds=$2 reps=$3
IFS=, read -r -a modes <<< "$4"
shift 4
OUT=$HERE/out/sync-$work
mkdir -p "$OUT"
cw=(); [[ "$work" == cw ]] && cw=(--constant-work)
"$BIN" --version > "$OUT/host.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$OUT/host.txt"
echo "nproc $(nproc)" >> "$OUT/host.txt"

rc=0
for ((r = 1; r <= rounds; r++)); do
  echo "=== round $r ($(date -u +%H:%M:%SZ))"
  for spec in "$@"; do
    IFS=: read -r circuit n <<< "$spec"
    for ((k = 0; k < ${#modes[@]}; k++)); do
      mode=${modes[$(((k + r) % ${#modes[@]}))]}
      sync=(); [[ "$mode" != auto ]] && sync=(G16_CUDA_SYNC="$mode")
      base=$OUT/$circuit-$mode-r$r
      env "${sync[@]}" G16_BENCH_CPU=1 "$BIN" bench --artifacts "$HERE/artifacts" \
        --variant "$circuit" --backend cuda --mode warm --reps "${n:-$reps}" --threads 8 \
        "${cw[@]}" --csv "$base.csv" < /dev/null > "$base.out" 2> "$base.err" \
        || { rc=1; echo "FAILED $circuit $mode r$r"; tail -3 "$base.err"; }
      grep -h median "$base.out" | sed "s/^/$mode r$r /"
    done
  done
done
echo "sync-ab done ($(date -u +%H:%M:%SZ)) rc $rc"
exit $rc
