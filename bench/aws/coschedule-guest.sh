#!/usr/bin/env bash
# Runs ON the GPU box, from the directory it was shipped in: CPU and GPU proof streams side by
# side, for the co-scheduling measurement (CARRY-03) and the OPT-20 ceiling.
#
# Every line of the plan file is one experiment: a tag, a circuit, `var` or `cw`
# (--constant-work), then one stream spec per concurrent process,
# `backend:threads:reps[:cpulist]`. All streams of a line start together; each is a
# `snarkrs bench --mode warm` under G16_BENCH_CPU=1, so it prints one `cpu-usage:` line a
# proof with wall-clock start/end, prove time and the process's CPU time.
# bench/aws/coschedule-report.py turns out/ into the tables.
#
# usage: coschedule-guest.sh <plan-file>
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
BIN=$HERE/bin/snarkrs
OUT=$HERE/out
PLAN=$1
mkdir -p "$OUT"

{
  echo "nproc $(nproc)"
  lscpu | grep -E '^(Model name|Thread|Core|Socket)'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$BIN" --version
} > "$OUT/host.txt" 2>&1
cat "$OUT/host.txt"

# Whole-box busy time, from /proc/stat's first line: everything but idle and iowait, in ticks.
busy() { awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $5+$6}' /proc/stat; }

while read -r tag circuit work specs; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  dir=$OUT/$tag
  rm -rf "$dir"; mkdir -p "$dir"
  cw=(); [[ "$work" == cw ]] && cw=(--constant-work)
  echo "=== $tag $circuit $work: $specs ($(date -u +%H:%M:%SZ))"
  read -r b0 i0 < <(busy)
  t0=$(date +%s.%N)
  pids=(); i=0
  for spec in $specs; do
    IFS=: read -r backend threads reps cpus <<< "$spec"
    pin=(); [[ -n "${cpus:-}" ]] && pin=(taskset -c "$cpus")
    G16_BENCH_CPU=1 "${pin[@]}" "$BIN" bench --artifacts "$HERE/artifacts" --variant "$circuit" \
      --backend "$backend" --mode warm --reps "$reps" --threads "$threads" "${cw[@]}" \
      --csv "$dir/$i-$backend.csv" < /dev/null > "$dir/$i-$backend.out" 2> "$dir/$i-$backend.err" &
    pids+=($!)
    echo "$i $spec" >> "$dir/streams.txt"
    i=$((i + 1))
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  read -r b1 i1 < <(busy)
  t1=$(date +%s.%N)
  echo "busy_ticks $((b1 - b0)) idle_ticks $((i1 - i0)) wall_s $(awk "BEGIN{print $t1 - $t0}") rc $rc" > "$dir/box.txt"
  cat "$dir/box.txt"
  grep -h 'median\|stages' "$dir"/*.out
  (( rc == 0 )) || { echo "FAILED $tag"; tail -5 "$dir"/*.err; }
done < "$PLAN"
echo "coschedule done ($(date -u +%H:%M:%SZ))"
