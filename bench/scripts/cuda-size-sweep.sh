#!/usr/bin/env bash
# The CUDA power-of-two size sweep (BUG-04): one process per domain and phase, running
# `size_sweep_point` from crates/cuda/tests/size_sweep_gpu.rs on a synthetic key. The `cuda`
# phase proves on the device and records prepare and proof times, the MSM runs, the free
# device memory through each MSM and the planned batch; the `cpu` phase proves the same key
# on the CPU and fails if the proof differs. Each process is sampled by nvidia-smi every
# 100 ms and its exit status kept, so a host OOM kill (137) or a hang (124) is a row too.
#
# usage, on the GPU box, with the zigbuilt test binary next to it:
#   cuda-size-sweep.sh <size_sweep_gpu binary> <out dir> <log2 domain>...
# e.g.
#   cuda-size-sweep.sh ./size_sweep_gpu-0123abcd out 16 17 18 19 20
#
# PHASES (default "cuda cpu") picks the phases, TIMEOUT (default 1500) bounds each process
# in seconds, and any G16_SWEEP_* in the environment reaches the test. Rows go to
# <out dir>/sweep.tsv, the records to <out dir>/2^<log>-<phase>.json, logs beside them.
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <test binary> <out dir> <log2 domain>..." >&2; exit 2; }
BIN=$(realpath "$1") OUT=$2; shift 2
PHASES=${PHASES:-cuda cpu} TIMEOUT=${TIMEOUT:-1500}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
[ -f "$OUT/sweep.tsv" ] || printf 'log\tphase\texit\tsmi_peak_mib\twall_ms\trecord\n' >"$OUT/sweep.tsv"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT/device.txt"
grep -E 'MemTotal|MemAvailable' /proc/meminfo | tee -a "$OUT/device.txt"
nproc | sed 's/^/cpus: /' | tee -a "$OUT/device.txt"

for log in "$@"; do
  for phase in $PHASES; do
    tag="2^$log-$phase"
    smi="$OUT/$tag-vram-mib.csv"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 >"$smi" &
    pid=$!
    rm -f "$OUT/$tag.json"
    t0=$(date +%s%N)
    G16_SWEEP_LOG=$log G16_SWEEP_PHASE=$phase G16_SWEEP_OUT=$OUT \
      timeout --kill-after=30 "$TIMEOUT" "$BIN" --ignored --exact size_sweep_point \
      --nocapture --test-threads=1 >"$OUT/$tag.log" 2>&1
    rc=$?
    wall=$(( ($(date +%s%N) - t0) / 1000000 ))
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    peak=$(sort -n "$smi" | tail -1)
    rec=$(cat "$OUT/$tag.json" 2>/dev/null || echo '{}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$log" "$phase" "$rc" "${peak:-}" "$wall" "$rec" \
      | tee -a "$OUT/sweep.tsv"
    # A cuda point that did not prove leaves nothing for the cpu phase to compare.
    if [ "$phase" = cuda ] && ! grep -q '"outcome":"proved"' "$OUT/$tag.json" 2>/dev/null; then
      break
    fi
  done
done
