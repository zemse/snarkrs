#!/usr/bin/env bash
# Validate and measure round 3's CUDA changes on a rented GPU, in one pass.
#
# Runs ON the box. Everything it needs was cross-compiled on the Mac and shipped, so this
# never invokes a compiler: see run-gpu-bench-prebuilt.sh for why that matters beyond cost.
#
# Two changes are under test and each has its own lever, because a single combined number
# cannot say which one paid:
#   G16_CUDA_FF_PTX=1        opt-in PTX carry-chain Montgomery multiply for Fr and Fq
#   G16_CUDA_WITNESS_REUSE=0 forces the OLD path, uploading the witness twice per proof
# So for the witness change, REUSE=0 is the before-arm and the default is the after-arm.
#
# The kernel cache is warmed before every timed section, and warmed AGAIN whenever a lever
# changes the kernel source, because NVRTC output is cached per source text in
# ~/.nv/ComputeCache. A first compile is ~113 s against ~0.18 s warm. Leaving that inside a
# timed region puts a two-minute outlier in exactly one rep of exactly one variant.
set -uo pipefail
cd ~/g16
export G16_ARTIFACTS=~/g16/bench/artifacts
BIN=./target/release/g16
OUT=~/g16/out; mkdir -p "$OUT"
log() { echo; echo "===== $* ====="; }

warm() { # warm <env assignments...>
  env "$@" $BIN prove --zkey bench/artifacts/tiny_mul/circuit.zkey \
    --witness bench/artifacts/tiny_mul/circuit.wtns \
    --proof /tmp/w.json --public /tmp/wp.json --backend cuda >/dev/null 2>&1
}

log "device"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nproc | xargs echo "vcpu:"

# ---------------------------------------------------------------- correctness
# Default configuration first. These binaries iterate G16_ARTIFACTS; if it were unset or
# wrong they would pass having checked nothing, which is why the override panics on a bad
# path rather than falling back.
for t in field_gpu pipeline_gpu adversarial_gpu; do
  log "correctness: $t (default config)"
  timeout 1200 ./tests/$t --test-threads=1 2>&1 | tail -25
done

log "correctness: field_gpu + pipeline_gpu with PTX carry chains on"
warm G16_CUDA_FF_PTX=1
for t in field_gpu pipeline_gpu; do
  G16_CUDA_FF_PTX=1 timeout 1200 ./tests/$t --test-threads=1 2>&1 | tail -20
done

# ---------------------------------------------------------------- measurement
bench() { # bench <tag> <mode> <env...>
  local tag=$1 mode=$2; shift 2
  log "bench: $tag ($mode)"
  warm "$@"
  env "$@" $BIN bench --artifacts bench/artifacts --reps 10 --mode "$mode" \
    --backend cuda --csv "$OUT/cuda-$tag.csv" 2>&1 | grep -E "median|====" | tail -20
}

# CPU on this box too: the history has a cpu row for this machine and the round changed the
# CPU MSM, so it should be re-measured on the same hardware rather than inferred from the Mac.
log "bench: cpu (both) on this box"
$BIN bench --artifacts bench/artifacts --reps 10 --mode both --backend cpu \
  --csv "$OUT/cpu-after.csv" 2>&1 | grep -E "median" | tail -12

bench after      both G16_NOOP=1                      # current defaults
bench ptx        both G16_CUDA_FF_PTX=1               # + PTX Montgomery
bench noreuse    warm G16_CUDA_WITNESS_REUSE=0        # before-arm for the witness change
bench reuse      warm G16_NOOP=1                      # after-arm, same mode for comparison

# ---------------------------------------------------------------- window sweep
# The default window rests on one data point, and TASKS.md records a forced c=8 beating the
# automatic choice by 9% at 2^18. One point is not a tune, so sweep the range on the two
# circuits that dominate the cost curve.
log "window sweep: G16_CUDA_MSM_C 6..16"
warm G16_NOOP=1
for c in 6 7 8 9 10 11 12 13 14 15 16; do
  G16_CUDA_MSM_C=$c $BIN bench --artifacts bench/artifacts --reps 6 --mode warm \
    --backend cuda --variant js_8x8_d32 --variant js_16x16_d32 \
    --csv "$OUT/sweep-c$c.csv" >/dev/null 2>&1
  echo "c=$c  $(python3 - "$OUT/sweep-c$c.csv" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for v in ("js_8x8_d32","js_16x16_d32"):
    ms=[float(r["ms"]) for r in rows if r["variant"]==v and r["mode"]=="warm"]
    if ms: print(f"{v} {statistics.median(ms):8.1f}ms", end="   ")
PY
)"
done

log "done. results in $OUT"
ls -la "$OUT"
