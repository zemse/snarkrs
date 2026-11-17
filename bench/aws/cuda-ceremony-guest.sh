#!/usr/bin/env bash
# Runs ON the GPU box, from the directory cuda-gate.sh shipped: the CUDA ceremony MSM gate.
#
# cuda-gate-guest.sh sources this after the prover gate; `cuda-gate.sh --ceremony-only` runs
# it on its own. The adapter's device tests (ceremony::tests in the lib), then on every
# shipped circuit: `groth16 setup --backend cuda` byte-identical to `--backend cpu`, and
# `zkey verify r1cs --backend cuda` passing with `--check exact` and `--check random` on a
# key setup and one CPU contribution made. Each cuda command runs twice, with the shipped
# crossover and with G16_CUDA_CER_MSM_MIN=0, which puts every MSM setup and the random
# check make on the device; G16_CUDA_TIME=1 has the backend print where its calls went.
# The CPU verify of the same key is timed beside each, for the comparison.
#
# Fixtures are public only: artifacts/<circuit>/circuit.r1cs and ptau/ppot_0080_NN.ptau.
#
# usage: cuda-ceremony-guest.sh
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
BIN=$HERE/bin
OUT=$HERE/out
STANDALONE=0
if ! declare -F step > /dev/null; then
  STANDALONE=1
  export RUST_BACKTRACE=1
  rm -rf "$OUT"; mkdir -p "$OUT"
  FAILED=()
  log() { echo; echo "=== $* ($(date -u +%H:%M:%SZ))"; }
  # The same step as cuda-gate-guest.sh's.
  step() {
    local name=$1 min=$2 rc t0=$SECONDS
    shift 2
    log "$name"
    timeout "${min}m" "$@" > "$OUT/$name.txt" 2>&1
    rc=$?
    grep -E "^test |^test result|panicked|SKIPPED|gate passed" "$OUT/$name.txt" | grep -vE " \.\.\. (ok|ignored.*)$"
    echo "  $name: rc=$rc in $((SECONDS - t0)) s"
    (( rc == 0 )) || FAILED+=("$name")
  }
  log "device"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
fi

# Its own process, so a cold box pays the MSM unit's compile here, under this cap.
step ceremony-lib 30 env G16_REQUIRE_TESTS=1 G16_CUDA_TIME=1 "$BIN/lib" --ignored \
  --test-threads=1 --nocapture ceremony::tests::
grep -Eq '^test result: ok\. 5 passed; 0 failed;' "$OUT/ceremony-lib.txt" \
  || FAILED+=("ceremony-lib-count")

# Device calls the backend reported in <file>, from its G16_CUDA_TIME line.
device_calls() {
  grep -oE 'cuda ceremony msm: [0-9]+ calls' "$1" | grep -oE '[0-9]+' | tail -1
}

# timed <out-file> <cmd...>: run under the cap, print nothing, leave rc in RC and seconds in DT.
timed() {
  local out=$1 t0=$SECONDS
  shift
  timeout 60m "$@" > "$out" 2>&1
  RC=$?
  DT=$((SECONDS - t0))
}

# ceremony_cli <circuit> <ptau file>
ceremony_cli() {
  local c=$1 ptau=$HERE/ptau/$2 d=$OUT/ceremony/$1 r1cs=$HERE/artifacts/$1/circuit.r1cs
  mkdir -p "$d"
  log "ceremony cli: $c against $2"
  timed "$d/setup-cpu.txt" "$BIN/snarkrs" groth16 setup "$r1cs" "$ptau" "$d/cpu.zkey" \
    --backend cpu
  echo "  setup cpu: rc=$RC in $DT s"
  (( RC == 0 )) || { FAILED+=("$c-setup-cpu"); return; }

  local min tag calls
  for min in shipped 0; do
    local env=(G16_CUDA_TIME=1)
    [[ $min == 0 ]] && env+=(G16_CUDA_CER_MSM_MIN=0)
    tag=min-$min
    timed "$d/setup-cuda-$tag.txt" env "${env[@]}" "$BIN/snarkrs" groth16 setup "$r1cs" \
      "$ptau" "$d/cuda-$tag.zkey" --backend cuda
    calls=$(device_calls "$d/setup-cuda-$tag.txt")
    if cmp -s "$d/cpu.zkey" "$d/cuda-$tag.zkey"; then same=identical; else same=DIFFERENT; fi
    echo "  setup cuda ($tag): rc=$RC in $DT s, ${calls:-no} device calls, $same to cpu"
    (( RC == 0 )) || FAILED+=("$c-setup-cuda-$tag")
    [[ $same == identical ]] || FAILED+=("$c-setup-cuda-$tag-bytes")
    rm -f "$d/cuda-$tag.zkey"
  done

  timed "$d/contribute.txt" "$BIN/snarkrs" zkey contribute "$d/cpu.zkey" "$d/final.zkey" \
    --name=gate -e=cuda-ceremony-gate --backend cpu
  echo "  contribute cpu: rc=$RC in $DT s"
  rm -f "$d/cpu.zkey"
  (( RC == 0 )) || { FAILED+=("$c-contribute"); return; }

  local check backend
  for check in exact random; do
    timed "$d/verify-cpu-$check.txt" "$BIN/snarkrs" zkey verify r1cs "$r1cs" "$ptau" \
      "$d/final.zkey" --backend cpu --check "$check" -v
    echo "  verify $check cpu: rc=$RC in $DT s"
    (( RC == 0 )) || FAILED+=("$c-verify-$check-cpu")
    for min in shipped 0; do
      local env=(G16_CUDA_TIME=1)
      [[ $min == 0 ]] && env+=(G16_CUDA_CER_MSM_MIN=0)
      tag=min-$min
      timed "$d/verify-cuda-$check-$tag.txt" env "${env[@]}" "$BIN/snarkrs" zkey verify r1cs \
        "$r1cs" "$ptau" "$d/final.zkey" --backend cuda --check "$check" -v
      calls=$(device_calls "$d/verify-cuda-$check-$tag.txt")
      echo "  verify $check cuda ($tag): rc=$RC in $DT s, ${calls:-no} device calls"
      (( RC == 0 )) || FAILED+=("$c-verify-$check-cuda-$tag")
      grep -q "recomputed on cuda" "$d/verify-cuda-$check-$tag.txt" \
        || FAILED+=("$c-verify-$check-cuda-$tag-backend")
      # With the crossover at zero the random check's combination is on the device
      # whatever the circuit, so a run that reports no device call did not test it.
      if [[ $check == random && $min == 0 ]] && (( ${calls:-0} == 0 )); then
        FAILED+=("$c-verify-random-cuda-$tag-no-device")
      fi
    done
  done
  rm -f "$d/final.zkey"
}

for c in tiny_mul js_2x2_d32; do
  ceremony_cli "$c" ppot_0080_15.ptau
done
# Shipped only by `cuda-gate.sh --ceremony-large`: cirPower 21, 3.7 GB resident at setup.
if [[ -f "$HERE/artifacts/anon-aadhaar/circuit.r1cs" && -f "$HERE/ptau/ppot_0080_21.ptau" ]]; then
  ceremony_cli anon-aadhaar ppot_0080_21.ptau
fi

if (( STANDALONE )); then
  log "result"
  if (( ${#FAILED[@]} )); then echo "GATE FAILED: ${FAILED[*]}"; exit 1; fi
  echo "GATE PASSED"
fi
