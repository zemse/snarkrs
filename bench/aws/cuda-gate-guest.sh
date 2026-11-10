#!/usr/bin/env bash
# Runs ON the GPU box, from the directory cuda-gate.sh shipped: the CUDA device gate.
#
# Every test binary and the CLI were cross-built on the Mac, so nothing here compiles Rust.
# The first run on a fresh box, or after a .cu/.cuh edit, still pays NVRTC and the driver
# JIT for each kernel unit (six to eleven minutes each for the MSM, constant-work MSM and
# FFT units on a T4). The warm step pays it, every unit at once, so the capped steps after
# it always load warm.
#
# usage: cuda-gate-guest.sh [--every-variant]
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
export G16_ARTIFACTS=$HERE/artifacts G16_GNARK_HARNESS=$HERE/gnark RUST_BACKTRACE=1
BIN=$HERE/bin
OUT=$HERE/out
rm -rf "$OUT"; mkdir -p "$OUT"
EVERY_VARIANT=0
[[ "${1:-}" == --every-variant ]] && EVERY_VARIANT=1
FAILED=()

log() { echo; echo "=== $* ($(date -u +%H:%M:%SZ))"; }

# step <name> <minutes> <cmd...>: run under a timeout, keep the whole output in out/<name>.txt,
# print the test lines and the result line, and record a failure on a non-zero exit.
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

# Every kernel unit compiled into the PTX and driver caches, outside the steps below. Cold
# on a g4dn.xlarge this is 21 minutes, the five units at once (26.5 one after another);
# warm, a few seconds.
step warm 30 env G16_CUDA_TIME=1 "$BIN/lib" --ignored --exact --test-threads=1 --nocapture \
  kernels::tests::every_unit_compiles_on_the_device
grep -E "^unit |^every unit" "$OUT/warm.txt" | sed 's/^/  /'

# The lib's device tests, without the two that need their own process or an hour: the
# legacy accumulator gate reads G16_CUDA_MSM_LEGACY_ACC once per process, and the FFT
# variant sweep is about 60 minutes cold.
step lib 10 env G16_REQUIRE_TESTS=1 "$BIN/lib" --test-threads=1
step lib-ignored 20 env G16_REQUIRE_TESTS=1 G16_CUDA_MSM_LEGACY_ACC=0 "$BIN/lib" --ignored \
  --test-threads=1 --nocapture \
  --skip constant_work_legacy_override_gate --skip every_variant_gives_the_same_points \
  --skip every_unit_compiles_on_the_device
step lib-legacy 5 env G16_CUDA_MSM_LEGACY_ACC=1 "$BIN/lib" --ignored --exact \
  --test-threads=1 --nocapture msm::tests::constant_work_legacy_override_gate
if (( EVERY_VARIANT )); then
  step lib-every-variant 90 "$BIN/lib" --ignored --exact --test-threads=1 --nocapture \
    fft::tests::every_variant_gives_the_same_points
fi

step adversarial 15 env G16_REQUIRE_TESTS=1 "$BIN/adversarial_gpu" --include-ignored \
  --test-threads=1 --nocapture
step pipeline 15 env G16_REQUIRE_TESTS=1 G16_CUDA_REQUIRED_FIXTURES=tiny_mul,sha256 \
  "$BIN/pipeline_gpu" --include-ignored --test-threads=1 --nocapture
step field 10 env G16_REQUIRE_TESTS=1 "$BIN/field_gpu" --test-threads=1
# gnark's own verifier is `go run` in the harness; without Go that half skips, so coverage is
# required only when Go is there to give it.
if command -v go >/dev/null; then
  step gnark 10 env G16_REQUIRE_TESTS=1 "$BIN/gnark_gpu" --test-threads=1 --nocapture
else
  step gnark 10 "$BIN/gnark_gpu" --test-threads=1 --nocapture
  grep "SKIPPED" "$OUT/gnark.txt" | grep -qv "no \`go\` on PATH" && FAILED+=("gnark-skip")
fi

# The CLI on CUDA with the fallback off, so a proof the GPU did not make cannot pass.
log "cli"
for c in tiny_mul sha256; do
  for cw in "" --constant-work; do
    tag=$c${cw:+-cw}
    "$BIN/snarkrs" groth16 prove "$G16_ARTIFACTS/$c/circuit.zkey" "$G16_ARTIFACTS/$c/circuit.wtns" \
      "$OUT/$tag.proof.json" "$OUT/$tag.public.json" --backend cuda --fallback false $cw \
      > "$OUT/$tag.prove.txt" 2>&1
    p=$?
    "$BIN/snarkrs" groth16 verify "$G16_ARTIFACTS/$c/vkey.json" "$OUT/$tag.public.json" \
      "$OUT/$tag.proof.json" > "$OUT/$tag.verify.txt" 2>&1
    v=$?
    echo "  $tag: prove rc=$p verify rc=$v $(tail -1 "$OUT/$tag.verify.txt")"
    (( p == 0 && v == 0 )) || FAILED+=("cli-$tag")
  done
done

# A budget refusal goes straight to the CPU and blames no device.
log "cli vram refusal"
G16_CUDA_VRAM_LIMIT=1 "$BIN/snarkrs" groth16 prove "$G16_ARTIFACTS/sha256/circuit.zkey" \
  "$G16_ARTIFACTS/sha256/circuit.wtns" "$OUT/refused.proof.json" "$OUT/refused.public.json" \
  --backend cuda > "$OUT/refused.prove.txt" 2>&1
p=$?
"$BIN/snarkrs" groth16 verify "$G16_ARTIFACTS/sha256/vkey.json" "$OUT/refused.public.json" \
  "$OUT/refused.proof.json" > "$OUT/refused.verify.txt" 2>&1
v=$?
grep "warning" "$OUT/refused.prove.txt" | sed 's/^/  /'
echo "  refused: prove rc=$p verify rc=$v $(tail -1 "$OUT/refused.verify.txt")"
(( p == 0 && v == 0 )) || FAILED+=("cli-refused")
[[ $(grep -c "warning" "$OUT/refused.prove.txt") == 1 ]] || FAILED+=("cli-refused-warnings")
grep -q "suspect\|proving again" "$OUT/refused.prove.txt" && FAILED+=("cli-refused-retry")

log "result"
if (( ${#FAILED[@]} )); then echo "GATE FAILED: ${FAILED[*]}"; exit 1; fi
echo "GATE PASSED"
