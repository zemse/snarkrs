#!/usr/bin/env bash
# The CUDA device gate on a T4, from this Mac: cross-build the CUDA test binaries and the CLI,
# ship them with the fixtures, and run cuda-gate-guest.sh on the box.
#
# This never calls aws. The box comes from t4.sh, which lives outside the repo: it starts the
# instance, holds the GPU lock while the gate runs, and stops the box at the last release.
# T4_BOX (and anything else t4.sh reads) passes through, so a lane gates on its own box.
#
# The gate: a warm-up that compiles every kernel unit into the box's caches (so a cold box
# or a .cu edit costs time, never a step's cap), the lib's tests and its ignored device
# tests, constant_work_legacy_override_gate in its own process with
# G16_CUDA_MSM_LEGACY_ACC=1, adversarial_gpu and pipeline_gpu with their ignored tests,
# field_gpu, gnark_gpu, and CLI prove and verify of tiny_mul and sha256, default and
# --constant-work, and that a proof refused for VRAM (G16_CUDA_VRAM_LIMIT=1) goes straight
# to the CPU. fft::tests::every_variant_gives_the_same_points is about 60
# minutes cold, so it runs only with --every-variant.
#
# usage: cuda-gate.sh [--every-variant] [--no-build]
#   env: G16_T4            path to t4.sh (required)
#        G16_T4_LANE       lease name on the box (default gate)
#        G16_ARTIFACTS     artifact root with tiny_mul and sha256 (default bench/artifacts)
#        CARGO_TARGET_DIR  (default target/cuda-gate)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
T4="${G16_T4:?set G16_T4 to the t4.sh helper}"
LANE="${G16_T4_LANE:-gate}"
ART="${G16_ARTIFACTS:-$HERE/bench/artifacts}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HERE/target/cuda-gate}"
TARGET="x86_64-unknown-linux-gnu.2.31"
GUEST_ARGS=()
BUILD=1
for a in "$@"; do
  case "$a" in
    --every-variant) GUEST_ARGS+=(--every-variant) ;;
    --no-build) BUILD=0 ;;
    *) echo "usage: cuda-gate.sh [--every-variant] [--no-build]" >&2; exit 2 ;;
  esac
done
[[ -x "$T4" ]] || { echo "G16_T4=$T4 is not executable" >&2; exit 2; }
for c in tiny_mul sha256; do
  for f in circuit.zkey circuit.wtns vkey.json; do
    [[ -f "$ART/$c/$f" ]] || { echo "missing $ART/$c/$f (set G16_ARTIFACTS)" >&2; exit 2; }
  done
done

SHIP="$CARGO_TARGET_DIR/ship"
JSON="$CARGO_TARGET_DIR/build.json"
cd "$HERE"
if (( BUILD )); then
  echo "==> cross-compiling the CUDA tests and the CLI for $TARGET"
  mkdir -p "$CARGO_TARGET_DIR"
  cargo zigbuild --release -p snarkrs-cuda --features cuda --tests --target "$TARGET" \
    --message-format=json > "$JSON"
  cargo zigbuild --release -p snarkrs --features cuda --bin snarkrs --target "$TARGET" \
    --message-format=json >> "$JSON"
fi

rm -rf "$SHIP"; mkdir -p "$SHIP/bin" "$SHIP/artifacts"
# The newest executable of each wanted target: the lib's test harness, the four device test
# binaries, and the CLI.
python3 - "$JSON" "$SHIP/bin" <<'PY'
import json, shutil, sys
want = {"lib", "adversarial_gpu", "pipeline_gpu", "field_gpu", "gnark_gpu", "snarkrs"}
found = {}
for line in open(sys.argv[1]):
    m = json.loads(line) if line.startswith("{") else {}
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    t = m["target"]
    name = "lib" if "lib" in t["kind"] and m["profile"]["test"] else t["name"]
    if name in want and (m["profile"]["test"] or name == "snarkrs"):
        found[name] = m["executable"]
missing = want - found.keys()
if missing:
    sys.exit(f"no executable for {sorted(missing)} in the build output")
for name, exe in found.items():
    shutil.copy2(exe, f"{sys.argv[2]}/{name}")
PY
file "$SHIP/bin/snarkrs" | grep -q 'ELF 64-bit.*x86-64' || { echo "not an x86-64 ELF, refusing to ship" >&2; exit 1; }
for c in tiny_mul sha256; do
  mkdir -p "$SHIP/artifacts/$c"
  cp -L "$ART/$c"/{circuit.zkey,circuit.wtns,vkey.json} "$SHIP/artifacts/$c/"
done
mkdir -p "$SHIP/gnark"
cp -RL test-support/gnark/{go.mod,go.sum,main.go,toy} "$SHIP/gnark/"
cp bench/aws/cuda-gate-guest.sh "$SHIP/"
echo "    $(du -sh "$SHIP" | cut -f1) to ship, $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (dirty)')"

# Released on every exit, so the box stops as soon as the gate is done or fails.
trap '"$T4" release "$LANE" || true' EXIT
"$T4" run "$LANE" "$SHIP" -- bash cuda-gate-guest.sh ${GUEST_ARGS[@]+"${GUEST_ARGS[@]}"}
