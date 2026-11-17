#!/usr/bin/env bash
# The CUDA device gate on a T4, from this Mac: cross-build the CUDA test binaries and the CLI,
# ship them with the fixtures, and run cuda-gate-guest.sh on the box.
#
# t4.sh uses an already-running, independently deadline-armed instance. Its detached job
# supervisor holds the guest GPU lock; exact-token cleanup stops the last released lease.
# A disconnect leaves the batch and lease intact for resume, never an unconditional release.
#
# The gate: a warm-up that compiles every kernel unit into the box's caches (so a cold box
# or a .cu edit costs time, never a step's cap), the lib's tests and its ignored device
# tests, constant_work_legacy_override_gate in its own process with
# G16_CUDA_MSM_LEGACY_ACC=1, adversarial_gpu and pipeline_gpu with their ignored tests,
# field_gpu, gnark_gpu, and CLI prove and verify of tiny_mul and sha256, default and
# --constant-work, and that a proof refused for VRAM (G16_CUDA_VRAM_LIMIT=1) goes straight
# to the CPU. fft::tests::every_variant_gives_the_same_points is about 60
# minutes cold, so it runs only with --every-variant. Then the ceremony MSM
# (cuda-ceremony-guest.sh): its device tests, and `groth16 setup` and `zkey verify r1cs` on
# cuda against cpu for tiny_mul and js_2x2_d32 on ppot_0080_15. --ceremony-large adds
# anon-aadhaar on ppot_0080_21 (2.6 GB more to ship); --ceremony-only runs the ceremony
# part alone and needs no gnark verifier.
#
# usage: cuda-gate.sh [--every-variant] [--ceremony-only] [--ceremony-large] [--no-build]
#   env: G16_T4            path to t4.sh (default bench/aws/t4.sh)
#        G16_T4_LANE       lease name on the box (default gate)
#        G16_ARTIFACTS     artifact root with tiny_mul and sha256 (default bench/artifacts)
#        G16_PTAU          ptau root with ppot_0080_15.ptau (default bench/ptau)
#        CARGO_TARGET_DIR  (default target/cuda-gate)
#        G16_GNARK_VERIFIER prebuilt Linux amd64 harness (default $CARGO_TARGET_DIR/gnark-verifier)
# Go builds are local, offline, and require Go >= 1.24 and cached modules. --no-build
# reuses the verifier too. An explicit G16_GNARK_VERIFIER is never rebuilt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
T4="${G16_T4:-$HERE/bench/aws/t4.sh}"
LANE="${G16_T4_LANE:-gate}"
ART="${G16_ARTIFACTS:-$HERE/bench/artifacts}"
PTAU="${G16_PTAU:-$HERE/bench/ptau}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HERE/target/cuda-gate}"
TARGET="x86_64-unknown-linux-gnu.2.31"
GUEST_ARGS=()
BUILD=1
CEREMONY_ONLY=0
CEREMONY_LARGE=0
for a in "$@"; do
  case "$a" in
    --every-variant) GUEST_ARGS+=(--every-variant) ;;
    --ceremony-only) CEREMONY_ONLY=1 ;;
    --ceremony-large) CEREMONY_LARGE=1 ;;
    --no-build) BUILD=0 ;;
    *) echo "usage: cuda-gate.sh [--every-variant] [--ceremony-only] [--ceremony-large] [--no-build]" >&2
       exit 2 ;;
  esac
done
[[ -x "$T4" ]] || { echo "G16_T4=$T4 is not executable" >&2; exit 2; }
if (( ! CEREMONY_ONLY )); then
  for c in tiny_mul sha256; do
    for f in circuit.zkey circuit.wtns vkey.json; do
      [[ -f "$ART/$c/$f" ]] || { echo "missing $ART/$c/$f (set G16_ARTIFACTS)" >&2; exit 2; }
    done
  done
fi
# The ceremony fixtures: circuit, ptau. Public only; nothing else is shipped for them.
CEREMONY=(tiny_mul:ppot_0080_15.ptau js_2x2_d32:ppot_0080_15.ptau)
(( CEREMONY_LARGE )) && CEREMONY+=(anon-aadhaar:ppot_0080_21.ptau)
for fx in "${CEREMONY[@]}"; do
  [[ -f "$ART/${fx%%:*}/circuit.r1cs" ]] || { echo "missing $ART/${fx%%:*}/circuit.r1cs" >&2; exit 2; }
  [[ -f "$PTAU/${fx#*:}" ]] || { echo "missing $PTAU/${fx#*:} (set G16_PTAU)" >&2; exit 2; }
done

SHIP="$CARGO_TARGET_DIR/ship"
JSON="$CARGO_TARGET_DIR/build.json"
cd "$HERE"
VERIFIER="${G16_GNARK_VERIFIER:-$CARGO_TARGET_DIR/gnark-verifier}"
if (( BUILD && ! CEREMONY_ONLY )) && [[ -z "${G16_GNARK_VERIFIER:-}" ]]; then
  echo "==> cross-compiling gnark's verifier locally for Linux amd64 (offline)"
  mkdir -p "$(dirname "$VERIFIER")"
  VERIFIER="$(cd "$(dirname "$VERIFIER")" && pwd)/$(basename "$VERIFIER")"
  (cd test-support/gnark && env CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOTOOLCHAIN=local \
    GOPROXY=off GOSUMDB=off go build -mod=readonly -o "$VERIFIER" .)
fi
if (( ! CEREMONY_ONLY )); then
  [[ -x "$VERIFIER" ]] || { echo "missing executable $VERIFIER (build locally or set G16_GNARK_VERIFIER)" >&2; exit 2; }
  file "$VERIFIER" | grep -q 'ELF 64-bit.*x86-64.*statically linked' || {
    echo "gnark verifier must be a static Linux x86-64 ELF" >&2; exit 2;
  }
fi
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
GUEST=cuda-ceremony-guest.sh
if (( ! CEREMONY_ONLY )); then
  GUEST=cuda-gate-guest.sh
  for c in tiny_mul sha256; do
    mkdir -p "$SHIP/artifacts/$c"
    cp -L "$ART/$c"/{circuit.zkey,circuit.wtns,vkey.json} "$SHIP/artifacts/$c/"
  done
  mkdir -p "$SHIP/gnark"
  cp -RL test-support/gnark/toy "$SHIP/gnark/"
  cp "$VERIFIER" "$SHIP/bin/gnark-verifier"
  cp bench/aws/cuda-gate-guest.sh "$SHIP/"
fi
mkdir -p "$SHIP/ptau"
for fx in "${CEREMONY[@]}"; do
  mkdir -p "$SHIP/artifacts/${fx%%:*}"
  cp -L "$ART/${fx%%:*}/circuit.r1cs" "$SHIP/artifacts/${fx%%:*}/"
  cp -L "$PTAU/${fx#*:}" "$SHIP/ptau/"
done
cp bench/aws/cuda-ceremony-guest.sh "$SHIP/"
echo "    $(du -sh "$SHIP" | cut -f1) to ship, $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (dirty)')"

# The runner releases its exact token only after terminal status and results are retrieved.
"$T4" run "$LANE" "$SHIP" -- bash -c \
  'set -euo pipefail; source "$HOME/g16-env.sh"; guest=$1; shift; exec bash "$guest" "$@"' \
  cuda-gate "$GUEST" ${GUEST_ARGS[@]+"${GUEST_ARGS[@]}"}
