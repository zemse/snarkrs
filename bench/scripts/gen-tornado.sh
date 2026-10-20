#!/usr/bin/env bash
# Rebuild bench/artifacts/tornado: Tornado Cash's withdraw circuit, Withdraw(20), a circom 1
# circuit of 28,275 constraints. These are BENCHMARK keys, never production.
#
# tornado-core pins circom 0.0.35 and tornadocash/circomlib c372f14. This compiles that
# circomlib with circom 0.5.x instead (CIRCOM1, checked by its --version), which needs
# tornado-circomlib-circom05.patch: Pedersen's Segment assigns windows[i].in after it has
# wired windows[i].base and .out, an order 0.0.35 tolerated and 0.5.x compiles into a wasm
# whose witness silently fails the r1cs. The patch feeds windows[i].in first and declares the
# sizes of the BASE and c_partial arrays. It is a fix for that old pin only: current iden3
# circomlib already feeds the window first and declares BASE[10][2].
#
# input.json is not rebuilt. Keep the existing $OUT/input.json, or pass INPUT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/artifacts/tornado"
VENDOR="$HERE/vendor"
PATCH="$HERE/scripts/tornado-circomlib-circom05.patch"
PTAU="${PTAU:-$HERE/ptau/local_19.ptau}"
INPUT="${INPUT:-$OUT/input.json}"
SNARKJS="${SNARKJS:-snarkjs}"
CORE_REV=1ef6a263ac6a0e476d063fcb269a9df65a1bd56a
CIRCOMLIB_REV=c372f14d324d57339c88451834bf2824e73bbdbc

: "${CIRCOM1:?set CIRCOM1 to a circom 0.5.x binary, e.g. node_modules/.bin/circom after npm i circom@0.5.46}"
case "$("$CIRCOM1" --version)" in
  0.5.*) ;;
  *) echo "need circom 0.5.x, have: $("$CIRCOM1" --version 2>&1)" >&2; exit 1 ;;
esac
[ -f "$INPUT" ] || { echo "tornado: no input.json at $INPUT" >&2; exit 1; }
[ -f "$PTAU" ] || { echo "tornado: no ptau at $PTAU (gen-artifacts.sh makes local_19.ptau)" >&2; exit 1; }

log() { printf '\n=== %s %s ===\n' "$(date +%T)" "$*"; }

mkdir -p "$VENDOR" "$OUT"
[ -d "$VENDOR/tornado-core" ] ||
  git clone https://github.com/tornadocash/tornado-core.git "$VENDOR/tornado-core"
[ -d "$VENDOR/circomlib-tornado-fork" ] ||
  git clone https://github.com/tornadocash/circomlib.git "$VENDOR/circomlib-tornado-fork"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$VENDOR/tornado-core" archive "$CORE_REV" circuits | tar -x -C "$WORK"
LIB="$WORK/node_modules/circomlib"
git clone --quiet --no-checkout "$VENDOR/circomlib-tornado-fork" "$LIB"
git -C "$LIB" checkout --quiet "$CIRCOMLIB_REV"
git -C "$LIB" apply "$PATCH" || { echo "tornado: $(basename "$PATCH") does not apply to circomlib $CIRCOMLIB_REV" >&2; exit 1; }
git -C "$LIB" apply --check -R "$PATCH" || { echo "tornado: $(basename "$PATCH") did not apply cleanly" >&2; exit 1; }

log "tornado compile ($("$CIRCOM1" --version))"
(cd "$WORK" && "$CIRCOM1" circuits/withdraw.circom \
  -r "$OUT/circuit.r1cs" -w "$OUT/circuit.wasm" -s "$OUT/circuit.sym")
"$SNARKJS" r1cs info "$OUT/circuit.r1cs" | tee "$OUT/r1cs-info.txt"

log "tornado witness"
[ "$INPUT" -ef "$OUT/input.json" ] || cp "$INPUT" "$OUT/input.json"
"$SNARKJS" wtns calculate "$OUT/circuit.wasm" "$OUT/input.json" "$OUT/circuit.wtns"
"$HERE/scripts/check-witness.sh" tornado "$OUT/circuit.r1cs" "$OUT/circuit.wtns"

log "tornado zkey ($(basename "$PTAU"))"
"$SNARKJS" groth16 setup "$OUT/circuit.r1cs" "$PTAU" "$WORK/circuit_0.zkey"
"$SNARKJS" zkey contribute "$WORK/circuit_0.zkey" "$OUT/circuit.zkey" \
    --name="bench" -e="tornado-benchmark-entropy-not-secure"
"$SNARKJS" zkey export verificationkey "$OUT/circuit.zkey" "$OUT/vkey.json"

"$SNARKJS" groth16 prove "$OUT/circuit.zkey" "$OUT/circuit.wtns" \
    "$WORK/proof.json" "$OUT/public.json"
"$SNARKJS" groth16 verify "$OUT/vkey.json" "$OUT/public.json" "$WORK/proof.json"
log "tornado done"
