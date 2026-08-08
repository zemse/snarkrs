#!/usr/bin/env bash
# Generate circom+snarkjs Groth16 artifacts for the joinsplit benchmark.
# One locally-generated 2^19 ptau covers every variant: snarkjs accepts a ptau
# larger than the circuit needs. These are BENCHMARK keys, never production.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/artifacts"
PTAU_DIR="$HERE/ptau"
CIRCOMLIB="$HERE/node_modules/circomlib/circuits"
MAXPOW="${MAXPOW:-19}"
mkdir -p "$OUT" "$PTAU_DIR"

# name:nIns:nOuts:depth:valueBits
VARIANTS="${VARIANTS:-
js_1x1_d8:1:1:8:64
js_2x2_d16:2:2:16:64
js_2x2_d32:2:2:32:64
js_8x8_d32:8:8:32:64
js_16x16_d32:16:16:32:64
}"

log() { printf '\n=== %s ===\n' "$*"; }

P="$PTAU_DIR/local_${MAXPOW}.ptau"
if [ ! -f "$P" ]; then
  log "generating local powers of tau 2^$MAXPOW (NOT a real ceremony)"
  snarkjs powersoftau new bn128 "$MAXPOW" "$PTAU_DIR/g0.ptau" -v
  snarkjs powersoftau contribute "$PTAU_DIR/g0.ptau" "$PTAU_DIR/g1.ptau" \
      --name="bench" -e="benchmark-entropy-not-secure" -v
  snarkjs powersoftau prepare phase2 "$PTAU_DIR/g1.ptau" "$P" -v
  rm -f "$PTAU_DIR/g0.ptau" "$PTAU_DIR/g1.ptau"
fi
log "ptau ready: $(ls -lh "$P" | awk '{print $5}')"

: > "$OUT/manifest.csv"
for v in $VARIANTS; do
  [ -z "$v" ] && continue
  IFS=: read -r NAME NIN NOUT DEPTH VBITS <<<"$v"
  log "$NAME (nIns=$NIN nOuts=$NOUT depth=$DEPTH)"
  d="$OUT/$NAME"; mkdir -p "$d"

  cat > "$d/circuit.circom" <<CIRC
pragma circom 2.1.4;
include "joinsplit.circom";
component main {public [merkleRoot, nullifiers, outCommitments, tokenId]} =
    JoinSplit($NIN, $NOUT, $DEPTH, $VBITS);
CIRC

  circom "$d/circuit.circom" --r1cs --wasm -o "$d" \
    -l "$CIRCOMLIB" -l "$HERE/circuits" > "$d/compile.log" 2>&1

  snarkjs r1cs info "$d/circuit.r1cs" | tee "$d/r1cs-info.txt"
  NC=$(awk '/# of Constraints/{print $NF}' "$d/r1cs-info.txt")

  log "$NAME zkey"
  snarkjs groth16 setup "$d/circuit.r1cs" "$P" "$d/circuit_0.zkey"
  snarkjs zkey contribute "$d/circuit_0.zkey" "$d/circuit.zkey" \
      --name="bench" -e="joinsplit-benchmark-entropy-not-secure"
  rm -f "$d/circuit_0.zkey"
  snarkjs zkey export verificationkey "$d/circuit.zkey" "$d/vkey.json"

  log "$NAME witness"
  node "$HERE/scripts/make-input.js" "$NIN" "$NOUT" "$DEPTH" > "$d/input.json"
  node "$d/circuit_js/generate_witness.js" "$d/circuit_js/circuit.wasm" \
       "$d/input.json" "$d/circuit.wtns"

  # Reference proof from snarkjs. This is the oracle: our prover must produce a
  # proof that verifies against the SAME vkey and public signals.
  snarkjs groth16 prove "$d/circuit.zkey" "$d/circuit.wtns" \
      "$d/proof.json" "$d/public.json"
  snarkjs groth16 verify "$d/vkey.json" "$d/public.json" "$d/proof.json"

  echo "$NAME,$NC,$NIN,$NOUT,$DEPTH" >> "$OUT/manifest.csv"
  log "$NAME done: $NC constraints, zkey $(ls -lh "$d/circuit.zkey" | awk '{print $5}')"
done

log "manifest"; cat "$OUT/manifest.csv"
