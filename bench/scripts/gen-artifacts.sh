#!/usr/bin/env bash
# Generate circom+snarkjs Groth16 artifacts for the joinsplit benchmark.
# One locally-generated 2^19 ptau covers every variant: snarkjs accepts a ptau
# larger than the circuit needs. These are BENCHMARK keys, never production.
#
# LARGE=1 builds the production-scale variant instead, into artifacts/large/, one level
# below where every test scans for variants, so only the tests that ask for it load it.
# Its domain is past the local ptau, so it is keyed against the PSE ppot_0080 file of
# its own power, fetched on first use. Expect about an hour and tens of GB of RAM.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/artifacts"
PTAU_DIR="$HERE/ptau"
CIRCOMLIB="$HERE/node_modules/circomlib/circuits"
MAXPOW="${MAXPOW:-19}"
PPOT_URL="https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080"

# name:nIns:nOuts:depth:valueBits
if [ "${LARGE:-0}" = 1 ]; then
  OUT="$HERE/artifacts/large"
  VARIANTS="${VARIANTS:-js_384x384_d32:384:384:32:64}"
  # snarkjs holds the whole 2^22 setup in the JS heap
  export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=65536}"
fi
VARIANTS="${VARIANTS:-
js_1x1_d8:1:1:8:64
js_2x2_d16:2:2:16:64
js_2x2_d32:2:2:32:64
js_8x8_d32:8:8:32:64
js_16x16_d32:16:16:32:64
}"
mkdir -p "$OUT" "$PTAU_DIR"

log() { printf '\n=== %s %s ===\n' "$(date +%T)" "$*"; }

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

touch "$OUT/manifest.csv"
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

  # --O2 explicitly: circom 2.2 defaults to --O1, which keeps the linear constraints and
  # doubles the count against the artifacts on disk (js_1x1_d8: 3,359 against 7,211).
  circom "$d/circuit.circom" --r1cs --wasm --O2 -o "$d" \
    -l "$CIRCOMLIB" -l "$HERE/circuits" > "$d/compile.log" 2>&1

  snarkjs r1cs info "$d/circuit.r1cs" | tee "$d/r1cs-info.txt"
  NC=$(awk '/# of Constraints/{print $NF}' "$d/r1cs-info.txt")
  NPUB=$(awk '/# of Public Inputs/{print $NF}' "$d/r1cs-info.txt")

  # snarkjs' domain: the constraints plus one per public input plus one
  POW=0; while [ $((1 << POW)) -lt $((NC + NPUB + 1)) ]; do POW=$((POW + 1)); done
  VP="$P"
  if [ "$POW" -gt "$MAXPOW" ]; then
    VP="$PTAU_DIR/ppot_0080_${POW}.ptau"
    if [ ! -f "$VP" ]; then
      log "fetching ppot_0080_${POW}.ptau"
      curl -fL -C - -o "$VP.part" "$PPOT_URL/ppot_0080_${POW}.ptau"
      mv "$VP.part" "$VP"
    fi
  fi

  log "$NAME zkey (domain 2^$POW, $(basename "$VP"))"
  snarkjs groth16 setup "$d/circuit.r1cs" "$VP" "$d/circuit_0.zkey"
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

  grep -v "^$NAME," "$OUT/manifest.csv" > "$OUT/manifest.csv.tmp" || true
  echo "$NAME,$NC,$NIN,$NOUT,$DEPTH" >> "$OUT/manifest.csv.tmp"
  mv "$OUT/manifest.csv.tmp" "$OUT/manifest.csv"
  log "$NAME done: $NC constraints, zkey $(ls -lh "$d/circuit.zkey" | awk '{print $5}')"
done

log "manifest"; cat "$OUT/manifest.csv"
