#!/usr/bin/env bash
# Abort unless a witness satisfies every constraint of its r1cs.
#
# `snarkjs wtns calculate` does not check `===` on circom 1's wasm, so a miscompiled
# circuit hands back a plausible witness that no prover can use, and the failure surfaces
# much later as a proof that does not verify. Run this after every
# `snarkjs wtns calculate` or `generate_witness.js`.
#
# usage: check-witness.sh <circuit name> <circuit.r1cs> <circuit.wtns>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <circuit name> <circuit.r1cs> <circuit.wtns>" >&2; exit 2; }
NAME=$1 R1CS=$2 WTNS=$3
SNARKJS="${SNARKJS:-snarkjs}"

if ! out=$("$SNARKJS" wtns check "$R1CS" "$WTNS" 2>&1) || ! grep -q "WITNESS IS CORRECT" <<<"$out"; then
  printf '%s\n' "$out" >&2
  echo "$NAME: $WTNS does not satisfy $R1CS" >&2
  exit 1
fi
echo "$NAME: witness satisfies the r1cs"
