#!/usr/bin/env bash
# Adapt the independent verifier to snarkrs bench's foreign-verifier interface.
set -euo pipefail
[[ $# == 5 && "$1" == groth16 && "$2" == verify ]] || {
  echo 'expected: groth16 verify vkey public proof' >&2
  exit 2
}
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/check_reference_public.py" "$3" "$4"
output=$("$HERE/rapidsnark-verify" "$3" "$4" "$5" 2>&1)
printf '%s\n' "$output"
[[ "$output" == *'Result: Valid proof'* ]] || exit 1
# Emit the marker only after the reference check and independent verification succeed.
echo 'OK!'
