#!/usr/bin/env bash
# RUSTC_WRAPPER: adds feature="asm" to ark-bn254 only, so its derived MontConfig takes ark-ff's asm path.
rustc=$1
shift
for a in "$@"; do
  if [[ $a == ark_bn254 ]]; then
    exec "$rustc" "$@" --cfg 'feature="asm"'
  fi
done
exec "$rustc" "$@"
