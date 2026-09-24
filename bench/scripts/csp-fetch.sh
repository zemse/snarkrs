#!/usr/bin/env bash
# Fetch the ethproofs client-side-proving benchmark artifacts.
#
# The zkeys are the published ones, not ones we generate: a locally-run setup would
# produce a different delta and a different key size, and `preprocessing_size` is one of
# the six numbers the benchmark reports. Every download is checked against the manifest
# the upstream repo ships, so a truncated fetch fails here rather than 40 minutes into a
# proving run.
#
# Sources:
#   circuits  github.com/ethereum/csp-benchmarks @ main, sparse (circom/ only)
#   zkeys     the same repo's `zkeys-v2` release
#
# Needs `circom` 2.2.3 on PATH for the one witness generator upstream does not ship.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$HERE/vendor/csp-benchmarks"
OUT="$HERE/artifacts/csp"
REPO="https://github.com/ethereum/csp-benchmarks.git"
RELEASE="https://github.com/ethereum/csp-benchmarks/releases/download/zkeys-v2"
CIRCOMLIB_SHA="35e54ea21da3e8762557234298dbb553c175ea8d"

# name:zkey basename. The zkey lands as circuit.zkey so `g16 bench` finds it by the same
# four filenames it looks for in every other artifact directory.
VARIANTS="${VARIANTS:-
sha256_128 sha256_256 sha256_512 sha256_1024 sha256_2048
keccak_128 keccak_256 keccak_512 keccak_1024 keccak_2048
poseidon_2 poseidon_4 poseidon_8 poseidon_12 poseidon_16
ecdsa_32
}"

log() { printf '\n=== %s ===\n' "$*"; }

# The circuit sources, and circomlib at the submodule's pinned commit. Blobless and cone
# sparse: the full history carries ~100 MB of generated witness-calculator C++ we do not
# need, and the checkout is 111 MB even so.
if [ ! -d "$VENDOR/.git" ]; then
  log "cloning csp-benchmarks sources"
  mkdir -p "$(dirname "$VENDOR")"
  git clone --filter=blob:none --no-checkout --depth 1 "$REPO" "$VENDOR"
  git -C "$VENDOR" sparse-checkout init --cone
  git -C "$VENDOR" sparse-checkout set circom
  git -C "$VENDOR" checkout
fi
if [ ! -f "$VENDOR/circom/circomlib/circuits/poseidon.circom" ]; then
  log "circomlib @ $CIRCOMLIB_SHA"
  rm -rf "$VENDOR/circom/circomlib"
  git clone --filter=blob:none https://github.com/iden3/circomlib.git "$VENDOR/circom/circomlib"
  git -C "$VENDOR/circom/circomlib" checkout --quiet "$CIRCOMLIB_SHA"
fi

# `ecdsa_32` is the one witness generator upstream does not keep in its tree: 57 MB of
# generated C++, because the width-12 comb table is inlined in the circuit. Compile it
# where `build.rs` expects to read it, with the flags upstream's own build script uses.
# The includes are relative to the circuit directory, so circom runs from there.
ecdsa="$VENDOR/circom/circuits/ecdsa"
if [ ! -s "$ecdsa/ecdsa_32/ecdsa_32.dat" ]; then
  log "compiling the ecdsa_32 witness generator (a few minutes)"
  # Pinned to 2.2.3, the version upstream states its constraint counts for. A later
  # circom compiles the same source to a different R1CS, and a witness from it does not
  # satisfy the published zkey: the failure surfaces as a proof that will not verify,
  # a long way from the cause.
  CIRCOM="$HERE/bin/circom2"
  [ -x "$CIRCOM" ] || CIRCOM="$(command -v circom || true)"
  [ -n "$CIRCOM" ] || { echo "circom 2.2.3 is not on PATH and bench/bin/circom2 is missing" >&2; exit 1; }
  "$CIRCOM" --version 2>&1 | grep -q '2[.]2[.]3' \
    || { echo "need circom 2.2.3, have: $("$CIRCOM" --version 2>&1)" >&2; exit 1; }
  staging="$(mktemp -d)"
  ( cd "$ecdsa" && "$CIRCOM" ecdsa_32.circom --c --O2 -o "$staging" )
  mkdir -p "$ecdsa/ecdsa_32"
  cp "$staging/ecdsa_32_cpp/ecdsa_32.cpp" "$staging/ecdsa_32_cpp/ecdsa_32.dat" "$ecdsa/ecdsa_32/"
  rm -rf "$staging"
fi

# checksums.sha256 maps a release asset to its path inside the upstream tree. We only
# want the basename and the digest.
manifest="$VENDOR/circom/checksums.sha256"
[ -f "$manifest" ] || { echo "missing $manifest" >&2; exit 1; }

want_digest() {  # $1 = zkey basename
  awk -v n="$1" '$2 ~ ("/" n "$") { print $1 }' "$manifest"
}

for name in $VARIANTS; do
  [ -z "$name" ] && continue
  d="$OUT/$name"
  mkdir -p "$d"
  asset="${name}_0001.zkey"
  want="$(want_digest "$asset")"
  [ -n "$want" ] || { echo "$asset not in checksums.sha256" >&2; exit 1; }

  if [ -f "$d/circuit.zkey" ]; then
    if [ "$got" = "$want" ]; then
      printf 'ok    %-14s zkey already present\n' "$name"
      continue
    fi
    echo "warn  $name: zkey present but digest differs, refetching" >&2
  fi

  log "$name: downloading $asset"
  curl --fail --location --retry 3 --retry-delay 5 -C - \
       "$RELEASE/$asset" --output "$d/circuit.zkey.part"
  if [ "$got" != "$want" ]; then
    echo "FAIL  $name: sha256 $got, expected $want" >&2
    exit 1
  fi
  mv "$d/circuit.zkey.part" "$d/circuit.zkey"
  printf 'ok    %-14s %s\n' "$name" "$(ls -lh "$d/circuit.zkey" | awk '{print $5}')"
done

log "all zkeys present and verified under $OUT"
