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
    got="$(shasum -a 256 "$d/circuit.zkey" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      printf 'ok    %-14s zkey already present\n' "$name"
      continue
    fi
    echo "warn  $name: zkey present but digest differs, refetching" >&2
  fi

  log "$name: downloading $asset"
  curl --fail --location --retry 3 --retry-delay 5 -C - \
       "$RELEASE/$asset" --output "$d/circuit.zkey.part"
  got="$(shasum -a 256 "$d/circuit.zkey.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "FAIL  $name: sha256 $got, expected $want" >&2
    exit 1
  fi
  mv "$d/circuit.zkey.part" "$d/circuit.zkey"
  printf 'ok    %-14s %s\n' "$name" "$(ls -lh "$d/circuit.zkey" | awk '{print $5}')"
done

log "all zkeys present and verified under $OUT"
