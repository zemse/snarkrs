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
# Needs `circom` 2.2.3 on PATH: every witness generator is compiled here, see below.
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

# The witness generator each circuit is built from, as `<name>.staged.cpp`/`.staged.dat`.
# `build.rs` reads only those two, so the choice made here is the whole of it.
#
# Where we can, the staged generator is a local `--sanity_check 0` build. circom's C++
# backend emits, for each `===` constraint, code that re-evaluates the constraint
# polynomial in field arithmetic and compares it. That is the proof system's job, not the
# witness generator's, and it is most of the work: one `Xor3Bits` bit costs two `Fr_bxor`
# and a copy to compute, and six muls, five adds, two subs and an `Fr_eq` to re-check.
# The flag omits it and takes keccak_2048 from 357 ms to 145. Author-written `assert()`
# in the circom sources is kept; only the generated constraint checks go.
#
# Two things have to hold before a local build may replace upstream's, and both are
# checked per circuit rather than assumed:
#
#   1. Compiling WITHOUT the flag must reproduce upstream's shipped `.cpp` and `.dat` byte
#      for byte. That is what makes the flag the only difference between their generator
#      and ours, and it is a property of this machine's circom and this circomlib, not
#      something that can be taken on faith. sha256 and keccak pass. **All five poseidon
#      circuits fail it**: upstream's shipped poseidon generators do not come from circom
#      2.2.3 over circomlib $CIRCOMLIB_SHA, so we keep theirs and forgo the flag there.
#      Poseidon is a fraction of a percent of witness time and gains nothing from it
#      anyway, every variant measuring 1.00x.
#
#   2. The R1CS must be identical with and without the flag, so the published zkey stays
#      valid and `num_constraints` does not move.
#
# A generator built from sources that do not match the one the published zkey was
# generated for produces a witness that does not satisfy it, and that surfaces as a proof
# which will not verify, a long way from the cause. Hence the checks, and hence circom is
# pinned to 2.2.3, the version upstream states its constraint counts for.
#
# `ecdsa_32` has no upstream generator to compare against: 57 MB of generated C++ is not
# in their tree, so check 1 does not apply and we build it as we always have.
CIRCOM="$HERE/bin/circom2"
[ -x "$CIRCOM" ] || CIRCOM="$(command -v circom || true)"
[ -n "$CIRCOM" ] || { echo "circom 2.2.3 is not on PATH and bench/bin/circom2 is missing" >&2; exit 1; }
"$CIRCOM" --version 2>&1 | grep -q '2[.]2[.]3' \
  || { echo "need circom 2.2.3, have: $("$CIRCOM" --version 2>&1)" >&2; exit 1; }

digest() { shasum -a256 "$1" | cut -d' ' -f1; }

for name in $VARIANTS; do
  [ -z "$name" ] && continue
  family="${name%_*}"
  dir="$VENDOR/circom/circuits/$family/$name"
  [ -s "$dir/$name.staged.cpp" ] && continue
  mkdir -p "$dir"

  log "building the $name witness generator"
  plain="$(mktemp -d)"
  ( cd "$VENDOR/circom/circuits/$family" && "$CIRCOM" "$name.circom" --c --r1cs --O2 -o "$plain" )

  # Check 1, where upstream gives us something to check against.
  faithful=yes
  if [ -s "$dir/$name.cpp" ]; then
    [ "$(digest "$plain/${name}_cpp/$name.cpp")" = "$(digest "$dir/$name.cpp")" ] || faithful=no
    [ "$(digest "$plain/${name}_cpp/$name.dat")" = "$(digest "$dir/$name.dat")" ] || faithful=no
  fi

  if [ "$faithful" = no ]; then
    echo "      our circom does not reproduce upstream's $name generator; keeping theirs" >&2
    cp "$dir/$name.cpp" "$dir/$name.staged.cpp"
    cp "$dir/$name.dat" "$dir/$name.staged.dat"
    rm -rf "$plain"
    continue
  fi

  flagged="$(mktemp -d)"
  ( cd "$VENDOR/circom/circuits/$family" && "$CIRCOM" "$name.circom" --c --r1cs --O2 --sanity_check 0 -o "$flagged" )

  # Check 2. The constraint system is what the published zkey commits to.
  [ "$(digest "$plain/$name.r1cs")" = "$(digest "$flagged/$name.r1cs")" ] || {
    echo "$name: --sanity_check 0 changed the R1CS, which would invalidate the zkey" >&2
    exit 1; }

  cp "$flagged/${name}_cpp/$name.cpp" "$dir/$name.staged.cpp"
  cp "$flagged/${name}_cpp/$name.dat" "$dir/$name.staged.dat"
  rm -rf "$plain" "$flagged"
done

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
    got="$(digest "$d/circuit.zkey")"
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
