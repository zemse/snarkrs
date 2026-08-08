#!/usr/bin/env bash
# Build rapidsnark (the CPU baseline) and our warm-mode wrapper around its library.
#
# Two local fixes are needed on a current macOS/arm64 box, neither of them ours to
# keep: CMake 4 rejects the cmake_minimum_required(<3.5) that rapidsnark's vendored
# json and pistache still declare, and build_gmp.sh aborts the whole build when the
# first GNU mirror serves a corrupt tarball instead of falling through to the other
# two (it verifies the checksum after the mirror loop has already broken out of it,
# and deletes the archive on failure). Both are worth upstreaming.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$HERE/vendor"
SRC="$VENDOR/rapidsnark"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

mkdir -p "$VENDOR"
if [ ! -d "$SRC" ]; then
  git clone --recursive --depth 1 https://github.com/iden3/rapidsnark.git "$SRC"
fi
cd "$SRC"
if [ ! -f build_gmp.sh.orig ]; then
  cp build_gmp.sh build_gmp.sh.orig
  patch -p0 < "$HERE/scripts/rapidsnark-gmp-mirror-fallback.patch"
fi

[ -d depends/gmp/package_macos_arm64 ] || ./build_gmp.sh macos_arm64
[ -x package_macos_arm64/bin/prover ]  || make macos_arm64

# Warm wrapper: rapidsnark's stock CLI reparses the zkey on every invocation, so it
# cannot measure steady state. Its library exposes the object API that proverServer
# uses, where the parse happens once in groth16_prover_create. Same proving code,
# the setup just lands outside the timed region.
PKG="$SRC/package_macos_arm64"
mkdir -p "$HERE/bin"
clang++ -std=c++17 -O3 -o "$HERE/bin/rapidsnark-warm" \
  "$HERE/wrappers/rapidsnark-warm/main.cpp" \
  -I"$PKG/include" -I"$SRC/src" -I"$SRC/depends/json/include" \
  -L"$PKG/lib" -lrapidsnark -lrapidsnark-fr-fq -lfr -lfq -lgmp \
  -Wl,-rpath,"$PKG/lib"

cp "$PKG/bin/prover"   "$HERE/bin/rapidsnark"
cp "$PKG/bin/verifier" "$HERE/bin/rapidsnark-verify"
echo "built: $HERE/bin/{rapidsnark,rapidsnark-verify,rapidsnark-warm}"
