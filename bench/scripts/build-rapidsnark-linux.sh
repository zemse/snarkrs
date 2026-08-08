#!/usr/bin/env bash
# Linux/x86_64 twin of build-rapidsnark.sh, for the EC2 GPU box.
#
# Same baseline, same warm wrapper, different toolchain triple: rapidsnark's Makefile
# calls the native target "host" and drops it in package/ rather than
# package_macos_arm64/. The build_gmp.sh mirror-fallback patch is applied here too,
# because the bug is in the script and not in the platform: it breaks out of the mirror
# loop on transfer success, verifies the checksum only after the loop, and deletes the
# archive on mismatch, so a single corrupt mirror kills the build with two good mirrors
# left untried.
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
  patch -p0 < "$HERE/scripts/rapidsnark-gmp-mirror-fallback.patch" || {
    echo "patch did not apply; continuing with upstream build_gmp.sh" >&2
    cp build_gmp.sh.orig build_gmp.sh
  }
fi

[ -d depends/gmp/package ] || ./build_gmp.sh host
[ -x package/bin/prover ]  || make host

PKG="$SRC/package"
mkdir -p "$HERE/bin"
cp "$PKG/bin/prover" "$HERE/bin/rapidsnark-prover"
cp "$PKG/bin/verifier" "$HERE/bin/rapidsnark-verify"

# Warm wrapper: the stock CLI reparses the zkey on every invocation, so it cannot measure
# steady state. Its library exposes the object API proverServer uses, where the parse
# happens once in groth16_prover_create and lands outside the timed region.
g++ -std=c++17 -O3 -o "$HERE/bin/rapidsnark-warm" \
  "$HERE/wrappers/rapidsnark-warm/main.cpp" \
  -I"$PKG/include" -I"$SRC/src" -I"$SRC/depends/json/include" \
  -L"$PKG/lib" -lrapidsnark -lrapidsnark-fr-fq -lfr -lfq -lgmp -lpthread -lomp \
  -Wl,-rpath,"$PKG/lib" \
  || g++ -std=c++17 -O3 -fopenmp -o "$HERE/bin/rapidsnark-warm" \
    "$HERE/wrappers/rapidsnark-warm/main.cpp" \
    -I"$PKG/include" -I"$SRC/src" -I"$SRC/depends/json/include" \
    -L"$PKG/lib" -lrapidsnark -lrapidsnark-fr-fq -lfr -lfq -lgmp -lpthread \
    -Wl,-rpath,"$PKG/lib"

echo "built: $HERE/bin/rapidsnark-prover $HERE/bin/rapidsnark-verify $HERE/bin/rapidsnark-warm"
