#!/usr/bin/env bash
# Run only inside a disposable Linux guest with the x86 GNU cross-toolchain.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux guest required' >&2; exit 1; }
SRC="${1:?rapidsnark source directory required}"
OUT="${2:?bundle directory required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ "$(git -C "$SRC" rev-parse HEAD)" == 81eddf1a536d26497b237c0b8a04fe90baf7e439 ]]
[[ "$(git -C "$SRC/depends/ffiasm" rev-parse HEAD)" == aa90166dc4c5a075b835a398e15cc1e06ac90e95 ]]
if git -C "$SRC" submodule status --recursive | grep -q '^[+-]'; then
    echo 'submodule revision mismatch' >&2; exit 1
fi
mkdir -p "$OUT/bin" "$OUT/metadata" "$OUT/sources" "$OUT/work"
if [[ "${G16_WRAPPER_ONLY:-0}" == 1 ]]; then
    exec > >(tee "$OUT/metadata/build-wrapper.log") 2>&1
else
    exec > >(tee "$OUT/metadata/build.log") 2>&1
fi
set -x
if [[ "${G16_WRAPPER_ONLY:-0}" == 1 ]]; then
    [[ -s "$OUT/work/rapidsnark-build/src/librapidsnark-fr-fq.a" ]]
    [[ -s "$SRC/depends/gmp/package/lib/libgmp.a" ]]
else
    cd "$OUT/work"
    if [[ ! -f gmp-6.3.0.tar.xz ]]; then
        curl -fL --retry 2 --max-time 120 -o gmp-6.3.0.tar.xz https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz
    fi
    printf '%s  %s\n' a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898 gmp-6.3.0.tar.xz | sha256sum -c -
    if [[ "${G16_REUSE_GMP:-0}" == 1 ]]; then
        [[ -s "$SRC/depends/gmp/package/lib/libgmp.a" && -s gmp-build/config.log ]]
        echo 'Reusing the retained guest GMP build; original configuration log retained.'
    else
        tar -xf gmp-6.3.0.tar.xz
        mkdir -p gmp-build
        cd gmp-build
        ../gmp-6.3.0/configure --build=aarch64-linux-gnu --host=x86_64-linux-gnu \
            --prefix="$SRC/depends/gmp/package" --with-pic --disable-shared --enable-static \
            --disable-fft --enable-fat CC=x86_64-linux-gnu-gcc CXX=x86_64-linux-gnu-g++ \
            CC_FOR_BUILD=gcc AR=x86_64-linux-gnu-ar RANLIB=x86_64-linux-gnu-ranlib CFLAGS='-O3'
        make -j4
        make install
    fi
    cp "$OUT/work/gmp-build/config.log" "$OUT/metadata/gmp-config.log"
    sha256sum "$SRC/depends/gmp/package/lib/libgmp.a" > "$OUT/metadata/gmp-library.sha256"
    cmake -S "$SRC" -B "$OUT/work/rapidsnark-build" \
        -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
        -DCMAKE_C_COMPILER=x86_64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=x86_64-linux-gnu-g++ \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-include sys/types.h' \
        -DCMAKE_EXE_LINKER_FLAGS=-static -DUSE_ASM=ON -DUSE_OPENMP=ON \
        -DOpenMP_gomp_LIBRARY="$(x86_64-linux-gnu-gcc -print-file-name=libgomp.a)"
    cmake --build "$OUT/work/rapidsnark-build" --target prover verifier rapidsnarkStaticFrFq -j4
    cp "$OUT/work/rapidsnark-build/src/prover" "$OUT/bin/rapidsnark"
    cp "$OUT/work/rapidsnark-build/src/verifier" "$OUT/bin/rapidsnark-verify"
fi
x86_64-linux-gnu-g++ -std=c++17 -O3 -fopenmp -static -include sys/types.h \
    "$HERE/main.cpp" -I"$SRC/src" -I"$SRC/build" -I"$SRC/depends/ffiasm/c" \
    -I"$SRC/depends/json/single_include" -I"$SRC/depends/gmp/package/include" \
    "$OUT/work/rapidsnark-build/src/librapidsnark-fr-fq.a" \
    "$SRC/depends/gmp/package/lib/libgmp.a" -lpthread -o "$OUT/bin/rapidsnark-warm"
cp "$HERE/main.cpp" "$HERE/build-linux-x86.sh" "$HERE/test_wrapper.py" \
    "$HERE/smoke-linux-x86.py" "$OUT/sources/"
git -C "$SRC" rev-parse HEAD > "$OUT/metadata/rapidsnark-revision.txt"
git -C "$SRC" submodule status --recursive > "$OUT/metadata/submodules.txt"
x86_64-linux-gnu-g++ --version > "$OUT/metadata/compiler.txt"
nasm -v > "$OUT/metadata/nasm.txt"
cmake --version > "$OUT/metadata/cmake.txt"
qemu-x86_64 --version > "$OUT/metadata/qemu.txt"
cp "$OUT/work/rapidsnark-build/CMakeCache.txt" "$OUT/metadata/"
for binary in "$OUT"/bin/*; do
    [[ -x "$binary" ]]
    x86_64-linux-gnu-readelf -h "$binary" | grep -q 'Machine:.*Advanced Micro Devices X86-64'
    if x86_64-linux-gnu-readelf -l "$binary" | grep -q INTERP ||
       x86_64-linux-gnu-readelf -d "$binary" | grep -q NEEDED; then
        echo "unexpected runtime ELF dependency: $binary" >&2; exit 1
    fi
    file "$binary"
    x86_64-linux-gnu-readelf -h -l -d "$binary"
done > "$OUT/metadata/elf.txt"
cd "$OUT"
find bin sources -type f ! -name '._*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
printf '\nBUILD_SUCCESS\n'
