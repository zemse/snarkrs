#!/usr/bin/env bash
# Benchmark CUDA on a rented GPU box without paying GPU rates to run a compiler.
#
# This is run-gpu-bench.sh with the build moved off the instance. The M2 Max cross-compiles
# the CUDA CLI to x86_64 linux in about 32 seconds via cargo-zigbuild, and the binary is
# shipped instead of the source. That is possible only because g16-cuda uses cudarc with
# `dynamic-loading`, so libcuda and libnvrtc are dlopened at run time rather than linked:
# nothing about the build needs a driver, a toolkit, or a GPU. See crates/g16-cuda/Cargo.toml.
#
# What that buys, in order of how much it matters:
#   * The instance's billed lifetime no longer includes a fat-LTO release build.
#   * The measured binary is byte-identical across runs and across instances, so a number
#     that moves is the code changing, not the box's toolchain changing. The old script
#     inherited whatever rustc the AMI happened to carry.
#   * The rental window shrinks to warm-up plus measurement, which is the part that
#     actually needs a GPU.
#
# What it does NOT change: the kernel cache still has to be warmed before timing. The first
# NVRTC compile of the MSM unit costs 113 seconds and every later one costs 0.18, because
# the driver caches NVRTC output in ~/.nv/ComputeCache. That is a run-time cost on the GPU
# and no amount of cross-compiling avoids it. Left un-warmed it puts a two minute outlier in
# the first cold rep of the first variant and nowhere else.
#
# usage: run-gpu-bench-prebuilt.sh <user@host> [reps]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${1:?usage: run-gpu-bench-prebuilt.sh ubuntu@<ip> [reps]}"
KEY="${G16_CUDA_KEY:-$HOME/.ssh/g16-cuda-bench.pem}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST")
RSH="ssh -i $KEY -o StrictHostKeyChecking=no"
REPS="${2:-10}"

# glibc 2.31 rather than the host's newest: the Deep Learning Base AMI is Ubuntu 22.04
# (glibc 2.35), and pinning below it leaves room to run the same binary on an older box
# without a rebuild. Pinning above it would fail at exec with a version error that looks
# nothing like its cause.
TARGET="x86_64-unknown-linux-gnu.2.31"
BIN="$HERE/target/x86_64-unknown-linux-gnu/release/g16"

echo "==> cross-compiling for $TARGET on this machine"
( cd "$HERE" && cargo zigbuild --release -p g16-cli --features cuda --target "$TARGET" )
file "$BIN" | grep -q 'ELF 64-bit.*x86-64' || { echo "not an x86-64 ELF, refusing to ship"; exit 1; }
echo "    $(ls -lh "$BIN" | awk '{print $5}')  $(cd "$HERE" && git rev-parse --short HEAD)"

# -L follows symlinks. bench/artifacts, bench/bin and friends are symlinks into the primary
# checkout in this worktree layout, and -a alone would ship the dangling links themselves.
echo "==> syncing artifacts and scripts (no source, no target/)"
rsync -azL -e "$RSH" \
  --exclude 'target/' --exclude '.git/' --exclude 'bench/vendor/' --exclude 'bench/ptau/' \
  --exclude 'bench/bin/' --exclude 'bench/node_modules/' --exclude 'circuit_js/' --exclude '*.r1cs' \
  "$HERE/bench/" "$HOST:~/g16/bench/"

echo "==> shipping the prebuilt binary"
"${SSH[@]}" 'mkdir -p ~/g16/target/release'
rsync -az -e "$RSH" "$BIN" "$HOST:~/g16/target/release/g16"
"${SSH[@]}" 'chmod +x ~/g16/target/release/g16 && ~/g16/target/release/g16 --version 2>/dev/null || true'

echo "==> warming the kernel cache so it is not inside the first rep"
"${SSH[@]}" 'cd ~/g16 && ./target/release/g16 prove \
  --zkey bench/artifacts/tiny_mul/circuit.zkey --witness bench/artifacts/tiny_mul/circuit.wtns \
  --proof /tmp/warm.json --public /tmp/warmp.json --backend cuda >/dev/null 2>&1 && echo warmed'

echo "==> benchmarking (reps=$REPS), nothing else should be running"
"${SSH[@]}" "cd ~/g16 && uptime && python3 bench/scripts/run-comparison.py --reps $REPS --backends cpu cuda"

echo "==> pulling results"
scp -q -i "$KEY" -o StrictHostKeyChecking=no "$HOST:~/g16/bench/results/comparison-*.csv" "$HERE/bench/results/"
ls -la "$HERE/bench/results/"

echo
echo "TERMINATE THE BOX NOW: ./bench/aws/terminate.sh <instance-id>"
