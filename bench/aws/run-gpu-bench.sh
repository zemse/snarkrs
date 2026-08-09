#!/usr/bin/env bash
# Ship this checkout to the GPU box, build it with CUDA, and run the full comparison.
#
# Two things this does that a bare `run-comparison.py` on the box would not.
#
# It warms the kernel cache before timing anything. The first NVRTC compile of the MSM unit
# takes 113 seconds and every subsequent one takes 0.18, because the NVIDIA driver caches
# NVRTC output in ~/.nv/ComputeCache (see bench/results/device-microbench.md). Leaving that
# inside the measurement would put a two minute outlier in the first cold rep of the first
# variant and nowhere else, which is neither the cold number anyone wants nor an honest
# average. The first-compile cost is reported separately instead, by --first-compile.
#
# It runs the benchmark with nothing else on the box. GPU benchmarking next to a concurrent
# build is how you get a constant that is wrong by 6x, which already happened once in this
# project on the CPU side.
#
# usage: run-gpu-bench.sh <user@host> [reps]
#        run-gpu-bench.sh <user@host> --first-compile   # measure a genuine cold compile
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${1:?usage: run-gpu-bench.sh ubuntu@<ip> [reps]}"
KEY="${G16_CUDA_KEY:-$HOME/.ssh/g16-cuda-bench.pem}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST")
REPS="${2:-10}"

echo "==> syncing"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude 'target/' --exclude '.git/' --exclude 'bench/vendor/' --exclude 'bench/ptau/' \
  --exclude 'bench/bin/' --exclude 'circuit_js/' --exclude '*.r1cs' \
  "$HERE/" "$HOST:~/g16/"

echo "==> building with cuda"
"${SSH[@]}" 'source ~/.cargo/env; cd ~/g16 && cargo build --release -p g16-cli --features cuda 2>&1 | tail -2'

if [ "${2:-}" = "--first-compile" ]; then
  echo "==> measuring a genuine first compile (clearing both caches)"
  "${SSH[@]}" 'source ~/.cargo/env; cd ~/g16 && rm -rf ~/.cache/g16-cuda ~/.nv && \
    /usr/bin/time -f "FIRST COMPILE + first proof: %e s" \
    ./target/release/g16 prove --zkey bench/artifacts/tiny_mul/circuit.zkey \
      --witness bench/artifacts/tiny_mul/circuit.wtns --proof /tmp/fc.json \
      --public /tmp/fcp.json --backend cuda 2>&1 | tail -2 && \
    /usr/bin/time -f "SECOND (both caches warm): %e s" \
    ./target/release/g16 prove --zkey bench/artifacts/tiny_mul/circuit.zkey \
      --witness bench/artifacts/tiny_mul/circuit.wtns --proof /tmp/fc.json \
      --public /tmp/fcp.json --backend cuda 2>&1 | tail -2'
  exit 0
fi

echo "==> warming the kernel cache so it is not inside the first rep"
"${SSH[@]}" 'source ~/.cargo/env; cd ~/g16 && ./target/release/g16 prove \
  --zkey bench/artifacts/tiny_mul/circuit.zkey --witness bench/artifacts/tiny_mul/circuit.wtns \
  --proof /tmp/warm.json --public /tmp/warmp.json --backend cuda >/dev/null 2>&1 && echo warmed'

echo "==> benchmarking (reps=$REPS), nothing else should be running"
"${SSH[@]}" "source ~/.cargo/env; cd ~/g16 && uptime && python3 bench/scripts/run-comparison.py --reps $REPS --backends cpu cuda"

echo "==> pulling results"
scp -q -i "$KEY" -o StrictHostKeyChecking=no "$HOST:~/g16/bench/results/comparison-*.csv" "$HERE/bench/results/"
ls -la "$HERE/bench/results/"
