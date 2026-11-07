#!/usr/bin/env bash
# The CUDA gnark path on a GPU box, end to end: the toy fixture through the device tests
# and gnark's own groth16.Verify, then SP1 v6.1.0 (2^24) through GnarkProver with
# construction by phase, a cold proof and warm ones by stage, peak host RSS per phase,
# the device budget against what construction took, and peak device memory sampled by
# nvidia-smi. Every step's output lands in <out dir>; it stops at the first failure.
#
# usage, from the repository root on the GPU box:
#   gnark-cuda-validate.sh <sp1 circuit dir> <sp1 witness dump> <out dir> [warm proofs]
# e.g.
#   bench/scripts/gnark-cuda-validate.sh ~/.sp1/circuits/groth16/v6.1.0 \
#       ~/sp1-groth16-fixtures/sp1-solution.bin ~/snarkrs-gnk04
#
# Needs: the CUDA driver, `go` on PATH for test-support/gnark, and about 15.3 GB of free
# device memory (the SP1 budget the example prints first; it refuses with less). On the
# host, construction holds 64 MiB of pinned staging, a run of the mapped dump, the 31 MB of
# infinity flags and, for its warm-up pass, a 504 MB zero witness; a proof holds the
# witness (1.5 GB, read whole as the caller's solver buffers would be) and the staging.
# On a shared box, wrap this in the box's own job runner.
set -euo pipefail
[ $# -ge 3 ] || {
  echo "usage: $0 <sp1 circuit dir> <sp1 witness dump> <out dir> [warm proofs]" >&2
  exit 2
}
SP1=$1 WIT=$2 OUT=$3 WARM=${4:-3}
ROOT=$(pwd)
mkdir -p "$OUT"
step() { echo "== $*" | tee -a "$OUT/summary.txt"; }

step "device"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv \
  | tee "$OUT/nvidia-smi.txt"

step "host tests (the kernel models, the dump layout, the SP1 budget)"
cargo test --release -p snarkrs-cuda --features cuda --lib -- --nocapture 2>&1 \
  | tee "$OUT/cuda-lib.txt"

step "device tests on the toy fixture, CPU as the oracle, gnark verify included"
G16_REQUIRE_TESTS=1 cargo test --release -p snarkrs-cuda --features cuda --test gnark_gpu \
  -- --nocapture --test-threads=1 2>&1 | tee "$OUT/gnark-gpu.txt"

step "build the example"
cargo build --release -p snarkrs-cuda --features cuda --example gnark_prove
EX=$ROOT/target/release/examples/gnark_prove

step "toy through GnarkProver, then gnark's groth16.Verify"
TOY=$ROOT/test-support/gnark/toy
"$EX" "$TOY/pk.dump" "$TOY/vk.bin" "$TOY/witness.bin" "$OUT/toy" "$WARM" 2>&1 | tee "$OUT/toy.txt"
(cd "$ROOT/test-support/gnark" && go run . verify "$TOY/vk.bin" "$OUT/toy/public.bin" \
  "$OUT/toy/proof.bin") 2>&1 | tee -a "$OUT/toy.txt"

step "SP1 v6.1.0 through GnarkProver, device memory sampled every 200 ms"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 >"$OUT/vram-mib.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
"$EX" "$SP1/groth16_pk.bin" "$SP1/groth16_vk.bin" "$WIT" "$OUT/sp1" "$WARM" 2>&1 | tee "$OUT/sp1.txt"
kill $SMI 2>/dev/null || true
step "peak device memory used, MiB, every process on the card: $(sort -n "$OUT/vram-mib.csv" | tail -1)"

step "SP1 proof through gnark's groth16.Verify"
(cd "$ROOT/test-support/gnark" && go run . verify "$SP1/groth16_vk.bin" "$OUT/sp1/public.bin" \
  "$OUT/sp1/proof.bin") 2>&1 | tee -a "$OUT/sp1.txt"

step "done: $OUT"
