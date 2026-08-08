#!/usr/bin/env bash
# Syntax and type check a CUDA translation unit on the remote GPU box.
#
# There is no CUDA compiler on an M2 Max, so this is the only way to find out whether a
# kernel is even valid before the integration build. It assembles the unit the same way
# kernels.rs does at run time (concatenate headers, then the kernel: NVRTC has no
# filesystem, so no .cu file may carry an #include of its own) and runs nvcc -ptx on the
# box. nvcc's front end is not byte-identical to NVRTC's, but it catches every syntax and
# type error, which is what this is for.
#
# usage: check-cu.sh <header.cuh>... <kernel.cu>
#   env: G16_CUDA_HOST  ubuntu@<ip>   (required)
#        G16_CUDA_KEY   path to the pem  (default ~/.ssh/g16-cuda-bench.pem)
set -euo pipefail
HOST="${G16_CUDA_HOST:?set G16_CUDA_HOST=ubuntu@<ip>}"
KEY="${G16_CUDA_KEY:-$HOME/.ssh/g16-cuda-bench.pem}"
ARCH="${G16_CUDA_ARCH:-sm_75}"

[ $# -ge 1 ] || { echo "usage: check-cu.sh <header.cuh>... <kernel.cu>" >&2; exit 2; }
for f in "$@"; do [ -f "$f" ] || { echo "no such file: $f" >&2; exit 2; }; done

UNIT="$(mktemp -d)/unit.cu"
cat "$@" > "$UNIT"
REMOTE="/tmp/g16-check-$$-$(basename "${!#}").cu"

scp -q -i "$KEY" -o StrictHostKeyChecking=no "$UNIT" "$HOST:$REMOTE"
ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST" \
  "nvcc -arch=$ARCH -ptx -o /dev/null '$REMOTE' 2>&1; rc=\$?; rm -f '$REMOTE'; \
   if [ \$rc -eq 0 ]; then echo 'COMPILES OK'; else echo \"nvcc FAILED rc=\$rc\"; fi; exit \$rc"
