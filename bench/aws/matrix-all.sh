#!/usr/bin/env bash
# Runs matrix.py for each instance type in turn, never two at once.
# Usage: [MAX_MINUTES=N] matrix-all.sh OUTDIR TYPE...
# Graviton types take OUTDIR/bundle-arm64, the rest OUTDIR/bundle.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=$1
shift
mkdir -p "$out/summary" "$out/logs"

for t in "$@"; do
  echo "######## $t start $(date -u +%H:%MZ)"
  python3 "$here/matrix.py" run --instance-type "$t" --bundle "$out/bundle" \
    --arm64-bundle "$out/bundle-arm64" --results "$out/results" --summary "$out/summary/$t.json" \
    --reps 10 --max-minutes "${MAX_MINUTES:-120}" \
    > "$out/logs/$t.log" 2>&1
  echo "######## $t exit $? $(date -u +%H:%MZ)"
done
echo "######## all done $(date -u +%H:%MZ)"
