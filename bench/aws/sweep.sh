#!/usr/bin/env bash
# Run the whole cross-machine sweep, N boxes at a time, and guarantee nothing survives it.
#
#   sweep.sh [reps] [max-parallel]
#
# Concurrency is capped because the two EC2 vCPU quotas (Standard, and G-and-VT) are
# account-wide and this IAM user cannot read them -- servicequotas:GetServiceQuota is
# denied, so the only way to learn a limit here is to hit it. bench-machine.sh retries a
# VcpuLimitExceeded rather than dropping the machine, so a too-high cap costs wall clock
# instead of losing a row.
#
# CPU and GPU boxes draw on separate quotas, so they are run as separate waves; that way a
# G-quota wall cannot block the CPU half of the sweep.
#
# The teardown at the end is unconditional and runs even if every lane failed. It is the
# third of three independent guarantees, after each lane's EXIT trap and each instance's
# own shutdown timer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

REPS="${1:-10}"
PAR="${2:-4}"

CPU_BOXES=(c7g.xlarge c8g.xlarge c7i.xlarge c7a.xlarge c7a.2xlarge c8g.4xlarge c7a.4xlarge)
GPU_BOXES=(g5g.xlarge g4dn.xlarge g4dn.2xlarge g6.xlarge g5.xlarge g6e.xlarge)
[ -n "${G16_ONLY:-}" ] && { CPU_BOXES=(); GPU_BOXES=(); for t in $G16_ONLY; do
    if [ -n "$(machine_field "$t" gpu)" ]; then GPU_BOXES+=("$t"); else CPU_BOXES+=("$t"); fi; done; }

teardown() {
  echo
  echo "================ TEARDOWN ================"
  "$HERE/terminate.sh" || true
}
trap teardown EXIT INT TERM

wave() {
  local name="$1"; shift
  local boxes=("$@")
  [ ${#boxes[@]} -eq 0 ] && return 0
  echo "=== wave: $name (${#boxes[@]} boxes, $PAR at a time) ==="
  local running=0
  for t in "${boxes[@]}"; do
    "$HERE/bench-machine.sh" "$t" "$REPS" >"$REPO_ROOT/bench/results/sweep/$t.out" 2>&1 &
    running=$((running+1))
    echo "  started $t (pid $!)"
    if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
  wait
  echo "=== wave $name complete ==="
  for t in "${boxes[@]}"; do
    printf '  %-14s %s\n' "$t" "$(tail -1 "$REPO_ROOT/bench/results/sweep/$t.out" 2>/dev/null)"
  done
}

mkdir -p "$REPO_ROOT/bench/results/sweep"
wave CPU "${CPU_BOXES[@]}"
wave GPU "${GPU_BOXES[@]}"

echo
echo "=== results collected ==="
find "$REPO_ROOT/bench/results/sweep" -name '*.csv' | sort | while read -r f; do
  printf '  %-52s %s rows\n' "${f#$REPO_ROOT/}" "$(($(wc -l < "$f") - 1))"
done
