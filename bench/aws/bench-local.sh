#!/usr/bin/env bash
# Run the same sweep phases on this machine, into the same schema as the EC2 boxes.
#
# The M2 Max is not rented, so there is no $/hour to read off a price list. It is carried
# in the sweep anyway because it is the only Metal data point that exists, and because a
# comparison that silently drops the machine you already own is not a buying guide.
#
# The imputed price written into meta.json is capex amortised at 100% utilisation:
# a 3,499 USD machine over 3 years is 3499/(3*365*24) = 0.1332 $/hr, plus ~40 W under
# this load at 0.12 $/kWh = 0.0048 $/hr. It is NOT the same kind of number as an EC2
# on-demand rate: it assumes the box is bought, kept busy, and worth nothing at the end.
# Every table that mixes the two says so.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO_ROOT"

REPS="${1:-10}"
NAME="${G16_LOCAL_NAME:-local-m2-max}"
PRICE="${G16_LOCAL_USD_HR:-0.1380}"
OUT="$REPO_ROOT/bench/results/sweep/$NAME"
mkdir -p "$OUT"

BACKENDS="${G16_LOCAL_BACKENDS:-cpu metal}"
FEATURES="--features metal"

echo "==> building"
cargo build --release -p g16-cli $FEATURES 2>&1 | tail -2

echo "==> correctness gate"
# Run once, keep the output, decide from it. Running the suite twice -- once to show and
# once to grep -- doubles the wall clock and, worse, lets the two runs disagree.
TLOG="$(mktemp)"
cargo test --release --workspace $FEATURES > "$TLOG" 2>&1 || true
grep -E '^test result' "$TLOG" | tail -20
TESTS=passed
grep -qE 'FAILED|^error\[|^error:' "$TLOG" && TESTS=FAILED
rm -f "$TLOG"
echo "    -> $TESTS"

for b in $BACKENDS; do
  echo "==> bench backend=$b"
  ./target/release/g16 bench --artifacts bench/artifacts --reps "$REPS" \
    --backend "$b" --mode both --csv "$OUT/$b.csv"
done

python3 - "$OUT/meta.json" "$NAME" "$PRICE" "$REPS" "$TESTS" <<'PY'
import json,subprocess,sys,os
out,name,price,reps,tests=sys.argv[1:6]
def sh(c):
    try: return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
    except Exception: return ""
json.dump(dict(instance_type=name, arch="arm64",
               vcpu=int(sh("sysctl -n hw.logicalcpu") or 0),
               cores=int(sh("sysctl -n hw.physicalcpu") or 0),
               mem_gib=round(int(sh("sysctl -n hw.memsize") or 0)/2**30),
               gpu="Apple M2 Max 38-core GPU", gpu_sku="M2Max",
               cpu_model=sh("sysctl -n machdep.cpu.brand_string"),
               usd_per_hour=float(price), price_basis="imputed capex, see bench-local.sh",
               region="local", reps=int(reps), tests=tests,
               git=sh("git rev-parse --short HEAD"),
               dirty=bool(sh("git status --porcelain"))),
          open(out,"w"), indent=2)
PY
echo "wrote $OUT/meta.json"
