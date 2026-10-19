#!/usr/bin/env bash
# Run the ethproofs client-side-proving benchmark against this prover.
#
#   bench/scripts/csp-fetch.sh              # once: zkeys + circuit sources
#   bench/scripts/csp-bench.sh              # every backend this build has
#   bench/scripts/csp-bench.sh --backend cpu --reps 5
#   bench/scripts/csp-bench.sh --target sha256 --input-size 128
#   SNARKJS_VERIFY=1 bench/scripts/csp-bench.sh     # also check proofs with snarkjs
#
# Writes one `{target}_{size}_g16_{backend}_metrics.json` per variant, in the schema the
# upstream collector reads, plus a `_breakdown.json` beside it carrying the phase split
# the upstream schema has nowhere to put. With SNARKJS_VERIFY=1 each variant's warm-up
# proof also goes through `snarkjs groth16 verify` before it is timed, and a reject
# stops the run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

CSP_BIN="bench/csp/target/release/snarkrs-csp"
OUT="bench/results/csp/metrics"
REPS=10
MEM_REPS=10
BACKENDS=""
FEATURES="${FEATURES:-metal}"
SNARKJS_VERIFY="${SNARKJS_VERIFY:-}"
SELECT=()

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)   BACKENDS="$BACKENDS $2"; shift 2 ;;
    --reps)      REPS="$2"; shift 2 ;;
    --mem-reps)  MEM_REPS="$2"; shift 2 ;;
    --out-dir)   OUT="$2"; shift 2 ;;
    --features)  FEATURES="$2"; shift 2 ;;
    --target)     SELECT+=(--target "$2"); shift 2 ;;
    --input-size) SELECT+=(--input-size "$2"); shift 2 ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# `metal` on macOS, `cuda` where there is a driver, cpu everywhere. Building with a
# feature the host cannot support fails at link time, so the default follows the host.
if [ -z "$BACKENDS" ]; then
  BACKENDS="cpu"
  case "$FEATURES" in *metal*) BACKENDS="$BACKENDS metal" ;; esac
  case "$FEATURES" in *cuda*)  BACKENDS="$BACKENDS cuda"  ;; esac
fi

ORACLE=()
if [ -n "$SNARKJS_VERIFY" ]; then
  if ! SNARKJS="$(command -v snarkjs)"; then
    echo "SNARKJS_VERIFY is set but snarkjs is not on PATH" >&2
    exit 1
  fi
  ORACLE=(--snarkjs "$SNARKJS")
fi

echo "=== building snarkrs-csp (features: $FEATURES) ==="
( cd bench/csp && cargo build --release ${FEATURES:+--features "$FEATURES"} )

# The vkey and public signals snarkjs checks against, regenerated so they match this build.
if [ -n "$SNARKJS_VERIFY" ]; then
  echo
  echo "=== vkey.json and public.json for the snarkjs check ==="
  "$CSP_BIN" artifacts ${SELECT[@]+"${SELECT[@]}"} \
    | grep --line-buffered -v -e '^Generating witness' -e 'allocating and retrying'
fi

for backend in $BACKENDS; do
  echo
  echo "=== $backend: $REPS timed reps, $MEM_REPS memory samples${SNARKJS_VERIFY:+, snarkjs check} ==="
  # The witness generator narrates every call on stdout and there is no flag to stop it.
  "$CSP_BIN" bench \
      --backend "$backend" --reps "$REPS" --mem-reps "$MEM_REPS" --out-dir "$OUT" \
      ${SELECT[@]+"${SELECT[@]}"} ${ORACLE[@]+"${ORACLE[@]}"} \
    | grep --line-buffered -v -e '^Generating witness' -e 'allocating and retrying'
done

echo
echo "=== comparison against the published circom row ==="
python3 bench/scripts/csp_report.py --metrics "$OUT"
