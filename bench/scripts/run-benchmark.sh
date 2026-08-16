#!/usr/bin/env bash
# One command that benchmarks this machine, whatever this machine is.
#
#   bench/scripts/run-benchmark.sh [--reps N] [--circuits "a b c"] [--no-download]
#                                  [--snarkjs-reps N] [--skip-external]
#                                  [--out FILE] [--note TEXT]
#                                  [--no-build] [--commit SHA]
#                                  [--max-foreign PCT] [--allow-loaded]
#
# It works out what backends the box can actually run, builds only those, fetches any
# missing proving keys from S3, benchmarks every circuit it has a complete artifact set
# for against rapidsnark and snarkjs as well as our own backends, and writes one markdown
# file named after the machine. Anything the box does not have is left blank in the table
# rather than reported as a zero or an error, because a missing measurement and a
# measurement of zero are not the same claim.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$HERE/bench"
ARTIFACTS="$BENCH/artifacts"
OUTDIR="$BENCH/results/machines"
BUCKET="${G16_BUCKET:-gpu-snark-bench}"
S3_PREFIX="${G16_S3_PREFIX:-artifacts}"
REPS="${G16_REPS:-15}"
# snarkjs is wasm and takes minutes at 10^5 constraints. A median over a few slow reps
# buys the same answer for a fraction of the wall clock.
SNARKJS_REPS="${G16_SNARKJS_REPS:-3}"
DOWNLOAD=1
EXTERNAL=1
OUT_OVERRIDE=""
NOTE=""
CIRCUITS="${G16_CIRCUITS:-sha256 keccak256 tornado rsa2048 anon-aadhaar}"

while [ $# -gt 0 ]; do
  case "$1" in
    --reps)         REPS="$2"; shift 2 ;;
    --circuits)     CIRCUITS="$2"; shift 2 ;;
    --snarkjs-reps) SNARKJS_REPS="$2"; shift 2 ;;
    --no-download)  DOWNLOAD=0; shift ;;
    --skip-external) EXTERNAL=0; shift ;;
    --out)          OUT_OVERRIDE="$2"; shift 2 ;;
    --note)         NOTE="$2"; shift 2 ;;
    --max-foreign)  MAX_FOREIGN="$2"; shift 2 ;;
    --allow-loaded) ALLOW_LOADED=1; shift ;;
    --commit)       COMMIT_OVERRIDE="$2"; shift 2 ;;
    --no-build)     NO_BUILD=1; shift ;;
    -h|--help)      sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- contention guard
# A timing taken while another tenant is on the box measures the other tenant. That is not
# hypothetical here: a parallel session started partway through a run, and because the
# external provers are timed last, rapidsnark absorbed nearly all of the interference and
# came out about 2x slow while our own number, taken minutes earlier, barely moved. The
# resulting table does not look contaminated. It looks like this prover winning by a mile,
# which is the most dangerous shape a wrong benchmark can take.
#
# The obvious guard, a load average ceiling, does not work mid-run. Proving saturates every
# core by design, so the run drives the load average above any sane ceiling within seconds
# and then trips on itself. run-comparison.py gets away with checking load once at startup;
# checking it repeatedly needs a measure that can tell our own work apart from somebody
# else's.
#
# So this counts CPU belonging to processes outside our own process group. Our children
# inherit it and everyone else's work does not, which is exactly the distinction the load
# average cannot make. The threshold is in percent of a single core, so 200 means two other
# cores' worth of foreign work.
#
# 200 rather than something tighter because a desktop is never actually at zero: the window
# server, the terminal and the editor together idle around 70% of a core here, and a ceiling
# below that can never pass. This is deliberately a coarse guard. It is not trying to detect
# a stray 5% background daemon, whose effect is inside the noise anyway. It is trying to
# catch the case that actually ruined a run, which was a parallel build sitting on ten of
# the twelve cores at over 1000%.
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
MAX_FOREIGN="${MAX_FOREIGN:-200}"
ALLOW_LOADED="${ALLOW_LOADED:-0}"
LOADED_ANYWAY=0
MY_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"

foreign_cpu() {
  ps -Ao pcpu=,pgid= | awk -v me="$MY_PGID" '$2 != me { s += $1 } END { printf "%.0f", s+0 }'
}

check_load() {
  local fc; fc="$(foreign_cpu)"
  [ "$fc" -gt "$MAX_FOREIGN" ] || return 0
  if [ "$ALLOW_LOADED" = "1" ]; then
    LOADED_ANYWAY=1
    echo "    WARNING: ${fc}% of a core in use by other processes during: $1" >&2
    return 0
  fi
  echo >&2
  echo "another process is using ${fc}% of a core (ceiling ${MAX_FOREIGN}%) on this ${CORES}-core box." >&2
  echo "Stopped during: $1" >&2
  echo "A timing taken now measures whatever else is running. Wait for the box to go" >&2
  echo "idle, or pass --allow-loaded to record anyway and mark the run as suspect." >&2
  exit 3
}

# ---------------------------------------------------------------- machine identity
# Same labels bench/scripts/run-comparison.py uses, so results from the two harnesses
# file under the same name instead of forking into two naming schemes.
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 && return
  fi
  if [ "$(uname -s)" = Darwin ]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null && return
  fi
  echo ""
}

detect_machine() {
  if [ "$(uname -s)" = Linux ]; then
    tok=$(curl -s -m 1 -X PUT "http://169.254.169.254/latest/api/token" \
          -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
    if [ -n "$tok" ]; then
      it=$(curl -s -m 1 "http://169.254.169.254/latest/meta-data/instance-type" \
           -H "X-aws-ec2-metadata-token: $tok" 2>/dev/null || true)
      if [ -n "$it" ]; then
        # The EC2 instance type on its own. It is the canonical name for the machine, it is
        # what you type to rent the same box again, and it already implies the accelerator.
        # An earlier version emitted aws-g4dn.2xlarge-tesla-t4, which buried a proper slug
        # inside a compound nobody can parse: a reader cannot tell where the instance type
        # ends and the GPU begins. The GPU is recorded as its own `accelerator:` field.
        echo "$it"
        return
      fi
    fi
  fi
  g=$(detect_gpu | tr ' ' '-')
  [ -n "$g" ] && echo "$g" || hostname
}

# Lowercase throughout: these strings become filenames and table keys, and a machine
# that reports "Apple M2 Max" on one path and "apple-m2-max" on another files its results
# in two places.
MACHINE="$(detect_machine | tr '[:upper:]' '[:lower:]')"

# The commit is part of the filename, not just a line inside it. Benchmarks age out of
# being re-runnable: a rented GPU box goes away, a laptop gets replaced, a circuit gets
# regenerated. Keying the file by commit means an old machine's numbers stay on disk and
# stay attributable instead of being overwritten by whatever ran last.
#
# --commit exists because the interesting machines are the ones without a checkout. A rented
# GPU box gets a cross-compiled binary shipped to it and no source and no .git, and the first
# run on one of those had to be coaxed through with a fake `git` on PATH answering
# `rev-parse`. Shimming git to get provenance right is a bad trade, so pass the commit the
# binary was built from instead.
if [ -n "${COMMIT_OVERRIDE:-}" ]; then
  COMMIT="$COMMIT_OVERRIDE"
else
  COMMIT="$(cd "$HERE" && git rev-parse --short HEAD 2>/dev/null || echo nogit)"
  if [ -n "$(cd "$HERE" && git status --porcelain 2>/dev/null)" ]; then
    COMMIT="${COMMIT}-dirty"
  fi
fi
OUT="$OUTDIR/${MACHINE}-${COMMIT}.md"
if [ -n "$OUT_OVERRIDE" ]; then OUT="$OUT_OVERRIDE"; fi

# ---------------------------------------------------------------- backend detection
# Presence is decided by what the box can run, not by what the OS could in principle
# support: a Linux host with no NVIDIA driver must not advertise cuda.
BACKENDS="cpu"
FEATURES=""
HAVE_METAL=0
HAVE_CUDA=0
if [ "$(uname -s)" = Darwin ]; then
  BACKENDS="$BACKENDS metal"; FEATURES="metal"; HAVE_METAL=1
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  BACKENDS="$BACKENDS cuda"
  FEATURES="${FEATURES:+$FEATURES,}cuda"; HAVE_CUDA=1
fi

# ---------------------------------------------------------------- external prover detection
# Same defensive rule as the backends: a prover counts as present only if the thing that
# runs it is here right now. Absent means its column comes out blank, not that the run
# fails. rapidsnark ships as three binaries built by bench/scripts/build-rapidsnark.sh,
# and any of them can be missing independently.
RAPIDSNARK="$BENCH/bin/rapidsnark"
RAPIDSNARK_WARM="$BENCH/bin/rapidsnark-warm"
RAPIDSNARK_VERIFY="$BENCH/bin/rapidsnark-verify"
[ -x "$RAPIDSNARK" ]        || RAPIDSNARK=""
[ -x "$RAPIDSNARK_WARM" ]   || RAPIDSNARK_WARM=""
[ -x "$RAPIDSNARK_VERIFY" ] || RAPIDSNARK_VERIFY=""

# snarkjs is usually installed by a node package manager whose bin directory is on an
# interactive PATH and nowhere else, so check the usual install locations too.
SNARKJS=""
if command -v snarkjs >/dev/null 2>&1; then
  SNARKJS="$(command -v snarkjs)"
else
  for cand in "$HOME/Library/pnpm/snarkjs" "$HOME/.local/share/pnpm/snarkjs" \
              "$HOME/.npm-global/bin/snarkjs" "/usr/local/bin/snarkjs" \
              "/opt/homebrew/bin/snarkjs"; do
    if [ -x "$cand" ]; then SNARKJS="$cand"; break; fi
  done
fi

# The list handed to the renderer: which external columns were attempted here. A prover
# not in this list is blank in the table; one in it that produced nothing shows `-`.
PROVERS=""
if [ "$EXTERNAL" -eq 1 ]; then
  if [ -n "$RAPIDSNARK" ] || [ -n "$RAPIDSNARK_WARM" ]; then PROVERS="$PROVERS rapidsnark"; fi
  if [ -n "$SNARKJS" ]; then PROVERS="$PROVERS snarkjs"; fi
fi
PROVERS="$(echo "$PROVERS" | xargs || true)"

# snarkjs never gets more reps than the run asked for overall.
if [ "$SNARKJS_REPS" -gt "$REPS" ]; then SNARKJS_REPS="$REPS"; fi

log "machine: $MACHINE"
echo "    backends available: $BACKENDS"
echo "    cargo features:     ${FEATURES:-<none>}"
echo "    external provers:   ${PROVERS:-<none>}"
echo "    rapidsnark:         ${RAPIDSNARK:-<missing>}"
echo "    rapidsnark-warm:    ${RAPIDSNARK_WARM:-<missing>}"
echo "    snarkjs:            ${SNARKJS:-<missing>}"
echo "    reps:               $REPS (snarkjs $SNARKJS_REPS)"

# ---------------------------------------------------------------- build
# --no-build for the same reason as --commit: a box holding a prebuilt binary has no source
# to build from, and running cargo there either fails or, worse, silently rebuilds something
# other than the binary that is about to be measured.
G16="$HERE/target/release/g16"
if [ "${NO_BUILD:-0}" = "1" ]; then
  log "using the prebuilt binary, not building"
  [ -x "$G16" ] || { echo "--no-build given but $G16 is not executable" >&2; exit 1; }
else
  log "building"
  if [ -n "$FEATURES" ]; then
    ( cd "$HERE" && cargo build --release -p g16-cli --features "$FEATURES" )
  else
    ( cd "$HERE" && cargo build --release -p g16-cli )
  fi
fi

# ---------------------------------------------------------------- artifacts
# Proving keys are hundreds of megabytes and are not in git. Fetch only what is missing,
# and only the two files proving actually reads: the .r1cs and the wasm witness generator
# are build inputs, not prover inputs.
fetch() {
  # Split, not one `local`: bash expands every argument to `local` before it assigns any
  # of them, so "$c" and "$f" are still unset while dest is being built.
  local c="$1"
  local f="$2"
  local dest="$ARTIFACTS/$c/$f"
  [ -s "$dest" ] && return 0
  [ "$DOWNLOAD" -eq 1 ] || { echo "    missing (download disabled): $c/$f"; return 1; }
  mkdir -p "$ARTIFACTS/$c"
  echo "    fetching s3://$BUCKET/$S3_PREFIX/$c/$f"
  if command -v aws >/dev/null 2>&1 && aws s3 cp "s3://$BUCKET/$S3_PREFIX/$c/$f" "$dest" --only-show-errors 2>/dev/null; then
    return 0
  fi
  # Fall back to unsigned HTTPS so a rented benchmark box needs no AWS credentials.
  curl -fsSL --retry 3 -o "$dest" \
    "https://${BUCKET}.s3.amazonaws.com/${S3_PREFIX}/${c}/${f}" && return 0
  rm -f "$dest"; echo "    could not fetch $c/$f"; return 1
}

log "artifacts"
READY=""
for c in $CIRCUITS; do
  # All five: the two big ones proving reads, plus the three small ones the bench
  # harness needs to verify a proof and label it with a constraint count.
  ok=1
  for f in circuit.zkey circuit.wtns vkey.json public.json r1cs-info.txt; do
    fetch "$c" "$f" || ok=0
  done
  if [ "$ok" -eq 1 ]; then
    READY="$READY $c"
    echo "    ready: $c"
  else
    echo "    skipping: $c"
  fi
done
READY="$(echo "$READY" | xargs || true)"
[ -n "$READY" ] || { echo "no circuits available, nothing to benchmark" >&2; exit 1; }

# ---------------------------------------------------------------- benchmark
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for b in $BACKENDS; do
  for c in $READY; do
    log "$c on $b"
    check_load "$c on $b"
    "$G16" bench --artifacts "$ARTIFACTS" --variant "$c" --backend "$b" \
                 --mode both --reps "$REPS" --csv "$TMP/${b}-${c}.csv" || \
      echo "    $c on $b failed, leaving it out of the table"
  done
done

# The external provers write into the same directory in the same schema, so the renderer
# reads all of it as one pile of rows and tells the tools apart by the `prover` column.
# A failure here must not take the run down: our own numbers are still worth writing out.
if [ -n "$PROVERS" ]; then
  for c in $READY; do
    log "$c on external provers ($PROVERS)"
    check_load "$c on external provers"
    python3 "$BENCH/scripts/bench_external.py" \
      --artifacts "$ARTIFACTS" --variant "$c" --csv "$TMP/external-${c}.csv" \
      --reps "$REPS" --snarkjs-reps "$SNARKJS_REPS" \
      --rapidsnark "$RAPIDSNARK" --rapidsnark-warm "$RAPIDSNARK_WARM" \
      --rapidsnark-verify "$RAPIDSNARK_VERIFY" --g16 "$G16" --snarkjs "$SNARKJS" \
      || echo "    external provers on $c failed, leaving those cells empty"
  done
else
  log "no external provers found, rapidsnark and snarkjs columns will be blank"
fi

# ---------------------------------------------------------------- report
mkdir -p "$(dirname "$OUT")"
# A run that was allowed through the load guard has to say so in the file itself. The
# person reading the table months from now is not the person who typed --allow-loaded.
if [ "$LOADED_ANYWAY" = "1" ]; then
  NOTE="SUSPECT: recorded with --allow-loaded while other processes were using more than ${MAX_FOREIGN}% of a core on this ${CORES}-core box. Timings here include whatever else was running and should not be compared against a run taken on an idle machine.${NOTE:+ }${NOTE}"
fi

# Detected once and recorded, so the file says what it ran on rather than leaving a reader
# to decode the slug.
GPU_NAME="$(detect_gpu)"
if [ "$(uname -s)" = Darwin ]; then
  CPU_NAME="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")"
else
  CPU_NAME="$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "")"
fi
# On a Mac detect_gpu falls back to the CPU brand string, which would make the two fields
# duplicates and imply a discrete accelerator that is not there.
[ "$GPU_NAME" = "$CPU_NAME" ] && GPU_NAME=""

python3 "$BENCH/scripts/render_machine.py" \
  --csv-dir "$TMP" --machine "$MACHINE" --reps "$REPS" \
  --gpu "$GPU_NAME" --cpu "$CPU_NAME" \
  --snarkjs-reps "$SNARKJS_REPS" \
  --backends "$BACKENDS" --provers "$PROVERS" --commit "$COMMIT" \
  --note "$NOTE" \
  --out "$OUT"

log "wrote $OUT"
cat "$OUT"
