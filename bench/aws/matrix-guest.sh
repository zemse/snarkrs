#!/usr/bin/env bash
# Runs on the rented host: every circuit, our CPU prover then rapidsnark, warm, all threads.
# Usage: matrix-guest.sh ROOT REPS CIRCUIT...
set -uo pipefail

root=$1 reps=$2
shift 2
out=$root/results
mkdir -p "$out"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$out/progress.log"; }

# Peak RSS of one child, recorded next to its log. GNU time is not on the stock image.
peak() {
  local rss=$1
  shift
  python3 - "$rss" "$@" <<'EOF'
import resource, subprocess, sys
r = subprocess.run(sys.argv[2:])
open(sys.argv[1], "w").write(f"{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n")
sys.exit(r.returncode)
EOF
}

lscpu > "$out/lscpu.txt"
cp /proc/cpuinfo "$out/cpuinfo.txt"
nproc > "$out/nproc.txt"
free -b > "$out/free.txt"
uname -a > "$out/uname.txt"
cp /etc/os-release "$out/os-release.txt"

flags=$(grep -m1 '^flags' /proc/cpuinfo)
if [[ " $flags " == *" adx "* && " $flags " == *" bmi2 "* ]]; then
  build=adx
else
  build=generic
fi
echo "$build" > "$out/build.txt"
snarkrs=$root/bin/snarkrs-$build
log "host $(nproc) cpus, build $build, reps $reps"

failed=0
for c in "$@"; do
  a=$root/artifacts/$c
  d=$out/$c
  mkdir -p "$d"

  log "$c ours bench"
  peak "$d/ours.rss" "$snarkrs" bench --artifacts "$root/artifacts" --variant "$c" \
    --mode warm --reps "$reps" --csv "$d/ours.csv" > "$d/ours.log" 2>&1
  echo $? > "$d/ours.exit"

  log "$c ours proof"
  "$snarkrs" groth16 prove "$a/circuit.zkey" "$a/circuit.wtns" \
    "$d/ours-proof.json" "$d/ours-public.json" > "$d/ours-prove.log" 2>&1
  echo $? > "$d/ours-prove.exit"

  log "$c rapidsnark"
  peak "$d/rs.rss" "$root/bin/rapidsnark-warm" "$a/circuit.zkey" "$a/circuit.wtns" \
    "$d/rs-last-proof.json" "$d/rs-last-public.json" "$reps" "$a/vkey.json" \
    --proof-dir "$d/rs-proofs" > "$d/rs.out" 2> "$d/rs.err"
  echo $? > "$d/rs.exit"

  for e in "$d"/*.exit; do
    [[ $(cat "$e") == 0 ]] || { failed=1; log "$c FAILED $(basename "$e" .exit)"; }
  done
done

log "done failed=$failed"
echo "$failed" > "$out/DONE"
