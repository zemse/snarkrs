#!/usr/bin/env bash
# Runs on the rented host: an interleaved A/B of every snarkrs build in the bundle, warm, all
# threads. Each round runs every circuit under every build, the build order reversing each
# round; then one proof per build and circuit, then rapidsnark once for reference.
# Usage: matrix-ab-guest.sh ROOT REPS ROUNDS CIRCUIT...
set -uo pipefail

root=$1 reps=$2 rounds=$3
shift 3
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

builds=()
for b in generic adx asm; do
  [[ -x $root/bin/snarkrs-$b ]] && builds+=("$b")
done
echo "${builds[*]}" > "$out/build.txt"
log "host $(nproc) cpus, builds ${builds[*]}, reps $reps, rounds $rounds"

failed=0
check() {
  [[ $(cat "$1") == 0 ]] || { failed=1; log "FAILED $1"; }
}

for ((r = 1; r <= rounds; r++)); do
  order=("${builds[@]}")
  if ((r % 2 == 0)); then
    order=()
    for ((i = ${#builds[@]} - 1; i >= 0; i--)); do order+=("${builds[i]}"); done
  fi
  for c in "$@"; do
    d=$out/$c
    mkdir -p "$d"
    for b in "${order[@]}"; do
      log "round $r $c $b bench"
      peak "$d/$b-r$r.rss" "$root/bin/snarkrs-$b" bench --artifacts "$root/artifacts" --variant "$c" \
        --mode warm --reps "$reps" --csv "$d/$b-r$r.csv" > "$d/$b-r$r.log" 2>&1
      echo $? > "$d/$b-r$r.exit"
      check "$d/$b-r$r.exit"
    done
  done
done

for c in "$@"; do
  a=$root/artifacts/$c
  d=$out/$c
  for b in "${builds[@]}"; do
    log "$c $b proof"
    "$root/bin/snarkrs-$b" groth16 prove "$a/circuit.zkey" "$a/circuit.wtns" \
      "$d/$b-proof.json" "$d/$b-public.json" > "$d/$b-prove.log" 2>&1
    echo $? > "$d/$b-prove.exit"
    check "$d/$b-prove.exit"
  done

  log "$c rapidsnark"
  peak "$d/rs.rss" "$root/bin/rapidsnark-warm" "$a/circuit.zkey" "$a/circuit.wtns" \
    "$d/rs-last-proof.json" "$d/rs-last-public.json" "$reps" "$a/vkey.json" \
    --proof-dir "$d/rs-proofs" > "$d/rs.out" 2> "$d/rs.err"
  echo $? > "$d/rs.exit"
  check "$d/rs.exit"
done

log "done failed=$failed"
echo "$failed" > "$out/DONE"
