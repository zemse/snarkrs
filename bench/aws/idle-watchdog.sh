#!/usr/bin/env bash
# Job leases avoid mistaking a quiet GPU stage for an idle machine.
set -euo pipefail

shutdown_reason() {
  local uptime="$1" now="$2" last="$3" done_at="$4" idle="$5" hard="$6" grace="$7"
  if (( uptime >= hard )); then
    echo hard-limit
  elif (( done_at > 0 && now - done_at >= grace )); then
    echo job-finished
  elif (( now - last >= idle )); then
    echo idle
  fi
}

main() {
  local state="${G16_WATCHDOG_STATE:-/var/lib/g16-watchdog}" now uptime last done_at reason
  local uptime_file="${G16_WATCHDOG_UPTIME:-/proc/uptime}"
  local shutdown="${G16_WATCHDOG_SHUTDOWN:-/sbin/shutdown}" idle hard grace
  read -r idle hard grace < "$state/config"
  for value in "$idle" "$hard" "$grace"; do
    [[ "$value" =~ ^[1-9][0-9]{0,8}$ ]] || { echo 'invalid watchdog configuration' >&2; return 1; }
  done
  now=$(date +%s)
  uptime=$(cut -d. -f1 "$uptime_file")
  last=$(stat -c %Y "$state/lease")
  done_at=0
  if [[ -f "$state/done" ]]; then
    done_at=$(stat -c %Y "$state/done")
  fi
  reason=$(shutdown_reason "$uptime" "$now" "$last" "$done_at" "$idle" "$hard" "$grace")
  if [[ -n "$reason" ]]; then
    logger -t g16-watchdog "poweroff: $reason"
    "$shutdown" -h now
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
