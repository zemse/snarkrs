#!/usr/bin/env bash
# Stop apt's own timers on a box that times anything, and keep them stopped.
#
# Ubuntu's apt-daily-upgrade.timer starts unattended-upgrade at a random time of day; on
# 2026-10-27 it ran mid-sweep and tripped run-benchmark.sh's contention guard. Every driver
# that launches a box embeds this file in its user-data ahead of its own apt work (see
# `quiet_apt_userdata` in lib.sh), so it also never waits on the dpkg lock. For a box that
# is already up, run it once: `ssh HOST 'sudo bash -s' < bench/aws/quiet-apt.sh`.
#
# Masking outlives a stop and start, which a user-data script does not: it runs on the
# first boot only. A running upgrade is stopped, not killed outright: unattended-upgrade
# finishes the package it is on when it gets SIGTERM, so dpkg is not left half done.
# /var/lib/g16-apt-quiet is written only once the lock is free, for a driver to wait on.

systemctl stop apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service \
  2>/dev/null || true
systemctl mask apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service \
  2>/dev/null || true
pkill -TERM -f /usr/bin/unattended-upgrade 2>/dev/null || true

g16_apt_busy() {
  if command -v fuser >/dev/null 2>&1; then
    fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock >/dev/null 2>&1
  else
    pgrep -f '/usr/bin/unattended-upgrade|/usr/lib/apt/apt.systemd.daily' >/dev/null 2>&1
  fi
}
g16_apt_waited=0
while g16_apt_busy && [ "$g16_apt_waited" -lt 900 ]; do
  sleep 5
  g16_apt_waited=$((g16_apt_waited + 5))
done
if g16_apt_busy; then
  echo "quiet-apt: the dpkg lock is still held after ${g16_apt_waited}s" >&2
else
  DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>/dev/null || true
  touch /var/lib/g16-apt-quiet
fi
