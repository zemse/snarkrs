#!/usr/bin/env bash
# Coordinate leases on an already-running T4. Start it separately through the native lifecycle.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/t4.py" "$@"
