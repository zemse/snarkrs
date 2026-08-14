#!/usr/bin/env bash
# Put the source and the prover inputs in S3 once, so thirteen boxes pull them in-region
# instead of thirteen uploads over a home connection.
#
# The artifact bundle deliberately omits circuit.r1cs and circuit_js/. Those are outputs of
# circom, not inputs to a prover, and they are 70 MB of the 164 MB for the largest variant.
# Dropping them takes the bundle from 293 MB to 114 MB and changes nothing that is measured.
#
# Source is taken from the working tree filtered through `git ls-files`, not `git archive`,
# so an uncommitted fix to the harness is what actually runs on the boxes. That is the right
# default while iterating; the run is pinned by recording `git rev-parse HEAD` and the dirty
# state in each box's meta.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO_ROOT"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "==> source bundle"
git ls-files -z | tar czf "$TMP/src.tgz" --null -T -
echo "    $(du -h "$TMP/src.tgz" | cut -f1)  ($(git ls-files | wc -l | tr -d ' ') tracked files)"

echo "==> artifact bundle"
( cd bench/artifacts
  files=(manifest.csv)
  for v in */; do v="${v%/}"
    for f in circuit.zkey circuit.wtns vkey.json r1cs-info.txt public.json proof.json; do
      [ -f "$v/$f" ] && files+=("$v/$f")
    done
  done
  tar czf "$TMP/artifacts.tgz" "${files[@]}" )
echo "    $(du -h "$TMP/artifacts.tgz" | cut -f1)"

aws_ s3 cp "$TMP/src.tgz"       "s3://$BUCKET/src.tgz"       --only-show-errors
aws_ s3 cp "$TMP/artifacts.tgz" "s3://$BUCKET/artifacts.tgz" --only-show-errors
aws_ s3 ls "s3://$BUCKET/"
echo "staged at $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' +dirty')"
