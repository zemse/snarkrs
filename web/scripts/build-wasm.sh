#!/usr/bin/env bash
# Builds the WebGPU prover to wasm and puts it in static/pkg, which the worker loads at
# runtime. Run this before `npm run build`; static/pkg is gitignored because it is fully
# reproducible from the workspace.
#
# The wasm-bindgen version check is the load-bearing part of this script. wgpu's WebGPU
# backend vendors its own generated bindings against one wasm-bindgen version. If the CLI
# that runs bindgen is a different version, the build succeeds, the page loads, and
# `requestAdapter()` quietly resolves to nothing: you get "no adapter on this machine" on a
# Mac that plainly has one. No compile error, no console warning.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

command -v wasm-pack >/dev/null || { echo "wasm-pack is not installed" >&2; exit 1; }
command -v wasm-opt  >/dev/null || { echo "wasm-opt is not installed (brew install binaryen)" >&2; exit 1; }

# The single source of truth for the pin: whatever cargo actually resolved.
want="$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/[",]/,""); print $3; exit}' "$root/Cargo.lock")"
echo "wasm-bindgen in Cargo.lock: $want"

# wasm-pack's own wasm-opt is off in crates/g16-wasm/Cargo.toml so the size before
# optimisation survives to be measured.
( cd "$root" && wasm-pack build crates/g16-wasm --target web --release \
    --out-dir pkg --out-name g16_wasm )

raw="$root/target/wasm32-unknown-unknown/release/g16_wasm.wasm"
bg="$root/crates/g16-wasm/pkg/g16_wasm_bg.wasm"

# rustc stamps the wasm-bindgen version into the schema section of its output and the CLI
# refuses to run against a section it does not understand. Comparing that stamp against
# Cargo.lock catches what the CLI cannot: a stale target/ from a build at another version.
got="$(strings -a "$raw" | grep -o '"version":"[0-9.]*"' | head -1 | cut -d'"' -f4 || true)"
[ -n "$got" ] || { echo "could not read a wasm-bindgen version out of $raw" >&2; exit 1; }
if [ "$got" != "$want" ]; then
  echo "wasm-bindgen skew: the .wasm was built against $got, Cargo.lock has $want." >&2
  echo "A skew against wgpu's vendored bindings fails at requestAdapter(), silently." >&2
  exit 1
fi
echo "wasm-bindgen stamped in the .wasm:  $got   (matches)"

before="$(wc -c < "$bg")"
wasm-opt -O -o "$bg.opt" "$bg"
mv "$bg.opt" "$bg"
after="$(wc -c < "$bg")"

rm -rf "$here/../static/pkg"
cp -R "$root/crates/g16-wasm/pkg" "$here/../static/pkg"
# wasm-pack writes a package.json into its output. Left in static/ it is a public file that
# claims this directory is an npm package, which it is not.
rm -f "$here/../static/pkg/package.json" "$here/../static/pkg/.gitignore" "$here/../static/pkg/README.md"

printf '\nrustc output          %10d bytes\n' "$(wc -c < "$raw")"
printf 'after wasm-bindgen    %10d bytes\n' "$before"
# 2.2 MB, not the 1.5 MB in design section 6. That budget was written for a product prover
# shipping one circuit; this build is 1.90 MB because merging the CPU-side optimisation work
# into main brought heavily unrolled field and MSM code into the same cdylib, and because
# this page also carries the CPU control backend. Worth 400 KB on a page whose next act is to
# download 262 MB of proving keys. The check stays, so a future regression still has to be
# noticed and argued for rather than sliding through.
printf 'after wasm-opt -O     %10d bytes   budget is 2200000\n' "$after"
[ "$after" -le 2200000 ] || { echo "over the 2.2 MB budget; see the note above before raising it." >&2; exit 1; }
