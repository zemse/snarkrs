#!/usr/bin/env bash
# Deploys the page to Vercel from this machine, wasm and all.
#
# Why from this machine and not from a git push: static/pkg is gitignored, and building it
# needs the Rust toolchain, wasm-pack and binaryen, none of which are in Vercel's build
# image. A git-linked deploy would therefore clone a checkout with no prover in it, build
# cleanly, and serve a page that 404s on g16_wasm_bg.wasm at init(). The CLI uploads the
# working tree instead, and .vercelignore is what keeps static/pkg in that upload.
#
# The wasm is rebuilt every time rather than reused, because the failure this guards against
# is a stale prover shipped next to fresh page code, which looks exactly like a working
# deploy. cargo caches, so an unchanged workspace costs seconds.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/.."

command -v vercel >/dev/null || { echo "vercel CLI is not installed (npm i -g vercel)" >&2; exit 1; }
vercel whoami >/dev/null 2>&1 || { echo "not logged in: run 'vercel login'" >&2; exit 1; }

./scripts/build-wasm.sh

# Belt and braces on top of build-wasm.sh's own checks: the upload is only as good as what
# is on disk at this moment.
[ -f static/pkg/g16_wasm_bg.wasm ] || { echo "static/pkg/g16_wasm_bg.wasm is missing" >&2; exit 1; }

# --prod so the deployment takes the project's domain. A bare `vercel` makes a preview URL
# with a random slug, which is not the thing anyone means by "deploy the site".
exec vercel deploy --prod "$@"
