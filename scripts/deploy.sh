#!/usr/bin/env bash
# Publishes the solver to https://www.michael-han.com/gto-solver/ by copying it into
# the michaellhan/michaellhan.github.io repo (folder: gto-solver/) and pushing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
gh repo clone michaellhan/michaellhan.github.io "$WORK/site" -- -q --depth 1
rm -rf "$WORK/site/gto-solver"
mkdir -p "$WORK/site/gto-solver/gto"
cp "$HERE/gto/static/index.html" "$HERE/gto/static/worker.js" "$WORK/site/gto-solver/"
cp "$HERE"/gto/*.py "$WORK/site/gto-solver/gto/"
# Stamp the worker's cache-busting version with its content hash, so a changed worker.js is never
# served from a browser's 4h-cached copy (which would load an outdated list of solver files).
WV="$(shasum "$HERE/gto/static/worker.js" | cut -c1-10)"
sed -i.bak -E "s/^const WORKER_VERSION = \"[^\"]*\";/const WORKER_VERSION = \"$WV\";/" "$WORK/site/gto-solver/index.html"
rm "$WORK/site/gto-solver/index.html.bak"
grep -q "const WORKER_VERSION = \"$WV\";" "$WORK/site/gto-solver/index.html"
cd "$WORK/site"
git add -A
if git diff --cached --quiet; then echo "Site already up to date."; exit 0; fi
git commit -q -m "gto-solver: deploy $(cd "$HERE" && git rev-parse --short HEAD)"
git push -q origin main
echo "Deployed. GitHub Pages will update https://www.michael-han.com/gto-solver/ in about a minute."
