#!/usr/bin/env bash
# Publishes the precomputed libraries (gto/static/flop_library: the heads-up and 3-way indexes, each spot's entries and
# report.json) to https://www.michael-han.com/gto-solver-library/, the GitHub Pages site of the
# michaellhan/gto-solver-library repo, as one commit with no history: each publish replaces the last (a force push of
# a fresh repo), so the repo holds a single copy of the library. (Copied into the site repo, every rebuilt entry stayed
# in its history: 1.08 GB by October 2026, when the library was 48 MB of the 87 MB published.) scripts/deploy.sh runs
# it before publishing the page, whose LIB_BASE points there (same origin: no CORS).
#   scripts/deploy_library.sh DIR   assembles the library in DIR (emptied first) and publishes it; nothing is pushed
#                                   when the published commit already holds the same files.
# BUILD_ONLY (any value, as deploy.sh's) only assembles it. The repo's Pages serve its main branch's root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO=michaellhan/gto-solver-library
OUT="${1:?usage: scripts/deploy_library.sh DIR}"
SRC="$HERE/gto/static/flop_library"
[ -f "$SRC/index.json" ] || { echo "No library to publish: $SRC/index.json is missing." >&2; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
cp -R "$SRC/." "$OUT/"
find "$OUT" -type f ! -name '*.json' ! -name '*.bin' -delete   # (a running build's .tmp files)
touch "$OUT/.nojekyll"   # served as they are (no Jekyll pass over ~13k files)
if [ -n "${BUILD_ONLY:-}" ]; then echo "Built $OUT (BUILD_ONLY: not published)."; exit 0; fi
cd "$OUT"
git init -q -b main
git add -A
if [ "$(gh api "repos/$REPO/commits/main" --jq .commit.tree.sha 2>/dev/null || true)" = "$(git write-tree)" ]; then
  echo "Library already up to date."
  exit 0
fi
git commit -q -m "library: gto-solver $(cd "$HERE" && git rev-parse --short HEAD), $(find . -name '*.bin' -not -path './.git/*' | wc -l | tr -d ' ') entries"
git push -q -f "https://github.com/$REPO.git" main
echo "Library published. GitHub Pages will update https://www.michael-han.com/gto-solver-library/ in about a minute."
