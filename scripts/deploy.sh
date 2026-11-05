#!/usr/bin/env bash
# Publishes the solver to https://www.michael-han.com/gto-solver/ by copying it into
# the michaellhan/michaellhan.github.io repo (folder: gto-solver/) and pushing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# BUILD_ONLY=<dir> builds the site into <dir>/site/gto-solver (as published: stamped versions, each view's URL) and
# stops there, without the checks, cloning or pushing: to try the static site locally, e.g.
#   BUILD_ONLY=/tmp/gto scripts/deploy.sh && python3 -m http.server -d /tmp/gto/site 8000   (then /gto-solver/hands/)
if [ -n "${BUILD_ONLY:-}" ]; then WORK="$BUILD_ONLY"; SKIP_CHECKS=1; mkdir -p "$WORK/site"; else WORK="$(mktemp -d)"; fi
# Published from an up-to-date checkout only: one behind origin/main would take back what was deployed since.
if [ -z "${BUILD_ONLY:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  git -C "$HERE" fetch -q origin main
  if ! git -C "$HERE" merge-base --is-ancestor origin/main HEAD; then
    echo "Deploy aborted: this checkout ($(git -C "$HERE" rev-parse --short HEAD)) is behind origin/main; rebase or pull first." >&2
    exit 1
  fi
fi

# Checks before anything is published: the unit tests, the page's JavaScript tests (when node is
# installed) and the quick sanity sweep (~3 min, rare preflop lines included; exits non-zero
# when a bug flag other than the info ones reproduces). SKIP_CHECKS=1 skips them (emergencies only);
# DRY_RUN=1 runs them and stops before cloning or pushing. PYTHON overrides the interpreter (default:
# the repo's .venv, else python3).
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HERE/.venv/bin/python" ]; then PYTHON="$HERE/.venv/bin/python"; else PYTHON=python3; fi
fi
if [ "${SKIP_CHECKS:-0}" = "1" ]; then
  echo "SKIP_CHECKS=1: skipping the unit tests and the quick sanity sweep."
else
  echo "Running unit tests (scripts/run_tests.py: unittest, each module in its own process)..."
  # a few tests run solves on a time budget, so a heavily loaded machine can fail one now and then; a
  # real bug fails again, so a failure is retried once before it blocks the deploy
  if ! (cd "$HERE" && "$PYTHON" scripts/run_tests.py); then
    echo "Unit tests failed; retrying once (time-budgeted tests can fail on a loaded machine)..."
    if ! (cd "$HERE" && "$PYTHON" scripts/run_tests.py); then
      echo "Deploy aborted: unit tests failed twice. Fix them, or SKIP_CHECKS=1 scripts/deploy.sh in an emergency." >&2
      exit 1
    fi
  fi
  if command -v node >/dev/null 2>&1; then
    echo "Running the page's JavaScript tests (node --test)..."
    if ! (cd "$HERE" && node --test 'tests/js/*.test.js' >/dev/null); then
      echo "Deploy aborted: the page's JavaScript tests failed (node --test 'tests/js/*.test.js'). Fix them, or SKIP_CHECKS=1 scripts/deploy.sh in an emergency." >&2
      exit 1
    fi
  else
    echo "node not found: skipping the page's JavaScript tests."
  fi
  echo "Running the quick sanity sweep..."
  if ! (cd "$HERE" && "$PYTHON" scripts/sanity_sweep.py --quick --out "$WORK/sanity_report.md"); then
    echo "Deploy aborted: the quick sanity sweep found bug flags (or failed); report in $WORK/sanity_report.md." >&2
    echo "Fix them, or SKIP_CHECKS=1 scripts/deploy.sh in an emergency." >&2
    exit 1
  fi
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: stopping before cloning or pushing the site."
  exit 0
fi

if [ -z "${BUILD_ONLY:-}" ]; then gh repo clone michaellhan/michaellhan.github.io "$WORK/site" -- -q --depth 1; fi
rm -rf "$WORK/site/gto-solver"
mkdir -p "$WORK/site/gto-solver/gto"
cp "$HERE/gto/static/index.html" "$HERE/gto/static/worker.js" "$HERE/gto/static/store.js" "$HERE/gto/static/sw.js" "$HERE/gto/static/health.json" "$WORK/site/gto-solver/"
# The modules the page loads on demand, one a line: the constant it loads the module with, then the files whose hash that
# is (the module first; a module that loads another hashes both). Each is copied beside the page and its constant stamped
# below, so a changed file is never served from a browser's cached copy. Removing a module is removing its line, and the
# constant with its code from the page.
MODULES="
HP_VERSION help.js
HN_VERSION hands.js
"
while read -r constant module _; do
  [ -z "$constant" ] || cp "$HERE/gto/static/$module" "$WORK/site/gto-solver/"
done <<< "$MODULES"
# the worker's Python code and treys in one zip (scripts/build_bundle.py); the separate gto/*.py files stay for pages
# cached from before the bundle (index.html is cached up to 10 minutes, with the old worker's version)
"$PYTHON" "$HERE/scripts/build_bundle.py" "$WORK/site/gto-solver/solver.zip" >/dev/null
cp "$HERE"/gto/*.py "$WORK/site/gto-solver/gto/"
# Stamp the worker's cache-busting version with its content hash (and solver.zip's, which the worker fetches as
# solver.zip?v=<this> and keeps in the Cache API under that URL), and each module's, so a changed file is never served
# from a browser's cached copy. Before the library: a stamp that fails stops the deploy with nothing published.
stamp() {
  sed -i.bak -E "s/^const $1 = \"[^\"]*\";/const $1 = \"$2\";/" "$WORK/site/gto-solver/index.html"
  rm "$WORK/site/gto-solver/index.html.bak"
  grep -q "const $1 = \"$2\";" "$WORK/site/gto-solver/index.html"
}
WV="$(cat "$HERE/gto/static/worker.js" "$WORK/site/gto-solver/solver.zip" | shasum | cut -c1-10)"
stamp WORKER_VERSION "$WV"
while read -r constant files; do
  [ -z "$constant" ] && continue
  hash="$(cd "$HERE/gto/static" && cat $files | shasum | cut -c1-10)"   # ($files unquoted: a list of names)
  stamp "$constant" "$hash"
done <<< "$MODULES"
# the storage layer the page loads before its own script (store.js: its <script src>'s ?v=)
TV="$(shasum "$HERE/gto/static/store.js" | cut -c1-10)"
sed -i.bak -E "s#<script src=\"store\.js\?v=[^\"]*\"></script>#<script src=\"store.js?v=$TV\"></script>#" "$WORK/site/gto-solver/index.html"
rm "$WORK/site/gto-solver/index.html.bak"
grep -q "<script src=\"store.js?v=$TV\"></script>" "$WORK/site/gto-solver/index.html"
# the precomputed libraries (gto/flop_library.py, gto/mw_library.py: the indexes and one small file per spot and flop,
# fetched by the worker when its spot comes up) are published first to their own site, gto-solver-library/ beside
# gto-solver/ (scripts/deploy_library.sh: one commit with no history, so rebuilt entries don't pile up in the site
# repo's), and the page's LIB_BASE points there. (BUILD_ONLY writes it to <dir>/site/gto-solver-library.)
if [ -f "$HERE/gto/static/flop_library/index.json" ]; then
  if [ -n "${BUILD_ONLY:-}" ]; then "$HERE/scripts/deploy_library.sh" "$WORK/site/gto-solver-library"; else "$HERE/scripts/deploy_library.sh" "$WORK/library"; fi
  sed -i.bak -E 's#^const LIB_BASE = "[^"]*";#const LIB_BASE = "../gto-solver-library/";#' "$WORK/site/gto-solver/index.html"
  rm "$WORK/site/gto-solver/index.html.bak"
  grep -q '^const LIB_BASE = "../gto-solver-library/";' "$WORK/site/gto-solver/index.html"
fi
# Each view's URL (/gto-solver/hands/...: the page's ROUTES, gto/routes.py) is a copy
# of the stamped page with a <base href="../"> pointing its relative URLs at gto-solver/ (GitHub Pages has no rewrites).
# (No 404.html: GitHub Pages only uses the site root's, which isn't this folder's.)
(cd "$HERE" && "$PYTHON" -m gto.routes "$WORK/site/gto-solver")
if [ -n "${BUILD_ONLY:-}" ]; then echo "Built $WORK/site/gto-solver (BUILD_ONLY: not published)."; exit 0; fi
cd "$WORK/site"
git add -A
if git diff --cached --quiet; then echo "Site already up to date."; exit 0; fi
git commit -q -m "gto-solver: deploy $(cd "$HERE" && git rev-parse --short HEAD)"
git push -q origin main
echo "Deployed. GitHub Pages will update https://www.michael-han.com/gto-solver/ in about a minute."
