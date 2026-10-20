#!/usr/bin/env bash
# Publishes the solver to https://www.michael-han.com/gto-solver/ by copying it into
# the michaellhan/michaellhan.github.io repo (folder: gto-solver/) and pushing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"

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
  echo "Running unit tests ($PYTHON -m unittest)..."
  # a few tests run solves on a time budget, so a heavily loaded machine can fail one now and then; a
  # real bug fails again, so a failure is retried once before it blocks the deploy
  if ! (cd "$HERE" && "$PYTHON" -m unittest -q); then
    echo "Unit tests failed; retrying once (time-budgeted tests can fail on a loaded machine)..."
    if ! (cd "$HERE" && "$PYTHON" -m unittest -q); then
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

gh repo clone michaellhan/michaellhan.github.io "$WORK/site" -- -q --depth 1
rm -rf "$WORK/site/gto-solver"
mkdir -p "$WORK/site/gto-solver/gto"
cp "$HERE/gto/static/index.html" "$HERE/gto/static/worker.js" "$HERE/gto/static/hh.js" "$HERE/gto/static/practice.js" "$WORK/site/gto-solver/"
cp "$HERE"/gto/*.py "$WORK/site/gto-solver/gto/"
# Stamp the worker's cache-busting version with its content hash (and practice.js's, loaded with the same
# version), so a changed worker.js is never served from a browser's 4h-cached copy (which would load an
# outdated list of solver files).
WV="$(cat "$HERE/gto/static/worker.js" "$HERE/gto/static/practice.js" | shasum | cut -c1-10)"
sed -i.bak -E "s/^const WORKER_VERSION = \"[^\"]*\";/const WORKER_VERSION = \"$WV\";/" "$WORK/site/gto-solver/index.html"
rm "$WORK/site/gto-solver/index.html.bak"
grep -q "const WORKER_VERSION = \"$WV\";" "$WORK/site/gto-solver/index.html"
# the same for the hand-history importer, which the page loads on demand
HV="$(shasum "$HERE/gto/static/hh.js" | cut -c1-10)"
sed -i.bak -E "s/^const HH_VERSION = \"[^\"]*\";/const HH_VERSION = \"$HV\";/" "$WORK/site/gto-solver/index.html"
rm "$WORK/site/gto-solver/index.html.bak"
grep -q "const HH_VERSION = \"$HV\";" "$WORK/site/gto-solver/index.html"
cd "$WORK/site"
git add -A
if git diff --cached --quiet; then echo "Site already up to date."; exit 0; fi
git commit -q -m "gto-solver: deploy $(cd "$HERE" && git rev-parse --short HEAD)"
git push -q origin main
echo "Deployed. GitHub Pages will update https://www.michael-han.com/gto-solver/ in about a minute."
