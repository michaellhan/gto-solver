#!/usr/bin/env bash
# Publishes the last full build of the solver as a frozen copy at https://www.michael-han.com/gto-solver-classic/, the
# folder gto-solver-classic/ of the michaellhan/michaellhan.github.io repo, beside gto-solver/. It's built once, from
# CLASSIC_COMMIT, by that commit's own scripts/deploy.sh (BUILD_ONLY) in a temporary worktree, and is never updated:
# scripts/deploy.sh wipes only gto-solver/ (rm -rf of that one path) and commits what's left, so it stays as it is.
#   scripts/publish_classic.sh                    clones the site repo, adds the folder, commits and pushes main
#   FORCE=1 scripts/publish_classic.sh            replaces a copy already there (it's frozen: a second publish is a mistake)
#   BUILD_ONLY=<dir> scripts/publish_classic.sh   writes <dir>/site/gto-solver-classic and stops: no clone, no push
# PYTHON overrides the interpreter (default: this checkout's .venv, else the main checkout's, else python3).
#
# The copy reads the live library, /gto-solver-library/, as a sibling (LIB_BASE "../gto-solver-library/", the same
# from /gto-solver-classic/ as from /gto-solver/, and from a route's copy through its <base href="../">), so the 334 MB
# library deploy.sh's BUILD_ONLY assembles is deleted, not published. It shares the origin, so IndexedDB "gto-store" and
# the localStorage keys are shared with gto-solver/ (one person's hands, seen by both). Its service worker has its own
# scope (sw.js registers relatively; a scope of /gto-solver/ doesn't cover /gto-solver-classic/).
#
# What differs from the build, each replaced by replace_once (which fails unless the text is there exactly once):
# - worker.js: caches.open("gto-solver") -> caches.open("gto-solver-classic"). worker.js's toCache deletes every key of
#   its cache whose path ends in /solver.zip, so two sites on one cache name evict each other's bundle on each visit.
#   Don't change the source's: Lizard's build requires caches.open("gto-solver") exactly once in gto/static/worker.js.
# - every page: sessionStorage "gto-isolate" -> "gto-isolate-classic". It's the once-a-tab mark for the first load
#   under the service worker, and sessionStorage is the origin's: a tab that had been on /gto-solver/ never registered
#   this folder's worker, and the page ran unisolated, on one thread, with its Stop restarting the worker.
set -euo pipefail

CLASSIC_COMMIT=1a14605e
SITE_REPO=michaellhan/michaellhan.github.io
FOLDER=gto-solver-classic
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISOLATE_OLD='if (sessionStorage.getItem("gto-isolate")) return false; sessionStorage.setItem("gto-isolate", "1");'
ISOLATE_NEW='if (sessionStorage.getItem("gto-isolate-classic")) return false; sessionStorage.setItem("gto-isolate-classic", "1");'

# FILE OLD NEW: OLD, literally, becomes NEW; fails, leaving the file as it was, unless OLD is there exactly once
replace_once() {
  local n
  n="$(FROM="$2" perl -0777 -ne '$f = $ENV{FROM}; $c = () = /\Q$f\E/g; print $c' "$1")"
  if [ "$n" != "1" ]; then
    echo "publish_classic: expected exactly one of [$2] in $1, found ${n:-0}." >&2
    return 1
  fi
  FROM="$2" TO="$3" perl -0777 -i -pe '$f = $ENV{FROM}; $t = $ENV{TO}; s/\Q$f\E/$t/' "$1"
}

# DIR: a build of gto-solver/ made into the frozen copy, and what it depends on checked
freeze() {
  local dir="$1" page
  replace_once "$dir/worker.js" 'caches.open("gto-solver")' 'caches.open("gto-solver-classic")'
  [ -f "$dir/index.html" ] || { echo "publish_classic: no $dir/index.html." >&2; return 1; }
  while IFS= read -r page; do
    replace_once "$page" "$ISOLATE_OLD" "$ISOLATE_NEW"
    # (the library, a sibling: deploy.sh points the page there only when it finds the library's index)
    grep -q '^const LIB_BASE = "../gto-solver-library/";' "$page" \
      || { echo "publish_classic: $page doesn't read the library from ../gto-solver-library/." >&2; return 1; }
  done < <(find "$dir" -name index.html)
  # a path keyed to the folder, in code (comments may name it)
  if grep -rnE --include='*.html' --include='*.js' --include='*.json' '["'"'"'`=]/gto-solver/|url\(/gto-solver/|michael-han\.com/gto-solver/' "$dir"; then
    echo "publish_classic: the lines above hard-code /gto-solver/." >&2
    return 1
  fi
}

python_for_build() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  local main
  main="$(dirname "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)")"
  for p in "$HERE/.venv/bin/python" "$main/.venv/bin/python"; do
    if [ -x "$p" ]; then echo "$p"; return; fi
  done
  echo python3
}

main() {
  local out="${BUILD_ONLY:-}" work src size
  work="$(mktemp -d)"
  src="$work/src"
  # shellcheck disable=SC2064
  trap "git -C '$HERE' worktree remove --force '$src' 2>/dev/null || true; rm -rf '$work'" EXIT

  # A frozen copy already there stops a real publish before anything is built.
  if [ -z "$out" ]; then
    gh repo clone "$SITE_REPO" "$work/site" -- -q --depth 1
    if [ -e "$work/site/$FOLDER" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "$FOLDER/ is already in $SITE_REPO, and it's frozen: a second publish is a mistake. FORCE=1 replaces it." >&2
      exit 1
    fi
  fi

  git -C "$HERE" rev-parse -q --verify "$CLASSIC_COMMIT^{commit}" >/dev/null || git -C "$HERE" fetch -q origin main
  git -C "$HERE" worktree add -q --detach "$src" "$CLASSIC_COMMIT"
  BUILD_ONLY="$work/build" PYTHON="$(python_for_build)" bash "$src/scripts/deploy.sh" >/dev/null
  rm -rf "$work/build/site/gto-solver-library"
  freeze "$work/build/site/gto-solver"
  size="$(du -sh "$work/build/site/gto-solver" | cut -f1), $(find "$work/build/site/gto-solver" -type f | wc -l | tr -d ' ') files"

  if [ -n "$out" ]; then
    mkdir -p "$out/site"
    rm -rf "${out:?}/site/$FOLDER"
    cp -R "$work/build/site/gto-solver" "$out/site/$FOLDER"
    echo "Built $out/site/$FOLDER ($size; BUILD_ONLY: not published)."
    exit 0
  fi

  rm -rf "${work:?}/site/$FOLDER"
  cp -R "$work/build/site/gto-solver" "$work/site/$FOLDER"
  git -C "$work/site" add -- "$FOLDER"
  if git -C "$work/site" diff --cached --quiet; then echo "$FOLDER/ is already this build."; exit 0; fi
  if git -C "$work/site" diff --cached --name-only | grep -qv "^$FOLDER/"; then
    echo "publish_classic: something outside $FOLDER/ was staged; nothing pushed." >&2
    exit 1
  fi
  git -C "$work/site" commit -q -m "$FOLDER: frozen copy of $CLASSIC_COMMIT"
  git -C "$work/site" push -q origin main
  echo "Published $FOLDER/ ($size). GitHub Pages will update https://www.michael-han.com/$FOLDER/ in about a minute."
}

# (sourced by tests/test_classic.py, for replace_once and freeze)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
