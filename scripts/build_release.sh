#!/usr/bin/env bash
# Builds dist/gto-solver.zip: the native version with double-click launchers.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/dist"; STAGE="$(mktemp -d)/gto-solver"
mkdir -p "$OUT" "$STAGE/gto/static" "$STAGE/examples"
cp "$HERE"/gto/*.py "$STAGE/gto/"
cp "$HERE"/gto/static/index.html "$HERE"/gto/static/worker.js "$STAGE/gto/static/"
cp "$HERE"/examples/*.json "$STAGE/examples/"
cp "$HERE"/requirements.txt "$HERE"/README.md "$HERE"/run.command "$HERE"/run.sh "$HERE"/run.bat "$STAGE/"
cat > "$STAGE/START HERE.txt" <<'TXT'
GTO Solver (native version)

  Mac:      double-click run.command   (if macOS blocks it: right-click > Open, once)
  Windows:  double-click run.bat
  Linux:    ./run.sh

Needs Python 3.9 or newer (https://www.python.org/downloads/). The first run creates a
private environment and installs numpy and treys; later runs start in a second.
The page opens at http://127.0.0.1:18877/ . Close the window to stop.
TXT
rm -f "$OUT/gto-solver.zip"
(cd "$(dirname "$STAGE")" && zip -q -r -X "$OUT/gto-solver.zip" gto-solver)
echo "built $OUT/gto-solver.zip ($(du -h "$OUT/gto-solver.zip" | cut -f1))"
