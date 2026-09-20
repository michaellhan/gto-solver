#!/usr/bin/env bash
# Builds the desktop app with PyInstaller. macOS: dist/GTO Solver.app + dist/GTO-Solver-mac.dmg
# Windows (run in Git Bash / CI): dist/GTO-Solver-windows.exe. Linux: dist/GTO-Solver-linux
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
PY="${PYTHON:-.venv-build/bin/python}"
[ -x "$PY" ] || PY=python
SEP=":"; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) SEP=";";; esac
rm -rf build "dist/GTO Solver" "dist/GTO Solver.app" dist/GTO-Solver-*
"$PY" -m PyInstaller --noconfirm --clean --windowed --name "GTO Solver" \
  --add-data "gto/static${SEP}gto/static" \
  --collect-submodules gto --hidden-import webview \
  $( [ "$(uname -s)" != "Darwin" ] && echo --onefile ) \
  app_entry.py
case "$(uname -s)" in
  Darwin)
    codesign --force --deep -s - "dist/GTO Solver.app" 2>/dev/null || true
    "dist/GTO Solver.app/Contents/MacOS/GTO Solver" --check
    rm -f dist/GTO-Solver-mac.dmg
    hdiutil create -quiet -volname "GTO Solver" -srcfolder "dist/GTO Solver.app" -ov -format UDZO dist/GTO-Solver-mac.dmg
    echo "built dist/GTO-Solver-mac.dmg ($(du -h dist/GTO-Solver-mac.dmg | cut -f1))";;
  MINGW*|MSYS*|CYGWIN*)
    mv "dist/GTO Solver.exe" dist/GTO-Solver-windows.exe
    echo "built dist/GTO-Solver-windows.exe";;
  *)
    mv "dist/GTO Solver" dist/GTO-Solver-linux
    echo "built dist/GTO-Solver-linux";;
esac
