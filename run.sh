#!/bin/bash
# Starts the GTO Solver on this computer and opens it in your browser. Usage: ./run.sh [port]
cd "$(dirname "$0")"
PORT="${1:-18877}"
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "Python 3 is required. Install it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close." _; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (about a minute)..."
  "$PY" -m venv .venv || { echo "Could not create a Python environment."; read -r -p "Press Enter to close." _; exit 1; }
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "Install failed."; read -r -p "Press Enter to close." _; exit 1; }
fi
echo "GTO Solver running at http://127.0.0.1:$PORT/  (close this window to stop)"
exec .venv/bin/python -m gto.server --port "$PORT" --open
