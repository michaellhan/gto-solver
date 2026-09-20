#!/bin/bash
# macOS launcher: double-click. First run creates a private Python environment (needs Python 3.9+).
cd "$(dirname "$0")"
exec ./run.sh "$@"
