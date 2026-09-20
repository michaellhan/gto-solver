@echo off
rem Windows launcher: double-click. First run creates a private Python environment (needs Python 3.9+ from python.org).
cd /d "%~dp0"
set PORT=18877
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv\Scripts\python.exe (
  echo First run: setting up ^(about a minute^)...
  %PY% -m venv .venv || (echo Python 3 is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
echo GTO Solver running at http://127.0.0.1:%PORT%/  (close this window to stop)
.venv\Scripts\python -m gto.server --port %PORT% --open
pause
