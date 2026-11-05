# GTO Solver

- Commit and push directly to `main`. Do not create feature branches or pull requests.
- Run `python scripts/run_tests.py` before pushing. It runs what `python -m unittest` finds, one module per process, because all of them in one process grow large enough for the machine's memory guard to stop the run.
- Deploy the site with `scripts/deploy.sh` (copies the page, worker and `gto/*.py` into
  `gto-solver/` of the michaellhan/michaellhan.github.io repo and pushes its `main`).
