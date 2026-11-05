# GTO Solver

- Commit and push directly to `main`. Do not create feature branches or pull requests.
- Run `python scripts/run_tests.py` before pushing major changes: solver, advisor or worker code, saved-data code, deploy scripts, a large page rewrite. It runs what `python -m unittest` finds, one module per process, because all of them in one process grow large enough for the machine's memory guard to stop the run.
- For small page or text changes, skip the full suite: run `node --test 'tests/js/*.test.js'` and `python -m unittest tests.test_routes tests.test_bundle tests.test_deploy tests.test_classic tests.test_progress`, then deploy with `SKIP_CHECKS=1 scripts/deploy.sh`.
- Deploy the site with `scripts/deploy.sh` (copies the page, worker and `gto/*.py` into
  `gto-solver/` of the michaellhan/michaellhan.github.io repo and pushes its `main`).
