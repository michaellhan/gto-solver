"""The browser's solver code (scripts/build_bundle.py's solver.zip, which gto/static/worker.js unpacks into Pyodide):
it holds every module the solver imports, eagerly or inside a function, and treys; each module imports from the zip
alone; the worker's startup imports leave the modules only some requests need for later; and the worker and deploy
script agree on the bundle's name."""
import ast
import glob
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_bundle  # noqa: E402

WORKER = open(os.path.join(ROOT, "gto", "static", "worker.js")).read()
# what the worker imports before it reports "Solving…" (its first runPython), and modules no startup should pay for
STARTUP = re.search(r'"import (gto\.[\w., ]+)\\n"', WORKER).group(1)
LAZY = ["gto.multiway_solver", "gto.pf_game", "gto.pf_game_data", "gto.pf_eqr", "gto.pf_solve", "gto.pf_realize",
        "gto.pf_realize_model", "gto.breakdown", "gto.exploit", "gto.native", "gto.cli", "gto.server"]


def run_from(zip_path, code):
    """Runs code in a fresh interpreter with the zip first on sys.path (numpy from site-packages)."""
    pre = f"import sys; sys.path.insert(0, {zip_path!r}); "
    return subprocess.run([sys.executable, "-c", pre + code], cwd=tempfile.gettempdir(), capture_output=True, text=True)


class TestBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.zip = os.path.join(cls.tmp.name, build_bundle.NAME)
        cls.sha = build_bundle.build(cls.zip)
        cls.names = set(zipfile.ZipFile(cls.zip).namelist())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deterministic(self):
        self.assertEqual(build_bundle.build(os.path.join(self.tmp.name, "again.zip")), self.sha)

    def test_every_module_and_its_imports(self):
        """Every gto/*.py (gto/__init__.py too) and treys are in the zip, and every module a gto module imports,
        at the top or inside a function, is one of them."""
        mods = sorted(glob.glob(os.path.join(ROOT, "gto", "*.py")))
        for p in mods:
            self.assertIn("gto/" + os.path.basename(p), self.names)
        for f in ("__init__", "card", "evaluator", "lookup"):
            self.assertIn(f"treys/{f}.py", self.names)
        for p in mods:
            for node in ast.walk(ast.parse(open(p).read())):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    targets = [node.module] if node.module else [a.name for a in node.names]
                    for t in targets:
                        self.assertIn(f"gto/{t.split('.')[0]}.py", self.names, f"{os.path.basename(p)} imports .{t}")

    def test_each_module_imports_from_the_zip(self):
        """Each module on its own (any order of first use works), from the zip, treys included."""
        for p in sorted(glob.glob(os.path.join(ROOT, "gto", "*.py"))):
            name = os.path.basename(p)[:-3]
            mod = "gto" if name == "__init__" else f"gto.{name}"
            r = run_from(self.zip, f"import {mod}, treys; assert {mod}.__file__.startswith({self.zip!r}), {mod}.__file__; "
                                   f"assert treys.__file__.startswith({self.zip!r})")
            self.assertEqual(r.returncode, 0, f"{mod}: {r.stderr[-800:]}")

    def test_startup_leaves_the_rest_for_later(self):
        """The worker's startup imports (advisor, solver, flop_library) don't pull in the modules only 3-way pots,
        preflop solves, locks, the pool or the Breakdown view use."""
        r = run_from(self.zip, f"import {STARTUP}; print(','.join(sorted(m for m in sys.modules if m.startswith('gto'))))")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        loaded = r.stdout.strip().split(",")
        for m in LAZY:
            self.assertNotIn(m, loaded)

    def test_worker_and_deploy_agree(self):
        """The worker fetches the bundle by the name the build writes; deploy.sh builds it next to the page and
        hashes it into the worker's version (a new bundle is a new URL, past the Cache API copy)."""
        self.assertIn(f"`{build_bundle.NAME}?v=${{VERSION}}`", WORKER)
        self.assertFalse(re.search(r"loadPackage\(.*micropip|pyimport\(\"micropip\"\)", WORKER), "micropip is back in the worker")
        deploy = open(os.path.join(ROOT, "scripts", "deploy.sh")).read()
        self.assertIn(f'scripts/build_bundle.py" "$WORK/site/gto-solver/{build_bundle.NAME}"', deploy)
        wv = re.search(r'^WV="\$\((.*)\)"$', deploy, re.M).group(1)
        self.assertIn(f"$WORK/site/gto-solver/{build_bundle.NAME}", wv)
        self.assertIn("worker.js", wv)


class TestStartupTables(unittest.TestCase):
    """The tables built at import, made cheaper for the worker's startup, are what they were."""

    def test_class_combos_match_a_scan(self):
        from gto import ranges
        from gto.cards import ALL_COMBOS, RANKS, rank, suit
        for a in RANKS:
            for b in RANKS:
                for sfx in ("", "s", "o"):
                    cls = a + b + sfx
                    if a == b and sfx:
                        continue
                    r1, r2 = RANKS.index(a), RANKS.index(b)
                    scan = [cb for cb in ALL_COMBOS if {rank(cb[0]), rank(cb[1])} == {r1, r2}
                            and (r1 == r2 or not sfx or (sfx == "s") == (suit(cb[0]) == suit(cb[1])))]
                    self.assertEqual(ranges._class_combos(cls), scan, cls)

    def test_pair_compat(self):
        import numpy as np
        from gto import pf_classes as P
        self.assertTrue(np.array_equal(P.PAIR_COMPAT, P.class_sums(P.COMPAT.astype(float)) / np.outer(P.N_IN, P.N_IN)))


if __name__ == "__main__":
    unittest.main()
