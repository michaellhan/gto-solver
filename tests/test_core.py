"""The compiled CFR core (gto/core.py, core/gtocore.c): the same solve as numpy's to rounding, the same EVs and
exploitability after it, numpy for the trees it doesn't run, and a browser module built from the current source."""
import gc
import os
import unittest
import weakref
from dataclasses import replace
from unittest import mock

import numpy as np

from gto import core, native
from gto.cards import cards
from gto.ranges import parse_range
from gto.solver import BucketEntry, Config, Decision, Chance, solve

OOP = parse_range("22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,64s+,54s,A8o+,KTo+,QTo+,JTo")
IP = parse_range("22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,85s+,74s+,63s+,53s+,43s,A2o+,K8o+,Q9o+,J9o+,T8o+,98o")
OOP_T = parse_range("66+,A9s+,KTs+,QTs+,JTs,T9s,98s,AJo+,KQo")
IP_T = parse_range("22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,ATo+,KJo+,QJo")
ON = core.lib() is not None


def _solve(board, r0, r1, pot, stack, cfg, iters, use_core=True):
    with mock.patch.object(core, "lib", return_value=core.lib() if use_core else None):
        return solve(cards(board), r0, r1, pot, stack, cfg=cfg, time_budget=1e9, max_iters=iters)


def _decisions(sol):
    out = []

    def walk(n):
        if isinstance(n, Decision):
            out.append(n)
            for c in n.children:
                walk(c)
        elif isinstance(n, (Chance, BucketEntry)):
            walk(n.child)
    walk(sol.root)
    return out


class TestWasm(unittest.TestCase):
    def test_module_built_from_the_current_source(self):
        # (python scripts/build_core.py after a change to core/gtocore.c)
        for name in ("core.wasm", "core-mt.wasm"):
            with open(os.path.join(core.ROOT, "gto", name), "rb") as f:
                self.assertIn(core.source_hash().encode(), f.read(), name)


@unittest.skipUnless(ON, "no C compiler for the native core")
class TestThreads(unittest.TestCase):
    def test_same_numbers_on_any_number_of_threads(self):
        # a turn by river class: the rows of its river nodes split between threads, each row's arithmetic the same; a
        # flop by hand buckets: the rows of its turn (the turn's classes, each a bucket game)
        for board, pot, stack, iters in (("Ks7d2cTh", 20, 90, 30), ("Kd7c2h", 5.5, 97.5, 20)):
            with self.subTest(board):
                got = []
                try:
                    for t in (1, 3, 8):
                        self.assertEqual(core.threads(t), t)
                        sol = _solve(board, OOP_T, IP_T, pot, stack, native.config("fast"), iters)
                        self.assertTrue(sol.solver.use_core and sol.solver._core_ok)
                        arr = np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()] for d in _decisions(sol)])
                        got.append((arr.view(np.int32), sol.exploitability()))
                finally:
                    core.threads(1)
                for arr, ex in got[1:]:
                    np.testing.assert_array_equal(arr, got[0][0])
                    self.assertEqual(ex, got[0][1])


@unittest.skipUnless(ON, "no C compiler for the native core")
class TestSameAsNumpy(unittest.TestCase):
    SPOTS = [
        ("river", "Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 60),
        ("turn by river class", "Ks7d2cTh", OOP_T, IP_T, 20, 90, native.config("fast"), 30),
        ("turn raked, all-ins", "Qs8d3c4h", OOP_T, IP_T, 30, 40,
         Config(river_classes=False, n_rivers=None, rake_rate=0.05, rake_cap=3.0), 25),
        # a flop by hand buckets (Fast's flop model): the turn by bucket, the river by strength; a 3bet pot's river
        # in the turn's buckets, raked
        ("flop by buckets", "Kd7c2h", OOP_T, IP_T, 5.5, 97.5, native.config("fast"), 20),
        ("flop by buckets, 3bet pot raked", "9h8h6c", OOP_T, IP_T, 20, 40,
         replace(native.config("fast"), rake_rate=0.05, rake_cap=3.0), 20),
    ]
    # the first iteration's regrets: the largest difference at a node over its largest regret (and 1). A flop's
    # turn and river utilities come through the bucket tables summed in another order than numpy's BLAS: ~1e-4 off
    # at every node, the root's regrets of 249 among them, which is 5e-5 of a turn node whose actions are worth
    # about the same (regrets of 2)
    REGRET_TOL = {"flop by buckets": 1e-4}

    def test_solve_and_evaluations(self):
        for name, board, r0, r1, pot, stack, cfg, iters in self.SPOTS:
            with self.subTest(name):
                # the first iteration's regrets the same to rounding, and every node's strategy after the second
                # but where the first left a hand tied (worth the same to every action: 0.4% of the by-class turn's
                # slots, 88 on the 8 class among them); later they part as numpy's own native and browser paths do
                # (a hand indifferent between actions goes one way or the other by rounding, the same nodes by the
                # same amounts), equally far from equilibrium
                a, b = (_decisions(_solve(board, r0, r1, pot, stack, cfg, 1, use_core=u)) for u in (True, False))
                self.assertEqual(len(a), len(b))
                scale = [float(np.abs(y.regret).max()) for y in b]     # (regrets weigh in the opponent's combos)
                self.assertLess(max(float(np.abs(x.regret - y.regret).max()) / (m + 1)
                                    for x, y, m in zip(a, b, scale)), self.REGRET_TOL.get(name, 1e-5))
                tied = [np.abs(y.regret).max(axis=1, keepdims=True) < 1e-5 * m for y, m in zip(b, scale)]
                a, b = (_solve(board, r0, r1, pot, stack, cfg, 2, use_core=u) for u in (True, False))
                self.assertIs(core._prog.s(), a.solver)
                sa, sb = ([np.where(t, 0, n.avg_strategy()) for t, n in zip(tied, _decisions(x))] for x in (a, b))
                self.assertLess(max(float(np.abs(x - y).max()) for x, y in zip(sa, sb)), 1e-4)
                a, b = (_solve(board, r0, r1, pot, stack, cfg, iters, use_core=u) for u in (True, False))
                self.assertEqual((a.iters, b.iters), (iters, iters))
                ea, eb = a.exploitability(), b.exploitability()
                self.assertLess(abs(ea - eb), 0.03 * eb + 0.01)
                # the passes after a solve, on one solver's sums: core vs numpy
                S, ev = a.solver, {}
                for use in (True, False):
                    S.use_core = use
                    ev[use] = (S.node_action_evs(S.root, a.r0, a.r1), S.node_action_br_evs(S.root, a.r0, a.r1),
                               S.exploitability(a.r0, a.r1))
                S.use_core = True
                tol = 2e-4 * pot
                self.assertLess(float(np.abs(ev[True][0] - ev[False][0]).max()), tol)
                self.assertLess(float(np.abs(ev[True][1] - ev[False][1]).max()), tol)
                self.assertAlmostEqual(ev[True][2], ev[False][2], delta=1e-3)
                self.assertGreater(ev[True][2], 0)

    def test_evaluation_below_the_root(self):
        sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 40)
        S = sol.solver
        node = sol.root.children[0]                 # after the first action
        self.assertIsInstance(node, Decision)
        r = [sol.r0, sol.r1]
        r[sol.root.player] = r[sol.root.player] * S.root.avg_strategy()[0, 0]
        a, b = (x[None, :].astype(np.float32) for x in r)
        for md, want in ((core.AVG, S._cfr(node, node.player, a, b, update=False, use_avg=True)),
                         (core.BR, S._best_response(node, node.player, a, b))):
            got = core.evaluate(S, [(node, node.player, r[0], r[1], md)])
            self.assertLess(float(np.abs(got[0] - want).max()), 1e-3)


@unittest.skipUnless(ON, "no C compiler for the native core")
class TestProgram(unittest.TestCase):
    def test_another_solve_replaces_it_and_it_comes_back(self):
        a = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 20)
        ea = a.exploitability()
        b = _solve("Qs8d3c4h2h", OOP, IP, 20, 90, native.config("fast"), 20)
        b.exploitability()
        self.assertIsNot(core._prog.s(), a.solver)
        self.assertEqual(a.exploitability(), ea)                  # rebuilt for a, the same answer
        self.assertIs(core._prog.s(), a.solver)
        a.extend(1e9, max_iters=40)                               # and a keeps solving from where it was
        self.assertEqual(a.iters, 40)
        self.assertLess(a.exploitability(), ea)

    def test_doesnt_keep_a_dropped_solve(self):
        sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        sol.exploitability()
        ref = weakref.ref(sol.solver)
        del sol
        gc.collect()
        self.assertIsNone(ref())

    def test_numpy_runs_what_it_doesnt(self):
        # rivers sampled every iteration: numpy, and the tree remembered as one the core doesn't run
        cfg = Config(river_classes=False, river_sample=4)
        sol = solve(cards("Ks7d2cTh"), OOP_T, IP_T, pot=20, stack=90, cfg=cfg, time_budget=1e9, max_iters=3)
        self.assertEqual(sol.iters, 3)
        self.assertFalse(sol.solver._core_ok)
        self.assertGreater(sol.exploitability(), 0)

    def test_leaf_heavy_trees_stay_with_blas(self):
        # natively numpy's batched leaf products beat the core on the library's flop; without BLAS the core runs it
        from gto import flop_library as fl
        sol = solve(cards("Kd7c2h"), OOP_T, IP_T, pot=5.5, stack=97.5, cfg=fl.library_config(), max_iters=0)
        S = sol.solver
        self.assertTrue(S.batch_leaves)
        self.assertIsNone(core._program(core.lib(), S, sol.r0, sol.r1))
        self.assertFalse(S._core_ok)
        S.batch_leaves, S._core_ok = False, True       # (as the browser's solver: no BLAS)
        prog = core._program(core.lib(), S, sol.r0, sol.r1)
        self.assertIsNotNone(prog)
        self.assertGreater(prog.leaf_rows, core.LEAF_ROWS_BLAS)

    def test_numpy_runs_what_the_arena_cant_hold(self):
        # over ARENA_MAX, or the arena failing to grow (the browser's memory.grow refused: the core traps)
        want = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5, use_core=False).exploitability()
        trap = mock.patch.object(type(core.lib()), "alloc", side_effect=RuntimeError("unreachable"))
        for patch in (mock.patch.object(core, "ARENA_MAX", 1 << 16), trap):
            with self.subTest(str(patch.attribute)), patch:
                core._prog = None
                sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
                self.assertIsNone(core._prog)
                self.assertFalse(sol.solver._core_ok)
                self.assertEqual(sol.iters, 5)
                self.assertAlmostEqual(sol.exploitability(), want, places=4)

    def test_busy_core_numpy_runs(self):
        with core._lock:
            sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
            self.assertIsNone(core.evaluate(sol.solver, [(sol.root, 0, sol.r0, sol.r1, core.AVG)]))
        self.assertEqual(sol.iters, 5)
        self.assertGreater(sol.exploitability(), 0)


class TestOff(unittest.TestCase):
    def test_env_turns_it_off(self):
        with mock.patch.dict(os.environ, {"GTO_CORE": "0"}):
            self.assertIsNone(core._load())


if __name__ == "__main__":
    unittest.main()
