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
from gto import flop_library as fl
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
        # flop by hand buckets: the rows of its turn (the turn's classes, each a bucket game); the library's flop: its
        # turns' rows, their leaves' products batched before the pass
        for board, pot, stack, iters, cfg in (("Ks7d2cTh", 20, 90, 30, native.config("fast")),
                                              ("Kd7c2h", 5.5, 97.5, 20, native.config("fast")),
                                              ("Kd7c2h", 5.5, 97.5, 6, fl.library_config())):
            with self.subTest(board, cfg=cfg.leaf_turns):
                got = []
                try:
                    for t in (1, 3, 8):
                        self.assertEqual(core.threads(t), t)
                        sol = _solve(board, OOP_T, IP_T, pot, stack, cfg, iters)
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
        # the library's flop (every turn line ending in an equity leaf: their products batched, trimmed to the ranges)
        # and a 3bet pot's (its river dealt, each card by buckets)
        ("library flop", "Kd7c2h", OOP_T, IP_T, 5.5, 97.5, fl.library_config(), 20),
        ("library flop, 3bet pot", "9h8h6c", OOP_T, IP_T, 20, 40, fl.library_config(20, 40), 10),
    ]
    # the first iteration's regrets: the largest difference at a node over its largest regret (and 1). A flop's
    # turn and river utilities come through the bucket tables summed in another order than numpy's BLAS: ~1e-4 off
    # at every node, the root's regrets of 249 among them, which is 5e-5 of a turn node whose actions are worth
    # about the same (regrets of 2)
    REGRET_TOL = {"flop by buckets": 1e-4}
    # the library's products on the core's own loops (no BLAS) summed in another order than numpy's sgemm: 2.2e-5
    REGRET_TOL_LOOPS = 5e-5
    # the second iteration's averages: in the 3bet pot's library flop a turn hand all but tied (regrets of -0.015,
    # 0.004 and 0.012 at a node worth 18) takes its first strategy from regrets 1.4e-5 apart, and its average 3e-4
    AVG_TOL = {"library flop, 3bet pot": 5e-4}

    def test_solve_and_evaluations(self):
        for name, board, r0, r1, pot, stack, cfg, iters, gemm in self._runs():
            with self.subTest(name, gemm=gemm):
                core.lib().L.core_gemm(gemm)
                # the first iteration's regrets the same to rounding, and every node's strategy after the second
                # but where the first left a hand tied (worth the same to every action: 0.4% of the by-class turn's
                # slots, 88 on the 8 class among them); later they part as numpy's own native and browser paths do
                # (a hand indifferent between actions goes one way or the other by rounding, the same nodes by the
                # same amounts), equally far from equilibrium
                a, b = (_solve(board, r0, r1, pot, stack, cfg, 1, use_core=u) for u in (True, False))
                if name.startswith("library"):
                    self.assertIs(core._prog.s(), a.solver)
                    self.assertGreater(core._prog.stacks, 0)
                # (a training pass's batched leaves give the combos outside a player's range 0: their regrets differ,
                # at the nodes played per combo)
                live = [np.ravel(r) > 0 for r in (b.r0, b.r1)]
                a, b = _decisions(a), _decisions(b)
                self.assertEqual(len(a), len(b))
                scale = [float(np.abs(y.regret).max()) for y in b]     # (regrets weigh in the opponent's combos)
                diff = [np.abs(x.regret - y.regret) for x, y in zip(a, b)]
                diff = [d[..., live[y.player]] if d.shape[-1] == live[y.player].size else d for d, y in zip(diff, b)]
                loops = gemm == 0 and name.startswith("library")
                tol = self.REGRET_TOL.get(name, self.REGRET_TOL_LOOPS if loops else 1e-5)
                self.assertLess(max(float(d.max()) / (m + 1) for d, m in zip(diff, scale)), tol)
                tied = [np.abs(y.regret).max(axis=1, keepdims=True) < 1e-5 * m for y, m in zip(b, scale)]
                a, b = (_solve(board, r0, r1, pot, stack, cfg, 2, use_core=u) for u in (True, False))
                self.assertIs(core._prog.s(), a.solver)
                sa, sb = ([np.where(t, 0, n.avg_strategy()) for t, n in zip(tied, _decisions(x))] for x in (a, b))
                self.assertLess(max(float(np.abs(x - y).max()) for x, y in zip(sa, sb)), self.AVG_TOL.get(name, 1e-4))
                a, b = (_solve(board, r0, r1, pot, stack, cfg, iters, use_core=u) for u in (True, False))
                self.assertEqual((a.iters, b.iters), (iters, iters))
                ea, eb = a.exploitability(), b.exploitability()
                # (rounding alone moves it: the river 60 iterations in at 0.47% of the pot against numpy's 0.51,
                # 0.197 and 0.195 at 100, 0.030 and 0.031 at 400)
                self.assertLess(abs(ea - eb), 0.1 * eb + 0.01)
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

    def _runs(self):
        """Each spot, the library's on both of the batched leaves' products (BLAS's and the core's own loops)."""
        blas = core.lib().L.core_gemm(-1)
        for spot in self.SPOTS:
            for gemm in ((1, 0) if blas and spot[0].startswith("library") else (blas,)):
                yield spot + (gemm,)
        core.lib().L.core_gemm(1)

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
class TestTables(unittest.TestCase):
    def test_same_tables_as_numpy(self):
        # a flop's tables built in the core (gto/core.py Tables) and in numpy: the flop's equity leaf and the turns' leaf
        # averages the same floats, every bucket the same, the bucket games' E to float32 rounding; Fast's model (every
        # third river) with a 3bet pot's river in the turn's buckets, and the browser's Accurate (every river)
        for cfg in (native.config("fast"), replace(native.config("fast"), river_buckets=64, bucket_river_step=1)):
            sol = solve(cards("Kd7c2h"), OOP_T, IP_T, pot=5.5, stack=97.5, cfg=cfg, time_budget=1e9, max_iters=0)
            ctx, dead, board = sol.solver.ctx, sol.solver.dead, tuple(cards("Kd7c2h"))
            got = {}
            for use in (True, False):
                ctx._stacks = {k: v for k, v in ctx._stacks.items() if k[0] == "S"}    # (the strengths kept)
                ctx._leaf = {}
                ctx.tables = core.Tables(core.lib()) if use else None
                try:
                    got[use] = [ctx._flop_leaf(board, dead)]
                    for t, _ in ctx.turn_classes(board, dead)[:4]:
                        b = board + (t,)
                        got[use] += ctx.bucket_table(b, dead, cfg.river_buckets)
                        for outer in (None, cfg.river_buckets):
                            got[use] += ctx.river_table(b, dead, cfg.river_strength, outer)
                finally:
                    ctx.tables = None
            self.assertEqual(len(got[True]), 1 + 4 * 9)
            for i, (a, b) in enumerate(zip(got[True], got[False])):
                with self.subTest(bucket_river_step=cfg.bucket_river_step, table=i):
                    self.assertEqual(a.shape, b.shape)
                    if i % 3 == 2:                               # (E)
                        self.assertLess(float(np.abs(a - b).max()), 5e-6)
                    else:
                        np.testing.assert_array_equal(a, b)


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

    def test_leaf_heavy_trees_batch_their_leaves(self):
        # natively the library's flop batches its equity leaves by matrix stack, trimmed to the ranges for training
        # (every turn line ends in one: 24 turn classes' matrices, read in place); without BLAS (the browser's solver)
        # one leaf at a time
        sol = solve(cards("Kd7c2h"), OOP_T, IP_T, pot=5.5, stack=97.5, cfg=fl.library_config(), max_iters=0)
        S = sol.solver
        self.assertTrue(S.batch_leaves)
        prog = core._program(core.lib(), S, sol.r0, sol.r1)
        self.assertGreater(prog.leaf_rows, core.LEAF_ROWS_BLAS)
        self.assertEqual(prog.stacks, 2)                  # (the flop's all-in leaves, the turns')
        self.assertEqual([len(x) for x in prog.live], [int((np.ravel(r) > 0).sum()) for r in (sol.r0, sol.r1)])
        r0 = np.where(np.ravel(sol.r0) > 0, sol.r0, 0.5)   # a combo outside the range: the training untrimmed
        prog.reaches(r0, sol.r1)
        self.assertIsNone(prog.live)
        S.batch_leaves, S._core_ok = False, True          # (as the browser's solver: no BLAS)
        core._prog = None
        prog = core._program(core.lib(), S, sol.r0, sol.r1)
        self.assertIsNotNone(prog)
        self.assertEqual(prog.stacks, 0)

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
