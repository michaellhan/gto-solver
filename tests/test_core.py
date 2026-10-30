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
from gto.cards import ALL_COMBOS, cards
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


def _chances(sol):
    out = []

    def walk(n):
        if isinstance(n, Decision):
            for c in n.children:
                walk(c)
        elif isinstance(n, (Chance, BucketEntry)):
            if isinstance(n, Chance):
                out.append(n)
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
        # turns' rows, their leaves' products batched before the pass; a 3bet pot's: its dealt rivers' deferred games,
        # their rows split between threads
        for board, pot, stack, iters, cfg in (("Ks7d2cTh", 20, 90, 30, native.config("fast")),
                                              ("Kd7c2h", 5.5, 97.5, 20, native.config("fast")),
                                              ("Kd7c2h", 5.5, 97.5, 6, fl.library_config()),
                                              ("9h8h6c", 20, 40, 4, fl.library_config(20, 40))):
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

    def test_discount_in_the_passes(self):
        # the DCFR discount applied in the iteration's passes (each regret and strategy sum at its last touch) or in
        # a pass of its own after them: the same floats, bit for bit; DCFR's two factors and DCFR+'s floor, a turn
        # by river class on 3 threads, a raked turn, a flop by hand buckets, and the library's 3bet pot (its dealt
        # rivers' deferred games walk their decisions twice a pass: discounted after the iteration either way)
        fast = native.config("fast")
        spots = (("Ks7d2cTh", 20, 90, 30, fast, 3), ("Ks7d2cTh", 20, 90, 30, replace(fast, cfr="dcfr+"), 1),
                 ("Qs8d3c4h", 30, 40, 20, Config(river_classes=False, n_rivers=None, rake_rate=0.05, rake_cap=3.0), 1),
                 ("Kd7c2h", 5.5, 97.5, 10, fast, 1), ("9h8h6c", 20, 40, 3, fl.library_config(20, 40), 1))
        for board, pot, stack, iters, cfg, t in spots:
            with self.subTest(board, cfr=cfg.cfr):
                got = []
                try:
                    core.threads(t)
                    for on in (1, 0):
                        self.assertEqual(core.lib().L.core_fuse(on), on)
                        sol = _solve(board, OOP_T, IP_T, pot, stack, cfg, iters)
                        self.assertTrue(sol.solver.use_core and sol.solver._core_ok)
                        got.append(np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()]
                                                   for d in _decisions(sol)]).view(np.int32))
                finally:
                    core.lib().L.core_fuse(1)
                    core.threads(1)
                np.testing.assert_array_equal(got[0], got[1])

    def test_rows_nobody_reaches(self):
        # a terminal's row the opponent's reach is all +0 on skips the sums (a fold's card-removal sums, a showdown's
        # running sums, a leaf's product) and writes the zeros they come to: the same floats bit for bit as summing
        # them, in training, the average strategies' EVs and the best responses. Rows that go unreached as a solve
        # settles (a turn on 3 threads, all-in leaves on a short stack, a raked turn dealing every river, a flop), and
        # an opponent of four combos holding the ace of clubs, its river unreached from the first iteration
        fast = native.config("fast")
        dealt = Config(river_classes=False, n_rivers=None, rake_rate=0.05, rake_cap=3.0)
        ace = cards("Ac")[0]
        few = np.zeros(1326)
        few[[i for i, (a, b) in enumerate(ALL_COMBOS) if ace in (a, b) and cards("Kc")[0] // 4 in (a // 4, b // 4)]] = 1
        self.assertEqual(int(few.sum()), 4)
        spots = (("Ks7d2cTh", 20, 90, 100, fast, IP_T, 3), ("Ks7d2cTh", 20, 25, 100, fast, IP_T, 1),
                 ("Qs8d3c4h", 30, 40, 60, dealt, IP_T, 1), ("Qs8d3c4h", 30, 40, 20, dealt, few, 1),
                 ("Kd7c2h", 5.5, 97.5, 10, fast, IP_T, 1))
        for board, pot, stack, iters, cfg, r1, t in spots:
            with self.subTest(board, stack=stack, combos=int((r1 > 0).sum())):
                got = []
                try:
                    core.threads(t)
                    for on in (1, 0):
                        self.assertEqual(core.lib().L.core_skip(on), on)
                        sol = _solve(board, OOP_T, r1, pot, stack, cfg, iters)
                        S = sol.solver
                        self.assertTrue(S.use_core and S._core_ok)
                        ev = [S.node_action_evs(S.root, sol.r0, sol.r1), S.node_action_br_evs(S.root, sol.r0, sol.r1)]
                        got.append((np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()]
                                                    for d in _decisions(sol)]).view(np.int32),
                                    [np.asarray(x) for x in ev], S.exploitability(sol.r0, sol.r1)))
                finally:
                    core.lib().L.core_skip(1)
                    core.threads(1)
                np.testing.assert_array_equal(got[0][0], got[1][0])
                for x, y in zip(got[0][1], got[1][1]):
                    np.testing.assert_array_equal(x, y)
                self.assertEqual(got[0][2], got[1][2])

    def test_lanes_hold_the_sums(self):
        # A helper runs its chunk of a chance node's rows in a lane of its own, sized by a serial pass over the chunks
        # in the first iteration. That pass sums every terminal row, reached or not: its second player's pass meets
        # the first's regret-matched zeros, and skipping the showdowns nobody reached there sized the lanes short of
        # a later chunk's (a trap in a helper: natively the process ends, in the browser the solve never answers).
        # The lanes the same with the skip on as off on 2, 4 and 8 threads, over turns full and trimmed (TRIMS'
        # third level trapped 28 of 112 such turns on 8 threads), a shallow 3bet pot's and a flop by hand buckets
        # (a river splits no rows), and each solve run on; and a limped turn advised at a 1 s budget (its tree
        # trimmed), which trapped every time. In a child process: a trap ends it.
        import json
        import subprocess
        import sys
        p = subprocess.run([sys.executable, "-c", "import tests.test_core as t; t._lanes_child()"], cwd=core.ROOT,
                           capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, PYTHONPATH=core.ROOT, GTO_THREADS="1"))
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        rows = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        self.assertEqual(len(rows), 3 * len(_lane_spots()) + 1)
        for r in rows[:-1]:
            self.assertGreater(r["lanes"][0][0], 0, r)
            self.assertEqual(r["lanes"][0], r["lanes"][1], r)
        self.assertTrue(rows[-1]["solver"]["iterations"] > 0, rows[-1])


def _lane_spots():
    btn = fl.spot_setup(next(s for s in fl.SPOTS if s[0] == "srp_btn_bb")[2])
    bb3 = fl.spot_setup(next(s for s in fl.SPOTS if s[0] == "3bp_bb_btn")[2])
    fast = native.config("fast")
    threaded = native.config("fast")
    threaded.bet_sizes["river"] = list(native.SPEEDS["accurate"][0]["bet_sizes"]["river"])
    shallow = native.config("fast")
    shallow.bet_sizes.update({k: list(v) for k, v in native.FAST_SPR_SIZES["turn"][1].items()})
    return (("Ks7d2cTh", OOP, IP, 9.1, 95.5, fast, 3), ("JcTd8d4h", btn["full0"], btn["full1"], 5.5, 97.5, threaded, 3),
            ("Ks7d2cTh", btn["full0"], btn["full1"], 5.5, 97.5, threaded, 0),
            ("9h8h6c5s", bb3["full0"], bb3["full1"], 20, 40, shallow, 0), ("Kd7c2h", OOP_T, IP_T, 5.5, 97.5, fast, 0))


def _lanes_child():
    """test_lanes_hold_the_sums' solves (a child process): a line of JSON each, the lanes after two iterations with
    the skip off and after twenty with it on; then the limped turn's request."""
    import json
    from gto.advisor import Advisor
    L = core.lib().L
    spots = _lane_spots()
    for t in (2, 4, 8):
        core.threads(t)
        for i, (board, r0, r1, pot, stack, cfg, trim) in enumerate(spots):
            cfg = Advisor._trim_cfg(cfg, ("flop", "turn", "river")[len(board) // 2 - 3], trim) if trim else cfg
            lanes = []
            for on, iters in ((0, 2), (1, 20)):
                L.core_skip(on)
                sol = solve(cards(board), r0, r1, pot, stack, cfg=cfg, time_budget=1e9, max_iters=iters)
                assert sol.solver.use_core and sol.solver._core_ok
                lanes.append([int(L.core_lanes(k)) for k in (0, 1)])
            print(json.dumps({"spot": i, "threads": t, "lanes": lanes}), flush=True)
    core.threads(8)
    F, X = (lambda p: {"pos": p, "action": "fold"}), (lambda p: {"pos": p, "action": "check"})
    C, B = (lambda p: {"pos": p, "action": "call"}), (lambda p, a: {"pos": p, "action": "bet", "amount": a})
    adv = native.advisor("fast", budget=1.0, prior_budget=1.0)
    res = adv.advise({"players": 6, "stack": 100, "hero": "SB", "hole": "QhJh", "board": "Ks7d2cTh",
                      "preflop": [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")],
                      "flop": [X("SB"), B("BB", 1.5), C("SB")], "turn": [], "river": []})
    print(json.dumps({"solver": res.get("solver"), "error": res.get("error")}), flush=True)


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

    def test_showdowns_on_any_number_of_combos(self):
        # a showdown's tables (gto/core.py _ranking_tables) lay the running sums out in four quarters, a remainder
        # after them: rivers of 1 to 323 combos (each remainder), the first iteration's regrets the same as numpy's
        # to rounding, and the passes after 20 iterations on one solver's sums (a raked river's showdowns add the
        # card-removal sums). (The solves themselves part after a few iterations in games this small, where a hand
        # is often tied between actions, as numpy's own native and browser paths do.)
        nz = np.nonzero(OOP * IP > 0)[0]
        for m in (1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 41, 103, 210):
            for rake in (0.0, 0.05):
                with self.subTest(combos=m, rake=rake):
                    r0, r1 = np.zeros(1326), np.zeros(1326)
                    r0[nz[:m]], r1[nz[-m:]] = 1.0, 0.5 + (np.arange(m) % 3) / 4
                    cfg = replace(native.config("fast"), rake_rate=rake, rake_cap=3.0 if rake else None)
                    a, b = (_solve("2c3d8hTs5s", r0, r1, 10, 50, cfg, 1, use_core=u) for u in (True, False))
                    self.assertIs(core._prog.s(), a.solver)
                    for x, y in zip(_decisions(a), _decisions(b)):
                        self.assertLess(float(np.abs(x.regret - y.regret).max()) / (np.abs(y.regret).max() + 1), 1e-5)
                    a = _solve("2c3d8hTs5s", r0, r1, 10, 50, cfg, 20)
                    S, ev = a.solver, {}
                    for use in (True, False):
                        S.use_core = use
                        ev[use] = (S.node_action_evs(S.root, a.r0, a.r1), S.node_action_br_evs(S.root, a.r0, a.r1),
                                   S.exploitability(a.r0, a.r1))
                    S.use_core = True
                    self.assertLess(float(np.abs(ev[True][0] - ev[False][0]).max()), 2e-5 * 10)
                    self.assertLess(float(np.abs(ev[True][1] - ev[False][1]).max()), 2e-5 * 10)
                    self.assertAlmostEqual(ev[True][2], ev[False][2], delta=1e-4)

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

    def test_sums_under_the_smallest_normal_count_as_none(self):
        # regret matching takes a sum of positive regrets (or of strategy sums) under float32's smallest normal for
        # none: its reciprocal overflowed, and a zero regret times inf made the strategy NaN
        sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        S, d = sol.solver, sol.root
        A = len(d.names)
        c = int(np.flatnonzero(np.ravel([sol.r0, sol.r1][d.player]) > 0)[0])
        tiny = np.float32(1e-40)
        self.assertTrue(0 < tiny < np.finfo(np.float32).tiny)
        d.ssum[0, :, c] = 0
        d.ssum[0, 0, c] = tiny
        np.testing.assert_array_equal(d.avg_strategy()[0, :, c], np.float32(1 / A))
        ev = {}
        for use in (True, False):
            S.use_core = use
            ev[use] = S.exploitability(sol.r0, sol.r1)
        S.use_core = True
        self.assertTrue(np.isfinite(ev[True]))
        self.assertAlmostEqual(ev[True], ev[False], delta=1e-3)
        d.regret[0, :, c] = 0
        d.regret[0, 0, c] = tiny
        np.testing.assert_array_equal(d.strategy()[0, :, c], np.float32(1 / A))
        sol.extend(1e9, max_iters=6)
        self.assertIs(core._prog.s(), S)
        self.assertTrue(np.isfinite(S._regret).all() and np.isfinite(S._ssum).all())

    def test_library_flop_where_its_regrets_shrank_to_denormals(self):
        # srp_utg_bb AcQdTh's library entry: rounding (numpy's own two leaf paths part as much) left the core's sum of
        # positive regrets at an all-in node, a hand whose opponent's reach there is all but 0, at 5e-41 by iteration
        # 87, whose reciprocal is inf: the strategies NaN from there, 73% of the pot from equilibrium at 300 where
        # numpy's solve is at 0.10
        s = fl.spot_setup(next(x for x in fl.SPOTS if x[0] == "srp_utg_bb")[2])
        cfg = fl.library_config()          # (the entry's tree then: the river blind)
        a, b = (_solve("AcQdTh", s["full0"], s["full1"], s["pot"], s["stack"], cfg, 100, use_core=u)
                for u in (True, False))
        b.solver.use_core = False
        for iters in (100, 200):
            for x in (a, b):
                x.extend(1e9, max_iters=iters)
            self.assertEqual((a.iters, b.iters), (iters, iters))
            self.assertTrue(np.isfinite(a.solver._regret).all())
            # (0.805 and 0.756% of the pot at 100, 0.204 and 0.202 at 200)
            self.assertLess(abs(a.exploitability() - b.exploitability()), 0.1)


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

    def test_turn_roots_leaf_over_every_river(self):
        # a turn solve's equity leaf (an all-in on the turn) over every river by _avg_showdown, in the core's tables
        # (built while it lays the program out) and in numpy: the floats the rivers' showdown matrices summed gave
        board = tuple(cards("Ks7d2cTh"))
        sol = solve(board, OOP_T, IP_T, pot=20, stack=90, cfg=native.config("fast"), time_budget=1e9, max_iters=0)
        ctx, dead = sol.solver.ctx, sol.solver.dead
        avail = [c for c in range(52) if c not in dead and c not in board]
        ref = np.zeros((ctx.n, ctx.n), dtype=np.float32)
        for c in avail:
            ref += ctx.showdown_matrix(board + (c,))
        h = ctx.member[avail].sum(axis=0)
        ref /= np.maximum(len(avail) - h[:, None] - h[None, :], 1.0).astype(np.float32)
        for use in (True, False):
            ctx._leaf = {}
            ctx.tables = core.Tables(core.lib()) if use else None
            try:
                with self.subTest(core=use):
                    np.testing.assert_array_equal(ctx.leaf_matrix(board, dead), ref)
            finally:
                ctx.tables = None


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

    def test_dealt_rivers_defer_their_games(self):
        # a 3bet pot's library flop: every dealt river's bucket game deferred (each turn line's, row by row together),
        # the same numbers as each game played where the walk reaches it
        got = []
        for defer in (True, False):
            with mock.patch.object(core, "DEFER_GAMES", defer):
                core._prog = None
                sol = _solve("9h8h6c", OOP_T, IP_T, 20, 40, fl.library_config(20, 40), 5)
                self.assertTrue(sol.solver.use_core and sol.solver._core_ok)
                entries = sum(isinstance(n.child, BucketEntry) for n in _chances(sol))
                self.assertGreater(entries, 10)
                self.assertEqual(len(core._prog.games), entries if defer else 0)
                arr = np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()] for d in _decisions(sol)])
                got.append((arr.view(np.int32), sol.exploitability()))
        core._prog = None
        np.testing.assert_array_equal(got[0][0], got[1][0])
        self.assertEqual(got[0][1], got[1][1])

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

    def test_an_iteration_under_the_clocks_resolution(self):
        """A first iteration the clock reads as 0 s (the browser's time.time() is whole milliseconds; a small river's
        iteration ~0.3 ms) runs on in the core: it had divided by it, and the error, taken for the core failing, turned
        the core off for the session (core._lib False: every later solve in numpy)."""
        with mock.patch.object(core, "_clock", lambda: 1.0):
            sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 40)
        self.assertIsNotNone(core.lib())
        self.assertEqual(sol.iters, 40)
        self.assertEqual(sol.solver.per_iter, core.PER_MIN)   # (run in the core: it sets per_iter)
        self.assertTrue(sol.solver._core_ok)

    def test_an_error_of_ours_in_run_leaves_the_core_on(self):
        """A Python error around the core in run() (here the factors' upload, once) runs that solve in numpy and leaves
        the core on: the next solve runs in it. A failure of the core itself (a run that didn't finish: CoreError)
        turns it off for the session, as a trap in the browser's module does."""
        want = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5, use_core=False).exploitability()
        real, calls = core.Program.iterate, []

        def once(prog, k, factors):
            calls.append(k)
            if len(calls) == 1:
                raise ValueError("ours")
            return real(prog, k, factors)
        with mock.patch.object(core.Program, "iterate", once):
            sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        self.assertEqual(calls, [1])                    # (numpy ran the rest: the core wasn't asked again)
        self.assertEqual(sol.iters, 5)
        self.assertAlmostEqual(sol.exploitability(), want, places=4)
        self.assertIsNotNone(core.lib())
        self.assertTrue(sol.solver._core_ok)
        nxt = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        self.assertIsNotNone(getattr(nxt.solver, "per_iter", None))   # (run in the core)
        L = core.lib()
        try:
            with mock.patch.object(type(L), "run", return_value=0):
                sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
            self.assertIsNone(core.lib())
            self.assertEqual(sol.iters, 5)
            self.assertAlmostEqual(sol.exploitability(), want, places=4)
        finally:
            core._lib = L

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
