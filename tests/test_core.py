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
from gto import solver as solver_mod
from gto.cards import ALL_COMBOS, COMBO_INDEX, cards
from gto.ranges import parse_range
from gto.solver import BucketEntry, Config, Decision, Chance, solve

OOP = parse_range("22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,64s+,54s,A8o+,KTo+,QTo+,JTo")
IP = parse_range("22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,85s+,74s+,63s+,53s+,43s,A2o+,K8o+,Q9o+,J9o+,T8o+,98o")
OOP_T = parse_range("66+,A9s+,KTs+,QTs+,JTs,T9s,98s,AJo+,KQo")
IP_T = parse_range("22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,ATo+,KJo+,QJo")
ON = core.lib() is not None


class _Flaky:
    """The native core as the browser's module fails: its `at`th call of `what` ("run" or "eval") raises CoreError (a
    trap there), and reload() gives another (worker.js: a new instance), whose `then`th call fails (None: none)."""

    def __init__(self, L, what, at, then=None):
        self.L, self.what, self.at, self.then, self.calls = L, what, at, then, 0

    def __getattr__(self, k):
        return getattr(self.L, k)

    def _call(self, name, *a):
        if name == self.what:
            self.calls += 1
            if self.calls == self.at:
                raise core.CoreError("the core failed: unreachable")
        return getattr(self.L, name)(*a)

    def run(self, *a):
        return self._call("run", *a)

    def eval(self, *a):
        return self._call("eval", *a)

    def reload(self):
        return _Flaky(self.L, self.what, self.then)


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

    def test_first_chance_nodes_in_one_job(self):
        # With helpers a training pass runs the rows of every first chance node (none above it) as one job: a walk
        # down keeps each node's reaches, the job masks or spreads each row and runs it, the thread finishing a node's
        # last row sums its utilities, and the pass proper takes them (core_subs). The same floats bit for bit as a
        # job a node, on 2, 4 and 8 threads: a turn by river class full and trimmed (TRIMS' third level), a raked turn
        # dealing every river, a flop by hand buckets and the library's flop (its leaves batched before the pass); its
        # lanes measured (core_lanes(2)). A 3bet pot's library flop (deferred games) runs a job a node either way.
        from gto.advisor import Advisor
        fast = native.config("fast")
        dealt = Config(river_classes=False, n_rivers=None, rake_rate=0.05, rake_cap=3.0)
        spots = (("Ks7d2cTh", 20, 90, 20, fast), ("Ks7d2cTh", 9.1, 95.5, 20, Advisor._trim_cfg(fast, "turn", 3)),
                 ("Qs8d3c4h", 30, 40, 15, dealt), ("Kd7c2h", 5.5, 97.5, 6, fast),
                 ("Kd7c2h", 5.5, 97.5, 4, fl.library_config()))
        L = core.lib().L
        for board, pot, stack, iters, cfg in spots:
            with self.subTest(board, pot=pot):
                got = []
                try:
                    for t in (2, 4, 8):
                        self.assertEqual(core.threads(t), t)
                        for on in (1, 0):
                            self.assertEqual(L.core_subs(on), on)
                            sol = _solve(board, OOP_T, IP_T, pot, stack, cfg, iters)
                            self.assertTrue(sol.solver.use_core and sol.solver._core_ok)
                            if on:
                                self.assertGreater(L.core_lanes(2), 0)
                            got.append(np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()]
                                                       for d in _decisions(sol)]).view(np.int32))
                finally:
                    L.core_subs(1)
                    core.threads(1)
                for g in got[1:]:
                    np.testing.assert_array_equal(g, got[0])

    def test_equity_leaves_in_the_subs_job(self):
        # A training pass with subs multiplies the equity leaves above them (an all-in called on the turn) in the subs'
        # job, every reach on a matrix at once, and the pass proper takes their products (core_leaves): the same floats
        # bit for bit as multiplying them in the pass, and as one thread, on 2, 4 and 8 threads (a turn by river class
        # full and trimmed, a shorter stack, the threaded page's tree with the Advisor's ranges); a product is taken
        # only where its reach is the one kept (core_leaves_taken counts them)
        from gto.advisor import Advisor
        fast = native.config("fast")
        threaded = native.config("fast")
        threaded.bet_sizes["river"] = list(native.SPEEDS["accurate"][0]["bet_sizes"]["river"])
        btn = fl.spot_setup(next(s for s in fl.SPOTS if s[0] == "srp_btn_bb")[2])
        spots = (("Ks7d2cTh", OOP_T, IP_T, 20, 90, 20, fast), ("Ks7d2cTh", OOP_T, IP_T, 20, 25, 20, fast),
                 ("Ks7d2cTh", OOP_T, IP_T, 9.1, 95.5, 20, Advisor._trim_cfg(fast, "turn", 3)),
                 ("JcTd8d4h", btn["full0"], btn["full1"], 5.5, 97.5, 12, threaded))
        L = core.lib().L
        for board, r0, r1, pot, stack, iters, cfg in spots:
            with self.subTest(board, pot=pot, stack=stack):
                got, taken = [], []
                try:
                    for t, ons in ((1, (1,)), (2, (1, 0)), (4, (1, 0)), (8, (1, 0))):
                        self.assertEqual(core.threads(t), t)
                        for on in ons:
                            self.assertEqual(L.core_leaves(on), on)
                            before = L.core_leaves_taken()
                            sol = _solve(board, r0, r1, pot, stack, cfg, iters)
                            self.assertTrue(sol.solver.use_core and sol.solver._core_ok)
                            taken.append((t, on, L.core_leaves_taken() - before))
                            got.append(np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()]
                                                       for d in _decisions(sol)]).view(np.int32))
                finally:
                    L.core_leaves(1)
                    core.threads(1)
                for g in got[1:]:
                    np.testing.assert_array_equal(g, got[0])
                for t, on, k in taken:
                    if t == 1 or not on:
                        self.assertEqual(k, 0, (t, on))
                    else:
                        self.assertGreater(k, 0, (t, on))

    def test_lanes_hold_the_sums(self):
        # A helper runs its rows of a training pass's job (every first chance node's: subs) in a lane of its own, sized
        # by a serial pass over them in the second iteration (the first runs a job a node, sizing a whole pass's lanes
        # for the evaluations). That pass sums every terminal row, reached or not: its second player's pass meets the
        # first's regret-matched zeros, and skipping the showdowns nobody reached there sized the lanes short of a
        # later chunk's (a trap in a helper: natively the process ends, in the browser the solve never answers).
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
            self.assertGreater(r["lanes"][0][2], 0, r)
            self.assertEqual(r["lanes"][0], r["lanes"][1], r)
        self.assertTrue(rows[-1]["solver"]["iterations"] > 0, rows[-1])

    def test_evaluations_in_any_order(self):
        # An evaluation from the root of a program no training pass sized (one built to evaluate another solve than
        # the arena's, or a solve stopped before its first iteration) sizes the lanes itself, each kind its own (an
        # average or a best response, either player): a turn's best response for the first player takes more scratch
        # than the second's, so sized by the second's alone the first's trapped (and the reverse on a trimmed tree).
        # Every kind of program the Advisor builds (evaluate-only, a presolve stopped before its first iteration, a
        # solve, a warm-started and a seeded one after their next iteration), every evaluation in every order (24), on
        # 2, 4 and 8 threads, a turn full and trimmed: the same floats as on one thread, no lane short (none grown);
        # and a lane set short of its measure grows to what its helpers held, with the same floats (the training's
        # and the evaluations'), not a trap. In a child process: a trap ends it.
        import json
        import subprocess
        import sys
        p = subprocess.run([sys.executable, "-c", "import tests.test_core as t; t._evals_child()"], cwd=core.ROOT,
                           capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, PYTHONPATH=core.ROOT, GTO_THREADS="1"))
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        rows = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        self.assertEqual(len([r for r in rows if "kind" in r]), 2 * 3 * 5, rows)
        for r in rows:
            if "kind" in r:
                self.assertEqual((r["orders"], r["same"], r["grown"]), (24, True, 0), r)
            else:
                self.assertEqual((r["same"], r["grew"]), (True, True), r)

    def test_evaluations_below_the_root(self):
        # An evaluation below the root (a node's action EVs: Solver.node_action_evs evaluates each child) runs its
        # chance nodes' rows on the helpers once the lanes are sized (a subtree's chance nodes are the root's, a chunk
        # of their rows the same scratch), serially before (measured there they'd leave out the rest of the tree).
        # Every evaluation at the root's children, of a program built to evaluate (lanes unsized) and of one after
        # an iteration, on 2, 4 and 8 threads, a turn full and trimmed: the same floats as one thread's, before and
        # after the root's evaluations size the lanes; on the helpers only once sized; no lane grown. In a child
        # process: a trap ends it.
        import json
        import subprocess
        import sys
        p = subprocess.run([sys.executable, "-c", "import tests.test_core as t; t._below_child()"], cwd=core.ROOT,
                           capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, PYTHONPATH=core.ROOT, GTO_THREADS="1"))
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        rows = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        self.assertEqual(len(rows), 2 * 3 * 2, rows)
        for r in rows:
            self.assertTrue(r["same"], r)
            self.assertEqual(r["grown"], 0, r)
            first, again = r["helped"]
            if r["kind"] == "evaluate-only":
                self.assertEqual(first, 0, r)     # (serially: no lanes yet)
            else:
                self.assertGreater(first, 0, r)
            self.assertGreater(again, 0, r)


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
                lanes.append([int(L.core_lanes(k)) for k in (0, 1, 2)])
            print(json.dumps({"spot": i, "threads": t, "lanes": lanes}), flush=True)
    core.threads(8)
    F, X = (lambda p: {"pos": p, "action": "fold"}), (lambda p: {"pos": p, "action": "check"})
    C, B = (lambda p: {"pos": p, "action": "call"}), (lambda p, a: {"pos": p, "action": "bet", "amount": a})
    adv = native.advisor("fast", budget=1.0, prior_budget=1.0)
    res = adv.advise({"players": 6, "stack": 100, "hero": "SB", "hole": "QhJh", "board": "Ks7d2cTh",
                      "preflop": [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")],
                      "flop": [X("SB"), B("BB", 1.5), C("SB")], "turn": [], "river": []})
    print(json.dumps({"solver": res.get("solver"), "error": res.get("error")}), flush=True)


def _evals_child():
    """test_evaluations_in_any_order's evaluations (a child process): a line of JSON a tree, thread count and kind of
    program; then the short lanes'."""
    import itertools
    import json
    from gto import solver as solver_mod
    from gto.advisor import Advisor
    L = core.lib().L
    fast = native.config("fast")
    evals = [(md, p) for md in (core.AVG, core.BR) for p in (0, 1)]

    def arrays(sol):
        return [a for d in _decisions(sol) for a in (d.regret, d.ssum)]

    def load(sol, saved, iters, horizon):
        for a, b in zip(arrays(sol), saved):
            a[...] = b
        sol.solver.iters, sol.solver.horizon = iters, horizon

    def seed(root, s):   # (a pure check or a pure bet a combo: the first pass's zeros, as Advisor._store_seed's)
        root.regret[0][0, 0::2] = root.regret[0][-1, 1::2] = np.float32(root.pot_now)

    def evaluate(sol, order):
        a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
        out = {}
        for md, p in order:
            u = core.evaluate(sol.solver, [(sol.root, p, a, b, md)])
            assert u is not None and core._lib is not False, (md, p)
            out[md, p] = u[0].view(np.int32).copy()
        return out

    for name, cfg in (("full", fast), ("trimmed", Advisor._trim_cfg(fast, "turn", 3))):
        core.threads(1)
        A = solve(cards("Ks7d2cTh"), OOP, IP, 9.1, 95.5, cfg=cfg, time_budget=1e9, max_iters=4)
        W = solve(cards("Ks7d2cTh"), OOP, IP, 9.1, 95.5, cfg=cfg, time_budget=1e9, max_iters=1, warm=A)
        D = solve(cards("Ks7d2cTh"), OOP, IP, 9.1, 95.5, cfg=cfg, time_budget=1e9, max_iters=1, seed=seed)
        a0 = [x.copy() for x in arrays(A)]
        st = (A.solver.iters, A.solver.horizon)

        def ready(kind):
            """the program of `kind` in the arena, its lanes unmeasured: the solve to evaluate"""
            load(A, a0, *st)
            core._prog = None
            sol = A
            if kind == "presolve stopped":
                try:
                    A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 1, min_iters=1, should_stop=lambda: True)
                except solver_mod.Cancelled:
                    pass
                assert core._prog is not None and core._prog.valid(core.lib(), A.solver)
            elif kind == "solve":
                A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 1, min_iters=1)
            elif kind in ("warm-started", "seeded"):
                sol = W if kind == "warm-started" else D
                load(sol, [np.zeros_like(x) for x in arrays(sol)], 0, 1)
                if sol is W:
                    solver_mod.warm_start(W.root, W.ctx, A.root, A.ctx)
                else:
                    seed(D.root, D.solver)
                sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=1, min_iters=1)
            return sol

        kinds = ("evaluate-only", "presolve stopped", "solve", "warm-started", "seeded")
        ref = {k: evaluate(ready(k), evals) for k in kinds}
        for t in (2, 4, 8):
            core.threads(t)
            for kind in kinds:
                before, same, orders = L.core_lanes_grown(-1, 0), True, 0
                for order in itertools.permutations(evals):
                    got = evaluate(ready(kind), order)
                    same &= all(np.array_equal(got[k], ref[kind][k]) for k in evals)
                    orders += 1
                print(json.dumps({"tree": name, "threads": t, "kind": kind, "orders": orders, "same": bool(same),
                                  "grown": int(L.core_lanes_grown(-1, 0) - before)}), flush=True)
        # a lane short of its measure (3/4 of it): the subs' in a training pass, then the whole pass's in an evaluation
        core.threads(4)
        load(A, a0, *st)
        core._prog = None
        A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 3, min_iters=1)
        want = [x.view(np.int32).copy() for x in arrays(A)], evaluate(A, evals)
        load(A, a0, *st)
        core._prog = None
        A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 2, min_iters=1)   # (both measured)
        lanes, grew = [int(L.core_lanes(k)) for k in (0, 2)], []
        before = L.core_lanes_grown(2, lanes[1] * 3 // 4)
        A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 3, min_iters=1)
        grew.append(L.core_lanes_grown(0, lanes[0] * 3 // 4) > before and L.core_lanes(2) > lanes[1] * 3 // 4)
        before = L.core_lanes_grown(-1, 0)
        got = [x.view(np.int32).copy() for x in arrays(A)], evaluate(A, evals)
        grew.append(L.core_lanes_grown(-1, 0) > before and L.core_lanes(0) > lanes[0] * 3 // 4)
        same = all(np.array_equal(x, y) for x, y in zip(got[0], want[0])) and \
            all(np.array_equal(got[1][e], want[1][e]) for e in evals)
        print(json.dumps({"tree": name, "lanes": lanes, "same": bool(same), "grew": all(grew), "each": grew}),
              flush=True)
    core.threads(1)


def _below_child():
    """test_evaluations_below_the_root's (a child process): a line of JSON a tree, thread count and kind of program."""
    import json
    from gto.advisor import Advisor
    L = core.lib().L
    fast = native.config("fast")
    evals = [(md, p) for md in (core.AVG, core.BR) for p in (0, 1)]
    for name, cfg in (("full", fast), ("trimmed", Advisor._trim_cfg(fast, "turn", 3))):
        core.threads(1)
        A = solve(cards("Ks7d2cTh"), OOP, IP, 9.1, 95.5, cfg=cfg, time_budget=1e9, max_iters=4)
        saved = [x.copy() for d in _decisions(A) for x in (d.regret, d.ssum)]
        st = (A.solver.iters, A.solver.horizon)
        a, b = A.r0[None, :].astype(np.float32), A.r1[None, :].astype(np.float32)

        def ready(kind):
            """A as it was, its program built for an evaluation alone or by one more iteration (lanes sized)"""
            for x, y in zip([x for d in _decisions(A) for x in (d.regret, d.ssum)], saved):
                x[...] = y
            A.solver.iters, A.solver.horizon = st
            core._prog = None
            if kind == "solve":
                A.solver.run(A.r0, A.r1, time_budget=1e9, max_iters=st[0] + 1, min_iters=1)

        def evaluate(nodes):
            out = []
            for md, p in evals:
                u = core.evaluate(A.solver, [(c, p, a, b, md) for c in nodes])
                assert u is not None and core._lib is not False, (md, p)
                out += [x.view(np.int32).copy() for x in u]
            return out

        ref = {}
        for kind in ("evaluate-only", "solve"):
            ready(kind)
            ref[kind] = evaluate(A.root.children), evaluate([A.root])
        for t in (2, 4, 8):
            core.threads(t)
            for kind in ("evaluate-only", "solve"):
                ready(kind)
                h0 = L.core_helped()
                first = evaluate(A.root.children)      # (evaluate-only: before any lane is sized)
                h1 = L.core_helped()
                root = evaluate([A.root])
                h2 = L.core_helped()
                again = evaluate(A.root.children)
                h3 = L.core_helped()
                same = all(np.array_equal(x, y) for x, y in zip(first + again + root, ref[kind][0] * 2 + ref[kind][1]))
                print(json.dumps({"tree": name, "threads": t, "kind": kind, "same": bool(same), "helped":
                                  [int(h1 - h0), int(h3 - h2)], "grown": int(L.core_lanes_grown(-1, 0))}), flush=True)
    core.threads(1)


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
    # (the 3bet pot's the same: 1.27e-4 off on regrets of 9 at its worst node since Config.force_allin merged the sizes
    # near the stack, 120 -> 88 decisions; 5.9e-6 before, the median 1.8e-7 either way)
    REGRET_TOL = {"flop by buckets": 1e-4, "flop by buckets, 3bet pot raked": 5e-5}
    # the library's products summed in another order than numpy's sgemm, on the core's own loops (no BLAS), or by BLAS
    # with the combos laid out by window (Solver._perm: 1.5e-5 on the 3bet pot's): 2.2e-5
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
                loops = name.startswith("library")
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
        # numpy's solve is at 0.10. (Compared from 200 iterations: at 100 rounding alone moves it 0.67-0.81, the
        # ranges nudged by an ulp; 0.19-0.20 at 200)
        s = fl.spot_setup(next(x for x in fl.SPOTS if x[0] == "srp_utg_bb")[2])
        cfg = fl.library_config()          # (the entry's tree then: the river blind)
        a, b = (_solve("AcQdTh", s["full0"], s["full1"], s["pot"], s["stack"], cfg, 100, use_core=u)
                for u in (True, False))
        b.solver.use_core = False
        for iters in (100, 200, 300):
            for x in (a, b):
                x.extend(1e9, max_iters=iters)
            self.assertEqual((a.iters, b.iters), (iters, iters))
            self.assertTrue(np.isfinite(a.solver._regret).all())
            # (0.632 and 0.756% of the pot at 100, 0.194 and 0.202 at 200, 0.096 and 0.102 at 300)
            if iters > 100:
                self.assertLess(abs(a.exploitability() - b.exploitability()), 0.1)


@unittest.skipUnless(ON, "no C compiler for the native core")
class TestWindows(unittest.TestCase):
    """Each player's combos alone (Solver._support; the core lays them out as windows): a combo with no weight in a
    player's range never has reach, so it's left out of that player's arithmetic, its regrets, sums and EVs 0, the
    rest the same to rounding; numpy leaves out the same. (Whatever their size here: WINDOW_MAX 1.)"""

    def setUp(self):
        p = mock.patch.object(solver_mod, "WINDOW_MAX", 1.0)
        p.start()
        self.addCleanup(p.stop)

    def test_combos_outside_a_range(self):
        # a turn by river class, OOP's narrow range against IP's wider one, and a hand OOP never holds at 1e-3 (the
        # Advisor's _with_hero for a hand outside its range): kept, with its EVs
        r0 = OOP_T.copy()
        hand = COMBO_INDEX[tuple(sorted(cards("7h2d")))]
        self.assertEqual(r0[hand], 0)
        r0[hand] = 1e-3
        pot = 20
        wins = []
        for use_core in (True, False):
            with self.subTest(use_core=use_core):
                a = _solve("Ks7d2cTh", r0, IP_T, pot, 90, native.config("fast"), 30, use_core=use_core)
                S = a.solver
                if use_core:
                    self.assertIs(core._prog.s(), S)
                    self.assertIsNotNone(core._prog.perm)
                win = S._win
                wins.append(win)
                rr = (np.ravel(a.r0) > 0, np.ravel(a.r1) > 0)
                for q in (0, 1):
                    self.assertTrue(win[q][rr[q]].all())                # (the ranges, and their images by class)
                self.assertLess(win[0].mean(), 0.75)
                ci = int(np.flatnonzero(S.ctx.active == hand)[0])
                self.assertTrue(win[0][ci])
                for d in _decisions(a):
                    if not d.bucketed:
                        self.assertFalse(d.regret[..., ~win[d.player]].any())
                        self.assertFalse(d.ssum[..., ~win[d.player]].any())
                p = S.root.player
                ev, br, ex = (S.node_action_evs(S.root, a.r0, a.r1), S.node_action_br_evs(S.root, a.r0, a.r1),
                              S.exploitability(a.r0, a.r1))
                self.assertFalse(ev[:, ~win[p]].any())
                self.assertFalse(br[:, ~win[p]].any())
                # in the range, the EVs of the same sums worked out on every combo (the windows off)
                keep = (S._win, S._wmask, S._perm, S._wins)
                S._win = S._wmask = S._perm = S._wins = None
                try:
                    ev1, br1, ex1 = (S.node_action_evs(S.root, a.r0, a.r1), S.node_action_br_evs(S.root, a.r0, a.r1),
                                     S.exploitability(a.r0, a.r1))
                finally:
                    S._win, S._wmask, S._perm, S._wins = keep
                m = rr[p]
                self.assertLess(float(np.abs(ev[:, m] - ev1[:, m]).max()), 1e-4 * pot)
                self.assertLess(float(np.abs(br[:, m] - br1[:, m]).max()), 1e-4 * pot)
                self.assertTrue(np.abs(ev1[:, ci]).max() > 0.1)          # (the 1e-3 hand's, worked out)
                self.assertAlmostEqual(ex, ex1, delta=1e-4)
        self.assertTrue(np.array_equal(*wins))

    def test_off(self):
        a = _solve("Ks7d2cTh", OOP_T, IP_T, 20, 90, native.config("fast"), 5)
        with mock.patch.object(core, "lib", return_value=core.lib()):
            S = a.solver
            self.assertIsNotNone(S._win)
            S.windows = False
            S.run(a.r0, a.r1, time_budget=1e9, max_iters=6)
            self.assertIsNone(S._win)
            self.assertIsNone(core._prog.perm)

    def test_a_download_leaves_the_arena_in_the_solvers_order(self):
        # a download (core.run's last step) reads the buffers back and leaves the arena's in the solver's order, not
        # laid out again: the next upload writes them anew, and an iteration or evaluation without one lays them out
        # first (_laid), as an upload would. A solve run on after evaluations is the same bit for bit.
        cfg = native.config("fast")
        a = _solve("Ks7d2cTh", OOP_T, IP_T, 20, 90, cfg, 6)
        S, prog = a.solver, core._prog
        self.assertIs(prog.s(), S)
        self.assertTrue(prog.runs)
        self.assertEqual(prog._plain, {prog.reg})      # (the sums uploaded since, for the solve's root EVs)
        raw = np.empty_like(S._regret)
        prog.L.read(prog.reg, raw)
        self.assertTrue(np.array_equal(raw.view(np.int32), S._regret.view(np.int32)))
        prog._laid(prog.reg)
        self.assertEqual(prog._plain, set())
        laid = np.empty_like(raw)
        prog.L.read(prog.reg, laid)
        prog._put(prog.reg, S._regret)
        prog.L.read(prog.reg, raw)
        self.assertTrue(np.array_equal(raw.view(np.int32), laid.view(np.int32)))
        self.assertFalse(np.array_equal(raw, S._regret))        # (laid out: another order)
        # an evaluation uploads the sums and leaves the regrets as the download did
        prog._get(prog.reg, S._regret)
        ex = S.exploitability(a.r0, a.r1)
        self.assertEqual(prog._plain, {prog.reg})
        S.run(a.r0, a.r1, time_budget=1e9, max_iters=12, min_iters=1)
        b = _solve("Ks7d2cTh", OOP_T, IP_T, 20, 90, cfg, 6)
        self.assertEqual(b.solver.exploitability(b.r0, b.r1), ex)
        b.solver.run(b.r0, b.r1, time_budget=1e9, max_iters=12, min_iters=1)
        self.assertEqual((S.iters, b.solver.iters), (12, 12))
        for x, y in ((S._regret, b.solver._regret), (S._ssum, b.solver._ssum)):
            self.assertTrue(np.array_equal(x.view(np.int32), y.view(np.int32)))

    def test_none_where_they_leave_out_little(self):
        # the players' combos over WINDOW_MAX of the tree's on average: no windows (here 0.51 and 1 of them, 0.75)
        a = _solve("Ks7d2cTh", OOP_T, IP_T, 20, 90, native.config("fast"), 5)
        self.assertAlmostEqual(float(a.solver._win.mean()), 0.754, delta=0.01)
        with mock.patch.object(solver_mod, "WINDOW_MAX", 0.7):
            b = _solve("Ks7d2cTh", OOP_T, IP_T, 20, 90, native.config("fast"), 5)
            self.assertIsNone(b.solver._win)
            self.assertIsNone(core._prog.perm)


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

    def test_a_batch_builds_each_table_the_same_on_any_number_of_threads(self):
        # a flop's tables in batches (Tables.batch, core_tables: as Program._tables_ahead builds them, the threads taking a
        # table each, the flop's equity leaf in chunks of rows) and one by one (core_table): the same floats, bit for
        # bit, on 1, 3 and 8 threads, the helpers' lanes never short (sized from the tables, not measured)
        cfg = native.config("fast")
        sol = solve(cards("Kd7c2h"), OOP_T, IP_T, pot=5.5, stack=97.5, cfg=cfg, time_budget=1e9, max_iters=0)
        ctx, dead, board = sol.solver.ctx, sol.solver.dead, tuple(cards("Kd7c2h"))
        turns = [board + (t,) for t, _ in ctx.turn_classes(board, dead)[:5]]
        K = cfg.river_buckets
        L = core.lib()

        def build(batch):
            ctx._stacks = {k: v for k, v in ctx._stacks.items() if k[0] == "S"}
            ctx._leaf = {}
            ctx.tables = core.Tables(L)
            try:
                if batch:
                    ctx.tables.batch(ctx, [ctx.table_request("leaf", board, dead)] +
                                     [ctx.table_request("bucket", b, dead, K) for b in turns])
                    ctx.tables.batch(ctx, [ctx.table_request("river", b, dead, cfg.river_strength, outer)
                                           for b in turns for outer in (None, K)])
                    self.assertEqual(len(ctx.tables.ready), 1 + 2 * len(turns))
                out = [ctx._flop_leaf(board, dead)]
                for b in turns:
                    out += ctx.bucket_table(b, dead, K)
                    for outer in (None, K):
                        out += ctx.river_table(b, dead, cfg.river_strength, outer)
                self.assertEqual(ctx.tables.ready, {})            # (each taken)
                return out
            finally:
                ctx.tables = None

        ref = build(False)
        try:
            for t in (1, 3, 8):
                with self.subTest(threads=t):
                    self.assertEqual(core.threads(t), t)
                    grown, helped = L.L.core_lanes_grown(-1, 0), L.L.core_helped()
                    got = build(True)
                    self.assertEqual(L.L.core_lanes_grown(-1, 0), grown)
                    self.assertEqual(L.L.core_helped() > helped, t > 1)
                    self.assertEqual(len(got), len(ref))
                    for a, b in zip(got, ref):
                        np.testing.assert_array_equal(a, b)
        finally:
            core.threads(1)

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
        with mock.patch.object(solver_mod, "WINDOW_MAX", 1.0):
            sol = solve(cards("Kd7c2h"), OOP_T, IP_T, pot=5.5, stack=97.5, cfg=fl.library_config(), max_iters=0)
        S = sol.solver
        self.assertIsNotNone(S._perm)
        self.assertTrue(S.batch_leaves)
        prog = core._program(core.lib(), S, sol.r0, sol.r1)
        self.assertGreater(prog.leaf_rows, core.LEAF_ROWS_BLAS)
        self.assertEqual(prog.stacks, 2)                  # (the flop's all-in leaves, the turns')
        # (each player's window: its range, on a flop)
        self.assertEqual([len(x) for x in prog.live], [int((np.ravel(r) > 0).sum()) for r in (sol.r0, sol.r1)])
        r0 = np.where(np.ravel(sol.r0) > 0, sol.r0, 0.5)   # a combo outside the range: an evaluation there is numpy's
        self.assertIsNone(core.evaluate(S, [(S.root, 0, r0, sol.r1, core.AVG)]))
        S.windows = False                                  # (without windows, the training untrimmed)
        S._windows(sol.r0, sol.r1)
        prog = core._program(core.lib(), S, sol.r0, sol.r1)
        self.assertEqual([len(x) for x in prog.live], [int((np.ravel(r) > 0).sum()) for r in (sol.r0, sol.r1)])
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
        turns it off for the session natively, with no new instance to start (in the browser it starts again:
        test_a_failed_run_starts_the_core_again)."""
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

    def test_a_failed_pass_after_the_solve_turns_the_core_off(self):
        """The core failing in the passes after a solve (core.evaluate: in the browser, say, a helper that failed
        holding rows, which the thread waiting on them traps on) runs them in numpy and, natively, turns the core off,
        as a failed run does: the answer still comes, and the next solve runs in numpy."""
        sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        want = sol.exploitability()
        L = core.lib()
        try:
            with mock.patch.object(type(L), "eval", side_effect=core.CoreError("the core failed")):
                got = sol.exploitability()
            self.assertIsNone(core.lib())
            self.assertAlmostEqual(got, want, places=4)
            nxt = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
            self.assertEqual(nxt.iters, 5)
            self.assertIsNone(getattr(nxt.solver, "per_iter", None))   # (numpy ran it)
        finally:
            core._lib = L

    def test_a_failed_run_starts_the_core_again(self):
        """In the browser a trap in a run (a helper that failed, the memory refusing to grow) starts the core again
        (core.restart: worker.js reload, a new instance on a fresh memory with new helpers) and runs the solve again in
        it, from the solver's arrays as they were before the run (a failed run's arena is never read back): the same
        floats as a run that didn't fail, failing on its first chunk or a later one. Failing again in the new instance,
        or once RESTARTS are used up, numpy runs the solve and the core is off, as after a failure before."""
        cfg = native.config("fast")
        go = lambda: solve(cards("Ks7d2cTh"), OOP_T, IP_T, 20, 90, cfg=cfg, time_budget=1e9, max_iters=40)
        L = core.lib()
        core._prog = None
        want = go()
        with mock.patch.object(core, "lib", return_value=None):
            want_np = go().exploitability()
        flat = lambda sol: np.concatenate([np.r_[d.regret.ravel(), d.ssum.ravel()] for d in _decisions(sol)]).view(np.int32)
        try:
            for at in (1, 2):
                with self.subTest(failing_chunk=at):
                    first = core._lib = _Flaky(L, "run", at)
                    core._restarts = 0
                    sol = go()
                    self.assertEqual(core._restarts, 1)
                    self.assertIsInstance(core.lib(), _Flaky)
                    self.assertIsNot(core.lib(), first)
                    self.assertEqual(first.calls, at)
                    self.assertEqual(sol.iters, 40)
                    self.assertIsNotNone(getattr(sol.solver, "per_iter", None))   # (run in the core)
                    np.testing.assert_array_equal(flat(sol), flat(want))
                    self.assertEqual(sol.exploitability(), want.exploitability())
            for name, restarts, then in (("again in the new instance", 0, 1), ("restarts used up", core.RESTARTS, None)):
                with self.subTest(name):
                    core._lib, core._restarts = _Flaky(L, "run", 2, then=then), restarts
                    sol = go()
                    self.assertIsNone(core.lib())
                    self.assertEqual(sol.iters, 40)
                    self.assertIsNone(getattr(sol.solver, "per_iter", None))   # (numpy ran it)
                    self.assertEqual(sol.exploitability(), want_np)
        finally:
            core._lib, core._restarts, core._prog = L, 0, None

    def test_a_failed_pass_after_the_solve_starts_the_core_again(self):
        """The passes after a solve (core.evaluate) the same: the core started again and the passes run in it."""
        sol = _solve("Ks7d2cTh5s", OOP, IP, 20, 90, native.config("fast"), 5)
        L = core.lib()
        want = sol.exploitability()
        try:
            core._lib, core._restarts, core._prog = _Flaky(L, "eval", 1), 0, None
            self.assertEqual(sol.exploitability(), want)
            self.assertEqual(core._restarts, 1)
            self.assertIsInstance(core.lib(), _Flaky)
            self.assertGreater(core.lib().calls, 0)   # (the passes ran in the new instance)
        finally:
            core._lib, core._restarts, core._prog = L, 0, None

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
