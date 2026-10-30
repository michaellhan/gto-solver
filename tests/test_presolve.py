"""Advisor.presolve: the solve your next decision on a street reads, run while the opponent is to act (the browser's
worker runs it while you enter their move and the page stops it when you do), so the advice that follows goes on from
it; only where the instant answer is offered (heads-up, at equilibrium) and the street is solved live."""
import unittest
from unittest import mock

import numpy as np

from gto import advisor as A
from gto import native
from gto.cards import cards as parse_cards
from gto.ranges import parse_range
from gto.solver import Cancelled, solve

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
LIMP = [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")]   # (no flop library: every street solved live)
# hero the big blind, in position after the flop: the small blind acts first on each street
PRE = {"players": 6, "stack": 100, "preflop": LIMP, "flop": [X("SB"), B("BB", 1.5), C("SB")], "turn": [], "river": [],
       "hero": "BB", "hole": "QhJh", "board": "Ks7d2cTh"}
CHECKED = dict(PRE, turn=[X("SB")])
OFF = dict(PRE, turn=[B("SB", 3)])   # (60% of the pot: the turn's tree has 33/75/125%)
SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")]


def advisor(budget=1.0):
    return native.advisor("fast", budget=budget, prior_budget=0.5)


def turn_solves(adv):
    return {k: s for k, s in adv._cache.items() if s is not None and s.ctx is not None and len(k[2]) == 4}


class TestPresolve(unittest.TestCase):
    def test_the_advice_goes_on_from_it(self):
        """The presolve makes the turn's solve with hero's cards as the advice would; the advice once the small blind
        checks finds it (no second turn solve) with its time counted against the budget, not run again."""
        adv = advisor()
        res = adv.presolve(PRE, "SB")
        self.assertEqual(res["presolved"], "turn")
        self.assertGreater(res["iterations"], 0)
        (key, sol), = turn_solves(adv).items()
        self.assertIn(key, adv._presolved)
        iters, secs = sol.iters, sol.seconds
        calls = []
        real = A.solve
        with mock.patch.object(A, "solve", lambda board, *a, **kw: calls.append(len(board)) or real(board, *a, **kw)):
            out = adv.advise(CHECKED)
        self.assertNotIn(4, calls)
        self.assertIs(turn_solves(adv)[key], sol)
        self.assertGreaterEqual(out["solver"]["iterations"], iters)
        self.assertGreaterEqual(sol.seconds, secs)
        self.assertNotIn(key, adv._presolved)   # (the hand's own now)
        self.assertTrue(out["recommended"])

    def test_stopped_part_way_kept_and_resumed(self):
        """Stopped in its first piece (before the first check, where the page's stop usually lands) the solve stays
        cached with the iterations it ran and their time, and the advice goes on from them."""
        adv = advisor(budget=2.0)
        adv.advise(dict(PRE, turn=[], board="Ks7d2c", flop=[X("SB")]))   # (the flop solved: the turn's narrowing)
        n = [None]
        real = A.solve

        def turn_first(board, *a, **kw):   # (the stop counts the turn solve's checks only)
            if len(board) == 4:
                n[0] = 0
            return real(board, *a, **kw)

        def stop():
            if n[0] is not None:
                n[0] += 1
            return (n[0] or 0) > 2
        adv.should_stop = stop
        with self.assertRaises(Cancelled), mock.patch.object(A, "solve", turn_first):
            adv.presolve(PRE, "SB")
        (key, sol), = turn_solves(adv).items()
        self.assertTrue(sol.stopped)
        self.assertGreater(sol.solver.iters, 0)
        self.assertGreater(sol.seconds, 0)
        self.assertIn(key, adv._presolved)
        self.assertFalse(adv._presolving)
        done = sol.solver.iters
        adv.should_stop = None
        out = adv.advise(CHECKED)
        self.assertIs(turn_solves(adv)[key], sol)
        self.assertFalse(sol.stopped)
        self.assertGreater(out["solver"]["iterations"], done)

    def test_a_move_off_its_tree_is_never_read(self):
        """The small blind leads a size the presolve's tree doesn't hold: the instant answer doesn't read the
        presolve's turn at its nearest size (as it does your own earlier advice's on the street; the flop's view has
        no turn lead, so there's none), and the advice solves the turn with the size in."""
        adv = advisor()
        adv.presolve(PRE, "SB")
        pre = set(turn_solves(adv))
        self.assertIsNone(adv.instant(OFF))
        held = set(adv._presolved)
        adv._presolved.clear()   # (as if your own advice had made it: read)
        self.assertEqual(adv.instant(OFF)["approximate"]["source"], "solve")
        adv._presolved |= held
        out = adv.advise(OFF)
        self.assertTrue(set(turn_solves(adv)) - pre)
        self.assertTrue(out["recommended"])

    def test_none_where_it_doesnt_apply(self):
        """Nothing solved: hero to act, hero's own seat, the flop not dealt, more than the equilibrium asked (pool,
        locks, custom sizes or ranges, a tournament, Study, a jam), a multiway pot, a flop the library holds."""
        adv = advisor()
        three = [F("UTG"), F("HJ"), F("CO"), C("BTN"), C("SB"), X("BB")]
        cases = [(CHECKED, "SB"), (PRE, "BB"), (dict(PRE, board="", flop=[], turn=[]), "SB"),
                 (dict(PRE, opponents="pool"), "SB"), (dict(PRE, locks={"SB": {"fold": 10}}), "SB"),
                 (dict(PRE, sizes={"turn": [50]}), "SB"), (dict(PRE, payouts=[50, 30, 20]), "SB"),
                 (dict(PRE, study=True), "SB"), (dict(PRE, jam=True), "SB"),
                 (dict(PRE, hero="BTN", preflop=three, flop=[X("SB"), X("BB"), X("BTN")]), "SB"),
                 (dict(PRE, hero="BTN", preflop=SRP, flop=[], board="Ks7d2c"), "BB")]
        calls = []
        with mock.patch.object(A, "solve", lambda *a, **kw: calls.append(1)), \
                mock.patch.object(A.Advisor, "_multiway_solve", lambda *a, **kw: calls.append(2)):
            for st, seat in cases:
                self.assertIsNone(adv.presolve(st, seat), (st, seat))
        self.assertEqual(calls, [])
        self.assertEqual(adv._cache, {})
        self.assertFalse(adv._presolving)


class TestCancelledSolve(unittest.TestCase):
    def test_the_solve_so_far_rides_on_the_exception(self):
        """solve() stopped part way raises Cancelled with the solve so far (e.sol): its iterations, tree and solver,
        which run on from there."""
        oop, ip = parse_range("66+,A9s+,KTs+,QTs+,AJo+"), parse_range("22+,A2s+,K9s+,Q9s+,ATo+")
        n = [0]

        def stop():
            n[0] += 1
            return n[0] > 2
        with self.assertRaises(Cancelled) as e:
            solve(parse_cards("Ks7d2cTh"), oop, ip, 20, 90, cfg=native.config("fast"), time_budget=1e9,
                  max_iters=500, should_stop=stop)
        sol = e.exception.sol
        self.assertGreater(sol.solver.iters, 0)
        self.assertEqual(sol.iters, sol.solver.iters)
        self.assertGreater(sol.cells, 0)
        start = sol.solver.iters
        sol.extend(1e9, max_iters=start + 5)
        self.assertEqual(sol.iters, start + 5)
        self.assertTrue(np.allclose(sol.strategy[:, sol.ctx.active].sum(0), 1, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
