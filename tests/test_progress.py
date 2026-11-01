"""Advisor.progress: a live street's answer in steps, for the browser to show while it solves (gto/static/worker.js
posts each as a partial): the request's own solve stopped at its looks (Advisor.first_look), read as it stands
(instant_from_own), then run on from there to the answer a request without looks gets."""
import json
import unittest
from unittest import mock

from gto import native
from gto import advisor as A
from gto.solver import Cancelled

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
BB3 = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), R("BB", 10), C("BTN")]
# a 3bet pot's turn (a shallow one: the turn's looks all the same), ~1 s natively to its 300th iteration
TURN = {"players": 6, "stack": 100, "preflop": BB3, "flop": [X("BB"), B("BTN", 6.8), C("BB")], "turn": [], "river": [],
        "hero": "BB", "hole": "QhJh", "board": "Ks7d2cTh"}
FLOP = dict(TURN, flop=[], board="Ks7d2c")   # (from the flop library: answered at once)
RIVER = dict(TURN, turn=[X("BB"), X("BTN")], board="Ks7d2cTh5s")
STUDY = dict({k: v for k, v in TURN.items() if k not in ("hero", "hole")}, study=True)   # (Study: no cards of its own)


def advisor(looks=None, settled=0.3):
    adv = native.advisor("fast", budget=30, prior_budget=0.5)
    adv.hu_settled = settled   # (the browser's Fast without threads; None: Accurate's, run to its 1000th iteration)
    if looks:
        adv.first_look = {"flop": looks, "turn": looks}
    return adv


def turn_iters(adv):
    return [s.iters for k, s in adv._cache.items() if len(k[2]) == 4]


class TestProgress(unittest.TestCase):
    def test_looks_then_the_same_answer(self):
        """Looks at the 100th and 200th iterations: each a partial read off the solve as it stands (source "solve",
        hero's mix, no EVs), nothing solved past it; then the answer, settled at the 300th, the same to the last digit
        as a request's without looks. Asked again (the worker's answer()), it comes from the cache."""
        cold = advisor().advise(TURN)
        self.assertEqual(cold["solver"]["iterations"], 300)
        adv = advisor((100, 200))
        got = []
        with mock.patch.object(A, "LOOK_MIN_S", 0.0):
            for it in (100, 200):
                got.append(adv.progress(TURN))
                self.assertEqual(turn_iters(adv), [it])
            got.append(adv.progress(TURN))
        for part, it in zip(got, (100, 200)):
            self.assertEqual(list(part), ["partial"])
            part = part["partial"]
            self.assertEqual(part["approximate"]["source"], "solve")
            self.assertEqual(part["approximate"]["iterations"], it)
            self.assertIn(f"({it} iterations)", part["approximate"]["label"])
            self.assertNotIn("solver", part)
            self.assertTrue(all("ev_bb" not in a for a in part["actions"]))
            self.assertAlmostEqual(sum(a["freq"] for a in part["actions"]), 1.0, delta=0.005)
            self.assertEqual([a["action"] for a in part["actions"]], [a["action"] for a in cold["actions"]])
        self.assertEqual(list(got[2]), ["result"])
        res = got[2]["result"]
        self.assertFalse(res.get("approximate"))
        self.assertEqual(res["solver"]["iterations"], 300)
        self.assertEqual(json.dumps(res["actions"]), json.dumps(cold["actions"]))
        self.assertEqual(json.dumps(adv.advise(TURN)["actions"]), json.dumps(cold["actions"]))
        self.assertEqual(turn_iters(adv), [300])

    def test_a_quick_solve_answers_in_full(self):
        """With under LOOK_MIN_S of the solve left to its first check at the look (a quick device, a small tree) it
        isn't stopped: progress answers in full, as advise does."""
        cold = advisor().advise(TURN)
        adv = advisor((100,))
        with mock.patch.object(A, "LOOK_MIN_S", 1e9):
            got = adv.progress(TURN)
        self.assertEqual(list(got), ["result"])
        self.assertEqual(got["result"]["solver"]["iterations"], 300)
        self.assertEqual(json.dumps(got["result"]["actions"]), json.dumps(cold["actions"]))

    def test_without_a_target(self):
        """Accurate's (no settled stop: on to its 1000th iteration): looks at the 100th and 300th, then the answer."""
        adv = advisor((100, 300), settled=None)
        with mock.patch.object(A, "LOOK_MIN_S", 0.0):
            got = [adv.progress(TURN) for _ in range(3)]
        self.assertEqual([g["partial"]["approximate"]["iterations"] for g in got[:2]], [100, 300])
        self.assertEqual(got[2]["result"]["solver"]["iterations"], 1000)
        self.assertFalse(got[2]["result"].get("approximate"))

    def test_study_looks(self):
        """Study's turn: each look the actor's whole range off the solve so far, nothing solved past it (a "node"
        answer: the grid, the combos and the range's frequencies, marked approximate, no EVs or equities), then
        Study's answer, the same as without looks."""
        cold = advisor().study(STUDY)
        adv = advisor((100, 200))
        got = []
        with mock.patch.object(A, "LOOK_MIN_S", 0.0):
            for it in (100, 200):
                got.append(adv.progress(STUDY))
                self.assertEqual(turn_iters(adv), [it])
            got.append(adv.progress(STUDY))
        for part, it in zip(got, (100, 200)):
            self.assertEqual(list(part), ["partial"])
            part = part["partial"]
            self.assertEqual((part["study"], part["actor"], part["street"]), ("node", "BB", "turn"))
            self.assertEqual(part["approximate"]["source"], "solve")
            self.assertEqual(part["approximate"]["iterations"], it)
            self.assertRegex(part["approximate"]["label"], rf"^this street's solve so far \({it} iterations, [\d.]+% of the pot")
            self.assertEqual((part["solver"]["iterations"], part["solver"]["exploitability_pct"]), (None, None))
            self.assertEqual(part["grid"]["actions"], cold["grid"]["actions"])
            self.assertEqual(sorted(part["combos"]), sorted(cold["combos"]))
            self.assertAlmostEqual(sum(a["freq"] for a in part["actions"]), 1.0, delta=0.01)
            self.assertTrue(all(a["ev_bb"] is None for a in part["actions"]))
            self.assertTrue(all(c["ev"] is None and c["evm"] is None and c["eq"] is None
                                for c in part["grid"]["cells"].values()))
            self.assertTrue(all(row[3] is None and row[4] is None for rows in part["combos"].values() for row in rows))
            self.assertTrue(all(c["eq"] is None for c in part["opponent_grid"]["cells"].values()))
            self.assertEqual((part["range_ev_bb"], part["range_eq"], part["opponent_eq"]), (None, None, None))
        self.assertEqual(list(got[2]), ["result"])
        res = got[2]["result"]
        self.assertFalse(res.get("approximate"))
        self.assertEqual(res["solver"]["iterations"], 300)
        for k in ("actions", "grid", "combos", "opponent_grid", "range_ev_bb", "range_eq"):
            self.assertEqual(json.dumps(res[k]), json.dumps(cold[k]), k)

    def test_stopped_at_a_look_and_after(self):
        """The page's stop (should_stop) during a look's run stops the request as it does an answer's: the solve is kept
        as it stands and the next request goes on from it."""
        adv = advisor((100,))
        real, inside, calls = A.solve, [], []

        def solve(board, *a, **kw):   # (the turn's solve running: stopped at its second check, after one iteration)
            inside.append(len(board))
            try:
                return real(board, *a, **kw)
            finally:
                inside.pop()
        adv.should_stop = lambda: inside[-1:] == [4] and (calls.append(1) or len(calls) >= 2)
        with mock.patch.object(A, "LOOK_MIN_S", 0.0):
            with mock.patch.object(A, "solve", solve), self.assertRaises(Cancelled):
                adv.progress(TURN)
            adv.should_stop = None
            done = turn_iters(adv)
            got = adv.progress(TURN)
        self.assertTrue(done and done[0] < 100, done)
        self.assertEqual(got["partial"]["approximate"]["iterations"], 100)

    def test_nothing_to_look_at(self):
        """None, nothing solved, where no look applies: no looks set, the river (quick anyway), Study's runout report,
        locks, a one-size view or custom ranges; a library flop answers in full."""
        self.assertIsNone(advisor().progress(TURN))
        adv = advisor((100,))
        for state in (RIVER, dict(STUDY, runouts=True), dict(TURN, locks={"x": 1}), dict(TURN, one_size=0),
                      dict(TURN, ranges={"BB": "AA"}), dict(TURN, board="")):
            self.assertIsNone(adv.progress(state), state)
        self.assertEqual(len(adv._cache), 0)
        got = adv.progress(FLOP)
        self.assertEqual(list(got), ["result"])
        self.assertEqual(got["result"]["street"], "flop")


if __name__ == "__main__":
    unittest.main()
