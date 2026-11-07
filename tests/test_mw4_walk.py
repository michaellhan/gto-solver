"""A 4-way flop narrows the next street through the solve your advice came from (Advisor._mw_walk), and nothing else
changes: no new solve, the betting rules where the solve can't read the line. scripts/bench_mw4_narrowing.py has what it
gains; here, on the hand that found it, what it does."""
import unittest

import numpy as np

from gto import advisor as A
from gto import native
from gto.cards import ALL_COMBOS, cards

PRE = [{"pos": "UTG", "action": "call"}, {"pos": "LJ", "action": "raise", "to": 4.0}, {"pos": "HJ", "action": "call"},
       {"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"},
       {"pos": "BB", "action": "call"}, {"pos": "UTG", "action": "call"}]
FLOP = "2s4d6c"
FOLD = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
CALL = lambda p: {"pos": p, "action": "call"}
CHECK = lambda *ps: [{"pos": p, "action": "check"} for p in ps]


def lead(amount, *rest):
    return [{"pos": "BB", "action": "bet", "amount": amount}] + list(rest)


LINE = lead(6.5, *FOLD("UTG"), CALL("LJ"), *FOLD("HJ"))     # (39% of the pot: the hand; the tree's bet is 33%)


def state(board, flop, turn=(), hole="As4s", hero="BB"):
    return {"players": 7, "stack": 100, "preflop": PRE, "board": board, "hero": hero, "hole": hole, "flop": list(flop),
            "turn": list(turn)}


class TestWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = native.advisor("fast", library=False)
        cls.adv.mw_max_iters, cls.adv.mw_budget = 75, 1e6
        res = cls.adv.advise(state(FLOP, []))        # the flop's advice: the solve paid for
        assert res.get("solved"), res.get("notes")
        cls.solves = len(cls.adv._mw_cache)

    def ranges(self, flop, turn="6h"):
        """The turn's ranges and notes (the rules' too): the answer itself isn't needed."""
        got = {}

        class Done(BaseException):
            pass
        walk = self.adv._mw_walk

        def probe(ranges, before, board, st, stk, hero, hole, streets, notes, current=False):
            rules = dict(ranges)
            ok = walk(ranges, before, board, st, stk, hero, hole, streets, notes, current=current)
            if st.name == "flop":
                got.update(rules=rules, ranges=dict(ranges), ok=ok, notes=list(notes))
                raise Done()
            return ok
        self.adv._mw_walk = probe
        try:
            self.adv.advise(state(FLOP + turn, flop))
        except Done:
            pass
        finally:
            del self.adv._mw_walk
        return got

    def test_the_call_reads_the_solve(self):
        got = self.ranges(LINE)
        self.assertTrue(got["ok"])
        self.assertEqual(len(self.adv._mw_cache), self.solves)      # no solve of its own
        self.assertTrue(any(n.startswith("flop: 4-way, ranges narrowed through the solve your advice came from")
                            for n in got["notes"]), got["notes"])
        sol, players, expl = next(h for h in self.adv._mw_lines.values() if len(h[1]) == 4)
        trust = A.trust_of(expl, "flop")
        self.assertGreater(trust, 0.5)
        node = sol.root
        R = sol.reach0[:, 0, :].astype(float).copy()
        for a in LINE:
            i = A._mw_pick(node, a)
            p = node.player
            R[p] *= (1 - self.adv.prior_smoothing) * node.avg_strategy()[0][i] + self.adv.prior_smoothing / len(node.names)
            node = node.children[i]
        lj = players.index("LJ")
        want = trust * sol.expand(R[lj]) + (1 - trust) * got["rules"]["LJ"]
        self.assertLess(np.abs(got["ranges"]["LJ"] - want).max(), 1e-6)
        self.assertGreater(np.abs(got["ranges"]["LJ"] - got["rules"]["LJ"]).max(), 0.05)

    def test_the_turn_answers_with_those_ranges(self):
        res = self.adv.advise(state(FLOP + "6h", LINE))
        self.assertIsNone(res.get("fallback"))
        self.assertTrue(any("4-way, ranges narrowed through" in n for n in res["notes"]), res["notes"])
        self.assertFalse(any("simple betting rules rather than a solve" in n for n in res["notes"]), res["notes"])

    def test_a_bet_far_over_the_size_stays_with_the_rules(self):
        got = self.ranges(lead(12.5, *FOLD("UTG"), CALL("LJ"), *FOLD("HJ")))    # (76%)
        self.assertFalse(got["ok"])
        self.assertTrue(any("too far from the 4-way solve's bet 33%" in n for n in got["notes"]), got["notes"])
        self.assertEqual({p: 1 for p in got["ranges"] if np.abs(got["ranges"][p] - got["rules"][p]).max() > 0}, {})

    def test_a_raise_the_tree_lacks_stays_with_the_rules(self):
        got = self.ranges(lead(5.5, {"pos": "UTG", "action": "raise", "to": 16.5}, *FOLD("LJ", "HJ"), CALL("BB")))
        self.assertFalse(got["ok"])
        self.assertEqual({p: 1 for p in got["ranges"] if np.abs(got["ranges"][p] - got["rules"][p]).max() > 0}, {})

    def test_half_the_pot_reads_the_solve(self):
        self.assertTrue(self.ranges(lead(8.0, *FOLD("UTG"), CALL("LJ"), *FOLD("HJ")))["ok"])   # (48%)

    def test_a_street_checked_through_stays_with_the_rules(self):
        got = self.ranges(CHECK("BB", "UTG", "LJ", "HJ"))
        self.assertFalse(got["ok"])
        self.assertEqual({p: 1 for p in got["ranges"] if np.abs(got["ranges"][p] - got["rules"][p]).max() > 0}, {})

    def test_no_flop_advice_no_solve(self):
        adv = native.advisor("fast", library=False)
        adv.mw_max_iters, adv.mw_budget = 40, 1e6
        res = adv.advise(state(FLOP + "6h", LINE))
        self.assertTrue(any("flop: multiway, ranges narrowed by simple betting rules" in n for n in res["notes"]), res["notes"])
        self.assertEqual([k for k in adv._mw_cache], [])            # (the turn heads-up: no multiway solve at all)

    def test_an_unsettled_solve_stays_with_the_rules(self):
        adv = native.advisor("fast", library=False)
        adv.mw_max_iters, adv.mw_budget = 5, 1e6
        adv.mw_max_expl = 1e9
        adv.advise(state(FLOP, []))
        self.assertEqual(A.trust_of(next(iter(adv._mw_lines.values()))[2], "flop"), 0.0)
        res = adv.advise(state(FLOP + "6h", LINE))
        self.assertFalse(any("4-way, ranges narrowed through" in n for n in res["notes"]), res["notes"])

    def test_the_floor_still_fills_a_thin_line(self):
        # a range the solve all but empties: _post_floor's fill, as after the rules
        from gto import preflop as pf
        before = np.zeros(len(ALL_COMBOS))
        before[:60] = 1.0
        after = before * 1e-3
        filled = A._post_floor(before, after, cards(FLOP + "6h"), "LJ", "flop", [])
        live = np.prod([1.0 - A.MEMBER_FULL[c] for c in cards(FLOP + "6h")], axis=0)
        self.assertGreater((filled * live).sum(), (after * live).sum() * 100)
        self.assertGreater(pf.MIN_LINE_COMBOS, (after * live).sum())

    def test_nobody_folded_is_filled_out(self):
        notes = self.adv.advise(state(FLOP + "6h", LINE))["notes"]
        self.assertFalse([n for n in notes if "HJ's line left" in n or "UTG's line left" in n], notes)

    def test_streets_the_walk_takes(self):
        class St:
            def __init__(self, name, n):
                self.name, self.players_start = name, list(range(n))
        self.assertTrue(A._mw_walks(St("flop", 3)) and A._mw_walks(St("turn", 3)) and A._mw_walks(St("flop", 4)))
        self.assertFalse(A._mw_walks(St("turn", 4)) or A._mw_walks(St("river", 4)) or A._mw_walks(St("flop", 5)))


class TestOffSize(unittest.TestCase):
    class Node:
        names = ["check", "bet 33%", "all-in"]
        amounts = [None, 5.5, 90.0]

    def test_sizes(self):
        n = self.Node()
        self.assertFalse(A._mw_off_size(n, 0, {"action": "check"}))
        self.assertFalse(A._mw_off_size(n, 1, {"action": "bet", "amount": 6.5}))
        self.assertFalse(A._mw_off_size(n, 1, {"action": "bet", "amount": 9.5}))
        self.assertTrue(A._mw_off_size(n, 1, {"action": "bet", "to": 12.5}))
        self.assertFalse(A._mw_off_size(n, 1, {"action": "bet"}))
        self.assertTrue(A._mw_off_size(n, 1, {"action": "all-in"}))
        self.assertFalse(A._mw_off_size(n, 2, {"action": "all-in"}))


if __name__ == "__main__":
    unittest.main()
