"""A hand outside your range has no equilibrium mix: a best response plays it one way at 100% however little that way is
ahead, so its answer goes by the EVs and says when the top ones are too near to call (advisor._read_off_range). On the
hand that found it: 7-handed, A4s in the big blind, a call the charts fold, then 2s4d6c 6h after a bet and a call: bet 75%
at 100% for 6.89 bb, 6.51 for a third of the pot, 6.24 for a check, in a 29.5 bb pot."""
import copy
import unittest

from gto import advisor as A
from gto.native import advisor

OFF = f"{A.OFF_RANGE}, so it was solved as a near-zero-weight combo"
FOLD = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
PRE = [{"pos": "UTG", "action": "call"}, {"pos": "LJ", "action": "raise", "to": 4}, {"pos": "HJ", "action": "call"}] \
    + FOLD("CO", "BTN", "SB") + [{"pos": "BB", "action": "call"}, {"pos": "UTG", "action": "call"}]
FLOP = [{"pos": "BB", "action": "bet", "amount": 5.5}, *FOLD("UTG"), {"pos": "LJ", "action": "call"}, *FOLD("HJ")]
HAND = {"players": 7, "hero": "BB", "hole": "As4s", "board": "2s4d6c6h", "stack": 100, "preflop": PRE, "flop": FLOP,
        "turn": [], "river": []}


def answer(pot=29.5, recommended="bet 75%", notes=(OFF,), expl=0.4, evs=(("check", 0.0, 6.24), ("bet 33%", 0.0, 6.51),
                                                                        ("bet 75%", 1.0, 6.89))):
    return {"street": "turn", "pot_bb": pot, "recommended": recommended, "sampled": recommended, "notes": list(notes),
            "solver": {"exploitability_pct": expl},
            "actions": [{"action": n, "freq": f, "ev_bb": e} for n, f, e in evs]}


class TestReadOffRange(unittest.TestCase):
    def read(self, **kw):
        res = answer(**kw)
        A._read_off_range(res)
        return res

    def test_the_hand_that_found_it(self):
        res = self.read()
        self.assertTrue(res["off_range"])
        self.assertEqual(res["close"], {"actions": ["bet 75%", "bet 33%", "check"], "within_pct": 3.0, "gap_pct": 2.2})

    def test_against_the_pool_only_the_near_ones(self):
        """9.75, 9.37 and 8.41: the check is 4.5% of the pot behind."""
        res = self.read(recommended="bet 33%", evs=(("check", 0.0, 8.41), ("bet 33%", 1.0, 9.75), ("bet 75%", 0.0, 9.37)))
        self.assertEqual(res["close"]["actions"], ["bet 33%", "bet 75%"])
        self.assertEqual(res["close"]["gap_pct"], 1.3)
        self.assertEqual(res["recommended"], "bet 33%")

    def test_a_clear_gap_is_not_close(self):
        res = self.read(evs=(("check", 0.0, 4.0), ("bet 75%", 1.0, 6.89)))
        self.assertTrue(res["off_range"])
        self.assertNotIn("close", res)

    def test_a_hand_in_its_range_is_left_alone(self):
        res = self.read(notes=("something else",))
        self.assertEqual(res, answer(notes=("something else",)))

    def test_the_solvers_numbers_stay(self):
        before = answer()
        res = copy.deepcopy(before)
        A._read_off_range(res)
        self.assertEqual(res["actions"], before["actions"])

    def test_a_mix_that_lags_its_evs_doesnt_fool_it(self):
        """check 100% at 1.92, bet 33% at 3.83: the top EV is the answer, nothing is close, and the mix stays shown."""
        res = self.read(pot=8.0, recommended="check", evs=(("check", 1.0, 1.92), ("bet 33%", 0.0, 3.83)))
        self.assertEqual((res["recommended"], res["sampled"]), ("bet 33%", "bet 33%"))
        self.assertNotIn("close", res)
        self.assertEqual(res["actions"][0]["freq"], 1.0)

    def test_a_mix_within_the_bar_is_close_with_the_top_ev_first(self):
        res = self.read(pot=8.0, recommended="check", evs=(("check", 1.0, 3.70), ("bet 33%", 0.0, 3.83)))
        self.assertEqual(res["recommended"], "bet 33%")
        self.assertEqual(res["close"]["actions"], ["bet 33%", "check"])

    def test_a_rough_solve_widens_the_bar_up_to_the_cap(self):
        evs = (("check", 0.0, 96.0), ("bet", 1.0, 100.0))    # (4% of a 100 bb pot)
        self.assertNotIn("close", self.read(pot=100.0, evs=evs, recommended="bet"))
        self.assertEqual(self.read(pot=100.0, evs=evs, recommended="bet", expl=4.2)["close"]["within_pct"], 4.2)
        far = (("check", 0.0, 94.0), ("bet", 1.0, 100.0))
        self.assertNotIn("close", self.read(pot=100.0, evs=far, recommended="bet", expl=20.0))
        self.assertEqual(self.read(pot=100.0, evs=evs, recommended="bet", expl=20.0)["close"]["within_pct"], A.CLOSE_MAX)

    def test_no_evs_no_call(self):
        """A quick 3-way library flop gives a hand outside its range the mix alone."""
        res = answer()
        for a in res["actions"]:
            del a["ev_bb"]
        A._read_off_range(res)
        self.assertTrue(res["off_range"])
        self.assertNotIn("close", res)
        self.assertEqual(res["recommended"], "bet 75%")

    def test_answers_that_arent_a_hands_pick(self):
        for res in ({"error": "x", "notes": [OFF]}, {"skipped": "x"}, {"study": "node", "actions": [], "notes": [OFF]}):
            before = copy.deepcopy(res)
            A._read_off_range(res)
            self.assertEqual(res, before)


class TestTheAnswers(unittest.TestCase):
    """advise's own answers: every surface reads these, so the page has nothing to work out."""

    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast", budget=1.5, prior_budget=1.0)
        cls.adv.mw_budget = 2.0

    def check(self, res):
        self.assertTrue(res.get("off_range"), res.get("notes"))
        top = max(res["actions"], key=lambda a: a["ev_bb"])
        self.assertEqual(next(a for a in res["actions"] if a["action"] == res["recommended"])["ev_bb"], top["ev_bb"])
        bar = res["close"]["within_pct"] if "close" in res else A.CLOSE_PCT
        near = {a["action"] for a in res["actions"] if (top["ev_bb"] - a["ev_bb"]) / res["pot_bb"] * 100 <= bar}
        self.assertEqual(set(res["close"]["actions"]) if "close" in res else {top["action"]}, near)

    def test_heads_up_turn(self):
        self.check(self.adv.advise(HAND))

    def test_heads_up_against_the_pool(self):
        self.check(self.adv.advise(dict(HAND, opponents="pool")))

    def test_multiway_flop(self):
        res = self.adv.advise(dict(HAND, board="2s4d6c", flop=[]))
        self.assertTrue(res.get("solved"), res.get("notes"))
        self.check(res)

    def test_a_hand_in_its_range_isnt_marked(self):
        res = self.adv.advise(dict(HAND, hole="AhAd", board="2s4d6c", flop=[], preflop=[
            {"pos": "UTG", "action": "fold"}, {"pos": "LJ", "action": "fold"}, {"pos": "HJ", "action": "fold"},
            {"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"},
            {"pos": "BB", "action": "raise", "to": 11}, {"pos": "BTN", "action": "call"}]))
        self.assertNotIn("off_range", res, res.get("notes"))
        self.assertNotIn("close", res)


if __name__ == "__main__":
    unittest.main()
