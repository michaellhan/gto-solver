"""Study's size menus (study.js SIZE_PRESETS, the request's "sizes"): up to advisor.STUDY_MAX_SIZES a street in Study
(the advice keeps MAX_SIZES); the flop library answers where the flop's sizes are its own, the flop is solved live where
they aren't."""
import unittest

from gto import advisor as A
from gto import native
from gto.cli import parse_actions

LINE = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"


def state(board="Kc7d2h", flop=(), **kw):
    return {"players": 6, "stack": 100, "preflop": parse_actions(LINE), "board": board, "flop": list(flop), "turn": [],
            "river": [], "study": True, **kw}


class TestStudySizes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = native.advisor("fast", budget=0.5, prior_budget=0.5)

    def test_more_sizes(self):
        """"More sizes": five flop bets, solved live (the library stores 33/75%)."""
        r = self.adv.advise(state(sizes={"flop": [25, 33, 50, 75, 125]}))
        self.assertEqual(r.get("study"), "node", r)
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 25%", "bet 33%", "bet 50%", "bet 75%", "bet 125%"])
        self.assertFalse(r.get("precomputed"))
        # facing the 50% bet, from the same solve
        x = {"pos": "BB", "action": "check"}
        r2 = self.adv.advise(state(flop=[x], sizes={"flop": [25, 33, 50, 75, 125]}))
        bet = next(a for a in r2["actions"] if a["action"] == "bet 50%")
        r3 = self.adv.advise(state(flop=[x, {"pos": "BTN", "action": "bet", "amount": bet["amount_bb"]}],
                                   sizes={"flop": [25, 33, 50, 75, 125]}))
        self.assertEqual(r3["actor"], "BB")
        self.assertEqual(r3["to_call_bb"], bet["amount_bb"])

    def test_library_where_sizes_match(self):
        """The library's own flop sizes, or other turn and river sizes alone: the flop answered from the library."""
        for sizes in (None, {"flop": [33, 75]}, {"turn": [50, 100]}, {"river": [33, 75, 125, 200]}):
            r = self.adv.advise(state(**({"sizes": sizes} if sizes else {})))
            self.assertTrue(r.get("precomputed"), sizes)
            self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 33%", "bet 75%"])

    def test_answer_at_once(self):
        """With a size menu: no answer at once on a flop (the library's reads have its sizes); a turn after the flop
        solved live with the menu's sizes, off that solve (as Standard's after a live flop)."""
        more = {"flop": [25, 33, 50, 75, 125]}
        self.assertIsNone(self.adv.instant(state(sizes=more)))
        x = {"pos": "BB", "action": "check"}
        r = self.adv.advise(state(flop=[x], sizes=more))
        b = max(r["actions"][1:], key=lambda a: a["freq"])
        line = [x, {"pos": "BTN", "action": "bet", "amount": b["amount_bb"]}, {"pos": "BB", "action": "call"}]
        self.adv.advise(state(board="Kc7d2hTs", flop=line, sizes=more))   # (the flop solved to narrow, cached)
        got = self.adv.instant(state(board="Kc7d2h9d", flop=line, sizes=more))
        self.assertEqual(got["approximate"]["source"], "prev")
        self.assertEqual(got["study"], "node")
        # the advice's custom sizes still get none
        adv = dict(state(board="Kc7d2h9d", flop=line, sizes=more), hero="BB", hole="QhJh")
        adv.pop("study")
        self.assertIsNone(self.adv.instant(adv))

    def test_limits(self):
        r = self.adv.advise(state(sizes={"flop": [20, 30, 40, 50, 60, 70]}))
        self.assertIn("at most 5 flop sizes", r.get("error", ""))
        # the advice keeps three a street
        adv = dict(state(sizes={"flop": [25, 50, 75, 125]}), hero="BB", hole="QhJh")
        adv.pop("study")
        self.assertIn("at most 3 flop sizes", self.adv.advise(adv).get("error", ""))
        self.assertEqual(A._custom_sizes({"flop": [25, 33, 50, 75, 125]}, A.STUDY_MAX_SIZES)["flop"],
                         [0.25, 0.33, 0.5, 0.75, 1.25])


if __name__ == "__main__":
    unittest.main()
