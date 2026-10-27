"""Study mode (Advisor.study, the page's solution browser): the hand as recorded, answered for whoever acts next
with their whole range: per hand class and combo the mix, EVs and equities, and what the hand needs next where
nobody acts."""
import unittest

import numpy as np

from gto.native import advisor

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
def combos(cls):
    """The class's combos: pairs 6, suited 4, offsuit 12."""
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    def study(self, **st):
        return self.adv.advise(dict({"players": 6, "stack": 100, "study": True}, **st))

    def check_totals(self, r):
        acts = r["actions"]
        self.assertAlmostEqual(sum(a["freq"] for a in acts), 1.0, delta=0.01)
        self.assertEqual(r["grid"]["actions"], [a["action"] for a in acts])
        # the range's mix per action is the grid's, weighted by each class's combos
        tot = sum(c["w"] * combos(k) for k, c in r["grid"]["cells"].items())
        self.assertAlmostEqual(tot, r["range_combos"], delta=0.01 * tot + 1)
        for i, a in enumerate(acts):
            share = sum(c["w"] * combos(k) * c["f"][i] for k, c in r["grid"]["cells"].items()) / tot
            self.assertAlmostEqual(share, a["freq"], delta=0.01)

    def test_preflop_whole_range(self):
        r = self.study()
        self.assertEqual((r["study"], r["street"], r["actor"]), ("node", "preflop", "UTG"))
        self.assertEqual(r["range_combos"], 1326.0)
        self.check_totals(r)
        self.assertEqual(r["grid"]["cells"]["AA"]["f"][r["grid"]["actions"].index("raise")], 1.0)
        r = self.study(preflop=PRE[:4])
        self.assertEqual(r["actor"], "SB")
        self.assertIn("call", r["grid"]["actions"])
        self.check_totals(r)

    def test_next_card_and_over(self):
        self.assertEqual(self.study(preflop=PRE), {"study": "deal", "street": "flop", "allin": False})
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "check"}]
        self.assertEqual(self.study(preflop=PRE, board="Kd7c2h", flop=flop)["street"], "turn")
        self.assertEqual(self.study(preflop=PRE[:5] + [{"pos": "BB", "action": "fold"}]),
                         {"study": "over", "street": "preflop", "winner": "BTN"})
        jam = PRE[:5] + [{"pos": "BB", "action": "all-in"}, {"pos": "BTN", "action": "call"}]
        self.assertEqual(self.study(preflop=jam), {"study": "deal", "street": "flop", "allin": True})
        self.assertEqual(self.study(preflop=jam, board="Kd7c2h4s9h")["study"], "over")
        self.assertIn("error", self.study(preflop=PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}],
                                         board="Kd7c2h"))

    def test_flop_from_the_library(self):
        r = self.study(preflop=PRE, board="Kd7c2h", flop=[{"pos": "BB", "action": "check"}])
        self.assertEqual((r["actor"], r["opponent"], r["position"]), ("BTN", "BB", "IP"))
        self.assertEqual(r["precomputed"]["spot"], "srp_btn_bb")
        self.check_totals(r)
        self.assertEqual(r["pot_bb"], 5.5)
        A = len(r["actions"])
        # each class's numbers are its combos', weighted by their reach
        for cls in ("AKo", "77", "QJs"):
            c, cs = r["grid"]["cells"][cls], r["combos"][cls]
            w = np.array([x[1] for x in cs])
            self.assertAlmostEqual(w.sum() / combos(cls), c["w"], delta=0.002)
            for a in range(A):
                self.assertAlmostEqual((w * [x[2][a] for x in cs]).sum() / w.sum(), c["f"][a], delta=0.002)
                self.assertAlmostEqual((w * [x[3][a] for x in cs]).sum() / w.sum(), c["ev"][a], delta=0.01)
            self.assertTrue(all(len(x[0]) == 4 and 0 <= x[4] <= 1 for x in cs))
        # no combo holds a board card; strong hands have the equity, and the two ranges' equities add up to one
        self.assertFalse(any("Kd" in x[0] or "7c" in x[0] for cs in r["combos"].values() for x in cs))
        self.assertGreater(r["grid"]["cells"]["77"]["eq"], 0.9)
        self.assertAlmostEqual(r["range_eq"] + r["opponent_eq"], 1.0, delta=0.01)
        self.assertAlmostEqual(r["opponent_combos"], sum(c["w"] * combos(k) for k, c in r["opponent_grid"]["cells"].items()),
                               delta=1.0)
        # at equilibrium the mix's EV is the best action's, within the solve's distance from it
        for k, c in r["grid"]["cells"].items():
            self.assertLess(max(c["ev"]) - c["evm"], 0.05 * r["pot_bb"] + 0.01, k)

    def test_hero_and_exploits_ignored(self):
        st = dict(preflop=PRE, board="Kd7c2h", flop=[{"pos": "BB", "action": "check"}])
        a = self.study(**st)
        b = self.study(hero="BB", hole="KhKs", opponents="pool", locks={"BB": {"fold_to_bet": 30}}, **st)
        self.assertEqual(a["grid"], b["grid"])
        self.assertEqual(b["actor"], "BTN")

    def test_breakdown(self):
        r = self.study(preflop=PRE, board="Kd7c2h", breakdown=True)
        self.assertEqual(r["actor"], "BB")
        self.assertEqual(r["breakdown"]["actions"], r["grid"]["actions"])
        self.assertNotIn("mine", r["breakdown"]["hero"])   # (no hand of ours)

    def test_turn_solved_live(self):
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 1.8},
                {"pos": "BB", "action": "call"}]
        r = self.study(preflop=PRE, board="Kd7c2h5s", flop=flop, turn=[{"pos": "BB", "action": "check"}], budget=1.0)
        self.assertEqual((r["street"], r["actor"]), ("turn", "BTN"))
        self.assertNotIn("precomputed", r)
        self.assertGreater(r["solver"]["iterations"], 0)
        self.check_totals(r)
        self.assertAlmostEqual(r["pot_bb"], 9.1, delta=0.01)

    def test_multiway_after_the_flop(self):
        pre = PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"},
                         {"pos": "BB", "action": "call"}]
        self.assertIn("heads-up", self.study(preflop=pre, board="Kd7c2h")["error"])


if __name__ == "__main__":
    unittest.main()
