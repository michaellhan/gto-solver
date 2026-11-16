"""Against a typical player (opponents "pool"), your range on later streets is the one your preflop advice plays there,
not the chart's: the pool-priced calls it adds (Advisor._priced's pool) are in it, so a hand you were told to call
enters the flop in your range instead of as a hand nobody expects."""
import unittest

import numpy as np

from gto.advisor import br_tie_mix, preflop_ranges
from gto.cards import ALL_COMBOS, hand_class
from gto.native import advisor
from gto.preflop import SEATS

F = lambda p: {"pos": p, "action": "fold"}
# the hand that found it: 8-handed, UTG opens to 6, HJ calls, CO calls 66 (pool advice: call 85%), KcTc8c checked to CO
OPEN = [{"pos": "UTG", "action": "raise", "to": 6}, F("UTG1"), F("LJ"), {"pos": "HJ", "action": "call"}]
PRE = OPEN + [{"pos": "CO", "action": "call"}, F("BTN"), F("SB"), F("BB")]
FLOP = [{"pos": "UTG", "action": "check"}, {"pos": "HJ", "action": "check"}]


class TestPoolHeroRange(unittest.TestCase):
    def setUp(self):
        self.adv = advisor("fast")   # (the site's: a 3-way flop by the 3-player solve)

    def test_your_range_holds_the_calls_your_advice_gives(self):
        pre = self.adv.advise({"players": 8, "hero": "CO", "hole": "6c6s", "board": "", "preflop": OPEN,
                               "opponents": "pool"})
        call = {a["action"]: a["freq"] for a in pre["actions"]}["call"]
        self.assertGreater(call, 0.5)
        r = preflop_ranges(PRE, SEATS[8], 100.0, "CO")["CO"]
        sixes = [i for i, c in enumerate(ALL_COMBOS) if hand_class(c) == "66"]
        self.assertAlmostEqual(float(np.mean(r[sixes])), call, places=2)
        # (equilibrium keeps the chart's range, where 66 folds)
        self.assertLess(float(np.mean(preflop_ranges(PRE, SEATS[8], 100.0)["CO"][sixes])), 0.05)

    def test_the_flop_answers_it_in_range(self):
        r = self.adv.advise({"players": 8, "hero": "CO", "hole": "6c6s", "board": "KcTc8c", "preflop": PRE,
                             "flop": FLOP, "opponents": "pool"})
        self.assertFalse(r.get("off_range"))
        self.assertGreater(r["hero_range_combos"], 30)
        # (Since multiway exploit answers became the best response to the pool (97e6ecf4), 66 bets here: the pool
        # raises and bluffs less than equilibrium, so betting outearns checking at every solve length, check about
        # 6.8 bb against 7.3-8.4 for the best bet. Which size is best is a near tie that flips with the iterations
        # (bet 33% 8.13-8.21 against 75% 7.79-8.34 from 57 to 529 iterations), so the test checks the EVs, not the mix.)
        ev = {a["action"]: a["ev_bb"] for a in r["actions"]}
        self.assertGreater(max(ev["bet 33%"], ev["bet 75%"]), ev["check"] + 0.3)

    def test_a_short_solve_shows_the_mix_its_evs_give(self):
        # Fast's answer at a 2 s budget (26 iterations natively, 5.6% of the pot from settled): its margin of 0.98 bb tied
        # check at 6.72 with bet 33% at 7.30 and bet 75% at 6.94, and a tied action kept its whole equilibrium weight,
        # so it showed check 98% (advisor.br_tie_mix). Whatever the iterations, the action shown most often is now within
        # half the margin of the best EV.
        adv = advisor("fast")
        adv.time_budget *= 0.4; adv.prior_budget *= 0.4
        adv.mw_budget = adv.mw_budget * 0.4 if adv.mw_budget else adv.mw_budget
        r = adv.advise({"players": 8, "hero": "CO", "hole": "6c6s", "board": "KcTc8c", "preflop": PRE, "flop": FLOP,
                        "opponents": "pool"})
        top = max(r["actions"], key=lambda a: a["freq"])
        best = max(a["ev_bb"] for a in r["actions"])
        self.assertIn("ev_tie", r)
        self.assertLessEqual(best - top["ev_bb"], r["ev_tie"] / 2 + 0.01, r["actions"])

    def test_tied_actions_keep_the_equilibrium_weight_the_noise_explains(self):
        evs = np.array([[6.72], [7.30], [6.94]])           # check, bet 33%, bet 75% (that answer's)
        eq = np.array([[0.98], [0.02], [0.0]])
        np.testing.assert_allclose(br_tie_mix(evs, eq, 0.98).ravel(), [0.40, 0.60, 0.0], atol=0.005)
        np.testing.assert_allclose(br_tie_mix(evs, eq, 0.1).ravel(), [0, 1, 0])   # (outside the margin: the best)
        # a gap far inside the margin keeps nearly the equilibrium's mix (AQo checking at 6.45 over bets at 6.40)
        np.testing.assert_allclose(br_tie_mix(np.array([[6.45], [6.40]]), np.array([[0.3], [0.7]]), 0.6).ravel(),
                                   [0.358, 0.642], atol=0.001)
        # an unreached hand (no equilibrium weight) plays the best; the mix sums to 1
        np.testing.assert_allclose(br_tie_mix(np.array([[1.0, 2.0], [3.0, 2.0]]), np.zeros((2, 2)), 0.5), [[0, 1], [1, 0]])
        m = br_tie_mix(np.random.default_rng(1).normal(size=(4, 50)), np.random.default_rng(2).random((4, 50)), 0.7)
        np.testing.assert_allclose(m.sum(axis=0), 1.0)


if __name__ == "__main__":
    unittest.main()
