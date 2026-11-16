"""Against a typical player (opponents "pool"), your range on later streets is the one your preflop advice plays there,
not the chart's: the pool-priced calls it adds (Advisor._priced's pool) are in it, so a hand you were told to call
enters the flop in your range instead of as a hand nobody expects."""
import unittest

import numpy as np

from gto.advisor import preflop_ranges
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


if __name__ == "__main__":
    unittest.main()
