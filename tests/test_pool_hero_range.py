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
        freq = {a["action"]: a["freq"] for a in r["actions"]}
        self.assertLess(freq["bet 75%"], 0.5)   # (off its range it bet 75% half the time, into two players)


if __name__ == "__main__":
    unittest.main()
