"""Your hand outside your preflop range: the chart still answers it at the spot, and the grid draws it (grid["hero"]),
where its cell's weight alone would leave it blank, like a hand the chart folds."""
import unittest

from gto.advisor import Advisor, Config

F = lambda p: {"pos": p, "action": "fold"}
# the hand that found it: 8-handed, LJ and CO limp, BTN raises J8s (the chart doesn't), SB 3bets to 15
PRE = [F("UTG"), F("UTG1"), {"pos": "LJ", "action": "call"}, F("HJ"), {"pos": "CO", "action": "call"},
       {"pos": "BTN", "action": "raise", "to": 5}, {"pos": "SB", "action": "raise", "to": 15}, F("BB"), F("LJ"), F("CO")]


class TestOffChartHand(unittest.TestCase):
    def setUp(self):
        self.adv = Advisor(Config(), time_budget=1.0, prior_budget=0.5)

    def ask(self, hole, **kw):
        return self.adv.advise({"players": 8, "hero": "BTN", "hole": hole, "board": "", "preflop": PRE, **kw})

    def test_hand_outside_the_range_is_drawn_from_its_answer(self):
        for kw in ({}, {"opponents": "pool"}):
            r = self.ask("Js8s", **kw)
            g = r["grid"]
            self.assertEqual(g["cells"]["J8s"]["w"], 0.0)   # (not in the range BTN arrives with)
            freq = {a["action"]: a["freq"] for a in r["actions"]}
            self.assertEqual(g["hero"], [freq[n] for n in g["actions"]])
            self.assertGreater(freq["call"], 0.0)
            self.assertTrue(any(n.startswith("J8s isn't in your range for this line") for n in r["notes"]))

    def test_hand_in_the_range_has_no_extra_cell(self):
        r = self.ask("JsTs")
        self.assertGreater(r["grid"]["cells"]["JTs"]["w"], 0.0)
        self.assertNotIn("hero", r["grid"])
        self.assertFalse(any("isn't in your range" in n for n in r["notes"]))


if __name__ == "__main__":
    unittest.main()
