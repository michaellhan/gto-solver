"""Answers whose ranges left the solves earlier in the hand say so (Advisor's "off_solution"), on the hand that showed
it: HJ KhQh against the BTN's flat on Ah2h7d 4d 8s. HJ bets 75% on the flop, which the solve takes with ~0% of HJ's
range, so HJ's range from there is the betting rules'; then 75% on the turn and a third of the pot on the river, and
the BTN jams. The fold is right, and the pot, the price, card removal and the river solve's convergence are checked
with it."""
import unittest

import numpy as np

from gto import native
from gto.advisor import _post_floor
from gto.cli import parse_actions

PRE = parse_actions("UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f")
BET = lambda p, a: {"pos": p, "action": "bet", "amount": a}
CALL = lambda p: {"pos": p, "action": "call"}
FLOP_75 = [BET("HJ", 5.0), CALL("BTN")]   # (77% of the 6.5 bb pot: the tree's 75%)


class TestOffSolutionHand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = native.advisor("fast", budget=1.5, prior_budget=1.5)

    def state(self, board, flop, turn=(), river=()):
        return {"players": 6, "stack": 100, "preflop": PRE, "board": board, "hero": "HJ", "hole": "KhQh",
                "flop": list(flop), "turn": list(turn), "river": list(river)}

    def test_rare_flop_bet_flagged(self):
        """The turn after HJ's 75% flop bet: the line is listed with the share the solve takes it with."""
        res = self.adv.advise(self.state("Ah2h7d4d", FLOP_75))
        rare = [o for o in res.get("off_solution", []) if o["why"] == "rare"]
        self.assertEqual([(o["street"], o["pos"], o["action"]) for o in rare], [("flop", "HJ", "bet 75%")])
        self.assertLess(rare[0]["share"], 10.0)
        self.assertTrue(any(n.startswith("flop: the solve rarely has HJ bet here") for n in res["notes"]))

    def test_solved_line_not_flagged(self):
        """The same turn after the third-pot bet the solve takes with nearly all of HJ's range: no rare line."""
        res = self.adv.advise(self.state("Ah2h7d4d", [BET("HJ", 2.0), CALL("BTN")]))
        self.assertEqual([o for o in res.get("off_solution", []) if o["why"] == "rare"], [])

    def test_river_jam_fold(self):
        """Facing the jam: a fold, from a settled river solve; pot 6.5 -> 16.5 -> 41.5, HJ's 13.5 and the BTN's 80
        all-in, 66.5 to call; the flop's rare line still listed; no BTN combo holds HJ's cards or the board's."""
        river = [BET("HJ", 13.5), {"pos": "BTN", "action": "all-in"}]
        res = self.adv.advise(self.state("Ah2h7d4d8s", FLOP_75, [BET("HJ", 12.5), CALL("BTN")], river))
        acts = {a["action"]: a for a in res["actions"]}
        self.assertEqual((res["recommended"], set(acts)), ("fold", {"fold", "call"}))
        self.assertGreater(acts["fold"]["freq"], 0.99)
        self.assertLess(acts["call"]["ev_bb"], 0.0)
        self.assertAlmostEqual(res["pot_bb"], 135.0)
        self.assertAlmostEqual(res["to_call_bb"], 66.5)
        self.assertAlmostEqual(acts["call"]["amount_bb"], 66.5)
        self.assertIn(("flop", "HJ", "bet 75%"),
                      {(o["street"], o.get("pos"), o.get("action")) for o in res["off_solution"] if o["why"] == "rare"})
        self.assertGreater(res["villain_range_combos"], 0.0)
        self.assertLess(res["solver"]["exploitability_pct"], 0.5)
        cells = res["villain_grid"]["cells"]
        # (a class's weight is its combos' mean: Kh and Qh are HJ's, so at most 3 of KQs's 4; one ace, four, seven,
        # eight or deuce on the board leaves at most 3 of its pair's 6)
        self.assertLessEqual(cells["KQs"]["w"], 0.75 + 1e-6)
        for pair in ("AA", "88", "77", "44", "22"):
            self.assertLessEqual(cells[pair]["w"], 0.5 + 1e-6, pair)

    def test_filled_line_recorded(self):
        """A line leaving under preflop.MIN_LINE_COMBOS is filled out, and listed with the combos it left."""
        before = np.zeros(1326)
        before[:60] = 1.0
        after = before.copy()
        after[6:] = 0.0
        off, notes = [], []
        r = _post_floor(before, after, [], "BTN", "flop", notes, off)
        self.assertGreater(r.sum(), 6.0)
        self.assertEqual(off, [{"street": "flop", "pos": "BTN", "why": "filled", "combos": 6.0}])
        self.assertEqual(_post_floor(before, before, [], "BTN", "flop", [], off).sum(), 60.0)   # (60 left: kept)
        self.assertEqual(len(off), 1)


if __name__ == "__main__":
    unittest.main()
