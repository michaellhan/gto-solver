"""Your equity once the betting is closed (state "equity": true, Advisor._equity_answer): checked through, called or
everyone all-in, against the ranges narrowed through the whole line, over the cards to come (breakdown.hand_equity)."""
import itertools
import unittest

import numpy as np

from gto.breakdown import hand_equity
from gto.cards import cards, COMBO_INDEX
from gto.evaluator import eval_board
from gto.native import advisor
from gto.pf_classes import equity_vs
from gto.ranges import parse_range, N_COMBOS

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
CHECKS = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "check"}]


def one(c):
    w = np.zeros(N_COMBOS); a, b = cards(c); w[COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
    return w


class TestHandEquity(unittest.TestCase):
    def test_preflop_heads_up_is_the_class_table(self):
        e = hand_equity(cards("AsAh"), [], [parse_range("KK")])
        self.assertAlmostEqual(e["equity"], 0.82, delta=0.01)
        self.assertTrue(e["exact"])
        w = parse_range("22+,ATs+,KQs")
        self.assertAlmostEqual(hand_equity(cards("Jh9h"), [], [w])["equity"],
                               float(equity_vs(w)[COMBO_INDEX[tuple(sorted(cards("Jh9h")))]]), places=9)

    def test_river_and_turn_against_counted_hands(self):
        # on the river: a share of the range, ties half
        e = hand_equity(cards("AhKh"), cards("Qs7h2cKd3s"), [parse_range("KK,KQ,AK")])
        self.assertEqual(e["runouts"], 1)
        self.assertAlmostEqual(e["equity"], e["win"] + e["tie"] / 2, places=9)
        # two villains holding one hand each (no shared cards, so independence is exact): every river counted
        hole, board, v = cards("JhTh"), cards("9h8c2d4s"), ["AsAd", "QcJc"]
        e = hand_equity(hole, board, [one(x) for x in v])
        share = []
        for r in set(range(52)) - set(hole + board + cards("AsAdQcJc")):
            s = eval_board(board + [r])
            me, them = s[COMBO_INDEX[tuple(sorted(hole))]], [s[COMBO_INDEX[tuple(sorted(cards(x)))]] for x in v]
            share.append(0.0 if max(them) > me else 1.0 / (1 + sum(t == me for t in them)))
        self.assertAlmostEqual(e["equity"], np.mean(share), places=9)
        self.assertEqual(e["runouts"], 46)

    def test_several_ranges_preflop_are_sampled(self):
        e = hand_equity(cards("9h9d"), [], [parse_range("22+,A2s+,KTs+"), parse_range("TT+,AK")], n_boards=1000)
        self.assertFalse(e["exact"])
        self.assertEqual(e["runouts"], 1000)
        self.assertLess(e["equity"], min(e["vs"]))   # beating both is harder than beating either


class TestEquityAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    def ask(self, **st):
        return self.adv.advise(dict({"players": 6, "stack": 100, "hero": "BTN", "hole": "AhKd", "equity": True}, **st))

    def test_preflop_closed(self):
        r = self.ask(preflop=PRE)
        self.assertEqual((r["street"], r["next"], r["allin"]), ("preflop", "flop", False))
        self.assertEqual([v["seat"] for v in r["villains"]], ["BB"])
        self.assertAlmostEqual(r["pot_bb"], 5.5)
        self.assertAlmostEqual(r["share_bb"], r["equity"] * 5.5, delta=0.01)
        # an all-in called: nobody can bet any more, and the jam's range is narrower than the flat's
        j = self.ask(hole="QsQd", preflop=PRE[:5] + [{"pos": "BB", "action": "all-in"}, {"pos": "BTN", "action": "call"}])
        self.assertTrue(j["allin"])
        self.assertLess(j["villains"][0]["combos"], r["villains"][0]["combos"])

    def test_flop_checked_through_narrows_through_it(self):
        r = self.ask(board="Ks7d2c", preflop=PRE, flop=CHECKS)
        self.assertEqual((r["street"], r["next"], r["runouts"]), ("flop", "turn", 1081))
        self.assertTrue(r["exact"])
        pre = self.ask(preflop=PRE)["villains"][0]["combos"]
        self.assertLess(r["villains"][0]["combos"], pre)   # (the board's cards out, and BB's check)
        self.assertGreater(r["equity"], 0.8)
        self.assertAlmostEqual(r["equity"], r["villains"][0]["equity"], places=4)

    def test_all_in_run_out_to_showdown(self):
        r = self.ask(board="Ks7d2c5h9s", preflop=PRE,
                     flop=[{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "all-in"}, {"pos": "BB", "action": "call"}])
        self.assertEqual((r["street"], r["next"], r["allin"], r["runouts"]), ("river", None, True, 1))
        self.assertAlmostEqual(r["pot_bb"], 200.5)

    def test_multiway(self):
        pre3 = PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
        r = self.ask(board="Ts6d2c", preflop=pre3, flop=[{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"},
                                                          {"pos": "BTN", "action": "check"}])
        self.assertEqual([v["seat"] for v in r["villains"]], ["SB", "BB"])
        self.assertNotIn("share_bb", r)
        self.assertLess(r["equity"], min(v["equity"] for v in r["villains"]))

    def test_only_once_the_betting_is_closed(self):
        self.assertIn("isn't over", self.ask(board="Ks7d2c", preflop=PRE, flop=CHECKS[:1])["error"])
        self.assertIn("isn't over", self.ask(preflop=PRE[:4])["error"])
        self.assertIn("folded", self.ask(preflop=PRE[:3] + [{"pos": "BTN", "action": "fold"}])["error"])
        # without "equity" a closed street is still no decision of yours
        r = self.adv.advise({"players": 6, "stack": 100, "hero": "BTN", "hole": "AhKd", "board": "Ks7d2c", "preflop": PRE, "flop": CHECKS})
        self.assertIn("closed", r["error"])


if __name__ == "__main__":
    unittest.main()
