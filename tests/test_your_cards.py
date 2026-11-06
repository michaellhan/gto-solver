"""Your cards aren't public: nobody's range or line depends on them. They come out only where your hand is counted
against the ranges, in the solve's row for it (card removal) and in what the answer reports."""
import unittest
from unittest import mock

import numpy as np

from gto import multiway_solver as MW
from gto import advisor as A
from gto.advisor import MEMBER_FULL
from gto.cards import cards
from gto.native import advisor

FOLDS = lambda *seats: [{"pos": s, "action": "fold"} for s in seats]
# the hand that found it: 7-handed, a 3bet pot three-way, where TT is a fifth of CO's line
PRE7 = FOLDS("UTG", "LJ", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5},
                                   {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"},
                                   {"pos": "CO", "action": "call"}]
FLOP7 = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 7.5}, {"pos": "BTN", "action": "call"},
         {"pos": "BB", "action": "call"}]
TURN7 = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 34.0}]
SEVEN = dict(players=7, hero="BTN", board="Ks7s2h2c", stack=100, preflop=PRE7, flop=FLOP7, turn=TURN7)
# heads-up: a 4bet pot's turn bet, BTN to act
PRE_4B = FOLDS("UTG", "LJ", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5}] \
    + FOLDS("SB", "BB") + [{"pos": "CO", "action": "raise", "to": 19}, {"pos": "BTN", "action": "call"}]
FOUR_BET = dict(players=6, hero="BTN", board="Ks7s2h2c", stack=100, preflop=PRE_4B,
                flop=[{"pos": "CO", "action": "bet", "amount": 10.0}, {"pos": "BTN", "action": "call"}],
                turn=[{"pos": "CO", "action": "bet", "amount": 25.0}])


def spy_multiway(state):
    """The ranges a fresh Fast advice hands the multiway solver, by seat order, and its answer."""
    seen = []
    real = MW.MultiwaySolver.__init__

    def init(self, board, ranges, *a, **k):
        seen.append([np.array(r, dtype=float) for r in ranges])
        return real(self, board, ranges, *a, **k)
    with mock.patch.object(MW.MultiwaySolver, "__init__", init):
        res = advisor("fast").advise(dict(state))
    return seen[-1], res


class TestLinesDontDependOnYourCards(unittest.TestCase):
    def test_multiway_solve_gets_the_same_ranges_whatever_you_hold(self):
        # with your cards taken out before the rules and the floor read the line, CO's turn range was 11.3 combos
        # holding TT where it's 9.7 less what TT blocks: more filler, less value, a call that isn't there
        tt, res = spy_multiway(dict(SEVEN, hole="ThTs"))
        ak, _ = spy_multiway(dict(SEVEN, hole="AhKd"))
        self.assertIn("multiway", res)
        for i in (0, 1):    # BB and CO; BTN's own is the same anyway
            np.testing.assert_array_equal(tt[i], ak[i])

    def test_the_floor_reads_the_whole_line(self):
        # the combos a line left, and so how much filler it gets, are the player's, not what's left of them after you
        notes = lambda h: [n for n in advisor("fast").advise(dict(SEVEN, hole=h))["notes"] if "filled out" in n]
        self.assertEqual(notes("ThTs"), notes("AhKd"))
        self.assertTrue(notes("ThTs"))

    def test_heads_up_solve_gets_the_same_ranges_whatever_you_hold(self):
        adv = advisor("fast")
        seen = {}
        real = adv._solve_cached

        def solve(board, r0, r1, *a, **k):
            seen.setdefault(len(board), []).append((np.array(r0, dtype=float), np.array(r1, dtype=float)))
            return real(board, r0, r1, *a, **k)
        adv._solve_cached = solve
        adv.advise(dict(FOUR_BET, hole="TdTc"))
        adv.advise(dict(FOUR_BET, hole="QdQc"))
        (a0, a1), (b0, b1) = seen[4][0], seen[4][-1]
        np.testing.assert_array_equal(a0, b0)      # CO's
        np.testing.assert_array_equal(a1, b1)      # and BTN's


    def test_one_size_rows_are_each_hands_own(self):
        # one solve serves every hand of a node: the view's rows, which hold a hand's line, can't
        river = dict(players=6, hero="BB", board="Ks7d2cTh3s", stack=100, turn=[{"pos": "BB", "action": "check"},
                                                                                 {"pos": "SB", "action": "check"}],
                     preflop=FOLDS("UTG", "LJ", "HJ", "CO", "BTN") + [{"pos": "SB", "action": "call"},
                                                                      {"pos": "BB", "action": "check"}],
                     flop=[{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}],
                     river=[{"pos": "SB", "action": "check"}])
        adv = advisor("fast")
        a = adv.advise(dict(river, hole="Kh9h", one_size="50%"))["one_size"]["rows"][0]["hand"]
        b = adv.advise(dict(river, hole="9c8c", one_size="50%"))["one_size"]["rows"][0]["hand"]
        self.assertNotEqual(a["ev"], b["ev"])


class TestYourCardsStillBlock(unittest.TestCase):
    def test_multiway_report_leaves_out_what_you_block(self):
        rs, res = spy_multiway(dict(SEVEN, hole="ThTs"))
        board = cards("Ks7s2h2c")
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        off = np.prod([1.0 - MEMBER_FULL[c] for c in cards("ThTs")], axis=0)
        self.assertAlmostEqual(res["opponent_range_combos"]["CO"], float((rs[1] * live * off).sum()), places=1)
        self.assertLess(res["opponent_range_combos"]["CO"], float((rs[1] * live).sum()) - 1.0)

    def test_heads_up_report_leaves_out_what_you_block(self):
        with mock.patch("gto.advisor.range_to_text", wraps=A.range_to_text) as shown:
            res = advisor("fast").advise(dict(FOUR_BET, hole="TdTc"))
        r = shown.call_args_list[-1].args[0]
        for c in cards("TdTc"):
            self.assertEqual(float(r[MEMBER_FULL[c] > 0].sum()), 0.0)
        self.assertAlmostEqual(res["villain_range_combos"], float(r.sum()), places=1)
        self.assertGreater(res["villain_range_combos"], 0.0)

    def test_a_range_you_block_whole(self):
        # nothing to count your hand against: the hand still answers
        pre = FOLDS("UTG", "LJ", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + FOLDS("SB") \
            + [{"pos": "BB", "action": "call"}]
        res = advisor("fast").advise(dict(players=6, hero="BTN", hole="ThTs", board="Ks7s2h", stack=100, preflop=pre,
                                          flop=[{"pos": "BB", "action": "check"}],
                                          ranges={"BB": "ThTc,ThTd,TsTc,TsTd,ThTs"}))
        self.assertNotIn("error", res)
        self.assertEqual(res["villain_range_combos"], 0.0)


class TestTheHand(unittest.TestCase):
    def test_tens_dont_read_a_game_where_co_cant_hold_tens(self):
        # With the opponents' ranges conditioned on TT, BTN's whole range played as if CO held none: its shove 21 combos
        # of 94 where it's 31 (AKo 2.7 of 6.4), so CO's JJ called 30% of the time, not 71%. With the lines read after
        # your cards, call +1.4 (100%) and all-in -0.8; in the game itself about 0 and -14.
        res = advisor("fast").advise(dict(SEVEN, hole="ThTs"))
        ev = {a["action"]: a["ev_bb"] for a in res["actions"]}
        self.assertLess(ev["call"], 0.5)
        self.assertLess(ev["all-in"], -8.0)
        self.assertEqual(res["recommended"], "fold")


if __name__ == "__main__":
    unittest.main()
