"""A hand off its range: the answer says so ("off_range"), and "wider" answers the same spot with your range holding hands
like yours, which is one of the assumptions the page checks the answer under (index.html's checkRanges)."""
import unittest

import numpy as np

from gto import advisor as A
from gto import preflop as pf
from gto.cards import COMBO_INDEX, cards
from gto.native import advisor
from gto.ranges import parse_range

FOLDS = lambda *seats: [{"pos": s, "action": "fold"} for s in seats]
CALL = lambda s: {"pos": s, "action": "call"}
# the hand that found it: 7-handed, UTG limps, LJ raises to 4, HJ and BB and UTG call; BB bets the flop, LJ calls
PRE = [CALL("UTG"), {"pos": "LJ", "action": "raise", "to": 4}, CALL("HJ")] + FOLDS("CO", "BTN", "SB") + [CALL("BB"), CALL("UTG")]
HAND = dict(players=7, hero="BB", hole="As4s", board="2s4d6c6h", stack=100, preflop=PRE,
            flop=[{"pos": "BB", "action": "bet", "amount": 6.5}] + FOLDS("UTG") + [CALL("LJ")] + FOLDS("HJ"))
SEATS7 = list(pf.SEATS[7])
OPEN = FOLDS("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + FOLDS("SB") + [CALL("BB")]
SEATS6 = list(pf.SEATS[6])


def combo(hole):
    return COMBO_INDEX[tuple(sorted(cards(hole)))]


class TestWiderPreflop(unittest.TestCase):
    def setUp(self):
        self.chart = A.preflop_ranges(PRE, SEATS7, 100.0)["BB"]
        self.wide = A._wider_preflop(PRE, SEATS7, 100.0, "BB", tuple(cards("As4s")))

    def test_it_holds_your_hand_and_the_whole_chart(self):
        self.assertEqual(self.chart[combo("As4s")], 0.0)
        self.assertEqual(self.wide[combo("As4s")], 1.0)
        self.assertTrue((self.wide >= self.chart - 1e-12).all())

    def test_it_holds_hands_like_yours_and_stays_within_its_limit(self):
        self.assertGreater(self.wide.sum(), self.chart.sum() + 10)
        self.assertLessEqual(self.wide.sum(), A.WIDER_MAX * self.chart.sum() + 1.01)
        self.assertGreater(self.wide[combo("As5s")], 0.0)      # (the suited aces that do as well)

    def test_a_call_leaves_out_what_the_chart_raises(self):
        self.assertEqual(self.wide[combo("AsAh")], self.chart[combo("AsAh")])

    def test_a_hand_in_the_range_needs_nothing(self):
        r = A.preflop_ranges(OPEN, SEATS6, 100.0)["BB"]
        hole = next(h for h in ("QsJs", "Td9d", "8h7h") if r[combo(h)] > 0.5)
        self.assertIsNone(A._wider_preflop(OPEN, SEATS6, 100.0, "BB", tuple(cards(hole))))


class TestWiderLine(unittest.TestCase):
    def test_a_street_that_left_your_hand_out_keeps_a_share(self):
        hole, before = tuple(cards("As4s")), np.ones(A.N_COMBOS)
        after = np.zeros(A.N_COMBOS)
        after[combo("KsKh")] = 1.0
        out = A._wider_line(before, after, hole)
        self.assertAlmostEqual(out[combo("As4s")], A.WIDER_KEEP)
        self.assertEqual(out[combo("KsKh")], 1.0)

    def test_a_street_that_kept_it_is_left_alone(self):
        before, after = np.ones(A.N_COMBOS), np.full(A.N_COMBOS, 0.5)
        self.assertIs(A._wider_line(before, after, tuple(cards("As4s"))), after)

    def test_a_hand_out_of_the_range_already_has_no_line_to_keep(self):
        before = np.ones(A.N_COMBOS)
        before[combo("As4s")] = 0.0
        after = np.zeros(A.N_COMBOS)
        self.assertIs(A._wider_line(before, after, tuple(cards("As4s"))), after)


class TestOffRange(unittest.TestCase):
    def test_the_marker(self):
        hole = tuple(cards("As4s"))
        r = np.ones(A.N_COMBOS)
        self.assertEqual(A._off_range(r, hole, r), {})
        self.assertEqual(A._off_range(r, (), r), {})      # (Study: no hand of your own)
        r[combo("As4s")] = 0.001
        self.assertEqual(A._off_range(r, hole, r)["off_range"], {"weight": 0.001, "at_street_start": 0.001})
        self.assertIs(A._off_range(r, hole, np.ones(A.N_COMBOS))["off_range"]["wider"], False)   # (left it on this street)


class TestTheHand(unittest.TestCase):
    """Three ranges, three answers: nothing here is hand-written for the spot."""
    @classmethod
    def setUpClass(cls):
        adv = advisor("fast")
        cls.chart = adv.advise(dict(HAND))
        cls.pool = adv.advise(dict(HAND, opponents="pool"))
        cls.wider = adv.advise(dict(HAND, wider=True, background=True))

    def test_off_its_range_where_the_ranges_leave_it_out(self):
        self.assertIn("off_range", self.chart)
        self.assertIn("off_range", self.pool)
        self.assertNotIn("wider", self.chart["off_range"])

    def test_a_wider_range_holds_it(self):
        self.assertNotIn("off_range", self.wider)
        self.assertGreater(self.wider["hero_range_combos"], self.chart["hero_range_combos"])

    def test_the_answer_turns_on_the_range(self):
        self.assertEqual(self.chart["recommended"], "bet 75%")
        self.assertEqual(self.pool["recommended"], "bet 33%")
        self.assertEqual(self.wider["recommended"], "check")

    def test_the_flags_dont_outlive_their_request(self):
        adv = advisor("fast")
        self.assertFalse(adv._wider or adv._background)


class TestInRange(unittest.TestCase):
    def test_a_hand_in_its_range_isnt_marked(self):
        r = A.preflop_ranges(OPEN, SEATS6, 100.0)["BB"]
        hole = next(h for h in ("QsJs", "Td9d", "8h7h") if r[combo(h)] > 0.5)
        res = advisor("fast").advise(dict(players=6, hero="BB", hole=hole, board="Ks7d2c", stack=100, preflop=OPEN))
        self.assertIn("actions", res)
        self.assertNotIn("off_range", res)

    def test_a_range_of_your_own_stands(self):
        res = advisor("fast").advise(dict(HAND, ranges={"BB": "22+,A2s+"}, wider=True))
        self.assertIn("actions", res)
        self.assertTrue(any("custom one" in n for n in res["notes"]))


class TestLaterStreets(unittest.TestCase):
    def test_a_line_the_solves_left_your_hand_out_of_keeps_a_share_of_it(self):
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}] + FOLDS("HJ", "CO", "BTN", "SB") + [CALL("BB")]
        st = dict(players=6, hero="BB", hole="Ad7s", board="8s2dQc8dTs", stack=100, preflop=pre, budget=0.8, prior_budget=0.8,
                  flop=[{"action": "bet", "amount": 1.8, "pos": "BB"}, {"action": "call", "pos": "UTG"}],
                  turn=[{"action": "check", "pos": "BB"}, {"action": "bet", "amount": 3.0, "pos": "UTG"}, {"action": "call", "pos": "BB"}],
                  river=[{"action": "check", "pos": "BB"}, {"action": "bet", "amount": 11.3, "pos": "UTG"}])
        adv = advisor("fast")
        self.assertGreater(A.preflop_ranges(pre, SEATS6, 100.0)["BB"][combo("Ad7s")], 0.1)      # (in range preflop)
        main = adv.advise(dict(st))
        self.assertIn("off_range", main)
        self.assertNotIn("wider", main["off_range"])
        self.assertNotIn("off_range", adv.advise(dict(st, wider=True, background=True)))


class TestCache(unittest.TestCase):
    """A follow-up's solves are the first to go, and what it reads of the hand's stays where it was."""
    def setUp(self):
        self.adv = A.Advisor()
        self.adv.cache_mb = 1e9

    def keep(self, key, bg=False):
        self.adv._background = bg
        self.adv._cache[key] = np.zeros(8)
        self.adv._keep("_cache", key, self.adv._cache[key])
        self.adv._background = False

    def order(self):
        return [k[2] for k in self.adv._held]

    def test_a_follow_ups_solve_goes_first(self):
        self.keep("flop")
        self.keep("turn")
        self.keep("theirs", bg=True)
        self.assertEqual(self.order(), ["theirs", "flop", "turn"])

    def test_reading_the_hands_solve_doesnt_move_it(self):
        self.keep("flop")
        self.keep("turn")
        self.adv._background = True
        self.adv._keep("_cache", "flop", None)
        self.adv._background = False
        self.assertEqual(self.order(), ["flop", "turn"])

    def test_the_shared_dict_stays_shared(self):
        held = self.adv._held
        self.keep("flop")
        self.keep("theirs", bg=True)
        self.assertIs(self.adv._held, held)


if __name__ == "__main__":
    unittest.main()
