"""A call's price counts what the players it is against still owe: a limper behind a raise is in the equity, so its chips
are in the pot (advisor._owed, Advisor._priced)."""
import unittest

from gto import advisor as A
from gto import preflop as pf
from gto.advisor import _owed, replay_hand
from gto.native import advisor

FOLDS = lambda *seats: [{"pos": s, "action": "fold"} for s in seats]
CALL = lambda p: {"pos": p, "action": "call"}
RAISE = lambda p, to: {"pos": p, "action": "raise", "to": to}
SEATS = pf.SEATS[7]
# the hand that found it: the big blind closes 4bb with a limper still to answer
LIMP = [CALL("UTG"), RAISE("LJ", 4), CALL("HJ")] + FOLDS("CO", "BTN", "SB")
NO_LIMP = FOLDS("UTG") + [RAISE("LJ", 4), CALL("HJ")] + FOLDS("CO", "BTN", "SB")


def bb(line, hole="As4s"):
    return advisor("fast").advise(dict(players=7, hero="BB", hole=hole, stack=100, board="", preflop=line))


def continues(res):
    """The big blind's share of hands that call or raise."""
    n = lambda k: 4 if k.endswith("s") else 12 if k.endswith("o") else 6
    cells = res["grid"]["cells"]
    return sum(n(k) * (v["f"][0] + v["f"][1]) for k, v in cells.items()) / sum(n(k) for k in cells)


def price(res):
    return next(n for n in res["notes"] if n.startswith("price"))


class TestOwed(unittest.TestCase):
    def test_a_limper_owes_the_raise(self):
        st = replay_hand(SEATS, 100.0, {"preflop": LIMP}, 1)[0]
        self.assertEqual(st.to_act, "BB")
        self.assertAlmostEqual(_owed(st, ["UTG", "LJ", "HJ"], 4.0, SEATS, 100.0), 3.0)

    def test_nobody_owes_after_a_call(self):
        st = replay_hand(SEATS, 100.0, {"preflop": NO_LIMP}, 1)[0]
        self.assertEqual(_owed(st, ["LJ", "HJ"], 4.0, SEATS, 100.0), 0.0)

    def test_a_short_limper_pays_what_it_has(self):
        t = pf.Table(SEATS, stacks={"UTG": 2.5})
        st = replay_hand(t, 100.0, {"preflop": LIMP}, 1)[0]
        self.assertAlmostEqual(_owed(st, ["UTG", "LJ", "HJ"], 4.0, t, 100.0), 1.5)

    def test_only_up_to_what_hero_matches(self):
        # a short hero's call for less doesn't win the rest of what the limper puts in
        st = replay_hand(SEATS, 100.0, {"preflop": LIMP}, 1)[0]
        self.assertAlmostEqual(_owed(st, ["UTG"], 2.0, SEATS, 100.0), 1.0)

    def test_an_all_in_player_owes_nothing(self):
        t = pf.Table(SEATS, stacks={"UTG": 1.0})
        st = replay_hand(t, 100.0, {"preflop": LIMP}, 1)[0]
        self.assertIn("UTG", st.allin)
        self.assertEqual(_owed(st, ["UTG"], 4.0, t, 100.0), 0.0)


class TestPrice(unittest.TestCase):
    def test_the_limper_is_in_the_price_as_in_the_equity(self):
        res = bb(LIMP)
        self.assertIn("price 18% of the pot against UTG, LJ, HJ", price(res))   # 3 / 16.5, was 3 / 13.5
        self.assertGreater(continues(res), 0.10)                                # was 3.7%

    def test_a_spot_nobody_owes_in_keeps_its_price(self):
        self.assertIn("price 24% of the pot against LJ, HJ", price(bb(NO_LIMP)))   # 3 / 12.5

    def test_the_answers_range_for_the_flop_widens_with_it(self):
        # the big blind's call was six combos, filled to the 8-combo floor, and every later street read that
        pre = LIMP + [CALL("BB"), CALL("UTG")]
        res = advisor("fast").advise(dict(players=7, hero="BB", hole="As4s", stack=100, board="2s4d6c", preflop=pre))
        self.assertGreater(res["hero_range_combos"], 60)


if __name__ == "__main__":
    unittest.main()
