"""The Breakdown view (gto/breakdown.py): hand categories, range equities, and the advisor's "breakdown" flag."""
import json
import os
import unittest
import numpy as np
from gto.breakdown import made_category, draw_category, categorize, range_equities, breakdown, CURVE_POINTS, MADE, DRAWS
from gto.cards import cards, COMBO_INDEX
from gto.ranges import parse_range, remove_blockers

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def made(hole, board):
    b = cards(board)
    idx = np.array([COMBO_INDEX[tuple(sorted(cards(hole)))]])
    return categorize(b, idx)[0][0]


def draw(hole, board):
    return draw_category(cards(hole), cards(board))


class Categories(unittest.TestCase):
    def test_made_hands(self):
        cases = [
            ("AhKh", "Ks7d2c", "top_tk"), ("KhQh", "Ks7d2c", "top_gk"), ("KhJh", "Ks7d2c", "top_gk"),
            ("Kh9h", "Ks7d2c", "top_wk"), ("AhKh", "As7d2c", "top_tk"),
            ("AhAd", "Ks7d2c", "overpair"), ("QhQd", "Ks7d2c", "middle"), ("8h8d", "Ks7d2c", "middle"),
            ("5h5d", "Ks7d2c", "weak"), ("7h6h", "Ks7d2c", "middle"), ("Ah2h", "Ks7d2c", "weak"),
            ("7h7c", "Ks7d2c", "set"), ("Kh7h", "Ks7d2c", "two_pair"), ("Ah3d", "Ks7d2c", "ace_high"),
            ("QhJh", "Ks7d2c", "nothing"),
            # wheel and broadway straights, flushes, full houses, quads
            ("Ah2d", "3c4d5h", "straight"), ("6h2d", "3c4d5h", "straight"), ("AhKd", "QcJdTh", "straight"),
            ("Ah2h", "Kh7h4h", "flush"), ("5h4h", "3h2hAh", "quads"), ("KhKd", "Ks7d7c", "full_house"),
            ("7h7s", "Ks7d7c", "quads"),
            # paired boards: the board's pair is everyone's
            ("Kh2d", "Ks7d7c", "top_wk"), ("Ah7h", "Ks7d7c", "trips"), ("AhQd", "Ks7d7c", "ace_high"),
            ("QhQd", "Ks7d7c", "middle"), ("8h8d", "7s7d2c", "overpair"), ("AhQd", "7s7d7c", "ace_high"),
            ("2h2d", "7s7d7c", "full_house"), ("2h3d", "7s7d7c", "nothing"), ("AhKd", "7s7d7c7h", "ace_high"), ("AhKd", "Ks7d7cKh", "full_house"),
            # river boards that play themselves
            ("2h2d", "9c8dTh7sJc", "board"), ("Qh2d", "9c8dTh7sJc", "straight"),
            ("Ac2d", "3h5h8h9hKh", "board"), ("2h3d", "4h5h8h9hKh", "board"), ("Qh3d", "4h5h8h9hKh", "flush"), ("Ad3d", "4h5h8h9hKh", "board"),
        ]
        for hole, board, want in cases:
            with self.subTest(hole=hole, board=board):
                self.assertEqual(made(hole, board), want)

    def test_draws(self):
        cases = [
            ("AhKh", "Qh7h2c", "fd"), ("Ah2d", "Qh7h2h", "fd"), ("AhKd", "Qh7h2h", "fd"), ("KdQd", "Qh7h2h", "none"),   # a monotone flop
            ("9h8d", "Tc7s2d", "oesd"), ("9h8d", "Jc7s2d", "gutshot"), ("Ah2d", "3c4s9d", "gutshot"),
            ("9h8h", "Th7h2c", "combo"), ("9h7d", "Jc8s5d", "oesd"),   # double gutter: J 9 8 7 5
            ("AhKd", "Jh9h4c", "backdoor"), ("AhKd", "QcJs4d", "gutshot"), ("KhQd", "JcTs4d", "oesd"), ("AhKd", "Qc8s4d", "backdoor"),
            ("2c3d", "Kh8h9s", "none"),
            # a four-flush turn: one card of the suit is a flush, none no draw; a four-straight board isn't a draw
            ("Ah2d", "Kh8h4h3h", "gutshot"), ("AdKd", "Kh8h4h3h", "none"), ("AdKc", "5h6s7d8c", "none"),
            ("9c2d", "5h6s7d8c", "none"), ("4c2d", "5h6s7d8c", "none"), ("Ad2d", "5h6s7d9c", "none"), ("Td2d", "5h6s7d9c", "gutshot"), ("Td8d", "5h6s7d9c", "none"),
            ("AhKh", "QhJh2c3d4s", "none"), ("Ts9d", "8h7s2d", "oesd"), ("Ts9d", "7h6s2d", "gutshot"),
        ]
        for hole, board, want in cases:
            with self.subTest(hole=hole, board=board):
                self.assertEqual(draw(hole, board), want)

    def test_ids_known(self):
        board = cards("Th9h2c")
        idx = np.nonzero(parse_range("22+,A2s+,K9s+,QTs+,J9s+,T8s+,97s+,ATo+,KJo+") > 0)[0]
        m, d = categorize(board, idx)
        self.assertTrue(set(m) <= {c for c, _ in MADE} and set(d) <= {c for c, _ in DRAWS})


class Equities(unittest.TestCase):
    def test_river_exact(self):
        e = lambda h, x: x[COMBO_INDEX[tuple(sorted(cards(h)))]]
        board = cards("Ks7d2c5h9s")
        e0, e1 = range_equities(board, parse_range("AA"), parse_range("QQ,K7"))
        # AhAd beats the 6 queens, loses to the 9 kings and sevens (Ks and 7d are on the board)
        self.assertAlmostEqual(e("AhAd", e0), 0.4, places=6)
        self.assertAlmostEqual(e("QhQd", e1), 0.0, places=6)
        self.assertAlmostEqual(e("Kh7h", e1), 1.0, places=6)
        self.assertTrue(np.isnan(e("KhKd", e0)) and np.isnan(e("AhAd", e1)))   # outside the ranges
        # ties split: both play K K 7 7 A
        e0, e1 = range_equities(cards("KsKd7c7h2s"), parse_range("AQ"), parse_range("AJ"))
        self.assertAlmostEqual(e("AhQh", e0), 0.5, places=6)
        self.assertAlmostEqual(e("AcJc", e1), 0.5, places=6)

    def test_turn_symmetric(self):
        board = cards("Ks7d2c5h")
        w0, w1 = (remove_blockers(parse_range(r), board) for r in ("AA,KQ,98s,66", "QQ,AK,T9s,77"))
        e0, e1 = range_equities(board, w0, w1)
        # the pair weights summed both ways: sum_i w0 e0 cnt0 + sum_j w1 e1 cnt1 = total; check the averages add to 1
        from gto.solver import MEMBER_FULL
        i0, i1 = np.nonzero(w0)[0], np.nonzero(w1)[0]
        ok = (MEMBER_FULL[:, i0].T @ MEMBER_FULL[:, i1]) == 0
        W = ok * w0[i0][:, None] * w1[i1][None, :]
        # each side's average weighted by its compatible opponent mass
        a0 = (e0[i0] * W.sum(1)).sum() / W.sum()
        a1 = (e1[i1] * W.sum(0)).sum() / W.sum()
        self.assertAlmostEqual(a0 + a1, 1.0, places=3)


    def test_runouts_named(self):
        """The payload says what the equities are over, for the views' wording: the river's board alone, every
        river card on the turn, a sample of the flop's runouts once the ranges are wide."""
        def bd(board, r0="AA,KQ,98s,66", r1="QQ,AK,T9s,77"):
            b = cards(board)
            w0, w1 = (remove_blockers(parse_range(r), b) for r in (r0, r1))
            return breakdown(b, ["check"], "BTN", w0, np.ones((1, 1326)) * (w0 > 0), np.zeros(1326), {"BB": w1}, 10.0)
        self.assertEqual(bd("Ks7d2c5h9s")["runouts"], [1, 1])
        self.assertEqual(bd("Ks7d2c5h")["runouts"], [48, 48])
        n, of = bd("Ks7d2c", "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,A2o+,K2o+,Q2o+", "22+,A2s+,K2s+,Q2s+,J2s+,A2o+,K2o+")["runouts"]
        self.assertEqual(of, 1176)
        self.assertLess(n, of)
        self.assertNotIn("runouts", breakdown(cards("Ks7d2c"), ["check"], "BTN", parse_range("AA"), np.ones((1, 1326)),
                                              np.zeros(1326), {"BB": parse_range("QQ")}, 10.0, equity=False))


class AdvisorFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gto.advisor import Advisor
        from gto.solver import Config
        cls.adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        with open(os.path.join(EX, "flop.json")) as f:
            cls.state = json.load(f)
        cls.plain = cls.adv.advise(dict(cls.state))
        cls.res = cls.adv.advise(dict(cls.state, breakdown=True))

    def test_flag_off_no_keys(self):
        self.assertNotIn("error", self.plain)
        self.assertNotIn("breakdown", self.plain)

    def test_same_solve(self):
        self.assertEqual(self.res["range_action_freq"], self.plain["range_action_freq"])
        self.assertEqual(self.res["solver"]["iterations"], self.plain["solver"]["iterations"])

    def test_payload(self):
        bd = self.res["breakdown"]
        self.assertEqual(bd["actions"], self.res["grid"]["actions"])
        h, (v,) = bd["hero"], bd["villains"]
        self.assertEqual(h["who"], "BTN"); self.assertEqual(v["who"], "BB")
        self.assertEqual(h["mine"], ["top_tk", "none"])   # AhQd on Qs8d3c
        for s, want in ((h, self.res["hero_range_combos"]), (v, self.res["villain_range_combos"])):
            self.assertAlmostEqual(sum(g["c"] for g in s["made"]), want, delta=0.2)
            self.assertAlmostEqual(sum(g["c"] for g in s["draws"]), want, delta=0.2)
            c = s["curve"]
            self.assertEqual(len(c), CURVE_POINTS)
            self.assertTrue(all(a >= b for a, b in zip(c, c[1:])), "curve sorted high to low")
            self.assertTrue(c[-1] <= s["eq"] <= c[0])
            # the categories' equities average to the range's
            self.assertAlmostEqual(sum(g["c"] * g["eq"] for g in s["made"]) / sum(g["c"] for g in s["made"]), s["eq"], delta=0.005)
            self.assertTrue(0 <= s["nuts"] <= 1)
        self.assertAlmostEqual(h["eq"] + v["eq"], 1.0, delta=0.03)
        # hero's mix over the categories adds up to the range's
        for a, name in enumerate(bd["actions"]):
            tot = sum(g["c"] * g["f"][a] for g in h["made"]) / sum(g["c"] for g in h["made"])
            self.assertAlmostEqual(tot, self.res["range_action_freq"][name], delta=0.005)
        for g in h["made"]:
            self.assertAlmostEqual(sum(g["f"]), 1.0, delta=0.01)
        self.assertAlmostEqual(h["ev"] + v["ev"], bd["pot"], delta=0.02)
        self.assertNotIn("f", v["made"][0])
        self.assertLess(len(json.dumps(bd)), 8000)

    def test_preflop_ignores_flag(self):
        st = dict(self.state, board="", flop=[], breakdown=True)
        self.assertNotIn("breakdown", self.adv.advise(st))


class MultiwayFlag(unittest.TestCase):
    def test_three_way_hero_mix_no_equity(self):
        from gto import advisor as A
        from tests.test_multiway_solver import PRE, CHECKS
        adv = A.Advisor(time_budget=1e9)
        adv.mw_max_iters = 10
        st = {"players": 6, "hero": "BB", "hole": "Kd9d", "board": "Kh7d2c9s4h", "stack": 100, "preflop": PRE,
              "flop": CHECKS, "turn": CHECKS, "river": [], "breakdown": True}
        r = adv.advise(st)
        self.assertTrue(r["solved"])
        bd = r["breakdown"]
        self.assertEqual(len(bd["villains"]), 2)
        self.assertNotIn("eq", bd["hero"])
        self.assertEqual(bd["hero"]["draws"], [])
        for a, name in enumerate(bd["actions"]):
            tot = sum(g["c"] * g["f"][a] for g in bd["hero"]["made"]) / sum(g["c"] for g in bd["hero"]["made"])
            self.assertAlmostEqual(tot, r["range_action_freq"][name], delta=0.005)


if __name__ == "__main__":
    unittest.main()
