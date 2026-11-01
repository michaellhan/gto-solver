"""Answers facing a bet or raise of a size the stored tree doesn't have (gto/off_tree.py), and Advisor.instant's use of
them on a library flop (advisor.instant_from_off_tree): read off the entry's stored decisions facing the sizes it has,
marked approximate; the flop's own solve, with the real size in its tree, still answers the request."""
import unittest

import numpy as np

from gto import advisor as A
from gto import flop_library as fl
from gto import native
from gto import off_tree as ot
from gto.cli import parse_actions


def src(names, strat, evs, C, T, reach):
    return {"names": names, "strat": np.asarray(strat, dtype=float), "evs": None if evs is None else
            np.asarray(evs, dtype=float), "C": C, "T": T, "reach": np.asarray(reach, dtype=float)}


def cont(mix, r):
    return float(((mix["call"] + mix["raise"] + mix["allin"]) * r).sum() / r.sum())


class TestOffTree(unittest.TestCase):
    def setUp(self):
        # four combos facing a 33% bet (C 1, T 4: pot 3) and a 75% bet (C 3, T 7: pot 4): folds, calls, raises
        self.r = np.array([1.0, 1.0, 1.0, 1.0])
        names = ["fold", "call", "raise to 9", "all-in"]
        self.small = src(names, [[0.0, 0.2, 0.0, 0.0], [0.0, 0.8, 1.0, 0.5], [0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]],
                         [[0, 0, 0, 0], [3.0, 1.0, 0.5, 0.0], [2.0, 0.0, -1, 0], [4.0, -2, -1, -0.2]], 1.0, 4.0, self.r)
        self.big = src(names, [[0.0, 0.6, 0.4, 1.0], [0.5, 0.4, 0.6, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
                       [[0, 0, 0, 0], [5.0, 0.2, 0.1, -2.0], [4.0, -1, -1, -3], [5.5, -3, -2, -3]], 3.0, 7.0, self.r)

    def test_kinds(self):
        self.assertEqual([ot.kind(n) for n in ("fold", "check", "call", "bet 33%", "raise to 9", "all-in")],
                         ["fold", "call", "call", "raise", "raise", "allin"])
        m = ot.by_kind(["fold", "call", "raise to 5", "raise to 9"], np.array([[0.1], [0.2], [0.3], [0.4]]))
        self.assertAlmostEqual(float(m["raise"][0]), 0.7)
        # a kind the decision lacks goes to the nearest it has: all-in to its raise, a raise it can't make to a call
        out = ot.onto(["fold", "call"], {"fold": np.array([0.2]), "call": np.array([0.3]), "raise": np.array([0.1]),
                                         "allin": np.array([0.4])})
        np.testing.assert_allclose(out[:, 0], [0.2, 0.8])
        out = ot.onto(["fold", "call", "raise to 9"], {"fold": np.array([0.0]), "call": np.array([0.5]),
                                                       "raise": np.array([0.0]), "allin": np.array([0.5])})
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 0.5])

    def test_price(self):
        self.assertAlmostEqual(ot.size_of(1.0, 4.0), 1 / 3)       # a third of the pot
        self.assertAlmostEqual(ot.size_of(3.7, 9.1), 3.7 / 5.4)   # a raise: its share of the pot after the call
        self.assertAlmostEqual(ot.mdf(1.0), 0.5)
        w = ot._weights([1 / 3, 0.75], 0.5)
        self.assertEqual([i for i, _ in w], [0, 1])
        self.assertAlmostEqual(w[1][1], np.log(1.5) / np.log(2.25))
        self.assertEqual(ot._weights([1 / 3, 0.75], 0.25), [(0, 1.0)])   # outside: the nearest alone
        self.assertEqual(ot._weights([1 / 3, 0.75], 2.0), [(1, 1.0)])

    def test_stored_size_reads_its_own_decision(self):
        """At a size the tree has, every method answers that decision's mix."""
        for x, s in ((1 / 3, self.small), (0.75, self.big)):
            C, T = s["C"], s["T"]
            want = ot.by_kind(s["names"], s["strat"])
            for method in ("near", "interp", "mdf", "prop"):
                got = ot.answer([self.small, self.big], C, T, method)
                for k in ot.KINDS:
                    np.testing.assert_allclose(got[k], want[k], atol=1e-9, err_msg=f"{method} {k} at {x}")

    def test_near(self):
        got = ot.answer([self.small, self.big], 2.0, 4.0, "near")   # (a 100% bet: the 75% one's)
        np.testing.assert_allclose(got["fold"], ot.by_kind(self.big["names"], self.big["strat"])["fold"])

    def test_mdf_target(self):
        """The range continues k(x) times the minimum defence (k from the stored sizes either side, linear in log x,
        the nearest's outside them), so it continues less the bigger the bet; its mix is a mix."""
        ks = [cont(ot.by_kind(s["names"], s["strat"]), self.r) / ot.mdf(ot.size_of(s["C"], s["T"]))
              for s in (self.small, self.big)]
        prev = 2.0
        for x in (0.2, 0.5, 1.0, 1.5, 3.0):
            C, T = x, 1.0 + x   # (pot 1)
            got = ot.answer([self.small, self.big], C, T, "mdf")
            w = ot._weights([1 / 3, 0.75], x)
            k = sum(wt * ks[i] for i, wt in w)
            self.assertAlmostEqual(cont(got, self.r), min(1.0, k * ot.mdf(x)), places=9)
            self.assertLess(cont(got, self.r), prev + 1e-12)
            prev = cont(got, self.r)
            tot = sum(got[k] for k in ot.KINDS)
            np.testing.assert_allclose(tot, 1.0)
            self.assertTrue(all((got[k] >= -1e-12).all() for k in ot.KINDS))

    def test_mdf_worst_fold_first(self):
        """Facing a bigger bet than the biggest stored, the continuing hands that realize the least of the pot (by
        their stored EVs) fold first; the best keep their mix."""
        got = ot.answer([self.small, self.big], 6.0, 10.0, "mdf")   # (a 150% bet)
        b = ot.by_kind(self.big["names"], self.big["strat"])
        self.assertAlmostEqual(float(got["fold"][0]), float(b["fold"][0]))   # (the strongest: unchanged)
        self.assertGreater(float(got["fold"][2]), float(b["fold"][2]) - 1e-12)
        self.assertGreaterEqual(float(got["fold"][1]), float(b["fold"][1]))
        # (combo 2 realizes less than combo 1, so it folds first: all of it before combo 1 folds any more)
        if got["fold"][1] > b["fold"][1] + 1e-9:
            self.assertAlmostEqual(float(got["fold"][2]), 1.0)
        # a smaller bet than the smallest stored: the folds that realize the most continue first, as calls
        got = ot.answer([self.small, self.big], 0.25, 1.5, "mdf")   # (a 20% bet)
        s = ot.by_kind(self.small["names"], self.small["strat"])
        self.assertLessEqual(float(got["fold"][3]), float(s["fold"][3]))
        self.assertGreaterEqual(float(got["call"][3]), float(s["call"][3]))

    def test_not_held(self):
        """NaN columns (combos a source doesn't hold) stay NaN, and don't count in the range's share."""
        s2 = dict(self.big, strat=self.big["strat"].copy())
        s2["strat"][:, 3] = np.nan
        got = ot.answer([self.small, s2], 2.0, 4.0, "mdf")
        self.assertTrue(np.isnan(got["call"][3]))
        self.assertFalse(np.isnan(got["call"][:3]).any())


# On the shipped library: the BTN's open called by the BB, BB to act facing the BTN's bet after a check.
SID = "srp_btn_bb"
LINE = dict((s[0], s[2]) for s in fl.SPOTS)[SID]
BOARD = "Kc7d2h"
X = lambda p: {"pos": p, "action": "check"}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
R = lambda p, a: {"pos": p, "action": "raise", "to": a}


class TestInstantOffTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib = fl.default_library()
        if lib is None or not any(s["id"] == SID for s in lib.index().get("spots", [])):
            raise unittest.SkipTest("the shipped library doesn't hold " + SID)
        cls.adv = native.advisor("fast", budget=0.5, prior_budget=0.5)

    def state(self, flop, hero, hole):
        return {"players": 6, "stack": 100, "preflop": parse_actions(LINE), "board": BOARD, "hero": hero,
                "hole": hole, "flop": flop, "turn": [], "river": []}

    def test_off_tree_bet(self):
        """A 55% or 100% bet (the tree has 33% and 75%): an approximate answer at once, read off the stored
        decisions; a hand folds more often the bigger the bet."""
        folds = []
        for amt in (1.5, 3.0, 5.5, 8.0):   # (27%, 55%, 100%, 145% of the 5.5 bb pot)
            res = self.adv.instant(self.state([X("BB"), B("BTN", amt)], "BB", "9s8s"))
            self.assertIsNotNone(res, amt)
            self.assertEqual(res["approximate"]["source"], "offtree")
            self.assertIn("33% and 75% bets", res["approximate"]["label"])
            self.assertAlmostEqual(sum(a["freq"] for a in res["actions"]), 1.0, places=2)
            self.assertEqual([a["action"] for a in res["actions"]][:2], ["fold", "call"])
            self.assertEqual(res["actions"][1]["amount_bb"], amt)
            raise_ = next(a for a in res["actions"] if a["action"].startswith("raise"))
            self.assertAlmostEqual(raise_["amount_bb"], round(amt + 0.5 * (5.5 + 2 * amt), 2), places=2)
            folds.append(res["actions"][0]["freq"])
        self.assertEqual(folds, sorted(folds))
        self.assertGreater(folds[-1], folds[0])

    def test_in_tree_unchanged(self):
        """A size the tree has (within 15%): no instant answer, as before (the library answers the request at once)."""
        self.assertIsNone(self.adv.instant(self.state([X("BB"), B("BTN", 2.0)], "BB", "9s8s")))
        res = self.adv.advise(self.state([X("BB"), B("BTN", 2.0)], "BB", "9s8s"))
        self.assertIn("precomputed", " ".join(res.get("notes", [])))

    def test_off_tree_raise(self):
        """A check-raise off the tree's 50% raise: read off the stored decision facing it."""
        res = self.adv.instant(self.state([X("BB"), B("BTN", 1.8), R("BB", 5.5)], "BTN", "AsQs"))
        self.assertIsNotNone(res)
        self.assertEqual(res["approximate"]["source"], "offtree")
        self.assertIn("raise", res["approximate"]["label"])
        self.assertAlmostEqual(res["actions"][1]["amount_bb"], 3.7)

    def test_not_off_tree(self):
        """Hero's own off-tree bet, a jam, or a hand off the entry's range: no such answer."""
        self.assertIsNone(self.adv.instant(self.state([X("BB"), B("BTN", 97.5)], "BB", "9s8s")))
        self.assertIsNone(self.adv.instant(self.state([X("BB"), B("BTN", 3.0)], "BB", "AhKd")))  # (BB 3bets AKo)
        self.assertIsNone(self.adv.instant(self.state([X("BB")], "BTN", "9s8s")))

    def test_own_solve_replaces_it(self):
        """The request solves the flop live with the real size in its tree; after it, no approximate answer."""
        s = self.state([X("BB"), B("BTN", 5.5)], "BB", "Qh7h")
        res = self.adv.advise(s)
        self.assertNotIn("approximate", res)
        self.assertTrue(any(a["action"] == "call" for a in res["actions"]))
        got = self.adv.instant(s)
        self.assertTrue(got is None or got["approximate"]["source"] != "offtree")


    def test_turn_off_the_earlier_solve(self):
        """The turn facing a bet off the tree of your earlier decision's solve: that solve's answers facing its sizes,
        read by the bet's price where it is bigger than all of them (Advisor.off_tree_later), else the nearest size's."""
        adv = native.advisor("fast", budget=2.0, prior_budget=1.0)   # (under ~1 s the turn's tree is cut to one size)
        flop = [X("BB"), B("BTN", 1.8), {"pos": "BB", "action": "call"}]
        st = lambda turn: {"players": 6, "stack": 100, "preflop": parse_actions(LINE), "board": BOARD + "Th",
                           "hero": "BB", "hole": "9s8s", "flop": flop, "turn": turn, "river": []}
        adv.advise(st([]))
        got = adv.instant(st([X("BB"), B("BTN", 18.2)]))   # (200% of the 9.1 bb pot: the tree has 33/75/125%)
        self.assertEqual(got["approximate"]["source"], "solve")
        self.assertIn("read for this 200% bet by its price", got["approximate"]["label"])
        self.assertAlmostEqual(sum(a["freq"] for a in got["actions"]), 1.0, places=2)
        self.assertEqual(got["actions"][1], {"action": "call", "freq": got["actions"][1]["freq"], "amount_bb": 18.2})
        # between two of its sizes (100%: 75% and 125%) the nearest's answer, by default (OFF_LATER)
        mid = adv.instant(st([X("BB"), B("BTN", 9.1)]))
        self.assertIn("the sizes since read at its nearest", mid["approximate"]["label"])
        adv.off_tree_later = "always"
        self.assertIn("read for this 100% bet by its price", adv.instant(st([X("BB"), B("BTN", 9.1)]))["approximate"]["label"])
        adv.off_tree_later = None
        old = adv.instant(st([X("BB"), B("BTN", 18.2)]))
        self.assertIn("the sizes since read at its nearest", old["approximate"]["label"])
        self.assertIsNone(adv.instant(st([X("BB"), B("BTN", 3.0)])))   # (33%: in its tree, read as before)


if __name__ == "__main__":
    unittest.main()
