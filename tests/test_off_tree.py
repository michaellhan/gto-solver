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
            for method in ("near", "interp", "mdf", "prop", "nearmdf"):
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

    def test_nearmdf(self):
        """Between two stored sizes "nearmdf" keeps the nearest size's mix, each combo only moving between folding and
        continuing, and the range continues as much as with mdf; outside them it is mdf."""
        mixes = [ot.by_kind(s["names"], s["strat"]) for s in (self.small, self.big)]
        for x, near in ((0.45, mixes[0]), (0.5, mixes[0]), (0.6, mixes[1])):
            C, T = x, 1.0 + x   # (pot 1)
            got = ot.answer([self.small, self.big], C, T, "nearmdf")
            want = ot.answer([self.small, self.big], C, T, "mdf")
            self.assertAlmostEqual(cont(got, self.r), cont(want, self.r), places=9)
            np.testing.assert_allclose(sum(got[k] for k in ot.KINDS), 1.0)
            c_near = near["call"] + near["raise"] + near["allin"]
            c_got = got["call"] + got["raise"] + got["allin"]
            for k in ("call", "raise", "allin"):   # (its continuing actions in the nearest's proportions)
                np.testing.assert_allclose(got[k] * c_near, near[k] * c_got, atol=1e-12)
        # (the strongest combo jams facing 33%, and keeps doing so facing 50%, where mdf mixes in 75%'s calls)
        got = ot.answer([self.small, self.big], 0.5, 1.5, "nearmdf")
        self.assertAlmostEqual(float(got["allin"][0]), 1.0)
        self.assertLess(float(ot.answer([self.small, self.big], 0.5, 1.5, "mdf")["allin"][0]), 0.9)
        for x in (0.2, 1.0, 1.5):
            got = ot.answer([self.small, self.big], x, 1.0 + x, "nearmdf")
            want = ot.answer([self.small, self.big], x, 1.0 + x, "mdf")
            for k in ot.KINDS:
                np.testing.assert_allclose(got[k], want[k], atol=1e-12)
        self.assertEqual(A.OFF_METHOD, "nearmdf")

    def test_not_held(self):
        """NaN columns (combos a source doesn't hold) stay NaN, and don't count in the range's share."""
        s2 = dict(self.big, strat=self.big["strat"].copy())
        s2["strat"][:, 3] = np.nan
        got = ot.answer([self.small, s2], 2.0, 4.0, "mdf")
        self.assertTrue(np.isnan(got["call"][3]))
        self.assertFalse(np.isnan(got["call"][:3]).any())

    def test_mix_from(self):
        """`mix_from`: the mix (and q) from the nearest of those sources; the range's continuing share still from all."""
        for x in (0.45, 0.2):
            C, T = x, 1.0 + x   # (pot 1: nearest the 33% bet)
            got = ot.answer([self.small, self.big], C, T, "nearmdf", mix_from=[1])
            want = ot.answer([self.small, self.big], C, T, "nearmdf")
            self.assertAlmostEqual(cont(got, self.r), cont(want, self.r), places=9)
            b = ot.by_kind(self.big["names"], self.big["strat"])
            c_b, c_got = b["call"] + b["raise"] + b["allin"], got["call"] + got["raise"] + got["allin"]
            for k in ("call", "raise", "allin"):   # (75%'s continuing actions, in its proportions)
                np.testing.assert_allclose(got[k] * c_b, b[k] * c_got, atol=1e-12)
        got = ot.answer([self.small, self.big], 0.45, 1.45, "near", mix_from=[1])
        np.testing.assert_allclose(got["fold"], ot.by_kind(self.big["names"], self.big["strat"])["fold"])

    def test_untaken_size_not_read(self):
        """Off a live solve, hero's mix facing a size the villain bets under OFF_SOURCE_LINE of their range (the flop
        library's STORE_LINE: it doesn't store those) isn't read: a 50% bet reads the 75% one's though 33% is nearer
        (the range's continuing share still by every size's price)."""
        from gto.solver import Decision
        self.assertEqual(A.OFF_SOURCE_LINE, fl.STORE_LINE)
        n = 4

        def dec(player, names, pot_now, bets, strat, children=None):
            d = Decision([None], player, names, children or [None] * len(names), pot_now, bets, n)
            d.ssum[0] = np.asarray(strat, dtype=np.float32)
            return d

        facing = lambda b, strat: dec(0, ["fold", "call", "raise 50%"], 10.0 + b, [0.0, b], strat)
        f33 = facing(3.3, [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
        f75 = facing(7.5, [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        f125 = facing(12.5, [[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        check = dec(0, ["check"], 10.0, [0.0, 0.0], [[1, 1, 1, 1]])
        evs = lambda ch, k: np.zeros((len(ch.names), n))
        one = np.ones(n)

        def villain(b33, b75=0.25, b125=None):   # (the BTN's share of their range on each line; the rest checks)
            b125 = 0.15 - b33 if b125 is None else b125
            return dec(1, ["check", "bet 33%", "bet 75%", "bet 125%"], 10.0, [0.0, 0.0],
                       [[1 - b33 - b75 - b125] * n, [b33] * n, [b75] * n, [b125] * n], [check, f33, f75, f125])

        mix, near, xs = A._off_tree_read(villain(0.01), 0, one, one, 5.0, 15.0, evs)   # (a 50% bet)
        self.assertIs(near, f75)
        np.testing.assert_allclose(xs, [0.33, 0.75, 1.25])
        self.assertAlmostEqual(float(mix["fold"][1]), 1.0)   # (75%'s mix: combo 1 folds, 33%'s calls everything)
        ks = [1.0 / ot.mdf(0.33), 0.5 / ot.mdf(0.75), 0.25 / ot.mdf(1.25)]
        k = sum(w * ks[i] for i, w in ot._weights(xs, 0.5))
        self.assertAlmostEqual(cont(mix, one), k * ot.mdf(0.5), places=6)
        want = ot.answer([dict(names=f.names, strat=f.avg_strategy0().astype(float), evs=evs(f, 0), C=f.bets[1],
                               T=f.pot_now, reach=one) for f in (f33, f75, f125)], 5.0, 15.0, "nearmdf", mix_from=[1, 2])
        for kk in ot.KINDS:
            np.testing.assert_allclose(mix[kk], want[kk], atol=1e-6)
        mix, near, xs = A._off_tree_read(villain(0.06), 0, one, one, 5.0, 15.0, evs)   # (taken 6%: read)
        self.assertIs(near, f33)
        # every size under it (the bets 12% of the range in all): all of them, as before
        mix, near, xs = A._off_tree_read(villain(0.04, 0.04, 0.04), 0, one, one, 5.0, 15.0, evs)
        self.assertIs(near, f33)
        # the river (OFF_LATER "above"): between two sizes the nearest's answer (None here), unless the villain doesn't
        # take the nearest: then read by the price
        self.assertIsNone(A._off_tree_read(villain(0.06), 0, one, one, 5.0, 15.0, evs, between=False, below=False))
        got = A._off_tree_read(villain(0.01), 0, one, one, 5.0, 15.0, evs, between=False, below=False)
        self.assertIs(got[1], f75)
        # a 100% bet with 33% taken, 75% and 125% not: the nearest's (75%) as before, not 33%'s two sizes below
        mix, near, xs = A._off_tree_read(villain(0.1, 0.01, 0.01), 0, one, one, 10.0, 20.0, evs)
        self.assertIs(near, f75)

    def test_readable(self):
        """The sizes a live solve's read takes its mix from: those taken, from the one just below the real size up."""
        T_, F_ = True, False
        self.assertEqual(ot.readable([0.33, 0.75, 1.25], [F_, T_, T_], 0.5), [1, 2])
        self.assertEqual(ot.readable([0.33, 0.75, 1.25], [F_, F_, T_], 0.4), [2])
        self.assertEqual(ot.readable([0.33, 0.75, 1.25, 2.5], [T_, F_, F_, T_], 1.0), [3])   # (not 33% for 100%)
        self.assertEqual(ot.readable([0.33, 0.75, 1.25], [T_, F_, F_], 1.0), [0, 1, 2])      # (none: all)
        self.assertEqual(ot.readable([0.33, 0.75, 1.25], [T_, T_, F_], 2.0), [0, 1, 2])      # (125% isn't: all)
        self.assertEqual(ot.readable([0.33, 0.75], [F_, T_], 0.2), [1])


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
        read by the bet's price, between two of them too (Advisor.off_tree_later "always"); "outside", between two
        the nearest size's; None, the nearest size's always."""
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
        # between two of its sizes (100%: 75% and 125%) by the price too, by default (OFF_LATER)
        mid = adv.instant(st([X("BB"), B("BTN", 9.1)]))
        self.assertIn("read for this 100% bet by its price", mid["approximate"]["label"])
        adv.off_tree_later = "outside"
        self.assertIn("the sizes since read at its nearest", adv.instant(st([X("BB"), B("BTN", 9.1)]))["approximate"]["label"])
        adv.off_tree_later = None
        old = adv.instant(st([X("BB"), B("BTN", 18.2)]))
        self.assertIn("the sizes since read at its nearest", old["approximate"]["label"])
        self.assertIsNone(adv.instant(st([X("BB"), B("BTN", 3.0)])))   # (33%: in its tree, read as before)

    def test_river_bigger_only(self):
        """The river reads by the price only a bet bigger than all of its solve's (OFF_LATER "above"): smaller, the
        nearest size's answer (a small river bet or raise read by the price gave up more), and between two (a tie)."""
        self.assertEqual(A.OFF_LATER, {"turn": "always", "river": "above"})
        adv = native.advisor("fast", budget=1.0, prior_budget=0.5)
        flop = [X("BB"), B("BTN", 1.8), {"pos": "BB", "action": "call"}]
        st = lambda river: {"players": 6, "stack": 100, "preflop": parse_actions(LINE), "board": BOARD + "Th3s",
                            "hero": "BB", "hole": "9s8s", "flop": flop, "turn": [X("BB"), X("BTN")], "river": river}
        adv.advise(st([]))
        # (the 9.1 bb pot: the river's tree bets 33/75/125/250%)
        big = adv.instant(st([X("BB"), B("BTN", 36.4)]))
        self.assertIn("read for this 400% bet by its price", big["approximate"]["label"])
        small = adv.instant(st([X("BB"), B("BTN", 1.4)]))
        self.assertIn("the sizes since read at its nearest", small["approximate"]["label"])
        adv.off_tree_later = dict(A.OFF_LATER, river="outside")
        self.assertIn("read for this 15% bet by its price", adv.instant(st([X("BB"), B("BTN", 1.4)]))["approximate"]["label"])
        mid = adv.instant(st([X("BB"), B("BTN", 9.1)]))   # (100%: between 75% and 125%)
        self.assertIn("the sizes since read at its nearest", mid["approximate"]["label"])


if __name__ == "__main__":
    unittest.main()
