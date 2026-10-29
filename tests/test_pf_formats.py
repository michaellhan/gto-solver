"""Other games' preflop charts (gto/pf_formats.py, gto/pf_format_data.py from scripts/build_pf_formats.py): the
decisions they cover, the moves added to the 100bb charts, and the advice playing them as the game's own charts."""
import unittest

import numpy as np

from gto import pf_formats as PF
from gto import preflop as pf
from gto.pf_classes import CLASS_OF, CLASSES

R = lambda p: {"pos": p, "action": "raise"}
F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
K = len(CLASSES)


class _Data:
    """Replaces the generated moves for a test."""

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        self.old = PF._DATA
        PF._DATA = self.data
        pf._STRAT_CACHE.clear()

    def __exit__(self, *a):
        PF._DATA = self.old
        pf._STRAT_CACHE.clear()


class TestKeys(unittest.TestCase):
    def test_line_keys(self):
        """The charts' heads-up lines: raise first in, facing an open, the opener facing the 3bet, the 3bettor
        facing the 4bet, the opener facing the 5bet; a call or a limp in the line is none of them."""
        L = PF.line_key
        self.assertEqual(L("UTG", []), "UTG|")
        self.assertEqual(L("BTN", [F("UTG"), F("HJ"), F("CO")]), "BTN|")
        self.assertEqual(L("BB", [R("BTN"), F("SB")]), "BB|BTN:r")
        self.assertEqual(L("BTN", [R("BTN"), F("SB"), R("BB")]), "BTN|BTN:r,BB:r")
        self.assertEqual(L("BB", [R("BTN"), R("BB"), R("BTN")]), "BB|BTN:r,BB:r,BTN:r")
        self.assertEqual(L("BTN", [R("BTN"), R("BB"), R("BTN"), R("BB")]), "BTN|BTN:r,BB:r,BTN:r,BB:r")
        for hero, acts in (("SB", [R("BTN"), R("BB")]), ("BB", [R("BTN"), C("SB")]), ("BTN", [R("BTN")]),
                           ("BB", [R("BTN"), R("SB"), R("BTN")]), ("BB", [C("SB")])):
            self.assertIsNone(L(hero, acts), (hero, acts))
        # a flat of the raise plays the chart against the raise alone; a limp or hero's own call doesn't
        self.assertEqual(PF.std_actions("BB", [R("CO"), C("BTN")]), [R("CO"), F("BTN")])
        self.assertIsNone(PF.std_actions("BB", [C("SB")]))
        self.assertIsNone(PF.std_actions("BTN", [R("CO"), C("BTN"), R("BB")]))

    def test_analog(self):
        """Depth and ante moves are read at the 6-max seats whose charts the table plays (preflop.chart_key)."""
        T8 = pf.Table(pf.SEATS[8])
        self.assertEqual(PF.analog("BB", [R("UTG1")], T8), "BB|UTG:r")
        self.assertEqual(PF.analog("LJ", [], T8), "UTG|")
        self.assertEqual(PF.analog("BB", [R("SB")], pf.Table(pf.SEATS[2])), "BB|SB:r")
        self.assertEqual(PF._seat8("UTG", pf.Table(pf.SEATS[9])), "UTG")
        self.assertEqual(PF._seat8("UTG2", pf.Table(pf.SEATS[9])), "UTG1")
        self.assertEqual(PF._seat8("BTN", pf.Table(pf.SEATS[9])), "BTN")

    def test_encoding(self):
        d = np.linspace(-0.6, 0.6, K)
        self.assertLess(np.abs(PF.decode(PF.encode(d)) - d).max(), 0.0026)


class TestMoves(unittest.TestCase):
    def test_moves(self):
        """Depth moves are linear between the solved depths (100bb is none), the ante's scale with the dead money,
        7-9 handed tables add their own; the default game and a line off the charts have none."""
        up = np.full(K, 0.2)
        data = {"depth": {40.0: {"BB|BTN:r": {"raise": up}}, 60.0: {"BB|BTN:r": {"raise": 0 * up}}},
                "ante": {40.0: {"BB|BTN:r": {"call": up}}, 100.0: {"BB|BTN:r": {"call": up}}, 20.0: {}},
                "players": {8.0: {"BB|UTG1:r": {"raise": up}}}}
        T = lambda n=6, d=100.0, a=0.0: pf.Table(pf.SEATS[n], a, False, None, d)
        with _Data(data):
            self.assertEqual(PF.moves("BB", [R("BTN"), F("SB")], T()), (None, {}))
            mv, cover = PF.moves("BB", [R("BTN"), F("SB")], T(d=50.0))
            self.assertAlmostEqual(float(mv["raise"][0]), 0.1)
            self.assertEqual(cover, {"depth": 50.0, "ante": 0.0})
            mv, cover = PF.moves("BB", [R("BTN"), F("SB")], T(d=40.0, a=PF.ANTE))
            self.assertAlmostEqual(float(mv["raise"][0]), 0.2)
            self.assertAlmostEqual(float(mv["call"][0]), 0.2)
            self.assertEqual(cover, {"depth": 40.0, "ante": PF.ANTE})
            mv, _ = PF.moves("BB", [R("BTN"), F("SB")], T(d=40.0, a=PF.ANTE / 2))
            self.assertAlmostEqual(float(mv["call"][0]), 0.1)
            self.assertEqual(PF.moves("BB", [R("BTN"), F("SB")], T(d=15.0)), (None, {}))
            self.assertEqual(PF.moves("BB", [R("CO"), F("BTN"), F("SB")], T(d=40.0)), (None, {}))
            mv, cover = PF.moves("BB", [R("UTG1")], T(n=8))
            self.assertAlmostEqual(float(mv["raise"][0]), 0.2)
            self.assertEqual(cover, {"depth": 100.0, "ante": 0.0, "base": "6max"})

    def test_strategy_is_the_chart_moved(self):
        """At a covered decision the advice's chart is the 100bb chart moved per class (each action in [0, 1], the
        continuing total at most 1); the default game's charts are untouched."""
        up = np.full(K, 0.1)
        data = {"depth": {40.0: {"BB|BTN:r": {"raise": up, "call": -up}}, 20.0: {}, 60.0: {}}}
        pre = [F("UTG"), F("HJ"), F("CO"), R("BTN"), F("SB")]
        base = pf.strategy_for("BB", pre, pf.Table(pf.SEATS[6]))
        with _Data(data):
            got = pf.strategy_for("BB", pre, pf.Table(pf.SEATS[6], 0.0, False, None, 40.0))
            want = PF.apply({k: v for k, v in base.items() if k != "fold"}, data["depth"][40.0]["BB|BTN:r"])
            for a in ("raise", "call"):
                self.assertLess(np.abs(pf.monotone(dict(want, fold=1 - sum(want.values())))[a] - got[a]).max(), 1e-9)
            self.assertGreater(got["raise"].mean(), base["raise"].mean() + 0.05)
            same = pf.strategy_for("BB", pre, pf.Table(pf.SEATS[6]))
            self.assertLess(np.abs(same["raise"] - base["raise"]).max(), 1e-12)

    def test_table_size_moves_the_6max_chart(self):
        """An 8-handed seat's chart is the 6-max chart of the seat it plays moved by the 8-handed solve, not the
        table's own rules (the 7+ handed EP chart) moved: UTG and UTG1 both from the 6-max UTG chart."""
        T8 = pf.Table(pf.SEATS[8])
        utg6 = pf.strategy_for("UTG", [], pf.Table(pf.SEATS[6]))["raise"]
        rules = pf.strategy_for("UTG", [], T8)["raise"]
        self.assertLess(rules.mean(), utg6.mean() - 0.02)        # the EP chart
        up = np.full(K, -0.01)
        with _Data({"players": {8.0: {"UTG|": {"raise": up}, "UTG1|": {"raise": 0 * up}}}}):
            got = pf.strategy_for("UTG", [], T8)["raise"]
            self.assertLess(np.abs(got - np.clip(utg6 - 0.01, 0, 1)).max(), 1e-9)
            got1 = pf.strategy_for("UTG1", [F("UTG")], T8)["raise"]
            self.assertLess(np.abs(got1 - utg6).max(), 1e-9)

    def test_advice_prices_from_the_game(self):
        """The advice re-prices a covered decision from the game's own (its depth, its ante), so a spot at its
        depth isn't re-priced for depth; one the moves don't cover still is."""
        from gto.advisor import Advisor
        up = np.zeros(K)
        data = {"depth": {40.0: {"BB|BTN:r": {"raise": up}}, 20.0: {}, 60.0: {}}}
        pre = [F("UTG"), F("HJ"), F("CO"), {"pos": "BTN", "action": "raise", "to": 2.5}, F("SB")]
        state = {"hero": "BB", "hole": "Ts9s", "preflop": pre, "stack": 40}
        adv = Advisor(time_budget=0.2, prior_budget=0.1)
        with _Data(data):
            r = adv.advise(dict(state))
            self.assertFalse(any("shallower than the charts" in n for n in r.get("notes", [])), r.get("notes"))
            self.assertIn("preflop: the 40bb chart (the 100bb chart moved by solves of this game)", r["notes"])
        with _Data({}):
            r = adv.advise(dict(state))
            self.assertTrue(any("shallower than the charts" in n for n in r.get("notes", [])), r.get("notes"))


class TestShipped(unittest.TestCase):
    def test_shipped_moves(self):
        """The generated moves (gto/pf_format_data.py): every dimension and depth, keys that are chart lines, and
        charts that stay in range; at 100bb 6-max and heads-up the charts are the hand-written ones."""
        from gto.pf_format_data import DELTAS
        if not DELTAS:
            self.skipTest("gto/pf_format_data.py holds no game yet")
        PF._DATA = None
        d = PF._data()
        for dim in ("depth", "ante", "players", "straddle", "hu_depth"):
            self.assertIn(dim, d)
        self.assertEqual(sorted(d["depth"]), [x for x in PF.DEPTHS if x != 100.0])
        self.assertEqual(sorted(d["hu_depth"]), [x for x in PF.HU_DEPTHS if x != 100.0])
        for dim, sets in d.items():
            for sub, nodes in sets.items():
                self.assertIn("UTG|" if dim != "hu_depth" else "SB|", nodes, (dim, sub))
                for key, mv in nodes.items():
                    hero, seq = key.split("|")
                    acts = [R(x.split(":")[0]) for x in seq.split(",") if x]
                    self.assertEqual(PF.line_key(hero, acts), key)
                    for a, v in mv.items():
                        self.assertEqual(v.shape, (K,))
                        self.assertLessEqual(np.abs(v).max(), 1.0)
        pf._STRAT_CACHE.clear()
        for n, stack, ante in ((6, 40.0, 0.0), (6, 20.0, 0.125), (8, 100.0, 0.0), (9, 40.0, 1 / 9), (2, 40.0, 0.0)):
            T = pf.Table(pf.SEATS[n], ante, False, None, stack)
            s = pf.strategy_for(pf.SEATS[n][-2], [F(p) for p in pf.SEATS[n][:-2]], T)
            tot = sum(s.values())
            self.assertLess(np.abs(tot - 1).max(), 1e-9)
            self.assertTrue(all(v.min() >= -1e-12 for v in s.values()))
        self.assertEqual(PF.moves("BTN", [], pf.Table(pf.SEATS[6])), (None, {}))
        self.assertEqual(PF.moves("SB", [], pf.Table(pf.SEATS[2])), (None, {}))


if __name__ == "__main__":
    unittest.main()
