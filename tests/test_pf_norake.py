"""No-rake charts (gto/pf_norake.py): without rake hero's preflop advice is the chart moved from the raked solves it was
fit to by Lizard's preflop solver's no-rake vs raked change at the spot; with rake, the chart as it was (re-priced for the
rake); the big blind's defence and the opener's calls of a blind's 3bet in position (written without rake) don't move;
the ranges entering the flop stay the charts' (the flop library's spots are keyed by them)."""
import unittest

import numpy as np  # noqa: E402

from gto import preflop as pf  # noqa: E402
from gto import pf_norake  # noqa: E402
from gto.advisor import Advisor, preflop_ranges  # noqa: E402

F = lambda p: {"pos": p, "action": "fold"}  # noqa: E731
R = lambda p: {"pos": p, "action": "raise"}  # noqa: E731
VS_BTN = [F("UTG"), F("HJ"), F("CO"), R("BTN")]


def shares(res):
    return {a["action"]: a["freq"] for a in res["actions"]}


class TestNoRakeCharts(unittest.TestCase):
    def test_spot_keys(self):
        T = pf.Table(pf.SEATS[6])
        self.assertEqual(pf_norake.spot_key("SB", VS_BTN, T), "6|SB|BTN")
        self.assertEqual(pf_norake.spot_key("BTN", [F("UTG"), F("HJ"), F("CO")], T), "6|BTN|")
        self.assertEqual(pf_norake.spot_key("CO", [F("UTG"), F("HJ"), R("CO"), R("BTN"), F("SB"), F("BB")], T), "6|CO|CO,BTN")
        # written without rake already: the big blind facing an open, the opener's calls in position of a blind's 3bet
        self.assertIsNone(pf_norake.spot_key("BB", VS_BTN + [F("SB")], T))
        self.assertIsNone(pf_norake.spot_key("BTN", VS_BTN + [R("SB"), F("BB")], T))
        H = pf.Table(pf.SEATS[2])
        self.assertEqual(pf_norake.spot_key("SB", [{"pos": "SB", "action": "call"}, R("BB")], H), "2|SB|L,BB")
        self.assertIn("6|SB|BTN", __import__("gto.pf_norake_data", fromlist=["DATA"]).DATA)

    def test_no_rake_defends_wider_and_rake_keeps_the_chart(self):
        adv = Advisor()
        base = {"hero": "SB", "board": "", "preflop": VS_BTN, "study": True}
        none = shares(adv.advise(dict(base)))
        raked = shares(adv.advise(dict(base, rake=0.05, rake_cap=4)))
        chart = pf.strategy_for("SB", VS_BTN, pf.Table(pf.SEATS[6]))
        self.assertAlmostEqual(raked["call"], float(np.mean(chart["call"])), delta=0.01)
        self.assertGreater(none["call"] + none["raise"], raked["call"] + raked["raise"] + 0.04)
        self.assertTrue(any("no rake" in n for n in adv.advise(dict(base))["notes"]))

    def test_unraked_charts_and_opens_barely_move(self):
        adv = Advisor()
        bb = shares(adv.advise({"hero": "BB", "board": "", "preflop": VS_BTN + [F("SB")], "study": True}))
        chart = pf.strategy_for("BB", VS_BTN + [F("SB")], pf.Table(pf.SEATS[6]))
        self.assertAlmostEqual(bb["call"], float(np.mean(chart["call"])), delta=0.01)
        btn = shares(adv.advise({"hero": "BTN", "board": "", "preflop": [F("UTG"), F("HJ"), F("CO")], "study": True}))
        self.assertAlmostEqual(btn["raise"], float(np.mean(pf.RFI["BTN"])), delta=0.02)

    def test_ranges_entering_the_flop_stay_the_charts(self):
        line = VS_BTN + [{"pos": "SB", "action": "call"}, F("BB")]
        T = pf.Table(pf.SEATS[6])
        r = preflop_ranges(line, T, 100.0)["SB"]
        chart = pf.strategy_for("SB", VS_BTN, T)["call"]
        self.assertAlmostEqual(float(r.sum()), float(chart.sum()), delta=1.0)
