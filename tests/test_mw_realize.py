"""A call re-drawn with more players in than the chart's: what it realizes moves with the price, a small pair's more
(advisor.MW_LEVEL..., _mw_realize), held to the flop solves in docs/mw_preflop_reference.json."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

from gto import advisor as A
from gto import preflop as pf
from gto.pf_classes import CLASSES, to_classes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import bench_mw_preflop as bench  # noqa: E402

idx = {c: i for i, c in enumerate(CLASSES)}
COMBO = lambda h: next(i for i, c in enumerate(A.ALL_COMBOS) if A.hand_class(c) == h)


def neutral():
    """The chart's level everywhere."""
    return mock.patch.multiple(A, MW_LEVEL=1.0, MW_SLOPE=0.0, MW_SMALL_PAIR=0.0)


def advice(spot):
    A._PF_CACHE.clear()
    try:
        return bench.advice(spot)
    finally:
        A._PF_CACHE.clear()


class TestRealize(unittest.TestCase):
    def test_a_dearer_price_realizes_more(self):
        self.assertLess(A._mw_realize(0.15, 100.0)[COMBO("KQo")], 1.0)
        self.assertGreater(A._mw_realize(0.27, 100.0)[COMBO("KQo")], 1.0)

    def test_a_small_pair_more_than_a_bigger_one(self):
        m = A._mw_realize(0.2, 100.0)
        self.assertAlmostEqual(m[COMBO("66")] - m[COMBO("77")], A.MW_SMALL_PAIR)
        self.assertEqual(m[COMBO("77")], m[COMBO("A4s")])

    def test_short_stacks_stay_at_the_chart_level(self):
        np.testing.assert_array_equal(A._mw_realize(0.27, A.MW_DEEP[0]), np.ones(len(A.ALL_COMBOS)))
        self.assertEqual(A._mw_realize(0.27, A.MW_DEEP[1])[0], A._mw_realize(0.27, 100.0)[0])

    def test_heads_up_doesnt_change(self):
        T = pf.SEATS[6]
        for hero, line in (("BB", "BTN raise 2.5"), ("BTN", "HJ raise 2.5"), ("BB", "SB raise 3"), ("CO", "CO raise 2.5, BTN raise 7.5")):
            pre = bench.full_line(T, hero, line)[0]
            A._PF_CACHE.clear()
            s = A.preflop_strategy(hero, pre, T, 100)
            with neutral():
                A._PF_CACHE.clear()
                b = A.preflop_strategy(hero, pre, T, 100)
            A._PF_CACHE.clear()
            for k in b:
                np.testing.assert_array_equal(s[k], b[k], line)

    def test_the_three_way_library_keeps_its_ranges(self):
        for spot in ("cc_co_btn",):
            c, r = advice(spot)
            with neutral():
                c0, r0 = advice(spot)
            np.testing.assert_array_equal(c, c0)
            np.testing.assert_array_equal(r, r0)

    def test_the_published_three_way_spot(self):
        """Poker Academy NL200, BTN open + SB call: the big blind continues 28.8%; it was 24.1% at the chart's level."""
        c, r = advice("cc_btn_sb")
        n = to_classes(np.ones(len(A.ALL_COMBOS)))
        self.assertAlmostEqual(float(((c + r) * n).sum() / n.sum()), 0.288, delta=0.04)

    def test_small_pairs_call_the_hand_that_found_it(self):
        c, _ = advice("bg7")
        with neutral():
            c0, _ = advice("bg7")
        self.assertGreater(c[idx["33"]], c0[idx["33"]] + 0.5)
        self.assertEqual(c[idx["A4s"]], 0.0)   # EV of a call -0.5bb, -0.2bb with the river's correction


class TestReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(bench.DATA) as fh:
            cls.data = json.load(fh)

    def test_the_data_is_whole(self):
        for key, r in self.data["spots"].items():
            for f in ("ev", "se", "river", "combos"):
                self.assertEqual(len(r[f]), len(CLASSES), key)
            self.assertGreaterEqual(min(b["flops"] for b in r["branches"]), 5, key)

    def test_the_advice_gives_up_less_than_before(self):
        now = before = 0.0
        fit = [k for k in self.data["spots"] if not k.endswith(("_wide", "_now")) and k not in ("ip_btn", "sb_cc")]
        for key in fit:
            r = self.data["spots"][key]
            ev, n = np.array(r["ev"]), np.array(r["combos"])
            a = bench.score(ev, *advice(key), n)
            b = bench.score(ev, np.array(r["advisor_before"]["call"]), np.array(r["advisor_before"]["raise"]), n)
            now += a[1] + a[2]
            before += b[1] + b[2]
        self.assertLess(now, before - 0.05)


if __name__ == "__main__":
    unittest.main()
