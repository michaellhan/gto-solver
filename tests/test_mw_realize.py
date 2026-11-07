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
    return bench.chart_level()


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

    def test_each_seat_before_the_big_blind_takes_some_off_up_to_those_measured(self):
        m = lambda seats: A._mw_realize(0.28, 100.0, seats)[COMBO("KQo")]
        self.assertAlmostEqual(m(0) - m(1), A.MW_SEAT)
        self.assertAlmostEqual(m(0) - m(A.MW_SEATS), A.MW_SEAT * A.MW_SEATS)
        self.assertEqual(m(A.MW_SEATS + 3), m(A.MW_SEATS))

    def test_the_seats_counted(self):
        """The button one, the cutoff two, the hijack at seven handed three; the blinds none: the big blind doesn't count."""
        want = dict(ip_btn=1, ip_btn_limp=1, ip_co=2, ip_co_limp=2, ip_hj=3, ip_hj_limp=3, sb_cc=0, sb_iso=0, bg7=0)
        real = A._mw_realize
        for spot, seats in want.items():
            got = []
            with mock.patch.object(A, "_mw_realize", lambda p, b, s=0: (got.append(s), real(p, b, s))[1]):
                advice(spot)
            self.assertEqual(got[-1], seats, spot)

    def test_the_blinds_dont_change(self):
        for spot in ("sb_cc", "cc_btn_sb", "iso1"):
            c, r = advice(spot)
            with mock.patch.object(A, "MW_SEAT", 0.0):
                c0, r0 = advice(spot)
            np.testing.assert_array_equal(c, c0)
            np.testing.assert_array_equal(r, r0)

    def test_the_hand_that_widened_the_hijack(self):
        """UTG limps, LJ raises to 4: HJ cold calls 29 combos at the chart's level, 95 without MW_SEAT."""
        T = pf.SEATS[7]
        pre = bench.full_line(T, "BB", "UTG call, LJ raise 4, HJ call")[0]

        def combos():
            A._PF_CACHE.clear()
            try:
                return float(A.preflop_ranges(pre, T, 100)["HJ"].sum())
            finally:
                A._PF_CACHE.clear()
        self.assertLess(combos(), 60)
        with mock.patch.object(A, "MW_SEAT", 0.0):
            self.assertGreater(combos(), 90)

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
        self.assertGreater(c[idx["33"]], c0[idx["33"]] + 0.2)   # (a 38% call of +0.6bb: at the bar since HJ's range narrowed)
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
        fit = [k for k in self.data["spots"] if not k.endswith(("_wide", "_now")) and k != "sb_cc" and k not in bench.IN_POSITION]
        for key in fit:
            r = self.data["spots"][key]
            ev, n = np.array(r["ev"]), np.array(r["combos"])
            a = bench.score(ev, *advice(key), n)
            b = bench.score(ev, np.array(r["advisor_before"]["call"]), np.array(r["advisor_before"]["raise"]), n)
            now += a[1] + a[2]
            before += b[1] + b[2]
        self.assertLess(now, before - 0.05)

    def test_no_branch_over_four_players(self):
        for key, r in self.data["spots"].items():
            self.assertLessEqual(max(len(b["players"]) for b in r["branches"]), bench.MAX_PLAYERS, key)

    def test_the_cap_on_players_is_on_the_branches(self):
        meta = dict(live=["CO", "HJ"], pend=[dict(pos=p, call=0.45, raise_=0.0) for p in ("SB", "BB")])
        got, _ = bench.branches(meta)
        self.assertIn(("SB",), [k for k, _ in got])
        self.assertNotIn(("SB", "BB"), [k for k, _ in got])       # five players
        self.assertAlmostEqual(sum(w for _, w in got), 1.0)

    def lost(self, spots, patch):
        with mock.patch.multiple(A, **patch):
            out = {}
            for s in spots:
                r = self.data["spots"][s]
                got = bench.score(np.array(r["ev"]), *advice(s), np.array(r["combos"]))
                out[s] = got[1] + got[2]
        return out

    def test_in_position_the_seat_term_gives_up_less(self):
        now = self.lost(bench.IN_POSITION, dict(MW_SEAT=A.MW_SEAT))
        without = self.lost(bench.IN_POSITION, dict(MW_SEAT=0.0))
        self.assertLess(sum(now.values()), sum(without.values()) - 0.04)
        self.assertLess(now[bench.HELD_OUT], without[bench.HELD_OUT] - 0.01)

    def test_a_comparison_is_paired_on_the_flops(self):
        """Both options read on the same resample: what moves both alike is no noise on their difference."""
        spot = "ip_btn"
        r = self.data["spots"][spot]
        ev = np.array(r["ev"])
        boots = np.array([ev + x for x in (-0.2, 0.0, 0.2)])
        with mock.patch.object(bench, "boot_evs", lambda s: boots):
            got = bench.compare([spot], {"same": dict(MW_SEAT=A.MW_SEAT), "again": dict(MW_SEAT=A.MW_SEAT), "off": dict(MW_SEAT=0.0)})
        self.assertEqual(got["same"][0], got["again"][0])
        self.assertEqual(got["again"][1], 0.0)
        self.assertGreater(got["off"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
