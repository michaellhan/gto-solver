"""Preflop locks: your read of an opponent (exploit.lock_preflop) reshapes their preflop strategy, the ranges replayed
through it, and hero's line re-drawn against them (Advisor._locked, _vs_jam)."""
import gc
import unittest
import numpy as np
from gto import preflop as pf
from gto.cards import ALL_COMBOS, hand_class
from gto.advisor import Advisor, preflop_ranges, _lock_spec
from gto.exploit import lock_preflop, LOCK_PRESETS, LOCK_FOLD_HALF

T = pf.Table(pf.SEATS[6])
NIT, STATION, MANIAC, OVERFOLDER = (LOCK_PRESETS[k] for k in ("nit", "station", "maniac", "overfolder"))
CLASS_N = {}
for _c in ALL_COMBOS:
    CLASS_N[hand_class(_c)] = CLASS_N.get(hand_class(_c), 0) + 1


def R(pos, action, to=None):
    return {"pos": pos, "action": action, **({"to": to} if to is not None else {})}


def cls(v, h):
    """Mean weight of hand class h in a 1326 vector."""
    return float(np.mean([v[i] for i, c in enumerate(ALL_COMBOS) if hand_class(c) == h]))


def share(res, act):
    """The range-wide frequency of an action in an answer's grid."""
    g = res["grid"]
    i = g["actions"].index(act)
    w = sum(c["w"] * CLASS_N[k] for k, c in g["cells"].items())
    return sum(c["w"] * CLASS_N[k] * c["f"][i] for k, c in g["cells"].items()) / w


def freq(res, h, act):
    g = res["grid"]
    return g["cells"][h]["f"][g["actions"].index(act)]


def acts(res):
    return {a["action"]: a["freq"] for a in res["actions"]}


UTG_VS_SB = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "fold"), R("BTN", "fold")]
BTN_FIRST = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold")]
BB_VS_BTN = BTN_FIRST + [R("BTN", "raise", 2.5), R("SB", "fold")]


class TestLockPreflop(unittest.TestCase):
    """exploit.lock_preflop at one decision: what each key moves, and in which order."""

    def setUp(self):
        n = 10
        self.reach = np.ones(n)
        self.eq = np.linspace(0.9, 0.3, n)        # combo 0 strongest
        z = np.zeros(n)
        # facing a raise: the top two raise, the next two mix raise and call, two call, the rest fold
        self.strat = {"raise": z + [1, 1, .5, .5, 0, 0, 0, 0, .3, 0], "call": z + [0, 0, .5, .5, 1, 1, 0, 0, 0, 0]}
        self.strat["fold"] = 1.0 - self.strat["raise"] - self.strat["call"]

    def lock(self, adj, n_raises=1, **kw):
        return lock_preflop(self.strat, adj, n_raises, self.reach, self.eq, **kw)

    def test_each_key(self):
        s0 = self.strat
        tot = lambda s, k: float(s[k].sum()) / len(self.reach)
        # raise -50: half the raises call, the mixed ones (and the bluff mixing with folds) first, weakest first
        s = self.lock({"raise": -50})
        self.assertAlmostEqual(tot(s, "raise"), tot(s0, "raise") / 2, places=6)
        self.assertAlmostEqual(tot(s, "fold"), tot(s0, "fold"), places=6)
        self.assertEqual(s["raise"][0], 1.0)                                        # a pure value raise stays
        self.assertLess(s["raise"][8], s0["raise"][8])                              # the weakest mixed raise goes first
        # raise +100: twice the raises, from the hands already mixing first, then the strongest calls
        s = self.lock({"raise": 100})
        self.assertAlmostEqual(tot(s, "raise"), 2 * tot(s0, "raise"), places=6)
        self.assertEqual(s["raise"][2], 1.0)
        # bluff -100: raises behind what continues (under 45-50% equity when called) fold, value stays
        s = self.lock({"bluff": -100})
        self.assertEqual(s["raise"][8], 0.0)
        self.assertAlmostEqual(s["fold"][8], 1.0)
        np.testing.assert_allclose(s["raise"][:4], s0["raise"][:4])
        s = self.lock({"bluff": 100})
        self.assertGreater(tot(s, "raise"), tot(s0, "raise"))
        self.assertTrue(np.all(s["raise"][:4] == s0["raise"][:4]))                   # value isn't a bluff
        # fold +20 (half or more of the range continues: points of the range), weakest calls first, then raises
        s = self.lock({"fold": 20})
        self.assertAlmostEqual(tot(s, "fold"), tot(s0, "fold") + 0.2, places=6)
        self.assertEqual(s["call"][2], s0["call"][2])                                # the strongest call stays
        # ...and where less continues, the shift shrinks with it: a cold seat's 7% doesn't all fold at +15
        cold = {"raise": np.r_[1.0, np.zeros(9)], "call": np.zeros(10)}
        cold["fold"] = 1.0 - cold["raise"]
        s = lock_preflop(cold, {"fold": 15}, 1, self.reach, self.eq)
        self.assertAlmostEqual(float(s["raise"][0]), 1.0 - 0.15 * (0.1 / LOCK_FOLD_HALF) / 0.1, places=6)
        s = self.lock({"fold": -30})
        self.assertAlmostEqual(tot(s, "fold"), tot(s0, "fold") - 0.3, places=6)
        # every hand's strategy still sums to one
        for adj in (NIT, STATION, MANIAC, OVERFOLDER, {"fold": 100, "raise": -100, "bluff": -100}):
            s = self.lock(adj)
            np.testing.assert_allclose(sum(s.values()), 1.0, atol=1e-9)
            self.assertTrue(all((v >= -1e-12).all() for v in s.values()), adj)

    def test_first_in(self):
        # bet scales opens (the weakest fold, the strongest folds open); fold, bluff and raise don't act first in
        z = np.zeros(10)
        rfi = {"raise": z + [1, 1, 1, 1, .5, 0, 0, 0, 0, 0]}
        rfi["fold"] = 1.0 - rfi["raise"]
        s = lock_preflop(rfi, {"bet": -25}, 0, self.reach, self.eq)
        self.assertAlmostEqual(float(s["raise"].sum()), 4.5 * 0.75, places=6)
        self.assertEqual(float(s["raise"][4]), 0.0)                                  # the mixed open goes first
        s = lock_preflop(rfi, {"bet": 50}, 0, self.reach, self.eq)
        self.assertAlmostEqual(float(s["raise"].sum()), 4.5 * 1.5, places=6)
        self.assertEqual(float(s["raise"][5]), 1.0)
        s = lock_preflop(rfi, {"fold": 50, "bluff": -100, "raise": -100}, 0, self.reach, self.eq)
        np.testing.assert_allclose(s["raise"], rfi["raise"])


class TestPreflopLocks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.adv = Advisor(time_budget=0.5, prior_budget=0.2)

    @classmethod
    def tearDownClass(cls):
        # (its cached solves: the suite runs in one process, where a class's advisor holds them to the end)
        del cls.adv
        gc.collect()

    def advise(self, hero, hole, preflop, locks=None, **kw):
        return self.adv.advise({"players": 6, "hero": hero, "hole": hole, "stack": 100, "preflop": preflop,
                                **({"locks": locks} if locks else {}), **kw})

    def test_nit_3bet_folds_tens(self):
        """UTG opens TT, a nit in the small blind 3bets: the chart calls, against the nit's QQ+/AK-ish range it folds."""
        pre = UTG_VS_SB + [R("SB", "raise", 10), R("BB", "fold")]
        chart, nit = preflop_ranges(pre, T, 100)["SB"], preflop_ranges(pre, T, 100, None, {"SB": NIT})["SB"]
        self.assertAlmostEqual(chart.sum(), 73.5, delta=1)
        self.assertLess(nit.sum(), 50)                       # 45 combos
        for h in ("AA", "KK", "QQ", "AKs", "AKo"):
            self.assertEqual(cls(nit, h), 1.0, h)            # its premiums all 3bet
        for h in ("99", "AQs", "A5s", "KQs"):
            self.assertLess(cls(nit, h), 0.05, h)            # its merged hands and bluffs don't
        self.assertLess(cls(nit, "TT"), 0.25)
        gto = self.advise("UTG", "TsTh", pre)
        self.assertEqual(acts(gto)["call"], 1.0)
        r = self.advise("UTG", "TsTh", pre, {"SB": NIT})
        self.assertEqual(r["locked"], {"SB": NIT})
        self.assertGreater(acts(r)["fold"], 0.25)            # 34% (the chart: never)
        self.assertEqual(r["actions"][1]["gto_freq"], 1.0)   # the chart's mix alongside
        self.assertTrue(any(n.startswith("locked: SB folds +15 pts to raises") and "45 combos (74 unlocked)" in n
                            for n in r["notes"]), r["notes"])
        self.assertTrue(any(n.startswith("calls re-drawn by equity against SB's locked range") for n in r["notes"]))
        # the whole continuing range shrinks: AQs, KQs and the middle pairs go, QQ+, AK and JJ stay
        self.assertGreater(share(r, "fold"), share(gto, "fold") + 0.15)
        for h in ("AQs", "KQs", "99", "88"):
            self.assertGreater(freq(r, h, "fold"), 0.6, h)
        for h in ("JJ", "QQ", "AKo"):
            self.assertLess(freq(r, h, "fold"), 0.05, h)
        # 3betting big (12bb, a worse price): the chart still calls, against the nit it folds
        big = UTG_VS_SB + [R("SB", "raise", 12), R("BB", "fold")]
        self.assertEqual(acts(self.advise("UTG", "TsTh", big))["call"], 1.0)
        self.assertGreater(acts(self.advise("UTG", "TsTh", big, {"SB": NIT}))["fold"], 0.8)
        # a maniac's 3bet is wide: TT 4bets for value; a station's is all value: TT folds more than the chart
        self.assertEqual(self.advise("UTG", "TsTh", pre, {"SB": MANIAC})["recommended"], "raise")
        self.assertGreater(acts(self.advise("UTG", "TsTh", pre, {"SB": STATION}))["fold"], 0.2)

    def test_station(self):
        """A station 3bets value only and calls wide: against its open hero 3bets thinner for value and bluffs less;
        facing its 3bet hero folds more."""
        st = preflop_ranges(UTG_VS_SB + [R("SB", "raise", 10)], T, 100, None, {"SB": STATION})["SB"]
        self.assertLess(st.sum(), 40)                        # QQ+, AK and a little AQo
        # facing hero's 3bet it calls far wider than the chart (the opener's range defending a 3bet)
        pre = BTN_FIRST[:2] + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold"), R("CO", "call")]
        self.assertGreater(preflop_ranges(pre, T, 100, None, {"CO": STATION})["CO"].sum(),
                           2 * preflop_ranges(pre, T, 100)["CO"].sum())
        # the big blind against a station's button open: thin value 3bets (A8o called before), fewer bluffs; the small
        # pairs keep flatting (they play for implied odds, which a station pays more: never new value raises)
        gto = self.advise("BB", "AhKh", BB_VS_BTN)
        r = self.advise("BB", "AhKh", BB_VS_BTN, {"BTN": STATION})
        self.assertEqual(freq(gto, "A8o", "raise"), 0.0)
        self.assertGreater(freq(r, "A8o", "raise"), 0.5)
        for h in ("22", "44", "55"):
            self.assertLessEqual(freq(r, h, "raise"), freq(gto, h, "raise") + 1e-9, h)
            self.assertGreaterEqual(freq(r, h, "call"), freq(gto, h, "call") - 0.05, h)
        for h in ("T9s", "JTs"):
            self.assertLess(freq(r, h, "raise"), freq(gto, h, "raise") - 0.2, h)
        self.assertTrue(any(n.startswith("raises re-drawn against BTN as locked: everyone folds") and "bluffs x0." in n
                            for n in r["notes"]), r["notes"])
        # first in with a station in the big blind: the pure steals (suited connectors) open less
        gto = self.advise("BTN", "9s6s", BTN_FIRST)
        r = self.advise("BTN", "9s6s", BTN_FIRST, {"BB": STATION})
        for h in ("65s", "54s", "76s"):
            self.assertLess(freq(r, h, "raise"), freq(gto, h, "raise") - 0.3, h)

    def test_nit_and_overfolder_get_stolen_from(self):
        # first in with a nit (or an overfolder) in the big blind: more opens
        for lk in (NIT, OVERFOLDER):
            gto = self.advise("BTN", "9s6s", BTN_FIRST)
            r = self.advise("BTN", "9s6s", BTN_FIRST, {"BB": lk})
            self.assertGreater(share(r, "raise"), share(gto, "raise") + 0.02, lk)
            self.assertEqual(freq(r, "Q8o", "raise"), 1.0)   # (half-opened by the chart)
            self.assertEqual(r["locked"], {"BB": lk})
            self.assertTrue(any(n.startswith("locked: BB") and "to act behind you" in n for n in r["notes"]))
        gto = self.advise("UTG", "9s6s", [])
        self.assertGreater(share(self.advise("UTG", "9s6s", [], {"BB": NIT}), "raise"), share(gto, "raise") + 0.005)
        # the big blind against a nit's button open: defends tighter (the open is tighter), 3bet-bluffs more
        gto = self.advise("BB", "AhKh", BB_VS_BTN)
        r = self.advise("BB", "AhKh", BB_VS_BTN, {"BTN": NIT})
        self.assertGreater(share(r, "fold"), share(gto, "fold") + 0.05)
        self.assertLess(freq(r, "Q7o", "call"), 0.1)
        # an overfolder opening: hero 3bets more against it (it folds more to the 3bet), its value 3bets (KQs) included
        pre = BTN_FIRST[:2] + [R("CO", "raise", 2.5)]
        gto, r = self.advise("BTN", "AhKh", pre), self.advise("BTN", "AhKh", pre, {"CO": OVERFOLDER})
        self.assertGreater(share(r, "raise"), share(gto, "raise") + 0.005)
        self.assertEqual(freq(r, "KQs", "raise"), freq(gto, "KQs", "raise"))
        self.assertTrue(any("so bluffs x1.1" in n for n in r["notes"]), r["notes"])

    def test_jam_from_a_locked_player(self):
        # CO opens, hero 3bets, CO jams 100bb: a nit's jam (its 4bet range as locked) is too strong for JJ
        pre = BTN_FIRST[:2] + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold"), R("CO", "all-in")]
        gto = self.advise("BTN", "JsJh", pre)
        r = self.advise("BTN", "JsJh", pre, {"CO": NIT})
        self.assertEqual(acts(gto)["call"], 1.0)
        self.assertEqual(acts(r)["fold"], 1.0)
        self.assertLess(r["equity"], gto["equity"] - 0.05)
        self.assertIn("locked: CO's jam is their raising range as you locked it", r["notes"])
        self.assertEqual(r["locked"], {"CO": NIT})
        self.assertGreater(self.advise("BTN", "JsJh", pre, {"CO": MANIAC})["equity"], gto["equity"])

    def test_ranges_after_the_preflop(self):
        """Postflop advice starts from the locked player's preflop range."""
        pre = BB_VS_BTN + [R("BB", "call")]
        chart = preflop_ranges(pre, T, 100)["BTN"]
        nit = preflop_ranges(pre, T, 100, None, {"BTN": NIT})["BTN"]
        maniac = preflop_ranges(pre, T, 100, None, {"BTN": MANIAC})["BTN"]
        self.assertLess(nit.sum(), 0.85 * chart.sum())       # opens 25% less
        self.assertGreater(maniac.sum(), 1.2 * chart.sum())
        np.testing.assert_array_equal(preflop_ranges(pre, T, 100)["BB"], preflop_ranges(pre, T, 100, None, {"BTN": NIT})["BB"])
        state = {"players": 6, "hero": "BB", "hole": "AhQd", "board": "Kd7c2h", "stack": 100, "preflop": pre, "flop": []}
        vw = lambda r: sum(c["w"] * CLASS_N[k] for k, c in r["villain_grid"]["cells"].items())
        gto = self.adv.advise(state)
        r = self.adv.advise(dict(state, locks={"BTN": NIT}))
        self.assertLess(vw(r), 0.85 * vw(gto))
        self.assertEqual(r["locked"], {"BTN": NIT})

    def test_cache_isolation_and_no_lock(self):
        pre = UTG_VS_SB + [R("SB", "raise", 10), R("BB", "fold")]
        before = self.advise("UTG", "TsTh", pre)
        r_before = preflop_ranges(pre, T, 100)
        locked = self.advise("UTG", "TsTh", pre, {"SB": NIT})
        self.assertNotEqual(acts(locked), acts(before))
        after = self.advise("UTG", "TsTh", pre)
        strip = lambda r: {k: v for k, v in r.items() if k not in ("seconds", "sampled")}
        self.assertEqual(strip(after), strip(before))
        self.assertNotIn("locked", after)
        self.assertNotIn("gto_freq", after["actions"][0])
        for p, v in preflop_ranges(pre, T, 100).items():
            np.testing.assert_array_equal(v, r_before[p])
        # the frozen spec: the preflop keys only, in their order, zeros dropped; none is ()
        self.assertEqual(_lock_spec({"SB": {"next_fold": 50, "raise": -30, "fold": 15, "bluff": 0}}),
                         (("SB", (("fold", 15.0), ("raise", -30.0))),))
        self.assertEqual(_lock_spec({"SB": {"fold_flop": 10}}), ())
        self.assertEqual(_lock_spec(None), ())
        # a lock on a seat that folded, or one that only acts after the flop, leaves the advice as it was
        for lk in ({"HJ": NIT}, {"SB": {"next_fold": 80, "fold_river": 10}}):
            r = self.advise("UTG", "TsTh", pre, lk)
            self.assertEqual(acts(r), acts(before), lk)
            self.assertNotIn("locked", r)
        r = self.advise("UTG", "TsTh", pre, {"SB": {"next_fold": 80}})
        self.assertIn("locked: SB's lock (next_fold) acts after the flop only", r["notes"])
        # on top of the pool: the pool's wider 3bet range, cut down
        pool = self.advise("UTG", "TsTh", pre, opponents="pool")
        r = self.advise("UTG", "TsTh", pre, {"SB": NIT}, opponents="pool")
        self.assertTrue(any(n.startswith("locked: SB") and "(on the pool)" in n for n in r["notes"]), r["notes"])
        self.assertGreater(share(r, "fold"), share(pool, "fold") + 0.15)
        # your own seat can't be locked, preflop as after
        self.assertIn("your own seat", self.advise("UTG", "TsTh", pre, {"UTG": NIT})["error"])

    def test_practice_opponent_ignores_locks(self):
        # a practice opponent's own preflop move is GTO's, whoever the player practising locked
        st = {"players": 6, "hero": "UTG", "hole": "AhAd", "stack": 100, "preflop": UTG_VS_SB, "locks": {"SB": NIT}}
        m = self.adv.play(st, "SB", "TsTh", seed=1)
        self.assertNotIn("error", m)
        chart = {a["action"]: a["freq"] for a in self.adv.advise({**st, "hero": "SB", "hole": "TsTh", "locks": None})["actions"]}
        self.assertEqual({a["action"]: a["freq"] for a in m["actions"]}, chart)


if __name__ == "__main__":
    unittest.main()
