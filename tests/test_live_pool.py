"""The typical live player ("opponents": "live", gto/exploit.py LIVE) beside the typical online one ("pool")."""
import unittest

import numpy as np

from gto import exploit as ex
from gto.advisor import Advisor, _pf_key, preflop_ranges

N = 1326


def share(strat, key, reach=None):
    w = np.ones(N) if reach is None else reach
    return float((w * strat[key]).sum() / w.sum())


class TestProfile(unittest.TestCase):
    def test_default_is_online_and_resets(self):
        self.assertEqual(ex.profile(), "online")
        t = ex.set_profile("live")
        self.assertEqual(ex.profile(), "live")
        self.assertEqual(ex._params("BLUFF_KEEP")[0], ex.LIVE["BLUFF_KEEP"])
        ex.reset_profile(t)
        self.assertEqual(ex.profile(), "online")
        self.assertIs(ex._params("BLUFF_KEEP")[0], ex.BLUFF_KEEP)
        t = ex.set_profile("nonsense")   # anything else is the online pool
        self.assertEqual(ex.profile(), "online")
        ex.reset_profile(t)

    def test_preflop_cache_keys_by_pool(self):
        pos = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        a = _pf_key("r", "BTN", [], pos, 100.0)
        t = ex.set_profile("live")
        try:
            b = _pf_key("r", "BTN", [], pos, 100.0)
        finally:
            ex.reset_profile(t)
        self.assertNotEqual(a, b)


class TestLivePreflop(unittest.TestCase):
    """First in it limps much of what it plays and plays more; facing an open it 3bets less and calls more."""

    def chart(self, n):
        rng = np.random.default_rng(n)
        e = rng.random(N)
        order = np.argsort(-e)
        raise_ = np.zeros(N); raise_[order[:300]] = 1.0       # the strongest 300 combos open
        return {"fold": 1.0 - raise_, "raise": raise_}, e

    def both(self, fn):
        online = fn()
        t = ex.set_profile("live")
        try:
            live = fn()
        finally:
            ex.reset_profile(t)
        return online, live

    def test_first_in(self):
        strat, e = self.chart(1)
        on, lv = self.both(lambda: ex.pool_preflop(strat, 0, "CO", np.ones(N), e, first_in=True))
        vpip = lambda s: share(s, "raise") + share(s, "call")
        self.assertGreater(vpip(lv), vpip(on))
        self.assertLess(share(lv, "raise"), share(on, "raise"))
        self.assertGreater(share(lv, "call"), 2 * share(on, "call"))
        # the limps are the chart's weakest opens: the strongest hand still raises
        top = int(np.argmax(e))
        self.assertAlmostEqual(lv["raise"][top], 1.0)
        for s in (on, lv):
            np.testing.assert_allclose(s["fold"] + s["call"] + s["raise"], 1.0, atol=1e-9)

    def test_facing_an_open(self):
        rng = np.random.default_rng(2)
        e = rng.random(N)
        order = np.argsort(-e)
        r = np.zeros(N); r[order[:80]] = 1.0
        c = np.zeros(N); c[order[80:200]] = 1.0
        strat = {"fold": 1.0 - r - c, "call": c, "raise": r}
        on, lv = self.both(lambda: ex.pool_preflop(strat, 1, "BTN", np.ones(N), e))
        self.assertLess(share(lv, "raise"), share(strat, "raise"))
        self.assertLess(share(lv, "raise"), share(on, "raise"))
        self.assertGreater(share(lv, "call"), share(on, "call"))
        # the 3bets it keeps are its strongest
        self.assertAlmostEqual(lv["raise"][int(np.argmax(e))], 1.0)

    def test_facing_a_3bet(self):
        rng = np.random.default_rng(3)
        e = rng.random(N)
        reach = (e > 0.6).astype(float)
        strat = {"fold": np.full(N, 0.6), "call": np.full(N, 0.3), "raise": np.full(N, 0.1)}
        on, lv = self.both(lambda: ex.pool_preflop(strat, 2, "CO", reach, e, opener=True))
        self.assertLess(share(lv, "fold", reach), share(on, "fold", reach))

    def test_ranges_differ(self):
        acts = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "call"}]
        pos = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        on, lv = self.both(lambda: preflop_ranges(acts, pos, 100.0, pool_except="BTN")["HJ"].sum())
        self.assertNotAlmostEqual(on, lv, places=0)   # its limps: a wider, stronger band live


class TestLiveAdvise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = Advisor(time_budget=2.0)

    def test_answers_and_resets(self):
        st = {"players": 6, "hero": "BTN", "hole": "AhJd", "board": "", "stack": 100,
              "preflop": [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "fold"}]}
        for o in ("pool", "live"):
            r = self.adv.advise(dict(st, opponents=o))
            self.assertIn(r["recommended"], ("raise", "call", "fold"))
            self.assertEqual(ex.profile(), "online")

    def test_river_bluffcatch(self):
        # BTN opens, BB calls; checked to the river, BB bets pot: live, it bluffs so little that a bluff catcher
        # gives up more often than against the online player
        R = lambda p, a, x=None: {"pos": p, "action": a, **({"to": x} if a == "raise" else {"amount": x} if x else {})}
        st = {"players": 6, "hero": "BTN", "hole": "Ks9s", "board": "Kd7c2h4s3d", "stack": 100,
              "preflop": [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5),
                          R("SB", "fold"), R("BB", "call")],
              "flop": [R("BB", "check"), R("BTN", "check")], "turn": [R("BB", "check"), R("BTN", "check")],
              "river": [R("BB", "bet", 5.5)]}
        fold = lambda r: sum(a["freq"] for a in r["actions"] if a["action"] == "fold")
        on, lv = fold(self.adv.advise(dict(st, opponents="pool"))), fold(self.adv.advise(dict(st, opponents="live")))
        self.assertGreaterEqual(lv, on)


if __name__ == "__main__":
    unittest.main()
