"""The solves an Advisor keeps for reuse stay under its memory budget (Advisor.cache_mb, advisor.CACHE_MB): kept without
bound, the browser's worker passed its 2 GB heap after a few hands and every later solve failed."""
import unittest

import numpy as np

from gto import advisor as A
from gto.advisor import Advisor, _held_bytes
from gto.native import advisors

MB = 2 ** 20


class _Slots:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class _Solve:
    def __init__(self, mb):
        self.tables = {"m": np.zeros(int(mb * MB) // 8)}


class TestHeldBytes(unittest.TestCase):
    def test_counts_each_array_once(self):
        x = np.zeros(1000)
        o = _Solve(1)
        o.view = o.tables["m"][:10]                       # a view: its base, already counted
        o.again = [x, x, (x,)]
        o.slots = _Slots(np.zeros(500), x)
        o.method = Advisor().advise                      # methods (and the advisor behind them) aren't followed
        self.assertEqual(_held_bytes(o), MB + 8000 + 4000)


class TestEviction(unittest.TestCase):
    def test_least_recently_used_go_first(self):
        adv = Advisor()
        adv.cache_mb = 2.5
        for k in "abc":
            adv._cache[k] = adv._by_spot["spot " + k] = _Solve(1)
            adv._keep("_cache", k, adv._cache[k])
        self.assertEqual(sorted(adv._cache), ["b", "c"])          # 3 MB was over: the oldest went
        adv._keep("_cache", "b", None)                             # b used again: c is now the oldest
        adv._cache["d"] = _Solve(1)
        adv._keep("_cache", "d", adv._cache["d"])
        self.assertEqual(sorted(adv._cache), ["b", "d"])
        self.assertEqual(sorted(adv._by_spot), ["spot b"])         # the spot index lets go of a and c too
        self.assertLessEqual(sum(adv._held.values()), 2.5 * MB)

    def test_the_latest_stay_whatever_they_hold(self):
        adv = Advisor()
        adv.cache_mb = 1
        for k in "ab":
            adv._cache[k] = _Solve(3)
            adv._keep("_cache", k, adv._cache[k])
        self.assertEqual(len(adv._cache), A.CACHE_KEEP)

    def test_multiway_lines_let_go(self):
        adv = Advisor()
        adv.cache_mb = 1.5
        for k in "ab":
            adv._mw_cache[k] = _Solve(1)
            adv._mw_lines["line " + k] = (adv._mw_cache[k], ["BB", "CO", "BTN"], 0.1)
            adv._keep("_mw_cache", k, adv._mw_cache[k])
        adv._cache["c"] = _Solve(0.2)
        adv._keep("_cache", "c", adv._cache["c"])
        self.assertEqual(list(adv._mw_cache), ["b"])
        self.assertEqual(list(adv._mw_lines), ["line b"])

    def test_speeds_share_one_budget(self):
        advs = advisors()
        fast, acc = advs["fast"], advs["accurate"]
        self.assertIs(fast._held, acc._held)
        fast.cache_mb = acc.cache_mb = 2.5
        fast._cache["old"] = _Solve(1)
        fast._keep("_cache", "old", fast._cache["old"])
        for k in "ab":
            acc._cache[k] = _Solve(1)
            acc._keep("_cache", k, acc._cache[k])
        self.assertEqual(fast._cache, {})                          # Fast's old hand goes first
        self.assertEqual(sorted(acc._cache), ["a", "b"])


class TestRealSolves(unittest.TestCase):
    def test_rivers_stay_under_the_budget_and_the_latest_is_reused(self):
        adv = Advisor(time_budget=0.3, prior_budget=0.3)
        adv.cache_mb = 0.0                                         # keep only the latest CACHE_KEEP
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        X = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "check"}]
        st = {"players": 6, "hero": "BB", "hole": "QhJh", "stack": 100, "preflop": pre, "flop": X, "turn": X, "river": []}
        for board in ("Ks7d2cTh5s", "Ad9c4s3h2d", "Jc8d3hQs6c", "Qd8s4c4h9s"):
            r = adv.advise(dict(st, board=board))
            self.assertNotIn("error", r)
            self.assertLessEqual(len(adv._cache), max(A.CACHE_KEEP, 1) + 2)    # (this river's and its narrowing streets')
            self.assertTrue(all(any(s is v for v in adv._cache.values()) for s in adv._by_spot.values()))
        n = len(adv._cache)
        again = adv.advise(dict(st, board="Qd8s4c4h9s"))
        self.assertEqual(len(adv._cache), n)                       # the latest spot is answered from its solve
        self.assertEqual([a["action"] for a in again["actions"]], [a["action"] for a in r["actions"]])


if __name__ == "__main__":
    unittest.main()
