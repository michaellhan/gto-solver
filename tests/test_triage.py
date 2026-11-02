"""The session analyzer's Quick triage (Advisor.triage, "triage": true): each decision answered from what is in hand
first (the charts, the flop library, a read at once), else a Quick solve, with where the answer came from and how far
the analyzer's deeper pass (the same decision at Fast) can be expected to agree: "triage": {source, confidence,
label}."""
import gc
import time
import unittest
from unittest import mock

from gto import advisor as A
from gto.native import advisor
from gto.solver import Cancelled

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
LIMP = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
        {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "check"}]
BASE = {"players": 6, "stack": 100, "budget": 8, "prior_budget": 3, "speed": "fast"}
FLOP = dict(BASE, hero="BB", hole="Ts9s", board="Kd7c2h", preflop=PRE, flop=[])
TURN = dict(BASE, hero="SB", hole="8h7h", board="Kd7c2h4s", preflop=LIMP,
            flop=[{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}], turn=[])
ACTS = [{"action": "check", "freq": 0.7}, {"action": "bet 75%", "freq": 0.3}]


class TestTriage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    @classmethod
    def tearDownClass(cls):
        del cls.adv
        gc.collect()

    def triage(self, st):
        return self.adv.advise(dict(st, triage=True))

    def test_preflop_from_the_charts(self):
        r = self.triage(dict(BASE, hero="BB", hole="AhKh", board="", preflop=PRE[:5]))
        self.assertEqual(r["triage"], {"source": "chart", "confidence": "high", "label": "the preflop charts"})
        self.assertTrue(r["actions"])

    def test_a_library_flop(self):
        r = self.triage(FLOP)
        self.assertIn("precomputed", r)
        self.assertEqual((r["triage"]["source"], r["triage"]["confidence"]), ("library", "high"))
        # the same answer as without triage (the library's own spot: Fast reads it too)
        plain = self.adv.advise(dict(FLOP))
        self.assertEqual([(a["action"], a["freq"]) for a in r["actions"]], [(a["action"], a["freq"]) for a in plain["actions"]])

    def test_a_quick_solve(self):
        """A limped pot's turn (no library, nothing in hand): a Quick solve within TRIAGE_BUDGETS, medium when it's
        within TRIAGE_SETTLED of equilibrium, else low."""
        t0 = time.time()
        with mock.patch.object(A.Advisor, "instant", return_value=None):
            r = self.triage(TURN)
        took = time.time() - t0
        t, sv = r["triage"], r["solver"]
        self.assertEqual(t["source"], "quick")
        self.assertIn("Quick solve", t["label"])
        x = sv.get("exploitability_pct")
        settled = x is not None and x <= A.TRIAGE_SETTLED and not sv.get("approximate")
        self.assertEqual(t["confidence"], "medium" if settled else "low")
        self.assertLess(took, sum(A.TRIAGE_BUDGETS["turn"]) + 2)

    def test_routes_and_confidence(self):
        """The reads at once by kind (the stored turn medium; the nearest stored flops, other bet sizes and an earlier
        street's solve low), a rougher Quick solve, the betting rules, and cancels and errors as they are."""
        adv = A.Advisor.__new__(A.Advisor)
        seen = []

        def run(instant=None, advise=None, st=TURN):
            seen.clear()

            def fake(state):
                seen.append(state)
                return dict(advise) if isinstance(advise, dict) else advise(state)
            with mock.patch.object(A.Advisor, "instant", lambda self, s: None if instant is None else dict(instant)), \
                 mock.patch.object(A.Advisor, "advise", lambda self, s: fake(s)):
                return A.Advisor.triage(adv, dict(st, triage=True))

        for src, conf in (("store", "medium"), ("near", "low"), ("offtree", "low"), ("prev", "low"), ("solve", "low")):
            r = run({"actions": ACTS, "approximate": {"source": src, "label": f"a {src} read"}})
            self.assertEqual(r["triage"], {"source": "approx", "confidence": conf, "label": f"a {src} read"}, src)
            self.assertEqual(seen, [])   # (read at once: nothing solved)
        # a Quick solve: its budgets, no "triage" passed on
        r = run(advise={"actions": ACTS, "solver": {"exploitability_pct": 0.6, "iterations": 120}})
        self.assertEqual((r["triage"]["source"], r["triage"]["confidence"]), ("quick", "medium"))
        self.assertNotIn("triage", seen[0])
        self.assertEqual((seen[0]["quick"], seen[0]["budget"], seen[0]["prior_budget"]), (True,) + A.TRIAGE_BUDGETS["turn"])
        for sv in ({"exploitability_pct": 2.5, "iterations": 40}, {"exploitability_pct": 0.5, "approximate": True}, {}):
            self.assertEqual(run(advise={"actions": ACTS, "solver": sv})["triage"]["confidence"], "low", sv)
        # the betting rules: a multiway spot no solve answered, or the catch-all
        for res in ({"actions": ACTS, "multiway": True, "solved": False}, {"actions": ACTS, "fallback": "rules"}):
            self.assertEqual(run(advise=res)["triage"], {"source": "rules", "confidence": "low", "label": "the betting rules"})
        # a borrowed library spot (read as this pot) is medium
        r = run(advise={"actions": ACTS, "precomputed": {"label": "BTN vs BB", "borrowed": True}})
        self.assertEqual((r["triage"]["source"], r["triage"]["confidence"]), ("library", "medium"))
        # cancels, errors and skips as they are, unmarked
        for res in ({"cancelled": True}, {"error": "boom"}, {"skipped": "street"}):
            self.assertEqual(run(advise=res), res)
        # a flop: the library first ("library_only"), its cancel as it is; then the reads, then the Quick solve
        self.assertEqual(run(advise={"cancelled": True}, st=FLOP), {"cancelled": True})
        self.assertTrue(seen[0]["library_only"])
        r = run(advise=lambda s: {"skipped": "not stored"} if s.get("library_only") else {"actions": ACTS, "solver": {"exploitability_pct": 0.4}}, st=FLOP)
        self.assertEqual([bool(s.get("library_only")) for s in seen], [True, False])
        self.assertEqual(seen[1]["budget"], A.TRIAGE_BUDGETS["flop"][0])
        # a read that raises is solved instead; a cancel inside it is the request's
        with mock.patch.object(A.Advisor, "instant", side_effect=ValueError("x")), \
             mock.patch.object(A.Advisor, "advise", lambda self, s: {"actions": ACTS, "solver": {"exploitability_pct": 0.4}}):
            self.assertEqual(A.Advisor.triage(adv, dict(TURN, triage=True))["triage"]["source"], "quick")
        with mock.patch.object(A.Advisor, "instant", side_effect=Cancelled()):
            with self.assertRaises(Cancelled):
                A.Advisor.triage(adv, dict(TURN, triage=True))


if __name__ == "__main__":
    unittest.main()
