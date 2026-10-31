"""Heads-up answers mark the EVs of actions hero's range almost never takes as rough (advisor.RARE_BRANCH, "rough_ev"):
the opponent's answer to such a branch is barely trained, so the page shows them with a "≈" (docs/cfr_research.md)."""
import gc
import json
import os
import unittest
from unittest import mock

from gto import advisor as A

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


class RoughEv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gto.solver import Config
        cls.adv = A.Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        with open(os.path.join(EX, "flop.json")) as f:
            cls.state = json.load(f)
        cls.res = cls.adv.advise(dict(cls.state))
        with mock.patch.object(A, "RARE_BRANCH", 0.1):   # (the same cached solve, every action under a tenth of the range)
            cls.tenth = cls.adv.advise(dict(cls.state))

    @classmethod
    def tearDownClass(cls):
        del cls.adv
        gc.collect()

    def test_flag_where_the_range_rarely_takes_the_action(self):
        for res, bar in ((self.res, A.RARE_BRANCH), (self.tenth, 0.1)):
            self.assertNotIn("error", res)
            rf = res["range_action_freq"]   # (the equilibrium's mix here: no exploit settings)
            for a in res["actions"]:
                self.assertIn(a.get("rough_ev"), (None, True))   # (the key only where it holds)
                if abs(rf[a["action"]] - bar) > 0.001:
                    self.assertEqual("rough_ev" in a, rf[a["action"]] < bar, (a, rf))
        self.assertTrue(any(a.get("rough_ev") for a in self.tenth["actions"]))
        self.assertFalse(all(a.get("rough_ev") for a in self.tenth["actions"]))

    def test_nothing_else_changes(self):
        strip = lambda r: [{k: v for k, v in a.items() if k != "rough_ev"} for a in r["actions"]]
        self.assertEqual(strip(self.res), strip(self.tenth))
        self.assertEqual(self.res["recommended"], self.tenth["recommended"])
        self.assertEqual(self.res["solver"]["iterations"], self.tenth["solver"]["iterations"])


if __name__ == "__main__":
    unittest.main()
